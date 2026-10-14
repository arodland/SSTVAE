// The live QRSSTVAE listener: sstvae/qrss/live.py's LiveListener and run()
// (the reference), on top of live.hpp's pieces. Feed it audio (`feed`), call
// `step(now)`; it keeps `state_dir` current: state.json (the tiles the
// window shows), tiles/<id>.png (their pictures) and listener.log.
//
//   live.py                       here
//   LiveListener.__init__         LiveListener(state_dir, cfg, store, passband, seams)
//   busy, close, log              busy, close, log
//   _load_previous                (the constructor)
//   tiles, write_state            tiles, write_state
//   feed, _note_silence,          feed (cheap: the stream front end, the ring
//   _sound_back, _heard           and the passband store's write)
//   backfill, recover             backfill, recover
//   _progress, active_slots       progress, active_slots
//   step, _interval               step, interval
//   _submit, _worker,             the two workers (below)
//   _finish_idle, idle            finish_idle, idle, wait_idle
//   _refresh, _finish,            the jobs; _hints is public as hints() (the
//   _finish_apply, _hints, ...    reference's tests call it)
//   _codec, _render               through a Renderer (qrss/codec/
//                                 listener_render.hpp has the codec's)
//   run                           run
//
// Decisions are the reference's, in its order: which slot is received when,
// which detection stands for a signal (by band centre), which tile a pass
// updates and whether it may, the ghost and headerless-SNR cuts, the hints
// and their USB dial shift, one end-of-slot receive per dial, the store's
// association and the provisional tiles a later header identifies, the
// picture's guess and accumulator. Log lines are the reference's texts;
// state.json is json.dumps(st, indent=1) of the reference's dict, its keys in
// its order (strict JSON: a value that is not finite is null, where the
// reference would write NaN).
//
// THREADS. One mutex guards all of the listener's state; every public
// method takes it itself, so any of them may be called from any thread
// (run() from one, the app's window reading tiles() from another, say).
// With cfg.background (qrss_listen's mode) two worker threads run the
// receives, at a lower priority than the caller's (nice +10 on Linux, the
// utility QoS class on macOS, below-normal on Windows): one the live
// receives, one the end-of-slot receives -- the reference's finish worker
// and its end-of-slot process in one, a C++ receive needing no process to
// run beside another. Without it, step() runs each receive before it
// returns (the reference's synchronous mode: what its tests drive). Either
// way a receive holds the lock only to copy what it needs (the slot's
// detections and candidates, the hints, the dial) and to apply what it
// found: the capture's read and preparation, the store's due passes (the
// hints), the receivers, the store's association, the codec's first load and
// every picture run without it, so feed() and step() never wait on a receive
// (nor on the store, whose own lock an association holds throughout). A
// picture is applied only if no newer update of its tile came in meanwhile
// (a per-tile generation), so a slow render never draws over a newer one.
//
// What blocks, and for how long:
//   feed          the lock, for the front end of one chunk, a ring write and
//                 the passband store's write (microseconds; a passband
//                 hour file's write when one fills)
//   step          the lock, for the bookkeeping (microseconds); without
//                 background the receives themselves (seconds to minutes)
//   write_state   the lock while the JSON is built; the file's write after
//   backfill      a passband store read (once, at startup)
//   recover       passband store reads (once, at startup), outside the lock
//   close         until the receives in flight return (the receivers take
//                 no cancel flag: a FULL end-of-slot receive is minutes)
//                 and the passband store's packer (at most CLOSE_PACKER_S)
// The clock seam is called with the lock held (it must not call back into
// the listener); the log sink never is (lines are handed to it in order
// once the lock is released, so a sink may read the listener).
//
// Departures from the reference (each one a fix, see port-plan 4.4):
//   * an exception in a live receive does not retire its slot: it is logged
//     once per slot and message and the slot is received again at its next
//     refresh (the reference set done, losing the slot's tiles for good on
//     one bad receive); an end-of-slot receive that throws still retires it;
//   * a receive's exception never leaves step(), background or not (the
//     reference's synchronous mode raised it to the caller);
//   * an unreadable state.json, or one whose structure is not the
//     reference's, starts the listener fresh (the reference raised from its
//     constructor);
//   * idle() is exact: a worker takes a job and says it is busy in one step
//     (the reference's worker took it from the queue, then set _busy, so
//     idle() could read true in between and a replay ran ahead of a receive).

#ifndef SSTVAE_QRSS_LISTENER_HPP
#define SSTVAE_QRSS_LISTENER_HPP

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "images/types.hpp"
#include "qrss/acquire.hpp"
#include "qrss/associate.hpp"
#include "qrss/frontend.hpp"
#include "qrss/header.hpp"
#include "qrss/live.hpp"
#include "qrss/passband.hpp"
#include "qrss/receiver.hpp"
#include "qrss/receiver_slot.hpp"
#include "qrss/rx_api.hpp"
#include "qrss/source.hpp"
#include "qrss/store.hpp"
#include "qrss/types.hpp"

namespace sstvae::qrss::live {

// How long close() waits for the passband store's packer to finish an hour.
inline constexpr double CLOSE_PACKER_S = 30.0;

// An exception a seam raised that is not a C++ one (the binding's Python
// callables): what() is its text (Python's str(e)), type_name() its type's
// name, which the listener's log lines show as the reference's do.
class SeamError : public std::runtime_error {
public:
    SeamError(std::string type_name, const std::string& text)
        : std::runtime_error(text), type_name_(std::move(type_name)) {}
    const std::string& type_name() const noexcept { return type_name_; }

private:
    std::string type_name_;
};

// The Python name of an exception: a SeamError's own; for the C++ ones the
// name of the Python exception the port raises in their place (ValueError
// for std::invalid_argument, KeyError for pyjson::KeyError, ...).
std::string error_name(const std::exception& e);
// The reference's f"{type(e).__name__}: {e}".
std::string error_text(const std::exception& e);

// What draws a tile's picture (live.py's _codec and render.load /
// check_codec_id / render, which the codec's implementation,
// codec::ListenerRenderer, does for real). Called from the workers, never
// with the listener's lock held, and from both at once: an implementation
// serialises itself.
class Renderer {
public:
    using Warn = std::function<void(const std::string&)>;
    virtual ~Renderer();
    // Ready the codec, loading it the first time (render.load(precision,
    // model, codec_id=codec_id, warn=warn) in the reference: codec_id is the
    // first picture's, and a mismatch then is a load error). nullopt when it
    // is loaded; else the error text the listener records as codec_error
    // (str(e), or the type's name when that is empty). The listener calls
    // it only while it has no codec_error, so it is never asked to try again.
    virtual std::optional<std::string> load(std::optional<std::int64_t> codec_id, const Warn& warn) = 0;
    // render.check_codec_id(codec, codec_id) once loaded: nullopt when a
    // picture of codec_id may be drawn (or there is nothing to check), else
    // the CodecMismatch's text.
    virtual std::optional<std::string> check(std::optional<std::int64_t> codec_id) = 0;
    // render.render(codec, S, W, mode): the picture of S and W (float64,
    // N_GROUPS x GROUP_LATENTS each); codec_id is the picture's (informative).
    // Throws std::exception whose what() the tile shows as "render failed: ...".
    virtual images::Picture render(std::span<const double> S, std::span<const double> W, std::int64_t mode,
                                   std::optional<std::int64_t> codec_id) = 0;
};

// The live receive: receiver.candidates(prep, sel, live_only=True,
// hints=hints), receiver.detect(prep, sel, **opt) and
// receiver.receive_slot(prep, sel, **opt), called exactly as _refresh calls
// them (opt: live_only, erase_s, cands, skip_hz; and estimator, erase_s,
// round_b=False, dets).
using CandidatesFn = std::function<std::vector<types::Detection>(const receiver::Prepared& prep, Selection sel,
                                                                 std::span<const acquire::Hint> hints)>;
using DetectFn = std::function<std::vector<types::Detection>(const receiver::Prepared& prep, Selection sel,
                                                             const receiver::DetectOptions& opt)>;
using ReceiveSlotFn = std::function<std::vector<types::PassResult>(const receiver::Prepared& prep, Selection sel,
                                                                   const receiver::SlotOptions& opt)>;
// The end-of-slot receive of one dial: finish_receive(x, ok, origin, q,
// config, dets, hints), config's frames the selection, its estimator the
// listener's and its rules the seams'.
using FinishFn = std::function<FinishResult(std::span<const cf> x, std::span<const std::uint8_t> ok,
                                            std::int64_t origin, std::int64_t q, const api::ReceiveConfig& config,
                                            std::span<const types::Detection> dets,
                                            std::span<const acquire::Hint> hints)>;
using AssociateFn = std::function<associate::Result(const types::PassResult& p, store::Store& store)>;

struct Seams {
    // time.time(): unix seconds, for every stamp the reference takes from the clock.
    std::function<double()> clock;
    // Where each log line goes ("<utc> <msg>"); empty: stderr, one line each.
    std::function<void(const std::string& line)> log;
    // The live receive; each empty one is the C++ receiver (receiver_slot.hpp
    // with api::slot_seams of the listener's ReceiveConfig).
    CandidatesFn candidates;
    DetectFn detect;
    ReceiveSlotFn receive_slot;
    // The end-of-slot receive; empty: live::finish_receive with its default
    // budgets and this noise.
    FinishFn finish;
    // The noise that fills what a capture did not hear (live.hpp's NoiseFn);
    // empty: default_noise.
    NoiseFn noise;
    // associate.associate(p, store); empty: associate::associate.
    AssociateFn associate;
    // The pictures; null: none (cfg.render is then ignored, the reference's
    // no-pictures listener).
    std::shared_ptr<Renderer> renderer;
    // The receiver's rules (rules.py's QRSS_OLD_EDGE_NOISE and
    // QRSS_OLD_GATING), for every receive.
    bool old_edge_noise = false;
    bool old_gating = false;
    // Lower the workers' priority (background mode).
    bool low_priority = true;
    // What the dial log takes off around a dial change (live.DIAL_GUARD_S,
    // which the reference's tests set at run time).
    DialGuard dial_guard = DIAL_GUARD_S;
};

class LiveListener {
public:
    // state_dir: where state.json, tiles/ and listener.log go (tiles/ is
    // created). store and passband null: none. The finished tiles of an
    // earlier run are read back from state.json. Throws when tiles/ cannot
    // be created (std::filesystem::filesystem_error).
    LiveListener(std::string state_dir, LiveConfig cfg = {}, std::shared_ptr<store::Store> store = nullptr,
                 std::shared_ptr<passband::PassbandStore> passband = nullptr, Seams seams = {});
    ~LiveListener();
    LiveListener(const LiveListener&) = delete;
    LiveListener& operator=(const LiveListener&) = delete;

    // --- what it is -------------------------------------------------------------------------
    const std::string& state_dir() const { return dir_; }
    const LiveConfig& config() const { return cfg_; }
    // cfg.specs(), cfg.selection(), receiver.is_joint(selection), and the
    // slot's length (frame_seconds of the specs: the longer frame).
    const std::vector<frame::FrameSpec>& specs() const { return specs_; }
    const std::vector<frame::FrameSpec>& selection() const { return sel_; }
    bool joint() const { return joint_; }
    double dur() const { return dur_; }
    // The ring (internally locked: a test may write into it directly).
    const FeRing& ring() const { return ring_; }
    FeRing& ring() { return ring_; }
    DialLog& dial() { return dial_; }
    // The clock seam's time.
    double clock() const;

    // --- state (copies, under the lock) -----------------------------------------------------
    // What the workers are doing, for state.json ("refresh <slot utc>",
    // "finish <slot utc>", both joined by "; "), nullopt when nothing.
    std::optional<std::string> busy() const;
    double now() const;
    std::optional<double> audio_end() const;
    // The stream front end's next audio index (fe.next_index): where the
    // next chunk follows on; nullopt before the first.
    std::optional<std::int64_t> next_index() const;
    std::string source() const;
    void set_source(std::string name);
    std::vector<std::string> lines() const;
    std::vector<std::int64_t> finished() const;   // sorted
    std::vector<SlotState> slots() const;         // by q
    std::vector<Tile> done_tiles() const;
    std::optional<std::string> codec_error() const;
    // Slots queued for recovery (recover), oldest queued first.
    std::vector<std::int64_t> recovering() const;
    // Every tile, live and done, newest slot first, then by frequency.
    std::vector<Tile> tiles() const;

    // --- logging and state ------------------------------------------------------------------
    // "<utc of now, else of the clock> <msg>": kept (the last 50), handed to
    // the log sink, and appended to listener.log (rotated to listener.log.1
    // past LOG_FILE_BYTES; a file that cannot be written is ignored).
    void log(const std::string& msg);
    // state.json, atomically (written as state.json.<pid>.tmp, then
    // replaced). Throws std::runtime_error when it cannot be written.
    void write_state(bool listening = true);

    // --- audio ------------------------------------------------------------------------------
    // Audio chunk x (8 kHz) whose last sample is at unix time t_end.
    void feed(std::span<const double> x, double t_end);
    // Refill the ring from the passband store; seconds read back.
    double backfill(double now);
    // Queue the end-of-slot receive (from the passband store) of every slot
    // that ended in the last `hours` while no listener finished it; the
    // slots queued.
    std::vector<std::int64_t> recover(double now, double hours = RECOVER_HOURS);

    // --- slots ------------------------------------------------------------------------------
    double progress(std::int64_t q) const;
    std::vector<std::int64_t> active_slots(double now) const;
    // Do whatever is due at `now` (unix seconds); true if anything changed.
    bool step(double now);
    // _hints(q, dial): where a later pass of a picture partly received is
    // due in slot q (on `dial`).
    std::vector<acquire::Hint> hints(std::int64_t q, std::optional<std::int64_t> dial);
    // When step() would next refresh a slot with no detections past
    // detect_until_s: _interval.
    double interval(std::int64_t q, double now) const;
    // _match: the tile of slot q (of either waveform) at the band centre of
    // a signal at f_hz in `waveform`, on the same dial (a copy), nullopt
    // when none.
    std::optional<Tile> match(std::int64_t q, double f_hz, std::optional<std::int64_t> dial = std::nullopt,
                              const std::string& waveform = "CE") const;

    // --- workers ----------------------------------------------------------------------------
    // No end-of-slot receive queued or running.
    bool finish_idle() const;
    // No receive queued or running.
    bool idle() const;
    // Block until idle() (or close()); false when timeout_s passed first.
    bool wait_idle(std::optional<double> timeout_s = std::nullopt) const;
    // run()'s end of a replay: max(every slot's end + 1 s, audio_end), or
    // nullopt with no slot.
    std::optional<double> replay_end() const;
    // Stop the workers: the receives in flight finish (and are applied),
    // nothing queued is run; then the passband store's packer gets
    // CLOSE_PACKER_S. Idempotent; the destructor calls it.
    void close();

private:
    struct Rec;           // a slot, and what the listener keeps beside it
    struct RenderJob;
    // store.due(q) as _hints reads it, taken without the listener's lock
    // (the store's own lock is held by an association for its whole
    // transaction, and feed() must not wait on that): the passes due, or
    // the text of the exception the store raised.
    struct DueRead {
        std::vector<store::Due> due;
        std::optional<std::string> error;
    };
    struct Job {
        bool finish = false;
        std::shared_ptr<Rec> s;
    };

    std::string dir_;
    LiveConfig cfg_;
    std::vector<frame::FrameSpec> specs_;
    std::vector<frame::FrameSpec> sel_;
    bool joint_ = false;
    double dur_ = 0.0;
    std::shared_ptr<store::Store> store_;
    std::shared_ptr<passband::PassbandStore> passband_;
    Seams seams_;
    api::ReceiveConfig rx_;        // the live receives' (frames, estimator, rules)
    api::ReceiveConfig finish_rx_; // the end-of-slot receive's (finish_config)
    FeRing ring_;
    DialLog dial_;

    // -- under mu_
    mutable std::mutex mu_;
    frontend::StreamFE fe_;
    std::deque<std::pair<double, double>> early_;
    std::optional<std::pair<std::int64_t, std::int64_t>> quiet_;
    std::vector<std::pair<std::int64_t, std::int64_t>> silent_;
    std::optional<std::int64_t> silent_from_;
    std::vector<std::shared_ptr<Rec>> slots_;            // in insertion order (the reference's dict)
    std::set<std::int64_t> finished_;
    std::vector<Tile> done_tiles_;
    std::vector<std::shared_ptr<Rec>> recover_;
    std::map<std::string, header::HeaderFields> headers_;
    std::map<std::string, std::uint64_t> render_gen_;
    std::uint64_t next_gen_ = 0;
    std::set<std::pair<std::int64_t, std::string>> refresh_errors_;
    std::optional<std::string> codec_error_;
    double now_ = 0.0;
    std::optional<double> audio_end_;
    std::string source_;
    std::vector<std::string> lines_;
    std::vector<std::string> sink_queue_;
    std::deque<Job> live_jobs_;
    std::deque<Job> finish_jobs_;
    std::optional<std::string> busy_live_;
    std::optional<std::string> busy_finish_;
    bool stopping_ = false;
    bool closed_ = false;
    std::condition_variable work_cv_;
    mutable std::condition_variable idle_cv_;

    std::mutex sink_mu_;    // one thread hands lines to the sink at a time
    std::mutex state_mu_;   // one state.json write at a time
    std::mutex png_mu_;     // one picture file write at a time
    std::mutex close_mu_;
    std::thread live_thread_;
    std::thread finish_thread_;

    using Lock = std::unique_lock<std::mutex>;

    double clock_locked() const;
    void log_locked(const std::string& msg);
    void deliver();
    void load_previous();
    std::vector<Tile> tiles_locked() const;
    std::shared_ptr<Rec> slot_locked(std::int64_t q) const;
    double progress_locked(std::int64_t q) const;
    double interval_locked(const Rec& s, double now) const;
    std::vector<std::int64_t> active_slots_locked(double now) const;
    bool finish_idle_locked() const;
    bool idle_locked() const;
    std::optional<double> replay_end_locked() const;

    void note_silence(std::int64_t a, std::span<const double> x);
    void sound_back(std::int64_t a);
    std::vector<std::pair<std::int64_t, std::int64_t>> heard_runs(std::int64_t n0, std::int64_t n);

    void submit(Lock& lk, bool finish, const std::shared_ptr<Rec>& s);
    void worker(bool finish);
    void run_job(const Job& j);
    void job_failed(const Job& j, const std::string& text);
    void refresh(const std::shared_ptr<Rec>& s);
    void finish(const std::shared_ptr<Rec>& s);
    std::vector<types::Detection> candidates_for(const std::shared_ptr<Rec>& s, const receiver::Prepared& prep,
                                                 std::span<const std::uint8_t> ok, std::int64_t origin);
    void identify_provisional();

    // which signal is which (all under the lock)
    double centre(const types::Detection& d) const;
    double centre(const types::PassResult& p) const;
    std::int64_t waveform_of(const types::PassResult& p) const;
    std::pair<std::string, std::string> kind(const types::PassResult& p) const;
    const frame::FrameSpec& spec_of(const types::PassResult& p) const;
    Tile* match_locked(Rec& s, double f_hz, std::optional<std::int64_t> dial, const std::string& waveform);
    Tile& tile_for(Rec& s, double f_hz, std::optional<std::int64_t> dial, const std::string& waveform,
                   const std::optional<std::string>& frame);
    bool takes(Tile& t, const types::PassResult& p);
    void keep(Rec& s, const types::Detection& d);
    DueRead read_due(std::int64_t q) const;
    std::vector<acquire::Hint> hints_locked(std::int64_t q, std::optional<std::int64_t> dial, const DueRead& due);
    void known_header_in_place(Rec& s, types::PassResult& p, std::optional<std::int64_t> dial);
    bool plausible(const Rec& s, const types::PassResult& p, std::optional<std::int64_t> dial) const;
    std::optional<RenderJob> update_tile(Tile& t, const types::PassResult& p, double heard,
                                         const std::optional<store::Key>& key, std::optional<double> progress);
    Tile* find_tile_locked(const std::string& id);
    void render(const RenderJob& job);
};

// Feed `source` into `listener` until it ends (or *stop is set). realtime:
// "now" is the listener's clock (live audio); otherwise it is the time of
// the newest sample (a replay), each chunk waits until no receive is queued
// or running, and at the end of a replay time runs on until every slot it
// touched has finished. Logs "listening: ..." first and "audio ended" last,
// writing state.json as the reference does (after a step that changed
// something, else every 2 s, and with listening false at the end).
void run(LiveListener& listener, Source& source, bool realtime, const std::atomic<bool>* stop = nullptr);

}  // namespace sstvae::qrss::live

#endif  // SSTVAE_QRSS_LISTENER_HPP

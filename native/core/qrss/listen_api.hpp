// The live QRSSTVAE listener as a library: qrss_listen.py's main() (the
// reference) behind a Session an application starts, feeds and stops in
// process, and which `sstvae-qrss listen` (apps/sstvae_qrss/listen.cpp) is
// built on.
//
//   qrss_listen.py                   here
//   _parser's options, defaults      Options
//   build_config                     build_config (Refusal: its SystemExit texts)
//   main: the three dir_locks, the   Session::start (the store's and the
//     Store, the PassbandStore,        passband's refusals are Notice lines,
//     the LiveListener                 as the script prints them)
//   main: --wav / stdin              Options::source (the CLI's factory); with
//                                    none, the session's PushSource, fed by push()
//   backfill, recover, "tiles in"    the session's thread, then live::run
//   live.run(...)                    the session's thread
//   except KeyboardInterrupt         Session::interrupt(): "stopped",
//                                    write_state(listening=False)
//   end of stdin: run()'s            Session::stop(): run()'s "audio ended" and
//     "audio ended"                    write_state(listening=False)
//   finally: listener.close()        the bounded close (below)
//
// HOW AN APPLICATION USES IT (the desktop app's QRSS window, say):
//
//   auto s = listen::Session::start(opt, sink);   // GUI thread: refusals throw here
//   s->push(samples);        // the capture thread, every chunk; never blocks
//   s->tiles(); s->lines();  // the GUI thread, on a timer (or read state.json)
//   s->stop();               // the GUI thread, at most ~stop_bound_s
//
// Which thread calls what, what blocks, and for how long:
//   start       the caller's thread: the dir locks (a mkdir, an open, a 10-byte
//               write each), the stores' directories, the listener's
//               construction (tiles/, reading state.json back) and the
//               source factory (a WavSource reads its whole file: about a
//               second for 30 minutes). Milliseconds for live audio. Then it
//               starts the session's thread and returns; that thread reads
//               the passband store back into the ring (backfill, for live
//               audio: a fraction of a second), queues recover()'s slots,
//               says "tiles in DIR" and runs live::run.
//   push        any thread (one at a time keeps the stamps in order: they are
//               serialised anyway): the clock, the Stamper's arithmetic and a
//               queue append. Microseconds; never waits on the listener, and
//               the queue is unbounded, so nothing is dropped. After stop() it
//               does nothing and returns false.
//   stop,       any thread, not from inside the sink (from on_ended it returns
//   interrupt   at once). Ends the audio, waits for the session's thread to
//               write its last state and for the listener to close, at most
//               stop_bound_s in all, plus the time to write state.json once
//               if the deadline finds the session's thread writing it, and
//               to finish a write of the listener's (state.json, a picture)
//               that the deadline finds under way. What has not finished by
//               then is abandoned (a receive in flight: minutes for a FULL
//               slot, since the receivers take no cancel flag): its results
//               are dropped, the listener writes nothing more (below) and
//               the locks are released. A receive that does finish
//               within the bound is kept: its tiles, store passes and log
//               lines land after "audio ended", and state.json is written
//               once more (still not listening) so that it shows them.
//               One that comes before the session's thread has read the
//               passband store back skips that and recover() (what they would
//               start, it would only abandon). Idempotent; a second call
//               waits for the first one's end (the same bound).
//   backlog     any thread: the samples pushed and not yet taken by the
//               listener (a size under a mutex). Growing means the listener
//               is falling behind its audio.
//   tiles,      any thread: a copy under the listener's lock (microseconds).
//   lines
//   ~Session    stop(), then lets go: threads still busy with an abandoned
//               receive finish on their own (they own what they use).
//
// The sink is called from the caller's thread during start() (the lock
// notices) and from the session's and the listener's threads afterwards,
// one line at a time (never two at once), never with a lock of the
// listener's held; it must not call stop() or destroy the session. Nothing
// reaches it once a stop has abandoned work, and an abandoning stop waits
// for the line being delivered, so the sink must not wait on the thread
// that stops the session either (a blocking queued call into a GUI thread
// that is inside stop() deadlocks: queue the line, do not block on it). on_ended is called once, after
// the locks are released, on the thread that ended the session: the
// session's own thread for the end of a replay or of a source, the caller's
// for stop() and interrupt(). It may call stop() (which then returns at
// once) but must not destroy the session or wait_ended(), nor wait on the
// thread that stops or destroys the session (~Session joins the session's
// thread, which may be the one running it: a blocking queued call into the
// GUI thread that is destroying the session deadlocks): ended() turns true
// once it has returned, so a waiter sees its effects.
//
// Abandoning, precisely. The C++ receivers cannot be interrupted, so a stop
// that finds one running gives up on it, and it gives up on the listener
// first: LiveListener::abandon(), called while the session still holds
// every lock, after which the listener appends nothing to listener.log,
// writes no state.json and no picture, changes no tile, starts no job,
// applies nothing a running job returns (the line saying the slot failed
// included), begins no store association and starts no further hour of
// the passband store's packing; a write of its own under way when it is
// called lands before it returns. The session's wrapped seams then make
// every further stage of the receive throw at its start ("ListenerStopped")
// and draw no picture, and nothing more reaches the sink. So what a stop
// that abandoned leaves on disk is what was there when it returned, apart
// from three things, all of them work already begun:
//   * the receive itself runs to its end on the listener's thread (CPU
//     only: it writes nothing), which ~Session lets go;
//   * a store association already under way is waited for to the deadline,
//     and if it is still running then, it completes after stop() returned
//     and the store's lock is released when it ends (a session started on
//     that store meanwhile runs without one, with its notice);
//   * an hour the packer was packing is given up at its next FLAC frame,
//     unless it had passed its checks and was being swapped into place,
//     which completes (renames within the passband directory), possibly
//     after the locks are released.
// And one thing left unwritten: a deadline that finds the session's thread
// neither waiting for audio nor on its last writes (a read-back or a step
// that outlasts the bound) leaves it where it is, so state.json stays as
// that thread last wrote it, saying "listening"; the thread writes nothing
// more and stops at its next read.
//
// Departures from the script, each deliberate:
//   * interrupt() is the script's KeyboardInterrupt, and the CLI sends it on
//     SIGTERM too (the script dies there with nothing written); an interrupt
//     reaches the session's thread when it next asks for audio, so one that
//     comes while a replay waits for its receives feeds one more chunk first,
//     and one during a replay's last wait (after the recording's end) ends as
//     "audio ended", the stop having released that wait;
//   * the stop is bounded, where the script's close() waits up to 5 s for
//     its end-of-slot process and then exits under its daemon threads;
//   * with no renderer (a build without the codec) and pictures asked for,
//     the session runs without them and says so once (a Notice).
//
// Qt-free; in sstvae_qrss. No mutable globals.

#ifndef SSTVAE_QRSS_LISTEN_API_HPP
#define SSTVAE_QRSS_LISTEN_API_HPP

#include <condition_variable>
#include <cstddef>
#include <deque>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "qrss/listener.hpp"
#include "qrss/live.hpp"
#include "qrss/source.hpp"
#include "qrss/sources.hpp"

namespace sstvae::qrss::listen {

// How long stop() and interrupt() may take (about 1.5 s, as asked of the app's
// window, whose own fallback kills the process at 3 s).
inline constexpr double STOP_BOUND_S = 1.5;
// The last part of the bound (a quarter of it, at most), kept for abandoning
// what has not finished: the session's thread's last writes, a write of the
// listener's under way, an association under way.
inline constexpr double SETTLE_S = 0.25;
// The PushSource's name, as state.json's "source" shows it.
inline constexpr const char* PUSH_NAME = "app capture (8 kHz)";
// The notice for pictures asked for with no renderer.
inline constexpr const char* NO_PICTURES_NOTICE =
    "this build renders no pictures (the codec is not compiled in); the tiles have none, as with --no-pictures";

// A refusal: the script's SystemExit text (exit status 1 in the CLI).
class Refusal : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

enum class LineKind {
    log,      // a listener log line, "<utc> <msg>" (also appended to listener.log)
    notice,   // what the script prints itself: a lock it did not get, "tiles in DIR"
};
using Sink = std::function<void(LineKind kind, const std::string& text)>;

// What a source factory may use: the session's clock (Options::clock) and a
// sleep that returns early once the session is stopping (for a WavSource's
// pacing, so a stop does not wait out a second of it).
struct SourceContext {
    live::Clock clock;
    live::Sleep sleep;
};
using SourceFactory = std::function<std::unique_ptr<live::Source>(const SourceContext& ctx)>;
// How pictures are drawn for a configuration (codec::listener_renderer, in
// sstvae_qrss_codec); null for none.
using RendererFactory = std::function<std::shared_ptr<live::Renderer>(const live::LiveConfig& cfg)>;

struct Options {
    // --- qrss_listen.py's options, with its defaults --------------------------------------------
    std::string frame = "full";                  // --frame: either waveform's name for a shape
    bool just_ce = false;                        // --just-ce
    bool just_l = false;                         // --just-l
    double refresh_s = 120.0;                    // --refresh
    std::optional<std::string> state_dir;        // --state (default STORE/live; "" is ".", as Path(""))
    std::optional<std::string> store_dir;        // --store (absent or "": store::default_root())
    bool no_store = false;                       // --no-store
    std::optional<std::string> passband_dir;     // --passband (absent or "": STORE/passband)
    bool no_passband = false;                    // --no-passband
    double recover_hours = 3.0;                  // --recover-hours (live audio only)
    std::optional<std::string> model;            // --model (none: the renderer's default)
    std::string precision = "fp32";              // --precision: fp32, fp16 or int8
    bool render = true;                          // false: --no-pictures
    bool old_edge_noise = false;                 // --old-edge-noise (or $QRSS_OLD_EDGE_NOISE)
    bool old_gating = false;                     // --old-gating (or $QRSS_OLD_GATING)

    // --- where the audio comes from ---------------------------------------------------------------
    // Made by start() after the locks and the listener, where the script
    // makes its source (so a refusal it throws comes in the script's order).
    // Empty: the session's PushSource, fed by push(), live.
    SourceFactory source;
    // The source is live audio ("now" is the clock; backfill and recover
    // first). False for a replay. Ignored for the PushSource (always live).
    bool realtime = true;
    std::string push_name = PUSH_NAME;
    double push_hold_s = live::STAMP_HOLD_S;

    // --- seams --------------------------------------------------------------------------------------
    // Unix seconds (time.time); empty: live::system_time. The listener's
    // clock, the PushSource's stamps and backfill/recover's "now".
    live::Clock clock;
    // The listener's seams, for tests (stub receivers, a noise, the dial
    // guard...). Its clock, log and renderer and the two rules are the
    // session's own and are ignored here.
    live::Seams seams;
    // Pictures (when render): empty, or a factory that makes none, and the
    // session runs without them (config().render false) and says so once
    // (NO_PICTURES_NOTICE).
    RendererFactory renderer;
    double stop_bound_s = STOP_BOUND_S;
    std::function<void()> on_ended;
};

// qrss_listen.build_config: the listener's configuration for `opt`, with
// nothing started. Refusal with the script's texts: "--frame 'x': one of
// full, medium, short, tiny, l-full, l-medium, l-short, l-tiny"; "--frame
// 'full-v1' is CE only: waveform L has no frame of that shape (L frames:
// l-full, l-medium, l-short, l-tiny); add --just-ce to receive it";
// "--refresh must be positive" (refresh_s <= 0; NaN passes, as there). And
// argparse's for what its parser would have refused: "argument --just-l: not
// allowed with argument --just-ce", "argument --precision: invalid choice:
// 'x' (choose from 'fp32', 'fp16', 'int8')".
live::LiveConfig build_config(const Options& opt);
// The store's root: Path(--store) if --store else default_root(), as str.
std::string store_root(const Options& opt);
// The tile directory: --state if given, else STORE/live, as str.
std::string state_dir(const Options& opt);

// Audio pushed by the application, stamped at arrival exactly as
// live::StreamSource stamps what it reads (one push is one read: the clock
// as it arrives, through a live::Stamper with hold_s), queued without
// bound. Thread-safe; push() and end() never wait on the reader.
class PushSource final : public live::Source {
public:
    explicit PushSource(std::string name = PUSH_NAME, live::Clock clock = {}, double hold_s = live::STAMP_HOLD_S);

    // Mono 8 kHz samples (float32, as an f32 stream carries them). False
    // (and nothing done) after end(); an empty push stamps nothing.
    bool push(std::span<const float> x);
    // The end of the audio: what the Stamper holds goes out, then eof()
    // once the queue is read. Idempotent.
    void end();

    // live::Source. next: std::invalid_argument for a negative or NaN timeout.
    std::optional<live::Chunk> next(double timeout_s) override;
    bool eof() const override;
    std::string name() const override { return name_; }
    // What the clock or the Stamper threw, which ended the stream; null otherwise.
    std::exception_ptr error() const;
    // Samples queued for next() (what the Stamper still holds not counted).
    std::size_t backlog() const;

private:
    std::string name_;
    live::Clock clock_;
    live::Stamper stamper_;
    mutable std::mutex mu_;
    std::condition_variable cv_;
    std::deque<live::Chunk> q_;
    bool ended_ = false;
    std::exception_ptr error_;
};

class Session {
public:
    // qrss_listen.py's main() up to its run(): build_config (Refusal), the
    // tile directory's lock (Refusal "another listener is writing tiles to
    // <dir>; stop it, or give this one its own --state DIR"), the store's
    // (taken, or a Notice "another listener is using the store in <root>;
    // this one runs without a store (pass --store DIR for one of its own)")
    // and the passband store's (taken, or "another listener is writing the
    // passband store in <dir>; this one keeps none"), the stores, the
    // listener, the source (whatever its factory throws, a Refusal for the
    // CLI's refusals); then the session's thread. Anything thrown leaves
    // nothing running and every lock released. Errors of the directories:
    // rawfile::FileError (a lock's), npz::FileError (the store's),
    // std::filesystem::filesystem_error (tiles/), passband::FileError and
    // std::invalid_argument (the passband store's meta.json).
    static std::unique_ptr<Session> start(Options opt, Sink sink = {});
    ~Session();
    Session(const Session&) = delete;
    Session& operator=(const Session&) = delete;

    // Feed the PushSource (see PushSource::push). False when this session
    // has another source, or has stopped.
    bool push(std::span<const float> x);
    // The PushSource's backlog (PushSource::backlog); 0 for another source.
    std::size_t backlog() const;
    // The end of the audio (the script's end of stdin), bounded (above).
    void stop();
    // KeyboardInterrupt: "stopped" and state.json not listening, bounded.
    void interrupt();

    // Started and not yet ended.
    bool running() const;
    // Ended: by the end of its source, an error, stop() or interrupt(); the
    // locks are released then, and on_ended has returned.
    bool ended() const;
    // Block until ended() (true), or timeout_s passes (false).
    bool wait_ended(std::optional<double> timeout_s = std::nullopt) const;
    // Ended by interrupt() ("stopped").
    bool interrupted() const;
    // A stop gave up on work still running (above).
    bool abandoned() const;
    // What ended the session's thread, if an exception (the script's
    // traceback): an error of the listener's (state.json unwritable, say)
    // or of the source. Null otherwise.
    std::exception_ptr error() const;

    const std::string& state_dir() const;
    // The store's root and the passband store's directory, when this
    // session holds them (nullopt: --no-store, --no-passband, or another
    // listener had it).
    std::optional<std::string> store_root() const;
    std::optional<std::string> passband_dir() const;
    const live::LiveConfig& config() const;
    // The listener's tiles (newest slot first) and its last log lines.
    std::vector<live::Tile> tiles() const;
    std::vector<std::string> lines() const;

    struct Core;   // (shared with the session's threads)

private:
    explicit Session(std::shared_ptr<Core> c) : c_(std::move(c)) {}
    std::shared_ptr<Core> c_;
};

}  // namespace sstvae::qrss::listen

#endif  // SSTVAE_QRSS_LISTEN_API_HPP

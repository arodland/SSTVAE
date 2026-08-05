// Continuous live reception: a rolling buffer in, decoded pictures out.
//
// Port of `sstvae/rx/engine.py`, whose docstring describes the strategy
// and still governs: every poll interval, try the preamble path
// (`Modem::demodulate`, needs the transmission's start to be in the
// buffer) and fall back to the preamble-free blind path
// (`Modem::demodulate_blind`, works from any long-enough mid-stream
// excerpt via the beacon side-channel). Because the buffer holds history
// from before sync was acquired, a mid-stream lock still decodes the
// frames that arrived *before* it -- retrospective decoding, not just
// from-here-on.
//
// A reception is finished when a fully-synced decode reports all its
// frames, when decoded progress stops advancing for `end_grace`
// seconds, or when the buffer holds audio past the point where the last
// frame of that transmission could still be arriving -- whichever comes
// first. All three run against the tracked reception retained *across*
// polls (`Pending`, in the .cpp), so none of them depends on the current
// poll having produced a decode at all; see `Pending` for why that is
// the whole ballgame.
//
// Headless, and knows nothing about audio devices or any UI: it reads a
// `RingBuffer` somebody else is filling, publishes live status into a
// `SharedState`, and hands finished receptions to a `Sink`. **Whether a
// finished reception is written to disk is the sink's decision, not the
// loop's** -- the CLI always saves, while a GUI may have an autosave
// checkbox and hold the picture for a Save button instead.
//
// Three things differ from the reference on purpose:
//
//   * **The decoder is a seam** (`Decoder`), where Python imports
//     `codec.reconstruct` directly. That keeps this file out of the
//     optional codec library -- the loop is pure state machine and
//     builds offline with `-DSSTVAE_BUILD_CODEC=OFF` -- and it lets the
//     tests drive the entire state machine with a stub decoder, so the
//     bookkeeping that decides "is this a new reception or one already
//     saved" is checkable without onnxruntime.
//   * **The lock is not optional.** Python's `SharedState` exposes a
//     mutex and a convention; here every read and write goes through
//     `get`/`update`, so there is no unlocked path to forget to take.
//   * **Waiting out a transmission does not copy the buffer.**
//     `decode_loop_low_cpu` polls `total_written()` rather than calling
//     `snapshot()` for its sample count, which in the reference copies
//     the whole 8 MB history once a second to read one integer.

#ifndef SSTVAE_RX_ENGINE_HPP
#define SSTVAE_RX_ENGINE_HPP

#include <condition_variable>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "config.hpp"
#include "images/types.hpp"
#include "modem/modem.hpp"
#include "rx/ringbuffer.hpp"
#include "util/event.hpp"

namespace sstvae::rx {

// Don't attempt a decode until there is at least this much audio.
inline constexpr double MIN_SECONDS_BEFORE_ATTEMPT = 3.0;

// How close two acquisitions' transmission-start sample positions must
// be to count as "the same reception" rather than a new one.
inline constexpr double SAME_RECEPTION_EPSILON_S = 1.0;

struct RxConfig {
    std::string out_dir = "received";
    double poll_interval = 5.0;
    double end_grace = 8.0;
    // Downscale saved pictures, e.g. {320, 240}. Unset = full size.
    std::optional<std::pair<int, int>> size;
    bool once = false;
    // A cap, not a fixed timescale: sync::BlindAccumulator runs one
    // decay timescale per mode (config::MODES), each capped at
    // min(mode.duration_s, blind_search_seconds), so by default (above
    // every mode's own duration) no mode's timescale is capped at all --
    // see decode_loop. Only useful to *shrink* below a mode's own
    // duration (a fast synthetic test's short buffer, e.g.); there is no
    // reliability reason to raise it past the longest mode's duration,
    // since there is no more real signal beyond that to integrate.
    double blind_search_seconds = config::MODES[config::N_MODES - 1].duration_s;

    // Widen the preamble-free search to config::BLIND_WIDE_MAX_OFFSET_HZ,
    // for a counterpart whose dial is off by hundreds of Hz. Opt-in
    // because unlike the preamble path -- which searches frequency for
    // free and so is always wide -- this one searches CFO directly and
    // its cost is linear in the number of bins.
    bool blind_wide = false;

    // Follow a carrier that moves during the transmission. Off by
    // default; the two gains suit different things, see
    // config::DRIFT_* and docs/todo.md.
    modem::DriftTrack drift_track = modem::DriftTrack::Off;
    // The largest fraction of wall time the loop may spend decoding.
    // **1.0 is off**, and off is the historical behaviour exactly: poll
    // every `poll_interval` no matter how long a poll takes.
    //
    // Below 1.0 the loop waits longer after a slow poll, so it backs
    // off on a machine where a decode costs seconds instead of
    // milliseconds. That is a phone: the desktop's decode is ~50 ms
    // against a 5 s interval (1% duty) and needs nothing, while on a
    // mid-range Android the same decode can approach the interval
    // itself, at which point the device is decoding continuously, the
    // UI starves, and -- the part that matters -- the extra polls buy
    // nothing, since each one re-decodes a picture that has grown by
    // one interval's worth of frames.
    //
    // Adaptive rather than a larger constant because the spread across
    // devices is the whole problem: a number slow enough for the worst
    // phone would make the best one needlessly stale, and neither is
    // knowable from here. The measurement is already published as
    // `Progress::last_decode_s`.
    double max_decode_duty = 1.0;
};

// How long `decode_loop` waits before its next poll, given what the
// last one cost. Exposed rather than kept internal so the backoff can
// be checked as arithmetic: the alternative is a test that runs the
// loop with a slow decoder and asserts on elapsed time, which would be
// asserting on latency instead of on the decision.
double poll_wait(const RxConfig& config, double last_cost_s);

// (stall metric, progress fraction) for one blind decode. Exposed for
// the same reason `poll_wait` is: it is arithmetic, and the alternative
// is inferring it from a whole decode run.
//
// Two different questions, deliberately answered by two different
// numbers. The metric is the count of confidently-received latents (see
// count_confident in the .cpp for why confidence is what makes it
// usable as a stall detector). The fraction is how far *into* the
// transmission we have got -- the last frame that decoded, over the
// frames expected -- and is not that count over the total. A count
// reads as a completion percentage and is not one: the erasures this
// path lives with (a fade, or simply not having heard the start) hold
// it down permanently, so a reception already at the transmission's
// last frame reports 70% and the bar never fills. The interleaver is
// why the two differ at all -- each frame's latents are scattered
// across the whole picture, so only the frame index says "how far".
//
// The denominator is mode C's frame count, the longest: the blind path
// has no header, so the real mode is unknown.
struct BlindProgress {
    int metric = 0;
    double frac = 0.0;
};
BlindProgress blind_progress(std::span<const double> weights_full);

// The `threading.Event` the reference stops on. Shared with the
// transmitter, which needs the same primitive for its cancel flag and
// its watchdog; the name stays because "stop flag" is what it is here.
using StopFlag = util::Event;

enum class Status { Listening, Receiving, Done };

const char* status_name(Status s);

// A reception the loop considers finished.
struct Reception {
    images::Picture image;
    // Unset when the mode is unknown, which is the blind path: there is
    // no header, so neither the mode nor the total frame count is known.
    std::optional<std::string> mode_name;
    std::string callsign;
    double snr_db = 0.0;
    std::optional<int> frames_received;
    std::optional<int> n_frames_expected;
};

// What the loop publishes for a UI to display.
//
// The picture is shared rather than copied because a waterfall reads
// this many times a second and a 640x480 RGB copy is 900 kB. Shared as
// `const`, so a displayed frame can never be mutated underneath the
// thread drawing it.
struct Progress {
    Status status = Status::Listening;
    std::optional<std::string> mode_name;
    std::optional<int> frames_received;
    std::optional<int> n_frames_expected;
    double progress_frac = 0.0;
    std::string callsign;
    double snr_db = 0.0;  // NaN until something has been measured
    std::shared_ptr<const images::Picture> image;
    std::optional<std::string> saved_path;
    double seconds_captured = 0.0;
    double last_decode_s = 0.0;
    // Completed poll cycles. Not in the reference; it is here because
    // "the receiver is alive but hearing nothing" and "the receiver is
    // wedged" are otherwise indistinguishable from outside, and because
    // it is what lets the tests assert on the state machine's decisions
    // without asserting on how fast it makes them.
    std::uint64_t polls = 0;

    Progress();
};

class SharedState {
public:
    Progress get() const;
    // Mutate under the lock. The only way in; see the header comment.
    void update(const std::function<void(Progress&)>& fn);

private:
    mutable std::mutex m_;
    Progress p_;
};

// Where a finished reception goes. Returns the path it was saved to, if
// the sink chose to save it at all.
using Sink = std::function<std::optional<std::string>(const Reception&)>;

// Where `decode_loop_diversity`'s per-latent branch-contribution image
// goes (see `sstvae::modem::diversity::contribution_image`). Takes the
// image and the picture's own saved path (nullopt if the picture sink
// declined to save), and returns the path it wrote, if any.
using DebugImageSink =
    std::function<std::optional<std::string>(const images::Picture&,
                                              const std::optional<std::string>&)>;

// Latents + weights -> picture. Supplied by the caller so this library
// need not link the codec; `sstvae-listen` passes one that calls
// `OnnxCodec::decode`.
using Decoder = std::function<images::Picture(std::span<const double> latents,
                                              std::span<const double> weights)>;

// "  SNR 12.3dB", or nothing when no SNR has been measured yet.
//
// Public because the GUI's status line and the CLI's stdout have to
// agree: two spellings of the same number read as two different
// measurements to an operator comparing notes on the air.
std::string fmt_snr(double snr_db);

// Parse a "WxH" size, as `receive.save_size` stores it. Nothing for an
// empty or unreadable string, which means "keep the full size" -- a
// typo must not silently produce a 0x0 picture.
std::optional<std::pair<int, int>> parse_size(const std::string& text);

// Unique output path, millisecond resolution.
//
// Millisecond and not second because two short-mode receptions can
// finish within the same second (both already sat complete in the
// buffer), and a second-resolution name would have the later silently
// overwrite the earlier.
std::string timestamped_path(const std::string& out_dir);

// The default sink: write every finished reception into a directory and
// report it on stdout. Exactly what the CLI listener did before saving
// became the sink's job.
Sink save_to_dir_sink(std::string out_dir,
                      std::optional<std::pair<int, int>> size = std::nullopt,
                      bool verbose = true);

// Writes the contribution image beside the picture it belongs to
// (`<name>_diversity.png`); a nullopt saved path (the picture sink
// declined to save) means there is nothing to write beside, so it
// writes nothing and returns nullopt.
DebugImageSink save_debug_image_to_dir_sink(bool verbose = true);

// Blocks until `stop` is set. Both loops are exception-safe in the sense
// that matters to a receiver: a decode that throws for one poll is not
// allowed to end the session.
void decode_loop(RingBuffer& ring, const Decoder& decode, SharedState& state,
                 const RxConfig& config, StopFlag& stop, const Sink& sink);

// Header-only variant: no blind fallback, and so no retrospective
// mid-stream decoding. While idle it searches only the newly-arrived
// slice of audio each poll rather than the whole buffer; once it locks
// it does no signal processing at all until enough audio has been
// captured for the whole transmission, then decodes and saves once.
void decode_loop_low_cpu(RingBuffer& ring, const Decoder& decode,
                         SharedState& state, const RxConfig& config,
                         StopFlag& stop, const Sink& sink);

// Two-branch counterpart of `decode_loop`: independently finds and
// demodulates the same transmission from `rings[0]` and `rings[1]` --
// different antennas/audio devices, independent noise and fading, no
// assumption of phase lock between them -- then maximal-ratio combines
// the branches (`sstvae::modem::diversity::combine_diversity_results`)
// before decoding. The two rings are assumed to start filling at the
// same wall-clock moment, so a reception position recorded in one
// ring's coordinate is directly comparable to the other's; see
// docs/diversity-reception.md.
//
// Kept separate from `decode_loop` rather than generalized to N rings:
// that function's state machine is the reference's load-bearing one
// (CLAUDE.md), and this keeps it unchanged for the single-device case
// at the cost of duplication here -- the same tradeoff the Python port
// makes (`sstvae/rx/engine.py`'s `decode_loop_diversity`).
//
// Each branch independently prefers a header lock and falls back to a
// blind one, same as `decode_loop` does for a single receiver
// (`find_branch_reception`) -- so a branch too weak for the preamble
// path can still contribute once enough audio has accumulated for a
// beacon superframe. Branches are matched by their *position*
// (`reception_start`, always expressed at the preamble's sample offset
// whichever path found it) within `SAME_RECEPTION_EPSILON_S`, the same
// criterion `decode_loop` itself uses -- for two blind locks this
// position agreement is a sanity check ("are these really the same
// transmission"), not an alignment requirement, since
// `BlindDemodResult::latents`/`weights` are already aligned by the
// beacon's absolute frame counter regardless of sample position (see
// `combine_diversity_results`). If only one branch locks (the other
// never acquires at all, or its lock is more than
// `SAME_RECEPTION_EPSILON_S` away -- most likely a spurious hit), that
// branch is used alone, same erasure/weight semantics as
// `combine_diversity_results` given a single branch.
//
// Progress/completion tracking mirrors `decode_loop`'s own preference: a
// header-locked combine (the common case) reports an exact frame count
// and finishes when it is reached; an all-blind combine (true duration
// unknown) finishes when progress stops advancing for `config.end_grace`
// seconds, the same stall detection `decode_loop` uses for its own
// blind path.
//
// `debug_sink`, if non-null, is handed `contribution_image` for every
// finished reception where both branches actually contributed --
// skipped when only one branch locked, since there is nothing to
// compare. `rings` must have exactly two elements.
void decode_loop_diversity(std::span<RingBuffer* const> rings, const Decoder& decode,
                           SharedState& state, const RxConfig& config,
                           StopFlag& stop, const Sink& sink,
                           const DebugImageSink* debug_sink = nullptr);

}  // namespace sstvae::rx

#endif

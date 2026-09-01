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
// frames, or when the buffer holds audio past the point where the last
// frame of that transmission could still be arriving -- whichever comes
// first. Progress stopping for `end_grace` seconds does not finish it:
// it *delivers* it, so autosave never waits on a signal that may not
// come back, and leaves it tracked until its scheduled end so a fade it
// recovers from still contributes. All of it runs against the tracked
// reception retained *across* polls (`Pending`, in the .cpp), so none of
// it depends on the current poll having produced a decode at all; see
// `Pending` for why that is the whole ballgame.
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

// (stall metric, frames decoded) for one decode's weights. Exposed for
// the same reason `poll_wait` is: it is arithmetic, and the alternative
// is inferring it from a whole decode run.
//
// The **metric** is the count of confidently-received latents (see
// decode_progress in the .cpp for why confidence is what makes it
// usable). **This is the number the stall clock watches, and nothing
// else may be substituted for it**: anything positional does not move
// when retrospective decoding fills in frames behind the furthest one,
// and any count over the *whole* legal frame range climbs on buffer
// growth alone, so a reception fed either would never stall or never
// stop stalling.
//
// **frames_decoded** is how many of the transmission's frames carried
// confident data. It is a *fill*, and it is what a UI shows beside the
// progress bar rather than as it: a fill reads as a completion
// percentage without being one, since the erasures both paths live with
// (a fade, or simply not having heard the start) hold it down
// permanently. The interleaver is why the two differ at all -- each
// frame's latents are scattered across the whole picture, so a latent
// count answers "how much" and only a frame index answers "how far".
struct DecodeProgress {
    int metric = 0;
    int frames_decoded = 0;
};
DecodeProgress decode_progress(std::span<const double> weights_full);

// How far through its schedule a transmission has got: the progress
// bar's numerator, and pure arithmetic on buffer positions.
//
// This is what the bar shows, because it is the only one of the
// available numbers that climbs with the clock rather than with the
// decoder's luck. The header path already reported nearly this --
// `DemodResult::frames_received` counts every frame whose samples are
// in the buffer, signal or noise -- so this is that number generalized
// to the blind path, which had been showing how far its furthest
// *decoded* frame reached and so stalled on the erasures that are its
// normal state.
//
// It counts from the transmission's own first frame, not from the audio
// we happened to capture: joining a transmission late leaves its early
// frames permanently unavailable, and a bar that starts at 60% and
// fills to 100% is honest where one that starts at 0 and can never
// reach 100 is not. The frames the join missed show up instead as the
// gap against frames_decoded, exactly like a fade's.
int frames_elapsed(std::int64_t start, std::int64_t total, int n_frames);

// The `threading.Event` the reference stops on. Shared with the
// transmitter, which needs the same primitive for its cancel flag and
// its watchdog; the name stays because "stop flag" is what it is here.
using StopFlag = util::Event;

// Waiting is a reception that lost sync and has already been delivered,
// but whose scheduled end has not arrived: neither receiving a signal
// (there isn't one) nor idle (a picture is still open for the rest of
// its frames). See `Pending` in the .cpp.
enum class Status { Listening, Receiving, Waiting, Done };

const char* status_name(Status s);

// A reception the loop considers finished.
struct Reception {
    images::Picture image;
    // Unset when the mode is unknown, which is the blind path: there is
    // no header, so neither the mode nor the total frame count is known.
    std::optional<std::string> mode_name;
    std::string callsign;
    double snr_db = 0.0;
    // How far through its schedule the transmission got, and how much
    // of it actually decoded. The first is the progress bar's numerator
    // and climbs with the clock; the second is a fill, and the gap
    // between them is what a fade -- or joining late -- costs. See
    // frames_elapsed and decode_progress.
    std::optional<int> frames_received;
    std::optional<int> frames_decoded;
    std::optional<int> n_frames_expected;
    // Set when this same reception has already been delivered once, to
    // the path named here: a fade ended it early, it recovered before
    // its scheduled end, and this is the better decode. **Replace what
    // is there rather than adding a second picture** -- one
    // transmission is one file, one gallery entry, one notification.
    std::optional<std::string> saved_path;
    // True on every delivery after the first, whether or not the sink
    // chose to save. saved_path cannot carry this by itself: a sink
    // that declined to save (a GUI with autosave off) returns no path,
    // and a redelivery would then read as a brand-new reception -- two
    // "reception complete" records for one transmission. saved_path
    // says where to write; this says whether it is the same reception
    // again.
    bool redelivery = false;
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
    // frames_received is how far the transmission has got (what the bar
    // shows); frames_decoded is how many frames carried confident data
    // (shown beside it, never as it). See frames_elapsed.
    std::optional<int> frames_received;
    std::optional<int> frames_decoded;
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
    // Blind-path observability, refreshed every poll the blind branch
    // runs (NaN / false when it didn't). The score is the accumulator's
    // best prominence whether or not it clears BLIND_SCORE_THRESHOLD --
    // a below-threshold score is otherwise invisible in live operation,
    // and a receiver that fails to acquire on real hardware gives no
    // number to compare against the threshold's calibration.
    // blind_locked distinguishes the two ways the blind path can be
    // silently stuck: score below threshold, and locked with the beacon
    // not decoding -- which mean opposite things (the second is a
    // payload/format problem, e.g. a pre-PROTOCOL_VERSION-4 sender, not
    // a weak signal).
    double blind_score = 0.0;  // NaN when the blind branch didn't run
    bool blind_locked = false;

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

}  // namespace sstvae::rx

#endif

#include "rx/engine.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <deque>
#include <filesystem>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "dsp/dsp.hpp"
#include "framing/framing.hpp"
#include "images/images.hpp"
#include "latents/latents.hpp"
#include "modem/modem.hpp"
#include "sync/sync.hpp"

namespace sstvae::rx {

namespace {

using Clock = std::chrono::steady_clock;
using config::FRAME_SAMPLES;
using config::FS;
using config::HEADER_SAMPLES;
using config::PREAMBLE_SAMPLES;

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

// Records how long a poll took, on every path out of the loop body --
// including the early `continue`s, which are cheap and must not leave
// an expensive poll's cost standing as the estimate.
class PollCost {
public:
    PollCost(double& out) : out_(out), t0_(Clock::now()) {}
    ~PollCost() { out_ = std::chrono::duration<double>(Clock::now() - t0_).count(); }

private:
    double& out_;
    Clock::time_point t0_;
};

// Mode C, the longest -- the fallback frame count wherever the real
// mode is unknown (a beacon carrying a mode index this receiver doesn't
// recognize; ordinarily the beacon's mode field names it).
constexpr int kMaxModeFrames = config::MODES[config::N_MODES - 1].n_frames;

// Weight a blind-path latent must clear to count as "confidently
// received" for the stall-detection progress metric -- see its use
// below. Matches the "good" cutoff tests already judge latents by.
constexpr double kProgressWeightThreshold = 0.5;

// How many finished receptions to remember. They only need to outlive
// their own audio in the ring buffer; 50 is the reference's bound.
constexpr std::size_t kFinishedHistory = 50;

double seconds_since(Clock::time_point t0) {
    return std::chrono::duration<double>(Clock::now() - t0).count();
}

std::optional<std::pair<double, double>> window_s(std::int64_t lo, std::int64_t hi) {
    return std::make_pair(static_cast<double>(lo) / FS, static_cast<double>(hi) / FS);
}

bool already_finished(std::int64_t pos, const std::deque<std::int64_t>& finished,
                      std::int64_t epsilon) {
    return std::any_of(finished.begin(), finished.end(),
                       [&](std::int64_t k) { return std::llabs(pos - k) <= epsilon; });
}

// Local [lo, hi) spans of the buffer with already-saved receptions'
// preambles carved out.
//
// Only the *preamble region* of a finished reception is excluded, not
// its whole duration: two transmissions can overlap in time, and
// blanking a finished one's full extent would hide an overlapping
// neighbour's preamble along with it.
std::vector<std::pair<std::int64_t, std::int64_t>> free_spans(
    std::int64_t n, std::int64_t buf_start,
    const std::deque<std::int64_t>& finished, std::int64_t epsilon) {
    std::vector<std::pair<std::int64_t, std::int64_t>> blocked;
    for (const std::int64_t p : finished) {
        const std::int64_t lo = p - buf_start - epsilon;
        const std::int64_t hi = p - buf_start + PREAMBLE_SAMPLES + epsilon;
        if (hi > 0 && lo < n) blocked.emplace_back(std::max<std::int64_t>(0, lo),
                                                   std::min<std::int64_t>(n, hi));
    }
    std::sort(blocked.begin(), blocked.end());

    std::vector<std::pair<std::int64_t, std::int64_t>> spans;
    std::int64_t cur = 0;
    for (const auto& [lo, hi] : blocked) {
        if (lo > cur) spans.emplace_back(cur, lo);
        cur = std::max(cur, hi);
    }
    if (cur < n) spans.emplace_back(cur, n);
    return spans;
}

struct FoundReception {
    modem::DemodResult result;
    std::int64_t start;  // absolute, in ring-buffer coordinates
};

// Decode the strongest preamble that is neither already saved nor a
// spurious peak.
//
// `sync::acquire` returns a single global argmax inside its window, so
// an already-decoded transmission still sitting in the buffer can
// outrank and hide a second one. Searching only *forward* of finished
// hits does not fix it either -- the strongest peak is often the later
// transmission, and stepping past it buries every earlier one. So:
// search each still-unclaimed span, and within a span step past any peak
// whose header will not decode (a correlation artefact inside a
// transmission's own frames rather than a real preamble).
//
// `demodulate` gets the same window the hit came from. It runs its own
// acquisition, and left to scan the whole buffer it can lock a different
// preamble than the one just vetted -- which is how an already-saved
// reception ends up decoded and written out a second time while the
// bookkeeping records some other position.
std::optional<FoundReception> find_new_reception(
    const modem::Modem& modem, std::span<const double> samples,
    std::span<const std::complex<double>> z, std::int64_t buf_start,
    const std::deque<std::int64_t>& finished, std::int64_t epsilon,
    modem::DriftTrack drift_track = modem::DriftTrack::Off, int max_tries = 4) {
    const auto n = static_cast<std::int64_t>(samples.size());
    int tries = 0;
    for (const auto& [span_lo, span_hi] : free_spans(n, buf_start, finished, epsilon)) {
        std::int64_t lo = span_lo;
        while (lo < span_hi && tries < max_tries) {
            sync::Acquisition acq{};
            try {
                acq = sync::acquire(z, config::PREAMBLE_THRESHOLD,
                                    config::ACQUIRE_MAX_BINS,
                                    sync::SearchWindow{lo, span_hi});
            } catch (const sync::SyncError&) {
                break;
            }
            ++tries;
            try {
                modem::DemodResult r =
                    modem.demodulate(samples, window_s(lo, span_hi), drift_track);
                const std::int64_t start = buf_start + r.preamble_start;
                return FoundReception{std::move(r), start};
            } catch (const sync::SyncError&) {
                lo = acq.preamble_start + PREAMBLE_SAMPLES;
            }
        }
    }
    return std::nullopt;
}

int count_nonzero(const std::vector<double>& v) {
    return static_cast<int>(std::count_if(v.begin(), v.end(),
                                          [](double w) { return w != 0.0; }));
}

void remember_finished(std::deque<std::int64_t>& finished, std::int64_t pos) {
    finished.push_back(pos);
    while (finished.size() > kFinishedHistory) finished.pop_front();
}

// The reception `decode_loop` is currently tracking, retained from one
// poll to the next.
//
// This exists so the three "is it finished?" tests can fire on a poll
// that decoded *nothing*, which is the whole reason it is a record
// rather than a handful of locals recomputed per poll. A reception stops
// producing a decode long before it stops being real: its audio scrolls
// out of the ring buffer, or -- much sooner -- its blind acquisition
// score falls back under BLIND_SCORE_THRESHOLD as the accumulator's
// evidence decays once the transmission is over. If the completion tests
// are only asked on polls that produced a decode, then in exactly the
// case that matters they are never asked again: the loop sits in
// Receiving forever, the sink is never called, and the picture that
// *was* decoded is never delivered or saved. An indefinite hang and
// autosave never firing are that one bug seen from two ends, so a poll
// that decodes nothing must go on counting against the reception rather
// than being skipped over.
//
// `image` and the fields beside it are the last good decode, held so the
// reception can still be delivered after its decodes have stopped.
// `deadline_abs` is a buffer position, not a time: past it, no real
// frame of this transmission can still be arriving (see decode_loop).
struct Pending {
    std::int64_t start = 0;         // absolute preamble-start position
    std::int64_t deadline_abs = 0;  // absolute buffer position
    // Whether the *most recent* decode of this reception came from the
    // blind path. Per-poll, not per-reception: a reception first found
    // blind can later decode over the preamble path (or vice versa),
    // and the stall-on-no-advance test below must follow the path
    // currently driving the record.
    bool blind = false;
    int metric = 0;                 // best progress metric seen
    // When the metric last stopped advancing; unset while it still is.
    std::optional<Clock::time_point> stable_since;
    std::shared_ptr<const images::Picture> image;
    std::optional<std::string> mode_name;
    std::optional<int> frames_received, n_frames_expected;
    std::string callsign;
    double snr_db = kNaN;
};

// Back to "listening", with the per-reception fields cleared.
void reset_to_listening(SharedState& state) {
    state.update([](Progress& s) {
        s.status = Status::Listening;
        s.mode_name.reset();
        s.frames_received.reset();
        s.n_frames_expected.reset();
        s.progress_frac = 0.0;
        s.callsign.clear();
        s.snr_db = kNaN;
    });
}

}  // namespace

BlindProgress blind_progress(std::span<const double> weights_full,
                             int n_frames_expected) {
    // The metric counts confidently-received latents only, not bare
    // nonzero ones: demodulate_blind assigns *some* nonzero weight to
    // essentially every legal abs_frame slot its ever-growing search
    // range touches, real signal or not (just small for noise, after
    // the med_h fix in modem.cpp), so a nonzero count keeps climbing
    // every poll purely from the buffer growing -- independent of
    // whether any new real data has arrived -- until buffer growth has
    // mapped the *entire* legal abs_frame range. That read as "stuck
    // receiving forever": the stall detector never saw a stable metric
    // to end_grace against.
    //
    // One past the highest absolute frame index any confidently-received
    // latent belongs to, or 0 if none does. See the header for why that,
    // and not the count, is the bar.
    const std::span<const std::int32_t> frame_of = framing::frame_of_latent();
    const std::size_t n = std::min(weights_full.size(), frame_of.size());
    BlindProgress out;
    int last = -1;
    for (std::size_t i = 0; i < n; ++i) {
        if (weights_full[i] > kProgressWeightThreshold) {
            ++out.metric;
            last = std::max(last, frame_of[i]);
        }
    }
    out.reach = last + 1;
    out.frac = static_cast<double>(out.reach) / n_frames_expected;
    return out;
}

double poll_wait(const RxConfig& config, double last_cost_s) {
    if (config.max_decode_duty >= 1.0 || last_cost_s <= 0.0) {
        return config.poll_interval;
    }
    // Floored, so a config asking for an absurd duty backs off by a
    // bounded amount rather than by minutes.
    const double d = std::max(0.05, config.max_decode_duty);
    return std::max(config.poll_interval, last_cost_s * (1.0 / d - 1.0));
}

std::string fmt_snr(double snr_db) {
    if (std::isnan(snr_db)) return "";
    char buf[32];
    std::snprintf(buf, sizeof buf, "  SNR %.1fdB", snr_db);
    return buf;
}

std::optional<std::pair<int, int>> parse_size(const std::string& text) {
    const std::size_t x = text.find('x');
    if (x == std::string::npos) return std::nullopt;
    try {
        std::size_t used = 0;
        const int w = std::stoi(text.substr(0, x), &used);
        if (used != x) return std::nullopt;
        const std::string tail = text.substr(x + 1);
        const int h = std::stoi(tail, &used);
        if (used != tail.size() || w <= 0 || h <= 0) return std::nullopt;
        return std::pair<int, int>{w, h};
    } catch (const std::exception&) {
        return std::nullopt;
    }
}

// --- primitives -------------------------------------------------------------

const char* status_name(Status s) {
    switch (s) {
        case Status::Listening: return "listening";
        case Status::Receiving: return "receiving";
        case Status::Done: return "done";
    }
    return "listening";
}

Progress::Progress() : snr_db(kNaN), blind_score(kNaN) {}

Progress SharedState::get() const {
    std::lock_guard<std::mutex> lock(m_);
    return p_;
}

void SharedState::update(const std::function<void(Progress&)>& fn) {
    std::lock_guard<std::mutex> lock(m_);
    fn(p_);
}

std::string timestamped_path(const std::string& out_dir) {
    using namespace std::chrono;
    const auto now = system_clock::now();
    const auto ms = duration_cast<milliseconds>(now.time_since_epoch()) % 1000;
    const std::time_t t = system_clock::to_time_t(now);
    std::tm tm{};
#if defined(_WIN32)
    localtime_s(&tm, &t);
#else
    localtime_r(&t, &tm);
#endif
    char stamp[32];
    std::strftime(stamp, sizeof stamp, "%Y%m%d_%H%M%S", &tm);

    char name[64];
    std::snprintf(name, sizeof name, "rx_%s_%03d.png", stamp,
                  static_cast<int>(ms.count()));
    return (std::filesystem::path(out_dir) / name).string();
}

Sink save_to_dir_sink(std::string out_dir, std::optional<std::pair<int, int>> size,
                      bool verbose) {
    return [out_dir = std::move(out_dir), size,
            verbose](const Reception& rec) -> std::optional<std::string> {
        const std::string path = timestamped_path(out_dir);
        std::filesystem::create_directories(std::filesystem::path(path).parent_path());
        const images::Picture& img =
            size ? images::resize(rec.image, size->first, size->second) : rec.image;
        images::save_png(img, path);
        if (verbose) {
            std::printf("saved %s (mode=%s, callsign=%s%s)\n", path.c_str(),
                        rec.mode_name ? rec.mode_name->c_str() : "unknown, blind sync",
                        rec.callsign.empty() ? "(none)" : rec.callsign.c_str(),
                        fmt_snr(rec.snr_db).c_str());
            std::fflush(stdout);
        }
        return path;
    };
}

// --- the full loop ----------------------------------------------------------

void decode_loop(RingBuffer& ring, const Decoder& decode, SharedState& state,
                 const RxConfig& config, StopFlag& stop, const Sink& sink) {
    const modem::Modem modem;
    const auto epsilon = static_cast<std::int64_t>(SAME_RECEPTION_EPSILON_S * FS);

    // Absolute (ring-buffer-coordinate) transmission-start position of
    // every reception already handled this run. The ring buffer goes on
    // holding a finished reception's audio for the whole buffer length
    // afterwards, so without this a still-buffered transmission would be
    // rediscovered, re-decoded and re-saved on every following poll.
    std::deque<std::int64_t> finished_starts;

    // The reception being tracked, unset while listening. Progress is
    // per-reception: carrying a previous transmission's metric across to
    // the next one can make a brand-new reception look as though it has
    // already stalled, and end it early -- so a different reception gets
    // a fresh record rather than an updated one.
    std::optional<Pending> pending;
    // Blind acquisition's persistent search state: a BlindAccumulator
    // folds in only the audio that is new since the last poll (see
    // sync::BlindAccumulator), so unlike the preamble path it carries
    // state across iterations of this loop instead of re-deriving
    // everything from the current snapshot each time. blind_acc_pushed
    // is the absolute (ring-buffer-coordinate) sample count already
    // folded in; unset means "nothing pushed yet, or the last push's
    // position fell out of the ring buffer's retained window" -- either
    // way there is a gap push() cannot bridge contiguously, so the right
    // response is a fresh accumulator over what is available now rather
    // than guessing at what filled it.
    std::optional<sync::BlindAccumulator> blind_acc;
    std::optional<std::int64_t> blind_acc_pushed;
    double last_poll_cost_s = 0.0;

    while (!stop.is_set()) {
        if (stop.wait(poll_wait(config, last_poll_cost_s))) break;
        const PollCost cost(last_poll_cost_s);

        std::uint64_t total = 0;
        const std::vector<double> samples = ring.snapshot(&total);
        state.update([&](Progress& s) {
            s.seconds_captured = static_cast<double>(total) / FS;
        });
        if (samples.size() < MIN_SECONDS_BEFORE_ATTEMPT * FS) continue;
        state.update([](Progress& s) { ++s.polls; });
        const auto buf_start =
            static_cast<std::int64_t>(total) - static_cast<std::int64_t>(samples.size());

        const Clock::time_point t0 = Clock::now();
        std::vector<double> latents_full, weights_full;
        bool have_latents = false;
        std::optional<std::string> mode_name;
        std::optional<int> n_frames_expected, frames_received;
        std::string callsign;
        double snr_db = kNaN;
        double progress_frac = 0.0;
        int progress_metric = 0;
        std::optional<std::int64_t> reception_start;
        double blind_score = kNaN;
        bool blind_locked = false;

        // Preamble path first: find and decode the strongest reception
        // that has not already been handled. Falls through to the blind
        // path if nothing there decodes (a corrupted header, or the only
        // hits are spurious correlation peaks).
        std::optional<FoundReception> found;
        try {
            const std::vector<std::complex<double>> z = dsp::to_baseband(samples);
            found = find_new_reception(modem, samples, z, buf_start, finished_starts,
                                       epsilon, config.drift_track);
        } catch (const std::exception&) {
            // One bad poll must not end the session.
            found = std::nullopt;
        }

        if (found) {
            const modem::DemodResult& r = found->result;
            latents_full = latents::pad_to_full(r.latents);
            weights_full = latents::pad_to_full(r.weights);
            have_latents = true;
            mode_name = std::string(r.mode.name);
            n_frames_expected = r.mode.n_frames;
            frames_received = r.frames_received;
            callsign = r.callsign;
            snr_db = r.snr_db;
            progress_frac = static_cast<double>(r.frames_received) / r.mode.n_frames;
            progress_metric = r.frames_received;
            reception_start = found->start;
        } else {
            // Fold whatever is new since the last poll into the running
            // accumulator -- O(new samples), not O(window) -- which is
            // what lets this run one decay timescale per mode (see
            // RxConfig::blind_search_seconds) rather than a single
            // one-size-fits-all window. The retrospective decode below
            // still covers the whole current buffer once locked, exactly
            // as before.
            if (!blind_acc || !blind_acc_pushed || *blind_acc_pushed < buf_start) {
                std::vector<std::optional<double>> timescales;
                for (const auto& mode : config::MODES)
                    timescales.push_back(
                        std::min(mode.duration_s, config.blind_search_seconds));
                // BLIND_SCORE_THRESHOLD from config, not a literal: a
                // hardcoded 4.0 survived here after the threshold moved
                // to 9.0 with the v3 pilot -- the fifth copy of exactly
                // the constant-drift hazard docs/todo-done.md records
                // four of -- leaving this loop accepting blind locks the
                // Python engine refuses.
                blind_acc.emplace(config.blind_wide ? config::BLIND_WIDE_MAX_OFFSET_HZ
                                                    : config::BLIND_MAX_OFFSET_HZ,
                                  config::BLIND_BIN_STEP_HZ, 8,
                                  config::BLIND_SCORE_THRESHOLD, std::nullopt,
                                  std::move(timescales));
                blind_acc_pushed = buf_start;
            }
            const auto new_lo = *blind_acc_pushed - buf_start;
            if (new_lo < static_cast<std::int64_t>(samples.size())) {
                const std::vector<double> new_raw(
                    samples.begin() + new_lo, samples.end());
                const std::vector<std::complex<double>> new_chunk =
                    dsp::to_baseband_at(new_raw, *blind_acc_pushed);
                blind_acc->push(new_chunk, *blind_acc_pushed);
                blind_acc_pushed = static_cast<std::int64_t>(total);
            }

            blind_score = blind_acc->best_score();
            std::optional<sync::BlindAcquisition> ba;
            try {
                // origin = buf_start: the accumulator folds in absolute
                // (ring-coordinate) phase, but the phase is about to be
                // used against `samples`, which starts at buf_start.
                // Without this the two coordinates agree only while the
                // session is younger than the ring buffer -- see
                // BlindAccumulator::result for the field failure that
                // found it.
                ba = blind_acc->result(buf_start);
            } catch (const sync::SyncError&) {
                ba = std::nullopt;
            }
            blind_locked = ba.has_value();

            std::optional<modem::BlindDemodResult> rb;
            if (ba) {
                try {
                    rb = modem.demodulate_blind(samples, std::nullopt, ba,
                                                config.drift_track);
                } catch (const sync::SyncError&) {
                    rb = std::nullopt;
                } catch (const std::exception&) {
                    rb = std::nullopt;
                }
            }

            if (rb && rb->beacon && rb->frame0_start) {
                // Record every reception in the same coordinate -- the
                // preamble start -- so finished_starts stays homogeneous.
                // The blind path locates absolute frame 0, which sits one
                // preamble+header later; without this correction one
                // transmission gets two labels 768 samples apart, and
                // free_spans blocks the wrong region.
                reception_start =
                    buf_start + *rb->frame0_start - PREAMBLE_SAMPLES - HEADER_SAMPLES;
                // Already handled: treat it as nothing decoded rather
                // than skipping the rest of the poll, so that whatever
                // reception is *currently* pending still gets its
                // completion tests run this time round.
                if (already_finished(*reception_start, finished_starts, epsilon)) {
                    reception_start.reset();
                } else {
                    latents_full = std::move(rb->latents);
                    weights_full = std::move(rb->weights);
                    have_latents = true;
                    callsign = rb->callsign;
                    snr_db = rb->snr_db;
                    // The beacon's mode field (PROTOCOL_VERSION 4) names
                    // the transmission's real mode, so the blind path no
                    // longer assumes mode C: the deadline, the progress
                    // denominator and the reported mode are all exact.
                    // An unknown index (a future mode) keeps the old
                    // assume-mode-C behaviour.
                    const int mi = rb->beacon->mode_index;
                    if (mi >= 0 && mi < config::N_MODES) {
                        const auto& spec =
                            config::MODES[static_cast<std::size_t>(mi)];
                        mode_name = std::string(spec.name);
                        n_frames_expected = spec.n_frames;
                    }
                    const BlindProgress bp = blind_progress(
                        weights_full,
                        n_frames_expected ? *n_frames_expected : kMaxModeFrames);
                    progress_metric = bp.metric;
                    progress_frac = bp.frac;
                    // The reach doubles as the reported frame counter --
                    // see BlindProgress for why leaving this unset froze
                    // half of every status line once the beacon's mode
                    // field supplied the other half.
                    frames_received = bp.reach;
                }
            }
        }

        const double decode_s = seconds_since(t0);
        state.update([&](Progress& s) {
            s.last_decode_s = decode_s;
            s.blind_score = blind_score;
            s.blind_locked = blind_locked;
        });

        bool advanced = false;
        const bool decoded = have_latents && reception_start.has_value();
        if (decoded) {
            // A different reception than the one being tracked: it takes
            // over, with a fresh record. Its very first poll must not
            // inherit the previous one's stall clock, or it can be
            // mistaken for a reception that has already stopped
            // advancing and be ended immediately.
            if (!pending || std::llabs(*reception_start - pending->start) > epsilon) {
                Pending p;
                p.start = *reception_start;
                pending = std::move(p);
            }
            // The deterministic backstop. The transmission's start is
            // known exactly -- from the header path's own acquisition,
            // or from the beacon on the blind path, both in the same
            // "preamble start" coordinate -- so the latest a real
            // frame of it can possibly still be arriving is its own
            // duration after that point, fixed the moment the start
            // is. The mode comes from the header, or from the beacon's
            // mode field on the blind path; only a beacon whose mode
            // index this receiver doesn't know (a future mode) falls
            // back to mode C's duration, the longest. That is unlike
            // "progress stopped advancing" below, which rides on the
            // decoder's own noise floor and is not guaranteed to ever
            // settle. Once the buffer holds audio past this deadline
            // there is provably no more real signal left to arrive
            // for this reception, done or not. Recomputed every decoded
            // poll (from pending->start, stable across polls) so a mode
            // learned on a later poll tightens it.
            const auto deadline_frames =
                static_cast<std::int64_t>(n_frames_expected ? *n_frames_expected
                                                            : kMaxModeFrames);
            pending->deadline_abs = pending->start + PREAMBLE_SAMPLES + HEADER_SAMPLES +
                                    deadline_frames * FRAME_SAMPLES;
            pending->blind = !found;
            if (progress_metric > pending->metric) {
                pending->metric = progress_metric;
                advanced = true;
            }
            pending->image = std::make_shared<const images::Picture>(
                decode(latents_full, weights_full));
            pending->mode_name = mode_name;
            pending->frames_received = frames_received;
            pending->n_frames_expected = n_frames_expected;
            pending->callsign = callsign;
            pending->snr_db = snr_db;
            state.update([&](Progress& s) {
                s.status = Status::Receiving;
                s.mode_name = mode_name;
                s.frames_received = frames_received;
                s.n_frames_expected = n_frames_expected;
                s.progress_frac = std::min(progress_frac, 1.0);
                s.callsign = callsign;
                s.snr_db = snr_db;
                s.image = pending->image;
            });
        }

        if (!pending) {
            state.update([](Progress& s) { s.status = Status::Listening; });
            continue;
        }

        // Two things end a reception short of its own frame count, and
        // both are timed against end_grace.
        //
        // It stopped decoding. That is how a reception actually ends in
        // the field: its audio scrolls out of the ring buffer, or its
        // blind acquisition score falls back under
        // BLIND_SCORE_THRESHOLD as the accumulator's evidence decays
        // once the transmission is over. A poll that decoded nothing is
        // not a neutral event -- it is the absence of progress, and is
        // timed as such. Resetting the clock there instead, which is
        // what the loop used to do from the branch that skipped the rest
        // of the poll entirely, is why a reception in that state could
        // never satisfy any completion test however long it sat.
        //
        // Or -- blind only, where there is no frames_received count to
        // finish on -- its decoded progress stopped advancing.
        // Deliberately *not*
        // extended to the header path: a spurious preamble-shaped lock
        // in noise goes on decoding the same handful of frames for as
        // long as it is looked at, and ending that on a stall would turn
        // a false lock into a reported, and autosaved, picture of noise
        // (see test_noise_produces_nothing). The header path does not
        // need it anyway -- frames_received climbs as the buffer grows,
        // so a real transmission reaches its count, and the deadline
        // below is the backstop for one that does not. Keyed on
        // pending->blind rather than "mode unknown": since the beacon's
        // mode field, the blind path knows the mode too, so
        // n_frames_expected no longer distinguishes the paths.
        const bool no_progress = !decoded || (pending->blind && !advanced);
        if (no_progress) {
            if (!pending->stable_since) pending->stable_since = Clock::now();
        } else {
            pending->stable_since.reset();
        }

        // Header path only: its frames_received is a contiguous decoded
        // count, so reaching the total genuinely means everything
        // arrived. The blind path's frames_received is a *reach* -- the
        // furthest frame decoded, with erasures routinely behind it --
        // and a reach at the last frame is exactly when retrospective
        // backfill is still improving the picture, so ending on it
        // would trade picture quality for nothing (the deadline already
        // bounds the wait).
        const bool complete = !pending->blind && pending->n_frames_expected &&
                              pending->frames_received &&
                              *pending->frames_received >= *pending->n_frames_expected;
        const bool stalled = pending->stable_since &&
                             seconds_since(*pending->stable_since) >= config.end_grace;
        if (!complete && !stalled &&
            static_cast<std::int64_t>(total) < pending->deadline_abs) {
            continue;
        }

        // Deliver only what has something in it. A reception that ended
        // with no confidently-received latent at all is not a picture,
        // and is deliberately *not* recorded in finished_starts either:
        // there is nothing to protect against re-saving, and a real
        // transmission that was merely weak on its first polls must stay
        // findable rather than be blocked out for as long as its audio
        // is in the buffer.
        const bool delivered = pending->metric > 0 && pending->image;
        if (delivered) {
            Reception rec;
            rec.image = *pending->image;
            rec.mode_name = pending->mode_name;
            rec.callsign = pending->callsign;
            rec.snr_db = pending->snr_db;
            rec.frames_received = pending->frames_received;
            rec.n_frames_expected = pending->n_frames_expected;
            const std::optional<std::string> saved_path = sink(rec);

            // Bookkeeping, not disk: this reception has been *handled*,
            // so it must never be rediscovered while its audio is still
            // in the buffer -- whether or not the sink chose to save it.
            remember_finished(finished_starts, pending->start);
            state.update([&](Progress& s) {
                s.status = Status::Done;
                s.saved_path = saved_path;
            });
        }
        pending.reset();

        if (delivered && config.once) {
            stop.set();
            break;
        }

        if (delivered) stop.wait(2.0);
        reset_to_listening(state);
    }
}

// --- the cheap loop ---------------------------------------------------------

void decode_loop_low_cpu(RingBuffer& ring, const Decoder& decode, SharedState& state,
                         const RxConfig& config, StopFlag& stop, const Sink& sink) {
    const modem::Modem modem;
    // Margin so a preamble cannot be missed by straddling the boundary
    // between one poll's search window and the next.
    constexpr double kSearchOverlapS = 2.0;
    std::int64_t last_search_pos = 0;  // absolute, ring.total_written coordinate

    while (!stop.is_set()) {
        if (stop.wait(config.poll_interval)) break;

        std::uint64_t total = 0;
        std::vector<double> samples = ring.snapshot(&total);
        if (samples.size() < MIN_SECONDS_BEFORE_ATTEMPT * FS) continue;
        auto buf_start =
            static_cast<std::int64_t>(total) - static_cast<std::int64_t>(samples.size());

        const std::int64_t search_from_abs = std::max(
            buf_start, last_search_pos - static_cast<std::int64_t>(kSearchOverlapS * FS));
        const std::int64_t search_lo = search_from_abs - buf_start;
        const auto search_hi = static_cast<std::int64_t>(samples.size());

        state.update([&](Progress& s) {
            ++s.polls;
            s.seconds_captured = static_cast<double>(total) / FS;
            if (s.status != Status::Receiving) s.status = Status::Listening;
        });

        sync::Acquisition acq{};
        try {
            acq = sync::acquire(dsp::to_baseband(samples), config::PREAMBLE_THRESHOLD,
                                config::ACQUIRE_MAX_BINS,
                                sync::SearchWindow{search_lo, search_hi});
        } catch (const std::exception&) {
            last_search_pos = static_cast<std::int64_t>(total);
            continue;
        }

        modem::DemodResult r;
        try {
            // The same window the hit came from, so demodulate cannot
            // lock a different (older, already-saved) preamble than the
            // one just found -- see find_new_reception above.
            r = modem.demodulate(samples, window_s(search_lo, search_hi),
                                 config.drift_track);
        } catch (const std::exception&) {
            last_search_pos = static_cast<std::int64_t>(total);
            continue;  // a spurious preamble-shaped hit; keep listening
        }

        const std::int64_t reception_start = buf_start + acq.preamble_start;
        const std::int64_t frames_end_abs = reception_start + PREAMBLE_SAMPLES +
                                            HEADER_SAMPLES +
                                            std::int64_t{r.mode.n_frames} * FRAME_SAMPLES;

        state.update([&](Progress& s) {
            s.status = Status::Receiving;
            s.mode_name = std::string(r.mode.name);
            s.frames_received = r.frames_received;
            s.n_frames_expected = r.mode.n_frames;
            s.progress_frac =
                std::min(static_cast<double>(r.frames_received) / r.mode.n_frames, 1.0);
            s.callsign = r.callsign;
            s.snr_db = r.snr_db;
        });

        // No further DSP until the whole transmission should have
        // arrived -- just wait, updating the status text cheaply. The
        // reference calls snapshot() here and throws the samples away;
        // total_written() is the same number without copying 8 MB.
        //
        // Bounded, because this waits on something outside the loop's
        // control: if capture stops -- the device is unplugged, the
        // stream dies, the host stops delivering callbacks -- total_now
        // stops advancing, and an unbounded wait here holds the receiver
        // in Receiving forever waiting for audio that will never come,
        // with no picture ever handed to the sink. The bound is how long
        // the audio still missing should take to arrive, plus end_grace;
        // past it, decode whatever did arrive.
        const Clock::time_point wait_deadline =
            Clock::now() +
            std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(
                static_cast<double>(frames_end_abs - static_cast<std::int64_t>(total)) / FS +
                config.end_grace));
        while (!stop.is_set()) {
            const std::uint64_t total_now = ring.total_written();
            if (static_cast<std::int64_t>(total_now) >= frames_end_abs) break;
            if (Clock::now() >= wait_deadline) break;
            state.update(
                [&](Progress& s) { s.seconds_captured = static_cast<double>(total_now) / FS; });
            stop.wait(std::min(1.0, config.poll_interval));
        }
        if (stop.is_set()) break;

        samples = ring.snapshot(&total);
        buf_start =
            static_cast<std::int64_t>(total) - static_cast<std::int64_t>(samples.size());
        // Never look for this reception's preamble again: without this,
        // the next poll resumes from where the search stood *before*
        // waiting out the transmission, re-finds the preamble that is
        // still in the buffer, and decodes and saves the same picture a
        // second time.
        last_search_pos = frames_end_abs;
        // Re-anchor the window on this reception in the grown buffer's
        // coordinates, so the final decode cannot lock a different
        // preamble.
        const std::int64_t lo = std::max<std::int64_t>(0, reception_start - buf_start);
        try {
            r = modem.demodulate(samples,
                                 window_s(lo, static_cast<std::int64_t>(samples.size())),
                                 config.drift_track);
        } catch (const std::exception&) {
            // The transmission was cut short or corrupted after all; go
            // back to listening rather than end the loop.
            state.update([](Progress& s) { s.status = Status::Listening; });
            continue;
        }

        auto img = std::make_shared<const images::Picture>(
            decode(latents::pad_to_full(r.latents), latents::pad_to_full(r.weights)));

        Reception rec;
        rec.image = *img;
        rec.mode_name = std::string(r.mode.name);
        rec.callsign = r.callsign;
        rec.snr_db = r.snr_db;
        rec.frames_received = r.frames_received;
        rec.n_frames_expected = r.mode.n_frames;
        const std::optional<std::string> saved_path = sink(rec);

        state.update([&](Progress& s) {
            s.status = Status::Done;
            s.image = img;
            s.frames_received = r.frames_received;
            s.progress_frac =
                std::min(static_cast<double>(r.frames_received) / r.mode.n_frames, 1.0);
            s.snr_db = r.snr_db;
            s.saved_path = saved_path;
        });

        if (config.once) {
            stop.set();
            break;
        }

        stop.wait(2.0);
        reset_to_listening(state);
    }
}

}  // namespace sstvae::rx

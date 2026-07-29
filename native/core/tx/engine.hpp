// The transmit sequence: encode, modulate, key the rig, play, unkey.
//
//     fit 640x480 -> encode -> modulate -> PTT on -> lead delay
//     -> play -> tail delay -> PTT off
//
// Port of `sstvae/tx/engine.py`. **The one rule this file exists to
// enforce is that PTT always comes back down.** A cancelled
// transmission, an exception anywhere in the chain, an audio device that
// stops calling back with the USB cable half out -- all of them must
// unkey. So the keyed region is a scope guard, *and* an independent
// watchdog thread drops PTT if the transmission runs past its known
// duration by a margin. A stuck transmitter is a hazard to the band and
// to the radio's finals, and it is not acceptable to rely on the happy
// path for that.
//
// The watchdog is not redundancy for its own sake: the scope guard only
// runs if control returns, and the failure it is there for is exactly
// the one where control does not.
//
// Headless and synchronous: `transmit()` blocks for the duration of the
// transmission (32-95 s), so callers run it on a worker thread and watch
// it through the state callback.
//
// As in `rx/engine.hpp`, the parts that would drag in a heavyweight
// dependency are seams: `Encoder` keeps onnxruntime out of this library
// and `Player` keeps the audio stack out, so the whole sequence --
// including the PTT guarantee -- is testable with neither a soundcard
// nor a radio nor a downloaded model.

#ifndef SSTVAE_TX_ENGINE_HPP
#define SSTVAE_TX_ENGINE_HPP

#include <atomic>
#include <functional>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "config.hpp"
#include "images/types.hpp"
#include "util/event.hpp"

namespace sstvae::tx {

// How long after the audio should have finished before the watchdog
// concludes something is wedged and unkeys anyway. Generous, because a
// resampled stream on a busy machine can legitimately lag by seconds.
inline constexpr double WATCHDOG_MARGIN_S = 15.0;

enum class TxPhase {
    Idle,
    Encoding,    // neural encoder; no progress fraction available
    Modulating,
    Keying,      // PTT up, waiting out the lead delay
    Sending,     // audio playing; progress is meaningful here
    Unkeying,
    Done,
    Cancelled,
    Failed,
};

const char* phase_name(TxPhase p);

// Thrown out of `prepare` when the operator cancelled mid-encode.
// Distinct from a failure because it is not an error to report -- the
// phase becomes Cancelled, not Failed.
class Cancelled : public std::runtime_error {
public:
    Cancelled() : std::runtime_error("transmission cancelled") {}
};

struct TxState {
    TxPhase phase = TxPhase::Idle;
    double progress = 0.0;  // 0..1, only meaningful during Sending
    std::string message;

    bool active() const {
        return phase != TxPhase::Idle && phase != TxPhase::Done &&
               phase != TxPhase::Cancelled && phase != TxPhase::Failed;
    }
};

struct TxConfig {
    std::string mode = "B";
    std::string callsign;
    std::string device;      // audio output device; empty = the default
    double level = 0.9;      // output peak, 0..1
    double ptt_lead_s = 0.3; // PTT up -> audio start (relay + ALC settling)
    double ptt_tail_s = 0.3; // audio end -> PTT down
    // Exposed rather than compiled in for the same reason
    // `Modem::modulate` exposes its clip headroom: the reference's tests
    // shorten it by patching a module constant, which is unreachable
    // from a test here. A watchdog whose only test has to wait out the
    // real 15 s margin is a watchdog that ends up untested.
    double watchdog_margin_s = WATCHDOG_MARGIN_S;
};

// Key or unkey. Null means no rig control -- audio only, VOX, or a
// dummy load.
using Ptt = std::function<void(bool on)>;

// Picture -> mode C's full-length unit-RMS latent vector.
using Encoder = std::function<std::vector<double>(const images::ImageArray&)>;

// Play a waveform, blocking until it finishes. Returns false if it
// stopped early because `should_stop` went true. `on_progress` is
// called from the audio callback.
using Player = std::function<bool(
    const std::string& device, std::span<const double> wave, int samplerate,
    const std::function<void(double)>& on_progress,
    const std::function<bool()>& should_stop,
    const std::function<void(const std::string&)>& on_error)>;

using OnState = std::function<void(const TxState&)>;
// May be called from the transmitting thread, from the audio callback,
// or from the watchdog thread -- so an implementation that touches a UI
// has to marshal, exactly as the reference's does.
using OnError = std::function<void(const std::string&)>;

// Drops PTT unconditionally if it is still up after `timeout_s`.
//
// Independent of the transmit path on purpose: it exists precisely for
// the cases where that path is stuck and its scope guard is never going
// to run. Python can make this a daemon thread and walk away; here the
// destructor stands it down and joins, which the event makes immediate
// rather than a wait for the timeout.
class PttWatchdog {
public:
    PttWatchdog(Ptt ptt, double timeout_s, std::function<void()> on_fire)
        : ptt_(std::move(ptt)), timeout_s_(timeout_s), on_fire_(std::move(on_fire)) {}

    ~PttWatchdog() {
        cancel();
        if (thread_.joinable()) thread_.join();
    }

    PttWatchdog(const PttWatchdog&) = delete;
    PttWatchdog& operator=(const PttWatchdog&) = delete;

    void start() {
        if (!ptt_) return;  // nothing to unkey
        thread_ = std::thread([this] {
            if (done_.wait(timeout_s_)) return;  // stood down in time
            try {
                ptt_(false);
            } catch (...) {
                // Already the emergency path; there is nowhere better to
                // report to than the callback below.
            }
            if (on_fire_) on_fire_();
        });
    }

    void cancel() { done_.set(); }

private:
    Ptt ptt_;
    double timeout_s_;
    std::function<void()> on_fire_;
    util::Event done_;
    std::thread thread_;
};

// Scale the modulator's output to the configured peak.
//
// Deliberately a plain peak scale and nothing else. `Modem::modulate`
// has already done the envelope clipping and band-limiting that sets the
// waveform's ~4.2 dB PAPR (config::CLIP_HEADROOM_DB, dsp::tx_condition);
// anything further here -- another clip, a compressor, a normalize to
// full scale -- would undo that conditioning and spray splatter into the
// adjacent channel.
std::vector<double> condition_for_output(std::span<const double> x, double level);

// One transmission at a time.
class TxEngine {
public:
    TxEngine(Ptt ptt, Player player, Encoder encode, OnState on_state = {},
             OnError on_error = {});

    TxState state() const;

    // Ask an in-flight transmission to stop. Safe from any thread.
    void cancel();
    bool cancelled() const;

    // Picture -> transmit waveform. Pure computation: no rig, no audio.
    // Separable so a UI can precompute while the operator is still
    // deciding, and so a test can check the waveform without keying
    // anything. Throws on failure, including on cancellation.
    std::vector<double> prepare(const images::Picture& image, const TxConfig& config);

    // The whole sequence. True if the transmission completed, false if
    // it was cancelled or failed. Does not throw for ordinary failures:
    // the phase becomes Failed and `on_error` is called, because the
    // caller is a worker thread whose only sensible response is to
    // report it.
    bool transmit(const images::Picture& image, const TxConfig& config);

private:
    bool keyed_send(const std::vector<double>& wave, const TxConfig& config);
    bool cancelled_result();
    void set(TxPhase phase, double progress, std::string message);
    void set_phase(TxPhase phase, std::string message);
    void key(bool on);
    void report_error(const std::string& msg) const;

    Ptt ptt_;
    Player player_;
    Encoder encode_;
    OnState on_state_;
    OnError on_error_;
    util::Event cancel_;

    // Phase and progress are atomic because progress is written from the
    // audio callback while a UI thread reads the pair. `message_` is not:
    // it is written only by the transmitting thread, and never between
    // the call into the player and its return -- so it is stable for the
    // whole window in which the callback can run.
    std::atomic<TxPhase> phase_{TxPhase::Idle};
    std::atomic<double> progress_{0.0};
    std::string message_;
};

}  // namespace sstvae::tx

#endif

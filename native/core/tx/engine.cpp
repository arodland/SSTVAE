#include "tx/engine.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <utility>

#include "images/images.hpp"
#include "modem/modem.hpp"

namespace sstvae::tx {

namespace {

using config::FS;

std::string sending_message(double duration_s) {
    char buf[64];
    std::snprintf(buf, sizeof buf, "sending (%.0f s)", duration_s);
    return buf;
}

}  // namespace

const char* phase_name(TxPhase p) {
    switch (p) {
        case TxPhase::Idle: return "idle";
        case TxPhase::Encoding: return "encoding";
        case TxPhase::Modulating: return "modulating";
        case TxPhase::Keying: return "keying";
        case TxPhase::Sending: return "sending";
        case TxPhase::Unkeying: return "unkeying";
        case TxPhase::Done: return "done";
        case TxPhase::Cancelled: return "cancelled";
        case TxPhase::Failed: return "failed";
    }
    return "idle";
}

std::vector<double> condition_for_output(std::span<const double> x, double level) {
    std::vector<double> out(x.begin(), x.end());
    double peak = 0.0;
    for (double v : out) peak = std::max(peak, std::abs(v));
    if (peak <= 0) return out;
    const double g = level / peak;
    for (double& v : out) v *= g;
    return out;
}

TxEngine::TxEngine(Ptt ptt, Player player, Encoder encode, OnState on_state,
                   OnError on_error)
    : ptt_(std::move(ptt)),
      player_(std::move(player)),
      encode_(std::move(encode)),
      on_state_(std::move(on_state)),
      on_error_(std::move(on_error)) {}

TxState TxEngine::state() const {
    TxState s;
    s.phase = phase_.load(std::memory_order_acquire);
    s.progress = progress_.load(std::memory_order_relaxed);
    s.message = message_;
    return s;
}

void TxEngine::cancel() { cancel_.set(); }

bool TxEngine::cancelled() const { return cancel_.is_set(); }

void TxEngine::set(TxPhase phase, double progress, std::string message) {
    progress_.store(progress, std::memory_order_relaxed);
    message_ = std::move(message);
    phase_.store(phase, std::memory_order_release);
    if (on_state_) on_state_(state());
}

void TxEngine::set_phase(TxPhase phase, std::string message) {
    set(phase, progress_.load(std::memory_order_relaxed), std::move(message));
}

void TxEngine::report_error(const std::string& msg) const {
    if (on_error_) on_error_(msg);
}

void TxEngine::key(bool on) {
    if (!ptt_) return;
    try {
        ptt_(on);
    } catch (const std::exception& e) {
        // Failing to key is a normal, reportable problem. Failing to
        // *unkey* is an emergency the operator has to know about now.
        if (on) {
            report_error(std::string("PTT on failed: ") + e.what());
        } else {
            report_error(std::string("PTT OFF FAILED: ") + e.what() +
                         " -- the rig may still be transmitting. Unkey it manually.");
        }
    }
}

std::vector<double> TxEngine::prepare(const images::Picture& image,
                                      const TxConfig& config) {
    const config::ModeSpec& spec = modem::mode_by_name(config.mode);

    set(TxPhase::Encoding, 0.0, "encoding image");
    const std::vector<double> flat = encode_(images::to_array(images::fit(image)));

    if (cancel_.is_set()) throw Cancelled();

    set(TxPhase::Modulating, 0.0,
        std::string("modulating mode ") + std::string(spec.name));
    if (flat.size() < static_cast<std::size_t>(spec.n_latents)) {
        throw std::runtime_error("encoder produced too few latents for this mode");
    }
    const modem::Modem m;
    const std::vector<double> wave = m.modulate(
        std::span<const double>(flat.data(), static_cast<std::size_t>(spec.n_latents)),
        spec, true, config.callsign);
    return condition_for_output(wave, config.level);
}

bool TxEngine::transmit(const images::Picture& image, const TxConfig& config) {
    cancel_.clear();
    std::vector<double> wave;
    try {
        wave = prepare(image, config);
    } catch (const Cancelled&) {
        set(TxPhase::Cancelled, 0.0, "cancelled");
        return false;
    } catch (const std::exception& e) {
        report_error(std::string("could not prepare transmission: ") + e.what());
        set(TxPhase::Failed, 0.0, e.what());
        return false;
    }

    if (cancel_.is_set()) {
        set(TxPhase::Cancelled, 0.0, "cancelled");
        return false;
    }

    return keyed_send(wave, config);
}

bool TxEngine::cancelled_result() {
    set_phase(TxPhase::Cancelled, "cancelled");
    return false;
}

bool TxEngine::keyed_send(const std::vector<double>& wave, const TxConfig& config) {
    const double duration_s = static_cast<double>(wave.size()) / FS;
    PttWatchdog watchdog(
        ptt_, config.ptt_lead_s + duration_s + config.ptt_tail_s + config.watchdog_margin_s,
        [this] {
            report_error(
                "PTT watchdog fired: transmission overran its expected duration, "
                "forcing the rig back to receive");
        });

    // The one guarantee this class makes. A scope guard rather than a
    // try/finally, so it holds for every exit from the keyed region --
    // return, exception, or the compiler's idea of one we did not think
    // of. The watchdog is stood down first, so a normal finish cannot
    // race it into firing.
    struct Unkey {
        TxEngine* self;
        PttWatchdog* watchdog;
        ~Unkey() {
            watchdog->cancel();
            self->key(false);
        }
    } unkey{this, &watchdog};

    try {
        set(TxPhase::Keying, 0.0, "keying rig");
        key(true);
        watchdog.start();
        if (cancel_.wait(config.ptt_lead_s)) return cancelled_result();

        set(TxPhase::Sending, 0.0, sending_message(duration_s));
        const bool completed = player_(
            config.device, std::span<const double>(wave.data(), wave.size()), FS,
            // Called from the audio callback: a relaxed store plus the
            // caller's callback. No allocation, no locks, no I/O --
            // underruns live here.
            [this](double frac) {
                progress_.store(frac, std::memory_order_relaxed);
                if (on_state_) on_state_(state());
            },
            [this] { return cancel_.is_set(); },
            [this](const std::string& msg) { report_error(msg); });
        if (!completed) return cancelled_result();

        set(TxPhase::Unkeying, 1.0, "unkeying");
        cancel_.wait(config.ptt_tail_s);
    } catch (const std::exception& e) {
        report_error(std::string("transmission failed: ") + e.what());
        set_phase(TxPhase::Failed, e.what());
        return false;
    }

    set(TxPhase::Done, 1.0, "sent");
    return true;
}

}  // namespace sstvae::tx

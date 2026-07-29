// The transmit sequence, and above all its one guarantee.
//
// Every test here that keys the rig checks where PTT ended up, because
// that is the property the module exists for: a stuck transmitter is a
// hazard to the band and to the radio's finals. The interesting cases
// are the ones where the transmission does *not* go well -- the player
// throwing, the operator cancelling, unkeying itself failing -- so those
// outnumber the happy path.
//
// No soundcard, no radio, no model: the engine takes its encoder, its
// player and its PTT as seams, so all three are stubs that record what
// they were asked to do. The waveform is still real, and the happy-path
// test demodulates it to prove that "sent" means a decodable
// transmission rather than an empty buffer.

#include <atomic>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "check.hpp"
#include "config.hpp"
#include "modem/modem.hpp"
#include "tx/engine.hpp"

using namespace sstvae;

namespace {

images::Picture test_picture() {
    // Already the target geometry, so `fit` is a copy and nothing here
    // depends on the scaler.
    images::Picture p(images::IMG_W, images::IMG_H);
    for (std::size_t i = 0; i < p.rgb.size(); ++i) {
        p.rgb[i] = static_cast<std::uint8_t>((i * 7919) & 0xFF);
    }
    return p;
}

// A stand-in for the neural encoder: deterministic, unit RMS, mode C
// length. The modem re-normalizes anyway; this just has to be plausible.
std::vector<double> stub_latents() {
    const int n = config::MODES[config::N_MODES - 1].n_latents;
    std::vector<double> out(static_cast<std::size_t>(n));
    std::uint64_t s = 99;
    for (double& v : out) {
        s += 0x9E3779B97F4A7C15ULL;
        std::uint64_t z = s;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        z = z ^ (z >> 31);
        v = static_cast<double>(z >> 11) / 9007199254740992.0 - 0.5;
    }
    double ms = 0.0;
    for (double v : out) ms += v * v;
    const double rms = std::sqrt(ms / static_cast<double>(out.size()));
    for (double& v : out) v /= rms;
    return out;
}

tx::Encoder good_encoder() {
    return [](const images::ImageArray&) { return stub_latents(); };
}

// Records every key/unkey in order, so a test can say not just "PTT is
// down" but "it went up once and came down once".
struct PttLog {
    std::mutex m;
    std::vector<bool> calls;
    bool fail_on_off = false;
    bool fail_on_on = false;

    tx::Ptt fn() {
        return [this](bool on) {
            {
                std::lock_guard<std::mutex> lock(m);
                calls.push_back(on);
            }
            if (on && fail_on_on) throw std::runtime_error("rig said no");
            if (!on && fail_on_off) throw std::runtime_error("socket is gone");
        };
    }

    std::vector<bool> snapshot() {
        std::lock_guard<std::mutex> lock(m);
        return calls;
    }

    bool ended_unkeyed() {
        const std::vector<bool> c = snapshot();
        return !c.empty() && c.back() == false;
    }
};

tx::TxConfig fast_config() {
    tx::TxConfig c;
    c.mode = "A";  // the shortest, ~32 s of audio to synthesize
    c.callsign = "KC2G";
    c.ptt_lead_s = 0.01;
    c.ptt_tail_s = 0.01;
    return c;
}

// --- the sequence -----------------------------------------------------------

void test_a_successful_transmission() {
    PttLog ptt;
    std::vector<double> played;
    std::string played_device;
    int played_rate = 0;

    tx::Player player = [&](const std::string& device, std::span<const double> wave,
                            int rate, const std::function<void(double)>& progress,
                            const std::function<bool()>&,
                            const std::function<void(const std::string&)>&) {
        played.assign(wave.begin(), wave.end());
        played_device = device;
        played_rate = rate;
        progress(0.5);
        return true;
    };

    std::vector<tx::TxPhase> phases;
    tx::TxEngine engine(ptt.fn(), player, good_encoder(),
                        [&](const tx::TxState& s) { phases.push_back(s.phase); });

    tx::TxConfig config = fast_config();
    config.device = "hw:1,0";
    const bool ok = engine.transmit(test_picture(), config);

    check::is_true(ok, "tx/ok: transmit reports success");
    check::is_true(engine.state().phase == tx::TxPhase::Done, "tx/ok: ends in Done");
    check::equal(engine.state().progress, 1.0, "tx/ok: progress finishes at 1");
    check::equal(played_rate, config::FS, "tx/ok: played at the modem's rate");
    check::equal(played_device, std::string("hw:1,0"), "tx/ok: the configured device");

    const std::vector<bool> keys = ptt.snapshot();
    check::equal(keys.size(), std::size_t{2}, "tx/ok: keyed once, unkeyed once");
    check::is_true(!keys.empty() && keys.front() == true, "tx/ok: keyed first");
    check::is_true(ptt.ended_unkeyed(), "tx/ok: PTT is down at the end");

    // The phases a UI would see, in order.
    const std::vector<tx::TxPhase> want = {
        tx::TxPhase::Encoding, tx::TxPhase::Modulating, tx::TxPhase::Keying,
        tx::TxPhase::Sending,  tx::TxPhase::Sending,  // the progress callback
        tx::TxPhase::Unkeying, tx::TxPhase::Done};
    check::is_true(phases == want, "tx/ok: the phase sequence a UI observes");

    // "Sent" has to mean a decodable transmission, not an empty buffer.
    check::is_true(!played.empty(), "tx/ok: something was played");
    double peak = 0.0;
    for (double v : played) peak = std::max(peak, std::abs(v));
    check::close(std::vector<double>{peak}, std::vector<double>{config.level}, 1e-12,
                 "tx/ok: scaled to the configured peak");

    const modem::Modem m;
    const modem::DemodResult r = m.demodulate(played);
    check::is_true(std::string(r.mode.name) == "A", "tx/ok: the mode goes out right");
    check::equal(r.callsign, std::string("KC2G"), "tx/ok: the beacon carries the callsign");
    check::equal(r.frames_received, r.mode.n_frames,
                 "tx/ok: every frame is present in the waveform");
}

void test_no_rig_control_still_transmits() {
    // PTT null: audio only, VOX, or a dummy load.
    tx::Player player = [](const std::string&, std::span<const double>, int,
                           const std::function<void(double)>&,
                           const std::function<bool()>&,
                           const std::function<void(const std::string&)>&) { return true; };
    tx::TxEngine engine(nullptr, player, good_encoder());
    check::is_true(engine.transmit(test_picture(), fast_config()),
                   "tx/norig: transmits with no rig attached");
    check::is_true(engine.state().phase == tx::TxPhase::Done, "tx/norig: Done");
}

// --- the guarantee ----------------------------------------------------------

void test_a_throwing_player_still_unkeys() {
    // The case the scope guard is for.
    PttLog ptt;
    std::string error;
    tx::Player player = [](const std::string&, std::span<const double>, int,
                           const std::function<void(double)>&,
                           const std::function<bool()>&,
                           const std::function<void(const std::string&)>&) -> bool {
        throw std::runtime_error("the sound device vanished");
    };
    tx::TxEngine engine(ptt.fn(), player, good_encoder(), {},
                        [&](const std::string& m) { error = m; });

    check::is_true(!engine.transmit(test_picture(), fast_config()),
                   "tx/throw: reports failure");
    check::is_true(engine.state().phase == tx::TxPhase::Failed, "tx/throw: phase Failed");
    check::is_true(ptt.ended_unkeyed(), "tx/throw: PTT still came back down");
    check::equal(ptt.snapshot().size(), std::size_t{2}, "tx/throw: exactly one up, one down");
    check::is_true(error.find("the sound device vanished") != std::string::npos,
                   "tx/throw: the operator is told why");
}

void test_cancelling_during_playback_unkeys() {
    PttLog ptt;
    tx::TxEngine* engine_ptr = nullptr;
    tx::Player player = [&](const std::string&, std::span<const double>, int,
                            const std::function<void(double)>& progress,
                            const std::function<bool()>& should_stop,
                            const std::function<void(const std::string&)>&) {
        progress(0.25);
        engine_ptr->cancel();
        // A real player notices between buffers and returns short.
        return !should_stop();
    };
    tx::TxEngine engine(ptt.fn(), player, good_encoder());
    engine_ptr = &engine;

    check::is_true(!engine.transmit(test_picture(), fast_config()),
                   "tx/cancel: a cancelled transmission is not a success");
    check::is_true(engine.state().phase == tx::TxPhase::Cancelled,
                   "tx/cancel: phase Cancelled, not Failed");
    check::is_true(ptt.ended_unkeyed(), "tx/cancel: PTT came back down");
    check::equal(ptt.snapshot().size(), std::size_t{2}, "tx/cancel: one up, one down");
}

void test_cancelling_during_encode_never_keys() {
    // Cancelled before the rig is ever touched: nothing should key at
    // all, which is a stronger claim than "it unkeyed".
    PttLog ptt;
    tx::TxEngine* engine_ptr = nullptr;
    tx::Encoder encoder = [&](const images::ImageArray&) {
        engine_ptr->cancel();
        return stub_latents();
    };
    tx::Player player = [](const std::string&, std::span<const double>, int,
                           const std::function<void(double)>&,
                           const std::function<bool()>&,
                           const std::function<void(const std::string&)>&) { return true; };
    tx::TxEngine engine(ptt.fn(), player, encoder);
    engine_ptr = &engine;

    check::is_true(!engine.transmit(test_picture(), fast_config()),
                   "tx/precancel: not a success");
    check::is_true(engine.state().phase == tx::TxPhase::Cancelled,
                   "tx/precancel: phase Cancelled");
    check::equal(ptt.snapshot().size(), std::size_t{0},
                 "tx/precancel: the rig was never keyed");
}

void test_a_failing_encoder_never_keys() {
    PttLog ptt;
    std::string error;
    tx::Encoder encoder = [](const images::ImageArray&) -> std::vector<double> {
        throw std::runtime_error("no model artifact");
    };
    tx::Player player = [](const std::string&, std::span<const double>, int,
                           const std::function<void(double)>&,
                           const std::function<bool()>&,
                           const std::function<void(const std::string&)>&) { return true; };
    tx::TxEngine engine(ptt.fn(), player, encoder, {},
                        [&](const std::string& m) { error = m; });

    check::is_true(!engine.transmit(test_picture(), fast_config()), "tx/encfail: fails");
    check::is_true(engine.state().phase == tx::TxPhase::Failed, "tx/encfail: phase Failed");
    check::equal(ptt.snapshot().size(), std::size_t{0},
                 "tx/encfail: a preparation failure never keys the rig");
    check::is_true(error.find("no model artifact") != std::string::npos,
                   "tx/encfail: the reason is reported");
}

void test_an_unknown_mode_is_refused_before_keying() {
    PttLog ptt;
    std::string error;
    tx::Player player = [](const std::string&, std::span<const double>, int,
                           const std::function<void(double)>&,
                           const std::function<bool()>&,
                           const std::function<void(const std::string&)>&) { return true; };
    tx::TxEngine engine(ptt.fn(), player, good_encoder(), {},
                        [&](const std::string& m) { error = m; });
    tx::TxConfig config = fast_config();
    config.mode = "Q";

    check::is_true(!engine.transmit(test_picture(), config), "tx/mode: refused");
    check::equal(ptt.snapshot().size(), std::size_t{0}, "tx/mode: nothing keyed");
    check::is_true(error.find("Q") != std::string::npos,
                   "tx/mode: the bad mode is named, not silently defaulted");
}

void test_a_failed_unkey_is_reported_as_an_emergency() {
    // Failing to key is a normal problem. Failing to *unkey* means the
    // rig may still be transmitting, and the operator has to be told to
    // go and do it by hand.
    PttLog ptt;
    ptt.fail_on_off = true;
    std::string error;
    tx::Player player = [](const std::string&, std::span<const double>, int,
                           const std::function<void(double)>&,
                           const std::function<bool()>&,
                           const std::function<void(const std::string&)>&) { return true; };
    tx::TxEngine engine(ptt.fn(), player, good_encoder(), {},
                        [&](const std::string& m) { error = m; });

    engine.transmit(test_picture(), fast_config());
    check::is_true(error.find("PTT OFF FAILED") != std::string::npos,
                   "tx/unkeyfail: reported loudly");
    check::is_true(error.find("Unkey it manually") != std::string::npos,
                   "tx/unkeyfail: the operator is told what to do about it");
}

// --- the watchdog -----------------------------------------------------------

void test_the_watchdog_unkeys_a_wedged_transmission() {
    // Tested directly rather than through `transmit`. The watchdog's
    // entire reason to exist is the case where the transmit path never
    // returns, so reaching it through a transmit that returns normally
    // would be testing the wrong thing -- and its timeout is
    // lead + duration + tail + margin, which for the shortest mode is
    // over half a minute.
    std::atomic<bool> unkeyed{false};
    std::atomic<bool> fired{false};
    {
        tx::PttWatchdog watchdog([&](bool on) { if (!on) unkeyed = true; }, 0.05,
                                 [&] { fired = true; });
        watchdog.start();
        // Deliberately no cancel: this is the wedged transmission.
        for (int i = 0; i < 400 && !fired; ++i) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }
    check::is_true(unkeyed.load(), "tx/watchdog: dropped PTT on its own");
    check::is_true(fired.load(), "tx/watchdog: reported that it fired");
}

void test_the_watchdog_stands_down_when_cancelled() {
    std::atomic<int> calls{0};
    std::atomic<bool> fired{false};
    {
        // A long timeout, cancelled immediately. If `cancel` did not
        // interrupt the wait, the destructor would block here for an
        // hour -- which the suite reports as a hang, and which is the
        // reason the event is a condition variable rather than a flag.
        tx::PttWatchdog watchdog([&](bool) { ++calls; }, 3600.0, [&] { fired = true; });
        watchdog.start();
        watchdog.cancel();
    }
    check::equal(calls.load(), 0, "tx/watchdog: a stood-down watchdog touches nothing");
    check::is_true(!fired.load(), "tx/watchdog: and does not report firing");
}

void test_a_normal_transmission_does_not_trip_the_watchdog() {
    // The complement of the two above: with the margin at its default
    // the watchdog must stay out of the way, and in particular must not
    // add a third PTT call after the sequence has already unkeyed.
    PttLog ptt;
    std::string error;
    tx::Player player = [](const std::string&, std::span<const double>, int,
                           const std::function<void(double)>&,
                           const std::function<bool()>&,
                           const std::function<void(const std::string&)>&) { return true; };
    tx::TxEngine engine(ptt.fn(), player, good_encoder(), {},
                        [&](const std::string& m) { error = m; });

    check::is_true(engine.transmit(test_picture(), fast_config()), "tx/nowd: succeeded");
    check::equal(ptt.snapshot().size(), std::size_t{2}, "tx/nowd: exactly two PTT calls");
    check::is_true(error.empty(), "tx/nowd: no watchdog complaint on a clean run");
}

// --- conditioning -----------------------------------------------------------

void test_condition_for_output_is_a_plain_peak_scale() {
    const std::vector<double> x = {0.0, 0.5, -0.25, 0.1};
    const std::vector<double> got = tx::condition_for_output(x, 0.9);
    check::close(got, std::vector<double>{0.0, 0.9, -0.45, 0.18}, 1e-12,
                 "tx/level: every sample scaled by one factor");

    // Silence must not become a division by zero.
    const std::vector<double> zeros(8, 0.0);
    check::is_true(tx::condition_for_output(zeros, 0.9) == zeros,
                   "tx/level: silence stays silence");

    // A quiet waveform is scaled *up* to the target -- this is a peak
    // set, not a limiter. The clipping that sets PAPR already happened
    // in the modulator, and repeating it here would splatter.
    const std::vector<double> quiet = {0.01, -0.02};
    const std::vector<double> loud = tx::condition_for_output(quiet, 0.9);
    check::close(std::vector<double>{loud[1]}, std::vector<double>{-0.9}, 1e-12,
                 "tx/level: quiet input is brought up");
}

}  // namespace

int main() {
    try {
        test_condition_for_output_is_a_plain_peak_scale();
        test_the_watchdog_stands_down_when_cancelled();
        test_the_watchdog_unkeys_a_wedged_transmission();
        test_a_failing_encoder_never_keys();
        test_an_unknown_mode_is_refused_before_keying();
        test_cancelling_during_encode_never_keys();
        test_a_successful_transmission();
        test_no_rig_control_still_transmits();
        test_a_throwing_player_still_unkeys();
        test_cancelling_during_playback_unkeys();
        test_a_failed_unkey_is_reported_as_an_emergency();
        test_a_normal_transmission_does_not_trip_the_watchdog();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FATAL: %s\n", e.what());
        return 1;
    }
    return check::report("tx engine");
}

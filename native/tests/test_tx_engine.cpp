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

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <mutex>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "check.hpp"
#include "config.hpp"
#include "dsp/dsp.hpp"
#include "dsp/leader.hpp"
#include "dsp/morse.hpp"
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

// A player that just captures the waveform it was handed.
tx::Player capturing_player(std::vector<double>& out) {
    return [&out](const std::string&, std::span<const double> wave, int,
                 const std::function<void(double)>&, const std::function<bool()>&,
                 const std::function<void(const std::string&)>&) {
        out.assign(wave.begin(), wave.end());
        return true;
    };
}

// The preamble detector's own metric, reimplemented here on purpose.
//
// `sync::acquire` does not expose it, and the leader's whole safety
// argument is a statement about this number -- so the test states it
// numerically rather than inferring it from a decode. A decode-based
// test cannot do the job: on a clean signal the true preamble reads
// exactly 1.000 and wins the argmax against *any* leader, so the
// dangerous designs pass. See dsp/leader.hpp for the measurements.
//
// Sliding lag-M autocorrelation over PREAMBLE_CORR_WINDOW, normalized by
// the window's own energy, as sstvae/modem/sync.py's _autocorr_metric.
// Returns the highest value over windows *starting* before `limit`.
//
// **Measured on the leader in front of a real transmission, not on the
// leader alone**, and that is not a nicety. The energy floor and the
// filter state both depend on what surrounds the leader, and measuring
// it in isolation understates it by a lot: the same 500 ms stretched
// sweep reads 0.104 standalone and 0.44 in the position it will actually
// occupy. The number that matters is the one a receiver computes on the
// signal it receives.
double peak_preamble_metric(std::span<const double> x, std::size_t limit) {
    const std::vector<dsp::cdouble> z = dsp::sync_lowpass(dsp::to_baseband(x));
    const std::size_t m = config::M;
    const std::size_t w = config::PREAMBLE_CORR_WINDOW;
    if (z.size() < m + w + 1) return 0.0;

    double mean_power = 0.0;
    for (const dsp::cdouble& v : z) mean_power += std::norm(v);
    mean_power /= static_cast<double>(z.size());
    const double floor_e = 1e-3 * static_cast<double>(w) * mean_power;

    dsp::cdouble acc{};
    double e1 = 0.0, e2 = 0.0;
    double best = 0.0;
    const std::size_t n = z.size() - m;
    for (std::size_t i = 0; i < n; ++i) {
        acc += z[i + m] * std::conj(z[i]);
        e1 += std::norm(z[i]);
        e2 += std::norm(z[i + m]);
        if (i >= w) {
            acc -= z[i - w + m] * std::conj(z[i - w]);
            e1 -= std::norm(z[i - w]);
            e2 -= std::norm(z[i - w + m]);
        }
        if (i + 1 < w) continue;
        const std::size_t start = i + 1 - w;
        if (start >= limit) break;
        const double energy =
            std::sqrt(std::max(e1, floor_e) * std::max(e2, floor_e)) + 1e-12;
        best = std::max(best, std::abs(acc) / energy);
    }
    return best;
}

// **The leader must not look like a preamble, at any duration.**
//
// This is the test that makes `VOX_SWEEP_S` load-bearing rather than
// decorative. Replace the repeated fixed-rate sweep with one sweep
// stretched over the requested duration -- the obvious implementation,
// and the one a reader would call a simplification -- and the leader's
// metric climbs with duration: 0.443 at the default 500 ms, already over
// the 0.42 threshold, and 0.969 at 10 s, which is a tone in all but
// name. Nothing else in the suite notices, because a stretched leader
// still decodes fine on a clean signal.
void test_the_vox_leader_never_looks_like_a_preamble() {
    // One real transmission, reused: the leader has to be measured in
    // the position it will occupy, and modulating five times is the
    // slowest thing this file could do for no extra information.
    std::vector<double> wave;
    tx::TxEngine engine(nullptr, capturing_player(wave), good_encoder());
    check::is_true(engine.transmit(test_picture(), fast_config()),
                   "tx/vox: baseline transmission for the metric");

    const auto gap_n =
        static_cast<std::size_t>(std::lround(dsp::VOX_LEAD_GAP_S * config::FS));

    for (const double seconds : {0.3, 0.5, 1.0, 3.0, 10.0}) {
        const std::vector<double> lead =
            dsp::vox_leader(seconds, config::FS, 1.0);
        check::equal(lead.size(),
                     static_cast<std::size_t>(std::lround(seconds * config::FS)),
                     "tx/vox: the leader is the requested length");

        std::vector<double> signal = lead;
        signal.insert(signal.end(), gap_n, 0.0);
        signal.insert(signal.end(), wave.begin(), wave.end());
        const double metric = peak_preamble_metric(signal, lead.size());

        char what[128];
        std::snprintf(what, sizeof what,
                      "tx/vox: %.1f s leader stays under the 0.42 threshold (%.3f)",
                      seconds, metric);
        check::is_true(metric < config::PREAMBLE_THRESHOLD, what);
        // Not merely under it: under the 0.358 peak that 3000 s of AWGN
        // produces through this detector (CLAUDE.md, PREAMBLE_REPEATS),
        // so the leader is no likelier to cause a false lock than the
        // silence it replaces.
        std::snprintf(what, sizeof what,
                      "tx/vox: %.1f s leader stays under the AWGN peak (%.3f)",
                      seconds, metric);
        check::is_true(metric < 0.358, what);
    }
}

// And the transmission behind it still decodes -- lock on the real
// preamble, every frame, callsign intact.
void test_a_transmission_behind_a_vox_leader_still_decodes() {
    std::vector<double> played;
    tx::TxEngine engine(nullptr, capturing_player(played), good_encoder());
    tx::TxConfig config = fast_config();
    config.vox_lead_s = 0.5;
    check::is_true(engine.transmit(test_picture(), config),
                   "tx/vox: transmits with the leader on");

    const modem::Modem m;
    const modem::DemodResult r = m.demodulate(played);
    check::is_true(std::string(r.mode.name) == "A", "tx/vox: the mode still decodes");
    check::equal(r.callsign, std::string("KC2G"), "tx/vox: the callsign survives");
    check::equal(r.frames_received, r.mode.n_frames,
                 "tx/vox: every frame is still there");
}

// Prepended before conditioning, at the wave's own peak -- so turning
// the leader on does not change the level the operator set. Same
// property the CW ID has, and the same reason.
void test_the_vox_leader_costs_only_its_own_airtime() {
    std::vector<double> plain;
    tx::TxEngine baseline(nullptr, capturing_player(plain), good_encoder());
    tx::TxConfig config = fast_config();
    // Off unless asked for: whenever PTT is under our control the lead
    // delay already covers the relay, and this is airtime.
    check::equal(config.vox_lead_s, 0.0, "tx/vox: off by default");
    check::is_true(baseline.transmit(test_picture(), config),
                   "tx/vox: baseline transmits");

    std::vector<double> with_lead;
    tx::TxEngine engine(nullptr, capturing_player(with_lead), good_encoder());
    config.vox_lead_s = 0.5;
    check::is_true(engine.transmit(test_picture(), config),
                   "tx/vox: transmits with the leader on");

    const auto lead_n = static_cast<std::size_t>(std::lround(0.5 * config::FS));
    const auto gap_n =
        static_cast<std::size_t>(std::lround(dsp::VOX_LEAD_GAP_S * config::FS));
    check::equal(with_lead.size(), plain.size() + lead_n + gap_n,
                 "tx/vox: exactly one leader plus one gap was prepended");

    bool gap_is_silent = true;
    for (std::size_t i = lead_n; i < lead_n + gap_n; ++i) {
        if (with_lead[i] != 0.0) gap_is_silent = false;
    }
    check::is_true(gap_is_silent, "tx/vox: the gap before the preamble is silence");

    double plain_peak = 0.0, lead_peak = 0.0;
    for (double v : plain) plain_peak = std::max(plain_peak, std::abs(v));
    for (double v : with_lead) lead_peak = std::max(lead_peak, std::abs(v));
    check::close(std::vector<double>{lead_peak}, std::vector<double>{plain_peak}, 1e-12,
                 "tx/vox: the transmit level is unchanged by the leader");
}

void test_cw_id_appends_after_the_transmission() {
    std::vector<double> plain;
    tx::TxEngine baseline(nullptr, capturing_player(plain), good_encoder());
    tx::TxConfig config = fast_config();
    config.cw_id = false;
    check::is_true(baseline.transmit(test_picture(), config),
                   "tx/cwid: baseline transmits");

    std::vector<double> with_id;
    tx::TxEngine engine(nullptr, capturing_player(with_id), good_encoder());
    config.cw_id = true;
    check::is_true(engine.transmit(test_picture(), config),
                   "tx/cwid: transmits with the ID on");

    // The default message is the whole point of the feature (issue #14):
    // it both identifies the station and advertises SSTVAE, not just the
    // bare callsign.
    check::equal(config.cw_message, std::string("SSTVAE DE {callsign}"),
                "tx/cwid: the default CW message advertises the mode and software");
    const std::vector<double> id_tone = dsp::generate_morse(
        "SSTVAE DE KC2G", config::FS, tx::CW_ID_WPM, tx::CW_ID_TONE_HZ, 1.0);
    check::is_true(!id_tone.empty(), "tx/cwid: SSTVAE DE KC2G produces a tone");
    const std::size_t gap_n =
        static_cast<std::size_t>(std::lround(tx::CW_ID_GAP_S * config::FS));

    check::equal(with_id.size(), plain.size() + gap_n + id_tone.size(),
                "tx/cwid: appended exactly one gap plus one ID's worth of samples");

    bool gap_is_silent = true;
    for (std::size_t i = plain.size(); i < plain.size() + gap_n; ++i) {
        if (with_id[i] != 0.0) gap_is_silent = false;
    }
    check::is_true(gap_is_silent, "tx/cwid: the 500 ms gap is silence");

    double tail_peak = 0.0;
    for (std::size_t i = plain.size() + gap_n; i < with_id.size(); ++i) {
        tail_peak = std::max(tail_peak, std::abs(with_id[i]));
    }
    check::is_true(tail_peak > 0.0, "tx/cwid: the ID tone actually plays");

    // Appended before conditioning: the overall peak (and so the
    // transmit level the operator set) is unchanged by adding the ID.
    double plain_peak = 0.0, with_id_peak = 0.0;
    for (double v : plain) plain_peak = std::max(plain_peak, std::abs(v));
    for (double v : with_id) with_id_peak = std::max(with_id_peak, std::abs(v));
    check::close(std::vector<double>{with_id_peak}, std::vector<double>{plain_peak}, 1e-12,
                "tx/cwid: adding the ID does not change the transmit peak");

    // The picture itself still decodes exactly as it would have.
    //
    // A steady CW tone is exactly periodic at whatever lag the preamble
    // correlator uses whenever its frequency is a multiple of the
    // carrier spacing (1000 Hz here, 20x50 Hz) -- so an unrestricted
    // search can find a higher-scoring "preamble" in the ID tone than
    // in the real one. That is the same class of event the sync layer
    // already tolerates (a spurious lock that fails the header and
    // never completes, see `test_rx_engine.cpp`'s
    // "noise never finishes a reception" case), not specific to this
    // feature -- but a real receiver already knows roughly where its
    // own transmission starts, so bound the search here rather than
    // relying on that tolerance to prove the picture decodes.
    const modem::Modem m;
    const modem::DemodResult r =
        m.demodulate(with_id, std::make_pair(0.0, static_cast<double>(plain.size()) /
                                                       config::FS));
    check::equal(r.callsign, std::string("KC2G"), "tx/cwid: beacon still decodes");
    check::equal(r.frames_received, r.mode.n_frames,
                "tx/cwid: every SSTVAE frame is still present");
}

void test_cw_id_does_nothing_with_no_callsign() {
    std::vector<double> played;
    tx::TxEngine engine(nullptr, capturing_player(played), good_encoder());
    tx::TxConfig config = fast_config();
    config.callsign = "";
    config.cw_id = true;
    check::is_true(engine.transmit(test_picture(), config),
                   "tx/cwid-none: still transmits");

    std::vector<double> plain;
    tx::TxEngine baseline(nullptr, capturing_player(plain), good_encoder());
    config.cw_id = false;
    check::is_true(baseline.transmit(test_picture(), config),
                   "tx/cwid-none: baseline transmits");

    check::equal(played.size(), plain.size(),
                "tx/cwid-none: no callsign means nothing is appended");
}

// The predicate the UI blocks Send on and the engine skips the ID on.
// Exactly one combination is bad; the three ways out all have to work,
// because the UI offers all three.
void test_cw_id_problem_names_only_the_broken_combination() {
    check::is_true(tx::cw_id_problem(true, "SSTVAE DE {callsign}", "").empty() == false,
                   "tx/cwid-problem: placeholder with no callsign is a problem");
    check::is_true(tx::cw_id_problem(true, "SSTVAE DE {callsign}", "KC2G").empty(),
                   "tx/cwid-problem: way out 1 -- set a callsign");
    check::is_true(tx::cw_id_problem(true, "SSTVAE DE KC2G", "").empty(),
                   "tx/cwid-problem: way out 2 -- write the call into the message");
    check::is_true(tx::cw_id_problem(false, "SSTVAE DE {callsign}", "").empty(),
                   "tx/cwid-problem: way out 3 -- turn CW ID off");
}

// The behaviour change that makes way out 2 real: before this, *any*
// empty callsign dropped the ID, so rewriting the template did nothing
// and the UI would have been offering an escape the engine ignored.
void test_cw_literal_message_is_sent_with_no_callsign() {
    std::vector<double> custom;
    tx::TxEngine engine(nullptr, capturing_player(custom), good_encoder());
    tx::TxConfig config = fast_config();
    config.callsign = "";
    config.cw_id = true;
    config.cw_message = "SSTVAE DE KC2G";
    check::is_true(engine.transmit(test_picture(), config),
                   "tx/cwid-literal: transmits");

    std::vector<double> plain;
    tx::TxEngine baseline(nullptr, capturing_player(plain), good_encoder());
    config.cw_id = false;
    check::is_true(baseline.transmit(test_picture(), config),
                   "tx/cwid-literal: baseline transmits");

    const std::vector<double> id_tone = dsp::generate_morse(
        "SSTVAE DE KC2G", config::FS, tx::CW_ID_WPM, tx::CW_ID_TONE_HZ, 1.0);
    const std::size_t gap_n =
        static_cast<std::size_t>(std::lround(tx::CW_ID_GAP_S * config::FS));
    check::equal(custom.size(), plain.size() + gap_n + id_tone.size(),
                "tx/cwid-literal: a message with no placeholder is keyed "
                "even with no callsign set");
}

void test_cw_message_is_customizable() {
    // issue #14: the operator can replace the default message. Every
    // `{callsign}` in it becomes the configured callsign; the rest goes
    // out verbatim.
    std::vector<double> custom;
    tx::TxEngine engine(nullptr, capturing_player(custom), good_encoder());
    tx::TxConfig config = fast_config();
    config.cw_id = true;
    config.cw_message = "DE {callsign} {callsign} SSTV TEST";
    check::is_true(engine.transmit(test_picture(), config),
                   "tx/cwmsg: transmits with a custom message");

    std::vector<double> plain;
    tx::TxEngine baseline(nullptr, capturing_player(plain), good_encoder());
    config.cw_id = false;
    check::is_true(baseline.transmit(test_picture(), config),
                   "tx/cwmsg: baseline transmits");

    const std::vector<double> id_tone = dsp::generate_morse(
        "DE KC2G KC2G SSTV TEST", config::FS, tx::CW_ID_WPM, tx::CW_ID_TONE_HZ, 1.0);
    check::is_true(!id_tone.empty(), "tx/cwmsg: the substituted message produces a tone");
    const std::size_t gap_n =
        static_cast<std::size_t>(std::lround(tx::CW_ID_GAP_S * config::FS));
    check::equal(custom.size(), plain.size() + gap_n + id_tone.size(),
                "tx/cwmsg: every {callsign} placeholder was substituted, not just the first");
}

void test_cw_message_with_no_placeholder_is_sent_as_is() {
    // A message with no `{callsign}` in it is still sent verbatim --
    // useful for an operator who wants to spell the callsign into the
    // message text themselves, or add nothing beyond a fixed phrase.
    std::vector<double> custom;
    tx::TxEngine engine(nullptr, capturing_player(custom), good_encoder());
    tx::TxConfig config = fast_config();
    config.cw_id = true;
    config.cw_message = "SSTVAE";
    check::is_true(engine.transmit(test_picture(), config),
                   "tx/cwmsg-fixed: transmits");

    std::vector<double> plain;
    tx::TxEngine baseline(nullptr, capturing_player(plain), good_encoder());
    config.cw_id = false;
    check::is_true(baseline.transmit(test_picture(), config),
                   "tx/cwmsg-fixed: baseline transmits");

    const std::vector<double> id_tone = dsp::generate_morse(
        "SSTVAE", config::FS, tx::CW_ID_WPM, tx::CW_ID_TONE_HZ, 1.0);
    const std::size_t gap_n =
        static_cast<std::size_t>(std::lround(tx::CW_ID_GAP_S * config::FS));
    check::equal(custom.size(), plain.size() + gap_n + id_tone.size(),
                "tx/cwmsg-fixed: the fixed message is sent with no substitution");
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
        test_the_vox_leader_never_looks_like_a_preamble();
        test_a_transmission_behind_a_vox_leader_still_decodes();
        test_the_vox_leader_costs_only_its_own_airtime();
        test_cw_id_appends_after_the_transmission();
        test_cw_id_does_nothing_with_no_callsign();
        test_cw_id_problem_names_only_the_broken_combination();
        test_cw_literal_message_is_sent_with_no_callsign();
        test_cw_message_is_customizable();
        test_cw_message_with_no_placeholder_is_sent_as_is();
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

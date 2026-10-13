// native/core/qrss/receiver_slot (the slot half of sstvae/qrss/receiver.py),
// wavread and rx_api.hpp, without Python.
//
// - The rules (ghost_of, aliases, ghosts, a3_lines, owned,
//   suspect_neighbour, suspect_beside) on records made by hand, each
//   decision on both sides of its boundary, the strongest-first order and
//   its ties (a stable sort).
// - The orchestration with every stage a stub that logs its call (the
//   SlotSeams), the clock a fake one: which candidates reach gate V and in
//   which order, with which noise exclusions (sorted, deduplicated), the
//   budget's accounting (an owned candidate is not counted, a merged one
//   is; max_candidates and budget_s break where the reference does, on the
//   reference's clock readings), the CE accept rechecked on the residual
//   ratio, the lead-ins and their symbol, the joint receiver's centres and
//   a waveform not listened for, and receive_slot's ghost / SUSPECT /
//   merge rules after tracking.
// - The receive end to end through rx_api.hpp: a CE TINY and an L l-tiny
//   pass made by the transmit API (api.hpp), summed with noise, received
//   jointly, CE alone, and a SHORT pass with its header; four threads at
//   once; and the two rules given per call while the environment says the
//   opposite (so they are not read from it).
// - ce_resid_ratio on a real gate-V track, against the same sum in long
//   double, and its 1.0 when no known symbol is usable.
// - wavread over the formats it takes, against the values wavio.read_wav
//   gives (integers over their dtype's maximum, channels averaged), and its
//   refusals.
//
// tests/test_qrss_native_receiver_slot.py compares all of it against the
// reference.

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <numbers>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "check.hpp"
#include "qrss/api.hpp"
#include "qrss/config.hpp"
#include "qrss/frame.hpp"
#include "qrss/frontend.hpp"
#include "qrss/picture.hpp"
#include "qrss/receiver.hpp"
#include "qrss/receiver_slot.hpp"
#include "qrss/rules.hpp"
#include "qrss/rx_api.hpp"
#include "qrss/rx_config.hpp"
#include "qrss/track.hpp"
#include "qrss/track_timing.hpp"
#include "qrss/types.hpp"
#include "qrss/wavread.hpp"

using namespace sstvae;
namespace rc = sstvae::qrss::receiver;
namespace fr = sstvae::qrss::frame;
namespace fe = sstvae::qrss::frontend;
namespace tk = sstvae::qrss::track;
namespace aq = sstvae::qrss::acquire;
namespace types = sstvae::qrss::types;
namespace cfg = sstvae::qrss::config;
namespace api = sstvae::qrss::api;
namespace pc = sstvae::qrss::picture;
namespace wav = sstvae::qrss::wav;
namespace rules = sstvae::qrss::rules;

namespace {

constexpr std::int64_t Q = 1954752;   // 2025-10-01T00:00Z
constexpr std::int64_t CE = cfg::WAVEFORM_CE;
constexpr std::int64_t L = cfg::WAVEFORM_L;
const double NaN = std::numeric_limits<double>::quiet_NaN();

template <typename E, typename F>
void throws(F f, const std::string& what) {
    try {
        f();
    } catch (const E&) {
        check::is_true(true, what);
        return;
    } catch (const std::exception& e) {
        check::fail(what, std::string("threw another type: ") + e.what());
        return;
    }
    check::fail(what, "did not throw");
}

std::string num(double x) {
    char b[64];
    std::snprintf(b, sizeof b, "%.9g", x);
    return b;
}

std::string nums(std::span<const double> v) {
    std::string s = "[";
    for (std::size_t i = 0; i < v.size(); ++i) s += (i ? " " : "") + num(v[i]);
    return s + "]";
}

void eqv(const std::vector<double>& got, const std::vector<double>& want, const std::string& what) {
    check::is_true(got == want, what + " (got " + nums(got) + ", want " + nums(want) + ")");
}

types::Detection det(double f, const char* method = "preamble", std::int64_t wf = CE, double a3 = NaN,
                     double z = NaN, std::optional<double> tau0 = 0.0) {
    types::Detection d;
    d.f_hz = f;
    d.method = method;
    d.waveform = wf;
    d.a3_z = a3;
    d.z_ref = z;
    if (tau0) {
        types::Timing t;
        t.tau0 = *tau0;
        t.z = z;
        d.timing = t;
    }
    return d;
}

types::PassResult pass(double f, double z, bool suspect, double tau0 = 0.0, std::int64_t wf = CE) {
    types::PassResult p;
    p.f_hz = f;
    p.waveform = wf;
    p.timing.tau0 = tau0;
    p.report.z_ref = z;
    p.report.suspect = suspect;
    p.frame = wf == L ? "l-tiny" : "tiny";
    return p;
}

std::vector<double> fs_of(const std::vector<types::Detection>& v) {
    std::vector<double> out;
    for (const auto& d : v) out.push_back(d.f_hz);
    return out;
}

std::vector<double> fs_of(const std::vector<types::PassResult>& v) {
    std::vector<double> out;
    for (const auto& p : v) out.push_back(p.f_hz);
    return out;
}

// --- the rules --------------------------------------------------------------------------------

void test_rules() {
    const double T = cfg::T_SYM;
    const double TL = fr::get_l("l-tiny").t_sym();
    // ghost_of: GHOST_HZ and GHOST_T symbols, strictly inside
    const double win = rc::GHOST_T * T * cfg::CH_FS;   // 22.734375 CH samples
    check::equal(win, 22.734375, "the ghost window");
    check::is_true(rc::ghost_of(1510.0, 10.0, 1500.0, 0.0), "a ghost: 10 Hz, 10 samples");
    check::is_true(!rc::ghost_of(1520.0, 0.0, 1500.0, 0.0), "GHOST_HZ away: no ghost");
    check::is_true(rc::ghost_of(1480.01, 0.0, 1500.0, 0.0), "just inside GHOST_HZ below");
    check::is_true(rc::ghost_of(1500.0, 22.7, 1500.0, 0.0) && !rc::ghost_of(1500.0, 22.75, 1500.0, 0.0),
                   "the timing window, CE");
    check::is_true(rc::ghost_of(1500.0, 45.4, 1500.0, 0.0, TL) && !rc::ghost_of(1500.0, 45.5, 1500.0, 0.0, TL),
                   "the timing window, L (in L symbols)");

    // _aliases: f +- k/T within ALIAS_HZ, timing within k + 1/2 symbols,
    // the stronger over ALIAS_RATIO times the weaker; strongest first
    const types::Detection A = det(1500.0, "preamble", CE, NaN, 100.0, 0.0);
    const auto keep = [&](std::vector<types::Detection> v) { return fs_of(rc::aliases(v)); };
    eqv(keep({A, det(1532.99, "tbd", CE, NaN, 30.0, 5.0)}), std::vector<double>{1500.0}, "a 1/T alias");
    eqv(keep({det(1532.99, "tbd", CE, NaN, 30.0, 5.0), A}), std::vector<double>{1500.0},
                 "a 1/T alias, listed first");
    eqv(keep({A, det(1467.0, "tbd", CE, NaN, 30.0, -5.0)}), std::vector<double>{1500.0}, "below, too");
    eqv(keep({A, det(1532.99, "tbd", CE, NaN, 34.0, 5.0)}), (std::vector<double>{1500.0, 1532.99}),
                 "not under 1/ALIAS_RATIO: kept");
    eqv(keep({A, det(1532.99, "tbd", CE, NaN, 30.0, 11.5)}), (std::vector<double>{1500.0, 1532.99}),
                 "past 1.5 symbols: kept");
    eqv(keep({A, det(1533.5, "tbd", CE, NaN, 30.0, 0.0)}), (std::vector<double>{1500.0, 1533.5}),
                 "past ALIAS_HZ: kept");
    eqv(keep({A, det(1565.98, "tbd", CE, NaN, 20.0, 18.9)}), std::vector<double>{1500.0},
                 "a 2/T alias within 2.5 symbols");
    eqv(keep({A, det(1565.98, "tbd", CE, NaN, 20.0, 19.0)}), (std::vector<double>{1500.0, 1565.98}),
                 "a 2/T alias past 2.5 symbols: kept");
    eqv(fs_of(rc::aliases(std::vector<types::Detection>{A, det(1516.49, "tbd", L, NaN, 20.0, 0.0)}, TL)),
                 std::vector<double>{1500.0}, "L's 1/T_L alias");
    check::is_true(rc::aliases_idx(std::vector<types::Detection>{det(1.0, "x", CE, NaN, 5.0), det(9.0, "x", CE, NaN, 7.0),
                                                                 det(20.0, "x", CE, NaN, 5.0)}) ==
                       std::vector<std::size_t>{1, 0, 2},
                   "strongest first, ties in their order");
    throws<std::invalid_argument>(
        [&] { rc::aliases(std::vector<types::Detection>{A, det(1532.99, "tbd", CE, NaN, 30.0, std::nullopt)}); },
        "an alias candidate without timing");
    eqv(keep({A, det(1600.0, "tbd", CE, NaN, 30.0, std::nullopt)}), (std::vector<double>{1500.0, 1600.0}),
                 "no timing, but far in frequency: its timing is never read");

    // _ghosts: the first of equal Z_ref stands
    const auto ghosts = [&](std::vector<types::Detection> v) { return fs_of(rc::ghosts(v)); };
    eqv(ghosts({A, det(1510.0, "tbd", CE, NaN, 5.0, 10.0)}), std::vector<double>{1500.0}, "a ghost");
    eqv(ghosts({A, det(1510.0, "tbd", CE, NaN, 5.0, 30.0)}), (std::vector<double>{1500.0, 1510.0}),
                 "outside the timing window: no ghost");
    eqv(ghosts({det(1500.0, "x", CE, NaN, 50.0), det(1510.0, "x", CE, NaN, 50.0)}),
                 std::vector<double>{1500.0}, "a tie: the first stands");
    eqv(ghosts({det(1510.0, "x", CE, NaN, 50.0), det(1500.0, "x", CE, NaN, 50.0)}),
                 std::vector<double>{1510.0}, "a tie, reversed");

    // _a3_lines: tbd candidates with a finite a3_z; the centre is the band's
    const std::vector<types::Detection> cands{det(1500.0, "tbd", CE, 5.0), det(1500.0, "tbd", L, 7.0),
                                              det(1600.0, "preamble", CE, 9.0), det(1700.0, "tbd", CE, NaN),
                                              det(1800.0, "tbd", CE, std::numeric_limits<double>::infinity())};
    const auto lines = rc::a3_lines(cands);
    check::equal(lines.size(), std::size_t{2}, "two A3 lines");
    check::is_true(lines.size() == 2 && lines[0].line == 1500.0 && lines[0].centre == 1500.0 && lines[0].z == 5.0 &&
                       lines[1].line == 1500.0 && lines[1].centre == 1490.5 && lines[1].z == 7.0,
                   "(line, centre, Z): an L line's centre is 9.5 Hz under it");

    // _owned: a whole-slot line in a preamble-found signal's band, no
    // stronger in A3 than that signal's own line
    const std::vector<types::Detection> ce_found{det(1500.0, "preamble")};
    const std::vector<rc::A3Line> own{{1500.1, 1500.1, 8.0}};
    check::is_true(rc::owned(det(1510.0, "tbd", CE, 5.0), ce_found, own, false), "owned: weaker than the own line");
    check::is_true(rc::owned(det(1510.0, "tbd", CE, 8.0), ce_found, own, false), "owned: as strong as it");
    check::is_true(!rc::owned(det(1510.0, "tbd", CE, 8.5), ce_found, own, false), "stronger: gated");
    check::is_true(!rc::owned(det(1510.0, "tbd", CE, 5.0), ce_found, own, true), "the old rule: gated");
    check::is_true(!rc::owned(det(1510.0, "hint", CE, 5.0), ce_found, own, false), "a hint: gated");
    check::is_true(!rc::owned(det(1510.0, "tbd", CE, NaN), ce_found, own, false), "no A3 Z: gated");
    check::is_true(!rc::owned(det(1520.0, "tbd", CE, 5.0), ce_found, own, false), "GHOST_HZ away: gated");
    check::is_true(!rc::owned(det(1510.0, "tbd", CE, 5.0), ce_found, {}, false), "the owner has no A3 line: gated");
    check::is_true(!rc::owned(det(1510.0, "tbd", CE, 5.0), std::vector<types::Detection>{det(1500.0, "tbd")}, own,
                              false),
                   "only a preamble-found signal owns its band");
    const std::vector<types::Detection> l_found{det(1500.0, "preamble", L)};
    const std::vector<rc::A3Line> l_own{{1500.0, 1490.5, 6.0}};
    check::is_true(rc::owned(det(1478.0, "tbd", CE, 5.0), l_found, l_own, false), "inside an L signal's band, low");
    check::is_true(rc::owned(det(1503.0, "tbd", CE, 5.0), l_found, l_own, false), "... and high");
    check::is_true(!rc::owned(det(1477.8, "tbd", CE, 5.0), l_found, l_own, false), "under L_OWN_BELOW_HZ");
    check::is_true(!rc::owned(det(1503.2, "tbd", CE, 5.0), l_found, l_own, false), "over L_OWN_ABOVE_HZ");
    check::is_true(rc::owned(det(1478.0, "tbd", CE, 5.0), l_found, std::vector<rc::A3Line>{{1300.0, 1493.0, 6.0}}, false),
                   "an own line by its centre");
    {
        // the environment, when old_gating is not given
        rules::set_old(rules::GATING, true);
        const bool env_old = !rc::owned(det(1510.0, "tbd", CE, 5.0), ce_found, own);
        rules::set_old(rules::GATING, false);
        const bool env_new = rc::owned(det(1510.0, "tbd", CE, 5.0), ce_found, own);
        check::is_true(env_old && env_new, "owned reads QRSS_OLD_GATING when not given it");
    }

    // _suspect_neighbour (frequencies) and _suspect_beside (centres)
    const std::vector<types::PassResult> out{pass(1500.0, 60.0, false)};
    check::is_true(rc::suspect_neighbour(pass(1510.0, 5.0, true), out), "a weak SUSPECT beside a sound pass");
    check::is_true(!rc::suspect_neighbour(pass(1510.0, 6.0, true), out), "at GHOST_Z_RATIO: kept");
    check::is_true(!rc::suspect_neighbour(pass(1510.0, 5.0, false), out), "not SUSPECT: kept");
    check::is_true(!rc::suspect_neighbour(pass(1521.0, 5.0, true), out), "GHOST_HZ away: kept");
    check::is_true(!rc::suspect_neighbour(pass(1510.0, 5.0, true), std::vector<types::PassResult>{pass(1500.0, 60.0, true)}),
                   "beside a SUSPECT pass: kept");
    const std::vector<types::PassResult> l_out{pass(1500.0, 60.0, false, 0.0, L)};
    check::is_true(rc::suspect_beside(pass(1475.0, 5.0, true), l_out) && !rc::suspect_neighbour(pass(1475.0, 5.0, true), l_out),
                   "beside an L pass by its centre (15.5 Hz), not by its pilot (25 Hz)");
}

// --- the orchestration over stubs -------------------------------------------------------------

// Every stage a stub logging its call. verify / verify_l accept the
// frequencies in `z` (with that Z_ref, tau0 0), the clock advances one
// second per reading.
struct Stub {
    std::vector<types::Detection> cands;
    std::map<double, double> z;
    double resid = 1.0;
    std::vector<std::string> log;
    double clock = 0.0;
    int now_calls = 0;
    std::map<double, types::PassResult> passes;   // receive_pass: f -> pass (absent: None)
    std::vector<types::Detection> detected;       // detect's answer

    std::optional<types::Detection> verified(const char* who, const types::Detection& d) {
        log.push_back(std::string(who) + " " + num(d.f_hz));
        const auto it = z.find(d.f_hz);
        if (it == z.end()) return std::nullopt;
        types::Detection v = det(d.f_hz, d.method.c_str(), d.waveform, NaN, it->second, 0.0);
        v.lead_in_s = d.lead_in_s;
        return v;
    }

    rc::SlotSeams seams() {
        rc::SlotSeams s;
        s.candidates = [this](const rc::Prepared&, rc::Selection sel, bool joint, bool live_only,
                              std::span<const aq::Hint> hints) {
            log.push_back(std::string("candidates ") + (joint ? "joint" : "ce") + " " + std::to_string(sel.size()) +
                          (live_only ? " live" : "") + " hints " + std::to_string(hints.size()));
            return cands;
        };
        s.pass.channel_for = [this](const rc::Prepared&, const fr::FrameSpec& spec, const types::Detection& d,
                                    std::span<const double> others,
                                    const std::optional<qrss::Rational>&) -> std::shared_ptr<const tk::ChanCapture> {
            log.push_back("chan " + spec.name() + " " + num(d.f_hz) + " " + nums(others));
            return nullptr;
        };
        s.pass.verify = [this](std::shared_ptr<const tk::ChanCapture>, const fr::FrameSpec&, const types::Detection& d,
                               std::span<const std::pair<double, double>>) { return verified("verify", d); };
        s.pass.verify_l = [this](std::shared_ptr<const tk::ChanCapture>, const fr::FrameSpec&, const types::Detection& d,
                                 std::span<const std::pair<double, double>>) { return verified("verify_l", d); };
        s.verify_track = [this](std::shared_ptr<const tk::ChanCapture> chan, const fr::FrameSpec& spec,
                                const types::Detection& v, std::span<const std::pair<double, double>>) {
            log.push_back("verify_track " + num(v.f_hz));
            return std::make_shared<tk::TrackResult>(spec, chan, types::Timing{}, tk::PathFn::from_parts(0.0, 1.0, {0.0}, std::nullopt));
        };
        s.ce_resid_ratio = [this](const tk::TrackResult&) {
            log.push_back("resid");
            return resid;
        };
        s.lead_in_span = [this](const rc::Prepared&, double f, double tau0_s, std::optional<double> t_sym) {
            log.push_back("lead_in " + num(f) + " " + num(tau0_s) + " " + (t_sym ? num(*t_sym) : "default"));
            return 2.5;
        };
        s.detect = [this](const rc::Prepared&, rc::Selection sel, bool joint, bool, std::optional<std::int64_t>,
                          std::optional<double>, std::span<const std::pair<double, double>>, std::span<const aq::Hint>) {
            log.push_back(std::string("detect ") + (joint ? "joint " : "ce ") + std::to_string(sel.size()));
            return detected;
        };
        s.receive_pass = [this](const rc::Prepared&, const fr::FrameSpec& spec, const types::Detection& d,
                                const rc::PassArgs& a) -> std::optional<types::PassResult> {
            log.push_back("receive_pass " + spec.name() + " " + num(d.f_hz) + " " + nums(a.exclude_hz) + " " +
                          a.estimator + (a.round_b ? "" : " round A"));
            const auto it = passes.find(d.f_hz);
            if (it == passes.end()) return std::nullopt;
            return it->second;
        };
        s.now = [this] {
            ++now_calls;
            clock += 1.0;
            return clock;
        };
        s.old_gating = false;
        return s;
    }
};

void expect_log(const Stub& s, const std::vector<std::string>& want, const std::string& what) {
    check::equal(s.log.size(), want.size(), what + ": the number of calls");
    for (std::size_t i = 0; i < std::min(s.log.size(), want.size()); ++i)
        check::equal(s.log[i], want[i], what + ": call " + std::to_string(i));
}

rc::Prepared stub_prep() {
    rc::Prepared p;
    p.q = Q;
    p.t0_index = 0.0;
    p.fs = cfg::FE_FS;
    return p;
}

// CE detect: P1, P2 preamble hits; T0 the owner's own A3 line and T1 a
// weaker line in its band (both owned); H1 a hint merging with P2 (counted);
// T3 refused at the gate; T4 accepted, its lead-in measured.
std::vector<types::Detection> ce_cands() {
    return {det(1500.0, "preamble"), det(1700.0, "preamble"),   det(1500.5, "tbd", CE, 6.0), det(1510.0, "tbd", CE, 3.0),
            det(1702.0, "hint"),     det(1300.0, "tbd", CE, 9.0), det(1900.0, "tbd", CE, 9.0)};
}

void test_detect_stubbed() {
    const rc::Prepared prep = stub_prep();
    const std::vector<fr::FrameSpec> tiny{fr::get("tiny")};
    const auto run = [&](Stub& s, const rc::DetectOptions& o) {
        const rc::SlotSeams seams = s.seams();
        return rc::detect(prep, tiny, o, seams);
    };
    const std::map<double, double> z{{1500.0, 50.0}, {1700.0, 30.0}, {1900.0, 8.0}};
    const std::vector<std::string> head{"candidates ce 1 hints 0", "chan tiny 1500 [1700]", "verify 1500",
                                        "chan tiny 1700 [1500]", "verify 1700"};
    {
        Stub s{ce_cands(), z};
        const auto got = run(s, {});
        std::vector<std::string> want = head;
        want.insert(want.end(), {"chan tiny 1300 [1500 1700]", "verify 1300", "chan tiny 1900 [1500 1700]",
                                 "verify 1900", "lead_in 1900 0 default"});
        expect_log(s, want, "CE detect");
        eqv(fs_of(got), (std::vector<double>{1500.0, 1700.0, 1900.0}), "CE detect: strongest first");
        check::is_true(got.size() == 3 && got[0].lead_in_s == 0.0 && got[2].lead_in_s == 2.5,
                       "a preamble hit keeps its lead-in; a whole-slot one is measured");
        check::equal(s.now_calls, 0, "no budget: the clock is never read");
    }
    {
        // max_candidates 3: P1, P2 and H1 counted (T0, T1 owned are not)
        Stub s{ce_cands(), z};
        rc::DetectOptions o;
        o.max_candidates = 3;
        const auto got = run(s, o);
        expect_log(s, head, "max_candidates 3");
        eqv(fs_of(got), (std::vector<double>{1500.0, 1700.0}), "max_candidates 3: two found");
        o.max_candidates = 0;
        Stub s0{ce_cands(), z};
        check::is_true(run(s0, o).empty(), "max_candidates 0: nothing gated");
        expect_log(s0, {"candidates ce 1 hints 0"}, "max_candidates 0");
    }
    {
        // budget 3.5 s on a clock of 1 s a reading: the deadline 1 + 3.5;
        // readings 2, 3, 4 pass (P1, P2, T0 owned), 5 stops at T1
        Stub s{ce_cands(), z};
        rc::DetectOptions o;
        o.budget_s = 3.5;
        const auto got = run(s, o);
        expect_log(s, head, "budget 3.5 s");
        check::equal(s.now_calls, 5, "budget 3.5 s: the deadline's reading and one per candidate until it passed");
        eqv(fs_of(got), (std::vector<double>{1500.0, 1700.0}), "budget 3.5 s: two found");
        Stub s0{ce_cands(), z};
        o.budget_s = 0.0;
        check::is_true(run(s0, o).empty() && s0.now_calls == 2, "budget 0: nothing gated");
    }
    {
        // the candidates given (no acquisition), skip_hz: T3 merges with a
        // known signal and every gate leaves it out of the noise
        Stub s{{}, z};
        rc::DetectOptions o;
        o.cands = ce_cands();
        o.skip_hz = {1300.2};
        const auto got = run(s, o);
        expect_log(s,
                   {"chan tiny 1500 [1300.2 1700]", "verify 1500", "chan tiny 1700 [1300.2 1500]", "verify 1700",
                    "chan tiny 1900 [1300.2 1500 1700]", "verify 1900", "lead_in 1900 0 default"},
                   "cands given, skip_hz");
        eqv(fs_of(got), (std::vector<double>{1500.0, 1700.0, 1900.0}), "cands given: the same found");
    }
    {
        // the old gating rule: T0 (merging with P1) and T1 gated
        Stub s{ce_cands(), z};
        rc::SlotSeams seams = s.seams();
        seams.old_gating = true;
        const auto got = rc::detect(prep, tiny, {}, seams);
        std::vector<std::string> want = head;
        want.insert(want.end(), {"chan tiny 1510 [1500 1700]", "verify 1510", "chan tiny 1300 [1500 1700]",
                                 "verify 1300", "chan tiny 1900 [1500 1700]", "verify 1900", "lead_in 1900 0 default"});
        expect_log(s, want, "old gating");
        eqv(fs_of(got), (std::vector<double>{1500.0, 1700.0, 1900.0}), "old gating: the same found");
    }
    {
        // after the gate: one detection per MERGE_HZ, then aliases and ghosts
        Stub s{{det(1500.0, "preamble"), det(1532.99, "preamble"), det(1503.0, "hint"), det(1512.0, "tbd", CE, NaN)},
               {{1500.0, 40.0}, {1532.99, 10.0}, {1512.0, 3.0}}};
        rc::SlotSeams seams = s.seams();
        const auto got = rc::detect(prep, tiny, {}, seams);
        eqv(fs_of(got), std::vector<double>{1500.0}, "an alias and a ghost dropped after the gate");
    }
}

// The joint receiver: LP an L preamble hit (centre 1490.5), CP a CE one; LT
// L's own A3 line and CT a CE line in L's band (owned); CH a hint at L's
// centre (merged, counted); W7 a waveform not listened for (counted); C1 a
// CE accept the residual ratio refuses, C2 one it keeps; L2 an L line.
void test_detect_joint_stubbed() {
    const rc::Prepared prep = stub_prep();
    const std::vector<fr::FrameSpec> specs{fr::get("tiny"), fr::get_l("l-tiny")};
    const std::vector<types::Detection> cands{
        det(1500.0, "preamble", L), det(1700.0, "preamble", CE), det(1500.2, "tbd", L, 6.0), det(1492.0, "tbd", CE, 5.0),
        det(1491.0, "hint", CE),    det(1250.0, "tbd", 7, NaN),   det(1300.0, "tbd", CE, 9.0), det(1100.0, "tbd", CE, 9.0),
        det(2000.0, "tbd", L, 9.0)};
    const std::map<double, double> z{{1500.0, 60.0}, {1700.0, 40.0}, {1300.0, 7.0}, {1100.0, 20.0}, {2000.0, 15.0}};
    Stub s{cands, z, 3.0};
    const rc::SlotSeams seams = s.seams();
    const auto got = rc::detect(prep, specs, {}, seams);
    const std::string T = num(cfg::T_SYM), TL = num(fr::get_l("l-tiny").t_sym());
    expect_log(s,
               {"candidates joint 2 hints 0", "chan l-tiny 1500 [1700]", "verify_l 1500", "chan tiny 1700 [1490.5]",
                "verify 1700", "verify_track 1700", "resid", "chan tiny 1300 [1490.5 1700]", "verify 1300",
                "verify_track 1300", "resid", "chan tiny 1100 [1490.5 1700]", "verify 1100", "verify_track 1100",
                "resid", "lead_in 1100 0 " + T, "chan l-tiny 2000 [1100 1490.5 1700]", "verify_l 2000",
                "lead_in 2000 0 " + TL},
               "joint detect");
    eqv(fs_of(got), (std::vector<double>{1500.0, 1700.0, 1100.0, 2000.0}), "joint detect: strongest first");
    check::is_true(got.size() == 4 && got[0].waveform == L && got[3].waveform == L && got[1].waveform == CE,
                   "joint detect: the waveforms kept");
    {
        // max_candidates 4: LP, CP, CH, W7 (LT and CT owned are not counted)
        Stub s4{cands, z, 3.0};
        rc::DetectOptions o;
        o.max_candidates = 4;
        const auto g4 = rc::detect_joint(prep, specs, o, s4.seams());
        eqv(fs_of(g4), (std::vector<double>{1500.0, 1700.0}), "joint max_candidates 4");
        check::equal(s4.log.size(), std::size_t{7}, "joint max_candidates 4: the gates of LP and CP only");
    }
    {
        // the residual ratio under CE_RESID_MIN: C1's accept stands
        Stub s1{cands, z, 1.5};
        const auto g1 = rc::detect(prep, specs, {}, s1.seams());
        eqv(fs_of(g1), (std::vector<double>{1500.0, 1700.0, 1100.0, 2000.0, 1300.0}),
                     "joint, R under CE_RESID_MIN: every CE accept kept");
    }
    {
        // gate on its own: a refused CE accept is None, a lead-in at the
        // frame's own symbol
        Stub g{{}, {{1300.0, 7.0}, {1100.0, 20.0}}, 3.0};
        const rc::SlotSeams gs = g.seams();
        const std::vector<double> others{1700.0};
        check::is_true(!rc::gate(prep, fr::get("tiny"), det(1300.0, "tbd"), others, {}, gs), "gate: refused on R");
        const auto v = rc::gate(prep, fr::get("tiny"), det(1100.0, "tbd"), others, {}, gs);
        check::is_true(v && v->lead_in_s == 2.5, "gate: accepted, its lead-in measured");
    }
}

// receive_slot after tracking: a ghost dropped, a weak SUSPECT neighbour
// dropped, two passes of one signal merged (the stronger standing), a pass
// refused; each receive_pass excluding the other detections.
void test_receive_slot_stubbed() {
    const rc::Prepared prep = stub_prep();
    const std::vector<fr::FrameSpec> tiny{fr::get("tiny")};
    Stub s;
    s.detected = {det(1500.0), det(1510.0), det(1560.0), det(1515.0), det(1562.0), det(1600.0)};
    s.passes = {{1500.0, pass(1500.02, 60.0, false, 0.0)},  {1510.0, pass(1508.0, 30.0, false, 5.0)},
                {1560.0, pass(1560.0, 5.0, true, 300.0)},   {1515.0, pass(1517.0, 4.0, true, 100.0)},
                {1562.0, pass(1562.0, 9.0, false, 200.0)}};
    rc::SlotOptions o;
    o.estimator = "decoupled";
    o.round_b = false;
    const auto got = rc::receive_slot(rc::CaptureRef(&prep), tiny, o, s.seams());
    expect_log(s,
               {"detect ce 1", "receive_pass tiny 1500 [1510 1560 1515 1562 1600] decoupled round A",
                "receive_pass tiny 1510 [1500 1560 1515 1562 1600] decoupled round A",
                "receive_pass tiny 1560 [1500 1510 1515 1562 1600] decoupled round A",
                "receive_pass tiny 1515 [1500 1510 1560 1562 1600] decoupled round A",
                "receive_pass tiny 1562 [1500 1510 1560 1515 1600] decoupled round A",
                "receive_pass tiny 1600 [1500 1510 1560 1515 1562] decoupled round A"},
               "receive_slot");
    eqv(fs_of(got), (std::vector<double>{1500.02, 1562.0}), "receive_slot: ghost, SUSPECT, merge");

    // dets given: no detection; the joint receiver with a waveform not
    // listened for received with the twin of the first frame
    Stub j;
    j.passes = {{1500.0, pass(1500.0, 60.0, false, 0.0, L)}, {1700.0, pass(1700.0, 30.0, false)}};
    rc::SlotOptions oj;
    oj.dets = std::vector<types::Detection>{det(1500.0, "preamble", L), det(1700.0)};
    const auto gj = rc::receive_slot_joint(prep, tiny, oj, j.seams());
    expect_log(j, {"receive_pass l-tiny 1500 [1700] joint", "receive_pass tiny 1700 [1490.5] joint"},
               "the joint receiver, dets given");
    eqv(fs_of(gj), (std::vector<double>{1500.0, 1700.0}), "the joint receiver: both");

    // across waveforms by centre: a weak SUSPECT CE pass 15.5 Hz from an L
    // pass's centre goes; within a waveform the ghost rule at its symbol
    Stub k;
    k.passes = {{1500.0, pass(1500.0, 60.0, false, 0.0, L)},
                {1475.0, pass(1475.0, 5.0, true, 500.0)},
                {1515.0, pass(1515.0, 9.0, false, 40.0, L)},
                {1491.0, pass(1491.0, 70.0, false, 900.0)}};
    rc::SlotOptions ok;
    ok.dets = std::vector<types::Detection>{det(1500.0, "preamble", L), det(1475.0), det(1515.0, "tbd", L),
                                            det(1491.0)};
    const std::vector<fr::FrameSpec> both{fr::get("tiny"), fr::get_l("l-tiny")};
    const auto gk = rc::receive_slot(rc::CaptureRef(&prep), both, ok, k.seams());
    eqv(fs_of(gk), std::vector<double>{1491.0},
                 "joint: SUSPECT beside dropped, L ghost dropped (40 < 45.5 samples), one centre the stronger");
}

// --- end to end -----------------------------------------------------------------------------

std::vector<double> closed_form_latents() {
    std::vector<double> v(3 * 52800);
    for (std::size_t i = 0; i < v.size(); ++i) {
        const double g = static_cast<double>(i);
        v[i] = std::sin(std::fmod(0.001 * g * g, 6.283)) + 0.3 * std::cos(g);
    }
    return v;
}

const pc::StoredPicture& picture() {
    static const pc::StoredPicture p = pc::StoredPicture::from_latents(closed_form_latents(), 0xD1D8, 1);
    return p;
}

std::vector<float> pass_audio(api::Waveform w, const char* frame, double freq) {
    api::PassRequest r;
    r.q = Q;
    r.waveform = w;
    r.freq_hz = freq;
    r.callsign = "K1ABC";
    r.frame = frame;
    return api::pass_audio(api::Source(picture()), r);
}

// Gaussian noise from a hash (splitmix64, Box-Muller): the same on every
// platform to the libm's last bits, which the checks here do not reach.
std::vector<double> noise(std::size_t n, double sigma, std::uint64_t seed) {
    std::vector<double> out(n);
    std::uint64_t s = seed;
    const auto next = [&] {
        std::uint64_t z = (s += 0x9E3779B97F4A7C15ull);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        z ^= z >> 31;
        return (static_cast<double>(z >> 11) + 0.5) / 9007199254740992.0;
    };
    for (std::size_t i = 0; i < n; i += 2) {
        const double r = sigma * std::sqrt(-2.0 * std::log(next())), a = 2.0 * std::numbers::pi * next();
        out[i] = r * std::cos(a);
        if (i + 1 < n) out[i + 1] = r * std::sin(a);
    }
    return out;
}

// Sum of the passes' audio at SNR2500 snr_db each (signal power over its
// keyed span, noise in 2500 Hz of the 4 kHz band).
std::vector<double> mix(const std::vector<std::vector<float>>& parts, double snr_db, std::uint64_t seed) {
    std::size_t n = 0;
    double p_sig = 0.0;
    for (const auto& x : parts) {
        n = std::max(n, x.size());
        double e = 0.0;
        std::size_t k = 0;
        for (const float v : x)
            if (v != 0.0f) {
                e += static_cast<double>(v) * v;
                ++k;
            }
        p_sig = std::max(p_sig, e / static_cast<double>(std::max<std::size_t>(k, 1)));
    }
    const double sigma = std::sqrt(p_sig / (2500.0 / 4000.0) / std::pow(10.0, snr_db / 10.0));
    std::vector<double> out = noise(n, sigma, seed);
    for (const auto& x : parts)
        for (std::size_t i = 0; i < x.size(); ++i) out[i] += x[i];
    return out;
}

const api::Capture& joint_capture() {
    static const api::Capture c = [] {
        const auto a = mix({pass_audio(api::Waveform::L, "tiny", 1500.0), pass_audio(api::Waveform::CE, "tiny", 1560.0)},
                           -3.0, 11);
        return api::capture_from_audio(a, Q);
    }();
    return c;
}

bool same_passes(const std::vector<types::PassResult>& a, const std::vector<types::PassResult>& b) {
    if (a.size() != b.size()) return false;
    for (std::size_t i = 0; i < a.size(); ++i)
        if (a[i].uid != b[i].uid || a[i].f_hz != b[i].f_hz || a[i].z != b[i].z || a[i].w != b[i].w ||
            a[i].report.z_ref != b[i].report.z_ref || a[i].report.snr2500_db != b[i].report.snr2500_db ||
            a[i].timing.tau0 != b[i].timing.tau0)
            return false;
    return true;
}

void test_receive_end_to_end() {
    const api::Capture& cap = joint_capture();
    check::equal(cap.fs, std::int64_t{cfg::FE_FS}, "the capture is the front end's");
    check::equal(cap.t0_index, 12.0 * cfg::FE_FS, "t0 12 s in");
    api::ReceiveConfig both;
    both.frames = {fr::get("tiny"), fr::get_l("l-tiny")};
    const auto got = api::receive_slot(cap, both);
    check::equal(got.size(), std::size_t{2}, "joint: two passes");
    if (got.size() == 2) {
        check::is_true(got[0].waveform != got[1].waveform, "joint: one of each waveform");
        for (const auto& p : got) {
            const double want = p.waveform == L ? 1500.0 : 1560.0;
            check::is_true(std::abs(p.f_hz - want) < 0.1, "joint: " + p.frame + " at its frequency (" + num(p.f_hz) + ")");
            check::equal(p.frame, std::string(p.waveform == L ? "l-tiny" : "tiny"), "joint: the frame");
            check::equal(p.q, Q, "joint: q");
            check::is_true(!p.cw.has_value() && !p.header.has_value(), "joint: TINY has no header or windows");
        }
    }
    // CE alone: the CE signal (L's is not CE's to receive)
    api::ReceiveConfig ce;
    ce.frames = {fr::get("tiny")};
    const auto got_ce = api::receive_slot(cap, ce);
    check::is_true(std::any_of(got_ce.begin(), got_ce.end(),
                               [](const types::PassResult& p) { return p.waveform == CE && std::abs(p.f_hz - 1560.0) < 0.1; }),
                   "CE alone: the CE pass");
    // live_only and max_candidates 0
    api::ReceiveConfig none = both;
    none.max_candidates = 0;
    check::is_true(api::receive_slot(cap, none).empty(), "max_candidates 0: nothing");
    api::ReceiveConfig live = both;
    live.live_only = true;
    check::equal(api::receive_slot(cap, live).size(), std::size_t{2}, "live_only: both, on their preambles");

    // the noise alone
    const api::Capture quiet = api::capture_from_audio(noise(joint_capture().fe.size() * 2, 0.1, 5), Q);
    check::is_true(api::receive_slot(quiet, both).empty(), "noise alone: nothing");

    // four threads at once, each the serial answer
    std::vector<std::vector<types::PassResult>> par(4);
    {
        std::vector<std::thread> ts;
        for (int i = 0; i < 4; ++i)
            ts.emplace_back([&, i] { par[static_cast<std::size_t>(i)] = api::receive_slot(cap, i % 2 ? ce : both); });
        for (auto& t : ts) t.join();
    }
    check::is_true(same_passes(par[0], got) && same_passes(par[2], got), "threads: the joint answer");
    check::is_true(same_passes(par[1], got_ce) && same_passes(par[3], got_ce), "threads: the CE answer");

    // float32 audio is widened exactly
    {
        std::vector<float> a32(1000);
        std::vector<double> a64(1000);
        for (std::size_t i = 0; i < a32.size(); ++i) a64[i] = a32[i] = static_cast<float>(std::sin(0.01 * double(i)));
        check::is_true(api::capture_from_audio(std::span<const float>(a32), Q, 5.0).fe ==
                           api::capture_from_audio(std::span<const double>(a64), Q, 5.0).fe,
                       "float32 audio widened exactly");
        check::equal(api::capture_from_audio(std::span<const double>(a64), Q, api::t0_s(Q) - 3.0).t0_index,
                     3.0 * cfg::FE_FS, "start_unix: t0 3 s in");
    }
}

void test_short_header() {
    const auto a = mix({pass_audio(api::Waveform::CE, "short", 1500.0)}, -6.0, 3);
    const api::Capture cap = api::capture_from_audio(a, Q);
    api::ReceiveConfig c;
    c.frames = {fr::get("short")};
    const auto got = api::receive_slot(cap, c);
    check::equal(got.size(), std::size_t{1}, "SHORT: one pass");
    if (got.size() == 1) {
        check::is_true(got[0].header.has_value() && got[0].header->callsign == "K1ABC", "SHORT: the header decoded");
        check::is_true(got[0].cw && got[0].cw->text && *got[0].cw->text == "K1ABC" && got[0].cw->agrees.value_or(false),
                       "SHORT: the callsign windows read and agree");
        check::is_true(std::abs(got[0].f_hz - 1500.0) < 0.05, "SHORT: at its frequency");
    }
}

// The rules given per call are this call's, whatever the environment says.
void test_rules_per_call() {
    // EDGE_NOISE matters at a band edge: a CE TINY at 2650 Hz
    const auto a = mix({pass_audio(api::Waveform::CE, "tiny", 2650.0)}, -3.0, 21);
    const api::Capture cap = api::capture_from_audio(a, Q);
    api::ReceiveConfig c;
    c.frames = {fr::get("tiny")};
    rules::set_old(rules::EDGE_NOISE, false);
    rules::set_old(rules::GATING, false);
    const auto now_rule = api::receive_slot(cap, c);
    api::ReceiveConfig old = c;
    old.old_edge_noise = true;
    const auto old_rule = api::receive_slot(cap, old);
    check::is_true(now_rule.size() == 1 && old_rule.size() == 1, "at 2650 Hz: a pass either way");
    if (now_rule.size() == 1 && old_rule.size() == 1)
        check::is_true(now_rule[0].report.snr2500_db < old_rule[0].report.snr2500_db,
                       "the old edge noise reads the floor low (SNR high): " + num(now_rule[0].report.snr2500_db) +
                           " < " + num(old_rule[0].report.snr2500_db));
    // the environment set: the config still decides
    rules::set_old(rules::EDGE_NOISE, true);
    rules::set_old(rules::GATING, true);
    const auto env_old_cfg_new = api::receive_slot(cap, c);
    const auto receiver_env = rc::receive_slot(rc::CaptureRef(&cap), c.frames);
    rules::set_old(rules::EDGE_NOISE, false);
    rules::set_old(rules::GATING, false);
    check::is_true(same_passes(env_old_cfg_new, now_rule), "QRSS_OLD_* set, the config not: today's rules");
    api::ReceiveConfig both_old = c;
    both_old.old_edge_noise = true;
    both_old.old_gating = true;
    check::is_true(same_passes(api::receive_slot(cap, both_old), receiver_env),
                   "the config's old rules are the environment's (the receiver reading it)");
}

// ce_resid_ratio on a real gate-V track.
void test_ce_resid_ratio() {
    const api::Capture& cap = joint_capture();
    const rc::Prepared prep = rc::prepare(cap);
    const fr::FrameSpec& tiny = fr::get("tiny");
    // the detections joint acquisition found (each with its path)
    const std::vector<fr::FrameSpec> both{tiny, fr::get_l("l-tiny")};
    const std::vector<types::Detection> found = rc::candidates(prep, both, true);
    const auto at = [&](double f, std::int64_t wf) -> std::optional<types::Detection> {
        for (const auto& c : found)
            if (c.waveform == wf && c.method == "preamble" && std::abs(c.f_hz - f) < 1.0) return c;
        return std::nullopt;
    };
    const auto dce = at(1560.0, CE), dl = at(1500.0, L);
    check::is_true(dce && dl, "joint acquisition: both preambles");
    if (!dce || !dl) return;
    const types::Detection d = *dce;
    const auto chan = std::make_shared<const tk::ChanCapture>(rc::channel_for(prep, tiny, d, std::vector<double>{1490.5}));
    const auto v = tk::verify(chan, tiny, d);
    check::is_true(v.has_value(), "the CE signal passes gate V");
    if (!v) return;
    tk::VerifyArgs va;
    va.fit = false;
    tk::TrackResult tr = tk::verify_track(chan, tiny, *v, va);
    const double R = rc::ce_resid_ratio(tr);
    check::is_true(R > 0.8 && R < rc::CE_RESID_MIN, "a CE signal's R is about 1 (" + num(R) + ")");
    // the same sum, by hand in long double
    const tk::Classes* cl = tr.ce_classes();
    const auto lay = fr::layout(tiny);
    const tk::Templates t =
        tk::templates(tiny, tr.pos, cl->mu, cl->nu, nullptr, false, tiny.keyed_start_pos() - tr.lead_in_s / cfg::T_SYM);
    long double num_s = 0, den_s = 0;
    std::size_t used = 0;
    for (std::size_t s = 0; s < cl->known.size(); ++s) {
        if (!cl->known[s]) continue;
        const std::int64_t g = lay->pos[s] - tr.pos[0];
        if (g < 0 || g >= static_cast<std::int64_t>(tr.pos.size()) || tr.psi[static_cast<std::size_t>(g)] < 0.5) continue;
        const auto k = static_cast<std::size_t>(g);
        const std::complex<long double> m(tr.m[k].real(), tr.m[k].imag()), u(tr.u[k].real(), tr.u[k].imag()),
            c(t.c[k].real(), t.c[k].imag());
        const long double N = tr.N[k], P = tr.P[k], u2 = std::norm(u), ci = c.imag(), c2 = std::norm(c);
        const long double V = ci * ci * (N * u2 + c2 * u2 * P + N * P) / 2;
        if (!(V > 0)) continue;
        const long double w = ci * ci * u2 / V, om = w * w * ci * ci * u2;
        num_s += om * std::norm(m - u * c);
        den_s += om * (N + c2 * P);
        ++used;
    }
    check::is_true(used > 10, "known symbols used: " + std::to_string(used));
    check::is_true(std::abs(static_cast<double>(num_s / den_s) - R) <= 1e-12 * R,
                   "R is the weighted residual ratio (" + num(static_cast<double>(num_s / den_s)) + ")");
    // the L signal under CE's gate: its accept (if any) is rechecked away
    types::Detection as_ce = *dl;
    as_ce.waveform = CE;
    const auto chl = std::make_shared<const tk::ChanCapture>(rc::channel_for(prep, tiny, as_ce, std::vector<double>{1560.0}));
    if (const auto vl = tk::verify(chl, tiny, as_ce)) {
        const double RL = rc::ce_resid_ratio(tk::verify_track(chl, tiny, *vl, va));
        check::is_true(RL > rc::CE_RESID_MIN, "an L signal under CE's gate: R over CE_RESID_MIN (" + num(RL) + ")");
    }
    // nothing usable: 1.0
    std::fill(tr.psi.begin(), tr.psi.end(), 0.25);
    check::equal(rc::ce_resid_ratio(tr), 1.0, "no known symbol usable: 1.0");
    tr.classes = std::monostate{};
    throws<std::invalid_argument>([&] { rc::ce_resid_ratio(tr); }, "a track without CE classes");
}

// --- wavread -----------------------------------------------------------------------------------

struct WavSpec {
    std::uint16_t tag = 1;
    std::uint16_t channels = 1;
    std::uint32_t rate = 8000;
    std::uint16_t bits = 16;
    std::uint16_t container = 2;   // bytes per sample
    bool extensible = false;
    bool list_chunk = false;
};

void put16(std::string& s, std::uint16_t v) {
    s.push_back(static_cast<char>(v & 0xff));
    s.push_back(static_cast<char>(v >> 8));
}
void put32(std::string& s, std::uint32_t v) {
    put16(s, static_cast<std::uint16_t>(v & 0xffff));
    put16(s, static_cast<std::uint16_t>(v >> 16));
}

std::string wav_bytes(const WavSpec& w, const std::string& data) {
    std::string fmt;
    const std::uint16_t align = static_cast<std::uint16_t>(w.channels * w.container);
    put16(fmt, w.extensible ? 0xFFFE : w.tag);
    put16(fmt, w.channels);
    put32(fmt, w.rate);
    put32(fmt, w.rate * align);
    put16(fmt, align);
    put16(fmt, w.bits);
    if (w.extensible) {
        put16(fmt, 22);
        put16(fmt, w.bits);
        put32(fmt, 0);
        put16(fmt, w.tag);
        fmt += std::string("\x00\x00\x00\x00\x10\x00\x80\x00\x00\xAA\x00\x38\x9B\x71", 14);
    }
    std::string body = "WAVE";
    body += "fmt ";
    put32(body, static_cast<std::uint32_t>(fmt.size()));
    body += fmt;
    if (w.list_chunk) {
        body += "LIST";
        put32(body, 5);
        body += std::string("INFOx", 5) + std::string(1, '\0');   // odd size, its pad byte
    }
    body += "data";
    put32(body, static_cast<std::uint32_t>(data.size()));
    body += data;
    std::string out = "RIFF";
    put32(out, static_cast<std::uint32_t>(body.size()));
    return out + body;
}

// a prefix of this run's own (the steady clock's reading at the first call)
const std::string& tmp_prefix() {
    static const std::string p =
        "sstvae_wavread_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + "_";
    return p;
}

std::string write_tmp(const std::string& name, const std::string& bytes) {
    const auto p = std::filesystem::temp_directory_path() / (tmp_prefix() + name);
    std::ofstream(p, std::ios::binary) << bytes;
    return p.string();
}

void test_wavread() {
    {
        std::string d;
        for (const std::int16_t v : {0, 16384, -32768, 32767}) put16(d, static_cast<std::uint16_t>(v));
        const auto x = wav::read_wav(write_tmp("i16.wav", wav_bytes({}, d)));
        eqv(x, (std::vector<double>{0.0, 16384.0 / 32767.0, -32768.0 / 32767.0, 1.0}), "int16 / 32767");
    }
    {
        const std::string d("\x00\x80\xff", 3);
        WavSpec w;
        w.bits = 8;
        w.container = 1;
        const auto x = wav::read_wav(write_tmp("u8.wav", wav_bytes(w, d)));
        eqv(x, (std::vector<double>{0.0, 128.0 / 255.0, 1.0}), "uint8 / 255 (scipy reads it unsigned)");
    }
    {
        // 24-bit, left-justified in int32
        const std::string d("\x00\x00\x40\x01\x00\x80", 6);
        WavSpec w;
        w.bits = 24;
        w.container = 3;
        const auto x = wav::read_wav(write_tmp("i24.wav", wav_bytes(w, d)));
        eqv(x, (std::vector<double>{1073741824.0 / 2147483647.0, -2147483392.0 / 2147483647.0}),
                     "24-bit / 2147483647, left-justified");
    }
    {
        std::string d;
        put32(d, 0x80000000u);
        put32(d, 123456789u);
        WavSpec w;
        w.bits = 32;
        w.container = 4;
        w.extensible = true;
        w.list_chunk = true;
        const auto x = wav::read_wav(write_tmp("i32.wav", wav_bytes(w, d)));
        eqv(x, (std::vector<double>{-2147483648.0 / 2147483647.0, 123456789.0 / 2147483647.0}),
                     "int32, extensible, a LIST chunk skipped");
    }
    {
        std::string d;
        const float f[3] = {0.1f, -0.75f, 3.0f};
        d.append(reinterpret_cast<const char*>(f), sizeof f);
        WavSpec w;
        w.tag = 3;
        w.bits = 32;
        w.container = 4;
        const auto x = wav::read_wav(write_tmp("f32.wav", wav_bytes(w, d)));
        eqv(x, (std::vector<double>{static_cast<double>(0.1f), -0.75, 3.0}), "float32 widened");
    }
    {
        // stereo: the channels averaged, after scaling
        std::string d;
        for (const std::int16_t v : {1000, 3000, -32768, 32767}) put16(d, static_cast<std::uint16_t>(v));
        WavSpec w;
        w.channels = 2;
        const auto x = wav::read_wav(write_tmp("st.wav", wav_bytes(w, d)));
        eqv(x, (std::vector<double>{(0.0 + 1000.0 / 32767.0 + 3000.0 / 32767.0) / 2.0,
                                             (0.0 + -32768.0 / 32767.0 + 1.0) / 2.0}),
                     "stereo averaged");
    }
    {
        // 16 kHz: resample_poly(x, 1, 2)
        std::string d;
        for (int i = 0; i < 1600; ++i)
            put16(d, static_cast<std::uint16_t>(static_cast<std::int16_t>(10000.0 * std::sin(2 * std::numbers::pi * 500.0 * i / 16000.0))));
        WavSpec w;
        w.rate = 16000;
        const auto x = wav::read_wav(write_tmp("r16.wav", wav_bytes(w, d)));
        check::equal(x.size(), std::size_t{800}, "16 kHz: half the samples");
        double err = 0.0;
        for (std::size_t i = 100; i < 700; ++i)
            err = std::max(err, std::abs(x[i] - 10000.0 / 32767.0 * std::sin(2 * std::numbers::pi * 500.0 * double(i) / 8000.0)));
        check::is_true(err < 1e-3, "16 kHz: the tone at 8 kHz (" + num(err) + ")");
    }
    // refusals
    try {
        wav::read_wav(write_tmp("missing-dir", "") + "/no/such.wav");
        check::fail("a missing file", "did not throw");
    } catch (const wav::ReadError& e) {
        check::is_true(e.err == ENOENT || e.err == ENOTDIR, "a missing file: ReadError with its errno");
    }
    throws<std::invalid_argument>([] { wav::read_wav(write_tmp("text.wav", "not a wav file at all")); }, "not a WAV");
    {
        WavSpec w;
        w.tag = 2;   // ADPCM
        throws<std::invalid_argument>([&] { wav::read_wav(write_tmp("adpcm.wav", wav_bytes(w, std::string(8, '\0')))); },
                                      "a format tag it does not take");
    }
    for (const char* n : {"i16.wav", "u8.wav", "i24.wav", "i32.wav", "f32.wav", "st.wav", "r16.wav", "text.wav",
                          "adpcm.wav", "missing-dir"}) {
        std::error_code ec;
        std::filesystem::remove(std::filesystem::temp_directory_path() / (tmp_prefix() + n), ec);
    }
}

}  // namespace

int main() {
    try {
        test_rules();
        test_detect_stubbed();
        test_detect_joint_stubbed();
        test_receive_slot_stubbed();
        test_wavread();
        test_receive_end_to_end();
        test_ce_resid_ratio();
        test_rules_per_call();
        test_short_header();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FATAL: %s\n", e.what());
        return 1;
    }
    return check::report("qrss_receiver_slot");
}

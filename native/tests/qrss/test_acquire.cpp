// native/core/qrss/acquire against sstvae/qrss/acquire.py's arbitration and
// acquire(), with no Python at test time:
//
// - the arbitration on random hit sets (splitmix, the recipe below in both
//   languages): the indices kept by each rule and by the joint path's
//   composition, pinned by digest; a hand-made alias and band-owned pair;
// - test_detections_order's and test_joint_arbitration's scenarios
//   (tests/test_qrss_lacquire.py) replayed here, their expected values
//   copied from those tests -- the latter through AcquireSeams, which is
//   how the stages are replaced without Python;
// - acquire() on test_acquire_a12.cpp's capture (CE TINY, splitmix noise,
//   0.4 s early): one frame with A3 and hints, joint from the raw capture
//   (the C++ blanker and AGC), joint with A3 (the shape verdicts and the
//   L copies), L alone; the candidates detections() makes of them. A1's
//   lines and the hints are pinned to the bit, A2's hits to wave 5's bar
//   (f_hz 1e-7 Hz, tau0_s 1e-6 s), A3's Gumbel p to 4e-16;
// - the same acquisition from two threads at once, and the errors.
// tests/test_qrss_native_acquire.py compares against Python directly.
// The pins (a scratch script; values printed with float.hex / repr):
//
//   x = capture(0.0, 1201); t0 = 12.0 * FE_FS + 0.4 * FE_FS   # test_acquire_a12.cpp's
//   cap = frontend.Capture(x, 0, t0); JT = (frame.TINY, lframe.L_TINY)
//   A.acquire(cap, frame.TINY, preprocessed=True, hints=(1500.3, 2000.0, 1700.0))
//   A.acquire(cap, JT, live_only=True, hints=(1500.0, (1600.0, 1), (1700.0, 0), (1800.0, 9)))
//   A.acquire(cap, JT, preprocessed=True)
//   A.acquire(cap, (lframe.L_TINY,), preprocessed=True, live_only=True, hints=(1500.0,))
//   ... .detections(frame.TINY), .detections(JT), .detections((lframe.L_TINY,))
//   A.live_end_s().hex()
//
//   def hits_of(t):                  # the random sets, as random_hits() below
//       u = splitmix(9000 + t, 64); n = int(u[0] % 12); out = []
//       for i in range(n):
//           a0 = int(u[1 + 5 * i]); w = a0 & 1; kind = (a0 >> 1) % 5
//           sign = 1 if (a0 >> 4) & 1 else -1; k = 1 + ((a0 >> 5) & 1)
//           tauc = (a0 >> 6) & 1; lamc = (a0 >> 7) % 3; pc = (a0 >> 9) % 3
//           tau_base = 0.3 if (a0 >> 11) & 1 else 0.0
//           u1, u2, u3, u4 = (unit(u[2 + 5 * i + j]) for j in range(4))
//           off = [(u1 - 0.5) * 12.0, (sign * k) / T_SYM + (u1 - 0.5) * 1.4,
//                  (sign * k) / L_T_SYM + (u1 - 0.5) * 1.4, -(2.0 + 22.0 * u1),
//                  (-22.1, -3.1, 5.0, -5.0)[(a0 >> 12) % 4]][kind]
//           tau = tau_base + ((u2 - 0.5) * 0.4 if tauc else 0.0)
//           lam = (100.0, 30.0, 5.0 + 595.0 * u3)[lamc]; p = (1e-10, 0.0, u4 * 1e-6)[pc]
//           out.append(A.PreambleHit(f_hz=1000.0 + off, drift_hz_per_min=0.0, tau0_s=tau, phase=0.0,
//                                    lam=lam, cell_lam=lam, p_value=p, passed=True, waveform=w))
//       return out
//   idx = []                         # for t in range(200): the kept indices of
//   # drop_symbol_rate_aliases(hits), (hits, L_T_SYM), drop_band_owned(hits), merge_hits(hits),
//   # each followed by -1, then of merge_hits(drop_band_owned(per-waveform aliases)), then -2
//   hashlib.sha256(np.array(idx, np.int64).tobytes()).hexdigest()

#include <algorithm>
#include <atomic>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <map>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

#include "infra_testgen.hpp"
#include "math/errors.hpp"
#include "qrss/acquire.hpp"
#include "qrss/ce.hpp"
#include "qrss/frame.hpp"
#include "qrss_check.hpp"

using namespace sstvae;
namespace acq = sstvae::qrss::acquire;
namespace types = sstvae::qrss::types;
namespace rxa = sstvae::qrss::rxconfig::acquire;
namespace config = sstvae::qrss::config;
namespace frame = sstvae::qrss::frame;
using cf = std::complex<float>;
using cd = std::complex<double>;

constexpr double FE = 4000.0;
constexpr std::int64_t CE = config::WAVEFORM_CE;
constexpr std::int64_t L = config::WAVEFORM_L;

static std::vector<cf> capture(double lead_in_s, std::uint64_t seed) {
    const auto n = static_cast<std::int64_t>(40 * FE);
    const auto& spec = frame::get("tiny");
    const std::vector<double> zeros(static_cast<std::size_t>(spec.n_data()), 0.0);
    const std::vector<double> sym = frame::assemble(spec, nullptr, zeros);
    qrss::ce::BasebandOptions opt;
    opt.lead_in_s = lead_in_s;
    const std::vector<cd> z = qrss::ce::baseband(sym, spec, FE, -12.0, n, opt);
    const auto un = static_cast<std::size_t>(n);
    const auto u = qrss_testgen::splitmix(seed, 2 * un);
    std::vector<cf> x(un);
    for (std::size_t i = 0; i < un; ++i) {
        const double nr = (qrss_testgen::unit(u[i]) - 0.5) * 9.8;
        const double ni = (qrss_testgen::unit(u[un + i]) - 0.5) * 9.8;
        const cd v = z[i] + cd(nr, ni);
        x[i] = cf(static_cast<float>(v.real()), static_cast<float>(v.imag()));
    }
    return x;
}

template <typename E, typename F>
static void throws(F f, const std::string& text, const std::string& what) {
    try {
        f();
    } catch (const E& e) {
        check::equal(std::string(e.what()), text, what);
        return;
    } catch (const std::exception& e) {
        check::fail(what, std::string("wrong exception: ") + e.what());
        return;
    }
    check::fail(what, "did not throw");
}

static void near(double got, double want, double tol, const std::string& what) {
    check::close(std::vector<double>{got}, std::vector<double>{want}, tol, what);
}

// test_qrss_lacquire._hit
static acq::PreambleHit hit(double f, double tau, double lam, double p, std::int64_t w) {
    acq::PreambleHit h;
    h.f_hz = f;
    h.tau0_s = tau;
    h.lam = lam;
    h.cell_lam = lam;
    h.p_value = p;
    h.passed = p < rxa::A2_PFA;
    h.waveform = w;
    return h;
}

static types::CarrierCandidate line(double f, std::optional<std::int64_t> w = std::nullopt) {
    return types::CarrierCandidate{f, 0.0, 0.0, w};
}

// (method, waveform, rounded f) of each candidate
using Order = std::vector<std::tuple<std::string, std::int64_t, long>>;
static Order order(const std::vector<types::Detection>& ds) {
    Order out;
    for (const auto& d : ds) out.emplace_back(d.method, d.waveform, std::lround(d.f_hz));
    return out;
}
static void order_is(const std::vector<types::Detection>& ds, const Order& want, const std::string& what) {
    const Order got = order(ds);
    check::equal(got.size(), want.size(), what + " count");
    for (std::size_t i = 0; i < std::min(got.size(), want.size()); ++i) {
        check::equal(std::get<0>(got[i]), std::get<0>(want[i]), what + " method " + std::to_string(i));
        check::equal(std::get<1>(got[i]), std::get<1>(want[i]), what + " waveform " + std::to_string(i));
        check::equal(std::get<2>(got[i]), std::get<2>(want[i]), what + " f " + std::to_string(i));
    }
}

// --- small functions ------------------------------------------------------------------------

static void test_small() {
    check::equal(acq::live_end_s(), 0x1.c819999999999p+4, "live_end_s");
    const frame::FrameSpec tiny = frame::get("tiny"), short_ = frame::get("short");
    const frame::FrameSpec ltiny = frame::get_l("l-tiny");
    const std::vector<frame::FrameSpec> jt{tiny, ltiny};
    const acq::FrameMap m = acq::frame_map_of(jt);
    check::equal(m.size(), std::size_t(2), "frame map");
    check::is_true(m[0].first == CE && m[1].first == L && m[1].second.name() == "l-tiny", "frame map order");
    const std::vector<frame::FrameSpec> lt{ltiny, tiny};
    check::is_true(acq::frame_map_of(lt)[0].first == L, "frame map keeps the frames' order");
    const std::vector<frame::FrameSpec> dup{tiny, short_}, none;
    throws<std::invalid_argument>([&] { (void)acq::frame_map_of(dup); },
                                  "acquire takes at most one frame per waveform, not 'tiny' and 'short'",
                                  "frame_map, two CE frames");
    throws<std::invalid_argument>([&] { (void)acq::frame_map_of(none); }, "acquire needs at least one frame",
                                  "frame_map, none");

    // _hint_items((1500, (1400.0, 1), (1300, 5)), {0: .., 1: ..}) == [(1500.0, [0, 1]), (1400.0, [1]), (1300.0, [])]
    const std::vector<acq::Hint> hints{{1500.0, std::nullopt}, {1400.0, L}, {1300.0, 5}};
    const auto items = acq::hint_items(hints, m);
    check::equal(items.size(), std::size_t(3), "hint_items count");
    check::is_true(items[0].first == 1500.0 && items[0].second == std::vector<std::int64_t>{CE, L}, "bare hint");
    check::is_true(items[1].first == 1400.0 && items[1].second == std::vector<std::int64_t>{L}, "L hint");
    check::is_true(items[2].first == 1300.0 && items[2].second.empty(), "unknown waveform hint");
    const std::vector<std::int64_t> only_l{L};
    check::is_true(acq::hint_items(hints, only_l)[0].second == only_l, "bare hint, L alone");
}

// --- arbitration ------------------------------------------------------------------------------

static std::vector<acq::PreambleHit> random_hits(int t) {
    const auto u = qrss_testgen::splitmix(static_cast<std::uint64_t>(9000 + t), 64);
    const auto n = static_cast<int>(u[0] % 12);
    std::vector<acq::PreambleHit> out;
    for (int i = 0; i < n; ++i) {
        const std::uint64_t a0 = u[static_cast<std::size_t>(1 + 5 * i)];
        const auto w = static_cast<std::int64_t>(a0 & 1);
        const auto kind = (a0 >> 1) % 5;
        const int sign = ((a0 >> 4) & 1) ? 1 : -1;
        const int k = 1 + static_cast<int>((a0 >> 5) & 1);
        const bool tauc = (a0 >> 6) & 1;
        const auto lamc = (a0 >> 7) % 3, pc = (a0 >> 9) % 3;
        const double tau_base = ((a0 >> 11) & 1) ? 0.3 : 0.0;
        double uu[4];
        for (int j = 0; j < 4; ++j) uu[j] = qrss_testgen::unit(u[static_cast<std::size_t>(2 + 5 * i + j)]);
        double off = 0.0;
        if (kind == 0)
            off = (uu[0] - 0.5) * 12.0;
        else if (kind == 1)
            off = static_cast<double>(sign * k) / config::T_SYM + (uu[0] - 0.5) * 1.4;
        else if (kind == 2)
            off = static_cast<double>(sign * k) / config::L_T_SYM + (uu[0] - 0.5) * 1.4;
        else if (kind == 3)
            off = -(2.0 + 22.0 * uu[0]);
        else
            off = std::vector<double>{-22.1, -3.1, 5.0, -5.0}[(a0 >> 12) % 4];
        const double tau = tau_base + (tauc ? (uu[1] - 0.5) * 0.4 : 0.0);
        const double lam = lamc == 0 ? 100.0 : lamc == 1 ? 30.0 : 5.0 + 595.0 * uu[2];
        const double p = pc == 0 ? 1e-10 : pc == 1 ? 0.0 : uu[3] * 1e-6;
        acq::PreambleHit h = hit(1000.0 + off, tau, lam, p, w);
        h.passed = true;
        out.push_back(h);
    }
    return out;
}

static void test_arbitration() {
    std::vector<std::int64_t> idx;
    std::vector<int> drops(5, 0);
    for (int t = 0; t < 200; ++t) {
        const std::vector<acq::PreambleHit> hits = random_hits(t);
        const std::vector<std::vector<std::size_t>> kept{
            acq::drop_symbol_rate_aliases_idx(hits), acq::drop_symbol_rate_aliases_idx(hits, config::L_T_SYM),
            acq::drop_band_owned_idx(hits), acq::merge_hits_idx(hits)};
        for (std::size_t j = 0; j < kept.size(); ++j) {
            for (const std::size_t i : kept[j]) idx.push_back(static_cast<std::int64_t>(i));
            idx.push_back(-1);
            drops[j] += static_cast<int>(hits.size() - kept[j].size());
        }
        // the joint path's composition, on indices into `hits`
        std::vector<std::size_t> per;
        for (const auto& [w, ts] : {std::pair{CE, config::T_SYM}, std::pair{L, config::L_T_SYM}}) {
            std::vector<std::size_t> of_w;
            std::vector<acq::PreambleHit> sub;
            for (std::size_t i = 0; i < hits.size(); ++i)
                if (hits[i].waveform == w) {
                    of_w.push_back(i);
                    sub.push_back(hits[i]);
                }
            for (const std::size_t i : acq::drop_symbol_rate_aliases_idx(sub, ts)) per.push_back(of_w[i]);
        }
        std::vector<acq::PreambleHit> ph;
        for (const std::size_t i : per) ph.push_back(hits[i]);
        const auto owned = acq::drop_band_owned_idx(ph);
        std::vector<acq::PreambleHit> oh;
        for (const std::size_t i : owned) oh.push_back(ph[i]);
        const auto merged = acq::merge_hits_idx(oh);
        for (const std::size_t i : merged) idx.push_back(static_cast<std::int64_t>(per[owned[i]]));
        idx.push_back(-2);
        drops[4] += static_cast<int>(hits.size() - merged.size());
        // the hit-returning forms are the indexed ones
        const auto mh = acq::merge_hits(hits);
        const auto mi = acq::merge_hits_idx(hits);
        bool same = mh.size() == mi.size();
        for (std::size_t i = 0; same && i < mh.size(); ++i) same = mh[i].f_hz == hits[mi[i]].f_hz;
        check::is_true(same, "merge_hits = merge_hits_idx " + std::to_string(t));
    }
    check::equal(idx.size(), std::size_t(6249), "random sets: indices");
    qrss_check::digest_is(idx, "9735b04a5b0491d7d3f08110f222a07546d7a9ddb0c6d0762ab99fcc64c180c9",
                          "random sets: the hits each rule keeps, in order");
    check::is_true(drops == std::vector<int>{7, 15, 72, 87, 145}, "random sets: drops per rule");

    // test_qrss_lacquire's pairs: L's -1/T_L alias 0.8 T_L off the pilot's timing is
    // band-owned (not an alias at CE's T); a CE hit beside it is not
    const acq::PreambleHit main = hit(1500.0, 0.30, 200.0, 1e-90, L);
    const acq::PreambleHit alias = hit(1500.0 - 1.0 / config::L_T_SYM, 0.30 + 0.8 * config::L_T_SYM, 40.0, 1e-12, L);
    const std::vector<acq::PreambleHit> pair{alias, main};
    check::is_true(acq::drop_band_owned_idx(pair) == std::vector<std::size_t>{1}, "band-owned alias dropped");
    check::equal(acq::drop_symbol_rate_aliases_idx(pair, config::L_T_SYM).size(), std::size_t(2),
                 "0.8 T_L off: not an alias");
    const std::vector<acq::PreambleHit> ce_pair{hit(1490.0, 0.30, 40.0, 1e-12, CE), main};
    check::equal(acq::drop_band_owned_idx(ce_pair).size(), std::size_t(2), "a CE hit in the band stays");
    // merge: the smaller p wins whatever the Lambda; strongest first after
    const std::vector<acq::PreambleHit> m{hit(1502.0, 0.31, 60.0, 1e-12, CE), hit(1500.0, 0.30, 50.0, 1e-30, L),
                                          hit(1300.0, 0.0, 80.0, 1e-8, L)};
    check::is_true(acq::merge_hits_idx(m) == std::vector<std::size_t>{2, 1}, "merge_hits");
    check::is_true(acq::merge_hits_idx(std::vector<acq::PreambleHit>{}).empty(), "merge_hits of nothing");
}

// --- detections -----------------------------------------------------------------------------

static acq::TbdHit tbd(double f, std::optional<std::int64_t> w, std::optional<double> l_f = std::nullopt) {
    acq::TbdHit h;
    h.path = types::FreqPath{{0.0, 100.0}, {f, f}, {1.0, 1.0}};
    if (l_f) h.l_path = types::FreqPath{{0.0, 100.0}, {*l_f, *l_f}, {1.0, 1.0}};
    h.z = 50.0;
    h.p_value = 1e-9;
    h.n_frames = 31;
    h.centre_hz = f;
    h.waveform = w;
    return h;
}

static void test_detections_order() {
    // test_qrss_lacquire.test_detections_order
    acq::Acquisition a;
    a.preamble = {hit(1000.0, 0.0, 50.0, 1e-12, CE), hit(1100.0, 0.0, 80.0, 1e-20, L)};
    a.tbd = {tbd(1200.0, CE), tbd(1300.0, std::nullopt, 1306.0), tbd(1400.0, L), tbd(1385.0, std::nullopt, 1404.0)};
    a.hinted = {line(1500.0, L)};
    const frame::FrameSpec tiny = frame::get("tiny"), ltiny = frame::get_l("l-tiny");
    const std::vector<frame::FrameSpec> jt{tiny, ltiny}, ce_only{tiny}, l_only{ltiny};
    order_is(a.detections(jt),
             {{"preamble", L, 1100}, {"preamble", CE, 1000}, {"hint", L, 1500}, {"tbd", CE, 1200},
              {"tbd", CE, 1300}, {"tbd", L, 1400}, {"tbd", CE, 1385}, {"tbd", L, 1306}},
             "joint");
    order_is(a.detections(ce_only),
             {{"preamble", CE, 1000}, {"tbd", CE, 1200}, {"tbd", CE, 1300}, {"tbd", CE, 1385}, {"tbd", CE, 1400}},
             "CE alone");
    order_is(a.detections(l_only),
             {{"preamble", L, 1100}, {"hint", L, 1500}, {"tbd", L, 1306}, {"tbd", L, 1400}, {"tbd", L, 1200}},
             "L alone");
    const auto dets = a.detections(jt);
    check::is_true(dets[0].timing.has_value(), "a preamble detection has timing");
    near(dets[0].path.t_s.back(), static_cast<double>(ltiny.keyed_end_pos()) * config::L_T_SYM, 1e-12,
         "an L preamble detection's path spans L's frame");
    const auto single = a.detections(tiny);
    std::vector<std::string> methods;
    for (const auto& d : single) methods.push_back(d.method);
    check::is_true(methods == std::vector<std::string>{"preamble", "preamble", "hint", "tbd", "tbd", "tbd", "tbd"},
                   "one frame: preamble as stored, hints, A3");
    // the same through the frame map
    order_is(acq::joint_detections(a, acq::frame_map_of(jt)), order(a.detections(jt)), "joint_detections");
    // test_frame_tuples: a hint on L's frame alone is CE's, in a tuple L's
    acq::Acquisition h;
    h.hinted = {types::CarrierCandidate{1500.0, 0.0, 1.0, std::nullopt}};
    const auto one = h.detections(ltiny);
    check::is_true(one.size() == 1 && one[0].waveform == CE, "hint, one L frame: CE");
    check::equal(one[0].path.t_s.back(), static_cast<double>(ltiny.keyed_end_pos()) * config::L_T_SYM,
                 "hint, one L frame: L's span");
    check::is_true(h.detections(l_only).at(0).waveform == L, "hint, (L,): L");
    const std::vector<frame::FrameSpec> two_ce{tiny, frame::get("short")};
    throws<std::invalid_argument>([&] { (void)a.detections(two_ce); },
                                  "acquire takes at most one frame per waveform, not 'tiny' and 'short'",
                                  "detections, two CE frames");
}

// test_qrss_lacquire.test_joint_arbitration, the stages replaced through the seams
static void test_joint_arbitration() {
    const std::map<std::pair<std::int64_t, double>, std::tuple<double, double, double>> passed{
        {{CE, 1000.0}, {500.0, 1e-200, 0.30}}, {{CE, 1033.0}, {60.0, 1e-9, 0.30}},
        {{L, 1000.0}, {20.0, 1e-8, 0.30}},     {{L, 1500.0}, {200.0, 1e-90, 0.70}},
        {{L, 1483.5}, {40.0, 1e-12, 0.75}},    {{L, 1516.5}, {30.0, 1e-10, 0.70}}};
    std::vector<std::pair<std::int64_t, double>> calls;
    std::map<double, double> leads;
    auto stub = [&](const types::CarrierCandidate& c, const acq::PreambleKind* k) {
        const std::int64_t w = k ? k->waveform : CE;
        calls.emplace_back(w, c.f_hz);
        auto it = passed.find({w, c.f_hz});
        const auto [lam, p, tau] = it == passed.end() ? std::tuple{5.0, 0.3, 0.0} : it->second;
        return hit(c.f_hz, tau, lam, p, w);
    };
    acq::AcquireSeams s;
    s.carrier_lines = [](const acq::FeView&, double, double, const acq::CarrierStats&) {
        return std::vector<types::CarrierCandidate>{line(1000.0), line(1033.0), line(1483.5), line(1516.5)};
    };
    s.pilot_lines = [](const acq::CarrierStats&, std::span<const types::CarrierCandidate>) {
        return std::vector<types::CarrierCandidate>{line(1500.0)};
    };
    s.line_search = [&](const acq::FeView&, double, const types::CarrierCandidate& c, double,
                        std::span<const acq::PreambleKind* const> kinds) {
        std::vector<acq::PreambleHit> out;
        for (const acq::PreambleKind* k : kinds) out.push_back(stub(c, k));
        return out;
    };
    s.search = [&](const acq::FeView&, double, const types::CarrierCandidate& c, double, const acq::PreambleKind* k) {
        return stub(c, k);
    };
    s.lead_in_span = [&](const acq::FeView&, double, double f, double, double, double, std::optional<double> t_sym) {
        leads[std::round(f * 10.0) / 10.0] = t_sym.value_or(config::T_SYM);
        return 0.0;
    };
    const std::vector<cf> zeros(static_cast<std::size_t>(40 * FE));
    const double t0 = 12.0 * FE;
    acq::AcquireOptions opt;
    opt.live_only = true;
    opt.preprocessed = true;
    const frame::FrameSpec tiny = frame::get("tiny"), ltiny = frame::get_l("l-tiny");
    const std::vector<frame::FrameSpec> jt{tiny, ltiny}, ce_only{tiny}, l_only{ltiny};

    auto pre = [](const acq::Acquisition& a) {
        std::vector<std::pair<std::int64_t, double>> out;
        for (const auto& h : a.preamble) out.emplace_back(h.waveform, h.f_hz);
        return out;
    };
    auto freqs = [&](std::int64_t w) {
        std::vector<double> out;
        for (const auto& [cw, f] : calls)
            if (cw == w && std::find(out.begin(), out.end(), f) == out.end()) out.push_back(f);
        std::sort(out.begin(), out.end());
        return out;
    };
    acq::Acquisition a = acq::acquire(zeros, t0, FE, jt, opt, s);
    check::is_true(pre(a) == std::vector<std::pair<std::int64_t, double>>{{CE, 1000.0}, {L, 1500.0}},
                   "joint arbitration: CE 1000 and L 1500");
    check::is_true(freqs(CE) == std::vector<double>{1000.0, 1033.0, 1483.5, 1516.5}, "CE's A2 at the lines");
    check::is_true(freqs(L) == std::vector<double>{1000.0, 1033.0, 1483.5, 1500.0, 1516.5},
                   "L's A2 at the lines and the pilot");
    check::is_true(leads == std::map<double, double>{{1000.0, config::T_SYM}, {1500.0, config::L_T_SYM}},
                   "each lead-in at its waveform's symbol");
    check::is_true(a.pilots.size() == 1 && a.pilots[0].f_hz == 1500.0, "pilots");
    calls.clear();
    a = acq::acquire(zeros, t0, FE, ce_only, opt, s);
    check::is_true(pre(a) == std::vector<std::pair<std::int64_t, double>>{{CE, 1000.0}}, "(TINY,): CE 1000");
    check::is_true(freqs(L).empty() && a.pilots.empty(), "(TINY,): no L search, no pilots");
    calls.clear();
    a = acq::acquire(zeros, t0, FE, l_only, opt, s);
    check::is_true(pre(a) == std::vector<std::pair<std::int64_t, double>>{{L, 1500.0}, {L, 1000.0}},
                   "(L_TINY,): L 1500 then L 1000");
    check::is_true(freqs(CE).empty(), "(L_TINY,): no CE search");
    // the same through acquire_joint and a Capture
    const acq::Acquisition b = acq::acquire_joint(zeros, t0, FE, acq::frame_map_of(l_only), opt, s);
    check::is_true(pre(b) == pre(a), "acquire_joint");
    qrss::frontend::Capture cap{zeros, 0, t0, 4000};
    check::is_true(pre(acq::acquire(cap, l_only, opt, s)) == pre(a), "acquire(Capture)");
}

// --- acquire() on a capture ---------------------------------------------------------------------

static void line_is(const types::CarrierCandidate& c, double f, double metric, std::optional<std::int64_t> w,
                    const std::string& what) {
    check::equal(c.f_hz, f, what + " f_hz");
    check::equal(c.drift_hz_per_min, 0.0, what + " drift");
    check::equal(c.metric, metric, what + " metric");
    check::is_true(c.waveform == w, what + " waveform");
}

static void hit_near(const acq::PreambleHit& h, double f, double tau0, double lam, double p, std::int64_t w,
                     const std::string& what) {
    near(h.f_hz, f, 1e-7, what + " f_hz");
    near(h.tau0_s, tau0, 1e-6, what + " tau0_s");
    near(h.lam, lam, 1e-9 * lam, what + " lam");
    near(h.p_value, p, 1e-6 * p, what + " p_value");
    check::equal(h.lead_in_s, 0.0, what + " lead_in_s");
    check::equal(h.waveform, w, what + " waveform");
    check::equal(h.n_groups, std::int64_t(2), what + " n_groups");
    check::is_true(h.passed, what + " passed");
}

static void tbd_is(const acq::TbdHit& h, double centre, double z, double p, std::optional<std::int64_t> w,
                   bool l_path, const std::string& what) {
    check::equal(h.centre_hz, centre, what + " centre_hz");
    check::equal(h.z, z, what + " z");
    near(h.p_value, p, 4e-16 * p, what + " p_value");
    check::equal(h.n_frames, std::int64_t(17), what + " n_frames");
    check::is_true(h.waveform == w, what + " waveform");
    check::equal(h.l_path.has_value(), l_path, what + " l_path");
}

// a candidate: method, waveform, f_hz (A3's polyfit line to 1e-12, a preamble hit's to 1e-7),
// the path's end, A3's z
static void det_is(const types::Detection& d, const std::string& method, std::int64_t w, double f, double t_end,
                   double a3_z, const std::string& what) {
    check::equal(d.method, method, what + " method");
    check::equal(d.waveform, w, what + " waveform");
    const double tol = method == "preamble" ? 1e-7 : method == "tbd" ? 1e-12 * f : 0.0;
    near(d.f_hz, f, tol, what + " f_hz");
    check::equal(d.path.t_s.back(), t_end, what + " path end");
    check::equal(d.timing.has_value(), method == "preamble", what + " timing");
    check::is_true(std::isnan(a3_z) ? std::isnan(d.a3_z) : d.a3_z == a3_z, what + " a3_z");
    check::equal(d.lead_in_s, 0.0, what + " lead_in_s");
}

static void test_acquire(const std::vector<cf>& x, double t0) {
    const frame::FrameSpec tiny = frame::get("tiny"), ltiny = frame::get_l("l-tiny");
    const std::vector<frame::FrameSpec> jt{tiny, ltiny}, l_only{ltiny};
    const double nan = std::nan("");
    const double ce_end = 0x1.aaccccccccccdp+5, l_end = 0x1.acbd70a3d70a4p+5, a3_end = 0x1.8000000000000p+4;
    const double z0 = 0x1.854da4c3a9316p+9, z1 = 0x1.218b8d97d63c2p+4, z2 = 0x1.00c6c2ee97dfcp+4;

    // one frame, A3, hints: 1500.3 beside the passed hit (skipped), 2000 and 1700 hinted
    acq::AcquireOptions opt;
    opt.preprocessed = true;
    opt.hints = {{1500.3, std::nullopt}, {2000.0, std::nullopt}, {1700.0, std::nullopt}};
    const acq::Acquisition a1 = acq::acquire(x, t0, FE, tiny, opt);
    check::equal(a1.lines.size(), std::size_t(7), "single: lines");
    check::equal(a1.lines.at(0).f_hz, 0x1.76ffbdffaf800p+10, "single: line 0");
    check::equal(a1.lines.at(1).f_hz, 0x1.79d0084e3f700p+10, "single: line 1");
    check::is_true(a1.pilots.empty(), "single: no pilots");
    check::equal(a1.hinted.size(), std::size_t(2), "single: hinted");
    line_is(a1.hinted.at(0), 2000.0, 0x1.1798900000000p+4, std::nullopt, "single: hinted 0");
    line_is(a1.hinted.at(1), 1700.0, 0x1.5cd07e0000000p+4, std::nullopt, "single: hinted 1");
    check::equal(a1.preamble.size(), std::size_t(1), "single: hits");
    hit_near(a1.preamble.at(0), 1500.0007352213397, -0.3999859723016677, 545.6062855769717, 5.43809922966975e-229,
             CE, "single: hit");
    check::equal(a1.tbd.size(), std::size_t(3), "single: A3 hits");
    tbd_is(a1.tbd.at(0), 0x1.742f99b58a800p+10, z0, 0.0, std::nullopt, false, "single: tbd 0");
    tbd_is(a1.tbd.at(1), 0x1.6a80000000000p+10, z1, 5.389005703930788e-07, std::nullopt, false, "single: tbd 1");
    tbd_is(a1.tbd.at(2), 0x1.8380000000000p+10, z2, 9.254418478854813e-06, std::nullopt, false, "single: tbd 2");
    {
        const auto d = a1.detections(tiny);
        check::equal(d.size(), std::size_t(6), "single: detections(TINY)");
        det_is(d.at(0), "preamble", CE, 1500.0007352213397, ce_end, nan, "single det 0");
        near(d.at(0).timing->tau0, -99.99649307541692, 1e-6 * 250, "single det 0 tau0");
        det_is(d.at(1), "hint", CE, 2000.0, ce_end, nan, "single det 1");
        det_is(d.at(2), "hint", CE, 1700.0, ce_end, nan, "single det 2");
        det_is(d.at(3), "tbd", CE, 1500.007034034647, a3_end, z0, "single det 3");
        det_is(d.at(4), "tbd", CE, 1489.8782220901874, a3_end, z1, "single det 4");
        det_is(d.at(5), "tbd", CE, 1510.54373955488, a3_end, z2, "single det 5");
        const auto j = a1.detections(jt);
        order_is(j,
                 {{"preamble", CE, 1500}, {"hint", CE, 2000}, {"hint", L, 2000}, {"hint", CE, 1700},
                  {"hint", L, 1700}, {"tbd", CE, 1500}, {"tbd", CE, 1490}, {"tbd", CE, 1511}, {"tbd", L, 1500},
                  {"tbd", L, 1490}, {"tbd", L, 1511}},
                 "single: detections(JT)");
        check::equal(j.at(2).path.t_s.back(), l_end, "single: an L hint's span");
    }

    // joint, from the raw capture (the blanker and AGC), live, hints of each kind
    acq::AcquireOptions jopt;
    jopt.live_only = true;
    jopt.hints = {{1500.0, std::nullopt}, {1600.0, L}, {1700.0, CE}, {1800.0, 9}};
    const acq::Acquisition a2 = acq::acquire(x, t0, FE, jt, jopt);
    check::equal(a2.lines.size(), std::size_t(7), "raw joint: lines");
    check::equal(a2.lines.at(0).f_hz, 0x1.76ffbc21ae000p+10, "raw joint: line 0");
    check::equal(a2.lines.at(1).f_hz, 0x1.79d03d905b000p+10, "raw joint: line 1");
    check::equal(a2.pilots.size(), std::size_t(5), "raw joint: pilots");
    check::equal(a2.pilots.at(0).f_hz, 0x1.7abebcd1ea000p+10, "raw joint: pilot 0");
    check::equal(a2.pilots.at(1).f_hz, 0x1.f6fc658544000p+10, "raw joint: pilot 1");
    check::equal(a2.hinted.size(), std::size_t(3), "raw joint: hinted");
    line_is(a2.hinted.at(0), 0x1.76ffbc21ae000p+10, 0x1.0330b40000000p+11, L, "raw joint: hinted 0 (refined)");
    line_is(a2.hinted.at(1), 1600.0, 0x1.4bb06a0000000p+4, L, "raw joint: hinted 1");
    line_is(a2.hinted.at(2), 1700.0, 0x1.5db2680000000p+4, CE, "raw joint: hinted 2");
    check::equal(a2.preamble.size(), std::size_t(1), "raw joint: hits");
    hit_near(a2.preamble.at(0), 1500.0006955771312, -0.39997935362842346, 546.1351575944603, 4.25987315430916e-229,
             CE, "raw joint: hit");
    check::is_true(a2.tbd.empty(), "raw joint: live, no A3");
    order_is(a2.detections(jt), {{"preamble", CE, 1500}, {"hint", L, 1500}, {"hint", L, 1600}, {"hint", CE, 1700}},
             "raw joint: detections");

    // joint, A3 with the shape verdicts
    acq::AcquireOptions popt;
    popt.preprocessed = true;
    const acq::Acquisition a3 = acq::acquire(x, t0, FE, jt, popt);
    check::equal(a3.pilots.size(), std::size_t(5), "joint A3: pilots");
    check::equal(a3.pilots.at(0).f_hz, 0x1.7abf037eab000p+10, "joint A3: pilot 0");
    check::equal(a3.preamble.size(), std::size_t(1), "joint A3: hits");
    check::is_true(a3.hinted.empty(), "joint A3: no hints");
    check::equal(a3.tbd.size(), std::size_t(3), "joint A3: A3 hits");
    tbd_is(a3.tbd.at(0), 0x1.742f99b58a800p+10, z0, 0.0, CE, true, "joint A3: tbd 0");
    tbd_is(a3.tbd.at(1), 0x1.6a80000000000p+10, z1, 5.389005703930788e-07, CE, true, "joint A3: tbd 1");
    tbd_is(a3.tbd.at(2), 0x1.8380000000000p+10, z2, 9.254418478854813e-06, L, true, "joint A3: tbd 2");
    {
        const auto d = a3.detections(jt);
        check::equal(d.size(), std::size_t(4), "joint A3: detections");
        det_is(d.at(0), "preamble", CE, 1500.0007352213397, ce_end, nan, "joint det 0");
        det_is(d.at(1), "tbd", CE, 1500.007034034647, a3_end, z0, "joint det 1");
        det_is(d.at(2), "tbd", CE, 1489.8782220901874, a3_end, z1, "joint det 2");
        det_is(d.at(3), "tbd", L, 1507.3568545271803, a3_end, z2, "joint det 3 (L, on its pilot)");
        const auto dl = a3.detections(l_only);
        check::equal(dl.size(), std::size_t(1), "joint A3: detections((L_TINY,)), one gate per pilot");
        det_is(dl.at(0), "tbd", L, 1507.3568545271803, a3_end, z2, "L det 0");
    }

    // L alone, live: a CE signal gives no L hit; the hint is A1's cell, refined
    acq::AcquireOptions lopt;
    lopt.preprocessed = true;
    lopt.live_only = true;
    lopt.hints = {{1500.0, std::nullopt}};
    const acq::Acquisition a4 = acq::acquire(x, t0, FE, l_only, lopt);
    check::is_true(a4.preamble.empty() && a4.tbd.empty(), "L alone: nothing passed");
    check::equal(a4.pilots.size(), std::size_t(5), "L alone: pilots");
    check::equal(a4.hinted.size(), std::size_t(1), "L alone: hinted");
    line_is(a4.hinted.at(0), 0x1.76ffbdffaf800p+10, 0x1.f816200000000p+10, L, "L alone: hinted 0");

    // the seams, filled with the native stages, change nothing
    acq::AcquireSeams s;
    std::atomic<int> n_lines{0}, n_search{0};
    s.carrier_lines = [&](const acq::FeView& fe, double t, double fs, const acq::CarrierStats& cs) {
        ++n_lines;
        (void)fe;
        (void)t;
        (void)fs;
        return acq::carrier_lines(cs);
    };
    s.search = [&](const acq::FeView& fe, double t, const types::CarrierCandidate& c, double fs,
                   const acq::PreambleKind* k) {
        ++n_search;
        return std::visit([&](auto v) { return acq::preamble_search(v, t, c, fs, rxa::A2_PFA, k ? *k : acq::ce_pre()); },
                          fe);
    };
    const acq::Acquisition a5 = acq::acquire(x, t0, FE, jt, jopt, s);
    check::is_true(n_lines == 1 && n_search >= 5, "seams called");
    check::is_true(a5.lines == a2.lines && a5.hinted == a2.hinted && a5.pilots == a2.pilots, "seams: same lines");
    check::is_true(a5.preamble.size() == 1 && a5.preamble[0].tau0_s == a2.preamble[0].tau0_s &&
                       a5.preamble[0].p_value == a2.preamble[0].p_value,
                   "seams: same hit");
}

static bool same(const acq::Acquisition& a, const acq::Acquisition& b) {
    if (!(a.lines == b.lines && a.hinted == b.hinted && a.pilots == b.pilots)) return false;
    if (a.preamble.size() != b.preamble.size() || a.tbd.size() != b.tbd.size()) return false;
    for (std::size_t i = 0; i < a.preamble.size(); ++i) {
        const auto &x = a.preamble[i], &y = b.preamble[i];
        if (x.f_hz != y.f_hz || x.tau0_s != y.tau0_s || x.lam != y.lam || x.p_value != y.p_value ||
            x.lead_in_s != y.lead_in_s || x.waveform != y.waveform)
            return false;
    }
    for (std::size_t i = 0; i < a.tbd.size(); ++i)
        if (!(a.tbd[i].path == b.tbd[i].path) || a.tbd[i].p_value != b.tbd[i].p_value) return false;
    return true;
}

static void test_threads(const std::vector<cf>& x, double t0) {
    const frame::FrameSpec tiny = frame::get("tiny"), ltiny = frame::get_l("l-tiny");
    const std::vector<frame::FrameSpec> jt{tiny, ltiny};
    acq::AcquireOptions opt;
    opt.preprocessed = true;
    opt.hints = {{1600.0, L}, {1700.0, std::nullopt}};
    const acq::Acquisition want = acq::acquire(x, t0, FE, jt, opt);
    std::vector<int> bad(2, 0);
    std::vector<std::thread> pool;
    for (int t = 0; t < 2; ++t)
        pool.emplace_back([&, t] {
            if (!same(acq::acquire(x, t0, FE, jt, opt), want)) ++bad[static_cast<std::size_t>(t)];
        });
    for (auto& th : pool) th.join();
    check::equal(bad[0] + bad[1], 0, "the same acquisition from 2 threads");
}

static void test_errors(const std::vector<cf>& x, double t0) {
    const frame::FrameSpec tiny = frame::get("tiny");
    acq::AcquireOptions opt;
    opt.preprocessed = true;
    opt.live_only = true;
    opt.hints = {{1650.0, CE}};
    // with hits to compare it with, abs(h.f_hz - f); without, float(f)
    throws<math::TypeError>([&] { (void)acq::acquire(x, t0, FE, tiny, opt); },
                            "unsupported operand type(s) for -: 'float' and 'tuple'", "a pair hint, one frame");
    const std::vector<cf> zeros(static_cast<std::size_t>(40 * FE));
    throws<math::TypeError>([&] { (void)acq::acquire(zeros, 12.0 * FE, FE, tiny, opt); },
                            "float() argument must be a string or a real number, not 'tuple'",
                            "a pair hint, one frame, no hits");
    opt.hints = {{1500.0, std::nullopt}, {1650.0, CE}};
    throws<math::TypeError>([&] { (void)acq::acquire(zeros, 12.0 * FE, FE, tiny, opt); },
                            "float() argument must be a string or a real number, not 'tuple'",
                            "a pair hint after a bare one");
    const std::vector<frame::FrameSpec> two_l{frame::get_l("l-tiny"), frame::get_l("l-short")}, none;
    throws<std::invalid_argument>([&] { (void)acq::acquire(zeros, 12.0 * FE, FE, two_l, opt); },
                                  "acquire takes at most one frame per waveform, not 'l-tiny' and 'l-short'",
                                  "two L frames");
    throws<std::invalid_argument>([&] { (void)acq::acquire(zeros, 12.0 * FE, FE, none, opt); },
                                  "acquire needs at least one frame", "no frames");
}

int main() {
    try {
        const std::vector<cf> x = capture(0.0, 1201);
        qrss_check::digest_is(x, "23438ab54e64dcd00accc29f12831eadb789a38e4235fdcfc1ba56d1a9c05230", "the capture");
        const double t0 = 12.0 * FE + 0.4 * FE;
        test_small();
        test_arbitration();
        test_detections_order();
        test_joint_arbitration();
        test_acquire(x, t0);
        test_threads(x, t0);
        test_errors(x, t0);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FATAL: %s\n", e.what());
        return 1;
    }
    return check::report("qrss_acquire");
}

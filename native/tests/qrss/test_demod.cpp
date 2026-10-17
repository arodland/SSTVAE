// native/core/qrss/demod against sstvae/qrss/demod.py: the outputs of every
// function on a deterministic synthetic track (demod_synth.hpp) pinned
// from Python by SHA-256 of their bytes, the kappa combination pinned as
// doubles, and the reference's refusals. tests/test_qrss_native_demod.py
// compares against Python on tracker runs. The pins (synth as in
// demod_synth.hpp, Q = 123456789):
//
//   tr = synth(frame.get(name), rho); lay = frame.layout(tr.spec)
//   demod.symbol_estimates(tr, lay.data); demod.data_estimates(tr); demod.calibrate(tr)
//   demod.extract(None, tr, tr.spec, Q, est[, calibrate_kappa=False])
//   ldemod._combine([(1.1, np.float64(0.01)), (0.9, np.float64(0.02))]), ...

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "demod_synth.hpp"
#include "qrss/demod.hpp"
#include "qrss/frame.hpp"
#include "qrss_check.hpp"

using namespace sstvae;
namespace dm = sstvae::qrss::demod;
namespace fr = sstvae::qrss::frame;
using qrss_check::digest_is;

constexpr std::uint64_t Q = 123456789;

template <typename E, typename F>
static void throws(F f, const std::string& text, const std::string& what) {
    try {
        f();
    } catch (const E& e) {
        check::equal(std::string(e.what()), text, what);
        return;
    } catch (const std::exception& e) {
        check::fail(what, std::string("threw another type: ") + e.what());
        return;
    }
    check::fail(what, "did not throw");
}

static void test_tiny() {
    const auto& spec = fr::get("tiny");
    const auto s = demod_synth::make(spec);
    const auto v = s.view();
    const auto lay = fr::layout(spec);
    const auto se = dm::symbol_estimates(v, lay->data);
    digest_is(se.y, "ad6b8569dd71d399447dbb55f66ab42e12c7bd6c2273f95c29bacd2743738fb9", "tiny symbol y");
    digest_is(se.s2n, "e4ae0dee40541f5875646bba9911def5a448874e15b438d185d145aad012bf96", "tiny symbol s2n");
    const auto de = dm::data_estimates(v);
    digest_is(de.y, "ad6b8569dd71d399447dbb55f66ab42e12c7bd6c2273f95c29bacd2743738fb9", "tiny data y");
    check::equal(de.extra, 0.0, "tiny extra");
    const auto cal = dm::calibrate(v);
    check::equal(cal.kappa, 0.5, "tiny kappa (clamped)");
    check::is_true(cal.suspect, "tiny suspect");
    const auto j = dm::extract(v, spec, Q, "joint");
    digest_is(j.z, "e4c622306b9db28783db004d926fe47461a34fb4b50ca75ca7f1a862ffd79704", "tiny joint z");
    digest_is(j.w, "54b8b315337a68bf3bef213bf3a8f5600a8eaaec0dfd6b542d0ca3c1821913a1", "tiny joint w");
    check::equal(j.llr.size(), std::size_t{0}, "tiny: no header LLRs");
    digest_is(j.s2, "5a1a350ea2d3bffca080de78a657bdda1d81ca0f597dd7413a2572f3a88ea62b", "tiny joint s2");
    check::is_true(!j.llr_block && !j.llr_spread, "tiny: no LLR parts");
    const auto p = dm::extract(v, spec, Q, "plain");
    digest_is(p.z, "9b841236a9879299cf8246ac34674e9bf173f9286a2b4ddc7b25a34579452b06", "tiny plain z");
    digest_is(p.w, "12b002476e1a84489fdb9fc58008d6169f09537f6a461b6a3a9c02346720236f", "tiny plain w");
    check::equal(p.estimator, std::string("plain"), "estimator echoed");
    // suspect: scale 1, so the uncalibrated extract is the same
    const auto nc = dm::extract(v, spec, Q, "joint", false);
    digest_is(nc.z, "e4c622306b9db28783db004d926fe47461a34fb4b50ca75ca7f1a862ffd79704", "tiny nocal z");
}

static void test_short_spread() {
    const auto& spec = fr::get("short");
    const auto s = demod_synth::make(spec, 0.054);
    const auto v = s.view();
    const auto lay = fr::layout(spec);
    const auto se = dm::symbol_estimates(v, lay->data);
    digest_is(se.y, "0242972999b7a72d882c03eeaa0a6c3597fddf480309e07adf5eba647b0cfd5c", "short symbol y");
    digest_is(se.s2n, "c90cbdbe8040b5b0bfd6167270f91393832d4e5ea7a0582c830c160582246955", "short symbol s2n");
    const auto de = dm::data_estimates(v);
    digest_is(de.y, "9e8bf8f9949388a81ab87b5d2cd9526aebbf64a96741ebf624a3448847b366d1", "short data y (spread)");
    digest_is(de.s2n, "89a65367f328a91d0a8d038b29823a9339e4265a08d1f1285629e707efa20cd3", "short data s2n");
    check::equal(de.extra, 0.054, "short extra = rho");
    const auto j = dm::extract(v, spec, Q, "joint");
    digest_is(j.z, "8878d8b9321fb2b0059bca989bcf0d303cb1451ba3301896d0531b9899ff1509", "short joint z");
    digest_is(j.w, "03216a6de830f53dd29112b4a17a910270b671e7c12c2b8227aebb47ea05e619", "short joint w");
    digest_is(j.llr, "8cd93acff95089ee03ef4626def9923d84adc0be6a089b94df1fb9b676c6229c", "short llr");
    digest_is(j.s2, "40a746c49b3a80c11302de10226f6899be7b12c3372218c5c4feb9d9fa6d1cb7", "short s2");
    check::is_true(j.llr_block && j.llr_spread, "short: both LLR parts");
    if (j.llr_block)
        digest_is(*j.llr_block, "35e96c66ff5c20a96c198882a1b4c4ec994563d3a4f741a0537db14be2c84752", "llr_block");
    if (j.llr_spread)
        digest_is(*j.llr_spread, "7a22d3823bf81f71be9d9f7f327b85ff4354829305bc272e770eddbb6f29ad1f", "llr_spread");
    check::equal(j.kappa, 0.5, "short kappa");
    check::is_true(j.suspect, "short suspect");
    const auto p = dm::extract(v, spec, Q, "plain");
    digest_is(p.z, "7c3534e1aa153328a0b9252f3713bd79756c8a5ee8e553df81abf8e284626f5f", "short plain z");
    digest_is(p.w, "e97ba9a081a03632e11ad8b69e66930a616ab778a298f1481a58798de61c5686", "short plain w");
}

static void test_combine() {
    using E = std::vector<std::pair<double, double>>;
    auto c = dm::combine(E{{1.1, 0.01}, {0.9, 0.02}});
    check::equal(c.kappa, 0x1.084fcace213f4p+0, "combine two");
    check::is_true(!c.suspect, "combine two: not suspect");
    c = dm::combine(E{{1.3, 0.004}});
    check::equal(c.kappa, 0x1.4b972e5cb972ep+0, "combine one");
    c = dm::combine(E{{3.0, 0.01}, {2.5, 0.5}});
    check::equal(c.kappa, 2.0, "clamped high");
    check::is_true(c.suspect, "suspect high");
    c = dm::combine(E{{0.2, 0.001}});
    check::equal(c.kappa, 0.5, "clamped low");
    check::is_true(c.suspect, "suspect low");
    c = dm::combine(E{});
    check::is_true(c == dm::Calibration{1.0, false}, "no estimates");
}

static void test_refusals() {
    const auto& l = fr::get_l("l-short");
    dm::TrackView v;
    v.spec = &l;
    throws<fr::WrongWaveform>([&] { (void)dm::calibrate(v); },
                              "demod.calibrate is CE's; 'l-short' is waveform 1: use ldemod.calibrate_l",
                              "calibrate on L");
    throws<fr::WrongWaveform>([&] { (void)dm::extract(v, l, Q); },
                              "demod.extract is CE's; 'l-short' is waveform 1: use ldemod.extract_l",
                              "extract on L");
    const std::vector<double> y(10, 0.5), s2(10, 1.0), s9(9, 1.0);
    throws<std::invalid_argument>([&] { (void)dm::block_estimate(y, s2, Q, "bad"); },
                                  "estimator must be 'joint' or 'plain', not 'bad'", "bad estimator");
    throws<std::invalid_argument>([&] { (void)dm::block_estimate(y, s2, Q, "", "None"); },
                                  "estimator must be 'joint' or 'plain', not None", "shown estimator");
    throws<std::invalid_argument>([&] { (void)dm::block_estimate(y, s9, Q); },
                                  "operands could not be broadcast together with shapes (9,) () (10,) ",
                                  "s2 of another length");
    throws<std::invalid_argument>([&] { (void)dm::spread_llr(y, s9, fr::get("short")); },
                                  "operands could not be broadcast together with shapes (10,) (9,) ",
                                  "spread_llr lengths");
    // erased everywhere: v inf, so w = 0 and z = 0
    const std::vector<double> inf(10, INFINITY);
    const auto be = dm::block_estimate(y, inf, Q);
    check::is_true(std::isinf(be.v[0]) && be.z[0] == 0.0, "an all-erased block");
    // numpy's index rules on a short grid
    const auto& t = fr::get("tiny");
    auto s = demod_synth::make(t);
    s.m.resize(5);
    throws<std::out_of_range>([&] { (void)dm::symbol_estimates(s.view(), fr::layout(t)->data); },
                              "index 701 is out of bounds for axis 0 with size 5", "a grid too short");
}

int main() {
    try {
        test_tiny();
        test_short_spread();
        test_combine();
        test_refusals();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FATAL: %s\n", e.what());
        return 1;
    }
    return check::report("qrss_demod");
}

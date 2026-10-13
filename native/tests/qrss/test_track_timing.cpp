// native/core/qrss/track_timing against values pinned from the Python
// reference (sstvae/qrss/track.py). The pass is a synthetic TINY frame:
//
//   x = np.where((np.arange(TINY.n_data) * 5 + 1) % 3 == 0, 1.0, -1.0)
//   sym = frame.assemble(TINY, None, x); i = np.arange(17000)
//   phi = ce.phase_grid(250, -3000, 17000, sym, TINY)
//   sig = np.exp(1j * (phi + 2 * np.pi * (0.02 * i / 250 + 0.5 * 0.001 * (i / 250) ** 2)))
//   noise = (((i * 7) % 19 - 9) / 8.0) + 1j * (((i * 3) % 17 - 8) / 8.0)
//   chan = ChanCapture(ch=(sig + noise).astype(np.complex64), fs=250, t0_index=3002.5,
//                      f_mix=Fraction(0))                 (and noise alone: refused)
//   flat = FreqPath(t_s=np.array([-10.0, 0.0, 1800.0]), f_hz=np.full(3, 1500.0), weight=np.ones(3))
//   det = Detection(1500.0, flat, None, np.nan, "tbd", 0.0)
//   track.fit_timing(chan, CH_FS, flat, TINY, t0_index=chan.t0_index, n_best=3)
//   track.fit_timing(chan, CH_FS, flat, TINY, tau0_init=-2.0, t0_index=chan.t0_index)
//   x = signal.oaconvolve(z_d * np.conj(track._lowpass(z_d, 0.3, CH_FS)), track._mf_taps(CH_FS), "same").imag
//   sig2 = max(float(np.median(x[2373:16250] ** 2)) / 0.4549, 1e-30)
//   track._segment_timing(x, sig2, track.known_symbols(TINY), 3002.7, 3e-5, T_SYM * 250, TINY.n_pos)
//   track._carrier_lines(z_d, chan, TINY, 20.0, 2.0); track.refine_path(chan, TINY, flat)
//   track.line_path(chan, TINY, flat) is flat          (a 53 s frame has no three 60 s segments)
//   track.verify(chan, CH_FS, TINY, det); track.verify_track(chan, TINY, det[, erase_s=((10.0, 20.0),)])
//
// with z_d = track.derotate(chan, track.freq_path_spline(flat)). The fit's
// grid is pinned bitwise against the reference's direct-path terms summed in
// k order ((a[:, None] * v).sum(axis=0) / norm, the script beside
// test_grid). tests/test_qrss_native_track_timing.py compares against the
// reference directly, on real passes.

#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <memory>
#include <numbers>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "qrss/ce.hpp"
#include "qrss/frame.hpp"
#include "qrss/rational.hpp"
#include "qrss/rx_config.hpp"
#include "qrss/track.hpp"
#include "qrss/track_timing.hpp"
#include "qrss/trackbase.hpp"
#include "qrss/types.hpp"
#include "qrss_check.hpp"

using namespace sstvae;
namespace tk = sstvae::qrss::track;
namespace fr = sstvae::qrss::frame;
namespace types = sstvae::qrss::types;
using cd = std::complex<double>;
using qrss_check::digest_is;

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

static void near(double got, double want, double tol, const std::string& what, bool relative = true) {
    const double lim = relative ? tol * std::abs(want) : tol;
    const bool ok = std::abs(got - want) <= lim;
    check::is_true(ok, what);
    if (!ok) std::fprintf(stderr, "   %s: got %.17g want %.17g\n", what.c_str(), got, want);
}

static std::shared_ptr<const tk::ChanCapture> synthetic_chan(bool with_signal) {
    const auto& TINY = fr::get("tiny");
    constexpr std::int64_t n = 17000;
    std::vector<double> x;
    for (std::int64_t k = 0; k < TINY.n_data(); ++k) x.push_back((k * 5 + 1) % 3 == 0 ? 1.0 : -1.0);
    const auto sym = fr::assemble(TINY, nullptr, x);
    const auto phi = qrss::ce::phase_grid(250, -3000, n, sym, TINY);
    auto chan = std::make_shared<tk::ChanCapture>();
    const double half_milli = 0.5 * 0.001;
    for (std::int64_t i = 0; i < n; ++i) {
        const double di = static_cast<double>(i);
        const double sq = (di / 250) * (di / 250);
        const double ph = phi[static_cast<std::size_t>(i)] + 2 * std::numbers::pi * (0.02 * di / 250 + half_milli * sq);
        const cd noise(static_cast<double>((i * 7) % 19 - 9) / 8.0, static_cast<double>((i * 3) % 17 - 8) / 8.0);
        const cd z = with_signal ? cd(std::cos(ph), std::sin(ph)) + noise : noise;
        chan->ch.emplace_back(static_cast<float>(z.real()), static_cast<float>(z.imag()));
    }
    chan->fs = 250;
    chan->t0_index = 3002.5;
    chan->f_mix = qrss::Rational(0);
    return chan;
}

static types::FreqPath flat_path_1500() {
    return types::FreqPath{{-10.0, 0.0, 1800.0}, {1500.0, 1500.0, 1500.0}, {1.0, 1.0, 1.0}};
}

static types::Detection synthetic_det() {
    types::Detection det;
    det.f_hz = 1500.0;
    det.path = flat_path_1500();
    det.z_ref = std::numeric_limits<double>::quiet_NaN();
    det.method = "tbd";
    return det;
}

static void test_parabola() {
    check::equal(tk::parabola(1.0, 3.0, 2.0), 0.16666666666666666, "parabola");
    check::equal(tk::parabola(1.0, 0.5, 2.0), 0.0, "parabola, a minimum");
    check::equal(tk::parabola(-4.0, 9.0, 7.5), 0.39655172413793105, "parabola, clipped side");
    check::equal(tk::parabola(std::numeric_limits<double>::infinity(), 1.0, 0.0), 0.0, "parabola, infinite");
}

// The grid against the reference's direct path, its terms summed in k order:
//   xs = (((np.arange(3000) * 11) % 23) - 11) / 16.0
//   kn = _Known(pos=np.arange(0, 300, 7), val=np.where(np.arange(0, 300, 7) % 3 == 0, 1.0, -1.0))
//   for d in (-2.5e-3, 0.0, 1.25e-3):
//       o = kn.pos * 7.578125 * (1 + d) + 0.375; idx = -40 + o[:, None] + np.arange(900)[None, :]
//       i0 = np.floor(idx).astype(np.int64); f = idx - i0; ok = (i0 >= 0) & (i0 + 1 < len(xs))
//       i0c = np.clip(i0, 0, len(xs) - 2); v = (xs[i0c] * (1 - f) + xs[i0c + 1] * f) * ok
//       row = (kn.val[:, None] * v).sum(axis=0) / 3.0
static void test_grid() {
    std::vector<double> xs(3000);
    for (std::int64_t i = 0; i < 3000; ++i) xs[static_cast<std::size_t>(i)] = static_cast<double>((i * 11) % 23 - 11) / 16.0;
    tk::Known kn;
    std::vector<double> val;
    for (std::int64_t p = 0; p < 300; p += 7) {
        kn.pos.push_back(p);
        val.push_back(p % 3 == 0 ? 1.0 : -1.0);
    }
    kn.val = val;
    const std::vector<double> deltas{-2.5e-3, 0.0, 1.25e-3};
    const auto Z = tk::fit_grid(xs, kn, -40, 900, 0.375, deltas, 7.578125, 3.0);
    check::equal(Z.size(), std::size_t{2700}, "grid size");
    check::equal(Z[900 + 100], -1.3326822916666667, "grid cell");
    digest_is(Z, "af33f2d418de9d285fcd27bc0a58afdf655c1e566bd2d2f9abe328fbef16c4e3", "grid, bitwise");
    // a non-finite value where the clip reads makes every term outside x NaN, as numpy's * ok does
    auto xn = xs;
    xn.back() = std::numeric_limits<double>::infinity();
    const auto Zn = tk::fit_grid(xn, kn, -40, 900, 0.375, deltas, 7.578125, 3.0);
    check::is_true(std::isnan(Zn[899]), "a lag reaching past x, its last sample infinite, is NaN");
    check::equal(Zn[0], Z[0], "a lag inside x is unchanged");
    throws<std::invalid_argument>([&] { (void)tk::fit_grid(xs, kn, 0, -1, 0.0, deltas, 1.0, 1.0); },
                                  "negative dimensions are not allowed", "negative nlag");
}

static void test_timing(const std::shared_ptr<const tk::ChanCapture>& chan) {
    const auto& TINY = fr::get("tiny");
    const tk::PathFn fn = tk::freq_path_spline(flat_path_1500());
    const auto z_d = tk::derotate(*chan, fn);
    tk::FitTimingArgs a;
    a.n_best = 3;
    tk::FitTimingTrace trace;
    const auto tms = tk::fit_timing(*chan, 250, fn, TINY, chan->t0_index, a, &trace);
    check::equal(tms.size(), std::size_t{3}, "three timings");
    check::equal(trace.nlag, std::int64_t{1255}, "blind lag count");
    check::equal(trace.base, std::int64_t{3002 - 627}, "base");
    check::equal(trace.peaks.size(), std::size_t{12}, "three peaks per carrier estimate");
    const double want_tau0[3] = {-2.9211359340711169, -139.55159350765598, 133.62722860717577};
    const double want_ppm[3] = {228.18026240730177, -228.18026240730177, 228.18026240730177};
    const double want_z[3] = {28.551819617974193, 7.2541885804806343, 6.7749310392892523};
    for (int k = 0; k < 3; ++k) {
        const std::string s = "timing " + std::to_string(k);
        near(tms[static_cast<std::size_t>(k)].tau0, want_tau0[k], 1e-6, s + " tau0", false);
        near(tms[static_cast<std::size_t>(k)].ppm, want_ppm[k], 1e-6, s + " ppm", false);
        near(tms[static_cast<std::size_t>(k)].z, want_z[k], 1e-9, s + " z");
        near(tms[static_cast<std::size_t>(k)].cov(1, 1), 13016.558038066272, 1e-15, s + " cov");
        check::equal(tms[static_cast<std::size_t>(k)].cov(0, 0), 0.25, s + " cov tau0");
    }
    // the span overload, and n_best 1
    const auto one = tk::fit_timing(z_d, 250, TINY, chan->t0_index);
    check::equal(one.size(), std::size_t{1}, "n_best 1: one timing");
    near(one[0].tau0, want_tau0[0], 1e-6, "n_best 1 tau0", false);
    // a local search (nlag <= 64: the reference's own direct path)
    tk::FitTimingArgs loc;
    loc.tau0_init = -2.0;
    const auto t1 = tk::fit_timing(z_d, 250, TINY, chan->t0_index, loc);
    near(t1[0].tau0, -2.9211359340711169, 1e-6, "local tau0", false);
    near(t1[0].ppm, 228.18026240730177, 1e-6, "local ppm", false);
    near(t1[0].z, 28.802463037220942, 1e-9, "local z");

    // _segment_timing on the 0.3 Hz carrier estimate's statistic
    const auto x = tk::fit_statistic(z_d, 0.3);
    // to rounding, not bitwise: _lowpass and the MF are FFT convolutions (track.hpp)
    check::equal(x.size(), std::size_t{17000}, "statistic length");
    near(x[4000], 0.43245180548001683, 1e-12, "statistic sample");
    const tk::Known kn = tk::known_symbols(TINY);
    const auto st = tk::segment_timing(x, 0.39904398031258814, kn, 3002.0 + 0.7, 3e-5, qrss::config::T_SYM * 250,
                                       TINY.n_pos());
    near(st.t0abs, 2999.9029338181422, 1e-9, "segment t0", false);
    near(st.delta, 1.1344063045815034e-05, 1e-12, "segment delta", false);
    near(st.score, 397.18189586816322, 1e-9, "segment score");
}

static void test_paths(const std::shared_ptr<const tk::ChanCapture>& chan) {
    const auto& TINY = fr::get("tiny");
    const types::FreqPath flat = flat_path_1500();
    const auto z_d = tk::derotate(*chan, tk::freq_path_spline(flat));
    const auto L = tk::carrier_lines(z_d, *chan, TINY, 20.0, 2.0);
    check::is_true(L.t == std::vector<double>{10.002000000000001, 30.001999999999999, 47.996000000000002}, "line times");
    const double want_f[3] = {0.043810335405613812, 0.062271469637837612, 0.084657194136299685};
    const double want_w[3] = {725.7413939000437, 9944.7703596760439, 7221.1690016115735};
    for (int k = 0; k < 3 && k < static_cast<int>(L.f.size()); ++k) {
        near(L.f[static_cast<std::size_t>(k)], want_f[k], 1e-9, "line f " + std::to_string(k), false);
        near(L.w[static_cast<std::size_t>(k)], want_w[k], 1e-9, "line ratio " + std::to_string(k));
    }
    check::equal(tk::carrier_lines(z_d, *chan, TINY, 20.0, 2.0, std::nullopt, 1e9).t.size(), std::size_t{0},
                 "no line at ratio 1e9");

    const auto rp = tk::refine_path(*chan, TINY, flat);
    check::is_true(!rp.unchanged, "refine_path measured a path");
    check::equal(rp.path.t_s.size(), std::size_t{73}, "refined path length");
    check::equal(rp.path.t_s.front(), -10.0, "refined path start");
    check::equal(rp.path.t_s.back(), 62.0, "refined path end");
    const std::size_t at[4] = {0, 10, 30, 72};
    const double want_rp[4] = {1500.0807040809434, 1500.0516172683294, 1500.0497120107414, 1500.1845741729344};
    for (int k = 0; k < 4; ++k)
        near(rp.path.f_hz[at[k]], want_rp[k], 1e-7, "refined path f " + std::to_string(k), false);
    check::is_true(rp.path.weight == std::vector<double>(73, 1.0), "refined path weights");
    tk::RefinePathArgs strict;
    strict.ratio = 1e9;
    check::is_true(tk::refine_path(*chan, TINY, flat, strict).unchanged, "no line: refine_path keeps the path");

    const auto lp = tk::line_path(*chan, TINY, flat);
    check::is_true(lp.unchanged && lp.path == flat, "a 53 s frame has no three 60 s segments: line_path keeps it");

    const types::FreqPath p4{{-10.0, 0.0, 30.0, 60.0}, {1500.0, 1500.5, 1501.0, 1500.25}, {1.0, 1.0, 1.0, 1.0}};
    const auto fp = tk::flat_path(p4);
    check::is_true(fp.t_s == std::vector<double>{-10.0, 0.0, 1800.0} && fp.weight == std::vector<double>(3, 1.0),
                   "flat path times and weights");
    near(fp.f_hz[0], 1500.3708333333327, 1e-13, "flat path f");
    check::is_true(fp.f_hz[0] == fp.f_hz[1] && fp.f_hz[1] == fp.f_hz[2], "flat path is flat");
}

static void test_gate(const std::shared_ptr<const tk::ChanCapture>& chan) {
    const auto& TINY = fr::get("tiny");
    const types::Detection det = synthetic_det();
    const auto v = tk::verify(chan, TINY, det);
    check::is_true(v.has_value(), "the synthetic pass is accepted");
    if (v) {
        near(v->z_ref, 514.71311243806178, 1e-8, "verify z_ref");
        near(v->timing->tau0, -2.6399606084187326, 1e-6, "verify tau0", false);
        near(v->timing->ppm, 28.379434522829428, 1e-6, "verify ppm", false);
        check::equal(v->path.t_s.size(), std::size_t{73}, "verify path (refined)");
        near(v->path.f_hz[0], 1500.0807040809434, 1e-7, "verify path f0", false);
        check::is_true(std::isnan(v->a3_z) && v->method == "tbd" && v->f_hz == 1500.0, "verify's other fields");
    }
    const auto vt = tk::verify_track(chan, TINY, det);
    near(vt.z_ref, 65.646759869557968, 1e-8, "verify_track z_ref");
    near(vt.q, 1.0782513897100879e-06, 1e-9, "verify_track q");
    near(vt.timing.tau0, -2.9211359340711169, 1e-6, "verify_track tau0", false);
    tk::VerifyArgs er;
    er.erase_s = {{10.0, 20.0}};
    near(tk::verify_track(chan, TINY, det, er).z_ref, 63.074376909088905, 1e-8, "verify_track erased z_ref");

    // noise alone is refused
    const auto noise = synthetic_chan(false);
    check::is_true(!tk::verify(noise, TINY, det).has_value(), "noise is refused");
    near(tk::verify_track(noise, TINY, det).z_ref, -1.5115636605271277, 1e-6, "noise z_ref");

    // reentrant: concurrent gates on one capture and on fresh ones agree
    std::vector<std::thread> th;
    std::vector<std::optional<types::Detection>> out(4);
    for (int i = 0; i < 4; ++i)
        th.emplace_back([&, i] { out[static_cast<std::size_t>(i)] = tk::verify(i % 2 ? synthetic_chan(true) : chan, TINY, det); });
    for (auto& t : th) t.join();
    for (const auto& o : out)
        check::is_true(o && v && o->z_ref == v->z_ref && o->timing == v->timing && o->path == v->path,
                       "a concurrent verify");

    // refusals: an L frame, before anything else is read
    const auto& L = fr::get_l("l-short");
    throws<fr::WrongWaveform>([&] { (void)tk::fit_timing(std::span<const cd>(), 250, L, 0.0); },
                              "track.fit_timing is CE's; 'l-short' is waveform 1: use ltrack.fit_timing_l",
                              "fit_timing, an L frame");
    throws<fr::WrongWaveform>([&] { (void)tk::verify(chan, L, det); },
                              "track.verify is CE's; 'l-short' is waveform 1: use ltrack.verify_l", "verify, an L frame");
    throws<fr::WrongWaveform>([&] { (void)tk::verify_track(chan, L, det); },
                              "track.verify_track is CE's; 'l-short' is waveform 1: use ltrack.verify_track_l",
                              "verify_track, an L frame");
}

int main() {
    try {
        test_parabola();
        test_grid();
        const auto chan = synthetic_chan(true);
        test_timing(chan);
        test_paths(chan);
        test_gate(chan);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FATAL: %s\n", e.what());
        return 1;
    }
    return check::report("qrss_track_timing");
}

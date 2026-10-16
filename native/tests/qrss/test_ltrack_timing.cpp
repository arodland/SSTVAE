// native/core/qrss/ltrack_timing against values pinned from the Python
// reference (sstvae/qrss/ltrack.py). The pins are ltrack_timing_pins.hpp,
// written by gen_ltrack_timing_pins.py beside this file (re-run it when the
// reference changes on purpose); every input here is rebuilt from the
// formulas that script uses -- exact binary fractions, correctly rounded
// decimal literals, or a function the port computes to the bit
// (lwave::composite, frame::assemble_l, known_symbols_l, make_classes_l,
// data_templates) -- so both sides start from the same doubles.
//
// Bitwise: _lag_phase and _clock_phase (signed zeros included), resid_ratio,
// and the coarse grid against the C++'s own summation order (each k sum
// numpy's sequential reduction, restated in the script). To rounding, its
// decisions exactly: the FFT convolutions (fit_mixed, the statistic), the
// grid against the reference's direct path, the fit on a synthetic l-tiny
// pass (the trace's cells, the timings), _segment_timing_l, refine_path_l
// and gate V (verify_track_l, verify_l: accepted, and noise refused).
// tests/test_qrss_native_ltrack_timing.py compares against the reference
// directly, on the receiver's own passes.

#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <variant>
#include <vector>

#include "ltrack_timing_pins.hpp"
#include "qrss/chan.hpp"
#include "qrss/qrss_config.hpp"
#include "qrss/frame.hpp"
#include "qrss/ltrack.hpp"
#include "qrss/ltrack_timing.hpp"
#include "qrss/lwave.hpp"
#include "qrss/rational.hpp"
#include "qrss/track.hpp"
#include "qrss/track_timing.hpp"
#include "qrss/trackbase.hpp"
#include "qrss/types.hpp"
#include "qrss_check.hpp"

using namespace sstvae;
namespace lt = sstvae::qrss::ltrack;
namespace tk = sstvae::qrss::track;
namespace fr = sstvae::qrss::frame;
namespace lw = sstvae::qrss::lwave;
namespace cfg = sstvae::qrss::config;
namespace types = sstvae::qrss::types;
namespace P = ltrack_timing_pins;
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

// |got - want| <= rel |want| + abs
static void near(double got, double want, double rel, const std::string& what, double abs = 0.0) {
    const bool ok = std::abs(got - want) <= rel * std::abs(want) + abs;
    check::is_true(ok, what);
    if (!ok) std::fprintf(stderr, "   %s: got %.17g want %.17g\n", what.c_str(), got, want);
}

static bool same_bits(double a, double b) { return std::signbit(a) == std::signbit(b) && a == b; }

static const fr::FrameSpec& TINY() { return fr::get_l("l-tiny"); }
static const fr::FrameSpec& SHORT() { return fr::get_l("l-short"); }
static double sps() { return cfg::L_T_SYM * cfg::CH_FS; }

// --- the mix's phases ------------------------------------------------------------------------------

static void test_phases() {
    std::vector<double> lag;
    for (std::int64_t k = -3000; k < 3000; ++k) lag.push_back(static_cast<double>(k) * 0.375 + 0.0625);
    qrss_check::libm_digest_is(lt::lag_phase(std::span<const double>(lag)), P::LAG_PHASE, "_lag_phase");
    const std::pair<double, const double*> scalars[] = {{0.0, P::LP_ZERO}, {-0.0, P::LP_NZERO}, {12.5, P::LP_HALF}};
    for (const auto& [v, want] : scalars) {
        const cd z = lt::lag_phase(v);
        check::is_true(same_bits(z.real(), want[0]) && same_bits(z.imag(), want[1]),
                       "_lag_phase(" + std::to_string(v) + "), signed zeros");
        check::is_true(z == lt::lag_phase(std::span<const double>(&v, 1))[0], "_lag_phase scalar = array");
    }
    std::vector<double> pos;
    for (std::int64_t p = 0; p < 30000; p += 7) pos.push_back(static_cast<double>(p));
    digest_is(lt::clock_phase(pos, 0.0), P::CP_0, "_clock_phase d 0");
    qrss_check::libm_digest_is(lt::clock_phase(pos, 2.5e-4), P::CP_P, "_clock_phase d 2.5e-4");
    qrss_check::libm_digest_is(lt::clock_phase(pos, -1.75e-4), P::CP_M, "_clock_phase d -1.75e-4");
}

// --- the statistic and the grid ---------------------------------------------------------------

static std::vector<cd> zc16k() {
    std::vector<cd> z;
    for (std::int64_t n = 0; n < 16000; ++n)
        z.emplace_back(static_cast<double>((n * 7) % 19 - 9) / 16.0, static_cast<double>((n * 3) % 17 - 8) / 32.0);
    return z;
}

static void test_statistic() {
    const auto zc = zc16k();
    const auto y = lt::fit_mixed(zc, 12.375);
    const auto x = lt::fit_statistic_l(zc, 12.375, 0.3);
    check::equal(y.size(), zc.size(), "fit_mixed size");
    check::equal(x.size(), zc.size(), "fit_statistic_l size");
    // to rounding: scipy's oaconvolve against pocketfft
    for (std::size_t j = 0; j < std::size(P::FX_IDX); ++j) {
        const auto k = static_cast<std::size_t>(P::FX_IDX[j]);
        const std::string at = " [" + std::to_string(k) + "]";
        near(y[k].real(), P::FM_RE[j], 0.0, "fit_mixed re" + at, 1e-13 * P::FM_SCALE);
        near(y[k].imag(), P::FM_IM[j], 0.0, "fit_mixed im" + at, 1e-13 * P::FM_SCALE);
        near(x[k].real(), P::FS_RE[j], 0.0, "statistic re" + at, 1e-13 * P::FS_SCALE);
        near(x[k].imag(), P::FS_IM[j], 0.0, "statistic im" + at, 1e-13 * P::FS_SCALE);
    }
    // the overload from fit_mixed is the same statistic
    check::is_true(lt::fit_statistic_l(zc, y, 0.3) == x, "fit_statistic_l from a fit_mixed");
}

static void test_grid() {
    const auto zc = zc16k();
    const tk::Known kn = lt::known_symbols_l(TINY());
    std::vector<double> deltas;
    for (int k = -3; k < 4; ++k) deltas.push_back(static_cast<double>(k) * 2.5e-5);
    const auto Z = lt::fit_grid_l(zc, kn, -40, 300, 0.375, deltas, sps(), 3.5);
    check::equal(Z.size(), deltas.size() * 300, "grid size");
    qrss_check::libm_digest_is(Z, P::GRID, "grid, bitwise in the C++'s order");
    for (std::size_t c = 0; c < std::size(P::GRID_REF); ++c) {
        const auto i = static_cast<std::size_t>(P::GRID_CELLS[2 * c]), j = static_cast<std::size_t>(P::GRID_CELLS[2 * c + 1]);
        near(Z[i * 300 + j], P::GRID_REF[c], 0.0, "grid against the reference's direct path, cell " + std::to_string(c),
             1e-13 * P::GRID_SCALE);
    }
    // a real known set is taken as complex (np.asarray(kn.val, complex128))
    tk::Known kr{kn.pos, std::vector<double>()};
    tk::Known kc{kn.pos, std::vector<cd>()};
    for (const cd& a : kn.cplx()) {
        std::get<0>(kr.val).push_back(a.real());
        std::get<1>(kc.val).emplace_back(a.real(), 0.0);
    }
    check::is_true(lt::fit_grid_l(zc, kr, -40, 30, 0.375, deltas, sps(), 3.5) ==
                       lt::fit_grid_l(zc, kc, -40, 30, 0.375, deltas, sps(), 3.5),
                   "a real known set");
    check::is_true(lt::fit_grid_l(zc, kn, -40, 0, 0.375, deltas, sps(), 3.5).empty(), "no lags");
    throws<std::invalid_argument>([&] { (void)lt::fit_grid_l(zc, kn, 0, -1, 0.0, deltas, 1.0, 1.0); },
                                  "negative dimensions are not allowed", "negative nlag");
}

// --- the synthetic l-tiny pass -------------------------------------------------------------------
// lwave.composite at 250 Hz from 12 s before t0, plus a structured "noise"
// ((i*7 % 19 - 9) + j (i*3 % 17 - 8)) / 4, as complex64.

constexpr std::int64_t N_CH = 17500;

static std::shared_ptr<tk::ChanCapture> synthetic_chan(bool with_signal, std::int64_t n_keep = N_CH) {
    const auto& T = TINY();
    std::vector<double> x;
    for (std::int64_t k = 0; k < T.n_data(); ++k) x.push_back((k * 5 + 1) % 3 == 0 ? 0.5 : -0.5);
    const auto sym = fr::assemble_l(T, nullptr, x);
    const auto sig = lw::composite(sym, T, cfg::CH_FS, -3000, N_CH, nullptr, 0.0);
    auto chan = std::make_shared<tk::ChanCapture>();
    for (std::int64_t i = 0; i < n_keep; ++i) {
        const cd noise(static_cast<double>((i * 7) % 19 - 9) / 4.0, static_cast<double>((i * 3) % 17 - 8) / 4.0);
        const cd z = with_signal ? sig[static_cast<std::size_t>(i)] + noise : noise;
        chan->ch.emplace_back(static_cast<float>(z.real()), static_cast<float>(z.imag()));
    }
    chan->fs = cfg::CH_FS;
    chan->t0_index = 3000.0;
    chan->f_mix = qrss::Rational(0);
    return chan;
}

static types::FreqPath flat_path() {
    return types::FreqPath{{-10.0, 0.0, 1800.0}, {1500.0, 1500.0, 1500.0}, {1.0, 1.0, 1.0}};
}

static types::Detection synthetic_det(std::optional<types::Timing> tm = std::nullopt) {
    types::Detection det;
    det.f_hz = 1500.0;
    det.path = flat_path();
    det.timing = std::move(tm);
    det.z_ref = std::numeric_limits<double>::quiet_NaN();
    det.method = "tbd";
    det.lead_in_s = 0.0;
    det.waveform = cfg::WAVEFORM_L;
    return det;
}

static types::Timing known_timing() {
    types::Timing tm;
    tm.tau0 = P::VT_TM[0];
    tm.ppm = P::VT_TM[1];
    tm.cov = math::Matrix(3, 3);
    return tm;
}

// tau0 to 1e-6 CH samples, ppm to 1e-6 samples of slip over the frame, z to 1e-8 relative
static void timing_near(const types::Timing& g, double tau0, double ppm, double z, const std::string& what) {
    near(g.tau0, tau0, 0.0, what + " tau0", 1e-6);
    near(g.ppm * 1e-6 * static_cast<double>(TINY().n_pos()) * sps(), ppm * 1e-6 * static_cast<double>(TINY().n_pos()) * sps(),
         0.0, what + " ppm (slip)", 1e-6);
    near(g.z, z, 1e-8, what + " z", 1e-8);
}

static void test_fit(const std::shared_ptr<const tk::ChanCapture>& chan) {
    const auto& T = TINY();
    const tk::PathFn fn = tk::freq_path_spline(flat_path());
    const auto z_d = tk::derotate(*chan, fn);
    lt::FitTimingLArgs a;
    a.n_best = 8;
    tk::FitTimingTrace tr;
    const auto tms = lt::fit_timing_l(*chan, cfg::CH_FS, fn, T, chan->t0_index, a, &tr);
    check::equal(tr.base, P::FT_BASE, "base");
    check::equal(tr.nlag, P::FT_NLAG, "blind lag count");
    check::equal(tr.nd, P::FT_ND, "clock errors");
    check::equal(tr.step, P::FT_STEP, "clock error step");
    check::equal(tr.frac0, 0.0, "frac0");
    check::equal(tr.sig2.size(), std::size(P::FT_SIG2), "a sig2 per carrier estimate");
    for (std::size_t k = 0; k < tr.sig2.size() && k < std::size(P::FT_SIG2); ++k)
        near(tr.sig2[k], P::FT_SIG2[k], 1e-12, "sig2 " + std::to_string(k));
    // the argmax cells and the runners-up, in the order found
    check::equal(tr.peaks.size() * 3, std::size(P::FT_CELLS), "peaks");
    for (std::size_t k = 0; k < tr.peaks.size() && 3 * k + 2 < std::size(P::FT_CELLS); ++k) {
        const auto& p = tr.peaks[k];
        check::is_true(static_cast<std::int64_t>(p.lp) == P::FT_CELLS[3 * k] && p.i == P::FT_CELLS[3 * k + 1] &&
                           p.j == P::FT_CELLS[3 * k + 2],
                       "peak cell " + std::to_string(k));
    }
    check::equal(static_cast<std::int64_t>(tms.size()), P::FT_N, "timings");
    for (std::size_t k = 0; k < tms.size() && k < std::size(P::FT_TAU0); ++k) {
        timing_near(tms[k], P::FT_TAU0[k], P::FT_PPM[k], P::FT_Z[k], "timing " + std::to_string(k));
        check::is_true(tms[k].gamma == 0.0 && tms[k].cov(0, 0) == 0.25 && tms[k].cov(2, 2) == 0.0,
                       "timing " + std::to_string(k) + " gamma, cov");
        check::equal(tms[k].cov(1, 1), (P::FT_STEP * 1e6) * (P::FT_STEP * 1e6), "timing cov ppm");
    }
    // the span overload, n_best 1
    const auto one = lt::fit_timing_l(z_d, cfg::CH_FS, T, chan->t0_index);
    check::equal(one.size(), std::size_t{1}, "n_best 1: one timing");
    timing_near(one[0], P::FT_TAU0[0], P::FT_PPM[0], P::FT_Z[0], "n_best 1");

    // a local search: the reference's own direct path
    lt::FitTimingLArgs loc;
    loc.tau0_init = P::FT_TAU0[0] + 0.75;
    tk::FitTimingTrace tl;
    const auto t1 = lt::fit_timing_l(z_d, cfg::CH_FS, T, chan->t0_index, loc, &tl);
    check::is_true(tl.nlag <= 64, "a local search's lags");
    timing_near(t1.at(0), P::FL[0], P::FL[1], P::FL[2], "local");
    // with a real known set (pos[::3], sign(Re val[::3])) and tau_range 5
    const tk::Known kn = lt::known_symbols_l(T);
    tk::Known kr{{}, std::vector<double>()};
    for (std::size_t k = 0; k < kn.pos.size(); k += 3) {
        kr.pos.push_back(kn.pos[k]);
        const double re = kn.cplx()[k].real();
        std::get<0>(kr.val).push_back((re > 0 ? 1.0 : re < 0 ? -1.0 : 0.0) + 0.0);
    }
    lt::FitTimingLArgs lr = loc;
    lr.known = &kr;
    lr.tau_range = 5.0;
    timing_near(lt::fit_timing_l(z_d, cfg::CH_FS, T, chan->t0_index, lr).at(0), P::FLR[0], P::FLR[1], P::FLR[2],
                "local, a real known set");

    // _segment_timing_l on the 0.3 Hz carrier estimate's statistic
    const double n_ref = static_cast<double>(P::FT_BASE);
    const auto x = lt::fit_statistic_l(z_d, n_ref, 0.3);
    for (std::size_t c = 0; c * 3 + 2 < std::size(P::SEG_IN); ++c) {
        const auto st = lt::segment_timing_l(x, P::FT_SIG2[1], kn, P::SEG_IN[3 * c], P::SEG_IN[3 * c + 1], sps(),
                                             T.n_pos(), n_ref, P::SEG_IN[3 * c + 2]);
        const std::string s = "segment " + std::to_string(c);
        near(st.t0abs, P::SEG_OUT[3 * c], 0.0, s + " t0", 1e-6);
        near(st.delta * static_cast<double>(T.n_pos()) * sps(), P::SEG_OUT[3 * c + 1] * static_cast<double>(T.n_pos()) * sps(),
             0.0, s + " slip", 1e-6);
        near(st.score, P::SEG_OUT[3 * c + 2], 1e-8, s + " score");
    }
}

static void test_path(const std::shared_ptr<const tk::ChanCapture>& chan) {
    const auto& T = TINY();
    const auto rp = lt::refine_path_l(*chan, T, flat_path());
    check::is_true(!rp.unchanged, "refine_path_l measured a path");
    check::equal(static_cast<std::int64_t>(rp.path.t_s.size()), P::RP_N, "refined path length");
    if (!rp.path.t_s.empty()) {
        check::equal(rp.path.t_s.front(), P::RP_T_ENDS[0], "refined path start");
        check::equal(rp.path.t_s.back(), P::RP_T_ENDS[1], "refined path end");
    }
    for (std::size_t k = 0; k < std::size(P::RP_IDX); ++k) {
        const auto at = static_cast<std::size_t>(P::RP_IDX[k]);
        if (at < rp.path.f_hz.size())
            near(rp.path.f_hz[at], P::RP_F[k], 0.0, "refined path f " + std::to_string(k), 1e-7);
    }
    check::is_true(rp.path.weight == std::vector<double>(rp.path.t_s.size(), 1.0), "refined path weights");
    // a capture cut short of two path segments: the path itself
    const auto cut = synthetic_chan(true, 3000 + 15 * cfg::CH_FS);
    const auto rc = lt::refine_path_l(*cut, T, flat_path());
    check::is_true(rc.unchanged && rc.path == flat_path(), "a short capture: refine_path_l keeps the path");
}

static void test_gate(const std::shared_ptr<const tk::ChanCapture>& chan) {
    const auto& T = TINY();
    const auto det = synthetic_det();
    const auto vt = lt::verify_track_l(chan, T, det);
    near(vt.q, P::VT[0], 1e-12, "verify_track_l q");
    near(vt.z_ref, P::VT[1], 1e-8, "verify_track_l z_ref");
    near(vt.resid_ratio, P::VT[2], 1e-8, "verify_track_l resid_ratio");
    near(vt.timing.tau0, P::VT[3], 0.0, "verify_track_l tau0", 1e-6);
    near(vt.timing.ppm, P::VT[4], 1e-6, "verify_track_l ppm");
    check::is_true(vt.chan == chan, "verify_track_l keeps the capture");

    const auto dk = synthetic_det(known_timing());
    tk::VerifyArgs nofit;
    nofit.fit = false;
    const auto vk = lt::verify_track_l(chan, T, dk, nofit);
    check::is_true(vk.timing == *dk.timing, "fit=False: the candidate's timing");
    near(vk.q, P::VTK[0], 1e-12, "verify_track_l (timing known) q");
    near(vk.z_ref, P::VTK[1], 1e-8, "verify_track_l (timing known) z_ref");
    near(vk.resid_ratio, P::VTK[2], 1e-8, "verify_track_l (timing known) resid_ratio");
    tk::VerifyArgs er = nofit;
    er.erase_s = {{10.0, 20.0}};
    const auto ve = lt::verify_track_l(chan, T, dk, er);
    near(ve.q, P::VTE[0], 1e-12, "verify_track_l (erased) q");
    near(ve.z_ref, P::VTE[1], 1e-8, "verify_track_l (erased) z_ref");
    near(ve.resid_ratio, P::VTE[2], 1e-8, "verify_track_l (erased) resid_ratio");

    const auto v = lt::verify_l(chan, T, det);
    check::is_true(v.has_value(), "the synthetic pass is accepted");
    if (v) {
        near(v->z_ref, P::V[0], 1e-8, "verify_l z_ref");
        near(v->timing->tau0, P::V[1], 0.0, "verify_l tau0", 1e-6);
        near(v->timing->ppm, P::V[2], 1e-6, "verify_l ppm");
        check::equal(static_cast<std::int64_t>(v->path.t_s.size()), P::V_PATH_N, "verify_l path (refined)");
        for (std::size_t k = 0; k < std::size(P::RP_IDX); ++k) {
            const auto at = static_cast<std::size_t>(P::RP_IDX[k]);
            if (at < v->path.f_hz.size())
                near(v->path.f_hz[at], P::V_PATH_F[k], 0.0, "verify_l path f " + std::to_string(k), 1e-7);
        }
        check::is_true(std::isnan(v->a3_z) && v->method == "tbd" && v->f_hz == 1500.0 &&
                           v->waveform == cfg::WAVEFORM_L && v->lead_in_s == 0.0,
                       "verify_l's other fields");
    }

    // noise alone is refused (after every base and the runners-up)
    const std::shared_ptr<const tk::ChanCapture> noise = synthetic_chan(false);
    qrss_check::digest_is(noise->ch, P::CAP_NOISE, "the noise capture");
    check::is_true(!lt::verify_l(noise, T, det).has_value(), "noise is refused");
    const auto vn = lt::verify_track_l(noise, T, det);
    near(vn.z_ref, P::VTN[0], 1e-8, "noise z_ref", 1e-8);
    near(vn.resid_ratio, P::VTN[1], 1e-8, "noise resid_ratio");

    // reentrant: concurrent gates on one capture and on fresh ones agree
    std::vector<std::thread> th;
    std::vector<std::optional<types::Detection>> out(4);
    for (int i = 0; i < 4; ++i)
        th.emplace_back([&, i] {
            const std::shared_ptr<const tk::ChanCapture> c = i % 2 ? synthetic_chan(true) : chan;
            out[static_cast<std::size_t>(i)] = lt::verify_l(c, T, det);
        });
    for (auto& t : th) t.join();
    for (const auto& o : out)
        check::is_true(o && v && o->z_ref == v->z_ref && o->timing == v->timing && o->path == v->path,
                       "a concurrent verify_l");
}

// --- resid_ratio on a synthetic track (bitwise) --------------------------------------------------

static void test_resid_ratio() {
    const auto& S = SHORT();
    const tk::LClasses cl = lt::make_classes_l(S);
    tk::LTrackResult tr(S, nullptr, known_timing(), tk::freq_path_spline(flat_path()));
    for (std::int64_t p = -cfg::SPAN; p < S.n_pos(); ++p) {
        const std::int64_t i = p + cfg::SPAN;
        tr.pos.push_back(p);
        tr.m.emplace_back(static_cast<double>((i * 7) % 13 - 6) * 0.125, static_cast<double>((i * 5) % 11 - 5) * 0.0625);
        tr.u.emplace_back(0.75 + static_cast<double>(i % 3) * 0.25, (static_cast<double>(i % 4) - 1.5) * 0.125);
        tr.P.push_back(0.001 * static_cast<double>(1 + i % 5));
        tr.N.push_back(0.25 + 0.0625 * static_cast<double>(i % 7));
        tr.psi.push_back(i % 97 == 0 ? 0.25 : 1.0);
    }
    tr.classes = cl;
    const auto tm = lt::data_templates(S, tr.pos, cl);
    check::equal(lt::resid_ratio(tr, tm.c, tm.gain, tm.dist), P::RR, "resid_ratio, bitwise");
    auto erased = tr;
    erased.psi.assign(erased.psi.size(), 0.0);
    check::equal(lt::resid_ratio(erased, tm.c, tm.gain, tm.dist), 1.0, "resid_ratio, no known symbol usable");
    auto ce = tr;
    ce.classes = std::monostate{};
    throws<std::invalid_argument>([&] { (void)lt::resid_ratio(ce, tm.c, tm.gain, tm.dist); },
                                  "resid_ratio: the track has no L classes", "resid_ratio, no L classes");
}

// --- refusals ----------------------------------------------------------------------------------------

static void test_refusals(const std::shared_ptr<const tk::ChanCapture>& chan) {
    const auto& ce = fr::get("tiny");
    const auto det = synthetic_det();
    throws<fr::WrongWaveform>([&] { (void)lt::fit_timing_l(std::span<const cd>(), cfg::CH_FS, ce, 0.0); },
                              P::ERR_FIT, "fit_timing_l, a CE frame");
    throws<fr::WrongWaveform>([&] { (void)lt::verify_l(chan, ce, det); }, P::ERR_VERIFY, "verify_l, a CE frame");
    throws<fr::WrongWaveform>([&] { (void)lt::verify_track_l(chan, ce, det); }, P::ERR_VTRACK,
                              "verify_track_l, a CE frame");
}

int main() {
    try {
        test_phases();
        test_statistic();
        test_grid();
        const std::shared_ptr<const tk::ChanCapture> chan = synthetic_chan(true);
        qrss_check::digest_is(chan->ch, P::CAP, "the capture");
        test_fit(chan);
        test_path(chan);
        test_gate(chan);
        test_resid_ratio();
        test_refusals(chan);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FATAL: %s\n", e.what());
        return 1;
    }
    return check::report("qrss_ltrack_timing");
}

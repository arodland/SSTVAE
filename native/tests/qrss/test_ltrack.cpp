// native/core/qrss/ltrack against values pinned from the Python reference
// (sstvae/qrss/ltrack.py). The pins are ltrack_pins.hpp, written by
// gen_ltrack_pins.py beside this file (re-run it when the reference
// changes on purpose); every input here is rebuilt from the formulas that
// script uses -- exact binary fractions, correctly rounded divisions, or a
// function the port computes to the bit (lwave::composite, frame::assemble_l)
// -- so both sides start from the same doubles. Digests are of the arrays'
// bytes, and they pin the functions ltrack.hpp lists as bitwise; the rest
// (pilot_filter, _pu_local, and the tracker runs built on them) is pinned
// to rounding, its decisions exactly. tests/test_qrss_native_ltrack.py
// compares against the reference directly, on real passes.

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
#include <vector>

#include "ltrack_pins.hpp"
#include "qrss/chan.hpp"
#include "qrss/config.hpp"
#include "qrss/frame.hpp"
#include "qrss/ltrack.hpp"
#include "qrss/lwave.hpp"
#include "qrss/rational.hpp"
#include "qrss/track.hpp"
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
namespace P = ltrack_pins;
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

// --- the exact-fraction inputs (the script's zc, zr, tm, bits, key) --------------------------------

struct Inputs {
    std::vector<double> zr;
    std::vector<cd> zc;
    types::Timing tm;
    std::vector<std::uint8_t> bits, key;
};

static Inputs inputs() {
    Inputs in;
    for (std::int64_t n = 0; n < 6000; ++n) {
        const double r = static_cast<double>((n * 7) % 19 - 9) / 16.0;
        in.zr.push_back(r);
        in.zc.emplace_back(r, static_cast<double>((n * 3) % 17 - 8) / 32.0);
    }
    in.tm.tau0 = 3.25;
    in.tm.ppm = 12.0;
    in.tm.gamma = 0.5;
    in.tm.cov = math::Matrix(3, 3);
    for (std::int64_t k = 0; k < 2474; ++k) in.bits.push_back(((k * 7 + 3) % 5) < 2 ? 1 : 0);
    for (std::int64_t k = 0; k < 192; ++k) in.key.push_back(((k * 5 + 2) % 7) < 3 ? 1 : 0);
    return in;
}

static const fr::FrameSpec& SHORT() { return fr::get_l("l-short"); }
static const fr::FrameSpec& TINY() { return fr::get_l("l-tiny"); }

static void test_tables() {
    const auto& t = lt::pilot_table();
    check::equal(t.sps(), P::TABLE_SPS, "pilot_table sps");
    check::equal(static_cast<std::int64_t>(t.half()), P::TABLE_HALF, "pilot_table half");
    check::equal(t.g0(), P::TABLE_G0, "pilot_table g0");
    digest_is(std::vector<double>(t.tab().begin(), t.tab().end()), P::TABLE_TAB, "pilot_table taps");
    digest_is(lt::pilot_taps(), P::TAPS, "_pilot_taps");
    // built once, and every thread sees the one table
    std::vector<const tk::MfTable*> seen(6, nullptr);
    std::vector<std::thread> th;
    for (std::size_t i = 0; i < seen.size(); ++i) th.emplace_back([&seen, i] { seen[i] = &lt::pilot_table(); });
    for (auto& x : th) x.join();
    for (std::size_t i = 0; i < seen.size(); ++i)
        check::is_true(seen[i] == &t, "pilot_table from thread " + std::to_string(i));
}

static void test_pilot_filter() {
    const Inputs in = inputs();
    std::vector<double> tp;
    for (std::int64_t k = -400; k < 7200; ++k) tp.push_back(static_cast<double>(k) * 0.875 + 0.0625);
    // to rounding: scipy's oaconvolve against pocketfft
    const auto y = lt::pilot_filter(std::span<const cd>(in.zc), tp);
    const auto yr = lt::pilot_filter(std::span<const double>(in.zr), tp);
    check::equal(y.size(), tp.size(), "pilot_filter size");
    for (std::size_t j = 0; j < std::size(P::PF_IDX); ++j) {
        const auto k = static_cast<std::size_t>(P::PF_IDX[j]);
        near(y[k].real(), P::PF_RE[j], 0.0, "pilot_filter re [" + std::to_string(k) + "]", 1e-13 * P::PF_SCALE);
        near(y[k].imag(), P::PF_IM[j], 0.0, "pilot_filter im [" + std::to_string(k) + "]", 1e-13 * P::PF_SCALE);
        near(yr[k], P::PFR[j], 0.0, "pilot_filter real [" + std::to_string(k) + "]", 1e-13 * P::PFR_SCALE);
    }
    check::is_true(y.front() == cd(0.0, 0.0) && y.back() == cd(0.0, 0.0), "pilot_filter: zero outside x");
    throws<std::out_of_range>(
        [&] { (void)lt::pilot_filter(std::span<const cd>(), std::span<const double>(tp).first(3)); },
        "index -1 is out of bounds for axis 0 with size 0", "pilot_filter, empty x");
    check::is_true(lt::pilot_filter(std::span<const cd>(in.zc), std::span<const double>()).empty(),
                   "pilot_filter, no tp");
}

static void test_mix_mf() {
    const Inputs in = inputs();
    const std::int64_t n_pos = SHORT().n_pos();
    digest_is(lt::data_mix(in.zc, -100.5, in.tm, n_pos), P::MIX_ALL, "data_mix");
    const std::vector<std::int64_t> idx{-5, 0, 7, 5999, 1234, -6000};
    digest_is(lt::data_mix(in.zc, -100.5, in.tm, n_pos, std::span<const std::int64_t>(idx)), P::MIX_IDX,
              "data_mix at idx");
    const std::vector<std::int64_t> bad{6000};
    throws<std::out_of_range>(
        [&] { (void)lt::data_mix(in.zc, -100.5, in.tm, n_pos, std::span<const std::int64_t>(bad)); },
        "index 6000 is out of bounds for axis 0 with size 6000", "data_mix, index past z_d");
    std::vector<double> pos;
    for (std::int64_t p = -20; p < 300; ++p) pos.push_back(static_cast<double>(p));
    digest_is(lt::data_mf(in.zc, -100.5, in.tm, pos, n_pos), P::MF_DENSE, "data_mf dense");
    const std::vector<double> sparse{0.0, 200.5, 377.25};
    digest_is(lt::data_mf(in.zc, -100.5, in.tm, sparse, n_pos), P::MF_SPARSE, "data_mf sparse");
    check::is_true(lt::data_mf(in.zc, -100.5, in.tm, std::vector<double>{}, n_pos).empty(), "data_mf, no pos");
    const auto far = lt::data_mf(in.zc, -100.5, in.tm, std::vector<double>{5000.0, 6000.0}, n_pos);
    check::is_true(far.size() == 2 && far[0] == cd(0.0, 0.0) && far[1] == cd(0.0, 0.0), "data_mf past z_d");
    throws<std::invalid_argument>(
        [&] {
            (void)lt::data_mf(in.zc, -100.5, in.tm, std::vector<double>{std::numeric_limits<double>::quiet_NaN()},
                              n_pos);
        },
        "cannot convert float NaN to integer", "data_mf, NaN position");
}

static void classes_are(const tk::LClasses& c, const char* mu, const char* nu, const char* known, const char* gain,
                        double rho, const std::string& what) {
    digest_is(c.mu, mu, what + " mu");
    digest_is(c.nu, nu, what + " nu");
    digest_is(c.known, known, what + " known");
    digest_is(c.gain, gain, what + " gain");
    check::equal(c.spread_rho, rho, what + " spread_rho");
}

static types::EmPrior prior_of(const fr::FrameSpec& spec) {
    types::EmPrior p;
    for (std::int64_t k = 0; k < spec.n_data(); ++k) {
        p.x_hat.push_back(static_cast<double>((k * 5) % 9 - 4) / 8.0);
        p.v.push_back(static_cast<double>((k * 3) % 7) / 5.0);
    }
    return p;
}

static void test_classes_templates() {
    const Inputs in = inputs();
    const auto& S = SHORT();
    const types::EmPrior prior = prior_of(S);
    const auto a = lt::make_classes_l(S);
    const auto b = lt::make_classes_l(S, &in.bits);
    classes_are(a, P::CL_A_MU, P::CL_A_NU, P::CL_A_KNOWN, P::CL_A_GAIN, P::CL_A_RHO, "classes A");
    classes_are(b, P::CL_B_MU, P::CL_B_NU, P::CL_B_KNOWN, P::CL_B_GAIN, P::CL_B_RHO, "classes B");
    classes_are(lt::make_classes_l(S, &in.bits, &prior), P::CL_C_MU, P::CL_C_NU, P::CL_C_KNOWN, P::CL_C_GAIN,
                P::CL_C_RHO, "classes B, prior");
    classes_are(lt::make_classes_l(S, nullptr, &prior), P::CL_D_MU, P::CL_D_NU, P::CL_D_KNOWN, P::CL_D_GAIN,
                P::CL_D_RHO, "classes A, prior");
    classes_are(lt::make_classes_l(TINY()), P::CL_TINY_MU, P::CL_TINY_NU, P::CL_TINY_KNOWN, P::CL_TINY_GAIN,
                P::CL_TINY_RHO, "classes l-tiny");
    check::is_true(!a.hdr_known && !a.hdr_bits && b.hdr_known && b.hdr_bits && *b.hdr_bits == in.bits,
                   "classes hdr flags");
    digest_is(lt::distortion(S), P::DIST, "_distortion");
    std::vector<std::int64_t> grid;
    for (std::int64_t p = -40; p < S.n_pos(); ++p) grid.push_back(p);
    auto t = lt::data_templates(S, grid, b, false);
    digest_is(t.c, P::TM_C, "data_templates c");
    digest_is(t.nu, P::TM_NU, "data_templates nu");
    digest_is(t.gain, P::TM_GAIN, "data_templates gain");
    digest_is(t.dist, P::TM_DIST, "data_templates dist");
    digest_is(t.is_sym, P::TM_SYM, "data_templates is_sym");
    t = lt::data_templates(S, grid, b, true);
    digest_is(t.c, P::TMW_C, "data_templates c, withheld");
    digest_is(t.nu, P::TMW_NU, "data_templates nu, withheld");
    digest_is(t.gain, P::TMW_GAIN, "data_templates gain, withheld");
    digest_is(t.dist, P::TMW_DIST, "data_templates dist, withheld");
    digest_is(t.is_sym, P::TMW_SYM, "data_templates is_sym, withheld");
    throws<std::out_of_range>([&] { (void)lt::data_templates(S, std::span<const std::int64_t>(), b); },
                              "index 0 is out of bounds for axis 0 with size 0", "data_templates, no grid");
    // known symbols
    const auto ks = lt::known_symbols_l(S);
    const auto ksb = lt::known_symbols_l(S, &in.bits);
    const auto kt = lt::known_symbols_l(TINY());
    check::equal(static_cast<std::int64_t>(ks.pos.size()), P::KN_S_N, "known_symbols_l size");
    check::equal(static_cast<std::int64_t>(ksb.pos.size()), P::KN_SB_N, "known_symbols_l size, bits");
    check::equal(static_cast<std::int64_t>(kt.pos.size()), P::KN_T_N, "known_symbols_l size, l-tiny");
    digest_is(ks.pos, P::KN_S_POS, "known_symbols_l pos");
    digest_is(ks.cplx(), P::KN_S_VAL, "known_symbols_l val");
    digest_is(ksb.pos, P::KN_SB_POS, "known_symbols_l pos, bits");
    digest_is(ksb.cplx(), P::KN_SB_VAL, "known_symbols_l val, bits");
    digest_is(kt.pos, P::KN_T_POS, "known_symbols_l pos, l-tiny");
    digest_is(kt.cplx(), P::KN_T_VAL, "known_symbols_l val, l-tiny");
    // refusals
    throws<fr::WrongWaveform>([] { (void)lt::make_classes_l(fr::get("tiny")); },
                              "ltrack.make_classes_l is L's; 'tiny' is waveform 0: use track's", "make_classes_l, CE");
    throws<fr::WrongWaveform>([] { (void)lt::known_symbols_l(fr::get("short")); },
                              "ltrack.known_symbols_l is L's; 'short' is waveform 0: use track's",
                              "known_symbols_l, CE");
    const std::vector<std::uint8_t> odd(2473, 1);
    throws<std::invalid_argument>([&] { (void)lt::make_classes_l(S, &odd); },
                                  "qpsk takes an even number of bits, not shape (2473,)", "make_classes_l, odd bits");
}

static void test_pilot_wave() {
    const Inputs in = inputs();
    const auto& S = SHORT();
    std::vector<double> tau;
    for (std::int64_t k = -2400; k < 2 * S.n_pos() + 400; ++k) tau.push_back(static_cast<double>(k) * 0.5);
    digest_is(lt::pilot_gain(tau, S), P::PG_0, "_pilot_gain");
    digest_is(lt::pilot_gain(tau, S, 5.0), P::PG_5, "_pilot_gain, lead-in");
    const lw::Keying key = lw::Keying::units(in.key);
    digest_is(lt::pilot_wave(tau, S), P::PW_0, "pilot_wave");
    digest_is(lt::pilot_wave(tau, S, &key), P::PW_K, "pilot_wave, keyed");
    digest_is(lt::pilot_wave(tau, S, &key, 5.0), P::PW_K5, "pilot_wave, keyed, lead-in");
    digest_is(lt::pilot_wave(tau, TINY(), nullptr, 2.5), P::PW_TINY, "pilot_wave, l-tiny");
    near(lt::keyed_power_db(TINY()), P::KEYED_DB_TINY, 1e-15, "keyed_power_db");
}

static void test_q_near() {
    const auto qg = tk::q_grid(cfg::L_T_SYM);
    digest_is(std::vector<double>(qg.begin(), qg.end()), P::Q_GRID, "q_grid(L_T_SYM)");
    for (std::size_t j = 0; j < std::size(P::QN_Q); ++j) {
        const auto [lo, hi] = lt::q_near_span(P::QN_Q[j], qg);
        check::equal(static_cast<std::int64_t>(lo), P::QN_SPAN[2 * j], "_q_near lo " + std::to_string(j));
        check::equal(static_cast<std::int64_t>(hi), P::QN_SPAN[2 * j + 1], "_q_near hi " + std::to_string(j));
        const auto near_q = lt::q_near(P::QN_Q[j], qg);
        check::is_true(near_q == std::vector<double>(qg.begin() + static_cast<std::ptrdiff_t>(lo),
                                                     qg.begin() + static_cast<std::ptrdiff_t>(hi)),
                       "_q_near values " + std::to_string(j));
    }
    throws<std::invalid_argument>([&] { (void)lt::q_near(0.0, qg); }, "math domain error", "_q_near, q = 0");
    throws<std::invalid_argument>([&] { (void)lt::q_near(1e-4, std::span<const double>()); },
                                  "attempt to get argmin of an empty sequence", "_q_near, no grid");
}

// --- the model on a synthetic track (the script's synth_track) ----------------------------------

static tk::PathFn flat_path() {
    const std::vector<double> t{-10.0, 0.0, 1800.0}, f{1500.0, 1500.0, 1500.0}, w{1.0, 1.0, 1.0};
    return tk::PathFn(t, f, std::span<const double>(w));
}

static tk::LTrackResult synth_track(const tk::LClasses& cl, const types::Timing& tm) {
    const auto& S = SHORT();
    tk::LTrackResult tr(S, nullptr, tm, flat_path());
    std::vector<cd> m_pl, c_pl;
    std::vector<double> N_pl, psi_pl;
    for (std::int64_t p = -cfg::SPAN; p < S.n_pos(); ++p) {
        const std::int64_t i = p + cfg::SPAN;
        tr.pos.push_back(p);
        tr.m.emplace_back(static_cast<double>((i * 7) % 13 - 6) * 0.125,
                          static_cast<double>((i * 5) % 11 - 5) * 0.0625);
        tr.N.push_back(0.25 + 0.0625 * static_cast<double>(i % 7));
        tr.psi.push_back(i % 97 == 0 ? 0.25 : 1.0);
        m_pl.emplace_back(static_cast<double>((i * 3) % 11 - 5) * 0.0625, static_cast<double>((i * 7) % 9 - 4) * 0.125);
        c_pl.emplace_back(0.375, static_cast<double>((i % 5) - 2) * 0.0625);
        N_pl.push_back(0.125 + 0.03125 * static_cast<double>(i % 5));
        psi_pl.push_back(i % 89 == 0 ? 0.125 : 1.0);
    }
    tr.m_pl = m_pl;
    tr.c_pl = c_pl;
    tr.N_pl = N_pl;
    tr.psi_pl = psi_pl;
    tr.classes = cl;
    return tr;
}

struct ModelPins {
    const char *c_all, *c, *nu, *gain, *dist, *k_pl, *is_sym, *is_known, *meas_d, *need, *meas_p, *measured;
    const char *s[4], *sd[4], *f[2], *fd[2], *fp[2], *fn[2];
    double pu0;
    const double* delta;
};

static void model_is(const lt::Model& md, const tk::LTrackResult& tr, std::span<const double> pu,
                     const std::vector<std::uint8_t>& mask, std::span<const cd> v, std::span<const double> Pv,
                     const ModelPins& p, const std::string& what) {
    digest_is(md.c_all, p.c_all, what + " c_all");
    digest_is(md.c, p.c, what + " c");
    digest_is(md.nu, p.nu, what + " nu");
    digest_is(md.gain, p.gain, what + " gain");
    digest_is(md.dist, p.dist, what + " dist");
    digest_is(md.k_pl, p.k_pl, what + " k_pl");
    digest_is(md.is_sym, p.is_sym, what + " is_sym");
    digest_is(md.is_known, p.is_known, what + " is_known");
    digest_is(md.meas_d, p.meas_d, what + " meas_d");
    digest_is(md.need, p.need, what + " need");
    digest_is(md.meas_p, p.meas_p, what + " meas_p");
    digest_is(md.measured, p.measured, what + " measured");
    check::is_true(tr.meas == md.meas_d && tr.meas_pl && *tr.meas_pl == md.meas_p, what + " sets tr.meas, meas_pl");
    auto s = md.streams(tr, pu);
    digest_is(s.mu_d, p.s[0], what + " streams mu_d");
    digest_is(s.R_d, p.s[1], what + " streams R_d");
    digest_is(s.mu_p, p.s[2], what + " streams mu_p");
    digest_is(s.R_p, p.s[3], what + " streams R_p");
    s = md.streams(tr, pu, cd(0.75, 0.5), true, true, &mask);
    digest_is(s.mu_d, p.sd[0], what + " streams(delta, mask) mu_d");
    digest_is(s.R_d, p.sd[1], what + " streams(delta, mask) R_d");
    digest_is(s.mu_p, p.sd[2], what + " streams(delta, mask) mu_p");
    digest_is(s.R_p, p.sd[3], what + " streams(delta, mask) R_p");
    auto f = md.fused(tr, pu);
    digest_is(f.mu, p.f[0], what + " fused mu");
    digest_is(f.R, p.f[1], what + " fused R");
    f = md.fused(tr, pu, cd(0.75, 0.5), true, true, &mask);
    digest_is(f.mu, p.fd[0], what + " fused(delta, mask) mu");
    digest_is(f.R, p.fd[1], what + " fused(delta, mask) R");
    f = md.fused(tr, pu, cd(1.0, 0.0), false);
    digest_is(f.mu, p.fp[0], what + " fused(pilot only) mu");
    digest_is(f.R, p.fp[1], what + " fused(pilot only) R");
    f = md.fused(tr, pu, cd(1.0, 0.0), true, false);
    digest_is(f.mu, p.fn[0], what + " fused(data only) mu");
    digest_is(f.R, p.fn[1], what + " fused(data only) R");
    check::equal(md.pu_initial(tr), p.pu0, what + " pu_initial");
    const cd d0s[3] = {cd(1.0, 0.0), cd(0.75, 0.5), cd(-1.25, -0.25)};
    for (int j = 0; j < 3; ++j) {
        const cd r = lt::fit_delta(md, tr, v, Pv, pu, d0s[j]);
        check::equal(r.real(), p.delta[2 * j], what + " _fit_delta re " + std::to_string(j));
        check::equal(r.imag(), p.delta[2 * j + 1], what + " _fit_delta im " + std::to_string(j));
    }
}

static void test_model() {
    const Inputs in = inputs();
    const auto& S = SHORT();
    const auto a = lt::make_classes_l(S);
    const auto b = lt::make_classes_l(S, &in.bits);
    const lw::Keying key = lw::Keying::units(in.key);
    std::vector<double> pu, Pv;
    std::vector<cd> v;
    std::vector<std::uint8_t> mask;
    for (std::int64_t i = 0; i < S.n_pos() + cfg::SPAN; ++i) {
        pu.push_back(0.5 + 0.125 * static_cast<double>(i % 3));
        v.emplace_back(0.75 + static_cast<double>(i % 3) * 0.25, (static_cast<double>(i % 4) - 1.5) * 0.125);
        Pv.push_back(0.001 * static_cast<double>(1 + i % 5));
        mask.push_back(i % 4 != 1 ? 1 : 0);
    }
    {
        auto tr = synth_track(a, in.tm);
        const lt::Model md(tr, S, a, false, nullptr, 0.0, 0);
        const ModelPins p{P::M1_C_ALL, P::M1_C, P::M1_NU, P::M1_GAIN, P::M1_DIST, P::M1_K_PL, P::M1_IS_SYM,
                          P::M1_IS_KNOWN, P::M1_MEAS_D, P::M1_NEED, P::M1_MEAS_P, P::M1_MEASURED,
                          {P::M1_S_MUD, P::M1_S_RD, P::M1_S_MUP, P::M1_S_RP},
                          {P::M1_SD_MUD, P::M1_SD_RD, P::M1_SD_MUP, P::M1_SD_RP},
                          {P::M1_F_MU, P::M1_F_R}, {P::M1_FD_MU, P::M1_FD_R}, {P::M1_FP_MU, P::M1_FP_R},
                          {P::M1_FN_MU, P::M1_FN_R}, P::M1_PU0, P::M1_DELTA};
        model_is(md, tr, pu, mask, v, Pv, p, "model 1");
        // _pu_local: to rounding (track's _smooth sums through BLAS)
        for (int pass = 0; pass < 2; ++pass) {
            std::vector<double> P1000;
            for (const double x : Pv) P1000.push_back(1000 * x);
            const auto none = std::vector<std::uint8_t>(md.measured.size(), 0);
            const auto pl = lt::pu_local(v, P1000, pass ? std::span<const std::uint8_t>(none) : md.measured);
            const double* want = pass ? P::PLN_LOC : P::PL_LOC;
            for (std::size_t j = 0; j < std::size(P::PL_IDX); ++j)
                near(pl.loc[static_cast<std::size_t>(P::PL_IDX[j])], want[j], 1e-13,
                     std::string("_pu_local") + (pass ? ", none measured " : " ") + std::to_string(j));
            near(pl.g, pass ? P::PLN_G : P::PL_G, 1e-13, pass ? "_pu_local g, none measured" : "_pu_local g");
        }
        // a track without its pilot stream
        auto bare = tr;
        bare.m_pl.reset();
        throws<math::TypeError>([&] { (void)md.streams(bare, pu); }, "'NoneType' object is not subscriptable",
                                "streams, no m_pl");
    }
    {
        auto tr = synth_track(b, in.tm);
        const lt::Model md(tr, S, b, true, &key, 0.0, 1);
        const ModelPins p{P::M2_C_ALL, P::M2_C, P::M2_NU, P::M2_GAIN, P::M2_DIST, P::M2_K_PL, P::M2_IS_SYM,
                          P::M2_IS_KNOWN, P::M2_MEAS_D, P::M2_NEED, P::M2_MEAS_P, P::M2_MEASURED,
                          {P::M2_S_MUD, P::M2_S_RD, P::M2_S_MUP, P::M2_S_RP},
                          {P::M2_SD_MUD, P::M2_SD_RD, P::M2_SD_MUP, P::M2_SD_RP},
                          {P::M2_F_MU, P::M2_F_R}, {P::M2_FD_MU, P::M2_FD_R}, {P::M2_FP_MU, P::M2_FP_R},
                          {P::M2_FN_MU, P::M2_FN_R}, P::M2_PU0, P::M2_DELTA};
        model_is(md, tr, pu, mask, v, Pv, p, "model 2");
    }
    // _hold_ends
    std::vector<double> R;
    const std::int64_t n = S.n_pos() + cfg::SPAN;
    for (std::int64_t i = 0; i < n; ++i)
        R.push_back((i < 37 || i > n - 50 || i % 11 == 0) ? std::numeric_limits<double>::infinity() : 1.0);
    auto hu = v;
    auto hP = Pv;
    check::is_true(lt::hold_ends(hu, hP, R), "_hold_ends");
    digest_is(hu, P::HOLD_U, "_hold_ends u");
    digest_is(hP, P::HOLD_P, "_hold_ends P");
    hu = v;
    hP = Pv;
    check::is_true(!lt::hold_ends(hu, hP, std::vector<double>(R.size(), std::numeric_limits<double>::infinity())) &&
                       hu == v && hP == Pv,
                   "_hold_ends, nothing measured");
}

// --- synthetic l-tiny passes ---------------------------------------------------------------------
//
// The script's capture(lead): l-tiny's symbols for latents +-0.5, through
// lwave.composite at 250 Hz from 12 s before t0, plus a structured
// "noise" ((i*7 % 19 - 9) + j (i*3 % 17 - 8)) / 16, as complex64.

constexpr std::int64_t N_CH = 17500;

static std::shared_ptr<tk::ChanCapture> synthetic_chan(double lead) {
    const auto& T = TINY();
    std::vector<double> x;
    for (std::int64_t k = 0; k < T.n_data(); ++k) x.push_back((k * 5 + 1) % 3 == 0 ? 0.5 : -0.5);
    const auto sym = fr::assemble_l(T, nullptr, x);
    const auto sig = lw::composite(sym, T, cfg::CH_FS, -3000, N_CH, nullptr, lead);
    auto chan = std::make_shared<tk::ChanCapture>();
    for (std::int64_t i = 0; i < N_CH; ++i) {
        const cd z = sig[static_cast<std::size_t>(i)] + cd(static_cast<double>((i * 7) % 19 - 9) / 16.0,
                                                           static_cast<double>((i * 3) % 17 - 8) / 16.0);
        chan->ch.emplace_back(static_cast<float>(z.real()), static_cast<float>(z.imag()));
    }
    chan->fs = cfg::CH_FS;
    chan->t0_index = 3000.0;
    chan->f_mix = qrss::Rational(0);
    return chan;
}

static types::Detection synthetic_det(double lead) {
    types::Detection det;
    det.f_hz = 1500.0;
    det.path = types::FreqPath{{-10.0, 0.0, 1800.0}, {1500.0, 1500.0, 1500.0}, {1.0, 1.0, 1.0}};
    types::Timing tm;
    tm.tau0 = 0.75;
    tm.ppm = 20.0;
    tm.cov = math::Matrix(3, 3);
    det.timing = tm;
    det.z_ref = std::numeric_limits<double>::quiet_NaN();
    det.method = "tbd";
    det.lead_in_s = lead;
    det.waveform = cfg::WAVEFORM_L;
    return det;
}

static void cplx_near(const std::vector<cd>& got, const double* want, double rel, const std::string& what) {
    for (std::size_t j = 0; j < std::size(P::TR_IDX); ++j) {
        const cd g = got[static_cast<std::size_t>(P::TR_IDX[j])];
        const double scale = std::abs(cd(want[2 * j], want[2 * j + 1]));
        near(g.real(), want[2 * j], 0.0, what + " re " + std::to_string(j), rel * scale);
        near(g.imag(), want[2 * j + 1], 0.0, what + " im " + std::to_string(j), rel * scale);
    }
}

static void test_synthetic_pass() {
    const auto chan = synthetic_chan(0.0);
    qrss_check::digest_is(chan->ch, P::CAP, "the capture");
    const auto& T = TINY();
    const auto det = synthetic_det(0.0);
    const tk::LTrackResult tr = lt::track_l(chan, T, det);
    // every decision is the reference's; the values to rounding (pilot_filter,
    // _smooth's BLAS sum and the refinement's linear algebra): ~1e-10 of the
    // arrays' scale measured, at the frame's last position where the timing's
    // rounding reaches furthest
    check::equal(tr.q, P::TR_Q, "q");
    check::equal(tr.lead_in_s, P::TR_LEAD, "lead_in_s");
    check::is_true(!tr.ook && !tr.keying, "no keying");
    check::equal(static_cast<std::int64_t>(tr.pos.size()), P::TR_NPOS, "grid");
    check::equal(tr.pos.front(), P::TR_P0, "grid start");
    digest_is(tr.meas, P::TR_MEAS, "meas");
    check::is_true(tr.meas_pl.has_value(), "meas_pl");
    if (tr.meas_pl) digest_is(*tr.meas_pl, P::TR_MEAS_PL, "meas_pl");
    digest_is(tr.c, P::TR_C, "c");
    near(tr.loglik, P::TR_LOGLIK, 1e-11, "loglik");
    near(tr.df_max, P::TR_DF_MAX, 1e-7, "df_max");
    near(tr.timing.tau0, P::TR_TIMING[0], 1e-9, "timing tau0");
    near(tr.timing.ppm, P::TR_TIMING[1], 1e-10, "timing ppm");
    check::equal(tr.timing.gamma, P::TR_TIMING[2], "timing gamma");
    near(tr.timing.z, P::TR_TIMING[3], 1e-11, "timing z");
    near(tr.delta.real(), P::TR_DELTA[0], 1e-11, "delta re");
    near(tr.delta.imag(), P::TR_DELTA[1], 1e-9, "delta im");
    cplx_near(tr.u, P::TR_U, 1e-9, "u");
    for (std::size_t j = 0; j < std::size(P::TR_IDX); ++j)
        near(tr.P[static_cast<std::size_t>(P::TR_IDX[j])], P::TR_P[j], 1e-10, "P " + std::to_string(j));
    check::is_true(tr.u_pl.has_value(), "u_pl");
    if (tr.u_pl) cplx_near(*tr.u_pl, P::TR_UPL, 1e-9, "u_pl");
    if (tr.m_pl) cplx_near(*tr.m_pl, P::TR_MPL, 1e-9, "m_pl");
    cplx_near(tr.m, P::TR_M, 1e-9, "m");
    near(tr.freq(0.0), P::TR_FREQ[0], 1e-14, "freq(0)");
    near(tr.freq(20.0), P::TR_FREQ[1], 1e-14, "freq(20)");
    near(tr.freq(50.0), P::TR_FREQ[2], 1e-14, "freq(50)");
    const auto rt = lt::refine_timing_l(tr);
    check::is_true(rt.has_value(), "refine_timing_l");
    if (rt) {
        near(rt->tau0, P::RT[0], 1e-9, "refined tau0");
        near(rt->ppm, P::RT[1], 1e-10, "refined ppm");
        check::equal(rt->gamma, P::RT[2], "refined gamma");
        near(rt->z, P::RT[3], 1e-10, "refined z");
    }
    const auto rep = lt::report_l(tr, 0.9, false);
    near(rep.offset_hz, P::REP[0], 1e-14, "report offset");
    // a drift of zero to rounding: the path is flat at 1500 Hz, one ulp of which is 2.3e-13
    near(rep.drift_hz_per_min, P::REP[1], 0.0, "report drift", 1e-10);
    near(rep.wander_hz_rms, P::REP[2], 1e-8, "report wander");
    near(rep.doppler_hz, P::REP[3], 1e-14, "report doppler");
    near(rep.ppm, P::REP[4], 1e-10, "report ppm");
    near(rep.snr2500_db, P::REP[5], 1e-11, "report snr");
    check::is_true(std::isnan(rep.z_ref) && rep.kappa == 0.9 && !rep.suspect, "report flags");
    near(lt::report_l(tr, 0.9, false, tk::noise_floor(*chan)).snr2500_db, P::REP[5], 1e-11, "report, floor given");
    // gate V's run: the known symbols withheld, delta 1, the timing unrefined
    lt::TrackLArgs wk;
    wk.withhold_known = true;
    const auto tw = lt::track_l(chan, T, det, wk);
    check::equal(tw.q, P::TW_Q, "withheld q");
    digest_is(tw.meas, P::TW_MEAS, "withheld meas");
    near(tw.loglik, P::TW_LOGLIK, 1e-11, "withheld loglik");
    check::is_true(tw.timing.tau0 == P::TW_TIMING[0] && tw.timing.ppm == P::TW_TIMING[1], "withheld timing");
    check::is_true(tw.delta == cd(1.0, 0.0), "withheld delta");
    cplx_near(tw.u, P::TW_U, 1e-9, "withheld u");
    lt::TrackLArgs nr;
    nr.refine = false;
    nr.outer = 0;
    const auto tn = lt::track_l(chan, T, det, nr);
    check::equal(tn.q, P::TN_Q, "unrefined q");
    near(tn.loglik, P::TN_LOGLIK, 1e-11, "unrefined loglik");
    near(tn.delta.real(), P::TN_DELTA[0], 1e-11, "unrefined delta re");
    near(tn.delta.imag(), P::TN_DELTA[1], 1e-10, "unrefined delta im");
    check::is_true(tn.timing.tau0 == 0.75 && tn.timing.ppm == 20.0, "unrefined timing");
    // reentrant: threads tracking the one capture get the one answer -- on the
    // capture whose noise table is filled, and on a fresh one whose empty
    // NoisePsdSlot they race to fill
    const std::shared_ptr<const tk::ChanCapture> fresh = synthetic_chan(0.0);
    check::is_true(!fresh->psd.get(), "a fresh capture has no noise table yet");
    std::vector<std::thread> th;
    std::vector<std::optional<tk::LTrackResult>> out(4);
    for (int i = 0; i < 4; ++i)
        th.emplace_back([&, i] { out[static_cast<std::size_t>(i)] = lt::track_l(i % 2 ? fresh : chan, T, det); });
    for (auto& t : th) t.join();
    for (const auto& o : out)
        check::is_true(o && o->q == tr.q && o->u == tr.u && o->P == tr.P && o->loglik == tr.loglik && o->N == tr.N &&
                           o->delta == tr.delta && o->timing.tau0 == tr.timing.tau0 && o->u_pl == tr.u_pl,
                       "a concurrent track_l");
    check::is_true(fresh->psd.get() != nullptr, "the race filled the fresh capture's table");
    // refusals
    auto no_timing = det;
    no_timing.timing.reset();
    throws<std::invalid_argument>([&] { (void)lt::track_l(chan, T, no_timing); },
                                  "track_l needs a timing (fit_timing_l, or gate V's)", "track_l, no timing");
    lt::TrackLArgs ook;
    ook.ook = true;
    throws<std::invalid_argument>([&] { (void)lt::track_l(chan, T, det, ook); },
                                  "waveform L keys its callsign by FSK only: cw_known must say ook=False",
                                  "track_l, ook");
    throws<fr::WrongWaveform>([&] { (void)lt::track_l(chan, fr::get("tiny"), det); },
                              "ltrack.track_l is L's; 'tiny' is waveform 0: use track's", "track_l, CE frame");
}

static void test_lead_in() {
    const auto& T = TINY();
    const auto with = synthetic_chan(3.0);
    const auto without = synthetic_chan(0.0);
    qrss_check::digest_is(with->ch, P::CAP_LEAD, "the capture with a lead-in");
    const auto det = synthetic_det(3.0);
    const auto tl = lt::track_l(with, T, det);
    check::equal(tl.lead_in_s, P::TL_LEAD, "lead-in kept");
    check::equal(tl.pos.front(), P::TL_P0, "grid from the lead-in");
    check::equal(tl.q, P::TL_Q, "lead-in q");
    if (tl.meas_pl) digest_is(*tl.meas_pl, P::TL_MEAS_PL, "lead-in meas_pl");
    near(tl.timing.tau0, P::TL_TIMING[0], 1e-9, "lead-in tau0");
    near(tl.timing.ppm, P::TL_TIMING[1], 1e-10, "lead-in ppm");
    const auto td = lt::track_l(without, T, det);
    check::equal(td.lead_in_s, P::TD_LEAD, "an unsupported lead-in dropped");
    check::equal(td.pos.front(), P::TD_P0, "grid without it");
    const tk::PathFn flat = flat_path();
    const auto z_lead = tk::derotate(*with, flat);
    const auto z0 = tk::derotate(*without, flat);
    const types::Timing tm = *det.timing;
    check::equal(lt::lead_supported_l(*with, z_lead, tm, T, 3.0), P::LS_LEAD != 0, "_lead_supported_l, there");
    check::equal(lt::lead_supported_l(*without, z0, tm, T, 3.0), P::LS_NONE != 0, "_lead_supported_l, absent");
    check::is_true(lt::lead_supported_l(*without, z0, tm, T, 0.5), "_lead_supported_l, too short to tell");
}

int main() {
    try {
        test_tables();
        test_pilot_filter();
        test_mix_mf();
        test_classes_templates();
        test_pilot_wave();
        test_q_near();
        test_model();
        test_synthetic_pass();
        test_lead_in();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FATAL: %s\n", e.what());
        return 1;
    }
    return check::report("qrss_ltrack");
}

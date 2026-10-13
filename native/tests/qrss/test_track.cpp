// native/core/qrss/track against values pinned from the Python reference
// (sstvae/qrss/track.py). Inputs are exact binary fractions, so Python
// makes the same doubles:
//
//   n = np.arange(6000); zr = ((n * 7) % 19 - 9) / 16.0
//   zc = zr + 1j * (((n * 3) % 17 - 8) / 32.0)
//   tm = Timing(tau0=3.25, ppm=12.0, gamma=0.5, z=0.0, cov=np.zeros((3, 3)))
//   pos = np.arange(-20, 300).astype(np.float64)
//   track.symbol_mf(zc, CH_FS, tm, pos, t0_index=-100.5, n_pos=frame.TINY.n_pos)    (and zr)
//   track.psi_at(np.where(n % 29 == 0, 0.0, 1.0), tm, pos, t0_index=-100.5, n_pos=...)
//   chan = ChanCapture(ch=zc.astype(np.complex64), fs=CH_FS, t0_index=-100.5, f_mix=Fraction(-3, 4))
//   track.derotate(chan, lambda t: 1500.25 + 0.001 * t)
//   mu = zc[:3000]; R = np.where(n[:3000] % 5 == 0, np.inf, 0.5 + (n[:3000] % 3) * 0.25)
//   track.kalman_rts(mu, R, 1e-4, 1.0[, causal=True]); track.select_q(mu, R, 1.0)
//   bits = (((np.arange(2474) * 7 + 3) % 5) < 2).astype(np.uint8)
//   track.known_symbols(frame.SHORT, bits); track.make_classes(frame.SHORT[, bits])
//   k = np.arange(frame.TINY.n_sym); mu = ((k * 5) % 9 - 4) / 4.0; nu = ((k * 3) % 7) / 8.0
//   track.templates(frame.TINY, np.arange(-30, 600), mu, nu)
//
// and a synthetic TINY pass, a CE signal at +0.02 Hz over a structured
// "noise" (the script is beside test_synthetic_pass). Digests are of the
// arrays' bytes (hashlib.sha256(a.tobytes())); the bitwise ones are the
// functions track.hpp lists as bitwise. tests/test_qrss_native_track.py
// compares against the reference directly, on real passes.

#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <memory>
#include <numbers>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "qrss/ce.hpp"
#include "qrss/frame.hpp"
#include "qrss/rational.hpp"
#include "qrss/track.hpp"
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

static void near(double got, double want, double rel, const std::string& what) {
    check::is_true(std::abs(got - want) <= rel * std::abs(want), what);
    if (!(std::abs(got - want) <= rel * std::abs(want)))
        std::fprintf(stderr, "   %s: got %.17g want %.17g\n", what.c_str(), got, want);
}

struct Inputs {
    std::vector<double> zr;
    std::vector<cd> zc;
    std::vector<double> pos;
    types::Timing tm;
};

static Inputs inputs() {
    Inputs in;
    for (std::int64_t n = 0; n < 6000; ++n) {
        const double r = static_cast<double>((n * 7) % 19 - 9) / 16.0;
        in.zr.push_back(r);
        in.zc.emplace_back(r, static_cast<double>((n * 3) % 17 - 8) / 32.0);
    }
    for (std::int64_t p = -20; p < 300; ++p) in.pos.push_back(static_cast<double>(p));
    in.tm.tau0 = 3.25;
    in.tm.ppm = 12.0;
    in.tm.gamma = 0.5;
    in.tm.cov = math::Matrix(3, 3);
    return in;
}

static std::vector<std::uint8_t> header_bits() {
    std::vector<std::uint8_t> b;
    for (std::int64_t k = 0; k < 2474; ++k) b.push_back(((k * 7 + 3) % 5) < 2 ? 1 : 0);
    return b;
}

static void test_tables() {
    const auto& t = tk::mf_table(qrss::config::T_SYM);
    check::equal(t.sps(), 7.578125, "mf_table sps");
    check::equal(t.half(), 62, "mf_table half");
    check::equal(t.g0(), 0x1.006d1ed38b121p+0, "mf_table g0");
    const std::span<const double> tab = t.tab();
    digest_is(std::vector<double>(tab.begin(), tab.end()),
              "96948942f8ea15676533a9e4b28e0240197a19c268995897419edcf12435ccaf", "mf_table taps");
    check::is_true(&tk::mf_table(qrss::config::T_SYM) == &t, "mf_table is built once");
    const auto& tl = tk::mf_table(2 * qrss::config::T_SYM);    // L's symbol, 0.060625 s
    check::equal(tl.half(), 123, "L mf_table half");
    check::equal(tl.g0(), 0x1.006cb85d5f6d6p+0, "L mf_table g0");
    digest_is(std::vector<double>(tl.tab().begin(), tl.tab().end()),
              "29dc05100aa2173abbd424002a3cc0b84224f3f01d209f5649ed037898e09218", "L mf_table taps");
    check::equal(tk::pulse_g0(qrss::config::T_SYM), 0x1.006d1ed38b121p+0, "pulse_g0");
    digest_is(tk::mf_taps(qrss::config::CH_FS, qrss::config::T_SYM),
              "cf40a862c38d4afc1ae0e231523aab2c6df167d03f5bdadcd5140f8e221f6d25", "mf_taps");
    digest_is(tk::mf_taps(qrss::config::CH_FS, 2 * qrss::config::T_SYM),
              "515869c657d0014727aed122240850156be4aa40dde6df2158693ce218edf426", "L mf_taps");
    // the cache under contention: every thread sees the one table
    std::vector<const tk::MfTable*> seen(8, nullptr);
    std::vector<std::thread> th;
    for (std::size_t i = 0; i < seen.size(); ++i)
        th.emplace_back([&seen, i] { seen[i] = &tk::mf_table(i % 2 ? 2 * qrss::config::T_SYM : qrss::config::T_SYM); });
    for (auto& x : th) x.join();
    for (std::size_t i = 0; i < seen.size(); ++i)
        check::is_true(seen[i] == (i % 2 ? &tl : &t), "mf_table from thread " + std::to_string(i));
}

static void test_mf_psi_derotate() {
    const Inputs in = inputs();
    const std::int64_t n_pos = fr::get("tiny").n_pos();
    digest_is(tk::symbol_mf(std::span<const cd>(in.zc), in.tm, in.pos, -100.5, n_pos),
              "01400c13c199b2c9b90335445a34d0aa3e23b4a16ebedbd03b1e3ba27550a611", "symbol_mf complex");
    digest_is(tk::symbol_mf(std::span<const double>(in.zr), in.tm, in.pos, -100.5, n_pos),
              "7b9be1963a24c95673e24d325c5c273993c48e589aac53559e210578aaed1b50", "symbol_mf real");
    std::vector<double> keep;
    for (std::int64_t n = 0; n < 6000; ++n) keep.push_back(n % 29 == 0 ? 0.0 : 1.0);
    digest_is(tk::psi_at(std::span<const double>(keep), in.tm, in.pos, -100.5, n_pos),
              "4ebbb4a32d44d7ad08db6e30add7465026318f039a56ada2c1892649ed57236d", "psi_at");
    tk::ChanCapture chan;
    for (const cd& z : in.zc) chan.ch.emplace_back(static_cast<float>(z.real()), static_cast<float>(z.imag()));
    chan.fs = qrss::config::CH_FS;
    chan.t0_index = -100.5;
    chan.f_mix = qrss::Rational(-3, 4);
    const tk::PathEval f = [](std::span<const double> t) {
        std::vector<double> out;
        for (const double x : t) out.push_back(1500.25 + 0.001 * x);
        return out;
    };
    digest_is(tk::derotate(chan, f), "de7c9223b415c232431e3e597f9a3cd60f725373c43d622ccafdf8c25da43699", "derotate");
}

static void test_filters() {
    const Inputs in = inputs();
    // to rounding: scipy's FFT convolution and np.convolve's BLAS sum
    const auto lp = tk::lowpass(std::span<const cd>(in.zc), 0.3);
    near(lp[0].real(), -0x1.d71dfe25bbd1dp-11, 1e-9, "lowpass [0]");
    near(lp[1000].imag(), -0x1.ec4b3c0a36622p-16, 1e-7, "lowpass [1000]");
    near(lp[5999].real(), -0x1.3a02e539ef9a5p-12, 1e-9, "lowpass [5999]");
    std::vector<double> x;
    for (std::size_t i = 0; i < 3000; ++i) x.push_back(std::norm(in.zc[i]));
    const auto sm = tk::smooth(x, 32);
    near(sm[0], 0x1.09c0000000000p-2, 1e-13, "smooth [0]");
    near(sm[1500], 0x1.2330000000000p-3, 1e-13, "smooth [1500]");
    near(sm[2999], 0x1.a0c0000000000p-3, 1e-13, "smooth [2999]");
}

static void test_kalman() {
    const Inputs in = inputs();
    const std::vector<cd> mu(in.zc.begin(), in.zc.begin() + 3000);
    std::vector<double> R;
    for (std::int64_t n = 0; n < 3000; ++n)
        R.push_back(n % 5 == 0 ? std::numeric_limits<double>::infinity() : 0.5 + static_cast<double>(n % 3) * 0.25);
    auto k = tk::kalman_rts(mu, R, 1e-4, 1.0);
    digest_is(k.u, "de2801e6a77882f59ce36667a71e95c12b1af5023093f54f49b8cf8a51f803f5", "kalman u");
    digest_is(k.P, "9d32270ea13d5184ab03db479c4a8b68c3c1d6ecc852f71c72fa2816ccff15fa", "kalman P");
    check::equal(k.loglik, -0x1.251a599c40739p+7, "kalman loglik");
    k = tk::kalman_rts(mu, R, 1e-4, 1.0, true);
    digest_is(k.u, "6878a479f4cef36d53eeeba14ae03917799610c18eef522bc6de7c610e34a201", "causal u");
    digest_is(k.P, "d1a557db84065d46fc86305b2a81c996c0005b3fec917ad8fdbeb7d7e22eeb2f", "causal P");
    check::equal(tk::kf_forward(mu, R, 1e-4, 1.0), -0x1.251a599c40739p+7, "kf_forward");
    const auto pass = tk::kf_forward_pass(mu, R, 1e-4, 1.0);
    check::equal(pass.loglik, -0x1.251a599c40739p+7, "kf_forward_pass loglik");
    check::equal(pass.xa.size(), std::size_t{3000}, "kf_forward_pass size");
    const auto sq = tk::select_q(mu, R, 1.0);
    check::is_true(sq.has_value(), "select_q");
    check::equal(sq->first, 0x1.dac586775972ep-31, "select_q q");
    check::equal(sq->second, 0x1.fc1ce9f331c21p+7, "select_q loglik");
    check::is_true(!tk::select_q(mu, R, 1.0, std::span<const double>()).has_value(), "select_q, no grid");
    const auto empty = tk::kalman_rts(std::span<const cd>(), std::span<const double>(), 0.1, 1.0);
    check::is_true(empty.u.empty() && empty.P.empty() && empty.loglik == 0.0, "kalman, nothing measured");
    throws<std::invalid_argument>([&] { (void)tk::kalman_rts(mu, std::vector<double>(3000, -1.0), 1e-4, 1.0); },
                                  "math domain error", "kalman, S < 0");
    throws<qrss::ZeroDivision>(
        [&] { (void)tk::kalman_rts(mu, std::vector<double>(3000, -10.0 - 1e-30), 1e-4, 1.0); },
        "float division by zero", "kalman, S == 0");
}

static void test_classes() {
    const auto bits = header_bits();
    const fr::HdrBits hb = fr::HdrBits::of(bits);
    const auto& SHORT = fr::get("short");
    const auto k = tk::known_symbols(SHORT, &hb);
    check::equal(k.pos.size(), std::size_t{3574}, "known size");
    digest_is(k.pos, "50e2d3e2b631007584380ac94fc7a4910e31e3240459a91c5dc6dce1c0a9bdfc", "known pos");
    digest_is(k.real(), "ef684674ef974a2a74ccfdceb95e07c62e97ea925ee77355975258669b2871dd", "known val");
    const auto kt = tk::known_symbols(fr::get("tiny"));
    check::equal(kt.pos.size(), std::size_t{729}, "known size, TINY");
    digest_is(kt.pos, "eb6a0e8ff73e80826d32ec8604c87c4adc023e663e9b545fc65529c4c172fb9b", "known pos, TINY");
    digest_is(kt.real(), "5f123f8a9d10d1293d8d966b11229c040d6b0838f1819c3feaff7c7692cc28fa", "known val, TINY");
    const auto b = tk::make_classes(SHORT, &hb);
    digest_is(b.mu, "9790af025987bbf88708e1fdf322516a500c4ed1020f8f25cc57eeb5cb7e614d", "classes B mu");
    digest_is(b.nu, "6a13cd1528c87ab3ad640f50c0ad05bc9be52b91047ce433ddbc9d81aad3e998", "classes B nu");
    digest_is(b.known, "14232f042707658f08cfa6aecba439447be37c7aad3685bf99e778e5b5e5a509", "classes B known");
    check::is_true(b.spread.has_value(), "classes B spread");
    if (b.spread)
        digest_is(*b.spread, "69ad2ec7979a6d07ee1ee27e9cf8555e7e071b4db76663d3d917e2e7b586486b", "classes B spread");
    check::is_true(b.hdr_known && b.spread_rho == 0.0, "classes B flags");
    const auto a = tk::make_classes(SHORT);
    digest_is(a.mu, "30edad907ab7513bd4710431e217ca3839a5494f8b090c09b7f1de8b0185d641", "classes A mu");
    digest_is(a.nu, "bd4c0bcac9006b82969ac86acab24172e3ae6cdc1c8455a7f6da812f3a827ad8", "classes A nu");
    digest_is(a.known, "c1f4281dee8a7016d3ca69089cc5c408be571b84255f94cde3171480102e6382", "classes A known");
    check::is_true(!a.spread && !a.hdr_known, "classes A flags");
    check::equal(a.spread_rho, 0x1.ba5e353f7ced9p-5, "classes A spread_rho");
    std::vector<std::int64_t> grid;
    for (std::int64_t p = -40; p < SHORT.n_pos(); ++p) grid.push_back(p);
    digest_is(tk::window_unknown(SHORT, std::span<const std::int64_t>(grid), true, qrss::config::SPAN),
              "076509aa099b1af2dede9b4bc81ed13fda02b092345937b95721e48d117deb10", "window_unknown");
    digest_is(tk::window_unknown(SHORT, std::span<const std::int64_t>(grid), false, qrss::config::SPAN),
              "e1116ebd43cf651abd88af5ba3a64b53e1195c3c93cccaab315e21025aa00759", "window_unknown, no OOK");
    const Inputs in = inputs();
    digest_is(tk::remove_spread(in.zc, -100.5, SHORT, in.tm, &*b.spread),
              "e225c500498d38e2e834785c9c15d79890fb511d7f46444a8d99f4320b157afe", "remove_spread");
    throws<fr::WrongWaveform>([] { (void)tk::make_classes(fr::get_l("l-short")); },
                              "track.make_classes is CE's; 'l-short' is waveform 1: use ltrack.make_classes_l",
                              "ce_only");
}

static void test_templates() {
    const auto& TINY = fr::get("tiny");
    std::vector<double> mu, nu;
    for (std::int64_t k = 0; k < TINY.n_sym(); ++k) {
        mu.push_back(static_cast<double>((k * 5) % 9 - 4) / 4.0);
        nu.push_back(static_cast<double>((k * 3) % 7) / 8.0);
    }
    std::vector<std::int64_t> grid;
    for (std::int64_t p = -30; p < 600; ++p) grid.push_back(p);
    const auto t = tk::templates(TINY, grid, mu, nu);
    check::equal(t.c.size(), std::size_t{630}, "templates size");
    // to rounding: numpy's SIMD exp, and 1 - e^-v cancelling where v is small
    check::is_true(std::abs(t.c[0].real()) <= 1e-15, "templates c[0]");
    near(t.c[300].imag(), -0x1.2ef5e4d0a2822p-1, 1e-12, "templates c[300]");
    near(t.sf[100], 0x1.bb6a3adcffb7dp-5, 1e-9, "templates sf[100]");
    near(t.sf[629], 0x1.2a839576d8692p-2, 1e-9, "templates sf[629]");
    throws<std::out_of_range>([&] { (void)tk::templates(TINY, std::span<const std::int64_t>(), mu, nu); },
                              "index 0 is out of bounds for axis 0 with size 0", "templates, no grid");
}

// The synthetic pass, as the reference makes it:
//
//   x = np.where((np.arange(TINY.n_data) * 5 + 1) % 3 == 0, 1.0, -1.0)
//   sym = frame.assemble(TINY, None, x); i = np.arange(16000)
//   phi = ce.phase_grid(250, -3000, 16000, sym, TINY)
//   sig = np.exp(1j * (phi + 2 * np.pi * 0.02 * i / 250))
//   noise = (((i * 7) % 19 - 9) / 8.0) + 1j * (((i * 3) % 17 - 8) / 8.0)
//   chan = ChanCapture(ch=(sig + noise).astype(np.complex64), fs=250, t0_index=3000.0, f_mix=Fraction(0))
//   flat = FreqPath(t_s=np.array([-10.0, 0.0, 1800.0]), f_hz=np.full(3, 1500.0), weight=np.ones(3))
//   det = Detection(1500.0, flat, Timing(0.0, 0.0, 0.0, 0.0, np.zeros((3, 3))), np.nan, "tbd", 0.0)
//   tr = track.track(chan, TINY, det); track.refine_timing(tr); track.report(tr, 0.9, False)
static std::shared_ptr<const tk::ChanCapture> synthetic_chan() {
    const auto& TINY = fr::get("tiny");
    std::vector<double> x;
    for (std::int64_t k = 0; k < TINY.n_data(); ++k) x.push_back((k * 5 + 1) % 3 == 0 ? 1.0 : -1.0);
    const auto sym = fr::assemble(TINY, nullptr, x);
    const auto phi = qrss::ce::phase_grid(250, -3000, 16000, sym, TINY);
    auto chan = std::make_shared<tk::ChanCapture>();
    for (std::int64_t i = 0; i < 16000; ++i) {
        const double ph = phi[static_cast<std::size_t>(i)] + 2 * std::numbers::pi * 0.02 * static_cast<double>(i) / 250;
        const cd z = cd(std::cos(ph), std::sin(ph)) + cd(static_cast<double>((i * 7) % 19 - 9) / 8.0,
                                                         static_cast<double>((i * 3) % 17 - 8) / 8.0);
        chan->ch.emplace_back(static_cast<float>(z.real()), static_cast<float>(z.imag()));
    }
    chan->fs = 250;
    chan->t0_index = 3000.0;
    chan->f_mix = qrss::Rational(0);
    return chan;
}

static types::Detection synthetic_det() {
    types::Detection det;
    det.f_hz = 1500.0;
    det.path = types::FreqPath{{-10.0, 0.0, 1800.0}, {1500.0, 1500.0, 1500.0}, {1.0, 1.0, 1.0}};
    types::Timing tm;
    tm.cov = math::Matrix(3, 3);
    det.timing = tm;
    det.z_ref = std::numeric_limits<double>::quiet_NaN();
    det.method = "tbd";
    return det;
}

static void test_synthetic_pass() {
    const auto chan = synthetic_chan();
    qrss_check::digest_is(chan->ch, "134cbc75afc51c58072f31270e8d75ec024b6a9d21d8424a801c1532de08feb2", "the capture");
    const auto& TINY = fr::get("tiny");
    const auto det = synthetic_det();
    const tk::TrackResult tr = tk::track(chan, TINY, det);
    // every decision is the reference's; the values to rounding (~1e-11 measured)
    check::equal(tr.q, 0x1.7e07eeb30f785p-6, "q");
    check::equal(tr.lead_in_s, 0.0, "lead_in_s");
    check::is_true(!tr.ook && !tr.keying, "no keying");
    check::equal(tr.pos.size(), std::size_t{1761}, "grid");
    check::equal(tr.pos.front(), std::int64_t{-8}, "grid start");
    digest_is(tr.meas, "5fb9aed931d78fd304398b51af348966b74caefa317b726b3252481fae5ec507", "meas");
    near(tr.loglik, -4349.926704242691, 1e-12, "loglik");
    near(tr.df_max, 0.0002691212870382946, 1e-8, "df_max");
    near(tr.timing.tau0, 0.03743309289846387, 1e-9, "timing tau0");
    near(tr.timing.ppm, -40.01153828525125, 1e-9, "timing ppm");
    check::equal(tr.timing.gamma, 0.0, "timing gamma");
    near(tr.u[0].real(), 0x1.03e23b977d940p-3, 1e-9, "u[0] re");
    near(tr.u[0].imag(), 0x1.d374c3814b47ap+0, 1e-9, "u[0] im");
    near(tr.u[500].real(), -0x1.4a773518cfb5ep-5, 1e-9, "u[500] re");
    near(tr.u[500].imag(), 0x1.0701894154fe6p+0, 1e-9, "u[500] im");
    near(tr.P[0], 0x1.104e248f68695p-9, 1e-9, "P[0]");
    near(tr.P[500], 0x1.94e9768d41262p-11, 1e-9, "P[500]");
    near(tr.freq(0.0), 0x1.7701371d1411ap+10, 1e-14, "freq(0)");
    near(tr.freq(20.0), 0x1.7701332324283p+10, 1e-14, "freq(20)");
    near(tr.freq(50.0), 0x1.770122834bb56p+10, 1e-14, "freq(50)");
    const auto rt = tk::refine_timing(tr);
    check::is_true(rt.has_value(), "refine_timing");
    if (rt) {
        near(rt->tau0, 0.002804528422325638, 1e-8, "refined tau0");
        near(rt->ppm, -26.07076952224611, 1e-9, "refined ppm");
        near(rt->z, 41240.487145206695, 1e-9, "refined z");
    }
    const auto rep = tk::report(tr, 0.9, false);
    near(rep.offset_hz, 1500.0189888664231, 1e-14, "report offset");
    check::is_true(std::abs(rep.drift_hz_per_min - -0.0019264591204400802) <= 1e-9, "report drift");
    near(rep.wander_hz_rms, 0.0069609643464083135, 1e-9, "report wander");
    check::equal(rep.doppler_hz, 3.0, "report doppler");
    near(rep.ppm, -40.01153828525125, 1e-9, "report ppm");
    near(rep.snr2500_db, 51.653716702940436, 1e-12, "report snr");
    check::is_true(std::isnan(rep.z_ref) && rep.kappa == 0.9 && rep.slip_free && !rep.suspect, "report flags");
    // reentrant: threads tracking the one capture get the one answer -- on
    // the capture whose noise table is already filled, and on a fresh one
    // whose empty NoisePsdSlot they race to fill
    const auto fresh = synthetic_chan();
    check::is_true(!fresh->psd.get(), "a fresh capture has no noise table yet");
    std::vector<std::thread> th;
    std::vector<std::optional<tk::TrackResult>> out(4);
    for (int i = 0; i < 4; ++i)
        th.emplace_back([&, i] { out[static_cast<std::size_t>(i)] = tk::track(i % 2 ? fresh : chan, TINY, det); });
    for (auto& t : th) t.join();
    for (const auto& o : out) {
        check::is_true(o && o->q == tr.q && o->u == tr.u && o->P == tr.P && o->loglik == tr.loglik && o->N == tr.N,
                       "a concurrent track");
    }
    check::is_true(fresh->psd.get() != nullptr, "the race filled the fresh capture's table");
    // refusals
    auto none = det;
    none.timing.reset();
    throws<std::invalid_argument>([&] { (void)tk::track(chan, TINY, none); },
                                  "track: no timing (det.timing is None)", "no timing");
    throws<fr::WrongWaveform>([&] { (void)tk::track(chan, fr::get_l("l-short"), det); },
                              "track.track is CE's; 'l-short' is waveform 1: use ltrack.track_l", "an L frame");
}

int main() {
    try {
        test_tables();
        test_mf_psi_derotate();
        test_filters();
        test_kalman();
        test_classes();
        test_templates();
        test_synthetic_pass();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FATAL: %s\n", e.what());
        return 1;
    }
    return check::report("qrss_track");
}

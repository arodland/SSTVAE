// native/core/qrss/lwave (waveform L's transmitter) against values the
// Python reference produced. No Python at test time.
//
// Inputs are exact in both languages: symbols
//   sym[i] = ((i*7919) % 2001 - 1000)/1000 + 1j ((i*104729) % 1999 - 999)/999
// (`seq_c`), the callsign "K1ABC/P" through morse.keying_units, and the
// frozen presets. Each pin was made with sstvae/qrss/lwave.py:
//
//   hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()   (digests)
//   float(v).hex()                                                  (values)
//
// Integer and table work (at_positions, data_grid, mix_turns, tau_grid,
// keyed_span_pos, window_keying, _interp_taps, resample) is pinned by
// digest: it is the same arithmetic in the same order everywhere. What goes
// through sin/cos (the pilot, cw_turns, id_weight, composite, band_mask) is
// bit-exact against numpy on glibc (tests/test_qrss_native_lwave.py) but
// another libm may round differently, so it is pinned by value at 1e-12.
// The clipper's FFT is pocketfft here and ducc0 in Python: frame_baseband
// is pinned at the plan's 1e-9 and, as an early warning, at 1e-12
// (measured ~1e-15).

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "qrss/qrss_config.hpp"
#include "qrss/filters.hpp"
#include "qrss/frame.hpp"
#include "qrss/lwave.hpp"
#include "qrss/morse.hpp"
#include "qrss/npsum.hpp"
#include "qrss/rational.hpp"
#include "qrss/tables.hpp"
#include "qrss_check.hpp"

using namespace sstvae;
namespace lw = sstvae::qrss::lwave;
namespace fr = sstvae::qrss::frame;
namespace cfg = sstvae::qrss::config;
using qrss_check::digest_is;
using cd = std::complex<double>;

namespace {

std::vector<cd> seq_c(std::int64_t n) {
    std::vector<cd> v(static_cast<std::size_t>(n));
    for (std::int64_t i = 0; i < n; ++i)
        v[static_cast<std::size_t>(i)] =
            cd(static_cast<double>((i * 7919) % 2001 - 1000) / 1000.0,
               static_cast<double>((i * 104729) % 1999 - 999) / 999.0);
    return v;
}

const qrss::Rational PILOT_HZ(cfg::L_PILOT_HZ.num, cfg::L_PILOT_HZ.den);

void near(double got, double want, double tol, const std::string& what) {
    ++check::checks;
    if (!(std::abs(got - want) <= tol)) {
        char buf[160];
        std::snprintf(buf, sizeof buf, "got %a, want %a (diff %.3g > %.3g)", got, want, std::abs(got - want), tol);
        check::fail(what, buf);
    }
}

double sum_re(const std::vector<cd>& z) {
    std::vector<double> r(z.size());
    for (std::size_t i = 0; i < z.size(); ++i) r[i] = z[i].real();
    return qrss::npsum::sum(std::span<const double>(r));
}

double sum_im(const std::vector<cd>& z) {
    std::vector<double> r(z.size());
    for (std::size_t i = 0; i < z.size(); ++i) r[i] = z[i].imag();
    return qrss::npsum::sum(std::span<const double>(r));
}

double sum_pow(const std::vector<cd>& z) {
    std::vector<double> r(z.size());
    for (std::size_t i = 0; i < z.size(); ++i) {
        const double a = lw::np_abs(z[i]);   // np.abs(z) ** 2: numpy squares
        r[i] = a * a;
    }
    return qrss::npsum::sum(std::span<const double>(r));
}

double sum_of(const std::vector<double>& v) { return qrss::npsum::sum(std::span<const double>(v)); }

void refuses(const std::function<void()>& fn, const std::string& want, const std::string& what) {
    try {
        fn();
    } catch (const std::invalid_argument& e) {
        check::equal(std::string(e.what()), want, what);
        return;
    }
    check::fail(what, "did not throw");
}

// Sample indices the value pins below were taken at.
const std::vector<std::size_t> IDX = {0,     1,     777,   2500,  9999,  15000, 21111, 30000,
                                      33333, 40000, 47777, 52000, 60000, 64000, 66067};

}  // namespace

static void test_constants() {
    check::equal(lw::a_pilot(), 0x1.ae2a81b75e576p-2, "A_PILOT");
    check::equal(lw::a_clip(), 0x1.15abc5bd2feafp+0, "A_CLIP");
    check::equal(lw::a_id(), 0x1.96632cbc9f77fp-1, "A_ID");
    check::equal(lw::g0(), 0x1.006cb85d5f6d6p+0, "G0");
    check::equal(lw::HALF, std::int64_t(7760), "HALF");
    check::equal(lw::CLIP_FS, 250, "CLIP_FS");
    const auto t = lw::pulse_table();
    check::equal(qrss_check::digest(std::vector<double>(t.begin(), t.end())), std::string(qrss::tables::L_PULSE_TABLE_SHA256),
                 "PULSE_TABLE");
    // The frozen interpolator is firwin(481, 125, fs=8000, ('kaiser', 12)).
    const auto taps = lw::interp_taps(1);
    check::equal(qrss_check::digest(taps), std::string(qrss::tables::L_INTERP_TAPS_SHA256), "_interp_taps(1)");
    const auto fw = qrss::filters::firwin(481, 125.0, qrss::filters::Window::kaiser(12.0),
                                          qrss::filters::PassZero::Yes, true, 8000.0);
    double worst = 0;
    for (std::size_t i = 0; i < fw.size(); ++i) worst = std::max(worst, std::abs(fw[i] - taps[i]));
    near(worst, 0.0, 1e-15, "the frozen taps are firwin's");
    digest_is(lw::interp_taps(32), "89949fc578fa813797fdf433348d65c42e099baedfe82faceca0d25fed673570", "_interp_taps(32)");
}

static void test_grid() {
    const auto& shrt = fr::get_l("l-short");
    const auto& tiny = fr::get_l("l-tiny");
    const auto sym_s = seq_c(shrt.n_sym()), sym_t = seq_c(tiny.n_sym());
    digest_is(lw::at_positions(sym_s, shrt), "c07b86b2a3c9f743526bd2d6e39704546e5384f9a30829248a90f317045eb25b",
              "at_positions l-short");
    digest_is(lw::at_positions(sym_t, tiny), "469f7f99336506d1bcaa0171f0d645eb71273e81bdb80f51d67939769acf5378",
              "at_positions l-tiny");
    digest_is(lw::data_grid(250, -3000, 70000, sym_s, shrt),
              "d681b99f8696dfcced944037d494e3b8a74d83706a9d2da96a3bccc16660f9bd", "data_grid 250 l-short");
    digest_is(lw::data_grid(4000, -1000, 40000, sym_t, tiny),
              "b5b24cc97fd4fca471c6951c166ffe8a04f12e1ebe19d0b99cde29c480b001bc", "data_grid 4000 l-tiny");
    digest_is(lw::data_grid(8000, 100000, 3000, sym_t, tiny),
              "c421e2c914623d959e9c199348e216a061f8e2d3d5f6711bf7057d5a54ad37ab", "data_grid past the frame");
    const auto v50 = seq_c(50);
    std::vector<double> r50(v50.size());
    for (std::size_t i = 0; i < r50.size(); ++i) r50[i] = v50[i].real();
    digest_is(lw::grid_sum(1000, -20, 300, std::span<const double>(r50), lw::pulse_table()),
              "8c863fd8f49df4f14bd008c8fb18d5633b7d3c2b852bbc5b587b074d824662e2", "_grid_sum, real");
    check::equal(lw::data_grid(250, 0, 0, sym_t, tiny).size(), std::size_t(0), "n = 0");
    check::equal(lw::data_grid(250, 0, -5, sym_t, tiny).size(), std::size_t(0), "n < 0");

    digest_is(lw::mix_turns(250, -3000, 70000, PILOT_HZ),
              "4fc7ef4bd17db73d58faf6bfbba980afdf90f3f9fcf6926367bb366f6884eaff", "mix_turns 250");
    digest_is(lw::mix_turns(8000, -123457, 20000, qrss::Rational(3, 7)),
              "983a1f3ce96877bfe93f26a2c2848f93f5dbb3762f320dd879e9a8ce82673046", "mix_turns 3/7 Hz");
    // A period past 3e9 (the reference's int64 products wrap there): exact.
    const auto big = lw::mix_turns(250, 0, 3, qrss::Rational(-3602879701896397, std::int64_t(1) << 55));
    const std::int64_t period = (std::int64_t(1) << 55) * 250;
    check::equal(big[1], static_cast<double>(period - 3602879701896397) / static_cast<double>(period),
                 "mix_turns, an int64-sized period");
    refuses([] { lw::mix_turns(0, 0, 3, PILOT_HZ); }, "mix_turns needs a positive fs, not 0", "mix_turns, fs 0");
    bool overflow = false;
    try {
        lw::mix_turns(4000, 0, 3, qrss::Rational(1, std::int64_t(1) << 55));
    } catch (const std::overflow_error&) {
        overflow = true;
    }
    check::is_true(overflow, "mix_turns, a period past int64");
    digest_is(lw::tau_grid(250, -3000, 70000), "83525166eafc65968dec23a9d6bfbfe7ac1a3dc2811cc61ca1de0a7147b8e0df",
              "tau_grid 250");
    digest_is(lw::tau_grid(4000, -123457, 20000), "6629b77b1a820abd6e0c3d197c8e984e8427738c56e6373e2709dd97da4b6e32",
              "tau_grid 4000");

    refuses([] { lw::grid_step(300); }, "fs must divide 16000, not 300", "fs 300");
    refuses([] { lw::grid_step(0); }, "fs must divide 16000, not 0", "fs 0");
    check::equal(lw::grid_step(250), std::int64_t(64), "fs 250");
    refuses([&] { lw::at_positions(std::span<const cd>(sym_t.data(), sym_t.size() - 1), tiny); },
            "sym has shape (876,), expected (877,)", "short sym");
    bool wrong = false;
    try {
        lw::at_positions(seq_c(fr::get("tiny").n_sym()), fr::get("tiny"));
    } catch (const fr::WrongWaveform&) {
        wrong = true;
    }
    check::is_true(wrong, "a CE frame is the wrong waveform");
}

static void test_keying() {
    const auto& shrt = fr::get_l("l-short");
    check::is_true(lw::keyed_span_pos(shrt) == std::pair<double, double>(-0x1.0p+3, 0x1.0798p+12), "span, no lead-in");
    check::is_true(lw::keyed_span_pos(shrt, 5.0) == std::pair<double, double>(-0x1.69e59bb61a645p+6, 0x1.0798p+12),
                   "span, 5 s lead-in");
    check::is_true(lw::keyed_span_pos(shrt, 10.0) == std::pair<double, double>(-0x1.59e59bb61a645p+7, 0x1.0798p+12),
                   "span, 10 s lead-in");
    refuses([&] { lw::keyed_span_pos(shrt, 11.0); }, "lead-in must be 0 to 10 s, not 11.0", "lead-in 11");
    refuses([&] { lw::keyed_span_pos(shrt, -0.5); }, "lead-in must be 0 to 10 s, not -0.5", "lead-in -0.5");
    refuses([&] { lw::keyed_span_pos(shrt, std::nan("")); }, "lead-in must be 0 to 10 s, not nan", "lead-in nan");

    const auto units = qrss::morse::keying_units("K1ABC/P");
    const auto k = lw::Keying::units(units);
    const auto w = lw::window_keying(&k, shrt);
    check::is_true(w.has_value(), "keying");
    digest_is(*w, "b79c8e9e39103fec9ac356fdfb1ef28821d373edb69c7f4fef6a1f2f513cadef", "window_keying, one mask");
    lw::Keying per;
    per.shape = {2, cfg::L_CW_WINDOW};
    per.values.assign(2 * cfg::L_CW_WINDOW, 0.0);
    for (int u = 0; u < cfg::L_CW_WINDOW; ++u) per.values[static_cast<std::size_t>(cfg::L_CW_WINDOW + u)] = units[u];
    digest_is(*lw::window_keying(&per, shrt), "44d18ea3a1c10d950dbe8215730e8456626283aa03c7fdfc46d3154b46b651b2",
              "window_keying, per window");
    check::is_true(!lw::window_keying(nullptr, shrt).has_value(), "no keying");
    lw::Keying bad = k;
    bad.values.pop_back();
    bad.shape = {cfg::L_CW_WINDOW - 1};
    refuses([&] { lw::window_keying(&bad, shrt); }, "keying must be 0/1 of shape (192,) or (2, 192), not (2, 191)",
            "191 units");
    bad = k;
    bad.values[10] = 2.0;
    refuses([&] { lw::window_keying(&bad, shrt); }, "keying must be 0/1 of shape (192,) or (2, 192), not (2, 192)",
            "a 2 in the mask");
    // No windows: the mask broadcasts to (0, 192) and none of its values is
    // looked at, so the reference accepts even this one.
    const auto none = lw::window_keying(&bad, fr::get_l("l-tiny"));
    check::is_true(none.has_value() && none->empty(), "a frame without windows takes any (192,) mask");
}

static void test_pilot() {
    const auto& shrt = fr::get_l("l-short");
    const auto units = qrss::morse::keying_units("K1ABC/P");
    const auto k = lw::Keying::units(units);
    const auto tau = lw::tau_grid(250, -3000, 70000);
    const double tol = 1e-12;

    const auto cw = lw::cw_turns(tau, shrt, &k);
    near(sum_of(cw), -0x1.c1841d00006fcp+20, 1e-6, "cw_turns sum");
    const double cw_want[] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0x1.b8p+5, -0x1.b8p+5, -0x1.b8p+5, -0x1.b8p+5, -0x1.b8p+6};
    for (std::size_t j = 0; j < IDX.size(); ++j) near(cw[IDX[j]], cw_want[j], tol, "cw_turns[" + std::to_string(IDX[j]) + "]");

    const auto W = lw::id_weight(tau, shrt, 5.0);
    near(sum_of(W), 0x1.0a2b000149fa6p+13, 1e-9, "id_weight sum");
    const double w_want[] = {1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1};
    for (std::size_t j = 0; j < IDX.size(); ++j) near(W[IDX[j]], w_want[j], tol, "id_weight[" + std::to_string(IDX[j]) + "]");

    const auto p = lw::pilot(tau, shrt, &k, 5.0);
    near(sum_re(p), 0x1.bf7c974c2174cp+14, 1e-9, "pilot sum, real");
    near(sum_im(p), 0x1.45d5a1ec79000p-11, 1e-9, "pilot sum, imaginary");
    near(p[2500].real(), lw::a_id(), tol, "pilot in the lead-in is A_ID");
    near(p[9999].real(), lw::a_pilot(), tol, "pilot in the data is A_PILOT");
    check::equal(p[0].real(), 0.0, "pilot before the lead-in");

    const auto sym = seq_c(shrt.n_sym());
    const auto c = lw::composite(sym, shrt, 250, -3000, 70000, &k, 5.0);
    near(sum_re(c), 0x1.bf7c85720932ap+14, 1e-9, "composite sum, real");
    near(sum_im(c), 0x1.69a8ceb1cfd80p-3, 1e-9, "composite sum, imaginary");
    const double c_re[] = {0,
                           0,
                           0,
                           0x1.96632cbc9f77fp-1,
                           0x1.9cdeccff91bd0p-4,
                           -0x1.d659e7d5066a0p-6,
                           -0x1.40593239fa178p-4,
                           0x1.1512c8aa074d4p-3,
                           0x1.57d7c7dd2c4a0p-6,
                           0x1.945fee6a1cf21p-1,
                           0x1.9569e3b9abf4cp-1,
                           0x1.1766319f5e566p-3,
                           -0x1.d333aa11589c0p-5,
                           0x1.11ded0fc6e3c6p+0,
                           0x1.96632cbc9f77fp-1};
    const double c_im[] = {0,
                           0,
                           0,
                           0,
                           0x1.92143e386ffa0p-2,
                           -0x1.c63d4781035dep-2,
                           -0x1.a4c54fac24d7bp-1,
                           0x1.b061a98c8102ap-3,
                           -0x1.99ef06acb01c8p-4,
                           -0x1.40edb4b6267cep-2,
                           -0x1.f3c97059f13d4p-1,
                           0x1.f59e234c6a9bap-1,
                           0x1.145ab525c4b93p-2,
                           -0x1.7265cbb1cc5b4p-2,
                           0};
    for (std::size_t j = 0; j < IDX.size(); ++j) {
        near(c[IDX[j]].real(), c_re[j], tol, "composite[" + std::to_string(IDX[j]) + "].real");
        near(c[IDX[j]].imag(), c_im[j], tol, "composite[" + std::to_string(IDX[j]) + "].imag");
    }
    refuses([&] { lw::composite(sym, shrt, 250, 0, 10, nullptr, 12.0); }, "lead-in must be 0 to 10 s, not 12.0",
            "composite, lead-in 12");

    std::vector<double> y;
    for (int i = 0; i <= 2000; ++i) y.push_back(-0.5 + 2.0 * i / 2000.0);
    std::vector<double> ry(y.size());
    for (std::size_t i = 0; i < y.size(); ++i) ry[i] = lw::raised(y[i]);
    near(sum_of(ry), 0x1.f44p+9, 1e-9, "_raised sum");   // np.sum(_raised(np.linspace(-0.5, 1.5, 2001)))
    check::equal(lw::raised(-1.0), 0.0, "_raised below 0");
    check::equal(lw::raised(2.0), 1.0, "_raised above 1");
    check::is_true(std::isnan(lw::raised(std::nan(""))), "_raised(nan)");
}

static void test_clipper() {
    // fftfreq and the band mask on the l-short frame's FFT size.
    const auto f = lw::fftfreq(66068, 1.0 / 250);
    digest_is(f, "15525c89dc8cdaa0b5529f99869088e75a2af4490574fa49ce3dd6a2ca40655d", "fftfreq(66068, 1/250)");
    const auto bm = lw::band_mask(f);
    near(sum_of(bm), 0x1.8d8c82690a2a6p+12, 1e-9, "band_mask sum");
    const double bm_want[] = {0x1.fffc8b10d43bbp-1, 0x1.ffefe949d1137p-1, 0x1.ffda0532d27c1p-1, 0x1.ffbadf9695a72p-1,
                              0x1.ff92799590a15p-1, 0x1.ff60d4a5e7e2ap-1, 0x1.ff25f29360c58p-1, 0x1.fee1d57f50e4fp-1,
                              0x1.fe947fe08a618p-1, 0x1.fe3df483450fap-1, 0x1.fdde3689048e0p-1, 0x1.fd7549687b470p-1};
    for (std::size_t j = 0; j < 12; ++j) near(bm[133 + j], bm_want[j], 1e-12, "band_mask[" + std::to_string(133 + j) + "]");
    check::equal(bm[0], 1.0, "band_mask at the pilot");

    // next_fast_len: scipy's values, and the smallest 11-smooth number >= n.
    const std::pair<std::int64_t, std::int64_t> nfl[] = {{1, 1},         {7, 7},         {13, 14},
                                                         {97, 98},       {1000, 1000},   {66068, 66150},
                                                         {67318, 67375}, {1000003, 1000188}};
    for (const auto& [n, want] : nfl) check::equal(lw::next_fast_len(n), want, "next_fast_len(" + std::to_string(n) + ")");
    auto smooth = [](std::int64_t m) {
        for (const std::int64_t p : {2, 3, 5, 7, 11})
            while (m % p == 0) m /= p;
        return m == 1;
    };
    bool ok = true;
    for (std::int64_t n = 1; n <= 5000; ++n) {
        std::int64_t want = n;
        while (!smooth(want)) ++want;
        ok = ok && lw::next_fast_len(n) == want;
    }
    check::is_true(ok, "next_fast_len is the next 11-smooth number up to 5000");

    const auto& tiny = fr::get_l("l-tiny");
    const auto sym_t = seq_c(tiny.n_sym());
    const auto z0 = lw::composite(sym_t, tiny, 250, -3000, 66068);
    const auto cl = lw::clip(z0);
    near(sum_re(cl), 0x1.2016922790511p+12, 1e-9, "clip sum, real");
    near(sum_im(cl), -0x1.22f6040f4a2e8p+2, 1e-9, "clip sum, imaginary");
    near(sum_pow(cl), 0x1.f3c95e19c48c7p+12, 1e-9, "clip power");
    check::is_true(lw::clip(z0, 250, 1e6) == z0, "nothing over the threshold: the input comes back");
    check::is_true(lw::clip(z0, 250, lw::a_clip(), std::span<const double>()) == z0, "no passes: the input");
    // np_abs and np_mul: numpy's forms.
    check::equal(lw::np_abs(cd(3, 4)), 5.0, "|3+4j|");
    check::is_true(std::isinf(lw::np_abs(cd(INFINITY, std::nan("")))), "|inf+nan j| is inf");
    check::is_true(std::isnan(lw::np_abs(cd(std::nan(""), 1.0))), "|nan+1j| is nan");
    const cd a(1 + 0x1p-30, 1 - 0x1p-30), b(1 - 0x1p-30, 1 + 0x1p-30);
    check::equal(lw::np_mul(a, b).real(), std::fma(a.real(), b.real(), -(a.imag() * b.imag())), "np_mul's fused real part");
}

struct FramePin {
    const char* name;
    double lead;
    bool keyed;
    std::int64_t n0, len, zeros;
    double s_re, s_im, s_pow;
    std::vector<std::size_t> idx;
    std::vector<double> re, im;
};

static void test_frame_baseband() {
    const std::vector<FramePin> pins = {
        {"l-tiny", 0.0, false, -1134, 15545, 2001, 0x1.6aee751fd0eaep+12, -0x1.7053b8079efb8p+2, 0x1.8ca7e533c1c62p+13,
         {1001, 2129, 3257, 4385, 5513, 6641, 7769, 8897, 10025, 11153, 12281, 13409},
         {0x1.b344c9a4da06cp-7, 0x1.bf8ca9bdf68e1p-3, -0x1.53ebf46714f8ap-1, -0x1.073fdfef59c6ap+0, 0x1.8086f05839e7bp-2,
          0x1.b63d6461e3580p-1, 0x1.352040f2b08a7p-2, 0x1.442ce3600e5e8p+0, 0x1.4dfcc980bb1e9p+0, -0x1.1db63097c91d2p-1,
          0x1.0db8162580898p-2, 0x1.cce9637eb0498p-1},
         {0x1.68502054f80ffp-8, 0x1.0622a8cce4226p+0, -0x1.308e0e84237bep-1, 0x1.2633afcf488efp-5, -0x1.e1f5b014400c1p-1,
          -0x1.98b3617e5739cp-4, -0x1.1345aa5825c30p-1, -0x1.30b92e4de26edp-1, -0x1.ea164bb5d5a1ep-3,
          -0x1.6d0197185a4b0p-2, -0x1.f38227397f3a3p-1, -0x1.e38501a1d5fdbp-4}},
        {"l-tiny", 5.0, false, -2384, 16795, 2001, 0x1.bb0b2fdeba4b0p+12, -0x1.7bc202e88f240p+2, 0x1.b5167eaecd10ep+13,
         {1001, 2233, 3465, 4697, 5929, 7161, 8393, 9625, 10857, 12089, 13321, 14553},
         {0x1.22028cf84d306p-7, 0x1.01ded115af6e2p+0, -0x1.2467bf8d2a50ap-1, 0x1.2be8d880f28ebp-1, 0x1.288f30beed08ep+0,
          -0x1.0aa76f8f7aa2ap-7, 0x1.4700566fc1433p+0, -0x1.807436e92ac14p-6, 0x1.10da8fb11e92cp+0,
          -0x1.2af690c2a8267p-8, 0x1.d1834c652a789p-1, -0x1.4d16a002d6257p-1},
         {-0x1.e1503c1de543fp-18, 0x1.e6175981f5635p-10, -0x1.b59f8a090eb47p-1, -0x1.5ab08f2f218c3p-2,
          0x1.54a7a04aeafcdp-1, 0x1.eef9f8837f381p-1, -0x1.3dcf541b4e884p-5, -0x1.fb3e4311136b2p-2, 0x1.006104247be2ap-3,
          0x1.3b8b5e799da9dp+0, 0x1.bbe3eda8a5f50p-4, -0x1.1a2dd65644ed8p-4}},
        {"l-short", 5.0, true, -2384, 67318, 2001, 0x1.d7fbe006aa6cdp+14, -0x1.cdd468367badep+3, 0x1.e4ec3d02f5bdbp+15,
         {1001, 6444, 11887, 17330, 22773, 28216, 33659, 39102, 44545, 49988, 55431, 60874},
         {0x1.21fa0dffbd3f4p-7, 0x1.4e538a0fe7853p-1, 0x1.3bf28813f3880p+0, 0x1.000887780ca9fp-1, 0x1.086fcff6aee5ap-1,
          0x1.57f0918de802ep-1, -0x1.b6f7c8ce9e9e2p-2, 0x1.b48de0fed075fp-1, 0x1.09e44aead4d5ep+0,
          -0x1.163d3fcdd7b7bp-2, 0x1.a65d831e6af49p-2, 0x1.0a5b1e08bf5bdp+0},
         {-0x1.0878a5ca874fcp-17, -0x1.25d829163e06dp-1, -0x1.4eedfc230a9b5p-2, 0x1.1d2e9f7ba7f79p+0,
          -0x1.0aeb5c30b6797p+0, -0x1.add827bc05521p-2, 0x1.e21aa92df149bp-1, 0x1.5db0c6d53753cp-1,
          -0x1.c17c01bd20e33p-1, -0x1.b95cc437297c6p-2, 0x1.f716bb53dcf57p-5, 0x1.3854336e41373p-1}},
    };
    const auto units = qrss::morse::keying_units("K1ABC/P");
    const auto k = lw::Keying::units(units);
    for (const auto& p : pins) {
        const auto& spec = fr::get_l(p.name);
        const std::string what = std::string(p.name) + " lead " + std::to_string(p.lead);
        const auto bb = lw::frame_baseband(seq_c(spec.n_sym()), spec, p.keyed ? &k : nullptr, p.lead);
        check::equal(bb.n0, p.n0, what + ": n0");
        check::equal(static_cast<std::int64_t>(bb.z.size()), p.len, what + ": length");
        check::equal(bb.fs, std::int64_t(250), what + ": fs");
        std::int64_t zeros = 0;
        for (const cd& v : bb.z) zeros += (v == cd(0, 0)) ? 1 : 0;
        check::equal(zeros, p.zeros, what + ": zeroed samples");
        for (const double tol : {1e-9, 1e-12}) {
            near(sum_re(bb.z), p.s_re, tol * 1e3, what + ": sum, real");
            near(sum_im(bb.z), p.s_im, tol * 1e3, what + ": sum, imaginary");
            near(sum_pow(bb.z), p.s_pow, tol * 1e3, what + ": power");
            for (std::size_t j = 0; j < p.idx.size(); ++j) {
                near(bb.z[p.idx[j]].real(), p.re[j], tol, what + ": z[" + std::to_string(p.idx[j]) + "].real");
                near(bb.z[p.idx[j]].imag(), p.im[j], tol, what + ": z[" + std::to_string(p.idx[j]) + "].imag");
            }
        }
        if (p.keyed) {
            // Chunk invariance, bit for bit: any split of the output is the same.
            const std::int64_t i0 = bb.n0 * 32 - 500, m = static_cast<std::int64_t>(bb.z.size()) * 32 + 1000;
            const auto whole = lw::resample(bb, cfg::FS, i0, m);
            std::vector<cd> parts(static_cast<std::size_t>(m));
            std::int64_t at = 0, step = 1;
            while (at < m) {
                const std::int64_t len = std::min(step, m - at);
                lw::resample_into(bb.z, bb.n0, bb.fs, cfg::FS, i0 + at,
                                  std::span<cd>(parts.data() + at, static_cast<std::size_t>(len)));
                at += len;
                step = step * 7 % 100003 + 1;
            }
            check::is_true(parts == whole, what + ": resample is chunk-invariant");
            const std::int64_t start = bb.n0 * 32;
            check::is_true(whole[static_cast<std::size_t>(start - i0 - 300)] == cd(0, 0), what + ": silent before the frame");
        }
    }
    refuses([&] { lw::frame_baseband(seq_c(10), fr::get_l("l-tiny")); }, "sym has shape (10,), expected (877,)",
            "frame_baseband, short sym");
}

static void test_resample() {
    const auto z = seq_c(3000);
    digest_is(lw::resample(z, -250, 250, 8000, -250 * 32 - 600, 3000 * 32 + 1200),
              "8a69b8470acb0cb52a1335bb5b1cf8a227f6142fc3cce987a89469736c4cb616", "resample to 8 kHz");
    digest_is(lw::resample(z, -250, 250, 4000, -1000, 20000),
              "2af7c78771846cc9468bf650bd9b841739997c3b8ae7ba91b9720112746c19cb", "resample to 4 kHz");
    digest_is(lw::resample(z, -250, 250, 250, -300, 3100),
              "706905fe95d01aa4b737c4d24c4271e25e3d3c4096e73c516a38abd632bd4281", "resample at its own rate");
    refuses([&] { lw::resample(z, -250, 250, 8001, 0, 10); }, "fs_out must be a multiple of 250", "8001 Hz");
    refuses([&] { lw::resample(z, -250, 250, 100, 0, 10); }, "fs_out must be a multiple of 250", "100 Hz");
    refuses([&] { lw::resample(z, -250, 250, 8000, 0, -1); }, "negative dimensions are not allowed", "m < 0");
    check::equal(lw::resample(z, -250, 250, 8000, 0, 0).size(), std::size_t(0), "m = 0");

    const auto tdc = lw::to_data_centre(z, -250, 250);
    near(sum_re(tdc), 0x1.b046c09002e2fp+3, 1e-9, "to_data_centre sum, real");
    near(sum_im(tdc), 0x1.346911c8cbdd8p+0, 1e-9, "to_data_centre sum, imaginary");
}

int main() {
    try {
        test_constants();
        test_grid();
        test_keying();
        test_pilot();
        test_clipper();
        test_frame_baseband();
        test_resample();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FATAL: %s\n", e.what());
        return 1;
    }
    return check::report("qrss_lwave");
}

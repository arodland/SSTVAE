// native/core/math/convolve against scipy 1.18.1: next_fast_len,
// _calc_oa_lens and lambertw(x, -1) as pinned integers and values;
// oaconvolve / fftconvolve against pinned samples and a direct O(n m)
// convolution; freqz against pinned values and a direct DTFT;
// maximum_filter1d against pinned outputs and a brute-force reflect.
// Inputs are splitmix64 counters (infra_testgen.hpp):
//
//   a = unit(splitmix(108, 3000)) - 0.5 + 1j * (unit(splitmix(109, 3000)) - 0.5)
//   b = unit(splitmix(110, 255)) - 0.5
//   bb = unit(splitmix(106, 41)) - 0.5
//   m = (splitmix(107, 50) % 5).astype(np.uint8)
//
// tests/test_qrss_native_frontend.py compares against scipy directly.

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <numbers>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "infra_testgen.hpp"
#include "math/convolve.hpp"
#include "qrss_check.hpp"

using namespace sstvae;
using cd = std::complex<double>;

template <typename E, typename F>
static void throws(F f, const std::string& text, const std::string& what) {
    try {
        f();
    } catch (const E& e) {
        check::equal(std::string(e.what()), text, what);
        return;
    } catch (const std::exception& e) {
        check::fail(what, std::string("threw the wrong type: ") + e.what());
        return;
    }
    check::fail(what, "did not throw");
}

static void near(cd got, cd want, double tol, const std::string& what) {
    ++check::checks;
    if (!(std::abs(got - want) <= tol)) {
        char buf[200];
        std::snprintf(buf, sizeof buf, "got {%.17g, %.17g}, want {%.17g, %.17g}", got.real(), got.imag(),
                      want.real(), want.imag());
        check::fail(what, buf);
    }
}

static std::vector<double> uniform(std::uint64_t seed, std::size_t n) {
    const auto z = qrss_testgen::splitmix(seed, n);
    std::vector<double> v(n);
    for (std::size_t i = 0; i < n; ++i) v[i] = qrss_testgen::unit(z[i]) - 0.5;
    return v;
}

template <typename A, typename B>
static std::vector<cd> direct(const std::vector<A>& a, const std::vector<B>& b) {
    std::vector<cd> out(a.size() + b.size() - 1);
    for (std::size_t i = 0; i < a.size(); ++i)
        for (std::size_t j = 0; j < b.size(); ++j) out[i + j] += cd(a[i]) * cd(b[j]);
    return out;
}

template <typename T>
static double worst(const std::vector<T>& got, const std::vector<cd>& want, std::size_t off = 0) {
    double w = got.size() + off <= want.size() ? 0.0 : HUGE_VAL;
    for (std::size_t i = 0; i < got.size() && i + off < want.size(); ++i)
        w = std::max(w, std::abs(cd(got[i]) - want[i + off]));
    return w;
}

static void test_lengths() {
    const std::int64_t t[] = {1, 7, 13, 2408, 1000001, 999983};
    const std::int64_t c[] = {1, 7, 14, 2420, 1000188, 1000000};
    const std::int64_t r[] = {1, 8, 15, 2430, 1012500, 1000000};
    for (int i = 0; i < 6; ++i) {
        check::equal(math::next_fast_len(t[i]), c[i], "next_fast_len " + std::to_string(t[i]));
        check::equal(math::next_fast_len(t[i], true), r[i], "next_fast_len real " + std::to_string(t[i]));
    }
    struct Oa {
        std::int64_t s1, s2, block, overlap, step1, step2;
    };
    const Oa oa[] = {{2000000, 255, 2420, 254, 2166, 255}, {40000, 121, 1050, 120, 930, 121},
                     {5000, 243, 2304, 242, 2062, 243},    {7000, 161, 1440, 160, 1280, 161},
                     {300000, 2001, 23520, 2000, 21520, 2001}, {9000, 481, 4900, 480, 4420, 481},
                     {7000, 321, 3125, 320, 2805, 321},    {100, 255, 354, -1, 100, 255},
                     {256, 255, 510, -1, 256, 255},        {255, 255, 509, -1, 255, 255}};
    for (const Oa& o : oa) {
        const auto l = math::calc_oa_lens(o.s1, o.s2);
        const std::string what = "calc_oa_lens(" + std::to_string(o.s1) + ", " + std::to_string(o.s2) + ")";
        check::equal(l.block_size, o.block, what + " block");
        check::equal(l.overlap, o.overlap, what + " overlap");
        check::equal(l.in1_step, o.step1, what + " step 1");
        check::equal(l.in2_step, o.step2, what + " step 2");
    }
    const std::pair<double, double> w[] = {
        {-0.3, -1.7813370234216275}, {-1e-3, -9.11800647040274}, {-1e-9, -23.89701958453166}};
    for (const auto& [x, y] : w) {
        ++check::checks;
        if (!(std::abs(math::lambertw_m1(x) - y) <= 1e-14 * std::abs(y)))
            check::fail("lambertw_m1(" + std::to_string(x) + ")", std::to_string(math::lambertw_m1(x)));
    }
    check::is_true(std::isnan(math::lambertw_m1(0.1)), "lambertw_m1 outside its branch");
}

static void test_convolutions() {
    const auto re = uniform(108, 3000), im = uniform(109, 3000);
    std::vector<cd> a(3000);
    for (std::size_t i = 0; i < a.size(); ++i) a[i] = {re[i], im[i]};
    const auto b = uniform(110, 255);

    const auto same = math::oaconvolve(std::span<const cd>(a), std::span<const double>(b), math::ConvMode::Same);
    check::equal(same.size(), std::size_t{3000}, "oaconvolve same length");
    near(same[0], {-0.5586437401687058, -0.8000720952577257}, 1e-13, "oa same [0]");
    near(same[1], {0.42050265838382955, 0.6008858817029435}, 1e-13, "oa same [1]");
    near(same[1500], {-0.11550593271480498, 2.2292350843312274}, 1e-13, "oa same [1500]");
    near(same[2999], {-1.0360082668121866, -0.013587885366917565}, 1e-13, "oa same [2999]");
    const auto ref = direct(a, b);
    check::is_true(worst(same, ref, 127) <= 1e-13, "oaconvolve same = the direct convolution's centre");

    const auto full = math::oaconvolve(std::span<const double>(re), std::span<const double>(b), math::ConvMode::Full);
    check::equal(full.size(), std::size_t{3254}, "oaconvolve full length");
    const std::pair<std::size_t, double> fp[] = {
        {0, 0.01945748202279377}, {1, -0.08986566058515526}, {1500, 2.4969738530938432}, {3253, -0.07091917744322171}};
    for (const auto& [i, v] : fp) near(full[i], v, 1e-13, "oa full real [" + std::to_string(i) + "]");
    check::is_true(worst(full, direct(re, b)) <= 1e-13, "oaconvolve real = the direct convolution");

    const std::vector<cd> a100(a.begin(), a.begin() + 100);
    const auto fc = math::fftconvolve(std::span<const cd>(a), std::span<const cd>(a100), math::ConvMode::Full);
    check::equal(fc.size(), std::size_t{3099}, "fftconvolve full length");
    near(fc[0], {0.050732924049709915, 0.12043279341291131}, 1e-13, "fft full [0]");
    near(fc[50], {1.7212806546972088, -1.231697932640473}, 1e-13, "fft full [50]");
    near(fc[3098], {-0.14677394676136188, -0.12475467908156557}, 1e-13, "fft full [3098]");
    check::is_true(worst(fc, direct(a, a100)) <= 1e-12, "fftconvolve complex = the direct convolution");

    // real x complex, the operands swapped, and the fallbacks
    const auto rc = math::oaconvolve(std::span<const double>(b), std::span<const cd>(a), math::ConvMode::Full);
    check::is_true(worst(rc, ref) <= 1e-13, "oaconvolve real x complex");
    const std::vector<double> one{2.5};
    const auto scaled = math::oaconvolve(std::span<const double>(re), std::span<const double>(one),
                                         math::ConvMode::Same);
    bool exact = scaled.size() == re.size();
    for (std::size_t i = 0; exact && i < re.size(); ++i) exact = scaled[i] == re[i] * 2.5;
    check::is_true(exact, "a length-1 operand is a plain product");
    const std::vector<double> eq(b.begin(), b.end());
    check::is_true(worst(math::oaconvolve(std::span<const double>(b), std::span<const double>(eq),
                                          math::ConvMode::Full),
                         direct(b, eq)) <= 1e-13,
                   "equal lengths fall back to fftconvolve");
    check::equal(math::oaconvolve(std::span<const double>(), std::span<const double>(b), math::ConvMode::Full).size(),
                 std::size_t{0}, "an empty operand");
}

static void test_freqz() {
    const auto bb = uniform(106, 41);
    const std::vector<double> f{0.0, 125.0, 1999.5, -3000.25};
    const auto h = math::freqz(bb, f, 8000.0);
    const cd want[] = {{-2.3668995087910503, 0.0},
                       {1.5931144760879663, 2.5102435768371194},
                       {2.174122730545415, -0.5149342753078352},
                       {1.0813666902120436, 1.1710492905601704}};
    check::equal(h.size(), std::size_t{4}, "freqz length");
    // bit for bit against numpy on the host the pins came from; an ulp of
    // tolerance here because cos/sin are libm's, which differ by an ulp
    // between platforms
    for (std::size_t i = 0; i < 4; ++i) near(h[i], want[i], 1e-14, "freqz [" + std::to_string(i) + "]");
    // a direct DTFT over many frequencies (more than one 64-point block)
    std::vector<double> g(203);
    for (std::size_t i = 0; i < g.size(); ++i) g[i] = -4000.0 + 39.5 * static_cast<double>(i);
    const auto hg = math::freqz(bb, g, 8000.0);
    double w = 0.0;
    for (std::size_t i = 0; i < g.size(); ++i) {
        cd d = 0.0;
        for (std::size_t k = 0; k < bb.size(); ++k)
            d += bb[k] * std::exp(cd(0.0, -2 * std::numbers::pi * g[i] / 8000.0 * static_cast<double>(k)));
        w = std::max(w, std::abs(hg[i] - d));
    }
    check::is_true(w <= 1e-12, "freqz = the direct DTFT (" + std::to_string(w) + ")");
    check::equal(math::freqz(bb, std::vector<double>{}, 8000.0).size(), std::size_t{0}, "freqz at no frequency");
}

static std::vector<std::uint8_t> brute_max(const std::vector<std::uint8_t>& x, std::int64_t size) {
    const auto n = static_cast<std::int64_t>(x.size());
    auto at = [&](std::int64_t i) {   // reflect: d c b a | a b c d | d c b a, repeatedly
        const std::int64_t p = 2 * n;
        i = ((i % p) + p) % p;
        return x[static_cast<std::size_t>(i < n ? i : p - 1 - i)];
    };
    std::vector<std::uint8_t> out(x.size());
    for (std::int64_t i = 0; i < n; ++i) {
        std::uint8_t m = 0;
        for (std::int64_t k = 0; k < size; ++k) m = std::max(m, at(i - size / 2 + k));
        out[static_cast<std::size_t>(i)] = m;
    }
    return out;
}

static void test_maximum_filter() {
    const auto z = qrss_testgen::splitmix(107, 50);
    std::vector<std::uint8_t> m(z.size());
    for (std::size_t i = 0; i < z.size(); ++i) m[i] = static_cast<std::uint8_t>(z[i] % 5);
    const std::vector<std::uint8_t> m_want{0, 1, 3, 2, 0, 2, 4, 1, 3, 1, 0, 2, 1, 3, 4, 2, 1, 1, 0, 4, 3, 1, 1, 3, 1,
                                           3, 3, 4, 1, 1, 4, 2, 0, 1, 4, 0, 0, 4, 1, 2, 0, 3, 0, 3, 1, 2, 0, 3, 3, 4};
    check::is_true(m == m_want, "maximum_filter1d input");
    const std::vector<std::uint8_t> s2{0, 1, 3, 3, 2, 2, 4, 4, 3, 3, 1, 2, 2, 3, 4, 4, 2, 1, 1, 4, 4, 3, 1, 3, 3,
                                       3, 3, 4, 4, 1, 4, 4, 2, 1, 4, 4, 0, 4, 4, 2, 2, 3, 3, 3, 3, 2, 2, 3, 3, 4};
    const std::vector<std::uint8_t> s3{1, 3, 3, 3, 2, 4, 4, 4, 3, 3, 2, 2, 3, 4, 4, 4, 2, 1, 4, 4, 4, 3, 3, 3, 3,
                                       3, 4, 4, 4, 4, 4, 4, 2, 4, 4, 4, 4, 4, 4, 2, 3, 3, 3, 3, 3, 2, 3, 3, 4, 4};
    const std::vector<std::uint8_t> s7{3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 4,
                                       4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3, 4, 4, 4, 4};
    check::is_true(math::maximum_filter1d(m, 1) == m, "maximum_filter1d size 1");
    check::is_true(math::maximum_filter1d(m, 2) == s2, "maximum_filter1d size 2 (scipy)");
    check::is_true(math::maximum_filter1d(m, 3) == s3, "maximum_filter1d size 3 (scipy)");
    check::is_true(math::maximum_filter1d(m, 7) == s7, "maximum_filter1d size 7 (scipy)");
    check::is_true(math::maximum_filter1d(m, 60) == std::vector<std::uint8_t>(50, 4), "size 60 (scipy)");
    check::is_true(math::maximum_filter1d(m, 101) == std::vector<std::uint8_t>(50, 4), "size 101 (scipy)");
    bool all = true;
    for (std::size_t n : {1, 2, 3, 5, 9}) {
        const std::vector<std::uint8_t> x(m.begin(), m.begin() + static_cast<std::ptrdiff_t>(n));
        for (std::int64_t size = 1; size <= 25; ++size) all = all && math::maximum_filter1d(x, size) == brute_max(x, size);
    }
    for (std::int64_t size = 1; size <= 25; ++size) all = all && math::maximum_filter1d(m, size) == brute_max(m, size);
    check::is_true(all, "maximum_filter1d = brute-force reflect, short lines and long windows included");
    check::is_true(math::maximum_filter1d(std::vector<std::uint8_t>{}, 3).empty(), "maximum_filter1d of nothing");
    throws<std::runtime_error>([&] { math::maximum_filter1d(m, 0); }, "incorrect filter size", "size 0");
}

int main() {
    try {
        test_lengths();
        test_convolutions();
        test_freqz();
        test_maximum_filter();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FATAL: %s\n", e.what());
        return 1;
    }
    return check::report("qrss_convolve");
}

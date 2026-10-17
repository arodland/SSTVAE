// native/core/math/special against scipy.special 1.18.1: gammaincc,
// gammainc, gammainccinv, gammaincinv and the cephes pieces underneath
// (gammaln, gamma, erfc, the Hurwitz zeta) through one pinned digest
// (bitwise), the edge cases scipy defines, and a few pinned values.
// tests/test_qrss_native_special.py compares against scipy directly.
// The digest:
//
//   out = io.BytesIO()
//   AS = [0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 10.95, 20.0, 50.0, 100.0, 300.0, 1000.0]
//   for j, a in enumerate(AS):
//       u = unit(splitmix(100 + j, 64))              # infra_testgen.hpp
//       x = u * (3.0 * a + 10.0)
//       out.write(special.gammaincc(a, x).tobytes())
//       out.write(special.gammainc(a, x).tobytes())
//       q1 = unit(splitmix(200 + j, 32))
//       out.write(special.gammainccinv(a, q1).tobytes())
//       out.write(special.gammainccinv(a, q1 * 1e-9).tobytes())
//       out.write(special.gammaincinv(a, q1).tobytes())
//   xs = unit(splitmix(300, 64)) * 60.0 - 20.0
//   out.write(special.gammaln(xs).tobytes())
//   out.write(special.gamma(xs).tobytes())
//   out.write(special.erfc(xs * 0.5).tobytes())
//   out.write(special.zeta(unit(splitmix(301, 64)) * 20.0 + 1.01,
//                          unit(splitmix(302, 64)) * 50.0 + 0.01).tobytes())
//   sha256(out.getvalue())
//
// And the tails under em.py's template statistic, against em_pins.hpp
// (gen_em_pins.py, which has the formulas): ndtri and norm.isf through
// one digest (bit for bit), CPython's math.lgamma through another, and
// gamma.logsf as a table to 1e-12 relative, with the edges scipy defines.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <span>
#include <string>
#include <vector>

#include "em_pins.hpp"
#include "infra_testgen.hpp"
#include "math/special.hpp"
#include "qrss_check.hpp"
#include "util/sha256.hpp"

using namespace sstvae;

template <typename T>
static void append(std::vector<std::uint8_t>& out, const T& v) {
    const auto* p = reinterpret_cast<const std::uint8_t*>(&v);
    out.insert(out.end(), p, p + sizeof(T));
}

static std::vector<double> units(std::uint64_t seed, std::size_t n) {
    const auto z = qrss_testgen::splitmix(seed, n);
    std::vector<double> u(n);
    for (std::size_t i = 0; i < n; ++i) u[i] = qrss_testgen::unit(z[i]);
    return u;
}

static void test_digest() {
    const std::vector<double> as = {0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 10.95, 20.0, 50.0, 100.0, 300.0, 1000.0};
    std::vector<std::uint8_t> out;
    for (std::size_t j = 0; j < as.size(); ++j) {
        const double a = as[j];
        const auto u = units(100 + j, 64);
        std::vector<double> x(u.size());
        for (std::size_t i = 0; i < u.size(); ++i) x[i] = u[i] * (3.0 * a + 10.0);
        for (const double v : math::gammaincc(a, std::span<const double>(x))) append(out, v);
        for (const double v : x) append(out, math::gammainc(a, v));
        const auto q1 = units(200 + j, 32);
        for (const double q : q1) append(out, math::gammainccinv(a, q));
        for (const double q : q1) append(out, math::gammainccinv(a, q * 1e-9));
        for (const double q : q1) append(out, math::gammaincinv(a, q));
    }
    auto xs = units(300, 64);
    for (double& v : xs) v = v * 60.0 - 20.0;
    for (const double v : xs) append(out, math::lgam(v));
    for (const double v : xs) append(out, math::gamma(v));
    for (const double v : xs) append(out, math::erfc(v * 0.5));
    const auto zx = units(301, 64), zq = units(302, 64);
    for (std::size_t i = 0; i < 64; ++i) append(out, math::zeta(zx[i] * 20.0 + 1.01, zq[i] * 50.0 + 0.01));
    qrss_check::libm_equal(util::to_hex(util::sha256(std::span<const std::uint8_t>(out))),
                 std::string("86cb884af3078f75f019b7df50c06edf8b245c3f9f300b4ac851b502cf4eeaf7"),
                 "gammaincc / gammainc / gammainccinv / gammaincinv / gammaln / gamma / erfc / zeta digest");
}

static void test_edges() {
    const double inf = std::numeric_limits<double>::infinity();
    const double nan = std::numeric_limits<double>::quiet_NaN();
    // scipy: gammaincc(0, 1) = 0, gammaincc(0, 0) = nan, gammaincc(2, 0) = 1,
    // gammaincc(inf, 3) = 1, gammaincc(inf, inf) = nan, gammaincc(2, inf) = 0,
    // gammaincc(-1, 1) = gammaincc(1, -1) = gammaincc(nan, 1) = nan
    check::equal(math::gammaincc(0.0, 1.0), 0.0, "gammaincc(0, 1)");
    check::is_true(std::isnan(math::gammaincc(0.0, 0.0)), "gammaincc(0, 0) is nan");
    check::equal(math::gammaincc(2.0, 0.0), 1.0, "gammaincc(2, 0)");
    check::equal(math::gammaincc(inf, 3.0), 1.0, "gammaincc(inf, 3)");
    check::is_true(std::isnan(math::gammaincc(inf, inf)), "gammaincc(inf, inf) is nan");
    check::equal(math::gammaincc(2.0, inf), 0.0, "gammaincc(2, inf)");
    check::is_true(std::isnan(math::gammaincc(-1.0, 1.0)), "gammaincc(-1, 1) is nan");
    check::is_true(std::isnan(math::gammaincc(1.0, -1.0)), "gammaincc(1, -1) is nan");
    check::is_true(std::isnan(math::gammaincc(nan, 1.0)), "gammaincc(nan, 1) is nan");
    // gammainccinv(3, 0) = inf, (3, 1) = 0, (3, 1.5) = (3, -0.1) = (-1, 0.5) = nan
    check::equal(math::gammainccinv(3.0, 0.0), inf, "gammainccinv(3, 0)");
    check::equal(math::gammainccinv(3.0, 1.0), 0.0, "gammainccinv(3, 1)");
    check::is_true(std::isnan(math::gammainccinv(3.0, 1.5)), "gammainccinv(3, 1.5) is nan");
    check::is_true(std::isnan(math::gammainccinv(3.0, -0.1)), "gammainccinv(3, -0.1) is nan");
    check::is_true(std::isnan(math::gammainccinv(-1.0, 0.5)), "gammainccinv(-1, 0.5) is nan");
    check::is_true(math::gammaincc(1.0, std::span<const double>()).empty(), "empty vector in, empty out");
}

static void test_pinned() {
    // float(special.X(...)).hex()
    check::equal(math::gammaincc(1.0, 2.0), 0x1.152aaa3bf81ccp-3, "gammaincc(1, 2) = exp(-2)");
    check::equal(math::gammaincc(10.95, 12.0), 0x1.5de2a03060b84p-2, "gammaincc(10.95, 12)");
    check::equal(math::gammaincc(4.0, 30.0), 0x1.003e3195e326bp-31, "gammaincc(4, 30)");
    // a1_threshold's shape for M = 11 at FE_FS, at the default pfa
    check::equal(math::gammainccinv(121 / (11 + 20 / 36.), 1e-4), 0x1.aef8bac165daap+4,
                 "gammainccinv(121 / (11 + 20/36), 1e-4)");
}

// gen_em_pins.ndtri_inputs / ndtri_digest
static void test_ndtri_isf() {
    std::vector<double> q;
    for (const int j : em_pins::NDTRI_J)
        for (const double u : units(400 + static_cast<std::uint64_t>(j), 32)) q.push_back(u * std::ldexp(1.0, -j));
    for (const double u : units(450, 64)) q.push_back(1.0 - u);
    for (const double v : em_pins::NDTRI_SUBNORMAL) q.push_back(v);
    std::vector<std::uint8_t> out;
    for (const double v : q) append(out, math::ndtri(v));
    for (const double v : q) append(out, math::norm_isf(v));
    check::equal(util::to_hex(util::sha256(std::span<const std::uint8_t>(out))), std::string(em_pins::NDTRI_DIGEST),
                 "ndtri / norm_isf digest");

    const double inf = std::numeric_limits<double>::infinity();
    // scipy: ndtri(0) = -inf, ndtri(1) = inf, ndtri(0.5) = 0, ndtri(-0.1) = ndtri(1.1) = ndtri(nan) = nan;
    // norm.isf(0) = inf, isf(1) = -inf, isf(0.5) = +0.0, nan outside [0, 1]
    check::equal(math::ndtri(0.0), -inf, "ndtri(0)");
    check::equal(math::ndtri(1.0), inf, "ndtri(1)");
    check::equal(math::ndtri(0.5), 0.0, "ndtri(0.5)");
    for (const double bad : {-0.1, 1.1, std::numeric_limits<double>::quiet_NaN()}) {
        check::is_true(std::isnan(math::ndtri(bad)), "ndtri outside [0, 1] is nan");
        check::is_true(std::isnan(math::norm_isf(bad)), "norm_isf outside [0, 1] is nan");
    }
    check::equal(math::norm_isf(0.0), inf, "norm_isf(0)");
    check::equal(math::norm_isf(1.0), -inf, "norm_isf(1)");
    check::is_true(math::norm_isf(0.5) == 0.0 && !std::signbit(math::norm_isf(0.5)), "norm_isf(0.5) is +0.0");
    // float(stats.norm.isf(1e-6)).hex()
    check::equal(math::norm_isf(1e-6), 0x1.30381a9799f7ep+2, "norm_isf(1e-6)");
}

// gen_em_pins.lgamma_inputs / lgamma_digest
static void test_py_lgamma() {
    std::vector<double> xs = units(500, 256);
    for (double& v : xs) v = v * 60.0 - 20.0;
    for (const double v : units(501, 64)) xs.push_back(v * 2e6);
    for (const double v : units(502, 64)) xs.push_back(v * 1e-3);
    for (int i = 1; i <= 200; ++i) xs.push_back(static_cast<double>(i) * 0.5 + 0.25);
    std::vector<std::uint8_t> out;
    for (const double v : xs) append(out, math::py_lgamma(v));
    qrss_check::libm_equal(util::to_hex(util::sha256(std::span<const std::uint8_t>(out))), std::string(em_pins::LGAMMA_DIGEST),
                 "math.lgamma digest");
    const double inf = std::numeric_limits<double>::infinity();
    // math.lgamma: nan for nan, inf for +-inf, 0 for 1 and 2; where Python
    // raises (a pole, an overflow) this returns inf
    check::is_true(std::isnan(math::py_lgamma(std::numeric_limits<double>::quiet_NaN())), "lgamma(nan)");
    check::equal(math::py_lgamma(inf), inf, "lgamma(inf)");
    check::equal(math::py_lgamma(-inf), inf, "lgamma(-inf)");
    check::equal(math::py_lgamma(1.0), 0.0, "lgamma(1)");
    check::equal(math::py_lgamma(2.0), 0.0, "lgamma(2)");
    check::equal(math::py_lgamma(0.0), inf, "lgamma(0): Python's ValueError");
    check::equal(math::py_lgamma(-3.0), inf, "lgamma(-3): Python's ValueError");
    check::equal(math::py_lgamma(1e308), inf, "lgamma(1e308): Python's OverflowError");
    // float(math.lgamma(0.5)).hex() = log(sqrt(pi))
    check::equal(math::py_lgamma(0.5), 0x1.250d048e7a1c0p-1, "lgamma(0.5)");
}

static void test_gamma_logsf() {
    double worst = 0.0;
    for (const em_pins::LogSf& r : em_pins::GAMMA_LOGSF) {
        const double got = math::gamma_logsf(r.x, r.a);
        char what[128];
        std::snprintf(what, sizeof what, "gamma.logsf(%a, %a)", r.x, r.a);
        if (r.logsf == 0.0 || std::isinf(r.logsf)) {
            check::equal(got, r.logsf, what);
            continue;
        }
        const double rel = std::abs(got - r.logsf) / std::abs(r.logsf);
        if (rel > worst) worst = rel;
        check::is_true(rel <= 1e-12, std::string(what) + " to 1e-12");
    }
    std::printf("gamma_logsf: worst relative error against scipy %.3g over %zu values\n", worst,
                std::size(em_pins::GAMMA_LOGSF));
    const double inf = std::numeric_limits<double>::infinity(), nan = std::numeric_limits<double>::quiet_NaN();
    // scipy: nan for a <= 0 or a nan x; 0 for x <= 0; -inf at x = inf
    check::is_true(std::isnan(math::gamma_logsf(1.0, 0.0)), "gamma.logsf(1, a=0) is nan");
    check::is_true(std::isnan(math::gamma_logsf(1.0, -2.0)), "gamma.logsf(1, a=-2) is nan");
    check::is_true(std::isnan(math::gamma_logsf(nan, 3.0)), "gamma.logsf(nan, 3) is nan");
    check::equal(math::gamma_logsf(0.0, 3.0), 0.0, "gamma.logsf(0, 3)");
    check::equal(math::gamma_logsf(-5.0, 3.0), 0.0, "gamma.logsf(-5, 3)");
    check::equal(math::gamma_logsf(-inf, 3.0), 0.0, "gamma.logsf(-inf, 3)");
    check::equal(math::gamma_logsf(inf, 3.0), -inf, "gamma.logsf(inf, 3)");
    check::equal(math::gamma_logsf(4800.0, 300.0), -inf, "gamma.logsf(4800, 300): gammaincc underflows");
}

int main() {
    try {
        test_digest();
        test_edges();
        test_pinned();
        test_ndtri_isf();
        test_py_lgamma();
        test_gamma_logsf();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FATAL: %s\n", e.what());
        return 1;
    }
    return check::report("qrss_special");
}

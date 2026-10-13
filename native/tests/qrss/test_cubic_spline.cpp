// native/core/math/cubic_spline against scipy.interpolate.CubicSpline
// 1.18.1 (complex values, not-a-knot, extrapolating): the coefficients
// and the values and first three derivatives, inside and outside the
// knots, on uniform and irregular knots, through one pinned digest
// (bitwise); the three-point case to tolerance (scipy's dense solve);
// the refusals with scipy's texts. tests/test_qrss_native_cubic_spline.py
// compares against scipy directly. The digest:
//
//   out = io.BytesIO()
//   for n in [2, 4, 5, 10, 161, 1000]:
//       z = splitmix(500 + n, 2 * n)                          # infra_testgen.hpp
//       y = (unit(z[:n]) - 0.5) + 1j * (unit(z[n:]) - 0.5)
//       x = np.arange(n, dtype=float) + 100.0
//       xi = np.cumsum(unit(splitmix(600 + n, n)) * 3.0 + 0.01)
//       for xx in (x, xi):
//           s = CubicSpline(xx, y)
//           out.write(s.c.tobytes())
//           e = xx[0] - 2.0 + unit(splitmix(700 + n, 50)) * (xx[-1] - xx[0] + 4.0)
//           for nu in range(4):
//               out.write(s(e, nu).tobytes())
//           out.write(s(xx, 0).tobytes())
//   sha256(out.getvalue())

#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "infra_testgen.hpp"
#include "math/cubic_spline.hpp"
#include "math/errors.hpp"
#include "qrss_check.hpp"
#include "util/sha256.hpp"

using namespace sstvae;
using cd = std::complex<double>;

template <typename T>
static void append_all(std::vector<std::uint8_t>& out, const std::vector<T>& v) {
    const auto* p = reinterpret_cast<const std::uint8_t*>(v.data());
    out.insert(out.end(), p, p + v.size() * sizeof(T));
}

template <typename E, typename F>
static void throws(F f, const std::string& text, const std::string& what) {
    try {
        f();
    } catch (const E& e) {
        check::equal(std::string(e.what()), text, what);
        return;
    }
    check::fail(what, "did not throw");
}

// check::equal renders through std::to_string, which has no complex overload
static void ceq(cd got, cd want, const std::string& what) {
    check::is_true(got == want, what);
    if (got != want)
        std::fprintf(stderr, "  got (%.17g, %.17g), want (%.17g, %.17g)\n", got.real(), got.imag(), want.real(),
                     want.imag());
}

static void test_digest() {
    std::vector<std::uint8_t> out;
    for (const std::size_t n : {2, 4, 5, 10, 161, 1000}) {
        const auto z = qrss_testgen::splitmix(500 + n, 2 * n);
        std::vector<cd> y(n);
        for (std::size_t i = 0; i < n; ++i)
            y[i] = cd(qrss_testgen::unit(z[i]) - 0.5, qrss_testgen::unit(z[n + i]) - 0.5);
        std::vector<double> x(n), xi(n);
        const auto zi = qrss_testgen::splitmix(600 + n, n);
        double acc = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            x[i] = static_cast<double>(i) + 100.0;
            acc += qrss_testgen::unit(zi[i]) * 3.0 + 0.01;  // np.cumsum: sequential
            xi[i] = acc;
        }
        const auto ze = qrss_testgen::splitmix(700 + n, 50);
        for (const auto* xx : {&x, &xi}) {
            const math::CubicSpline s(*xx, y);
            append_all(out, s.c());
            std::vector<double> e(50);
            for (std::size_t i = 0; i < 50; ++i)
                e[i] = (*xx)[0] - 2.0 + qrss_testgen::unit(ze[i]) * ((*xx)[n - 1] - (*xx)[0] + 4.0);
            for (int nu = 0; nu < 4; ++nu) append_all(out, s(std::span<const double>(e), nu));
            append_all(out, s(std::span<const double>(*xx), 0));
        }
    }
    check::equal(util::to_hex(util::sha256(std::span<const std::uint8_t>(out))),
                 std::string("9063596b61f38f1d23160312b16afe5caae69c2de0db442a54b93095384c22b2"),
                 "CubicSpline coefficients / values / derivatives digest");
}

static void test_three_points() {
    // CubicSpline([0., 1., 3.], [1+2j, -0.5+0.25j, 2-1j])(2.0) == (-0.16666666666666663-0.75j)
    // (scipy.linalg.solve's dense 3x3: agreement to rounding, not to the bit)
    const math::CubicSpline s(std::vector<double>{0.0, 1.0, 3.0}, std::vector<cd>{{1, 2}, {-0.5, 0.25}, {2, -1}});
    const cd v = s(2.0);
    check::is_true(std::abs(v - cd(-0.16666666666666663, -0.75)) < 1e-14, "three points, s(2.0)");
    check::is_true(std::abs(s(1.0) - cd(-0.5, 0.25)) < 1e-15, "three points interpolate");
}

static void test_behaviour() {
    // two points: the line through them, extended both ways; derivatives past
    // the cubic term are 0; NaN in, NaN (+0j) out
    const math::CubicSpline s(std::vector<double>{1.0, 3.0}, std::vector<cd>{{1, 0}, {5, -2}});
    ceq(s(2.0), cd(3.0, -1.0), "line, midpoint");
    ceq(s(5.0), cd(9.0, -4.0), "line, extrapolated right");
    ceq(s(-1.0), cd(-3.0, 2.0), "line, extrapolated left");
    ceq(s(2.0, 1), cd(2.0, -1.0), "line, slope");
    ceq(s(2.0, 4), cd(0.0, 0.0), "fourth derivative of a cubic");
    const cd nan_out = s(std::numeric_limits<double>::quiet_NaN());
    check::is_true(std::isnan(nan_out.real()) && nan_out.imag() == 0.0, "NaN -> nan+0j");
    check::equal(s.size(), std::size_t(2), "size");
    check::equal(s.c().size(), std::size_t(4), "c is 4 x (n - 1)");
}

static void test_refusals() {
    const std::vector<double> x2 = {0.0, 1.0};
    const std::vector<cd> y2 = {{1, 0}, {2, 0}};
    const double inf = std::numeric_limits<double>::infinity();
    throws<std::invalid_argument>([] { math::CubicSpline(std::vector<double>{0.0}, std::vector<cd>{{1, 0}}); },
                                  "`x` must contain at least 2 elements.", "one point");
    throws<std::invalid_argument>([&] { math::CubicSpline(x2, std::vector<cd>{{1, 0}}); },
                                  "The length of `y` along `axis`=0 doesn't match the length of `x`",
                                  "length mismatch");
    throws<std::invalid_argument>([&] { math::CubicSpline(std::vector<double>{0.0, inf}, y2); },
                                  "`x` must contain only finite values.", "infinite x");
    throws<std::invalid_argument>([&] { math::CubicSpline(x2, std::vector<cd>{{1, 0}, {2, inf}}); },
                                  "`y` must contain only finite values.", "infinite y");
    throws<std::invalid_argument>([&] { math::CubicSpline(std::vector<double>{1.0, 1.0}, y2); },
                                  "`x` must be strictly increasing sequence.", "repeated x");
    throws<std::invalid_argument>([&] { math::CubicSpline(std::vector<double>{1.0, 0.0}, y2); },
                                  "`x` must be strictly increasing sequence.", "decreasing x");
    // a slope that overflows: (1e308 - -1e308) / 1 is inf
    throws<std::invalid_argument>(
        [] { math::CubicSpline(std::vector<double>{0.0, 1.0}, std::vector<cd>{{-1e308, 0}, {1e308, 0}}); },
        "`dydx` must contain only finite values.", "infinite slope");
    const math::CubicSpline s(x2, y2);
    throws<std::invalid_argument>([&] { (void)s(0.5, -1); }, "Order of derivative cannot be negative",
                                  "negative derivative");
}

int main() {
    try {
        test_digest();
        test_three_points();
        test_behaviour();
        test_refusals();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FATAL: %s\n", e.what());
        return 1;
    }
    return check::report("qrss_cubic_spline");
}

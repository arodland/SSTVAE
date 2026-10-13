// native/core/math/npstats against numpy: median, mean, interp and polyval
// through one pinned digest (bitwise), polyfit against pinned values, and
// the refusals. tests/test_qrss_native_math.py compares against numpy
// directly. The digest:
//
//   out = io.BytesIO()
//   for n in list(range(1, 41)) + [100, 1001]:
//       z = splitmix(2000 + n, n)                     # infra_testgen.hpp
//       v = unit(z) * 2000.0 - 1000.0
//       v[::7] = np.round(v[::7])                     # ties for the median
//       v32 = v.astype(np.float32)
//       xp = np.sort(v); fp = v[::-1].copy()
//       x = unit(splitmix(3000 + n, 64)) * 2400.0 - 1200.0
//       for r in (np.median(v), np.mean(v), np.median(v32), np.mean(v32)):
//           out.write(np.asarray(r).tobytes())
//       out.write(np.interp(x, xp, fp).tobytes())
//       out.write(np.interp(x, xp, fp, left=-7.0, right=9.0).tobytes())
//       out.write(np.polyval(v[:4], x).tobytes())
//   sha256(out.getvalue())

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "infra_testgen.hpp"
#include "math/errors.hpp"
#include "math/npstats.hpp"
#include "qrss_check.hpp"
#include "util/sha256.hpp"

using namespace sstvae;

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

template <typename T>
static void append(std::vector<std::uint8_t>& out, const T& v) {
    const auto* p = reinterpret_cast<const std::uint8_t*>(&v);
    out.insert(out.end(), p, p + sizeof(T));
}

template <typename T>
static void append_all(std::vector<std::uint8_t>& out, const std::vector<T>& v) {
    for (const T& x : v) append(out, x);
}

static void test_digest() {
    std::vector<std::size_t> lengths;
    for (std::size_t n = 1; n <= 40; ++n) lengths.push_back(n);
    lengths.push_back(100);
    lengths.push_back(1001);
    std::vector<std::uint8_t> out;
    for (const std::size_t n : lengths) {
        const auto z = qrss_testgen::splitmix(2000 + n, n);
        std::vector<double> v(n);
        for (std::size_t i = 0; i < n; ++i) v[i] = qrss_testgen::unit(z[i]) * 2000.0 - 1000.0;
        for (std::size_t i = 0; i < n; i += 7) v[i] = std::nearbyint(v[i]);
        std::vector<float> v32(v.begin(), v.end());
        std::vector<double> xp = v;
        std::sort(xp.begin(), xp.end());
        std::vector<double> fp(v.rbegin(), v.rend());
        const auto zx = qrss_testgen::splitmix(3000 + n, 64);
        std::vector<double> x(64);
        for (std::size_t i = 0; i < 64; ++i) x[i] = qrss_testgen::unit(zx[i]) * 2400.0 - 1200.0;
        append(out, math::median(std::span<const double>(v)));
        append(out, math::mean(std::span<const double>(v)));
        append(out, math::median(std::span<const float>(v32)));
        append(out, math::mean(std::span<const float>(v32)));
        append_all(out, math::interp(std::span<const double>(x), xp, fp));
        append_all(out, math::interp(std::span<const double>(x), xp, fp, -7.0, 9.0));
        const std::vector<double> p(v.begin(), v.begin() + static_cast<std::ptrdiff_t>(std::min<std::size_t>(4, n)));
        append_all(out, math::polyval(p, std::span<const double>(x)));
    }
    check::equal(util::to_hex(util::sha256(std::span<const std::uint8_t>(out))),
                 std::string("802c57a94a49c7181cbf65419f949a83954413b539126d02199ccd9c9042e6d8"),
                 "median / mean / interp / polyval digest");
}

static void test_edges() {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    check::is_true(std::isnan(math::median(std::vector<double>{})), "median([]) is NaN");
    check::is_true(std::isnan(math::median(std::vector<double>{1.0, nan, 2.0})), "a NaN makes the median NaN");
    check::is_true(!std::signbit(math::median(std::vector<double>{-0.0})), "median([-0.0]) is +0.0, as numpy's");
    check::equal(math::median(std::vector<double>{3.0, 1.0, 2.0}), 2.0, "odd median");
    check::equal(math::median(std::vector<double>{4.0, 1.0, 3.0, 2.0}), 2.5, "even median");
    check::equal(math::median(std::vector<float>{1.0f, 2.0f}), 1.5f, "float32 median stays float32");
    check::is_true(std::isnan(math::mean(std::vector<double>{})), "mean([]) is NaN");
    // interp: one sample point (no NaN check, numpy's len == 1 branch)
    check::equal(math::interp(5.0, std::vector<double>{1.0}, std::vector<double>{7.0}), 7.0, "one point: inside");
    check::equal(math::interp(-5.0, std::vector<double>{1.0}, std::vector<double>{7.0}, -1.0, 2.0), -1.0,
                 "one point: left");
    check::is_true(std::isnan(math::interp(nan, std::vector<double>{0.0, 1.0}, std::vector<double>{0.0, 1.0})),
                   "interp(nan) is nan");
    throws<std::invalid_argument>([] { math::interp(1.0, std::vector<double>{}, std::vector<double>{}); },
                                  "array of sample points is empty", "interp, empty xp");
    throws<std::invalid_argument>(
        [] { math::interp(1.0, std::vector<double>{0.0, 1.0}, std::vector<double>{1.0}); },
        "fp and xp are not of the same length.", "interp, length mismatch");
}

static void test_polyfit() {
    // np.polyfit([0.0, 1.0, 2.0, 4.0], [1.0, 3.2, 4.9, 9.1], 1, w=[1.0, 2.0, 1.0, 0.5])
    const std::vector<double> x = {0.0, 1.0, 2.0, 4.0}, y = {1.0, 3.2, 4.9, 9.1}, w = {1.0, 2.0, 1.0, 0.5};
    check::close(math::polyfit(x, y, 1, std::span<const double>(w)),
                 std::vector<double>{1.973076923076923, 1.1461538461538463}, 1e-14, "weighted line");
    check::close(math::polyfit(std::vector<double>{0.0, 1.0, 2.0}, std::vector<double>{1.0, 3.0, 5.0}, 1),
                 std::vector<double>{2.0, 1.0}, 1e-14, "exact line");
    // np.polyfit([1.0], [1.0], 1) -> [0.5, 0.5] (min-norm); x=[3,3] -> [0.25, 0.75]
    check::close(math::polyfit(std::vector<double>{1.0}, std::vector<double>{1.0}, 1),
                 std::vector<double>{0.5, 0.5}, 1e-14, "one point, degree 1");
    check::close(math::polyfit(std::vector<double>{3.0, 3.0}, std::vector<double>{1.0, 2.0}, 1),
                 std::vector<double>{0.25, 0.75}, 1e-14, "two equal x");
    const auto c = math::polyfit(std::vector<double>{0.0, 1.0}, std::vector<double>{std::nan(""), 2.0}, 1);
    check::is_true(std::isnan(c[0]) && std::isnan(c[1]), "NaN y: NaN coefficients");
    throws<math::LinAlgError>(
        [] { math::polyfit(std::vector<double>{0.0, 0.0}, std::vector<double>{1.0, 2.0}, 1); },
        "SVD did not converge in Linear Least Squares", "x all zero");
    throws<math::LinAlgError>(
        [] { math::polyfit(std::vector<double>{std::nan(""), 1.0}, std::vector<double>{1.0, 2.0}, 1); },
        "SVD did not converge in Linear Least Squares", "NaN x");
    throws<std::invalid_argument>(
        [] { math::polyfit(std::vector<double>{0.0, 1.0}, std::vector<double>{1.0, 2.0}, -1); },
        "expected deg >= 0", "negative degree");
    throws<math::TypeError>([] { math::polyfit(std::vector<double>{}, std::vector<double>{}, 1); },
                            "expected non-empty vector for x", "empty x");
    throws<math::TypeError>(
        [] { math::polyfit(std::vector<double>{0.0, 1.0}, std::vector<double>{1.0}, 1); },
        "expected x and y to have same length", "length mismatch");
    check::equal(math::polyval(std::vector<double>{2.0, -1.0, 0.5}, 3.0), (2.0 * 3.0 - 1.0) * 3.0 + 0.5, "polyval");
}

int main() {
    try {
        test_digest();
        test_edges();
        test_polyfit();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FATAL: %s\n", e.what());
        return 1;
    }
    return check::report("qrss_npstats");
}

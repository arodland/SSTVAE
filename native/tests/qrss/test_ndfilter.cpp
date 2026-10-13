// native/core/math/ndfilter against scipy.ndimage.median_filter(...,
// mode="reflect") 1.18.1, float64 and float32, through one pinned digest
// (bitwise) -- windows narrower than, equal to and wider than the array,
// including scipy's lim2 == 0 case -- plus small pinned cases and the
// refusal. tests/test_qrss_native_ndfilter.py compares against scipy
// directly. The digest:
//
//   out = io.BytesIO()
//   for n, w in [(1, 3), (2, 5), (3, 7), (4, 9), (5, 5), (5, 11), (6, 13), (7, 13),
//                (13, 13), (14, 13), (100, 7), (1201, 161), (9601, 321)]:
//       v = np.round(unit(splitmix(400 + n + w, n)) * 10.0 * 4) / 4   # infra_testgen.hpp; ties
//       out.write(ndimage.median_filter(v, size=w, mode="reflect").tobytes())
//       out.write(ndimage.median_filter(v.astype(np.float32), size=w, mode="reflect").tobytes())
//   sha256(out.getvalue())

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "infra_testgen.hpp"
#include "math/ndfilter.hpp"
#include "qrss_check.hpp"
#include "util/sha256.hpp"

using namespace sstvae;

template <typename T>
static void append_all(std::vector<std::uint8_t>& out, const std::vector<T>& v) {
    const auto* p = reinterpret_cast<const std::uint8_t*>(v.data());
    out.insert(out.end(), p, p + v.size() * sizeof(T));
}

static void test_digest() {
    const std::vector<std::pair<std::size_t, int>> cases = {
        {1, 3},   {2, 5},   {3, 7},    {4, 9},    {5, 5},     {5, 11},    {6, 13},
        {7, 13},  {13, 13}, {14, 13}, {100, 7}, {1201, 161}, {9601, 321}};
    std::vector<std::uint8_t> out;
    for (const auto& [n, w] : cases) {
        const auto z = qrss_testgen::splitmix(400 + n + static_cast<std::size_t>(w), n);
        std::vector<double> v(n);
        for (std::size_t i = 0; i < n; ++i) v[i] = std::nearbyint(qrss_testgen::unit(z[i]) * 10.0 * 4) / 4;
        const std::vector<float> v32(v.begin(), v.end());
        append_all(out, math::median_filter(std::span<const double>(v), w));
        append_all(out, math::median_filter(std::span<const float>(v32), w));
    }
    check::equal(util::to_hex(util::sha256(std::span<const std::uint8_t>(out))),
                 std::string("c401cde7753d6c5caae0eb314d7d83628be300175bebfbe279dd3114419d59ea"),
                 "median_filter digest");
}

static void test_small() {
    // v = np.arange(n, dtype=float)[::-1] * 2.0 + np.array([0.5, -1.0, 3.0, 0.0, 1.5, -2.0])[:n]
    // ndimage.median_filter(v, size=w, mode="reflect")
    const std::vector<std::pair<std::vector<double>, std::pair<int, std::vector<double>>>> cases = {
        {{4.5, 1.0, 3.0}, {7, {3.0, 3.0, 3.0}}},
        {{2.5, -1.0}, {5, {-1.0, 2.5}}},
        {{6.5, 3.0, 5.0, 0.0}, {9, {3.0, 5.0, 3.0, 5.0}}},
        {{8.5, 5.0, 7.0, 2.0, 1.5}, {5, {7.0, 7.0, 5.0, 2.0, 2.0}}},
        {{10.5, 7.0, 9.0, 4.0, 3.5, -2.0}, {13, {4.0, 4.0, 4.0, 7.0, 7.0, 7.0}}},
    };
    for (const auto& [v, ww] : cases) {
        const auto& [w, want] = ww;
        const std::string what = "median_filter n=" + std::to_string(v.size()) + " w=" + std::to_string(w);
        check::equal(math::median_filter(std::span<const double>(v), w) == want, true, what);
        const std::vector<float> v32(v.begin(), v.end()), want32(want.begin(), want.end());
        check::equal(math::median_filter(std::span<const float>(v32), w) == want32, true, what + " float32");
    }
    // a width-1 window is the identity; an empty array is empty
    const std::vector<double> v = {3.0, -1.0, 2.0};
    check::equal(math::median_filter(std::span<const double>(v), 1) == v, true, "size 1 is the identity");
    check::is_true(math::median_filter(std::span<const double>(), 5).empty(), "empty in, empty out");
    // reflect is half-sample symmetric: [1, 9, 2] w3 -> [1, 2, 2] ((1)1 9 2(2))
    check::equal(math::median_filter(std::span<const double>(std::vector<double>{1.0, 9.0, 2.0}), 3) ==
                     std::vector<double>{1.0, 2.0, 2.0},
                 true, "reflect at both ends");
}

static void test_refusal() {
    for (const int size : {0, 4, -3}) {
        try {
            (void)math::median_filter(std::span<const double>(std::vector<double>{1.0, 2.0}), size);
            check::fail("size " + std::to_string(size), "did not throw");
        } catch (const std::invalid_argument& e) {
            check::equal(std::string(e.what()),
                         "median_filter: size must be odd and positive, not " + std::to_string(size),
                         "size " + std::to_string(size) + " refused");
        }
    }
}

int main() {
    try {
        test_digest();
        test_small();
        test_refusal();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FATAL: %s\n", e.what());
        return 1;
    }
    return check::report("qrss_ndfilter");
}

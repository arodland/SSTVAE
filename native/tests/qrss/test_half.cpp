// native/core/qrss/half against numpy's float16, through pinned digests
// (tests/test_qrss_native_infra.py compares the two directly).
//
//   h = np.arange(65536, dtype=np.uint16)
//   sha256(h.view(np.float16).astype(np.float64).tobytes())   -> to_double
//   sha256(h.view(np.float16).astype(np.float32).tobytes())   -> to_float
//
// from_double over ~1e6 inputs (`inputs()` below, splitmix from
// infra_testgen.hpp): every positive finite half, the midpoint to the next
// one (65536 past the last) and the doubles either side of it -- the ties,
// and the values a float32 detour would round to a tie and then the wrong
// way -- all of them negated too; 500,000 raw 64-bit patterns (NaN
// payloads, inf, subnormal and huge doubles); 250,000 values scaled by
// 2^-30..2^33. As numpy:
//
//   pos = np.arange(0x7c00, dtype=np.uint16); lo = pos.view(np.float16).astype(np.float64)
//   hi = np.append(lo[1:], 65536.0); mid = (lo + hi) / 2
//   a = np.stack([lo, mid, np.nextafter(mid, 0.0), np.nextafter(mid, np.inf)], 1).reshape(-1)
//   a = np.concatenate([a, -a]); raw = splitmix(1, 500000).view(np.float64)
//   z = splitmix(2, 250000); k = (z & np.uint64(63)).astype(np.int64) - 30
//   x = np.concatenate([a, raw, np.ldexp(unit(z) * 2.0 - 1.0, k)])
//   sha256(x.astype(np.float16).tobytes())                                      -> from_double
//   keep = ~np.isnan(x) & ((np.abs(x) <= np.finfo(np.float32).max) | np.isinf(x))
//   sha256(x[keep].astype(np.float32).astype(np.float16).tobytes())             -> from_float

#include <bit>
#include <cmath>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <vector>

#include "infra_testgen.hpp"
#include "qrss/half.hpp"
#include "qrss_check.hpp"

using namespace sstvae;
namespace half = sstvae::qrss::half;
using qrss_check::digest_is;

static std::vector<double> inputs() {
    std::vector<double> lo, mid;
    for (std::uint32_t b = 0; b < 0x7c00; ++b) lo.push_back(half::to_double(static_cast<std::uint16_t>(b)));
    std::vector<double> a;
    for (std::size_t i = 0; i < lo.size(); ++i) {
        const double hi = i + 1 < lo.size() ? lo[i + 1] : 65536.0;
        const double m = (lo[i] + hi) / 2;
        a.insert(a.end(), {lo[i], m, std::nextafter(m, 0.0), std::nextafter(m, std::numeric_limits<double>::infinity())});
    }
    const std::size_t n = a.size();
    for (std::size_t i = 0; i < n; ++i) a.push_back(-a[i]);
    for (const std::uint64_t z : qrss_testgen::splitmix(1, 500000)) a.push_back(std::bit_cast<double>(z));
    for (const std::uint64_t z : qrss_testgen::splitmix(2, 250000)) {
        const int k = static_cast<int>(z & 63u) - 30;
        a.push_back(std::ldexp(qrss_testgen::unit(z) * 2.0 - 1.0, k));
    }
    return a;
}

static void test_every_half() {
    std::vector<double> d(65536);
    std::vector<float> f(65536);
    for (std::uint32_t b = 0; b < 65536; ++b) {
        d[b] = half::to_double(static_cast<std::uint16_t>(b));
        f[b] = half::to_float(static_cast<std::uint16_t>(b));
    }
    digest_is(d, "abaa35fb7387cc874a8d8464aa18cd64baa87781a69f1c96a5aa5e0626d48a26", "to_double, all 65536");
    digest_is(f, "f4fdd084f85448d28c84f20fabf4022ba938e40b7f382d2727dec6f41ac6267a", "to_float, all 65536");
    // Back again, bit for bit, NaN payloads included.
    int bad_d = 0, bad_f = 0;
    for (std::uint32_t b = 0; b < 65536; ++b) {
        bad_d += half::from_double(d[b]) != b;
        bad_f += half::from_float(f[b]) != b;
    }
    check::equal(bad_d, 0, "from_double(to_double(h)) == h");
    check::equal(bad_f, 0, "from_float(to_float(h)) == h");
}

static void test_conversions() {
    const std::vector<double> x = inputs();
    check::equal(x.size(), std::size_t(1003952), "input count");
    digest_is(half::from_double(x), "48a64fc1b04dcde9a4524ce08505e80f8f98b3698a44aed07330973d334b34cd",
              "from_double");
    std::vector<std::uint16_t> h32;
    const double fmax = std::numeric_limits<float>::max();
    for (const double v : x)
        if (!std::isnan(v) && (std::fabs(v) <= fmax || std::isinf(v)))
            h32.push_back(half::from_float(static_cast<float>(v)));
    check::equal(h32.size(), std::size_t(784939), "float input count");
    digest_is(h32, "c1fb23240b38e89d51457cb4bff414031d641637ea274913f6cc0237bbe72f6d", "from_float");
}

static void test_landmarks() {
    // Values whose answers are known without numpy.
    check::equal(half::from_double(1.0), std::uint16_t(0x3c00), "1.0");
    check::equal(half::from_double(-2.0), std::uint16_t(0xc000), "-2.0");
    check::equal(half::from_double(65504.0), std::uint16_t(0x7bff), "max half");
    check::equal(half::from_double(65519.99), std::uint16_t(0x7bff), "just under the overflow tie");
    check::equal(half::from_double(65520.0), std::uint16_t(0x7c00), "the overflow tie rounds to inf");
    check::equal(half::from_double(1e300), std::uint16_t(0x7c00), "overflow");
    check::equal(half::from_double(-1e300), std::uint16_t(0xfc00), "-overflow");
    check::equal(half::from_double(0x1p-24), std::uint16_t(0x0001), "smallest subnormal");
    check::equal(half::from_double(0x1p-25), std::uint16_t(0x0000), "half of it ties to even 0");
    check::equal(half::from_double(std::nextafter(0x1p-25, 1.0)), std::uint16_t(0x0001), "just above rounds up");
    check::equal(half::from_double(3 * 0x1p-25), std::uint16_t(0x0002), "1.5 subnormal units tie to 2");
    check::equal(half::from_double(-0.0), std::uint16_t(0x8000), "-0");
    check::equal(half::from_double(0x1p-14), std::uint16_t(0x0400), "smallest normal");
    check::equal(half::from_double(std::nextafter(0x1p-14, 0.0)), std::uint16_t(0x0400),
                 "just under the smallest normal rounds up into it");
    // 1 + 2^-11 is the tie between 1 and 1 + 2^-10: to even (1). One double
    // ulp above it must go up -- the case a float32 detour gets wrong.
    check::equal(half::from_double(1.0 + 0x1p-11), std::uint16_t(0x3c00), "tie to even");
    check::equal(half::from_double(1.0 + 0x1p-11 + 0x1p-52), std::uint16_t(0x3c01), "above the tie");
    check::equal(half::from_double(1.0 + 3 * 0x1p-11), std::uint16_t(0x3c02), "tie to even, upward");
    check::equal(half::from_double(std::numeric_limits<double>::infinity()), std::uint16_t(0x7c00), "inf");
    check::equal(half::from_double(std::bit_cast<double>(0x7ff8000000000000ULL)), std::uint16_t(0x7e00), "qNaN");
    check::equal(half::from_double(std::bit_cast<double>(0x7ff0000000000001ULL)), std::uint16_t(0x7c01),
                 "a NaN whose top payload bits are zero stays a NaN");
    check::equal(std::bit_cast<std::uint64_t>(half::to_double(std::uint16_t(0x7c01))), 0x7ff0040000000000ULL,
                 "half NaN payload to double");
}

int main() {
    try {
        test_every_half();
        test_conversions();
        test_landmarks();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FATAL: %s\n", e.what());
        return 1;
    }
    return check::report("qrss_half");
}

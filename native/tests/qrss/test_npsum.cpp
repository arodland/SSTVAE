// native/core/qrss/npsum against numpy's np.sum / np.mean, through one
// pinned digest over every length 1..300 and a few long ones (bitwise;
// tests/test_qrss_native_infra.py compares against numpy directly):
//
//   out = io.BytesIO()
//   for n in list(range(1, 301)) + [1000, 8191, 8192, 8193, 16385, 50600, 100000]:
//       z = splitmix(1000 + n, n)                     # infra_testgen.hpp
//       v = unit(z) * 2000.0 - 1000.0
//       v32 = v.astype(np.float32)
//       k = (z & np.uint64(63)).astype(np.int64) - 32
//       w32 = np.ldexp(v, k).astype(np.float32)       # wide range: float64 sums round
//       c = v + 1j * v[::-1]
//       for r in (np.sum(v), np.mean(v), np.sum(v32), np.mean(v32),
//                 np.sum(w32, dtype=np.float64), np.mean(w32, dtype=np.float64),
//                 np.sum(c), np.sum(c.astype(np.complex64))):
//           out.write(r.tobytes())
//   sha256(out.getvalue())

#include <cmath>
#include <complex>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

#include "infra_testgen.hpp"
#include "qrss/npsum.hpp"
#include "qrss_check.hpp"
#include "util/sha256.hpp"

using namespace sstvae;
namespace npsum = sstvae::qrss::npsum;

template <typename T>
static void append(std::vector<std::uint8_t>& out, const T& v) {
    const auto* p = reinterpret_cast<const std::uint8_t*>(&v);
    out.insert(out.end(), p, p + sizeof(T));
}

static void test_digest() {
    std::vector<std::size_t> lengths;
    for (std::size_t n = 1; n <= 300; ++n) lengths.push_back(n);
    for (const std::size_t n : {1000, 8191, 8192, 8193, 16385, 50600, 100000}) lengths.push_back(n);
    std::vector<std::uint8_t> out;
    for (const std::size_t n : lengths) {
        const auto z = qrss_testgen::splitmix(1000 + n, n);
        std::vector<double> v(n);
        std::vector<float> v32(n), w32(n);
        std::vector<std::complex<double>> c(n);
        std::vector<std::complex<float>> c64(n);
        for (std::size_t i = 0; i < n; ++i) {
            v[i] = qrss_testgen::unit(z[i]) * 2000.0 - 1000.0;
            v32[i] = static_cast<float>(v[i]);
            w32[i] = static_cast<float>(std::ldexp(v[i], static_cast<int>(z[i] & 63u) - 32));
        }
        for (std::size_t i = 0; i < n; ++i) {
            c[i] = {v[i], v[n - 1 - i]};
            c64[i] = {static_cast<float>(c[i].real()), static_cast<float>(c[i].imag())};
        }
        append(out, npsum::sum(v));
        append(out, npsum::mean(v));
        append(out, npsum::sum(v32));
        append(out, npsum::mean(v32));
        append(out, npsum::sum_as<double>(std::span<const float>(w32)));
        append(out, npsum::mean_as_double(std::span<const float>(w32)));
        append(out, npsum::sum(c));
        append(out, npsum::sum(c64));
    }
    check::equal(util::to_hex(util::sha256(std::span<const std::uint8_t>(out))),
                 std::string("7aecbbe9866d497ac1634396decc49a5e563fffdd3a8895bcf04d72a67406a3d"),
                 "np.sum / np.mean digest");
}

static void test_structure() {
    // Fewer than 8 values: left to right from the identity.
    const std::vector<double> a = {1e16, 1.0, 1.0, -1e16};
    check::equal(npsum::sum(a), ((1e16 + 1.0) + 1.0) + -1e16, "n < 8 is sequential");
    // 8..128: eight accumulators, combined pairwise, then the tail.
    std::vector<double> b(13);
    for (std::size_t i = 0; i < b.size(); ++i) b[i] = std::ldexp(1.0 + static_cast<double>(i), static_cast<int>(i % 5) * 20);
    double r[8];
    for (int k = 0; k < 8; ++k) r[k] = b[static_cast<std::size_t>(k)];
    double want = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
    for (std::size_t i = 8; i < b.size(); ++i) want += b[i];
    check::equal(npsum::sum(b), 0.0 + want, "8 accumulators then the tail");
    // Empty: the identity; the mean is NaN as numpy's is.
    check::equal(npsum::sum(std::vector<double>{}), 0.0, "empty sum");
    check::is_true(std::isnan(npsum::mean(std::vector<double>{})), "empty mean is NaN");
    // -0.0 sums to +0.0, as np.sum([-0.0]) does (the identity is +0).
    check::is_true(!std::signbit(npsum::sum(std::vector<double>{-0.0, -0.0})), "-0 sums to +0");
    // The buffered cast restarts the pairwise tree every 8192 values:
    //   f = np.ones(8193, np.float32); f[0] = 2.0**60; f[8192] = -2.0**60
    //   np.sum(f, dtype=np.float64) -> 7936.0 (one tree over all 8193: 7808.0)
    std::vector<float> f(8193, 1.0f);
    f[0] = 0x1p60f;
    f[8192] = -0x1p60f;
    check::equal(npsum::sum_as<double>(std::span<const float>(f)), 7936.0, "buffered chunks of 8192");
    std::vector<double> fd(f.begin(), f.end());
    check::equal(npsum::sum(fd), 7808.0, "unbuffered: one tree");
    const std::vector<std::uint8_t> bits = {1, 0, 1, 1};
    check::equal(npsum::mean_as_double(std::span<const std::uint8_t>(bits)), 0.75, "mean of a mask");
}

int main() {
    try {
        test_digest();
        test_structure();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FATAL: %s\n", e.what());
        return 1;
    }
    return check::report("qrss_npsum");
}

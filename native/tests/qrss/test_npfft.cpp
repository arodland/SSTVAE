// native/core/math/npfft against numpy 2.5.3's numpy.fft (pocketfft C++):
// fft, ifft, zero-padded fft, fft of real input, rfft, irfft, rfftfreq and
// fftfreq over powers of two, primes (Bluestein) and mixed radices,
// through one pinned digest (bitwise); fft_rows and fft_inplace against
// fft; the refusals with numpy's texts. tests/test_qrss_native_npfft.py
// compares against numpy directly. The digest:
//
//   out = io.BytesIO()
//   for n in [1, 2, 3, 5, 8, 16, 17, 97, 100, 160, 1009, 2000, 4801, 16000]:
//       z = splitmix(800 + n, 2 * n)                          # infra_testgen.hpp
//       a = (unit(z[:n]) - 0.5) + 1j * (unit(z[n:]) - 0.5)
//       r = unit(z[:n]) - 0.5
//       out.write(np.fft.fft(a).tobytes())
//       out.write(np.fft.ifft(a).tobytes())
//       out.write(np.fft.fft(a, 2 * n + 1).tobytes())
//       out.write(np.fft.fft(r).tobytes())
//       out.write(np.fft.rfft(r).tobytes())
//       out.write(np.fft.irfft(a[: n // 2 + 1], n).tobytes())
//       out.write(np.fft.rfftfreq(n, 0.25).tobytes())
//       out.write(np.fft.fftfreq(n, 1 / 4000).tobytes())
//   sha256(out.getvalue())
//
// and in single precision (numpy 2.5 computes the forward transforms of
// complex64/float32 in double and rounds; the inverses in float):
//
//   out = io.BytesIO()
//   for n in [1, 2, 3, 4, 5, 16, 17, 100, 1009, 2000]:
//       z = splitmix(1800 + n, 2 * n)
//       a = ((unit(z[:n]) - 0.5) + 1j * (unit(z[n:]) - 0.5)).astype(np.complex64)
//       r = (unit(z[:n]) - 0.5).astype(np.float32)
//       for v in (np.fft.fft(a), np.fft.ifft(a), np.fft.fft(a, 2 * n + 1), np.fft.ifft(a, 2 * n + 1),
//                 np.fft.fft(r), np.fft.rfft(r), np.fft.irfft(a[: n // 2 + 1], n)):
//           out.write(v.tobytes())
//   sha256(out.getvalue())

#include <complex>
#include <cstdint>
#include <cstdio>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "infra_testgen.hpp"
#include "math/npfft.hpp"
#include "qrss_check.hpp"
#include "util/sha256.hpp"

using namespace sstvae;
using cd = std::complex<double>;

template <typename T>
static void append_all(std::vector<std::uint8_t>& out, const std::vector<T>& v) {
    const auto* p = reinterpret_cast<const std::uint8_t*>(v.data());
    out.insert(out.end(), p, p + v.size() * sizeof(T));
}

template <typename F>
static void refuses(F f, const std::string& text, const std::string& what) {
    try {
        f();
    } catch (const std::invalid_argument& e) {
        check::equal(std::string(e.what()), text, what);
        return;
    }
    check::fail(what, "did not throw");
}

static void test_digest() {
    std::vector<std::uint8_t> out;
    for (const std::size_t n : {1, 2, 3, 5, 8, 16, 17, 97, 100, 160, 1009, 2000, 4801, 16000}) {
        const auto z = qrss_testgen::splitmix(800 + n, 2 * n);
        std::vector<cd> a(n);
        std::vector<double> r(n);
        for (std::size_t i = 0; i < n; ++i) {
            r[i] = qrss_testgen::unit(z[i]) - 0.5;
            a[i] = cd(r[i], qrss_testgen::unit(z[n + i]) - 0.5);
        }
        const auto N = static_cast<std::int64_t>(n);
        append_all(out, math::fft(std::span<const cd>(a)));
        append_all(out, math::ifft(std::span<const cd>(a)));
        append_all(out, math::fft(std::span<const cd>(a), 2 * N + 1));
        append_all(out, math::fft(std::span<const double>(r)));
        append_all(out, math::rfft(std::span<const double>(r)));
        append_all(out, math::irfft(std::span<const cd>(a.data(), n / 2 + 1), N));
        append_all(out, math::rfftfreq(N, 0.25));
        append_all(out, math::fftfreq(N, 1.0 / 4000));
    }
    qrss_check::libm_equal(util::to_hex(util::sha256(std::span<const std::uint8_t>(out))),
                 std::string("87d1b00584a3ebafb34ad60194620fe2b89b0c9c9d9fe521be3774de58f304d9"),
                 "fft / ifft / rfft / irfft / fftfreq digest");
}

static void test_single_digest() {
    using cf = std::complex<float>;
    std::vector<std::uint8_t> out;
    for (const std::size_t n : {1, 2, 3, 4, 5, 16, 17, 100, 1009, 2000}) {
        const auto z = qrss_testgen::splitmix(1800 + n, 2 * n);
        std::vector<cf> a(n);
        std::vector<float> r(n);
        for (std::size_t i = 0; i < n; ++i) {
            const double re = qrss_testgen::unit(z[i]) - 0.5;
            a[i] = cf(static_cast<float>(re), static_cast<float>(qrss_testgen::unit(z[n + i]) - 0.5));
            r[i] = static_cast<float>(re);
        }
        const auto N = static_cast<std::int64_t>(n);
        append_all(out, math::fft(std::span<const cf>(a)));
        append_all(out, math::ifft(std::span<const cf>(a)));
        append_all(out, math::fft(std::span<const cf>(a), 2 * N + 1));
        append_all(out, math::ifft(std::span<const cf>(a), 2 * N + 1));
        append_all(out, math::fft(std::span<const float>(r)));
        append_all(out, math::rfft(std::span<const float>(r)));
        append_all(out, math::irfft(std::span<const cf>(a.data(), n / 2 + 1), N));
    }
    qrss_check::libm_equal(util::to_hex(util::sha256(std::span<const std::uint8_t>(out))),
                 std::string("32b785ca02b4fc607b96221752c72e4a70b13a680876c17654817d6dd9ed4054"),
                 "single-precision fft / ifft / rfft / irfft digest");
}

static void test_rows_and_inplace() {
    const std::size_t rows = 3, cols = 50;
    const auto z = qrss_testgen::splitmix(850, 2 * rows * cols);
    std::vector<cd> a(rows * cols);
    for (std::size_t i = 0; i < a.size(); ++i)
        a[i] = cd(qrss_testgen::unit(z[i]) - 0.5, qrss_testgen::unit(z[a.size() + i]) - 0.5);
    for (const std::int64_t n : {50, 40, 64}) {
        const auto got = math::fft_rows(a, rows, cols, n);
        bool same = got.size() == rows * static_cast<std::size_t>(n);
        for (std::size_t r = 0; same && r < rows; ++r) {
            const auto one = math::fft(std::span<const cd>(a.data() + r * cols, cols), n);
            for (std::size_t k = 0; k < one.size(); ++k) same = same && got[r * static_cast<std::size_t>(n) + k] == one[k];
        }
        check::is_true(same, "fft_rows is fft per row, n=" + std::to_string(n));
    }
    std::vector<cd> buf(a.begin(), a.begin() + static_cast<std::ptrdiff_t>(cols));
    math::fft_inplace(buf);
    check::is_true(buf == math::fft(std::span<const cd>(a.data(), cols)), "fft_inplace is fft");
    // irfft past the input reads zeros; a short rfft round trip is exact enough
    const std::vector<double> r = {1.0, -2.0, 0.5, 3.0};
    const auto back = math::irfft(math::rfft(std::span<const double>(r)), 4);
    check::close(back, r, 1e-15, "irfft(rfft(r))");
    check::equal(math::irfft(std::vector<cd>{{1.0, 0.0}}, 3) == std::vector<double>(3, 1.0 / 3.0), true,
                 "irfft of a DC term zero-extends");
}

static void test_refusals() {
    const std::vector<cd> a = {{1, 0}, {2, 0}};
    refuses([&] { (void)math::fft(std::span<const cd>(a), 0); }, "Invalid number of FFT data points (0) specified.",
            "fft n = 0");
    refuses([&] { (void)math::ifft(std::span<const cd>(a), -3); },
            "Invalid number of FFT data points (-3) specified.", "ifft n = -3");
    refuses([] { (void)math::fft(std::span<const cd>()); }, "Invalid number of FFT data points (0) specified.",
            "fft of nothing");
    refuses([] { (void)math::irfft(std::vector<cd>{{1, 0}}); }, "Invalid number of FFT data points (0) specified.",
            "irfft of one point, default n");
    refuses([] { (void)math::irfft(std::span<const cd>()); }, "Invalid number of FFT data points (-2) specified.",
            "irfft of nothing, default n");
    // numpy reads past the end of an empty input here; the port refuses
    refuses([] { (void)math::irfft(std::span<const cd>(), 4); }, "irfft: empty input", "irfft of nothing, n = 4");
}

int main() {
    try {
        test_digest();
        test_single_digest();
        test_rows_and_inplace();
        test_refusals();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FATAL: %s\n", e.what());
        return 1;
    }
    return check::report("qrss_npfft");
}

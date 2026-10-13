#include "math/convolve.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "dsp/fft.hpp"   // pocketfft_hdronly (its one include site)
#include "math/npcomplex.hpp"

namespace sstvae::math {

namespace {

using cd = std::complex<double>;

template <typename T>
inline constexpr bool is_complex_v = !std::is_floating_point_v<T>;

// --- transforms (pocketfft, one thread: the receiver runs on workers) --------

// rows x n, row-major: forward or inverse complex transforms of each row,
// the inverse scaled by fct.
void c2c_rows(const std::vector<cd>& in, std::vector<cd>& out, std::size_t rows, std::size_t n, bool forward,
              double fct) {
    out.resize(rows * n);
    if (rows == 0 || n == 0) return;
    const pocketfft::shape_t shape{rows, n};
    const pocketfft::stride_t stride{static_cast<std::ptrdiff_t>(n * sizeof(cd)),
                                     static_cast<std::ptrdiff_t>(sizeof(cd))};
    pocketfft::c2c(shape, stride, stride, pocketfft::shape_t{1}, forward, in.data(), out.data(), fct, 1);
}

// rows x n real -> rows x (n/2 + 1) complex, forward.
void r2c_rows(const std::vector<double>& in, std::vector<cd>& out, std::size_t rows, std::size_t n) {
    const std::size_t nc = n / 2 + 1;
    out.resize(rows * nc);
    if (rows == 0 || n == 0) return;
    const pocketfft::shape_t shape{rows, n};
    const pocketfft::stride_t sr{static_cast<std::ptrdiff_t>(n * sizeof(double)),
                                 static_cast<std::ptrdiff_t>(sizeof(double))};
    const pocketfft::stride_t sc{static_cast<std::ptrdiff_t>(nc * sizeof(cd)),
                                 static_cast<std::ptrdiff_t>(sizeof(cd))};
    pocketfft::r2c(shape, sr, sc, pocketfft::shape_t{1}, true, in.data(), out.data(), 1.0, 1);
}

// rows x (n/2 + 1) complex -> rows x n real, inverse, scaled by fct.
void c2r_rows(const std::vector<cd>& in, std::vector<double>& out, std::size_t rows, std::size_t n, double fct) {
    const std::size_t nc = n / 2 + 1;
    out.resize(rows * n);
    if (rows == 0 || n == 0) return;
    const pocketfft::shape_t shape{rows, n};
    const pocketfft::stride_t sc{static_cast<std::ptrdiff_t>(nc * sizeof(cd)),
                                 static_cast<std::ptrdiff_t>(sizeof(cd))};
    const pocketfft::stride_t sr{static_cast<std::ptrdiff_t>(n * sizeof(double)),
                                 static_cast<std::ptrdiff_t>(sizeof(double))};
    pocketfft::c2r(shape, sc, sr, pocketfft::shape_t{1}, false, in.data(), out.data(), fct, 1);
}

// The rows (each `step` samples of x, zero-padded to n; x itself padded
// with zeros to rows*step) as a rows x n buffer.
template <typename T>
std::vector<T> padded_rows(std::span<const T> x, std::size_t rows, std::size_t step, std::size_t n) {
    std::vector<T> buf(rows * n, T{});
    for (std::size_t r = 0; r < rows; ++r) {
        const std::size_t a = r * step;
        if (a >= x.size()) break;
        const std::size_t len = std::min(step, x.size() - a);
        std::copy(x.begin() + static_cast<std::ptrdiff_t>(a), x.begin() + static_cast<std::ptrdiff_t>(a + len),
                  buf.begin() + static_cast<std::ptrdiff_t>(r * n));
    }
    return buf;
}

// Full complex spectra of the rows, as scipy.fft.fftn sees them: a complex
// row by a complex transform, a real one as ducc's c2c does a real input
// (the real transform, then the upper half as conjugates).
template <typename T>
std::vector<cd> complex_spectra(std::span<const T> x, std::size_t rows, std::size_t step, std::size_t n) {
    std::vector<cd> out;
    if constexpr (is_complex_v<T>) {
        c2c_rows(padded_rows<cd>(x, rows, step, n), out, rows, n, true, 1.0);
    } else {
        std::vector<cd> half;
        r2c_rows(padded_rows<double>(x, rows, step, n), half, rows, n);
        const std::size_t nc = n / 2 + 1;
        out.assign(rows * n, cd{});
        for (std::size_t r = 0; r < rows; ++r) {
            cd* o = out.data() + r * n;
            const cd* h = half.data() + r * nc;
            for (std::size_t j = 0; j < nc && j < n; ++j) o[j] = h[j];
            for (std::size_t j = nc; j < n; ++j) o[j] = std::conj(h[n - j]);
        }
    }
    return out;
}

// The convolution of the rows of in1 and in2 (rows1 and rows2 rows, one of
// them 1, broadcast), each zero-padded to n, back in the time domain:
// max(rows1, rows2) x n. Real when both inputs are.
template <typename R, typename T1, typename T2>
std::vector<R> freq_domain_conv(std::span<const T1> in1, std::size_t rows1, std::size_t step1,
                                std::span<const T2> in2, std::size_t rows2, std::size_t step2, std::size_t n) {
    const std::size_t rows = std::max(rows1, rows2);
    const double fct = 1.0 / static_cast<double>(n);
    if constexpr (is_complex_v<R>) {
        const std::vector<cd> sp1 = complex_spectra<T1>(in1, rows1, step1, n);
        const std::vector<cd> sp2 = complex_spectra<T2>(in2, rows2, step2, n);
        std::vector<cd> prod(rows * n);
        for (std::size_t r = 0; r < rows; ++r) {
            const cd* a = sp1.data() + (rows1 == 1 ? 0 : r) * n;
            const cd* b = sp2.data() + (rows2 == 1 ? 0 : r) * n;
            cd* p = prod.data() + r * n;
            for (std::size_t j = 0; j < n; ++j) p[j] = np::mul(a[j], b[j]);
        }
        std::vector<cd> out;
        c2c_rows(prod, out, rows, n, false, fct);
        return out;
    } else {
        const std::size_t nc = n / 2 + 1;
        std::vector<cd> sp1, sp2;
        r2c_rows(padded_rows<double>(in1, rows1, step1, n), sp1, rows1, n);
        r2c_rows(padded_rows<double>(in2, rows2, step2, n), sp2, rows2, n);
        std::vector<cd> prod(rows * nc);
        for (std::size_t r = 0; r < rows; ++r) {
            const cd* a = sp1.data() + (rows1 == 1 ? 0 : r) * nc;
            const cd* b = sp2.data() + (rows2 == 1 ? 0 : r) * nc;
            cd* p = prod.data() + r * nc;
            for (std::size_t j = 0; j < nc; ++j) p[j] = np::mul(a[j], b[j]);
        }
        std::vector<double> out;
        c2r_rows(prod, out, rows, n, fct);
        return out;
    }
}

// in1 * in2 with numpy's promotion and broadcasting (one of them length 1).
template <typename R, typename A, typename B>
R times(const A& a, const B& b) {
    if constexpr (is_complex_v<R>) {
        return np::mul(cd(a), cd(b));
    } else {
        return a * b;
    }
}

template <typename R, typename T1, typename T2>
std::vector<R> broadcast_product(std::span<const T1> in1, std::span<const T2> in2) {
    const std::size_t n = std::max(in1.size(), in2.size());
    std::vector<R> out(n);
    for (std::size_t i = 0; i < n; ++i)
        out[i] = times<R>(in1[in1.size() == 1 ? 0 : i], in2[in2.size() == 1 ? 0 : i]);
    return out;
}

// _apply_conv_mode on a 'full' result of length s1 + s2 - 1 (or, from the
// broadcast product, max(s1, s2)).
template <typename R>
std::vector<R> apply_mode(std::vector<R> ret, std::size_t s1, ConvMode mode) {
    if (mode == ConvMode::Full) return ret;
    const std::size_t start = (ret.size() - s1) / 2;
    return std::vector<R>(ret.begin() + static_cast<std::ptrdiff_t>(start),
                          ret.begin() + static_cast<std::ptrdiff_t>(start + s1));
}

template <typename R, typename T1, typename T2>
std::vector<R> fftconvolve_impl(std::span<const T1> in1, std::span<const T2> in2, ConvMode mode) {
    if (in1.empty() || in2.empty()) return {};
    const std::size_t s1 = in1.size(), s2 = in2.size();
    if (s1 == 1 || s2 == 1) return apply_mode(broadcast_product<R>(in1, in2), s1, mode);
    const std::size_t shape = s1 + s2 - 1;
    const auto fshape = static_cast<std::size_t>(next_fast_len(static_cast<std::int64_t>(shape), !is_complex_v<R>));
    std::vector<R> ret = freq_domain_conv<R>(in1, 1, s1, in2, 1, s2, fshape);
    ret.resize(shape);
    return apply_mode(std::move(ret), s1, mode);
}

template <typename R, typename T1, typename T2>
std::vector<R> oaconvolve_impl(std::span<const T1> in1, std::span<const T2> in2, ConvMode mode) {
    if (in1.empty() || in2.empty()) return {};
    const std::int64_t s1 = static_cast<std::int64_t>(in1.size()), s2 = static_cast<std::int64_t>(in2.size());
    if (s1 == s2) return fftconvolve_impl<R>(in1, in2, mode);
    if (s1 == 1 || s2 == 1) return apply_mode(broadcast_product<R>(in1, in2), in1.size(), mode);
    const std::int64_t shape_final = s1 + s2 - 1;
    const OaLens lens = calc_oa_lens(s1, s2);
    if (lens.in1_step == s1 && lens.in2_step == s2) return fftconvolve_impl<R>(in1, in2, mode);
    const std::int64_t block = lens.block_size, overlap = lens.overlap;
    auto steps = [&](std::int64_t s, std::int64_t step) -> std::int64_t {
        if (s <= step) return 1;
        auto n = static_cast<std::int64_t>(std::ceil(static_cast<double>(s + 1) / static_cast<double>(step)));
        if ((block - overlap) * n < shape_final) n += 1;
        return n;
    };
    const std::int64_t nsteps1 = steps(s1, lens.in1_step), nsteps2 = steps(s2, lens.in2_step);
    const auto nb = static_cast<std::size_t>(block);
    std::vector<R> ret = freq_domain_conv<R>(in1, static_cast<std::size_t>(nsteps1),
                                             static_cast<std::size_t>(lens.in1_step), in2,
                                             static_cast<std::size_t>(nsteps2),
                                             static_cast<std::size_t>(lens.in2_step), nb);
    // overlap-add: each row's last `overlap` samples onto the next row's first
    const std::size_t rows = static_cast<std::size_t>(std::max(nsteps1, nsteps2));
    const auto ov = static_cast<std::size_t>(overlap);
    const std::size_t keep = nb - ov;
    for (std::size_t r = 1; r < rows; ++r) {
        R* dst = ret.data() + r * nb;
        const R* src = ret.data() + (r - 1) * nb + keep;
        for (std::size_t j = 0; j < ov && j < keep; ++j) dst[j] = dst[j] + src[j];
    }
    std::vector<R> flat;
    flat.reserve(rows * keep);
    for (std::size_t r = 0; r < rows; ++r)
        flat.insert(flat.end(), ret.begin() + static_cast<std::ptrdiff_t>(r * nb),
                    ret.begin() + static_cast<std::ptrdiff_t>(r * nb + keep));
    flat.resize(static_cast<std::size_t>(std::min<std::int64_t>(shape_final, static_cast<std::int64_t>(flat.size()))));
    return apply_mode(std::move(flat), in1.size(), mode);
}

bool smooth(std::int64_t n, bool real) {
    for (const std::int64_t p : {2, 3, 5})
        while (n % p == 0) n /= p;
    if (!real)
        for (const std::int64_t p : {7, 11})
            while (n % p == 0) n /= p;
    return n == 1;
}

}  // namespace

std::int64_t next_fast_len(std::int64_t target, bool real) {
    if (target < 1) throw std::invalid_argument("Target cannot be negative or zero");
    std::int64_t n = target;
    while (!smooth(n, real)) ++n;
    return n;
}

double lambertw_m1(double x) {
    // scipy.special.lambertw (xsf's lambert_w.h) for real x on the k = -1
    // branch: the initial guess log(-x) for |x| <= 1/e, then Halley's
    // method at tol 1e-8 (the third-order steps leave it at full precision).
    constexpr double EXPN1 = 0.36787944117144232159553;   // exp(-1)
    if (std::isnan(x) || !(x < 0.0) || -x > EXPN1) return std::numeric_limits<double>::quiet_NaN();
    double w = std::log(-x);
    for (int i = 0; i < 100; ++i) {
        const double ew = std::exp(w);
        const double wew = w * ew;
        const double wewz = wew - x;
        const double wn = w - wewz / (wew + ew - (w + 2.0) * wewz / (2.0 * w + 2.0));
        if (std::fabs(wn - w) <= 1e-8 * std::fabs(wn)) return wn;
        w = wn;
    }
    return std::numeric_limits<double>::quiet_NaN();
}

OaLens calc_oa_lens(std::int64_t s1, std::int64_t s2) {
    const OaLens fallback{s1 + s2 - 1, -1, s1, s2};
    if (s1 == s2 || s1 == 1 || s2 == 1) return fallback;
    bool swapped = false;
    if (s2 > s1) {
        std::swap(s1, s2);
        swapped = true;
    }
    if (static_cast<double>(s2) >= static_cast<double>(s1) / 2.0) return fallback;
    const std::int64_t overlap = s2 - 1;
    const double opt_size = -static_cast<double>(overlap) * lambertw_m1(-1.0 / (2.0 * std::numbers::e * static_cast<double>(overlap)));
    const std::int64_t block = next_fast_len(static_cast<std::int64_t>(std::ceil(opt_size)));
    if (block >= s1) return fallback;
    if (!swapped) return {block, overlap, block - s2 + 1, s2};
    return {block, overlap, s2, block - s2 + 1};
}

std::vector<cd> oaconvolve(std::span<const cd> in1, std::span<const double> in2, ConvMode mode) {
    return oaconvolve_impl<cd>(in1, in2, mode);
}
std::vector<cd> oaconvolve(std::span<const cd> in1, std::span<const cd> in2, ConvMode mode) {
    return oaconvolve_impl<cd>(in1, in2, mode);
}
std::vector<cd> oaconvolve(std::span<const double> in1, std::span<const cd> in2, ConvMode mode) {
    return oaconvolve_impl<cd>(in1, in2, mode);
}
std::vector<double> oaconvolve(std::span<const double> in1, std::span<const double> in2, ConvMode mode) {
    return oaconvolve_impl<double>(in1, in2, mode);
}

std::vector<cd> fftconvolve(std::span<const cd> in1, std::span<const double> in2, ConvMode mode) {
    return fftconvolve_impl<cd>(in1, in2, mode);
}
std::vector<cd> fftconvolve(std::span<const cd> in1, std::span<const cd> in2, ConvMode mode) {
    return fftconvolve_impl<cd>(in1, in2, mode);
}
std::vector<cd> fftconvolve(std::span<const double> in1, std::span<const cd> in2, ConvMode mode) {
    return fftconvolve_impl<cd>(in1, in2, mode);
}
std::vector<double> fftconvolve(std::span<const double> in1, std::span<const double> in2, ConvMode mode) {
    return fftconvolve_impl<double>(in1, in2, mode);
}

std::vector<cd> freqz(std::span<const double> b, std::span<const double> f, double fs) {
    constexpr double two_pi = 2.0 * 3.141592653589793;   // 2 * math.pi, exactly as Python forms it
    std::vector<cd> out(f.size());
    const std::size_t nb = b.size();
    for (std::size_t i = 0; i < f.size(); ++i) {
        const double w = two_pi * f[i] / fs;
        // exp(-1j * w): -1j * (w + 0j) is (+0.0, -w) for a finite w, and
        // glibc's cexp of that is (cos(-w), sin(-w)) exactly (exp(0) = 1)
        const cd z = np::mul(cd(-0.0, -1.0), cd(w, 0.0));
        const cd zm1 = std::isfinite(w) ? cd(std::cos(z.imag()), std::sin(z.imag())) : std::exp(z);
        cd h = nb ? cd(b[nb - 1], 0.0) : cd(0.0, 0.0);
        for (std::size_t k = 2; k <= nb; ++k) {
            const cd p = np::mul(h, zm1);
            h = cd(b[nb - k] + p.real(), 0.0 + p.imag());
        }
        out[i] = np::div(h, cd(1.0, 0.0));
    }
    return out;
}

std::vector<std::uint8_t> maximum_filter1d(std::span<const std::uint8_t> x, std::int64_t size) {
    if (size < 1) throw std::runtime_error("incorrect filter size");
    const auto n = static_cast<std::int64_t>(x.size());
    std::vector<std::uint8_t> out(x.size());
    if (n == 0) return out;
    const std::int64_t left = size / 2;
    // 'reflect': period 2n, the second half mirrored
    auto at = [&](std::int64_t p) -> std::uint8_t {
        std::int64_t m = p % (2 * n);
        if (m < 0) m += 2 * n;
        return x[static_cast<std::size_t>(m < n ? m : 2 * n - 1 - m)];
    };
    // a monotonic deque over the extended line, positions -left .. n-1+right
    std::deque<std::pair<std::int64_t, std::uint8_t>> dq;
    std::int64_t next = -left;
    for (std::int64_t i = 0; i < n; ++i) {
        const std::int64_t lo = i - left, hi = lo + size - 1;
        for (; next <= hi; ++next) {
            const std::uint8_t v = at(next);
            while (!dq.empty() && dq.back().second <= v) dq.pop_back();
            dq.emplace_back(next, v);
        }
        while (dq.front().first < lo) dq.pop_front();
        out[static_cast<std::size_t>(i)] = dq.front().second;
    }
    return out;
}

}  // namespace sstvae::math

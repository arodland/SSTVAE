// numpy.fft (fft, ifft, rfft, irfft, fftfreq, rfftfreq): see npfft.hpp.
// The loops follow numpy 2.x's numpy/fft/_pocketfft_umath.cpp (fft_loop,
// rfft_impl, irfft_loop): copy the input into the output buffer, cropped
// or zero-padded, then one plan->exec(buffer, fct, direction) -- in the
// precision numpy's ufunc type resolution picks (see the single-precision
// overloads below).

#include "math/npfft.hpp"

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <string>

#include "dsp/fft.hpp"

namespace sstvae::math {
namespace {

namespace pd = pocketfft::detail;

std::size_t points(std::optional<std::int64_t> n, std::size_t fallback) {
    const std::int64_t v = n ? *n : static_cast<std::int64_t>(fallback);
    if (v < 1) throw std::invalid_argument("Invalid number of FFT data points (" + std::to_string(v) + ") specified.");
    return static_cast<std::size_t>(v);
}

template <typename T>
void c2c_inplace(std::complex<T>* data, std::size_t n, T fct, bool forward) {
    auto plan = pd::get_plan<pd::pocketfft_c<T>>(n);
    plan->exec(reinterpret_cast<pd::cmplx<T>*>(data), fct, forward);
}

// numpy's fct: 1 forward; for the inverse (norm=None, swapped to
// "forward") reciprocal(n, dtype=real_dtype) -- 1/n rounded to T.
template <typename T>
T inverse_fct(std::size_t n) {
    return static_cast<T>(1) / static_cast<T>(n);
}

template <typename T>
std::vector<std::complex<T>> c2c(std::span<const std::complex<T>> x, std::optional<std::int64_t> n, bool forward) {
    const std::size_t npts = points(n, x.size());
    std::vector<std::complex<T>> out(npts, std::complex<T>(0, 0));
    std::copy_n(x.begin(), std::min(npts, x.size()), out.begin());
    c2c_inplace<T>(out.data(), npts, forward ? T(1) : inverse_fct<T>(npts), forward);
    return out;
}

template <typename T>
std::vector<std::complex<T>> c2c_real(std::span<const T> x, std::optional<std::int64_t> n) {
    std::vector<std::complex<T>> z(x.size());
    for (std::size_t k = 0; k < x.size(); ++k) z[k] = std::complex<T>(x[k], 0);
    return c2c<T>(z, n, true);
}

template <typename T>
std::vector<std::complex<T>> rfft_impl(std::span<const T> x, std::optional<std::int64_t> n) {
    const std::size_t npts = points(n, x.size());
    const std::size_t nout = npts / 2 + 1;
    // rfft_impl: the real data at offset 1 of a buffer of 2*nout values
    // (zeros past the input), exec in place, then R0 moved down, I0 = 0.
    std::vector<std::complex<T>> out(nout, std::complex<T>(0, 0));
    T* buf = reinterpret_cast<T*>(out.data());
    std::copy_n(x.begin(), std::min(npts, x.size()), buf + 1);
    auto plan = pd::get_plan<pd::pocketfft_r<T>>(npts);
    plan->exec(buf + 1, T(1), true);
    out[0] = std::complex<T>(out[0].imag(), 0);
    return out;
}

template <typename T>
std::vector<T> irfft_impl(std::span<const std::complex<T>> x, std::optional<std::int64_t> n) {
    const std::size_t nin = x.size();
    const std::int64_t dflt = (static_cast<std::int64_t>(nin) - 1) * 2;
    if (!n && dflt < 1)
        throw std::invalid_argument("Invalid number of FFT data points (" + std::to_string(dflt) + ") specified.");
    const std::size_t nout = points(n, static_cast<std::size_t>(dflt));
    // numpy reads x[0] of an empty input regardless (past the end); refuse.
    if (nin == 0) throw std::invalid_argument("irfft: empty input");
    // irfft_loop: R0, then (R1, I1) ... up to (nout - 1) / 2 pairs (zeros
    // past the input), then R(nout/2) for even nout.
    std::vector<T> out(nout, T(0));
    out[0] = x[0].real();
    if (nout > 1) {
        const std::size_t pairs = (nout - 1) / 2;
        const std::size_t have = std::min(pairs, nin - 1);
        for (std::size_t k = 0; k < have; ++k) {
            out[1 + 2 * k] = x[1 + k].real();
            out[2 + 2 * k] = x[1 + k].imag();
        }
        if (nout % 2 == 0) out[nout - 1] = (nout / 2 >= nin) ? T(0) : x[nout / 2].real();
    }
    auto plan = pd::get_plan<pd::pocketfft_r<T>>(nout);
    plan->exec(out.data(), inverse_fct<T>(nout), false);
    return out;
}

template <typename T>
std::vector<std::complex<T>> fft_rows_impl(std::span<const std::complex<T>> a, std::size_t rows, std::size_t cols,
                                           std::optional<std::int64_t> n) {
    if (a.size() != rows * cols) throw std::invalid_argument("fft_rows: a holds " + std::to_string(a.size()) +
                                                             " values, not rows*cols = " + std::to_string(rows * cols));
    const std::size_t npts = points(n, cols);
    std::vector<std::complex<T>> out(rows * npts, std::complex<T>(0, 0));
    const std::size_t used = std::min(npts, cols);
    for (std::size_t r = 0; r < rows; ++r) {
        std::copy_n(a.begin() + static_cast<std::ptrdiff_t>(r * cols), used,
                    out.begin() + static_cast<std::ptrdiff_t>(r * npts));
        c2c_inplace<T>(out.data() + r * npts, npts, T(1), true);
    }
    return out;
}

}  // namespace

std::vector<std::complex<double>> fft(std::span<const std::complex<double>> x, std::optional<std::int64_t> n) {
    return c2c<double>(x, n, true);
}
std::vector<std::complex<double>> fft(std::span<const double> x, std::optional<std::int64_t> n) {
    return c2c_real<double>(x, n);
}
std::vector<std::complex<double>> ifft(std::span<const std::complex<double>> x, std::optional<std::int64_t> n) {
    return c2c<double>(x, n, false);
}
std::vector<std::complex<double>> rfft(std::span<const double> x, std::optional<std::int64_t> n) {
    return rfft_impl<double>(x, n);
}
std::vector<double> irfft(std::span<const std::complex<double>> x, std::optional<std::int64_t> n) {
    return irfft_impl<double>(x, n);
}

// Single precision as numpy 2.5 actually runs it. _raw_fft passes fct as
// the Python int 1 to the forward ufuncs, which the legacy type resolver
// reads as int64 -- castable safely only to the double loop ('Dd->D',
// 'dd->D') -- so a forward transform of complex64/float32 is computed in
// double and rounded to complex64 once. The inverses get fct =
// np.reciprocal(n, dtype=float32), a float32, which selects the float loop
// ('Ff->F', 'Ff->f'): those really run pocketfft's float plans.
template <typename T>
std::vector<std::complex<float>> narrow(const std::vector<std::complex<T>>& v) {
    std::vector<std::complex<float>> out(v.size());
    for (std::size_t k = 0; k < v.size(); ++k)
        out[k] = std::complex<float>(static_cast<float>(v[k].real()), static_cast<float>(v[k].imag()));
    return out;
}

std::vector<std::complex<float>> fft(std::span<const std::complex<float>> x, std::optional<std::int64_t> n) {
    const std::vector<std::complex<double>> wide(x.begin(), x.end());
    return narrow(c2c<double>(wide, n, true));
}
std::vector<std::complex<float>> fft(std::span<const float> x, std::optional<std::int64_t> n) {
    const std::vector<double> wide(x.begin(), x.end());
    return narrow(c2c_real<double>(wide, n));
}
std::vector<std::complex<float>> ifft(std::span<const std::complex<float>> x, std::optional<std::int64_t> n) {
    return c2c<float>(x, n, false);
}
std::vector<std::complex<float>> rfft(std::span<const float> x, std::optional<std::int64_t> n) {
    const std::vector<double> wide(x.begin(), x.end());
    return narrow(rfft_impl<double>(wide, n));
}
std::vector<float> irfft(std::span<const std::complex<float>> x, std::optional<std::int64_t> n) {
    return irfft_impl<float>(x, n);
}

void fft_inplace(std::span<std::complex<double>> data) {
    const std::size_t npts = points(std::nullopt, data.size());
    c2c_inplace<double>(data.data(), npts, 1.0, true);
}

std::vector<std::complex<double>> fft_rows(std::span<const std::complex<double>> a, std::size_t rows,
                                           std::size_t cols, std::optional<std::int64_t> n) {
    return fft_rows_impl<double>(a, rows, cols, n);
}
std::vector<std::complex<float>> fft_rows(std::span<const std::complex<float>> a, std::size_t rows,
                                          std::size_t cols, std::optional<std::int64_t> n) {
    const std::vector<std::complex<double>> wide(a.begin(), a.end());  // forward: the double loop, as fft
    return narrow(fft_rows_impl<double>(wide, rows, cols, n));
}

std::vector<double> fftfreq(std::int64_t n, double d) {
    const double val = 1.0 / (static_cast<double>(n) * d);
    std::vector<double> out(static_cast<std::size_t>(std::max<std::int64_t>(n, 0)));
    const std::int64_t N = (n - 1) / 2 + 1;  // n >= 1: floor division
    for (std::int64_t i = 0; i < n; ++i) {
        const std::int64_t k = i < N ? i : -(n / 2) + (i - N);
        out[static_cast<std::size_t>(i)] = static_cast<double>(k) * val;
    }
    return out;
}

std::vector<double> rfftfreq(std::int64_t n, double d) {
    const double val = 1.0 / (static_cast<double>(n) * d);
    const std::int64_t N = n / 2 + 1;
    std::vector<double> out(static_cast<std::size_t>(std::max<std::int64_t>(N, 0)));
    for (std::int64_t i = 0; i < N; ++i) out[static_cast<std::size_t>(i)] = static_cast<double>(i) * val;
    return out;
}

}  // namespace sstvae::math

// numpy.fft (fft, ifft, rfft, irfft, fftfreq, rfftfreq): see npfft.hpp.
// The loops follow numpy 2.x's numpy/fft/_pocketfft_umath.cpp (fft_loop,
// rfft_impl, irfft_loop): copy the input into the output buffer, cropped
// or zero-padded, then one plan->exec(buffer, fct, direction).

#include "math/npfft.hpp"

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <string>

#include "dsp/fft.hpp"

namespace sstvae::math {
namespace {

using cdouble = std::complex<double>;
namespace pd = pocketfft::detail;

std::size_t points(std::optional<std::int64_t> n, std::size_t fallback) {
    const std::int64_t v = n ? *n : static_cast<std::int64_t>(fallback);
    if (v < 1) throw std::invalid_argument("Invalid number of FFT data points (" + std::to_string(v) + ") specified.");
    return static_cast<std::size_t>(v);
}

void c2c_inplace(cdouble* data, std::size_t n, double fct, bool forward) {
    auto plan = pd::get_plan<pd::pocketfft_c<double>>(n);
    plan->exec(reinterpret_cast<pd::cmplx<double>*>(data), fct, forward);
}

std::vector<cdouble> c2c(std::span<const cdouble> x, std::optional<std::int64_t> n, bool forward) {
    const std::size_t npts = points(n, x.size());
    std::vector<cdouble> out(npts, cdouble(0.0, 0.0));
    std::copy_n(x.begin(), std::min(npts, x.size()), out.begin());
    // numpy's fct: 1 forward; reciprocal(n) for the inverse (norm=None).
    c2c_inplace(out.data(), npts, forward ? 1.0 : 1.0 / static_cast<double>(npts), forward);
    return out;
}

}  // namespace

std::vector<cdouble> fft(std::span<const cdouble> x, std::optional<std::int64_t> n) { return c2c(x, n, true); }

std::vector<cdouble> fft(std::span<const double> x, std::optional<std::int64_t> n) {
    std::vector<cdouble> z(x.size());
    for (std::size_t k = 0; k < x.size(); ++k) z[k] = cdouble(x[k], 0.0);
    return c2c(z, n, true);
}

std::vector<cdouble> ifft(std::span<const cdouble> x, std::optional<std::int64_t> n) { return c2c(x, n, false); }

void fft_inplace(std::span<cdouble> data) {
    const std::size_t npts = points(std::nullopt, data.size());
    c2c_inplace(data.data(), npts, 1.0, true);
}

std::vector<cdouble> rfft(std::span<const double> x, std::optional<std::int64_t> n) {
    const std::size_t npts = points(n, x.size());
    const std::size_t nout = npts / 2 + 1;
    // rfft_impl: the real data at offset 1 of a buffer of 2*nout doubles
    // (zeros past the input), exec in place, then R0 moved down, I0 = 0.
    std::vector<cdouble> out(nout, cdouble(0.0, 0.0));
    double* buf = reinterpret_cast<double*>(out.data());
    std::copy_n(x.begin(), std::min(npts, x.size()), buf + 1);
    auto plan = pd::get_plan<pd::pocketfft_r<double>>(npts);
    plan->exec(buf + 1, 1.0, true);
    out[0] = cdouble(out[0].imag(), 0.0);
    return out;
}

std::vector<double> irfft(std::span<const cdouble> x, std::optional<std::int64_t> n) {
    const std::size_t nin = x.size();
    const std::int64_t dflt = (static_cast<std::int64_t>(nin) - 1) * 2;
    if (!n && dflt < 1)
        throw std::invalid_argument("Invalid number of FFT data points (" + std::to_string(dflt) + ") specified.");
    const std::size_t nout = points(n, static_cast<std::size_t>(dflt));
    // numpy reads x[0] of an empty input regardless (past the end); refuse.
    if (nin == 0) throw std::invalid_argument("irfft: empty input");
    // irfft_loop: R0, then (R1, I1) ... up to (nout - 1) / 2 pairs (zeros
    // past the input), then R(nout/2) for even nout.
    std::vector<double> out(nout, 0.0);
    out[0] = x[0].real();
    if (nout > 1) {
        const std::size_t pairs = (nout - 1) / 2;
        const std::size_t have = std::min(pairs, nin - 1);
        for (std::size_t k = 0; k < have; ++k) {
            out[1 + 2 * k] = x[1 + k].real();
            out[2 + 2 * k] = x[1 + k].imag();
        }
        if (nout % 2 == 0) out[nout - 1] = (nout / 2 >= nin) ? 0.0 : x[nout / 2].real();
    }
    auto plan = pd::get_plan<pd::pocketfft_r<double>>(nout);
    plan->exec(out.data(), 1.0 / static_cast<double>(nout), false);
    return out;
}

std::vector<cdouble> fft_rows(std::span<const cdouble> a, std::size_t rows, std::size_t cols,
                              std::optional<std::int64_t> n) {
    if (a.size() != rows * cols) throw std::invalid_argument("fft_rows: a holds " + std::to_string(a.size()) +
                                                             " values, not rows*cols = " + std::to_string(rows * cols));
    const std::size_t npts = points(n, cols);
    std::vector<cdouble> out(rows * npts, cdouble(0.0, 0.0));
    const std::size_t used = std::min(npts, cols);
    for (std::size_t r = 0; r < rows; ++r) {
        std::copy_n(a.begin() + static_cast<std::ptrdiff_t>(r * cols), used,
                    out.begin() + static_cast<std::ptrdiff_t>(r * npts));
        c2c_inplace(out.data() + r * npts, npts, 1.0, true);
    }
    return out;
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

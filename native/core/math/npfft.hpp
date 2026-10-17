// numpy.fft as the QRSSTVAE receiver calls it: fft, ifft, rfft, irfft
// with numpy's `n` (crop or zero-pad the input to n points; irfft's
// default 2 (m - 1)) and default norm ("backward": the forward transform
// unscaled, the inverse multiplied by fct = 1/n inside the transform),
// plus fftfreq and rfftfreq.
//
// numpy 2.x computes these with pocketfft's C++ header
// (numpy/fft/_pocketfft_umath.cpp): one cached plan per length, exec(c,
// fct, direction), real input to fft promoted to complex (a c2c
// transform), rfft/irfft through the real plan in FFTPACK order. This
// calls the same plan classes of the vendored header
// (native/third_party/pocketfft) the same way, so the transforms are
// numpy's to the bit when the two headers' arithmetic agrees -- which
// tests/test_qrss_native_npfft.py measures (it is not a property of the
// API: a pocketfft revision that changed a twiddle or a pass would move
// the last bits, and the test would say so).
//
// Errors: numpy's ValueError "Invalid number of FFT data points (n)
// specified." for n < 1 (std::invalid_argument). Thread-safe: pocketfft's
// plan cache is mutex-protected.

#ifndef SSTVAE_MATH_NPFFT_HPP
#define SSTVAE_MATH_NPFFT_HPP

#include <complex>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

namespace sstvae::math {

// np.fft.fft(x, n): forward c2c of x cropped or zero-padded to n (default
// len(x)). Real x is promoted to complex first, as numpy's ufunc does.
std::vector<std::complex<double>> fft(std::span<const std::complex<double>> x,
                                      std::optional<std::int64_t> n = std::nullopt);
std::vector<std::complex<double>> fft(std::span<const double> x, std::optional<std::int64_t> n = std::nullopt);

// np.fft.ifft(x, n): backward c2c with fct = 1/n.
std::vector<std::complex<double>> ifft(std::span<const std::complex<double>> x,
                                       std::optional<std::int64_t> n = std::nullopt);

// np.fft.rfft(x, n): n // 2 + 1 bins of the real transform.
std::vector<std::complex<double>> rfft(std::span<const double> x, std::optional<std::int64_t> n = std::nullopt);

// np.fft.irfft(x, n): n real points (default 2 (len(x) - 1)), fct = 1/n.
// The imaginary parts of x[0] (and of x[n/2], n even) are ignored, as in
// numpy.
std::vector<double> irfft(std::span<const std::complex<double>> x, std::optional<std::int64_t> n = std::nullopt);

// The same for complex64/float32 input, returning complex64 (irfft:
// float32), with numpy 2.5's precisions -- which are not uniform. The
// forward transforms (fft, rfft) receive fct as a Python int, which
// numpy's legacy ufunc type resolution can cast safely only to the double
// loop, so they are computed in double and rounded once; the inverses
// (ifft, irfft) receive fct = np.reciprocal(n, dtype=float32), select the
// float loop, and run pocketfft's float plans with that fct. Measured, not
// read from the release notes (which say all four run natively in float).
// A float32 signal multiplied by a float64 window first is complex128, and
// takes the overloads above.
std::vector<std::complex<float>> fft(std::span<const std::complex<float>> x,
                                     std::optional<std::int64_t> n = std::nullopt);
std::vector<std::complex<float>> fft(std::span<const float> x, std::optional<std::int64_t> n = std::nullopt);
std::vector<std::complex<float>> ifft(std::span<const std::complex<float>> x,
                                      std::optional<std::int64_t> n = std::nullopt);
std::vector<std::complex<float>> rfft(std::span<const float> x, std::optional<std::int64_t> n = std::nullopt);
std::vector<float> irfft(std::span<const std::complex<float>> x, std::optional<std::int64_t> n = std::nullopt);

// np.fft.fft of data, in place (forward, unscaled): the transform numpy
// applies to each row once its input is copied in, for callers that
// reuse one buffer across many frames.
void fft_inplace(std::span<std::complex<double>> data);

// np.fft.fft(a, n, axis=1) of a row-major rows x cols matrix: rows x n.
std::vector<std::complex<double>> fft_rows(std::span<const std::complex<double>> a, std::size_t rows,
                                           std::size_t cols, std::optional<std::int64_t> n = std::nullopt);
std::vector<std::complex<float>> fft_rows(std::span<const std::complex<float>> a, std::size_t rows,
                                          std::size_t cols, std::optional<std::int64_t> n = std::nullopt);

// np.fft.fftfreq(n, d) and np.fft.rfftfreq(n, d): integer k times
// val = 1.0 / (n * d).
std::vector<double> fftfreq(std::int64_t n, double d = 1.0);
std::vector<double> rfftfreq(std::int64_t n, double d = 1.0);

}  // namespace sstvae::math

#endif  // SSTVAE_MATH_NPFFT_HPP

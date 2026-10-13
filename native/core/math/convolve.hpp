// scipy's FFT convolutions, frequency response and 1-D maximum filter as
// the QRSSTVAE receiver calls them (scipy 1.18.1):
//
//   scipy.signal.oaconvolve(in1, in2, mode)   frontend.audio_to_fe/fe_to_audio,
//                                             track, ltrack, acquire
//   scipy.signal.fftconvolve(in1, in2, mode)  (oaconvolve's fallbacks)
//   scipy.signal.freqz(b, worN=array, fs=fs)  frontend's noise-floor masks
//   scipy.ndimage.maximum_filter1d(x, size)   frontend.blank
//
// 1-D only, modes 'full' and 'same', real or complex operands.
//
// What is copied from scipy, so the results agree to rounding:
//
// * oaconvolve's block arithmetic exactly: `_calc_oa_lens`'s optimal
//   block (-K W_{-1}(-1/(2 e K)) through scipy.fft.next_fast_len, K the
//   overlap), the step and padding counts, the fallbacks to fftconvolve
//   (equal lengths, one block) and to a plain product (a length-1
//   operand), the overlap-add, and the centring of 'same'. So the same
//   samples meet in the same transforms of the same sizes (2420 for the
//   front end's 255 taps).
// * The transforms: scipy's are ducc0's; these are pocketfft's
//   (dsp/fft.hpp), the same algorithm family. A real operand of a complex
//   convolution is transformed as ducc's c2c does a real input (a real
//   transform with the other half filled by conjugates), a real
//   convolution through r2c/c2r, the inverse scaled by 1/n inside the
//   transform. The spectra multiply as numpy does (npcomplex.hpp, fused).
//   The two FFT libraries round differently, so the results agree to
//   ~1e-16 of the output's scale, not to the bit.
// * freqz with an array worN, bit for bit: w = 2 pi f / fs, z^-1 =
//   exp(-1j w) (cos and sin of -w, as glibc's cexp gives them), Horner's
//   rule from the last coefficient with numpy's fused complex multiply,
//   and the division by the denominator 1.
// * maximum_filter1d with mode 'reflect', origin 0, on an integer array:
//   out[i] = max of x[i - size//2 .. i - size//2 + size - 1] with the
//   line extended by reflection (d c b a | a b c d | d c b a), repeatedly
//   for a line shorter than the window.

#ifndef SSTVAE_MATH_CONVOLVE_HPP
#define SSTVAE_MATH_CONVOLVE_HPP

#include <complex>
#include <cstdint>
#include <span>
#include <vector>

namespace sstvae::math {

enum class ConvMode { Full, Same };

// scipy.fft.next_fast_len(target, real): the smallest n >= target whose
// prime factors are all 2, 3, 5, 7 or 11 (2, 3 or 5 when real).
// std::invalid_argument for target < 1.
std::int64_t next_fast_len(std::int64_t target, bool real = false);

// scipy.special.lambertw(x, k=-1).real for x in [-1/e, 0): the lower
// branch. NaN outside it.
double lambertw_m1(double x);

// scipy.signal._signaltools._calc_oa_lens(s1, s2). overlap is -1 for
// scipy's None (the fallback).
struct OaLens {
    std::int64_t block_size;
    std::int64_t overlap;
    std::int64_t in1_step;
    std::int64_t in2_step;
};
OaLens calc_oa_lens(std::int64_t s1, std::int64_t s2);

// scipy.signal.oaconvolve(in1, in2, mode) and fftconvolve, 1-D. An empty
// operand gives an empty result (scipy's np.asarray([])).
std::vector<std::complex<double>> oaconvolve(std::span<const std::complex<double>> in1,
                                             std::span<const double> in2, ConvMode mode);
std::vector<std::complex<double>> oaconvolve(std::span<const std::complex<double>> in1,
                                             std::span<const std::complex<double>> in2, ConvMode mode);
std::vector<std::complex<double>> oaconvolve(std::span<const double> in1,
                                             std::span<const std::complex<double>> in2, ConvMode mode);
std::vector<double> oaconvolve(std::span<const double> in1, std::span<const double> in2, ConvMode mode);

std::vector<std::complex<double>> fftconvolve(std::span<const std::complex<double>> in1,
                                              std::span<const double> in2, ConvMode mode);
std::vector<std::complex<double>> fftconvolve(std::span<const std::complex<double>> in1,
                                              std::span<const std::complex<double>> in2, ConvMode mode);
std::vector<std::complex<double>> fftconvolve(std::span<const double> in1,
                                              std::span<const std::complex<double>> in2, ConvMode mode);
std::vector<double> fftconvolve(std::span<const double> in1, std::span<const double> in2, ConvMode mode);

// signal.freqz(b, worN=f, fs=fs)[1]: H at the frequencies f (in the units
// of fs) of the FIR filter b (denominator 1).
std::vector<std::complex<double>> freqz(std::span<const double> b, std::span<const double> f, double fs);

// ndimage.maximum_filter1d(x, size) (mode 'reflect', origin 0) on an
// integer (or bool-as-uint8) array. std::runtime_error (scipy's
// RuntimeError) "incorrect filter size" for size < 1, as scipy refuses it.
std::vector<std::uint8_t> maximum_filter1d(std::span<const std::uint8_t> x, std::int64_t size);

}  // namespace sstvae::math

#endif  // SSTVAE_MATH_CONVOLVE_HPP

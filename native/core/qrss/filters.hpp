// scipy-compatible FIR filters for QRSSTVAE: upfirdn, firwin with
// scipy's own windows, resample_poly, and the Cephes Bessel functions
// i0/i0e. Ports of what sstvae/qrss takes from scipy 1.18:
//
//   scipy.signal.upfirdn         ce._grid_sum, lwave._grid_sum, lwave.resample
//   scipy.signal.firwin          lwave._interp_taps, frontend._FE_TAPS, the
//                                channeliser's decimator, acquire's notch
//   scipy.signal.resample_poly   ce.to_audio, frontend.channelise(_keep)
//   scipy.signal.windows.hann    acquire/frontend spectral frames (sym=False)
//   scipy.special.i0, i0e        the Kaiser window; cwid._ln_i0
//
// Python (numpy and scipy) is the reference, and these are written to
// agree with it bit for bit, not merely to a tolerance:
//
// - upfirdn copies the loop of scipy's _upfirdn_apply.pyx exactly: the
//   transposed, flipped polyphase taps, the zero-pad skip on the left,
//   the out[y] = out[y] + x*h accumulation order, and the "flush" loop
//   that adds cval*h past the right end. Complex samples multiply as C99
//   does, (ac - bd) + i(ad + bc), with real taps cast to complex first
//   (imaginary part +0.0), which is how scipy sees them.
// - firwin follows _fir_filter_design.firwin's numpy operation order:
//   np.sinc as sin(pi x)/(pi x) with eps at zero, h built band by band,
//   the window applied, and the scale taken as numpy's pairwise sum.
// - The windows follow scipy.signal.windows: general_cosine over
//   np.linspace(-pi, pi, M) (numpy's own linspace arithmetic) and the
//   Kaiser window through Cephes i0, not a power series.
// - i0/i0e are Cephes' Chebyshev expansions as scipy.special ships them
//   (xsf), including i0's overflow to inf past log(DBL_MAX) and NaN at inf.
//
// What the bitwise claim rests on: libm sin/cos/exp/sqrt (numpy's float64
// sin and cos call libm on x86-64; scipy's i0 calls std::exp), and no
// FMA contraction (-ffp-contract=off on sstvae_qrss).
//
// Errors are std::invalid_argument carrying scipy's ValueError text.

#ifndef SSTVAE_QRSS_FILTERS_HPP
#define SSTVAE_QRSS_FILTERS_HPP

#include <complex>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

namespace sstvae::qrss::filters {

// --- special functions --------------------------------------------------------

// scipy.special.i0: the modified Bessel function of order 0 (Cephes).
double i0(double x);

// scipy.special.i0e: exp(-|x|) i0(x) (Cephes).
double i0e(double x);

// --- windows ------------------------------------------------------------------

// The windows firwin and resample_poly are given in sstvae/qrss.
struct Window {
    enum class Kind { Hamming, Hann, Kaiser };
    Kind kind = Kind::Hamming;
    double beta = 0.0;   // Kaiser only
    // A '_symmetric' / '_periodic' suffix on scipy's window name, which
    // overrides get_window's fftbins (and so firwin's symmetric window).
    std::optional<bool> sym;

    static Window hamming() { return {Kind::Hamming, 0.0, std::nullopt}; }
    static Window hann() { return {Kind::Hann, 0.0, std::nullopt}; }
    static Window kaiser(double beta) { return {Kind::Kaiser, beta, std::nullopt}; }
};

// scipy.signal.windows.general_hamming(M, alpha, sym): alpha + (1 - alpha)
// cos over np.linspace(-pi, pi, M). M < 0 is refused; M <= 1 is ones(M).
std::vector<double> general_hamming(std::int64_t M, double alpha, bool sym = true);

// scipy.signal.windows.hann / hamming / kaiser.
std::vector<double> hann(std::int64_t M, bool sym = true);
std::vector<double> hamming(std::int64_t M, bool sym = true);
std::vector<double> kaiser(std::int64_t M, double beta, bool sym = true);

// scipy.signal.get_window(window, Nx, fftbins): Nx must be positive. The
// window is symmetric when fftbins is false, unless window.sym says otherwise.
std::vector<double> get_window(const Window& window, std::int64_t Nx, bool fftbins = true);

// --- firwin -------------------------------------------------------------------

// scipy's pass_zero: True (Yes), False (No), or one of the four strings.
// (Not `True`/`False`: X11 defines those as macros.)
enum class PassZero { Yes, No, Lowpass, Highpass, Bandpass, Bandstop };

// scipy.signal.kaiser_atten / kaiser_beta.
double kaiser_atten(std::int64_t numtaps, double width);
double kaiser_beta(double a);

// scipy.signal.firwin(numtaps, cutoff, width=, window=, pass_zero=, scale=,
// fs=). cutoff is in the units of fs (fs = 2 when absent, i.e. relative to
// Nyquist); `width`, when given, replaces the window by a Kaiser window
// designed for that transition width.
std::vector<double> firwin(std::int64_t numtaps, std::span<const double> cutoff,
                           const Window& window = Window::hamming(),
                           PassZero pass_zero = PassZero::Yes, bool scale = true,
                           std::optional<double> fs = std::nullopt,
                           std::optional<double> width = std::nullopt);

inline std::vector<double> firwin(std::int64_t numtaps, double cutoff,
                                  const Window& window = Window::hamming(),
                                  PassZero pass_zero = PassZero::Yes, bool scale = true,
                                  std::optional<double> fs = std::nullopt,
                                  std::optional<double> width = std::nullopt) {
    return firwin(numtaps, std::span<const double>(&cutoff, 1), window, pass_zero, scale, fs, width);
}

// --- upfirdn ------------------------------------------------------------------

// scipy's _output_len: ceil(((in_len - 1) * up + len_h) / down), by floor
// division, so it is negative for some empty inputs (which upfirdn refuses
// as numpy refuses a negative dimension). In int64 that wraps, as Cython's
// does; down == 0 throws std::domain_error and INT64_MIN // -1
// std::overflow_error (Cython's ZeroDivisionError and OverflowError)
// rather than trapping.
std::int64_t upfirdn_output_len(std::int64_t len_h, std::int64_t in_len, std::int64_t up,
                                std::int64_t down);

// scipy.signal.upfirdn(h, x, up, down, mode='constant', cval=cval) on a 1-D
// signal, in the output type T (float, double, complex<float> or
// complex<double>): taps and samples are already cast to it, as scipy
// casts both to result_type(h, x, float32).
template <typename T>
std::vector<T> upfirdn(std::span<const T> h, std::span<const T> x, std::int64_t up = 1,
                       std::int64_t down = 1, T cval = T{});

// Real float64 taps over complex128 samples (the L transmitter's case):
// the taps become complex with imaginary part +0.0 first, as in scipy.
std::vector<std::complex<double>> upfirdn(std::span<const double> h,
                                          std::span<const std::complex<double>> x,
                                          std::int64_t up = 1, std::int64_t down = 1);

// --- resample_poly ------------------------------------------------------------

// scipy.signal.resample_poly(x, up, down, window=window, padtype='constant',
// cval=cval) on a 1-D signal of floating type T. The filter is
// firwin(2*10*max(up,down)+1, 1/max(up,down), window) cast to T and then
// multiplied by up in T (after up/down are reduced by their gcd), as scipy
// does for a floating x. up == down returns a copy.
template <typename T>
std::vector<T> resample_poly(std::span<const T> x, std::int64_t up, std::int64_t down,
                             const Window& window = Window::kaiser(5.0), T cval = T{});

// resample_poly's designed filter for an already-reduced up/down, in T and
// multiplied by up: firwin(2*half_len + 1, 1/max_rate, window) with
// half_len = 10*max(up, down), cast to T and then scaled in T.
template <typename T>
std::vector<T> resample_poly_filter(std::int64_t up, std::int64_t down,
                                    const Window& window = Window::kaiser(5.0));

// The part of resample_poly after the filter is designed: `h` is the
// filter already multiplied by `up`, in T; up and down are already reduced;
// half_len is scipy's (10*max(up,down) for a designed filter, or
// (len(h) - 1) // 2 for one given as an array).
template <typename T>
std::vector<T> resample_poly_taps(std::span<const T> x, std::int64_t up, std::int64_t down,
                                  std::span<const T> h, std::int64_t half_len, T cval = T{});

}  // namespace sstvae::qrss::filters

#endif  // SSTVAE_QRSS_FILTERS_HPP

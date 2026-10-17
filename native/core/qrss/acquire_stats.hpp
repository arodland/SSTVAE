// The QRSSTVAE receiver's acquisition statistics: the pure-array functions
// of sstvae/qrss/acquire.py that A1 (carrier lines), A2 (preamble search)
// and A3 (track-before-detect) share, ported for the A1/A2 and A3 lanes
// to build on.
//
//   acquire.py               here
//   _frame_len, frame_window frame_len, frame_window
//   spectrogram              spectrogram (float32 frames x bins)
//   band_bins                band_bins
//   cfar_normalise           cfar_normalise
//   spectral_shape           spectral_shape
//   _parabola                parabola (float64 and float32 arithmetic)
//   a1_gamma, a1_threshold   a1_gamma, a1_threshold
//   a2_threshold             a2_threshold
//   preamble_scales          preamble_scales
//   preamble_p               preamble_p
//   tbd_gumbel, gumbel_p     tbd_gumbel, gumbel_p
//   _band_cov                band_cov (cached per fs, thread-safe)
//
// (track._parabola, which ltrack imports, is the same function as
// acquire._parabola; it belongs to the track lane.)
//
// Bit-exact against the reference -- the same numpy/scipy arithmetic in
// the same order (core/math's ports of gammaincc/gammainccinv,
// median_filter, np.median, numpy's pairwise sums and pocketfft FFT; numpy
// float32 arithmetic kept in float32 where NEP 50 keeps it there) -- with
// one exception: gumbel_p calls np.exp and np.expm1, which numpy runs
// through its own SIMD (AVX-512) kernels rather than libm on a machine
// that has them, so it agrees to an ulp or two, not to the bit.
//
// Errors are the reference's: ValueError (std::invalid_argument),
// IndexError (std::out_of_range), ZeroDivisionError (qrss::ZeroDivision).
// Thread-safe: band_cov's cache is mutex-protected and holds immutable
// tables; everything else is pure.

#ifndef SSTVAE_QRSS_ACQUIRE_STATS_HPP
#define SSTVAE_QRSS_ACQUIRE_STATS_HPP

#include <array>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <utility>
#include <vector>

#include "qrss/qrss_config.hpp"
#include "qrss/rx_config.hpp"

namespace sstvae::qrss::acquire {

namespace rxa = sstvae::qrss::rxconfig::acquire;

// A row-major 2-D numpy array (rows x cols) of T: float32 for the
// spectrogram and its normalised forms, as numpy's float32[n_frames, n_bins].
template <typename T>
struct Grid {
    std::size_t rows = 0;
    std::size_t cols = 0;
    std::vector<T> data;

    Grid() = default;
    Grid(std::size_t r, std::size_t c) : rows(r), cols(c), data(r * c, T(0)) {}
    T& operator()(std::size_t i, std::size_t j) { return data[i * cols + j]; }
    const T& operator()(std::size_t i, std::size_t j) const { return data[i * cols + j]; }
    std::span<const T> row(std::size_t i) const { return {data.data() + i * cols, cols}; }
    bool operator==(const Grid&) const = default;
};
using Grid32 = Grid<float>;

// int(round(STFT_FRAME_S * fs)): samples in one 4 s STFT frame.
std::int64_t frame_len(double fs);

// scipy.signal.windows.hann(frame_len(fs), sym=False).
std::vector<double> frame_window(double fs);

// spectrogram(x, fs, starts, bins): float32[len(starts), n_bins] of
// |FFT(w * frame)|^2 for frames starting at round(starts[i]) (half to
// even; samples outside x count 0), keeping the fftfreq bins `bins`
// (negative indices count from the end, as numpy's; all L bins when
// absent). complex64 input is widened exactly, as numpy's `* w` does,
// and windowed by numpy's complex product with w promoted to w + 0j
// (re*w - im*0, re*0 + im*w), so a non-finite sample spreads NaN exactly
// as numpy's does (an inf real part makes a NaN imaginary one). Real
// input (float64, or float32 widened) is windowed as a real product and
// promoted to complex afterwards, as np.fft.fft promotes it.
Grid32 spectrogram(std::span<const std::complex<float>> x, double fs, std::span<const double> starts,
                   std::optional<std::span<const std::int64_t>> bins = std::nullopt);
Grid32 spectrogram(std::span<const std::complex<double>> x, double fs, std::span<const double> starts,
                   std::optional<std::span<const std::int64_t>> bins = std::nullopt);
Grid32 spectrogram(std::span<const double> x, double fs, std::span<const double> starts,
                   std::optional<std::span<const std::int64_t>> bins = std::nullopt);

// band_bins(fs, band_hz, centre_hz): the fftfreq indices of a 4 s frame
// whose audio frequency (fftfreq + centre_hz) lies in [band_hz[0],
// band_hz[1]], in ascending frequency, and those frequencies.
struct BandBins {
    std::vector<std::int64_t> sel;
    std::vector<double> f_hz;
};
BandBins band_bins(double fs, std::array<double, 2> band_hz = rxa::BAND_HZ,
                   double centre_hz = config::FE_CENTER_HZ);

// cfar_normalise(P, local_bins): each frame divided by its median / ln 2
// (in P's precision), 0 for a frame whose median is not > 0; then, when
// local_bins is present and nonzero, each bin divided by spectral_shape
// (in float64) and the result rounded to float32.
Grid32 cfar_normalise(const Grid<float>& P, std::optional<int> local_bins = rxa::CFAR_LOCAL_BINS);
Grid32 cfar_normalise(const Grid<double>& P, std::optional<int> local_bins = rxa::CFAR_LOCAL_BINS);

// spectral_shape(Pn, local_bins, rows): float64[n_bins], the median over
// the frames selected by `rows` (a mask of length Pn.rows; all frames
// when absent) / ln 2, median-filtered over local_bins bins (reflected;
// the window odd and at most the largest odd count <= n_bins), 1 where
// that is not > 0 or no frame is selected. A mask of the wrong length is
// std::out_of_range (numpy's IndexError).
std::vector<double> spectral_shape(const Grid<float>& Pn, int local_bins = rxa::CFAR_LOCAL_BINS,
                                   std::optional<std::span<const std::uint8_t>> rows = std::nullopt);
std::vector<double> spectral_shape(const Grid<double>& Pn, int local_bins = rxa::CFAR_LOCAL_BINS,
                                   std::optional<std::span<const std::uint8_t>> rows = std::nullopt);

// _parabola(ym, y0, yp): the vertex offset, clipped to [-1/2, 1/2], of the
// parabola through (-1, ym), (0, y0), (1, yp); 0 unless its curvature is
// negative and finite. The float overload computes in float32, as numpy
// does for np.float32 arguments (acquire.carrier_lines' log levels).
double parabola(double ym, double y0, double yp);
double parabola(float ym, float y0, float yp);

// a1_gamma(M, fs): (shape, scale) of A1's statistic on noise.
std::pair<double, double> a1_gamma(std::int64_t M, double fs = config::FE_FS);

// a1_threshold(M, pfa, fs) = gammainccinv(shape, pfa) * scale.
double a1_threshold(std::int64_t M, double pfa = rxa::A1_PFA, double fs = config::FE_FS);

// a2_threshold(pfa, n_chunks, n_scales) = gammainccinv(n_chunks, pfa / n_scales).
double a2_threshold(double pfa = rxa::A2_PFA, std::int64_t n_chunks = config::N_PRE / rxa::A2_CHUNK,
                    std::int64_t n_scales = static_cast<std::int64_t>(rxa::A2_SCALES.size()));

// One scale of preamble_scales: k groups, Lambda_k[n_df, n_tau] row-major.
struct Scale {
    std::int64_t k = 0;
    std::vector<double> lam;
};

// How numpy adds up preamble_scales' runs (sum over g) and groups (sum
// over k), which is a property of the arrays' memory layout, not of the
// values: when Sn = S / sqrt(var)[None] comes out with its chunk axis
// innermost in memory, each reduction is one call of numpy's pairwise sum
// (onto 0); otherwise the iterator puts the chunk axis outside and adds
// one term at a time, ((0 + a0) + a1) + ..., which differs in the last
// bit from 8 terms up (runs of g >= 5 complex, k >= 9 groups). The S
// acquire._preamble_sums builds (an einsum output) is Fortran-ordered and
// takes the sequential order; a C-ordered S the pairwise one.
enum class SumOrder { Pairwise, Sequential };

// preamble_scales(S, var, scales): S complex128[n_df, n_tau, nc] and var
// float64[n_tau, nc], row-major. Each scale g >= 1 (0 is ZeroDivision,
// negative std::invalid_argument).
std::vector<Scale> preamble_scales(std::span<const std::complex<double>> S, std::size_t n_df, std::size_t n_tau,
                                   std::size_t nc, std::span<const double> var,
                                   std::span<const int> scales = rxa::A2_SCALES,
                                   SumOrder order = SumOrder::Pairwise);

// preamble_p(S, var, scales): per cell the best scale's gammaincc(k,
// Lambda_k) times the number of scales, capped at 1 (NaN stays NaN), the
// index of that scale (the first minimum; a NaN counts as the minimum,
// as np.argmin's), and the scales themselves.
struct PreambleP {
    std::vector<double> p;            // [n_df, n_tau]
    std::vector<std::int64_t> best;   // [n_df, n_tau]
    std::vector<Scale> st;
};
PreambleP preamble_p(std::span<const std::complex<double>> S, std::size_t n_df, std::size_t n_tau, std::size_t nc,
                     std::span<const double> var, std::span<const int> scales = rxa::A2_SCALES,
                     SumOrder order = SumOrder::Pairwise);

// tbd_gumbel(n_frames): (mu, beta) of A3's Z on noise, interpolated in
// sqrt(n_frames) through the committed table (extrapolated outside it).
// n_frames < 0 is std::invalid_argument "math domain error" (math.sqrt's).
std::pair<double, double> tbd_gumbel(std::int64_t n_frames);

// gumbel_p(z, n_frames) = -expm1(-exp(-(z - mu) / beta)).
double gumbel_p(double z, std::int64_t n_frames);

// _band_cov(n_bins, fs): (v0, v1), the variance of a sum of n_bins
// adjacent normalised bins of one Hann frame and the covariance of two
// such sums in consecutive (50%-overlapping) frames, on white noise.
std::pair<double, double> band_cov(std::int64_t n_bins, double fs = config::FE_FS);

}  // namespace sstvae::qrss::acquire

#endif  // SSTVAE_QRSS_ACQUIRE_STATS_HPP

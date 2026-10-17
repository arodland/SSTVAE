// Timing and verification, the rest of sstvae/qrss/track.py (design 6.3 V,
// 6.4, 6.5 step 1): the coarse timing fit (fit_timing, _segment_timing,
// _parabola), gate V (verify, verify_track), the frequency-path
// measurements on the frame (_carrier_lines, refine_path, line_path) and
// _flat_path. The tracker itself (track, refine_timing, templates, ...) is
// track.hpp's; this builds on it.
//
// Not a format module: nothing here goes on the air. Constants are
// rx_config.hpp's (generated from track.py).
//
// Python is normative. Where the two agree and why they do not agree to
// the bit:
//
// * fit_timing correlates the known symbols with x at every lag by the
//   DIRECT sparse gather -- the reference's own path for nlag <= 64 --
//   for every search, where the reference switches to FFT correlation
//   (588 rfft/irfft pairs of 2^19 points on a FULL blind search) above
//   64 lags. The two are the same sum: each term a_k ((1 - f) x[i] + f
//   x[i + 1]) is the reference's direct-path product; the sum over k runs
//   in k order where the reference's direct path is BLAS dgemv's order
//   and its FFT path a transform. So the grid agrees to rounding (a few
//   ulps of the largest term, measured in the parity test), with one
//   structural exception: a term whose interpolation pair straddles the
//   end of x counts 0 here and in the reference's direct path, while its
//   FFT path keeps the half inside. The receiver's captures are padded
//   past both ends of every lag searched (receiver.channel_for), so no
//   such term arises there.
// * _segment_timing, the lstsq line fits, the BLAS dot products, scipy's
//   FFT convolutions (_lowpass, the dense MF) and PathFn's spline fits
//   agree to rounding. Every decision (the argmax cells, the runners-up,
//   the segments kept, the outlier cut, the Newton branch, the
//   accept/reject) is the reference's on inputs whose deciding values are
//   not within rounding of a threshold.
// * _carrier_lines' spectra are numpy's to the bit given the same z
//   (pocketfft, np.hanning through libm cos, numpy's median); the z they
//   are taken of comes through a fitted PathFn, so refine_path and
//   line_path agree to rounding (1e-7 Hz is the bar).
// * line_path's search sums are numpy's (a sequential sum over segments,
//   which is what .sum(axis=0) of that array is).
//
// Errors are the reference's types: frame::WrongWaveform for an L frame
// (Python's AssertionError from _ce_only), std::invalid_argument for
// ValueError, std::out_of_range for IndexError, math::LinAlgError.
//
// Thread safety: every function is reentrant; a ChanCapture's noise table
// is filled once (its NoisePsdSlot) whichever thread asks first.

#ifndef SSTVAE_QRSS_TRACK_TIMING_HPP
#define SSTVAE_QRSS_TRACK_TIMING_HPP

#include <complex>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <utility>
#include <vector>

#include "qrss/chan.hpp"
#include "qrss/frame.hpp"
#include "qrss/rx_config.hpp"
#include "qrss/track.hpp"
#include "qrss/trackbase.hpp"
#include "qrss/types.hpp"

namespace sstvae::qrss::track {

// --- the coarse timing fit ------------------------------------------------------

// _parabola(ym, y0, yp): the vertex offset in [-1/2, 1/2] of the parabola
// through (-1, ym), (0, y0), (1, yp); 0 unless its curvature is negative
// and finite. The same function as acquire._parabola (acquire_stats.hpp's
// parabola, float64 arithmetic).
double parabola(double ym, double y0, double yp);

// fit_timing's keyword arguments. known: tmpl_fn's known symbols (null:
// the frame's own, known_symbols(spec)); tau0_init narrows the lag search
// to tau_range (default 3) CH samples either side; ppm_range the clock
// errors searched; n_best the peaks kept per carrier estimate (and the
// timings returned).
struct FitTimingArgs {
    const Known* known = nullptr;
    std::optional<double> tau0_init;
    double ppm_range = 250.0;
    std::optional<double> tau_range;
    std::int64_t n_best = 1;
};

// One peak as fit_timing found it, for tests and diagnostics: the carrier
// estimate (index into CARRIER_LP_HZ), the grid cell (clock-error index i,
// lag index j) with its parabola offsets and Z, _segment_timing's inputs
// and what it returned.
struct FitPeak {
    std::size_t lp = 0;
    std::int64_t i = 0, j = 0;
    double di = 0.0, dj = 0.0;
    double zmax = 0.0;
    double t0abs_in = 0.0, delta_in = 0.0;
    double t0abs = 0.0, delta = 0.0, score = 0.0;
};

// What one fit_timing call computed on the way (filled when asked for).
struct FitTimingTrace {
    std::int64_t base = 0, nlag = 0, nd = 0;
    double frac0 = 0.0, step = 0.0;
    std::vector<double> sig2;          // per carrier estimate
    std::vector<FitPeak> peaks;        // in the order found
};

// fit_timing (design 6.4 steps 1-2) on the derotated channel z_d (the
// capture's samples times exp(-2 pi j theta_path)). CE only. Returns the
// timings strongest first: one for n_best <= 1 (the reference's single
// Timing), else up to n_best. Timing.z is the peak's correlation in units
// of its noise sd; cov is diag(0.25, (step 1e6)^2, 0).
std::vector<types::Timing> fit_timing(std::span<const std::complex<double>> z_d, int ch_fs,
                                      const frame::FrameSpec& spec, double t0_index,
                                      const FitTimingArgs& args = {}, FitTimingTrace* trace = nullptr);
// The same with z_d = derotate(chan, fn).
std::vector<types::Timing> fit_timing(const ChanCapture& chan, int ch_fs, const PathFn& fn,
                                      const frame::FrameSpec& spec, double t0_index,
                                      const FitTimingArgs& args = {}, FitTimingTrace* trace = nullptr);

// fit_timing(n_best = 1) and fit_timing(n_best = args.n_best) from one
// correlation grid (the reference's _fit_timing_tries): gate V's blind
// candidate tracks the best timing first and the runners-up only when it
// fails, and the grid is nearly all of either call's cost. Each is that
// call's result exactly (n_best = 1's peaks are every carrier estimate's
// first, which a search for more finds first too). The trace is the
// n_best search's.
struct FitTimingTries {
    types::Timing best;                  // fit_timing(..., n_best = 1)
    std::vector<types::Timing> tries;    // fit_timing(..., n_best)
};
FitTimingTries fit_timing_tries(std::span<const std::complex<double>> z_d, int ch_fs, const frame::FrameSpec& spec,
                                double t0_index, const FitTimingArgs& args = {}, FitTimingTrace* trace = nullptr);

// One peak of the grid search (fit_timing's, or fit_timing_l's): its rank
// k in its carrier estimate, _segment_timing's score, the grid's Z there
// and the refined (t0abs, delta).
struct TimingPeak {
    std::int64_t k = 0;
    double score = 0.0, zmax = 0.0, t0abs = 0.0, delta = 0.0;
};
// _timings: the timings from the peaks (in the order found), strongest
// score first (a stable sort, NaN last), one per sps of lag, up to
// max(1, n_best); cov diag(0.25, (step 1e6)^2, 0).
std::vector<types::Timing> timings_of(std::vector<TimingPeak> peaks, std::int64_t n_best, double sps, double step,
                                      double t0_index);
// _fit_timing_tries' pair from the peaks: timings_of the rank-0 peaks
// with n_best 1, and of every peak with n_best.
FitTimingTries tries_of(std::vector<TimingPeak> peaks, std::int64_t n_best, double sps, double step,
                        double t0_index);

// The statistic fit_timing correlates: Im(MF{z_d conj(LP(z_d))}) with LP
// the _lowpass at lp_hz and MF the dense symbol MF (_mf_taps), through
// scipy's oaconvolve "same".
std::vector<double> fit_statistic(std::span<const std::complex<double>> z_d, double lp_hz, int fs = config::CH_FS);

// The coarse grid, row-major (deltas.size() x nlag): Z[i, j] = sum_k a_k
// ((1 - f) x[n] + f x[n + 1]) / norm with n + f = base + o_k + j, o_k =
// pos_k sps (1 + deltas[i]) + frac0, a term 0 unless n >= 0 and n + 1 <
// len(x) (the reference's direct path).
std::vector<double> fit_grid(std::span<const double> x, const Known& known, std::int64_t base, std::int64_t nlag,
                             double frac0, std::span<const double> deltas, double sps, double norm);

// _segment_timing: (t0abs, delta) refined from the local timing offsets of
// seg_s segments of the known symbols, and the final line's score (sum of
// the segments' Z^2 at it, negatives counted 0).
struct SegmentTiming {
    double t0abs = 0.0;
    double delta = 0.0;
    double score = 0.0;
};
SegmentTiming segment_timing(std::span<const double> x, double sig2, const Known& known, double t0abs, double delta,
                             double sps, std::int64_t n_pos, double seg_s = rxconfig::track::SEG_TIMING_S);

// --- the frequency path, measured on the frame ----------------------------------

// _carrier_lines: (t, f, ratio) of the carrier line in each seg_s segment of
// z over the frame, each searched +-span_hz around the last one found;
// med_band (lo, hi) moves the median's region.
struct CarrierLines {
    std::vector<double> t;
    std::vector<double> f;
    std::vector<double> w;
};
CarrierLines carrier_lines(std::span<const std::complex<double>> z, const ChanCapture& chan,
                           const frame::FrameSpec& spec, double seg_s, double span_hz,
                           std::optional<std::pair<double, double>> med_band = std::nullopt, double ratio = 10.0);

// A frequency path a measurement returned; `unchanged` when it is the
// path it was given (the reference returns that very object).
struct PathResult {
    types::FreqPath path;
    bool unchanged = false;
};

// refine_path's keyword arguments.
struct RefinePathArgs {
    double seg_s = rxconfig::track::PATH_SEG_S;
    double span_hz = rxconfig::track::PATH_SPAN_HZ;
    std::optional<std::pair<double, double>> med_band;
    double ratio = 10.0;
    double fine_ratio = rxconfig::track::PATH_FINE_RATIO;
    double fine_share = rxconfig::track::PATH_FINE_SHARE;
};

// refine_path: the carrier's frequency path measured on the frame (path
// plus each segment's residual line, and a finer second pass on a strong
// carrier).
PathResult refine_path(const ChanCapture& chan, const frame::FrameSpec& spec, const types::FreqPath& path,
                       const RefinePathArgs& args = {});

// line_path: the best straight frequency line over the whole frame near
// path's.
PathResult line_path(const ChanCapture& chan, const frame::FrameSpec& spec, const types::FreqPath& path);

// _flat_path: path's frequency at t = 0, flat from -10 s to 1800 s.
types::FreqPath flat_path(const types::FreqPath& path);

// --- gate V ------------------------------------------------------------------------

// verify / verify_track's keyword arguments. erase_s: spans (a, b),
// seconds after t0, that were not heard. noise_psd: how the capture's
// noise table is computed if its slot is empty (empty: the C++ front end).
struct VerifyArgs {
    bool fit = true;
    std::optional<double> tau_range;
    std::vector<std::pair<double, double>> erase_s;
    NoisePsdFn noise_psd;
};

// verify_track: the tracker run behind verify (known symbols withheld, no
// refinement, no re-centring) with tr.z_ref set over what was heard. CE
// only.
TrackResult verify_track(std::shared_ptr<const ChanCapture> chan, const frame::FrameSpec& spec,
                         const types::Detection& cand, const VerifyArgs& args = {});

// verify: gate V (design 6.3). The candidate with its measured path, timing
// and z_ref, or nullopt (Python's None) when Z_ref never exceeds Z_ACCEPT.
// The result's a3_z is NaN (the reference builds a fresh Detection). CE
// only.
std::optional<types::Detection> verify(std::shared_ptr<const ChanCapture> chan, const frame::FrameSpec& spec,
                                       const types::Detection& cand, const VerifyArgs& args = {});

}  // namespace sstvae::qrss::track

#endif  // SSTVAE_QRSS_TRACK_TIMING_HPP

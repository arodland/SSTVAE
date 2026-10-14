// Timing and verification for waveform L, the rest of sstvae/qrss/ltrack.py
// (docs/qrss/waveform-l.md; design 6.3 V, 6.4): the receiver-time mix's
// phases (_lag_phase, _clock_phase), the coarse timing fit (fit_timing_l,
// _segment_timing_l), the pilot's frequency path measured on the frame
// (refine_path_l), the residual check (resid_ratio) and gate V
// (verify_track_l, verify_l). The tracker itself (track_l and its pieces)
// is ltrack.hpp's; CE's counterparts are track_timing.hpp's, whose gather
// kernel this reuses.
//
// Not a format module: nothing here goes on the air. Constants are
// rx_config.hpp's (generated from ltrack.py and track.py).
//
// Python is normative. What agrees to the bit with the reference on the
// same inputs, and what does not:
//
// * Bitwise: lag_phase and clock_phase (np.exp of a purely imaginary
//   argument is (cos, sin) of it through libm), resid_ratio (numpy's
//   pairwise sums), and the composition of verify_track_l and verify_l:
//   with the C++ primitives they call assigned over the reference's, the
//   reference's verify_track_l and verify_l are these to the bit.
// * fit_timing_l correlates the known symbols with the complex statistic x
//   at every lag by the DIRECT sparse gather -- the reference's own path
//   for nlag <= 64 -- for every search, where the reference switches to a
//   complex FFT correlation (2 x 4 transforms of 2^19 points per clock
//   error on an L FULL blind search) above 64 lags. The gather is
//   track::fit_grid's, run four times per clock error (the real and
//   imaginary parts of g_k = conj(a_k) clock_phase(pos_k, d) against the
//   real and imaginary parts of x), so the sum over k runs in k order where
//   the reference's direct path is BLAS zgemv's order and its FFT path a
//   transform: the grid agrees to rounding. One structural exception, as
//   CE's: a term whose interpolation pair straddles either end of x (its
//   pair x[-1], x[0] or x[len - 1], x[len]) counts 0 here and in the
//   reference's direct path, while its FFT path keeps the half inside --
//   measured 1.4% of the grid's scale at such cells on a capture starting
//   inside the blind search. The receiver's captures are zero-padded past
//   both ends of every lag searched (receiver.channel_for pads L as it
//   pads CE: the channel spans about 13.5 s before t0, against the blind
//   search's 2.5 s, and 3 s past its farthest lag at 260 ppm of slip after
//   the frame, zero-filled where the capture does not reach), so no such
//   term arises there (test_receiver_channel_holds_every_lag pins that). A
//   non-finite sample of x is the other: numpy's
//   complex `v * ok` makes both parts NaN where either is not finite, the
//   four real gathers keep the finite part (a capture holding inf or NaN).
// * _segment_timing_l's correlations are BLAS products there (to
//   rounding); its line fits are track_timing's (math::lstsq for LAPACK).
//   scipy's FFT convolutions (the dense MF, _lowpass) agree to rounding.
//   Every decision (the argmax cells, the runners-up, the segments kept,
//   the outlier cut, the bases tried, the runners-up of gate V, the
//   accept/reject) is the reference's on inputs whose deciding values are
//   not within rounding of a threshold.
// * refine_path_l is track::refine_path with the pilot's median band
//   (to rounding through a fitted PathFn: 1e-7 Hz is the bar).
//
// Errors are the reference's types: frame::WrongWaveform for a CE frame
// (Python's AssertionError from _l_only), std::invalid_argument for
// ValueError, std::out_of_range for IndexError, qrss::ZeroDivision,
// math::LinAlgError.
//
// Thread safety: every function is reentrant; a ChanCapture's noise table
// is filled once (its NoisePsdSlot) whichever thread asks first.

#ifndef SSTVAE_QRSS_LTRACK_TIMING_HPP
#define SSTVAE_QRSS_LTRACK_TIMING_HPP

#include <complex>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <vector>

#include "qrss/chan.hpp"
#include "qrss/frame.hpp"
#include "qrss/ltrack.hpp"
#include "qrss/rx_config.hpp"
#include "qrss/track_timing.hpp"
#include "qrss/trackbase.hpp"
#include "qrss/types.hpp"

namespace sstvae::qrss::ltrack {

// --- the receiver-time mix's phases ----------------------------------------------

// _lag_phase: exp(-j 2 pi f_p lag / CH_FS), f_p / CH_FS = 12207/242500 turn
// per CH sample, the turns reduced to [0, 1) first.
std::vector<cplx> lag_phase(std::span<const double> lag);
cplx lag_phase(double lag);

// _clock_phase: exp(-j 2 pi f_p T d pos), f_p T = 12207/16000 turn per L
// symbol, for each position under the clock error d.
std::vector<cplx> clock_phase(std::span<const double> pos, double d);

// --- the coarse timing fit ----------------------------------------------------------

// fit_timing_l's keyword arguments (track::FitTimingArgs's meaning; a real
// `known` is taken as complex, as np.asarray(kn.val, complex128) takes it).
using FitTimingLArgs = track::FitTimingArgs;

// The data band mixed up in receiver time and matched-filtered:
// oaconvolve(z_d conj(lag_phase(n - n_ref)), mf_taps(fs, L_T_SYM), "same").
std::vector<cplx> fit_mixed(std::span<const cplx> z_d, double n_ref, int fs = config::CH_FS);

// The statistic fit_timing_l correlates at one carrier estimate:
// fit_mixed(z_d, n_ref) conj(LP(z_d)), LP the _lowpass at lp_hz.
std::vector<cplx> fit_statistic_l(std::span<const cplx> z_d, double n_ref, double lp_hz, int fs = config::CH_FS);
// The same from a fit_mixed already computed.
std::vector<cplx> fit_statistic_l(std::span<const cplx> z_d, std::span<const cplx> y, double lp_hz,
                                  int fs = config::CH_FS);

// The coarse grid, row-major (deltas.size() x nlag): Z[i, j] =
// Re(lag_phase(j) sum_k g_k ((1 - f) x[n] + f x[n + 1])) / norm, g_k =
// conj(a_k) clock_phase(pos_k, deltas[i]), n + f = base + o_k + j, o_k =
// pos_k sps (1 + deltas[i]) + frac0, a term 0 unless n >= 0 and n + 1 <
// len(x) (the reference's direct path).
std::vector<double> fit_grid_l(std::span<const cplx> x, const track::Known& known, std::int64_t base,
                               std::int64_t nlag, double frac0, std::span<const double> deltas, double sps,
                               double norm);

// _segment_timing_l: track::segment_timing on L's complex x, each local lag
// with its mix phase (about n_ref) and each position with its clock phase.
track::SegmentTiming segment_timing_l(std::span<const cplx> x, double sig2, const track::Known& known, double t0abs,
                                      double delta, double sps, std::int64_t n_pos, double n_ref,
                                      double seg_s = rxconfig::ltrack::SEG_TIMING_S);

// fit_timing_l (design 6.4 steps 1-2 for L) on the derotated channel z_d.
// L only. Returns the timings strongest first: one for n_best <= 1 (the
// reference's single Timing), else up to n_best. Timing.z is the peak's
// correlation in units of its noise sd; cov is diag(0.25, (step 1e6)^2, 0).
// trace (when given) records what was computed on the way (its peaks'
// lp index into CARRIER_LP_HZ_L).
std::vector<types::Timing> fit_timing_l(std::span<const cplx> z_d, int ch_fs, const frame::FrameSpec& spec,
                                        double t0_index, const FitTimingLArgs& args = {},
                                        track::FitTimingTrace* trace = nullptr);
// The same with z_d = derotate(chan, fn).
std::vector<types::Timing> fit_timing_l(const track::ChanCapture& chan, int ch_fs, const track::PathFn& fn,
                                        const frame::FrameSpec& spec, double t0_index,
                                        const FitTimingLArgs& args = {}, track::FitTimingTrace* trace = nullptr);
// fit_timing_l(n_best = 1) and fit_timing_l(n_best = args.n_best) from one
// correlation grid, for verify_l (the reference's _fit_timing_tries_l;
// track::fit_timing_tries).
track::FitTimingTries fit_timing_tries_l(std::span<const cplx> z_d, int ch_fs, const frame::FrameSpec& spec,
                                         double t0_index, const FitTimingLArgs& args = {},
                                         track::FitTimingTrace* trace = nullptr);

// --- gate V --------------------------------------------------------------------------

// refine_path_l: track::refine_path on the pilot, the line's median taken
// from PATH_MED_LO_HZ to 4 PATH_SPAN_HZ + 10 Hz (above the pilot, clear of
// the data band below it).
track::PathResult refine_path_l(const track::ChanCapture& chan, const frame::FrameSpec& spec,
                                const types::FreqPath& path);

// resid_ratio: the known symbols' residual power over the model's, weighted
// as each symbol weighs in Z_ref; c_known, gain and dist on tr's grid (from
// data_templates with the known symbols known). 1.0 when no known symbol is
// usable. tr must carry L classes (std::invalid_argument otherwise).
double resid_ratio(const track::LTrackResult& tr, std::span<const cplx> c_known, std::span<const double> gain,
                   std::span<const double> dist);

// verify_track_l: the tracker run behind verify_l (known symbols withheld,
// no refinement, no re-centring) with tr.resid_ratio and tr.z_ref (Z_ref
// over what was heard, divided by sqrt(max(1, resid_ratio))) set. L only.
track::LTrackResult verify_track_l(std::shared_ptr<const track::ChanCapture> chan, const frame::FrameSpec& spec,
                                   const types::Detection& cand, const track::VerifyArgs& args = {});

// verify_l: gate V for an L candidate. The candidate with its measured
// path, timing and z_ref and waveform L, or nullopt (Python's None) when
// Z_ref never exceeds Z_ACCEPT. The result's a3_z is NaN (the reference
// builds a fresh Detection). L only.
std::optional<types::Detection> verify_l(std::shared_ptr<const track::ChanCapture> chan,
                                         const frame::FrameSpec& spec, const types::Detection& cand,
                                         const track::VerifyArgs& args = {});

}  // namespace sstvae::qrss::ltrack

#endif  // SSTVAE_QRSS_LTRACK_TIMING_HPP

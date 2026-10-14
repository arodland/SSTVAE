// The tracker core of sstvae/qrss/track.py (design 6.3 V, 6.4, 6.5; WP6):
// derotation, the symbol matched filter and its tables, the CE templates,
// the complex-gain Kalman filter and RTS smoother with the process-noise
// search, the tracker run itself (`track`) with its timing refinement and
// frequency re-centring, Z_ref, erasure and the pass report. The records
// (ChanCapture, PathFn, MfTable, Classes, TrackResult, ...) are
// trackbase.hpp's; the noise floor is chan.hpp's.
//
// Not here: fit_timing, _segment_timing, _parabola, verify, verify_track,
// refine_path, line_path, _carrier_lines and _flat_path are
// track_timing.hpp's; genie_track (a test aid) is not ported.
//
// Python is normative. What agrees to the bit with the reference on the
// same inputs, and what does not:
//
// * Bitwise: derotate (np.exp of a purely imaginary argument is (cos, sin)
//   of it, the complex multiplies numpy's fused ones), mf_table,
//   symbol_mf (einsum's own summation order: a complex operand sums its
//   products in order, real times real in two SSE lanes of four-product
//   blocks), psi_at, known_symbols, make_classes, remove_spread, the
//   window and position bookkeeping, _kf_forward, kalman_rts and select_q
//   (Python's float and complex arithmetic: a float times a complex is
//   CPython 3.13's complex(k, 0) times it, unfused), _measurements,
//   _pu_initial, z_ref_stat, erase.
// * To rounding: templates and anything downstream of it (np.exp of a real
//   array is numpy's SIMD exp, not libm's), _lowpass (scipy's FFT
//   convolution), _smooth (np.convolve sums through BLAS ddot), np.angle
//   (numpy's SIMD arctan2), polyfit and the linear algebra (math/linalg's
//   Jacobi and LU for LAPACK). So `track`, `refine_timing`, `_recenter`
//   and `report` agree to rounding, and every decision they take is the
//   reference's (the same q, the same Newton branch, the same breaks) on
//   any input whose deciding values are not within rounding of a
//   threshold. How far "rounding" carries: m, c, N, psi ~1e-11; u ~1e-9
//   and P up to ~1e-6 relative on real passes, because the Kalman pass
//   amplifies an ulp in the process noise Pu (the smoother's mean, a BLAS
//   sum) about 1e9-fold -- its diffuse initial velocity variance cancels
//   down to q Pu. On the inputs the C++ run measured, the reference's own
//   kalman_rts returns this run's u, P and loglik to the bit.
//
// Errors are the reference's types: frame::WrongWaveform for a CE-only
// function given an L frame ("track.X is CE's; 'name' is waveform 1: use
// Y", Python's AssertionError), std::invalid_argument for ValueError,
// std::out_of_range for IndexError, std::overflow_error for
// OverflowError, qrss::ZeroDivision for ZeroDivisionError and
// math::LinAlgError.
//
// Thread safety: every function is reentrant. mf_table's two tables are
// built once (function-local statics) and immutable; a ChanCapture's
// noise table is its thread-safe NoisePsdSlot.

#ifndef SSTVAE_QRSS_TRACK_HPP
#define SSTVAE_QRSS_TRACK_HPP

#include <complex>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

#include "qrss/ce.hpp"
#include "qrss/chan.hpp"
#include "qrss/config.hpp"
#include "qrss/frame.hpp"
#include "qrss/rx_config.hpp"
#include "qrss/trackbase.hpp"
#include "qrss/types.hpp"

namespace sstvae::qrss::track {

using cplx = std::complex<double>;

// _ce_only: frame::WrongWaveform "{what} is CE's; {spec.name!r} is waveform
// {n}: use {instead}" unless spec is a CE frame.
void ce_only(const frame::FrameSpec& spec, std::string_view what, std::string_view instead);

// --- derotation ---------------------------------------------------------------

// A frequency path evaluated at many times (audio Hz at receiver times).
using PathEval = std::function<std::vector<double>(std::span<const double>)>;

// z_d = CH * exp(-2 pi j theta), theta the integral of f(t) - f_mix from t0
// (in turns, the running sum interpolated at the fractional t0 index).
std::vector<cplx> derotate(const ChanCapture& chan, const PathFn& fn);
std::vector<cplx> derotate(const ChanCapture& chan, const PathEval& fn);
// The same over samples `ch` held at double precision (what
// chan.ch.astype(complex128) gives for a capture whose CH is not complex64),
// on grid's fs, t0_index and f_mix (grid.ch is not read).
std::vector<cplx> derotate(std::span<const cplx> ch, const ChanCapture& grid, const PathFn& fn);
std::vector<cplx> derotate(std::span<const cplx> ch, const ChanCapture& grid, const PathEval& fn);

// --- the symbol matched filter --------------------------------------------------

// _pulse_at: the unit-energy RRC of the waveform with period t_sym (CE's
// T_SYM or L's L_T_SYM) at x symbols; std::invalid_argument "no symbol
// pulse with period {t_sym!r} s" for any other.
double pulse_at(double x, double t_sym);
std::vector<double> pulse_at(std::span<const double> x, double t_sym);
// _pulse_g0: CE's G0 at T_SYM, else L's.
double pulse_g0(double t_sym);

// mf_table: the symbol MF at period t_sym (sps = t_sym CH_FS, half =
// ceil(SPAN sps) + 1, taps every 1/MF_RES CH sample). Built once per
// waveform. Any other t_sym: std::invalid_argument "no symbol pulse with
// period ..." (or Python's int() errors for a non-finite SPAN * sps).
const MfTable& mf_table(double t_sym = config::T_SYM);

// symbol_mf: m_p = (1/sps) sum_n z(n) tab(n - t_p), t_p = t0_index +
// rx_index(timing, p) at the table's period; `table` null is CE's
// mf_table(). Samples outside z count zero; a non-finite t_p gives NaN.
// An empty z with any position is std::out_of_range ("index -1 is out of
// bounds for axis 0 with size 0").
std::vector<cplx> symbol_mf(std::span<const cplx> z, const types::Timing& timing, std::span<const double> pos,
                            double t0_index, std::int64_t n_pos, const MfTable* table = nullptr);
std::vector<cplx> symbol_mf(std::span<const double> z, const types::Timing& timing, std::span<const double> pos,
                            double t0_index, std::int64_t n_pos, const MfTable* table = nullptr);

// psi_at: MF{keep}(t_p)/g0 clipped to [0, 1] (nullopt: no blanker, 1).
std::vector<double> psi_at(std::optional<std::span<const double>> keep, const types::Timing& timing,
                           std::span<const double> pos, double t0_index, std::int64_t n_pos,
                           const MfTable* table = nullptr);
// The same over a capture's own keep (float32, read as float64).
std::vector<double> psi_at(const ChanCapture& chan, const types::Timing& timing, std::span<const double> pos,
                           std::int64_t n_pos, const MfTable* table = nullptr);

// _lowpass: zero-phase Hann-windowed-sinc low-pass of cutoff hz (n =
// int(round(4 fs / hz)) | 1 taps), through scipy's oaconvolve "same".
std::vector<cplx> lowpass(std::span<const cplx> x, double hz, double fs = config::CH_FS);
std::vector<double> lowpass(std::span<const double> x, double hz, double fs = config::CH_FS);
// The taps _lowpass applies.
std::vector<double> lowpass_taps(double hz, double fs = config::CH_FS);

// _mf_taps: the symbol MF at integer samples of rate fs.
std::vector<double> mf_taps(double fs = config::CH_FS, double t_sym = config::T_SYM);

// --- known symbols and classes ---------------------------------------------------

// known_symbols: preamble, references and spare (+ the header bits once
// decoded: their 1 - 2 b, as make_classes' mu holds them -- so a header
// make_classes refuses, one of the wrong length say, is its error here
// too). CE only.
Known known_symbols(const frame::FrameSpec& spec, const frame::HdrBits* hdr_bits = nullptr);

// make_classes: round A (hdr_bits null) or round B classes, `prior` the
// data's EM prior. CE only. Assignments broadcast as numpy's do (a value
// array of one element fills every slot; another length is
// std::invalid_argument "shape mismatch: ...").
Classes make_classes(const frame::FrameSpec& spec, const frame::HdrBits* hdr_bits = nullptr,
                     const types::EmPrior* prior = nullptr);

// remove_spread: z * exp(-j phi_h), the known spread header's phase taken
// off at the sender time each CH sample was sent (t0_index the channel's).
// CE only; `spread` null returns z unchanged (as complex128).
std::vector<cplx> remove_spread(std::span<const cplx> z, double t0_index, const frame::FrameSpec& spec,
                                const types::Timing& timing, const std::vector<double>* spread);

// templates: (c, sf) at integer positions grid_pos: c = MF{E[s]} and sf =
// max(MF{a^2 (1 - e^-v)}/G0, 0). `keying` a valid keying
// (ce::window_keying) or null; the keyed span starts at keyed_from_pos
// (nullopt: spec.keyed_start_pos). CE only. An empty grid is
// std::out_of_range.
struct Templates {
    std::vector<cplx> c;
    std::vector<double> sf;
};
Templates templates(const frame::FrameSpec& spec, std::span<const std::int64_t> grid_pos, std::span<const double> mu,
                    std::span<const double> nu, const ce::Keying* keying = nullptr, bool ook = false,
                    std::optional<double> keyed_from_pos = std::nullopt);

// --- Kalman filter and RTS smoother ------------------------------------------------

// _kf_forward with store=True: the forward pass and its predictions.
struct KfPass {
    double loglik = 0.0;
    std::vector<cplx> xa, xb;           // filtered state
    std::vector<double> pa, pb, pc;     // filtered covariance
    std::vector<cplx> fx1, fx2;         // predicted state
    std::vector<double> fa, fb, fc;     // predicted covariance
};
// _kf_forward(store=False): the innovation log-likelihood alone.
double kf_forward(std::span<const cplx> mu, std::span<const double> R, double q, double Pu);
KfPass kf_forward_pass(std::span<const cplx> mu, std::span<const double> R, double q, double Pu);

// kalman_rts: (u_hat, P, loglik); causal returns the forward filter.
struct KalmanResult {
    std::vector<cplx> u;
    std::vector<double> P;
    double loglik = 0.0;
};
KalmanResult kalman_rts(std::span<const cplx> mu, std::span<const double> R, double q, double Pu,
                        bool causal = false);

// select_q: (q, loglik) of greatest innovation likelihood over `grid`
// (the first of equals); nullopt for an empty grid (Python's None).
std::optional<std::pair<double, double>> select_q(std::span<const cplx> mu, std::span<const double> R, double Pu,
                                                  std::span<const double> grid = Q_GRID);

// --- the tracker's pieces ------------------------------------------------------------

// _window_unknown: positions whose filter support reaches window units
// that may be keyed (0/1 per position).
std::vector<std::uint8_t> window_unknown(const frame::FrameSpec& spec, std::span<const std::int64_t> grid_pos,
                                         bool ook_possible = true, std::int64_t reach = config::SPAN);
std::vector<std::uint8_t> window_unknown(const frame::FrameSpec& spec, std::span<const double> grid_pos,
                                         bool ook_possible = true, double reach = config::SPAN);

// _smooth: a length-n moving average, edge-padded (n <= 1: a copy).
std::vector<double> smooth(std::span<const double> x, std::int64_t n);

// _measurements: (mu, R) for the Kalman filter from tr's m, c (or `c`),
// psi, N, meas (or `meas`).
struct Measurements {
    std::vector<cplx> mu;
    std::vector<double> R;
};
Measurements measurements(const TrackResult& tr, std::span<const double> kappa, std::span<const double> sf,
                          std::span<const double> pu_loc, const std::vector<cplx>* c = nullptr,
                          const std::vector<std::uint8_t>* meas = nullptr);

// _pu_initial: the first mean-power estimate.
double pu_initial(const TrackResult& tr);

// _recenter: the frequency u_hat still rotates at (30 s windows) added to
// the path, and the largest |df|. fn nullopt: no window qualified, the
// path is tr.freq unchanged (and df_max 0).
struct Recenter {
    std::optional<PathFn> fn;
    double df_max = 0.0;
};
Recenter recenter(const TrackResult& tr);

// _lead_supported: whether the claimed lead-in carrier is there.
bool lead_supported(const ChanCapture& chan, std::span<const cplx> z_d, const types::Timing& tm,
                    const frame::FrameSpec& spec, double lead, const NoisePsdFn& noise_psd = {});

// --- the tracker -----------------------------------------------------------------------

// track()'s keyword arguments. classes nullopt is round A
// (make_classes(spec, prior=prior)); keying/ook are cw_known (keying a
// valid keying, nullopt with ook false is cw_known None); timing and freq
// nullopt are det's; lead_in_s nullopt is det.lead_in_s; z_d null is
// derotate(chan, freq). noise_psd computes the capture's noise table if
// its slot is empty (empty: the C++ frontend.noise_psd).
struct TrackArgs {
    std::optional<Classes> classes;
    std::optional<ce::Keying> keying;
    bool ook = false;
    std::optional<types::EmPrior> prior;
    std::optional<types::Timing> timing;
    std::optional<PathFn> freq;
    bool withhold_known = false;
    int outer = 2;
    bool refine = true;
    std::optional<double> lead_in_s;
    std::shared_ptr<const std::vector<cplx>> z_d;
    NoisePsdFn noise_psd;
};

// track: one tracker run over a pass (design 6.5). CE only. No timing
// (neither args.timing nor det.timing) is std::invalid_argument.
TrackResult track(std::shared_ptr<const ChanCapture> chan, const frame::FrameSpec& spec,
                  const types::Detection& det, const TrackArgs& args = {});

// refine_timing: Newton steps on the known symbols' likelihood given u_hat
// (u_ref/P_ref, default tr's own); nullopt is Python's None. CE only.
std::optional<types::Timing> refine_timing(const TrackResult& tr, int iters = 3,
                                           const std::vector<cplx>* u_ref = nullptr,
                                           const std::vector<double>* P_ref = nullptr);

// --- the verification gate and the report ---------------------------------------------

// _layout_of: the frame's layout in either waveform.
std::shared_ptr<const frame::Layout> layout_of(const frame::FrameSpec& spec);

// z_ref_stat over tr's known positions (tr's classes' known, either
// waveform): d_known nullopt is CE's d = j Im(c_known).
double z_ref_stat(const TrackResult& tr, std::span<const cplx> c_known,
                  std::optional<std::span<const cplx>> d_known = std::nullopt);

// erase: psi = 0 for tr's symbols within T of any span (a, b), seconds
// after t0. Nothing changes for no spans.
void erase(TrackResult& tr, std::span<const std::pair<double, double>> spans);

// report: the pass's TrackReport with the capture's noise floor given
// (what tr.chan.noise_floor() returns), or computed from tr.chan.
types::TrackReport report(const TrackResult& tr, double kappa, bool suspect, double noise_floor);
types::TrackReport report(const TrackResult& tr, double kappa = 1.0, bool suspect = false,
                          const NoisePsdFn& noise_psd = {});

}  // namespace sstvae::qrss::track

#endif  // SSTVAE_QRSS_TRACK_HPP

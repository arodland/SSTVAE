// The waveform L tracker of sstvae/qrss/ltrack.py (docs/qrss/waveform-l.md):
// the pilot filter, the data-band mix and matched filter, L's classes and
// templates, the pilot's template, the two-stream gain model (`_Model`),
// the tracker run itself (`track_l`) with delta's fit, the lead-in check
// and the timing refinement (`refine_timing_l`), the report and the known
// symbols. The records (ChanCapture, PathFn, MfTable, LClasses,
// LTrackResult, ...) are trackbase.hpp's; the Kalman filter, select_q,
// derotate, symbol_mf, psi_at, smooth and recenter are track.hpp's (what
// ltrack.py imports from track.py).
//
// The coarse timing and gate V (fit_timing_l, _segment_timing_l,
// _lag_phase, _clock_phase, refine_path_l, verify_track_l, resid_ratio,
// verify_l) are ltrack_timing.hpp's; genie_track_l (a test aid) is not
// ported.
//
// Python is normative. What agrees to the bit with the reference on the
// same inputs, and what does not:
//
// * Bitwise: pilot_table and its taps (ce::rrc, numpy's pairwise sums),
//   data_mix (np.exp of a purely imaginary argument is (cos, sin) of it,
//   the multiply numpy's fused one), data_mf (track::symbol_mf on the
//   mixed samples, the dense and sparse paths filling the same samples),
//   make_classes_l, _distortion, data_templates, _pilot_gain, pilot_wave
//   (lwave::pilot and id_weight are bitwise; a float times a complex array
//   promotes the float, and the division by A_ID G_REF_ABS is numpy's
//   complex division by (d, 0): a multiply by its reciprocal), _q_near,
//   _Model and its streams/fused/pu_initial, _hold_ends, _fit_delta
//   (CPython 3.13's complex arithmetic: a float is complex(f, 0), products
//   unfused, quotients by Smith's method dividing by the denominator, abs
//   is hypot), known_symbols_l, report_l's correction.
// * To rounding: pilot_filter (scipy's oaconvolve through ducc0 against
//   pocketfft here, ~1e-16 of the output's scale), _pu_local (np.convolve
//   in track's _smooth sums through BLAS), the linear algebra of
//   refine_timing_l (math/linalg's Jacobi and LU for LAPACK), and so
//   _lead_supported_l, track_l, refine_timing_l and report_l. Every
//   decision they take is the reference's on any input whose deciding
//   values are not within rounding of a threshold; with the reference's
//   own pilot_filter, _smooth, _recenter and refine_timing_l substituted
//   for its leaves, track_l is the reference to the bit.
//
// Errors are the reference's types: frame::WrongWaveform for a CE frame
// ("ltrack.X is L's; 'name' is waveform 0: use track's", Python's
// AssertionError), std::invalid_argument for ValueError, std::out_of_range
// for IndexError, std::overflow_error for OverflowError, qrss::ZeroDivision
// for ZeroDivisionError and math::LinAlgError.
//
// Thread safety: every function is reentrant. pilot_table is built once (a
// function-local static) and immutable; a ChanCapture's noise table is its
// thread-safe NoisePsdSlot.

#ifndef SSTVAE_QRSS_LTRACK_HPP
#define SSTVAE_QRSS_LTRACK_HPP

#include <complex>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

#include "qrss/chan.hpp"
#include "qrss/qrss_config.hpp"
#include "qrss/frame.hpp"
#include "qrss/lwave.hpp"
#include "qrss/rx_config.hpp"
#include "qrss/trackbase.hpp"
#include "qrss/types.hpp"

namespace sstvae::qrss::ltrack {

using cplx = std::complex<double>;

// _l_only: frame::WrongWaveform "{what} is L's; {spec.name!r} is waveform
// {n}: use track's" unless spec is an L frame.
void l_only(const frame::FrameSpec& spec, std::string_view what);

// --- the filters -------------------------------------------------------------

// pilot_table: the unit-energy RRC (alpha 0.15, +-8 of its symbols) at
// PILOT_K T_L, read at positions in L symbols (t_sym L_T_SYM, sps 30.3125,
// half 244); g0 its output for a constant 1. Built once.
const track::MfTable& pilot_table();

// _pilot_taps: the pilot filter at integer CH samples over its sps (2 half
// + 1 taps).
std::vector<double> pilot_taps();

// pilot_filter: the pilot filter's output of x (CH samples) at fractional
// indices tp: oaconvolve(x, taps, "same") read by linear interpolation,
// samples outside x counting zero. An empty x with any tp is
// std::out_of_range (numpy's IndexError on y[-1]).
std::vector<cplx> pilot_filter(std::span<const cplx> x, std::span<const double> tp);
std::vector<double> pilot_filter(std::span<const double> x, std::span<const double> tp);

// data_mix: z_d exp(+j 2 pi f_p T tau(n)) at CH indices idx (nullopt: all
// of z_d), tau the sender time of n under `timing` (t0_index the capture's).
// An index outside z_d is std::out_of_range (negative ones wrap, as numpy's).
std::vector<cplx> data_mix(std::span<const cplx> z_d, double t0_index, const types::Timing& timing,
                           std::int64_t n_pos, std::optional<std::span<const std::int64_t>> idx = std::nullopt);

// data_mf: the data-band matched filter at positions `pos` (L's symbol MF on
// z_d re-mixed for this timing). A non-finite read-out time is Python's
// error from math.floor/ceil (std::invalid_argument for NaN,
// std::overflow_error for infinity).
std::vector<cplx> data_mf(std::span<const cplx> z_d, double t0_index, const types::Timing& timing,
                          std::span<const double> pos, std::int64_t n_pos);
// data_mf's reach: the [lo, hi) of a z_d of n_z samples its filters read at
// `pos` (nullopt for no positions, or hi <= lo, where data_mf gives zeros);
// a non-finite read-out time refused as data_mf refuses it. data_mf mixes
// only the samples the filters reach when len(pos) (2 half + 1) < hi - lo.
std::optional<std::pair<std::int64_t, std::int64_t>> data_mf_reach(std::size_t n_z, double t0_index,
                                                                   const types::Timing& timing,
                                                                   std::span<const double> pos, std::int64_t n_pos);

// --- classes and templates ---------------------------------------------------------

// make_classes_l: round A (hdr_bits null) or round B classes; `prior` an
// EmPrior (x_hat, v per latent) for the data. hdr_bits are the coded bits
// as uint8 (np.asarray(hdr_bits, dtype=np.uint8), 1-D); an odd count is
// sequences::qpsk's std::invalid_argument when the frame has a header.
track::LClasses make_classes_l(const frame::FrameSpec& spec, const std::vector<std::uint8_t>* hdr_bits = nullptr,
                               const types::EmPrior* prior = nullptr);

// _distortion: float64[n_sym], the clipper's distortion at the data MF per
// symbol (D_PASS_L, D_HDR_L at the header, D_REF_L at the known symbols).
std::vector<double> distortion(const frame::FrameSpec& spec);

// data_templates: (c, nu, gain, dist, is_sym) on the grid. An empty grid is
// std::out_of_range.
struct DataTemplates {
    std::vector<cplx> c;
    std::vector<double> nu, gain, dist;
    std::vector<std::uint8_t> is_sym;
};
DataTemplates data_templates(const frame::FrameSpec& spec, std::span<const std::int64_t> grid,
                             const track::LClasses& classes, bool withhold_known = false);

// _pilot_gain: the clipper's gain on the pilot at sender time tau.
std::vector<double> pilot_gain(std::span<const double> tau, const frame::FrameSpec& spec, double lead_in_s = 0.0);

// pilot_wave: the pilot on u's scale, _pilot_gain lwave.pilot / (A_ID
// G_REF_ABS). `keying` as lwave::pilot takes it (null: unknown).
std::vector<cplx> pilot_wave(std::span<const double> tau, const frame::FrameSpec& spec,
                             const lwave::Keying* keying = nullptr, double lead_in_s = 0.0);

// _q_near: the grid's points within two of q's (by |log g - log q|). q <= 0
// is std::invalid_argument "math domain error".
std::vector<double> q_near(double q, std::span<const double> grid);
// The [lo, hi) of grid that q_near returns.
std::pair<std::size_t, std::size_t> q_near_span(double q, std::span<const double> grid);

// --- the tracker's pieces ------------------------------------------------------------

// _Model: what one tracker run holds between measurements (templates,
// masks, the pilot's noise ratio k_pl), on tr.pos. The constructor sets
// tr.meas and tr.meas_pl, as the reference's does; the methods read tr's
// measurements (m, psi, N; m_pl, c_pl, psi_pl, N_pl) when called. Masks are
// 0/1 per grid position.
struct Streams {
    std::vector<cplx> mu_d;
    std::vector<double> R_d;
    std::vector<cplx> mu_p;
    std::vector<double> R_p;
};
struct Fused {
    std::vector<cplx> mu;
    std::vector<double> R;
};
class Model {
public:
    Model(track::LTrackResult& tr, const frame::FrameSpec& spec, const track::LClasses& classes, bool withhold_known,
          const lwave::Keying* keying, double lead, std::int64_t n_lead);
    // The arrays, filled by hand (the binding's _Model reads them back).
    Model() = default;

    // streams: (mu, R) of each stream per position (R inf where none, the
    // pilot's divided by delta), no floor yet; a stream turned off is all
    // (0, inf). data_mask (0/1) narrows the data stream.
    Streams streams(const track::LTrackResult& tr, std::span<const double> pu_loc, cplx delta = {1.0, 0.0},
                    bool data = true, bool pilot = true, const std::vector<std::uint8_t>* data_mask = nullptr) const;
    // fused: the streams by inverse variance plus the model-error floor.
    Fused fused(const track::LTrackResult& tr, std::span<const double> pu_loc, cplx delta = {1.0, 0.0},
                bool data = true, bool pilot = true, const std::vector<std::uint8_t>* data_mask = nullptr) const;
    // pu_initial: the first mean-power estimate.
    double pu_initial(const track::LTrackResult& tr) const;

    std::vector<cplx> c_all, c;
    std::vector<double> nu, gain, dist, k_pl;
    std::vector<std::uint8_t> is_sym, is_known, meas_d, need, meas_p, measured;
};

// _pu_local: (Pu_loc, Pu_g), the local signal power and its mean over the
// positions that measure anything.
struct PuLocal {
    std::vector<double> loc;
    double g = 0.0;
};
PuLocal pu_local(std::span<const cplx> u, std::span<const double> P, std::span<const std::uint8_t> measured);

// _hold_ends: u and P held at their first and last measured (finite R)
// values outside them; false (nothing changed) when no R is finite.
bool hold_ends(std::vector<cplx>& u, std::vector<double>& P, std::span<const double> R);

// _lead_supported_l: whether a claimed lead-in's pilot is there. noise_psd
// computes the capture's noise table if its slot is empty.
bool lead_supported_l(const track::ChanCapture& chan, std::span<const cplx> z_d, const types::Timing& tm,
                      const frame::FrameSpec& spec, double lead, const track::NoisePsdFn& noise_psd = {});

// _fit_delta: delta from the data band's measurements against a pilot-only
// track v (of u delta), shrunk to 1 by DELTA_PRIOR_SD. qrss::ZeroDivision
// for a zero delta0 (Python's complex division by zero).
cplx fit_delta(const Model& model, const track::LTrackResult& tr, std::span<const cplx> v, std::span<const double> Pv,
               std::span<const double> pu_loc, cplx delta0 = {1.0, 0.0});

// --- the tracker -----------------------------------------------------------------------

// track_l()'s keyword arguments. classes nullopt is round A
// (make_classes_l(spec, prior=prior)); keying/ook are cw_known (keying as
// np.asarray(k, dtype=np.uint8) gives it, nullopt with ook false is
// cw_known None or (None, False); ook true is refused); timing and freq
// nullopt are det's; lead_in_s nullopt is det.lead_in_s; z_d null is
// derotate(chan, freq). noise_psd computes the capture's noise table if its
// slot is empty (empty: the C++ frontend.noise_psd).
struct TrackLArgs {
    std::optional<track::LClasses> classes;
    std::optional<lwave::Keying> keying;
    bool ook = false;
    std::optional<types::EmPrior> prior;
    std::optional<types::Timing> timing;
    std::optional<track::PathFn> freq;
    bool withhold_known = false;
    int outer = 2;
    bool refine = true;
    std::optional<double> lead_in_s;
    std::shared_ptr<const std::vector<cplx>> z_d;
    track::NoisePsdFn noise_psd;
};

// track_l: one L tracker run over a pass. L only. No timing (neither
// args.timing nor det.timing) is std::invalid_argument "track_l needs a
// timing (fit_timing_l, or gate V's)"; ook keying is std::invalid_argument
// "waveform L keys its callsign by FSK only: cw_known must say ook=False".
track::LTrackResult track_l(std::shared_ptr<const track::ChanCapture> chan, const frame::FrameSpec& spec,
                            const types::Detection& det, const TrackLArgs& args = {});

// refine_timing_l: Newton steps on the known symbols' likelihood given a
// gain track (u_ref/P_ref, default tr.u_pl/P_pl, else tr.u/P), delta's
// phase profiled out; nullopt is Python's None. L only.
std::optional<types::Timing> refine_timing_l(const track::LTrackResult& tr, int iters = 3,
                                             const std::vector<cplx>* u_ref = nullptr,
                                             const std::vector<double>* P_ref = nullptr);

// --- the report and the known symbols -----------------------------------------------------

// report_l: track::report with SNR2500 on the frame's keyed power
// (L_KEYED_POWER by the frame's name, 1.0 for one not in the table), the
// capture's noise floor given (what tr.chan.noise_floor() returns) or
// computed from tr.chan.
types::TrackReport report_l(const track::LTrackResult& tr, double kappa, bool suspect, double noise_floor);
types::TrackReport report_l(const track::LTrackResult& tr, double kappa = 1.0, bool suspect = false,
                            const track::NoisePsdFn& noise_psd = {});
// The correction report_l adds to track::report's snr2500_db for a frame.
double keyed_power_db(const frame::FrameSpec& spec);

// known_symbols_l: preamble and references (+ the header symbols once
// decoded), complex. hdr_bits as in make_classes_l.
track::Known known_symbols_l(const frame::FrameSpec& spec, const std::vector<std::uint8_t>* hdr_bits = nullptr);

}  // namespace sstvae::qrss::ltrack

#endif  // SSTVAE_QRSS_LTRACK_HPP

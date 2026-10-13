// The QRSSTVAE receiver's acquisition stages A1 (carrier lines) and A2
// (preamble search), and the lead-in measurement: sstvae/qrss/acquire.py
// (the reference) from `CarrierStats` to `lead_in_span`, on the shared
// statistics of acquire_stats.hpp. A3 (track-before-detect) and the
// arbitration (`acquire`, `merge_hits`, ...) are not here.
//
//   acquire.py                    here
//   CarrierStats                  CarrierStats
//   carrier_line_stats            carrier_line_stats
//   carrier_lines                 carrier_lines
//   hint_line, _cell_line         hint_line, cell_line
//   pilot_lines                   pilot_lines
//   _notch_taps, _mf_taps,        notch_taps, mf_taps, l_mf_taps
//     _l_mf_taps                    (CH_FS tables built once)
//   PreambleKind, CE_PRE, L_PRE,  PreambleKind, ce_pre(), l_pre(),
//     PREAMBLE_KINDS                preamble_kind(waveform)
//   PreambleHit (.to_detection)   PreambleHit (.to_detection)
//   _A2Input, _A2Channel          A2Input, A2Channel
//   preamble_channel, _a2_input,  preamble_channel, a2_input, a2_mf
//     _a2_mf
//   _chunks, _norm, _templates    chunks, norm, templates
//   preamble_lambda,              preamble_lambda, preamble_sums
//     _preamble_sums
//   _fine_sums, _coherent,        fine_sums, coherent, phase_line
//     _phase_line
//   preamble_search               preamble_search
//   lead_in_span                  lead_in_span
//
// How close to the reference, and why:
//
// * A1 is bit-exact: the float32 spectrogram, medians and statistic sums
//   are numpy's float32 arithmetic in numpy's order (acquire_stats.hpp),
//   and the vertex parabola takes numpy's float32 log, which is not libm's
//   (math/nplog.hpp). One exception: carrier_lines walks the bins in
//   np.argsort(best)[::-1] order, and numpy's argsort is not stable, so
//   two bins with *equal* float32 statistics above threshold may be taken
//   in the other order here (ties are broken by the higher bin first).
// * A2's arrays agree to rounding, not to the bit, in three places: the
//   chunk phases go through np.angle, which numpy computes with its own
//   SIMD arctan2 (std::atan2 here: 1 ulp apart in ~7% of arguments);
//   oaconvolve's FFTs (math/convolve.hpp); and the chunk sums S, which
//   the reference's einsum("fcs,tcs->ftc", optimize=True) hands, in numpy
//   2.5, to bmm_einsum -- a batched matmul, i.e. the BLAS's zgemm once per
//   chunk, whose blocking and fused multiply-adds are the BLAS's own and
//   not reproducible here (nor across BLAS builds and CPUs); S is the same
//   sum taken symbol by symbol, ~2e-15 relative from it. Everything else
//   is numpy's arithmetic in numpy's order: the fused complex multiply of
//   arrays, the plain one of numpy scalars, numpy's complex exp of a
//   (0, y) argument (cos y, sin y) and its pairwise sums. preamble_p runs
//   in the order numpy adds the Fortran-ordered S that _preamble_sums
//   builds (SumOrder::Sequential), so on the same S its p is the
//   reference's to the bit.
//
// Errors are the reference's: std::invalid_argument for ValueError,
// std::out_of_range for IndexError, std::overflow_error for OverflowError,
// qrss::ZeroDivision for ZeroDivisionError, math::TypeError for TypeError.
// Thread-safe: the tap tables and the two preamble kinds are built once
// (function-local statics) and never written again; everything else is
// pure.

#ifndef SSTVAE_QRSS_ACQUIRE_A12_HPP
#define SSTVAE_QRSS_ACQUIRE_A12_HPP

#include <array>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "math/cubic_spline.hpp"
#include "qrss/acquire_stats.hpp"
#include "qrss/config.hpp"
#include "qrss/rational.hpp"
#include "qrss/rx_config.hpp"
#include "qrss/types.hpp"

namespace sstvae::qrss::acquire {

using cf = std::complex<float>;
using cd = std::complex<double>;

// --- A1 ---------------------------------------------------------------------

// A1's statistic: stat(d, b) sums M normalised frames along drift d through
// bin b, b the bin of the frequency at t0.
struct CarrierStats {
    Grid32 stat;                 // float32[n_drift, n_bin]
    std::vector<double> f_hz;    // audio Hz of each bin, at t0
    std::vector<double> drifts;  // Hz/min
    std::int64_t M = 0;
};

// carrier_line_stats(fe, t0_index, fs, span_s=, drifts=, band_hz=). fe is
// complex64 (the front end) or complex128.
CarrierStats carrier_line_stats(std::span<const cf> fe, double t0_index, double fs = config::FE_FS,
                                std::array<double, 2> span_s = rxa::A1_SPAN_S,
                                std::span<const double> drifts = rxa::A1_DRIFTS,
                                std::array<double, 2> band_hz = rxa::BAND_HZ);
CarrierStats carrier_line_stats(std::span<const cd> fe, double t0_index, double fs = config::FE_FS,
                                std::array<double, 2> span_s = rxa::A1_SPAN_S,
                                std::span<const double> drifts = rxa::A1_DRIFTS,
                                std::array<double, 2> band_hz = rxa::BAND_HZ);

// carrier_lines(..., stats=cs): up to max_cands lines over the Gamma CFAR
// threshold, strongest first, at least min_sep_hz apart.
std::vector<types::CarrierCandidate> carrier_lines(const CarrierStats& cs, double fs = config::FE_FS,
                                                   double pfa = rxa::A1_PFA,
                                                   std::int64_t max_cands = rxa::A1_MAX_CANDS,
                                                   double min_sep_hz = rxa::A1_MIN_SEP_HZ);

// hint_line(cs, f_hz, fs, pfa=, span_hz=). `refined`, when given, says
// whether the line came from a cell (`cell_line`, whose f_hz the reference
// computes as an np.float64) rather than the hint itself (a float).
types::CarrierCandidate hint_line(const CarrierStats& cs, double f_hz, double fs = config::FE_FS,
                                  double pfa = rxa::A1_PFA, double span_hz = rxa::HINT_SPAN_HZ,
                                  bool* refined = nullptr);

// _cell_line(cs, b, d, v). b and d index as numpy does (negative from the
// end; std::out_of_range past either end).
types::CarrierCandidate cell_line(const CarrierStats& cs, std::int64_t b, std::int64_t d, double v);

// pilot_lines(cs, lines, sep_hz=): only each line's f_hz is read.
std::vector<types::CarrierCandidate> pilot_lines(const CarrierStats& cs,
                                                 std::span<const types::CarrierCandidate> lines,
                                                 double sep_hz = rxa::A2_ALIAS_HZ);

// --- A2: filters and preamble kinds -------------------------------------------

// signal.firwin(NOTCH_TAPS, NOTCH_HZ, fs=CH_FS, window="hann").
const std::vector<double>& notch_taps();
// _mf_taps(fs): p(k/(fs T))/(fs T), |k| <= int(8 fs T). The CH_FS table is
// built once; another fs is computed on each call.
std::vector<double> mf_taps(double fs = config::CH_FS);
// _l_mf_taps(fs): L's unit-energy RRC at sps = fs T_L, cut at +-SPAN symbols.
std::vector<double> l_mf_taps(double fs = config::CH_FS);

// What A2 needs to know about one waveform's preamble (acquire.PreambleKind).
// seq is complex; `real` says the reference's seq is a real (float64) array,
// as CE's +-1 is, which changes how numpy promotes it in a product.
struct PreambleKind {
    std::int64_t waveform = config::WAVEFORM_CE;
    std::vector<cd> seq;
    bool real = false;
    double t_sym = config::T_SYM;
    std::int64_t chunk = rxa::A2_CHUNK;
    double data_offset_hz = 0.0;
    double phase_offset = 0.0;
    std::vector<double> mf_taps;

    // len(seq) // chunk (ZeroDivision for chunk 0).
    std::int64_t n_chunks() const;
    // WAVEFORM_NAMES[waveform] (std::out_of_range "<w>" otherwise: KeyError).
    std::string name() const;
};

const PreambleKind& ce_pre();
const PreambleKind& l_pre();
// PREAMBLE_KINDS[waveform]; std::out_of_range naming the waveform otherwise.
const PreambleKind& preamble_kind(std::int64_t waveform);

// --- A2: records --------------------------------------------------------------

struct PreambleHit {
    double f_hz = 0.0;
    double drift_hz_per_min = 0.0;
    double tau0_s = 0.0;
    double phase = 0.0;
    double lam = 0.0;
    double cell_lam = 0.0;
    double p_value = 0.0;
    bool passed = false;
    std::int64_t n_groups = config::N_PRE / rxa::A2_CHUNK;
    std::optional<Rational> f_mix;                    // None
    double tau0_sd_s = std::numeric_limits<double>::quiet_NaN();
    std::optional<std::vector<cd>> chunk_sums;        // None
    double lead_in_s = 0.0;
    std::int64_t waveform = config::WAVEFORM_CE;

    // PreambleHit.to_detection(duration_s).
    types::Detection to_detection(double duration_s = 1800.0) const;
};

// The channel at one line, drift and mixing residual removed, notched.
struct A2Input {
    std::vector<cd> y;
    double t0_ch = 0.0;
    Rational f_mix;
};

// The notched, matched-filtered channel A2 searches. local is the
// reference's `local` (None when absent).
struct A2Channel {
    std::vector<cd> m;
    double t0_ch = 0.0;
    double sigma2 = 0.0;
    Rational f_mix;
    std::optional<std::vector<double>> local;
    PreambleKind kind;
};

// --- A2: the channel ----------------------------------------------------------

// _a2_input(fe, t0_index, f_audio_hz, drift_hz_per_min, fs). fs is used as
// the reference uses it: int(fs) for the decimation and the channeliser,
// the value itself in the span arithmetic.
A2Input a2_input(std::span<const cf> fe, double t0_index, double f_audio_hz, double drift_hz_per_min = 0.0,
                 double fs = config::FE_FS);
A2Input a2_input(std::span<const cd> fe, double t0_index, double f_audio_hz, double drift_hz_per_min = 0.0,
                 double fs = config::FE_FS);

// _a2_mf(inp, kind).
A2Channel a2_mf(const A2Input& inp, const PreambleKind& kind = ce_pre());

// preamble_channel(fe, t0_index, f_audio_hz, drift, fs, kind=).
A2Channel preamble_channel(std::span<const cf> fe, double t0_index, double f_audio_hz,
                           double drift_hz_per_min = 0.0, double fs = config::FE_FS,
                           const PreambleKind& kind = ce_pre());
A2Channel preamble_channel(std::span<const cd> fe, double t0_index, double f_audio_hz,
                           double drift_hz_per_min = 0.0, double fs = config::FE_FS,
                           const PreambleKind& kind = ce_pre());

// --- A2: templates and sums ---------------------------------------------------

// _chunks(kind): zero-mean chunk templates b (nc x ck, row-major); real
// when kind.real (imaginary parts 0).
struct Chunks {
    std::size_t nc = 0, ck = 0;
    std::vector<cd> b;
    bool real = false;
};
Chunks chunks(const PreambleKind& kind);

// _norm(b): sum |b|^2 per chunk, numpy's b * conj(b) and pairwise sum.
std::vector<double> norm(const Chunks& b);

// _templates(df_hz, kind): (b, W[n_df, nc, ck] row-major, norm).
struct Templates {
    Chunks b;
    std::vector<cd> W;
    std::vector<double> norm;
};
Templates templates(std::span<const double> df_hz = rxa::A2_DF_HZ, const PreambleKind& kind = ce_pre());

// _preamble_sums(chan, df_hz, tau_s): Lambda[n_df, n_tau], the tau offsets
// in CH samples from t0, S[n_df, n_tau, nc] and var[n_tau, nc], all
// row-major here (the reference's S and Lambda are Fortran-ordered; the
// binding hands them back that way).
struct Sums {
    std::size_t n_df = 0, n_tau = 0, nc = 0;
    std::vector<double> lam;
    std::vector<double> taus;
    std::vector<cd> S;
    std::vector<double> var;
};
Sums preamble_sums(const A2Channel& chan, std::span<const double> df_hz = rxa::A2_DF_HZ,
                   double tau_s = rxa::A2_TAU_S);
// preamble_lambda: the same (var left empty).
Sums preamble_lambda(const A2Channel& chan, std::span<const double> df_hz = rxa::A2_DF_HZ,
                     double tau_s = rxa::A2_TAU_S);

// A callable spline over the MF output (CubicSpline's __call__ on a vector).
using SplineFn = std::function<std::vector<cd>(std::span<const double>)>;
SplineFn spline_fn(const math::CubicSpline& spline);

// _fine_sums(spline, chan, tau, df, rate): (S_c, norm).
std::pair<std::vector<cd>, std::vector<double>> fine_sums(const SplineFn& spline, const A2Channel& chan, double tau,
                                                          double df, double rate = 0.0);
// _coherent(spline, chan, tau, df, rate).
double coherent(const SplineFn& spline, const A2Channel& chan, double tau, double df, double rate);
// _phase_line(S, dt): (a, b, sd_b). `fitted`, when given, says whether the
// line was fitted (the reference's a is then an np.float64) rather than
// returned as (0.0, 0.0, inf) for want of weight.
std::tuple<double, double, double> phase_line(std::span<const cd> S, double dt, bool* fitted = nullptr);

// --- A2: the search -------------------------------------------------------------

// preamble_search(fe, t0_index, cand, fs, pfa=, kind=, inp=): the best
// (tau0, df) cell at cand, refined. With `inp` the channel is not
// recomputed (fe, t0_index and fs are then unused, as in the reference).
PreambleHit preamble_search(const A2Input& inp, const types::CarrierCandidate& cand, double pfa = rxa::A2_PFA,
                            const PreambleKind& kind = ce_pre());
PreambleHit preamble_search(std::span<const cf> fe, double t0_index, const types::CarrierCandidate& cand,
                            double fs = config::FE_FS, double pfa = rxa::A2_PFA,
                            const PreambleKind& kind = ce_pre());
PreambleHit preamble_search(std::span<const cd> fe, double t0_index, const types::CarrierCandidate& cand,
                            double fs = config::FE_FS, double pfa = rxa::A2_PFA,
                            const PreambleKind& kind = ce_pre());

// --- lead-in ----------------------------------------------------------------------

// frontend.noise_psd as lead_in_span calls it: (ch, CH_FS, fs_in=, audio_hz=).
// Empty: frontend::noise_psd itself.
using NoisePsdFn = std::function<std::vector<float>(std::span<const cf> ch, std::int64_t fs, std::int64_t fs_in,
                                                    double audio_hz)>;

// lead_in_span(fe, t0_index, f_hz, drift, tau0_s, fs, z_min=, t_sym=).
double lead_in_span(std::span<const cf> fe, double t0_index, double f_hz, double drift_hz_per_min = 0.0,
                    double tau0_s = 0.0, double fs = config::FE_FS, double z_min = rxa::LEAD_Z,
                    double t_sym = config::T_SYM, const NoisePsdFn& noise_psd = {});
double lead_in_span(std::span<const cd> fe, double t0_index, double f_hz, double drift_hz_per_min = 0.0,
                    double tau0_s = 0.0, double fs = config::FE_FS, double z_min = rxa::LEAD_Z,
                    double t_sym = config::T_SYM, const NoisePsdFn& noise_psd = {});

}  // namespace sstvae::qrss::acquire

#endif  // SSTVAE_QRSS_ACQUIRE_A12_HPP

// Symbol extraction, weights, the plain and joint block estimators and the
// header LLRs of waveform CE (design 6.6, WP6). Port of
// sstvae/qrss/demod.py (the reference), whose module docstring is the
// definition: for every non-reference stream symbol k at position p, from
// the tracker's matched-filter output m_p, smoothed gain u_p (error
// variance P_p), noise N_p and unblanked share psi_p,
//
//     y_k  = Im(m_p conj(u_p)) / (K |u_p|^2 psi_p)
//     s2_k = [N_p/2 + (A0^2 + K^2) P_p/2] / (K^2 |u_p|^2 psi_p^2) + D_PASS
//
// erased (y 0, s2 inf) where psi_p < 0.5 or |u_p|^2 is under U_MIN_REL of
// its median; then the kappa calibration on the pass's own symbols, the
// per-precoder-block estimators and the header LLRs.
//
// Not a format module. The pinned calibrations (D_PASS, D_REF, KAPPA_*,
// U_MIN_REL) are rx_config.hpp's, generated from demod.py.
//
// How close to Python, and why:
//
// - **Bitwise** (on the same libm, x86-64 glibc here): every array
//   operation is numpy's elementwise arithmetic in numpy's order --
//   complex multiplies as numpy's SIMD loop does them (fused, lwave::np_mul),
//   |u| as numpy's complex absolute (lwave::np_abs), x ** 2 on an array as
//   a square, np.median and np.mean (math::median, math::mean, numpy's
//   pairwise sum), np.bincount as a left-to-right sum per bin from 0, the
//   WHT in the precoder's order (precoder::wht). Python's scalar float
//   `** 2` is C pow, kept as a real pow call here (a compiler folds
//   pow(x, 2.0) into x * x, which is not always pow's rounding).
// - np.exp in data_estimates is numpy's own AVX-512 exp, which is 1 ulp
//   from libm's in about 5% of arguments; the only arguments that reach it
//   are -beta^2 rho / 2 for the frames' rho (0 and 0.054), where the two
//   agree. Elsewhere the result would hold to 1e-16 relative.
// - So symbol_estimates, data_estimates, spread_llr, calibrate,
//   block_estimate and extract come out as Python's to the bit given the
//   same track; extract's float32 outputs are the float64 results rounded
//   once, as numpy's astype rounds them.
//
// The track is read through a `TrackView`: its frame, its grid of
// positions and the per-position arrays the demodulator reads, plus the
// two things it reads of the classes (CE's spread_rho, L's hdr_bits).
// `view(tr)` gives one of a TrackResult; `extract`'s m argument (Python's
// "m: the matched-filter outputs, None: the track's own") is a view whose
// m is replaced. Indexing follows numpy: a negative index counts from the
// end, anything else out of range is std::out_of_range with numpy's
// IndexError text.
//
// Refusals are the reference's: frame::WrongWaveform (Python's
// AssertionError) for a frame of the other waveform, std::invalid_argument
// (ValueError) for an unknown estimator and for arrays that do not
// broadcast.

#ifndef SSTVAE_QRSS_DEMOD_HPP
#define SSTVAE_QRSS_DEMOD_HPP

#include <array>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "qrss/frame.hpp"
#include "qrss/rx_config.hpp"
#include "qrss/trackbase.hpp"

namespace sstvae::qrss::demod {

using cplx = std::complex<double>;
using frame::FrameSpec;

// The pinned calibrations (rx_config.hpp, generated from demod.py).
inline constexpr double D_PASS = rxconfig::demod::D_PASS;
inline constexpr double D_REF = rxconfig::demod::D_REF;
inline constexpr std::array<double, 2> KAPPA_RANGE = rxconfig::demod::KAPPA_RANGE;
inline constexpr double KAPPA_PRIOR_SD = rxconfig::demod::KAPPA_PRIOR_SD;
inline constexpr double KAPPA_S2_MAX = rxconfig::demod::KAPPA_S2_MAX;
inline constexpr double KAPPA_S2_REL = rxconfig::demod::KAPPA_S2_REL;
inline constexpr double U_MIN_REL = rxconfig::demod::U_MIN_REL;

// What the demodulators (CE's here, L's in ldemod) read of a tracker run.
// Non-owning: the arrays belong to the TrackResult (or whatever the view
// was made from) and must outlive the view.
struct TrackView {
    const FrameSpec* spec = nullptr;          // tr.spec
    std::span<const std::int64_t> pos;        // the grid's positions; gi(p) = p - pos[0]
    std::span<const cplx> m;                  // per grid position
    std::span<const cplx> u;
    std::span<const cplx> c;                  // CE's template (calibrate reads Im c / K)
    std::span<const double> P;
    std::span<const double> N;
    std::span<const double> psi;
    std::span<const std::uint8_t> meas;       // 0/1 per grid position; empty is None
    double spread_rho = 0.0;                  // float(getattr(classes, "spread_rho", 0) or 0)
    std::optional<std::span<const std::uint8_t>> hdr_bits;   // L's classes.hdr_bits; nullopt is None
};

// The view of a TrackResult: CE Classes give their spread_rho, LClasses
// their spread_rho and hdr_bits, no classes neither.
TrackView view(const track::TrackResult& tr);

// --- numpy's arithmetic, shared with ldemod -------------------------------------

namespace np {
// Python's float ** float (C pow, never folded into a multiply).
double py_pow(double x, double y);
// numpy's complex128 division a / b (CDOUBLE_divide: Smith's form; a real
// divisor, cast to (d, 0), comes out as a multiply by 1/d).
cplx cdiv(cplx a, cplx b);
// x[i] with numpy's index rules for an axis of `size` (negative from the
// end); std::out_of_range "index I is out of bounds for axis 0 with size N".
// Inline: it runs once per element of every gather (a third of extract's
// time while it was a call).
[[noreturn]] void index_error(std::int64_t i, std::size_t size);
inline std::size_t index(std::int64_t i, std::size_t size) {
    const auto n = static_cast<std::int64_t>(size);
    const std::int64_t j = i < 0 ? i + n : i;
    if (j < 0 || j >= n) [[unlikely]]
        index_error(i, size);
    return static_cast<std::size_t>(j);
}
// The grid indices of stream symbols `stream_idx`: layout.pos[idx] -
// int(tr.pos[0]), as signed numbers (an array indexed by them checks each
// with `index`).
std::vector<std::int64_t> grid_of(const TrackView& tr, std::span<const std::int64_t> stream_idx);
// |u[i]|^2 for the grid positions >= 0 (tr.u[tr.pos >= 0]), as numpy's
// boolean index does it ("boolean index did not match ..." when u and pos
// differ in length).
std::vector<double> u2_at_nonneg(const TrackView& tr);
}  // namespace np

// --- symbol estimates ------------------------------------------------------------

// (y, s2 noise part, s2 total) for stream symbols `stream_idx`, uncalibrated.
struct SymbolEstimates {
    std::vector<double> y, s2n, s2;
};
SymbolEstimates symbol_estimates(const TrackView& tr, std::span<const std::int64_t> stream_idx);

// (y, s2 noise part, extra variance) of the data symbols, uncalibrated.
// With spread_rho > 0 (a spread header not yet known) y is corrected for
// the lowered gain exp(-beta^2 rho / 2) and extra is rho; else extra 0.
struct DataEstimates {
    std::vector<double> y, s2n;
    double extra = 0.0;
};
DataEstimates data_estimates(const TrackView& tr);

// float64[2474]: coded-bit LLRs from the spread header on the data symbols
// (y and s2 per data symbol, s2 inf where erased), unwhitened and folded
// over the repeats. y, s2 and spec.n_data() must agree in length
// (std::invalid_argument with numpy's broadcast text).
std::vector<double> spread_llr(std::span<const double> y, std::span<const double> s2, const FrameSpec& spec);

// --- calibration ------------------------------------------------------------------

struct Calibration {
    double kappa = 1.0;
    bool suspect = false;
    friend bool operator==(const Calibration&, const Calibration&) = default;
};

// (kappa, variance) estimates, combined by inverse variance, shrunk to 1
// (prior sd KAPPA_PRIOR_SD) and clamped to KAPPA_RANGE; suspect when the
// raw estimate is more than 3 sd outside it. No estimates: (1, false).
// Shared with ldemod (its _combine).
Calibration combine(std::span<const std::pair<double, double>> ests);

// calibrate: (kappa, suspect) from the residuals of the pass's own
// symbols. frame::WrongWaveform for an L frame ("demod.calibrate is CE's;
// 'l-short' is waveform 1: use ldemod.calibrate_l").
Calibration calibrate(const TrackView& tr);

// --- block estimators --------------------------------------------------------------

// The estimator's name checked: "joint" or "plain", else
// std::invalid_argument "estimator must be 'joint' or 'plain', not <shown>"
// (`shown` empty: the name's repr).
void check_estimator(std::string_view estimator, std::string_view shown = {});

// (z, v): air-order latents and their variance from precoded-domain y, s2
// (s2 inf where erased), per precoder block of the n = len(y) data latents.
struct BlockEstimate {
    std::vector<double> z, v;
};
BlockEstimate block_estimate(std::span<const double> y, std::span<const double> s2, std::uint64_t q,
                             std::string_view estimator = "joint", std::string_view shown = {});

// --- extract ------------------------------------------------------------------------

// (z, w, llr, diag) of one pass. z, w float32[n_data] in air order; llr
// float32[2474] (positive means bit 0; 0 values without a header).
// diag: kappa, suspect, the data's y and s2 in the precoded domain, the
// header symbols' y, the estimator, and the block and spread parts of the
// LLRs (nullopt where Python's are None). ldemod's extract_l returns the
// same record.
struct Extraction {
    std::vector<float> z, w, llr;
    double kappa = 1.0;
    bool suspect = false;
    std::vector<double> y, s2, y_hdr;
    std::string estimator;
    std::optional<std::vector<double>> llr_block, llr_spread;
};

// extract(m, track, spec, q, estimator, calibrate_kappa): `spec` is the
// argument the reference takes beside the track (the header block, the
// spread copy); the symbol estimates read tr.spec. For Python's non-None
// m, pass a view with m replaced. frame::WrongWaveform for an L `spec`
// ("demod.extract is CE's; ...: use ldemod.extract_l").
Extraction extract(const TrackView& tr, const FrameSpec& spec, std::uint64_t q, std::string_view estimator = "joint",
                   bool calibrate_kappa = true, std::string_view shown = {});
inline Extraction extract(const track::TrackResult& tr, const FrameSpec& spec, std::uint64_t q,
                          std::string_view estimator = "joint", bool calibrate_kappa = true) {
    return extract(view(tr), spec, q, estimator, calibrate_kappa);
}

// The parts of extract after the calibration, shared with ldemod: block
// estimates of the data, w = 1/v (0 where v is not finite and positive),
// z = 0 where w is, all cast to float32. `ex` receives z, w; s2 is the
// calibrated total.
void finish_data(Extraction& ex, std::span<const double> y, std::span<const double> s2, std::uint64_t q,
                 std::string_view estimator, std::string_view shown);

}  // namespace sstvae::qrss::demod

#endif  // SSTVAE_QRSS_DEMOD_HPP

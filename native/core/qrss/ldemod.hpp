// Symbol extraction, weights and header LLRs for waveform L: the L
// counterpart of demod. Port of sstvae/qrss/ldemod.py (the reference),
// whose module docstring is the definition. A stream symbol of class gain
// g (G_*_L: the clipper's gain on that class against the references) gives
//
//     r = m conj(u) / (|u|^2 psi g),        E[r | a] = a
//     y = lframe.unpair(r)                    two latents (or coded bits) per symbol
//     s2n = (N + g^2 P) / (g^2 |u|^2 psi^2)   per latent, the same for I and Q
//
// erased (r 0, s2n inf) where psi < 0.5 or |u|^2 is under U_MIN_REL of its
// median. Everything after the symbol estimates is CE's on the x scale
// (demod.hpp: block_estimate, spread_llr, the kappa combination), with the
// spread header's copy either rho of extra variance (header unknown) or
// subtracted (known).
//
// Not a format module; the pinned constants (G_*_L, PILOT_GAIN_*, D_*_L)
// are rx_config.hpp's, generated from ldemod.py.
//
// Parity is demod.hpp's: numpy's elementwise arithmetic in numpy's order,
// so bitwise given the same track on the same libm -- including numpy's
// complex division by a real array, which multiplies by the reciprocal
// (demod::np::cdiv), and Python's sum() of the kappa estimates.
//
// The track is a demod::TrackView (its meas is the tracker's mask, empty
// for None; hdr_bits the classes' known header bits, nullopt while
// unknown). Refusals: frame::WrongWaveform (Python's AssertionError,
// "<what> is L's; 'short' is waveform 0: use demod's") from
// kappa_estimates_l, calibrate_l and extract_l on a CE frame.

#ifndef SSTVAE_QRSS_LDEMOD_HPP
#define SSTVAE_QRSS_LDEMOD_HPP

#include <complex>
#include <cstdint>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

#include "qrss/demod.hpp"
#include "qrss/frame.hpp"
#include "qrss/rx_config.hpp"

namespace sstvae::qrss::ldemod {

using cplx = std::complex<double>;
using demod::TrackView;
using frame::FrameSpec;

// The pinned constants (rx_config.hpp, generated from ldemod.py).
inline constexpr double G_REF_L = rxconfig::ldemod::G_REF_L;
inline constexpr double G_HDR_L = rxconfig::ldemod::G_HDR_L;
inline constexpr double G_DATA_L = rxconfig::ldemod::G_DATA_L;
inline constexpr double G_PRE_L = rxconfig::ldemod::G_PRE_L;
inline constexpr double PILOT_GAIN_PRE = rxconfig::ldemod::PILOT_GAIN_PRE;
inline constexpr double PILOT_GAIN_HDR = rxconfig::ldemod::PILOT_GAIN_HDR;
inline constexpr double PILOT_GAIN_DATA = rxconfig::ldemod::PILOT_GAIN_DATA;
inline constexpr double PILOT_GAIN_ID = rxconfig::ldemod::PILOT_GAIN_ID;
inline constexpr double G_REF_ABS = rxconfig::ldemod::G_REF_ABS;
inline constexpr double D_PASS_L = rxconfig::ldemod::D_PASS_L;
inline constexpr double D_REF_L = rxconfig::ldemod::D_REF_L;
inline constexpr double D_HDR_L = rxconfig::ldemod::D_HDR_L;
inline constexpr double D_PRE_L = rxconfig::ldemod::D_PRE_L;
inline constexpr double D_PASS_LX = rxconfig::ldemod::D_PASS_LX;

// The data's clipper distortion on the frame's x scale: D_PASS_L (1 + rho).
double d_pass_x(const FrameSpec& spec);

// (r complex128, s2n float64) per stream symbol in `stream_idx`,
// uncalibrated, for class gain `gain`. Erased symbols: r = 0, s2n = inf.
struct SymbolEstimatesL {
    std::vector<cplx> r;
    std::vector<double> s2n;
};
SymbolEstimatesL symbol_estimates_l(const TrackView& tr, std::span<const std::int64_t> stream_idx, double gain);

// (y_x, s2n_x, extra) per data latent (the precoded domain, x scale),
// uncalibrated: the spread copy as rho of extra variance while the header
// is unknown, subtracted (extra 0, y 0 where erased) once it is known.
demod::DataEstimates data_estimates_l(const TrackView& tr);

// (y_h, s2n_h) float64[2474] in coded-bit order (+1 for bit 0), uncalibrated.
struct HeaderEstimates {
    std::vector<double> y, s2n;
};
HeaderEstimates header_estimates_l(const TrackView& tr);

// The (kappa, variance) estimates calibrate_l combines: the references'
// (when 16 are usable), then the data's (256).
std::vector<std::pair<double, double>> kappa_estimates_l(const TrackView& tr);

// _combine: demod::combine.
inline demod::Calibration combine(std::span<const std::pair<double, double>> ests) { return demod::combine(ests); }

// (kappa, suspect) of an L pass.
demod::Calibration calibrate_l(const TrackView& tr);

// extract_l(m, track, spec, q, estimator, calibrate_kappa): demod::extract's
// contract for an L pass (a view with m replaced for a non-None m).
demod::Extraction extract_l(const TrackView& tr, const FrameSpec& spec, std::uint64_t q,
                            std::string_view estimator = "joint", bool calibrate_kappa = true,
                            std::string_view shown = {});
inline demod::Extraction extract_l(const track::TrackResult& tr, const FrameSpec& spec, std::uint64_t q,
                                   std::string_view estimator = "joint", bool calibrate_kappa = true) {
    return extract_l(demod::view(tr), spec, q, estimator, calibrate_kappa);
}

}  // namespace sstvae::qrss::ldemod

#endif  // SSTVAE_QRSS_LDEMOD_HPP

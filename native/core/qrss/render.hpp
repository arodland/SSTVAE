// The codec boundary's arithmetic: an accumulator's (S, W) -> the decoder's
// two planes. Port of the pure part of sstvae/qrss/render.py (g_shrink,
// reference_w, decoder_planes; the reference), whose module docstring is
// the definition:
//
//     latent = g(W) S / W,  g(W) = G_MAX sqrt(W / (W + G_KNEE))
//     weight = min(sqrt(W / W~), 1),  W~ the largest of the groups' median
//              W over their sent latents (1 for every heard latent when W~ is 0)
//
// never-sent latents, groups above the mode and groups without data at
// latent 0, weight 0. render() itself (the codec call) is not here.
//
// Not a format module; G_MAX and G_KNEE are rx_config.hpp's.
//
// Bitwise: numpy's elementwise arithmetic in numpy's order on float64,
// np.median (math::median), np.maximum's NaN rule (a NaN W stays NaN, and
// -0.0 becomes +0.0 as numpy's maximum gives it), then one rounding to
// float32 as astype does -- so the planes are Python's to the bit.

#ifndef SSTVAE_QRSS_RENDER_HPP
#define SSTVAE_QRSS_RENDER_HPP

#include <cstdint>
#include <span>
#include <utility>
#include <vector>

#include "qrss/rx_config.hpp"

namespace sstvae::qrss::render {

inline constexpr double G_MAX = rxconfig::render::G_MAX;
inline constexpr double G_KNEE = rxconfig::render::G_KNEE;
inline constexpr int N_GROUPS = rxconfig::render::N_GROUPS;
inline constexpr int GROUP_LATENTS = rxconfig::render::GROUP_LATENTS;

// g_shrink: G_MAX sqrt(W / (W + G_KNEE)) of max(W, 0), elementwise.
std::vector<double> g_shrink(std::span<const double> W);

// reference_w: W~ for W float64[3 * 52800] (any shape of that size in
// Python; flat here), 0.0 when no group up to `mode` has data. Python's
// errors: std::invalid_argument "cannot reshape array of size N into shape
// (3,52800)", std::out_of_range "index 3 is out of bounds for axis 0 with
// size 3" for a mode above 2 (a negative mode has no groups: 0.0).
double reference_w(std::span<const double> W, std::int64_t mode);

// decoder_planes: (latents, weights), float32[158400] each, canonical flat
// order. std::invalid_argument "mode must be 0..2, not <mode>", the reshape
// text above, or "W must be finite and non-negative, S finite".
std::pair<std::vector<float>, std::vector<float>> decoder_planes(std::span<const double> S,
                                                                 std::span<const double> W, std::int64_t mode);
// The same from an accumulator's float32 planes (widened exactly, as
// np.asarray(..., dtype=float64) does).
std::pair<std::vector<float>, std::vector<float>> decoder_planes(std::span<const float> S, std::span<const float> W,
                                                                 std::int64_t mode);

}  // namespace sstvae::qrss::render

#endif  // SSTVAE_QRSS_RENDER_HPP

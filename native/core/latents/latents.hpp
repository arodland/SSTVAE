// Latent-vector bookkeeping: the operations that are about the *shape*
// of a latent vector rather than about the neural network.
//
// These live outside `core/codec/` deliberately. The codec is behind
// `-DSSTVAE_BUILD_CODEC` because it drags in onnxruntime, and the
// receive engine needs to extend a short mode's latents to full length
// on every poll -- which is a memcpy, not an inference. Leaving it in
// the codec would have made the rx state machine unbuildable offline
// (and untestable without a downloaded artifact) over a reshape.
//
// `codec.hpp` re-exports both names, so `codec::pad_to_full` still
// resolves.

#ifndef SSTVAE_LATENTS_LATENTS_HPP
#define SSTVAE_LATENTS_LATENTS_HPP

#include <vector>

#include "config.hpp"

namespace sstvae::latents {

// Mode C's length -- the model-facing contract, and the size every
// vector handed to the decoder must have.
inline constexpr int N_LATENTS =
    config::LATENT_CHANNELS * config::LATENT_H * config::LATENT_W;

// Extend a mode A/B latent (or weight) vector to mode C's length.
// The modes are nested, so a shorter one is a full-length vector whose
// tail never arrived -- which is exactly what weight 0 means.
std::vector<double> pad_to_full(const std::vector<double>& vec, double fill = 0.0);

}  // namespace sstvae::latents

#endif

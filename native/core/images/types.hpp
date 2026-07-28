// Picture geometry and the two in-memory picture representations.
//
// Split out from both `images.hpp` and `codec.hpp` so that neither
// depends on the other for them. The codec is behind an option
// (SSTVAE_BUILD_CODEC) because it drags in onnxruntime; picture types
// must not, or an offline build loses the ability to talk about a
// picture at all.

#ifndef SSTVAE_IMAGES_TYPES_HPP
#define SSTVAE_IMAGES_TYPES_HPP

#include <cstdint>
#include <vector>

#include "config.hpp"

namespace sstvae::images {

// Target resolution. `sstvae/images.py` states these as literals; here
// they are tied to the latent grid, which is what actually fixes them
// -- the decoder upsamples by 16 in each axis, so the picture size is
// not an independent choice, and a static_assert is more honest than a
// second copy of the numbers.
inline constexpr int UPSAMPLE = 16;
inline constexpr int IMG_W = config::LATENT_W * UPSAMPLE;
inline constexpr int IMG_H = config::LATENT_H * UPSAMPLE;
static_assert(IMG_W == 640 && IMG_H == 480, "picture geometry moved");

// 8-bit RGB, interleaved, row-major -- the layout PIL hands back, so a
// picture crosses the binding without a transpose.
struct Picture {
    int width = 0;
    int height = 0;
    std::vector<std::uint8_t> rgb;  // height * width * 3

    Picture() = default;
    Picture(int w, int h)
        : width(w), height(h), rgb(static_cast<std::size_t>(w) * h * 3) {}
    bool empty() const { return rgb.empty(); }
};

// Planar float in [0,1], (3, H, W) -- what the encoder graph takes, and
// what `images::to_array` produces.
struct ImageArray {
    int width = 0;
    int height = 0;
    std::vector<float> chw;  // 3 * height * width
};

}  // namespace sstvae::images

#endif

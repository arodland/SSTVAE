// Framing pictures for the codec: load, fit to the target geometry,
// convert to the array the encoder wants.
//
// The counterpart of `sstvae/images.py`. Two of the three operations
// are exact and are checked against it; the third, scaling, deliberately
// is not -- see native/third_party/stb/README.md for why that is a
// decision rather than an omission.
//
// Font handling from images.py is *not* here. It exists to draw text on
// pictures, which is the overlay renderer's job, and it lands in
// core/overlay/ with the rest of the drawing code.

#ifndef SSTVAE_IMAGES_IMAGES_HPP
#define SSTVAE_IMAGES_IMAGES_HPP

#include <string>
#include <vector>

#include "images/types.hpp"

namespace sstvae::images {

// The smallest input accepted, upscaled from there -- parity with
// classic 320x240 SSTV sources. Matches images.py.
inline constexpr int MIN_W = 320;
inline constexpr int MIN_H = 240;

// Scale to an arbitrary size, ignoring aspect ratio -- the operation
// `Image.resize((w, h))` performs. Used for the receiver's optional
// "save at 320x240" and as the first half of `fit`.
//
// Already-correct input is returned untouched.
Picture resize(const Picture& img, int width, int height);

// Any picture -> exactly IMG_W x IMG_H RGB, by scaling to cover the
// target and centre-cropping. Deterministic and aspect-preserving.
//
// Already-correct input is returned untouched, which is the path the
// parity tests use.
Picture fit(const Picture& img);

// IMG_W x IMG_H RGB -> (3, IMG_H, IMG_W) float32 in [0,1].
// Exact: a transpose and a divide by 255.
ImageArray to_array(const Picture& img);

// Open any stb-readable picture (PNG, JPEG, BMP, GIF, TGA, ...).
// Throws with the file name and stb's reason on failure.
Picture load(const std::string& path);

// Cover-resize, centre-crop, convert -- `images.load_image`.
ImageArray load_array(const std::string& path);

// Write a PNG. Throws on failure.
void save_png(const Picture& img, const std::string& path);

}  // namespace sstvae::images

#endif

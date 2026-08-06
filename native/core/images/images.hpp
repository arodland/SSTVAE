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

#include <cstddef>
#include <cstdint>
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

// How a picture is framed into the target rectangle.
//
// The pictures the codec sends are 4:3, and anything else has to lose
// something. The default here is what this function has always done
// silently -- scale to cover, crop the centre -- expressed as data so
// the operator can move it. There is no letterbox option: padding
// would spend airtime on black, and the decision (2026-08-01) was that
// an operator who wants the whole frame pads the file themselves.
struct Framing {
    // Multiplier on the *cover* scale -- the smallest scale that fills
    // the target. 1.0 is the tightest framing that keeps the picture
    // full-bleed; above 1.0 crops in further. Below 1.0 would expose
    // edges with nothing behind them, so callers clamp there.
    double zoom = 1.0;
    // Centre of the crop window in normalized source coordinates.
    // (0.5, 0.5) is the middle of the picture, which is what the
    // parameterless overload has always used.
    double center_x = 0.5;
    double center_y = 0.5;
};

// Any picture -> exactly IMG_W x IMG_H RGB, by scaling to cover the
// target and cropping. Deterministic and aspect-preserving.
//
// Already-correct input is returned untouched, which is the path the
// parity tests use.
//
// **The default framing is byte-identical to what this produced before
// framing existed** -- the crop offset uses `floor`, which is what the
// old `(scaled - target) / 2` integer division was for a non-negative
// numerator. `test_framing` holds it to that by reimplementing the old
// formula rather than by calling this function twice.
Picture fit(const Picture& img);
Picture fit(const Picture& img, const Framing& framing);

// IMG_W x IMG_H RGB -> (3, IMG_H, IMG_W) float32 in [0,1].
// Exact: a transpose and a divide by 255.
ImageArray to_array(const Picture& img);

// The largest picture file that will be opened at all. `load` reads the
// whole file into memory before decoding -- both the decoder and the
// EXIF parser want the bytes -- so an absurd file is refused up front
// rather than allocated for. A GiB is far above any real photograph and
// far below what would trouble a desktop, so it is a guard against a
// mistake (a WAV, a disk image, a truncated download of something else)
// rather than a limit anyone should meet.
inline constexpr std::size_t MAX_FILE_BYTES = 1024u * 1024u * 1024u;

// The EXIF orientation tag (0x0112), 1..8, as it appears in a JPEG's
// APP1 segment. 1 is the identity: stored top-left, no transform.
// Values 5..8 exchange width and height.
//
// Returns 1 for a file with no EXIF, no orientation tag, a value
// outside 1..8, or metadata too corrupt to parse -- an unreadable tag
// costs the operator a rotation, never the picture. `data` is the whole
// file, not a segment.
int exif_orientation(const std::uint8_t* data, std::size_t len);

// Undo an EXIF orientation, producing the upright picture. Orientation
// 1 (and anything outside 1..8) returns the input untouched.
//
// The counterpart of Pillow's `ImageOps.exif_transpose`, and required
// to agree with it exactly -- `tests/test_native_parity.py -k exif`
// holds both implementations to the same eight results, since a
// disagreement here is a rotated transmission rather than an error.
Picture apply_orientation(const Picture& img, int orientation);

// Open any stb-readable picture (PNG, JPEG, BMP, GIF, TGA, ...).
// Throws with the file name and stb's reason on failure.
//
// JPEG orientation metadata is applied here, so everything downstream
// -- `fit`'s framing above all -- sees the picture the way it was
// taken. stb decodes pixels and ignores EXIF entirely; the tag is read
// separately (third_party/easyexif).
Picture load(const std::string& path);

// Cover-resize, centre-crop, convert -- `images.load_image`.
ImageArray load_array(const std::string& path);

// Write a PNG. Throws on failure.
void save_png(const Picture& img, const std::string& path);

}  // namespace sstvae::images

#endif

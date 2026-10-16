#include "images/images.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <vector>

// Vendored; read for the orientation tag alone. See
// native/third_party/easyexif/README.md.
#include "easyexif/exif.h"

#define STB_IMAGE_IMPLEMENTATION
#define STBI_FAILURE_USERMSG
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image_resize2.h"

namespace sstvae::images {
namespace {

// Whole file into memory. Both consumers -- the stb decoder and the
// EXIF parser -- want the bytes rather than a path, and a picture is
// already being held decoded at several times this size.
std::vector<std::uint8_t> read_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot read " + path);
    std::vector<std::uint8_t> bytes;
    in.seekg(0, std::ios::end);
    const std::streamoff size = in.tellg();
    if (size < 0) throw std::runtime_error("cannot read " + path);
    // Refused before the allocation, not after -- the point is to not
    // reserve a gigabyte on the strength of a file name. The message
    // names the size, because the usual cause is a file that is not a
    // picture at all and the operator will recognise it by its length.
    if (static_cast<std::uintmax_t>(size) > MAX_FILE_BYTES) {
        throw std::runtime_error(path + " is " + std::to_string(size) +
                                 " bytes; the limit for a picture file is " +
                                 std::to_string(MAX_FILE_BYTES));
    }
    in.seekg(0, std::ios::beg);
    bytes.resize(static_cast<std::size_t>(size));
    if (size > 0) {
        in.read(reinterpret_cast<char*>(bytes.data()), size);
        if (!in) throw std::runtime_error("cannot read " + path);
    }
    return bytes;
}

}  // namespace

Picture resize(const Picture& img, int width, int height) {
    if (img.width == width && img.height == height) return img;
    if (img.width <= 0 || img.height <= 0) {
        throw std::runtime_error("cannot resize an empty picture");
    }
    if (width <= 0 || height <= 0) {
        throw std::runtime_error("resize target must be positive");
    }
    Picture out(width, height);
    if (stbir_resize_uint8_srgb(img.rgb.data(), img.width, img.height, 0, out.rgb.data(),
                                width, height, 0, STBIR_RGB) == nullptr) {
        throw std::runtime_error("image resize failed");
    }
    return out;
}

Picture fit(const Picture& img) { return fit(img, Framing{}); }

double min_zoom(int width, int height) {
    if (width <= 0 || height <= 0) return 1.0;
    const double src = static_cast<double>(width) / height;
    const double target = static_cast<double>(IMG_W) / IMG_H;
    // contain / cover, which is the ratio of the two aspects whichever
    // way round they are.
    return std::min(src, target) / std::max(src, target);
}

Picture fit(const Picture& img, const Framing& framing) {
    // The identity short-circuit only applies to the default framing:
    // an operator who has zoomed or panned an already-4:3 picture means
    // it, and returning the original would silently ignore them.
    const bool defaulted = framing.zoom == 1.0 && framing.center_x == 0.5 &&
                           framing.center_y == 0.5;
    if (defaulted && img.width == IMG_W && img.height == IMG_H) return img;
    if (img.width <= 0 || img.height <= 0) {
        throw std::runtime_error("cannot fit an empty picture");
    }

    // Scale relative to *cover* the target, then crop -- the same shape
    // of operation as images.py, though not the same filter. Below
    // `min_zoom` there is nothing left to reveal, only more black.
    const double zoom =
        std::max(min_zoom(img.width, img.height), framing.zoom);
    const double scale = std::max(static_cast<double>(IMG_W) / img.width,
                                  static_cast<double>(IMG_H) / img.height) *
                         zoom;
    // At zoom 1 and above the scaled picture covers the target, and the
    // floor of IMG_W/IMG_H is the old guard against rounding a pixel
    // short of it. Below 1 an axis is *meant* to come up short, so the
    // guard would be exactly the bug -- it would silently cancel the
    // padding the operator asked for.
    const int floor_w = zoom >= 1.0 ? IMG_W : 1;
    const int floor_h = zoom >= 1.0 ? IMG_H : 1;
    const int sw = std::max(floor_w, static_cast<int>(std::lround(img.width * scale)));
    const int sh = std::max(floor_h, static_cast<int>(std::lround(img.height * scale)));

    const Picture scaled = resize(img, sw, sh);

    // `floor`, not `lround`: for the default centre this has to reduce
    // to the old `(sw - IMG_W) / 2` integer division exactly, and those
    // agree only under truncation of a non-negative value. With an odd
    // `sw` they differ by one pixel, so adding framing would have
    // silently shifted every odd-intermediate picture -- 1920x1080
    // scales to 853 wide, so that is most photographs. `test_framing`
    // pins this against the old formula written out; the Python parity
    // suite cannot, since it deliberately skips `fit_image` wherever
    // resampling is involved (PIL LANCZOS vs stb).
    //
    // The two clamp bounds are `0` and `scaled - target` in whichever
    // order they come: when the scaled picture is the larger the window
    // must stay inside it (the original rule), and when it is the
    // smaller the same expression keeps the *picture* inside the
    // window, so the padding never lands all on one side with the
    // photograph half off the canvas.
    const auto offset = [](double center, int scaled_size, int target) {
        const double raw = center * scaled_size - target / 2.0;
        const int slack = scaled_size - target;
        return std::clamp(static_cast<int>(std::floor(raw)), std::min(0, slack),
                          std::max(0, slack));
    };
    const int left = offset(framing.center_x, sw, IMG_W);
    const int top = offset(framing.center_y, sh, IMG_H);

    // Zero-initialized, which is the black the uncovered part is padded
    // with -- so the copy below only has to skip what it cannot fill.
    Picture out(IMG_W, IMG_H);
    const int x0 = std::max(0, -left);            // first covered column
    const int x1 = std::min(IMG_W, sw - left);    // one past the last
    for (int y = 0; y < IMG_H; ++y) {
        const int sy = y + top;
        if (sy < 0 || sy >= sh || x1 <= x0) continue;
        const std::uint8_t* src =
            scaled.rgb.data() + (static_cast<std::size_t>(sy) * sw + (x0 + left)) * 3;
        std::copy(src, src + static_cast<std::size_t>(x1 - x0) * 3,
                  out.rgb.begin() + (static_cast<std::size_t>(y) * IMG_W + x0) * 3);
    }
    return out;
}

ImageArray to_array(const Picture& img) {
    if (img.width != IMG_W || img.height != IMG_H) {
        throw std::runtime_error("to_array wants a fitted " + std::to_string(IMG_W) +
                                 "x" + std::to_string(IMG_H) + " picture");
    }
    ImageArray out;
    out.width = img.width;
    out.height = img.height;
    out.chw.resize(static_cast<std::size_t>(3) * IMG_H * IMG_W);

    // (H, W, 3) interleaved -> (3, H, W) planar, /255.
    //
    // The divide is exact in the sense that matters: 255 is a power-of-
    // two-free integer, so v/255.0f is correctly rounded from an exact
    // integer numerator and matches numpy's float32 divide bit for bit.
    const std::size_t plane = static_cast<std::size_t>(IMG_H) * IMG_W;
    for (std::size_t i = 0; i < plane; ++i) {
        for (int c = 0; c < 3; ++c) {
            out.chw[c * plane + i] = static_cast<float>(img.rgb[i * 3 + c]) / 255.0f;
        }
    }
    return out;
}

namespace {

// PNG stores EXIF in an `eXIf` chunk holding a bare TIFF stream -- no
// `Exif\0\0` signature, which is what a JPEG APP1 segment carries and
// what easyexif's segment entry point expects. So the chunk is located
// here and the signature prepended, leaving the TIFF parsing (byte
// order, IFD offsets, bounds) where it belongs: in the vendored code.
//
// Walking PNG chunks is not the same kind of undertaking as walking a
// TIFF: every chunk is a 4-byte big-endian length followed by a 4-byte
// type, so the whole walk is bounds-checkable by inspection and there
// are no internal offsets that can point anywhere.
//
// Pillow honours this chunk, so not reading it would mean a tagged PNG
// rotating in `sstvae/images.py` and not here -- exactly the silent,
// consequential disagreement the parity test exists to prevent.
std::vector<std::uint8_t> png_exif_segment(const std::uint8_t* data, std::size_t len) {
    static constexpr std::uint8_t SIGNATURE[8] = {0x89, 0x50, 0x4E, 0x47,
                                                  0x0D, 0x0A, 0x1A, 0x0A};
    if (len < sizeof(SIGNATURE) || !std::equal(SIGNATURE, SIGNATURE + sizeof(SIGNATURE), data)) {
        return {};
    }
    std::size_t pos = sizeof(SIGNATURE);
    while (pos + 8 <= len) {
        const std::size_t size = (static_cast<std::size_t>(data[pos]) << 24) |
                                 (static_cast<std::size_t>(data[pos + 1]) << 16) |
                                 (static_cast<std::size_t>(data[pos + 2]) << 8) |
                                 static_cast<std::size_t>(data[pos + 3]);
        const std::uint8_t* type = data + pos + 4;
        const std::size_t body = pos + 8;
        // The 4-byte CRC follows the body. Overflow is not reachable --
        // `size` is at most 2^32-1 and `body` at most `len` -- but the
        // comparison is written so it would not matter if it were.
        if (size > len || body > len - size || body + size + 4 > len) break;
        if (std::equal(type, type + 4, "eXIf")) {
            std::vector<std::uint8_t> out = {'E', 'x', 'i', 'f', 0x00, 0x00};
            out.insert(out.end(), data + body, data + body + size);
            return out;
        }
        pos = body + size + 4;
    }
    return {};
}

}  // namespace

int exif_orientation(const std::uint8_t* data, std::size_t len) {
    if (data == nullptr || len == 0) return 1;
    easyexif::EXIFInfo info;
    // Anything either parser objects to -- not a JPEG or PNG, no EXIF, a
    // truncated or corrupt segment -- is an unrotated picture, not a
    // failure. The return code is deliberately not distinguished: there
    // is nothing a caller could do differently with "no EXIF" than with
    // "corrupt".
    const std::vector<std::uint8_t> png = png_exif_segment(data, len);
    const int status =
        png.empty()
            ? info.parseFrom(data, static_cast<unsigned>(len))
            : info.parseFromEXIFSegment(png.data(), static_cast<unsigned>(png.size()));
    if (status != PARSE_EXIF_SUCCESS) return 1;
    const int value = static_cast<int>(info.Orientation);
    // 0 is easyexif's "tag absent"; anything above 8 is undefined in the
    // spec and has been seen in the wild from buggy writers.
    return (value >= 1 && value <= 8) ? value : 1;
}

Picture apply_orientation(const Picture& img, int orientation) {
    if (orientation <= 1 || orientation > 8 || img.empty()) return img;

    // Each case is the *inverse* map: where in the source does the
    // destination pixel come from. Written out rather than composed from
    // flip/transpose flags because the composition order is exactly the
    // thing that is easy to get backwards, and eight explicit lines can
    // be read against the spec's table one at a time.
    //
    // 5..8 exchange the axes, so the output's geometry changes -- which
    // is the whole point, and why this must run before `fit` decides
    // what to crop.
    const int sw = img.width, sh = img.height;
    const bool swap_axes = orientation >= 5;
    Picture out(swap_axes ? sh : sw, swap_axes ? sw : sh);

    for (int dy = 0; dy < out.height; ++dy) {
        for (int dx = 0; dx < out.width; ++dx) {
            int sx = 0, sy = 0;
            switch (orientation) {
                case 2: sx = sw - 1 - dx; sy = dy;             break;  // mirror
                case 3: sx = sw - 1 - dx; sy = sh - 1 - dy;    break;  // 180
                case 4: sx = dx;          sy = sh - 1 - dy;    break;  // flip
                case 5: sx = dy;          sy = dx;             break;  // transpose
                case 6: sx = dy;          sy = sh - 1 - dx;    break;  // 90 CW
                case 7: sx = sw - 1 - dy; sy = sh - 1 - dx;    break;  // transverse
                case 8: sx = sw - 1 - dy; sy = dx;             break;  // 90 CCW
                default: sx = dx;         sy = dy;             break;
            }
            const std::size_t src = (static_cast<std::size_t>(sy) * sw + sx) * 3;
            const std::size_t dst = (static_cast<std::size_t>(dy) * out.width + dx) * 3;
            out.rgb[dst] = img.rgb[src];
            out.rgb[dst + 1] = img.rgb[src + 1];
            out.rgb[dst + 2] = img.rgb[src + 2];
        }
    }
    return out;
}

Picture load(const std::string& path) {
    // Read the file once and decode from memory, because the orientation
    // tag and the pixels come from the same bytes: stb ignores EXIF, and
    // easyexif wants the whole file. `stbi_load(path, ...)` would mean
    // opening it twice.
    std::vector<std::uint8_t> bytes = read_file(path);

    int w = 0, h = 0, channels = 0;
    // Forced to 3 channels: an RGBA or greyscale source is converted the
    // way `Image.convert("RGB")` would.
    std::uint8_t* data = stbi_load_from_memory(bytes.data(), static_cast<int>(bytes.size()),
                                               &w, &h, &channels, 3);
    if (data == nullptr) {
        throw std::runtime_error("cannot read " + path + ": " + stbi_failure_reason());
    }
    Picture out(w, h);
    std::copy(data, data + static_cast<std::size_t>(w) * h * 3, out.rgb.begin());
    stbi_image_free(data);
    return apply_orientation(out, exif_orientation(bytes.data(), bytes.size()));
}

ImageArray load_array(const std::string& path) { return to_array(fit(load(path))); }

void save_png(const Picture& img, const std::string& path) {
    if (stbi_write_png(path.c_str(), img.width, img.height, 3, img.rgb.data(),
                       img.width * 3) == 0) {
        throw std::runtime_error("cannot write " + path);
    }
}

}  // namespace sstvae::images

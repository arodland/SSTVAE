#include "images/images.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#define STB_IMAGE_IMPLEMENTATION
#define STBI_FAILURE_USERMSG
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image_resize2.h"

namespace sstvae::images {

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

Picture load(const std::string& path) {
    int w = 0, h = 0, channels = 0;
    // Forced to 3 channels: an RGBA or greyscale source is converted the
    // way `Image.convert("RGB")` would.
    std::uint8_t* data = stbi_load(path.c_str(), &w, &h, &channels, 3);
    if (data == nullptr) {
        throw std::runtime_error("cannot read " + path + ": " + stbi_failure_reason());
    }
    Picture out(w, h);
    std::copy(data, data + static_cast<std::size_t>(w) * h * 3, out.rgb.begin());
    stbi_image_free(data);
    return out;
}

ImageArray load_array(const std::string& path) { return to_array(fit(load(path))); }

void save_png(const Picture& img, const std::string& path) {
    if (stbi_write_png(path.c_str(), img.width, img.height, 3, img.rgb.data(),
                       img.width * 3) == 0) {
        throw std::runtime_error("cannot write " + path);
    }
}

}  // namespace sstvae::images

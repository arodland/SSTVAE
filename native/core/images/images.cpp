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

Picture fit(const Picture& img) {
    if (img.width == IMG_W && img.height == IMG_H) return img;
    if (img.width <= 0 || img.height <= 0) {
        throw std::runtime_error("cannot fit an empty picture");
    }

    // Scale to *cover* the target, then centre-crop -- the same shape of
    // operation as images.py, though not the same filter.
    const double scale = std::max(static_cast<double>(IMG_W) / img.width,
                                  static_cast<double>(IMG_H) / img.height);
    const int sw = std::max(IMG_W, static_cast<int>(std::lround(img.width * scale)));
    const int sh = std::max(IMG_H, static_cast<int>(std::lround(img.height * scale)));

    const Picture scaled = resize(img, sw, sh);

    Picture out(IMG_W, IMG_H);
    const int left = (sw - IMG_W) / 2;
    const int top = (sh - IMG_H) / 2;
    for (int y = 0; y < IMG_H; ++y) {
        const std::uint8_t* src =
            scaled.rgb.data() + (static_cast<std::size_t>(y + top) * sw + left) * 3;
        std::copy(src, src + static_cast<std::size_t>(IMG_W) * 3,
                  out.rgb.begin() + static_cast<std::size_t>(y) * IMG_W * 3);
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

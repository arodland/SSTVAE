#include "composition.hpp"

#include <algorithm>

namespace sstvae::androidapp {

Composition& Composition::instance() {
    static Composition c;
    return c;
}

bool Composition::set_source(const std::string& path, std::string* error) {
    images::Picture loaded;
    try {
        loaded = images::load(path);
    } catch (const std::exception& e) {
        if (error) *error = e.what();
        return false;
    }
    if (loaded.width < images::MIN_W || loaded.height < images::MIN_H) {
        // The same floor `images.py` has kept since classic SSTV
        // sources. Refused rather than upscaled silently, because at
        // this point the operator can go and pick a better file, and
        // discovering it after a 95 s transmission cannot.
        if (error) {
            *error = "that picture is " + std::to_string(loaded.width) + "x" +
                     std::to_string(loaded.height) + "; the smallest accepted is " +
                     std::to_string(images::MIN_W) + "x" + std::to_string(images::MIN_H);
        }
        return false;
    }

    std::lock_guard<std::mutex> lk(mu_);
    source_ = std::move(loaded);
    path_ = path;
    framing_ = images::Framing{};
    return true;
}

void Composition::clear() {
    std::lock_guard<std::mutex> lk(mu_);
    source_ = images::Picture{};
    path_.clear();
    framing_ = images::Framing{};
}

bool Composition::has_source() const {
    std::lock_guard<std::mutex> lk(mu_);
    return source_.width > 0 && source_.height > 0;
}

std::string Composition::source_path() const {
    std::lock_guard<std::mutex> lk(mu_);
    return path_;
}

int Composition::source_width() const {
    std::lock_guard<std::mutex> lk(mu_);
    return source_.width;
}

int Composition::source_height() const {
    std::lock_guard<std::mutex> lk(mu_);
    return source_.height;
}

// Half the crop window's extent in normalized source coordinates, on
// each axis. At zoom 1 the window covers the full extent of whichever
// axis is tight, so its centre cannot move on that axis at all -- which
// is why this is computed rather than clamped to a constant.
std::pair<double, double> Composition::half_extents(double zoom) const {
    const double aspect_src = source_.height > 0
                                  ? static_cast<double>(source_.width) / source_.height
                                  : 1.0;
    const double aspect_dst = static_cast<double>(images::IMG_W) / images::IMG_H;
    double half_x = 0.5 / zoom;
    double half_y = 0.5 / zoom;
    if (aspect_src > aspect_dst) {
        // Source is wider than the target: the crop is full height and
        // narrower than full width, so only x has room to move.
        half_x *= aspect_dst / aspect_src;
    } else {
        half_y *= aspect_src / aspect_dst;
    }
    return {half_x, half_y};
}

void Composition::set_framing(const images::Framing& framing) {
    std::lock_guard<std::mutex> lk(mu_);
    // The floor is `min_zoom`, not 1.0: below cover the picture is
    // letterboxed into the frame rather than cropped, and at `min_zoom`
    // exactly all of it is visible. Zooming further out would only add
    // black, so that is where the travel stops -- and it is the same
    // function `images::fit` clamps to, so the preview and the
    // transmitted picture cannot disagree about where the end is.
    framing_.zoom = std::max(images::min_zoom(source_.width, source_.height),
                             framing.zoom);
    const auto [half_x, half_y] = half_extents(framing_.zoom);
    // **A half-extent of 0.5 or more means the axis has no slack**, and
    // below zoom 1 it can exceed 0.5 -- the window is then wider than
    // the source on that axis, and the picture is centred in it with
    // black either side. The clamp range would be inverted, which is
    // undefined behaviour for `std::clamp` rather than a no-op, so the
    // centre is pinned instead.
    const auto pin = [](double value, double half) {
        return half >= 0.5 ? 0.5 : std::clamp(value, half, 1.0 - half);
    };
    framing_.center_x = pin(framing.center_x, half_x);
    framing_.center_y = pin(framing.center_y, half_y);
}

void Composition::pan(double frac_x, double frac_y) {
    images::Framing next;
    {
        std::lock_guard<std::mutex> lk(mu_);
        const auto [half_x, half_y] = half_extents(framing_.zoom);
        next = framing_;
        next.center_x += frac_x * 2.0 * half_x;
        next.center_y += frac_y * 2.0 * half_y;
    }
    set_framing(next);
}

images::Framing Composition::framing() const {
    std::lock_guard<std::mutex> lk(mu_);
    return framing_;
}

images::Picture Composition::preview() const {
    std::lock_guard<std::mutex> lk(mu_);
    if (source_.width <= 0 || source_.height <= 0) return {};
    return images::fit(source_, framing_);
}

}  // namespace sstvae::androidapp

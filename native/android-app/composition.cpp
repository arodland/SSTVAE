#include "composition.hpp"

#include <algorithm>

#include "overlay/render.hpp"
#include "session.hpp"

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

void Composition::set_template(overlay::Doc doc) {
    std::lock_guard<std::mutex> lk(mu_);
    template_ = std::move(doc);
}

overlay::Doc Composition::template_doc() const {
    std::lock_guard<std::mutex> lk(mu_);
    return template_;
}

void Composition::set_fields(overlay::Fields fields) {
    std::lock_guard<std::mutex> lk(mu_);
    fields_ = std::move(fields);
}

overlay::Fields Composition::fields() const {
    std::lock_guard<std::mutex> lk(mu_);
    return fields_;
}

void Composition::set_reply_target(const std::string& path, const std::string& callsign,
                                   double snr_db) {
    std::lock_guard<std::mutex> lk(mu_);
    reply_target_set_ = true;
    reply_target_path_ = path;
    reply_target_callsign_ = callsign;
    reply_target_snr_db_ = snr_db;
}

bool Composition::has_reply_target() const {
    std::lock_guard<std::mutex> lk(mu_);
    return reply_target_set_;
}

std::string Composition::reply_target_callsign() const {
    std::lock_guard<std::mutex> lk(mu_);
    return reply_target_callsign_;
}

double Composition::reply_target_snr_db() const {
    std::lock_guard<std::mutex> lk(mu_);
    return reply_target_snr_db_;
}

images::Picture Composition::preview() const {
    // Read what Session knows *before* taking our own lock, so this
    // never holds two objects' locks at once -- Session has no call path
    // back into Composition today, but there is no reason to rely on
    // that staying true.
    const Session::LastReception last = Session::instance().last_reception();

    images::Picture base;
    overlay::Doc doc;
    overlay::Fields fields;
    bool has_target = false;
    std::string target_path;
    {
        std::lock_guard<std::mutex> lk(mu_);
        if (source_.width <= 0 || source_.height <= 0) return {};
        base = images::fit(source_, framing_);
        doc = template_;
        fields = fields_;
        has_target = reply_target_set_;
        target_path = reply_target_path_;
    }
    // "None": the picture goes out unmodified, and at no rendering cost
    // at all -- today's behaviour, byte for byte.
    if (doc.items.empty()) return base;

    // With no explicit reply target, `last_rx` means the newest
    // reception -- the same rule the desktop's editor uses when nothing
    // has been chosen for it either.
    const std::string last_rx_path = has_target ? target_path : last.path;

    const images::Picture* last_rx = nullptr;
    {
        std::lock_guard<std::mutex> lk(last_rx_mu_);
        if (last_rx_path.empty()) {
            last_rx_cache_.reset();
            last_rx_cache_path_.clear();
        } else if (last_rx_path != last_rx_cache_path_) {
            try {
                last_rx_cache_ = images::load(last_rx_path);
                last_rx_cache_path_ = last_rx_path;
            } catch (const std::exception&) {
                // A reception's PNG that cannot be re-read (deleted,
                // corrupt) draws nothing rather than failing the whole
                // preview -- the same "a template that insets the most
                // recent picture is perfectly valid on a session that
                // has not had one" rule `overlay::render` documents.
                last_rx_cache_.reset();
                last_rx_cache_path_.clear();
            }
        }
        if (last_rx_cache_) last_rx = &*last_rx_cache_;
    }

    return overlay::render(base, overlay::substitute(doc, fields), last_rx);
}

}  // namespace sstvae::androidapp

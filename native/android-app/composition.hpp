// The picture being prepared for transmission, and how it is framed.
//
// **Process-wide, for the same reason `Session` is.** Picking a photo
// and cropping it is minutes of the operator's attention, and an
// activity destroyed on a rotation would otherwise throw all of it away.
// So the composition outlives any view, and `Transmitter` is a view over
// it exactly as `Listener` is a view over `Session`.
//
// **There is no overlay here, and that is a decision** (Andrew,
// 2026-08-09). The desktop composes: insets, captions, a callsign drawn
// into the picture. This app deliberately does not, not even an
// automatic callsign -- the station is identified by the beacon carrier,
// which every receiver decodes whether or not anyone can read text in a
// picture, and optionally by a CW ID that a human can copy by ear.
// Burning a callsign into the pixels identifies the station only to
// someone who already decoded the picture, and spends the codec's
// bit budget doing it. So the operator picks an image, crops it, and it
// goes out unmodified.
//
// What survives of the desktop's rule is the part that matters: the
// preview **is** the output of the same `images::fit` the transmitter
// will run, not a toolkit-drawn imitation of a crop. There is no second
// representation that can drift from what goes on the air.

#ifndef SSTVAE_ANDROID_COMPOSITION_HPP
#define SSTVAE_ANDROID_COMPOSITION_HPP

#include <mutex>
#include <string>
#include <utility>

#include "images/images.hpp"
#include "images/types.hpp"

namespace sstvae::androidapp {

class Composition {
public:
    static Composition& instance();

    // Load a picture file. False and `error` on anything unreadable --
    // which on this platform includes a file the picker handed us from a
    // provider we cannot open, so it is a normal outcome and not an
    // assertion. Resets the framing: a crop is about one picture.
    bool set_source(const std::string& path, std::string* error);
    void clear();

    bool has_source() const;
    std::string source_path() const;
    // Source dimensions, for a UI that has to bound a pan.
    int source_width() const;
    int source_height() const;

    // Zoom is clamped at `images::min_zoom` below -- the point at which
    // the whole source is visible, letterboxed -- and the centre is
    // clamped so the crop window stays inside the picture wherever
    // there is slack to move in, which is what stops a drag walking the
    // frame off the image.
    void set_framing(const images::Framing& framing);
    images::Framing framing() const;

    // Move the crop window by a fraction of *its own* extent -- so
    // `pan(-0.25, 0)` shifts the visible content a quarter of a preview
    // width to the right, whatever the source's aspect and the zoom.
    //
    // In those units rather than in normalized source coordinates
    // because the conversion needs the crop window's size, which needs
    // the aspect ratio and the zoom, both of which live here. A UI doing
    // that arithmetic would be a second copy of the geometry that
    // `set_framing` clamps against, and the two would drift.
    void pan(double frac_x, double frac_y);

    // Exactly what will be transmitted: IMG_W x IMG_H, through the same
    // call the transmitter makes. Empty if there is no source.
    images::Picture preview() const;

private:
    Composition() = default;
    Composition(const Composition&) = delete;
    Composition& operator=(const Composition&) = delete;

    // Half the crop window's extent on each axis, in normalized source
    // coordinates. `mu_` must be held.
    std::pair<double, double> half_extents(double zoom) const;

    mutable std::mutex mu_;
    images::Picture source_;
    std::string path_;
    images::Framing framing_;
};

}  // namespace sstvae::androidapp

#endif

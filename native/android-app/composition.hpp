// The picture being prepared for transmission, and how it is framed.
//
// **Process-wide, for the same reason `Session` is.** Picking a photo
// and cropping it is minutes of the operator's attention, and an
// activity destroyed on a rotation would otherwise throw all of it away.
// So the composition outlives any view, and `Transmitter` is a view over
// it exactly as `Listener` is a view over `Session`.
//
// **There was no overlay here, and that was a decision** (Andrew,
// 2026-08-09) -- reversed 2026-09-14, and the reversal is worth reading
// rather than skipping. The station is still identified by the beacon
// carrier and, optionally, a CW ID; neither of those can say **whom**
// a transmission is addressed to, and a QSO reply is addressed --
// "W1XYZ de KC2G" is what makes an over a reply rather than a second
// broadcast. `docs/overlay-templates.md` is the design: a *template* is
// an ordinary `overlay::Doc` with `{theircall}`-style placeholders, so
// the operator never free-hand composes on this screen -- there is
// still no editor here, only a template picker and a small fields row
// (step 4 is the editor, and is not built). An automatic callsign
// caption is still not offered: `{mycall}` exists only inside a
// template the operator chose, on purpose.
//
// What survived unchanged from the original design: the preview **is**
// the output of the same pipeline the transmitter runs (`images::fit`,
// then, with a template active, `overlay::render` over the substituted
// document), not a toolkit-drawn imitation. There is no second
// representation that can drift from what goes on the air.

#ifndef SSTVAE_ANDROID_COMPOSITION_HPP
#define SSTVAE_ANDROID_COMPOSITION_HPP

#include <mutex>
#include <optional>
#include <string>
#include <utility>

#include "images/images.hpp"
#include "images/types.hpp"
#include "overlay/model.hpp"
#include "overlay/template.hpp"

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
    // call the transmitter makes -- `images::fit`, then, with a
    // template active, `overlay::render` over the substituted document.
    // Empty if there is no source.
    images::Picture preview() const;

    // --- templates (docs/overlay-templates.md) --------------------------
    //
    // Composition owns these too, for the reason it owns the picture
    // and the framing: `preview()` has to be able to render the whole
    // composite with nothing else supplied. `Transmitter` is the policy
    // -- which template is chosen, what the operator typed -- and pushes
    // the result here, the same division `OverlayEditor::set_fields`
    // draws on the desktop.

    // The raw template: placeholders intact, exactly as a phone editor
    // (step 4, not built) would show them. Empty items means "None",
    // the original unmodified-picture behaviour.
    void set_template(overlay::Doc doc);
    overlay::Doc template_doc() const;

    // What fills the current template's holes right now. Recomputed by
    // `Transmitter` on every relevant change (a keystroke, a template
    // switch, a fresh reception's SNR) and pushed whole, rather than
    // tracked incrementally here.
    void set_fields(overlay::Fields fields);
    overlay::Fields fields() const;

    // The reception Reply was pressed on, if any. Its callsign and SNR
    // are what `Transmitter` reads to fill `{theircall}` (once, as a
    // seed -- the field stays editable after) and `{snr}` (every time,
    // never typed); a `last_rx` item in the template resolves to *this*
    // picture rather than whichever reception is newest. Choosing a
    // different *source* picture (Choose/Camera) does **not** clear
    // this -- replying with a freshly taken photo as the outgoing
    // picture is the ordinary case, not an edge one.
    void set_reply_target(const std::string& path, const std::string& callsign,
                          double snr_db);
    bool has_reply_target() const;
    std::string reply_target_callsign() const;
    double reply_target_snr_db() const;

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
    overlay::Doc template_;
    overlay::Fields fields_;
    bool reply_target_set_ = false;
    std::string reply_target_path_;
    std::string reply_target_callsign_;
    double reply_target_snr_db_ = 0.0;

    // The `last_rx` picture, cached and reloaded only when the path it
    // should show actually changes -- `preview()` may run once per
    // crop-drag frame or per keystroke in a field, and a render must
    // not decode a PNG from disk that often. Separate from `mu_` so a
    // slow decode never blocks a pan or a field edit.
    mutable std::mutex last_rx_mu_;
    mutable std::string last_rx_cache_path_;
    mutable std::optional<images::Picture> last_rx_cache_;
};

}  // namespace sstvae::androidapp

#endif

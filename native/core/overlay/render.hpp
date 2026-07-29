// Drawing an overlay Doc onto a picture.
//
// The counterpart of `sstvae/overlay/render.py`, and it keeps that
// module's central property: **what the editor previews is what this
// produces**, because the editor previews this output rather than
// painting its own imitation. `item_bbox` is shared with the editor for
// the same reason -- selection handles positioned by separate geometry
// drift away from what is drawn, and the operator sees the drift before
// anyone else does.
//
// QtGui only, never QtWidgets: `tools/check_layering.py` enforces it,
// so an overlay stays renderable from the command line under
// QT_QPA_PLATFORM=offscreen.
//
// Text is where this *cannot* match the reference and is not asked to.
// PIL lays out glyphs with FreeType and Qt with its own stack, so
// identical pixels were never available at any price; the rule from
// `docs/native-app.md` applies -- the on-air format is normative,
// everything above the modem may use native idioms. What is preserved
// is the document's meaning: PIL's two-letter anchors, multi-line
// alignment and spacing, stroke, and rotation all do here what the
// document says they do.

#ifndef SSTVAE_OVERLAY_RENDER_HPP
#define SSTVAE_OVERLAY_RENDER_HPP

#include "images/types.hpp"
#include "overlay/model.hpp"

namespace sstvae::overlay {

// Pixel rectangle an item occupies.
struct Bbox {
    int x = 0;
    int y = 0;
    int w = 0;
    int h = 0;
};

// Where `item` lands on a canvas of this size.
//
// `last_rx` supplies the aspect ratio for a "last_rx" image item; with
// none, 4:3 is assumed so an editor still has a handle to grab before
// anything has been received.
Bbox item_bbox(int canvas_w, int canvas_h, const Item& item,
               const images::Picture* last_rx = nullptr);

// Draw `doc` over `base`, returning a new picture.
//
// `base` is used as given: the document's coordinates are fractions of
// whatever canvas it receives, so framing to the transmit size is the
// caller's job (`images::fit`). A `last_rx` item with nothing received
// yet draws nothing rather than failing -- a template that insets the
// most recent picture is perfectly valid on a session that has not had
// one.
//
// Requires a live QGuiApplication: Qt's font database is a
// platform-integration service, and text drawn without one is not a
// warning but a crash.
images::Picture render(const images::Picture& base, const Doc& doc,
                       const images::Picture* last_rx = nullptr);

}  // namespace sstvae::overlay

#endif

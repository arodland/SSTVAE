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
// **Rotation is not applied**, and callers that draw a selection need
// `item_quad` instead. This is the un-turned box: the extent the item
// would occupy at rotation 0, which is what the layout arithmetic and
// the renderer both start from.
//
// `last_rx` supplies the aspect ratio for a "last_rx" image item; with
// none, 4:3 is assumed so an editor still has a handle to grab before
// anything has been received.
Bbox item_bbox(int canvas_w, int canvas_h, const Item& item,
               const images::Picture* last_rx = nullptr);

// A point in canvas pixels.
struct Point {
    double x = 0.0;
    double y = 0.0;
};

// The point `item` turns about, in canvas pixels.
//
// **Not the box's centre, and the two kinds differ.** `render` turns
// text about the anchor the document pins -- the point `(x, y)`
// addresses -- and an image about its own centre. That asymmetry is
// real and is kept: changing it would change how an already-saved
// document draws, and the anchor is what makes a template's
// corner-pinned text stay pinned when it is turned.
//
// It is published because an editor cannot guess it. A rotate handle
// that orbits any other point sends the item somewhere the drawing does
// not go, which is the same class of drift `item_bbox` living here
// rather than in the editor exists to prevent.
Point item_pivot(int canvas_w, int canvas_h, const Item& item,
                 const images::Picture* last_rx = nullptr);

// `item_bbox`'s four corners after rotation, in canvas pixels:
// top-left, top-right, bottom-right, bottom-left of the *un-turned*
// box, each carried round `item_pivot`.
//
// Convex by construction (it is a rectangle), so `quad_contains` can
// use a sign test.
struct Quad {
    Point corner[4];
};

Quad item_quad(int canvas_w, int canvas_h, const Item& item,
               const images::Picture* last_rx = nullptr);

// Whether a canvas-pixel point falls inside `quad`.
bool quad_contains(const Quad& quad, double x, double y);

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

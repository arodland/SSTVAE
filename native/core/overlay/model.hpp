// The overlay document: what to draw on top of a picture.
//
// The document only -- rendering lives with the editor in Phase 3. Two
// design choices carry over from `sstvae/overlay/model.py`, and both
// exist so that *templates* (saving an overlay and reapplying it to
// tomorrow's picture) stay a UI-only change rather than a redesign:
//
// **Coordinates are normalized** to 0..1 of the canvas and sizes are
// fractions of it, so a document is resolution-independent.
//
// **Image insets are late-bound references, not pasted bitmaps.**
// `ImageItem::source` is "last_rx" or a path, resolved at render time.
// A template saying "inset the most recent received picture, bottom
// left" therefore still means that next week -- which is the entire
// point, and impossible if the editor flattened the bitmap in at
// composition time.

#ifndef SSTVAE_OVERLAY_MODEL_HPP
#define SSTVAE_OVERLAY_MODEL_HPP

#include <optional>
#include <string>
#include <variant>
#include <vector>

#include "images/types.hpp"

namespace sstvae::overlay {

// The overlay's coordinate space is the transmitted frame itself, so
// what the editor shows is what goes on the air.
inline constexpr int CANVAS_W = images::IMG_W;
inline constexpr int CANVAS_H = images::IMG_H;

inline constexpr int DOC_VERSION = 1;

// Resolved at render time rather than stored, so the reference stays
// meaningful in a saved template.
inline constexpr const char* SOURCE_LAST_RX = "last_rx";

// A run of burned-in text.
//
// `text` may contain newlines: a station's callsign, grid and name are
// one item, not three stacked by hand.
struct TextItem {
    std::string text;
    double x = 0.03;
    double y = 0.03;
    // Cap height as a fraction of canvas height, so text scales with
    // the frame.
    double size = 0.08;
    std::string color = "#ffffff";
    std::string stroke_color = "#000000";
    double stroke_width = 0.12;  // fraction of the glyph size
    // Font path; empty = the default face.
    std::string font;
    // Which point of the text box (x, y) positions, in PIL's two-letter
    // convention ("la" = left/ascender, "mm" = middle/middle). This is
    // what lets a template pin text to a corner without knowing how
    // long the string will be.
    std::string anchor = "la";
    std::string align = "left";   // between lines, once there is more than one
    double line_spacing = 0.15;   // extra gap, fraction of size
    double rotation = 0.0;        // degrees, counter-clockwise
};

// A picture inset -- typically the last received image, so an operator
// can send back what they just got.
struct ImageItem {
    std::string source = SOURCE_LAST_RX;  // "last_rx" or a filesystem path
    double x = 0.68;
    double y = 0.68;
    double width = 0.28;    // fraction of canvas width; height follows aspect
    double border = 0.004;  // fraction of canvas width; 0 for none
    std::string border_color = "#ffffff";
    double opacity = 1.0;
    double rotation = 0.0;
    std::string anchor = "la";
};

// A filled and/or stroked rectangle.
//
// Fill and stroke are independent, each "none" | "solid" | "gradient" --
// the kind and its colour(s) travel together as flat fields, matching
// this file's style rather than a nested gradient struct. A gradient is
// linear only: two colours and an angle. The angle is counter-clockwise,
// matching `rotation`, and deliberately: the gradient is painted into
// the item's own unrotated layer and rotated with it
// (`native/core/overlay/render.cpp`), so a gradient's angle and the
// item's rotation add exactly as the two numbers suggest they should.
struct RectItem {
    double x = 0.1;
    double y = 0.1;
    double width = 0.3;   // fraction of canvas width
    double height = 0.2;  // fraction of canvas height
    double rotation = 0.0;  // degrees, counter-clockwise
    std::string anchor = "la";

    std::string fill_kind = "none";  // "none" | "solid" | "gradient"
    std::string fill_color = "#ffffff";
    std::string fill_color2 = "#000000";  // the gradient's second stop
    double fill_angle = 0.0;

    std::string stroke_kind = "none";  // "none" | "solid" | "gradient"
    std::string stroke_color = "#ffffff";
    std::string stroke_color2 = "#000000";
    double stroke_angle = 0.0;
    double stroke_width = 0.006;  // fraction of canvas width
};

using Item = std::variant<TextItem, ImageItem, RectItem>;

// An ordered list of items, drawn back to front.
//
// `name` is what makes a document a *template* (`overlay/template.hpp`):
// a saved layout the operator picks by name. Written only when set, so
// an unnamed document serializes exactly as it did before templates
// existed, and a template opens as a plain document in any build with
// this model.
struct Doc {
    std::vector<Item> items;
    int version = DOC_VERSION;
    std::string name;

    bool empty() const { return items.empty(); }
};

// What was skipped while loading. Unknown item kinds and unknown fields
// are ignored for forward compatibility, but -- as with settings --
// ignoring quietly makes a hand-edited document's typo invisible.
struct Note {
    std::string where;
    std::string problem;
};

// Parse a document.
//
// Unlike `settings::load`, this *can* fail, and deliberately: a
// document is a file the operator explicitly opened, so malformed JSON
// or a version this build cannot understand is worth reporting rather
// than silently replacing with an empty overlay. Unknown items and
// fields within a readable document are still skipped and noted.
Doc from_json(const std::string& text, std::vector<Note>* notes = nullptr);

std::string to_json(const Doc& doc, int indent = 2);

}  // namespace sstvae::overlay

#endif

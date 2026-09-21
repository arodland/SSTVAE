// Overlay rendering.
//
// There is no pixel oracle here and there deliberately is not one: PIL
// and Qt lay out glyphs with different engines, so byte equality with
// the reference was never available (see render.hpp). What *is*
// checkable is everything the document promises -- that an item lands
// where its coordinates and anchor say, that the canvas is otherwise
// untouched, that a missing source draws nothing rather than failing,
// and above all that `item_bbox` agrees with what `render` draws, since
// the editor positions its selection handles with the former and the
// operator judges them against the latter.

#include <QFile>
#include <QGuiApplication>
#include <QImage>
#include <QString>
#include <QTemporaryDir>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "check.hpp"
#include "images/types.hpp"
#include "overlay/render.hpp"

using namespace sstvae;

namespace {

images::Picture solid(int w, int h, std::uint8_t r, std::uint8_t g,
                      std::uint8_t b) {
    images::Picture p(w, h);
    for (std::size_t i = 0; i < p.rgb.size(); i += 3) {
        p.rgb[i] = r;
        p.rgb[i + 1] = g;
        p.rgb[i + 2] = b;
    }
    return p;
}

struct Rgb {
    int r = 0;
    int g = 0;
    int b = 0;
};

Rgb pixel(const images::Picture& p, int x, int y) {
    const std::size_t i = (static_cast<std::size_t>(y) * p.width + x) * 3;
    return Rgb{p.rgb[i], p.rgb[i + 1], p.rgb[i + 2]};
}

bool same(const Rgb& a, const Rgb& b) {
    return a.r == b.r && a.g == b.g && a.b == b.b;
}

// How many pixels differ from the untouched background.
int painted(const images::Picture& out, const images::Picture& base) {
    int n = 0;
    for (std::size_t i = 0; i < out.rgb.size(); i += 3) {
        if (out.rgb[i] != base.rgb[i] || out.rgb[i + 1] != base.rgb[i + 1] ||
            out.rgb[i + 2] != base.rgb[i + 2]) {
            ++n;
        }
    }
    return n;
}

// Every painted pixel, as a bounding box. What was actually drawn,
// which is the thing item_bbox is claiming to predict.
overlay::Bbox painted_bbox(const images::Picture& out,
                           const images::Picture& base) {
    int x0 = out.width, y0 = out.height, x1 = -1, y1 = -1;
    for (int y = 0; y < out.height; ++y) {
        for (int x = 0; x < out.width; ++x) {
            if (!same(pixel(out, x, y), pixel(base, x, y))) {
                x0 = std::min(x0, x);
                y0 = std::min(y0, y);
                x1 = std::max(x1, x);
                y1 = std::max(y1, y);
            }
        }
    }
    if (x1 < 0) return overlay::Bbox{0, 0, 0, 0};
    return overlay::Bbox{x0, y0, x1 - x0 + 1, y1 - y0 + 1};
}

void test_an_empty_document_changes_nothing() {
    const images::Picture base = solid(64, 48, 10, 20, 30);
    const images::Picture out = overlay::render(base, overlay::Doc{});
    check::equal(out.width, base.width, "render/empty: width kept");
    check::equal(out.height, base.height, "render/empty: height kept");
    check::equal(painted(out, base), 0,
                 "render/empty: not one pixel touched");
}

void test_an_image_inset_lands_where_the_document_says() {
    const images::Picture base = solid(200, 150, 0, 0, 0);
    const images::Picture inset = solid(40, 30, 255, 0, 0);

    overlay::ImageItem item;
    item.source = overlay::SOURCE_LAST_RX;
    item.x = 0.25;
    item.y = 0.5;
    item.width = 0.25;  // 50 px wide, 4:3 source -> ~38 px tall
    item.border = 0.0;
    item.anchor = "la";

    overlay::Doc doc;
    doc.items.push_back(item);
    const images::Picture out = overlay::render(base, doc, &inset);

    const overlay::Bbox drawn = painted_bbox(out, base);
    check::equal(drawn.x, 50, "render/image: left edge at x*width");
    check::equal(drawn.y, 75, "render/image: top edge at y*height");
    check::equal(drawn.w, 50, "render/image: width is width*canvas");

    // The inset's own colour, not a blend: opacity is 1 and the source
    // is opaque, so a pixel inside it must be exactly the source.
    const Rgb inside = pixel(out, 60, 80);
    check::is_true(same(inside, Rgb{255, 0, 0}),
                   "render/image: drawn at full opacity");
    // And outside is still background.
    check::is_true(same(pixel(out, 10, 10), Rgb{0, 0, 0}),
                   "render/image: nothing painted outside it");
}

void test_the_anchor_moves_the_inset_not_the_picture() {
    const images::Picture base = solid(200, 150, 0, 0, 0);
    const images::Picture inset = solid(40, 30, 0, 255, 0);

    overlay::ImageItem item;
    item.x = 0.5;
    item.y = 0.5;
    item.width = 0.25;
    item.border = 0.0;
    item.anchor = "rb";  // right/bottom: the item ends at the point

    overlay::Doc doc;
    doc.items.push_back(item);
    const overlay::Bbox drawn =
        painted_bbox(overlay::render(base, doc, &inset), base);

    check::equal(drawn.x + drawn.w, 100,
                 "render/anchor: right edge sits on x");
    check::equal(drawn.y + drawn.h, 75,
                 "render/anchor: bottom edge sits on y");
}

void test_item_bbox_predicts_what_render_draws() {
    const images::Picture base = solid(200, 150, 0, 0, 0);
    const images::Picture inset = solid(40, 30, 0, 0, 255);

    // The property the editor depends on, over anchors that move the
    // item in both axes -- a handle drawn from item_bbox has to land on
    // the pixels render() paints.
    for (const std::string anchor : {"la", "mm", "rb", "lb", "rm"}) {
        overlay::ImageItem item;
        item.x = 0.5;
        item.y = 0.5;
        item.width = 0.3;
        item.border = 0.01;
        item.anchor = anchor;

        overlay::Doc doc;
        doc.items.push_back(item);
        const overlay::Bbox predicted =
            overlay::item_bbox(base.width, base.height, doc.items.front(), &inset);
        const overlay::Bbox drawn =
            painted_bbox(overlay::render(base, doc, &inset), base);

        const std::string what = "render/bbox: anchor " + anchor;
        check::equal(drawn.x, predicted.x, what + " predicts x");
        check::equal(drawn.y, predicted.y, what + " predicts y");
        check::equal(drawn.w, predicted.w, what + " predicts width");
        check::equal(drawn.h, predicted.h, what + " predicts height");
    }
}

void test_a_border_surrounds_the_inset() {
    const images::Picture base = solid(200, 150, 0, 0, 0);
    const images::Picture inset = solid(40, 30, 255, 0, 0);

    overlay::ImageItem item;
    item.x = 0.25;
    item.y = 0.25;
    item.width = 0.25;
    item.border = 0.02;  // 4 px
    item.border_color = "#00ff00";
    item.anchor = "la";

    overlay::Doc doc;
    doc.items.push_back(item);
    const images::Picture out = overlay::render(base, doc, &inset);
    const overlay::Bbox drawn = painted_bbox(out, base);

    check::is_true(same(pixel(out, drawn.x + 1, drawn.y + 1), Rgb{0, 255, 0}),
                   "render/border: the frame is the border colour");
    check::is_true(
        same(pixel(out, drawn.x + drawn.w / 2, drawn.y + drawn.h / 2),
             Rgb{255, 0, 0}),
        "render/border: the picture is inside it");
}

void test_a_missing_source_draws_nothing() {
    const images::Picture base = solid(64, 48, 7, 7, 7);

    // A template that insets "the last received picture" is valid on a
    // session where nothing has been received; it must draw nothing
    // rather than fail.
    overlay::Doc doc;
    doc.items.push_back(overlay::ImageItem{});
    check::equal(painted(overlay::render(base, doc, nullptr), base), 0,
                 "render/missing: last_rx with nothing received is a no-op");

    overlay::ImageItem missing;
    missing.source = "/nonexistent/definitely-not-here.png";
    overlay::Doc doc2;
    doc2.items.push_back(missing);
    check::equal(painted(overlay::render(base, doc2, nullptr), base), 0,
                 "render/missing: an unreadable path is a no-op");
}

void test_text_is_drawn_and_bounded_where_predicted() {
    const images::Picture base = solid(320, 240, 0, 0, 0);

    overlay::TextItem item;
    item.text = "W1AW";
    item.x = 0.1;
    item.y = 0.1;
    item.size = 0.15;
    item.color = "#ffffff";
    item.stroke_width = 0.0;

    overlay::Doc doc;
    doc.items.push_back(item);
    const images::Picture out = overlay::render(base, doc);
    check::is_true(painted(out, base) > 0, "render/text: something was drawn");

    // Not exact: item_bbox is the font's metric box, which includes the
    // ascender and descender whether or not this string reaches them,
    // so the ink is a subset. Containment is the property the editor
    // needs -- a handle must not cut the glyphs off.
    const overlay::Bbox predicted =
        overlay::item_bbox(base.width, base.height, doc.items.front());
    const overlay::Bbox ink = painted_bbox(out, base);
    check::is_true(ink.x >= predicted.x && ink.y >= predicted.y &&
                       ink.x + ink.w <= predicted.x + predicted.w &&
                       ink.y + ink.h <= predicted.y + predicted.h,
                   "render/text: the ink lies inside item_bbox");
}

void test_empty_text_still_has_a_handle() {
    // An item being typed into must stay selectable; a zero-size box
    // cannot be grabbed.
    overlay::TextItem item;
    item.text = "";
    const overlay::Bbox box = overlay::item_bbox(640, 480, overlay::Item{item});
    check::is_true(box.w >= 1 && box.h >= 1,
                   "render/text: an empty item still has a grabbable box");
}

void test_items_draw_back_to_front() {
    const images::Picture base = solid(200, 150, 0, 0, 0);
    const images::Picture red = solid(40, 30, 255, 0, 0);

    // A big red inset, then a smaller one on top of it wearing a thick
    // green border. Where the border lands it is over the first item's
    // pixels, so seeing green there is exactly "the later item wins".
    overlay::ImageItem under;
    under.x = 0.1;
    under.y = 0.1;
    under.width = 0.5;
    under.border = 0.0;

    overlay::ImageItem over;
    over.x = 0.2;
    over.y = 0.2;
    over.width = 0.15;
    over.border = 0.02;
    over.border_color = "#00ff00";

    overlay::Doc doc;
    doc.items.push_back(under);
    doc.items.push_back(over);
    const images::Picture out = overlay::render(base, doc, &red);

    const overlay::Bbox top =
        overlay::item_bbox(base.width, base.height, doc.items.back(), &red);
    check::is_true(same(pixel(out, top.x + 1, top.y + 1), Rgb{0, 255, 0}),
                   "render/order: the later item covers the earlier one");
    // And the first item is still there where the second does not reach.
    const overlay::Bbox bottom =
        overlay::item_bbox(base.width, base.height, doc.items.front(), &red);
    check::is_true(same(pixel(out, bottom.x + 1, bottom.y + 1), Rgb{255, 0, 0}),
                   "render/order: and does not erase the rest of it");
}

// A file-backed inset is decoded once per path.
//
// `item_bbox` is called by `OverlayEditor::hit_test` on every mouse
// move over the canvas and again by every paint, and it used to run
// `QImage::load()` on the item's path each time -- so moving the
// pointer across the composer re-read and re-decoded every inset from
// disk.
//
// **Deleting the file is how "cached" is made observable.** There is no
// counter to assert on, and adding one would be a test-only hook on a
// hot path; but a decode that does not happen cannot notice that its
// source is gone. The stale answer this pins is the documented
// contract, not an accident: an inset is identified by its path, and
// content changes behind that path are deliberately not watched (a stat
// per mouse move to catch a case an operator fixes by re-adding the
// item).
void test_a_file_inset_is_decoded_once() {
    // **`QTemporaryDir`, not `$TMPDIR` and not a fixed name.** The first
    // version of this reached for `getenv("TMPDIR")` with `/tmp` as the
    // fallback, which on Windows is neither set nor a directory -- so
    // `save()` failed, the item fell back to the 0.75 aspect a missing
    // source gets, and the test failed on one platform and nowhere
    // else. A unique directory also keeps the cache honest: it is keyed
    // on the path and lives for the process, so a fixed name shared
    // with another test would serve one test's pixels to another.
    QTemporaryDir dir;
    check::is_true(dir.isValid(), "cache: a temporary directory was made");
    if (!dir.isValid()) return;
    const QString qpath = dir.filePath(QStringLiteral("inset.png"));
    const std::string path = qpath.toStdString();

    // Deliberately not 4:3, so the aspect it reports could only have
    // come from this file.
    {
        QImage source(80, 20, QImage::Format_RGB888);
        source.fill(Qt::magenta);
        check::is_true(source.save(qpath), "cache: the fixture file was written");
    }

    overlay::ImageItem item;
    item.source = path;
    item.width = 0.25;
    item.border = 0.0;
    item.anchor = "la";
    const overlay::Item boxed = item;

    const overlay::Bbox first = overlay::item_bbox(200, 150, boxed, nullptr);
    // 50 px wide at 80x20 -> 12 or 13 px tall; whatever it is, it is
    // this file's aspect and not the 0.75 fallback a failed load gives.
    check::is_true(first.h < first.w / 2,
                   "cache: the source's own aspect was used");

    check::is_true(QFile::remove(qpath), "cache: the fixture file was removed");

    const overlay::Bbox second = overlay::item_bbox(200, 150, boxed, nullptr);
    check::equal(second.w, first.w, "cache: width unchanged after the file went");
    check::equal(second.h, first.h,
                 "cache: and height -- so it was not re-read from disk");

    // And the drawing path shares the cache, not just the measuring one.
    const images::Picture base = solid(200, 150, 0, 0, 0);
    overlay::Doc doc;
    doc.items.push_back(item);
    const images::Picture out = overlay::render(base, doc, nullptr);
    check::is_true(painted(out, base) > 0,
                   "cache: render still draws it after the file went");
}

// --- rectangles ---------------------------------------------------------

void test_a_solid_rect_fills_its_bbox_and_nothing_else() {
    const images::Picture base = solid(200, 150, 0, 0, 0);

    overlay::RectItem item;
    item.x = 0.1;
    item.y = 0.1;
    item.width = 0.3;
    item.height = 0.2;
    item.fill_kind = "solid";
    item.fill_color = "#ff0000";

    overlay::Doc doc;
    doc.items.push_back(item);
    const images::Picture out = overlay::render(base, doc);
    const overlay::Bbox predicted =
        overlay::item_bbox(base.width, base.height, doc.items.front());
    const overlay::Bbox drawn = painted_bbox(out, base);

    check::equal(drawn.x, predicted.x, "render/rect: left edge at x*width");
    check::equal(drawn.y, predicted.y, "render/rect: top edge at y*height");
    check::equal(drawn.w, predicted.w, "render/rect: width matches item_bbox");
    check::equal(drawn.h, predicted.h, "render/rect: height matches item_bbox");
    check::is_true(same(pixel(out, drawn.x + 2, drawn.y + 2), Rgb{255, 0, 0}),
                   "render/rect: filled with the solid colour");
    check::is_true(same(pixel(out, 5, 5), Rgb{0, 0, 0}),
                   "render/rect: nothing painted outside it");
}

void test_a_rect_with_no_fill_or_stroke_draws_nothing() {
    const images::Picture base = solid(64, 48, 9, 9, 9);
    overlay::Doc doc;
    doc.items.push_back(overlay::RectItem{});  // fill_kind/stroke_kind default "none"
    check::equal(painted(overlay::render(base, doc), base), 0,
                 "render/rect: an all-\"none\" rect is a legal no-op");
}

void test_a_gradient_rect_interpolates_between_its_two_colours() {
    const images::Picture base = solid(200, 100, 0, 0, 0);

    overlay::RectItem item;
    item.x = 0.0;
    item.y = 0.0;
    item.width = 1.0;
    item.height = 1.0;
    item.fill_kind = "gradient";
    item.fill_color = "#ff0000";
    item.fill_color2 = "#0000ff";
    item.fill_angle = 0.0;  // left (colour1) to right (colour2)

    overlay::Doc doc;
    doc.items.push_back(item);
    const images::Picture out = overlay::render(base, doc);

    const Rgb left = pixel(out, 2, 50);
    const Rgb right = pixel(out, 197, 50);
    check::is_true(left.r > left.b, "render/rect: left edge leans toward colour1");
    check::is_true(right.b > right.r, "render/rect: right edge leans toward colour2");
}

void test_a_stroke_only_rect_draws_an_outline_not_a_fill() {
    const images::Picture base = solid(200, 150, 0, 0, 0);

    overlay::RectItem item;
    item.x = 0.1;
    item.y = 0.1;
    item.width = 0.3;
    item.height = 0.2;
    item.stroke_kind = "solid";
    item.stroke_color = "#00ff00";
    item.stroke_width = 0.02;  // 4 px

    overlay::Doc doc;
    doc.items.push_back(item);
    const images::Picture out = overlay::render(base, doc);
    const overlay::Bbox box =
        overlay::item_bbox(base.width, base.height, doc.items.front());

    check::is_true(same(pixel(out, box.x + 1, box.y + box.h / 2), Rgb{0, 255, 0}),
                   "render/rect: the stroke is on the edge");
    check::is_true(same(pixel(out, box.x + box.w / 2, box.y + box.h / 2),
                        Rgb{0, 0, 0}),
                   "render/rect: the centre is untouched with no fill");
}

void test_rect_json_roundtrips_through_the_model() {
    overlay::RectItem item;
    item.x = 0.2;
    item.width = 0.4;
    item.height = 0.25;
    item.fill_kind = "gradient";
    item.fill_color = "#112233";
    item.fill_color2 = "#445566";
    item.fill_angle = 45.0;
    item.stroke_kind = "solid";
    item.stroke_color = "#778899";
    item.stroke_width = 0.01;
    item.rotation = 12.0;

    overlay::Doc doc;
    doc.items.push_back(item);
    const overlay::Doc back = overlay::from_json(overlay::to_json(doc));
    check::equal(back.items.size(), std::size_t{1}, "rect/json: one item survives");
    const overlay::RectItem* r = std::get_if<overlay::RectItem>(&back.items[0]);
    check::is_true(r != nullptr, "rect/json: item kind is \"rect\"");
    if (r == nullptr) return;
    check::equal(r->fill_color, item.fill_color, "rect/json: fill_color round-trips");
    check::equal(r->fill_kind, item.fill_kind, "rect/json: fill_kind round-trips");
    check::equal(r->stroke_width, item.stroke_width,
                 "rect/json: stroke_width round-trips");
}


// --- text style and radial gradients --------------------------------------

images::Picture render_one(const overlay::Item& item, const images::Picture& base) {
    overlay::Doc doc;
    doc.items.push_back(item);
    return overlay::render(base, doc);
}

// Mean (red - blue) over the painted pixels inside one rectangle of the
// canvas: positive leans toward a red first stop, negative toward a blue
// second one. `ink` says how many pixels that was, so a band that
// happened to hold no glyph cannot pass as "neutral".
struct Lean {
    double value = 0.0;
    int ink = 0;
};

Lean lean(const images::Picture& out, const images::Picture& base, int x0, int y0,
          int x1, int y1) {
    Lean l;
    double sum = 0.0;
    for (int y = std::max(0, y0); y < std::min(out.height, y1); ++y) {
        for (int x = std::max(0, x0); x < std::min(out.width, x1); ++x) {
            const Rgb p = pixel(out, x, y);
            if (same(p, pixel(base, x, y))) continue;
            sum += p.r - p.b;
            ++l.ink;
        }
    }
    l.value = l.ink > 0 ? sum / l.ink : 0.0;
    return l;
}

int count_exact(const images::Picture& out, const Rgb& colour) {
    int n = 0;
    for (int y = 0; y < out.height; ++y)
        for (int x = 0; x < out.width; ++x)
            if (same(pixel(out, x, y), colour)) ++n;
    return n;
}

// Wide capitals with a red-to-blue gradient and no stroke, so every
// painted pixel is fill.
overlay::TextItem gradient_text(const char* text, double size) {
    overlay::TextItem item;
    item.text = text;
    item.x = 0.05;
    item.y = 0.2;
    item.size = size;
    item.stroke_width = 0.0;
    item.fill_kind = "gradient";
    item.color = "#ff0000";
    item.fill_color2 = "#0000ff";
    return item;
}

void test_solid_text_never_reads_the_gradient_fields() {
    // Every document written before these fields existed has a solid
    // fill, and its pixels must not depend on fields it could not have
    // set. That is what "an unstyled document renders as before" comes
    // down to in code: the solid path takes the brush it always did and
    // reads nothing else.
    const images::Picture base = solid(320, 240, 10, 20, 30);
    overlay::TextItem plain;
    plain.text = "W1AW";
    plain.size = 0.2;
    overlay::TextItem noisy = plain;
    noisy.fill_color2 = "#00ff00";
    noisy.fill_angle = 77.0;
    noisy.fill_gradient = "radial";
    check::is_true(render_one(plain, base).rgb == render_one(noisy, base).rgb,
                   "render/style: a solid fill ignores the gradient fields");

    // And a fill kind this build does not know draws solid rather than
    // nothing -- the opposite of a rect's rule, deliberately: a caption
    // that vanishes on an older build is worse than one drawn flat.
    overlay::TextItem future = plain;
    future.fill_kind = "shimmer";
    check::is_true(render_one(future, base).rgb == render_one(plain, base).rgb,
                   "render/style: an unknown fill kind draws solid, not nothing");
}

void test_bold_and_italic_change_the_glyphs() {
    const images::Picture base = solid(320, 240, 0, 0, 0);
    overlay::TextItem regular;
    regular.text = "Wave";
    regular.size = 0.25;
    regular.stroke_width = 0.0;
    overlay::TextItem bold = regular;
    bold.bold = true;
    overlay::TextItem italic = regular;
    italic.italic = true;

    const int ink = painted(render_one(regular, base), base);
    check::is_true(ink > 0, "render/style: the regular face draws");
    check::is_true(painted(render_one(bold, base), base) > ink * 11 / 10,
                   "render/style: bold adds ink (" + std::to_string(ink) + " regular)");
    check::is_true(render_one(italic, base).rgb != render_one(regular, base).rgb,
                   "render/style: italic changes the glyphs");
}

void test_underline_draws_below_the_baseline_and_inside_the_handle() {
    // Capitals only, so nothing the plain text draws reaches below its
    // baseline, and any ink lower down is the underline.
    const images::Picture base = solid(320, 240, 0, 0, 0);
    overlay::TextItem plain;
    plain.text = "WAVE";
    plain.size = 0.2;
    plain.stroke_width = 0.0;
    overlay::TextItem underlined = plain;
    underlined.underline = true;

    const overlay::Bbox plain_ink = painted_bbox(render_one(plain, base), base);
    const images::Picture out = render_one(underlined, base);
    const overlay::Bbox ink = painted_bbox(out, base);
    check::is_true(ink.y + ink.h > plain_ink.y + plain_ink.h,
                   "render/style: the underline adds ink below the baseline");

    // `item_bbox` extends to cover it: the handle must not clip a line
    // the renderer draws.
    const overlay::Bbox handle =
        overlay::item_bbox(base.width, base.height, overlay::Item{underlined});
    check::is_true(ink.x >= handle.x && ink.y >= handle.y &&
                       ink.x + ink.w <= handle.x + handle.w &&
                       ink.y + ink.h <= handle.y + handle.h,
                   "render/style: an underline lies inside item_bbox");
}

void test_outline_text_draws_the_stroke_and_not_the_fill() {
    const images::Picture base = solid(320, 240, 0, 0, 0);
    overlay::TextItem filled;
    filled.text = "WAVE";
    filled.size = 0.3;
    filled.color = "#ff0000";
    filled.stroke_color = "#00ff00";
    overlay::TextItem outline = filled;
    outline.fill_kind = "none";

    // Glyph interiors are exactly the fill colour; anti-aliasing only
    // blends at the edges.
    const Rgb red{255, 0, 0};
    const images::Picture solid_out = render_one(filled, base);
    const images::Picture outline_out = render_one(outline, base);
    check::is_true(count_exact(solid_out, red) > 0, "render/style: a solid fill is there");
    check::equal(count_exact(outline_out, red), 0, "render/style: \"none\" draws no fill");
    check::is_true(painted(outline_out, base) > 0, "render/style: but it draws the stroke");

    outline.stroke_width = 0.0;
    check::equal(painted(render_one(outline, base), base), 0,
                 "render/style: no fill and no stroke draws nothing");
}

void test_outline_text_is_hollow() {
    // **The check above cannot see this, and did not.** A Qt pen
    // straddles the glyph path, so with no fill over it the pen's inner
    // half paints the glyph interiors in the *stroke* colour -- no fill
    // colour anywhere, test passed, and "none" rendered as solid
    // letters. What "outlined text" means is that the inside of every
    // stem is background, so that is what is asserted: take the pixels
    // well inside the ink of a filled render (an exact-fill pixel whose
    // eight neighbours are exact-fill too) and require every one of them
    // untouched in the outline render. A regular face at a stroke wide
    // enough to swallow its stems is the case that failed.
    const images::Picture base = solid(640, 480, 0, 0, 0);
    overlay::TextItem filled;
    filled.text = "HOLLOW";
    filled.size = 0.3;
    filled.color = "#ff0000";
    filled.stroke_width = 0.0;
    const images::Picture ink = render_one(filled, base);

    overlay::TextItem outline = filled;
    outline.fill_kind = "none";
    outline.stroke_color = "#00ff00";
    outline.stroke_width = 0.06;
    const images::Picture out = render_one(outline, base);

    const Rgb red{255, 0, 0};
    int interior = 0;
    int painted_inside = 0;
    for (int y = 1; y + 1 < ink.height; ++y) {
        for (int x = 1; x + 1 < ink.width; ++x) {
            bool deep = true;
            for (int dy = -1; dy <= 1 && deep; ++dy)
                for (int dx = -1; dx <= 1 && deep; ++dx)
                    deep = same(pixel(ink, x + dx, y + dy), red);
            if (!deep) continue;
            ++interior;
            if (!same(pixel(out, x, y), pixel(base, x, y))) ++painted_inside;
        }
    }
    check::is_true(interior > 1000, "render/style: the glyphs have an interior to check (" +
                                        std::to_string(interior) + " px)");
    check::equal(painted_inside, 0, "render/style: outline-only text is hollow");
    check::is_true(painted(out, base) > 0, "render/style: and the outline is drawn");
}

void test_a_linear_text_gradient_runs_counter_clockwise_from_its_angle() {
    const images::Picture base = solid(640, 240, 0, 0, 0);
    const overlay::TextItem across = gradient_text("MMMMM", 0.3);
    const images::Picture out = render_one(across, base);
    const overlay::Bbox ink = painted_bbox(out, base);
    const int third = ink.w / 3;
    const Lean left = lean(out, base, ink.x, ink.y, ink.x + third, ink.y + ink.h);
    const Lean right = lean(out, base, ink.x + ink.w - third, ink.y, ink.x + ink.w,
                            ink.y + ink.h);
    check::is_true(left.ink > 0 && right.ink > 0, "render/gradient: ink at both ends");
    check::is_true(left.value > 0.0, "render/gradient: angle 0 starts red on the left");
    check::is_true(right.value < 0.0, "render/gradient: and ends blue on the right");

    // **90 runs bottom to top**, the counter-clockwise sense a rect's
    // gradient and every item's rotation use. A clockwise implementation
    // passes the check above and puts red at the top here.
    overlay::TextItem upward = across;
    upward.fill_angle = 90.0;
    const images::Picture up = render_one(upward, base);
    const int half = ink.h / 2;
    const Lean top = lean(up, base, ink.x, ink.y, ink.x + ink.w, ink.y + half);
    const Lean bottom = lean(up, base, ink.x, ink.y + half, ink.x + ink.w, ink.y + ink.h);
    check::is_true(top.ink > 0 && bottom.ink > 0, "render/gradient: ink top and bottom");
    check::is_true(bottom.value > top.value,
                   "render/gradient: angle 90 is red at the bottom (counter-clockwise)");
}

void test_a_radial_text_gradient_is_centred() {
    const images::Picture base = solid(640, 240, 0, 0, 0);
    overlay::TextItem radial = gradient_text("MMMMM", 0.3);
    radial.fill_gradient = "radial";
    const images::Picture out = render_one(radial, base);
    const overlay::Bbox ink = painted_bbox(out, base);
    const int third = ink.w / 3;
    const Lean left = lean(out, base, ink.x, ink.y, ink.x + third, ink.y + ink.h);
    const Lean middle = lean(out, base, ink.x + third, ink.y, ink.x + ink.w - third,
                             ink.y + ink.h);
    const Lean right = lean(out, base, ink.x + ink.w - third, ink.y, ink.x + ink.w,
                            ink.y + ink.h);
    check::is_true(middle.value > left.value && middle.value > right.value,
                   "render/gradient: radial is reddest in the middle, not at one end");
    check::is_true(out.rgb != render_one(gradient_text("MMMMM", 0.3), base).rgb,
                   "render/gradient: radial is not the linear ramp");
}

void test_a_text_gradient_turns_with_a_rotated_item() {
    // Rotated a quarter turn counter-clockwise about its centre, the
    // text's own left-to-right ramp runs bottom to top on screen. Built
    // in unrotated screen space instead, it would stay left to right
    // across a now-narrow column and the two halves would barely differ.
    const images::Picture base = solid(480, 480, 0, 0, 0);
    overlay::TextItem item = gradient_text("MMMMM", 0.2);
    item.anchor = "mm";
    item.x = 0.5;
    item.y = 0.5;
    item.rotation = 90.0;
    const images::Picture out = render_one(item, base);
    const overlay::Bbox ink = painted_bbox(out, base);
    check::is_true(ink.h > ink.w, "render/gradient: the rotated text stands upright");
    const int half = ink.h / 2;
    const Lean top = lean(out, base, ink.x, ink.y, ink.x + ink.w, ink.y + half);
    const Lean bottom = lean(out, base, ink.x, ink.y + half, ink.x + ink.w, ink.y + ink.h);
    check::is_true(bottom.value > 0.0 && top.value < 0.0,
                   "render/gradient: the ramp turned with the text");
}

void test_a_radial_rect_is_centred_and_so_is_its_stroke() {
    const images::Picture base = solid(200, 100, 0, 0, 0);
    overlay::RectItem rect;
    rect.x = 0.0;
    rect.y = 0.0;
    rect.width = 1.0;
    rect.height = 1.0;
    rect.fill_kind = "gradient";
    rect.fill_color = "#ff0000";
    rect.fill_color2 = "#0000ff";
    rect.fill_gradient = "radial";
    const images::Picture out = render_one(rect, base);
    const Rgb centre = pixel(out, 100, 50);
    const Rgb corner = pixel(out, 0, 0);
    check::is_true(centre.r > 200 && centre.b < 55,
                   "render/rect: radial starts at the first colour in the centre");
    check::is_true(corner.b > 200 && corner.r < 55,
                   "render/rect: and reaches the second at the corners");
    // The linear ramp is at its midpoint there, which is what makes the
    // centre check above specific to radial.
    rect.fill_gradient = "linear";
    const Rgb linear_centre = pixel(render_one(rect, base), 100, 50);
    check::is_true(std::abs(linear_centre.r - linear_centre.b) < 40,
                   "render/rect: (a linear ramp is half-way at the centre)");

    overlay::RectItem outline;
    outline.x = 0.1;
    outline.y = 0.1;
    outline.width = 0.8;
    outline.height = 0.8;
    outline.stroke_kind = "gradient";
    outline.stroke_color = "#ff0000";
    outline.stroke_color2 = "#0000ff";
    outline.stroke_width = 0.05;
    const images::Picture linear_stroke = render_one(outline, base);
    outline.stroke_gradient = "radial";
    check::is_true(render_one(outline, base).rgb != linear_stroke.rgb,
                   "render/rect: a stroke can be radial too");
}

// A font file this machine is known to have, or empty.
std::string known_font_file() {
    for (const char* path : {"/usr/share/fonts/TTF/DejaVuSans.ttf",
                             "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                             "/usr/share/fonts/dejavu/DejaVuSans.ttf",
                             "/Library/Fonts/Arial.ttf",
                             "C:/Windows/Fonts/arial.ttf"}) {
        if (QFile::exists(QString::fromUtf8(path))) return path;
    }
    return std::string();
}

void test_a_family_is_requested_and_a_font_file_still_wins() {
    // A monospace "i" is as wide as an "m", so the request shows up in
    // the measured width wherever a fixed-pitch face exists -- and the
    // style hint is what finds one on platforms whose databases do not
    // know the generic keywords as family names.
    overlay::TextItem sans;
    sans.text = "iiiiii";
    overlay::TextItem mono = sans;
    mono.font_family = "monospace";
    const int sans_w = overlay::item_bbox(640, 480, overlay::Item{sans}).w;
    const int mono_w = overlay::item_bbox(640, 480, overlay::Item{mono}).w;
    check::is_true(mono_w > sans_w, "render/font: \"monospace\" is honoured (" +
                                        std::to_string(mono_w) + " vs " +
                                        std::to_string(sans_w) + ")");

    const std::string file = known_font_file();
    if (file.empty()) {
        std::fprintf(stderr, "SKIP render/font: no known font file on this machine, "
                             "so font-beats-family is untested here\n");
        return;
    }
    overlay::TextItem from_file = sans;
    from_file.font = file;
    overlay::TextItem both = from_file;
    both.font_family = "monospace";
    check::equal(overlay::item_bbox(640, 480, overlay::Item{both}).w,
                 overlay::item_bbox(640, 480, overlay::Item{from_file}).w,
                 "render/font: a font file wins over a family request");
}

}  // namespace

int main(int argc, char** argv) {
    check::report_crashes_instead_of_prompting();
    // Qt's font database is platform integration, so text needs an
    // application object; offscreen so this runs on a CI box with no
    // display. Set before constructing, not in the environment of
    // whatever launched us.
    qputenv("QT_QPA_PLATFORM", "offscreen");
    const QGuiApplication app(argc, argv);

    test_an_empty_document_changes_nothing();
    test_an_image_inset_lands_where_the_document_says();
    test_the_anchor_moves_the_inset_not_the_picture();
    test_item_bbox_predicts_what_render_draws();
    test_a_border_surrounds_the_inset();
    test_a_missing_source_draws_nothing();
    test_text_is_drawn_and_bounded_where_predicted();
    test_empty_text_still_has_a_handle();
    test_items_draw_back_to_front();
    test_a_file_inset_is_decoded_once();
    test_a_solid_rect_fills_its_bbox_and_nothing_else();
    test_a_rect_with_no_fill_or_stroke_draws_nothing();
    test_a_gradient_rect_interpolates_between_its_two_colours();
    test_a_stroke_only_rect_draws_an_outline_not_a_fill();
    test_rect_json_roundtrips_through_the_model();
    test_solid_text_never_reads_the_gradient_fields();
    test_bold_and_italic_change_the_glyphs();
    test_underline_draws_below_the_baseline_and_inside_the_handle();
    test_outline_text_draws_the_stroke_and_not_the_fill();
    test_outline_text_is_hollow();
    test_a_linear_text_gradient_runs_counter_clockwise_from_its_angle();
    test_a_radial_text_gradient_is_centred();
    test_a_text_gradient_turns_with_a_rotated_item();
    test_a_radial_rect_is_centred_and_so_is_its_stroke();
    test_a_family_is_requested_and_a_font_file_still_wins();

    return check::report("overlay rendering");
}

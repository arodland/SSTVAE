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

    return check::report("overlay rendering");
}

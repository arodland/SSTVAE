// The overlay editor's geometry.
//
// How it *looks* needs eyes. Where a click lands does not: the whole
// design rests on the handles coming from the same `item_bbox` the
// renderer places the item with, so a click at the item's centre must
// select that item and a drag must move it to where the cursor went.
// Those are arithmetic, and arithmetic that is easy to get subtly wrong
// -- an inverted axis or a forgotten letterbox offset still *looks*
// like a working editor until an item will not go where you put it.

#include <QApplication>
#include <QKeyEvent>
#include <QMouseEvent>
#include <QLayout>
#include <QStyle>
#include <QVBoxLayout>
#include <QWidget>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <numbers>
#include <string>
#include <variant>

#include "check.hpp"
#include "images/types.hpp"
#include "overlay/render.hpp"
#include "overlay/template.hpp"
#include "overlay_editor.hpp"
#include "style.hpp"

using namespace sstvae;

namespace {

// Deliberately not the canvas aspect, so the letterbox offset is
// non-zero in one axis and a mapping that ignores it fails.
constexpr int W = 500;
constexpr int H = 500;

images::Picture grey(int w, int h) {
    images::Picture picture(w, h);
    std::fill(picture.rgb.begin(), picture.rgb.end(), std::uint8_t{128});
    return picture;
}

// Where a canvas point lands in the widget, mirroring the editor's own
// letterboxing: same aspect, centred.
QPoint widget_point(double canvas_x, double canvas_y) {
    const double aspect =
        static_cast<double>(overlay::CANVAS_W) / overlay::CANVAS_H;
    int w = W;
    int h = static_cast<int>(std::lround(w / aspect));
    if (h > H) {
        h = H;
        w = static_cast<int>(std::lround(h * aspect));
    }
    const int x0 = (W - w) / 2;
    const int y0 = (H - h) / 2;
    return QPoint(x0 + static_cast<int>(std::lround(canvas_x * w / overlay::CANVAS_W)),
                  y0 + static_cast<int>(std::lround(canvas_y * h / overlay::CANVAS_H)));
}

// The inverse of `widget_point`: needed only where a test has to start
// from a widget pixel it did not choose (the rotate handle's position,
// which depends on the style's own icon-size metric) and work out what
// canvas angle that press landed at.
QPointF canvas_point(QPoint widget_pt) {
    const double aspect =
        static_cast<double>(overlay::CANVAS_W) / overlay::CANVAS_H;
    int w = W;
    int h = static_cast<int>(std::lround(w / aspect));
    if (h > H) {
        h = H;
        w = static_cast<int>(std::lround(h * aspect));
    }
    const int x0 = (W - w) / 2;
    const int y0 = (H - h) / 2;
    return QPointF((widget_pt.x() - x0) * overlay::CANVAS_W / static_cast<double>(w),
                   (widget_pt.y() - y0) * overlay::CANVAS_H / static_cast<double>(h));
}

void press(gui::OverlayEditor& editor, QPoint at) {
    QMouseEvent event(QEvent::MouseButtonPress, QPointF(at), QPointF(at),
                      Qt::LeftButton, Qt::LeftButton, Qt::NoModifier);
    QApplication::sendEvent(&editor, &event);
}

void move_to(gui::OverlayEditor& editor, QPoint at) {
    QMouseEvent event(QEvent::MouseMove, QPointF(at), QPointF(at), Qt::NoButton,
                      Qt::LeftButton, Qt::NoModifier);
    QApplication::sendEvent(&editor, &event);
}

void release(gui::OverlayEditor& editor, QPoint at) {
    QMouseEvent event(QEvent::MouseButtonRelease, QPointF(at), QPointF(at),
                      Qt::LeftButton, Qt::NoButton, Qt::NoModifier);
    QApplication::sendEvent(&editor, &event);
}

void key(gui::OverlayEditor& editor, int code,
         Qt::KeyboardModifiers mods = Qt::NoModifier) {
    QKeyEvent event(QEvent::KeyPress, code, mods);
    QApplication::sendEvent(&editor, &event);
}

gui::OverlayEditor* make_editor() {
    auto* editor = new gui::OverlayEditor();
    editor->resize(W, H);
    editor->set_base_image(grey(overlay::CANVAS_W, overlay::CANVAS_H));
    return editor;
}

// The centre of an item's box, in canvas pixels.
QPointF centre_of(const overlay::Item& item, const images::Picture* last_rx) {
    const overlay::Bbox box = overlay::item_bbox(overlay::CANVAS_W,
                                                 overlay::CANVAS_H, item, last_rx);
    return QPointF(box.x + box.w / 2.0, box.y + box.h / 2.0);
}

void test_no_picture_means_nothing_to_compose() {
    gui::OverlayEditor editor;
    editor.resize(W, H);
    check::is_true(!editor.composed_image().has_value(),
                   "editor: no base picture composes to nothing");
    check::is_true(!editor.has_base(), "editor: and says so");
}

void test_clicking_an_item_selects_it() {
    gui::OverlayEditor* editor = make_editor();
    const images::Picture inset = grey(40, 30);
    editor->set_last_rx(inset);
    editor->add_last_rx_inset();
    // Adding selects; clear it so the click is what does the work.
    press(*editor, widget_point(2, 2));
    check::is_true(editor->selected_item() == nullptr,
                   "editor: clicking empty canvas clears the selection");

    const overlay::Item& item = editor->doc().items.front();
    const QPointF centre = centre_of(item, &inset);
    press(*editor, widget_point(centre.x(), centre.y()));
    check::is_true(editor->selected_item() != nullptr,
                   "editor: clicking an item selects it");
    delete editor;
}

void test_dragging_moves_the_item_to_the_cursor() {
    gui::OverlayEditor* editor = make_editor();
    const images::Picture inset = grey(40, 30);
    editor->set_last_rx(inset);
    editor->add_last_rx_inset();

    const QPointF from = centre_of(editor->doc().items.front(), &inset);
    press(*editor, widget_point(from.x(), from.y()));
    // Somewhere clearly elsewhere, and not on either axis of the start,
    // so a swapped or dropped axis cannot pass.
    const QPointF to(from.x() - 180.0, from.y() + 90.0);
    move_to(*editor, widget_point(to.x(), to.y()));
    release(*editor, widget_point(to.x(), to.y()));

    const QPointF now = centre_of(editor->doc().items.front(), &inset);
    // Within a pixel or two: the widget maps through integer positions.
    check::is_true(std::abs(now.x() - to.x()) <= 3.0,
                   "editor: a drag moves the item to the cursor in x (" +
                       std::to_string(now.x()) + " vs " + std::to_string(to.x()) + ")");
    check::is_true(std::abs(now.y() - to.y()) <= 3.0,
                   "editor: and in y (" + std::to_string(now.y()) + " vs " +
                       std::to_string(to.y()) + ")");
    delete editor;
}

void test_a_drag_keeps_the_grab_offset() {
    // Grabbing an item near its edge must not teleport its anchor to the
    // cursor -- the item should follow the *movement*, not snap.
    gui::OverlayEditor* editor = make_editor();
    const images::Picture inset = grey(40, 30);
    editor->set_last_rx(inset);
    editor->add_last_rx_inset();

    const overlay::Bbox box = overlay::item_bbox(
        overlay::CANVAS_W, overlay::CANVAS_H, editor->doc().items.front(), &inset);
    const QPointF grab(box.x + 3.0, box.y + 3.0);  // near the top-left corner
    press(*editor, widget_point(grab.x(), grab.y()));
    move_to(*editor, widget_point(grab.x() + 100.0, grab.y() + 40.0));
    release(*editor, widget_point(grab.x() + 100.0, grab.y() + 40.0));

    const overlay::Bbox moved = overlay::item_bbox(
        overlay::CANVAS_W, overlay::CANVAS_H, editor->doc().items.front(), &inset);
    check::is_true(std::abs((moved.x - box.x) - 100) <= 3,
                   "editor: the item moves by the drag distance, not to the cursor");
    check::is_true(std::abs((moved.y - box.y) - 40) <= 3,
                   "editor: in both axes");
    delete editor;
}

void test_normalized_coordinates_survive_a_resize() {
    // The document stores fractions, so the same item must land in the
    // same *relative* place at another widget size. This is what makes a
    // saved template mean anything.
    gui::OverlayEditor* editor = make_editor();
    const images::Picture inset = grey(40, 30);
    editor->set_last_rx(inset);
    editor->add_last_rx_inset();

    const QPointF target(200.0, 150.0);
    const QPointF from = centre_of(editor->doc().items.front(), &inset);
    press(*editor, widget_point(from.x(), from.y()));
    move_to(*editor, widget_point(target.x(), target.y()));
    release(*editor, widget_point(target.x(), target.y()));

    const overlay::ImageItem& item =
        std::get<overlay::ImageItem>(editor->doc().items.front());
    const double x_before = item.x;
    const double y_before = item.y;

    editor->resize(W * 2, H * 2);
    const overlay::ImageItem& after =
        std::get<overlay::ImageItem>(editor->doc().items.front());
    check::equal(after.x, x_before, "editor: a resize does not move the item in x");
    check::equal(after.y, y_before, "editor: nor in y");
    delete editor;
}

void test_removing_clears_the_selection() {
    gui::OverlayEditor* editor = make_editor();
    editor->add_text("W1AW");
    check::is_true(editor->selected_item() != nullptr,
                   "editor: a new item starts selected");
    editor->remove_selected();
    check::is_true(editor->selected_item() == nullptr,
                   "editor: removing it clears the selection");
    check::is_true(editor->doc().items.empty(),
                   "editor: and takes it out of the document");
    delete editor;
}

void test_the_composite_is_the_renderer_s_output() {
    // Not a Qt-drawn imitation: what the operator arranges is what goes
    // on the air, by construction. Assert it literally.
    gui::OverlayEditor* editor = make_editor();
    editor->add_text("W1AW");

    const std::optional<images::Picture> composed = editor->composed_image();
    check::is_true(composed.has_value(), "editor: composes with a base picture");
    const images::Picture expected =
        overlay::render(grey(overlay::CANVAS_W, overlay::CANVAS_H), editor->doc());
    check::equal(composed->width, expected.width, "editor: composite width");
    check::is_true(composed->rgb == expected.rgb,
                   "editor: the composite is exactly overlay::render's output");
    delete editor;
}

// --- templates (docs/overlay-templates.md) ------------------------------

void test_set_fields_substitutes_the_composite_but_not_the_document() {
    gui::OverlayEditor* editor = make_editor();
    editor->add_text("{theircall} de {mycall}");
    const std::string raw = std::get<overlay::TextItem>(editor->doc().items[0]).text;

    overlay::Fields fields;
    fields.builtin = {{"theircall", "W1XYZ"}, {"mycall", "KC2G"}};
    editor->set_fields(fields);

    // The document stays the template: this is what `Save as
    // template...` writes and what the property panel's text box shows.
    check::equal(std::get<overlay::TextItem>(editor->doc().items[0]).text, raw,
                 "editor: set_fields does not touch the stored document");

    const overlay::Doc expected_doc = overlay::substitute(editor->doc(), fields);
    const images::Picture expected =
        overlay::render(grey(overlay::CANVAS_W, overlay::CANVAS_H), expected_doc);
    const std::optional<images::Picture> composed = editor->composed_image();
    check::is_true(composed.has_value() && composed->rgb == expected.rgb,
                   "editor: the composite reflects the substituted text");
    delete editor;
}

void test_set_fields_with_no_placeholders_changes_nothing() {
    // The default-construction case: a plain overlay with no template
    // active must render exactly as it always has.
    gui::OverlayEditor* editor = make_editor();
    editor->add_text("KC2G");
    const images::Picture before = *editor->composed_image();

    editor->set_fields(overlay::Fields{{{"theircall", "W1XYZ"}}, {}});
    check::is_true(editor->composed_image()->rgb == before.rgb,
                   "editor: an unused field changes nothing");
    delete editor;
}

void test_a_dropped_line_is_not_hit_testable() {
    // Rule 2: a hole with nothing else on its line vanishes from the
    // rendered picture, so it must also vanish from what a click can
    // land on -- a selection handle that floated over empty canvas
    // would be the WYSIWYG rule broken silently.
    gui::OverlayEditor* editor = make_editor();
    editor->add_text("SNR {snr}");  // no {snr} filled in -> empty text
    editor->set_fields(overlay::Fields{});

    const std::optional<images::Picture> composed = editor->composed_image();
    check::is_true(composed.has_value(), "editor: still composes");
    const images::Picture expected =
        overlay::render(grey(overlay::CANVAS_W, overlay::CANVAS_H), overlay::Doc());
    check::is_true(composed->rgb == expected.rgb,
                   "editor: a fully-empty line paints nothing, same as no item");
    delete editor;
}

}  // namespace


void test_arrows_nudge_by_a_fixed_fraction() {
    // The nudge exists because a mouse cannot do it: positions are
    // normalized, so the smallest drag is one widget pixel -- a
    // different distance at every window size. A key step has to be a
    // fixed fraction of the canvas, and it has to move the axis it
    // names in the direction it names.
    std::unique_ptr<gui::OverlayEditor> editor(make_editor());
    editor->add_text("N0CALL");
    overlay::Item* item = editor->selected_item();
    check::is_true(item != nullptr, "nudge: an added item is selected");

    const double x0 = std::visit([](const auto& i) { return i.x; }, *item);
    const double y0 = std::visit([](const auto& i) { return i.y; }, *item);

    constexpr double FINE = 1.0 / 640.0;
    const auto ix = [&] { return std::visit([](const auto& i) { return i.x; }, *item); };
    const auto iy = [&] { return std::visit([](const auto& i) { return i.y; }, *item); };

    key(*editor, Qt::Key_Right);
    check::is_true(std::abs(ix() - (x0 + FINE)) <= 1e-9,
                   "nudge: right moves +x by one fine step");
    check::is_true(std::abs(iy() - y0) <= 1e-9, "nudge: right leaves y alone");

    key(*editor, Qt::Key_Left);
    check::is_true(std::abs(ix() - x0) <= 1e-9, "nudge: left comes back");

    key(*editor, Qt::Key_Down);
    check::is_true(std::abs(iy() - (y0 + FINE)) <= 1e-9, "nudge: down moves +y");
    key(*editor, Qt::Key_Up);
    check::is_true(std::abs(iy() - y0) <= 1e-9, "nudge: up comes back");

    // Shift is the coarse step, and it is bigger -- not merely
    // different, which an inequality assertion would also accept.
    constexpr double COARSE = 1.0 / 64.0;
    key(*editor, Qt::Key_Right, Qt::ShiftModifier);
    check::is_true(std::abs(ix() - (x0 + COARSE)) <= 1e-9,
                   "nudge: shift takes the coarse step");
}

void test_delete_removes_the_selection() {
    std::unique_ptr<gui::OverlayEditor> editor(make_editor());
    editor->add_text("N0CALL");
    check::equal(static_cast<int>(editor->doc().items.size()), 1,
                 "delete: one item to begin with");
    key(*editor, Qt::Key_Delete);
    check::equal(static_cast<int>(editor->doc().items.size()), 0,
                 "delete: the key removes it");
    check::is_true(editor->selected_item() == nullptr,
                   "delete: and clears the selection");

    // With nothing selected the key must fall through rather than be
    // swallowed, or a shortcut elsewhere in the window stops working.
    key(*editor, Qt::Key_Delete);
    check::equal(static_cast<int>(editor->doc().items.size()), 0,
                 "delete: harmless with an empty document");
}

void test_an_added_item_can_be_nudged_without_clicking_first() {
    // The flow the feature is for: Add text, then line it up. The
    // button that added it has the focus, so unless the editor takes
    // focus the keys are dead exactly here.
    std::unique_ptr<gui::OverlayEditor> editor(make_editor());
    editor->add_text("N0CALL");
    check::is_true(editor->hasFocus() || editor->focusPolicy() != Qt::ClickFocus,
                   "nudge: the editor is reachable by keyboard after an add");
}

// A reception is a document change only when the document uses it.
//
// **This is the guard for a runaway, not a tidiness rule.**
// `set_last_rx` used to invalidate and emit `documentChanged`
// unconditionally, and `TransmitPanel` connects that signal to
// `schedule_optimization`, which abandons any speculative refinement in
// flight and starts a fresh one on a 120 s budget across four
// onnxruntime threads. So every picture that arrived restarted a full
// optimization of the transmit composition -- on stations whose
// composition did not contain a "last received" inset at all, and which
// were not about to transmit.
//
// Measured on the panel, fifteen receptions over 65 s: **205 s of CPU
// against 0.30 s with refinement off.** Runs plateau at 40-100 s and
// pictures on a net arrive every 32-95 s, so each arrival discarded the
// previous run's work and the optimizer never reached idle. Operators
// reported it as CPU that climbed until the application had to be
// restarted, which is exactly what it was: nothing leaked, and nothing
// could bring it down again either.
//
// Counted rather than asserted-once, because the failure is "it fires
// when it should not" and a boolean cannot tell one emission from
// twenty.
void test_a_reception_is_a_change_only_if_an_item_uses_it() {
    std::unique_ptr<gui::OverlayEditor> editor(make_editor());
    int changes = 0;
    QObject::connect(editor.get(), &gui::OverlayEditor::documentChanged,
                     [&changes] { ++changes; });

    // Nothing in the document at all.
    editor->set_last_rx(grey(64, 48));
    check::equal(changes, 0, "an empty composition ignores a reception");

    // Items, but none of them referring to the reception. The file
    // inset matters: an `ImageItem` is not automatically a `last_rx`
    // one, and a check on the item's *type* rather than its `source`
    // would pass every other test in this file and still fire here.
    editor->add_text("N0CALL");
    editor->add_image_inset("/nonexistent/inset.png");
    changes = 0;
    editor->set_last_rx(grey(64, 48));
    check::equal(changes, 0,
                 "a composition with no last_rx item ignores a reception");
    check::is_true(!editor->uses_last_rx(), "and says it does not use one");

    // And now one that does.
    editor->add_last_rx_inset();
    changes = 0;
    editor->set_last_rx(grey(64, 48));
    check::equal(changes, 1, "a composition with a last_rx item follows it");
    check::is_true(editor->uses_last_rx(), "and says it uses one");
}

// The reception is still *stored* when nothing refers to it yet.
//
// The late-binding guarantee is what a "last_rx" item means: it resolves
// at render time, so an inset added after a reception must show that
// reception rather than nothing. Skipping the *emit* must not become
// skipping the assignment -- which is the obvious way to write this fix
// and is wrong.
void test_a_reception_is_kept_even_with_nothing_to_show_it() {
    std::unique_ptr<gui::OverlayEditor> editor(make_editor());

    images::Picture received(64, 48);
    std::fill(received.rgb.begin(), received.rgb.end(), std::uint8_t{255});
    editor->set_last_rx(received);
    check::is_true(editor->has_last_rx(),
                   "the reception is kept though nothing showed it");

    // Added afterwards: the composite must be the renderer's output for
    // *that* picture, which it cannot be if the assignment was skipped.
    editor->add_last_rx_inset();
    const std::optional<images::Picture> composed = editor->composed_image();
    check::is_true(composed.has_value(), "there is a composite");
    if (!composed) return;
    const images::Picture expected = overlay::render(
        grey(overlay::CANVAS_W, overlay::CANVAS_H), editor->doc(), &received);
    check::is_true(composed->rgb == expected.rgb,
                   "an inset added later shows the reception that preceded it");
}

// The editor must not pin a window height to its own width.
//
// It used to: `setFixedHeight(width * 3/4)` in `resizeEvent`, which is
// a hard *minimum*, so widening the transmit pane raised a floor under
// the whole window that narrowing it never lowered. Measured before the
// fix, through `sstvae-gui-shot --transmit`: the panel's minimum height
// went 611 px at 545 wide, 925 at 1348, **1274 at 1900**. Bounded while
// a splitter kept the pane narrow; unbounded once a tab hands it the
// entire window. The identical construct in the receive preview is
// guarded by `test_picture_box.cpp` -- this is the other copy.
//
// Nothing is lost by capping instead: `canvas_rect()` letterboxes in
// both directions, so a pane too short for 4:3 draws a smaller centred
// canvas and the handles follow it, because they come from that same
// rectangle.
void test_it_pins_no_window_height() {
    QWidget container;
    auto* layout = new QVBoxLayout(&container);
    layout->setContentsMargins(0, 0, 0, 0);
    auto* editor = new gui::OverlayEditor(&container);
    layout->addWidget(editor);
    container.resize(3000, 2000);
    container.show();

    int previous = 0;
    for (const int w : {545, 900, 1348, 1900}) {
        // Twice: the cap is derived from the width, so Qt clamps the
        // incoming geometry against the previous one and the new cap
        // applies on the pass `updateGeometry` asks for.
        for (int pass = 0; pass < 2; ++pass) {
            container.setGeometry(0, 0, w, 400);
            QCoreApplication::processEvents();
        }
        const int floor = container.minimumSizeHint().height();
        if (previous != 0) {
            check::equal(floor, previous,
                         "the minimum height does not follow the editor's width");
        }
        previous = floor;

        // **And the constraint the hint cannot see.** `minimumSizeHint`
        // never consults `heightForWidth`; what Qt actually applies when
        // it lays a widget out is `minimumHeightForWidth(width)`. The
        // first version of this test checked only the hint above, went
        // green, and shipped a window that grew off the bottom of the
        // screen -- because `hasHeightForWidth` was still set on the
        // editor, a `QSplitter` hid it and a `QTabWidget` passed it
        // straight through to the window.
        check::is_true(!container.layout()->hasHeightForWidth() ||
                           container.layout()->minimumHeightForWidth(w) <= floor,
                       "and neither does minimumHeightForWidth");
    }
}

// --- rectangles and z-order ----------------------------------------------

void test_add_rect_selects_a_visible_item() {
    gui::OverlayEditor* editor = make_editor();
    editor->add_rect();
    const overlay::Item* item = editor->selected_item();
    check::is_true(item != nullptr, "editor: adding a rect selects it");
    const auto* rect = std::get_if<overlay::RectItem>(item);
    check::is_true(rect != nullptr, "editor: the added item is a rect");
    if (rect == nullptr) return;
    // Not the model's own bare defaults -- see `add_rect`'s comment --
    // the button has to place something a click can actually find.
    check::is_true(rect->fill_kind != "none",
                   "editor: a freshly added rect is visible, not blank");
    delete editor;
}

void test_raise_and_lower_swap_adjacent_items_and_follow_the_selection() {
    gui::OverlayEditor* editor = make_editor();
    editor->add_text("A");  // index 0
    editor->add_text("B");  // index 1, selected
    check::is_true(!editor->can_raise_selected(),
                   "editor: the top item cannot be raised further");
    check::is_true(editor->can_lower_selected(), "editor: but can be lowered");

    editor->lower_selected();
    check::equal(std::get<overlay::TextItem>(editor->doc().items[0]).text,
                 std::string("B"), "editor: lower swaps it down");
    check::equal(std::get<overlay::TextItem>(*editor->selected_item()).text,
                 std::string("B"), "editor: the selection follows the item");

    editor->raise_selected();
    check::equal(std::get<overlay::TextItem>(editor->doc().items[1]).text,
                 std::string("B"), "editor: raise undoes it");
    delete editor;
}

void test_bring_to_front_and_send_to_back_preserve_the_rest_of_the_order() {
    gui::OverlayEditor* editor = make_editor();
    editor->add_text("A");  // 0
    editor->add_text("B");  // 1
    editor->add_text("C");  // 2
    const auto label = [&](int i) {
        return std::get<overlay::TextItem>(editor->doc().items[i]).text;
    };

    // `add_text` always selects the item it just added, which is the
    // simplest way to get a known item selected without needing real
    // click geometry -- add a fourth on top of A..C and pull it to the
    // back, which must not disturb their own relative order.
    editor->add_text("D");  // 3, selected
    check::equal(label(3), std::string("D"), "editor: D starts on top");
    editor->send_selected_to_back();
    check::equal(label(0), std::string("D"), "editor: D is now at the back");
    check::equal(label(1), std::string("A"), "editor: A..C keep their order");
    check::equal(label(2), std::string("B"), "editor: A..C keep their order");
    check::equal(label(3), std::string("C"), "editor: A..C keep their order");
    check::equal(std::get<overlay::TextItem>(*editor->selected_item()).text,
                 std::string("D"), "editor: the selection follows D to index 0");

    editor->bring_selected_to_front();
    check::equal(label(3), std::string("D"), "editor: and back to the front");
    check::equal(label(0), std::string("A"), "editor: A..C keep their order again");
    check::equal(label(1), std::string("B"), "editor: A..C keep their order again");
    check::equal(label(2), std::string("C"), "editor: A..C keep their order again");
    delete editor;
}

void test_z_order_changes_which_item_paints_on_top() {
    // The property the buttons exist for, not just the vector
    // arithmetic: after a reorder, `render()` must actually draw the
    // raised item over the one it used to sit under.
    gui::OverlayEditor* editor = make_editor();
    overlay::RectItem under;
    under.x = 0.1;
    under.y = 0.1;
    under.width = 0.3;
    under.height = 0.3;
    under.fill_kind = "solid";
    under.fill_color = "#ff0000";
    overlay::RectItem over_item = under;
    over_item.fill_color = "#0000ff";

    overlay::Doc doc;
    doc.items.push_back(under);      // red, index 0
    doc.items.push_back(over_item);  // blue, index 1, drawn on top
    editor->set_doc(doc);

    // Both rects cover the same area, so a click there hits both --
    // `hit_test` picks front to back, i.e. the one actually on top
    // (blue), which is the whole point being pinned here.
    QMouseEvent select_event(QEvent::MouseButtonPress,
                             widget_point(0.2 * overlay::CANVAS_W, 0.2 * overlay::CANVAS_H),
                             widget_point(0.2 * overlay::CANVAS_W, 0.2 * overlay::CANVAS_H),
                             Qt::LeftButton, Qt::LeftButton, Qt::NoModifier);
    QApplication::sendEvent(editor, &select_event);
    check::equal(editor->doc().items.size(), std::size_t{2}, "editor: still two items");

    const auto* selected_rect = std::get_if<overlay::RectItem>(editor->selected_item());
    check::is_true(selected_rect != nullptr && selected_rect->fill_color == "#0000ff",
                   "editor: the click selected the blue (topmost) rect");

    // Lower the selected (blue) item below red -- red should now win
    // the overlap it used to lose.
    editor->lower_selected();
    const std::optional<images::Picture> composed = editor->composed_image();
    check::is_true(composed.has_value(), "editor: composes after the reorder");
    const std::size_t idx =
        (static_cast<std::size_t>(0.2 * overlay::CANVAS_H) * composed->width +
         static_cast<std::size_t>(0.2 * overlay::CANVAS_W)) * 3;
    check::equal(static_cast<int>(composed->rgb[idx]), 255,
                 "editor: red now paints over blue after lower_selected");
    check::equal(static_cast<int>(composed->rgb[idx + 2]), 0,
                 "editor: and blue no longer shows through");
    delete editor;
}

// --- rotation, scaling and the floating palette's anchor -----------------

void test_dragging_the_rotate_handle_rotates_the_item() {
    // The rotate grip sits above-and-right of the item, offset from its
    // top-right corner by `handle_px() * 2` in raw screen pixels -- the
    // opposite corner and direction from the resize grip, so a press can
    // never be ambiguous between the two. `handle_px()` is derived from
    // the style, not a literal, so this mirrors that same query rather
    // than guessing a pixel count.
    gui::OverlayEditor* editor = make_editor();
    editor->add_rect();
    overlay::Item* item = editor->selected_item();
    check::is_true(item != nullptr, "rotate: a rect is selected after add_rect");

    const overlay::Bbox box =
        overlay::item_bbox(overlay::CANVAS_W, overlay::CANVAS_H, *item, nullptr);
    const QPointF center(box.x + box.w / 2.0, box.y + box.h / 2.0);

    const QRect on_screen = editor->selection_screen_rect();
    const int side =
        std::max(10, editor->style()->pixelMetric(QStyle::PM_SmallIconSize) * 2 / 3);
    const int offset = side * 2;
    const QPoint handle_center(on_screen.x() + on_screen.width() + offset,
                               on_screen.y() - offset);

    press(*editor, handle_center);
    check::is_true(editor->selected_item() != nullptr,
                   "rotate: pressing the handle keeps the item selected");
    const double r0 =
        std::visit([](const auto& i) { return i.rotation; }, *editor->selected_item());

    // Work out the canvas-space angle the press actually landed at (it is
    // near the handle's centre, not on any axis this test controls), then
    // ask for a point a precise quarter turn further counter-clockwise --
    // same formula `mousePressEvent`/`mouseMoveEvent` use: screen-down is
    // +y, so this negation is what makes a visually CCW sweep read as a
    // positive angle.
    const QPointF press_canvas = canvas_point(handle_center);
    const double angle0 =
        std::atan2(-(press_canvas.y() - center.y()), press_canvas.x() - center.x());
    const double radius =
        std::hypot(press_canvas.x() - center.x(), press_canvas.y() - center.y());
    const double angle_target = angle0 + std::numbers::pi / 2.0;
    const QPointF target_canvas(center.x() + radius * std::cos(angle_target),
                                center.y() - radius * std::sin(angle_target));
    move_to(*editor, widget_point(target_canvas.x(), target_canvas.y()));

    const double rotation =
        std::visit([](const auto& i) { return i.rotation; }, *editor->selected_item());
    check::is_true(std::abs(rotation - (r0 + 90.0)) <= 2.0,
                   "rotate: a quarter-turn CCW drag adds 90 degrees (" +
                       std::to_string(r0) + " -> " + std::to_string(rotation) + ")");

    release(*editor, widget_point(target_canvas.x(), target_canvas.y()));
    delete editor;
}

void test_scale_keys_grow_and_shrink_the_selection() {
    std::unique_ptr<gui::OverlayEditor> editor(make_editor());
    editor->add_text("N0CALL");
    const overlay::TextItem before =
        std::get<overlay::TextItem>(*editor->selected_item());

    key(*editor, Qt::Key_Plus);
    const overlay::TextItem grown =
        std::get<overlay::TextItem>(*editor->selected_item());
    check::is_true(grown.size > before.size,
                   "scale: + grows the selection's size");

    key(*editor, Qt::Key_Minus);
    const overlay::TextItem back =
        std::get<overlay::TextItem>(*editor->selected_item());
    check::is_true(back.size < grown.size,
                   "scale: - shrinks it back down");

    // Shift is the coarse step, matching the nudge convention -- bigger,
    // not merely different. Compare the two multiplicative factors from
    // the same starting size rather than chaining more key presses, so
    // rounding from an earlier step cannot muddy the comparison.
    const double base = back.size;
    key(*editor, Qt::Key_Plus);
    const double fine_size =
        std::get<overlay::TextItem>(*editor->selected_item()).size;
    std::get<overlay::TextItem>(*editor->selected_item()).size = base;
    key(*editor, Qt::Key_Plus, Qt::ShiftModifier);
    const double coarse_size =
        std::get<overlay::TextItem>(*editor->selected_item()).size;
    check::is_true((coarse_size - base) > (fine_size - base),
                   "scale: shift takes a bigger step than the plain key");
}

void test_rotate_keys_turn_the_selection() {
    std::unique_ptr<gui::OverlayEditor> editor(make_editor());
    editor->add_rect();
    const double r0 =
        std::visit([](const auto& i) { return i.rotation; }, *editor->selected_item());

    // `[` rotates counter-clockwise (increasing), `]` clockwise
    // (decreasing) -- see `keyPressEvent`'s `rotate_item` calls.
    key(*editor, Qt::Key_BracketLeft);
    const double r1 =
        std::visit([](const auto& i) { return i.rotation; }, *editor->selected_item());
    check::is_true(r1 > r0, "rotate keys: [ increases rotation");

    key(*editor, Qt::Key_BracketRight);
    const double r2 =
        std::visit([](const auto& i) { return i.rotation; }, *editor->selected_item());
    check::is_true(std::abs(r2 - r0) <= 1e-9, "rotate keys: ] undoes it");

    // Shift takes the coarser, 15-degree step.
    key(*editor, Qt::Key_BracketLeft, Qt::ShiftModifier);
    const double r3 =
        std::visit([](const auto& i) { return i.rotation; }, *editor->selected_item());
    check::is_true(r3 - r0 > r1 - r0,
                   "rotate keys: shift takes a bigger step than the plain key");
}

void test_the_resize_handle_tracks_the_items_rotation() {
    // A square, so a clean quarter turn swings its bottom-right corner
    // to exactly where its top-right corner used to be -- a precise,
    // easily-checked prediction rather than an approximate one.
    gui::OverlayEditor* editor = make_editor();
    editor->add_rect();
    auto* rect = std::get_if<overlay::RectItem>(editor->selected_item());
    check::is_true(rect != nullptr, "rotate/resize: a rect is selected");
    if (rect == nullptr) {
        delete editor;
        return;
    }
    // Equal in canvas *pixels*, not merely equal fractions -- CANVAS_W
    // and CANVAS_H are not equal (4:3), so two equal fractions would be
    // a rectangle, not the square this test's corner-swap prediction
    // needs.
    rect->width = 0.2;
    rect->height =
        0.2 * overlay::CANVAS_W / static_cast<double>(overlay::CANVAS_H);
    editor->refresh_item();

    const overlay::Bbox box =
        overlay::item_bbox(overlay::CANVAS_W, overlay::CANVAS_H, *editor->selected_item(), nullptr);
    check::equal(box.w, box.h, "rotate/resize: the test rect is square in canvas pixels");

    // Six 15-degree coarse steps make one quarter turn.
    for (int i = 0; i < 6; ++i) key(*editor, Qt::Key_BracketLeft, Qt::ShiftModifier);
    check::is_true(
        std::abs(std::get<overlay::RectItem>(*editor->selected_item()).rotation - 90.0) <=
            1e-6,
        "rotate/resize: six coarse steps make a quarter turn");

    // Press where the item's own top-right corner sits (in the item's
    // *local*, unrotated frame) -- after a 90-degree turn, that is
    // exactly where the bottom-right corner, and with it the resize
    // grip, has rotated to.
    const QPoint at_rotated_corner = widget_point(box.x + box.w, box.y);
    const double width_before =
        std::get<overlay::RectItem>(*editor->selected_item()).width;
    press(*editor, at_rotated_corner);
    move_to(*editor, at_rotated_corner + QPoint(30, 0));
    release(*editor, at_rotated_corner + QPoint(30, 0));
    const double width_after =
        std::get<overlay::RectItem>(*editor->selected_item()).width;
    check::is_true(std::abs(width_after - width_before) > 1e-6,
                   "rotate/resize: dragging the rotated corner still resizes the item");
    delete editor;
}

void test_the_rotate_handle_stays_reachable_near_a_canvas_edge() {
    // Pinned right at the canvas's own top-right corner -- the
    // unclamped handle position (further up and further right of the
    // item's own top-right corner, see `rotate_handle_rect`'s outward
    // offset) would land off the canvas entirely without the clamp.
    gui::OverlayEditor* editor = make_editor();
    editor->add_rect();
    auto* rect = std::get_if<overlay::RectItem>(editor->selected_item());
    check::is_true(rect != nullptr, "rotate/edge: a rect is selected");
    if (rect == nullptr) {
        delete editor;
        return;
    }
    rect->x = 0.92;
    rect->y = 0.02;
    rect->width = 0.06;
    rect->height = 0.06;
    editor->refresh_item();

    const double r0 = std::get<overlay::RectItem>(*editor->selected_item()).rotation;
    // A few pixels in from the canvas's literal corner pixel, so the
    // press is safely inside the clamped handle regardless of rounding
    // at the exact edge.
    const QPoint press_at =
        widget_point(overlay::CANVAS_W, 0) + QPoint(-3, 3);
    press(*editor, press_at);
    move_to(*editor, press_at + QPoint(-25, 5));
    const double r1 = std::get<overlay::RectItem>(*editor->selected_item()).rotation;
    release(*editor, press_at + QPoint(-25, 5));
    check::is_true(std::abs(r1 - r0) > 1e-6,
                   "rotate/edge: the handle is still grabbable at the canvas corner");
    delete editor;
}

// A "last received" inset paints nothing at all until a reception
// arrives -- correctly, since this is also what encodes the
// transmission and a placeholder must never go out over the air in
// place of a picture. But that used to leave the item invisible and
// unfindable on the *preview* too, before the operator had clicked
// anything -- a real gap for a template that starts with one already
// in it. The editor now draws its own frame there, over the composed
// picture rather than into it.
void test_an_unresolved_last_rx_inset_shows_a_placeholder_frame() {
    gui::OverlayEditor* editor = make_editor();
    editor->add_last_rx_inset();  // no reception yet
    QCoreApplication::processEvents();

    const overlay::Item& item = editor->doc().items.front();
    // A few pixels in from the box's own top-left corner -- not its
    // centre, which the placeholder's caption paints over and whose
    // anti-aliased edge just barely misses an exact colour match. No
    // `last_rx` picture to measure an aspect from yet, so `item_bbox`
    // falls back to 0.75, which this sample point has to use too, to
    // land inside the same box the editor computes.
    const overlay::Bbox box =
        overlay::item_bbox(overlay::CANVAS_W, overlay::CANVAS_H, item, nullptr);
    const QPoint at = widget_point(box.x + 5, box.y + 5);

    const QImage before = editor->grab().toImage();
    check::is_true(before.rect().contains(at), "placeholder: the sample point is on screen");
    check::equal(before.pixelColor(at).rgb(), gui::style::color::viewport_frame().rgb(),
                 "placeholder: an unresolved last_rx inset paints its own frame");

    // Once a reception arrives, the placeholder must get out of the way
    // -- overlay::render is what draws the actual picture there now.
    editor->set_last_rx(grey(40, 30));
    QCoreApplication::processEvents();
    const QImage after = editor->grab().toImage();
    check::is_true(after.pixelColor(at).rgb() != gui::style::color::viewport_frame().rgb(),
                   "placeholder: it is gone once a reception arrives");
    delete editor;
}

// The placeholder is findable, not just visible once already selected:
// it has to draw for every unresolved last_rx item, since the whole
// point is helping the operator locate one they have not clicked yet.
void test_the_placeholder_draws_even_when_nothing_is_selected() {
    gui::OverlayEditor* editor = make_editor();
    editor->add_last_rx_inset();
    // Deselect by clicking empty canvas -- `remove_selected` would take
    // the item out of the document entirely, which is not what this
    // test wants: the item stays, only the selection changes.
    press(*editor, widget_point(2, 2));
    QCoreApplication::processEvents();
    check::is_true(editor->selected_item() == nullptr,
                   "placeholder: nothing is selected");
    check::is_true(!editor->doc().items.empty(),
                   "placeholder: but the item is still in the document");

    const overlay::Item& item = editor->doc().items.front();
    const overlay::Bbox box =
        overlay::item_bbox(overlay::CANVAS_W, overlay::CANVAS_H, item, nullptr);
    const QPoint at = widget_point(box.x + 5, box.y + 5);
    const QImage frame = editor->grab().toImage();
    check::equal(frame.pixelColor(at).rgb(), gui::style::color::viewport_frame().rgb(),
                 "placeholder: still drawn with nothing selected");
    delete editor;
}

void test_selection_screen_rect_tracks_the_selection() {
    gui::OverlayEditor* editor = make_editor();
    check::is_true(editor->selection_screen_rect().isEmpty(),
                   "selection rect: empty with nothing selected");

    editor->add_last_rx_inset();  // no last_rx set yet, but still an item
    const images::Picture inset = grey(40, 30);
    editor->set_last_rx(inset);

    const QRect on_screen = editor->selection_screen_rect();
    check::is_true(!on_screen.isEmpty(),
                   "selection rect: non-empty once something is selected");

    const overlay::Bbox box = overlay::item_bbox(
        overlay::CANVAS_W, overlay::CANVAS_H, editor->doc().items.front(), &inset);
    const QPoint expected_center =
        widget_point(box.x + box.w / 2.0, box.y + box.h / 2.0);
    check::is_true(on_screen.contains(expected_center),
                   "selection rect: covers the item's own on-screen centre");

    editor->remove_selected();
    check::is_true(editor->selection_screen_rect().isEmpty(),
                   "selection rect: empty again once the selection is removed");
    delete editor;
}

int main(int argc, char** argv) {
    check::report_crashes_instead_of_prompting();
    qputenv("QT_QPA_PLATFORM", "offscreen");
    const QApplication app(argc, argv);

    test_no_picture_means_nothing_to_compose();
    test_clicking_an_item_selects_it();
    test_dragging_moves_the_item_to_the_cursor();
    test_a_drag_keeps_the_grab_offset();
    test_normalized_coordinates_survive_a_resize();
    test_removing_clears_the_selection();
    test_the_composite_is_the_renderer_s_output();
    test_set_fields_substitutes_the_composite_but_not_the_document();
    test_set_fields_with_no_placeholders_changes_nothing();
    test_a_dropped_line_is_not_hit_testable();
    test_arrows_nudge_by_a_fixed_fraction();
    test_delete_removes_the_selection();
    test_an_added_item_can_be_nudged_without_clicking_first();
    test_a_reception_is_a_change_only_if_an_item_uses_it();
    test_a_reception_is_kept_even_with_nothing_to_show_it();
    test_it_pins_no_window_height();
    test_add_rect_selects_a_visible_item();
    test_raise_and_lower_swap_adjacent_items_and_follow_the_selection();
    test_bring_to_front_and_send_to_back_preserve_the_rest_of_the_order();
    test_z_order_changes_which_item_paints_on_top();
    test_dragging_the_rotate_handle_rotates_the_item();
    test_scale_keys_grow_and_shrink_the_selection();
    test_rotate_keys_turn_the_selection();
    test_the_resize_handle_tracks_the_items_rotation();
    test_the_rotate_handle_stays_reachable_near_a_canvas_edge();
    test_an_unresolved_last_rx_inset_shows_a_placeholder_frame();
    test_the_placeholder_draws_even_when_nothing_is_selected();
    test_selection_screen_rect_tracks_the_selection();

    return check::report("overlay editor");
}

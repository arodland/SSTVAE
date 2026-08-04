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
#include <QVBoxLayout>
#include <QWidget>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <string>
#include <variant>

#include "check.hpp"
#include "images/types.hpp"
#include "overlay/render.hpp"
#include "overlay_editor.hpp"

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
    test_arrows_nudge_by_a_fixed_fraction();
    test_delete_removes_the_selection();
    test_an_added_item_can_be_nudged_without_clicking_first();
    test_it_pins_no_window_height();

    return check::report("overlay editor");
}

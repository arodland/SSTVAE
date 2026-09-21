// The right-click item menu.
//
// What a menu *looks* like needs eyes. What it writes does not, and the
// failures available here are all of the silent kind: a control that
// reads a field and writes it straight back on open, a wheel step that
// is the platform's rather than the documented one, a "clear" that also
// clears somebody's callsign. Each of those leaves a menu that still
// looks and behaves like a working one.

#include <QAction>
#include <QApplication>
#include <QMenu>
#include <QPushButton>
#include <QSpinBox>
#include <QToolButton>
#include <QWheelEvent>
#include <QWidgetAction>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <string>
#include <variant>

#include "check.hpp"
#include "images/types.hpp"
#include "item_menu.hpp"
#include "overlay/model.hpp"
#include "overlay_editor.hpp"

using namespace sstvae;

namespace {

images::Picture grey(int w, int h) {
    images::Picture picture(w, h);
    std::fill(picture.rgb.begin(), picture.rgb.end(), std::uint8_t{128});
    return picture;
}

struct Fixture {
    std::unique_ptr<gui::OverlayEditor> editor;
    std::unique_ptr<gui::ItemMenu> menu;
    int edits = 0;

    Fixture() : editor(new gui::OverlayEditor()) {
        editor->resize(500, 400);
        editor->set_base_image(grey(overlay::CANVAS_W, overlay::CANVAS_H));
        menu.reset(new gui::ItemMenu(editor.get()));
        QObject::connect(editor.get(), &gui::OverlayEditor::documentChanged,
                         [this] { ++edits; });
    }

    // Open the menu over the selection, then close the popup again --
    // the controls keep whatever the open loaded into them, and a test
    // driving a visible menu would be testing the window manager.
    void open() {
        menu->popup_for(editor->selected_item(), QPoint(10, 10));
        menu->close();
    }

    overlay::TextItem* open_on_text(const std::string& text = "N0CALL") {
        editor->add_text(text);
        open();
        return std::get_if<overlay::TextItem>(editor->selected_item());
    }

    overlay::RectItem* open_on_rect() {
        editor->add_rect();
        open();
        return std::get_if<overlay::RectItem>(editor->selected_item());
    }

    // The text of the item at a given depth -- index 0 is the bottom
    // layer, since `Doc::items` is drawn back to front.
    std::string text_at(int index) {
        return std::get<overlay::TextItem>(editor->doc().items[index]).text;
    }

    // The whole stack as one string, so a wrong order fails with the
    // order it got rather than with "false".
    std::string stack() {
        std::string out;
        for (int i = 0; i < static_cast<int>(editor->doc().items.size()); ++i) {
            out += text_at(i);
        }
        return out;
    }

    template <typename T>
    T* find(const char* name) {
        return menu->findChild<T*>(QString::fromLatin1(name));
    }
};

void wheel(QWidget* widget, int notches, Qt::KeyboardModifiers modifiers = Qt::NoModifier) {
    const QPointF at(5, 5);
    QWheelEvent event(at, widget->mapToGlobal(at.toPoint()), QPoint(0, 0),
                      QPoint(0, notches * 120), Qt::NoButton, modifiers, Qt::NoScrollPhase,
                      false);
    QApplication::sendEvent(widget, &event);
}

// A text item's non-style fields, for the "and nothing else moved" half
// of the clear test.
std::string positional(const overlay::TextItem& text) {
    return text.text + "|" + std::to_string(text.x) + "|" + std::to_string(text.y) + "|" +
           std::to_string(text.size) + "|" + std::to_string(text.rotation) + "|" +
           text.align + "|" + text.anchor + "|" + std::to_string(text.line_spacing) + "|" +
           text.font;
}

// The wheel steps are the menu's, not the platform's.
//
// A `QSpinBox` already answers a wheel with its own `singleStep`, and a
// `QMenu` answers one by scrolling itself -- so "it responds to the
// wheel" is not evidence that either of the documented steps is what
// happened. Both are asserted in the field's own unit *and* in the
// document, because the field is in pixels of the transmitted frame and
// the document is in fractions of it: a conversion dropped in either
// direction leaves the spin box looking perfectly correct.
void test_the_wheel_steps_are_the_documented_ones() {
    Fixture fix;
    overlay::TextItem* text = fix.open_on_text();
    auto* size = fix.find<QSpinBox>("menu_size");
    auto* rotation = fix.find<QSpinBox>("menu_rotation");
    check::is_true(text != nullptr && size != nullptr && rotation != nullptr,
                   "menu: a text item with size and rotation fields");
    if (text == nullptr || size == nullptr || rotation == nullptr) return;

    // 0.08 of a 480-line canvas.
    check::equal(size->value(), 38, "menu: size opens in canvas pixels");

    wheel(size, 1);
    check::equal(size->value(), 39, "menu: one notch is one pixel");
    check::is_true(std::abs(text->size - 39.0 / overlay::CANVAS_H) < 1e-12,
                   "menu: and it reaches the document as a fraction");

    wheel(size, 1, Qt::ShiftModifier);
    check::equal(size->value(), 49, "menu: shift is ten pixels");
    wheel(size, -1, Qt::ShiftModifier);
    check::equal(size->value(), 39, "menu: and back down");

    // Clamped at the editor's own limits (0.01..1.5 of the height).
    wheel(size, 100, Qt::ShiftModifier);
    check::equal(size->value(), 720, "menu: size stops at the top");
    wheel(size, -1000);
    check::equal(size->value(), 5, "menu: and at the bottom");

    check::equal(rotation->value(), 0, "menu: rotation opens at zero");
    wheel(rotation, 1);
    check::equal(rotation->value(), 1, "menu: one notch is one degree");
    check::equal(text->rotation, 1.0, "menu: and reaches the document");

    // Shift is a *destination*, not a bigger step: from 1 degree the
    // next multiple of 15 upward is 15, and downward is 0. A "+15" step
    // would give 16 and -14, and both still look like a working snap
    // until you try to square something up.
    wheel(rotation, 1, Qt::ShiftModifier);
    check::equal(rotation->value(), 15, "menu: shift snaps up to 15");
    wheel(rotation, -1, Qt::ShiftModifier);
    check::equal(rotation->value(), 0, "menu: and back down to 0");
    wheel(rotation, -1, Qt::ShiftModifier);
    check::equal(rotation->value(), -15, "menu: through zero as well");

    wheel(rotation, 1000);
    check::equal(rotation->value(), 180, "menu: rotation stops at 180");
    wheel(rotation, -1000);
    check::equal(rotation->value(), -180, "menu: and at -180");
}

// A rect's stroke width is pixels of the frame's *width*, which is what
// the document's fraction is of -- a conversion through the height would
// read 3 where it should read 4, and look entirely plausible.
void test_the_rect_stroke_width_is_in_canvas_pixels() {
    Fixture fix;
    overlay::RectItem* rect = fix.open_on_rect();
    auto* width = fix.find<QSpinBox>("menu_rect_stroke_width");
    check::is_true(rect != nullptr && width != nullptr, "menu: a rect stroke width");
    if (rect == nullptr || width == nullptr) return;

    // 0.006 of 640.
    check::equal(width->value(), 4, "menu: rect stroke opens in canvas pixels");
    wheel(width, 1);
    check::equal(width->value(), 5, "menu: one notch is one pixel");
    check::is_true(std::abs(rect->stroke_width - 5.0 / overlay::CANVAS_W) < 1e-12,
                   "menu: a fraction of the canvas width");
}

// Opening the menu must not be an edit.
//
// Every control is filled from the item when the menu opens, and each of
// those `setValue`/`setChecked` calls fires the same signal a real edit
// does. Without the loading guard the menu writes the item back to
// itself on open -- which is invisible for a field that round-trips, and
// *not* invisible for the size, which the menu shows in whole canvas
// pixels. A size of 0.0801 would come back as 38/480.
void test_opening_the_menu_edits_nothing() {
    Fixture fix;
    fix.editor->add_text("N0CALL");
    auto* text = std::get_if<overlay::TextItem>(fix.editor->selected_item());
    check::is_true(text != nullptr, "menu: a text item");
    if (text == nullptr) return;
    // Deliberately not a whole number of canvas pixels: 38.448.
    text->size = 0.0801;
    text->rotation = 7.5;
    const std::string before = positional(*text);

    fix.edits = 0;
    fix.open();
    check::equal(fix.edits, 0, "menu: opening it is not an edit");
    check::equal(positional(*text), before, "menu: and the item is what it was");

    fix.editor->add_rect();
    auto* rect = std::get_if<overlay::RectItem>(fix.editor->selected_item());
    if (rect == nullptr) return;
    rect->stroke_width = 0.0071;  // 4.544 px
    rect->rotation = -2.5;
    fix.edits = 0;
    fix.open();
    check::equal(fix.edits, 0, "menu: nor over a rect");
    check::equal(rect->stroke_width, 0.0071, "menu: whose stroke is untouched");
    check::equal(rect->rotation, -2.5, "menu: and rotation");
}

// Solid -> linear -> radial -> outline for text, writing the kind *and*
// the gradient's shape -- radial is a field of its own, and a button
// that set only the kind would draw every "radial" as linear.
void test_the_text_fill_mode_cycles() {
    Fixture fix;
    overlay::TextItem* text = fix.open_on_text();
    auto* button = fix.find<QPushButton>("menu_fill_mode");
    auto* angle = fix.find<QSpinBox>("menu_fill_angle");
    auto* to = fix.find<QPushButton>("menu_fill_to");
    check::is_true(text != nullptr && button != nullptr && angle != nullptr && to != nullptr,
                   "menu: it has a fill-mode button");
    if (text == nullptr || button == nullptr || angle == nullptr || to == nullptr) return;

    check::equal(button->text().toStdString(), std::string("SOLID"), "menu: opens solid");
    check::is_true(!to->isEnabled() && !angle->isEnabled(),
                   "menu: a solid fill has no second stop or angle");

    button->click();
    check::equal(text->fill_kind, std::string("gradient"), "menu: solid -> gradient");
    check::equal(text->fill_gradient, std::string("linear"), "menu: a linear one");
    check::equal(button->text().toStdString(), std::string("LINEAR"), "menu: label follows");
    check::is_true(to->isEnabled() && angle->isEnabled(), "menu: linear has both");

    button->click();
    check::equal(text->fill_kind, std::string("gradient"), "menu: still a gradient");
    check::equal(text->fill_gradient, std::string("radial"), "menu: linear -> radial");
    check::is_true(to->isEnabled() && !angle->isEnabled(), "menu: radial has no angle");

    button->click();
    check::equal(text->fill_kind, std::string("none"), "menu: radial -> outline");
    check::equal(button->text().toStdString(), std::string("OUTLINE"), "menu: says so");

    button->click();
    check::equal(text->fill_kind, std::string("solid"), "menu: outline -> solid");

    // The angle writes the document.
    button->click();
    angle->setValue(45);
    check::equal(text->fill_angle, 45.0, "menu: the angle reaches the document");
}

// A rect starts empty and cycles from there; fill and stroke are two
// sets of fields, and each row must write its own.
void test_the_rect_modes_cycle_independently() {
    Fixture fix;
    overlay::RectItem* rect = fix.open_on_rect();
    auto* fill = fix.find<QPushButton>("menu_rect_fill_mode");
    auto* stroke = fix.find<QPushButton>("menu_rect_stroke_mode");
    auto* stroke_angle = fix.find<QSpinBox>("menu_rect_stroke_angle");
    check::is_true(rect != nullptr && fill != nullptr && stroke != nullptr &&
                       stroke_angle != nullptr,
                   "menu: a rect with fill and stroke modes");
    if (rect == nullptr || fill == nullptr || stroke == nullptr || stroke_angle == nullptr)
        return;

    // `add_rect` places a solid rect so it can be seen; the cycle
    // itself starts from empty.
    check::equal(fill->text().toStdString(), std::string("SOLID"), "menu: a new rect is solid");
    fill->click();
    fill->click();
    fill->click();
    check::equal(rect->fill_kind, std::string("none"), "menu: solid -> ... -> none");
    check::equal(fill->text().toStdString(), std::string("NONE"), "menu: and says so");
    fill->click();
    check::equal(rect->fill_kind, std::string("solid"), "menu: none -> solid");
    check::equal(rect->stroke_kind, std::string("none"), "menu: and the stroke is untouched");

    stroke->click();
    stroke->click();
    stroke->click();
    check::equal(rect->stroke_kind, std::string("gradient"), "menu: stroke -> gradient");
    check::equal(rect->stroke_gradient, std::string("radial"), "menu: a radial one");
    check::equal(rect->fill_kind, std::string("solid"), "menu: the fill kind is its own");
    stroke->click();
    check::equal(rect->stroke_kind, std::string("none"), "menu: radial -> none");

    stroke->click();
    stroke->click();  // linear
    stroke_angle->setValue(-30);
    check::equal(rect->stroke_angle, -30.0, "menu: the stroke angle is the stroke's");
    check::equal(rect->fill_angle, 0.0, "menu: and not the fill's");
}

// The stroke toggle is the only control on `stroke_width`, and zero is
// the off switch -- so turning it off has to remember the width
// somewhere or the operator loses it. It must remember the width they
// *chose*, not the default, which is the version of this that passes a
// test written with the default still in place.
void test_the_stroke_toggle_keeps_the_width() {
    Fixture fix;
    overlay::TextItem* text = fix.open_on_text();
    auto* on = fix.find<QToolButton>("menu_stroke_on");
    auto* width = fix.find<QSpinBox>("menu_stroke_width");
    check::is_true(text != nullptr && on != nullptr && width != nullptr,
                   "menu: it has a stroke toggle and width");
    if (text == nullptr || on == nullptr || width == nullptr) return;

    check::is_true(on->isChecked(), "menu: a new item has a stroke");
    check::equal(width->value(), 12, "menu: 0.12 of the glyph size");

    width->setValue(30);
    check::is_true(std::abs(text->stroke_width - 0.30) < 1e-12,
                   "menu: the width reaches the document");

    on->setChecked(false);
    check::equal(text->stroke_width, 0.0, "menu: off is a width of zero");
    check::equal(width->value(), 0, "menu: and the field says zero");
    check::is_true(!width->isEnabled(), "menu: with nothing to set");

    on->setChecked(true);
    check::is_true(std::abs(text->stroke_width - 0.30) < 1e-12,
                   "menu: back on restores the width that was chosen");
    check::equal(width->value(), 30, "menu: and the field with it");
    check::is_true(width->isEnabled(), "menu: enabled again");
}

// Clear resets the appearance and *only* the appearance.
//
// The untouched fields are asserted explicitly rather than left implied:
// the obvious way to write this button is `*text = overlay::TextItem{}`
// with the text copied back, which passes every assertion about the
// style fields and silently moves the item to the top-left corner at the
// default size.
void test_clear_resets_the_style_fields_only() {
    Fixture fix;
    fix.editor->add_text("W1AW/KH6");
    auto* text = std::get_if<overlay::TextItem>(fix.editor->selected_item());
    check::is_true(text != nullptr, "menu: a text item");
    if (text == nullptr) return;

    text->bold = true;
    text->italic = true;
    text->underline = true;
    text->font_family = "monospace";
    text->fill_kind = "gradient";
    text->fill_gradient = "radial";
    text->color = "#ff0000";
    text->fill_color2 = "#00ff00";
    text->fill_angle = 120.0;
    text->stroke_color = "#0000ff";
    text->stroke_width = 0.4;
    text->x = 0.4;
    text->y = 0.6;
    text->size = 0.2;
    text->rotation = 33.0;
    text->align = "center";
    text->anchor = "mm";
    text->line_spacing = 0.5;
    text->font = "/fonts/sent-with-the-template.ttf";
    const std::string untouched = positional(*text);

    fix.open();
    auto* clear = fix.find<QPushButton>("menu_clear");
    auto* bold = fix.find<QToolButton>("menu_bold");
    check::is_true(clear != nullptr && bold != nullptr, "menu: it has a clear button");
    if (clear == nullptr || bold == nullptr) return;
    clear->click();

    const overlay::TextItem fresh;
    check::equal(text->bold, fresh.bold, "clear: bold");
    check::equal(text->italic, fresh.italic, "clear: italic");
    check::equal(text->underline, fresh.underline, "clear: underline");
    check::equal(text->font_family, fresh.font_family, "clear: font family");
    check::equal(text->fill_kind, fresh.fill_kind, "clear: fill kind");
    check::equal(text->fill_gradient, fresh.fill_gradient, "clear: gradient shape");
    check::equal(text->color, fresh.color, "clear: colour");
    check::equal(text->fill_color2, fresh.fill_color2, "clear: the far stop");
    check::equal(text->fill_angle, fresh.fill_angle, "clear: fill angle");
    check::equal(text->stroke_color, fresh.stroke_color, "clear: stroke colour");
    check::equal(text->stroke_width, fresh.stroke_width, "clear: stroke width");
    check::is_true(!bold->isChecked(), "clear: and the controls say so");

    check::equal(positional(*text), untouched,
                 "clear: and nothing else moved -- text, place, size, rotation, "
                 "align, anchor, line spacing, font file");
}

// What each kind of item is offered.
//
// Ordering is the one thing on this menu that means anything for a
// picture; bold and a gradient do not. Opening nothing at all would be
// the other defensible choice and is the worse one -- a right-click that
// does nothing reads as broken.
void test_each_kind_of_item_gets_its_own_submenus() {
    Fixture fix;
    auto* format = fix.find<QMenu>("menu_format");
    auto* style = fix.find<QMenu>("menu_style");
    auto* layers = fix.find<QMenu>("menu_layers");
    auto* text_row = fix.find<QPushButton>("menu_clear");
    auto* rect_row = fix.find<QPushButton>("menu_rect_fill_mode");
    check::is_true(format != nullptr && style != nullptr && layers != nullptr &&
                       text_row != nullptr && rect_row != nullptr,
                   "menu: it has all three submenus");
    if (format == nullptr || style == nullptr || layers == nullptr || text_row == nullptr ||
        rect_row == nullptr)
        return;

    // Whether a style row is offered is its action's visibility; the
    // widget itself is only ever shown by the menu that holds it.
    const auto offered = [&](QWidget* row) {
        for (QAction* action : style->actions()) {
            auto* holder = qobject_cast<QWidgetAction*>(action);
            if (holder != nullptr && holder->defaultWidget()->isAncestorOf(row))
                return action->isVisible();
        }
        return false;
    };

    fix.editor->add_image_inset("/nonexistent/inset.png");
    fix.open();
    check::is_true(!format->menuAction()->isVisible(), "menu: no Format for a picture");
    check::is_true(!style->menuAction()->isVisible(), "menu: no Style either");
    check::is_true(layers->menuAction()->isVisible(), "menu: but Layers");

    fix.editor->add_rect();
    fix.open();
    check::is_true(!format->menuAction()->isVisible(), "menu: no Format for a rect");
    check::is_true(style->menuAction()->isVisible(), "menu: but Style");
    check::is_true(offered(rect_row) && !offered(text_row), "menu: the rect's rows");

    fix.editor->add_text("N0CALL");
    fix.open();
    check::is_true(format->menuAction()->isVisible(), "menu: text gets Format back");
    check::is_true(offered(text_row) && !offered(rect_row), "menu: and the text's rows");
}

// A style row that is not offered must not be *painted* either.
//
// `QMenu` skips a hidden action when it lays itself out and never hides
// that action's widget, so a row shown once for a text item stays
// visible -- at the menu's top-left, over the rect's own rows -- the
// next time Style opens for a rect. Only a menu that has actually been
// shown shows this, which is why the submenus are popped up here.
void test_a_hidden_style_row_is_not_painted() {
    Fixture fix;
    auto* style = fix.find<QMenu>("menu_style");
    auto* clear = fix.find<QPushButton>("menu_clear");
    auto* rect_mode = fix.find<QPushButton>("menu_rect_fill_mode");
    check::is_true(style != nullptr && clear != nullptr && rect_mode != nullptr,
                   "menu: the style rows");
    if (style == nullptr || clear == nullptr || rect_mode == nullptr) return;
    QWidget* text_row = clear->parentWidget();
    QWidget* rect_rows = rect_mode->parentWidget()->parentWidget();

    const auto show_style = [&] {
        fix.menu->popup_for(fix.editor->selected_item(), QPoint(10, 10));
        style->popup(QPoint(40, 40));
        QApplication::processEvents();
    };
    const auto hide_style = [&] {
        style->close();
        fix.menu->close();
        QApplication::processEvents();
    };

    fix.editor->add_text("N0CALL");
    show_style();
    check::is_true(text_row->isVisible() && !rect_rows->isVisible(),
                   "menu: text shows the text row alone");
    hide_style();

    fix.editor->add_rect();
    show_style();
    check::is_true(rect_rows->isVisible(), "menu: a rect shows its rows");
    check::is_true(!text_row->isVisible(), "menu: and not the text row it showed last time");
    hide_style();

    fix.editor->add_text("N0CALL");
    show_style();
    check::is_true(text_row->isVisible() && !rect_rows->isVisible(),
                   "menu: and back again for text");
    hide_style();
}

// The Layers row acts on the editor, and Shift changes both what the
// buttons say and what they do.
//
// The label has to change *before* the click -- a control that silently
// does something other than what it says is worse than one that cannot
// do it at all -- which is why the modifier is a polled state rather
// than something read off the click event.
void test_the_layer_row_reorders_and_shift_relabels() {
    // **Four items, and the selection starts two from each end.** With
    // three, every shifted move is also reachable in one step -- from
    // the middle of a stack of three, "down one" and "to the bottom"
    // land in the same place, so a menu that ignored Shift entirely
    // passed. This is the fixture that tells them apart.
    Fixture fix;
    fix.editor->add_text("A");
    fix.editor->add_text("B");
    fix.editor->add_text("C");
    fix.editor->add_text("D");
    fix.open();  // on D, which is on top
    check::equal(fix.stack(), std::string("ABCD"), "layers: added back to front");

    auto* up = fix.find<QPushButton>("menu_layer_up");
    auto* down = fix.find<QPushButton>("menu_layer_down");
    check::is_true(up != nullptr && down != nullptr, "menu: it has layer buttons");
    if (up == nullptr || down == nullptr) return;
    check::is_true(!up->isEnabled() && down->isEnabled(),
                   "layers: nothing above the top item");

    const std::string plain_up = up->text().toStdString();
    const std::string plain_down = down->text().toStdString();

    down->click();
    check::equal(fix.stack(), std::string("ABDC"), "layers: down moves one layer");
    check::is_true(up->isEnabled(), "layers: and there is something above it now");

    fix.menu->set_layer_modifiers(Qt::ShiftModifier);
    check::is_true(up->text().toStdString() != plain_up,
                   "layers: shift says the button will do something else");
    check::is_true(down->text().toStdString() != plain_down, "layers: both of them");

    // Two layers to go, so one step short of the bottom is "ADBC" and
    // the jump is "DABC" -- the assertion that made Shift load-bearing.
    down->click();
    check::equal(fix.stack(), std::string("DABC"), "layers: with shift, down goes all the way");
    check::is_true(!down->isEnabled(), "layers: and there is nothing below it");
    up->click();
    check::equal(fix.stack(), std::string("ABCD"), "layers: and up goes all the way back");

    fix.menu->set_layer_modifiers(Qt::NoModifier);
    check::equal(up->text().toStdString(), plain_up, "layers: releasing shift says so again");
    check::equal(down->text().toStdString(), plain_down, "layers: on both");
    down->click();
    check::equal(fix.stack(), std::string("ABDC"), "layers: one layer at a time again");
}

// The menu edits the editor's selection, so it refuses to open over
// anything else. `OverlayEditor::contextMenuEvent` selects what was
// right-clicked before it emits, so this never fires in the app -- it is
// the assertion that keeps that a requirement rather than a habit.
void test_it_refuses_an_item_that_is_not_the_selection() {
    Fixture fix;
    fix.editor->add_text("A");
    fix.editor->add_text("B");  // selected
    overlay::Item* other = &const_cast<overlay::Doc&>(fix.editor->doc()).items[0];

    fix.menu->popup_for(other, QPoint(10, 10));
    check::is_true(!fix.menu->isVisible(), "menu: not over an unselected item");
    fix.menu->popup_for(nullptr, QPoint(10, 10));
    check::is_true(!fix.menu->isVisible(), "menu: nor over nothing");

    fix.menu->popup_for(fix.editor->selected_item(), QPoint(10, 10));
    check::is_true(fix.menu->isVisible(), "menu: but it does over the selection");
    fix.menu->close();
}

// Bold, italic and underline are document fields, not a Qt font effect
// painted into the preview -- which is the property that keeps what the
// operator arranges and what goes on the air the same picture.
void test_the_weight_toggles_write_the_document() {
    Fixture fix;
    overlay::TextItem* text = fix.open_on_text();
    auto* bold = fix.find<QToolButton>("menu_bold");
    auto* italic = fix.find<QToolButton>("menu_italic");
    auto* underline = fix.find<QToolButton>("menu_underline");
    check::is_true(text != nullptr && bold != nullptr && italic != nullptr &&
                       underline != nullptr,
                   "menu: it has the three weight toggles");
    if (text == nullptr || bold == nullptr || italic == nullptr || underline == nullptr)
        return;

    fix.edits = 0;
    bold->setChecked(true);
    check::is_true(text->bold, "menu: bold reaches the document");
    check::is_true(!text->italic && !text->underline,
                   "menu: and only bold -- the three are not one flag");
    italic->setChecked(true);
    underline->setChecked(true);
    check::is_true(text->italic && text->underline, "menu: and the other two");
    bold->setChecked(false);
    check::is_true(!text->bold && text->italic, "menu: turning one off leaves the others");
    check::equal(fix.edits, 4, "menu: each is one edit, announced once");
}

// The family is a nested menu of generic keywords; picking one writes the
// field, and a family the menu does not list is shown as itself rather
// than claimed to be the default.
void test_the_family_menu_writes_the_document() {
    Fixture fix;
    fix.editor->add_text("N0CALL");
    auto* text = std::get_if<overlay::TextItem>(fix.editor->selected_item());
    auto* family = fix.find<QPushButton>("menu_family");
    check::is_true(text != nullptr && family != nullptr && family->menu() != nullptr,
                   "menu: a family button with its own menu");
    if (text == nullptr || family == nullptr || family->menu() == nullptr) return;

    text->font_family = "DejaVu Serif";
    fix.open();
    check::equal(family->text().toStdString(), std::string("DejaVu Serif"),
                 "menu: an unlisted family shows as itself");

    for (QAction* action : family->menu()->actions()) {
        if (action->text() == QStringLiteral("Monospace")) action->trigger();
    }
    check::equal(text->font_family, std::string("monospace"), "menu: picking writes the keyword");
    check::equal(family->text().toStdString(), std::string("Monospace"), "menu: and shows it");
}

// The button's text changes with the selection, so its width must cover
// the longest label rather than whichever one happens to be showing --
// sized to "Default", "Sans Serif" was clipped.
void test_the_family_button_fits_every_label() {
    Fixture fix;
    fix.editor->add_text("N0CALL");
    auto* family = fix.find<QPushButton>("menu_family");
    check::is_true(family != nullptr, "menu: a family button");
    if (family == nullptr || family->menu() == nullptr) return;
    fix.open();
    const int room = family->minimumWidth();
    for (QAction* action : family->menu()->actions()) {
        action->trigger();
        check::is_true(family->sizeHint().width() <= room,
                       ("menu: the family button fits " + action->text()).toStdString());
    }
}

// Remove goes through the editor, so the selection and the document stay
// one story.
void test_remove_removes_the_selection() {
    Fixture fix;
    fix.editor->add_text("A");
    fix.editor->add_text("B");
    fix.open();
    auto* remove = fix.find<QAction>("menu_remove");
    check::is_true(remove != nullptr, "menu: it has Remove");
    if (remove == nullptr) return;
    remove->trigger();
    check::equal(fix.stack(), std::string("A"), "menu: Remove takes the selected item");
    check::is_true(fix.editor->selected_item() == nullptr, "menu: and the selection with it");
}

}  // namespace

int main(int argc, char** argv) {
    check::report_crashes_instead_of_prompting();
    qputenv("QT_QPA_PLATFORM", "offscreen");
    const QApplication app(argc, argv);

    test_the_wheel_steps_are_the_documented_ones();
    test_the_rect_stroke_width_is_in_canvas_pixels();
    test_opening_the_menu_edits_nothing();
    test_the_text_fill_mode_cycles();
    test_the_rect_modes_cycle_independently();
    test_the_stroke_toggle_keeps_the_width();
    test_clear_resets_the_style_fields_only();
    test_each_kind_of_item_gets_its_own_submenus();
    test_a_hidden_style_row_is_not_painted();
    test_the_layer_row_reorders_and_shift_relabels();
    test_it_refuses_an_item_that_is_not_the_selection();
    test_the_weight_toggles_write_the_document();
    test_the_family_menu_writes_the_document();
    test_the_family_button_fits_every_label();
    test_remove_removes_the_selection();

    return check::report("item menu");
}

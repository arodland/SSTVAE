// The right-click menu for an item on the composer.
//
// A `QMenu` with up to three submenus -- Format, Style, Layers -- each
// holding rows of *live controls* rather than a list of menu items, plus
// Remove. It opens only on a right-click (`OverlayEditor::
// contextMenuRequested`); a left-click selects and shows handles, and
// nothing else.
//
// What an item gets depends on what it is: text gets all three (weight,
// slant, underline, family and size; fill, stroke and rotation; order),
// a rectangle gets Style (its fill and stroke rows, and rotation) and
// Layers, and a picture inset gets Layers only -- ordering being the one
// thing here that means anything for a picture.
//
// **Everything here must be a document field the renderer honours.** The
// editor previews `overlay::render()`'s own output, so a control that
// painted an effect into the preview alone would be showing the operator
// something that is not going on the air. Every control writes a field
// on the item and then re-renders.
//
// Two constructions are deliberate and neither is the obvious one:
//
// **No `QComboBox` inside a `QWidgetAction`.** Its popup is a second
// window over a menu that holds a mouse grab, and on some styles opening
// it dismisses the menu underneath. The font family is a nested `QMenu`
// instead -- the one popup a menu is *made* of, which works on every
// style. The failure it avoids is style-dependent, so a machine where
// the combo happened to work would prove nothing about the next one.
//
// **The item is not stored.** Every edit asks the editor for its
// selection afresh, because the Layers row rotates items inside a
// `std::vector` -- a pointer captured when the menu opened would point at
// a different item by the time the next control was touched.

#ifndef SSTVAE_GUI_ITEM_MENU_HPP
#define SSTVAE_GUI_ITEM_MENU_HPP

#include <QMenu>
#include <QPoint>
#include <QString>

#include <string>

#include "overlay/model.hpp"

class QAbstractButton;
class QAction;
class QEvent;
class QObject;
class QPushButton;
class QSpinBox;
class QTimer;
class QToolButton;
class QWidget;
class QWidgetAction;

namespace sstvae::gui {

class OverlayEditor;

class ItemMenu : public QMenu {
    Q_OBJECT

public:
    explicit ItemMenu(OverlayEditor* editor, QWidget* parent = nullptr);

    // Open the menu over `item`, which must be the editor's current
    // selection -- `OverlayEditor::contextMenuEvent` selects what was
    // clicked before it emits, so that holds by construction. Anything
    // else is refused rather than guessed at, because editing an item
    // other than the selected one is silently wrong.
    void popup_for(overlay::Item* item, const QPoint& global_pos);

    // What the Layers row says and does, from the modifiers held *now*.
    //
    // Shift turns "Bring forward" into "Bring to front", and the label
    // has to change before the click to be honest about what the click
    // will do -- so this is polled on a timer while the submenu is open
    // rather than read from the click event. Public because that makes
    // it the seam a test drives, the way `Waterfall::tick()` is a slot so
    // a frame can be rendered without waiting on a timer.
    void set_layer_modifiers(Qt::KeyboardModifiers modifiers);

protected:
    // The wheel steps on the size, rotation and stroke-width fields. A
    // `QMenu` eats wheel events to scroll itself, so these are
    // intercepted before either the menu or the spin box's own handler
    // sees them.
    bool eventFilter(QObject* watched, QEvent* event) override;

private:
    // The selected item, or null when the selection is gone or the
    // controls are being filled. Every edit goes through these rather
    // than through a pointer captured when the menu opened.
    overlay::Item* current_item();
    template <typename T>
    T* current();
    // Re-render. Every control on this menu writes through here.
    void edited();

    void build_format();
    void build_text_style();
    void build_rect_style();
    void build_layers();

    // Fill every control from the item, with `loading_` set so their
    // change signals do not write the value straight back.
    void load_text(const overlay::TextItem& text);
    void load_rect(const overlay::RectItem& rect);
    // Enable each control only where it means something: a colour with
    // no fill to colour, an angle on a radial gradient, a width on no
    // stroke.
    void update_enabled();
    void update_layer_row();

    // A colour well: a button that paints its colour and, when clicked,
    // edits `field` on the selected `T`. A pointer to member rather than
    // a reference into the item: the colour dialog is modal and runs an
    // event loop, so the item is resolved again after it returns.
    QPushButton* color_well(QWidget* parent, const QString& name, const QString& tip);
    template <typename T>
    void pick_color(QAbstractButton* well, std::string T::* field);

    // One wheel notch on a field, in the units that field shows.
    void step_rotation(QSpinBox* field, int notches, bool coarse);

    OverlayEditor* editor_ = nullptr;

    QMenu* format_menu_ = nullptr;
    QMenu* style_menu_ = nullptr;
    QMenu* layers_menu_ = nullptr;
    QWidgetAction* text_style_action_ = nullptr;
    QWidgetAction* rect_style_action_ = nullptr;
    QAction* remove_action_ = nullptr;

    // --- Format (text) --------------------------------------------------
    QToolButton* bold_ = nullptr;
    QToolButton* italic_ = nullptr;
    QToolButton* underline_ = nullptr;
    QPushButton* family_ = nullptr;
    QSpinBox* size_ = nullptr;

    // --- Style (text) ---------------------------------------------------
    QPushButton* fill_from_ = nullptr;
    QPushButton* fill_mode_ = nullptr;
    QPushButton* fill_to_ = nullptr;
    QSpinBox* fill_angle_ = nullptr;
    QToolButton* stroke_on_ = nullptr;
    QPushButton* stroke_color_ = nullptr;
    QSpinBox* stroke_width_ = nullptr;
    QSpinBox* rotation_ = nullptr;

    // --- Style (rectangle) ----------------------------------------------
    QPushButton* rect_fill_mode_ = nullptr;
    QPushButton* rect_fill_color_ = nullptr;
    QPushButton* rect_fill_to_ = nullptr;
    QSpinBox* rect_fill_angle_ = nullptr;
    QPushButton* rect_stroke_mode_ = nullptr;
    QPushButton* rect_stroke_color_ = nullptr;
    QPushButton* rect_stroke_to_ = nullptr;
    QSpinBox* rect_stroke_angle_ = nullptr;
    QSpinBox* rect_stroke_width_ = nullptr;
    QSpinBox* rect_rotation_ = nullptr;

    // --- Layers ----------------------------------------------------------
    QPushButton* layer_up_ = nullptr;
    QPushButton* layer_down_ = nullptr;
    // Polls `QApplication::keyboardModifiers` while Layers is open.
    QTimer* shift_watch_ = nullptr;
    bool shifted_ = false;

    // What the text stroke toggle restores when switched back on.
    // `stroke_width` is the only stroke field -- zero *is* the off
    // switch, because two fields that can disagree about whether there
    // is a stroke is one field too many -- so turning it off has to
    // remember the width somewhere, and that is here, not the document.
    double stroke_restore_ = 0.12;

    // Set while the controls are being filled from an item.
    bool loading_ = false;
};

}  // namespace sstvae::gui

#endif

// Composing an overlay on the picture about to be sent.
//
// **The preview is `overlay::render()`'s output, not a Qt-drawn
// imitation of it.** That is the property the whole design rests on:
// what the operator arranges is what goes on the air, by construction
// rather than by two pieces of code agreeing. The reference says the
// same thing about its `QGraphicsView`; here it falls out even more
// directly, because a plain painted widget *has* no scene to drift.
//
// Selection handles come from `overlay::item_bbox`, which is the same
// geometry the renderer uses to place the item -- so a handle cannot
// sit somewhere other than the thing it selects.
//
// Coordinates in the document are normalized 0..1, and stay that way
// through dragging: the widget maps to canvas space and back, and never
// stores a pixel position. A saved template therefore means the same
// thing at any size.

#ifndef SSTVAE_GUI_OVERLAY_EDITOR_HPP
#define SSTVAE_GUI_OVERLAY_EDITOR_HPP

#include <QPolygon>
#include <QRect>
#include <QWidget>


#include <optional>
#include <string>

#include "images/types.hpp"
#include "overlay/model.hpp"
#include "overlay/render.hpp"
#include "overlay/template.hpp"

namespace sstvae::gui {

class OverlayEditor : public QWidget {
    Q_OBJECT

public:
    explicit OverlayEditor(QWidget* parent = nullptr);
    ~OverlayEditor() override;

    QSize sizeHint() const override;
    // **No `heightForWidth`, deliberately.** It was here, with a
    // comment saying a `QSplitter` ignores it -- true, and the reason it
    // did no visible harm for as long as it did. A `QTabWidget`
    // *propagates* it, so the tabbed layout handed the window a
    // minimum height of `width * 3/4`: 1840 px at 2020 px wide,
    // measured. The window grew off the bottom of the screen and did
    // not come back when the layout was switched again, because Qt
    // lowers a minimum without resizing.
    //
    // It is also the one form of the ratchet a test on
    // `minimumSizeHint()` cannot see, since that never consults
    // `heightForWidth` -- Qt applies `minimumHeightForWidth` at layout
    // time instead. `test_overlay_editor.cpp` measures a *container's*
    // `minimumHeightForWidth` for that reason.
    //
    // The 4:3 shape is kept by `resizeEvent` (a maximum, which
    // constrains nothing upward) and by `canvas_rect()`, which
    // letterboxes both ways.

    // The picture the overlay sits on, already framed to the transmit
    // size by the caller.
    void set_base_image(const images::Picture& image);
    bool has_base() const { return !base_.empty(); }

    // The most recent reception, for a "last_rx" inset. Late-bound on
    // purpose: an item referring to it keeps meaning "the most recent
    // one" rather than freezing today's picture into the document.
    void set_last_rx(const images::Picture& image);
    bool has_last_rx() const { return last_rx_.has_value(); }

    // Whether anything in the document resolves to the last reception.
    // Public so a test can state the condition `set_last_rx` turns on.
    bool uses_last_rx() const;

    void add_text(const std::string& text);
    void add_image_inset(const std::string& path);
    void add_last_rx_inset();
    void add_rect();
    void remove_selected();
    void clear_overlay();

    // --- stacking order -------------------------------------------------
    //
    // `doc_.items` is drawn back to front, so "raise" means "later in
    // the vector". These act on whatever is selected and keep the
    // selection following it -- a reorder that dropped the selection
    // would be indistinguishable from one that silently failed.
    void raise_selected();
    void lower_selected();
    void bring_selected_to_front();
    void send_selected_to_back();
    // Whether raise/lower/front/back would do anything right now, for
    // the buttons that call them -- there is nothing to raise above the
    // top item or lower below the bottom one.
    bool can_raise_selected() const;
    bool can_lower_selected() const;

    // The selected item, or null. A pointer into the document, so the
    // property editor mutates it in place and calls `refresh_item`.
    overlay::Item* selected_item();
    void refresh_item();

    const overlay::Doc& doc() const { return doc_; }
    void set_doc(overlay::Doc doc);

    // Values to fill a template's holes with (docs/overlay-templates.md).
    // **The document being edited stays the raw template** -- the text
    // box shows the literal `{theircall}`, and `doc()`/`set_doc()` and
    // everything the property panel writes into a selected item's
    // `.text` are untouched by this. What changes is the *painted*
    // canvas and the geometry derived from it: `composed_image()`,
    // paint, hit-testing and the selection handle all substitute first,
    // because that is what is actually drawn -- so a handle still sits
    // on the thing it selects even though the item's own bbox (in raw
    // text) may measure a different extent than the substituted text
    // that is on screen. Defaults to empty fields, which is a no-op for
    // any document with no placeholders in it, so a plain (non-template)
    // overlay is unaffected.
    void set_fields(overlay::Fields fields);
    const overlay::Fields& fields() const { return fields_; }

    // Base plus overlay, or nothing if no picture has been chosen.
    std::optional<images::Picture> composed_image() const;

    // The current selection's on-screen rectangle, in this widget's own
    // coordinates -- what a floating panel anchoring itself "near the
    // selected item" should position against. Empty if nothing is
    // selected.
    QRect selection_screen_rect() const;

signals:
    // Null when the selection was cleared.
    void selectionChanged(overlay::Item* item);

    // The composition changed: a new base picture, an item added,
    // moved, resized, edited or removed. Emitted per mouse move during
    // a drag, so anything expensive downstream must debounce -- which
    // is what `optimize::Speculative` is for.
    //
    // Deliberately *not* emitted by `select()`: selection handles are
    // drawn over the widget, not into `composed_image()`, so choosing a
    // different item changes nothing that would be transmitted.
    void documentChanged();

    // A right-click landed on an item -- or on one of the selection's
    // grips, which belong to the selected item. The item is already
    // selected when this fires, so a menu opened from it edits what the
    // operator actually clicked; `global_pos` is where to open it.
    // Nothing is emitted for a right-click on empty canvas.
    void contextMenuRequested(overlay::Item* item, const QPoint& global_pos);

protected:
    void paintEvent(QPaintEvent* event) override;
    void mousePressEvent(QMouseEvent* event) override;
    void mouseMoveEvent(QMouseEvent* event) override;
    void mouseReleaseEvent(QMouseEvent* event) override;
    // Right-click: hit-test as a press does, select, and hand the item
    // to whoever offers a menu. The editor opens none itself -- it has
    // no business knowing what is on one.
    void contextMenuEvent(QContextMenuEvent* event) override;
    // Delete removes the selection; the arrows nudge it. Nudging is
    // what a mouse cannot do: items are placed in normalized
    // coordinates, so the smallest useful drag is one widget pixel,
    // which is a different distance on every window size. A key press
    // is a fixed fraction of the canvas, so two callsigns can actually
    // be lined up.
    void keyPressEvent(QKeyEvent* event) override;
    void resizeEvent(QResizeEvent* event) override;

private:
    // **Rotate is a separate drag mode from Resize**, not a modifier on
    // it: the two grips sit at different corners (see `rotate_handle_rect`
    // vs `handle_rect`) precisely so a press can never be ambiguous
    // between them.
    enum class Drag { None, Move, Resize, Rotate };

    void rerender();
    // What is actually painted for `item`: substituted per `fields_`.
    // See `set_fields`.
    overlay::Item rendered(const overlay::Item& item) const;
    // Where the canvas is drawn inside the widget, letter-boxed.
    QRect canvas_rect() const;
    // Widget point -> canvas pixel. Outside the canvas is still mapped;
    // callers check the rect.
    QPointF to_canvas(const QPointF& widget_point) const;
    int hit_test(const QPointF& canvas_point) const;
    // The index a right-click at this widget point belongs to: the
    // selected item if the point is on its rotate or resize grip, else
    // whatever `hit_test` finds, else -1. The grips are tested first and
    // in `mousePressEvent`'s order, because they sit outside the item's
    // own area and would otherwise never be reachable.
    int hit_index(const QPointF& widget_point) const;
    // `item`'s bbox, mapped into this widget's own pixel coordinates --
    // shared by `paintEvent`'s selection box and `selection_screen_rect`,
    // which must agree about where the item sits on screen. **Not
    // rotated** -- an axis-aligned approximation is all
    // `selection_screen_rect` needs (a floating panel anchoring "near"
    // the item), and `paintEvent`'s own outline uses
    // `item_screen_polygon` instead, which is.
    QRect item_screen_rect(const overlay::Item& item) const;
    // The dashed selection outline: `box`'s four corners, rotated around
    // its own centre by `rotation` and mapped to screen pixels -- so the
    // outline turns with the item instead of staying axis-aligned while
    // the picture underneath it visibly rotates.
    QPolygon item_screen_polygon(const overlay::Bbox& box, double rotation) const;
    QRect handle_rect(const overlay::Bbox& box, double rotation) const;
    // Above and outside the top-right corner, offset from `handle_rect`
    // deliberately (see `Drag::Rotate`).
    QRect rotate_handle_rect(const overlay::Bbox& box, double rotation) const;
    // The grip's side, from the style rather than a pixel literal --
    // see the .cpp.
    int handle_px() const;
    // `point` (canvas space) rotated by `rotation_degrees` around
    // `centre` (canvas space), in the same sense `overlay::render`
    // rotates a painted item -- see the .cpp for the derivation. Shared
    // by the outline and both grips, so all three always agree about
    // where the item's corners actually are.
    static QPointF rotate_around(const QPointF& point, const QPointF& centre,
                                 double rotation_degrees);
    // Cursor feedback for the no-drag path of mouseMoveEvent.
    void update_hover_cursor(const QPointF& point);
    void select(int index);
    // The keyboard-shortcut forms of the two mouse drags: `+`/`-` and
    // `[`/`]` in `keyPressEvent`. Multiplicative and additive
    // respectively, matching what dragging the corresponding handle
    // does, and clamped to the same bounds `mouseMoveEvent` uses.
    static void scale_item(overlay::Item& item, double factor);
    static void rotate_item(overlay::Item& item, double delta_degrees);

    overlay::Doc doc_;
    overlay::Fields fields_;
    images::Picture base_;
    std::optional<images::Picture> last_rx_;
    // The rendered composite, cached because rendering is not free and a
    // repaint happens on every mouse move during a drag.
    images::Picture composed_;
    bool composed_valid_ = false;

    int selected_ = -1;
    Drag drag_ = Drag::None;
    // Canvas-space offset from the item's anchor to the grab point, so a
    // drag does not snap the item's corner to the cursor.
    QPointF grab_offset_;
    double resize_start_ = 0.0;
    // A rectangle resizes both axes together (see `mouseMoveEvent`), so
    // its starting height rides alongside `resize_start_`'s width. Text
    // and image items leave this at 0 and never read it.
    double resize_start_height_ = 0.0;
    QPointF resize_origin_;
    // Rotate drag state: the item's own rotation and the pointer's
    // angle around the pivot (radians, screen sense already flipped --
    // see the .cpp), both captured the moment the grip was grabbed. The
    // live rotation is `rotate_start_rotation_` plus however far the
    // pointer's angle has moved since, around `rotate_center_`.
    double rotate_start_rotation_ = 0.0;
    double rotate_start_pointer_angle_ = 0.0;
    QPointF rotate_center_;
};

}  // namespace sstvae::gui

#endif

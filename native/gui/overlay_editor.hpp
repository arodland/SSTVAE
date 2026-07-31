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

#include <QRect>
#include <QWidget>

#include <optional>
#include <string>

#include "images/types.hpp"
#include "overlay/model.hpp"
#include "overlay/render.hpp"

namespace sstvae::gui {

class OverlayEditor : public QWidget {
    Q_OBJECT

public:
    explicit OverlayEditor(QWidget* parent = nullptr);
    ~OverlayEditor() override;

    QSize sizeHint() const override;

    // The picture the overlay sits on, already framed to the transmit
    // size by the caller.
    void set_base_image(const images::Picture& image);
    bool has_base() const { return !base_.empty(); }

    // The most recent reception, for a "last_rx" inset. Late-bound on
    // purpose: an item referring to it keeps meaning "the most recent
    // one" rather than freezing today's picture into the document.
    void set_last_rx(const images::Picture& image);
    bool has_last_rx() const { return last_rx_.has_value(); }

    void add_text(const std::string& text);
    void add_image_inset(const std::string& path);
    void add_last_rx_inset();
    void remove_selected();
    void clear_overlay();

    // The selected item, or null. A pointer into the document, so the
    // property editor mutates it in place and calls `refresh_item`.
    overlay::Item* selected_item();
    void refresh_item();

    const overlay::Doc& doc() const { return doc_; }
    void set_doc(overlay::Doc doc);

    // Base plus overlay, or nothing if no picture has been chosen.
    std::optional<images::Picture> composed_image() const;

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

protected:
    void paintEvent(QPaintEvent* event) override;
    void mousePressEvent(QMouseEvent* event) override;
    void mouseMoveEvent(QMouseEvent* event) override;
    void mouseReleaseEvent(QMouseEvent* event) override;

private:
    enum class Drag { None, Move, Resize };

    void rerender();
    // Where the canvas is drawn inside the widget, letter-boxed.
    QRect canvas_rect() const;
    // Widget point -> canvas pixel. Outside the canvas is still mapped;
    // callers check the rect.
    QPointF to_canvas(const QPointF& widget_point) const;
    int hit_test(const QPointF& canvas_point) const;
    QRect handle_rect(const overlay::Bbox& box) const;
    void select(int index);

    overlay::Doc doc_;
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
    QPointF resize_origin_;
};

}  // namespace sstvae::gui

#endif

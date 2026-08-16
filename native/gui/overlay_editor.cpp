#include "overlay_editor.hpp"

#include <QImage>
#include <QKeyEvent>
#include <QMouseEvent>
#include <QPainter>
#include <QPen>
#include <QResizeEvent>
#include <QStyle>

#include <algorithm>
#include <cmath>
#include <utility>
#include <variant>

#include "images/images.hpp"
#include "overlay/render.hpp"
#include "style.hpp"

namespace sstvae::gui {

OverlayEditor::OverlayEditor(QWidget* parent) : QWidget(parent) {
    setMinimumSize(320, 240);
    // **No `setHeightForWidth`.** It is the third form of the same
    // ratchet as `setFixedHeight` (see picture_box.hpp), and the most
    // deceptive, because `minimumSizeHint()` never consults it -- a
    // test asserting on the hint sees nothing wrong. What Qt actually
    // applies at layout time is `minimumHeightForWidth(width)`, and
    // with the flag set that is `width * 3/4`.
    //
    // It stayed harmless for as long as it did because a `QSplitter`
    // does not propagate `hasHeightForWidth` and a `QTabWidget` does.
    // So the tabbed layout exposed it to the window for the first time,
    // and measured on the container: at 900 px wide it demanded 1107 px
    // of height, at 1400 px 1375, at **2020 px 1840**. The window grew
    // past the bottom of the screen, and switching back to side by side
    // did not shrink it again -- Qt lowers a minimum without resizing.
    //
    // Nothing is lost. `resizeEvent` caps the height at 4:3 (a maximum,
    // which constrains nothing upward) and `canvas_rect()` letterboxes
    // in both directions, so the canvas is still exactly 4:3 at any
    // shape this widget is given.
    // Expanding both ways: this is what absorbs the pane's spare
    // room. It imposes no cap of its own -- `canvas_rect()` centres
    // a 4:3 canvas in whatever it is given.
    setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    // **Tracking on, so the cursor can answer.** Nothing in this widget
    // announced itself: items drag, the corner grip resizes, the arrow
    // keys nudge, and the pointer stayed an arrow over all of it.
    // `CropView` in this same application deliberately sets a cursor and
    // explains why -- a widget that moves things under the pointer has
    // to say so before the pointer is pressed.
    setMouseTracking(true);
    setToolTip(tr("Drag an item to move it, or its corner to resize. Arrow "
                  "keys nudge the selection (Shift for a coarser step); "
                  "Delete removes it."));
    // Strong, not ClickFocus: the arrow keys and Delete are useless to
    // an operator who cannot get focus onto this widget, and ClickFocus
    // keeps it out of the Tab chain entirely.
    setFocusPolicy(Qt::StrongFocus);
}

OverlayEditor::~OverlayEditor() = default;

QSize OverlayEditor::sizeHint() const {
    return QSize(overlay::CANVAS_W, overlay::CANVAS_H);
}

void OverlayEditor::set_base_image(const images::Picture& image) {
    base_ = image;
    composed_valid_ = false;
    update();
    emit documentChanged();
}

bool OverlayEditor::uses_last_rx() const {
    for (const overlay::Item& item : doc_.items) {
        const auto* image = std::get_if<overlay::ImageItem>(&item);
        if (image != nullptr && image->source == overlay::SOURCE_LAST_RX) {
            return true;
        }
    }
    return false;
}

void OverlayEditor::set_last_rx(const images::Picture& image) {
    // Always stored, whether or not anything refers to it yet: the
    // reference is late-bound, so an inset added *later* must show the
    // newest reception rather than the one that happened to arrive
    // while the item existed.
    last_rx_ = image;

    // **But `documentChanged` only when the composition actually
    // depends on it.** A "last_rx" item is resolved at render time, so
    // a new reception genuinely changes what an existing item shows --
    // and that is worth re-rendering and re-optimizing for. With no
    // such item, nothing about what would be transmitted has moved, and
    // saying otherwise is expensive rather than merely untidy:
    // `TransmitPanel` connects this signal to `schedule_optimization`,
    // which abandons any speculative run in flight and starts a fresh
    // one on a 120 s budget across four onnxruntime threads.
    //
    // Measured on the panel, fifteen receptions arriving over 65 s:
    // **205 s of CPU with this emitted unconditionally against 0.30 s
    // with refinement off**, on a composition no item of which used the
    // received picture. Runs plateau at 40-100 s and pictures on a net
    // arrive every 32-95 s, so each arrival discarded the previous
    // run's work and restarted -- the optimizer never reached idle and
    // the only way back was to restart the application, which is
    // exactly what operators reported.
    if (!uses_last_rx()) return;
    composed_valid_ = false;
    update();
    emit documentChanged();
}

void OverlayEditor::add_text(const std::string& text) {
    overlay::TextItem item;
    item.text = text;
    doc_.items.push_back(item);
    select(static_cast<int>(doc_.items.size()) - 1);
    // The button that ran this has the focus, so the keyboard would
    // otherwise be dead on exactly the flow the nudge exists for --
    // add a callsign, then line it up against another.
    setFocus(Qt::OtherFocusReason);
    emit documentChanged();
}

void OverlayEditor::add_image_inset(const std::string& path) {
    overlay::ImageItem item;
    item.source = path;
    doc_.items.push_back(item);
    select(static_cast<int>(doc_.items.size()) - 1);
    // The button that ran this has the focus, so the keyboard would
    // otherwise be dead on exactly the flow the nudge exists for --
    // add a callsign, then line it up against another.
    setFocus(Qt::OtherFocusReason);
    emit documentChanged();
}

void OverlayEditor::add_last_rx_inset() {
    overlay::ImageItem item;  // defaults to SOURCE_LAST_RX
    doc_.items.push_back(item);
    select(static_cast<int>(doc_.items.size()) - 1);
    // The button that ran this has the focus, so the keyboard would
    // otherwise be dead on exactly the flow the nudge exists for --
    // add a callsign, then line it up against another.
    setFocus(Qt::OtherFocusReason);
    emit documentChanged();
}

void OverlayEditor::remove_selected() {
    if (selected_ < 0 || selected_ >= static_cast<int>(doc_.items.size())) return;
    doc_.items.erase(doc_.items.begin() + selected_);
    select(-1);
    emit documentChanged();
}

void OverlayEditor::clear_overlay() {
    doc_.items.clear();
    select(-1);
    emit documentChanged();
}

overlay::Item* OverlayEditor::selected_item() {
    if (selected_ < 0 || selected_ >= static_cast<int>(doc_.items.size())) {
        return nullptr;
    }
    return &doc_.items[selected_];
}

void OverlayEditor::refresh_item() {
    composed_valid_ = false;
    update();
    emit documentChanged();
}

void OverlayEditor::set_doc(overlay::Doc doc) {
    doc_ = std::move(doc);
    select(-1);
    emit documentChanged();
}

void OverlayEditor::select(int index) {
    selected_ = index;
    composed_valid_ = false;
    update();
    emit selectionChanged(selected_item());
}

std::optional<images::Picture> OverlayEditor::composed_image() const {
    if (base_.empty()) return std::nullopt;
    return overlay::render(base_, doc_, last_rx_ ? &*last_rx_ : nullptr);
}

void OverlayEditor::rerender() {
    if (base_.empty()) {
        composed_ = images::Picture();
    } else {
        composed_ = overlay::render(base_, doc_, last_rx_ ? &*last_rx_ : nullptr);
    }
    composed_valid_ = true;
}

QRect OverlayEditor::canvas_rect() const {
    // Letter-boxed, preserving the canvas aspect: the document's
    // coordinates are fractions of the transmitted frame, so stretching
    // the preview would put a handle somewhere the item is not.
    const double aspect =
        static_cast<double>(overlay::CANVAS_W) / overlay::CANVAS_H;
    int w = width();
    int h = static_cast<int>(std::lround(w / aspect));
    if (h > height()) {
        h = height();
        w = static_cast<int>(std::lround(h * aspect));
    }
    return QRect((width() - w) / 2, (height() - h) / 2, std::max(1, w),
                 std::max(1, h));
}

QPointF OverlayEditor::to_canvas(const QPointF& widget_point) const {
    const QRect rect = canvas_rect();
    const double sx = static_cast<double>(overlay::CANVAS_W) / rect.width();
    const double sy = static_cast<double>(overlay::CANVAS_H) / rect.height();
    return QPointF((widget_point.x() - rect.x()) * sx,
                   (widget_point.y() - rect.y()) * sy);
}

int OverlayEditor::hit_test(const QPointF& point) const {
    // Front to back, so the item drawn on top is the one you grab --
    // the same order the eye resolves an overlap in.
    for (int i = static_cast<int>(doc_.items.size()) - 1; i >= 0; --i) {
        const overlay::Bbox box =
            overlay::item_bbox(overlay::CANVAS_W, overlay::CANVAS_H, doc_.items[i],
                               last_rx_ ? &*last_rx_ : nullptr);
        if (point.x() >= box.x && point.x() < box.x + box.w &&
            point.y() >= box.y && point.y() < box.y + box.h) {
            return i;
        }
    }
    return -1;
}

// The side of the square resize grip.
//
// **Not a constant 10.** A logical-pixel literal is a fixed physical
// size only on the display it was chosen on; on a HiDPI panel at 200%
// scaling it is a ~3 mm target for the one gesture in this widget that
// needs precision. `PM_SmallIconSize` is the style's own answer to "how
// big is a small thing here", so it tracks both the screen and the
// font. Floored at the old value so nothing gets *worse*.
int OverlayEditor::handle_px() const {
    return std::max(10, style()->pixelMetric(QStyle::PM_SmallIconSize) * 2 / 3);
}

QRect OverlayEditor::handle_rect(const overlay::Bbox& box) const {
    const QRect rect = canvas_rect();
    const double sx = static_cast<double>(rect.width()) / overlay::CANVAS_W;
    const double sy = static_cast<double>(rect.height()) / overlay::CANVAS_H;
    const int x = rect.x() + static_cast<int>(std::lround((box.x + box.w) * sx));
    const int y = rect.y() + static_cast<int>(std::lround((box.y + box.h) * sy));
    const int side = handle_px();
    return QRect(x - side / 2, y - side / 2, side, side);
}

void OverlayEditor::paintEvent(QPaintEvent*) {
    QPainter painter(this);
    const QRect rect = canvas_rect();

    if (!composed_valid_) rerender();
    if (composed_.empty()) {
        // **Draw the empty canvas as a 4:3 box, not as nothing.** The
        // two panes are locked to the same width so the pictures are
        // the same size, but an empty composer that painted only its
        // own background made one side a dark rectangle and the other
        // a void -- so they measured equal and did not read equal. The
        // same fill and the same disabled text as `PictureBox`, which
        // is the receive side's empty state, so the pair is symmetric
        // before either has a picture in it.
        // Viewport, then the 4:3 frame inside it -- the composer has to
        // show the shape it will send before anything is in it, for the
        // same reason the receive box does.
        painter.fillRect(this->rect(), style::color::viewport());
        painter.fillRect(rect, style::color::viewport_frame());
        painter.setPen(style::color::viewport_edge());
        painter.drawRect(rect.adjusted(0, 0, -1, -1));
        painter.setPen(style::color::viewport_text());
        painter.drawText(rect, Qt::AlignCenter, tr("Choose an image to send"));
        return;
    }

    // The same fill as the empty state, so the viewport around the
    // canvas does not change colour the moment a picture arrives.
    painter.fillRect(this->rect(), style::color::viewport());
    painter.setRenderHint(QPainter::SmoothPixmapTransform, true);
    painter.drawImage(rect, style::to_qimage(composed_));

    if (overlay::Item* item = const_cast<OverlayEditor*>(this)->selected_item()) {
        const overlay::Bbox box = overlay::item_bbox(
            overlay::CANVAS_W, overlay::CANVAS_H, *item,
            last_rx_ ? &*last_rx_ : nullptr);
        const double sx = static_cast<double>(rect.width()) / overlay::CANVAS_W;
        const double sy = static_cast<double>(rect.height()) / overlay::CANVAS_H;
        const QRect on_screen(
            rect.x() + static_cast<int>(std::lround(box.x * sx)),
            rect.y() + static_cast<int>(std::lround(box.y * sy)),
            std::max(1, static_cast<int>(std::lround(box.w * sx))),
            std::max(1, static_cast<int>(std::lround(box.h * sy))));

        // Two-tone, so the outline is visible over both a bright and a
        // dark picture without knowing which it is.
        painter.setPen(QPen(QColor(0, 0, 0, 160), 3));
        painter.drawRect(on_screen);
        painter.setPen(QPen(QColor(255, 255, 255, 230), 1, Qt::DashLine));
        painter.drawRect(on_screen);
        painter.fillRect(handle_rect(box), QColor(255, 255, 255, 230));
        painter.setPen(QPen(QColor(0, 0, 0, 200), 1));
        painter.drawRect(handle_rect(box));
    }
}

void OverlayEditor::mousePressEvent(QMouseEvent* event) {
    if (event->button() != Qt::LeftButton) return;
    const QPointF point = event->position();

    // The grip first: it sits on the item's corner, so testing the item
    // before the handle would make the corner unresizable.
    if (overlay::Item* item = selected_item()) {
        const overlay::Bbox box = overlay::item_bbox(
            overlay::CANVAS_W, overlay::CANVAS_H, *item,
            last_rx_ ? &*last_rx_ : nullptr);
        if (handle_rect(box).contains(point.toPoint())) {
            drag_ = Drag::Resize;
            resize_origin_ = to_canvas(point);
            resize_start_ = std::holds_alternative<overlay::TextItem>(*item)
                                ? std::get<overlay::TextItem>(*item).size
                                : std::get<overlay::ImageItem>(*item).width;
            return;
        }
    }

    const QPointF canvas = to_canvas(point);
    const int index = hit_test(canvas);
    if (index != selected_) select(index);
    if (index < 0) {
        drag_ = Drag::None;
        return;
    }

    drag_ = Drag::Move;
    overlay::Item& item = doc_.items[index];
    const double x = std::visit([](const auto& i) { return i.x; }, item);
    const double y = std::visit([](const auto& i) { return i.y; }, item);
    grab_offset_ = QPointF(canvas.x() - x * overlay::CANVAS_W,
                           canvas.y() - y * overlay::CANVAS_H);
}

// What the pointer is over, expressed as a cursor.
//
// Split out and called from the no-drag path of `mouseMoveEvent`: the
// grip first, exactly as `mousePressEvent` tests it, so the cursor
// cannot promise a resize where a press would start a move.
void OverlayEditor::update_hover_cursor(const QPointF& point) {
    if (overlay::Item* item = selected_item()) {
        const overlay::Bbox box = overlay::item_bbox(
            overlay::CANVAS_W, overlay::CANVAS_H, *item,
            last_rx_ ? &*last_rx_ : nullptr);
        if (handle_rect(box).contains(point.toPoint())) {
            setCursor(Qt::SizeFDiagCursor);
            return;
        }
    }
    setCursor(hit_test(to_canvas(point)) >= 0 ? Qt::SizeAllCursor
                                              : Qt::ArrowCursor);
}

void OverlayEditor::mouseMoveEvent(QMouseEvent* event) {
    if (drag_ == Drag::None) {
        update_hover_cursor(event->position());
        return;
    }
    overlay::Item* item = selected_item();
    if (item == nullptr) return;
    const QPointF canvas = to_canvas(event->position());

    if (drag_ == Drag::Move) {
        // Stored normalized, never as pixels: that is what keeps a
        // document meaningful at another resolution.
        const double x = (canvas.x() - grab_offset_.x()) / overlay::CANVAS_W;
        const double y = (canvas.y() - grab_offset_.y()) / overlay::CANVAS_H;
        std::visit(
            [x, y](auto& i) {
                // Clamped loosely rather than to 0..1: an item may hang
                // off the edge deliberately, but it must not be dragged
                // somewhere it can never be grabbed again.
                i.x = std::clamp(x, -0.5, 1.5);
                i.y = std::clamp(y, -0.5, 1.5);
            },
            *item);
    } else {
        // Resize from the grabbed corner: the change in distance from
        // the item's anchor scales the size.
        const double x0 = std::visit([](const auto& i) { return i.x; }, *item) *
                          overlay::CANVAS_W;
        const double start = std::max(1.0, resize_origin_.x() - x0);
        const double now = std::max(1.0, canvas.x() - x0);
        const double factor = now / start;
        if (auto* text = std::get_if<overlay::TextItem>(item)) {
            text->size = std::clamp(resize_start_ * factor, 0.01, 1.5);
        } else if (auto* image = std::get_if<overlay::ImageItem>(item)) {
            image->width = std::clamp(resize_start_ * factor, 0.02, 2.0);
        }
    }
    composed_valid_ = false;
    update();
    emit selectionChanged(item);
    emit documentChanged();
}

void OverlayEditor::mouseReleaseEvent(QMouseEvent* event) {
    Q_UNUSED(event);
    drag_ = Drag::None;
}

void OverlayEditor::resizeEvent(QResizeEvent* event) {
    QWidget::resizeEvent(event);
    // **Nothing here, deliberately.** This widget used to cap its own
    // height at 4:3 -- first with `setFixedHeight`, which is a hard
    // *minimum* and made a wide pane raise a window floor that
    // narrowing never lowered (measured 611 px at 545 wide, 925 at
    // 1348, 1274 at 1900); then with a maximum, which was safe but was
    // still one of two caps on one property, so whichever `resizeEvent`
    // ran last decided the size.
    //
    // Now the canvas is drawn *inside* the widget rather than being the
    // widget, so none of it is needed: `canvas_rect()` letterboxes in
    // both directions, the composer is exactly 4:3 at any shape this is
    // handed, and nothing is imposed upward on the window.
}

void OverlayEditor::keyPressEvent(QKeyEvent* event) {
    overlay::Item* item = selected_item();
    if (item == nullptr) {
        QWidget::keyPressEvent(event);
        return;
    }

    if (event->key() == Qt::Key_Delete || event->key() == Qt::Key_Backspace) {
        remove_selected();
        event->accept();
        return;
    }

    // A fraction of the canvas, not a pixel: positions are normalized,
    // so a fixed step means the same nudge whatever the window size.
    // Shift is the coarse step, for getting somewhere; the fine one is
    // roughly a canvas pixel at 640 wide.
    constexpr double FINE = 1.0 / 640.0;
    constexpr double COARSE = 1.0 / 64.0;
    const double step =
        (event->modifiers() & Qt::ShiftModifier) ? COARSE : FINE;

    double dx = 0.0;
    double dy = 0.0;
    switch (event->key()) {
        case Qt::Key_Left: dx = -step; break;
        case Qt::Key_Right: dx = step; break;
        case Qt::Key_Up: dy = -step; break;
        case Qt::Key_Down: dy = step; break;
        default:
            QWidget::keyPressEvent(event);
            return;
    }

    // The same clamp a drag uses, so an item cannot be nudged somewhere
    // a drag could not have put it.
    std::visit(
        [dx, dy](auto& i) {
            i.x = std::clamp(i.x + dx, -0.5, 1.5);
            i.y = std::clamp(i.y + dy, -0.5, 1.5);
        },
        *item);
    refresh_item();
    event->accept();
}

}  // namespace sstvae::gui

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
#include <numbers>
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
    setToolTip(tr("Drag an item to move it, its corner to resize, or the "
                  "small circle above it to rotate. Arrow keys nudge the "
                  "selection (Shift for a coarser step); +/- scales it, "
                  "[ and ] rotate it; Delete removes it."));
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

void OverlayEditor::add_rect() {
    overlay::RectItem item;
    // A visible default rather than the model's own defaults (fill and
    // stroke both "none"): every other Add button places something the
    // operator can immediately see and select, and an invisible
    // rectangle looks exactly like the button doing nothing.
    item.fill_kind = "solid";
    item.fill_color = "#ffffff";
    doc_.items.push_back(item);
    select(static_cast<int>(doc_.items.size()) - 1);
    setFocus(Qt::OtherFocusReason);
    emit documentChanged();
}

void OverlayEditor::remove_selected() {
    if (selected_ < 0 || selected_ >= static_cast<int>(doc_.items.size())) return;
    doc_.items.erase(doc_.items.begin() + selected_);
    select(-1);
    emit documentChanged();
}

bool OverlayEditor::can_raise_selected() const {
    return selected_ >= 0 && selected_ + 1 < static_cast<int>(doc_.items.size());
}

bool OverlayEditor::can_lower_selected() const {
    return selected_ > 0 && selected_ < static_cast<int>(doc_.items.size());
}

void OverlayEditor::raise_selected() {
    if (!can_raise_selected()) return;
    std::swap(doc_.items[selected_], doc_.items[selected_ + 1]);
    // The item moved with the swap; the selection index follows it so
    // the same item stays selected rather than whatever is now sitting
    // at the old index.
    select(selected_ + 1);
    emit documentChanged();
}

void OverlayEditor::lower_selected() {
    if (!can_lower_selected()) return;
    std::swap(doc_.items[selected_], doc_.items[selected_ - 1]);
    select(selected_ - 1);
    emit documentChanged();
}

void OverlayEditor::bring_selected_to_front() {
    if (!can_raise_selected()) return;
    // Rotate the range [selected_, end) left by one: the selected item
    // lands at the back of the vector (drawn last, i.e. on top) and
    // everything above it shifts down one slot to make room, keeping
    // their own relative order -- a single `std::swap` against the last
    // element would instead trade places with whatever was on top,
    // silently reordering the items in between.
    std::rotate(doc_.items.begin() + selected_, doc_.items.begin() + selected_ + 1,
               doc_.items.end());
    select(static_cast<int>(doc_.items.size()) - 1);
    emit documentChanged();
}

void OverlayEditor::send_selected_to_back() {
    if (!can_lower_selected()) return;
    // The mirror image: rotate [begin, selected_] right by one, same
    // reasoning.
    std::rotate(doc_.items.begin(), doc_.items.begin() + selected_,
               doc_.items.begin() + selected_ + 1);
    select(0);
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

void OverlayEditor::set_fields(overlay::Fields fields) {
    fields_ = std::move(fields);
    composed_valid_ = false;
    update();
    // A field change is a composition change exactly like a drag: the
    // picture that would be sent is different now, so the speculative
    // optimizer must be told, debounced the same way a text edit is
    // (`TransmitPanel` connects a field box's textChanged to
    // `set_fields` followed by nothing else -- this signal is the only
    // notification either needs).
    emit documentChanged();
}

// What is actually drawn for `item`: `.text` substituted, everything
// else (position, size, rotation, anchor, color, the image source)
// copied through untouched, since substitution only ever touches
// TextItem::text. Wrapping a single item in a one-item Doc reuses
// `overlay::substitute` rather than duplicating its rules here, at the
// cost of one short-lived vector -- cheap next to the font-metrics work
// `item_bbox` already does on every one of these call sites.
overlay::Item OverlayEditor::rendered(const overlay::Item& item) const {
    overlay::Doc one;
    one.items.push_back(item);
    return overlay::substitute(one, fields_).items[0];
}

void OverlayEditor::select(int index) {
    selected_ = index;
    composed_valid_ = false;
    update();
    emit selectionChanged(selected_item());
}

std::optional<images::Picture> OverlayEditor::composed_image() const {
    if (base_.empty()) return std::nullopt;
    return overlay::render(base_, overlay::substitute(doc_, fields_),
                           last_rx_ ? &*last_rx_ : nullptr);
}

QRect OverlayEditor::item_screen_rect(const overlay::Item& item) const {
    const overlay::Bbox box = overlay::item_bbox(
        overlay::CANVAS_W, overlay::CANVAS_H, rendered(item), last_rx_ ? &*last_rx_ : nullptr);
    const QRect rect = canvas_rect();
    const double sx = static_cast<double>(rect.width()) / overlay::CANVAS_W;
    const double sy = static_cast<double>(rect.height()) / overlay::CANVAS_H;
    return QRect(rect.x() + static_cast<int>(std::lround(box.x * sx)),
                rect.y() + static_cast<int>(std::lround(box.y * sy)),
                std::max(1, static_cast<int>(std::lround(box.w * sx))),
                std::max(1, static_cast<int>(std::lround(box.h * sy))));
}

QRect OverlayEditor::selection_screen_rect() const {
    const overlay::Item* item = const_cast<OverlayEditor*>(this)->selected_item();
    if (item == nullptr) return QRect();
    return item_screen_rect(*item);
}

// Same transform `overlay::render` applies to a rotated item's pixels
// (`painter.rotate(-item.rotation)` about the item's own centre): the
// document's angle is counter-clockwise, as PIL's is, and
// `QTransform::rotate` turns the other way, hence the sign flip here
// too. Working this out algebraically rather than pushing points through
// a `QTransform` keeps the handles and the outline in plain canvas-space
// arithmetic, matching every other geometry helper in this file.
QPointF OverlayEditor::rotate_around(const QPointF& point, const QPointF& centre,
                                     double rotation_degrees) {
    if (rotation_degrees == 0.0) return point;
    const double rad = rotation_degrees * std::numbers::pi / 180.0;
    const double c = std::cos(rad);
    const double s = std::sin(rad);
    const double dx = point.x() - centre.x();
    const double dy = point.y() - centre.y();
    return QPointF(centre.x() + dx * c + dy * s, centre.y() - dx * s + dy * c);
}

QPolygon OverlayEditor::item_screen_polygon(const overlay::Bbox& box,
                                            double rotation) const {
    const QRect rect = canvas_rect();
    const double sx = static_cast<double>(rect.width()) / overlay::CANVAS_W;
    const double sy = static_cast<double>(rect.height()) / overlay::CANVAS_H;
    const QPointF centre(box.x + box.w / 2.0, box.y + box.h / 2.0);
    const QPointF corners[4] = {
        QPointF(box.x, box.y), QPointF(box.x + box.w, box.y),
        QPointF(box.x + box.w, box.y + box.h), QPointF(box.x, box.y + box.h)};
    QPolygon polygon;
    for (const QPointF& corner : corners) {
        const QPointF rotated = rotate_around(corner, centre, rotation);
        polygon << QPoint(rect.x() + static_cast<int>(std::lround(rotated.x() * sx)),
                          rect.y() + static_cast<int>(std::lround(rotated.y() * sy)));
    }
    return polygon;
}

void OverlayEditor::scale_item(overlay::Item& item, double factor) {
    if (auto* text = std::get_if<overlay::TextItem>(&item)) {
        text->size = std::clamp(text->size * factor, 0.01, 1.5);
    } else if (auto* rect = std::get_if<overlay::RectItem>(&item)) {
        // Both axes together, matching the corner-drag resize -- see
        // `mouseMoveEvent`'s identical reasoning.
        rect->width = std::clamp(rect->width * factor, 0.02, 2.0);
        rect->height = std::clamp(rect->height * factor, 0.02, 2.0);
    } else if (auto* image = std::get_if<overlay::ImageItem>(&item)) {
        image->width = std::clamp(image->width * factor, 0.02, 2.0);
    }
}

void OverlayEditor::rotate_item(overlay::Item& item, double delta_degrees) {
    std::visit(
        [delta_degrees](auto& i) {
            // Normalized into (-180, 180] rather than left to wind up
            // past 360 -- the same range the old rotation spin box
            // offered, and a small negative angle is easier to read
            // than its 350-odd-degree equivalent.
            double normalized = std::fmod(i.rotation + delta_degrees, 360.0);
            if (normalized > 180.0) normalized -= 360.0;
            if (normalized <= -180.0) normalized += 360.0;
            i.rotation = normalized;
        },
        item);
}

void OverlayEditor::rerender() {
    if (base_.empty()) {
        composed_ = images::Picture();
    } else {
        composed_ = overlay::render(base_, overlay::substitute(doc_, fields_),
                                    last_rx_ ? &*last_rx_ : nullptr);
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
    // the same order the eye resolves an overlap in. Against the
    // *rendered* bbox (see `set_fields`), so a hole a template dropped
    // (rule 2) cannot be clicked, and one substitution lengthened is
    // grabbable over its whole painted extent.
    for (int i = static_cast<int>(doc_.items.size()) - 1; i >= 0; --i) {
        const overlay::Bbox box =
            overlay::item_bbox(overlay::CANVAS_W, overlay::CANVAS_H, rendered(doc_.items[i]),
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

QRect OverlayEditor::handle_rect(const overlay::Bbox& box, double rotation) const {
    // The item's own bottom-right corner, rotated around the box's
    // centre so the grip stays on that corner as the item turns -- the
    // same reason the selection outline is a rotated polygon rather than
    // a static rect. `rotate_around` is a no-op at zero rotation, so this
    // costs nothing on an unrotated item.
    const QPointF centre(box.x + box.w / 2.0, box.y + box.h / 2.0);
    const QPointF corner =
        rotate_around(QPointF(box.x + box.w, box.y + box.h), centre, rotation);

    const QRect rect = canvas_rect();
    const double sx = static_cast<double>(rect.width()) / overlay::CANVAS_W;
    const double sy = static_cast<double>(rect.height()) / overlay::CANVAS_H;
    const int x = rect.x() + static_cast<int>(std::lround(corner.x() * sx));
    const int y = rect.y() + static_cast<int>(std::lround(corner.y() * sy));
    const int side = handle_px();
    return QRect(x - side / 2, y - side / 2, side, side);
}

QRect OverlayEditor::rotate_handle_rect(const overlay::Bbox& box, double rotation) const {
    // The item's own top-right corner, rotated the same way `handle_rect`
    // rotates the bottom-right one -- the two stay clear of each other
    // at any angle because they track different corners of the same box,
    // not because either one's own screen offset (below) rotates with
    // it. That offset is therefore left fixed (up and to the right, in
    // screen pixels): its only job is to sit this grip visibly apart
    // from the resize one, which rotating the corner already guarantees.
    const QPointF centre(box.x + box.w / 2.0, box.y + box.h / 2.0);
    const QPointF corner = rotate_around(QPointF(box.x + box.w, box.y), centre, rotation);

    const QRect rect = canvas_rect();
    const double sx = static_cast<double>(rect.width()) / overlay::CANVAS_W;
    const double sy = static_cast<double>(rect.height()) / overlay::CANVAS_H;
    const int side = handle_px();
    const int offset = side * 2;
    int x = rect.x() + static_cast<int>(std::lround(corner.x() * sx)) + offset;
    int y = rect.y() + static_cast<int>(std::lround(corner.y() * sy)) - offset;

    // Clamped within the canvas: an item near an edge would otherwise
    // push this handle past the picture (or the widget) entirely, since
    // the offset above is added on top of wherever the rotated corner
    // itself already landed.
    x = std::clamp(x - side / 2, rect.x(), rect.x() + std::max(0, rect.width() - side));
    y = std::clamp(y - side / 2, rect.y(), rect.y() + std::max(0, rect.height() - side));
    return QRect(x, y, side, side);
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
    // canvas does not change color the moment a picture arrives.
    painter.fillRect(this->rect(), style::color::viewport());
    painter.setRenderHint(QPainter::SmoothPixmapTransform, true);
    painter.drawImage(rect, style::to_qimage(composed_));

    // **A "last received" inset with nothing received yet paints
    // nothing at all** -- correctly: `overlay::render` is also what
    // encodes the transmission, so it must never draw a placeholder
    // that could go out over the air in place of a picture. But that
    // leaves the item invisible on this preview too, before the
    // operator has clicked anything to find it -- a real gap for a
    // template that starts with one already in it (docs/overlay-
    // templates.md's "Reply with picture"), unreachable by anything on
    // screen until a reception arrives. So the *editor* draws its own
    // frame here, over the composed picture rather than into it, for
    // every such item -- not only the selected one, so it is findable
    // before it is clicked. Same look as the empty-canvas state above
    // and `PictureBox`'s own "no picture" frame.
    if (!last_rx_) {
        for (const overlay::Item& doc_item : doc_.items) {
            const auto* image = std::get_if<overlay::ImageItem>(&doc_item);
            if (image == nullptr || image->source != overlay::SOURCE_LAST_RX) continue;
            const QRect box = item_screen_rect(doc_item);
            painter.fillRect(box, style::color::viewport_frame());
            painter.setPen(style::color::viewport_edge());
            painter.drawRect(box.adjusted(0, 0, -1, -1));
            painter.setPen(style::color::viewport_text());
            painter.drawText(box, Qt::AlignCenter | Qt::TextWordWrap,
                             tr("No picture received yet"));
        }
    }

    if (overlay::Item* item = const_cast<OverlayEditor*>(this)->selected_item()) {
        const overlay::Bbox box = overlay::item_bbox(
            overlay::CANVAS_W, overlay::CANVAS_H, rendered(*item),
            last_rx_ ? &*last_rx_ : nullptr);
        const double rotation =
            std::visit([](const auto& i) { return i.rotation; }, *item);
        const QPolygon on_screen = item_screen_polygon(box, rotation);

        // Two-tone, so the outline is visible over both a bright and a
        // dark picture without knowing which it is.
        painter.setPen(QPen(QColor(0, 0, 0, 160), 3));
        painter.drawPolygon(on_screen);
        painter.setPen(QPen(QColor(255, 255, 255, 230), 1, Qt::DashLine));
        painter.drawPolygon(on_screen);
        painter.setPen(QPen(QColor(0, 0, 0, 200), 1));
        painter.setBrush(QColor(255, 255, 255, 230));
        painter.drawRect(handle_rect(box, rotation));
        // The rotate grip is a circle rather than a square, so the two
        // read as different kinds of control at a glance rather than as
        // two identical squares that happen to do different things.
        painter.drawEllipse(rotate_handle_rect(box, rotation));
    }
}

void OverlayEditor::mousePressEvent(QMouseEvent* event) {
    if (event->button() != Qt::LeftButton) return;
    const QPointF point = event->position();

    // The grips first: they sit outside the item's own area, so testing
    // the item before them would make neither one reachable.
    if (overlay::Item* item = selected_item()) {
        const overlay::Bbox box = overlay::item_bbox(
            overlay::CANVAS_W, overlay::CANVAS_H, rendered(*item),
            last_rx_ ? &*last_rx_ : nullptr);
        const double rotation =
            std::visit([](const auto& i) { return i.rotation; }, *item);
        if (rotate_handle_rect(box, rotation).contains(point.toPoint())) {
            drag_ = Drag::Rotate;
            const QPointF center(box.x + box.w / 2.0, box.y + box.h / 2.0);
            rotate_center_ = center;
            rotate_start_rotation_ = rotation;
            const QPointF canvas = to_canvas(point);
            // Screen y grows downward, so this negates it: a positive
            // angle here then means "counter-clockwise as the operator
            // sees it", matching the document's own rotation sense.
            rotate_start_pointer_angle_ =
                std::atan2(-(canvas.y() - center.y()), canvas.x() - center.x());
            return;
        }
        if (handle_rect(box, rotation).contains(point.toPoint())) {
            drag_ = Drag::Resize;
            resize_origin_ = to_canvas(point);
            resize_start_height_ = 0.0;
            if (const auto* text = std::get_if<overlay::TextItem>(item)) {
                resize_start_ = text->size;
            } else if (const auto* rect = std::get_if<overlay::RectItem>(item)) {
                resize_start_ = rect->width;
                resize_start_height_ = rect->height;
            } else {
                resize_start_ = std::get<overlay::ImageItem>(*item).width;
            }
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
            overlay::CANVAS_W, overlay::CANVAS_H, rendered(*item),
            last_rx_ ? &*last_rx_ : nullptr);
        const double rotation =
            std::visit([](const auto& i) { return i.rotation; }, *item);
        if (rotate_handle_rect(box, rotation).contains(point.toPoint())) {
            // Qt has no built-in rotate cursor; a cross is at least not
            // one of the shapes already claimed by move or resize.
            setCursor(Qt::CrossCursor);
            return;
        }
        if (handle_rect(box, rotation).contains(point.toPoint())) {
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
    } else if (drag_ == Drag::Resize) {
        // Resize from the grabbed corner: the change in distance from
        // the item's anchor scales the size.
        const double x0 = std::visit([](const auto& i) { return i.x; }, *item) *
                          overlay::CANVAS_W;
        const double start = std::max(1.0, resize_origin_.x() - x0);
        const double now = std::max(1.0, canvas.x() - x0);
        const double factor = now / start;
        if (auto* text = std::get_if<overlay::TextItem>(item)) {
            text->size = std::clamp(resize_start_ * factor, 0.01, 1.5);
        } else if (auto* rect = std::get_if<overlay::RectItem>(item)) {
            // Both axes scale together, from the same horizontal drag
            // distance every other item type resizes with -- so the
            // rectangle's own aspect ratio is preserved rather than
            // stretching only its width.
            rect->width = std::clamp(resize_start_ * factor, 0.02, 2.0);
            rect->height = std::clamp(resize_start_height_ * factor, 0.02, 2.0);
        } else if (auto* image = std::get_if<overlay::ImageItem>(item)) {
            image->width = std::clamp(resize_start_ * factor, 0.02, 2.0);
        }
    } else {  // Drag::Rotate
        const double angle = std::atan2(-(canvas.y() - rotate_center_.y()),
                                        canvas.x() - rotate_center_.x());
        const double delta_deg =
            (angle - rotate_start_pointer_angle_) * 180.0 / std::numbers::pi;
        // Not `rotate_item` (which adds to the item's *current* value):
        // a drag reads back the same delta on every move, so it must be
        // applied against the rotation captured at the press, not
        // compounded onto whatever the previous move step left behind.
        std::visit(
            [this, delta_deg](auto& i) {
                double normalized =
                    std::fmod(rotate_start_rotation_ + delta_deg, 360.0);
                if (normalized > 180.0) normalized -= 360.0;
                if (normalized <= -180.0) normalized += 360.0;
                i.rotation = normalized;
            },
            *item);
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

    const bool coarse = event->modifiers() & Qt::ShiftModifier;

    // +/- and [/] are the keyboard forms of dragging the resize and
    // rotate grips -- multiplicative and additive respectively, exactly
    // what each drag does, and clamped/normalized by the same
    // `scale_item`/`rotate_item` the grips would end up at.
    switch (event->key()) {
        case Qt::Key_Plus:
        case Qt::Key_Equal:
            scale_item(*item, coarse ? 1.10 : 1.02);
            refresh_item();
            event->accept();
            return;
        case Qt::Key_Minus:
        case Qt::Key_Underscore:
            scale_item(*item, 1.0 / (coarse ? 1.10 : 1.02));
            refresh_item();
            event->accept();
            return;
        case Qt::Key_BracketLeft:
            rotate_item(*item, coarse ? 15.0 : 1.0);
            refresh_item();
            event->accept();
            return;
        case Qt::Key_BracketRight:
            rotate_item(*item, coarse ? -15.0 : -1.0);
            refresh_item();
            event->accept();
            return;
        default:
            break;
    }

    // A fraction of the canvas, not a pixel: positions are normalized,
    // so a fixed step means the same nudge whatever the window size.
    // Shift is the coarse step, for getting somewhere; the fine one is
    // roughly a canvas pixel at 640 wide.
    constexpr double FINE = 1.0 / 640.0;
    constexpr double COARSE = 1.0 / 64.0;
    const double step = coarse ? COARSE : FINE;

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

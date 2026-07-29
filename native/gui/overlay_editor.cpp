#include "overlay_editor.hpp"

#include <QImage>
#include <QMouseEvent>
#include <QPainter>
#include <QPen>

#include <algorithm>
#include <cmath>
#include <utility>

#include "images/images.hpp"
#include "overlay/render.hpp"

namespace sstvae::gui {

namespace {

// Side of the square resize grip, in widget pixels.
constexpr int HANDLE = 10;

QImage to_qimage(const images::Picture& picture) {
    const QImage view(picture.rgb.data(), picture.width, picture.height,
                      picture.width * 3, QImage::Format_RGB888);
    return view.copy();
}

}  // namespace

OverlayEditor::OverlayEditor(QWidget* parent) : QWidget(parent) {
    setMinimumSize(320, 240);
    setMouseTracking(false);
    setFocusPolicy(Qt::ClickFocus);
}

OverlayEditor::~OverlayEditor() = default;

QSize OverlayEditor::sizeHint() const {
    return QSize(overlay::CANVAS_W, overlay::CANVAS_H);
}

void OverlayEditor::set_base_image(const images::Picture& image) {
    base_ = image;
    composed_valid_ = false;
    update();
}

void OverlayEditor::set_last_rx(const images::Picture& image) {
    last_rx_ = image;
    // A "last_rx" item is resolved at render time, so a new reception
    // changes what an existing item shows -- which is the point.
    composed_valid_ = false;
    update();
}

void OverlayEditor::add_text(const std::string& text) {
    overlay::TextItem item;
    item.text = text;
    doc_.items.push_back(item);
    select(static_cast<int>(doc_.items.size()) - 1);
}

void OverlayEditor::add_image_inset(const std::string& path) {
    overlay::ImageItem item;
    item.source = path;
    doc_.items.push_back(item);
    select(static_cast<int>(doc_.items.size()) - 1);
}

void OverlayEditor::add_last_rx_inset() {
    overlay::ImageItem item;  // defaults to SOURCE_LAST_RX
    doc_.items.push_back(item);
    select(static_cast<int>(doc_.items.size()) - 1);
}

void OverlayEditor::remove_selected() {
    if (selected_ < 0 || selected_ >= static_cast<int>(doc_.items.size())) return;
    doc_.items.erase(doc_.items.begin() + selected_);
    select(-1);
}

void OverlayEditor::clear_overlay() {
    doc_.items.clear();
    select(-1);
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
}

void OverlayEditor::set_doc(overlay::Doc doc) {
    doc_ = std::move(doc);
    select(-1);
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

QRect OverlayEditor::handle_rect(const overlay::Bbox& box) const {
    const QRect rect = canvas_rect();
    const double sx = static_cast<double>(rect.width()) / overlay::CANVAS_W;
    const double sy = static_cast<double>(rect.height()) / overlay::CANVAS_H;
    const int x = rect.x() + static_cast<int>(std::lround((box.x + box.w) * sx));
    const int y = rect.y() + static_cast<int>(std::lround((box.y + box.h) * sy));
    return QRect(x - HANDLE / 2, y - HANDLE / 2, HANDLE, HANDLE);
}

void OverlayEditor::paintEvent(QPaintEvent*) {
    QPainter painter(this);
    const QRect rect = canvas_rect();

    if (!composed_valid_) rerender();
    if (composed_.empty()) {
        painter.fillRect(this->rect(), palette().window());
        painter.setPen(palette().color(QPalette::Disabled, QPalette::WindowText));
        painter.drawText(this->rect(), Qt::AlignCenter,
                         tr("Choose a picture to send"));
        return;
    }

    painter.fillRect(this->rect(), Qt::black);
    painter.setRenderHint(QPainter::SmoothPixmapTransform, true);
    painter.drawImage(rect, to_qimage(composed_));

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

void OverlayEditor::mouseMoveEvent(QMouseEvent* event) {
    if (drag_ == Drag::None) return;
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
}

void OverlayEditor::mouseReleaseEvent(QMouseEvent* event) {
    Q_UNUSED(event);
    drag_ = Drag::None;
}

}  // namespace sstvae::gui

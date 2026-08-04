#include "crop_dialog.hpp"

#include <QDialogButtonBox>
#include <QHBoxLayout>
#include <QImage>
#include <QLabel>
#include <QMouseEvent>
#include <QPainter>
#include <QPainterPath>
#include <QPixmap>
#include <QPushButton>
#include <QSlider>
#include <QVBoxLayout>
#include <QWheelEvent>

#include <algorithm>
#include <cmath>

namespace sstvae::gui {

namespace {

// Zoom is exposed as integer slider steps because QSlider is integral;
// 100 steps is 1.00x (exactly cover) and the top is a 4x crop, which is
// tighter than anyone sensibly wants and still leaves the arithmetic
// well away from the single-pixel end.
constexpr int ZOOM_MIN = 100;
constexpr int ZOOM_MAX = 400;

double steps_to_zoom(int steps) { return steps / 100.0; }
int zoom_to_steps(double zoom) {
    return std::clamp(static_cast<int>(std::lround(zoom * 100.0)), ZOOM_MIN, ZOOM_MAX);
}

QPixmap to_pixmap(const images::Picture& picture) {
    if (picture.empty()) return QPixmap();
    const QImage view(picture.rgb.data(), picture.width, picture.height,
                      picture.width * 3, QImage::Format_RGB888);
    return QPixmap::fromImage(view.copy());
}

}  // namespace

// --- CropView ---------------------------------------------------------------

CropView::CropView(QWidget* parent) : QWidget(parent) {
    setMinimumSize(280, 210);
    // A move cursor, not a hand: `paintEvent` holds the picture still
    // and moves the window over it, so what the pointer carries is the
    // *window*. A hand would promise the other model -- picture sliding
    // under a fixed frame -- and the two want opposite drag signs.
    setCursor(Qt::SizeAllCursor);
}

QSize CropView::sizeHint() const { return {520, 390}; }

void CropView::set_source(const images::Picture& source) {
    source_ = source;
    pixmap_ = to_pixmap(source);
    update();
}

void CropView::set_framing(const images::Framing& framing) {
    framing_ = framing;
    clamp_center();
    update();
}

QRectF CropView::source_rect() const {
    if (source_.empty()) return {};
    const double sw = source_.width;
    const double sh = source_.height;
    const double scale = std::min(width() / sw, height() / sh);
    const double w = sw * scale;
    const double h = sh * scale;
    return {(width() - w) / 2.0, (height() - h) / 2.0, w, h};
}

QRectF CropView::crop_rect() const {
    const QRectF area = source_rect();
    if (area.isEmpty()) return {};

    // The crop window's size as a *fraction of the source*, which is
    // what the framing means: at zoom 1 the window is the largest 4:3
    // rectangle that fits, and zoom shrinks it from there.
    const double src_aspect =
        static_cast<double>(source_.width) / source_.height;
    const double target_aspect =
        static_cast<double>(images::IMG_W) / images::IMG_H;

    double frac_w = 1.0;
    double frac_h = 1.0;
    if (src_aspect > target_aspect) {
        frac_w = target_aspect / src_aspect;  // wider than 4:3: lose width
    } else {
        frac_h = src_aspect / target_aspect;  // taller: lose height
    }
    const double zoom = std::max(1.0, framing_.zoom);
    frac_w /= zoom;
    frac_h /= zoom;

    const double w = area.width() * frac_w;
    const double h = area.height() * frac_h;
    const double cx = area.left() + area.width() * framing_.center_x;
    const double cy = area.top() + area.height() * framing_.center_y;
    return {cx - w / 2.0, cy - h / 2.0, w, h};
}

void CropView::clamp_center() {
    framing_.zoom = std::clamp(framing_.zoom, steps_to_zoom(ZOOM_MIN),
                               steps_to_zoom(ZOOM_MAX));
    if (source_.empty()) return;
    const QRectF area = source_rect();
    if (area.isEmpty()) return;
    const QRectF crop = crop_rect();
    // Half the window's size, as a fraction of the source: the centre
    // cannot come closer to an edge than that without the window
    // hanging over nothing.
    const double half_w = crop.width() / area.width() / 2.0;
    const double half_h = crop.height() / area.height() / 2.0;
    framing_.center_x = std::clamp(framing_.center_x, half_w, 1.0 - half_w);
    framing_.center_y = std::clamp(framing_.center_y, half_h, 1.0 - half_h);
}

void CropView::paintEvent(QPaintEvent* event) {
    Q_UNUSED(event);
    QPainter painter(this);
    painter.fillRect(rect(), palette().window());
    if (pixmap_.isNull()) return;

    const QRectF area = source_rect();
    // Smooth, like every other scaled picture in this app: a full-size
    // photograph into a ~520 px preview is a 6x downscale, and
    // nearest-neighbour point-sampling moires fine detail badly -- in a
    // dialog whose whole job is showing the operator what they are
    // giving up.
    painter.setRenderHint(QPainter::SmoothPixmapTransform, true);
    painter.drawPixmap(area, pixmap_, QRectF(pixmap_.rect()));

    // Everything outside the window is dimmed rather than hidden, so
    // the operator can see what is being given up -- which is the whole
    // complaint this dialog answers.
    const QRectF crop = crop_rect();
    QPainterPath outside;
    outside.addRect(area);
    QPainterPath inside;
    inside.addRect(crop);
    painter.fillPath(outside.subtracted(inside), QColor(0, 0, 0, 140));

    // A two-tone border, the same idiom the overlay editor's selection
    // uses, so it reads on any picture.
    painter.setPen(QPen(QColor(0, 0, 0, 160), 3));
    painter.drawRect(crop);
    painter.setPen(QPen(QColor(255, 255, 255, 230), 1));
    painter.drawRect(crop);
}

void CropView::mousePressEvent(QMouseEvent* event) {
    if (event->button() != Qt::LeftButton) {
        QWidget::mousePressEvent(event);
        return;
    }
    dragging_ = true;
    drag_last_ = event->position();
}

void CropView::mouseMoveEvent(QMouseEvent* event) {
    if (!dragging_) return;
    const QRectF area = source_rect();
    if (area.isEmpty()) return;
    const QPointF delta = event->position() - drag_last_;
    drag_last_ = event->position();
    // The window follows the pointer, because the window is what moves
    // on screen: `source_rect` has no framing dependence, so the
    // picture is fixed and `crop_rect` is what the centre moves. The
    // opposite sign would be right for a widget that panned the
    // picture instead -- and it is the sign this had, so dragging right
    // selected further *left*.
    framing_.center_x += delta.x() / area.width();
    framing_.center_y += delta.y() / area.height();
    clamp_center();
    update();
    emit framingChanged();
}

void CropView::mouseReleaseEvent(QMouseEvent* event) {
    if (event->button() == Qt::LeftButton) dragging_ = false;
    QWidget::mouseReleaseEvent(event);
}

void CropView::wheelEvent(QWheelEvent* event) {
    // Accumulated to whole notches rather than one step per event. A
    // conventional wheel delivers 120 units per detent, but a trackpad
    // or a high-resolution wheel delivers a handful of units per event
    // -- so "one event, one 1.1x step" turns a small two-finger gesture
    // into a run from 1x to the 4x ceiling.
    wheel_accum_ += event->angleDelta().y();
    constexpr int NOTCH = 120;
    bool moved = false;
    while (wheel_accum_ >= NOTCH) {
        wheel_accum_ -= NOTCH;
        framing_.zoom *= 1.1;
        moved = true;
    }
    while (wheel_accum_ <= -NOTCH) {
        wheel_accum_ += NOTCH;
        framing_.zoom /= 1.1;
        moved = true;
    }
    if (moved) {
        clamp_center();
        update();
        emit framingChanged();
    }
    event->accept();
}

// --- CropDialog -------------------------------------------------------------

CropDialog::CropDialog(const images::Picture& source,
                       const images::Framing& initial, QWidget* parent)
    : QDialog(parent) {
    setWindowTitle(tr("Frame picture"));

    auto* layout = new QVBoxLayout(this);

    summary_ = new QLabel(this);
    summary_->setText(tr("%1x%2 source, sent as %3x%4. Drag to reposition; "
                         "the wheel or the slider zooms.")
                          .arg(source.width)
                          .arg(source.height)
                          .arg(images::IMG_W)
                          .arg(images::IMG_H));
    summary_->setWordWrap(true);
    layout->addWidget(summary_);

    view_ = new CropView(this);
    view_->set_source(source);
    view_->set_framing(initial);
    connect(view_, &CropView::framingChanged, this, &CropDialog::on_view_changed);
    layout->addWidget(view_, 1);

    auto* zoom_row = new QHBoxLayout();
    zoom_row->addWidget(new QLabel(tr("Zoom:"), this));
    zoom_ = new QSlider(Qt::Horizontal, this);
    zoom_->setRange(ZOOM_MIN, ZOOM_MAX);
    zoom_->setValue(zoom_to_steps(view_->framing().zoom));
    connect(zoom_, &QSlider::valueChanged, this, &CropDialog::on_zoom_changed);
    zoom_row->addWidget(zoom_, 1);
    auto* reset = new QPushButton(tr("Reset"), this);
    reset->setToolTip(tr("Back to the centred, full-width framing"));
    connect(reset, &QPushButton::clicked, this, &CropDialog::reset_framing);
    zoom_row->addWidget(reset);
    layout->addLayout(zoom_row);

    auto* buttons =
        new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel, this);
    connect(buttons, &QDialogButtonBox::accepted, this, &QDialog::accept);
    connect(buttons, &QDialogButtonBox::rejected, this, &QDialog::reject);
    // Enter accepts, and the default framing is the centre crop this
    // function has always done -- so an operator who does not care pays
    // one keypress and gets exactly the old behaviour.
    buttons->button(QDialogButtonBox::Ok)->setDefault(true);
    layout->addWidget(buttons);
}

images::Framing CropDialog::framing() const { return view_->framing(); }

void CropDialog::on_zoom_changed(int steps) {
    if (syncing_) return;
    images::Framing framing = view_->framing();
    framing.zoom = steps_to_zoom(steps);
    view_->set_framing(framing);
}

void CropDialog::on_view_changed() {
    // The wheel changes zoom inside the view, so the slider has to
    // follow -- guarded, or its valueChanged would drive the view back.
    syncing_ = true;
    zoom_->setValue(zoom_to_steps(view_->framing().zoom));
    syncing_ = false;
}

void CropDialog::reset_framing() {
    view_->set_framing(images::Framing{});
    on_view_changed();
}

}  // namespace sstvae::gui

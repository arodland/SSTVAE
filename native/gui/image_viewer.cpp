#include "image_viewer.hpp"

#include <QGuiApplication>
#include <QKeyEvent>
#include <QLabel>
#include <QMouseEvent>
#include <QPainter>
#include <QPalette>
#include <QScreen>
#include <QVBoxLayout>
#include <QWheelEvent>

#include <algorithm>
#include <cmath>

#include "style.hpp"

namespace sstvae::gui {

ImageView::ImageView(QWidget* parent) : QWidget(parent) {
    setObjectName(QStringLiteral("image_view"));
    setAutoFillBackground(true);
    QPalette dark = palette();
    dark.setColor(QPalette::Window, style::color::viewport());
    setPalette(dark);
    setFocusPolicy(Qt::StrongFocus);
    setCursor(Qt::OpenHandCursor);
    setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
}

QSize ImageView::sizeHint() const {
    return pixmap_.isNull() ? QSize(640, 480) : pixmap_.size();
}

void ImageView::set_pixmap(const QPixmap& pixmap) {
    pixmap_ = pixmap;
    fit();
}

double ImageView::fit_zoom() const {
    if (pixmap_.isNull() || width() <= 0 || height() <= 0) return 1.0;
    return std::clamp(std::min(static_cast<double>(width()) / pixmap_.width(),
                               static_cast<double>(height()) / pixmap_.height()),
                      MIN_ZOOM, MAX_ZOOM);
}

void ImageView::fit() {
    fit_ = true;
    zoom_ = fit_zoom();
    clamp_offset();
    update();
    emit zoomChanged(zoom_);
}

void ImageView::zoom_about(QPointF anchor, double factor) {
    if (pixmap_.isNull()) return;
    const double z = std::clamp(zoom_ * factor, MIN_ZOOM, MAX_ZOOM);
    if (z == zoom_) return;
    const QPointF p = to_picture(anchor);
    zoom_ = z;
    fit_ = false;
    offset_ = anchor - p * zoom_;
    clamp_offset();
    update();
    emit zoomChanged(zoom_);
}

void ImageView::set_zoom(double zoom) {
    if (zoom_ <= 0) return;
    zoom_about(QPointF(width() / 2.0, height() / 2.0), zoom / zoom_);
}

void ImageView::pan(QPointF delta) {
    offset_ += delta;
    clamp_offset();
    update();
}

QPointF ImageView::to_widget(QPointF p) const { return offset_ + p * zoom_; }
QPointF ImageView::to_picture(QPointF w) const { return (w - offset_) / zoom_; }

QRectF ImageView::picture_rect() const {
    return QRectF(offset_, QSizeF(pixmap_.width() * zoom_, pixmap_.height() * zoom_));
}

// A picture narrower than the window sits centred; a wider one can be
// dragged only as far as its own edges, so it never leaves the window.
void ImageView::clamp_offset() {
    const double pw = pixmap_.width() * zoom_;
    const double ph = pixmap_.height() * zoom_;
    const auto axis = [](double off, double pic, double win) {
        if (pic <= win) return (win - pic) / 2.0;
        return std::clamp(off, win - pic, 0.0);
    };
    offset_.setX(axis(offset_.x(), pw, width()));
    offset_.setY(axis(offset_.y(), ph, height()));
}

void ImageView::paintEvent(QPaintEvent* event) {
    QWidget::paintEvent(event);
    if (pixmap_.isNull()) return;
    QPainter painter(this);
    painter.setRenderHint(QPainter::SmoothPixmapTransform, zoom_ < 2.0);
    painter.drawPixmap(picture_rect(), pixmap_, QRectF(pixmap_.rect()));
}

void ImageView::resizeEvent(QResizeEvent* event) {
    QWidget::resizeEvent(event);
    if (fit_) {
        fit();
    } else {
        clamp_offset();
    }
}

void ImageView::wheelEvent(QWheelEvent* event) {
    const double notches = event->angleDelta().y() / 120.0;
    if (notches == 0.0) return;
    zoom_about(event->position(), std::pow(WHEEL_STEP, notches));
    event->accept();
}

void ImageView::mousePressEvent(QMouseEvent* event) {
    if (event->button() != Qt::LeftButton) return;
    dragging_ = true;
    drag_from_ = event->position();
    setCursor(Qt::ClosedHandCursor);
}

void ImageView::mouseMoveEvent(QMouseEvent* event) {
    if (!dragging_) return;
    pan(event->position() - drag_from_);
    drag_from_ = event->position();
}

void ImageView::mouseReleaseEvent(QMouseEvent* event) {
    if (event->button() != Qt::LeftButton) return;
    dragging_ = false;
    setCursor(Qt::OpenHandCursor);
}

void ImageView::mouseDoubleClickEvent(QMouseEvent* event) {
    if (event->button() != Qt::LeftButton) return;
    // Fitted: 100% about the point clicked. Otherwise back to fitting.
    if (fit_) {
        zoom_about(event->position(), 1.0 / zoom_);
        fit_ = false;
    } else {
        fit();
    }
}

void ImageView::keyPressEvent(QKeyEvent* event) {
    const QPointF centre(width() / 2.0, height() / 2.0);
    switch (event->key()) {
    case Qt::Key_Plus:
    case Qt::Key_Equal: zoom_about(centre, WHEEL_STEP); break;
    case Qt::Key_Minus: zoom_about(centre, 1.0 / WHEEL_STEP); break;
    case Qt::Key_0: fit(); break;
    case Qt::Key_1: set_zoom(1.0); break;
    default: QWidget::keyPressEvent(event); return;
    }
    event->accept();
}

// --- the window ------------------------------------------------------------------------

ImageViewer::ImageViewer(const QPixmap& pixmap, const QString& title, QWidget* parent)
    : QDialog(parent, Qt::Window) {
    setObjectName(QStringLiteral("image_viewer"));
    setWindowTitle(title);
    auto* box = new QVBoxLayout(this);
    box->setContentsMargins(0, 0, 0, 0);
    box->setSpacing(0);
    view_ = new ImageView(this);
    box->addWidget(view_, 1);
    caption_ = new QLabel(this);
    caption_->setObjectName(QStringLiteral("image_viewer_caption"));
    caption_->setContentsMargins(6, 3, 6, 3);
    box->addWidget(caption_);
    connect(view_, &ImageView::zoomChanged, this, &ImageViewer::update_caption);
    view_->set_pixmap(pixmap);

    // Big enough to show the picture at 100% where the screen allows it.
    QSize want = pixmap.size() + QSize(0, caption_->sizeHint().height());
    if (const QScreen* screen = parent != nullptr ? parent->screen() : QGuiApplication::primaryScreen()) {
        want = want.boundedTo(screen->availableGeometry().size() * 0.85);
    }
    resize(want.expandedTo(QSize(320, 240)));
    view_->setFocus();
    update_caption();
}

void ImageViewer::add_footer(QWidget* footer) {
    footer->setParent(this);
    footer->setContentsMargins(6, 0, 6, 6);
    static_cast<QVBoxLayout*>(layout())->addWidget(footer);
}

void ImageViewer::update_caption() {
    const QPixmap& p = view_->pixmap();
    caption_->setText(tr("%1 x %2, %3%%4   Wheel zooms, drag pans, double-click fits or "
                         "shows 100%")
                          .arg(p.width())
                          .arg(p.height())
                          .arg(std::lround(view_->zoom() * 100.0))
                          .arg(view_->fitted() ? tr(" (fit)") : QString()));
}

ImageViewer* open_image_viewer(const QPixmap& pixmap, const QString& title, QWidget* parent) {
    if (pixmap.isNull()) return nullptr;
    auto* viewer = new ImageViewer(pixmap, title, parent);
    viewer->setAttribute(Qt::WA_DeleteOnClose);
    viewer->show();
    return viewer;
}

}  // namespace sstvae::gui

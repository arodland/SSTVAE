#include "picture_box.hpp"

#include <QLabel>
#include <QPalette>
#include <QPainter>
#include <QResizeEvent>
#include <QSizePolicy>

#include <algorithm>

#include "images/images.hpp"

namespace sstvae::gui {

PictureBox::PictureBox(const QString& text, QWidget* parent) : QWidget(parent) {
    // **The dark fills the whole widget**, not just the picture. The
    // spare width in a pane wider than 4:3 needs then reads as viewport
    // rather than as a gap someone forgot to close -- and it lets this
    // widget stop caring about its own height, which is what retired
    // the cap that kept fighting the layout.
    //
    // Palette, not a stylesheet: a stylesheet anywhere makes Qt wrap the
    // application style in QStyleSheetStyle, which resets padding to
    // zero across every combo and spin box in the app -- including the
    // settings dialog, which has nothing to do with this widget.
    setAutoFillBackground(true);
    QPalette dark = palette();
    dark.setColor(QPalette::Window, QColor(0x20, 0x20, 0x24));
    dark.setColor(QPalette::WindowText, QColor(0x88, 0x88, 0x88));
    setPalette(dark);

    label_ = new QLabel(text, this);
    label_->setAlignment(Qt::AlignCenter);
    // Transparent: the box behind it is already the right colour, and a
    // second opaque rectangle would put a visible seam at the 4:3 edge.
    label_->setAttribute(Qt::WA_TranslucentBackground);

    // Expanding in both directions: this is the thing that should absorb
    // the pane's spare room. It asks for 4:3 through `sizeHint` and is
    // content with anything.
    setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
}

QSize PictureBox::sizeHint() const {
    return QSize(width(), std::max(MIN_H, width() * images::IMG_H / images::IMG_W));
}

void PictureBox::set_picture(const QPixmap& picture) {
    source_ = picture;
    rescale();
}

QRect PictureBox::picture_rect() const { return label_->geometry(); }

void PictureBox::resizeEvent(QResizeEvent* event) {
    QWidget::resizeEvent(event);

    // The largest 4:3 rectangle that fits, centred. No height cap, no
    // external limit, nothing imposed upward -- this widget's only job
    // is to place a rectangle inside whatever it was handed.
    int w = width();
    int h = height();
    if (w * images::IMG_H > h * images::IMG_W) {
        w = h * images::IMG_W / images::IMG_H;
    } else {
        h = w * images::IMG_H / images::IMG_W;
    }
    label_->setGeometry((width() - w) / 2, (height() - h) / 2, w, h);

    // Rescaled here rather than from the panel's resizeEvent, which is a
    // staleness trap: by this point the label's geometry is the one it
    // will keep.
    rescale();
}

void PictureBox::paintEvent(QPaintEvent* event) {
    QWidget::paintEvent(event);
    if (!source_.isNull()) return;  // the picture is its own frame

    QPainter painter(this);
    const QRect frame = label_->geometry();
    // A shade lighter than the viewport, plus a hairline: enough to read
    // as "the picture goes here" without competing with a picture once
    // one arrives. Colours from the palette this widget already carries,
    // never a stylesheet -- one stylesheet anywhere re-styles every combo
    // and spin box in the application.
    painter.fillRect(frame, QColor(0x31, 0x31, 0x3a));
    painter.setPen(QColor(0x55, 0x55, 0x61));
    painter.drawRect(frame.adjusted(0, 0, -1, -1));
}

void PictureBox::rescale() {
    if (source_.isNull()) return;
    label_->setPixmap(
        source_.scaled(label_->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
}

}  // namespace sstvae::gui

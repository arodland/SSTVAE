// Choosing which part of a non-4:3 picture goes on the air.
//
// The codec sends 4:3, and `images::fit` has always resolved anything
// else by scaling to cover and cropping the centre -- silently. A 16:9
// photograph lost a quarter of its width and a portrait lost most of
// its height, with no indication that anything had been discarded and
// no way to choose differently. Worse, the editor letterboxes the
// *already-cropped* canvas inside its widget, which reads as "your
// photo was letterboxed" when it was cut.
//
// So: when a chosen picture is not 4:3, this dialog opens on top of
// the original with an aspect-locked window over it. Accepting the
// default is exactly the old behaviour, one keypress away, which is
// what keeps it out of the way of an operator in a hurry.
//
// The zoom travels *out* as well as in, down to the point where the
// window covers the whole source and the parts of the canvas the
// picture cannot fill go out black. That reverses the 2026-08-01 "no
// letterbox" decision -- it is airtime spent on black, but which
// trade-off is right belongs to the operator, and the alternative was
// padding the file in another program. The end of the travel is
// `images::min_zoom`, shared with `images::fit` so the preview and the
// transmitted picture agree about where it is.
//
// The dialog owns no picture data beyond the source it is handed; what
// it returns is a `images::Framing`, so the caller keeps the original
// and can re-open this to adjust.

#ifndef SSTVAE_GUI_CROP_DIALOG_HPP
#define SSTVAE_GUI_CROP_DIALOG_HPP

#include <QDialog>
#include <QWidget>

#include "images/images.hpp"
#include "images/types.hpp"

class QLabel;
class QSlider;

namespace sstvae::gui {

// The interactive part: the source picture with a draggable,
// aspect-locked crop window over it and the excluded area dimmed.
//
// A painted QWidget rather than a QGraphicsView, for the same reason
// the overlay editor is one: there is a single thing to draw and a
// scene would be a second representation that can drift from the
// geometry `images::fit` actually uses.
class CropView : public QWidget {
    Q_OBJECT

public:
    explicit CropView(QWidget* parent = nullptr);

    void set_source(const images::Picture& source);
    void set_framing(const images::Framing& framing);
    const images::Framing& framing() const { return framing_; }
    // The end of the outward travel for the current source, which the
    // dialog needs to size its slider.
    double min_zoom() const;

    QSize sizeHint() const override;

signals:
    void framingChanged();

protected:
    void paintEvent(QPaintEvent* event) override;
    void mousePressEvent(QMouseEvent* event) override;
    void mouseMoveEvent(QMouseEvent* event) override;
    void mouseReleaseEvent(QMouseEvent* event) override;
    void wheelEvent(QWheelEvent* event) override;

private:
    // Where the whole source picture is drawn inside this widget,
    // letterboxed to preserve its aspect -- and shrunk far enough that
    // the *widest* crop window still fits, so the window's edges never
    // leave the widget and the operator can see the black it is
    // enclosing.
    QRectF source_rect() const;
    // The crop window's size as a fraction of the source, at a given
    // zoom. Above 1 in an axis means the window overhangs the picture,
    // which is what `fit` pads.
    QSizeF crop_fractions(double zoom) const;
    // The crop window in widget coordinates, derived from `framing_` --
    // never stored, so what is drawn cannot disagree with what is
    // returned.
    QRectF crop_rect() const;
    // Keep the window inside the picture: the centre's legal range
    // shrinks as the window grows.
    void clamp_center();

    images::Picture source_;
    QPixmap pixmap_;
    images::Framing framing_;
    bool dragging_ = false;
    QPointF drag_last_;
    // Wheel deltas below one detent are kept rather than acted on, so a
    // trackpad's small events add up to the same step a mouse notch
    // gives in one.
    int wheel_accum_ = 0;
};

class CropDialog : public QDialog {
    Q_OBJECT

public:
    // `source` is the picture as loaded, at its own size.
    CropDialog(const images::Picture& source, const images::Framing& initial,
               QWidget* parent = nullptr);

    images::Framing framing() const;

private slots:
    void on_zoom_changed(int steps);
    void on_view_changed();
    void reset_framing();

private:
    CropView* view_ = nullptr;
    QSlider* zoom_ = nullptr;
    QLabel* summary_ = nullptr;
    bool syncing_ = false;
};

}  // namespace sstvae::gui

#endif

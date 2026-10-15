// A picture in a window of its own, to look at closely: double-click a
// received picture (the SSTVAE preview, a QRSS tile) to open one.
//
// The mouse wheel zooms about the pointer, so the spot under it stays
// put; dragging pans; a double-click toggles between fitting the window
// and one picture pixel per screen pixel. Keys: + and - zoom about the
// centre, 0 fits, 1 is 100%, Escape closes. Past 200% the pixels are
// drawn as squares rather than smoothed, since the reason to zoom that
// far into a decoded picture is to see what the decoder actually made.

#ifndef SSTVAE_GUI_IMAGE_VIEWER_HPP
#define SSTVAE_GUI_IMAGE_VIEWER_HPP

#include <QDialog>
#include <QPixmap>
#include <QPointF>
#include <QWidget>

class QLabel;

namespace sstvae::gui {

class ImageView : public QWidget {
    Q_OBJECT

public:
    static constexpr double MIN_ZOOM = 0.05;
    static constexpr double MAX_ZOOM = 32.0;
    static constexpr double WHEEL_STEP = 1.25;   // per notch

    explicit ImageView(QWidget* parent = nullptr);

    void set_pixmap(const QPixmap& pixmap);
    const QPixmap& pixmap() const { return pixmap_; }

    // Screen pixels per picture pixel.
    double zoom() const { return zoom_; }
    bool fitted() const { return fit_; }
    // Zoom by `factor`, keeping the picture point under `anchor` (a
    // widget position) where it is.
    void zoom_about(QPointF anchor, double factor);
    void set_zoom(double zoom);   // about the centre
    void fit();
    // Move the picture by `delta` screen pixels.
    void pan(QPointF delta);

    // Where picture point `p` (picture pixels) is drawn, and back.
    QPointF to_widget(QPointF p) const;
    QPointF to_picture(QPointF w) const;
    // The picture's rectangle on screen.
    QRectF picture_rect() const;

    QSize sizeHint() const override;

signals:
    void zoomChanged(double zoom);

protected:
    void paintEvent(QPaintEvent* event) override;
    void resizeEvent(QResizeEvent* event) override;
    void wheelEvent(QWheelEvent* event) override;
    void mousePressEvent(QMouseEvent* event) override;
    void mouseMoveEvent(QMouseEvent* event) override;
    void mouseReleaseEvent(QMouseEvent* event) override;
    void mouseDoubleClickEvent(QMouseEvent* event) override;
    void keyPressEvent(QKeyEvent* event) override;

private:
    double fit_zoom() const;
    void clamp_offset();

    QPixmap pixmap_;
    double zoom_ = 1.0;
    QPointF offset_;   // widget position of the picture's top-left
    bool fit_ = true;
    bool dragging_ = false;
    QPointF drag_from_;
};

class ImageViewer : public QDialog {
    Q_OBJECT

public:
    ImageViewer(const QPixmap& pixmap, const QString& title, QWidget* parent = nullptr);
    ImageView* view() const { return view_; }
    // A row of the caller's under the caption (a QRSS picture's buttons).
    void add_footer(QWidget* footer);

private:
    void update_caption();

    ImageView* view_ = nullptr;
    QLabel* caption_ = nullptr;
};

// Open a viewer for `pixmap` (deleted when closed) and return it, or
// nullptr for an empty pixmap.
ImageViewer* open_image_viewer(const QPixmap& pixmap, const QString& title, QWidget* parent);

}  // namespace sstvae::gui

#endif

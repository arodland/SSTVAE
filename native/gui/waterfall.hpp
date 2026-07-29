// Scrolling spectrum display, plus the input level meter beside it.
//
// A port of `sstvae/gui/waterfall.py`, including its central sizing
// decision, which is not cosmetic:
//
// **The backing image is kept exactly the widget's size, so the painter
// never rescales it.** Scrolling is clean when the destination is an
// integer multiple of the source: at a k-times upscale, source row i
// lands at exactly i*k and a one-row shift moves the picture by exactly
// k pixels. The bad direction is downscaling, which is where this
// widget was -- 640 rows into a pane a few hundred pixels tall. There,
// source row i lands at floor(i * height / rows), a one-row shift moves
// the picture by a fraction of a pixel, and every frame re-quantises
// differently, so the rows crawl and shimmer. k = 1 is the simplest
// member of that family and the best one here: the scroll advances a
// single pixel per tick, and k > 1 would jump k pixels and hold k times
// less history for the same pane.
//
// The frequency axis has the same problem in the same direction -- 384
// bins squeezed into a ~280 px column -- so rows are reduced to the
// widget's width when they are computed rather than by the painter; see
// `dsp::reduce_to_width`.
//
// History depth therefore follows the widget height: at ~20 fps a
// 700-pixel pane holds a bit over half a minute.
//
// Audio comes from the same RingBuffer the decoder reads, via `tail()`
// -- a display-sized slice, never `snapshot()`. The decode loop already
// tore holes in its own audio once by copying the whole buffer under
// the lock; a widget repainting 20 times a second must not go near it.

#ifndef SSTVAE_GUI_WATERFALL_HPP
#define SSTVAE_GUI_WATERFALL_HPP

#include <QImage>
#include <QWidget>

#include <memory>

namespace sstvae::rx {
class RingBuffer;
}

namespace sstvae::gui {

class Waterfall : public QWidget {
    Q_OBJECT

public:
    explicit Waterfall(QWidget* parent = nullptr, int fps = 20);
    ~Waterfall() override;

    QSize sizeHint() const override;

    // The buffer to read; may be null, and is replaced wholesale when
    // receive restarts (a fresh buffer after a transmission is how the
    // tail of our own signal is kept out of the decoder).
    void set_ring(std::shared_ptr<rx::RingBuffer> ring);
    void clear();

protected:
    void paintEvent(QPaintEvent* event) override;
    void resizeEvent(QResizeEvent* event) override;

private slots:
    // A slot rather than a plain member so a test can drive one frame
    // directly instead of waiting on the timer -- which would make the
    // test a stopwatch, and this project does not assert on time.
    void tick();

private:
    void ensure_image();
    void draw_band_markers(QPainter& painter);
    void draw_level_meter(QPainter& painter);

    std::shared_ptr<rx::RingBuffer> ring_;
    // Exactly widget-sized; see the header comment.
    QImage image_;
    double peak_ = 0.0;
    bool clipping_ = false;
};

}  // namespace sstvae::gui

#endif

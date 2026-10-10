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
// The frequency axis has the same problem, and had it in the same
// direction while this widget was a narrow column: 384 bins squeezed
// into ~280 px. Rows are therefore reduced to the widget's width when
// they are computed rather than by the painter (`dsp::reduce_to_width`,
// peak-hold when shrinking so a one-bin carrier cannot be sampled
// away). As a strip it is usually *wider* than 384 px and that function
// interpolates instead -- the shrinking path still matters whenever the
// pane is dragged narrow, which the splitter allows.
//
// History depth therefore follows the widget height: at ~20 fps the
// default 160-pixel strip holds about eight seconds, and dragging the
// splitter down buys more.
//
// The rows are kept (dsp::WaterfallHistory, ten minutes) in a feed the
// QRSSTVAE spectrogram shares (spectrum_feed.hpp), so a resize or a
// taller strip is redrawn from them rather than left black.
//
// Audio comes from the same RingBuffer the decoder reads, via `tail()`
// -- a display-sized slice, never `snapshot()`. The decode loop already
// tore holes in its own audio once by copying the whole buffer under
// the lock; a widget repainting 20 times a second must not go near it.

#ifndef SSTVAE_GUI_WATERFALL_HPP
#define SSTVAE_GUI_WATERFALL_HPP

#include <QImage>
#include <QString>
#include <QWidget>

class QTimer;

#include <cstdint>
#include <memory>
#include <vector>

namespace sstvae::rx {
class RingBuffer;
}

namespace sstvae::gui {

class SpectrumFeed;

// "<-45s", "<-30m", "<-2h05m": how long ago a dial change was, for the
// marks on this strip and the QRSS spectrogram.
QString dial_mark_label(double age_s);

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

    // Draw from a shared feed (spectrum_feed.hpp) instead of this
    // widget's own: rows arrive when the feed pumps, and the widget's
    // timer stops. `set_ring` then sets the feed's ring.
    void set_feed(SpectrumFeed* feed);
    SpectrumFeed* feed() const { return feed_; }

    // The clip indicator latches: a peak at or over unity marks the
    // display until the operator clicks the meter. The instantaneous
    // red bar reverts as soon as the peak passes, which for a 20 fps
    // meter means a clipped over could come and go entirely unseen.
    // Latched state is what turns "was it clipping while I was away?"
    // into a question with an answer -- which is also why replacing
    // the ring does *not* clear it.
    bool clip_latched() const { return clip_latched_; }

    // Times (unix seconds) the radio's dial was changed: drawn as green
    // lines across the strip at the row that was newest then, labelled
    // with how long ago ("<-3m").
    void set_dial_marks(std::vector<double> times);
    void clear_clip();

protected:
    void paintEvent(QPaintEvent* event) override;
    void resizeEvent(QResizeEvent* event) override;
    void mousePressEvent(QMouseEvent* event) override;

private slots:
    // A slot rather than a plain member so a test can drive one frame
    // directly instead of waiting on the timer -- which would make the
    // test a stopwatch, and this project does not assert on time.
    void tick();

private:
    void ensure_image();
    void draw_band_markers(QPainter& painter);
    void draw_level_meter(QPainter& painter);
    void draw_disabled_scrim(QPainter& painter);
    void draw_dial_marks(QPainter& painter);

    // Paint the rows the feed has gained since the last call.
    void draw_new_rows();
    void paint_row(uchar* dst, const std::vector<double>& db, int width) const;

    std::unique_ptr<SpectrumFeed> own_feed_;
    SpectrumFeed* feed_ = nullptr;
    QTimer* timer_ = nullptr;
    // One past the newest history row painted at the top of the image.
    std::uint64_t drawn_ = 0;
    // Exactly widget-sized; see the header comment.
    QImage image_;
    double peak_ = 0.0;
    bool clipping_ = false;
    bool clip_latched_ = false;
    std::vector<double> dial_marks_;
    // Wall time when the newest row was drawn, so a mark's row is
    // counted back from it (and freezes with the rows when capture stops).
    double newest_row_time_ = 0.0;
};

}  // namespace sstvae::gui

#endif

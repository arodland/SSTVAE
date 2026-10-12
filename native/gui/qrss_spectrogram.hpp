// The QRSSTVAE mode's spectrogram, after Argo, with a "Time Lens" after
// Glissando: frequency up the screen, time across it, the newest audio
// entering at the left edge and unfurling to the right as it ages.
//
// The time axis is not linear (dsp::TimeScale). It is a bubble lens: the
// newest audio, at the left, is magnified to about a second per pixel,
// and the magnification falls away smoothly, with no edge, to fifteen
// seconds per pixel for hours of history, where a carrier too weak to
// see second by second shows up in the average. The mouse wheel over the
// time axis changes the scale under the pointer (the lens's or the
// history's), and over the frequency axis zooms frequency about the
// pointer; a double-click on the frequency axis goes back to the whole
// QRSS range. Dragging an axis pans it, the point grabbed staying under
// the pointer: the frequency axis up and down, the time axis back into
// the history (which freezes the view there; a double-click on the time
// axis, or dragging back to the present, goes live again).
//
// Inverse greyscale by default, stronger darker, which is what weak
// signal work wants; the sun/moon in the lower right corner flips it.
// Signals the QRSS listener has synced on (its tiles) are drawn in shades
// of red over their own frequency and pass instead of grey.
//
// The audio is the receive pane's capture ring, through the spectrum
// feed this shares with the waterfall (spectrum_feed.hpp), so this hears
// what the listener hears and keeps six hours of it. The arithmetic is in
// core/dsp/slow_spectrogram.hpp; this file draws it.

#ifndef SSTVAE_GUI_QRSS_SPECTROGRAM_HPP
#define SSTVAE_GUI_QRSS_SPECTROGRAM_HPP

#include <QImage>
#include <QRect>
#include <QWidget>

#include <cstdint>
#include <functional>
#include <memory>
#include <unordered_map>
#include <vector>

#include "dsp/slow_spectrogram.hpp"

class QTimer;

namespace sstvae::rx {
class RingBuffer;
}

namespace sstvae::gui {

class SpectrumFeed;

class QrssSpectrogram : public QWidget {
    Q_OBJECT

public:
    // A synced signal: its carrier and the time its pass covers.
    // A signal's pass: its carrier (CE) or pilot (L), its time, and how
    // far below and above f_hz it reaches.
    struct Marker {
        double f_hz = 0.0;
        double t0 = 0.0;
        double t1 = 0.0;
        double below_hz = MARK_HALF_HZ;
        double above_hz = MARK_HALF_HZ;
    };
    // Half the width of a CE signal: what a marker colours either side
    // of its carrier.
    static constexpr double MARK_HALF_HZ = 25.0;
    // An L signal sits from 22.1 Hz below its pilot to 3.1 Hz above.
    static constexpr double MARK_L_BELOW_HZ = 22.1;
    static constexpr double MARK_L_ABOVE_HZ = 3.1;
    static constexpr double F_LO = 300.0;
    static constexpr double F_HI = 2700.0;
    // Strength is drawn as how many noise deviations a pixel stands
    // above its column's noise floor: nothing at Z_FLOOR, full at
    // Z_FLOOR + Z_RANGE.
    static constexpr double Z_FLOOR = 1.5;
    static constexpr double Z_RANGE = 10.0;

    explicit QrssSpectrogram(QWidget* parent = nullptr);
    ~QrssSpectrogram() override;

    void set_ring(std::shared_ptr<rx::RingBuffer> ring);
    // Draw from a shared feed (spectrum_feed.hpp) instead of this
    // widget's own; `set_ring` then sets the feed's ring.
    void set_feed(SpectrumFeed* feed);
    SpectrumFeed* feed() const { return feed_; }
    // Off in SSTVAE mode: no redraw timer.
    void set_active(bool on);
    // Audio whose last sample was captured at `end_time` (unix seconds).
    void push_audio(const std::vector<double>& samples, double end_time);
    void set_markers(std::vector<Marker> markers);
    // Times (unix seconds) the radio's dial was changed: green lines up
    // the plot, each labelled with how long ago ("<-30m").
    void set_dial_marks(std::vector<double> times);
    // For tests: the wall clock, in unix seconds.
    void set_clock(std::function<double()> clock);

    // True: light background, stronger signals darker (the default).
    bool inverted() const { return inverted_; }
    void set_inverted(bool on);

    const dsp::TimeScale& scale() const { return scale_; }
    dsp::TimeScale& scale() { return scale_; }
    double f_lo() const { return f_lo_; }
    double f_hi() const { return f_hi_; }
    void set_frequency_range(double lo, double hi);
    // The time at the plot's left edge when panned back, or live
    // (following the clock).
    bool live() const { return view_end_ <= 0.0; }
    double view_end() const { return live() ? now() : view_end_; }
    // Pin the left edge at `t`; at or past now, or before the oldest
    // audio, it is clamped (now means live).
    void set_view_end(double t);
    void go_live();
    const dsp::SlowSpectrogram& data() const;

    // Where things are, for tests and for the mouse.
    QRect plot_rect() const;
    QRect time_axis_rect() const;
    QRect freq_axis_rect() const;
    QRect theme_button_rect() const;
    // The plot as it is drawn now (rendered if out of date).
    const QImage& image();
    // Pixel in the plot for a time and a frequency.
    QPoint plot_point(double t, double f_hz) const;
    double now() const;
    // For tests: how many columns the last redraw computed (the rest of
    // the history came from the cache).
    int last_computed_columns() const { return last_computed_columns_; }

    QSize sizeHint() const override;
    QSize minimumSizeHint() const override;

public slots:
    // Read what the ring has gained, and redraw.
    void pump();

protected:
    void paintEvent(QPaintEvent* event) override;
    void resizeEvent(QResizeEvent* event) override;
    void wheelEvent(QWheelEvent* event) override;
    void mousePressEvent(QMouseEvent* event) override;
    void mouseDoubleClickEvent(QMouseEvent* event) override;
    void mouseMoveEvent(QMouseEvent* event) override;
    void mouseReleaseEvent(QMouseEvent* event) override;
    void leaveEvent(QEvent* event) override;

private:
    void update_cursor(QPoint pos);
    void render();
    void invalidate();
    double freq_at_y(double y) const;
    double y_of_freq(double f) const;

    std::unique_ptr<SpectrumFeed> own_feed_;
    SpectrumFeed* feed_ = nullptr;
    dsp::TimeScale scale_;
    std::vector<Marker> markers_;
    std::vector<double> dial_marks_;
    std::function<double()> clock_;
    QTimer* timer_ = nullptr;
    QImage image_;
    bool dirty_ = true;
    bool inverted_ = true;
    double f_lo_ = F_LO;
    double f_hi_ = F_HI;
    double rendered_now_ = 0.0;
    double view_end_ = 0.0;   // 0: live
    enum class Drag { None, Time, Freq };
    Drag drag_ = Drag::None;
    double drag_t_ = 0.0;     // the time grabbed on the time axis
    double drag_f_ = 0.0;     // the frequency grabbed on the frequency axis
    // The history's block grid as last drawn (see render()).
    double x_of_time(double t) const;
    double time_at_x(double x) const;
    int render_lens_end_ = 0;
    long long render_k0_ = 0;
    double render_spp_ = 0.0;
    // What a cached history block's pixels depend on.
    struct CacheKey {
        int w = 0, h = 0;
        double f_lo = 0, f_hi = 0;
        bool inverted = false;
        double history_spp = 0, lens_spp = 0;
        int lens_px = 0;
        const SpectrumFeed* feed = nullptr;
        std::vector<Marker> markers;
        bool operator==(const CacheKey& o) const {
            if (markers.size() != o.markers.size()) return false;
            for (std::size_t i = 0; i < markers.size(); ++i) {
                if (markers[i].f_hz != o.markers[i].f_hz || markers[i].t0 != o.markers[i].t0 ||
                    markers[i].t1 != o.markers[i].t1) {
                    return false;
                }
            }
            return w == o.w && h == o.h && f_lo == o.f_lo && f_hi == o.f_hi &&
                   inverted == o.inverted && history_spp == o.history_spp &&
                   lens_spp == o.lens_spp && lens_px == o.lens_px && feed == o.feed;
        }
    };
    CacheKey cache_key_;
    std::unordered_map<long long, std::vector<QRgb>> history_cache_;
    int last_computed_columns_ = 0;
    double cache_newest_ = 0.0;
};

}  // namespace sstvae::gui

#endif

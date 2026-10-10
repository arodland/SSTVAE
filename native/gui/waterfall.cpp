#include "waterfall.hpp"

#include <QColor>
#include <QDateTime>
#include <QFontMetrics>
#include <QMouseEvent>
#include <QPaintEvent>
#include <QPainter>
#include <QPen>
#include <QResizeEvent>
#include <QSizePolicy>
#include <QTimer>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <utility>
#include <vector>

#include "config.hpp"
#include "dsp/spectrum.hpp"
#include "rx/ringbuffer.hpp"
#include "spectrum_feed.hpp"
#include "style.hpp"

namespace sstvae::gui {

namespace {

// The occupied band, marked so the operator can see whether the signal
// is sitting where the modem expects it.
constexpr double BAND_LO_HZ = config::CARRIER0 - config::RS;
constexpr double BAND_HI_HZ = config::CARRIER0 + config::NC * config::RS;

constexpr double DB_FLOOR = -95.0;
constexpr double DB_CEIL = -20.0;

using Rgb = std::array<std::uint8_t, 3>;

// 256-entry black -> blue -> green -> yellow -> white ramp.
const std::array<Rgb, 256>& colormap() {
    static const std::array<Rgb, 256> lut = [] {
        struct Stop {
            double at;
            Rgb color;
        };
        constexpr std::array<Stop, 5> stops{{{0.00, {0, 0, 0}},
                                             {0.25, {0, 0, 140}},
                                             {0.50, {0, 170, 90}},
                                             {0.75, {245, 235, 40}},
                                             {1.00, {255, 255, 255}}}};
        std::array<Rgb, 256> out{};
        for (int i = 0; i < 256; ++i) {
            const double x = i / 255.0;
            std::size_t k = 0;
            while (k + 2 < stops.size() && x > stops[k + 1].at) ++k;
            const Stop& lo = stops[k];
            const Stop& hi = stops[k + 1];
            const double t = (x - lo.at) / (hi.at - lo.at);
            for (int ch = 0; ch < 3; ++ch) {
                out[i][ch] = static_cast<std::uint8_t>(
                    lo.color[ch] + t * (hi.color[ch] - lo.color[ch]));
            }
        }
        return out;
    }();
    return lut;
}

}  // namespace

Waterfall::Waterfall(QWidget* parent, int fps) : QWidget(parent) {
    // Narrow minimum: the frequency axis is scaled to whatever width it
    // is given, and demanding all 384 bins as pixels would make this
    // strip the floor under the whole receive pane -- and, since the
    // two panes share a splitter whose minimum is the *sum* of theirs,
    // under the window.
    setMinimumWidth(160);
    setMinimumHeight(90);
    // A strip across the top of the receive pane rather than a column
    // down its side, so it takes the width it is given and does not
    // fight the picture below it for height.
    setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Preferred);

    own_feed_ = std::make_unique<SpectrumFeed>();
    feed_ = own_feed_.get();
    timer_ = new QTimer(this);
    connect(timer_, &QTimer::timeout, this, &Waterfall::tick);
    timer_->start(std::max(1, 1000 / std::max(1, fps)));
}

void Waterfall::set_feed(SpectrumFeed* feed) {
    if (feed == nullptr || feed == feed_) return;
    if (feed_ != own_feed_.get()) disconnect(feed_, nullptr, this, nullptr);
    feed_ = feed;
    timer_->stop();
    connect(feed_, &SpectrumFeed::pumped, this, &Waterfall::draw_new_rows);
    // Redraw from the shared history at once.
    const auto hold = feed_->lock();
    drawn_ = feed_->fast().count();
    image_ = QImage();
    ensure_image();
    update();
}

Waterfall::~Waterfall() = default;

// Wide and shallow: history depth follows the height, so a strip holds
// less of it than the old full-height column did -- about eight
// seconds at 20 fps, which is the span that matters for "is someone
// transmitting right now" and for setting soundcard gain. The splitter
// above it makes that the operator's call rather than this number's.
QSize Waterfall::sizeHint() const { return QSize(520, 160); }

void Waterfall::set_ring(std::shared_ptr<rx::RingBuffer> ring) {
    feed_->set_ring(std::move(ring));
}

void Waterfall::clear() {
    if (!image_.isNull()) image_.fill(Qt::black);
    peak_ = 0.0;
    clipping_ = false;
    update();
}

// The caller holds the feed's lock: this may redraw from its history.
void Waterfall::ensure_image() {
    // Device pixels, not logical ones.
    //
    // The rule this widget is built on is that the backing image is
    // exactly the size it is drawn at, so the painter never rescales
    // it -- and on a HiDPI screen "the size it is drawn at" is the
    // widget's size times the device pixel ratio. Sized in logical
    // pixels instead, the image is half resolution and Qt scales it up
    // on every repaint: blurry, slower, and the one-row scroll lands on
    // a half-pixel boundary so the rows shimmer as they move. That is
    // the same class of fault as the downscaling this widget was
    // written to avoid, arriving by a different route.
    const qreal dpr = devicePixelRatioF();
    const int w = std::max(1, static_cast<int>(std::lround(width() * dpr)));
    const int h = std::max(1, static_cast<int>(std::lround(height() * dpr)));
    if (image_.width() == w && image_.height() == h &&
        qFuzzyCompare(image_.devicePixelRatio(), dpr)) {
        return;
    }

    // Carry the history across a resize rather than blanking it: redrawn
    // from the stored rows, at the new width and as deep as the new
    // height, so dragging the strip taller shows older rows. A row the
    // store no longer has is copied from the old image, its columns
    // point-resampled.
    QImage grown(w, h, QImage::Format_RGB888);
    grown.setDevicePixelRatio(dpr);
    grown.fill(Qt::black);
    const dsp::WaterfallHistory& history = feed_->fast();
    const int old_w = image_.isNull() ? 0 : image_.width();
    for (int y = 0; y < h; ++y) {
        uchar* dst = grown.scanLine(y);
        const std::uint64_t seq = drawn_ - 1 - static_cast<std::uint64_t>(y);
        if (static_cast<std::uint64_t>(y) < drawn_ && history.has(seq)) {
            paint_row(dst, history.row(seq), w);
        } else if (y < (image_.isNull() ? 0 : image_.height())) {
            const uchar* src = image_.constScanLine(y);
            for (int x = 0; x < w; ++x) {
                const int sx = std::min(old_w - 1, x * old_w / w);
                std::copy_n(src + sx * 3, 3, dst + x * 3);
            }
        }
    }
    image_ = std::move(grown);
}

void Waterfall::paint_row(uchar* dst, const std::vector<double>& db, int width) const {
    const std::vector<double> reduced = dsp::reduce_to_width(db, width);
    if (static_cast<int>(reduced.size()) < width) return;
    const std::array<Rgb, 256>& lut = colormap();
    for (int x = 0; x < width; ++x) {
        const double norm =
            std::clamp((reduced[x] - DB_FLOOR) / (DB_CEIL - DB_FLOOR), 0.0, 1.0);
        const Rgb& color = lut[static_cast<std::size_t>(norm * 255.0)];
        std::copy_n(color.data(), 3, dst + x * 3);
    }
}

void Waterfall::tick() {
    if (feed_ == own_feed_.get()) feed_->pump();
    draw_new_rows();
}

void Waterfall::draw_new_rows() {
    const auto hold = feed_->lock();
    const dsp::WaterfallHistory& history = feed_->fast();
    const std::uint64_t count = history.count();
    if (count == drawn_) return;
    peak_ = feed_->peak();
    // The audio layer hands back floats; anything at or over unity has
    // already been clipped somewhere upstream in the capture chain, so
    // this reports the soundcard's problem rather than ours.
    clipping_ = peak_ >= 0.99;
    if (clipping_) clip_latched_ = true;

    ensure_image();
    const int h = image_.height();
    const std::uint64_t from =
        std::max<std::uint64_t>({drawn_, history.first(),
                                 count > static_cast<std::uint64_t>(h) ? count - h : 0});
    // One row = one pixel, so each is a 1 px scroll. Bottom-up, in
    // place: row h-1 is overwritten first, so no row is read after it
    // has been clobbered.
    const std::size_t stride = static_cast<std::size_t>(image_.bytesPerLine());
    for (std::uint64_t seq = from; seq < count; ++seq) {
        for (int y = h - 1; y > 0; --y) {
            std::copy_n(image_.constScanLine(y - 1), stride, image_.scanLine(y));
        }
        paint_row(image_.scanLine(0), history.row(seq), image_.width());
    }
    drawn_ = count;
    newest_row_time_ = static_cast<double>(QDateTime::currentMSecsSinceEpoch()) / 1000.0;
    update();
}

void Waterfall::resizeEvent(QResizeEvent* event) {
    QWidget::resizeEvent(event);
    const auto hold = feed_->lock();
    ensure_image();
}

void Waterfall::paintEvent(QPaintEvent* event) {
    Q_UNUSED(event);
    {
        const auto hold = feed_->lock();
        ensure_image();
    }
    QPainter painter(this);
    // 1:1 by construction, so this is a blit and not a rescale: the
    // image carries the same device pixel ratio as the painter, so
    // drawing it at the origin puts one image pixel on one device pixel.
    painter.drawImage(QPointF(0, 0), image_);
    draw_band_markers(painter);
    draw_dial_marks(painter);
    draw_level_meter(painter);
    draw_disabled_scrim(painter);
}

void Waterfall::set_dial_marks(std::vector<double> times) {
    dial_marks_ = std::move(times);
    update();
}

void Waterfall::draw_dial_marks(QPainter& painter) {
    if (dial_marks_.empty() || newest_row_time_ <= 0.0) return;
    const double rows_per_px = std::max(1e-9, image_.devicePixelRatio());
    const double fps = std::max(1, feed_->fps());
    const QColor green(0x3c, 0xe0, 0x6e);
    for (const double t : dial_marks_) {
        const double age = newest_row_time_ - t;
        if (age < 0) continue;
        const int y = static_cast<int>(std::lround(age * fps / rows_per_px));
        if (y >= height()) continue;
        painter.setPen(QPen(green, 2));
        painter.drawLine(0, y, width(), y);
        const QString label = dial_mark_label(age);
        painter.setPen(QColor(0, 0, 0, 180));
        painter.drawText(5, y + 13, label);
        painter.setPen(green);
        painter.drawText(4, y + 12, label);
    }
}

QString dial_mark_label(double age_s) {
    const long long m = static_cast<long long>(std::floor(age_s / 60.0));
    if (age_s < 60.0) return QStringLiteral("<-%1s").arg(static_cast<long long>(age_s));
    if (m < 60) return QStringLiteral("<-%1m").arg(m);
    return QStringLiteral("<-%1h%2m").arg(m / 60).arg(m % 60, 2, 10, QLatin1Char('0'));
}

void Waterfall::draw_disabled_scrim(QPainter& painter) {
    // `setEnabled(false)` alone does nothing to a custom-painted widget:
    // Qt greys the *standard* controls it draws itself, and a QLabel
    // dims its pixmap, but a paintEvent that blits an image and strokes
    // hard-coded colours renders pixel-identically either way. This
    // widget is disabled exactly once -- while transmitting -- and the
    // whole point of showing it then is that a paused receiver must not
    // look like a wedged one, which is the state a frozen spectrum with
    // no scrim is indistinguishable from.
    if (isEnabled()) return;
    painter.fillRect(rect(), QColor(0, 0, 0, 150));
    painter.setPen(QColor(220, 220, 220, 220));
    const QString label = tr("Paused — transmitting");
    const QRect box = rect();
    painter.drawText(box, Qt::AlignCenter, label);
}

void Waterfall::draw_band_markers(QPainter& painter) {
    const int w = width();
    const int h = height();
    const double scale = w / dsp::WATERFALL_DISPLAY_HZ;

    QPen pen(QColor(255, 255, 255, 110));
    pen.setStyle(Qt::DashLine);
    painter.setPen(pen);
    for (const double hz : {BAND_LO_HZ, BAND_HI_HZ}) {
        const int x = static_cast<int>(hz * scale);
        painter.drawLine(x, 0, x, h);
    }

    // The strip is user-resizable, so the caption has to earn its
    // place: drop it rather than let it run off the edge or overprint
    // the spectrum.
    const QString label = tr("SSTVAE %1-%2 Hz")
                              .arg(BAND_LO_HZ, 0, 'f', 0)
                              .arg(BAND_HI_HZ, 0, 'f', 0);
    const int label_x = static_cast<int>(BAND_LO_HZ * scale) + 4;
    // The latched CLIP marker shares this baseline at the right edge;
    // when it is up, the caption yields early rather than overprinting
    // it -- at 195-240 px both fit the old guard and neither is legible.
    const int reserved =
        clip_latched_
            ? painter.fontMetrics().horizontalAdvance(tr("CLIP")) + 20
            : 12;
    if (label_x + painter.fontMetrics().horizontalAdvance(label) < w - reserved) {
        painter.setPen(QColor(0, 0, 0, 160));
        painter.drawText(label_x + 1, 15, label);  // shadow, for contrast
        painter.setPen(QColor(255, 255, 255, 220));
        painter.drawText(label_x, 14, label);
    }

    // A 1 kHz grid, so the operator can read where a signal sits. Drawn
    // with a shadow: the ticks sit over whatever the spectrum happens to
    // be doing at the bottom of the pane, which is often bright.
    for (int hz = 1000; hz < static_cast<int>(dsp::WATERFALL_DISPLAY_HZ);
         hz += 1000) {
        const int x = static_cast<int>(hz * scale);
        const QString tick = QString::number(hz / 1000) + QStringLiteral("k");
        painter.setPen(QColor(0, 0, 0, 150));
        painter.drawLine(x + 1, h - 8, x + 1, h);
        painter.drawText(x + 4, h - 3, tick);
        painter.setPen(QColor(255, 255, 255, 190));
        painter.drawLine(x, h - 8, x, h);
        painter.drawText(x + 3, h - 4, tick);
    }
}

void Waterfall::draw_level_meter(QPainter& painter) {
    // A thin bar down the right edge: enough to set soundcard gain,
    // which is the one audio adjustment that actually matters.
    const int w = width();
    const int h = height();
    constexpr int bar_w = 8;
    const int x0 = w - bar_w - 2;
    painter.fillRect(x0, 2, bar_w, h - 4, QColor(0, 0, 0, 140));

    // dBFS, so the useful range is not crushed into the top of a linear
    // bar.
    const double db = 20.0 * std::log10(std::max(peak_, 1e-6));
    const double frac = std::clamp((db + 60.0) / 60.0, 0.0, 1.0);
    const int filled = static_cast<int>((h - 4) * frac);

    // From the shared table: this bar, the CLIP marker and the PTT lamp
    // are one idea in three places, and were three unrelated literals.
    QColor color = style::color::ok();
    if (clipping_) {
        color = style::color::danger_bright();
    } else if (frac > 0.85) {
        color = style::color::caution();
    }
    painter.fillRect(x0, h - 2 - filled, bar_w, filled, color);

    if (clip_latched_) {
        // Latched, not momentary: stays until the meter is clicked, so
        // clipping that happened while nobody was looking still gets
        // reported. Same shadowed-text idiom as the band caption.
        const QString label = tr("CLIP");
        const int text_x =
            x0 - painter.fontMetrics().horizontalAdvance(label) - 4;
        painter.setPen(QColor(0, 0, 0, 160));
        painter.drawText(text_x + 1, 15, label);
        QColor clip = style::color::danger_bright();
        clip.setAlpha(230);
        painter.setPen(clip);
        painter.drawText(text_x, 14, label);
    }
}

void Waterfall::clear_clip() {
    clip_latched_ = false;
    update();
}

void Waterfall::mousePressEvent(QMouseEvent* event) {
    // A click on (or near) the meter clears the latch. The whole right
    // edge counts -- an 8 px bar is not a precision target.
    if (clip_latched_ && event->position().x() >= width() - 30) {
        clear_clip();
        return;
    }
    QWidget::mousePressEvent(event);
}

}  // namespace sstvae::gui

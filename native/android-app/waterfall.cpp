#include "waterfall.hpp"

#include <QPainter>

#include <algorithm>
#include <cmath>
#include <vector>

#include "config.hpp"
#include "dsp/spectrum.hpp"
#include "session.hpp"

namespace {

using sstvae::androidapp::Session;

// dB range mapped onto the colour ramp. Fixed rather than
// auto-scaling: an auto-scaled waterfall renders an empty band and a
// busy one identically, which defeats the one job this display has.
constexpr double kFloorDb = -90.0;
constexpr double kCeilDb = -20.0;

QRgb ramp(double db) {
    const double t = std::clamp((db - kFloorDb) / (kCeilDb - kFloorDb), 0.0, 1.0);
    // Black -> blue -> green -> yellow -> white. Monotone in luminance,
    // so a stronger signal is always a brighter pixel even in grey.
    const double r = std::clamp(2.2 * t - 0.9, 0.0, 1.0);
    const double g = std::clamp(1.8 * t - 0.35, 0.0, 1.0);
    const double b = t < 0.5 ? std::clamp(2.0 * t, 0.0, 1.0)
                             : std::clamp(2.0 * t - 1.0, 0.0, 1.0);
    return qRgb(static_cast<int>(255 * r), static_cast<int>(255 * g),
                static_cast<int>(255 * b));
}

// x for a frequency, given the display's span.
double x_for_hz(double hz, int width) {
    return width * hz / sstvae::dsp::WATERFALL_DISPLAY_HZ;
}

}  // namespace

Waterfall::Waterfall(QQuickItem* parent) : QQuickPaintedItem(parent) {
    // ~10 fps. The desktop runs ~20; a phone gains nothing from the
    // extra frames and pays for them in battery, and this display's
    // information rate is set by the FFT, not the repaint.
    timer_.setInterval(100);
    connect(&timer_, &QTimer::timeout, this, &Waterfall::tick);
    timer_.start();
}

void Waterfall::ensure_history(int w, int h) {
    if (w <= 0 || h <= 0) return;
    if (history_.width() == w && history_.height() == h) return;

    // **Keep the history across a resize.** Blanking it is easy, looks
    // deliberate, and throws away the minute of band activity the
    // operator was using to tune.
    QImage grown(w, h, QImage::Format_RGB32);
    grown.fill(Qt::black);
    if (!history_.isNull()) {
        QPainter p(&grown);
        p.drawImage(QRect(0, 0, w, h), history_,
                    QRect(0, 0, history_.width(), history_.height()));
    }
    history_ = grown;
}

void Waterfall::tick() {
    const int w = static_cast<int>(width());
    const int h = static_cast<int>(height());
    if (w <= 0 || h <= 0) return;
    ensure_history(w, h);

    const std::vector<double> block =
        Session::instance().audio_tail(sstvae::dsp::WATERFALL_NFFT);
    if (static_cast<int>(block.size()) < sstvae::dsp::WATERFALL_NFFT) {
        update();
        return;
    }

    const std::vector<double> db =
        sstvae::dsp::spectrum_db(block, sstvae::dsp::WATERFALL_BINS);
    const std::vector<double> row = sstvae::dsp::reduce_to_width(db, w);

    // Scroll **down**: history moves away from the newest row, which is
    // painted at the top. An in-place row copy is easy to write in the
    // other direction and the result still looks like a moving display,
    // which is exactly why it is worth being deliberate about.
    std::memmove(history_.scanLine(1), history_.scanLine(0),
                 static_cast<std::size_t>(history_.bytesPerLine()) * (h - 1));
    auto* top = reinterpret_cast<QRgb*>(history_.scanLine(0));
    for (int x = 0; x < w && x < static_cast<int>(row.size()); ++x) {
        top[x] = ramp(row[x]);
    }
    update();
}

void Waterfall::geometryChange(const QRectF& newGeometry, const QRectF& oldGeometry) {
    QQuickPaintedItem::geometryChange(newGeometry, oldGeometry);
    ensure_history(static_cast<int>(newGeometry.width()),
                   static_cast<int>(newGeometry.height()));
}

void Waterfall::paint(QPainter* painter) {
    const int w = static_cast<int>(width());
    const int h = static_cast<int>(height());
    if (w <= 0 || h <= 0) return;

    if (history_.isNull()) {
        painter->fillRect(0, 0, w, h, Qt::black);
    } else {
        painter->drawImage(0, 0, history_);
    }

    // The band edges, which are what makes this a tuning instrument
    // rather than a pretty picture: with no frequency readout, "is the
    // signal between the lines" is the entire tuning interface.
    const double lo = sstvae::config::CARRIER0;
    const double hi = sstvae::config::CARRIER0 +
                      (sstvae::config::NC - 1) * sstvae::config::RS;
    QPen pen(QColor(255, 255, 255, 110));
    pen.setStyle(Qt::DashLine);
    painter->setPen(pen);
    painter->drawLine(QPointF(x_for_hz(lo, w), 0), QPointF(x_for_hz(lo, w), h));
    painter->drawLine(QPointF(x_for_hz(hi, w), 0), QPointF(x_for_hz(hi, w), h));
}

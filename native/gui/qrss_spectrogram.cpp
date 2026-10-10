#include "qrss_spectrogram.hpp"

#include <QDateTime>
#include <QMouseEvent>
#include <QPainter>
#include <QLinearGradient>
#include <QPainterPath>
#include <QTimeZone>
#include <QTimer>
#include <QWheelEvent>

#include <algorithm>
#include <cmath>
#include <limits>
#include <iterator>
#include <thread>
#include <utility>
#include <vector>

#include "rx/ringbuffer.hpp"
#include "spectrum_feed.hpp"

namespace sstvae::gui {

namespace {

constexpr int FREQ_AXIS_W = 52;
constexpr int TIME_AXIS_H = 22;
constexpr int BUTTON = 24;
constexpr double WHEEL_STEP = 1.25;
constexpr double MIN_SPAN_HZ = 20.0;

QString clock_label(double t, bool seconds) {
    const QDateTime u = QDateTime::fromSecsSinceEpoch(static_cast<qint64>(std::floor(t)),
                                                      QTimeZone::utc());
    return u.toString(seconds ? QStringLiteral("HH:mm:ss") : QStringLiteral("HH:mm'Z'"));
}

// The colour for strength v in [0, 1].
QRgb shade(double v, bool inverted, bool marked) {
    const auto c = [](double x) { return std::clamp(static_cast<int>(std::lround(x)), 0, 255); };
    // Marked: the same ramp in red, from the same background, so only
    // the signal's own energy is red and not the band around it.
    if (inverted) {
        if (marked) return qRgb(c(255 - 115 * v), c(255 * (1 - v)), c(255 * (1 - v)));
        return qRgb(c(255 * (1 - v)), c(255 * (1 - v)), c(255 * (1 - v)));
    }
    if (marked) return qRgb(c(255 * std::pow(v, 0.7)), c(40 * v), c(40 * v));
    return qRgb(c(255 * v), c(255 * v), c(255 * v));
}

}  // namespace

QrssSpectrogram::QrssSpectrogram(QWidget* parent) : QWidget(parent) {
    setObjectName(QStringLiteral("qrss_spectrogram"));
    setMouseTracking(true);
    clock_ = [] { return static_cast<double>(QDateTime::currentMSecsSinceEpoch()) / 1000.0; };
    own_feed_ = std::make_unique<SpectrumFeed>();
    feed_ = own_feed_.get();
    timer_ = new QTimer(this);
    timer_->setInterval(1000);
    connect(timer_, &QTimer::timeout, this, &QrssSpectrogram::pump);
    timer_->start();
    setToolTip(tr("QRSS spectrogram: newest at the left. The Time Lens there magnifies the "
                  "newest audio to about a second per pixel and eases smoothly out to "
                  "fifteen seconds per pixel for the history.\n"
                  "Mouse wheel over the time axis: change the scale under the pointer.\n"
                  "Mouse wheel over the frequency axis: zoom; double-click it to see the "
                  "whole range.\nThe sun/moon flips the colours. Red: a signal the QRSS "
                  "listener has synced on."));
}

QrssSpectrogram::~QrssSpectrogram() = default;

void QrssSpectrogram::set_clock(std::function<double()> clock) {
    clock_ = std::move(clock);
    own_feed_->set_clock(clock_);
    invalidate();
}

double QrssSpectrogram::now() const { return clock_(); }

void QrssSpectrogram::set_ring(std::shared_ptr<rx::RingBuffer> ring) {
    feed_->set_ring(std::move(ring));
}

void QrssSpectrogram::set_feed(SpectrumFeed* feed) {
    if (feed == nullptr) return;
    feed_ = feed;
    invalidate();
}

void QrssSpectrogram::set_active(bool on) {
    if (on) {
        if (!timer_->isActive()) timer_->start();
        invalidate();
    } else {
        timer_->stop();
    }
}

const dsp::SlowSpectrogram& QrssSpectrogram::data() const { return feed_->slow(); }

void QrssSpectrogram::push_audio(const std::vector<double>& samples, double end_time) {
    feed_->push_audio(samples, end_time);
    invalidate();
}

void QrssSpectrogram::pump() {
    // A shared feed pumps itself; this one only redraws.
    if (feed_ == own_feed_.get()) feed_->pump();
    // The picture moves with the clock even with no audio.
    invalidate();
}

void QrssSpectrogram::set_markers(std::vector<Marker> markers) {
    markers_ = std::move(markers);
    invalidate();
}

void QrssSpectrogram::set_inverted(bool on) {
    if (on == inverted_) return;
    inverted_ = on;
    invalidate();
}

void QrssSpectrogram::set_frequency_range(double lo, double hi) {
    lo = std::clamp(lo, 0.0, dsp::SlowSpectrogram::MAX_HZ - MIN_SPAN_HZ);
    hi = std::clamp(hi, lo + MIN_SPAN_HZ, dsp::SlowSpectrogram::MAX_HZ);
    f_lo_ = lo;
    f_hi_ = hi;
    invalidate();
}

void QrssSpectrogram::invalidate() {
    dirty_ = true;
    update();
}

QSize QrssSpectrogram::sizeHint() const { return {760, 480}; }
QSize QrssSpectrogram::minimumSizeHint() const { return {240, 160}; }

QRect QrssSpectrogram::plot_rect() const {
    return {FREQ_AXIS_W, 0, std::max(1, width() - FREQ_AXIS_W),
            std::max(1, height() - TIME_AXIS_H)};
}
QRect QrssSpectrogram::time_axis_rect() const {
    return {FREQ_AXIS_W, height() - TIME_AXIS_H, std::max(1, width() - FREQ_AXIS_W),
            TIME_AXIS_H};
}
QRect QrssSpectrogram::freq_axis_rect() const {
    return {0, 0, FREQ_AXIS_W, std::max(1, height() - TIME_AXIS_H)};
}
QRect QrssSpectrogram::theme_button_rect() const {
    const QRect p = plot_rect();
    return {p.right() - BUTTON - 5, p.bottom() - BUTTON - 5, BUTTON, BUTTON};
}

double QrssSpectrogram::freq_at_y(double y) const {
    const double h = plot_rect().height();
    return f_hi_ - (f_hi_ - f_lo_) * y / h;
}
double QrssSpectrogram::y_of_freq(double f) const {
    const double h = plot_rect().height();
    return (f_hi_ - f) / (f_hi_ - f_lo_) * h;
}

QPoint QrssSpectrogram::plot_point(double t, double f_hz) const {
    const double x = x_of_time(t);
    return {static_cast<int>(std::floor(x)), static_cast<int>(std::floor(y_of_freq(f_hz)))};
}

const QImage& QrssSpectrogram::image() {
    if (dirty_ || image_.size() != plot_rect().size()) render();
    return image_;
}

void QrssSpectrogram::render() {
    dirty_ = false;
    // The feed's worker may be pumping; it waits while this reads.
    const auto hold = feed_->lock();
    const QRect p = plot_rect();
    const int w = p.width();
    const int h = p.height();
    image_ = QImage(w, h, QImage::Format_RGB32);
    const double t_now = now();
    rendered_now_ = t_now;
    const QRgb gap = inverted_ ? qRgb(232, 232, 232) : qRgb(28, 28, 32);
    image_.fill(gap);

    // Which bins each row averages over.
    std::vector<int> b0(static_cast<std::size_t>(h));
    std::vector<int> b1(static_cast<std::size_t>(h));
    std::vector<double> row_f(static_cast<std::size_t>(h));
    for (int y = 0; y < h; ++y) {
        const double fa = freq_at_y(y + 1);
        const double fb = freq_at_y(y);
        int lo = static_cast<int>(std::floor(fa / dsp::SlowSpectrogram::BIN_HZ));
        int hi = static_cast<int>(std::ceil(fb / dsp::SlowSpectrogram::BIN_HZ));
        lo = std::clamp(lo, 0, dsp::SlowSpectrogram::BINS - 1);
        hi = std::clamp(std::max(hi, lo + 1), lo + 1, dsp::SlowSpectrogram::BINS);
        b0[static_cast<std::size_t>(y)] = lo;
        b1[static_cast<std::size_t>(y)] = hi;
        row_f[static_cast<std::size_t>(y)] = 0.5 * (fa + fb);
    }

    // **Only the Time Lens is recomputed every second** (Jeff,
    // 2026-10-10). The history to its right is drawn on a grid of
    // history_spp-second blocks anchored to the clock, not to "now": a
    // block keeps its pixels until the grid moves on by one, when the
    // whole history shifts a pixel and only the newest block is new. So
    // a finished block is computed once and kept (`history_cache_`) until
    // something that changes how it looks changes.
    const int lens_end = std::min(w, scale_.lens_px());
    const double spp = scale_.history_spp();
    const double lens_age = scale_.age_at(lens_end);
    const long long k0 = static_cast<long long>(std::floor((t_now - lens_age) / spp));
    render_lens_end_ = lens_end;
    render_k0_ = k0;
    render_spp_ = spp;
    CacheKey key{w, h, f_lo_, f_hi_, inverted_, spp, scale_.lens_spp(), scale_.lens_px(),
                 feed_, markers_};
    // A block is final once the store holds a column past its end.
    const double newest = data().newest();
    // A store that went back in time was cleared: nothing kept holds.
    if (!(key == cache_key_) || newest < cache_newest_) {
        history_cache_.clear();
        cache_key_ = std::move(key);
    }
    cache_newest_ = newest;

    struct Job {
        int x;
        double t_lo, t_hi;
        long long block;   // or LLONG_MIN in the lens
        std::vector<QRgb> px;
        bool drawn = false;
    };
    std::vector<Job> jobs;
    std::vector<int> cached_x;
    std::vector<long long> cached_k;
    for (int x = 0; x < lens_end; ++x) {
        jobs.push_back({x, t_now - scale_.age_at(x + 1), t_now - scale_.age_at(x),
                        std::numeric_limits<long long>::min(), {}});
    }
    for (int x = lens_end; x < w; ++x) {
        const long long k = k0 - (x - lens_end);
        if (history_cache_.count(k) != 0) {
            cached_x.push_back(x);
            cached_k.push_back(k);
        } else {
            jobs.push_back({x, static_cast<double>(k) * spp, static_cast<double>(k + 1) * spp, k,
                            {}});
        }
    }
    last_computed_columns_ = static_cast<int>(jobs.size());

    // The columns to compute are independent, so they are spread over
    // several cores: the store is only read here (the feed's worker
    // waits on the lock), and each job has its own output.
    const auto compute = [&](std::size_t j_begin, std::size_t j_end) {
        std::vector<float> rows(static_cast<std::size_t>(h));
        std::vector<double> cumsum(dsp::SlowSpectrogram::BINS + 1);
        std::vector<const Marker*> here;
        for (std::size_t j = j_begin; j < j_end; ++j) {
            Job& job = jobs[j];
            const double t_lo = job.t_lo;
            const double t_hi = job.t_hi;
            int n_cols = 0;
            const std::vector<float> col = data().mean_between(t_lo, t_hi, &n_cols);
            if (col.empty()) continue;
            cumsum[0] = 0.0;
            for (int k = 0; k < dsp::SlowSpectrogram::BINS; ++k) cumsum[k + 1] = cumsum[k] + col[k];
            for (int y = 0; y < h; ++y) {
                const int lo = b0[static_cast<std::size_t>(y)];
                const int hi = b1[static_cast<std::size_t>(y)];
                rows[static_cast<std::size_t>(y)] =
                    static_cast<float>((cumsum[hi] - cumsum[lo]) / (hi - lo));
            }
            // How far above the noise each row is, in units of the noise's
            // own roughness: a mean of n power values of noise varies by
            // about 1/sqrt(n) of the floor, so the same weak signal stands
            // out as well at 15 s a pixel as at 1, and better.
            const double floor_p = std::max(1e-30, static_cast<double>(dsp::median_of(rows)));
            // Which markers this column's time falls in.
            here.clear();
            for (const Marker& m : markers_) {
                if (m.t0 < t_hi && m.t1 > t_lo) here.push_back(&m);
            }
            job.px.resize(static_cast<std::size_t>(h));
            for (int y = 0; y < h; ++y) {
                const double n = static_cast<double>(n_cols) *
                                 (b1[static_cast<std::size_t>(y)] - b0[static_cast<std::size_t>(y)]);
                const double z = (rows[static_cast<std::size_t>(y)] / floor_p - 1.0) * std::sqrt(n);
                const double v = std::clamp((z - Z_FLOOR) / Z_RANGE, 0.0, 1.0);
                bool marked = false;
                for (const Marker* m : here) {
                    if (std::abs(row_f[static_cast<std::size_t>(y)] - m->f_hz) <= MARK_HALF_HZ) {
                        marked = true;
                        break;
                    }
                }
                job.px[static_cast<std::size_t>(y)] = shade(v, inverted_, marked);
            }
            job.drawn = true;
        }
    };
    const std::size_t n_jobs = jobs.size();
    const std::size_t n_threads = std::min<std::size_t>(
        std::clamp(static_cast<int>(std::thread::hardware_concurrency()), 1, 4),
        std::max<std::size_t>(1, n_jobs / 16));
    const std::size_t per = (n_jobs + n_threads - 1) / n_threads;
    std::vector<std::thread> threads;
    for (std::size_t i = 1; i < n_threads; ++i) {
        const std::size_t lo = std::min(n_jobs, i * per);
        const std::size_t hi = std::min(n_jobs, lo + per);
        if (lo < hi) threads.emplace_back(compute, lo, hi);
    }
    compute(0, std::min(n_jobs, per));
    for (std::thread& t : threads) t.join();

    uchar* const bits = image_.bits();
    const qsizetype stride = image_.bytesPerLine();
    const auto put = [&](int x, const std::vector<QRgb>& px) {
        for (int y = 0; y < h; ++y) {
            reinterpret_cast<QRgb*>(bits + y * stride)[x] = px[static_cast<std::size_t>(y)];
        }
    };
    for (Job& job : jobs) {
        if (job.drawn) put(job.x, job.px);
        // Keep a finished block, drawn or empty (an empty one is a gap).
        if (job.block != std::numeric_limits<long long>::min() && job.t_hi <= newest) {
            history_cache_[job.block] = job.drawn ? std::move(job.px) : std::vector<QRgb>{};
        }
    }
    for (std::size_t i = 0; i < cached_x.size(); ++i) {
        const std::vector<QRgb>& px = history_cache_[cached_k[i]];
        if (!px.empty()) put(cached_x[i], px);
    }
    // Forget blocks that have scrolled off the right.
    const long long oldest = k0 - (w - lens_end);
    for (auto it = history_cache_.begin(); it != history_cache_.end();) {
        it = it->first < oldest ? history_cache_.erase(it) : std::next(it);
    }
}

double QrssSpectrogram::x_of_time(double t) const {
    const double t_now = rendered_now_ > 0 ? rendered_now_ : now();
    if (render_spp_ > 0 && t_now - t > scale_.age_at(render_lens_end_)) {
        return render_lens_end_ + static_cast<double>(render_k0_ + 1) - t / render_spp_;
    }
    return scale_.x_of(t_now - t);
}

double QrssSpectrogram::time_at_x(double x) const {
    if (render_spp_ > 0 && x >= render_lens_end_) {
        return (static_cast<double>(render_k0_ + 1) - (x - render_lens_end_)) * render_spp_;
    }
    return rendered_now_ - scale_.age_at(x);
}

void QrssSpectrogram::resizeEvent(QResizeEvent* event) {
    QWidget::resizeEvent(event);
    invalidate();
}

void QrssSpectrogram::paintEvent(QPaintEvent*) {
    QPainter painter(this);
    const QPalette& pal = palette();
    painter.fillRect(rect(), pal.color(QPalette::Window));
    const QRect p = plot_rect();
    painter.drawImage(p.topLeft(), image());
    const QColor ink = inverted_ ? QColor(40, 40, 40) : QColor(220, 220, 220);
    const QColor accent = pal.color(QPalette::Highlight);

    // The Time Lens: no edge to draw, since it has none, so a wash
    // across the top fading out with the magnification says where it is.
    {
        const int lens_w = std::min(scale_.lens_px(), p.width());
        QLinearGradient wash(p.left(), 0, p.left() + lens_w, 0);
        QColor strong = accent;
        strong.setAlpha(90);
        QColor none = accent;
        none.setAlpha(0);
        wash.setColorAt(0.0, strong);
        wash.setColorAt(1.0, none);
        painter.fillRect(QRect(p.left(), p.top(), lens_w, 14), wash);
        QFont small = font();
        small.setPointSizeF(std::max(6.5, small.pointSizeF() * 0.8));
        painter.setFont(small);
        painter.setPen(ink);
        painter.drawText(QRect(p.left() + 4, p.top(), lens_w, 14), Qt::AlignLeft | Qt::AlignVCenter,
                         tr("Time Lens  %1 s/px").arg(scale_.lens_spp(), 0, 'g', 3));
        if (lens_w < p.width()) {
            painter.drawText(QRect(p.left() + lens_w, p.top(), 160, 14),
                             Qt::AlignLeft | Qt::AlignVCenter,
                             tr("%1 s/px").arg(scale_.history_spp(), 0, 'g', 3));
        }
    }
    // Now.
    painter.setPen(QPen(accent, 2));
    painter.drawLine(p.left() + 1, p.top(), p.left() + 1, p.bottom());

    QFont axis_font = font();
    axis_font.setPointSizeF(std::max(6.5, axis_font.pointSizeF() * 0.8));
    painter.setFont(axis_font);
    const QColor axis_ink = pal.color(QPalette::WindowText);
    const QColor grid = pal.color(QPalette::Mid);

    // Time ticks: as dense as the local scale allows, in UTC.
    {
        const QRect a = time_axis_rect();
        static constexpr double STEPS[] = {5,   10,  15,   30,   60,   120,  300,  600,
                                           900, 1800, 3600, 7200, 10800, 21600};
        int last_x = -1000;
        double prev_t = rendered_now_;
        for (int x = 1; x < p.width(); ++x) {
            const double spp = scale_.spp_at(x);
            double step = STEPS[std::size(STEPS) - 1];
            for (double s : STEPS) {
                if (s / spp >= 70.0) {
                    step = s;
                    break;
                }
            }
            const double t = time_at_x(x);
            if (std::floor(t / step) != std::floor(prev_t / step) && x - last_x >= 60) {
                const double tick = std::floor(prev_t / step) * step;
                const int px = p.left() + x;
                painter.setPen(grid);
                painter.drawLine(px, a.top(), px, a.top() + 4);
                painter.setPen(axis_ink);
                painter.drawText(QRect(px - 40, a.top() + 3, 80, a.height() - 3),
                                 Qt::AlignHCenter | Qt::AlignTop, clock_label(tick, step < 60));
                last_x = x;
            }
            prev_t = t;
        }
    }
    // Frequency ticks.
    {
        const QRect a = freq_axis_rect();
        static constexpr double STEPS[] = {1, 2, 5, 10, 20, 50, 100, 200, 500, 1000};
        const double hz_per_px = (f_hi_ - f_lo_) / std::max(1, p.height());
        double step = 1000;
        for (double s : STEPS) {
            if (s / hz_per_px >= 28.0) {
                step = s;
                break;
            }
        }
        for (double f = std::ceil(f_lo_ / step) * step; f <= f_hi_; f += step) {
            const int y = p.top() + static_cast<int>(std::lround(y_of_freq(f)));
            painter.setPen(grid);
            painter.drawLine(a.right() - 4, y, a.right(), y);
            painter.setPen(axis_ink);
            painter.drawText(QRect(0, y - 8, a.width() - 6, 16), Qt::AlignRight | Qt::AlignVCenter,
                             QString::number(f, 'f', step < 1 ? 1 : 0));
        }
    }

    // The sun/moon: a moon on the light scheme (go dark), a sun on the
    // dark one (go light).
    {
        const QRectF b = QRectF(theme_button_rect()).adjusted(0.5, 0.5, -0.5, -0.5);
        painter.setRenderHint(QPainter::Antialiasing, true);
        painter.setPen(QPen(ink, 1));
        painter.setBrush(inverted_ ? QColor(255, 255, 255, 200) : QColor(0, 0, 0, 160));
        painter.drawRoundedRect(b, 5, 5);
        const QPointF c = b.center();
        if (inverted_) {
            QPainterPath moon;
            moon.addEllipse(c, 7, 7);
            QPainterPath bite;
            bite.addEllipse(c + QPointF(4, -3), 6.5, 6.5);
            painter.setPen(Qt::NoPen);
            painter.setBrush(QColor(50, 50, 60));
            painter.drawPath(moon.subtracted(bite));
        } else {
            painter.setPen(QPen(QColor(255, 200, 40), 1.6));
            painter.setBrush(QColor(255, 200, 40));
            painter.drawEllipse(c, 4, 4);
            for (int i = 0; i < 8; ++i) {
                const double a = i * 3.14159265358979 / 4;
                painter.drawLine(c + QPointF(6 * std::cos(a), 6 * std::sin(a)),
                                 c + QPointF(9 * std::cos(a), 9 * std::sin(a)));
            }
        }
    }
}

void QrssSpectrogram::wheelEvent(QWheelEvent* event) {
    const QPoint pos = event->position().toPoint();
    const double steps = event->angleDelta().y() / 120.0;
    if (steps == 0) return;
    const double factor = std::pow(WHEEL_STEP, -steps);   // wheel up: finer
    if (time_axis_rect().contains(pos)) {
        const int x = pos.x() - plot_rect().left();
        if (x < scale_.lens_px() / 2) {
            scale_.set_lens_spp(scale_.lens_spp() * factor);
        } else {
            scale_.set_history_spp(scale_.history_spp() * factor);
        }
        invalidate();
        event->accept();
        return;
    }
    if (freq_axis_rect().contains(pos)) {
        const double f = freq_at_y(pos.y() - plot_rect().top());
        const double lo = f - (f - f_lo_) * factor;
        const double hi = f + (f_hi_ - f) * factor;
        set_frequency_range(lo, hi);
        event->accept();
        return;
    }
    QWidget::wheelEvent(event);
}

void QrssSpectrogram::mousePressEvent(QMouseEvent* event) {
    if (event->button() == Qt::LeftButton &&
        theme_button_rect().contains(event->position().toPoint())) {
        set_inverted(!inverted_);
        event->accept();
        return;
    }
    QWidget::mousePressEvent(event);
}

void QrssSpectrogram::mouseDoubleClickEvent(QMouseEvent* event) {
    if (freq_axis_rect().contains(event->position().toPoint())) {
        set_frequency_range(F_LO, F_HI);
        event->accept();
        return;
    }
    QWidget::mouseDoubleClickEvent(event);
}

}  // namespace sstvae::gui

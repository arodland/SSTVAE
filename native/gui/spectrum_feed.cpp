#include "spectrum_feed.hpp"

#include <QDateTime>
#include <QTimer>

#include <algorithm>
#include <cmath>
#include <utility>

#include "rx/ringbuffer.hpp"

namespace sstvae::gui {

SpectrumFeed::SpectrumFeed(QObject* parent) : QObject(parent) {
    clock_ = [] { return static_cast<double>(QDateTime::currentMSecsSinceEpoch()) / 1000.0; };
}

SpectrumFeed::~SpectrumFeed() = default;

void SpectrumFeed::set_ring(std::shared_ptr<rx::RingBuffer> ring) {
    if (ring == ring_) return;
    ring_ = std::move(ring);
    // Start from what the new ring has now; the old one's audio is
    // already stored.
    cursor_ = 0;
    if (ring_) ring_->read_since(0, &cursor_);
    slow_.restart();
    peak_ = 0.0;
}

void SpectrumFeed::push_audio(const std::vector<double>& samples, double end_time) {
    slow_.push(samples, end_time);
}

void SpectrumFeed::start(int fps) {
    if (timer_ == nullptr) {
        timer_ = new QTimer(this);
        connect(timer_, &QTimer::timeout, this, &SpectrumFeed::pump);
    }
    timer_->start(std::max(1, 1000 / std::max(1, fps)));
}

void SpectrumFeed::set_enabled(bool waterfall, bool qrss) {
    fast_on_ = waterfall;
    if (qrss && !slow_on_) {
        if (ring_) cursor_ = ring_->total_written();
        slow_.restart();
    }
    slow_on_ = qrss;
}

void SpectrumFeed::set_clock(std::function<double()> clock) { clock_ = std::move(clock); }

void SpectrumFeed::pump() {
    if (ring_) {
        const double t = now();
        // The waterfall: the newest block, a display-sized slice (never
        // the whole ring; see waterfall.hpp).
        const std::vector<double> block =
            fast_on_ ? ring_->tail(dsp::WATERFALL_NFFT) : std::vector<double>{};
        if (static_cast<int>(block.size()) == dsp::WATERFALL_NFFT) {
            peak_ = 0.0;
            for (const double sample : block) peak_ = std::max(peak_, std::abs(sample));
            fast_.push(t, dsp::spectrum_db(block, dsp::WATERFALL_BINS));
        }
        // The QRSS store: every sample, in order.
        if (!slow_on_) {
            emit pumped();
            return;
        }
        std::uint64_t total = 0;
        std::vector<double> x = ring_->read_since(cursor_, &total);
        if (total < cursor_) {
            x = ring_->read_since(0, &total);
            slow_.restart();
        }
        cursor_ = total;
        if (!x.empty()) slow_.push(x, t);
    }
    emit pumped();
}

}  // namespace sstvae::gui

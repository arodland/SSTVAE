#include "spectrum_feed.hpp"

#include <QDateTime>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <utility>

#include "rx/ringbuffer.hpp"

namespace sstvae::gui {

SpectrumFeed::SpectrumFeed(QObject* parent) : QObject(parent) {
    clock_ = [] { return static_cast<double>(QDateTime::currentMSecsSinceEpoch()) / 1000.0; };
}

SpectrumFeed::~SpectrumFeed() { stop(); }

void SpectrumFeed::set_ring(std::shared_ptr<rx::RingBuffer> ring) {
    const std::lock_guard<std::mutex> hold(mutex_);
    if (ring == ring_) return;
    ring_ = std::move(ring);
    // Start from what the new ring has now; the old one's audio is
    // already stored.
    cursor_ = ring_ ? ring_->total_written() : 0;
    slow_.restart();
    peak_ = 0.0;
}

void SpectrumFeed::push_audio(const std::vector<double>& samples, double end_time) {
    const std::lock_guard<std::mutex> hold(mutex_);
    slow_.push(samples, end_time);
}

void SpectrumFeed::start(int fps) {
    fps_.store(std::clamp(fps, 1, 60));
    if (worker_.joinable()) return;   // the running worker reads the new rate
    stopping_.store(false);
    worker_ = std::thread([this] {
        auto next = std::chrono::steady_clock::now();
        while (!stopping_.load()) {
            pump();
            next += std::chrono::microseconds(1000000 / std::max(1, fps_.load()));
            const auto now = std::chrono::steady_clock::now();
            // Behind by more than a pump: start counting again from now
            // rather than run a burst to catch up.
            if (next < now) next = now;
            std::unique_lock<std::mutex> lk(wake_mutex_);
            wake_.wait_until(lk, next, [this] { return stopping_.load(); });
        }
    });
}

void SpectrumFeed::stop() {
    {
        const std::lock_guard<std::mutex> lk(wake_mutex_);
        stopping_.store(true);
    }
    wake_.notify_all();
    if (worker_.joinable()) worker_.join();
}

void SpectrumFeed::set_enabled(bool waterfall, bool qrss) {
    const std::lock_guard<std::mutex> hold(mutex_);
    fast_on_ = waterfall;
    if (qrss && !slow_on_) {
        if (ring_) cursor_ = ring_->total_written();
        slow_.restart();
    }
    slow_on_ = qrss;
}

void SpectrumFeed::set_clock(std::function<double()> clock) {
    const std::lock_guard<std::mutex> hold(mutex_);
    clock_ = std::move(clock);
}

double SpectrumFeed::now() const {
    const std::lock_guard<std::mutex> hold(mutex_);
    return clock_();
}

void SpectrumFeed::pump() {
    {
        const std::lock_guard<std::mutex> hold(mutex_);
        if (ring_) {
            const double t = clock_();
            // The waterfall: the newest block, a display-sized slice
            // (never the whole ring; see waterfall.hpp).
            const std::vector<double> block =
                fast_on_ ? ring_->tail(dsp::WATERFALL_NFFT) : std::vector<double>{};
            if (static_cast<int>(block.size()) == dsp::WATERFALL_NFFT) {
                peak_ = 0.0;
                for (const double sample : block) peak_ = std::max(peak_, std::abs(sample));
                fast_.push(t, dsp::spectrum_db(block, dsp::WATERFALL_BINS));
            }
            // The QRSS store: every sample, in order.
            if (slow_on_) {
                std::uint64_t total = 0;
                std::vector<double> x = ring_->read_since(cursor_, &total);
                if (total < cursor_) {
                    x = ring_->read_since(0, &total);
                    slow_.restart();
                }
                cursor_ = total;
                if (!x.empty()) slow_.push(x, t);
            }
        }
    }
    emit pumped();
}

}  // namespace sstvae::gui

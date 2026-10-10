// One spectrum feed for both displays (Jeff, 2026-10-10): the waterfall
// across the top of the window and the QRSSTVAE spectrogram are drawn
// from the same audio, read from the capture ring once, and both kept in
// memory -- the waterfall's rows for ten minutes (dsp::WaterfallHistory),
// the QRSS columns for six hours (dsp::SlowSpectrogram), together under
// 100 MB.
//
// The two displays need two transforms, not one, and this is the reason
// there are two stores: the waterfall is for tuning and "is someone
// transmitting", so it is 20 rows a second of a 1024-point FFT (7.8 Hz,
// an eighth of a second); a QRSS carrier is a hertz wide and takes
// minutes to see, so the spectrogram is a column a second of an
// 8192-point FFT (1 Hz). One FFT shape for both would blur the waterfall
// by a whole second or widen the QRSS bins eightfold, losing 9 dB on a
// carrier.
//
// Each display can also run with a feed of its own (the default), which
// is how their tests drive them; the main window makes one and hands it
// to both.
//
// **`start()` pumps on a worker thread of its own** (Jeff, 2026-10-10),
// so the FFTs never run on the screen's thread. Everything the feed
// holds is behind one mutex: a reader of `slow()`/`fast()`/`peak()`
// while the worker runs holds `lock()` for as long as it uses them.
// `pumped` is emitted from the worker and so reaches a widget queued.

#ifndef SSTVAE_GUI_SPECTRUM_FEED_HPP
#define SSTVAE_GUI_SPECTRUM_FEED_HPP

#include <QObject>

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include "dsp/slow_spectrogram.hpp"
#include "dsp/spectrum.hpp"

namespace sstvae::rx {
class RingBuffer;
}

namespace sstvae::gui {

class SpectrumFeed : public QObject {
    Q_OBJECT

public:
    explicit SpectrumFeed(QObject* parent = nullptr);
    ~SpectrumFeed() override;

    // The capture ring to read; null while not receiving. Replacing it
    // keeps everything already stored.
    void set_ring(std::shared_ptr<rx::RingBuffer> ring);
    const std::shared_ptr<rx::RingBuffer>& ring() const { return ring_; }
    // Audio straight into the QRSS store, whose last sample was captured
    // at `end_time` (for tests and for audio not from the ring).
    void push_audio(const std::vector<double>& samples, double end_time);
    // Pump on the worker thread, `fps` times a second (the waterfall's
    // row rate); again to change the rate.
    void start(int fps);
    void stop();
    int fps() const { return fps_.load(); }
    // Held while reading the stores when the worker may be running.
    std::unique_lock<std::mutex> lock() const { return std::unique_lock<std::mutex>(mutex_); }
    void set_clock(std::function<double()> clock);
    // Which stores to fill: the waterfall's rows (SSTVAE mode) and the
    // QRSS columns (QRSSTVAE mode). Both on by default. A store switched
    // back on starts from the audio arriving then; the gap stays a gap.
    void set_enabled(bool waterfall, bool qrss);
    bool waterfall_enabled() const { return fast_on_; }
    bool qrss_enabled() const { return slow_on_; }
    double now() const;

    const dsp::SlowSpectrogram& slow() const { return slow_; }
    const dsp::WaterfallHistory& fast() const { return fast_; }
    // The newest waterfall block's peak sample, for the level meter.
    double peak() const { return peak_; }

public slots:
    // One waterfall row from the newest audio, and whatever the ring has
    // gained into the QRSS store.
    void pump();

signals:
    void pumped();

private:
    dsp::SlowSpectrogram slow_;
    dsp::WaterfallHistory fast_;
    std::shared_ptr<rx::RingBuffer> ring_;
    std::uint64_t cursor_ = 0;
    std::function<double()> clock_;
    mutable std::mutex mutex_;
    std::thread worker_;
    std::mutex wake_mutex_;
    std::condition_variable wake_;
    std::atomic<bool> stopping_{false};
    std::atomic<int> fps_{0};
    double peak_ = 0.0;
    bool fast_on_ = true;
    bool slow_on_ = true;
};

}  // namespace sstvae::gui

#endif

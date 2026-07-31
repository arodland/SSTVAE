#include "audio/qt/qtaudio.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <thread>

#include <QAudioDevice>
#include <QAudioFormat>
#include <QAudioSink>
#include <QAudioSource>
#include <QCoreApplication>
#include <QEventLoop>
#include <QIODevice>
#include <QMediaDevices>
#include <QObject>
#include <QThread>
#include <QTimer>

#include "dsp/dsp.hpp"

namespace sstvae::audio::qt {

namespace {

using Clock = std::chrono::steady_clock;

double seconds_since(Clock::time_point t) {
    return std::chrono::duration<double>(Clock::now() - t).count();
}

std::string to_std(const QString& s) { return s.toStdString(); }

std::vector<std::string> names_of(const QList<QAudioDevice>& devices) {
    std::vector<std::string> out;
    out.reserve(static_cast<std::size_t>(devices.size()));
    for (const QAudioDevice& d : devices) out.push_back(to_std(d.description()));
    return out;
}

// Qt's sample format <-> ours. Unknown formats are rejected rather than
// guessed: reading a device's bytes with the wrong interpretation
// produces noise that still decodes as "a signal", which is the failure
// mode this project keeps having to dig out.
std::optional<SampleFormat> from_qt(QAudioFormat::SampleFormat f) {
    switch (f) {
        case QAudioFormat::Float: return SampleFormat::Float;
        case QAudioFormat::Int16: return SampleFormat::Int16;
        case QAudioFormat::Int32: return SampleFormat::Int32;
        case QAudioFormat::UInt8: return SampleFormat::UInt8;
        default: return std::nullopt;
    }
}

// A mono format at `rate` the device actually accepts. Float first
// because it needs no scaling; Int16 is what most radio interfaces
// advertise.
QAudioFormat choose_format(const QAudioDevice& device, int rate) {
    for (const QAudioFormat::SampleFormat sf :
         {QAudioFormat::Float, QAudioFormat::Int16, QAudioFormat::Int32,
          QAudioFormat::UInt8}) {
        QAudioFormat fmt;
        fmt.setSampleRate(rate);
        fmt.setChannelCount(1);
        fmt.setSampleFormat(sf);
        if (device.isFormatSupported(fmt)) return fmt;
    }
    // Nothing mono worked; take the device's own preference and mix down
    // whatever it gives us.
    return device.preferredFormat();
}

std::string error_text(QAudio::Error e) {
    switch (e) {
        case QAudio::OpenError:
            return "could not open the audio device (in use, or gone?)";
        case QAudio::IOError:
            return "audio device I/O error; the device may have been unplugged";
        // UnderrunError is deliberately absent: Qt 6.11 deprecates it and
        // documents that it is never emitted, so naming it here would buy
        // a deprecation warning for a branch that cannot be taken.
        case QAudio::FatalError: return "the audio device stopped working";
        default: return "audio error";
    }
}

QAudioDevice pick(const QList<QAudioDevice>& devices, const std::string& wanted,
                  const QAudioDevice& fallback, const char* direction,
                  const Report& report) {
    if (devices.isEmpty()) {
        throw std::runtime_error(std::string("no audio ") + direction + " devices found");
    }
    const std::vector<std::string> names = names_of(devices);
    const std::optional<std::size_t> idx = match_device(names, wanted);
    if (!wanted.empty() && !idx && report) {
        report(std::string("[audio ") + direction + "] no device matching \"" + wanted +
               "\"; using the system default");
    }
    return idx ? devices.at(static_cast<qsizetype>(*idx)) : fallback;
}

}  // namespace

std::vector<std::string> input_device_names() {
    return names_of(QMediaDevices::audioInputs());
}

std::vector<std::string> output_device_names() {
    return names_of(QMediaDevices::audioOutputs());
}

std::string default_input_name() {
    return to_std(QMediaDevices::defaultAudioInput().description());
}

std::string default_output_name() {
    return to_std(QMediaDevices::defaultAudioOutput().description());
}

// --- capture ----------------------------------------------------------------

// Lives on the capture thread. Everything it touches -- the source, the
// QIODevice, the resampler -- is created and used there, so the only
// cross-thread state is the ring buffer (which is built for exactly one
// writer) and two atomics.
class CaptureWorker : public QObject {
public:
    CaptureWorker(QAudioDevice device, rx::RingBuffer& ring, int samplerate,
                  Report on_error)
        : device_(std::move(device)),
          ring_(ring),
          samplerate_(samplerate),
          report_(std::move(on_error)) {}

    // Called on the capture thread once its event loop is running.
    void start() {
        const int rate = device_.preferredFormat().sampleRate() > 0
                             ? device_.preferredFormat().sampleRate()
                             : samplerate_;
        const QAudioFormat fmt = choose_format(device_, rate);

        const std::optional<SampleFormat> sf = from_qt(fmt.sampleFormat());
        if (!sf) {
            fail("unsupported sample format from the device");
            return;
        }
        format_ = *sf;
        channels_ = fmt.channelCount();
        rate_.store(fmt.sampleRate(), std::memory_order_relaxed);

        if (fmt.sampleRate() != samplerate_) {
            const Ratio r = resample_ratio(fmt.sampleRate(), samplerate_);
            resampler_ = std::make_unique<StreamResampler>(r.up, r.down);
        }

        source_ = std::make_unique<QAudioSource>(device_, fmt);
        source_->setBufferSize(static_cast<qsizetype>(
            BUFFER_SECONDS * fmt.sampleRate() * channels_ * fmt.bytesPerSample()));
        io_ = source_->start();
        if (io_ == nullptr) {
            fail("could not start capture on \"" + to_std(device_.description()) + "\"");
            return;
        }
        connect(io_, &QIODevice::readyRead, this, &CaptureWorker::drain);
        started_.store(true, std::memory_order_release);
    }

    void shutdown() {
        if (io_ != nullptr) {
            disconnect(io_, &QIODevice::readyRead, this, &CaptureWorker::drain);
            io_ = nullptr;
        }
        if (source_) {
            source_->stop();
            source_.reset();
        }
    }

    bool started() const { return started_.load(std::memory_order_acquire); }
    bool failed() const { return failed_.load(std::memory_order_acquire); }
    const std::string& error() const { return error_; }
    int rate() const { return rate_.load(std::memory_order_relaxed); }
    std::uint64_t captured() const { return captured_.load(std::memory_order_relaxed); }

private:
    void fail(const std::string& why) {
        error_ = why;
        failed_.store(true, std::memory_order_release);
        if (report_) report_("[audio in] " + why);
    }

    void check_error() {
        const QAudio::Error e = source_->error();
        if (e != QAudio::NoError && e != last_error_) {
            last_error_ = e;
            if (report_) report_("[audio in] " + error_text(e));
        }
    }

    void drain() {
        check_error();
        const QByteArray raw = io_->readAll();
        if (raw.isEmpty()) return;

        const std::vector<double> mono = bytes_to_mono(
            std::span<const std::byte>(reinterpret_cast<const std::byte*>(raw.constData()),
                                       static_cast<std::size_t>(raw.size())),
            format_, channels_);
        if (mono.empty()) return;
        captured_.fetch_add(mono.size(), std::memory_order_relaxed);

        if (resampler_) {
            const std::vector<double> out = (*resampler_)(mono);
            if (!out.empty()) ring_.write(out);
        } else {
            ring_.write(mono);
        }
    }

    QAudioDevice device_;
    rx::RingBuffer& ring_;
    int samplerate_;
    Report report_;

    std::unique_ptr<QAudioSource> source_;
    QIODevice* io_ = nullptr;
    std::unique_ptr<StreamResampler> resampler_;
    SampleFormat format_ = SampleFormat::Float;
    int channels_ = 1;
    QAudio::Error last_error_ = QAudio::NoError;

    std::atomic<bool> started_{false};
    std::atomic<bool> failed_{false};
    std::string error_;
    std::atomic<int> rate_{0};
    std::atomic<std::uint64_t> captured_{0};
};

struct InputStream::Impl {
    QThread thread;
    std::unique_ptr<CaptureWorker> worker;
    std::string device_name;
};

InputStream::InputStream(const std::string& device_name, rx::RingBuffer& ring,
                         int samplerate, Report on_error)
    : impl_(std::make_unique<Impl>()) {
    const QAudioDevice device =
        pick(QMediaDevices::audioInputs(), device_name,
             QMediaDevices::defaultAudioInput(), "in", on_error);
    impl_->device_name = to_std(device.description());

    impl_->worker = std::make_unique<CaptureWorker>(device, ring, samplerate, on_error);
    impl_->worker->moveToThread(&impl_->thread);
    QObject::connect(&impl_->thread, &QThread::started, impl_->worker.get(),
                     &CaptureWorker::start);
    impl_->thread.start();

    // Wait for the worker to open the device, so a failure is reported
    // by the constructor rather than as silence. Bounded so a wedged
    // backend cannot hang the caller: the receiver would then never
    // start, which is at least visible.
    const Clock::time_point t0 = Clock::now();
    while (!impl_->worker->started() && !impl_->worker->failed()) {
        if (seconds_since(t0) > 5.0) {
            stop();
            throw std::runtime_error("timed out opening capture on \"" +
                                     impl_->device_name + "\"");
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    if (impl_->worker->failed()) {
        const std::string why = impl_->worker->error();
        stop();
        throw std::runtime_error(why);
    }

    if (on_error) {
        const int rate = impl_->worker->rate();
        on_error("[audio in] " + impl_->device_name + " at " + std::to_string(rate) +
                 " Hz" + (rate == samplerate
                              ? ""
                              : ", resampled to " + std::to_string(samplerate) + " Hz"));
    }
}

InputStream::~InputStream() { stop(); }

void InputStream::stop() {
    if (!impl_ || !impl_->thread.isRunning()) return;
    // Shut the device down *on its own thread*: QAudioSource is not
    // thread-affine by accident, and tearing it down from here would be
    // the same class of bug as the one this whole layer exists to avoid.
    // The functor form, not the by-name one: naming a slot as a string
    // needs moc, and this worker deliberately has no Q_OBJECT -- every
    // connection it makes is the compile-checked pointer-to-member form.
    CaptureWorker* worker = impl_->worker.get();
    QMetaObject::invokeMethod(worker, [worker] { worker->shutdown(); },
                              Qt::BlockingQueuedConnection);
    impl_->thread.quit();
    impl_->thread.wait();
}

int InputStream::device_rate() const { return impl_->worker->rate(); }
std::string InputStream::device_name() const { return impl_->device_name; }
std::uint64_t InputStream::samples_captured() const { return impl_->worker->captured(); }

// --- playback ---------------------------------------------------------------

namespace {

// Lives on its own thread, like CaptureWorker: on Windows, QAudioSink's
// WASAPI backend needs an event loop pumped on the thread that owns the
// sink to actually move written bytes to the device -- push mode still
// relies on timer-driven bookkeeping internally. Without one, write()
// happily queues into the sink's buffer and nothing ever drains, which
// reads as total silence (no meter deflection, nothing on the monitor)
// until the stream is torn down, at which point the buffered tail
// escapes in one burst. That is exactly what a plain std::thread with no
// event loop -- which is what TxEngine calls this from -- produces.
// PulseAudio/PipeWire/CoreAudio service the stream from their own native
// thread and never showed it, which is why the loopback shakedown never
// caught it.
class PlaybackWorker : public QObject {
public:
    PlaybackWorker(QAudioDevice device, std::vector<std::byte> data, QAudioFormat fmt,
                   qsizetype frame_bytes, std::function<void(double)> on_progress,
                   std::function<bool()> should_stop, Report on_error)
        : device_(std::move(device)),
          data_(std::move(data)),
          fmt_(fmt),
          frame_bytes_(frame_bytes),
          on_progress_(std::move(on_progress)),
          should_stop_(std::move(should_stop)),
          report_(std::move(on_error)),
          timer_(this) {}

    // Called on the playback thread once its event loop is running.
    void start() {
        sink_ = std::make_unique<QAudioSink>(device_, fmt_);
        sink_->setBufferSize(static_cast<qsizetype>(BUFFER_SECONDS * fmt_.sampleRate() *
                                                     fmt_.channelCount() *
                                                     fmt_.bytesPerSample()));
        io_ = sink_->start();
        if (io_ == nullptr) {
            if (report_) {
                report_("could not start playback on \"" + to_std(device_.description()) +
                        "\"");
            }
            finish(false);
            return;
        }
        connect(&timer_, &QTimer::timeout, this, &PlaybackWorker::feed);
        timer_.start(5);
    }

    void shutdown() {
        timer_.stop();
        io_ = nullptr;
        if (sink_) {
            sink_->stop();
            sink_.reset();
        }
    }

    bool finished() const { return finished_.load(std::memory_order_acquire); }
    bool succeeded() const { return success_.load(std::memory_order_relaxed); }

private:
    void finish(bool ok) {
        timer_.stop();
        success_.store(ok, std::memory_order_relaxed);
        finished_.store(true, std::memory_order_release);
    }

    void feed() {
        if (should_stop_ && should_stop_()) {
            finish(false);
            return;
        }

        if (phase_ == Phase::Writing) {
            const auto total = static_cast<qsizetype>(data_.size());
            const qsizetype free = sink_->bytesFree();
            // Whole frames only; a partial frame would desync the stream.
            const qsizetype chunk =
                free > 0 ? (std::min(free, total - pos_) / frame_bytes_) * frame_bytes_ : 0;
            if (chunk > 0) {
                const qsizetype written =
                    io_->write(reinterpret_cast<const char*>(data_.data() + pos_), chunk);
                if (written < 0) {
                    finish(false);
                    return;
                }
                pos_ += written;
                if (on_progress_) {
                    on_progress_(
                        std::min(1.0, static_cast<double>(pos_) / static_cast<double>(total)));
                }
            }
            if (pos_ >= total) {
                // Let the device drain what is still buffered, or the
                // tail of the transmission is cut off mid-picture.
                phase_ = Phase::Draining;
                drain_deadline_ = Clock::now() + std::chrono::duration_cast<Clock::duration>(
                                                     std::chrono::duration<double>(
                                                         BUFFER_SECONDS + 1.0));
            }
            return;
        }

        if (sink_->bytesFree() >= sink_->bufferSize() || Clock::now() >= drain_deadline_) {
            finish(true);
        }
    }

    QAudioDevice device_;
    std::vector<std::byte> data_;
    QAudioFormat fmt_;
    qsizetype frame_bytes_;
    std::function<void(double)> on_progress_;
    std::function<bool()> should_stop_;
    Report report_;

    std::unique_ptr<QAudioSink> sink_;
    QIODevice* io_ = nullptr;
    QTimer timer_;
    qsizetype pos_ = 0;

    enum class Phase { Writing, Draining };
    Phase phase_ = Phase::Writing;
    Clock::time_point drain_deadline_;

    std::atomic<bool> success_{false};
    std::atomic<bool> finished_{false};
};

}  // namespace

bool play(const std::string& device_name, std::span<const double> samples, int samplerate,
          const std::function<void(double)>& on_progress,
          const std::function<bool()>& should_stop, const Report& on_error) {
    const QAudioDevice device =
        pick(QMediaDevices::audioOutputs(), device_name,
             QMediaDevices::defaultAudioOutput(), "out", on_error);

    const QAudioFormat fmt = choose_format(device, samplerate);
    const int rate = fmt.sampleRate();
    const std::optional<SampleFormat> sf = from_qt(fmt.sampleFormat());
    if (!sf) throw std::runtime_error("unsupported output sample format");

    std::vector<double> x(samples.begin(), samples.end());
    if (rate != samplerate) {
        const Ratio r = resample_ratio(samplerate, rate);
        x = dsp::resample_poly(x, r.up, r.down);
        if (on_error) {
            on_error("[audio out] " + to_std(device.description()) + " wants " +
                     std::to_string(rate) + " Hz; resampled from " +
                     std::to_string(samplerate) + " Hz");
        }
    }

    const std::vector<std::byte> buf = mono_to_bytes(x, *sf, fmt.channelCount());
    const auto frame_bytes =
        static_cast<qsizetype>(std::max(1, fmt.channelCount() * fmt.bytesPerSample()));

    // A generous ceiling so a wedged device cannot block transmit
    // forever. TxEngine's watchdog is the real backstop; this keeps the
    // ordinary case from depending on it.
    const double budget_s =
        static_cast<double>(x.size()) / std::max(rate, 1) + 30.0 + BUFFER_SECONDS;

    QThread thread;
    auto worker = std::make_unique<PlaybackWorker>(device, buf, fmt, frame_bytes, on_progress,
                                                    should_stop, on_error);
    worker->moveToThread(&thread);
    QObject::connect(&thread, &QThread::started, worker.get(), &PlaybackWorker::start);
    thread.start();

    const Clock::time_point deadline =
        Clock::now() +
        std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(budget_s));
    while (!worker->finished() && Clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }

    // Shut the sink down *on its own thread*, same reasoning as
    // InputStream::stop(): QAudioSink is not thread-affine by accident.
    PlaybackWorker* w = worker.get();
    QMetaObject::invokeMethod(w, [w] { w->shutdown(); }, Qt::BlockingQueuedConnection);
    thread.quit();
    thread.wait();

    return worker->finished() && worker->succeeded();
}

}  // namespace sstvae::audio::qt

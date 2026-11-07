#include "tci/tci.hpp"

#include <QElapsedTimer>
#include <QTimer>
#include <QUrl>
#include <QWebSocket>

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

#include "audio/audio.hpp"
#include "config.hpp"
#include "dsp/dsp.hpp"
#include "rx/ringbuffer.hpp"

namespace sstvae::tci {

namespace {

static_assert(std::endian::native == std::endian::little,
              "TCI's samples are little-endian; copied as they are");

// The stream header: 16 little-endian uint32s ahead of the samples
// (receiver, sample_rate, format, codec, crc, length, type, channels,
// reserved...).
constexpr qsizetype HEADER = 64;
enum Field { RECEIVER = 0, SAMPLE_RATE = 1, FORMAT = 2, LENGTH = 5, TYPE = 6, CHANNELS = 7 };
enum Type : std::uint32_t { RX_AUDIO = 1, TX_AUDIO = 2, TX_CHRONO = 3 };
constexpr std::uint32_t FLOAT32 = 3;

constexpr int TICK_MS = 20;
constexpr qint64 CHRONO_GAP_MS = 500;  // sending with no TX_CHRONO this long: fed on our clock
constexpr int RETRY_MS = 2000;

std::array<std::uint32_t, 16> header(const QByteArray& f) {
    std::array<std::uint32_t, 16> h;
    std::memcpy(h.data(), f.constData(), sizeof h);
    return h;
}

void put(QByteArray& f, int i, std::uint32_t v) { std::memcpy(f.data() + 4 * i, &v, 4); }

// TCI's sample types; int24 (1) is one CapturePipeline lacks.
std::optional<audio::SampleFormat> sample_format(std::uint32_t f) {
    switch (f) {
        case 0: return audio::SampleFormat::Int16;
        case 2: return audio::SampleFormat::Int32;
        case 3: return audio::SampleFormat::Float;
        default: return std::nullopt;
    }
}

}  // namespace

std::optional<std::pair<std::string, int>> parse_address(const std::string& s) {
    const auto colon = s.rfind(':');
    const std::string host = s.substr(0, colon);
    if (host.empty()) return std::nullopt;
    if (colon == std::string::npos) return std::pair{host, DEFAULT_PORT};
    const std::string port = s.substr(colon + 1);
    if (port.empty() || port.size() > 5 ||
        port.find_first_not_of("0123456789") != std::string::npos) {
        return std::nullopt;
    }
    const int p = std::stoi(port);
    if (p < 1 || p > 65535) return std::nullopt;
    return std::pair{host, p};
}

struct Tci::Worker : QObject {
    explicit Worker(Status on_status) : on_status(std::move(on_status)) {}

    const Status on_status;

    std::atomic<bool> ready{false};          // read by set_ptt/play's callers
    std::atomic<std::size_t> tx_pos{0};      // samples of tx_buf sent so far
    mutable std::mutex mu;
    std::optional<double> dial;  // mu

    // the rest on the worker's thread only
    QUrl url;
    std::string modulation;
    bool enabled = false, retry_pending = false;
    QWebSocket* ws = nullptr;
    QTimer* tick = nullptr;
    QElapsedTimer clock;
    std::string last_error;
    std::shared_ptr<rx::RingBuffer> ring;
    std::unique_ptr<audio::CapturePipeline> pipe;
    std::uint32_t pipe_format = 0, pipe_channels = 0, pipe_rate = 0;
    bool format_warned = false;
    std::vector<float> tx_buf;
    bool tx_active = false;
    qint64 last_chrono = -1, tx_started = 0, push_from = 0;

    void report(const std::string& text, bool error) {
        if (on_status) on_status(text, error);
    }

    void init() {
        clock.start();
        ws = new QWebSocket(QString(), QWebSocketProtocol::VersionLatest, this);
        connect(ws, &QWebSocket::disconnected, this, [this] { lost(); });
        connect(ws, &QWebSocket::errorOccurred, this, [this](QAbstractSocket::SocketError) {
            const std::string e = ws->errorString().toStdString();
            if (enabled && e != last_error) {
                last_error = e;
                report("TCI " + url.toString().toStdString() + ": " + e + " (retrying every " +
                           std::to_string(RETRY_MS / 1000) + " s)",
                       true);
            }
            lost();
        });
        connect(ws, &QWebSocket::textMessageReceived, this, [this](const QString& m) { text(m); });
        connect(ws, &QWebSocket::binaryMessageReceived, this,
                [this](const QByteArray& f) { binary(f); });
        tick = new QTimer(this);
        tick->setTimerType(Qt::PreciseTimer);
        connect(tick, &QTimer::timeout, this, [this] { on_tick(); });
        tick->start(TICK_MS);
    }

    void start(const QUrl& u, std::string mod) {
        enabled = false;  // the abort below is not a loss to report or retry
        ws->abort();
        url = u;
        modulation = std::move(mod);
        enabled = true;
        last_error.clear();
        open();
    }

    void stop() {
        enabled = false;
        // A close frame, not abort(): the server sees an orderly goodbye
        // rather than a dropped connection.
        if (ws->state() == QAbstractSocket::ConnectedState) {
            ws->close();
        } else {
            ws->abort();
        }
        reset();
        const std::lock_guard<std::mutex> lock(mu);
        dial.reset();
    }

    void open() {
        retry_pending = false;
        if (enabled && ws->state() == QAbstractSocket::UnconnectedState) ws->open(url);
    }

    void reset() {
        ready = false;
        pipe.reset();
    }

    void lost() {
        if (ready && enabled) {
            report("TCI: connection lost (retrying every " + std::to_string(RETRY_MS / 1000) +
                       " s)",
                   true);
        }
        reset();
        if (!enabled || retry_pending) return;
        retry_pending = true;
        QTimer::singleShot(RETRY_MS, this, [this] { open(); });
    }

    // On the worker's thread, before it quits: what lives on it is
    // deleted here.
    void close() {
        enabled = false;
        delete tick;
        tick = nullptr;
        if (ws) ws->abort();
        delete ws;
        ws = nullptr;
    }

    void send(const QString& cmd) { ws->sendTextMessage(cmd); }

    bool set_ptt(bool on) {
        if (!ready) return false;
        send(on ? QStringLiteral("trx:0,true,tci;") : QStringLiteral("trx:0,false;"));
        return true;
    }

    bool begin_tx(std::vector<float> buf) {
        if (!ready) return false;
        tx_buf = std::move(buf);
        tx_pos = 0;
        tx_active = true;
        tx_started = push_from = clock.elapsed();
        return true;
    }

    void end_tx() {
        tx_active = false;
        tx_buf.clear();
    }

    void text(const QString& message) {
        for (const QString& part : message.split(QLatin1Char(';'), Qt::SkipEmptyParts)) {
            const QString cmd = part.trimmed();
            const qsizetype colon = cmd.indexOf(QLatin1Char(':'));
            const QString name = cmd.left(colon).toLower();
            const QStringList args =
                colon < 0 ? QStringList() : cmd.mid(colon + 1).split(QLatin1Char(','));
            if (name == QLatin1String("ready")) {
                ready = true;
                last_error.clear();
                report("TCI: connected to " + url.toString().toStdString(), false);
                // One command a message; the server answers each with its
                // value (a rate it lacks: the one it keeps).
                send(QStringLiteral("audio_samplerate:%1;").arg(RATE));
                send(QStringLiteral("audio_stream_sample_type:float32;"));
                send(QStringLiteral("audio_stream_channels:1;"));
                if (!modulation.empty()) {
                    send(QStringLiteral("modulation:0,%1;").arg(QString::fromStdString(modulation)));
                }
                send(QStringLiteral("audio_start:0;"));
            } else if (name == QLatin1String("vfo") && args.size() == 3 &&
                       args[0] == QLatin1String("0") && args[1] == QLatin1String("0")) {
                bool ok = false;
                const double hz = args[2].toDouble(&ok);
                const std::lock_guard<std::mutex> lock(mu);
                if (ok) dial = hz;
            } else if (name == QLatin1String("audio_samplerate") && args.size() == 1 &&
                       args[0].toInt() != RATE) {
                // Our TX frames state their rate; converting them is the
                // server's business. RX is resampled whatever it sends.
                report("TCI: the server keeps its audio at " + args[0].toStdString() +
                           " Hz, not " + std::to_string(RATE),
                       false);
            }
        }
    }

    void binary(const QByteArray& f) {
        if (f.size() < HEADER) return;
        const auto h = header(f);
        if (h[RECEIVER] != 0) return;
        if (h[TYPE] == RX_AUDIO) {
            if (!ring) return;
            const auto fmt = sample_format(h[FORMAT]);
            const std::uint32_t r = h[SAMPLE_RATE];
            // 0 before TCI 1.9, which was stereo
            const std::uint32_t ch = h[CHANNELS] ? h[CHANNELS] : 2;
            if (!fmt || r == 0) {
                if (!format_warned) {
                    report("TCI: RX audio as type " + std::to_string(h[FORMAT]) + " at " +
                               std::to_string(r) + " Hz: want int16, int32 or float32",
                           true);
                }
                format_warned = true;
                return;
            }
            if (!pipe || h[FORMAT] != pipe_format || ch != pipe_channels || r != pipe_rate) {
                pipe = std::make_unique<audio::CapturePipeline>(
                    *fmt, static_cast<int>(ch), static_cast<int>(r), config::FS);
                pipe_format = h[FORMAT];
                pipe_channels = ch;
                pipe_rate = r;
            }
            // The samples by the frame's size, not `length`: its meaning
            // (per channel or not) changed in 1.9.
            const std::vector<double> out = (*pipe)(
                {reinterpret_cast<const std::byte*>(f.constData() + HEADER),
                 static_cast<std::size_t>(f.size() - HEADER)});
            if (!out.empty()) ring->write(out);
        } else if (h[TYPE] == TX_CHRONO) {
            last_chrono = clock.elapsed();
            if (!tx_active) return;
            // The reply echoes length and channels; before 1.9 (channels
            // 0) length was per channel, in stereo.
            const std::uint32_t length = h[LENGTH], ch = h[CHANNELS];
            send_tx(ch ? length / ch : length, ch ? ch : 2, length, ch);
        }
    }

    // `frames` of tx_buf (silence past its end), on every one of
    // `channels`, as one TX_AUDIO frame.
    void send_tx(std::size_t frames, std::uint32_t channels, std::uint32_t length,
                 std::uint32_t channels_field) {
        QByteArray out(HEADER + static_cast<qsizetype>(frames * channels * sizeof(float)), '\0');
        put(out, SAMPLE_RATE, static_cast<std::uint32_t>(RATE));
        put(out, FORMAT, FLOAT32);
        put(out, LENGTH, length);
        put(out, TYPE, TX_AUDIO);
        put(out, CHANNELS, channels_field);
        const std::size_t pos = tx_pos;
        char* d = out.data() + HEADER;
        for (std::size_t i = pos; i < pos + frames && i < tx_buf.size(); ++i) {
            for (std::uint32_t c = 0; c < channels; ++c, d += sizeof(float)) {
                std::memcpy(d, &tx_buf[i], sizeof(float));
            }
        }
        tx_pos = pos + frames;
        ws->sendBinaryMessage(out);
    }

    void on_tick() {
        if (!tx_active || !ready) return;
        const qint64 now = clock.elapsed();
        // A server that never sends TX_CHRONO is fed at once; one that
        // has, only once it misses this over's.
        if (last_chrono < 0 || now - std::max(last_chrono, tx_started) > CHRONO_GAP_MS) {
            const auto frames = static_cast<std::size_t>((now - push_from) * RATE / 1000);
            if (frames) send_tx(frames, 1, static_cast<std::uint32_t>(frames), 1);
        }
        push_from = now;
    }
};

Tci::Tci(Status on_status) : w_(new Worker(std::move(on_status))) {
    w_->moveToThread(&thread_);
    QObject::connect(&thread_, &QThread::started, w_, [w = w_] { w->init(); });
    thread_.start();
}

Tci::~Tci() {
    QMetaObject::invokeMethod(w_, [w = w_] { w->close(); }, Qt::BlockingQueuedConnection);
    thread_.quit();
    thread_.wait();
    delete w_;
}

void Tci::start(const std::string& host, int port, const std::string& modulation) {
    const QUrl url(QStringLiteral("ws://%1:%2").arg(QString::fromStdString(host)).arg(port));
    QMetaObject::invokeMethod(w_, [w = w_, url, modulation] { w->start(url, modulation); });
}

void Tci::stop() {
    QMetaObject::invokeMethod(w_, [w = w_] { w->stop(); });
}

void Tci::set_capture(std::shared_ptr<rx::RingBuffer> ring) {
    QMetaObject::invokeMethod(w_, [w = w_, ring] {
        w->ring = ring;
        w->pipe.reset();
    });
}

void Tci::set_ptt(bool on) {
    bool ok = false;
    QMetaObject::invokeMethod(w_, [this, on, &ok] { ok = w_->set_ptt(on); },
                              Qt::BlockingQueuedConnection);
    if (!ok) throw std::runtime_error("TCI: not connected");
}

bool Tci::play(std::span<const double> samples, int samplerate,
               const std::function<void(double)>& on_progress,
               const std::function<bool()>& should_stop,
               const std::function<void(const std::string&)>& on_error) {
    // Resampled once, up front, as audio::qt::play does: the whole
    // waveform is in hand.
    std::vector<double> x(samples.begin(), samples.end());
    if (samplerate != RATE) {
        const audio::Ratio r = audio::resample_ratio(samplerate, RATE);
        x = dsp::resample_poly(x, r.up, r.down);
    }
    std::vector<float> buf(x.begin(), x.end());
    const std::size_t total = buf.size();

    bool ok = false;
    QMetaObject::invokeMethod(
        w_, [this, &buf, &ok] { ok = w_->begin_tx(std::move(buf)); },
        Qt::BlockingQueuedConnection);
    if (!ok) throw std::runtime_error("TCI: not connected");

    // A generous ceiling so a server that stops asking cannot block
    // transmit forever; TxEngine's watchdog is the real backstop.
    using Clock = std::chrono::steady_clock;
    const auto deadline = Clock::now() + std::chrono::seconds(30) +
                          std::chrono::milliseconds(total * 1000 / RATE);
    bool completed = false;
    while (true) {
        const std::size_t pos = w_->tx_pos;
        if (on_progress) on_progress(std::min(1.0, static_cast<double>(pos) / std::max<std::size_t>(total, 1)));
        if (pos >= total) {
            completed = true;
            break;
        }
        if (should_stop && should_stop()) break;
        if (!w_->ready) {
            if (on_error) on_error("TCI: connection lost while sending");
            break;
        }
        if (Clock::now() > deadline) {
            if (on_error) on_error("TCI: the server stopped taking transmit audio");
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    QMetaObject::invokeMethod(w_, [w = w_] { w->end_tx(); }, Qt::BlockingQueuedConnection);
    return completed;
}

std::optional<double> Tci::frequency_hz() const {
    const std::lock_guard<std::mutex> lock(w_->mu);
    return w_->dial;
}

}  // namespace sstvae::tci

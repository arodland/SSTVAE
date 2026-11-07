// The TCI client (core/tci/) against a fake server on loopback.
//
// What it checks is what a radio would notice: the setup commands after
// READY, the dial, RX audio reaching the ring at FS, PTT as `trx`, and a
// transmission arriving whole and in order -- both answering TX_CHRONO
// and, for a server that never sends one, pushed on the client's clock.

#include <QCoreApplication>
#include <QElapsedTimer>
#include <QHostAddress>
#include <QTimer>
#include <QWebSocket>
#include <QWebSocketServer>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <numbers>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "check.hpp"
#include "config.hpp"
#include "dsp/dsp.hpp"
#include "rx/ringbuffer.hpp"
#include "tci/tci.hpp"

using namespace sstvae;

namespace {

bool pump_until(const std::function<bool()>& done, int ms = 5000) {
    QElapsedTimer t;
    t.start();
    while (!done() && t.elapsed() < ms) {
        QCoreApplication::processEvents(QEventLoop::AllEvents, 10);
    }
    return done();
}

QByteArray frame(std::uint32_t type, std::uint32_t rate, std::uint32_t length,
                 std::uint32_t channels, const std::vector<float>& samples = {}) {
    std::array<std::uint32_t, 16> h{};
    h[1] = rate;
    h[2] = 3;  // float32
    h[5] = length;
    h[6] = type;
    h[7] = channels;
    QByteArray f(64 + static_cast<qsizetype>(samples.size() * sizeof(float)), '\0');
    std::memcpy(f.data(), h.data(), 64);
    if (!samples.empty()) std::memcpy(f.data() + 64, samples.data(), samples.size() * sizeof(float));
    return f;
}

// One client at a time; records its commands and TX audio.
struct FakeServer {
    QWebSocketServer server{QStringLiteral("fake"), QWebSocketServer::NonSecureMode};
    QWebSocket* client = nullptr;
    QStringList commands;
    std::vector<float> tx;
    std::vector<std::size_t> tx_frames;  // samples per TX_AUDIO frame
    QTimer chrono;

    explicit FakeServer(bool send_chrono) {
        server.listen(QHostAddress::LocalHost, 0);
        QObject::connect(&server, &QWebSocketServer::newConnection, [this] {
            client = server.nextPendingConnection();
            QObject::connect(client, &QWebSocket::textMessageReceived,
                             [this](const QString& m) { commands << m; });
            QObject::connect(client, &QWebSocket::binaryMessageReceived, [this](const QByteArray& f) {
                std::uint32_t type = 0;
                std::memcpy(&type, f.constData() + 24, 4);
                if (type != 2) return;
                const auto n = static_cast<std::size_t>(f.size() - 64) / sizeof(float);
                const auto* p = reinterpret_cast<const float*>(f.constData() + 64);
                tx.insert(tx.end(), p, p + n);
                tx_frames.push_back(n);
            });
            client->sendTextMessage(QStringLiteral("protocol:ExpertSDR3,2.1;device:fake;ready;"));
        });
        // 2048 samples at 48 kHz, as ExpertSDR3 asks
        chrono.setInterval(43);
        if (send_chrono) {
            QObject::connect(&chrono, &QTimer::timeout, [this] {
                if (client) client->sendBinaryMessage(frame(3, tci::RATE, 2048, 1));
            });
            chrono.start();
        }
    }

    int port() const { return server.serverPort(); }
    bool saw(const char* cmd) const { return commands.contains(QString::fromLatin1(cmd)); }
};

// Half a second of a 1 kHz tone at FS.
std::vector<double> tone() {
    std::vector<double> x(config::FS / 2);
    for (std::size_t i = 0; i < x.size(); ++i) {
        x[i] = 0.5 * std::sin(2 * std::numbers::pi * 1000.0 * static_cast<double>(i) / config::FS);
    }
    return x;
}

// play() blocks, and the server lives on this thread: run it beside the
// event loop.
bool transmit(tci::Tci& t, const std::vector<double>& wave) {
    bool ok = false;
    bool done = false;
    std::thread th([&] {
        ok = t.play(wave, config::FS, {}, {}, {});
        done = true;
    });
    pump_until([&] { return done; }, 10000);
    th.join();
    return ok;
}

void check_tx(const FakeServer& s, const std::vector<double>& wave, const char* what) {
    const std::vector<double> want = dsp::resample_poly(wave, tci::RATE / config::FS, 1);
    check::is_true(s.tx.size() >= want.size(),
                   std::string(what) + ": all of it sent (" + std::to_string(s.tx.size()) +
                       " of " + std::to_string(want.size()) + ")");
    double worst = 0;
    for (std::size_t i = 0; i < want.size() && i < s.tx.size(); ++i) {
        worst = std::max(worst, std::abs(want[i] - s.tx[i]));
    }
    check::is_true(worst < 1e-6, std::string(what) + ": in order, unaltered (worst " +
                                     std::to_string(worst) + ")");
}

void test_chrono() {
    check::current_step = "chrono";
    FakeServer s(true);
    tci::Tci t;
    bool threw = false;
    try {
        t.set_ptt(true);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    check::is_true(threw, "PTT before connecting throws");

    t.start("127.0.0.1", s.port(), "digu");
    check::is_true(pump_until([&] { return s.saw("audio_start:0;"); }), "audio started");
    check::is_true(s.saw("audio_samplerate:48000;") && s.saw("audio_stream_sample_type:float32;") &&
                       s.saw("audio_stream_channels:1;") && s.saw("modulation:0,digu;"),
                   "setup commands: " + s.commands.join(QLatin1Char(' ')).toStdString());

    s.client->sendTextMessage(QStringLiteral("vfo:0,0,14230000;vfo:0,1,7000000;"));
    check::is_true(pump_until([&] { return t.frequency_hz().has_value(); }) &&
                       *t.frequency_hz() == 14230000.0,
                   "the dial is VFO A of transceiver 0");

    // RX: a 48 kHz tone reaches the ring at FS
    auto ring = std::make_shared<rx::RingBuffer>(10.0);
    t.set_capture(ring);
    pump_until([] { return false; }, 100);
    const std::vector<double> x = dsp::resample_poly(tone(), tci::RATE / config::FS, 1);
    for (std::size_t i = 0; i < x.size(); i += 2048) {
        std::vector<float> chunk(x.begin() + static_cast<std::ptrdiff_t>(i),
                                 x.begin() + static_cast<std::ptrdiff_t>(std::min(i + 2048, x.size())));
        s.client->sendBinaryMessage(frame(1, tci::RATE, static_cast<std::uint32_t>(chunk.size()), 1, chunk));
    }
    std::uint64_t total = 0;
    pump_until([&] {
        ring->snapshot(&total);
        return total > config::FS / 2 - 400;
    });
    check::is_true(total > config::FS / 2 - 400 && total <= config::FS / 2,
                   "RX audio reaches the ring at FS (" + std::to_string(total) + ")");
    const std::vector<double> got = ring->snapshot();
    double sum = 0;
    for (std::size_t i = 200; i < got.size() - 200; ++i) sum += got[i] * got[i];
    const double rms = std::sqrt(sum / static_cast<double>(got.size() - 400));
    check::is_true(std::abs(rms - 0.5 / std::sqrt(2.0)) < 0.01, "the tone survives (rms " + std::to_string(rms) + ")");

    t.set_ptt(true);
    check::is_true(pump_until([&] { return s.saw("trx:0,true,tci;"); }), "keyed with trx");
    const std::vector<double> wave = tone();
    check::is_true(transmit(t, wave), "play() finished, chrono-paced");
    check_tx(s, wave, "chrono-paced");
    bool all_answers = !s.tx_frames.empty();
    for (const std::size_t n : s.tx_frames) all_answers = all_answers && n == 2048;
    check::is_true(all_answers, "every TX frame answered a TX_CHRONO, none pushed");
    t.set_ptt(false);
    check::is_true(pump_until([&] { return s.saw("trx:0,false;"); }), "released with trx");

    t.stop();
    check::is_true(pump_until([&] { return !t.frequency_hz().has_value(); }), "stop() forgets the dial");
}

void test_pushed() {
    check::current_step = "pushed";
    FakeServer s(false);
    tci::Tci t;
    t.start("127.0.0.1", s.port(), "");
    check::is_true(pump_until([&] { return s.saw("audio_start:0;"); }), "audio started");
    check::is_true(!s.commands.join(QString()).contains(QStringLiteral("modulation")),
                   "no modulation asked: the operator's stays");
    const std::vector<double> wave = tone();
    check::is_true(transmit(t, wave), "play() finished with no TX_CHRONO");
    pump_until([] { return false; }, 100);
    check_tx(s, wave, "pushed");
}

void test_address() {
    check::current_step = "address";
    using P = std::pair<std::string, int>;
    check::is_true(tci::parse_address("sdr.lan") == P{"sdr.lan", 50001}, "HOST: port 50001");
    check::is_true(tci::parse_address("10.0.0.5:40001") == P{"10.0.0.5", 40001}, "HOST:PORT");
    for (const char* bad : {":40001", "sdr:", "sdr:0", "sdr:65536", "sdr:4x", ""}) {
        check::is_true(!tci::parse_address(bad), std::string("not an address: ") + bad);
    }
}

}  // namespace

int main(int argc, char** argv) {
    check::report_crashes_instead_of_prompting();
    check::Watchdog watchdog(120.0, "tci");
    QCoreApplication app(argc, argv);
    test_address();
    test_chrono();
    test_pushed();
    return check::report("tci");
}

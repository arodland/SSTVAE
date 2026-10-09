// The QRSS window, in the parts with a right answer: the audio reaches
// the listener sample for sample (across a fresh ring, as after a
// transmission), and the tiles say what the listener's state file says.
// How a tile looks is a judgement call and is not asserted.

#include <QApplication>
#include <QDateTime>
#include <QDir>
#include <QElapsedTimer>
#include <QFile>
#include <QImage>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QTemporaryDir>
#include <QThread>

#include <cstdio>
#include <cstring>
#include <functional>
#include <memory>
#include <vector>

#include "check.hpp"
#include "qrss_window.hpp"
#include "rx/ringbuffer.hpp"

using namespace sstvae;

namespace {

QJsonObject tile_json(const QString& id, double f_hz, const QString& call, double progress,
                      const QString& image, int rev) {
    QJsonObject t;
    t[QStringLiteral("id")] = id;
    t[QStringLiteral("q")] = 1990584;
    t[QStringLiteral("slot_utc")] = QStringLiteral("2026-10-09T06:00:00Z");
    t[QStringLiteral("frame")] = QStringLiteral("full");
    t[QStringLiteral("f_hz")] = f_hz;
    t[QStringLiteral("status")] = QStringLiteral("receiving");
    t[QStringLiteral("snr_db")] = -12.5;
    if (!call.isEmpty()) {
        t[QStringLiteral("callsign")] = call;
        t[QStringLiteral("grid")] = QStringLiteral("FN42");
        t[QStringLiteral("picture_id")] = QStringLiteral("98f0da1e");
        t[QStringLiteral("mode")] = QStringLiteral("A");
        t[QStringLiteral("passes")] = 3;
    } else {
        t[QStringLiteral("callsign")] = QJsonValue();
    }
    t[QStringLiteral("progress")] = progress;
    t[QStringLiteral("heard")] = progress;
    t[QStringLiteral("received")] = progress;
    t[QStringLiteral("image")] = image.isEmpty() ? QJsonValue() : QJsonValue(image);
    t[QStringLiteral("image_rev")] = rev;
    return t;
}

// Write state.json with a given modification time, so two writes inside
// one clock tick are still seen as two.
void write_state(const QString& dir, const QJsonArray& tiles, qint64 mtime_ms) {
    QJsonObject st;
    st[QStringLiteral("version")] = 1;
    st[QStringLiteral("listening")] = true;
    st[QStringLiteral("source")] = QStringLiteral("test");
    st[QStringLiteral("tiles")] = tiles;
    QFile f(QDir(dir).filePath(QStringLiteral("state.json")));
    f.open(QIODevice::WriteOnly | QIODevice::Truncate);
    f.write(QJsonDocument(st).toJson());
    f.setFileTime(QDateTime::fromMSecsSinceEpoch(mtime_ms), QFileDevice::FileModificationTime);
    f.close();
}

void test_tiles_follow_the_state_file() {
    QTemporaryDir dir;
    QDir(dir.path()).mkpath(QStringLiteral("tiles"));
    QImage img(64, 48, QImage::Format_RGB32);
    img.fill(Qt::red);
    img.save(QDir(dir.path()).filePath(QStringLiteral("tiles/a.png")));

    gui::QrssWindow w;
    w.set_state_dir(dir.path());
    check::equal(w.tile_count(), 0, "qrss/tiles: none without a state file");

    QJsonArray tiles;
    tiles.append(tile_json(QStringLiteral("a"), 1500.0, QStringLiteral("N0CALL"), 0.35,
                           QStringLiteral("tiles/a.png"), 2));
    tiles.append(tile_json(QStringLiteral("b"), 1550.2, QString(), 0.02, QString(), 0));
    write_state(dir.path(), tiles, 1'000'000);
    w.reload();
    check::equal(w.tile_count(), 2, "qrss/tiles: one per listed signal");
    gui::QrssTile* a = w.tile(QStringLiteral("a"));
    gui::QrssTile* b = w.tile(QStringLiteral("b"));
    check::is_true(a && b, "qrss/tiles: found by id");
    if (!a || !b) return;
    check::equal(a->title().toStdString(), std::string("N0CALL  FN42"),
                 "qrss/tiles: a decoded header names the station");
    check::equal(b->title().toStdString(), std::string("Signal at 1550.2 Hz"),
                 "qrss/tiles: before the header, the frequency");
    check::is_true(a->has_picture(), "qrss/tiles: the picture is loaded");
    check::is_true(!b->has_picture(), "qrss/tiles: no picture before the header");
    check::equal(a->progress_percent(), 35, "qrss/tiles: progress through the pass");
    check::equal(a->progress_text().toStdString(), std::string("35% of the pass"),
                 "qrss/tiles: and in words, beside the bar");
    check::is_true(a->details().contains(QStringLiteral("3 passes")),
                   "qrss/tiles: says how many passes the picture combines");

    // The listener drops a signal: so does the window.
    QJsonArray one;
    one.append(tile_json(QStringLiteral("a"), 1500.0, QStringLiteral("N0CALL"), 0.5,
                         QStringLiteral("tiles/a.png"), 3));
    write_state(dir.path(), one, 2'000'000);
    w.reload();
    QApplication::processEvents();
    check::equal(w.tile_count(), 1, "qrss/tiles: a signal no longer listed goes");
    check::equal(w.tile(QStringLiteral("a"))->progress_percent(), 50,
                 "qrss/tiles: and the one left is updated in place");

    // A half-written file (not JSON) is ignored, not treated as empty.
    {
        QFile f(QDir(dir.path()).filePath(QStringLiteral("state.json")));
        f.open(QIODevice::WriteOnly | QIODevice::Truncate);
        f.write("{\"tiles\": [");
        f.setFileTime(QDateTime::fromMSecsSinceEpoch(3'000'000),
                      QFileDevice::FileModificationTime);
    }
    w.reload();
    check::equal(w.tile_count(), 1, "qrss/tiles: a torn state file changes nothing");
}

std::vector<double> ramp(double from, int n) {
    std::vector<double> v(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) v[static_cast<std::size_t>(i)] = (from + i) * 1e-4;
    return v;
}

bool wait_until(const std::function<bool()>& done, int ms) {
    QElapsedTimer t;
    t.start();
    while (!done()) {
        if (t.elapsed() > ms) return false;
        QApplication::processEvents(QEventLoop::AllEvents, 50);
        QThread::msleep(10);
    }
    return true;
}

void test_audio_reaches_the_listener_once() {
#ifdef _WIN32
    std::printf("skip: qrss/feed needs a POSIX shell\n");
#else
    QTemporaryDir dir;
    const QString out = QDir(dir.path()).filePath(QStringLiteral("audio.raw"));
    gui::QrssWindow w;
    w.set_state_dir(dir.path());
    // A stand-in listener that records its stdin ("--state DIR" lands in
    // $0 and $1, which it ignores).
    w.set_command(QStringLiteral("sh -c \"cat > '%1'\"").arg(out));

    auto ring = std::make_shared<rx::RingBuffer>(1.0, 1000);
    ring->write(ramp(-500, 500));       // heard before the listener: not sent
    w.set_ring(ring);
    w.start_listener();
    check::is_true(wait_until([&] { return w.listener_running(); }, 5000),
                   "qrss/feed: the listener starts");

    std::vector<double> want;
    int next = 0;
    for (int block : {300, 1, 999, 250}) {        // across the 1000-sample wrap
        ring->write(ramp(next, block));
        auto r = ramp(next, block);
        want.insert(want.end(), r.begin(), r.end());
        next += block;
        w.pump_audio();
    }
    // Half duplex: the receive pane replaces the ring after transmitting.
    auto fresh = std::make_shared<rx::RingBuffer>(1.0, 1000);
    w.set_ring(fresh);
    fresh->write(ramp(next, 400));
    auto r = ramp(next, 400);
    want.insert(want.end(), r.begin(), r.end());
    w.pump_audio();
    w.pump_audio();                                // nothing new: nothing sent

    w.stop_listener();
    check::is_true(wait_until([&] { return !w.listener_running(); }, 10000),
                   "qrss/feed: closing its input ends the listener");

    QFile f(out);
    f.open(QIODevice::ReadOnly);
    const QByteArray bytes = f.readAll();
    std::vector<float> got(static_cast<std::size_t>(bytes.size()) / sizeof(float));
    std::memcpy(got.data(), bytes.data(), got.size() * sizeof(float));
    check::equal(got.size(), want.size(), "qrss/feed: every sample, once");
    std::vector<double> gotd(got.begin(), got.end());
    std::vector<double> wantf;
    for (double v : want) wantf.push_back(static_cast<float>(v));
    check::close(gotd, wantf, 0.0, "qrss/feed: in order, as float32");
#endif
}

}  // namespace

int main(int argc, char** argv) {
    check::report_crashes_instead_of_prompting();
    qputenv("QT_QPA_PLATFORM", "offscreen");
    const QApplication app(argc, argv);
    test_tiles_follow_the_state_file();
    test_audio_reaches_the_listener_once();
    return check::report("qrss window");
}

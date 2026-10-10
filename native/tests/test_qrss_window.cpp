// The QRSS window, in the parts with a right answer: the audio reaches
// the listener sample for sample (across a fresh ring, as after a
// transmission), and the tiles say what the listener's state file says.
// How a tile looks is a judgement call and is not asserted.

#include <QApplication>
#include <QColor>
#include <QDateTime>
#include <QDir>
#include <QElapsedTimer>
#include <QFile>
#include <QImage>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QLabel>
#include <QMouseEvent>
#include <QSplitter>
#include <QWheelEvent>
#include <QPushButton>
#include <QTemporaryDir>
#include <QThread>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <memory>
#include <optional>
#include <random>
#include <vector>

#include "check.hpp"
#include "qrss_page.hpp"
#include "qrss_spectrogram.hpp"
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
    t[QStringLiteral("mean_w_db")] = -7.8;
    if (!call.isEmpty()) {
        t[QStringLiteral("callsign")] = call;
        t[QStringLiteral("note")] = QStringLiteral("stored (header)");
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
    check::is_true(f.open(QIODevice::WriteOnly | QIODevice::Truncate),
                   "qrss/tiles: state file opened for writing");
    f.write(QJsonDocument(st).toJson());
    // Flushed first: a flush at close would stamp the real time over this
    // one, and two writes inside one filesystem tick would look like one.
    f.flush();
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
    QJsonObject with_dial = tile_json(QStringLiteral("a"), 1500.0, QStringLiteral("N0CALL"), 0.35,
                                      QStringLiteral("tiles/a.png"), 2);
    with_dial[QStringLiteral("dial_hz")] = 10'138'700;
    tiles.append(with_dial);
    QJsonObject no_dial = tile_json(QStringLiteral("b"), 1550.2, QString(), 0.02, QString(), 0);
    no_dial[QStringLiteral("dial_hz")] = QJsonValue();
    tiles.append(no_dial);
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
    check::equal(a->weight_text().toStdString(), std::string("-7.8 dB"),
                 "qrss/tiles: the mean weight beside the title");
    check::is_true(!a->details().contains(QStringLiteral("SNR")) &&
                       !a->details().contains(QStringLiteral("-12.5")) &&
                       !a->details().contains(QStringLiteral("weight")),
                   "qrss/tiles: no SNR on the tile, and the weight not twice");
    check::is_true(a->has_picture(), "qrss/tiles: the picture is loaded");
    check::is_true(!b->has_picture(), "qrss/tiles: no picture before the header");
    check::equal(a->progress_percent(), 35, "qrss/tiles: progress through the pass");
    check::is_true(!a->details().contains(QStringLiteral("% of")),
                   "qrss/tiles: no percentages in words; the bar says it");
    check::is_true(a->details().contains(QStringLiteral("3 passes")),
                   "qrss/tiles: says how many passes the picture combines");
    check::is_true(a->details().contains(QStringLiteral("10.1387 MHz + 1500 Hz")) &&
                       !a->details().contains(QStringLiteral("RF ")),
                   "qrss/tiles: on the air as dial + audio offset (USB)");
    check::is_true(b->details().contains(QStringLiteral("· 1550 Hz")) &&
                       !b->details().contains(QStringLiteral("MHz")),
                   "qrss/tiles: and the offset alone when the dial is unknown");
    check::is_true(
        a->details().contains(QStringLiteral("Saved: matched to its picture by its header")),
        "qrss/tiles: how it was filed, in words");

    // A finished frame whose end-of-slot receive is running.
    {
        QJsonArray fin;
        QJsonObject fa = tile_json(QStringLiteral("a"), 1500.0, QStringLiteral("N0CALL"), 1.0,
                                   QStringLiteral("tiles/a.png"), 2);
        fa[QStringLiteral("status")] = QStringLiteral("finishing");
        fin.append(fa);
        QJsonObject fb = tile_json(QStringLiteral("b"), 1550.2, QString(), 1.0, QString(), 0);
        fb[QStringLiteral("status")] = QStringLiteral("finishing");
        fin.append(fb);
        write_state(dir.path(), fin, 1'500'000);
        w.reload();
        check::is_true(w.tile(QStringLiteral("a"))->details().contains(QStringLiteral("Finishing")),
                       "qrss/tiles: finishing is said so");
    }

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
        check::is_true(f.open(QIODevice::WriteOnly | QIODevice::Truncate),
                       "qrss/tiles: torn state file opened for writing");
        f.write("{\"tiles\": [");
        f.flush();
        f.setFileTime(QDateTime::fromMSecsSinceEpoch(3'000'000),
                      QFileDevice::FileModificationTime);
    }
    w.reload();
    check::equal(w.tile_count(), 1, "qrss/tiles: a torn state file changes nothing");
}

void test_the_confidence_band() {
    QTemporaryDir dir;
    gui::QrssWindow w;
    w.set_state_dir(dir.path());
    QJsonObject t = tile_json(QStringLiteral("c"), 1500.0, QStringLiteral("N0CALL"), 0.7,
                              QString(), 0);
    // The preamble (nothing measured), noise, 0 dB, strong, and a
    // stretch still to come.
    t[QStringLiteral("confidence")] = QJsonArray{QJsonValue(), -20.0, 0.0, 12.0, QJsonValue()};
    QJsonArray tiles;
    tiles.append(t);
    write_state(dir.path(), tiles, 5'000'000);
    w.reload();
    gui::QrssTile* tile = w.tile(QStringLiteral("c"));
    check::is_true(tile != nullptr && tile->pass_bar() != nullptr, "qrss/bar: every tile has one");
    if (tile == nullptr || tile->pass_bar() == nullptr) return;
    gui::PassBar* bar = tile->pass_bar();
    check::equal(bar->width(), tile->width() - 12, "qrss/bar: as wide as the picture");
    const QImage img = bar->grab().toImage();
    const double step = (bar->width() - 2) / 5.0;
    auto at = [&](int bin) { return img.pixelColor(1 + static_cast<int>((bin + 0.5) * step), 4); };
    check::is_true(at(0) == QColor(0, 0, 0),
                   "qrss/bar: past the preamble, nothing measured is black");
    check::is_true(at(1) == QColor(0, 0, 0), "qrss/bar: no better than noise is black");
    check::is_true(at(2) == gui::PassBar::color(0.0) && at(2).red() > 200 && at(2).green() < 30,
                   "qrss/bar: 0 dB per latent is red");
    check::is_true(at(3) == QColor(255, 255, 255), "qrss/bar: strong is white");
    check::is_true(at(4) == bar->palette().color(QPalette::Window),
                   "qrss/bar: what is still to come is not drawn");
    const QColor blue = gui::PassBar::color(-5.0);
    const QColor yellow = gui::PassBar::color(5.0);
    check::is_true(blue.blue() > 200 && blue.red() < 30 && yellow.red() > 200 &&
                       yellow.green() > 200 && yellow.blue() < 30,
                   "qrss/bar: blue low, yellow high, between red and white");
    check::is_true(gui::PassBar::color(-40.0) == QColor(0, 0, 0) &&
                       gui::PassBar::color(30.0) == QColor(255, 255, 255),
                   "qrss/bar: clamped at both ends");
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

    // SSTVAE mode: inactive, the listener goes and nothing is fed.
    w.set_active(false);
    check::is_true(wait_until([&] { return !w.listener_running(); }, 10000),
                   "qrss/feed: inactive (SSTVAE mode) stops the listener");

    QFile f(out);
    check::is_true(f.open(QIODevice::ReadOnly), "qrss/feed: the listener's input file opened");
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

// The main window's QRSSTVAE mode: the signals embedded rather than a
// window of their own, one Start/Stop that asks the receive pane, and the
// schedule's next send with the button that opens it.
void test_the_qrss_tab() {
    QWidget host;
    auto* signals_view = new gui::QrssWindow(nullptr, Qt::Widget);
    auto* page = new gui::QrssPage(signals_view, &host);
    host.resize(900, 600);
    host.show();
    QApplication::processEvents();
    check::is_true(!signals_view->isWindow() && signals_view->isVisible() &&
                       signals_view->parentWidget() == page->splitter(),
                   "tab: the QRSS signals are inside the tab, not a window");
    auto* listen = page->findChild<QPushButton*>(QStringLiteral("qrss_listen"));
    auto* schedule = page->findChild<QPushButton*>(QStringLiteral("qrss_schedule_button"));
    auto* summary = page->findChild<QLabel*>(QStringLiteral("qrss_schedule_summary"));
    check::is_true(listen && schedule && summary, "tab: Start receiving, the schedule, Schedule...");
    if (!(listen && schedule && summary)) return;
    std::vector<bool> asked;
    QObject::connect(page, &gui::QrssPage::listenRequested, [&asked](bool on) { asked.push_back(on); });
    int opened = 0;
    QObject::connect(page, &gui::QrssPage::scheduleRequested, [&opened] { ++opened; });
    listen->click();
    check::is_true(asked.size() == 1 && asked[0], "tab: Start asks to start receiving");
    check::is_true(listen->text().contains(QStringLiteral("Start")),
                   "tab: but shows receiving only once it is");
    page->set_listening(true);
    listen->click();
    check::is_true(asked.size() == 2 && !asked[1] && listen->text().contains(QStringLiteral("Stop")),
                   "tab: then Stop stops it");
    schedule->click();
    check::equal(opened, 1, "tab: Schedule... asks for the schedule");
    page->set_schedule_summary(QStringLiteral("Next send: x"));
    check::is_true(summary->text() == QStringLiteral("Next send: x"), "tab: and shows the next send");

    // One switch: the listener's own Start/Stop row and its log are
    // details, hidden until asked for.
    auto* details = page->findChild<QPushButton*>(QStringLiteral("qrss_details"));
    auto* row = signals_view->findChild<QWidget*>(QStringLiteral("qrss_listener_row"));
    auto* log = signals_view->findChild<QWidget*>(QStringLiteral("qrss_log"));
    check::is_true(details && row && log, "tab: a Listener details toggle, the row and the log");
    if (!(details && row && log)) return;
    check::is_true(!row->isVisible() && !log->isVisible() && !signals_view->details_visible(),
                   "tab: the listener's own controls are hidden by default");
    details->click();
    QApplication::processEvents();
    check::is_true(row->isVisible() && log->isVisible(), "tab: Listener details shows them");
    details->click();
    QApplication::processEvents();
    check::is_true(!row->isVisible() && !log->isVisible(), "tab: and hides them again");
}

// Headerless passes go in the Provisional column, and the "mode A
// assumed" note under them goes (the column says it); a header moves the
// tile across to Received.
void test_provisional_tiles() {
    QTemporaryDir dir;
    gui::QrssWindow w;
    w.set_state_dir(dir.path());
    QJsonArray tiles;
    tiles.append(tile_json(QStringLiteral("a"), 1500.0, QStringLiteral("N0CALL"), 0.4, QString(), 0));
    QJsonObject b = tile_json(QStringLiteral("b"), 1550.0, QString(), 0.3, QString(), 0);
    b[QStringLiteral("note")] =
        QStringLiteral("stored (provisional); no header yet: picture assumes the first pass of "
                       "a mode A send");
    tiles.append(b);
    write_state(dir.path(), tiles, 1'000'000);
    w.reload();
    gui::QrssTile* ta = w.tile(QStringLiteral("a"));
    gui::QrssTile* tb = w.tile(QStringLiteral("b"));
    check::is_true(ta && tb, "provisional: both tiles");
    if (!(ta && tb)) return;
    check::is_true(ta->parentWidget() == w.received_area() && !ta->provisional(),
                   "provisional: a tile with a header is Received");
    check::is_true(tb->parentWidget() == w.provisional_area() && tb->provisional(),
                   "provisional: a headerless one is Provisional");
    check::is_true(!tb->details().contains(QStringLiteral("mode A")) &&
                       tb->details().contains(
                           QStringLiteral("Saved, not matched to a picture yet")) &&
                       !tb->details().contains(QStringLiteral("stored")),
                   "provisional: the mode A note goes, the rest in plain words");
    QJsonArray later;
    later.append(tile_json(QStringLiteral("b"), 1550.0, QStringLiteral("K1ABC"), 0.6, QString(), 0));
    write_state(dir.path(), later, 2'000'000);
    w.reload();
    check::is_true(w.tile(QStringLiteral("b"))->parentWidget() == w.received_area(),
                   "provisional: a header moves it to Received");
}

// The pass bar: filled to the share of the pass so far, black where
// latents are missing (not heard or erased), none for the preamble or
// for what has not arrived yet.
void test_the_pass_bar() {
    gui::PassBar bar;
    std::vector<std::optional<double>> bins(120, std::optional<double>(3.0));
    bins[0] = std::nullopt;        // the preamble
    bins[10] = std::nullopt;       // not heard
    bins[20] = -30.0;              // heard, all erased
    bins[30] = -12.0;              // weak, but there
    for (int i = 60; i < 120; ++i) bins[static_cast<std::size_t>(i)] = std::nullopt;   // to come
    bar.set(bins, 0.5);
    const std::vector<bool>& h = bar.holes();
    check::is_true(!h[0] && h[10] && h[20] && !h[30] && !h[5],
                   "pass bar: holes where latents are missing, not the preamble or weak data");
    check::is_true(!h[70], "pass bar: what has not arrived yet is not a hole");
    bar.resize(240, 8);
    QImage img(bar.size(), QImage::Format_RGB32);
    bar.render(&img);
    const int x10 = 1 + static_cast<int>(10.5 * 238 / 120);
    const QRgb hole = img.pixel(x10, 4);
    check::is_true(hole == qRgb(0, 0, 0), "pass bar: missing latents are black");
    check::is_true(QColor(img.pixel(1, 4)) == gui::PassBar::unmeasured(),
                   "pass bar: the preamble is grey");
    check::is_true(img.pixel(1 + static_cast<int>(40.5 * 238 / 120), 4) != hole &&
                       img.pixel(200, 4) != img.pixel(1 + static_cast<int>(40.5 * 238 / 120), 4),
                   "pass bar: filled to the progress, empty after");
}

// One tile per picture: a later pass that combines earlier ones hides
// them; passes not combined into it stay.
void test_combined_passes_hide_their_parts() {
    QTemporaryDir dir;
    gui::QrssWindow w;
    w.set_state_dir(dir.path());
    const auto pass = [](const QString& id, const QString& slot, int passes, const QString& pid) {
        QJsonObject t = tile_json(id, 1204.0, QStringLiteral("KC2G"), 1.0, QString(), 0);
        t[QStringLiteral("slot_utc")] = slot;
        t[QStringLiteral("passes")] = passes;
        t[QStringLiteral("picture_id")] = pid;
        return t;
    };
    QJsonArray tiles;   // newest first, as the listener lists them
    tiles.append(pass(QStringLiteral("c"), QStringLiteral("2026-10-10T21:00:00Z"), 2,
                      QStringLiteral("cbc634ec")));
    tiles.append(pass(QStringLiteral("b"), QStringLiteral("2026-10-10T20:30:00Z"), 1,
                      QStringLiteral("cbc634ec")));
    tiles.append(pass(QStringLiteral("o"), QStringLiteral("2026-10-10T20:15:00Z"), 1,
                      QStringLiteral("0000beef")));
    write_state(dir.path(), tiles, 1'000'000);
    w.reload();
    gui::QrssTile* c = w.tile(QStringLiteral("c"));
    gui::QrssTile* b = w.tile(QStringLiteral("b"));
    gui::QrssTile* o = w.tile(QStringLiteral("o"));
    check::is_true(c && b && o, "combined: all three tiles known");
    if (!(c && b && o)) return;
    check::is_true(!c->isHidden() && b->isHidden(),
                   "combined: the two-pass tile shows, the pass it includes does not");
    check::is_true(!o->isHidden(), "combined: another picture's tile stays");
    // Two single passes of one picture, not combined: both stay.
    QJsonArray apart;
    apart.append(pass(QStringLiteral("c"), QStringLiteral("2026-10-10T21:00:00Z"), 1,
                      QStringLiteral("cbc634ec")));
    apart.append(pass(QStringLiteral("b"), QStringLiteral("2026-10-10T20:30:00Z"), 1,
                      QStringLiteral("cbc634ec")));
    write_state(dir.path(), apart, 2'000'000);
    w.reload();
    check::is_true(!w.tile(QStringLiteral("c"))->isHidden() &&
                       !w.tile(QStringLiteral("b"))->isHidden(),
                   "combined: passes not combined both show");
}

// The spectrogram: a tone shows where it is, darker on the default
// (inverse) scheme, red where a synced signal is, and the controls do
// what they say.
void test_the_spectrogram() {
    constexpr double NOW = 1791594000.0 + 600.0;
    gui::QrssSpectrogram sg;
    sg.set_clock([] { return NOW; });
    sg.resize(800, 500);
    sg.show();
    QApplication::processEvents();
    // Two minutes of a tone at 1500 Hz in quiet noise, ending now.
    {
        std::mt19937 rng(4);
        std::normal_distribution<double> noise(0.0, 0.05);
        std::vector<double> x(8000 * 120);
        for (std::size_t i = 0; i < x.size(); ++i) {
            x[i] = 0.05 * std::sin(2.0 * 3.14159265358979 * 1500.0 * i / 8000.0) + noise(rng);
        }
        sg.push_audio(x, NOW);
    }
    const QImage& img = sg.image();
    const QPoint on = sg.plot_point(NOW - 30, 1500.0);
    const QPoint off = sg.plot_point(NOW - 30, 1100.0);
    const QPoint old = sg.plot_point(NOW - 1000, 1500.0);
    check::is_true(img.rect().contains(on) && img.rect().contains(old),
                   "spectrogram: the points are on the plot");
    if (!img.rect().contains(on)) return;
    check::is_true(qGray(img.pixel(on)) < 100 && qGray(img.pixel(off)) > 180,
                   "spectrogram: inverse by default, the tone dark on a light ground");
    check::is_true(on.x() < 60, "spectrogram: the newest audio at the left");
    check::is_true(img.pixel(old) == img.pixel(sg.plot_point(NOW - 1000, 1100.0)),
                   "spectrogram: before the audio began, nothing drawn");

    // The sun/moon flips it.
    {
        const QPoint c = sg.theme_button_rect().center();
        QMouseEvent press(QEvent::MouseButtonPress, QPointF(c), QPointF(sg.mapToGlobal(c)),
                          Qt::LeftButton, Qt::LeftButton, Qt::NoModifier);
        QApplication::sendEvent(&sg, &press);
    }
    check::is_true(!sg.inverted() && qGray(sg.image().pixel(on)) > 150 &&
                       qGray(sg.image().pixel(off)) < 60,
                   "spectrogram: the sun/moon makes it light on dark");
    sg.set_inverted(true);

    // A synced signal at 1500 Hz from five minutes ago: red there.
    sg.set_markers({{1500.0, NOW - 300, NOW + 1500}});
    const QRgb r = sg.image().pixel(on);
    check::is_true(qRed(r) > qGreen(r) + 80 && qRed(r) > qBlue(r) + 80,
                   "spectrogram: a synced signal is red");
    check::is_true(qGray(sg.image().pixel(off)) > 180, "spectrogram: and only there");

    // Wheel over the time axis: the scale under the pointer.
    const auto wheel = [&sg](QPoint at, int delta) {
        QWheelEvent e(QPointF(at), QPointF(sg.mapToGlobal(at)), QPoint(), QPoint(0, delta),
                      Qt::NoButton, Qt::NoModifier, Qt::NoScrollPhase, false);
        QApplication::sendEvent(&sg, &e);
    };
    const QRect axis = sg.time_axis_rect();
    wheel(QPoint(axis.left() + 600, axis.center().y()), -120);
    check::is_true(std::abs(sg.scale().history_spp() - 15.0 * 1.25) < 1e-9 &&
                       sg.scale().lens_spp() == 1.0,
                   "spectrogram: wheel over the history changes the history's scale");
    wheel(QPoint(axis.left() + 50, axis.center().y()), 120);
    check::is_true(std::abs(sg.scale().lens_spp() - 1.0 / 1.25) < 1e-9,
                   "spectrogram: and over the lens, the lens's");
    const QRect faxis = sg.freq_axis_rect();
    wheel(QPoint(faxis.center().x(), faxis.center().y()), 120);
    check::is_true(sg.f_hi() - sg.f_lo() < 2400.0 - 1.0, "spectrogram: wheel over frequency zooms");
    {
        const QPoint c = faxis.center();
        QMouseEvent dbl(QEvent::MouseButtonDblClick, QPointF(c), QPointF(sg.mapToGlobal(c)),
                        Qt::LeftButton, Qt::LeftButton, Qt::NoModifier);
        QApplication::sendEvent(&sg, &dbl);
    }
    check::is_true(sg.f_lo() == gui::QrssSpectrogram::F_LO && sg.f_hi() == gui::QrssSpectrogram::F_HI,
                   "spectrogram: double-click goes back to the whole range");

    // Dragging the axes pans them, the grabbed point under the pointer.
    const auto mouse = [&sg](QEvent::Type type, QPoint at, Qt::MouseButtons held) {
        QMouseEvent e(type, QPointF(at), QPointF(sg.mapToGlobal(at)),
                      type == QEvent::MouseMove ? Qt::NoButton : Qt::LeftButton, held,
                      Qt::NoModifier);
        QApplication::sendEvent(&sg, &e);
    };
    const auto drag = [&mouse](QPoint from, QPoint to) {
        mouse(QEvent::MouseButtonPress, from, Qt::LeftButton);
        mouse(QEvent::MouseMove, to, Qt::LeftButton);
        mouse(QEvent::MouseButtonRelease, to, Qt::NoButton);
    };
    sg.set_frequency_range(1000.0, 1600.0);
    {
        const int h = sg.plot_rect().height();
        // 1300 Hz is half way up; dragged down a quarter of the plot it
        // shows 150 Hz higher.
        drag(QPoint(faxis.center().x(), sg.plot_rect().top() + h / 2),
             QPoint(faxis.center().x(), sg.plot_rect().top() + h / 2 + h / 4));
        check::is_true(std::abs(sg.f_lo() - 1150.0) < 3.0 && std::abs(sg.f_hi() - 1750.0) < 3.0,
                       "spectrogram: dragging the frequency axis down pans it up, span kept");
        // Dragged far, it stops at the top of the range.
        drag(QPoint(faxis.center().x(), sg.plot_rect().top() + 2),
             QPoint(faxis.center().x(), sg.plot_rect().top() + 20 * h));
        check::is_true(sg.f_hi() == dsp::SlowSpectrogram::MAX_HZ &&
                           std::abs(sg.f_hi() - sg.f_lo() - 600.0) < 1e-6,
                       "spectrogram: and stops at the top of the range");
    }
    sg.set_frequency_range(gui::QrssSpectrogram::F_LO, gui::QrssSpectrogram::F_HI);
    check::is_true(sg.live(), "spectrogram: live to start with");
    {
        // Grab a minute ago on the time axis and drag it to the left
        // edge: the view ends a minute ago, frozen there.
        const int y = axis.center().y();
        const int x60 = sg.plot_point(NOW - 60, 1500.0).x();
        drag(QPoint(axis.left() + x60, y), QPoint(axis.left(), y));
        check::is_true(!sg.live() && std::abs(sg.view_end() - (NOW - 60)) < 3.0,
                       "spectrogram: dragging the time axis left goes back in time");
        sg.image();   // plot_point reads the view as last drawn
        const QPoint tone = sg.plot_point(NOW - 90, 1500.0);
        check::is_true(tone.x() < 60 && qGray(sg.image().pixel(tone)) < 100,
                       "spectrogram: and the older audio is drawn at the left");
        // Dragging right past the present goes live again.
        drag(QPoint(axis.left() + 5, y), QPoint(axis.left() + 400, y));
        check::is_true(sg.live(), "spectrogram: dragging back past now goes live");
        drag(QPoint(axis.left() + x60, y), QPoint(axis.left(), y));
        const QPoint c(axis.left() + 300, y);
        QMouseEvent dbl(QEvent::MouseButtonDblClick, QPointF(c), QPointF(sg.mapToGlobal(c)),
                        Qt::LeftButton, Qt::LeftButton, Qt::NoModifier);
        QApplication::sendEvent(&sg, &dbl);
        check::is_true(sg.live(), "spectrogram: double-click on the time axis goes live");
        // No further back than the audio kept.
        sg.set_view_end(NOW - 86400.0);
        check::is_true(sg.view_end() > NOW - 200.0,
                       "spectrogram: no further back than the oldest audio");
        sg.go_live();
    }
}

// Each redraw recomputes the Time Lens; the history to its right is
// kept, block by block, and only moves when its grid does.
void test_the_history_is_kept() {
    double now = 1791594000.0 + 1200.0;
    gui::QrssSpectrogram sg;
    sg.set_clock([&now] { return now; });
    sg.scale().set_history_spp(2.0);   // a short history, so little audio
    sg.resize(800, 300);
    sg.show();
    QApplication::processEvents();
    {
        std::mt19937 rng(9);
        std::normal_distribution<double> noise(0.0, 0.05);
        std::vector<double> x(8000 * 900);
        for (std::size_t i = 0; i < x.size(); ++i) {
            x[i] = 0.05 * std::sin(2.0 * 3.14159265358979 * 1500.0 * i / 8000.0) + noise(rng);
        }
        sg.push_audio(x, now);
    }
    const int w = sg.image().width();
    check::equal(sg.last_computed_columns(), w, "history: the first redraw computes it all");
    const double t_old = now - 800.0;
    const QPoint p1 = sg.plot_point(t_old, 1500.0);
    const QRgb before = sg.image().pixel(p1);
    check::is_true(p1.x() > sg.scale().lens_px() && qGray(before) < 100,
                   "history: the tone shows in the history");
    now += 1.0;
    sg.pump();
    const QImage& after = sg.image();
    check::is_true(sg.last_computed_columns() <= sg.scale().lens_px() + 2,
                   "history: a second later only the Time Lens (and the newest block) is redone");
    const QPoint p2 = sg.plot_point(t_old, 1500.0);
    check::is_true(std::abs(p2.x() - p1.x()) <= 1 && after.pixel(p2) == before,
                   "history: and the kept history draws the same");
}

// The tab: the spectrogram and the received pane across a divider, and
// the tiles' passes reach the spectrogram as red.
void test_the_tab_layout() {
    QWidget host;
    auto* signals_view = new gui::QrssWindow(nullptr, Qt::Widget);
    auto* page = new gui::QrssPage(signals_view, &host);
    host.resize(1300, 700);
    host.show();
    QApplication::processEvents();
    check::is_true(page->splitter() && page->splitter()->count() == 2 &&
                       page->splitter()->widget(0) == page->spectrogram() &&
                       page->splitter()->widget(1) == signals_view,
                   "tab: spectrogram, then the received pane, across a divider");
    const QList<int> sizes = page->splitter()->sizes();
    check::is_true(sizes.size() == 2 && sizes[0] > sizes[1],
                   "tab: the spectrogram the larger to start with");
}

int main(int argc, char** argv) {
    check::report_crashes_instead_of_prompting();
    qputenv("QT_QPA_PLATFORM", "offscreen");
    const QApplication app(argc, argv);
    test_tiles_follow_the_state_file();
    test_the_confidence_band();
    test_audio_reaches_the_listener_once();
    test_the_qrss_tab();
    test_provisional_tiles();
    test_the_pass_bar();
    test_combined_passes_hide_their_parts();
    test_the_spectrogram();
    test_the_history_is_kept();
    test_the_tab_layout();
    return check::report("qrss window");
}

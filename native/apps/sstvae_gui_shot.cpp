// Render the application's windows to PNG files, headless.
//
// A GUI's layout bugs are not the kind a test catches. This does not
// try to: it just makes *looking* cheap, so a layout can be checked at
// several sizes without a display, a compositor, or a human. It found
// two things the day it was written -- help text clipped mid-sentence
// by a form layout with too little height, and every combo and spin box
// in the dialog rendering with zero left padding because a stylesheet
// somewhere else in the app had swapped the platform style out.
//
// A tool rather than a ctest, for the same reason `sstvae-audio-check`
// is: there is no assertion to make. "Is this laid out well" has no
// oracle, and a screenshot test that compares against a stored PNG
// fails on every font and theme it was not recorded with.
//
// Only widgets that can be built without touching the network or a
// radio are offered. `MainWindow` is deliberately absent: it starts a
// model load and opens the rig, neither of which belongs in a
// screenshot tool.

#include <QApplication>
#include <QPixmap>
#include <QStringList>
#include <QLayout>
#include <QLabel>
#include <QProgressBar>
#include <QTabWidget>

#include <cmath>
#include <cstdio>
#include <string>

#include "app_state.hpp"
#include "banner.hpp"
#include "crop_dialog.hpp"
#include "log/log.hpp"
#include "log_pane.hpp"
#include "main_window.hpp"
#include "pane_container.hpp"
#include "rx_panel.hpp"
#include "settings/settings.hpp"
#include "settings_dialog.hpp"
#include "tx_panel.hpp"

namespace {

void usage() {
    std::fprintf(stderr,
                 "usage: sstvae-gui-shot [--out DIR] [--size WxH] [--tab N]\n"
                 "\n"
                 "  --out DIR    where to write the PNGs (default: .)\n"
                 "  --size WxH   window size (default: the window's own)\n"
                 "  --tab N      only this settings tab; default is all\n"
                 "  --transmit   also shoot the transmit panel\n"
                 "  --receive    also shoot the receive panel\n"
                 "  --crop       also shoot the framing dialog\n"
                 "  --window     also shoot the whole main window\n"
                 "  --log        also shoot the log pane and error banner\n"
                 "  --panes      also shoot both pane layouts, and report the\n"
                 "               minimum width each one imposes on the window\n"
                 "\n"
                 "Writes settings-<n>-<name>.png, one per tab.\n");
}

// A configuration with the optional controls switched on, so the shots
// show the dialog at its fullest rather than at its emptiest.
sstvae::settings::Config demo_config() {
    sstvae::settings::Config config;
    config.callsign = "N0CALL";
    config.rig.enabled = true;
    config.rig.device = "/dev/ttyUSB0";
    config.rig.baud = 38400;
    // On, so the transmit shot exercises the refinement wiring as well
    // as showing the control in its non-default state.
    config.transmit.optimize = true;
    return config;
}

}  // namespace

int main(int argc, char** argv) {
    // Before QApplication: the platform plugin is chosen at construction.
    qputenv("QT_QPA_PLATFORM", "offscreen");
    QApplication app(argc, argv);

    QString out = QStringLiteral(".");
    int width = 0;
    int height = 0;
    int only_tab = -1;
    bool transmit = false;
    bool receive = false;
    bool crop = false;
    bool window = false;
    bool log_widgets = false;
    bool panes = false;

    const QStringList args = QCoreApplication::arguments();
    for (int i = 1; i < args.size(); ++i) {
        const QString& arg = args[i];
        if (arg == QLatin1String("--out") && i + 1 < args.size()) {
            out = args[++i];
        } else if (arg == QLatin1String("--size") && i + 1 < args.size()) {
            const QStringList parts = args[++i].split(QLatin1Char('x'));
            if (parts.size() != 2) {
                usage();
                return 2;
            }
            width = parts[0].toInt();
            height = parts[1].toInt();
        } else if (arg == QLatin1String("--tab") && i + 1 < args.size()) {
            only_tab = args[++i].toInt();
        } else if (arg == QLatin1String("--transmit")) {
            transmit = true;
        } else if (arg == QLatin1String("--receive")) {
            receive = true;
        } else if (arg == QLatin1String("--crop")) {
            crop = true;
        } else if (arg == QLatin1String("--window")) {
            window = true;
        } else if (arg == QLatin1String("--log")) {
            log_widgets = true;
        } else if (arg == QLatin1String("--panes")) {
            panes = true;
        } else {
            usage();
            return 2;
        }
    }

    const sstvae::settings::Config config = demo_config();
    sstvae::gui::SettingsDialog dialog(config);
    if (width > 0 && height > 0) dialog.resize(width, height);
    dialog.show();

    auto* tabs = dialog.findChild<QTabWidget*>();
    if (tabs == nullptr) {
        std::fprintf(stderr, "sstvae-gui-shot: no tab widget found\n");
        return 1;
    }

    // Opt-in, because unlike the settings dialog this one needs an
    // AppState -- which begins resolving the checkpoint as soon as it is
    // constructed, and on a cold cache that means an HTTP download. A
    // smoke test that proves Qt links should not also depend on the
    // network being up.
    if (transmit) {
        sstvae::gui::AppState state;
        sstvae::gui::TransmitPanel panel(&state);
        panel.resize(width > 0 ? width : 1000, height > 0 ? height : 700);
        // The banner *inside the panel's layout*, with the longest
        // message the app can produce -- a wrapped QLabel that clips
        // in a layout looks fine shot standalone, so the layout is
        // what has to be under the camera.
        QMetaObject::invokeMethod(
            &panel, "on_error", Qt::DirectConnection,
            Q_ARG(QString,
                  QStringLiteral("PTT OFF FAILED: read timeout — the rig "
                                 "may still be transmitting. Unkey it "
                                 "manually.")));
        panel.show();
        app.processEvents();
        const QString path = QStringLiteral("%1/transmit.png").arg(out);
        panel.grab().save(path);
        // The minimum is reported because it is a real constraint on
        // the window: the two panes sit in a splitter, whose minimum is
        // the *sum* of its children's, so a pane that quietly demands
        // 700 px sets the floor for the whole application.
        std::printf("%s (min %dx%d)\n", path.toLocal8Bit().constData(),
                    panel.minimumSizeHint().width(),
                    panel.minimumSizeHint().height());
    }

    // The receive pane at the width the dual-pane split gives it --
    // the shape the waterfall strip and the picture have to share.
    // `MainWindow` itself is still not a target here (it starts a model
    // load and opens the rig), so the *arrangement* of the two panes is
    // reviewed by reading; each pane's own layout is reviewed here.
    if (receive) {
        sstvae::gui::AppState state;
        sstvae::gui::ReceivePanel panel(&state);
        // Populated, not pristine: an empty pane hides exactly what has
        // to be looked at -- the card at its full length, the banner
        // inside the layout, and the longest status line the panel can
        // produce, each of which sets a width floor. The transmit shot
        // below injects its worst message for the same reason.
        panel.fill_for_screenshot();
        panel.resize(width > 0 ? width : 540, height > 0 ? height : 700);
        panel.show();
        app.processEvents();
        const QString path = QStringLiteral("%1/receive.png").arg(out);
        panel.grab().save(path);
        std::printf("%s (min %dx%d)\n", path.toLocal8Bit().constData(),
                    panel.minimumSizeHint().width(),
                    panel.minimumSizeHint().height());
    }

    // Both pane arrangements, with the *real* panels in them.
    //
    // This is the one target that reports a number worth acting on
    // rather than a picture worth looking at. The tabbed layout exists
    // because a `QSplitter`'s minimum width is the sum of its
    // children's where a `QTabWidget`'s is the max, and that claim is
    // only interesting at the sizes the actual panels demand --
    // `test_pane_container.cpp` proves the structure holds with toy
    // panes, but only these two say whether the saving is enough to
    // reach a real laptop panel. Re-run it whenever either panel grows
    // a control.
    if (panes) {
        sstvae::gui::AppState state;
        auto* rx = new sstvae::gui::ReceivePanel(&state);
        auto* tx = new sstvae::gui::TransmitPanel(&state);
        rx->fill_for_screenshot();
        sstvae::gui::PaneContainer container(rx, QStringLiteral("Receive"), tx,
                                             QStringLiteral("Transmit"));
        container.set_control_strips(rx->control_strip(), tx->control_strip());
        container.resize(width > 0 ? width : 1360, height > 0 ? height : 760);
        container.show();
        app.processEvents();

        // The strips are what make the pictures equal, so their heights
        // belong next to the pictures': when the two differ, this says
        // by how much and which way, instead of leaving it to be
        // inferred from the picture sizes.
        std::printf("strips:   receive %d  transmit %d%s\n",
                    rx->control_strip()->height(), tx->control_strip()->height(),
                    rx->control_strip()->height() == tx->control_strip()->height()
                        ? "  (equal)" : "  <-- NOT EQUAL");
        std::printf("pictures: receive %dx%d  transmit %dx%d\n",
                    rx->picture_area()->width(), rx->picture_area()->height(),
                    tx->picture_area()->width(), tx->picture_area()->height());
        const QString split_path = QStringLiteral("%1/panes-split.png").arg(out);
        container.grab().save(split_path);
        const QSize split_min = container.minimumSizeHint();
        std::printf("%s (min %dx%d)\n", split_path.toLocal8Bit().constData(),
                    split_min.width(), split_min.height());

        container.set_mode(sstvae::gui::PaneLayout::Tabs);
        app.processEvents();
        const QString tabs_path = QStringLiteral("%1/panes-tabs.png").arg(out);
        container.grab().save(tabs_path);
        const QSize tabs_min = container.minimumSizeHint();
        // **Both axes.** Reporting only the width measures the axis this
        // layout improves and stays silent on the one it can wreck: a
        // pane that pins its height to its width (see `picture_box.hpp`)
        // does its worst damage exactly when a tab hands it the whole
        // window, which is the case a width-only number cannot see.
        std::printf("%s (min %dx%d; %d px narrower, %d px %s)\n",
                    tabs_path.toLocal8Bit().constData(), tabs_min.width(),
                    tabs_min.height(), split_min.width() - tabs_min.width(),
                    std::abs(split_min.height() - tabs_min.height()),
                    tabs_min.height() > split_min.height() ? "TALLER" : "shorter");

        // **The third number, and the one that actually broke a
        // window.** A minimum size *hint* never consults
        // `heightForWidth`; what Qt applies when it lays a widget out
        // is `minimumHeightForWidth(width)`. A widget asking to be 4:3
        // through that shows up here and nowhere else -- and a
        // QSplitter hides it while a QTabWidget passes it to the
        // window, which is how the tabbed layout grew past the bottom
        // of the screen.
        //
        // **This is not expected to read zero**, and an earlier version
        // of this comment said it was, which would have had the next
        // person chasing a ratchet that is not there. Wrapped labels
        // legitimately have a height that depends on width, and
        // `QWidgetItem` reports a *preferred* heightForWidth where no
        // minimum is defined. What matters is that the number does not
        // grow with width -- that is the ratchet -- and that the round
        // trip below leaves the window the size it was.
        for (const auto mode : {sstvae::gui::PaneLayout::Split,
                                sstvae::gui::PaneLayout::Tabs}) {
            container.set_mode(mode);
            for (int k = 0; k < 3; ++k) {
                container.setGeometry(0, 0, container.width(), container.height());
                app.processEvents();
            }
            const int hfw = container.layout()->hasHeightForWidth()
                                ? container.layout()->minimumHeightForWidth(container.width())
                                : 0;
            std::printf("%s: minimumHeightForWidth(%d) = %d%s\n",
                        mode == sstvae::gui::PaneLayout::Split ? "split" : "tabs ",
                        container.width(), hfw,
                        hfw > container.height() ? "   <-- forces the window taller" : "");
        }
    }

    // The whole window. Opt-in and last, because unlike everything
    // else here it constructs a MainWindow, which starts a model load
    // and opens the rig -- so it wants a warm model cache and rig
    // control off, and a failure puts a modal on screen that a
    // screenshot run cannot dismiss. Worth having anyway: the thing
    // being decided *is* the window, and whether it fits a given screen
    // height cannot be answered by shooting the panes separately.
    if (window) {
        sstvae::gui::MainWindow win;
        win.resize(width > 0 ? width : 1360, height > 0 ? height : 900);
        win.show();
        for (int i = 0; i < 40; ++i) app.processEvents();
        const QString path = QStringLiteral("%1/window.png").arg(out);
        win.grab().save(path);
        std::printf("%s (min %dx%d)\n", path.toLocal8Bit().constData(),
                    win.minimumSizeHint().width(), win.minimumSizeHint().height());
        // **The regression this window actually had**: switching layout
        // grew it past the bottom of the screen and switching back did
        // not shrink it again, because Qt lowers a minimum without
        // resizing. Toggle twice and the size must come back unchanged.
        {
            auto* rxp = win.findChild<sstvae::gui::ReceivePanel*>();
            auto* txp = win.findChild<sstvae::gui::TransmitPanel*>();
            if (rxp != nullptr && txp != nullptr) {
                const QSize a = rxp->picture_area()->size();
                const QSize b = txp->picture_area()->size();
                std::printf("  pictures: receive %dx%d  transmit %dx%d%s\n",
                            a.width(), a.height(), b.width(), b.height(),
                            a == b ? "  (equal)" : "  <-- NOT EQUAL");
            }
        }
        auto* panes = win.findChild<sstvae::gui::PaneContainer*>();
        if (panes != nullptr) {
            const QSize before = win.size();
            for (const auto m : {sstvae::gui::PaneLayout::Tabs,
                                 sstvae::gui::PaneLayout::Split}) {
                panes->set_mode(m);
                for (int i = 0; i < 20; ++i) app.processEvents();
                std::printf("  after %-5s window %dx%d\n",
                            m == sstvae::gui::PaneLayout::Tabs ? "tabs" : "split",
                            win.width(), win.height());
            }
            std::printf("  %s\n", win.size() == before
                                       ? "round trip: size unchanged"
                                       : "ROUND TRIP CHANGED THE WINDOW SIZE");
        }
    }

    // The framing dialog, on a 16:9 source -- the case it exists for,
    // where a quarter of the width is being given up and the dimmed
    // region is what the operator is deciding about.
    if (crop) {
        constexpr int SW = 1600;
        constexpr int SH = 900;
        sstvae::images::Picture source(SW, SH);
        for (int y = 0; y < SH; ++y) {
            for (int x = 0; x < SW; ++x) {
                const std::size_t i = (static_cast<std::size_t>(y) * SW + x) * 3;
                // A coarse checker plus a gradient: structured enough
                // that the crop window's edges are obvious, and not so
                // busy that the dimming is hard to read.
                const bool check = ((x / 100) + (y / 100)) % 2 == 0;
                source.rgb[i] = static_cast<unsigned char>(check ? 210 : 60);
                source.rgb[i + 1] = static_cast<unsigned char>(x * 255 / SW);
                source.rgb[i + 2] = static_cast<unsigned char>(y * 255 / SH);
            }
        }
        // Both ends of the zoom: the crop the dialog was written for,
        // and the fully zoomed-out framing, where the window overhangs
        // the picture and the overhang is the black that goes on the
        // air. The second is the one worth eyes -- padding drawn as the
        // dialog's background instead of black would look plausible and
        // be a lie about what is transmitted.
        sstvae::images::Framing framings[2];
        framings[1].zoom = sstvae::images::min_zoom(SW, SH);
        const char* names[2] = {"crop", "crop-zoomed-out"};
        for (int i = 0; i < 2; ++i) {
            sstvae::gui::CropDialog dialog(source, framings[i]);
            dialog.resize(width > 0 ? width : 640, height > 0 ? height : 560);
            dialog.show();
            app.processEvents();
            const QString path = QStringLiteral("%1/%2.png")
                                     .arg(out, QLatin1String(names[i]));
            dialog.grab().save(path);
            std::printf("%s (min %dx%d)\n", path.toLocal8Bit().constData(),
                        dialog.minimumSizeHint().width(),
                        dialog.minimumSizeHint().height());
        }
    }

    // The observability widgets, with representative content: the pane
    // filled from a preloaded log, and the banner showing the message
    // whose survivability is the whole reason it exists.
    if (log_widgets) {
        sstvae::log::StatusLog log;
        log.append("app", sstvae::log::Severity::Warning,
                   "config: receive.decode_every: not a number; using default");
        log.append("rig", sstvae::log::Severity::Info,
                   "Rig: 14.2300 MHz (Elecraft K4)");
        log.append("rx", sstvae::log::Severity::Info,
                   "sync acquired: mode B de KD8XYZ");
        log.append("rx", sstvae::log::Severity::Info,
                   "reception complete: mode B de KD8XYZ, 32/32 frames, SNR "
                   "8.3 dB — saved /home/op/Pictures/rx/KD8XYZ-B-14230.png");
        log.append("tx", sstvae::log::Severity::Info, "keying rig");
        log.append("tx", sstvae::log::Severity::Error,
                   "PTT OFF FAILED: read timeout — the rig may still be "
                   "transmitting. Unkey it manually.");
        sstvae::gui::LogPane pane(&log);
        pane.resize(width > 0 ? width : 1000, 180);
        pane.show();
        app.processEvents();
        const QString pane_path = QStringLiteral("%1/log-pane.png").arg(out);
        pane.grab().save(pane_path);
        std::printf("%s\n", pane_path.toLocal8Bit().constData());

        sstvae::gui::ErrorBanner banner;
        banner.show_error(QStringLiteral(
            "PTT OFF FAILED: read timeout — the rig may still be "
            "transmitting. Unkey it manually."));
        banner.resize(width > 0 ? width : 1000, banner.sizeHint().height());
        banner.show();
        app.processEvents();
        const QString banner_path = QStringLiteral("%1/banner.png").arg(out);
        banner.grab().save(banner_path);
        std::printf("%s\n", banner_path.toLocal8Bit().constData());
    }

    for (int i = 0; i < tabs->count(); ++i) {
        if (only_tab >= 0 && i != only_tab) continue;
        tabs->setCurrentIndex(i);
        // Let the layout settle before grabbing: a tab that has just
        // been shown has not been laid out yet, and the shot would be of
        // the previous one's geometry.
        app.processEvents();
        const QString name = tabs->tabText(i).toLower().replace(QLatin1Char(' '),
                                                               QLatin1Char('-'));
        const QString path =
            QStringLiteral("%1/settings-%2-%3.png").arg(out).arg(i).arg(name);
        if (!dialog.grab().save(path)) {
            std::fprintf(stderr, "sstvae-gui-shot: could not write %s\n",
                         path.toLocal8Bit().constData());
            return 1;
        }
        std::printf("%s\n", path.toLocal8Bit().constData());
    }
    return 0;
}

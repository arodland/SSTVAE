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
#include <QTabWidget>

#include <cstdio>
#include <string>

#include "app_state.hpp"
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

    // The transmit panel, which needs an AppState but touches neither
    // the network nor the radio until Send is pressed.
    {
        sstvae::gui::AppState state;
        sstvae::gui::TransmitPanel panel(&state);
        panel.resize(width > 0 ? width : 1000, height > 0 ? height : 700);
        panel.show();
        app.processEvents();
        const QString path = QStringLiteral("%1/transmit.png").arg(out);
        panel.grab().save(path);
        std::printf("%s\n", path.toLocal8Bit().constData());
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

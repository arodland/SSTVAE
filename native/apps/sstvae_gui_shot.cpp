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
#include <QComboBox>
#include <QDateTime>
#include <QDir>
#include <QFile>
#include <QImage>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QTimeZone>
#include <QPixmap>
#include <QStringList>
#include <QLayout>
#include <QAction>
#include <QLabel>
#include <QMenu>
#include <QPoint>
#include <QProgressBar>
#include <QTabWidget>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <numbers>
#include <random>
#include <optional>
#include <string>
#include <vector>

#include "app_state.hpp"
#include "banner.hpp"
#include "crop_dialog.hpp"
#include "images/types.hpp"
#include "item_menu.hpp"
#include "qrss_schedule.hpp"
#include "qrss_editor.hpp"
#include "qrss_page.hpp"
#include "qrss_spectrogram.hpp"
#include "info_button.hpp"
#include "qrss_schedule_window.hpp"
#include "qrss_sends.hpp"
#include "qrss_timeline.hpp"
#include "images/images.hpp"
#include "qrss_window.hpp"
#include "log/log.hpp"
#include "log_pane.hpp"
#include "main_window.hpp"
#include "overlay_editor.hpp"
#include "pane_container.hpp"
#include "rx_panel.hpp"
#include "settings/settings.hpp"
#include "settings_dialog.hpp"
#include "share_dialog.hpp"
#include "overlay/template_catalog.hpp"
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
                 "  --qrss-tab   also shoot the QRSSTVAE mode on two hours of made-up\n"
                 "               signals\n"
                 "  --window     also shoot the whole main window\n"
                 "  --log        also shoot the log pane and error banner\n"
                 "  --panes      also shoot both pane layouts, and report the\n"
                 "               minimum width each one imposes on the window\n"
                 "  --share      also shoot the template share window\n"
                 "  --item-menu  also shoot the composer's right-click menu\n"
                 "               and its submenus, for a text item and a rect\n"
                 "  --qrss DIR   also shoot the QRSS signals window showing the\n"
                 "               listener tiles in DIR (qrss_listen.py --state)\n"
                 "  --qrss-schedule DIR  also shoot the QRSS schedule window over\n"
                 "               DIR/qrss_schedule.json and DIR/qrss_sends (demo\n"
                 "               Sends and sends if empty), its menu and the Editor\n"
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
    bool share = false;
    bool receive = false;
    bool crop = false;
    bool qrss_tab = false;
    bool window = false;
    bool log_widgets = false;
    bool panes = false;
    bool item_menu = false;
    QString qrss_dir;
    QString schedule_dir;

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
        } else if (arg == QLatin1String("--qrss-tab")) {
            qrss_tab = true;
        } else if (arg == QLatin1String("--crop")) {
            crop = true;
        } else if (arg == QLatin1String("--window")) {
            window = true;
        } else if (arg == QLatin1String("--log")) {
            log_widgets = true;
        } else if (arg == QLatin1String("--panes")) {
            panes = true;
        } else if (arg == QLatin1String("--share")) {
            share = true;
        } else if (arg == QLatin1String("--item-menu")) {
            item_menu = true;
        } else if (arg == QLatin1String("--qrss") && i + 1 < args.size()) {
            qrss_dir = args[++i];
        } else if (arg == QLatin1String("--qrss-schedule") && i + 1 < args.size()) {
            schedule_dir = args[++i];
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

    // **A popup is invisible to a normal widget render**, which is why
    // this is its own option: the item menu is the one surface of the
    // composer that `--transmit` cannot show, because it is not in any
    // layout. Unlike the transmit panel it needs no `AppState` -- the
    // menu's only collaborator is the editor -- so it cannot touch the
    // network.
    // The one window whose content is generated rather than laid out --
    // a code that is too dense or too small is not something a test can
    // tell you. Needs no `AppState`: it is handed a document.
    if (share) {
        sstvae::overlay::Doc doc =
            sstvae::overlay::load_builtin_templates(
                (QCoreApplication::applicationDirPath() + QStringLiteral("/templates"))
                    .toStdString())
                .at(2);
        sstvae::gui::ShareDialog dialog(doc);
        dialog.resize(width > 0 ? width : 520, height > 0 ? height : 700);
        dialog.show();
        app.processEvents();
        const QString path = QStringLiteral("%1/share.png").arg(out);
        dialog.grab().save(path);
        std::printf("%s\n", path.toLocal8Bit().constData());
    }

    // Reads tiles a listener already wrote; starts no listener.
    if (!qrss_dir.isEmpty()) {
        sstvae::gui::QrssWindow qrss;
        qrss.set_state_dir(qrss_dir);
        qrss.resize(width > 0 ? width : 820, height > 0 ? height : 640);
        qrss.show();
        app.processEvents();
        const QString path = QStringLiteral("%1/qrss.png").arg(out);
        qrss.grab().save(path);
        std::printf("%s (%d tiles)\n", path.toLocal8Bit().constData(), qrss.tile_count());
    }

    // Sends nothing: the scheduler's timer is off and its transmitter
    // refuses.
    if (!schedule_dir.isEmpty()) {
        namespace qs = sstvae::gui::qrss_schedule;
        sstvae::gui::QrssScheduler scheduler(
            std::filesystem::path(schedule_dir.toStdString()) / "qrss_schedule.json",
            [](const qs::Entry&, const std::vector<sstvae::gui::qrss_tx::Pass>&, std::string* why) {
                *why = "a screenshot sends nothing";
                return sstvae::gui::QrssScheduler::Start::Failed;
            },
            {}, 0);
        // Three Sends, each a coloured gradient with a caption, made
        // through the Editor itself so its save path is what is shot.
        sstvae::gui::QrssSends sends(std::filesystem::path(schedule_dir.toStdString()) /
                                     "qrss_sends");
        const auto gradient = [](int r, int g, int b) {
            sstvae::images::Picture pic(sstvae::overlay::CANVAS_W, sstvae::overlay::CANVAS_H);
            for (int y = 0; y < pic.height; ++y) {
                for (int x = 0; x < pic.width; ++x) {
                    const std::size_t i = (static_cast<std::size_t>(y) * pic.width + x) * 3;
                    const int k = 40 + x * 180 / pic.width;
                    pic.rgb[i] = static_cast<std::uint8_t>(k * r / 255);
                    pic.rgb[i + 1] = static_cast<std::uint8_t>(k * g / 255);
                    pic.rgb[i + 2] = static_cast<std::uint8_t>(k * b / 255);
                }
            }
            return pic;
        };
        sstvae::gui::QrssEditor::Context context;
        context.default_text = config.callsign;
        sstvae::gui::QrssEditor editor(&sends, context);
        editor.resize(width > 0 ? width : 900, height > 0 ? height : 640);
        std::vector<std::string> ids;
        if (sends.sends().empty()) {
            const struct { const char* label; const char* text; int r, g, b; } demo[] = {
                {"Sunset", "AG7EW CN85", 255, 150, 60},
                {"Antenna farm", "630 m test", 80, 200, 120},
                {"CQ card", "CQ QRSS", 90, 140, 255},
            };
            for (const auto& d : demo) {
                editor.start_from(gradient(d.r, d.g, d.b), {}, d.label);
                editor.editor()->add_text(d.text);
                ids.push_back(editor.save());
            }
        } else {
            for (const auto& s : sends.sends()) ids.push_back(s.id);
        }
        if (scheduler.entries().empty() && ids.size() >= 2) {
            const double first = qs::earliest_slot(scheduler.now()) + 3600.0;
            const auto picture_of = [&](const std::string& id) {
                auto p = sstvae::images::load(sends.find(id)->picture.string());
                return p;
            };
            qs::Entry aa;
            aa.label = sends.find(ids[0])->label;
            aa.send_id = ids[0];
            aa.first_slot = first;
            aa.count = 2;
            scheduler.add(aa, picture_of(ids[0]));
            qs::Entry b;
            b.label = sends.find(ids[1])->label;
            b.send_id = ids[1];
            b.mode = "B";
            b.freq_hz = 1650.0;
            b.first_slot = first + 3 * 3600.0;
            b.count = 0;
            b.every_min = 240;
            scheduler.add(b, picture_of(ids[1]));
            qs::FreqChange c;
            c.at = first + 2 * 3600.0 + 120.0;
            c.dial_hz = 7'038'600.0;
            scheduler.add_change(c);
        }
        sstvae::gui::QrssScheduleWindow w(
            &scheduler, &sends,
            [] { return sstvae::gui::QrssScheduleWindow::Defaults{"A", 1500.0}; }, context);
        w.resize(width > 0 ? width : 1000, height > 0 ? height : 720);
        w.show();
        if (!ids.empty()) w.select_send(ids[0]);
        app.processEvents();
        const QString path = QStringLiteral("%1/qrss-schedule.png").arg(out);
        w.grab().save(path);
        std::printf("%s (%zu scheduled, %zu sends)\n", path.toLocal8Bit().constData(),
                    scheduler.entries().size(), sends.sends().size());

        // The (i) cards beside Mode and Waveform, each for both choices.
        for (const char* name : {"schedule_waveform_info", "schedule_mode_info"}) {
            auto* info = w.findChild<sstvae::gui::InfoButton*>(QString::fromLatin1(name));
            auto* combo = w.findChild<QComboBox*>(QString::fromLatin1(
                std::string(name) == "schedule_waveform_info" ? "schedule_waveform" : "schedule_mode"));
            if (info == nullptr || combo == nullptr) continue;
            for (int i = 0; i < combo->count(); ++i) {
                combo->setCurrentIndex(i);
                info->show_card();
                app.processEvents();
                const QString ipath = QStringLiteral("%1/%2-%3.png")
                                          .arg(out, QString::fromLatin1(name), combo->currentData().toString());
                if (info->card() != nullptr) info->card()->grab().save(ipath);
                std::printf("%s\n", ipath.toLocal8Bit().constData());
                info->hide_card();
            }
            combo->setCurrentIndex(0);
        }

        // The right-click menu on the first block, and its two submenus.
        const auto& blocks = w.timeline()->blocks();
        if (!blocks.empty()) {
            QMenu* menu = w.menu_for(blocks.front().entry, blocks.front().send);
            menu->popup(QPoint(80, 80));
            app.processEvents();
            const QString mpath = QStringLiteral("%1/qrss-schedule-menu.png").arg(out);
            menu->grab().save(mpath);
            std::printf("%s\n", mpath.toLocal8Bit().constData());
            for (const char* name : {"menu_change_mode", "menu_repeat_in"}) {
                if (auto* sub = menu->findChild<QMenu*>(QString::fromLatin1(name))) {
                    sub->popup(QPoint(260, 80));
                    app.processEvents();
                    const QString spath =
                        QStringLiteral("%1/qrss-schedule-%2.png").arg(out, QString::fromLatin1(name));
                    sub->grab().save(spath);
                    std::printf("%s\n", spath.toLocal8Bit().constData());
                    sub->hide();
                }
            }
            menu->hide();
            delete menu;
        }

        // The Editor, open on the first Send.
        if (!ids.empty() && editor.open_send(ids[0])) {
            editor.show();
            app.processEvents();
            const QString epath = QStringLiteral("%1/qrss-editor.png").arg(out);
            editor.grab().save(epath);
            std::printf("%s\n", epath.toLocal8Bit().constData());
            editor.hide();
        }
    }

    if (item_menu) {
        sstvae::gui::OverlayEditor editor;
        editor.resize(width > 0 ? width : 640, height > 0 ? height : 480);
        sstvae::images::Picture base(sstvae::overlay::CANVAS_W, sstvae::overlay::CANVAS_H);
        std::fill(base.rgb.begin(), base.rgb.end(), std::uint8_t{96});
        editor.set_base_image(base);
        sstvae::gui::ItemMenu menu(&editor);

        // Each submenu separately: they are what the menu actually is,
        // and a shot of three entries says nothing about the rows of
        // controls behind them. Style differs by item, so it is shot for
        // both kinds that have one.
        const auto shoot = [&](const char* kind) {
            menu.popup_for(editor.selected_item(), QPoint(80, 80));
            app.processEvents();
            const QString root = QStringLiteral("%1/item-menu-%2.png")
                                     .arg(out, QString::fromLatin1(kind));
            menu.grab().save(root);
            std::printf("%s\n", root.toLocal8Bit().constData());
            for (const char* name : {"menu_format", "menu_style", "menu_layers"}) {
                auto* sub = menu.findChild<QMenu*>(QString::fromLatin1(name));
                if (sub == nullptr || !sub->menuAction()->isVisible()) continue;
                sub->popup(QPoint(200, 200));
                app.processEvents();
                const QString path = QStringLiteral("%1/item-menu-%2-%3.png")
                                         .arg(out, QString::fromLatin1(kind),
                                              QString::fromLatin1(name + 5));
                sub->grab().save(path);
                std::printf("%s (%dx%d)\n", path.toLocal8Bit().constData(),
                            sub->sizeHint().width(), sub->sizeHint().height());
                sub->close();
            }
            menu.close();
        };
        editor.add_text("N0CALL");
        shoot("text");
        editor.add_rect();
        shoot("rect");
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
        // The QRSSTVAE mode.
        {
            using Mode = sstvae::gui::MainWindow::UiMode;
            const Mode was = win.ui_mode();
            win.set_ui_mode(Mode::Qrsstvae);
            for (int i = 0; i < 10; ++i) app.processEvents();
            const QString qpath = QStringLiteral("%1/window-qrss.png").arg(out);
            win.grab().save(qpath);
            std::printf("%s\n", qpath.toLocal8Bit().constData());
            win.set_ui_mode(Mode::Sstvae);
            for (int i = 0; i < 10; ++i) app.processEvents();
            win.set_ui_mode(was);
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
    // The QRSSTVAE mode, on made-up signals: two hours of noise with a
    // CE-like signal (a carrier wandering over 50 Hz) on two passes at
    // 1500 Hz, a weaker one at 1100 Hz, and a drifting plain carrier;
    // and tiles for the 1500 Hz passes (one with a header, one without).
    if (qrss_tab) {
        const double now = std::floor(QDateTime::currentSecsSinceEpoch() / 900.0) * 900.0 + 500.0;
        const QString state = QStringLiteral("%1/qrss-tab-state").arg(out);
        QDir().mkpath(state + QStringLiteral("/tiles"));
        {
            QImage pic(320, 240, QImage::Format_RGB32);
            for (int y = 0; y < 240; ++y) {
                for (int x = 0; x < 320; ++x) pic.setPixel(x, y, qRgb(40 + x / 2, 90 + y / 3, 160));
            }
            pic.save(state + QStringLiteral("/tiles/a.png"));
        }
        const auto iso = [](double t) {
            return QDateTime::fromSecsSinceEpoch(static_cast<qint64>(t), QTimeZone::utc())
                .toString(Qt::ISODate);
        };
        QJsonArray tiles;
        {
            QJsonObject a;
            a[QStringLiteral("id")] = QStringLiteral("a");
            a[QStringLiteral("slot_utc")] = iso(now - 500.0 - 1800.0);
            a[QStringLiteral("f_hz")] = 1500.0;
            a[QStringLiteral("status")] = QStringLiteral("complete");
            a[QStringLiteral("snr_db")] = -18.4;
            a[QStringLiteral("callsign")] = QStringLiteral("AG7EW");
            a[QStringLiteral("grid")] = QStringLiteral("CN85");
            a[QStringLiteral("picture_id")] = QStringLiteral("98f0da1e");
            a[QStringLiteral("mode")] = QStringLiteral("A");
            a[QStringLiteral("passes")] = 2;
            a[QStringLiteral("progress")] = 1.0;
            a[QStringLiteral("heard")] = 1.0;
            a[QStringLiteral("received")] = 0.96;
            a[QStringLiteral("image")] = QStringLiteral("tiles/a.png");
            a[QStringLiteral("image_rev")] = 1;
            tiles.append(a);
            QJsonObject b;
            b[QStringLiteral("id")] = QStringLiteral("b");
            b[QStringLiteral("slot_utc")] = iso(now - 500.0);
            b[QStringLiteral("f_hz")] = 1100.0;
            b[QStringLiteral("status")] = QStringLiteral("receiving");
            b[QStringLiteral("snr_db")] = -24.0;
            b[QStringLiteral("progress")] = 0.28;
            b[QStringLiteral("heard")] = 0.28;
            b[QStringLiteral("received")] = 0.2;
            b[QStringLiteral("note")] = QStringLiteral(
                "no header yet: picture assumes the first pass of a mode A send");
            tiles.append(b);
        }
        {
            QJsonObject st;
            st[QStringLiteral("version")] = 1;
            st[QStringLiteral("listening")] = true;
            st[QStringLiteral("source")] = QStringLiteral("app");
            st[QStringLiteral("tiles")] = tiles;
            QFile f(state + QStringLiteral("/state.json"));
            if (f.open(QIODevice::WriteOnly | QIODevice::Truncate)) {
                f.write(QJsonDocument(st).toJson());
                f.close();
            }
        }
        auto* qw = new sstvae::gui::QrssWindow(nullptr, Qt::Widget);
        qw->set_state_dir(state);
        sstvae::gui::QrssPage page(qw);
        page.spectrogram()->set_clock([now] { return now; });
        page.spectrogram()->set_dial_marks({now - 5400.0, now - 1800.0});
        page.set_schedule_summary(QStringLiteral("Next send: \"Sunset\", mode A at 17:45Z."));
        // The audio, a minute at a time.
        std::mt19937 rng(7);
        std::normal_distribution<double> noise(0.0, 1.0);
        std::uniform_real_distribution<double> wander(-25.0, 25.0);
        constexpr double SPAN = 2 * 3600.0;
        double phase_a = 0;
        double phase_b = 0;
        double phase_c = 0;
        double fa = 0;
        double fb = 0;
        for (double t0 = now - SPAN; t0 < now; t0 += 60.0) {
            std::vector<double> x(8000 * 60);
            for (std::size_t i = 0; i < x.size(); ++i) {
                const double t = t0 + static_cast<double>(i) / 8000.0;
                if (i % 160 == 0) {
                    fa = wander(rng);
                    fb = wander(rng);
                }
                double v = noise(rng);
                const double slot_a = std::fmod(t - (now - 500.0 - 1800.0) + 7200.0, 3600.0);
                if (t > now - 500.0 - 5400.0 && slot_a >= 0 && slot_a < 1783.0) {
                    phase_a += 2 * std::numbers::pi * (1500.0 + fa) / 8000.0;
                    v += 0.25 * std::cos(phase_a);
                }
                if (t > now - 500.0) {
                    phase_b += 2 * std::numbers::pi * (1100.0 + fb) / 8000.0;
                    v += 0.12 * std::cos(phase_b);
                }
                const double fc = 1850.0 + 6.0 * std::sin(2 * std::numbers::pi * t / 2400.0);
                phase_c += 2 * std::numbers::pi * fc / 8000.0;
                v += 0.05 * std::cos(phase_c);
                x[i] = 0.1 * v;
            }
            page.spectrogram()->push_audio(x, t0 + 60.0);
        }
        page.resize(width > 0 ? width : 1400, height > 0 ? height : 720);
        page.show();
        qw->reload();
        for (int i = 0; i < 5; ++i) app.processEvents();
        const QString path = QStringLiteral("%1/qrss-tab.png").arg(out);
        page.grab().save(path);
        std::printf("%s\n", path.toLocal8Bit().constData());
        page.spectrogram()->set_inverted(false);
        app.processEvents();
        const QString dark = QStringLiteral("%1/qrss-tab-dark.png").arg(out);
        page.spectrogram()->grab().save(dark);
        std::printf("%s\n", dark.toLocal8Bit().constData());
    }

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

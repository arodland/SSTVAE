#include "main_window.hpp"

#include <QAction>
#include <QCloseEvent>
#include <QKeySequence>
#include <QLabel>
#include <QMenu>
#include <QMenuBar>
#include <QMessageBox>
#include <QStatusBar>
#include <QTabWidget>
#include <QWidget>

#include "app_state.hpp"
#include "rx_panel.hpp"
#include "tx_panel.hpp"
#include "settings_dialog.hpp"

namespace sstvae::gui {

namespace {

constexpr auto APP_NAME = "SSTVAE";

}  // namespace

MainWindow::MainWindow(QWidget* parent) : QMainWindow(parent) {
    state_ = new AppState(this);
    setWindowTitle(QString::fromLatin1(APP_NAME));
    resize(1100, 800);

    tabs_ = new QTabWidget(this);
    rx_panel_ = new ReceivePanel(state_, tabs_);
    tabs_->addTab(rx_panel_, tr("Receive"));
    tx_panel_ = new TransmitPanel(state_, tabs_);
    tabs_->addTab(tx_panel_, tr("Transmit"));

    // Half duplex: our own transmission must not be decoded back into a
    // received picture. Frequency polling pauses too -- the answer is
    // not interesting mid-over, and it keeps CAT chatter off the wire
    // while keyed.
    connect(tx_panel_, &TransmitPanel::transmitStarted, rx_panel_,
            &ReceivePanel::suspend_for_transmit);
    connect(tx_panel_, &TransmitPanel::transmitFinished, rx_panel_,
            &ReceivePanel::resume_after_transmit);
    connect(tx_panel_, &TransmitPanel::transmitStarted, state_,
            &AppState::pause_rig_polling);
    connect(tx_panel_, &TransmitPanel::transmitFinished, state_,
            &AppState::resume_rig_polling);
    // The most recent picture becomes available as a transmit inset.
    connect(rx_panel_, &ReceivePanel::imageReceived, tx_panel_,
            &TransmitPanel::set_last_rx_image);
    setCentralWidget(tabs_);

    build_menu();
    build_status_bar();

    connect(state_, &AppState::modelLoaded, this, &MainWindow::on_model_loaded);
    connect(state_, &AppState::rigStatus, rig_label_, &QLabel::setText);
    connect(rx_panel_, &ReceivePanel::receptionSaved, this,
            [this](const QString& path) {
                statusBar()->showMessage(tr("Saved %1").arg(path), 5000);
            });

    state_->load_model_async();
    state_->connect_rig();
    update_station_label();
}

MainWindow::~MainWindow() = default;

void MainWindow::build_menu() {
    // The explicit NoRole calls are load-bearing on macOS. Qt's Cocoa
    // plugin pattern-matches action text and moves anything looking
    // like Preferences or Quit into the application menu. Both of this
    // menu's actions match, so Qt emptied the File menu -- and macOS
    // hides an empty menu, leaving no way to reach Settings at all.
    //
    // The shortcuts are the belt to that braces: the platform-correct
    // sequences (Cmd+, and Cmd+Q on macOS, Ctrl+Q elsewhere), so
    // Settings stays reachable even if a platform menu bar misbehaves
    // again.
    QMenu* menu = menuBar()->addMenu(tr("&File"));

    QAction* settings_action = menu->addAction(tr("&Settings..."));
    settings_action->setMenuRole(QAction::NoRole);
    settings_action->setShortcut(QKeySequence::Preferences);
    connect(settings_action, &QAction::triggered, this, &MainWindow::open_settings);

    menu->addSeparator();

    QAction* quit_action = menu->addAction(tr("&Quit"));
    quit_action->setMenuRole(QAction::NoRole);
    quit_action->setShortcut(QKeySequence::Quit);
    connect(quit_action, &QAction::triggered, this, &MainWindow::close);
}

void MainWindow::build_status_bar() {
    auto* bar = new QStatusBar(this);
    setStatusBar(bar);
    station_label_ = new QLabel(QString(), this);
    rig_label_ = new QLabel(tr("Rig control off"), this);
    model_label_ = new QLabel(tr("Loading model..."), this);
    for (QLabel* label : {station_label_, rig_label_, model_label_}) {
        bar->addPermanentWidget(label);
    }
}

void MainWindow::update_station_label() {
    const std::string& callsign = state_->config().callsign;
    station_label_->setText(
        tr("Callsign: %1")
            .arg(callsign.empty() ? tr("(no callsign set)")
                                  : QString::fromStdString(callsign)));
}

void MainWindow::on_model_loaded() {
    if (state_->model() == nullptr) {
        model_label_->setText(tr("Model failed to load"));
        QMessageBox::critical(
            this, tr("Could not load the model"),
            tr("%1\n\nSet a checkpoint path in Settings, or check your network "
               "connection for the published checkpoint.")
                .arg(state_->model_error()));
        return;
    }
    model_label_->setText(tr("Model ready"));
}

void MainWindow::open_settings() {
    SettingsDialog dialog(state_->config(), this);
    if (dialog.exec() != QDialog::Accepted) return;

    // The sequence is the window's, not the dialog's: apply, save,
    // relabel, reconnect the rig, and reload the model *only* if the
    // checkpoint actually changed -- reloading unconditionally would
    // re-download or re-open it every time the operator adjusted an
    // unrelated setting.
    const std::string previous_model = state_->config().model_path;
    const std::string previous_precision = state_->config().precision;
    dialog.apply_to(state_->config());
    state_->save_config();
    update_station_label();
    rx_panel_->sync_from_config();
    state_->connect_rig();

    if (state_->config().model_path != previous_model ||
        state_->config().precision != previous_precision) {
        model_label_->setText(tr("Loading model..."));
        state_->load_model_async();
    }
}

void MainWindow::closeEvent(QCloseEvent* event) {
    if (tx_panel_->transmitting()) {
        const auto answer = QMessageBox::question(
            this, tr("Transmitting"),
            tr("A transmission is in progress. Stop it and quit?"));
        if (answer != QMessageBox::Yes) {
            event->ignore();
            return;
        }
        tx_panel_->cancel();
    }
    rx_panel_->stop();
    state_->disconnect_rig();
    state_->save_config();
    QMainWindow::closeEvent(event);
}

}  // namespace sstvae::gui

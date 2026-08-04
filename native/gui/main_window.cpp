#include "main_window.hpp"

#include <QAction>
#include <QCloseEvent>
#include <QDateTime>
#include <QDockWidget>
#include <QGroupBox>
#include <QKeySequence>
#include <QLabel>
#include <QMenu>
#include <QMenuBar>
#include <QActionGroup>
#include <QPainter>
#include <QSplitter>
#include <QSplitterHandle>
#include <QMessageBox>
#include <QScreen>
#include <QStatusBar>
#include <QTimer>
#include <QVBoxLayout>
#include <QWidget>

#include <algorithm>

#include "app_state.hpp"
#include "log_pane.hpp"
#include "pane_container.hpp"
#include "waterfall.hpp"
#include "rig/hamlib.hpp"
#include "rx_panel.hpp"
#include "settings_dialog.hpp"
#include "tx/engine.hpp"
#include "tx_panel.hpp"

namespace sstvae::gui {

namespace {

constexpr auto APP_NAME = "SSTVAE";

// The spectrum strip before anyone has dragged it. Deliberately
// shallow: at 150 px most of it was black most of the time, and the
// height is worth more to the pictures.
constexpr int DEFAULT_WATERFALL_H = 96;

// A splitter handle that looks like one.
//
// Qt's default handle is a flat gap: on a dark theme it is invisible,
// so the waterfall was resizable with nothing on screen saying so. Three
// short dashes in the middle is the plain desktop idiom for a grip, and
// painting it keeps us out of stylesheets -- one stylesheet anywhere
// makes Qt wrap the application style and strips the padding from every
// combo and spin box in the app.
class GripHandle : public QSplitterHandle {
public:
    GripHandle(Qt::Orientation orientation, QSplitter* parent)
        : QSplitterHandle(orientation, parent) {
        setCursor(orientation == Qt::Horizontal ? Qt::SplitHCursor : Qt::SplitVCursor);
    }

protected:
    void paintEvent(QPaintEvent* event) override {
        QSplitterHandle::paintEvent(event);
        QPainter painter(this);
        painter.setPen(QPen(palette().color(QPalette::Mid), 1));
        const int cx = width() / 2;
        const int cy = height() / 2;
        // Three marks, 6 px long, 3 px apart, across the drag axis.
        for (int i = -1; i <= 1; ++i) {
            if (orientation() == Qt::Vertical) {
                painter.drawLine(cx - 12 + i * 12, cy, cx - 6 + i * 12, cy);
            } else {
                painter.drawLine(cx, cy - 12 + i * 12, cx, cy - 6 + i * 12);
            }
        }
    }
};

class GripSplitter : public QSplitter {
public:
    using QSplitter::QSplitter;

protected:
    QSplitterHandle* createHandle() override {
        return new GripHandle(orientation(), this);
    }
};

}  // namespace

MainWindow::MainWindow(QWidget* parent) : QMainWindow(parent) {
    state_ = new AppState(this);
    setWindowTitle(QString::fromLatin1(APP_NAME));
    // Wider than the tabbed window it replaces, because two panes are
    // side by side now and their minimums add (see main_window.hpp).
    // The height is unchanged from the tabbed window.
    resize(1360, 800);

    // **The spectrum spans the whole window, above both panes** -- and
    // above the *tab widget* too, so it is on screen in both layouts.
    // It is the one widget here whose job is resolving detail across a
    // frequency axis, and inside the receive pane it had about a third
    // of the width to do it in. The receive panel still drives it (it
    // owns the ring buffer), so ownership moved and control did not.
    //
    // Keeping it outside the tabs is deliberate: tabs exist for a
    // narrow screen, and what you give up there is seeing the band
    // while you compose. The strip is the cheapest way to keep it.
    waterfall_ = new Waterfall(this);
    // A floor, and no ceiling: the operator sets the height now, and a
    // maximum would silently ignore a value they had chosen.
    waterfall_->setMinimumHeight(60);

    rx_panel_ = new ReceivePanel(state_);
    tx_panel_ = new TransmitPanel(state_);
    rx_panel_->attach_waterfall(waterfall_);

    // **Each pane is named.** The tabs this replaced carried the only
    // labels that said which half was which, and dropping them left two
    // similar-looking columns of controls whose identity you had to
    // infer from a button caption ("Start receiving", "Send"). Driving
    // it settles the question: inferring is not the same as seeing.
    //
    // The titles go on the container rather than inside the panels
    // because that is where the pairing lives -- neither panel has any
    // business knowing it sits beside the other one -- and it is also
    // what lets the same two words be a group-box title in one layout
    // and a tab label in the other.
    panes_ = new PaneContainer(rx_panel_, tr("Receive"), tx_panel_, tr("Transmit"), this);
    // Equal panes are not equal pictures on their own -- the two halves
    // carry different controls beneath. Matching the *strips* is what
    // makes the pictures match, without taking space off either.
    panes_->set_control_strips(rx_panel_->control_strip(),
                               tx_panel_->control_strip());

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
    // **The waterfall's height is the operator's** (decided
    // 2026-08-03). A fixed strip was the alternative and the
    // recommendation; a handle won because 150 px is too much most of
    // the time and not enough occasionally, and something set once
    // costs nothing afterwards. The position is remembered in
    // `ui.waterfall_height`.
    //
    // A vertical handle here is unrelated to the horizontal one that
    // was removed to lock the panes equal -- that one decided which
    // pane won, and this one only decides how much spectrum history is
    // on screen.
    stack_ = new GripSplitter(Qt::Vertical, this);
    stack_->addWidget(waterfall_);
    stack_->addWidget(panes_);
    stack_->setStretchFactor(0, 0);
    stack_->setStretchFactor(1, 1);
    // Neither half may be dragged out of existence: a waterfall of zero
    // height reads as a broken capture, and the panes are the app.
    stack_->setChildrenCollapsible(false);
    stack_->setHandleWidth(6);
    setCentralWidget(stack_);
    restore_waterfall_height();
    connect(stack_, &QSplitter::splitterMoved, this,
            [this](int, int) { remember_waterfall_height(); });

    build_menu();
    build_status_bar();
    build_log_dock();

    connect(state_, &AppState::modelLoaded, this, &MainWindow::on_model_loaded);
    connect(state_, &AppState::rigStatus, this, &MainWindow::on_rig_status);
    connect(state_, &AppState::modelProgress, this,
            &MainWindow::on_model_progress);
    // The PTT lamp follows the transmit phase; the rig chip greys while
    // polling is paused so a frozen frequency does not read as current.
    connect(tx_panel_, &TransmitPanel::stateChanged, this,
            [this](int phase, double, const QString&) { on_tx_state(phase); });
    connect(tx_panel_, &TransmitPanel::sendFinished, this,
            [this](bool) { on_tx_state(static_cast<int>(tx::TxPhase::Idle)); });
    connect(tx_panel_, &TransmitPanel::transmitStarted, this,
            [this] { rig_label_->setEnabled(false); });
    connect(tx_panel_, &TransmitPanel::transmitFinished, this,
            [this] { rig_label_->setEnabled(true); });
    connect(rx_panel_, &ReceivePanel::receptionSaved, this,
            [this](const QString& path) {
                statusBar()->showMessage(tr("Saved %1").arg(path), 5000);
            });

    // The receive status is mirrored into the status bar while tabbed,
    // so composing a picture is not done blind through someone else's
    // over -- which is the one thing the side-by-side layout buys and
    // tabs give straight back. Connected in both layouts: a switch mid
    // reception then has a current line to show rather than a stale one
    // or none.
    connect(rx_panel_, &ReceivePanel::statusChanged, this, &MainWindow::on_rx_status);
    connect(panes_, &PaneContainer::modeChanged, this, &MainWindow::on_layout_changed);
    on_layout_changed(panes_->mode());

    panes_->set_mode(startup_layout());
    // After the layout is chosen, since the chosen one decides the size.
    fit_to_screen();

    state_->load_model_async();
    state_->connect_rig();
    update_station_label();
}

MainWindow::~MainWindow() {
    // Destruction order is load-bearing. `state_` is this window's
    // first child and the panels live inside the central widget, its
    // second -- so QObject's forward-order child deletion would destroy
    // AppState *before* the panels, whose teardown still uses it
    // (~ReceivePanel calls stop(); ~TransmitPanel joins the transmit
    // and optimizer workers, which may be mid-fetch through the
    // AppState-captured progress hook). Deleting the central widget by
    // hand first means every panel -- and every worker thread they own
    // -- is gone while AppState is still alive.
    //
    // **The panels' signals are severed first, and that is not
    // belt-and-braces.** `~ReceivePanel` calls `stop()`, which writes
    // its status line, which now emits `statusChanged` -- and the slot
    // for it reaches back into `panes_`. But the panels are children of
    // `panes_`, so by the time they are being destroyed `~PaneContainer`
    // has already run its body and destroyed its `QString` members,
    // while remaining a live QObject whose slots still fire. Without
    // this the last act of every quit is `set_first_note` reading a
    // destructed QString. Qt's automatic disconnect does not help: it
    // happens in `~QObject`, which is after the derived members are
    // gone.
    disconnect(rx_panel_, nullptr, this, nullptr);
    disconnect(tx_panel_, nullptr, this, nullptr);
    delete takeCentralWidget();
}

void MainWindow::build_layout_menu() {
    // A submenu with two exclusive checkable entries rather than one
    // "Tabbed layout" toggle: the two arrangements are peers, and a
    // checkbox would make the operator work out which state its unticked
    // form means.
    //
    // Deliberately no "Auto" entry, even though "auto" is a value the
    // config holds. Auto is a *first-run guess*, and offering it back to
    // someone who has already seen the window would be asking them to
    // choose "whatever you think" over an answer they can see. Picking
    // either entry writes a concrete value and ends the guessing.
    QMenu* layout = view_menu_->addMenu(tr("&Layout"));
    auto* group = new QActionGroup(this);
    group->setExclusive(true);

    split_action_ = layout->addAction(tr("&Side by side"));
    split_action_->setCheckable(true);
    split_action_->setActionGroup(group);
    connect(split_action_, &QAction::triggered, this,
            [this] { choose_layout(PaneLayout::Split); });

    tabs_action_ = layout->addAction(tr("&Tabbed"));
    tabs_action_->setCheckable(true);
    tabs_action_->setActionGroup(group);
    connect(tabs_action_, &QAction::triggered, this,
            [this] { choose_layout(PaneLayout::Tabs); });

    view_menu_->addSeparator();
}

PaneLayout MainWindow::startup_layout() const {
    // What the side-by-side arrangement actually needs, measured rather
    // than guessed: the panels' minimums move whenever either gains a
    // control, and a hardcoded breakpoint would silently stop matching
    // the layout it was chosen for.
    const QSize required = panes_->minimumSizeHint();
    const QSize available = available_screen();
    const PaneLayout mode =
        resolve_layout(state_->config().ui.layout, available, required);
    if (mode == PaneLayout::Tabs && state_->config().ui.layout == "auto") {
        // Which axis, and by how much. "Starting in tabs" on its own
        // reads as an arbitrary decision; the numbers make it checkable
        // and tell the operator what would have to change.
        state_->log_event(
            "app", log::Severity::Info,
            tr("screen is %1x%2 and side by side needs %3x%4; starting in tabs "
               "(View > Layout)")
                .arg(available.width())
                .arg(available.height())
                .arg(required.width())
                .arg(required.height()));
    }
    return mode;
}

QSize MainWindow::available_screen() const {
    const QScreen* screen = this->screen();
    return screen != nullptr ? screen->availableGeometry().size() : QSize();
}

void MainWindow::fit_to_screen() {
    const QSize available = available_screen();
    if (!available.isValid()) return;
    // **The window may not open larger than the screen.** If it does,
    // the menu bar goes off the edge with everything else, so View >
    // Layout -- the way out of a layout that does not fit -- cannot be
    // reached from inside the application at all. That is what an
    // operator actually hit.
    const QSize want = size();
    const QSize fitted = want.boundedTo(available);
    if (fitted == want) return;
    resize(fitted);
    state_->log_event("app", log::Severity::Info,
                      tr("window would not fit the %1x%2 screen; opened at %3x%4")
                          .arg(available.width())
                          .arg(available.height())
                          .arg(fitted.width())
                          .arg(fitted.height()));
}

void MainWindow::on_layout_changed(PaneLayout mode) {
    const bool tabbed = mode == PaneLayout::Tabs;
    // The receive summary earns its place in the status bar only when
    // the receive pane can be off screen. In split mode it would be the
    // same words twice, an arm's length apart.
    rx_status_label_->setVisible(tabbed);
    if (split_action_ != nullptr) split_action_->setChecked(!tabbed);
    if (tabs_action_ != nullptr) tabs_action_->setChecked(tabbed);
}

void MainWindow::on_rx_status(const QString& text) {
    rx_status_text_ = text;
    // Written unconditionally, *not* only while the label is visible.
    // `QStatusBar` hides every non-permanent widget for the duration of
    // a `showMessage` -- and this window posts one for five seconds
    // after each saved reception. Skipping the update while hidden
    // means an operator who presses Stop inside that window gets the
    // message replaced by the line from *before* it, which in tabbed
    // mode is their only view of a receiver that is no longer running.
    rx_status_label_->setText(text);
    // The tab label carries the short form, because the status bar is
    // at the other end of the window from the tab the operator is
    // deciding whether to look at.
    panes_->set_first_note(rx_panel_->listening() ? tr("listening") : QString());
}

void MainWindow::choose_layout(PaneLayout mode) {
    // An explicit choice ends "auto". Auto is a guess about the screen,
    // and a person who has just answered the question should not be
    // re-guessed at on the next launch -- including when they pick the
    // layout auto would have picked anyway, since the screen may be a
    // different one next time.
    state_->config().ui.layout = mode == PaneLayout::Tabs ? "tabs" : "split";
    state_->save_config();
    panes_->set_mode(mode);
}

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

    // View > Status log: the dock's own toggle action, so the menu
    // entry and the dock's close button cannot disagree about state.
    // The action is added in build_log_dock(), which runs after this;
    // the menu pointer is kept on the window via findChild-free means.
    view_menu_ = menuBar()->addMenu(tr("&View"));
    build_layout_menu();

    QMenu* help = menuBar()->addMenu(tr("&Help"));
    QAction* about = help->addAction(tr("&About SSTVAE"));
    about->setMenuRole(QAction::NoRole);  // as above: keep it in this menu
    connect(about, &QAction::triggered, this, &MainWindow::show_about);
}

void MainWindow::restore_waterfall_height() {
    const int wanted = state_->config().ui.waterfall_height;
    // 0 means the operator has never dragged it. Qt's own initial split
    // is a reasonable first answer, so leave it alone rather than
    // inventing a default that would then look like a chosen value.
    // Never dragged: pick a shallow default rather than letting Qt
    // split the window evenly. The strip only needs enough rows to see
    // a signal arrive -- history is what the handle is for.
    const int height = wanted > 0 ? wanted : DEFAULT_WATERFALL_H;
    const int total = stack_->height();
    // Before the window is shown `height()` is not meaningful yet; the
    // sizes are applied anyway and Qt scales them to the real height on
    // the first layout, which is what makes this work from a
    // constructor.
    const int rest = std::max(1, total - height);
    stack_->setSizes({height, rest});
}

void MainWindow::remember_waterfall_height() {
    const QList<int> sizes = stack_->sizes();
    if (sizes.isEmpty()) return;
    if (state_->config().ui.waterfall_height == sizes.front()) return;
    state_->config().ui.waterfall_height = sizes.front();
    // Saved on the drag rather than at exit: an app that is killed --
    // or that crashes in a backend -- still owes the operator the
    // layout they set.
    state_->save_config();
}

void MainWindow::show_about() {
    // The version of the *library that talks to the radio* matters as
    // much as ours: "which Hamlib" is the first question any rig
    // problem raises, and an operator should not have to find a
    // terminal to answer it. Same for the log's location, which is
    // what a bug report needs attached.
    QString text = tr("<b>SSTVAE</b><br>"
                      "Image transmission over HF radio.<br><br>"
                      "Hamlib %1<br>Qt %2")
                       .arg(QString::fromStdString(rig::hamlib_version()),
                            QString::fromLatin1(qVersion()));
    const QString log_note = state_->log_file_note();
    if (log_note.isEmpty()) {
        text += tr("<br><br>Status log: %1").arg(state_->log_file_path());
    } else {
        text += QStringLiteral("<br><br>") + log_note;
    }
    QMessageBox::about(this, tr("About SSTVAE"), text);
}

void MainWindow::build_status_bar() {
    auto* bar = new QStatusBar(this);
    setStatusBar(bar);
    // The PTT lamp: hidden except while the rig is keyed. Bold red
    // text on the standard background -- a state indicator, not chrome
    // -- because "the radio is transmitting" is the one state in the
    // app that must be visible at a glance.
    ptt_label_ = new QLabel(tr("TX"), this);
    QFont ptt_font = ptt_label_->font();
    ptt_font.setBold(true);
    ptt_label_->setFont(ptt_font);
    QPalette ptt_palette = ptt_label_->palette();
    ptt_palette.setColor(QPalette::WindowText, QColor(0xb3, 0x26, 0x1e));
    ptt_label_->setPalette(ptt_palette);
    ptt_label_->hide();

    // The receive summary, for tabbed mode -- hidden in split mode,
    // where the receive pane is on screen saying the same thing.
    // `Ignored` so a long reception line ("Receiving mode C: frame
    // 431/440 (98%) -- SNR 12.4 dB de W1AW") cannot pin the window's
    // minimum width, which is the whole reason the tabbed layout exists.
    rx_status_label_ = new QLabel(QString(), this);
    rx_status_label_->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Preferred);
    rx_status_label_->hide();
    bar->addWidget(rx_status_label_, 1);

    station_label_ = new QLabel(QString(), this);
    rig_label_ = new QLabel(tr("Rig control off"), this);
    model_label_ = new QLabel(tr("Loading model..."), this);
    for (QLabel* label : {ptt_label_, station_label_, rig_label_, model_label_}) {
        bar->addPermanentWidget(label);
    }

    rig_error_timer_ = new QTimer(this);
    rig_error_timer_->setInterval(5000);
    connect(rig_error_timer_, &QTimer::timeout, this,
            &MainWindow::refresh_rig_error_age);
}

void MainWindow::build_log_dock() {
    log_pane_ = new LogPane(&state_->status_log(), this);
    log_pane_->set_file_note(state_->log_file_note());
    connect(state_, &AppState::logEntry, log_pane_, &LogPane::append);

    log_dock_ = new QDockWidget(tr("Status log"), this);
    // **One row, not two.** A QDockWidget's own title bar is a full row
    // of height carrying nothing but a word and a close button, and the
    // pane already had a row of its own for the filter and Copy. At the
    // bottom of the window those two rows cost more than the log they
    // introduce. The pane's row becomes the title bar, with the name
    // and the close button folded into it.
    log_dock_->setTitleBarWidget(log_pane_->take_title_row(tr("Status log"), log_dock_));
    // Closable so it can be put away; not floatable or movable -- it is
    // a log strip, not a tool window, and the bottom is its place.
    log_dock_->setFeatures(QDockWidget::DockWidgetClosable);
    log_dock_->setAllowedAreas(Qt::BottomDockWidgetArea);
    log_dock_->setWidget(log_pane_);
    addDockWidget(Qt::BottomDockWidgetArea, log_dock_);
    view_menu_->addAction(log_dock_->toggleViewAction());

    // An error re-opens a closed dock: the log is where the detail
    // lives, and an error with the log hidden would be exactly the
    // silent failure this pane exists to end.
    connect(state_, &AppState::logEntry, this,
            [this](qlonglong, const QString&, int severity, const QString&) {
                if (severity == static_cast<int>(log::Severity::Error)) {
                    log_dock_->show();
                }
            });

    // Queued is mandatory, not a preference: the signal is emitted
    // under the StatusLog lock, and this slot appends to that same log.
    connect(
        state_, &AppState::fileLogFailed, this,
        [this](const QString& what) {
            log_pane_->set_file_note(tr("(log file unavailable: %1)").arg(what));
            state_->log_event("app", log::Severity::Error,
                              tr("file log failed: %1").arg(what));
        },
        Qt::QueuedConnection);
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
    // Refinement needs a codec to start from, so a run that could not
    // be armed at startup (or after a checkpoint change) is armed here
    // rather than waiting for the operator to edit something.
    tx_panel_->sync_from_config();
}

void MainWindow::on_rig_status(const QString& text, bool error) {
    if (error) {
        rig_error_text_ = text;
        rig_error_since_ms_ = QDateTime::currentMSecsSinceEpoch();
        rig_label_->setText(text);
        rig_error_timer_->start();
    } else {
        rig_error_text_.clear();
        rig_error_timer_->stop();
        rig_label_->setText(text);
    }
}

void MainWindow::refresh_rig_error_age() {
    if (rig_error_text_.isEmpty()) return;
    const qint64 seconds =
        (QDateTime::currentMSecsSinceEpoch() - rig_error_since_ms_) / 1000;
    // The controller deduplicates identical failures and backs its poll
    // off to a minute, so without this a stale error is indistinguishable
    // from a fresh one.
    rig_label_->setText(tr("%1 (%2 s ago)").arg(rig_error_text_).arg(seconds));
}

void MainWindow::on_model_progress(qlonglong received, qlonglong total) {
    // The hook is global, so this fires for *every* artifact -- the
    // decoder during the initial load, but also the encoder fetched
    // lazily on the first Send and the gradient graph on the first
    // refinement. The fetcher always ends with a final
    // (size, size) call, which is the reset point: without it the
    // label would read "downloading" for the rest of the session
    // after any of those later fetches.
    if (total > 0 && received >= total) {
        model_label_->setText(state_->model() != nullptr
                                  ? tr("Model ready")
                                  : tr("Loading model..."));
        return;
    }
    if (total > 0) {
        model_label_->setText(tr("Model: downloading %1 / %2 MB")
                                  .arg(received / 1e6, 0, 'f', 1)
                                  .arg(total / 1e6, 0, 'f', 1));
    } else {
        model_label_->setText(
            tr("Model: downloading %1 MB").arg(received / 1e6, 0, 'f', 1));
    }
}

void MainWindow::on_tx_state(int phase) {
    // Keyed is Keying through Unkeying inclusive: the lamp exists for
    // "the radio is (or should be) transmitting right now".
    const auto p = static_cast<tx::TxPhase>(phase);
    const bool keyed = p == tx::TxPhase::Keying || p == tx::TxPhase::Sending ||
                       p == tx::TxPhase::Unkeying;
    ptt_label_->setVisible(keyed);
}

void MainWindow::open_settings() {
    SettingsDialog dialog(state_->config(), this);
    // Test CAT / Test PTT results are shown in a message box the
    // operator dismisses; the log keeps what it said.
    connect(&dialog, &SettingsDialog::rigTestFinished, state_,
            [this](bool ok, const QString& message) {
                state_->log_event("rig",
                                  ok ? log::Severity::Info : log::Severity::Error,
                                  tr("rig test: %1").arg(message));
            });
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
    // Picture refinement can be switched on or off here, and either way
    // it has to take effect now: on, and a run starts for what is
    // already composed; off, and any refined latents are discarded.
    tx_panel_->sync_from_config();
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

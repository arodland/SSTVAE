// The application window.
//
// **The receive and transmit panels sit side by side in a splitter,
// not in tabs.** The reference (and this port until now) put them in a
// `QTabWidget`, which meant composing a picture was done blind: no
// waterfall, no decode progress, no incoming preview. On the band that
// is the wrong trade -- you prepare the next transmission *while*
// listening, and the thing you most want to see while composing is
// whether someone else is sending.
//
// **The price is the window's minimum width, and it is not small.** A
// `QSplitter`'s minimum is the *sum* of its children's, not the max a
// tab widget got away with, and a collapsed pane still counts because
// it is not hidden. Measured with `sstvae-gui-shot`, which prints each
// pane's `minimumSizeHint`: transmit 545, receive 464, so the window
// floor is ~1015 px against ~738 before. Progress-tier labels are
// therefore set `QSizePolicy::Ignored` wherever they would otherwise
// pin a width -- that alone took transmit from 738 to 545.
//
// **Tabs remain available for a screen that cannot meet that floor**
// (`ui.layout`, View > Layout). Scrolling inside the panes was the
// other candidate and treats the symptom: the floor is a property of
// the arrangement, and only the arrangement can move it. See
// `pane_container.hpp` for the switch and why "auto" measures the
// screen rather than the window.
//
// The wiring between the panels lives here rather than in either of
// them, because all of it is about the pair: half duplex (our own
// transmission must not be decoded back into a received picture),
// pausing frequency polling while keyed, and handing the most recent
// received picture to the transmitter as an overlay inset.

#ifndef SSTVAE_GUI_MAIN_WINDOW_HPP
#define SSTVAE_GUI_MAIN_WINDOW_HPP

#include <QMainWindow>
#include <QSize>

#include "pane_container.hpp"

class QAction;
class QSplitter;
class QDockWidget;
class QLabel;
class QMenu;
class QTimer;

namespace sstvae::gui {

class AppState;
class LogPane;
class ReceivePanel;
class Waterfall;
class TransmitPanel;

class MainWindow : public QMainWindow {
    Q_OBJECT

public:
    explicit MainWindow(QWidget* parent = nullptr);
    ~MainWindow() override;

public slots:
    void open_settings();

protected:
    void closeEvent(QCloseEvent* event) override;

private:
    void build_menu();
    void build_layout_menu();
    // Switch layout and record the operator's choice. Distinct from
    // `PaneContainer::set_mode`, which is told what to do: this is the
    // path that also stops the setting being "auto", because a person
    // has now answered the question auto was guessing at.
    void choose_layout(PaneLayout mode);
    // What the config and the screen between them say to start in.
    PaneLayout startup_layout() const;
    // The screen's usable area, or an invalid size if the platform did
    // not report one.
    QSize available_screen() const;
    // Shrink the window to the screen if it opened larger, and say so.
    void fit_to_screen();
    void on_layout_changed(PaneLayout mode);
    // The receive status line, mirrored into the status bar while
    // tabbed -- see `on_rx_status`.
    void on_rx_status(const QString& text);
    void build_status_bar();
    void build_log_dock();
    void update_station_label();
    void on_model_loaded();
    void on_rig_status(const QString& text, bool error);
    void on_model_progress(qlonglong received, qlonglong total);
    void on_tx_state(int phase);
    void show_about();
    // The waterfall's share of the window, persisted so a handle set
    // once stays set. Saved on every drag rather than at exit, because
    // an app that is killed still owes the operator their layout.
    void restore_waterfall_height();
    void remember_waterfall_height();
    void refresh_rig_error_age();

    AppState* state_ = nullptr;
    QSplitter* stack_ = nullptr;
    PaneContainer* panes_ = nullptr;
    Waterfall* waterfall_ = nullptr;
    ReceivePanel* rx_panel_ = nullptr;
    TransmitPanel* tx_panel_ = nullptr;
    QLabel* ptt_label_ = nullptr;
    QLabel* station_label_ = nullptr;
    QLabel* rig_label_ = nullptr;
    QLabel* model_label_ = nullptr;
    QDockWidget* log_dock_ = nullptr;
    // Set only while an error is forcing the log dock open, so the
    // toggle that causes is not mistaken for the operator's own.
    bool auto_opening_log_ = false;
    // `receive.auto_start`, still owed. Set once at construction and
    // cleared the first time it is acted on, because it describes how
    // the app *starts* -- the model also loads when the operator changes
    // the checkpoint, and that must not restart a session they stopped.
    // It survives a failed load, so fixing a bad model path in Settings
    // still gets the listening station it asked for.
    bool auto_start_pending_ = false;
    LogPane* log_pane_ = nullptr;
    QMenu* view_menu_ = nullptr;
    QAction* split_action_ = nullptr;
    QAction* tabs_action_ = nullptr;

    // The receive panel's own status line, shown in the status bar
    // *only* while tabbed. Kept even in split mode so a switch mid
    // reception has something to display at once rather than waiting
    // for the next poll.
    QLabel* rx_status_label_ = nullptr;
    QString rx_status_text_;

    // The rig label's error-age display: a CAT failure keeps its text,
    // and this pair appends "(N s ago)" so a stale error cannot pass
    // for a fresh one (the controller deduplicates identical repeats,
    // so the text alone never says how old it is).
    QString rig_error_text_;
    qint64 rig_error_since_ms_ = 0;
    QTimer* rig_error_timer_ = nullptr;
};

}  // namespace sstvae::gui

#endif

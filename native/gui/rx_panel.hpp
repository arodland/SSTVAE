// Receive panel: waterfall, live picture, and the reception controls.
//
// A port of `sstvae/gui/rx_panel.py`, keeping its threading shape,
// which is the part with the hazards in it:
//
// **The decode loop runs on a plain worker thread and reaches this
// widget only through queued signals.** Nothing here touches a widget
// from the decode thread.
//
// **The sink saves on the decode thread rather than signalling the GUI
// to save.** Writing a PNG is file I/O, and the UI thread must not
// stall on a slow or full disk mid-reception.
//
// **Saving is the sink's job, not the loop's**, because the autosave
// checkbox may mean holding the picture for the Save button instead of
// writing it. Autosave is read through the config at reception time
// rather than captured at start, so toggling it takes effect on the
// very next reception without restarting the loop.

#ifndef SSTVAE_GUI_RX_PANEL_HPP
#define SSTVAE_GUI_RX_PANEL_HPP

#include <QPixmap>
#include <QPointer>
#include <QWidget>

#include <atomic>
#include <memory>
#include <optional>
#include <string>
#include <thread>

#include "images/types.hpp"
#include "picture_box.hpp"
// Complete type, not a forward declaration: QPointer<T> instantiates
// static_casts through QObject and needs to know T derives from it.
#include "waterfall.hpp"
#include "rx/engine.hpp"
#include "rx/ringbuffer.hpp"

class QCheckBox;
class QLabel;
class QProgressBar;
class QPushButton;

namespace sstvae::audio::qt {
class InputStream;
}

namespace sstvae::gui {

class AppState;
class ErrorBanner;

class ReceivePanel : public QWidget {
    Q_OBJECT

public:
    explicit ReceivePanel(AppState* state, QWidget* parent = nullptr);
    ~ReceivePanel() override;

    bool listening() const;

    // The spectrum strip now spans the whole window rather than living
    // in this pane -- it is the one widget whose job is resolving
    // detail across a frequency axis, and a third of the window was
    // never enough for it. The panel still drives it (it owns the ring
    // buffer), so ownership moved to MainWindow while control did not.
    // Null until attached, and every use is guarded.
    void attach_waterfall(Waterfall* waterfall) { waterfall_ = waterfall; }
    // Null-safe accessor. **Not defensive programming for its own
    // sake**: moving the strip up to the window made it a *sibling* of
    // this panel rather than a child, and it is added to the central
    // widget's layout first -- so at teardown `~QWidget` deletes it
    // before the panel whose `stop()` still calls `set_ring(nullptr)`.
    // A `QPointer` goes null when the widget dies, which turns a
    // use-after-free at every exit into a no-op.
    Waterfall* fall() const { return waterfall_.data(); }
    // The 4:3 picture area. Read by tests and by the screenshot tool;
    // nothing sizes it from outside any more.
    QWidget* picture_area() const;
    // Everything below the picture, as one widget, so the container can
    // hold it to the same height as the transmit pane's strip -- which
    // is what makes the two pictures the same size.
    QWidget* control_strip() const;

    // Re-read the settings the panel mirrors in its own controls. The
    // autosave checkbox exists in both places, so a change made in the
    // dialog has to show up here or the two disagree about what is on.
    void sync_from_config();

public slots:
    bool start();
    void stop();
    void save_current();
    // Reveal the last saved picture in the desktop's file manager.
    void open_saved_folder();

    // Fill every text surface with the longest content it can hold, for
    // `sstvae-gui-shot`. A slot for the same reason `Waterfall::tick`
    // is one: the layout facts worth looking at -- whether the status
    // line, the card and the five-button controls row fit a narrow pane
    // -- are invisible on a freshly constructed panel, where all three
    // are empty. Touches no engine and starts nothing.
    void fill_for_screenshot();

    // Half duplex. Our own audio would otherwise be decoded straight
    // back into a "received" picture.
    void suspend_for_transmit();
    void resume_after_transmit();

signals:
    void receptionSaved(const QString& path);
    // Newest *complete* picture, for the transmit panel's inset. A
    // partial one would be a poor keepsake.
    void imageReceived(const images::Picture& image);
    void listeningChanged(bool listening);
    // Whatever the panel's own status line now reads. Emitted for every
    // change, so a second display of it (the status bar, while the
    // window is tabbed and this pane may be behind the other one) cannot
    // drift from the first.
    void statusChanged(const QString& text);

    // Emitted from the decode thread; connected queued to the slots
    // below so the work lands on the GUI thread.
    void receptionFinished(const QString& saved_path);
    void errorOccurred(const QString& message);

protected:
    // Only to keep the error banner sitting over the top of the picture.
    void resizeEvent(QResizeEvent* event) override;

private slots:
    void refresh_status();
    void on_reception(const QString& saved_path);
    void on_error(const QString& message);
    void on_autosave_toggled(bool on);

private:
    void build_ui();
    // Keeps the floating error banner across the top of the picture.
    void place_banner();
    // The one way the status line is written. A bare `status_->setText`
    // would leave the status bar's copy stale on whichever of the five
    // call sites someone forgot.
    void set_status(const QString& text);
    void show_image(const images::Picture& image);
    void set_displayed(const images::Picture& image, const std::string& callsign,
                       const std::optional<std::string>& mode_name);
    std::optional<std::string> handle_reception(const rx::Reception& reception);
    std::optional<std::string> save_reception(const rx::Reception& reception);
    void save_audio_beside(const std::string& image_path);

    AppState* app_ = nullptr;

    // --- widgets
    ErrorBanner* banner_ = nullptr;
    PictureBox* preview_ = nullptr;
    QWidget* strip_ = nullptr;
    QLabel* status_ = nullptr;
    // Diversity-only lock lamps ("Primary"/"Secondary"), hidden unless
    // diversity_active_ -- a single-receiver session has no second
    // branch for these to describe.
    QLabel* branch_a_label_ = nullptr;
    QLabel* branch_b_label_ = nullptr;
    QLabel* last_card_ = nullptr;
    QProgressBar* progress_ = nullptr;
    QPointer<Waterfall> waterfall_;
    QPushButton* start_button_ = nullptr;
    QPushButton* stop_button_ = nullptr;
    QPushButton* save_button_ = nullptr;
    QPushButton* folder_button_ = nullptr;
    QCheckBox* autosave_ = nullptr;

    // --- reception
    std::shared_ptr<rx::RingBuffer> ring_;
    std::unique_ptr<audio::qt::InputStream> stream_;
    // Diversity reception's second branch (docs/diversity-reception.md,
    // settings::ReceiveConfig::diversity_enabled); null otherwise. The
    // waterfall only ever follows ring_, the primary branch -- a second
    // spectrum strip for the diversity device is not implemented here.
    std::shared_ptr<rx::RingBuffer> ring2_;
    std::unique_ptr<audio::qt::InputStream> stream2_;
    // Set for the lifetime of a diversity session (mirrors ring2_'s
    // presence), so the lamps can be shown/hidden once at start()/stop()
    // rather than re-checked every poll.
    bool diversity_active_ = false;
    std::unique_ptr<rx::SharedState> shared_;
    rx::StopFlag stop_flag_;
    std::thread thread_;
    std::atomic<bool> running_{false};

    // The picture currently in the preview, and what to name it. Not
    // necessarily a *finished* reception: the loop reconstructs on every
    // poll, so there is usually a picture on screen long before the
    // transmission ends -- and on a marginal signal the operator may
    // want the version on screen now rather than whatever it settles on.
    std::optional<images::Picture> displayed_;
    std::string displayed_callsign_;
    std::optional<std::string> displayed_mode_;
    // Identity check, to avoid repainting a picture already on screen.
    const images::Picture* shown_ = nullptr;

    // Handed to the GUI thread by the sink, which runs on the decode
    // thread; guarded by the same lock the reception signal implies.
    std::optional<rx::Reception> last_reception_;
    std::mutex reception_mutex_;

    std::optional<std::string> last_saved_path_;
    bool suspended_for_tx_ = false;
    // Edge detection for the log: the engine has no "sync acquired"
    // event, only a status that polls as "receiving", so the transition
    // is detected here and logged once per acquisition.
    bool was_receiving_ = false;
};

}  // namespace sstvae::gui

#endif

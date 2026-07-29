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
#include <QWidget>

#include <atomic>
#include <memory>
#include <optional>
#include <string>
#include <thread>

#include "images/types.hpp"
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
class Waterfall;

class ReceivePanel : public QWidget {
    Q_OBJECT

public:
    explicit ReceivePanel(AppState* state, QWidget* parent = nullptr);
    ~ReceivePanel() override;

    bool listening() const;

    // Re-read the settings the panel mirrors in its own controls. The
    // autosave checkbox exists in both places, so a change made in the
    // dialog has to show up here or the two disagree about what is on.
    void sync_from_config();

public slots:
    bool start();
    void stop();
    void save_current();

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

    // Emitted from the decode thread; connected queued to the slots
    // below so the work lands on the GUI thread.
    void receptionFinished(const QString& saved_path);
    void errorOccurred(const QString& message);

protected:
    void resizeEvent(QResizeEvent* event) override;

private slots:
    void refresh_status();
    void on_reception(const QString& saved_path);
    void on_error(const QString& message);
    void on_autosave_toggled(bool on);

private:
    void build_ui();
    void show_image(const images::Picture& image);
    void set_displayed(const images::Picture& image, const std::string& callsign,
                       const std::optional<std::string>& mode_name);
    std::optional<std::string> handle_reception(const rx::Reception& reception);
    std::optional<std::string> save_reception(const rx::Reception& reception);
    void save_audio_beside(const std::string& image_path);

    AppState* app_ = nullptr;

    // --- widgets
    QLabel* preview_ = nullptr;
    QLabel* status_ = nullptr;
    QProgressBar* progress_ = nullptr;
    Waterfall* waterfall_ = nullptr;
    QPushButton* start_button_ = nullptr;
    QPushButton* stop_button_ = nullptr;
    QPushButton* save_button_ = nullptr;
    QCheckBox* autosave_ = nullptr;

    // --- reception
    std::shared_ptr<rx::RingBuffer> ring_;
    std::unique_ptr<audio::qt::InputStream> stream_;
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
    // Unscaled, so a resize rescales from the original rather than
    // compounding losses.
    QPixmap preview_pixmap_;
    // Identity check, to avoid repainting a picture already on screen.
    const images::Picture* shown_ = nullptr;

    // Handed to the GUI thread by the sink, which runs on the decode
    // thread; guarded by the same lock the reception signal implies.
    std::optional<rx::Reception> last_reception_;
    std::mutex reception_mutex_;

    std::optional<std::string> last_saved_path_;
    bool suspended_for_tx_ = false;
};

}  // namespace sstvae::gui

#endif

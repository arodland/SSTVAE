// Transmit panel: pick a picture, compose an overlay, send it.
//
// A port of `sstvae/gui/tx_panel.py`. The transmission itself runs on a
// worker thread; `TxEngine` may call back from that thread, from the
// audio callback, or from its watchdog, so every callback here marshals
// to the GUI thread through a queued signal.

#ifndef SSTVAE_GUI_TX_PANEL_HPP
#define SSTVAE_GUI_TX_PANEL_HPP

#include <QWidget>

#include <atomic>
#include <memory>
#include <optional>
#include <string>
#include <thread>

#include "images/types.hpp"
#include "optimize/speculative.hpp"
#include "overlay/model.hpp"
#include "tx/engine.hpp"

class QComboBox;
class QDoubleSpinBox;
class QGroupBox;
class QLabel;
class QPlainTextEdit;
class QProgressBar;
class QPushButton;
class QSlider;
class QTimer;

namespace sstvae::gui {

class AppState;
class OverlayEditor;

// The output level is stored as a peak amplitude (`transmit.level`,
// 0..1) because that is what the transmitter scales to, but it is
// *shown* in dB relative to full scale, which is the unit the operator
// is already working in at the radio. 0 dB is 1.0.
inline constexpr double LEVEL_MIN_DB = -30.0;
inline constexpr double LEVEL_STEP_DB = 0.5;
// Writing the config on every tick of a drag would be dozens of atomic
// saves; the in-memory value updates immediately and the file follows
// once the operator settles.
inline constexpr int LEVEL_SAVE_DELAY_MS = 500;

double level_to_db(double level);
double db_to_level(double db);

class TransmitPanel : public QWidget {
    Q_OBJECT

public:
    explicit TransmitPanel(AppState* state, QWidget* parent = nullptr);
    ~TransmitPanel() override;

    bool transmitting() const;

public slots:
    void send();
    // Re-arm speculative optimization for the current composition.
    // Cheap and idempotent: the debounce inside `Speculative` is what
    // keeps a drag from starting a run per mouse move.
    void schedule_optimization();
    void cancel();
    void choose_image();
    void load_image(const QString& path);
    // The receive panel's newest complete picture, for a "last_rx" inset.
    void set_last_rx_image(const images::Picture& image);

signals:
    void transmitStarted();
    void transmitFinished();

    // Emitted from the optimizer's worker thread; queued to the slot
    // below, exactly like the transmit callbacks above.
    void optimizerProgressed();

    // Emitted from the transmitting thread; queued to the slots below.
    void stateChanged(int phase, double progress, const QString& message);
    void errorOccurred(const QString& message);
    void sendFinished(bool ok);

private slots:
    void on_optimizer_progress();
    void on_state(int phase, double progress, const QString& message);
    void on_error(const QString& message);
    void on_finished(bool ok);
    void on_selection(overlay::Item* item);
    void on_mode_changed();
    void on_level_changed(int steps);

private:
    void build_ui();
    // Everything from "the operator pressed Send" onwards, once any
    // optimization has settled. Split out because Send may have to wait
    // for the optimizer first, and that wait must not block the GUI.
    void begin_transmit(const images::Picture& picture,
                        std::vector<double> latents);
    void rebuild_optimizer();
    QWidget* build_side_panel();
    QGroupBox* build_properties(QWidget* parent);
    QWidget* build_send_bar();
    void update_level_label();
    overlay::Item* editing_item();

    AppState* app_ = nullptr;
    OverlayEditor* editor_ = nullptr;

    QPushButton* choose_button_ = nullptr;
    QLabel* image_label_ = nullptr;
    QPushButton* add_rx_button_ = nullptr;

    QGroupBox* properties_ = nullptr;
    QPlainTextEdit* text_edit_ = nullptr;
    QComboBox* align_combo_ = nullptr;
    QDoubleSpinBox* size_spin_ = nullptr;
    QDoubleSpinBox* rotation_spin_ = nullptr;
    QPushButton* color_button_ = nullptr;
    // Set while the property widgets are being filled from an item, so
    // their change signals do not write straight back into it.
    bool loading_properties_ = false;

    QComboBox* mode_combo_ = nullptr;
    QSlider* level_slider_ = nullptr;
    QLabel* level_label_ = nullptr;
    QTimer* save_level_timer_ = nullptr;
    QPushButton* send_button_ = nullptr;
    QPushButton* cancel_button_ = nullptr;
    QProgressBar* progress_ = nullptr;
    QLabel* status_ = nullptr;

    std::unique_ptr<tx::TxEngine> engine_;
    std::thread thread_;
    std::atomic<bool> running_{false};

    // Null when the feature is off or the model has not loaded yet.
    std::unique_ptr<optimize::Speculative> optimizer_;
    // Set between pressing Send and the optimizer settling; polled by
    // `wait_timer_`, which is the only thing that may start the
    // transmission in that window.
    QTimer* wait_timer_ = nullptr;
    bool awaiting_optimizer_ = false;
    // The composition as it was when Send was pressed. Send commits to
    // that picture: editing during the wait (or during transmission)
    // applies to the *next* send, not this one.
    std::optional<images::Picture> send_picture_;
    // An edit arrived while a send was in progress; re-arm once it is
    // over rather than moving the ground under it.
    bool restart_after_send_ = false;
};

}  // namespace sstvae::gui

#endif

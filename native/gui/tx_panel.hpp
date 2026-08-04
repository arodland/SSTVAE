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

#include "images/images.hpp"
#include "images/types.hpp"
#include "optimize/speculative.hpp"
#include "overlay/model.hpp"
#include "tx/engine.hpp"

class QColor;
class QComboBox;
class QDoubleSpinBox;
class QDragEnterEvent;
class QDropEvent;
class QGroupBox;
class QLabel;
class QPlainTextEdit;
class QProgressBar;
class QPushButton;
class QSlider;
class QTimer;

namespace sstvae::gui {

class AppState;
class ErrorBanner;
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
    // The 4:3 canvas. Read by tests and by the screenshot tool.
    QWidget* picture_area() const;
    // Everything below the canvas, as one widget, so the container can
    // hold it to the same height as the receive pane's strip.
    QWidget* control_strip() const;

public slots:
    void send();
    // The settings dialog was accepted, or the model finished loading.
    // Both can turn refinement on or off underneath us, and both must
    // take effect at once rather than at the next edit.
    void sync_from_config();
    // Re-arm speculative optimization for the current composition.
    // Cheap and idempotent: the debounce inside `Speculative` is what
    // keeps a drag from starting a run per mouse move.
    void schedule_optimization();
    void cancel();
    void choose_image();
    void load_image(const QString& path);
    // Re-open the framing dialog for the picture already loaded. The
    // original is kept, so this starts from the source rather than
    // cropping what the last framing produced.
    void choose_framing();
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

protected:
    // A dropped picture file loads exactly as a chosen one does.
    void dragEnterEvent(QDragEnterEvent* event) override;
    void dropEvent(QDropEvent* event) override;

protected:
    // Only to keep the error banner sitting over the top of the picture.
    void resizeEvent(QResizeEvent* event) override;

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
    // Keeps the floating error banner across the top of the picture.
    void place_banner();
    // Paint the current text colour onto the Colour button.
    void set_color_swatch(const QColor& color);
    // True while the picture is committed to a send in progress.
    bool picture_locked() const;
    void set_picture_controls_enabled(bool on);
    // Push `framing_` through `images::fit` into the editor, and
    // relabel the picture honestly.
    void apply_framing();
    // Everything from "the operator pressed Send" onwards, once any
    // optimization has settled. Split out because Send may have to wait
    // for the optimizer first, and that wait must not block the GUI.
    void begin_transmit(const images::Picture& picture,
                        std::vector<double> latents);
    void rebuild_optimizer();
    QWidget* build_tool_row();
    QGroupBox* build_properties(QWidget* parent);
    QWidget* build_send_bar();
    void update_level_label();
    overlay::Item* editing_item();

    AppState* app_ = nullptr;
    OverlayEditor* editor_ = nullptr;
    ErrorBanner* banner_ = nullptr;

    QPushButton* choose_button_ = nullptr;
    QPushButton* frame_button_ = nullptr;
    QLabel* image_label_ = nullptr;
    QPushButton* add_rx_button_ = nullptr;

    // The picture as loaded, at its own size, plus how it is framed
    // into the transmit canvas. Kept apart so re-framing starts from
    // the source instead of compounding crops.
    std::optional<images::Picture> source_;
    QString source_path_;
    images::Framing framing_;

    QWidget* strip_ = nullptr;
    QGroupBox* properties_ = nullptr;
    QPlainTextEdit* text_edit_ = nullptr;
    QComboBox* align_combo_ = nullptr;
    QDoubleSpinBox* size_spin_ = nullptr;
    QDoubleSpinBox* rotation_spin_ = nullptr;
    QPushButton* color_button_ = nullptr;
    // What the swatch currently shows, so a drag's per-mouse-move
    // selectionChanged does not rebuild an identical icon.
    QColor swatch_color_;
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

    // Edge detection for the log: `on_state` fires on every playback
    // progress tick, so phase transitions are logged only when the
    // phase actually changes.
    int last_logged_phase_ = -1;
    // The optimizer's finished result is logged once per run, not once
    // per progress callback.
    bool optimizer_result_logged_ = false;
};

}  // namespace sstvae::gui

#endif

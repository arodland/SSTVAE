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
#include <string>
#include <thread>

#include "images/types.hpp"
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
    void cancel();
    void choose_image();
    void load_image(const QString& path);
    // The receive panel's newest complete picture, for a "last_rx" inset.
    void set_last_rx_image(const images::Picture& image);

signals:
    void transmitStarted();
    void transmitFinished();

    // Emitted from the transmitting thread; queued to the slots below.
    void stateChanged(int phase, double progress, const QString& message);
    void errorOccurred(const QString& message);
    void sendFinished(bool ok);

private slots:
    void on_state(int phase, double progress, const QString& message);
    void on_error(const QString& message);
    void on_finished(bool ok);
    void on_selection(overlay::Item* item);
    void on_mode_changed();
    void on_level_changed(int steps);

private:
    void build_ui();
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
};

}  // namespace sstvae::gui

#endif

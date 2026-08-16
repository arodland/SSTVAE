// The settings dialog.
//
// Reads a configuration and writes it back only through `apply_to`,
// which the caller invokes on OK -- so Cancel really cancels.
//
// The device pickers store the device *name* rather than an index:
// indices are renumbered whenever a USB device is plugged or unplugged,
// so a saved index silently comes back pointing at a different
// soundcard.
//
// The Rig tab is the one part that is not a port of
// `sstvae/gui/settings_dialog.py`. That dialog configured a *rigctld
// socket*; this one configures Hamlib directly, and is modelled on
// WSJT-X's Radio tab because that is the set a real radio needs and the
// one operators already know. See `settings::RigConfig`.

#ifndef SSTVAE_GUI_SETTINGS_DIALOG_HPP
#define SSTVAE_GUI_SETTINGS_DIALOG_HPP

#include <QDialog>

#include <string>

#include "settings/settings.hpp"

class QCheckBox;
class QComboBox;
class QDoubleSpinBox;
class QLabel;
class QLineEdit;
class QPushButton;
class QWidget;

namespace sstvae::gui {

class SettingsDialog : public QDialog {
    Q_OBJECT

public:
    explicit SettingsDialog(const settings::Config& config,
                            QWidget* parent = nullptr);
    ~SettingsDialog() override;

    // Copy the dialog's values into `config`.
    void apply_to(settings::Config& config) const;

private slots:
    void sync_precision_enabled();
    void refresh_devices();
    void sync_ptt_enabled();
    // Show what the current filename template would actually produce.
    void sync_filename_preview();
    void test_cat();
    void test_ptt();
    void on_rig_test_finished(bool ok, const QString& message);
    void download_all_models();
    void on_model_download_finished(bool ok, const QString& message);

signals:
    // Emitted from the worker thread the rig test runs on; queued, so
    // the message box opens on the GUI thread. Nothing on the GUI
    // thread may block on the rig, not even for a test button.
    void rigTestFinished(bool ok, const QString& message);
    // Emitted from the worker thread that fetches the model artifacts;
    // queued for the same reason -- a download is real network I/O, and
    // the GUI thread must not wait on it.
    void modelDownloadFinished(bool ok, const QString& message);

private:
    QWidget* model_tab();
    QWidget* audio_tab();
    QWidget* rig_tab();
    QWidget* folders_tab();
    QWidget* receive_tab();
    QWidget* transmit_tab();

    void fill_device_combo(QComboBox* combo, bool input,
                           const std::string& current);
    void run_rig_test(bool key_ptt);
    settings::RigConfig pending_rig() const;
    int rig_model_number() const;
    void set_rig_test_busy(bool busy);
    void set_download_busy(bool busy);

    settings::Config config_;

    // Model
    QLineEdit* model_path_ = nullptr;
    QComboBox* precision_ = nullptr;
    QLabel* precision_note_ = nullptr;
    QPushButton* download_all_ = nullptr;

    // Audio
    QComboBox* input_device_ = nullptr;
    QComboBox* output_device_ = nullptr;

    // Rig
    QCheckBox* rig_enabled_ = nullptr;
    QComboBox* rig_model_ = nullptr;
    QLabel* rig_model_note_ = nullptr;
    QLineEdit* rig_device_ = nullptr;
    QComboBox* rig_baud_ = nullptr;
    QComboBox* data_bits_ = nullptr;
    QComboBox* stop_bits_ = nullptr;
    QComboBox* parity_ = nullptr;
    QComboBox* handshake_ = nullptr;
    QCheckBox* dtr_forced_ = nullptr;
    QComboBox* dtr_ = nullptr;
    QCheckBox* rts_forced_ = nullptr;
    QComboBox* rts_ = nullptr;
    QComboBox* ptt_method_ = nullptr;
    QLineEdit* ptt_device_ = nullptr;
    QComboBox* rig_mode_ = nullptr;
    QDoubleSpinBox* poll_interval_s_ = nullptr;
    QDoubleSpinBox* ptt_lead_ = nullptr;
    QDoubleSpinBox* ptt_tail_ = nullptr;
    QPushButton* test_cat_ = nullptr;
    QPushButton* test_ptt_ = nullptr;

    // Folders
    QLineEdit* receive_dir_ = nullptr;
    QLineEdit* transmit_dir_ = nullptr;
    QLineEdit* template_dir_ = nullptr;

    // Transmit
    QLineEdit* callsign_ = nullptr;
    QCheckBox* optimize_ = nullptr;
    QCheckBox* cw_id_ = nullptr;
    QLineEdit* cw_message_ = nullptr;
    QCheckBox* vox_lead_ = nullptr;

    // Receive
    QCheckBox* auto_start_ = nullptr;
    QCheckBox* autosave_ = nullptr;
    QCheckBox* save_audio_ = nullptr;
    QCheckBox* low_cpu_ = nullptr;
    QCheckBox* blind_wide_ = nullptr;
    QComboBox* drift_track_ = nullptr;
    QLineEdit* filename_template_ = nullptr;
    QLabel* filename_preview_ = nullptr;
    QLineEdit* save_size_ = nullptr;
    QDoubleSpinBox* buffer_seconds_ = nullptr;
    QDoubleSpinBox* poll_interval_ = nullptr;
};

}  // namespace sstvae::gui

#endif

// The QRSS schedule window, opened by the "Schedule..." button beside
// Send.
//
// Three parts, in order of what the operator looks at most:
//
// - **Upcoming**, across the top: the schedule as a timeline of quarter
//   hours (gui/qrss_timeline.hpp), each send a block with its picture,
//   mode and repeat number. Click a block to select it and its Send; drag
//   it to another slot to move it; right-click it to remove it, change
//   its mode or set how soon it repeats. Clicking an empty slot with a
//   Send selected sets the form's start time to that slot.
// - **Sends**, on the left: pictures prepared for the schedule and kept
//   on disk (gui/qrss_sends.hpp). New and Open file
//   open the Editor (gui/qrss_editor.hpp); a Send on the schedule is
//   tinted, and selecting one tints its blocks on the timeline.
// - **Schedule**, on the right: mode, carrier, first quarter hour and
//   repeats for the selected Send, enabled once a Send is chosen, then
//   the list of what is scheduled, with Pause, Resume and Remove.
//
// Times are quarter hours in UTC, as everywhere else in QRSS, with the
// local time beside them. A scheduled entry keeps its own copy of the
// Send's picture, so every send of it has the same picture ID.

#ifndef SSTVAE_GUI_QRSS_SCHEDULE_WINDOW_HPP
#define SSTVAE_GUI_QRSS_SCHEDULE_WINDOW_HPP

#include <QDialog>
#include <QPushButton>
#include <QWidget>

#include <functional>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "images/types.hpp"
#include "overlay/model.hpp"
#include "qrss_editor.hpp"
#include "qrss_schedule.hpp"

class QComboBox;
class QDateTimeEdit;
class QDoubleSpinBox;
class QGroupBox;
class QLabel;
class QListWidget;
class QMenu;
class QPushButton;
class QScrollArea;
class QSpinBox;
class QTimer;

namespace sstvae::gui {

class QrssScheduler;
class QrssSends;
class QrssTimeline;

class InfoButton;
class QrssClockFace;
class SendPicture;

// Test / Tune (Jeff, 2026-10-15): a plain button until pressed, then
// wholly red with a bar inside it that runs down over the test, labelled
// "Halt Test" while the tone is on the air.
class TestTuneButton : public QPushButton {
    Q_OBJECT

public:
    explicit TestTuneButton(QWidget* parent = nullptr);
    // `remaining`: the fraction of the test still to go, 0..1.
    void set_running(bool running, double remaining = 1.0);
    bool running() const { return running_; }
    double remaining() const { return remaining_; }

protected:
    void paintEvent(QPaintEvent* event) override;

private:
    bool running_ = false;
    double remaining_ = 1.0;
};

// "Add frequency change" and its kin (Jeff, 2026-10-15): the time on a
// clock face like At Time's, in UTC or local time by a toggle, beside a
// field to type it in. The wheel steps 15 minutes, an hour with Shift and
// a minute with Ctrl.
class QrssFreqChangeDialog : public QDialog {
    Q_OBJECT

public:
    QrssFreqChangeDialog(const QString& title, double at, double dial_hz, bool with_time, bool utc,
                         QWidget* parent = nullptr);
    double at() const { return t_; }
    double dial_hz() const;
    bool utc() const { return utc_; }
    void set_utc(bool utc);
    void set_time(double t);
    QrssClockFace* face() const { return face_; }

private:
    void show_time();

    double t_ = 0.0;
    bool utc_ = true;
    bool syncing_ = false;
    QrssClockFace* face_ = nullptr;
    QDateTimeEdit* edit_ = nullptr;
    QLabel* zone_ = nullptr;
    QLabel* readout_ = nullptr;
    QPushButton* utc_button_ = nullptr;
    QPushButton* local_button_ = nullptr;
    QDoubleSpinBox* freq_ = nullptr;
};

class QrssScheduleWindow : public QWidget {
    Q_OBJECT

public:
    struct Defaults {
        std::string mode = "A";
        double freq_hz = 1500.0;
        double dial_hz = 0.0;   // the radio's dial now, 0 if unknown
        std::string waveform = "L";   // the default since 2026-10-12 (Jeff)
    };
    // How far ahead the timeline reaches.
    static constexpr double TIMELINE_HOURS = 48.0;

    QrssScheduleWindow(QrssScheduler* scheduler, QrssSends* sends,
                       std::function<Defaults()> defaults, QrssEditor::Context editor_context,
                       QWidget* parent = nullptr);

    // The form's first quarter hour, as Add would use it.
    double chosen_slot() const;
    void set_chosen_slot(double slot);
    // The selected Send, or empty.
    std::string selected_send() const;
    void select_send(const std::string& id);
    QrssTimeline* timeline() const { return timeline_; }

    // The Editor, modal: a new Send, a picture file, or an existing
    // Send. Each selects what was saved.
    void new_send();
    void new_from_file(const QString& path);
    void edit_send(const std::string& id);
    void delete_send(const std::string& id);

    // What the timeline's right-click menu does. Each returns empty on
    // success, else why not (also shown in the window).
    std::string remove_entry(const std::string& entry);
    std::string remove_from(const std::string& entry, int send);
    std::string change_mode(const std::string& entry, const std::string& mode);
    // The audio carrier frequency, every send of the entry.
    std::string change_offset(const std::string& entry, double hz);
    std::string change_waveform(const std::string& entry, const std::string& waveform);
    std::string repeat_every(const std::string& entry, int minutes);
    std::string move_send(const std::string& entry, int send, double slot);
    // The "Repeat in" choices for a mode, in minutes: never sooner than
    // its passes take (a mode B send lasts an hour).
    static std::vector<int> repeat_choices(const std::string& mode);
    // The menu a right-click on send `send` of `entry` opens. The caller
    // owns it.
    QMenu* menu_for(const std::string& entry, int send);
    // A Send dropped on the timeline at `slot`: scheduled there as the
    // form says. Empty on success, else why not (also shown).
    std::string drop_send(const std::string& send_id, double slot);
    // The entry selected on the timeline ("" for none).
    const std::string& selected_entry() const { return selected_entry_; }

    // Frequency changes (green on the timeline). Each returns empty on
    // success, else why not (also shown in the window).
    std::string add_freq_change(double at, double dial_hz);
    std::string move_freq_change(const std::string& id, double at);
    std::string retune_freq_change(const std::string& id, double dial_hz);
    void remove_freq_change(const std::string& id);
    // The menus a right-click on a change, and on empty time, open. The
    // caller owns them.
    QMenu* change_menu_for(const std::string& id);
    QMenu* empty_menu_for(double t);

    // What Test / Tune does: `start` keys the tone (empty when it
    // started, else why not), `halt` stops it.
    struct TestTune {
        std::function<std::string()> start;
        std::function<void()> halt;
    };
    void set_test_tune(TestTune hooks) { test_tune_ = std::move(hooks); }
    TestTuneButton* test_tune_button() const { return test_; }

public slots:
    void refresh();
    // The tone started, ticked or ended.
    void set_test_state(bool running, double remaining);

protected:
    void showEvent(QShowEvent* event) override;
    bool eventFilter(QObject* watched, QEvent* event) override;

private:
    void build();
    QWidget* build_sends(QWidget* parent);
    QWidget* build_form(QWidget* parent);
    void update_summary();
    // The clocks moved the first send by `seconds`, kept no earlier
    // than the earliest a send can be added now.
    void step_slot(int seconds);
    void sync_repeat();
    // Empty when added, else why not (shown in the summary and logged).
    std::string add();
    void refresh_sends();
    void refresh_timeline();
    void on_send_selected();
    void on_block_clicked(const QString& entry, int send);
    void select_entry(const std::string& entry);
    void form_from(const qrss_schedule::Entry& e);
    void show_result(const std::string& why, const QString& done);
    QString local_and_utc(double slot) const;
    void open_editor(const std::function<bool(QrssEditor&)>& prepare);
    // Asks for a change's time and dial frequency; false if cancelled.
    // Without the time, only the frequency is asked for.
    bool ask_freq_change(const QString& title, double* at, double* dial_hz,
                         bool with_time = true);

    QrssScheduler* scheduler_ = nullptr;
    QrssSends* sends_ = nullptr;
    std::function<Defaults()> defaults_;
    QrssEditor::Context editor_context_;
    std::string selected_entry_;

    QrssTimeline* timeline_ = nullptr;
    QScrollArea* timeline_scroll_ = nullptr;
    QLabel* message_ = nullptr;

    QListWidget* sends_list_ = nullptr;
    QPushButton* edit_send_ = nullptr;
    QPushButton* delete_send_ = nullptr;

    // The Send's choices on the left, At Time (the clocks) on the right,
    // Add under the right, level with the bottom of the left (Jeff,
    // 2026-10-12, from the mockup).
    QGroupBox* form_box_ = nullptr;
    QGroupBox* at_box_ = nullptr;
    QLabel* send_name_ = nullptr;
    SendPicture* picture_ = nullptr;
    QComboBox* mode_ = nullptr;
    QSpinBox* freq_ = nullptr;
    QComboBox* waveform_ = nullptr;
    InfoButton* mode_info_ = nullptr;      // (i) cards beside the choices
    InfoButton* waveform_info_ = nullptr;
    QSpinBox* times_ = nullptr;     // repeats after the first send: 0 = once
    QLabel* every_label_ = nullptr;
    QSpinBox* every_ = nullptr;     // minutes; at the mode's own length, back to back
    QLabel* every_unit_ = nullptr;
    QrssClockFace* clock_utc_ = nullptr;
    QrssClockFace* clock_local_ = nullptr;
    QLabel* utc_readout_ = nullptr;
    QLabel* local_caption_ = nullptr;
    QLabel* local_readout_ = nullptr;
    double chosen_slot_ = 0.0;
    QLabel* summary_ = nullptr;
    QPushButton* add_ = nullptr;
    TestTuneButton* test_ = nullptr;
    TestTune test_tune_;
    bool freq_utc_ = true;   // the frequency change dialog's last UTC/local choice
    QTimer* clock_timer_ = nullptr;
};

}  // namespace sstvae::gui

#endif

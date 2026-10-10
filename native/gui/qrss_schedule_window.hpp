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

#include <QWidget>

#include <functional>
#include <optional>
#include <string>
#include <vector>

#include "images/types.hpp"
#include "overlay/model.hpp"
#include "qrss_editor.hpp"

class QComboBox;
class QDateEdit;
class QGroupBox;
class QLabel;
class QListWidget;
class QMenu;
class QPushButton;
class QScrollArea;
class QSpinBox;
class QTimer;
class QTreeWidget;

namespace sstvae::gui {

class QrssScheduler;
class QrssSends;
class QrssTimeline;

class QrssScheduleWindow : public QWidget {
    Q_OBJECT

public:
    struct Defaults {
        std::string mode = "A";
        double freq_hz = 1500.0;
        double dial_hz = 0.0;   // the radio's dial now, 0 if unknown
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
    std::string repeat_every(const std::string& entry, int minutes);
    std::string move_send(const std::string& entry, int send, double slot);
    // The "Repeat in" choices for a mode, in minutes: never sooner than
    // its passes take (a mode B send lasts an hour).
    static std::vector<int> repeat_choices(const std::string& mode);
    // The menu a right-click on send `send` of `entry` opens. The caller
    // owns it.
    QMenu* menu_for(const std::string& entry, int send);

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

public slots:
    void refresh();

protected:
    void showEvent(QShowEvent* event) override;
    bool eventFilter(QObject* watched, QEvent* event) override;

private:
    void build();
    QWidget* build_sends(QWidget* parent);
    QWidget* build_form(QWidget* parent);
    void update_summary();
    void fill_times();
    void add();
    void set_selected_enabled(bool on);
    void remove_selected();
    void refresh_sends();
    void refresh_timeline();
    void on_send_selected();
    void on_block_clicked(const QString& entry, int send);
    void select_entry(const std::string& entry);
    void show_result(const std::string& why, const QString& done);
    QString local_and_utc(double slot) const;
    void open_editor(const std::function<bool(QrssEditor&)>& prepare);
    // Asks for a change's time and dial frequency; false if cancelled.
    bool ask_freq_change(const QString& title, double* at, double* dial_hz);

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

    QGroupBox* form_box_ = nullptr;
    QLabel* thumb_ = nullptr;
    QLabel* send_name_ = nullptr;
    QComboBox* mode_ = nullptr;
    QSpinBox* freq_ = nullptr;
    QDateEdit* date_ = nullptr;
    QComboBox* time_ = nullptr;
    QSpinBox* count_ = nullptr;
    QComboBox* every_ = nullptr;
    QLabel* summary_ = nullptr;
    QPushButton* add_ = nullptr;
    QTreeWidget* table_ = nullptr;
    QPushButton* pause_ = nullptr;
    QPushButton* resume_ = nullptr;
    QPushButton* remove_ = nullptr;
    QTimer* clock_timer_ = nullptr;
};

}  // namespace sstvae::gui

#endif

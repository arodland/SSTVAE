// Two buttons on a received QRSSTVAE picture (Jeff, 2026-10-15, from the
// "Improve and Find Buttons" mockup): **Improve**, the leave-one-out EM
// refine, and **Find missed passes**, the retroactive search of the 48 h
// passband store. Both are the C++ ports behind qrss/em_api.hpp, run in
// this process on a worker thread against the listener's own store
// (they take no lock the listener holds, so it keeps listening).
//
// `Controller` is one per QRSS window and keeps one run at a time per
// picture: pressing the other button while one runs queues it, so Find
// then Improve (or the reverse) runs back to back. When a run ends the
// picture is drawn again from the store (the codec, as the listener draws
// it) to a file of the app's own beside the listener's tiles, and the
// tile shows that until the listener draws the picture itself again.
//
// `EmBar` is the widget: the two buttons, each with a (?) card saying what
// it does, when to use it and whether it can run here (greyed out with
// the reason when it cannot), then a progress line with Cancel while one
// runs, then a line for each run's result. A tile has one and the picture
// viewer gets another for the same picture.

#ifndef SSTVAE_GUI_QRSS_EM_HPP
#define SSTVAE_GUI_QRSS_EM_HPP

#include <QHash>
#include <QObject>
#include <QString>
#include <QWidget>

#include <atomic>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <optional>
#include <thread>
#include <utility>

class QLabel;
class QProgressBar;
class QPushButton;

namespace sstvae::qrss::em {
struct Hooks;
}

namespace sstvae::gui {

class InfoButton;

namespace qrss_em {

enum class Kind { Improve, Find };

// A picture, as a tile of state.json describes it.
struct Target {
    QString callsign;
    QString picture_id;   // 8 hex digits
    QString mode;         // A, B, C
    QString frame;        // the tile's frame: "full", "l-full", ...
    QString waveform;     // "CE" or "L"
    QString status;       // receiving, finishing, complete, lost
    int passes = 1;
    // A picture the store holds: a callsign and a picture ID.
    bool valid() const { return !callsign.isEmpty() && !picture_id.isEmpty(); }
    // "<CALL>_<pid>", the store's name for it.
    QString key() const { return callsign + QLatin1Char('_') + picture_id; }
};

// What a finished run says.
struct Outcome {
    QString line;                      // "Improved: +0.12 dB over 3 rounds", or the error
    bool ok = true;                    // false: the line is an error
    QString image;                     // the picture drawn again ("" when not)
    std::optional<double> mean_w_db;   // the picture's mean weight now
};

// Runs one job, on a worker thread; `hooks` carries progress and cancel.
using Runner = std::function<Outcome(Kind, const Target&, const qrss::em::Hooks& hooks)>;

// Where the CE search looks: the tile's frame without its "l-".
QString ce_frame(const QString& frame);

class Controller : public QObject {
    Q_OBJECT

public:
    // `state_dir`: the listener's tile directory, where the redrawn
    // pictures go (STATE/em/).
    explicit Controller(QString state_dir, QObject* parent = nullptr);
    ~Controller() override;   // cancels what runs and waits for it

    void set_state_dir(const QString& dir) { state_dir_ = dir; }
    // The listener's store: store::default_root(), as the listener's own default.
    QString store_root() const { return store_root_; }
    void set_store_root(const QString& root) { store_root_ = root; }
    QString passband_dir() const;
    // For tests: what runs a job (default: em_api and the codec).
    void set_runner(Runner runner) { runner_ = std::move(runner); }

    // "" when `kind` can run on `t`, else why not, for the card.
    QString unavailable(Kind kind, const Target& t) const;

    struct State {
        std::optional<Kind> running;
        std::deque<Kind> queued;
        std::int64_t done = 0;
        std::int64_t total = 0;
        QString what;                 // the progress line
        QString improve_line, find_line;
        bool improve_ok = true, find_ok = true;
        QString image;                // the latest redrawn picture
        std::optional<double> mean_w_db;
    };
    // The picture's state, or null before anything was pressed for it.
    const State* state(const QString& key) const;

    // Run `kind` now, or after the run in progress for the same picture.
    // False when it cannot run (unavailable) or is already running or queued.
    bool start(Kind kind, const Target& t);
    // Stop the picture's run (what it finished is kept) and drop its queue.
    void cancel(const QString& key);
    bool busy() const { return !jobs_.isEmpty(); }

signals:
    // Anything about the picture changed: progress, a result, a picture.
    void changed(const QString& key);

private:
    struct Job {
        Kind kind;
        Target target;
        std::shared_ptr<std::atomic<bool>> cancel;
        std::thread thread;
    };
    void launch(const QString& key, Kind kind, const Target& t);
    void finished(const QString& key, Kind kind, const Outcome& out);
    Outcome run_default(Kind kind, const Target& t, const qrss::em::Hooks& hooks) const;

    QString state_dir_;
    QString store_root_;
    Runner runner_;
    QHash<QString, State> states_;
    QHash<QString, Target> targets_;
    QHash<QString, std::shared_ptr<Job>> jobs_;
};

class EmBar : public QWidget {
    Q_OBJECT

public:
    explicit EmBar(Controller* controller, QWidget* parent = nullptr);
    // The picture this bar is for; hidden while `t` is not valid.
    void set_target(const Target& t);
    const Target& target() const { return target_; }

    QPushButton* improve_button() const { return improve_; }
    QPushButton* find_button() const { return find_; }
    InfoButton* improve_info() const { return improve_info_; }
    InfoButton* find_info() const { return find_info_; }
    QPushButton* cancel_button() const { return cancel_; }
    QLabel* progress_label() const { return work_; }
    QLabel* result_label() const { return result_; }
    // The two cards' text now.
    QString card(Kind kind) const;

signals:
    // The picture was drawn again: the file.
    void pictureChanged(const QString& path, std::optional<double> mean_w_db);

private:
    void refresh();

    Controller* controller_;
    Target target_;
    QWidget* actions_ = nullptr;
    QPushButton* improve_ = nullptr;
    QPushButton* find_ = nullptr;
    InfoButton* improve_info_ = nullptr;
    InfoButton* find_info_ = nullptr;
    QWidget* working_ = nullptr;
    QLabel* work_ = nullptr;
    QPushButton* cancel_ = nullptr;
    QProgressBar* bar_ = nullptr;
    QLabel* result_ = nullptr;
    QString shown_image_;
};

}  // namespace qrss_em
}  // namespace sstvae::gui

#endif

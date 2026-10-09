// The QRSS window: every QRSSTVAE signal in the passband, as a tile
// whose picture fills in while its half-hour pass arrives.
//
// The QRSSTVAE receiver is Python (`sstvae/qrss/`, `qrss_listen.py`) and
// is not ported; this window is its front end, in two halves that are
// deliberately independent:
//
//   * **Feed.** "Start listener" runs `qrss_listen.py` as a child
//     process and pipes this app's own capture audio into it: every
//     sample the receive pane's ring buffer gets, once, as float32 at
//     8 kHz (`RingBuffer::read_since`). So QRSS hears exactly what the
//     receive pane hears, through one sound-card stream, and stops
//     hearing while the app transmits (half duplex replaces the ring;
//     the listener sees a hole in time and erases it). Nothing flows
//     unless the receive pane is listening.
//   * **Tiles.** The listener writes `state.json` and `tiles/*.png` to a
//     directory; the window polls that file and shows what it says. It
//     does so whether or not it started the listener, so one run from a
//     terminal (`arecord ... | qrss_listen.py`) shows up here too.
//
// Above the modem, so no parity claim: the tile layout is this file's
// own. What has a right answer is tested in `test_qrss_window.cpp`: the
// audio reaches the child sample for sample, and the tiles say what the
// state file says.

#ifndef SSTVAE_GUI_QRSS_WINDOW_HPP
#define SSTVAE_GUI_QRSS_WINDOW_HPP

#include <QHash>
#include <QPointer>
#include <QProcess>
#include <QString>
#include <QStringList>
#include <QWidget>

#include <cstdint>
#include <memory>

class QCloseEvent;
class QJsonObject;
class QLabel;
class QLineEdit;
class QProgressBar;
class QPlainTextEdit;
class QPushButton;
class QScrollArea;
class QTimer;

namespace sstvae::rx {
class RingBuffer;
}

namespace sstvae::gui {

class FlowLayout;
class QrssTile;

class QrssWindow : public QWidget {
    Q_OBJECT

public:
    explicit QrssWindow(QWidget* parent = nullptr);
    ~QrssWindow() override;

    // The receive pane's capture ring, or null when it is not listening.
    // Called by ReceivePanel wherever it hands the waterfall its ring.
    void set_ring(std::shared_ptr<rx::RingBuffer> ring);

    // Where the listener writes its tiles. Default: $QRSSTVAE_HOME/live,
    // else the platform's generic data directory + /qrsstvae/live (on
    // Linux ~/.local/share/qrsstvae/live, the listener's own default).
    QString state_dir() const { return state_dir_; }
    void set_state_dir(const QString& dir);
    static QString default_state_dir();

    // The command that starts the listener, before `--state DIR` is
    // appended: $SSTVAE_QRSS_LISTEN if set, else a Python (the repo's
    // .venv if there is one, else python3) and qrss_listen.py, looked
    // for beside the executable and up to four directories above it.
    static QString default_command();
    QString command() const;
    void set_command(const QString& command);

    bool listener_running() const;
    int tile_count() const { return tiles_.size(); }
    // The tile for a listener tile id, or null.
    QrssTile* tile(const QString& id) const { return tiles_.value(id, nullptr); }

public slots:
    void start_listener();
    void stop_listener();
    // Re-read state.json now (the timer does this every second).
    void reload();
    // Move whatever the ring has gained to the listener (the timer does
    // this every 250 ms). Public so a test need not wait for it.
    void pump_audio();

signals:
    void listenerStateChanged(bool running);

protected:
    void closeEvent(QCloseEvent* event) override;

private:
    void update_header();
    void append_log(const QString& text);
    void on_finished(int code, QProcess::ExitStatus status);

    QString state_dir_;
    QLineEdit* command_edit_ = nullptr;
    QPushButton* start_button_ = nullptr;
    QPushButton* stop_button_ = nullptr;
    QLabel* status_label_ = nullptr;
    QLabel* empty_label_ = nullptr;
    QScrollArea* scroll_ = nullptr;
    QWidget* tiles_host_ = nullptr;
    FlowLayout* flow_ = nullptr;
    QPlainTextEdit* log_ = nullptr;
    QTimer* poll_timer_ = nullptr;
    QTimer* feed_timer_ = nullptr;
    QPointer<QProcess> proc_;
    QHash<QString, QrssTile*> tiles_;
    QString stderr_rest_;

    std::shared_ptr<rx::RingBuffer> ring_;
    std::uint64_t fed_ = 0;            // ring samples already sent
    qint64 samples_sent_ = 0;
    qint64 samples_dropped_ = 0;
    qint64 state_mtime_ = -1;
    QString state_summary_;
};

// One signal: picture, what it is, how far through its pass.
class QrssTile : public QWidget {
    Q_OBJECT

public:
    explicit QrssTile(QWidget* parent = nullptr);
    // Apply one tile object from state.json; `dir` resolves its image.
    void update_from(const QJsonObject& tile, const QString& dir);
    QString title() const;
    QString details() const;
    bool has_picture() const;
    int progress_percent() const;

private:
    QLabel* picture_ = nullptr;
    QLabel* title_ = nullptr;
    QLabel* details_ = nullptr;
    QProgressBar* progress_ = nullptr;
    int image_rev_ = -1;
    QString image_path_;
};

}  // namespace sstvae::gui

#endif

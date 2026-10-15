// The QRSS window: every QRSSTVAE signal in the passband, as a tile
// whose picture fills in while its half-hour pass arrives.
//
// The QRSSTVAE receiver is a separate process: `sstvae-qrss listen`, the
// C++ port, or `qrss_listen.py`, the Python it was ported from (same
// arguments, same files). This window is its front end, in two halves
// that are deliberately independent:
//
//   * **Feed.** "Start listener" runs the listener as a child
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
#include <optional>
#include <vector>

#include "qrss_spectrogram.hpp"

class QCloseEvent;
class QJsonObject;
class QLabel;
class QLineEdit;
class QVBoxLayout;
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
namespace qrss_em {
class Controller;
class EmBar;
}  // namespace qrss_em

class QrssWindow : public QWidget {
    Q_OBJECT

public:
    // A window of its own by default; the main window embeds it in its
    // QRSSTVAE mode with `Qt::Widget`.
    explicit QrssWindow(QWidget* parent = nullptr, Qt::WindowFlags flags = Qt::Window);
    ~QrssWindow() override;

    // The receive pane's capture ring, or null when it is not listening.
    // Called by ReceivePanel wherever it hands the waterfall its ring.
    void set_ring(std::shared_ptr<rx::RingBuffer> ring);
    // The QRSSTVAE mode's spectrogram: handed the same ring, and the
    // tiles' frequencies and passes to draw in red.
    void set_spectrogram(QrssSpectrogram* spectrogram);
    // The two tile columns, for tests.
    QWidget* provisional_area() const { return prov_host_; }
    QWidget* received_area() const { return tiles_host_; }

    // Where the listener writes its tiles. Default: $QRSSTVAE_HOME/live,
    // else the platform's generic data directory + /qrsstvae/live (on
    // Linux ~/.local/share/qrsstvae/live, the listener's own default).
    QString state_dir() const { return state_dir_; }
    void set_state_dir(const QString& dir);
    static QString default_state_dir();

    // The command that starts the listener, before `--state DIR` is
    // appended: $SSTVAE_QRSS_LISTEN if set, else `sstvae-qrss listen`
    // (unless SSTVAE_QRSS_BACKEND=python), else a Python (the repo's
    // .venv if there is one, else python3) and qrss_listen.py; each
    // looked for beside the executable and up to four directories above.
    static QString default_command();
    QString command() const;
    void set_command(const QString& command);

    // The listener's command row and its output log. Shown in a window
    // of its own; the QRSSTVAE mode hides them behind a toggle.
    void set_details_visible(bool on);
    // Off in SSTVAE mode: the listener this window started is stopped
    // and nothing polls or feeds it. On again: polling resumes from the
    // audio arriving now (the main window restarts the listener).
    void set_active(bool on);
    bool active() const { return active_; }
    bool details_visible() const;

    bool listener_running() const;
    int tile_count() const { return tiles_.size(); }
    // The tile for a listener tile id, or null.
    QrssTile* tile(const QString& id) const { return tiles_.value(id, nullptr); }
    // What runs the tiles' Improve and Find missed passes (qrss_em.hpp).
    qrss_em::Controller* em() const { return em_; }

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
    void update_markers(std::vector<QrssSpectrogram::Marker> markers);

    QString state_dir_;
    QWidget* listener_row_ = nullptr;
    bool active_ = true;
    QLineEdit* command_edit_ = nullptr;
    QPushButton* start_button_ = nullptr;
    QPushButton* stop_button_ = nullptr;
    QLabel* status_label_ = nullptr;
    QLabel* empty_label_ = nullptr;
    QScrollArea* scroll_ = nullptr;
    QWidget* tiles_host_ = nullptr;
    FlowLayout* flow_ = nullptr;
    QScrollArea* prov_scroll_ = nullptr;
    QWidget* prov_host_ = nullptr;
    QVBoxLayout* prov_list_ = nullptr;
    QLabel* prov_empty_ = nullptr;
    QPointer<QrssSpectrogram> spectrogram_;
    std::vector<QrssSpectrogram::Marker> markers_;
    QPlainTextEdit* log_ = nullptr;
    QTimer* poll_timer_ = nullptr;
    QTimer* feed_timer_ = nullptr;
    QPointer<QProcess> proc_;
    QHash<QString, QrssTile*> tiles_;
    qrss_em::Controller* em_ = nullptr;
    QString stderr_rest_;

    std::shared_ptr<rx::RingBuffer> ring_;
    std::uint64_t fed_ = 0;            // ring samples already sent
    qint64 samples_sent_ = 0;
    qint64 samples_dropped_ = 0;
    qint64 state_mtime_ = -1;
    QString state_summary_;
};

// How far through its pass a signal is, and how well each stretch of it
// came through, in one bar under the title (Jeff, 2026-10-10). Its
// filled length is the share of the pass so far; each stretch is
// coloured by the listener's `confidence`, the per-latent SNR in dB
// (receiver.confidence_db): black at -10 dB or less, so a stretch whose
// latents are missing (not heard, or erased) is black, then blue at -5,
// red at 0, yellow at +5 and white at +10 dB or more. The preamble,
// which has no latents, is grey; what is still to come is empty.
class PassBar : public QWidget {
public:
    explicit PassBar(QWidget* parent = nullptr);
    void set(const std::vector<std::optional<double>>& bins, double progress);
    double progress() const { return progress_; }
    // Each bin up to `progress`: true where latents are missing (black).
    const std::vector<bool>& holes() const { return holes_; }
    // The colour for a per-latent SNR, and for the preamble.
    static QColor color(double snr_db);
    static QColor unmeasured();

protected:
    void paintEvent(QPaintEvent* event) override;

private:
    std::vector<bool> holes_;
    std::vector<std::optional<double>> bins_;
    double progress_ = 0.0;
};

// One signal: picture, what it is, how far through its pass.
class QrssTile : public QWidget {
    Q_OBJECT

public:
    static constexpr int WIDTH = 252;
    // `em`: what runs the picture's Improve and Find missed passes (none
    // without one).
    explicit QrssTile(QWidget* parent = nullptr, qrss_em::Controller* em = nullptr);
    // No header yet: the picture is a guess (the Provisional column).
    bool provisional() const { return provisional_; }
    // Apply one tile object from state.json; `dir` resolves its image.
    void update_from(const QJsonObject& tile, const QString& dir);
    QString title() const;
    // The mean weight shown at the right of the title ("" when unknown).
    QString weight_text() const;
    // Everything under the bar, the note included.
    QString details() const;
    // The listener's note, in plain words (smaller print).
    QString note() const;
    bool has_picture() const;
    int progress_percent() const;
    PassBar* pass_bar() const { return progress_; }
    // The picture file shown, at its full size ("" before one).
    QString image_path() const { return image_path_; }
    // Improve and Find missed passes, under the lines; null without a controller.
    qrss_em::EmBar* em_bar() const { return em_bar_; }

protected:
    // Opens the picture in a viewer to zoom and pan (image_viewer.hpp).
    void mouseDoubleClickEvent(QMouseEvent* event) override;

private:
    QLabel* picture_ = nullptr;
    QLabel* title_ = nullptr;
    QLabel* weight_ = nullptr;
    QLabel* details_ = nullptr;
    QLabel* note_ = nullptr;
    PassBar* progress_ = nullptr;
    int image_rev_ = -1;
    bool provisional_ = true;
    QString image_path_;
    qrss_em::Controller* em_ = nullptr;
    qrss_em::EmBar* em_bar_ = nullptr;
    // The listener's image_rev when the app last drew the picture again
    // (Improve, Find): until the listener draws it itself, the app's stays.
    int em_rev_ = -1;
};

}  // namespace sstvae::gui

#endif

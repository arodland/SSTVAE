#include "qrss_window.hpp"

#include <QCloseEvent>
#include <QCoreApplication>
#include <QDateTime>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QFont>
#include <QHBoxLayout>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QLabel>
#include <QLineEdit>
#include <QMouseEvent>
#include <QPainter>
#include <QPixmap>
#include <QPlainTextEdit>
#include <QProcessEnvironment>
#include <QProgressBar>
#include <QPushButton>
#include <QScrollArea>
#include <QSet>
#include <QStandardPaths>
#include <QTimer>
#include <QVBoxLayout>

#include <algorithm>
#include <cmath>
#include <iterator>
#include <optional>
#include <utility>
#include <vector>

#include "flow_layout.hpp"
#include "image_viewer.hpp"
#include "qrss_spectrogram.hpp"
#include "rx/ringbuffer.hpp"

namespace sstvae::gui {

namespace {

constexpr int PICTURE_W = 240;
constexpr int PICTURE_H = 180;
// Audio the child has not read yet, beyond which we stop queueing it:
// 4 MB is two minutes at 8 kHz float32. Past that the listener is not
// reading at all, and an unbounded queue would only grow this process.
constexpr qint64 MAX_BACKLOG_BYTES = 4 << 20;
constexpr int LOG_LINES = 400;
// A QRSS pass, from its quarter hour: what a tile's red covers.
constexpr double PASS_S = 1782.7;
// What the listener appends to a headerless tile's note; the Provisional
// column says it instead.
const QString GUESS_NOTE =
    QStringLiteral("no header yet: picture assumes the first pass of a mode A send");

QString number_or(const QJsonValue& v, int decimals, const QString& unit,
                  const QString& none = QStringLiteral("?")) {
    if (!v.isDouble()) return none;
    return QString::number(v.toDouble(), 'f', decimals) + unit;
}

// The inverse of QProcess::splitCommand, for showing a command line.
QString join_command(const QStringList& args) {
    QStringList out;
    for (QString a : args) {
        if (a.isEmpty() || a.contains(QLatin1Char(' ')) || a.contains(QLatin1Char('"'))) {
            a.replace(QLatin1Char('"'), QStringLiteral("\"\"\""));
            a = QLatin1Char('"') + a + QLatin1Char('"');
        }
        out << a;
    }
    return out.join(QLatin1Char(' '));
}

QString hhmm(const QString& iso) {
    // "2026-10-09T06:00:00Z" -> "06:00Z"; anything else as it is.
    return iso.size() >= 16 ? iso.mid(11, 5) + QStringLiteral("Z") : iso;
}

}  // namespace

// --- confidence band ---------------------------------------------------------------------

namespace {

constexpr int BAND_H = 2;

struct Stop {
    double db;
    int r, g, b;
};
constexpr Stop BAND_STOPS[] = {
    {-10.0, 0, 0, 0},       // no better than noise
    {-5.0, 0, 40, 255},     // low
    {0.0, 230, 0, 0},
    {5.0, 255, 220, 0},
    {10.0, 255, 255, 255},  // high
};

}  // namespace

ConfidenceBand::ConfidenceBand(QWidget* parent) : QWidget(parent) {
    setFixedSize(PICTURE_W, BAND_H);
    setToolTip(tr("Confidence through the pass, left to right: the SNR of each stretch's "
                  "picture numbers. Black is no better than noise (-10 dB or less), then "
                  "blue, red, yellow, and white at +10 dB or more. Grey: nothing to measure "
                  "there (the preamble, or audio not heard)."));
}

void ConfidenceBand::set(std::vector<std::optional<double>> bins, double progress) {
    bins_ = std::move(bins);
    progress_ = progress;
    update();
}

QColor ConfidenceBand::color(double snr_db) {
    const auto& lo = BAND_STOPS[0];
    const auto& hi = BAND_STOPS[std::size(BAND_STOPS) - 1];
    if (!(snr_db > lo.db)) return QColor(lo.r, lo.g, lo.b);
    if (snr_db >= hi.db) return QColor(hi.r, hi.g, hi.b);
    for (std::size_t i = 1; i < std::size(BAND_STOPS); ++i) {
        const Stop& a = BAND_STOPS[i - 1];
        const Stop& b = BAND_STOPS[i];
        if (snr_db <= b.db) {
            const double f = (snr_db - a.db) / (b.db - a.db);
            auto mix = [f](int x, int y) {
                return static_cast<int>(std::lround(x + (y - x) * f));
            };
            return QColor(mix(a.r, b.r), mix(a.g, b.g), mix(a.b, b.b));
        }
    }
    return QColor(hi.r, hi.g, hi.b);
}

QColor ConfidenceBand::unmeasured() { return QColor(128, 128, 128); }

void ConfidenceBand::paintEvent(QPaintEvent*) {
    const int n = static_cast<int>(bins_.size());
    if (n == 0) return;
    QPainter painter(this);
    for (int i = 0; i < n; ++i) {
        const int x0 = static_cast<int>(std::lround(static_cast<double>(i) * width() / n));
        const int x1 = static_cast<int>(std::lround(static_cast<double>(i + 1) * width() / n));
        if (x1 <= x0) continue;
        const std::optional<double>& v = bins_[static_cast<std::size_t>(i)];
        if (v) {
            painter.fillRect(x0, 0, x1 - x0, height(), color(*v));
        } else if (static_cast<double>(i) / n < progress_) {
            painter.fillRect(x0, 0, x1 - x0, height(), unmeasured());
        }
    }
}

// --- tile -------------------------------------------------------------------------------

QrssTile::QrssTile(QWidget* parent) : QWidget(parent) {
    auto* box = new QVBoxLayout(this);
    box->setContentsMargins(6, 6, 6, 6);
    box->setSpacing(3);
    picture_ = new QLabel(this);
    picture_->setFixedSize(PICTURE_W, PICTURE_H);
    picture_->setAlignment(Qt::AlignCenter);
    picture_->setFrameShape(QFrame::Box);
    picture_->setText(tr("Waiting for the header"));
    picture_->setWordWrap(true);
    band_ = new ConfidenceBand(this);
    // Small print under the picture, as Glissando has it: the picture
    // is what the tile is for.
    QFont small = font();
    small.setPointSizeF(std::max(6.5, small.pointSizeF() * 0.78));
    title_ = new QLabel(this);
    QFont bold = font();
    bold.setPointSizeF(std::max(7.0, bold.pointSizeF() * 0.88));
    bold.setBold(true);
    title_->setFont(bold);
    details_ = new QLabel(this);
    details_->setFont(small);
    details_->setWordWrap(true);
    details_->setFixedWidth(PICTURE_W);
    details_->setTextInteractionFlags(Qt::TextSelectableByMouse);
    progress_ = new QProgressBar(this);
    progress_->setRange(0, 100);
    progress_->setFixedWidth(PICTURE_W);
    // The text goes beside the bar, not on it: drawn on the bar, the
    // fill's edge splits it ("55% of th|e pass") on most styles.
    progress_->setTextVisible(false);
    progress_->setFixedHeight(8);
    progress_text_ = new QLabel(this);
    progress_text_->setFont(small);
    // The band sits right under the picture it describes.
    auto* picture_and_band = new QVBoxLayout;
    picture_and_band->setSpacing(1);
    picture_and_band->addWidget(picture_);
    picture_and_band->addWidget(band_);
    box->addLayout(picture_and_band);
    box->addWidget(title_);
    box->addWidget(progress_text_);
    box->addWidget(progress_);
    box->addWidget(details_);
    box->addStretch(1);
    setFixedWidth(WIDTH);
}

void QrssTile::update_from(const QJsonObject& t, const QString& dir) {
    const QString call = t.value(QStringLiteral("callsign")).toString();
    const QString grid = t.value(QStringLiteral("grid")).toString();
    const double f_hz = t.value(QStringLiteral("f_hz")).toDouble();
    const QString status = t.value(QStringLiteral("status")).toString();
    if (!call.isEmpty()) {
        title_->setText(grid.isEmpty() ? call : call + QStringLiteral("  ") + grid);
    } else {
        title_->setText(tr("Signal at %1 Hz").arg(f_hz, 0, 'f', 1));
    }

    const double progress = t.value(QStringLiteral("progress")).toDouble();
    progress_->setValue(static_cast<int>(std::lround(100.0 * std::clamp(progress, 0.0, 1.0))));
    progress_text_->setText(tr("%1% of the pass").arg(progress_->value()));
    std::vector<std::optional<double>> bins;
    for (const QJsonValue& v : t.value(QStringLiteral("confidence")).toArray()) {
        bins.push_back(v.isDouble() ? std::optional<double>(v.toDouble()) : std::nullopt);
    }
    band_->set(std::move(bins), std::clamp(progress, 0.0, 1.0));

    QStringList lines;
    lines << tr("Slot %1 · %2 Hz · SNR %3")
                 .arg(hhmm(t.value(QStringLiteral("slot_utc")).toString()))
                 .arg(f_hz, 0, 'f', 1)
                 .arg(number_or(t.value(QStringLiteral("snr_db")), 1, tr(" dB")));
    // On the air: the dial the listener heard it on plus its audio
    // frequency (USB). Unknown dial (null): nothing.
    if (const QJsonValue dial = t.value(QStringLiteral("dial_hz")); dial.isDouble() && f_hz > 0) {
        lines << tr("RF %1 MHz (dial %2)")
                     .arg((dial.toDouble() + f_hz) / 1e6, 0, 'f', 6)
                     .arg(dial.toDouble() / 1e6, 0, 'f', 6);
    }
    const QString pid = t.value(QStringLiteral("picture_id")).toString();
    if (!pid.isEmpty()) {
        const int passes = t.value(QStringLiteral("passes")).toInt(1);
        lines << tr("Picture %1, mode %2 · %3")
                     .arg(pid, t.value(QStringLiteral("mode")).toString(),
                          passes == 1 ? tr("1 pass") : tr("%1 passes combined").arg(passes));
    }
    const QString cw = t.value(QStringLiteral("cw_text")).toString();
    if (!cw.isEmpty() && cw != call) lines << tr("Morse ID read: %1").arg(cw);
    const double received = t.value(QStringLiteral("received")).toDouble();
    const double heard = t.value(QStringLiteral("heard")).toDouble();
    QString state;
    if (status == QStringLiteral("complete")) {
        state = tr("Complete");
    } else if (status == QStringLiteral("lost")) {
        state = tr("Lost (not found at the end of the pass)");
    } else {
        state = tr("Receiving");
    }
    lines << tr("%1 · %2% of latents · heard %3% of the pass")
                 .arg(state)
                 .arg(static_cast<int>(std::lround(100.0 * received)))
                 .arg(static_cast<int>(std::lround(100.0 * heard)));
    const QString wdb = number_or(t.value(QStringLiteral("mean_w_db")), 1, tr(" dB"), QString());
    if (!wdb.isEmpty()) lines << tr("Mean weight %1").arg(wdb);
    QString note = t.value(QStringLiteral("note")).toString();
    note.remove(QStringLiteral("; ") + GUESS_NOTE).remove(GUESS_NOTE);
    if (!note.isEmpty()) lines << note;
    provisional_ = pid.isEmpty();
    details_->setText(lines.join(QLatin1Char('\n')));

    const QString image = t.value(QStringLiteral("image")).toString();
    const int rev = t.value(QStringLiteral("image_rev")).toInt();
    if (!image.isEmpty()) {
        const QString path = QDir(dir).filePath(image);
        if (rev != image_rev_ || path != image_path_) {
            QPixmap px(path);
            if (!px.isNull()) {
                picture_->setPixmap(px.scaled(PICTURE_W, PICTURE_H, Qt::KeepAspectRatio,
                                              Qt::SmoothTransformation));
                image_rev_ = rev;
                image_path_ = path;
            }
        }
    } else if (image_rev_ < 0) {
        const bool header = !t.value(QStringLiteral("picture_id")).toString().isEmpty();
        picture_->setText(header ? tr("Header decoded; picture next refresh")
                          : status == QStringLiteral("receiving")
                              ? tr("Waiting for the header")
                              : tr("No header decoded: nowhere to place the picture"));
    }
}

QString QrssTile::title() const { return title_->text(); }
QString QrssTile::details() const { return details_->text(); }
bool QrssTile::has_picture() const {
    return !picture_->pixmap().isNull();
}
void QrssTile::mouseDoubleClickEvent(QMouseEvent* event) {
    if (event->button() == Qt::LeftButton && !image_path_.isEmpty()) {
        // From the file, at full size: the tile shows a reduction.
        open_image_viewer(QPixmap(image_path_), title_->text(), this);
        return;
    }
    QWidget::mouseDoubleClickEvent(event);
}

int QrssTile::progress_percent() const { return progress_->value(); }
QString QrssTile::progress_text() const { return progress_text_->text(); }

// --- window ----------------------------------------------------------------------------

QrssWindow::QrssWindow(QWidget* parent, Qt::WindowFlags flags) : QWidget(parent, flags) {
    setWindowTitle(tr("QRSS signals"));
    state_dir_ = default_state_dir();

    auto* box = new QVBoxLayout(this);

    // The listener's own controls: its command and a Start/Stop for the
    // process alone. On the QRSSTVAE mode one Start runs both the audio and
    // the listener, so these and the log are details the page hides.
    listener_row_ = new QWidget(this);
    listener_row_->setObjectName(QStringLiteral("qrss_listener_row"));
    auto* row = new QHBoxLayout(listener_row_);
    row->setContentsMargins(0, 0, 0, 0);
    row->addWidget(new QLabel(tr("Listener:"), this));
    command_edit_ = new QLineEdit(default_command(), this);
    command_edit_->setToolTip(
        tr("The command that runs qrss_listen.py; \"--state DIR\" is added. "
           "Set SSTVAE_QRSS_LISTEN to change the default."));
    row->addWidget(command_edit_, 1);
    start_button_ = new QPushButton(tr("Start listener"), this);
    stop_button_ = new QPushButton(tr("Stop"), this);
    stop_button_->setEnabled(false);
    row->addWidget(start_button_);
    row->addWidget(stop_button_);
    box->addWidget(listener_row_);

    status_label_ = new QLabel(this);
    status_label_->setWordWrap(true);
    status_label_->setTextInteractionFlags(Qt::TextSelectableByMouse);
    box->addWidget(status_label_);

    // Two columns of tiles. **Provisional**, at the far left: passes
    // heard without a header, whose picture is only a guess (the first
    // pass of a mode A send) until a header places it -- the column says
    // so once, instead of a note under every tile. Then **Received**:
    // everything placed by a header.
    auto* columns = new QHBoxLayout;
    columns->setSpacing(6);
    auto* prov_box = new QVBoxLayout;
    prov_box->setSpacing(2);
    auto* prov_title = new QLabel(tr("Provisional"), this);
    prov_title->setToolTip(tr("Passes heard without a header. The picture assumes the first "
                              "pass of a mode A send until a header (or a repeat) places it."));
    QFont heading = prov_title->font();
    heading.setBold(true);
    prov_title->setFont(heading);
    prov_box->addWidget(prov_title);
    prov_scroll_ = new QScrollArea(this);
    prov_scroll_->setObjectName(QStringLiteral("qrss_provisional"));
    prov_scroll_->setWidgetResizable(true);
    prov_scroll_->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    prov_scroll_->setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOn);
    prov_host_ = new QWidget(prov_scroll_);
    prov_list_ = new QVBoxLayout(prov_host_);
    prov_list_->setContentsMargins(0, 0, 0, 0);
    prov_list_->setSpacing(4);
    prov_empty_ = new QLabel(tr("None"), prov_host_);
    prov_empty_->setAlignment(Qt::AlignHCenter | Qt::AlignTop);
    prov_list_->addWidget(prov_empty_);
    prov_list_->addStretch(1);
    prov_scroll_->setWidget(prov_host_);
    prov_scroll_->setFixedWidth(QrssTile::WIDTH + 24);
    prov_box->addWidget(prov_scroll_, 1);
    columns->addLayout(prov_box);

    auto* recv_box = new QVBoxLayout;
    recv_box->setSpacing(2);
    auto* recv_title = new QLabel(tr("Received"), this);
    recv_title->setFont(heading);
    recv_box->addWidget(recv_title);
    scroll_ = new QScrollArea(this);
    scroll_->setObjectName(QStringLiteral("qrss_received"));
    scroll_->setWidgetResizable(true);
    scroll_->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    tiles_host_ = new QWidget(scroll_);
    flow_ = new FlowLayout(tiles_host_, 6, 8, 8);
    empty_label_ = new QLabel(
        tr("No QRSS signals yet. A signal appears here about 25 s after its "
           "quarter hour, once its preamble has been heard."),
        tiles_host_);
    empty_label_->setWordWrap(true);
    flow_->addWidget(empty_label_);
    scroll_->setWidget(tiles_host_);
    recv_box->addWidget(scroll_, 1);
    columns->addLayout(recv_box, 1);
    box->addLayout(columns, 1);

    log_ = new QPlainTextEdit(this);
    log_->setObjectName(QStringLiteral("qrss_log"));
    log_->setReadOnly(true);
    log_->setMaximumBlockCount(LOG_LINES);
    log_->setMaximumHeight(110);
    log_->setPlaceholderText(tr("Listener output"));
    box->addWidget(log_);

    connect(start_button_, &QPushButton::clicked, this, &QrssWindow::start_listener);
    connect(stop_button_, &QPushButton::clicked, this, &QrssWindow::stop_listener);

    poll_timer_ = new QTimer(this);
    poll_timer_->setInterval(1000);
    connect(poll_timer_, &QTimer::timeout, this, &QrssWindow::reload);
    poll_timer_->start();
    feed_timer_ = new QTimer(this);
    feed_timer_->setInterval(250);
    connect(feed_timer_, &QTimer::timeout, this, &QrssWindow::pump_audio);
    feed_timer_->start();

    resize(820, 640);
    update_header();
}

QrssWindow::~QrssWindow() {
    if (proc_) {
        // As stop_listener, but synchronously: the app is going away.
        proc_->disconnect(this);
        proc_->closeWriteChannel();
        if (!proc_->waitForFinished(3000)) {
            proc_->terminate();
            if (!proc_->waitForFinished(2000)) proc_->kill();
        }
    }
}

QString QrssWindow::default_state_dir() {
    const QByteArray home = qgetenv("QRSSTVAE_HOME");
    if (!home.isEmpty()) return QDir(QString::fromLocal8Bit(home)).filePath(QStringLiteral("live"));
    const QString base = QStandardPaths::writableLocation(QStandardPaths::GenericDataLocation);
    return QDir(base).filePath(QStringLiteral("qrsstvae/live"));
}

QString QrssWindow::default_command() {
    const QByteArray env = qgetenv("SSTVAE_QRSS_LISTEN");
    if (!env.isEmpty()) return QString::fromLocal8Bit(env);
    QDir dir(QCoreApplication::applicationDirPath());
    for (int up = 0; up <= 4; ++up) {
        const QString script = dir.filePath(QStringLiteral("qrss_listen.py"));
        if (QFileInfo::exists(script)) {
            QString python = QStringLiteral("python3");
            for (const char* venv : {".venv/bin/python", ".venv/Scripts/python.exe"}) {
                const QString p = dir.filePath(QString::fromLatin1(venv));
                if (QFileInfo::exists(p)) {
                    python = p;
                    break;
                }
            }
            return join_command({python, script});
        }
        if (!dir.cdUp()) break;
    }
    return QStringLiteral("python3 qrss_listen.py");
}

void QrssWindow::set_details_visible(bool on) {
    listener_row_->setVisible(on);
    log_->setVisible(on);
}

void QrssWindow::set_active(bool on) {
    if (on == active_) return;
    active_ = on;
    if (on) {
        // Skip what the ring gathered while off rather than send it late.
        if (ring_) fed_ = ring_->total_written();
        poll_timer_->start();
        feed_timer_->start();
        reload();
    } else {
        stop_listener();
        poll_timer_->stop();
        feed_timer_->stop();
    }
}

bool QrssWindow::details_visible() const { return !listener_row_->isHidden(); }

QString QrssWindow::command() const { return command_edit_->text(); }
void QrssWindow::set_command(const QString& command) { command_edit_->setText(command); }

void QrssWindow::set_state_dir(const QString& dir) {
    state_dir_ = dir;
    state_mtime_ = -1;
    reload();
}

bool QrssWindow::listener_running() const {
    return proc_ && proc_->state() != QProcess::NotRunning;
}

void QrssWindow::set_spectrogram(QrssSpectrogram* spectrogram) {
    spectrogram_ = spectrogram;
    if (spectrogram_) {
        spectrogram_->set_ring(ring_);
        spectrogram_->set_markers(markers_);
    }
}

void QrssWindow::update_markers(std::vector<QrssSpectrogram::Marker> markers) {
    markers_ = std::move(markers);
    if (spectrogram_) spectrogram_->set_markers(markers_);
}

void QrssWindow::set_ring(std::shared_ptr<rx::RingBuffer> ring) {
    ring_ = std::move(ring);
    if (spectrogram_) spectrogram_->set_ring(ring_);
    // A new ring counts from 0, and anything already in it was heard
    // before we knew about it; start from now rather than send a burst
    // of old audio stamped with the current time.
    fed_ = ring_ ? ring_->total_written() : 0;
    update_header();
}

void QrssWindow::start_listener() {
    if (listener_running()) return;
    QStringList args = QProcess::splitCommand(command());
    if (args.isEmpty()) {
        append_log(tr("No listener command."));
        return;
    }
    const QString program = args.takeFirst();
    args << QStringLiteral("--state") << state_dir_;
    QDir().mkpath(state_dir_);

    auto* proc = new QProcess(this);
    proc->setProcessChannelMode(QProcess::MergedChannels);
    QProcessEnvironment env = QProcessEnvironment::systemEnvironment();
    env.insert(QStringLiteral("PYTHONUNBUFFERED"), QStringLiteral("1"));
    proc->setProcessEnvironment(env);
    for (const QString& a : args) {
        if (a.endsWith(QStringLiteral("qrss_listen.py"))) {
            proc->setWorkingDirectory(QFileInfo(a).absolutePath());
            break;
        }
    }
    connect(proc, &QProcess::readyReadStandardOutput, this, [this, proc] {
        stderr_rest_ += QString::fromUtf8(proc->readAllStandardOutput());
        const int cut = stderr_rest_.lastIndexOf(QLatin1Char('\n'));
        if (cut < 0) return;
        for (const QString& line : stderr_rest_.left(cut).split(QLatin1Char('\n')))
            append_log(line);
        stderr_rest_ = stderr_rest_.mid(cut + 1);
    });
    connect(proc, &QProcess::finished, this, &QrssWindow::on_finished);
    connect(proc, &QProcess::errorOccurred, this, [this, proc](QProcess::ProcessError e) {
        if (e == QProcess::FailedToStart) {
            append_log(tr("Could not start %1: %2").arg(proc->program(), proc->errorString()));
            proc->deleteLater();
            update_header();
            emit listenerStateChanged(false);
        }
    });
    proc_ = proc;
    fed_ = ring_ ? ring_->total_written() : 0;
    samples_sent_ = samples_dropped_ = 0;
    append_log(tr("$ %1").arg(join_command(QStringList{program} + args)));
    proc->start(program, args);
    update_header();
    emit listenerStateChanged(true);
}

void QrssWindow::stop_listener() {
    if (!proc_) return;
    // Closing stdin is the listener's "audio ended": it writes its state
    // as not listening and exits. A terminate follows if it does not.
    proc_->closeWriteChannel();
    QPointer<QProcess> p = proc_;
    QTimer::singleShot(3000, this, [p] {
        if (p && p->state() != QProcess::NotRunning) p->terminate();
    });
    QTimer::singleShot(8000, this, [p] {
        if (p && p->state() != QProcess::NotRunning) p->kill();
    });
}

void QrssWindow::on_finished(int code, QProcess::ExitStatus status) {
    if (!stderr_rest_.isEmpty()) append_log(stderr_rest_);
    stderr_rest_.clear();
    append_log(status == QProcess::CrashExit ? tr("Listener stopped (crashed).")
                                             : tr("Listener exited with code %1.").arg(code));
    if (proc_) proc_->deleteLater();
    proc_ = nullptr;
    update_header();
    emit listenerStateChanged(false);
}

void QrssWindow::pump_audio() {
    if (!ring_) return;
    std::uint64_t total = 0;
    std::vector<double> x = ring_->read_since(fed_, &total);
    if (total < fed_) {
        fed_ = 0;  // a fresh ring: its count started over
        x = ring_->read_since(0, &total);
    }
    fed_ = total;
    // Starting counts: QProcess holds what is written until the child runs.
    if (x.empty() || !listener_running()) return;
    if (proc_->bytesToWrite() > MAX_BACKLOG_BYTES) {
        samples_dropped_ += static_cast<qint64>(x.size());
        return;
    }
    QByteArray bytes(static_cast<qsizetype>(x.size() * sizeof(float)), Qt::Uninitialized);
    auto* out = reinterpret_cast<float*>(bytes.data());
    for (std::size_t i = 0; i < x.size(); ++i) out[i] = static_cast<float>(x[i]);
    // float32 little-endian: every platform this app ships on is
    // little-endian, which the listener's --format f32 assumes.
    proc_->write(bytes);
    samples_sent_ += static_cast<qint64>(x.size());
}

void QrssWindow::append_log(const QString& text) {
    if (!text.isEmpty()) log_->appendPlainText(text);
}

void QrssWindow::reload() {
    const QString path = QDir(state_dir_).filePath(QStringLiteral("state.json"));
    const QFileInfo info(path);
    if (!info.exists()) {
        if (state_mtime_ != -2) {
            state_summary_.clear();
            state_mtime_ = -2;
            update_header();
        }
        return;
    }
    const qint64 mtime = info.lastModified().toMSecsSinceEpoch();
    if (mtime == state_mtime_) return;
    QFile f(path);
    if (!f.open(QIODevice::ReadOnly)) return;
    QJsonParseError err{};
    const QJsonDocument doc = QJsonDocument::fromJson(f.readAll(), &err);
    if (err.error != QJsonParseError::NoError || !doc.isObject()) return;  // mid-write: next time
    state_mtime_ = mtime;
    const QJsonObject st = doc.object();

    // Tiles, in the order the listener lists them (newest slot first).
    QStringList order;
    std::vector<QrssSpectrogram::Marker> markers;
    // Each tile's picture (station, picture ID, mode) and how many passes
    // it shows, to hide the ones a later combined tile already includes.
    struct Combined {
        QString id, picture, slot;
        int passes = 1;
    };
    std::vector<Combined> pictures;
    for (const QJsonValue& v : st.value(QStringLiteral("tiles")).toArray()) {
        const QJsonObject t = v.toObject();
        const QString id = t.value(QStringLiteral("id")).toString();
        if (id.isEmpty() || order.contains(id)) continue;
        order << id;
        if (const QString pid = t.value(QStringLiteral("picture_id")).toString(); !pid.isEmpty()) {
            pictures.push_back({id,
                                t.value(QStringLiteral("callsign")).toString() + QLatin1Char('/') +
                                    pid + QLatin1Char('/') + t.value(QStringLiteral("mode")).toString(),
                                t.value(QStringLiteral("slot_utc")).toString(),
                                t.value(QStringLiteral("passes")).toInt(1)});
        }
        QrssTile* tile = tiles_.value(id, nullptr);
        if (!tile) {
            tile = new QrssTile(tiles_host_);
            tiles_.insert(id, tile);
        }
        tile->update_from(t, state_dir_);
        // Its pass, for the spectrogram's red.
        const QDateTime slot = QDateTime::fromString(
            t.value(QStringLiteral("slot_utc")).toString(), Qt::ISODate);
        const double f_hz = t.value(QStringLiteral("f_hz")).toDouble();
        if (slot.isValid() && f_hz > 0) {
            const double t0 = static_cast<double>(slot.toSecsSinceEpoch());
            markers.push_back({f_hz, t0, t0 + PASS_S});
        }
    }
    for (auto it = tiles_.begin(); it != tiles_.end();) {
        if (!order.contains(it.key())) {
            it.value()->deleteLater();
            it = tiles_.erase(it);
        } else {
            ++it;
        }
    }
    // Re-lay out in order: take everything out, put it back, each tile
    // in its column.
    while (QLayoutItem* item = flow_->takeAt(0)) delete item;
    while (QLayoutItem* item = prov_list_->takeAt(0)) delete item;
    QStringList received;
    QStringList provisional;
    // **One tile per picture** (Jeff, 2026-10-10): a later pass of the
    // same picture shows every earlier pass combined into it ("2 passes
    // combined"), so the earlier tiles it includes are hidden, not shown
    // beside it. The listener combines passes across dials too, so the
    // dial is not part of the picture. An earlier tile is included when a
    // later one shows more passes than it does; two later tiles of one
    // pass each (not combined) both stay.
    QSet<QString> superseded;
    for (const Combined& older : pictures) {
        for (const Combined& newer : pictures) {
            if (newer.picture == older.picture && newer.slot > older.slot &&
                newer.passes > older.passes) {
                superseded.insert(older.id);
                break;
            }
        }
    }
    for (const QString& id : order) {
        QrssTile* tile = tiles_.value(id);
        if (!tile->provisional() && superseded.contains(id)) {
            tile->hide();
            continue;
        }
        (tile->provisional() ? provisional : received) << id;
    }
    empty_label_->setVisible(received.isEmpty());
    if (received.isEmpty()) flow_->addWidget(empty_label_);
    for (const QString& id : received) {
        QrssTile* tile = tiles_.value(id);
        tile->setParent(tiles_host_);
        flow_->addWidget(tile);
        tile->show();
    }
    prov_empty_->setVisible(provisional.isEmpty());
    if (provisional.isEmpty()) prov_list_->addWidget(prov_empty_);
    for (const QString& id : provisional) {
        QrssTile* tile = tiles_.value(id);
        tile->setParent(prov_host_);
        prov_list_->addWidget(tile);
        tile->show();
    }
    prov_list_->addStretch(1);
    update_markers(markers);

    QStringList parts;
    const bool listening = st.value(QStringLiteral("listening")).toBool();
    parts << (listening ? tr("Listener: listening (%1)")
                              .arg(st.value(QStringLiteral("source")).toString())
                        : tr("Listener: not listening"));
    const QJsonArray slot_list = st.value(QStringLiteral("slots")).toArray();
    QStringList on_air;
    for (const QJsonValue& v : slot_list) {
        const QJsonObject s = v.toObject();
        on_air << tr("%1 at %2%")
                      .arg(hhmm(s.value(QStringLiteral("slot_utc")).toString()))
                      .arg(static_cast<int>(
                          std::lround(100.0 * s.value(QStringLiteral("progress")).toDouble())));
    }
    if (!on_air.isEmpty()) parts << tr("Slots on the air: %1").arg(on_air.join(QStringLiteral(", ")));
    const QString busy = st.value(QStringLiteral("busy")).toString();
    if (!busy.isEmpty()) parts << tr("Working on: %1").arg(busy);
    const QString codec_error = st.value(QStringLiteral("codec_error")).toString();
    if (!codec_error.isEmpty()) parts << tr("No pictures: %1").arg(codec_error);
    const QJsonArray log = st.value(QStringLiteral("log")).toArray();
    if (!log.isEmpty()) parts << log.last().toString();
    state_summary_ = parts.join(QLatin1Char('\n'));
    update_header();
}

void QrssWindow::update_header() {
    const bool running = listener_running();
    start_button_->setEnabled(!running);
    stop_button_->setEnabled(running);
    command_edit_->setEnabled(!running);
    QStringList lines;
    if (running) {
        lines << (ring_ ? tr("Feeding the receive pane's audio (%1 s sent%2).")
                              .arg(samples_sent_ / 8000)
                              .arg(samples_dropped_
                                       ? tr(", %1 s dropped: listener not reading")
                                             .arg(samples_dropped_ / 8000)
                                       : QString())
                        : tr("No audio: press Listen in the receive pane so the "
                             "listener hears the radio."));
    }
    lines << tr("Tiles: %1").arg(QDir::toNativeSeparators(state_dir_));
    if (!state_summary_.isEmpty()) lines << state_summary_;
    status_label_->setText(lines.join(QLatin1Char('\n')));
}

void QrssWindow::closeEvent(QCloseEvent* event) {
    // Closing the window only hides it; the listener keeps running and
    // View > QRSSTVAE brings it back.
    hide();
    event->ignore();
}

}  // namespace sstvae::gui

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
#include <QPixmap>
#include <QPlainTextEdit>
#include <QProcessEnvironment>
#include <QProgressBar>
#include <QPushButton>
#include <QScrollArea>
#include <QStandardPaths>
#include <QTimer>
#include <QVBoxLayout>

#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

#include "flow_layout.hpp"
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
    title_ = new QLabel(this);
    QFont bold = title_->font();
    bold.setBold(true);
    title_->setFont(bold);
    details_ = new QLabel(this);
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
    box->addWidget(picture_);
    box->addWidget(title_);
    box->addWidget(progress_text_);
    box->addWidget(progress_);
    box->addWidget(details_);
    box->addStretch(1);
    setFixedWidth(PICTURE_W + 12);
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

    QStringList lines;
    lines << tr("Slot %1 · %2 Hz · SNR %3")
                 .arg(hhmm(t.value(QStringLiteral("slot_utc")).toString()))
                 .arg(f_hz, 0, 'f', 1)
                 .arg(number_or(t.value(QStringLiteral("snr_db")), 1, tr(" dB")));
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
    const QString note = t.value(QStringLiteral("note")).toString();
    if (!note.isEmpty()) lines << note;
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
int QrssTile::progress_percent() const { return progress_->value(); }
QString QrssTile::progress_text() const { return progress_text_->text(); }

// --- window ----------------------------------------------------------------------------

QrssWindow::QrssWindow(QWidget* parent) : QWidget(parent, Qt::Window) {
    setWindowTitle(tr("QRSS signals"));
    state_dir_ = default_state_dir();

    auto* box = new QVBoxLayout(this);

    auto* row = new QHBoxLayout;
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
    box->addLayout(row);

    status_label_ = new QLabel(this);
    status_label_->setWordWrap(true);
    status_label_->setTextInteractionFlags(Qt::TextSelectableByMouse);
    box->addWidget(status_label_);

    scroll_ = new QScrollArea(this);
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
    box->addWidget(scroll_, 1);

    log_ = new QPlainTextEdit(this);
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

void QrssWindow::set_ring(std::shared_ptr<rx::RingBuffer> ring) {
    ring_ = std::move(ring);
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
    for (const QJsonValue& v : st.value(QStringLiteral("tiles")).toArray()) {
        const QJsonObject t = v.toObject();
        const QString id = t.value(QStringLiteral("id")).toString();
        if (id.isEmpty() || order.contains(id)) continue;
        order << id;
        QrssTile* tile = tiles_.value(id, nullptr);
        if (!tile) {
            tile = new QrssTile(tiles_host_);
            tiles_.insert(id, tile);
        }
        tile->update_from(t, state_dir_);
    }
    for (auto it = tiles_.begin(); it != tiles_.end();) {
        if (!order.contains(it.key())) {
            it.value()->deleteLater();
            it = tiles_.erase(it);
        } else {
            ++it;
        }
    }
    // Re-lay out in order: take everything out, put it back.
    while (QLayoutItem* item = flow_->takeAt(0)) delete item;
    empty_label_->setVisible(order.isEmpty());
    if (order.isEmpty()) flow_->addWidget(empty_label_);
    for (const QString& id : order) {
        QrssTile* tile = tiles_.value(id);
        flow_->addWidget(tile);
        tile->show();
    }

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
    // View > QRSS signals brings it back.
    hide();
    event->ignore();
}

}  // namespace sstvae::gui

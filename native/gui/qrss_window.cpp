#include "qrss_window.hpp"

#include <QCloseEvent>
#include <QCoreApplication>
#include <QDateTime>
#include <QDir>
#include <QEvent>
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
#include "qrss_em.hpp"
#include "qrss_spectrogram.hpp"
#include "qrss_tx.hpp"
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
constexpr double PASS_S = qrss_tx::PASS_S;
// What the listener appends to a headerless tile's note; the Provisional
// column says it instead.
const QString GUESS_NOTE =
    QStringLiteral("no header yet: picture assumes the first pass of a mode A send");

QString number_or(const QJsonValue& v, int decimals, const QString& unit,
                  const QString& none = QStringLiteral("?")) {
    if (!v.isDouble()) return none;
    return QString::number(v.toDouble(), 'f', decimals) + unit;
}

// The listener's note in plain words (Jeff, 2026-10-10): "stored (rule)"
// says how the pass was filed with others of its picture
// (sstvae/qrss/associate.py); anything not recognised is shown as is.
QString plain_note(const QString& note) {
    static const std::pair<const char*, const char*> STORED[] = {
        {"stored (header)", QT_TR_NOOP("Saved: matched to its picture by its header")},
        {"stored (soft-header)",
         QT_TR_NOOP("Saved: header read by adding up several passes")},
        {"stored (corr)", QT_TR_NOOP("Saved: matched to a stored picture by its contents")},
        {"stored (already a member)",
         QT_TR_NOOP("Somehow I already received all of this image!")},
        {"stored (provisional)", QT_TR_NOOP("Kept; waiting to find out which picture this is")},
    };
    QString out = note;
    for (const auto& [from, to] : STORED) {
        out.replace(QString::fromLatin1(from), QCoreApplication::translate("QrssTile", to));
    }
    if (out.startsWith(QStringLiteral("store failed: "))) {
        out = QCoreApplication::translate("QrssTile", "Could not save: %1")
                  .arg(out.mid(QStringLiteral("store failed: ").size()));
    } else if (out.startsWith(QStringLiteral("render failed: "))) {
        out = QCoreApplication::translate("QrssTile", "Could not draw the picture: %1")
                  .arg(out.mid(QStringLiteral("render failed: ").size()));
    }
    return out;
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

// --- pass bar ---------------------------------------------------------------------------

namespace {
constexpr int PASS_BAR_H = 8;
// The preamble at the start of a pass carries no latents: not a hole.
constexpr double PREAMBLE_S = 20.0;
// receiver.CONF_FLOOR_DB: a bin whose latents were all erased.
constexpr double CONF_FLOOR_DB = -30.0;
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

QColor PassBar::color(double weight_db) {
    const auto& lo = BAND_STOPS[0];
    const auto& hi = BAND_STOPS[std::size(BAND_STOPS) - 1];
    if (!(weight_db > lo.db)) return QColor(lo.r, lo.g, lo.b);
    if (weight_db >= hi.db) return QColor(hi.r, hi.g, hi.b);
    for (std::size_t i = 1; i < std::size(BAND_STOPS); ++i) {
        const Stop& a = BAND_STOPS[i - 1];
        const Stop& b = BAND_STOPS[i];
        if (weight_db <= b.db) {
            const double f = (weight_db - a.db) / (b.db - a.db);
            auto mix = [f](int x, int y) {
                return static_cast<int>(std::lround(x + (y - x) * f));
            };
            return QColor(mix(a.r, b.r), mix(a.g, b.g), mix(a.b, b.b));
        }
    }
    return QColor(hi.r, hi.g, hi.b);
}

QColor PassBar::unmeasured() { return QColor(128, 128, 128); }


PassBar::PassBar(QWidget* parent) : QWidget(parent) {
    setObjectName(QStringLiteral("pass_bar"));
    setFixedSize(PICTURE_W, PASS_BAR_H);
    setToolTip(tr("The pass so far, left to right, coloured by how well each stretch came "
                  "through: black is no better than noise (or nothing there: not heard, or "
                  "lost), then blue, red, yellow, and white at +10 dB or more per latent. "
                  "Grey: the preamble."));
}

void PassBar::set(const std::vector<std::optional<double>>& bins, double progress) {
    progress_ = std::clamp(progress, 0.0, 1.0);
    bins_ = bins;
    holes_.assign(bins.size(), false);
    const double n = static_cast<double>(bins.size());
    for (std::size_t i = 0; i < bins.size(); ++i) {
        const double start = static_cast<double>(i) / n;
        const double end = static_cast<double>(i + 1) / n;
        if (start >= progress_) break;
        // A bin wholly inside the preamble has nothing to miss.
        if (end * PASS_S <= PREAMBLE_S) continue;
        holes_[i] = !bins[i] || *bins[i] <= CONF_FLOOR_DB + 0.01;
    }
    update();
}

void PassBar::paintEvent(QPaintEvent*) {
    QPainter painter(this);
    const QPalette& pal = palette();
    const int w = width();
    const int h = height();
    // The track is the window's colour, not Base: Base is white on most
    // themes, which is also the strongest confidence.
    painter.fillRect(rect(), pal.color(QPalette::Window));
    painter.setPen(pal.color(QPalette::Mid));
    painter.drawRect(0, 0, w - 1, h - 1);
    const int fill = static_cast<int>(std::lround(progress_ * (w - 2)));
    const int n = static_cast<int>(bins_.size());
    if (n == 0) {
        // No confidence yet: the length alone.
        painter.fillRect(1, 1, fill, h - 2, unmeasured());
        return;
    }
    for (int i = 0; i < n; ++i) {
        const int x0 = 1 + static_cast<int>(std::lround(static_cast<double>(i) * (w - 2) / n));
        const int x1 = std::min(
            1 + fill, 1 + static_cast<int>(std::lround(static_cast<double>(i + 1) * (w - 2) / n)));
        if (x1 <= x0) continue;
        const std::optional<double>& v = bins_[static_cast<std::size_t>(i)];
        const QColor c = v ? color(*v)
                         : holes_[static_cast<std::size_t>(i)] ? color(CONF_FLOOR_DB)
                                                               : unmeasured();
        painter.fillRect(x0, 1, x1 - x0, h - 2, c);
    }
}

// --- tile -------------------------------------------------------------------------------

QrssTile::QrssTile(QWidget* parent, qrss_em::Controller* em) : QWidget(parent), em_(em) {
    auto* box = new QVBoxLayout(this);
    box->setContentsMargins(6, 6, 6, 6);
    box->setSpacing(3);
    picture_ = new QLabel(this);
    picture_->setFixedSize(PICTURE_W, PICTURE_H);
    picture_->setAlignment(Qt::AlignCenter);
    picture_->setFrameShape(QFrame::Box);
    picture_->setText(tr("Waiting for the header"));
    picture_->setWordWrap(true);
    // Small print under the picture, as Glissando has it: the picture
    // is what the tile is for.
    QFont small = font();
    small.setPointSizeF(std::max(6.5, small.pointSizeF() * 0.78));
    title_ = new QLabel(this);
    QFont bold = font();
    bold.setPointSizeF(std::max(7.0, bold.pointSizeF() * 0.88));
    bold.setBold(true);
    title_->setFont(bold);
    // The picture's mean weight (per-latent SNR, receiver.mean_w_db), the
    // number most worth a glance: on the title line, at the right, in the
    // same bold. Not the carrier's SNR, which the spectrogram shows
    // better (Jeff, 2026-10-10).
    weight_ = new QLabel(this);
    weight_->setObjectName(QStringLiteral("tile_weight"));
    weight_->setFont(bold);
    weight_->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
    details_ = new QLabel(this);
    details_->setFont(small);
    details_->setWordWrap(true);
    details_->setFixedWidth(PICTURE_W);
    details_->setTextInteractionFlags(Qt::TextSelectableByMouse);
    // The listener's longer notes ("Saved: ...", errors) in smaller print
    // still, under the rest (Jeff, 2026-10-11).
    note_ = new QLabel(this);
    note_->setObjectName(QStringLiteral("tile_note"));
    QFont smaller = font();
    smaller.setPointSizeF(std::max(6.0, smaller.pointSizeF() * 0.68));
    note_->setFont(smaller);
    note_->setWordWrap(true);
    note_->setFixedWidth(PICTURE_W);
    note_->setTextInteractionFlags(Qt::TextSelectableByMouse);
    // **The bar says it all** (Jeff, 2026-10-10): its length is how far
    // through the pass, its colours how well each stretch came through
    // (missing latents are black); no words, and no second strip.
    progress_ = new PassBar(this);
    box->addWidget(picture_);
    auto* title_row = new QHBoxLayout;
    title_row->setContentsMargins(0, 0, 0, 0);
    title_row->addWidget(title_, 1);
    title_row->addWidget(weight_);
    box->addLayout(title_row);
    box->addWidget(progress_);
    box->addWidget(details_);
    box->addWidget(note_);
    if (em_ != nullptr) {
        em_bar_ = new qrss_em::EmBar(em_, this);
        em_bar_->setFixedWidth(PICTURE_W);
        box->addWidget(em_bar_);
        connect(em_bar_, &qrss_em::EmBar::pictureChanged, this,
                [this](const QString& path, std::optional<double> mean_w_db) {
                    QPixmap px(path);
                    if (px.isNull()) return;
                    picture_->setPixmap(
                        px.scaled(PICTURE_W, PICTURE_H, Qt::KeepAspectRatio, Qt::SmoothTransformation));
                    image_path_ = path;
                    em_rev_ = std::max(0, image_rev_);
                    if (mean_w_db) weight_->setText(QString::number(*mean_w_db, 'f', 1) + tr(" dB"));
                });
    }
    box->addStretch(1);
    setFixedWidth(WIDTH);
}

void QrssTile::update_from(const QJsonObject& t, const QString& dir) {
    const QString call = t.value(QStringLiteral("callsign")).toString();
    const QString grid = t.value(QStringLiteral("grid")).toString();
    // Any number in a tile can be null (the listener writes a NaN or an
    // infinity so): unknown, never 0.
    const QJsonValue f_hz = t.value(QStringLiteral("f_hz"));
    const QString status = t.value(QStringLiteral("status")).toString();
    if (!call.isEmpty()) {
        title_->setText(grid.isEmpty() ? call : call + QStringLiteral("  ") + grid);
    } else {
        title_->setText(tr("Signal at %1").arg(number_or(f_hz, 1, tr(" Hz"), tr("? Hz"))));
    }

    const double progress = t.value(QStringLiteral("progress")).toDouble();
    std::vector<std::optional<double>> bins;
    for (const QJsonValue& v : t.value(QStringLiteral("confidence")).toArray()) {
        bins.push_back(v.isDouble() ? std::optional<double>(v.toDouble()) : std::nullopt);
    }
    progress_->set(bins, progress);

    const int rev = t.value(QStringLiteral("image_rev")).toInt();
    // The app drew the picture again: its picture and weight stay until
    // the listener draws one of its own.
    if (em_rev_ >= 0 && rev != em_rev_) em_rev_ = -1;
    const bool em_shown = em_rev_ >= 0;
    if (!em_shown) {
        weight_->setText(number_or(t.value(QStringLiteral("mean_w_db")), 1, tr(" dB"), QString()));
    }
    QStringList lines;
    // Where it is: the dial plus the audio offset (USB), as an operator
    // would say it ("10.141 MHz + 1500 Hz", Jeff, 2026-10-10); the offset
    // alone when the dial is unknown (no rig control: null).
    QString where = number_or(f_hz, 0, tr(" Hz"), tr("? Hz"));
    if (const QJsonValue dial = t.value(QStringLiteral("dial_hz")); dial.isDouble()) {
        QString mhz = QString::number(dial.toDouble() / 1e6, 'f', 6);
        while (mhz.endsWith(QLatin1Char('0'))) mhz.chop(1);
        if (mhz.endsWith(QLatin1Char('.'))) mhz.chop(1);
        where = tr("%1 MHz + %2").arg(mhz, where);
    }
    // CE is the default and goes unsaid, as on the schedule's timeline.
    if (t.value(QStringLiteral("waveform")).toString() == QLatin1String("L")) {
        where += tr(" · waveform L");
    }
    lines << tr("Slot %1 · %2").arg(hhmm(t.value(QStringLiteral("slot_utc")).toString()), where);
    const QString pid = t.value(QStringLiteral("picture_id")).toString();
    if (!pid.isEmpty()) {
        const int passes = t.value(QStringLiteral("passes")).toInt(1);
        lines << tr("Picture %1, mode %2 · %3")
                     .arg(pid, t.value(QStringLiteral("mode")).toString(),
                          passes == 1 ? tr("1 pass") : tr("%1 passes combined").arg(passes));
    }
    const QString cw = t.value(QStringLiteral("cw_text")).toString();
    // Only while the header's callsign is not decoded: once it is, the
    // Morse ID adds nothing (Jeff, 2026-10-10).
    if (!cw.isEmpty() && call.isEmpty()) lines << tr("Morse ID read: %1").arg(cw);
    QString state;
    if (status == QStringLiteral("complete")) {
        state = tr("Complete");
    } else if (status == QStringLiteral("finishing")) {
        // The frame has ended; the end-of-slot receive is running.
        state = tr("Finishing");
    } else if (status == QStringLiteral("lost")) {
        state = tr("Lost (not found at the end of the pass)");
    } else {
        state = tr("Receiving");
    }
    lines << state;
    QString note = t.value(QStringLiteral("note")).toString();
    note.remove(QStringLiteral("; ") + GUESS_NOTE).remove(GUESS_NOTE);
    note = plain_note(note);
    note_->setText(note);
    note_->setVisible(!note.isEmpty());
    provisional_ = pid.isEmpty();
    if (em_bar_ != nullptr) {
        qrss_em::Target target;
        target.callsign = call;
        target.picture_id = pid;
        target.mode = t.value(QStringLiteral("mode")).toString();
        target.frame = t.value(QStringLiteral("frame")).toString();
        target.waveform = t.value(QStringLiteral("waveform")).toString();
        target.status = status;
        target.passes = t.value(QStringLiteral("passes")).toInt(1);
        em_bar_->set_target(target);
    }
    details_->setText(lines.join(QLatin1Char('\n')));

    const QString image = t.value(QStringLiteral("image")).toString();
    if (em_shown) {
        // The app's picture, above.
    } else if (!image.isEmpty()) {
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
                          : status == QStringLiteral("receiving") ||
                                    status == QStringLiteral("finishing")
                              ? tr("Waiting for the header")
                              : tr("No header decoded: nowhere to place the picture"));
    }
}

QString QrssTile::title() const { return title_->text(); }
QString QrssTile::weight_text() const { return weight_->text(); }
QString QrssTile::details() const {
    return note_->text().isEmpty() ? details_->text()
                                   : details_->text() + QLatin1Char('\n') + note_->text();
}
QString QrssTile::note() const { return note_->text(); }
bool QrssTile::has_picture() const {
    return !picture_->pixmap().isNull();
}
void QrssTile::mouseDoubleClickEvent(QMouseEvent* event) {
    if (event->button() == Qt::LeftButton && !image_path_.isEmpty()) {
        // From the file, at full size: the tile shows a reduction.
        ImageViewer* viewer = open_image_viewer(QPixmap(image_path_), title_->text(), this);
        if (viewer != nullptr && em_bar_ != nullptr && em_bar_->target().valid()) {
            auto* bar = new qrss_em::EmBar(em_, viewer);
            bar->set_target(em_bar_->target());
            viewer->add_footer(bar);
            connect(bar, &qrss_em::EmBar::pictureChanged, viewer,
                    [viewer](const QString& path, std::optional<double>) {
                        const QPixmap px(path);
                        if (!px.isNull()) viewer->view()->set_pixmap(px);
                    });
            // The tile's bar follows the tile; this one keeps the picture
            // the viewer was opened on.
        }
        return;
    }
    QWidget::mouseDoubleClickEvent(event);
}

int QrssTile::progress_percent() const {
    return static_cast<int>(std::lround(100.0 * progress_->progress()));
}

// --- window ----------------------------------------------------------------------------

QrssWindow::QrssWindow(QWidget* parent, Qt::WindowFlags flags) : QWidget(parent, flags) {
    setWindowTitle(tr("QRSS signals"));
    state_dir_ = default_state_dir();
    em_ = new qrss_em::Controller(state_dir_, this);

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
        tr("The command that runs the QRSSTVAE listener (sstvae-qrss listen, or "
           "qrss_listen.py); \"--state DIR\" is added. Set SSTVAE_QRSS_LISTEN to "
           "change the default."));
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
    // Cards line up by their tops: a short "receiving" card no longer
    // sits lower than a finished one beside it.
    flow_->set_align_top(true);
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
    // The C++ listener (`sstvae-qrss listen`, the same arguments and
    // files as qrss_listen.py) unless SSTVAE_QRSS_BACKEND=python, as for
    // sending. Looked for beside the executable and up to four
    // directories above it: a macOS bundle's executable sits three
    // below the build directory that holds sstvae-qrss.
    const bool python_backend = qgetenv("SSTVAE_QRSS_BACKEND") == "python";
#ifdef Q_OS_WIN
    const QString native = QStringLiteral("sstvae-qrss.exe");
#else
    const QString native = QStringLiteral("sstvae-qrss");
#endif
    QDir dir(QCoreApplication::applicationDirPath());
    QString script;
    for (int up = 0; up <= 4; ++up) {
        const QString exe = dir.filePath(native);
        if (!python_backend && QFileInfo(exe).isExecutable() && QFileInfo(exe).isFile()) {
            return join_command({exe, QStringLiteral("listen")});
        }
        const QString py = dir.filePath(QStringLiteral("qrss_listen.py"));
        if (script.isEmpty() && QFileInfo::exists(py)) {
            QString python = QStringLiteral("python3");
            for (const char* venv : {".venv/bin/python", ".venv/Scripts/python.exe"}) {
                const QString p = dir.filePath(QString::fromLatin1(venv));
                if (QFileInfo::exists(p)) {
                    python = p;
                    break;
                }
            }
            script = join_command({python, py});
            if (python_backend) return script;
        }
        if (!dir.cdUp()) break;
    }
    if (!script.isEmpty()) return script;
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
    em_->set_state_dir(dir);
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
            tile = new QrssTile(tiles_host_, em_);
            tiles_.insert(id, tile);
        }
        tile->update_from(t, state_dir_);
        // Its pass, for the spectrogram's red.
        const QDateTime slot = QDateTime::fromString(
            t.value(QStringLiteral("slot_utc")).toString(), Qt::ISODate);
        const double f_hz = t.value(QStringLiteral("f_hz")).toDouble();
        if (slot.isValid() && f_hz > 0) {
            const double t0 = static_cast<double>(slot.toSecsSinceEpoch());
            QrssSpectrogram::Marker m{f_hz, t0, t0 + PASS_S};
            if (t.value(QStringLiteral("waveform")).toString() == QLatin1String("L")) {
                m.below_hz = QrssSpectrogram::MARK_L_BELOW_HZ;
                m.above_hz = QrssSpectrogram::MARK_L_ABOVE_HZ;
            }
            markers.push_back(m);
        }
    }
    for (auto it = tiles_.begin(); it != tiles_.end();) {
        if (!order.contains(it.key())) {
            // Hidden now, not only when the deferred delete comes round:
            // until then it would still be drawn where it was, under the
            // tile laid out into its place.
            it.value()->hide();
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
    // Lay the tiles out now and repaint both columns whole. Left to the
    // posted layout request, tiles that moved could stay drawn where
    // they were, over the ones that moved into their place, until a
    // scroll repainted the pane (Jeff, 2026-10-11).
    flow_->activate();
    prov_list_->activate();
    tiles_host_->update();
    prov_host_->update();
    scroll_->viewport()->update();
    prov_scroll_->viewport()->update();
    // And tell both scroll areas the content's height changed. A flow
    // layout's minimum is one card whatever the count, so Qt sees no
    // size change to pass on by itself, and new cards could arrive with
    // no scroll bar to reach them (Jeff, 2026-10-11).
    tiles_host_->updateGeometry();
    prov_host_->updateGeometry();
    QCoreApplication::postEvent(scroll_, new QEvent(QEvent::LayoutRequest));
    QCoreApplication::postEvent(prov_scroll_, new QEvent(QEvent::LayoutRequest));
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
        const QJsonValue progress = s.value(QStringLiteral("progress"));
        on_air << tr("%1 at %2%")
                      .arg(hhmm(s.value(QStringLiteral("slot_utc")).toString()),
                           progress.isDouble()
                               ? QString::number(std::lround(100.0 * progress.toDouble()))
                               : QStringLiteral("?"));
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
    // Never shorter than it has been: a line coming and going moved both
    // tile columns up and down with it, and a column moved that way could
    // stay drawn over itself until scrolled (Jeff, 2026-10-11).
    const int need = status_label_->width() > 0
                         ? status_label_->heightForWidth(status_label_->width())
                         : status_label_->sizeHint().height();
    if (need > status_label_->minimumHeight()) status_label_->setMinimumHeight(need);
}

void QrssWindow::closeEvent(QCloseEvent* event) {
    // Closing the window only hides it; the listener keeps running and
    // View > QRSSTVAE brings it back.
    hide();
    event->ignore();
}

}  // namespace sstvae::gui

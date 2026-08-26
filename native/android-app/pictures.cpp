#include "pictures.hpp"

#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QJsonDocument>
#include <QJsonObject>
#include <QStandardPaths>

#include <cstring>

#include "composition.hpp"
#include "rx/engine.hpp"
#include "session.hpp"

namespace {

using sstvae::androidapp::Composition;
using sstvae::androidapp::Session;

QString pictures_dir() {
    return QStandardPaths::writableLocation(QStandardPaths::AppDataLocation) +
           QStringLiteral("/pictures");
}

// A reception's picture, converted for display. `Picture` is tightly
// packed RGB888 and `QImage` wants a stride, so this copies rather than
// wrapping -- correct here, because the engine may replace the picture
// under us the moment the lock is released.
QImage to_qimage(const sstvae::images::Picture& p) {
    if (p.width <= 0 || p.height <= 0) return {};
    QImage img(p.width, p.height, QImage::Format_RGB888);
    const int row = p.width * 3;
    for (int y = 0; y < p.height; ++y) {
        std::memcpy(img.scanLine(y), p.rgb.data() + static_cast<std::size_t>(y) * row,
                    row);
    }
    return img;
}

}  // namespace

PictureProvider::PictureProvider() : QQuickImageProvider(QQuickImageProvider::Image) {}

QImage PictureProvider::requestImage(const QString& id, QSize* size, const QSize&) {
    QImage out;
    if (id.startsWith(QStringLiteral("live/"))) {
        const sstvae::rx::Progress p = Session::instance().progress();
        if (p.image) out = to_qimage(*p.image);
    } else if (id.startsWith(QStringLiteral("file/"))) {
        out.load(id.mid(5));
    } else if (id.startsWith(QStringLiteral("compose/"))) {
        // **The composition preview is `images::fit`'s own output**, run
        // with the framing the transmitter will use, rather than a
        // scaled-and-clipped QML Image imitating a crop. The desktop's
        // rule, for the desktop's reason: a second representation of the
        // picture is a second thing that can disagree with what goes on
        // the air, and here the whole screen exists to decide exactly
        // that. Cheap enough to re-run per drag -- it is one stb resize
        // to 640x480 -- and the QML side loads it asynchronously.
        out = to_qimage(Composition::instance().preview());
    }
    if (size) *size = out.size();
    return out;
}

PictureList::PictureList(QObject* parent) : QAbstractListModel(parent) { refresh(); }

int PictureList::rowCount(const QModelIndex&) const { return entries_.size(); }

QHash<int, QByteArray> PictureList::roleNames() const {
    return {{PathRole, "path"},         {ReceivedRole, "received"},
            {CallsignRole, "callsign"}, {ModeRole, "mode"},
            {SnrRole, "snr"},           {FramesRole, "frames"},
            {SummaryRole, "summary"}};
}

QVariant PictureList::data(const QModelIndex& index, int role) const {
    if (index.row() < 0 || index.row() >= entries_.size()) return {};
    const PictureEntry& e = entries_[index.row()];
    switch (role) {
        case PathRole:
            return e.path;
        case ReceivedRole:
            return e.received;
        case CallsignRole:
            return e.callsign.isEmpty() ? QStringLiteral("no callsign") : e.callsign;
        case ModeRole:
            return e.mode;
        case SnrRole:
            return e.snr_db;
        case FramesRole: {
            QString s = QStringLiteral("%1/%2").arg(e.frames_received)
                            .arg(e.frames_expected);
            if (e.frames_decoded > 0 && e.frames_expected > 0) {
                s += QStringLiteral("  (%1% decoded)")
                         .arg(100.0 * e.frames_decoded / e.frames_expected, 0, 'f', 0);
            }
            return s;
        }
        case SummaryRole: {
            // One line, because a gallery row has room for one. The
            // callsign leads: on a phone the first question about a
            // received picture is who sent it.
            QString s = e.callsign.isEmpty() ? QStringLiteral("unknown") : e.callsign;
            if (!e.mode.isEmpty()) s += QStringLiteral("  mode %1").arg(e.mode);
            s += QStringLiteral("  %1 dB").arg(e.snr_db, 0, 'f', 1);
            if (e.frames_expected > 0) {
                s += QStringLiteral("  %1/%2")
                         .arg(e.frames_received)
                         .arg(e.frames_expected);
            }
            return s;
        }
        default:
            return {};
    }
}

void PictureList::refresh() {
    beginResetModel();
    entries_.clear();

    QDir dir(pictures_dir());
    // Newest first: the reception someone wants is almost always the
    // last one.
    const QStringList names =
        dir.entryList({QStringLiteral("*.png")}, QDir::Files, QDir::Name | QDir::Reversed);
    for (const QString& name : names) {
        PictureEntry e;
        e.path = dir.filePath(name);
        e.received = QFileInfo(name).completeBaseName();

        QFile meta(dir.filePath(QFileInfo(name).completeBaseName() + ".json"));
        if (meta.open(QIODevice::ReadOnly)) {
            const QJsonObject o = QJsonDocument::fromJson(meta.readAll()).object();
            e.callsign = o.value("callsign").toString();
            e.mode = o.value("mode").toString();
            e.snr_db = o.value("snr_db").toDouble();
            e.frames_received = o.value("frames_received").toInt();
            e.frames_decoded = o.value("frames_decoded").toInt();
            e.frames_expected = o.value("frames_expected").toInt();
        }
        entries_.push_back(e);
    }
    endResetModel();
    emit changed();
}

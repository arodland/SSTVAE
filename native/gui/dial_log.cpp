#include "dial_log.hpp"

#include <QByteArray>
#include <QDateTime>
#include <QDir>
#include <QFile>
#include <QJsonDocument>
#include <QJsonObject>
#include <QList>
#include <QSaveFile>

#include <cmath>
#include <utility>

namespace sstvae::gui {

namespace {

struct Line {
    double t = 0.0;
    long long hz = 0;
    bool ok = false;
};

Line parse(const QByteArray& text) {
    const QJsonObject o = QJsonDocument::fromJson(text).object();
    Line l;
    if (!o.contains(QStringLiteral("t")) || !o.contains(QStringLiteral("dial_hz"))) return l;
    l.t = o.value(QStringLiteral("t")).toDouble();
    l.hz = static_cast<long long>(std::llround(o.value(QStringLiteral("dial_hz")).toDouble()));
    l.ok = l.hz > 0;
    return l;
}

}  // namespace

DialLog::DialLog(QString dir) { set_dir(dir); }

void DialLog::set_dir(const QString& dir) {
    dir_ = dir;
    last_hz_.reset();
    if (dir_.isEmpty()) return;
    read_last();
    trim(static_cast<double>(QDateTime::currentMSecsSinceEpoch()) / 1000.0);
}

QString DialLog::path() const {
    return dir_.isEmpty() ? QString() : QDir(dir_).filePath(QStringLiteral("dial_changes.jsonl"));
}

void DialLog::read_last() {
    QFile f(path());
    if (!f.open(QIODevice::ReadOnly)) return;
    const QList<QByteArray> lines = f.readAll().split('\n');
    for (auto it = lines.crbegin(); it != lines.crend(); ++it) {
        const Line l = parse(*it);
        if (l.ok) {
            last_hz_ = l.hz;
            return;
        }
    }
}

bool DialLog::record(double t, double hz, const char* why) {
    if (dir_.isEmpty() || !(hz > 0.0)) return false;
    const long long rounded = std::llround(hz);
    if (last_hz_ && *last_hz_ == rounded) return false;
    QDir().mkpath(dir_);
    QJsonObject o;
    o[QStringLiteral("t")] = t;
    o[QStringLiteral("dial_hz")] = static_cast<qint64>(rounded);
    o[QStringLiteral("why")] = QString::fromLatin1(why);
    const QByteArray line = QJsonDocument(o).toJson(QJsonDocument::Compact) + '\n';
    QFile f(path());
    if (!f.open(QIODevice::WriteOnly | QIODevice::Append)) return false;
    // One write per line, so a reader never sees half of one.
    if (f.write(line) != line.size()) return false;
    f.close();
    last_hz_ = rounded;
    if (++writes_since_trim_ >= 50) trim(t);
    return true;
}

void DialLog::trim(double now) {
    writes_since_trim_ = 0;
    QFile f(path());
    if (!f.open(QIODevice::ReadOnly)) return;
    const QList<QByteArray> lines = f.readAll().split('\n');
    f.close();
    const double cutoff = now - KEEP_S;
    int newest_old = -1;
    bool any_old = false;
    for (int i = 0; i < lines.size(); ++i) {
        const Line l = parse(lines[i]);
        if (!l.ok) continue;
        if (l.t < cutoff) {
            if (newest_old >= 0) any_old = true;
            newest_old = i;
        }
    }
    if (!any_old) return;   // nothing to drop beyond the line kept
    QByteArray out;
    for (int i = 0; i < lines.size(); ++i) {
        const Line l = parse(lines[i]);
        if (!l.ok) continue;
        if (l.t < cutoff && i != newest_old) continue;
        out += lines[i].trimmed() + '\n';
    }
    QSaveFile save(path());
    if (!save.open(QIODevice::WriteOnly)) return;
    if (save.write(out) != out.size()) {
        save.cancelWriting();
        return;
    }
    save.commit();
}

}  // namespace sstvae::gui

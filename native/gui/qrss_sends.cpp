#include "qrss_sends.hpp"

#include <QDateTime>
#include <QFile>
#include <QJsonDocument>
#include <QJsonObject>
#include <QSaveFile>
#include <QUuid>

#include <algorithm>
#include <exception>
#include <system_error>
#include <utility>

#include "images/images.hpp"

namespace sstvae::gui {

namespace {

double now_s() { return static_cast<double>(QDateTime::currentMSecsSinceEpoch()) / 1000.0; }

QString qpath(const std::filesystem::path& p) { return QString::fromStdString(p.string()); }

}  // namespace

QrssSends::QrssSends(std::filesystem::path dir, QObject* parent)
    : QObject(parent), dir_(std::move(dir)) {
    reload();
}

void QrssSends::reload() {
    sends_.clear();
    std::error_code ec;
    if (!std::filesystem::is_directory(dir_, ec)) {
        emit changed();
        return;
    }
    for (const auto& entry : std::filesystem::directory_iterator(dir_, ec)) {
        const std::filesystem::path p = entry.path();
        if (p.extension() != ".json") continue;
        QFile f(qpath(p));
        if (!f.open(QIODevice::ReadOnly)) continue;
        const QJsonObject o = QJsonDocument::fromJson(f.readAll()).object();
        QrssSend s;
        s.id = p.stem().string();
        s.label = o.value(QStringLiteral("label")).toString().toStdString();
        s.created = o.value(QStringLiteral("created")).toDouble();
        s.updated = o.value(QStringLiteral("updated")).toDouble(s.created);
        s.picture = dir_ / (s.id + ".png");
        s.base = dir_ / (s.id + "-base.png");
        // A Send is its on-air picture; without it there is nothing to
        // schedule.
        if (!std::filesystem::exists(s.picture, ec)) continue;
        if (s.label.empty()) s.label = "picture " + s.id.substr(0, 4);
        sends_.push_back(std::move(s));
    }
    std::sort(sends_.begin(), sends_.end(), [](const QrssSend& a, const QrssSend& b) {
        return a.updated != b.updated ? a.updated > b.updated : a.id < b.id;
    });
    emit changed();
}

const QrssSend* QrssSends::find(const std::string& id) const {
    for (const QrssSend& s : sends_) {
        if (s.id == id) return &s;
    }
    return nullptr;
}

bool QrssSends::write(const QrssSend& send, const images::Picture& base,
                      const overlay::Doc& doc, const images::Picture& picture,
                      std::string* why) {
    std::error_code ec;
    std::filesystem::create_directories(dir_, ec);
    try {
        images::save_png(picture, send.picture.string());
        images::save_png(base, send.base.string());
    } catch (const std::exception& e) {
        if (why) *why = std::string("could not save the picture: ") + e.what();
        return false;
    }
    QJsonObject o;
    o[QStringLiteral("version")] = 1;
    o[QStringLiteral("label")] = QString::fromStdString(send.label);
    o[QStringLiteral("created")] = send.created;
    o[QStringLiteral("updated")] = send.updated;
    o[QStringLiteral("overlay")] =
        QJsonDocument::fromJson(QByteArray::fromStdString(overlay::to_json(doc))).object();
    QSaveFile f(qpath(dir_ / (send.id + ".json")));
    const QByteArray text = QJsonDocument(o).toJson(QJsonDocument::Indented);
    if (!f.open(QIODevice::WriteOnly) || f.write(text) != text.size() || !f.commit()) {
        if (why) *why = "could not save " + (dir_ / (send.id + ".json")).string();
        return false;
    }
    return true;
}

std::string QrssSends::add(const std::string& label, const images::Picture& base,
                           const overlay::Doc& doc, const images::Picture& picture,
                           std::string* why) {
    QrssSend s;
    do {
        s.id = QUuid::createUuid().toString(QUuid::Id128).left(12).toStdString();
    } while (find(s.id) != nullptr);
    s.label = label.empty() ? "picture " + s.id.substr(0, 4) : label;
    s.created = s.updated = now_s();
    s.picture = dir_ / (s.id + ".png");
    s.base = dir_ / (s.id + "-base.png");
    if (!write(s, base, doc, picture, why)) return {};
    reload();
    return s.id;
}

bool QrssSends::update(const std::string& id, const std::string& label,
                       const images::Picture& base, const overlay::Doc& doc,
                       const images::Picture& picture, std::string* why) {
    const QrssSend* old = find(id);
    if (old == nullptr) {
        if (why) *why = "that Send no longer exists";
        return false;
    }
    QrssSend s = *old;
    if (!label.empty()) s.label = label;
    s.updated = now_s();
    if (!write(s, base, doc, picture, why)) return false;
    reload();
    return true;
}

bool QrssSends::rename(const std::string& id, const std::string& label) {
    const QrssSend* s = find(id);
    if (s == nullptr || label.empty()) return false;
    const std::optional<images::Picture> base = load_base(id);
    images::Picture picture;
    try {
        picture = images::load(s->picture.string());
    } catch (const std::exception&) {
        return false;
    }
    return update(id, label, base ? *base : picture, load_doc(id), picture);
}

void QrssSends::remove(const std::string& id) {
    if (find(id) == nullptr) return;
    std::error_code ec;
    for (const char* suffix : {".json", ".png", "-base.png"}) {
        std::filesystem::remove(dir_ / (id + suffix), ec);
    }
    reload();
}

std::optional<images::Picture> QrssSends::load_base(const std::string& id) const {
    const QrssSend* s = find(id);
    if (s == nullptr) return std::nullopt;
    try {
        return images::load(s->base.string());
    } catch (const std::exception&) {
    }
    // An older or hand-made Send with no base: the picture itself, so
    // reopening it still works (its overlay is then baked in).
    try {
        return images::load(s->picture.string());
    } catch (const std::exception&) {
        return std::nullopt;
    }
}

overlay::Doc QrssSends::load_doc(const std::string& id) const {
    QFile f(qpath(dir_ / (id + ".json")));
    if (!f.open(QIODevice::ReadOnly)) return {};
    const QJsonObject o = QJsonDocument::fromJson(f.readAll()).object();
    const QJsonValue v = o.value(QStringLiteral("overlay"));
    if (!v.isObject()) return {};
    try {
        return overlay::from_json(QJsonDocument(v.toObject()).toJson().toStdString());
    } catch (const std::exception&) {
        return {};
    }
}

}  // namespace sstvae::gui

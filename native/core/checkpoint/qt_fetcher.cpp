#include "checkpoint/qt_fetcher.hpp"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <memory>
#include <string>
#include <utility>

#include <QByteArray>
#include <QCoreApplication>
#include <QCryptographicHash>
#include <QEventLoop>
#include <QFile>
#include <QNetworkAccessManager>
#include <QNetworkReply>
#include <QNetworkRequest>
#include <QString>
#include <QUrl>

namespace sstvae::checkpoint {

namespace fs = std::filesystem;

namespace {

// Enough hops for the Hub's 302 to its CDN, with room to spare, and few
// enough that a redirect loop ends as an error rather than a hang.
constexpr int kMaxRedirects = 5;

std::string to_std(const QString& s) { return s.toStdString(); }

// The LFS object hash the Hub states on its 302. Absent for small
// non-LFS files, where the etag is a git blob sha1 and not comparable
// to the content -- so a missing or wrong-shaped value means "no
// checksum available" rather than "checksum failed".
QByteArray linked_sha256(QNetworkReply& reply) {
    QByteArray etag = reply.rawHeader("x-linked-etag");
    etag.replace('"', "");
    // The cast is the point: std::isxdigit takes an int that must be
    // representable as unsigned char, and plain `char` is signed here,
    // so any byte above 0x7F in a malformed header would be UB.
    const bool hex64 =
        etag.size() == 64 && std::all_of(etag.cbegin(), etag.cend(), [](char c) {
            return std::isxdigit(static_cast<unsigned char>(c)) != 0;
        });
    return hex64 ? etag : QByteArray();
}

struct Response {
    std::unique_ptr<QNetworkReply> reply;
    QByteArray expected_sha256;
};

// Follows redirects by hand so the intermediate headers stay visible;
// see the header on why that matters.
Response get_following_redirects(QNetworkAccessManager& nam, const QUrl& start,
                                 const OnProgress& on_progress) {
    QUrl url = start;
    QByteArray sha;
    for (int hop = 0; hop <= kMaxRedirects; ++hop) {
        QNetworkRequest request(url);
        request.setAttribute(QNetworkRequest::RedirectPolicyAttribute,
                             QNetworkRequest::ManualRedirectPolicy);
        request.setHeader(QNetworkRequest::UserAgentHeader, QStringLiteral("sstvae"));

        std::unique_ptr<QNetworkReply> reply(nam.get(request));
        // Live progress while the loop below runs. Redirect hops have no
        // body to speak of, so in practice this reports the artifact
        // itself; before this connection existed the callback fired
        // exactly once, at 100%, which is no progress indication at all.
        if (on_progress) {
            QObject::connect(reply.get(), &QNetworkReply::downloadProgress,
                             [&on_progress](qint64 received, qint64 total) {
                                 on_progress(received, total < 0 ? 0 : total);
                             });
        }
        QEventLoop loop;
        QObject::connect(reply.get(), &QNetworkReply::finished, &loop, &QEventLoop::quit);
        loop.exec();

        if (sha.isEmpty()) sha = linked_sha256(*reply);

        const QVariant redirect =
            reply->attribute(QNetworkRequest::RedirectionTargetAttribute);
        if (!redirect.isValid()) return {std::move(reply), sha};
        url = url.resolved(redirect.toUrl());
    }
    throw CheckpointError("too many redirects fetching " + to_std(start.toString()));
}

}  // namespace

Fetcher qt_fetcher(OnProgress on_progress) {
    return [on_progress = std::move(on_progress)](
               std::string_view filename) -> std::string {
        if (QCoreApplication::instance() == nullptr) {
            throw CheckpointError(
                "downloading needs a QCoreApplication; construct one before "
                "resolving the model, or pass --model with a local artifact");
        }

        const std::string name(filename);
        const fs::path dir(cache_dir());
        std::error_code ec;
        fs::create_directories(dir, ec);
        if (ec) {
            throw CheckpointError("cannot create the model cache at " + dir.string() +
                                  ": " + ec.message());
        }

        QNetworkAccessManager nam;
        const QUrl url(QString::fromStdString(artifact_url(filename)));
        Response response = get_following_redirects(nam, url, on_progress);
        QNetworkReply& reply = *response.reply;

        if (reply.error() != QNetworkReply::NoError) {
            throw CheckpointError("could not fetch " + name + ": " +
                                  to_std(reply.errorString()));
        }
        const int status =
            reply.attribute(QNetworkRequest::HttpStatusCodeAttribute).toInt();
        if (status != 200) {
            throw CheckpointError("could not fetch " + name + ": HTTP " +
                                  std::to_string(status));
        }

        const QByteArray body = reply.readAll();
        if (body.isEmpty()) {
            throw CheckpointError("could not fetch " + name + ": empty response");
        }
        if (on_progress) on_progress(body.size(), body.size());

        if (!response.expected_sha256.isEmpty()) {
            const QByteArray got =
                QCryptographicHash::hash(body, QCryptographicHash::Sha256).toHex();
            if (got.compare(response.expected_sha256.toLower()) != 0) {
                throw CheckpointError(
                    "checksum mismatch on " + name + ": the Hub said " +
                    response.expected_sha256.toStdString() +
                    " but the bytes hash to " + got.toStdString() +
                    ". The download was corrupted; try again.");
            }
        }

        // Atomic: a partial file left in the cache would be found by
        // find_cached on the next run and handed to onnxruntime.
        const fs::path final_path = dir / name;
        const fs::path partial = dir / (name + ".part");
        {
            QFile out(QString::fromStdString(partial.string()));
            if (!out.open(QIODevice::WriteOnly | QIODevice::Truncate)) {
                throw CheckpointError("cannot write " + partial.string() + ": " +
                                      to_std(out.errorString()));
            }
            if (out.write(body) != body.size() || !out.flush()) {
                throw CheckpointError("short write to " + partial.string() + ": " +
                                      to_std(out.errorString()));
            }
        }
        fs::rename(partial, final_path, ec);
        if (ec) {
            fs::remove(partial, ec);
            throw CheckpointError("cannot place " + final_path.string() + ": " +
                                  ec.message());
        }
        return final_path.string();
    };
}

void install_qt_fetcher(OnProgress on_progress) {
    set_default_fetcher(qt_fetcher(std::move(on_progress)));
}

}  // namespace sstvae::checkpoint

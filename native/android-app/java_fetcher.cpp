#include "java_fetcher.hpp"

#include <QCryptographicHash>
#include <QFile>
#include <QJniEnvironment>
#include <QJniObject>

#include <filesystem>
#include <mutex>
#include <string>

namespace sstvae::androidapp {
namespace {

namespace fs = std::filesystem;
using checkpoint::CheckpointError;

constexpr const char* kFetcherClass = "org/cleverdomain/sstvae/ModelFetcher";

// The progress hook, reached from `ModelFetcher.nativeProgress` on
// whichever thread is downloading. A mutex rather than an atomic
// because it guards a `std::function`, and the download is one call
// deep -- there is no contention to speak of.
std::mutex g_progress_mu;
OnProgress g_progress;

void set_progress(OnProgress p) {
    std::lock_guard<std::mutex> lk(g_progress_mu);
    g_progress = std::move(p);
}

std::string hash_file(const fs::path& path) {
    QFile f(QString::fromStdString(path.string()));
    if (!f.open(QIODevice::ReadOnly)) {
        throw CheckpointError("cannot read back " + path.string());
    }
    QCryptographicHash hash(QCryptographicHash::Sha256);
    if (!hash.addData(&f)) {
        throw CheckpointError("cannot hash " + path.string());
    }
    return hash.result().toHex().toStdString();
}

}  // namespace

extern "C" JNIEXPORT void JNICALL
Java_org_cleverdomain_sstvae_ModelFetcher_nativeProgress(JNIEnv*, jclass,
                                                         jlong received, jlong total) {
    OnProgress p;
    {
        std::lock_guard<std::mutex> lk(g_progress_mu);
        p = g_progress;
    }
    if (p) p(received, total);
}

checkpoint::Fetcher java_fetcher(OnProgress on_progress) {
    set_progress(std::move(on_progress));

    return [](std::string_view filename) {
        const std::string name(filename);
        const fs::path dir(checkpoint::cache_dir());
        std::error_code ec;
        fs::create_directories(dir, ec);
        if (ec) {
            throw CheckpointError("cannot create the model cache at " + dir.string() +
                                  ": " + ec.message());
        }
        const fs::path final_path = dir / name;
        const fs::path partial = dir / (name + ".part");

        const std::string url = checkpoint::artifact_url(filename);
        QJniObject jurl = QJniObject::fromString(QString::fromStdString(url));
        QJniObject jpart =
            QJniObject::fromString(QString::fromStdString(partial.string()));

        QJniObject sha = QJniObject::callStaticObjectMethod(
            kFetcherClass, "download",
            "(Ljava/lang/String;Ljava/lang/String;)Ljava/lang/String;", jurl.object(),
            jpart.object());

        // A Java exception is the transport's way of reporting every
        // failure, so it has to become ours before anything else looks
        // at the file. `checkAndClearExceptions` swallows the message,
        // which is exactly what the operator needs, so it is pulled out
        // first.
        QJniEnvironment env;
        if (env->ExceptionCheck()) {
            QJniObject ex = QJniObject::fromLocalRef(env->ExceptionOccurred());
            env->ExceptionClear();
            std::string detail;
            if (ex.isValid()) {
                QJniObject msg =
                    ex.callObjectMethod("getMessage", "()Ljava/lang/String;");
                if (msg.isValid()) detail = msg.toString().toStdString();
            }
            fs::remove(partial, ec);
            throw CheckpointError("could not fetch " + name +
                                  (detail.empty() ? "" : ": " + detail));
        }

        // Verified here, not in Java. A truncated or corrupted download
        // that reached the cache would be found by `find_cached` on the
        // next run and handed to onnxruntime, failing a long way from
        // its cause.
        const std::string expected = sha.isValid() ? sha.toString().toStdString() : "";
        if (!expected.empty()) {
            std::string got = hash_file(partial);
            std::string want = expected;
            for (char& c : want) c = static_cast<char>(std::tolower(c));
            if (got != want) {
                fs::remove(partial, ec);
                throw CheckpointError("checksum mismatch on " + name +
                                      ": the Hub said " + want +
                                      " but the bytes hash to " + got +
                                      ". The download was corrupted; try again.");
            }
        }

        // Atomic, for the same reason.
        fs::rename(partial, final_path, ec);
        if (ec) {
            fs::remove(partial, ec);
            throw CheckpointError("cannot place " + final_path.string() + ": " +
                                  ec.message());
        }
        return final_path.string();
    };
}

void install_java_fetcher(OnProgress on_progress) {
    checkpoint::set_default_fetcher(java_fetcher(std::move(on_progress)));
}

}  // namespace sstvae::androidapp

#include "assets.hpp"

#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>

#include <QJniEnvironment>
#include <QJniObject>
#include <QtCore/qcoreapplication_platform.h>

#include "checkpoint/checkpoint.hpp"

namespace sstvae::androidapp::assets {

namespace {

AAssetManager* g_manager = nullptr;

std::string asset_path(const std::string& part) {
    // The same filename the fetcher would have downloaded, so a bundled
    // artifact and a fetched one are interchangeable and the version
    // pin lives in exactly one place (`checkpoint::DEFAULT_REVISION`).
    return "models/" + checkpoint::onnx_filename(part);
}

}  // namespace

bool init() {
    if (g_manager != nullptr) return true;
    QJniObject ctx = QNativeInterface::QAndroidApplication::context();
    if (!ctx.isValid()) return false;
    QJniObject mgr =
        ctx.callObjectMethod("getAssets", "()Landroid/content/res/AssetManager;");
    if (!mgr.isValid()) return false;
    // A global reference, deliberately never released: `AAssetManager*`
    // is only valid while the Java object behind it is, and this one has
    // to outlive every codec in the process. One reference, for the life
    // of the app.
    QJniEnvironment env;
    jobject global = env->NewGlobalRef(mgr.object());
    g_manager = AAssetManager_fromJava(env.jniEnv(), global);
    if (env.checkAndClearExceptions()) g_manager = nullptr;
    return g_manager != nullptr;
}

std::optional<codec::ModelBlob> model_blob(const std::string& part) {
    if (g_manager == nullptr) return std::nullopt;
    const std::string name = asset_path(part);

    AAsset* a = AAssetManager_open(g_manager, name.c_str(), AASSET_MODE_STREAMING);
    // Missing is a normal answer -- a build with
    // `SSTVAE_ANDROID_BUNDLE_MODELS=OFF` has no such asset, and the
    // codec falls through to the fetcher. Not logged as an error for
    // that reason.
    if (a == nullptr) return std::nullopt;

    const off64_t len = AAsset_getLength64(a);
    codec::ModelBlob blob;
    blob.name = "bundled " + name;
    if (len > 0) {
        blob.data.resize(static_cast<std::size_t>(len));
        // Read rather than `AAsset_getBuffer`, and the difference is
        // only visible in the build: AAPT deflates these to 92% of
        // their size (fp16 weights barely compress), so `getBuffer`
        // would inflate into a heap block owned by the open asset and
        // keep it for as long as the asset stayed open. Reading into
        // the blob puts that same memory somewhere with a lifetime --
        // the codec drops it the moment the session is built.
        const int got = AAsset_read(a, blob.data.data(), blob.data.size());
        if (got != static_cast<int>(blob.data.size())) {
            AAsset_close(a);
            return std::nullopt;
        }
    }
    AAsset_close(a);
    if (blob.data.empty()) return std::nullopt;
    return blob;
}

}  // namespace sstvae::androidapp::assets

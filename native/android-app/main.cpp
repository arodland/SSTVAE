// Tier 0's entry point.
//
// Proving ground at this stage: that Qt Quick and `sstvae_core` build and
// deploy together for Android. The waveform constants come from
// `config.hpp`, which is generated from `sstvae/config.py` -- so if this
// renders the right numbers on a phone, the shared core really is linked
// rather than merely compiled.

#include <QDir>
#include <QGuiApplication>
#include <QQmlApplicationEngine>
#include <QQmlContext>
#include <QStandardPaths>
#include <QString>

#include <cstdlib>

#include "config.hpp"
#include "pictures.hpp"

namespace {

// Point the model cache at app-private storage.
//
// `checkpoint::cache_dir()` reads `SSTVAE_MODEL_CACHE` first and
// otherwise follows the platform convention -- which on Android means
// the XDG branch, and `$HOME` is not a useful place on a phone. Setting
// the override is better than exporting `XDG_CACHE_HOME` and hoping:
// this is not an XDG platform, and saying so directly leaves nothing to
// infer. Must happen before anything resolves an artifact.
void set_model_cache() {
    const QString dir =
        QStandardPaths::writableLocation(QStandardPaths::CacheLocation) +
        QStringLiteral("/models");
    QDir().mkpath(dir);
    qputenv("SSTVAE_MODEL_CACHE", dir.toUtf8());
}

}  // namespace

int main(int argc, char* argv[]) {
    QGuiApplication app(argc, argv);
    set_model_cache();

    using namespace sstvae;
    const QString waveform =
        QStringLiteral("%1 carriers x %2 Hz\n%3-%4 Hz\n%5 Hz, frame %6 samples")
            .arg(config::NC)
            .arg(config::RS)
            .arg(config::CARRIER0)
            .arg(config::CARRIER0 + (config::NC - 1) * config::RS)
            .arg(config::FS)
            .arg(config::FRAME_SAMPLES);

    QQmlApplicationEngine engine;
    // Serves both the reception in progress and saved ones; see
    // pictures.hpp for why the two come from different places.
    engine.addImageProvider(QStringLiteral("sstvae"), new PictureProvider);
    engine.rootContext()->setContextProperty("waveformText", waveform);
    engine.loadFromModule("SSTVAE", "Main");
    if (engine.rootObjects().isEmpty()) return 1;
    return app.exec();
}

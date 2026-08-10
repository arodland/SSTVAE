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
    // QSettings keys off these, so they have to be set before anything
    // reads a preference -- an unnamed application writes to a file
    // named after the executable and silently loses everything the
    // next time that name changes.
    QCoreApplication::setOrganizationName(QStringLiteral("SSTVAE"));
    QCoreApplication::setOrganizationDomain(QStringLiteral("cleverdomain.org"));
    QCoreApplication::setApplicationName(QStringLiteral("SSTVAE"));
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
    const int rc = app.exec();

    // **End the process without unwinding.**
    //
    // Returning from here runs static destructors and then lets Android
    // tear the process down around threads that are still using what
    // they destroy. Measured on API 36, backing out of the app:
    //
    //     FORTIFY: pthread_mutex_lock called on a destroyed mutex
    //     Fatal signal 6 (SIGABRT) in tid ... (hwuiTask1)
    //     tombstoned: received crash request
    //
    // Not our thread and not our mutex -- it is Android's own HWUI
    // render threads outliving the graphics state -- but it is a
    // tombstone all the same, so an ordinary exit was being recorded as
    // a native crash, which is exactly what Play's vitals count.
    //
    // `_Exit` skips both: no static destructors, no atexit handlers, no
    // teardown for a late thread to race. Nothing is lost by it. Every
    // `QSettings` write in this app goes through a temporary that syncs
    // when it dies, so no preference is pending here; the audio device,
    // PTT and the engine threads are the service's to release and it
    // does that on its stop path, while the world is still standing;
    // and the rest is memory the kernel reclaims anyway.
    //
    // Third instance of one rule in this codebase, and the reason it
    // keeps recurring: at teardown, *not running code* is the reliable
    // option. `Session` is deliberately immortal for the same reason
    // (it aborted in `~OnnxCodec` from an atexit handler), and
    // `check::Watchdog` calls `std::_Exit` rather than unwinding
    // because a wedged library is somewhere unwinding can hang.
    std::_Exit(rc);
}

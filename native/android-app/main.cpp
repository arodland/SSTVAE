// Tier 0's entry point.
//
// Proving ground at this stage: that Qt Quick and `sstvae_core` build and
// deploy together for Android. The waveform constants come from
// `config.hpp`, which is generated from `sstvae/config.py` -- so if this
// renders the right numbers on a phone, the shared core really is linked
// rather than merely compiled.

#include <QGuiApplication>
#include <QQmlApplicationEngine>
#include <QQmlContext>
#include <QString>

#include "config.hpp"

int main(int argc, char* argv[]) {
    QGuiApplication app(argc, argv);

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
    engine.rootContext()->setContextProperty("waveformText", waveform);
    engine.loadFromModule("SSTVAE", "Main");
    if (engine.rootObjects().isEmpty()) return 1;
    return app.exec();
}

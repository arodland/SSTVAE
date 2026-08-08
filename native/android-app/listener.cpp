#include "listener.hpp"

#include <QJniEnvironment>
#include <QJniObject>
#include <QtCore/qcoreapplication_platform.h>

#include <cmath>

#include "audio/android/androidaudio.hpp"
#include "rx/engine.hpp"
#include "session.hpp"

namespace {

using namespace sstvae;
using sstvae::androidapp::Session;

constexpr const char* kServiceClass = "org/cleverdomain/sstvae/ListenerService";

// The label for "let the platform decide", which has to be
// distinguishable from a device that merely happens to be listed first.
const QString kSystemDefault = QStringLiteral("System default");

// Hand the audio layer the VM and an application Context.
//
// Qt already defines `JNI_OnLoad`, so the layer cannot define its own
// without clashing -- which is why `set_java_vm` is a function rather
// than something the library arranges for itself. Qt exposes both
// pieces we need, so this is four lines rather than a second entry
// point.
bool init_audio_bridge(QString* error) {
    audio::android::set_java_vm(
        reinterpret_cast<audio::android::JavaVM_*>(QJniEnvironment::javaVM()));

    QJniObject ctx = QNativeInterface::QAndroidApplication::context();
    if (!ctx.isValid()) {
        *error = QStringLiteral("no Android context");
        return false;
    }
    QJniObject::callStaticMethod<void>("org/cleverdomain/sstvae/AudioBridge", "init",
                                       "(Landroid/content/Context;)V", ctx.object());
    if (QJniEnvironment().checkAndClearExceptions()) {
        *error = QStringLiteral("AudioBridge.init failed");
        return false;
    }
    return true;
}

}  // namespace

Listener::Listener(QObject* parent) : QObject(parent) {
    if (!init_audio_bridge(&error_)) return;
    refreshDevices();

    // 500 ms, which is a *display* refresh and not a decode cadence --
    // the engine polls on its own 5 s schedule and this only reads what
    // it published.
    poll_.setInterval(500);
    connect(&poll_, &QTimer::timeout, this, &Listener::changed);
    poll_.start();
}

// Nothing to tear down. **Destroying the view must not end the
// session** -- that is the point of the service owning it, and a
// `stop()` here would quietly undo the whole arrangement the first time
// the activity was destroyed on a rotation.
Listener::~Listener() = default;

QStringList Listener::inputDevices() const { return devices_; }

bool Listener::listening() const { return Session::instance().running(); }

QString Listener::lastError() const {
    const std::string e = Session::instance().last_error();
    return e.empty() ? error_ : QString::fromStdString(e);
}

void Listener::refreshDevices() {
    devices_.clear();
    devices_ << kSystemDefault;
    try {
        for (const std::string& n : audio::android::input_device_names()) {
            devices_ << QString::fromStdString(n);
        }
    } catch (const std::exception& e) {
        error_ = QString::fromStdString(e.what());
    }
    emit devicesChanged();
    emit changed();
}

void Listener::start(const QString& deviceName) {
    error_.clear();
    const QString want = deviceName == kSystemDefault ? QString{} : deviceName;
    QJniObject ctx = QNativeInterface::QAndroidApplication::context();
    QJniObject::callStaticMethod<void>(
        kServiceClass, "startListening", "(Landroid/content/Context;Ljava/lang/String;)V",
        ctx.object(), QJniObject::fromString(want).object<jstring>());
    if (QJniEnvironment().checkAndClearExceptions()) {
        error_ = QStringLiteral("could not start the listening service");
    }
    emit changed();
}

void Listener::stop() {
    QJniObject ctx = QNativeInterface::QAndroidApplication::context();
    QJniObject::callStaticMethod<void>(kServiceClass, "stopListening",
                                       "(Landroid/content/Context;)V", ctx.object());
    if (QJniEnvironment().checkAndClearExceptions()) {
        error_ = QStringLiteral("could not stop the listening service");
    }
    emit changed();
}

QString Listener::status() const {
    Session& s = Session::instance();
    if (!s.running()) return QStringLiteral("idle");
    const rx::Progress p = s.progress();
    QString out = QString::fromLatin1(rx::status_name(p.status));
    out += QStringLiteral("   polls %1").arg(p.polls);
    out += QStringLiteral("   ring %1 s").arg(p.seconds_captured, 0, 'f', 1);
    if (p.frames_received.value_or(0) > 0) {
        out += QStringLiteral("\nframes %1").arg(*p.frames_received);
        if (p.n_frames_expected.value_or(0) > 0) {
            out += QStringLiteral("/%1").arg(*p.n_frames_expected);
        }
    }
    if (p.mode_name) {
        out += QStringLiteral("   mode %1").arg(QString::fromStdString(*p.mode_name));
    }
    if (!p.callsign.empty()) {
        out += QStringLiteral("   %1").arg(QString::fromStdString(p.callsign));
    }
    if (!std::isnan(p.snr_db)) {
        out += QStringLiteral("   SNR %1 dB").arg(p.snr_db, 0, 'f', 1);
    }
    return out;
}

QString Listener::audioRoute() const {
    Session& s = Session::instance();
    if (!s.running()) return {};
    QString out = QStringLiteral("%1 Hz -> %2")
                      .arg(s.device_rate())
                      .arg(QString::fromStdString(s.routed_device()));
    const std::string w = s.routing_warning();
    if (!w.empty()) out += QStringLiteral("\n! %1").arg(QString::fromStdString(w));
    return out;
}

QString Listener::level() const {
    Session& s = Session::instance();
    if (!s.running()) return {};
    const double peak = s.peak_level();
    const QString db = peak > 0.0
                           ? QStringLiteral("%1 dBFS").arg(20 * std::log10(peak), 0, 'f', 0)
                           : QStringLiteral("silent");
    // Both numbers, always: quiet and silent have the same mean level
    // and are not the same failure.
    return QStringLiteral("peak %1   %2% near-zero")
        .arg(db)
        .arg(100.0 * s.near_zero_fraction(), 0, 'f', 1);
}

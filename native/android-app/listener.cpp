#include "listener.hpp"

#include <QJniEnvironment>
#include <QJniObject>
#include <QtCore/qcoreapplication_platform.h>

#include <cmath>

#include "config.hpp"
#include "images/types.hpp"

namespace {

using namespace sstvae;

// Hand the audio layer the VM and an application Context.
//
// Qt already defines `JNI_OnLoad`, so the layer cannot define its own
// without clashing -- which is why `set_java_vm` is a function rather
// than something the library arranges for itself. Qt exposes both pieces
// we need, so this is four lines rather than a second entry point.
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

Listener::~Listener() { stop(); }

QStringList Listener::inputDevices() const { return devices_; }

void Listener::refreshDevices() {
    devices_.clear();
    // Empty first: "whatever the platform routes to" is a real choice and
    // has to be distinguishable from a device that happens to be listed
    // first.
    devices_ << QStringLiteral("System default");
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
    if (stream_) return;
    error_.clear();

    ring_ = std::make_unique<rx::RingBuffer>(130.0);
    state_ = std::make_unique<rx::SharedState>();
    stop_ = std::make_unique<rx::StopFlag>();

    // No codec yet: the decoder is a seam, so the whole state machine
    // runs and still reports mode, frames, SNR and callsign without one.
    // That is most of what validating this layer needs, and it is the
    // same reason `rx::Decoder` exists at all.
    rx::Decoder decoder = [](std::span<const double>, std::span<const double>) {
        return images::Picture{};
    };
    rx::RxConfig cfg;
    cfg.poll_interval = 5.0;
    rx::Sink sink = [](const rx::Reception&) -> std::optional<std::string> {
        return std::nullopt;
    };

    const std::string want =
        deviceName == QStringLiteral("System default") ? std::string{}
                                                       : deviceName.toStdString();
    try {
        stream_ = std::make_unique<audio::android::InputStream>(
            want, *ring_, config::FS,
            [this](const std::string&) { emit changed(); },
            [this](const std::string& m) { error_ = QString::fromStdString(m); });
    } catch (const std::exception& e) {
        error_ = QString::fromStdString(e.what());
        ring_.reset();
        state_.reset();
        stop_.reset();
        emit changed();
        return;
    }

    thread_ = std::thread([this, decoder, cfg, sink] {
        try {
            rx::decode_loop(*ring_, decoder, *state_, cfg, *stop_, sink);
        } catch (const std::exception&) {
            // One bad session must not take the process down; the error
            // surfaces as the loop simply having stopped advancing.
        }
    });
    emit changed();
}

void Listener::stop() {
    if (stream_) stream_->stop();
    if (stop_) stop_->set();
    if (thread_.joinable()) thread_.join();
    stream_.reset();
    ring_.reset();
    state_.reset();
    stop_.reset();
    emit changed();
}

QString Listener::status() const {
    if (!state_) return QStringLiteral("idle");
    const rx::Progress p = state_->get();
    QString s = QString::fromLatin1(rx::status_name(p.status));
    s += QStringLiteral("   polls %1").arg(p.polls);
    s += QStringLiteral("   ring %1 s").arg(p.seconds_captured, 0, 'f', 1);
    if (p.frames_received.value_or(0) > 0) {
        s += QStringLiteral("\nframes %1").arg(*p.frames_received);
        if (p.n_frames_expected.value_or(0) > 0) {
            s += QStringLiteral("/%1").arg(*p.n_frames_expected);
        }
    }
    if (p.mode_name) s += QStringLiteral("   mode %1").arg(QString::fromStdString(*p.mode_name));
    if (!p.callsign.empty()) {
        s += QStringLiteral("   %1").arg(QString::fromStdString(p.callsign));
    }
    if (!std::isnan(p.snr_db)) s += QStringLiteral("   SNR %1 dB").arg(p.snr_db, 0, 'f', 1);
    return s;
}

QString Listener::audioRoute() const {
    if (!stream_) return {};
    QString s = QStringLiteral("%1 Hz -> %2")
                    .arg(stream_->device_rate())
                    .arg(QString::fromStdString(stream_->routed_device()));
    const std::string w = stream_->routing_warning();
    if (!w.empty()) s += QStringLiteral("\n! %1").arg(QString::fromStdString(w));
    return s;
}

QString Listener::level() const {
    if (!stream_) return {};
    const double peak = stream_->peak_level();
    const QString db = peak > 0.0
                           ? QStringLiteral("%1 dBFS").arg(20 * std::log10(peak), 0, 'f', 0)
                           : QStringLiteral("silent");
    // Both numbers, always: quiet and silent have the same mean level and
    // are not the same failure.
    return QStringLiteral("peak %1   %2% near-zero")
        .arg(db)
        .arg(100.0 * stream_->near_zero_fraction(), 0, 'f', 1);
}

#include "listener.hpp"

#include <QDir>
#include <QJniEnvironment>
#include <QJniObject>
#include <QPermissions>
#include <QSettings>
#include <QStandardPaths>
#include <QtCore/qcoreapplication_platform.h>

#include <cmath>

#include "assets.hpp"
#include "audio/android/androidaudio.hpp"
#include "rx/engine.hpp"
#include "session.hpp"

namespace {

using namespace sstvae;
using sstvae::androidapp::Session;

constexpr const char* kServiceClass = "org/cleverdomain/sstvae/ListenerService";

// Hold the screen while a session is running and this window is up.
// A window flag rather than a wake lock, so the platform drops it on
// backgrounding by itself -- see ScreenOn.java.
void request_notification_permission() {
    if (QNativeInterface::QAndroidApplication::sdkVersion() < 33) return;
    QJniObject ctx = QNativeInterface::QAndroidApplication::context();
    if (!ctx.isValid()) return;
    QJniObject::callStaticMethod<void>("org/cleverdomain/sstvae/Permissions",
                                       "requestNotifications",
                                       "(Landroid/content/Context;)V", ctx.object());
    QJniEnvironment().checkAndClearExceptions();
}

void keep_screen_on(bool on) {
    QJniObject ctx = QNativeInterface::QAndroidApplication::context();
    if (!ctx.isValid()) return;
    QJniObject::callStaticMethod<void>("org/cleverdomain/sstvae/ScreenOn", "set",
                                       "(Landroid/content/Context;Z)V", ctx.object(),
                                       static_cast<jboolean>(on));
    QJniEnvironment().checkAndClearExceptions();
}

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

namespace {
constexpr auto kTechnicalKey = "ui/showTechnical";
constexpr auto kGalleryKey = "receive/saveToGallery";
}  // namespace

void Listener::setShowTechnical(bool on) {
    if (on == technical_) return;
    technical_ = on;
    QSettings().setValue(QLatin1String(kTechnicalKey), on);
    // The notification is built from the session, not from here, and
    // it is drawn by a service that may outlive this object -- so the
    // flag has to reach the session rather than being consulted only
    // where the QML asks for it. Otherwise the shade would keep
    // reporting poll counts with the switch off.
    Session::instance().set_show_technical(on);
    emit changed();
}

// Exporting is done by the service, on the Java side where the Context
// is, so this only records the choice and hands it to the session for
// the service to read. Turning it on does not export the receptions
// already on disk: they can go out through Share, and silently copying a
// backlog into someone's camera roll is exactly the surprise the switch
// exists to prevent.
void Listener::setSaveToGallery(bool on) {
    if (on == gallery_) return;
    gallery_ = on;
    QSettings().setValue(QLatin1String(kGalleryKey), on);
    Session::instance().set_save_to_gallery(on);
    // A stale failure from the last time it was on would otherwise
    // outlive the setting itself.
    if (!on) Session::instance().set_gallery_error(std::string());
    emit changed();
}

QString Listener::galleryError() const {
    return QString::fromStdString(Session::instance().gallery_error());
}

Listener::Listener(QObject* parent) : QObject(parent) {
    technical_ = QSettings().value(QLatin1String(kTechnicalKey), false).toBool();
    Session::instance().set_show_technical(technical_);
    gallery_ = QSettings().value(QLatin1String(kGalleryKey), false).toBool();
    Session::instance().set_save_to_gallery(gallery_);
    // Resolve the AssetManager here, on the UI thread, because that is
    // the only thread with a Java context to ask -- after this the
    // bundled models are reachable from the model thread with no JNI at
    // all. A false return means no bundled artifacts, which the codec
    // treats as "fetch them" rather than as a failure, so it is not
    // fatal and deliberately does not set `error_`.
    androidapp::assets::init();

    if (!init_audio_bridge(&error_)) return;
    refreshDevices();

    // Receptions land in app-private storage, and that stays true even
    // with `saveToGallery` on: the sidecar is what makes a picture
    // answerable a week later, and MediaStore has nowhere to put it. The
    // shared-gallery copy is a mirror written by the service (see
    // `Gallery.java`), never the original.
    const QString pics =
        QStandardPaths::writableLocation(QStandardPaths::AppDataLocation) +
        QStringLiteral("/pictures");
    QDir().mkpath(pics);
    Session::instance().set_picture_dir(pics.toStdString());

    // The ongoing notification needs POST_NOTIFICATIONS from API 33.
    // Fire and forget: the answer is not waited on because denial is
    // not fatal -- it costs the notification, not the session -- and
    // the service is written to treat it that way.
    request_notification_permission();

    // Start fetching the decoder now rather than at the first
    // reception. It is a ~9 MB download on a first run, and the moment
    // a picture is arriving is the worst possible time to begin it.
    loadModel();

    // 500 ms, which is a *display* refresh and not a decode cadence --
    // the engine polls on its own 5 s schedule and this only reads what
    // it published.
    poll_.setInterval(500);
    connect(&poll_, &QTimer::timeout, this, [this] {
        // Bump the image id only when the engine actually published a
        // different picture. Bumping every tick would re-decode and
        // re-upload a 900 kB image twice a second for nothing.
        const auto p = Session::instance().progress();
        if (p.image.get() != last_image_) {
            last_image_ = p.image.get();
            ++live_id_;
        }
        // Driven from the poll rather than from start()/stop(),
        // because the session can also end without the UI asking --
        // a capture error, or the service being stopped from the
        // notification. Tracking the button would leave the screen
        // pinned on after one of those.
        const bool live = Session::instance().running();
        if (live != screen_held_) {
            screen_held_ = live;
            keep_screen_on(live);
        }
        emit changed();
    });
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

// Ask for the microphone, then start.
//
// **Nothing is pre-granted on an ordinary sideload** -- `adb install
// -g` had been hiding this throughout development, which is its own
// small lesson about testing the install path people will actually
// use. Qt's permission type is used rather than raw JNI because the
// result arrives asynchronously through the activity, and Qt already
// owns that plumbing.
//
// The request is made from `start()` rather than at launch, so the
// prompt arrives when the operator has just asked to listen and its
// reason is self-evident. It is also required to be here: a
// microphone-typed foreground service may not be started without the
// permission, so the service call has to wait for the answer.
void Listener::start(const QString& deviceName) {
    error_.clear();
    const QString want = deviceName == kSystemDefault ? QString{} : deviceName;

    const QMicrophonePermission mic;
    switch (qApp->checkPermission(mic)) {
        case Qt::PermissionStatus::Undetermined:
            qApp->requestPermission(mic, this, [this, deviceName](const QPermission& p) {
                if (p.status() == Qt::PermissionStatus::Granted) {
                    start(deviceName);
                } else {
                    error_ = tr("Microphone access is required to receive.");
                    emit changed();
                }
            });
            return;
        case Qt::PermissionStatus::Denied:
            // Android stops showing the prompt after a denial, so
            // repeating the request here would look like the button
            // doing nothing at all. Say where to fix it instead.
            error_ = tr("Microphone access is denied. Enable it in "
                        "Settings > Apps > SSTVAE > Permissions.");
            emit changed();
            return;
        case Qt::PermissionStatus::Granted:
            break;
    }
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

// What the status line says with the technical detail switched off.
//
// Less, but not merely less: the plain line answers "is a picture
// coming and how far along is it", which is the question the operator
// has, while the technical one answers "is the receiver working",
// which is the question a bug has. Empty while merely listening,
// because the placeholder under the waterfall already says so and
// repeating it twice on one screen is noise.
QString Listener::plain_status() const {
    const rx::Progress p = Session::instance().progress();
    if (p.status == rx::Status::Done) return QStringLiteral("Picture complete");
    if (p.status != rx::Status::Receiving) return {};

    QString out = QStringLiteral("Receiving a picture");
    if (!p.callsign.empty()) {
        out = QStringLiteral("Receiving from %1").arg(QString::fromStdString(p.callsign));
    }
    // A percentage rather than a frame count. Frames are the unit the
    // modem thinks in and mean nothing to the operator; the fraction is
    // already computed for both the header and the blind path, which
    // are counted differently and would need two branches here.
    if (p.progress_frac > 0.0) {
        out += QStringLiteral("  %1%").arg(100.0 * p.progress_frac, 0, 'f', 0);
    }
    return out;
}

QString Listener::status() const {
    Session& s = Session::instance();
    if (!s.running()) return technical_ ? QStringLiteral("idle") : QString();
    if (!technical_) return plain_status();
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
    // The sample rate is a fact about the driver, not about the
    // radio; the device *name* is what an operator is checking. The
    // warning below is neither -- it fires only when the audio is
    // genuinely coming from somewhere other than was asked for, which
    // is actionable at any level of interest, so it is never hidden.
    QString out = technical_ ? QStringLiteral("%1 Hz -> %2")
                                   .arg(s.device_rate())
                                   .arg(QString::fromStdString(s.routed_device()))
                             : QString::fromStdString(s.routed_device());
    const std::string w = s.routing_warning();
    if (!w.empty()) out += QStringLiteral("\n! %1").arg(QString::fromStdString(w));
    return out;
}

QString Listener::level() const {
    Session& s = Session::instance();
    // The whole line is diagnostics -- peak, near-zero fraction,
    // capture drift, decode cost. There is no plain-language half of
    // it worth keeping: the one judgement an operator needs from it is
    // already on the meter as a colour and a word.
    if (!s.running() || !technical_) return {};
    const double peak = s.peak_level();
    const QString db = peak > 0.0
                           ? QStringLiteral("%1 dBFS").arg(20 * std::log10(peak), 0, 'f', 0)
                           : QStringLiteral("silent");
    // Both numbers, always: quiet and silent have the same mean level
    // and are not the same failure.
    QString out = QStringLiteral("peak %1   %2% near-zero")
                      .arg(db)
                      .arg(100.0 * s.near_zero_fraction(), 0, 'f', 1);
    // Shown always, not only when bad. A number that appears only on
    // failure teaches nobody what healthy looks like, and this is the
    // one measurement that distinguishes "the band is quiet" from "the
    // capture path is eating your audio".
    const double ppm = s.capture_drift_ppm();
    if (ppm != 0.0) {
        out += QStringLiteral("\ncapture %1%2 ppm")
                   .arg(ppm > 0 ? "+" : "")
                   .arg(ppm, 0, 'f', 0);
        if (ppm < -1000.0) out += QStringLiteral("  DROPPING AUDIO");
    }
    // **"dsp", not "decode".** `last_decode_s` is measured before the
    // codec runs (see rx/engine.cpp, and the Python reference it
    // mirrors), so it is sync plus demodulation and no inference at
    // all. Labelling it "decode" sent one investigation straight at
    // onnxruntime when every millisecond of it was in the DSP. The
    // adaptive backoff uses the true whole-poll cost, which is
    // measured separately and is not this number.
    const double dsp = s.progress().last_decode_s;
    if (dsp > 0.0) {
        out += QStringLiteral("   dsp %1 s").arg(dsp, 0, 'f', 1);
    }
    return out;
}

void Listener::loadModel() {
    Session::instance().load_model_async();
    emit changed();
}

bool Listener::modelReady() const {
    return Session::instance().model_state() == sstvae::androidapp::ModelState::Ready;
}

QString Listener::modelStatus() const {
    using sstvae::androidapp::ModelState;
    Session& s = Session::instance();
    switch (s.model_state()) {
        case ModelState::Ready:
            return QStringLiteral("model ready");
        case ModelState::Loading:
            return QStringLiteral("loading model...");
        case ModelState::Downloading: {
            const auto [got, total] = s.model_progress();
            if (total > 0) {
                return QStringLiteral("downloading model  %1%")
                    .arg(100.0 * static_cast<double>(got) / static_cast<double>(total),
                         0, 'f', 0);
            }
            return QStringLiteral("downloading model  %1 kB").arg(got / 1024);
        }
        case ModelState::Failed:
            // Named as a *decode* consequence rather than a load error,
            // because that is what it costs the operator: the station
            // still hears everything, and only the picture is missing.
            return QStringLiteral("no model - receiving without pictures\n%1")
                .arg(QString::fromStdString(s.model_error()));
        case ModelState::Absent:
        default:
            return QStringLiteral("no model");
    }
}

bool Listener::hasLiveImage() const {
    const auto p = Session::instance().progress();
    return p.image && p.image->width > 0;
}

void Listener::sharePicture(const QString& path, const QString& caption) {
    QJniObject ctx = QNativeInterface::QAndroidApplication::context();
    if (!ctx.isValid()) return;
    QJniObject::callStaticMethod<void>(
        "org/cleverdomain/sstvae/Sharing", "share",
        "(Landroid/content/Context;Ljava/lang/String;Ljava/lang/String;)V", ctx.object(),
        QJniObject::fromString(path).object<jstring>(),
        QJniObject::fromString(caption).object<jstring>());
    if (QJniEnvironment().checkAndClearExceptions()) {
        error_ = tr("Could not share that picture.");
        emit changed();
    }
}

double Listener::peakLevel() const { return Session::instance().peak_level(); }
double Listener::driftPpm() const { return Session::instance().capture_drift_ppm(); }

// The threshold the meter turns red on. -1000 ppm is the project's own
// scale: a clean path measured +211 ppm, and 3500 ppm of loss cost 5 dB
// and the picture. Past this it is audio being dropped, not a crystal
// being imprecise.
bool Listener::droppingAudio() const {
    return Session::instance().capture_drift_ppm() < -1000.0;
}

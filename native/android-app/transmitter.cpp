#include "transmitter.hpp"

#include <QJniEnvironment>
#include <QJniObject>
#include <QMetaObject>
#include <QSettings>
#include <QtCore/qcoreapplication_platform.h>

#include <jni.h>

#include <atomic>
#include <cmath>

#include "audio/android/androidaudio.hpp"
#include "composition.hpp"
#include "config.hpp"
#include "dsp/leader.hpp"
#include "session.hpp"
#include "tx/engine.hpp"

namespace {

using namespace sstvae;
using sstvae::androidapp::Composition;
using sstvae::androidapp::ModelState;
using sstvae::androidapp::Session;

constexpr const char* kServiceClass = "org/cleverdomain/sstvae/ListenerService";
constexpr const char* kPickerClass = "org/cleverdomain/sstvae/ImagePicker";

// The label for "let the platform decide", which has to be
// distinguishable from a device that merely happens to be listed first.
const QString kSystemDefault = QStringLiteral("System default");

// One spelling per setting, used for both the read and the write. The
// characteristic settings bug is a field displayed and never written
// back, and a key spelled twice is how it happens.
constexpr auto kCallsign = "station/callsign";
constexpr auto kMode = "transmit/mode";
constexpr auto kLevel = "transmit/level";
constexpr auto kCwId = "transmit/cwId";
constexpr auto kCwMessage = "transmit/cwMessage";
constexpr auto kVoxLead = "transmit/voxLeadSeconds";
constexpr auto kOutputDevice = "audio/outputDevice";
// Versioned in the name, so that if the prompt ever has to say
// something materially different it can be asked again by bumping the
// key rather than by adding a second flag beside a stale one.
constexpr auto kAcknowledged = "transmit/firstTransmitAcknowledgedV1";

// The instance the picker's result goes to.
//
// A pointer rather than plumbing, because the alternative is worse: the
// activity result arrives on the Android UI thread through a static Java
// callback, and there is no `this` to carry along it. QML creates
// exactly one Transmitter, and the pointer is cleared in the destructor,
// so a result arriving after the view is gone is dropped rather than
// delivered to freed memory -- which is a real sequence here, since the
// picker can outlive a rotation.
std::atomic<Transmitter*> g_active{nullptr};

const sstvae::config::ModeSpec* find_mode(const QString& name) {
    for (const auto& m : config::MODES) {
        if (QString::fromLatin1(m.name.data(), static_cast<int>(m.name.size())) == name) {
            return &m;
        }
    }
    return nullptr;
}

}  // namespace

Transmitter::Transmitter(QObject* parent) : QObject(parent) {
    QSettings s;
    callsign_ = s.value(QLatin1String(kCallsign)).toString();
    mode_ = s.value(QLatin1String(kMode), QStringLiteral("B")).toString();
    level_ = s.value(QLatin1String(kLevel), 0.9).toDouble();
    cw_id_ = s.value(QLatin1String(kCwId), false).toBool();
    cw_message_ =
        s.value(QLatin1String(kCwMessage), QStringLiteral("SSTVAE DE {callsign}")).toString();
    // **Off by default**, because it is only right for a VOX-keyed
    // station and it is airtime everyone else would pay for silently.
    vox_lead_s_ = s.value(QLatin1String(kVoxLead), 0.0).toDouble();
    acknowledged_ = s.value(QLatin1String(kAcknowledged), false).toBool();
    device_ = s.value(QLatin1String(kOutputDevice)).toString();
    if (find_mode(mode_) == nullptr) mode_ = QStringLiteral("B");

    refreshDevices();

    // A display refresh, like the receive side's. The engine publishes
    // progress from the audio callback and this only reads it.
    poll_.setInterval(500);
    connect(&poll_, &QTimer::timeout, this, [this] { emit changed(); });
    poll_.start();

    g_active.store(this);
}

Transmitter::~Transmitter() { g_active.store(nullptr); }

void Transmitter::bump() {
    ++preview_id_;
    emit changed();
}

// --- the picture ------------------------------------------------------

bool Transmitter::hasPicture() const { return Composition::instance().has_source(); }
double Transmitter::zoom() const { return Composition::instance().framing().zoom; }

double Transmitter::minZoom() const {
    const Composition& c = Composition::instance();
    return images::min_zoom(c.source_width(), c.source_height());
}
double Transmitter::centerX() const { return Composition::instance().framing().center_x; }
double Transmitter::centerY() const { return Composition::instance().framing().center_y; }

void Transmitter::setZoom(double z) {
    images::Framing f = Composition::instance().framing();
    f.zoom = z;
    Composition::instance().set_framing(f);
    bump();
}

void Transmitter::panBy(double dx, double dy) {
    Composition::instance().pan(dx, dy);
    bump();
}

void Transmitter::clearPicture() {
    Composition::instance().clear();
    bump();
}

void Transmitter::pickImage() {
    QJniObject ctx = QNativeInterface::QAndroidApplication::context();
    if (!ctx.isValid()) return;
    QJniObject::callStaticMethod<void>(kPickerClass, "pick",
                                       "(Landroid/content/Context;)V", ctx.object());
    if (QJniEnvironment().checkAndClearExceptions()) {
        error_ = tr("Could not open the picture chooser.");
        emit changed();
    }
}

void Transmitter::takePhoto() {
    QJniObject ctx = QNativeInterface::QAndroidApplication::context();
    if (!ctx.isValid()) return;
    QJniObject::callStaticMethod<void>(kPickerClass, "capture",
                                       "(Landroid/content/Context;)V", ctx.object());
    if (QJniEnvironment().checkAndClearExceptions()) {
        error_ = tr("Could not open the camera.");
        emit changed();
    }
}

void Transmitter::onPicked(const QString& path, const QString& error) {
    if (!error.isEmpty()) {
        error_ = error;
        emit changed();
        return;
    }
    // Both empty is the picker being backed out of, which is the most
    // ordinary thing that happens there. Keep whatever was already
    // composed rather than treating it as a failed load.
    if (path.isEmpty()) return;
    std::string why;
    if (!Composition::instance().set_source(path.toStdString(), &why)) {
        error_ = QString::fromStdString(why);
    } else {
        error_.clear();
    }
    bump();
}

// --- settings ---------------------------------------------------------

void Transmitter::setCallsign(const QString& c) {
    // Upper case, because that is how a callsign is written and because
    // the beacon's character set is A-Z, 0-9 and '/' -- lower case would
    // be dropped on the way out, which reads as the app losing it.
    const QString up = c.toUpper();
    if (up == callsign_) return;
    callsign_ = up;
    QSettings().setValue(QLatin1String(kCallsign), callsign_);
    emit changed();
}

void Transmitter::setMode(const QString& m) {
    if (m == mode_ || find_mode(m) == nullptr) return;
    mode_ = m;
    QSettings().setValue(QLatin1String(kMode), mode_);
    emit changed();
}

QStringList Transmitter::modes() const {
    QStringList out;
    for (const auto& m : config::MODES) {
        out << QString::fromLatin1(m.name.data(), static_cast<int>(m.name.size()));
    }
    return out;
}

void Transmitter::setLevel(double v) {
    if (std::abs(v - level_) < 1e-9) return;
    level_ = v;
    QSettings().setValue(QLatin1String(kLevel), level_);
    emit changed();
}

void Transmitter::setCwId(bool on) {
    if (on == cw_id_) return;
    cw_id_ = on;
    QSettings().setValue(QLatin1String(kCwId), cw_id_);
    emit changed();
}

void Transmitter::setCwMessage(const QString& m) {
    if (m == cw_message_) return;
    cw_message_ = m;
    QSettings().setValue(QLatin1String(kCwMessage), cw_message_);
    emit changed();
}

void Transmitter::setVoxLead(double s) {
    if (std::abs(s - vox_lead_s_) < 1e-9) return;
    vox_lead_s_ = s;
    QSettings().setValue(QLatin1String(kVoxLead), vox_lead_s_);
    emit changed();
}

void Transmitter::setOutputDevice(const QString& d) {
    if (d == device_) return;
    device_ = d;
    QSettings().setValue(QLatin1String(kOutputDevice), device_);
    emit changed();
}

void Transmitter::refreshDevices() {
    devices_.clear();
    devices_ << kSystemDefault;
    try {
        for (const std::string& n : audio::android::output_device_names()) {
            devices_ << QString::fromStdString(n);
        }
    } catch (const std::exception& e) {
        error_ = QString::fromStdString(e.what());
    }
    emit devicesChanged();
    emit changed();
}

// --- the over ---------------------------------------------------------

void Transmitter::loadEncoder() {
    Session::instance().preload_encoder_async();
    emit changed();
}

bool Transmitter::encoderReady() const {
    return Session::instance().encoder_state() == ModelState::Ready;
}

QString Transmitter::encoderStatus() const {
    Session& s = Session::instance();
    switch (s.encoder_state()) {
        case ModelState::Ready:
            return QStringLiteral("encoder ready");
        case ModelState::Loading:
            return QStringLiteral("loading encoder...");
        case ModelState::Downloading: {
            const auto [got, total] = s.model_progress();
            if (total > 0) {
                return QStringLiteral("downloading encoder  %1%")
                    .arg(100.0 * static_cast<double>(got) / static_cast<double>(total),
                         0, 'f', 0);
            }
            return QStringLiteral("downloading encoder  %1 kB").arg(got / 1024);
        }
        case ModelState::Failed:
            return QStringLiteral("no encoder - cannot transmit\n%1")
                .arg(QString::fromStdString(s.encoder_error()));
        case ModelState::Absent:
        default:
            // Named as what it costs rather than as a state, and phrased
            // as a download because that is what tapping will do: the
            // encoder is a separate 9 MB artifact a receive-only station
            // never fetches.
            return QStringLiteral("encoder not downloaded yet");
    }
}

bool Transmitter::transmitting() const { return Session::instance().transmitting(); }

QString Transmitter::cwIdProblem() const {
    return QString::fromStdString(tx::cw_id_problem(
        cw_id_, cw_message_.toStdString(), callsign_.toStdString()));
}

void Transmitter::acknowledgeFirstTransmit() {
    if (acknowledged_) return;
    acknowledged_ = true;
    QSettings().setValue(QLatin1String(kAcknowledged), true);
    emit changed();
}

bool Transmitter::canSend() const {
    // **The CW check blocks; the first-transmit prompt does not.** They
    // are different kinds of thing: a broken CW ID is a setting that
    // cannot do what it says, and the only fix is in Settings, so Send
    // stays disabled with the reason on screen. The prompt is something
    // to read once, and it is reached *through* Send -- disabling the
    // button would leave nothing to press to get to it.
    return hasPicture() && encoderReady() && !transmitting() &&
           cwIdProblem().isEmpty();
}

QString Transmitter::txStatus() const {
    Session& s = Session::instance();
    const tx::TxState t = s.tx_state();
    if (s.transmitting()) {
        if (t.phase == tx::TxPhase::Sending) {
            return QStringLiteral("Transmitting  %1%").arg(100.0 * t.progress, 0, 'f', 0);
        }
        return QString::fromStdString(t.message.empty()
                                          ? std::string(tx::phase_name(t.phase))
                                          : t.message);
    }
    switch (t.phase) {
        case tx::TxPhase::Done:
            return QStringLiteral("Sent");
        case tx::TxPhase::Cancelled:
            return QStringLiteral("Cancelled");
        case tx::TxPhase::Failed:
            return QStringLiteral("Transmission failed");
        default:
            return {};
    }
}

double Transmitter::txProgress() const {
    return Session::instance().tx_state().progress;
}

QString Transmitter::airtime() const {
    const config::ModeSpec* m = find_mode(mode_);
    if (m == nullptr) return {};
    double seconds = m->duration_s;
    if (vox_lead_s_ > 0.0) seconds += vox_lead_s_ + dsp::VOX_LEAD_GAP_S;
    // The CW ID's length depends on the message, so it is deliberately
    // not counted here rather than guessed at: a figure that is
    // sometimes wrong is worse than one that is consistently the
    // picture's own airtime.
    return QStringLiteral("%1 s").arg(seconds, 0, 'f', 0);
}

QString Transmitter::lastError() const {
    const std::string e = Session::instance().last_error();
    return e.empty() ? error_ : QString::fromStdString(e);
}

void Transmitter::send() {
    if (!canSend()) return;
    error_.clear();

    Session::TxRequest req;
    // **Composed now, and committed to.** The transmitting thread gets
    // the picture as it was at the tap; moving the crop afterwards
    // belongs to the next over, which is the rule the desktop settled on
    // for the same reason -- what is in flight has to keep describing
    // the picture going out.
    req.picture = Composition::instance().preview();
    req.mode = mode_.toStdString();
    req.callsign = callsign_.toStdString();
    req.level = level_;
    req.cw_id = cw_id_;
    req.cw_message = cw_message_.toStdString();
    req.vox_lead_s = vox_lead_s_;
    req.output_device = device_ == kSystemDefault ? std::string{} : device_.toStdString();
    Session::instance().stage_transmit(std::move(req));

    QJniObject ctx = QNativeInterface::QAndroidApplication::context();
    QJniObject::callStaticMethod<void>(kServiceClass, "transmit",
                                       "(Landroid/content/Context;)V", ctx.object());
    if (QJniEnvironment().checkAndClearExceptions()) {
        error_ = tr("Could not start the transmit service.");
    }
    emit changed();
}

void Transmitter::cancel() {
    QJniObject ctx = QNativeInterface::QAndroidApplication::context();
    QJniObject::callStaticMethod<void>(kServiceClass, "cancelTransmit",
                                       "(Landroid/content/Context;)V", ctx.object());
    QJniEnvironment().checkAndClearExceptions();
    emit changed();
}

// --- the picker's result ----------------------------------------------
//
// Arrives on the Android UI thread from a static Java callback, so it is
// marshalled onto the Transmitter's own thread before anything is read
// or written. `Composition::set_source` decodes a photograph, which is
// not work for whichever thread the platform happened to call us on.
extern "C" JNIEXPORT void JNICALL Java_org_cleverdomain_sstvae_ImagePicker_nativePicked(
    JNIEnv* env, jclass, jstring jpath, jstring jerror) {
    const auto to_qstring = [env](jstring s) {
        if (s == nullptr) return QString{};
        const char* c = env->GetStringUTFChars(s, nullptr);
        QString out = QString::fromUtf8(c == nullptr ? "" : c);
        if (c != nullptr) env->ReleaseStringUTFChars(s, c);
        return out;
    };
    const QString path = to_qstring(jpath);
    const QString error = to_qstring(jerror);

    Transmitter* t = g_active.load();
    if (t == nullptr) return;
    QMetaObject::invokeMethod(
        t, [t, path, error] { t->onPicked(path, error); }, Qt::QueuedConnection);
}

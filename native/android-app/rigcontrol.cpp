#include "rigcontrol.hpp"

#include <QLatin1String>
#include <QPointer>
#include <QSettings>
#include <QGuiApplication>
#include <QVariantMap>

#include <algorithm>
#include <memory>
#include <optional>

#include "rig/android/androidrig.hpp"
#include "rig/bridged.hpp"
#include "rig/hamlib.hpp"
#include "session.hpp"

// `SSTVAE_ANDROID_RIG=OFF` is a supported configuration -- the one
// escape hatch for a build environment whose NDK cannot cross-compile
// Hamlib's autotools tarball. It costs CAT and keeps the app, which is
// the right way round. Only the calls that need libhamlib are guarded:
// the transport, the settings and the whole of this screen still build,
// so turning it off is not a second code path to keep working.

using namespace sstvae;
// Targeted, alongside the namespace directive, exactly as `listener.cpp`
// and `transmitter.cpp` do it: `Session` is in `sstvae::androidapp`,
// which `using namespace sstvae` does not reach into.
using sstvae::androidapp::Session;

namespace {

// Spelled once each, per the discipline `transmitter.hpp` records: a
// setting displayed and written back through two different spellings is
// a setting that silently resets.
constexpr auto kEnabledKey = "rig/enabled";
constexpr auto kConnectionKey = "rig/connection";
constexpr auto kModelKey = "rig/model";
constexpr auto kDeviceKey = "rig/device";
constexpr auto kHostKey = "rig/host";
constexpr auto kBaudKey = "rig/baud";
constexpr auto kPttKey = "rig/ptt";

// How many rows a model search returns. Hamlib knows several hundred
// rigs; a phone list view of all of them is a scroll, not a picker.
constexpr int kMaxModelRows = 60;

rig::PttMethod ptt_from(const QString& name) {
    if (name == QLatin1String("cat")) return rig::PttMethod::Cat;
    if (name == QLatin1String("dtr")) return rig::PttMethod::Dtr;
    if (name == QLatin1String("rts")) return rig::PttMethod::Rts;
    return rig::PttMethod::Vox;
}

QVariantMap model_row(const rig::RigModel& m) {
    QVariantMap row;
    row[QStringLiteral("number")] = m.model;
    row[QStringLiteral("label")] = QString::fromStdString(m.label());
    return row;
}

// Cached because `rig::list_models()` walks and initialises every
// backend Hamlib has -- it is not cheap, and this is called on every
// keystroke of the search box. The first call is on the UI thread when
// the rig screen first opens, which is the same place the desktop pays
// it.
const std::vector<rig::RigModel>& all_models() {
#ifdef SSTVAE_ANDROID_HAVE_RIG
    static const std::vector<rig::RigModel> models = rig::list_models();
#else
    static const std::vector<rig::RigModel> models;
#endif
    return models;
}

bool have_rig_support() {
#ifdef SSTVAE_ANDROID_HAVE_RIG
    return true;
#else
    return false;
#endif
}

}  // namespace

RigControl::RigControl(QObject* parent) : QObject(parent) {
    load();

    // A guarded pointer, not a raw capture: the callback is queued, so
    // an event can still be in flight when a rotation destroys this
    // view. Clearing the callback in the destructor closes the window
    // for new ones and this closes it for the one already posted.
    QPointer<RigControl> self(this);
    rig::android::set_permission_callback(
        [self](const std::string& id, bool granted) {
            // Arrives on Android's broadcast thread. Hopping to this
            // object's thread is what makes touching `devices_` legal;
            // a queued connection is the mechanism Qt provides for it.
            QMetaObject::invokeMethod(
                qApp,
                [self, id = QString::fromStdString(id), granted] {
                    if (self) self->publish_permission(id, granted);
                },
                Qt::QueuedConnection);
        });

    // Slower than the receive screen's, because there is nothing here
    // that moves quickly: the rig itself is polled every 10 s, so a
    // faster refresh would redraw the same frequency several times.
    poll_.setInterval(1000);
    connect(&poll_, &QTimer::timeout, this, &RigControl::changed);
    poll_.start();

    // A rig session survives this object, so a screen rotation must not
    // reopen the radio -- but a *first* launch with rig control on
    // should connect without the operator pressing anything.
    if (!have_rig_support()) enabled_ = false;
    if (enabled_ && !Session::instance().rig_running()) connectRig();
}

RigControl::~RigControl() {
    // Deliberately does **not** stop the rig. The session outlives this
    // view by design: a rotation during an over must not drop PTT, and
    // this destructor runs on exactly that path.
    rig::android::set_permission_callback({});
}

// --- settings ---------------------------------------------------------------

void RigControl::load() {
    QSettings s;
    enabled_ = s.value(QLatin1String(kEnabledKey), false).toBool();
    connection_ = s.value(QLatin1String(kConnectionKey), connection_).toString();
    model_ = s.value(QLatin1String(kModelKey), model_).toInt();
    device_ = s.value(QLatin1String(kDeviceKey), QString()).toString();
    host_ = s.value(QLatin1String(kHostKey), QString()).toString();
    baud_ = s.value(QLatin1String(kBaudKey), 0).toInt();
    ptt_ = s.value(QLatin1String(kPttKey), ptt_).toString();
}

void RigControl::save() {
    QSettings s;
    s.setValue(QLatin1String(kEnabledKey), enabled_);
    s.setValue(QLatin1String(kConnectionKey), connection_);
    s.setValue(QLatin1String(kModelKey), model_);
    s.setValue(QLatin1String(kDeviceKey), device_);
    s.setValue(QLatin1String(kHostKey), host_);
    s.setValue(QLatin1String(kBaudKey), baud_);
    s.setValue(QLatin1String(kPttKey), ptt_);
}

void RigControl::setEnabled(bool on) {
    if (on && !have_rig_support()) return;
    if (on == enabled_) return;
    enabled_ = on;
    save();
    if (!on) {
        Session::instance().stop_rig();
    } else {
        connectRig();
    }
    emit changed();
}

void RigControl::setConnection(const QString& kind) {
    if (kind == connection_) return;
    connection_ = kind;
    error_.clear();
    // The device belongs to the kind that was showing when it was
    // chosen, so carrying it across would offer a Bluetooth address to
    // the USB opener.
    device_.clear();
    save();
    refreshDevices();
    emit changed();
}

QStringList RigControl::connections() const {
    return {QStringLiteral("usb"), QStringLiteral("bluetooth"),
            QStringLiteral("network")};
}

void RigControl::setModel(int m) {
    if (m == model_) return;
    model_ = m;
    save();
    emit changed();
}

QString RigControl::modelLabel() const {
    for (const rig::RigModel& m : all_models()) {
        if (m.model == model_) return QString::fromStdString(m.label());
    }
    return QStringLiteral("model %1").arg(model_);
}

void RigControl::setDevice(const QString& id) {
    if (id == device_) return;
    device_ = id;
    save();
    emit changed();
    emit devicesChanged();  // `devicePermitted` follows the selection
}

QString RigControl::deviceLabel() const {
    for (const QVariant& row : devices_) {
        const QVariantMap m = row.toMap();
        if (m[QStringLiteral("id")].toString() == device_) {
            return m[QStringLiteral("label")].toString();
        }
    }
    return device_;
}

bool RigControl::devicePermitted() const {
    if (connection_ == QLatin1String("network")) return true;
    if (device_.isEmpty()) return false;
    for (const QVariant& row : devices_) {
        const QVariantMap m = row.toMap();
        if (m[QStringLiteral("id")].toString() == device_) {
            return m[QStringLiteral("permitted")].toBool();
        }
    }
    return false;
}

void RigControl::setHost(const QString& h) {
    if (h == host_) return;
    host_ = h;
    save();
    emit changed();
}

void RigControl::setBaud(int b) {
    if (b == baud_) return;
    baud_ = b;
    save();
    emit changed();
}

QStringList RigControl::bauds() const {
    // "Default" first, and it is not a synonym for one of the numbers:
    // it means take whatever the chosen Hamlib backend would have set a
    // serial port to, which `rig::serial_defaults` reads out of that
    // backend's own caps. Right for almost every radio, and the reason
    // a rate is a thing the operator can leave alone.
    return {QStringLiteral("Default"), QStringLiteral("1200"),
            QStringLiteral("2400"),    QStringLiteral("4800"),
            QStringLiteral("9600"),    QStringLiteral("19200"),
            QStringLiteral("38400"),   QStringLiteral("57600"),
            QStringLiteral("115200")};
}

void RigControl::setPttMethod(const QString& m) {
    if (m == ptt_) return;
    ptt_ = m;
    save();
    emit changed();
}

QStringList RigControl::pttMethods() const {
    QStringList out{QStringLiteral("vox"), QStringLiteral("cat")};
    // **DTR and RTS are not offered over Bluetooth**, because RFCOMM has
    // no modem control lines at all. Offering them and failing at the
    // moment somebody presses transmit is the worst available order to
    // discover it in.
    if (connection_ != QLatin1String("bluetooth")) {
        out << QStringLiteral("dtr") << QStringLiteral("rts");
    }
    return out;
}

// --- models -----------------------------------------------------------------

QVariantList RigControl::findModels(const QString& query) const {
    const QString needle = query.trimmed();
    QVariantList out;

    if (needle.isEmpty()) {
        // Never a blank screen. The dummy is what makes the whole path
        // exercisable with no radio attached, and NET rigctl is the
        // "share the radio with the station PC" answer -- both are worth
        // finding without knowing to search for them.
        for (const rig::RigModel& m : all_models()) {
            if (m.model == rig::MODEL_DUMMY || m.model == rig::MODEL_NET_RIGCTL) {
                out.append(model_row(m));
            }
        }
        for (const rig::RigModel& m : all_models()) {
            if (static_cast<int>(out.size()) >= kMaxModelRows) break;
            if (m.model == rig::MODEL_DUMMY || m.model == rig::MODEL_NET_RIGCTL) continue;
            out.append(model_row(m));
        }
        return out;
    }

    for (const rig::RigModel& m : all_models()) {
        if (static_cast<int>(out.size()) >= kMaxModelRows) break;
        const QString label = QString::fromStdString(m.label());
        if (label.contains(needle, Qt::CaseInsensitive)) out.append(model_row(m));
    }
    return out;
}

// --- devices ----------------------------------------------------------------

void RigControl::refreshDevices() {
    devices_.clear();
    if (connection_ != QLatin1String("network")) {
        std::vector<rig::android::SerialDevice> found;
        try {
            found = connection_ == QLatin1String("bluetooth")
                        ? rig::android::bluetooth_devices()
                        : rig::android::usb_devices();
        } catch (const std::exception&) {
            // An empty list, not an error banner: "no radio plugged in"
            // and "the Bluetooth permission has not been granted" are
            // both ordinary states of this screen, and the screen says
            // so in its own words.
            found.clear();
        }
        for (const rig::android::SerialDevice& d : found) {
            QVariantMap row;
            row[QStringLiteral("id")] = QString::fromStdString(d.id);
            row[QStringLiteral("label")] = QString::fromStdString(d.label);
            row[QStringLiteral("permitted")] = d.permitted;
            devices_.append(row);
        }
    }
    emit devicesChanged();
    emit changed();
}

void RigControl::requestPermission() {
    if (connection_ == QLatin1String("network")) return;
    const std::string id = connection_ == QLatin1String("bluetooth") && device_.isEmpty()
                               ? std::string("bt:")
                               : device_.toStdString();
    if (id.empty()) return;
    try {
        rig::android::request_permission(id);
    } catch (const std::exception&) {
        // The dialog could not be raised; the screen keeps showing the
        // device as not permitted, which is the truth.
    }
}

void RigControl::publish_permission(const QString& id, bool granted) {
    Q_UNUSED(id);
    Q_UNUSED(granted);
    // Re-enumerate rather than patching the row: a USB grant can arrive
    // for a device that has since been unplugged, and the answer to "is
    // it usable now" is the enumeration, not the broadcast.
    refreshDevices();
}

// --- the session ------------------------------------------------------------

bool RigControl::connectRig() {
    rig::HamlibConfig config;
    config.model = model_;
    config.baud = baud_;
    config.ptt_method = ptt_from(ptt_);

    std::shared_ptr<rig::SerialTransport> transport;
    if (connection_ == QLatin1String("network")) {
        config.device = host_.trimmed().toStdString();
    } else {
        config.device = device_.toStdString();
        if (config.device.empty()) {
            error_ = tr("Choose a device first.");
            emit changed();
            return false;
        }
        // The line settings the transport has to be given, with every
        // "Default" filled from what this Hamlib backend would itself
        // have used -- because over a socket Hamlib never touches the
        // hardware and so cannot apply them. This is also where the
        // control-line conflicts are caught (PTT by RTS with hardware
        // handshaking, and so on): Hamlib's own checks for those sit
        // inside its serial path and never run here.
        rig::SerialDefaults defaults;
#ifdef SSTVAE_ANDROID_HAVE_RIG
        defaults = rig::serial_defaults(model_);
#endif
        rig::SerialParams params;
        try {
            params = rig::resolve_serial_params(config, defaults);
        } catch (const std::exception& e) {
            // A control-line conflict -- PTT by RTS with hardware
            // handshaking, and the two like it. **Reported here, because
            // nothing downstream will**: `make_bridged_backend` does not
            // resolve line settings (the transport is built with them
            // before it is handed over), so an earlier draft that left
            // this to `start_rig` swallowed the message and connected
            // with a silently wrong configuration.
            error_ = QString::fromStdString(e.what());
            emit changed();
            return false;
        }
        transport = rig::android::make_transport(params);
    }

    error_.clear();
    const bool ok = Session::instance().start_rig(config, std::move(transport));
    emit changed();
    return ok;
}

void RigControl::disconnectRig() {
    Session::instance().stop_rig();
    emit changed();
}

// --- live state -------------------------------------------------------------

bool RigControl::running() const { return Session::instance().rig_running(); }
bool RigControl::canKey() const { return Session::instance().rig_can_key(); }

QString RigControl::status() const {
    if (!error_.isEmpty()) return error_;
    return QString::fromStdString(Session::instance().rig_status());
}

bool RigControl::failed() const {
    return !error_.isEmpty() || Session::instance().rig_failed();
}

bool RigControl::bluetoothReady() const {
    return rig::android::has_permission("bt:");
}

QString RigControl::frequency() const {
    const std::optional<double> hz = Session::instance().rig_frequency_hz();
    if (!hz) return {};
    return QStringLiteral("%1 MHz").arg(*hz / 1e6, 0, 'f', 6);
}

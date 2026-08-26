// CAT and PTT, as QML sees it.
//
// A view, owning nothing, like `Listener` and `Transmitter`: the
// `RigController` lives in `Session` because keying has to outlive a
// rotation, and everything here is derived on demand or written
// straight through to `QSettings`.
//
// **What is behind it is the whole reason this file exists.**
// `docs/android.md` recorded rig control as structurally impossible on
// Android, because Hamlib's serial layer opens a device path and an
// unprivileged app is given none. Hamlib takes a socket for any backend
// instead (`core/rig/transport.hpp`), so the radio list here is the
// desktop's radio list, unmodified and unpatched. The three connection
// kinds below are what a phone can actually offer:
//
//   * **USB** -- a serial adapter or a radio's own USB port, through the
//     USB host API and `usb-serial-for-android`. CAT, plus DTR or RTS
//     keying.
//   * **Bluetooth** -- an RFCOMM link to a radio with a serial profile.
//     CAT keying only: RFCOMM has no modem control lines to assert, and
//     the transport says so rather than failing silently.
//   * **Network** -- a host and port handed to Hamlib directly, with no
//     bridge in the path. That covers a station PC running `rigctld`
//     (model 2) and a serial-over-TCP server pointed at with a native
//     backend, which are the same thing in Hamlib and are the same
//     thing here.
//
// The settings live in `QSettings` for the reason `Transmitter`
// records: the desktop's `config.json` schema is a rig section, folder
// paths and a window layout, and its hand-editability buys nothing on a
// phone. The discipline that does carry over is that every setting
// displayed is written back through the same key it was read from, with
// the key spelled once as a constant.

#ifndef SSTVAE_ANDROID_RIGCONTROL_HPP
#define SSTVAE_ANDROID_RIGCONTROL_HPP

#include <QObject>
#include <QString>
#include <QStringList>
#include <QTimer>
#include <QVariantList>
#include <QtQml/qqmlregistration.h>

class RigControl : public QObject {
    Q_OBJECT
    QML_ELEMENT

    // Whether the operator wants rig control at all. Off is the
    // shipping default and not a degraded mode: the app was built to
    // work with no CAT, and a phone acoustically coupled to a handheld
    // has no radio to talk to.
    Q_PROPERTY(bool enabled READ enabled WRITE setEnabled NOTIFY changed)

    // "usb" | "bluetooth" | "network". **This, not the shape of the
    // device string, is what decides whether a bridge is built** --
    // see `rig::make_bridged_backend`, where an earlier draft sniffed
    // the string and got it wrong for every USB device.
    Q_PROPERTY(QString connection READ connection WRITE setConnection NOTIFY changed)
    Q_PROPERTY(QStringList connections READ connections CONSTANT)

    // The Hamlib model number, and its label for display. Two
    // properties rather than an index into a list: the list is
    // hundreds of rows and is filtered by a search box, so an index
    // would mean something different after every keystroke.
    Q_PROPERTY(int model READ model WRITE setModel NOTIFY changed)
    Q_PROPERTY(QString modelLabel READ modelLabel NOTIFY changed)

    // {id, label, permitted} per row, for the current connection kind.
    // Empty for "network".
    Q_PROPERTY(QVariantList devices READ devices NOTIFY devicesChanged)
    Q_PROPERTY(QString device READ device WRITE setDevice NOTIFY changed)
    Q_PROPERTY(QString deviceLabel READ deviceLabel NOTIFY changed)
    // Whether the app may open the selected device right now. False
    // puts a "Grant access" button in front of the connect button
    // rather than letting the connection fail with a permission error.
    Q_PROPERTY(bool devicePermitted READ devicePermitted NOTIFY devicesChanged)

    // host[:port] for the network kind. Hamlib defaults the port to
    // 4532, which is rigctld's, so a bare host is the common case.
    Q_PROPERTY(QString host READ host WRITE setHost NOTIFY changed)

    Q_PROPERTY(int baud READ baud WRITE setBaud NOTIFY changed)
    Q_PROPERTY(QStringList bauds READ bauds CONSTANT)

    // "vox" | "cat" | "dtr" | "rts". `vox` means *do not key at all*,
    // because the operator's audio is doing it -- not "key by VOX".
    Q_PROPERTY(QString pttMethod READ pttMethod WRITE setPttMethod NOTIFY changed)
    Q_PROPERTY(QStringList pttMethods READ pttMethods NOTIFY changed)

    // "mic" | "data": which audio input CAT keying selects on a radio
    // that keys the two differently (a TS-480 and friends). Meaningless
    // on every other rig, which is what `pttAudioSupported` says.
    Q_PROPERTY(QString pttAudio READ pttAudio WRITE setPttAudio NOTIFY changed)
    Q_PROPERTY(bool pttAudioSupported READ pttAudioSupported NOTIFY changed)

    // --- live state --------------------------------------------------
    Q_PROPERTY(bool running READ running NOTIFY changed)
    // One line an operator can act on, which `running` cannot be.
    //
    // **`running` means a session is configured, not that the radio is
    // answering** -- the desktop's distinction, kept for the desktop's
    // reason. On hardware that read as "Connected" with the cable
    // pulled out, which is the most misleading thing this screen could
    // say. This separates them: "Connecting", "Connected", "Device
    // disconnected", "Not responding".
    Q_PROPERTY(QString connectionState READ connectionState NOTIFY changed)
    Q_PROPERTY(bool failed READ failed NOTIFY changed)
    Q_PROPERTY(QString status READ status NOTIFY changed)
    // The dial frequency as text, empty when unknown. Text rather than a
    // number because "unknown" is a state the UI has to render and a
    // sentinel double is how that becomes 0.000 000 MHz on screen.
    Q_PROPERTY(QString frequency READ frequency NOTIFY changed)
    // Whether an over will be keyed by the rig. Read by the transmit
    // screen, which says so rather than leaving the operator to work out
    // whether the VOX leader is still doing the job.
    Q_PROPERTY(bool canKey READ canKey NOTIFY changed)
    // Hamlib's own trace, kept where an operator can read and send it.
    //
    // **This is a diagnostic, and it exists because of a bug this
    // screen could not explain.** A Kenwood worked over USB and two
    // Icoms did not, and the artifact that answers "how far did open
    // get, and which CAT command did the radio refuse" is Hamlib's
    // trace -- which the library writes to stderr, i.e. nowhere on a
    // phone. Off by default: at `RIG_DEBUG_TRACE` it is a line per
    // frame and there is nothing here a working station needs.
    Q_PROPERTY(bool debugLog READ debugLog WRITE setDebugLog NOTIFY changed)
    Q_PROPERTY(QString logText READ logText NOTIFY changed)

    // Whether Bluetooth can be listed at all. Distinct from "no paired
    // radios", which is what an empty list looks like either way -- and
    // telling an operator to go and pair a radio they already paired is
    // the kind of wrong advice that ends in a bug report.
    Q_PROPERTY(bool bluetoothReady READ bluetoothReady NOTIFY devicesChanged)

public:
    explicit RigControl(QObject* parent = nullptr);
    ~RigControl() override;

    bool enabled() const { return enabled_; }
    void setEnabled(bool on);

    QString connection() const { return connection_; }
    void setConnection(const QString& kind);
    QStringList connections() const;

    int model() const { return model_; }
    void setModel(int m);
    QString modelLabel() const;

    QVariantList devices() const { return devices_; }
    QString device() const { return device_; }
    void setDevice(const QString& id);
    QString deviceLabel() const;
    bool devicePermitted() const;

    QString host() const { return host_; }
    void setHost(const QString& h);

    int baud() const { return baud_; }
    void setBaud(int b);
    QStringList bauds() const;

    QString pttMethod() const { return ptt_; }
    void setPttMethod(const QString& m);
    QStringList pttMethods() const;

    QString pttAudio() const { return ptt_audio_; }
    void setPttAudio(const QString& a);
    bool pttAudioSupported() const;

    bool running() const;
    QString connectionState() const;
    bool failed() const;
    QString status() const;
    QString frequency() const;
    bool canKey() const;
    bool bluetoothReady() const;

    bool debugLog() const { return debug_log_; }
    void setDebugLog(bool on);
    QString logText() const;
    Q_INVOKABLE void clearLog();

    // Rows matching `query`, capped -- Hamlib knows several hundred
    // rigs and a phone list view of all of them is not a picker. An
    // empty query returns the most useful few rather than nothing, so
    // the screen is never blank.
    Q_INVOKABLE QVariantList findModels(const QString& query) const;

    // Re-enumerate. Called when the screen appears and after a
    // permission answer; USB devices come and go with the cable.
    Q_INVOKABLE void refreshDevices();

    // Raise the system permission dialog for the selected device.
    // Returns immediately -- nothing here waits on a human.
    Q_INVOKABLE void requestPermission();

    // Open the radio with the current settings, or close it. `connect`
    // returns immediately: reaching the rig can take its full timeout
    // and the answer arrives through `status`.
    Q_INVOKABLE bool connectRig();
    Q_INVOKABLE void disconnectRig();

signals:
    void changed();
    void devicesChanged();

private:
    void save();
    void load();
    void publish_permission(const QString& id, bool granted);

    // Called once per poll tick. Rebuilds the rig session when the radio
    // has stopped answering and the device is reachable again.
    void maybe_reconnect();

    // Install or remove the Hamlib debug sink to match `debug_log_`.
    void apply_debug_log();

    bool enabled_ = false;
    QString connection_ = QStringLiteral("usb");
    int model_ = 1;  // Hamlib's dummy: connects with no radio attached
    QString device_;
    QString host_;
    int baud_ = 0;  // 0 = whatever the chosen backend would have used
    QString ptt_ = QStringLiteral("cat");
    QString ptt_audio_ = QStringLiteral("mic");
    bool debug_log_ = false;

    // False when `init_rig_bridge` could not hand the layer a VM and a
    // Context. Distinct from the compile-time `SSTVAE_ANDROID_HAVE_RIG`:
    // that says the build has Hamlib, this says the platform side came
    // up. Either way the switch will not turn on, and `status` says why.
    bool layer_ok_ = true;

    // Reconnection state, in whole poll ticks (the timer is 1 Hz) so
    // there is no clock to reason about.
    //
    // `reconnect_wait_` is only spent on attempts that actually ran: a
    // device that is simply absent costs nothing and is retried on the
    // very next tick, which is what makes replugging feel immediate
    // rather than "somewhere in the next 30 seconds".
    int reconnect_wait_ = 0;
    int reconnect_step_ = 0;
    // Cached by `maybe_reconnect` so `connectionState` can distinguish
    // "unplugged" from "not answering" without a JNI call of its own on
    // every property read.
    bool device_present_ = true;

    QVariantList devices_;
    // Set by a failed `connectRig` and shown in place of the session's
    // status, which for a configuration that never reached `start_rig`
    // would be the *previous* session's text or nothing at all.
    QString error_;
    QTimer poll_;
};

#endif

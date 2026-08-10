// The transmit side, as QML sees it.
//
// A view, owning nothing -- the counterpart of `Listener`. The picture
// and its framing live in `Composition`, the engine lives in `Session`,
// and both outlive this object, so a rotation in the middle of composing
// or in the middle of an over loses nothing. Every property is derived
// on demand for the same reason it is on the receive side.
//
// Starting an over goes *through the service*, never straight to
// `Session`, exactly as starting capture does. The picture does not fit
// in an Intent, so the sequence is stage-then-ask: this object stages the
// request on the session and asks the service to send it.
//
// The station settings live here rather than in `core/settings/`.
// **QSettings, not the desktop's config.json**, which is a real
// divergence and a deliberate one: that file's schema is a rig section,
// folder paths and a window layout, none of which exist on a phone,
// and its hand-editability -- the property the round-trip discipline
// protects -- buys nothing on a device with no text editor and no shared
// machine. The app already stored `ui/showTechnical` this way. What does
// carry over is the discipline itself: every setting displayed is
// written back through the same key it was read from, with the key
// spelled once as a constant.

#ifndef SSTVAE_ANDROID_TRANSMITTER_HPP
#define SSTVAE_ANDROID_TRANSMITTER_HPP

#include <QObject>
#include <QString>
#include <QStringList>
#include <QTimer>
#include <QtQml/qqmlregistration.h>

class Transmitter : public QObject {
    Q_OBJECT
    QML_ELEMENT

    // --- the picture -----------------------------------------------
    Q_PROPERTY(bool hasPicture READ hasPicture NOTIFY changed)
    // Bumped whenever the framing moves, because QML caches images by
    // URL -- the same reason the live receive preview carries an id.
    Q_PROPERTY(int previewId READ previewId NOTIFY changed)
    Q_PROPERTY(double zoom READ zoom WRITE setZoom NOTIFY changed)
    // The bottom of the zoom travel for the picture currently loaded:
    // the point at which all of it is visible, letterboxed. 1.0 for a
    // 4:3 source, where there is nothing to letterbox. Exposed so the
    // slider and the pinch stop exactly where `images::fit` clamps,
    // rather than at a constant that would be wrong for every aspect
    // but one.
    Q_PROPERTY(double minZoom READ minZoom NOTIFY changed)
    Q_PROPERTY(double centerX READ centerX NOTIFY changed)
    Q_PROPERTY(double centerY READ centerY NOTIFY changed)

    // --- station settings -------------------------------------------
    Q_PROPERTY(QString callsign READ callsign WRITE setCallsign NOTIFY changed)
    Q_PROPERTY(QString mode READ mode WRITE setMode NOTIFY changed)
    Q_PROPERTY(QStringList modes READ modes CONSTANT)
    Q_PROPERTY(double level READ level WRITE setLevel NOTIFY changed)
    Q_PROPERTY(bool cwId READ cwId WRITE setCwId NOTIFY changed)
    Q_PROPERTY(QString cwMessage READ cwMessage WRITE setCwMessage NOTIFY changed)
    // Seconds of leader, 0 for none. A duration rather than a switch
    // because VOX circuits differ and the operator is the one who can
    // hear whether the start of the transmission survived.
    Q_PROPERTY(double voxLead READ voxLead WRITE setVoxLead NOTIFY changed)
    Q_PROPERTY(QStringList outputDevices READ outputDevices NOTIFY devicesChanged)
    Q_PROPERTY(QString outputDevice READ outputDevice WRITE setOutputDevice NOTIFY changed)

    // --- the over ----------------------------------------------------
    Q_PROPERTY(bool encoderReady READ encoderReady NOTIFY changed)
    Q_PROPERTY(QString encoderStatus READ encoderStatus NOTIFY changed)
    Q_PROPERTY(bool transmitting READ transmitting NOTIFY changed)
    Q_PROPERTY(bool canSend READ canSend NOTIFY changed)
    Q_PROPERTY(QString txStatus READ txStatus NOTIFY changed)
    Q_PROPERTY(double txProgress READ txProgress NOTIFY changed)
    // How long this mode's transmission will take, as text, so the
    // operator can decide whether to start one now.
    Q_PROPERTY(QString airtime READ airtime NOTIFY changed)
    Q_PROPERTY(QString lastError READ lastError NOTIFY changed)

public:
    explicit Transmitter(QObject* parent = nullptr);
    ~Transmitter() override;

    bool hasPicture() const;
    int previewId() const { return preview_id_; }
    double zoom() const;
    double minZoom() const;
    double centerX() const;
    double centerY() const;
    void setZoom(double z);

    QString callsign() const { return callsign_; }
    void setCallsign(const QString& c);
    QString mode() const { return mode_; }
    void setMode(const QString& m);
    QStringList modes() const;
    double level() const { return level_; }
    void setLevel(double v);
    bool cwId() const { return cw_id_; }
    void setCwId(bool on);
    QString cwMessage() const { return cw_message_; }
    void setCwMessage(const QString& m);
    double voxLead() const { return vox_lead_s_; }
    void setVoxLead(double s);
    QStringList outputDevices() const { return devices_; }
    QString outputDevice() const { return device_; }
    void setOutputDevice(const QString& d);

    bool encoderReady() const;
    QString encoderStatus() const;
    bool transmitting() const;
    bool canSend() const;
    QString txStatus() const;
    double txProgress() const;
    QString airtime() const;
    QString lastError() const;

    // Drag motion, as a fraction of the *preview's* own width and
    // height -- so QML passes `-dx/width, -dy/height` and never has to
    // know the crop window's size in source coordinates, which depends
    // on the aspect ratio and the zoom and is clamped in `Composition`
    // where both are known.
    Q_INVOKABLE void panBy(double dx, double dy);
    Q_INVOKABLE void pickImage();
    Q_INVOKABLE void takePhoto();
    Q_INVOKABLE void clearPicture();
    Q_INVOKABLE void refreshDevices();
    Q_INVOKABLE void loadEncoder();
    Q_INVOKABLE void send();
    Q_INVOKABLE void cancel();

    // Called from the picker's JNI callback, on the Android UI thread.
    void onPicked(const QString& path, const QString& error);

signals:
    void devicesChanged();
    void changed();

private:
    void bump();

    QStringList devices_;
    QString error_;
    QTimer poll_;
    int preview_id_ = 0;

    QString callsign_;
    QString mode_;
    double level_ = 0.9;
    bool cw_id_ = false;
    QString cw_message_;
    double vox_lead_s_ = 0.0;
    QString device_;
};

#endif

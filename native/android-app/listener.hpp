// The receive session, as QML sees it.
//
// **A view, owning nothing.** The session lives in `Session` and its
// lifetime is guaranteed by `ListenerService`; this class holds no ring
// buffer, no stream and no engine thread, and caches nothing they say.
// Every property is derived on demand, so an instance created after a
// reception started is in exactly the same position as one that was
// there all along -- which is what "the UI is a detachable view" has to
// mean in practice for it to survive a rotation or the screen going
// off.
//
// Starting and stopping go *through the service*, never straight to
// `Session`. That is the whole ownership inversion: if the UI could
// start a session directly, the session would belong to something
// Android may destroy mid-reception.

#ifndef SSTVAE_ANDROID_LISTENER_HPP
#define SSTVAE_ANDROID_LISTENER_HPP

#include <QObject>
#include <QString>
#include <QStringList>
#include <QTimer>
#include <QtQml/qqmlregistration.h>

class Listener : public QObject {
    Q_OBJECT
    QML_ELEMENT

    Q_PROPERTY(QStringList inputDevices READ inputDevices NOTIFY devicesChanged)
    Q_PROPERTY(bool listening READ listening NOTIFY changed)
    Q_PROPERTY(QString status READ status NOTIFY changed)
    Q_PROPERTY(QString audioRoute READ audioRoute NOTIFY changed)
    Q_PROPERTY(QString level READ level NOTIFY changed)
    // Numbers, not a parsed string. The QML meter binds to these; the
    // text line stays for the detail that will not fit on a bar.
    Q_PROPERTY(double peakLevel READ peakLevel NOTIFY changed)
    Q_PROPERTY(double driftPpm READ driftPpm NOTIFY changed)
    Q_PROPERTY(bool droppingAudio READ droppingAudio NOTIFY changed)
    Q_PROPERTY(QString lastError READ lastError NOTIFY changed)
    Q_PROPERTY(QString modelStatus READ modelStatus NOTIFY changed)
    Q_PROPERTY(bool modelReady READ modelReady NOTIFY changed)
    // Bumped whenever the engine publishes a new frame. QML caches
    // images by URL, so the *id* has to change or the preview freezes
    // on the first frame it ever showed.
    Q_PROPERTY(int liveImageId READ liveImageId NOTIFY changed)
    Q_PROPERTY(bool hasLiveImage READ hasLiveImage NOTIFY changed)
    // **Off by default, and the default is the whole point.** Poll
    // counts, ppm, dBFS and ring depth are how this app was debugged
    // and they are the reason several bugs were findable at all -- but
    // they are also the first thing an operator sees, and a screen that
    // opens with "peak -23 dBFS 4.1% near-zero / capture +180 ppm"
    // reads as equipment rather than as a radio. Kept in full, one
    // switch away, because the failures they catch (dropped samples
    // above all) are invisible without them and do not announce
    // themselves.
    Q_PROPERTY(bool showTechnical READ showTechnical WRITE setShowTechnical
                   NOTIFY changed)
    // Mirror finished receptions into `Pictures/SSTVAE`, which is what
    // puts them in Google Photos' "On this device" collections. Off by
    // default -- see `Session::set_save_to_gallery` for why that is not
    // just conservatism.
    Q_PROPERTY(bool saveToGallery READ saveToGallery WRITE setSaveToGallery
                   NOTIFY changed)
    Q_PROPERTY(QString galleryError READ galleryError NOTIFY changed)

public:
    explicit Listener(QObject* parent = nullptr);
    ~Listener() override;

    QStringList inputDevices() const;
    bool listening() const;
    QString status() const;
    QString audioRoute() const;
    QString level() const;
    double peakLevel() const;
    double driftPpm() const;
    bool droppingAudio() const;
    QString lastError() const;
    QString modelStatus() const;
    bool modelReady() const;
    int liveImageId() const { return live_id_; }
    bool hasLiveImage() const;
    bool showTechnical() const { return technical_; }
    void setShowTechnical(bool on);
    bool saveToGallery() const { return gallery_; }
    void setSaveToGallery(bool on);
    QString galleryError() const;

    Q_INVOKABLE void refreshDevices();
    Q_INVOKABLE void loadModel();
    // Hand a saved reception to the system share sheet. On the UI
    // rather than the gallery model because it is an action, not data.
    Q_INVOKABLE void sharePicture(const QString& path, const QString& caption);
    Q_INVOKABLE void start(const QString& deviceName);
    Q_INVOKABLE void stop();

signals:
    void devicesChanged();
    void changed();

private:
    QString plain_status() const;

    QStringList devices_;
    QString error_;
    QTimer poll_;
    int live_id_ = 0;
    bool screen_held_ = false;
    bool technical_ = false;
    bool gallery_ = false;
    const void* last_image_ = nullptr;
};

#endif

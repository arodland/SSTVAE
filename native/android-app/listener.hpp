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
    Q_PROPERTY(QString lastError READ lastError NOTIFY changed)

public:
    explicit Listener(QObject* parent = nullptr);
    ~Listener() override;

    QStringList inputDevices() const;
    bool listening() const;
    QString status() const;
    QString audioRoute() const;
    QString level() const;
    QString lastError() const;

    Q_INVOKABLE void refreshDevices();
    Q_INVOKABLE void start(const QString& deviceName);
    Q_INVOKABLE void stop();

signals:
    void devicesChanged();
    void changed();

private:
    QStringList devices_;
    QString error_;
    QTimer poll_;
};

#endif

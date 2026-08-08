// The receive session, as QML sees it.
//
// **Deliberately thin, and deliberately owning nothing that matters.**
// Tier 0's shape (docs/android.md) is that the foreground service owns
// the engine and the UI is a detachable view, which inverts the
// desktop's `AppState`. This class is a placeholder for the service in
// that arrangement, so it is written to the rule already: every property
// it exposes is *derived* from `rx::SharedState` or the `InputStream` on
// demand, and none of it is accumulated by watching. When the service
// arrives, what moves is the ownership of `ring_`/`engine_`, not the way
// the UI reads them.

#ifndef SSTVAE_ANDROID_LISTENER_HPP
#define SSTVAE_ANDROID_LISTENER_HPP

#include <QObject>
#include <QString>
#include <QStringList>
#include <QTimer>
#include <QtQml/qqmlregistration.h>

#include <memory>
#include <optional>
#include <thread>

#include "audio/android/androidaudio.hpp"
#include "rx/engine.hpp"
#include "rx/ringbuffer.hpp"

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
    bool listening() const { return stream_ != nullptr; }
    QString status() const;
    QString audioRoute() const;
    QString level() const;
    QString lastError() const { return error_; }

    Q_INVOKABLE void refreshDevices();
    Q_INVOKABLE void start(const QString& deviceName);
    Q_INVOKABLE void stop();

signals:
    void devicesChanged();
    void changed();

private:
    QStringList devices_;
    QString error_;

    // Everything below moves to the service. Nothing in the UI holds a
    // copy of any of it.
    std::unique_ptr<sstvae::rx::RingBuffer> ring_;
    std::unique_ptr<sstvae::audio::android::InputStream> stream_;
    std::unique_ptr<sstvae::rx::SharedState> state_;
    std::unique_ptr<sstvae::rx::StopFlag> stop_;
    std::thread thread_;
    QTimer poll_;
};

#endif

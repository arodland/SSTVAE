// The received-pictures gallery, and the live preview beside it.
//
// **Both read from disk, not from shared state**, and that is the
// design rather than an implementation detail. `rx/engine` wipes mode,
// callsign, SNR and frame count two seconds after a reception; on a
// phone the operator is usually not looking then, so anything that
// lived only in memory would be gone by the time they picked the
// handset up. Each reception is a PNG plus a JSON sidecar, and the
// sidecar is what this model reads -- which means a picture opened a
// week later still answers "who sent this, and how well did it come
// through".
//
// The live preview is the one thing that *does* come from shared state,
// because it is by definition about right now. It goes through a
// `QQuickImageProvider` with a monotonic id in the URL: QML caches by
// URL, so a stable one would show the first frame forever.

#ifndef SSTVAE_ANDROID_PICTURES_HPP
#define SSTVAE_ANDROID_PICTURES_HPP

#include <QAbstractListModel>
#include <QImage>
#include <QQuickImageProvider>
#include <QString>
#include <QVector>
#include <QtQml/qqmlregistration.h>

// Serves both `image://sstvae/live/<n>` (the reception in progress) and
// `image://sstvae/file/<path>` (a saved one).
class PictureProvider : public QQuickImageProvider {
public:
    PictureProvider();
    QImage requestImage(const QString& id, QSize* size, const QSize& requested) override;
};

struct PictureEntry {
    QString path;
    QString received;
    QString callsign;
    QString mode;
    double snr_db = 0.0;
    int frames_received = 0;
    int frames_expected = 0;
};

class PictureList : public QAbstractListModel {
    Q_OBJECT
    QML_ELEMENT

    Q_PROPERTY(int count READ rowCount NOTIFY changed)

public:
    enum Roles {
        PathRole = Qt::UserRole + 1,
        ReceivedRole,
        CallsignRole,
        ModeRole,
        SnrRole,
        FramesRole,
        SummaryRole,
    };

    explicit PictureList(QObject* parent = nullptr);

    int rowCount(const QModelIndex& parent = {}) const override;
    QVariant data(const QModelIndex& index, int role) const override;
    QHash<int, QByteArray> roleNames() const override;

    // Cheap enough to call whenever the screen is shown: a directory
    // listing and a few hundred bytes of JSON each. Deliberately not a
    // filesystem watcher -- a reception finishing while the gallery is
    // open is rare, and a watcher is a background wakeup on a phone.
    Q_INVOKABLE void refresh();

signals:
    void changed();

private:
    QVector<PictureEntry> entries_;
};

#endif

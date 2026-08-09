// The waterfall, which on this platform is the tuning instrument.
//
// **Not a diagnostic** (docs/android.md). With no CAT there is no
// frequency readout, no rig chip, nothing at all to say where the radio
// is pointed -- so this display is the only tuning feedback the
// operator has, and it is more important here than on the desktop
// where it competes with a rig panel. That is why it gets real
// vertical extent, and why the band markers are drawn on it rather
// than left implicit.
//
// The arithmetic is `core/dsp/spectrum.cpp`, shared with the desktop
// and Qt-free. In particular `reduce_to_width` is **peak-hold when
// shrinking, not point-sampling**: the carriers are one or two bins
// wide and about six apart, so taking every k'th bin drops some
// outright and leaves a ragged comb -- which reads as a *reception*
// problem and sends the next person to debug the modem. On a display
// whose whole job is "are you tuned right", that is the worst available
// lie.
//
// A `QQuickPaintedItem` over a scrolling `QImage`, so history is a
// buffer rather than a retained list of frames. Two things the
// desktop's version got wrong first and this inherits the fixes for:
// scrolling must move history **down**, and a resize must keep the
// history rather than blank it.

#ifndef SSTVAE_ANDROID_WATERFALL_HPP
#define SSTVAE_ANDROID_WATERFALL_HPP

#include <QImage>
#include <QQuickPaintedItem>
#include <QTimer>
#include <QtQml/qqmlregistration.h>

class Waterfall : public QQuickPaintedItem {
    Q_OBJECT
    QML_ELEMENT

public:
    explicit Waterfall(QQuickItem* parent = nullptr);

    void paint(QPainter* painter) override;

public slots:
    // A slot so a test can render one frame instead of waiting on the
    // timer -- the desktop's rule, and it is what makes "the tone
    // paints at the x its frequency says" checkable without a
    // stopwatch.
    void tick();

protected:
    void geometryChange(const QRectF& newGeometry, const QRectF& oldGeometry) override;

private:
    void ensure_history(int w, int h);

    QImage history_;
    QTimer timer_;
};

#endif

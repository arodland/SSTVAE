// The schedule window's "At Time" pickers (Jeff, 2026-10-12): two clock
// faces, one UTC and one local, showing the first send's quarter hour.
// The mouse wheel over either moves it 15 minutes a notch, an hour with
// Shift held; the arrow keys do the same on a focused face. The faces
// only show and ask: the window owns the time and clamps it.
//
// ShiftKeyCap is the "Shift" key drawn in the hint under them; it turns
// red while Shift is held anywhere in the application, so the operator
// can see which step the wheel will take.

#ifndef SSTVAE_GUI_QRSS_CLOCK_HPP
#define SSTVAE_GUI_QRSS_CLOCK_HPP

#include <QString>
#include <QWidget>

namespace sstvae::gui {

class QrssClockFace : public QWidget {
    Q_OBJECT

public:
    static constexpr int FACE = 132;   // the dial's diameter, px
    static constexpr int STEP_S = 900;
    static constexpr int SHIFT_STEP_S = 3600;

    // `utc` false shows the computer's local time.
    explicit QrssClockFace(bool utc, QWidget* parent = nullptr);

    void set_time(double t);
    double time() const { return t_; }
    bool utc() const { return utc_; }
    // "16:30Z" or "09:30 PDT", and "Mon 12 Oct", as drawn under the dial.
    QString readout() const;
    QString date_text() const;

    QSize sizeHint() const override;
    QSize minimumSizeHint() const override { return sizeHint(); }

signals:
    // Move the time by `seconds` (positive is later).
    void stepRequested(int seconds);

protected:
    void paintEvent(QPaintEvent* event) override;
    void wheelEvent(QWheelEvent* event) override;
    void keyPressEvent(QKeyEvent* event) override;

private:
    bool utc_;
    double t_ = 0.0;
    int wheel_accum_ = 0;   // a high-resolution wheel's eighths, to whole notches
};

class ShiftKeyCap : public QWidget {
    Q_OBJECT

public:
    explicit ShiftKeyCap(QWidget* parent = nullptr);
    ~ShiftKeyCap() override;
    bool down() const { return down_; }
    void set_down(bool on);
    QSize sizeHint() const override;
    QSize minimumSizeHint() const override { return sizeHint(); }

protected:
    void paintEvent(QPaintEvent* event) override;
    bool eventFilter(QObject* watched, QEvent* event) override;

private:
    bool down_ = false;
};

}  // namespace sstvae::gui

#endif

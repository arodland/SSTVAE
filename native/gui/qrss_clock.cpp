#include "qrss_clock.hpp"

#include <QApplication>
#include <QDateTime>
#include <QEvent>
#include <QFont>
#include <QKeyEvent>
#include <QPainter>
#include <QPainterPath>
#include <QPalette>
#include <QPen>
#include <QTimeZone>
#include <QWheelEvent>

#include <algorithm>
#include <cmath>

#include "style.hpp"

namespace sstvae::gui {

namespace {

QDateTime at(double t, bool utc) {
    const QDateTime d = QDateTime::fromSecsSinceEpoch(static_cast<qint64>(std::llround(t)),
                                                      QTimeZone::utc());
    return utc ? d : d.toLocalTime();
}

}  // namespace

// --- the clock face -------------------------------------------------------------

QrssClockFace::QrssClockFace(bool utc, QWidget* parent) : QWidget(parent), utc_(utc) {
    setObjectName(utc ? QStringLiteral("at_time_utc") : QStringLiteral("at_time_local"));
    setFocusPolicy(Qt::StrongFocus);
    setCursor(Qt::SizeVerCursor);
    setToolTip(tr("Turn the mouse wheel here to move the first send 15 minutes; hold Shift "
                  "to move it an hour. The arrow keys do the same."));
}

void QrssClockFace::set_utc(bool utc) {
    if (utc == utc_) return;
    utc_ = utc;
    update();
}

int QrssClockFace::step_for(Qt::KeyboardModifiers mods) const {
    if (minute_steps_ && (mods & Qt::ControlModifier)) return CTRL_STEP_S;
    return (mods & Qt::ShiftModifier) ? SHIFT_STEP_S : STEP_S;
}

void QrssClockFace::set_time(double t) {
    if (t == t_) return;
    t_ = t;
    update();
}

QString QrssClockFace::readout() const {
    const QDateTime d = at(t_, utc_);
    return utc_ ? d.toString(QStringLiteral("HH:mm'Z'"))
                : d.toString(QStringLiteral("HH:mm ")) + d.timeZoneAbbreviation();
}

QString QrssClockFace::date_text() const {
    return at(t_, utc_).toString(QStringLiteral("ddd d MMM"));
}

QSize QrssClockFace::sizeHint() const { return {FACE + 4, FACE + 4}; }

void QrssClockFace::paintEvent(QPaintEvent*) {
    QPainter p(this);
    p.setRenderHint(QPainter::Antialiasing);
    const QPalette& pal = palette();
    const QColor ink = pal.color(QPalette::Text);
    const QColor muted = style::secondary_text(pal);
    const double r = std::min(width(), height()) / 2.0 - 2.0;
    p.translate(width() / 2.0, height() / 2.0);

    p.setPen(QPen(hasFocus() ? pal.color(QPalette::Highlight) : pal.color(QPalette::Mid),
                  hasFocus() ? 2.5 : 1.5));
    p.setBrush(pal.color(QPalette::Base));
    p.drawEllipse(QPointF(0, 0), r, r);

    // Minute ticks; the quarter hours, the steps the wheel takes, are
    // the longest and darkest.
    for (int i = 0; i < 60; ++i) {
        const double a = i / 60.0 * 2.0 * M_PI;
        const bool quarter = i % 15 == 0;
        const bool five = i % 5 == 0;
        const double r1 = r * (quarter ? 0.80 : five ? 0.86 : 0.91);
        p.setPen(QPen(quarter ? ink : muted, quarter ? 2.5 : five ? 1.4 : 0.7));
        p.drawLine(QPointF(std::sin(a) * r1, -std::cos(a) * r1),
                   QPointF(std::sin(a) * r * 0.96, -std::cos(a) * r * 0.96));
    }
    QFont small = font();
    small.setPointSizeF(std::max(6.0, small.pointSizeF() * 0.8));
    p.setFont(small);
    p.setPen(muted);
    for (int i = 1; i <= 12; ++i) {
        const double a = i / 12.0 * 2.0 * M_PI;
        const QPointF c(std::sin(a) * r * 0.66, -std::cos(a) * r * 0.66);
        p.drawText(QRectF(c.x() - 10, c.y() - 8, 20, 16), Qt::AlignCenter, QString::number(i));
    }
    const QDateTime d = at(t_, utc_);
    const int h = d.time().hour();
    const int m = d.time().minute();
    p.drawText(QRectF(-20, r * 0.30, 40, 14), Qt::AlignCenter,
               h < 12 ? QStringLiteral("AM") : QStringLiteral("PM"));

    const double ah = ((h % 12) + m / 60.0) / 12.0 * 2.0 * M_PI;
    const double am = m / 60.0 * 2.0 * M_PI;
    p.setPen(QPen(ink, 4.5, Qt::SolidLine, Qt::RoundCap));
    p.drawLine(QPointF(0, 0), QPointF(std::sin(ah) * r * 0.48, -std::cos(ah) * r * 0.48));
    p.setPen(QPen(style::color::danger(), 2.5, Qt::SolidLine, Qt::RoundCap));
    p.drawLine(QPointF(0, 0), QPointF(std::sin(am) * r * 0.82, -std::cos(am) * r * 0.82));
    p.setPen(Qt::NoPen);
    p.setBrush(ink);
    p.drawEllipse(QPointF(0, 0), 3.5, 3.5);
}

void QrssClockFace::wheelEvent(QWheelEvent* event) {
    // Shift turns some wheels sideways (Qt reports x then), so either.
    const QPoint d = event->angleDelta();
    wheel_accum_ += d.y() != 0 ? d.y() : d.x();
    const int notches = wheel_accum_ / 120;
    if (notches == 0) {
        event->accept();
        return;
    }
    wheel_accum_ -= notches * 120;
    emit stepRequested(notches * step_for(event->modifiers()));
    event->accept();
}

void QrssClockFace::keyPressEvent(QKeyEvent* event) {
    const int step = step_for(event->modifiers());
    switch (event->key()) {
        case Qt::Key_Up:
        case Qt::Key_Right:
            emit stepRequested(step);
            return;
        case Qt::Key_Down:
        case Qt::Key_Left:
            emit stepRequested(-step);
            return;
        default:
            QWidget::keyPressEvent(event);
    }
}

// --- the Shift key cap ----------------------------------------------------------

ShiftKeyCap::ShiftKeyCap(QWidget* parent, Qt::Key key) : QWidget(parent), key_(key) {
    if (key == Qt::Key_Shift) setObjectName(QStringLiteral("at_time_shift"));
    setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);
    qApp->installEventFilter(this);
}

ShiftKeyCap::~ShiftKeyCap() {
    if (qApp) qApp->removeEventFilter(this);
}

void ShiftKeyCap::set_down(bool on) {
    if (on == down_) return;
    down_ = on;
    update();
}

QString ShiftKeyCap::label() const {
    if (key_ != Qt::Key_Control) return QStringLiteral("Shift");
#ifdef Q_OS_MACOS
    return QStringLiteral("Cmd");
#else
    return QStringLiteral("Ctrl");
#endif
}

QSize ShiftKeyCap::sizeHint() const {
    QFont f = font();
    f.setPointSizeF(std::max(6.0, f.pointSizeF() * 0.85));
    const QFontMetrics fm(f);
    return {fm.horizontalAdvance(label()) + 10, fm.height() + 4};
}

void ShiftKeyCap::paintEvent(QPaintEvent*) {
    QPainter p(this);
    p.setRenderHint(QPainter::Antialiasing);
    const QPalette& pal = palette();
    const QRectF box = QRectF(rect()).adjusted(0.5, 0.5, -0.5, -1.5);
    p.setPen(QPen(down_ ? style::color::danger() : pal.color(QPalette::Mid), 1));
    p.setBrush(down_ ? style::color::danger() : pal.color(QPalette::Base));
    p.drawRoundedRect(box, 3, 3);
    // The key's lower lip.
    if (!down_) {
        p.setPen(QPen(pal.color(QPalette::Mid), 1));
        p.drawLine(box.bottomLeft() + QPointF(2, 1), box.bottomRight() + QPointF(-2, 1));
    }
    QFont f = font();
    f.setPointSizeF(std::max(6.0, f.pointSizeF() * 0.85));
    p.setFont(f);
    p.setPen(down_ ? style::color::on_danger() : pal.color(QPalette::Text));
    p.drawText(box, Qt::AlignCenter, label());
}

bool ShiftKeyCap::eventFilter(QObject* watched, QEvent* event) {
    switch (event->type()) {
        case QEvent::KeyPress:
        case QEvent::KeyRelease: {
            auto* k = static_cast<QKeyEvent*>(event);
            if (k->key() == key_) set_down(event->type() == QEvent::KeyPress);
            break;
        }
        case QEvent::Wheel:
            set_down(static_cast<QWheelEvent*>(event)->modifiers() &
                     (key_ == Qt::Key_Control ? Qt::ControlModifier : Qt::ShiftModifier));
            break;
        case QEvent::ApplicationDeactivate:
            set_down(false);
            break;
        default:
            break;
    }
    return QWidget::eventFilter(watched, event);
}

}  // namespace sstvae::gui

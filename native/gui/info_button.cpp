#include "info_button.hpp"

#include <QApplication>
#include <QEnterEvent>
#include <QFrame>
#include <QGuiApplication>
#include <QKeyEvent>
#include <QLabel>
#include <QMouseEvent>
#include <QPainter>
#include <QPen>
#include <QScreen>
#include <QTimer>
#include <QVBoxLayout>

#include <algorithm>
#include <utility>

namespace sstvae::gui {

// The card: a frameless tooltip-style window holding one rich-text label.
class InfoCard : public QFrame {
public:
    explicit InfoCard(QWidget* owner)
        : QFrame(owner, Qt::ToolTip | Qt::FramelessWindowHint) {
        setObjectName(QStringLiteral("info_card"));
        setAttribute(Qt::WA_ShowWithoutActivating);
        setFrameShape(QFrame::Box);
        setLineWidth(1);
        // The window's own colours, not the tooltip's yellow.
        QPalette pal = owner->palette();
        pal.setColor(QPalette::Window, pal.color(QPalette::Base));
        pal.setColor(QPalette::WindowText, pal.color(QPalette::Text));
        setPalette(pal);
        setAutoFillBackground(true);
        auto* layout = new QVBoxLayout(this);
        layout->setContentsMargins(14, 12, 14, 12);
        label_ = new QLabel(this);
        label_->setObjectName(QStringLiteral("info_card_text"));
        label_->setTextFormat(Qt::RichText);
        label_->setWordWrap(true);
        label_->setFixedWidth(InfoButton::CARD_W - 28);
        layout->addWidget(label_);
        setMouseTracking(true);
    }
    void set_html(const QString& html) {
        label_->setText(html);
        adjustSize();
    }

private:
    QLabel* label_;
};

InfoButton::InfoButton(std::function<QString()> html, QWidget* parent)
    : QToolButton(parent), html_(std::move(html)) {
    setObjectName(QStringLiteral("info_button"));
    setCursor(Qt::WhatsThisCursor);
    setAutoRaise(true);
    setFocusPolicy(Qt::StrongFocus);
    show_timer_ = new QTimer(this);
    show_timer_->setSingleShot(true);
    show_timer_->setInterval(SHOW_DELAY_MS);
    connect(show_timer_, &QTimer::timeout, this, &InfoButton::show_card);
    hide_timer_ = new QTimer(this);
    hide_timer_->setSingleShot(true);
    hide_timer_->setInterval(HIDE_DELAY_MS);
    connect(hide_timer_, &QTimer::timeout, this, [this] {
        if (pinned_) return;
        // Not while the pointer is on the card: it can be read there.
        if (card_ != nullptr && card_->isVisible() &&
            card_->rect().contains(card_->mapFromGlobal(QCursor::pos()))) {
            return;
        }
        hide_card();
    });
    connect(this, &QToolButton::clicked, this, [this] {
        if (pinned_ && card_shown()) {
            hide_card();
            return;
        }
        show_card();
        pinned_ = true;
    });
}

InfoButton::~InfoButton() {
    if (qApp) qApp->removeEventFilter(this);
    delete card_;
}

bool InfoButton::card_shown() const { return card_ != nullptr && card_->isVisible(); }

QWidget* InfoButton::card() const { return card_; }

void InfoButton::show_card() {
    show_timer_->stop();
    hide_timer_->stop();
    if (card_ == nullptr) {
        card_ = new InfoCard(this);
        card_->installEventFilter(this);
    }
    card_->set_html(html());
    place_card();
    card_->show();
    // A press elsewhere, or Esc, closes a pinned card.
    qApp->installEventFilter(this);
    update();
}

void InfoButton::hide_card() {
    show_timer_->stop();
    hide_timer_->stop();
    pinned_ = false;
    if (card_ != nullptr) card_->hide();
    if (qApp) qApp->removeEventFilter(this);
    update();
}

void InfoButton::refresh() {
    if (card_shown()) {
        card_->set_html(html());
        place_card();
    }
}

void InfoButton::place_card() {
    // Under the (i), its right edge near the (i)'s; on screen whatever
    // the window's position.
    const QSize sz = card_->sizeHint();
    QPoint at = mapToGlobal(QPoint(width() + 6 - sz.width(), height() + 6));
    if (const QScreen* s = screen()) {
        const QRect avail = s->availableGeometry();
        if (at.y() + sz.height() > avail.bottom()) at.setY(mapToGlobal(QPoint(0, 0)).y() - 6 - sz.height());
        at.setX(std::clamp(at.x(), avail.left(), std::max(avail.left(), avail.right() - sz.width())));
        at.setY(std::max(at.y(), avail.top()));
    }
    card_->move(at);
    card_->resize(sz);
}

QSize InfoButton::sizeHint() const { return {22, 22}; }

void InfoButton::paintEvent(QPaintEvent*) {
    QPainter p(this);
    p.setRenderHint(QPainter::Antialiasing);
    const QPalette& pal = palette();
    const bool hot = underMouse() || card_shown() || hasFocus();
    const QRectF r = QRectF(rect()).adjusted(1.5, 1.5, -1.5, -1.5);
    const double d = std::min(r.width(), r.height());
    const QRectF circle(r.center().x() - d / 2, r.center().y() - d / 2, d, d);
    p.setPen(QPen(hot ? pal.color(QPalette::Highlight) : pal.color(QPalette::Mid), 1));
    p.setBrush(hot ? pal.color(QPalette::Highlight) : pal.color(QPalette::Base));
    p.drawEllipse(circle);
    QFont f(QStringLiteral("serif"));
    f.setStyleHint(QFont::Serif);
    f.setItalic(true);
    f.setBold(true);
    f.setPixelSize(std::max(9, static_cast<int>(d * 0.62)));
    p.setFont(f);
    p.setPen(hot ? pal.color(QPalette::HighlightedText) : pal.color(QPalette::Text));
    p.drawText(circle, Qt::AlignCenter, QStringLiteral("i"));
}

#if QT_VERSION >= QT_VERSION_CHECK(6, 0, 0)
void InfoButton::enterEvent(QEnterEvent* event) {
#else
void InfoButton::enterEvent(QEvent* event) {
#endif
    hide_timer_->stop();
    if (!card_shown()) show_timer_->start();
    update();
    QToolButton::enterEvent(event);
}

void InfoButton::leaveEvent(QEvent* event) {
    show_timer_->stop();
    if (card_shown() && !pinned_) hide_timer_->start();
    update();
    QToolButton::leaveEvent(event);
}

bool InfoButton::eventFilter(QObject* watched, QEvent* event) {
    if (watched == card_ && event->type() == QEvent::Leave && !pinned_) {
        hide_timer_->start();
        return false;
    }
    if (watched == card_ && event->type() == QEvent::Enter) {
        hide_timer_->stop();
        return false;
    }
    if (!card_shown()) return false;
    if (event->type() == QEvent::KeyPress &&
        static_cast<QKeyEvent*>(event)->key() == Qt::Key_Escape) {
        hide_card();
        return true;
    }
    if (event->type() == QEvent::MouseButtonPress) {
        const QPoint g = static_cast<QMouseEvent*>(event)->globalPosition().toPoint();
        const bool on_card = card_->frameGeometry().contains(g);
        const bool on_me = rect().contains(mapFromGlobal(g));
        if (!on_card && !on_me) hide_card();
    }
    if (event->type() == QEvent::ApplicationDeactivate) hide_card();
    return false;
}

}  // namespace sstvae::gui

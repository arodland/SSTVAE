#include "info_button.hpp"

#include <QApplication>
#include <QColor>
#include <QEnterEvent>
#include <QFrame>
#include <QGuiApplication>
#include <QKeyEvent>
#include <QLabel>
#include <QMouseEvent>
#include <QPainter>
#include <QPalette>
#include <QPen>
#include <QScreen>
#include <QTimer>
#include <QVBoxLayout>

#include <algorithm>
#include <utility>

#include "style.hpp"

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

void InfoButton::set_glyph(const QString& glyph) {
    glyph_ = glyph;
    update();
}

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
    p.drawText(circle, Qt::AlignCenter, glyph_);
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

// --- card text ----------------------------------------------------------------------------

namespace info {

namespace {
bool dark_theme(const QPalette& pal) { return pal.color(QPalette::Base).lightness() < 128; }
}  // namespace

QString hex(const QColor& c) { return c.name(QColor::HexRgb); }
QString good_colour(const QPalette& pal) {
    return dark_theme(pal) ? QStringLiteral("#5cc983") : QStringLiteral("#23824a");
}
QString cost_colour(const QPalette& pal) {
    return dark_theme(pal) ? QStringLiteral("#f0a648") : QStringLiteral("#a85d00");
}

QString bullet_list(const QStringList& items) {
    QString out;
    // A table, not <ul>: a label's list indent is a fixed 40 px.
    for (const QString& t : items) {
        out += QStringLiteral("<tr><td valign=\"top\" style=\"padding-right:4px;\">\u2022</td>"
                              "<td valign=\"top\" style=\"padding-bottom:3px;\">%1</td></tr>")
                   .arg(t.toHtmlEscaped());
    }
    return QStringLiteral("<table cellspacing=\"0\" cellpadding=\"0\" style=\"margin-top:3px;\">%1</table>")
        .arg(out);
}

QString info_html(const QPalette& pal, const QString& title, const QString& what,
                  const QStringList& good, const QStringList& cost, const QString& extra) {
    return info_html(pal, title, what, QString(), true, QObject::tr("Better at"), good,
                     QObject::tr("Costs"), cost, extra);
}

QString info_html(const QPalette& pal, const QString& title, const QString& what,
                  const QString& now, bool ok, const QString& good_head, const QStringList& good,
                  const QString& cost_head, const QStringList& cost, const QString& extra) {
    const QString muted = hex(style::secondary_text(pal));
    const auto head = [](const QString& colour, const QString& text) {
        return QStringLiteral("<span style=\"color:%1; font-size:small; "
                              "font-weight:600; letter-spacing:1px;\">%2</span>")
            .arg(colour, text.toUpper().toHtmlEscaped());
    };
    const QString now_line =
        now.isEmpty() ? QString()
                      : QStringLiteral("<p style=\"margin:8px 0 0 0; color:%1; font-weight:600;\">%2</p>")
                            .arg(ok ? good_colour(pal) : cost_colour(pal), now.toHtmlEscaped());
    return QStringLiteral(
               "<p style=\"margin:0; font-weight:700;\">%1</p>"
               "<p style=\"margin:2px 0 0 0; color:%2;\">%3</p>%4"
               // The headings get a row of their own, and the gap above them is
               // cell padding: a paragraph's bottom margin before the table
               // pushed one column's heading lower than the other's (Jeff,
               // 2026-10-15).
               "<table width=\"100%\" cellspacing=\"0\" cellpadding=\"0\"><tr>"
               "<td width=\"50%\" valign=\"bottom\" style=\"padding-top:8px;\">%5</td>"
               "<td width=\"50%\" valign=\"bottom\" style=\"padding-top:8px; padding-left:10px;\">%6</td>"
               "</tr><tr>"
               "<td width=\"50%\" valign=\"top\">%7</td>"
               "<td width=\"50%\" valign=\"top\" style=\"padding-left:10px;\">%8</td>"
               "</tr></table>%9")
        .arg(title.toHtmlEscaped(), muted, what.toHtmlEscaped(), now_line,
             head(good_colour(pal), good_head), head(cost_colour(pal), cost_head), bullet_list(good),
             bullet_list(cost), extra);
}

QString footnote(const QPalette& pal, const QString& text) {
    return QStringLiteral("<p style=\"margin:8px 0 0 0; color:%1; font-size:small;\">%2</p>")
        .arg(hex(style::secondary_text(pal)), text.toHtmlEscaped());
}

}  // namespace info

}  // namespace sstvae::gui

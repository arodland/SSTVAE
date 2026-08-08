#include "banner.hpp"

#include <QHBoxLayout>
#include <QLabel>
#include <QPushButton>
#include <QPalette>
#include <QStyle>

#include "style.hpp"

namespace sstvae::gui {

ErrorBanner::ErrorBanner(QWidget* parent) : QFrame(parent) {
    setFrameStyle(QFrame::StyledPanel | QFrame::Plain);
    // **Opaque, and with its own colours.** It used to sit on the pane
    // background and inherit it; now it floats over the picture, which
    // is nearly black in every theme -- so on a light desktop it drew
    // dark bold text on a dark picture and was barely readable, with
    // the picture's viewport showing through behind it. An alert
    // surface has to carry its own contrast wherever it lands.
    setAutoFillBackground(true);
    QPalette alert = palette();
    alert.setColor(QPalette::Window, style::color::danger_surface());
    alert.setColor(QPalette::WindowText, style::color::on_danger());
    setPalette(alert);

    auto* layout = new QHBoxLayout(this);
    layout->setContentsMargins(8, 4, 8, 4);

    // **Clicks fall through, except on Dismiss.** This banner floats
    // over the picture area, and on the transmit side that area is the
    // composing canvas -- so a sticky error, which is dismissed only by
    // hand, swallowed every click on the top ~40 px of it and an overlay
    // item placed near the top edge could not be selected at all. The
    // attribute is per widget and hit-testing reaches children first, so
    // the button below stays live while the frame and its two labels do
    // not.
    setAttribute(Qt::WA_TransparentForMouseEvents);

    icon_ = new QLabel(this);
    icon_->setAttribute(Qt::WA_TransparentForMouseEvents);
    const int size = style()->pixelMetric(QStyle::PM_SmallIconSize);
    icon_->setPixmap(style()
                         ->standardIcon(QStyle::SP_MessageBoxCritical)
                         .pixmap(size, size));
    layout->addWidget(icon_);

    text_ = new QLabel(this);
    text_->setWordWrap(true);
    text_->setAttribute(Qt::WA_TransparentForMouseEvents);
    // Inherit the banner's own palette rather than the window's.
    text_->setForegroundRole(QPalette::WindowText);
    // Bold rather than coloured: readable in every theme, and the icon
    // already says "error".
    QFont font = text_->font();
    font.setBold(true);
    text_->setFont(font);
    layout->addWidget(text_, 1);

    auto* dismiss = new QPushButton(tr("Dismiss"), this);
    connect(dismiss, &QPushButton::clicked, this, &ErrorBanner::clear);
    layout->addWidget(dismiss);

    hide();
}

void ErrorBanner::show_error(const QString& message) {
    text_->setText(message);
    show();
}

void ErrorBanner::clear() {
    text_->clear();
    hide();
}

QString ErrorBanner::message() const { return text_->text(); }

}  // namespace sstvae::gui

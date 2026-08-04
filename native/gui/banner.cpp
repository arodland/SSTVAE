#include "banner.hpp"

#include <QHBoxLayout>
#include <QLabel>
#include <QPushButton>
#include <QPalette>
#include <QStyle>

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
    alert.setColor(QPalette::Window, QColor(0x7a, 0x1f, 0x1a));
    alert.setColor(QPalette::WindowText, QColor(0xff, 0xf2, 0xf0));
    setPalette(alert);

    auto* layout = new QHBoxLayout(this);
    layout->setContentsMargins(8, 4, 8, 4);

    icon_ = new QLabel(this);
    const int size = style()->pixelMetric(QStyle::PM_SmallIconSize);
    icon_->setPixmap(style()
                         ->standardIcon(QStyle::SP_MessageBoxCritical)
                         .pixmap(size, size));
    layout->addWidget(icon_);

    text_ = new QLabel(this);
    text_->setWordWrap(true);
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

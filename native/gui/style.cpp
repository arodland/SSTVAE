#include "style.hpp"

#include <QEvent>
#include <QFontMetrics>
#include <QHBoxLayout>
#include <QHelpEvent>
#include <QImage>
#include <QPaintEvent>
#include <QPainter>
#include <QPalette>
#include <QPixmap>
#include <QToolButton>
#include <QToolTip>
#include <QVBoxLayout>
#include <QWidget>

#include <algorithm>
#include <cmath>
#include <utility>

#include "banner.hpp"

namespace sstvae::gui::style {

namespace {

// The one red. Everything else in `color::` is derived from it, so
// "the red" is a single edit rather than three literals in three files
// that were already 3% apart from each other.
const QColor DANGER(0xb3, 0x26, 0x1e);

double srgb_to_linear(int channel) {
    const double v = channel / 255.0;
    return v <= 0.03928 ? v / 12.92 : std::pow((v + 0.055) / 1.055, 2.4);
}

}  // namespace

// --- colour ------------------------------------------------------------------

double relative_luminance(const QColor& color) {
    return 0.2126 * srgb_to_linear(color.red()) +
           0.7152 * srgb_to_linear(color.green()) +
           0.0722 * srgb_to_linear(color.blue());
}

double contrast_ratio(const QColor& a, const QColor& b) {
    double lighter = relative_luminance(a);
    double darker = relative_luminance(b);
    if (lighter < darker) std::swap(lighter, darker);
    return (lighter + 0.05) / (darker + 0.05);
}

QColor secondary_text(const QPalette& palette) {
    const QColor fg = palette.color(QPalette::WindowText);
    const QColor bg = palette.color(QPalette::Window);

    // Walk from the theme's own text colour toward its background and
    // keep the last step that still clears the threshold. Blending
    // toward the background only ever lowers contrast, so the first
    // failure is the boundary and there is nothing beyond it worth
    // checking.
    //
    // Starting *at* `fg` is what makes the pathological case safe: a
    // theme whose own body text is already below 4.5:1 gets its own
    // colour back rather than something worse.
    constexpr int STEPS = 24;
    QColor best = fg;
    for (int step = 1; step <= STEPS; ++step) {
        const double t = static_cast<double>(step) / STEPS;
        const QColor candidate(
            static_cast<int>(std::lround(fg.red() + t * (bg.red() - fg.red()))),
            static_cast<int>(std::lround(fg.green() + t * (bg.green() - fg.green()))),
            static_cast<int>(std::lround(fg.blue() + t * (bg.blue() - fg.blue()))));
        if (contrast_ratio(candidate, bg) < MIN_CONTRAST) break;
        best = candidate;
    }
    return best;
}

namespace color {

QColor danger() { return DANGER; }
QColor danger_surface() { return DANGER.darker(150); }
QColor on_danger() { return QColor(0xff, 0xf2, 0xf0); }
QColor danger_bright() { return DANGER.lighter(160); }
QColor caution() { return QColor(0xff, 0xbe, 0x3c); }
QColor ok() { return QColor(0x5a, 0xdc, 0x78); }

QColor viewport() { return QColor(0x20, 0x20, 0x24); }
QColor viewport_frame() { return QColor(0x31, 0x31, 0x3a); }
QColor viewport_edge() { return QColor(0x55, 0x55, 0x61); }
QColor viewport_text() { return QColor(0x88, 0x88, 0x88); }

}  // namespace color

// --- text --------------------------------------------------------------------

void dim(QWidget* widget) {
    if (widget == nullptr) return;
    QPalette dimmed = widget->palette();
    const QColor quiet = secondary_text(dimmed);
    dimmed.setColor(QPalette::WindowText, quiet);
    dimmed.setColor(QPalette::Text, quiet);
    widget->setPalette(dimmed);
}

void undim(QWidget* widget) {
    // A default-constructed QPalette has every entry unresolved, so
    // this clears the override rather than pinning today's colours --
    // which is what a saved-and-restored copy would have done.
    if (widget != nullptr) widget->setPalette(QPalette());
}

QLabel* note(const QString& text, QWidget* parent) {
    auto* label = new QLabel(text, parent);
    label->setWordWrap(true);
    dim(label);
    return label;
}

QWidget* note_with_detail(const QString& summary, const QString& detail,
                          QWidget* parent) {
    auto* holder = new QWidget(parent);
    auto* layout = new QVBoxLayout(holder);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->setSpacing(2);

    layout->addWidget(note(summary, holder));

    QLabel* body = note(detail, holder);
    body->hide();

    // Full contrast, deliberately: it is a control, and the argument
    // for dimming the prose is exactly the argument against dimming the
    // thing that reveals it.
    auto* more = new QToolButton(holder);
    more->setAutoRaise(true);
    more->setCheckable(true);
    more->setToolButtonStyle(Qt::ToolButtonTextBesideIcon);
    more->setArrowType(Qt::RightArrow);
    more->setText(QObject::tr("More"));
    QObject::connect(more, &QToolButton::toggled, holder, [more, body](bool on) {
        more->setArrowType(on ? Qt::DownArrow : Qt::RightArrow);
        more->setText(on ? QObject::tr("Less") : QObject::tr("More"));
        body->setVisible(on);
    });
    layout->addWidget(more, 0, Qt::AlignLeft);
    layout->addWidget(body);
    return holder;
}

QWidget* row(QWidget* parent, std::initializer_list<QWidget*> widgets,
             int stretch_last) {
    auto* holder = new QWidget(parent);
    auto* layout = new QHBoxLayout(holder);
    layout->setContentsMargins(0, 0, 0, 0);
    int index = 0;
    for (QWidget* widget : widgets) {
        layout->addWidget(widget, index == stretch_last ? 1 : 0);
        ++index;
    }
    if (stretch_last < 0) layout->addStretch(1);
    return holder;
}

void ElidingLabel::paintEvent(QPaintEvent* event) {
    const QRect area = contentsRect();
    const QString full = text();
    const QString shown =
        fontMetrics().elidedText(full, Qt::ElideRight, area.width());
    elided_ = shown != full;
    if (!elided_) {
        QLabel::paintEvent(event);
        return;
    }
    QPainter painter(this);
    painter.setPen(palette().color(foregroundRole()));
    painter.drawText(area, static_cast<int>(alignment()), shown);
}

bool ElidingLabel::event(QEvent* event) {
    // The full text on hover, but only when there is something to
    // recover and only when the caller has not set a tooltip of its own
    // -- a control that explains itself should keep saying what it
    // says.
    if (event->type() == QEvent::ToolTip && toolTip().isEmpty()) {
        auto* help = static_cast<QHelpEvent*>(event);
        if (elided_) {
            QToolTip::showText(help->globalPos(), text(), this);
        } else {
            QToolTip::hideText();
        }
        return true;
    }
    return QLabel::event(event);
}

// --- pictures ----------------------------------------------------------------

QImage to_qimage(const images::Picture& picture) {
    if (picture.empty()) return QImage();
    // `copy()`, not the view: the view borrows `picture.rgb`, which the
    // caller is free to destroy the moment this returns.
    const QImage view(picture.rgb.data(), picture.width, picture.height,
                      picture.width * 3, QImage::Format_RGB888);
    return view.copy();
}

QPixmap to_pixmap(const images::Picture& picture) {
    const QImage image = to_qimage(picture);
    return image.isNull() ? QPixmap() : QPixmap::fromImage(image);
}

QString fmt_snr_db(double snr_db) {
    if (std::isnan(snr_db)) return QObject::tr("SNR --");
    return QObject::tr("SNR %1 dB").arg(snr_db, 0, 'f', 1);
}

// --- layout ------------------------------------------------------------------

void place_over(ErrorBanner* banner, const QWidget* target) {
    if (banner == nullptr || target == nullptr) return;
    const QRect over = target->geometry();
    const int wanted = banner->heightForWidth(over.width());
    banner->setGeometry(over.x(), over.y(), over.width(),
                        std::max(banner->sizeHint().height(), wanted));
    banner->raise();
}

}  // namespace sstvae::gui::style

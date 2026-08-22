#include "style.hpp"

#include <QEvent>
#include <QFontMetrics>
#include <QHBoxLayout>
#include <QHelpEvent>
#include <QMouseEvent>
#include <QImage>
#include <QPaintEvent>
#include <QPainter>
#include <QPalette>
#include <QPixmap>
#include <QStyle>
#include <QToolButton>
#include <QToolTip>
#include <QVBoxLayout>
#include <QWidget>

#include <algorithm>
#include <cmath>
#include <functional>
#include <utility>

#include "banner.hpp"

namespace sstvae::gui::style {

namespace {

// The gap between a disclosure's triangle and the text beside it, and
// therefore also the tail of the gutter every note reserves.
constexpr int GUTTER_SPACING = 4;

// The one red. Everything else in `color::` is derived from it, so
// "the red" is a single edit rather than three literals in three files
// that were already 3% apart from each other.
const QColor DANGER(0xb3, 0x26, 0x1e);

double srgb_to_linear(int channel) {
    const double v = channel / 255.0;
    return v <= 0.03928 ? v / 12.92 : std::pow((v + 0.055) / 1.055, 2.4);
}

// A wrapped note that runs a callback when clicked, so a disclosure's
// summary is as clickable as its triangle.
//
// No Q_OBJECT: it needs no signal of its own, only a virtual override
// and a stored callable -- the same reason `PictureBox` and
// `ElidingLabel` have none.
class ClickableNote : public QLabel {
public:
    ClickableNote(const QString& text, QWidget* parent, std::function<void()> on_click)
        : QLabel(text, parent), on_click_(std::move(on_click)) {
        setWordWrap(true);
        // The one cue that this paragraph is also a control. Without it
        // the triangle beside it is the only thing that looks live, and
        // the click target is 16 px wide.
        setCursor(Qt::PointingHandCursor);
    }

protected:
    void mouseReleaseEvent(QMouseEvent* event) override {
        // Only a release that lands inside counts -- pressing here and
        // dragging away is how a person changes their mind.
        if (event->button() == Qt::LeftButton && rect().contains(event->pos()) &&
            on_click_) {
            on_click_();
        }
        QLabel::mouseReleaseEvent(event);
    }

private:
    std::function<void()> on_click_;
};

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
// `ok()` is a light, bright green -- measured 11.95:1 against black,
// 1.76:1 against white -- so its companion text is dark rather than
// `on_danger()`'s light, and near-black rather than pure black to keep
// the same "tinted toward the hue" idea `on_danger()` uses.
QColor on_ok() { return QColor(0x0a, 0x33, 0x16); }

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

int note_gutter(const QWidget* reference) {
    const QStyle* s = reference != nullptr ? reference->style() : nullptr;
    const int arrow = s != nullptr ? s->pixelMetric(QStyle::PM_IndicatorWidth) : 14;
    return arrow + GUTTER_SPACING;
}

namespace {

// A note without the gutter, for use *inside* a disclosure where the
// gutter is already accounted for by the row it sits in.
QLabel* bare_note(const QString& text, QWidget* parent) {
    auto* label = new QLabel(text, parent);
    label->setWordWrap(true);
    dim(label);
    return label;
}

}  // namespace

QLabel* note(const QString& text, QWidget* parent) {
    QLabel* label = bare_note(text, parent);
    // **Every note starts in the same column, whether or not it has a
    // triangle.** A disclosure's summary is pushed right by its arrow,
    // so without reserving the same space here a plain note and a
    // summary in the same form began 20 px apart -- two kinds of help
    // text in two columns, which is the disconnection this whole change
    // is about. Reserving the gutter turns the triangle into a hanging
    // indent instead: text aligned, handles in the margin beside it.
    label->setContentsMargins(note_gutter(parent), 0, 0, 0);
    return label;
}

QWidget* note_with_detail(const QString& summary, const QString& detail,
                          QWidget* parent) {
    auto* holder = new QWidget(parent);
    auto* layout = new QVBoxLayout(holder);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->setSpacing(2);

    QLabel* body = bare_note(detail, holder);
    body->hide();

    // **The triangle leads the summary; there is no separate "More"
    // line.** It used to be one: summary, then a `▸ More` button on a
    // row of its own, indented *further* than the text it opened, with
    // the same amount of whitespace above it as below. Nothing tied it
    // to anything, so it read as a control floating in the margin
    // rather than as a handle on the paragraph above.
    //
    // A triangle at the head of the first line is the disclosure idiom
    // every desktop uses, and it puts the affordance where the thing it
    // reveals will appear.
    auto* arrow = new QToolButton(holder);
    arrow->setAutoRaise(true);
    arrow->setCheckable(true);
    arrow->setArrowType(Qt::RightArrow);
    arrow->setFocusPolicy(Qt::TabFocus);

    // The whole summary is the click target, not just the triangle.
    // A triangle is a small thing to hit, and the paragraph beside it
    // is what the operator is actually reading when they decide they
    // want more of it.
    auto* head = new ClickableNote(summary, holder,
                                   [arrow] { arrow->toggle(); });
    dim(head);  // it is a note first and a control second

    QObject::connect(arrow, &QToolButton::toggled, holder,
                     [arrow, body](bool on) {
                         arrow->setArrowType(on ? Qt::DownArrow : Qt::RightArrow);
                         arrow->setToolTip(on ? QObject::tr("Show less")
                                              : QObject::tr("Show more"));
                         body->setVisible(on);
                     });
    arrow->setToolTip(QObject::tr("Show more"));

    // Pinned to the gutter's width, so the triangle and the text beside
    // it land on exactly the columns a plain `note` reserves. Left to
    // its own size hint the two drift apart with the platform style.
    const int gutter = note_gutter(parent);
    arrow->setFixedWidth(gutter - GUTTER_SPACING);

    auto* head_row = new QWidget(holder);
    auto* head_layout = new QHBoxLayout(head_row);
    head_layout->setContentsMargins(0, 0, 0, 0);
    head_layout->setSpacing(GUTTER_SPACING);
    // Top-aligned: against a summary that wraps to three lines, a
    // vertically centred triangle points at the middle of the paragraph
    // instead of at its start.
    head_layout->addWidget(arrow, 0, Qt::AlignTop);
    head_layout->addWidget(head, 1);

    // The detail lines up with the summary rather than with the
    // triangle, so the two paragraphs read as one column of text with a
    // handle beside it.
    body->setContentsMargins(gutter, 0, 0, 0);

    layout->addWidget(head_row);
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

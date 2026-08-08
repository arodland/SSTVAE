// The shared appearance helpers.
//
// Two of them make claims that are checkable, and both replaced code
// that made the same claim and was wrong:
//
//   1. **`secondary_text` is a contrast guarantee, not a taste.** What
//      it replaced -- `QPalette::Disabled, QPalette::WindowText` --
//      measured **1.62:1** against the window on the default light
//      theme, where WCAG AA wants 4.5:1 and normal labels in the same
//      dialog measured 18.3:1. That colour is the style's answer to
//      "this control is unavailable", and the settings dialog's help
//      text is not an unavailable control; borrowing it made the
//      application's entire documentation surface the least readable
//      thing in the window. A guarantee nobody can check is a
//      preference wearing a lab coat, so it is checked here -- on a
//      light palette and a dark one, because a colour chosen against
//      one theme is exactly the bug being fixed.
//
//   2. **`ElidingLabel` must shorten rather than clip**, and must not
//      lose the text while doing it. A label that clips reads as text
//      that ends, so a status line cut after "de KD8X" reads as a
//      callsign that did not decode.
//
// Neither has an oracle in a screenshot: a badly-contrasted label is
// still a label, and clipped text and elided text are both short.

#include <QApplication>
#include <QColor>
#include <QLabel>
#include <QPalette>
#include <QPixmap>
#include <QPoint>
#include <QString>
#include <QWidget>

#include <cmath>
#include <string>
#include <utility>

#include "check.hpp"
#include "style.hpp"

using namespace sstvae;
using namespace sstvae::gui;

namespace {

// **The Disabled group is set deliberately, and to the measured
// values.** `#bebebe` on `#efefef` is what the real light theme
// actually produced -- 1.62:1, off a render of the Transmit settings
// tab. Without it these fixtures cannot reproduce the bug at all:
// `setColor(role, c)` writes every colour group, so a palette built
// from two calls has a Disabled foreground identical to its Active one,
// and the old implementation would look fine here while being
// unreadable on screen.
QPalette light_palette() {
    QPalette p;
    p.setColor(QPalette::Window, QColor(0xef, 0xef, 0xef));
    p.setColor(QPalette::WindowText, QColor(0x00, 0x00, 0x00));
    p.setColor(QPalette::Disabled, QPalette::WindowText, QColor(0xbe, 0xbe, 0xbe));
    return p;
}

QPalette dark_palette() {
    QPalette p;
    p.setColor(QPalette::Window, QColor(0x2b, 0x2b, 0x2b));
    p.setColor(QPalette::WindowText, QColor(0xf0, 0xf0, 0xf0));
    p.setColor(QPalette::Disabled, QPalette::WindowText, QColor(0x50, 0x50, 0x50));
    return p;
}

void test_contrast_ratio_matches_wcag() {
    // The two ends, which the formula pins exactly: black on white is
    // 21:1 and anything on itself is 1:1. If these move, the ratio is
    // not WCAG's and the threshold below means nothing.
    check::is_true(
        std::abs(style::contrast_ratio(QColor(Qt::black), QColor(Qt::white)) - 21.0) <
            1e-9,
        "contrast: black on white is 21:1");
    check::is_true(
        std::abs(style::contrast_ratio(QColor(0x77, 0x77, 0x77),
                                       QColor(0x77, 0x77, 0x77)) -
                 1.0) < 1e-12,
        "contrast: a colour against itself is 1:1");
}

void test_secondary_text_is_legible() {
    for (const auto& [palette, name] :
         {std::pair{light_palette(), "light"}, std::pair{dark_palette(), "dark"}}) {
        const QColor quiet = style::secondary_text(palette);
        const QColor bg = palette.color(QPalette::Window);
        const double ratio = style::contrast_ratio(quiet, bg);
        check::is_true(ratio >= style::MIN_CONTRAST,
                       std::string("secondary_text clears 4.5:1 on ") + name);

        // And it is *dimmer* than body text, or it is not doing its
        // other job -- a "quiet" colour identical to the loud one would
        // pass the check above and change nothing on screen.
        const QColor body = palette.color(QPalette::WindowText);
        check::is_true(ratio < style::contrast_ratio(body, bg),
                       std::string("secondary_text is quieter than body text on ") +
                           name);
    }

    // The pathological palette: body text that already fails. The
    // answer is the theme's own colour, never something worse -- this
    // is the one case where walking further would make a bad theme
    // unreadable rather than merely poor.
    QPalette bad;
    bad.setColor(QPalette::Window, QColor(0xef, 0xef, 0xef));
    bad.setColor(QPalette::WindowText, QColor(0xd0, 0xd0, 0xd0));
    check::is_true(style::secondary_text(bad) == bad.color(QPalette::WindowText),
                   "secondary_text on an already-failing palette returns it "
                   "unchanged");
}

void test_eliding_label_shortens_without_losing_text() {
    const QString full = QStringLiteral(
        "Receiving mode C: frame 220/220 (100%)  ·  SNR 8.3 dB  ·  de KD8XYZ");

    QWidget host;
    host.resize(1200, 100);
    auto* label = new style::ElidingLabel(full, &host);
    label->setGeometry(0, 0, 1000, 24);
    host.show();

    // Wide enough: nothing is touched, and the widget paints its text
    // the way any QLabel would.
    check::is_true(label->fontMetrics().horizontalAdvance(full) <= 1000,
                   "the fixture text fits at 1000 px");

    // **`text()` still returns the whole thing at any width.** This is
    // the property that made `paintEvent` the right place to elide: a
    // subclass shadowing the non-virtual `setText` would be bypassed
    // through a `QLabel*`, of which Qt holds many, and would also make
    // `text()` lie to anything that reads it back.
    label->setGeometry(0, 0, 120, 24);
    check::equal(label->text().toStdString(), full.toStdString(),
                 "elided: text() is untouched");

    // And it renders differently from a plain QLabel given the same
    // text at the same width.
    //
    // **This is the assertion that means anything.** Elision happens at
    // paint time by design, so no size hint and no property will answer
    // -- and comparing the render against itself at two widths would
    // only be measuring the geometry we just set. A plain QLabel here
    // *clips*: it paints as many whole pixels of the string as fit and
    // stops. So the two images differ exactly when this class is doing
    // its job, and are identical the moment `paintEvent` stops eliding.
    QLabel clipping(full, &host);
    clipping.setGeometry(0, 0, 120, 24);
    check::is_true(label->grab().toImage() != clipping.grab().toImage(),
                   "elided: renders differently from a label that clips");

    // Given room, it goes back to being an ordinary label.
    label->setGeometry(0, 0, 1000, 24);
    clipping.setGeometry(0, 0, 1000, 24);
    check::is_true(label->grab().toImage() == clipping.grab().toImage(),
                   "not elided: identical to a plain label when the text fits");
}

// A disclosure's summary and a plain note start in the same column.
//
// **This is arithmetic, not taste, which is why it is here.** The rest
// of "does this dialog group properly" has no oracle and belongs to
// `sstvae-gui-shot` and a pair of eyes. But two kinds of help text
// starting 20 px apart is a number, and it is the exact fault this
// gutter was added to fix: without it a plain note began at the left
// edge while a summary was pushed right by its triangle, so the same
// form showed help in two columns.
void test_notes_and_disclosures_share_a_column() {
    QWidget host;
    host.resize(600, 400);

    QLabel* plain = style::note(QStringLiteral("A plain note."), &host);
    QWidget* disclosed = style::note_with_detail(
        QStringLiteral("A summary."), QStringLiteral("The detail."), &host);

    // Where the *text* begins in each, relative to the widget's own
    // left edge: a margin for the plain label, and the position the
    // head row's layout gives the summary for the disclosure.
    const int plain_x = plain->contentsMargins().left();

    QLabel* summary = nullptr;
    for (QLabel* label : disclosed->findChildren<QLabel*>()) {
        if (label->text() == QLatin1String("A summary.")) summary = label;
    }
    check::is_true(summary != nullptr, "the disclosure has its summary");

    host.show();
    plain->setGeometry(0, 0, 500, 40);
    disclosed->setGeometry(0, 40, 500, 40);
    QCoreApplication::processEvents();

    // mapTo, not x(): the summary sits inside a head row inside the
    // holder, so its own x() is relative to a parent that is not the
    // one the plain note is measured against.
    const int summary_x = summary->mapTo(disclosed, QPoint(0, 0)).x() +
                          summary->contentsMargins().left();
    check::equal(summary_x, plain_x,
                 "a disclosure's summary starts where a plain note does");
    check::is_true(plain_x > 0, "the gutter is actually reserved");
}

}  // namespace

int main(int argc, char** argv) {
    check::report_crashes_instead_of_prompting();
    qputenv("QT_QPA_PLATFORM", "offscreen");
    QApplication app(argc, argv);

    test_contrast_ratio_matches_wcag();
    test_secondary_text_is_legible();
    test_eliding_label_shortens_without_losing_text();
    test_notes_and_disclosures_share_a_column();
    return check::report("style");
}

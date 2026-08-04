// The receive/transmit layout switch.
//
// Four claims, each of which fails invisibly if it is wrong:
//
//   1. The panels survive a mode switch. They are not toys -- the real
//      ones own a decode thread and a transmit thread -- so a container
//      that deletes its children when it is replaced would crash a long
//      way from here, at exit or at the next callback.
//   2. They come back *visible*. Reparenting through a null parent sets
//      Qt's explicit-hidden flag, and a widget hidden that way stays
//      hidden when it is added to a visible layout. The window would
//      look empty and nothing would have failed.
//   3. Tabs really are narrower. This is the entire justification for
//      the feature: a QSplitter's minimum is the sum of its children's,
//      a QTabWidget's is the max. If it did not hold, "small screen
//      mode" would be a menu entry that changed nothing.
//   4. "auto" resolves against the screen and honours an explicit
//      setting. The dangerous case is a screen we could not measure --
//      it must not read as "tiny".
//
// Deliberately built from two plain widgets rather than the real
// panels: the switch is container logic, and pulling in AppState would
// start a model load and open the rig.

#include <QApplication>
#include <QLabel>
#include <QPushButton>
#include <QSize>
#include <QSizePolicy>
#include <QPointer>
#include <QSize>
#include <QSizePolicy>
#include <QVBoxLayout>
#include <QWidget>

#include <string>

#include "check.hpp"
#include "flow_layout.hpp"
#include "pane_container.hpp"

using namespace sstvae;
using gui::PaneLayout;

namespace {

// A pane with a real minimum width, since that is what the third claim
// above is about. A bare QWidget has a minimum of zero and both layouts
// would agree on nothing.
QWidget* pane(int minimum_width) {
    auto* w = new QWidget;
    auto* layout = new QVBoxLayout(w);
    auto* label = new QLabel(QStringLiteral("x"));
    label->setMinimumWidth(minimum_width);
    layout->addWidget(label);
    return w;
}

void test_panels_survive_and_stay_visible() {
    auto* first = pane(200);
    auto* second = pane(200);
    QPointer<QWidget> first_alive(first);
    QPointer<QWidget> second_alive(second);

    gui::PaneContainer container(first, QStringLiteral("Receive"), second,
                                 QStringLiteral("Transmit"));
    container.resize(900, 500);
    container.show();

    check::equal(static_cast<int>(container.mode()),
                 static_cast<int>(PaneLayout::Split), "starts side by side");

    // Switch both ways twice: a container that leaks or deletes on one
    // direction only would pass a single switch.
    for (int round = 0; round < 2; ++round) {
        container.set_mode(PaneLayout::Tabs);
        check::is_true(!first_alive.isNull() && !second_alive.isNull(),
                       "panels survive the switch to tabs");
        check::is_true(first->isVisible(), "the current tab's panel is visible");
        // The second tab is *correctly* hidden -- that is what a tab
        // widget is -- so what is checked is that it is still owned and
        // becomes visible when selected, not that it is on screen.
        check::is_true(second->parentWidget() != nullptr,
                       "the background tab's panel is still parented");

        container.set_mode(PaneLayout::Split);
        check::is_true(!first_alive.isNull() && !second_alive.isNull(),
                       "panels survive the switch back to side by side");
        check::is_true(first->isVisible() && second->isVisible(),
                       "both panels are visible side by side");
    }
}

void test_tabs_are_narrower_than_side_by_side() {
    // Equal minimums, so the arithmetic is unambiguous: the sum is
    // twice the max, and any result between them would mean the layout
    // is not doing what the whole feature assumes.
    auto* first = pane(300);
    auto* second = pane(300);
    gui::PaneContainer container(first, QStringLiteral("Receive"), second,
                                 QStringLiteral("Transmit"));

    const int split_min = container.minimumSizeHint().width();
    container.set_mode(PaneLayout::Tabs);
    const int tabs_min = container.minimumSizeHint().width();

    // Not "smaller by some tolerance" -- the claim is structural. Tabs
    // hold one pane's minimum where the splitter holds two, so the gap
    // has to be about a whole pane. Checked loosely (chrome, handle and
    // group-box margins differ between the two) but not so loosely that
    // a few pixels would pass.
    check::is_true(tabs_min < split_min,
                   "tabs have a smaller minimum width than side by side");
    check::is_true(tabs_min < split_min * 3 / 4,
                   "the saving is about a pane, not a margin");
}

void test_resolve_layout() {
    // An explicit setting wins over the screen, in both directions:
    // wanting both halves on a small panel is allowed, and so is
    // wanting tabs on a large one.
    check::equal(static_cast<int>(gui::resolve_layout("split", QSize(800, 1200), QSize(1015, 700))),
                 static_cast<int>(PaneLayout::Split),
                 "an explicit 'split' is honoured on a screen too small for it");
    check::equal(static_cast<int>(gui::resolve_layout("tabs", QSize(3840, 1200), QSize(1015, 700))),
                 static_cast<int>(PaneLayout::Tabs),
                 "an explicit 'tabs' is honoured on a large screen");

    check::equal(static_cast<int>(gui::resolve_layout("auto", QSize(1920, 1200), QSize(1015, 700))),
                 static_cast<int>(PaneLayout::Split), "auto: it fits");
    check::equal(static_cast<int>(gui::resolve_layout("auto", QSize(800, 1200), QSize(1015, 700))),
                 static_cast<int>(PaneLayout::Tabs), "auto: it does not fit");
    // Exactly the available width is a fit: a maximised window is a
    // legitimate way to run this app, and availableGeometry has already
    // taken the panels and docks off.
    check::equal(static_cast<int>(gui::resolve_layout("auto", QSize(1015, 1200), QSize(1015, 700))),
                 static_cast<int>(PaneLayout::Split), "auto: exactly fits");

    // **Height is a constraint too**, and this is what that adds: a
    // screen wide enough for side by side but too short for it must
    // still choose tabs. Before this, only width was compared -- fine
    // while the split layout's height floor was fixed, and wrong once
    // wrapping made that floor rise as the window narrows.
    check::equal(static_cast<int>(gui::resolve_layout("auto", QSize(1920, 600),
                                                     QSize(1015, 800))),
                 static_cast<int>(PaneLayout::Tabs),
                 "auto: wide enough but too short still picks tabs");
    check::equal(static_cast<int>(gui::resolve_layout("auto", QSize(1920, 900),
                                                     QSize(1015, 800))),
                 static_cast<int>(PaneLayout::Split),
                 "auto: room in both axes stays side by side");
    // An unmeasurable screen is an unanswered question in either axis.
    check::equal(static_cast<int>(gui::resolve_layout("auto", QSize(1920, 0),
                                                     QSize(1015, 800))),
                 static_cast<int>(PaneLayout::Split),
                 "auto: an unmeasurable height falls to side by side");

    // The failure that must not read as "small screen". A screen we
    // could not measure is an unanswered question, and answering it
    // with the fallback layout would quietly demote every operator
    // whose platform did not report a geometry.
    check::equal(static_cast<int>(gui::resolve_layout("auto", QSize(0, 1200), QSize(1015, 700))),
                 static_cast<int>(PaneLayout::Split),
                 "auto: an unmeasurable screen falls to side by side");
}

// A stand-in pane: a picture that takes the room, over a control strip
// of a given height. The two real panes differ only in how tall their
// strips are, which is exactly what this reproduces.
struct Pane {
    QWidget* root;
    QWidget* picture;
    QWidget* strip;
};

Pane make_pane(int strip_rows) {
    auto* root = new QWidget;
    auto* layout = new QVBoxLayout(root);
    layout->setContentsMargins(0, 0, 0, 0);
    auto* picture = new QWidget(root);
    picture->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    picture->setMinimumSize(80, 60);
    layout->addWidget(picture, 1);
    auto* strip = new QWidget(root);
    auto* strip_layout = new QVBoxLayout(strip);
    strip_layout->setContentsMargins(0, 0, 0, 0);
    for (int i = 0; i < strip_rows; ++i) {
        // **A wrapping row of real buttons, not a fixed-height label.**
        // The first version of this fixture used labels of a fixed
        // height, so the strips could not wrap and the test passed
        // without ever exercising the case that broke: at narrow widths
        // the two strips reflow to *different* row counts, and a single
        // measurement of their heights is then wrong. Different button
        // counts per row make the two sides wrap at different widths,
        // which is exactly the real panes' situation.
        auto* row = new QWidget(strip);
        auto* flow = new gui::FlowLayout(row);
        for (int b = 0; b < 3 + i * 2; ++b) {
            auto* button = new QPushButton(QStringLiteral("button %1").arg(b), row);
            flow->addWidget(button);
        }
        strip_layout->addWidget(row);
    }
    layout->addWidget(strip);
    return {root, picture, strip};
}

// **The property the whole layout exists to provide**, and the one that
// had no test while two bugs hid behind it: the two pictures are the
// same size.
//
// It is asserted on *unequal* panes -- three control rows against one --
// because equal-by-accident is exactly what a fixture of two identical
// panes would prove. The real panes differ the same way: the composer
// carries tools, properties and a send bar where the monitor carries a
// status line, a card and its buttons.
void test_the_two_pictures_are_the_same_size() {
    Pane rx = make_pane(1);
    Pane tx = make_pane(3);

    gui::PaneContainer container(rx.root, QStringLiteral("Receive"), tx.root,
                                 QStringLiteral("Transmit"));
    container.set_control_strips(rx.strip, tx.strip);
    container.show();

    // Deliberately spanning the widths where the rows wrap differently:
    // 760 and 900 are narrow enough that the two strips take different
    // numbers of lines, which is where a single-pass measurement failed.
    for (const QSize size : {QSize(760, 900), QSize(900, 1100), QSize(1200, 700),
                             QSize(1600, 900), QSize(2000, 620)}) {
        for (int pass = 0; pass < 2; ++pass) {
            container.setGeometry(0, 0, size.width(), size.height());
            QCoreApplication::processEvents();
        }
        const std::string at = " at " + std::to_string(size.width()) + "x" +
                               std::to_string(size.height());
        check::equal(rx.strip->height(), tx.strip->height(),
                     "the two control strips are the same height" + at);
        check::equal(rx.picture->width(), tx.picture->width(),
                     "the two pictures are the same width" + at);
        check::equal(rx.picture->height(), tx.picture->height(),
                     "the two pictures are the same height" + at);
        // Not merely equal -- *large*. Shrinking both to a postage stamp
        // would satisfy every assertion above, and is precisely the
        // failure the previous implementation shipped.
        check::is_true(rx.picture->height() > size.height() / 3,
                       "and the picture uses the room it was given" + at);
    }
}

// Equal sizes must survive the layout switch, which is where the last
// implementation lost them: the mode change did not re-run the sizing,
// so a picture kept the size it had while sharing the window.
void test_sizes_survive_a_mode_round_trip() {
    Pane rx = make_pane(1);
    Pane tx = make_pane(3);
    gui::PaneContainer container(rx.root, QStringLiteral("Receive"), tx.root,
                                 QStringLiteral("Transmit"));
    container.set_control_strips(rx.strip, tx.strip);
    container.show();
    for (int pass = 0; pass < 2; ++pass) {
        container.setGeometry(0, 0, 1400, 800);
        QCoreApplication::processEvents();
    }
    const QSize before = rx.picture->size();

    container.set_mode(PaneLayout::Tabs);
    QCoreApplication::processEvents();
    container.set_mode(PaneLayout::Split);
    for (int pass = 0; pass < 2; ++pass) {
        container.setGeometry(0, 0, 1400, 800);
        QCoreApplication::processEvents();
    }

    check::equal(rx.picture->width(), before.width(),
                 "a round trip through tabs leaves the picture's width alone");
    check::equal(rx.picture->height(), before.height(),
                 "and its height");
    check::equal(rx.picture->height(), tx.picture->height(),
                 "and the two are still equal afterwards");
}

// **A widget that must not pin the window's width still has to be
// visible.** `QSizePolicy::Ignored` is how several progress-tier
// surfaces avoid setting a width floor -- and `QWidgetItem::sizeHint()`
// answers zero for an Ignored axis, which a layout with stretch to give
// can interpret sensibly and this one cannot. Laid out literally, the
// transmit status label and the send progress bar were 0 px wide: the
// refiner's running "+x.x dB" report simply stopped appearing, and an
// operator noticed before any test did.
void test_ignored_width_widgets_are_still_laid_out() {
    QWidget host;
    auto* flow = new gui::FlowLayout(&host);
    auto* normal = new QPushButton(QStringLiteral("Send"), &host);
    auto* ignored = new QLabel(QStringLiteral("Refining picture... +3.2 dB est."), &host);
    ignored->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Preferred);
    flow->addWidget(normal);
    flow->addWidget(ignored);
    host.resize(900, 80);
    host.show();
    QCoreApplication::processEvents();

    check::is_true(ignored->width() > 0,
                   "an Ignored-width widget is given a real width");
    check::is_true(ignored->width() >= ignored->sizeHint().width() / 2,
                   "and enough of one to read its text");
    // ...while still not raising the floor, which is the reason it was
    // marked Ignored in the first place. The layout's minimum must come
    // from the *other* item.
    check::is_true(flow->minimumSize().width() <= normal->sizeHint().width() + 4,
                   "and it still does not pin the layout's minimum width");
}

}  // namespace

int main(int argc, char** argv) {
    check::report_crashes_instead_of_prompting();
    qputenv("QT_QPA_PLATFORM", "offscreen");
    const QApplication app(argc, argv);

    test_panels_survive_and_stay_visible();
    test_tabs_are_narrower_than_side_by_side();
    test_resolve_layout();
    test_the_two_pictures_are_the_same_size();
    test_sizes_survive_a_mode_round_trip();
    test_ignored_width_widgets_are_still_laid_out();

    return check::report("pane container");
}

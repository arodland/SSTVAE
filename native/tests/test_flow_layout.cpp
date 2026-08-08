// The wrapping row used by both panels' control strips.
//
// Three properties, all of which have been wrong at some point and none
// of which a screenshot settles on its own:
//
//   1. **Items on one line are vertically centred, not top-aligned.**
//      A flow layout learns a line's height only once the line is
//      finished, so the obvious implementation -- set each item's
//      geometry as you visit it -- pins everything to the top. In a row
//      that mixes buttons with text, the text then floats above the
//      buttons beside it. Reported from the running app, on "No image
//      selected" between two buttons and on "Level:" beside its slider.
//
//   2. **The minimum width is the widest item, not the sum**, which is
//      the whole reason this layout exists: both panes share one width
//      floor, so a strip that demanded the total width of its buttons
//      would set that floor for the entire window.
//
//   3. **An `Ignored` horizontal policy must not lay out at zero
//      width.** `QWidgetItem::sizeHint` reports 0 on an ignored axis;
//      several widgets carry that policy precisely so they cannot pin
//      the window's width, and they vanished -- the transmit status
//      label and the send progress bar were both laid out 0 px wide and
//      simply were not on screen.

#include <QApplication>
#include <QLabel>
#include <QPushButton>
#include <QSizePolicy>
#include <QWidget>

#include "check.hpp"
#include "flow_layout.hpp"

using namespace sstvae;
using namespace sstvae::gui;

namespace {

void test_a_line_is_vertically_centred() {
    QWidget host;
    auto* flow = new FlowLayout(&host);

    // A tall item and a short one on the same line, which is the shape
    // of every control strip in the app: buttons beside a caption.
    auto* tall = new QPushButton(QStringLiteral("Tall"), &host);
    tall->setFixedSize(80, 40);
    auto* shortest = new QLabel(QStringLiteral("Short"), &host);
    shortest->setFixedSize(60, 16);
    flow->addWidget(tall);
    flow->addWidget(shortest);

    host.resize(400, 100);
    host.show();
    flow->setGeometry(QRect(0, 0, 400, 100));

    check::equal(tall->geometry().height(), 40, "the tall item keeps its height");
    check::equal(shortest->geometry().height(), 16, "the short item keeps its height");
    check::is_true(tall->geometry().top() == shortest->geometry().top() - 12,
                   "the short item is centred against the tall one");
    // Stated as centres too, because the offset above is only the right
    // number if both are on the same line to begin with.
    check::equal(shortest->geometry().center().y(), tall->geometry().center().y(),
                 "their centres agree");
}

void test_each_line_is_centred_independently() {
    QWidget host;
    auto* flow = new FlowLayout(&host);

    // Two lines: a tall pair, then a short pair forced onto the next
    // line. The second line must centre against *its own* height, not
    // against the first line's -- a single line-height variable reused
    // across the loop is the easy way to get that wrong.
    auto* a = new QLabel(QStringLiteral("A"), &host);
    a->setFixedSize(120, 40);
    auto* b = new QLabel(QStringLiteral("B"), &host);
    b->setFixedSize(120, 10);
    auto* c = new QLabel(QStringLiteral("C"), &host);
    c->setFixedSize(120, 20);
    auto* d = new QLabel(QStringLiteral("D"), &host);
    d->setFixedSize(120, 12);
    for (QLabel* w : {a, b, c, d}) flow->addWidget(w);

    host.resize(260, 200);
    host.show();
    flow->setGeometry(QRect(0, 0, 260, 200));

    check::is_true(a->geometry().top() < c->geometry().top(),
                   "the third item wrapped to a second line");
    check::equal(b->geometry().center().y(), a->geometry().center().y(),
                 "line one is centred on itself");
    check::equal(d->geometry().center().y(), c->geometry().center().y(),
                 "line two is centred on itself");
}

void test_the_floor_is_the_widest_item() {
    QWidget host;
    auto* flow = new FlowLayout(&host);
    for (const int width : {60, 150, 90}) {
        auto* button = new QPushButton(&host);
        button->setFixedSize(width, 24);
        flow->addWidget(button);
    }
    check::equal(flow->minimumSize().width(), 150,
                 "the minimum is the widest item, not the sum");
}

void test_an_ignored_width_still_gets_laid_out() {
    QWidget host;
    auto* flow = new FlowLayout(&host);
    auto* label = new QLabel(QStringLiteral("A status line of some length"), &host);
    label->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Preferred);
    flow->addWidget(label);

    host.resize(400, 60);
    host.show();
    flow->setGeometry(QRect(0, 0, 400, 60));

    check::is_true(label->geometry().width() > 0,
                   "an Ignored width is laid out at its widget's hint, not zero");
    // ...and still does not raise the floor, which is the reason the
    // policy is Ignored in the first place.
    check::equal(flow->minimumSize().width(), 0,
                 "an Ignored width contributes nothing to the minimum");
}

}  // namespace

int main(int argc, char** argv) {
    check::report_crashes_instead_of_prompting();
    qputenv("QT_QPA_PLATFORM", "offscreen");
    QApplication app(argc, argv);

    test_a_line_is_vertically_centred();
    test_each_line_is_centred_independently();
    test_the_floor_is_the_widest_item();
    test_an_ignored_width_still_gets_laid_out();
    return check::report("flow layout");
}

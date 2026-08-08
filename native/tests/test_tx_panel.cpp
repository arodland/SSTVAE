// The transmit pane's control strip, which must not change height.
//
// **This is the guard for a bug that broke the other pane.** The
// "Selected item" row used to hide itself when nothing was selected,
// while the comment where it is built said the opposite in bold. Two
// things followed, and the second is why this file exists:
//
//   1. The row is ~90 px, so the canvas jumped under the pointer on
//      every select and deselect -- on a composing surface, where the
//      thing being clicked is the thing that moves.
//
//   2. `PaneContainer::equalise_strips` -- the entire mechanism that
//      makes the received image and the composed image the same size --
//      runs only from `set_control_strips` and from a resize. Nothing
//      re-runs it when a strip's *content* changes height. So from the
//      first click on the canvas, both strips held minimums computed
//      for a layout that no longer existed and the two images silently
//      stopped matching, until something happened to resize the window.
//
// `test_pane_container.cpp` cannot catch this: it drives the container
// with stand-in panes whose strips are static, which is right for what
// it tests and blind to what this tests. The check has to be on the
// real panel.
//
// Deliberately a height *invariant* rather than a "the box is visible"
// assertion: what the layout cares about is the number, and hiding the
// row by some other means later would break the panes the same way.

#include <QApplication>
#include <QLayout>
#include <QWidget>

#include <string>

#include "app_state.hpp"
#include "check.hpp"
#include "overlay_editor.hpp"
#include "tx_panel.hpp"

using namespace sstvae;
using namespace sstvae::gui;

namespace {

// The strip's height at the width it will actually have.
//
// Not `sizeHint()`: the rows inside wrap, so the height is a function
// of the width, and a hint taken at an unconstrained width reports one
// line -- the same trap `PaneContainer::strip_height_for_width` records.
int strip_height(QWidget* strip) {
    QLayout* layout = strip->layout();
    if (layout != nullptr && layout->hasHeightForWidth() && strip->width() > 0) {
        return layout->minimumHeightForWidth(strip->width());
    }
    return strip->sizeHint().height();
}

void test_the_strip_height_survives_a_selection() {
    // `AppState`'s constructor loads settings and opens the log; it
    // does not fetch a model, and `TransmitPanel` with no codec simply
    // does not arm the optimizer. So this needs no network and no
    // radio.
    AppState state;
    QWidget host;
    host.resize(900, 700);
    auto* panel = new TransmitPanel(&state, &host);
    panel->setGeometry(0, 0, 900, 700);
    host.show();
    QCoreApplication::processEvents();

    QWidget* strip = panel->control_strip();
    const int idle = strip_height(strip);
    check::is_true(idle > 0, "the strip has a height to begin with");

    // Adding a text item selects it, which is the transition that used
    // to reveal the properties row.
    auto* editor = panel->findChild<OverlayEditor*>();
    check::is_true(editor != nullptr, "the panel has an overlay editor");
    editor->add_text(std::string("KD8XYZ"));
    QCoreApplication::processEvents();
    check::equal(strip_height(strip), idle,
                 "the strip is the same height with an item selected");

    // And back to nothing selected, which is the transition that used
    // to hide the row again. Through `remove_selected`, which is a real
    // gesture (the Remove button, or Delete) and reaches the same state
    // as clicking empty canvas -- `select` itself is private, and this
    // test has no business reaching past the panel's own surface.
    editor->remove_selected();
    QCoreApplication::processEvents();
    check::equal(strip_height(strip), idle,
                 "the strip is the same height with nothing selected");
}

}  // namespace

int main(int argc, char** argv) {
    check::report_crashes_instead_of_prompting();
    qputenv("QT_QPA_PLATFORM", "offscreen");
    QApplication app(argc, argv);

    test_the_strip_height_survives_a_selection();
    return check::report("transmit panel");
}

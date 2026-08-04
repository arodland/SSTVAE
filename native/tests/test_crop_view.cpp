// The framing dialog's interaction arithmetic.
//
// Everything here has a right answer and none of it needs eyes, which
// is the same line `test_overlay_editor.cpp` draws. The one that
// matters most is the drag direction: `paintEvent` holds the picture
// still and moves the crop window, so a drag to the right must select
// a window further right. The first version of this widget had the
// opposite sign -- taken from a comment describing a widget that pans
// the picture instead -- so dragging right selected further left, and
// nothing in the repository could see it.

#include <QApplication>
#include <QMouseEvent>
#include <QWheelEvent>

#include <cmath>

#include "check.hpp"
#include "crop_dialog.hpp"
#include "images/images.hpp"

namespace check = sstvae::check;
using sstvae::gui::CropView;
using sstvae::images::Framing;
using sstvae::images::Picture;

namespace {

constexpr int W = 520;
constexpr int H = 390;

Picture flat(int width, int height) {
    Picture p(width, height);
    for (std::size_t i = 0; i < p.rgb.size(); ++i) {
        p.rgb[i] = static_cast<std::uint8_t>(i % 251);
    }
    return p;
}

void drag(CropView& view, double from_x, double from_y, double dx, double dy) {
    const QPointF from(from_x, from_y);
    const QPointF to(from_x + dx, from_y + dy);
    QMouseEvent press(QEvent::MouseButtonPress, from, view.mapToGlobal(from),
                      Qt::LeftButton, Qt::LeftButton, Qt::NoModifier);
    QApplication::sendEvent(&view, &press);
    QMouseEvent move(QEvent::MouseMove, to, view.mapToGlobal(to), Qt::NoButton,
                     Qt::LeftButton, Qt::NoModifier);
    QApplication::sendEvent(&view, &move);
    QMouseEvent release(QEvent::MouseButtonRelease, to, view.mapToGlobal(to),
                        Qt::LeftButton, Qt::NoButton, Qt::NoModifier);
    QApplication::sendEvent(&view, &release);
}

void wheel(CropView& view, int delta, int times) {
    for (int i = 0; i < times; ++i) {
        QWheelEvent event(QPointF(W / 2.0, H / 2.0),
                          view.mapToGlobal(QPointF(W / 2.0, H / 2.0)),
                          QPoint(0, 0), QPoint(0, delta), Qt::NoButton,
                          Qt::NoModifier, Qt::NoScrollPhase, false);
        QApplication::sendEvent(&view, &event);
    }
}

void test_drag_moves_the_window_with_the_pointer() {
    CropView view;
    view.resize(W, H);
    view.set_source(flat(1600, 900));  // 16:9: room to pan horizontally
    view.set_framing(Framing{});

    const double start = view.framing().center_x;
    drag(view, W / 2.0, H / 2.0, 40, 0);
    check::is_true(view.framing().center_x > start,
                   "crop/drag: dragging right selects further right");

    const double after_right = view.framing().center_x;
    drag(view, W / 2.0, H / 2.0, -40, 0);
    check::is_true(view.framing().center_x < after_right,
                   "crop/drag: dragging left comes back");
}

void test_drag_is_vertical_too_when_there_is_room() {
    CropView view;
    view.resize(W, H);
    view.set_source(flat(900, 1600));  // portrait: room to pan vertically
    view.set_framing(Framing{});

    const double start = view.framing().center_y;
    drag(view, W / 2.0, H / 2.0, 0, 30);
    check::is_true(view.framing().center_y > start,
                   "crop/drag: dragging down selects further down");
}

void test_drag_stops_at_the_edges() {
    CropView view;
    view.resize(W, H);
    view.set_source(flat(1600, 900));
    view.set_framing(Framing{});

    // Far beyond the picture: the window must stop at the edge rather
    // than hang over nothing, which `images::fit` would then clamp
    // differently from what was previewed.
    drag(view, W / 2.0, H / 2.0, 5000, 5000);
    const Framing f = view.framing();
    check::is_true(f.center_x <= 1.0 && f.center_x >= 0.0,
                   "crop/drag: centre_x stays inside the picture");
    check::is_true(f.center_y <= 1.0 && f.center_y >= 0.0,
                   "crop/drag: centre_y stays inside the picture");
    // A 16:9 source at zoom 1 has no vertical slack at all, so the
    // vertical centre cannot have moved.
    check::is_true(std::abs(f.center_y - 0.5) < 1e-9,
                   "crop/drag: no vertical travel when there is no slack");
}

void test_a_trackpad_gesture_does_not_slam_the_zoom() {
    CropView view;
    view.resize(W, H);
    view.set_source(flat(1600, 900));
    view.set_framing(Framing{});

    // Ten high-resolution events of 4 units each -- a third of one
    // physical detent. Treating each event as a full notch ran this to
    // the 4x ceiling; accumulating leaves it untouched.
    wheel(view, 4, 10);
    check::is_true(std::abs(view.framing().zoom - 1.0) < 1e-9,
                   "crop/wheel: a third of a notch does not zoom");

    // And a real notch does move it, once.
    wheel(view, 120, 1);
    check::is_true(view.framing().zoom > 1.0,
                   "crop/wheel: one full notch zooms in");
    check::is_true(view.framing().zoom < 1.2,
                   "crop/wheel: by one step, not several");
}

void test_zoom_is_bounded() {
    CropView view;
    view.resize(W, H);
    view.set_source(flat(1600, 900));
    view.set_framing(Framing{});

    wheel(view, 120, 200);
    check::is_true(view.framing().zoom <= 4.0 + 1e-9,
                   "crop/wheel: zoom stops at the ceiling");
    wheel(view, -120, 400);
    check::is_true(view.framing().zoom >= 1.0 - 1e-9,
                   "crop/wheel: and never below cover");
}

void test_an_empty_source_is_harmless() {
    // The dialog is never opened without a picture, but a widget that
    // divides by a zero-sized source would be a crash rather than a
    // wrong picture, so it is worth one assertion.
    CropView view;
    view.resize(W, H);
    view.set_framing(Framing{});
    drag(view, 10, 10, 20, 20);
    wheel(view, 120, 1);
    const Framing f = view.framing();
    check::is_true(!std::isnan(f.center_x) && !std::isnan(f.center_y),
                   "crop/empty: no NaN from a source-less view");
}

}  // namespace

int main(int argc, char** argv) {
    check::report_crashes_instead_of_prompting();
    qputenv("QT_QPA_PLATFORM", "offscreen");
    const QApplication app(argc, argv);

    test_drag_moves_the_window_with_the_pointer();
    test_drag_is_vertical_too_when_there_is_room();
    test_drag_stops_at_the_edges();
    test_a_trackpad_gesture_does_not_slam_the_zoom();
    test_zoom_is_bounded();
    test_an_empty_source_is_harmless();

    return check::report("crop view");
}

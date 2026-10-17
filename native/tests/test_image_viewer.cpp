// The picture viewer: double-click a received picture, zoom with the
// wheel about the pointer, drag to pan.
//
// "About the pointer" is the part that looks right when it is wrong --
// a zoom about the corner or the centre still zooms -- so it is checked
// as arithmetic: the picture point under the pointer before a wheel
// notch is the one under it after.

#include <QApplication>
#include <QImage>
#include <QJsonObject>
#include <QMouseEvent>
#include <QPixmap>
#include <QTemporaryDir>
#include <QWheelEvent>
#include <QWidget>

#include <cmath>

#include "check.hpp"
#include "image_viewer.hpp"
#include "picture_box.hpp"
#include "qrss_window.hpp"

using namespace sstvae;

namespace {

QPixmap picture() {
    QImage img(640, 480, QImage::Format_RGB32);
    img.fill(Qt::darkGreen);
    return QPixmap::fromImage(img);
}

void send_wheel(QWidget& w, QPointF at, int delta) {
    QWheelEvent e(at, w.mapToGlobal(at), QPoint(), QPoint(0, delta), Qt::NoButton,
                  Qt::NoModifier, Qt::NoScrollPhase, false);
    QApplication::sendEvent(&w, &e);
}

void send_mouse(QWidget& w, QEvent::Type type, QPointF at, Qt::MouseButton button = Qt::LeftButton) {
    QMouseEvent e(type, at, w.mapToGlobal(at), button,
                  type == QEvent::MouseButtonRelease ? Qt::NoButton : Qt::LeftButton,
                  Qt::NoModifier);
    QApplication::sendEvent(&w, &e);
}

bool close_to(QPointF a, QPointF b) { return std::abs(a.x() - b.x()) < 1e-6 && std::abs(a.y() - b.y()) < 1e-6; }

gui::ImageViewer* open_viewer() {
    for (QWidget* w : QApplication::topLevelWidgets()) {
        if (auto* v = qobject_cast<gui::ImageViewer*>(w); v != nullptr && v->isVisible()) return v;
    }
    return nullptr;
}

void close_viewers() {
    for (QWidget* w : QApplication::topLevelWidgets()) {
        if (qobject_cast<gui::ImageViewer*>(w) != nullptr) w->close();
    }
    QApplication::processEvents();
    QApplication::sendPostedEvents(nullptr, QEvent::DeferredDelete);
}

void test_zoom_and_pan() {
    gui::ImageView view;
    view.resize(400, 300);
    view.show();
    view.set_pixmap(picture());
    check::is_true(view.fitted() && std::abs(view.zoom() - 400.0 / 640.0) < 1e-9,
                   "opens fitted to the window");
    check::is_true(close_to(view.picture_rect().topLeft(), QPointF(0, 0)),
                   "fitted, it fills the window");

    const QPointF at(100, 80);
    const QPointF under = view.to_picture(at);
    send_wheel(view, at, 120);
    check::is_true(std::abs(view.zoom() - 400.0 / 640.0 * 1.25) < 1e-9 && !view.fitted(),
                   "a wheel notch zooms in by a step");
    check::is_true(close_to(view.to_picture(at), under), "about the pointer: the spot under it stays");
    send_wheel(view, QPointF(300, 200), 360);
    const QPointF before = view.to_picture(QPointF(200, 150));
    send_mouse(view, QEvent::MouseButtonPress, QPointF(200, 150));
    send_mouse(view, QEvent::MouseMove, QPointF(170, 130));
    send_mouse(view, QEvent::MouseButtonRelease, QPointF(170, 130));
    check::is_true(close_to(view.to_picture(QPointF(170, 130)), before),
                   "dragging moves the picture with the pointer");

    // Never off the window: a far drag stops at the picture's edge.
    send_mouse(view, QEvent::MouseButtonPress, QPointF(10, 10));
    send_mouse(view, QEvent::MouseMove, QPointF(5000, 5000));
    send_mouse(view, QEvent::MouseButtonRelease, QPointF(5000, 5000));
    check::is_true(close_to(view.picture_rect().topLeft(), QPointF(0, 0)),
                   "a drag stops at the picture's edge");

    // Zoomed out past the window, it sits centred.
    for (int i = 0; i < 10; ++i) send_wheel(view, QPointF(10, 10), -120);
    const QRectF r = view.picture_rect();
    check::is_true(r.width() < 400 && std::abs(r.center().x() - 200) < 1e-6 &&
                       std::abs(r.center().y() - 150) < 1e-6,
                   "smaller than the window, it is centred");

    send_mouse(view, QEvent::MouseButtonDblClick, QPointF(200, 150));
    check::is_true(view.fitted(), "double-click goes back to fitting");
    send_mouse(view, QEvent::MouseButtonDblClick, QPointF(50, 40));
    check::is_true(!view.fitted() && std::abs(view.zoom() - 1.0) < 1e-9,
                   "and again shows 100%");
}

void test_double_click_opens_it() {
    QWidget host;
    auto* box = new gui::PictureBox(QStringLiteral("Nothing"), &host);
    host.resize(400, 300);
    box->resize(400, 300);
    host.show();
    QApplication::processEvents();
    send_mouse(*box, QEvent::MouseButtonDblClick, QPointF(200, 150));
    check::is_true(open_viewer() == nullptr, "no picture, no viewer");
    box->set_picture(picture());
    send_mouse(*box, QEvent::MouseButtonDblClick, QPointF(200, 150));
    gui::ImageViewer* v = open_viewer();
    check::is_true(v != nullptr && v->view()->pixmap().width() == 640,
                   "double-clicking the received picture opens it, full size");
    close_viewers();
    check::is_true(open_viewer() == nullptr, "and closing deletes it");

    // A QRSS tile opens its picture file, not the reduced copy it shows.
    QTemporaryDir dir;
    check::is_true(picture().save(dir.filePath(QStringLiteral("t.png"))), "wrote a tile picture");
    gui::QrssTile tile;
    QJsonObject t;
    t.insert(QStringLiteral("image"), QStringLiteral("t.png"));
    t.insert(QStringLiteral("image_rev"), 1);
    t.insert(QStringLiteral("picture_id"), QStringLiteral("abc"));
    tile.update_from(t, dir.path());
    tile.show();
    send_mouse(tile, QEvent::MouseButtonDblClick, QPointF(20, 20));
    v = open_viewer();
    check::is_true(v != nullptr && v->view()->pixmap().size() == QSize(640, 480),
                   "double-clicking a QRSS tile opens its picture at full size");
    close_viewers();
}

}  // namespace

int main(int argc, char** argv) {
    check::report_crashes_instead_of_prompting();
    qputenv("QT_QPA_PLATFORM", "offscreen");
    const QApplication app(argc, argv);

    test_zoom_and_pan();
    test_double_click_opens_it();

    return check::report("image viewer");
}

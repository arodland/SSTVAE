#include "compose_tools.hpp"

#include <QCoreApplication>
#include <QDir>
#include <QFont>
#include <QPainter>
#include <QPainterPath>
#include <QPen>
#include <QPixmap>
#include <QPolygonF>
#include <QStyle>
#include <QWidget>

#include <algorithm>
#include <cmath>
#include <functional>
#include <vector>

namespace sstvae::gui::compose {

const char* const IMAGE_FILTER =
    "Images (*.png *.jpg *.jpeg *.webp *.bmp *.gif *.heic *.heif *.avif *.jp2 *.j2k *.jpf);;All files (*)";

// --- the tool palette's icons -------------------------------------------
//
// Hand-drawn rather than shipped as asset files: four tiny glyphs are
// cheaper to draw once than to source, license and keep in step with
// the platform's icon theme (which several of this project's target
// platforms do not reliably have one of). `PM_LargeIconSize` and
// `devicePixelRatioF`, so a HiDPI screen gets a crisp glyph.

namespace {

QIcon draw_tool_icon(const QWidget* metrics,
                     const std::function<void(QPainter&, int, const QColor&)>& paint) {
    const int size = metrics->style()->pixelMetric(QStyle::PM_LargeIconSize);
    const qreal dpr = metrics->devicePixelRatioF();
    QPixmap pixmap(static_cast<int>(std::lround(size * dpr)),
                   static_cast<int>(std::lround(size * dpr)));
    pixmap.setDevicePixelRatio(dpr);
    pixmap.fill(Qt::transparent);
    QPainter painter(&pixmap);
    painter.setRenderHint(QPainter::Antialiasing, true);
    // Palette, not a literal color: a hand-drawn icon that ignored the
    // palette would go invisible on a dark theme.
    paint(painter, size, metrics->palette().color(QPalette::WindowText));
    return QIcon(pixmap);
}

// The "picture" glyph shared by the image-inset and last-received tools
// -- a frame, a sun and a mountain range, the generic photo icon every
// platform already uses this shape for.
void draw_picture_glyph(QPainter& p, int size, const QColor& ink) {
    const QRectF frame(size * 0.12, size * 0.12, size * 0.76, size * 0.76);
    p.setPen(QPen(ink, std::max(1.0, size / 16.0)));
    p.setBrush(Qt::NoBrush);
    p.drawRoundedRect(frame, size * 0.06, size * 0.06);

    p.setPen(Qt::NoPen);
    p.setBrush(ink);
    p.drawEllipse(QPointF(frame.left() + frame.width() * 0.3,
                          frame.top() + frame.height() * 0.32),
                 size * 0.07, size * 0.07);

    QPainterPath mountains;
    mountains.moveTo(frame.left() + frame.width() * 0.06, frame.bottom() - frame.height() * 0.1);
    mountains.lineTo(frame.left() + frame.width() * 0.38, frame.bottom() - frame.height() * 0.55);
    mountains.lineTo(frame.left() + frame.width() * 0.58, frame.bottom() - frame.height() * 0.3);
    mountains.lineTo(frame.left() + frame.width() * 0.78, frame.bottom() - frame.height() * 0.52);
    mountains.lineTo(frame.right() - frame.width() * 0.06, frame.bottom() - frame.height() * 0.1);
    mountains.closeSubpath();
    p.drawPath(mountains);
}

// Line glyphs on a 24-unit grid, as the mockup drew them: each path is
// a list of polylines.
QIcon line_icon(const QWidget* metrics, int size,
                const std::vector<std::vector<QPointF>>& lines, const QColor& hover = {}) {
    const qreal dpr = metrics->devicePixelRatioF();
    const auto draw = [&](const QColor& ink) {
        QPixmap pixmap(static_cast<int>(std::lround(size * dpr)),
                       static_cast<int>(std::lround(size * dpr)));
        pixmap.setDevicePixelRatio(dpr);
        pixmap.fill(Qt::transparent);
        QPainter p(&pixmap);
        p.setRenderHint(QPainter::Antialiasing, true);
        const double k = size / 24.0;
        p.setPen(QPen(ink, 1.7 * k, Qt::SolidLine, Qt::RoundCap, Qt::RoundJoin));
        p.setBrush(Qt::NoBrush);
        for (const auto& line : lines) {
            QPolygonF poly;
            for (const QPointF& pt : line) poly << pt * k;
            p.drawPolyline(poly);
        }
        return pixmap;
    };
    QIcon icon(draw(metrics->palette().color(QPalette::WindowText)));
    if (hover.isValid()) icon.addPixmap(draw(hover), QIcon::Active);
    return icon;
}

}  // namespace

QIcon new_send_icon(const QWidget* metrics, int size) {
    return line_icon(metrics, size,
                     {{{6, 3}, {14, 3}, {18, 7}, {18, 21}, {6, 21}, {6, 3}},
                      {{14, 3}, {14, 7}, {18, 7}},
                      {{12, 11}, {12, 17}},
                      {{9, 14}, {15, 14}}});
}

QIcon open_file_icon(const QWidget* metrics, int size) {
    return line_icon(metrics, size,
                     {{{3, 7}, {3, 5}, {9, 5}, {11, 7}, {19, 7}, {19, 10}},
                      {{3, 7}, {3, 19}, {18, 19}, {21, 10}, {7, 10}, {4, 19}}});
}

QIcon edit_icon(const QWidget* metrics, int size) {
    return line_icon(metrics, size,
                     {{{4, 20}, {5, 16}, {16, 5}, {19, 8}, {8, 19}, {4, 20}},
                      {{14, 7}, {17, 10}}});
}

QIcon delete_icon(const QWidget* metrics, int size) {
    // Red while the pointer is over it: the one that cannot be taken back.
    return line_icon(metrics, size,
                     {{{4, 7}, {20, 7}},
                      {{9, 7}, {9, 4}, {15, 4}, {15, 7}},
                      {{6, 7}, {7, 20}, {17, 20}, {18, 7}},
                      {{10, 11}, {10, 17}},
                      {{14, 11}, {14, 17}}},
                     QColor(192, 53, 43));
}

QIcon text_tool_icon(const QWidget* metrics) {
    return draw_tool_icon(metrics, [](QPainter& p, int size, const QColor& ink) {
        QFont font = p.font();
        font.setBold(true);
        font.setPixelSize(static_cast<int>(size * 0.72));
        p.setFont(font);
        p.setPen(ink);
        p.drawText(QRect(0, 0, size, size), Qt::AlignCenter, QStringLiteral("T"));
    });
}

QIcon image_tool_icon(const QWidget* metrics) {
    return draw_tool_icon(metrics, [](QPainter& p, int size, const QColor& ink) {
        draw_picture_glyph(p, size, ink);
    });
}

QIcon last_rx_tool_icon(const QWidget* metrics) {
    return draw_tool_icon(metrics, [](QPainter& p, int size, const QColor& ink) {
        draw_picture_glyph(p, size, ink);
        // A small curved "history" arrow badge over the bottom-right
        // corner, so the glyph reads as "that picture again" rather
        // than a second, redundant photo icon.
        const QRectF badge(size * 0.48, size * 0.48, size * 0.46, size * 0.46);
        QPainterPath arrow;
        arrow.arcMoveTo(badge, 30);
        arrow.arcTo(badge, 30, 260);
        p.setPen(QPen(ink, std::max(1.0, size / 14.0)));
        p.setBrush(Qt::NoBrush);
        p.drawPath(arrow);
        const QPointF tip = arrow.currentPosition();
        QPolygonF head;
        head << tip << QPointF(tip.x() - size * 0.09, tip.y() - size * 0.02)
             << QPointF(tip.x() - size * 0.01, tip.y() + size * 0.09);
        p.setPen(Qt::NoPen);
        p.setBrush(ink);
        p.drawPolygon(head);
    });
}

QIcon rect_tool_icon(const QWidget* metrics) {
    return draw_tool_icon(metrics, [](QPainter& p, int size, const QColor& ink) {
        const QRectF box(size * 0.15, size * 0.26, size * 0.7, size * 0.48);
        p.setPen(Qt::NoPen);
        p.setBrush(QColor(ink.red(), ink.green(), ink.blue(), 90));
        p.drawRect(box);
        p.setBrush(Qt::NoBrush);
        p.setPen(QPen(ink, std::max(1.0, size / 14.0)));
        p.drawRect(box);
    });
}


std::filesystem::path builtin_templates_dir() {
    // Copied at build time (`sstvae_copy_builtin_templates` in
    // native/CMakeLists.txt) from `sstvae/overlay/templates/`, the one
    // place the three ship from. Where an *installed* app keeps its
    // data is a platform convention, tried first; beside the executable
    // is the build tree, the tests, sstvae-gui-shot and the Windows
    // package, and is the fallback everywhere.
    //
    // macOS: a bundle's Contents/Resources -- "beside the executable" is
    // Contents/MacOS there, and codesign refuses data in it (see the
    // CMake function). Linux: <prefix>/share/sstvae/templates, resolved
    // from the executable's own prefix, so one rule covers a distro
    // package at /usr, a hand install at /usr/local or /opt, and the
    // AppDir -- a packager expects /usr/share/sstvae, not a data
    // directory under /usr/bin.
    const QString beside = QCoreApplication::applicationDirPath();
#if defined(Q_OS_MACOS)
    const QString installed = QDir::cleanPath(beside + QStringLiteral("/../Resources/templates"));
#elif defined(Q_OS_UNIX)
    const QString installed =
        QDir::cleanPath(beside + QStringLiteral("/../share/sstvae/templates"));
#else
    const QString installed;
#endif
    if (!installed.isEmpty() && QDir(installed).exists()) return installed.toStdString();
    return (beside + QStringLiteral("/templates")).toStdString();
}

}  // namespace sstvae::gui::compose

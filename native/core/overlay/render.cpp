#include "overlay/render.hpp"

#include <QColor>
#include <QFont>
#include <QFontDatabase>
#include <QFontMetricsF>
#include <QImage>
#include <QPainter>
#include <QPainterPath>
#include <QSize>
#include <QString>
#include <QStringList>
#include <QTransform>

#include <algorithm>
#include <cmath>
#include <map>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace sstvae::overlay {

namespace {

// ---------------------------------------------------------------------
// Pictures in and out

QImage to_qimage(const images::Picture& p) {
    if (p.empty()) return QImage();
    // The Picture's rows are tightly packed; QImage would otherwise
    // assume a 4-byte-aligned stride and shear the picture. copy()
    // because the wrapper does not own the bytes.
    const QImage view(p.rgb.data(), p.width, p.height, p.width * 3,
                      QImage::Format_RGB888);
    return view.copy().convertToFormat(QImage::Format_ARGB32_Premultiplied);
}

images::Picture from_qimage(const QImage& image) {
    const QImage rgb = image.convertToFormat(QImage::Format_RGB888);
    images::Picture out(rgb.width(), rgb.height());
    for (int y = 0; y < rgb.height(); ++y) {
        std::copy_n(rgb.constScanLine(y), static_cast<std::size_t>(rgb.width()) * 3,
                    out.rgb.data() + static_cast<std::size_t>(y) * rgb.width() * 3);
    }
    return out;
}

QColor color_of(const std::string& text, const QColor& fallback) {
    const QColor c(QString::fromStdString(text));
    return c.isValid() ? c : fallback;
}

// ---------------------------------------------------------------------
// Fonts
//
// The document names a font by *path*, because that is what PIL takes
// and what a saved template carries. Qt matches by family, so a path
// has to be registered with the font database first and the family it
// contributed read back out. Cached: registering is file I/O, and a
// text item is re-measured on every drag in the editor.

QString family_for(const std::string& path) {
    if (path.empty()) return QString();
    static std::mutex mutex;
    static std::map<std::string, QString> cache;
    const std::lock_guard<std::mutex> lock(mutex);
    const auto it = cache.find(path);
    if (it != cache.end()) return it->second;

    QString family;
    const int id = QFontDatabase::addApplicationFont(QString::fromStdString(path));
    if (id != -1) {
        const QStringList families = QFontDatabase::applicationFontFamilies(id);
        if (!families.isEmpty()) family = families.front();
    }
    cache.emplace(path, family);
    return family;
}

QFont font_for(const TextItem& item, int size_px) {
    QFont font;
    const QString family = family_for(item.font);
    if (!family.isEmpty()) font.setFamily(family);
    // setPixelSize, not setPointSize: the document sizes text as a
    // fraction of canvas height, so the answer must not depend on the
    // DPI of whatever screen happens to be attached.
    font.setPixelSize(std::max(1, size_px));
    return font;
}

// ---------------------------------------------------------------------
// Text layout
//
// PIL's two-letter anchor, kept because it is what the document says
// and what the reference's saved files contain. First letter is
// horizontal (l/m/r), second vertical: 'a' ascender, 't' top, 'm'
// middle, 's' baseline, 'b' bottom, 'd' descender.

struct TextLayout {
    QStringList lines;
    double line_height = 0.0;  // baseline to baseline
    double width = 0.0;
    double height = 0.0;
    double ascent = 0.0;
    double left = 0.0;  // top-left of the block, anchor applied
    double top = 0.0;
};

TextLayout layout_text(const TextItem& item, const QFont& font, int size_px,
                       double x, double y) {
    const QFontMetricsF fm(font);
    TextLayout out;
    out.lines = QString::fromStdString(item.text).split(QLatin1Char('\n'));
    if (out.lines.isEmpty()) out.lines << QString();

    const double spacing = std::max(0.0, item.line_spacing * size_px);
    out.ascent = fm.ascent();
    out.line_height = fm.height() + spacing;
    for (const QString& line : out.lines)
        out.width = std::max(out.width, fm.horizontalAdvance(line));
    out.height = fm.height() +
                 out.line_height * static_cast<double>(out.lines.size() - 1);

    const char h = item.anchor.empty() ? 'l' : item.anchor[0];
    const char v = item.anchor.size() > 1 ? item.anchor[1] : 'a';

    out.left = x;
    if (h == 'm') out.left = x - out.width / 2.0;
    else if (h == 'r') out.left = x - out.width;

    switch (v) {
        case 'm': out.top = y - out.height / 2.0; break;
        case 's': out.top = y - out.ascent; break;
        case 'b':
        case 'd': out.top = y - out.height; break;
        default: out.top = y; break;  // 'a' ascender, 't' top
    }
    return out;
}

// One path for the whole block, so the stroke is drawn under *all* the
// glyphs before any of them is filled. Stroking and filling line by
// line would let a descender's outline cut across the line below it.
QPainterPath text_path(const TextItem& item, const QFont& font,
                       const TextLayout& layout) {
    const QFontMetricsF fm(font);
    QPainterPath path;
    for (int i = 0; i < layout.lines.size(); ++i) {
        const QString& line = layout.lines[i];
        if (line.isEmpty()) continue;
        double x = layout.left;
        if (item.align == "center")
            x += (layout.width - fm.horizontalAdvance(line)) / 2.0;
        else if (item.align == "right")
            x += layout.width - fm.horizontalAdvance(line);
        const double baseline =
            layout.top + layout.ascent + layout.line_height * i;
        path.addText(QPointF(x, baseline), font, line);
    }
    return path;
}

void draw_text(QPainter& painter, const TextItem& item, int canvas_w,
               int canvas_h) {
    if (item.text.empty()) return;
    const int size_px = static_cast<int>(std::lround(item.size * canvas_h));
    const QFont font = font_for(item, size_px);
    const double stroke =
        std::max(0.0, item.stroke_width * item.size * canvas_h);
    const double x = std::lround(item.x * canvas_w);
    const double y = std::lround(item.y * canvas_h);

    painter.save();
    if (item.rotation != 0.0) {
        // About the anchor point, which is the point the document
        // actually pins. (The reference rotates a padded layer and
        // composites it at the anchor, which is close but not the same;
        // this is the version an editor can show a handle for.) Negated
        // because the document's angle is counter-clockwise, as PIL's
        // is, and QTransform::rotate turns the other way.
        painter.translate(x, y);
        painter.rotate(-item.rotation);
        painter.translate(-x, -y);
    }

    const TextLayout layout = layout_text(item, font, size_px, x, y);
    const QPainterPath path = text_path(item, font, layout);
    if (stroke > 0.0) {
        QPen pen(color_of(item.stroke_color, Qt::black));
        // PIL's stroke_width is a radius, drawn outside the glyph; a
        // Qt pen straddles the path, so half of a width-2w pen lands
        // outside. Same visual weight rather than a coincidence.
        pen.setWidthF(stroke * 2.0);
        pen.setJoinStyle(Qt::RoundJoin);
        painter.strokePath(path, pen);
    }
    painter.fillPath(path, color_of(item.color, Qt::white));
    painter.restore();
}

// ---------------------------------------------------------------------
// Image insets

// A file-backed inset, decoded once.
//
// **Keyed on the path and never invalidated**, which is the whole
// contract: a document names an inset by path, so the only thing that
// changes which pixels an item shows is the path changing. Watching the
// file for content changes is deliberately not attempted -- it would
// mean a stat on every call, on a path that runs on every mouse move,
// to catch a case (the operator overwriting a file the composition
// already refers to, in place, mid-session) that costs nothing to fix
// by re-adding the inset.
//
// The decode used to happen on every call, and `item_bbox` is a caller:
// `OverlayEditor::hit_test` runs it per item on **every mouse move over
// the canvas**, and `paintEvent` runs it again. So merely moving the
// pointer across the composer re-read and re-decoded every inset from
// disk.
//
// A failed load is cached too. The alternative is retrying a missing
// file at mouse-move rate, which is the same pathology with syscalls
// instead of a decode.
//
// Returned by value: `QImage` is copy-on-write, so this is a refcount
// bump, and it means no caller holds a reference into a cache another
// thread could evict.
QImage cached_file_image(const std::string& path) {
    // Small and FIFO-bounded rather than unbounded. A source photograph
    // is not small -- 4000x3000 is 48 MB as ARGB32 -- and this is a
    // process-lifetime cache in an application that runs for days, so
    // an operator cycling through a folder of pictures must not be able
    // to grow it without limit. Eight is far more than any composition
    // uses at once.
    constexpr std::size_t CAPACITY = 8;
    static std::mutex mutex;
    static std::vector<std::pair<std::string, QImage>> cache;

    const std::lock_guard<std::mutex> lock(mutex);
    for (const auto& entry : cache) {
        if (entry.first == path) return entry.second;
    }

    QImage loaded;
    if (loaded.load(QString::fromStdString(path))) {
        loaded = loaded.convertToFormat(QImage::Format_ARGB32_Premultiplied);
    } else {
        loaded = QImage();
    }
    if (cache.size() >= CAPACITY) cache.erase(cache.begin());
    cache.emplace_back(path, loaded);
    return loaded;
}

QImage resolve_source(const ImageItem& item, const images::Picture* last_rx) {
    if (item.source == SOURCE_LAST_RX) {
        return last_rx != nullptr ? to_qimage(*last_rx) : QImage();
    }
    if (item.source.empty()) return QImage();
    return cached_file_image(item.source);
}

// The source's dimensions, without materializing it.
//
// `item_bbox` wants nothing from the source but its aspect ratio, and
// it is the call that runs on every mouse move. For a "last_rx" item
// the answer is already on the `Picture`, so going through
// `resolve_source` converted a 640x480 reception into a 1.2 MB ARGB32
// `QImage` in order to read two integers back off it.
QSize source_size(const ImageItem& item, const images::Picture* last_rx) {
    if (item.source == SOURCE_LAST_RX) {
        if (last_rx == nullptr || last_rx->empty()) return QSize();
        return QSize(last_rx->width, last_rx->height);
    }
    if (item.source.empty()) return QSize();
    return cached_file_image(item.source).size();
}

// The inset at its drawn size, border included; the anchor is applied
// by the caller, which is also what item_bbox does.
QSize inset_size(const ImageItem& item, int canvas_w, double aspect) {
    const int w = std::max(1, static_cast<int>(std::lround(item.width * canvas_w)));
    const int h = std::max(1, static_cast<int>(std::lround(w * aspect)));
    const int border =
        std::max(0, static_cast<int>(std::lround(item.border * canvas_w)));
    return QSize(w + 2 * border, h + 2 * border);
}

void apply_anchor(const std::string& anchor, int w, int h, int& x, int& y) {
    const char horizontal = anchor.empty() ? 'l' : anchor[0];
    const char vertical = anchor.size() > 1 ? anchor[1] : 'a';
    if (horizontal == 'm') x -= w / 2;
    else if (horizontal == 'r') x -= w;
    if (vertical == 'm') y -= h / 2;
    else if (vertical == 'b' || vertical == 'd') y -= h;
}

void draw_image(QPainter& painter, const ImageItem& item, int canvas_w,
                int canvas_h, const images::Picture* last_rx) {
    const QImage src = resolve_source(item, last_rx);
    if (src.isNull()) return;

    const int border =
        std::max(0, static_cast<int>(std::lround(item.border * canvas_w)));
    const int target_w =
        std::max(1, static_cast<int>(std::lround(item.width * canvas_w)));
    const int target_h = std::max(
        1, static_cast<int>(std::lround(static_cast<double>(target_w) *
                                        src.height() / src.width())));

    QImage inset(target_w + 2 * border, target_h + 2 * border,
                 QImage::Format_ARGB32_Premultiplied);
    inset.fill(border > 0 ? color_of(item.border_color, Qt::white)
                          : QColor(Qt::transparent));
    {
        QPainter inset_painter(&inset);
        inset_painter.setRenderHint(QPainter::SmoothPixmapTransform, true);
        inset_painter.drawImage(QRect(border, border, target_w, target_h), src);
    }

    int x = static_cast<int>(std::lround(item.x * canvas_w));
    int y = static_cast<int>(std::lround(item.y * canvas_h));
    apply_anchor(item.anchor, inset.width(), inset.height(), x, y);

    painter.save();
    painter.setOpacity(std::clamp(item.opacity, 0.0, 1.0));
    if (item.rotation != 0.0) {
        const QPointF centre(x + inset.width() / 2.0, y + inset.height() / 2.0);
        painter.translate(centre);
        painter.rotate(-item.rotation);
        painter.translate(-centre);
    }
    painter.drawImage(QPoint(x, y), inset);
    painter.restore();
}

}  // namespace

Bbox item_bbox(int canvas_w, int canvas_h, const Item& item,
               const images::Picture* last_rx) {
    if (const TextItem* text = std::get_if<TextItem>(&item)) {
        const int size_px =
            static_cast<int>(std::lround(text->size * canvas_h));
        const QFont font = font_for(*text, size_px);
        const double stroke =
            std::max(0.0, text->stroke_width * text->size * canvas_h);
        const double x = std::lround(text->x * canvas_w);
        const double y = std::lround(text->y * canvas_h);
        // A space when empty, so an item being typed into still has a
        // handle to select rather than collapsing to nothing.
        TextItem measured = *text;
        if (measured.text.empty()) measured.text = " ";
        const TextLayout layout = layout_text(measured, font, size_px, x, y);
        return Bbox{static_cast<int>(std::lround(layout.left - stroke)),
                    static_cast<int>(std::lround(layout.top - stroke)),
                    std::max(1, static_cast<int>(
                                    std::lround(layout.width + 2 * stroke))),
                    std::max(1, static_cast<int>(
                                    std::lround(layout.height + 2 * stroke)))};
    }

    const ImageItem& image = std::get<ImageItem>(item);
    // Dimensions only -- see `source_size`. This is the mouse-move path.
    const QSize src = source_size(image, last_rx);
    const double aspect =
        src.isEmpty() ? 0.75
                      : static_cast<double>(src.height()) / src.width();
    const QSize size = inset_size(image, canvas_w, aspect);
    int x = static_cast<int>(std::lround(image.x * canvas_w));
    int y = static_cast<int>(std::lround(image.y * canvas_h));
    apply_anchor(image.anchor, size.width(), size.height(), x, y);
    return Bbox{x, y, size.width(), size.height()};
}

images::Picture render(const images::Picture& base, const Doc& doc,
                       const images::Picture* last_rx) {
    if (base.empty()) return base;
    QImage canvas = to_qimage(base);
    {
        QPainter painter(&canvas);
        painter.setRenderHint(QPainter::Antialiasing, true);
        painter.setRenderHint(QPainter::TextAntialiasing, true);
        painter.setRenderHint(QPainter::SmoothPixmapTransform, true);
        for (const Item& item : doc.items) {
            if (const TextItem* text = std::get_if<TextItem>(&item)) {
                draw_text(painter, *text, base.width, base.height);
            } else if (const ImageItem* image = std::get_if<ImageItem>(&item)) {
                draw_image(painter, *image, base.width, base.height, last_rx);
            }
        }
    }
    return from_qimage(canvas);
}

}  // namespace sstvae::overlay

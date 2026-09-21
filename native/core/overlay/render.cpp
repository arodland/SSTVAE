#include "overlay/render.hpp"

#include <QBrush>
#include <QColor>
#include <QFont>
#include <QFontDatabase>
#include <QFontMetricsF>
#include <QImage>
#include <QLinearGradient>
#include <QPainter>
#include <QPainterPath>
#include <QPen>
#include <QRadialGradient>
#include <QRectF>
#include <QSize>
#include <QString>
#include <QStringList>
#include <QTransform>

#include <algorithm>
#include <cmath>
#include <map>
#include <mutex>
#include <numbers>
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

// The document's four generic keywords. Qt's fontconfig backend
// understands them as family names, but the other platforms' font
// databases do not, so the style hint rides along: it is what makes
// "monospace" mean a fixed-pitch face on a machine where no family is
// called that. Anything else is a real family name, passed through.
void apply_family_request(QFont& font, const std::string& request) {
    const QString name = QString::fromStdString(request);
    struct Generic {
        const char* keyword;
        QFont::StyleHint hint;
    };
    static constexpr Generic GENERICS[] = {
        {"sans-serif", QFont::SansSerif},
        {"serif", QFont::Serif},
        {"monospace", QFont::Monospace},
        {"cursive", QFont::Cursive},
    };
    font.setFamily(name);
    for (const Generic& g : GENERICS) {
        if (name.compare(QLatin1String(g.keyword), Qt::CaseInsensitive) == 0) {
            font.setStyleHint(g.hint);
            return;
        }
    }
}

QFont font_for(const TextItem& item, int size_px) {
    QFont font;
    // `font` (a path) wins over `font_family`: a template that ships its
    // own face is naming the exact file it needs. The family request is
    // the fallback when there is no path *or* the path yielded no family,
    // so an unreadable file degrades to the operator's choice of face
    // rather than to the default one.
    const QString family = family_for(item.font);
    if (!family.isEmpty()) font.setFamily(family);
    else if (!item.font_family.empty()) apply_family_request(font, item.font_family);
    // Qt synthesizes a weight or a slant the face does not have, so
    // these always do something visible.
    font.setBold(item.bold);
    font.setItalic(item.italic);
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

// Where a line's underline sits, from the font's own metrics. Shared by
// the drawing and by `item_bbox`, so a selection handle cannot stop
// short of a line the renderer draws.
//
// Never thinner than a pixel: `lineWidth()` is 0 or fractional for some
// faces at small sizes, and an underline that vanished there would make
// the toggle look broken.
QRectF underline_rect(const QFontMetricsF& fm, double x, double baseline,
                      double width) {
    return QRectF(x, baseline + fm.underlinePos(), width,
                  std::max(1.0, fm.lineWidth()));
}

// The whole block as two paths, both stroked before either is filled,
// so the stroke lies under *all* the glyphs. Stroking and filling line
// by line would let a descender's outline cut across the line below.
//
// **The underline is its own path, not a rect added to the glyphs'.**
// `QPainterPath::addText` adds outlines only -- a font's underline is a
// decoration Qt draws separately, so `QFont::setUnderline` renders
// nothing through this path -- and a rect sharing the glyphs' path
// fights them over the fill rule: under odd-even a descender crossing
// it is punched out, under winding a contour running the other way
// cancels it. Two paths is what a union looks like without asking Qt to
// compute one.
struct TextShape {
    QPainterPath glyphs;
    QPainterPath underline;
};

TextShape text_shape(const TextItem& item, const QFont& font,
                     const TextLayout& layout) {
    const QFontMetricsF fm(font);
    TextShape shape;
    for (int i = 0; i < layout.lines.size(); ++i) {
        const QString& line = layout.lines[i];
        if (line.isEmpty()) continue;
        const double advance = fm.horizontalAdvance(line);
        double x = layout.left;
        if (item.align == "center") x += (layout.width - advance) / 2.0;
        else if (item.align == "right") x += layout.width - advance;
        const double baseline =
            layout.top + layout.ascent + layout.line_height * i;
        shape.glyphs.addText(QPointF(x, baseline), font, line);
        if (item.underline) {
            shape.underline.addRect(underline_rect(fm, x, baseline, advance));
        }
    }
    return shape;
}

// Declared here, defined with the other gradient code under "Rectangles":
// text and rects fill through the same geometry.
QBrush gradient_brush(const QRectF& box, const std::string& c1, const std::string& c2,
                      double angle_deg, const std::string& shape);

// The glyph fill, in `RectItem`'s terms. A solid fill -- every document
// written before these fields existed -- takes exactly the path it
// always did, and never reads the gradient fields.
//
// **An unrecognised kind draws solid, not nothing.** That is the
// opposite of a rect's rule (`rect_brush` treats an unknown kind as
// "none"), and deliberately: a caption that vanishes on a build that
// does not know some later fill kind is worse than one drawn flat, and
// the whole point of text is to be read.
//
// A gradient spans the layout box rather than the ink, so it does not
// shift as the text is edited, and it is built in the painter's current
// space -- already rotated for the item -- so it turns with the text,
// which is what "painted into the item's unrotated layer" means for a
// rect.
QBrush text_fill_brush(const TextItem& item, const TextLayout& layout) {
    if (item.fill_kind == "none") return QBrush(Qt::NoBrush);
    if (item.fill_kind == "gradient") {
        return gradient_brush(QRectF(layout.left, layout.top, layout.width, layout.height),
                              item.color, item.fill_color2, item.fill_angle,
                              item.fill_gradient);
    }
    return QBrush(color_of(item.color, Qt::white));
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

    // Computed before rotating: the layout does not depend on the
    // painter's transform, and the block's own centre is what the
    // rotation below needs.
    const TextLayout layout = layout_text(item, font, size_px, x, y);

    painter.save();
    if (item.rotation != 0.0) {
        // About the text block's own centre, matching `draw_rect`/
        // `draw_image` -- not the raw anchor point this used to pivot
        // on, which for the common "top-left" anchor swung the whole
        // block out from under the selection box instead of turning it
        // in place (found via the editor's own rotate handle, which
        // made the mismatch obvious for the first time). Negated
        // because the document's angle is counter-clockwise, as PIL's
        // is, and QTransform::rotate turns the other way.
        const QPointF centre(layout.left + layout.width / 2.0,
                             layout.top + layout.height / 2.0);
        painter.translate(centre);
        painter.rotate(-item.rotation);
        painter.translate(-centre);
    }

    const TextShape shape = text_shape(item, font, layout);
    if (stroke > 0.0) {
        QPen pen(color_of(item.stroke_color, Qt::black));
        // PIL's stroke_width is a radius, drawn outside the glyph; a
        // Qt pen straddles the path, so half of a width-2w pen lands
        // outside. Same visual weight rather than a coincidence.
        pen.setWidthF(stroke * 2.0);
        pen.setJoinStyle(Qt::RoundJoin);
        painter.strokePath(shape.glyphs, pen);
        if (item.underline) painter.strokePath(shape.underline, pen);
    }
    const QBrush fill = text_fill_brush(item, layout);
    if (fill.style() != Qt::NoBrush) {
        painter.fillPath(shape.glyphs, fill);
        if (item.underline) painter.fillPath(shape.underline, fill);
    }
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

// ---------------------------------------------------------------------
// Rectangles

// A two-colour gradient over `box`, for rects and text alike.
//
// **Linear** runs along `angle_deg`, counter-clockwise to match the
// items' `rotation`: 0 runs left to right, 90 bottom to top. The extent
// formula matches `sstvae/overlay/render.py`'s `_gradient_layer` so the
// two implementations agree on where each colour lands, even though Qt
// interpolates continuously where PIL's numpy version is per-pixel.
//
// **Radial** is centred on the box and reaches `c2` at its corners -- a
// radius of half the diagonal, so the whole box is inside the ramp and
// no corner is left clamped at the far colour. It has no angle. Anything
// but "radial" is linear: that is how a build that predates some later
// shape degrades, and how an older build reads a radial one.
QBrush gradient_brush(const QRectF& box, const std::string& c1, const std::string& c2,
                      double angle_deg, const std::string& shape) {
    const QColor from = color_of(c1, Qt::white);
    const QColor to = color_of(c2, Qt::black);
    const QPointF centre = box.center();
    if (shape == "radial") {
        const double radius = std::max(1.0, std::hypot(box.width(), box.height()) / 2.0);
        QRadialGradient grad(centre, radius);
        grad.setColorAt(0.0, from);
        grad.setColorAt(1.0, to);
        return QBrush(grad);
    }
    const double theta = angle_deg * std::numbers::pi / 180.0;
    const double ux = std::cos(theta);
    const double uy = -std::sin(theta);
    const double extent = (std::abs(ux) * box.width() + std::abs(uy) * box.height()) / 2.0;
    const double ext = extent > 0.0 ? extent : 1.0;
    QLinearGradient grad(QPointF(centre.x() - ux * ext, centre.y() - uy * ext),
                         QPointF(centre.x() + ux * ext, centre.y() + uy * ext));
    grad.setColorAt(0.0, from);
    grad.setColorAt(1.0, to);
    return QBrush(grad);
}

QBrush rect_brush(const std::string& kind, const std::string& color,
                  const std::string& color2, double angle, const std::string& shape,
                  double w, double h) {
    if (kind == "solid") return QBrush(color_of(color, Qt::white));
    if (kind == "gradient") {
        return gradient_brush(QRectF(0, 0, w, h), color, color2, angle, shape);
    }
    return QBrush(Qt::NoBrush);
}

void draw_rect(QPainter& painter, const RectItem& item, int canvas_w, int canvas_h) {
    if (item.fill_kind == "none" && item.stroke_kind == "none") return;

    const int iw = std::max(1, static_cast<int>(std::lround(item.width * canvas_w)));
    const int ih = std::max(1, static_cast<int>(std::lround(item.height * canvas_h)));
    const double sw = std::max(0.0, item.stroke_width * canvas_w);

    // Painted into its own unrotated layer and then rotated as a whole
    // by the painter transform below, exactly as `draw_image` does --
    // which is also why `item_bbox` below reports the *unrotated*
    // extent: the two must describe the same thing a handle sits on.
    QImage layer(iw, ih, QImage::Format_ARGB32_Premultiplied);
    layer.fill(Qt::transparent);
    {
        QPainter lp(&layer);
        if (item.fill_kind != "none") {
            lp.fillRect(QRectF(0, 0, iw, ih),
                        rect_brush(item.fill_kind, item.fill_color, item.fill_color2,
                                   item.fill_angle, item.fill_gradient, iw, ih));
        }
        if (item.stroke_kind != "none" && sw > 0.0) {
            QPen pen(rect_brush(item.stroke_kind, item.stroke_color, item.stroke_color2,
                                item.stroke_angle, item.stroke_gradient, iw, ih),
                     sw);
            pen.setJoinStyle(Qt::MiterJoin);
            lp.setPen(pen);
            lp.setBrush(Qt::NoBrush);
            // A Qt pen straddles the path -- inset by half its width so
            // the stroke's outer edge lands on the item's own declared
            // bounds instead of spilling `sw/2` past them.
            lp.drawRect(QRectF(sw / 2.0, sw / 2.0, iw - sw, ih - sw));
        }
    }

    int x = static_cast<int>(std::lround(item.x * canvas_w));
    int y = static_cast<int>(std::lround(item.y * canvas_h));
    apply_anchor(item.anchor, layer.width(), layer.height(), x, y);

    painter.save();
    if (item.rotation != 0.0) {
        const QPointF centre(x + layer.width() / 2.0, y + layer.height() / 2.0);
        painter.translate(centre);
        painter.rotate(-item.rotation);
        painter.translate(-centre);
    }
    painter.drawImage(QPoint(x, y), layer);
    painter.restore();
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
        // An underline can sit below the font's descent, and the handle
        // must not clip a line the renderer draws. The last line's is the
        // lowest, and the same `underline_rect` placed it.
        double bottom = layout.top + layout.height;
        if (text->underline) {
            const double baseline = layout.top + layout.ascent +
                                    layout.line_height * (layout.lines.size() - 1);
            bottom = std::max(bottom, underline_rect(QFontMetricsF(font), layout.left,
                                                     baseline, layout.width)
                                          .bottom());
        }
        return Bbox{static_cast<int>(std::lround(layout.left - stroke)),
                    static_cast<int>(std::lround(layout.top - stroke)),
                    std::max(1, static_cast<int>(
                                    std::lround(layout.width + 2 * stroke))),
                    std::max(1, static_cast<int>(std::lround(
                                    bottom - layout.top + 2 * stroke)))};
    }

    if (const RectItem* rect = std::get_if<RectItem>(&item)) {
        // Unrotated dimensions, like the `ImageItem` branch below -- an
        // existing simplification kept for consistency, not a gap
        // specific to rectangles: see `draw_rect`, which paints into an
        // unrotated layer and rotates it with a painter transform, so
        // the handle and the paint describe the same unrotated box.
        const int iw = std::max(1, static_cast<int>(std::lround(rect->width * canvas_w)));
        const int ih = std::max(1, static_cast<int>(std::lround(rect->height * canvas_h)));
        int x = static_cast<int>(std::lround(rect->x * canvas_w));
        int y = static_cast<int>(std::lround(rect->y * canvas_h));
        apply_anchor(rect->anchor, iw, ih, x, y);
        return Bbox{x, y, iw, ih};
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
            } else if (const RectItem* rect = std::get_if<RectItem>(&item)) {
                draw_rect(painter, *rect, base.width, base.height);
            } else if (const ImageItem* image = std::get_if<ImageItem>(&item)) {
                draw_image(painter, *image, base.width, base.height, last_rx);
            }
        }
    }
    return from_qimage(canvas);
}

}  // namespace sstvae::overlay

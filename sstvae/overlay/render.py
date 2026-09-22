"""Rendering an `OverlayDoc` onto a picture.

Pure PIL, no Qt: the GUI editor draws its own live preview with Qt
items, but what actually gets transmitted is rendered here, so the
result is identical whether it came from the editor, a future saved
template, or a command-line `--overlay doc.json`.

Note the codec is trained for exactly this kind of content -- see the
burned-in text augmentation in `sstvae/data.py` and its comment on why
the training text is deliberately unstructured. Composition happens
*before* encoding, so the overlay is part of the picture the network
codes, not something laid on afterwards.
"""

import math
import os
from dataclasses import dataclass, replace
from functools import lru_cache

import numpy as np
from PIL import Image, ImageChops, ImageColor, ImageDraw, ImageFont

from .model import ImageItem, OverlayDoc, RectItem, SOURCE_LAST_RX, TextItem

# Same font search the training overlays use, so the GUI's default face
# matches what the model was trained on rather than being an arbitrary
# second choice.
from ..images import AVAILABLE_FONTS, find_font_face, open_image


@lru_cache(maxsize=64)
def _load_font(path: str | None, size: int):
    size = max(1, int(size))
    candidates = [path] if path else []
    candidates += list(AVAILABLE_FONTS)
    for p in candidates:
        if p and os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except OSError:
                continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1 has a bitmap-only default
        return ImageFont.load_default()


def _text_draw_kwargs(item, size_px: int) -> dict:
    """The PIL keyword arguments for drawing (or measuring) a text item.

    In one place so `item_bbox` and `_render_text` cannot disagree about
    geometry -- the editor's selection handle comes from the former and
    the picture from the latter. PIL applies `align` and `spacing` only
    when the string actually spans lines.
    """
    return {
        "anchor": item.anchor,
        "align": item.align,
        "spacing": max(0, round(item.line_spacing * size_px)),
    }


# --- text style -------------------------------------------------------------
#
# An item that asks for nothing -- no family, not bold, not italic, a solid
# fill, no underline -- takes exactly the path it always did, down to the
# same PIL call, so every document written before these fields existed
# renders byte-for-byte as before. Note that its face is the *training*
# face (`images.AVAILABLE_FONTS[0]`, DejaVu Sans **Bold** where installed),
# not a regular weight: that predates the style fields and is kept. A
# request, on the other hand, is honoured literally -- `italic` alone asks
# for a regular-weight italic -- which is also what the Qt renderer does.

# A style the chosen file does not carry is synthesized, as Qt does: bold
# by growing the glyphs, italic by shearing them.
_FAKE_BOLD = 0.03  # glyph growth, fraction of the font size
_ITALIC_SHEAR = 0.2  # about 11 degrees
# PIL exposes no underline metrics. These are DejaVu Sans's proportions;
# Qt reads the face's own, which is one more reason the two renderers
# agree on text in meaning rather than to the pixel.
_UNDERLINE_OFFSET = 0.08  # below the baseline, fraction of the font size
_UNDERLINE_THICKNESS = 0.05


@dataclass(frozen=True)
class _Face:
    font: ImageFont.FreeTypeFont
    fake_bold: int  # px the glyphs must be grown by; 0 for none
    fake_italic: bool


@lru_cache(maxsize=64)
def _face(path: str | None, family: str, bold: bool, italic: bool, size_px: int) -> _Face:
    """The font for a text item, and whatever style it still has to fake.

    `font` (a path) wins over `font_family`, and a path that cannot be
    read falls through to the family request -- the precedence the Qt
    renderer's `font_for` uses.
    """
    grow = max(1, round(size_px * _FAKE_BOLD))
    if path and os.path.exists(path):
        return _Face(_load_font(path, size_px), grow if bold else 0, italic)
    if not family and not bold and not italic:
        return _Face(_load_font(None, size_px), 0, False)
    found = find_font_face(family, bold, italic)
    if found is None:
        return _Face(_load_font(None, size_px), grow if bold else 0, italic)
    file, has_bold, has_italic = found
    return _Face(_load_font(file, size_px), grow if bold and not has_bold else 0,
                 italic and not has_italic)


def _face_for(item: TextItem, size_px: int) -> _Face:
    return _face(item.font, item.font_family, item.bold, item.italic, size_px)


def _is_plain(item: TextItem, face: _Face) -> bool:
    """Whether PIL can draw this item in a single native call.

    An unknown fill kind counts as solid here, deliberately: a caption
    that vanishes on a build that predates some later kind is worse than
    one drawn flat. A rect treats an unknown kind as "none".
    """
    if not isinstance(face.font, ImageFont.FreeTypeFont):
        return True  # Pillow's old bitmap default can be drawn, not styled
    return (item.fill_kind not in ("none", "gradient") and not item.underline
            and not face.fake_bold and not face.fake_italic)


def _layout_lines(item: TextItem, font, size_px: int, stroke: int,
                  x: float, y: float) -> list[tuple[float, float, str, str]]:
    """Where PIL puts each line of `item.text`, as (x, y, anchor, line).

    **A copy of Pillow's own multi-line layout** (`ImageText._split` in
    Pillow 12, `ImageDraw.multiline_text` before it -- the same arithmetic
    since 9.2): lines `getbbox("A")[3] + stroke + spacing` apart, aligned
    by width, each drawn with the block's anchor. It exists because the
    styled path draws each line separately -- its fill, stroke and
    underline masks must share one layout, and PIL's multi-line spacing
    moves with the stroke width each mask is drawn at -- and it must land
    exactly where PIL's single call would. `test_overlay.py` pins that,
    so a Pillow that changes the formula fails a test instead of shifting
    styled text by a few pixels.
    """
    anchor = item.anchor or "la"
    lines = item.text.split("\n")
    if len(lines) == 1:
        return [(x, y, anchor, lines[0])]
    if anchor[1] in "tb":
        raise ValueError("anchor not supported for multiline text")  # as PIL does
    spacing = _text_draw_kwargs(item, size_px)["spacing"]
    step = font.getbbox("A", stroke_width=stroke)[3] + stroke + spacing
    widths = [font.getlength(line) for line in lines]
    widest = max(widths)
    top = y
    if anchor[1] == "m":
        top -= (len(lines) - 1) * step / 2.0
    elif anchor[1] == "d":
        top -= (len(lines) - 1) * step
    placed = []
    for line, width in zip(lines, widths):
        slack = widest - width
        left = x + {"center": slack / 2.0, "right": slack}.get(item.align, 0.0)
        left -= {"m": slack / 2.0, "r": slack}.get(anchor[0], 0.0)
        placed.append((left, top, anchor, line))
        top += step
    return placed


def _underlines(font, size_px: int, placed) -> list[tuple[int, int, int, int]]:
    """One (x0, y0, x1, y1) bar per non-empty line, under its baseline."""
    ascent, descent = font.getmetrics()
    offset = max(1, round(size_px * _UNDERLINE_OFFSET))
    thickness = max(1, round(size_px * _UNDERLINE_THICKNESS))
    bars = []
    for lx, ly, anchor, line in placed:
        if not line:
            continue  # the Qt renderer skips empty lines too
        width = font.getlength(line)
        left = lx - {"m": width / 2.0, "r": width}.get(anchor[0], 0.0)
        vertical = anchor[1]
        if vertical == "a":
            baseline = ly + ascent
        elif vertical == "s":
            baseline = ly
        elif vertical == "d":
            baseline = ly - descent
        elif vertical == "m":
            baseline = ly + (ascent - descent) / 2.0
        else:  # "t" / "b", single-line only: measured from the ink
            box = font.getbbox(line, anchor="ls")
            baseline = ly - (box[1] if vertical == "t" else box[3])
        top = round(baseline) + offset
        bars.append((round(left), top, round(left + width), top + thickness))
    return bars


def _union(boxes):
    return (min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes))


def _styled_geometry(item: TextItem, face: _Face, size_px: int, stroke: int):
    """Everything the styled path draws, laid out about the origin.

    Returns `(placed, bars, layout_box, extent)`: the lines, the underline
    bars, the block's metric box (what a gradient spans -- the text, not
    the ink, so the ramp does not slide as the text is edited), and the
    full extent of the paint, stroke, underline and synthesized style
    included. `item_bbox` reports the extent, so a handle cannot clip
    anything this draws.
    """
    font = face.font
    measure = ImageDraw.Draw(Image.new("L", (1, 1)))
    placed = _layout_lines(item, font, size_px, stroke, 0.0, 0.0)
    bars = _underlines(font, size_px, placed) if item.underline else []
    layout_box = _union([measure.textbbox((lx, ly), line, font=font, anchor=anchor)
                         for lx, ly, anchor, line in placed])
    grow = stroke + face.fake_bold
    extent = _union(
        [measure.textbbox((lx, ly), line, font=font, anchor=anchor, stroke_width=grow)
         for lx, ly, anchor, line in placed]
        + [(b[0] - stroke, b[1] - stroke, b[2] + stroke, b[3] + stroke) for b in bars])
    if face.fake_italic:
        lean = _ITALIC_SHEAR * (extent[3] - extent[1]) / 2.0
        extent = (extent[0] - lean, extent[1], extent[2] + lean, extent[3])
    return placed, bars, layout_box, extent


def _text_mask(size, placed, font, stroke: int, bars, bar_grow: int,
               offset: tuple[int, int]) -> Image.Image:
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    ox, oy = offset
    for lx, ly, anchor, line in placed:
        if line:
            draw.text((lx + ox, ly + oy), line, font=font, anchor=anchor, fill=255,
                      stroke_width=stroke, stroke_fill=255)
    for x0, y0, x1, y1 in bars:
        draw.rectangle((x0 + ox - bar_grow, y0 + oy - bar_grow,
                        x1 + ox + bar_grow - 1, y1 + oy + bar_grow - 1), fill=255)
    return mask


def _through(paint: Image.Image, mask: Image.Image) -> Image.Image:
    """`paint` with `mask` as its coverage.

    Not `layer.paste(paint, mask=mask)`: that interpolates every channel
    toward the transparent black underneath, so an anti-aliased edge
    comes out darker as well as more transparent.
    """
    out = paint.copy()
    out.putalpha(ImageChops.multiply(out.getchannel("A"), mask))
    return out


def _composite_clipped(canvas: Image.Image, layer: Image.Image, x: int, y: int) -> None:
    """`alpha_composite` at (x, y), which may lie partly off the canvas."""
    sx, sy = max(0, -x), max(0, -y)
    if sx >= layer.width or sy >= layer.height:
        return
    canvas.alpha_composite(layer, (max(0, x), max(0, y)), (sx, sy))


def _render_styled_text(canvas: Image.Image, item: TextItem, face: _Face,
                        size_px: int, stroke: int) -> None:
    w, h = canvas.size
    x, y = round(item.x * w), round(item.y * h)
    placed, bars, layout_box, extent = _styled_geometry(item, face, size_px, stroke)

    # The layer is the extent plus a margin, with the text's origin at
    # (ox, oy) inside it.
    pad = 2
    ox, oy = pad - math.floor(extent[0]), pad - math.floor(extent[1])
    size = (math.ceil(extent[2]) + ox + pad, math.ceil(extent[3]) + oy + pad)
    font = face.font
    interior = _text_mask(size, placed, font, face.fake_bold, bars, 0, (ox, oy))
    filled = item.fill_kind != "none"

    layer = Image.new("RGBA", size, (0, 0, 0, 0))
    if stroke > 0:
        silhouette = _text_mask(size, placed, font, stroke + face.fake_bold, bars,
                                stroke, (ox, oy))
        # Under a fill the whole silhouette is painted and the fill laid
        # over it, as PIL's own stroke is; with no fill only the ring is.
        ring = silhouette if filled else ImageChops.subtract(silhouette, interior)
        stroke_paint = Image.new("RGBA", size, ImageColor.getcolor(item.stroke_color, "RGBA"))
        layer.alpha_composite(_through(stroke_paint, ring))
    if filled:
        if item.fill_kind == "gradient":
            box = (layout_box[0] + ox, layout_box[1] + oy,
                   layout_box[2] - layout_box[0], layout_box[3] - layout_box[1])
            fill_paint = _gradient_layer(size[0], size[1], item.color, item.fill_color2,
                                         item.fill_angle, item.fill_gradient, box)
        else:
            fill_paint = Image.new("RGBA", size, ImageColor.getcolor(item.color, "RGBA"))
        layer.alpha_composite(_through(fill_paint, interior))

    # The block's own centre -- the stroked text box, underline and
    # synthesized slant excluded -- is the pivot, as for the plain path.
    text_box = _union([ImageDraw.Draw(Image.new("L", (1, 1))).textbbox(
        (lx, ly), line, font=font, anchor=anchor, stroke_width=stroke)
        for lx, ly, anchor, line in placed])
    pivot_x = ox + (text_box[0] + text_box[2]) / 2.0
    pivot_y = oy + (text_box[1] + text_box[3]) / 2.0

    if face.fake_italic:
        # Leaning right: each row samples from its left by its height
        # above the pivot. The extent already made room for it.
        layer = layer.transform(layer.size, Image.AFFINE,
                                (1, _ITALIC_SHEAR, -_ITALIC_SHEAR * pivot_y, 0, 1, 0),
                                resample=Image.BICUBIC)

    if not item.rotation:
        _composite_clipped(canvas, layer, x - ox, y - oy)
        return

    # Rotate about the pivot: pad the layer until the pivot is its exact
    # centre, which `rotate(expand=True)` then keeps fixed.
    half_w = math.ceil(max(pivot_x, layer.width - pivot_x))
    half_h = math.ceil(max(pivot_y, layer.height - pivot_y))
    centred = Image.new("RGBA", (2 * half_w, 2 * half_h), (0, 0, 0, 0))
    centred.alpha_composite(layer, (round(half_w - pivot_x), round(half_h - pivot_y)))
    rotated = centred.rotate(item.rotation, resample=Image.BICUBIC, expand=True)
    _composite_clipped(canvas, rotated,
                       round(x + pivot_x - ox - rotated.width / 2.0),
                       round(y + pivot_y - oy - rotated.height / 2.0))


def _resolve_source(source: str, last_rx: Image.Image | None) -> Image.Image | None:
    """Late binding: turn an `ImageItem.source` reference into pixels.

    Returning None (rather than raising) for a missing source is
    deliberate -- a template referring to the last received image is
    perfectly valid on a session where nothing has been received yet,
    and should simply draw nothing.
    """
    if source == SOURCE_LAST_RX:
        return last_rx
    if not source or not os.path.exists(source):
        return None
    try:
        # Upright, like the main picture -- an inset is usually a
        # photograph too, and one of the two arriving sideways would be
        # the more confusing outcome.
        return open_image(source).convert("RGB")
    except (OSError, ValueError):
        # ValueError is `open_image`'s size refusal. An inset that is too
        # large to open draws nothing, like every other unusable source
        # here -- refusing to render the whole composition over one
        # decorative element would be the worse failure.
        return None


def _render_text(canvas: Image.Image, item: TextItem) -> None:
    if not item.text:
        return
    w, h = canvas.size
    size_px = round(item.size * h)
    face = _face_for(item, size_px)
    stroke = max(0, round(item.stroke_width * item.size * h))
    if not _is_plain(item, face):
        _render_styled_text(canvas, item, face, size_px, stroke)
        return

    font = face.font
    x, y = round(item.x * w), round(item.y * h)
    kw = _text_draw_kwargs(item, size_px)

    if not item.rotation:
        ImageDraw.Draw(canvas).text(
            (x, y), item.text, font=font, fill=item.color,
            stroke_width=stroke, stroke_fill=item.stroke_color, **kw,
        )
        return

    # Rotated text needs its own layer: PIL cannot rotate a draw call.
    pad = stroke * 2 + 4
    box = ImageDraw.Draw(Image.new("RGBA", (1, 1))).textbbox(
        (0, 0), item.text, font=font, stroke_width=stroke, **kw
    )
    layer = Image.new(
        "RGBA", (box[2] - box[0] + 2 * pad, box[3] - box[1] + 2 * pad), (0, 0, 0, 0)
    )
    ImageDraw.Draw(layer).text(
        (pad - box[0], pad - box[1]), item.text, font=font, fill=item.color,
        stroke_width=stroke, stroke_fill=item.stroke_color, **kw,
    )

    # About the text block's own centre, matching `_render_rect`/
    # `_render_image` -- not the point this used to pivot around
    # (pasting the *rotated*, bigger layer at the same offset the
    # unrotated one used), which let the block visibly swing away from
    # its own anchor as the angle grew, most obviously at 90 degrees.
    # `rotate(expand=True)` keeps the pre-rotation layer's own centre
    # fixed and only grows the canvas symmetrically around it, so the
    # fix is just placing that layer so its centre lands on the centre
    # of the *unrotated* bbox -- the same point `item_bbox` reports and
    # the editor's selection box and rotate handle are drawn around.
    centre_x = x + (box[0] + box[2]) / 2.0
    centre_y = y + (box[1] + box[3]) / 2.0
    layer = layer.rotate(item.rotation, resample=Image.BICUBIC, expand=True)
    paste_x = round(centre_x - layer.width / 2.0)
    paste_y = round(centre_y - layer.height / 2.0)
    canvas.alpha_composite(layer, (paste_x, paste_y))


def _gradient_layer(w: int, h: int, color1: str, color2: str, angle_deg: float,
                    shape: str = "linear",
                    box: tuple[float, float, float, float] | None = None) -> Image.Image:
    """A two-colour gradient as an RGBA layer, `w` x `h`.

    The gradient's geometry is `box` -- (x, y, width, height) within the
    layer, the whole layer by default -- and it is clamped to its end
    colours beyond it, as Qt's is. A text item's box is its metric box,
    while the layer also covers stroke, underline and synthesized bold.

    **Linear** runs from `color1` to `color2` along `angle_deg`,
    counter-clockwise, matching `RectItem.rotation`: 0 is left to right,
    90 bottom to top. **Radial** is centred on the box and reaches
    `color2` at its corners, with no angle. Anything but "radial" is
    linear -- which is also how an older build reads a radial one.

    Computed with numpy rather than PIL, which has no gradient primitive
    -- a full-layer array is cheap next to the font and image work the
    rest of this module already does per item.
    """
    c1 = np.array(ImageColor.getcolor(color1, "RGBA"), dtype=np.float32)
    c2 = np.array(ImageColor.getcolor(color2, "RGBA"), dtype=np.float32)
    bx, by, bw, bh = box if box is not None else (0, 0, w, h)
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = bx + (bw - 1) / 2.0, by + (bh - 1) / 2.0
    if shape == "radial":
        radius = math.hypot(bw, bh) / 2.0 or 1.0
        t = np.clip(np.hypot(xs - cx, ys - cy) / radius, 0.0, 1.0)[..., None]
    else:
        theta = math.radians(angle_deg)
        ux, uy = math.cos(theta), -math.sin(theta)
        proj = (xs - cx) * ux + (ys - cy) * uy
        extent = (abs(ux) * bw + abs(uy) * bh) / 2.0 or 1.0
        t = np.clip(proj / extent * 0.5 + 0.5, 0.0, 1.0)[..., None]
    arr = c1[None, None, :] * (1.0 - t) + c2[None, None, :] * t
    return Image.fromarray(np.round(arr).astype(np.uint8), "RGBA")


def _rect_fill_layer(w: int, h: int, kind: str, color: str, color2: str,
                     angle: float, shape: str = "linear") -> Image.Image | None:
    if kind == "solid":
        layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        ImageDraw.Draw(layer).rectangle((0, 0, w - 1, h - 1),
                                        fill=ImageColor.getcolor(color, "RGBA"))
        return layer
    if kind == "gradient":
        return _gradient_layer(w, h, color, color2, angle, shape)
    return None


def _render_rect(canvas: Image.Image, item: RectItem) -> None:
    w, h = canvas.size
    iw = max(1, round(item.width * w))
    ih = max(1, round(item.height * h))
    sw = max(0, round(item.stroke_width * w))

    layer = Image.new("RGBA", (iw, ih), (0, 0, 0, 0))

    fill = _rect_fill_layer(iw, ih, item.fill_kind, item.fill_color,
                            item.fill_color2, item.fill_angle, item.fill_gradient)
    if fill is not None:
        layer.alpha_composite(fill)

    if item.stroke_kind != "none" and sw > 0:
        stroke_src = _rect_fill_layer(iw, ih, "solid" if item.stroke_kind == "solid"
                                      else "gradient", item.stroke_color,
                                      item.stroke_color2, item.stroke_angle,
                                      item.stroke_gradient)
        # An outline band `sw` px thick, centred on the edge -- the same
        # place `ImageDraw.rectangle(outline=..., width=...)` puts it,
        # which is what a solid stroke draws with directly.
        mask = Image.new("L", (iw, ih), 0)
        ImageDraw.Draw(mask).rectangle((0, 0, iw - 1, ih - 1), outline=255, width=sw)
        layer.paste(stroke_src, (0, 0), mask)

    if layer.getbbox() is None:
        return  # every field is "none" -- a legal, invisible rectangle

    if item.rotation:
        layer = layer.rotate(item.rotation, resample=Image.BICUBIC, expand=True)

    x, y = round(item.x * w), round(item.y * h)
    if item.anchor.startswith("m"):
        x -= layer.width // 2
    elif item.anchor.startswith("r"):
        x -= layer.width
    if item.anchor.endswith("m"):
        y -= layer.height // 2
    elif item.anchor.endswith(("b", "d")):
        y -= layer.height
    canvas.alpha_composite(layer, (x, y))


def _render_image(canvas: Image.Image, item: ImageItem,
                  last_rx: Image.Image | None) -> None:
    src = _resolve_source(item.source, last_rx)
    if src is None:
        return
    w, h = canvas.size
    target_w = max(1, round(item.width * w))
    target_h = max(1, round(target_w * src.height / src.width))
    inset = src.convert("RGBA").resize((target_w, target_h), Image.LANCZOS)

    border = max(0, round(item.border * w))
    if border:
        framed = Image.new(
            "RGBA",
            (target_w + 2 * border, target_h + 2 * border),
            item.border_color,
        )
        framed.paste(inset, (border, border))
        inset = framed

    if item.opacity < 1.0:
        alpha = inset.getchannel("A").point(
            lambda a: round(a * max(0.0, min(1.0, item.opacity)))
        )
        inset.putalpha(alpha)

    if item.rotation:
        inset = inset.rotate(item.rotation, resample=Image.BICUBIC, expand=True)

    x, y = round(item.x * w), round(item.y * h)
    if item.anchor.startswith("m"):
        x -= inset.width // 2
    elif item.anchor.startswith("r"):
        x -= inset.width
    if item.anchor.endswith("m"):
        y -= inset.height // 2
    elif item.anchor.endswith("b") or item.anchor.endswith("d"):
        y -= inset.height
    canvas.alpha_composite(inset, (x, y))


def item_bbox(canvas_size: tuple[int, int], item,
              last_rx: Image.Image | None = None) -> tuple[int, int, int, int]:
    """Pixel (x, y, w, h) an item occupies on a canvas of `canvas_size`.

    Lives here rather than in the editor so the on-screen selection
    handles are positioned by the same geometry that draws the item --
    otherwise the two drift apart and the handle stops matching what the
    operator sees.
    """
    w, h = canvas_size
    x, y = round(item.x * w), round(item.y * h)

    if isinstance(item, TextItem):
        size_px = round(item.size * h)
        face = _face_for(item, size_px)
        font = face.font
        stroke = max(0, round(item.stroke_width * item.size * h))
        if not _is_plain(item, face):
            measured = item if item.text else _with_text(item, " ")
            _, _, _, extent = _styled_geometry(measured, face, size_px, stroke)
            x0, y0 = math.floor(x + extent[0]), math.floor(y + extent[1])
            x1, y1 = math.ceil(x + extent[2]), math.ceil(y + extent[3])
            return x0, y0, max(1, x1 - x0), max(1, y1 - y0)
        box = ImageDraw.Draw(Image.new("RGB", (1, 1))).textbbox(
            (x, y), item.text or " ", font=font, stroke_width=stroke,
            **_text_draw_kwargs(item, size_px),
        )
        return box[0], box[1], max(1, box[2] - box[0]), max(1, box[3] - box[1])

    if isinstance(item, RectItem):
        # Unrotated dimensions, like the `ImageItem` branch below --
        # neither accounts for `rotation` in the handle it hands the
        # editor, which is an existing simplification kept here for
        # consistency rather than a gap specific to rectangles.
        iw = max(1, round(item.width * w))
        ih = max(1, round(item.height * h))
        if item.anchor.startswith("m"):
            x -= iw // 2
        elif item.anchor.startswith("r"):
            x -= iw
        if item.anchor.endswith("m"):
            y -= ih // 2
        elif item.anchor.endswith(("b", "d")):
            y -= ih
        return x, y, iw, ih

    src = _resolve_source(item.source, last_rx)
    aspect = (src.height / src.width) if src else 0.75
    iw = max(1, round(item.width * w))
    ih = max(1, round(iw * aspect))
    border = max(0, round(item.border * w))
    iw += 2 * border
    ih += 2 * border
    if item.anchor.startswith("m"):
        x -= iw // 2
    elif item.anchor.startswith("r"):
        x -= iw
    if item.anchor.endswith("m"):
        y -= ih // 2
    elif item.anchor.endswith(("b", "d")):
        y -= ih
    return x, y, iw, ih


def _with_text(item: TextItem, text: str) -> TextItem:
    return replace(item, text=text)


def render(base: Image.Image, doc: OverlayDoc,
           last_rx: Image.Image | None = None) -> Image.Image:
    """Draw `doc` over `base` and return a new RGB image.

    `base` is used as-is; the caller is responsible for having framed it
    to the transmit size (`sstvae.images.fit_image`), because the
    document's normalized coordinates are relative to whatever it is
    given. `last_rx` supplies the pixels for any item whose source is
    `"last_rx"`.
    """
    canvas = base.convert("RGBA")
    for item in doc.items:
        if isinstance(item, TextItem):
            _render_text(canvas, item)
        elif isinstance(item, RectItem):
            _render_rect(canvas, item)
        elif isinstance(item, ImageItem):
            _render_image(canvas, item, last_rx)
    return canvas.convert("RGB")

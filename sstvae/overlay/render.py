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

import os
from functools import lru_cache

from PIL import Image, ImageDraw, ImageFont

from .model import ImageItem, OverlayDoc, SOURCE_LAST_RX, TextItem

# Same font search the training overlays use, so the GUI's default face
# matches what the model was trained on rather than being an arbitrary
# second choice.
from ..images import AVAILABLE_FONTS, open_image


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
    font = _load_font(item.font, size_px)
    stroke = max(0, round(item.stroke_width * item.size * h))
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
    layer = layer.rotate(item.rotation, resample=Image.BICUBIC, expand=True)
    canvas.alpha_composite(layer, (x - pad, y - pad))


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
        font = _load_font(item.font, size_px)
        stroke = max(0, round(item.stroke_width * item.size * h))
        box = ImageDraw.Draw(Image.new("RGB", (1, 1))).textbbox(
            (x, y), item.text or " ", font=font, stroke_width=stroke,
            **_text_draw_kwargs(item, size_px),
        )
        return box[0], box[1], max(1, box[2] - box[0]), max(1, box[3] - box[1])

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
        elif isinstance(item, ImageItem):
            _render_image(canvas, item, last_rx)
    return canvas.convert("RGB")

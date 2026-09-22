"""The overlay document: what to draw on top of a picture.

Two design choices here exist to make *templates* -- saving an overlay
and reapplying it to tomorrow's picture -- a later UI-only change rather
than a redesign:

**Coordinates are normalized** to 0..1 of the canvas, and sizes are
fractions of it. A document is therefore resolution-independent, so the
same one frames correctly whatever the base image was, and the editor
can be any size on screen without baking its pixel geometry in.

**Image insets are late-bound references, not pasted bitmaps.**
`ImageItem.source` is `"last_rx"` or a file path, resolved at render
time. A saved template that says "inset the most recent received image,
bottom left" therefore keeps meaning that next week -- which is the
whole point of a template, and would be impossible if the editor
flattened the bitmap in at composition time.

Nothing in this module (or in render.py) imports Qt, so the document and
its rendering stay testable and reusable from the command line.
"""

import json
from dataclasses import asdict, dataclass, field

from ..images import IMG_H, IMG_W

# The overlay's coordinate space is the transmitted frame itself, so
# what the editor shows is what goes on the air.
CANVAS_W, CANVAS_H = IMG_W, IMG_H

# Still 1 after fields were added, deliberately: readers drop fields they
# do not know, and every field added since the first release is written
# only when it differs from its default (`_SPARSE_FIELDS`). A document
# using no newer feature is therefore byte-identical to what an older
# build writes, and one that does degrades there -- a styled caption drawn
# plain, a radial gradient drawn linear -- instead of being refused. The
# same trade `RectItem` made when it arrived without a bump.
DOC_VERSION = 1

# Resolved at render time rather than stored, so the reference stays
# meaningful in a saved template.
SOURCE_LAST_RX = "last_rx"


@dataclass
class TextItem:
    """A run of burned-in text.

    `text` may contain newlines; a station's callsign, grid and name are
    one item, not three stacked by hand.

    `size` is the cap height as a fraction of canvas height, so text
    scales with the frame. `anchor` names which point of the text box
    (x, y) positions, in PIL's two-letter convention ("la" = left/
    ascender, "mm" = middle/middle), which is what lets a template pin
    text to a corner without knowing how long the string will be.
    `align` is how the lines sit relative to each other, which only
    matters once there is more than one.

    The style fields reproduce, at their defaults, exactly what a text
    item drew before they existed. `font_family` is a family name or a
    generic keyword ("sans-serif", "serif", "monospace", "cursive"), and
    `font` (a path) wins when both are set -- a template shipping its own
    face names the exact file it needs. The fill uses `RectItem`'s terms,
    with `color` as the first stop in every kind (it is what "solid"
    always drew) and "none" leaving only the stroke, for outlined text.
    The stroke stays solid.
    """

    text: str = ""
    x: float = 0.03
    y: float = 0.03
    size: float = 0.08
    color: str = "#ffffff"
    stroke_color: str = "#000000"
    stroke_width: float = 0.12  # fraction of the glyph size
    font: str | None = None  # path; None = the bundled/default face
    anchor: str = "la"
    align: str = "left"  # left | center | right, between lines
    line_spacing: float = 0.15  # extra gap between lines, fraction of size
    rotation: float = 0.0  # degrees, counter-clockwise

    bold: bool = False
    italic: bool = False
    underline: bool = False
    font_family: str = ""  # empty = no request
    fill_kind: str = "solid"  # "none" | "solid" | "gradient"
    fill_color2: str = "#000000"  # the gradient's second stop
    fill_angle: float = 0.0  # counter-clockwise, like rotation
    fill_gradient: str = "linear"  # "linear" | "radial"

    type: str = field(default="text", init=False)


@dataclass
class ImageItem:
    """A picture inset -- typically the last received image, so an
    operator can send back what they just got."""

    source: str = SOURCE_LAST_RX  # "last_rx" or a filesystem path
    x: float = 0.68
    y: float = 0.68
    width: float = 0.28  # fraction of canvas width; height follows aspect
    border: float = 0.004  # fraction of canvas width; 0 for none
    border_color: str = "#ffffff"
    opacity: float = 1.0
    rotation: float = 0.0
    anchor: str = "la"
    type: str = field(default="image", init=False)


@dataclass
class RectItem:
    """A filled and/or stroked rectangle.

    Fill and stroke are independent and each is "none", "solid" or
    "gradient" -- a plain toggle would need a second field for "what
    colour", so the kind and the colour(s) travel together per the
    project's flat-dataclass style (see `TextItem`). A gradient has two
    colours and is linear or radial. A linear one runs along its angle,
    which uses the same counter-clockwise convention as `rotation`
    below, and deliberately so -- the gradient is drawn into the item's
    own unrotated layer and then rotated with it (`render.py`), so a
    gradient's angle and the item's rotation add exactly as the two
    numbers suggest they should. A radial one is centred on the item and
    reaches the second colour at its corners; it has no angle.

    Radial is a field of its own rather than a fourth kind so that an
    older build, which drops fields it does not know, still draws the
    gradient -- as a linear one -- instead of losing the fill.
    """

    x: float = 0.1
    y: float = 0.1
    width: float = 0.3   # fraction of canvas width
    height: float = 0.2  # fraction of canvas height
    rotation: float = 0.0  # degrees, counter-clockwise
    anchor: str = "la"

    fill_kind: str = "none"  # "none" | "solid" | "gradient"
    fill_color: str = "#ffffff"
    fill_color2: str = "#000000"  # the gradient's second stop
    fill_angle: float = 0.0
    fill_gradient: str = "linear"  # "linear" | "radial"

    stroke_kind: str = "none"  # "none" | "solid" | "gradient"
    stroke_color: str = "#ffffff"
    stroke_color2: str = "#000000"
    stroke_angle: float = 0.0
    stroke_gradient: str = "linear"  # "linear" | "radial"
    stroke_width: float = 0.006  # fraction of canvas width

    type: str = field(default="rect", init=False)


_ITEM_TYPES = {"text": TextItem, "image": ImageItem, "rect": RectItem}

# Fields added after the format's first release, written only when they
# differ from their defaults -- the rule `OverlayDoc.name` already
# follows. A document that uses none of them then serializes exactly as
# it did before they existed, which is what lets DOC_VERSION stay at 1.
# The C++ writer (`put_unless_default` in native/core/overlay/model.cpp)
# omits the same fields by the same test, and tests/test_native_overlay.py
# holds the two to identical output.
_SPARSE_FIELDS = {
    TextItem: ("bold", "italic", "underline", "font_family", "fill_kind",
               "fill_color2", "fill_angle", "fill_gradient"),
    RectItem: ("fill_gradient", "stroke_gradient"),
}


def _item_dict(item) -> dict:
    out = asdict(item)
    fields = type(item).__dataclass_fields__
    for name in _SPARSE_FIELDS.get(type(item), ()):
        if out[name] == fields[name].default:
            del out[name]
    return out


@dataclass
class OverlayDoc:
    """An ordered list of items, drawn back to front.

    `name` is what makes a document a *template* (see `template.py`):
    a saved layout the operator picks by name. It is written only when
    set, so an unnamed document serializes exactly as it did before
    templates existed, and a template opens as a plain document in any
    build that has this model.
    """

    items: list = field(default_factory=list)
    version: int = DOC_VERSION
    name: str = ""

    def to_dict(self) -> dict:
        out = {
            "version": self.version,
            "items": [_item_dict(i) for i in self.items],
        }
        if self.name:
            out["name"] = self.name
        return out

    @classmethod
    def from_dict(cls, data: dict) -> "OverlayDoc":
        version = int(data.get("version", DOC_VERSION))
        name = data.get("name", "")
        if not isinstance(name, str):
            name = ""  # a wrong type costs the name, not the document
        if version > DOC_VERSION:
            raise ValueError(
                f"overlay document version {version} is newer than this "
                f"build understands (max {DOC_VERSION})"
            )
        items = []
        for raw in data.get("items", []):
            raw = dict(raw)
            kind = raw.pop("type", "text")
            item_cls = _ITEM_TYPES.get(kind)
            if item_cls is None:
                continue  # forward compatibility: ignore unknown item kinds
            # Drop unknown fields rather than crashing, so a document
            # written by a later version still mostly renders.
            known = {f for f in item_cls.__dataclass_fields__ if f != "type"}
            items.append(item_cls(**{k: v for k, v in raw.items() if k in known}))
        return cls(items=items, version=version, name=name)

    def to_json(self, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls, text: str) -> "OverlayDoc":
        return cls.from_dict(json.loads(text))

    def is_empty(self) -> bool:
        return not self.items

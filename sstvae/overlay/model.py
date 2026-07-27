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


_ITEM_TYPES = {"text": TextItem, "image": ImageItem}


@dataclass
class OverlayDoc:
    """An ordered list of items, drawn back to front."""

    items: list = field(default_factory=list)
    version: int = DOC_VERSION

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "items": [asdict(i) for i in self.items],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "OverlayDoc":
        version = int(data.get("version", DOC_VERSION))
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
        return cls(items=items, version=version)

    def to_json(self, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls, text: str) -> "OverlayDoc":
        return cls.from_dict(json.loads(text))

    def is_empty(self) -> bool:
        return not self.items

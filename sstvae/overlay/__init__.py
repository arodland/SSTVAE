"""Composing station text and insets onto a picture before transmit."""

from .model import CANVAS_H, CANVAS_W, ImageItem, OverlayDoc, RectItem, TextItem
from .render import item_bbox, render
from .template import (
    BUILTIN_FIELDS,
    Fields,
    Placeholders,
    builtin_templates,
    format_snr,
    load_template,
    placeholders,
    substitute,
)

__all__ = [
    "BUILTIN_FIELDS",
    "CANVAS_H",
    "CANVAS_W",
    "Fields",
    "ImageItem",
    "OverlayDoc",
    "Placeholders",
    "RectItem",
    "TextItem",
    "builtin_templates",
    "format_snr",
    "item_bbox",
    "load_template",
    "placeholders",
    "render",
    "substitute",
]

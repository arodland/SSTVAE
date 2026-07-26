"""Overlay document round-tripping and rendering."""

import numpy as np
from PIL import Image

from sstvae.overlay import CANVAS_H, CANVAS_W, ImageItem, OverlayDoc, TextItem, render


def base(color=(20, 40, 60)):
    return Image.new("RGB", (CANVAS_W, CANVAS_H), color)


def test_json_roundtrip_preserves_items():
    doc = OverlayDoc(items=[
        TextItem(text="N0CALL", x=0.1, y=0.2, size=0.09, color="#ff0000"),
        ImageItem(source="last_rx", x=0.7, width=0.25, rotation=5.0),
    ])
    back = OverlayDoc.from_json(doc.to_json())

    assert len(back.items) == 2
    assert isinstance(back.items[0], TextItem)
    assert back.items[0].text == "N0CALL"
    assert back.items[0].color == "#ff0000"
    assert isinstance(back.items[1], ImageItem)
    assert back.items[1].source == "last_rx"
    assert back.items[1].rotation == 5.0


def test_unknown_fields_and_item_types_are_ignored():
    """Forward compatibility: a document from a later build should still
    render what this build understands rather than failing to load."""
    doc = OverlayDoc.from_dict({
        "version": 1,
        "items": [
            {"type": "text", "text": "HI", "glow": 3},        # unknown field
            {"type": "hologram", "wow": True},                 # unknown type
        ],
    })
    assert len(doc.items) == 1
    assert doc.items[0].text == "HI"


def test_empty_doc_leaves_the_image_untouched():
    src = base()
    out = render(src, OverlayDoc())
    assert np.array_equal(np.asarray(src), np.asarray(out))


def test_text_changes_pixels_where_it_is_drawn_and_not_elsewhere():
    src = base()
    doc = OverlayDoc(items=[TextItem(text="TEST", x=0.02, y=0.02, size=0.15)])
    out = np.asarray(render(src, doc))
    before = np.asarray(src)

    top_left = np.any(out[: CANVAS_H // 3, : CANVAS_W // 2] != before[: CANVAS_H // 3, : CANVAS_W // 2])
    bottom_right = np.any(out[CANVAS_H // 2 :, CANVAS_W // 2 :] != before[CANVAS_H // 2 :, CANVAS_W // 2 :])
    assert top_left, "text was not drawn"
    assert not bottom_right, "text leaked into the far corner"


def test_last_rx_inset_is_drawn_from_the_supplied_image():
    src = base()
    rx = Image.new("RGB", (640, 480), (255, 0, 255))
    doc = OverlayDoc(items=[ImageItem(source="last_rx", x=0.6, y=0.6, width=0.3, border=0)])
    out = np.asarray(render(src, doc, last_rx=rx))

    # Somewhere in the lower right there should now be magenta.
    region = out[int(0.6 * CANVAS_H) :, int(0.6 * CANVAS_W) :]
    assert np.any(np.all(region == (255, 0, 255), axis=-1))


def test_missing_last_rx_renders_nothing_rather_than_failing():
    """A template asking for the last received image is valid on a
    session where nothing has been received yet."""
    src = base()
    doc = OverlayDoc(items=[ImageItem(source="last_rx")])
    out = render(src, doc, last_rx=None)
    assert np.array_equal(np.asarray(src), np.asarray(out))


def test_missing_file_source_renders_nothing():
    src = base()
    doc = OverlayDoc(items=[ImageItem(source="/nonexistent/nope.png")])
    assert np.array_equal(np.asarray(src), np.asarray(render(src, doc)))


def test_normalized_coordinates_scale_with_the_canvas():
    """The point of normalized coordinates: the same document frames the
    same way whatever size it is rendered at -- which is what will make
    saved templates portable."""
    doc = OverlayDoc(items=[ImageItem(source="last_rx", x=0.5, y=0.5, width=0.25, border=0)])
    rx = Image.new("RGB", (100, 100), (0, 255, 0))

    small = np.asarray(render(Image.new("RGB", (320, 240), "black"), doc, rx))
    large = np.asarray(render(Image.new("RGB", (640, 480), "black"), doc, rx))

    def green_fraction(a):
        return float(np.mean(np.all(a == (0, 255, 0), axis=-1)))

    # The inset covers the same *proportion* of the frame at either size.
    assert green_fraction(small) > 0
    assert np.isclose(green_fraction(small), green_fraction(large), atol=0.005)


def test_render_returns_rgb():
    out = render(base(), OverlayDoc(items=[TextItem(text="X")]))
    assert out.mode == "RGB"
    assert out.size == (CANVAS_W, CANVAS_H)

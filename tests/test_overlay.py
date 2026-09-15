"""Overlay document round-tripping and rendering."""

import numpy as np
from PIL import Image

from sstvae.overlay import CANVAS_H, CANVAS_W, ImageItem, OverlayDoc, RectItem, TextItem, render


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


# --- rectangles -----------------------------------------------------------


def test_rect_json_roundtrip():
    doc = OverlayDoc(items=[
        RectItem(x=0.1, y=0.2, width=0.3, height=0.15, fill_kind="solid",
                 fill_color="#ff0000", stroke_kind="gradient",
                 stroke_color="#00ff00", stroke_color2="#0000ff",
                 stroke_angle=30.0, rotation=10.0),
    ])
    back = OverlayDoc.from_json(doc.to_json())
    assert isinstance(back.items[0], RectItem)
    assert back.items[0] == doc.items[0]


def test_rect_with_every_field_none_draws_nothing():
    src = base()
    doc = OverlayDoc(items=[RectItem(x=0.1, y=0.1, width=0.3, height=0.2)])
    assert np.array_equal(np.asarray(src), np.asarray(render(src, doc)))


def test_solid_rect_fills_its_area_and_nothing_else():
    src = base()
    doc = OverlayDoc(items=[
        RectItem(x=0.1, y=0.1, width=0.2, height=0.2, fill_kind="solid", fill_color="#ff0000"),
    ])
    out = np.asarray(render(src, doc))
    inside = out[int(0.15 * CANVAS_H), int(0.15 * CANVAS_W)]
    outside = out[int(0.5 * CANVAS_H), int(0.5 * CANVAS_W)]
    assert tuple(inside) == (255, 0, 0)
    assert tuple(outside) == tuple(np.asarray(src)[int(0.5 * CANVAS_H), int(0.5 * CANVAS_W)])


def test_gradient_rect_interpolates_between_its_two_colors():
    src = base()
    doc = OverlayDoc(items=[
        RectItem(x=0.0, y=0.0, width=1.0, height=1.0, fill_kind="gradient",
                 fill_color="#ff0000", fill_color2="#0000ff", fill_angle=0.0),
    ])
    out = np.asarray(render(src, doc))
    left = out[CANVAS_H // 2, 2]
    right = out[CANVAS_H // 2, CANVAS_W - 3]
    # Left edge close to color1 (red), right edge close to color2 (blue).
    assert left[0] > left[2]
    assert right[2] > right[0]


def test_stroke_only_rect_draws_an_outline_not_a_fill():
    src = base()
    doc = OverlayDoc(items=[
        RectItem(x=0.1, y=0.1, width=0.3, height=0.2, stroke_kind="solid",
                 stroke_color="#00ff00", stroke_width=0.02),
    ])
    out = np.asarray(render(src, doc))
    before = np.asarray(src)
    center_y, center_x = int(0.2 * CANVAS_H), int(0.25 * CANVAS_W)
    edge_y, edge_x = int(0.1 * CANVAS_H), int(0.25 * CANVAS_W)
    assert np.array_equal(out[center_y, center_x], before[center_y, center_x])
    assert tuple(out[edge_y, edge_x]) == (0, 255, 0)


def test_rect_bbox_ignores_rotation_like_the_image_item_does():
    """Matches the existing (documented) simplification for ImageItem --
    the selection handle is sized from the unrotated extent."""
    from sstvae.overlay import item_bbox
    item = RectItem(x=0.1, y=0.1, width=0.3, height=0.2, rotation=45.0)
    assert item_bbox((CANVAS_W, CANVAS_H), item) == (
        round(0.1 * CANVAS_W), round(0.1 * CANVAS_H),
        round(0.3 * CANVAS_W), round(0.2 * CANVAS_H))


# --- templates (docs/overlay-templates.md) ------------------------------
#
# The three rules and the placeholder grammar, stated against the spec.
# The C++ implementation is held to the same outputs in
# tests/test_native_overlay.py; these are what "the same" is measured
# against, so a rule wrong in both at once still fails here.

from sstvae.overlay import Fields, builtin_templates, format_snr, placeholders, substitute
from sstvae.overlay.template import BUILTIN_FIELDS, normalize_label, substitute_text


def _sub(text, builtin=None, custom=None):
    return substitute_text(text, Fields(builtin or {}, custom or {}))


def test_builtin_placeholders_are_replaced():
    assert _sub("de {mycall}", {"mycall": "KC2G"}) == "de KC2G"


def test_unknown_placeholders_are_left_literally():
    """Rule 1: a typo must show in the preview, not vanish on the air."""
    f = {"mycall": "KC2G"}
    assert _sub("de {mycal}", f) == "de {mycal}"
    assert _sub("{MYCALL}", f) == "{MYCALL}"
    assert _sub("{my call}", f) == "{my call}"


def test_doubled_braces_are_literal():
    assert _sub("{{mycall}}", {"mycall": "KC2G"}) == "{mycall}"
    assert _sub("a {{ b }} c") == "a { b } c"


def test_unclosed_and_stray_braces_are_literal():
    f = {"mycall": "KC2G"}
    assert _sub("{mycall", f) == "{mycall"
    assert _sub("} {mycall}", f) == "} KC2G"
    assert _sub("{{mycall}", f) == "{mycall}"


def test_a_line_whose_placeholders_are_all_empty_is_dropped_whole():
    """Rule 2, including the literal text on that line: `SNR ` goes with
    its placeholder, since a label with nothing after it is worse than
    no line."""
    f = {"theircall": "W1XYZ", "mycall": "KC2G"}
    assert _sub("{theircall} de {mycall}\nSNR {snr}\n{field Comment}", f) == "W1XYZ de KC2G"


def test_a_line_without_placeholders_is_never_touched():
    assert _sub("CQ CQ CQ\n\nde {mycall}") == "CQ CQ CQ\n"


def test_a_line_with_any_filled_placeholder_is_kept_intact():
    assert _sub("{theircall} de {mycall}", {"mycall": "KC2G"}) == " de KC2G"


def test_whitespace_only_and_missing_values_are_empty():
    assert _sub("{field Comment}", custom={"Comment": "   "}) == ""
    assert _sub("x\n{field Comment}", custom={"Comment": "   "}) == "x"
    assert _sub("SNR {snr}") == ""
    assert _sub("{theircall}\n{snr}") == ""


def test_custom_fields_by_label():
    assert _sub("{field Comment}", custom={"Comment": "TNX FER PIC"}) == "TNX FER PIC"
    assert _sub("QTH {field Their QTH}", custom={"Their QTH": "Boston"}) == "QTH Boston"
    assert _sub("QTH {field  Their   QTH }", custom={"Their QTH": "Boston"}) == "QTH Boston"
    assert normalize_label("  a \t b  ") == "a b"


def test_the_same_label_twice_is_one_field():
    doc = OverlayDoc(items=[TextItem(text="{field Comment}\n{field Comment} again")])
    assert placeholders(doc).custom == ["Comment"]
    out = substitute(doc, Fields(custom={"Comment": "hi"}))
    assert out.items[0].text == "hi\nhi again"


def test_malformed_field_declarations_are_unknown_placeholders():
    assert _sub("{field}", custom={"": "x"}) == "{field}"
    assert _sub("{field }", custom={"": "x"}) == "{field }"
    assert _sub("{fieldx}") == "{fieldx}"


def test_builtin_and_custom_namespaces_do_not_collide():
    out = _sub("{mycall} {field mycall}", {"mycall": "KC2G"}, {"mycall": "custom"})
    assert out == "KC2G custom"


def test_placeholders_in_first_appearance_order_without_repeats():
    doc = OverlayDoc(items=[
        TextItem(text="{snr} {mycall}\n{field Comment} {nonsense}"),
        ImageItem(),
        TextItem(text="{mycall} {field QTH} {field Comment}"),
    ])
    p = placeholders(doc)
    assert p.builtin == ["snr", "mycall"]
    assert p.custom == ["Comment", "QTH"]
    assert all(name in BUILTIN_FIELDS for name in p.builtin)


def test_substitute_returns_a_copy_and_leaves_images_alone():
    """Rule 3."""
    doc = OverlayDoc(name="Reply", items=[TextItem(text="{theircall} de {mycall}"), ImageItem()])
    out = substitute(doc, Fields({"theircall": "W1XYZ", "mycall": "KC2G"}))
    assert doc.items[0].text == "{theircall} de {mycall}"
    assert out.items[0].text == "W1XYZ de KC2G"
    assert isinstance(out.items[1], ImageItem)
    assert out.name == "Reply"


def test_substitute_leaves_rects_alone_and_counts_no_placeholders():
    doc = OverlayDoc(items=[RectItem(fill_kind="solid", fill_color="#ff0000")])
    assert placeholders(doc).builtin == []
    assert placeholders(doc).custom == []
    out = substitute(doc, Fields({"mycall": "KC2G"}))
    assert out.items[0] == doc.items[0]


def test_name_round_trips_and_is_omitted_when_empty():
    assert OverlayDoc.from_json(OverlayDoc(name="CQ").to_json()).name == "CQ"
    assert "name" not in OverlayDoc().to_dict()
    assert OverlayDoc.from_dict({"version": 1, "items": [], "name": 7}).name == ""


def test_format_snr():
    assert format_snr(12.4) == "12 dB"
    assert format_snr(12.5) == "12 dB"  # half to even, like numpy and nearbyint
    assert format_snr(13.5) == "14 dB"
    assert format_snr(-2.6) == "-3 dB"
    assert format_snr(-0.3) == "0 dB"
    assert format_snr(None) == ""


def test_shipped_templates():
    cq, reply, picture = builtin_templates()
    assert [d.name for d in (cq, reply, picture)] == ["CQ", "Reply", "Reply with picture"]
    # Every built-in carries a Comment line; only Reply asks for their call.
    for doc in (cq, reply, picture):
        assert placeholders(doc).custom == ["Comment"]
    assert placeholders(cq).builtin == ["mycall", "grid"]
    assert placeholders(reply).builtin == ["theircall", "mycall", "snr"]
    assert picture.items[0].text == reply.items[0].text
    assert len(picture.items) == 2 and picture.items[1].source == "last_rx"

    filled = substitute(reply, Fields({"theircall": "W1XYZ", "mycall": "KC2G", "snr": "12 dB"}))
    assert filled.items[0].text == "W1XYZ de KC2G\nSNR 12 dB"
    filled = substitute(cq, Fields({"mycall": "KC2G"}, {"Comment": "QRZ?"}))
    assert filled.items[0].text == "CQ CQ CQ\nde KC2G\nQRZ?"

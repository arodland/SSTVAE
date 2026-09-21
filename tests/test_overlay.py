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


# --- text style and radial gradients --------------------------------------


def test_text_style_and_radial_gradients_round_trip():
    text = TextItem(text="KC2G", bold=True, italic=True, underline=True,
                    font_family="serif", fill_kind="gradient", fill_color2="#123456",
                    fill_angle=30.0, fill_gradient="radial")
    rect = RectItem(fill_kind="gradient", fill_gradient="radial",
                    stroke_kind="gradient", stroke_gradient="radial")
    back = OverlayDoc.from_json(OverlayDoc(items=[text, rect]).to_json())
    assert back.items == [text, rect]


def test_style_fields_are_written_only_when_set():
    """Like `name`: an unstyled document is exactly what an older build
    writes. (The C++ writer is held to the same rule, per field, in
    test_native_overlay.py.)"""
    text_item, rect_item = OverlayDoc(items=[TextItem(), RectItem()]).to_dict()["items"]
    for key in ("bold", "italic", "underline", "font_family", "fill_kind",
                "fill_color2", "fill_angle", "fill_gradient"):
        assert key not in text_item, key
    assert "fill_gradient" not in rect_item and "stroke_gradient" not in rect_item
    # A rect's own fill fields predate this change and are always written.
    assert rect_item["fill_kind"] == "none"
    assert OverlayDoc(items=[TextItem(bold=True)]).to_dict()["items"][0]["bold"] is True


# --- text style: the Python renderer --------------------------------------
#
# The styled path draws each line itself, from a copy of Pillow's own
# multi-line layout, so its fill, stroke and underline masks share one
# layout. The first test is what makes that copy safe to keep: a Pillow
# that changed its line-spacing formula would shift styled text by a few
# pixels and every other test here would still pass.

from dataclasses import replace  # noqa: E402

import pytest  # noqa: E402

from sstvae.images import find_font_face  # noqa: E402
from sstvae.overlay import item_bbox  # noqa: E402

BG = (20, 40, 60)


def ink(out: Image.Image, background=BG) -> np.ndarray:
    """A boolean mask of every pixel the overlay touched."""
    return np.any(np.asarray(out) != np.array(background, dtype=np.uint8), axis=2)


def ink_box(out: Image.Image, background=BG) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(ink(out, background))
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def render_item(item, size=(640, 480)) -> Image.Image:
    return render(Image.new("RGB", size, BG), OverlayDoc(items=[item]))


@pytest.mark.parametrize("anchor", ["la", "ma", "ra", "ls", "mm", "md"])
@pytest.mark.parametrize("align", ["left", "center", "right"])
@pytest.mark.parametrize("stroke_width", [0.12, 0.0])
def test_the_styled_path_lays_lines_out_exactly_where_pillow_does(anchor, align, stroke_width):
    # A one-colour gradient is drawn by the styled path and looks like a
    # solid fill, so the two renders must agree to rounding -- a line one
    # pixel off shows up as a full-strength difference along every edge.
    # With no stroke under it, the fill's anti-aliased edge meets bare
    # canvas, which is also where compositing it wrongly would show.
    text = "Wide first line\nmid\nlast one"
    plain = TextItem(text=text, x=0.5, y=0.4, size=0.07, anchor=anchor, align=align,
                     line_spacing=0.3, color="#ffcc00", stroke_width=stroke_width)
    styled = replace(plain, fill_kind="gradient", fill_color2="#ffcc00")
    a = np.asarray(render_item(plain), dtype=np.int16)
    b = np.asarray(render_item(styled), dtype=np.int16)
    assert np.abs(a - b).max() <= 2, (anchor, align, stroke_width, int(np.abs(a - b).max()))


def test_solid_text_never_reads_the_gradient_fields():
    plain = TextItem(text="W1AW", size=0.2)
    noisy = TextItem(text="W1AW", size=0.2, fill_color2="#00ff00", fill_angle=77.0,
                     fill_gradient="radial")
    assert np.array_equal(np.asarray(render_item(plain)), np.asarray(render_item(noisy)))
    # And an unknown kind draws solid rather than nothing, as in C++: a
    # caption that vanishes on an older build is worse than one drawn flat.
    future = TextItem(text="W1AW", size=0.2, fill_kind="shimmer")
    assert np.array_equal(np.asarray(render_item(plain)), np.asarray(render_item(future)))


def test_underline_draws_a_bar_under_every_line():
    # Capitals, so nothing of the plain text reaches below a baseline and
    # every row the underline adds is new ink. Two lines, so the bars must
    # come in two separate groups -- one under each -- which is what the
    # copied multi-line layout is for.
    plain = TextItem(text="WAVE\nMAST", size=0.12, stroke_width=0.0, line_spacing=0.5)
    under = TextItem(text="WAVE\nMAST", size=0.12, stroke_width=0.0, line_spacing=0.5,
                     underline=True)
    rows_plain = set(np.nonzero(ink(render_item(plain)).any(axis=1))[0])
    added = sorted(set(np.nonzero(ink(render_item(under)).any(axis=1))[0]) - rows_plain)
    groups = [[added[0]]]
    for r in added[1:]:
        if r == groups[-1][-1] + 1:
            groups[-1].append(r)
        else:
            groups.append([r])
    assert len(groups) == 2, groups
    first_band_bottom = max(r for r in rows_plain if r < groups[0][0])
    assert groups[0][0] > first_band_bottom, "the first bar sits below the first line"
    assert groups[1][0] > max(rows_plain), "the second sits below the last line"


def test_outline_text_is_hollow():
    # Every pixel deep inside a filled render's ink must be background in
    # the outline render -- "none" means outlined, not filled in the
    # stroke colour. (The same assertion caught the C++ renderer.)
    filled = TextItem(text="HOLLOW", size=0.3, color="#ff0000", stroke_width=0.0)
    outline = TextItem(text="HOLLOW", size=0.3, fill_kind="none", stroke_color="#00ff00",
                       stroke_width=0.06)
    red = np.all(np.asarray(render_item(filled)) == (255, 0, 0), axis=2)
    deep = red.copy()
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            deep &= np.roll(np.roll(red, dy, axis=0), dx, axis=1)
    assert deep.sum() > 1000
    out_ink = ink(render_item(outline))
    assert not (out_ink & deep).any(), "the glyph interiors stay background"
    assert out_ink.any(), "and the outline is drawn"
    outline.stroke_width = 0.0
    assert not ink(render_item(outline)).any(), "no fill and no stroke draws nothing"


def _font_file(stem: str) -> str | None:
    found = find_font_face({"DejaVuSans": "sans-serif", "DejaVuSansMono": "monospace"}[stem],
                           False, False)
    return found[0] if found and found[0].endswith(f"/{stem}.ttf") else None


def test_bold_and_italic_change_the_glyphs_real_face_or_synthesized():
    # With a family, bold and italic select real faces where installed;
    # with a font *file*, which carries one face only, they are synthesized
    # as Qt does. Both must visibly do something.
    regular = TextItem(text="Wave", size=0.25, stroke_width=0.0, font_family="serif")
    bold = TextItem(text="Wave", size=0.25, stroke_width=0.0, font_family="serif", bold=True)
    italic = TextItem(text="Wave", size=0.25, stroke_width=0.0, font_family="serif",
                      italic=True)
    base_ink = ink(render_item(regular)).sum()
    assert ink(render_item(bold)).sum() > base_ink * 1.1
    assert not np.array_equal(np.asarray(render_item(italic)), np.asarray(render_item(regular)))

    path = _font_file("DejaVuSans")
    if path is None:
        pytest.skip("DejaVuSans.ttf not installed; synthesized styles untested here")
    from_file = TextItem(text="Wave", size=0.25, stroke_width=0.0, font=path)
    fake_bold = TextItem(text="Wave", size=0.25, stroke_width=0.0, font=path, bold=True)
    assert ink(render_item(fake_bold)).sum() > ink(render_item(from_file)).sum() * 1.1

    # A synthesized slant leans *right*: vertical stems, so the top of the
    # ink sits further right than the bottom by an amount a shear the
    # wrong way (or none) cannot produce.
    def top_minus_bottom(item):
        mask = ink(render_item(item))
        rows = np.nonzero(mask.any(axis=1))[0]
        quarter = (rows[-1] - rows[0]) // 4
        xs_top = np.nonzero(mask[rows[0]:rows[0] + quarter])[1]
        xs_bottom = np.nonzero(mask[rows[-1] - quarter:rows[-1] + 1])[1]
        return xs_top.mean() - xs_bottom.mean()
    upright = TextItem(text="IIII", size=0.25, stroke_width=0.0, font=path)
    slanted = TextItem(text="IIII", size=0.25, stroke_width=0.0, font=path, italic=True)
    assert abs(top_minus_bottom(upright)) < 1.0
    assert top_minus_bottom(slanted) > 5.0, "a synthesized italic leans right"


def test_styled_ink_stays_inside_item_bbox():
    """The editor's handle is item_bbox; it must not clip anything the
    styled path draws -- underline, synthesized weight or slant included."""
    path = _font_file("DejaVuSans")
    variants = [
        TextItem(text="Under\nlined", size=0.1, underline=True),
        TextItem(text="Grad", size=0.15, fill_kind="gradient", fill_gradient="radial"),
        TextItem(text="Bold", size=0.15, font_family="serif", bold=True, underline=True),
    ]
    if path is not None:
        variants += [TextItem(text="Slant", size=0.15, font=path, italic=True),
                     TextItem(text="Heavy", size=0.15, font=path, bold=True, underline=True)]
    for item in variants:
        x, y, w, h = item_bbox((640, 480), item)
        box = ink_box(render_item(item))
        assert box[0] >= x and box[1] >= y and box[2] <= x + w and box[3] <= y + h, (
            item.text, box, (x, y, w, h))


def _lean(out: Image.Image, box) -> float:
    """Mean (red - blue) over the ink inside `box`."""
    arr = np.asarray(out, dtype=np.float64)
    region = ink(out)[box[1]:box[3], box[0]:box[2]]
    px = arr[box[1]:box[3], box[0]:box[2]][region]
    return float((px[:, 0] - px[:, 2]).mean())


def _ramp(**kw) -> TextItem:
    return TextItem(text="MMMMM", x=0.05, y=0.2, size=0.3, stroke_width=0.0,
                    fill_kind="gradient", color="#ff0000", fill_color2="#0000ff", **kw)


def test_a_linear_text_gradient_runs_counter_clockwise_from_its_angle():
    out = render_item(_ramp())
    x0, y0, x1, y1 = ink_box(out)
    third = (x1 - x0) // 3
    assert _lean(out, (x0, y0, x0 + third, y1)) > 0, "angle 0 starts red on the left"
    assert _lean(out, (x1 - third, y0, x1, y1)) < 0, "and ends blue on the right"
    # 90 runs bottom to top -- the counter-clockwise sense of `rotation`
    # and of a rect's gradient. A clockwise one puts red on top.
    up = render_item(_ramp(fill_angle=90.0))
    x0, y0, x1, y1 = ink_box(up)
    half = (y0 + y1) // 2
    assert _lean(up, (x0, half, x1, y1)) > _lean(up, (x0, y0, x1, half))


def _deep(mask: np.ndarray) -> np.ndarray:
    """Pixels of `mask` whose eight neighbours are in it too."""
    out = mask.copy()
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            out &= np.roll(np.roll(mask, dy, axis=0), dx, axis=1)
    return out


@pytest.mark.parametrize("shape", ["linear", "radial"])
def test_a_stroke_does_not_move_the_ramp(shape):
    # The gradient spans the text's own box, not its paint -- as in the Qt
    # renderer, where it spans the layout box. So the fill colour deep in
    # the glyphs is the same under a heavy stroke as under none; a ramp
    # stretched over the stroke's extent would shift every one of them.
    bare = replace(_ramp(fill_gradient=shape), text="MM", size=0.2)
    heavy = replace(bare, stroke_width=0.3, stroke_color="#00ff00")
    a, b = render_item(bare), render_item(heavy)
    # "Inside" from an exact-colour solid render, eroded: exact means fully
    # covered, where merely "not background" would admit anti-aliased
    # edge pixels whose partial coverage lets the stroke show through.
    solid = replace(bare, fill_kind="solid", color="#ff00ff")
    inside = _deep(np.all(np.asarray(render_item(solid)) == (255, 0, 255), axis=2))
    assert inside.sum() > 1000
    diff = np.abs(np.asarray(a, dtype=np.int16) - np.asarray(b, dtype=np.int16))[inside]
    assert diff.max() <= 2, int(diff.max())


def test_a_radial_text_gradient_is_centred():
    out = render_item(_ramp(fill_gradient="radial"))
    x0, y0, x1, y1 = ink_box(out)
    third = (x1 - x0) // 3
    middle = _lean(out, (x0 + third, y0, x1 - third, y1))
    assert middle > _lean(out, (x0, y0, x0 + third, y1))
    assert middle > _lean(out, (x1 - third, y0, x1, y1))


def test_a_text_gradient_turns_with_a_rotated_item():
    # A quarter turn counter-clockwise puts the text's own left-to-right
    # ramp bottom to top on screen.
    out = render_item(replace(_ramp(), x=0.5, y=0.5, anchor="mm", rotation=90.0, size=0.2))
    x0, y0, x1, y1 = ink_box(out)
    assert y1 - y0 > x1 - x0, "the rotated text stands upright"
    half = (y0 + y1) // 2
    assert _lean(out, (x0, half, x1, y1)) > 0 > _lean(out, (x0, y0, x1, half))


def test_a_radial_rect_is_centred_and_so_is_its_stroke():
    rect = RectItem(x=0.0, y=0.0, width=1.0, height=1.0, fill_kind="gradient",
                    fill_color="#ff0000", fill_color2="#0000ff", fill_gradient="radial")
    out = np.asarray(render(Image.new("RGB", (200, 100), BG), OverlayDoc(items=[rect])))
    assert out[50, 100, 0] > 200 and out[50, 100, 2] < 55, "the first colour at the centre"
    assert out[0, 0, 2] > 200 and out[0, 0, 0] < 55, "the second at the corners"

    def stroked(shape):
        item = RectItem(x=0.1, y=0.1, width=0.8, height=0.8, stroke_kind="gradient",
                        stroke_color="#ff0000", stroke_color2="#0000ff", stroke_width=0.05,
                        stroke_gradient=shape)
        return np.asarray(render(Image.new("RGB", (200, 100), BG), OverlayDoc(items=[item])))
    assert not np.array_equal(stroked("linear"), stroked("radial")), "a stroke can be radial too"


def test_a_family_is_requested_and_a_font_file_still_wins():
    if _font_file("DejaVuSansMono") is None:
        pytest.skip("DejaVu Sans Mono not installed")
    sans = TextItem(text="iiiiii", font_family="sans-serif")
    mono = TextItem(text="iiiiii", font_family="monospace")
    assert item_bbox((640, 480), mono)[2] > item_bbox((640, 480), sans)[2]
    path = _font_file("DejaVuSans")
    if path is None:
        pytest.skip("DejaVuSans.ttf not installed")
    from_file = TextItem(text="iiiiii", font=path)
    both = TextItem(text="iiiiii", font=path, font_family="monospace")
    assert item_bbox((640, 480), both)[2] == item_bbox((640, 480), from_file)[2]

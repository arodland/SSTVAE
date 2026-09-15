"""The C++ overlay document reader against the Python one.

The *document* only. Rendering lands in Phase 3 with the editor, so that
`item_bbox` has a single implementation shared between the drawn picture
and the editor's selection handles -- which is the property
`sstvae/overlay/render.py` exists to guarantee, and which two separate
ports would quietly break.

What matters here is that a saved overlay (and, later, a template) means
the same thing to both apps. A template is specifically a document meant
to outlive the session that wrote it, so a field silently dropped on
load is a template that degrades every time it is opened.
"""

import json

import pytest

from sstvae.overlay.model import DOC_VERSION, ImageItem, OverlayDoc, RectItem, TextItem


def _cpp(native):
    if not hasattr(native, "overlay"):
        pytest.skip("built without the overlay module")
    return native.overlay


def test_canvas_matches_the_transmitted_frame(native):
    """The overlay's coordinate space is the frame itself, so what the
    editor shows is what goes on the air."""
    from sstvae.overlay.model import CANVAS_H, CANVAS_W

    cpp = _cpp(native)
    assert (cpp.CANVAS_W, cpp.CANVAS_H) == (CANVAS_W, CANVAS_H)
    assert cpp.DOC_VERSION == DOC_VERSION


def test_empty_document_round_trips(native):
    cpp = _cpp(native)
    text, notes = cpp.round_trip(OverlayDoc().to_json())
    assert not notes
    assert json.loads(text) == OverlayDoc().to_dict()


def test_default_items_round_trip(native):
    cpp = _cpp(native)
    doc = OverlayDoc(items=[TextItem(text="KC2G"), ImageItem()])
    text, notes = cpp.round_trip(doc.to_json())
    assert not notes
    assert json.loads(text) == doc.to_dict()


def test_a_fully_specified_document_round_trips(native):
    """Every field non-default, so a reader that ignored the file and
    returned defaults could not pass.

    Multi-line text and a rotation are in here on purpose: they are the
    two things the renderer treats specially, so they are the two most
    likely to be dropped by a document reader written alongside it.
    """
    cpp = _cpp(native)
    doc = OverlayDoc(items=[
        TextItem(
            text="KC2G\nFN31pr\nAndrew",
            x=0.11, y=0.77, size=0.055,
            color="#ffcc00", stroke_color="#101010", stroke_width=0.2,
            font="/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
            anchor="mm", align="center", line_spacing=0.3, rotation=-7.5,
        ),
        ImageItem(
            source="/home/op/pictures/last.png",
            x=0.62, y=0.05, width=0.33, border=0.01,
            border_color="#00ff88", opacity=0.65, rotation=12.0, anchor="rb",
        ),
    ])
    text, notes = cpp.round_trip(doc.to_json())
    assert not notes, notes
    assert json.loads(text) == doc.to_dict()


def test_a_fully_specified_rect_round_trips(native):
    """Every field non-default, matching the text/image test above --
    fill and stroke each get a turn at "gradient" so both colours and
    the angle are exercised, not just the solid defaults."""
    cpp = _cpp(native)
    doc = OverlayDoc(items=[
        RectItem(
            x=0.15, y=0.62, width=0.4, height=0.22, rotation=33.0, anchor="mm",
            fill_kind="gradient", fill_color="#112233", fill_color2="#445566",
            fill_angle=45.0,
            stroke_kind="solid", stroke_color="#ff00ff", stroke_color2="#00ffff",
            stroke_angle=90.0, stroke_width=0.015,
        ),
    ])
    text, notes = cpp.round_trip(doc.to_json())
    assert not notes, notes
    assert json.loads(text) == doc.to_dict()


def test_last_rx_reference_is_preserved_verbatim(native):
    """The late-bound reference is the whole point of a template.

    If a reader ever resolved this to a path at load time, a saved
    template would stop meaning "the most recent received picture" and
    start meaning "that one picture from last Tuesday".
    """
    from sstvae.overlay.model import SOURCE_LAST_RX

    cpp = _cpp(native)
    doc = OverlayDoc(items=[ImageItem(source=SOURCE_LAST_RX)])
    text, _ = cpp.round_trip(doc.to_json())
    assert json.loads(text)["items"][0]["source"] == SOURCE_LAST_RX


def test_documents_the_python_side_writes_are_readable(native):
    """A spread of documents built through the reference's own API."""
    cpp = _cpp(native)
    docs = [
        OverlayDoc(),
        OverlayDoc(items=[TextItem()]),
        OverlayDoc(items=[ImageItem()]),
        OverlayDoc(items=[RectItem()]),
        OverlayDoc(items=[TextItem(text="a"), TextItem(text="b"), ImageItem()]),
        OverlayDoc(items=[TextItem(text="", size=0.0, rotation=360.0)]),
        OverlayDoc(items=[RectItem(fill_kind="solid"), RectItem(stroke_kind="gradient")]),
    ]
    for doc in docs:
        text, notes = cpp.round_trip(doc.to_json())
        assert not notes, (doc.to_dict(), notes)
        assert json.loads(text) == doc.to_dict()


def test_cpp_output_is_readable_by_the_reference(native):
    """The other direction: what C++ writes, Python must accept."""
    cpp = _cpp(native)
    doc = OverlayDoc(items=[
        TextItem(text="W1AW", rotation=15.0),
        ImageItem(opacity=0.5),
        RectItem(fill_kind="gradient", fill_angle=60.0),
    ])
    text, _ = cpp.round_trip(doc.to_json())
    assert OverlayDoc.from_json(text).to_dict() == doc.to_dict()


def test_unknown_item_kinds_and_fields_are_skipped_and_reported(native):
    """Forward compatibility, matching the reference -- but noisier.

    Python drops these silently. Reporting them is what makes a
    hand-edited document's typo visible instead of mysterious.
    """
    cpp = _cpp(native)
    data = {
        "version": 1,
        "items": [
            {"type": "text", "text": "hi", "future_field": 3},
            {"type": "hologram", "wow": True},
            {"type": "image", "source": "last_rx"},
        ],
    }
    text, notes = cpp.round_trip(json.dumps(data))
    got = json.loads(text)

    assert [i["type"] for i in got["items"]] == ["text", "image"]
    assert got["items"][0]["text"] == "hi"
    reported = " ".join(f"{w}: {p}" for w, p in notes)
    assert "future_field" in reported
    assert "hologram" in reported


def test_a_newer_document_version_is_refused_by_both(native):
    """Unlike the config, a document the operator explicitly opened
    should fail loudly rather than silently becoming an empty overlay."""
    cpp = _cpp(native)
    data = json.dumps({"version": DOC_VERSION + 1, "items": []})

    with pytest.raises(ValueError):
        OverlayDoc.from_json(data)
    with pytest.raises(RuntimeError, match="newer than this build"):
        cpp.round_trip(data)


@pytest.mark.parametrize("broken", [
    "", "{", "[1,2,3]", "null", '{"items": 5}', "not json",
])
def test_malformed_documents_raise(native, broken):
    cpp = _cpp(native)
    with pytest.raises(RuntimeError):
        cpp.round_trip(broken)


def test_a_bad_field_does_not_discard_the_whole_document(native):
    """One wrong type should cost that field, not the overlay."""
    cpp = _cpp(native)
    data = {
        "version": 1,
        "items": [{"type": "text", "text": "keep me", "x": "not a number"}],
    }
    text, notes = cpp.round_trip(json.dumps(data))
    got = json.loads(text)["items"][0]

    assert got["text"] == "keep me"
    assert got["x"] == TextItem().x, "the bad value should leave the default"
    assert any("x" in where for where, _ in notes)


# --- templates ----------------------------------------------------------
#
# The substitution rules are pure string processing with a spec, so the
# two implementations are held to *identical* output on a corpus that
# covers every branch of the grammar: escapes, unknown placeholders,
# custom labels with odd whitespace, the dropped-line rule in each of its
# cases, and the shipped templates.

TEMPLATE_CORPUS = [
    "de {mycall}",
    "de {mycal}",
    "{MYCALL} {my call} {}",
    "{{mycall}} a {{ b }} c",
    "{mycall",
    "} {mycall}",
    "{{mycall}",
    "{theircall} de {mycall}\nSNR {snr}\n{field Comment}",
    "CQ CQ CQ\n\nde {mycall}",
    "{theircall} de {mycall}",
    "{field Comment}",
    "x\n{field Comment}",
    "{theircall}\n{snr}",
    "QTH {field  Their   QTH }",
    "{field Comment}\n{field Comment} again",
    "{field}",
    "{field }",
    "{fieldx}",
    "{mycall} {field mycall}",
    "{snr} {mycall}\n{field Comment} {nonsense}",
    "{utc} {date} {mode} {grid} {name}",
    "{ {mycall} }",
    "{{{mycall}}}",
    "trailing newline\n",
    "",
]

TEMPLATE_FIELD_SETS = [
    ({}, {}),
    ({"mycall": "KC2G"}, {}),
    ({"theircall": "W1XYZ", "mycall": "KC2G", "snr": "12 dB"}, {"Comment": "TNX"}),
    ({"mycall": "KC2G"}, {"Comment": "   ", "Their QTH": "Boston", "mycall": "custom"}),
    ({"utc": "12:34", "date": "2026-09-14", "mode": "B", "grid": "FN31", "name": "Andrew"}, {}),
]


def test_template_substitution_agrees_on_the_corpus(native):
    from sstvae.overlay import Fields
    from sstvae.overlay.template import substitute_text

    cpp = _cpp(native)
    for text in TEMPLATE_CORPUS:
        for builtin, custom in TEMPLATE_FIELD_SETS:
            want = substitute_text(text, Fields(builtin, custom))
            got = cpp.substitute_text(text, builtin, custom)
            assert got == want, (text, builtin, custom)


def test_placeholders_agree_on_the_corpus(native):
    from sstvae.overlay import placeholders

    cpp = _cpp(native)
    for text in TEMPLATE_CORPUS:
        doc = OverlayDoc(items=[TextItem(text=text), ImageItem(), TextItem(text="{mycall}")])
        want = placeholders(doc)
        got_builtin, got_custom = cpp.placeholders(doc.to_json())
        assert (list(got_builtin), list(got_custom)) == (want.builtin, want.custom), text


def test_whole_document_substitution_agrees(native):
    from sstvae.overlay import Fields, builtin_templates, substitute

    cpp = _cpp(native)
    for doc in builtin_templates():
        for builtin, custom in TEMPLATE_FIELD_SETS:
            want = substitute(doc, Fields(builtin, custom)).to_dict()
            got = json.loads(cpp.substitute(doc.to_json(), builtin, custom))
            assert got == want, (doc.name, builtin, custom)


def test_named_documents_round_trip_through_cpp(native):
    cpp = _cpp(native)
    doc = OverlayDoc(name="Reply with picture", items=[TextItem(text="{theircall}")])
    text, notes = cpp.round_trip(doc.to_json())
    assert not notes
    assert json.loads(text) == doc.to_dict()
    assert OverlayDoc.from_json(text).name == "Reply with picture"


def test_format_snr_agrees(native):
    from sstvae.overlay import format_snr

    cpp = _cpp(native)
    for value in [None, 0.0, 0.5, 1.5, 2.5, 12.4, 12.5, 13.5, -0.3, -2.5, -2.6, 27.49, 99.5]:
        assert cpp.format_snr(value) == format_snr(value), value

"""Editing behaviour of the overlay panel.

Guards two things that only show up when a human types into the box:
the caret must stay where they put it, and text must be able to span
lines.
"""

import numpy as np
import pytest

pytest.importorskip("PySide6")

from PIL import Image  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from sstvae.overlay import CANVAS_H, CANVAS_W, OverlayDoc, TextItem, item_bbox, render  # noqa: E402


_APP = None  # module-level so the QApplication outlives every widget;
# letting it be collected first aborts the interpreter on exit


@pytest.fixture(scope="module")
def qapp():
    global _APP
    _APP = QApplication.instance() or QApplication([])
    return _APP


class _FakeAppState:
    """Just the parts TransmitPanel reads.

    Deliberately not a real `AppState`: that one starts a background
    checkpoint download and opens a rig connection on construction,
    neither of which has anything to do with editing text.
    """

    def __init__(self):
        from sstvae.gui.settings import Config

        self.config = Config()
        self.model = None

    def ptt(self):
        return None

    def save_config(self):
        pass


@pytest.fixture
def panel(qapp):
    from sstvae.gui.tx_panel import TransmitPanel

    widget = TransmitPanel(_FakeAppState())
    widget.editor.set_base_image(
        Image.new("RGB", (CANVAS_W, CANVAS_H), (30, 30, 30))
    )
    yield widget
    widget.deleteLater()
    qapp.processEvents()


# --- caret behaviour ----------------------------------------------------

def test_editing_the_middle_of_the_text_keeps_the_caret_there(panel):
    """The regression: every keystroke rebuilt the handles, which fired
    selectionChanged, which reloaded the text box and dropped the caret
    to the end -- so you could only ever append."""
    panel.editor.add_text("N0CALL")
    box = panel.text_edit

    cursor = box.textCursor()
    cursor.setPosition(0)
    box.setTextCursor(cursor)

    box.insertPlainText("X")

    assert box.toPlainText() == "XN0CALL"
    assert box.textCursor().position() == 1, (
        "caret jumped away from where the character was typed"
    )
    assert panel.editor.doc.items[0].text == "XN0CALL"


def test_typing_several_characters_at_the_front_keeps_their_order(panel):
    panel.editor.add_text("CALL")
    box = panel.text_edit
    cursor = box.textCursor()
    cursor.setPosition(0)
    box.setTextCursor(cursor)

    for ch in "W1":
        box.insertPlainText(ch)

    assert box.toPlainText() == "W1CALL"


def test_rebuilding_handles_does_not_report_a_selection_change(panel):
    panel.editor.add_text("HI")
    seen = []
    panel.editor.selectionChanged.connect(seen.append)

    panel.editor.refresh_item()

    assert seen == [], "a document refresh is not a selection change"


def test_selecting_a_different_item_still_updates_the_panel(panel):
    """The suppression must not go so far that real selection changes
    stop reaching the property panel."""
    panel.editor.add_text("FIRST")
    panel.editor.add_text("SECOND")
    assert panel.text_edit.toPlainText() == "SECOND"

    seen = []
    panel.editor.selectionChanged.connect(seen.append)
    panel.editor._handles[0].setSelected(True)

    assert seen, "selecting another item reported nothing"
    assert panel.text_edit.toPlainText() == "FIRST"


# --- multi-line ---------------------------------------------------------

def test_multiline_text_reaches_the_document(panel):
    panel.editor.add_text("N0CALL")
    panel.text_edit.setPlainText("N0CALL\nFN20\nAndrew")
    assert panel.editor.doc.items[0].text == "N0CALL\nFN20\nAndrew"


def test_multiline_renders_taller_than_one_line():
    base = Image.new("RGB", (CANVAS_W, CANVAS_H), "black")
    one = render(base, OverlayDoc(items=[TextItem(text="ONE", size=0.08)]))
    three = render(base, OverlayDoc(items=[TextItem(text="ONE\nTWO\nSIX", size=0.08)]))

    def ink_rows(img):
        a = np.asarray(img).sum(axis=2)
        return np.count_nonzero(a.sum(axis=1) > 0)

    assert ink_rows(three) > 2 * ink_rows(one), "extra lines were not drawn"


def test_bbox_tracks_the_extra_lines(panel):
    """The selection handle has to grow with the text, or it stops
    matching what is on screen."""
    one = item_bbox((CANVAS_W, CANVAS_H), TextItem(text="ONE", size=0.08))
    three = item_bbox((CANVAS_W, CANVAS_H), TextItem(text="ONE\nTWO\nSIX", size=0.08))
    assert three[3] > 2 * one[3]


def test_alignment_moves_the_shorter_line():
    """A short line above a long one sits in a different place under each
    alignment."""
    base = Image.new("RGB", (CANVAS_W, CANVAS_H), "black")

    def first_line_centre(align):
        doc = OverlayDoc(items=[TextItem(text="X\nWIDE TEXT", size=0.08, align=align)])
        a = np.asarray(render(base, doc)).sum(axis=2)
        rows = np.flatnonzero(a.sum(axis=1) > 0)
        cols = np.flatnonzero(a[rows[0] + 5] > 0)
        return (cols[0] + cols[-1]) / 2

    left, centre, right = (first_line_centre(a) for a in ("left", "center", "right"))
    assert left < centre < right, (
        f"alignment had no effect (left={left}, centre={centre}, right={right})"
    )


def test_multiline_survives_a_json_roundtrip():
    doc = OverlayDoc(items=[TextItem(text="A\nB", align="center", line_spacing=0.3)])
    back = OverlayDoc.from_json(doc.to_json())
    assert back.items[0].text == "A\nB"
    assert back.items[0].align == "center"
    assert back.items[0].line_spacing == 0.3

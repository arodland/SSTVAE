"""Saving whatever picture is on screen.

The decode loop reconstructs on every poll, so a picture exists long
before a transmission finishes — and on a marginal signal it may never
reach "complete" at all. Save has to follow the preview, not the
reception state machine.
"""


import pytest

pytest.importorskip("PySide6")

from PIL import Image  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from sstvae.gui.settings import Config  # noqa: E402
from sstvae.rx import Reception  # noqa: E402

_APP = None


@pytest.fixture(scope="module")
def qapp():
    global _APP
    _APP = QApplication.instance() or QApplication([])
    return _APP


class _FakeAppState:
    def __init__(self, tmp_path):
        self.config = Config()
        self.config.folders.receive_dir = str(tmp_path)
        self.model = None
        self.current_frequency_hz = 14_340_000.0

    def ptt(self):
        return None

    def save_config(self):
        pass


@pytest.fixture
def panel(qapp, tmp_path):
    from sstvae.gui.rx_panel import ReceivePanel

    widget = ReceivePanel(_FakeAppState(tmp_path))
    yield widget
    widget.deleteLater()
    qapp.processEvents()


def picture(colour=(10, 20, 30)):
    return Image.new("RGB", (640, 480), colour)


def make_reception(**kw):
    defaults = dict(
        image=picture((90, 10, 10)), mode_name="B", callsign="N0CALL",
        snr_db=12.5, frames_received=440, n_frames_expected=440,
    )
    defaults.update(kw)
    return Reception(**defaults)


# --- availability --------------------------------------------------------

def test_save_starts_disabled(panel):
    assert not panel.save_btn.isEnabled()


def test_a_partial_decode_enables_save(panel):
    """The behaviour asked for: mid-reception there is a picture, so it
    must be savable without waiting for the transmission to end."""
    panel._state.status = "receiving"
    panel._state.image = picture()
    panel._state.callsign = "W1AW"
    panel._state.mode_name = "C"
    panel._thread = _AlwaysAlive()

    panel._refresh_status()

    assert panel.save_btn.isEnabled()
    assert panel._displayed is not None
    assert panel._displayed.callsign == "W1AW"
    assert panel._displayed.mode_name == "C"


def test_a_completed_reception_also_enables_save(panel):
    panel._on_reception(make_reception(), None)
    assert panel.save_btn.isEnabled()
    assert panel._displayed.callsign == "N0CALL"


def test_save_stays_available_after_the_receiver_stops(panel):
    """Stopping the receiver doesn't take the picture off the screen, so
    it shouldn't take away the button either."""
    panel._on_reception(make_reception(), None)
    panel.stop()
    assert panel.save_btn.isEnabled()


# --- what actually gets written ------------------------------------------

def test_saving_writes_the_displayed_picture(panel, tmp_path, monkeypatch):
    partial = picture((7, 8, 9))
    panel._set_displayed(partial, "W1AW", "C")

    out = tmp_path / "partial.png"
    monkeypatch.setattr(
        "sstvae.gui.rx_panel.QFileDialog.getSaveFileName",
        lambda *a, **k: (str(out), ""),
    )
    panel.save_current()

    assert out.exists()
    with Image.open(out) as saved:
        assert saved.convert("RGB").getpixel((0, 0)) == (7, 8, 9)


def test_the_suggested_name_uses_the_partial_receptions_details(panel, monkeypatch):
    panel._set_displayed(picture(), "W1AW", "C")
    seen = {}

    def fake_dialog(parent, title, suggested, filt):
        seen["suggested"] = suggested
        return ("", "")  # operator cancels

    monkeypatch.setattr(
        "sstvae.gui.rx_panel.QFileDialog.getSaveFileName", fake_dialog
    )
    panel.save_current()

    assert "W1AW" in seen["suggested"]
    assert "14.340MHz" in seen["suggested"]


def test_a_later_decode_replaces_what_gets_saved(panel):
    """Successive polls improve the picture; save must follow the newest
    one rather than pinning the first."""
    panel._set_displayed(picture((1, 1, 1)), "W1AW", "C")
    panel._set_displayed(picture((2, 2, 2)), "W1AW", "C")
    assert panel._displayed.image.getpixel((0, 0)) == (2, 2, 2)


def test_saving_with_nothing_displayed_is_a_no_op(panel):
    panel._displayed = None
    panel.save_current()  # must not raise


def test_only_finished_receptions_feed_the_transmit_inset(panel):
    """A half-decoded picture is a poor 'last received image' to send
    back, so the inset signal stays tied to completion."""
    emitted = []
    panel.imageReceived.connect(emitted.append)

    panel._set_displayed(picture(), "W1AW", "C")
    assert emitted == []

    panel._on_reception(make_reception(), None)
    assert len(emitted) == 1


class _AlwaysAlive:
    """Stands in for the decode thread so `listening` reads True."""

    def is_alive(self):
        return True

    def join(self, timeout=None):
        return None

"""The GUI's reception sink: autosave, naming, and failure handling.

These need PySide6 (the sink carries results back on Qt signals) but no
display -- the Qt offscreen platform is forced in conftest.
"""

from pathlib import Path

import pytest

pytest.importorskip("PySide6")

from PIL import Image  # noqa: E402

from sstvae.gui.rx_panel import _GuiSink, _unique  # noqa: E402
from sstvae.gui.settings import Config  # noqa: E402
from sstvae.rx import Reception  # noqa: E402


def make_reception(callsign="N0CALL", mode="B"):
    return Reception(
        image=Image.new("RGB", (640, 480), (10, 20, 30)),
        mode_name=mode,
        callsign=callsign,
        snr_db=12.5,
        frames_received=440,
        n_frames_expected=440,
    )


def make_sink(cfg, freq_hz=14340000.0):
    from sstvae.gui.rx_panel import _Signals

    signals = _Signals()
    seen = []
    signals.reception.connect(lambda rec, path: seen.append((rec, path)))
    errors = []
    signals.error.connect(errors.append)
    return _GuiSink(signals, lambda: (cfg, freq_hz)), seen, errors


def test_autosave_writes_a_file_named_from_the_template(tmp_path):
    cfg = Config()
    cfg.folders.receive_dir = str(tmp_path)
    cfg.receive.autosave = True
    sink, seen, errors = make_sink(cfg)

    path = sink.on_reception(make_reception())

    assert path is not None
    saved = Path(path)
    assert saved.exists()
    assert saved.parent == tmp_path
    assert "14.340MHz" in saved.name
    assert "N0CALL" in saved.name
    assert not errors
    assert seen[0][1] == path


def test_autosave_off_keeps_the_picture_but_writes_nothing(tmp_path):
    """The Save button needs the reception in hand even though nothing
    was written -- that is the whole point of the autosave checkbox."""
    cfg = Config()
    cfg.folders.receive_dir = str(tmp_path)
    cfg.receive.autosave = False
    sink, seen, _ = make_sink(cfg)

    assert sink.on_reception(make_reception()) is None
    assert list(tmp_path.iterdir()) == []
    assert len(seen) == 1 and seen[0][1] is None


def test_the_directory_is_created_on_demand(tmp_path):
    cfg = Config()
    cfg.folders.receive_dir = str(tmp_path / "new" / "deep")
    sink, _, _ = make_sink(cfg)
    assert Path(sink.on_reception(make_reception())).exists()


def test_no_rig_means_no_frequency_in_the_name(tmp_path):
    cfg = Config()
    cfg.folders.receive_dir = str(tmp_path)
    sink, _, _ = make_sink(cfg, freq_hz=None)
    assert "MHz" not in Path(sink.on_reception(make_reception())).name


def test_save_size_is_applied(tmp_path):
    cfg = Config()
    cfg.folders.receive_dir = str(tmp_path)
    cfg.receive.save_size = "320x240"
    sink, _, _ = make_sink(cfg)
    with Image.open(sink.on_reception(make_reception())) as img:
        assert img.size == (320, 240)


def test_a_save_failure_is_reported_rather_than_killing_the_decode_loop(tmp_path):
    """The sink runs on the decode thread; an unhandled OSError there
    would take the whole receiver down."""
    cfg = Config()
    blocker = tmp_path / "blocked"
    blocker.write_text("I am a file, not a directory")
    cfg.folders.receive_dir = str(blocker)
    sink, seen, errors = make_sink(cfg)

    assert sink.on_reception(make_reception()) is None
    assert errors and "could not save" in errors[0]
    assert len(seen) == 1, "the picture is still delivered to the UI"


def test_two_receptions_in_the_same_second_do_not_overwrite(tmp_path):
    cfg = Config()
    cfg.folders.receive_dir = str(tmp_path)
    sink, _, _ = make_sink(cfg)

    first = sink.on_reception(make_reception())
    second = sink.on_reception(make_reception())

    assert first != second
    assert len(list(tmp_path.iterdir())) == 2


def test_unique_leaves_a_free_name_alone(tmp_path):
    path = tmp_path / "rx.png"
    assert _unique(path) == path
    path.touch()
    assert _unique(path) == tmp_path / "rx_2.png"

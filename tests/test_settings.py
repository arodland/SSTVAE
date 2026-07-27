"""Config persistence and received-image filenames."""

import json
from datetime import datetime, timezone

from sstvae.gui.settings import Config, format_filename


def test_roundtrip_preserves_nested_sections(tmp_path):
    cfg = Config(callsign="N0CALL")
    cfg.rig.enabled = True
    cfg.rig.port = 4999
    cfg.receive.autosave = False
    cfg.transmit.mode = "C"
    path = cfg.save(tmp_path / "config.json")

    back = Config.load(path)
    assert back.callsign == "N0CALL"
    assert back.rig.enabled is True
    assert back.rig.port == 4999
    assert back.receive.autosave is False
    assert back.transmit.mode == "C"
    # Nested sections must come back as dataclasses, not dicts.
    assert back.rig.host == "127.0.0.1"


def test_missing_file_gives_defaults(tmp_path):
    cfg = Config.load(tmp_path / "nothing-here.json")
    assert cfg.callsign == ""
    assert cfg.transmit.mode == "B"


def test_corrupt_file_does_not_stop_startup(tmp_path):
    """A broken config must not prevent the app from launching -- the
    settings dialog is how the operator would fix it."""
    path = tmp_path / "config.json"
    path.write_text("{ this is not json")
    assert Config.load(path).transmit.mode == "B"


def test_unknown_keys_are_ignored_not_fatal(tmp_path):
    """An older build reading a newer build's config keeps the settings
    it understands instead of refusing or wiping them."""
    path = tmp_path / "config.json"
    path.write_text(json.dumps({
        "callsign": "W1AW",
        "future_option": {"nested": 1},
        "rig": {"port": 4533, "unknown_rig_field": "x"},
    }))
    cfg = Config.load(path)
    assert cfg.callsign == "W1AW"
    assert cfg.rig.port == 4533


def test_save_is_atomic_and_leaves_no_temp_file(tmp_path):
    path = tmp_path / "config.json"
    Config(callsign="A").save(path)
    Config(callsign="B").save(path)
    assert Config.load(path).callsign == "B"
    assert list(tmp_path.iterdir()) == [path], "a .tmp file was left behind"


def test_save_creates_the_directory(tmp_path):
    path = tmp_path / "deep" / "nested" / "config.json"
    Config().save(path)
    assert path.exists()


# --- filenames ---------------------------------------------------------

WHEN = datetime(2026, 7, 26, 1, 15, 42, tzinfo=timezone.utc)
TEMPLATE = Config().receive.filename_template


def test_default_filename_matches_the_projects_recording_convention():
    name = format_filename(TEMPLATE, callsign="N0CALL", freq_hz=14340000.0, when=WHEN)
    assert name == "2026-07-26_011542Z_14.340MHz_N0CALL"


def test_absent_fields_drop_out_without_leaving_empty_separators():
    # No rig connected and no callsign decoded.
    assert format_filename(TEMPLATE, when=WHEN) == "2026-07-26_011542Z"
    assert format_filename(TEMPLATE, callsign="N0CALL", when=WHEN) == (
        "2026-07-26_011542Z_N0CALL"
    )
    assert format_filename(TEMPLATE, freq_hz=7074000.0, when=WHEN) == (
        "2026-07-26_011542Z_7.074MHz"
    )


def test_slashes_in_a_portable_callsign_do_not_become_directories():
    name = format_filename("{callsign}", callsign="N0CALL/P", when=WHEN)
    assert "/" not in name
    assert name == "N0CALL-P"


def test_unknown_template_field_is_left_alone_rather_than_crashing():
    assert format_filename("{date}_{nope}", when=WHEN) == "2026-07-26_{nope}"


def test_template_that_renders_empty_still_yields_a_name():
    assert format_filename("{callsign}", when=WHEN) == "2026-07-26_011542Z"

"""The C++ config reader.

This used to be a two-implementation test: the native reader against
`sstvae.gui.settings.Config`, checking that a `config.json` written by
either app was understood by the other. The Python GUI is gone
(2026-08-01), so there is only one implementation and nothing to
compare it to. What is left is the part that was always the real
content -- a settings reader's bug is a field it accepts and then
quietly drops, and that is checkable without a second implementation.

`NON_DEFAULT` is how: a complete config in which **no field holds its
default**, required to survive a round trip byte for byte. A dropped
field comes back as its default, and no default appears in the fixture,
so it cannot survive. `test_the_fixture_holds_no_default` keeps that
property true as fields are added -- it compares the fixture against the
reader's own defaults key by key, so a *new* setting that nobody added
here fails immediately rather than being silently untested.

Two things outlive the Python app for their own reasons. The v1 rig
section must still **migrate quietly**: the shape changed on 2026-07-29
(it described a rigctld socket -- host, port, spawn_local -- which is
the one part of rig control the native app does not have, since it links
libhamlib in-process), and it is not the operator's fault that their
file is old, so the dead keys are recognized-and-ignored rather than
reported. And the filename templates are checked against **frozen
expected strings** rather than against a second implementation, which
is the stronger form: it pins what the operator sees rather than only
that two pieces of code agree.

The reader also reports what it ignored. Python dropped unknown or
ill-typed keys silently, which is the right *effect* (an old build must
not wipe a new build's settings) but makes a typo in a hand-edited
config invisible. Those notes are tested here, because a diagnostic
nobody checks is a diagnostic that stops being true.
"""

import copy
import json
import os
import subprocess
from pathlib import Path

import pytest

# Where conftest drops the built extension, for the subprocess test below.
NATIVE_PYTHON_DIR = Path(__file__).resolve().parent.parent / "native" / "build" / "python"


def _cpp(native):
    if not hasattr(native, "settings"):
        pytest.skip("built without the settings module")
    return native.settings


# A complete v2 config with nothing at its default. Paths are absolute
# and invented so the fixture does not depend on $HOME -- the *defaults*
# for the folders do, which is exactly why they must not be reused here.
NON_DEFAULT = {
    "callsign": "KC2G",
    "model_path": "/opt/models/v3",
    "precision": "int8",
    "audio": {
        "backend": "portaudio",
        "input_device": "USB Audio CODEC",
        "output_device": "SSTVAE-Loopback",
        "samplerate": 8000,  # see test_samplerate_is_refused_rather_than_obeyed
    },
    "rig": {
        "enabled": True,
        "model": 3073,
        "device": "COM5",
        "baud": 115200,
        "data_bits": "eight",
        "stop_bits": "two",
        "parity": "even",
        "handshake": "hardware",
        "dtr": "high",
        "rts": "low",
        "ptt_method": "rts",
        "ptt_device": "COM7",
        "mode": "pkt_usb",
        "ptt_lead_s": 0.45,
        "ptt_tail_s": 0.25,
        "poll_interval_s": 2.5,
    },
    "folders": {
        "receive_dir": "/srv/sstv/in",
        "transmit_dir": "/srv/sstv/out",
        "template_dir": "/srv/sstv/tpl",
    },
    "receive": {
        "autosave": False,
        "low_cpu": True,
        "buffer_seconds": 95.5,
        "poll_interval": 11.0,
        "blind_search_seconds": 30.0,
        "end_grace": 6.5,
        "save_size": "320x240",
        "save_audio": True,
        "filename_template": "{callsign}_{date}",
    },
    "transmit": {
        "mode": "C",
        "level": 0.72,
        "optimize": True,
        "cw_id": True,
        "cw_message": "TEST DE {callsign}",
    },
    "ui": {
        # Neither is a default: the default layout is "auto" and the
        # default waterfall height is 0 ("never dragged").
        "layout": "tabs",
        "waterfall_height": 140,
    },
    "version": 2,
}


def _defaults(cpp) -> dict:
    return json.loads(cpp.defaults_json())


def _flatten(data: dict, prefix: str = "") -> dict:
    out = {}
    for key, value in data.items():
        path = f"{prefix}{key}"
        if isinstance(value, dict):
            out.update(_flatten(value, path + "."))
        else:
            out[path] = value
    return out


def test_the_fixture_holds_no_default(native):
    """The guard that makes every other test here mean something.

    `samplerate` is the one exception and is not a real one: it is the
    ring buffer's rate, fixed by the modem, so there is no other legal
    value to put in the fixture. It has its own test below.
    """
    cpp = _cpp(native)
    defaults = _flatten(_defaults(cpp))
    fixture = _flatten(NON_DEFAULT)

    assert set(fixture) == set(defaults), (
        "the fixture and the reader disagree about which settings exist; "
        "a new field must be added here with a non-default value")

    same = {key for key, value in fixture.items()
            if key not in ("version", "audio.samplerate") and defaults[key] == value}
    assert not same, f"these fixture fields are at their default: {sorted(same)}"


def test_a_realistic_hand_edited_config_survives(native):
    """Every field set to something other than its default.

    A round trip of the defaults would pass even if the reader ignored
    the file entirely and returned defaults -- which is exactly the bug
    worth catching. Nothing here equals a default, so a dropped field
    cannot come back looking right.
    """
    cpp = _cpp(native)
    got_text, notes = cpp.round_trip(json.dumps(NON_DEFAULT))
    assert not notes, notes
    assert json.loads(got_text) == NON_DEFAULT


def test_the_defaults_round_trip(native):
    """Weaker than the above by construction, but it is what an operator
    who has never opened the settings dialog is running."""
    cpp = _cpp(native)
    defaults = _defaults(cpp)
    got_text, notes = cpp.round_trip(json.dumps(defaults))
    assert not notes, f"the reader objected to its own defaults: {notes}"
    assert json.loads(got_text) == defaults


def test_a_v1_rig_section_migrates_quietly(native):
    """A config the Python app wrote must not read as a pile of typos.

    The rig section changed shape; the operator did not do anything
    wrong, so the dead keys are recognized-and-ignored rather than
    reported. The one thing that does carry over is the model number,
    which v1 stored as a string.
    """
    cpp = _cpp(native)
    data = copy.deepcopy(NON_DEFAULT)
    data["version"] = 1
    data["rig"] = {
        "enabled": True,
        "host": "10.0.0.5",
        "port": 4533,
        "spawn_local": True,
        "model": "2043",
        "device": "/dev/ttyACM1",
        "baud": 38400,
        "ptt_lead_s": 0.45,
        "ptt_tail_s": 0.25,
        "poll_interval_s": 2.5,
    }

    got_text, notes = cpp.round_trip(json.dumps(data))
    got = json.loads(got_text)
    rig = got["rig"]

    assert not [n for n in notes if n[0].startswith("rig.")], (
        f"migrating a v1 rig section should be quiet: {notes}")
    # A string model number is read as the number it is.
    assert rig["model"] == 2043
    # And the fields that mean the same thing in both schemas survive.
    assert rig["enabled"] is True
    assert rig["device"] == "/dev/ttyACM1"
    assert rig["baud"] == 38400
    assert rig["ptt_lead_s"] == 0.45
    assert rig["poll_interval_s"] == 2.5
    # Migrating the rig section must not disturb anything else.
    assert {k: v for k, v in got.items() if k not in ("rig", "version")} == \
        {k: v for k, v in NON_DEFAULT.items() if k not in ("rig", "version")}


def test_unknown_keys_are_kept_harmless_but_reported(native):
    """An older build reading a newer config.

    It must not fail and must not adopt garbage, but -- unlike the
    reference the port started from -- it should say what it skipped.
    """
    cpp = _cpp(native)
    data = copy.deepcopy(NON_DEFAULT)
    data["some_future_option"] = 42
    data["receive"]["future_nested"] = True

    got_text, notes = cpp.round_trip(json.dumps(data))
    reported = {key for key, _ in notes}
    assert "some_future_option" in reported
    assert "receive.future_nested" in reported
    # The known settings still came through.
    assert json.loads(got_text) == NON_DEFAULT


def test_wrong_types_fall_back_to_defaults_and_are_reported(native):
    cpp = _cpp(native)
    defaults = _defaults(cpp)
    data = copy.deepcopy(defaults)
    data["callsign"] = 12345          # should be a string
    data["rig"]["baud"] = "115200"    # should be an integer
    data["receive"]["autosave"] = "yes"  # should be a boolean
    data["transmit"] = "B"            # should be an object

    got_text, notes = cpp.round_trip(json.dumps(data))
    got = json.loads(got_text)
    reported = {key: problem for key, problem in notes}

    assert "callsign" in reported and "string" in reported["callsign"]
    assert "rig.baud" in reported and "integer" in reported["rig.baud"]
    assert "receive.autosave" in reported
    assert "transmit" in reported
    assert got == defaults, "a bad value should leave the default in place"


def test_a_corrupt_file_still_yields_a_usable_config(native):
    """Loading must never fail: the settings dialog is how it gets fixed."""
    cpp = _cpp(native)
    defaults = _defaults(cpp)
    for broken in ("", "{", "null", "[1,2,3]", "not json at all"):
        got_text, notes = cpp.round_trip(broken)
        assert json.loads(got_text) == defaults
        assert notes, f"{broken!r} parsed silently"


def test_samplerate_is_refused_rather_than_obeyed(native):
    """The one setting where honouring the file would be wrong.

    `samplerate` is the ring buffer's rate, fixed by the modem. A config
    naming anything else fills the ring with wrong-rate audio that
    decodes to nothing, and the symptom looks like a broken radio.
    """
    cpp = _cpp(native)
    from sstvae.config import FS

    data = copy.deepcopy(NON_DEFAULT)
    data["audio"]["samplerate"] = 48000
    got_text, notes = cpp.round_trip(json.dumps(data))

    assert json.loads(got_text)["audio"]["samplerate"] == FS
    assert any("samplerate" in key for key, _ in notes)


def test_atomic_save_round_trips_through_a_real_file(native, tmp_path):
    cpp = _cpp(native)
    data = copy.deepcopy(NON_DEFAULT)
    data["callsign"] = "N6MTS"
    path = tmp_path / "sub" / "config.json"

    assert cpp.save_and_load(json.dumps(data), str(path)) == "N6MTS"
    assert path.exists(), "save should create missing parent directories"
    assert not list(path.parent.glob("*.tmp")), "the temp file should be gone"
    # And what landed on disk is the whole config, not just the field asked for.
    assert json.loads(path.read_text()) == data


# Frozen expected output rather than a second implementation. These are
# what the operator sees in their receive directory, so pinning the
# strings is the point; `{callsign}` alone falling back to the timestamp
# template (rather than producing an empty name) is the case worth
# having written down.
FILENAME_CASES = [
    ("{date}_{time}Z_{freq}_{callsign}", {}, "2026-07-28_011542Z"),
    ("{date}_{time}Z_{freq}_{callsign}", {"callsign": "KC2G"},
     "2026-07-28_011542Z_KC2G"),
    ("{date}_{time}Z_{freq}_{callsign}", {"callsign": "KC2G", "freq_hz": 14340000.0},
     "2026-07-28_011542Z_14.340MHz_KC2G"),
    ("{date}_{time}Z_{freq}_{callsign}", {"freq_hz": 7043500.0},
     "2026-07-28_011542Z_7.043MHz"),
    ("{callsign}_{mode}", {"callsign": "W1AW/2", "mode": "B"}, "W1AW-2_B"),
    ("{callsign}", {}, "2026-07-28_011542Z"),
    ("fixed_name", {}, "fixed_name"),
    ("{unknown}_{callsign}", {"callsign": "KC2G"}, "{unknown}_KC2G"),
]


@pytest.mark.parametrize("template,kwargs,want", FILENAME_CASES)
def test_filename_templates(native, template, kwargs, want):
    from datetime import datetime, timezone

    cpp = _cpp(native)
    when = datetime(2026, 7, 28, 1, 15, 42, tzinfo=timezone.utc)
    got = cpp.format_filename(
        template,
        callsign=kwargs.get("callsign", ""),
        freq_hz=kwargs.get("freq_hz"),
        mode=kwargs.get("mode", ""),
        when=int(when.timestamp()),
    )
    assert got == want


def test_config_path_is_platform_appropriate(native):
    """Compared against **platformdirs**, which is where the path
    convention actually comes from.

    It caught a real bug when it was first written this way:
    platformdirs maps `user_config_dir` to `user_data_dir` on Windows
    with `roaming=False`, so the correct directory is `AppData\\Local`,
    not `AppData\\Roaming`.
    """
    platformdirs = pytest.importorskip(
        "platformdirs", reason="the reference for this path convention")

    cpp = _cpp(native)
    want = platformdirs.user_config_dir("sstvae", appauthor=False)
    assert cpp.config_path() == str(Path(want) / "config.json")


def test_config_dir_honours_xdg_config_home(native, tmp_path, monkeypatch):
    """platformdirs honours XDG_CONFIG_HOME on macOS as well as Linux.

    Skipped on Windows, where it has no meaning and platformdirs ignores
    it. Run in a subprocess because the C++ reads the environment at
    call time and pytest's monkeypatch cannot reach a child's copy
    otherwise -- it can, but only if the child inherits it, which is the
    point being checked.
    """
    import sys

    if sys.platform.startswith("win"):
        pytest.skip("XDG_CONFIG_HOME is not a Windows concept")
    platformdirs = pytest.importorskip("platformdirs")

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    result = subprocess.run(
        [sys.executable, "-c",
         "import sstvae_native; print(sstvae_native.settings.config_path())"],
        capture_output=True, text=True, check=True,
        env={**os.environ, "PYTHONPATH": str(NATIVE_PYTHON_DIR)})

    want = Path(platformdirs.user_config_dir("sstvae", appauthor=False)) / "config.json"
    assert result.stdout.strip() == str(want)
    assert str(tmp_path) in result.stdout

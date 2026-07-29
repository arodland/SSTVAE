"""The C++ config reader against the Python one.

Not a parity test in the sense the modem ones are -- exact behaviour is
not required here. What *is* required is that the two apps can share a
machine: a `config.json` written by either must be understood by the
other, with every setting surviving the trip. Anything else is a
silently reverted preference.

The C++ side adds something the reference does not have: it reports
what it ignored. Python drops unknown or ill-typed keys silently, which
is the right effect (an old build must not wipe a new build's settings)
but makes a typo in a hand-edited config invisible. Those notes are
tested here too, because a diagnostic nobody checks is a diagnostic
that stops being true.
"""

import json
import os
import subprocess
from pathlib import Path

import pytest

from sstvae.gui.settings import Config, format_filename

# Where conftest drops the built extension, for the subprocess test below.
NATIVE_PYTHON_DIR = Path(__file__).resolve().parent.parent / "native" / "build" / "python"


def _cpp(native):
    if not hasattr(native, "settings"):
        pytest.skip("built without the settings module")
    return native.settings


def test_python_defaults_load_into_cpp_unchanged(native):
    """The whole default config, field by field.

    Written as a dict comparison rather than a string one so a
    formatting difference (indentation, key order) does not read as a
    settings difference -- only the values matter.
    """
    cpp = _cpp(native)
    want = Config().to_dict()
    got_text, notes = cpp.round_trip(json.dumps(want))
    got = json.loads(got_text)

    assert not notes, f"C++ objected to the reference defaults: {notes}"
    assert got == want


def test_cpp_defaults_load_into_python_unchanged(native):
    """And the same trip in the other direction."""
    cpp = _cpp(native)
    defaults = json.loads(cpp.defaults_json())
    rebuilt = Config.from_dict(defaults).to_dict()
    assert rebuilt == defaults


def test_a_realistic_hand_edited_config_survives(native):
    """Every field set to something other than its default.

    A round trip of the defaults would pass even if the reader ignored
    the file entirely and returned defaults -- which is exactly the bug
    worth catching. Nothing here equals a default.
    """
    cpp = _cpp(native)
    cfg = Config()
    cfg.callsign = "KC2G"
    cfg.model_path = "/opt/models/v2"
    cfg.precision = "int8"
    cfg.audio.backend = "portaudio"
    cfg.audio.input_device = "USB Audio CODEC"
    cfg.audio.output_device = "SSTVAE-Loopback"
    cfg.rig.enabled = True
    cfg.rig.host = "10.0.0.5"
    cfg.rig.port = 4533
    cfg.rig.spawn_local = True
    cfg.rig.model = "2043"
    cfg.rig.device = "/dev/ttyACM1"
    cfg.rig.baud = 38400
    cfg.rig.ptt_lead_s = 0.45
    cfg.rig.ptt_tail_s = 0.25
    cfg.rig.poll_interval_s = 2.5
    cfg.folders.receive_dir = "/srv/sstv/in"
    cfg.folders.transmit_dir = "/srv/sstv/out"
    cfg.folders.template_dir = "/srv/sstv/tpl"
    cfg.receive.autosave = False
    cfg.receive.low_cpu = True
    cfg.receive.buffer_seconds = 95.5
    cfg.receive.poll_interval = 11.0
    cfg.receive.blind_search_seconds = 30.0
    cfg.receive.end_grace = 6.5
    cfg.receive.save_size = "320x240"
    cfg.receive.save_audio = True
    cfg.receive.filename_template = "{callsign}_{date}"
    cfg.transmit.mode = "C"
    cfg.transmit.level = 0.72

    want = cfg.to_dict()
    got_text, notes = cpp.round_trip(json.dumps(want))
    assert not notes, notes
    assert json.loads(got_text) == want


def test_unknown_keys_are_kept_harmless_but_reported(native):
    """An older build reading a newer config.

    It must not fail and must not adopt garbage, but -- unlike the
    reference -- it should say what it skipped.
    """
    cpp = _cpp(native)
    data = Config().to_dict()
    data["some_future_option"] = 42
    data["receive"]["future_nested"] = True

    got_text, notes = cpp.round_trip(json.dumps(data))
    reported = {key for key, _ in notes}
    assert "some_future_option" in reported
    assert "receive.future_nested" in reported
    # The known settings still came through.
    assert json.loads(got_text) == Config().to_dict()


def test_wrong_types_fall_back_to_defaults_and_are_reported(native):
    cpp = _cpp(native)
    data = Config().to_dict()
    data["callsign"] = 12345          # should be a string
    data["rig"]["port"] = "4532"      # should be an integer
    data["receive"]["autosave"] = "yes"  # should be a boolean
    data["transmit"] = "B"            # should be an object

    got_text, notes = cpp.round_trip(json.dumps(data))
    got = json.loads(got_text)
    reported = {key: problem for key, problem in notes}

    assert "callsign" in reported and "string" in reported["callsign"]
    assert "rig.port" in reported and "integer" in reported["rig.port"]
    assert "receive.autosave" in reported
    assert "transmit" in reported
    assert got == Config().to_dict(), "a bad value should leave the default in place"


def test_a_corrupt_file_still_yields_a_usable_config(native):
    """Loading must never fail: the settings dialog is how it gets fixed."""
    cpp = _cpp(native)
    for broken in ("", "{", "null", "[1,2,3]", "not json at all"):
        got_text, notes = cpp.round_trip(broken)
        assert json.loads(got_text) == Config().to_dict()
        assert notes, f"{broken!r} parsed silently"


def test_samplerate_is_refused_rather_than_obeyed(native):
    """The one setting where honouring the file would be wrong.

    `samplerate` is the ring buffer's rate, fixed by the modem. A config
    naming anything else fills the ring with wrong-rate audio that
    decodes to nothing, and the symptom looks like a broken radio. The
    reference documents this; here it is enforced.
    """
    cpp = _cpp(native)
    from sstvae.config import FS

    data = Config().to_dict()
    data["audio"]["samplerate"] = 48000
    got_text, notes = cpp.round_trip(json.dumps(data))

    assert json.loads(got_text)["audio"]["samplerate"] == FS
    assert any("samplerate" in key for key, _ in notes)


def test_atomic_save_round_trips_through_a_real_file(native, tmp_path):
    cpp = _cpp(native)
    data = Config().to_dict()
    data["callsign"] = "N6MTS"
    path = tmp_path / "sub" / "config.json"

    assert cpp.save_and_load(json.dumps(data), str(path)) == "N6MTS"
    assert path.exists(), "save should create missing parent directories"
    assert not list(path.parent.glob("*.tmp")), "the temp file should be gone"
    # And the file it wrote is readable by the reference.
    assert Config.load(path).callsign == "N6MTS"


@pytest.mark.parametrize("template,kwargs", [
    ("{date}_{time}Z_{freq}_{callsign}", {}),
    ("{date}_{time}Z_{freq}_{callsign}", {"callsign": "KC2G"}),
    ("{date}_{time}Z_{freq}_{callsign}", {"callsign": "KC2G", "freq_hz": 14340000.0}),
    ("{date}_{time}Z_{freq}_{callsign}", {"freq_hz": 7043500.0}),
    ("{callsign}_{mode}", {"callsign": "W1AW/2", "mode": "B"}),
    ("{callsign}", {}),
    ("fixed_name", {}),
    ("{unknown}_{callsign}", {"callsign": "KC2G"}),
])
def test_filename_templates_match_the_reference(native, template, kwargs):
    """Filenames are what the operator actually sees, so these do have
    to agree -- a receive directory where the two apps name files
    differently is a mess to sort out later."""
    from datetime import datetime, timezone

    cpp = _cpp(native)
    when = datetime(2026, 7, 28, 1, 15, 42, tzinfo=timezone.utc)
    want = format_filename(template, when=when, **kwargs)
    got = cpp.format_filename(
        template,
        callsign=kwargs.get("callsign", ""),
        freq_hz=kwargs.get("freq_hz"),
        mode=kwargs.get("mode", ""),
        when=int(when.timestamp()),
    )
    assert got == want


def test_config_path_is_platform_appropriate(native):
    """Both apps must look in the same place, or an operator who runs
    each of them once ends up with two configs and no idea why.

    Compared against **platformdirs**, not against
    `sstvae.gui.settings.config_path()`. That function has an ImportError
    fallback to `~/.config` on every platform, and platformdirs lives in
    the `gui` extra -- so in a `[cli]`-only environment it reports the
    fallback, and asserting against it would pin the native app to a
    directory the GUI never looks in. The first version of this test did
    exactly that and passed on Linux, where the two happen to agree.

    It caught a real bug when fixed: platformdirs maps `user_config_dir`
    to `user_data_dir` on Windows with `roaming=False`, so the correct
    directory is `AppData\\Local`, not `AppData\\Roaming`.
    """
    platformdirs = pytest.importorskip(
        "platformdirs", reason="the reference for this path lives in the gui extra")

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

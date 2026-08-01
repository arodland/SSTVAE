"""Persistent application configuration.

Deliberately plain dataclasses over JSON rather than QSettings: the
config is then readable, diffable, portable between machines, and
loadable by headless code that never starts Qt.

Two robustness rules, both learned from config files that eat
themselves:

* **Writes are atomic** (temp file + os.replace), so losing power or
  hitting a disk-full mid-save leaves the previous config intact rather
  than a truncated one.
* **Unknown keys are ignored, not fatal.** Running an older build
  against a config a newer one wrote must not wipe the operator's
  settings -- worst case a new option reverts to its default.
"""

import json
import os
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import get_type_hints

from ..config import FS

CONFIG_VERSION = 1


def config_dir() -> Path:
    try:
        from platformdirs import user_config_dir
    except ImportError:
        # Good enough on Linux/macOS, and keeps the app importable
        # without the dependency; the GUI extra installs platformdirs.
        base = os.environ.get("XDG_CONFIG_HOME") or (Path.home() / ".config")
        return Path(base) / "sstvae"
    return Path(user_config_dir("sstvae", appauthor=False))


def config_path() -> Path:
    return config_dir() / "config.json"


AUDIO_BACKENDS = ("qt", "portaudio")


@dataclass
class AudioConfig:
    # Which library talks to the soundcard. "qt" (QtMultimedia) is the
    # default because its realtime side is C++ inside Qt, so our Python
    # never sits on the audio thread. A PortAudio *callback* does, and
    # losing the GIL race to Qt's painting drops audio silently on hosts
    # with no buffer behind them (JACK) -- measured at 5 dB of SNR and a
    # mangled picture. See sstvae/gui/qtaudio.py.
    #
    # "portaudio" is kept because the two enumerate different devices:
    # Qt does not list PulseAudio/PipeWire *monitor* sources, so a
    # loopback needs `module-remap-source` to be visible to it, while
    # PortAudio sees monitors directly. Useful for testing.
    backend: str = "qt"
    # Device *description* under the active backend; None = system
    # default. Stored as a name rather than an opaque id so the config
    # stays hand-editable and survives a backend change -- the two
    # backends spell some devices differently, and a name that no longer
    # resolves is kept and flagged rather than silently reset.
    input_device: str | None = None
    output_device: str | None = None
    # What lands in the ring buffer. Fixed by the modem -- this is not a
    # device setting, and changing it produces silent garbage.
    samplerate: int = FS


@dataclass
class RigConfig:
    enabled: bool = False
    host: str = "127.0.0.1"
    port: int = 4532
    spawn_local: bool = False  # start our own rigctld rather than reuse one
    model: str = "1"  # Hamlib rig model number; 1 is the dummy rig
    device: str = "/dev/ttyUSB0"
    baud: int = 19200
    ptt_lead_s: float = 0.3
    ptt_tail_s: float = 0.3
    poll_interval_s: float = 5.0  # frequency readback for filenames/status


@dataclass
class FolderConfig:
    receive_dir: str = str(Path.home() / "SSTVAE" / "received")
    transmit_dir: str = str(Path.home() / "Pictures")
    # Reserved: overlay templates are not implemented yet, but the
    # location is configured now so saved templates land somewhere
    # predictable when they are.
    template_dir: str = str(Path.home() / "SSTVAE" / "templates")


@dataclass
class ReceiveConfig:
    autosave: bool = True
    low_cpu: bool = False
    buffer_seconds: float = 130.0
    poll_interval: float = 5.0
    blind_search_seconds: float = 25.0
    end_grace: float = 8.0
    save_size: str | None = None  # e.g. "320x240"; None keeps 640x480
    # Diagnostic: also write the captured audio beside each received
    # image, as float32 at FS with nothing rescaled. Answers "is the
    # picture bad because the audio was bad?" without needing the
    # hardware again -- the dump decodes offline with sstvae_decode.py.
    save_audio: bool = False
    # Fields available: date, time, freq, callsign, mode. Missing ones
    # (no rig, no callsign decoded) drop out of the name rather than
    # leaving an empty gap.
    filename_template: str = "{date}_{time}Z_{freq}_{callsign}"


@dataclass
class TransmitConfig:
    mode: str = "B"
    level: float = 0.9
    # Transmit-time latent optimization (docs/latent-optimization.md).
    # The **native** app implements it; this GUI is frozen, so the key
    # exists here only so the two agree about the config file -- a
    # config written by one must not read as a typo to the other. That
    # agreement is what `tests/test_native_settings.py` checks.
    optimize: bool = False


@dataclass
class Config:
    callsign: str = ""
    # None = the published ONNX artifacts, fetched and cached on first use.
    # May also be a .pt checkpoint, a .onnx artifact, or a directory of
    # exported .onnx files -- see sstvae.checkpoint.resolve_onnx.
    model_path: str | None = None
    # ONNX precision. Purely local: every precision decodes every other
    # precision's transmissions, so this never has to match the far end.
    # Ignored when model_path names a .pt (that is the torch backend).
    precision: str = "fp16"
    audio: AudioConfig = field(default_factory=AudioConfig)
    rig: RigConfig = field(default_factory=RigConfig)
    folders: FolderConfig = field(default_factory=FolderConfig)
    receive: ReceiveConfig = field(default_factory=ReceiveConfig)
    transmit: TransmitConfig = field(default_factory=TransmitConfig)
    version: int = CONFIG_VERSION

    # --- persistence ---------------------------------------------------
    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Config":
        return _build(cls, data or {})

    @classmethod
    def load(cls, path: Path | None = None) -> "Config":
        """Read the config, falling back to defaults for anything absent
        or unparseable. Never raises: a corrupt config must not stop the
        application from starting, since the settings dialog is how the
        operator would fix it."""
        path = path or config_path()
        try:
            with open(path, encoding="utf-8") as fh:
                return cls.from_dict(json.load(fh))
        except FileNotFoundError:
            return cls()
        except (OSError, ValueError, TypeError):
            return cls()

    def save(self, path: Path | None = None) -> Path:
        path = Path(path or config_path())
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2, sort_keys=False)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)  # atomic on POSIX and Windows
        return path


def codec_precision(config) -> str | None:
    """The `precision=` to hand `codec.load_codec` for this config.

    `None` for a `.pt`, because that selects the torch backend, which has
    no precision and rejects the argument rather than ignoring it. The
    settings dialog greys the control out in the same case, so the user
    sees the same rule the code applies.
    """
    if config.model_path and Path(config.model_path).suffix == ".pt":
        return None
    return config.precision


def _build(cls, data: dict):
    """Construct a (possibly nested) config dataclass from a dict,
    ignoring unknown keys and keeping defaults for missing ones."""
    # Resolved rather than read off `field.type`, which is a *string*
    # whenever annotations are lazy -- then every nested section would
    # silently come back as a plain dict.
    hints = get_type_hints(cls)
    kwargs = {}
    for f in fields(cls):
        if f.name not in data:
            continue
        value = data[f.name]
        hint = hints.get(f.name)
        if is_dataclass(hint) and isinstance(value, dict):
            kwargs[f.name] = _build(hint, value)
        else:
            kwargs[f.name] = value
    return cls(**kwargs)


def format_filename(template: str, *, callsign: str = "", freq_hz: float | None = None,
                    mode: str | None = None, when=None) -> str:
    """Render a received-image filename from the configured template.

    Fields the reception didn't supply are dropped along with any
    separator that would be left stranded, so a decode with no callsign
    and no rig connected yields `20260726_011542Z` rather than
    `20260726_011542Z__`.

    The default template reproduces the naming already used for this
    project's off-air recordings, e.g.
    `2026-07-26_011542Z_14.340MHz_N0CALL.png`.
    """
    from datetime import datetime, timezone

    when = when or datetime.now(timezone.utc)
    values = {
        "date": when.strftime("%Y-%m-%d"),
        "time": when.strftime("%H%M%S"),
        "freq": f"{freq_hz / 1e6:.3f}MHz" if freq_hz else "",
        "callsign": (callsign or "").strip().replace("/", "-"),
        "mode": mode or "",
    }
    parts = []
    for chunk in template.split("_"):
        try:
            rendered = chunk.format(**values)
        except (KeyError, IndexError):
            rendered = chunk  # unknown field: leave the literal text
        if rendered:
            parts.append(rendered)
    return "_".join(parts) or when.strftime("%Y-%m-%d_%H%M%SZ")

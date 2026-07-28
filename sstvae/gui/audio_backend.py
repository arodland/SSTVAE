"""Chooses between the QtMultimedia and PortAudio soundcard backends.

One place that knows both, so the receive and transmit panels do not have
to. See `AudioConfig.backend` for why there are two, and
`sstvae/gui/qtaudio.py` for why Qt is the default.

The two enumerate devices differently and that is not hidden here,
because it is not hideable: Qt does not list PulseAudio/PipeWire
*monitor* sources at all, so a loopback has to be published as a real
source (`module-remap-source`) before Qt can see it, while PortAudio
lists monitors directly. Whichever backend is active, the *name* stored
in the config is what that backend calls the device.
"""

from ..config import FS


def is_qt(config) -> bool:
    return getattr(config.audio, "backend", "qt") == "qt"


def backend_name(config) -> str:
    return "QtMultimedia" if is_qt(config) else "PortAudio"


def list_input_labels(config) -> list[str]:
    """Selectable input device names, for the settings picker."""
    if is_qt(config):
        from .qtaudio import device_labels, list_input_devices

        return device_labels(list_input_devices())
    from ..audio import list_devices

    return [d.name for d in list_devices("input")]


def list_output_labels(config) -> list[str]:
    if is_qt(config):
        from .qtaudio import device_labels, list_output_devices

        return device_labels(list_output_devices())
    from ..audio import list_devices

    return [d.name for d in list_devices("output")]


def open_input_stream(config, ring, on_error=None):
    """Start capture per `config`. Returns (stream, actual_rate).

    Both backends return something with `stop()` and `close()`, which is
    all the receive panel uses.
    """
    audio = config.audio
    if is_qt(config):
        from .qtaudio import open_input_stream as _open

        return _open(audio.input_device or None, ring, audio.samplerate,
                     on_error=on_error)
    from ..audio import open_input_stream as _open

    return _open(audio.input_device or None, ring, audio.samplerate,
                 on_error=on_error)


def player_for(config):
    """The callable to hand `TxEngine(player=...)`.

    `TxEngine` keeps PTT down through its own try/finally and watchdog
    regardless of which player this is, so swapping backends cannot
    weaken that invariant.
    """
    if is_qt(config):
        from .qtaudio import play

        return play
    from ..audio import play

    return play


__all__ = ["FS", "backend_name", "is_qt", "list_input_labels",
           "list_output_labels", "open_input_stream", "player_for"]

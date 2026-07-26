"""Soundcard I/O: device enumeration and stream opening.

Everything here is optional at import time -- `sounddevice` (PortAudio)
is only needed by the live tools, so it is imported inside the functions
rather than at module scope. That keeps `import sstvae.audio` working on
a machine with no PortAudio, which matters because the settings dialog
wants to *report* that rather than fail to start.

The awkward part these helpers exist to hide: SSTVAE runs at 8 kHz, and
plenty of devices refuse that rate outright. PulseAudio and most modern
backends resample transparently, but ALSA hardware devices and some
Windows drivers do not, so both directions fall back to the device's
native rate with polyphase resampling on the way past.
"""

import sys
import threading
from dataclasses import dataclass
from math import gcd

import numpy as np

from .config import FS


class AudioUnavailable(RuntimeError):
    """PortAudio/sounddevice could not be loaded or found no devices."""


@dataclass(frozen=True)
class DeviceInfo:
    index: int
    name: str
    channels: int
    default_samplerate: float
    is_default: bool

    def label(self) -> str:
        star = " (default)" if self.is_default else ""
        return f"{self.index}: {self.name}{star}"


def _sd():
    try:
        import sounddevice as sd
    except Exception as e:  # ImportError, or PortAudio missing at load
        raise AudioUnavailable(
            f"sounddevice/PortAudio is not available: {e}\n"
            "Install the GUI extra (pip install -e .[gui]) and your platform's "
            "PortAudio library."
        ) from e
    return sd


def list_devices(kind: str) -> list[DeviceInfo]:
    """Input or output devices, in PortAudio index order.

    `kind` is "input" or "output". Devices with no channels in the
    requested direction are left out -- a card's playback side has no
    business appearing in the capture picker.
    """
    if kind not in ("input", "output"):
        raise ValueError(f"kind must be 'input' or 'output', got {kind!r}")
    sd = _sd()
    key = f"max_{kind}_channels"
    try:
        default_idx = sd.default.device[0 if kind == "input" else 1]
    except Exception:
        default_idx = None
    out = []
    for i, d in enumerate(sd.query_devices()):
        if d[key] < 1:
            continue
        out.append(
            DeviceInfo(
                index=i,
                name=d["name"],
                channels=d[key],
                default_samplerate=float(d["default_samplerate"]),
                is_default=(i == default_idx),
            )
        )
    if not out:
        raise AudioUnavailable(f"no {kind} devices found")
    return out


def _rate_fallback(device, kind: str) -> int:
    """The device's own preferred sample rate."""
    sd = _sd()
    info = sd.query_devices(device, kind)
    return int(round(info["default_samplerate"]))


def resample_ratio(src_rate: int, dst_rate: int) -> tuple[int, int]:
    """(up, down) for `scipy.signal.resample_poly` to convert audio at
    `src_rate` into audio at `dst_rate`.

    Spelled out with both rates named because the two directions are
    inverses of each other and sharing one "ratio to the device" helper
    between capture and playback silently got playback backwards: a
    transmission was decimated 48000->8000 instead of interpolated
    8000->48000, so 32 seconds of audio became 0.9 seconds of noise.
    """
    g = gcd(src_rate, dst_rate)
    return dst_rate // g, src_rate // g


def open_input_stream(device, ring, samplerate: int = FS, on_error=None):
    """Open an InputStream feeding `ring` (a `sstvae.rx.RingBuffer`).

    Returns (stream, actual_rate). Tries the requested rate directly
    first; falls back to the device's default rate with per-chunk
    polyphase resampling if that is rejected.
    """
    sd = _sd()
    report = on_error or (lambda msg: print(msg, file=sys.stderr))

    def make_callback(resample_fn=None):
        def callback(indata, frames, time_info, status):
            if status:
                report(f"[audio in] {status}")
            mono = indata[:, 0] if indata.ndim > 1 else indata
            ring.write(resample_fn(mono) if resample_fn else mono)

        return callback

    try:
        stream = sd.InputStream(
            samplerate=samplerate, channels=1, dtype="float32",
            device=device, callback=make_callback(),
        )
        stream.start()
        return stream, samplerate
    except Exception as e:
        from scipy.signal import resample_poly

        native = _rate_fallback(device, "input")
        # Capture direction: the device hands us `native`, we want `samplerate`.
        up, down = resample_ratio(native, samplerate)
        report(
            f"[audio in] {samplerate} Hz rejected ({e}); falling back to "
            f"device default {native} Hz with resampling"
        )
        stream = sd.InputStream(
            samplerate=native, channels=1, dtype="float32", device=device,
            callback=make_callback(lambda x: resample_poly(x, up, down)),
        )
        stream.start()
        return stream, native


def play(device, samples: np.ndarray, samplerate: int = FS, on_progress=None,
         should_stop=None, on_error=None) -> bool:
    """Play `samples` (float, at `samplerate`) to `device`, blocking
    until it has finished or been stopped.

    `on_progress(fraction)` is called from the audio callback as the
    buffer drains, `should_stop()` is polled to allow cancellation.
    Returns True if playback completed, False if it was stopped early.

    Resamples up front rather than per-chunk when the device rejects the
    requested rate -- unlike capture, the whole waveform is already in
    hand, and one clean resample avoids polyphase edge effects at every
    block boundary.
    """
    sd = _sd()
    report = on_error or (lambda msg: print(msg, file=sys.stderr))
    x = np.asarray(samples, dtype=np.float32).reshape(-1)

    try:
        stream_rate = samplerate
        sd.check_output_settings(device=device, samplerate=samplerate,
                                 channels=1, dtype="float32")
    except Exception as e:
        from scipy.signal import resample_poly

        native = _rate_fallback(device, "output")
        # Playback direction: we hold `samplerate`, the device wants
        # `native` -- the opposite of the capture case above.
        up, down = resample_ratio(samplerate, native)
        report(
            f"[audio out] {samplerate} Hz rejected ({e}); resampling to "
            f"device default {native} Hz"
        )
        x = resample_poly(x, up, down).astype(np.float32)
        stream_rate = native

    total = len(x)
    pos = 0
    done = threading.Event()

    def callback(outdata, frames, time_info, status):
        nonlocal pos
        if status:
            report(f"[audio out] {status}")
        if should_stop is not None and should_stop():
            outdata[:] = 0
            raise sd.CallbackAbort
        n = min(frames, total - pos)
        outdata[:n, 0] = x[pos : pos + n]
        if n < frames:
            outdata[n:, 0] = 0
        pos += n
        if on_progress is not None:
            on_progress(pos / total if total else 1.0)
        if pos >= total:
            raise sd.CallbackStop

    stream = sd.OutputStream(
        samplerate=stream_rate, channels=1, dtype="float32",
        device=device, callback=callback, finished_callback=done.set,
    )
    with stream:
        # Poll rather than block forever: a device that stops calling
        # back (USB unplugged mid-transmission) must not wedge the
        # caller, because the caller is what drops PTT.
        while not done.wait(0.1):
            if should_stop is not None and should_stop():
                break
            if not stream.active:
                break
    return pos >= total

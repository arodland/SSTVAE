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

# Host APIs where a Python capture callback is not safe to rely on.
#
# Our PortAudio callback is Python, so it runs on the host's realtime
# audio thread and must take the GIL to do anything at all. Any other
# thread holding the GIL -- the Qt thread converting a 640x480 preview to
# a QPixmap and painting it, right after every decode poll -- stops it
# from running. Hosts with a large software buffer (PulseAudio, PipeWire's
# own device) absorb that invisibly. **JACK does not**: its period is a
# couple of milliseconds with nothing queued behind it, so the audio is
# skipped, silently, with no status flag to report.
#
# Measured on a PipeWire-JACK device: ~200-350 samples lost on every
# decode poll, tracking `poll_interval` exactly (change the interval and
# the losses follow it), costing 5 dB of SNR and a mangled picture, while
# still syncing and reporting every frame received. The identical code was
# clean headless, which is what made it look like a GUI decode bug.
#
# The proper fix is to keep Python off the realtime thread. PortAudio's
# *blocking* API would do that -- its own C callback fills an internal
# ring -- but `stream.read()` **corrupts the heap on the JACK backend**
# (`malloc(): invalid size`) at every blocksize and latency tried, so it
# is not available. The real fix is a capture layer whose realtime side
# is not Python at all; see docs/native-app.md on QtMultimedia.
FRAGILE_HOST_APIS = ("jack",)


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


def host_api_name(device) -> str | None:
    """The PortAudio host API a device belongs to, lowercased."""
    sd = _sd()
    try:
        info = sd.query_devices(device, "input")
        return sd.query_hostapis(info["hostapi"])["name"].lower()
    except Exception:
        return None


def warn_if_fragile_host(device, report) -> str | None:
    """Say so if this device's host API cannot be captured reliably.

    Worth a warning rather than a refusal: it *mostly* works, and a
    user who has deliberately chosen JACK may know what they are doing.
    But the failure mode is a quietly mangled picture, which nobody
    would attribute to their device choice unaided. See
    `FRAGILE_HOST_APIS`.
    """
    api = host_api_name(device)
    if api and any(bad in api for bad in FRAGILE_HOST_APIS):
        report(
            f"[audio in] this device is on the '{api}' host API, which has no "
            "buffering behind its realtime period. Pictures may decode noisy "
            "or mangled even though sync succeeds. Prefer the 'pulse' or "
            "'pipewire' entry for the same device."
        )
        return api
    return None


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


class StreamResampler:
    """`resample_poly` for a stream that arrives in chunks.

    **Calling `resample_poly` on each callback chunk independently is
    wrong**, and wrong in a way that decodes rather than fails. It is an
    FIR polyphase filter, so an isolated chunk is zero-padded at both
    ends and every chunk boundary gets a filter transient; at 44.1 kHz
    into 8 kHz the filter is 8821 taps against ~186 output samples per
    chunk. On a real 66 s recording that cost **4.7 dB of SNR** (+2.4 dB
    one-shot, -2.3 dB per-chunk) and mangled the picture, while still
    syncing and reporting every frame received. `play()` sidesteps this
    by resampling the whole waveform up front; capture cannot.

    Per-chunk rounding is the second half of the bug: each chunk's output
    length is rounded up independently, so the stream gains samples --
    684 of them over that same recording, a 0.13% clock error the timing
    tracker then has to fight.

    This keeps `pad` input samples of context on *each* side of every
    block it emits, and only emits whole multiples of `down` input
    samples, so the output is sample-for-sample what one-shot resampling
    of the whole stream would have produced. Costs `2 * pad` input
    samples of latency, ~20 ms at 44.1 kHz.
    """

    def __init__(self, up: int, down: int):
        self.up = up
        self.down = down
        # scipy's default filter is 20*max(up, down)+1 taps in the
        # upsampled domain; this is its span measured in *input* samples.
        span = -(-(20 * max(up, down) + 1) // up)
        # Round up to a whole number of `down`, so the samples skipped at
        # the head are an exact integer and no phase error accumulates.
        self.pad = -(-span // down) * down
        self._buf = np.zeros(self.pad, dtype=np.float64)

    def __call__(self, chunk: np.ndarray) -> np.ndarray:
        from scipy.signal import resample_poly

        self._buf = np.concatenate(
            [self._buf, np.asarray(chunk, dtype=np.float64).reshape(-1)])
        usable = len(self._buf) - 2 * self.pad
        n = (usable // self.down) * self.down if usable > 0 else 0
        if n <= 0:
            return np.empty(0)
        block = self._buf[: 2 * self.pad + n]
        y = resample_poly(block, self.up, self.down)
        skip = self.pad * self.up // self.down
        take = n * self.up // self.down
        self._buf = self._buf[n:]
        return y[skip:skip + take]


def open_input_stream(device, ring, samplerate: int = FS, on_error=None):
    """Open an InputStream feeding `ring` (a `sstvae.rx.RingBuffer`).

    Returns (stream, actual_rate).

    **Opens at the device's own rate and resamples here, rather than
    asking the device for 8 kHz.** Almost no capture hardware is natively
    8 kHz, so requesting it does not avoid a resampler -- it just hands
    the job to whichever one happens to be in the audio stack, and their
    quality varies enormously. JACK cannot resample at all: a JACK stream
    only ever runs at the server's rate, whatever was asked for. Doing
    the conversion in `StreamResampler` makes capture quality a property
    of this code rather than of the user's audio configuration.

    `samplerate` is what lands in `ring`, and the modem fixes it at `FS`
    -- it is not a device setting and changing it produces silent
    garbage.

    This uses PortAudio's **callback** API. The blocking API would keep
    Python off the host's realtime thread, which would be better -- see
    the note on `warn_if_fragile_host` -- but `stream.read()` corrupts
    the heap on PortAudio's JACK backend at every blocksize and latency
    tried, so it is not an option.
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

    def open_at(rate: int):
        resampler = None
        if rate != samplerate:
            resampler = StreamResampler(*resample_ratio(rate, samplerate))
        stream = sd.InputStream(
            samplerate=rate, channels=1, dtype="float32", device=device,
            callback=make_callback(resampler),
        )
        stream.start()
        return stream, rate

    warn_if_fragile_host(device, report)

    native = None
    try:
        native = _rate_fallback(device, "input")
    except Exception as e:  # device gone, or no rate advertised
        report(f"[audio in] could not query the device's rate ({e})")

    if native and native != samplerate:
        try:
            return open_at(native)
        except Exception as e:
            report(
                f"[audio in] {native} Hz (the device's own rate) failed ({e}); "
                f"asking for {samplerate} Hz instead \u2014 note this leaves the "
                "rate conversion to the audio stack"
            )

    try:
        return open_at(samplerate)
    except Exception as e:
        if native is None or native == samplerate:
            raise
        report(f"[audio in] {samplerate} Hz rejected ({e}); using {native} Hz")
        return open_at(native)


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

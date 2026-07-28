"""Soundcard capture and playback through QtMultimedia.

**Why this exists rather than PortAudio.** A `sounddevice` callback is a
Python function running on the host's realtime audio thread, so it has to
take the GIL before it can do anything. Whenever another thread holds the
GIL -- the Qt thread converting a 640x480 preview to a QPixmap and
painting it, right after every decode poll -- the callback cannot run. A
host with a large software buffer absorbs that invisibly; JACK, whose
period is a couple of milliseconds with nothing queued behind it, simply
loses the audio, silently, with no status flag. Measured cost on a
PipeWire-JACK device: 200-350 samples per decode poll, 5 dB of SNR and a
mangled picture, while sync succeeded and every frame was reported.

`QAudioSource` is pull-based: Qt's C++ backend fills an internal buffer
on the realtime thread and we drain it from the event loop whenever we
get round to it. Python is off the realtime path entirely, which is a
structural fix rather than a tuning exercise. PortAudio's blocking API
would have achieved the same thing, but `stream.read()` corrupts the heap
on its JACK backend at every blocksize and latency tried.

This module is the *only* place in the send/receive path that may import
Qt, which is why it lives under `sstvae/gui/`. Everything with real logic
in it -- `bytes_to_mono`, `match_device`, and the resampling itself --
is deliberately Qt-free and tested without a QApplication.

Two things carried over from the PortAudio implementation because they
are properties of the problem, not of the library:

- **Open at the device's own rate and resample here.** Qt will happily
  accept an 8 kHz `QAudioFormat` and let its backend convert; that hands
  the job to a resampler whose quality we do not control. See
  `sstvae.audio.StreamResampler`.
- **Resampling must be stateful across reads.** Per-block
  `resample_poly` cost 4.7 dB on a real recording.
"""

import time

import numpy as np
from PySide6.QtCore import QObject
from PySide6.QtMultimedia import (
    QAudioFormat,
    QAudioSink,
    QAudioSource,
    QMediaDevices,
)

from ..audio import StreamResampler, resample_ratio
from ..config import FS

# How much audio Qt should buffer for us. This is the slack that makes
# the whole design work: we can be this far late in draining it without
# losing a sample, which comfortably covers a decode poll's ~170 ms of
# demodulate plus the preview conversion that follows it.
#
# Measured on a K4 RX A capture with a thread deliberately holding the
# GIL: clean through 400 ms of blocking at 1.0 s of buffer, and losing
# samples at 800 ms. 2 s is therefore generous rather than tight, and it
# costs 384 KB. Nothing here is latency-sensitive -- the decode loop polls
# every 5 seconds.
BUFFER_SECONDS = 2.0

# Qt sample formats we can convert, and how to scale them to [-1, 1].
_FORMATS = {
    "Float": (np.float32, 1.0, 0.0),
    "Int16": (np.int16, 32768.0, 0.0),
    "Int32": (np.int32, 2147483648.0, 0.0),
    "UInt8": (np.uint8, 128.0, 128.0),
}


def bytes_to_mono(raw, fmt: str, channels: int) -> np.ndarray:
    """Raw interleaved device bytes -> mono float64 in [-1, 1].

    Takes the format as a plain string so this is testable without Qt.
    Channels are mixed down rather than picking the first, so a device
    that puts the signal only on the right channel still works.
    """
    if fmt not in _FORMATS:
        raise ValueError(f"unsupported sample format {fmt!r}")
    dtype, scale, offset = _FORMATS[fmt]
    a = np.frombuffer(raw, dtype=dtype)
    if channels > 1:
        # Drop a trailing partial frame rather than misaligning every
        # sample after it.
        a = a[: len(a) // channels * channels].reshape(-1, channels)
        a = a.mean(axis=1)
    return (a.astype(np.float64) - offset) / scale


def match_device(descriptions: list[str], wanted: str | None) -> int | None:
    """Index of the device to use, or None for "the system default".

    Matching is by description rather than by Qt's opaque device id,
    because the id is not stable across backends and the config file has
    to stay human-editable. Exact match wins; otherwise a unique
    case-insensitive substring match, so a saved "K4 RX A" still finds
    "K4 RX A" after Qt decorates the name.
    """
    if not wanted:
        return None
    for i, d in enumerate(descriptions):
        if d == wanted:
            return i
    low = wanted.lower()
    hits = [i for i, d in enumerate(descriptions) if low in d.lower()]
    return hits[0] if len(hits) == 1 else None


def list_input_devices():
    return list(QMediaDevices.audioInputs())


def list_output_devices():
    return list(QMediaDevices.audioOutputs())


def device_labels(devices) -> list[str]:
    return [d.description() for d in devices]


def _format_name(sample_format) -> str:
    return getattr(sample_format, "name", str(sample_format))


def choose_format(device, rate: int) -> QAudioFormat:
    """A mono format at `rate` that the device actually accepts.

    Float first because it needs no scaling; Int16 is the common
    fallback and is what most radio interfaces advertise.
    """
    for candidate in ("Float", "Int16", "Int32", "UInt8"):
        sf = getattr(QAudioFormat.SampleFormat, candidate, None)
        if sf is None:
            continue
        fmt = QAudioFormat()
        fmt.setSampleRate(rate)
        fmt.setChannelCount(1)
        fmt.setSampleFormat(sf)
        if device.isFormatSupported(fmt):
            return fmt
    # Nothing mono worked; fall back to the device's own preference and
    # mix down whatever it gives us.
    return device.preferredFormat()


class QtInputStream(QObject):
    """Captures into a `sstvae.rx.RingBuffer`.

    Deliberately API-compatible with what `sstvae.audio.open_input_stream`
    returned -- `stop()` and `close()` -- so the receive panel does not
    care which backend it got.
    """

    def __init__(self, device, ring, samplerate: int = FS, on_error=None,
                 parent=None):
        super().__init__(parent)
        self._ring = ring
        self._report = on_error or (lambda msg: None)
        self._resampler = None
        self._samples = 0
        self._last_error = "NoError"

        rate = device.preferredFormat().sampleRate() or samplerate
        fmt = choose_format(device, rate)
        self.rate = fmt.sampleRate()
        self._fmt_name = _format_name(fmt.sampleFormat())
        self._channels = fmt.channelCount()
        if self.rate != samplerate:
            self._resampler = StreamResampler(*resample_ratio(self.rate, samplerate))

        self._source = QAudioSource(device, fmt, self)
        self._source.setBufferSize(
            int(BUFFER_SECONDS * self.rate * self._channels * fmt.bytesPerSample())
        )
        # No `errorOccurred` signal on QAudioSource in PySide6; errors
        # surface as a state change with `error()` set.
        # `stateChanged` is deliberately *not* connected: PySide cannot
        # marshal `QAudio::State` into any Python slot in this build
        # ("parameter 0 ... cannot be converted"), not even a *args
        # lambda. `error()` is a cheap getter, so the read path polls it
        # instead and we lose nothing.
        self._io = self._source.start()
        if self._io is None:
            raise RuntimeError(
                f"could not start capture on {device.description()!r}")
        self._io.readyRead.connect(self._on_ready)

    def _check_error(self) -> None:
        err = self._source.error()
        name = getattr(err, "name", str(err))
        if name != "NoError" and name != self._last_error:
            self._last_error = name
            self._report(f"[audio in] {device_error_text(name)}")

    def _on_ready(self) -> None:
        self._check_error()
        raw = self._io.readAll()
        if raw.isEmpty():
            return
        mono = bytes_to_mono(bytes(raw.data()), self._fmt_name, self._channels)
        if mono.size == 0:
            return
        self._samples += mono.size
        self._ring.write(self._resampler(mono) if self._resampler else mono)

    @property
    def samples_captured(self) -> int:
        return self._samples

    def stop(self) -> None:
        try:
            self._io.readyRead.disconnect(self._on_ready)
        except (RuntimeError, TypeError):
            pass
        try:
            self._source.stop()
        except RuntimeError:
            pass

    def close(self) -> None:
        self.stop()


def device_error_text(name: str) -> str:
    return {
        "OpenError": "could not open the audio device (in use, or gone?)",
        "IOError": "audio device I/O error; the device may have been unplugged",
        "UnderrunError": "audio underrun",
        "FatalError": "the audio device stopped working",
    }.get(name, f"audio error: {name}")


def _write_all(io, sink, data: bytes, frame_bytes: int, on_progress, should_stop,
               deadline: float) -> bool:
    """Push `data` into `io`, pacing on `bytesFree`. True if it all went.

    Push mode rather than pull: the transmit engine calls this from its
    own worker thread, which has no Qt event loop, so nothing would
    deliver a `readyRead`-style signal. Explicit writes need no event
    loop at all.
    """
    total = len(data)
    pos = 0
    while pos < total:
        if should_stop and should_stop():
            return False
        if time.monotonic() > deadline:
            return False
        free = sink.bytesFree()
        if free <= 0:
            time.sleep(0.005)
            continue
        # Whole frames only; a partial frame would desync the stream.
        chunk = (min(free, total - pos) // frame_bytes) * frame_bytes
        if chunk <= 0:
            time.sleep(0.005)
            continue
        written = io.write(data[pos:pos + chunk])
        if written < 0:
            return False
        pos += written
        if on_progress:
            on_progress(min(1.0, pos / total))
        time.sleep(0.001)
    return True


def play(device_name, samples: np.ndarray, samplerate: int = FS,
         on_progress=None, should_stop=None, on_error=None) -> bool:
    """Play `samples` (float, at `samplerate`) to the named output device.

    Signature matches `sstvae.audio.play` so it can be injected into
    `TxEngine` through its existing `player` seam -- which means the
    PTT-always-comes-back-down invariant is unaffected: that lives in
    `TxEngine`'s try/finally plus its independent watchdog, and does not
    depend on which player is in use.

    Resamples up front rather than per chunk. Unlike capture the whole
    waveform is in hand, so one clean conversion avoids polyphase edge
    effects entirely -- and note the K4's transmit device advertises
    12 kHz, so this path is not hypothetical.
    """
    report = on_error or (lambda msg: None)
    devices = list_output_devices()
    if not devices:
        raise RuntimeError("no audio output devices found")
    idx = match_device(device_labels(devices), device_name)
    if device_name and idx is None:
        report(f"[audio out] no output device matching {device_name!r}; "
               "using the system default")
    device = devices[idx] if idx is not None else QMediaDevices.defaultAudioOutput()

    x = np.asarray(samples, dtype=np.float64).reshape(-1)
    fmt = choose_format(device, samplerate)
    rate = fmt.sampleRate()
    if rate != samplerate:
        from scipy.signal import resample_poly

        up, down = resample_ratio(samplerate, rate)
        x = resample_poly(x, up, down)
        report(f"[audio out] {device.description()} wants {rate} Hz; "
               f"resampled from {samplerate} Hz")

    fmt_name = _format_name(fmt.sampleFormat())
    if fmt_name not in _FORMATS:
        raise RuntimeError(f"unsupported output format {fmt_name}")
    dtype, scale, offset = _FORMATS[fmt_name]
    if fmt_name == "Float":
        buf = np.clip(x, -1.0, 1.0).astype(np.float32)
    else:
        # Scale then clip in the integer domain, so a sample at exactly
        # +1.0 does not wrap to full-scale negative.
        buf = np.clip(np.round(x * (scale - 1) + offset),
                      np.iinfo(dtype).min, np.iinfo(dtype).max).astype(dtype)
    if fmt.channelCount() > 1:
        buf = np.repeat(buf[:, None], fmt.channelCount(), axis=1).reshape(-1)

    sink = QAudioSink(device, fmt)
    sink.setBufferSize(int(BUFFER_SECONDS * rate * fmt.channelCount()
                           * fmt.bytesPerSample()))
    io = sink.start()
    if io is None:
        raise RuntimeError(f"could not start playback on {device.description()!r}")
    frame_bytes = max(1, fmt.channelCount() * fmt.bytesPerSample())
    # Generous ceiling so a wedged device cannot block transmit forever.
    # TxEngine's watchdog is the real backstop; this keeps us from
    # relying on it in the ordinary case.
    deadline = time.monotonic() + len(x) / max(rate, 1) + 30.0
    try:
        done = _write_all(io, sink, buf.tobytes(), frame_bytes,
                          on_progress, should_stop, deadline)
        if done:
            # Let the device drain what is still buffered, or the tail of
            # the transmission is cut off mid-picture.
            drain_until = time.monotonic() + BUFFER_SECONDS + 1.0
            while (sink.bytesFree() < sink.bufferSize()
                   and time.monotonic() < drain_until):
                if should_stop and should_stop():
                    return False
                time.sleep(0.01)
        return done
    finally:
        try:
            sink.stop()
        except RuntimeError:
            pass


def open_input_stream(device_name, ring, samplerate: int = FS, on_error=None):
    """Open capture on the named device. Returns (stream, actual_rate).

    Mirrors `sstvae.audio.open_input_stream`'s signature so the receive
    panel is backend-agnostic. `device_name` is a description as shown by
    `device_labels`, or None for the system default.
    """
    report = on_error or (lambda msg: None)
    devices = list_input_devices()
    if not devices:
        raise RuntimeError("no audio input devices found")
    idx = match_device(device_labels(devices), device_name)
    if device_name and idx is None:
        report(
            f"[audio in] no input device matching {device_name!r}; "
            "using the system default"
        )
    device = devices[idx] if idx is not None else QMediaDevices.defaultAudioInput()
    stream = QtInputStream(device, ring, samplerate, on_error=report)
    report(f"[audio in] {device.description()} at {stream.rate} Hz"
           + ("" if stream.rate == samplerate else f", resampled to {samplerate} Hz"))
    return stream, stream.rate

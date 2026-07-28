"""Sample-rate conversion around devices that refuse 8 kHz.

The bug these guard against: capture and playback need *inverse*
resampling ratios, and a single shared "ratio to the device" helper got
playback backwards. A 32-second transmission was decimated 48000->8000
instead of interpolated 8000->48000, so it played as 0.9 seconds of
noise -- the rig keyed, the progress bar shot to 100%, and nothing
intelligible went out.

Nothing here touches real hardware: `sstvae.audio._sd` is replaced with
a fake PortAudio that rejects 8 kHz, which is exactly what an Elecraft
K4's USB codec (and most ALSA hardware devices) does.
"""

import numpy as np
import pytest

from sstvae import audio
from sstvae.config import FS


# --- the ratio itself ---------------------------------------------------

def test_ratio_upsamples_for_playback():
    """8 kHz out to a 48 kHz device must interpolate by 6, not decimate."""
    up, down = audio.resample_ratio(8000, 48000)
    assert (up, down) == (6, 1)


def test_ratio_downsamples_for_capture():
    assert audio.resample_ratio(48000, 8000) == (1, 6)


def test_ratio_directions_are_inverses():
    for native in (44100, 48000, 96000, 22050):
        up, down = audio.resample_ratio(FS, native)
        assert audio.resample_ratio(native, FS) == (down, up)


@pytest.mark.parametrize("native", [44100, 48000, 96000, 22050, 16000])
def test_resampling_preserves_duration(native):
    """The property that actually matters: a signal keeps its wall-clock
    length through the conversion."""
    from scipy.signal import resample_poly

    seconds = 4.0
    x = np.sin(2 * np.pi * 1500 * np.arange(int(seconds * FS)) / FS)
    up, down = audio.resample_ratio(FS, native)
    y = resample_poly(x, up, down)
    assert y.size / native == pytest.approx(seconds, rel=0.01)


# --- the playback path --------------------------------------------------

class FakePortAudio:
    """Minimal sounddevice stand-in for a device that only does 48 kHz."""

    class CallbackStop(Exception):
        pass

    class CallbackAbort(Exception):
        pass

    def __init__(self, native=48000, accepts=()):
        self.native = native
        self.accepts = set(accepts)
        self.played = []  # every block handed to the device
        self.stream_rate = None
        self.closed = False
        self.hostapi_name = "ALSA"

    def check_output_settings(self, device=None, samplerate=None, **kw):
        if samplerate not in self.accepts:
            raise RuntimeError(f"Invalid sample rate {samplerate}")

    def query_devices(self, device=None, kind=None):
        return {"default_samplerate": float(self.native), "name": "Fake TX",
                "hostapi": 0}

    def OutputStream(self, samplerate=None, callback=None,
                     finished_callback=None, **kw):
        self.stream_rate = samplerate
        fake = self

        class Stream:
            active = True

            def __enter__(self_inner):
                # Drain the whole buffer the way a real device would.
                blocks = 0
                while blocks < 100_000:
                    out = np.zeros((1024, 1))
                    try:
                        callback(out, 1024, None, None)
                    except (FakePortAudio.CallbackStop, FakePortAudio.CallbackAbort):
                        fake.played.append(out.copy())
                        break
                    fake.played.append(out.copy())
                    blocks += 1
                self_inner.active = False
                if finished_callback:
                    finished_callback()
                return self_inner

            def __exit__(self_inner, *exc):
                return False

        return Stream()

    def query_hostapis(self, index=None):
        return {"name": self.hostapi_name}

    def InputStream(self, samplerate=None, callback=None, **kw):
        """Capture counterpart. Rejects rates the device doesn't do, which
        is what pushes `open_input_stream` onto its resampling path."""
        if samplerate not in self.accepts and samplerate != self.native:
            raise RuntimeError(f"Invalid sample rate {samplerate}")
        self.stream_rate = samplerate
        self.capture_callback = callback
        fake = self

        class Stream:
            def start(self_inner):
                fake.started = True

            def stop(self_inner):
                fake.closed = True

            def close(self_inner):
                fake.closed = True

        return Stream()

    def feed(self, signal, blocksize=1024):
        """Hand `signal` to the capture callback in blocks, as a device would."""
        for i in range(0, len(signal), blocksize):
            block = np.asarray(signal[i:i + blocksize], dtype=np.float32)
            self.capture_callback(block.reshape(-1, 1), len(block), None, None)

    def total_samples_played(self) -> int:
        return sum(b.shape[0] for b in self.played)


@pytest.fixture
def fake_pa(monkeypatch):
    pa = FakePortAudio()
    monkeypatch.setattr(audio, "_sd", lambda: pa)
    return pa


def test_playback_to_a_48k_only_device_keeps_the_right_duration(fake_pa):
    """The regression: 32 s of audio must still take 32 s of device time."""
    seconds = 4.0
    x = np.sin(2 * np.pi * 1500 * np.arange(int(seconds * FS)) / FS)

    assert audio.play(None, x, samplerate=FS) is True

    assert fake_pa.stream_rate == 48000, "should have fallen back to the native rate"
    played_seconds = fake_pa.total_samples_played() / fake_pa.stream_rate
    assert played_seconds == pytest.approx(seconds, rel=0.02), (
        f"played {played_seconds:.2f} s of audio, expected {seconds:.2f} s -- "
        "the resample ratio is inverted"
    )


def test_playback_survives_the_resampling_intact(fake_pa):
    """Duration alone isn't enough: the tone has to still be there.

    Decimating 8 kHz by 6 destroys a 1500 Hz carrier entirely, so this
    fails loudly on the inverted ratio even if the length were fixed.
    """
    x = np.sin(2 * np.pi * 1500 * np.arange(2 * FS) / FS)
    audio.play(None, x, samplerate=FS)

    out = np.concatenate([b[:, 0] for b in fake_pa.played])
    spec = np.abs(np.fft.rfft(out * np.hanning(len(out))))
    peak_hz = np.argmax(spec) * fake_pa.stream_rate / len(out)
    assert peak_hz == pytest.approx(1500, abs=20), (
        f"dominant tone came out at {peak_hz:.0f} Hz, not 1500 Hz"
    )


def test_no_resampling_when_the_device_accepts_8k(fake_pa):
    fake_pa.accepts = {FS}
    x = np.zeros(FS)
    audio.play(None, x, samplerate=FS)
    assert fake_pa.stream_rate == FS
    assert fake_pa.total_samples_played() == pytest.approx(FS, rel=0.2)


def test_progress_reaches_one(fake_pa):
    seen = []
    audio.play(None, np.zeros(2 * FS), samplerate=FS, on_progress=seen.append)
    assert seen and seen[-1] == pytest.approx(1.0)


# --- capture-side resampling ---------------------------------------------
#
# The regression these guard against decoded rather than failed: a
# recording that `sstvae_decode.py` handled cleanly came out of the live
# receiver as a mangled picture, still reporting every frame received.
# `resample_poly` was being called on each callback chunk independently,
# so every chunk boundary carried a polyphase transient (8821 taps against
# ~186 output samples at 44.1 kHz) and each chunk's length was rounded up
# on its own. Cost: 4.7 dB of SNR and 684 spurious samples over 66 s.

RATE_PAIRS = [(44100, FS), (48000, FS), (22050, FS), (16000, FS)]


@pytest.mark.parametrize("native,target", RATE_PAIRS)
@pytest.mark.parametrize("blocksize", [735, 1024, 4096])
def test_streaming_resampler_matches_one_shot_exactly(native, target, blocksize):
    """Chunking must not change a single sample.

    Exact equality is the right bar: this is the same filter over the
    same data, so any difference at all is a boundary artifact.
    """
    from scipy.signal import resample_poly

    rng = np.random.default_rng(0)
    t = np.arange(int(native * 1.5)) / native
    x = (np.sin(2 * np.pi * 1500 * t) + 0.3 * rng.standard_normal(len(t)))

    up, down = audio.resample_ratio(native, target)
    want = resample_poly(x, up, down)

    rs = audio.StreamResampler(up, down)
    got = np.concatenate([rs(x[i:i + blocksize])
                          for i in range(0, len(x), blocksize)])

    assert len(got) <= len(want)
    # Only the unflushed tail may be missing, never more.
    assert len(want) - len(got) < 2 * rs.pad * up // down + down
    assert np.array_equal(got, want[:len(got)])


@pytest.mark.parametrize("native,target", RATE_PAIRS)
def test_streaming_resampler_does_not_drift(native, target):
    """Output length must track input length, not accumulate rounding."""
    up, down = audio.resample_ratio(native, target)
    rs = audio.StreamResampler(up, down)
    n = 0
    for _ in range(200):
        n += len(rs(np.zeros(1024)))
    expected = 200 * 1024 * up / down
    assert abs(n - expected) < 2 * rs.pad * up / down + down, (
        "per-chunk rounding is accumulating a clock error"
    )


def test_capture_from_a_44k_only_device_reproduces_one_shot(fake_pa):
    """End to end through `open_input_stream`, via the fake device."""
    from scipy.signal import resample_poly

    from sstvae.rx import RingBuffer

    fake_pa.native = 44100
    fake_pa.accepts = {44100}

    t = np.arange(44100) / 44100
    x = np.sin(2 * np.pi * 1200 * t)

    ring = RingBuffer(10.0)
    stream, rate = audio.open_input_stream(None, ring, FS, on_error=lambda m: None)
    assert rate == 44100, "should have fallen back to the device rate"

    fake_pa.feed(x, blocksize=1024)
    up, down = audio.resample_ratio(44100, FS)
    want = resample_poly(x, up, down)
    got, total = ring.snapshot()
    assert total > 0
    assert np.allclose(got[:total], want[:total], atol=1e-6), (
        "captured audio differs from one-shot resampling -- chunk-boundary "
        "artifacts are back"
    )


def test_capture_prefers_the_device_rate_even_when_8k_is_offered(fake_pa):
    """The PipeWire/ALSA regression.

    Asking a device for 8 kHz does not avoid a resampler -- it delegates
    to whatever the audio stack provides, and ALSA's default is linear
    interpolation. The same loopback decoded cleanly via PulseAudio and
    produced a mangled picture via PipeWire/ALSA. So we open at the
    device's own rate and convert here, where the quality is ours.
    """
    from sstvae.rx import RingBuffer

    fake_pa.native = 48000
    fake_pa.accepts = {FS, 48000}  # device would happily give us 8 kHz

    ring = RingBuffer(10.0)
    _, rate = audio.open_input_stream(None, ring, FS, on_error=lambda m: None)

    assert rate == 48000, (
        "should open at the device's native rate and resample in "
        "StreamResampler, not let the audio stack convert to 8 kHz"
    )
    assert fake_pa.stream_rate == 48000


def test_capture_opens_directly_when_the_device_is_natively_8k(fake_pa):
    """No pointless conversion when the device really is 8 kHz."""
    from sstvae.rx import RingBuffer

    fake_pa.native = FS
    fake_pa.accepts = {FS}

    ring = RingBuffer(10.0)
    stream, rate = audio.open_input_stream(None, ring, FS, on_error=lambda m: None)
    assert rate == FS

    x = np.sin(2 * np.pi * 1200 * np.arange(FS) / FS)
    fake_pa.feed(x, blocksize=1024)
    got, total = ring.snapshot()
    assert np.allclose(got[:total], x[:total], atol=1e-6), "must pass through untouched"


def test_a_jack_device_is_flagged(fake_pa):
    """JACK has no buffering behind its realtime period, and our capture
    callback is Python. Losing the GIL race there drops audio silently --
    a mangled picture with sync intact, which nobody would blame on their
    device choice. Warn; don't refuse."""
    from sstvae.rx import RingBuffer

    fake_pa.hostapi_name = "JACK Audio Connection Kit"
    fake_pa.accepts = {48000}
    msgs = []
    audio.open_input_stream(None, RingBuffer(10.0), FS, on_error=msgs.append)
    assert any("host API" in m for m in msgs), msgs
    assert any("pipewire" in m for m in msgs), "should name the alternatives"


def test_an_ordinary_device_is_not_flagged(fake_pa):
    from sstvae.rx import RingBuffer

    fake_pa.accepts = {48000}
    msgs = []
    audio.open_input_stream(None, RingBuffer(10.0), FS, on_error=msgs.append)
    assert not any("host API" in m for m in msgs), msgs

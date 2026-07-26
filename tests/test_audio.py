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

    def check_output_settings(self, device=None, samplerate=None, **kw):
        if samplerate not in self.accepts:
            raise RuntimeError(f"Invalid sample rate {samplerate}")

    def query_devices(self, device=None, kind=None):
        return {"default_samplerate": float(self.native), "name": "Fake TX"}

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

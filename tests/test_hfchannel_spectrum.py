import numpy as np
import pytest

from sstvae import hfchannel
from sstvae.config import FS


@pytest.mark.parametrize("spread", [0.1, 1.0, 2.0])
def test_gaussian_taps_match_f1487_spread(spread):
    """F.1487 defines the frequency spread as 2 sigma of a Gaussian
    Doppler spectrum. The Butterworth generator this replaced read
    3.0 Hz at a 2 Hz setting."""
    g = hfchannel._gaussian_taps(
        int(FS * 600 / spread), spread, np.random.default_rng(1)
    )[::40]
    P = np.abs(np.fft.fft(g)) ** 2
    f = np.fft.fftfreq(len(g), 40 / FS)
    two_sigma = 2 * np.sqrt(np.sum(P * f**2) / np.sum(P))
    assert abs(two_sigma / spread - 1) < 0.05
    assert abs(np.mean(np.abs(g) ** 2) - 1) < 0.01


def test_clock_offset_is_band_limited():
    """A tone resampled by the clock-offset model stays a clean tone.
    np.interp left linear-interpolation products at ~-23 dB."""
    n = np.arange(FS * 4)
    x = np.sin(2 * np.pi * 1987.0 * n / FS)
    y = hfchannel.sample_clock_offset(x, 100.0)
    f = 1987.0 * len(x) / len(y)  # the ratio after rounding to whole samples
    ref = np.exp(-2j * np.pi * f * np.arange(len(y)) / FS)
    mid = slice(FS, len(y) - FS)  # away from the circular wrap
    a = 2 * np.mean(y[mid] * ref[mid])
    resid = y[mid] - np.real(a * np.conj(ref[mid]))
    assert 10 * np.log10(np.mean(resid**2) / np.mean(y[mid] ** 2)) < -60

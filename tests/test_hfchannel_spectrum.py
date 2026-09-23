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


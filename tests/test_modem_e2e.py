import numpy as np
import pytest

from sstvae import hfchannel
from sstvae.config import FS, MODES, FRAMES_PER_GROUP, LATENTS_PER_FRAME
from sstvae.modem import Modem


def _latent_snr_db(sent, got, w=None):
    mask = np.ones_like(sent, dtype=bool) if w is None else (w > 0)
    err = np.mean((sent[mask] - got[mask]) ** 2)
    return 10 * np.log10(np.mean(sent[mask] ** 2) / err)


@pytest.fixture(scope="module")
def modem():
    return Modem()


def _unit_latents(mode, seed=0):
    rng = np.random.default_rng(seed)
    lat = rng.normal(size=MODES[mode].n_latents)
    return lat / np.sqrt(np.mean(lat**2))


def test_clean_loopback(modem):
    lat = _unit_latents("A")
    x = modem.modulate(lat, "A")
    r = modem.demodulate(x)
    assert r.mode.name == "A"
    assert r.frames_received == FRAMES_PER_GROUP
    assert abs(r.freq_offset) < 1.0
    # Ceiling set by TX clip-and-filter distortion, which the decoder
    # network is trained through. Weighted mask excludes the small
    # permanently-dropped-for-the-beacon-carrier fraction (weight 0 by
    # design, not a channel error) alongside any real erasures.
    assert _latent_snr_db(lat, r.latents, r.weights) > 18


@pytest.mark.parametrize("df", [-50.0, 47.5, 50.0])
def test_freq_offset(modem, df):
    lat = _unit_latents("A", seed=1)
    x = modem.modulate(lat, "A")
    y = hfchannel.apply_channel(x, freq_offset_hz=df)
    r = modem.demodulate(y)
    assert abs(r.freq_offset - df) < 1.0
    assert _latent_snr_db(lat, r.latents, r.weights) > 17


def test_awgn_latent_snr(modem):
    lat = _unit_latents("A", seed=2)
    x = modem.modulate(lat, "A")
    y = hfchannel.apply_channel(x, snr_db=10.0, freq_offset_hz=-31.0)
    r = modem.demodulate(y)
    snr = _latent_snr_db(lat, r.latents)
    # 10 dB channel SNR (3 kHz ref) spread over 24 carriers x 50 baud:
    # expect per-latent SNR in the same ballpark, allow generous slop.
    assert 8 < snr < 20


def test_truncated_reception(modem):
    """Early-stopped mode C: first ~33 s covers group 0, rest erased."""
    lat = _unit_latents("C", seed=3)
    x = modem.modulate(lat, "C")
    r = modem.demodulate(x[: int(33 * FS)])
    assert r.mode.name == "C"
    assert FRAMES_PER_GROUP <= r.frames_received <= FRAMES_PER_GROUP + 20
    w = r.weights
    n0 = MODES["A"].n_latents
    assert np.all(w[-n0:] == 0)  # last group never arrived
    assert (w > 0).sum() >= FRAMES_PER_GROUP * LATENTS_PER_FRAME
    assert _latent_snr_db(lat, r.latents, w) > 18


def test_zeroed_frames(modem):
    """Blanked audio spans decode as erasures, rest survives."""
    lat = _unit_latents("A", seed=4)
    x = modem.modulate(lat, "A")
    y = hfchannel.apply_channel(x, spans=[(5.0, 6.0), (12.0, 12.5)])
    r = modem.demodulate(y)
    good = r.weights > 0.5
    assert good.sum() > 0.8 * lat.size
    assert _latent_snr_db(lat, r.latents, r.weights > 0.5) > 18


@pytest.mark.parametrize("ppm", [-50.0, 50.0])
def test_sample_clock_offset(modem, ppm):
    lat = _unit_latents("A", seed=5)
    x = modem.modulate(lat, "A")
    y = hfchannel.apply_channel(x, ppm=ppm)
    r = modem.demodulate(y)
    assert r.frames_received >= FRAMES_PER_GROUP - 1
    assert _latent_snr_db(lat, r.latents, r.weights) > 15


def test_fading_smoke(modem):
    """MPP fading at decent SNR: most latents recovered with sane weights."""
    lat = _unit_latents("A", seed=6)
    x = modem.modulate(lat, "A")
    y = hfchannel.apply_channel(x, snr_db=15.0, fading_preset="mpp", seed=9)
    r = modem.demodulate(y)
    assert r.frames_received >= FRAMES_PER_GROUP - 10
    strong = r.weights > 0.7
    assert strong.sum() > 0.3 * lat.size
    assert _latent_snr_db(lat, r.latents, strong) > 5


def test_no_signal_raises(modem):
    from sstvae.modem import SyncError

    rng = np.random.default_rng(11)
    with pytest.raises(SyncError):
        modem.demodulate(rng.normal(size=8 * FS))

import numpy as np
import pytest

from sstvae import hfchannel
from sstvae.config import FS, MODES, FRAMES_PER_GROUP, LATENTS_PER_FRAME
from sstvae.modem import Modem

from conftest import (
    SNR_MARGIN_DB,
    combine_snr_db,
    latent_snr_db as _latent_snr_db,
    snr_floor_db,
    unit_latents as _unit_latents,
)

# SNR each impairment reaches on its own with clipping disabled. These
# characterize the modem/channel, not the clip setting, so they stay put
# as CLIP_HEADROOM_DB moves; conftest folds in the clip floor. See
# conftest for the combination model.
AWGN_10DB_ONLY_DB = 8.9
ZEROED_SPANS_ONLY_DB = 24.1
PPM_50_ONLY_DB = 26.4
MPP_FADING_15DB_ONLY_DB = 13.7


@pytest.fixture(scope="module")
def modem():
    return Modem()


def test_loopback_without_clipping_is_near_lossless(unclipped_floor_db):
    """Modem health, independent of CLIP_HEADROOM_DB: with clip-and-filter
    disabled the round trip is limited only by equalization, the cyclic
    prefix and numerical error. This is the one latent-SNR assertion that
    should never move when the clip headroom is retuned."""
    assert unclipped_floor_db > 30, f"unclipped loopback {unclipped_floor_db:.1f} dB"


def test_clean_loopback(modem, clip_floor_db):
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
    assert _latent_snr_db(lat, r.latents, r.weights) > snr_floor_db(clip_floor_db)


def test_clipping_only_costs_snr(clip_floor_db, unclipped_floor_db):
    """Sanity on the clip floor itself: clipping may cost quality but can
    never add any, whatever headroom is configured."""
    assert clip_floor_db <= unclipped_floor_db + 0.5


@pytest.mark.parametrize("df", [-50.0, 47.5, 50.0])
def test_freq_offset(modem, df, clip_floor_db):
    lat = _unit_latents("A", seed=1)
    x = modem.modulate(lat, "A")
    y = hfchannel.apply_channel(x, freq_offset_hz=df)
    r = modem.demodulate(y)
    assert abs(r.freq_offset - df) < 1.0
    # A corrected CFO costs essentially nothing, so this stays clip-limited.
    assert _latent_snr_db(lat, r.latents, r.weights) > snr_floor_db(clip_floor_db)


def test_awgn_latent_snr(modem, clip_floor_db):
    lat = _unit_latents("A", seed=2)
    x = modem.modulate(lat, "A")
    y = hfchannel.apply_channel(x, snr_db=10.0, freq_offset_hz=-31.0)
    r = modem.demodulate(y)
    snr = _latent_snr_db(lat, r.latents)
    # 10 dB channel SNR spread over 24 carriers x 50 baud.
    # Bounded above as well: beating the noise-plus-clipping prediction
    # would mean the channel is not actually being applied.
    expected = combine_snr_db(clip_floor_db, AWGN_10DB_ONLY_DB)
    assert expected - SNR_MARGIN_DB < snr < expected + 2.0


def test_truncated_reception(modem, clip_floor_db):
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
    # Frames that did arrive are undamaged, so this is clip-limited.
    assert _latent_snr_db(lat, r.latents, w) > snr_floor_db(clip_floor_db)


def test_zeroed_frames(modem, clip_floor_db):
    """Blanked audio spans decode as erasures, rest survives."""
    lat = _unit_latents("A", seed=4)
    x = modem.modulate(lat, "A")
    y = hfchannel.apply_channel(x, spans=[(5.0, 6.0), (12.0, 12.5)])
    r = modem.demodulate(y)
    good = r.weights > 0.5
    assert good.sum() > 0.8 * lat.size
    assert _latent_snr_db(lat, r.latents, r.weights > 0.5) > snr_floor_db(
        clip_floor_db, ZEROED_SPANS_ONLY_DB
    )


@pytest.mark.parametrize("ppm", [-50.0, 50.0])
def test_sample_clock_offset(modem, ppm, clip_floor_db):
    lat = _unit_latents("A", seed=5)
    x = modem.modulate(lat, "A")
    y = hfchannel.apply_channel(x, ppm=ppm)
    r = modem.demodulate(y)
    assert r.frames_received >= FRAMES_PER_GROUP - 1
    assert _latent_snr_db(lat, r.latents, r.weights) > snr_floor_db(
        clip_floor_db, PPM_50_ONLY_DB
    )


def test_fading_smoke(modem, clip_floor_db):
    """MPP fading at decent SNR: most latents recovered with sane weights."""
    lat = _unit_latents("A", seed=6)
    x = modem.modulate(lat, "A")
    y = hfchannel.apply_channel(x, snr_db=15.0, fading_preset="mpp", seed=9)
    r = modem.demodulate(y)
    assert r.frames_received >= FRAMES_PER_GROUP - 10
    strong = r.weights > 0.7
    assert strong.sum() > 0.3 * lat.size
    assert _latent_snr_db(lat, r.latents, strong) > snr_floor_db(
        clip_floor_db, MPP_FADING_15DB_ONLY_DB
    )


def test_no_signal_raises(modem):
    from sstvae.modem import SyncError

    rng = np.random.default_rng(11)
    with pytest.raises(SyncError):
        modem.demodulate(rng.normal(size=8 * FS))

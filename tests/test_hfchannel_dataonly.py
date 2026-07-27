import numpy as np
import pytest

from sstvae import hfchannel
from sstvae.config import MODES
from sstvae.modem import Modem

from conftest import snr_floor_db

# SNR this scenario reaches with clipping disabled (see conftest).
MPG_FADING_15DB_ONLY_DB = 14.7


@pytest.fixture(scope="module")
def modem():
    return Modem()


def _unit_latents(mode, seed=0):
    rng = np.random.default_rng(seed)
    lat = rng.normal(size=MODES[mode].n_latents)
    return lat / np.sqrt(np.mean(lat**2))


def test_detect_mode_by_length(modem):
    for name in ["A", "B", "C"]:
        x = modem.modulate(_unit_latents(name), name)
        assert hfchannel.detect_mode_by_length(x).name == name


def test_detect_mode_rejects_arbitrary_length():
    with pytest.raises(ValueError):
        hfchannel.detect_mode_by_length(np.zeros(12345))


def test_mask_protects_only_preamble_and_header(modem):
    x = modem.modulate(_unit_latents("A"), "A")
    mask = hfchannel.data_sample_mask(MODES["A"], len(x))
    # Preamble+header+lead-in/out are a tiny fraction of a ~30s
    # transmission; everything else (all frames, pilots included) is
    # corruptible.
    assert mask.mean() > 0.97


def test_protect_sync_survives_extreme_noise(modem):
    """SNR so low the header/frames would never lock in a normal
    apply_channel call, but with --protect-sync sync must still work."""
    lat = _unit_latents("A", seed=1)
    x = modem.modulate(lat, "A")

    # Sanity: this SNR really does break normal (unprotected) sync.
    from sstvae.modem import SyncError

    y_unprotected = hfchannel.apply_channel(x, snr_db=-15, seed=2)
    with pytest.raises(SyncError):
        modem.demodulate(y_unprotected)

    y = hfchannel.apply_channel_data_only(x, snr_db=-15, seed=2)
    r = modem.demodulate(y)
    assert r.mode.name == "A"
    assert r.frames_received == r.mode.n_frames
    # Data itself should be devastated at this SNR.
    err = np.mean((lat - r.latents) ** 2)
    snr_db_out = 10 * np.log10(np.mean(lat**2) / err)
    assert snr_db_out < 5


def test_protect_sync_survives_extreme_fading(modem):
    """Preamble/header stay protected so acquisition locks even under
    disturbed fading + low SNR; frames may legitimately be degraded
    (not asserting quality here, just that it doesn't fail to decode
    at all)."""
    lat = _unit_latents("A", seed=3)
    x = modem.modulate(lat, "A")
    y = hfchannel.apply_channel_data_only(
        x, fading_preset="mpd", snr_db=0.0, seed=4
    )
    r = modem.demodulate(y)
    assert r.mode.name == "A"
    assert r.sync_metric > 0.5
    assert r.frames_received == r.mode.n_frames


def test_protect_sync_fading_equalization_is_not_self_defeating(modem, clip_floor_db):
    """The bug this mask design fixes: protecting pilots but not data
    under fading made the equalizer trust a clean channel estimate
    while the data went through a totally different faded channel,
    producing garbage regardless of SNR. With pilots corrupted
    consistently alongside their data, decent SNR + mild fading should
    give a reasonably good reconstruction, not noise."""
    lat = _unit_latents("A", seed=8)
    x = modem.modulate(lat, "A")
    y = hfchannel.apply_channel_data_only(
        x, fading_preset="mpg", snr_db=15.0, seed=9
    )
    r = modem.demodulate(y)
    assert r.mode.name == "A"
    good = r.weights > 0.7
    assert good.sum() > 0.3 * lat.size
    err = np.mean((lat[good] - r.latents[good]) ** 2)
    snr_db_out = 10 * np.log10(np.mean(lat[good] ** 2) / err)
    floor = snr_floor_db(clip_floor_db, MPG_FADING_15DB_ONLY_DB)
    assert snr_db_out > floor, (
        f"expected ~{floor:.1f} dB or better, got {snr_db_out:.1f} dB"
    )


def test_protect_sync_clean_data_is_bit_exact(modem):
    """With no impairments requested, output should equal the input
    exactly (mask splicing shouldn't perturb anything on its own)."""
    lat = _unit_latents("B", seed=5)
    x = modem.modulate(lat, "B")
    y = hfchannel.apply_channel_data_only(x)
    assert np.allclose(x, y)


def test_protect_sync_freq_offset_applied_globally(modem):
    lat = _unit_latents("A", seed=6)
    x = modem.modulate(lat, "A")
    y = hfchannel.apply_channel_data_only(x, freq_offset_hz=35.0, snr_db=20.0, seed=7)
    r = modem.demodulate(y)
    assert abs(r.freq_offset - 35.0) < 1.0


def test_ppm_rejected_via_cli_contract():
    """apply_channel_data_only has no ppm parameter at all (see
    docstring: resampling breaks the fixed-layout assumption)."""
    import inspect

    assert "ppm" not in inspect.signature(
        hfchannel.apply_channel_data_only
    ).parameters

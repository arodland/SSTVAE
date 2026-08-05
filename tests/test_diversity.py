import numpy as np
import pytest

from sstvae import hfchannel
from sstvae.modem import Modem, SyncError
from sstvae.modem.diversity import combine_demod_results, demodulate_diversity

from conftest import latent_snr_db, unit_latents


@pytest.fixture(scope="module")
def modem():
    return Modem()


def test_single_branch_is_identity(modem):
    lat = unit_latents("A")
    r = modem.demodulate(modem.modulate(lat, "A"))
    combined = combine_demod_results([r])
    assert combined is r


def test_needs_at_least_one_branch():
    with pytest.raises(ValueError):
        combine_demod_results([])


def test_mode_mismatch_raises(modem):
    ra = modem.demodulate(modem.modulate(unit_latents("A"), "A"))
    rb = modem.demodulate(modem.modulate(unit_latents("B"), "B"))
    with pytest.raises(ValueError):
        combine_demod_results([ra, rb])


def test_two_identical_clean_branches_dont_distort_the_signal(modem):
    """Two branches fed the *same* clean recording aren't a real
    diversity scenario (their "noise" is perfectly correlated, not
    independent, so MRC's independence assumption doesn't hold) -- but
    the combined latents should still equal the input exactly, and the
    combined weight should never fall outside [0, 1]."""
    lat = unit_latents("A")
    x = modem.modulate(lat, "A")
    r1 = modem.demodulate(x)
    r2 = modem.demodulate(x)
    combined = combine_demod_results([r1, r2])
    np.testing.assert_allclose(combined.latents, r1.latents, atol=1e-9)
    assert combined.weights.min() >= 0.0
    assert combined.weights.max() <= 1.0 + 1e-9


def test_combined_weight_never_exceeds_one(modem):
    lat = unit_latents("A")
    x = modem.modulate(lat, "A")
    r1 = modem.demodulate(hfchannel.apply_channel(x, snr_db=15.0, seed=1))
    r2 = modem.demodulate(hfchannel.apply_channel(x, snr_db=15.0, seed=2))
    combined = combine_demod_results([r1, r2])
    assert combined.weights.max() <= 1.0 + 1e-9


def test_erasure_in_both_branches_stays_erased(modem):
    lat = unit_latents("A")
    r1 = modem.demodulate(modem.modulate(lat, "A"))
    r2 = modem.demodulate(modem.modulate(lat, "A"))
    r1.weights[:10] = 0.0
    r2.weights[:10] = 0.0
    combined = combine_demod_results([r1, r2])
    assert np.all(combined.weights[:10] == 0.0)
    assert np.all(combined.latents[:10] == 0.0)


def test_diversity_gain_under_independent_awgn(modem):
    """Two branches at the same nominal SNR, independent noise: combined
    latent SNR should land close to the +3 dB MRC prediction (branch
    SNRs sum in linear terms) and beat either branch alone."""
    lat = unit_latents("A")
    x = modem.modulate(lat, "A")
    snr_db = 6.0
    r1 = modem.demodulate(hfchannel.apply_channel(x, snr_db=snr_db, seed=10))
    r2 = modem.demodulate(hfchannel.apply_channel(x, snr_db=snr_db, seed=20))
    s1 = latent_snr_db(lat, r1.latents, r1.weights)
    s2 = latent_snr_db(lat, r2.latents, r2.weights)
    combined = combine_demod_results([r1, r2])
    s_combined = latent_snr_db(lat, combined.latents, combined.weights)
    assert s_combined > max(s1, s2) + 1.5
    predicted = 10 * np.log10(10 ** (s1 / 10) + 10 ** (s2 / 10))
    assert abs(s_combined - predicted) < 1.5


def test_diversity_gain_under_independent_fading(modem):
    """Independent Watterson fading per branch: a deep fade on one
    branch shouldn't sink the combined decode the way it sinks that
    branch alone."""
    lat = unit_latents("A", seed=3)
    x = modem.modulate(lat, "A")
    r1 = modem.demodulate(
        hfchannel.apply_channel(x, snr_db=10.0, fading_preset="mpp", seed=11)
    )
    r2 = modem.demodulate(
        hfchannel.apply_channel(x, snr_db=10.0, fading_preset="mpp", seed=41)
    )
    s1 = latent_snr_db(lat, r1.latents, r1.weights)
    s2 = latent_snr_db(lat, r2.latents, r2.weights)
    combined = combine_demod_results([r1, r2])
    s_combined = latent_snr_db(lat, combined.latents, combined.weights)
    assert s_combined > max(s1, s2)


def test_demodulate_diversity_matches_manual_combine(modem):
    lat = unit_latents("A")
    x = modem.modulate(lat, "A")
    streams = [
        hfchannel.apply_channel(x, snr_db=8.0, seed=100),
        hfchannel.apply_channel(x, snr_db=8.0, seed=200),
    ]
    manual = combine_demod_results([modem.demodulate(s) for s in streams])
    via_helper = demodulate_diversity(modem, streams)
    np.testing.assert_allclose(via_helper.latents, manual.latents)
    np.testing.assert_allclose(via_helper.weights, manual.weights)


def test_demodulate_diversity_drops_a_dead_branch(modem):
    """One branch is pure noise (never acquires); the other is clean.
    The combine should fall back to the surviving branch rather than
    raising."""
    lat = unit_latents("A")
    x = modem.modulate(lat, "A")
    dead = np.random.default_rng(0).normal(scale=0.05, size=len(x))
    good = modem.demodulate(x)
    result = demodulate_diversity(modem, [dead, x])
    np.testing.assert_allclose(result.latents, good.latents)


def test_demodulate_diversity_raises_if_every_branch_fails(modem):
    dead = np.random.default_rng(0).normal(scale=0.05, size=48000)
    with pytest.raises(SyncError):
        demodulate_diversity(modem, [dead, dead.copy()])

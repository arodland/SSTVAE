import numpy as np
import pytest

from sstvae import hfchannel
from sstvae.config import FRAMES_PER_GROUP, LATENT_GROUPS, MODES, NC_LATENT
from sstvae.modem import Modem, SyncError
from sstvae.modem import framing
from sstvae.modem.diversity import (
    branch_contribution,
    combine_blind_results,
    combine_demod_results,
    combine_diversity_results,
    contribution_image,
    demodulate_diversity,
)

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


def test_branch_contribution_columns_sum_to_one_or_zero(modem):
    lat = unit_latents("A")
    x = modem.modulate(lat, "A")
    r1 = modem.demodulate(hfchannel.apply_channel(x, snr_db=8.0, seed=1))
    r2 = modem.demodulate(hfchannel.apply_channel(x, snr_db=8.0, seed=2))
    frac = branch_contribution([r1, r2])
    assert frac.shape == (2, len(r1.latents))
    totals = frac.sum(axis=0)
    # Every column is either fully accounted for (weight somewhere) or
    # fully erased on both branches (0) -- never a fraction that would
    # imply a third, uncounted source.
    assert np.all((np.isclose(totals, 1.0)) | (np.isclose(totals, 0.0)))


def test_branch_contribution_single_branch_is_its_own_erasure_mask(modem):
    lat = unit_latents("A")
    r = modem.demodulate(modem.modulate(lat, "A"))
    frac = branch_contribution([r])
    np.testing.assert_array_equal(frac[0], (r.weights > 0).astype(float))


def test_branch_contribution_favors_the_stronger_branch(modem):
    """A branch that's essentially noise everywhere should get ~0 credit
    against a clean branch, latent by latent."""
    lat = unit_latents("A")
    x = modem.modulate(lat, "A")
    clean = modem.demodulate(x)
    noisy = modem.demodulate(hfchannel.apply_channel(x, snr_db=-5.0, seed=7))
    frac = branch_contribution([clean, noisy])
    assert frac[0].mean() > frac[1].mean()


def test_contribution_image_shape_and_needs_two_branches(modem):
    lat = unit_latents("A")
    x = modem.modulate(lat, "A")
    r1 = modem.demodulate(hfchannel.apply_channel(x, snr_db=8.0, seed=1))
    r2 = modem.demodulate(hfchannel.apply_channel(x, snr_db=8.0, seed=2))

    with pytest.raises(ValueError):
        contribution_image([r1])

    img = contribution_image([r1, r2], scale=1)
    assert img.size == (r1.mode.n_frames, NC_LATENT)
    assert img.mode == "RGB"
    arr = np.asarray(img)
    assert arr[:, :, 1].max() == 0  # green channel unused
    # Every carrier carries data in every frame (unlike the old
    # decoder-channel indexing, where the interleaver's scatter left
    # most cells with no coverage at all) -- at a decent SNR almost
    # every cell should be lit, black only from real erasure.
    lit = arr.sum(axis=-1) > 0
    assert lit.mean() > 0.9


def test_contribution_image_black_means_erased_not_missing_coverage(modem):
    """Unlike the old decoder-channel-indexed image, every carrier is
    used in every frame, so a black cell can only mean both branches
    erased that carrier that frame -- never "the interleaver didn't
    touch it". Verified directly rather than trusting real fading to
    produce one."""
    lat = unit_latents("A")
    r1 = modem.demodulate(modem.modulate(lat, "A"))
    r2 = modem.demodulate(modem.modulate(lat, "A"))
    r1.weights[:] = 0.0
    r2.weights[:] = 0.0
    img = contribution_image([r1, r2], scale=1)
    assert np.asarray(img).sum() == 0


def test_contribution_image_no_stair_stepping_across_mode_b_groups(modem):
    """Mode B transmits group 0's frames, then group 1's, as two
    sequential blocks over disjoint *decoder-channel* ranges -- the old
    channel-indexed image showed that as a staircase (each block lit
    only its own channel range). Carrier index is the same physical set
    throughout, so every row must show data in frames from *both*
    groups, not just one."""
    lat = unit_latents("B")
    x = modem.modulate(lat, "B")
    r1 = modem.demodulate(hfchannel.apply_channel(x, snr_db=10.0, seed=31))
    r2 = modem.demodulate(hfchannel.apply_channel(x, snr_db=10.0, seed=32))
    img = contribution_image([r1, r2], scale=1)
    arr = np.asarray(img)

    n_frames = r1.mode.n_frames
    early_frame = n_frames // 4  # well inside group 0's block
    late_frame = 3 * n_frames // 4  # well inside group 1's block
    lit_early = arr[:, early_frame, :].sum(axis=-1) > 0
    lit_late = arr[:, late_frame, :].sum(axis=-1) > 0
    assert lit_early.mean() > 0.9
    assert lit_late.mean() > 0.9


def test_contribution_image_pure_branch_is_pure_hue_peaking_at_full_brightness(modem):
    """One branch dead, the other clean: every covered cell should be
    pure red (no blue at all -- the dead branch never contributes a
    share), and the strongest cell should read at essentially full
    brightness, since brightness is normalized to this reception's own
    peak. Individual cells needn't all hit 255 themselves -- brightness
    tracks the *good* branch's own per-carrier weight, which on a real
    (if unfaded) channel varies slightly carrier to carrier, and that
    variation is exactly what this feature exists to show."""
    lat = unit_latents("A")
    x = modem.modulate(lat, "A")
    good = modem.demodulate(x)
    dead = modem.demodulate(x)
    dead.weights[:] = 0.0
    dead.snr_db = float("-inf")

    img = contribution_image([good, dead], scale=1)
    arr = np.asarray(img).astype(np.int32)
    lit = arr.sum(axis=-1) > 0
    assert np.all(arr[:, :, 2][lit] == 0)  # no blue at all: pure hue
    assert arr[:, :, 0][lit].max() >= 250  # the peak cell is near-saturated
    assert arr[:, :, 0][lit].min() > 0  # but nothing lit is fully dark


def test_contribution_image_darkens_when_both_branches_fade_together(modem):
    """The feature this test exists for: two branches splitting a
    latent evenly (magenta hue either way) must still draw differently
    depending on how *much* either had to offer -- a carrier both
    branches faded on should go dark, not stay a bright, fully-mixed
    magenta just because the split was even. Constructed directly
    rather than via real fading so the two frames being compared are
    known exactly: one frame's underlying latents pinned to full
    weight on both branches, everything else (including a second,
    distinct frame used as the "faded" comparison) at a low baseline,
    identical between branches throughout -- so the fractional hue is
    exactly 50/50 red/blue everywhere, and only the brightness differs.
    """
    lat = unit_latents("A")
    x = modem.modulate(lat, "A")
    r1 = modem.demodulate(x)
    r2 = modem.demodulate(x)

    baseline = 0.05
    r1.weights[:] = baseline
    r2.weights[:] = baseline
    r1.snr_db = r2.snr_db = 10.0

    f_strong, f_weak = 0, r1.mode.n_frames // 2
    _, idx_strong = framing.slot_range_for_frame(f_strong)
    r1.weights[idx_strong] = 1.0
    r2.weights[idx_strong] = 1.0

    img = contribution_image([r1, r2], scale=1)
    arr = np.asarray(img).astype(np.int32)
    col_strong = arr[:, f_strong, :]
    col_weak = arr[:, f_weak, :]

    # Hue is even in both columns: red and blue nearly equal, since both
    # branches carry the identical weight everywhere.
    assert np.allclose(col_strong[:, 0], col_strong[:, 2], atol=2)
    assert np.allclose(col_weak[:, 0], col_weak[:, 2], atol=2)

    # But the faded column is far dimmer than the full-strength one,
    # despite the identical 50/50 split -- brightness tracks overall
    # strength, not just how evenly it was shared.
    assert col_weak.sum() > 0  # still lit, not erased -- baseline > 0
    assert col_strong.sum() > 5 * col_weak.sum()
    # An even 50/50 split caps each channel at ~half brightness; the
    # peak column's R+B should still sum to essentially full (255).
    assert (col_strong[:, 0] + col_strong[:, 2]).max() >= 250


# --- blind-acquisition branches ------------------------------------------

@pytest.fixture(scope="module")
def blind_pair(modem):
    """Two branches of the same mode-A transmission, each demodulated
    *blind* (no preamble/header used) -- the whole 32 s buffer is enough
    for MIN_FRAMES_FOR_SYNC (~10.5 s) to see a full beacon superframe."""
    lat = unit_latents("A", seed=42)
    x = modem.modulate(lat, "A", callsign="BLIND")
    a = hfchannel.apply_channel(x, snr_db=8.0, seed=101)
    b = hfchannel.apply_channel(x, snr_db=8.0, seed=202)
    return lat, modem.demodulate_blind(a), modem.demodulate_blind(b)


def test_blind_results_actually_locked(blind_pair):
    """Sanity check on the fixture itself: if the blind lock silently
    stopped working, every test built on it would pass vacuously."""
    _, ra, rb = blind_pair
    assert ra.beacon is not None and rb.beacon is not None
    assert ra.callsign == "BLIND" and rb.callsign == "BLIND"


def test_combine_blind_single_branch_is_identity(blind_pair):
    _, ra, _ = blind_pair
    combined = combine_blind_results([ra])
    assert combined is ra


def test_combine_blind_needs_at_least_one_branch():
    with pytest.raises(ValueError):
        combine_blind_results([])


def test_combine_blind_two_branches_are_directly_aligned(blind_pair, modem):
    """No sample-timebase matching needed for blind branches: they place
    latents by the beacon's absolute frame counter, so combining two
    independent locks of the same transmission should recover (close to)
    every latent mode A actually carries, same as the header path does."""
    lat, ra, rb = blind_pair
    combined = combine_blind_results([ra, rb])
    assert combined.weights.max() <= 1.0 + 1e-9
    n_a = MODES["A"].n_latents
    # Every mode-A latent actually carried on air (n_tx_latents -- the
    # rest is DROPPED_LATENTS_PER_GROUP, a permanent erasure, never
    # transmitted at all) should have landed somewhere in the combined
    # full-C-sized weight array.
    assert np.count_nonzero(combined.weights[:n_a]) == MODES["A"].n_tx_latents
    s = latent_snr_db(lat, combined.latents[:n_a], combined.weights[:n_a])
    sa = latent_snr_db(lat, ra.latents[:n_a], ra.weights[:n_a])
    sb = latent_snr_db(lat, rb.latents[:n_a], rb.weights[:n_a])
    assert s > max(sa, sb) - 0.5  # combining shouldn't ever do meaningfully worse


# --- combine_diversity_results: any mix of header/blind -------------------

def test_diversity_results_pure_headered_matches_combine_demod_results(modem):
    lat = unit_latents("A")
    x = modem.modulate(lat, "A")
    ra = modem.demodulate(hfchannel.apply_channel(x, snr_db=8.0, seed=1))
    rb = modem.demodulate(hfchannel.apply_channel(x, snr_db=8.0, seed=2))
    via_diversity = combine_diversity_results([ra, rb])
    via_demod = combine_demod_results([ra, rb])
    np.testing.assert_allclose(via_diversity.latents, via_demod.latents)
    np.testing.assert_allclose(via_diversity.weights, via_demod.weights)


def test_diversity_results_pure_blind_matches_combine_blind_results(blind_pair):
    _, ra, rb = blind_pair
    via_diversity = combine_diversity_results([ra, rb])
    via_blind = combine_blind_results([ra, rb])
    np.testing.assert_allclose(via_diversity.latents, via_blind.latents)
    np.testing.assert_allclose(via_diversity.weights, via_blind.weights)


def test_diversity_results_mixed_header_and_blind(modem):
    """One branch header-locked, the other only blind-locked -- the
    result should still be a DemodResult (the header's mode is
    authoritative), sized to that mode, with the blind branch's data
    folded in rather than ignored."""
    lat = unit_latents("A", seed=7)
    x = modem.modulate(lat, "A", callsign="MIXED")
    headered = modem.demodulate(hfchannel.apply_channel(x, snr_db=6.0, seed=11))
    blind = modem.demodulate_blind(hfchannel.apply_channel(x, snr_db=6.0, seed=22))
    assert blind.beacon is not None  # fixture sanity

    combined = combine_diversity_results([headered, blind])
    assert combined.mode.name == "A"
    assert len(combined.latents) == MODES["A"].n_latents
    assert combined.weights.max() <= 1.0 + 1e-9

    s_combined = latent_snr_db(lat, combined.latents, combined.weights)
    s_headered = latent_snr_db(lat, headered.latents, headered.weights)
    assert s_combined > s_headered - 0.5


def test_diversity_results_mixed_beats_either_branch_alone(modem):
    lat = unit_latents("A", seed=9)
    x = modem.modulate(lat, "A")
    headered = modem.demodulate(hfchannel.apply_channel(x, snr_db=6.0, seed=33))
    blind = modem.demodulate_blind(hfchannel.apply_channel(x, snr_db=6.0, seed=44))
    combined = combine_diversity_results([headered, blind])

    s_combined = latent_snr_db(lat, combined.latents, combined.weights)
    s_headered = latent_snr_db(lat, headered.latents, headered.weights)
    n_a = MODES["A"].n_latents
    s_blind = latent_snr_db(lat, blind.latents[:n_a], blind.weights[:n_a])
    assert s_combined > max(s_headered, s_blind) + 1.0


def test_diversity_results_rejects_mismatched_header_modes(modem):
    ra = modem.demodulate(modem.modulate(unit_latents("A"), "A"))
    rb = modem.demodulate(modem.modulate(unit_latents("B"), "B"))
    with pytest.raises(ValueError):
        combine_diversity_results([ra, rb])


def test_diversity_results_rejects_unknown_branch_type(modem):
    ra = modem.demodulate(modem.modulate(unit_latents("A"), "A"))
    with pytest.raises(TypeError):
        combine_diversity_results([ra, object()])


def test_branch_contribution_and_image_accept_mixed_branches(modem):
    lat = unit_latents("A", seed=13)
    x = modem.modulate(lat, "A")
    headered = modem.demodulate(hfchannel.apply_channel(x, snr_db=8.0, seed=55))
    blind = modem.demodulate_blind(hfchannel.apply_channel(x, snr_db=8.0, seed=66))

    frac = branch_contribution([headered, blind])
    assert frac.shape == (2, len(blind.latents))  # full C range, blind sets the size

    img = contribution_image([headered, blind], scale=1)
    # Mixed/blind combos can't assume the header's mode range, so the
    # image spans mode C's full frame count.
    assert img.size == (LATENT_GROUPS * FRAMES_PER_GROUP, NC_LATENT)

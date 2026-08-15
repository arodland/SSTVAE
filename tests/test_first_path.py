"""Timing must lock to the *first* path, not the strongest one.

Both acquisition paths find timing by taking the argmax of a correlation
against a known reference. On a two-path channel that is the strongest
path, which is not the first one, and which of the two is stronger
changes as the channel fades -- so the argmax lands `delay` samples late
whenever the echo wins, putting the early path's energy ahead of the
demodulation window where the cyclic prefix cannot cover it. On the
blind path, which re-acquires every poll, that showed up as a live
picture alternating between a clean decode and a mushy one for as long
as the reception lasted. See config.FIRST_PATH_SEARCH.

The channel here is a *deterministic* two-path echo rather than a
Watterson preset: the effect is a property of the geometry, and fading
would only add an RNG that decides how often the test exercises it.

Every test that asserts on the first path also asserts that the argmax
is somewhere else, because otherwise it would pass just as happily with
first-path selection deleted -- on a single-path channel the two are the
same answer by construction (that is what test_single_path_is_untouched
pins down).
"""

import numpy as np
import pytest

from sstvae.config import (
    FIRST_PATH_SEARCH,
    FRAME_SAMPLES,
    HEADER_SAMPLES,
    LEADIN_SAMPLES,
    MODES,
    NCP,
    PREAMBLE_SAMPLES,
)
from sstvae.modem import Modem
from sstvae.modem.dsp import to_baseband
from sstvae.modem.ofdm import pilot_template
from sstvae.modem.sync import (
    BlindAcquisition,
    acquire,
    acquire_blind,
    first_path,
)

# Second path stronger than the first. It has to be strictly stronger,
# not merely present: at equal power the argmax is already ambiguous and
# the test would be measuring a tie-break.
ECHO_GAIN = 1.4


def _tx(mode="B", seed=0):
    modem = Modem()
    lat = np.random.default_rng(seed).normal(size=MODES[mode].n_latents)
    lat /= np.sqrt(np.mean(lat**2))
    return modem.modulate(lat, mode, callsign="N0CALL")


def _echo(x, delay, gain=ECHO_GAIN):
    late = np.concatenate([np.zeros(delay), x[: len(x) - delay]])
    return (x + gain * late) / np.sqrt(1 + gain**2)


def _pilot_fold(x):
    """The blind path's per-phase matched-filter power, one CFO bin at
    zero offset -- the array acquire_blind takes its argmax of."""
    from scipy.fft import fft, ifft, next_fast_len

    z = to_baseband(x)
    kernel = np.conj(pilot_template()[::-1])
    m = len(kernel)
    n_fft = next_fast_len(len(z) + m - 1)
    mf = ifft(fft(z, n_fft) * fft(kernel, n_fft))[m - 1 : m - 1 + len(z) - m + 1]
    p2 = np.abs(mf) ** 2
    n = len(p2) // FRAME_SAMPLES
    return p2[: n * FRAME_SAMPLES].reshape(n, FRAME_SAMPLES).sum(axis=0)


# 2.0 ms and 4.0 ms: the second-path delays of the mpp and mpd presets,
# which is where this was found. Both sit inside the cyclic prefix, so
# syncing to the first path is not merely better but correct.
DELAYS = [16, 32]


@pytest.mark.parametrize("delay", DELAYS)
def test_blind_acquisition_takes_the_early_path(delay):
    x = _tx()
    clean_phase = int(np.argmax(_pilot_fold(x)))

    fold = _pilot_fold(_echo(x, delay))
    assert int(np.argmax(fold)) == (clean_phase + delay) % FRAME_SAMPLES, (
        "the echo is supposed to win the argmax -- without that this test "
        "passes with first-path selection removed"
    )

    ba = acquire_blind(to_baseband(_echo(x, delay)))
    assert ba.frame_start % FRAME_SAMPLES == clean_phase % FRAME_SAMPLES


@pytest.mark.parametrize("delay", DELAYS)
def test_preamble_acquisition_takes_the_early_path(delay):
    x = _tx()
    assert acquire(to_baseband(x)).preamble_start == LEADIN_SAMPLES
    acq = acquire(to_baseband(_echo(x, delay)))
    assert acq.preamble_start == LEADIN_SAMPLES


@pytest.mark.parametrize("delay", DELAYS)
def test_the_early_path_decodes_better_than_the_late_one(delay):
    """The reason any of this matters, stated as latent SNR.

    Deliberately a *comparison* rather than an absolute bar: a two-path
    channel with this much delay spread has real frequency-selective
    nulls in it, so the echo costs picture quality whichever path we sync
    to, and an absolute threshold would be measuring the nulls. What
    first-path selection owns is the difference between the two timings,
    which is 1.2-2.2 dB and which pre-fix went the wrong way for as long
    as the echo stayed the stronger path.
    """
    modem = Modem()
    lat = np.random.default_rng(0).normal(size=MODES["B"].n_latents)
    lat /= np.sqrt(np.mean(lat**2))
    x = modem.modulate(lat, "B", callsign="N0CALL")
    y = _echo(x, delay)

    def latent_snr(acq):
        r = modem.demodulate_blind(y, acquisition=acq)
        n = min(len(r.latents), len(lat))
        w, est, truth = r.weights[:n], r.latents[:n], lat[:n]
        err = (est - truth) * w
        return 10 * np.log10(np.sum((truth * w) ** 2) / (np.sum(err**2) + 1e-20))

    # The comparison point is the *argmax*, derived independently, not
    # `early.frame_start + delay`. Anchoring the late candidate on what
    # acquire_blind returned makes the test insensitive to which path it
    # picked -- verified, that version passes with first-path selection
    # stubbed out, because both candidates then slide by the same delay.
    early = acquire_blind(to_baseband(y))
    argmax_phase = int(np.argmax(_pilot_fold(y)))
    late = BlindAcquisition(
        frame_start=early.frame_start
        + (argmax_phase - early.frame_start) % FRAME_SAMPLES,
        freq_offset=early.freq_offset,
        metric=early.metric,
    )
    snr_early, snr_late = latent_snr(early), latent_snr(late)
    assert snr_early > snr_late + 0.8, (
        f"{delay}-sample echo: first path {snr_early:.2f} dB, "
        f"strongest path {snr_late:.2f} dB"
    )


def test_single_path_is_untouched():
    """Where the argmax is right, first-path selection returns it --
    exactly, not approximately. This is what makes the change free on
    awgn rather than merely cheap."""
    fold = _pilot_fold(_tx())
    peak = int(np.argmax(fold))
    assert first_path(fold, peak, cyclic=True) == peak


def test_first_path_ignores_a_peak_outside_the_cyclic_prefix():
    """A path further ahead than the CP cannot be equalized whichever one
    we sync to, so there is nothing to win by reaching for it -- and
    reaching would let the preamble path, whose template is periodic with
    M, walk back a whole period."""
    profile = np.zeros(FRAME_SAMPLES)
    peak = 800
    profile[peak] = 1.0
    profile[peak - (FIRST_PATH_SEARCH + 1)] = 0.9
    assert first_path(profile, peak, cyclic=True) == peak

    profile[peak - FIRST_PATH_SEARCH] = 0.9
    assert first_path(profile, peak, cyclic=True) == peak - FIRST_PATH_SEARCH


def test_first_path_needs_a_local_maximum_not_just_a_level():
    """A plain 'earliest sample above the threshold' walks down the
    argmax's own correlation skirt and answers a few samples early on
    every channel, single-path ones included -- measured, 0.27 dB at mpd.
    A monotone ramp up to the peak has no earlier local maximum in it."""
    profile = np.zeros(FRAME_SAMPLES)
    peak = 800
    ramp = np.linspace(0.4, 1.0, FIRST_PATH_SEARCH + 1)
    profile[peak - FIRST_PATH_SEARCH : peak + 1] = ramp
    assert first_path(profile, peak, cyclic=True) == peak


def test_blind_frame_start_still_lands_on_a_useful_window():
    """First-path selection moves the reported timing, so the invariant
    the rest of the blind path relies on -- frame_start is a pilot's
    useful-window start, one CP into its frame -- has to survive it."""
    x = _tx()
    ba = acquire_blind(to_baseband(_echo(x, 16)))
    frames_start = LEADIN_SAMPLES + PREAMBLE_SAMPLES + HEADER_SAMPLES
    assert (ba.frame_start - frames_start) % FRAME_SAMPLES == NCP

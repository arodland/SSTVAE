"""The beacon's soft chips must be maximal-ratio, not equalized.

`demodulate` divides each carrier by its pilot channel estimate. On the
23 latent carriers that is what the confidence weights are for; the
beacon carrier gets no weight, so an equalized chip from a carrier in a
fade is amplified noise carrying a *large* magnitude -- and Golay's soft
ML decode reads magnitude as confidence, so one faded chip outvotes the
four good ones in its codeword.

The damage is graded rather than catastrophic (`floor` bounds a total
null, so the worst case is a carrier a little *above* the floor), which
is why this is a rate over channel realizations and not a single
crafted frame: a deterministic version of this test passed against the
broken code.

Measured with `scripts/beacon_soft_sweep.py`, mode B, 60 trials/cell,
bit-exact beacon decode over short blind windows:

    channel  SNR   real(y)   real(raw*conj(h))
    awgn     0.0      0.80        0.93
    awgn    -2.0      0.30        0.52
    mpp      8.0      0.58        0.93
    mpp      4.0      0.27        0.65
    mpd      8.0      0.40        0.92
    mpd      4.0      0.13        0.60

The cell below is mpd 8 dB: 0.40 against 0.92, so the bound sits several
sigma from both.
"""

import numpy as np

from sstvae import hfchannel
from sstvae.config import (
    FRAME_SAMPLES,
    HEADER_SAMPLES,
    LEADIN_SAMPLES,
    MODES,
    PREAMBLE_SAMPLES,
)
from sstvae.modem import Modem
from sstvae.modem.beacon import MIN_FRAMES_FOR_SYNC

CALLSIGN = "W1AW"
CHIPS_PER_FRAME = 5
TRIALS = 25
MIN_DECODES = 18  # shipped ~0.92 (23/25); equalized ~0.40 (10/25)


def _blind_window(mode, preset, snr_db, seed, frames=MIN_FRAMES_FOR_SYNC + 4):
    """A short mid-transmission window, the case where the beacon is
    actually load-bearing: one superframe copy, no preamble. Over a whole
    transmission six repetitions decode 100% either way."""
    rng = np.random.default_rng(seed)
    latents = rng.standard_normal(mode.n_latents)
    latents /= np.sqrt(np.mean(latents**2))
    tx = Modem().modulate(latents, mode, callsign=CALLSIGN)
    rx = hfchannel.apply_channel(tx, snr_db=snr_db, fading_preset=preset, seed=seed)

    start_frame = mode.n_frames // 3
    lo = LEADIN_SAMPLES + PREAMBLE_SAMPLES + HEADER_SAMPLES + start_frame * FRAME_SAMPLES
    win = rx[lo : lo + frames * FRAME_SAMPLES]
    return win, start_frame


def test_beacon_decodes_through_fading_on_a_short_window():
    mode = MODES["B"]
    m = Modem()
    ok = 0
    for t in range(TRIALS):
        win, start_frame = _blind_window(mode, "mpd", 8.0, 1000 * t)
        b = m.demodulate_blind(win).beacon
        ok += bool(
            b is not None
            and b.callsign == CALLSIGN
                and b.frame_index - b.chip_offset // CHIPS_PER_FRAME == start_frame
        )
    assert ok >= MIN_DECODES, f"{ok}/{TRIALS} beacon decodes, expected >= {MIN_DECODES}"


def test_multi_repetition_combining_is_coherent():
    """`_decode_combined` sums raw chips, not their signs.

    Only reachable with >= 3 superframe repetitions and only when no
    single repetition decodes alone, so this needs a window long enough
    for the grid (SUPERFRAME_LEN is 36.2 frames) and a channel harsh
    enough that single-shot fails -- 150 frames at mpd 0 dB. A whole
    transmission carries ~12 repetitions, where every variant is 1.00.

    Measured (`scripts/beacon_combine_sweep.py`, 40 trials/cell), mpd
    0 dB: equalized chips + sign 0.47, equalized + raw 0.17, MRC + sign
    0.57, MRC + raw 0.85. The two changes compound and neither is safe
    without the other, which is why this test and the one above are a
    pair.
    """
    mode = MODES["B"]
    m = Modem()
    ok = 0
    for t in range(TRIALS):
        win, start_frame = _blind_window(mode, "mpd", 0.0, 500 + 1000 * t, frames=150)
        b = m.demodulate_blind(win).beacon
        ok += bool(
            b is not None
            and b.callsign == CALLSIGN
                and b.frame_index - b.chip_offset // CHIPS_PER_FRAME == start_frame
        )
    assert ok >= 17, f"{ok}/{TRIALS} beacon decodes, expected >= 17"

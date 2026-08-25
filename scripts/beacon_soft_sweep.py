#!/usr/bin/env python3
"""Beacon decode success rate vs SNR, for the three soft-value weightings.

    python scripts/beacon_soft_sweep.py --trials 40

The beacon's soft chips come off the pilot channel estimate, and how
they are weighted decides whether a faded carrier poisons its Golay
codeword.

Now a single-column regression instrument: the comparison it was written
for is settled and the winner is what ships. Measured 2026-08-25, mode B,
60 trials/cell -- old `real(y)` against the shipped `real(raw*conj(h))`:

    awgn  0.0  0.77 -> 0.97     mpp  8.0  0.60 -> 0.90
    awgn -2.0  0.20 -> 0.47     mpp  4.0  0.23 -> 0.58
    mpg 12.0   0.67 -> 0.72     mpd  8.0  0.42 -> 0.90
    mpg  4.0   0.23 -> 0.28     mpd  4.0  0.05 -> 0.50

mpg moves least, as expected: at 0.5 ms delay the whole 1150 Hz band is
inside one coherence bandwidth, so there is no per-carrier fade to
deweight -- the beacon carrier fades with everything else.

Success is a *bit-exact* beacon result: callsign, mode and the absolute
frame index all correct. A wrong-but-returned beacon is counted as a
failure, not a decode, since downstream that is worse than nothing.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sstvae import hfchannel  # noqa: E402
from sstvae.config import (  # noqa: E402
    FRAME_SAMPLES,
    HEADER_SAMPLES,
    LEADIN_SAMPLES,
    MODES,
    PREAMBLE_SAMPLES,
)
from sstvae.modem import Modem  # noqa: E402
from sstvae.modem.beacon import MIN_FRAMES_FOR_SYNC  # noqa: E402

CALLSIGN = "K1ABC"
CHIPS_PER_FRAME = 5
# A hair over the window that *guarantees* a full superframe copy.
WINDOW_FRAMES = MIN_FRAMES_FOR_SYNC + 4


def one_trial(mode, preset, snr_db, seed):
    """True if this channel realization's beacon decodes bit-exactly.

    A short *blind* window, mid-transmission: that is where the beacon is
    actually load-bearing and where it is not already saved by six
    superframe repetitions. Over a whole mode-B transmission every
    weighting decodes 100% of the time in every cell, which measures
    nothing.

    """
    rng = np.random.default_rng(seed)
    m = Modem()
    latents = rng.standard_normal(mode.n_latents).astype(np.float64)
    latents /= np.sqrt(np.mean(latents**2))
    tx = m.modulate(latents, mode, callsign=CALLSIGN)
    rx = hfchannel.apply_channel(tx, snr_db=snr_db, fading_preset=preset, seed=seed)

    # Window of WINDOW_FRAMES starting at frame `start_frame`, aligned to
    # the frame grid so the truth for the decoded absolute frame index is
    # exactly start_frame.
    start_frame = mode.n_frames // 3
    frame0 = LEADIN_SAMPLES + PREAMBLE_SAMPLES + HEADER_SAMPLES
    lo = frame0 + start_frame * FRAME_SAMPLES
    win = rx[lo : lo + WINDOW_FRAMES * FRAME_SAMPLES]

    b = m.demodulate_blind(win).beacon
    return bool(
        b is not None
        and b.callsign == CALLSIGN
        and b.mode_index == mode.index
        and b.frame_index - b.chip_offset // CHIPS_PER_FRAME == start_frame
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--mode", default="B")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    mode = MODES[args.mode]
    cells = [(None, s) for s in (6.0, 3.0, 0.0, -2.0)]
    cells += [(p, s) for p in ("mpg", "mpp", "mpd") for s in (12.0, 8.0, 4.0)]

    print(f"mode {args.mode}, {args.trials} trials/cell, beacon bit-exact decode rate")
    print(f"{'channel':>10} {'SNR':>5}  {'rate':>6}")
    for preset, snr in cells:
        ok = sum(
            one_trial(mode, preset, snr, args.seed + 1000 * t)
            for t in range(args.trials)
        )
        print(f"{preset or 'awgn':>10} {snr:5.1f}  {ok / args.trials:6.2f}")


if __name__ == "__main__":
    main()

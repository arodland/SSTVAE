#!/usr/bin/env python3
"""Equal-gain (sign) vs coherent combining in the beacon's multi-repetition
fallback, now that the soft chips are maximal-ratio.

    python scripts/beacon_combine_sweep.py --trials 40

`_decode_combined` sums the *sign* of each repetition's chips because
the old equalized soft values could be enormous and random when a
repetition's channel estimate sat near a fade null -- one such
repetition dominated a raw sum and wrecked it. The chips are now
`real(raw * conj(h))`, so a faded repetition arrives *small* instead,
which is exactly what coherent combining wants. This asks whether the
workaround is still earning its keep.

Two measurements, because they answer different questions:

  forced   the fallback in isolation: single-shot decoding is disabled,
           so every trial exercises the combining path. This is the
           discriminating one -- good statistics, no confound.
  end2end  real `beacon.decode()`, harsh cells only. Confirms whether
           any difference survives single-shot decoding usually winning
           first, which is what actually ships.

Success is bit-exact: callsign, mode and absolute frame index.
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
from sstvae.modem import Modem, SyncError  # noqa: E402
from sstvae.modem import beacon  # noqa: E402
from sstvae.modem import modem as modem_mod  # noqa: E402

CALLSIGN = "K1ABC"
FRAME0 = LEADIN_SAMPLES + PREAMBLE_SAMPLES + HEADER_SAMPLES
CHIPS_PER_FRAME = 5
MODES_TO_COMBINE = ["sign", "raw", "norm"]


def soft_chips(mode, preset, snr_db, seed, window_frames):
    """One channel realization's beacon soft-chip stream, demodulated
    the way the receiver does it. Shared across the combining variants,
    so the comparison is paired -- identical chips in, only the
    combining differs.

    A window, not the whole transmission, and the *blind* path. A full
    mode-B transmission carries ~12 superframe repetitions, at which
    every variant is 1.00 in every cell; and going harsh enough to move
    that number breaks preamble acquisition long before it breaks the
    beacon. The regime where combining decides is 3-4 repetitions
    (SUPERFRAME_LEN is 36.2 frames), which is what --window sets.
    """
    rng = np.random.default_rng(seed)
    latents = rng.standard_normal(mode.n_latents)
    latents /= np.sqrt(np.mean(latents**2))
    m = Modem()
    tx = m.modulate(latents, mode, callsign=CALLSIGN)
    rx = hfchannel.apply_channel(tx, snr_db=snr_db, fading_preset=preset, seed=seed)

    # The soft chips aren't exposed, and adding a parameter to the
    # shipped API for a sweep would be the wrong trade. Capture them
    # from the one call the demodulator makes.
    captured = {}
    real_decode = beacon.decode

    def capture(chips, *a, **kw):
        captured["chips"] = chips
        return real_decode(chips, *a, **kw)

    start_frame = mode.n_frames // 3
    lo = FRAME0 + start_frame * FRAME_SAMPLES
    win = rx[lo : lo + window_frames * FRAME_SAMPLES]

    modem_mod.beacon.decode = capture
    try:
        m.demodulate_blind(win)
    except SyncError:
        # Blind acquisition, not the beacon: orthogonal to what is being
        # measured, so the trial is excluded rather than counted as a
        # combining failure.
        return None
    finally:
        modem_mod.beacon.decode = real_decode
    return captured.get("chips"), start_frame


def _ok(b, mode, expect_frame0=0):
    return bool(
        b is not None
        and b.callsign == CALLSIGN
        and b.mode_index == mode.index
        and b.frame_index - b.chip_offset // CHIPS_PER_FRAME == expect_frame0
    )


def run(mode, cells, trials, seed0, forced, window_frames):
    print(f"\n{'forced fallback' if forced else 'end to end'}: mode {mode.name}, "
          f"{window_frames}-frame window, {trials} trials/cell, bit-exact beacon decode")
    print(f"{'channel':>10} {'SNR':>5}  "
          + "  ".join(f"{c:>6}" for c in MODES_TO_COMBINE) + "       n")
    for preset, snr in cells:
        tally = {c: 0 for c in MODES_TO_COMBINE}
        n = 0
        for t in range(trials):
            got = soft_chips(mode, preset, snr, seed0 + 1000 * t, window_frames)
            if got is None or got[0] is None:
                continue
            chips, start_frame = got
            n += 1
            for c in MODES_TO_COMBINE:
                beacon.COMBINE_MODE = c
                if forced:
                    b = _combined_only(chips)
                else:
                    b = beacon.decode(chips)
                tally[c] += _ok(b, mode, start_frame)
        rates = "  ".join(
            f"{tally[c] / n:6.2f}" if n else "     -" for c in MODES_TO_COMBINE
        )
        print(f"{preset or 'awgn':>10} {snr:5.1f}  {rates}  {n:>3}/{trials}")


def _combined_only(chips):
    """`beacon.decode()` with the single-shot path removed, so the
    combining fallback is always what answers."""
    for off in beacon.find_sync(chips):
        r = beacon._decode_combined(chips, off)
        if r is not None:
            return r
    for off in beacon._folded_sync_phases(chips):
        r = beacon._decode_combined(chips, off)
        if r is not None:
            return r
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--mode", default="B")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--window", type=int, default=150,
                    help="window length in frames (~4 superframe repetitions)")
    args = ap.parse_args()

    mode = MODES[args.mode]
    # Harsh only: anywhere comfortable, every variant is 1.00 and the
    # fallback never runs at all.
    cells = [(None, s) for s in (0.0, -2.0, -4.0)]
    cells += [(p, s) for p in ("mpg", "mpp", "mpd") for s in (8.0, 4.0, 0.0)]

    run(mode, cells, args.trials, args.seed, True, args.window)
    run(mode, cells, args.trials, args.seed, False, args.window)


if __name__ == "__main__":
    main()

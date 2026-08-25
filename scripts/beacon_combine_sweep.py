#!/usr/bin/env python3
"""Beacon multi-repetition combining: a regression instrument.

    python scripts/beacon_combine_sweep.py --trials 40

`_decode_combined` sums each repetition's raw chip values -- coherent
(maximal-ratio) combining. It summed *signs* until 2026-08-25, correctly
for what it was given: the chips then came off the zero-forcing
equalizer, so a repetition near a fade null contributed an enormous,
essentially random magnitude and dominated a raw sum. The chips are
`real(raw * conj(h))` now, so a faded repetition arrives small instead,
and the ordering inverts.

Measured 2026-08-25, mode B, 40 trials/cell, forced fallback:

    chips           equalized  equalized      MRC      MRC
    combining            sign        raw     sign      raw
    awgn -4.0            0.45       0.20     0.50     0.65
    mpg   0.0            0.72       0.30     0.75     0.82
    mpp   0.0            0.57       0.23     0.72     0.93
    mpd   0.0            0.40       0.15     0.53     0.90

That comparison is settled, so this now measures the shipped path only.
To re-run it, swap the two lines in `beacon._decode_combined` /
`_search_counter_chunk` back to `np.sign(...)` and/or revert
`modem.py`'s beacon chip to `np.real(y[BEACON_CARRIER])`; a knob in
product code for a one-off comparison would be the wrong trade.

Two measurements, because they answer different questions:

  forced   the fallback in isolation: single-shot decoding is disabled,
           so every trial exercises the combining path. Good statistics,
           no confound.
  end2end  real `beacon.decode()`. Confirms whether any difference
           survives single-shot decoding usually winning first.

The window matters and is not a detail. `_decode_combined` is dead code
below 3 repetitions (`_repetition_grid`) and saturated above ~6, and
SUPERFRAME_LEN is 36.2 frames -- so a whole mode-B transmission (~12
repetitions) scores 1.00 for every variant in every cell. It runs on
the blind path because going harsh enough to move that number breaks
preamble acquisition long before it breaks the beacon.

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
    print(f"{'channel':>10} {'SNR':>5}  {'rate':>6}       n")
    for preset, snr in cells:
        ok = 0
        n = 0
        for t in range(trials):
            got = soft_chips(mode, preset, snr, seed0 + 1000 * t, window_frames)
            if got is None or got[0] is None:
                continue
            chips, start_frame = got
            n += 1
            b = _combined_only(chips) if forced else beacon.decode(chips)
            ok += _ok(b, mode, start_frame)
        rate = f"{ok / n:6.2f}" if n else "     -"
        print(f"{preset or 'awgn':>10} {snr:5.1f}  {rate}  {n:>3}/{trials}")


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

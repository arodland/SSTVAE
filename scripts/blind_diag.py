#!/usr/bin/env python3
"""Offline blind-acquisition diagnostic for a recorded capture.

Feeds a WAV through the same BlindAccumulator the live receive loop
uses (per-mode decay timescales, chunked pushes) and prints what the
loop itself never shows: the score trajectory over time, the winning
CFO and phase, the shape of the fold's peak, and what happens when the
final state is handed to the demodulator. Built for the field failure
mode where blind acquisition underperforms on real hardware while the
header path is perfect -- the shapes distinguish the mechanisms:

  - clean narrow peak (mainlobe ~+-4 samples), score below threshold
    -> the pilot genuinely isn't arriving at the strength the SNR
    implies, or something broadband is raising the median floor
  - peak smeared over tens of samples, score *decaying* as more audio
    accumulates, lock phase walking between polls -> sample-clock
    mismatch (only blind suffers; the header path has a drift tracker;
    measured: 200 ppm reads 34 at 5 s and 6.6 at 35 s, phase walking
    ~1.6 samples/s)
  - several distinct peaks                     -> multipath / room echo
  - peak at a range-edge CFO bin               -> station offset outside
    the +-55 Hz default search (try --wide)
  - healthy lock, beacon never decodes         -> payload-side problem

Usage:
    python scripts/blind_diag.py capture.wav [--wide] [--chunk-s 5]

The capture should contain the blind-only reception (transmission with
its start missing). Any sample rate; resampled to FS if needed.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from sstvae import wavio
from sstvae.config import (
    BLIND_MAX_OFFSET_HZ,
    BLIND_SCORE_THRESHOLD,
    BLIND_WIDE_MAX_OFFSET_HZ,
    FRAME_SAMPLES,
    FS,
    MODES,
    MODES_BY_INDEX,
)
from sstvae.modem import Modem
from sstvae.modem.dsp import to_baseband, to_baseband_at
from sstvae.modem.sync import BlindAccumulator, SyncError


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("wav", type=Path)
    ap.add_argument("--wide", action="store_true",
                    help=f"search +-{BLIND_WIDE_MAX_OFFSET_HZ:.0f} Hz instead of "
                         f"+-{BLIND_MAX_OFFSET_HZ:.0f}")
    ap.add_argument("--chunk-s", type=float, default=5.0,
                    help="push granularity, like the live loop's poll (default 5)")
    args = ap.parse_args()

    x = wavio.read_wav(str(args.wav))  # mono float at FS, resampled if needed
    print(f"{len(x) / FS:.1f} s of audio")

    # The same construction rx/engine.py uses.
    timescales = [m.duration_s for m in MODES.values()]
    acc = BlindAccumulator(
        max_offset_hz=BLIND_WIDE_MAX_OFFSET_HZ if args.wide else BLIND_MAX_OFFSET_HZ,
        window_s=timescales,
    )

    chunk = int(args.chunk_s * FS)
    print(f"\nscore trajectory (lock threshold {BLIND_SCORE_THRESHOLD}):")
    pos = 0
    while pos < len(x):
        acc.push(to_baseband_at(x[pos : pos + chunk], pos), pos)
        pos += chunk
        score = acc.best_score()
        try:
            r = acc.result()
            lock = f"LOCK  cfo {r.freq_offset:+6.1f} Hz  phase {r.frame_start}"
        except SyncError:
            lock = ""
        print(f"  t={pos / FS:6.1f}s  score {score:6.2f}  {lock}")

    # The fold's peak shape, from the accumulator's own state: which
    # mechanism is suppressing the score is usually visible here.
    scores = acc._folded.max(axis=2) / (np.median(acc._folded, axis=2) + 1e-12)
    t, i = np.unravel_index(np.argmax(scores), scores.shape)
    row = acc._folded[t, i]
    peak = int(np.argmax(row))
    med = float(np.median(row))
    print(f"\nbest cell: timescale {timescales[t]:.0f}s, "
          f"cfo bin {acc._freqs[i]:+.1f} Hz "
          f"({'RANGE EDGE' if i in (0, len(acc._freqs) - 1) else 'interior'})")
    print(f"peak/median {row[peak] / (med + 1e-12):.2f}")
    print("fold around the peak (values / median, offsets in samples):")
    for lo in range(-24, 25, 8):
        vals = "  ".join(
            f"{row[(peak + j) % FRAME_SAMPLES] / (med + 1e-12):6.2f}"
            for j in range(lo, lo + 8)
        )
        print(f"  {lo:+4d}..{lo + 7:+4d}: {vals}")
    above3 = int(np.count_nonzero(row > 3 * med))
    print(f"bins above 3x median: {above3} "
          f"(~10 = clean peak; several tens = smeared or multiple paths)")

    # Hand the final state to the demodulator, exactly as the loop would.
    try:
        r = acc.result()
    except SyncError as e:
        print(f"\nno lock: {e}")
        return 0
    rb = Modem().demodulate_blind(x, acquisition=r)
    if rb.beacon is None:
        print("\nlocked, but the beacon did not decode from this window")
    else:
        spec = MODES_BY_INDEX.get(rb.beacon.mode_index)
        print(f"\nbeacon: frame {rb.beacon.frame_index}, "
              f"mode {spec.name if spec else '?'}, "
              f"callsign '{rb.beacon.callsign}', snr {rb.snr_db:.1f} dB")

    # And the header path on the same audio, for the cross-check number:
    # its freq offset measures the true station offset, and on an
    # acoustic bench that is nearly all sample-clock mismatch
    # (offset_hz / 1525 = the clock error).
    try:
        rh = Modem().demodulate(x)
        print(f"header path: mode {rh.mode.name}, {rh.frames_received} frames, "
              f"freq offset {rh.freq_offset:+.2f} Hz, snr {rh.snr_db:.1f} dB")
    except SyncError:
        print("header path: no preamble in this capture (expected for a "
              "blind-only recording)")

    _native_cross_check(x, args, timescales, chunk)
    return 0


def _native_cross_check(x, args, timescales, chunk) -> None:
    """Everything above again, through the C++ implementations, if the
    extension module is built (tools/build_native.sh) -- for the field
    case where the Python listener decodes a capture the native app
    fails on: same file, both implementations, side by side. The suite's
    parity tests all pass on synthetic audio, so if the two disagree
    here, the disagreement is a property of this real capture and this
    file IS the reproduction -- keep it."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent /
                           "native" / "build" / "python"))
    try:
        import sstvae_native as native
    except ImportError:
        print("\n[no native cross-check: extension module not built -- "
              "tools/build_native.sh]")
        return

    print("\n--- native (C++) cross-check ---")
    acc = native.sync.BlindAccumulator(
        max_offset_hz=BLIND_WIDE_MAX_OFFSET_HZ if args.wide else BLIND_MAX_OFFSET_HZ,
        window_s=[float(t) for t in timescales],
    )
    pos = 0
    while pos < len(x):
        acc.push(to_baseband_at(x[pos : pos + chunk], pos), pos)
        pos += chunk
        score = acc.best_score()
        try:
            frame_start, freq_offset, metric = acc.result()
            lock = f"LOCK  cfo {freq_offset:+6.1f} Hz  phase {frame_start}"
        except Exception:
            lock = ""
        print(f"  t={pos / FS:6.1f}s  score {score:6.2f}  {lock}")

    try:
        acq = acc.result()
    except Exception as e:
        print(f"native: no lock: {e}")
        return
    d = native.modem.demodulate_blind(
        np.asarray(x, dtype=np.float64), None,
        (int(acq[0]), float(acq[1]), float(acq[2])), "off")
    if d["beacon"] is None:
        print("native: locked, but the beacon did not decode from this window")
    else:
        chip_offset, frame_index, callsign, mode_index = d["beacon"]
        spec = MODES_BY_INDEX.get(mode_index)
        print(f"native beacon: frame {frame_index}, "
              f"mode {spec.name if spec else '?'}, callsign '{callsign}', "
              f"snr {d['snr_db']:.1f} dB")


if __name__ == "__main__":
    raise SystemExit(main())

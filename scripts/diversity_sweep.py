#!/usr/bin/env python3
"""Measure the gain from diversity reception: two independent receive
branches of the *same* transmission (independent noise/fading, each
re-acquiring its own copy of the preamble), maximal-ratio combined via
`sstvae.modem.diversity.combine_demod_results`.

    python scripts/diversity_sweep.py                # AWGN + mpp tables
    python scripts/diversity_sweep.py --csv sweep.csv

Runs entirely in the latent domain (no codec/model download needed --
this is a modem-level measurement, and `latent_snr_db` is the same
metric `tests/conftest.py` calibrates every end-to-end test against).
Per point this demodulates 2 x --trials waveforms and combines them, so
it costs about twice `scripts/snr_sweep.py` per point plus the combine,
which is cheap.

See docs/diversity-reception.md for the derivation this is checking and
for the last published numbers.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sstvae import hfchannel  # noqa: E402
from sstvae.config import MODES, SNR_REF_BW_HZ  # noqa: E402
from sstvae.modem import Modem, SyncError  # noqa: E402
from sstvae.modem.diversity import combine_demod_results  # noqa: E402

AWGN_SNRS = [6.0, 3.0, 0.0, -2.0, -4.0]
FADING_SNRS = [10.0, 6.0, 3.0]
DEFAULT_TRIALS = 12


def unit_latents(mode: str, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    lat = rng.normal(size=MODES[mode].n_latents)
    return lat / np.sqrt(np.mean(lat**2))


def latent_snr_db(sent: np.ndarray, got: np.ndarray, w: np.ndarray | None = None) -> float:
    mask = np.ones_like(sent, dtype=bool) if w is None else (w > 0)
    if not mask.any():
        return float("-inf")
    err = np.mean((sent[mask] - got[mask]) ** 2)
    if err <= 0:
        return float("inf")
    return 10 * np.log10(np.mean(sent[mask] ** 2) / err)


def run_point(modem, spec, snr_db, fading, trials, seed_base):
    """Mean single-branch SNR (the better of the two, what an operator
    would get from either antenna alone), mean combined SNR, and how
    many trials both branches acquired."""
    single, combined_scores, both_locked = [], [], 0
    for t in range(trials):
        lat = unit_latents(spec.name, seed_base + t)
        x = modem.modulate(lat, spec)
        y1 = hfchannel.apply_channel(
            x, snr_db=snr_db, fading_preset=fading, seed=seed_base + 1000 + t
        )
        y2 = hfchannel.apply_channel(
            x, snr_db=snr_db, fading_preset=fading, seed=seed_base + 2000 + t
        )
        try:
            r1 = modem.demodulate(y1)
            r2 = modem.demodulate(y2)
        except SyncError:
            continue
        both_locked += 1
        s1 = latent_snr_db(lat, r1.latents, r1.weights)
        s2 = latent_snr_db(lat, r2.latents, r2.weights)
        single.append(max(s1, s2))
        combined = combine_demod_results([r1, r2])
        combined_scores.append(latent_snr_db(lat, combined.latents, combined.weights))
    mean_single = float(np.mean(single)) if single else None
    mean_combined = float(np.mean(combined_scores)) if combined_scores else None
    return mean_single, mean_combined, both_locked


def fmt(single, combined, locked, total):
    if single is None:
        return "—"
    gain = combined - single
    text = f"{single:5.1f} -> {combined:5.1f} dB (+{gain:.1f})"
    if locked != total:
        text += f" [{locked}/{total}]"
    return text


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--modes", default="ABC")
    ap.add_argument("--fading", default="mpp", help="preset for the second table")
    ap.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    ap.add_argument("--seed", type=int, default=5000)
    ap.add_argument("--csv", type=Path, default=None)
    args = ap.parse_args()

    modem = Modem()
    modes = [MODES[m] for m in args.modes]
    print(f"{args.trials} trials/point, latent SNR referenced to "
          f"{SNR_REF_BW_HZ:.0f} Hz noise bandwidth\n", file=sys.stderr)

    rows = []
    tables = {}
    for label, snrs, fading in (
        ("awgn", AWGN_SNRS, None),
        (args.fading, FADING_SNRS, args.fading),
    ):
        table = {}
        for spec in modes:
            cells = []
            for snr_db in snrs:
                single, combined, locked = run_point(
                    modem, spec, snr_db, fading, args.trials, args.seed
                )
                cells.append((single, combined, locked, args.trials))
                rows.append({
                    "channel": label, "mode": spec.name, "snr_db": snr_db,
                    "single_branch_db": "" if single is None else f"{single:.2f}",
                    "combined_db": "" if combined is None else f"{combined:.2f}",
                    "gain_db": "" if single is None else f"{combined - single:.2f}",
                    "both_locked": f"{locked}/{args.trials}",
                })
                print(
                    f"  {label:>4} mode {spec.name} {snr_db:>5.1f} dB: "
                    f"{fmt(single, combined, locked, args.trials)}",
                    file=sys.stderr,
                )
            table[spec.name] = cells
        tables[label] = (snrs, table)

    for label in ("awgn", args.fading):
        snrs, table = tables[label]
        heads = " | ".join(f"{s:.0f} dB" for s in snrs)
        print(f"\n**{label}** (single branch -> combined, both in "
              f"latent SNR dB):\n")
        print(f"| Mode | {heads} |")
        print("|---|" + "---|" * len(snrs))
        for spec in modes:
            cells = " | ".join(
                "—" if s is None else f"{s:.1f} -> {c:.1f} (+{c - s:.1f})"
                for s, c, _, _ in table[spec.name]
            )
            print(f"| {spec.name} | {cells} |")

    if args.csv:
        import csv

        with open(args.csv, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nwrote {args.csv}", file=sys.stderr)


if __name__ == "__main__":
    main()

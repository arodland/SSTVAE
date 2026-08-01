#!/usr/bin/env python3
"""Paired A/B roundtrip comparison of two .pt checkpoints.

    python scripts/ab_checkpoint_sweep.py A.pt B.pt --csv ab.csv

Runs the whole path per point -- encode, modulate, simulated channel,
demodulate, decode -- for both checkpoints over the same validation
images and the *same channel seeds*, so every cell is a paired
comparison and the difference is attributable to the model alone.

The paired standard error is over per-image deltas, which is much
tighter than differencing two independent means: the image-to-image
PSNR spread (several dB) cancels.

SNRs are in `config.SNR_REF_BW_HZ`.
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sstvae import hfchannel  # noqa: E402
from sstvae.codec import load_codec, pad_to_full, reconstruct  # noqa: E402
from sstvae.config import MODES, SNR_REF_BW_HZ  # noqa: E402
from sstvae.images import fit_image  # noqa: E402
from sstvae.modem import Modem, SyncError  # noqa: E402

# (label, snr_db or None, fading preset or None)
CONDITIONS = [
    ("clean", None, None),
    ("awgn 20 dB", 20.0, None),
    ("awgn 10 dB", 10.0, None),
    ("awgn 6 dB", 6.0, None),
    ("awgn 3 dB", 3.0, None),
    ("awgn 0 dB", 0.0, None),
    ("mpg 10 dB", 10.0, "mpg"),
    ("mpp 10 dB", 10.0, "mpp"),
    ("mpp 6 dB", 6.0, "mpp"),
    ("mpd 6 dB", 6.0, "mpd"),
]


def psnr(a, b) -> float:
    x, y = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    mse = np.mean((x - y) ** 2)
    return float("inf") if mse == 0 else 10 * np.log10(255.0**2 / mse)


def load_images(repo: str, split: str, n: int):
    from datasets import load_dataset

    ds = load_dataset(repo, split=split)
    return [fit_image(ds[i]["image"]) for i in range(min(n, len(ds)))]


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("model_a")
    ap.add_argument("model_b")
    ap.add_argument("--dataset", default="arodland/coco640-sstvae")
    ap.add_argument("--split", default="validation")
    ap.add_argument("--n-images", type=int, default=32)
    ap.add_argument("--modes", default="ABC")
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--csv", type=Path, default=None)
    args = ap.parse_args()

    images = load_images(args.dataset, args.split, args.n_images)
    print(
        f"{len(images)} images from {args.dataset}:{args.split}, "
        f"SNR referenced to {SNR_REF_BW_HZ:.0f} Hz",
        file=sys.stderr,
    )

    names = [args.model_a, args.model_b]
    codecs = [load_codec(p) for p in names]
    modem = Modem()
    specs = [MODES[m] for m in args.modes]

    # Encode once per (model, image); every mode is a prefix of the same
    # latent vector.
    latents = [[c.encode(img) for img in images] for c in codecs]

    rows = []
    for spec in specs:
        waves = [
            [modem.modulate(lat[: spec.n_latents], spec) for lat in per_model]
            for per_model in latents
        ]
        for label, snr_db, fading in CONDITIONS:
            # Per-image PSNR for each model; NaN where sync failed.
            got = np.full((2, len(images)), np.nan)
            for i, src in enumerate(images):
                for m in range(2):
                    y = waves[m][i]
                    if snr_db is not None or fading is not None:
                        y = hfchannel.apply_channel(
                            y, snr_db=snr_db, fading_preset=fading,
                            seed=args.seed + i,
                        )
                    try:
                        r = modem.demodulate(y)
                    except SyncError:
                        continue
                    out = reconstruct(
                        codecs[m], pad_to_full(r.latents), pad_to_full(r.weights)
                    )
                    got[m, i] = psnr(src, out)

            sync = [int(np.count_nonzero(~np.isnan(got[m]))) for m in range(2)]
            means = [
                float(np.nanmean(got[m])) if sync[m] else float("nan")
                for m in range(2)
            ]
            # Paired delta over images where *both* models synced.
            both = ~np.isnan(got[0]) & ~np.isnan(got[1])
            n_both = int(np.count_nonzero(both))
            if n_both:
                d = got[1][both] - got[0][both]
                delta = float(d.mean())
                sem = float(d.std(ddof=1) / np.sqrt(n_both)) if n_both > 1 else float("nan")
                wins = int(np.count_nonzero(d > 0))
            else:
                delta = sem = float("nan")
                wins = 0

            rows.append({
                "mode": spec.name, "condition": label,
                "psnr_a": means[0], "psnr_b": means[1],
                "delta": delta, "sem": sem,
                "n_paired": n_both, "b_wins": wins,
                "sync_a": f"{sync[0]}/{len(images)}",
                "sync_b": f"{sync[1]}/{len(images)}",
            })
            print(
                f"  mode {spec.name} {label:>10}: "
                f"A {means[0]:5.2f}  B {means[1]:5.2f}  "
                f"delta {delta:+5.2f} +/- {sem:.2f}  "
                f"B wins {wins}/{n_both}  "
                f"sync {sync[0]}/{sync[1]} of {len(images)}",
                file=sys.stderr, flush=True,
            )

    print(f"\nA = {names[0]}\nB = {names[1]}\n")
    print("PSNR dB, mean over the validation images; delta = B - A, paired.\n")
    print("| Mode | Condition | A | B | delta | +/- | B wins | sync A/B |")
    print("|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(
            f"| {r['mode']} | {r['condition']} | {r['psnr_a']:.2f} | "
            f"{r['psnr_b']:.2f} | {r['delta']:+.2f} | {r['sem']:.2f} | "
            f"{r['b_wins']}/{r['n_paired']} | {r['sync_a']} / {r['sync_b']} |"
        )

    if args.csv:
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {args.csv}", file=sys.stderr)


if __name__ == "__main__":
    main()

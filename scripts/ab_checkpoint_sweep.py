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

**Both metrics are reported by default, and that is deliberate.** PSNR
alone cannot judge anything that trades distortion for perceptual
quality -- a perceptual loss term is *supposed* to cost PSNR (PE loss's
own paper gives up 0.33 dB for a 5% LPIPS improvement), so a comparison
scored on PSNR alone answers a question nobody asked. `--no-lpips` opts
out for speed; an unavailable `lpips` is an error rather than a silently
dropped column, because a table missing the metric you meant to decide
on looks exactly like a table where the metric came out even. This costs
no new dependency: `load_images` imports `datasets`, so the script has
always required the `train` extra, which is where `lpips` lives too.

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


def make_lpips(device: str):
    """A `(PIL, PIL) -> float` LPIPS scorer, or raise with what to install.

    Two choices worth stating, because both differ from what
    `scripts/train.py` does with the same package:

    - **VGG, matching the training objective's net**, not AlexNet.
      AlexNet is what the LPIPS README recommends for reporting, but a
      metric that disagrees with the term being optimized would make a
      loss sweep unreadable in the one case it exists to read.
    - **Whole 640x480 frames, not train.py's random 256px crop.** That
      crop is a compute-and-calibration tradeoff for an objective
      evaluated millions of times; an evaluation wants determinism and
      full coverage, and a random crop would put sampling noise inside a
      paired delta whose whole point is that it has none.
    """
    try:
        import lpips as lpips_pkg
        import torch
    except Exception as exc:  # noqa: BLE001 - environment, not logic
        # Not just ImportError: a torch/torchvision version mismatch
        # raises RuntimeError from deep inside torchvision's registration
        # (`operator torchvision::nms does not exist`), which is the
        # likelier failure on a ROCm box where the two are pinned
        # separately. train.py's own make_lpips swallows exactly this and
        # trains MSE-only, so the message names both possibilities.
        raise SystemExit(
            f"could not load LPIPS ({type(exc).__name__}: {exc}).\n"
            "Install the train extra (`uv sync --extra train`) or fix a "
            "torch/torchvision version mismatch, or pass --no-lpips."
        ) from exc

    net = lpips_pkg.LPIPS(net="vgg").to(device).eval()

    def to_tensor(img):
        x = np.asarray(img, dtype=np.float32) / 255.0
        t = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).to(device)
        return t * 2.0 - 1.0

    def score(a, b) -> float:
        with torch.no_grad():
            return float(net(to_tensor(a), to_tensor(b)).item())

    return score


def pick_device(requested: str) -> str:
    if requested != "auto":
        return requested
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def paired(got: np.ndarray, lower_is_better: bool) -> dict:
    """Per-model means and the paired B-A delta over the images where
    *both* models synced. `got` is (2, n_images), NaN where sync failed.

    `wins` counts images where B is **better**, which is the opposite
    comparison for LPIPS -- a column that silently means one thing in
    one row and its negation in the next is worse than no column.
    """
    sync = [int(np.count_nonzero(~np.isnan(got[m]))) for m in range(2)]
    means = [
        float(np.nanmean(got[m])) if sync[m] else float("nan") for m in range(2)
    ]
    both = ~np.isnan(got[0]) & ~np.isnan(got[1])
    n_both = int(np.count_nonzero(both))
    if not n_both:
        return {"a": means[0], "b": means[1], "delta": float("nan"),
                "sem": float("nan"), "wins": 0, "n_paired": 0, "sync": sync}
    d = got[1][both] - got[0][both]
    return {
        "a": means[0], "b": means[1],
        "delta": float(d.mean()),
        "sem": float(d.std(ddof=1) / np.sqrt(n_both)) if n_both > 1 else float("nan"),
        "wins": int(np.count_nonzero(d < 0 if lower_is_better else d > 0)),
        "n_paired": n_both, "sync": sync,
    }


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
    ap.add_argument(
        "--lpips",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="also score LPIPS (VGG, whole frame). On by default: a "
        "perceptual-loss comparison decided on PSNR alone is the "
        "failure this exists to prevent",
    )
    ap.add_argument(
        "--device",
        default="auto",
        help="torch device for LPIPS (auto = cuda if available)",
    )
    args = ap.parse_args()

    lpips_fn = make_lpips(pick_device(args.device)) if args.lpips else None

    images = load_images(args.dataset, args.split, args.n_images)
    print(
        f"{len(images)} images from {args.dataset}:{args.split}, "
        f"SNR referenced to {SNR_REF_BW_HZ:.0f} Hz"
        + (f", LPIPS on {pick_device(args.device)}" if lpips_fn else ""),
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
            # Per-image PSNR (and LPIPS) for each model; NaN where sync
            # failed. Both metrics score the same decoded picture, so a
            # sync failure is one hole in both, never two different ones.
            got = np.full((2, len(images)), np.nan)
            got_lp = np.full((2, len(images)), np.nan)
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
                    if lpips_fn is not None:
                        got_lp[m, i] = lpips_fn(src, out)

            p = paired(got, lower_is_better=False)
            row = {
                "mode": spec.name, "condition": label,
                "psnr_a": p["a"], "psnr_b": p["b"],
                "delta": p["delta"], "sem": p["sem"],
                "n_paired": p["n_paired"], "b_wins": p["wins"],
                "sync_a": f"{p['sync'][0]}/{len(images)}",
                "sync_b": f"{p['sync'][1]}/{len(images)}",
            }
            line = (
                f"  mode {spec.name} {label:>10}: "
                f"PSNR A {p['a']:5.2f} B {p['b']:5.2f} "
                f"d {p['delta']:+5.2f} +/- {p['sem']:.2f} "
                f"B wins {p['wins']}/{p['n_paired']}"
            )
            if lpips_fn is not None:
                q = paired(got_lp, lower_is_better=True)
                row.update({
                    "lpips_a": q["a"], "lpips_b": q["b"],
                    "lpips_delta": q["delta"], "lpips_sem": q["sem"],
                    "lpips_b_wins": q["wins"],
                })
                line += (
                    f" | LPIPS A {q['a']:.4f} B {q['b']:.4f} "
                    f"d {q['delta']:+.4f} +/- {q['sem']:.4f} "
                    f"B wins {q['wins']}/{q['n_paired']}"
                )
            rows.append(row)
            print(
                line + f" | sync {p['sync'][0]}/{p['sync'][1]} of {len(images)}",
                file=sys.stderr, flush=True,
            )

    print(f"\nA = {names[0]}\nB = {names[1]}\n")
    print("Mean over the validation images; delta = B - A, paired.")
    print("PSNR dB, higher is better.", end=" ")
    if lpips_fn is not None:
        # Stated because the delta's sign flips meaning between the two
        # columns and the reader has no other way to know.
        print("LPIPS (VGG, whole frame), **lower** is better, "
              "so a negative delta is B winning.", end=" ")
    print("\"B wins\" counts images where B is better, either way.\n")

    head = "| Mode | Condition | PSNR A | PSNR B | delta | +/- | B wins |"
    if lpips_fn is not None:
        head += " LPIPS A | LPIPS B | delta | +/- | B wins |"
    head += " sync A/B |"
    print(head)
    print("|" + "---|" * (head.count("|") - 1))
    for r in rows:
        cells = (
            f"| {r['mode']} | {r['condition']} | {r['psnr_a']:.2f} | "
            f"{r['psnr_b']:.2f} | {r['delta']:+.2f} | {r['sem']:.2f} | "
            f"{r['b_wins']}/{r['n_paired']} |"
        )
        if lpips_fn is not None:
            cells += (
                f" {r['lpips_a']:.4f} | {r['lpips_b']:.4f} | "
                f"{r['lpips_delta']:+.4f} | {r['lpips_sem']:.4f} | "
                f"{r['lpips_b_wins']}/{r['n_paired']} |"
            )
        print(cells + f" {r['sync_a']} / {r['sync_b']} |")

    if args.csv:
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {args.csv}", file=sys.stderr)


if __name__ == "__main__":
    main()

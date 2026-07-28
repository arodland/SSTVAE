#!/usr/bin/env python3
"""Measure PSNR vs channel SNR, end to end, and print the README tables.

    python scripts/snr_sweep.py                     # the published tables
    python scripts/snr_sweep.py --csv sweep.csv     # also dump raw rows

Runs the whole path per point -- encode, modulate, simulated channel,
demodulate, decode -- and averages PSNR over a fixed set of validation
images the model never saw in training.

This exists because the README's performance tables were originally
produced ad hoc, which made them impossible to re-derive when the SNR
convention changed. SNRs are in `config.SNR_REF_BW_HZ`.

Costs a few minutes: the transmit waveform for each (image, mode) is
built once and reused across every SNR point, and each image is encoded
only once (the modes are nested prefixes of the same latent vector), so
the demodulator dominates the runtime.
"""

import argparse
import io
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image  # noqa: E402

from sstvae import hfchannel  # noqa: E402
from sstvae.codec import MODEL_HELP, load_torch_model, pad_to_full, reconstruct  # noqa: E402
from sstvae.config import MODES, SNR_REF_BW_HZ  # noqa: E402
from sstvae.images import fit_image, image_to_tensor  # noqa: E402
from sstvae.models import SSTVAE  # noqa: E402
from sstvae.modem import Modem, SyncError  # noqa: E402

DEFAULT_IMAGES = Path("data/val2017.zip")
# COCO val2017: never in the training split (which is train2017). Taken
# in sorted order rather than sampled, so the set is stable across runs
# and machines.
# 25 rather than a handful: near the acquisition threshold the useful
# output is a success *rate*, and six attempts can only ever resolve it
# to within a sixth.
DEFAULT_N_IMAGES = 25
AWGN_SNRS = [None, 20.0, 10.0, 6.0, 3.0, 0.0, -2.0]  # None = no noise
FADING_SNRS = [20.0, 10.0, 6.0]


def load_images(source: Path, n: int) -> list[Image.Image]:
    """`source` may be a zip (COCO's val2017.zip) or a directory."""
    if source.suffix == ".zip":
        with zipfile.ZipFile(source) as z:
            names = sorted(x for x in z.namelist() if x.lower().endswith((".jpg", ".png")))
            blobs = [z.read(name) for name in names[:n]]
        return [fit_image(Image.open(io.BytesIO(b))) for b in blobs]
    files = sorted(
        p for p in source.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    return [fit_image(Image.open(p)) for p in files[:n]]


def psnr(a: Image.Image, b: Image.Image) -> float:
    x, y = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    mse = np.mean((x - y) ** 2)
    return float("inf") if mse == 0 else 10 * np.log10(255.0**2 / mse)


def encode_all(model, images) -> list[np.ndarray]:
    """One full-length latent vector per image; every mode is a prefix."""
    out = []
    for img in images:
        with torch.no_grad():
            z = model.encoder(image_to_tensor(img)[None])
        out.append(SSTVAE.latents_to_flat(z)[0].numpy().astype(np.float64))
    return out


def run_point(model, modem, waveforms, images, snr_db, fading, seed_base):
    """Mean PSNR over the image set, and how many acquired sync."""
    scores, acquired = [], 0
    for i, (wave, src) in enumerate(zip(waveforms, images)):
        y = wave
        if snr_db is not None or fading is not None:
            y = hfchannel.apply_channel(
                wave, snr_db=snr_db, fading_preset=fading, seed=seed_base + i
            )
        try:
            r = modem.demodulate(y)
        except SyncError:
            continue
        acquired += 1
        out = reconstruct(model, pad_to_full(r.latents), pad_to_full(r.weights))
        scores.append(psnr(src, out))
    return (float(np.mean(scores)) if scores else None), acquired


def fmt(value, acquired=None, total=None) -> str:
    """PSNR, with the acquisition rate whenever it wasn't 100%.

    The mean is over the images that actually synced, so on its own it
    flatters the points near threshold -- a failed acquisition is no
    picture at all, not a low-PSNR one. Quoted together, the pair says
    the useful thing: how often you get a picture, and how good it is
    when you do.
    """
    if value is None:
        return "—"
    text = f"{value:.1f}"
    if acquired is not None and total is not None and acquired != total:
        text += f" ({acquired}/{total})"
    return text


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=None, help=MODEL_HELP)
    ap.add_argument("--images", type=Path, default=DEFAULT_IMAGES)
    ap.add_argument("--n-images", type=int, default=DEFAULT_N_IMAGES)
    ap.add_argument("--modes", default="ABC")
    ap.add_argument("--fading", default="mpp", help="preset for the second table")
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--csv", type=Path, default=None)
    args = ap.parse_args()

    images = load_images(args.images, args.n_images)
    if not images:
        raise SystemExit(f"no images found in {args.images}")
    print(f"{len(images)} images, SNR referenced to {SNR_REF_BW_HZ:.0f} Hz\n",
          file=sys.stderr)

    model = load_torch_model(args.model)
    modem = Modem()
    latents = encode_all(model, images)
    modes = [MODES[m] for m in args.modes]

    rows = []
    tables = {}
    for label, snrs, fading in (
        ("awgn", AWGN_SNRS, None),
        (args.fading, FADING_SNRS, args.fading),
    ):
        table = {}
        for spec in modes:
            # Built once per (image, mode) and reused at every SNR.
            waves = [modem.modulate(lat[: spec.n_latents], spec) for lat in latents]
            cells = []
            for snr_db in snrs:
                mean_psnr, acquired = run_point(
                    model, modem, waves, images, snr_db, fading, args.seed
                )
                cells.append((mean_psnr, acquired, len(images)))
                rows.append({
                    "channel": label, "mode": spec.name,
                    "snr_db": "clean" if snr_db is None else snr_db,
                    "psnr_db": "" if mean_psnr is None else f"{mean_psnr:.2f}",
                    "acquired": f"{acquired}/{len(images)}",
                })
                print(
                    f"  {label:>4} mode {spec.name} "
                    f"{'clean' if snr_db is None else f'{snr_db:>5.1f} dB'}: "
                    f"PSNR {fmt(mean_psnr):>5}  sync {acquired}/{len(images)}",
                    file=sys.stderr,
                )
            table[spec.name] = cells
        tables[label] = (snrs, table)

    snrs, table = tables["awgn"]
    heads = " | ".join("clean" if s is None else f"{s:.0f} dB" for s in snrs)
    print(f"**AWGN** (SNR in a {SNR_REF_BW_HZ / 1000:.1f} kHz noise bandwidth):\n")
    print(f"| Mode | Time | {heads} |")
    print("|---|---|" + "---|" * len(snrs))
    for spec in modes:
        cells = " | ".join(fmt(*c) for c in table[spec.name])
        print(f"| {spec.name} | {spec.duration_s:.0f} s | {cells} |")

    snrs, table = tables[args.fading]
    heads = " | ".join(f"{s:.0f} dB" for s in snrs)
    print(f"\n**Watterson `{args.fading}`**:\n")
    print(f"| Mode | {heads} |")
    print("|---|" + "---|" * len(snrs))
    for spec in modes:
        cells = " | ".join(fmt(*c) for c in table[spec.name])
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

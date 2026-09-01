#!/usr/bin/env python3
"""Measure PSNR vs channel SNR, end to end, and print the README tables.

    python scripts/snr_sweep.py                     # the published tables
    python scripts/snr_sweep.py --csv sweep.csv     # also dump raw rows
    python scripts/snr_sweep.py a.pt b.pt \
        --conditions awgn,mpg,mpp,mpd,mps           # compare two checkpoints

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

Two things differ from the July version. **Decoding uses the same
fallback ladder a real receiver does** -- header, then `demodulate_blind`
-- because scoring only the header path reports "no picture" exactly
where a real station falls back to blind and still shows one, which
censors the low-SNR end of every table. And the SNR grid **descends
until acquisition drops below --acq-stop** rather than stopping at a
hardcoded floor, so each (mode, condition) is swept to its own threshold
instead of to a guess made for a different channel.
"""

import argparse
import io
import sys
import zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image  # noqa: E402

from sstvae import hfchannel  # noqa: E402
from sstvae.codec import MODEL_HELP, load_codec, pad_to_full, reconstruct  # noqa: E402
from sstvae.config import MODES, SNR_REF_BW_HZ  # noqa: E402
from sstvae.images import fit_image  # noqa: E402
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


def encode_all(codec, images) -> list[np.ndarray]:
    """One full-length latent vector per image; every mode is a prefix."""
    return [codec.encode(img) for img in images]


def decode(modem, y):
    """Header first, then blind -- the ladder a live receiver uses.

    Scoring only the header path makes a table that says "no picture"
    where a real station still gets one, which is exactly the low-SNR
    end the sweep exists to measure.
    """
    try:
        return modem.demodulate(y), "hdr"
    except SyncError:
        pass
    try:
        return modem.demodulate_blind(y), "blind"
    except Exception:
        return None, "fail"


def run_point(codec, modem, waveforms, images, snr_db, fading, seed_base):
    """Mean PSNR over the image set, and how each image was acquired."""
    scores = []
    n = {"hdr": 0, "blind": 0, "fail": 0}
    for i, (wave, src) in enumerate(zip(waveforms, images)):
        y = wave
        if snr_db is not None or fading is not None:
            y = hfchannel.apply_channel(
                wave, snr_db=snr_db, fading_preset=fading, seed=seed_base + i
            )
        r, how = decode(modem, y)
        n[how] += 1
        if r is None:
            continue
        out = reconstruct(codec, pad_to_full(r.latents), pad_to_full(r.weights))
        scores.append(psnr(src, out))
    acquired = n["hdr"] + n["blind"]
    return (float(np.mean(scores)) if scores else None), acquired, n


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
    ap.add_argument("models", nargs="*", default=[], help=MODEL_HELP)
    ap.add_argument("--model", default=None, help="single-model form (legacy)")
    ap.add_argument("--labels", default=None, help="comma-separated model names")
    ap.add_argument("--images", type=Path, default=DEFAULT_IMAGES)
    ap.add_argument("--n-images", type=int, default=DEFAULT_N_IMAGES)
    ap.add_argument("--modes", default="ABC")
    ap.add_argument(
        "--conditions",
        default="awgn,mpp",
        help="comma-separated: awgn plus any preset in hfchannel.FADING_PRESETS "
        "(mpg, mpp, mpd, mps)",
    )
    ap.add_argument("--snr-start", type=float, default=20.0)
    ap.add_argument("--snr-step", type=float, default=2.0, help="descent step, dB")
    ap.add_argument("--snr-floor", type=float, default=-14.0, help="hard stop")
    ap.add_argument(
        "--acq-stop",
        type=float,
        default=0.5,
        help="stop descending a (model, condition, mode) once the fraction of "
        "images that acquire at all -- header or blind -- falls below this",
    )
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--csv", type=Path, default=None)
    args = ap.parse_args()

    paths = list(args.models) or [args.model]
    labels = (args.labels.split(",") if args.labels
              else [Path(p).parent.name or Path(p).stem if p else "default"
                    for p in paths])
    if len(labels) != len(paths):
        ap.error(f"--labels has {len(labels)} names for {len(paths)} models")

    images = load_images(args.images, args.n_images)
    if not images:
        raise SystemExit(f"no images found in {args.images}")
    print(f"{len(images)} images, SNR referenced to {SNR_REF_BW_HZ:.0f} Hz",
          file=sys.stderr)

    codecs = [load_codec(p) for p in paths]
    modem = Modem()
    modes = [MODES[m] for m in args.modes]
    conds = [c.strip() for c in args.conditions.split(",") if c.strip()]

    # Encode once per (model, image): mode-independent, and the modes are
    # nested prefixes of the same vector. Modulate once per (model, mode)
    # and reuse across every condition and SNR -- the demodulator is what
    # costs, so nothing below this line should re-encode or re-modulate.
    all_latents = [encode_all(c, images) for c in codecs]
    waves_by_mode = {
        spec.name: [
            [modem.modulate(lat[: spec.n_latents], spec) for lat in lats]
            for lats in all_latents
        ]
        for spec in modes
    }

    # (label, condition, mode, snr) -> (psnr, acquired, counts)
    results = {}
    rows = []
    for cond in conds:
        fading = None if cond == "awgn" else cond
        for spec in modes:
            waves = waves_by_mode[spec.name]
            snr = args.snr_start
            while snr >= args.snr_floor:
                alive = False
                for mi, label in enumerate(labels):
                    mean_psnr, acquired, n = run_point(
                        codecs[mi], modem, waves[mi], images, snr, fading, args.seed
                    )
                    rate = acquired / len(images)
                    alive = alive or rate >= args.acq_stop
                    results[(label, cond, spec.name, snr)] = (mean_psnr, acquired, n)
                    rows.append({
                        "model": label, "channel": cond, "mode": spec.name,
                        "snr_db": snr,
                        "psnr_db": "" if mean_psnr is None else f"{mean_psnr:.2f}",
                        "acquired": acquired, "n_images": len(images),
                        "n_hdr": n["hdr"], "n_blind": n["blind"], "n_fail": n["fail"],
                    })
                    print(
                        f"  {cond:>4} mode {spec.name} {snr:>6.1f} dB  {label:<14} "
                        f"PSNR {fmt(mean_psnr):>6}  acq {acquired}/{len(images)} "
                        f"(hdr {n['hdr']} blind {n['blind']})",
                        file=sys.stderr, flush=True,
                    )
                if not alive:
                    break          # both models past the --acq-stop threshold
                snr -= args.snr_step

    print(f"\nPSNR dB, mean over images that produced a picture; "
          f"acq = header-or-blind acquisitions out of {len(images)}.")
    print("Descent stops once every model is below "
          f"{args.acq_stop:.0%} acquisition.\n")
    for cond in conds:
        print(f"\n### {cond}\n")
        head = " | ".join(f"{l} PSNR | {l} acq" for l in labels)
        delta = " | delta" if len(labels) == 2 else ""
        print(f"| Mode | SNR | {head}{delta} |")
        print("|---|---|" + "---|" * (2 * len(labels) + (1 if len(labels) == 2 else 0)))
        for spec in modes:
            snrs = sorted({k[3] for k in results if k[1] == cond and k[2] == spec.name},
                          reverse=True)
            for snr in snrs:
                cells, vals = [], []
                for label in labels:
                    v = results.get((label, cond, spec.name, snr))
                    if v is None:
                        cells += ["—", "—"]
                        vals.append(None)
                        continue
                    mp, acq, n = v
                    cells.append("—" if mp is None else f"{mp:.2f}")
                    cells.append(f"{acq}/{len(images)}")
                    vals.append(mp)
                d = ""
                if len(labels) == 2:
                    d = (f" | {vals[1] - vals[0]:+.2f}"
                         if vals[0] is not None and vals[1] is not None else " | —")
                print(f"| {spec.name} | {snr:.0f} | " + " | ".join(cells) + d + " |")

    if args.csv:
        import csv

        with open(args.csv, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nwrote {args.csv}", file=sys.stderr)


if __name__ == "__main__":
    main()

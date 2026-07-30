#!/usr/bin/env python3
"""Measure the photographic vs non-photographic codec gap at fp32.

    python scripts/gen_nonphoto.py --out /tmp/nonphoto --per-class 8
    python scripts/eval_nonphoto.py --nonphoto /tmp/nonphoto

Codec-only, no modem: encode -> (per-mode truncation) -> decode, so the
number isolates what the *model* does to content it never trained on
(docs/todo.md "Non-photographic content", item 1). The channel adds the
same latent-domain noise to every content class; the training-data gap
is in the clean reconstruction, and measuring it there keeps the run to
seconds instead of the full sweep's minutes. Per mode because the modes
truncate latent groups, and detail-heavy synthetic content may lean on
the later groups differently than photographs do.

Baseline is COCO val2017 (never in training), loaded the same way
snr_sweep.py does. Output is mean PSNR per content class per mode, and
the gap against the COCO baseline.
"""

import argparse
import io
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image  # noqa: E402

from sstvae.codec import load_codec, pad_to_full, reconstruct  # noqa: E402
from sstvae.config import MODES  # noqa: E402
from sstvae.images import fit_image  # noqa: E402

DEFAULT_COCO = Path("data/val2017.zip")


def load_coco(source: Path, n: int) -> list[tuple[str, Image.Image]]:
    with zipfile.ZipFile(source) as z:
        names = sorted(x for x in z.namelist() if x.lower().endswith((".jpg", ".png")))
        blobs = [z.read(name) for name in names[:n]]
    return [("coco", fit_image(Image.open(io.BytesIO(b)))) for b in blobs]


def load_nonphoto(source: Path) -> list[tuple[str, Image.Image]]:
    out = []
    for p in sorted(source.iterdir()):
        if p.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            continue
        cls = p.stem.rsplit("_", 1)[0]
        out.append((cls, fit_image(Image.open(p))))
    return out


def psnr(a: Image.Image, b: Image.Image) -> float:
    x, y = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    mse = np.mean((x - y) ** 2)
    return float("inf") if mse == 0 else 10 * np.log10(255.0**2 / mse)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--nonphoto", type=Path, required=True,
                    help="directory from gen_nonphoto.py")
    ap.add_argument("--coco", type=Path, default=DEFAULT_COCO)
    ap.add_argument("--n-coco", type=int, default=25)
    ap.add_argument("--model", default=None)
    ap.add_argument("--precision", default=None,
                    help="ONNX backend only (a .pt model is torch fp32 "
                    "already); the published-artifact runs in docs/todo.md "
                    "use --precision fp32")
    ap.add_argument("--csv", type=Path, default=None,
                    help="also dump one row per (image, mode)")
    args = ap.parse_args()

    codec = load_codec(args.model, precision=args.precision)
    images = load_coco(args.coco, args.n_coco) + load_nonphoto(args.nonphoto)

    rows = []  # (class, mode, image_index, psnr)
    scores: dict[tuple[str, str], list[float]] = defaultdict(list)
    counters: dict[str, int] = defaultdict(int)
    for cls, img in images:
        idx = counters[cls]
        counters[cls] += 1
        z_full = codec.encode(img)
        for mode in MODES.values():
            z = pad_to_full(z_full[: mode.n_latents])
            w = pad_to_full(np.ones(mode.n_latents))
            p = psnr(img, reconstruct(codec, z, w))
            scores[cls, mode.name].append(p)
            rows.append((cls, mode.name, idx, p))

    classes = ["coco"] + sorted(c for c in counters if c != "coco")
    modes = list(MODES)
    name_w = max(len(c) for c in classes) + 2
    print(f"codec-only PSNR (dB), model={args.model or 'published'} "
          f"precision={args.precision or 'default'}; "
          f"gap vs coco in parentheses")
    print(f"{'class':<{name_w}}  n  " + "".join(f"{'mode ' + m:>16}" for m in modes))
    for cls in classes:
        cells = []
        for m in modes:
            mean = float(np.mean(scores[cls, m]))
            if cls == "coco":
                cells.append(f"{mean:>16.2f}")
            else:
                gap = mean - float(np.mean(scores['coco', m]))
                cells.append(f"{mean:>8.2f} ({gap:+5.2f})")
        print(f"{cls:<{name_w}} {counters[cls]:>2}  " + "".join(cells))

    if args.csv:
        with open(args.csv, "w") as f:
            f.write("class,mode,image,psnr_db\n")
            for cls, m, idx, p in rows:
                f.write(f"{cls},{m},{idx},{p:.4f}\n")
        print(f"wrote {len(rows)} rows to {args.csv}")


if __name__ == "__main__":
    main()

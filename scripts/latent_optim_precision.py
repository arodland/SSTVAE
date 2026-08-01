#!/usr/bin/env python3
"""Do optimized latents survive a lower-precision decoder?

The sender optimizes against fp32 and **cannot know what the receiver
loaded** — fp16 is the app's default and int8 is published. Ordinary
encoder latents cost −0.006/−0.318 dB there (`docs/onnx.md`), but
latents deliberately tuned against one decoder carry no such guarantee:
the whole premise of the feature is finding inputs that particular
weights respond well to, which is exactly the kind of thing
quantisation could blunt.

So: optimize once against fp32, transmit, and decode the *same*
receptions with all three precisions. What matters is not the absolute
PSNR at each precision — that is already known — but whether the **gain
over the encoder's latents** survives.

    python scripts/latent_optim_precision.py wonder_wheel.jpg w0nycert.png \\
        --model sstvae-np1-epoch317.pt --onnx-dir out/onnx --opt-snr 5
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from latent_optim_prototype import optimize_latents  # noqa: E402

from sstvae import hfchannel  # noqa: E402
from sstvae.codec import load_codec, load_torch_model, pad_to_full  # noqa: E402
from sstvae.config import MODES  # noqa: E402
from sstvae.images import fit_image, image_to_array  # noqa: E402
from sstvae.modem import Modem  # noqa: E402

PRECISIONS = ("fp32", "fp16", "int8")


def psnr(mse: float) -> float:
    return -10 * math.log10(max(mse, 1e-12))


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("images", nargs="+")
    ap.add_argument("--model", default=None, help="checkpoint .pt to optimize against")
    ap.add_argument("--onnx-dir", required=True,
                    help="directory of {stem}-{encoder,decoder}-{prec}.onnx "
                         "exported from the SAME checkpoint")
    ap.add_argument("--mode", choices=sorted(MODES), default="B")
    ap.add_argument("--opt-snr", type=float, default=5.0)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--snr", type=float, nargs="+", default=[3.0, 9.0])
    ap.add_argument("--fading", nargs="+", default=["none", "mpp"])
    ap.add_argument("--seeds", type=int, default=25)
    args = ap.parse_args()

    torch.manual_seed(0)
    model = load_torch_model(args.model)
    spec = MODES[args.mode]
    modem = Modem()
    codecs = {p: load_codec(args.onnx_dir, precision=p) for p in PRECISIONS}
    cells = [(f, s) for f in args.fading for s in args.snr]

    for path in args.images:
        name = Path(path).stem
        target = image_to_array(fit_image(Image.open(path)))
        img_t = torch.from_numpy(target).unsqueeze(0)
        print(f"\n{'=' * 74}\n{name}  (mode {spec.name}, objective "
              f"{args.opt_snr:g} dB, {args.seeds} seeds)\n{'=' * 74}")

        z0, z_opt, _ = optimize_latents(
            model, img_t, args.mode, steps=args.steps,
            channel_snr_db=args.opt_snr, verbose=False)
        waves = {
            lbl: modem.modulate(
                model.latents_to_flat(z)[0].numpy()[: spec.n_latents], spec)
            for lbl, z in (("encoder", z0), ("optimized", z_opt))}

        # acc[precision][label][cell] -> list of psnr
        acc = {p: {l: {c: [] for c in cells} for l in waves} for p in PRECISIONS}
        for fading, snr in cells:
            fp = None if fading == "none" else fading
            for seed in range(args.seeds):
                for lbl, wave in waves.items():
                    rx = hfchannel.apply_channel(wave, snr_db=snr,
                                                 fading_preset=fp, seed=seed)
                    try:
                        r = modem.demodulate(rx)
                    except Exception:
                        continue
                    lat = pad_to_full(r.latents)
                    wgt = pad_to_full(r.weights)
                    # One reception, decoded three ways -- so the
                    # precision comparison is not also a channel
                    # comparison.
                    for p in PRECISIONS:
                        arr = np.asarray(codecs[p].decode(lat, wgt),
                                         dtype=np.float32).transpose(2, 0, 1) / 255
                        acc[p][lbl][(fading, snr)].append(
                            psnr(float(((arr - target) ** 2).mean())))

        head = "  ".join(f"{f}@{s:g}".rjust(9) for f, s in cells)
        print(f"\n  {'precision':<12}{head}     mean")
        for p in PRECISIONS:
            for lbl in ("encoder", "optimized"):
                vals = [np.mean(acc[p][lbl][c]) for c in cells]
                print(f"  {p + ' ' + lbl:<12}"
                      + "  ".join(f"{v:9.2f}" for v in vals)
                      + f"  {np.mean(vals):7.2f}")
            d = [np.mean(acc[p]["optimized"][c]) - np.mean(acc[p]["encoder"][c])
                 for c in cells]
            print(f"  {p + ' DELTA':<12}" + "  ".join(f"{v:+9.2f}" for v in d)
                  + f"  {np.mean(d):+7.2f}\n")


if __name__ == "__main__":
    main()

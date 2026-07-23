#!/usr/bin/env python3
"""Decode received SSTVAE audio into an image.

    python sstvae_decode.py rx.wav out.png --model runs/s1/checkpoint.pt
    python sstvae_decode.py rx.wav out.png --model ... --snapshots 4

--snapshots N additionally writes out_1.png..out_N.png reconstructed
from successive prefixes of the transmission (progressive decode).
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from sstvae import wavio
from sstvae.config import FS, LATENT_CHANNELS, LATENTS_PER_FRAME, MODES
from sstvae.models import SSTVAE
from sstvae.modem import Modem, framing
from sstvae_encode import load_model


def reconstruct(model: SSTVAE, latents: np.ndarray, weights: np.ndarray) -> Image.Image:
    """Full-length (mode C sized) latent/weight vectors -> PIL image."""
    z = SSTVAE.flat_to_latents(torch.from_numpy(latents).float()[None])
    w = SSTVAE.flat_to_latents(torch.from_numpy(weights).float()[None])
    with torch.no_grad():
        img = model.decoder(z * (w > 0), w)[0]
    arr = (img.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
    return Image.fromarray(arr)


def pad_to_full(vec: np.ndarray, fill: float = 0.0) -> np.ndarray:
    full = MODES["C"].n_latents
    out = np.full(full, fill)
    out[: len(vec)] = vec
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input", help="received WAV")
    ap.add_argument("output", help="output image (png/jpg)")
    ap.add_argument("--model", required=True, help="checkpoint.pt")
    ap.add_argument("--snapshots", type=int, default=0)
    ap.add_argument(
        "--search-start", type=float, default=None,
        help="limit preamble search to after this time (s)",
    )
    ap.add_argument(
        "--search-end", type=float, default=None,
        help="limit preamble search to before this time (s)",
    )
    args = ap.parse_args()

    x = wavio.read_wav(args.input)
    search = None
    if args.search_start is not None or args.search_end is not None:
        search = (args.search_start or 0.0, args.search_end or len(x) / FS)
    model = load_model(args.model)
    r = Modem().demodulate(x, search_s=search)
    print(
        f"mode {r.mode.name}, {r.frames_received}/{r.mode.n_frames} frames, "
        f"freq offset {r.freq_offset:+.1f} Hz, sync metric {r.sync_metric:.2f}"
    )

    latents = pad_to_full(r.latents)
    weights = pad_to_full(r.weights)
    reconstruct(model, latents, weights).save(args.output)
    print(f"wrote {args.output}")

    if args.snapshots > 0:
        # Frame index of each canonical latent: which point in the
        # transmission it arrived at.
        slot_frame = np.arange(r.mode.n_latents) // LATENTS_PER_FRAME
        frame_of_latent = framing.deinterleave(slot_frame.astype(float), r.mode)
        out = Path(args.output)
        for k in range(1, args.snapshots + 1):
            cutoff = r.mode.n_frames * k / args.snapshots
            w_k = r.weights * (frame_of_latent < cutoff)
            img = reconstruct(model, latents, pad_to_full(w_k))
            path = out.with_stem(f"{out.stem}_{k:03d}")
            img.save(path)
            print(f"wrote {path} (first {cutoff:.0f} frames)")


if __name__ == "__main__":
    main()

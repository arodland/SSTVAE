#!/usr/bin/env python3
"""Encode an image into SSTVAE transmit audio.

    python sstvae_encode.py photo.jpg tx.wav --mode B
"""

import argparse

import numpy as np
import torch

from sstvae import wavio
from sstvae.codec import MODEL_HELP, load_model  # noqa: F401  (re-exported)
from sstvae.config import MODES
from sstvae.images import load_image
from sstvae.models import SSTVAE
from sstvae.modem import Modem


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("image")
    ap.add_argument("output", help="WAV file for transmission")
    ap.add_argument("--mode", choices=sorted(MODES), default="B")
    ap.add_argument("--model", default=None, help=MODEL_HELP)
    ap.add_argument(
        "--callsign", default="",
        help="up to 8 chars, sent continuously on the beacon carrier "
        "alongside the resync frame counter (see sstvae/modem/beacon.py)",
    )
    args = ap.parse_args()

    spec = MODES[args.mode]
    model = load_model(args.model)
    img = load_image(args.image)[None]
    with torch.no_grad():
        z = model.encoder(img)
    flat = SSTVAE.latents_to_flat(z)[0].numpy().astype(np.float64)

    x = Modem().modulate(flat[: spec.n_latents], spec, callsign=args.callsign)
    wavio.write_wav(args.output, x)
    print(
        f"wrote {args.output}: mode {spec.name}, {spec.n_latents} latents, "
        f"{spec.duration_s:.1f} s"
    )


if __name__ == "__main__":
    main()

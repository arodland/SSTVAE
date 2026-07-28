#!/usr/bin/env python3
"""Encode an image into SSTVAE transmit audio.

    python sstvae_encode.py photo.jpg tx.wav --mode B
"""

import argparse

from sstvae import wavio
from sstvae.checkpoint import PRECISIONS
from sstvae.codec import (  # noqa: F401  (MODEL_HELP re-exported)
    MODEL_HELP,
    PRECISION_HELP,
    load_codec,
)
from sstvae.config import MODES
from sstvae.images import load_image
from sstvae.modem import Modem


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("image")
    ap.add_argument("output", help="WAV file for transmission")
    ap.add_argument("--mode", choices=sorted(MODES), default="B")
    ap.add_argument("--model", default=None, help=MODEL_HELP)
    ap.add_argument("--precision", choices=PRECISIONS, default=None,
                    help=PRECISION_HELP)
    ap.add_argument(
        "--callsign", default="",
        help="up to 8 chars, sent continuously on the beacon carrier "
        "alongside the resync frame counter (see sstvae/modem/beacon.py)",
    )
    args = ap.parse_args()

    spec = MODES[args.mode]
    codec = load_codec(args.model, precision=args.precision)
    flat = codec.encode(load_image(args.image))

    x = Modem().modulate(flat[: spec.n_latents], spec, callsign=args.callsign)
    wavio.write_wav(args.output, x)
    print(
        f"wrote {args.output}: mode {spec.name}, {spec.n_latents} latents, "
        f"{spec.duration_s:.1f} s"
    )


if __name__ == "__main__":
    main()

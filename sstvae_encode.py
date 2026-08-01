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
    ap.add_argument(
        "--optimize", nargs="?", type=float, const=20.0, default=None,
        metavar="SECONDS",
        help="spend up to SECONDS (default 20) improving the latents for "
        "this particular picture before transmitting -- costs nothing on "
        "air and needs no change at the receiver "
        "(docs/latent-optimization.md). Longer is better up to a plateau "
        "at ~90 s on a fast desktop; the default 20 s captures about two "
        "thirds of the available gain, and it stops early once the loss "
        "flattens. Fetches an extra 18 MB artifact the first time.",
    )
    args = ap.parse_args()

    spec = MODES[args.mode]
    codec = load_codec(args.model, precision=args.precision)
    image = load_image(args.image)
    flat = codec.encode(image)

    if args.optimize is not None:
        from sstvae.latent_optim import optimize

        print(f"optimizing latents (up to {args.optimize:g} s)...", flush=True)
        r = optimize(flat, image, args.mode, model=args.model,
                     time_budget_s=args.optimize)
        flat = r.latents
        print(
            f"  {r.steps} steps in {r.seconds:.1f} s ({r.stop_reason}); "
            f"objective improved {r.gain_db:.2f} dB"
        )

    x = Modem().modulate(flat[: spec.n_latents], spec, callsign=args.callsign)
    wavio.write_wav(args.output, x)
    print(
        f"wrote {args.output}: mode {spec.name}, {spec.n_latents} latents, "
        f"{spec.duration_s:.1f} s"
    )


if __name__ == "__main__":
    main()

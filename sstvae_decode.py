#!/usr/bin/env python3
"""Decode received SSTVAE audio into an image.

    python sstvae_decode.py rx.wav out.png
    python sstvae_decode.py rx.wav out.png --model ... --snapshots 4

--snapshots N additionally writes out_1.png..out_N.png reconstructed
from successive prefixes of the transmission (progressive decode).
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from sstvae import wavio
from sstvae.checkpoint import PRECISIONS
from sstvae.codec import (  # noqa: F401  (re-exported)
    MODEL_HELP,
    PRECISION_HELP,
    load_codec,
    pad_to_full,
    reconstruct,
)
from sstvae.config import DRIFT_TRACK_MODES, FS, LATENTS_PER_FRAME
from sstvae.modem import Modem, framing


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input", help="received WAV")
    ap.add_argument("output", help="output image (png/jpg)")
    ap.add_argument("--model", default=None, help=MODEL_HELP)
    ap.add_argument("--precision", choices=PRECISIONS, default=None,
                    help=PRECISION_HELP)
    ap.add_argument("--snapshots", type=int, default=0)
    ap.add_argument(
        "--search-start", type=float, default=None,
        help="limit preamble search to after this time (s)",
    )
    ap.add_argument(
        "--search-end", type=float, default=None,
        help="limit preamble search to before this time (s)",
    )
    ap.add_argument(
        "--size", type=str, default=None,
        help="resize output image, e.g. 320x240 (classic SSTV size)",
    )
    ap.add_argument(
        "--drift-track", choices=DRIFT_TRACK_MODES, default="off",
        help="follow a carrier that drifts during the transmission: 'slow' for a "
        "drifting rig, 'fast' for rapid wander (VHF/satellite). Off by default -- "
        "the untracked receiver absorbs about +-2 Hz of residual on its own.",
    )
    args = ap.parse_args()
    out_size = None
    if args.size:
        w, h = args.size.lower().split("x")
        out_size = (int(w), int(h))

    x = wavio.read_wav(args.input)
    search = None
    if args.search_start is not None or args.search_end is not None:
        search = (args.search_start or 0.0, args.search_end or len(x) / FS)
    model = load_codec(args.model, precision=args.precision)
    r = Modem().demodulate(x, search_s=search, drift_track=args.drift_track)
    print(
        f"mode {r.mode.name}, {r.frames_received}/{r.mode.n_frames} frames, "
        f"freq offset {r.freq_offset:+.1f} Hz, sync metric {r.sync_metric:.2f}, "
        f"SNR {r.snr_db:.1f} dB"
    )
    if r.beacon is not None:
        cs = f"'{r.callsign}'" if r.callsign else "(none sent)"
        print(f"beacon: frame {r.beacon.frame_index}, callsign {cs}")
    else:
        print("beacon: no superframe decoded (short/noisy reception)")

    latents = pad_to_full(r.latents)
    weights = pad_to_full(r.weights)

    def save(img, path):
        if out_size is not None:
            img = img.resize(out_size, Image.LANCZOS)
        img.save(path)

    save(reconstruct(model, latents, weights), args.output)
    print(f"wrote {args.output}")

    if args.snapshots > 0:
        # Frame index of each canonical latent: which point in the
        # transmission it arrived at. Latents that never got an on-air
        # slot (see config.DROPPED_LATENTS_PER_GROUP) come back as 0
        # here, but they're always weight-0 anyway so the bogus frame
        # index never actually gates anything in.
        slot_frame = np.arange(r.mode.n_tx_latents) // LATENTS_PER_FRAME
        frame_of_latent, _ = framing.deinterleave(slot_frame.astype(float), r.mode)
        out = Path(args.output)
        for k in range(1, args.snapshots + 1):
            cutoff = r.mode.n_frames * k / args.snapshots
            w_k = r.weights * (frame_of_latent < cutoff)
            img = reconstruct(model, latents, pad_to_full(w_k))
            path = out.with_stem(f"{out.stem}_{k:03d}")
            save(img, path)
            print(f"wrote {path} (first {cutoff:.0f} frames)")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""How much picture quality costs you for tuning in late.

    python scripts/late_join_sweep.py

Distinct from a *late lock* (`sync.acquire_blind`), where the receiver
recorded the whole transmission but only worked out where it was partway
through — that loses nothing. This measures the other case: the frames
before you started listening never existed, so they decode as erasures.

The channel is clean, to isolate the cost of the missing frames from the
cost of noise. Frames are dropped by zeroing their latents' weights,
which is exactly the state a late joiner's decoder is in.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sstvae.codec import MODEL_HELP, load_model, pad_to_full, reconstruct  # noqa: E402
from sstvae.config import (  # noqa: E402
    FRAME_SAMPLES,
    FRAMES_PER_GROUP,
    FS,
    LATENTS_PER_FRAME,
    MODES,
)
from sstvae.data import image_to_tensor  # noqa: E402
from sstvae.models import SSTVAE  # noqa: E402
from sstvae.modem import Modem, framing  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from snr_sweep import DEFAULT_IMAGES, load_images, psnr  # noqa: E402

JOIN_SECONDS = [0, 16, 32, 48, 64, 80]


def frame_of_each_latent(spec) -> np.ndarray:
    """Which frame carried each canonical latent.

    The interleaver scatters a frame's latents across the whole picture,
    so this is not a contiguous region -- which is the point: a late join
    costs detail everywhere rather than a missing band.
    """
    slot = np.arange(spec.n_tx_latents) // LATENTS_PER_FRAME
    frame, _ = framing.deinterleave(slot.astype(float), spec)
    return pad_to_full(frame)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=None, help=MODEL_HELP)
    ap.add_argument("--images", type=Path, default=DEFAULT_IMAGES)
    ap.add_argument("--n-images", type=int, default=12)
    ap.add_argument("--mode", default="C")
    args = ap.parse_args()

    images = load_images(args.images, args.n_images)
    model = load_model(args.model)
    modem = Modem()
    spec = MODES[args.mode]
    frame_of_latent = frame_of_each_latent(spec)

    group_s = FRAMES_PER_GROUP * FRAME_SAMPLES / FS
    print(
        f"mode {spec.name}, clean channel, {len(images)} images; "
        f"one group = {group_s:.1f} s",
        file=sys.stderr,
    )

    scores = {j: [] for j in JOIN_SECONDS}
    for img in images:
        with torch.no_grad():
            z = model.encoder(image_to_tensor(img)[None])
        flat = SSTVAE.latents_to_flat(z)[0].numpy().astype(np.float64)
        r = modem.demodulate(modem.modulate(flat[: spec.n_latents], spec))
        latents, weights = pad_to_full(r.latents), pad_to_full(r.weights)
        for join_s in JOIN_SECONDS:
            cutoff = join_s * FS / FRAME_SAMPLES
            masked = weights * (frame_of_latent >= cutoff)
            scores[join_s].append(psnr(img, reconstruct(model, latents, masked)))

    print(f"\n| joined at | {' | '.join(f'{j} s' for j in JOIN_SECONDS)} |")
    print("|---|" + "---|" * len(JOIN_SECONDS))
    cells = " | ".join(f"{np.mean(scores[j]):.1f}" for j in JOIN_SECONDS)
    print(f"| PSNR (dB) | {cells} |")
    print(
        f"\nGroup boundaries at {group_s:.0f} s and {2 * group_s:.0f} s.",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()

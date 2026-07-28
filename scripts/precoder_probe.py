#!/usr/bin/env python3
"""Measure what a slot-domain precoder does to PAPR, TX side only.

Settles the open question in docs/slot-domain-precoder.md before any of
the RX/erasure-accounting work is done: SC-FDMA's 3-4 dB crest-factor
win comes from constant-modulus (QPSK/QAM) inputs, and our latents are
Gaussian-ish analog, so the gain here may be small or nil.

Builds the transmit waveform exactly as Modem.modulate does, but
optionally spreads each OFDM symbol's 23 complex latent carriers with a
unitary transform first, and reports envelope PAPR before and after the
clipper. Nothing about the receiver changes, so this measures the
*achievable* PAPR, not end-to-end quality.

    python scripts/precoder_probe.py --model checkpoint.pt --image img.jpg
    python scripts/precoder_probe.py --synthetic      # Gaussian latents
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sstvae.config import (
    BEACON_CARRIER,
    CHIPS_PER_FRAME,
    CLIP_HEADROOM_DB,
    DATA_SYMS_PER_FRAME,
    LATENTS_PER_FRAME,
    LEADIN_SAMPLES,
    LEADOUT_SAMPLES,
    MODES,
    NC,
    NC_LATENT,
    SYMS_PER_FRAME,
)
from sstvae.modem import beacon, dsp, framing, ofdm
from sstvae.modem.modem import Modem


def dft_matrix(n: int) -> np.ndarray:
    """Unitary n-point DFT — the SC-FDMA / DFT-s-OFDM spreader."""
    k = np.arange(n)
    return np.exp(-2j * np.pi * np.outer(k, k) / n) / np.sqrt(n)


def hadamard_matrix(n: int) -> np.ndarray:
    """Unitary Walsh-Hadamard spreader, zero-padded to n.

    Real-valued and constant-modulus in a different sense than the DFT;
    worth a look because it mixes I and Q differently.
    """
    m = 1 << (n - 1).bit_length()
    h = np.ones((1, 1))
    while h.shape[0] < m:
        h = np.block([[h, h], [h, -h]])
    return h[:n, :n] / np.sqrt(n)


PRECODERS = {
    "none": lambda n: np.eye(n),
    "dft": dft_matrix,
    "hadamard": hadamard_matrix,
}


def build_waveform(latents: np.ndarray, spec, precoder: np.ndarray) -> np.ndarray:
    """Modem.modulate, with each symbol's latent carriers precoded."""
    m = Modem()
    rms = np.sqrt(np.mean(latents**2))
    if rms > 0:
        latents = latents / rms
    slots = framing.interleave(latents, spec)
    n_f = spec.n_frames
    chips = beacon.chip_stream(0, n_f, "PROBE")
    symbols = np.empty((n_f * SYMS_PER_FRAME, NC), dtype=np.complex128)
    for f in range(n_f):
        sl = slots[f * LATENTS_PER_FRAME : (f + 1) * LATENTS_PER_FRAME]
        symbols[f * SYMS_PER_FRAME] = m.pilot
        data = np.empty((DATA_SYMS_PER_FRAME, NC), dtype=np.complex128)
        # (DATA_SYMS_PER_FRAME, NC_LATENT): one row = one OFDM symbol's
        # 23 complex carriers = the 46 real slots that share a peak event.
        car = framing.slots_to_symbols(sl)
        data[:, :NC_LATENT] = car @ precoder.T
        data[:, BEACON_CARRIER] = chips[f * CHIPS_PER_FRAME : (f + 1) * CHIPS_PER_FRAME]
        symbols[f * SYMS_PER_FRAME + 1 : (f + 1) * SYMS_PER_FRAME] = data
    hdr = framing.header_symbol(spec)
    return np.concatenate(
        [
            np.zeros(LEADIN_SAMPLES),
            ofdm.preamble_waveform(),
            ofdm.modulate_symbols(np.stack([hdr, hdr])),
            ofdm.modulate_symbols(symbols),
            np.zeros(LEADOUT_SAMPLES),
        ]
    )


def latents_from_model(model_path: str, image: str | None, spec) -> np.ndarray:
    import torch
    from PIL import Image

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from sstvae.codec import load_torch_model as load_model
    from sstvae.models import SSTVAE

    model = load_model(model_path)
    if image:
        img = Image.open(image).convert("RGB").resize((640, 480))
        x = torch.from_numpy(np.asarray(img)).permute(2, 0, 1)[None].float() / 255
    else:
        x = torch.rand(1, 3, 480, 640)
    with torch.no_grad():
        z = model.encoder(x)
    flat = SSTVAE.latents_to_flat(z)[0].numpy().astype(np.float64)
    return flat[: spec.n_latents]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", help="checkpoint for real latents")
    ap.add_argument("--image", help="image to encode (default: random noise image)")
    ap.add_argument(
        "--synthetic",
        action="store_true",
        help="unit-variance Gaussian latents instead of a model's",
    )
    ap.add_argument(
        "--constant-modulus",
        action="store_true",
        help="+-1 slots, i.e. QPSK carriers. Not a real waveform for us "
        "-- it is the control: this is the input SC-FDMA is designed "
        "for, so DFT spreading must show its known 3-4 dB win here. If "
        "it does not, the probe is wrong, not the theory",
    )
    ap.add_argument("--mode", default="A", choices=list(MODES))
    ap.add_argument("--clip-headroom-db", type=float, default=CLIP_HEADROOM_DB)
    args = ap.parse_args()

    spec = MODES[args.mode]
    rng = np.random.default_rng(0)
    if args.constant_modulus:
        latents = rng.choice([-1.0, 1.0], size=spec.n_latents)
        src = "QPSK control (constant modulus)"
    elif args.synthetic or not args.model:
        latents = rng.standard_normal(spec.n_latents)
        src = "synthetic Gaussian"
    else:
        latents = latents_from_model(args.model, args.image, spec)
        src = f"{args.model}"
    kurt = np.mean(latents**4) / np.mean(latents**2) ** 2
    print(f"mode {args.mode}, {spec.n_latents} latents from {src}")
    print(f"latent kurtosis {kurt:.3f} (3.0 = Gaussian)\n")

    # Post-clip PAPR is pinned by the clipper at any headroom, so it is
    # not the figure of merit -- clipping *distortion* is. Measured as
    # waveform NMSE through tx_condition, which maps to latent SNR
    # because OFDM demod is linear and orthogonal.
    print(
        f"{'precoder':10s} {'PAPR pre-clip':>14s} {'PAPR post-clip':>15s}"
        f" {'clip SNR':>10s}"
    )
    base = None
    for name, make in PRECODERS.items():
        x = build_waveform(latents, spec, make(NC_LATENT))
        y = dsp.tx_condition(x, args.clip_headroom_db)
        xn = x / np.sqrt(np.mean(x**2))
        yn = y / np.sqrt(np.mean(y**2))
        clip_snr = 10 * np.log10(np.mean(xn**2) / np.mean((yn - xn) ** 2))
        pre, post = dsp.papr_db(x), dsp.papr_db(y)
        if base is None:
            base = (pre, post, clip_snr)
            delta = ""
        else:
            delta = (
                f"   ({pre - base[0]:+.2f} / {post - base[1]:+.2f}"
                f" / {clip_snr - base[2]:+.2f} dB)"
            )
        print(f"{name:10s} {pre:13.2f} {post:14.2f} {clip_snr:9.2f}{delta}")


if __name__ == "__main__":
    main()

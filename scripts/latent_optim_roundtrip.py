#!/usr/bin/env python3
"""End-to-end check on transmit-time latent optimization.

`scripts/latent_optim_prototype.py` measures in the latent domain,
through the *differentiable* channel that training uses. That is the
objective being optimized, so it cannot be the evidence that
optimization works -- optimizing MSE at an assumed SNR and then
reporting MSE at that same SNR is not a measurement.

This runs the real thing instead: encode -> modulate -> HF channel ->
demodulate -> decode, comparing the picture that comes out against the
picture that went in, for encoder latents and optimized latents through
identical channels (same seed, so the two see the *same* noise and
fading realization, not merely the same distribution).

It also reports PAPR of both waveforms, which is free here and is the
measurement `docs/latent-optimization.md` wants: latents ride OFDM
carriers, so a changed latent distribution could in principle change
the envelope statistics. Expected to be small -- the clipper absorbs it
and stage-2 trained through that same clipper -- but expected and
measured are different words.

    python scripts/latent_optim_roundtrip.py wonder_wheel.jpg w0nycert.png \\
        --model sstvae-np1-epoch317.pt --mode B --snr 6 12 --fading none mpp
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
from sstvae.codec import load_torch_model, pad_to_full  # noqa: E402
from sstvae.config import MODES  # noqa: E402
from sstvae.images import fit_image, image_to_array  # noqa: E402
from sstvae.modem import Modem, dsp  # noqa: E402


def psnr(mse: float) -> float:
    return -10 * math.log10(max(mse, 1e-12))


def decode_to_array(model, latents, weights) -> np.ndarray:
    """Received latents -> (3,H,W) float array, via the torch decoder.

    The torch decoder rather than `codec.reconstruct` on purpose: this
    script already holds the checkpoint, and going through PIL uint8
    would quantize the thing being measured.
    """
    z = torch.from_numpy(pad_to_full(latents).astype(np.float32))
    w = torch.from_numpy(pad_to_full(weights).astype(np.float32))
    z = model.flat_to_latents(z[None])
    w = model.flat_to_latents(w[None])
    with torch.no_grad():
        return model.decoder(z * w, w)[0].numpy()


def run_cell(model, wave: np.ndarray, target: np.ndarray, snr, fading, seed):
    """One transmission through one channel realization."""
    rx = hfchannel.apply_channel(wave, snr_db=snr, fading_preset=fading,
                                 seed=seed)
    try:
        r = Modem().demodulate(rx)
    except Exception as e:  # a failed acquisition is a result, not a crash
        return {"ok": False, "why": type(e).__name__ + f": {e}"[:60]}
    img = decode_to_array(model, r.latents, r.weights)
    mse = float(((img - target) ** 2).mean())
    return {"ok": True, "psnr": psnr(mse), "frames": r.frames_received,
            "n_frames": r.mode.n_frames, "snr_meas": r.snr_db,
            "mode": r.mode.name, "img": img}


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("images", nargs="+")
    ap.add_argument("--model", default=None)
    ap.add_argument("--mode", choices=sorted(MODES), default="B")
    ap.add_argument("--snr", type=float, nargs="+", default=[3.0, 9.0])
    ap.add_argument("--fading", nargs="+", default=["none", "mpp"],
                    choices=["none", "mpg", "mpp", "mpd"])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--opt-snr", nargs="+", default=["none"],
                    metavar="DB|none",
                    help="objective SNR(s): optimize through the "
                         "differentiable channel at this SNR. 'none' is "
                         "the clean-decoder objective. More than one "
                         "value sweeps them and prints a matrix.")
    ap.add_argument("--reg-weight", type=float, nargs="+", default=[0.0],
                    help="L2 pull toward the encoder's latents. More than "
                         "one value sweeps them (crossed with --opt-snr).")
    ap.add_argument("--callsign", default="")
    ap.add_argument("--save-dir", default=None,
                    help="write the A/B transmit WAVs and decoded pictures here")
    args = ap.parse_args()

    torch.manual_seed(0)
    model = load_torch_model(args.model)
    spec = MODES[args.mode]
    modem = Modem()
    save = Path(args.save_dir) if args.save_dir else None
    if save:
        save.mkdir(parents=True, exist_ok=True)

    opt_snrs = [None if s.lower() == "none" else float(s) for s in args.opt_snr]
    cells = [(f, s) for f in args.fading for s in args.snr]
    configs = [(o, r) for o in opt_snrs for r in args.reg_weight]

    def label_of(o, r=None):
        s = "clean" if o is None else f"{o:g} dB"
        if r is not None and len(args.reg_weight) > 1:
            s += f" r={r:g}"
        return s

    def evaluate(wave, target):
        """One waveform over the whole channel grid -> {cell: mean psnr}."""
        out, locks = {}, {}
        for fading, snr in cells:
            fp = None if fading == "none" else fading
            vals, fails = [], 0
            for seed in range(args.seeds):
                r = run_cell(model, wave, target, snr, fp, seed)
                if r["ok"]:
                    vals.append(r["psnr"])
                else:
                    fails += 1
            out[(fading, snr)] = float(np.mean(vals)) if vals else float("nan")
            locks[(fading, snr)] = fails
        return out, locks

    for path in args.images:
        name = Path(path).stem
        target = image_to_array(fit_image(Image.open(path)))
        img_t = torch.from_numpy(target).unsqueeze(0)
        print(f"\n{'=' * 78}\n{name}  (mode {spec.name}, {args.seeds} seeds, "
              f"{args.steps} steps)\n{'=' * 78}")

        # The encoder's latents are the same whatever the objective is,
        # so its baseline is measured once and every row compares to it.
        with torch.no_grad():
            z_enc = optimize_latents(model, img_t, args.mode, steps=0,
                                     verbose=False)[0]
        base_wave = modem.modulate(
            model.latents_to_flat(z_enc)[0].numpy()[: spec.n_latents], spec,
            callsign=args.callsign)
        base, base_locks = evaluate(base_wave, target)
        base_papr = dsp.papr_db(base_wave)

        print(f"\n  encoder baseline (PAPR {base_papr:.2f} dB)")
        print("   " + "  ".join(f"{f}@{s:g}".rjust(9) for f, s in cells))
        print("   " + "  ".join(f"{base[c]:9.2f}" for c in cells))
        print("   " + "  ".join(
            f"{('(' + str(base_locks[c]) + ' nolock)') if base_locks[c] else '':>9}"
            for c in cells))

        width = max(10, max(len(label_of(o, r)) for o, r in configs) + 2)
        print(f"\n  delta vs encoder, by objective:")
        print("   " + "objective".ljust(width)
              + "  ".join(f"{f}@{s:g}".rjust(9) for f, s in cells)
              + "     mean   PAPR")
        for o, reg in configs:
            _, z_opt, _ = optimize_latents(
                model, img_t, args.mode, steps=args.steps, lr=args.lr,
                channel_snr_db=o, reg_weight=reg, verbose=False)
            wave = modem.modulate(
                model.latents_to_flat(z_opt)[0].numpy()[: spec.n_latents],
                spec, callsign=args.callsign)
            got, _ = evaluate(wave, target)
            deltas = [got[c] - base[c] for c in cells]
            print("   " + label_of(o, reg).ljust(width)
                  + "  ".join(f"{d:+9.2f}" for d in deltas)
                  + f"  {np.nanmean(deltas):+7.2f}"
                  + f"  {dsp.papr_db(wave) - base_papr:+6.2f}")
            if save:
                from sstvae import wavio
                wavio.write_wav(
                    str(save / f"{name}-opt{label_of(o, reg).replace(' ', '')}.wav"),
                    wave)

    if save:
        print(f"\nwrote WAVs and decoded pictures to {save}")


if __name__ == "__main__":
    main()

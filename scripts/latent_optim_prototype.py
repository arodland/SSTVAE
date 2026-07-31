#!/usr/bin/env python3
"""Prototype: per-image latent optimization against a frozen decoder.

The encoder is an amortized approximation — trained to minimize average
loss over the training set, not necessarily the loss for any single
image. This script checks how big that gap actually is: it encodes one
image normally, then treats the latents as a free parameter and runs
gradient descent through the (frozen) decoder to directly minimize
reconstruction loss on this image, optionally under the same
differentiable channel model used in training (`latent_channel.py`) so
the result stays meaningful for what actually gets transmitted rather
than just the noiseless decode.

Usage:
    python scripts/latent_optim_prototype.py IMAGE.jpg --out sheet.png
    python scripts/latent_optim_prototype.py IMAGE.jpg --model ckpt.pt \\
        --steps 500 --channel-snr-db 8 --out sheet.png
"""

import argparse
import math

import numpy as np
import torch
from PIL import Image, ImageDraw

from sstvae import wavio
from sstvae.codec import load_torch_model
from sstvae.config import MODES, CHANNELS_PER_GROUP
from sstvae.images import fit_image, image_to_array, font
from sstvae.latent_channel import ChannelConfig, apply_latent_channel
from sstvae.modem import Modem


def psnr(mse: float) -> float:
    return -10 * math.log10(max(mse, 1e-12))


def mask_weights(like: torch.Tensor, active_channels: int) -> torch.Tensor:
    """(B,C,H,W)-shaped all-ones/zeros weight mask for the first
    `active_channels` channels -- what mode A/B leaves untransmitted is
    unweighted, exactly as the decoder sees a truncated mode."""
    w = torch.zeros_like(like)
    w[:, :active_channels] = 1.0
    return w


def project_unit_rms(z: torch.Tensor, active_channels: int) -> torch.Tensor:
    """Renormalize the active channels to unit RMS (channels beyond
    `active_channels` are assumed already zero) -- matches
    `Modem.modulate`'s normalization, which is over the transmitted
    (already-truncated) latent vector only, not the full 132-channel
    tensor."""
    active = z[:, :active_channels]
    rms = active.flatten(1).pow(2).mean(dim=1, keepdim=True).sqrt().clamp_min(1e-6)
    return z / rms[:, :, None, None]


def mc_channel_mse(model, z, img, snr_db, weights_mode, n=8):
    """Mean reconstruction MSE over `n` channel draws at a fixed SNR."""
    cfg = ChannelConfig(snr_db_range=(snr_db, snr_db), erasure_rate_max=0.0,
                         p_truncate=0.0)
    total = 0.0
    with torch.no_grad():
        for _ in range(n):
            noisy, w, _ = apply_latent_channel(z, cfg)
            noisy, w = noisy * weights_mode, w * weights_mode
            recon = model.decoder(noisy, w)
            total += torch.nn.functional.mse_loss(recon, img).item()
    return total / n


def make_sheet(orig: np.ndarray, enc_recon: np.ndarray, opt_recon: np.ndarray,
               labels: list[str]) -> Image.Image:
    def to_pil(arr):
        return Image.fromarray(
            (arr.transpose(1, 2, 0) * 255).clip(0, 255).astype("uint8"))

    panels = [to_pil(orig), to_pil(enc_recon), to_pil(opt_recon)]
    w, h = panels[0].size
    pad, label_h = 8, 28
    sheet = Image.new("RGB", (w * 3 + pad * 4, h + label_h + pad * 2), "white")
    draw = ImageDraw.Draw(sheet)
    f = font(18)
    for i, (panel, label) in enumerate(zip(panels, labels)):
        x = pad + i * (w + pad)
        sheet.paste(panel, (x, label_h + pad))
        draw.text((x, 4), label, fill="black", font=f)
    return sheet


def optimize_latents(model, img, mode: str, *, steps: int = 300,
                     lr: float = 0.02, channel_snr_db: float | None = None,
                     channel_samples: int = 4, reg_weight: float = 0.0,
                     verbose: bool = True):
    """Encoder latents -> better latents for *this* image.

    Returns `(z0, z_opt, weights_mode)`, all mode-masked and projected
    onto the unit-RMS shell, so a caller can modulate either one
    without further conditioning. Shared with
    `scripts/latent_optim_roundtrip.py` -- two copies of this loop would
    let the measurement and the thing being measured drift apart.
    """
    spec = MODES[mode]
    active = spec.groups * CHANNELS_PER_GROUP

    with torch.no_grad():
        z0 = model.encoder(img)
        weights_mode = mask_weights(z0, active)
        z0 = project_unit_rms(z0 * weights_mode, active)

    z_opt = z0.clone().detach().requires_grad_(True)
    opt = torch.optim.Adam([z_opt], lr=lr)
    cfg = None
    if channel_snr_db is not None:
        cfg = ChannelConfig(snr_db_range=(channel_snr_db,) * 2,
                            erasure_rate_max=0.0, p_truncate=0.0)

    for step in range(steps):
        opt.zero_grad()
        if cfg is not None:
            loss = 0.0
            for _ in range(channel_samples):
                noisy, w, _ = apply_latent_channel(z_opt, cfg)
                loss = loss + torch.nn.functional.mse_loss(
                    model.decoder(noisy * weights_mode, w * weights_mode), img)
            loss = loss / channel_samples
        else:
            loss = torch.nn.functional.mse_loss(
                model.decoder(z_opt, weights_mode), img)
        (loss + reg_weight * (z_opt - z0).pow(2).mean()).backward()
        opt.step()
        with torch.no_grad():
            z_opt.copy_(project_unit_rms(z_opt, active))
        if verbose and (step + 1) % max(1, steps // 10) == 0:
            print(f"  step {step + 1:4d}/{steps}  recon_loss={loss.item():.6f}")

    return z0, z_opt.detach(), weights_mode


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image")
    ap.add_argument("--model", default=None, help="checkpoint .pt (default: published)")
    ap.add_argument("--out", default="latent_optim_sheet.png")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--channel-snr-db", type=float, default=None,
                     help="optimize under AWGN at this SNR (per-latent, "
                          "unit-RMS convention) instead of the clean decoder")
    ap.add_argument("--channel-samples", type=int, default=4,
                     help="Monte Carlo noise draws per optimizer step "
                          "when --channel-snr-db is set")
    ap.add_argument("--reg-weight", type=float, default=0.0,
                     help="L2 penalty pulling optimized latents back "
                          "toward the encoder's output. **Measured to do "
                          "nothing** when the objective includes the "
                          "channel, and to cost 0.3-1.0 dB above 1e-3; "
                          "kept only so that result stays reproducible. "
                          "See docs/latent-optimization.md.")
    ap.add_argument("--eval-snr-db", type=float, default=10.0,
                     help="SNR to evaluate both latents at for the "
                          "under-channel comparison, regardless of "
                          "--channel-snr-db")
    ap.add_argument("--mode", choices=sorted(MODES), default="C",
                     help="transmit mode: restricts optimization and both "
                          "reconstructions to this mode's latent groups "
                          "(A=1 group, B=2, C=3, all coarse-to-fine "
                          "prefixes), and is the mode any --wav-out is "
                          "modulated as")
    ap.add_argument("--callsign", default="",
                     help="passed through to Modem.modulate's beacon")
    ap.add_argument("--wav-out", default=None,
                     help="write the optimized latents through the real "
                          "modem to this WAV file")
    ap.add_argument("--wav-out-baseline", default=None,
                     help="also modulate the plain encoder latents to "
                          "this WAV file, for A/B comparison on air")
    args = ap.parse_args()

    torch.manual_seed(0)
    model = load_torch_model(args.model)
    spec = MODES[args.mode]
    active_channels = spec.groups * CHANNELS_PER_GROUP

    pil_img = fit_image(Image.open(args.image))
    arr = image_to_array(pil_img)
    img = torch.from_numpy(arr).unsqueeze(0)

    z0, z_opt, weights_mode = optimize_latents(
        model, img, args.mode, steps=args.steps, lr=args.lr,
        channel_snr_db=args.channel_snr_db,
        channel_samples=args.channel_samples, reg_weight=args.reg_weight)

    with torch.no_grad():
        enc_recon = model.decoder(z0, weights_mode)
        enc_mse = torch.nn.functional.mse_loss(enc_recon, img).item()
        opt_recon = model.decoder(z_opt, weights_mode)
        opt_mse = torch.nn.functional.mse_loss(opt_recon, img).item()
        z_dist = (z_opt - z0).norm().item()
        z0_rms = z0.flatten(1).pow(2).mean().sqrt().item()
        zopt_rms = z_opt.flatten(1).pow(2).mean().sqrt().item()

    print()
    print("=== Clean-channel reconstruction ===")
    print(f"  encoder   MSE={enc_mse:.6f}  PSNR={psnr(enc_mse):.2f} dB")
    print(f"  optimized MSE={opt_mse:.6f}  PSNR={psnr(opt_mse):.2f} dB")
    print(f"  delta     {psnr(opt_mse) - psnr(enc_mse):+.2f} dB")
    print(f"  ||z_opt - z_encoder||_2 = {z_dist:.3f}   "
          f"(rms: encoder={z0_rms:.4f}, optimized={zopt_rms:.4f})")

    print()
    print(f"=== Under simulated channel, SNR={args.eval_snr_db:.1f} dB "
          f"(mean of 32 draws) ===")
    enc_ch_mse = mc_channel_mse(model, z0, img, args.eval_snr_db, weights_mode, n=32)
    opt_ch_mse = mc_channel_mse(model, z_opt, img, args.eval_snr_db, weights_mode, n=32)
    print(f"  encoder   MSE={enc_ch_mse:.6f}  PSNR={psnr(enc_ch_mse):.2f} dB")
    print(f"  optimized MSE={opt_ch_mse:.6f}  PSNR={psnr(opt_ch_mse):.2f} dB")
    print(f"  delta     {psnr(opt_ch_mse) - psnr(enc_ch_mse):+.2f} dB")

    labels = ["original", f"encoder ({psnr(enc_mse):.1f} dB)",
              f"optimized ({psnr(opt_mse):.1f} dB)"]
    sheet = make_sheet(arr, enc_recon[0].numpy(), opt_recon[0].numpy(), labels)
    sheet.save(args.out)
    print(f"\nsheet saved to {args.out}")

    if args.wav_out or args.wav_out_baseline:
        modem = Modem()

        def to_wav(z, path):
            flat = model.latents_to_flat(z.detach())[0].numpy()
            x = modem.modulate(flat[: spec.n_latents], spec, callsign=args.callsign)
            wavio.write_wav(path, x)
            print(f"wrote {path}: mode {spec.name}, {spec.n_latents} latents, "
                  f"{spec.duration_s:.1f} s")

        if args.wav_out:
            to_wav(z_opt, args.wav_out)
        if args.wav_out_baseline:
            to_wav(z0, args.wav_out_baseline)


if __name__ == "__main__":
    main()

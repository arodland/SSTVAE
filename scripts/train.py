#!/usr/bin/env python3
"""Stage-1 training: autoencoder through the latent channel model.

Local smoke test (any torch device, incl. ROCm):
    python scripts/train.py --smoke --out runs/smoke

Real run (image folder, GPU):
    python scripts/train.py --data /path/to/images --epochs 60 --out runs/s1
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sstvae.data import FolderDataset, HFHubDataset, SyntheticDataset
from sstvae.latent_channel import ChannelConfig, apply_latent_channel
from sstvae.models import SSTVAE

_LUMA_WEIGHTS = torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)


def chroma(img: torch.Tensor) -> torch.Tensor:
    """Per-pixel color offset from gray (img minus its luma), broadcast
    back to 3 channels. Penalizing its MSE directly targets desaturation
    (regression toward gray) without HSV's singularities near black/white."""
    luma = (img * _LUMA_WEIGHTS.to(img.device, img.dtype)).sum(dim=1, keepdim=True)
    return img - luma


@torch.no_grad()
def evaluate(model, loader, device, max_batches=16):
    """Val PSNR at fixed channel settings: clean, and 8 dB + 20% erasures."""
    model.eval()
    cfgs = {
        "clean": None,
        "8dB_e20": ChannelConfig(
            snr_db_range=(8.0, 8.0), erasure_rate_max=0.2, p_truncate=0.0
        ),
    }
    mse = {k: 0.0 for k in cfgs}
    n = 0
    for bi, img in enumerate(loader):
        if bi >= max_batches:
            break
        img = img.to(device)
        z = model.encoder(img)
        for k, cfg in cfgs.items():
            if cfg is None:
                noisy, w = z, torch.ones_like(z)
            else:
                g = torch.Generator(device=device).manual_seed(bi)
                noisy, w, _conf = apply_latent_channel(z, cfg, generator=g)
            recon = model.decoder(noisy, w)
            mse[k] += F.mse_loss(recon, img).item() * img.shape[0]
        n += img.shape[0]
    model.train()
    return {k: -10 * torch.tensor(v / n).log10().item() for k, v in mse.items()}


def push_checkpoint(repo: str, out: Path, epoch: int) -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo, exist_ok=True, private=True)
    for name in ["checkpoint.pt", "metrics.jsonl", f"samples_{epoch:03d}.png"]:
        p = out / name
        if p.exists():
            api.upload_file(
                path_or_fileobj=str(p), path_in_repo=name, repo_id=repo
            )


@torch.no_grad()
def dump_samples(model, imgs, out_path, device):
    """Save [original | clean recon | noisy recon] grid for fixed images."""
    from torchvision.utils import save_image

    model.eval()
    imgs = imgs.to(device)
    z = model.encoder(imgs)
    clean = model.decoder(z, torch.ones_like(z))
    g = torch.Generator(device=device).manual_seed(0)
    cfg = ChannelConfig(snr_db_range=(8.0, 8.0), erasure_rate_max=0.2, p_truncate=0.0)
    noisy, w, _conf = apply_latent_channel(z, cfg, generator=g)
    rough = model.decoder(noisy, w)
    rows = torch.cat([imgs, clean, rough], dim=0)
    save_image(rows, out_path, nrow=imgs.shape[0])
    model.train()


def pick_device() -> torch.device:
    # torch.cuda covers ROCm builds as well
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_lpips(device):
    try:
        import lpips

        return lpips.LPIPS(net="vgg").to(device).eval()
    except Exception:
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=str, default=None, help="image folder")
    ap.add_argument(
        "--hf-dataset",
        type=str,
        default=None,
        help="Hub dataset repo (train/validation splits), e.g. arodland/coco320-sstvae",
    )
    ap.add_argument(
        "--push-to-hub",
        type=str,
        default=None,
        help="Hub model repo to upload checkpoint/metrics after each epoch",
    )
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument(
        "--epoch-size",
        type=int,
        default=None,
        help="random images per epoch (default: full dataset); "
        "gives frequent checkpoints/samples on big datasets",
    )
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--lpips-weight", type=float, default=0.5)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--smoke", action="store_true", help="tiny run on synthetic data")
    ap.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="bfloat16 autocast (halves activation VRAM)",
    )
    ap.add_argument(
        "--stage2",
        action="store_true",
        help="train through the differentiable OFDM waveform channel "
        "(clip/PAPR, fading, pilot-EQ residuals) instead of the "
        "latent-AWGN model; start from a stage-1 checkpoint",
    )
    ap.add_argument(
        "--papr-weight",
        type=float,
        default=0.002,
        help="weight on the continuous linear-ratio PAPR penalty "
        "(RADE-style: no hinge/target, just peak/mean power kept "
        "small and always-active so it can't dominate reconstruction "
        "loss — see radae/radae_base.py's distortion_loss). Default "
        "chosen so pre+post-clip ratios (~13 early in training) "
        "contribute roughly 2% of total loss, not 300%.",
    )
    ap.add_argument(
        "--chroma-weight",
        type=float,
        default=2.0,
        help="weight on MSE of the color-offset-from-gray vector; "
        "counters RGB-MSE's blind spot for desaturation (higher = "
        "more resistant to washing out saturated colors)",
    )
    ap.add_argument("--resume", type=str, default=None)
    ap.add_argument(
        "--seed",
        type=int,
        default=0,
        help="global RNG seed — sampler order and channel-noise draws "
        "become comparable across runs (not bit-exact GPU determinism, "
        "just enough for a fair A/B comparison)",
    )
    args = ap.parse_args()
    torch.manual_seed(args.seed)

    val_dataset = None
    if args.smoke:
        args.width = min(args.width, 32)
        args.epochs = 2
        args.batch = min(args.batch, 8)
        dataset = SyntheticDataset(n=128)
    elif args.hf_dataset:
        dataset = HFHubDataset(args.hf_dataset, split="train")
        val_dataset = HFHubDataset(args.hf_dataset, split="validation")
    elif args.data:
        dataset = FolderDataset(args.data)
    else:
        ap.error("--data or --hf-dataset is required unless --smoke")

    device = pick_device()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"device={device}, images={len(dataset)}, width={args.width}")

    model = SSTVAE(width=args.width).to(device)
    start_epoch = 0
    if args.resume:
        path = args.resume
        if path.startswith("hf://"):  # e.g. hf://arodland/sstvae-s1
            from huggingface_hub import hf_hub_download

            path = hf_hub_download(repo_id=path[5:], filename="checkpoint.pt")
        ckpt = torch.load(path, map_location=device)
        model.load_state_dict(ckpt["model"])
        start_epoch = ckpt.get("epoch", -1) + 1
        print(f"resumed from {args.resume} at epoch {start_epoch}")

    # Keep metrics.jsonl continuous across resumes: if this invocation's
    # push target already has history and we don't have it locally
    # (e.g. resuming in a fresh directory), pull it down first so we
    # append instead of silently truncating the pushed record.
    metrics_path = out / "metrics.jsonl"
    if args.push_to_hub and not metrics_path.exists():
        from huggingface_hub import hf_hub_download

        try:
            prior = hf_hub_download(repo_id=args.push_to_hub, filename="metrics.jsonl")
            metrics_path.write_bytes(Path(prior).read_bytes())
            print(f"seeded {metrics_path} from existing {args.push_to_hub}")
        except Exception:
            pass  # no prior history on the hub target — fine, start fresh

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    # T_max is this invocation's epoch count, not a global target, so the
    # cosine schedule restarts fresh on every resume rather than
    # continuing the decay from before.
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    sampler = None
    if args.epoch_size and args.epoch_size < len(dataset):
        # Fresh random subset every epoch (RandomSampler reshuffles).
        sampler = torch.utils.data.RandomSampler(
            dataset, replacement=False, num_samples=args.epoch_size
        )
    loader = DataLoader(
        dataset,
        batch_size=args.batch,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=0 if args.smoke else args.workers,
        drop_last=True,
    )
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset, batch_size=args.batch, shuffle=False, num_workers=2
        )
    lpips_fn = None if args.smoke else make_lpips(device)
    if lpips_fn is None and not args.smoke:
        print("lpips unavailable; training with MSE only")
    ch_cfg = ChannelConfig()
    wave_ch = None
    if args.stage2:
        from sstvae.waveform_channel import WaveformChannel

        wave_ch = WaveformChannel().to(device)

    step = 0
    for epoch in range(start_epoch, start_epoch + args.epochs):
        model.train()
        t0, ep_loss, n_batches = time.time(), 0.0, 0
        ep_papr_post, ep_papr_pre = 0.0, 0.0
        for img in loader:
            img = img.to(device, non_blocking=True)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=args.amp):
                z = model.encoder(img)
            papr_loss = 0.0
            if wave_ch is not None:
                # Waveform chain runs in fp32 (complex ops + autocast
                # don't mix); the networks stay under autocast.
                flat = model.latents_to_flat(z.float())
                noisy_flat, w_flat, papr_pre_db, papr_post_db, conf = wave_ch(flat)
                noisy = model.flat_to_latents(noisy_flat)
                w = model.flat_to_latents(w_flat)
                # RADE-style PAPR penalty: continuous linear peak/mean
                # power ratio, no hinge/target, small fixed weight (see
                # radae/radae_base.py's distortion_loss: `loss +=
                # (0.125/18) * PAPR` with PAPR = peak_power/av_power,
                # unhinged). A dB-scale hinge loss was tried first and
                # got stuck: log-compression flattens gradient for the
                # worst peaks (the opposite of what you want), and a
                # hinge either contributes 0 or grows unbounded past its
                # target, which let it balloon to ~3x the reconstruction
                # loss and still not move. Both pre- and post-clip still
                # contribute (post-clip is what RADE penalizes; pre-clip
                # is our own addition — see WaveformChannel._clip_filter
                # for why post-clip alone gives weak gradient once
                # clipping is active).
                papr_pre_ratio = 10 ** (papr_pre_db / 10)
                papr_post_ratio = 10 ** (papr_post_db / 10)
                papr_loss = args.papr_weight * (
                    papr_pre_ratio.mean() + papr_post_ratio.mean()
                )
            else:
                noisy, w, conf = apply_latent_channel(z, ch_cfg)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=args.amp):
                recon = model.decoder(noisy.to(z.dtype), w.to(z.dtype))
                recon = recon.float()
                loss = F.mse_loss(recon, img) + papr_loss
                if args.chroma_weight:
                    # Scaled by per-sample channel confidence (SNR/erasure/
                    # fading — NOT truncation): a clean mode-A-only sample
                    # gets the full penalty (excellent should mean
                    # saturated, regardless of how much was truncated),
                    # while a genuinely noisy sample is allowed to hedge
                    # toward gray instead of hallucinating color speckle.
                    chroma_mse = F.mse_loss(
                        chroma(recon), chroma(img), reduction="none"
                    ).mean(dim=(1, 2, 3))
                    loss = loss + args.chroma_weight * (
                        conf.to(chroma_mse.dtype) * chroma_mse
                    ).mean()
                if lpips_fn is not None:
                    # LPIPS is calibrated on small patches (~64-256 px);
                    # a random 256x256 crop keeps it at its trained scale
                    # and saves ~5x compute vs the full 640x480 frame.
                    ch = min(256, img.shape[-2])
                    cw = min(256, img.shape[-1])
                    top = int(torch.randint(0, img.shape[-2] - ch + 1, (1,)))
                    left = int(torch.randint(0, img.shape[-1] - cw + 1, (1,)))
                    rc = recon[..., top : top + ch, left : left + cw]
                    ic = img[..., top : top + ch, left : left + cw]
                    loss = loss + args.lpips_weight * lpips_fn(
                        rc * 2 - 1, ic * 2 - 1
                    ).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_loss += loss.item()
            if wave_ch is not None:
                ep_papr_post += papr_post_db.mean().item()
                ep_papr_pre += papr_pre_db.mean().item()
            n_batches += 1
            step += 1
        sched.step()
        avg = ep_loss / max(n_batches, 1)
        record = {"epoch": epoch, "train_loss": avg, "seconds": time.time() - t0}
        if wave_ch is not None:
            record["papr_db"] = ep_papr_post / max(n_batches, 1)
            record["papr_pre_db"] = ep_papr_pre / max(n_batches, 1)
        if val_loader is not None:
            record.update({f"val_psnr_{k}": v for k, v in
                           evaluate(model, val_loader, device).items()})
        print(
            f"epoch {epoch}: loss={avg:.5f}"
            + "".join(f" {k}={v:.2f}" for k, v in record.items()
                      if k.startswith("val_psnr") or k.startswith("papr"))
            + f" [{record['seconds']:.1f}s]"
        )
        with metrics_path.open("a") as fh:
            fh.write(json.dumps(record) + "\n")
        torch.save(
            {"model": model.state_dict(), "width": args.width, "epoch": epoch},
            out / "checkpoint.pt",
        )
        if epoch % 2 == 0 or epoch == start_epoch + args.epochs - 1:
            sample_src = val_dataset if val_dataset is not None else dataset
            sample_imgs = torch.stack([sample_src[i] for i in range(4)])
            dump_samples(model, sample_imgs, out / f"samples_{epoch:03d}.png", device)
        if args.push_to_hub:
            try:
                push_checkpoint(args.push_to_hub, out, epoch)
            except Exception as e:
                print(f"hub push failed (will retry next epoch): {e}")

    (out / "train_config.json").write_text(json.dumps(vars(args), indent=2))
    print(f"done; checkpoint at {out/'checkpoint.pt'}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Train the optional post-decoder refiner against a FROZEN codec.

Pairs are generated on the fly: image -> encoder -> stage-1 latent
channel -> decoder -> (degraded reconstruction, reference). The codec
never updates, so the refiner stays a pure receive-side option and any
existing checkpoint's pictures are unchanged with it disabled.

The loss is deliberately conservative (Charbonnier + light LPIPS, no
adversarial term): for a radio mode, plausible invented detail is worse
than visible degradation — watch the nonphoto/text metrics for exactly
that failure before believing a PSNR win.

Local smoke test (ROCm: use --no-amp):
    python scripts/train_refiner.py --smoke --out /tmp/refiner-smoke

Real run (frozen stage-1/2 checkpoint required):
    python scripts/train_refiner.py --codec runs/s1/checkpoint.pt \\
        --hf-dataset arodland/coco640-sstvae --epochs 40 --out runs/refiner
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sstvae.config import LATENT_CHANNELS, CHANNELS_PER_GROUP
from sstvae.data import (
    FolderDataset,
    HFHubDataset,
    NonPhotoDataset,
    SyntheticDataset,
)
from sstvae.latent_channel import ChannelConfig, apply_latent_channel
from sstvae.models import Refiner, SSTVAE

_LUMA_WEIGHTS = torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)


def chroma(img: torch.Tensor) -> torch.Tensor:
    """Same anti-desaturation term as scripts/train.py (see it for why)."""
    luma = (img * _LUMA_WEIGHTS.to(img.device, img.dtype)).sum(dim=1, keepdim=True)
    return img - luma


def charbonnier(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    return ((a - b).pow(2) + eps * eps).sqrt().mean()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_frozen_codec(path_arg: str, device) -> tuple[SSTVAE, str]:
    """Load the codec checkpoint the refiner is trained against, frozen.
    Returns (model, sha256) — the sha is stamped into the refiner
    checkpoint so a refiner can't silently run on a different codec's
    output (same reasoning as OnnxCodec's source_sha256 cross-check)."""
    path = path_arg
    if path.startswith("hf://"):
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(repo_id=path[5:], filename="checkpoint.pt")
    path = Path(path)
    sha = sha256_file(path)
    ckpt = torch.load(path, map_location=device)
    codec = SSTVAE(width=ckpt.get("width", 128)).to(device)
    codec.load_state_dict(ckpt["model"])
    codec.eval().requires_grad_(False)
    return codec, sha


def truncate_to_group0(
    noisy: torch.Tensor, w: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Zero all but group 0 — a deterministic mode-A reception, unlike
    ChannelConfig's random per-sample truncation."""
    group_idx = torch.arange(LATENT_CHANNELS, device=noisy.device)
    mask = (group_idx // CHANNELS_PER_GROUP == 0).view(1, -1, 1, 1).to(noisy.dtype)
    return noisy * mask, w * mask


@torch.no_grad()
def degrade(codec, img, cfg, generator=None):
    """image -> (base recon, weights, confidence) through the frozen codec."""
    z = codec.encoder(img)
    noisy, w, conf = apply_latent_channel(z, cfg, generator=generator)
    return codec.decoder(noisy, w), w, conf


@torch.no_grad()
def evaluate(codec, refiner, loader, device, max_batches=16):
    """Base vs refined PSNR at fixed conditions. The *delta* per bucket
    is the deliverable — modeA_8dB (most-damaged common case) is where
    the refiner has to earn its download."""
    refiner.eval()
    buckets = {
        "clean": ChannelConfig(snr_db_range=(22.0, 22.0), erasure_rate_max=0.0,
                               p_truncate=0.0),
        "8dB_e20": ChannelConfig(snr_db_range=(8.0, 8.0), erasure_rate_max=0.2,
                                 p_truncate=0.0),
        "modeA_8dB": ChannelConfig(snr_db_range=(8.0, 8.0), erasure_rate_max=0.2,
                                   p_truncate=0.0),
    }
    mse = {}
    n = 0
    for bi, img in enumerate(loader):
        if bi >= max_batches:
            break
        img = img.to(device)
        z = codec.encoder(img)
        for k, cfg in buckets.items():
            g = torch.Generator(device=device).manual_seed(bi)
            noisy, w, conf = apply_latent_channel(z, cfg, generator=g)
            if k == "modeA_8dB":
                noisy, w = truncate_to_group0(noisy, w)
            base = codec.decoder(noisy, w)
            refined = refiner(base, w, conf)
            mse[f"{k}_base"] = mse.get(f"{k}_base", 0.0) + \
                F.mse_loss(base, img).item() * img.shape[0]
            mse[f"{k}_ref"] = mse.get(f"{k}_ref", 0.0) + \
                F.mse_loss(refined, img).item() * img.shape[0]
        n += img.shape[0]
    refiner.train()
    return {k: -10 * torch.tensor(v / n).log10().item() for k, v in mse.items()}


@torch.no_grad()
def evaluate_nonphoto(codec, refiner, imgs, device, batch=8):
    """The hallucination watch: text/line-art is where an over-eager
    refiner invents plausible wrong detail. A refined PSNR *below* base
    here is disqualifying even if the photo buckets improved."""
    refiner.eval()
    cfg = ChannelConfig(snr_db_range=(8.0, 8.0), erasure_rate_max=0.2, p_truncate=0.0)
    mse = {"np_8dB_base": 0.0, "np_8dB_ref": 0.0}
    for b0 in range(0, len(imgs), batch):
        img = imgs[b0 : b0 + batch].to(device)
        g = torch.Generator(device=device).manual_seed(b0)
        base, w, conf = degrade(codec, img, cfg, generator=g)
        refined = refiner(base, w, conf)
        mse["np_8dB_base"] += F.mse_loss(base, img).item() * img.shape[0]
        mse["np_8dB_ref"] += F.mse_loss(refined, img).item() * img.shape[0]
    refiner.train()
    n = len(imgs)
    return {k: -10 * torch.tensor(v / n).log10().item() for k, v in mse.items()}


@torch.no_grad()
def dump_samples(codec, refiner, imgs, out_path, device):
    """[original | base recon | refined] at the 8dB+e20 condition."""
    from torchvision.utils import save_image

    refiner.eval()
    imgs = imgs.to(device)
    cfg = ChannelConfig(snr_db_range=(8.0, 8.0), erasure_rate_max=0.2, p_truncate=0.0)
    g = torch.Generator(device=device).manual_seed(0)
    base, w, conf = degrade(codec, imgs, cfg, generator=g)
    refined = refiner(base, w, conf)
    save_image(torch.cat([imgs, base, refined], dim=0), out_path, nrow=imgs.shape[0])
    refiner.train()


def pick_device() -> torch.device:
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


def push_checkpoint(repo: str, out: Path, epoch: int) -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo, exist_ok=True, private=True)
    for name in ["refiner.pt", "metrics.jsonl", f"samples_{epoch:03d}.png"]:
        p = out / name
        if p.exists():
            api.upload_file(path_or_fileobj=str(p), path_in_repo=name, repo_id=repo)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--codec",
        type=str,
        default=None,
        help="frozen SSTVAE checkpoint the refiner is trained against "
        "(.pt path or hf://repo); required unless --smoke",
    )
    ap.add_argument("--data", type=str, default=None, help="image folder")
    ap.add_argument("--hf-dataset", type=str, default=None,
                    help="Hub dataset repo (train/validation splits)")
    ap.add_argument("--push-to-hub", type=str, default=None,
                    help="Hub model repo to upload refiner/metrics after each epoch")
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--epoch-size", type=int, default=None,
                    help="random images per epoch (default: full dataset)")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument(
        "--augment",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="train-only augmentation (see sstvae/data.py); never on validation",
    )
    ap.add_argument(
        "--nonphoto-frac",
        type=float,
        default=0.0,
        help="fraction of the training mix from sstvae/nonphoto.py — worth "
        "sweeping here too: the refiner must at minimum not *hurt* text",
    )
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument(
        "--lpips-weight",
        type=float,
        default=0.2,
        help="kept lower than the codec's 0.5 on purpose: perceptual loss "
        "is the hallucination knob, and this model's whole pitch is "
        "faithfulness (see module docstring)",
    )
    ap.add_argument("--chroma-weight", type=float, default=2.0,
                    help="same anti-desaturation term as scripts/train.py, "
                    "confidence-scaled the same way")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--smoke", action="store_true",
                    help="tiny run on synthetic data with a random codec")
    ap.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="bfloat16 autocast",
    )
    ap.add_argument("--resume", type=str, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)

    val_dataset = None
    if args.smoke:
        args.width = min(args.width, 16)
        args.epochs = 2
        args.batch = min(args.batch, 4)
        dataset = SyntheticDataset(n=64)
    elif args.hf_dataset:
        dataset = HFHubDataset(args.hf_dataset, split="train", augment=args.augment)
        val_dataset = HFHubDataset(args.hf_dataset, split="validation")
    elif args.data:
        dataset = FolderDataset(args.data, augment=args.augment)
    else:
        ap.error("--data or --hf-dataset is required unless --smoke")
    if args.nonphoto_frac and not args.smoke:
        if not 0.0 < args.nonphoto_frac < 1.0:
            ap.error("--nonphoto-frac must be in (0, 1)")
        n_extra = round(args.nonphoto_frac / (1 - args.nonphoto_frac) * len(dataset))
        dataset = torch.utils.data.ConcatDataset(
            [dataset, NonPhotoDataset(n_extra, salt="train")]
        )
        print(f"nonphoto mix: {n_extra} synthetic images "
              f"({args.nonphoto_frac:.0%} of {len(dataset)})")

    device = pick_device()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.codec:
        codec, codec_sha = load_frozen_codec(args.codec, device)
    elif args.smoke:
        codec = SSTVAE(width=32).to(device).eval().requires_grad_(False)
        codec_sha = "smoke-random-codec"
        print("smoke: random-weight codec (pipeline check only)")
    else:
        ap.error("--codec is required unless --smoke")
    print(f"device={device}, images={len(dataset)}, refiner width={args.width}, "
          f"codec sha256:{codec_sha[:16]}...")

    refiner = Refiner(width=args.width).to(device)
    start_epoch = 0
    if args.resume:
        rpath = args.resume
        if rpath.startswith("hf://"):
            from huggingface_hub import hf_hub_download

            rpath = hf_hub_download(repo_id=rpath[5:], filename="refiner.pt")
        ckpt = torch.load(rpath, map_location=device)
        if ckpt.get("codec_sha256") not in (codec_sha, None):
            raise SystemExit(
                f"refiner checkpoint was trained against codec "
                f"{ckpt['codec_sha256'][:16]}..., but --codec is "
                f"{codec_sha[:16]}... — refusing to continue against a "
                "different codec's output distribution"
            )
        refiner.load_state_dict(ckpt["model"])
        start_epoch = ckpt.get("epoch", -1) + 1
        print(f"resumed from {args.resume} at epoch {start_epoch}")

    metrics_path = out / "metrics.jsonl"
    if args.push_to_hub and not metrics_path.exists():
        from huggingface_hub import hf_hub_download

        try:
            prior = hf_hub_download(repo_id=args.push_to_hub, filename="metrics.jsonl")
            metrics_path.write_bytes(Path(prior).read_bytes())
            print(f"seeded {metrics_path} from existing {args.push_to_hub}")
        except Exception:
            pass  # no prior history on the hub target — fine, start fresh

    opt = torch.optim.AdamW(refiner.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    sampler = None
    if args.epoch_size and args.epoch_size < len(dataset):
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
    nonphoto_val = None
    if not args.smoke:
        np_ds = NonPhotoDataset(48, salt="val")
        nonphoto_val = torch.stack([np_ds[i] for i in range(len(np_ds))])
    lpips_fn = None if args.smoke else make_lpips(device)
    if lpips_fn is None and not args.smoke:
        print("lpips unavailable; training without perceptual term")
    # Same distribution the codec trained under — the refiner must cover
    # everything the decoder can emit, including truncated modes.
    ch_cfg = ChannelConfig()

    for epoch in range(start_epoch, start_epoch + args.epochs):
        refiner.train()
        t0, ep_loss, n_batches = time.time(), 0.0, 0
        for img in loader:
            img = img.to(device, non_blocking=True)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=args.amp):
                base, w, conf = degrade(codec, img, ch_cfg)
                base = base.detach()
                refined = refiner(base, w, conf)
                refined = refined.float()
                loss = charbonnier(refined, img)
                if args.chroma_weight:
                    chroma_mse = F.mse_loss(
                        chroma(refined), chroma(img), reduction="none"
                    ).mean(dim=(1, 2, 3))
                    loss = loss + args.chroma_weight * (
                        conf.to(chroma_mse.dtype) * chroma_mse
                    ).mean()
                if lpips_fn is not None:
                    ch = min(256, img.shape[-2])
                    cw = min(256, img.shape[-1])
                    top = int(torch.randint(0, img.shape[-2] - ch + 1, (1,)))
                    left = int(torch.randint(0, img.shape[-1] - cw + 1, (1,)))
                    rc = refined[..., top : top + ch, left : left + cw]
                    ic = img[..., top : top + ch, left : left + cw]
                    loss = loss + args.lpips_weight * lpips_fn(
                        rc * 2 - 1, ic * 2 - 1
                    ).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(refiner.parameters(), 1.0)
            opt.step()
            ep_loss += loss.item()
            n_batches += 1
        sched.step()
        avg = ep_loss / max(n_batches, 1)
        record = {"epoch": epoch, "train_loss": avg, "seconds": time.time() - t0}
        if val_loader is not None:
            record.update({f"val_psnr_{k}": v for k, v in
                           evaluate(codec, refiner, val_loader, device).items()})
        if nonphoto_val is not None:
            record.update({f"val_psnr_{k}": v for k, v in
                           evaluate_nonphoto(codec, refiner, nonphoto_val,
                                             device).items()})
        print(
            f"epoch {epoch}: loss={avg:.5f}"
            + "".join(f" {k}={v:.2f}" for k, v in record.items()
                      if k.startswith("val_psnr"))
            + f" [{record['seconds']:.1f}s]"
        )
        with metrics_path.open("a") as fh:
            fh.write(json.dumps(record) + "\n")
        torch.save(
            {
                "model": refiner.state_dict(),
                "width": args.width,
                "epoch": epoch,
                "codec_sha256": codec_sha,
                "codec_width": codec.width,
            },
            out / "refiner.pt",
        )
        if epoch % 2 == 0 or epoch == start_epoch + args.epochs - 1:
            sample_src = val_dataset if val_dataset is not None else dataset
            sample_imgs = torch.stack([sample_src[i] for i in range(4)])
            dump_samples(codec, refiner, sample_imgs,
                         out / f"samples_{epoch:03d}.png", device)
        if args.push_to_hub:
            try:
                push_checkpoint(args.push_to_hub, out, epoch)
            except Exception as e:
                print(f"hub push failed (will retry next epoch): {e}")

    (out / "train_config.json").write_text(json.dumps(vars(args), indent=2))
    print(f"done; refiner at {out/'refiner.pt'}")


if __name__ == "__main__":
    main()

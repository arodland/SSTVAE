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

from sstvae.config import CHANNELS_PER_GROUP, CLIP_HEADROOM_DB
from sstvae.data import (
    FolderDataset,
    HFHubDataset,
    NonPhotoDataset,
    SyntheticDataset,
    overlay_text_batch,
)
from sstvae.latent_channel import ChannelConfig, apply_latent_channel
from sstvae.models import SSTVAE
from sstvae.pe_loss import pe_loss

_LUMA_WEIGHTS = torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)


def chroma(img: torch.Tensor) -> torch.Tensor:
    """Per-pixel color offset from gray (img minus its luma), broadcast
    back to 3 channels. Penalizing its MSE directly targets desaturation
    (regression toward gray) without HSV's singularities near black/white."""
    luma = (img * _LUMA_WEIGHTS.to(img.device, img.dtype)).sum(dim=1, keepdim=True)
    return img - luma


def lpips_sum(lpips_fn, recon, img, chunk=4):
    """Summed **full-frame** LPIPS over a batch, scored in chunks.

    Full frames, unlike the training term's random 256 px crop: a
    validation metric wants determinism and whole-picture coverage, and a
    random crop would put sampling noise into a number whose job is to be
    compared epoch to epoch. It is also the choice
    `scripts/ab_checkpoint_sweep.py` makes, so the two are comparable.

    Chunked because VGG activations at 640x480 are ~9x a 256 px crop's,
    and the val batch size was chosen for the autoencoder rather than
    for this.
    """
    total = 0.0
    for i in range(0, recon.shape[0], chunk):
        total += lpips_fn(
            recon[i : i + chunk] * 2 - 1, img[i : i + chunk] * 2 - 1
        ).sum().item()
    return total


@torch.no_grad()
def evaluate(model, loader, device, max_batches=16, lpips_fn=None):
    """Val PSNR at fixed channel settings: clean, 8 dB + 20% erasures, and
    clean-channel with burned-in text. Plus LPIPS on the same pictures if
    `lpips_fn` is given.

    The `text` variant deliberately uses the *unmodified* validation
    images plus a seeded overlay, so it tracks how well burned-in text
    survives without perturbing `clean` — whose value carries a long
    run-to-run history worth keeping comparable.

    Returns `(psnr, lpips)`; `lpips` is empty when no scorer was passed.
    Two dicts rather than one because the caller prefixes them
    differently, and a PSNR and an LPIPS under one key set would be
    telling the reader that higher is better for both.
    """
    model.eval()
    cfgs = {
        "clean": None,
        "8dB_e20": ChannelConfig(
            snr_db_range=(8.0, 8.0), erasure_rate_max=0.2, p_truncate=0.0
        ),
    }
    mse = {k: 0.0 for k in cfgs}
    mse["text"] = 0.0
    lp = {k: 0.0 for k in mse}
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
            if lpips_fn is not None:
                lp[k] += lpips_sum(lpips_fn, recon, img)
        timg = overlay_text_batch(img, seed=bi)
        tz = model.encoder(timg)
        trecon = model.decoder(tz, torch.ones_like(tz))
        mse["text"] += F.mse_loss(trecon, timg).item() * img.shape[0]
        if lpips_fn is not None:
            lp["text"] += lpips_sum(lpips_fn, trecon, timg)
        n += img.shape[0]
    model.train()
    return (
        {k: -10 * torch.tensor(v / n).log10().item() for k, v in mse.items()},
        {k: v / n for k, v in lp.items()} if lpips_fn is not None else {},
    )


@torch.no_grad()
def evaluate_waveform(model, loader, device, channels, max_batches=8,
                      lpips_fn=None):
    """Val PSNR *through the waveform channel* — the one stage 2 trains for.

    `evaluate()` above measures on the stage-1 latent channel whatever
    `--stage2` says, and that channel contains no clipper, so it cannot
    see the thing stage 2 is optimizing against. A stage-2 run judged on
    it alone is being scored by a channel it stopped training for.

    Fixed conditions and a per-batch seed, so the fading draws and the
    noise are identical across epochs and across runs: the point is a
    paired comparison, not an unbiased estimate of on-air PSNR. For that
    use scripts/ab_checkpoint_sweep.py, which runs the real modem.

    Returns `(psnr, lpips)` like `evaluate()`. This is the pair that
    matters for anything trading distortion against perceptual quality:
    `val_psnr_wave_mp8` is already how a stage-2 run is ranked, and
    `val_lpips_wave_mp8` is the other half of that judgement.
    """
    model.eval()
    mse = {k: 0.0 for k in channels}
    lp = {k: 0.0 for k in channels}
    n = 0
    # WaveformChannel.forward draws from the *global* RNG (unlike
    # apply_latent_channel, which takes a generator), so seeding it for a
    # repeatable channel would otherwise reach into the training stream
    # and change what the next epoch sees. Save and put back.
    cpu_state = torch.get_rng_state()
    cuda_state = (torch.cuda.get_rng_state_all()
                  if torch.cuda.is_available() else None)
    try:
        for bi, img in enumerate(loader):
            if bi >= max_batches:
                break
            img = img.to(device)
            z = model.encoder(img).float()
            flat = model.latents_to_flat(z)
            for k, ch in channels.items():
                torch.manual_seed(bi)
                noisy_flat, w_flat, _pre, _post, _conf = ch(flat)
                recon = model.decoder(
                    model.flat_to_latents(noisy_flat),
                    model.flat_to_latents(w_flat),
                )
                mse[k] += F.mse_loss(recon, img).item() * img.shape[0]
                if lpips_fn is not None:
                    lp[k] += lpips_sum(lpips_fn, recon, img)
            n += img.shape[0]
    finally:
        torch.set_rng_state(cpu_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)
        model.train()
    return (
        {k: -10 * torch.tensor(v / n).log10().item() for k, v in mse.items()},
        {k: v / n for k, v in lp.items()} if lpips_fn is not None else {},
    )


@torch.no_grad()
def evaluate_modes(model, loader, device, channel, max_batches=8):
    """Waveform-channel PSNR per transmitted mode: A/B/C = 1/2/3 groups.

    Every other metric in this file evaluates at full depth — the eval
    configs all set p_truncate=0.0 — so until this existed, *nothing*
    logged by this project had ever scored mode A or B, while training
    ran at p_truncate=0.5 and optimized all three. Mode B is expected to
    be the most-used on air (Andrew, 2026-08-14), so the mode the metrics
    covered was the one operators reach for least.

    Truncation is deterministic here, not sampled: mode k is exactly the
    trailing-group weight mask the channel applies at `keep=k`, which is
    what makes A/B/C comparable across epochs rather than a lottery.
    """
    model.eval()
    mse = {m: 0.0 for m in ("A", "B", "C")}
    n = 0
    cpu_state = torch.get_rng_state()
    cuda_state = (torch.cuda.get_rng_state_all()
                  if torch.cuda.is_available() else None)
    try:
        for bi, img in enumerate(loader):
            if bi >= max_batches:
                break
            img = img.to(device)
            z = model.encoder(img).float()
            torch.manual_seed(bi)
            noisy_flat, w_flat, _pre, _post, _conf = channel(model.latents_to_flat(z))
            noisy, w = model.flat_to_latents(noisy_flat), model.flat_to_latents(w_flat)
            gidx = torch.arange(w.shape[1], device=device) // CHANNELS_PER_GROUP
            for keep, name in enumerate(("A", "B", "C"), start=1):
                wk = w * (gidx[None, :, None, None] < keep)
                mse[name] += F.mse_loss(model.decoder(noisy, wk), img).item() * img.shape[0]
            n += img.shape[0]
    finally:
        torch.set_rng_state(cpu_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)
        model.train()
    return {k: -10 * torch.tensor(v / n).log10().item() for k, v in mse.items()}


@torch.no_grad()
def evaluate_nonphoto(model, imgs, device, batch=8):
    """Clean and 8 dB + 20%-erasure PSNR on the fixed non-photo val set
    (salt="val", disjoint from both training and the measured eval set).

    Logged whether or not --nonphoto-frac is on, so a photo-only
    baseline records the same metric the mixture-ratio sweep compares.
    """
    model.eval()
    cfg = ChannelConfig(snr_db_range=(8.0, 8.0), erasure_rate_max=0.2, p_truncate=0.0)
    mse = {"np_clean": 0.0, "np_8dB_e20": 0.0}
    for b0 in range(0, len(imgs), batch):
        img = imgs[b0 : b0 + batch].to(device)
        z = model.encoder(img)
        recon = model.decoder(z, torch.ones_like(z))
        mse["np_clean"] += F.mse_loss(recon, img).item() * img.shape[0]
        g = torch.Generator(device=device).manual_seed(b0)
        noisy, w, _conf = apply_latent_channel(z, cfg, generator=g)
        mse["np_8dB_e20"] += F.mse_loss(model.decoder(noisy, w), img).item() * img.shape[0]
    model.train()
    n = len(imgs)
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
    ap.add_argument(
        "--augment",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="train-only random zoom/pan crop + hflip + color jitter "
        "(see sstvae/data.py); never applied to the validation split",
    )
    ap.add_argument(
        "--nonphoto-frac",
        type=float,
        default=0.0,
        help="fraction of the training mix drawn from procedural "
        "non-photographic content (sstvae/nonphoto.py: test cards, "
        "callsign cards, text, line art, gradients, charts) — the "
        "classes measured 3-7 dB behind COCO on a photo-only model "
        "(docs/todo.md). 0 disables; sweep it, don't guess it, and "
        "watch val_psnr_clean for photo regression",
    )
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
        "contribute roughly 2%% of total loss, not 300%%.",
    )
    ap.add_argument(
        "--clip-headroom-db",
        type=float,
        default=CLIP_HEADROOM_DB,
        help="envelope clip threshold above mean envelope power, fed to "
        "WaveformChannel's Stage2Config (default matches the on-air "
        "modem's sstvae.config.CLIP_HEADROOM_DB). Genie-sweep testing "
        "(scripts/genie_papr_sweep.py) found this knob costs far less "
        "PSNR per dB of PAPR than pushing pre-clip crest factor down "
        "via --papr-weight, so prefer lowering this directly.",
    )
    ap.add_argument(
        "--chroma-weight",
        type=float,
        default=2.0,
        help="weight on MSE of the color-offset-from-gray vector; "
        "counters RGB-MSE's blind spot for desaturation (higher = "
        "more resistant to washing out saturated colors)",
    )
    ap.add_argument(
        "--pe-alpha",
        type=float,
        default=0.0,
        help="penalty factor of the PE loss (sstvae/pe_loss.py, "
        "doi:10.26599/CVM.2025.9450475), which replaces the flat MSE "
        "reconstruction term with one that amplifies error on pixels "
        "sitting on the blurred side of an edge. 0 disables (exactly, "
        "not approximately). The paper's 2.0 came from L1 backbones on "
        "an unstated intensity scale and does not transfer -- our p=2 "
        "squares the weight -- so sweep it, don't guess it, and watch "
        "PSNR, which this trades away for perceptual quality by design",
    )
    ap.add_argument(
        "--pe-conf-scale",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="scale --pe-alpha by the per-sample channel confidence, the "
        "same treatment --chroma-weight gets and for the same reason: "
        "the paper's premise is that blur is regression-to-the-mean on "
        "an ill-posed inverse, but here it is *also* correct MMSE "
        "hedging when the channel destroyed the information. Amplifying "
        "edge error on a badly-faded sample asks the decoder to invent "
        "edges. --no-pe-conf-scale gives the paper's loss unmodified",
    )
    ap.add_argument(
        "--optimizer",
        choices=["adamw", "muon"],
        default="adamw",
        help="muon = orthogonalized momentum on the conv weights, AdamW "
        "on biases and GroupNorm gains (sstvae/muon.py). Uses the "
        "match_rms_adamw lr scaling, so --lr keeps its AdamW meaning "
        "and a muon run is directly comparable to an adamw one at the "
        "same flags",
    )
    ap.add_argument(
        "--muon-adjust-lr-fn",
        choices=["match_rms_adamw", "original", "none"],
        default="match_rms_adamw",
        help="'original' puts the lr back on Muon's native ~0.02 scale; "
        "only useful with a matching --lr",
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
        dataset = HFHubDataset(args.hf_dataset, split="train", augment=args.augment)
        val_dataset = HFHubDataset(args.hf_dataset, split="validation")
    elif args.data:
        dataset = FolderDataset(args.data, augment=args.augment)
    else:
        ap.error("--data or --hf-dataset is required unless --smoke")
    if args.nonphoto_frac and not args.smoke:
        if not 0.0 < args.nonphoto_frac < 1.0:
            ap.error("--nonphoto-frac must be in (0, 1)")
        # Sized so nonphoto is `frac` of the combined pool; a RandomSampler
        # over the concatenation then keeps that fraction per epoch in
        # expectation, whether or not --epoch-size subsamples.
        n_extra = round(args.nonphoto_frac / (1 - args.nonphoto_frac) * len(dataset))
        dataset = torch.utils.data.ConcatDataset(
            [dataset, NonPhotoDataset(n_extra, salt="train")]
        )
        print(f"nonphoto mix: {n_extra} synthetic images "
              f"({args.nonphoto_frac:.0%} of {len(dataset)})")

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

    if args.optimizer == "muon":
        from sstvae.muon import Muon, build_param_groups

        opt = Muon(build_param_groups(model, lr=args.lr),
                   lr=args.lr, adjust_lr_fn=args.muon_adjust_lr_fn)
        n_muon = len(opt.param_groups[0]["params"])
        print(f"optimizer=muon ({n_muon} matrices, "
              f"adjust_lr_fn={args.muon_adjust_lr_fn}) + AdamW on "
              f"{len(opt.param_groups[1]['params'])} 1D params")
    else:
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
    nonphoto_val = None
    if not args.smoke:
        np_ds = NonPhotoDataset(48, salt="val")  # 8 per class, fixed
        nonphoto_val = torch.stack([np_ds[i] for i in range(len(np_ds))])
    lpips_fn = None if args.smoke else make_lpips(device)
    if lpips_fn is None and not args.smoke:
        print("lpips unavailable; training with MSE only")
    ch_cfg = ChannelConfig()
    wave_ch = None
    wave_val = {}
    if args.stage2:
        from sstvae.waveform_channel import Stage2Config, WaveformChannel

        wave_ch = WaveformChannel(
            Stage2Config(clip_headroom_db=args.clip_headroom_db)
        ).to(device)
        # Two fixed operating points rather than the training config's
        # ranges: an epoch-to-epoch comparison wants the same channel
        # every time, and the training range would resample SNR, fading
        # and truncation on every call.
        _val_common = dict(
            clip_headroom_db=args.clip_headroom_db,
            snr_db_range=(8.0, 8.0),
            p_truncate=0.0,
            erasure_bursts_mean=0.0,
        )
        wave_val = {
            "wave_awgn8": WaveformChannel(
                Stage2Config(p_fading=0.0, **_val_common)
            ).to(device),
            "wave_mp8": WaveformChannel(
                Stage2Config(
                    p_fading=1.0,
                    doppler_range_hz=(1.0, 1.0),
                    delay_range_ms=(2.0, 2.0),
                    **_val_common,
                )
            ).to(device),
        }

    step = 0
    for epoch in range(start_epoch, start_epoch + args.epochs):
        model.train()
        t0, ep_loss, n_batches = time.time(), 0.0, 0
        ep_papr_post, ep_papr_pre, ep_papr_loss = 0.0, 0.0, 0.0
        ep_lpips = 0.0
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
                if args.papr_weight:
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
                if args.pe_alpha:
                    # PE loss (sstvae/pe_loss.py) replaces the flat MSE
                    # rather than adding a term: it *is* the distortion
                    # loss, reweighted per pixel. alpha=0 gives back
                    # F.mse_loss exactly, so this branch is the only
                    # difference between a PE run and a baseline one.
                    alpha = args.pe_alpha
                    if args.pe_conf_scale:
                        alpha = alpha * conf.to(recon.dtype).view(-1, 1, 1, 1)
                    recon_loss = pe_loss(recon, img, alpha)
                else:
                    recon_loss = F.mse_loss(recon, img)
                loss = recon_loss + papr_loss
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
                    lp = lpips_fn(rc * 2 - 1, ic * 2 - 1).mean()
                    # Logged *unweighted*, so train_lpips stays comparable
                    # across runs with a different --lpips-weight -- the
                    # same reason ep_papr_loss is tracked separately.
                    ep_lpips += lp.detach().item()
                    loss = loss + args.lpips_weight * lp
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_loss += loss.item()
            if wave_ch is not None:
                ep_papr_post += papr_post_db.mean().item()
                ep_papr_pre += papr_pre_db.mean().item()
                # Tracked separately so train_loss stays comparable
                # against runs with a different --papr-weight: the PAPR
                # term moves the total for a reason that has nothing to
                # do with reconstruction quality.
                ep_papr_loss += (papr_loss.detach().item()
                                 if torch.is_tensor(papr_loss) else papr_loss)
            n_batches += 1
            step += 1
        sched.step()
        avg = ep_loss / max(n_batches, 1)
        record = {"epoch": epoch, "train_loss": avg, "seconds": time.time() - t0}
        if wave_ch is not None:
            record["papr_db"] = ep_papr_post / max(n_batches, 1)
            record["papr_pre_db"] = ep_papr_pre / max(n_batches, 1)
            # Deliberately still just the PAPR term: this field has
            # run-to-run history, and subtracting the LPIPS and chroma
            # terms too would silently change what old numbers meant.
            record["recon_loss"] = avg - ep_papr_loss / max(n_batches, 1)
        if lpips_fn is not None:
            # The training term's random-crop LPIPS, so it is *not* on the
            # same scale as the whole-frame val_lpips_* below. Compare it
            # against itself across epochs, never against those.
            record["train_lpips"] = ep_lpips / max(n_batches, 1)
        if val_loader is not None:
            psnr_v, lpips_v = evaluate(model, val_loader, device,
                                       lpips_fn=lpips_fn)
            record.update({f"val_psnr_{k}": v for k, v in psnr_v.items()})
            record.update({f"val_lpips_{k}": v for k, v in lpips_v.items()})
            if wave_val:
                psnr_w, lpips_w = evaluate_waveform(model, val_loader, device,
                                                    wave_val, lpips_fn=lpips_fn)
                record.update({f"val_psnr_{k}": v for k, v in psnr_w.items()})
                record.update({f"val_lpips_{k}": v for k, v in lpips_w.items()})
                # Per-mode, on the fading channel only: A/B/C differ by
                # how much data arrived, which the AWGN cell would show
                # identically for twice the compute.
                record.update({f"val_psnr_mode{k}": v for k, v in
                               evaluate_modes(model, val_loader, device,
                                              wave_val["wave_mp8"]).items()})
        if nonphoto_val is not None:
            record.update({f"val_psnr_{k}": v for k, v in
                           evaluate_nonphoto(model, nonphoto_val, device).items()})
        print(
            f"epoch {epoch}: loss={avg:.5f}"
            + "".join(f" {k}={v:.2f}" for k, v in record.items()
                      if k.startswith("val_psnr") or k.endswith("_db"))
            # LPIPS gets its own pass at .4f: the values run 0.05-0.3 and
            # the difference being looked for is in the third decimal, so
            # the .2f the dB figures want would print most of them equal.
            + "".join(f" {k}={v:.4f}" for k, v in record.items()
                      if "lpips" in k)
            # not in the loop above: the weight is O(1e-3), so the .2f
            # the dB figures want would print every value as 0.00.
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

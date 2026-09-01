#!/usr/bin/env python3
"""Paired A/B roundtrip comparison of two or more .pt checkpoints.

    python scripts/ab_checkpoint_sweep.py A.pt B.pt --csv ab.csv
    python scripts/ab_checkpoint_sweep.py ctl.pt d25.pt d50.pt \
        --content coco,nonphoto --metrics psnr,lpips,ms_ssim

Runs the whole path per point -- encode, modulate, simulated channel,
demodulate, decode -- for every checkpoint over the same images and the
*same channel seeds*, so every cell is a paired comparison and the
difference is attributable to the model alone.

The paired standard error is over per-image deltas, which is much
tighter than differencing two independent means: the image-to-image
spread (several dB of PSNR) cancels.

With more than two checkpoints, every delta is against the *first* one
-- pass the control first and the table reads as "what each variant did
to the baseline".

On judging a perceptual-loss change (the reason --metrics exists): a
metric that is in the training objective cannot referee it. LPIPS is in
every current run's loss at equal weight, so it is fair *between* these
runs but flatters them all against anything trained without it;
`haarpsi` (and `gmsd`, `fsim`) are in no run's loss and are the
disinterested judges; ms_ssim saturates here and is kept only as a
familiar reference point. DISTS is available and deliberately NOT in the
default set -- scoring a --dists-weight sweep on DISTS is marking your
own homework, and it will report a win whether or not the picture got
better. Read it as "did the term do what it was asked to", never as
evidence that it should ship. The eye is still the judge of record for
text and faces; --dump-dir is what feeds it.

SNRs are in `config.SNR_REF_BW_HZ`.
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sstvae import hfchannel  # noqa: E402
from sstvae.codec import load_codec, pad_to_full, reconstruct  # noqa: E402
from sstvae.config import MODES, SNR_REF_BW_HZ  # noqa: E402
from sstvae.images import fit_image  # noqa: E402
from sstvae.modem import Modem, SyncError  # noqa: E402

# (label, snr_db or None, fading preset or None)
CONDITIONS = [
    ("mpp -3 dB", -3.0, "mpp"),
    ("mpp 0 dB", 0.0, "mpp"),
    ("mpd 6 dB", 6.0, "mpd"),
    ("mpp 6 dB", 6.0, "mpp"),
    ("mpp 10 dB", 10.0, "mpp"),
    ("mpg 10 dB", 10.0, "mpg"),
    ("awgn 0 dB", 0.0, None),
    ("awgn 3 dB", 3.0, None),
    ("awgn 6 dB", 6.0, None),
    ("awgn 10 dB", 10.0, None),
    ("awgn 20 dB", 20.0, None),
    ("clean", None, None),
]


def psnr(a, b) -> float:
    x, y = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    mse = np.mean((x - y) ** 2)
    return float("inf") if mse == 0 else 10 * np.log10(255.0**2 / mse)


# ---------------------------------------------------------------- judges

def _to_nchw(images, scale_to_pm1: bool):
    """PIL list -> float32 NCHW tensor, [0,1] or [-1,1]."""
    import torch

    a = np.stack([np.asarray(im, dtype=np.float32) for im in images]) / 255.0
    t = torch.from_numpy(a).permute(0, 3, 1, 2).contiguous()
    return t * 2 - 1 if scale_to_pm1 else t


class _TorchJudge:
    """A batched torch perceptual metric over PIL pairs.

    Chunked because a 32-image VGG forward at 640x480 will not fit
    anywhere pleasant; the chunk is larger on a GPU, where the per-call
    overhead rather than the memory is what dominates.

    These are the only part of this sweep a GPU helps: the codec is
    CPU-only by construction (see `codec.load_torch_model`) and the
    modem is NumPy. In a run with perceptual metrics the judges are the
    bulk of the wall clock, so they are worth placing deliberately.
    """

    def __init__(self, build, scale_to_pm1: bool, device="cpu", chunk=None):
        self._build, self._pm1, self._device = build, scale_to_pm1, device
        self._chunk = chunk if chunk else (16 if str(device) != "cpu" else 4)
        self._net = None

    def __call__(self, refs, outs):
        import torch

        if self._net is None:
            net = self._build()
            # piq's functional wrappers are plain closures with no
            # parameters -- they follow their inputs' device. Only the
            # module-based judges (LPIPS, DISTS) carry VGG weights.
            self._net = net.to(self._device) if hasattr(net, "to") else net
        got = np.full(len(refs), np.nan, dtype=float)
        with torch.no_grad():
            for i in range(0, len(refs), self._chunk):
                r = _to_nchw(refs[i: i + self._chunk], self._pm1).to(self._device)
                o = _to_nchw(outs[i: i + self._chunk], self._pm1).to(self._device)
                v = self._net(o, r).flatten()
                # A metric whose reduction collapses the batch returns one
                # number for the chunk, which would silently land in the
                # first slot and leave the rest untouched -- piq's modules
                # default to reduction='mean' and do exactly that. Assert
                # rather than trust: the symptom is a plausible-looking
                # column of zeros, not an error.
                if len(v) != r.shape[0]:
                    raise RuntimeError(
                        f"judge returned {len(v)} scores for a chunk of "
                        f"{r.shape[0]}; it needs a per-image reduction"
                    )
                got[i: i + len(v)] = v.cpu().numpy()
        return got


def _build_lpips():
    import lpips

    # Already per-image: returns (N, 1, 1, 1).
    return lpips.LPIPS(net="vgg").eval()


def _build_dists():
    from piq import DISTS

    # reduction='none' -- the default is 'mean', which returns one scalar
    # for the whole batch (see the check in _TorchJudge.__call__).
    return DISTS(reduction="none")


def _piq_functional(name: str):
    """Wrap a piq functional metric to the (x, y) -> per-image tensor shape
    the judge protocol expects. data_range=1 matches _to_nchw."""

    def build():
        import piq
        import torch

        fn = getattr(piq, name)

        def f(x, y):
            return torch.stack([
                fn(x[i: i + 1], y[i: i + 1], data_range=1.0)
                for i in range(x.shape[0])
            ])

        return f

    return build


# name -> (callable factory, higher_is_better)
JUDGES = {
    "psnr": (None, True),  # handled inline, no torch
    "lpips": (lambda d: _TorchJudge(_build_lpips, True, d), False),
    "dists": (lambda d: _TorchJudge(_build_dists, False, d), False),
    # Non-objective judges. ms_ssim saturates above ~0.99 on
    # reconstructions this good (deltas land in the 4th decimal and say
    # nothing), so haarpsi is the default disinterested one -- it keeps
    # usable dynamic range at high quality. gmsd is its lower-is-better
    # counterpart, kept as a cross-check with a different failure mode.
    "ms_ssim": (lambda d: _TorchJudge(_piq_functional("multi_scale_ssim"), False, d), True),
    "haarpsi": (lambda d: _TorchJudge(_piq_functional("haarpsi"), False, d), True),
    "gmsd": (lambda d: _TorchJudge(_piq_functional("gmsd"), False, d), False),
    "fsim": (lambda d: _TorchJudge(_piq_functional("fsim"), False, d), True),
}


# --------------------------------------------------------------- content

def pick_device(requested: str):
    """'auto' -> the GPU if there is one. torch.cuda covers ROCm builds."""
    import torch

    if requested != "auto":
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_coco(repo: str, split: str, n: int):
    from datasets import load_dataset

    ds = load_dataset(repo, split=split)
    return [("coco", fit_image(ds[i]["image"])) for i in range(min(n, len(ds)))]


def load_nonphoto(per_class: int, salt: str = "eval"):
    """Procedural operator content, labelled per class.

    Reported per class rather than pooled because the classes are the
    question: `callsign` and `text` are where a texture-tolerant loss is
    most likely to do harm, and averaging them into `testcard` and
    `gradient` is exactly how that would be hidden.
    """
    from sstvae import nonphoto

    return [
        (cls, nonphoto.generate(cls, i, salt=salt))
        for cls in nonphoto.CLASSES
        for i in range(per_class)
    ]


def load_dir(path: Path, n: int):
    from PIL import Image

    files = sorted(
        p for p in path.iterdir()
        if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp")
    )[:n]
    return [(path.name, fit_image(Image.open(p))) for p in files]


def abs_diff_tile(a, b, gain: int):
    """Per-channel |a - b|, multiplied up so a one-or-two-level difference
    is actually visible, and clipped at white.

    Kept in RGB rather than reduced to luma on purpose: a difference that
    lives in one channel is a colour shift, which is precisely what a
    chroma-weighted loss can introduce and what a greyscale panel would
    hide. The true peak is returned so the caption can state it -- the
    gain is fixed rather than auto-normalized so two strips are
    comparable, and without the peak a saturated panel and a merely
    bright one look the same.
    """
    from PIL import Image

    d = np.abs(np.asarray(a, dtype=np.int16) - np.asarray(b, dtype=np.int16))
    tile = Image.fromarray(np.clip(d * gain, 0, 255).astype(np.uint8))
    return tile, int(d.max())


def save_strip(out_path: Path, src, recons, labels, diff_gain: int = 8):
    """[source | model0 | model1 | ... | amplified difference] for the eye.

    The difference panel is only drawn for a two-model run: with more
    than two there is no single pair to difference, and picking one
    silently would be worse than omitting it.
    """
    from PIL import Image, ImageDraw

    tiles = [(src, "source")]
    for r, lab in zip(recons, labels):
        tiles.append((r if r is not None else Image.new("RGB", src.size), lab))
    if len(recons) == 2 and all(r is not None for r in recons):
        tile, peak = abs_diff_tile(recons[0], recons[1], diff_gain)
        tiles.append(
            (tile, f"|{labels[0]}-{labels[1]}| x{diff_gain}  peak {peak}/255")
        )

    w, h = src.size
    strip = Image.new("RGB", (w * len(tiles), h + 14), (0, 0, 0))
    d = ImageDraw.Draw(strip)
    for i, (t, lab) in enumerate(tiles):
        strip.paste(t, (i * w, 14))
        d.text((i * w + 4, 2), lab, fill=(255, 255, 255))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    strip.save(out_path)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("models", nargs="+", help="2+ checkpoints; deltas are vs the first")
    ap.add_argument("--labels", default=None, help="comma-separated names for the models")
    ap.add_argument("--dataset", default="arodland/coco640-sstvae")
    ap.add_argument("--split", default="validation")
    ap.add_argument("--n-images", type=int, default=32)
    ap.add_argument(
        "--content",
        default="coco",
        help="comma-separated: coco, nonphoto, and/or a directory path "
        "(e.g. a folder of faces). Each is scored and reported separately",
    )
    ap.add_argument(
        "--nonphoto-per-class",
        type=int,
        default=4,
        help="images per non-photo class (6 classes: see sstvae/nonphoto.py)",
    )
    ap.add_argument(
        "--metrics",
        default="psnr,lpips",
        help="comma-separated: psnr, lpips, haarpsi, gmsd, fsim, ms_ssim, "
        "dists. See the module "
        "docstring on why dists is not a default and ms_ssim is the "
        "disinterested one",
    )
    ap.add_argument("--modes", default="ABC")
    ap.add_argument(
        "--conditions",
        default=None,
        help="comma-separated substrings to keep (e.g. 'clean,awgn 6,mpp 6'); "
        "default runs all of them",
    )
    ap.add_argument(
        "--device",
        default="auto",
        help="device for the perceptual judges: auto (default), cpu, cuda. "
        "Only the judges move -- the codec is CPU-only by construction "
        "(codec.load_torch_model: a 640x480 pass is a few ms against "
        "~270 ms of NumPy DSP) and the modem is NumPy, so a psnr-only run "
        "is CPU-bound either way. The judges themselves are ~15x on a GPU, "
        "which is ~2x on the whole sweep. Expect the FIRST gpu run to be "
        "slower than cpu: ROCm/MIOpen compiles a kernel per tensor shape "
        "(measured 99 s cold against 18 s warm and 37 s on cpu). That "
        "cache persists in ~/.cache/miopen, so it is a one-time cost, not "
        "a reason to pass --device cpu"
    )
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--csv", type=Path, default=None)
    ap.add_argument(
        "--dump-dir",
        type=Path,
        default=None,
        help="write [source | each model | amplified difference] strips "
        "here for eyeballing; the difference panel is drawn only for a "
        "two-model run",
    )
    ap.add_argument("--dump-per-class", type=int, default=2)
    ap.add_argument(
        "--diff-gain",
        type=int,
        default=8,
        help="amplification for the strip's |model0 - model1| panel "
        "(fixed, not auto-normalized, so strips stay comparable; the "
        "caption states the true peak)",
    )
    args = ap.parse_args()

    if len(args.models) < 2:
        ap.error("need at least two checkpoints to compare")
    labels = (
        args.labels.split(",") if args.labels
        else [Path(m).parent.name or Path(m).stem for m in args.models]
    )
    if len(labels) != len(args.models):
        ap.error(f"--labels has {len(labels)} names for {len(args.models)} models")

    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    for m in metrics:
        if m not in JUDGES:
            ap.error(f"unknown metric {m!r}; choose from {sorted(JUDGES)}")
    judges = {}
    need_torch = [m for m in metrics if JUDGES[m][0] is not None]
    device = pick_device(args.device) if need_torch else "cpu"
    if need_torch:
        print(f"perceptual judges on {device}", file=sys.stderr)
    for m in metrics:
        factory, _hb = JUDGES[m]
        if factory is None:
            continue
        try:
            judges[m] = factory(device)
        except Exception as e:  # a missing lpips/piq should not kill the run
            print(f"metric {m} unavailable ({e}); skipping", file=sys.stderr)
    metrics = [m for m in metrics if m == "psnr" or m in judges]
    if "dists" in metrics:
        print(
            "NOTE: dists is in the metric set. If any model was trained with "
            "--dists-weight, this metric is its own training objective and "
            "cannot referee the comparison.",
            file=sys.stderr,
        )

    # Content sets, each a list of (class_name, PIL).
    content = []
    for spec in args.content.split(","):
        spec = spec.strip()
        if spec == "coco":
            content += load_coco(args.dataset, args.split, args.n_images)
        elif spec == "nonphoto":
            content += load_nonphoto(args.nonphoto_per_class)
        elif spec:
            content += load_dir(Path(spec), args.n_images)
    if not content:
        ap.error("no images selected by --content")
    classes = sorted({c for c, _ in content})

    conds = CONDITIONS
    if args.conditions:
        keep = [c.strip() for c in args.conditions.split(",")]
        conds = [c for c in CONDITIONS if any(k in c[0] for k in keep)]
        if not conds:
            ap.error(f"--conditions matched nothing in {[c[0] for c in CONDITIONS]}")

    print(
        f"{len(content)} images ({', '.join(classes)}), "
        f"{len(args.models)} models, metrics {','.join(metrics)}, "
        f"SNR referenced to {SNR_REF_BW_HZ:.0f} Hz",
        file=sys.stderr,
    )

    codecs = [load_codec(p) for p in args.models]
    modem = Modem()
    specs = [MODES[m] for m in args.modes]
    images = [im for _c, im in content]
    img_class = np.array([c for c, _im in content])
    nm = len(args.models)

    # Encode once per (model, image); every mode is a prefix of the same
    # latent vector.
    latents = [[c.encode(img) for img in images] for c in codecs]

    rows = []
    for spec in specs:
        waves = [
            [modem.modulate(lat[: spec.n_latents], spec) for lat in per_model]
            for per_model in latents
        ]
        for label, snr_db, fading in conds:
            # Keep the decoded pictures: the perceptual judges are batched
            # and the dump needs them anyway.
            recons = [[None] * len(images) for _ in range(nm)]
            for i in range(len(images)):
                for m in range(nm):
                    y = waves[m][i]
                    if snr_db is not None or fading is not None:
                        y = hfchannel.apply_channel(
                            y, snr_db=snr_db, fading_preset=fading,
                            seed=args.seed + i,
                        )
                    try:
                        r = modem.demodulate(y)
                    except SyncError:
                        continue
                    recons[m][i] = reconstruct(
                        codecs[m], pad_to_full(r.latents), pad_to_full(r.weights)
                    )

            ok = np.array([[recons[m][i] is not None for i in range(len(images))]
                           for m in range(nm)])

            if args.dump_dir:
                for cls in classes:
                    idx = np.flatnonzero(img_class == cls)[: args.dump_per_class]
                    for i in idx:
                        save_strip(
                            args.dump_dir / spec.name /
                            f"{label.replace(' ', '_')}_{cls}_{i}.png",
                            images[i], [recons[m][i] for m in range(nm)], labels,
                            diff_gain=args.diff_gain,
                        )

            for metric in metrics:
                _f, higher_better = JUDGES[metric]
                scores = np.full((nm, len(images)), np.nan)
                for m in range(nm):
                    sel = np.flatnonzero(ok[m])
                    if not len(sel):
                        continue
                    if metric == "psnr":
                        scores[m, sel] = [psnr(images[i], recons[m][i]) for i in sel]
                    else:
                        scores[m, sel] = judges[metric](
                            [images[i] for i in sel],
                            [recons[m][i] for i in sel],
                        )

                for cls in classes:
                    csel = img_class == cls
                    base = scores[0][csel]
                    for m in range(nm):
                        cur = scores[m][csel]
                        both = ~np.isnan(base) & ~np.isnan(cur)
                        n_both = int(np.count_nonzero(both))
                        mean = float(np.nanmean(cur)) if np.any(~np.isnan(cur)) else float("nan")
                        if m == 0 or not n_both:
                            delta = sem = float("nan")
                            wins = 0
                        else:
                            d = cur[both] - base[both]
                            delta = float(d.mean())
                            sem = (float(d.std(ddof=1) / np.sqrt(n_both))
                                   if n_both > 1 else float("nan"))
                            wins = int(np.count_nonzero(d > 0 if higher_better else d < 0))
                        rows.append({
                            "content": cls, "mode": spec.name, "condition": label,
                            "metric": metric, "model": labels[m], "value": mean,
                            "delta": delta, "sem": sem, "n_paired": n_both,
                            "wins": wins,
                            "sync": f"{int(np.count_nonzero(ok[m] & csel))}/"
                                    f"{int(np.count_nonzero(csel))}",
                                    })
                    if metric == metrics[0]:
                        print(
                            f"  mode {spec.name} {label:>10} {cls:>9}: "
                            + "  ".join(
                                f"{labels[m]} {float(np.nanmean(scores[m][csel])):5.2f}"
                                for m in range(nm)
                            ),
                            file=sys.stderr, flush=True,
                        )

    print("\n".join(f"{chr(65 + i)} = {m}  ({labels[i]})"
                    for i, m in enumerate(args.models)))
    print(f"\nbaseline = {labels[0]}; delta = model - baseline, paired per image.")
    print("psnr/ms_ssim higher is better; lpips/dists lower is better.\n")
    print("| Content | Mode | Condition | Metric | Model | Value | delta | +/- | wins | sync |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        d = "--" if r["model"] == labels[0] else f"{r['delta']:+.4f}"
        s = "--" if r["model"] == labels[0] else f"{r['sem']:.4f}"
        w = "--" if r["model"] == labels[0] else f"{r['wins']}/{r['n_paired']}"
        print(
            f"| {r['content']} | {r['mode']} | {r['condition']} | {r['metric']} | "
            f"{r['model']} | {r['value']:.4f} | {d} | {s} | {w} | {r['sync']} |"
        )

    if args.csv:
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {args.csv}", file=sys.stderr)


if __name__ == "__main__":
    main()

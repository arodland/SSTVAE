#!/usr/bin/env python3
"""Export a `.pt` checkpoint to the published ONNX artifacts.

Run at **publish time**, from the checkpoint being published, so the
artifacts cannot drift from it. See `docs/onnx.md` for why this exists
and what each precision costs; the short version is that `onnxruntime`
is 27 MB against torch's 336 MB, and the receiving path only ever needs
two convolutional passes.

Six artifacts per checkpoint -- {encoder, decoder} x {fp32, fp16, int8}
-- named after the checkpoint they came from:

    v1.pt  ->  v1-encoder-fp32.onnx   v1-decoder-fp32.onnx
               v1-encoder-fp16.onnx   v1-decoder-fp16.onnx
               v1-encoder-int8.onnx   v1-decoder-int8.onnx

All three precisions are published with every revision because every
precision decodes on every other precision's receiver (measured: fp32
ONNX and torch agree to ~1e-6, roughly 112 dB below the channel noise).
Publishing them saves third parties from rolling their own export, which
is the case that actually risks divergence. **There is one on-air
format, and the precisions are not variants of it.**

Usage:

    scripts/export_onnx.py                      # published checkpoint
    scripts/export_onnx.py --model out/best.pt
    scripts/export_onnx.py --push               # upload to the Hub

Nothing is uploaded without `--push`. Every artifact is verified against
the torch model before it is written, and a tolerance breach fails the
run rather than publishing a bad codec.
"""

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sstvae import checkpoint  # noqa: E402
from sstvae.codec import load_model  # noqa: E402
from sstvae.config import LATENT_CHANNELS, LATENT_H, LATENT_W  # noqa: E402
from sstvae.images import IMG_H, IMG_W  # noqa: E402

OPSET = 17
PRECISIONS = ("fp32", "fp16", "int8")

# The yardstick for quantisation error is not fp32, it is the channel:
# at the modem's operating point there is this much RMS noise on
# unit-RMS latents. Quantisation noise is just one more small additive
# source on a channel that already carries a much larger one.
CHANNEL_NOISE_RMS = 0.367

# Gates, not targets. Measured values (docs/onnx.md) sit far below these;
# they exist to catch a broken export, not to police the last decimal.
#
#   latent_rms    RMS encoder error on unit-RMS latents. Compare against
#                 CHANNEL_NOISE_RMS -- that is the yardstick, not fp32.
#   quality_db    PSNR *lost* against the source image, running the whole
#                 pipeline at this precision versus running it in torch.
#
# The second one is the metric that means something. PSNR of an ONNX
# reconstruction against the torch reconstruction is a difference with no
# natural scale: int8 scores ~24 dB on it while costing ~0.15 dB of
# actual picture quality, so gating on it would reject a good artifact.
#
# The int8 gates are loose because int8 really is coarse here. Measured
# 2026-07-27 over 10 COCO val images: latent RMS 1.88e-01, costing
# 0.28 dB of picture. `quantize_dynamic` turns every Conv into
# `ConvInteger`, which supports only a **per-tensor** weight scale --
# `per_channel=True` is silently a no-op on this graph, verified. Static
# (calibrated) quantisation is the lever if int8 accuracy ever matters;
# it is not implemented here.
TOLERANCES = {
    "fp32": {"latent_rms": 1e-4, "quality_db": 0.02},
    "fp16": {"latent_rms": 5e-3, "quality_db": 0.05},
    "int8": {"latent_rms": 0.25, "quality_db": 0.50},
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def probe_images(n: int, seed: int = 0) -> torch.Tensor:
    """Deterministic stand-in images for verification, in [0,1].

    Low-frequency content rather than white noise: a convolutional
    autoencoder trained on photographs behaves quite differently on
    broadband noise, and a verification input that the model has no
    idea what to do with tells you little about whether the export is
    faithful. Pass `--images` for the real thing when it matters.
    """
    g = torch.Generator().manual_seed(seed)
    yy = torch.linspace(0, 1, IMG_H)[:, None]
    xx = torch.linspace(0, 1, IMG_W)[None, :]
    out = []
    for _ in range(n):
        img = torch.zeros(3, IMG_H, IMG_W)
        for c in range(3):
            acc = torch.zeros(IMG_H, IMG_W)
            for _ in range(6):
                fx, fy = torch.rand(2, generator=g) * 6
                ph = torch.rand(1, generator=g) * 6.283
                amp = torch.rand(1, generator=g)
                acc += amp * torch.sin(6.283 * (fx * xx + fy * yy) + ph)
            img[c] = acc
        img = (img - img.amin()) / (img.amax() - img.amin() + 1e-8)
        out.append(img)
    return torch.stack(out)


def load_images(paths: list[Path]) -> torch.Tensor:
    from PIL import Image

    from sstvae.images import fit_image, image_to_tensor

    return torch.stack([image_to_tensor(fit_image(Image.open(p))) for p in paths])


def export_fp32(module: torch.nn.Module, args: tuple, path: Path,
                input_names: list[str], output_names: list[str]) -> None:
    """Export one module at fp32.

    `dynamo=True` is the default and the maintained path; the legacy
    TorchScript exporter produces a numerically identical graph, so
    there is no reason to opt out. `external_data=False` is **not** the
    default and matters: without it the weights land in a `.onnx.data`
    sidecar, and four artifacts that must arrive together are worse than
    two.
    """
    torch.onnx.export(
        module, args, str(path),
        input_names=input_names, output_names=output_names,
        opset_version=OPSET, dynamo=True, external_data=False,
    )


def convert_fp16(src: Path, dst: Path) -> None:
    import onnx
    from onnxconverter_common import float16

    model = onnx.load(str(src))
    # keep_io_types: the caller still hands us fp32 arrays and gets fp32
    # back. The precision is an internal storage choice, not part of the
    # interface -- codec.py should not need to know which file it loaded.
    onnx.save(float16.convert_float_to_float16(model, keep_io_types=True), str(dst))


def convert_int8(src: Path, dst: Path) -> None:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(str(src), str(dst), weight_type=QuantType.QInt8)


def stamp_metadata(path: Path, props: dict) -> None:
    """Record provenance inside the artifact.

    Which checkpoint an `.onnx` came from is exactly the question that
    gets asked when two stations disagree, and an answer that lives in
    the file cannot be separated from it.
    """
    import onnx

    model = onnx.load(str(path))
    for k, v in props.items():
        entry = model.metadata_props.add()
        entry.key, entry.value = str(k), str(v)
    onnx.save(model, str(path))


def run_onnx(path: Path, feeds: dict) -> np.ndarray:
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 4  # measured best; 1 was ~5x worse
    opts.log_severity_level = 3
    sess = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    if len(sess.get_inputs()) == 1:
        return sess.run(None, {name: feeds["primary"]})[0]
    second = sess.get_inputs()[1].name
    return sess.run(None, {name: feeds["primary"], second: feeds["secondary"]})[0]


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return float("inf") if mse == 0 else 10.0 * float(np.log10(1.0 / mse))


def verify(model, images: torch.Tensor, paths: dict, precision: str) -> dict:
    """Compare one precision's encoder and decoder against torch.

    One image at a time: the graphs are exported at a fixed batch of 1,
    which is what the app actually does and what keeps the export in the
    easy, fully-static regime.
    """
    lat_errs, vs_torch, torch_q, onnx_q = [], [], [], []
    for i in range(images.shape[0]):
        one = images[i : i + 1]
        src = one.numpy().astype(np.float32)
        with torch.no_grad():
            ref_latents = model.encoder(one)
            weights = torch.ones_like(ref_latents)
            ref_image = model.decoder(ref_latents, weights).numpy()

        got_latents = run_onnx(paths["encoder"], {"primary": src})
        lat_errs.append(got_latents - ref_latents.numpy())

        w_np = weights.numpy().astype(np.float32)
        # Decoder alone, from the *torch* latents: isolates decoder error
        # instead of compounding the encoder's.
        vs_torch.append(psnr(
            run_onnx(paths["decoder"],
                     {"primary": ref_latents.numpy().astype(np.float32),
                      "secondary": w_np}),
            ref_image,
        ))
        # Whole pipeline at this precision -- what a station running these
        # artifacts actually gets -- measured against the source image.
        onnx_q.append(psnr(
            run_onnx(paths["decoder"],
                     {"primary": got_latents.astype(np.float32),
                      "secondary": w_np}),
            src,
        ))
        torch_q.append(psnr(ref_image, src))

    lat_err = np.concatenate([e.ravel() for e in lat_errs])
    latent_rms = float(np.sqrt(np.mean(lat_err ** 2)))
    quality_db = float(np.mean(torch_q) - np.mean(onnx_q))

    tol = TOLERANCES[precision]
    return {
        "precision": precision,
        "latent_rms": latent_rms,
        "latent_max": float(np.abs(lat_err).max()),
        "latent_vs_channel_db": (
            float("inf") if latent_rms == 0
            else 20.0 * float(np.log10(CHANNEL_NOISE_RMS / latent_rms))
        ),
        "torch_psnr_db": float(np.mean(torch_q)),
        "onnx_psnr_db": float(np.mean(onnx_q)),
        "quality_lost_db": quality_db,
        "decoder_vs_torch_psnr_db": float(np.mean(vs_torch)),
        "encoder_mb": paths["encoder"].stat().st_size / 1e6,
        "decoder_mb": paths["decoder"].stat().st_size / 1e6,
        "ok": latent_rms <= tol["latent_rms"] and quality_db <= tol["quality_db"],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=None,
                    help="checkpoint .pt; defaults to the published one")
    ap.add_argument("--out", default="onnx", type=Path,
                    help="output directory (default: ./onnx)")
    ap.add_argument("--stem", default=None,
                    help="artifact name prefix; defaults to the checkpoint stem "
                         "(v1.pt -> v1-encoder-fp16.onnx)")
    ap.add_argument("--precisions", default=",".join(PRECISIONS),
                    help=f"comma-separated subset of {','.join(PRECISIONS)}")
    ap.add_argument("--images", type=Path, default=None,
                    help="directory of images to verify against; synthetic "
                         "low-frequency probes are used if omitted")
    ap.add_argument("--n-probe", type=int, default=4,
                    help="number of verification images (default 4)")
    ap.add_argument("--push", action="store_true",
                    help="upload the artifacts to the Hub after verifying")
    ap.add_argument("--repo", default=checkpoint.DEFAULT_REPO,
                    help=f"Hub repo to push to (default {checkpoint.DEFAULT_REPO})")
    args = ap.parse_args()

    precisions = [p.strip() for p in args.precisions.split(",") if p.strip()]
    for p in precisions:
        if p not in PRECISIONS:
            ap.error(f"unknown precision {p!r}; choose from {', '.join(PRECISIONS)}")

    ckpt_path = Path(checkpoint.resolve(args.model))
    stem = args.stem or ckpt_path.stem
    ckpt_sha = sha256(ckpt_path)
    print(f"checkpoint  {ckpt_path.name}  sha256:{ckpt_sha[:16]}...")

    model = load_model(args.model)
    model.eval()

    if args.images:
        files = sorted(
            p for p in args.images.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
        )[: args.n_probe]
        if not files:
            ap.error(f"no images found in {args.images}")
        images = load_images(files)
        print(f"verifying against {len(files)} image(s) from {args.images}")
    else:
        images = probe_images(args.n_probe)
        print(f"verifying against {args.n_probe} synthetic probe(s)")

    args.out.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    results = []

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        base = {
            "encoder": tmp / "encoder-fp32.onnx",
            "decoder": tmp / "decoder-fp32.onnx",
        }

        print("exporting fp32 graphs...")
        export_fp32(model.encoder, (images[:1],), base["encoder"],
                    ["image"], ["latents"])
        z0 = torch.zeros(1, LATENT_CHANNELS, LATENT_H, LATENT_W)
        export_fp32(model.decoder, (z0, torch.ones_like(z0)), base["decoder"],
                    ["latents", "weights"], ["image"])

        for precision in precisions:
            paths = {}
            for part in ("encoder", "decoder"):
                dst = args.out / f"{stem}-{part}-{precision}.onnx"
                if precision == "fp32":
                    shutil.copyfile(base[part], dst)
                elif precision == "fp16":
                    convert_fp16(base[part], dst)
                else:
                    convert_int8(base[part], dst)
                stamp_metadata(dst, {
                    "sstvae.source_checkpoint": ckpt_path.name,
                    "sstvae.source_sha256": ckpt_sha,
                    "sstvae.part": part,
                    "sstvae.precision": precision,
                    "sstvae.opset": OPSET,
                    "sstvae.torch_version": torch.__version__,
                })
                paths[part] = dst

            r = verify(model, images, paths, precision)
            results.append(r)
            written.extend(paths.values())
            flag = "ok" if r["ok"] else "FAIL"
            print(
                f"  {precision:>4}  {r['encoder_mb']:5.1f} + {r['decoder_mb']:4.1f} MB  "
                f"latent RMS {r['latent_rms']:.2e} "
                f"({r['latent_vs_channel_db']:5.1f} dB under channel)  "
                f"picture {r['onnx_psnr_db']:6.2f} dB "
                f"({r['quality_lost_db']:+.3f} vs torch)  [{flag}]"
            )

    manifest = args.out / f"{stem}-onnx-manifest.json"
    manifest.write_text(json.dumps({
        "source_checkpoint": ckpt_path.name,
        "source_sha256": ckpt_sha,
        "opset": OPSET,
        "torch_version": torch.__version__,
        "channel_noise_rms": CHANNEL_NOISE_RMS,
        "artifacts": {
            p.name: {"sha256": sha256(p), "bytes": p.stat().st_size} for p in written
        },
        "verification": results,
    }, indent=2) + "\n")
    print(f"wrote {len(written)} artifact(s) + manifest to {args.out}/")

    failed = [r["precision"] for r in results if not r["ok"]]
    if failed:
        print(f"\nFAILED verification: {', '.join(failed)} -- nothing pushed",
              file=sys.stderr)
        return 1

    if args.push:
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(args.repo, exist_ok=True, repo_type="model")
        for p in [*written, manifest]:
            print(f"  uploading {p.name}")
            api.upload_file(path_or_fileobj=str(p), path_in_repo=p.name,
                            repo_id=args.repo,
                            commit_message=f"ONNX artifacts for {ckpt_path.name}")
        print(f"pushed to https://huggingface.co/{args.repo}")
    else:
        print("(not pushed; pass --push to upload)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Prototype: export the decoder's vector-Jacobian product as ONNX.

Per-image latent optimization (`scripts/latent_optim_prototype.py`)
needs `d(loss)/d(latents)` through the decoder. Doing that in the native
app would otherwise mean either an autodiff engine in C++ or a
training-capable onnxruntime build. Neither is necessary: the decoder's
weights are frozen, so the only gradient wanted is the one with respect
to its *input*, and that is a fixed chain of ops whose backward
formulas are themselves ordinary tensor arithmetic.

So this builds a module whose forward is `(z, weights, grad_recon) ->
(recon, grad_z)` -- the backward pass written out as forward ops -- and
exports *that*. The runtime side then needs one ORT session and an Adam
loop, both of which are arithmetic rather than architecture.

**The backward formulas are hand-derived on purpose.** Calling
`torch.autograd.grad` inside the module and exporting through it means
exporting a double-backward trace, which hits ops with no ONNX symbolic
registered. Written as forward ops, every one of them is a Conv,
ConvTranspose, or elementwise op the exporter already handles.

Checked three ways, since a wrong gradient still produces a plausible
picture (it just optimizes slowly or to the wrong place):
  1. every formula against `torch.autograd.grad` on the real decoder,
  2. the exported ONNX graph against the torch module,
  3. an end-to-end Adam loop driven *only* through onnxruntime, with
     numpy standing in for the C++ that would run it, which is the
     actual feasibility claim.

    python scripts/decoder_vjp_prototype.py --model ckpt.pt \\
        --onnx-out /tmp/decoder-grad.onnx --image photo.jpg
"""

import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from sstvae.codec import load_torch_model
from sstvae.config import LATENT_CHANNELS, LATENT_H, LATENT_W
from sstvae.images import IMG_H, IMG_W  # noqa: F401  (referenced in prose)
from sstvae.models.decoder_vjp import DecoderVJP

OPSET = 17


# The module itself now lives in `sstvae/models/decoder_vjp.py` so the
# publish script and this one cannot drift. What stays here is the
# verification, which is the part with an oracle.


# --- checks ---------------------------------------------------------------


def check_against_autograd(model, vjp, z, weights, target) -> None:
    z_ref = z.clone().requires_grad_(True)
    ref_recon = model.decoder(z_ref, weights)
    ref_mse = F.mse_loss(ref_recon, target)
    (ref_grad,) = torch.autograd.grad(ref_mse, z_ref)

    with torch.no_grad():
        recon, grad_z, mse = vjp(z, weights, target)

    r_err = (recon - ref_recon).abs().max().item()
    g_err = (grad_z - ref_grad).abs().max().item()
    g_scale = ref_grad.abs().max().item()
    print("  torch VJP vs autograd:")
    print(f"    recon   max|diff| = {r_err:.3e}")
    print(f"    mse     {mse.item():.8f} vs {ref_mse.item():.8f}")
    print(f"    grad_z  max|diff| = {g_err:.3e}  "
          f"(rel {g_err / max(g_scale, 1e-30):.3e}, scale {g_scale:.3e})")
    assert r_err < 1e-5, "reconstruction disagrees"
    assert g_err < 1e-5 * max(g_scale, 1e-9), "gradient disagrees with autograd"


def check_onnx(path: Path, vjp, z, weights, target) -> None:
    import onnxruntime as ort

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    o_recon, o_grad, o_mse = sess.run(
        ["recon", "grad_z", "mse"],
        {"z": z.numpy(), "weights": weights.numpy(), "target": target.numpy()})

    with torch.no_grad():
        t_recon, t_grad, t_mse = vjp(z, weights, target)

    r_err = np.abs(o_recon - t_recon.numpy()).max()
    g_err = np.abs(o_grad - t_grad.numpy()).max()
    print("  ONNX vs torch VJP:")
    print(f"    recon   max|diff| = {r_err:.3e}")
    print(f"    grad_z  max|diff| = {g_err:.3e}")
    print(f"    mse     {float(o_mse):.8f} vs {t_mse.item():.8f}")
    assert r_err < 1e-5 and g_err < 1e-8, "ONNX graph disagrees with torch"


def optimize_through_onnx(path: Path, z0: np.ndarray, weights: np.ndarray,
                          target: np.ndarray, lr: float, active_channels: int,
                          *, max_steps: int = 1000, time_budget_s: float = 20.0,
                          patience: int = 10, min_rel_gain: float = 2e-3,
                          verbose: bool = True) -> tuple[np.ndarray, list, str]:
    """The whole optimization loop with no torch: one ORT session per
    step, plus Adam in numpy. **This is the part that would be C++** --
    everything here is elementwise arithmetic over a fixed-size buffer
    and one `Run()`, with no autodiff machinery anywhere.

    Stops on whichever comes first: a plateau (no `min_rel_gain`
    relative improvement on the best loss within `patience` steps), the
    wall-clock budget, or `max_steps`. **Not a fixed step count** -- per
    step cost varies by an order of magnitude across the machines this
    would ship to, so a count that is a few seconds on a desktop is
    minutes on a small board. All three bounds are kept: the plateau
    test is the one that should normally fire, the budget is what makes
    the feature safe to run inside a transmit workflow, and `max_steps`
    only backstops a loss that never plateaus.

    Returns the *best* iterate, not the last. The loss reported at a
    step is the one for the latents that went into it, so the final
    update is always unmeasured -- returning it would occasionally ship
    a step past the minimum.
    """
    import onnxruntime as ort

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    # Bind by position, not by name: the prototype export calls the
    # first input `z` and the published artifact calls it `latents`
    # (matching the decoder's own convention). Positions are the part
    # that is actually contractual, and it is what the C++ caller will
    # do too.
    in_names = [i.name for i in sess.get_inputs()]
    out_names = [o.name for o in sess.get_outputs()]

    z = z0.copy()
    m = np.zeros_like(z)
    v = np.zeros_like(z)
    b1, b2, eps = 0.9, 0.999, 1e-8
    history: list[float] = []

    best_mse, best_z, best_step = math.inf, z.copy(), 0
    started = time.perf_counter()
    stop = f"max_steps ({max_steps})"

    for step in range(1, max_steps + 1):
        # One call gives the picture, the loss and the gradient.
        _, grad_z, mse = sess.run(
            out_names, dict(zip(in_names, (z, weights, target))))
        mse = float(mse)
        history.append(mse)

        if mse < best_mse * (1 - min_rel_gain):
            best_step = step
        if mse < best_mse:
            best_mse, best_z = mse, z.copy()

        if verbose and (step == 1 or step % 10 == 0):
            print(f"    step {step:4d}  mse={mse:.6f}  "
                  f"psnr={-10 * math.log10(max(mse, 1e-12)):.2f} dB  "
                  f"[{time.perf_counter() - started:.1f}s]")

        if step - best_step >= patience:
            stop = f"plateau (no {min_rel_gain:.1%} gain in {patience} steps)"
            break
        if time.perf_counter() - started >= time_budget_s:
            stop = f"time budget ({time_budget_s:.0f}s)"
            break

        m = b1 * m + (1 - b1) * grad_z
        v = b2 * v + (1 - b2) * grad_z * grad_z
        z = z - lr * (m / (1 - b1 ** step)) / (
            np.sqrt(v / (1 - b2 ** step)) + eps)

        active = z[:, :active_channels]
        rms = np.sqrt((active * active).mean(axis=(1, 2, 3), keepdims=True))
        z = (z / np.maximum(rms, 1e-6)).astype(np.float32)

    if verbose:
        print(f"    stopped: {stop} after {len(history)} steps, "
              f"{time.perf_counter() - started:.1f}s; best at step {best_step}")
    return best_z, history, stop


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=None, help="checkpoint .pt")
    ap.add_argument("--onnx-out", default="/tmp/decoder-grad.onnx")
    ap.add_argument("--image", default=None,
                    help="optional: run the ORT-only optimization loop "
                         "against this image")
    ap.add_argument("--max-steps", type=int, default=1000)
    ap.add_argument("--time-budget", type=float, default=20.0,
                    help="seconds; the loop stops on whichever of "
                         "plateau/budget/max-steps comes first")
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--min-rel-gain", type=float, default=2e-3)
    ap.add_argument("--lr", type=float, default=0.02)
    args = ap.parse_args()

    torch.manual_seed(0)
    model = load_torch_model(args.model)
    vjp = DecoderVJP(model.decoder).eval()

    z = torch.randn(1, LATENT_CHANNELS, LATENT_H, LATENT_W)
    z = z / z.flatten(1).pow(2).mean(dim=1).sqrt().view(-1, 1, 1, 1)
    weights = torch.ones_like(z)
    target = torch.rand(1, 3, IMG_H, IMG_W)

    print("== 1. hand-derived backward vs torch autograd ==")
    check_against_autograd(model, vjp, z, weights, target)

    # A truncated mode exercises the weight mask, which is also the
    # chain-rule factor on grad_z -- a mask applied to the forward pass
    # but forgotten in the backward would pass the check above.
    print("\n  same, mode A (one group, rest masked off):")
    w_a = torch.zeros_like(z)
    w_a[:, :44] = 1.0
    check_against_autograd(model, vjp, z * w_a, w_a, target)

    print("\n== 2. ONNX export ==")
    out = Path(args.onnx_out)
    torch.onnx.export(
        vjp, (z, weights, target), str(out),
        input_names=["z", "weights", "target"],
        output_names=["recon", "grad_z", "mse"],
        opset_version=OPSET, dynamo=True, external_data=False,
    )
    print(f"  wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")
    check_onnx(out, vjp, z, weights, target)

    if args.image:
        from PIL import Image

        from sstvae.images import fit_image, image_to_array

        print("\n== 3. optimization driven only by onnxruntime + numpy ==")
        arr = image_to_array(fit_image(Image.open(args.image)))
        target = arr[None].astype(np.float32)
        with torch.no_grad():
            z0 = model.encoder(torch.from_numpy(target)).numpy()
            base = model.decoder(
                torch.from_numpy(z0),
                torch.ones(1, LATENT_CHANNELS, LATENT_H, LATENT_W)).numpy()
        base_mse = float(((base - target) ** 2).mean())
        print(f"    encoder baseline mse={base_mse:.6f}  "
              f"psnr={-10 * math.log10(base_mse):.2f} dB")
        _, hist, _ = optimize_through_onnx(
            out, z0, np.ones_like(z0), target, args.lr, LATENT_CHANNELS,
            max_steps=args.max_steps, time_budget_s=args.time_budget,
            patience=args.patience, min_rel_gain=args.min_rel_gain)
        print(f"    gain: "
              f"{-10 * math.log10(min(hist)) + 10 * math.log10(base_mse):+.2f} dB")


if __name__ == "__main__":
    main()

"""The codec boundary: an accumulator's (S, W) -> the decoder's two
planes -> a picture (design 5, spec section 8 "Decoder input").

SSTVAE's decoder takes latents and per-latent weights in [0, 1], weight
0 meaning erased (`codec.decode`). Build step 1 measured what it wants
from an accumulator (`measure/decoder-measurements.md`):

- **latent = g(W) * S / W** with g(W) = 0.785 sqrt(W / (W + 0.39)) from
  each latent's own W -- the shrinkage SSTVAE's own modem delivers on
  AWGN, which the decoder was trained on. Feeding the unbiased S/W
  costs 0.7 dB of PSNR everywhere.
- **weight = min(sqrt(W / W~), 1)** with *one* W~ per picture: the
  largest of the groups' median W over their 50,600 sent latents.
  Normalising each group to its own median instead makes a weak
  refinement group cost up to 2.2 dB.
- Never-sent latents (2,200 per group) and groups without data get
  weight 0 (and latent 0).

Only this module, `picture.py` (for the codec ID) and `qrss_encode.py`
touch the codec. No torch: `sstvae.codec` runs on onnxruntime.
"""

from __future__ import annotations

import os

import numpy as np

from . import picture

G_MAX = 0.785                    # g(W) -> 0.785 as W -> infinity
G_KNEE = 0.39                    # g(W) = G_MAX sqrt(W / (W + G_KNEE))
N_GROUPS = picture.N_GROUPS
GROUP_LATENTS = picture.GROUP_LATENTS


def g_shrink(W) -> np.ndarray:
    """Decoder input gain 0.785 sqrt(W / (W + 0.39)), from each latent's own W."""
    W = np.maximum(np.asarray(W, dtype=np.float64), 0.0)
    return G_MAX * np.sqrt(W / (W + G_KNEE))


def reference_w(W, mode: int) -> float:
    """W~: the largest median W over the sent latents of the groups with data.

    0.0 when no group has data.
    """
    W = np.asarray(W, dtype=np.float64).reshape(N_GROUPS, GROUP_LATENTS)
    meds = [np.median(W[g, picture.air_to_canonical(g)])
            for g in range(int(mode) + 1) if np.any(W[g] > 0)]
    return float(max(meds)) if meds else 0.0


def decoder_planes(S, W, mode: int) -> tuple[np.ndarray, np.ndarray]:
    """(latents, weights), float32[158400] each, canonical flat order.

    S, W: an accumulator's float32[3, 52800]. Group g sits at
    [g*52800, (g+1)*52800), each group (channel, h, w) C-order, which is
    `sstvae.latents.flat_to_latents`'s layout. Groups above the mode and
    groups with no data are zero with weight 0, as are never-sent
    latents and any latent with W = 0. If W~ is 0 (a group whose
    median sent latent is unheard -- under half a pass), every heard
    latent gets weight 1.
    """
    mode = int(mode)
    if not 0 <= mode < N_GROUPS:
        raise ValueError(f"mode must be 0..{N_GROUPS - 1}, not {mode}")
    S = np.asarray(S, dtype=np.float64).reshape(N_GROUPS, GROUP_LATENTS)
    W = np.asarray(W, dtype=np.float64).reshape(N_GROUPS, GROUP_LATENTS)
    if np.any(W < 0) or not np.all(np.isfinite(W)) or not np.all(np.isfinite(S)):
        raise ValueError("W must be finite and non-negative, S finite")
    sent = np.zeros((N_GROUPS, GROUP_LATENTS), dtype=bool)
    for g in range(mode + 1):
        if np.any(W[g] > 0):
            sent[g, picture.air_to_canonical(g)] = True
    heard = sent & (W > 0)
    lat = np.zeros_like(W)
    lat[heard] = g_shrink(W[heard]) * S[heard] / W[heard]
    w_ref = reference_w(W, mode)
    wt = np.zeros_like(W)
    if w_ref > 0:
        wt[heard] = np.minimum(np.sqrt(W[heard] / w_ref), 1.0)
    else:
        wt[heard] = 1.0
    return lat.astype(np.float32).ravel(), wt.astype(np.float32).ravel()


def render(codec, S, W, mode: int):
    """The accumulated picture: codec.decode(*decoder_planes(S, W, mode)) -> PIL image."""
    lat, wt = decoder_planes(S, W, mode)
    return codec.decode(lat, wt)


def render_accumulator(codec, acc):
    """`render` of a `store.Accumulator`."""
    return render(codec, acc.S, acc.W, acc.mode)


def model_dir() -> str | None:
    """$QRSSTVAE_MODEL_DIR, else None (the published artifacts via the HF cache)."""
    return os.environ.get("QRSSTVAE_MODEL_DIR") or None


def load(precision: str = "fp32", path: str | None = None):
    """The decoder-side codec: `codec.load_codec(model_dir(), precision=...)`.

    `path` overrides the model directory (a CLI's --model).
    """
    from sstvae import codec

    return codec.load_codec(path if path is not None else model_dir(),
                            precision=precision)

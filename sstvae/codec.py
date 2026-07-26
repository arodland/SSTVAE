"""Loading the model and turning latents back into a picture.

The encoder/decoder network *is* the codec, so every tool -- the CLI
scripts, the live listener, the GUI -- needs these three things. They
used to live in `sstvae_encode.py` / `sstvae_decode.py` at the top
level, which meant anything inside the package that wanted them had to
import a *script*. They live here now; the scripts re-export them so
their command lines and any existing imports are unchanged.
"""

import numpy as np
import torch
from PIL import Image

from . import checkpoint
from .config import MODES
from .models import SSTVAE

MODEL_HELP = (
    "checkpoint.pt; defaults to the published checkpoint, downloaded and "
    "cached on first use"
)


def load_model(path: str | None = None) -> SSTVAE:
    """`path` may be None, in which case the published checkpoint is used.

    Always loads onto the CPU: the encoder/decoder passes are a few
    milliseconds on one 640x480 image, so a GPU buys nothing and would
    drag ROCm/CUDA initialization into short-lived CLI runs and into the
    GUI process.
    """
    ckpt = torch.load(checkpoint.resolve(path), map_location="cpu")
    model = SSTVAE(width=ckpt.get("width", 128))
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def reconstruct(model: SSTVAE, latents: np.ndarray, weights: np.ndarray) -> Image.Image:
    """Full-length (mode C sized) latent/weight vectors -> PIL image."""
    z = SSTVAE.flat_to_latents(torch.from_numpy(latents).float()[None])
    w = SSTVAE.flat_to_latents(torch.from_numpy(weights).float()[None])
    with torch.no_grad():
        img = model.decoder(z * (w > 0), w)[0]
    arr = (img.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
    return Image.fromarray(arr)


def pad_to_full(vec: np.ndarray, fill: float = 0.0) -> np.ndarray:
    """Extend a mode A/B latent (or weight) vector to mode C's length.

    The modes are nested -- mode A's latents are a prefix of mode C's --
    so a shorter mode is just a full-length vector whose tail never
    arrived, which is exactly what weight 0 means to the decoder.
    """
    full = MODES["C"].n_latents
    out = np.full(full, fill)
    out[: len(vec)] = vec
    return out

"""Framing pictures for the codec, and finding a font to draw on them.

Split out of `data.py` so the receive and transmit paths don't have to
import the training dataset machinery. `data.py` pulls in torchvision
and `torch.utils.data` to build its augmentation pipeline; everything
here is PIL plus a single array conversion, and a station that only
sends and receives pictures has no use for the rest.

`data.py` imports from this module, so training keeps working unchanged
and there is one definition of the target geometry rather than two.

**This module must import without torch.** The codec runs on ONNX now
(`docs/onnx.md`), so the send path is numpy end to end; an unconditional
`import torch` here would drag 336 MB back into every station that only
wants to send a picture. The `*_tensor` functions still exist for
training and import torch lazily -- use `image_to_array` / `load_image`
everywhere else.
"""

import os
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image, ImageFont

if TYPE_CHECKING:  # pragma: no cover - typing only, never imported at runtime
    import torch

# Target resolution. The latent grid (40x30) is fixed by the modem's
# capacity; at x16 downsampling that means 640x480 images. Images as
# small as MIN_W x MIN_H are accepted (and upscaled) to keep parity
# with classic 320x240 SSTV sources.
IMG_W, IMG_H = 640, 480
MIN_W, MIN_H = 320, 240

FONT_CANDIDATES = (
    "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/TTF/DejaVuSans.ttf",
    "/usr/share/fonts/TTF/DejaVuSerif.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/liberation/LiberationSerif-Regular.ttf",
    "/usr/share/fonts/liberation/LiberationMono-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
)
AVAILABLE_FONTS = tuple(p for p in FONT_CANDIDATES if os.path.exists(p))


@lru_cache(maxsize=128)
def font(size: int, idx: int = 0):
    """A scalable font at `size`. Falls back to Pillow's built-in scalable
    default, so this works in a bare container with no font packages."""
    if AVAILABLE_FONTS:
        return ImageFont.truetype(AVAILABLE_FONTS[idx % len(AVAILABLE_FONTS)], size)
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1 has a bitmap-only default
        return ImageFont.load_default()


def fit_image(img: Image.Image) -> Image.Image:
    """Any image -> exactly IMG_W x IMG_H RGB, by scaling to cover the
    target and centre-cropping (deterministic, aspect-preserving).

    Used by callers that already hold an image in memory -- the GUI,
    which composes text and insets onto the picture before transmitting
    -- so they go through the same framing as a file loaded from disk
    instead of a subtly different resize.
    """
    if img.mode != "RGB":
        img = img.convert("RGB")
    if img.size == (IMG_W, IMG_H):
        return img
    scale = max(IMG_W / img.width, IMG_H / img.height)
    img = img.resize(
        (round(img.width * scale), round(img.height * scale)), Image.LANCZOS
    )
    left = (img.width - IMG_W) // 2
    top = (img.height - IMG_H) // 2
    return img.crop((left, top, left + IMG_W, top + IMG_H))


def image_to_array(img: Image.Image) -> np.ndarray:
    """IMG_W x IMG_H RGB image -> (3, IMG_H, IMG_W) float32 in [0,1]."""
    return np.array(img, dtype=np.float32).transpose(2, 0, 1) / 255.0


def load_image(path: str | Path) -> np.ndarray:
    """Open any PIL-readable image -> (3, IMG_H, IMG_W) float32 in [0,1].

    Deterministic: cover-resize then centre-crop. For the augmented
    training variant see `sstvae.data.load_image`.
    """
    return image_to_array(fit_image(Image.open(path)))


def image_to_tensor(img: Image.Image) -> "torch.Tensor":
    """As `image_to_array`, but a torch tensor. **Training only.**

    Imports torch on call rather than at module import, so the send and
    receive paths -- which use `image_to_array` -- stay torch-free. See
    this module's docstring.
    """
    import torch

    return torch.from_numpy(image_to_array(img))

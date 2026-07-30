"""Training data: a folder of images, or a synthetic set for smoke tests."""

import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import Dataset
from torchvision import transforms as T

# The geometry and the font search live in `images.py`, which the
# receive/transmit paths import without dragging torchvision and the
# dataset machinery in with them. Re-exported here so training code and
# existing imports are unchanged.
from .images import (  # noqa: F401
    AVAILABLE_FONTS as _AVAILABLE_FONTS,
    IMG_H,
    IMG_W,
    MIN_H,
    MIN_W,
    fit_image,
    font as _font,
    image_to_tensor,
)

# Train-only augmentation: mild zoom/pan (scale relative to the source
# image's own area, so small MIN_W x MIN_H sources aren't cropped much
# further before upscaling), a coin-flip mirror (fine for a pixel
# reconstruction loss -- no OCR/semantic task cares that text ends up
# mirrored), and mild color jitter. Never applied to validation data.
_RANDOM_CROP = T.RandomResizedCrop(
    (IMG_H, IMG_W), scale=(0.8, 1.0), ratio=(IMG_W / IMG_H, IMG_W / IMG_H)
)
_COLOR_JITTER = T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.02)
_AUGMENT = T.Compose([_RANDOM_CROP, T.RandomHorizontalFlip(p=0.5), _COLOR_JITTER])

# --- burned-in text --------------------------------------------------------
# Real SSTV pictures nearly always carry burned-in text, which is
# high-contrast high-frequency content unlike anything in a natural-photo
# corpus. Measured on the epoch-126 checkpoint, an overlay cost ~1.1 dB
# PSNR against the same photo without one — and PSNR understates it,
# since ringing around glyph edges destroys legibility well before it
# moves MSE much. Flat illustration content measured *easier* than
# photos, so text is the content gap actually worth training on.
#
# Deliberately unstructured — random strings, sizes, positions, colors —
# rather than realistic callsign/grid/frequency layouts. Training on a
# fixed station-text template would let the decoder learn the template
# and hallucinate plausible-looking glyphs instead of faithfully coding
# whatever text is present, and real-world overlays vary far more than
# any template would. Drawn after the flip so text is never mirrored,
# and after the jitter so it stays the crisp overlay a station burns in.
TEXT_OVERLAY_P = 0.5

_TEXT_CHARS = (
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz"
    "0123456789"
    "0123456789"  # digits weighted up: overlays are numeral-heavy
    "/-.:# "
)


def draw_random_text(img: Image.Image, rng: random.Random | None = None) -> Image.Image:
    """Burn one to three blocks of random text into a copy of `img`."""
    rng = rng or random.Random()
    img = img.copy()
    d = ImageDraw.Draw(img, "RGBA")
    w, h = img.size

    for _ in range(rng.randint(1, 3)):
        text = "".join(
            rng.choice(_TEXT_CHARS) for _ in range(rng.randint(2, 22))
        ).strip() or "0"
        font = _font(rng.randint(11, 72), rng.randrange(64))
        fill = (rng.randrange(256), rng.randrange(256), rng.randrange(256))
        # Anchor anywhere, including slightly off-edge so partly-clipped
        # text is seen too.
        x, y = rng.randint(-30, max(-29, w - 20)), rng.randint(-15, max(-14, h - 20))

        roll = rng.random()
        if roll < 0.3:
            # Translucent backing panel.
            pad = rng.randint(3, 14)
            b = d.textbbox((x, y), text, font=font)
            d.rectangle(
                [b[0] - pad, b[1] - pad, b[2] + pad, b[3] + pad],
                fill=(rng.randrange(256), rng.randrange(256), rng.randrange(256),
                      rng.randint(120, 255)),
            )
            d.text((x, y), text, font=font, fill=fill)
        elif roll < 0.85:
            # Contrasting outline, so the glyphs stay legible over any
            # background rather than vanishing into it.
            lum = 0.299 * fill[0] + 0.587 * fill[1] + 0.114 * fill[2]
            stroke = (0, 0, 0) if lum > 110 else (255, 255, 255)
            d.text((x, y), text, font=font, fill=fill,
                   stroke_width=rng.randint(1, 3), stroke_fill=stroke)
        else:
            d.text((x, y), text, font=font, fill=fill)
    return img


def overlay_text_batch(imgs: torch.Tensor, seed: int = 0) -> torch.Tensor:
    """(B,3,H,W) in [0,1] -> same with deterministic random text burned in.

    Seeded so the overlay is identical every call, making the resulting
    metric comparable across epochs and runs. For evaluation only —
    training uses the unseeded path via `_augment_image`.
    """
    out = []
    for i, t in enumerate(imgs):
        pil = Image.fromarray(
            (t.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        )
        pil = draw_random_text(pil, random.Random(seed * 10007 + i))
        out.append(
            torch.from_numpy(np.array(pil)).permute(2, 0, 1).float().div_(255.0)
        )
    return torch.stack(out).to(imgs.device)


def _augment_image(img: Image.Image, rng: random.Random | None = None) -> Image.Image:
    img = _AUGMENT(img)
    rng = rng or random.Random()
    if rng.random() < TEXT_OVERLAY_P:
        img = draw_random_text(img, rng)
    return img


def load_image(path: str | Path, augment: bool = False) -> torch.Tensor:
    """Open any PIL-readable image -> (3, IMG_H, IMG_W) float in [0,1].

    Without augmentation: resizes to cover the target then center-crops,
    preserving aspect (deterministic -- use for eval/validation).
    With augmentation: random zoom/pan crop + hflip + color jitter, then
    burned-in random text with probability TEXT_OVERLAY_P (training only).
    """
    img = Image.open(path).convert("RGB")
    img = _augment_image(img) if augment else fit_image(img)
    return image_to_tensor(img)


class FolderDataset(Dataset):
    EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

    def __init__(self, root: str | Path, augment: bool = False):
        self.paths = sorted(
            p for p in Path(root).rglob("*") if p.suffix.lower() in self.EXTS
        )
        if not self.paths:
            raise FileNotFoundError(f"no images under {root}")
        self.augment = augment

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return load_image(self.paths[i], augment=self.augment)


class HFHubDataset(Dataset):
    """Images from a Hub dataset repo (e.g. arodland/coco320-sstvae).

    Expects an `image` column already sized IMG_W x IMG_H (or close;
    with augment=True the random-crop transform re-derives its own
    zoom/pan/flip/jitter view instead of just resizing to fit).
    """

    def __init__(self, repo: str, split: str = "train", augment: bool = False):
        from datasets import load_dataset

        self.ds = load_dataset(repo, split=split)
        self.augment = augment

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        img = self.ds[i]["image"].convert("RGB")
        if self.augment:
            img = _augment_image(img)
        elif img.size != (IMG_W, IMG_H):
            img = img.resize((IMG_W, IMG_H), Image.LANCZOS)
        return torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0


class NonPhotoDataset(Dataset):
    """Procedural operator-content images from `sstvae/nonphoto.py`:
    test cards, callsign cards, text blocks, line art, gradients, charts.

    Mixed into photographic training via `--nonphoto-frac` — the classes
    measured 3–7 dB behind COCO at fp32 on a photo-only model
    (docs/todo.md "Non-photographic content"). Generated on the fly, so
    `n` is a knob, not a directory size; deterministic per (index, salt),
    with the salt keeping train/val/eval splits disjoint by construction.

    Deliberately unaugmented: no mirror (mirrored callsigns), no color
    jitter (a test card's saturated primaries *are* the content), and
    the generators already randomize geometry per index.
    """

    def __init__(self, n: int, salt: str = "train"):
        from . import nonphoto

        self._gen = nonphoto.generate_index
        self.n = n
        self.salt = salt

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        img = self._gen(i, salt=self.salt)
        return torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0


class SyntheticDataset(Dataset):
    """Procedural gradients/shapes/noise — pipeline smoke tests only."""

    def __init__(self, n: int = 1024, seed: int = 0):
        self.n = n
        self.seed = seed

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        rng = np.random.default_rng(self.seed * 100003 + i)
        y, x = np.mgrid[0:IMG_H, 0:IMG_W].astype(np.float32)
        img = np.zeros((3, IMG_H, IMG_W), dtype=np.float32)
        for ch in range(3):
            gx, gy = rng.uniform(-1, 1, 2)
            img[ch] = 0.5 + 0.25 * (gx * x / IMG_W + gy * y / IMG_H)
        for _ in range(rng.integers(3, 9)):
            cx, cy = rng.uniform(0, IMG_W), rng.uniform(0, IMG_H)
            r = rng.uniform(10, 80)
            color = rng.uniform(0, 1, 3).astype(np.float32)
            mask = ((x - cx) ** 2 + (y - cy) ** 2) < r**2
            img[:, mask] = color[:, None]
        img += rng.normal(0, 0.02, img.shape).astype(np.float32)
        return torch.from_numpy(np.clip(img, 0, 1))

"""Training data: a folder of images, or a synthetic set for smoke tests."""

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

# Target resolution. The latent grid (40x30) is fixed by the modem's
# capacity; at x16 downsampling that means 640x480 images. Images as
# small as MIN_W x MIN_H are accepted (and upscaled) to keep parity
# with classic 320x240 SSTV sources.
IMG_W, IMG_H = 640, 480
MIN_W, MIN_H = 320, 240

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


def load_image(path: str | Path, augment: bool = False) -> torch.Tensor:
    """Open any PIL-readable image -> (3, IMG_H, IMG_W) float in [0,1].

    Without augmentation: resizes to cover the target then center-crops,
    preserving aspect (deterministic -- use for eval/validation).
    With augmentation: random zoom/pan crop + hflip + color jitter (use
    for training only).
    """
    img = Image.open(path).convert("RGB")
    if augment:
        img = _AUGMENT(img)
    else:
        scale = max(IMG_W / img.width, IMG_H / img.height)
        img = img.resize(
            (round(img.width * scale), round(img.height * scale)), Image.LANCZOS
        )
        left = (img.width - IMG_W) // 2
        top = (img.height - IMG_H) // 2
        img = img.crop((left, top, left + IMG_W, top + IMG_H))
    return torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0


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
            img = _AUGMENT(img)
        elif img.size != (IMG_W, IMG_H):
            img = img.resize((IMG_W, IMG_H), Image.LANCZOS)
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

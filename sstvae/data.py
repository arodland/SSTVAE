"""Training data: a folder of images, or a synthetic set for smoke tests."""

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

IMG_W, IMG_H = 320, 240


def load_image(path: str | Path) -> torch.Tensor:
    """Open any PIL-readable image -> (3, IMG_H, IMG_W) float in [0,1].

    Resizes to cover the target then center-crops, preserving aspect.
    """
    img = Image.open(path).convert("RGB")
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

    def __init__(self, root: str | Path):
        self.paths = sorted(
            p for p in Path(root).rglob("*") if p.suffix.lower() in self.EXTS
        )
        if not self.paths:
            raise FileNotFoundError(f"no images under {root}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return load_image(self.paths[i])


class HFHubDataset(Dataset):
    """Images from a Hub dataset repo (e.g. arodland/coco320-sstvae).

    Expects an `image` column already sized IMG_W x IMG_H.
    """

    def __init__(self, repo: str, split: str = "train"):
        from datasets import load_dataset

        self.ds = load_dataset(repo, split=split)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        img = self.ds[i]["image"].convert("RGB")
        if img.size != (IMG_W, IMG_H):
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

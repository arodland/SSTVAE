#!/usr/bin/env python3
"""Build the coco320 HF dataset with proper train/validation splits and
push it to the Hub.

The prepped folder mixes train2017 and val2017 images; split membership
is recovered from the original zips' file lists.

    python scripts/build_hf_dataset.py data/coco320 \
        --val-zip data/val2017.zip --repo arodland/coco320-sstvae
"""

import argparse
import zipfile
from pathlib import Path

from datasets import Dataset, DatasetDict, Features, Image

CARD = """\
---
license: cc-by-4.0
task_categories: [image-to-image]
pretty_name: COCO 2017 at 320x240 for SSTVAE
---

# coco320-sstvae

COCO 2017 images resized (cover) and center-cropped to 320x240, JPEG
quality 92. Built for training the SSTVAE radio autoencoder
(image-over-HF-radio). Splits follow the original COCO train2017 /
val2017 membership. Images smaller than 320x240 were dropped.

Original images: https://cocodataset.org — see COCO terms of use;
image copyrights belong to their Flickr owners.
"""


def zip_stems(zip_path: Path) -> set[str]:
    with zipfile.ZipFile(zip_path) as zf:
        return {Path(n).stem for n in zf.namelist() if n.lower().endswith(".jpg")}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("folder", help="prepped 320x240 image folder")
    ap.add_argument("--val-zip", required=True, help="original val2017.zip")
    ap.add_argument("--repo", required=True)
    ap.add_argument("--public", action="store_true")
    args = ap.parse_args()

    val_stems = zip_stems(Path(args.val_zip))
    paths = sorted(Path(args.folder).glob("*.jpg"))
    train_files = [str(p) for p in paths if p.stem not in val_stems]
    val_files = [str(p) for p in paths if p.stem in val_stems]
    print(f"train: {len(train_files)}, validation: {len(val_files)}")

    features = Features({"image": Image()})
    dd = DatasetDict(
        {
            "train": Dataset.from_dict({"image": train_files}, features=features),
            "validation": Dataset.from_dict({"image": val_files}, features=features),
        }
    )
    dd.push_to_hub(args.repo, private=not args.public)

    from huggingface_hub import HfApi

    HfApi().upload_file(
        path_or_fileobj=CARD.encode(),
        path_in_repo="README.md",
        repo_id=args.repo,
        repo_type="dataset",
    )
    print(f"pushed to https://huggingface.co/datasets/{args.repo}")


if __name__ == "__main__":
    main()

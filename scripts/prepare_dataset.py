#!/usr/bin/env python3
"""Prepare training images: extract from a zip (or walk a folder),
resize-and-center-crop to the target resolution (sstvae.data.IMG_W/H).

    python scripts/prepare_dataset.py data/val2017.zip data/coco640
    python scripts/prepare_dataset.py data/train2017.zip data/coco640

Images smaller than MIN_W x MIN_H (classic SSTV size) are skipped;
anything between MIN and target size is upscaled by the cover-resize.
Processing is multiprocess; re-runs skip already-written files.
"""

import argparse
import io
import sys
import zipfile
from multiprocessing import Pool
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sstvae.data import IMG_H, IMG_W, MIN_H, MIN_W

_zip_path: str | None = None
_zf: zipfile.ZipFile | None = None


def _process(job: tuple[str, str]) -> bool:
    global _zf
    name, out_path = job
    try:
        if _zip_path is not None:
            if _zf is None:
                _zf = zipfile.ZipFile(_zip_path)
            data = _zf.read(name)
            img = Image.open(io.BytesIO(data))
        else:
            img = Image.open(name)
        img = img.convert("RGB")
        # Accept anything classic-SSTV sized or larger; smaller-than-target
        # images get upscaled by the cover-resize below.
        if img.width < MIN_W or img.height < MIN_H:
            return False
        scale = max(IMG_W / img.width, IMG_H / img.height)
        img = img.resize(
            (round(img.width * scale), round(img.height * scale)), Image.LANCZOS
        )
        left = (img.width - IMG_W) // 2
        top = (img.height - IMG_H) // 2
        img = img.crop((left, top, left + IMG_W, top + IMG_H))
        img.save(out_path, quality=92)
        return True
    except Exception as e:
        print(f"skip {name}: {e}", file=sys.stderr)
        return False


def _init(zip_path: str | None):
    global _zip_path
    _zip_path = zip_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("source", help="zip file or image folder")
    ap.add_argument("outdir")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    src = Path(args.source)

    if src.suffix == ".zip":
        with zipfile.ZipFile(src) as zf:
            names = [
                n
                for n in zf.namelist()
                if n.lower().endswith((".jpg", ".jpeg", ".png"))
            ]
        zip_arg = str(src)
    else:
        names = [
            str(p)
            for p in src.rglob("*")
            if p.suffix.lower() in (".jpg", ".jpeg", ".png")
        ]
        zip_arg = None

    jobs = []
    for n in names:
        dest = out / (Path(n).stem + ".jpg")
        if not dest.exists():
            jobs.append((n, str(dest)))
    print(f"{len(names)} images, {len(jobs)} to process")

    done = 0
    with Pool(args.workers, initializer=_init, initargs=(zip_arg,)) as pool:
        for i, ok in enumerate(pool.imap_unordered(_process, jobs, chunksize=64)):
            done += ok
            if (i + 1) % 5000 == 0:
                print(f"{i + 1}/{len(jobs)} processed, {done} written")
    print(f"done: {done} written to {out}")


if __name__ == "__main__":
    main()

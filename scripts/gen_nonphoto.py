#!/usr/bin/env python3
"""Write a set of procedural non-photographic images to a directory.

    python scripts/gen_nonphoto.py --out /tmp/nonphoto --per-class 8

CLI front end for `sstvae/nonphoto.py`, which holds the generators (and
the reasoning). Deterministic per (class, index, salt); the default
salt "eval" is the measured evaluation set in docs/todo.md, and is
disjoint from the "train"/"val" salts `data.NonPhotoDataset` uses.

Filenames are `<class>_<i>.png`, so an evaluator can group by class
with a split on the last underscore.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sstvae import nonphoto  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--per-class", type=int, default=8)
    ap.add_argument("--salt", default="eval")
    ap.add_argument("--classes", nargs="*", default=list(nonphoto.CLASSES),
                    choices=list(nonphoto.CLASSES))
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    for cls in args.classes:
        for i in range(args.per_class):
            img = nonphoto.generate(cls, i, salt=args.salt)
            img.save(args.out / f"{cls}_{i:03d}.png")
    print(f"wrote {len(args.classes) * args.per_class} images to {args.out}")


if __name__ == "__main__":
    main()

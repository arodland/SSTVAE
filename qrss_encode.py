#!/usr/bin/env python3
"""Encode an image once into a stored QRSSTVAE picture (.qrsp).

    python qrss_encode.py photo.jpg pic.qrsp --mode A

The picture is encoded once and kept as the fp16 latents every pass will
send; repeats never re-encode (a re-encode would send a different picture
under the same ID). Prints the picture ID. The codec ID is read from the
model's `sstvae.source_sha256` metadata; a model without it is refused
unless --codec-id is given, because receivers must decode with the same
checkpoint.
"""

import argparse
import os

from sstvae.checkpoint import PRECISIONS, resolve_onnx
from sstvae.codec import MODEL_HELP, load_codec
from sstvae.images import load_image
from sstvae.qrss import picture


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("image")
    ap.add_argument("output", help="stored picture (.qrsp)")
    ap.add_argument("--model", default=None, help=MODEL_HELP)
    ap.add_argument("--precision", choices=PRECISIONS, default="fp16")
    ap.add_argument("--mode", choices=picture.MODE_NAMES, default="A",
                    help="A sends group 0, B groups 0-1, C groups 0-2 (one per slot)")
    ap.add_argument("--codec-id", default=None, metavar="HEX",
                    help="16-bit codec ID, for a model without source metadata")
    args = ap.parse_args()
    if not os.path.isfile(args.image):
        raise SystemExit(f"{args.image}: no such file")

    codec_id = None
    if args.codec_id is not None:
        codec_id = int(args.codec_id, 16)
        if not 0 <= codec_id < 1 << 16:
            raise SystemExit(f"--codec-id must be 16 bits, not {args.codec_id}")
    else:
        enc = resolve_onnx("encoder", args.model, args.precision)
        codec_id = picture.codec_id_from_onnx(enc, args.precision)
        if codec_id is None:
            raise SystemExit(f"{enc} carries no sstvae.source_sha256 metadata; pass "
                             "--codec-id HEX so receivers know which decoder to use")

    codec = load_codec(args.model, precision=args.precision)
    try:
        img = load_image(args.image)
    except (OSError, ValueError) as e:
        raise SystemExit(f"{args.image}: {e}") from None
    flat = codec.encode(img)
    mode = picture.MODE_NAMES.index(args.mode)
    sp = picture.StoredPicture.from_latents(flat, codec_id, mode)
    picture.save_qrsp(args.output, sp)
    print(f"wrote {args.output}: mode {args.mode} ({mode + 1} segment"
          f"{'s' if mode else ''}), codec {codec_id:04x}, picture ID {sp.id_hex}")


if __name__ == "__main__":
    main()

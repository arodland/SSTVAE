#!/usr/bin/env python3
"""Rasterize native/packaging/sstvae.svg into the shipped icon formats.

    tools/gen_icons.py

Three platforms want three containers and none of them wants an SVG:

    sstvae.ico          Windows -- linked into the .exe as a resource,
                        and used by the NSIS installer for its shortcuts
    sstvae.icns         macOS -- Contents/Resources, named by the
                        Info.plist's CFBundleIconFile
    icons/*.png         freedesktop -- hicolor/<size>x<size>/apps, which
                        is what a Linux menu, an AppImage's thumbnail and
                        a taskbar all read

Generated and committed, like config.hpp and the golden vectors, so that
a plain `cmake` build needs neither Python nor librsvg. Unlike those two
it is deliberately **not** a CI gate: the check would have to be a
byte-comparison of rasterized output, and librsvg's antialiasing is not
promised to be stable across versions, so the gate would fail on a
librsvg upgrade with no icon having changed. Re-run this by hand when
the SVG changes, and look at the result -- which is the only check that
means anything for an icon anyway.

Everything this writes is a derivative of the SVG, which is **licensed
artwork and not under the project's license** (see NOTICE). So each
output gets an SPDX sidecar written beside it here, rather than by hand:
adding a size later would otherwise put an unlabelled non-free file into
a repository whose root LICENSE says Artistic-2.0, and nothing would
notice.

Requires `rsvg-convert` (librsvg) and Pillow. Rasterizing every size
from the vector rather than downscaling one big PNG is the point: at 16
and 32 pixels a Lanczos reduction of a 1024px render is a grey blur,
where the renderer given a 16px viewport snaps the same shapes to the
pixel grid.
"""

from __future__ import annotations

import shutil
import struct
import subprocess
import sys
from io import BytesIO
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
SVG = ROOT / "native" / "packaging" / "sstvae.svg"
OUT = ROOT / "native" / "packaging"

# What each container needs. The union is rasterized once and shared.
PNG_SIZES = (16, 32, 48, 128, 256)
ICO_SIZES = (16, 24, 32, 48, 64, 128, 256)

# The icns element types, by the *pixel* size of their payload. Note that
# several sizes appear twice: `ic13` is "128 points at 2x" and `ic08` is
# "256 points at 1x", both 256 pixels, and macOS picks between them by
# the display's scale factor. Omitting the @2x forms is what makes an
# otherwise fine icon look soft on a Retina display.
ICNS_TYPES = (
    (b"icp4", 16),
    (b"icp5", 32),
    (b"ic11", 32),
    (b"ic12", 64),
    (b"ic07", 128),
    (b"ic13", 256),
    (b"ic08", 256),
    (b"ic14", 512),
    (b"ic09", 512),
    (b"ic10", 1024),
)


# The REUSE sidecar written beside every generated file. Kept in step
# with native/packaging/sstvae.svg.license, which is the hand-written one
# for the source artwork; if that changes, change this.
SIDECAR = """\
# The SSTVAE application icon is licensed artwork, NOT under the
# project's Artistic License 2.0. Rendered from
# native/packaging/sstvae.svg by tools/gen_icons.py; a derivative work,
# so the same restriction applies. See NOTICE at the repository root.
SPDX-FileCopyrightText: Licensed artwork, all rights reserved by its author
SPDX-License-Identifier: LicenseRef-SSTVAE-Branding
"""


def rasterize(size: int) -> Image.Image:
    """The SVG rendered at exactly `size` pixels square."""
    png = subprocess.run(
        ["rsvg-convert", "-w", str(size), "-h", str(size), str(SVG)],
        check=True,
        stdout=subprocess.PIPE,
    ).stdout
    image = Image.open(BytesIO(png)).convert("RGBA")
    if image.size != (size, size):  # librsvg honours -w/-h, but be sure
        raise SystemExit(f"gen_icons: rsvg-convert produced {image.size} for {size}")
    return image


def write_icns(path: Path, images: dict[int, Image.Image]) -> None:
    """Assemble an .icns by hand, one PNG payload per element.

    Pillow can write .icns, but only from a single image -- it upscales
    whatever it is given to 1024 and derives the rest, so the small
    elements come out as blurred reductions of a blur. The container
    itself is trivial: a header, then (4-byte type, big-endian length
    *including* the 8-byte header, payload) repeated.
    """
    elements: list[tuple[bytes, bytes]] = []
    for kind, size in ICNS_TYPES:
        buf = BytesIO()
        images[size].save(buf, "png")
        elements.append((kind, buf.getvalue()))

    # A table of contents is optional, but it is what lets a reader index
    # the file without walking it, and Finder writes one.
    toc = b"".join(kind + struct.pack(">I", len(data) + 8) for kind, data in elements)
    body = struct.pack(">I", len(toc) + 8) + toc
    body = b"TOC " + body
    for kind, data in elements:
        body += kind + struct.pack(">I", len(data) + 8) + data
    path.write_bytes(b"icns" + struct.pack(">I", len(body) + 8) + body)


def main() -> int:
    if not SVG.is_file():
        raise SystemExit(f"gen_icons: no source at {SVG}")
    if shutil.which("rsvg-convert") is None:
        raise SystemExit(
            "gen_icons: rsvg-convert not found (Debian/Ubuntu: librsvg2-bin, "
            "Arch: librsvg, macOS: brew install librsvg)"
        )

    sizes = sorted({*PNG_SIZES, *ICO_SIZES, *(s for _, s in ICNS_TYPES)})
    images = {size: rasterize(size) for size in sizes}

    png_dir = OUT / "icons"
    png_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for size in PNG_SIZES:
        path = png_dir / f"sstvae-{size}.png"
        images[size].save(path, "png", optimize=True)
        written.append(path)

    ico = OUT / "sstvae.ico"
    # `sizes` selects which frames to emit; `append_images` supplies the
    # natively rendered one for each, so nothing is downscaled here.
    images[max(ICO_SIZES)].save(
        ico,
        "ico",
        sizes=[(s, s) for s in ICO_SIZES],
        append_images=[images[s] for s in ICO_SIZES],
    )
    written.append(ico)

    icns = OUT / "sstvae.icns"
    write_icns(icns, images)
    written.append(icns)

    for path in written:
        sidecar = path.with_name(path.name + ".license")
        sidecar.write_text(SIDECAR)
        print(f"{path.relative_to(ROOT)} ({path.stat().st_size} bytes) + .license")
    return 0


if __name__ == "__main__":
    sys.exit(main())

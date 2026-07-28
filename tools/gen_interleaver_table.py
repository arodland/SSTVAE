#!/usr/bin/env python3
"""Emit `native/core/framing/interleaver_table.cpp` from the frozen data.

    tools/gen_interleaver_table.py            # write the table
    tools/gen_interleaver_table.py --check    # exit 1 if it would change

**Reads `sstvae/modem/interleaver_perms.npy`, not numpy's RNG.** That
file is the on-air format (see the note in `sstvae/modem/framing.py`);
this script only re-encodes it as C++ so the application can carry it in
`.rodata`. So a `--check` failure means the C++ and the frozen data have
diverged — an encoding bug, or someone editing the generated file — and
never that numpy changed its mind about what a permutation is.

That distinction is the whole point of the split. An earlier draft of
this script derived the table from `framing._PERMS`, which at the time
re-seeded numpy at import: a future numpy that changed its stream would
then have failed CI, and the obvious fix would have been to regenerate
the table and silently change what the radio transmits.

`std::uint16_t` because `GROUP_LATENTS` is 52,800, comfortably inside
16 bits. 3 x 50,600 x 2 = 304 KB of table against a ~75 MB application:
not worth a more compact encoding, and any such encoding would need
decoding code that could itself be wrong.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DEFAULT_OUT = ROOT / "native" / "core" / "framing" / "interleaver_table.cpp"
PER_LINE = 24


def render() -> str:
    from sstvae.config import (
        GROUP_LATENTS,
        LATENT_GROUPS,
        TRANSMIT_LATENTS_PER_GROUP,
    )
    from sstvae.modem import framing

    perms = np.load(framing._PERMS_PATH)
    if perms.shape != (LATENT_GROUPS, TRANSMIT_LATENTS_PER_GROUP):
        raise SystemExit(
            f"{framing._PERMS_PATH} has shape {perms.shape}, expected "
            f"({LATENT_GROUPS}, {TRANSMIT_LATENTS_PER_GROUP}) for this config.py"
        )
    if perms.min() < 0 or perms.max() >= GROUP_LATENTS:
        raise SystemExit("frozen permutation contains an out-of-range index")
    if GROUP_LATENTS > 0xFFFF:
        raise SystemExit("GROUP_LATENTS no longer fits in uint16; widen the "
                         "table type in interleaver_table.hpp too")

    out: list[str] = []
    w = out.append
    w("// GENERATED FILE -- DO NOT EDIT.\n")
    w("//\n")
    w("// Written by tools/gen_interleaver_table.py from\n")
    w("// sstvae/modem/interleaver_perms.npy, which is the frozen on-air\n")
    w("// interleave -- not from numpy's RNG. See sstvae/modem/framing.py\n")
    w("// for why that distinction matters. CI regenerates this file and\n")
    w("// fails if it differs from the frozen data.\n")
    w("//\n")
    w("// Each group's permutation is truncated to the transmittable budget;\n")
    w("// the entries beyond it are the latents that group permanently drops\n")
    w("// (weight 0), and are defined by their absence.\n")
    w("\n")
    w('#include "framing/interleaver_table.hpp"\n')
    w("\n")
    w("namespace sstvae::framing {\n")
    w("\n")
    w("// Flat rather than nested: one initializer list of "
      f"{perms.size} values\n")
    w("// compiles substantially faster than three nested ones, and the\n")
    w("// accessor in the header hides the indexing.\n")
    w("const std::uint16_t TX_PERMS_DATA[N_GROUPS * TX_PERM_LEN] = {\n")
    flat = perms.reshape(-1)
    for g in range(LATENT_GROUPS):
        w(f"    // group {g}\n")
        row = perms[g]
        for i in range(0, len(row), PER_LINE):
            w("    " + ",".join(str(int(v)) for v in row[i:i + PER_LINE]) + ",\n")
    w("};\n")
    w("\n")
    w("}  // namespace sstvae::framing\n")
    del flat
    return "".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--check", action="store_true",
                    help="exit 1 if the file is missing or out of date")
    args = ap.parse_args()

    text = render()
    if args.check:
        try:
            current = args.out.read_text(encoding="utf-8")
        except FileNotFoundError:
            print(f"{args.out} does not exist; run tools/gen_interleaver_table.py",
                  file=sys.stderr)
            return 1
        if current != text:
            print(f"{args.out} disagrees with the frozen interleaver data.\n"
                  "Re-run tools/gen_interleaver_table.py and commit the result.\n"
                  "Note this is an encoding check: it compares C++ against\n"
                  "sstvae/modem/interleaver_perms.npy, which is the format.",
                  file=sys.stderr)
            return 1
        print(f"{args.out.name} is up to date")
        return 0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    print(f"wrote {args.out} ({len(text) / 1024:.0f} KiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

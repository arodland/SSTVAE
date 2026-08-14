#!/usr/bin/env python3
"""Freeze the RNG-derived part of the on-air format, and audit it.

    tools/freeze_format_constants.py            # write the frozen data
    tools/freeze_format_constants.py --verify   # does numpy still agree?

## What this is for

The per-group interleaver permutations (`INTERLEAVER_SEED + g`) were
produced by seeding numpy, and they are **part of the on-air format** --
two stations must agree on them exactly or the picture is noise.

(The pilot sequence used to be here too, as a seeded QPSK draw. As of
PROTOCOL_VERSION 3 it is Zadoff-Chu -- a closed form over exact
integers, `config.PILOT_PHASE_NUM` -- so there is no generator whose
behaviour it could depend on and nothing here to audit. That is a
strictly better place to be than freezing a draw, and where any future
format constant should aim.)

Deriving them at import time makes numpy's PCG64, its bounded-integer
draw and its shuffle loop part of that format. numpy commits to stream
stability, but "commits to" is not "cannot change", and the failure mode
if it ever did is the worst kind: no error, just stations on different
numpy versions unable to decode each other.

So the values are written down, and the program reads what is written
down:

* `sstvae/modem/interleaver_perms.npy` -- 3 x 50,600 uint16, too large
  for a literal, committed beside the module that uses it.

## `--verify` is information, not a gate

It re-derives them from the seed with the *current* numpy and reports
whether they still match. **A mismatch does not mean the frozen data is
wrong.** It means numpy changed, and the correct response is to change
nothing: the format is what is frozen. Record the finding, and note that
anything which re-derives these values (an old checkout, a third-party
reimplementation) is now incompatible.

That is why this is a script you run rather than a test that fails.
Wiring it into CI would express exactly the wrong relationship -- it
would make numpy's current behaviour authoritative over the format, and
the obvious way to "fix" a red build would be to regenerate the frozen
data and silently change what the radio transmits.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PERMS_PATH = ROOT / "sstvae" / "modem" / "interleaver_perms.npy"


def derive_interleaver_perms(seed: int, group_latents: int, groups: int,
                             transmit_per_group: int) -> np.ndarray:
    """How interleaver_perms.npy was originally produced.

    Only the transmittable prefix is kept: the remainder of each
    permutation is the set of latents that group permanently drops, and
    it is defined by its absence (weight 0), so storing it would be
    storing something nothing reads.
    """
    return np.array(
        [
            np.random.default_rng(seed + g).permutation(group_latents)[
                :transmit_per_group
            ]
            for g in range(groups)
        ],
        dtype=np.uint16,
    )


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--verify", action="store_true",
                    help="re-derive from the seeds and report whether numpy "
                         "still agrees (informational -- see the docstring)")
    args = ap.parse_args()

    from sstvae.config import (
        GROUP_LATENTS,
        INTERLEAVER_SEED,
        LATENT_GROUPS,
        TRANSMIT_LATENTS_PER_GROUP,
    )

    if GROUP_LATENTS > np.iinfo(np.uint16).max:
        raise SystemExit("GROUP_LATENTS no longer fits in uint16; widen the "
                         "frozen permutation dtype and regenerate")

    derived_perms = derive_interleaver_perms(
        INTERLEAVER_SEED, GROUP_LATENTS, LATENT_GROUPS, TRANSMIT_LATENTS_PER_GROUP)

    if args.verify:
        ok = True
        if not PERMS_PATH.exists():
            ok = False
            print(f"interleaver:      {PERMS_PATH} is missing")
        else:
            frozen_perms = np.load(PERMS_PATH)
            if np.array_equal(frozen_perms, derived_perms):
                print(f"interleaver:      numpy {np.__version__} still agrees "
                      f"({frozen_perms.shape[0]} x {frozen_perms.shape[1]})")
            else:
                ok = False
                diffs = int(np.sum(frozen_perms != derived_perms))
                print(f"interleaver:      DIFFER under numpy {np.__version__} "
                      f"({diffs} of {frozen_perms.size} entries)")

        if not ok:
            print("\nnumpy no longer reproduces the frozen format constants.")
            print("This does NOT mean the frozen data is wrong, and it must")
            print("NOT be regenerated: those values are the on-air format and")
            print("changing them would break compatibility with every station")
            print("and every recording. What it means is that re-deriving them")
            print("from the seeds is no longer equivalent -- so anything that")
            print("still does (an old checkout, a third-party implementation)")
            print("is now incompatible with this one. Record the numpy version")
            print("at which it changed and move on.")
        # Deliberately 0 either way: this reports, it does not gate.
        return 0

    PERMS_PATH.parent.mkdir(parents=True, exist_ok=True)
    if PERMS_PATH.exists():
        existing = np.load(PERMS_PATH)
        if not np.array_equal(existing, derived_perms):
            raise SystemExit(
                f"{PERMS_PATH} already exists and differs from what this numpy "
                f"({np.__version__}) derives.\n"
                "Refusing to overwrite: the committed file is the on-air "
                "format, and replacing it would change what this program "
                "transmits. Run --verify to see the difference."
            )
        print(f"{PERMS_PATH.name} already frozen and unchanged")
        return 0

    np.save(PERMS_PATH, derived_perms, allow_pickle=False)
    print(f"wrote {PERMS_PATH} {derived_perms.shape} "
          f"({PERMS_PATH.stat().st_size / 1024:.0f} KiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

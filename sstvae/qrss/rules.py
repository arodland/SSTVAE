"""Receiver rules a command line can set back to their earlier form.

Not a format module: none of this changes what goes on the air.

Two receiver changes of 2026-10-12 each have a switch that restores the
rule they replaced, so a recording can be compared old against new and
a fix that misbehaves on the band can be turned off without a code
change:

- EDGE_NOISE: `frontend.noise_psd` takes its noise bins only where the
  front end is flat (FE_FLAT_DB), and where neighbours leave too few of
  those, from the filter's skirt as well, each bin compensated for it.
  Before, a signal below about 430 Hz or above about 2570 Hz took half
  its bins raw from that skirt, so its noise floor read low and its SNR
  high, by 9.6 dB at 2695 Hz on white noise.
- GATING: `receiver.detect` does not gate a whole-slot candidate in the
  own band of a signal it found on its preamble and no stronger in the
  whole-slot search than that signal's own line (a strong CE's ghosts
  about 12 Hz either side, the L copy of its carrier, a line in an L
  signal's data band), nor count it against max_candidates. Before,
  each was gated and then refused or dropped as a ghost, at up to a
  minute each on FULL.

qrss_listen.py and qrss_receive.py take --old-edge-noise and
--old-gating, qrss_decode.py --old-edge-noise only (its --em and --retro
passes search for no candidates); with both, and --just-ce, the receiver
is the one of 2026-10-11 bit for bit. The switches live in the
environment so that the listener's end-of-slot process, which is
spawned, follows its parent.
"""

from __future__ import annotations

import os

EDGE_NOISE = "QRSS_OLD_EDGE_NOISE"
GATING = "QRSS_OLD_GATING"
RULES = (EDGE_NOISE, GATING)


def old(rule: str) -> bool:
    """Whether `rule` is set back to its earlier form."""
    return os.environ.get(rule, "") not in ("", "0")


def set_old(rule: str, on: bool = True) -> None:
    """Set `rule` back to its earlier form (on) or to the current one (off),
    for this process and any it starts afterwards."""
    if rule not in RULES:
        raise ValueError(f"unknown rule {rule!r}: one of {', '.join(RULES)}")
    if on:
        os.environ[rule] = "1"
    else:
        os.environ.pop(rule, None)


def add_arguments(ap, which=RULES) -> None:
    """--old-edge-noise and --old-gating (those of `which`), on a command
    line's parser."""
    if EDGE_NOISE in which:
        ap.add_argument("--old-edge-noise", action="store_true",
                        help="take the noise floor near 300 and 2700 Hz as before 2026-10-12, "
                             "partly off the front end's filter skirt")
    if GATING in which:
        ap.add_argument("--old-gating", action="store_true",
                        help="verify every candidate as before 2026-10-12, including those "
                             "inside a signal found already")


def apply(args) -> None:
    """Set back the rules a command line asked for (`add_arguments`)."""
    if getattr(args, "old_edge_noise", False):
        set_old(EDGE_NOISE)
    if getattr(args, "old_gating", False):
        set_old(GATING)

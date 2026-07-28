#!/usr/bin/env python3
"""Emit `native/core/config.hpp` from `sstvae/config.py`.

CLAUDE.md: *"All waveform/latent numbers must agree through this
module."* Two hand-maintained copies of `config.py` would be the single
most likely source of a silent on-air incompatibility, and the failure
would be invisible in both test suites if both were edited consistently
but wrongly. So the C++ side does not get its own copy -- it gets a
generated one, and CI regenerates it and asserts the diff is empty.

    tools/gen_config_header.py            # write the header
    tools/gen_config_header.py --check    # exit 1 if it would change

Two things here are less obvious than the scalar copying:

**The pilot sequence is generated, not ported.** `ofdm.pilot_sequence()`
draws from `np.random.default_rng(PILOT_SEED)`, so reproducing it in C++
means reimplementing PCG64 *and* numpy's bounded-integer algorithm, and
then owning that forever as a compatibility surface. But the sequence is
a *format constant* -- 24 fixed QPSK phases that both ends must agree
on -- not an algorithm anyone needs to run. Emitting the values makes the
contract explicit and deletes the dependency on numpy's RNG internals.
The same argument applies to the interleaver permutations later, which
are far larger; those get their own generated artifact rather than
bloating this header.

**Derived values are emitted, not recomputed.** `M = FS // RS` could be
written as an expression in C++, but then a future edit to config.py that
changes the relationship silently leaves the C++ expression stale-but-
plausible. Emitting the computed value means the generator is the only
thing that knows how a number is derived.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DEFAULT_OUT = ROOT / "native" / "core" / "config.hpp"

# Scalars copied verbatim, in the order they appear in config.py so the
# generated file reads like the original and diffs stay local.
INT_NAMES = [
    "FS", "RS", "NC", "M", "NCP", "NSYM", "CARRIER0", "FCENTER",
    "SYMS_PER_FRAME", "DATA_SYMS_PER_FRAME", "BEACON_CARRIER", "NC_LATENT",
    "CHIPS_PER_FRAME", "FRAME_SAMPLES", "LATENTS_PER_FRAME",
    "BEACON_COUNTER_BITS", "BEACON_CALLSIGN_CHARS", "BEACON_CALLSIGN_CHAR_BITS",
    "BEACON_CALLSIGN_BITS", "BEACON_CRC_BITS",
    "PREAMBLE_CP", "PREAMBLE_SAMPLES", "HEADER_SYMS", "HEADER_SAMPLES",
    "LEADIN_SAMPLES", "LEADOUT_SAMPLES",
    "LATENT_H", "LATENT_W", "LATENT_GROUPS", "CHANNELS_PER_GROUP",
    "LATENT_CHANNELS", "GROUP_LATENTS", "FRAMES_PER_GROUP",
    "TRANSMIT_LATENTS_PER_GROUP", "DROPPED_LATENTS_PER_GROUP",
    "DEMOD_BACKOFF", "INTERLEAVER_SEED", "PILOT_SEED", "PROTOCOL_VERSION",
]
DOUBLE_NAMES = ["CLIP_HEADROOM_DB", "SNR_REF_BW_HZ"]


def _hexfloat(x: float) -> str:
    """A double as a C++ hex float literal.

    Decimal round-tripping of doubles is exact in principle (17
    significant digits) but depends on the compiler's strtod being
    correctly rounded. Hex literals are exact by construction, and this
    header is the one place where a one-ulp difference would be both
    possible and maddening to find.
    """
    return float.hex(float(x))


def _emit_double(name: str, value: float, comment: str = "") -> str:
    tail = f"  // {comment}" if comment else ""
    return (f"inline constexpr double {name} = {_hexfloat(value)};"
            f"  // {value!r}{tail}\n")


def render() -> str:
    import numpy as np

    from sstvae import config as cfg
    from sstvae.modem import ofdm

    out: list[str] = []
    w = out.append

    w("// GENERATED FILE -- DO NOT EDIT.\n")
    w("//\n")
    w("// Written by tools/gen_config_header.py from sstvae/config.py, which\n")
    w("// is the single source of truth for every waveform and latent number\n")
    w("// in this project. Editing this file by hand will be reverted by the\n")
    w("// next generator run, and CI fails if the two disagree.\n")
    w("//\n")
    w("// To change a constant: edit sstvae/config.py, re-run the generator,\n")
    w("// and commit both.\n")
    w("\n")
    w("#pragma once\n")
    w("\n")
    w("#include <array>\n")
    w("#include <complex>\n")
    w("#include <cstddef>\n")
    w("#include <string_view>\n")
    w("\n")
    w("namespace sstvae::config {\n")
    w("\n")

    w("// --- scalars ---------------------------------------------------------\n")
    for name in INT_NAMES:
        value = getattr(cfg, name)
        w(f"inline constexpr int {name} = {int(value)};\n")
    w("\n")
    for name in DOUBLE_NAMES:
        w(_emit_double(name, getattr(cfg, name)))
    w("\n")

    tx_lo, tx_hi = cfg.TX_BANDPASS
    w("// Post-clip transmit filter, Hz.\n")
    w(_emit_double("TX_BANDPASS_LO", tx_lo))
    w(_emit_double("TX_BANDPASS_HI", tx_hi))
    w("\n")

    w("// --- beacon sync word ------------------------------------------------\n")
    w("// Barker-13: a clean, unambiguous chip-level autocorrelation peak, so\n")
    w("// superframe phase is recoverable from any contiguous run of frames.\n")
    sync = ", ".join(str(int(v)) for v in cfg.BEACON_SYNC)
    w(f"inline constexpr int BEACON_SYNC_LEN = {len(cfg.BEACON_SYNC)};\n")
    w("inline constexpr std::array<int, BEACON_SYNC_LEN> BEACON_SYNC = "
      f"{{{sync}}};\n")
    w("\n")

    w("// --- pilot sequence --------------------------------------------------\n")
    w("// GENERATED VALUES, not a generated algorithm. Python draws these from\n")
    w("// np.random.default_rng(PILOT_SEED), but they are a *format constant*:\n")
    w("// 24 fixed QPSK symbols both ends must agree on. Reimplementing PCG64\n")
    w("// and numpy's bounded-integer draw in C++ to rederive them would make\n")
    w("// numpy's RNG internals part of the on-air format, which they are not.\n")
    w("//\n")
    w("// Emitted as the quadrant indices rather than as phases or as complex\n")
    w("// values, so C++ evaluates the *same expression* Python does:\n")
    w("//     phase = pi/4 + pi/2 * k\n")
    w("// Emitting phases would mean emitting np.angle()'s principal value,\n")
    w("// which wraps 7pi/4 to -pi/4 -- mathematically identical, but a\n")
    w("// different argument to exp() and so a different last ulp for no\n")
    w("// reason. Residual divergence is then libm's alone.\n")
    quadrants = np.rint(
        (np.angle(ofdm.pilot_sequence()) - np.pi / 4) / (np.pi / 2)
    ).astype(int) % 4
    # Verify the emitted indices actually reproduce the sequence, so a
    # future change to pilot_sequence() cannot silently ship a header
    # describing a waveform nobody transmits.
    rebuilt = np.exp(1j * (np.pi / 4 + np.pi / 2 * quadrants))
    err = np.max(np.abs(rebuilt - ofdm.pilot_sequence()))
    if err > 1e-12:
        raise SystemExit(
            f"pilot quadrant recovery failed (max error {err:.3g}); "
            "ofdm.pilot_sequence() is no longer a plain QPSK draw and this "
            "generator needs updating"
        )
    w(f"inline constexpr std::array<int, NC> PILOT_QUADRANTS = {{\n")
    for i in range(0, len(quadrants), 12):
        w("    " + ", ".join(str(k) for k in quadrants[i:i + 12]) + ",\n")
    w("};\n")
    w("\n")

    w("// --- modes -----------------------------------------------------------\n")
    w("struct ModeSpec {\n")
    w("    std::string_view name;\n")
    w("    int index;\n")
    w("    int groups;          // latent channel groups transmitted\n")
    w("    int n_frames;\n")
    w("    int n_latents;       // model-facing contract (groups * GROUP_LATENTS)\n")
    w("    int n_tx_latents;    // actually carried on air (23-carrier capacity)\n")
    w("    double duration_s;\n")
    w("};\n")
    w("\n")
    modes = sorted(cfg.MODES.values(), key=lambda m: m.index)
    w(f"inline constexpr int N_MODES = {len(modes)};\n")
    w("inline constexpr std::array<ModeSpec, N_MODES> MODES = {{\n")
    for m in modes:
        w(f'    {{"{m.name}", {m.index}, {m.groups}, {m.n_frames}, '
          f"{m.n_latents}, {m.n_tx_latents}, {_hexfloat(m.duration_s)}}},"
          f"  // ~{m.duration_s:.0f} s\n")
    w("}};\n")
    w("\n")
    w("// Modes are indexed by their on-air index, which is also their position\n")
    w("// in the table; the static_assert keeps that true if a mode is ever\n")
    w("// added out of order.\n")
    w("static_assert(MODES[0].index == 0 && MODES[N_MODES - 1].index == N_MODES - 1,\n")
    w('              "MODES must be stored in on-air index order");\n')
    w("\n")

    w("// --- invariants ------------------------------------------------------\n")
    w("// Restated from config.py's own asserts, checked at compile time here.\n")
    w("static_assert(M == FS / RS);\n")
    w("static_assert(NSYM == M + NCP);\n")
    w("static_assert(FRAME_SAMPLES == SYMS_PER_FRAME * NSYM);\n")
    w("static_assert(LATENTS_PER_FRAME == NC_LATENT * DATA_SYMS_PER_FRAME * 2);\n")
    w("static_assert(LATENT_CHANNELS == LATENT_GROUPS * CHANNELS_PER_GROUP);\n")
    w("static_assert(GROUP_LATENTS == CHANNELS_PER_GROUP * LATENT_H * LATENT_W);\n")
    w("static_assert(FRAMES_PER_GROUP * NC * DATA_SYMS_PER_FRAME * 2 == GROUP_LATENTS,\n")
    w('              "FRAMES_PER_GROUP is pinned to the pre-beacon capacity");\n')
    w("static_assert(TRANSMIT_LATENTS_PER_GROUP == FRAMES_PER_GROUP * LATENTS_PER_FRAME);\n")
    w("static_assert(DROPPED_LATENTS_PER_GROUP ==\n")
    w("                  GROUP_LATENTS - TRANSMIT_LATENTS_PER_GROUP &&\n")
    w("              DROPPED_LATENTS_PER_GROUP >= 0 &&\n")
    w("              DROPPED_LATENTS_PER_GROUP < GROUP_LATENTS,\n")
    w('              "the per-group remainder is a permanent erasure");\n')
    w("static_assert(BEACON_CARRIER == NC - 1 && NC_LATENT == NC - 1,\n")
    w('              "one carrier is reserved for the beacon side-channel");\n')
    w("\n")
    w("}  // namespace sstvae::config\n")

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
            print(f"{args.out} does not exist; run tools/gen_config_header.py",
                  file=sys.stderr)
            return 1
        if current != text:
            print(f"{args.out} is out of date with sstvae/config.py.\n"
                  "Re-run tools/gen_config_header.py and commit the result.",
                  file=sys.stderr)
            return 1
        print(f"{args.out} is up to date")
        return 0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    print(f"wrote {args.out} ({len(text.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

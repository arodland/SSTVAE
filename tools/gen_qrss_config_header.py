#!/usr/bin/env python3
"""Emit `native/core/qrss/qrss_config.hpp` and `native/core/qrss/tables.cpp`
from `sstvae/qrss`, the C++ port's copy of every QRSSTVAE format constant.

    tools/gen_qrss_config_header.py            # write both files
    tools/gen_qrss_config_header.py --check    # exit 1 if either would change

The same rule as `tools/gen_config_header.py` and for the same reason: a
hand-copied on-air number in C++ is the most likely way for two stations
to disagree silently, so the C++ gets a generated copy and `--check`
says when it is stale. Everything here is read from the Python modules,
never retyped.

- **Doubles are hex literals**, exact by construction. The ones derived
  from `math.exp` at import (A0, K_LIN, CARRIER_FRAC, USEFUL_FRAC) must
  not be recomputed with `std::exp`: libms disagree in the last place.
- **Rationals stay rationals** (`Ratio{num, den}`), so mixing and pilot
  offsets stay in exact integer turns, as `L_PILOT_HZ` = 12207/970 is in
  Python.
- **Frame presets are data**: every CE and L preset, the version-1
  frames and the prototypes, by name. The C++ derives lengths from them
  with the same arithmetic as `frame.FrameSpec`, and the parity tests
  compare every derived length.
- **The polar information set and the Morse table** are format tables.
- **`tables.cpp` holds frozen float tables** (`ce.PULSE_TABLE`,
  `lwave.PULSE_TABLE`, the L interpolator's unit-gain taps). They are
  computed at import through libm `sin`/`cos`, so freezing them makes the
  C++ bit-identical to Python on the generating machine, which is the
  "format constants are frozen data" rule in CLAUDE.md.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OUT_DIR = ROOT / "native" / "core" / "qrss"
CONFIG_OUT = OUT_DIR / "qrss_config.hpp"
TABLES_OUT = OUT_DIR / "tables.cpp"

BANNER = ("// GENERATED FILE -- DO NOT EDIT.\n"
          "// Written by tools/gen_qrss_config_header.py from sstvae/qrss.\n"
          "// Regenerate with: python tools/gen_qrss_config_header.py\n")

INT_NAMES = [
    "FS", "T_NUM", "T_DEN", "SPAN",
    "N_PRE", "N_HDR", "REF_PERIOD", "N_HDR_BITS", "N_INFO_BITS", "POLAR_N",
    "FORMAT_VERSION", "WAVEFORM_CE", "WAVEFORM_L", "LEAD_IN_MAX_S",
    "CW_WINDOW", "CW_UNIT", "CW_UNITS", "CW_LEAD_UNITS", "CW_ROOM_UNITS",
    "INT8_SCALE", "FE_FS", "FE_CENTER_HZ", "CH_FS", "CH_DECIM",
    "L_T_NUM", "L_N_PRE", "L_N_HDR", "L_CW_WINDOW",
]
DOUBLE_NAMES = [
    "T_SYM", "A0", "K_LIN", "CARRIER_FRAC", "USEFUL_FRAC", "SPREAD_RHO",
    "Z_ACCEPT", "L_T_SYM", "L_CLIP_HEADROOM_DB", "L_CLIP_GUARD_HZ",
]
RATIO_NAMES = ["ALPHA", "BETA", "L_EDGE_HZ", "L_PILOT_HZ", "L_PILOT_FRAC"]
INT_TUPLE_NAMES = ["FORMAT_VERSIONS", "CW_AFTER_FULL", "L_CW_AFTER_FULL"]
DOUBLE_TUPLE_NAMES = ["CARRIER_BAND_HZ", "L_CLIP_OVERSHOOT"]
DOMAIN_NAMES = ["PREAMBLE_DOMAIN", "REFERENCE_DOMAIN", "SCRAMBLE_DOMAIN",
                "SPREAD_DOMAIN", "L_PREAMBLE_DOMAIN", "L_REFERENCE_DOMAIN"]


def _hex(x: float) -> str:
    """A double as a C++ hex float literal (exact)."""
    x = float(x)
    if x == 0.0:
        return "-0.0" if str(x).startswith("-") else "0.0"
    return float.hex(x)


def _cstr(b: bytes) -> str:
    s = b.decode("ascii")
    assert all(32 <= ord(c) < 127 and c not in '"\\' for c in s), s
    return f'"{s}"'


def _presets():
    from sstvae.qrss import frame, lframe

    out = []
    for group, specs in (("PRESETS", frame.PRESETS), ("LEGACY", frame.LEGACY),
                         ("PROTOTYPES", frame.PROTOTYPES), ("L_PRESETS", lframe.PRESETS)):
        for name, s in specs.items():
            out.append((group, s))
    return out


def render_config() -> str:
    from sstvae.qrss import constants as k
    from sstvae.qrss import morse, polar, precoder, sequences

    lines: list[str] = []
    w = lines.append
    w(BANNER)
    w("//\n// Every QRSSTVAE format constant, for the C++ port (native/core/qrss/).\n"
      "// SSTVAE's own numbers (TRANSMIT_LATENTS_PER_GROUP and friends) are in\n"
      "// ../config.hpp, generated from sstvae/config.py, and are not restated.\n\n")
    w("#pragma once\n\n#include <array>\n#include <cstdint>\n#include <string_view>\n\n")
    w("namespace sstvae::qrss::config {\n\n")
    w("// An exact rational, as fractions.Fraction is in Python.\n")
    w("struct Ratio {\n    std::int64_t num;\n    std::int64_t den;\n"
      "    constexpr double value() const { return static_cast<double>(num) / static_cast<double>(den); }\n};\n\n")

    w("// --- scalars (sstvae/qrss/constants.py) ---\n")
    for name in INT_NAMES:
        v = getattr(k, name)
        assert isinstance(v, int) and not isinstance(v, bool), name
        w(f"inline constexpr int {name} = {v};\n")
    for name in DOUBLE_NAMES:
        v = getattr(k, name)
        w(f"inline constexpr double {name} = {_hex(v)};  // {float(v)!r}\n")
    for name in RATIO_NAMES:
        v = Fraction(getattr(k, name))
        w(f"inline constexpr Ratio {name}{{{v.numerator}, {v.denominator}}};  // {float(v)!r}\n")
    for name in INT_TUPLE_NAMES:
        v = tuple(int(x) for x in getattr(k, name))
        w(f"inline constexpr std::array<int, {len(v)}> {name}{{{', '.join(map(str, v))}}};\n")
    for name in DOUBLE_TUPLE_NAMES:
        v = tuple(float(x) for x in getattr(k, name))
        w(f"inline constexpr std::array<double, {len(v)}> {name}{{{', '.join(map(_hex, v))}}};"
          f"  // {v!r}\n")
    w("\n// --- SHA-256 domain strings (sstvae/qrss/sequences.py) ---\n")
    for name in DOMAIN_NAMES:
        w(f"inline constexpr std::string_view {name} = {_cstr(getattr(sequences, name))};\n")

    w("\n// The Python module's own consistency asserts.\n")
    w("static_assert(T_DEN % FS == 0 && FE_FS == CH_FS * CH_DECIM, \"rates nest\");\n")
    w("static_assert(CW_LEAD_UNITS + CW_ROOM_UNITS + CW_LEAD_UNITS == CW_UNITS);\n")
    w("static_assert(L_PILOT_HZ.num == 12207 && L_PILOT_HZ.den == 970 && 2 * L_CW_WINDOW == CW_WINDOW);\n")

    w("\n// --- frame presets (frame.PRESETS, LEGACY, PROTOTYPES; lframe.PRESETS) ---\n")
    w("// The parameters only; lengths are derived in frame.cpp exactly as\n"
      "// frame.FrameSpec / lframe.LFrameSpec derive them. `hdr_rho` is the\n"
      "// resolved value (FrameSpec.__post_init__ has already applied the default).\n")
    w("enum class PresetGroup { Presets, Legacy, Prototypes, LPresets };\n")
    w("struct PresetDef {\n    std::string_view name;\n    PresetGroup group;\n    int waveform;\n"
      "    int n_pre;\n    int n_hdr;\n    int n_data;\n    int n_win;\n    std::array<int, 4> cw_after;\n"
      "    double hdr_rho;\n};\n")
    pres = _presets()
    group_enum = {"PRESETS": "Presets", "LEGACY": "Legacy", "PROTOTYPES": "Prototypes",
                  "L_PRESETS": "LPresets"}
    w(f"inline constexpr std::array<PresetDef, {len(pres)}> FRAME_PRESETS{{{{\n")
    for group, s in pres:
        cw = tuple(int(c) for c in s.cw_after)
        assert len(cw) <= 4, s.name
        cwp = cw + (0,) * (4 - len(cw))
        w(f"    {{{_cstr(s.name.encode())}, PresetGroup::{group_enum[group]}, {int(s.waveform)}, "
          f"{s.n_pre}, {s.n_hdr}, {s.n_data}, {len(cw)}, {{{', '.join(map(str, cwp))}}}, "
          f"{_hex(s.hdr_rho)}}},\n")
    w("}};\n")

    info = tuple(int(i) for i in polar.INFO_SET)
    assert list(info) == sorted(info) and len(info) == k.N_INFO_BITS
    digest = polar.INFO_SET_SHA256
    w("\n// --- polar information set (sstvae/qrss/polar.py), ascending ---\n")
    w(f"inline constexpr std::string_view POLAR_INFO_SET_SHA256 = \"{digest}\";\n")
    w(f"inline constexpr std::array<int, {len(info)}> POLAR_INFO_SET{{{{\n")
    for i in range(0, len(info), 12):
        w("    " + ", ".join(map(str, info[i:i + 12])) + ",\n")
    w("}};\n")

    w("\n// --- precoder (sstvae/qrss/precoder.py) ---\n")
    w(f"inline constexpr int PRECODER_BLOCK = {int(precoder.BLOCK)};\n")

    w("\n// --- Morse (sstvae/qrss/morse.py), in the table's own order ---\n")
    w("struct MorseCode {\n    char ch;\n    std::string_view code;\n};\n")
    w(f"inline constexpr int MAX_CALL_CHARS = {int(morse.MAX_CALL_CHARS)};\n")
    w(f"inline constexpr std::array<MorseCode, {len(morse.MORSE)}> MORSE{{{{\n")
    for ch, code in morse.MORSE.items():
        w(f"    {{'{ch}', \"{code}\"}},\n")
    w("}};\n")

    w("\n// --- module constants (picture.py, beaconfile.py, ce.py, lwave.py) ---\n")
    w("// Defined in those modules rather than constants.py; the C++ modules take\n"
      "// them from here so no copy is typed by hand.\n")
    from sstvae.qrss import beaconfile, ce, lwave, picture
    names = tuple(picture.MODE_NAMES)
    w(f"inline constexpr std::array<std::string_view, {len(names)}> MODE_NAMES{{"
      f"{', '.join(_cstr(n.encode()) for n in names)}}};\n")
    for cname, v in [("QRSP_VERSION", picture.QRSP_VERSION),
                     ("BEACON_FILE_VERSION", beaconfile.FILE_VERSION),
                     ("BEACON_FLAG_OOK", beaconfile.FLAG_OOK),
                     ("BEACON_HEAD_BYTES", beaconfile.HEAD_BYTES),
                     ("CE_OOK_FIRST_KEYED_UNIT", ce.OOK_FIRST_KEYED_UNIT),
                     ("CE_OOK_LAST_KEYED_UNIT", ce.OOK_LAST_KEYED_UNIT),
                     ("L_ID_RAMP_UNITS", lwave.ID_RAMP_UNITS),
                     ("L_INTERP_HALF", lwave.INTERP_HALF)]:
        assert isinstance(v, int) and not isinstance(v, bool), cname
        w(f"inline constexpr int {cname} = {v};\n")
    w(f"inline constexpr std::string_view BEACON_MAGIC = {_cstr(beaconfile.MAGIC)};\n")
    import inspect
    margin = inspect.signature(lwave.frame_baseband).parameters["margin_s"].default
    for cname, v in [("L_AVG_POWER", lwave.L_AVG_POWER), ("L_KEY_RAMP_S", lwave.KEY_RAMP_S),
                     ("L_MARGIN_S", margin)]:
        w(f"inline constexpr double {cname} = {_hex(float(v))};  // {float(v)!r}\n")

    w("\n// --- a pass's timing (frontend.slot_t0, tx.transmit_audio's defaults, frame.FULL) ---\n")
    w("// Slot q's quarter hour is SLOT_S * q (unix seconds) and its t0 T0_OFFSET_S\n"
      "// later; a pass's audio starts TX_PRE_S before t0 and ends TX_POST_S after\n"
      "// the keying. FULL_DURATION_S is the FULL frame's, t0 to its end, the same\n"
      "// for CE and L. For the app (qrss/api.hpp), so no copy is typed by hand.\n")
    from sstvae.qrss import frame, frontend, lframe, tx
    slot_s, t0_offset = frontend.slot_t0(1) - frontend.slot_t0(0), frontend.slot_t0(0)
    params = inspect.signature(tx.transmit_audio).parameters
    full_s = frame.FULL.duration_s
    assert lframe.get("l-full").duration_s == full_s, "CE and L FULL frames last the same"
    for cname, v in [("SLOT_S", slot_s), ("T0_OFFSET_S", t0_offset),
                     ("TX_CARRIER_HZ", params["carrier_hz"].default)]:
        assert isinstance(v, int) and not isinstance(v, bool), cname
        w(f"inline constexpr int {cname} = {v};\n")
    for cname, v in [("TX_PRE_S", params["pre_s"].default), ("TX_POST_S", params["post_s"].default),
                     ("TX_AMPLITUDE", params["amplitude"].default),
                     ("TX_CHUNK_S", params["chunk_s"].default), ("FULL_DURATION_S", full_s)]:
        w(f"inline constexpr double {cname} = {_hex(float(v))};  // {float(v)!r}\n")

    w("\n}  // namespace sstvae::qrss::config\n")
    return "".join(lines)


def _tables():
    """(name, comment, float64 array) for every frozen table."""
    import numpy as np

    from sstvae.qrss import ce, lwave

    interp = np.asarray(lwave._interp_taps(1), dtype=np.float64) \
        if hasattr(lwave, "_interp_taps") else None
    out = [
        ("CE_PULSE_TABLE", "ce.PULSE_TABLE: p(m/485), m = -3880..3880", np.asarray(ce.PULSE_TABLE)),
        ("L_PULSE_TABLE", "lwave.PULSE_TABLE: p(m/970), m = -7760..7760", np.asarray(lwave.PULSE_TABLE)),
    ]
    if interp is not None:
        out.append(("L_INTERP_TAPS", "lwave interpolator, unit gain (multiply by `up`)", interp))
    for name, _, a in out:
        assert a.dtype == np.float64 and a.ndim == 1, name
    return out


def render_tables() -> tuple[str, str]:
    """tables.cpp and the matching declarations for tables.hpp."""
    tabs = _tables()
    src: list[str] = [BANNER, "\n#include \"qrss/tables.hpp\"\n\nnamespace sstvae::qrss::tables {\n"]
    hdr: list[str] = [BANNER, "//\n// Frozen float tables of the QRSSTVAE format. Defined in tables.cpp.\n\n"
                      "#pragma once\n\n#include <array>\n#include <string_view>\n\n"
                      "namespace sstvae::qrss::tables {\n\n"]
    for name, comment, a in tabs:
        digest = hashlib.sha256(a.astype("<f8").tobytes()).hexdigest()
        hdr.append(f"// {comment}\n// sha256 of the little-endian float64 bytes: {digest}\n")
        hdr.append(f"extern const std::array<double, {len(a)}> {name};\n")
        hdr.append(f"inline constexpr std::string_view {name}_SHA256 = \"{digest}\";\n\n")
        src.append(f"\nconst std::array<double, {len(a)}> {name}{{{{\n")
        vals = [_hex(v) for v in a.tolist()]
        for i in range(0, len(vals), 4):
            src.append("    " + ", ".join(vals[i:i + 4]) + ",\n")
        src.append("}};\n")
    hdr.append("}  // namespace sstvae::qrss::tables\n")
    src.append("\n}  // namespace sstvae::qrss::tables\n")
    return "".join(src), "".join(hdr)


def _tables_agree(old: str, new: str, tol: float = 1e-15) -> bool:
    """Same tables, values within `tol` (a few ulp of the pulse's peak)."""
    import re

    def parse(text):
        out = {}
        for m in re.finditer(r"const std::array<double, (\d+)> (\w+)\{\{(.*?)\}\};", text, re.S):
            vals = [float.fromhex(v) if "0x" in v else float(v)
                    for v in re.findall(r"-?0x[0-9a-fA-Fp.+-]+|-?\d+\.\d+", m.group(3))]
            out[m.group(2)] = vals
        return out

    a, b = parse(old), parse(new)
    if a.keys() != b.keys():
        return False
    return all(len(a[k]) == len(b[k]) and max((abs(x - y) for x, y in zip(a[k], b[k])), default=0.0) <= tol
               for k in a)


def outputs() -> dict[Path, str]:
    src, hdr = render_tables()
    return {CONFIG_OUT: render_config(), TABLES_OUT: src, OUT_DIR / "tables.hpp": hdr}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true",
                    help="exit 1 if a generated file is missing or stale")
    args = ap.parse_args()
    stale = []
    for path, text in outputs().items():
        old = path.read_text(encoding="utf-8") if path.exists() else None
        if old == text:
            continue
        if args.check and path == TABLES_OUT and old is not None and _tables_agree(old, text):
            # Frozen data: libm sin/cos on another machine may move the
            # last place, and the committed table is then the format,
            # not the regeneration (CLAUDE.md, "Format constants are
            # frozen data"). Only a real change fails the check.
            continue
        if args.check:
            stale.append(path)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
            print(f"wrote {path.relative_to(ROOT)}")
    if stale:
        for p in stale:
            print(f"stale: {p.relative_to(ROOT)}; run tools/gen_qrss_config_header.py",
                  file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

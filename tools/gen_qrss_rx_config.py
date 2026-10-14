#!/usr/bin/env python3
"""Emit `native/core/qrss/rx_config.hpp` from the QRSSTVAE receiver's
modules: every module-level constant of `sstvae/qrss/{track, ltrack,
frontend, acquire, demod, ldemod, cwid, receiver, render, rules, ce,
lwave}` that has a C++ spelling.

    tools/gen_qrss_rx_config.py            # write the header
    tools/gen_qrss_rx_config.py --check    # exit 1 if it would change

The receiver's calibrations (KAPPA_SELF, the D_* and G_* gains, the
Gumbel tables, Q_SPREADS_HZ, ...) are measured on the exact Python
statistic, so the C++ must use the same numbers: this header is their
one copy, generated rather than typed (`tools/gen_qrss_config_header.py`
does the same for the format constants, in `config.hpp`; this is its
receive-side twin, kept separate so the format header's CI gate does not
move whenever a receiver calibration does).

Rules:

- One namespace per module, `sstvae::qrss::rxconfig::<module>`, holding
  that module's names in source order. A constant a module imports is
  not repeated (its own module emits it).
- A private name (`_FP_NUM`) loses its leading underscores and gains a
  trailing one (`FP_NUM_`): a leading underscore and a capital is
  reserved in C++.
- bool, int (int, or std::int64_t past int32), float (hex literal,
  exact), Fraction (config::Ratio), str (std::string_view); a tuple or
  list of one of those kinds, a 1-D numeric ndarray of at most 256
  values, and a dict of str to number (an array of NamedValue, in
  insertion order). Anything else (caches, patterns, dataclasses, large
  tables) is listed in a comment as not emitted.
- Floats computed at import through libm (ce.G0, lwave.A_CLIP, ...) can
  move in the last place between machines; `--check` therefore accepts
  a regenerated float within 4 ulp of the committed one, like the format
  generator's frozen tables, and the committed value is the one the C++
  uses.
"""

from __future__ import annotations

import argparse
import ast
import importlib
import math
import re
import sys
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OUT = ROOT / "native" / "core" / "qrss" / "rx_config.hpp"

BANNER = ("// GENERATED FILE -- DO NOT EDIT.\n"
          "// Written by tools/gen_qrss_rx_config.py from sstvae/qrss.\n"
          "// Regenerate with: python tools/gen_qrss_rx_config.py\n")

MODULES = ("track", "ltrack", "frontend", "acquire", "demod", "ldemod", "cwid", "receiver",
           "render", "rules", "ce", "lwave", "live", "store", "associate")

MAX_ARRAY = 256


def _hex(x: float) -> str:
    x = float(x)
    if math.isnan(x):
        return "std::numeric_limits<double>::quiet_NaN()"
    if math.isinf(x):
        return ("-" if x < 0 else "") + "std::numeric_limits<double>::infinity()"
    if x == 0.0:
        return "-0.0" if math.copysign(1.0, x) < 0 else "0.0"
    return float.hex(x)


def _cstr(s: str) -> str:
    assert all(32 <= ord(c) < 127 for c in s), s
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _cname(name: str) -> str:
    stripped = name.lstrip("_")
    return stripped + "_" if stripped != name else name


def _own_names(module) -> list[str]:
    """Top-level names the module assigns itself, in source order."""
    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    out: list[str] = []

    def add(t):
        if isinstance(t, ast.Name):
            if t.id not in out:
                out.append(t.id)
        elif isinstance(t, (ast.Tuple, ast.List)):
            for e in t.elts:
                add(e)

    for node in tree.body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                add(t)
        elif isinstance(node, ast.AnnAssign):
            add(node.target)
    return [n for n in out if n.lstrip("_") and n.lstrip("_")[0].isupper()
            and n.lstrip("_").upper() == n.lstrip("_")]


def _is_int(v) -> bool:
    import numpy as np
    return (isinstance(v, (int, np.integer)) and not isinstance(v, (bool, np.bool_)))


def _is_float(v) -> bool:
    import numpy as np
    return isinstance(v, (float, np.floating))


def _int_decl(name: str, v: int) -> str:
    v = int(v)
    ty = "int" if -2**31 <= v < 2**31 else "std::int64_t"
    return f"inline constexpr {ty} {name} = {v};\n"


def _render_value(name: str, v) -> str | None:
    import numpy as np
    c = _cname(name)
    if isinstance(v, (bool, np.bool_)):
        return f"inline constexpr bool {c} = {'true' if v else 'false'};\n"
    if _is_int(v):
        return _int_decl(c, v)
    if _is_float(v):
        return f"inline constexpr double {c} = {_hex(v)};  // {float(v)!r}\n"
    if isinstance(v, Fraction):
        return (f"inline constexpr config::Ratio {c}{{{v.numerator}, {v.denominator}}};"
                f"  // {float(v)!r}\n")
    if isinstance(v, str):
        return f"inline constexpr std::string_view {c} = {_cstr(v)};\n"
    if isinstance(v, np.ndarray):
        if v.ndim != 1 or v.size > MAX_ARRAY or v.dtype.kind not in "iuf":
            return None
        v = tuple(v.tolist())
    if isinstance(v, (tuple, list)):
        if not v:
            return None
        if all(_is_int(e) for e in v):
            ints = [int(e) for e in v]
            ty = "int" if all(-2**31 <= e < 2**31 for e in ints) else "std::int64_t"
            return (f"inline constexpr std::array<{ty}, {len(ints)}> {c}{{"
                    f"{', '.join(map(str, ints))}}};\n")
        if all(_is_int(e) or _is_float(e) for e in v):
            vals = [float(e) for e in v]
            body = ", ".join(_hex(e) for e in vals)
            return (f"inline constexpr std::array<double, {len(vals)}> {c}{{{body}}};"
                    f"  // {tuple(vals)!r}\n")
        if all(isinstance(e, str) for e in v):
            return (f"inline constexpr std::array<std::string_view, {len(v)}> {c}{{"
                    f"{', '.join(_cstr(e) for e in v)}}};\n")
        return None
    if isinstance(v, dict):
        if not v or not all(isinstance(k, str) for k in v):
            return None
        if not all(_is_int(e) or _is_float(e) for e in v.values()):
            return None
        rows = ", ".join(f"NamedValue{{{_cstr(k)}, {_hex(float(x))}}}" for k, x in v.items())
        return (f"inline constexpr std::array<NamedValue, {len(v)}> {c}{{{{{rows}}}}};"
                f"  // {dict((k, float(x)) for k, x in v.items())!r}\n")
    return None


def render() -> str:
    lines = [BANNER]
    w = lines.append
    w("//\n// The QRSSTVAE receiver's module constants (calibrations, settings and\n"
      "// thresholds), one namespace per Python module. Not a format: nothing\n"
      "// here goes on the air (config.hpp has those). Private Python names\n"
      "// (_FP_NUM) appear with the leading underscores moved to the end (FP_NUM_).\n\n")
    w("#pragma once\n\n#include <array>\n#include <cstddef>\n#include <cstdint>\n#include <limits>\n"
      "#include <string_view>\n\n#include \"qrss/config.hpp\"\n\n")
    w("namespace sstvae::qrss::rxconfig {\n\n")
    w("// One entry of a Python dict of str to number, in insertion order.\n")
    w("struct NamedValue {\n    std::string_view name;\n    double value;\n};\n\n")
    w("// The value for `name`, or `fallback` (dict.get).\n")
    w("template <std::size_t N>\nconstexpr double lookup(const std::array<NamedValue, N>& table, "
      "std::string_view name, double fallback) {\n"
      "    for (const NamedValue& e : table)\n        if (e.name == name) return e.value;\n"
      "    return fallback;\n}\n")
    for mod_name in MODULES:
        m = importlib.import_module(f"sstvae.qrss.{mod_name}")
        w(f"\n// --- sstvae/qrss/{mod_name}.py ---\nnamespace {mod_name} {{\n")
        skipped = []
        for name in _own_names(m):
            if not hasattr(m, name):
                continue
            text = _render_value(name, getattr(m, name))
            if text is None:
                skipped.append(f"{name} ({type(getattr(m, name)).__name__})")
            else:
                w(text)
        if skipped:
            w("// not emitted: " + ", ".join(skipped) + "\n")
        w(f"}}  // namespace {mod_name}\n")
    w("\n}  // namespace sstvae::qrss::rxconfig\n")
    return "".join(lines)


_HEX = re.compile(r"-?0x[0-9a-fA-F]+(?:\.[0-9a-fA-F]*)?p[+-]?\d+")


def _agree(old: str, new: str, ulps: int = 4) -> bool:
    """The same header up to hex doubles (and their repr comments) within
    `ulps`: libm-derived constants may move in the last place."""
    def skeleton(text):
        body = "\n".join(line.split("  //")[0] for line in text.splitlines())
        return _HEX.sub("H", body), [float.fromhex(x) for x in _HEX.findall(body)]

    (so, vo), (sn, vn) = skeleton(old), skeleton(new)
    if so != sn or len(vo) != len(vn):
        return False
    for a, b in zip(vo, vn):
        if a != b and abs(a - b) > ulps * math.ulp(max(abs(a), abs(b))):
            return False
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true",
                    help="exit 1 if the generated header is missing or stale")
    args = ap.parse_args()
    text = render()
    old = OUT.read_text(encoding="utf-8") if OUT.exists() else None
    if old == text:
        return 0
    if args.check:
        if old is not None and _agree(old, text):
            return 0
        print(f"stale: {OUT.relative_to(ROOT)}; run tools/gen_qrss_rx_config.py", file=sys.stderr)
        return 1
    OUT.write_text(text, encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

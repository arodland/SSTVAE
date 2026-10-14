#!/usr/bin/env python3
"""The pins of native/tests/qrss/test_decode.cpp, from qrss_decode.py itself.

Writes native/tests/qrss/decode_pins.hpp, which is committed so the ctest
needs no Python. Re-run it from the repository root if the cases below or
qrss_decode.py's helpers change:

    python native/tests/qrss/gen_decode_pins.py

native/core/qrss/decode.hpp ports qrss_decode.py's parse_key, acc_lines and
the line --pass prints. These pins hold them to the script without a store
or a subprocess (tests/test_qrss_native_decode.py runs the two programs side
by side on real stores):

- str.upper() of callsigns (SpecialCasing's expansions, a surrogate
  escape kept), and f"{v:08x}" of ints of any size (the sign inside the
  width);
- parse_key's results and its two refusals, by their text;
- acc_lines of accumulators built here (no store, so no member_wave):
  the mean W over the heard latents, per-segment medians, members whose
  fields are of any JSON type, and the exceptions the script ends with;
- the --pass line (its mean W in float32, max(.., 1e-30), NaN kept).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

import qrss_decode  # noqa: E402
from sstvae.qrss.constants import WAVEFORM_NAMES  # noqa: E402
from sstvae.qrss.store import Accumulator  # noqa: E402

OUT = Path(__file__).with_name("decode_pins.hpp")

UPPERS = [b"k1abc/p", "straße".encode(), "ﬃ".encode(), "ŉ".encode(), "ǰ".encode(),
          "ΐ".encode(), "ᾀᾳ".encode(), "ıiİ".encode(), "µ".encode(),
          "ſ".encode(), "ẞÿ".encode(), b"ab\xffc", b"\xc3", b"\xed\xa0\x80x", b"\xf4\x90\x80\x80",
          "\U00010428".encode(), "ა".encode(), "ͅ".encode(), b""]

HEXES = [("0", 8), ("1", 8), ("-1", 8), ("-5", 8), ("3735928559", 8), ("4294967296", 8), ("53720", 4),
         ("7", 4), ("-123456789", 4), (str(2 ** 128 - 1), 8), (str(-(2 ** 70)), 8), ("9223372036854775807", 8),
         ("-9223372036854775808", 8)]

KEYS = [b"K1ABC:1a2b3c4d", b"k1abc/p:0x1F", b"a:b:c", b"K1: 1_0 ", b"K1:-1", b"K1:+0X_f", b"K1",
        b":12", b"K1:", b"K1:zz", b"K1:0x", b"K1:1__0", b"K'1\"", b"K1:'", b"stra\xc3\x9fe:ff",
        "K1:１ａ".encode(), b"K\xff:1", b"K1:1\xff", b"K1:" + b"f" * 40, b"K1:\t1\n"]

# (mode, codec_id, callsign, picture_id, members, foreign, runs of heard W: (g, start, count, value))
ACCS = [
    (2, 0xD1D8, "K1ABC/P", 0x1A2B3C4D,
     [{"uid": "u1", "segment": 0, "method": "header", "z_ref": 10.0, "mean_w_db": -3.0103, "em_round": 0},
      {"uid": "u2", "segment": 2, "method": "correlation", "z_ref": 12, "mean_w_db": 1.5, "em_round": 2}],
     [{"source": "remote"}], [(0, 0, 100, 2.0), (0, 100, 50, 0.25), (2, 5, 1, 0.5)]),
    (0, 7, "FULL", 0, [], [], []),
    (1, 0xFFFF, "W1AW", 2 ** 32 - 1, [], [{}, {}], [(0, 0, 60000, 1.0), (1, 3, 2, 1e-3)]),
    (-1, 1, "M", 5, [{"uid": "u", "segment": "s", "method": 3, "z_ref": True, "mean_w_db": 0, "em_round": 1.5}],
     [], [(1, 0, 10, 4.0)]),
    (-3, 1, "M", 5, [{"uid": "u", "segment": None, "method": None, "z_ref": -0.05, "mean_w_db": -1e-9,
                      "em_round": None}], [], []),
    (2, 1, "M", 5, [{"uid": "u", "segment": [1, 2], "method": {"a": [True, None]}, "z_ref": 1e300,
                     "mean_w_db": 12345.678, "em_round": "x"}], [], [(2, 0, 3, 1.0)]),
    (3, 1, "M", 5, [], [], []),
    (-4, 1, "M", 5, [], [], []),
    (0, 1, "M", 5, [{"uid": "u", "segment": 0, "method": "m", "z_ref": "x", "mean_w_db": 0, "em_round": 0}], [], []),
    (0, 1, "M", 5, [{"uid": "u", "segment": 0, "method": "m", "z_ref": None, "mean_w_db": 0, "em_round": 0}], [],
     []),
    (0, 1, "M", 5, [{"uid": "u", "segment": 0, "method": "m", "z_ref": [1], "mean_w_db": 0, "em_round": 0}], [], []),
    (0, 1, "M", 5, [{"uid": "u", "segment": 0, "method": "m", "z_ref": 1, "mean_w_db": {}, "em_round": 0}], [], []),
    (0, 1, "M", 5, [{"segment": 0, "method": "m", "z_ref": 1, "mean_w_db": 0, "em_round": 0}], [], []),
    (0, 1, "M", 5, [{"uid": "u", "segment": 0, "z_ref": None, "mean_w_db": 0, "em_round": 0}], [], []),
    (0, -2, "M", -5, [], [], []),
]

# (uid, waveform, frame, runs of w: (count, value))
PASSES = [("1_2_3", 0, "short", [(2, 0.5), (1, 1.5), (1, 3.0), (1, 0.0)]),
          ("u", 1, "l-short", [(4, 2.0)]),
          ("u", 0, "full", [(10, 0.0)]),
          ("u", 7, "x", [(3, 1e-3)]),
          ("u", 0, "short", [(2, 1.0), (1, float("nan"))]),
          ("u", 0, "short", [(1, float("inf"))]),
          ("u", 0, "short", [(1, -1.0)])]


def cstr(b) -> str:
    if isinstance(b, str):
        b = b.encode("utf-8", "surrogateescape")
    out = []
    prev_octal = False
    for c in b:
        if c == 0x22 or c == 0x5C:
            out.append("\\" + chr(c))
            prev_octal = False
        elif 0x20 <= c < 0x7F and not (prev_octal and chr(c) in "01234567"):
            out.append(chr(c))
            prev_octal = False
        else:
            out.append("\\%03o" % c)
            prev_octal = True
    return '"' + "".join(out) + '"'


def fs(b: bytes) -> str:
    return b.decode("utf-8", "surrogateescape")


def runs_text(runs) -> str:
    return ";".join(",".join(repr(float(x)) if isinstance(x, float) else str(x) for x in r) for r in runs)


def acc_case(mode, codec_id, call, pid, members, foreign, runs):
    acc = Accumulator.empty((call, pid), 0, codec_id, 0.0)
    acc.mode = mode
    acc.members = json.loads(json.dumps(members))
    acc.foreign = foreign
    for g, start, count, value in runs:
        acc.W.reshape(-1)[g * acc.W.shape[1] + start:g * acc.W.shape[1] + start + count] = value
    try:
        return False, "\n".join(qrss_decode.acc_lines(acc))
    except Exception as e:   # noqa: BLE001 -- the script lets it escape
        return True, f"{type(e).__name__}: {e}" if not isinstance(e, KeyError) else f"KeyError: {e}"


def pass_line(uid, waveform, frame_name, runs):
    w = np.concatenate([np.full(n, v, dtype=np.float32) for n, v in runs])
    p = SimpleNamespace(uid=uid, waveform=waveform, frame=frame_name, w=w)
    # qrss_decode.py main(), the --pass branch
    return (f"pass {p.uid}: {WAVEFORM_NAMES.get(p.waveform, p.waveform)} {p.frame} frame, "
            "mean W "
            f"{10 * np.log10(max(float(np.mean(p.w)), 1e-30)):+.2f} dB")


def main() -> None:
    lines = [
        "// GENERATED by native/tests/qrss/gen_decode_pins.py from qrss_decode.py",
        f"// (CPython {sys.version.split()[0]}, numpy {np.__version__}). Do not edit.",
        "",
        "#pragma once",
        "",
        "namespace decode_pins {",
        "",
        "// str.upper() of a callsign (bytes as a file name's: a surrogate escape kept)",
        "struct Upper {",
        "    const char* in;",
        "    const char* out;",
        "};",
        "// format(int(decimal), f\"0{width}x\")",
        "struct Hex {",
        "    const char* decimal;",
        "    int width;",
        "    const char* out;",
        "};",
        "// parse_key(in): (callsign, picture ID in decimal), or its SystemExit's text",
        "struct Key {",
        "    const char* in;",
        "    bool ok;",
        "    const char* callsign;",
        "    const char* decimal;",
        "    const char* refusal;",
        "};",
        "// acc_lines of an accumulator: members and foreign as JSON, W's heard",
        "// latents as \"g,start,count,value\" runs joined by ';'; the lines joined by",
        "// newlines, or (error) the escaping exception as \"Class: message\"",
        "struct Acc {",
        "    long long mode;",
        "    long long codec_id;",
        "    const char* callsign;",
        "    long long picture_id;",
        "    const char* members;",
        "    const char* foreign;",
        "    const char* runs;",
        "    bool error;",
        "    const char* out;",
        "};",
        "// the --pass line of a pass whose w is \"count,value\" runs joined by ';'",
        "struct Pass {",
        "    const char* uid;",
        "    long long waveform;",
        "    const char* frame;",
        "    const char* runs;",
        "    const char* line;",
        "};",
        "",
        "constexpr Upper UPPERS[] = {",
    ]
    for b in UPPERS:
        lines.append(f"    {{{cstr(b)}, {cstr(fs(b).upper())}}},")
    lines += ["};", "", "constexpr Hex HEXES[] = {"]
    for dec, width in HEXES:
        lines.append(f"    {{{cstr(dec)}, {width}, {cstr(format(int(dec), f'0{width}x'))}}},")
    lines += ["};", "", "constexpr Key KEYS[] = {"]
    for b in KEYS:
        try:
            call, pid = qrss_decode.parse_key(fs(b))
            lines.append(f"    {{{cstr(b)}, true, {cstr(call)}, {cstr(str(pid))}, \"\"}},")
        except SystemExit as e:
            lines.append(f"    {{{cstr(b)}, false, \"\", \"\", {cstr(str(e.code))}}},")
    lines += ["};", "", "constexpr Acc ACCS[] = {"]
    for mode, codec_id, call, pid, members, foreign, runs in ACCS:
        error, out = acc_case(mode, codec_id, call, pid, members, foreign, runs)
        lines.append(f"    {{{mode}, {codec_id}, {cstr(call)}, {pid}LL, {cstr(json.dumps(members))}, "
                     f"{cstr(json.dumps(foreign))}, {cstr(runs_text(runs))}, {'true' if error else 'false'}, "
                     f"{cstr(out)}}},")
    lines += ["};", "", "constexpr Pass PASSES[] = {"]
    for uid, waveform, frame_name, runs in PASSES:
        lines.append(f"    {{{cstr(uid)}, {waveform}, {cstr(frame_name)}, {cstr(runs_text(runs))}, "
                     f"{cstr(pass_line(uid, waveform, frame_name, runs))}}},")
    lines += ["};", "", "}  // namespace decode_pins", ""]
    OUT.write_text("\n".join(lines))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()

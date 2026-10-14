#!/usr/bin/env python3
"""Writes native/tests/golden_qrss/live_pins.json: what sstvae/qrss/live.py
makes of a set of dial files and tiles, for native/tests/qrss/test_live.cpp
to check the C++ port (native/core/qrss/live) against without Python.

- "dial": for each dial file (its bytes, as hex), the reference's
  DialLog(path).refresh() points, and for the first file its spans, by_dial
  and span_at at a few times (DIAL_GUARD_S as it is). A file holding a dial
  past int64, which the reference keeps as a Python int and the port skips
  (the one deviation left in the parse), names those lines in "skips": its
  points are the reference's of the file without them, and the C++ must
  find the same in the whole file.
- "tiles": for each tile, asdict(tile) (NaN and Infinity tokens where its
  floats are not finite, which the C++ reads as Tile::from_json takes them)
  and json.dumps(tile.to_json()), the text the C++'s to_json must dump to.

Floats are written as Python writes them (repr: shortest round trip), so
both sides read back the same doubles.

    PYTHONPATH=. python tools/gen_qrss_live_pins.py            # rewrites the file
    PYTHONPATH=. python tools/gen_qrss_live_pins.py --check    # exit 1 if stale
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sstvae.qrss import live  # noqa: E402

OUT = ROOT / "native" / "tests" / "golden_qrss" / "live_pins.json"
INT64 = range(-2 ** 63, 2 ** 63)


def line(t, hz, **kw) -> str:
    return json.dumps({"t": t, "dial_hz": hz, **kw})


# Lines a dial file may hold that are not a plain change: skipped, or taken
# as float() and round() take them.
ODD = [
    "not json", "[1, 2]", '"x"', "null", "{}", '{"t": 1}', '{"dial_hz": 5}', '{"t": "x", "dial_hz": 1}',
    '{"t": "12.5", "dial_hz": "14070000"}', '{"t": " 13 ", "dial_hz": "1_400"}', '{"t": true, "dial_hz": 14070000}',
    '{"t": 14, "dial_hz": false}', '{"t": NaN, "dial_hz": 1}', '{"t": Infinity, "dial_hz": 1}',
    '{"t": 15, "dial_hz": NaN}', '{"t": 16, "dial_hz": -3}', '{"t": 17, "dial_hz": 0.4}', '{"t": 18, "dial_hz": 0.5}',
    '{"t": 19, "dial_hz": 2.5}', '{"t": 20, "dial_hz": 3.5}', '{"t": 21, "dial_hz": 0}',
    '{"t": 22, "dial_hz": 14070000, "x": [1, {"y": null}]}', '{"t": 1e400, "dial_hz": 3}',
    '{"t": "inf", "dial_hz": 3}', '{"t": "nan", "dial_hz": 3}', '{"t": 23, "dial_hz": "nan"}',
    '{"t": 1, "t": 24, "dial_hz": 5}', '{"t": 1E3, "dial_hz": 7}', '{"t": -0.0, "dial_hz": 7}',
    '   {"t": 25, "dial_hz": 9}   ', "", "  ", '{"t": 26, "dial_hz": 9.0}', '{"t": 26, "dial_hz": 10}',
    '{"t": 123456789012345678901234567890, "dial_hz": 11}', '{"t": 27, "dial_hz": 123456789012}',
    '{"t": [27], "dial_hz": 5}', '{"t": {"a": 1}, "dial_hz": 5}', '{"t": 28, "dial_hz": "0x10"}',
    '{"t": 29, "dial_hz": "1e3"}', '{"t": "\\u0031\\u0032", "dial_hz": 5}', '{"t": 30, "dial_hz": 6',
    '{"t": 31, "dial_hz": 12, "seq": 123456789012345678901234567890}',
    '{"t": -123456789012345678901234567890, "dial_hz": 13}', '{"t": 32, "dial_hz": 14, "s": "-99999999999999999999"}',
    '{"t": 33, "dial_hz": 15, "a": [9223372036854775807, -9223372036854775808, 9223372036854775808]}',
    '{"t": 34.5e1, "dial_hz": 16, "b": -0, "c": 1.5e+3}',
    # an infinite dial, a time past float's range: OverflowError, skipped
    '{"t": 35, "dial_hz": 1e400}', '{"t": 36, "dial_hz": "-Infinity"}', '{"t": 1' + "0" * 400 + ', "dial_hz": 17}',
]

# (name, bytes, skips): skips are the lines (as text) the port skips and the
# reference keeps, a dial past int64.
DIALS = [
    ("mixed",
     b'{"t": 10, "dial_hz": 7038600}\n{"t": 5, "dial_hz": 7038600}\ngarbage\n'
     b'{"t": 20, "dial_hz": 14095600.5}\n{"t": 21, "dial_hz": 14095601.5}\n'
     b'{"t": "30", "dial_hz": "10138700"}\n{"t": 40, "dial_hz": 0}\n{"t": NaN, "dial_hz": 5}\n'
     b'{"t": 50, "dial_hz": true}\n{"t": 60, "dial_hz": 10138700}\n[1, 2]\n{"dial_hz": 1}\n{"t": 70}\n'
     b'{"t": 123456789012345678901234567890, "dial_hz": 3}\n'
     b'{"t": 80, "dial_hz": 4, "x": 123456789012345678901234567890}\n{"t": 90, "dial_hz": 5\r\n'
     b'{"t": 95, "dial_hz": 6}\xc2\x85{"t": 96, "dial_hz": 7}\xe2\x80\xa8{"t": 97, "dial_hz": 8}\n'
     b'{"t": 98, "dial_hz": 9}\xff\n', []),
    ("odd lines", ("\n".join([line(100.0, 10_141_000), *ODD, line(200, 10_140_900), line(150.5, 10_141_000),
                              line(150.5, 14_070_000)]) + "\n").encode(), []),
    ("crlf", ("\r\n".join([line(1, 5), line(2, 6), line(3, 6), line(4, 5)]) + "\r\n").encode(), []),
    ("separators",
     "﻿{\"t\": 1, \"dial_hz\": 5}\x0b{\"t\": 2, \"dial_hz\": 6}\x0c{\"t\": 3, \"dial_hz\": 7}\x1c"
     "{\"t\": 4, \"dial_hz\": 8}\x1d{\"t\": 5, \"dial_hz\": 9}\x1e{\"t\": 6, \"dial_hz\": 10}\x85"
     "{\"t\": 7, \"dial_hz\": 11} {\"t\": 8, \"dial_hz\": 12} {\"t\": 9, \"dial_hz\": 13}\r"
     "{\"t\": 10, \"dial_hz\": 14}".encode(), []),
    ("invalid utf-8",
     b'{"t": 1, "dial_hz": 5}\n\xff\xfe{"t": 2, "dial_hz": 6}\n{"t": 3, "dial_hz": 7, "c": "\xe9"}\n'
     b'{"t": 4, "dial_hz": 8, "c": "\xed\xa0\x80"}\n{"t": 5, "dial\xc0\x80": 9}\n{"t": 6, "dial_hz": 10}\n'
     b'{"t": 7, "dial_hz": 11, "c": "\xe2\x82"}\n{"t": 8, "dial_hz": 12, "c": "\xf0\x9f\x98\x80"}', []),
    ("partial", ("\n".join([line(1, 5), line(2, 6)]) + "\n").encode() + b'{"t": 3, "dial_', []),
    ("unsorted", ("\n".join(line(t, hz) for t, hz in ((5, 1), (3, 2), (5, 3), (4, 2), (1, 2), (5, 1)))).encode(), []),
    ("a big int beside a string",
     b'{"s": "\\"12345678901234567890123", "t": 1e1, "dial_hz": 2, "n": -98765432109876543210}', []),
    ("dials past int64",
     ("\n".join([line(1, 3), '{"t": 5, "dial_hz": 123456789012345678901234567890}', line(6, 1e30),
                 line(7, 9223372036854775807), line(8, 3), line(9, 9223372036854774784),
                 line(10, 99999999999999999999.5)]) + "\n").encode(),
     ['{"t": 5, "dial_hz": 123456789012345678901234567890}', line(6, 1e30), line(7, 9223372036854775807),
      line(10, 99999999999999999999.5)]),
    ("empty", b"", []),
]

QUERIES = {"spans": [(0.0, 1e30), (-math.inf, math.inf), (20.0, 30.0), (96.5, 96.5), (1e31, 1e32)],
           "span_at": [0.0, 25.0, 96.5, 97.0, 1e30, -math.inf, math.inf]}


def points(data: bytes) -> list:
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / live.DIAL_FILE
        p.write_bytes(data)
        log = live.DialLog(p)
        log.refresh()
        return log, [[t, hz] for t, hz in log.points]


def dial_pins() -> list:
    out = []
    for i, (name, data, skips) in enumerate(DIALS):
        log, pts = points(data)
        past = [p for p in pts if p[1] not in INT64]
        if skips:
            assert past, name
            kept = b"\n".join(x for x in data.split(b"\n") if x.decode() not in skips)
            log, pts = points(kept)
        assert all(p[1] in INT64 for p in pts), (name, pts)
        assert not past or skips, (name, past)
        pin = {"name": name, "hex": data.hex(), "points": pts}
        if i == 0:
            pin["spans"] = [{"a": a, "b": b, "spans": [list(s) for s in log.spans(a, b)],
                             "by_dial": [[dial, [list(s) for s in spans]] for dial, spans in log.by_dial(a, b)]}
                            for a, b in QUERIES["spans"]]
            pin["span_at"] = [{"t": t, "span": list(log.span_at(t))} for t in QUERIES["span_at"]]
        out.append(pin)
    return out


def tiles():
    t = live.Tile
    full = t(id="1954752-1500", q=1954752, slot_utc="2025-10-01T00:00:00Z", frame="l-tiny", f_hz=1509.5,
             dial_hz=10_140_000, status="complete", snr_db=-12.25, z_ref=17.5, callsign="K1ABC/P",
             grid="FN42", picture_id="1234abcd", mode="B", segment=1, cw_text="K1ABC", progress=1.0, heard=0.875,
             received=0.5, mean_w_db=-3.125, confidence=[None, -1.5, 2.0, None, 1e-300, 0.1], passes=2,
             image="tiles/x.png", image_rev=3, updated=1759276800.123, note="stored (corr)", uid="u_1",
             waveform="L")
    yield "defaults", t(id="a", q=1, slot_utc="", frame="tiny", f_hz=1500.0)
    yield "every field", full
    yield "a receiving tile", t(id="a", q=7, slot_utc="2026-10-09T06:00Z", frame="tiny", f_hz=1234.5,
                                dial_hz=14095600, snr_db=-3.25, z_ref=8.0, callsign="N0CALL", mode="B", segment=2,
                                progress=0.5, heard=1.0, received=0.25, mean_w_db=-1.5,
                                confidence=[None, 1.5, -2.0], passes=2, image="x.png", image_rev=3,
                                updated=1760000000.125, note="n", uid="u", waveform="L")
    yield "text and odd floats", dataclasses.replace(
        full, note='quote " backslash \\ tab \t nl \n ctl \x01 \x7f é ü 日本 \U0001f600', callsign="", grid=None,
        dial_hz=None, q=-5, f_hz=-0.0, updated=0.1 + 0.2, progress=1 / 3, heard=2 ** -1074, received=1e16,
        mean_w_db=1e-7, snr_db=123456789.0)
    yield "big ints", dataclasses.replace(full, dial_hz=2 ** 62, segment=0, passes=0, image_rev=2 ** 53 + 1,
                                          confidence=[])
    yield "snr and z_ref not finite", dataclasses.replace(full, snr_db=math.nan, z_ref=math.inf)
    yield "every float not finite", dataclasses.replace(
        full, f_hz=math.nan, progress=-math.inf, heard=math.inf, received=math.nan, mean_w_db=-math.inf,
        updated=math.nan, confidence=[math.nan, None, math.inf, 1.0, -math.inf])


def tile_pins() -> list:
    return [{"name": name, "tile": dataclasses.asdict(t), "json": json.dumps(t.to_json())} for name, t in tiles()]


def build() -> str:
    pins = {"dial_guard_s": list(live.DIAL_GUARD_S), "dial": dial_pins(), "tiles": tile_pins()}
    return json.dumps(pins, indent=1) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--check", action="store_true", help="exit 1 if the committed file is stale")
    args = ap.parse_args()
    text = build()
    if args.check:
        if not OUT.exists() or OUT.read_text() != text:
            print(f"{OUT.relative_to(ROOT)} is stale: run tools/gen_qrss_live_pins.py", file=sys.stderr)
            return 1
        return 0
    tmp = OUT.with_name(OUT.name + ".tmp")
    tmp.write_text(text)
    tmp.replace(OUT)
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

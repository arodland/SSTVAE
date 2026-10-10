#!/usr/bin/env python3
"""Read-only check of a QRSSTVAE listener's passband store, for a slot whose
tile reads less than 100% although the whole pass seemed to be heard.

    python3 scripts/qrss_check_store.py                       # the last 3 hours, default store
    python3 scripts/qrss_check_store.py --slot 2026-10-10T20:30Z
    python3 scripts/qrss_check_store.py --store DIR --hours 6

Needs only Python 3 and numpy. Writes nothing. For each slot whose pass
window (t0 - 15 s to t0 + 1787.7 s, t0 = the quarter hour + 1 s) overlaps
the period it covers, it reports:

  - how much of the frame the store holds, and every hole in it (an audio
    stall the listener treated as missing audio shows as a hole of a
    second or more, often followed by "missing" / "ahead of the clock"
    lines in the listener's log);
  - int16 clipping in the stored stream (the store saturates at +-4 in
    front-end units), runs of exact zeros (audio written as silence), and
    one-second stretches far below the slot's median power (a dropout,
    the receive muted for a transmission, a tune);
  - the median level per minute, so an AGC jump, a tune or a band change
    stands out;
  - the dial changes the app logged in that slot, and the guard around
    each that the listener counts as unheard;
  - the tile the listener has for the slot (received, heard, SNR) and the
    log lines in state.json that bear on missing audio.
"""

import argparse
import json
import math
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

FRAME_S = 1782.7          # FULL frame
LEAD_S, TAIL_S = 15.0, 5.0
GUARD_S = (2.0, 30.0)     # the listener's dial-change guard (before, after)
HOUR_RE = re.compile(r"^fe_(\d+)\.i16$")


def utc(t: float) -> str:
    return datetime.fromtimestamp(t, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_utc(s: str) -> float:
    s = s.strip().replace("Z", "+00:00")
    d = datetime.fromisoformat(s)
    if d.tzinfo is None:
        d = d.replace(tzinfo=timezone.utc)
    return d.timestamp()


class Store:
    def __init__(self, root: Path):
        self.root = root
        m = json.loads((root / "meta.json").read_text())
        self.fs, self.scale = int(m["fs"]), float(m["scale"])
        self.per_hour = 3600 * self.fs

    def hours(self) -> list[int]:
        return sorted(int(m.group(1)) for p in self.root.iterdir()
                      if (m := HOUR_RE.match(p.name)))

    def ranges(self, hour: int) -> list[tuple[int, int]]:
        p = self.root / f"fe_{hour}.json"
        return [tuple(r) for r in json.loads(p.read_text())] if p.exists() else []

    def read(self, a: float, b: float):
        """(int16 I/Q (n, 2), written mask) for unix times [a, b)."""
        n0, n1 = int(round(a * self.fs)), int(round(b * self.fs))
        x = np.zeros((n1 - n0, 2), np.int16)
        ok = np.zeros(n1 - n0, bool)
        n = n0
        while n < n1:
            hour, k = divmod(n, self.per_hour)
            m = min(n1 - n, self.per_hour - k)
            p = self.root / f"fe_{hour}.i16"
            if p.exists():
                mm = np.memmap(p, dtype=np.int16, mode="r", shape=(self.per_hour, 2))
                for lo, hi in self.ranges(hour):
                    lo, hi = max(lo, k), min(hi, k + m)
                    if lo < hi:
                        x[n - n0 + lo - k:n - n0 + hi - k] = mm[lo:hi]
                        ok[n - n0 + lo - k:n - n0 + hi - k] = True
                del mm
            n += m
        return x, ok


def runs(mask: np.ndarray):
    """(start, end) index pairs of the True runs of mask."""
    m = np.concatenate(([0], mask.astype(np.int8), [0]))
    d = np.diff(m)
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def dial_points(path: Path):
    pts = []
    if path.exists():
        for line in path.read_text(errors="replace").splitlines():
            try:
                d = json.loads(line)
                pts.append((float(d["t"]), int(round(float(d["dial_hz"]))), d.get("why")))
            except (ValueError, KeyError, TypeError):
                continue
    return sorted(pts)


def check_slot(st: Store, q: int, state: dict | None, dials, out, quiet: bool = False) -> None:
    fs = st.fs
    t0 = 900 * q + 1
    a, b = t0 - LEAD_S, t0 + FRAME_S + TAIL_S
    x, ok = st.read(a, b)
    org = int(round(LEAD_S * fs))
    fr = slice(org, org + int(round(FRAME_S * fs)))
    okf = ok[fr]
    if quiet and not ok.any():
        return
    print(f"\n=== slot {utc(t0 - 1)}  (t0 {utc(t0)}, frame ends {utc(t0 + FRAME_S)})", file=out)
    if not ok.any():
        print("  nothing stored", file=out)
        return
    print(f"  frame stored: {100 * okf.mean():.2f}%  "
          f"({okf.sum() / fs:.1f} s of {FRAME_S:.1f} s)", file=out)
    holes = [(s, e) for s, e in runs(~ok)]
    if holes:
        print(f"  holes ({len(holes)}, {sum(e - s for s, e in holes) / fs:.1f} s in all; "
              "seconds after t0):", file=out)
        for s, e in holes[:60]:
            print(f"    {(s - org) / fs:9.2f} .. {(e - org) / fs:9.2f}  "
                  f"{(e - s) / fs:7.2f} s   {utc(a + s / fs)}", file=out)
        if len(holes) > 60:
            print(f"    ... {len(holes) - 60} more", file=out)
    else:
        print("  no holes", file=out)
    xi = x[ok].astype(np.int32)
    clip = int(np.sum(np.abs(xi) >= 32767))
    print(f"  clipped I/Q values: {clip} of {2 * len(xi)}"
          + (f" ({100 * clip / (2 * len(xi)):.3f}%)" if clip else ""), file=out)
    zero = ok & (x[:, 0] == 0) & (x[:, 1] == 0)
    zr = [(s, e) for s, e in runs(zero) if e - s >= fs // 10]
    if zr:
        print(f"  exact-zero runs of 0.1 s or more: {len(zr)}, "
              f"{sum(e - s for s, e in zr) / fs:.1f} s", file=out)
        for s, e in zr[:20]:
            print(f"    {(s - org) / fs:9.2f} .. {(e - org) / fs:9.2f}", file=out)
    p = (x[:, 0].astype(np.float64) ** 2 + x[:, 1].astype(np.float64) ** 2) / st.scale ** 2
    sec = len(p) // fs
    ps = p[:sec * fs].reshape(sec, fs).mean(axis=1)
    oks = ok[:sec * fs].reshape(sec, fs).all(axis=1)
    if oks.any():
        med = float(np.median(ps[oks]))
        low = oks & (ps < 1e-3 * med)
        if low.any():
            print(f"  seconds 30 dB or more below the median power: {int(low.sum())}", file=out)
            for s, e in runs(low)[:20]:
                print(f"    {s - LEAD_S:9.0f} .. {e - LEAD_S:9.0f} s", file=out)
        mins = []
        for m0 in range(0, sec, 60):
            sl = slice(m0, min(sec, m0 + 60))
            if oks[sl].any():
                mins.append(f"{10 * math.log10(max(float(np.median(ps[sl][oks[sl]])), 1e-12)):6.1f}")
            else:
                mins.append("   ---")
        print(f"  median power per minute (dB re 1 FE unit, slot median "
              f"{10 * math.log10(max(med, 1e-12)):.1f}):", file=out)
        for i in range(0, len(mins), 10):
            print("    " + " ".join(mins[i:i + 10]), file=out)
    ch = [(t, hz, why) for t, hz, why in dials if a - 3600 <= t <= b]
    if ch:
        print("  dial log:", file=out)
        for t, hz, why in ch:
            inside = a <= t <= b
            print(f"    {utc(t)} ({t - t0:+8.1f} s) {hz / 1e6:.6f} MHz {why or ''}"
                  + ("" if inside else "  (before the slot)"), file=out)
        changes = [(t, hz) for i, (t, hz, _) in enumerate(dials)
                   if i and dials[i - 1][1] != hz and a <= t <= b]
        if changes:
            lost = sum(min(b, t + GUARD_S[1]) - max(a, t - GUARD_S[0]) for t, _ in changes)
            print(f"  {len(changes)} dial change(s) in the slot: about {lost:.0f} s "
                  "counted as unheard by the guard", file=out)
    if state:
        for t in state.get("tiles", []):
            if int(t.get("q", -1)) == q:
                print(f"  tile {t.get('id')}: {t.get('status')}, f {t.get('f_hz'):.1f} Hz, "
                      f"dial {t.get('dial_hz')}, received {100 * (t.get('received') or 0):.1f}% "
                      f"of latents, heard {100 * (t.get('heard') or 0):.1f}%, "
                      f"SNR {t.get('snr_db')}, {t.get('callsign')}", file=out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    home = os.environ.get("QRSSTVAE_HOME")
    root = Path(home) if home else Path.home() / ".local" / "share" / "qrsstvae"
    ap.add_argument("--store", type=Path, default=root,
                    help=f"the listener's store (default {root})")
    ap.add_argument("--passband", type=Path, default=None, help="default STORE/passband")
    ap.add_argument("--state", type=Path, default=None, help="default STORE/live")
    ap.add_argument("--hours", type=float, default=3.0, help="look back this far (default 3)")
    ap.add_argument("--slot", default=None, help="only this slot (its quarter hour, UTC)")
    args = ap.parse_args()
    pb = args.passband or args.store / "passband"
    live = args.state or args.store / "live"
    if not (pb / "meta.json").exists():
        sys.exit(f"no passband store in {pb}")
    st = Store(pb)
    print(f"passband store {pb}: {st.fs} Hz I/Q int16, scale {st.scale:g}")
    hs = st.hours()
    for h in hs:
        rs = st.ranges(h)
        got = sum(hi - lo for lo, hi in rs) / st.fs
        size = (pb / f"fe_{h}.i16").stat()
        print(f"  {utc(3600 * h)}  {got:7.1f} s written in {len(rs)} range(s), "
              f"{size.st_size / 1e6:.1f} MB apparent, "
              f"{getattr(size, 'st_blocks', 0) * 512 / 1e6:.1f} MB on disk")
    state = None
    try:
        state = json.loads((live / "state.json").read_text())
    except (OSError, ValueError):
        print(f"no state.json in {live}")
    dials = dial_points(live / "dial_changes.jsonl")
    print(f"dial log: {len(dials)} line(s)" if dials else "dial log: none")
    if args.slot:
        qs = [int(math.floor((parse_utc(args.slot) + 1) / 900.0))]
    else:
        now = time.time()
        qs = range(int((now - args.hours * 3600 - FRAME_S) // 900),
                   int((now - FRAME_S - TAIL_S) // 900) + 1)
    for q in qs:
        check_slot(st, q, state, dials, sys.stdout, quiet=not args.slot)
    if state:
        keys = ("missing", "ahead of the clock", "restarts", "stopped", "listening",
                "read back", "frame over", "heard")
        lines = [s for s in state.get("log", []) if any(k in s for k in keys)]
        if lines:
            print("\nlistener log (the last lines state.json keeps):")
            for s in lines:
                print("  " + s)


if __name__ == "__main__":
    main()

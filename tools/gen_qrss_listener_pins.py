#!/usr/bin/env python3
"""Writes native/tests/golden_qrss/listener_pins.json: scripted stub
scenarios run through sstvae/qrss/live.py's LiveListener, for
native/tests/qrss/test_listener.cpp to replay through the C++ port
(native/core/qrss/listener) without Python.

Each scenario is data the C++ replays as it is: the listener's
configuration, a state.json and a dial file to start from, the audio fed
(each chunk a formula, so no random numbers cross: sample k of a chunk of
seed s is 1e-3 ((h(k + 7919 s) / 2^32) - 0.5) with h(i) = i 2654435761 mod
2^32, exact in float64 on both sides), the steps, and what each stub
receiver answers, call by call (candidates, detect, the live receive_slot
and finish_receive, each a list of answers consumed in order, the last
repeated). The reference's listener runs with those stubs in place of
receiver.candidates / .detect / .receive_slot and live.finish_receive,
with time.time() frozen at CLOCK; what is pinned is what it did:

- every stub call with its arguments (detect's candidates and skipped
  centres, the live receive's dets and erased spans, finish_receive's
  share of the capture heard, origin, dets and hints);
- each step's result, and after the script the log lines, the tiles
  (Tile.to_json), state.json's and listener.log's text, byte for byte,
  and the tiles' pictures (size, and the values their pixels take: the
  fake codec's pictures are one value each);
- for "restart" scenarios, a second listener on the same directory: the
  tiles it reads back, and those left after a step past keep_done_s.

    PYTHONPATH=. python tools/gen_qrss_listener_pins.py     # rewrites the file
    PYTHONPATH=. python tools/gen_qrss_listener_pins.py --check
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sstvae.qrss import frame, live  # noqa: E402
from sstvae.qrss import receiver as RX  # noqa: E402
from sstvae.qrss.constants import FS, WAVEFORM_CE, WAVEFORM_L  # noqa: E402
from sstvae.qrss.frontend import PB_TAIL_S, slot_t0  # noqa: E402

OUT = ROOT / "native" / "tests" / "golden_qrss" / "listener_pins.json"
CLOCK = 1_760_000_000.0
Q = 1954752                      # tests' Q_TEST: 2025-10-01T00:00:00Z
T0 = float(slot_t0(Q))
TINY_S = live.frame_seconds(frame.TINY)
END = T0 + TINY_S + PB_TAIL_S + 0.5


def audio(n: int, seed: int) -> np.ndarray:
    i = np.arange(n, dtype=np.uint64) + np.uint64(7919 * seed)
    h = (i * np.uint64(2654435761)) % np.uint64(2 ** 32)
    return 1e-3 * (h.astype(np.float64) / 4294967296.0 - 0.5)


def det(f, waveform=None, z=None, method=None):
    d = {"f_hz": f}
    if waveform is not None:
        d.update(waveform=waveform, z_ref=z if z is not None else 10.0, method=method or "preamble")
    return d


def pas(f, z=20.0, snr=-12.0, n=frame.TINY.n_data, header=None, waveform=None, uid=None, cw=None):
    p = {"f_hz": f, "z_ref": z, "snr": snr, "n": n, "uid": uid or f"u{f}"}
    if header is not None:
        p["header"] = header
    if waveform is not None:
        p["waveform"] = waveform
    if cw is not None:
        p["cw"] = cw
    return p


HDR = dict(callsign="AG7EW", grid="CN85", picture_id=0x621AC873, mode=1, segment=0, codec_id=0xD1D8)

SCENARIOS = [
    dict(name="bookkeeping",
         cfg=dict(spec="tiny", refresh_s=10.0, first_s=25.0),
         feeds=[[76 * FS, 1, T0 + 60.0]],
         steps=[T0 - 20.0, T0 + 30.0, T0 + 35.0, T0 + 41.0, END],
         candidates=[[det(1500.0), det(1610.0)]],
         detect=[[det(1500.0), det(1610.0)]],
         receive=[[pas(1500.3), pas(1611.0)], [pas(1501.9), pas(1609.5)]],
         finish=[[[pas(1500.0), pas(1610.0)], 1.0]],
         restart=True),
    dict(name="recheck",
         cfg=dict(spec="tiny", refresh_s=10.0, first_s=25.0, detect_until_s=30.0, recheck_until_s=45.0),
         feeds=[[76 * FS, 2, T0 + 60.0]],
         steps=[T0 + 26.0, T0 + 36.0, T0 + 46.0, END],
         candidates=[[det(1500.0), det(1600.0)]],
         detect=[[det(1500.0)], [det(1600.0)], []],
         receive=[[pas(1500.0)], [pas(1500.0), pas(1600.0)]],
         finish=[[[pas(1500.0), pas(1600.0, header=HDR)], 1.0]]),
    dict(name="late_join_hint",
         cfg=dict(spec="tiny"),
         state={"tiles": [dict(id=f"p{Q - 2}_1600.0", q=Q - 2, slot_utc="", frame="tiny", f_hz=1600.0,
                               status="complete", mode="B", segment=0, callsign="KC2G",
                               updated=float(slot_t0(Q)))]},
         feeds=[[int((TINY_S + 5 - 0.65 * TINY_S) * FS), 3, T0 + TINY_S + 5]],
         steps=[T0 + 0.65 * TINY_S, END],
         candidates=[[]], detect=[[]], receive=[[]],
         finish=[[[pas(1600.0, header=dict(HDR, callsign="KC2G", segment=1))], 0.35]]),
    dict(name="two_dials",
         cfg=dict(spec="tiny"),
         dial_guard=[2.0, 2.0],
         dial_file=(json.dumps({"t": T0 - 600.0, "dial_hz": 10_138_700, "why": "start"}) + "\n"
                    + json.dumps({"t": T0 + 20.0, "dial_hz": 14_095_600, "why": "schedule"}) + "\n"),
         feeds=[[int((END + 4.5 - (T0 - 16.0)) * FS), 4, END + 4.5]],
         steps=[T0 + 30.0, END],
         candidates=[[det(1500.0)]], detect=[[det(1500.0)]],
         receive=[[pas(1500.0)]],
         finish=[[[pas(1500.0)], 0.4], [[pas(1500.0)], 0.6]]),
    dict(name="waveform_switch",
         cfg=dict(spec="tiny", waveforms=[WAVEFORM_CE, WAVEFORM_L], refresh_s=10.0),
         feeds=[[140 * FS, 5, T0 + 125.0]],
         steps=[T0 + 26.0, T0 + 36.0, T0 + 46.0, T0 + 56.0],
         candidates=[[]], detect=[[det(1500.0, WAVEFORM_CE, 10.0)]],
         receive=[[pas(1500.0, waveform=WAVEFORM_CE, z=10.0), pas(1560.0, waveform=WAVEFORM_CE, z=9.0)],
                  [pas(1509.5, waveform=WAVEFORM_L, z=20.0)],
                  [pas(1500.2, waveform=WAVEFORM_CE, z=15.0)],
                  [pas(1500.1, waveform=WAVEFORM_CE, z=12.0)]],
         finish=[[[pas(1500.1, waveform=WAVEFORM_CE, z=12.0)], 1.0]]),
    dict(name="ghost_and_snr_cut",
         cfg=dict(spec="tiny", refresh_s=10.0),
         feeds=[[76 * FS, 6, T0 + 60.0]],
         steps=[T0 + 26.0, T0 + 36.0, END],
         candidates=[[]], detect=[[det(1500.0)]],
         receive=[[pas(1500.0, header=HDR), pas(1700.0, snr=3.1), pas(5000.0), pas(250.0),
                   pas(1800.0, snr=2.0, cw="AG7EW"), pas(1900.0, snr=-5.9)],
                  [pas(1500.4), pas(1700.2, snr=-6.0)]],
         finish=[[[pas(1500.0), pas(1700.0, snr=-6.5)], 1.0]]),
    dict(name="provisional_picture",
         cfg=dict(spec="tiny", refresh_s=10.0, render=True),
         feeds=[[76 * FS, 7, T0 + 60.0]],
         steps=[T0 + 26.0, T0 + 36.0, T0 + 46.0, END],
         candidates=[[]], detect=[[det(1500.0)]],
         receive=[[pas(1500.0, n=-1)], [pas(1500.0, n=-1, header=dict(HDR, segment=1))],
                  [pas(1500.0, n=-1)]],
         finish=[[[pas(1500.0, n=-1)], 1.0]]),
    # a later pass due (an earlier slot's tile), through the live refresh's
    # acquisition: said once, the hints passed on, CE and joint
    *(dict(name=f"due_hint_{name}",
           cfg=dict(spec="tiny", refresh_s=5.0, **({} if ws is None else {"waveforms": ws})),
           state={"tiles": [dict(id=f"p{Q - 2 * k}_{f}", q=Q - 2 * k, slot_utc="", frame="tiny", f_hz=f,
                                 status="complete", mode=m, segment=0, callsign="KC2G",
                                 updated=float(slot_t0(Q)))
                            for k, f, m in ((1, 1600.0, "B"), (2, 1720.0, "C"))]},
           feeds=[[66 * FS, 8, T0 + 50.0]],
           steps=[T0 + 26.0, T0 + 31.0, T0 + 36.0, END],
           candidates=[[det(1500.0, wf), det(1600.0, wf, method="hint"), det(1720.0, wf)]],
           detect=[[det(1500.0, wf), det(1720.0, wf)]],
           receive=[[pas(1500.0, waveform=wf), pas(1720.0, waveform=wf, header=dict(HDR, segment=2))]],
           finish=[[[pas(1500.0, waveform=wf), pas(1600.0, waveform=wf)], 1.0]])
      for name, ws, wf in (("ce", None, None), ("joint", [WAVEFORM_CE, WAVEFORM_L], WAVEFORM_CE))),
]


def _ns_header(h):
    return None if h is None else SimpleNamespace(**h)


class _Det:
    def __init__(self, d):
        self.f_hz = d["f_hz"]
        for k in ("waveform", "z_ref", "method"):
            if k in d:
                setattr(self, k, d[k])


class _Pass:
    def __init__(self, d):
        from sstvae.qrss import picture

        n = picture.SENT if d["n"] == -1 else d["n"]
        self.f_hz, self.uid = d["f_hz"], d["uid"]
        self.report = SimpleNamespace(snr2500_db=d["snr"], z_ref=d["z_ref"], suspect=False)
        self.header = _ns_header(d.get("header"))
        self.cw = SimpleNamespace(text=d["cw"]) if "cw" in d else None
        if "waveform" in d:
            self.waveform = d["waveform"]
            self.frame = "l-tiny" if d["waveform"] == WAVEFORM_L else "tiny"
        self.w = np.ones(n, dtype=np.float32)
        self.z = (np.arange(n, dtype=np.float32) % 7 - 3) / 4


class _Codec:
    """A fake codec: picture k is 8x6 of value 37 k + (sum of the weight plane) % 200."""

    def __init__(self):
        self.k = 0

    def decode(self, lat, wt):
        from PIL import Image

        v = np.full((6, 8, 3), (37 * self.k + int(np.asarray(wt, dtype=np.float64).sum()) % 200) % 256,
                    np.uint8)
        self.k += 1
        return Image.fromarray(v)


def _pictures(d: Path) -> dict:
    """tiles/*.png: {name: [width, height, the values its pixels take]}."""
    from PIL import Image

    out = {}
    for p in sorted((d / "tiles").glob("*.png")):
        with Image.open(p) as im:
            a = np.asarray(im.convert("RGB"))
        out[p.name] = [int(a.shape[1]), int(a.shape[0]), [int(v) for v in np.unique(a)]]
    return out


def run(sc: dict, root: Path) -> dict:
    calls = []
    queues = {k: list(sc[k]) for k in ("candidates", "detect", "receive", "finish")}

    def take(k):
        q = queues[k]
        return q.pop(0) if len(q) > 1 else q[0]

    def candidates(prep, spec, live_only=False, hints=()):
        calls.append(["candidates", prep.q, len(prep.fe), [list(h) if isinstance(h, tuple) else h
                                                            for h in hints]])
        return [_Det(d) for d in take("candidates")]

    def detect(prep, spec, live_only=False, *, erase_s=(), cands=None, skip_hz=(), **kw):
        calls.append(["detect", prep.q, len(prep.fe), live_only, [[float(a), float(b)] for a, b in erase_s],
                      None if cands is None else [c.f_hz for c in cands], [float(f) for f in skip_hz]])
        return [_Det(d) for d in take("detect")]

    def receive_slot(prep, spec, *, estimator="joint", erase_s=(), round_b=True, dets=None, **kw):
        calls.append(["receive", prep.q, len(prep.fe), estimator, round_b,
                      [[float(a), float(b)] for a, b in erase_s], [d.f_hz for d in dets]])
        return [_Pass(p) for p in take("receive")]

    def finish_receive(x, ok, origin, q, spec, estimator, dets, hints=()):
        calls.append(["finish", q, len(x), int(np.count_nonzero(ok)), int(origin), estimator,
                      [d.f_hz for d in dets], [list(h) if isinstance(h, tuple) else h for h in hints]])
        passes, heard = take("finish")
        return [_Pass(p) for p in passes], heard

    saved = (RX.candidates, RX.detect, RX.receive_slot, live.finish_receive, live.time, live.DIAL_GUARD_S)
    RX.candidates, RX.detect, RX.receive_slot, live.finish_receive = (candidates, detect, receive_slot,
                                                                       finish_receive)
    live.time = SimpleNamespace(time=lambda: CLOCK, sleep=lambda s: None)
    if "dial_guard" in sc:
        live.DIAL_GUARD_S = tuple(sc["dial_guard"])
    try:
        d = root / sc["name"]
        d.mkdir(parents=True)
        if "state" in sc:
            (d / "state.json").write_text(json.dumps(sc["state"]))
        if "dial_file" in sc:
            (d / live.DIAL_FILE).write_text(sc["dial_file"])
        c = dict(sc["cfg"])
        c["spec"] = frame.get(c["spec"])
        if "waveforms" in c:
            c["waveforms"] = tuple(c["waveforms"])
        c.setdefault("render", False)
        cfg = live.LiveConfig(**c)
        sink = []
        L = live.LiveListener(d, cfg, log=sink.append)
        if cfg.render:
            L.codec = _Codec()
        for n, seed, t_end in sc["feeds"]:
            L.feed(audio(n, seed), t_end)
        steps = [L.step(t) for t in sc["steps"]]
        L.write_state()
        log = d / live.LOG_FILE
        out = dict(calls=calls, steps=steps, sink=sink, lines=L.lines,
                   tiles=json.dumps([t.to_json() for t in L.tiles()]),
                   state=(d / "state.json").read_text(), finished=sorted(L.finished),
                   log=log.read_text() if log.exists() else None, pictures=_pictures(d))
        if sc.get("restart"):
            L2 = live.LiveListener(d, cfg, log=lambda m: None)
            out["restart_tiles"] = json.dumps([t.to_json() for t in L2.tiles()])
            L2.step(L2.tiles()[0].updated + cfg.keep_done_s + 1)
            out["expired_tiles"] = json.dumps([t.to_json() for t in L2.tiles()])
        return out
    finally:
        (RX.candidates, RX.detect, RX.receive_slot, live.finish_receive, live.time,
         live.DIAL_GUARD_S) = saved


def build() -> str:
    with tempfile.TemporaryDirectory() as tmp:
        pins = dict(clock=CLOCK, q=Q, scenarios=[])
        for sc in SCENARIOS:
            pins["scenarios"].append(dict(sc, expect=run(sc, Path(tmp))))
    return json.dumps(pins, indent=1) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="exit 1 if the file is stale")
    a = ap.parse_args()
    text = build()
    if a.check:
        ok = OUT.exists() and OUT.read_text() == text
        print("up to date" if ok else f"{OUT} is stale")
        sys.exit(0 if ok else 1)
    OUT.write_text(text)
    print(f"wrote {OUT} ({len(text)} bytes)")


if __name__ == "__main__":
    main()

"""Parity of native/core/qrss/listener (sstvae_native.qrss.listener) with
sstvae/qrss/live.py's LiveListener and run(), side by side: no row
replaces the reference's listener, so each scenario runs once through
each and the two are compared (tests/qrss_native/_listener.py builds the
C++ one the way a test builds the reference's).

The contract (the bar):

- stub scenarios, after test_qrss_live.py and test_qrss_live_l.py, with the
  receivers replaced as those tests replace them (the C++ listener's seams
  look the module's functions up at each call, and its end-of-slot receive
  is the C++ finish_receive's decisions over them): every receiver call
  with its arguments, the step results, the tiles, the log lines,
  state.json byte for byte (the clock frozen for both: time.time() is
  only ever a stamp), listener.log, the tiles' pictures pixel for pixel,
  the store tree (_store.compare_trees) and the passband files. Covered:
  merging by band centre across refreshes and the end of the slot, the
  waveform switch of one signal, ghosts of a neighbouring slot, the
  headerless SNR cut, hints moved by a USB dial change, a slot heard on two
  dials, a refused preamble checked again and acquisition kept once final,
  a late join's hint search, a later pass due (said once, the hints
  passed to acquisition) and a store whose due passes or association
  raise (the text logged or noted is str(e)), recover and backfill from
  the passband store, provisional pictures and the redraw a header
  brings, resync and early audio, digital silence as holes, and run()'s
  replay to its end;
- an exception in a live refresh: the C++ logs it once per slot and
  message and retries the slot (its `done` stays false), as the
  reference's worker does; the reference's synchronous listener raises it
  out of step();
- the real receiver on TINY CE and TINY L slots heard half and then whole
  (the C++ listener on the C++ receiver, the reference on its own): one
  tile each, f_hz within 0.1 Hz, snr_db within 0.3 dB, callsign, picture
  ID, segment, status, waveform and frame equal (the half-heard pass's
  progress and heard equal);
- with SSTVAE's codec built (native/build-qrss-codec), the C++ renderer's
  tile pictures equal the reference's pixel for pixel.

native/tests/qrss/test_listener.cpp pins key scenarios without Python
(tools/gen_qrss_listener_pins.py), with threads.
"""

from __future__ import annotations

import json
import math
import os
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from qrss_helpers import Q_TEST, unit_rms_latents
from qrss_native import _listener, _store, require

import sstvae.qrss.associate as assoc
from sstvae.qrss import channel as chm
from sstvae.qrss import frame, frontend, lframe, live, picture, precoder
from sstvae.qrss import receiver as RX
from sstvae.qrss.channel import ChannelConfig
from sstvae.qrss.constants import FE_FS, FS, WAVEFORM_CE, WAVEFORM_L
from sstvae.qrss.frontend import PB_LEAD_S, PB_TAIL_S, slot_t0

CE, L_ = WAVEFORM_CE, WAVEFORM_L
TINY, L_TINY = frame.TINY, lframe.L_TINY
CLOCK = 1_760_000_000.0          # time.time() for both listeners (a stamp, never a decision)


@pytest.fixture(scope="module")
def LN(native):
    return require(native, "listener")


def _hiss(n):
    """n samples of a quiet band: faint, but heard (exact zeros are not)."""
    return 1e-3 * np.random.default_rng(n).standard_normal(n)


class _Det:
    def __init__(self, f, waveform=None, z=None, method=None):
        self.f_hz = f
        if waveform is not None:
            self.waveform, self.z_ref, self.method = waveform, z, method or "preamble"


class _Report:
    def __init__(self, z, snr=-12.0):
        self.snr2500_db, self.z_ref, self.suspect = snr, z, False


class _Pass:
    def __init__(self, f, z=20.0, header=None, n=TINY.n_data, snr=-12.0, waveform=None, cw=None,
                 uid=None):
        self.f_hz, self.report, self.header, self.cw = f, _Report(z, snr), header, cw
        if waveform is not None:
            self.waveform, self.frame = waveform, "l-tiny" if waveform == L_ else "tiny"
        self.uid = uid or f"u{f}"
        self.w = np.ones(n, dtype=np.float32)
        self.z = np.zeros(n, dtype=np.float32)


def _hdr(callsign="AG7EW", picture_id=0x621AC873, mode=1, segment=0, codec_id=0xD1D8):
    return SimpleNamespace(callsign=callsign, grid="CN85", picture_id=picture_id, mode=mode,
                           segment=segment, codec_id=codec_id)


# --- running a scenario on both --------------------------------------------------------------


class Side:
    """One implementation's run of a scenario: builds its listeners (and
    stores) under its own directory, collects its log sink's lines."""

    def __init__(self, kind, native, root, mp):
        self.kind, self.native, self.root, self.mp = kind, native, root, mp
        self.sink: list[str] = []
        self.made = []

    @property
    def cpp(self):
        return self.kind == "cpp"

    def listener(self, sub="live", cfg=None, store=None, passband=None, codec=None, stubbed=True,
                 stub_associate=False):
        d = self.root / sub
        if self.cpp:
            r = None if codec is None else _listener.CodecRenderer(codec)
            L = _listener.listener(self.native, d, cfg, store, passband, self.sink.append,
                                   stubbed=stubbed, clock=lambda: CLOCK, renderer=r,
                                   stub_associate=stub_associate)
        else:
            L = live.LiveListener(d, cfg, store=store, passband=passband, log=self.sink.append)
            if codec is not None:
                L.codec = codec
        self.made.append(L)
        return L

    def store(self, sub="store", clock=None):
        if self.cpp:
            return _store.store_class(self.native.qrss)(self.root / sub, clock=clock or (lambda: CLOCK))
        from sstvae.qrss.store import Store

        return Store(self.root / sub, clock=clock or (lambda: CLOCK))

    def close(self):
        for L in self.made:
            L.close()


def both(native, tmp_path, monkeypatch, scenario):
    """scenario(side) on the reference and on the C++; their results."""
    _listener.frozen_time(monkeypatch, CLOCK)
    out = {}
    for kind in ("py", "cpp"):
        side = Side(kind, native, tmp_path / kind, monkeypatch)
        try:
            out[kind] = scenario(side)
        finally:
            side.close()
        out[kind + "_sink"] = side.sink
    return out


def snapshot(L, d):
    """Everything a listener shows: lines, tiles, finished, slots, state.json's
    text, listener.log and the tiles' pictures (pixels)."""
    from PIL import Image

    L.write_state()
    d = Path(d)
    log = d / live.LOG_FILE
    pics = {}
    for p in sorted((d / "tiles").glob("*.png")):
        with Image.open(p) as im:
            pics[p.name] = np.asarray(im.convert("RGB")).tobytes()
    return dict(lines=list(L.lines), tiles=[t.to_json() for t in L.tiles()],
                finished=sorted(L.finished), slots=sorted(L.slots), busy=L.busy,
                codec_error=L.codec_error, now=L.now, audio_end=L.audio_end,
                state=(d / "state.json").read_text(),
                log=log.read_text() if log.exists() else None, pictures=pics,
                stray=sorted(p.name for p in d.iterdir() if ".tmp" in p.name))


def same(a, b, what=""):
    """a == b, saying where they part."""
    if isinstance(a, dict) and isinstance(b, dict):
        assert list(a) == list(b), f"{what}: keys {list(a)} != {list(b)}"
        for k in a:
            same(a[k], b[k], f"{what}.{k}")
        return
    if isinstance(a, str) and isinstance(b, str) and a != b and a.startswith("{"):
        try:
            same(json.loads(a), json.loads(b), what + "(json)")
        except ValueError:
            pass
    if isinstance(a, list) and isinstance(b, list) and a != b:
        assert len(a) == len(b), f"{what}: {len(a)} items != {len(b)}\n{a}\n{b}"
        for k, (x, y) in enumerate(zip(a, b)):
            same(x, y, f"{what}[{k}]")
    assert a == b, f"{what}: {a!r} != {b!r}"


def check(out, *keys):
    for k in keys or [k for k in out if not k.endswith("_sink") and k in ("py",)]:
        pass
    same(out["py"], out["cpp"], "result")
    same(out["py_sink"], out["cpp_sink"], "sink")


# --- the stub scenarios --------------------------------------------------------------------


def test_bookkeeping_merge_finish_and_restart(LN, native, tmp_path, monkeypatch):
    """test_slot_bookkeeping_with_a_stub_receiver: two signals whose
    frequencies move under MERGE_HZ across refreshes keep their tiles, the
    end of the slot is the whole slot, round B; state.json, and a restart
    that keeps the finished tiles until keep_done_s."""
    def scenario(side):
        calls = []

        def detect(prep, spec, live_only=False, **kw):
            calls.append(("detect", live_only, sorted(kw)))
            return [_Det(1500.0), _Det(1610.0)]

        freqs = iter([(1500.3, 1611.0), (1501.9, 1609.5), (1500.0, 1610.0)])

        def receive_slot(prep, spec, live_only=False, estimator="joint", *, erase_s=(),
                         round_b=True, dets=None, max_candidates=None, budget_s=None, hints=()):
            calls.append(("receive", round_b, [tuple(map(float, e)) for e in erase_s],
                          None if dets is None else [d.f_hz for d in dets], max_candidates,
                          budget_s, list(hints), prep.q, int(prep.t0_index), len(prep.fe)))
            return [_Pass(f) for f in next(freqs)]

        side.mp.setattr(RX, "detect", detect)
        side.mp.setattr(RX, "receive_slot", receive_slot)
        cfg = live.LiveConfig(spec=TINY, refresh_s=10.0, first_s=25.0, render=False)
        L = side.listener(cfg=cfg)
        t0 = slot_t0(Q_TEST)
        L.feed(_hiss(76 * FS), t0 + 60.0)
        steps = [L.step(t) for t in (t0 - 20.0, t0 + 30.0, t0 + 35.0, t0 + 41.0,
                                     t0 + live.frame_seconds(TINY) + PB_TAIL_S + 0.5)]
        snap = snapshot(L, side.root / "live")
        L2 = side.listener(cfg=cfg)
        again = [t.to_json() for t in L2.tiles()]
        L2.step(L2.tiles()[0].updated + cfg.keep_done_s + 1)
        return dict(calls=calls, steps=steps, snap=snap, again=again, expired=L2.tiles(),
                    snap2=snapshot(L2, side.root / "live"))

    out = both(native, tmp_path, monkeypatch, scenario)
    check(out)
    assert len(out["py"]["again"]) == 2 and out["py"]["expired"] == []


def test_refused_preamble_and_acquisition_kept_once_final(LN, native, tmp_path, monkeypatch):
    """test_a_refused_preamble_is_checked_again and
    test_acquisition_is_kept_only_once_it_is_final."""
    def scenario(side):
        acq, calls = [], []
        side.mp.setattr(RX, "candidates",
                        lambda *a, **k: acq.append(sorted(k)) or [_Det(1500.0), _Det(1600.0)])
        answers = iter([[_Det(1500.0)], [_Det(1600.0)]])

        def detect(prep, spec, live_only=False, *, erase_s=(), cands=None, skip_hz=(), hints=()):
            calls.append((tuple(c.f_hz for c in cands), tuple(skip_hz)))
            return next(answers)

        side.mp.setattr(RX, "detect", detect)
        side.mp.setattr(RX, "receive_slot",
                        lambda prep, spec, *, dets=None, **k: [_Pass(d.f_hz) for d in dets])
        cfg = live.LiveConfig(spec=TINY, refresh_s=10.0, first_s=25.0, detect_until_s=30.0,
                              recheck_until_s=45.0, render=False)
        L = side.listener(cfg=cfg)
        t0 = slot_t0(Q_TEST)
        L.feed(_hiss(76 * FS), t0 + 60.0)
        for t in (26.0, 36.0, 46.0):
            L.step(t0 + t)
        first = dict(acq=acq, calls=calls, snap=snapshot(L, side.root / "live"))
        # acquisition final only once the audio is past its reach
        acq2 = []
        side.mp.setattr(RX, "candidates", lambda *a, **k: acq2.append(1) or [])
        side.mp.setattr(RX, "detect", lambda *a, **k: [])
        side.mp.setattr(RX, "receive_slot", lambda *a, **k: [])
        from sstvae.qrss import acquire as acquisition

        L = side.listener("live2", cfg=live.LiveConfig(spec=TINY, refresh_s=5.0, first_s=25.0,
                                                       render=False))
        reach = acquisition.live_end_s() + 1.0
        L.feed(_hiss(int((reach - 0.5 + 16) * FS)), t0 + reach - 0.5)
        L.step(t0 + 26.0)
        a = (list(acq2), L.slots[Q_TEST].cands)
        L.feed(_hiss(FS), t0 + reach + 0.5)
        L.step(t0 + 31.0)
        b = (list(acq2), L.slots[Q_TEST].cands)
        L.step(t0 + 36.0)
        return dict(first=first, a=a, b=b, c=list(acq2), snap=snapshot(L, side.root / "live2"))

    out = both(native, tmp_path, monkeypatch, scenario)
    check(out)
    assert out["py"]["first"]["calls"] == [((1500.0, 1600.0), ()), ((1500.0, 1600.0), (1500.0,))]
    assert out["py"]["b"] == ([1, 1], []) and out["py"]["c"] == [1, 1]


def _seed_done_tiles(d, tiles):
    """state.json holding finished tiles (what an earlier run left)."""
    d.mkdir(parents=True, exist_ok=True)
    (d / "state.json").write_text(json.dumps({"tiles": [t.to_json() for t in tiles]}))


def _part1(q, f, mode="B", dial=None, waveform="CE", frame_name="tiny"):
    """A tile of an earlier slot whose header said it was part 1 of `mode`."""
    return live.Tile(id=f"p{q}_{f}", q=q, slot_utc="", frame=frame_name, f_hz=f, dial_hz=dial,
                     status="complete", mode=mode, segment=0, updated=float(slot_t0(q + 2)),
                     callsign="KC2G", waveform=waveform)


def test_late_join_searches_a_hint_only(LN, native, tmp_path, monkeypatch):
    """test_a_slot_mostly_unheard_gets_no_whole_slot_search and
    test_a_late_join_searches_where_a_later_pass_is_due: joined after the
    preamble, the end of the slot looks only at the hint an earlier slot's
    tile gives (read back from state.json)."""
    def scenario(side):
        calls = []

        def detect(*a, hints=(), skip_hz=(), **k):
            calls.append(("detect", list(hints), list(skip_hz), sorted(k)))
            return [_Det(f) for f in hints]

        side.mp.setattr(RX, "detect", detect)
        side.mp.setattr(RX, "receive_slot", lambda *a, dets=None, **k: calls.append(
            ("receive", [d.f_hz for d in dets], sorted(k))) or [])
        q = Q_TEST
        out = {}
        for name, seed in (("nohint", []), ("hint", [_part1(q - 2, 1600.0)])):
            d = side.root / name
            if seed:
                _seed_done_tiles(d, seed)
            L = side.listener(name, cfg=live.LiveConfig(spec=TINY, render=False))
            t0 = slot_t0(q)
            dur = live.frame_seconds(TINY)
            late = t0 + 0.65 * dur
            L.feed(_hiss(int((t0 + dur + 5 - late) * FS)), t0 + dur + 5)
            L.step(late)
            L.step(t0 + dur + PB_TAIL_S + 0.5)
            out[name] = dict(calls=list(calls), snap=snapshot(L, d))
            calls.clear()
        return out

    out = both(native, tmp_path, monkeypatch, scenario)
    check(out)
    assert out["py"]["nohint"]["calls"] == []
    assert [c[:2] for c in out["py"]["hint"]["calls"]] == [("detect", [1600.0]), ("receive", [1600.0])]


def test_a_later_pass_due_through_the_refresh(LN, native, tmp_path, monkeypatch):
    """test_a_later_pass_is_listened_for_where_it_is_due through the live
    refresh: the hint an earlier slot's tile gives reaches acquisition, said
    once ("a later pass is due at"), CE and joint; and a store whose
    accumulator cannot be read says "hints: <str(e)>" and the tiles' hint
    still stands."""
    def scenario(side):
        calls = []

        joint = []

        def candidates(prep, spec, live_only=False, hints=()):
            calls.append(("cand", prep.q, live_only, list(hints)))
            fs = [1500.0] + [h[0] if joint[0] else h for h in hints]
            return [_Det(f, CE, 20.0) if joint[0] else _Det(f) for f in fs]

        def detect(prep, spec, live_only=False, *, cands=None, hints=(), **kw):
            calls.append(("detect", [c.f_hz for c in cands or ()], list(hints), sorted(kw)))
            return list(cands or ())

        side.mp.setattr(RX, "candidates", candidates)
        side.mp.setattr(RX, "detect", detect)
        side.mp.setattr(RX, "receive_slot", lambda *a, dets=None, **k: calls.append(
            ("receive", [d.f_hz for d in dets or ()])) or [])
        q = Q_TEST
        out = {}
        # (an accumulator whose structure is not a dict of lists of dicts
        # raises TypeError out of the reference's store.due, where the C++
        # store skips it: the two part there, in the store, not here)
        bad = ({"mode": 2, "members": [{"segment": None, "uid": "x"}]},)
        for name, ws, acc in (("ce", None, None), ("joint", (CE, L_), None),
                              *((f"bad{k}", None, b) for k, b in enumerate(bad))):
            joint[:] = [ws is not None]
            d = side.root / name
            _seed_done_tiles(d, [_part1(q - 2, 1600.0), _part1(q - 4, 1720.0, "C")])
            store = None
            if acc is not None:
                store = side.store(f"store_{name}")
                (side.root / f"store_{name}" / "acc" / "KC2G_00000011.json").write_text(json.dumps(acc))
            L = side.listener(name, cfg=live.LiveConfig(spec=TINY, waveforms=ws, refresh_s=5.0,
                                                         render=False), store=store)
            t0 = slot_t0(q)
            L.feed(_hiss(66 * FS), t0 + 50.0)
            steps = [L.step(t) for t in (t0 + 26.0, t0 + 31.0, t0 + 50.0)]
            out[name] = dict(calls=list(calls), steps=steps, snap=snapshot(L, d))
            calls.clear()
        return out

    out = both(native, tmp_path, monkeypatch, scenario)
    check(out)
    py = out["py"]
    assert any("a later pass is due at 1600.0, 1720.0 Hz" in ln for ln in py["ce"]["snap"]["lines"])
    assert any("a later pass is due at 1600.0 Hz (CE), 1720.0 Hz (CE)" in ln
               for ln in py["joint"]["snap"]["lines"])
    assert [c[3] for c in py["ce"]["calls"] if c[0] == "cand"][0] == [1600.0, 1720.0]
    lines = [ln.split(" ", 1)[1] for ln in py["bad0"]["snap"]["lines"]]
    assert "hints: int() argument must be a string, a bytes-like object or a real number, not 'NoneType'" in lines
    assert any("a later pass is due at 1600.0, 1720.0 Hz" in ln for ln in lines)


def test_a_slot_heard_on_two_dials(LN, native, tmp_path, monkeypatch):
    """test_a_slot_heard_on_two_dials_is_received_on_each: live on the dial
    the slot started on, the end of the slot once per dial."""
    monkeypatch.setattr(live, "DIAL_GUARD_S", (2.0, 2.0))

    def scenario(side):
        calls = []
        side.mp.setattr(RX, "detect", lambda prep, spec, live_only=False, **kw: [_Det(1500.0)])

        def receive_slot(prep, spec, live_only=False, estimator="joint", *, erase_s=(),
                         round_b=True, dets=None, max_candidates=None, budget_s=None, hints=()):
            calls.append((round_b, bool(dets), [tuple(map(float, e)) for e in erase_s]))
            return [_Pass(1500.0)]

        side.mp.setattr(RX, "receive_slot", receive_slot)
        d = side.root / "live"
        d.mkdir(parents=True)
        q = Q_TEST
        t0 = slot_t0(q)
        (d / live.DIAL_FILE).write_text(
            json.dumps({"t": t0 - 600.0, "dial_hz": 10_138_700, "why": "start"}) + "\n"
            + json.dumps({"t": t0 + 20.0, "dial_hz": 14_095_600, "why": "schedule"}) + "\n")
        L = side.listener(cfg=live.LiveConfig(spec=TINY, render=False))
        end = t0 + live.frame_seconds(TINY) + PB_TAIL_S
        start = t0 - PB_LEAD_S - 1.0
        L.feed(_hiss(int((end + 5 - start) * FS)), end + 5)
        L.step(t0 + 30.0)
        L.step(end + 0.5)
        return dict(calls=calls, snap=snapshot(L, d))

    out = both(native, tmp_path, monkeypatch, scenario)
    check(out)
    assert [c[:2] for c in out["py"]["calls"]] == [(False, True), (True, True), (True, False)]
    assert any("on 2 dials" in s for s in out["py"]["snap"]["lines"])


def test_one_signal_one_tile_whichever_waveform(LN, native, tmp_path, monkeypatch):
    """test_one_signal_is_one_tile_whichever_waveform: the stronger reading
    by Z_ref says the waveform; a weaker one of the other is not applied,
    live or at the end (where the tile goes lost)."""
    def scenario(side):
        answers = iter([[_Pass(1500.0, waveform=CE, z=10.0), _Pass(1560.0, waveform=CE, z=9.0)],
                        [_Pass(1509.5, waveform=L_, z=20.0)],
                        [_Pass(1500.2, waveform=CE, z=15.0)],
                        [_Pass(1500.1, waveform=CE, z=12.0)]])
        side.mp.setattr(RX, "candidates", lambda *a, **k: [])
        side.mp.setattr(RX, "detect", lambda *a, **k: [_Det(1500.0, CE, 10.0)])
        side.mp.setattr(RX, "receive_slot", lambda *a, **k: next(answers))
        L = side.listener(cfg=live.LiveConfig(spec=TINY, waveforms=(CE, L_), refresh_s=10.0,
                                              render=False))
        t0 = slot_t0(Q_TEST)
        end = t0 + L.dur + PB_TAIL_S + 0.5
        L.feed(_hiss(int((end - t0 + 15) * FS)), end)
        snaps = []
        for t in (t0 + 26.0, t0 + 36.0, t0 + 46.0, end):
            L.step(t)
            snaps.append([x.to_json() for x in L.tiles()])
        return dict(snaps=snaps, snap=snapshot(L, side.root / "live"))

    out = both(native, tmp_path, monkeypatch, scenario)
    check(out)
    assert any("is waveform L (Z_ref 20.0 against 10.0 as CE)" in m for m in out["py"]["snap"]["lines"])


def test_ghosts_and_the_headerless_snr_cut(LN, native, tmp_path, monkeypatch):
    """Two overlapping FULL slots (the receivers stubbed by slot): slot q's
    signal has a callsign; the next slot's pass at its frequency is its
    ghost (no tile), and one 60 Hz away is not. Passes too strong for a
    header that did not decode, out of the band or not finite get no tile;
    one with a CW callsign does."""
    def scenario(side):
        q = Q_TEST
        answers = {
            q: [_Pass(1500.0, header=_hdr(), n=10), _Pass(1700.0, snr=3.1, n=10),
                _Pass(float("inf"), n=10), _Pass(5000.0, n=10),
                _Pass(1800.0, snr=2.0, n=10, cw=SimpleNamespace(text="AG7EW")),
                _Pass(1900.0, snr=float("nan"), n=10)],
            q + 1: [_Pass(1512.0, n=10), _Pass(1560.0, n=10)],
        }
        side.mp.setattr(RX, "candidates", lambda *a, **k: [])
        side.mp.setattr(RX, "detect", lambda prep, *a, **k: [_Det(1500.0)])
        side.mp.setattr(RX, "receive_slot", lambda prep, *a, **k: answers.get(prep.q, []))
        cfg = live.LiveConfig(spec=frame.FULL, refresh_s=1000.0, render=False)
        L = side.listener(cfg=cfg)
        t0 = slot_t0(q)
        n = int((900 + 100) * FE_FS)      # (from slot q's lead: the slot before is joined late, never prepared)
        x = np.ones(n, np.complex64)
        L.ring.write(int((t0 - 20) * FE_FS), x)
        L.step(t0 + 30.0)
        a = [x.to_json() for x in L.tiles()]
        L.step(slot_t0(q + 1) + 30.0)
        return dict(a=a, snap=snapshot(L, side.root / "live"))

    out = both(native, tmp_path, monkeypatch, scenario)
    check(out)
    fs = sorted((t["q"] - Q_TEST, round(t["f_hz"])) for t in out["py"]["snap"]["tiles"])
    assert fs == [(0, 1500), (0, 1800), (1, 1560)]


def test_hints_moved_by_the_dial(LN, native, tmp_path, monkeypatch):
    """test_hints_name_where_a_later_pass_is_due and test_hints_carry_their_
    waveform: a picture in the store and earlier slots' tiles (from
    state.json), moved by a USB dial change since, inside the band, one per
    MERGE_HZ; the joint listener's carry their waveform."""
    from qrss_fakes import make_pass_result, synthetic_picture

    def scenario(side):
        q = Q_TEST
        store = side.store(clock=lambda: float(slot_t0(q)))
        sp = synthetic_picture(31, mode=1)
        p = make_pass_result(picture.air_values(sp.segs[0], 0), 1.0, seed=1, q=q, f_hz=1205.47)
        store.add_pass(p)
        store.attach(("KC2G", sp.picture_id), p.uid, 0, "header", mode=1, codec_id=sp.codec_id)
        out = {}
        for ws in (None, (CE, L_), (L_,)):
            d = side.root / f"live{ws}"
            _seed_done_tiles(d, [_part1(q, 1205.6), _part1(q, 1500.0, "C", dial=10_141_000),
                                 _part1(q, 2650.0, "C"), _part1(q, 900.0, "A"),
                                 _part1(q, 2509.5, "B", dial=10_141_000, waveform="L",
                                        frame_name="l-tiny")])
            (d / live.DIAL_FILE).write_text(
                json.dumps({"t": slot_t0(q) - 600.0, "dial_hz": 10_141_000}) + "\n"
                + json.dumps({"t": slot_t0(q + 1), "dial_hz": 10_140_900}) + "\n")
            L = side.listener(f"live{ws}", cfg=live.LiveConfig(spec=TINY, waveforms=ws, render=False),
                              store=store)
            L.dial.refresh()
            out[str(ws)] = [[L._hints(q + k, L.dial.dial_at(slot_t0(q + k))) for k in range(7)],
                            L._hints(q + 2, None)]
        return out

    out = both(native, tmp_path, monkeypatch, scenario)
    check(out)
    assert out["py"]["None"][0][2] == [1305.47, 1600.0]


def test_provisional_picture_redrawn_from_a_header(LN, native, tmp_path, monkeypatch):
    """test_a_headerless_pass_gets_a_provisional_picture through the live
    receive: a headerless pass is drawn as segment 0 of mode A and marked
    GUESS_NOTE; the header a later refresh decodes redraws it unmarked (a
    fake codec, so both draw the same picture); a header decoded once
    stays with its signal when a later receive misses it."""
    from PIL import Image

    class Codec:
        def __init__(self):
            self.decoded = []

        def decode(self, lat, wt):
            k = len(self.decoded)
            self.decoded.append((np.asarray(lat).sum(), np.asarray(wt).sum()))
            v = np.full((48, 64, 3), (37 * k + int(np.asarray(wt).sum()) % 200) % 256, np.uint8)
            v[:, :, 1] = np.arange(64, dtype=np.uint8)[None, :]
            return Image.fromarray(v)

    def scenario(side):
        hdrs = iter([None, _hdr(segment=1), None, None])
        side.mp.setattr(RX, "candidates", lambda *a, **k: [])
        side.mp.setattr(RX, "detect", lambda *a, **k: [_Det(1500.0)])

        def receive_slot(*a, **k):
            p = _Pass(1500.0, header=next(hdrs), n=picture.SENT)
            p.z = np.linspace(-1, 1, picture.SENT).astype(np.float32)
            return [p]

        side.mp.setattr(RX, "receive_slot", receive_slot)
        codec = Codec()
        L = side.listener(cfg=live.LiveConfig(spec=TINY, refresh_s=10.0, render=True), codec=codec)
        t0 = slot_t0(Q_TEST)
        end = t0 + live.frame_seconds(TINY) + PB_TAIL_S + 0.5
        L.feed(_hiss(int((end - t0 + 15) * FS)), end)
        snaps = []
        for t in (t0 + 26.0, t0 + 36.0, t0 + 46.0, end):
            L.step(t)
            snaps.append([x.to_json() for x in L.tiles()])
        return dict(snaps=snaps, decoded=codec.decoded, snap=snapshot(L, side.root / "live"))

    out = both(native, tmp_path, monkeypatch, scenario)
    check(out)
    notes = [s[0]["note"] for s in out["py"]["snaps"]]
    assert live.GUESS_NOTE in notes[0] and live.GUESS_NOTE not in notes[1]
    assert out["py"]["snaps"][-1][0]["image_rev"] >= 2 and out["py"]["snap"]["pictures"]


def test_passes_filed_in_the_store(LN, native, tmp_path, monkeypatch):
    """Genie passes of a picture (qrss_fakes) as the live and end-of-slot
    receive, filed by each listener's own associate in its own store: the
    tiles, their pictures (from the accumulator, a fake codec) and the
    store trees the same."""
    from PIL import Image
    from qrss_fakes import make_header, make_pass_result, synthetic_picture

    class Codec:
        def decode(self, lat, wt):
            v = np.clip(np.asarray(lat)[:3, :48, :64].transpose(1, 2, 0) * 60 + 128, 0, 255)
            return Image.fromarray(v.astype(np.uint8))

    sp = synthetic_picture(5, mode=1)

    def scenario(side):
        q = Q_TEST
        hdr = make_header(sp, segment=0)
        passes = [make_pass_result(picture.air_values(sp.segs[0], 0), 2.0, seed=s, q=q,
                                   f_hz=1500.0 + 0.1 * s, hdr=hdr, decoded=s != 1,
                                   hdr_es_n0_db=3.0) for s in range(3)]
        it = iter(passes)
        side.mp.setattr(RX, "candidates", lambda *a, **k: [])
        side.mp.setattr(RX, "detect", lambda *a, **k: [_Det(1500.0)])
        side.mp.setattr(RX, "receive_slot", lambda *a, **k: [next(it)])
        store = side.store()
        L = side.listener(cfg=live.LiveConfig(spec=TINY, refresh_s=10.0, render=True), store=store,
                          codec=Codec())
        t0 = slot_t0(q)
        end = t0 + live.frame_seconds(TINY) + PB_TAIL_S + 0.5
        L.feed(_hiss(int((end - t0 + 15) * FS)), end)
        for t in (t0 + 26.0, t0 + 36.0, end):
            L.step(t)
        return dict(snap=snapshot(L, side.root / "live"))

    out = both(native, tmp_path, monkeypatch, scenario)
    check(out)
    _store.compare_trees(tmp_path / "py" / "store", tmp_path / "cpp" / "store")
    assert (tmp_path / "py" / "store" / "acc").is_dir()


def test_a_store_that_fails_keeps_the_tile(LN, native, tmp_path, monkeypatch):
    """A pass whose association raises (each listener's own associate, over
    an accumulator of its picture that has lost a field): the tile is kept,
    complete, noted "store failed: <str(e)>" -- a KeyError's text its key's
    repr alone, as str() gives it."""
    from qrss_fakes import make_header, make_pass_result, synthetic_picture

    from sstvae.qrss.store import key_name

    sp = synthetic_picture(6, mode=1)

    def scenario(side):
        q = Q_TEST
        hdr = make_header(sp, segment=0)
        p = make_pass_result(picture.air_values(sp.segs[0], 0), 2.0, seed=4, q=q, f_hz=1500.0, hdr=hdr,
                             decoded=True, hdr_es_n0_db=3.0)
        side.mp.setattr(RX, "candidates", lambda *a, **k: [])
        side.mp.setattr(RX, "detect", lambda *a, **k: [_Det(1500.0)])
        side.mp.setattr(RX, "receive_slot", lambda *a, **k: [p])
        store = side.store()
        acc = side.root / "store" / "acc" / (key_name((hdr.callsign, hdr.picture_id)) + ".json")
        acc.write_text(json.dumps({"mode": 1, "codec_id": sp.codec_id}))
        L = side.listener(cfg=live.LiveConfig(spec=TINY, render=False), store=store)
        t0 = slot_t0(q)
        end = t0 + live.frame_seconds(TINY) + PB_TAIL_S + 0.5
        L.feed(_hiss(int((end - t0 + 15) * FS)), end)
        for t in (t0 + 26.0, end):
            L.step(t)
        return dict(snap=snapshot(L, side.root / "live"))

    out = both(native, tmp_path, monkeypatch, scenario)
    check(out)
    (t,) = out["py"]["snap"]["tiles"]
    assert t["status"] == "complete" and t["note"] == "store failed: 'callsign'", t["note"]


def test_recover_and_backfill_from_the_passband_store(LN, native, tmp_path, monkeypatch):
    """test_a_restarted_listener_finishes_a_slot_that_ended_while_it_was_
    stopped and _reads_the_preamble_back, the receivers stubbed: recover
    lists the slot once, a restart that finished it lists nothing, one
    too long ago nothing; backfill refills the ring from the store; a slot
    the store holds is not received again."""
    def scenario(side):
        q = Q_TEST
        side.mp.setattr(RX, "candidates", lambda *a, **k: [])
        side.mp.setattr(RX, "detect", lambda *a, **k: [_Det(1500.0)])
        side.mp.setattr(RX, "receive_slot", lambda *a, dets=None, **k: [_Pass(1500.0)])
        t0 = slot_t0(q)
        pb = frontend.PassbandStore(side.root / "passband")
        x = (np.random.default_rng(1).standard_normal(int(80 * FE_FS)) * 0.1).astype(np.complex64)
        pb.write(t0 - 15.0, x, expire=False)
        now = t0 + live.frame_seconds(TINY) + PB_TAIL_S + 600.0
        cfg = live.LiveConfig(spec=TINY, render=False)
        L = side.listener(cfg=cfg, passband=pb)
        r1, r2 = L.recover(now), L.recover(now)
        L.step(now)
        snap = snapshot(L, side.root / "live")
        L2 = side.listener(cfg=cfg, passband=pb)
        r3 = L2.recover(now)
        r4 = side.listener("live3", cfg=cfg, passband=pb).recover(now, hours=0.1)
        L5 = side.listener("live5", cfg=cfg, passband=pb)
        got = L5.backfill(t0 + 30.0)
        _, ok = L5.ring.read(int((t0 - 20) * FE_FS), int((t0 + 30) * FE_FS))
        L5.step(t0 + 30.0)
        return dict(r=(r1, r2, r3, r4), snap=snap, backfill=got, ok=ok.tobytes(),
                    snap5=snapshot(L5, side.root / "live5"))

    out = both(native, tmp_path, monkeypatch, scenario)
    check(out)
    assert out["py"]["r"] == ([Q_TEST], [], [], [])


def test_feed_resync_early_audio_and_digital_silence(LN, native, tmp_path, monkeypatch):
    """test_clock_follows_on_and_restarts_after_a_gap, test_digital_silence_
    is_not_heard and the Stamper's bursts: the stream's index, the ring's
    holes, the passband store's files and the log lines the same."""
    def scenario(side):
        out = {}
        L = side.listener("a", cfg=live.LiveConfig(spec=TINY, render=False))
        t = 1_000_000.0
        idx = []
        L.feed(_hiss(FS), t)
        L.feed(_hiss(FS), t + 1.3)
        L.feed(_hiss(FS), t + 10.0)
        for k in range(10):
            L.feed(_hiss(FS), t + 10.5 + 0.01 * k)
        idx.append(L.next_index if side.cpp else L.fe.next_index)
        for k in range(40):
            L.feed(_hiss(FS), t + 19.0 + k)
        idx.append(L.next_index if side.cpp else L.fe.next_index)
        _, ok = L.ring.read(int((t - 1) * FE_FS), int((t + 60) * FE_FS))
        out["a"] = dict(idx=idx, ok=ok.tobytes(), snap=snapshot(L, side.root / "a"))
        pb = frontend.PassbandStore(side.root / "passband")
        L = side.listener("b", cfg=live.LiveConfig(spec=TINY, render=False), passband=pb)
        sound = _hiss(20 * FS)
        sound[5 * FS:5 * FS + FS // 2] = 0.0
        audio = np.concatenate([sound[:10 * FS], np.zeros(60 * FS), sound[10 * FS:]])
        for k in range(0, len(audio), 2000):
            L.feed(audio[k:k + 2000], t + (k + 2000) / FS)
        _, ok = L.ring.read(int(t * FE_FS), int((t + 81) * FE_FS))
        L.close()
        out["b"] = dict(ok=ok.tobytes(), snap=snapshot(L, side.root / "b"),
                        passband=_store.tree(side.root / "passband"))
        st = live.Stamper()
        L = side.listener("c", cfg=live.LiveConfig(spec=TINY, render=False))
        T = 1_000_000.0
        chunks = ([(_hiss(2000), T + (k + 1) * 2000 / FS) for k in range(80)]
                  + [(_hiss(2000), T + 23.0 + 1e-4 * k) for k in range(12)]
                  + [(_hiss(2000), T + 23.0 + (k + 1) * 2000 / FS) for k in range(160)]
                  + [(_hiss(2000), T + 80.0 + (k + 1) * 2000 / FS) for k in range(40)])
        for x, te in chunks:
            for y, e in st.push(x, te):
                L.feed(y, e)
        for y, e in st.flush():
            L.feed(y, e)
        out["c"] = dict(idx=L.next_index if side.cpp else L.fe.next_index,
                        snap=snapshot(L, side.root / "c"))
        return out

    out = both(native, tmp_path, monkeypatch, scenario)
    check(out)
    assert any("silence" in m for m in out["py"]["b"]["snap"]["lines"])
    assert any("ahead of the clock" in m for m in out["py"]["a"]["snap"]["lines"])


class _Replay:
    """A replay: the audio of a TINY slot in half-second chunks, a None (no
    audio) among them."""

    name = "replay"

    def __init__(self, t0, secs):
        self.t0, self.secs = t0, secs

    def chunks(self):
        n = FS // 2
        for k in range(int(self.secs * 2)):
            if k == 7:
                yield None, 0.0
            yield _hiss(n + k), self.t0 + (k + 1) * 0.5


def test_run_replays_to_the_end(LN, native, tmp_path, monkeypatch):
    """run(listener, source, realtime=False): fed chunk by chunk, stepped
    to the newest sample, and at the replay's end stepped on until its slot
    finished; the log, the tiles and state.json the same."""
    def scenario(side):
        calls = []
        side.mp.setattr(RX, "candidates", lambda *a, **k: [])
        side.mp.setattr(RX, "detect", lambda *a, **k: [_Det(1500.0, CE, 10.0)])
        side.mp.setattr(RX, "receive_slot", lambda prep, *a, dets=None, **k: calls.append(
            (prep.q, len(prep.fe), None if dets is None else len(dets))) or [_Pass(1500.0, waveform=CE)])
        cfg = live.LiveConfig(spec=TINY, waveforms=(CE, L_), refresh_s=15.0, render=False)
        L = side.listener(cfg=cfg)
        t0 = slot_t0(Q_TEST)
        src = _Replay(t0 - 20.0, 60.0)
        if side.cpp:
            native.qrss.listener.run(L, src, False)
        else:
            live.run(L, src, realtime=False)
        return dict(calls=calls, snap=snapshot(L, side.root / "live"))

    out = both(native, tmp_path, monkeypatch, scenario)
    check(out)
    assert Q_TEST in out["py"]["snap"]["finished"]
    assert out["py"]["snap"]["lines"][-1].endswith("audio ended")


def test_the_log_file_rotates(LN, native, tmp_path, monkeypatch):
    """listener.log moved to listener.log.1 past LOG_FILE_BYTES, one kept."""
    def scenario(side):
        L = side.listener(cfg=live.LiveConfig(spec=TINY, render=False))
        pad = "x" * 4000
        for k in range(1100):
            L.log(f"line {k} {pad}")
        d = side.root / "live"
        return dict(now=(d / live.LOG_FILE).read_bytes(),
                    old=(d / (live.LOG_FILE + ".1")).read_bytes(), lines=L.lines)

    out = both(native, tmp_path, monkeypatch, scenario)
    assert out["py"]["now"] == out["cpp"]["now"] and out["py"]["old"] == out["cpp"]["old"]
    assert out["py"]["lines"] == out["cpp"]["lines"]


def test_a_bad_state_file_starts_fresh(LN, native, tmp_path, monkeypatch):
    """A state.json that is not the schema (a list where an object is
    wanted, a tile that is not an object): both start fresh, since the
    listener must start. What both read: tiles of every kind, the old
    headerless strong ones dropped, receiving/finishing ones not listed,
    finished slots kept."""
    tiles = [
        live.Tile(id="a", q=Q_TEST, slot_utc="", frame="tiny", f_hz=1500.0, status="complete",
                  callsign="K1ABC", updated=CLOCK),
        live.Tile(id="b", q=Q_TEST, slot_utc="", frame="tiny", f_hz=1600.0, status="lost",
                  snr_db=-3.0, updated=CLOCK),
        live.Tile(id="c", q=Q_TEST + 1, slot_utc="", frame="tiny", f_hz=1700.0, status="finishing"),
        live.Tile(id="d", q=Q_TEST + 2, slot_utc="", frame="tiny", f_hz=1800.0, status="complete",
                  cw_text="AG7EW", snr_db=2.0, updated=CLOCK),
    ]

    def scenario(side):
        d = side.root / "live"
        d.mkdir(parents=True)
        st = {"tiles": [t.to_json() for t in tiles], "finished": [Q_TEST - 4, "7"]}
        st["tiles"][0]["unknown"] = 1
        (d / "state.json").write_text(json.dumps(st))
        L = side.listener(cfg=live.LiveConfig(spec=TINY, render=False))
        return dict(tiles=[t.to_json() for t in L.tiles()], finished=sorted(L.finished))

    out = both(native, tmp_path, monkeypatch, scenario)
    same(out["py"], out["cpp"])
    for bad in ("[1, 2]", '{"tiles": [1]}', '{"finished": 5}'):
        d = tmp_path / f"bad{len(bad)}"
        d.mkdir()
        (d / "state.json").write_text(bad)
        P = live.LiveListener(d, live.LiveConfig(spec=TINY, render=False), log=lambda m: None)
        assert P.tiles() == [] and P.finished == set()
        P.close()
        L = _listener.listener(native, d, live.LiveConfig(spec=TINY, render=False), log=lambda m: None)
        assert L.tiles() == [] and L.finished == set()
        L.close()


def test_a_tile_of_another_shape_is_skipped_alone(LN, native, tmp_path, monkeypatch):
    """A state.json tile whose value is not of its field's JSON type (a null
    where the field cannot be None, a string for a number) is skipped alone
    and the rest read, a number taken as its field's type (an int as a
    float, a bool as 0 or 1): the same tiles and finished slots from both.
    live.py kept such a tile, Tile(**...), where the C++ skipped it."""
    good = dict(id="good", q=Q_TEST, slot_utc="", frame="tiny", f_hz=1500.0, status="complete",
                callsign="K1ABC", updated=CLOCK)
    nums = dict(good, id="nums", f_hz=1700, passes=True, heard=True, dial_hz=False, snr_db=3,
                image_rev=False, confidence=[1, None, True, 2.5])
    bad = [("f_hz", None), ("heard", None), ("progress", None), ("received", None), ("updated", None),
           ("q", None), ("passes", None), ("image_rev", None), ("q", 3.5), ("dial_hz", 1.5),
           ("segment", "1"), ("id", 5), ("status", None), ("note", None), ("waveform", None),
           ("callsign", 7), ("snr_db", "x"), ("confidence", None), ("confidence", [1.0, "x"]),
           ("f_hz", "1500"), ("slot_utc", None), ("uid", 12), ("mode", 1), ("frame", None)]
    tiles = [good, nums]
    for i, (field, value) in enumerate(bad):
        t = dict(good, id=f"bad{i}", f_hz=1600.0 + i, status="shown" if field == "q" else "lost")
        t[field] = value
        tiles.append(t)

    def scenario(side):
        d = side.root / "live"
        d.mkdir(parents=True)
        (d / "state.json").write_text(json.dumps({"tiles": tiles, "finished": [Q_TEST - 4]}))
        L = side.listener(cfg=live.LiveConfig(spec=TINY, render=False))
        return dict(tiles=[t.to_json() for t in L.tiles()], finished=sorted(L.finished))

    out = both(native, tmp_path, monkeypatch, scenario)
    same(out["py"], out["cpp"])
    assert [t["id"] for t in out["py"]["tiles"]] == ["good", "nums"]
    n = out["py"]["tiles"][1]
    assert (n["f_hz"], n["passes"], n["heard"], n["dial_hz"], n["confidence"]) == (1700.0, 1, 1.0, 0,
                                                                                  [1.0, None, 1.0, 2.5])


# --- where the two differ by design ---------------------------------------------------------


def test_an_exception_in_a_refresh_is_logged_once_and_retried(LN, native, tmp_path, monkeypatch):
    """The C++ listener: a live refresh that raises is logged once per slot
    and message, the slot stays live and is received again at its next
    refresh, as the reference's worker does; the reference's synchronous
    listener raises it out of step()."""
    _listener.frozen_time(monkeypatch, CLOCK)
    n = {"k": 0}

    def receive_slot(*a, **k):
        n["k"] += 1
        if n["k"] <= 2:
            raise RuntimeError("the receiver fell over")
        return [_Pass(1500.0)]

    monkeypatch.setattr(RX, "candidates", lambda *a, **k: [])
    monkeypatch.setattr(RX, "detect", lambda *a, **k: [_Det(1500.0)])
    monkeypatch.setattr(RX, "receive_slot", receive_slot)
    lines = []
    cfg = live.LiveConfig(spec=TINY, refresh_s=5.0, render=False)
    L = _listener.listener(native, tmp_path / "c", cfg, log=lines.append)
    t0 = slot_t0(Q_TEST)
    L.feed(_hiss(76 * FS), t0 + 60.0)
    L.step(t0 + 26.0)
    L.step(t0 + 31.0)
    failed = [m for m in lines if "RuntimeError" in m]
    assert len(failed) == 1 and failed[0].endswith(
        "slot 2025-10-01T00:00:00Z: RuntimeError: the receiver fell over"), failed
    assert not L.slots[Q_TEST].done and L.tiles() == []
    L.step(t0 + 36.0)
    assert [round(t.f_hz) for t in L.tiles()] == [1500] and n["k"] == 3
    L.step(t0 + live.frame_seconds(TINY) + PB_TAIL_S + 0.5)
    assert Q_TEST in L.finished
    L.close()
    # the reference, synchronous: out of step()
    n["k"] = 0
    P = live.LiveListener(tmp_path / "p", cfg, log=lambda m: None)
    P.feed(_hiss(76 * FS), t0 + 60.0)
    with pytest.raises(RuntimeError):
        P.step(t0 + 26.0)


def test_background_receives_with_close_in_flight(LN, native, tmp_path, monkeypatch):
    """background: a slow end-of-slot receive does not hold up feed() or
    write_state (busy names it), the slot finishes once it returns, and
    close() with a live receive in flight returns."""
    import threading

    _listener.frozen_time(monkeypatch, CLOCK)
    release, started = threading.Event(), threading.Event()

    def slow(*a, **k):
        started.set()
        release.wait(20)
        return []

    monkeypatch.setattr(RX, "receive_slot", slow)
    monkeypatch.setattr(RX, "detect", lambda *a, **k: [])
    cfg = live.LiveConfig(spec=TINY, render=False, background=True)
    L = _listener.listener(native, tmp_path, cfg, log=lambda m: None)
    t0 = slot_t0(Q_TEST)
    end = t0 + live.frame_seconds(TINY) + PB_TAIL_S + 0.5
    L.feed(_hiss(int((end - t0 + 15) * FS)), end)
    L.step(t0 + 30.0)
    assert L.wait_idle(10.0)
    L.step(end)
    assert started.wait(10)
    L.feed(_hiss(FS), end + 1.0)
    L.write_state()
    st = json.loads((tmp_path / "state.json").read_text())
    assert st["busy"] and "finish" in st["busy"] and not L.idle()
    release.set()
    assert L.wait_idle(10.0)
    L.step(end + 2.0)
    assert Q_TEST in L.finished and L.idle()
    # close() while a receive is in flight
    release.clear()
    started.clear()
    monkeypatch.setattr(RX, "detect", lambda *a, **k: [_Det(1500.0)])
    L2 = _listener.listener(native, tmp_path / "two", cfg, log=lambda m: None)
    L2.feed(_hiss(76 * FS), t0 + 60.0)
    L2.step(t0 + 30.0)
    assert started.wait(10)
    threading.Timer(0.5, release.set).start()
    L2.close()
    L2.close()                    # idempotent
    L.close()


# --- the real receiver ----------------------------------------------------------------------


def _tiny_sim(waveform, snr_db=-6.0):
    q = Q_TEST
    if waveform == L_:
        a = unit_rms_latents(L_TINY.n_data, 0)
        sym = lframe.assemble(L_TINY, None, precoder.precode(a, q))
        return chm.simulate(sym, L_TINY, q, ChannelConfig(snr_db=snr_db, seed=2), carrier_hz=1509.5)
    a = unit_rms_latents(TINY.n_data, 0)
    sym = frame.assemble(TINY, None, precoder.precode(a, q))
    return chm.simulate(sym, TINY, q, ChannelConfig(snr_db=snr_db, seed=2), carrier_hz=1500.0)


@pytest.mark.parametrize("waveform", [CE, L_], ids=["ce", "l"])
def test_the_real_receiver_half_heard_then_whole(LN, native, tmp_path, monkeypatch, waveform):
    """test_tiny_slot_half_heard_then_whole on each listener's own receiver:
    one tile, its frequency within 0.1 Hz and SNR within 0.3 dB of the
    reference's, the rest of what it shows the same."""
    _listener.frozen_time(monkeypatch, CLOCK)
    sim = _tiny_sim(waveform)
    spec = L_TINY if waveform == L_ else TINY
    ws = (L_,) if waveform == L_ else None
    q = Q_TEST
    t0 = slot_t0(q)
    n0 = t0 * FE_FS - int(round(sim.t0_index))
    half = int(round(sim.t0_index + 0.5 * live.frame_seconds(spec) * FE_FS))
    cut = (half - sim.t0_index) / FE_FS
    got = {}
    for kind in ("py", "cpp"):
        cfg = live.LiveConfig(spec=TINY, waveforms=ws, render=False)
        d = tmp_path / kind
        if kind == "py":
            L = live.LiveListener(d, cfg, log=lambda m: None)
        else:
            L = _listener.listener(native, d, cfg, log=lambda m: None, stubbed=False, clock=lambda: CLOCK)
        L.ring.write(n0, sim.fe[:half])
        L.step(t0 + cut)
        first = [t.to_json() for t in L.tiles()]
        L.ring.write(n0 + half, sim.fe[half:])
        L.step(t0 + L.dur + PB_TAIL_S + 1.0)
        got[kind] = (first, [t.to_json() for t in L.tiles()])
        L.close()
    for a, b in zip(got["py"], got["cpp"]):
        assert len(a) == len(b) == 1, (a, b)
        (x,), (y,) = a, b
        assert abs(x["f_hz"] - y["f_hz"]) < 0.1 and abs(x["snr_db"] - y["snr_db"]) < 0.3
        for k in ("id", "q", "status", "callsign", "picture_id", "segment", "mode", "waveform",
                  "frame", "progress", "heard", "dial_hz", "image", "passes"):
            assert x[k] == y[k], (k, x[k], y[k])
        assert abs(x["received"] - y["received"]) < 0.02
    assert got["py"][1][0]["status"] == "complete" and got["py"][0][0]["status"] == "receiving"


# --- the codec -----------------------------------------------------------------------------


def test_the_codec_renderer_draws_the_references_pictures(LN, native, tmp_path, monkeypatch):
    """With SSTVAE's codec: the C++ ListenerRenderer (lazy load, the codec
    ID checked) draws each tile's picture pixel for pixel as the reference's
    render.load / render.render; a picture of another codec is refused
    with the reference's note."""
    if not hasattr(LN, "ListenerRenderer"):
        pytest.skip("sstvae_native built without SSTVAE's codec (native/build-qrss-codec)")
    from qrss_fakes import make_header, make_pass_result, synthetic_picture

    from sstvae.qrss import render

    from sstvae import checkpoint

    try:
        ref_codec = render.load("fp32")
        have = render.decoder_codec_id(ref_codec)
        # the C++ is handed the file the reference loaded: its own cache
        # (SSTVAE_MODEL_CACHE) is not huggingface_hub's, and the artifact
        # compared must be one
        decoder = checkpoint.resolve_onnx("decoder", render.model_dir(), "fp32")
    except BaseException as e:      # noqa: BLE001 - no model to compare with
        if os.environ.get("SSTVAE_REQUIRE_CODEC"):
            raise
        pytest.skip(f"no codec: {e}")
    _listener.frozen_time(monkeypatch, CLOCK)
    sp = synthetic_picture(3, mode=0, codec_id=have if have is not None else 0xD1D8)

    def passes():
        hdr = make_header(sp, segment=0)
        p = make_pass_result(picture.air_values(sp.segs[0], 0), 4.0, seed=0, q=Q_TEST, f_hz=1500.0,
                             hdr=hdr, decoded=True)
        bad = make_pass_result(picture.air_values(sp.segs[0], 0), 4.0, seed=1, q=Q_TEST, f_hz=1700.0,
                               hdr=make_header(synthetic_picture(4, mode=0, codec_id=0x1234)),
                               decoded=True)
        return [p, bad]

    monkeypatch.setattr(RX, "candidates", lambda *a, **k: [])
    monkeypatch.setattr(RX, "detect", lambda *a, **k: [_Det(1500.0), _Det(1700.0)])
    monkeypatch.setattr(RX, "receive_slot", lambda *a, **k: passes())
    t0 = slot_t0(Q_TEST)
    cfg = live.LiveConfig(spec=TINY, refresh_s=10.0, render=True)
    snaps = {}
    for kind in ("py", "cpp"):
        d = tmp_path / kind
        if kind == "py":
            L = live.LiveListener(d, cfg, log=lambda m: None)
        else:
            L = _listener.listener(native, d, cfg, log=lambda m: None, clock=lambda: CLOCK,
                                   renderer=LN.ListenerRenderer(cfg.precision, decoder))
        L.feed(_hiss(76 * FS), t0 + 60.0)
        L.step(t0 + 26.0)
        snaps[kind] = snapshot(L, d)
        L.close()
    same(snaps["py"], snaps["cpp"])
    assert len(snaps["py"]["pictures"]) == 1
    notes = {round(t["f_hz"]): t["note"] for t in snaps["py"]["tiles"]}
    assert "1234" in notes[1700]

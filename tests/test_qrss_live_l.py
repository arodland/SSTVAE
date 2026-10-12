"""The live listener with both waveforms (`LiveConfig.waveforms`,
`qrss_listen.py --just-ce / --just-l`).

Fast: what each command line listens for (parsed in process: no
listener, no subprocess), the frames every receiver call is handed
(stub receivers that record them, as in test_qrss_live.py), the slot's
length with each selection, one tile per signal whichever waveform it is
read as, tiles and state.json carrying the waveform, hints carrying
theirs, a tile's picture counting a stored member read as the other
waveform once, and L's band edge. CE alone must make exactly the calls
it made before L, with the CE frame itself. Slow: real slots holding a
CE and an L signal, through the real receiver: one written to the ring,
and one heard as audio across the slot's end with each selection
holding L.
"""

import importlib.util
import json
import os
import pickle

import numpy as np
import pytest

from qrss_helpers import Q_TEST, REPO_ROOT, latent_snr_db, unit_rms_latents
from sstvae.qrss import channel as chm
from sstvae.qrss import frame, frontend, lframe, live, precoder
from sstvae.qrss import receiver as RX
from sstvae.qrss.channel import ChannelConfig, Neighbour
from sstvae.qrss.constants import FE_FS, FS, L_T_SYM, WAVEFORM_CE, WAVEFORM_L
from sstvae.qrss.frontend import PB_LEAD_S, PB_TAIL_S, slot_t0

CE, L = WAVEFORM_CE, WAVEFORM_L
TINY, L_TINY = frame.TINY, lframe.L_TINY
SELECTIONS = (None, (CE,), (L,), (CE, L))


def _hiss(n):
    """n samples of a quiet band: faint, but heard (exact zeros are not)."""
    return 1e-3 * np.random.default_rng(n).standard_normal(n)


def _listener(path, waveforms=(CE, L), **kw):
    cfg = live.LiveConfig(spec=TINY, waveforms=waveforms, render=False, **kw)
    return live.LiveListener(path, cfg, log=lambda m: None)


class _Det:
    def __init__(self, f, waveform=CE, z=10.0, method="preamble"):
        self.f_hz, self.waveform, self.z_ref, self.method = f, waveform, z, method


class _Report:
    def __init__(self, z, snr=-12.0):
        self.snr2500_db, self.z_ref, self.suspect = snr, z, False


class _Pass:
    def __init__(self, f, waveform=CE, z=20.0, header=None, snr=-12.0):
        self.f_hz, self.waveform, self.report = f, waveform, _Report(z, snr)
        self.frame = "l-tiny" if waveform == L else "tiny"
        self.header, self.cw, self.uid = header, None, f"u{f}"
        self.w = np.ones(TINY.n_data, dtype=np.float32)
        self.z = np.zeros(TINY.n_data, dtype=np.float32)


@pytest.fixture
def qrss_listen():
    """qrss_listen.py as a module (its main not run), with the thread
    settings its import puts in the environment taken out again."""
    names = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
    before = {v: os.environ.get(v) for v in names}
    spec = importlib.util.spec_from_file_location("qrss_listen", REPO_ROOT / "qrss_listen.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for v, x in before.items():
        if x is None:
            os.environ.pop(v, None)
        else:
            os.environ[v] = x
    return mod


def test_the_default_config_is_ce_alone():
    """LiveConfig() is the listener as it was before L: CE alone, the receiver
    handed the frame itself; any selection holding L is handed the tuple."""
    cfg = live.LiveConfig()
    assert cfg.waveforms is None and cfg.specs == (frame.FULL,) and cfg.selection is frame.FULL
    want = {None: (TINY,), (CE,): (TINY,), (L,): (L_TINY,), (CE, L): (TINY, L_TINY),
            (L, CE): (TINY, L_TINY)}
    for ws, specs in want.items():
        c = live.LiveConfig(spec=TINY, waveforms=ws)
        assert c.specs == specs
        assert c.selection is TINY if L not in (ws or ()) else c.selection == specs
        # the receiver's own rule for what a selection is
        assert RX.frames(c.selection) == specs
        assert RX.is_joint(c.selection) == (L in (ws or ()))
    with pytest.raises(ValueError):
        live.LiveConfig(spec=frame.FULL_V1, waveforms=(CE, L)).specs   # no L twin
    with pytest.raises(ValueError):
        live.LiveConfig(waveforms=()).specs
    assert live.LiveConfig(spec=frame.FULL_V1, waveforms=(CE,)).selection is frame.FULL_V1


def test_a_slot_lasts_as_long_as_the_longer_frame(tmp_path):
    """frame_seconds(L_FULL) in L's symbols; the listener's slot (ring,
    active slots, progress, the capture read) is the longest enabled frame's,
    and with CE alone exactly today's."""
    assert live.frame_seconds(lframe.L_FULL) == lframe.L_FULL.keyed_end_pos * L_T_SYM
    assert live.frame_seconds(lframe.L_FULL) == pytest.approx(1782.65, abs=0.01)
    assert live.frame_seconds((TINY, L_TINY)) == live.frame_seconds(L_TINY) > \
        live.frame_seconds(TINY)
    t0 = slot_t0(Q_TEST)
    for k, ws in enumerate(SELECTIONS):
        lst = _listener(tmp_path / str(k), ws)
        dur = max(live.frame_seconds(s) for s in lst.specs)
        assert lst.dur == dur
        assert lst.ring.n == int((dur + PB_LEAD_S + PB_TAIL_S + 300.0) * FE_FS)
        assert Q_TEST in lst.active_slots(t0 + dur + PB_TAIL_S - 0.01)
        assert Q_TEST not in lst.active_slots(t0 + dur + PB_TAIL_S)
        lst.now = t0 + dur / 2
        assert lst._progress(Q_TEST) == pytest.approx(0.5)
        x, _, origin = live.capture_read(lst.ring, Q_TEST, lst.selection)
        assert len(x) - origin == int(round((dur + PB_TAIL_S) * FE_FS))
        if L not in (ws or ()):
            assert lst.dur == live.frame_seconds(TINY) and lst.selection is TINY
    # a restarted listener finishes a slot only once its longer frame has ended
    pb = frontend.PassbandStore(tmp_path / "passband")
    pb.write(t0 - 15.0, np.full(int(80 * FE_FS), 0.1, np.complex64), expire=False)
    end = t0 + live.frame_seconds(L_TINY) + PB_TAIL_S
    for k, now in enumerate((end - 0.01, end + 0.01)):
        lst = live.LiveListener(tmp_path / f"r{k}", live.LiveConfig(spec=TINY, waveforms=(CE, L),
                                                                    render=False),
                                passband=pb, log=lambda m: None)
        assert lst.recover(now) == ([Q_TEST] if k else [])


def test_the_command_line_picks_the_waveforms(qrss_listen, tmp_path):
    """Neither flag: both waveforms (the app passes only --state); --just-ce:
    CE alone, today's listener; --just-l: L alone. --frame names the CE
    frame, and a shape L has no frame of needs --just-ce."""
    build = qrss_listen.build_config
    args, cfg = build(["--state", str(tmp_path)])
    assert args.state == tmp_path and cfg.background and cfg.finish_process
    assert cfg.waveforms == (CE, L) and cfg.selection == (frame.FULL, lframe.L_FULL)
    _, cfg = build(["--just-ce"])
    assert cfg.specs == (frame.FULL,) and cfg.selection is frame.FULL
    _, cfg = build(["--just-l", "--frame", "short"])
    assert cfg.spec is frame.SHORT and cfg.selection == (lframe.L_SHORT,)
    _, cfg = build(["--frame", "tiny", "--refresh", "30"])
    assert cfg.selection == (TINY, L_TINY) and cfg.refresh_s == 30.0
    # L's name for a shape names it as well (qrss_receive.py takes either)
    _, cfg = build(["--just-l", "--frame", "l-short"])
    assert cfg.spec is frame.SHORT and cfg.selection == (lframe.L_SHORT,)
    _, cfg = build(["--frame", "l-tiny"])
    assert cfg.spec is TINY and cfg.selection == (TINY, L_TINY)
    with pytest.raises(SystemExit, match="l-full"):
        build(["--frame", "l-huge"])
    with pytest.raises(SystemExit, match="CE only.*--just-ce"):
        build(["--frame", "full-v1"])
    with pytest.raises(SystemExit, match="CE only"):
        build(["--frame", "full-spreadonly", "--just-l"])
    _, cfg = build(["--frame", "full-v1", "--just-ce"])
    assert cfg.selection is frame.FULL_V1
    with pytest.raises(SystemExit):
        build(["--just-ce", "--just-l"])          # (argparse: one or the other)


class _InProcess:
    """`_FinishProcess` without the process: the arguments go through pickle."""

    def call(self, args):
        return live.finish_receive(*pickle.loads(pickle.dumps(args)))

    def close(self):
        pass


@pytest.mark.parametrize("ws", SELECTIONS, ids=["none", "ce", "l", "both"])
def test_every_receiver_call_gets_the_selection(tmp_path, monkeypatch, ws):
    """Live acquisition, detection, the live receive and the end-of-slot
    receive (its arguments pickled, as for the child process) are each
    handed the selection: the CE frame itself for CE alone, else the
    tuple; a tile takes the pass's waveform and frame, and its confidence
    the pass's own frame."""
    wf = L if ws == (L,) else CE
    calls, conf = [], []

    def candidates(prep, spec, live_only=False, hints=()):
        calls.append(("candidates", spec))
        return [_Det(1500.0, wf)]

    def detect(prep, spec, live_only=False, **kw):
        calls.append(("detect", spec))
        return [_Det(1500.0, wf)]

    def receive_slot(prep, spec, live_only=False, estimator="joint", *, dets=None, **kw):
        calls.append(("receive", spec, dets is None, kw.get("max_candidates"),
                      kw.get("budget_s")))
        return [_Pass(1500.0, wf, z=20.0 + len(calls))]

    def confidence_db(p, spec, elapsed_s=None, n=RX.CONF_BINS):
        conf.append((spec, elapsed_s))
        return [None] * n

    monkeypatch.setattr(RX, "candidates", candidates)
    monkeypatch.setattr(RX, "detect", detect)
    monkeypatch.setattr(RX, "receive_slot", receive_slot)
    monkeypatch.setattr(RX, "confidence_db", confidence_db)
    lst = _listener(tmp_path, ws)
    lst._proc = _InProcess()
    t0 = slot_t0(Q_TEST)
    end = t0 + lst.dur + PB_TAIL_S + 0.5
    lst.feed(_hiss(int((end - t0 + 15) * FS)), end)
    lst.step(t0 + 30.0)
    lst.step(end)
    assert [c[0] for c in calls] == ["candidates", "detect", "receive", "receive"]
    for c in calls:
        if L in (ws or ()):
            assert isinstance(c[1], tuple) and c[1] == lst.specs
        elif c is calls[-1]:
            assert c[1] == TINY and not isinstance(c[1], tuple)   # (a pickled copy, as today)
        else:
            assert c[1] is TINY                   # today's call, with the frame itself
    assert calls[-1][2:] == (True, live.FINISH_MAX_CANDIDATES, live.FINISH_BUDGET_S)
    (tile,) = lst.tiles()
    assert tile.status == "complete"
    assert (tile.waveform, tile.frame) == (("L", "l-tiny") if wf == L else ("CE", "tiny"))
    assert [c[0] for c in conf] == [L_TINY if wf == L else TINY] * 2
    # seconds after t0 when the audio was read, whichever frame is the longer
    assert [c[1] for c in conf] == [pytest.approx(30.0, abs=1e-9), pytest.approx(lst.dur)]
    lst.write_state()
    st = json.loads((tmp_path / "state.json").read_text())
    assert st["waveforms"] == [{CE: "CE", L: "L"}[s.waveform] for s in lst.specs]
    assert st["frame"] == "tiny" and st["tiles"][0]["waveform"] == tile.waveform


def test_tiles_carry_their_waveform_and_frame(tmp_path):
    """A tile's waveform and frame go through state.json and back; a tile
    written before L had one loads as CE."""
    lst = _listener(tmp_path)
    t = live.Tile(id="a", q=Q_TEST, slot_utc="2025-10-01T00:00:00Z", frame="l-tiny",
                  f_hz=1509.5, status="complete", callsign="K1ABC", waveform="L",
                  updated=slot_t0(Q_TEST) + 100.0)
    assert live.Tile(**json.loads(json.dumps(t.to_json()))) == t
    lst.done_tiles.append(t)
    lst.write_state()
    st = json.loads((tmp_path / "state.json").read_text())
    assert st["waveforms"] == ["CE", "L"] and st["frame"] == "tiny"
    assert (st["tiles"][0]["waveform"], st["tiles"][0]["frame"]) == ("L", "l-tiny")
    old = {k: v for k, v in st["tiles"][0].items() if k != "waveform"}
    old.update(id="b", frame="tiny", f_hz=1700.0)
    st["tiles"].append(old)
    (tmp_path / "state.json").write_text(json.dumps(st))
    again = {t.id: t for t in _listener(tmp_path).done_tiles}
    assert (again["a"].waveform, again["a"].frame) == ("L", "l-tiny")
    assert (again["b"].waveform, again["b"].frame) == ("CE", "tiny")
    legacy = live.LiveListener(tmp_path / "ce", live.LiveConfig(spec=TINY, render=False),
                               log=lambda m: None)
    legacy.write_state()
    assert json.loads((tmp_path / "ce" / "state.json").read_text())["waveforms"] == ["CE"]


def test_one_signal_is_one_tile_whichever_waveform(tmp_path, monkeypatch):
    """A CE and an L pass at one band centre (an L pilot 9.5 Hz above it)
    are one tile: the stronger by Z_ref says which waveform it is, and a
    weaker reading as the other waveform is not applied to it, live or at
    the end of the slot (where the tile then goes lost, nothing stored)."""
    answers = iter([[_Pass(1500.0, CE, z=10.0), _Pass(1560.0, CE, z=9.0)],
                    [_Pass(1509.5, L, z=20.0)],
                    [_Pass(1500.2, CE, z=15.0)],
                    [_Pass(1500.1, CE, z=12.0)]])          # the end of the slot
    monkeypatch.setattr(RX, "candidates", lambda *a, **k: [])
    monkeypatch.setattr(RX, "detect", lambda *a, **k: [_Det(1500.0)])
    monkeypatch.setattr(RX, "receive_slot", lambda *a, **k: next(answers))
    monkeypatch.setattr(RX, "confidence_db", lambda p, spec, *a, **k: [])
    lines = []
    lst = live.LiveListener(tmp_path, live.LiveConfig(spec=TINY, waveforms=(CE, L),
                                                      refresh_s=10.0, render=False),
                            log=lines.append)
    t0 = slot_t0(Q_TEST)
    end = t0 + lst.dur + PB_TAIL_S + 0.5
    lst.feed(_hiss(int((end - t0 + 15) * FS)), end)
    lst.step(t0 + 26.0)
    tiles = {round(t.f_hz): t for t in lst.tiles()}
    assert sorted(tiles) == [1500, 1560] and tiles[1500].waveform == "CE"
    tid = tiles[1500].id
    lst.step(t0 + 36.0)                       # L, stronger: the tile becomes L
    t = lst.slots[Q_TEST].tiles[tid]
    assert (t.waveform, t.frame, t.f_hz, t.z_ref) == ("L", "l-tiny", 1509.5, 20.0)
    assert any("is waveform L (Z_ref 20.0 against 10.0 as CE)" in m for m in lines)
    lst.step(t0 + 46.0)                       # CE again, weaker: not applied
    assert (t.waveform, t.f_hz, t.z_ref) == ("L", 1509.5, 20.0)
    assert len(lst.slots[Q_TEST].tiles) == 2
    lst.step(end)
    by = {x.id: x for x in lst.tiles()}
    assert by[tid].status == "lost" and by[tid].waveform == "L"
    assert any("as waveform CE is not kept" in m for m in lines)


def test_a_signal_keeps_its_header_whichever_waveform_reads_it(tmp_path):
    """A header one receive decoded stays with its signal (its band centre),
    whichever waveform a later receive reads it as; a pass of the other
    waveform only as strong as the tile does not take the tile over."""
    lst = _listener(tmp_path)
    s = live.SlotState(Q_TEST)
    lst.slots = {Q_TEST: s}
    t = lst._tile_for(s, 1509.5, waveform="L", frame="l-tiny")
    t.z_ref, hdr = 20.0, object()
    lst._headers[t.id] = hdr
    assert lst._known_header(s, _Pass(1509.6, L)).header is hdr
    assert lst._known_header(s, _Pass(1500.0, CE)).header is hdr     # its centre, read as CE
    assert lst._known_header(s, _Pass(1509.5, CE)).header is None    # 9.5 Hz from it
    assert not lst._takes(t, _Pass(1500.0, CE, z=20.0)) and (t.waveform, t.frame) == \
        ("L", "l-tiny")
    assert lst._takes(t, _Pass(1500.0, CE, z=20.5)) and (t.waveform, t.frame) == ("CE", "tiny")


def test_the_tile_picture_knows_a_member_by_its_centre(tmp_path):
    """The picture of a tile whose transmission the store holds already, read
    as the other waveform, counts that member once: the joint listener asks
    `Store.twin` by band centre (its waveform), CE alone as before L."""
    from types import SimpleNamespace

    from PIL import Image

    from sstvae.qrss import picture

    class Codec:
        def decode(self, lat, wt):
            return Image.new("RGB", (64, 48))

    hdr = SimpleNamespace(segment=0, mode=0, codec_id=0xD1D8, callsign="K1ABC", grid="FN42",
                          picture_id=7)
    S = np.zeros((picture.N_GROUPS, picture.GROUP_LATENTS), dtype=np.float32)
    acc = SimpleNamespace(mode=0, codec_id=0xD1D8, S=S, W=np.ones_like(S), members=[{}],
                          uids=["the CE reading"])
    for k, (ws, wf, want) in enumerate([((CE, L), L, {"waveform": L}), (None, CE, {})]):
        asked = []
        store = SimpleNamespace(has_accumulator=lambda key: key == ("K1ABC", 7),
                                accumulator=lambda key: acc,
                                twin=lambda *a, asked=asked, **kw: asked.append((a[1:], kw))
                                or {"uid": "x"})
        lst = live.LiveListener(tmp_path / str(k), live.LiveConfig(spec=TINY, waveforms=ws,
                                                                   render=True),
                                store=store, log=lambda m: None)
        lst.codec = Codec()
        p = _Pass(1509.5 if wf == L else 1500.0, wf, header=hdr)
        p.q, p.z, p.w = Q_TEST, np.zeros(picture.SENT, np.float32), np.ones(picture.SENT,
                                                                          np.float32)
        t = live.Tile(id="t", q=Q_TEST, slot_utc="x", frame=p.frame, f_hz=p.f_hz)
        lst._render(t, p)
        assert asked == [((Q_TEST, p.f_hz, 0), want)] and t.passes == 1


class _Quiet:
    name = "nothing"

    def chunks(self):
        return iter(())


def test_run_says_what_it_listens_for(tmp_path):
    want = {None: "waveform CE (tiny)", (L,): "waveform L (l-tiny)",
            (CE, L): "waveforms CE (tiny) and L (l-tiny)"}
    for k, (ws, text) in enumerate(want.items()):
        lines = []
        lst = live.LiveListener(tmp_path / str(k), live.LiveConfig(spec=TINY, waveforms=ws,
                                                                   render=False),
                                log=lines.append)
        live.run(lst, _Quiet(), realtime=False)
        assert lines[0].endswith(f"listening: nothing, frame tiny, {text}")


def test_detections_are_kept_one_per_signal_by_centre(tmp_path):
    """s.dets holds one detection per band centre: the newest of a waveform,
    and one of the other waveform only when it is stronger."""
    lst = _listener(tmp_path)
    s = live.SlotState(Q_TEST)
    ce, l_weak, l_strong = _Det(1500.0, CE, 8.0), _Det(1509.5, L, 6.0), _Det(1509.5, L, 9.0)
    for d in (ce, l_weak, _Det(1509.5, L, 8.0)):     # weaker, then as strong: held
        lst._keep(s, d)
    assert s.dets == [ce]
    lst._keep(s, l_strong)
    assert s.dets == [l_strong]
    newer = _Det(1509.4, L, 5.0)                  # its own waveform: the newest stands
    lst._keep(s, newer)
    other = _Det(1530.0, CE, 3.0)                 # 30.1 Hz from the L signal's centre
    lst._keep(s, other)
    assert s.dets == [newer, other]
    legacy = live.LiveListener(tmp_path / "ce", live.LiveConfig(spec=TINY, render=False),
                               log=lambda m: None)
    s = live.SlotState(Q_TEST)
    for f in (1500.0, 1503.0, 1510.0):
        legacy._keep(s, _Det(f))
    assert [d.f_hz for d in s.dets] == [1503.0, 1510.0]


def test_the_recheck_window_knows_signals_by_centre(tmp_path, monkeypatch):
    """Past detect_until_s, a refused candidate is checked again with the
    signals found skipped by band centre: a CE candidate at an L signal's
    centre is that signal, not one to check again."""
    def run(cands):
        calls = []
        answers = iter([[_Det(1509.5, L)], []])
        monkeypatch.setattr(RX, "candidates", lambda *a, **k: list(cands))

        def detect(prep, spec, live_only=False, *, erase_s=(), cands=None, skip_hz=()):
            calls.append([round(f, 2) for f in skip_hz])
            return next(answers)

        monkeypatch.setattr(RX, "detect", detect)
        monkeypatch.setattr(RX, "receive_slot", lambda *a, dets=None, **k: [])
        cfg = live.LiveConfig(spec=TINY, waveforms=(CE, L), refresh_s=10.0, first_s=25.0,
                              detect_until_s=30.0, recheck_until_s=45.0, render=False)
        lst = live.LiveListener(tmp_path / str(len(cands)), cfg, log=lambda m: None)
        t0 = slot_t0(Q_TEST)
        lst.feed(_hiss(76 * FS), t0 + 60.0)
        lst.step(t0 + 26.0)
        lst.step(t0 + 36.0)
        return calls

    assert run([_Det(1509.5, L), _Det(1500.0, CE)]) == [[]]
    assert run([_Det(1509.5, L), _Det(1500.0, CE), _Det(1600.0, CE)]) == [[], [1500.0]]


class _Due:
    def __init__(self, rows):
        self.rows = rows

    def due(self, q):
        return [dict(r) for r in self.rows]


def test_hints_carry_their_waveform(tmp_path):
    """On the joint path a hint is (f, waveform), from the store's pass or an
    earlier tile, moved by the dial change, inside the band for its
    waveform, one per band centre; a waveform not listened for gives none,
    and CE alone gets bare frequencies, as before L."""
    q = Q_TEST
    rows = [dict(f_hz=1205.47, q=q, waveform=L, frame="l-tiny"),
            dict(f_hz=1700.0, q=q, waveform=CE),
            dict(f_hz=1709.0, q=q, waveform=L),         # the same signal as 1700 CE
            dict(f_hz=220.0, q=q, waveform=L),          # 320 Hz on the new dial: data below 300
            dict(f_hz=210.0, q=q),                      # (no waveform: CE) 310 Hz
            dict(f_hz=2200.0, q=q, frame="l-tiny"),     # the frame says L
            dict(f_hz=2000.0, q=q, waveform=CE, frame="l-tiny"),   # ...and decides (D5)
            dict(f_hz=1000.0, q=q, waveform=L, frame="no-such-frame")]
    dial = tmp_path / "dial"
    dial.mkdir()
    (dial / live.DIAL_FILE).write_text(
        json.dumps({"t": slot_t0(q) - 600.0, "dial_hz": 10_141_000}) + "\n"
        + json.dumps({"t": slot_t0(q + 1), "dial_hz": 10_140_900}) + "\n")

    def hints(ws):
        d = tmp_path / str(ws)
        d.mkdir()
        (d / live.DIAL_FILE).write_text((dial / live.DIAL_FILE).read_text())
        lst = live.LiveListener(d, live.LiveConfig(spec=TINY, waveforms=ws, render=False),
                                store=_Due(rows), log=lambda m: None)
        lst.dial.refresh()
        lst.done_tiles.append(live.Tile(id="p1", q=q, slot_utc="", frame="l-tiny", f_hz=2509.5,
                                        status="complete", mode="B", segment=0, waveform="L",
                                        dial_hz=10_141_000))
        return lst._hints(q + 2, lst.dial.dial_at(slot_t0(q + 2)))

    assert hints((CE, L)) == [(310.0, CE), (1100.0, L), (1305.47, L), (1800.0, CE),
                              (2100.0, L), (2300.0, L), (2609.5, L)]
    assert hints((L,)) == [(1100.0, L), (1305.47, L), (1809.0, L), (2100.0, L), (2300.0, L),
                           (2609.5, L)]
    assert hints(None) == [310.0, 1800.0]
    assert hints((CE,)) == [310.0, 1800.0]


def test_l_band_edge_and_neighbouring_ghosts(tmp_path):
    """An L pass is in the band only with its data (22.07 Hz below the
    pilot) above 300 Hz; the neighbouring-slot ghost rule compares band
    centres, whichever waveform the tile and the pass are."""
    lst = _listener(tmp_path)
    s, other = live.SlotState(Q_TEST), live.SlotState(Q_TEST + 1)
    lst.slots = {Q_TEST: s, Q_TEST + 1: other}
    lst.now = slot_t0(Q_TEST + 1)
    assert not lst._plausible(s, _Pass(322.0, L)) and lst._plausible(s, _Pass(322.1, L))
    assert lst._plausible(s, _Pass(300.0, CE)) and not lst._plausible(s, _Pass(299.9, CE))
    assert lst._plausible(s, _Pass(2700.0, L)) and not lst._plausible(s, _Pass(2700.1, L))
    assert live.in_band(322.1, L) and not live.in_band(322.0, L) and live.in_band(300.0)
    t = lst._tile_for(other, 1509.5, waveform="L", frame="l-tiny")
    t.callsign = "K1ABC"
    assert not lst._plausible(s, _Pass(1500.0, CE))     # its centre: that signal
    assert lst._plausible(s, _Pass(1525.0, CE))         # 25 Hz from it (15.5 from its pilot)
    assert lst._plausible(s, _Pass(1540.0, L))
    assert not lst._plausible(s, _Pass(1519.0, L))      # centre 1509.5, 9.5 from it
    assert lst._match(other, 1500.0) is t and lst._match(other, 1509.5) is None
    near = other.tiles["n"] = live.Tile(id="n", q=Q_TEST + 1, slot_utc="", frame="tiny",
                                        f_hz=1503.5)     # CE, 3.5 Hz above the L one's centre
    assert lst._match(other, 1500.5) is t and lst._match(other, 1503.0) is near
    assert lst._match(other, 1510.0, waveform="L") is t          # (centre 1500.5)
    assert [live._waveform_id(w) for w in ("CE", "L", "?", CE, L)] == [CE, L, CE, CE, L]
    assert live.centre_of(1509.5, "L") == RX.centre_hz(_Pass(1509.5, L)) == 1500.0


def test_the_end_of_slot_receive_takes_the_tuple(monkeypatch):
    """finish_receive with a tuple: one receive_slot call with the tuple and
    the one budget; heard in part, the hints are detected with the tuple and
    the known signals skipped by band centre."""
    calls = []
    monkeypatch.setattr(RX, "receive_slot", lambda prep, spec, *a, dets=None, **k: calls.append(
        ("receive", spec, dets, k.get("max_candidates"), k.get("budget_s"), k.get("hints")))
        or [])
    monkeypatch.setattr(RX, "detect", lambda prep, spec, *a, skip_hz=(), hints=(), **k: (
        calls.append(("detect", spec, list(skip_hz), list(hints))) or [_Det(1600.0, L)]))
    sel = (TINY, L_TINY)
    q, origin = Q_TEST, 15 * FE_FS
    n = origin + int((live.frame_seconds(sel) + PB_TAIL_S) * FE_FS)
    rng = np.random.default_rng(1)
    x = (rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(np.complex64)
    hints = [(1600.0, L)]
    _, heard = live.finish_receive(x, np.ones(n, bool), origin, q, sel, "joint", [], hints)
    assert heard == 1.0
    assert calls == [("receive", sel, None, live.FINISH_MAX_CANDIDATES, live.FINISH_BUDGET_S,
                      hints)]
    calls.clear()
    ok = np.zeros(n, bool)
    ok[:origin + 20 * FE_FS] = True                    # the preamble, and no more
    dets = [_Det(1509.5, L)]
    live.finish_receive(x, ok, origin, q, sel, "joint", dets, hints)
    assert calls[0] == ("detect", sel, [1500.0], hints)
    assert calls[1][:2] == ("receive", sel) and [d.f_hz for d in calls[1][2]] == [1509.5, 1600.0]


@pytest.mark.slow
def test_a_slot_with_both_waveforms_through_the_real_receiver(tmp_path):
    """A TINY slot at -6 dB with an L neighbour 200 Hz above at the same
    power: the joint listener gives each its tile, of its own waveform."""
    q = Q_TEST
    a = unit_rms_latents(TINY.n_data, 0)
    sym = frame.assemble(TINY, None, precoder.precode(a, q))
    sim = chm.simulate(sym, TINY, q, ChannelConfig(
        snr_db=-6.0, seed=2, neighbours=(Neighbour(df_hz=200.0, rel_db=0.0, seed=4,
                                                   waveform=L),)), carrier_hz=1500.0)
    lst = _listener(tmp_path)
    t0 = slot_t0(q)
    lst.ring.write(t0 * FE_FS - int(round(sim.t0_index)), sim.fe)
    lst.step(t0 + 30.0)
    lst.step(t0 + lst.dur + frontend.PB_TAIL_S + 1.0)
    tiles = sorted(lst.tiles(), key=lambda t: t.f_hz)
    assert [(t.waveform, t.frame) for t in tiles] == [("CE", "tiny"), ("L", "l-tiny")]
    assert abs(tiles[0].f_hz - 1500.0) < 0.2 and abs(tiles[1].f_hz - 1700.0) < 0.2
    assert all(t.status == "complete" and t.received == 1.0 for t in tiles)


def _mixed_audio(snr_db, seed, df_hz):
    """(8 kHz audio from t0 - 12 s, CE latents, L latents): a TINY pass at
    1500 Hz and an l-tiny one with its pilot df_hz above, each at snr_db
    (SNR2500 on its own keyed power) against one noise."""
    q = Q_TEST
    a_ce, a_l = unit_rms_latents(TINY.n_data, seed), unit_rms_latents(L_TINY.n_data, seed + 1)
    ce = chm.simulate(frame.assemble(TINY, None, precoder.precode(a_ce, q)), TINY, q,
                      ChannelConfig(seed=seed), carrier_hz=1500.0, post_s=12.0)
    el = chm.simulate(lframe.assemble(L_TINY, None, precoder.precode(a_l, q)), L_TINY, q,
                      ChannelConfig(seed=seed + 1), carrier_hz=1500.0 + df_hz, post_s=12.0)
    assert ce.t0_index == el.t0_index
    p = ce.truth.signal_power
    fe = np.zeros(max(len(ce.fe), len(el.fe)), np.complex128)
    fe[:len(ce.fe)] += ce.fe
    fe[:len(el.fe)] += el.fe * np.sqrt(p / el.truth.signal_power)
    nv = p * FE_FS / (2500.0 * 10 ** (snr_db / 10))
    fe += np.sqrt(nv / 2) * np.random.default_rng([seed, 9]).standard_normal(
        2 * len(fe)).view(np.complex128)
    return chm.fe_to_audio(fe), a_ce, a_l


@pytest.mark.slow
@pytest.mark.parametrize("ws", [(CE, L), (L,)], ids=["both", "l"])
def test_a_mixed_slot_heard_as_audio(tmp_path, ws):
    """A TINY and an l-tiny signal 80 Hz apart at -12 dB, fed as audio in
    half-second chunks on past the slot's end: each waveform listened for
    gets its own tile, complete, of its own waveform and frame, with W
    honest about its latents; --just-l does not take the CE one as L."""
    audio, a_ce, a_l = _mixed_audio(-12.0, 31, 80.0)
    lst = _listener(tmp_path, ws, refresh_s=15.0)
    got = []
    apply = lst._finish_apply
    lst._finish_apply = lambda s, results: (got.extend(p for _, ps, _ in results for p in ps),
                                            apply(s, results))
    t0 = slot_t0(Q_TEST)
    n = FS // 2
    for k in range(0, len(audio), n):
        x = audio[k:k + n]
        t_end = t0 - 12.0 + (k + len(x) - 1) / FS
        lst.feed(x, t_end)
        lst.step(t_end)
    assert t_end > t0 + lst.dur + PB_TAIL_S and Q_TEST in lst.finished
    want = [("CE", "tiny", 1500.0), ("L", "l-tiny", 1580.0)][2 - len(ws):]
    tiles = sorted(lst.tiles(), key=lambda t: t.f_hz)
    assert [(t.waveform, t.frame, round(t.f_hz)) for t in tiles] == want
    for t in tiles:
        p = next(p for p in got if abs(p.f_hz - t.f_hz) < 0.01)
        lat = latent_snr_db(p.z, a_l if p.waveform == L else a_ce)
        assert t.status == "complete" and t.received == 1.0 and t.heard == 1.0
        assert abs(t.mean_w_db - lat) < 1.5 and lat > (5.5 if p.waveform == L else 3.0)
        conf = [c for c in t.confidence if c is not None]
        assert len(conf) > 60 and abs(np.median(conf) - t.mean_w_db) < 1.0
    lst.write_state()
    st = json.loads((tmp_path / "state.json").read_text())
    assert st["waveforms"] == ["CE", "L"][2 - len(ws):]
    assert [t["waveform"] for t in sorted(st["tiles"], key=lambda t: t["f_hz"])] == \
        [w[0] for w in want]

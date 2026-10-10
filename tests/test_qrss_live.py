"""The live listener (`sstvae.qrss.live`, `qrss_listen.py`).

Fast: the streaming front end equals the batch one sample for sample,
the ring and the slot capture account for what was and was not heard,
the clock follows on or restarts, and the slot bookkeeping (tiles,
their ids across refreshes, finishing, the state file) runs against a
stub receiver. One TINY slot goes through the real receiver half heard
and then whole (about 5 s): the half-heard pass carries no weight in
blocks it has not heard yet.
"""

import json

import numpy as np
import pytest

from qrss_helpers import Q_TEST, unit_rms_latents
from sstvae.qrss import channel as chm
from sstvae.qrss import frame, frontend, live, precoder
from sstvae.qrss import receiver as RX
from sstvae.qrss.channel import ChannelConfig
from sstvae.qrss.constants import FE_FS, FS, T_SYM
from sstvae.qrss.frontend import slot_t0

TINY = frame.TINY


def test_stream_fe_equals_batch():
    x = np.random.default_rng(3).standard_normal(30000)
    ref = frontend.audio_to_fe(x)
    s = live.StreamFE(block=4096)
    out, a = [], 0
    for n in (100, 3000, 777, 5000, 16000, 5123):
        out += s.push(a, x[a:a + n])
        a += n
    n0 = out[0][0]
    y = np.concatenate([o[1] for o in out])
    assert n0 == live.HALO // 2 and len(y) > 14000
    assert np.array_equal(y, ref[n0:n0 + len(y)])


def test_stream_fe_restarts_on_the_grid():
    s = live.StreamFE()
    s.push(0, np.zeros(4000))
    out = s.push(10_007, np.zeros(4000))      # not following on
    assert s.a0 % live.ALIGN == 0
    assert out and out[0][0] == (10_016 + live.HALO) // 2


def test_ring_wraps_and_holes_read_as_unheard():
    r = live.FeRing(span_s=1.0)                # 4000 samples
    r.write(100, np.arange(3000, dtype=np.complex64))
    r.write(3100, np.arange(3000, 6000, dtype=np.complex64))   # wraps
    x, ok = r.read(2100, 6100)
    assert ok.all() and np.array_equal(x.real, np.arange(2000, 6000))
    x, ok = r.read(1000, 2200)                 # partly older than the ring
    assert not ok[:1100].any() and ok[1100:].all()
    r.write(9000, np.ones(100, dtype=np.complex64))           # a hole before it
    x, ok = r.read(6100, 9100)
    assert not ok[:2900].any() and ok[2900:].all()


def test_slot_capture_fills_and_erases_what_was_not_heard():
    q = Q_TEST
    t0 = slot_t0(q)
    r = live.FeRing(live.frame_seconds(TINY) + 60)
    rng = np.random.default_rng(0)
    a = (t0 - 15) * FE_FS
    n = 40 * FE_FS                              # heard from t0 - 15 s to t0 + 25 s
    r.write(a, (rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(np.complex64))
    prep, erase, heard = live.slot_capture(r, q, TINY)
    assert prep.t0_index == 15 * FE_FS
    assert erase == [(25.0, live.frame_seconds(TINY) + frontend.PB_TAIL_S)]
    assert heard == pytest.approx(25.0 / live.frame_seconds(TINY), rel=1e-3)
    p = np.abs(prep.fe) ** 2
    # heard and filled parts at the same normalised level (median power 1)
    assert np.median(p[:30 * FE_FS]) == pytest.approx(1.0, rel=0.05)
    assert np.median(p[45 * FE_FS:]) == pytest.approx(1.0, rel=0.05)


def test_clock_follows_on_and_restarts_after_a_gap(tmp_path):
    L = live.LiveListener(tmp_path, live.LiveConfig(spec=TINY, render=False),
                          log=lambda m: None)
    t = 1_000_000.0
    L.feed(np.zeros(FS), t)
    first = L.fe.next_index
    L.feed(np.zeros(FS), t + 1.0 + 0.3)        # 0.3 s of clock disagreement: follows on
    assert L.fe.next_index == first + FS
    L.feed(np.zeros(FS), t + 10.0)             # 7.7 s missing: restarts there
    assert L.fe.next_index == int(round((t + 10.0) * FS)) + 1
    _, ok = L.ring.read(int((t + 2.5) * FE_FS), int((t + 8.5) * FE_FS))
    assert not ok.any()
    assert any("missing" in s for s in L.lines)
    # a burst (ten seconds read at once) follows on: a pipe loses nothing
    nxt = L.fe.next_index
    for k in range(10):
        L.feed(np.zeros(FS), t + 10.5 + 0.01 * k)
    assert L.fe.next_index == nxt + 10 * FS
    # audio early for EARLY_WINDOW_S on end: the sound card runs fast, step back
    t2 = t + 20.0
    for k in range(40):
        L.feed(np.zeros(FS), t2 - 1.0 + k)      # the stream 2 s ahead of every stamp
    assert any("ahead of the clock" in s for s in L.lines)


class _Det:
    def __init__(self, f):
        self.f_hz = f


class _Report:
    def __init__(self, z, snr=-12.0):
        self.snr2500_db, self.z_ref = snr, z


class _Pass:
    def __init__(self, f, z=20.0, header=None, n=TINY.n_data, snr=-12.0):
        self.f_hz, self.report, self.header, self.cw = f, _Report(z, snr), header, None
        self.uid = f"u{f}"
        self.w = np.ones(n, dtype=np.float32)
        self.z = np.zeros(n, dtype=np.float32)


def test_slot_bookkeeping_with_a_stub_receiver(tmp_path, monkeypatch):
    calls = []

    def detect(prep, spec, live_only=False):
        calls.append(("detect", live_only))
        return [_Det(1500.0), _Det(1610.0)]

    freqs = iter([(1500.3, 1611.0), (1501.9, 1609.5), (1500.0, 1610.0)])

    def receive_slot(prep, spec, live_only=False, estimator="joint", *, erase_s=(),
                     round_b=True, dets=None, max_candidates=None, budget_s=None):
        calls.append(("receive", round_b, bool(erase_s)))
        return [_Pass(f) for f in next(freqs)]

    monkeypatch.setattr(RX, "detect", detect)
    monkeypatch.setattr(RX, "receive_slot", receive_slot)
    cfg = live.LiveConfig(spec=TINY, refresh_s=10.0, first_s=25.0, render=False)
    L = live.LiveListener(tmp_path, cfg, log=lambda m: None)
    q = Q_TEST
    t0 = slot_t0(q)
    L.feed(np.zeros(76 * FS), t0 + 60.0)       # the whole TINY slot heard
    assert not L.step(t0 - 20.0)
    assert L.step(t0 + 30.0) and set(L.slots) == {q}
    tiles = {round(t.f_hz): t.id for t in L.tiles()}
    assert len(tiles) == 2
    L.step(t0 + 35.0)                          # not yet due
    L.step(t0 + 41.0)                          # due: frequencies moved under 5 Hz
    assert sorted(t.id for t in L.tiles()) == sorted(tiles.values())
    assert [c[1] for c in calls if c[0] == "receive"] == [False, False]   # live: round A
    L.step(t0 + live.frame_seconds(TINY) + frontend.PB_TAIL_S + 0.5)
    assert calls[-1] == ("receive", True, False)        # the whole slot, round B, nothing erased
    assert q not in L.slots and q in L.finished
    assert {t.status for t in L.tiles()} == {"complete"}
    L.write_state()
    st = json.loads((tmp_path / "state.json").read_text())
    assert [t["id"] for t in st["tiles"]] == [t.id for t in L.tiles()]
    assert st["frame"] == "tiny" and st["listening"]
    # a restart keeps the finished tiles; they expire after keep_done_s
    L2 = live.LiveListener(tmp_path, cfg, log=lambda m: None)
    assert len(L2.tiles()) == 2
    L2.step(L2.tiles()[0].updated + cfg.keep_done_s + 1)
    assert L2.tiles() == []


def test_implausible_and_neighbour_passes_get_no_tile(tmp_path):
    L = live.LiveListener(tmp_path, live.LiveConfig(spec=TINY, render=False),
                          log=lambda m: None)
    s = live.SlotState(Q_TEST)
    other = live.SlotState(Q_TEST + 1)
    L.slots = {Q_TEST: s, Q_TEST + 1: other}
    L.now = slot_t0(Q_TEST + 1)
    t = L._tile_for(other, 1500.0)
    assert L._plausible(s, _Pass(1500.0))      # the neighbour has no header yet
    t.callsign = "K1ABC"
    assert not L._plausible(s, _Pass(1512.0))  # its signal, at the wrong timing
    assert L._plausible(s, _Pass(1560.0))
    assert not L._plausible(s, _Pass(float("inf")))
    assert not L._plausible(s, _Pass(5000.0))
    # strong enough that its header would have decoded: a carrier or a voice
    assert not L._plausible(s, _Pass(1560.0, snr=3.1))
    assert not L._plausible(s, _Pass(1560.0, snr=float("nan")))


def test_tiny_slot_half_heard_then_whole(tmp_path):
    """The real receiver on a TINY slot at -6 dB: half heard, the blocks not
    yet heard carry no weight; heard whole, every latent does."""
    q = Q_TEST
    a = unit_rms_latents(TINY.n_data, 0)
    sym = frame.assemble(TINY, None, precoder.precode(a, q))
    sim = chm.simulate(sym, TINY, q, ChannelConfig(snr_db=-6.0, seed=2), carrier_hz=1500.0)
    L = live.LiveListener(tmp_path, live.LiveConfig(spec=TINY, render=False),
                          log=lambda m: None)
    t0 = slot_t0(q)
    n0 = t0 * FE_FS - int(round(sim.t0_index))
    half = int(round(sim.t0_index + 0.5 * live.frame_seconds(TINY) * FE_FS))
    L.ring.write(n0, sim.fe[:half])
    cut = (half - sim.t0_index) / FE_FS
    L.step(t0 + cut)
    (tile,) = L.tiles()
    assert abs(tile.f_hz - 1500.0) < 0.2 and tile.status == "receiving"
    # the preamble is 20 of the frame's 53 s, so a quarter of the data is heard
    assert 0.1 < tile.received < 0.45 and tile.heard == pytest.approx(0.5, abs=0.01)
    prep, erase, _ = live.slot_capture(L.ring, q, TINY)
    p = RX.receive_slot(prep, TINY, erase_s=erase, round_b=False, dets=list(L.slots[q].dets))[0]
    # data symbol k of a block: each 64-latent block is spread over 64
    # consecutive data symbols, so a block whose symbols all come after
    # the cut has W = 0, and one wholly before it has W > 0
    lay = frame.layout(TINY)
    t_data = np.asarray(lay.pos)[np.asarray(lay.data)] * T_SYM
    bi = precoder.block_index(TINY.n_data)
    late = np.array([t_data[bi == b].min() > cut for b in bi])
    early = np.array([t_data[bi == b].max() < cut - 1.0 for b in bi])
    assert late.any() and early.any()
    assert np.all(p.w[late] == 0) and np.all(p.w[early] > 0)

    L.ring.write(n0 + half, sim.fe[half:])
    L.step(t0 + live.frame_seconds(TINY) + frontend.PB_TAIL_S + 1.0)
    (tile,) = L.tiles()
    assert tile.status == "complete" and tile.received == 1.0 and tile.heard == 1.0


def test_one_listener_per_directory(tmp_path):
    a = live.dir_lock(tmp_path)
    assert a is not None
    assert live.dir_lock(tmp_path) is None          # a second writer is refused
    a.close()
    b = live.dir_lock(tmp_path)                     # and allowed once the first is gone
    assert b is not None
    b.close()


def test_atomic_writes_from_two_writers_do_not_collide(tmp_path):
    """Two writers of one passband index file: each write's temporary file
    is its own, so neither os.replace moves the other's away."""
    import threading
    p = tmp_path / "fe_1.json"
    errors = []

    def write(k):
        try:
            for i in range(300):
                frontend._atomic_write(p, f"[{k}, {i}]")
        except OSError as e:
            errors.append(e)

    ts = [threading.Thread(target=write, args=(k,)) for k in range(2)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert not errors
    assert json.loads(p.read_text())[1] == 299


def test_a_slot_mostly_unheard_gets_no_whole_slot_search(tmp_path, monkeypatch):
    """Joined late (after the preamble, a third of the slot heard): the end
    of the slot receives only what the preamble found, which here is
    nothing, rather than searching the slot; on a phone band that search
    ran for longer than a slot."""
    calls = []
    monkeypatch.setattr(RX, "detect", lambda *a, **k: calls.append("detect") or [])
    monkeypatch.setattr(RX, "receive_slot", lambda *a, **k: calls.append(("receive", k)) or [])
    L = live.LiveListener(tmp_path, live.LiveConfig(spec=TINY, render=False),
                          log=lambda m: None)
    q = Q_TEST
    t0 = slot_t0(q)
    dur = live.frame_seconds(TINY)
    late = t0 + 0.65 * dur
    L.feed(np.zeros(int((t0 + dur + 5 - late) * FS)), t0 + dur + 5)
    L.step(late)
    L.step(t0 + dur + frontend.PB_TAIL_S + 0.5)
    assert q in L.finished and calls == []
    assert any("no whole-slot search" in s for s in L.lines)


def test_receives_run_on_the_worker_and_state_keeps_moving(tmp_path, monkeypatch):
    """With background on, a slow receive does not hold up feeding or
    state.json, which says what the worker is doing."""
    import threading
    release = threading.Event()
    started = threading.Event()

    def slow(*a, **k):
        started.set()
        release.wait(10)
        return []

    monkeypatch.setattr(RX, "receive_slot", slow)
    monkeypatch.setattr(RX, "detect", lambda *a, **k: [])
    L = live.LiveListener(tmp_path, live.LiveConfig(spec=TINY, render=False, background=True),
                          log=lambda m: None)
    q = Q_TEST
    t0 = slot_t0(q)
    end = t0 + live.frame_seconds(TINY) + frontend.PB_TAIL_S + 0.5
    L.feed(np.zeros(int((end - t0 + 15) * FS)), end)
    with L.lock:
        L.step(t0 + 30.0)             # the slot starts (its refresh finds nothing)
    for _ in range(100):
        if L.idle():
            break
        import time as _t
        _t.sleep(0.05)
    with L.lock:
        L.step(end)
    assert started.wait(5)
    with L.lock:                      # the loop can still feed and write state
        L.feed(np.zeros(FS), end + 1.0)
        L.write_state()
    st = json.loads((tmp_path / "state.json").read_text())
    assert st["busy"] and "finish" in st["busy"]
    release.set()
    for _ in range(100):
        if L.idle():
            break
        import time as _t
        _t.sleep(0.05)
    with L.lock:
        L.step(end + 2.0)
    assert q in L.finished and L.idle()


def test_a_restarted_listener_reads_the_preamble_back(tmp_path):
    """A listener started mid-slot refills its ring from the passband store
    an earlier one wrote, so the slot's preamble (and its tile) is not lost."""
    q = Q_TEST
    a = unit_rms_latents(TINY.n_data, 0)
    sym = frame.assemble(TINY, None, precoder.precode(a, q))
    sim = chm.simulate(sym, TINY, q, ChannelConfig(snr_db=-6.0, seed=2), carrier_hz=1500.0)
    t0 = slot_t0(q)
    n0 = t0 * FE_FS - int(round(sim.t0_index))
    half = int(round(sim.t0_index + 0.5 * live.frame_seconds(TINY) * FE_FS))
    pb = frontend.PassbandStore(tmp_path / "passband")
    pb.write(n0 / FE_FS, sim.fe[:half], expire=False)      # what the first listener heard
    now = (n0 + half) / FE_FS
    L = live.LiveListener(tmp_path / "live", live.LiveConfig(spec=TINY, render=False),
                          passband=pb, log=lambda m: None)
    assert L.step(now) is not None and not L.tiles()     # nothing heard: no preamble
    L = live.LiveListener(tmp_path / "live2", live.LiveConfig(spec=TINY, render=False),
                          passband=pb, log=lambda m: None)
    assert L.backfill(now) == pytest.approx(half / FE_FS, abs=1.0)
    L.step(now)
    (tile,) = L.tiles()
    assert abs(tile.f_hz - 1500.0) < 0.2 and tile.status == "receiving"

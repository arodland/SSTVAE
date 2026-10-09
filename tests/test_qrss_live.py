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
    def __init__(self, z):
        self.snr2500_db, self.z_ref = -12.0, z


class _Pass:
    def __init__(self, f, z=20.0, header=None, n=TINY.n_data):
        self.f_hz, self.report, self.header, self.cw = f, _Report(z), header, None
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
                     round_b=True, dets=None):
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

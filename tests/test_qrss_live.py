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


def _steady(t, secs, n=2000):
    """(chunk, read time) of audio heard from t on, read as it arrives (n samples a chunk)."""
    return [(np.zeros(n), t + (k + 1) * n / FS) for k in range(int(round(secs * FS / n)))]


def _feed(L, chunks, stamper=None):
    for x, t in chunks:
        for y, te in (stamper.push(x, t) if stamper is not None else [(x, t)]):
            L.feed(y, te)


def test_a_burst_after_a_stall_is_not_missing_audio(tmp_path):
    """The app's audio pump held up for 3 s, then the 3 s arriving at once:
    stamped by read time, the first piece reads 2.75 s late and the
    listener leaves a hole; the stamper sees the backlog drain and the
    stream follows on, with nothing missing."""
    T = 1_000_000.0
    stall = [(np.zeros(2000), T + 23.0 + 1e-4 * k) for k in range(12)]   # heard T+20..T+23
    chunks = _steady(T, 20.0) + stall + _steady(T + 23.0, 40.0)
    old = live.LiveListener(tmp_path / "old", live.LiveConfig(spec=TINY, render=False),
                            log=lambda m: None)
    _feed(old, chunks)
    assert any("missing" in s for s in old.lines)          # what stamping by read time did
    L = live.LiveListener(tmp_path / "new", live.LiveConfig(spec=TINY, render=False),
                          log=lambda m: None)
    st = live.Stamper()
    _feed(L, chunks, st)
    for y, te in st.flush():
        L.feed(y, te)
    assert not any("missing" in s or "ahead" in s for s in L.lines)
    assert L.fe.next_index == int(round((T + 63.0) * FS)) + 1     # on the clock at the end
    _, ok = L.ring.read(int((T + 1.0) * FE_FS), int((T + 62.0) * FE_FS))
    assert ok.all()


def test_missing_audio_is_still_missing(tmp_path):
    """10 s that never arrive (the receive muted): held, then a hole of 10 s
    with the audio after it on the clock."""
    T = 1_000_000.0
    chunks = _steady(T, 20.0) + _steady(T + 30.0, 20.0)
    L = live.LiveListener(tmp_path, live.LiveConfig(spec=TINY, render=False),
                          log=lambda m: None)
    _feed(L, chunks, live.Stamper())
    (gap,) = [s for s in L.lines if "missing" in s]
    assert "10.0 s of audio missing" in gap
    _, ok = L.ring.read(int((T + 32.0) * FE_FS), int((T + 44.0) * FE_FS))
    assert ok.all()
    _, ok = L.ring.read(int((T + 21.0) * FE_FS), int((T + 29.0) * FE_FS))
    assert not ok.any()


def test_audio_queued_for_a_starting_listener_starts_when_it_was_heard(tmp_path):
    """8 s of audio queued while the listener started up, read at once,
    then audio as it arrives: the stream starts where the queue did."""
    T = 1_000_000.0
    queued = [(np.zeros(2000), T + 8.0 + 1e-4 * k) for k in range(32)]
    L = live.LiveListener(tmp_path, live.LiveConfig(spec=TINY, render=False),
                          log=lambda m: None)
    _feed(L, queued + _steady(T + 8.0, 40.0), live.Stamper())
    assert not any("missing" in s or "ahead" in s for s in L.lines)
    _, ok = L.ring.read(int((T + 1.0) * FE_FS), int((T + 47.0) * FE_FS))
    assert ok.all()
    assert L.fe.next_index > int(round((T + 47.0) * FS))


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
    # complete: every bin of the band has a value (W = 1 is 0 dB)
    conf = st["tiles"][0]["confidence"]
    assert len(conf) == RX.CONF_BINS and conf[-1] == 0.0
    # a restart keeps the finished tiles; they expire after keep_done_s
    L2 = live.LiveListener(tmp_path, cfg, log=lambda m: None)
    assert len(L2.tiles()) == 2
    L2.step(L2.tiles()[0].updated + cfg.keep_done_s + 1)
    assert L2.tiles() == []


def test_an_empty_slot_is_searched_less_often(tmp_path):
    """Every refresh_s until detect_until_s; after it, a slot where nothing
    was found is searched every empty_refresh_s, and one with a signal is
    still received every refresh_s."""
    cfg = live.LiveConfig(render=False)            # FULL: 120 s, 25 s, 300 s, 300 s
    end = live.frame_seconds(cfg.spec)

    def refreshes(found_at):
        L = live.LiveListener(tmp_path / f"f{found_at}", cfg, log=lambda m: None)
        t0 = slot_t0(Q_TEST)
        seen = []

        def refresh(s):
            if s.q != Q_TEST:
                return                            # (the slot before, still on the air)
            seen.append(round(L.now - t0))
            if found_at is not None and len(seen) >= found_at:
                s.dets = [_Det(1500.0)]

        L._refresh = refresh
        for t in np.arange(0.0, end, 5.0):
            L.step(t0 + t)
        return seen

    assert refreshes(None) == [25, 145, 265, 565, 865, 1165, 1465, 1765]
    found = refreshes(2)                          # found at the second search
    assert found[:2] == [25, 145] and np.all(np.diff(found) == 120) and found[-1] > end - 120


def test_confidence_over_the_pass():
    """The band under a tile: per-latent SNR over time, from W and the header LLRs."""
    from types import SimpleNamespace

    spec = frame.FULL
    lay = frame.layout(spec)
    t_data = lay.pos[lay.data] * T_SYM
    w = np.where(t_data < 900.0, 10 ** 0.4, 0.0).astype(np.float32)   # 4 dB, then nothing
    s2 = 0.5                                                          # header at 3 dB
    rng = np.random.default_rng(3)
    x = rng.choice([-1.0, 1.0], len(lay.hdr_bits))
    llr = (2 * (x + rng.normal(0.0, s2 ** 0.5, len(x))) / s2).astype(np.float32)
    psi = np.ones(spec.n_pos, dtype=np.float32)
    psi[int(1200 / T_SYM):int(1300 / T_SYM)] = 0.0                    # not heard
    p = SimpleNamespace(w=w, hdr_llr=llr, psi=psi)
    c = RX.confidence_db(p, spec)
    bin_s = spec.keyed_end_pos * T_SYM / RX.CONF_BINS

    def at(t):
        return c[int(t / bin_s)]

    assert len(c) == RX.CONF_BINS
    assert c[0] is None                                   # the preamble: nothing to measure
    assert abs(at(50.0) - 3.0) < 0.6                      # header, from its LLRs
    assert at(500.0) == 4.0                               # data, from W
    assert at(1000.0) == RX.CONF_FLOOR_DB                 # heard, no information
    assert at(1250.0) is None                             # not heard
    live_c = RX.confidence_db(p, spec, elapsed_s=600.0)   # 600 s into the pass
    assert at(500.0) == live_c[int(500.0 / bin_s)] and live_c[int(700.0 / bin_s)] is None


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


def _until(cond, timeout=10.0):
    import time as _t
    deadline = _t.monotonic() + timeout
    while not cond() and _t.monotonic() < deadline:
        _t.sleep(0.02)
    assert cond()


def test_a_finishing_slot_shows_the_whole_pass_meanwhile(tmp_path, monkeypatch):
    """The end-of-slot receive takes minutes. Meanwhile the tile reads
    "finishing", and one more live receive of the whole frame takes it to
    the pass's end, where it used to stay at the last refresh (96% for
    five minutes in a field test). A tile's progress is when its audio was read,
    not when its receive returned."""
    import threading

    reads = []

    def receive_slot(*a, **k):
        reads.append(L.now)
        with L.lock:
            L.now += 10.0             # the loop steps on while a receive runs
        return [_Pass(1500.0)]

    monkeypatch.setattr(RX, "detect", lambda *a, **k: [_Det(1500.0)])
    monkeypatch.setattr(RX, "receive_slot", receive_slot)
    L = live.LiveListener(tmp_path, live.LiveConfig(spec=TINY, render=False, background=True),
                          log=lambda m: None)
    release = threading.Event()
    finish = L._finish
    L._finish = lambda s: (release.wait(10), finish(s))
    q = Q_TEST
    t0 = slot_t0(q)
    dur = live.frame_seconds(TINY)
    end = t0 + dur + frontend.PB_TAIL_S + 0.5
    L.feed(np.zeros(int((end - t0 + 15) * FS)), end)
    live_idle = lambda: L._busy["live"] is None and L._jobs.empty()   # noqa: E731
    with L.lock:
        L.step(t0 + 30.0)
    _until(lambda: live_idle() and reads)
    (tile,) = L.tiles()
    assert tile.status == "receiving" and tile.progress == pytest.approx(30.0 / dur)
    with L.lock:
        L.step(end)
    assert tile.status == "finishing"
    _until(lambda: live_idle() and len(reads) == 2)
    assert tile.status == "finishing" and tile.progress == 1.0 and tile.heard == 1.0
    with L.lock:
        L.step(end + 1.0)             # no second final refresh
    assert len(reads) == 2
    release.set()
    _until(L.idle)
    with L.lock:
        L.step(end + 2.0)
    assert q in L.finished and tile.status == "complete"
    # a listener stopped while it finished re-finishes the slot, not lists the tile
    t = live.Tile(id="x", q=q + 1, slot_utc="", frame="tiny", f_hz=1500.0, status="finishing")
    (tmp_path / "s2").mkdir()
    (tmp_path / "s2" / "state.json").write_text(json.dumps({"tiles": [t.to_json()]}))
    L2 = live.LiveListener(tmp_path / "s2", live.LiveConfig(spec=TINY, render=False),
                           log=lambda m: None)
    assert L2.tiles() == [] and q + 1 not in L2.finished
    L2.close()
    L.close()


def test_the_end_of_slot_receive_runs_in_its_own_process(tmp_path):
    """With finish_process, the end-of-slot receive runs in a child process
    while the live worker stays free, gets the same passes, and a child that
    dies is replaced at the next receive."""
    import os
    import time as _t

    q = Q_TEST
    a = unit_rms_latents(TINY.n_data, 0)
    sym = frame.assemble(TINY, None, precoder.precode(a, q))
    sim = chm.simulate(sym, TINY, q, ChannelConfig(snr_db=-10.0, seed=2), carrier_hz=1500.0)
    t0 = slot_t0(q)
    n0 = t0 * FE_FS - int(round(sim.t0_index))
    end = t0 + live.frame_seconds(TINY) + frontend.PB_TAIL_S + 0.5
    cfg = live.LiveConfig(spec=TINY, render=False, background=True, finish_process=True)
    L = live.LiveListener(tmp_path, cfg, log=lambda m: None)
    try:
        L.ring.write(n0, sim.fe)
        L._refresh = lambda s: None           # (live receives are not what is tested)
        got = []
        apply = L._finish_apply
        L._finish_apply = lambda s, results: (got.append(results), apply(s, results))
        with L.lock:
            L.step(t0 + 30.0)                 # the slot starts
            L.step(end)                       # and ends
        s = L.slots[q]
        child = None
        deadline = _t.monotonic() + 120.0
        while not (s.done and L.idle()) and _t.monotonic() < deadline:
            if child is None and L._proc._proc is not None:
                child = L._proc._proc.pid
                assert L._busy["live"] is None and "finish" in L.busy
            _t.sleep(0.05)
        assert s.done and L.idle(), "the end-of-slot receive did not finish"
        assert child is not None and child != os.getpid()
        with L.lock:
            L.step(end + 1.0)
        assert q in L.finished
        (tile,) = L.tiles()
        assert tile.status == "complete" and abs(tile.f_hz - 1500.0) < 0.5
        # the same receive in this process gives the same pass
        x, ok, origin = live.capture_read(L.ring, q, TINY)
        here, heard = live.finish_receive(x, ok, origin, q, TINY, cfg.estimator, [])
        ((_, there, heard2),) = got[0]
        assert heard == heard2 and len(here) == len(there) == 1
        assert np.array_equal(here[0].z, there[0].z) and np.array_equal(here[0].w, there[0].w)
        # a child that dies is replaced
        L._proc._proc.kill()
        L._proc._proc.join(10.0)
        empty = (np.zeros(len(x), np.complex64), np.zeros(len(x), bool), origin, q, TINY,
                 cfg.estimator, [])
        assert L._proc.call(empty) == ([], 0.0)
    finally:
        L.close()
    assert L._proc._proc is None


def test_the_dial_log_splits_time_into_epochs(tmp_path):
    """The app's dial file: lines in any order, a bad or half-written line
    skipped, a repeated dial not a change, the first dial in force before
    it, DIAL_GUARD_S taken off around each change, re-read on change."""
    path = tmp_path / live.DIAL_FILE
    D = live.DialLog(path)
    D.refresh()
    assert D.spans(0.0, 100.0) == [(-np.inf, np.inf, None)]
    assert D.span_at(50.0) == (-np.inf, np.inf, None)
    lines = [{"t": 1000.0, "dial_hz": 10_140_000, "why": "start"},
             {"t": 3000.0, "dial_hz": 7_039_000},
             {"t": 2000.0, "dial_hz": 14_096_000, "why": "menu"},
             {"t": 1500.0, "dial_hz": 10_140_000, "why": "radio"},     # no change
             {"t": 2500.0}]
    path.write_text("\n".join(json.dumps(d) for d in lines) + '\n{"t": 3500.0, "dial')
    D.refresh()
    assert D.points == [(1000.0, 10_140_000), (2000.0, 14_096_000), (3000.0, 7_039_000)]
    g0, g1 = live.DIAL_GUARD_S
    assert D.span_at(0.0) == (-np.inf, 2000.0 - g0, 10_140_000)          # before the first line
    assert D.span_at(2010.0) == (2000.0 + g1, 3000.0 - g0, 14_096_000)   # lo after t: settling
    assert D.span_at(9999.0) == (3000.0 + g1, np.inf, 7_039_000)
    assert D.spans(1900.0, 2100.0) == [(-np.inf, 2000.0 - g0, 10_140_000),
                                       (2000.0 + g1, 3000.0 - g0, 14_096_000)]
    assert D.spans(3100.0, 3200.0) == [(3000.0 + g1, np.inf, 7_039_000)]
    # the app's 48 h trim replaces the file: read again by path
    # a dial left and come back to is one dial, heard twice
    assert D.by_dial(0.0, 9999.0) == [(10_140_000, [(-np.inf, 2000.0 - g0)]),
                                      (14_096_000, [(2000.0 + g1, 3000.0 - g0)]),
                                      (7_039_000, [(3000.0 + g1, np.inf)])]
    path.write_text("\n".join(json.dumps({"t": t, "dial_hz": hz}) for t, hz in
                              [(1000.0, 7_040_000), (2000.0, 7_041_000), (2001.0, 7_040_000)]))
    D.refresh()
    assert D.by_dial(0.0, 9999.0) == [(7_040_000, [(-np.inf, 2000.0 - g0), (2001.0 + g1, np.inf)])]
    assert D.dial_at(2000.5) == 7_041_000
    path.write_text(json.dumps({"t": 5000.0, "dial_hz": 3_568_600}) + "\n")
    D.refresh()
    assert D.points == [(5000.0, 3_568_600)]
    path.unlink()
    D.refresh()
    assert D.points == []


def test_restrict_keeps_only_what_was_heard_on_one_dial():
    q = Q_TEST
    t0 = slot_t0(q)
    origin = 10 * FE_FS
    ok = np.ones(100 * FE_FS, dtype=bool)
    ok[50 * FE_FS:55 * FE_FS] = False                 # a hole stays a hole
    assert live.restrict(ok, origin, q, [(-np.inf, np.inf)]) is ok
    r = live.restrict(ok, origin, q, [(t0 + 20.0, t0 + 60.0)])
    assert np.flatnonzero(np.diff(r.astype(np.int8))).tolist() == [
        30 * FE_FS - 1, 50 * FE_FS - 1, 55 * FE_FS - 1, 70 * FE_FS - 1]
    assert not live.restrict(ok, origin, q, [(t0 + 200.0, np.inf)]).any()
    assert not live.restrict(ok, origin, q, []).any()
    r = live.restrict(ok, origin, q, [(-np.inf, t0 + 5.0), (t0 + 80.0, np.inf)])
    assert r[:15 * FE_FS].all() and not r[15 * FE_FS:90 * FE_FS].any() and r[90 * FE_FS:].all()
    assert live.same_dial(None, 7_040_000) and not live.same_dial(7_040_000, 14_096_000)


def test_a_slot_heard_on_two_dials_is_received_on_each(tmp_path, monkeypatch):
    """The dial changes twenty seconds into a slot. Live, the slot is
    received on the dial it started on with everything after the change
    erased; at its end each dial is received on its own, and a signal at
    the same audio frequency on the other dial is another tile."""
    monkeypatch.setattr(live, "DIAL_GUARD_S", (2.0, 2.0))   # (TINY is 53 s: room after it)
    calls = []

    def detect(prep, spec, live_only=False):
        return [_Det(1500.0)]

    def receive_slot(prep, spec, live_only=False, estimator="joint", *, erase_s=(),
                     round_b=True, dets=None, max_candidates=None, budget_s=None):
        calls.append((round_b, bool(dets), list(erase_s)))
        return [_Pass(1500.0)]

    monkeypatch.setattr(RX, "detect", detect)
    monkeypatch.setattr(RX, "receive_slot", receive_slot)
    L = live.LiveListener(tmp_path, live.LiveConfig(spec=TINY, render=False),
                          log=lambda m: None)
    q = Q_TEST
    t0 = slot_t0(q)
    end = t0 + live.frame_seconds(TINY) + frontend.PB_TAIL_S
    (tmp_path / live.DIAL_FILE).write_text(
        json.dumps({"t": t0 - 600.0, "dial_hz": 10_138_700, "why": "start"}) + "\n"
        + json.dumps({"t": t0 + 20.0, "dial_hz": 14_095_600, "why": "schedule"}) + "\n")
    start = t0 - frontend.PB_LEAD_S - 1.0
    L.feed(np.zeros(int((end + 5 - start) * FS)), end + 5)
    L.step(t0 + 30.0)
    (tile,) = L.tiles()
    assert tile.dial_hz == 10_138_700
    ((round_b, dets, erase),) = calls
    assert not round_b and dets and erase[0] == (18.0, live.frame_seconds(TINY) + frontend.PB_TAIL_S)
    calls.clear()
    L.step(end + 0.5)
    assert q in L.finished
    assert [c[:2] for c in calls] == [(True, True), (True, False)]   # its signal; then a search
    assert calls[0][2][0][0] == 18.0 and calls[1][2][0] == (-frontend.PB_LEAD_S, 22.0)
    tiles = sorted(L.tiles(), key=lambda t: t.dial_hz)
    assert [t.dial_hz for t in tiles] == [10_138_700, 14_095_600]
    assert all(t.status == "complete" and t.f_hz == 1500.0 for t in tiles)
    assert any("on 2 dials" in s for s in L.lines)
    L.write_state()
    st = json.loads((tmp_path / "state.json").read_text())
    assert sorted(t["dial_hz"] for t in st["tiles"]) == [10_138_700, 14_095_600]


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


def test_a_headerless_pass_gets_a_provisional_picture(tmp_path):
    """No header: the picture is drawn as segment 0 of a mode A send and
    marked so; once a header decodes it is redrawn from it, unmarked."""
    from types import SimpleNamespace

    from PIL import Image

    from sstvae.qrss import picture

    decoded = []

    class Codec:
        def decode(self, lat, wt):
            decoded.append(np.asarray(wt))
            return Image.new("RGB", (64, 48))

    L = live.LiveListener(tmp_path, live.LiveConfig(spec=TINY, render=True),
                          log=lambda m: None)
    L.codec = Codec()
    t = live.Tile(id="t", q=Q_TEST, slot_utc="x", frame="full", f_hz=1500.0)
    p = _Pass(1500.0, n=picture.SENT)
    p.z = np.ones(picture.SENT, dtype=np.float32)
    L._render(t, p)
    assert t.image == "tiles/t.png" and (tmp_path / t.image).is_file()
    assert live.GUESS_NOTE in t.note and len(decoded) == 1
    L._render(t, p)
    assert t.note.count(live.GUESS_NOTE) == 1            # marked once, however often redrawn
    p.header = SimpleNamespace(segment=1, mode=1, codec_id=0xD1D8, callsign="AG7EW",
                               grid="CN85", picture_id=7)
    L._render(t, p)
    assert t.note == "" and t.image_rev == 3
    # the guess put the pass in group 0; the header puts it in group 1
    assert not np.array_equal(decoded[0], decoded[2])


def test_a_headerless_pass_the_store_matched_is_drawn_as_its_picture(tmp_path):
    """Section 9: a headerless pass matched to a picture by the store (a
    combined header, or its latents) shows that picture, not the guess."""
    from types import SimpleNamespace

    from PIL import Image

    from sstvae.qrss import picture

    decoded = []

    class Codec:
        def decode(self, lat, wt):
            decoded.append(np.asarray(wt))
            return Image.new("RGB", (64, 48))

    p = _Pass(1500.0, n=picture.SENT)
    S = np.zeros((picture.N_GROUPS, picture.GROUP_LATENTS), dtype=np.float32)
    W = np.zeros_like(S)
    W[1] = 1.0                                     # the store placed it in group 1
    acc = SimpleNamespace(mode=1, codec_id=0xD1D8, S=S, W=W, members=[{}, {}],
                          uids=[p.uid, "other"])
    key = ("AG7EW", 7)
    store = SimpleNamespace(has_accumulator=lambda k: k == key, accumulator=lambda k: acc)
    L = live.LiveListener(tmp_path, live.LiveConfig(spec=TINY, render=True), store=store,
                          log=lambda m: None)
    L.codec = Codec()
    t = live.Tile(id="t", q=Q_TEST, slot_utc="x", frame="full", f_hz=1500.0,
                  note="stored (corr)")
    L._render(t, p, key=key)
    assert t.note == "stored (corr)" and t.callsign == "AG7EW" and t.mode == "B"
    assert t.passes == 2 and len(decoded) == 1


def test_a_restarted_listener_finishes_a_slot_that_ended_while_it_was_stopped(tmp_path):
    """A slot heard by an earlier listener that stopped before its frame
    ended is received from the passband store on the next start, once."""
    q = Q_TEST
    a = unit_rms_latents(TINY.n_data, 0)
    sym = frame.assemble(TINY, None, precoder.precode(a, q))
    sim = chm.simulate(sym, TINY, q, ChannelConfig(snr_db=-6.0, seed=2), carrier_hz=1500.0)
    t0 = slot_t0(q)
    n0 = t0 * FE_FS - int(round(sim.t0_index))
    pb = frontend.PassbandStore(tmp_path / "passband")
    pb.write(n0 / FE_FS, sim.fe, expire=False)        # the earlier listener heard it all
    now = t0 + live.frame_seconds(TINY) + frontend.PB_TAIL_S + 600.0
    cfg = live.LiveConfig(spec=TINY, render=False)
    L = live.LiveListener(tmp_path / "live", cfg, passband=pb, log=lambda m: None)
    assert L.recover(now) == [q]
    assert not L.recover(now)                          # queued once
    L.step(now)
    (tile,) = L.tiles()
    assert abs(tile.f_hz - 1500.0) < 0.2 and tile.status == "complete"
    L.write_state()
    assert q in json.loads((tmp_path / "live" / "state.json").read_text())["finished"]
    L2 = live.LiveListener(tmp_path / "live", cfg, passband=pb, log=lambda m: None)
    assert L2.recover(now) == []                        # finished already, by the first
    assert live.LiveListener(tmp_path / "live3", cfg, passband=pb,
                             log=lambda m: None).recover(now, hours=0.1) == []   # too long ago


def test_a_slot_the_store_holds_is_not_received_again(tmp_path):
    """An earlier listener that filed a slot's passes and stopped before it
    recorded the slot as finished: the restart does not receive it again,
    which would file a second copy of each pass under its picture."""
    from types import SimpleNamespace

    q = Q_TEST
    pb = frontend.PassbandStore(tmp_path / "passband")
    t0 = slot_t0(q)
    pb.write(t0 - 15.0, np.ones(int(80 * FE_FS), np.complex64), expire=False)
    now = t0 + live.frame_seconds(TINY) + frontend.PB_TAIL_S + 600.0
    cfg = live.LiveConfig(spec=TINY, render=False)
    held = SimpleNamespace(pass_uids=lambda: [f"{q}_1500000_0123abcd", "junk"])
    L = live.LiveListener(tmp_path / "a", cfg, store=held, passband=pb, log=lambda m: None)
    assert L.recover(now) == [] and q in L.finished
    other = SimpleNamespace(pass_uids=lambda: [f"{q - 4}_1500000_0123abcd"])
    L = live.LiveListener(tmp_path / "b", cfg, store=other, passband=pb, log=lambda m: None)
    assert L.recover(now) == [q]


def test_a_provisional_tile_takes_the_picture_a_later_header_finds(tmp_path, monkeypatch):
    """A pass filed provisional, then pulled into its picture when a later
    pass's header decodes: the older tile takes the picture's callsign, ID
    and mode (so the app hides it behind the newer one); others stay."""
    from types import SimpleNamespace

    import sstvae.qrss.associate as assoc

    key = ("AG7EW", 0x7B202AAD)
    hdr = SimpleNamespace(callsign="AG7EW", grid="CN85", picture_id=key[1], mode=1,
                          segment=1, codec_id=0xD1D8)
    filed = set()

    def associate(p, store):
        if p.header is None:
            return None, "prov"
        filed.add("old")                  # the header pulls the provisional pass in
        return key, "header"

    monkeypatch.setattr(assoc, "associate", associate)
    acc = SimpleNamespace(mode=1, member=lambda u: {"uid": u, "segment": 0, "method": "corr"})
    store = SimpleNamespace(find_member=lambda u: key if u == "old" and filed else None,
                            accumulator=lambda k: acc)
    L = live.LiveListener(tmp_path, live.LiveConfig(spec=TINY, render=False), store=store,
                          log=lambda m: None)
    L.now = slot_t0(Q_TEST + 4)
    first, second = live.SlotState(Q_TEST), live.SlotState(Q_TEST + 4)
    old, stray = _Pass(1500.0), _Pass(1700.0)
    old.uid, stray.uid = "old", "stray"
    L._finish_apply(first, [(None, [old, stray], 1.0)])
    L.done_tiles.extend(first.tiles.values())
    by_f = {round(t.f_hz): t for t in L.done_tiles}
    assert by_f[1500].note == "stored (provisional)" and by_f[1500].uid == "old"
    new = _Pass(1500.0, header=hdr)
    new.uid = "new"
    L._finish_apply(second, [(None, [new], 1.0)])
    t = by_f[1500]
    assert (t.callsign, t.picture_id, t.mode, t.segment) == ("AG7EW", "7b202aad", "B", 0)
    assert t.note == "stored (corr)" and t.passes == 1
    assert by_f[1700].picture_id is None and by_f[1700].note == "stored (provisional)"
    (n,) = second.tiles.values()
    assert n.note == "stored (header)" and n.uid == "new"


def test_a_header_decoded_once_stays_with_its_signal(tmp_path, monkeypatch):
    """A later live receive that misses the header keeps the one decoded
    earlier: the tile keeps its callsign and is not redrawn as the guess,
    and the end-of-slot pass goes to the store with it."""
    from types import SimpleNamespace

    hdr = SimpleNamespace(callsign="AG7EW", grid="CN85", picture_id=0x621AC873, mode=1,
                          segment=0, codec_id=0xD1D8)
    headers = iter([hdr, None, None])

    def receive_slot(prep, spec, live_only=False, estimator="joint", *, erase_s=(),
                     round_b=True, dets=None, max_candidates=None, budget_s=None):
        return [_Pass(1500.0, header=next(headers))]

    monkeypatch.setattr(RX, "detect", lambda prep, spec, live_only=False, **kw: [_Det(1500.0)])
    monkeypatch.setattr(RX, "receive_slot", receive_slot)
    stored = []

    import sstvae.qrss.associate as assoc
    monkeypatch.setattr(assoc, "associate", lambda p, store: stored.append(p.header) or
                        (("AG7EW", 1), "header"))
    store = SimpleNamespace(has_accumulator=lambda k: False)
    cfg = live.LiveConfig(spec=TINY, refresh_s=10.0, first_s=25.0, render=False)
    L = live.LiveListener(tmp_path, cfg, store=store, log=lambda m: None)
    t0 = slot_t0(Q_TEST)
    L.feed(np.zeros(76 * FS), t0 + 60.0)
    L.step(t0 + 30.0)
    L.step(t0 + 41.0)                                   # this receive missed the header
    (t,) = L.tiles()
    assert t.callsign == "AG7EW" and t.mode == "B" and live.GUESS_NOTE not in t.note
    L.step(t0 + live.frame_seconds(TINY) + frontend.PB_TAIL_S + 0.5)
    assert stored == [hdr]


def test_a_late_joined_pass_is_kept_for_the_store(tmp_path):
    """A pass whose header was never heard (the listener started mid-slot)
    is kept: when it is strong if its callsign windows read a call, and
    whatever a slot other than its two neighbours decoded at its frequency
    (the same picture sent again later), so the store can match it."""
    from types import SimpleNamespace

    L = live.LiveListener(tmp_path, live.LiveConfig(spec=TINY, render=False),
                          log=lambda m: None)
    s = live.SlotState(Q_TEST)
    later = live.SlotState(Q_TEST + 2)
    L.slots = {Q_TEST: s, Q_TEST + 2: later}
    L.now = slot_t0(Q_TEST + 2)
    L._tile_for(later, 1500.0).callsign = "AG7EW"    # the repeat, half an hour later
    assert L._plausible(s, _Pass(1503.0))
    strong = _Pass(1700.0, snr=2.0)
    assert not L._plausible(s, strong)
    strong.cw = SimpleNamespace(text="AG7EW")
    assert L._plausible(s, strong)

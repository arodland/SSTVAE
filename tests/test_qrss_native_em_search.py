"""Parity of native/core/qrss/em_search (sstvae_native.qrss.em_search) with
the template search and retroactive detection of sstvae/qrss/em.py:
`_candidates`, `_chunk_template`, `_search_chan`, `_search_input`,
`_from_passband`, `template_search`, `receive_template` and `retro_detect`.
tests/qrss_native/em_search.py substitutes them under --native-qrss=rx.

Each side gets its own store over the same files (the reference's
sstvae.qrss.store.Store, and the native one on a copy of its directory),
the same captures, and the same PassbandStore directory (written by the
reference's class; the C++ reads its files).

The contract (the bar), and in brackets what was measured (2026-10-14):

- **_search_chan**: the same (lag, m), lam within 1e-9 relative, on random
  captures and templates (complex, several chunk counts and lengths,
  drift bins, a K = 0 template) [bit for bit: the per-row FFTs are
  pocketfft's, as numpy's, and the products and |C|^2 numpy's fused and
  SIMD-hypot arithmetic];
- **_chunk_template**: starts exactly, energies and templates within 1e-12
  of the scale (ce_rx's bar: the mean template's np.exp is libm's here)
  [worst printed], the empty-span refusal's type and text;
- **_candidates, _search_input, _from_passband**: equal (hypotheses,
  Prepared arrays, the mix as a Fraction, the slot's capture) and the same
  refusals' texts; a non-store passes through as the very object;
- **template_search** on TINY (fast) and SHORT (slow): a weak pass, one at
  the Bonferroni gate, a different picture from the same station, noise, a
  stored pass (its own 250 Hz capture and mix), a 250 Hz capture with its
  mix, a candidate outside the capture's CH reach (and the same reached
  through `freqs`), a PassbandStore slot by `q`, `segments`, narrower
  options: the same detection or none, the same segment and cell
  (f_hz, ppm, tau0, drift) exactly, the same chunks and cells, Z within
  0.05 [worst |dZ| printed: 0 to 4e-15], lam within 1e-9 relative, the
  same refusals;
- **receive_template**: the pass within the receiver lane's bars (f_hz
  1e-6 Hz, z / w / ch to 1e-6 of the scale, timing, report), the same
  header, frame, q, mix and estimator [worst printed];
- **retro_detect** over a PassbandStore of seven slots -- a weak pass
  live acquisition misses (found, received, attached by corr), a slot
  already held, a stored pass within RETRO_SAME_HZ, coverage below 0.9,
  noise, an L pass stored in the slot (its band centre) and an L pass
  stored in another slot: the same slots found and passed over (every log
  line, its numbers within the bars), each find's pass within the bars,
  the same key and rule, the same members attached afterwards;
- the hooks: progress per slot of the list, a cancel before any slot or
  hypothesis, a Python `_seams` stub's answers taken as the reference
  takes them (rows file's seams).

The false-alarm recheck (C++ against Python Z on noise-only captures, the
statistic's distribution) is `test_false_alarm_noise` here (slow, a few
captures) and tools-free in native/tests/qrss/test_em_search.cpp (pins
from native/tests/qrss/gen_em_search_pins.py).
"""

import dataclasses
import functools
import math
import re
import shutil
from fractions import Fraction
from types import SimpleNamespace

import numpy as np
import pytest
from qrss_fakes import make_header, make_pass_result, synthetic_picture
from qrss_helpers import Q_TEST, unit_rms_latents
from qrss_native import _rx, require
from qrss_native import _store as H

from sstvae.qrss import channel as chm
from sstvae.qrss import em, frame, frontend, header, lframe, morse, track, tx
from sstvae.qrss import receiver as RX
from sstvae.qrss import store as S
from sstvae.qrss.channel import ChannelConfig
from sstvae.qrss.constants import CH_FS, T_SYM, WAVEFORM_CE, WAVEFORM_L

CALL = "K1ABC"
Z_BAR = 0.05
LAM_REL = 1e-9
PASS_BAR = 1e-6
SPECS = {"tiny": frame.TINY, "short": frame.SHORT}
# per frame: a weak pass's SNR (found), one at the gate, and the noise level's reference
WEAK = {"tiny": -30.0, "short": -30.0}
GATE = {"tiny": -32.0, "short": -36.0}
WORST: dict = {}


def _worst(key, err):
    WORST[key] = max(WORST.get(key, 0.0), float(err))


@pytest.fixture(scope="module", autouse=True)
def _print_worst():
    yield
    if WORST:
        print("\nworst:", ", ".join(f"{k} {v:.3g}" for k, v in sorted(WORST.items())))


class Clock:
    def __init__(self, t=1.76e9):
        self.t = float(t)

    def __call__(self):
        return self.t


@pytest.fixture(scope="module")
def ns(native):
    require(native, "store")
    return require(native, "em_search")


@pytest.fixture(scope="module")
def NStore(native, ns):
    return H.store_class(native.qrss)


# --- fixtures: pictures, captures, stores ----------------------------------------------------

@functools.lru_cache(maxsize=4)
def _picture(seed=0):
    sp = synthetic_picture(seed)
    return sp, make_header(sp, callsign=CALL)


def _key(h):
    return (h.callsign, int(h.picture_id))


@functools.lru_cache(maxsize=16)
def _sim(name, snr, q, seed, pic=0, preset="quiet"):
    spec = SPECS[name]
    sp, h = _picture(pic)
    sym = tx.slot_symbols(sp, q, spec, header=h if spec.n_hdr else None)
    return chm.simulate(sym, spec, q, ChannelConfig(snr_db=snr, preset=preset, seed=seed),
                        keying=morse.keying_units(CALL) if spec.n_win else None)


def _cap(name, snr, q, seed, pic=0):
    sim = _sim(name, snr, q, seed, pic)
    return frontend.Capture(sim.fe, q, sim.t0_index)


def _noise(name, q, seed):
    """A noise-only capture of the frame's length (the reference test's recipe)."""
    sim = _sim(name, WEAK[name], Q_TEST + 11, 7)
    rng = np.random.default_rng(10_000 + seed)
    n = len(sim.fe)
    sd = np.sqrt(sim.truth.noise_var_fe / 2)
    fe = (sd * (rng.standard_normal(n) + 1j * rng.standard_normal(n))).astype(np.complex64)
    return frontend.Capture(fe, q, sim.t0_index)


def _template_pass():
    sp, h = _picture(0)
    p = make_pass_result(tx.slot_air_latents(sp, 0)[:frame.FULL.n_data], 10 ** 0.5, hdr=h,
                         decoded=True, seed=1)
    return p, h


def _pair(root, NStore, extra=()):
    """(reference store, native store on a copy, header): the picture's
    accumulator (one +5 dB FULL pass, q = Q_TEST - 1) and `extra` passes
    stored unattached."""
    st = S.Store(root / "py", clock=Clock())
    p, h = _template_pass()
    st.add_pass(p)
    st.attach(_key(h), p.uid, h.segment, "test", mode=h.mode, codec_id=h.codec_id)
    for x in extra:
        st.add_pass(x)
    shutil.copytree(root / "py", root / "cpp")
    return st, NStore(root / "cpp", clock=Clock()), h


@pytest.fixture(scope="module")
def stores(tmp_path_factory, NStore):
    return _pair(tmp_path_factory.mktemp("em_search_stores"), NStore)


def _outcome(fn, *a, **k):
    try:
        return "ok", fn(*a, **k)
    except Exception as e:  # noqa: BLE001 - the reference's own type and text
        return type(e).__name__, str(e)


# --- comparisons ------------------------------------------------------------------------------

def _rel(a, b):
    a = np.asarray(a, dtype=np.complex128 if np.iscomplexobj(a) or np.iscomplexobj(b) else np.float64)
    b = np.asarray(b, dtype=a.dtype)
    assert a.shape == b.shape, (a.shape, b.shape)
    if a.size == 0:
        return 0.0
    scale = max(float(np.max(np.abs(b))), 1e-300)
    return float(np.max(np.abs(a - b))) / scale


def _num(x, y, tol, what, key=None):
    x, y = float(x), float(y)
    if math.isnan(y) or math.isnan(x):
        assert math.isnan(x) and math.isnan(y), (what, x, y)
        return
    err = abs(x - y) / max(1.0, abs(y))
    if key:
        _worst(key, err)
    assert err <= tol, (what, x, y)


def _det_close(c, p, what):
    """A C++ Detection (to_ref'd) against the reference's."""
    assert (c is None) == (p is None), what
    if p is None:
        return
    assert (c.method, c.lead_in_s) == (p.method, p.lead_in_s) == ("template", 0.0), what
    assert c.f_hz == p.f_hz and type(c.f_hz) is float, (what, c.f_hz, p.f_hz)
    assert np.array_equal(c.path.t_s, p.path.t_s) and np.array_equal(c.path.weight, p.path.weight), what
    e = float(np.max(np.abs(np.asarray(c.path.f_hz) - np.asarray(p.path.f_hz))))
    _worst("det path f_hz (Hz)", e)
    assert e <= 1e-9, what
    tc, tp = c.timing, p.timing
    assert (tc.tau0, tc.ppm, tc.gamma) == (tp.tau0, tp.ppm, tp.gamma), (what, tc, tp)
    assert np.array_equal(np.asarray(tc.cov), np.asarray(tp.cov)), what
    assert abs(tc.z - tp.z) <= Z_BAR and abs(c.z_ref - p.z_ref) <= Z_BAR, what


def _search_close(c, p, what):
    """template_search(..., return_stats=True) results: the C++'s against the reference's."""
    (dc, zc, ic), (dp, zp, ip) = c, p
    dc = _rx.to_ref(dc)
    assert isinstance(zc, float), what
    e = abs(zc - zp)
    _worst("template_search Z", e)
    assert e <= Z_BAR, (what, zc, zp)
    assert sorted(ic) == sorted(ip), (what, ic, ip)
    if ip:
        for k in ("chunks", "segment"):
            assert ic[k] == ip[k] and type(ic[k]) is int, (what, k, ic[k], ip[k])
        for k in ("cells", "f_hz", "ppm", "tau0", "drift"):     # the cell, exactly
            assert ic[k] == ip[k] and type(ic[k]) is float, (what, k, ic[k], ip[k])
        _num(ic["lam"], ip["lam"], LAM_REL, (what, "lam"), "template_search lam (rel)")
    _det_close(dc, dp, what)


def _search_both(ns, st, nst, key, cap, spec, **kw):
    p = _outcome(em.template_search, st, key, cap, spec, return_stats=True, **kw)
    c = _outcome(ns.template_search, nst, key, cap, spec, return_stats=True, **kw)
    assert c[0] == p[0], (c, p)
    if p[0] != "ok":
        assert c[1] == p[1], (c, p)
        return None
    _search_close(c[1], p[1], kw)
    return p[1]


def _pass_close(g, r, what):
    """A C++ PassResult (to_ref'd) against the reference's, at the receiver lane's bars."""
    assert (g is None) == (r is None), what
    if r is None:
        return
    assert type(g) is type(r), what
    assert (g.q, g.frame, g.waveform, g.estimator, g.em_round, g.ch_fs) == \
        (r.q, r.frame, r.waveform, r.estimator, r.em_round, r.ch_fs), what
    assert g.uid.rsplit("_", 1)[0] == r.uid.rsplit("_", 1)[0], what     # uid: its q and f fields only
    assert g.header == r.header and g.cw is None and r.cw is None, what
    e = abs(g.f_hz - r.f_hz)
    _worst("pass f_hz (Hz)", e)
    assert e <= 1e-6, what
    assert abs(g.timing.tau0 - r.timing.tau0) <= 1e-6 * CH_FS, what
    for f in ("ppm", "gamma", "z"):
        _num(getattr(g.timing, f), getattr(r.timing, f), PASS_BAR, (what, f), f"pass timing.{f}")
    for f in ("offset_hz", "drift_hz_per_min", "wander_hz_rms", "doppler_hz", "ppm", "snr2500_db", "z_ref",
              "kappa"):
        _num(getattr(g.report, f), getattr(r.report, f), PASS_BAR, (what, f), f"pass report.{f}")
    assert (g.report.slip_free, g.report.suspect) == (r.report.slip_free, r.report.suspect), what
    for f in ("z", "w", "hdr_llr", "psi", "ch"):
        a, b = getattr(g, f), getattr(r, f)
        assert np.asarray(a).dtype == np.asarray(b).dtype, (what, f)
        e = _rel(a, b)
        _worst(f"pass {f}", e)
        assert e <= PASS_BAR, (what, f, e)
    assert g.ch_t0_index == r.ch_t0_index and type(g.ch_t0_index) is float, what
    assert g.f_mix_hz == r.f_mix_hz and isinstance(g.f_mix_hz, Fraction), what


_NUM = re.compile(r"-?\d+(?:\.\d+)?")


def _lines_close(c, p):
    """Log lines: the same words, every number within 0.1 (Z's one decimal) or
    exactly (q, the segment, the key)."""
    assert len(c) == len(p), (c, p)
    for a, b in zip(c, p):
        assert _NUM.sub("#", a) == _NUM.sub("#", b), (a, b)
        for x, y in zip(_NUM.findall(a), _NUM.findall(b)):
            if "." in y:
                assert abs(float(x) - float(y)) <= 0.11, (a, b)
            else:
                assert x == y, (a, b)


# --- _search_chan -------------------------------------------------------------------------------

def _chan_case(seed, K, chunk, lag, m, drift):
    rng = np.random.default_rng(seed)
    n = max(K * chunk + 2 * lag + 50, 10)
    z = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    starts = np.sort(rng.integers(-lag - 20, n + 20, K)).astype(np.int64)
    E = (rng.standard_normal((K, chunk)) + 1j * rng.standard_normal((K, chunk))) * 10.0 ** rng.uniform(-2, 2, (K, 1))
    en = np.sum(np.abs(E) ** 2, axis=1)
    db = rng.integers(-3, 4, K).astype(np.int64) if drift else np.zeros(K, dtype=np.int64)
    return z, float(rng.uniform(0.1, 10.0)), starts, E, en, lag, m, db


@pytest.mark.parametrize("case", [
    (1, 1, 500, 625, 9, False), (2, 7, 500, 625, 9, False), (3, 26, 500, 625, 9, True),
    (4, 5, 137, 40, 3, True), (5, 3, 64, 0, 0, False), (6, 4, 500, 900, 2, False),
    (7, 2, 1000, 625, 12, True), (8, 0, 500, 625, 9, False),
])
def test_search_chan(ns, case):
    args = _chan_case(*case)
    p = _outcome(em._search_chan, *args)
    c = _outcome(ns._search_chan, *args)
    assert c[0] == p[0], (c, p)
    if p[0] != "ok":
        assert c[1] == p[1]
        return
    (lc, gc, mc), (lp, gp, mp) = c[1], p[1]
    assert (gc, mc) == (gp, mp) and type(gc) is int and type(mc) is int, (c, p)
    e = abs(lc - lp) / max(abs(lp), 1e-300)
    _worst("_search_chan lam (rel)", e)
    assert e <= LAM_REL and type(lc) is float, (c, p)


def test_search_chan_capture_shorter_than_its_chunks(ns):
    rng = np.random.default_rng(11)
    z = rng.standard_normal(300) + 1j * rng.standard_normal(300)
    E = rng.standard_normal((3, 200)) + 1j * rng.standard_normal((3, 200))
    args = (z, 1.0, np.array([-500, 150, 900]), E, np.sum(np.abs(E) ** 2, axis=1), 30, 2,
            np.zeros(3, dtype=np.int64))
    assert ns._search_chan(*args) == em._search_chan(*args)


# --- _chunk_template ----------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["tiny", "short"])
def test_chunk_template(ns, stores, name):
    st, _, h = stores
    spec = SPECS[name]
    acc = st.accumulator(_key(h))
    idx = S.canonical_index(0, spec.n_data)
    prior = em.prior_from(acc.S[0, idx], acc.W[0, idx], Q_TEST + 11)
    cls = [track.make_classes(spec, None, prior)]
    if spec.has_header:
        cls.append(track.make_classes(spec, header.encode(h), prior))
    keyings = [(None, False)] + ([(morse.keying_units(CALL), False), (morse.keying_units(CALL), True)]
                                 if spec.n_win else [])
    n_full = int(spec.keyed_end_pos * T_SYM * CH_FS * 1.01) + 4000
    worst_e = worst_en = 0.0
    for cl in cls:
        for keying, ook in keyings:
            for t0, ppm, n_ch, chunk in ((2000.25, 0.0, n_full, 500), (1234.5, 37.5, n_full, 500),
                                         (-40.25, -149.95313964386128, n_full, 137),
                                         (2000.0, 0.0, n_full // 3, 500), (2000.0, 0.0, 10, 500),
                                         (2000.0, 0.0, n_full, 1)):
                args = (spec, cl.mu, cl.nu, keying, ook, t0, ppm, n_ch, chunk)
                p = _outcome(em._chunk_template, *args)
                c = _outcome(ns._chunk_template, *args)
                assert c[0] == p[0], (c[0], p[0], args[5:])
                if p[0] != "ok":
                    assert c[1] == p[1]
                    continue
                (sc, Ec, enc), (sp_, Ep, enp) = c[1], p[1]
                assert np.array_equal(sc, sp_) and sc.dtype == sp_.dtype, args[5:]
                assert Ec.shape == Ep.shape and Ec.dtype == Ep.dtype == np.complex128, args[5:]
                assert enc.dtype == enp.dtype, args[5:]
                if Ep.size:
                    worst_e = max(worst_e, float(np.max(np.abs(Ec - Ep))) / max(float(np.max(np.abs(Ep))), 1e-300))
                    worst_en = max(worst_en, float(np.max(np.abs(enc - enp) / np.abs(enp))))
    _worst("_chunk_template E (of the scale)", worst_e)
    _worst("_chunk_template en (rel)", worst_en)
    assert worst_e <= 1e-12 and worst_en <= 1e-12


# --- _candidates, _search_input, _from_passband -------------------------------------------------

def test_candidates(ns, tmp_path, NStore):
    """Members' frequencies, drifts and clocks, deduplicated; an L member and a
    member whose pass file is gone give none; extra frequencies after."""
    sp, h = _picture(0)
    a = tx.slot_air_latents(sp, 0)[:frame.TINY.n_data]
    passes = []
    for k, (f, drift, ppm) in enumerate(((1500.0, 0.0, 0.0), (1500.1, 0.02, 0.5), (1500.3, 0.0, 0.0),
                                         (1500.1, 0.2, 0.0), (1500.1, 0.0, 3.0), (1498.0, -0.4, -12.5))):
        p = make_pass_result(a, 2.0, seed=40 + k, f_hz=f)
        p = dataclasses.replace(p, report=dataclasses.replace(p.report, drift_hz_per_min=drift),
                                timing=dataclasses.replace(p.timing, ppm=ppm))
        passes.append(p)
    pl = make_pass_result(unit_rms_latents(lframe.L_TINY.n_data, 5), 1.0, seed=5, q=Q_TEST + 30,
                          f_hz=1509.5, waveform=WAVEFORM_L)
    gone = make_pass_result(a, 2.0, seed=60, f_hz=1600.0)
    trees = []
    for sub in ("py", "cpp"):
        st = (S.Store if sub == "py" else NStore)(tmp_path / sub, clock=Clock())
        for p in passes + [pl, gone]:
            st.add_pass(p)
            st.attach(_key(h), p.uid, 0, "test", mode=h.mode, codec_id=h.codec_id)
        (tmp_path / sub / "passes" / f"{gone.uid}.npz").unlink()
        trees.append(st)
    st, nst = trees
    for extra in ((), (1500.05, 1600.0, 1600.2, 1500.0), [np.float64(1700.0), 1700]):
        got = ns._candidates(nst, nst.accumulator(_key(h)), extra)
        ref = em._candidates(st, st.accumulator(_key(h)), extra)
        assert got == ref and all(type(x) is float for c in got for x in c), (extra, got, ref)


def test_search_input(ns, stores):
    cap = _cap("tiny", WEAK["tiny"], Q_TEST + 11, 7)
    prep = RX.prepare(cap)
    det = em.template_search(stores[0], _key(stores[2]), cap, frame.TINY)
    stored = em.receive_template(stores[0], _key(stores[2]), cap, frame.TINY, det=det)
    ch = track.make_chan(prep.fe, prep.keep, prep.t0_index, prep.fs, 1501.5, -40.0, 100.0)
    ch_cap = frontend.Capture(ch.ch, cap.q, ch.t0_index, fs=CH_FS)
    bare = dataclasses.replace(stored, ch=None)
    empty = dataclasses.replace(stored, ch=np.zeros(0, np.complex64))
    cases = [(cap, None), (cap, 1500), (prep, None), (ch_cap, ch.f_mix), (ch_cap, None), (ch_cap, 1501.5),
             (ch_cap, "1501.5"), (stored, None), (stored, 1234), (bare, None), (empty, None)]
    for k, (c, fm) in enumerate(cases):
        p = _outcome(em._search_input, c, fm)
        n = _outcome(ns._search_input, c, fm)
        assert n[0] == p[0], (k, n, p)
        if p[0] != "ok":
            assert n[1] == p[1], k
            continue
        (gp, gm), (rp, rm) = _rx.to_ref(n[1]), p[1]
        if c is prep:
            assert gp is c                     # a Prepared is handed back as itself
        assert type(gm) is type(rm) and gm == rm, (k, gm, rm)
        assert (gp.q, gp.t0_index, gp.fs) == (rp.q, rp.t0_index, rp.fs), k
        assert (gp.keep is None) == (rp.keep is None), k
        if rp.keep is not None:
            assert np.array_equal(gp.keep, rp.keep), k
        e = _rel(gp.fe, rp.fe)
        _worst("_search_input fe", e)
        assert e <= PASS_BAR and np.asarray(gp.fe).dtype == np.asarray(rp.fe).dtype, k


def test_from_passband(ns, tmp_path):
    spec = frame.TINY
    q = Q_TEST + 11
    pb = frontend.PassbandStore(tmp_path / "pb")
    pb.write_capture(_cap("tiny", WEAK["tiny"], q, 7), expire=False)
    c = ns._from_passband(pb, q, spec)
    r = em._from_passband(pb, q, spec)
    assert type(c) is type(r) is frontend.Capture
    assert (c.q, c.t0_index, c.fs) == (r.q, r.t0_index, r.fs) and type(c.t0_index) is type(r.t0_index)
    assert c.fe.dtype == r.fe.dtype and c.fe.tobytes() == r.fe.tobytes()
    for qq in (None, q + 400):                              # the refusals' texts
        a, b = _outcome(ns._from_passband, pb, qq, spec), _outcome(em._from_passband, pb, qq, spec)
        assert a == b and a[0] == "ValueError", (a, b)
    c = ns._from_passband(pb, float(q), spec)              # int(q), as the reference
    assert c.q == q and type(c.q) is int and c.fe.tobytes() == r.fe.tobytes()
    thing = object()
    assert ns._from_passband(thing, q, spec) is thing and ns._from_passband("not a store", None, spec) == "not a store"


# --- template_search ----------------------------------------------------------------------------

def _search_cases(name):
    """label -> (cap factory, kwargs)."""
    spec = SPECS[name]
    q = Q_TEST + 11

    def weak():
        return _cap(name, WEAK[name], q, 7)

    def ch(f_mix):
        prep = RX.prepare(weak())
        c = track.make_chan(prep.fe, prep.keep, prep.t0_index, prep.fs, f_mix, -40.0,
                            spec.keyed_end_pos * T_SYM + 60.0)
        return frontend.Capture(c.ch, q, c.t0_index, fs=CH_FS), c.f_mix

    return {
        "weak": (weak, {}),
        "gate": (lambda: _cap(name, GATE[name], q, 7), {}),
        "other-picture": (lambda: _cap(name, WEAK[name] + 6.0, q, 8, pic=1), {}),
        "noise": (lambda: _noise(name, q, 3), {}),
        "ch-mix": (lambda: ch(1501.5)[0], {"f_mix_hz": ch(1501.5)[1]}),
        "ch-reach": (lambda: ch(1500.0 + 150.0)[0], {"f_mix_hz": ch(1650.0)[1]}),
        "ch-reach-freqs": (lambda: ch(1500.0 + 150.0)[0], {"f_mix_hz": ch(1650.0)[1], "freqs": (1640.0,)}),
        "freqs": (weak, {"freqs": (1500.6, 1503.0)}),
        "segments": (weak, {"segments": [0, 1]}),
        "options": (weak, {"tau_s": 0.5, "df_hz": 0.1, "ppm_steps": 1, "z_accept": 12.0}),
    }


FAST_LABELS = ["weak", "gate", "other-picture", "noise", "ch-mix", "ch-reach", "ch-reach-freqs", "freqs",
               "segments", "options"]


@pytest.mark.parametrize("label", FAST_LABELS)
def test_template_search_tiny(ns, stores, label):
    st, nst, h = stores
    make, kw = _search_cases("tiny")[label]
    r = _search_both(ns, st, nst, _key(h), make(), frame.TINY, **kw)
    if label in ("weak", "ch-mix", "freqs"):
        assert r is not None and r[0] is not None          # the fixtures mean what they say
    if label in ("noise", "ch-reach"):
        assert r[0] is None
    if label == "ch-reach":
        assert r == (None, 0.0, {})                         # no candidate inside the capture


@pytest.mark.slow
@pytest.mark.parametrize("label", ["weak", "gate", "other-picture", "noise", "ch-mix", "ch-reach-freqs"])
def test_template_search_short(ns, stores, label):
    st, nst, h = stores
    make, kw = _search_cases("short")[label]
    _search_both(ns, st, nst, _key(h), make(), frame.SHORT, **kw)


def test_template_search_stored_pass_and_passband(ns, stores, tmp_path):
    """A stored pass (its 250 Hz capture and mix) and a PassbandStore slot by q."""
    st, nst, h = stores
    key = _key(h)
    spec = frame.TINY
    q = Q_TEST + 11
    cap = _cap("tiny", WEAK["tiny"], q, 7)
    det = em.template_search(st, key, cap, spec)
    stored = em.receive_template(st, key, cap, spec, det=det)
    r = _search_both(ns, st, nst, key, stored, spec)
    assert r[0] is not None
    pb = frontend.PassbandStore(tmp_path / "pb")
    pb.write_capture(cap, expire=False)
    r = _search_both(ns, st, nst, key, pb, spec, q=q)
    assert r[0] is not None
    _search_both(ns, st, nst, key, pb, spec, q=float(q))
    for kw in ({}, {"q": q + 400}):                         # refusals: the slot's q, a slot not stored
        _search_both(ns, st, nst, key, pb, spec, **kw)


def test_template_search_refusals(ns, stores):
    st, nst, h = stores
    key = _key(h)
    cap = _cap("tiny", WEAK["tiny"], Q_TEST + 11, 7)
    prep = RX.prepare(cap)
    ch = track.make_chan(prep.fe, prep.keep, prep.t0_index, prep.fs, 1500.0, -40.0, 100.0)
    ch_cap = frontend.Capture(ch.ch, cap.q, ch.t0_index, fs=CH_FS)
    _search_both(ns, st, nst, key, ch_cap, frame.TINY)                       # a 250 Hz capture, no mix
    for spec in (lframe.L_TINY, lframe.L_SHORT):                             # CE-only, before the capture
        a = _outcome(em.template_search, st, key, None, spec)
        b = _outcome(ns.template_search, nst, key, None, spec)
        assert a == b and a[0] == "ValueError"
    _search_both(ns, st, nst, ("NOBODY", 1), cap, frame.TINY)                # no accumulator: nothing
    _search_both(ns, st, nst, key, cap, frame.TINY, segments=[])
    _search_both(ns, st, nst, key, cap, frame.TINY, segments=[2])            # a segment without data
    c = _outcome(ns.template_search, nst, key, cap, frame.TINY, tau_s=-1.0)  # deviation: refused
    assert c[0] == "ValueError" and "tau_s" in c[1]


def test_template_search_cancel(ns, native, stores):
    st, nst, h = stores
    flag = native.qrss.em.CancelFlag()
    flag.set()
    with pytest.raises(native.qrss.em.Cancelled):
        ns.template_search(nst, _key(h), _cap("tiny", WEAK["tiny"], Q_TEST + 11, 7), frame.TINY, cancel=flag)


# --- receive_template ---------------------------------------------------------------------------

def _receive_both(ns, st, nst, key, cap, spec, **kw):
    p = _outcome(em.receive_template, st, key, cap, spec, **kw)
    c = _outcome(ns.receive_template, nst, key, cap, spec, **kw)
    assert c[0] == p[0], (c, p)
    if p[0] != "ok":
        assert c[1] == p[1], (c, p)
        return None
    _pass_close(_rx.to_ref(c[1]), p[1], kw)
    return p[1]


def test_receive_template_tiny(ns, stores):
    st, nst, h = stores
    key = _key(h)
    spec = frame.TINY
    cap = _cap("tiny", WEAK["tiny"], Q_TEST + 11, 7)
    det = em.template_search(st, key, cap, spec)
    assert det is not None
    p = _receive_both(ns, st, nst, key, cap, spec, det=det)
    assert p is not None
    _receive_both(ns, st, nst, key, cap, spec)                               # searched here
    _receive_both(ns, st, nst, key, cap, spec, det=det, estimator="plain", segment=0)
    _receive_both(ns, st, nst, key, p, spec, det=det)                         # a stored pass's capture
    assert _receive_both(ns, st, nst, key, _noise("tiny", Q_TEST + 11, 3), spec) is None
    _receive_both(ns, st, nst, key, cap, spec, bogus=1)                      # template_search's TypeError
    make, kw = _search_cases("tiny")["ch-mix"]
    _receive_both(ns, st, nst, key, make(), spec, **kw)


@pytest.mark.slow
def test_receive_template_short(ns, stores):
    st, nst, h = stores
    key = _key(h)
    cap = _cap("short", WEAK["short"], Q_TEST + 11, 7)
    det = em.template_search(st, key, cap, frame.SHORT)
    p = _receive_both(ns, st, nst, key, cap, frame.SHORT, det=det)
    assert p is not None  # header parity is _pass_close's; at -30 dB it may not decode


# --- retro_detect -------------------------------------------------------------------------------

def _retro_world(name, root, NStore):
    """A PassbandStore of seven slots and the two stores (see the module docstring)."""
    spec = SPECS[name]
    snr = WEAK[name]
    qA, qH, qS, qC, qN, qL, qO = (Q_TEST + 1, Q_TEST - 1, Q_TEST + 2, Q_TEST + 3, Q_TEST + 5, Q_TEST + 6,
                                  Q_TEST + 7)
    sp, h = _picture(0)
    a = tx.slot_air_latents(sp, 0)[:spec.n_data]
    l_spec = lframe.get_any("l-" + name)
    extra = [
        make_pass_result(a, 2.0, seed=70, q=qS, f_hz=1500.4),                          # stored, same slot
        make_pass_result(unit_rms_latents(l_spec.n_data, 5), 1.0, seed=71, q=qL,
                         f_hz=1500.0 - RX.L_EXCLUDE_OFFSET_HZ, waveform=WAVEFORM_L),   # L, its centre
        make_pass_result(unit_rms_latents(l_spec.n_data, 6), 1.0, seed=72, q=qO + 1,
                         f_hz=1500.0 - RX.L_EXCLUDE_OFFSET_HZ, waveform=WAVEFORM_L),   # L, another slot
    ]
    st, nst, h = _pair(root, NStore, extra)
    pb = frontend.PassbandStore(root / "pb")
    for k, q in enumerate((qA, qH, qS, qL, qO)):
        pb.write_capture(_cap(name, snr, q, 20 + k), expire=False)
    c = _cap(name, snr, qC, 30)
    pb.write_capture(frontend.Capture(c.fe[:int(0.6 * len(c.fe))], qC, c.t0_index), expire=False)
    pb.write_capture(_noise(name, qN, 31), expire=False)
    return st, nst, h, pb


def _retro_both(ns, st, nst, h, pb, spec, **kw):
    key = _key(h)
    lp, lc = [], []
    fp = em.retro_detect(st, key, pb, spec, log=lp.append, **kw)
    fc = ns.retro_detect(nst, key, pb, spec, log=lc.append, **kw)
    _lines_close(lc, lp)
    assert len(fc) == len(fp), (lc, lp)
    for (pc, kc, rc), (pp, kp, rp) in zip(fc, fp):
        _pass_close(_rx.to_ref(pc), pp, "retro find")
        assert kc == kp and rc == rp, (kc, kp, rc, rp)
    mc = sorted((nst.pass_info(m["uid"])["q"], m["segment"]) for m in nst.accumulator(key).members)
    mp = sorted((st.pass_info(m["uid"])["q"], m["segment"]) for m in st.accumulator(key).members)
    assert mc == mp
    return fp, lp


def test_retro_detect_tiny(ns, tmp_path, NStore):
    st, nst, h, pb = _retro_world("tiny", tmp_path, NStore)
    found, log = _retro_both(ns, st, nst, h, pb, frame.TINY)
    print("\n".join(log))
    assert [p.q for p, _, _ in found] == [Q_TEST + 1, Q_TEST + 7]
    assert sum("already stored" in s for s in log) == 2 and sum("nothing" in s for s in log) == 1
    found, log = _retro_both(ns, st, nst, h, pb, frame.TINY)                 # a second run: nothing new
    assert found == []
    _retro_both(ns, st, nst, h, pb, frame.TINY, slots=[Q_TEST + 5, float(Q_TEST + 2)], ppm_steps=1)


@pytest.mark.slow
def test_retro_detect_short(ns, tmp_path, NStore):
    st, nst, h, pb = _retro_world("short", tmp_path, NStore)
    found, log = _retro_both(ns, st, nst, h, pb, frame.SHORT)
    print("\n".join(log))
    assert [p.q for p, _, _ in found] == [Q_TEST + 1, Q_TEST + 7]


def test_retro_detect_hooks_and_seams(ns, native, tmp_path, NStore):
    """progress per slot of the list and cancel before a slot; a Python pb and
    `_seams` taken as the reference takes them (a stubbed search)."""
    st, nst, h = _pair(tmp_path, NStore)
    key = _key(h)
    seen = []
    pb = SimpleNamespace(slots=lambda dur: [Q_TEST - 1, Q_TEST + 3, Q_TEST + 4],
                         capture=lambda q, dur: (None, 0.0) if q == Q_TEST + 4 else (("cap", q), 1.0))

    def template_search(store, k, cap, spec, return_stats=False, **search):
        seen.append((cap, return_stats, search))
        return None, 2.5, {}

    lp = []
    import sstvae.qrss.em as ref_em
    mp = pytest.MonkeyPatch()
    mp.setattr(ref_em, "template_search", template_search)
    try:
        assert em.retro_detect(st, key, pb, frame.TINY, log=lp.append, tau_s=1.0) == []
    finally:
        mp.undo()
    ref_seen, seen[:] = list(seen), []
    lc, prog = [], []
    out = ns.retro_detect(nst, key, pb, frame.TINY, log=lc.append, tau_s=1.0,
                          progress=lambda d, t, w: prog.append((d, t, w)),
                          _seams={"template_search": template_search})
    assert out == [] and lc == lp == [f"slot q={Q_TEST + 3}: nothing (Z 2.5)"]
    assert seen == ref_seen == [(("cap", Q_TEST + 3), True, {"tau_s": 1.0})]
    assert prog == [(k + 1, 3, f"slot q={q}") for k, q in enumerate((Q_TEST - 1, Q_TEST + 3, Q_TEST + 4))]
    flag = native.qrss.em.CancelFlag()
    flag.set()
    with pytest.raises(native.qrss.em.Cancelled):
        ns.retro_detect(nst, key, pb, frame.TINY, cancel=flag, _seams={"template_search": template_search})
    # a search seam's detection without info["segment"]: the reference's KeyError, at the receive
    det = em.template_search(st, key, _cap("tiny", WEAK["tiny"], Q_TEST + 11, 7), frame.TINY)
    pb1 = SimpleNamespace(capture=lambda q, dur: (("cap", q), 1.0))
    stub = {"template_search": lambda *a, **k: (det, 9.0, {})}
    with pytest.raises(KeyError, match="segment"):
        ns.retro_detect(nst, key, pb1, frame.TINY, slots=[Q_TEST + 3], _seams=stub)
    with pytest.raises(TypeError):                         # the C++ search does not take it
        ns.retro_detect(nst, key, pb1, frame.TINY, slots=[Q_TEST + 3], bogus=1)
    for spec in (lframe.L_TINY,):
        a = _outcome(em.retro_detect, st, key, None, spec)
        b = _outcome(ns.retro_detect, nst, key, None, spec)
        assert a == b and a[0] == "ValueError"


# --- false alarms -------------------------------------------------------------------------------

@pytest.mark.slow
def test_false_alarm_noise(ns, stores):
    """Noise-only captures against the picture's accumulator: the C++'s Z within
    0.05 of the reference's on each, and no detection (the full recheck, 40 of
    each frame, is the lane's report; this is a few)."""
    st, nst, h = stores
    key = _key(h)
    zs = {}
    for name, n in (("tiny", 8), ("short", 4)):
        spec = SPECS[name]
        for seed in range(n):
            cap = _noise(name, Q_TEST + 11, 500 + seed)
            r = _search_both(ns, st, nst, key, cap, spec)
            assert r[0] is None
            zs.setdefault(name, []).append(r[1])
    print({k: (max(v), float(np.mean(v))) for k, v in zs.items()})

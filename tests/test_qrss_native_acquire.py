"""Parity of native/core/qrss/acquire (sstvae_native.qrss.acquire) with
sstvae/qrss/acquire.py's arbitration and acquire(), held side by side in
one process:

- the arbitration (`drop_symbol_rate_aliases`, `drop_band_owned`,
  `merge_hits`) keeps the very same hit objects in the same order, on
  crafted sets (every rule's boundary, ties in the sort keys) and on
  random ones; `_hint_items` and `live_end_s` equal, errors included;
- `Acquisition.detections` (one frame and a tuple) on the reference's own
  Acquisitions: the same candidates in the same order, method, waveform,
  path and timing exact, an A3 line's polyfit frequency to 1e-12;
- acquire() on the C++ (A1, A2 of both kinds, hints, arbitration,
  lead-in, A3), single and joint, live_only or not, on CE and L SHORT at
  -15 dB, a raw capture holding both, noise, and strong signals whose
  symbol-rate alias and band-owned neighbour pass A2 and must be dropped:
  A1's lines, the pilots and the hints exact, type for type; the preamble
  hits to wave 5's A2 bar (f_hz 1e-7 Hz, tau0_s 1e-6 s), the same lead-in;
  A3 exact but its Gumbel p (4e-16);
- the seams: with the reference's stages handed to the C++ (only this
  module's rows applied), the C++ acquisition is the reference's to the
  bit; a replaced noise_psd / shape_frames is called;
- the reference's errors, type and text;
- the whole callers with spies on the native entry points (installed
  before the rows, so a row that kept the reference shows a zero):
  acquire.acquire single and joint with every acquire row, the reference
  acquire with only the arbitration rows, and (slow) receiver.detect and
  receive_slot on CE SHORT and on a mixed CE + L SHORT capture.

tests/qrss_native/acquire.py substitutes these under --native-qrss=rx.
"""

import dataclasses

import numpy as np
import pytest
import test_qrss_acquire as TA
import test_qrss_lacquire as TL
from qrss_native import _owner, _rx, require
from test_qrss_native_acquire_a12 import _cands_same, _hit_close, _same

from sstvae.qrss import acquire as A
from sstvae.qrss import frame, frontend, lframe
from sstvae.qrss.constants import CH_FS, FE_FS, L_T_SYM, T_SYM, WAVEFORM_CE, WAVEFORM_L
from sstvae.qrss.types import CarrierCandidate, Detection, FreqPath

CE, L = WAVEFORM_CE, WAVEFORM_L
JOINT = (frame.SHORT, lframe.L_SHORT)


@pytest.fixture(scope="module")
def N(native):
    return require(native, "acquire")


def _hit(f, tau, lam, p, waveform):
    return A.PreambleHit(f_hz=f, drift_hz_per_min=0.0, tau0_s=tau, phase=0.0, lam=lam,
                         cell_lam=lam, p_value=p, passed=p < A.A2_PFA, waveform=waveform)


def _outcome(fn, *args, **kw):
    try:
        out = fn(*args, **kw)
    except (ValueError, TypeError, IndexError, KeyError) as e:
        return type(e), str(e)
    return "ok", out


def _nan_eq(a, b):
    return a == b or (a != a and b != b)


# --- comparisons --------------------------------------------------------------------------------

def _hit_same(r, g, what):
    """A preamble hit to the bit, type for type."""
    assert type(g) is type(r) is A.PreambleHit, what
    for f in dataclasses.fields(r):
        a, b = getattr(r, f.name), getattr(g, f.name)
        if isinstance(a, np.ndarray):
            _same(b, a, f"{what}.{f.name}")
        else:
            assert type(b) is type(a) and _nan_eq(a, b), (what, f.name, a, b)


def _tbd_same(r, g, what, p_rtol):
    assert type(g) is type(r) is A.TbdHit, what
    for f in dataclasses.fields(r):
        a, b = getattr(r, f.name), getattr(g, f.name)
        if f.name == "p_value":
            assert type(b) is type(a) is float and abs(b - a) <= p_rtol * abs(a), (what, a, b)
        elif dataclasses.is_dataclass(a):
            assert type(b) is type(a), (what, f.name)
            for ff in dataclasses.fields(a):
                _same(getattr(b, ff.name), getattr(a, ff.name), f"{what}.{f.name}.{ff.name}")
        else:
            assert type(b) is type(a) and _nan_eq(a, b), (what, f.name, a, b)


def _acq_check(r, g, what, exact=False):
    """exact: every field to the bit (the stages were the reference's);
    else A2's hits to wave 5's bar and A3's Gumbel p to 4e-16."""
    assert type(g) is A.Acquisition, what
    _cands_same(r.lines, g.lines, f"{what} lines")
    _cands_same(r.pilots, g.pilots, f"{what} pilots")
    _cands_same(r.hinted, g.hinted, f"{what} hinted")
    assert len(r.preamble) == len(g.preamble), (what, r.preamble, g.preamble)
    for i, (a, b) in enumerate(zip(r.preamble, g.preamble)):
        if exact:
            _hit_same(a, b, f"{what} preamble[{i}]")
        else:
            _hit_close(a, b, f"{what} preamble[{i}]")
            assert b.lead_in_s == a.lead_in_s, (what, a.lead_in_s, b.lead_in_s)
    assert len(r.tbd) == len(g.tbd), (what, len(r.tbd), len(g.tbd))
    for i, (a, b) in enumerate(zip(r.tbd, g.tbd)):
        _tbd_same(a, b, f"{what} tbd[{i}]", 0.0 if exact else 4e-16)


def _det_check(r, g, what, hit_tol=None, raw=False):
    """Two candidate lists. hit_tol None: a preamble detection exact, else
    its frequency (and path) within hit_tol Hz and its timing within
    1e-6 s; an A3 line's frequency (polyfit) to 1e-12 either way. raw: g
    is the binding's own output, whose hint frequencies are floats."""
    assert len(r) == len(g), (what, [(d.method, d.waveform, d.f_hz) for d in r],
                              [(d.method, d.waveform, d.f_hz) for d in g])
    for i, (a, b) in enumerate(zip(r, g)):
        w = f"{what}[{i}]"
        assert type(b) is type(a) is Detection, w
        assert (b.method, b.waveform) == (a.method, a.waveform), (w, a.method, b.method, a.waveform, b.waveform)
        assert type(b.waveform) is type(a.waveform), w
        assert type(b.lead_in_s) is type(a.lead_in_s) and b.lead_in_s == a.lead_in_s, (w, a.lead_in_s, b.lead_in_s)
        assert _nan_eq(a.z_ref, b.z_ref) and _nan_eq(a.a3_z, b.a3_z), w
        if raw and a.method == "hint":
            assert type(b.f_hz) is float and b.f_hz == a.f_hz, (w, a.f_hz, b.f_hz)
        else:
            assert type(b.f_hz) is type(a.f_hz), (w, type(a.f_hz), type(b.f_hz))
        tol = 0.0
        if a.method == "tbd":
            tol = 1e-12 * abs(a.f_hz)
        elif a.method == "preamble" and hit_tol is not None:
            tol = hit_tol
        assert abs(b.f_hz - a.f_hz) <= tol, (w, a.f_hz, b.f_hz)
        for k in ("t_s", "f_hz", "weight"):
            x, y = getattr(a.path, k), getattr(b.path, k)
            if a.method == "preamble" and hit_tol is not None and k == "f_hz":
                assert x.dtype == y.dtype and x.shape == y.shape, w
                assert np.max(np.abs(y - x)) <= 4 * hit_tol, (w, x, y)     # + drift t / 60
            else:
                _same(y, x, f"{w}.path.{k}")
        assert (a.timing is None) == (b.timing is None), w
        if a.timing is not None:
            ta, tb = a.timing, b.timing
            if a.method == "preamble" and hit_tol is not None:
                assert abs(tb.tau0 - ta.tau0) <= 1e-6 * CH_FS, (w, ta.tau0, tb.tau0)
                assert abs(tb.z - ta.z) <= 1e-9 * abs(ta.z), w
                np.testing.assert_allclose(tb.cov, ta.cov, rtol=1e-5, atol=0.0)
                assert (tb.ppm, tb.gamma) == (ta.ppm, ta.gamma), w
            else:
                for f in ("tau0", "ppm", "gamma", "z"):
                    x, y = getattr(ta, f), getattr(tb, f)
                    assert type(y) is type(x) and _nan_eq(x, y), (w, f, x, y)
                _same(tb.cov, ta.cov, f"{w}.cov")


# --- small functions ----------------------------------------------------------------------------

def test_live_end_s(N):
    assert type(N.live_end_s()) is float and N.live_end_s() == A.live_end_s()


def test_hint_items(N):
    specs = A.frame_map(JOINT)
    for hints in ((), (1500,), (1500, (1400.0, 1), (1300, 5)), [np.float64(1200.5), (np.float32(7.25), 0)],
                  ([1210.0, 1.9], (1.0, True), ("1300.5", "1")), (np.int64(1200), (1.5, np.int8(1)))):
        for sp in (specs, A.frame_map((frame.TINY,)), A.frame_map((lframe.L_TINY,)), {L: lframe.L_TINY, CE: frame.TINY}):
            r, g = A._hint_items(hints, sp), N._hint_items(hints, sp)
            assert len(r) == len(g), (hints, r, g)
            for (rf, rw), (gf, gw) in zip(r, g):
                assert type(gf) is float and gf == rf and gw == rw and all(type(x) is int for x in gw), (r, g)
    for bad in (((1, 2, 3),), ((1,),), ("abc",), (("abc", 1),), ((1.0, None),), (None,), ((1.0, "x"),),
                (1500.0, [1.0, 2.0, 3.0]), ({1: 2},)):
        r, g = _outcome(A._hint_items, bad, specs), _outcome(N._hint_items, bad, specs)
        assert r == g, (bad, r, g)
        assert r[0] != "ok" or bad == ({1: 2},), (bad, r)


def test_arbitration_crafted(N):
    """Each rule at its boundaries and on ties: the same objects kept, in
    the same order."""
    k1 = 1.0 / T_SYM
    sets = {
        "alias": [_hit(1000.0, 0.30, 500.0, 1e-200, CE), _hit(1000.0 + k1, 0.30, 60.0, 1e-9, CE),
                  _hit(1000.0 - 2 * k1 + 0.49, 0.31, 100.0, 1e-9, CE), _hit(1000.0 + k1 + 0.5, 0.30, 10.0, 1e-9, CE),
                  _hit(1000.0 - k1, 0.30 + T_SYM / 2, 10.0, 1e-9, CE), _hit(1000.0 - k1, 0.30, 500.0 / 3, 1e-9, CE),
                  _hit(1000.0 + 3 * k1, 0.30, 1.0, 1e-9, CE), _hit(1000.0 + k1, 0.30, 60.0, 1e-9, CE)],
        "band": [_hit(1500.0, 0.70, 200.0, 1e-90, L), _hit(1500.0 - 3.1, 0.70, 40.0, 1e-12, L),
                 _hit(1500.0 - 22.1, 0.70 + 3 * L_T_SYM, 30.0, 1e-10, L),
                 _hit(1500.0 - 22.1, 0.70 + 2.99 * L_T_SYM, 30.0, 1e-10, L),
                 _hit(1500.0 - 3.09, 0.70, 30.0, 1e-10, L), _hit(1500.0 - 22.11, 0.70, 30.0, 1e-10, L),
                 _hit(1500.0 - 10.0, 0.70, 30.0, 1e-10, CE), _hit(1500.0 + 10.0, 0.70, 30.0, 1e-10, L),
                 _hit(1500.0 - 16.5, 0.70, 400.0, 1e-10, L), _hit(1500.0 - 10.0, 0.70, 30.0, 1e-10, L)],
        "merge": [_hit(1500.0, 0.30, 50.0, 1e-30, CE), _hit(1500.0, 0.31, 80.0, 1e-30, L),
                  _hit(1505.0, 0.30, 90.0, 1e-40, L), _hit(1504.99, 0.30 + L_T_SYM / 2, 90.0, 1e-40, CE),
                  _hit(1495.01, 0.30 - L_T_SYM / 2 + 1e-9, 90.0, 1e-40, CE), _hit(1300.0, 0.0, 80.0, 1e-30, CE),
                  _hit(1300.0, 0.0, 80.0, 1e-30, L), _hit(1302.0, 0.01, 80.0, 0.0, CE)],
    }
    for what, hits in sets.items():
        for fn, kw in (("drop_symbol_rate_aliases", {}), ("drop_symbol_rate_aliases", {"t_sym": L_T_SYM}),
                       ("drop_band_owned", {}), ("merge_hits", {})):
            for order in (hits, hits[::-1], hits[1:] + hits[:1]):
                r = getattr(A, fn)(order, **kw)
                g = getattr(N, fn)(order, **kw)
                assert len(r) == len(g) and all(a is b for a, b in zip(r, g)), (what, fn, kw, r, g)
                assert type(g) is list
    # every rule dropped something somewhere above
    assert len(A.drop_symbol_rate_aliases(sets["alias"])) < len(sets["alias"])
    assert len(A.drop_band_owned(sets["band"])) < len(sets["band"])
    assert len(A.merge_hits(sets["merge"])) < len(sets["merge"])
    for fn in ("drop_symbol_rate_aliases", "drop_band_owned", "merge_hits"):
        assert getattr(N, fn)([]) == [] and getattr(N, fn)(()) == []
        h = sets["merge"][0]
        assert getattr(N, fn)(iter([h]))[0] is h


def test_arbitration_random(N):
    """Random hit sets crowded about the aliases, the L band and the merge
    window, with tied Lambdas and p-values: the same objects kept."""
    rng = np.random.default_rng(41)
    dropped = dict.fromkeys(("a", "al", "b", "m"), 0)
    for trial in range(300):
        base = float(rng.uniform(900.0, 1200.0))
        hits = []
        for _ in range(int(rng.integers(0, 14))):
            w = int(rng.choice([CE, L]))
            kind = int(rng.integers(6))
            if kind == 0:
                f = base + rng.uniform(-6.0, 6.0)
            elif kind == 1:
                f = base + rng.choice([-2, -1, 1, 2]) / T_SYM + rng.uniform(-0.7, 0.7)
            elif kind == 2:
                f = base + rng.choice([-2, -1, 1, 2]) / L_T_SYM + rng.uniform(-0.7, 0.7)
            elif kind == 3:
                f = base - rng.uniform(2.0, 24.0)
            elif kind == 4:
                f = base + float(rng.choice([-22.1, -3.1, 5.0, -5.0]))
            else:
                f = rng.uniform(800.0, 1300.0)
            tau = float(rng.choice([0.0, 0.3])) + float(rng.choice([0.0, rng.uniform(-0.2, 0.2)]))
            lam = float(rng.choice([100.0, 30.0, rng.uniform(5.0, 600.0)]))
            p = float(rng.choice([1e-10, 0.0, 10.0 ** -rng.uniform(5.0, 250.0)]))
            hits.append(_hit(float(f), tau, lam, p, w))
        for key, fn, kw in (("a", "drop_symbol_rate_aliases", {}), ("al", "drop_symbol_rate_aliases",
                                                                     {"t_sym": L_T_SYM}),
                            ("b", "drop_band_owned", {}), ("m", "merge_hits", {})):
            r = getattr(A, fn)(hits, **kw)
            g = getattr(N, fn)(hits, **kw)
            assert len(r) == len(g) and all(a is b for a, b in zip(r, g)), (trial, fn, kw)
            dropped[key] += len(hits) - len(r)
        # the joint path's composition
        per = []
        for w, t in ((CE, T_SYM), (L, L_T_SYM)):
            per += A.drop_symbol_rate_aliases([h for h in hits if h.waveform == w], t)
        r = A.merge_hits(A.drop_band_owned(per))
        g = N.merge_hits(N.drop_band_owned(per))
        assert len(r) == len(g) and all(a is b for a, b in zip(r, g)), trial
    assert min(dropped.values()) >= 20, dropped


# --- detections ---------------------------------------------------------------------------------

def _crafted_acq():
    """test_detections_order's Acquisition."""
    def tbd(f, w, l_f=None):
        t = np.array([0.0, 100.0])
        path = FreqPath(t_s=t, f_hz=np.full(2, f), weight=np.ones(2))
        lp = None if l_f is None else FreqPath(t_s=t, f_hz=np.full(2, l_f), weight=np.ones(2))
        return A.TbdHit(path=path, z=50.0, p_value=1e-9, n_frames=31, centre_hz=f, waveform=w, l_path=lp)
    return A.Acquisition(
        lines=[], preamble=[_hit(1000.0, 0.0, 50.0, 1e-12, CE), _hit(1100.0, 0.0, 80.0, 1e-20, L)],
        tbd=[tbd(1200.0, CE), tbd(1300.0, None, l_f=1306.0), tbd(1400.0, L), tbd(1385.0, None, l_f=1404.0),
             tbd(1700.0, 5), tbd(1705.0, L)],
        hinted=[CarrierCandidate(1500.0, 0.0, 0.0, waveform=L), CarrierCandidate(1600.0, 1.5, 3.0),
                CarrierCandidate(np.float64(1650.25), -0.5, 3.0, waveform=CE),
                CarrierCandidate(1800.0, 0.0, 0.0, waveform=7)])


SPEC_SETS = (frame.TINY, lframe.L_TINY, frame.SHORT, JOINT, (frame.TINY,), (lframe.L_TINY,),
             (lframe.L_SHORT, frame.FULL), [frame.TINY, lframe.L_TINY])


def test_detections_crafted(N):
    """test_detections_order's Acquisition (with a hit whose waveform no
    frame names and hints of every kind), every frame and tuple."""
    acq = _crafted_acq()
    for spec in SPEC_SETS:
        r = acq.detections(spec)
        g = _rx.to_ref(N._detections(acq, spec))
        _det_check(r, g, f"crafted {getattr(spec, 'name', spec)}", raw=True)
        if isinstance(spec, (tuple, list)):
            g2 = _rx.to_ref(N._joint_detections(acq, A.frame_map(spec)))
            _det_check(acq._joint_detections(A.frame_map(spec)), g2, "joint", raw=True)
    order = [(d.method, d.waveform, round(d.f_hz)) for d in _rx.to_ref(N._detections(acq, (frame.TINY,
                                                                                         lframe.L_TINY)))]
    assert order[:8] == [("preamble", L, 1100), ("preamble", CE, 1000), ("hint", L, 1500),
                         ("hint", CE, 1600), ("hint", L, 1600), ("hint", CE, 1650), ("tbd", CE, 1200),
                         ("tbd", CE, 1300)], order
    for bad in ((frame.TINY, frame.SHORT), (), [lframe.L_TINY, lframe.L_SHORT]):
        r, g = _outcome(acq.detections, bad), _outcome(N._detections, acq, bad)
        assert r == g and r[0] is ValueError, (bad, r, g)


# --- captures and the reference's acquisitions ------------------------------------------------

@pytest.fixture(scope="module")
def caps():
    """Prepared FE (fe, t0_index): CE and L SHORT at -15 dB, noise, CE and
    L SHORT at 0 dB (whose aliases pass A2); and a raw capture holding an
    L SHORT pass at -12 dB with a CE one of the same power 60 Hz above."""
    from qrss_fakes import l_scenario

    from sstvae.qrss.channel import Neighbour

    x, t0 = TA._capture(1650.0, -15.0, shift_s=0.7, seed=3)
    xl, t0l = TL._capture(TL._l_clean("l-short"), 1203.4, -15.0, shift_s=-0.4, seed=5)
    noise = TA._noise_fe(len(x), np.random.default_rng(16))
    xs, t0s = TA._capture(1650.0, 0.0, shift_s=0.2, seed=9)
    xls, t0ls = TL._capture(TL._l_clean("l-short"), 1203.4, 0.0, shift_s=0.1, seed=10)
    _, sim = l_scenario("l-short", -12.0, 1, cfg=(("neighbours", (Neighbour(60.0, 0.0, seed=7,
                                                                             waveform=CE),)),))
    return {"ce": (x, t0), "l": (xl, t0l), "noise": (noise, t0), "ce0": (xs, t0s), "l0": (xls, t0ls),
            "mixed": (sim.fe, sim.t0_index)}


# name: (capture, spec, kwargs) -- the acquisitions compared
RUNS = {
    # one frame: CE's acquisition, A3 over the frame, hints A2 does and does not pass
    "ce single a3": ("ce", frame.SHORT, {"preprocessed": True, "hints": (1650.5, 1203.4, 2100.0)}),
    # joint, A3 with shape_specs, lead-ins at L's symbol, hints of each kind and a waveform
    # no frame names
    "l joint a3": ("l", JOINT, {"preprocessed": True, "hints": ((1210.0, CE), 1203.4, 1190.0, (1800.0, 7))}),
    # a raw capture (the C++ blanker and AGC) with a CE and an L signal: one hit each
    "mixed raw joint": ("mixed", JOINT, {"live_only": True, "hints": ((1560.0, CE),)}),
    # a 0 dB CE signal hinted at f -+ 1/T: the +1/T alias passes A2 and is dropped
    "ce0 single aliases": ("ce0", frame.SHORT, {"preprocessed": True, "live_only": True,
                                                "hints": (1650.0 - 1 / T_SYM, 1650.0 + 1 / T_SYM)}),
    # a 0 dB L signal hinted beside its -1/T_L alias: the alias passes A2-L (a hint offset) and
    # is the stronger signal's band
    "l0 band": ("l0", (lframe.L_SHORT,), {"preprocessed": True, "live_only": True, "hints": ((1188.0, L),)}),
    # slow
    "ce joint": ("ce", JOINT, {"preprocessed": True, "live_only": True,
                               "hints": (1650.0, (1203.4, L), 2100.0, (1800.0, 7), (1700, CE))}),
    "l single": ("l", frame.SHORT, {"preprocessed": True, "live_only": True}),
    "noise joint": ("noise", JOINT, {"preprocessed": True, "live_only": True, "hints": (1500.0, (1300.0, L))}),
    "l alone": ("l", (lframe.L_SHORT,), {"preprocessed": True, "live_only": True, "hints": (1203.0,)}),
    "ce0 aliases": ("ce0", JOINT, {"preprocessed": True, "live_only": True,
                                   "hints": tuple(1650.0 + k / T_SYM for k in (-2, -1, 1, 2))}),
    "l0 band joint": ("l0", JOINT, {"preprocessed": True, "live_only": True, "hints": (1188.0, (1220.5, L))}),
    "noise single a3": ("noise", frame.SHORT, {"preprocessed": True}),
}
FAST = ("ce single a3", "l joint a3", "mixed raw joint", "ce0 single aliases", "l0 band")
SLOW = tuple(n for n in RUNS if n not in FAST)
_ARB = ("drop_symbol_rate_aliases", "drop_band_owned", "merge_hits")


def _cap(caps, name):
    fe, t0 = caps[name]
    return frontend.Capture(fe, 0, t0)


class _Ref:
    """The reference's acquisitions, each made once when first asked for,
    and what its arbitration dropped in each (to show a rule was
    exercised)."""

    def __init__(self, caps):
        self.caps = caps
        self.acq = {}
        self.dropped = {}

    def __call__(self, name):
        if name not in self.acq:
            cap, spec, kw = RUNS[name]
            counts = dict.fromkeys(_ARB, 0)
            real = {n: getattr(A, n) for n in _ARB}

            def counted(fn, key):
                def f(hits, *a):
                    hits = list(hits)
                    kept = fn(hits, *a)
                    counts[key] += len(hits) - len(kept)
                    return kept
                return f
            for n, fn in real.items():
                setattr(A, n, counted(fn, n))
            try:
                self.acq[name] = A.acquire(_cap(self.caps, cap), spec, **kw)
            finally:
                for n, fn in real.items():
                    setattr(A, n, fn)
            self.dropped[name] = counts
        return self.acq[name]


@pytest.fixture(scope="module")
def ref(caps):
    return _Ref(caps)


# --- detections on real acquisitions --------------------------------------------------------------

def test_detections_on_reference_acquisitions(N, ref):
    n = {"preamble": 0, "hint": 0, "tbd": 0}
    for name in FAST:
        acq = ref(name)
        for spec in SPEC_SETS:
            r = acq.detections(spec)
            _det_check(r, _rx.to_ref(N._detections(acq, spec)), f"{name} {spec}", raw=True)
            for d in r:
                n[d.method] += 1
    assert min(n.values()) >= 10, n


# --- acquire() ----------------------------------------------------------------------------------

def _on_the_cpp(N, caps, ref, names, joint_too=()):
    n_hits = n_tbd = n_hinted = 0
    for name in names:
        cap, spec, kw = RUNS[name]
        r = ref(name)
        g = _rx.to_ref(N.acquire(_cap(caps, cap), spec, **kw))
        _acq_check(r, g, name)
        n_hits += len(r.preamble)
        n_tbd += len(r.tbd)
        n_hinted += len(r.hinted)
        if name in joint_too:
            g2 = _rx.to_ref(N._acquire_joint(_cap(caps, cap), A.frame_map(spec), live_only=kw.get(
                "live_only", False), preprocessed=kw.get("preprocessed", False), hints=kw.get("hints", ())))
            _acq_check(r, g2, f"{name} _acquire_joint")
    return n_hits, n_tbd, n_hinted


def test_acquire_on_the_cpp(N, caps, ref):
    """The C++ acquire() with its own stages against the reference."""
    n_hits, n_tbd, n_hinted = _on_the_cpp(N, caps, ref, FAST)    # _acquire_joint: the whole caller
    assert n_hits >= 6 and n_tbd >= 2 and n_hinted >= 4, (n_hits, n_tbd, n_hinted)
    d = ref.dropped
    # the strong signals' aliases passed A2 and were dropped by their rules
    assert d["ce0 single aliases"]["drop_symbol_rate_aliases"] >= 1, d
    assert d["l0 band"]["drop_band_owned"] >= 1, d
    # the mixed capture: both signals found, each in its waveform
    got = sorted((h.waveform, round(h.f_hz)) for h in ref("mixed raw joint").preamble)
    assert got == [(CE, 1560), (L, 1500)], got


@pytest.mark.slow
def test_acquire_on_the_cpp_more(N, caps, ref):
    """The other acquisitions: noise, an L signal under CE's acquisition,
    L alone, a 0 dB CE signal's +-1/T and +-2/T aliases."""
    n_hits, _t, n_hinted = _on_the_cpp(N, caps, ref, SLOW, joint_too=("ce joint", "noise joint"))
    assert n_hits >= 5 and n_hinted >= 8, (n_hits, n_hinted)
    assert ref.dropped["ce0 aliases"]["drop_symbol_rate_aliases"] >= 1, ref.dropped
    assert ref.dropped["l0 band joint"]["drop_band_owned"] >= 1, ref.dropped


def test_acquire_errors(N, caps):
    fe, t0 = caps["ce"]
    zeros = frontend.Capture(np.zeros(40 * FE_FS, np.complex64), 0, TA.PRE_S * FE_FS)
    cap = frontend.Capture(fe, 0, t0)
    cases = [
        (zeros, (frame.TINY, frame.SHORT), {}),
        (zeros, (), {}),
        (zeros, [lframe.L_TINY, lframe.L_SHORT], {}),
        (cap, frame.SHORT, {"preprocessed": True, "live_only": True, "hints": (1650.0, [1203.4, L])}),
        (zeros, frame.TINY, {"preprocessed": True, "live_only": True, "hints": ((1650.0, CE),)}),
        (zeros, frame.TINY, {"preprocessed": True, "live_only": True, "hints": ([1650.0, CE],)}),
        (zeros, (frame.TINY,), {"preprocessed": True, "live_only": True, "hints": ((1, 2, 3),)}),
        (zeros, (frame.TINY,), {"preprocessed": True, "live_only": True, "hints": (("x", 0),)}),
        (zeros, (frame.TINY,), {"preprocessed": True, "live_only": True, "hints": ((1500.0, None),)}),
    ]
    for c, spec, kw in cases:
        r = _outcome(A.acquire, c, spec, **kw)
        g = _outcome(lambda *a, **k: _rx.to_ref(N.acquire(*a, **k)), c, spec, **kw)
        assert r[0] != "ok" and r == g, (spec, kw, r, g)


# --- the seams ----------------------------------------------------------------------------------

def _set_rows(monkeypatch, *tables):
    for table in tables:
        for (target, attr), fn in table.items():
            monkeypatch.setattr(_owner(target), attr, fn)


def _every_row(native, monkeypatch):
    from qrss_native import acquire as rows_acq
    from qrss_native import acquire_a12, acquire_stats, acquire_tbd
    from qrss_native import frontend as rows_fe

    q = native.qrss
    _set_rows(monkeypatch, rows_fe.rows(q), rows_acq.rows(q), acquire_a12.rows(q), acquire_stats.rows(q),
              acquire_tbd.rows(q))


def test_acquire_with_the_reference_stages(N, native, caps, ref, monkeypatch):
    """Only this module's rows applied: the C++ acquisition calls every
    stage back in Python (the reference's), so its result is the
    reference's to the bit -- the order of the calls, the arguments, the
    arbitration and the assembly all the C++'s."""
    from qrss_native import acquire as rows

    names = ("ce single a3", "l joint a3", "mixed raw joint")
    want = {name: ref(name) for name in names}
    calls = []
    real = native.qrss.acquire.acquire

    def spy(*a, **kw):
        calls.append(sorted(kw["_seams"]))
        return real(*a, **kw)
    monkeypatch.setattr(native.qrss.acquire, "acquire", spy)
    _set_rows(monkeypatch, rows.rows(native.qrss))
    for name in names:
        cap, spec, kw = RUNS[name]
        g = A.acquire(_cap(caps, cap), spec, **kw)
        _acq_check(want[name], g, f"{name} (reference stages)", exact=True)
    assert len(calls) == len(names)
    assert {"carrier_line_stats", "carrier_lines", "pilot_lines", "hint_line", "a1_threshold", "_a2_input",
            "preamble_search", "lead_in_span", "track_before_detect", "noise_psd", "shape_frames", "blank",
            "normalise"} == set(calls[0]), calls[0]


def test_replaced_noise_psd_and_shape_frames(N, native, caps, ref, monkeypatch):
    """Every row applied, then frontend.noise_psd and acquire.shape_frames
    replaced by wrappers over the native ones: the C++ lead-in and A3 call
    them back, and the result is the all-native one."""
    name = "l joint a3"                          # A3 with shape_specs, a lead-in
    r = ref(name)
    cap, spec, kw = RUNS[name]
    _every_row(native, monkeypatch)
    want = A.acquire(_cap(caps, cap), spec, **kw)
    seen = {"noise_psd": 0, "shape_frames": 0}

    def counted(fn, key):
        def f(*a, **k):
            seen[key] += 1
            return fn(*a, **k)
        return f
    monkeypatch.setattr(frontend, "noise_psd", counted(frontend.noise_psd, "noise_psd"))
    monkeypatch.setattr(A, "shape_frames", counted(A.shape_frames, "shape_frames"))
    got = A.acquire(_cap(caps, cap), spec, **kw)
    _acq_check(want, got, "replaced", exact=True)
    assert seen["noise_psd"] >= len(want.preamble) >= 1 and seen["shape_frames"] == 1, seen
    _acq_check(r, got, "replaced vs reference")


# --- the whole callers, with spies -------------------------------------------------------------

_ENTRY = ("acquire", "_acquire_joint", "_hint_items", "drop_symbol_rate_aliases", "drop_band_owned",
          "merge_hits", "live_end_s", "_detections", "_joint_detections")


def _spy(native, monkeypatch, names=_ENTRY):
    """Counters on native.qrss.acquire's entry points, installed before the
    rows are built (some rows bind the function itself); `seams` records
    each acquire's _seams argument."""
    mod = native.qrss.acquire
    counts = dict.fromkeys(names, 0)
    seams = []
    for name in names:
        def counted(*a, _fn=getattr(mod, name), _name=name, **kw):
            counts[_name] += 1
            if "_seams" in kw:
                seams.append(kw["_seams"])
            return _fn(*a, **kw)
        monkeypatch.setattr(mod, name, counted)
    return counts, seams


def test_acquire_whole_caller(N, native, caps, ref, monkeypatch):
    """acquire.acquire (single and joint), _acquire_joint, detections and
    _joint_detections, live_end_s, with every acquire row (and frontend's)
    assigned: the reference's results, every row reached, every stage the
    C++'s (no seams)."""
    names = ("ce single a3", "mixed raw joint", "l0 band")
    specs = {name: (RUNS[name][1], frame.TINY, (lframe.L_TINY,)) for name in names}
    # the reference's candidates, before the rows replace Acquisition.detections
    want = {name: [ref(name).detections(sp) for sp in specs[name]] for name in names}
    want_joint = ref("l0 band")._joint_detections(A.frame_map(RUNS["l0 band"][1]))
    counts, seams = _spy(native, monkeypatch)
    _every_row(native, monkeypatch)
    for name in names:
        cap, spec, kw = RUNS[name]
        g = A.acquire(_cap(caps, cap), spec, **kw)
        r = ref(name)
        _acq_check(r, g, f"{name} (whole caller)")
        for c in g.lines + g.pilots:
            assert type(c.f_hz) is np.float64, (name, c)
        for sp, w in zip(specs[name], want[name]):
            _det_check(w, g.detections(sp), f"{name} dets", hit_tol=1e-7)
        if name == "l0 band":
            m = A.frame_map(spec)
            g2 = A._acquire_joint(_cap(caps, cap), m, live_only=kw["live_only"], preprocessed=True,
                                  hints=kw["hints"])
            _acq_check(r, g2, f"{name} _acquire_joint (whole caller)")
            _det_check(want_joint, g2._joint_detections(m), f"{name} joint dets", hit_tol=1e-7)
    assert A.live_end_s() == A.A2_TAU_S + A.N_PRE * T_SYM + A._A2_MARGIN_S
    assert seams and all(s is None for s in seams), seams
    # every entry point reached; the arbitration inside acquire() is the C++'s, so
    # the Python-facing ones are reached only by their own callers (next test)
    assert counts["acquire"] == 3 and counts["_acquire_joint"] == 1, counts
    assert counts["_detections"] == 9 and counts["_joint_detections"] == 1 and counts["live_end_s"] == 1, counts
    assert counts["_hint_items"] == counts["drop_band_owned"] == 0, counts


def test_reference_acquire_with_the_arbitration_rows(N, native, caps, ref, monkeypatch):
    """The reference's acquire() with only the arbitration rows assigned
    (its stages Python): every one called, and the acquisition the
    reference's to the bit (lead_in_s set on the hits they returned)."""
    from qrss_native import acquire as rows_acq

    names = ("mixed raw joint", "l0 band")
    want = {name: ref(name) for name in names}
    keep = ("_hint_items", "drop_symbol_rate_aliases", "drop_band_owned", "merge_hits", "live_end_s")
    counts, _seams = _spy(native, monkeypatch, keep)
    _set_rows(monkeypatch, {k: v for k, v in rows_acq.rows(native.qrss).items() if k[1] in keep})
    for name in names:
        cap, spec, kw = RUNS[name]
        g = A.acquire(_cap(caps, cap), spec, **kw)
        _acq_check(want[name], g, f"{name} (arbitration rows)", exact=True)
    assert A.live_end_s() == A.A2_TAU_S + A.N_PRE * T_SYM + A._A2_MARGIN_S
    assert counts == {"_hint_items": 2, "drop_symbol_rate_aliases": 3, "drop_band_owned": 2, "merge_hits": 2,
                      "live_end_s": 1}, counts


# --- receiver, on the C++ (slow) ----------------------------------------------------------------

def _passes_close(rps, gps):
    assert len(rps) == len(gps) >= 1, (rps, gps)
    for rp, gp in zip(rps, gps):
        assert (rp.frame, rp.waveform, rp.estimator, rp.em_round, rp.f_mix_hz) == \
            (gp.frame, gp.waveform, gp.estimator, gp.em_round, gp.f_mix_hz)
        assert rp.header == gp.header and rp.cw == gp.cw
        assert abs(gp.f_hz - rp.f_hz) <= 1e-7 and abs(gp.timing.tau0 - rp.timing.tau0) <= 1e-6 * CH_FS
        for f in ("z", "w", "hdr_llr", "psi"):
            a, b = getattr(rp, f), getattr(gp, f)
            assert a.dtype == b.dtype and a.shape == b.shape, f
            if a.size:
                err = float(np.max(np.abs(b.astype(np.float64) - a)))
                assert err <= 1e-3 * max(float(np.max(np.abs(a))), 1.0), (f, err)


def _receiver_case(native, monkeypatch, which):
    import test_qrss_receiver as TR
    from qrss_fakes import l_scenario
    from qrss_helpers import Q_TEST

    from sstvae.qrss import receiver as RX
    from sstvae.qrss.channel import Neighbour

    if which == "ce":
        _a, sim = TR.scenario("short", -12.0, 1)
        spec = frame.SHORT
    else:
        _a, sim = l_scenario("l-short", -12.0, 1, cfg=(("neighbours", (Neighbour(60.0, 0.0, seed=7,
                                                                                 waveform=CE),)),))
        spec = JOINT

    def run():
        cap = frontend.Capture(sim.fe, Q_TEST, sim.t0_index)
        dets = RX.detect(RX.prepare(cap), spec)
        return dets, RX.receive_slot(cap, spec)

    rdets, rps = run()
    counts, seams = _spy(native, monkeypatch)
    _every_row(native, monkeypatch)
    gdets, gps = run()
    assert counts["acquire"] == 2 and counts["_detections"] == 2, counts
    assert all(s is None for s in seams), seams
    _det_check(rdets, gdets, f"{which} detect", hit_tol=1e-7)
    _passes_close(rps, gps)
    return rps


@pytest.mark.slow
def test_receiver_ce_short_on_the_cpp(native, N, monkeypatch):
    """receiver.detect and receive_slot on a CE SHORT pass at -12 dB with
    every acquire row assigned: the same detections and the same pass."""
    ps = _receiver_case(native, monkeypatch, "ce")
    assert [p.waveform for p in ps] == [CE]


@pytest.mark.slow
def test_receiver_mixed_joint_on_the_cpp(native, N, monkeypatch):
    """receiver.detect and receive_slot with (SHORT, L_SHORT) on an L SHORT
    pass with a CE one 60 Hz above: the same detections and both passes."""
    ps = _receiver_case(native, monkeypatch, "mixed")
    assert sorted((p.waveform, round(p.f_hz)) for p in ps) == [(CE, 1560), (L, 1500)]

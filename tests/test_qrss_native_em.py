"""Parity of native/core/qrss/em (sstvae_native.qrss.em) with the shared
half of sstvae/qrss/em.py -- the references, what every member is known to
have sent, a stored pass's frame, the paths and the template search's tail
statistics -- and of the tail functions under them
(native/core/math/special: ndtri, norm_isf, gamma_logsf, CPython's
lgamma). tests/qrss_native/em.py substitutes them under --native-qrss=rx.

The contract (the bar), and in brackets what was measured (2026-10-14):

- **the references**: lmmse_latents and prior_from bit for bit (values,
  dtypes, shapes, numpy scalars for 0-d, the broadcast refusal's text) on
  random, degenerate and non-finite inputs; em_prior bit for bit on stores
  written by either implementation, read by either, and its refusals
  [all equal]; acc_mean_w_db equal, or within an ulp where numpy's SIMD
  log10 and the C library's part, the C++'s always 10 math.log10 of the mean
  [worst printed];
- **what every member sent**: known_header's and known_keying's decisions
  (HeaderFields, keying units and OOK) identical on fixtures -- a member's
  decoded header, the soft header from summed LLRs, a key mismatch, a
  segment beyond the mode, mixed CE / L members, OOK against FSK votes,
  refused callsigns -- on both implementations' stores [all equal];
- **a stored pass's frame**: pass_spec (the module's own preset object),
  is_l and _ce_only's text on every frame name, header versions 1 and 2,
  real and duck-typed passes [all equal];
- **the paths**: _psi_frame and _flat bit for bit; _path_of's times bit for
  bit and its frequencies to 1e-12 of their scale (the tracker's path is
  evaluated by the C++ spline, the trackbase lane's bar) [worst printed];
- **the tail statistics**: _gauss_z and template_z within 1e-12 relative
  (gamma.logsf takes numpy's SIMD log / log1p in the reference and the C
  library's here) over Z 0 to 40, n_cells 1 to 1e9, n_chunks 1 to 900,
  and, beyond log_p -700 (the Mills ratio), within 0.01 of the exact Z as
  the reference's docstring promises; their refusals' texts; ndtri,
  norm_isf and Python's lgamma bit for bit with scipy and math; gamma.logsf
  within 1e-12 relative [worst printed];
- **the hooks**: log lines and progress in order, a cancel set during
  step k stopping the call at step k + 1 with Cancelled, a hook's
  exception passing through, refusals of a hook of the wrong type;
- **the whole caller**: em.em_refine (the reference's, a later lane's
  port) over this lane's rows with the store's: the same history, log and
  files.

The same functions on pinned inputs, without Python, are
native/tests/qrss/test_em.cpp's and test_special.cpp's (pins from
native/tests/qrss/gen_em_pins.py).
"""

import copy
import dataclasses
import functools
import math
import threading
from fractions import Fraction
from types import SimpleNamespace

import numpy as np
import pytest
from qrss_fakes import make_header, make_pass_result, synthetic_picture
from qrss_helpers import Q_TEST
from qrss_native import _rx, require
from qrss_native import _store as H

import qrss_native
import test_qrss_multipass as tm
from sstvae.qrss import em, frame, header, lframe, morse, picture
from sstvae.qrss import store as S
from sstvae.qrss.constants import WAVEFORM_CE, WAVEFORM_L
from sstvae.qrss.types import CwIdResult

WORST: dict = {}


def _worst(key, err):
    WORST[key] = max(WORST.get(key, 0.0), float(err))


@pytest.fixture(scope="module", autouse=True)
def _print_worst():
    yield
    if WORST:
        print("\nworst:", ", ".join(f"{k} {v:.3g}" for k, v in sorted(WORST.items())))


@pytest.fixture(scope="module")
def ne(native):
    require(native, "store")
    return require(native, "em")


@pytest.fixture(scope="module")
def NStore(native, ne):
    return H.store_class(native.qrss)


@pytest.fixture(scope="module")
def row(native, ne):
    from qrss_native import em as rows_em

    return {name: fn for (_t, name), fn in rows_em.rows(native.qrss).items()}


@pytest.fixture(scope="module")
def sp(native, ne):
    return native.qrss.special


def _typed(x):
    """A value with its type: numpy scalars against Python floats are part of
    the contract."""
    if isinstance(x, tuple):
        return tuple(_typed(v) for v in x)
    if isinstance(x, list):
        return [_typed(v) for v in x]
    if isinstance(x, dict):
        return {k: _typed(v) for k, v in x.items()}
    if isinstance(x, np.ndarray):
        return ("nd", x.dtype.str, x.shape, x.tobytes())
    if dataclasses.is_dataclass(x) and not isinstance(x, type):
        return (type(x).__name__, _typed(dataclasses.asdict(x)))
    if isinstance(x, np.generic):
        return (type(x).__name__, np.asarray(x).tobytes())
    return (type(x).__name__, repr(x))


def _outcome(fn, *a, **k):
    try:
        return "ok", _typed(fn(*a, **k))
    except Exception as e:  # noqa: BLE001 - the reference's own type and text
        return type(e).__name__, str(e)


# --- the references --------------------------------------------------------------------------------

def _lmmse_inputs():
    rng = np.random.default_rng(5)
    out = []
    for n in (0, 1, 7, 1024, 4096):
        s = rng.standard_normal(n) * 3
        w = rng.uniform(-1, 5, n)
        w[rng.random(n) < 0.2] = 0.0
        out.append((s, w))
    s = rng.standard_normal(64)
    w = rng.uniform(0, 5, 64)
    s[3], w[4], w[5], w[6], s[7], w[8], w[9], s[10] = np.nan, np.nan, np.inf, -np.inf, np.inf, -0.0, -1.0, -0.0
    out += [(s, w), (s[:1], w), (s, w[:1]), (s.reshape(8, 8), w.reshape(8, 8)), (s.reshape(8, 8), w[:8]),
            (s[:8], w.reshape(8, 8)), (s.reshape(2, 4, 8), w[:8].reshape(1, 8)), (s[:3], w[:4]),
            (s.reshape(8, 8), w[:7]), (2.0, 1.0), (2.0, w[:5]), (s[:5], -3.0), (list(s[:5]), list(w[:5])),
            (s.astype(np.float32), w.astype(np.float32)), (np.arange(5), np.arange(5)), ([], []),
            (np.float32(1.5), np.float64(0.25)), (True, 2)]
    return out


def test_lmmse_and_prior_from(row):
    """lmmse_latents and prior_from bit for bit: values, dtypes, shapes,
    numpy scalars, broadcasting and its refusal."""
    for i, (s, w) in enumerate(_lmmse_inputs()):
        assert _outcome(row["lmmse_latents"], s, w) == _outcome(em.lmmse_latents, s, w), i
    rng = np.random.default_rng(6)
    for n in (1024, 4096, 16384, 50600, 1, 7, 0, 100):
        s = rng.standard_normal(n) * 2
        w = rng.uniform(0, 4, n) * (rng.random(n) > 0.25)
        if n > 10:
            w[5], s[6] = np.nan, np.inf
        for q in (Q_TEST, 0, 1, 2 ** 40, -1, np.int64(77), 2 ** 64 - 1):
            assert _outcome(row["prior_from"], s, w, q) == _outcome(em.prior_from, s, w, q), (n, q)
    # length-1 broadcasting and lists, as the reference's arithmetic does
    s = rng.standard_normal(1024)
    for a, b in ((s, [0.5]), (s[:1], np.full(1024, 2.0)), (list(s), list(np.abs(s)))):
        assert _outcome(row["prior_from"], a, b, Q_TEST) == _outcome(em.prior_from, a, b, Q_TEST)


def _pair(NStore, root, passes, attach, *, py_cls=S.Store):
    """The same passes stored and attached by the reference (root/py) and the
    C++ (root/cpp)."""
    out = []
    for cls, sub in ((py_cls, "py"), (NStore, "cpp")):
        st = cls(root / sub, clock=tm.Clock())
        for p in passes:
            st.add_pass(p)
        for key, uid, g, mode, codec in attach:
            st.attach(key, uid, g, "test", mode=mode, codec_id=codec)
        out.append(st)
    return out


def _stores(NStore, root, passes, attach):
    """(reference store, C++ store) pairs over both trees: each
    implementation reading what it wrote and what the other wrote."""
    py, cpp = _pair(NStore, root, passes, attach)
    return [(py, cpp), (S.Store(root / "cpp", clock=tm.Clock()), NStore(root / "py", clock=tm.Clock()))]


@functools.lru_cache(maxsize=None)
def _picture(seed, mode=0):
    spic = synthetic_picture(seed, mode=mode)
    return spic, make_header(spic, callsign="K1ABC")


def _key(h):
    return (h.callsign, int(h.picture_id))


def test_em_prior_and_mean_w(row, native, NStore, tmp_path):
    """em_prior (leave one out) bit for bit on both stores, its refusals,
    and acc_mean_w_db."""
    spic, h = _picture(0)
    a = picture.air_values(spic.segs[0], 0)
    ps = [make_pass_result(a[:4096], 0.5 + s, seed=s) for s in range(3)]
    ps.append(make_pass_result(a[:4096], 2.0, seed=3, erase_after=1000))
    ps.append(make_pass_result(a[:1024], 1.5, seed=4))                    # a TINY member
    att = [(_key(h), p.uid, 0, h.mode, h.codec_id) for p in ps]
    other = make_pass_result(a[:4096], 1.0, seed=9)
    for py, cpp in _stores(NStore, tmp_path, ps + [other], att):
        acc_py, acc_c = py.accumulator(_key(h)), cpp.accumulator(_key(h))
        for p in ps + [other]:
            for spec in (frame.SHORT, frame.TINY, frame.MEDIUM, lframe.L_SHORT):
                for q in (Q_TEST, p.q, 3):
                    ref = _outcome(em.em_prior, acc_py, p.uid, py, q, spec)
                    assert _outcome(row["em_prior"], acc_c, p.uid, cpp, q, spec) == ref, (p.uid, spec.name)
        _mean_w_same(row, acc_c, acc_py)
    empty_py = S.Accumulator.empty(("NONE", 1), 0, 1, 0.0)
    empty_c = native.qrss.store.Accumulator.empty(("NONE", 1), 0, 1, 0.0)
    assert row["acc_mean_w_db"](empty_c) == em.acc_mean_w_db(empty_py) == float("-inf")


def _mean_w_same(row, acc_c, acc_py):
    g, r = row["acc_mean_w_db"](acc_c), em.acc_mean_w_db(acc_py)
    W = np.asarray(acc_py.W, np.float64)
    m = float(np.mean(W[W > 0]))
    assert type(g) is float and g == 10 * math.log10(m)
    assert r == g or abs(r - g) <= math.ulp(g), (r, g)
    _worst("acc_mean_w_db ulp", abs(r - g) / math.ulp(g))


# --- what every member pass is known to have sent -----------------------------------------------------

def test_known_header(row, NStore, tmp_path):
    """known_header's decisions on both stores: a member's decoded header, the
    soft header of summed LLRs, a key mismatch, segments outside the mode,
    mixed CE / L members, LLRs of no header."""
    spic, h = _picture(0)
    sp1, h1 = _picture(1, mode=1)
    sp2, h2 = _picture(2)
    a = picture.air_values(spic.segs[0], 0)
    weak = [make_pass_result(a[:4096], 0.5, seed=s, hdr=h, hdr_es_n0_db=-16.0) for s in range(6)]
    assert all(header.decode(p.hdr_llr.astype(float)) is None for p in weak)
    lweak = [make_pass_result(a[:4096], 0.5, seed=20 + s, hdr=h, hdr_es_n0_db=-16.0, waveform=WAVEFORM_L)
             for s in range(6)]
    decoded1 = make_pass_result(a[:4096], 0.5, seed=30, hdr=h1, decoded=True)
    decoded1_seg1 = make_pass_result(a[:4096], 0.5, seed=31, hdr=dataclasses.replace(h1, segment=1), decoded=True)
    foreign = make_pass_result(a[:4096], 0.5, seed=32, hdr=h2, decoded=True)          # another picture's header
    # ... decoded, and its LLRs, which still join the soft sum (only a matching
    # decoded header ends the search)
    foreign_llr_decoded = make_pass_result(a[:4096], 0.5, seed=33, hdr=h2, decoded=True, hdr_es_n0_db=-3.0)
    foreign_llr = [make_pass_result(a[:4096], 0.5, seed=40 + s, hdr=h2, hdr_es_n0_db=-6.0) for s in range(2)]
    nothing = [make_pass_result(a[:4096], 0.5, seed=50 + s) for s in range(2)]
    tiny = make_pass_result(a[:1024], 0.5, seed=60)
    tiny.hdr_llr = np.ones(10, np.float32)                                            # not a header's length
    strong = make_pass_result(a[:4096], 0.5, seed=61, hdr=h, hdr_es_n0_db=0.0)
    scen = {
        "soft": [(p, 0) for p in weak],
        "soft-L": [(p, 0) for p in lweak],
        "soft-mixed": [(p, 0) for p in weak[:3] + lweak[:3]],
        "soft-split": [(p, k % 2) for k, p in enumerate(weak)],                       # segments 0, 1, 0, ...
        "soft-short": [(p, 0) for p in weak[:2]],                                     # too few: no header
        "decoded": [(decoded1, 0)],
        "decoded-later": [(nothing[0], 0), (decoded1_seg1, 1)],
        "foreign": [(foreign, 0)],
        "foreign-llr": [(p, 0) for p in foreign_llr],
        "foreign+soft": [(foreign, 0)] + [(p, 0) for p in weak],
        "foreign-decoded-llr+soft": [(foreign_llr_decoded, 0)] + [(p, 0) for p in weak],
        "nothing": [(p, 0) for p in nothing] + [(tiny, 0)],
        "strong-after-weak": [(weak[0], 1), (strong, 0)],
    }
    keys = {"decoded": h1, "decoded-later": h1}
    for name, members in scen.items():
        hk = keys.get(name, h)
        att = [(_key(hk), p.uid, g, 2, hk.codec_id) for p, g in members]
        for py, cpp in _stores(NStore, tmp_path / name, [p for p, _ in members], att):
            acc_py, acc_c = py.accumulator(_key(hk)), cpp.accumulator(_key(hk))
            for g in (0, 1, 2, -1, 5):
                ref = _outcome(em.known_header, py, acc_py, g)
                assert _outcome(row["known_header"], cpp, acc_c, g) == ref, (name, g)
            if name in ("soft", "decoded", "soft-L", "soft-split"):
                assert _outcome(em.known_header, py, acc_py, 0)[1][0] == "HeaderFields", name


def _cw(keying, z=8.0, agrees=True):
    return CwIdResult(text="K1ABC", z_match=z, keying=keying, agrees=agrees)


def test_known_keying(row, NStore, tmp_path):
    """known_keying's (units, ook) or None: OOK against FSK votes, votes that
    do not count, members of the other waveform, refused callsigns."""
    spic, h = _picture(0)
    a = picture.air_values(spic.segs[0], 0)
    mk = functools.partial(make_pass_result, a[:4096], 0.5)
    variants = {
        "fsk": [_cw("fsk")], "ook": [_cw("ook")], "ook2-fsk1": [_cw("ook"), _cw("ook"), _cw("fsk")],
        "tie": [_cw("ook"), _cw("fsk")], "weak": [_cw("ook", z=6.0), _cw("ook", z=float("nan"))],
        "disagree": [_cw("ook", agrees=False), _cw("ook", agrees=None)], "none": [None, None],
        "other-keying": [_cw("xyz"), _cw("xyz")], "other+fsk": [_cw("xyz"), _cw("xyz"), _cw("fsk")],
    }
    calls = (None, "K1ABC", "K1ABC  ", "k1abc", "", "TOOLONGCALL", "K1/ABC", "K 1", "1", "00000000", 5, b"K1ABC",
             "K1ÄBC")
    seen = set()
    for name, cws in variants.items():
        for wf_members in (WAVEFORM_CE, WAVEFORM_L, "mixed"):
            ps = []
            for k, cw in enumerate(cws):
                wf = (WAVEFORM_CE, WAVEFORM_L)[k % 2] if wf_members == "mixed" else wf_members
                ps.append(dataclasses.replace(mk(seed=100 + k, waveform=wf), cw=cw))
            att = [(_key(h), p.uid, 0, h.mode, h.codec_id) for p in ps]
            sub = tmp_path / f"{name}-{wf_members}"
            for py, cpp in _stores(NStore, sub, ps, att):
                acc_py, acc_c = py.accumulator(_key(h)), cpp.accumulator(_key(h))
                for call in calls:
                    for wf in (WAVEFORM_CE, WAVEFORM_L, 7):
                        ref = _outcome(em.known_keying, py, acc_py, call, wf)
                        assert _outcome(row["known_keying"], cpp, acc_c, call, wf) == ref, (name, call, wf)
                        seen.add(ref if ref[1] == ("NoneType", "None") else ref[1][1])
                assert _outcome(row["known_keying"], cpp, acc_c, "K1ABC") == \
                    _outcome(em.known_keying, py, acc_py, "K1ABC")
    # every decision was reached: None, FSK and OOK
    assert seen >= {("ok", ("NoneType", "None")), ("bool", "False"), ("bool", "True")}, seen


# --- a stored pass's frame -------------------------------------------------------------------------------

def _all_specs():
    return (list(frame.PRESETS.values()) + list(frame.LEGACY.values()) + list(frame.PROTOTYPES.values())
            + list(lframe.PRESETS.values()))


def test_pass_spec_is_l_ce_only(row, ne, NStore, tmp_path):
    """pass_spec (the preset object itself), is_l and _ce_only's refusal on
    every frame name and header version, duck-typed and stored passes."""
    _, h = _picture(0)
    names = [s.name for s in _all_specs()] + ["nope", "l-nope", "", "FULL"]
    hdrs = (None, h, dataclasses.replace(h, version=1), dataclasses.replace(h, version=0),
            dataclasses.replace(h, version=3), SimpleNamespace(version=1))
    presets = _all_specs()
    for name in names:
        for hd in hdrs:
            p = SimpleNamespace(frame=name, header=hd)
            ref = _outcome(em.pass_spec, p)
            assert _outcome(row["pass_spec"], p) == ref, (name, hd)
            if ref[0] == "ok":
                r, g = em.pass_spec(p), row["pass_spec"](p)
                assert type(g) is type(r) and g == r
                if any(r is v for v in presets):
                    assert g is r, (name, hd)
            assert _outcome(row["is_l"], p) == _outcome(em.is_l, p), (name, hd)
    for spec in presets:
        for what in ("em.pass_path", "x 'y' \"z\"", ""):
            assert _outcome(row["_ce_only"], spec, what) == _outcome(em._ce_only, spec, what)
            if _outcome(em.pass_spec, SimpleNamespace(frame=spec.name, header=None))[0] == "ok":
                nspec = ne.pass_spec(SimpleNamespace(frame=spec.name, header=None))   # the native spec as input
                assert _outcome(row["_ce_only"], nspec, what) == _outcome(em._ce_only, spec, what)
    # stored passes, as em_refine hands them over
    spic, h = _picture(0)
    a = picture.air_values(spic.segs[0], 0)
    ps = [make_pass_result(a[:n], 0.5, seed=k, hdr=hd, decoded=True, waveform=wf)
          for k, (n, hd, wf) in enumerate([(4096, h, WAVEFORM_CE), (4096, dataclasses.replace(h, version=1), WAVEFORM_CE),
                                           (4096, dataclasses.replace(h, version=1), WAVEFORM_L),
                                           (1024, None, WAVEFORM_CE)])]
    py, cpp = _pair(NStore, tmp_path, ps, [])
    for p in ps:
        r, g = em.pass_spec(py.load_pass(p.uid)), row["pass_spec"](cpp.load_pass(p.uid))
        assert g is r, p.frame
        assert row["is_l"](cpp.load_pass(p.uid)) is em.is_l(py.load_pass(p.uid))


# --- the paths ---------------------------------------------------------------------------------------------

@functools.lru_cache(maxsize=None)
def _tracks():
    """The reference's tracks of a CE TINY and an l-tiny pass (round A, and
    B where it ran)."""
    from qrss_demod_tracks import _ce_sim, flat_det
    from qrss_fakes import l_scenario

    from sstvae.qrss import frontend
    from sstvae.qrss import receiver as RX

    out = []
    spec, sim = _ce_sim("tiny", -6.0, 3)
    prep = RX.prepare(frontend.Capture(sim.fe, Q_TEST, sim.t0_index))
    out += RX.receive_pass(prep, spec, flat_det(1500.0), return_tracks=True)[1]
    _, sim = l_scenario("l-tiny", -6.0, 3)
    prep = RX.prepare(frontend.Capture(sim.fe, Q_TEST, sim.t0_index))
    out += RX.receive_pass(prep, lframe.L_TINY, flat_det(1500.0, WAVEFORM_L), return_tracks=True)[1]
    return out


def _path_close(g, r, what):
    assert type(g) is type(r)
    assert g.t_s.dtype == r.t_s.dtype and np.array_equal(g.t_s, r.t_s), what
    assert np.array_equal(g.weight, r.weight) and g.weight.dtype == r.weight.dtype, what
    assert g.f_hz.dtype == r.f_hz.dtype and g.f_hz.shape == r.f_hz.shape, what
    if r.f_hz.size:
        err = float(np.max(np.abs(g.f_hz - r.f_hz)) / np.max(np.abs(r.f_hz)))
        _worst("_path_of f_hz", err)
        assert err <= 1e-12, (what, err)


def test_psi_frame_path_of_flat(row):
    """_psi_frame bit for bit, _path_of (times exact, the path to 1e-12) on
    real tracks with their own and shifted timing, the empty grid's
    IndexError, and _flat."""
    trs = _tracks()
    assert len(trs) >= 2
    for k, tr in enumerate(trs):
        g, r = row["_psi_frame"](tr), em._psi_frame(tr)
        assert g.dtype == r.dtype == np.float32 and np.array_equal(g, r), k
        _path_close(row["_path_of"](tr), em._path_of(tr), k)
        for d_tau, ppm in ((0.37, 0.0), (-5.25, 120.0), (1e-9, -80.0)):
            t2 = copy.copy(tr)
            t2.timing = dataclasses.replace(tr.timing, tau0=tr.timing.tau0 + d_tau, ppm=ppm)
            _path_close(row["_path_of"](t2), em._path_of(t2), (k, d_tau, ppm))
        t3 = copy.copy(tr)
        t3.pos = tr.pos[:1]
        _path_close(row["_path_of"](t3), em._path_of(t3), (k, "one position"))
        t4 = copy.copy(tr)
        t4.pos = tr.pos[:0]
        assert _outcome(row["_path_of"], t4) == _outcome(em._path_of, t4)
    for f in (1500.0, 1500.25, np.float32(3.3), Fraction(3, 2), 7, -0.0, float("nan"), float("inf"), np.int64(9)):
        assert _outcome(row["_flat"], f) == _outcome(em._flat, f), f


# --- the tail statistics -------------------------------------------------------------------------------------

def _rel(g, r):
    if g == r or (math.isnan(g) and math.isnan(r)):
        return 0.0
    return abs(g - r) / max(abs(r), 1e-300)


def test_special_tails(sp):
    """ndtri / norm_isf / lgamma bit for bit with scipy and math; gamma.logsf
    to 1e-12 relative (numpy's SIMD log in the reference)."""
    from scipy import special, stats

    q = np.concatenate([np.logspace(np.log10(0.5), -300, 3000), [0.0, 1.0, 0.5, 5e-324, 1e-310, 2.2250738585072014e-308,
                                                                np.nan, -0.1, 1.1, 0.135, 0.8646647167633873,
                                                                1 - 1e-16, np.inf, -np.inf]])
    q = np.concatenate([q, 1.0 - q[:1500]])
    for g, r in ((sp.ndtri(q), special.ndtri(q)), (sp.norm_isf(q), stats.norm.isf(q))):
        assert np.array_equal(g.view(np.int64), r.view(np.int64)) or \
            np.array_equal(g, r, equal_nan=True) and np.array_equal(np.signbit(g), np.signbit(r))
    assert type(sp.ndtri(0.25)) is np.float64
    xs = np.concatenate([np.random.default_rng(1).uniform(-60, 60, 4000), np.arange(-4, 1001, 0.5),
                         [1e-21, -1e-21, 2.5e305, 1e308, np.inf, -np.inf, np.nan, 1e-320]])
    for x, v in zip(xs, sp.py_lgamma(xs)):
        try:
            r = math.lgamma(x)
        except (ValueError, OverflowError):
            r = math.inf
        assert v == r or (math.isnan(v) and math.isnan(r)), x
    worst = 0.0
    for a in (1, 2, 3, 5, 10, 33, 100, 300, 891, 900, 1000, 0.5):
        x = np.concatenate([np.linspace(0, 8 * a + 4000, 1500), [-1.0, 0.0, np.inf, np.nan, a, a - 1e-9, 5e-324]])
        g, r = sp.gamma_logsf(x, a), stats.gamma.logsf(x, a)
        assert np.array_equal(np.isnan(g), np.isnan(r)) and np.array_equal(np.isinf(g), np.isinf(r)), a
        fin = np.isfinite(r) & (r != 0)
        assert np.array_equal(g[r == 0], r[r == 0])
        if fin.any():
            worst = max(worst, float(np.max(np.abs(g[fin] - r[fin]) / np.abs(r[fin]))))
    for a in (0.0, -1.0, np.nan):
        assert np.isnan(sp.gamma_logsf(3.0, a)) and np.isnan(stats.gamma.logsf(3.0, a))
    _worst("gamma_logsf rel", worst)
    assert worst <= 1e-12


def test_gauss_z(row):
    """_gauss_z within 1e-12 relative everywhere, and beyond -700 (the Mills
    ratio) within 0.01 of the exact Z."""
    from scipy import stats

    lps = np.concatenate([-np.logspace(-6, 4, 4000), [0.0, math.log(0.5), math.log(0.5) - 1e-16, -700.0,
                                                       np.nextafter(-700.0, 0), np.nextafter(-700.0, -np.inf),
                                                       -1e300, -np.inf, np.inf, np.nan, 5.0, -745.2, -746.0]])
    worst = 0.0
    for lp in lps:
        lp = float(lp)
        g, r = _outcome(row["_gauss_z"], lp), _outcome(em._gauss_z, lp)
        assert g[0] == r[0] == "ok" and g[1][0] == r[1][0] == "float", lp
        zg, zr = row["_gauss_z"](lp), em._gauss_z(lp)
        e = _rel(zg, zr)
        worst = max(worst, e)
        assert e <= 1e-12, (lp, zg, zr)
        if -1e5 < lp <= -700.0:
            # the exact Z for this tail: |logsf(Z) - log_p| / Z is the error in Z
            assert abs(float(stats.norm.logsf(zg)) - lp) / zg < 0.01, lp
    _worst("_gauss_z rel", worst)


def test_template_z(row):
    """template_z within 1e-12 relative over Z 0..40, n_cells 1..1e9,
    n_chunks 1..900, far tails through the leading-term fallback, and the
    refusals' texts."""
    from scipy import stats

    worst = 0.0
    for K in (1, 2, 3, 5, 10, 33, 100, 300, 891, 900):
        for n_cells in (1.0, 0.0, 10.0, 1e4, 1e9, 0.5):
            lams = [stats.gamma.isf(10.0 ** -e, K) for e in np.linspace(0.0, 350.0, 36)]
            lams += [K * f for f in (0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 64.0)] + [K + 2000.0, K + 1e5]
            for lam in lams:
                lam = float(lam)
                zg, zr = row["template_z"](lam, K, n_cells), em.template_z(lam, K, n_cells)
                assert type(zg) is type(zr) is float
                e = _rel(zg, zr)
                worst = max(worst, e)
                assert e <= 1e-12, (lam, K, n_cells, zg, zr)
    _worst("template_z rel", worst)
    zs = [row["template_z"](300 * f, 300, 1e5) for f in (2.0, 4.0, 8.0, 16.0)]
    assert all(np.isfinite(zs)) and all(np.diff(zs) > 0)
    for args in ((0.0, 5, 1.0), (-1.0, 5, 1.0), (3.0, 0, 1.0), (0.0, 0, 1.0), (-1.0, 0, 1.0), (float("nan"), 5, 1.0),
                 (float("inf"), 5, 1.0), (float("inf"), 1, 1.0), (5.0, 5, float("nan")), (5.0, 5, -3.0),
                 (5.0, -2, 1.0), (1e6, 1, 1.0), (1e6, 900, float("inf")), (np.float64(400.0), np.int64(300), 1e3),
                 (400, 300, 1000)):
        assert _outcome(row["template_z"], *args) == _outcome(em.template_z, *args), args


# --- the hooks -------------------------------------------------------------------------------------------------

def test_hooks(ne):
    """log and progress in order; a cancel set during step k (from another
    thread) stops the call at step k + 1; a hook's exception passes through."""
    lines, prog = [], []
    ne._run_hooks(3, log=lines.append, progress=lambda d, t, w: prog.append((d, t, w)))
    assert lines == ["step 0 of 3", "step 1 of 3", "step 2 of 3"]
    assert prog == [(1, 3, "steps"), (2, 3, "steps"), (3, 3, "steps")]
    ne._run_hooks(2)                                             # no hooks at all
    flag = ne.CancelFlag()
    assert not flag.is_set()
    lines = []

    def log(line):
        lines.append(line)
        if line == "step 1 of 5":
            t = threading.Thread(target=flag.set)
            t.start()
            t.join()

    with pytest.raises(ne.Cancelled, match="cancelled"):
        ne._run_hooks(5, log=log, cancel=flag)
    assert lines == ["step 0 of 5", "step 1 of 5"] and flag.is_set()
    assert issubclass(ne.Cancelled, RuntimeError)
    with pytest.raises(ne.Cancelled):
        ne._run_hooks(1, log=lines.append, cancel=flag)            # already set: before any step
    flag.clear()
    ne._run_hooks(1, cancel=flag)

    def boom(line):
        raise ZeroDivisionError("from the hook")

    with pytest.raises(ZeroDivisionError, match="from the hook"):
        ne._run_hooks(3, log=boom)
    with pytest.raises(TypeError):
        ne._run_hooks(1, log=5)
    with pytest.raises(TypeError):
        ne._run_hooks(1, progress="x")
    with pytest.raises(TypeError):
        ne._run_hooks(1, cancel=threading.Event())


# --- the whole caller ------------------------------------------------------------------------------------------

def _apply_rows(native, mp):
    """Store, associate and this lane's rows over their targets and every
    from-import site in sstvae.qrss, through `mp`."""
    from qrss_native import em as rows_em

    H.apply_store_rows(native, mp)
    reference = qrss_native._import_reference()
    replaced = {}
    for key, repl in rows_em.rows(native.qrss).items():
        owner = qrss_native._owner(key[0])
        orig = getattr(owner, key[1])
        mp.setattr(owner, key[1], repl)
        replaced[id(orig)] = (orig, repl)
    for m in reference:
        for k, v in list(vars(m).items()):
            hit = replaced.get(id(v))
            if hit is not None and hit[0] is v:
                mp.setattr(m, k, hit[1])


def _em_run(root, passes, key, h):
    st = S.Store(root, clock=tm.Clock())                         # whichever Store the module holds
    for p in passes:
        st.add_pass(p)
        st.attach(key, p.uid, 0, "test", mode=h.mode, codec_id=h.codec_id)
    log = []
    hist = em.em_refine(st, key, rounds=3, log=log.append)
    return hist, log


def test_em_refine_whole_caller(native, ne, tmp_path):
    """em.em_refine (the reference's loop) over this lane's rows and the
    store's: the same history and log, and the same files, as with every
    function Python. Synthetic and L members (no capture / not CE), so the
    loop reaches acc_mean_w_db, known_header, known_keying and pass_spec."""
    spic, h = _picture(0)
    a = picture.air_values(spic.segs[0], 0)
    ps = [make_pass_result(a[:4096], 0.5 + s, seed=s, hdr=h, hdr_es_n0_db=-16.0) for s in range(4)]
    ps.append(dataclasses.replace(make_pass_result(a[:4096], 1.0, seed=7, waveform=WAVEFORM_L, hdr=h, decoded=True),
                                  ch=np.ones(16, np.complex64), cw=_cw("fsk")))
    ref = _em_run(tmp_path / "py", ps, _key(h), h)
    used = {}
    with pytest.MonkeyPatch.context() as mp:
        _apply_rows(native, mp)
        for name in ("acc_mean_w_db", "known_header", "known_keying", "pass_spec"):
            fn = getattr(em, name)

            def spy(*a_, _fn=fn, _name=name, **k):
                used[_name] = used.get(_name, 0) + 1
                return _fn(*a_, **k)
            mp.setattr(em, name, spy)
        got = _em_run(tmp_path / "cpp", ps, _key(h), h)
    assert set(used) == {"acc_mean_w_db", "known_header", "known_keying", "pass_spec"}, used
    assert got[1] == ref[1] and len(ref[1]) == 1 and "waveform L" in ref[1][0]
    assert len(got[0]) == len(ref[0]) == 2
    for g, r in zip(got[0], ref[0]):
        assert g == r or abs(g - r) <= math.ulp(r), (got[0], ref[0])
    H.compare_trees(tmp_path / "py", tmp_path / "cpp")

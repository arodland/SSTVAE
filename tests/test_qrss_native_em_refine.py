"""Parity of native/core/qrss/em_refine (sstvae_native.qrss.em_refine) with
the EM loop of sstvae/qrss/em.py: pass_path, rereceive and em_refine.
tests/qrss_native/em_refine.py substitutes them under --native-qrss=rx.

The passes are CE TINY and SHORT passes simulated by sstvae/qrss/channel.py
(seeds, SNRs, the "quiet" and "disturbed" presets) and received by the
reference at the nominal frequency and timing (test_qrss_em.forced_pass:
the oracle's detection, no acquisition); every input a function takes
beyond the pass -- the prior, the path, the header, the callsign keying --
is the reference's, injected into both implementations alike.

The contract (the bar), and in brackets what was measured (2026-10-14):

- **pass_path**: the same branch (the line or its refinement) wherever the
  two candidates' Z_ref are not within 1e-7 of each other, both candidate
  paths with the times and weights bit for bit and the frequencies within
  1e-7 Hz, both Z_ref within 1e-7 of their scale (the track lanes' path
  and Z_ref bars), and the path chosen [worst printed]; a caller's channel
  given its noise table, as the reference's object is;
- **rereceive**, with the reference's prior and path injected, split and
  not, with and without a header, with and without the callsign keying,
  on the receive_pass branch (no header, or TINY, a frame without one) and
  the two-round one: the stage 2 bars of the receiver's parity file -- z,
  w, hdr_llr and psi to 1e-6 of their scale (z and w elementwise too),
  f_hz to 1e-6 Hz, tau0 to 1e-6 s, the timing's and report's numbers to
  1e-6 relative, the covariance to 1e-4, uid, header, cw, q, frame,
  estimator and em_round equal -- and the path it ended on within 1e-6 s
  and 1e-6 Hz [worst printed];
- **em_refine**, the whole caller, A/B on two copies of one store of 2 and
  3 passes (TINY in the fast tier, SHORT in the slow one): the same number
  of rounds, the history within 0.05 dB per entry [worst printed], the
  same passes re-received in the same order, the log lines the same text
  with their numbers within 0.011 dB (printed to 0.01), and every stored
  pass's z and w within 1e-5 of their scale after the rounds [worst
  printed];
- **refusals and edges**: a key with no accumulator (the same KeyError
  text), an accumulator whose members keep no capture (history of two
  equal entries, nothing logged, the same files), an L member (the
  reference's line, in round 1 only), rounds 0 and below (the history
  alone), a falsy estimator (the pass's own), an L pass and a pass without
  a capture given to rereceive (None, or (None, None)), an L pass given to
  pass_path (the same ValueError);
- **the seam**: em_refine through the rows with em.rereceive replaced by a
  deterministic spy calls the spy as the reference does (the same passes,
  rounds, priors bit for bit, headers and keying), and writes the same
  files;
- **the hooks** (the C++'s own): progress (done, total, what) after each
  pass with total = rounds x eligible members and the early-stop call;
  a cancel set from the progress hook stops the loop at the next poll with
  em.Cancelled and leaves a store whose every file loads, the passes
  stored before it updated and the rest untouched.

The same functions on a pinned TINY fixture, without Python, are
native/tests/qrss/test_em_refine.cpp's (pins from
native/tests/qrss/gen_em_refine_pins.py).
"""

import dataclasses
import functools
import math
import re
import shutil
import time
from fractions import Fraction

import numpy as np
import pytest
import qrss_native
import test_qrss_em as te
import test_qrss_multipass as tm
from qrss_fakes import make_pass_result
from qrss_helpers import Q_TEST
from qrss_native import _rx, require
from qrss_native import _store as H

from sstvae.qrss import em, frame, morse
from sstvae.qrss import store as S
from sstvae.qrss import track as T
from sstvae.qrss.constants import CH_FS, WAVEFORM_L

WORST: dict = {}
PASS_BAR = 1e-6          # the stage 2 bars (the receiver's parity file)
PATH_BAR = 1e-7          # the track lanes' path and Z_ref bars
STORE_BAR = 1e-5         # a stored pass after em_refine's rounds
HIST_BAR = 0.05          # dB, per history entry


def _worst(key, err):
    WORST[key] = max(WORST.get(key, 0.0), float(err))


@pytest.fixture(scope="module", autouse=True)
def _print_worst():
    yield
    if WORST:
        print("\nworst:", ", ".join(f"{k} {v:.3g}" for k, v in sorted(WORST.items())))


@pytest.fixture(scope="module")
def Er(native):
    require(native, "store")
    require(native, "em")
    return require(native, "em_refine")


@pytest.fixture(scope="module")
def NStore(native, Er):
    return H.store_class(native.qrss)


@pytest.fixture(scope="module")
def row(native, Er):
    from qrss_native import em_refine as rows_er

    return {name: fn for (_t, name), fn in rows_er.rows(native.qrss).items()}


# --- the passes -----------------------------------------------------------------------------------

# label: (frame, SNR dB, seed, preset)
CASES = {
    "tiny-6": ("tiny", -6.0, 1, "quiet"),
    "tiny-12d": ("tiny", -12.0, 2, "disturbed"),
    "tiny-15": ("tiny", -15.0, 3, None),
    "short-6": ("short", -6.0, 1, "quiet"),
    "short-12d": ("short", -12.0, 2, "disturbed"),
}
SLOW = {"short-12d"}
CALL = te.CALL


def _labels(only):
    return [pytest.param(k, marks=pytest.mark.slow) if k in SLOW else k for k in only]


@functools.cache
def _pass(label, i=0):
    """Pass i of the label's set: its own slot (q) and seed, received by the
    reference at the nominal frequency and timing."""
    name, snr, seed, preset = CASES[label]
    spec = frame.get(name)
    q = Q_TEST + 7 * i
    sim = te.simulate_pass(spec, snr, q, seed + 100 * i, preset)
    p = te.forced_pass(sim, spec, q)
    assert p is not None, (label, i)
    return p


def _spec(label):
    return frame.get(CASES[label][0])


def _hdr():
    return te.picture_and_header()[1]


def _cw():
    return (np.asarray(morse.keying_units(CALL), np.uint8), False)


@functools.cache
def _prior(label):
    """A prior as em_prior makes one: the truth seen through two passes'
    worth of noise (prior_from of S = W (a + n), W = 2), in pass 0's slot."""
    spec = _spec(label)
    a = te.air(spec)
    rng = np.random.default_rng(17)
    W = np.full(spec.n_data, 2.0)
    Sv = W * (a + rng.standard_normal(spec.n_data) / np.sqrt(W))
    return em.prior_from(Sv, W, _pass(label).q)


# --- comparison -----------------------------------------------------------------------------------

def _rel(a, b, what):
    """max |a - b| over max(1, max |b|), the same shape and dtype; NaN and
    infinities only where the other has them."""
    a, b = np.asarray(a), np.asarray(b)
    assert a.shape == b.shape, (what, a.shape, b.shape)
    assert a.dtype == b.dtype, (what, a.dtype, b.dtype)
    if a.size == 0:
        return 0.0
    ac, bc = a.astype(np.complex128), b.astype(np.complex128)
    same = (ac == bc) | (np.isnan(ac) & np.isnan(bc))
    if same.all():
        return 0.0
    assert np.isfinite(ac[~same]).all() and np.isfinite(bc[~same]).all(), what
    fin = np.isfinite(bc)
    scale = max(1.0, float(np.max(np.abs(bc[fin])))) if fin.any() else 1.0
    return float(np.max(np.abs(ac[~same] - bc[~same]))) / scale


def _elementwise(a, b):
    """max |a - b| / |b| over the elements that differ."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    d = a != b
    if not d.any():
        return 0.0
    return float(np.max(np.abs(a[d] - b[d]) / np.abs(b[d])))


def _num(x, y, tol, what, key):
    """|x - y| <= tol max(1, |y|) (NaN with NaN)."""
    x, y = float(x), float(y)
    if math.isnan(y) or math.isnan(x):
        assert math.isnan(x) and math.isnan(y), (what, x, y)
        return
    err = abs(x - y) / max(1.0, abs(y))
    _worst(key, err)
    assert err <= tol, (what, x, y)


def _abs(x, y, tol, what, key):
    err = abs(float(x) - float(y))
    _worst(key, err)
    assert err <= tol, (what, x, y)


def _path_close(g, r, what, key, tol=PATH_BAR, t_tol=0.0):
    assert type(g) is type(r), what
    assert g.t_s.shape == r.t_s.shape and g.t_s.dtype == r.t_s.dtype, (what, g.t_s.shape, r.t_s.shape)
    if t_tol == 0.0:
        assert np.array_equal(g.t_s, r.t_s), what
    else:
        _abs(np.max(np.abs(g.t_s - r.t_s), initial=0.0), 0.0, t_tol, (what, "t_s"), f"{key} t_s (s, abs)")
    _abs(np.max(np.abs(g.f_hz - r.f_hz), initial=0.0), 0.0, tol, (what, "f_hz"), f"{key} f_hz (Hz, abs)")
    assert g.f_hz.dtype == r.f_hz.dtype == np.float64, what
    assert np.array_equal(g.weight, r.weight), what


def _pass_close(g, r, what, key="rereceive", bar=PASS_BAR):
    assert (g is None) == (r is None), what
    if r is None:
        return
    assert type(g) is type(r), what
    assert (g.uid, g.q, g.frame, g.waveform, g.estimator, g.em_round, g.ch_fs) == \
        (r.uid, r.q, r.frame, r.waveform, r.estimator, r.em_round, r.ch_fs), what
    assert type(g.q) is int and type(g.ch_fs) is int and type(g.waveform) is int, what
    assert type(g.em_round) is int, what
    assert g.header == r.header, what
    assert (g.cw is None) == (r.cw is None), what
    if r.cw is not None:
        assert (g.cw.text, g.cw.keying, g.cw.agrees) == (r.cw.text, r.cw.keying, r.cw.agrees), what
        _num(g.cw.z_match, r.cw.z_match, bar, (what, "z_match"), f"{key} cw.z_match")
    _abs(g.f_hz, r.f_hz, 1e-6, (what, "f_hz"), f"{key} f_hz (Hz, abs)")
    _abs(g.timing.tau0, r.timing.tau0, 1e-6 * CH_FS, (what, "tau0"), f"{key} timing.tau0 (CH samples, abs)")
    for f in ("ppm", "gamma", "z"):
        _num(getattr(g.timing, f), getattr(r.timing, f), bar, (what, f), f"{key} timing.{f}")
    e = _rel(np.asarray(g.timing.cov, np.float64), np.asarray(r.timing.cov, np.float64), (what, "cov"))
    _worst(f"{key} timing.cov", e)
    assert e <= 1e-4, (what, "cov", e)
    for f in ("offset_hz", "drift_hz_per_min", "wander_hz_rms", "doppler_hz", "ppm", "snr2500_db", "z_ref",
              "kappa"):
        _num(getattr(g.report, f), getattr(r.report, f), bar, (what, f), f"{key} report.{f}")
    assert (g.report.slip_free, g.report.suspect) == (r.report.slip_free, r.report.suspect), what
    for f in ("z", "w", "hdr_llr", "psi", "ch"):
        e = _rel(getattr(g, f), getattr(r, f), (what, f))
        _worst(f"{key} {f}", e)
        assert e <= bar, (what, f, e)
    for f in ("z", "w"):
        e = _elementwise(getattr(g, f), getattr(r, f))
        _worst(f"{key} {f} (elementwise)", e)
        assert e <= bar, (what, f, "elementwise", e)
    assert g.ch_t0_index == r.ch_t0_index and type(g.ch_t0_index) is float, what
    assert g.f_mix_hz == r.f_mix_hz and isinstance(g.f_mix_hz, Fraction), what


# --- pass_path ------------------------------------------------------------------------------------

def _ref_trace(p):
    """What the reference's pass_path measures: (line, refined, z_line,
    z_refined), by its own calls."""
    from sstvae.qrss import receiver as RX

    spec = em.pass_spec(p)
    det = RX.detection_from_pass(p)
    chan = RX.channel_for(RX.capture_from_pass(p), spec, det, f_mix_hz=p.f_mix_hz)
    line = T.line_path(chan, spec, em._flat(p.f_hz))
    refined = T.refine_path(chan, spec, line)
    zs = [T.verify_track(chan, spec, dataclasses.replace(det, path=pth), fit=False).z_ref
          for pth in (line, refined)]
    return line, refined, zs[0], zs[1]


# TINY at -15 dB with no fading puts q at the tracker's grid floor, where the
# reference itself moves Z_ref by ~8e-7 relative for a 1e-13-sample nudge of
# its own timing: the track lane holds Z_ref to 1e-5 there
# (test_qrss_native_track_timing.py), and so does this file.
Z_REF_BAR = {"tiny-15": 1e-5}


@pytest.mark.parametrize("label", _labels(["tiny-6", "tiny-12d", "tiny-15", "short-6", "short-12d"]))
def test_pass_path(Er, row, label):
    """The same candidates, Z_ref and branch; the same path chosen."""
    p = _pass(label)
    ref = _ref_path(label)
    line, refined, z_line, z_ref_ = _ref_trace(p)
    choice = 1 if z_ref_ > z_line else 0
    _path_close(ref, (line, refined)[choice], (label, "the reference's own choice"), "self", tol=0.0)
    tr = Er._pass_path_trace(p)
    _path_close(_rx.to_ref(tr["line"]), line, (label, "line"), "pass_path line")
    _path_close(_rx.to_ref(tr["refined"]), refined, (label, "refined"), "pass_path refined")
    bar = Z_REF_BAR.get(label, PATH_BAR)
    key = "pass_path Z_ref" + (" (q floor)" if label in Z_REF_BAR else "")
    _num(tr["z_line"], z_line, bar, (label, "z_line"), key)
    _num(tr["z_refined"], z_ref_, bar, (label, "z_refined"), key)
    if abs(z_ref_ - z_line) > 2 * bar * max(1.0, abs(z_line)):
        assert tr["choice"] == choice, (label, z_line, z_ref_, tr)
    got = row["pass_path"](p)
    _path_close(got, ref, (label, "pass_path"), "pass_path")


def test_pass_path_given_chan(Er, row):
    """pass_path(p, chan): the caller's channel (the reference's object) is
    used as it is and keeps the noise table the C++ computed."""
    from sstvae.qrss import receiver as RX

    p = _pass("tiny-6")
    spec = em.pass_spec(p)
    det = RX.detection_from_pass(p)
    mk = functools.partial(RX.channel_for, RX.capture_from_pass(p), spec, det, f_mix_hz=p.f_mix_hz)
    c_ref, c_cpp = mk(), mk()
    ref = em.pass_path(p, c_ref)
    assert c_cpp._psd is None
    got = row["pass_path"](p, c_cpp)
    _path_close(got, ref, "pass_path(chan)", "pass_path")
    assert c_ref._psd is not None and c_cpp._psd is not None
    for a, b in zip(c_cpp._psd, c_ref._psd):
        e = _rel(a, b, "chan._psd")
        _worst("chan._psd", e)
        assert e <= 1e-12
    # a dict channel too (the C++'s own record)
    d = _rx.to_ref(Er.pass_path(p, {"__type__": "ChanCapture", "ch": c_ref.ch, "fs": c_ref.fs,
                                    "t0_index": c_ref.t0_index, "f_mix": c_ref.f_mix, "keep": None,
                                    "exclude_hz": (), "_psd": None}))
    _path_close(d, ref, "pass_path(dict chan)", "pass_path")


# --- rereceive ------------------------------------------------------------------------------------

# variant: (prior, header, callsign keying, split, path injected)
VARIANTS = {
    "plain": (False, False, False, True, True),
    "prior": (True, False, False, True, True),
    "hdr-split": (True, True, False, True, True),
    "hdr-split-cw": (True, True, True, True, True),
    "hdr-nosplit-cw": (True, True, True, False, True),
    "hdr-noprior-cw": (False, True, True, True, True),
    "own-path": (True, True, True, True, False),
}
# (label, variant) pairs of the fast tier; the rest are slow. TINY has no
# header, so its "hdr" variants run the receive_pass branch with a header
# given (the reference's `not spec.has_header`); SHORT's receive_pass branch
# ("plain", "prior") is the slow tier's, TINY covering it in the fast one.
FAST = {("tiny-6", v) for v in VARIANTS} | {("tiny-12d", "prior"), ("tiny-12d", "hdr-split"),
                                             ("tiny-15", "own-path"),
                                             ("short-6", "hdr-split-cw"), ("short-6", "hdr-nosplit-cw"),
                                             ("short-6", "hdr-noprior-cw")}


def _rr_params():
    out = []
    for label in CASES:
        for v in VARIANTS:
            mark = () if (label, v) in FAST else (pytest.mark.slow,)
            out.append(pytest.param(label, v, marks=mark, id=f"{label}-{v}"))
    return out


@functools.cache
def _ref_path(label):
    return em.pass_path(_pass(label))


@pytest.mark.parametrize("label,variant", _rr_params())
def test_rereceive(Er, row, label, variant):
    """rereceive with the reference's prior, path, header and keying: the
    same pass and final path, within the stage 2 bars."""
    use_prior, use_hdr, use_cw, split, inject = VARIANTS[variant]
    p = _pass(label)
    args = (_prior(label) if use_prior else None, _hdr() if use_hdr else None, _cw() if use_cw else None)
    kw = {"path": _ref_path(label) if inject else None, "split": split, "return_path": True}
    t = time.perf_counter()
    ref, ref_path = em.rereceive(p, *args, None, 2, **kw)
    t_ref = time.perf_counter() - t
    t = time.perf_counter()
    got, got_path = row["rereceive"](p, *args, None, 2, **kw)
    t_cpp = time.perf_counter() - t
    _worst(f"time {CASES[label][0]} rereceive C++/Python", t_cpp / t_ref)
    _pass_close(got, ref, (label, variant))
    assert got.em_round == 2
    _path_close(got_path, ref_path, (label, variant, "path"), "rereceive path", tol=1e-6, t_tol=1e-6)
    # without return_path: the pass alone, the same
    if variant == "plain":
        _pass_close(row["rereceive"](p, *args, None, 2, path=kw["path"], split=split), ref, (label, "no path"))


def test_rereceive_estimator_and_round(Er, row):
    """estimator "plain" against the pass's "joint"; a falsy estimator is the
    pass's own; em_round None keeps the pass's."""
    p = _pass("tiny-6")
    path = _ref_path("tiny-6")
    for est in ("plain", "", None):
        ref = em.rereceive(p, _prior("tiny-6"), None, None, est, None, path=path)
        got = row["rereceive"](p, _prior("tiny-6"), None, None, est, None, path=path)
        _pass_close(got, ref, ("estimator", est))
        assert got.estimator == ("plain" if est else p.estimator)
        assert got.em_round == p.em_round
    for bad in ("nope",):
        with pytest.raises(ValueError) as r:
            em.rereceive(p, None, None, None, bad, path=path)
        with pytest.raises(ValueError) as g:
            row["rereceive"](p, None, None, None, bad, path=path)
        assert str(g.value) == str(r.value)


def test_rereceive_refusals(Er, row):
    """An L pass and a pass without a capture: None (or (None, None)); an L
    pass to pass_path: the reference's ValueError."""
    p = _pass("tiny-6")
    a = te.air(frame.SHORT)
    pl = dataclasses.replace(make_pass_result(a, 1.0, seed=5, waveform=WAVEFORM_L), ch=np.ones(16, np.complex64))
    for q in (pl, dataclasses.replace(p, ch=np.zeros(0, np.complex64))):
        assert em.rereceive(q, None) is None and row["rereceive"](q, None) is None
        assert em.rereceive(q, None, return_path=True) == (None, None) == row["rereceive"](q, None, return_path=True)
    with pytest.raises(ValueError) as r:
        em.pass_path(pl)
    with pytest.raises(ValueError) as g:
        row["pass_path"](pl)
    assert str(g.value) == str(r.value) and "CE-only" in str(r.value)
    # a callsign keying with no units is nothing the reference makes: refused by name
    with pytest.raises(NotImplementedError):
        Er.rereceive(p, None, None, (None, True), path=_rx_path(_ref_path("tiny-6")))


def _rx_path(fp):
    return {"__type__": "FreqPath", "t_s": fp.t_s, "f_hz": fp.f_hz, "weight": fp.weight}


def test_rereceive_cancel(Er, native):
    """A cancel already set: em.Cancelled, nothing returned."""
    flag = native.qrss.em.CancelFlag()
    flag.set()
    with pytest.raises(native.qrss.em.Cancelled):
        Er.rereceive(_pass("tiny-6"), None, cancel=flag)


# --- em_refine, the whole caller ------------------------------------------------------------------

def _base_store(root, passes, h):
    st = S.Store(root, clock=tm.Clock())
    for p in passes:
        st.add_pass(p)
        st.attach(te.key_of(h), p.uid, 0, "test", mode=h.mode, codec_id=h.codec_id)
    return st


_LINE = re.compile(r"^round (\d+) pass (\S+): mean W ([+-]\d+\.\d\d) -> ([+-]\d+\.\d\d) dB$")


def _logs_close(got, ref, what):
    assert len(got) == len(ref), (what, got, ref)
    for g, r in zip(got, ref):
        mg, mr = _LINE.match(g), _LINE.match(r)
        if mr is None:
            assert g == r, what
            continue
        assert mg is not None, (what, g)
        assert mg.group(1, 2) == mr.group(1, 2), (what, g, r)
        for k in (3, 4):
            _abs(float(mg.group(k)), float(mr.group(k)), 0.011, (what, g, r), "em_refine log (dB)")


def _stores_close(py_root, cpp_root, NStore, key, what, bar=STORE_BAR):
    sp, sc = S.Store(py_root, clock=tm.Clock()), NStore(cpp_root, clock=tm.Clock())
    ap, ac = sp.accumulator(key), sc.accumulator(key)
    assert list(ap.uids) == list(ac.uids), what
    for uid in ap.uids:
        r, g = sp.load_pass(uid), sc.load_pass(uid)
        assert (g.em_round, g.estimator, g.header) == (r.em_round, r.estimator, r.header), (what, uid)
        for f in ("z", "w"):
            e = _rel(getattr(g, f), getattr(r, f), (what, uid, f))
            _worst(f"em_refine stored {f}", e)
            assert e <= bar, (what, uid, f, e)
    for f in ("S", "W"):
        e = _rel(np.asarray(getattr(ac, f), np.float64), np.asarray(getattr(ap, f), np.float64), (what, f))
        _worst(f"em_refine accumulator {f}", e)
        assert e <= bar, (what, f, e)


def _ab(tmp_path, NStore, Er, passes, rounds=3, tol_db=em.EM_TOL_DB, **kw):
    """em_refine on two copies of one store: (reference history, log),
    (C++ history, log), and the two roots."""
    h = _hdr()
    key = te.key_of(h)
    _base_store(tmp_path / "base", passes, h)
    shutil.copytree(tmp_path / "base", tmp_path / "py")
    shutil.copytree(tmp_path / "base", tmp_path / "cpp")
    lr, lg = [], []
    t = time.perf_counter()
    hr = em.em_refine(S.Store(tmp_path / "py", clock=tm.Clock()), key, rounds, tol_db, log=lr.append, **kw)
    t_ref = time.perf_counter() - t
    t = time.perf_counter()
    hg = Er.em_refine(NStore(tmp_path / "cpp", clock=tm.Clock()), key, rounds, tol_db, log=lg.append, **kw)
    t_cpp = time.perf_counter() - t
    _worst("time em_refine C++/Python", t_cpp / t_ref)
    return (hr, lr), (hg, lg), key


def _check_ab(ref, got, what):
    (hr, lr), (hg, lg) = ref, got
    assert len(hg) == len(hr), (what, hg, hr)
    assert all(type(x) is float for x in hg), what
    for g, r in zip(hg, hr):
        _abs(g, r, HIST_BAR, (what, hg, hr), "em_refine history (dB)")
    _logs_close(lg, lr, what)


@pytest.mark.parametrize("label,n", [("tiny-6", 2), ("tiny-12d", 3),
                                     pytest.param("short-6", 2, marks=pytest.mark.slow),
                                     pytest.param("short-12d", 3, marks=pytest.mark.slow)])
def test_em_refine_ab(Er, NStore, tmp_path, label, n):
    """em_refine on two copies of an n-pass store: the same rounds, history,
    passes re-received, log and stored passes."""
    passes = [_pass(label, i) for i in range(n)]
    ref, got, key = _ab(tmp_path, NStore, Er, passes, rounds=3)
    _check_ab(ref, got, (label, n))
    assert any(_LINE.match(x) for x in ref[1]), ref[1]
    _stores_close(tmp_path / "py", tmp_path / "cpp", NStore, key, (label, n))


@pytest.mark.parametrize("tol_db,rounds,kw", [(-math.inf, 2, {}), (math.inf, 3, {}),
                                              (em.EM_TOL_DB, 2, {"use_header": False}),
                                              (em.EM_TOL_DB, 1, {"estimator": "plain"}),
                                              (em.EM_TOL_DB, 0, {}), (em.EM_TOL_DB, -2, {})])
def test_em_refine_options(Er, NStore, tmp_path, tol_db, rounds, kw):
    """tol_db that never stops and one that stops after round 1, use_header
    off, an estimator, no rounds at all: the reference's outcomes."""
    passes = [_pass("tiny-6", i) for i in range(2)]
    ref, got, key = _ab(tmp_path, NStore, Er, passes, rounds=rounds, tol_db=tol_db, **kw)
    _check_ab(ref, got, (tol_db, rounds, kw))
    want = {(-math.inf, 2): 3, (math.inf, 3): 2}.get((tol_db, rounds))
    if want is not None:
        assert len(ref[0]) == want, ref[0]
    if rounds <= 0:
        assert len(ref[0]) == 1 and ref[1] == []
    _stores_close(tmp_path / "py", tmp_path / "cpp", NStore, key, (tol_db, rounds, kw))


def test_em_refine_refusals_and_skips(Er, NStore, tmp_path):
    """A key with no accumulator; members without a capture; an L member."""
    h = _hdr()
    key = te.key_of(h)
    a = te.air(frame.SHORT)
    synth = [make_pass_result(a, 0.5 + s, seed=s, hdr=h, hdr_es_n0_db=-16.0) for s in range(3)]
    pl = dataclasses.replace(make_pass_result(a, 1.0, seed=7, waveform=WAVEFORM_L, hdr=h, decoded=True),
                             ch=np.ones(16, np.complex64))
    for name, passes, lines in (("synthetic", synth, 0), ("with L", synth + [pl], 1)):
        root = tmp_path / name
        _base_store(root / "base", passes, h)
        shutil.copytree(root / "base", root / "py")
        shutil.copytree(root / "base", root / "cpp")
        lr, lg = [], []
        hr = em.em_refine(S.Store(root / "py", clock=tm.Clock()), key, 3, log=lr.append)
        hg = Er.em_refine(NStore(root / "cpp", clock=tm.Clock()), key, 3, log=lg.append)
        assert lg == lr and len(lr) == lines, (name, lr, lg)
        if lines:
            assert lr == [f"pass {pl.uid}: waveform L, not re-received (EM is CE-only)"]
        assert len(hg) == len(hr) == 2 and hr[0] == hr[1], (hr, hg)
        for g, r in zip(hg, hr):
            assert g == r or abs(g - r) <= 4 * math.ulp(r), (hg, hr)
        H.compare_trees(root / "py", root / "cpp")
    # no accumulator under the key
    bad = ("NOPE", 7)
    with pytest.raises(KeyError) as r:
        em.em_refine(S.Store(tmp_path / "synthetic" / "py"), bad)
    with pytest.raises(KeyError) as g:
        Er.em_refine(NStore(tmp_path / "synthetic" / "cpp"), bad)
    assert str(g.value) == str(r.value)
    # key[1] as the reference reads it: int(key[1])
    hr = em.em_refine(S.Store(tmp_path / "synthetic" / "py"), (key[0], str(key[1])), 1)
    hg = Er.em_refine(NStore(tmp_path / "synthetic" / "cpp"), [key[0], float(key[1])], 1)
    assert len(hr) == len(hg) == 2


# --- the seam (a spy on em.rereceive) -------------------------------------------------------------

def _sha(x):
    import hashlib

    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()[:16]


def _seam_run(tmp, passes, h):
    calls = []

    def rereceive(p, prior, hdr, cw, estimator=None, r=1, path=None, return_path=False):
        calls.append((p.uid, r, _sha(prior.x_hat), _sha(prior.v), repr(hdr),
                      None if cw is None else (_sha(cw[0]), cw[1]), estimator, path is None))
        new = dataclasses.replace(p, z=(p.z * 0.9 + 0.01 * r).astype(np.float32),
                                  w=(p.w * (1.0 + float(np.mean(prior.v)))).astype(np.float32), em_round=r)
        pth = em._flat(1500.0 + r)
        return (new, pth) if return_path else new

    st = S.Store(tmp, clock=tm.Clock())                       # whichever Store the module holds
    for p in passes:
        st.add_pass(p)
        st.attach(te.key_of(h), p.uid, 0, "test", mode=h.mode, codec_id=h.codec_id)
    em_rr = em.rereceive
    em.rereceive = rereceive
    try:
        lines = []
        hist = em.em_refine(st, te.key_of(h), rounds=3, log=lines.append)
    finally:
        em.rereceive = em_rr
    return hist, calls, lines


def test_em_refine_seam(native, Er, tmp_path, monkeypatch):
    """em_refine through the rows with em.rereceive replaced: the C++ calls
    the replacement as the reference does, with the same priors (bit for
    bit), headers, keying and paths, and stores what it returns."""
    from qrss_native import em as rows_em
    from qrss_native import em_refine as rows_er

    h = _hdr()
    a = te.air(frame.SHORT)
    ch = np.exp(1j * np.arange(16) / 3.0).astype(np.complex64)
    passes = [dataclasses.replace(make_pass_result(a, 0.5, seed=s, hdr=h, hdr_es_n0_db=-16.0), ch=ch)
              for s in range(3)]
    ref = _seam_run(tmp_path / "py", passes, h)
    with monkeypatch.context() as mp:
        H.apply_store_rows(native, mp)
        reference = qrss_native._import_reference()
        replaced = {}
        for rows in (rows_em.rows(native.qrss), rows_er.rows(native.qrss)):
            for k, repl in rows.items():
                owner = qrss_native._owner(k[0])
                orig = getattr(owner, k[1])
                mp.setattr(owner, k[1], repl)
                replaced[id(orig)] = (orig, repl)
        for m in reference:
            for k, v in list(vars(m).items()):
                hit = replaced.get(id(v))
                if hit is not None and hit[0] is v:
                    mp.setattr(m, k, hit[1])
        assert getattr(em.em_refine, "__qrss_native__", False)
        got = _seam_run(tmp_path / "cpp", passes, h)
    assert got[1] == ref[1] and len(ref[1]) >= 3, (got[1], ref[1])
    assert got[2] == ref[2]
    assert len(got[0]) == len(ref[0])
    for g, r in zip(got[0], ref[0]):
        assert g == r or abs(g - r) <= 4 * math.ulp(r), (got[0], ref[0])
    H.compare_trees(tmp_path / "py", tmp_path / "cpp")


# --- the hooks (the C++'s own) --------------------------------------------------------------------

def _hook_store(tmp_path, NStore, n=2):
    h = _hdr()
    passes = [_pass("tiny-6", i) for i in range(n)]
    _base_store(tmp_path / "s", passes, h)
    return NStore(tmp_path / "s", clock=tm.Clock()), te.key_of(h), [p.uid for p in passes]


def test_em_refine_progress(Er, NStore, tmp_path):
    """progress after each pass, total = rounds x eligible members; an early
    stop says so once."""
    st, key, _uids = _hook_store(tmp_path / "a", NStore)
    acc_uids = list(st.accumulator(key).uids)
    seen = []
    hist = Er.em_refine(st, key, 2, -math.inf, progress=lambda *a: seen.append(a))
    assert len(hist) == 3
    want = [(i + 1, 4, f"round {r} pass {u}") for i, (r, u) in enumerate((r, u) for r in (1, 2) for u in acc_uids)]
    assert seen == want, seen
    st, key, _uids = _hook_store(tmp_path / "b", NStore)
    acc_uids = list(st.accumulator(key).uids)
    seen = []
    hist = Er.em_refine(st, key, 3, math.inf, progress=lambda *a: seen.append(a))
    assert len(hist) == 2
    assert seen == [(1, 6, f"round 1 pass {acc_uids[0]}"), (2, 6, f"round 1 pass {acc_uids[1]}"),
                    (2, 2, "converged after round 1")], seen
    # no eligible member, or no rounds: no call at all
    seen = []
    assert len(Er.em_refine(st, key, 0, progress=lambda *a: seen.append(a))) == 1 and seen == []


def test_em_refine_cancel(native, Er, NStore, tmp_path):
    """A cancel set from the progress hook stops the loop at the next poll:
    em.Cancelled, the first pass stored (em_round 1), the other untouched,
    and every file of the store loads."""
    st, key, _uids = _hook_store(tmp_path, NStore, n=3)
    acc_uids = list(st.accumulator(key).uids)
    before = {u: st.load_pass(u) for u in acc_uids}
    flag = native.qrss.em.CancelFlag()
    lines = []

    def progress(done, total, what):
        flag.set()

    with pytest.raises(native.qrss.em.Cancelled):
        Er.em_refine(st, key, 3, progress=progress, cancel=flag, log=lines.append)
    assert len(lines) == 1 and lines[0].startswith(f"round 1 pass {acc_uids[0]}:"), lines
    for root_store in (S.Store(tmp_path / "s"), NStore(tmp_path / "s")):
        acc = root_store.accumulator(key)
        assert list(acc.uids) == acc_uids
        for u in acc_uids:
            p = root_store.load_pass(u)
            assert p.em_round == (1 if u == acc_uids[0] else 0), (u, p.em_round)
            if u != acc_uids[0]:
                assert np.array_equal(p.z, before[u].z) and np.array_equal(p.w, before[u].w)
    # set before the call: nothing at all
    flag2 = native.qrss.em.CancelFlag()
    flag2.set()
    with pytest.raises(native.qrss.em.Cancelled):
        Er.em_refine(st, key, 3, cancel=flag2)
    assert st.load_pass(acc_uids[1]).em_round == 0


def test_rows_signatures(row):
    """The rows take the reference's arguments, defaults included."""
    import inspect

    def params(fn):
        return [(p.name, p.kind, p.default) for p in inspect.signature(fn).parameters.values()]

    for name in ("pass_path", "rereceive", "em_refine"):
        assert params(row[name]) == params(getattr(em, name)), name

"""Parity of native/core/qrss/ltrack (sstvae_native.qrss.ltrack) with the
waveform L tracker of sstvae/qrss/ltrack.py, held side by side in one
process.

The inputs are the Python receiver's own: an l-tiny pass at -12 dB and an
l-short pass at -6 dB through `receiver.receive_pass`, with spies on
`ltrack.track_l` and `ltrack.refine_timing_l` recording every call (gate
V's withheld runs, round A, round B with the header and callsign known,
and the refinements inside them) and what each returned. The rows are
then fed exactly those arguments, and an l-tiny pass at -6 dB with a 9 s
lead-in (the one configuration reaching _lead_supported_l) is received
on the rows. The slow tier adds the same at -12 dB with +1 Hz/min drift
and +100 ppm on the receiver's clock (l-short) and at -6 dB with
-1.5 Hz/min and -100 ppm on the sender's (l-tiny), and l-medium's
quadratic-timing branch.

The contract, and what was measured (2026-10-13, these passes):

- bit for bit (dtype and shape included): pilot_table and _pilot_taps,
  data_mix, data_mf (dense and sparse), make_classes_l, _distortion,
  data_templates, _pilot_gain, pilot_wave, _q_near, known_symbols_l,
  _Model's arrays and its streams / fused / pu_initial, _fit_delta,
  _hold_ends;
- pilot_filter (scipy's oaconvolve through ducc0 against pocketfft) to
  1e-12 of its output's scale, _pu_local (np.convolve's BLAS sum) 1e-13;
- track_l on the receivers' calls: the same q, keying, lead-in, grid and
  measurement masks, every decision alike; z_d, m, c, N, psi, u, P, pu and
  the pilot stream's arrays to 1e-9 relative (measured 4.4e-10 worst, at
  the frame's end where the timing's rounding reaches furthest; P 1.4e-12),
  delta 1e-9, the timing to 1e-6 CH samples (measured 1.6e-12), the path
  to 1e-9 Hz. With the reference's own pilot_filter, _smooth, _recenter,
  freq_path_spline and refine_timing_l swapped for the rows (the leaves
  that are to rounding), the reference's track_l is the C++ one bit for
  bit, every field of every call;
- refine_timing_l: the same None outcomes, the timing to 1e-6 CH samples
  (measured 1.1e-16), z to 1e-9; report_l's fields to 1e-9 relative, or
  absolute under 1 (the drift and the wander are differences of a 1500 Hz
  path: measured 9e-13 Hz);
- receive_pass with the rows over ltrack's names, end to end: the same
  passes and headers, f_hz to 1e-6 Hz, latents to 1e-6, with every row
  reached (counting spies installed before the rows), in three
  configurations: every row; every row but track_l (the reference's
  tracker run over the rows' pieces); the leaves alone;
- the reference's refusals, type and text; and numpy's own semantics at
  the edges: a numpy scalar for a 0-d tau, index or tp, an index's or a
  position array's own shape, a boolean mask, a CE frame (the
  AttributeError / AssertionError lframe.layout raises on one).

tests/qrss_native/ltrack.py substitutes these under --native-qrss=rx.
"""

import copy
import dataclasses
import math
from types import SimpleNamespace

import numpy as np
import pytest

from qrss_demod_tracks import flat_det, rel
from qrss_fakes import L_HDR, l_scenario
from qrss_helpers import Q_TEST
from qrss_native import require
from sstvae.qrss import frame, frontend, header, lframe
from sstvae.qrss import ltrack as LT
from sstvae.qrss import receiver as RX
from sstvae.qrss import track as T
from sstvae.qrss.constants import CH_FS, L_T_SYM, SPAN, WAVEFORM_L
from sstvae.qrss.types import EmPrior, Timing

LTRACK = "sstvae.qrss.ltrack"
ARRAYS = ("z_d", "m", "c", "N", "psi", "u", "P", "pu", "m_pl", "c_pl", "N_pl", "psi_pl", "u_pl", "P_pl")
MASKS = ("pos", "meas", "meas_pl")


@pytest.fixture(scope="module")
def Ln(native):
    return require(native, "ltrack")


@pytest.fixture(scope="module")
def row(native, Ln):
    from qrss_native.ltrack import rows

    return {name: fn for (_t, name), fn in rows(native.qrss).items()}


@pytest.fixture(scope="module")
def leaves(native, Ln):
    """The rows the swap test puts under the reference's track_l: ltrack's
    pilot_filter and refine_timing_l, track's _smooth and _recenter and
    trackbase's freq_path_spline."""
    from qrss_native.track import rows as trows
    from qrss_native.trackbase import rows as tbrows

    t = {name: fn for (_m, name), fn in trows(native.qrss).items()}
    tb = {name: fn for (_m, name), fn in tbrows(native.qrss).items()}
    return t, tb


def _record(name, snr, cfg=(), lead_in=0.0):
    """The Python receiver on one L pass, every track_l and refine_timing_l
    call recorded (arguments and a copy of what was passed and returned)."""
    spec = lframe.get(name)
    _, sim = l_scenario(name, snr, 1, lead_in=lead_in, cfg=cfg)
    prep = RX.prepare(frontend.Capture(sim.fe, Q_TEST, sim.t0_index))
    det = dataclasses.replace(flat_det(1500.0, WAVEFORM_L), lead_in_s=lead_in)
    calls, rcalls = [], []
    orig, orig_r = LT.track_l, LT.refine_timing_l

    def spy(*a, **k):
        tr = orig(*a, **k)
        calls.append((a, dict(k), copy.copy(tr)))
        return tr

    def spy_r(tr, *a, **k):
        snap = copy.copy(tr)
        out = orig_r(tr, *a, **k)
        rcalls.append((snap, a, dict(k), out))
        return out

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(LT, "track_l", spy)
        mp.setattr(LT, "refine_timing_l", spy_r)
        res = RX.receive_pass(prep, spec, det, return_tracks=True)
    assert res is not None, name
    return SimpleNamespace(spec=spec, sim=sim, prep=prep, det=det, pr=res[0], tracks=res[1],
                           calls=calls, rcalls=rcalls)


@pytest.fixture(scope="module")
def passes(Ln):
    return {"l-tiny": _record("l-tiny", -12.0), "l-short": _record("l-short", -6.0)}


@pytest.fixture(scope="module")
def hard_passes(Ln):
    """The slow tier's: drift and a clock error each way."""
    return {"l-short drift +100 ppm rx": _record("l-short", -12.0,
                                                 (("drift_hz_per_min", 1.0), ("rx_ppm", 100.0))),
            "l-tiny drift -100 ppm tx": _record("l-tiny", -6.0,
                                                (("drift_hz_per_min", -1.5), ("tx_ppm", -100.0)))}


def _tracks(passes):
    """(label, LTrackResult) of every track_l call's result."""
    return [(f"{n}/{i}", c[2]) for n, p in passes.items() for i, c in enumerate(p.calls)]


# --- comparisons ------------------------------------------------------------------------------

def _same(a, b, what=""):
    if a is None or b is None:
        assert a is None and b is None, what
        return
    a, b = np.asarray(a), np.asarray(b)
    assert a.dtype == b.dtype, (what, a.dtype, b.dtype)
    assert a.shape == b.shape, (what, a.shape, b.shape)
    if a.dtype.kind in "fc":
        av = a.view(a.real.dtype) if a.dtype.kind == "c" else a
        bv = b.view(b.real.dtype) if b.dtype.kind == "c" else b
        an, bn = np.isnan(av), np.isnan(bv)
        assert np.array_equal(an, bn), what
        assert av[~an].tobytes() == bv[~bn].tobytes(), what
    else:
        assert np.array_equal(a, b), what


def _scaled(g, r):
    """max |g - r| over max |r|."""
    g, r = np.asarray(g), np.asarray(r)
    assert g.shape == r.shape and g.dtype == r.dtype
    return float(np.max(np.abs(g - r)) / max(float(np.max(np.abs(r))), 1e-300)) if r.size else 0.0


def _outcome(fn, *args, **kw):
    try:
        out = fn(*args, **kw)
    except (ValueError, TypeError, IndexError, ZeroDivisionError, OverflowError, AssertionError,
            AttributeError, np.linalg.LinAlgError) as e:
        return type(e), str(e)
    return "ok", out


def _same_error(r, g, what=""):
    assert r[0] != "ok" and g[:2] == r[:2], (what, r[:2], g[:2])


def _alike(r, g, what=""):
    """Both raised alike, or both returned equal values (arrays bit for bit)."""
    if r[0] != "ok":
        return _same_error(r, g, what)
    assert g[0] == "ok", (what, g)
    if isinstance(r[1], np.ndarray):
        _same(g[1], r[1], what)
    else:
        assert g[1] == r[1] and type(g[1]) is type(r[1]), (what, g[1], r[1])


def _slip(tm, spec):
    return tm.ppm * 1e-6 * spec.n_pos * L_T_SYM * CH_FS


def _timing_agrees(g, r, spec, what, tol=1e-6):
    assert (g is None) == (r is None), what
    if r is None:
        return
    assert abs(g.tau0 - r.tau0) <= tol, (what, g.tau0, r.tau0)
    assert abs(_slip(g, spec) - _slip(r, spec)) <= tol, (what, g.ppm, r.ppm)
    assert abs(g.gamma - r.gamma) <= tol, (what, g.gamma, r.gamma)
    assert abs(g.z - r.z) <= 1e-9 * max(abs(r.z), 1.0), (what, g.z, r.z)
    cg, cr = np.asarray(g.cov), np.asarray(r.cov)
    assert cg.shape == cr.shape == (3, 3), what
    assert np.array_equal(cg == 0, cr == 0), what
    assert np.max(np.abs(cg - cr)) <= 1e-6 * max(float(np.max(np.abs(cr))), 1e-300), what


def _track_agrees(g, r, what, worst=None):
    """g (the row's) against r (the reference's): every decision alike,
    the arrays to 1e-9 relative."""
    spec = r.spec
    assert (g.q, g.lead_in_s, g.ook) == (r.q, r.lead_in_s, r.ook), what
    _same(g.keying, r.keying, f"{what} keying")
    for f in MASKS:
        _same(getattr(g, f), getattr(r, f), f"{what} {f}")
    for f in ARRAYS:
        a, b = getattr(g, f), getattr(r, f)
        assert (a is None) == (b is None), (what, f)
        if b is None:
            continue
        e = rel(a, b)
        if worst is not None:
            worst[f] = max(worst.get(f, 0.0), e)
        assert e <= 1e-9, (what, f, e)
    assert abs(g.delta - r.delta) <= 1e-9 * abs(r.delta), (what, g.delta, r.delta)
    assert abs(g.loglik - r.loglik) <= 1e-9 * max(abs(r.loglik), 1.0), (what, g.loglik, r.loglik)
    assert abs(g.df_max - r.df_max) <= 1e-9, (what, g.df_max, r.df_max)
    _timing_agrees(g.timing, r.timing, spec, what)
    t = np.linspace(-20.0, spec.n_pos * L_T_SYM + 20.0, 41)
    assert np.max(np.abs(np.asarray(g.freq(t)) - np.asarray(r.freq(t)))) <= 1e-9, what
    assert g.spec is r.spec and g.chan is r.chan, what


def _same_track(g, r, what):
    """Bit for bit: every array field, the scalars and the timing."""
    for f in MASKS + ARRAYS:
        _same(getattr(g, f), getattr(r, f), f"{what} {f}")
    for f in ("q", "loglik", "df_max", "lead_in_s", "delta", "ook"):
        assert getattr(g, f) == getattr(r, f), (what, f, getattr(g, f), getattr(r, f))
    for f in ("tau0", "ppm", "gamma", "z"):
        assert getattr(g.timing, f) == getattr(r.timing, f), (what, f)
    _same(g.timing.cov, r.timing.cov, f"{what} cov")
    _same(g.keying, r.keying, f"{what} keying")


# --- tables and filters ---------------------------------------------------------------------

def test_tables(Ln, row):
    r, g = LT.pilot_table(), row["pilot_table"]()
    assert (g.t_sym, g.sps, g.half, g.g0) == (r.t_sym, r.sps, r.half, r.g0)
    _same(g.tab, r.tab, "tab")
    assert not g.tab.flags.writeable
    assert row["pilot_table"]() is g, "one table"
    _same(row["_pilot_taps"](), LT._pilot_taps(), "_pilot_taps")


def test_pilot_filter(Ln, row, passes):
    """To 1e-12 of the output's scale on what the tracker filters (z_d,
    and the keep as float64). A complex64 x is 1e-6: scipy transforms it
    in single precision, the C++ in double (measured 9.4e-8)."""
    worst = {}
    for label, tr in _tracks(passes):
        tp = tr.chan.t0_index + T.rx_index(tr.timing, tr.pos, tr.spec.n_pos, CH_FS, t_sym=L_T_SYM)
        for kind, x in (("z_d", tr.z_d), ("real", np.abs(tr.z_d)), ("complex64", tr.chan.ch)):
            r, g = LT.pilot_filter(x, tp), row["pilot_filter"](x, tp)
            assert g.dtype == r.dtype and g.shape == r.shape, label
            worst[kind] = max(worst.get(kind, 0.0), _scaled(g, r))
            assert np.array_equal(g == 0, r == 0), label
    assert worst["z_d"] <= 1e-12 and worst["real"] <= 1e-12 and worst["complex64"] <= 1e-6, worst
    tr = passes["l-tiny"].calls[-1][2]
    for tp in (3000.25, np.array([[10.5, -3.0], [1e6, 2.0]]), np.zeros(0), [5, 6]):
        r, g = LT.pilot_filter(tr.z_d, tp), row["pilot_filter"](tr.z_d, tp)
        assert g.shape == r.shape and g.dtype == r.dtype and type(g) is type(r), repr(tp)
        assert _scaled(g, r) <= 1e-12
    for x, tp in ((np.zeros(0, complex), [1.0]), (np.zeros((3, 3)), [1.0]), (np.zeros(0), []),
                  (5.0, [0.0])):
        r, g = _outcome(LT.pilot_filter, x, tp), _outcome(row["pilot_filter"], x, tp)
        if r[0] == "ok":
            assert g[0] == "ok" and np.asarray(g[1]).shape == np.asarray(r[1]).shape
        else:
            _same_error(r, g, repr(x))


@pytest.mark.filterwarnings("ignore:invalid value encountered:RuntimeWarning")  # the reference on a NaN position
def test_data_mix_and_mf(Ln, row, passes):
    rng = np.random.default_rng(3)
    for label, tr in _tracks(passes):
        n_pos = tr.spec.n_pos
        _same(row["data_mix"](tr.z_d, tr.chan, tr.timing, n_pos),
              LT.data_mix(tr.z_d, tr.chan, tr.timing, n_pos), label)
        idx = rng.integers(-len(tr.z_d), len(tr.z_d), 200)
        _same(row["data_mix"](tr.z_d, tr.chan, tr.timing, n_pos, idx),
              LT.data_mix(tr.z_d, tr.chan, tr.timing, n_pos, idx), label)
        moved = dataclasses.replace(tr.timing, tau0=tr.timing.tau0 + 0.37, ppm=tr.timing.ppm - 3.1)
        for pos in (tr.pos, tr.pos[::7] + 0.25, np.array([5.0, n_pos - 3.5]), np.zeros(0),
                    np.array([n_pos + 5000.0])):
            for tm in (tr.timing, moved):
                _same(row["data_mf"](tr.z_d, tr.chan, tm, pos, n_pos),
                      LT.data_mf(tr.z_d, tr.chan, tm, pos, n_pos), label)
    tr = passes["l-tiny"].calls[-1][2]
    for args in ((tr.z_d, tr.chan, tr.timing, tr.spec.n_pos, [len(tr.z_d)]),):
        _same_error(_outcome(LT.data_mix, *args), _outcome(row["data_mix"], *args))
    for pos in ([np.nan], [np.inf]):
        args = (tr.z_d, tr.chan, tr.timing, pos, tr.spec.n_pos)
        _same_error(_outcome(LT.data_mf, *args), _outcome(row["data_mf"], *args), repr(pos))
    # numpy's indexing: a scalar back for a 0-d index, the index's own shape,
    # a boolean mask (and what broadcasting it makes of tau), and the refusals
    n_z = len(tr.z_d)
    for idx in (np.int64(5), 7, np.array(-1), np.arange(12).reshape(3, 4), [3, 4], [],
                np.array([], dtype=np.int64), np.array([n_z - 1, -n_z]), np.array([1.0, 2.0]),
                np.zeros(n_z, bool), np.arange(n_z) % 3 == 0, np.ones(n_z, bool), np.array([True]),
                np.zeros(10, bool), np.zeros((n_z, 1), bool), np.array(True), np.array(False)):
        args = (tr.z_d, tr.chan, tr.timing, tr.spec.n_pos, idx)
        _alike(_outcome(LT.data_mix, *args), _outcome(row["data_mix"], *args), ("data_mix", np.shape(idx)))
    # positions of any shape: none (0-d), zeros where nothing reaches, or
    # the reference's own broadcasting and einsum refusals
    for pos in (np.float64(3.0), 3.0, [], [1, 2], np.zeros((0, 3)), np.full((3, 2), 1e7),
                np.arange(20.0).reshape(20, 1), np.arange(40.0).reshape(20, 2), np.arange(40.0).reshape(20, 2, 1),
                np.arange(6.0).reshape(2, 3, 1), np.arange(6.0).reshape(1, 2, 3), np.arange(-20, 140.0).reshape(10, 16),
                np.arange(4.0).reshape(2, 2), np.array([[0.0], [150.0]]), np.arange(-20, 140.0).reshape(1, 160),
                np.arange(5000.0).reshape(5000, 1)):
        args = (tr.z_d, tr.chan, tr.timing, pos, tr.spec.n_pos)
        _alike(_outcome(LT.data_mf, *args), _outcome(row["data_mf"], *args), ("data_mf", np.shape(pos)))


# --- classes, templates, the pilot's template -----------------------------------------------------

def _classes_same(g, r, what):
    for f in ("mu", "nu", "known", "gain"):
        _same(getattr(g, f), getattr(r, f), f"{what} {f}")
    _same(g.hdr_bits, r.hdr_bits, f"{what} hdr_bits")
    assert (g.hdr_known, g.spread, g.spread_rho) == (r.hdr_known, r.spread, r.spread_rho), what
    assert type(g) is type(r), what


def test_classes_templates_known(Ln, row, passes):
    rng = np.random.default_rng(1)
    bits = header.encode(L_HDR)
    for name in ("l-tiny", "l-short", "l-medium", "l-full"):
        spec = lframe.get(name)
        prior = EmPrior(x_hat=rng.standard_normal(spec.n_data) * 0.5,
                        v=rng.uniform(-0.2, 1.3, spec.n_data))
        for hb, pr in ((None, None), (bits, None), (bits, prior), (None, prior)):
            r, g = LT.make_classes_l(spec, hb, pr), row["make_classes_l"](spec, hb, pr)
            _classes_same(g, r, name)
            for wk in (False, True):
                grid = np.arange(-SPAN - 7, spec.n_pos)
                for x, y in zip(row["data_templates"](spec, grid, r, wk), LT.data_templates(spec, grid, r, wk)):
                    _same(x, y, f"{name} templates")
        _same(row["_distortion"](spec), LT._distortion(spec), name)
        for hb in (None, bits):
            kr, kg = LT.known_symbols_l(spec, hb), row["known_symbols_l"](spec, hb)
            _same(kg.pos, kr.pos, name)
            _same(kg.val, kr.val, name)
    # the receivers' own classes, on their grids
    for label, tr in _tracks(passes):
        for wk in (False, True):
            for x, y in zip(row["data_templates"](tr.spec, tr.pos, tr.classes, wk),
                            LT.data_templates(tr.spec, tr.pos, tr.classes, wk)):
                _same(x, y, label)
    # the bits object stays the caller's array, as np.asarray keeps it
    g = row["make_classes_l"](lframe.L_SHORT, bits)
    assert g.hdr_bits is LT.make_classes_l(lframe.L_SHORT, bits).hdr_bits or \
        np.array_equal(g.hdr_bits, bits)
    # refusals
    # a CE frame: _l_only's AssertionError, or what lframe.layout raises on one
    # (an AttributeError for TINY, the header count's AssertionError for FULL)
    cl = LT.make_classes_l(lframe.L_SHORT)
    for spec in (frame.TINY, frame.FULL):
        for fn, args in (("make_classes_l", ()), ("known_symbols_l", ()), ("_distortion", ()),
                         ("data_templates", (np.arange(-8, 50), cl))):
            _alike(_outcome(getattr(LT, fn), spec, *args), _outcome(row[fn], spec, *args), (spec.name, fn))
        for fn in ("_pilot_gain", "pilot_wave"):
            _alike(_outcome(getattr(LT, fn), np.arange(5.0), spec), _outcome(row[fn], np.arange(5.0), spec),
                   (spec.name, fn))
    for bad in (bits[:-1], list(bits) + [1, 0]):
        _same_error(_outcome(LT.make_classes_l, lframe.L_SHORT, bad),
                    _outcome(row["make_classes_l"], lframe.L_SHORT, bad), "bad bits")
    _same_error(_outcome(LT.data_templates, lframe.L_SHORT, np.zeros(0, np.int64), g),
                _outcome(row["data_templates"], lframe.L_SHORT, np.zeros(0, np.int64), g))
    for spec in (frame.TINY, frame.FULL, lframe.L_SHORT, lframe.L_FULL):
        _alike(_outcome(LT._l_only, spec, "x.y"), _outcome(row["_l_only"], spec, "x.y"), spec.name)


def test_pilot_wave(Ln, row, passes):
    keyings = [None] + [c[2].keying for p in passes.values() for c in p.calls if c[2].keying is not None]
    assert len(keyings) > 1, "a round B with the callsign keying"
    for name in ("l-tiny", "l-short", "l-full"):
        spec = lframe.get(name)
        tau = np.concatenate([np.arange(-400.0, spec.n_pos + 50, 0.37), [np.nan, -1e9]])
        for lead in (0.0, 4.2, 10.0):
            _same(row["_pilot_gain"](tau, spec, lead), LT._pilot_gain(tau, spec, lead), name)
            for k in keyings if spec.n_win else [None]:
                _same(row["pilot_wave"](tau, spec, k, lead), LT.pilot_wave(tau, spec, k, lead), name)
    spec = lframe.L_SHORT
    for tau in (3.5, np.float64(-2.0), np.ones((2, 3)), np.zeros(0), [1, 2]):
        for lead in (0.0, 11.0, -1.0):
            for fn in ("_pilot_gain", "pilot_wave"):
                _alike(_outcome(getattr(LT, fn), tau, spec, *((None, lead) if fn == "pilot_wave" else (lead,))),
                       _outcome(row[fn], tau, spec, *((None, lead) if fn == "pilot_wave" else (lead,))),
                       (fn, repr(tau), lead))


def test_q_near(Ln, row):
    for t_sym in (L_T_SYM, T.T_SYM):
        qg = T.q_grid(t_sym)
        qs = list(qg) + [q * f for q in qg for f in (0.7, 1.3, 1.0 + 1e-15)] + [1e-300, 1e300, math.inf]
        for q in qs:
            r, g = LT._q_near(q, qg), row["_q_near"](q, qg)
            assert g == r and all(a is b for a, b in zip(g, r)), q
        for q in (0.0, -1.0):
            _same_error(_outcome(LT._q_near, q, qg), _outcome(row["_q_near"], q, qg))
    _same_error(_outcome(LT._q_near, 1e-4, ()), _outcome(row["_q_near"], 1e-4, ()))


# --- the model and the pieces between measurements ------------------------------------------------

def _model_args(call):
    """_Model's constructor arguments as track_l built them for one call."""
    a, k, tr = call
    spec = a[1]
    lead = tr.lead_in_s
    n_lead = int(math.floor(min(lead, LT.LEAD_KEEP_S) / L_T_SYM))
    return spec, tr.classes, bool(k.get("withhold_known", False)), tr.keying, lead, n_lead


def test_model(Ln, row, passes):
    for label, call in [(f"{n}/{i}", c) for n, p in passes.items() for i, c in enumerate(p.calls)]:
        spec, cl, wk, key, lead, n_lead = _model_args(call)
        tr_r, tr_g = copy.copy(call[2]), copy.copy(call[2])
        mr = LT._Model(tr_r, spec, cl, wk, key, lead, n_lead)
        mg = row["_Model"](tr_g, spec, cl, wk, key, lead, n_lead)
        for f in ("c_all", "c", "nu", "gain", "dist", "k_pl", "is_sym", "is_known", "meas_d", "need",
                  "meas_p", "measured"):
            _same(getattr(mg, f), getattr(mr, f), f"{label} {f}")
        _same(tr_g.meas, tr_r.meas, label)
        _same(tr_g.meas_pl, tr_r.meas_pl, label)
        pu = call[2].pu
        mask = (np.arange(len(pu)) % 3) != 0
        for kw in ({}, dict(delta=call[2].delta), dict(data=False), dict(pilot=False),
                   dict(delta=0.5 - 0.25j, data_mask=mask)):
            for (x, y), (xr, yr) in zip(mg.streams(pu, **kw), mr.streams(pu, **kw)):
                _same(x, xr, f"{label} streams {kw}")
                _same(y, yr, f"{label} streams {kw}")
            for x, y in zip(mg.fused(pu, **kw), mr.fused(pu, **kw)):
                _same(x, y, f"{label} fused {kw}")
        assert mg.pu_initial() == mr.pu_initial(), label
        # the methods read the model's arrays (and its track) when called
        for md in (mg, mr):
            md.meas_d = md.meas_d & mask
            md.tr.N = md.tr.N * 2.0
        for x, y in zip(mg.fused(pu), mr.fused(pu)):
            _same(x, y, f"{label} edited")
        assert mg.pu_initial() == mr.pu_initial(), label
        # _fit_delta, on either model: bit for bit
        v, Pv = call[2].u_pl, call[2].P_pl
        if v is None:
            v, Pv = call[2].u, call[2].P
        for d0 in (1.0 + 0.0j, call[2].delta, -0.3 + 2.0j):
            r = LT._fit_delta(mr, v, Pv, pu, d0)
            for md in (mg, mr):
                g = row["_fit_delta"](md, v, Pv, pu, d0)
                assert g == r and type(g) is complex, (label, g, r)
        _alike(_outcome(LT._fit_delta, mr, v, Pv, pu, 0), _outcome(row["_fit_delta"], mg, v, Pv, pu, 0),
               f"{label} delta0 0")


def test_pu_local_and_hold_ends(Ln, row, passes):
    for label, tr in _tracks(passes):
        if tr.u is None:
            continue
        meas = (tr.meas | tr.meas_pl) if tr.meas_pl is not None else tr.meas
        for m in (meas, np.zeros(len(meas), bool)):
            (lg, gg), (lr, gr) = row["_pu_local"](tr.u, tr.P, m), LT._pu_local(tr.u, tr.P, m)
            assert lg.dtype == lr.dtype and lg.shape == lr.shape
            assert rel(lg, lr) <= 1e-13 and abs(gg - gr) <= 1e-13 * gr, label
        R = np.where((np.arange(len(tr.u)) % 13 == 0) | (np.arange(len(tr.u)) < 40), np.inf, 1.0)
        R[-30:] = np.inf
        for args in ((tr.u, tr.P, R), (tr.u.real.copy(), tr.P, R), (tr.u, tr.P, np.full(len(R), np.inf))):
            g, r = row["_hold_ends"](*args), LT._hold_ends(*args)
            _same(g[0], r[0], label)
            _same(g[1], r[1], label)
        g = row["_hold_ends"](tr.u, tr.P, np.full(len(R), np.inf))
        assert g[0] is tr.u and g[1] is tr.P, "nothing measured: the inputs back"


def test_lead_supported(Ln, row, passes):
    for name, p in passes.items():
        tr = p.calls[-1][2]
        for lead in (0.0, 0.5, 1.3, 3.0, 9.0, 12.0):
            for tm in (tr.timing, dataclasses.replace(tr.timing, tau0=tr.timing.tau0 + 40)):
                _alike(_outcome(LT._lead_supported_l, tr.chan, tr.z_d, tm, p.spec, lead),
                       _outcome(row["_lead_supported_l"], tr.chan, tr.z_d, tm, p.spec, lead), (name, lead))


# --- the tracker --------------------------------------------------------------------------------

def test_track_l_on_the_receivers_calls(Ln, row, passes):
    worst = {}
    for name, p in passes.items():
        assert len(p.calls) >= 2, name
        for i, (a, k, ref) in enumerate(p.calls):
            g = row["track_l"](*a, **k)
            _track_agrees(g, ref, f"{name}/{i} {sorted(k)}", worst)
            assert g.classes is (a[3] if len(a) > 3 and a[3] is not None else g.classes)
            if "timing" in k and k["timing"] is not None and ref.timing is k["timing"]:
                assert g.timing is k["timing"]
    assert all(v <= 1e-9 for v in worst.values()), worst


def test_track_l_is_the_reference_up_to_its_leaves(Ln, row, leaves, passes):
    """The reference's track_l with the rows' pilot_filter, refine_timing_l,
    _smooth, _recenter and freq_path_spline is the C++ one bit for bit."""
    t, tb = leaves
    for name, p in passes.items():
        for i, (a, k, _ref) in enumerate(p.calls):
            g = row["track_l"](*a, **k)
            with pytest.MonkeyPatch.context() as mp:
                mp.setattr(LT, "pilot_filter", row["pilot_filter"])
                mp.setattr(LT, "refine_timing_l", row["refine_timing_l"])
                mp.setattr(LT, "_smooth", t["_smooth"])
                mp.setattr(LT, "freq_path_spline", tb["freq_path_spline"])
                mp.setattr(T, "_recenter", t["_recenter"])
                h = LT.track_l(*a, **k)
            _same_track(g, h, f"{name}/{i}")


def test_track_l_arguments(Ln, row, passes):
    p = passes["l-tiny"]
    (a, k, ref) = p.calls[-1]
    chan, spec, det = a[0], a[1], a[2]
    tm = ref.timing
    det_t = dataclasses.replace(det, timing=tm)
    pr = EmPrior(x_hat=np.linspace(-0.5, 0.5, spec.n_data), v=np.full(spec.n_data, 0.6))
    z_d = T.derotate(chan, ref.freq)
    cases = [
        ("defaults", (chan, spec, det_t), {}),
        ("prior", (chan, spec, det_t, None, None, pr), {}),
        ("timing kw", (chan, spec, det), dict(timing=tm)),
        ("freq and z_d", (chan, spec, det_t), dict(freq=ref.freq, z_d=z_d)),
        ("no refine", (chan, spec, det_t), dict(refine=False, outer=0)),
        ("outer 3", (chan, spec, det_t), dict(outer=3)),
        ("lead-in claimed", (chan, spec, det_t), dict(lead_in_s=3.0)),
        ("lead-in clamped", (chan, spec, det_t), dict(lead_in_s=-2.0)),
        ("cw_known (None, False)", (chan, spec, det_t, None, (None, False)), {}),
        ("withheld", (chan, spec, det_t), dict(withhold_known=True)),
    ]
    sp = passes["l-short"]
    b = [c for c in sp.calls if c[2].keying is not None]
    if b:
        a2, k2, r2 = b[-1]
        cases.append(("keyed, no refine", a2, dict(k2, refine=False)))
    for what, args, kw in cases:
        r = LT.track_l(*args, **kw)
        g = row["track_l"](*args, **kw)
        _track_agrees(g, r, what)
        if "z_d" in kw:
            assert g.z_d is kw["z_d"] or not np.array_equal(r.z_d, kw["z_d"])
    # refusals
    no_tm = dataclasses.replace(det, timing=None)
    for args, kw in (((chan, spec, no_tm), {}),
                     ((chan, spec, det_t, None, (None, True)), {}),
                     ((chan, frame.TINY, det_t), {}),
                     ((chan, spec, det_t, None, (np.ones(5), False)), {})):
        _same_error(_outcome(LT.track_l, *args, **kw), _outcome(row["track_l"], *args, **kw), repr(kw))


def test_refine_timing_l(Ln, row, passes):
    n_some = 0
    for name, p in passes.items():
        for i, (tr, a, k, out) in enumerate(p.rcalls):
            g = row["refine_timing_l"](tr, *a, **k)
            _timing_agrees(g, out, p.spec, f"{name}/{i}")
            n_some += out is not None
        for j, (_a, _k, ref) in enumerate(p.calls):
            for kw in ({}, dict(iters=1), dict(u_ref=ref.u * 0.9, P_ref=ref.P * 2.0), dict(iters=0)):
                r, g = _outcome(LT.refine_timing_l, ref, **kw), _outcome(row["refine_timing_l"], ref, **kw)
                if r[0] != "ok":
                    _same_error(r, g)
                    continue
                assert g[0] == "ok"
                _timing_agrees(g[1], r[1], p.spec, f"{name} call {j} {sorted(kw)}")
    assert n_some > 0
    # off the peak: the gradient branch (an indefinite Hessian) and its None
    tr = passes["l-short"].calls[-1][2]
    for dtau, dslip in ((0.0, 4.0), (10.0, 0.0), (7.0, 0.0), (2.0, -6.0), (300.0, 0.0)):
        t2 = copy.copy(tr)
        t2.timing = dataclasses.replace(tr.timing, tau0=tr.timing.tau0 + dtau,
                                        ppm=tr.timing.ppm + dslip / (tr.spec.n_pos * L_T_SYM * CH_FS) * 1e6)
        _timing_agrees(row["refine_timing_l"](t2), LT.refine_timing_l(t2), tr.spec, (dtau, dslip))
    # refusals and Nones the receiver does not reach
    for mutate in (lambda t: setattr(t, "u_pl", None), lambda t: setattr(t, "classes", None),
                   lambda t: setattr(t, "delta", 0j), lambda t: setattr(t, "N", None),
                   lambda t: setattr(t, "psi", np.zeros_like(t.psi)), lambda t: setattr(t, "spec", frame.TINY)):
        t2 = copy.copy(tr)
        mutate(t2)
        r, g = _outcome(LT.refine_timing_l, t2), _outcome(row["refine_timing_l"], t2)
        if r[0] != "ok":
            _same_error(r, g)
        else:
            assert g[0] == "ok"
            _timing_agrees(g[1], r[1], tr.spec, "mutated")


def test_report_l(Ln, row, passes):
    for label, tr in _tracks(passes):
        if tr.u is None:
            continue
        for kappa, suspect in ((1.0, False), (0.7, True)):
            r, g = LT.report_l(tr, kappa, suspect), row["report_l"](tr, kappa, suspect)
            assert type(g) is type(r)
            # relative, or absolute under 1: the drift and the wander are
            # differences of a ~1500 Hz path (measured 9e-13 Hz on the wander)
            for f in ("offset_hz", "drift_hz_per_min", "wander_hz_rms", "doppler_hz", "ppm", "snr2500_db",
                      "kappa"):
                x, y = getattr(g, f), getattr(r, f)
                assert abs(x - y) <= 1e-9 * max(abs(y), 1.0), (label, f, x, y)
            assert (g.slip_free, g.suspect) == (r.slip_free, r.suspect), label
            assert (math.isnan(g.z_ref) and math.isnan(r.z_ref)) or g.z_ref == r.z_ref, label


# --- the whole caller ---------------------------------------------------------------------------

def _counted(table, used):
    def wrap(name, fn):
        if isinstance(fn, type):
            class Counted(fn):
                def __init__(self, *a, **k):
                    used[name] = used.get(name, 0) + 1
                    super().__init__(*a, **k)
            return Counted

        def f(*a, **k):
            used[name] = used.get(name, 0) + 1
            return fn(*a, **k)
        return f
    return {name: wrap(name, fn) for name, fn in table.items()}


def _receive_agrees(g, r, what):
    assert g is not None, what
    assert g.uid.rsplit("_", 1)[0] == r.uid.rsplit("_", 1)[0], what
    assert (g.q, g.frame, g.waveform, g.estimator) == (r.q, r.frame, r.waveform, r.estimator), what
    assert g.header == r.header and g.cw == r.cw, what
    assert abs(g.f_hz - r.f_hz) <= 1e-6, what
    for f, tol in (("z", 1e-6), ("w", 1e-6), ("hdr_llr", 1e-4), ("psi", 1e-6)):
        a, b = np.asarray(getattr(g, f)), np.asarray(getattr(r, f))
        assert a.shape == b.shape and a.dtype == b.dtype, (what, f)
        assert a.size == 0 or np.max(np.abs(a.astype(np.float64) - b)) <= tol, (what, f)
    assert abs(g.timing.tau0 - r.timing.tau0) <= 1e-6 and abs(g.timing.ppm - r.timing.ppm) <= 1e-6, what
    for f in ("offset_hz", "doppler_hz", "snr2500_db", "z_ref", "kappa"):
        x, y = getattr(g.report, f), getattr(r.report, f)
        assert abs(x - y) <= 1e-6 * max(abs(y), 1.0), (what, f)
    assert (g.report.slip_free, g.report.suspect) == (r.report.slip_free, r.report.suspect), what


# which rows each configuration installs, and which of them it must reach
CONFIGS = {
    "every row": (None, {"track_l", "make_classes_l", "report_l", "known_symbols_l", "data_templates"}),
    "every row but track_l": ("track_l", {"_l_only", "_Model", "_fit_delta", "_pu_local", "_hold_ends",
                                          "_q_near", "refine_timing_l", "pilot_filter", "data_mf",
                                          "pilot_wave", "make_classes_l", "report_l"}),
    "the leaves": ("leaves", {"_pilot_taps", "data_mix", "_distortion", "_pilot_gain"}),
}
LEAVES = ("_pilot_taps", "data_mix", "_distortion", "_pilot_gain", "pilot_table")


@pytest.mark.parametrize("config", list(CONFIGS))
def test_receive_pass_on_the_rows(native, passes, monkeypatch, config):
    """receiver.receive_pass on l-tiny and l-short with the rows of
    tests/qrss_native/ltrack.py over ltrack's names (counting spies
    installed before them): the same passes and headers, f_hz to 1e-6 Hz,
    the latents to 1e-6, and every row the configuration is for reached."""
    from qrss_native.ltrack import rows

    table = {name: fn for (t, name), fn in rows(native.qrss).items() if t == LTRACK}
    assert len(table) == len(rows(native.qrss))
    leave_out, must = CONFIGS[config]
    if leave_out == "leaves":
        table = {n: f for n, f in table.items() if n in LEAVES}
    elif leave_out is not None:
        table.pop(leave_out)
    used: dict = {}
    for name, fn in _counted(table, used).items():
        monkeypatch.setattr(LT, name, fn)
    names = ("l-tiny",) if leave_out == "leaves" else tuple(passes)
    for name in names:
        p = passes[name]
        got = RX.receive_pass(p.prep, p.spec, p.det, return_tracks=True)
        _receive_agrees(got[0], p.pr, f"{config}: {name}")
        assert len(got[1]) == len(p.tracks), name
    assert must <= set(used), (config, sorted(must - set(used)))


# --- the slow tier --------------------------------------------------------------------------------

@pytest.mark.slow
def test_track_l_with_drift_and_clock_errors(Ln, row, leaves, hard_passes):
    """The receivers' calls on the two drifting passes, as on the fast ones;
    and the swap test there (measured: arrays 4.4e-10, P 1.4e-12, pu
    1.1e-11; bit for bit with the leaves swapped)."""
    t, tb = leaves
    worst = {}
    for name, p in hard_passes.items():
        for i, (a, k, ref) in enumerate(p.calls):
            g = row["track_l"](*a, **k)
            _track_agrees(g, ref, f"{name}/{i}", worst)
            with pytest.MonkeyPatch.context() as mp:
                mp.setattr(LT, "pilot_filter", row["pilot_filter"])
                mp.setattr(LT, "refine_timing_l", row["refine_timing_l"])
                mp.setattr(LT, "_smooth", t["_smooth"])
                mp.setattr(LT, "freq_path_spline", tb["freq_path_spline"])
                mp.setattr(T, "_recenter", t["_recenter"])
                h = LT.track_l(*a, **k)
            _same_track(g, h, f"{name}/{i} swapped")
        for i, (tr, a, k, out) in enumerate(p.rcalls):
            _timing_agrees(row["refine_timing_l"](tr, *a, **k), out, p.spec, f"{name} refine {i}")
        for label, tr in [(f"{name}/{i}", c[2]) for i, c in enumerate(p.calls) if c[2].u is not None]:
            r, g = LT.report_l(tr, 0.8, False), row["report_l"](tr, 0.8, False)
            assert abs(g.snr2500_db - r.snr2500_db) <= 1e-9 * abs(r.snr2500_db), label
    assert all(v <= 1e-9 for v in worst.values()), worst


def test_lead_in_pass(native, Ln, row, monkeypatch):
    """A 9 s lead-in claimed by the detection: _lead_supported_l keeps it,
    the grid starts before the frame, and the receiver on the rows is the
    reference's (the only configuration reaching _lead_supported_l)."""
    from qrss_native.ltrack import rows

    p = _record("l-tiny", -6.0, lead_in=9.0)
    assert any(c[2].lead_in_s > 0 for c in p.calls)
    for i, (a, k, ref) in enumerate(p.calls):
        _track_agrees(row["track_l"](*a, **k), ref, f"lead-in {i}")
    table = {name: fn for (_t, name), fn in rows(native.qrss).items() if name != "track_l"}
    used: dict = {}
    for name, fn in _counted(table, used).items():
        monkeypatch.setattr(LT, name, fn)
    got = RX.receive_pass(p.prep, p.spec, p.det, return_tracks=True)
    _receive_agrees(got[0], p.pr, "lead-in")
    assert "_lead_supported_l" in used


@pytest.mark.slow
def test_refine_timing_l_quadratic(Ln, row):
    """l-medium (over 303 s): the third dimension, gamma, kept where it
    raises the known symbols' Z by more than 1 (measured: tau0 within
    2.2e-16, gamma identical)."""
    spec = lframe.get("l-medium")
    _, sim = l_scenario("l-medium", -6.0, 1)
    prep = RX.prepare(frontend.Capture(sim.fe, Q_TEST, sim.t0_index))
    det = flat_det(1500.0, WAVEFORM_L)
    chan = RX.channel_for(prep, spec, det)
    g = LT.genie_track_l(chan, spec, sim.truth, 1500.0)
    tr = LT.track_l(chan, spec, dataclasses.replace(det, timing=g.timing), refine=False, outer=0)
    kept = 0
    for dg in (0.0, 4.0, 8.0, -6.0):
        t2 = copy.copy(tr)
        t2.timing = dataclasses.replace(tr.timing, gamma=tr.timing.gamma + dg)
        r, c = LT.refine_timing_l(t2), row["refine_timing_l"](t2)
        _timing_agrees(c, r, spec, dg)
        kept += r is not None and r.cov[2, 2] > 0
    assert kept > 0, "gamma's branch reached"

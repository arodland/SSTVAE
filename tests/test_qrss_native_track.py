"""Parity of native/core/qrss/track (sstvae_native.qrss.track) with the
tracker core of sstvae/qrss/track.py, held side by side in one process.

The inputs are the Python receiver's own: a CE TINY pass at -12 dB and a
SHORT pass at -6 dB through `receiver.receive_pass`, with a spy on
`track.track` recording every call it makes (gate V's withheld runs,
round A, round B with the header and callsign known) and what each
returned. The C++ `track` is then fed exactly those arguments.

The contract, and what was measured (2026-10-13, these passes):

- bit for bit (dtype and shape included): mf_table's taps, _pulse_at,
  _mf_taps, symbol_mf (complex and real z, CE / L / complex tables), psi_at,
  derotate (given the same path values), known_symbols, make_classes,
  remove_spread, _window_unknown, _kf_forward, kalman_rts, select_q (the
  same element of the grid), _measurements, _pu_initial, erase;
- templates: c to 1e-10 relative (measured 3e-16), sf to 1e-9 elementwise
  and 1e-14 of its scale (measured 1.9e-10 where v is small: numpy's SIMD
  exp of a real array is an ulp from libm's, and 1 - e^-v cancels);
  _lowpass (scipy's FFT convolution) to 1e-12 -- a complex64 input 1e-6,
  scipy transforming it in single precision where the C++ stays in double
  -- and _smooth (np.convolve through BLAS ddot) to 1e-13;
- track(): the same q, keying, lead-in and outcome of every decision; the
  path and timing to 1e-9 Hz / 1e-6 CH samples; z_d, m, c, N, psi to 1e-9
  relative (measured 2.5e-10 -- the re-centred spline, refitted by the
  C++). u, P and pu are what the reference's own last Kalman step makes of
  the C++ run's inputs, to the bit; against the Python run, measured u
  6.7e-10, P 8.4e-7 and pu 1.1e-8 relative, which is the reference's own
  conditioning: a 1-ulp change of Pu_g (np.convolve's BLAS sum against a
  sequential one) moves the reference's P by 8.4e-7 on the TINY gate-V run
  (the diffuse initial velocity covariance, 0.1 Pu, cancelled down to
  ~q Pu, keeps about seven digits). numpy's SIMD exp in the templates' sf
  seeds the same amplification: 2.0e-5 in P on a drifting lead-in pass's
  gate-V run. Swapping the C++ _smooth and templates into the reference
  makes its track the C++ one bit for bit, every field of every call;
- refine_timing: the same None outcomes, timing to 1e-6 CH samples
  (measured 2e-13); z_ref_stat to 1e-8; report's fields to 1e-9 relative,
  the drift as 1e-9 Hz/min absolute (the slope of a near-flat 1500 Hz
  line: measured 3e-12 Hz/min);
- receive_pass on the C++ rows, end to end: the same passes and headers,
  f_hz to 1e-6 Hz, latents to 1e-6;
- the reference's refusals, type and text, for an L frame and for bad
  arguments.

tests/qrss_native/track.py substitutes these under --native-qrss=rx.
"""

import copy
import dataclasses
import math
from types import SimpleNamespace

import numpy as np
import pytest

from qrss_demod_tracks import CALL, HDR, _ce_sim, flat_det, rel, ulps
from qrss_helpers import Q_TEST
from qrss_native import require
from sstvae.qrss import frame, frontend, header, lframe
from sstvae.qrss import ltrack as LT
from sstvae.qrss import receiver as RX
from sstvae.qrss import track as T
from sstvae.qrss.constants import CH_FS, L_T_SYM, SPAN, T_SYM
from sstvae.qrss.types import EmPrior, Timing

TRACK = "sstvae.qrss.track"


@pytest.fixture(scope="module")
def Tn(native):
    return require(native, "track")


@pytest.fixture(scope="module")
def row(native, Tn):
    from qrss_native.track import rows

    return {name: fn for (_t, name), fn in rows(native.qrss).items()}


@pytest.fixture(scope="module")
def passes(Tn):
    """{name: pass}: the Python receiver on CE TINY and SHORT, every
    track.track call recorded (arguments and a copy of the result taken
    as it returned)."""
    out = {}
    for name, snr in (("tiny", -12.0), ("short", -6.0)):
        spec, sim = _ce_sim(name, snr, 1)
        prep = RX.prepare(frontend.Capture(sim.fe, Q_TEST, sim.t0_index))
        calls = []
        orig = T.track

        def spy(*a, _orig=orig, _calls=calls, **k):
            tr = _orig(*a, **k)
            _calls.append((a, dict(k), copy.copy(tr)))
            return tr

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(T, "track", spy)
            res = RX.receive_pass(prep, spec, flat_det(1500.0), return_tracks=True)
        assert res is not None, name
        out[name] = SimpleNamespace(spec=spec, sim=sim, prep=prep, pr=res[0], tracks=res[1],
                                    calls=calls)
    return out


def _tracks(passes):
    """(label, TrackResult) of every finished track."""
    return [(f"{n}/{'AB'[i]}", tr) for n, p in passes.items() for i, tr in enumerate(p.tracks)]


# --- comparisons ------------------------------------------------------------------------------

def _same(a, b, what=""):
    a, b = np.asarray(a), np.asarray(b)
    assert a.dtype == b.dtype, (what, a.dtype, b.dtype)
    assert a.shape == b.shape, (what, a.shape, b.shape)
    if a.dtype.kind in "fc":
        av = a.view(a.real.dtype) if a.dtype.kind == "c" else a
        bv = b.view(b.real.dtype) if b.dtype.kind == "c" else b
        an, bn = np.isnan(av), np.isnan(bv)
        assert np.array_equal(an, bn), what
        assert av[~an].tobytes() == bv[~bn].tobytes(), (what, ulps(a, b))
    else:
        assert np.array_equal(a, b), what


def _outcome(fn, *args, **kw):
    try:
        out = fn(*args, **kw)
    except (ValueError, TypeError, IndexError, ZeroDivisionError, OverflowError, AssertionError,
            AttributeError, np.linalg.LinAlgError) as e:
        return type(e), str(e)
    return "ok", out


def _same_error(r, g, what=""):
    assert r[0] != "ok" and g[:2] == r[:2], (what, r[:2], g[:2])


def _same_outcome(r, g, what=""):
    """Both raised alike, or both returned equal records (a short header is not refused)."""
    if r[0] != "ok":
        return _same_error(r, g, what)
    assert g[0] == "ok", (what, g)
    for f in ("pos", "val", "known", "mu", "nu", "spread"):
        if hasattr(r[1], f):
            _same(getattr(g[1], f), getattr(r[1], f), f"{what} {f}")


def _timing(tau0=0.0, ppm=0.0, gamma=0.0):
    return Timing(tau0=tau0, ppm=ppm, gamma=gamma, z=0.0, cov=np.zeros((3, 3)))


# --- tables and the matched filter --------------------------------------------------------------

def test_tables(Tn, row):
    for t_sym in (T_SYM, L_T_SYM):
        r, g = T.mf_table(t_sym), Tn.mf_table(t_sym)
        assert (g["sps"], g["half"], g["g0"]) == (r.sps, r.half, r.g0)
        _same(g["tab"], r.tab, f"mf_table {t_sym}")
        assert not g["tab"].flags.writeable
        tb = row["mf_table"](t_sym)
        assert row["mf_table"](float(t_sym)) is tb and isinstance(tb, T.MfTable)
        _same(Tn._mf_taps(CH_FS, t_sym), T._mf_taps(CH_FS, t_sym), "mf_taps")
        _same(Tn._mf_taps(16000, t_sym), T._mf_taps(16000, t_sym), "mf_taps 16 kHz")
        assert Tn._pulse_g0(t_sym) == T._pulse_g0(t_sym)
        x = np.concatenate([np.linspace(-9.0, 9.0, 4000), [8.0, -8.0, 0.0, np.nan, np.inf]])
        _same(Tn._pulse_at(x, t_sym), T._pulse_at(x, t_sym), "pulse_at")
        _same(Tn._pulse_at(x.reshape(-1, 3), t_sym), T._pulse_at(x.reshape(-1, 3), t_sym), "2-D")
    assert row["mf_table"]() is row["mf_table"](T_SYM)
    for bad in (0.05, 1.0):
        _same_error(_outcome(T.mf_table, bad), _outcome(Tn.mf_table, bad), f"mf_table {bad}")
        _same_error(_outcome(T._pulse_at, 0.5, bad), _outcome(Tn._pulse_at, 0.5, bad), "pulse_at")


def test_symbol_mf_and_psi(Tn, row, passes):
    for label, tr in _tracks(passes):
        n_pos, t0 = tr.spec.n_pos, tr.chan.t0_index
        pos = tr.pos.astype(np.float64)
        kw = dict(t0_index=t0, n_pos=n_pos)
        _same(Tn.symbol_mf(tr.z_d, CH_FS, tr.timing, pos, **kw),
              T.symbol_mf(tr.z_d, CH_FS, tr.timing, pos, **kw), f"{label} symbol_mf")
        keep = tr.chan.keep if tr.chan.keep is not None else np.ones(len(tr.chan.ch), np.float32)
        _same(Tn.symbol_mf(keep, CH_FS, tr.timing, pos, **kw),
              T.symbol_mf(keep, CH_FS, tr.timing, pos, **kw), f"{label} real z")
        _same(Tn.psi_at(keep, tr.timing, pos, **kw), T.psi_at(keep, tr.timing, pos, **kw), "psi")
        _same(Tn.psi_at(None, tr.timing, pos, **kw), T.psi_at(None, tr.timing, pos, **kw), "psi 1")
    # off the grid: fractional, far outside, NaN; the L table, a complex
    # table and the row's own table objects (used without a copy)
    tr = passes["tiny"].tracks[0]
    z = tr.z_d
    pos = np.concatenate([np.linspace(-40.0, 1800.0, 997), [1e6, -1e6, np.nan, np.inf]])
    tm = Timing(tau0=3.25, ppm=-61.0, gamma=7.5, z=0.0, cov=np.zeros((3, 3)))
    kw = dict(t0_index=tr.chan.t0_index, n_pos=tr.spec.n_pos)
    lt = T.mf_table(L_T_SYM)
    tables = [None, T.mf_table(), lt, row["mf_table"](L_T_SYM),
              T.MfTable(t_sym=lt.t_sym, sps=lt.sps, half=lt.half, tab=(1 + 2j) * lt.tab, g0=lt.g0)]
    for k, tb in enumerate(tables):
        for zz in (z, z.real.copy(), z.astype(np.complex64)):
            _same(Tn.symbol_mf(zz, CH_FS, tm, pos, table=tb, **kw),
                  T.symbol_mf(zz, CH_FS, tm, pos, table=tb, **kw), f"table {k} {zz.dtype}")
    _same(Tn.psi_at(tr.chan.keep, tm, pos, table=lt, **kw), T.psi_at(tr.chan.keep, tm, pos, table=lt, **kw))
    # refusals: CH_FS, an empty capture
    _same_error(_outcome(T.symbol_mf, z, 4000, tm, pos, **kw), _outcome(Tn.symbol_mf, z, 4000, tm, pos, **kw))
    _same_error(_outcome(T.symbol_mf, z[:0], CH_FS, tm, pos[:3], **kw),
                _outcome(Tn.symbol_mf, z[:0], CH_FS, tm, pos[:3], **kw), "empty z")
    _same(Tn.symbol_mf(z, CH_FS, tm, pos[:0], **kw), T.symbol_mf(z, CH_FS, tm, pos[:0], **kw), "no positions")


def test_derotate(Tn, passes):
    for name, p in passes.items():
        tr = p.tracks[-1]
        chan = tr.chan
        ref_fn = tr.freq                          # the reference's PathFn
        lam = lambda t, f=ref_fn: f(t)            # noqa: E731  (Python evaluates the path)
        _same(Tn.derotate(chan, lam), T.derotate(chan, lam), f"{name} callable")
        from qrss_native import _rx
        nat = _rx.NativePathFn(np.linspace(-20, 300, 40), 1500.0 + 0.01 * np.sin(np.arange(40)),
                               np.linspace(1, 2, 40))
        _same(Tn.derotate(chan, nat), T.derotate(chan, nat), f"{name} native path")
        # the reference's spline, evaluated by the C++: to rounding
        a, b = Tn.derotate(chan, ref_fn), T.derotate(chan, ref_fn)
        assert np.max(np.abs(a - b)) <= 1e-9 * np.max(np.abs(b)), name
        ones = dataclasses.replace(chan, ch=np.ones(len(chan.ch), dtype=np.complex128) * (1 + 1e-12j))
        _same(Tn.derotate(ones, lam), T.derotate(ones, lam), f"{name} complex128 capture")
    empty = dataclasses.replace(chan, ch=np.zeros(0, np.complex64))
    _same(Tn.derotate(empty, lam), T.derotate(empty, lam), "empty")
    # a result of another length is not broadcast by the reference (np.interp refuses it)
    const = lambda t: 1500.25                                    # noqa: E731
    two = lambda t: np.array([1500.25, 1500.0])                  # noqa: E731
    for f, what in ((const, "a constant"), (lambda t: np.array([1500.25]), "(1,)"), (two, "(2,)")):
        _same_error(_outcome(T.derotate, chan, f), _outcome(Tn.derotate, chan, f), what)
    _same(Tn.derotate(empty, const), T.derotate(empty, const), "a constant, empty")
    _same_error(_outcome(T.derotate, empty, two), _outcome(Tn.derotate, empty, two), "(2,), empty")
    one = dataclasses.replace(chan, ch=chan.ch[:1])
    _same(Tn.derotate(one, const), T.derotate(one, const), "a constant, one sample")


def test_lowpass_and_smooth(Tn, passes):
    z = passes["tiny"].tracks[0].z_d
    worst, single = 0.0, 0.0
    for hz in (0.03, 0.3, 1.5, 3.0):
        for x in (z, z.real, z.astype(np.complex64)):
            r, g = T._lowpass(x, hz), Tn._lowpass(x, hz)
            assert g.dtype == r.dtype and g.shape == r.shape
            e = float(np.max(np.abs(g - r)) / np.max(np.abs(r)))
            if x.dtype == np.complex64:
                single = max(single, e)
            else:
                worst = max(worst, e)
    assert worst <= 1e-12, worst
    # scipy transforms a complex64 input in single precision (the C++ in double): float32 noise
    assert single <= 1e-6, single
    _same_error(_outcome(T._lowpass, z, 0.0), _outcome(Tn._lowpass, z, 0.0), "hz 0")
    x = np.abs(z[:5000]) ** 2
    for n in (0, 1, 2, 3, 32, 99, 4999, 5000, 7000):
        r, g = T._smooth(x, n), Tn._smooth(x, n)
        assert g.dtype == r.dtype and g.shape == r.shape, n
        assert np.max(np.abs(g - r)) <= 1e-13 * np.max(np.abs(r)), n
    _same(Tn._smooth(x, 1), x, "n 1")
    _same_error(_outcome(T._smooth, x[:0], 4), _outcome(Tn._smooth, x[:0], 4), "empty")


# --- known symbols, classes, the spread header, templates ----------------------------------------

def _classes_same(g, r, what):
    for f in ("mu", "nu", "known", "spread"):
        if getattr(r, f) is None:
            assert getattr(g, f) is None, (what, f)
        else:
            _same(getattr(g, f), getattr(r, f), f"{what} {f}")
    assert (g.hdr_known, g.spread_rho) == (r.hdr_known, r.spread_rho), what
    assert type(g.spread_rho) is type(r.spread_rho), what


def test_known_symbols_and_classes(Tn, row):
    bits = header.encode(HDR)
    rng = np.random.default_rng(5)
    for name in ("tiny", "short", "full", "short-v1", "full-spreadonly"):
        spec = frame.get(name)
        prior = EmPrior(x_hat=rng.standard_normal(spec.n_data), v=rng.uniform(-0.5, 1.5, spec.n_data))
        for b in (None, bits, list(bits)):
            k_r, k_g = T.known_symbols(spec, b), row["known_symbols"](spec, b)
            _same(k_g.pos, k_r.pos, f"{name} known pos")
            _same(k_g.val, k_r.val, f"{name} known val")
            for pr in (None, prior):
                _classes_same(row["make_classes"](spec, b, pr), T.make_classes(spec, b, pr), name)
    # header bits of other shapes: a short header is not refused by known_symbols (its pos and
    # val then differ in length, as the reference's do); leading unit axes broadcast in
    # make_classes unless the spread copy refuses them
    for name in ("short", "short-v1", "full-spreadonly", "tiny"):
        spec = frame.get(name)
        for bad in (bits[:-1], np.zeros((2, 2474)), np.zeros((1, 2474)), np.zeros((2474, 1)),
                    np.zeros((1, 1)), 3.0):
            _same_outcome(_outcome(T.make_classes, spec, bad), _outcome(row["make_classes"], spec, bad),
                          f"{name} bits {np.shape(bad)}")
            _same_outcome(_outcome(T.known_symbols, spec, bad), _outcome(row["known_symbols"], spec, bad),
                          f"{name} bits {np.shape(bad)}")
    spec = frame.SHORT
    pr = EmPrior(x_hat=np.zeros(spec.n_data + 1), v=np.zeros(spec.n_data))
    _same_error(_outcome(T.make_classes, spec, None, pr), _outcome(Tn.make_classes, spec, None, pr), "prior")
    pr = EmPrior(x_hat=np.zeros(1) + 0.5, v=np.ones(1) * 2.0)      # one element: broadcast
    _classes_same(row["make_classes"](spec, None, pr), T.make_classes(spec, None, pr), "broadcast")


def test_remove_spread(Tn, passes):
    spec = frame.SHORT
    cl = T.make_classes(spec, header.encode(HDR))
    tr = passes["short"].tracks[-1]
    for tm in (tr.timing, _timing(3.5, 70.0, 12.0)):
        _same(Tn.remove_spread(tr.z_d, tr.chan, spec, tm, cl.spread),
              T.remove_spread(tr.z_d, tr.chan, spec, tm, cl.spread), "spread")
    z = tr.z_d[:4000].real.copy()
    chan = SimpleNamespace(t0_index=-3000.25)
    _same(Tn.remove_spread(z, chan, spec, tr.timing, cl.spread),
          T.remove_spread(z, chan, spec, tr.timing, cl.spread), "real z, far from the frame")
    assert Tn.remove_spread(z, None, spec, None, None) is z
    _same_error(_outcome(T.remove_spread, tr.z_d, tr.chan, spec, tr.timing, cl.spread[:-1]),
                _outcome(Tn.remove_spread, tr.z_d, tr.chan, spec, tr.timing, cl.spread[:-1]), "shape")


def _nrel(g, r):
    return float(np.max(np.abs(g - r)) / max(float(np.max(np.abs(r))), 1e-300)) if r.size else 0.0


def test_templates(Tn, passes):
    # sf = MF{a^2 (1 - e^-v)}: numpy's SIMD exp and libm's differ by an ulp, and 1 - e^-v
    # cancels where v is small, so sf is held elementwise to 1e-9 and to the scale to 1e-14
    worst = {"c": 0.0, "sf": 0.0, "sf_norm": 0.0}
    for name, p in passes.items():
        spec = p.spec
        for a, k, ref in p.calls:
            cl = ref.classes
            keying, ook = T._keying_tuple(k.get("cw_known"))
            for kf in (None, spec.keyed_start_pos - 2.5, -50.0):
                for mu, nu in ((cl.mu, cl.nu), (np.where(cl.known, 0.0, cl.mu), np.where(cl.known, 1.0, cl.nu))):
                    r = T.templates(spec, ref.pos, mu, nu, keying, ook, kf)
                    g = Tn.templates(spec, ref.pos, mu, nu, keying, ook, kf)
                    for i, f in enumerate(("c", "sf")):
                        assert g[i].dtype == r[i].dtype and g[i].shape == r[i].shape
                        worst[f] = max(worst[f], rel(g[i], r[i]))
                    worst["sf_norm"] = max(worst["sf_norm"], _nrel(g[1], r[1]))
        # ook, and a partial grid
        keying = np.zeros((spec.n_win, 192), np.uint8) if spec.n_win else None
        grid = np.arange(-30, 600, dtype=np.int64)
        r = T.templates(spec, grid, ref.classes.mu, ref.classes.nu, keying, True)
        g = Tn.templates(spec, grid, ref.classes.mu, ref.classes.nu, keying, True)
        worst["c"] = max(worst["c"], rel(g[0], r[0]))
        worst["sf"] = max(worst["sf"], rel(g[1], r[1]))
        worst["sf_norm"] = max(worst["sf_norm"], _nrel(g[1], r[1]))
    assert worst["c"] <= 1e-10 and worst["sf"] <= 1e-9 and worst["sf_norm"] <= 1e-14, worst
    spec = frame.SHORT
    mu, nu = np.zeros(spec.n_sym), np.ones(spec.n_sym)
    for args in ((np.arange(0), mu, nu), (np.arange(4), mu[:-1], nu), (np.arange(4), mu, nu[1:]),
                 (np.arange(4), mu, nu, np.zeros((3, 192))), (np.arange(4), mu, nu, np.full(192, 2))):
        _same_error(_outcome(T.templates, spec, *args), _outcome(Tn.templates, spec, *args), str(args[1:]))


# --- the Kalman filter -----------------------------------------------------------------------------

def _final_run(tr, templates=T.templates):
    """(mu, R, Pu_g) of the tracker's last Kalman run, rebuilt from tr (with the run's own
    implementation of the templates: numpy's SIMD exp moves sf by ulps)."""
    spec = tr.spec
    known = tr.classes.known
    is_known = np.zeros(len(tr.pos), bool)
    is_known[np.asarray(frame.layout(spec).pos)[known] - tr.pos[0]] = True
    withheld = not tr.meas[is_known].any() and is_known.any()
    keyed_from = spec.keyed_start_pos - tr.lead_in_s / T_SYM
    mu, nu = tr.classes.mu, tr.classes.nu
    if withheld:
        mu, nu = np.where(known, 0.0, mu), np.where(known, 1.0, nu)
    _, sf = templates(spec, tr.pos, mu, nu, tr.keying, tr.ook, keyed_from)
    kappa = np.where(is_known & (not withheld), T.KAPPA_SELF_KNOWN, T.KAPPA_SELF)
    Pu_g = max(float(np.mean(tr.pu[tr.pos >= -SPAN])), 1e-30)
    m, R = T._measurements(tr, kappa, sf, tr.pu)
    return m, R, Pu_g


def test_kalman(Tn, passes):
    for name, p in passes.items():
        for a, k, ref in p.calls:
            mu, R, Pu_g = _final_run(ref)
            u, P, ll = T.kalman_rts(mu, R, ref.q, Pu_g)
            _same(u, ref.u, "the rebuilt run is the tracker's")
            for causal in (False, True):
                r, g = T.kalman_rts(mu, R, ref.q, Pu_g, causal), Tn.kalman_rts(mu, R, ref.q, Pu_g, causal)
                _same(g[0], r[0], "u")
                _same(g[1], r[1], "P")
                assert g[2] == r[2]
            for store in (False, True):
                r, g = T._kf_forward(mu, R, ref.q, Pu_g, store), Tn._kf_forward(mu, R, ref.q, Pu_g, store)
                assert g == r
            for grid in (T.Q_GRID, T.Q_GRID[3:6], (0.01, 0.01), (ref.q,), T.q_grid(L_T_SYM)):
                r, g = T.select_q(mu, R, Pu_g, grid), Tn.select_q(mu, R, Pu_g, grid)
                assert g == r and type(g[0]) is type(r[0]), grid
    # the smoother test's synthetic gain; nothing measured; edges
    rng = np.random.default_rng(4)
    n = 3000
    mu = np.exp(1j * np.cumsum(rng.standard_normal(n)) * 0.05) + 0.5 * (rng.standard_normal(n) + 1j * rng.standard_normal(n))
    R = np.where(rng.random(n) < 0.2, np.inf, 0.5)
    for args in ((mu, R, 1e-4, 1.0), (mu, np.full(n, np.inf), 1e-4, 1.0), (mu[:1], R[:1], 0.1, 2.0),
                 (mu[:0], R[:0], 0.1, 1.0), (mu.real.copy(), R, 3e-3, 0.7)):
        r, g = T.kalman_rts(*args), Tn.kalman_rts(*args)
        for i in (0, 1):
            _same(g[i], r[i], f"edge {len(args[0])}")
        assert g[2] == r[2]
        assert Tn.select_q(*args[:2], args[3]) == T.select_q(*args[:2], args[3])
    assert Tn.select_q(mu, R, 1.0, ()) is None and T.select_q(mu, R, 1.0, ()) is None
    for bad in (np.full(n, -1.0), np.full(n, -10.0 - 1e-30)):     # S < 0 and S == 0
        _same_error(_outcome(T.kalman_rts, mu, bad, 1e-4, 1.0), _outcome(Tn.kalman_rts, mu, bad, 1e-4, 1.0))


# --- the tracker's pieces ------------------------------------------------------------------------

def test_window_unknown(Tn):
    for spec in (frame.SHORT, frame.FULL, frame.TINY, lframe.L_SHORT):
        grid = np.arange(-40, spec.n_pos)
        for ook in (True, False):
            for reach in (SPAN, 0, 16, 2.5):
                for g in (grid, grid.astype(np.float64) + 0.5):
                    _same(Tn._window_unknown(spec, g, ook, reach), T._window_unknown(spec, g, ook, reach))


def test_measurements_and_pu(Tn, passes):
    for name, p in passes.items():
        for a, k, ref in p.calls:
            tr = copy.copy(ref)
            _same(np.array(Tn._pu_initial(tr)), np.array(T._pu_initial(tr)), "pu_initial")
            kappa = np.linspace(0.1, 1.0, len(tr.pos))
            sf = np.linspace(0.0, 0.3, len(tr.pos))
            r = T._measurements(tr, kappa, sf, tr.pu)
            g = Tn._measurements(tr, kappa, sf, tr.pu)
            _same(g[0], r[0], "mu")
            _same(g[1], r[1], "R")
            meas = np.ones(len(tr.pos), bool)
            r = T._measurements(tr, kappa, sf, tr.pu, c=tr.c * 0.5, meas=meas)
            g = Tn._measurements(tr, kappa, sf, tr.pu, c=tr.c * 0.5, meas=meas)
            _same(g[0], r[0], "mu, c and meas given")
            _same(g[1], r[1], "R, c and meas given")
    tr = copy.copy(ref)
    tr.meas = np.zeros(len(tr.pos), bool)
    assert Tn._pu_initial(tr) == T._pu_initial(tr) == 1e-30


def test_recenter(Tn, row, passes):
    for label, tr in _tracks(passes):
        rf, rd = T._recenter(tr)
        gf, gd = row["_recenter"](tr)
        assert abs(gd - rd) <= 1e-9 * abs(rd), (label, gd, rd)
        t = np.linspace(-30.0, tr.spec.n_pos * T_SYM + 30.0, 2001)
        assert np.max(np.abs(gf(t) - rf(t))) <= 1e-9, label
    still = copy.copy(tr)
    still.u = np.zeros(len(tr.pos), np.complex128)
    fn, df = row["_recenter"](still)
    assert fn is still.freq and df == 0.0 and T._recenter(still) == (still.freq, 0.0)


def test_keying_tuple_and_lead(Tn, passes):
    k = np.ones(192, np.int64)
    for cw in (None, (None, 0), (k, 1), [k.tolist(), np.bool_(True)], (np.zeros((2, 192)), 0.0)):
        r, g = T._keying_tuple(cw), Tn._keying_tuple(cw)
        assert type(g[1]) is type(r[1]) and g[1] == r[1]
        assert (g[0] is None) == (r[0] is None)
        if r[0] is not None:
            _same(g[0], r[0], "keying")
    for bad in ((k,), (k, 0, 1), 5):
        _same_error(_outcome(T._keying_tuple, bad), _outcome(Tn._keying_tuple, bad), str(bad))
    for name, p in passes.items():
        tr = p.tracks[0]
        for lead in (0.0, 0.1, 0.13, 1.0, 5.0, 8.0, 10.0):
            for tm in (tr.timing, _timing(tr.timing.tau0 + 400.0)):
                assert Tn._lead_supported(tr.chan, tr.z_d, tm, p.spec, lead) == \
                    T._lead_supported(tr.chan, tr.z_d, tm, p.spec, lead), (name, lead)


# --- track ---------------------------------------------------------------------------------------

def test_track_on_the_receivers_calls(Tn, row, passes):
    worst = {}
    for name, p in passes.items():
        assert len(p.calls) >= 2, name
        for i, (a, k, ref) in enumerate(p.calls):
            got = row["track"](*a, **k)
            what = f"{name} call {i}"
            assert got.q == ref.q and got.lead_in_s == ref.lead_in_s and got.ook == ref.ook, what
            assert (got.keying is None) == (ref.keying is None), what
            if ref.keying is not None:
                _same(got.keying, ref.keying, what)
            _same(got.pos, ref.pos, what)
            _same(got.meas, ref.meas, what)
            assert got.chan is ref.chan and got.spec is ref.spec and got.classes is ref.classes, what
            for f in ("z_d", "m", "c", "N", "psi"):
                worst[f] = max(worst.get(f, 0.0), rel(getattr(got, f), getattr(ref, f)))
            for f in ("u", "P", "pu"):
                worst[f] = max(worst.get(f, 0.0), rel(getattr(got, f), getattr(ref, f)))
            # u, P: the reference's own last Kalman step on the C++ run's inputs
            mu, R, Pu_g = _final_run(got, Tn.templates)
            u, P, ll = T.kalman_rts(mu, R, got.q, Pu_g)
            _same(got.u, u, what + " u")
            _same(got.P, P, what + " P")
            assert got.loglik == ll, what
            assert abs(got.df_max - ref.df_max) <= 1e-9 * max(abs(ref.df_max), 1e-3), what
            tm, rt = got.timing, ref.timing
            slip = (tm.ppm - rt.ppm) * 1e-6 * p.spec.n_pos * T_SYM * CH_FS
            assert abs(tm.tau0 - rt.tau0) <= 1e-6 and abs(slip) <= 1e-6 and abs(tm.gamma - rt.gamma) <= 1e-6
            assert (tm is rt) == (ref.timing is (k.get("timing") or a[2].timing)), what
            t = np.linspace(-30.0, p.spec.n_pos * T_SYM + 30.0, 2001)
            assert np.max(np.abs(got.freq(t) - ref.freq(t))) <= 1e-9, what
            assert (got.freq is k.get("freq")) == (ref.freq is k.get("freq")), what
    for f in ("z_d", "m", "c", "N", "psi"):
        assert worst[f] <= 1e-9, (f, worst)
    for f, tol in (("u", 1e-8), ("P", 1e-5), ("pu", 1e-6)):
        assert worst[f] <= tol, (f, worst)


def test_track_arguments(Tn, row, passes):
    p = passes["short"]
    a, k, ref = p.calls[-1]                      # round B: classes, callsign keying, a path
    chan, spec, det = a[:3]
    variants = [
        dict(timing=k["timing"], freq=k["freq"], outer=0, refine=False),
        dict(timing=k["timing"], lead_in_s=0.5, outer=1),
        dict(timing=k["timing"], withhold_known=True, outer=0, refine=False, z_d=ref.z_d),
    ]
    for kw in variants:
        r = T.track(chan, spec, det, a[3], k.get("cw_known"), **kw)
        g = row["track"](chan, spec, det, a[3], k.get("cw_known"), **kw)
        assert g.q == r.q and g.lead_in_s == r.lead_in_s, kw
        for f in ("m", "c", "psi", "N"):
            assert rel(getattr(g, f), getattr(r, f)) <= 1e-9, (kw, f)
    # round A from scratch with a prior, det's own timing
    det2 = dataclasses.replace(det, timing=k["timing"])
    prior = EmPrior(x_hat=np.zeros(spec.n_data), v=np.full(spec.n_data, 0.5))
    r = T.track(chan, spec, det2, None, None, prior, outer=0, refine=False)
    g = row["track"](chan, spec, det2, None, None, prior, outer=0, refine=False)
    assert g.q == r.q and g.timing is det2.timing
    _classes_same(g.classes, r.classes, "prior")
    # refusals: no timing, a bad keying, an L frame
    det0 = dataclasses.replace(det, timing=None)
    _same_error(_outcome(T.track, chan, spec, det0), _outcome(row["track"], chan, spec, det0), "no timing")
    bad = (np.full(192, 3), 0)
    _same_error(_outcome(T.track, chan, spec, det2, None, bad), _outcome(row["track"], chan, spec, det2, None, bad))


# --- timing refinement, Z_ref, erase, report --------------------------------------------------

def test_refine_timing(Tn, row, passes):
    outcomes = []
    for name, p in passes.items():
        for a, k, ref in p.calls:
            for kw in ({}, dict(iters=1), dict(u_ref=ref.u * 0.9, P_ref=ref.P * 2.0)):
                r, g = T.refine_timing(ref, **kw), row["refine_timing"](ref, **kw)
                outcomes.append(r is None)
                assert (g is None) == (r is None), (name, kw)
                if r is None:
                    continue
                n_pos, sps = p.spec.n_pos, p.spec.t_sym * CH_FS
                assert abs(g.tau0 - r.tau0) <= 1e-6 and abs(g.gamma - r.gamma) <= 1e-6
                assert abs((g.ppm - r.ppm) * 1e-6 * n_pos * sps) <= 1e-6
                assert abs(g.z - r.z) <= 1e-9 * abs(r.z)
                assert np.max(np.abs(g.cov - r.cov)) <= 1e-6 * np.max(np.abs(r.cov))
    assert True in outcomes and False in outcomes        # gate V's withheld runs refine to nothing
    # a timing far off, noise, too few known symbols, MEDIUM's gamma branch
    tr = copy.copy(passes["short"].tracks[0])
    rng = np.random.default_rng(9)
    cases = [dataclasses.replace(tr, timing=_timing(tr.timing.tau0 + 3.0, tr.timing.ppm)),
             dataclasses.replace(tr, z_d=(rng.standard_normal(len(tr.z_d)) + 1j * rng.standard_normal(len(tr.z_d))) * 0.01),
             dataclasses.replace(tr, meas=np.zeros(len(tr.pos), bool)),
             dataclasses.replace(tr, classes=None),
             dataclasses.replace(tr, spec=frame.MEDIUM)]
    for c in cases:
        r, g = _outcome(T.refine_timing, c), _outcome(row["refine_timing"], c)
        assert g[0] == r[0], (r, g)
        if r[0] == "ok":
            assert (g[1] is None) == (r[1] is None), (r, g)
            if r[1] is not None:
                assert abs(g[1].tau0 - r[1].tau0) <= 1e-6 and abs(g[1].gamma - r[1].gamma) <= 1e-6


def test_z_ref_erase_report(Tn, row, passes):
    for label, tr in _tracks(passes):
        spec = tr.spec
        c_known, _ = T.templates(spec, tr.pos, tr.classes.mu, tr.classes.nu, None, False)
        for d in (None, c_known, 1j * c_known.imag):
            r, g = T.z_ref_stat(tr, c_known, d), Tn.z_ref_stat(tr, c_known, d)
            assert abs(g - r) <= 1e-8 * max(abs(r), 1.0), (label, g, r)
        for kappa, suspect in ((1.0, False), (0.73, True)):
            r, g = T.report(tr, kappa, suspect), row["report"](tr, kappa, suspect)
            for f in ("offset_hz", "wander_hz_rms", "doppler_hz", "ppm", "snr2500_db", "z_ref", "kappa"):
                assert abs(getattr(g, f) - getattr(r, f)) <= 1e-9 * abs(getattr(r, f)), (label, f)
            assert abs(g.drift_hz_per_min - r.drift_hz_per_min) <= 1e-9, label
            assert (g.slip_free, g.suspect) == (r.slip_free, r.suspect), label
        for spans in ((), [(10.0, 12.0)], [(5.0, 6.0), (np.float64(30.0), 31)], [(-1e9, 1e9)]):
            a, b = copy.copy(tr), copy.copy(tr)
            T.erase(a, spans)
            assert Tn.erase(b, spans) is None
            _same(b.psi, a.psi, f"{label} erase")
            assert (b.psi is tr.psi) == (a.psi is tr.psi)
    tr = copy.copy(tr)
    for bad in (np.array([[1.0, 2.0]]), [(1.0, 2.0, 3.0)], [1.0]):
        _same_error(_outcome(T.erase, copy.copy(tr), bad), _outcome(Tn.erase, copy.copy(tr), bad), str(bad))
    none = copy.copy(tr)
    none.psi = np.zeros(len(tr.psi))
    assert Tn.z_ref_stat(none, c_known) == T.z_ref_stat(none, c_known) == 0.0


def test_lunits_fakes(Tn):
    """test_qrss_lunits' fake tracks: chan None, classes a SimpleNamespace,
    an L frame (z_ref_stat reads lframe's layout)."""
    rng = np.random.default_rng(3)
    for spec, mod in ((frame.TINY, frame), (lframe.L_TINY, lframe)):
        known = np.zeros(spec.n_sym, bool)
        known[mod.layout(spec).known_idx] = True
        n = spec.n_pos + SPAN
        u = 0.7 * np.exp(1j * np.linspace(0, 3, n))
        c = rng.standard_normal(n) + 1j * rng.standard_normal(n)
        m = u * c + 0.5 * (rng.standard_normal(n) + 1j * rng.standard_normal(n))
        tr = T.TrackResult(spec=spec, chan=None, timing=_timing(), freq=None, pos=np.arange(-SPAN, spec.n_pos),
                           z_d=None, m=m, c=c, u=u, P=np.full(n, 0.01), N=np.full(n, 0.5), psi=np.ones(n),
                           meas=None, pu=None, classes=SimpleNamespace(known=known))
        for d in (None, c, 1j * c.imag):
            r, g = T.z_ref_stat(tr, c, d), Tn.z_ref_stat(tr, c, d)
            assert abs(g - r) <= 1e-12 * abs(r), (spec.name, g, r)


def test_ce_only(Tn, row):
    spec = lframe.L_SHORT
    calls = [
        (T.track, row["track"], (None, spec, None)),
        (T.make_classes, row["make_classes"], (spec,)),
        (T.known_symbols, row["known_symbols"], (spec,)),
        (T.templates, Tn.templates, (spec, np.arange(4), None, None)),
        (T.remove_spread, Tn.remove_spread, (np.zeros(4), None, spec, None, None)),
        (T.refine_timing, row["refine_timing"], (SimpleNamespace(spec=spec),)),
    ]
    for ref, nat, args in calls:
        r, g = _outcome(ref, *args), _outcome(nat, *args)
        assert r[0] is AssertionError and g == r, (r, g)


# --- the whole caller --------------------------------------------------------------------------

def test_receive_pass_on_the_cpp(native, passes, monkeypatch):
    """receiver.receive_pass on TINY and SHORT with every row of
    tests/qrss_native/track.py assigned over track's names (and ltrack's
    from-imports of them): the same passes, headers and callsign, f_hz to
    1e-6 Hz and the latents to 1e-6."""
    from qrss_native.track import rows

    table = rows(native.qrss)
    assert all(target == TRACK for target, _ in table)
    used = {"track": 0}
    for (_t, name), fn in table.items():
        monkeypatch.setattr(T, name, fn)
    row_track = table[(TRACK, "track")]

    def counted(*a, **k):
        used["track"] += 1
        return row_track(*a, **k)

    monkeypatch.setattr(T, "track", counted)
    for name in ("derotate", "kalman_rts", "select_q", "_smooth", "_keying_tuple"):
        monkeypatch.setattr(LT, name, table[(TRACK, name)])
    for name, p in passes.items():
        before = used["track"]
        res = RX.receive_pass(p.prep, p.spec, flat_det(1500.0), return_tracks=True)
        assert used["track"] - before == len(p.calls), name
        assert res is not None, name
        g, r = res[0], p.pr
        assert g.uid.rsplit("_", 1)[0] == r.uid.rsplit("_", 1)[0], name
        assert (g.q, g.frame, g.waveform, g.estimator) == (r.q, r.frame, r.waveform, r.estimator)
        assert g.header == r.header and g.cw == r.cw, name
        assert abs(g.f_hz - r.f_hz) <= 1e-6, name
        for f, tol in (("z", 1e-6), ("w", 1e-6), ("hdr_llr", 1e-4), ("psi", 1e-6)):
            a, b = np.asarray(getattr(g, f)), np.asarray(getattr(r, f))
            assert a.shape == b.shape and a.dtype == b.dtype, (name, f)
            assert a.size == 0 or np.max(np.abs(a.astype(np.float64) - b)) <= tol, (name, f)
        _same(g.ch, r.ch, name)
        assert abs(g.timing.tau0 - r.timing.tau0) <= 1e-6 and abs(g.timing.ppm - r.timing.ppm) <= 1e-6
        for f in ("offset_hz", "doppler_hz", "snr2500_db", "z_ref", "kappa"):
            assert abs(getattr(g.report, f) - getattr(r.report, f)) <= 1e-6 * max(abs(getattr(r.report, f)), 1.0), f
        assert (g.report.slip_free, g.report.suspect) == (r.report.slip_free, r.report.suspect)
        assert len(res[1]) == len(p.tracks), name


# --- probes: sizes, inputs and paths the receiver's two passes do not reach ---------------------

def test_symbol_mf_at_full_size(Tn):
    """FULL's grid (sample indices past 3.5e5 with t0 and a large tau0),
    random timings with gamma, fractional positions, the CE and L tables,
    complex128, complex64 and a float32 keep: bit for bit (measured, every
    trial). Every fifth position, so the whole range is reached cheaply."""
    rng = np.random.default_rng(7)
    spec = frame.FULL
    n = int(spec.n_pos * T_SYM * CH_FS) + 6000
    z = (rng.standard_normal(n) + 1j * rng.standard_normal(n)) * 10 ** rng.uniform(-3, 3)
    keep = ((rng.random(n) > 0.05) * rng.uniform(0.5, 1.0, n)).astype(np.float32)
    lt = T.mf_table(L_T_SYM)
    for trial in range(4):
        tm = Timing(tau0=rng.uniform(-300, 300), ppm=rng.uniform(-250, 250), gamma=rng.uniform(-60, 60),
                    z=0.0, cov=np.zeros((3, 3)))
        pos = np.arange(-SPAN - 264, spec.n_pos, 5).astype(np.float64)
        if trial % 2:
            pos += rng.uniform(-0.5, 0.5, len(pos))
        kw = dict(t0_index=rng.uniform(1000, 3000), n_pos=spec.n_pos, table=lt if trial >= 2 else None)
        for zz in (z, z.astype(np.complex64), keep):
            _same(Tn.symbol_mf(zz, CH_FS, tm, pos, **kw), T.symbol_mf(zz, CH_FS, tm, pos, **kw),
                  f"trial {trial} {zz.dtype}")
        _same(Tn.psi_at(keep, tm, pos, **kw), T.psi_at(keep, tm, pos, **kw), f"trial {trial} psi")


def test_kalman_fuzz(Tn):
    """kalman_rts (both modes), select_q and _kf_forward(store) on random
    gains across twelve decades, with inf, NaN, -inf and zero variances, a
    NaN measurement, q and Pu of 0: bit for bit, or the same refusal."""
    rng = np.random.default_rng(11)

    def outcome(f, *a):
        try:
            return "ok", f(*a)
        except Exception as e:                                   # noqa: BLE001  (compared below)
            return type(e), str(e)

    for trial in range(160):
        n = int(rng.choice([0, 1, 2, 3, 5, 17, 200, 1200]))
        mu = (rng.standard_normal(n) + 1j * rng.standard_normal(n)) * 10 ** rng.uniform(-6, 6)
        R = 10 ** rng.uniform(-8, 4, n)
        R[rng.random(n) < 0.3] = np.inf
        if n:
            for every, arr, v in ((7, R, np.nan), (11, mu, np.nan), (13, R, -np.inf), (17, R, 0.0)):
                if trial % every == 0:
                    arr[rng.integers(n)] = v
        q = float(rng.choice([0.0, 1e-12, 1e-6, 1e-3, 0.5, 10 ** rng.uniform(-14, 0)]))
        Pu = float(rng.choice([0.0, 1e-30, 1.0, 10 ** rng.uniform(-10, 6)]))
        what = f"trial {trial} n {n} q {q} Pu {Pu}"
        for causal in (False, True):
            r, g = outcome(T.kalman_rts, mu, R, q, Pu, causal), outcome(Tn.kalman_rts, mu, R, q, Pu, causal)
            assert g[0] == r[0], (what, r, g)
            if r[0] != "ok":
                assert g == r, what
                continue
            _same(g[1][0], r[1][0], what + " u")
            _same(g[1][1], r[1][1], what + " P")
            _same(np.float64(g[1][2]), np.float64(r[1][2]), what + " loglik")
        r, g = outcome(T.select_q, mu, R, Pu), outcome(Tn.select_q, mu, R, Pu)
        assert g[0] == r[0] and (r[0] != "ok" or g[1][0] == r[1][0]), (what, r, g)
        if r[0] == "ok":
            _same(np.float64(g[1][1]), np.float64(r[1][1]), what + " select_q loglik")
        r, g = outcome(T._kf_forward, mu, R, q, Pu, True), outcome(Tn._kf_forward, mu, R, q, Pu, True)
        assert g[0] == r[0], (what, r, g)
        if r[0] == "ok":
            _same(np.float64(g[1][0]), np.float64(r[1][0]), what + " forward loglik")
            for x, y in zip(g[1][1], r[1][1], strict=True):
                _same(np.asarray(x), np.asarray(y), what + " stored")


@pytest.fixture(scope="module")
def lead_pass(Tn):
    """A TINY pass with 9 s of keyed lead-in and 1.5 Hz/min of drift at
    -9 dB, the reference's track calls recorded as in `passes`."""
    from qrss_helpers import unit_rms_latents
    from sstvae.qrss import channel as chm, morse
    from sstvae.qrss.channel import ChannelConfig
    from sstvae.qrss.precoder import precode

    spec = frame.TINY
    a = unit_rms_latents(spec.n_data, 1)
    sym = frame.assemble(spec, header.encode(HDR) if spec.n_hdr else None, precode(a, Q_TEST))
    sim = chm.simulate(sym, spec, Q_TEST, ChannelConfig(snr_db=-9.0, seed=6, drift_hz_per_min=1.5),
                       carrier_hz=1500.0, lead_in_s=9.0,
                       keying=morse.keying_units(CALL) if spec.n_win else None)
    prep = RX.prepare(frontend.Capture(sim.fe, Q_TEST, sim.t0_index))
    calls = []
    orig = T.track

    def spy(*a, **k):
        tr = orig(*a, **k)
        calls.append((a, dict(k), copy.copy(tr)))
        return tr

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(T, "track", spy)
        res = RX.receive_pass(prep, spec, flat_det(1500.0), return_tracks=True)
    assert res is not None and calls
    return SimpleNamespace(spec=spec, prep=prep, calls=calls)


def _track_agrees(Tn, g, r, what, spec, worst):
    """track()'s contract for one call: the decisions equal, the values to
    rounding, u and P what the reference's last Kalman step makes of the
    C++ run's own inputs (bit for bit)."""
    assert g.q == r.q and g.lead_in_s == r.lead_in_s and g.ook == r.ook, what
    _same(g.pos, r.pos, what)
    _same(g.meas, r.meas, what)
    for f in ("z_d", "m", "c", "N", "psi", "u", "P", "pu"):
        worst[f] = max(worst.get(f, 0.0), rel(getattr(g, f), getattr(r, f)))
    mu, R, Pu_g = _final_run(g, Tn.templates)
    u, P, ll = T.kalman_rts(mu, R, g.q, Pu_g)
    _same(g.u, u, what + " u")
    _same(g.P, P, what + " P")
    assert g.loglik == ll, what
    tm, rt = g.timing, r.timing
    slip = (tm.ppm - rt.ppm) * 1e-6 * spec.n_pos * T_SYM * CH_FS
    assert abs(tm.tau0 - rt.tau0) <= 1e-6 and abs(slip) <= 1e-6 and abs(tm.gamma - rt.gamma) <= 1e-6, what
    t = np.linspace(-30.0, spec.n_pos * T_SYM + 30.0, 2001)
    assert np.max(np.abs(g.freq(t) - r.freq(t))) <= 1e-9, what


def test_track_with_a_lead_in(Tn, row, lead_pass):
    """The lead-in branch (`_lead_supported` true, the grid extended before
    the frame, LEAD_IN_MAX_S's clamp) and the receiver's own calls on a
    drifting capture: the reference's P on the gate-V run sits 2.0e-5 from
    the C++ (u 2.7e-9, pu 8.9e-8; measured), its own conditioning on an
    ulp of numpy's SIMD exp in sf, so the bar is the conditional
    bit-for-bit check (and the swap test below)."""
    spec = lead_pass.spec
    worst = {}
    for i, (a, k, ref) in enumerate(lead_pass.calls):
        _track_agrees(Tn, row["track"](*a, **k), ref, f"call {i}", spec, worst)
    a, k, _ = lead_pass.calls[-1]
    grids = {}
    for lead in (2.0, 9.0, 12.0):
        kw = dict(k, lead_in_s=lead)
        r, g = T.track(*a, **kw), row["track"](*a, **kw)
        _track_agrees(Tn, g, r, f"lead {lead}", spec, worst)
        grids[lead] = (r.lead_in_s, len(r.pos))
    assert grids[9.0][1] > grids[2.0][1] > spec.n_pos + SPAN, grids    # supported, not refused
    assert grids[12.0] == (10.0, grids[9.0][1]), grids                 # clamped to LEAD_IN_MAX_S
    for f in ("z_d", "m", "c", "N", "psi"):
        assert worst[f] <= 1e-9, (f, worst)
    assert worst["u"] <= 1e-8 and worst["P"] <= 1e-4 and worst["pu"] <= 1e-6, worst


def test_track_is_the_reference_up_to_smooth_and_templates(Tn, row, passes, lead_pass):
    """The reference's track with the C++ `_smooth` and `templates` assigned
    over its own is the C++ track bit for bit: every array field, q,
    loglik, df_max and the recentred path, on every recorded call (gate V's
    withheld runs, rounds A and B, the lead-in pass). So the two seeds named
    in the module docstring -- np.convolve's BLAS sum and numpy's SIMD exp
    in sf -- are where the C++ tracker's arrays depart from the reference,
    and everything downstream of them (the Kalman iterations, recentring's
    spline refit, the lowpass) is the reference's arithmetic exactly. The
    one other seed is refine_timing's Newton step, math::solve's LU against
    OpenBLAS's getrf (the math lane's documented to-rounding linalg): the
    refined timing to an ulp (measured tau0 1.4e-17, ppm 1.9e-16, the
    covariance 2e-18 relative), well under the sample grid's own rounding,
    so the arrays it feeds stay bit for bit."""
    calls = [(n, i, c) for n, p in passes.items() for i, c in enumerate(p.calls)]
    calls += [("lead", i, c) for i, c in enumerate(lead_pass.calls)]
    fields = ("pos", "meas", "z_d", "m", "c", "N", "psi", "u", "P", "pu")
    for name, i, (a, k, _ref) in calls:
        what = f"{name} call {i}"
        g = row["track"](*a, **k)
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(T, "_smooth", Tn._smooth)
            mp.setattr(T, "templates", Tn.templates)
            h = T.track(*a, **k)
        for f in fields:
            _same(getattr(g, f), getattr(h, f), f"{what} {f}")
        assert (g.q, g.loglik, g.df_max, g.lead_in_s, g.ook) == (h.q, h.loglik, h.df_max, h.lead_in_s, h.ook), what
        tm, ht = g.timing, h.timing
        for f in ("tau0", "ppm", "gamma", "z"):
            assert abs(getattr(tm, f) - getattr(ht, f)) <= 1e-12 * max(abs(getattr(ht, f)), 1.0), (what, f)
        assert np.max(np.abs(tm.cov - ht.cov)) <= 1e-12 * np.max(np.abs(ht.cov)), what
        t = np.linspace(-30.0, h.spec.n_pos * T_SYM + 30.0, 2001)
        _same(g.freq(t), h.freq(t), what + " path")


def test_track_round_b_variants(Tn, row, passes):
    """SHORT's round B (callsign keying, a recentred path) with the keying
    taken as OOK, the path moved 45 mHz and ramped, the timing moved 1.7
    samples and 3 ppm, and four outer iterations."""
    p = passes["short"]
    a, k, ref = p.calls[-1]
    assert k.get("cw_known") is not None and k.get("freq") is not None
    tt = np.linspace(-10.0, p.spec.n_pos * T_SYM + 10.0, 400)
    off = T.PathFn(tt, k["freq"](tt) + 0.045 + 0.0002 * tt)
    tm0 = k["timing"]
    moved = Timing(tau0=tm0.tau0 + 1.7, ppm=tm0.ppm + 3.0, gamma=0.0, z=0.0, cov=np.zeros((3, 3)))
    worst = {}
    for what, kw in (("ook", dict(k, cw_known=(k["cw_known"][0], True))),
                     ("path off, outer 4", dict(k, freq=off, outer=4)),
                     ("timing moved", dict(k, timing=moved))):
        r, g = T.track(*a, **kw), row["track"](*a, **kw)
        _track_agrees(Tn, g, r, what, p.spec, worst)
    for f in ("z_d", "m", "c", "N", "psi"):
        assert worst[f] <= 1e-9, (f, worst)
    assert worst["u"] <= 1e-8 and worst["P"] <= 1e-4 and worst["pu"] <= 1e-6, worst


def test_l_receive_pass_on_the_cpp(native, monkeypatch):
    """receiver.receive_pass on an L TINY pass at -12 dB with every row over
    track's names and ltrack's from-imports of them: ltrack reaches
    symbol_mf, psi_at, kalman_rts, select_q, derotate, _smooth, _recenter,
    erase, z_ref_stat and report through them, and the pass is the
    reference's (measured: z, w and psi identical, f_hz 2e-13 Hz)."""
    from qrss_fakes import l_scenario
    from qrss_native.track import rows
    from sstvae.qrss.constants import WAVEFORM_L

    spec = lframe.get("l-tiny")
    _, sim = l_scenario("l-tiny", -12.0, 1)
    prep = RX.prepare(frontend.Capture(sim.fe, Q_TEST, sim.t0_index))
    det = flat_det(1500.0, WAVEFORM_L)
    ref = RX.receive_pass(prep, spec, det, return_tracks=True)
    assert ref is not None
    table = rows(native.qrss)
    used = {}

    def counted(name, fn):
        def f(*a, **k):
            used[name] = used.get(name, 0) + 1
            return fn(*a, **k)
        return f

    wrapped = {name: counted(name, fn) for (_t, name), fn in table.items()}
    from_imports = ("derotate", "kalman_rts", "select_q", "_smooth", "_keying_tuple")
    for name in from_imports:                    # ltrack's own bindings are track's functions
        assert getattr(LT, name) is getattr(T, name), name
    for name, fn in wrapped.items():
        monkeypatch.setattr(T, name, fn)
    for name in from_imports:
        monkeypatch.setattr(LT, name, wrapped[name])
    got = RX.receive_pass(prep, spec, det, return_tracks=True)
    assert got is not None
    reached = {"symbol_mf", "psi_at", "kalman_rts", "select_q", "derotate", "_smooth", "_recenter",
               "erase", "z_ref_stat", "report"}
    assert reached <= set(used), sorted(set(reached) - set(used))
    g, r = got[0], ref[0]
    assert g.uid.rsplit("_", 1)[0] == r.uid.rsplit("_", 1)[0]
    assert (g.q, g.frame, g.waveform, g.estimator) == (r.q, r.frame, r.waveform, r.estimator)
    assert g.header == r.header and g.cw == r.cw
    assert abs(g.f_hz - r.f_hz) <= 1e-6
    for f in ("z", "w", "hdr_llr", "psi"):
        x, y = np.asarray(getattr(g, f)), np.asarray(getattr(r, f))
        assert x.shape == y.shape and x.dtype == y.dtype, f
        assert x.size == 0 or np.max(np.abs(x.astype(np.float64) - y)) <= 1e-6, f
    assert abs(g.timing.tau0 - r.timing.tau0) <= 1e-6 and abs(g.timing.ppm - r.timing.ppm) <= 1e-6
    for f in ("offset_hz", "doppler_hz", "snr2500_db", "z_ref", "kappa"):
        assert abs(getattr(g.report, f) - getattr(r.report, f)) <= 1e-6 * max(abs(getattr(r.report, f)), 1.0), f
    assert (g.report.slip_free, g.report.suspect) == (r.report.slip_free, r.report.suspect)
    assert len(got[1]) == len(ref[1])

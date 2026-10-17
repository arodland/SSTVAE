"""Parity of native/core/qrss/trackbase (sstvae_native.qrss.trackbase)
with the tracker's shared vocabulary in sstvae/qrss/track.py and ltrack.py.

- q_of_spread, spread_of_q, q_grid and Q_GRID to the bit;
- rx_index and _tau_of_rx to 1e-12 relative (the same arithmetic in the
  same order, so in practice to the bit);
- PathFn within 1e-10 Hz (the line's coefficients are np.polyfit's to
  ~1e-15 relative, its SVD not being LAPACK's; given the same parts the
  evaluation is the same doubles): the line and the spline, 1-8 and more points,
  knot intervals without data, repeated times, weights, extrapolation on
  both sides, the reference's refusals; NativePathFn (the row) is the
  reference's class, pickles and copies;
- ChanCapture's maps, MfTable's validation, and every record (ChanCapture,
  MfTable, _Known, Classes, LClasses, TrackResult, LTrackResult) across
  Python -> C++ -> Python unchanged.
"""

import copy
import math
import pickle
from fractions import Fraction

import numpy as np
import pytest

from qrss_native import _rx, require
from sstvae.qrss import constants as C
from sstvae.qrss import frame, lframe, ltrack, track
from sstvae.qrss import types as qt

L_T_SYM = ltrack.L_T_SYM


@pytest.fixture(scope="module")
def tb(native, request):
    if request.config.getoption("--native-qrss", default=None):
        pytest.skip("--native-qrss replaces the reference this file compares against")
    return require(native, "trackbase")


def _same(a, b):
    a, b = np.asarray(a), np.asarray(b)
    assert a.dtype == b.dtype, (a.dtype, b.dtype)
    assert a.shape == b.shape, (a.shape, b.shape)
    an = np.isnan(a) if a.dtype.kind in "fc" else np.zeros(a.shape, bool)
    bn = np.isnan(b) if b.dtype.kind in "fc" else np.zeros(b.shape, bool)
    assert np.array_equal(an, bn)
    assert a[~an].tobytes() == b[~bn].tobytes(), (a, b)


def _rel(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    assert a.shape == b.shape
    assert np.array_equal(np.isnan(a), np.isnan(b))
    ok = ~np.isnan(a)
    if not ok.any():
        return 0.0
    return float(np.max(np.abs(a[ok] - b[ok]) / np.maximum(np.abs(b[ok]), 1e-300)))


def _normwise(a, b):
    """max |a - b| over max |b|: a small coefficient beside a large one
    (a slope of 1e-3 Hz/s beside 1500 Hz) is only determined to the
    larger one's rounding, by any least-squares solver."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return float(np.max(np.abs(a - b)) / max(float(np.max(np.abs(b))), 1e-300))


def _outcome(fn, *args, **kw):
    try:
        out = fn(*args, **kw)
    except (ValueError, TypeError, IndexError, ZeroDivisionError, OverflowError,
            np.linalg.LinAlgError) as e:
        return type(e), str(e)
    return "ok", out


# --- process noise ------------------------------------------------------------------------------


def test_q_maps_bit_exact(tb):
    rng = np.random.default_rng(1)
    for s in [0.0, 0.01, 0.1, 1.0, 3.0, 17.0, *rng.uniform(0, 5, 200)]:
        for t_sym in (C.T_SYM, L_T_SYM, 1.0):
            assert tb.q_of_spread(float(s), t_sym) == track.q_of_spread(float(s), t_sym)
            q = track.q_of_spread(float(s), t_sym)
            assert tb.spread_of_q(q, t_sym) == track.spread_of_q(q, t_sym)
    for t_sym in (C.T_SYM, L_T_SYM, 0.5):
        assert tb.q_grid(t_sym) == track.q_grid(t_sym)
    assert tb.q_grid() == track.q_grid()
    assert tb.Q_GRID == track.Q_GRID
    assert tb.Q_SPREADS_HZ == track.Q_SPREADS_HZ
    assert tb.KNOTS_S == track.KNOTS_S and tb.MF_RES == track.MF_RES


# --- timing -------------------------------------------------------------------------------------


def _timings(rng, n=30):
    out = [qt.Timing(tau0=0.0, ppm=0.0, gamma=0.0, z=0.0, cov=np.zeros((3, 3)))]
    for _ in range(n):
        out.append(qt.Timing(tau0=float(rng.uniform(-5e3, 5e4)), ppm=float(rng.uniform(-200, 200)),
                             gamma=float(rng.uniform(-30, 30)), z=1.0, cov=np.eye(3)))
    return out


def test_rx_index_and_tau_of_rx(tb):
    rng = np.random.default_rng(2)
    worst = 0.0
    for tm in _timings(rng):
        for n_pos, t_sym in ((frame.SHORT.n_pos, C.T_SYM), (lframe.get("l-short").n_pos, L_T_SYM)):
            tau = np.concatenate([np.arange(-50, n_pos + 50, 7), rng.uniform(-10, n_pos, 100)])
            want = track.rx_index(tm, tau, n_pos, t_sym=t_sym)
            got = tb.rx_index(tm, tau, n_pos, t_sym=t_sym)
            worst = max(worst, _rel(got, want))
            want = track.rx_index(tm, tau.astype(np.int64), n_pos, C.CH_FS, t_sym)
            worst = max(worst, _rel(tb.rx_index(tm, tau.astype(np.int64), n_pos, C.CH_FS, t_sym), want))
            rel = track.rx_index(tm, tau, n_pos, t_sym=t_sym)
            want = track._tau_of_rx(tm, rel, n_pos, t_sym)
            got = tb._tau_of_rx(tm, rel, n_pos, t_sym)
            worst = max(worst, _rel(got, want))
            # shapes: 0-d gives a numpy scalar, 2-D stays 2-D
            g0, w0 = tb.rx_index(tm, 3.0, n_pos), track.rx_index(tm, 3.0, n_pos)
            assert type(g0) is type(w0) is np.float64
            assert tb.rx_index(tm, tau[:20].reshape(4, 5), n_pos).shape == (4, 5)
            assert tb._tau_of_rx(tm, rel[:20].reshape(4, 5), n_pos).shape == (4, 5)
    assert worst <= 1e-12, worst


# --- PathFn -------------------------------------------------------------------------------------


def _pathfn_cases():
    rng = np.random.default_rng(3)
    cases = []
    for n in range(1, 9):                                   # 1-8 points: line, then the first spline
        t = np.sort(rng.uniform(-30, 30 + 20 * n, n))
        cases.append((f"n{n}", t, 1500 + 0.02 * t + rng.standard_normal(n) * 0.1, None, 10.0))
    for n in (9, 20, 100, 1000):
        t = np.sort(rng.uniform(0, 600, n))
        f = 1500 + 0.01 * t + 0.3 * np.sin(t / 40)
        w = rng.uniform(0, 3, n)
        cases.append((f"n{n}", t, f, w, 10.0))
        cases.append((f"n{n}-knots60", t, f, w, 60.0))
    t = np.r_[np.linspace(0, 30, 20), np.linspace(200, 260, 20)]   # a gap: knots without data
    cases.append(("gap", t, 1500 + 0.001 * t, None, 10.0))
    t = np.r_[0.0, 3.5, np.sort(rng.uniform(60, 100, 30)), 100.0]   # scipy's skipped rows
    cases.append(("skipped-rows", t, 1500 + np.sin(t / 7), None, 10.0))
    t = np.repeat(np.linspace(0, 100, 15), 2)                      # repeated times
    cases.append(("repeats", t, 1500 + 0.01 * t + np.tile([0.0, 0.1], 15), None, 10.0))
    t = rng.permutation(np.linspace(0, 120, 40))                   # unsorted
    cases.append(("unsorted", t, 1500 + 0.002 * t ** 1.5, rng.uniform(-1, 2, 40), 10.0))
    t = np.linspace(0, 90, 30)
    cases.append(("zero-weights", t, 1500 + 0.01 * t, np.zeros(30), 10.0))
    cases.append(("short-span", np.linspace(0, 9.9, 30), np.full(30, 1501.0), None, 10.0))
    cases.append(("exact-span", np.linspace(0, 10, 30), np.linspace(1500, 1501, 30), None, 10.0))
    cases.append(("two-equal-t", np.array([5.0, 5.0, 9.0]), np.array([1.0, 2.0, 3.0]), None, 10.0))
    cases.append(("ints", np.arange(12), np.arange(12) * 2 + 1500, np.ones(12, int), 3))
    return cases


CASES = _pathfn_cases()


@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
def test_pathfn_matches_reference(tb, case):
    _, t, f, w, knots = case
    ref = track.PathFn(t, f, w, knots)
    nat = tb.PathFn(t, f, w, knots)
    assert nat.lo == ref.lo and nat.hi == ref.hi
    assert _normwise(nat.line, ref._line) <= 1e-13      # np.polyfit's lstsq: SVD to rounding
    assert (nat.spline is None) == (ref._spl is None)
    if ref._spl is not None:
        st, sc, sk = nat.spline
        _same(st, ref._spl.t)
        assert sk == ref._spl.k
        assert np.max(np.abs(sc - ref._spl.c)) <= 1e-10
    span = ref.hi - ref.lo
    x = np.concatenate([np.linspace(ref.lo - span - 50, ref.hi + span + 50, 2001), np.asarray(t, float),
                        [ref.lo, ref.hi, np.nan]])
    want, got = ref(x), nat(x)
    assert np.array_equal(np.isnan(got), np.isnan(want))
    ok = ~np.isnan(want)
    assert np.max(np.abs(got[ok] - want[ok])) <= 1e-10
    # through the row's class, and through freq_path_spline
    row = _rx.NativePathFn(t, f, w, knots)
    assert isinstance(row, track.PathFn)
    assert np.max(np.abs(row(x[ok]) - want[ok])) <= 1e-10
    if w is not None and knots == track.KNOTS_S:
        fp = qt.FreqPath(t_s=np.asarray(t, float), f_hz=np.asarray(f, float), weight=np.asarray(w, float))
        assert np.max(np.abs(tb.freq_path_spline(fp)(x[ok]) - track.freq_path_spline(fp)(x[ok]))) <= 1e-10
    # scalars
    for s in (ref.lo - 7.0, ref.lo, 0.5 * (ref.lo + ref.hi), ref.hi, ref.hi + 7.0):
        assert abs(float(nat(s)) - float(ref(s))) <= 1e-10


@pytest.mark.parametrize("args", [
    ([], [], None, 10.0),                              # IndexError
    ([np.nan], [1500.0], None, 10.0),                  # ValueError: int(nan // knots)
    ([np.nan, np.nan], [1.0, 2.0], None, 10.0),        # LinAlgError from polyfit
    ([0.0, 0.0], [1.0, 2.0], None, 10.0),              # LinAlgError: x all zero
    ([1.0, 2.0], [1.0, 2.0], None, 0.0),               # ZeroDivisionError
    ([0.0, 1.0], [1.0, 2.0], [np.nan, 1.0], 10.0),     # LinAlgError (a NaN weight)
    ([0.0, 1e300], [1.0, 2.0], None, 1e-300),          # OverflowError
    ([5.0], [1.0], None, 10.0),
])
def test_pathfn_refusals(tb, args):
    want = _outcome(track.PathFn, *args)
    got = _outcome(tb.PathFn, *args)
    assert got[0] == want[0], (got, want)
    if got[0] != "ok":
        assert got[1] == want[1]


def test_pathfn_length_mismatch_is_refused(tb):
    """Python indexes f and w by t's order and so drops (or IndexErrors on)
    a different length; the port refuses it outright."""
    with pytest.raises(ValueError, match="same length"):
        tb.PathFn([0.0, 1.0, 2.0], [1.0, 2.0, 3.0, 4.0])
    with pytest.raises(ValueError, match="same length"):
        tb.PathFn([0.0, 1.0, 2.0], [1.0, 2.0, 3.0], [1.0])


def test_native_pathfn_is_the_reference_class(tb):
    t = np.linspace(0, 200, 50)
    fn = _rx.NativePathFn(t, 1500 + np.sin(t / 30))
    ref = track.PathFn(t, 1500 + np.sin(t / 30))
    x = np.linspace(-50, 250, 301)
    for other in (pickle.loads(pickle.dumps(fn)), copy.deepcopy(fn), copy.copy(fn),
                  pickle.loads(pickle.dumps(fn._native))):
        _same(other(x), fn(x))
    assert _normwise(fn._line, ref._line) <= 1e-13
    assert fn.lo == ref.lo and fn.hi == ref.hi
    np.testing.assert_allclose(fn._spl.c, ref._spl.c, rtol=0, atol=1e-10)
    assert np.max(np.abs(fn._inside(x) - ref._inside(x))) <= 1e-10
    # rx_types takes a reference PathFn without refitting: same values, same parts
    back = tb.roundtrip(ref, "PathFn")
    assert np.max(np.abs(back(x) - ref(x))) <= 1e-10
    line_only = track.PathFn([0.0, 1.0, 2.0], [1500.0, 1500.5, 1501.5])
    _same(tb.roundtrip(line_only, "PathFn")(x), line_only(x))      # polyval: the same doubles


# --- the narrow channel and the records -----------------------------------------------------------


def _chan(rng, psd=False, keep=True):
    n = 2000
    c = track.ChanCapture(
        ch=(rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(np.complex64), fs=C.CH_FS,
        t0_index=1234.5, f_mix=Fraction(96017, 64),
        keep=rng.uniform(0, 1, n).astype(np.float32) if keep else None, exclude_hz=(1490.0, 1512.5))
    if psd:
        c._psd = (np.linspace(0, 8, 5), rng.uniform(1, 2, 5))
    return c


def _same_record(a, b, path=""):
    import dataclasses

    if isinstance(a, track.PathFn):
        assert isinstance(b, track.PathFn)
        assert a.lo == b.lo and a.hi == b.hi
        x = np.linspace(a.lo - 30, a.hi + 30, 101)
        _same(np.asarray(a(x)), np.asarray(b(x)))
        return
    if dataclasses.is_dataclass(a) and not isinstance(a, type):
        assert type(a) is type(b), (path, type(a), type(b))
        for f in dataclasses.fields(a):
            _same_record(getattr(a, f.name), getattr(b, f.name), f"{path}.{f.name}")
        return
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        _same(a, b)
        return
    if isinstance(a, (tuple, list)):
        assert type(a) is type(b) and len(a) == len(b), (path, a, b)
        for i, (x, y) in enumerate(zip(a, b)):
            _same_record(x, y, f"{path}[{i}]")
        return
    if isinstance(a, float) and math.isnan(a):
        assert isinstance(b, float) and math.isnan(b), path
        return
    assert type(a) is type(b), (path, type(a), type(b))
    assert a == b, (path, a, b)


def test_chan_capture(tb):
    rng = np.random.default_rng(4)
    for psd in (False, True):
        for keep in (False, True):
            c = _chan(rng, psd, keep)
            _same_record(_rx.to_ref(tb.roundtrip(c)), c)
    c = _chan(rng)
    n = np.r_[np.arange(-10, 30), 1234.5, 1e6]
    _same(tb.chan_t_s(c, n), c.t_s(n))
    _same(tb.chan_t_s(c, 7), c.t_s(7))
    assert tb.chan_mix_audio_hz(c) == c.mix_audio_hz
    c.f_mix = Fraction(-1, 3)
    assert tb.chan_mix_audio_hz(c) == c.mix_audio_hz


def test_mf_table(tb):
    for t_sym in (C.T_SYM, L_T_SYM):
        ref = track.mf_table(t_sym)
        _same_record(_rx.to_ref(tb.roundtrip(ref)), ref)
    half = 3
    good = np.linspace(0, 1, 2 * half * track.MF_RES + 1)
    ok = _rx.to_ref(tb.mf_table_check(C.T_SYM, 8.0, half, good, 1.0))
    _same_record(ok, track.MfTable(C.T_SYM, 8.0, half, good, 1.0))
    cplx = good * np.exp(1j * good)
    _same_record(_rx.to_ref(tb.mf_table_check(C.T_SYM, 8.0, half, cplx, 1.0)),
                 track.MfTable(C.T_SYM, 8.0, half, cplx, 1.0))
    for bad in (good[:-1], np.zeros(0)):
        want = _outcome(track.MfTable, C.T_SYM, 8.0, half, bad, 1.0)
        got = _outcome(tb.mf_table_check, C.T_SYM, 8.0, half, bad, 1.0)
        assert got == want


def test_known_and_classes(tb):
    spec = frame.SHORT
    k = track.known_symbols(spec)
    _same_record(_rx.to_ref(tb.roundtrip(k, "Known")), k)
    kc = track._Known(pos=np.arange(5, dtype=np.int64), val=np.exp(1j * np.arange(5.0)))
    _same_record(_rx.to_ref(tb.roundtrip(kc, "Known")), kc)
    rng = np.random.default_rng(5)
    hdr = rng.integers(0, 2, 2474).astype(np.uint8)      # the polar codeword
    for cl in (track.make_classes(spec), track.make_classes(spec, hdr_bits=hdr),
               track.make_classes(frame.get("full"), hdr_bits=rng.integers(0, 2, 2474).astype(np.uint8))):
        _same_record(_rx.to_ref(tb.roundtrip(cl)), cl)
    lspec = lframe.get("l-short")
    for hb in (None, rng.integers(0, 2, 2474).astype(np.uint8)):
        lc = ltrack.make_classes_l(lspec, hdr_bits=hb)
        _same_record(_rx.to_ref(tb.roundtrip(lc)), lc)


def _track_result(rng, cls, spec, filled=True):
    n = 300
    pos = np.arange(-40, n - 40, dtype=np.int64)
    t = np.linspace(-30, 200, 40)
    fields = dict(
        spec=spec, chan=_chan(rng, psd=filled), timing=qt.Timing(1234.5, 3.0, 0.5, 9.0, np.eye(3) * 0.1),
        freq=track.PathFn(t, 1500 + 0.01 * t), pos=pos,
        z_d=rng.standard_normal(2000) + 1j * rng.standard_normal(2000),
        m=(rng.standard_normal(n) + 1j) if filled else None,
        c=rng.standard_normal(n) + 0j if filled or cls is track.TrackResult else None,
        u=(rng.standard_normal(n) + 2j) if filled else None,
        P=rng.uniform(0, 1, n) if filled else None, N=rng.uniform(0, 1, n) if filled else None,
        psi=rng.uniform(0, 1, n) if filled else None,
        meas=rng.uniform(0, 1, n) > 0.5 if filled or cls is track.TrackResult else None,
        pu=rng.uniform(0, 1, n) if filled else None,
        q=1e-6, loglik=-123.5, ook=filled and cls is track.TrackResult, lead_in_s=8.0,
        z_ref=float("nan") if not filled else 12.5, df_max=0.25,
        keying=rng.integers(0, 2, 40).astype(np.uint8) if filled else None,
        report=qt.TrackReport(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, True, False) if filled else None,
    )
    if cls is track.TrackResult:
        fields["classes"] = track.make_classes(spec) if filled else None
    else:
        fields["classes"] = ltrack.make_classes_l(spec) if filled else None
        if filled:
            fields.update(m_pl=rng.standard_normal(n) + 1j, c_pl=rng.standard_normal(n) + 0j,
                          N_pl=rng.uniform(0, 1, n), psi_pl=rng.uniform(0, 1, n),
                          meas_pl=rng.uniform(0, 1, n) > 0.5, delta=0.5 - 0.25j,
                          u_pl=rng.standard_normal(n) + 0j, P_pl=rng.uniform(0, 1, n), resid_ratio=0.75)
    return cls(**fields)


@pytest.mark.parametrize("filled", [True, False])
def test_track_results_both_ways(tb, filled):
    rng = np.random.default_rng(6)
    for cls, spec in ((track.TrackResult, frame.SHORT), (ltrack.LTrackResult, lframe.get("l-short"))):
        tr = _track_result(rng, cls, spec, filled)
        back = _rx.to_ref(tb.roundtrip(tr))
        assert back.spec is spec                       # the module's own preset object
        _same_record(back, tr)
        info = tb.track_result_info(tr)
        assert info["p_lo"] == tr.p_lo
        _same(info["gi"], tr.gi(tr.pos))
        assert info["doppler_hz"] == tr.doppler_hz
        _same(info["t_pos_s"], tr.t_pos_s())
        p = np.array([-3, 0, 17, 250])
        info = tb.track_result_info(tr, p)
        _same(info["gi"], tr.gi(p))
        _same(info["t_pos_s"], tr.t_pos_s(p))
        assert info["is_l"] == (cls is ltrack.LTrackResult)

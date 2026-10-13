"""Parity of native/core/math (sstvae_native.qrss.math) with numpy and scipy.

The receiver's modules lean on a handful of numpy/scipy numerics; the
port has its own (core/math), and this holds them against the libraries
they stand in for: median, mean, average, interp and polyval to the bit; polyfit,
lstsq, solve, inv, eigh and eigvalsh to ~1e-12 (LAPACK's blocking and
summation orders are not reproduced, so the last bits differ);
make_lsq_spline to ~1e-10 of its coefficients, with scipy's refusals
(type and text); BSpline evaluation and derivative; LAPACK's dlartg
to the bit.
"""

import warnings

import numpy as np
import pytest

from qrss_native import require

interpolate = pytest.importorskip("scipy.interpolate")


@pytest.fixture(scope="module")
def nm(native, request):
    if request.config.getoption("--native-qrss", default=None):
        pytest.skip("--native-qrss replaces the reference this file compares against")
    return require(native, "math")


def _same(a, b):
    """Bit for bit, dtype and shape included (any NaN equals any NaN)."""
    a, b = np.asarray(a), np.asarray(b)
    assert a.dtype == b.dtype, (a.dtype, b.dtype)
    assert a.shape == b.shape, (a.shape, b.shape)
    an, bn = np.isnan(a), np.isnan(b)
    assert np.array_equal(an, bn)
    assert a[~an].tobytes() == b[~bn].tobytes(), (a, b)


def _close(a, b, rel, what=""):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    assert a.shape == b.shape, (what, a.shape, b.shape)
    assert np.array_equal(np.isnan(a), np.isnan(b)), what
    ok = ~np.isnan(a)
    scale = max(1.0, float(np.max(np.abs(b[ok])))) if ok.any() else 1.0
    err = float(np.max(np.abs(a[ok] - b[ok]))) / scale if ok.any() else 0.0
    assert err <= rel, (what, err)
    return err


def _outcome(fn, *args, **kw):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = fn(*args, **kw)
    except (ValueError, TypeError, IndexError, np.linalg.LinAlgError) as e:
        return type(e), str(e)
    return "ok", out


def _arrays(seed):
    rng = np.random.default_rng(seed)
    out = []
    for n in (0, 1, 2, 3, 4, 5, 10, 101, 1000, 1001):
        x = rng.standard_normal(n) * 10 ** rng.uniform(-3, 3)
        out.append(x)
        if n:
            y = x.copy()
            y[rng.integers(n)] = -0.0
            out.append(y)
            out.append(np.round(x))            # ties
    out.append(np.array([np.nan, 1.0, 2.0]))
    out.append(np.array([np.inf, -np.inf, 1.0]))
    out.append(np.array([np.inf, 1.0, 2.0]))
    out.append(np.array([-0.0]))
    out.append(np.array([0.0, -0.0]))
    return out


# --- median, mean -------------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_median_and_mean_bit_exact(nm, dtype):
    for x in _arrays(1):
        x = x.astype(dtype)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            want_med, want_mean = np.median(x), np.mean(x)
        _same(nm.median(x), want_med)
        _same(nm.mean(x), want_mean)


def test_average_bit_exact(nm):
    """np.average(a, weights=w), as track/ltrack's timing fit and acquire's
    drift fit call it: pairwise sums of a * w and of w."""
    rng = np.random.default_rng(12)
    for x in _arrays(4):
        for w in (rng.uniform(0, 5, len(x)), rng.uniform(16, 900, len(x)), np.round(rng.uniform(0, 3, len(x)))):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    want = ("ok", np.average(x, weights=w))
                except ZeroDivisionError as e:
                    want = (ZeroDivisionError, str(e))
            try:
                got = ("ok", nm.average(x, w))
            except ZeroDivisionError as e:
                got = (ZeroDivisionError, str(e))
            assert got[0] == want[0], (x, w, got, want)
            if got[0] == "ok":
                _same(got[1], want[1])
            else:
                assert got[1] == want[1]
    # a NaN weight sum is not zero: it passes through
    _same(nm.average([1.0, 2.0], [np.nan, 1.0]), np.average([1.0, 2.0], weights=[np.nan, 1.0]))
    with pytest.raises(TypeError) as want:
        np.average([1.0, 2.0], weights=[1.0])
    with pytest.raises(TypeError) as got:
        nm.average([1.0, 2.0], [1.0])
    assert str(got.value) == str(want.value)


# --- interp -------------------------------------------------------------------------------------


def test_interp_bit_exact(nm):
    rng = np.random.default_rng(2)
    for n in (1, 2, 3, 5, 17, 200):
        xp = np.sort(rng.uniform(-50, 50, n))
        if n > 3:
            xp[2] = xp[1]                      # a repeated sample point
        fp = rng.standard_normal(n)
        x = np.concatenate([rng.uniform(-80, 80, 300), xp, [np.nan, np.inf, -np.inf, -0.0]])
        for left, right in ((None, None), (-7.0, 9.0), (np.nan, None)):
            _same(nm.interp(x, xp, fp, left, right), np.interp(x, xp, fp, left, right))
        _same(nm.interp(3.25, xp, fp), np.interp(3.25, xp, fp))
        _same(nm.interp(x.reshape(-1, 2) if len(x) % 2 == 0 else x, xp, fp),
              np.interp(x.reshape(-1, 2) if len(x) % 2 == 0 else x, xp, fp))
    # NaN sample values and inf slopes
    xp = np.array([0.0, 1.0, 2.0, 3.0])
    fp = np.array([0.0, np.inf, 1.0, np.nan])
    x = np.linspace(-1, 4, 41)
    _same(nm.interp(x, xp, fp), np.interp(x, xp, fp))


def test_interp_refusals(nm):
    for args in (([1.0], [], []), ([1.0], [0.0, 1.0], [1.0])):
        assert _outcome(nm.interp, *args) == _outcome(np.interp, *args)


# --- polyfit, polyval ---------------------------------------------------------------------------


def test_polyfit_matches_numpy(nm):
    rng = np.random.default_rng(3)
    worst = 0.0
    for trial in range(200):
        n = int(rng.integers(2, 60))
        deg = int(rng.integers(0, min(n, 4)))
        x = np.sort(rng.uniform(-100, 300, n))
        y = 1500 + rng.standard_normal(n) * 0.3 + 0.01 * x
        w = None if trial % 2 else rng.uniform(0, 4, n)
        want = np.polyfit(x, y, deg, w=w)
        got = nm.polyfit(x, y, deg, None, w)
        worst = max(worst, _close(got, want, 1e-11, f"trial {trial}"))
    assert worst < 1e-11


@pytest.mark.parametrize("x,y,deg,w", [
    ([1.0], [1.0], 1, None),                        # one point: min-norm
    ([3.0, 3.0], [1.0, 2.0], 1, None),              # x all equal
    ([0.0, 0.0], [1.0, 2.0], 1, None),              # x all zero: LinAlgError
    ([0.0, 1.0], [np.nan, 2.0], 1, None),           # NaN y: NaN coefficients
    ([0.0, 1.0], [np.inf, 2.0], 1, None),
    ([np.nan, 1.0], [1.0, 2.0], 1, None),           # NaN x: LinAlgError
    ([0.0, 1.0, 2.0], [1.0, 2.0, 4.0], 1, [0.0, 0.0, 0.0]),
    ([0.0, 1.0, 2.0], [1.0, 2.0, 4.0], -1, None),   # ValueError
    ([], [], 1, None),                              # TypeError
    ([0.0, 1.0], [1.0], 1, None),                   # TypeError
    ([0.0, 1.0], [1.0, 2.0], 1, [1.0]),             # TypeError
])
def test_polyfit_edges(nm, x, y, deg, w):
    got = _outcome(nm.polyfit, x, y, deg, None, w)
    want = _outcome(np.polyfit, np.asarray(x, dtype=float), np.asarray(y, dtype=float), deg, w=w)
    assert got[0] == want[0], (got, want)
    if got[0] == "ok":
        _close(got[1], want[1], 1e-12)
    else:
        assert got[1] == want[1]


def test_polyval_bit_exact(nm):
    rng = np.random.default_rng(4)
    x = np.concatenate([rng.uniform(-1e3, 1e3, 500), [np.inf, -np.inf, np.nan, 0.0, -0.0]])
    for deg in range(5):
        p = rng.standard_normal(deg + 1)
        _same(nm.polyval(p, x), np.polyval(p, x))
        _same(nm.polyval(p, 2.5), np.polyval(p, np.asarray(2.5)))


# --- lstsq, solve, inv --------------------------------------------------------------------------


def test_lstsq_matches_numpy(nm):
    rng = np.random.default_rng(5)
    for m, n in ((5, 3), (3, 3), (2, 4), (40, 2), (1, 2), (6, 1), (1, 1)):
        for trial in range(20):
            a = rng.standard_normal((m, n)) * 10 ** rng.uniform(-2, 2, n)
            b = rng.standard_normal(m)
            want = np.linalg.lstsq(a, b, rcond=None)
            got = nm.lstsq(a, b)
            _close(got[0], want[0], 1e-10, (m, n, "x"))
            assert got[2] == want[2]
            _close(got[3], want[3], 1e-12, (m, n, "s"))
            assert np.asarray(got[1]).shape == np.asarray(want[1]).shape
            if np.size(want[1]):
                _close(got[1], want[1], 1e-9, (m, n, "residuals"))


def test_lstsq_edges(nm):
    a = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])     # rank 1: min-norm
    b = np.array([1.0, 2.0, 3.0])
    want, got = np.linalg.lstsq(a, b, rcond=None), nm.lstsq(a, b)
    _close(got[0], want[0], 1e-12)
    assert got[2] == want[2] == 1
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        want = np.linalg.lstsq(np.eye(2), np.array([np.nan, 1.0]), rcond=None)
    got = nm.lstsq(np.eye(2), np.array([np.nan, 1.0]))
    assert np.all(np.isnan(got[0])) and np.all(np.isnan(want[0])) and got[2] == want[2]
    for bad in (np.array([[np.nan, 1.0], [0.0, 1.0]]), np.array([[np.inf, 1.0], [0.0, 1.0]])):
        assert _outcome(nm.lstsq, bad, np.ones(2)) == _outcome(np.linalg.lstsq, bad, np.ones(2), rcond=None)
    # rcond: an explicit value, and LAPACK's "<= 0 or >= 1 means eps"
    a = np.diag([1.0, 1e-9, 1e-17])
    for rc in (1e-8, 1e-12, -1.0, 2.0):
        assert nm.lstsq(a, np.ones(3), rc)[2] == np.linalg.lstsq(a, np.ones(3), rcond=rc)[2]
    # ... whose eps is DLAMCH's 2^-53, not np.finfo's 2^-52: 1.5e-16 sits between them
    a = np.diag([1.0, 1.5e-16])
    for rc in (-1.0, 0.0, 1.0, 2.3e-16):
        assert nm.lstsq(a, np.ones(2), rc)[2] == np.linalg.lstsq(a, np.ones(2), rcond=rc)[2], rc


def test_solve_and_inv_match_numpy(nm):
    rng = np.random.default_rng(6)
    for n in (1, 2, 3, 4, 6):
        for _ in range(20):
            a = rng.standard_normal((n, n)) + n * np.eye(n)
            b = rng.standard_normal(n)
            _close(nm.solve(a, b), np.linalg.solve(a, b), 1e-12)
            _close(nm.inv(a), np.linalg.inv(a), 1e-12)
    sing = np.array([[1.0, 2.0], [2.0, 4.0]])
    assert _outcome(nm.solve, sing, np.ones(2)) == _outcome(np.linalg.solve, sing, np.ones(2))
    assert _outcome(nm.inv, sing) == _outcome(np.linalg.inv, sing)
    assert _outcome(nm.inv, np.zeros((2, 3))) == _outcome(np.linalg.inv, np.zeros((2, 3)))
    # NaN in: NaN out, no error. Which entries come out NaN is OpenBLAS's
    # kernel behaviour (its dscal by 1/NaN leaves a zero a zero: [nan, 1]
    # here, where plain LU arithmetic -- the port's -- gives [nan, nan]).
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        nan_a = np.array([[np.nan, 1.0], [0.0, 1.0]])
        assert np.any(np.isnan(nm.solve(nan_a, np.ones(2))))
        assert np.any(np.isnan(np.linalg.solve(nan_a, np.ones(2))))


# --- eigh, eigvalsh -----------------------------------------------------------------------------


def test_eigh_matches_numpy(nm):
    rng = np.random.default_rng(7)
    for n in (1, 2, 3, 4, 6):
        for _ in range(30):
            g = rng.standard_normal((n, n)) * 10 ** rng.uniform(-3, 3)
            a = g @ g.T if n > 1 else g * g
            a[np.triu_indices(n, 1)] = rng.standard_normal(n * (n - 1) // 2)   # ignored: lower is read
            want_w, want_v = np.linalg.eigh(a)
            got_w, got_v = nm.eigh(a)
            scale = max(1e-300, float(np.max(np.abs(want_w))))
            assert np.max(np.abs(got_w - want_w)) / scale < 1e-12
            _close(nm.eigvalsh(a), np.linalg.eigvalsh(a), 1e-12 * scale)
            # eigenvectors: up to sign, where the eigenvalue is well separated
            low = np.tril(a) + np.tril(a, -1).T
            assert np.max(np.abs(low @ got_v - got_v * got_w)) / scale < 1e-12
            gaps = np.diff(want_w)
            for i in range(n):
                sep = min([abs(g_) for g_ in (gaps[i - 1] if i else np.inf, gaps[i] if i < n - 1 else np.inf)])
                if sep > 1e-6 * scale:
                    assert abs(abs(got_v[:, i] @ want_v[:, i]) - 1.0) < 1e-8


def test_eigh_non_finite(nm):
    a = np.array([[1.0, 0.0], [np.nan, 1.0]])
    assert np.all(np.isnan(nm.eigvalsh(a))) and np.all(np.isnan(np.linalg.eigvalsh(a)))
    assert np.all(np.isnan(nm.eigh(a)[0]))
    upper_only = np.array([[1.0, np.nan], [0.0, 1.0]])           # the upper triangle is not read
    _same(nm.eigvalsh(upper_only), np.linalg.eigvalsh(upper_only))
    assert nm.eigvalsh(np.zeros((0, 0))).shape == (0,)


# --- make_lsq_spline, BSpline -------------------------------------------------------------------


def _knots(lo, hi, step):
    n_int = int((hi - lo) // step)
    inner = lo + step * np.arange(1, n_int + 1)
    inner = inner[inner < hi - 0.25 * step]
    return np.concatenate([[lo] * 4, inner, [hi] * 4])


def test_make_lsq_spline_matches_scipy(nm):
    rng = np.random.default_rng(8)
    worst = 0.0
    for trial in range(150):
        n = int(rng.integers(8, 400))
        x = np.sort(rng.uniform(-200, 600, n))
        y = 1500 + np.cumsum(rng.standard_normal(n)) * 0.01
        w = np.sqrt(rng.uniform(0, 3, n) + 1e-9) if trial % 3 else None
        t = _knots(x[0], x[-1], float(rng.choice([10.0, 30.0, 60.0])))
        want = _outcome(interpolate.make_lsq_spline, x, y, t, k=3, w=w)
        got = _outcome(nm.make_lsq_spline, x, y, t, 3, w)
        assert got[0] == want[0], (trial, got, want)
        if got[0] != "ok":
            assert got[1] == want[1]
            continue
        gt, gc, gk = got[1]
        _same(gt, want[1].t)
        assert gk == want[1].k
        assert np.array_equal(np.isfinite(gc), np.isfinite(want[1].c))
        if np.all(np.isfinite(want[1].c)):
            worst = max(worst, _close(gc, want[1].c, 1e-10, trial))
            xs = np.linspace(x[0] - 50, x[-1] + 50, 333)
            _close(nm.bspline_eval(gt, gc, gk, xs), want[1](xs), 1e-10)
    assert worst < 1e-10


def test_make_lsq_spline_data_gaps(nm):
    """Knot intervals with no data (Schoenberg-Whitney fails): scipy's QR
    then gives non-finite coefficients -- or, when a row's offset passes
    its index, finite wrong ones -- and the port must give the same."""
    rng = np.random.default_rng(9)
    agree = 0
    for trial in range(80):
        a = np.sort(rng.uniform(0, 40, int(rng.integers(4, 30))))
        b = np.sort(rng.uniform(rng.uniform(60, 200), 300, int(rng.integers(4, 30))))
        x = np.concatenate([a, b])
        y = 1500 + 0.01 * x + rng.standard_normal(len(x)) * 0.05
        t = _knots(x[0], x[-1], 10.0)
        want = _outcome(interpolate.make_lsq_spline, x, y, t, k=3)
        got = _outcome(nm.make_lsq_spline, x, y, t, 3)
        assert got[0] == want[0], (trial, got, want)
        if got[0] == "ok":
            assert np.array_equal(np.isfinite(got[1][1]), np.isfinite(want[1].c)), trial
            if np.all(np.isfinite(want[1].c)):
                _close(got[1][1], want[1].c, 1e-8, trial)
            agree += 1
    assert agree > 0


@pytest.mark.parametrize("case", ["nan_x", "inf_w", "few", "neg_k", "t_desc", "y_len", "oob_lo",
                                  "oob_hi", "w_len", "x_desc", "nc"])
def test_make_lsq_spline_refusals(nm, case):
    x = np.linspace(0.0, 10.0, 20)
    y = np.sin(x)
    t = np.r_[[0.0] * 4, 3.0, 5.0, [10.0] * 4]
    w = np.ones_like(x)
    k = 3
    if case == "nan_x":
        x = x.copy()
        x[3] = np.nan
    elif case == "inf_w":
        w = w.copy()
        w[0] = np.inf
    elif case == "few":
        x, y, w = x[:3], y[:3], w[:3]
    elif case == "neg_k":
        k = -1
    elif case == "t_desc":
        t = np.r_[[0.0] * 4, 5.0, 3.0, [10.0] * 4]
    elif case == "y_len":
        y = y[:-1]
    elif case == "oob_lo":
        x = np.r_[-1.0, x[1:]]
    elif case == "oob_hi":
        x = np.r_[x[:-1], 11.0]
    elif case == "w_len":
        w = w[:-1]
    elif case == "x_desc":
        x = x.copy()
        x[[4, 5]] = x[[5, 4]]
    elif case == "nc":
        x, y, w = x[:5], y[:5], w[:5]
        t = np.r_[[0.0] * 4, 1.0, 2.0, 3.0, [10.0] * 4]
    want = _outcome(interpolate.make_lsq_spline, x, y, t, k=k, w=w)
    got = _outcome(nm.make_lsq_spline, x, y, t, k, w)
    assert got[0] == want[0], (got, want)
    if case.startswith("oob"):
        assert got[1].startswith("Out of bounds w/ x = ")   # the array's text is approximate
    else:
        assert got[1] == want[1]


def test_bspline_eval_and_derivative(nm):
    rng = np.random.default_rng(10)
    for k in (1, 2, 3):
        for _ in range(20):
            n_in = int(rng.integers(0, 8))
            t = np.r_[[0.0] * (k + 1), np.sort(rng.uniform(0.5, 9.5, n_in)), [10.0] * (k + 1)]
            c = rng.standard_normal(len(t) - k - 1)
            ref = interpolate.BSpline(t, c, k)
            xs = np.r_[np.linspace(-3, 13, 401), t, np.nan]
            _close(nm.bspline_eval(t, c, k, xs), ref(xs), 1e-13)
            for nu in range(1, k + 1):
                dt, dc, dk = nm.bspline_derivative(t, c, k, nu)
                d = ref.derivative(nu)
                _same(dt, d.t)
                assert dk == d.k
                _close(dc, d.c, 1e-12)
                _close(nm.bspline_eval(dt, dc, dk, xs), d(xs), 1e-11)
    # internal repeated knots: not differentiable to that order
    t = np.r_[[0.0] * 4, 5.0, 5.0, 5.0, 5.0, [10.0] * 4]
    c = rng.standard_normal(len(t) - 4)
    want = _outcome(interpolate.BSpline(t, c, 3).derivative, 1)
    got = _outcome(nm.bspline_derivative, t, c, 3, 1)
    assert got[0] == want[0] and (got[0] == "ok" or got[1] == want[1]), (got, want)
    # splder's errstate(invalid, divide): an infinite coefficient over a zero-width
    # span is inf/0, an exact inf with no divide-by-zero flag, so it passes; inf - inf
    # and 0/0 raise; a NaN passes quietly
    for c in ([1, 2, 3, 1, np.inf, 1, 2, 3], [1, 2, 3, np.inf, 1, 1, 2, 3], [1, 2, 3, np.nan, 1, 1, 2, 3],
              [1, 2, 3, 1, 1, 1, 2, 3], [1, 2, 3, np.inf, np.inf, 1, 2, 3]):
        c = np.array(c, float)
        want = _outcome(interpolate.BSpline(t, c, 3).derivative, 1)
        got = _outcome(nm.bspline_derivative, t, c, 3, 1)
        assert got[0] == want[0], (c, got, want)
        if got[0] == "ok":
            _same(got[1][1], want[1].c)
        else:
            assert got[1] == want[1], (c, got, want)


def test_dlartg_bit_exact(nm):
    lapack = pytest.importorskip("scipy.linalg.lapack")
    rng = np.random.default_rng(11)
    vals = list(rng.standard_normal(200) * 10.0 ** rng.integers(-300, 300, 200))
    vals += [0.0, -0.0, 1.0, -1.0, 1e-310, -1e-310, 1e308, -1e308, 5e-324]
    for _ in range(3000):
        f, g = (float(v) for v in rng.choice(vals, 2))
        want = lapack.dlartg(f, g)
        got = nm.dlartg(f, g)
        _same(np.array(got), np.array(want))


def test_make_lsq_spline_reproduces_scipys_skipped_rows(nm):
    """scipy's QR reduction (qr_reduce) rotates row i only into columns
    offset[i]..min(i, nc)-1, so a row whose band offset is past its own
    index -- a couple of points, then a gap of several knot spacings -- is
    left unreduced and back-substituted as if its band started on the
    diagonal. The result is finite and wrong (residual far above the
    least-squares optimum), and PathFn's non-finite guard does not catch
    it. The port reproduces it exactly; this pins both facts."""
    t = np.r_[[0.0] * 4, np.arange(10.0, 100.0, 10.0), [100.0] * 4]
    rng = np.random.default_rng(1)
    x = np.r_[0.0, 3.5, np.sort(rng.uniform(60.0, 100.0, 30)), 100.0]
    y = np.sin(x / 7) + 0.01 * x
    want = interpolate.make_lsq_spline(x, y, t, k=3)
    got_t, got_c, _ = nm.make_lsq_spline(x, y, t, 3)
    assert np.all(np.isfinite(want.c))
    _same(got_c, want.c)
    design = interpolate.BSpline.design_matrix(x, t, 3).toarray()
    best = np.linalg.lstsq(design, y, rcond=None)[0]
    assert np.sum((design @ want.c - y) ** 2) > 100 * np.sum((design @ best - y) ** 2)

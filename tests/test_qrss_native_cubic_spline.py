"""Parity of native/core/math/cubic_spline (sstvae_native.qrss.cubic_spline)
with scipy.interpolate.CubicSpline: complex values, bc_type "not-a-knot",
extrapolating -- the one shape acquire.preamble_search builds
(`CubicSpline(np.arange(lo, hi), chan.m[lo:hi])`, evaluated at
fractional sample positions).

Bit for bit for every n != 3: the coefficients, the values and the
derivatives (nu 0..4) inside the knots, at them, outside them and at
NaN, on integer, uniform and irregular knots, complex128 and complex64
y. n == 3 is scipy's dense 3x3 solve, held to 1e-13. scipy's ValueError
texts, in its order; the shapes the port does not cover raise
NotImplementedError rather than computing something else.
"""

import warnings

import numpy as np
import pytest

from qrss_native import require

interpolate = pytest.importorskip("scipy.interpolate")


@pytest.fixture(scope="module")
def cs(native):
    return require(native, "cubic_spline")


def _same(a, b, what=""):
    a, b = np.asarray(a), np.asarray(b)
    assert a.dtype == b.dtype, (what, a.dtype, b.dtype)
    assert a.shape == b.shape, (what, a.shape, b.shape)
    an, bn = np.isnan(a), np.isnan(b)
    assert np.array_equal(an, bn), what
    bad = np.flatnonzero(a[~an] != b[~bn])
    assert bad.size == 0, (what, bad.size, a[~an][bad[:3]], b[~bn][bad[:3]])
    assert a[~an].tobytes() == b[~bn].tobytes(), (what, "signed zeros")


def _knots(rng, n, kind):
    if kind == "int":
        lo = int(rng.integers(0, 5000))
        return np.arange(lo, lo + n)
    if kind == "uniform":
        return np.arange(n) * 0.25 + 100.0
    return np.cumsum(rng.uniform(0.01, 3.0, n)) - 7.0


def _points(rng, x):
    span = x[-1] - x[0]
    return np.concatenate([
        rng.uniform(x[0] - 3.0, x[-1] + 3.0, 300),
        x.astype(float),
        (x[:-1] + x[1:]) / 2.0,
        [x[0] - 1e6, x[-1] + 1e6, x[0] - span, np.nan, np.nextafter(x[-1], np.inf), np.nextafter(x[0], -np.inf)],
    ])


@pytest.mark.parametrize("kind", ["int", "uniform", "irregular"])
def test_bitwise(cs, kind):
    rng = np.random.default_rng({"int": 1, "uniform": 2, "irregular": 3}[kind])
    for n in (2, 4, 5, 6, 7, 10, 33, 161, 500, 2000):
        x = _knots(rng, n, kind)
        y = (rng.standard_normal(n) + 1j * rng.standard_normal(n)) * 10 ** rng.uniform(-3, 3)
        for yy in (y, y.astype(np.complex64)):
            ref, got = interpolate.CubicSpline(x, yy), cs.CubicSpline(x, yy)
            _same(got.c, ref.c, f"c n={n} {kind} {yy.dtype}")
            _same(got.x, ref.x, f"x n={n} {kind}")
            e = _points(rng, x)
            for nu in range(5):
                _same(got(e, nu), ref(e, nu), f"nu={nu} n={n} {kind} {yy.dtype}")


def test_preamble_search_shape(cs):
    """chan.m is a band-limited complex64 channel at CH_FS; the spline runs
    over np.arange(lo, hi) (int64) and is read at t0 + tau + t * CH_FS."""
    rng = np.random.default_rng(4)
    for _ in range(5):
        n = int(rng.integers(500, 3000))
        t = np.arange(n)
        m = (np.exp(2j * np.pi * rng.uniform(-0.05, 0.05) * t) * rng.uniform(0.1, 10)
             + 0.1 * (rng.standard_normal(n) + 1j * rng.standard_normal(n))).astype(np.complex64)
        lo = int(rng.integers(0, n // 4))
        hi = int(rng.integers(3 * n // 4, n))
        ref, got = interpolate.CubicSpline(np.arange(lo, hi), m[lo:hi]), cs.CubicSpline(np.arange(lo, hi), m[lo:hi])
        q = lo + 80.3 + np.arange(10 * 66) * 0.25 * 3.79
        _same(got(q.reshape(10, 66)), ref(q.reshape(10, 66)), "preamble_search reads")
        _same(got(float(q[5])), ref(float(q[5])), "a scalar read is 0-d")


def test_three_points(cs):
    """Not-a-knot through three points is one parabola, so the cubic
    coefficient is rounding noise in both (8.9e-16 for one of these), and
    far extrapolation magnifies that noise by x**3: compared near the
    knots, and the coefficients against their own scale."""
    rng = np.random.default_rng(5)
    for _ in range(20):
        x = np.cumsum(rng.uniform(0.1, 3.0, 3))
        y = rng.standard_normal(3) + 1j * rng.standard_normal(3)
        ref, got = interpolate.CubicSpline(x, y), cs.CubicSpline(x, y)
        assert np.max(np.abs(got.c - ref.c)) <= 1e-13 * np.max(np.abs(ref.c))
        e = np.concatenate([rng.uniform(x[0] - 3.0, x[-1] + 3.0, 300), x, [np.nan]])
        for nu in range(4):
            a, b = got(e, nu), ref(e, nu)
            ok = ~np.isnan(b)
            assert np.array_equal(np.isnan(a), np.isnan(b))
            assert np.max(np.abs(a[ok] - b[ok])) <= 1e-13 * max(1.0, np.max(np.abs(b[ok])))


def _outcome(fn, *args, **kw):
    try:
        with warnings.catch_warnings(), np.errstate(all="ignore"):
            warnings.simplefilter("ignore")
            out = fn(*args, **kw)
    except (ValueError, TypeError, IndexError, np.linalg.LinAlgError) as e:
        return type(e), str(e)
    return "ok", out


def test_value_errors_match_scipy(cs):
    y2 = np.array([1 + 0j, 2 + 1j])
    cases = [
        (np.array([0.0]), np.array([1 + 0j])),
        (np.array([0.0, 1.0]), np.array([1 + 0j])),
        (np.array([0.0, np.inf]), y2),
        (np.array([0.0, np.nan]), y2),
        (np.array([0.0, 1.0]), np.array([1 + 0j, np.inf + 0j])),
        (np.array([0.0, 1.0]), np.array([1 + 0j, complex(0, np.nan)])),
        (np.array([1.0, 1.0]), y2),
        (np.array([1.0, 0.0]), y2),
        (np.array([0.0, 1.0, 1.0, 2.0]), np.ones(4, complex)),
        (np.array([0.0, 1.0]), np.array([-1e308 + 0j, 1e308 + 0j])),
        (np.array([0.0, 5e-324]), y2),
        (np.ones((2, 2)), y2),
        (np.array([0.0, 1j]), y2),
        (np.array([0.0, 1.0, 1.0]), np.ones(2, complex)),    # length checked before order
        (np.array([np.inf, 1.0]), np.array([1 + 0j, np.nan])),   # x checked before y
    ]
    for x, y in cases:
        r = _outcome(interpolate.CubicSpline, x, y)
        g = _outcome(cs.CubicSpline, x, y)
        assert r[0] == g[0], (x, y, r, g)
        if r[0] != "ok":
            assert r[1] == g[1], (x, y, r, g)
    s_ref, s = interpolate.CubicSpline([0.0, 1.0, 2.0, 4.0], np.ones(4, complex)), cs.CubicSpline(
        [0.0, 1.0, 2.0, 4.0], np.ones(4, complex))
    r, g = _outcome(s_ref, 0.5, -1), _outcome(s, 0.5, -1)
    assert r[0] == g[0] and r[1] == g[1], (r, g)


def test_unported_shapes_refuse(cs):
    x = np.arange(5.0)
    with pytest.raises(NotImplementedError):
        cs.CubicSpline(x, np.arange(5.0))               # real y
    with pytest.raises(NotImplementedError):
        cs.CubicSpline(x, np.ones(5, complex), bc_type="natural")
    with pytest.raises(NotImplementedError):
        cs.CubicSpline(x, np.ones(5, complex), extrapolate=False)
    with pytest.raises(NotImplementedError):
        cs.CubicSpline(x, np.ones((5, 2), complex))
    s = cs.CubicSpline(x, np.ones(5, complex))
    with pytest.raises(NotImplementedError):
        s(1.5, extrapolate=False)
    assert s.axis == 0 and s.extrapolate is True and s.c.shape == (4, 4)

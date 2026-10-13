"""Parity of native/core/math/special (sstvae_native.qrss.special) with
scipy.special: gammaincc, gammainc, gammainccinv, gammaincinv and the
cephes pieces under them (gammaln, gamma, erfc, the Hurwitz zeta), bit
for bit, over broad grids, the arguments acquire.py actually passes (A1's
and A2's thresholds and A2's per-cell p-values), and scipy's edge cases
(NaN, infinities, domain errors that come out NaN).

The port runs the same xsf/cephes operations over the same glibc libm
calls, so on Linux x86-64 the agreement is exact; on another libm it
would be to that libm's ulp, and this file would say so.
"""

import warnings

import numpy as np
import pytest

from qrss_native import require

special = pytest.importorskip("scipy.special")


@pytest.fixture(scope="module")
def sp(native):
    return require(native, "special")


def _same(a, b, what=""):
    """Bit for bit, dtype and shape included (any NaN equals any NaN)."""
    a, b = np.asarray(a), np.asarray(b)
    assert a.dtype == b.dtype, (what, a.dtype, b.dtype)
    assert a.shape == b.shape, (what, a.shape, b.shape)
    an, bn = np.isnan(a), np.isnan(b)
    assert np.array_equal(an, bn), (what, np.flatnonzero(an != bn)[:5])
    bad = np.flatnonzero(a[~an].view(np.uint64) != b[~bn].view(np.uint64))
    assert bad.size == 0, (what, bad.size, a[~an][bad[:3]], b[~bn][bad[:3]])


def _ref(fn, *args):
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        return fn(*args)


A_GRID = np.concatenate([
    [1e-300, 1e-20, 1e-6, 1e-3, 0.01, 0.1, 0.25, 0.5, 0.75, 0.999, 1.0, 1.001, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0],
    [7.0, 9.999, 10.0, 10.95, 11.0, 19.5, 20.0, 25.0, 30.0, 50.0, 99.0, 100.0, 150.0, 199.5, 200.0, 201.0],
    [300.0, 500.0, 1000.0, 1e4, 1e5, 1e6, 1e10],
])


def test_gammaincc_and_gammainc_grid(sp):
    rng = np.random.default_rng(1)
    for a in A_GRID:
        x = np.concatenate([
            [0.0, 1e-300, 1e-10, 0.5 * a, 0.9 * a, a, 1.1 * a, 2 * a, a + 1, a - 1, 1e300, np.inf],
            rng.uniform(0, 3 * a + 10, 200),
            a * np.exp(rng.uniform(-8, 3, 200)),
            a + np.sqrt(max(a, 1.0)) * rng.standard_normal(200),   # Temme's region
        ])
        x = x[x >= 0]
        _same(sp.gammaincc(a, x), _ref(special.gammaincc, a, x), f"gammaincc a={a}")
        _same(sp.gammainc(a, x), _ref(special.gammainc, a, x), f"gammainc a={a}")


def test_inverses_grid(sp):
    rng = np.random.default_rng(2)
    q = np.concatenate([[0.0, 1e-300, 1e-200, 1e-100, 1e-30, 1e-16, 1e-9, 1e-7, 1e-4, 0.01, 0.1, 0.25],
                        [0.5, 0.75, 0.9, 0.99, 1 - 1e-9, 1 - 1e-16, 1.0],
                        rng.uniform(0, 1, 100), 10.0 ** rng.uniform(-300, 0, 100)])
    for a in A_GRID[A_GRID < 1e9]:
        _same(sp.gammainccinv(a, q), _ref(special.gammainccinv, a, q), f"gammainccinv a={a}")
        _same(sp.gammaincinv(a, q), _ref(special.gammaincinv, a, q), f"gammaincinv a={a}")


def test_the_arguments_acquire_passes(sp):
    """A1: gammainccinv(M^2 / (M + d), pfa); A2: gammainccinv(n_chunks,
    pfa / n_scales) and gammaincc(k, Lambda) for the scales' k."""
    from sstvae.qrss import acquire as A

    shapes = []
    for fs in (A.FE_FS, 500.0, 250.0):
        for M in range(1, 301):
            shapes.append(A.a1_gamma(M, fs)[0])
    shapes = np.array(shapes)
    for pfa in (A.A1_PFA, 1e-3, 1e-6):
        _same(sp.gammainccinv(shapes, pfa), _ref(special.gammainccinv, shapes, pfa), f"A1 pfa={pfa}")
    k = np.arange(1, 41, dtype=float)
    for pfa in (A.A2_PFA / 4, A.A2_PFA, 1e-4 / 3):
        _same(sp.gammainccinv(k, pfa), _ref(special.gammainccinv, k, pfa), f"A2 pfa={pfa}")
    lam = np.concatenate([np.linspace(0, 60, 6001), np.linspace(60, 2000, 2000)])
    for kk in (1, 2, 3, 5, 10, 20, 33):
        _same(sp.gammaincc(float(kk), lam), _ref(special.gammaincc, float(kk), lam), f"gammaincc k={kk}")
        _same(sp.gammaincc_many(float(kk), lam), _ref(special.gammaincc, float(kk), lam), f"many k={kk}")


def test_edges(sp):
    nan, inf = np.nan, np.inf
    a = np.array([0.0, 0.0, 2.0, inf, inf, 2.0, -1.0, 1.0, nan, 1.0, 0.0, inf, -inf, 1e-320])
    x = np.array([1.0, 0.0, 0.0, 3.0, inf, inf, 1.0, -1.0, 1.0, nan, inf, 0.0, 1.0, 1.0])
    _same(sp.gammaincc(a, x), _ref(special.gammaincc, a, x), "gammaincc edges")
    _same(sp.gammainc(a, x), _ref(special.gammainc, a, x), "gammainc edges")
    q = np.array([0.0, 1.0, 1.5, -0.1, 0.5, 0.5, nan, 0.5, 0.0, 1.0])
    aq = np.array([3.0, 3.0, 3.0, 3.0, -1.0, 0.0, 3.0, nan, inf, inf])
    _same(sp.gammainccinv(aq, q), _ref(special.gammainccinv, aq, q), "gammainccinv edges")
    _same(sp.gammaincinv(aq, q), _ref(special.gammaincinv, aq, q), "gammaincinv edges")


def test_shapes_and_types(sp):
    """A 0-d result is an np.float64, as a scipy ufunc returns; arrays
    broadcast and keep their shape."""
    r = sp.gammaincc(2.0, 3.0)
    assert type(r) is np.float64 and r == special.gammaincc(2.0, 3.0)
    assert type(sp.gammainccinv(np.float64(5.0), 1e-4)) is np.float64
    a = np.array([[1.0], [2.0], [3.0]])
    x = np.array([0.5, 1.0, 4.0, 9.0])
    _same(sp.gammaincc(a, x), special.gammaincc(a, x), "broadcast")
    _same(sp.gammaincc(2, np.arange(5)), special.gammaincc(2, np.arange(5)), "integers")
    _same(sp.gammaincc(2.0, np.zeros((0, 3))), special.gammaincc(2.0, np.zeros((0, 3))), "empty")
    with pytest.raises(ValueError):
        sp.gammaincc(np.ones(3), np.ones(4))


def test_cephes_pieces(sp):
    rng = np.random.default_rng(3)
    xs = np.concatenate([rng.uniform(-200, 200, 2000), 10.0 ** rng.uniform(-300, 2.5, 500),
                         [0.0, -0.0, 1.0, 2.0, -1.0, -2.5, 0.5, 1e-300, 171.0, 171.7, 172.0, 1e10, -1e10],
                         [np.inf, -np.inf, np.nan], -np.arange(1, 30) - 1e-12])
    _same(sp.lgam(xs), _ref(special.gammaln, xs), "gammaln")
    _same(sp.gamma(xs), _ref(special.gamma, xs), "gamma")
    e = np.concatenate([rng.uniform(-30, 30, 2000), [0.0, -0.0, 0.5, 0.84375, 1.25, 2.857, 6.0, 26.6, 27.3]])
    _same(sp.erfc(e), _ref(special.erfc, e), "erfc")
    zx = np.concatenate([rng.uniform(1.0, 60.0, 1000), [1.0, 1.0 + 1e-12, 2.0, 0.5, -1.0, 100.0, np.inf]])
    zq = np.concatenate([10.0 ** rng.uniform(-3, 4, 1000), [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]])
    _same(sp.zeta(zx, zq), _ref(special.zeta, zx, zq), "zeta")
    _same(sp.zeta(3.0, np.array([0.0, -1.0, -1.5, 1e-300])),
          _ref(special.zeta, 3.0, np.array([0.0, -1.0, -1.5, 1e-300])), "zeta, q <= 0")

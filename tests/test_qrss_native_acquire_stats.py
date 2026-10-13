"""Parity of native/core/qrss/acquire_stats (sstvae_native.qrss.acquire_stats)
with sstvae/qrss/acquire.py's shared statistics, held side by side in one
process: bit for bit (dtype and shape included) for the STFT frames and
their CFAR normalisation, the band bins, the spectral shape, the A1/A2
thresholds, A2's per-cell p-values and scales, A3's Gumbel table and band
covariance, and the vertex parabola in float64, float32 and NEP 50's mixed
arithmetic; gumbel_p to a few ulps (numpy's SIMD exp/expm1 are not
glibc's); the reference's errors, type and text.

tests/qrss_native/acquire_stats.py substitutes these for acquire's own
functions under --native-qrss=rx, where tests/test_qrss_acquire.py and
tests/test_qrss_lacquire.py then run against the C++.
"""

import numpy as np
import pytest

from qrss_native import require
from sstvae.qrss import acquire as A


@pytest.fixture(scope="module")
def Q(native):
    return require(native, "acquire_stats")


def _same(a, b, what=""):
    a, b = np.asarray(a), np.asarray(b)
    assert a.dtype == b.dtype, (what, a.dtype, b.dtype)
    assert a.shape == b.shape, (what, a.shape, b.shape)
    an, bn = np.isnan(a), np.isnan(b)
    assert np.array_equal(an, bn), what
    assert a[~an].tobytes() == b[~bn].tobytes(), (what, np.count_nonzero(a[~an] != b[~bn]))


def _same_scalar(r, g, what=""):
    assert type(r) is type(g), (what, type(r), type(g))
    _same(r, g, what)


def _outcome(fn, *args, **kw):
    try:
        out = fn(*args, **kw)
    except (ValueError, TypeError, IndexError, ZeroDivisionError) as e:
        return type(e), str(e)
    return "ok", out


def test_scalars(Q):
    for fs in (A.FE_FS, 4000.0, 500, 250):
        assert Q._frame_len(fs) == A._frame_len(fs) and type(Q._frame_len(fs)) is int
        _same(Q.frame_window(fs), A.frame_window(fs), f"frame_window {fs}")
        for M in (1, 2, 3, 11, 50, 100, 333):
            _same(np.array(Q.a1_gamma(M, fs)), np.array(A.a1_gamma(M, fs)), f"a1_gamma {M} {fs}")
            _same_scalar(Q.a1_threshold(M, fs=fs), A.a1_threshold(M, fs=fs), f"a1_threshold {M} {fs}")
            for pfa in (1e-2, 1e-6):
                _same_scalar(Q.a1_threshold(M, pfa, fs), A.a1_threshold(M, pfa, fs), f"a1_threshold {M} {pfa}")
        for n in (0, 1, 2, 3, 5, 76, 321, 1000):
            _same(np.array(Q._band_cov(n, fs)), np.array(A._band_cov(n, fs)), f"band_cov {n} {fs}")
    _same(np.array(Q.a1_gamma(7)), np.array(A.a1_gamma(7)), "a1_gamma default fs")
    for name, args in (("a1_gamma", (11,)), ("a1_gamma", (1, 500)), ("tbd_gumbel", (31,)), ("_band_cov", (5,))):
        r, g = getattr(A, name)(*args), getattr(Q, name)(*args)
        assert [type(v) for v in g] == [type(v) for v in r], (name, r, g)   # np.float64 vs float
    _same(np.array(Q._band_cov(9)), np.array(A._band_cov(9)), "band_cov default fs")
    _same_scalar(Q.a2_threshold(), A.a2_threshold(), "a2_threshold()")
    for pfa in (1e-9, 1e-7, 1e-4, 0.3):
        for nch in (1, 2, 5, 10, 25):
            for ns in (1, 3, 4):
                _same_scalar(Q.a2_threshold(pfa, nch, ns), A.a2_threshold(pfa, nch, ns), f"a2 {pfa} {nch} {ns}")


def test_gumbel(Q):
    worst = 0.0
    for n in (0, 1, 10, 31, 50, 132, 200, 331, 896, 1500, 2000, 5000):
        r, g = A.tbd_gumbel(n), Q.tbd_gumbel(n)
        _same(np.array(g), np.array(r), f"tbd_gumbel {n}")
        assert type(g[0]) is type(r[0]) is float
        for z in (-5.0, 0.0, 5.0, 10.0, 20.0, 50.0, 200.0, np.inf, -np.inf):
            rp, gp = A.gumbel_p(z, n), Q.gumbel_p(z, n)
            assert type(gp) is type(rp)
            if rp == gp:
                continue
            worst = max(worst, abs(gp - rp) / abs(rp))
    assert worst <= 4e-16, worst


def test_parabola(Q):
    rng = np.random.default_rng(1)
    for _ in range(3000):
        v = rng.standard_normal(3) * 10 ** rng.uniform(-3, 3)
        _same_scalar(Q._parabola(*v), A._parabola(*v), "float64")
        _same_scalar(Q._parabola(*map(float, v)), A._parabola(*map(float, v)), "python floats")
        v32 = v.astype(np.float32)
        _same_scalar(Q._parabola(*v32), A._parabola(*v32), "float32")
        lg = np.log(np.maximum(v32, 1e-9))           # carrier_lines' log levels
        _same_scalar(Q._parabola(*lg), A._parabola(*lg), "float32 log")
        mixed = (float(v[0]), v32[1], float(v[2]))
        _same_scalar(Q._parabola(*mixed), A._parabola(*mixed), "mixed float/float32")
    for v in ((1.0, 2.0, 1.0), (1.0, 0.0, 1.0), (0.0, 1.0, 1.9), (np.nan, 1.0, 0.0), (np.inf, 1.0, 0.0),
              (0.0, 0.0, 0.0)):
        _same_scalar(Q._parabola(*v), A._parabola(*v), f"edge {v}")


def test_band_bins(Q):
    for fs in (A.FE_FS, 500, 250, 1000.0):
        for band in (A.BAND_HZ, (1000.0, 1100.0), A.L_BAND_HZ, (1400, 1600)):
            for c in (A.FE_CENTER_HZ, 0.0, 12.5):
                r, g = A.band_bins(fs, band, c), Q.band_bins(fs, band, c)
                _same(g[0], r[0], "sel")
                _same(g[1], r[1], "f")
        r, g = A.band_bins(fs), Q.band_bins(fs)
        _same(g[0], r[0], "sel default")
        _same(g[1], r[1], "f default")


@pytest.mark.parametrize("fs,n", [(4000, 60000), (500, 9000)])
def test_spectrogram_cfar_shape(Q, fs, n):
    rng = np.random.default_rng(fs)
    x = (rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(np.complex64)
    L = A._frame_len(fs)
    starts = np.concatenate([rng.uniform(-L, n, 30), [0.5, 1.5, 2.5, -L - 3, n + 5, n - L / 2, -0.5]])
    sel, _f = A.band_bins(fs)
    for bins in (None, sel, np.arange(-50, 50) % L, np.array([-1, -3, 5]), np.array([], dtype=np.int64)):
        _same(Q.spectrogram(x, fs, starts, bins), A.spectrogram(x, fs, starts, bins), f"spectrogram {fs}")
    _same(Q.spectrogram(x, fs, starts[:7], sel, chunk=3), A.spectrogram(x, fs, starts[:7], sel, chunk=3), "chunk")
    _same(Q.spectrogram(x, fs, np.array([]), sel), A.spectrogram(x, fs, np.array([]), sel), "no frames")
    x128 = x.astype(np.complex128) * 1.0000001
    _same(Q.spectrogram(x128, fs, starts[:5], sel), A.spectrogram(x128, fs, starts[:5], sel), "complex128")
    xr = rng.standard_normal(n)
    _same(Q.spectrogram(xr, fs, starts[:5], sel), A.spectrogram(xr, fs, starts[:5], sel), "real")

    P = A.spectrogram(x, fs, np.arange(0, n - L, L // 2), sel)
    P[3] = 0                                         # a dead frame
    for lb in (A.CFAR_LOCAL_BINS, None, 0, 5, 1000):
        _same(Q.cfar_normalise(P, lb), A.cfar_normalise(P, lb), f"cfar {lb}")
    _same(Q.cfar_normalise(P), A.cfar_normalise(P), "cfar default")
    P64 = P.astype(np.float64)
    _same(Q.cfar_normalise(P64), A.cfar_normalise(P64), "cfar float64")
    E = A.cfar_normalise(P)
    for lb in (A.CFAR_LOCAL_BINS, 1, 7, 2):
        for rows in (None, np.arange(len(E)) % 3 != 0, np.zeros(len(E), bool)):
            _same(Q.spectral_shape(E, lb, rows), A.spectral_shape(E, lb, rows), f"shape {lb}")
    _same(Q.spectral_shape(E), A.spectral_shape(E), "shape default")
    for k in (1, 2, 3, 4):
        _same(Q.spectral_shape(E[:, :k]), A.spectral_shape(E[:, :k]), f"shape {k} bins")
    _same(Q.spectral_shape(E[:, :4].astype(np.float64)), A.spectral_shape(E[:, :4].astype(np.float64)), "f64")


def test_preamble_p(Q):
    rng = np.random.default_rng(3)
    for trial in range(8):
        n_df, n_tau, nc = 21, int(rng.integers(1, 200)), 10
        S = (rng.standard_normal((n_df, n_tau, nc)) + 1j * rng.standard_normal((n_df, n_tau, nc))) \
            * 10 ** rng.uniform(-3, 3)
        var = rng.uniform(0.1, 4, (n_tau, nc)) * np.abs(S).mean() ** 2
        if trial == 1:
            S[3, 0, :] += 50 * np.sqrt(var[0])      # p underflows at every scale
        if trial == 2:
            S[5, 0, 2] = np.nan                     # argmin takes the NaN
        for sc in (A.A2_SCALES, (1,), (3,), (2, 5), (20,), (10, 1)):
            rs, gs = A.preamble_scales(S, var, sc), Q.preamble_scales(S, var, sc)
            assert len(rs) == len(gs)
            for (rk, rl), (gk, gl) in zip(rs, gs):
                assert rk == gk and type(gk) is int
                _same(gl, rl, f"scales {sc} k={rk}")
            rp, gp = A.preamble_p(S, var, sc), Q.preamble_p(S, var, sc)
            _same(gp[0], rp[0], f"p {sc}")
            _same(gp[1], rp[1], f"best {sc}")
            for (rk, rl), (gk, gl) in zip(rp[2], gp[2]):
                assert rk == gk
                _same(gl, rl, f"st {sc}")
        _same(Q.preamble_p(S, var)[0], A.preamble_p(S, var)[0], "default scales")


def test_errors_match(Q):
    x10 = np.zeros(10, np.complex64)
    S = np.ones((2, 3, 10), complex)
    v = np.ones((3, 10))
    E = np.ones((4, 5), np.float32)
    cases = [
        ("a2_threshold", (1e-4, 5, 0)),
        ("tbd_gumbel", (-1,)),
        ("spectrogram", (np.zeros(0, np.complex64), 500, np.array([0.0]))),
        ("spectrogram", (x10, 500, np.array([0.0]), np.array([2000]))),
        ("spectrogram", (x10, 500, np.array([0.0]), np.array([-2001]))),
        ("spectral_shape", (E, 3, np.ones(3, bool))),
        ("preamble_scales", (S, v, (0,))),
        ("preamble_p", (S, v, (0,))),
        ("preamble_p", (S, v, ())),
    ]
    for name, args in cases:
        r, g = _outcome(getattr(A, name), *args), _outcome(getattr(Q, name), *args)
        assert r[0] == g[0], (name, r, g)
        if r[0] != "ok":
            assert r[1] == g[1], (name, r, g)

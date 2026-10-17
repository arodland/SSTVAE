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
    except (ValueError, TypeError, IndexError, ZeroDivisionError, OverflowError) as e:
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



def _layout(a, order):
    """A copy of a whose memory order, slowest axis first, is `order`."""
    return np.ascontiguousarray(a.transpose(order)).transpose(np.argsort(order))


def test_preamble_p_follows_numpys_layout(Q):
    """numpy's sums in preamble_scales run pairwise when Sn = S / sqrt(var)
    comes out with its chunk axis innermost and one term at a time
    otherwise -- the last bit of Lambda depends on S's memory layout, and
    the S acquire._preamble_sums makes (an einsum output) is
    Fortran-ordered. Every axis order of S and var, degenerate lengths,
    complex64, a list, and the real _preamble_sums output, for both kinds."""
    import itertools
    rng = np.random.default_rng(17)
    for shape in ((21, 37, 10), (1, 37, 10), (21, 1, 10), (1, 1, 10), (5, 4, 3), (3, 4, 1)):
        S0 = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
        v0 = rng.uniform(0.5, 2, shape[1:])
        for order in itertools.permutations(range(3)):
            for vorder in ((0, 1), (1, 0)):
                S, var = _layout(S0, order), _layout(v0, vorder)
                for sc in (A.A2_SCALES, (1,), (3,)):
                    rp, gp = A.preamble_p(S, var, sc), Q.preamble_p(S, var, sc)
                    for (rk, rl), (gk, gl) in zip(rp[2], gp[2]):
                        _same(gl, rl, f"layout {shape} {order} {vorder} {sc} k={rk}")
                    _same(gp[0], rp[0], f"p layout {shape} {order} {vorder}")
                    _same(gp[1], rp[1], f"best layout {shape} {order} {vorder}")
    S, var = _layout(S0, (2, 1, 0)).astype(np.complex64), v0
    _same(Q.preamble_scales(S, var)[0][1], A.preamble_scales(S, var)[0][1], "complex64")
    _same(Q.preamble_scales(S.tolist(), var.tolist())[1][1], A.preamble_scales(np.array(S.tolist()), var.tolist())[1][1],
          "lists")

    import test_qrss_acquire as TA
    x, t0 = TA._capture(1650.0, -15.0, shift_s=0.7, seed=3)
    for kind in (A.CE_PRE, A.L_PRE):
        chan = A.preamble_channel(x, t0, 1650.0, 0.0, kind=kind)
        _lam, _taus, S, var = A._preamble_sums(chan)
        assert S.flags.f_contiguous and not S.flags.c_contiguous       # the case this exists for
        rp, gp = A.preamble_p(S, var), Q.preamble_p(S, var)
        _same(gp[0], rp[0], f"real p {kind.waveform}")
        _same(gp[1], rp[1], f"real best {kind.waveform}")
        for (rk, rl), (gk, gl) in zip(rp[2], gp[2]):
            _same(gl, rl, f"real Lambda {kind.waveform} k={rk}")
        S[3, 7, :] *= 1e3                                               # p underflows there
        _same(Q.preamble_p(S, var)[0], A.preamble_p(S, var)[0], f"real p, strong {kind.waveform}")

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
        # review probes: the reference's own order of refusals
        ("spectrogram", (x10, 500, np.array([0.0]), None, 0)),
        ("spectrogram", (x10, 500, np.array([]), None, 0)),
        ("spectrogram", (x10, np.nan, np.array([0.0]), None, 0)),
        ("spectrogram", (x10, -1.0, np.array([0.0]))),
        ("spectrogram", (x10, 500, 0.0)),
        ("spectrogram", (x10, 500, np.array([0.0]), np.int64(3))),
        ("spectrogram", (x10, 500, 0.0, np.int64(3))),
        ("spectral_shape", (E, -4)),
        ("spectral_shape", (E, 2 ** 40)),
        ("cfar_normalise", (E, -1)),
        ("cfar_normalise", (E, 0.5)),
        ("cfar_normalise", (E, 2 ** 40)),
    ]
    for name, args in cases:
        r, g = _outcome(getattr(A, name), *args), _outcome(getattr(Q, name), *args)
        assert r[0] == g[0], (name, r, g)
        if r[0] != "ok":
            assert r[1] == g[1], (name, r, g)
        else:
            _same(g[1], r[1], name)


# --- Review probes: non-finite and odd inputs, the receiver's real shapes,
# --- and acquire's own A1/A2/A3 run end to end on the C++.

def test_spectrogram_nonfinite_samples_and_real_dtypes(Q):
    """numpy windows complex input by the complex product with w promoted
    to w + 0j, so an inf real part makes a NaN imaginary one (and a NaN row
    rather than an inf); real input is windowed as a real product and only
    made complex by the FFT, so the same inf stays inf. Integer, bool,
    float16 and list input widen as numpy's do."""
    rng = np.random.default_rng(11)
    n, fs = 9000, 500
    x = (rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(np.complex64)
    x[100] = np.inf
    x[3000] = complex(1, np.inf)
    x[5000] = complex(-np.inf, 2)
    x[7000] = complex(np.nan, 0)
    starts = np.array([0.0, 50.0, 2500.0, 4100.0, 6000.0, 6500.0, 8000.0])
    sel = np.arange(-40, 40)
    with np.errstate(invalid="ignore", over="ignore"):
        for xx in (x, x.astype(np.complex128), x.real.copy(), x.real.astype(np.float64),
                   x.imag.astype(np.float16)):
            for bins in (None, sel):
                _same(Q.spectrogram(xx, fs, starts, bins), A.spectrogram(xx, fs, starts, bins),
                      f"non-finite {xx.dtype}")
    xi = rng.integers(-30000, 30000, n).astype(np.int16)
    for xx in (xi, xi.astype(np.int64), xi > 0, list(x[200:3000].astype(complex)), [float(v) for v in xi[:3000]]):
        _same(Q.spectrogram(xx, fs, starts[:3], sel), A.spectrogram(xx, fs, starts[:3], sel), "dtype")


def test_spectrogram_extreme_starts(Q):
    """Starts past either end, ones whose idx = st + k wraps int64 in numpy
    (2^63 - 1024 plus k >= 1024), and NaN/inf/out-of-range starts that
    numpy casts to INT64_MIN: every one an all-zero frame, as numpy's."""
    rng = np.random.default_rng(12)
    n, fs = 3000, 250
    x = (rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(np.complex64)
    starts = np.array([1e18, -1e18, 9223372036854774784.0, -9223372036854774784.0, np.nan, np.inf, -np.inf,
                       1e19, -1e19, 0.5, -0.5, 1.5, 2.5, n - 0.5, n + 0.5, -999.5, -1000.5, -1001.5])
    with np.errstate(invalid="ignore"):
        _same(Q.spectrogram(x, fs, starts), A.spectrogram(x, fs, starts), "extreme starts")
        _same(Q.spectrogram(x, fs, list(starts), np.array([0, 1, -1])),
              A.spectrogram(x, fs, list(starts), np.array([0, 1, -1])), "extreme starts, list")


def test_non_finite_fs_is_rounds_error(Q):
    """int(round(fs * 4)) of a NaN/inf fs: ValueError/OverflowError, in
    round()'s words -- and _band_cov must not answer a NaN fs from another
    fs's cached tables (NaN is equivalent to every key of an ordered map)."""
    for fs in (A.FE_FS, 500.0):
        Q._band_cov(5, fs)                           # something in the cache
    for name, args in (("_frame_len", (np.nan,)), ("_frame_len", (np.inf,)), ("frame_window", (np.nan,)),
                       ("frame_window", (-np.inf,)), ("_band_cov", (5, np.nan)), ("_band_cov", (5, np.inf)),
                       ("a1_gamma", (11, np.nan)), ("a1_threshold", (11, 1e-4, np.inf)),
                       ("band_bins", (np.nan,)), ("spectrogram", (np.ones(10, np.complex64), np.nan, [0.0]))):
        r, g = _outcome(getattr(A, name), *args), _outcome(getattr(Q, name), *args)
        assert r[0] == g[0] and r[0] != "ok", (name, args, r, g)
        assert r[1] == g[1], (name, args, r, g)


def test_band_cov_cache_eviction(Q):
    """More distinct fs than the cache holds (8), revisited in another
    order: every answer the reference's, whatever was evicted."""
    fss = [250.0 + 37.5 * i for i in range(13)] + [A.FE_FS]
    for order in (fss, fss[::-1], fss[::3] + fss[1::3] + fss[2::3]):
        for fs in order:
            for nb in (1, 9, 76):
                _same(np.array(Q._band_cov(nb, fs)), np.array(A._band_cov(nb, fs)), f"band_cov {nb} {fs}")


def test_a1_gamma_large_m(Q):
    """M * M is Python's exact product rounded once: no int64 overflow."""
    for M in (0, 1, 10 ** 5, 3037000499, 3037000500, 2 ** 40 + 1, 2 ** 52 - 1):
        with np.errstate(divide="ignore", invalid="ignore"):
            r, g = A.a1_gamma(M), Q.a1_gamma(M)
        _same(np.array(g), np.array(r), f"a1_gamma {M}")


def test_parabola_argument_types(Q):
    """NEP 50 over every mix the reference's callers can pass: Python int
    and float (weak), np.float32, np.float64, 0-d arrays of either."""
    rng = np.random.default_rng(13)
    kinds = (int, float, np.float32, np.float64, lambda v: np.array(v, np.float32), lambda v: np.array(v))
    for _ in range(400):
        v = rng.standard_normal(3) * 10 ** rng.uniform(-2, 2)
        v[1] = abs(v[1]) + 1                        # mostly a peak
        ks = rng.integers(0, len(kinds), 3)
        args = []
        for val, k in zip(v, ks):
            f = kinds[k]
            args.append(f(round(val)) if f is int else f(val))
        _same_scalar(Q._parabola(*args), A._parabola(*args), f"types {[type(a) for a in args]} {args}")


@pytest.mark.filterwarnings("ignore:Mean of empty slice:RuntimeWarning",
                            "ignore:invalid value encountered:RuntimeWarning")
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_cfar_edge_shapes(Q, dtype):
    """cfar_normalise and spectral_shape on empty, one- and two-bin grids,
    odd and even frame counts, frames that are more than half zero (median
    0: a dead frame), all-zero grids, NaN and inf in a frame, and every
    local_bins the reference accepts or refuses."""
    rng = np.random.default_rng(14)
    base = rng.exponential(size=(9, 41)).astype(dtype)
    halfzero = base.copy()
    halfzero[2, :21] = 0                             # median 0
    halfzero[5, :20] = 0                             # median > 0
    nanrow = base.copy()
    nanrow[4, 7] = np.nan
    infrow = base.copy()
    infrow[1, 3] = np.inf
    infrow[6, :] = np.inf
    cases = {"rows0": base[:0], "cols0": base[:, :0], "cols1": base[:, :1], "cols2": base[:, :2],
             "cols3": base[:, :3], "even": base[:8], "one": base[:1], "halfzero": halfzero,
             "zeros": np.zeros_like(base), "nan": nanrow, "inf": infrow}
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        for name, P in cases.items():
            for lb in (A.CFAR_LOCAL_BINS, None, 0, 1, 2, 3, 4, -1, -4, 1000):
                r, g = _outcome(A.cfar_normalise, P, lb), _outcome(Q.cfar_normalise, P, lb)
                assert r[0] == g[0], (name, lb, r, g)
                if r[0] == "ok":
                    _same(g[1], r[1], f"cfar {name} {lb}")
                else:
                    assert r[1] == g[1], (name, lb, r, g)
                for rows in (None, np.arange(len(P)) % 2 == 0, (np.arange(len(P)) % 3).tolist()):
                    r, g = _outcome(A.spectral_shape, P, lb or 1, rows), _outcome(Q.spectral_shape, P, lb or 1, rows)
                    assert r[0] == g[0], (name, lb, r, g)
                    if r[0] == "ok":
                        _same(g[1], r[1], f"shape {name} {lb}")
                    else:
                        assert r[1] == g[1], (name, lb, r, g)


def test_a3_short_shape(Q):
    """tbd_emissions' frames on a SHORT pass at FE_FS: 132-ish 4 s frames,
    the band's 9601 bins, a fractional t0 index -- spectrogram then
    cfar_normalise with the local shape, bit for bit."""
    from sstvae.qrss import frame
    rng = np.random.default_rng(15)
    fs = A.FE_FS
    t0 = 12 * fs + 0.37
    t = A.tbd_frame_starts(frame.SHORT, None, t0, fs)
    n = int(t0 + (t[-1] + A.STFT_FRAME_S) * fs) + 10
    x = ((rng.standard_normal(n) + 1j * rng.standard_normal(n)) / np.sqrt(2)).astype(np.complex64)
    x[::7] *= 1.3                                   # a little structure
    sel, _f = A.band_bins(fs)
    P = A.spectrogram(x, fs, t0 + t * fs, sel)
    _same(Q.spectrogram(x, fs, t0 + t * fs, sel), P, "SHORT spectrogram")
    _same(Q.cfar_normalise(P), A.cfar_normalise(P), "SHORT cfar")


def _deep_same(r, g, what):
    import dataclasses
    if dataclasses.is_dataclass(r) and not isinstance(r, type):
        assert type(g) is type(r), (what, type(r), type(g))
        for f in dataclasses.fields(r):
            if type(r) is A.TbdHit and f.name == "p_value":     # gumbel_p: numpy's SIMD exp/expm1
                a, b = r.p_value, g.p_value
                assert type(b) is type(a) and abs(b - a) <= 4e-16 * abs(a), (what, a, b)
                continue
            _deep_same(getattr(r, f.name), getattr(g, f.name), f"{what}.{f.name}")
    elif isinstance(r, (list, tuple)):
        assert type(g) is type(r) and len(g) == len(r), (what, r, g)
        for i, (a, b) in enumerate(zip(r, g)):
            _deep_same(a, b, f"{what}[{i}]")
    elif isinstance(r, np.ndarray):
        _same(g, r, what)
    elif isinstance(r, float):
        assert type(g) is type(r) and (r == g or (np.isnan(r) and np.isnan(g))), (what, r, g)
    else:
        assert type(g) is type(r) and r == g, (what, r, g)


def test_acquire_end_to_end_on_the_cpp(Q, native, monkeypatch):
    """A1 (carrier_line_stats, carrier_lines), A2 (preamble_search, both
    preamble kinds -- CubicSpline included) and A3 (track_before_detect) on
    a -15 dB SHORT capture, with every row of tests/qrss_native/
    acquire_stats.py assigned over acquire's own names: each result equal,
    field for field, to the Python run's. This is the path the rows exist
    for, through acquire's real callers rather than one function at a time."""
    import test_qrss_acquire as TA
    from qrss_native.acquire_stats import rows
    from sstvae.qrss import frame

    x, t0 = TA._capture(1650.0, -15.0, shift_s=0.7, seed=3)
    noise = TA._noise_fe(len(x), np.random.default_rng(16))

    def run():
        out = []
        for fe in (x, noise):
            out.append(A.carrier_line_stats(fe, t0))
            cands = A.carrier_lines(fe, t0)
            out.append(cands)
            probe = list(cands[:2]) + [TA.CarrierCandidate(1650.0, 0.0, 0.0)]
            for c in probe:
                for kind in (A.CE_PRE, A.L_PRE):
                    out.append(A.preamble_search(fe, t0, c, kind=kind))
            out.append(A.track_before_detect(fe, t0, frame.SHORT, cands, all_hits=True))
        return out

    ref = run()
    table = rows(native.qrss)
    assert all(target == "sstvae.qrss.acquire" for target, _ in table)
    for (_target, name), fn in table.items():
        monkeypatch.setattr(A, name, fn)
    got = run()
    _deep_same(ref, got, "acquire")

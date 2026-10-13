"""Parity of native/core/qrss/render (sstvae_native.qrss.render) with the
arithmetic of sstvae/qrss/render.py: g(W), W~ and the decoder's planes.

decoder_planes is held to the float32 bit pattern (the decoder sees these
bytes), on float32 accumulators as the store holds them and on float64,
every mode, groups heard and unheard, latents never sent, W~ = 0 and
W = 0 inside a sent group; measured equal on all of them. reference_w
and g_shrink are compared exactly (np.median's partition and the same
float64 expression; -0.0, NaN and negative W follow np.maximum).
The codec call (`render`) stays Python and reaches the native planes
through the module.
"""

import numpy as np
import pytest

import qrss_demod_probes as P
from qrss_native import require
from sstvae.qrss import picture, render

NG, GL = render.N_GROUPS, render.GROUP_LATENTS


@pytest.fixture(scope="module")
def rd(native):
    return require(native, "render")


def _bits_equal(a, b):
    a, b = np.asarray(a), np.asarray(b)
    assert a.dtype == b.dtype and a.shape == b.shape, (a.dtype, b.dtype, a.shape, b.shape)
    assert a.tobytes() == b.tobytes()


def _accumulators():
    rng = np.random.default_rng(20261013)
    out = []
    W = rng.uniform(0, 5, (NG, GL)).astype(np.float32)
    S = rng.standard_normal((NG, GL)).astype(np.float32)
    out.append(("random", S, W))
    W2 = W.copy()
    W2[1] = 0.0                                          # group 1 unheard
    W2[2, ::7] = 0.0                                     # holes in a heard group
    out.append(("holes", S, W2))
    W3 = np.zeros((NG, GL), np.float32)
    W3[0, picture.air_to_canonical(0)[: GL // 4]] = 2.0  # W~ = 0: median sent latent unheard
    out.append(("w-ref-zero", S, W3))
    out.append(("nothing", S, np.zeros((NG, GL), np.float32)))
    W4 = (rng.exponential(1.0, (NG, GL)) * (rng.random((NG, GL)) < 0.6)).astype(np.float64)
    out.append(("float64", rng.standard_normal((NG, GL)) * 3.0, W4))
    out.append(("flat", S.ravel(), W.ravel()))
    return out


@pytest.mark.parametrize("mode", (0, 1, 2))
def test_decoder_planes(rd, mode):
    for label, S, W in _accumulators():
        a = render.decoder_planes(S, W, mode)
        b = rd.decoder_planes(S, W, mode)
        assert isinstance(b, tuple) and len(b) == 2, label
        _bits_equal(b[0], a[0])
        _bits_equal(b[1], a[1])
        assert rd.reference_w(W, mode) == render.reference_w(W, mode), label


def test_mode_is_taken_through_int(rd):
    _, S, W = _accumulators()[0]
    for mode in (np.int64(1), 2.0, True):
        a, b = render.decoder_planes(S, W, mode), rd.decoder_planes(S, W, mode)
        _bits_equal(b[0], a[0])
        assert rd.reference_w(W, mode) == render.reference_w(W, mode)


def test_g_shrink(rd):
    w = np.array([-0.0, 0.0, 5e-324, 1e-300, 0.39, 0.5, 1.0, 7.25, 1e300, np.inf, -3.0, np.nan])
    with np.errstate(invalid="ignore"):              # inf / inf (the native raises no warning)
        _bits_equal(rd.g_shrink(w), render.g_shrink(w))
    for x in ([1.0, 2.0], [[0.5, 1.5], [2.5, 3.5]], np.float32([0.25, 4.0]), np.arange(5)):
        _bits_equal(rd.g_shrink(x), render.g_shrink(x))
    a, b = render.g_shrink(2.0), rd.g_shrink(2.0)
    assert np.shape(b) == np.shape(a) == () and float(b) == float(a)


def test_reference_w_edges(rd):
    W = np.ones(NG * GL)
    for mode in (-1, -5):
        assert rd.reference_w(W, mode) == render.reference_w(W, mode) == 0.0


def _raises_alike(f_py, f_nat, exc):
    with pytest.raises(exc) as ea:
        f_py()
    with pytest.raises(exc) as eb:
        f_nat()
    assert str(eb.value) == str(ea.value)


def test_refusals(rd):
    ok = np.ones(NG * GL)
    for mode in (3, -1, 7):
        _raises_alike(lambda: render.decoder_planes(ok, ok, mode),
                      lambda: rd.decoder_planes(ok, ok, mode), ValueError)
    _raises_alike(lambda: render.decoder_planes(ok[:5], ok, 0),
                  lambda: rd.decoder_planes(ok[:5], ok, 0), ValueError)
    _raises_alike(lambda: render.decoder_planes(ok, ok[:-1], 0),
                  lambda: rd.decoder_planes(ok, ok[:-1], 0), ValueError)
    for bad, which in ((-1.0, "W"), (np.inf, "W"), (np.nan, "W"), (np.inf, "S"), (np.nan, "S")):
        x = ok.copy()
        x[7] = bad
        S, W = (ok, x) if which == "W" else (x, ok)
        _raises_alike(lambda: render.decoder_planes(S, W, 0), lambda: rd.decoder_planes(S, W, 0),
                      ValueError)
    _raises_alike(lambda: render.reference_w(ok, 3), lambda: rd.reference_w(ok, 3), IndexError)
    _raises_alike(lambda: render.reference_w(ok[:5], 0), lambda: rd.reference_w(ok[:5], 0),
                  ValueError)


# ---- adversarial probes (review stage) ----------------------------------------------

def test_probe_extreme_accumulators(rd):
    """Random accumulators with denormal, float32-edge and huge W, -0.0, and S
    up to 1e300: latents that overflow float32 (S/W past 3.4e38) become inf
    by the same cast, the rest bit-identical; float64 and float32 stores,
    every mode."""
    rng = np.random.default_rng(5)
    for trial in range(6):
        W = rng.exponential(1, (NG, GL)) * (rng.random((NG, GL)) < 0.7)
        S = rng.standard_normal((NG, GL)) * rng.choice([1, 1e30, 1e300], (NG, GL))
        k = rng.integers(0, NG * GL, 200)
        W.flat[k] = rng.choice([5e-324, 1e-310, 1.4e-45, 1e-40, 3.4e38, 1e300, -0.0], 200)
        mode = trial % 3
        for dt in (np.float64, np.float32):
            Sd, Wd = S.astype(dt), W.astype(dt)
            a = P.outcome(lambda: render.decoder_planes(Sd, Wd, mode))
            b = P.outcome(lambda: rd.decoder_planes(Sd, Wd, mode))
            if P.same_refusal(a, b):                     # 1e300 is inf in float32: refused
                continue
            _bits_equal(b[0], a[0])
            _bits_equal(b[1], a[1])
            assert rd.reference_w(Wd, mode) == render.reference_w(Wd, mode)

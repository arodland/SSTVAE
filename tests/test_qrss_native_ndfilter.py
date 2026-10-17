"""Parity of native/core/math/ndfilter (sstvae_native.qrss.ndfilter) with
scipy.ndimage.median_filter(x, size=w, mode="reflect") on 1-D arrays,
float64 and float32, bit for bit: every array length against every odd
window up to well past it (scipy's 1-D fast path and its generic path),
ties and signed zeros (which of two equal values is returned is visible
in the bits), and the sizes acquire.spectral_shape uses. Plus the port's
refusals. The one case held to the value rather than the bits is a
window of exactly 2 n + 1 (scipy's lim2 == 0), where scipy reads one
element past the array (see native/core/math/ndfilter.hpp).
"""

import numpy as np
import pytest

from qrss_native import require

ndimage = pytest.importorskip("scipy.ndimage")


@pytest.fixture(scope="module")
def nf(native):
    return require(native, "ndfilter")


def _same(a, b, what=""):
    a, b = np.asarray(a), np.asarray(b)
    assert a.dtype == b.dtype, (what, a.dtype, b.dtype)
    assert a.shape == b.shape, (what, a.shape, b.shape)
    assert a.tobytes() == b.tobytes(), (what, a, b)


def _both(nf, v, w, what):
    got, want = nf.median_filter(v, w), ndimage.median_filter(v, size=w, mode="reflect")
    if len(v) > 1 and w == 2 * len(v) + 1:
        # lim2 == 0: scipy's fast path primes its heap with v[len(v)], a
        # read past the end, so its tie-breaking (the sign of a zero)
        # depends on that memory: the value, not the bits
        assert got.dtype == want.dtype and np.array_equal(got, want, equal_nan=True), (what, got, want)
        return
    _same(got, want, what)


def test_every_length_and_window(nf):
    rng = np.random.default_rng(1)
    for n in range(1, 41):
        for w in range(1, 2 * n + 8, 2):
            v = rng.standard_normal(n)
            _both(nf, v, w, f"n={n} w={w}")
            _both(nf, v.astype(np.float32), w, f"n={n} w={w} float32")
            t = np.round(v * 2) / 2                       # ties
            _both(nf, t, w, f"n={n} w={w} ties")


def test_signed_zeros_and_ties(nf):
    rng = np.random.default_rng(2)
    for n in (3, 7, 20, 101):
        for w in (3, 5, 7, 21):
            v = rng.choice(np.array([0.0, -0.0, 1.0, -1.0]), n)
            _both(nf, v, w, f"zeros n={n} w={w}")
            _both(nf, v.astype(np.float32), w, f"zeros n={n} w={w} float32")


def test_nan_follows_scipy(nf):
    """A NaN breaks the comparisons both of scipy's selections rely on;
    the port runs the same comparisons, so what comes out is scipy's."""
    rng = np.random.default_rng(4)
    for n in (2, 3, 5, 9, 30, 200):
        for w in (3, 5, 9, 31, 2 * n + 3):
            if w == 2 * n + 1:
                continue    # lim2 == 0: a NaN-free heap is not promised there
            v = rng.standard_normal(n)
            v[rng.integers(n, size=max(1, n // 7))] = np.nan
            _both(nf, v, w, f"nan n={n} w={w}")
            _both(nf, v.astype(np.float32), w, f"nan n={n} w={w} float32")


def test_acquire_sizes(nf):
    """spectral_shape filters a 1-D level over the band's bins with
    w = CFAR_LOCAL_BINS (or the largest odd count that fits)."""
    from sstvae.qrss import acquire as A

    rng = np.random.default_rng(3)
    for n in (9601, 1201, 801, 400, 321, 320):
        lvl = rng.exponential(1.0, n) / np.log(2.0)
        for w in (A.CFAR_LOCAL_BINS, 161, 7, 1):
            ww = min(w, n if n % 2 else n - 1)
            _both(nf, lvl, ww, f"n={n} w={ww}")
            _both(nf, lvl.astype(np.float32), ww, f"n={n} w={ww} float32")


def test_empty_and_refusals(nf):
    _both(nf, np.zeros(0), 5, "empty")
    with pytest.raises(ValueError, match="size must be odd and positive, not 4"):
        nf.median_filter(np.ones(5), 4)
    with pytest.raises(ValueError, match="size must be odd and positive, not 0"):
        nf.median_filter(np.ones(5), 0)
    with pytest.raises(ValueError, match="only mode 'reflect'"):
        nf.median_filter(np.ones(5), 3, mode="nearest")
    with pytest.raises(ValueError, match="only a 1-D array"):
        nf.median_filter(np.ones((2, 5)), 3)

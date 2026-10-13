"""Parity of native/core/math/npfft (sstvae_native.qrss.npfft) with
numpy.fft, bit for bit: fft, ifft, rfft and irfft at every length to 300
and at the receiver's (16000-point STFT frames, 9601-point bands, primes
that take pocketfft's Bluestein path), cropped and zero-padded through
`n`, real and complex input, double and single precision; fft along the
last axis of a 2-D array;
fftfreq and rfftfreq; numpy's refusals, in its words.

numpy 2.x's FFT is pocketfft's C++ header and the port calls the
vendored copy's same plan classes the same way, so agreement to the bit
is a measurement of the two headers' arithmetic agreeing, not a property
of the API -- which is why this checks so many lengths.
"""

import numpy as np
import pytest

from qrss_native import require


@pytest.fixture(scope="module")
def nf(native):
    return require(native, "npfft")


def _same(a, b, what=""):
    a, b = np.asarray(a), np.asarray(b)
    assert a.dtype == b.dtype, (what, a.dtype, b.dtype)
    assert a.shape == b.shape, (what, a.shape, b.shape)
    assert a.tobytes() == b.tobytes(), (what, np.flatnonzero(a.ravel() != b.ravel())[:5])


LENGTHS = list(range(1, 301)) + [320, 480, 500, 512, 1000, 1009, 1024, 2000, 4000, 4801, 8000, 9601, 10007,
                                 16000, 16384, 32000, 65536]


def test_c2c_every_length(nf):
    rng = np.random.default_rng(1)
    for n in LENGTHS:
        a = rng.standard_normal(n) + 1j * rng.standard_normal(n)
        r = rng.standard_normal(n)
        _same(nf.fft(a), np.fft.fft(a), f"fft {n}")
        _same(nf.ifft(a), np.fft.ifft(a), f"ifft {n}")
        _same(nf.fft(r), np.fft.fft(r), f"fft real {n}")
        if n <= 300:
            for m in (1, max(1, n // 2), n + 1, 2 * n + 1):
                _same(nf.fft(a, m), np.fft.fft(a, m), f"fft {n} -> {m}")
                _same(nf.ifft(a, m), np.fft.ifft(a, m), f"ifft {n} -> {m}")


def test_single_precision(nf):
    """numpy 2.x transforms complex64/float32 in single precision (float
    plans, fct = reciprocal(n, dtype=float32)) and returns complex64 /
    float32; multiplied by a float64 window first, the same data is
    complex128 and takes the double path."""
    rng = np.random.default_rng(2)
    for n in LENGTHS[::3] + [16000, 9601]:
        a = (rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(np.complex64)
        r = rng.standard_normal(n).astype(np.float32)
        _same(nf.fft(a), np.fft.fft(a), f"fft complex64 {n}")
        _same(nf.ifft(a), np.fft.ifft(a), f"ifft complex64 {n}")
        _same(nf.fft(r), np.fft.fft(r), f"fft float32 {n}")
        _same(nf.rfft(r), np.fft.rfft(r), f"rfft float32 {n}")
        c = np.fft.rfft(r)
        if n > 1:                          # one bin: default n is 0, refused (test_refusals)
            _same(nf.irfft(c), np.fft.irfft(c), f"irfft complex64 {n}")
        _same(nf.irfft(c, n), np.fft.irfft(c, n), f"irfft complex64 {n} n")
        if n <= 300:
            _same(nf.fft(a, 2 * n + 1), np.fft.fft(a, 2 * n + 1), f"fft complex64 {n} padded")
            _same(nf.ifft(a, n + 3), np.fft.ifft(a, n + 3), f"ifft complex64 {n} padded")
    a = (rng.standard_normal(16000) + 1j * rng.standard_normal(16000)).astype(np.complex64)
    w = np.hanning(16000)
    _same(nf.fft(a * w), np.fft.fft(a * w), "windowed complex64 is complex128")
    m = (rng.standard_normal((4, 100)) + 1j * rng.standard_normal((4, 100))).astype(np.complex64)
    _same(nf.fft(m, 128), np.fft.fft(m, 128), "rows complex64")
    _same(nf.fft(np.arange(10)), np.fft.fft(np.arange(10)), "integers are double")


def test_real_transforms(nf):
    rng = np.random.default_rng(3)
    for n in LENGTHS:
        r = rng.standard_normal(n)
        _same(nf.rfft(r), np.fft.rfft(r), f"rfft {n}")
        c = np.fft.rfft(r) + 0.25j        # imag of the DC (and Nyquist) bins ignored
        if n > 1:
            _same(nf.irfft(c), np.fft.irfft(c), f"irfft default {n}")
        _same(nf.irfft(c, n), np.fft.irfft(c, n), f"irfft {n}")
        if n <= 300:
            for m in (1, 2, max(1, n // 2), n + 1, 2 * n + 3):
                _same(nf.rfft(r, m), np.fft.rfft(r, m), f"rfft {n} -> {m}")
                _same(nf.irfft(c, m), np.fft.irfft(c, m), f"irfft {n} -> {m}")


def test_rows(nf):
    rng = np.random.default_rng(4)
    for rows, cols in ((1, 7), (3, 50), (46, 160), (5, 2000), (0, 8)):
        a = rng.standard_normal((rows, cols)) + 1j * rng.standard_normal((rows, cols))
        _same(nf.fft(a), np.fft.fft(a), f"rows {rows}x{cols}")
        _same(nf.fft(a, axis=1), np.fft.fft(a, axis=1), f"rows {rows}x{cols} axis=1")
        _same(nf.fft(a, cols + 5), np.fft.fft(a, cols + 5), f"rows {rows}x{cols} padded")


def test_freqs(nf):
    for n in list(range(1, 50)) + [160, 2000, 9601, 16000, 16001]:
        for d in (1.0, 0.25, 1 / 4000, 1 / 500, 1 / 250, 3.7):
            _same(nf.fftfreq(n, d), np.fft.fftfreq(n, d), f"fftfreq {n} {d}")
            _same(nf.rfftfreq(n, d), np.fft.rfftfreq(n, d), f"rfftfreq {n} {d}")
        _same(nf.fftfreq(n), np.fft.fftfreq(n), f"fftfreq {n}")


def _outcome(fn, *args, **kw):
    try:
        out = fn(*args, **kw)
    except (ValueError, TypeError, IndexError) as e:
        return type(e), str(e)
    return "ok", out


def test_refusals_match_numpy(nf):
    a = np.array([1 + 0j, 2 + 0j])
    for fn, args in [("fft", (a, 0)), ("ifft", (a, -3)), ("fft", (np.zeros(0, complex),)),
                     ("ifft", (np.zeros(0, complex),)), ("rfft", (np.ones(4), 0)),
                     ("irfft", (np.array([1 + 0j]),)), ("irfft", (np.zeros(0, complex),)),
                     ("irfft", (a, 0)), ("fft", (np.ones((2, 3), complex), 0)),
                     ("fft", (np.ones((2, 3), complex), -2)), ("fft", (np.ones((3, 0), complex),)),
                     ("fft", (np.ones((2, 3), np.complex64), 0))]:
        r, g = _outcome(getattr(np.fft, fn), *args), _outcome(getattr(nf, fn), *args)
        assert r[0] == g[0] and r[1] == g[1], (fn, args, r, g)
    # numpy reads x[0] of an empty input past its end when n is given; the
    # port refuses instead of returning that memory's transform
    with pytest.raises(ValueError, match="irfft: empty input"):
        nf.irfft(np.zeros(0, complex), 4)
    with pytest.raises(NotImplementedError):
        nf.fft(a, norm="ortho")
    with pytest.raises(NotImplementedError):
        nf.ifft(np.ones((2, 3), complex))
    with pytest.raises(NotImplementedError):
        nf.fft(np.ones(4, np.clongdouble))

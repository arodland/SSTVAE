"""Parity of native/core/qrss/lwave_rx (sstvae_native.qrss.lwave_rxf) with
the receive side of sstvae/qrss/lwave.py: PULSE_NORM and L's exact symbol
matched filter.

Contract 1e-12; measured 2026-10-13: PULSE_NORM equal to the bit (and to
the generated rxconfig value), matched_filter bit-identical for real and
complex input on every grid tested, positions off both ends, every chunk
size; a constant gives lwave.G0 to the bit. (`to_data_centre` is the
transmit module's and is tested with it; `loopback_stats` is a
diagnostic and stays Python.)
"""

import numpy as np
import pytest

from qrss_native import require
from sstvae.qrss import lwave

FS = (250, 500, 1000, 4000, 16000)


@pytest.fixture(scope="module")
def lr(native):
    return require(native, "lwave_rxf")


def test_pulse_norm(lr):
    assert lr.PULSE_NORM == lwave.PULSE_NORM
    assert type(lr.PULSE_NORM) is float


@pytest.mark.parametrize("fs", FS)
def test_matched_filter(lr, fs):
    rng = np.random.default_rng(fs + 1)
    n = 4 * fs
    zc = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    pos = np.arange(-30, 160)
    for z in (zc, zc.real, zc.astype(np.complex64), np.arange(n) % 5):
        for n0 in (-fs // 2, 0, 41):
            a = lwave.matched_filter(z, fs, n0, pos)
            b = lr.matched_filter(z, fs, n0, pos)
            assert b.tobytes() == a.tobytes() and b.dtype == a.dtype


def test_matched_filter_chunks_and_edges(lr):
    rng = np.random.default_rng(9)
    z = rng.standard_normal(3000) + 1j * rng.standard_normal(3000)
    pos = rng.integers(-40, 300, 2500)
    for chunk in (1, 64, 2048, 4096):
        a = lwave.matched_filter(z, 250, -250, pos, chunk)
        b = lr.matched_filter(z, 250, -250, pos, chunk=chunk)
        assert b.tobytes() == a.tobytes()
    assert lr.matched_filter(z, 250, 0, []).shape == (0,)
    g = lr.matched_filter(np.ones(64000), 16000, -32000, [0])[0]
    assert g == lwave.matched_filter(np.ones(64000), 16000, -32000, [0])[0]
    assert abs(g - lwave.G0) <= 1e-12


def _raises_alike(f_py, f_nat, exc):
    with pytest.raises(exc) as ea:
        f_py()
    with pytest.raises(exc) as eb:
        f_nat()
    assert str(eb.value) == str(ea.value)


def test_refusals(lr):
    z = np.ones(100)
    _raises_alike(lambda: lwave.matched_filter(z, 3000, 0, [1]), lambda: lr.matched_filter(z, 3000, 0, [1]),
                  ValueError)
    _raises_alike(lambda: lwave.matched_filter(z, 250, 0, [1], 0),
                  lambda: lr.matched_filter(z, 250, 0, [1], 0), ValueError)
    _raises_alike(lambda: lwave.matched_filter([], 250, 0, [1]), lambda: lr.matched_filter([], 250, 0, [1]),
                  IndexError)

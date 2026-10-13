"""Parity of native/core/qrss/ce_rx (sstvae_native.qrss.ce_rx) with the
receive side of sstvae/qrss/ce.py: the exact symbol matched filter and the
mean waveform (closed form and on the exact grid).

Contract 1e-12 (absolute, the values being O(1)); measured 2026-10-13:

- matched_filter: bit-identical, real and complex input, every grid
  tested (250 Hz to 16 kHz), positions off both ends, every chunk size
  (the per-row sums are numpy's pairwise ones, the scale its last
  multiply);
- mean_template and _mean_template_grid: worst 1.6e-16, ~95% of values
  bit-identical. The rest is np.exp: numpy's SIMD exp is one ulp from
  libm's on ~5% of arguments, and exp(-v/2) is the only transcendental
  outside the exact phase arithmetic.
"""

import numpy as np
import pytest

import qrss_demod_probes as P
from qrss_native import require
from sstvae.qrss import ce, frame, morse

CALL = "K1ABC"
FS = (250, 500, 1000, 2000, 8000, 16000)


@pytest.fixture(scope="module")
def cr(native):
    return require(native, "ce_rx")


def _close(b, a, tol=1e-12):
    a, b = np.asarray(a), np.asarray(b)
    assert a.dtype == b.dtype and a.shape == b.shape, (a.dtype, b.dtype, a.shape, b.shape)
    err = float(np.max(np.abs(b - a))) if a.size else 0.0
    assert err <= tol, err
    return err


@pytest.mark.parametrize("fs", FS)
def test_matched_filter(cr, fs):
    rng = np.random.default_rng(fs)
    n = 5 * fs
    zc = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    pos = np.arange(-30, 200)                 # T = 1/33 s: off the start and past the end
    for z in (zc, zc.real, zc.astype(np.complex64), zc.real.astype(np.float32),
              np.arange(n) % 7, list(zc.real[:300])):
        for n0 in (-fs, 0, 37):
            a = ce.matched_filter(z, fs, n0, pos)
            b = cr.matched_filter(z, fs, n0, pos)
            assert b.tobytes() == a.tobytes() and b.dtype == a.dtype


def test_matched_filter_chunks_and_positions(cr):
    rng = np.random.default_rng(7)
    z = rng.standard_normal(4000) + 1j * rng.standard_normal(4000)
    pos = rng.integers(-50, 400, 3000)        # unordered, repeated
    for chunk in (1, 3, 100, 2048, 5000):
        a = ce.matched_filter(z, 250, -500, pos, chunk)
        b = cr.matched_filter(z, 250, -500, pos, chunk=chunk)
        assert b.tobytes() == a.tobytes()
    assert not cr.matched_filter(z, 250, 0, pos, -3).any()          # range() runs no chunk
    assert cr.matched_filter(z, 250, 0, []).shape == ce.matched_filter(z, 250, 0, []).shape == (0,)
    assert cr.matched_filter(np.ones(32000), 16000, -16000, [0])[0] == ce.G0


def _raises_alike(f_py, f_nat, exc):
    with pytest.raises(exc) as ea:
        f_py()
    with pytest.raises(exc) as eb:
        f_nat()
    assert str(eb.value) == str(ea.value)


def test_matched_filter_refusals(cr):
    z = np.ones(100)
    _raises_alike(lambda: ce.matched_filter(z, 3000, 0, [1]), lambda: cr.matched_filter(z, 3000, 0, [1]),
                  ValueError)
    _raises_alike(lambda: ce.matched_filter(z, 250, 0, [1], 0), lambda: cr.matched_filter(z, 250, 0, [1], 0),
                  ValueError)
    _raises_alike(lambda: ce.matched_filter([], 250, 0, [1]), lambda: cr.matched_filter([], 250, 0, [1]),
                  IndexError)


def _symbols(spec, seed):
    rng = np.random.default_rng(seed)
    mu = rng.standard_normal(spec.n_sym)
    nu = rng.uniform(0, 1.5, spec.n_sym)
    lay = frame.layout(spec)
    mu[lay.known_idx] = np.asarray(lay.known_val)
    nu[lay.known_idx] = 0.0
    return mu, nu


@pytest.mark.parametrize("name", ("tiny", "short"))
def test_mean_template(cr, name):
    spec = frame.get(name)
    mu, nu = _symbols(spec, 3)
    tau = np.linspace(-20.0, spec.n_pos + 5.0, 40001)
    keyings = [None, morse.keying_units(CALL)] if spec.n_win else [None]
    worst = 0.0
    for keying in keyings:
        for ook in (False, True):
            a = ce.mean_template(tau, mu, nu, spec, keying, ook)
            worst = max(worst, _close(cr.mean_template(tau, mu, nu, spec, keying, ook), a))
    # unordered and 2-D times, a lead-in, a scalar
    rng = np.random.default_rng(1)
    t2 = rng.permutation(tau[::7]).reshape(-1, 3)[:1000]
    _close(cr.mean_template(t2, mu, nu, spec, keyings[-1]), ce.mean_template(t2, mu, nu, spec, keyings[-1]))
    _close(cr.mean_template(tau[:3000], mu, nu, spec, keyed_from_pos=-15.5),
           ce.mean_template(tau[:3000], mu, nu, spec, keyed_from_pos=-15.5))
    a, b = ce.mean_template(3.25, mu, nu, spec), cr.mean_template(3.25, mu, nu, spec)
    assert np.shape(b) == np.shape(a) == () and abs(b - a) <= 1e-12
    assert worst <= 1e-12


@pytest.mark.parametrize("fs", (250, 16000))
def test_mean_template_grid(cr, fs):
    spec = frame.get("short")
    mu, nu = _symbols(spec, 4)
    n = 120_000 if fs == 16000 else 70_000
    for keying, ook in ((None, False), (morse.keying_units(CALL), False), (morse.keying_units(CALL), True)):
        for n0 in (-fs * 2, 3 * fs):
            a = ce._mean_template_grid(fs, n0, n, mu, nu, spec, keying, ook)
            _close(cr._mean_template_grid(fs, n0, n, mu, nu, spec, keying, ook), a)


def test_mean_template_refusals(cr):
    spec = frame.get("tiny")
    mu, nu = _symbols(spec, 5)
    _raises_alike(lambda: ce.mean_template([0.0], mu[:3], nu, spec),
                  lambda: cr.mean_template([0.0], mu[:3], nu, spec), ValueError)
    _raises_alike(lambda: ce.mean_template([0.0], mu, nu[:-1], spec),
                  lambda: cr.mean_template([0.0], mu, nu[:-1], spec), ValueError)
    _raises_alike(lambda: ce._mean_template_grid(3000, 0, 5, mu, nu, spec),
                  lambda: cr._mean_template_grid(3000, 0, 5, mu, nu, spec), ValueError)
    short = frame.get("short")
    ms, ns_ = _symbols(short, 6)
    _raises_alike(lambda: ce.mean_template([0.0], ms, ns_, short, np.ones((2, 5))),
                  lambda: cr.mean_template([0.0], ms, ns_, short, np.ones((2, 5))), ValueError)


# ---- adversarial probes (review stage) ----------------------------------------------

@pytest.mark.parametrize("label,args", P.mf_cases(), ids=lambda v: v if isinstance(v, str) else "")
def test_probe_matched_filter_argument_forms(cr, label, args):
    P.check_mf_case(ce.matched_filter, cr.matched_filter, args)


def test_probe_matched_filter_refuses_2d_pos(cr):
    """A 2-D pos: the reference fails a broadcast; the port refuses it outright
    (both ValueError; a flattened read had returned values)."""
    z = np.ones(500)
    with pytest.raises(ValueError):
        ce.matched_filter(z, 250, 0, [[1, 2], [3, 4]])
    with pytest.raises(ValueError):
        cr.matched_filter(z, 250, 0, [[1, 2], [3, 4]])


def test_probe_matched_filter_full_pass(cr):
    """A FULL pass's whole grid at 250 Hz, positions -8 .. n_pos + 8 (int64
    index arithmetic at full size), real and complex."""
    spec = frame.get("full")
    rng = np.random.default_rng(11)
    pos = np.arange(-8, spec.n_pos + 8)
    n0 = int(np.floor((pos[0] - 10) * 485 / 64))
    n = int(np.ceil((pos[-1] + 10) * 485 / 64)) - n0
    z = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    for zz in (z, z.real.copy()):
        a, b = ce.matched_filter(zz, 250, n0, pos), cr.matched_filter(zz, 250, n0, pos)
        assert b.dtype == a.dtype and b.tobytes() == a.tobytes()


@pytest.mark.parametrize("name", ("medium", "full", "short-v1", "full-spreadonly"))
def test_probe_mean_template_presets(cr, name):
    """Every other preset, at sparse times through the whole frame and a
    lead-in, and non-finite times (NaN, +-inf, 1e300)."""
    spec = frame.get(name)
    mu, nu = _symbols(spec, 12)
    rng = np.random.default_rng(13)
    tau = np.concatenate([rng.uniform(-40.0, spec.n_pos + 40.0, 4000),
                          [np.nan, np.inf, -np.inf, 1e300, -1e300, 2.0**62, -0.0, 0.5, -0.5]])
    keying = morse.keying_units(CALL) if spec.n_win else None
    for kf in (None, -30.25):
        a = P.outcome(lambda: ce.mean_template(tau, mu, nu, spec, keying, keyed_from_pos=kf))
        b = cr.mean_template(tau, mu, nu, spec, keying, keyed_from_pos=kf)
        P.rel(b.real, a.real, tol=np.inf)                 # the same non-finite values
        P.rel(b.imag, a.imag, tol=np.inf)
        fin = np.isfinite(a)
        _close(b[fin], a[fin])


def test_probe_mean_template_non_finite_symbols(cr):
    """NaN/inf in mu or nu and negative nu: the same non-finite pattern in the
    closed form, the rest to 1e-12 relative."""
    spec = frame.get("tiny")
    mu, nu = _symbols(spec, 14)
    tau = np.linspace(-20, spec.n_pos + 5, 30001)
    for what in ("mu nan", "mu inf", "nu inf", "nu nan", "nu negative"):
        m2, n2 = mu.copy(), nu.copy()
        if what == "mu nan":
            m2[100] = np.nan
        elif what == "mu inf":
            m2[100] = np.inf
        elif what == "nu inf":
            n2[100] = np.inf
        elif what == "nu nan":
            n2[100] = np.nan
        else:
            n2[100:110] = -5.0
        a = P.outcome(lambda: ce.mean_template(tau, m2, n2, spec))
        b = P.outcome(lambda: cr.mean_template(tau, m2, n2, spec))
        P.rel(b.real, a.real, what)
        P.rel(b.imag, a.imag, what)


def test_probe_mean_template_refuses_an_l_frame(cr):
    """ce's mean template on an L frame: Python's frame.layout fails its
    assertion (after mu's shape check, before nu's); the port had laid it
    out as L and returned values."""
    from sstvae.qrss import lframe
    for name in lframe.PRESETS:
        spec = lframe.get(name)
        mu, nu = np.zeros(spec.n_sym), np.ones(spec.n_sym)
        _raises_alike(lambda: ce.mean_template([0.0, 1.0], mu, nu, spec),
                      lambda: cr.mean_template([0.0, 1.0], mu, nu, spec), AssertionError)
        _raises_alike(lambda: ce.mean_template([0.0], mu, nu[:3], spec),
                      lambda: cr.mean_template([0.0], mu, nu[:3], spec), AssertionError)
        _raises_alike(lambda: ce.mean_template([0.0], mu[:3], nu, spec),
                      lambda: cr.mean_template([0.0], mu[:3], nu, spec), ValueError)
        _raises_alike(lambda: ce._mean_template_grid(250, -50, 100, mu, nu, spec),
                      lambda: cr._mean_template_grid(250, -50, 100, mu, nu, spec), AssertionError)

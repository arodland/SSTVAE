"""C++ waveform L transmitter (native/core/qrss/lwave) against sstvae.qrss.lwave.

Three classes, as native/core/qrss/lwave.hpp states them:

- **bit for bit**: the integer and table work (at_positions, data_grid and
  _grid_sum through scipy's upfirdn, mix_turns, tau_grid, keyed_span_pos,
  window_keying, _interp_taps), resample given the same frame, and -- with
  numpy's own sin/cos (libm here) and its fused complex multiply -- the
  pilot, cw_turns, id_weight, composite and band_mask;
- **tolerance**: clip and frame_baseband, a whole-frame FFT (pocketfft
  against scipy.fft's ducc0) and a pow; measured near 1e-15, held to 1e-9
  as the plan sets it, and the measured figure is asserted well under it
  too so a drift is seen long before it matters;
- **refusals**: ValueError texts compared in full.

Skips when the extension is not built; `pytest --native-qrss=tx` runs the
rest of the QRSS suites against the same code.
"""

from fractions import Fraction

import numpy as np
import pytest
import scipy.fft

from qrss_native import require
from sstvae.qrss import frame as F
from sstvae.qrss import lframe as LF
from sstvae.qrss import lwave as ref
from sstvae.qrss import morse
from sstvae.qrss.constants import FE_FS, FS, L_CW_WINDOW, L_PILOT_HZ, L_T_SYM

L_NAMES = ("l-tiny", "l-short", "l-medium", "l-full")
CALL = "K1ABC/P"


@pytest.fixture(scope="module")
def nat(native, request):
    if request.config.getoption("--native-qrss", default=None):
        pytest.skip("--native-qrss replaces the reference this file compares against")
    return require(native, "lwave")


def _sym(spec, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(spec.n_sym) + 1j * rng.standard_normal(spec.n_sym)) / np.sqrt(2)


def _same(a, b):
    """Bit for bit, dtype and shape included."""
    a, b = np.asarray(a), np.asarray(b)
    assert a.dtype == b.dtype, (a.dtype, b.dtype)
    assert a.shape == b.shape, (a.shape, b.shape)
    assert a.tobytes() == b.tobytes(), np.max(np.abs(a - b)) if a.size else 0


def _outcome(fn, *args, **kw):
    try:
        out = fn(*args, **kw)
    except (ValueError, TypeError, OverflowError) as e:
        return type(e), str(e)
    return "ok", out


def _keyings(spec):
    k = morse.keying_units(CALL)
    yield None
    if spec.n_win:
        yield k
        per = np.zeros((spec.n_win, L_CW_WINDOW), dtype=np.uint8)
        per[0] = k
        per[-1] = morse.keying_units("W1AW")
        yield per
        yield k.astype(bool)
        yield k.astype(np.float32)


def _frame_tau(spec, fs, lead=5.0):
    lo, hi = ref.keyed_span_pos(spec, lead)
    n0 = int(np.floor((lo * L_T_SYM - 1.0) * fs))
    n1 = int(np.ceil((hi * L_T_SYM + 1.0) * fs))
    return n0, n1 - n0


# --- constants -----------------------------------------------------------------------------

def test_constants(nat):
    for name in ("GRID_FS", "HALF", "CLIP_FS", "G0", "A_PILOT", "A_CLIP", "L_AVG_POWER", "A_ID",
                 "ID_RAMP_UNITS", "KEY_RAMP_S", "INTERP_HALF"):
        a, b = getattr(nat, name), getattr(ref, name)
        assert type(a) is type(b) and a == b, name
    _same(nat.PULSE_TABLE, ref.PULSE_TABLE)
    assert not nat.PULSE_TABLE.flags.writeable


def test_interp_taps(nat):
    for up in (1, 2, 16, 32, 7):
        _same(nat._interp_taps(up), ref._interp_taps(up))


# --- the data on a sample grid -------------------------------------------------------------

def test_grid_step(nat):
    for fs in (250, 500, 1000, 4000, 8000, 16000, 1, 300, 0, -250, 32000, 250.0, np.int64(500)):
        assert _outcome(nat._grid_step, fs) == _outcome(ref._grid_step, fs), fs


def test_grid_sum(nat):
    rng = np.random.default_rng(3)
    for n_pos in (0, 1, 40, 300):
        vc = rng.standard_normal(n_pos) + 1j * rng.standard_normal(n_pos)
        vr = rng.standard_normal(n_pos)
        for fs in (250, 1000, 8000):
            for n0, n in ((-300, 900), (0, 1), (5000, 10), (-10**6, 50), (17, 0), (17, -4)):
                for vals in (vc, vr):
                    _same(nat._grid_sum(fs, n0, n, vals, ref.PULSE_TABLE),
                          ref._grid_sum(fs, n0, n, vals, ref.PULSE_TABLE))


@pytest.mark.parametrize("name", L_NAMES)
def test_at_positions_and_data_grid(nat, name):
    spec = LF.get(name)
    sym = _sym(spec, 1)
    _same(nat.at_positions(sym, spec), ref.at_positions(sym, spec))
    n0, n = _frame_tau(spec, 250)
    _same(nat.data_grid(250, n0, n, sym, spec), ref.data_grid(250, n0, n, sym, spec))
    for fs in (FE_FS, FS):
        _same(nat.data_grid(fs, -4000, 30000, sym, spec), ref.data_grid(fs, -4000, 30000, sym, spec))


def test_at_positions_refusals(nat):
    spec = LF.get("l-tiny")
    for bad in (np.zeros(spec.n_sym - 1), np.zeros((1, spec.n_sym)), 0.0):
        assert _outcome(nat.at_positions, bad, spec) == _outcome(ref.at_positions, bad, spec)
    with pytest.raises(AssertionError):           # the reference fails on a CE frame too
        nat.at_positions(np.zeros(F.TINY.n_sym), F.TINY)
    with pytest.raises(Exception):
        ref.at_positions(np.zeros(F.TINY.n_sym), F.TINY)


def _exact_turns(fs, n0, n, f):
    f = Fraction(f)
    period = f.denominator * fs
    r = [((k % period) * (f.numerator % period)) % period for k in range(n0, n0 + n)]
    return np.array([float(x) / float(period) for x in r], dtype=np.float64)


def test_mix_turns(nat):
    for f in (L_PILOT_HZ, Fraction(-12207, 970), Fraction(3, 7), 5, 0, 12.5, -0.1, Fraction(10**6 + 1, 3)):
        for fs in (250, 4000, 8000, 1):
            for n0, n in ((0, 1000), (-123457, 5000), (10**12, 300), (7, 0)):
                period = Fraction(f).denominator * fs
                if period > 2**63 - 1:
                    # numpy cannot hold the period; both refuse it.
                    assert _outcome(nat.mix_turns, fs, n0, n, f)[0] is OverflowError
                    assert _outcome(ref.mix_turns, fs, n0, n, f)[0] is OverflowError
                elif (period - 1) ** 2 > 2**63 - 1:
                    # The reference's int64 products wrap here (-0.1 is
                    # Fraction(-3602879701896397, 2**55)); the port's are exact.
                    _same(nat.mix_turns(fs, n0, n, f), _exact_turns(fs, n0, n, f))
                else:
                    _same(nat.mix_turns(fs, n0, n, f), ref.mix_turns(fs, n0, n, f))
    assert ref.mix_turns(250, 0, 3, L_PILOT_HZ).tobytes() == _exact_turns(250, 0, 3, L_PILOT_HZ).tobytes()


def test_tau_grid(nat):
    for fs in (250, 4000, 8000):
        for n0, n in ((0, 1000), (-123457, 5000), (10**9, 300), (3, 0)):
            _same(nat.tau_grid(fs, n0, n), ref.tau_grid(fs, n0, n))


# --- pilot -----------------------------------------------------------------------------------

def test_keyed_span_pos(nat):
    for spec in (*(LF.get(x) for x in L_NAMES), F.SHORT, F.TINY):
        for lead in (0.0, 0, 5.0, 10, 2.5, np.float64(7.25), True):
            assert nat.keyed_span_pos(spec, lead) == ref.keyed_span_pos(spec, lead)
            assert [type(x) for x in nat.keyed_span_pos(spec, lead)] == \
                [type(x) for x in ref.keyed_span_pos(spec, lead)]
        for lead in (-0.5, 10.5, 11, float("nan"), float("inf"), -0.0):
            assert _outcome(nat.keyed_span_pos, spec, lead) == _outcome(ref.keyed_span_pos, spec, lead)
    spec = LF.get("l-short")
    assert nat.keyed_span_pos(spec) == ref.keyed_span_pos(spec)


@pytest.mark.filterwarnings("ignore::numpy.exceptions.ComplexWarning")
def test_window_keying(nat):
    spec = LF.get("l-medium")
    k = morse.keying_units(CALL)
    cases = [None, k, np.stack([k, k]), k.astype(bool), k.astype(np.int64), k.astype(np.float64),
             k.astype(complex), list(k), np.stack([k, k, k]), k[:-1], np.full(L_CW_WINDOW, 2),
             np.full(L_CW_WINDOW, 0.5), k.astype(complex) + 1j, np.zeros((2, 2, L_CW_WINDOW)),
             np.ones(384), np.zeros(0)]
    for c in cases:
        a, b = _outcome(nat.window_keying, c, spec), _outcome(ref.window_keying, c, spec)
        if a[0] == "ok":
            assert b[0] == "ok"
            if a[1] is None:
                assert b[1] is None
            else:
                _same(a[1], b[1])
        else:
            assert a == b


def test_raised(nat):
    y = np.array([-np.inf, -1.0, -0.0, 0.0, 1e-300, 0.25, 0.5, 0.75, 1.0, 1.5, np.inf, np.nan])
    y = np.concatenate([y, np.linspace(-0.5, 1.5, 4001)])
    _same(nat._raised(y), ref._raised(y))
    for v in (0.3, -2.0, 0.0):
        a, b = nat._raised(v), ref._raised(v)
        assert type(a) is type(b) and a == b


@pytest.mark.parametrize("name", ["l-short", "l-medium"])
def test_pilot_pieces(nat, name):
    spec = LF.get(name)
    n0, n = _frame_tau(spec, 250)
    tau = ref.tau_grid(250, n0, n)
    tau2 = tau.reshape(-1, 2) if n % 2 == 0 else tau[:-1].reshape(-1, 2)
    for kk in _keyings(spec):
        _same(nat.cw_turns(tau, spec, kk), ref.cw_turns(tau, spec, kk))
        _same(nat.cw_turns(tau2, spec, kk), ref.cw_turns(tau2, spec, kk))
        for lead in (0.0, 5.0):
            _same(nat.pilot(tau, spec, kk, lead), ref.pilot(tau, spec, kk, lead))
    for lead in (0.0, 2.0, 10.0):
        _same(nat.id_weight(tau, spec, lead), ref.id_weight(tau, spec, lead))
        _same(nat.id_weight(tau2, spec, lead), ref.id_weight(tau2, spec, lead))
    k = morse.keying_units(CALL)
    _same(nat.pilot(tau, spec, k, 3.0, 0.2), ref.pilot(tau, spec, k, 3.0, 0.2))
    _same(nat.pilot(tau, spec, k, 3.0, 0.0), ref.pilot(tau, spec, k, 3.0, 0.0))
    _same(nat.pilot(tau, spec), ref.pilot(tau, spec))


def test_pilot_on_a_ce_frame(nat):
    """The reference's pilot functions take any frame; so do these."""
    tau = np.linspace(-200, 9000, 20001)
    k = morse.keying_units(CALL)
    _same(nat.id_weight(tau, F.SHORT, 5.0), ref.id_weight(tau, F.SHORT, 5.0))
    _same(nat.cw_turns(tau, F.SHORT, k), ref.cw_turns(tau, F.SHORT, k))
    _same(nat.pilot(tau, F.SHORT, k, 5.0), ref.pilot(tau, F.SHORT, k, 5.0))


def test_pilot_refusals(nat):
    spec = LF.get("l-short")
    tau = np.linspace(-10, 100, 50)
    for lead in (-1.0, 12, float("nan")):
        assert _outcome(nat.pilot, tau, spec, None, lead) == _outcome(ref.pilot, tau, spec, None, lead)
    bad = np.full(L_CW_WINDOW, 3)
    assert _outcome(nat.pilot, tau, spec, bad) == _outcome(ref.pilot, tau, spec, bad)
    assert _outcome(nat.cw_turns, tau, spec, bad) == _outcome(ref.cw_turns, tau, spec, bad)


# --- composite, band mask, clipper ----------------------------------------------------------

@pytest.mark.parametrize("name", ["l-tiny", "l-short"])
def test_composite(nat, name):
    spec = LF.get(name)
    sym = _sym(spec, 2)
    for lead in (0.0, 5.0):
        n0, n = _frame_tau(spec, 250, lead)
        for kk in _keyings(spec):
            _same(nat.composite(sym, spec, 250, n0, n, keying=kk, lead_in_s=lead),
                  ref.composite(sym, spec, 250, n0, n, keying=kk, lead_in_s=lead))
    _same(nat.composite(sym, spec, FS, -2000, 40000), ref.composite(sym, spec, FS, -2000, 40000))
    _same(nat.composite(sym, spec, 250, 0, 0), ref.composite(sym, spec, 250, 0, 0))
    for bad in (dict(lead_in_s=11), dict(lead_in_s=-1)):
        assert _outcome(nat.composite, sym, spec, 250, 0, 10, **bad) == \
            _outcome(ref.composite, sym, spec, 250, 0, 10, **bad)


def test_band_mask(nat):
    for n, d in ((66068, 1 / 250), (8192, 1 / 250), (1001, 1 / 4000), (12, 0.5)):
        f = np.fft.fftfreq(n, d)
        _same(nat._fftfreq(n, d), f)
        _same(nat.band_mask(f), ref.band_mask(f))
        _same(nat.band_mask(f, 2.5, 0.5), ref.band_mask(f, 2.5, 0.5))
        sf = scipy.fft.fftfreq(n, d)
        assert np.max(np.abs(nat.band_mask(sf) - ref.band_mask(sf))) <= 1e-15
    for v in (-14.0, 0.0, 1.0, 0.75):
        a, b = nat.band_mask(v), ref.band_mask(v)
        assert type(a) is type(b) and a == b


def test_next_fast_len(nat):
    for n in [*range(1, 3000), 66068, 67318, 10**6 + 7, 2**31 + 1]:
        assert nat._next_fast_len(n) == scipy.fft.next_fast_len(n), n


def test_numpy_arithmetic(nat):
    """np.abs and complex * as numpy computes them (the clipper's own pieces)."""
    rng = np.random.default_rng(4)
    z = rng.standard_normal(20000) * 10.0 ** rng.uniform(-200, 200, 20000) \
        + 1j * rng.standard_normal(20000) * 10.0 ** rng.uniform(-200, 200, 20000)
    special = np.array([0, -0.0, np.inf, -np.inf, np.nan, 1e-310, 1e308], dtype=float)
    grid = np.empty((special.size, special.size), dtype=np.complex128)
    grid.real, grid.imag = special[:, None], special[None, :]
    z = np.concatenate([z, grid.ravel(), [3 + 4j, 1e300 + 1e300j]])
    _same(nat._abs(z), np.abs(z))
    w = rng.standard_normal(z.size) + 1j * rng.standard_normal(z.size)
    for a, b in ((z[:20000], w[:20000]), (w, w[::-1]), (w.real + 0j, w)):
        _same(nat._mul(a, b), a * b)


def _frame_z0(spec, lead, kk):
    sym = _sym(spec, 5)
    lo, hi = ref.keyed_span_pos(spec, lead)
    t_lo, t_hi = lo * L_T_SYM - ref.KEY_RAMP_S, hi * L_T_SYM + ref.KEY_RAMP_S
    n0 = int(np.floor((t_lo - 4.0) * 250))
    n1 = int(np.ceil((t_hi + 4.0) * 250))
    return sym, ref.composite(sym, spec, 250, n0, n1 - n0, keying=kk, lead_in_s=lead)


@pytest.mark.parametrize("name", ["l-tiny", "l-short"])
def test_clip(nat, name):
    spec = LF.get(name)
    k = morse.keying_units(CALL) if spec.n_win else None
    _, z0 = _frame_z0(spec, 5.0, k)
    err = np.max(np.abs(nat.clip(z0) - ref.clip(z0)))
    assert err <= 1e-9
    assert err <= 1e-13                                   # measured ~1e-15
    for ov in ((1.0,), (2.0,), (1.5,), (0.5, 3.0), ()):
        err = np.max(np.abs(nat.clip(z0, overshoot=ov) - ref.clip(z0, overshoot=ov)))
        assert err <= 1e-13, ov
    err = np.max(np.abs(nat.clip(z0, 250, 0.9, guard_hz=2.0) - ref.clip(z0, 250, 0.9, guard_hz=2.0)))
    assert err <= 1e-13
    # Nothing over the threshold: every pass is skipped, the input comes back.
    _same(nat.clip(z0, threshold=1e6), z0)
    _same(ref.clip(z0, threshold=1e6), z0)
    # The input is never modified.
    before = z0.copy()
    nat.clip(z0)
    _same(z0, before)


def test_clip_is_exact_but_for_fft_and_pow(nat, monkeypatch):
    """Where the clipper's tolerance comes from: with numpy.fft (pocketfft,
    as the port's dsp::fft) in place of scipy.fft (ducc0) in the reference,
    a pass of overshoot 1 or 2 -- a copy and a square -- is bit for bit;
    1.5 goes through pow, numpy's own (SVML on AVX-512) against libm."""
    import types

    spec = LF.get("l-short")
    _, z0 = _frame_z0(spec, 5.0, morse.keying_units(CALL))
    monkeypatch.setattr(ref, "sfft", types.SimpleNamespace(
        next_fast_len=scipy.fft.next_fast_len, fftfreq=scipy.fft.fftfreq,
        fft=np.fft.fft, ifft=np.fft.ifft))
    for ov in ((1.0,), (2.0,), (1.0, 2.0)):
        _same(nat.clip(z0, overshoot=ov), ref.clip(z0, overshoot=ov))
    assert np.max(np.abs(nat.clip(z0) - ref.clip(z0))) <= 1e-13


def test_division_by_a_id_is_numpys():
    """frame_baseband's final `z /= A_ID` is numpy's complex division by
    (A_ID + 0j), which the port writes out as Smith's form; numpy's loop
    gives exactly that (and, here, z * (1/A_ID))."""
    rng = np.random.default_rng(10)
    z = (rng.standard_normal(100000) + 1j * rng.standard_normal(100000)) * 10.0 ** rng.uniform(-5, 5, 100000)
    w = z.copy()
    w /= ref.A_ID
    rat = 0.0 / ref.A_ID
    scl = 1.0 / (ref.A_ID + 0.0 * rat)
    want = np.empty_like(z)
    want.real, want.imag = (z.real + z.imag * rat) * scl, (z.imag - z.real * rat) * scl
    _same(w, want)


@pytest.mark.parametrize("name", ["l-tiny", "l-short"])
@pytest.mark.parametrize("lead", [0.0, 5.0])
def test_frame_baseband(nat, name, lead):
    spec = LF.get(name)
    sym = _sym(spec, 6)
    for kk in (None, morse.keying_units(CALL)) if spec.n_win else (None,):
        a = ref.frame_baseband(sym, spec, keying=kk, lead_in_s=lead)
        b = nat.frame_baseband(sym, spec, keying=kk, lead_in_s=lead)
        assert (b.n0, b.fs) == (a.n0, a.fs) and type(b.n0) is int and type(b.fs) is int
        assert b.z.dtype == np.complex128 and b.z.shape == a.z.shape
        assert not b.z.flags.writeable
        np.testing.assert_array_equal(b.z == 0, a.z == 0)  # the same samples zeroed
        err = np.max(np.abs(b.z - a.z))
        assert err <= 1e-9
        assert err <= 1e-13                                # measured ~1e-15


def test_frame_baseband_arguments(nat):
    spec = LF.get("l-tiny")
    sym = _sym(spec, 7)
    for kw in (dict(fs=500), dict(ramp_s=0.2), dict(margin_s=2.5), dict(fs=1000, margin_s=1.0)):
        a, b = ref.frame_baseband(sym, spec, **kw), nat.frame_baseband(sym, spec, **kw)
        assert (b.n0, b.fs, b.z.shape) == (a.n0, a.fs, a.z.shape), kw
        assert np.max(np.abs(b.z - a.z)) <= 1e-13, kw
    for bad in (dict(lead_in_s=10.5), dict(lead_in_s=-2)):
        assert _outcome(nat.frame_baseband, sym, spec, **bad) == \
            _outcome(ref.frame_baseband, sym, spec, **bad)
    assert _outcome(nat.frame_baseband, sym[:-1], spec) == _outcome(ref.frame_baseband, sym[:-1], spec)
    with pytest.raises(TypeError):
        nat.frame_baseband(sym, spec, None)               # keyword-only, as the reference


# --- to 8 kHz ---------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def short_bb():
    spec = LF.get("l-short")
    return ref.frame_baseband(_sym(spec, 8), spec, keying=morse.keying_units(CALL), lead_in_s=5.0)


def test_resample(nat, short_bb):
    bb = short_bb
    for fs_out in (FS, FE_FS, 250, 500):
        up = fs_out // bb.fs
        for i0, m in ((bb.n0 * up - 1000, len(bb.z) * up + 2000), (0, 5000), (-10**7, 100),
                      (bb.n0 * up + 7, 1), (3, 0)):
            _same(nat.resample(bb, fs_out, i0, m), ref.resample(bb, fs_out, i0, m))


def test_resample_chunks(nat, short_bb):
    """Any chunking of the output gives the same bits (the reference's
    windowed upfirdn does too: each sample is its own ascending sum)."""
    bb = short_bb
    i0, m = bb.n0 * 32 - 500, len(bb.z) * 32 + 1000
    whole = nat.resample(bb, FS, i0, m)
    rng = np.random.default_rng(9)
    cuts = np.sort(rng.choice(np.arange(1, m), 40, replace=False))
    edges = [0, *cuts.tolist(), m]
    parts = [nat.resample(bb, FS, i0 + a, b - a) for a, b in zip(edges[:-1], edges[1:])]
    _same(np.concatenate(parts), whole)
    _same(np.concatenate([nat.resample(bb, FS, i0 + a, 4000) for a in range(0, m, 4000)])[:m], whole)


def test_resample_refusals(nat, short_bb):
    bb = short_bb
    for fs_out in (300, 8001, 8000.5, 100):
        assert _outcome(nat.resample, bb, fs_out, 0, 10) == _outcome(ref.resample, bb, fs_out, 0, 10)
    with pytest.raises(ValueError):
        nat.resample(bb, FS, 0, -1)
    with pytest.raises(ValueError):
        ref.resample(bb, FS, 0, -1)


def test_baseband_object(nat, short_bb):
    b = nat.Baseband(short_bb.z, short_bb.n0, short_bb.fs)
    assert (b.n0, b.fs) == (short_bb.n0, short_bb.fs)
    _same(nat.resample(b, FS, 0, 1000), ref.resample(short_bb, FS, 0, 1000))


def test_to_data_centre(nat, short_bb):
    _same(nat.to_data_centre(short_bb), ref.to_data_centre(short_bb))

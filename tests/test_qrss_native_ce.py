"""C++ CE transmit side (native/core/qrss/ce) against sstvae.qrss.ce.

What is pure arithmetic is compared bit for bit everywhere: `_grid_sum` and
so `phase_grid`/`var_grid` (a direct sum in upfirdn's ascending order, on
the same table), the OOK amplitude, the keyed span, `_mulmod`, the
carrier's fraction, and every refusal's type and text. What goes through
sin/cos -- the closed-form pulse, `_hann_int`, the ramp, every phasor, so
`baseband` and `to_audio` -- is bit for bit when numpy's float64 trig is
the C library's (probed below; true of the reference platform) and within
1e-12 otherwise, and `to_audio` additionally needs numpy's complex multiply
to be the fused one the C++ reproduces. Skips when the extension is not
built; `pytest --native-qrss=tx` runs the rest of the QRSS suites against
the same code.
"""

import math
from decimal import Decimal
from fractions import Fraction

import numpy as np
import pytest

from qrss_native import require
from sstvae.qrss import ce as C
from sstvae.qrss import frame as F
from sstvae.qrss import morse
from sstvae.qrss.constants import CW_UNITS, LEAD_IN_MAX_S, T_NUM, T_SYM

S, T = F.SHORT, F.TINY
EDGE = F.FrameSpec("edge", n_data=4096, cw_after=(3301, 7670))   # a window at the very end


def _libm_trig() -> bool:
    """numpy's float64 sin/cos are the C library's (what the C++ calls)."""
    x = np.concatenate([np.linspace(-60.0, 60.0, 20011), np.random.default_rng(0).uniform(0, 7, 5000)])
    return (np.array_equal(np.cos(x), [math.cos(v) for v in x])
            and np.array_equal(np.sin(x), [math.sin(v) for v in x]))


def _fused_cmul() -> bool:
    """numpy's complex multiply forms Re as fma(ar, br, -(ai*bi)), as the C++ does."""
    rng = np.random.default_rng(1)
    a = rng.standard_normal(4096) + 1j * rng.standard_normal(4096)
    b = np.exp(1j * rng.uniform(0, 2 * np.pi, 4096))
    want = [math.fma(p.real, q.real, -(p.imag * q.imag)) for p, q in zip(a, b)]
    return np.array_equal((a * b).real, want)


LIBM = _libm_trig()
FUSED = _fused_cmul()


@pytest.fixture(scope="module")
def nat(native):
    return require(native, "ce")


def _same(got, want, what=""):
    """Equal bytes, dtype and shape (so -0.0 != 0.0 and NaNs compare)."""
    got, want = np.asarray(got), np.asarray(want)
    assert got.dtype == want.dtype, (what, got.dtype, want.dtype)
    assert got.shape == want.shape, (what, got.shape, want.shape)
    assert np.ascontiguousarray(got).tobytes() == np.ascontiguousarray(want).tobytes(), what


def _close(got, want, tol, exact, what=""):
    """Bit for bit when `exact`, else within `tol` (same dtype and shape either way)."""
    if exact:
        _same(got, want, what)
        return
    got, want = np.asarray(got), np.asarray(want)
    assert got.dtype == want.dtype and got.shape == want.shape, what
    if got.size:
        assert np.max(np.abs(got - want)) <= tol, (what, np.max(np.abs(got - want)))


def _raises_alike(f_nat, f_ref):
    """Both raise the same exception type with the same text."""
    with pytest.raises(Exception) as want:
        f_ref()
    with pytest.raises(type(want.value)) as got:
        f_nat()
    assert type(got.value) is type(want.value)
    assert str(got.value) == str(want.value)


def _sym(spec, seed=7):
    return np.random.default_rng(seed).standard_normal(spec.n_sym)


# --- the pulse ---------------------------------------------------------------------

def test_module_values(nat):
    tab = np.array_equal(nat.PULSE_TABLE, C.PULSE_TABLE)
    assert tab or not LIBM, "the frozen table is the reference's wherever numpy's trig is libm's"
    _close(nat.PULSE_TABLE, C.PULSE_TABLE, 1e-15, LIBM, "PULSE_TABLE")
    _close(nat.PULSE_NORM, C.PULSE_NORM, 4e-16, LIBM, "PULSE_NORM")
    _close(nat.G0, C.G0, 4e-16, tab, "G0")
    for name in ("GRID_FS", "HALF", "OOK_FIRST_KEYED_UNIT", "OOK_LAST_KEYED_UNIT", "FSK_SHIFT_HZ", "_ALPHA", "_BETA"):
        assert getattr(nat, name) == getattr(C, name), name
    assert not nat.PULSE_TABLE.flags.writeable


def test_rrc_and_pulse(nat):
    rng = np.random.default_rng(2)
    m = np.arange(-C.HALF - 600, C.HALF + 600)
    special = np.array([0.0, -0.0, 1e-13, -1e-13, 5 / 3, -5 / 3, 5 / 3 + 1e-11, 8.0, -8.0, 8.0000001,
                        np.nextafter(8.0, 9.0), 1e3, -1e3])
    for tau in (m / T_NUM, rng.uniform(-10, 10, 50000), special, rng.uniform(-9, 9, (30, 7))):
        _close(nat._rrc(tau), C._rrc(tau), 1e-15, LIBM, "_rrc")
        _close(nat.pulse(tau), C.pulse(tau), 1e-15, LIBM, "pulse")
    _close(nat.pulse(0.25), C.pulse(0.25), 1e-15, LIBM, "a scalar is a 0-d array")
    _close(nat._rrc([0.5, 1.5]), C._rrc([0.5, 1.5]), 1e-15, LIBM, "a list")


def test_pulse_offsets(nat):
    rng = np.random.default_rng(3)
    d = np.concatenate([rng.uniform(-0.5, 0.5, 4000), [0.0, -0.0, 0.5, -0.5, 1 / 3, -1 / 3, 1e-13]])
    for square in (False, True):
        got = list(nat._pulse_offsets(d, square))
        want = list(C._pulse_offsets(d, square))
        assert [j for j, _ in got] == [j for j, _ in want] == list(range(-8, 9))
        for (j, g), (_, w) in zip(got, want):
            _close(g, w, 1e-15, LIBM, f"_pulse_offsets j={j} square={square}")


# --- phase on the grid (exact) and in closed form ----------------------------------

@pytest.mark.parametrize("spec", [S, T, EDGE], ids=lambda s: s.name)
def test_grid_sum_is_upfirdn(nat, spec):
    """The direct sum is upfirdn's, bit for bit, on the same table and values."""
    vals = C.at_positions(_sym(spec), spec)
    sq = C.PULSE_TABLE ** 2
    for fs, n0, n in [(8000, -1940, 480000), (16000, -3880, 300000), (4000, 0, 50000),
                      (2000, -7, 9999), (250, -61, 9000), (125, 3, 4000), (1, 0, 300),
                      (8000, 0, 1), (8000, -10 ** 6, 5000), (8000, spec.n_pos * 243 - 3000, 9000),
                      (16000, 1, 0), (8000, 7, -3)]:
        _same(nat._grid_sum(fs, n0, n, vals, C.PULSE_TABLE), C._grid_sum(fs, n0, n, vals, C.PULSE_TABLE),
              f"_grid_sum fs={fs} n0={n0} n={n}")
        _same(nat._grid_sum(fs, n0, min(n, 20000), vals, sq), C._grid_sum(fs, n0, min(n, 20000), vals, sq),
              f"_grid_sum squared fs={fs} n0={n0}")


def test_phase_and_var_grid(nat):
    exact = np.array_equal(nat.PULSE_TABLE, C.PULSE_TABLE)
    sym = _sym(S)
    var = np.random.default_rng(8).uniform(0, 2, S.n_sym)
    # 60 s at 8 kHz from 1940 samples before t0, the transmitter's own call
    _close(nat.phase_grid(8000, -1940, 480000, sym, S), C.phase_grid(8000, -1940, 480000, sym, S),
           1e-12, exact, "phase_grid 60 s")
    _close(nat.var_grid(8000, -1940, 480000, var, S), C.var_grid(8000, -1940, 480000, var, S),
           1e-12, exact, "var_grid 60 s")
    for fs in (16000, 4000, 250):
        _close(nat.phase_grid(fs, 5, 20000, sym, S), C.phase_grid(fs, 5, 20000, sym, S), 1e-12, exact, fs)
    _close(nat.phase_grid(8000, 0, 100, list(sym), S), C.phase_grid(8000, 0, 100, list(sym), S),
           1e-12, exact, "a list of symbols")
    _same(nat.at_positions(sym, S), C.at_positions(sym, S), "at_positions")


def test_phase_at(nat):
    rng = np.random.default_rng(4)
    sym = _sym(S)
    var = rng.uniform(0, 2, S.n_sym)
    k = np.arange(-12, S.n_pos + 12)
    taus = [
        rng.uniform(-20, S.n_pos + 20, 60000),                      # unordered
        np.concatenate([k, k + 1 / 3, k - 1 / 3, k + 0.5, k - 0.5]),  # the singular offsets, the half-way points
        rng.uniform(-5, 60, (40, 25)),                               # 2-D keeps its shape
        np.array([-1e9, 1e9, -9.6, S.n_pos + 8.4]),                 # nowhere near the frame
        np.array(1234.25),
    ]
    for tau in taus:
        got, gv = nat.phase_at(tau, sym, S)
        want, wv = C.phase_at(tau, sym, S)
        assert gv is None and wv is None
        _close(got, want, 1e-12, LIBM, "phase_at")
        got, gv = nat.phase_at(tau, sym, S, var=var, chunk=1001)
        want, wv = C.phase_at(tau, sym, S, var=var, chunk=1001)
        _close(got, want, 1e-12, LIBM, "phase_at with var")
        _close(gv, wv, 1e-12, LIBM, "phase_at's v")
    # The closed form agrees with the table path at the grid times.
    tau = np.arange(-1940, 60000 - 1940) * 2 / T_NUM
    phi, _ = nat.phase_at(tau, sym, S)
    assert np.max(np.abs(phi - nat.phase_grid(8000, -1940, 60000, sym, S))) < 1e-12


def test_small_refusals(nat):
    for fs in (3, 0, -8000, 32000, 3000):
        _raises_alike(lambda: nat._grid_step(fs), lambda: C._grid_step(fs))
    assert [nat._grid_step(fs) for fs in (16000, 8000, 250, 1)] == [1, 2, 64, 16000]
    for bad in (np.zeros(5), np.zeros((2, 3)), np.zeros(T.n_sym + 1), 0.0):
        _raises_alike(lambda: nat._check_sym(bad, T), lambda: C._check_sym(bad, T))
        _raises_alike(lambda: nat._check_sym(bad, T, "var"), lambda: C._check_sym(bad, T, "var"))
        _raises_alike(lambda: nat.phase_grid(8000, 0, 10, bad, T), lambda: C.phase_grid(8000, 0, 10, bad, T))
        _raises_alike(lambda: nat.at_positions(bad, T), lambda: C.at_positions(bad, T))
    _raises_alike(lambda: nat.phase_grid(3000, 0, 10, _sym(T), T), lambda: C.phase_grid(3000, 0, 10, _sym(T), T))
    _raises_alike(lambda: nat.var_grid(8000, 0, 10, np.zeros(3), T), lambda: C.var_grid(8000, 0, 10, np.zeros(3), T))
    _raises_alike(lambda: nat.phase_at([0.0], _sym(T), T, var=np.zeros(4)),
                  lambda: C.phase_at([0.0], _sym(T), T, var=np.zeros(4)))


# --- the callsign windows ----------------------------------------------------------

def test_hann_int(nat):
    y = np.concatenate([np.linspace(-1.5, 2.5, 40001), [-0.5, 0.5, -0.0, np.nextafter(0.5, 1)]])
    _close(nat._hann_int(y), C._hann_int(y), 2e-16, LIBM, "_hann_int")


def test_window_keying(nat):
    k = morse.keying_units("K1ABC/P")
    two = np.stack([k, morse.keying_units("W1AW")])
    accept = [None, k, k.astype(np.int64), k.astype(bool), k.astype(np.float64), list(k), two,
              two.astype(np.int32), np.zeros(CW_UNITS), np.ones((2, CW_UNITS), np.uint8)]
    for x in accept:
        got, want = nat.window_keying(x, S), C.window_keying(x, S)
        if want is None:
            assert got is None
        else:
            _same(got, want, "window_keying")
    _same(nat.window_keying(k, T), C.window_keying(k, T), "no windows: (0, 192)")
    refuse = [np.zeros(191), np.full(CW_UNITS, 2), np.full(CW_UNITS, -1), np.full(CW_UNITS, 0.5),
              np.zeros((3, CW_UNITS)), np.zeros((2, 191)), np.zeros((1, CW_UNITS)), np.zeros((2, 2, CW_UNITS)),
              np.full(CW_UNITS, np.nan)]
    for x in refuse:
        _raises_alike(lambda: nat.window_keying(x, S), lambda: C.window_keying(x, S))


def test_window_units(nat):
    tau = np.linspace(4990, 8500, 3511)
    for w in (0, 1, -1, -2):
        _same(nat.window_units(tau, S, w), C.window_units(tau, S, w), f"window_units w={w}")
    assert nat.window_units(5100.0, S, 0) == C.window_units(5100.0, S, 0)
    for w in (2, -3):
        _raises_alike(lambda: nat.window_units(tau, S, w), lambda: C.window_units(tau, S, w))
    _raises_alike(lambda: nat.window_units(tau, T, 0), lambda: C.window_units(tau, T, 0))


@pytest.mark.parametrize("call", ["K1ABC/P", "W1AW", "N0CALL", None])
def test_cw_phase_turns(nat, call):
    k = None if call is None else morse.keying_units(call)
    for spec in (S, EDGE):
        n0, n = -1940, spec.n_pos * 243 + 4000
        tau = (np.arange(n0, n0 + n, dtype=np.int64) * 2) / T_NUM       # the 8 kHz grid, as baseband makes it
        rng = np.random.default_rng(5)
        mixed = rng.permutation(tau[1200000:1300000])                    # unordered: the masked path
        for t in (tau, mixed, tau[1214000:1216100].reshape(30, 70)):
            got_th, got_a = nat.cw_phase_turns(t, spec, k, ook=True)
            want_th, want_a = C.cw_phase_turns(t, spec, k, ook=True)
            _same(got_a, want_a, f"OOK amplitude {call} {spec.name}")
            _same(got_th, want_th, f"OOK theta {call}")
            got_th, got_a = nat.cw_phase_turns(t, spec, k)
            want_th, want_a = C.cw_phase_turns(t, spec, k)
            _same(got_a, want_a, "FSK amplitude")
            _close(got_th, want_th, 1e-12, LIBM, f"FSK theta {call} {spec.name}")
    two = None if k is None else np.stack([k, morse.keying_units("K9Z")])
    tau = np.linspace(4990, 8500, 30000)
    _close(nat.cw_phase_turns(tau, S, two)[0], C.cw_phase_turns(tau, S, two)[0], 1e-12, LIBM, "one mask per window")
    _close(nat.cw_phase_turns(4999.75, S, k)[0], C.cw_phase_turns(4999.75, S, k)[0], 1e-12, LIBM, "0-d")


# --- the keyed span and its ramp ---------------------------------------------------

def test_keyed_span_pos(nat):
    leads = [*np.linspace(0.0, LEAD_IN_MAX_S, 81), 0, 10, -0.0, 5e-324, np.float64(2.5), np.nextafter(10.0, 0)]
    for spec in (S, T, EDGE):
        for lead in leads:
            got, want = nat.keyed_span_pos(spec, lead), C.keyed_span_pos(spec, lead)
            assert type(got) is tuple and len(got) == 2
            assert [float(v).hex() for v in got] == [float(v).hex() for v in want], (spec.name, lead)
        assert nat.keyed_span_pos(spec) == C.keyed_span_pos(spec)
    for lead in (10.5, -1, 11, -1e-300, np.nextafter(10.0, 11), float("nan"), float("inf"), -float("inf")):
        _raises_alike(lambda: nat.keyed_span_pos(S, lead), lambda: C.keyed_span_pos(S, lead))


def test_ramp_envelope(nat):
    for ramp_s in (0.05, 0.0, 0.2):
        ramp = ramp_s / T_SYM
        for lead in (0.0, 3.0):
            lo, hi = C.keyed_span_pos(S, lead)
            tau = np.concatenate([np.linspace(lo - 3, lo + 3, 20001), np.linspace(hi - 3, hi + 3, 20001),
                                  [lo, hi, lo - ramp, hi + ramp, np.nextafter(lo, -1e9), np.nextafter(hi, -1e9)]])
            _close(nat._ramp_envelope(tau, lo, hi, ramp), C._ramp_envelope(tau, lo, hi, ramp), 1e-15, LIBM,
                   f"_ramp_envelope {ramp_s} s")


def test_turns_to_phasor(nat):
    rng = np.random.default_rng(6)
    t = np.concatenate([rng.uniform(-5, 5, 20000), rng.uniform(-1e7, 1e7, 2000), [0.0, -0.0, 0.25, 0.5, 1.0, -1e-17]])
    _close(nat._turns_to_phasor(t), C._turns_to_phasor(t), 2e-16, LIBM, "_turns_to_phasor")
    _close(nat._turns_to_phasor(t.reshape(-1, 2)), C._turns_to_phasor(t.reshape(-1, 2)), 2e-16, LIBM, "2-D")


# --- baseband ----------------------------------------------------------------------

def _stream_len(spec, start_s, fs):
    return int(round((spec.keyed_end_pos * T_SYM + 0.5 - start_s) * fs))


K = morse.keying_units("K1ABC/P")
BB = [
    # (spec, fs, start_s, keyword arguments) -- n runs to past the keyed end
    (T, 8000, -1.0, {}),
    (T, 8000, 0.0, dict(ramp_s=0.0)),
    (T, 8000, 3.3, dict(lead_in_s=2.5)),
    (T, 8000, -12.0, dict(lead_in_s=10.0)),
    (T, 4000, -0.3, {}),
    (T, 250, -0.3, {}),
    (T, 16000, 0.5, {}),
    (T, 8000, 3.3e-5, {}),                                      # off the grid: closed form
    (T, 8000, -0.75, dict(time_scale=1.0001)),                  # a clock offset: closed form
    (T, 3000, -0.5, {}),                                        # 3 kHz does not divide 16 kHz
    (S, 8000, -1.0, {}),
    (S, 8000, -1.0, dict(keying=K)),
    (S, 8000, -1.0, dict(keying=K, ook=True)),
    (S, 8000, -1.0, dict(ook=True)),
    (S, 8000, 140.0, dict(keying=np.stack([K, morse.keying_units("W1AW")]), lead_in_s=1.0)),
    (S, 8000, 140.0001, dict(keying=K, ook=True, time_scale=0.99995)),
    (EDGE, 4000, 200.0, dict(keying=K)),
]


@pytest.mark.parametrize("case", BB, ids=[f"{s.name}-{fs}-{st}-{'-'.join(kw)}" for s, fs, st, kw in BB])
def test_baseband(nat, case):
    spec, fs, start_s, kw = case
    sym = _sym(spec)
    n = _stream_len(spec, start_s, fs)
    if kw.get("time_scale", 1.0) != 1.0 or fs == 3000:
        n = min(n, 120000)                                       # the closed form is the slow path
    got = nat.baseband(sym, spec, fs, start_s, n, **kw)
    want = C.baseband(sym, spec, fs, start_s, n, **kw)
    _close(got, want, 1e-12, LIBM, "baseband")
    assert np.count_nonzero(want) > 0


def test_baseband_refusals(nat):
    sym = _sym(T)
    cases = [
        (np.zeros(5), T, 8000, 0.0, 10, {}),
        (sym, T, 8000, 0.0, -1, {}),
        (sym, T, 8000, 0.0, 10, dict(lead_in_s=11.0)),
        (sym, T, 8000, 0.0, 10, dict(lead_in_s=-1)),
        (sym, S, 8000, 0.0, 10, {}),
        (_sym(S), S, 8000, 0.0, 10, dict(keying=np.zeros(191))),
        (_sym(S), S, 8000, 0.0, 10, dict(keying=np.full(CW_UNITS, 3))),
        (np.zeros(5), S, 8000, 0.0, 10, dict(keying=np.zeros(191))),   # sym first
        (sym, T, 8000, 0.0, 10, dict(lead_in_s=11.0, keying=np.zeros(3))),   # the lead-in before the keying
        (sym, T, 8000, float("nan"), 10, {}),
        (sym, T, 8000, float("inf"), 10, {}),
        (sym, T, 8000, 0.0, 10, dict(chunk=0)),
    ]
    for s, spec, fs, st, n, kw in cases:
        _raises_alike(lambda: nat.baseband(s, spec, fs, st, n, **kw), lambda: C.baseband(s, spec, fs, st, n, **kw))
    _same(nat.baseband(sym, T, 8000, 0.0, 0), C.baseband(sym, T, 8000, 0.0, 0), "n = 0")
    _same(nat.baseband(sym, T, 8000, -100.0, 50), C.baseband(sym, T, 8000, -100.0, 50), "nothing keyed")


def test_baseband_chunk_invariance(nat):
    """The chunk only decides where the work splits: bit for bit in the C++."""
    sym = _sym(S)
    for start_s, n, kw in [(150.0, 120000, dict(keying=K)), (150.0, 120000, dict(keying=K, ook=True)),
                           (-0.75, 30000, {}), (150.0001, 20000, dict(keying=K, time_scale=1.0001))]:
        ref = nat.baseband(sym, S, 8000, start_s, n, **kw)
        for chunk in (1 << 18, 4096, 1000, 7):
            _same(nat.baseband(sym, S, 8000, start_s, n, chunk=chunk, **kw), ref, f"chunk {chunk}")
    small = nat.baseband(sym, S, 8000, -0.3, 2000, chunk=1)
    _same(small, nat.baseband(sym, S, 8000, -0.3, 2000), "chunk 1 across the ramp")


# --- to_audio ----------------------------------------------------------------------

CARRIERS = [(1200, 1), (1500, 1), (2650, 1), C.carrier_fraction(1500.3)]


@pytest.mark.parametrize("carrier", CARRIERS, ids=[f"{n}/{d}" for n, d in CARRIERS])
def test_to_audio(nat, carrier):
    num, den = carrier
    rng = np.random.default_rng(9)
    z_bb = C.baseband(_sym(T), T, 8000, -1.0, 100000)
    z_rand = rng.standard_normal(30000) + 1j * rng.standard_normal(30000)
    exact = LIBM and FUSED
    for z in (z_bb, z_rand):
        for n0 in (0, 1, 123456789, -5, 2 ** 40, 79999):
            for amp in (0.5, 0.25):
                _close(nat.to_audio(z, 8000, num, den, amp, n0), C.to_audio(z, 8000, num, den, amp, n0),
                       1e-12, exact, f"to_audio {num}/{den} n0={n0}")
    for fs in (250, 4000, 2000, 16000):
        z = z_rand[:4000]
        _close(nat.to_audio(z, fs, num, den), C.to_audio(z, fs, num, den), 1e-12, exact, f"to_audio from {fs} Hz")
    _close(nat.to_audio(list(z_rand[:10]), 8000, num, den), C.to_audio(list(z_rand[:10]), 8000, num, den),
           1e-12, exact, "a list")
    _close(nat.to_audio(z_rand[:10].real, 8000, num, den), C.to_audio(z_rand[:10].real, 8000, num, den),
           1e-12, exact, "real input")


def test_to_audio_chunked_and_refusals(nat):
    rng = np.random.default_rng(10)
    z = rng.standard_normal(50000) + 1j * rng.standard_normal(50000)
    whole = nat.to_audio(z, 8000, 15003, 10, 0.5, 777)
    parts = np.concatenate([nat.to_audio(z[i:i + 4097], 8000, 15003, 10, 0.5, 777 + i)
                            for i in range(0, len(z), 4097)])
    _same(parts, whole, "to_audio in chunks with the running count")
    for den in (0, -10):
        _raises_alike(lambda: nat.to_audio(z, 8000, 1500, den), lambda: C.to_audio(z, 8000, 1500, den))
    _same(nat.to_audio(z[:0], 8000, 1500), C.to_audio(z[:0], 8000, 1500), "empty")


def test_mulmod(nat):
    rng = np.random.default_rng(11)
    for _ in range(3000):
        m = int(rng.integers(1, 2 ** 63 - 1))
        a, b = int(rng.integers(0, m)), int(rng.integers(0, m))
        assert nat._mulmod(a, b, m) == (a * b) % m
    for a, b, m in [(0, 5, 7), (2 ** 62, 2 ** 62, 2 ** 63 - 25), (2 ** 63 - 26, 2 ** 63 - 26, 2 ** 63 - 25),
                    (79999, 15003, 80000), (123456789012345, 987654321098765, 8000 << 20)]:
        assert nat._mulmod(a, b, m) == (a * b) % m


# --- the carrier --------------------------------------------------------------------

def test_check_carrier(nat):
    accept = [300, 1500, 2700, 1500.3, 300.0, 2700.0, "1500", "2650.5", np.float64(1200.5), np.int64(1800),
              Fraction(3001, 2), True * 1500]
    for f in accept:
        got, want = nat.check_carrier(f), C.check_carrier(f)
        assert type(got) is float and got == want, f
    refuse = [299.99, 2700.01, float("nan"), float("inf"), -float("inf"), -1500, 0, "5000", "nan", 1e300,
              np.float64(250.0), Fraction(6001, 2), 3000]
    for f in refuse:
        _raises_alike(lambda: nat.check_carrier(f), lambda: C.check_carrier(f))
    for f in ("abc", None, [1500]):
        _raises_alike(lambda: nat.check_carrier(f), lambda: C.check_carrier(f))


def test_carrier_fraction(nat):
    accept = [1200, 1500.3, "1500.3", "2650", 300, 2700, "3/7", 1234.56789, 1500.0000001, "1e3", "-1500",
              Fraction(15003, 10), Fraction(1, 3), np.int64(1500), np.float64(1500.3), 0.1, 1 / 3, 2650.123456789,
              "1500.0000001", "  1500  ", 2 ** 40, 2 ** 70, Decimal("1500.3"), np.uint64(2 ** 63 + 5), True,
              "1_500.3", "+1500", "1500.", ".5", "1E3", "-3/7", "\t1500\n", "1_000/3",
              # past int64 before it is limited: the C++ hands these to Fraction
              "2650.123456789123456789", "1500.30000000000000000001", "99999999999999999999", "1/99999999999999999999",
              "１５００", 1e300, 5e-324, -1e19]
    for c in accept:
        got, want = nat.carrier_fraction(c), C.carrier_fraction(c)
        assert type(got) is tuple and got == want, (c, got, want)
        assert [type(v) for v in got] == [type(v) for v in want], c
    for c in ("abc", "1/0", float("nan"), float("inf"), None, "", [1500], Decimal("nan"),
              np.longdouble(1500.3), np.float32(1500.5), "3/-7", "0x10", "1,5", "\ud800"):
        _raises_alike(lambda: nat.carrier_fraction(c), lambda: C.carrier_fraction(c))

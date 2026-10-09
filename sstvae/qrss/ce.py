"""Waveform CE: the phase pulse, phase synthesis at any rate, the
callsign-window phase, baseband and audio, the mean-waveform template
and the noiseless loopback (design 2.4, 2.7, 2.8; spec 2.5, 2.7, 2.8).

A **format module**: this is what goes on the air.

    s(t) = a(t) * exp(j * [phi(t) + 2 pi theta_cw(t)])
    phi(t) = beta * sum_s x_s * p((t - t0)/T - pos(s))

with x the assembled stream symbols (`frame.assemble`), pos(s) their
positions (windows included), p the unit-energy RRC (alpha = 0.15) cut
at +-8 symbols and renormalised on the 16 kHz grid, and beta = 0.8 rad
rms. theta_cw is the callsign-window phase in turns (FSK keying; 0 for
on-off keying), a(t) the keying envelope: 1 from t0 - 8T (or the start
of an optional lead-in) to the end of the frame, following the Morse in
an on-off-keyed window.

Time is in symbols from t0 throughout: tau = (t - t0)/T, so stream
symbol s is centred at tau = pos(s). On a sample grid at a rate fs that
divides 16000, sample N (counted from t0) sits at 16 kHz grid index
(16000/fs)*N and the pulse offset to position p is the integer
m = (16000/fs)*N - 485*p, so `PULSE_TABLE[m + 3880]` is exact at every
such rate (decision D1). `phase_grid` uses that; `phase_at` evaluates
the pulse in closed form at any tau (clock offsets, templates), and the
two agree to rounding.

The window phase is carried *unwrapped*: after window w it is minus the
number of key-down units sent so far, an integer, so exp() never sees
the difference while a frequency synthesizer (`si5351`) that steers on
phase differences sees a continuous phase.
"""

import math
from fractions import Fraction

import numpy as np
from scipy.signal import resample_poly, upfirdn

from .constants import (
    ALPHA,
    BETA,
    CH_FS,
    CW_UNIT,
    CW_UNITS,
    CW_WINDOW,
    FS,
    K_LIN,
    LEAD_IN_MAX_S,
    SPAN,
    T_DEN,
    T_NUM,
    T_SYM,
)
from .frame import FrameSpec, layout

_ALPHA = float(ALPHA)
_BETA = float(BETA)
GRID_FS = T_DEN                       # 16 kHz: the grid the pulse table lives on
HALF = SPAN * T_NUM                   # 3880 grid steps = 8 symbols

# On-off keying (design 2.7, decision D14 as adopted in spec rev 9): the
# carrier stays on in units 0..4 and 184..191 and follows the Morse in
# units 5..183, where 5..7 are key-up so the call's first element stands
# apart from the plain carrier before it.
OOK_FIRST_KEYED_UNIT = 5
OOK_LAST_KEYED_UNIT = 183
FSK_SHIFT_HZ = -1.0 / (CW_UNIT * T_SYM)   # -16.495 Hz: one cycle per unit, below


# --- pulse ------------------------------------------------------------------------

def _rrc(tau) -> np.ndarray:
    """The textbook unit-energy-in-continuous-time RRC h(tau), uncut, unscaled."""
    tau = np.asarray(tau, dtype=np.float64)
    a = _ALPHA
    out = np.empty_like(tau)
    zero = np.abs(tau) < 1e-12
    den = np.pi * tau * (1.0 - (4.0 * a * tau) ** 2)
    sing = ~zero & (np.abs(1.0 - (4.0 * a * tau) ** 2) < 1e-9)
    reg = ~zero & ~sing
    t = tau[reg]
    out[reg] = (np.sin(np.pi * t * (1 - a)) + 4 * a * t * np.cos(np.pi * t * (1 + a))) / den[reg]
    out[zero] = 1 - a + 4 * a / np.pi
    out[sing] = a / np.sqrt(2) * ((1 + 2 / np.pi) * np.sin(np.pi / (4 * a))
                                  + (1 - 2 / np.pi) * np.cos(np.pi / (4 * a)))
    return out


_M = np.arange(-HALF, HALF + 1)
# c^2 = (1/485) sum_m h(m/485)^2 over the cut pulse: unit energy on the grid.
PULSE_NORM = float(np.sqrt(np.sum(_rrc(_M / T_NUM) ** 2) / T_NUM))   # 0.9999137


def pulse(tau) -> np.ndarray:
    """p(tau) = h(tau)/c for |tau| <= 8 symbols, else 0 (float64)."""
    tau = np.asarray(tau, dtype=np.float64)
    return np.where(np.abs(tau) <= SPAN, _rrc(tau) / PULSE_NORM, 0.0)


PULSE_TABLE = pulse(_M / T_NUM)                 # (7761,) p(m/485), m = -3880..3880
PULSE_TABLE.setflags(write=False)
G0 = float(PULSE_TABLE.sum() / T_NUM)           # 1.001665: matched-filter gain of a constant
_PULSE_TABLE_SQ = PULSE_TABLE ** 2


def _grid_step(fs: int) -> int:
    """16 kHz grid steps per sample at rate fs; ValueError unless fs divides 16000."""
    fs = int(fs)
    if fs <= 0 or GRID_FS % fs:
        raise ValueError(f"fs must divide {GRID_FS}, not {fs}")
    return GRID_FS // fs


# --- symbols on the position axis ------------------------------------------------------

def _check_sym(sym, spec: FrameSpec, name="sym") -> np.ndarray:
    sym = np.asarray(sym, dtype=np.float64)
    if sym.shape != (spec.n_sym,):
        raise ValueError(f"{name} has shape {sym.shape}, expected ({spec.n_sym},)")
    return sym


def at_positions(sym, spec: FrameSpec) -> np.ndarray:
    """float64[n_pos]: stream-order values placed at their positions, 0 in windows."""
    sym = _check_sym(sym, spec)
    out = np.zeros(spec.n_pos, dtype=np.float64)
    out[layout(spec).pos] = sym
    return out


# --- phase ----------------------------------------------------------------------------------

def _pulse_offsets(d: np.ndarray, square: bool):
    """Yield (j, p(d - j)) for j = -8..8, d in [-1/2, 1/2], two trig calls in all.

    sin(pi (d - j) c) = sin(pi d c) cos(pi j c) - cos(pi d c) sin(pi j c), so the
    17 offsets share one sin/cos pair per argument; tau = d - j. Only j = +-8
    can reach past the cut, and only j = 0 (tau = 0) and j = +-2 (tau =
    -+5/3, where 4 alpha |tau| = 1) can meet a removable singularity.
    """
    a = _ALPHA
    c1, c2 = np.pi * (1 - a), np.pi * (1 + a)
    s1, k1 = np.sin(c1 * d), np.cos(c1 * d)
    s2, k2 = np.sin(c2 * d), np.cos(c2 * d)
    for j in range(-SPAN, SPAN + 1):
        tau = d - j
        sn = s1 * math.cos(c1 * j) - k1 * math.sin(c1 * j)
        cs = k2 * math.cos(c2 * j) + s2 * math.sin(c2 * j)
        den = (np.pi * PULSE_NORM) * tau * (1.0 - (4.0 * a * tau) ** 2)
        if j in (0, 2, -2):
            bad = np.abs(den) < 1e-9
            if bad.any():
                den = np.where(bad, 1.0, den)
                p = (sn + 4 * a * tau * cs) / den
                p[bad] = _rrc(tau[bad]) / PULSE_NORM
            else:
                p = (sn + 4 * a * tau * cs) / den
        else:
            p = (sn + 4 * a * tau * cs) / den
        if abs(j) == SPAN:
            p = np.where(np.abs(tau) <= SPAN, p, 0.0)
        yield j, (p * p if square else p)


def phase_at(tau_pos, sym, spec: FrameSpec, var=None, chunk: int = 1 << 19):
    """(phi_mean, v) at arbitrary times tau_pos = (t - t0)/T (float64, any shape).

    phi_mean = beta * sum_s sym_s p(tau - pos_s); with `var` (per stream symbol)
    also v = beta^2 * sum_s var_s p(tau - pos_s)^2, else v is None. Evaluated
    in closed form at 17 offsets per sample.
    """
    tau = np.asarray(tau_pos, dtype=np.float64)
    pad = 2 * SPAN + 2
    vals = np.pad(at_positions(sym, spec), pad)
    vvar = None if var is None else np.pad(
        at_positions(_check_sym(var, spec, "var"), spec), pad)
    flat = tau.reshape(-1)
    phi = np.zeros(flat.shape, dtype=np.float64)
    v = None if vvar is None else np.zeros(flat.shape, dtype=np.float64)
    n_pos = spec.n_pos
    for lo in range(0, len(flat), chunk):
        t = flat[lo:lo + chunk]
        centre = np.floor(t + 0.5)
        ci = centre.astype(np.int64)
        near = (ci >= -SPAN - 1) & (ci <= n_pos + SPAN)
        if not near.any():
            continue
        if not near.all():
            t, ci, centre = t[near], ci[near], centre[near]
        d = t - centre
        base = ci + pad                         # index of position ci in the padded array
        acc = np.zeros(len(t))
        for j, p in _pulse_offsets(d, square=False):
            acc += vals[base + j] * p
        out = np.zeros(min(chunk, len(flat) - lo))
        out[near] = _BETA * acc
        phi[lo:lo + chunk] = out
        if v is not None:
            acc = np.zeros(len(t))
            for j, p2 in _pulse_offsets(d, square=True):
                acc += vvar[base + j] * p2
            out = np.zeros(min(chunk, len(flat) - lo))
            out[near] = _BETA ** 2 * acc
            v[lo:lo + chunk] = out
    phi = phi.reshape(tau.shape)
    return phi, (None if v is None else v.reshape(tau.shape))


def _grid_sum(fs: int, n0: int, n: int, vals: np.ndarray, table: np.ndarray) -> np.ndarray:
    """sum_p vals[p] * table[L*(n0+i) - 485 p + 3880] for i < n, exactly, by upfirdn.

    vals is per position (n_pos,). Upsampling the positions by 485 puts
    position p at grid index 485 p; the table is the filter; keeping every
    L-th output gives the samples. A few leading zeros on the filter align
    the decimation phase with sample n0.
    """
    L = _grid_step(fs)
    n0, n = int(n0), int(n)
    if n <= 0:
        return np.zeros(0)
    n_pos = len(vals)
    # Positions that can reach samples [n0, n0 + n): local window [p_lo, p_hi).
    p_lo = (L * n0) // T_NUM - SPAN - 1
    p_hi = -(-(L * (n0 + n - 1)) // T_NUM) + SPAN + 2
    a, b = max(p_lo, 0), min(p_hi, n_pos)
    if a >= b:
        return np.zeros(n)
    x = np.zeros(p_hi - p_lo)
    x[a - p_lo:b - p_lo] = vals[a:b]
    z0 = (T_NUM * p_lo - HALF) % L
    h = np.concatenate([np.zeros(z0), table])
    first = (L * n0 - T_NUM * p_lo + HALF + z0) // L      # exact: divisible by L
    y = upfirdn(h, x, up=T_NUM, down=L)
    out = y[first:first + n]
    if len(out) < n:
        out = np.concatenate([out, np.zeros(n - len(out))])
    return out


def phase_grid(fs: int, n0: int, n: int, sym, spec: FrameSpec) -> np.ndarray:
    """phi at samples t = t0 + (n0 + i)/fs, i < n, fs dividing 16000 (exact table path)."""
    return _BETA * _grid_sum(fs, n0, n, at_positions(sym, spec), PULSE_TABLE)


def var_grid(fs: int, n0: int, n: int, var, spec: FrameSpec) -> np.ndarray:
    """v = beta^2 sum_s var_s p^2 on the same grid as `phase_grid`."""
    vals = at_positions(_check_sym(var, spec, "var"), spec)
    return _BETA ** 2 * _grid_sum(fs, n0, n, vals, _PULSE_TABLE_SQ)


# --- callsign windows -------------------------------------------------------------------

def _hann_int(y: np.ndarray) -> np.ndarray:
    """R(y): the integral of the Hann CDF centred on 0 with unit width."""
    y = np.asarray(y, dtype=np.float64)
    mid = (y + 0.5) ** 2 / 2 - (np.cos(2 * np.pi * y) + 1) / (4 * np.pi ** 2)
    return np.where(y < -0.5, 0.0, np.where(y > 0.5, y, mid))


def window_keying(keying, spec: FrameSpec) -> np.ndarray | None:
    """(n_win, 192) uint8 from None, one (192,) mask, or one mask per window."""
    if keying is None:
        return None
    k = np.asarray(keying)
    if k.ndim == 1:
        k = np.broadcast_to(k, (spec.n_win, len(k)))
    if k.shape != (spec.n_win, CW_UNITS) or np.any((k != 0) & (k != 1)):
        raise ValueError(f"keying must be 0/1 of shape ({CW_UNITS},) or "
                         f"({spec.n_win}, {CW_UNITS}), not {k.shape}")
    return k.astype(np.uint8)


def window_units(tau_pos, spec: FrameSpec, w: int) -> np.ndarray:
    """x = (tau - (P_w - 1/2)) / 2: time in Morse units from the start of window w."""
    p_w = spec.win_start_pos[w]
    return (np.asarray(tau_pos, dtype=np.float64) - (p_w - 0.5)) / CW_UNIT


def cw_phase_turns(tau_pos, spec: FrameSpec, keying, ook: bool = False):
    """(theta_cw in turns, a) at times tau_pos (design 2.7).

    FSK: theta = -sum_u k[u] (R(x - u) - R(x - u - 1)) within each window
    (x in units), accumulated across windows so it is continuous: after
    window w it is the integer -(key-down units so far). a = 1.
    OOK: theta = 0; a = 0 in units 5..183 of each window except where the
    keying is down. `keying` None means no call keyed (plain carrier for
    FSK, carrier off in units 5..183 for OOK).
    """
    tau = np.asarray(tau_pos, dtype=np.float64)
    theta = np.zeros(tau.shape)
    a = np.ones(tau.shape)
    k_all = window_keying(keying, spec)
    flat_t, flat_th, flat_a = tau.reshape(-1), theta.reshape(-1), a.reshape(-1)
    ordered = flat_t.size < 2 or bool(flat_t[-1] >= flat_t[0] and
                                      np.all(flat_t[1:] >= flat_t[:-1]))
    for w in range(spec.n_win):
        k = np.zeros(CW_UNITS, np.uint8) if k_all is None else k_all[w]
        p0 = spec.win_start_pos[w] - 0.5
        p1 = p0 + CW_WINDOW
        if ordered:                              # the usual case: a time axis
            i0, i1 = np.searchsorted(flat_t, [p0, p1])
            inside = slice(i0, i1)
            after = slice(i1, None)
        else:
            inside = (flat_t >= p0) & (flat_t < p1)
            after = flat_t >= p1
        x = (flat_t[inside] - p0) / CW_UNIT
        if ook:
            u = np.floor(x).astype(np.int64)
            keyed = (u >= OOK_FIRST_KEYED_UNIT) & (u <= OOK_LAST_KEYED_UNIT)
            flat_a[inside] = np.where(keyed, k[np.clip(u, 0, CW_UNITS - 1)], 1.0)
            continue
        total = int(k.sum())
        if total == 0:
            continue
        flat_th[after] -= total
        f = np.floor(x).astype(np.int64)
        # Units at or before f - 2 are complete (R(y) - R(y - 1) = 1 for y >= 3/2),
        # units after f + 1 have not started; f - 1, f, f + 1 are in transition.
        cum = np.concatenate([[0], np.cumsum(k, dtype=np.int64)])   # cum[i] = sum k[:i]
        th = -cum[np.clip(f - 1, 0, CW_UNITS)].astype(np.float64)
        for j in (-1, 0, 1):
            u = f + j
            ok = (u >= 0) & (u < CW_UNITS)
            ku = np.where(ok, k[np.clip(u, 0, CW_UNITS - 1)], 0)
            y = x - u
            th -= ku * (_hann_int(y) - _hann_int(y - 1))
        flat_th[inside] += th
    return theta, a


# --- keyed span ------------------------------------------------------------------------------

def keyed_span_pos(spec: FrameSpec, lead_in_s: float = 0.0) -> tuple[float, float]:
    """[start, end) of the keyed carrier in symbols from t0 (sender's clock).

    Keying starts 8 symbols before t0, or `lead_in_s` seconds earlier still
    with a lead-in (spec 2.8: the lead-in lasts lead_in_s and ends where the
    preamble's first pulse begins). It ends with the frame (design 2.2).
    """
    lead = float(lead_in_s)
    if not 0.0 <= lead <= LEAD_IN_MAX_S:
        raise ValueError(f"lead-in must be 0 to {LEAD_IN_MAX_S} s, not {lead_in_s}")
    return spec.keyed_start_pos - lead / T_SYM, spec.keyed_end_pos


def _ramp_envelope(tau: np.ndarray, lo: float, hi: float, ramp_sym: float) -> np.ndarray:
    """1 on [lo, hi), raised-cosine ramps of ramp_sym outside it, 0 beyond."""
    env = ((tau >= lo) & (tau < hi)).astype(np.float64)
    if ramp_sym > 0:
        up = (tau >= lo - ramp_sym) & (tau < lo)
        env[up] = 0.5 * (1 - np.cos(np.pi * (tau[up] - (lo - ramp_sym)) / ramp_sym))
        dn = (tau >= hi) & (tau < hi + ramp_sym)
        env[dn] = 0.5 * (1 + np.cos(np.pi * (tau[dn] - hi) / ramp_sym))
    return env


def _turns_to_phasor(turns: np.ndarray) -> np.ndarray:
    """exp(j 2 pi turns), with the turns reduced to [0, 1) first."""
    ang = 2 * np.pi * (turns - np.floor(turns))
    out = np.empty(ang.shape, dtype=np.complex128)
    out.real = np.cos(ang)
    out.imag = np.sin(ang)
    return out


def baseband(sym, spec: FrameSpec, fs: int, start_s: float, n: int, *, keying=None,
             ook: bool = False, lead_in_s: float = 0.0, ramp_s: float = 0.05,
             time_scale: float = 1.0, chunk: int = 1 << 18) -> np.ndarray:
    """complex128[n]: the CE signal at t_i = t0 + (start_s + i/fs) * time_scale.

    a * exp(j(phi + 2 pi theta_cw)), zero where unkeyed, with a raised-cosine
    amplitude ramp of `ramp_s` just outside the keyed span (where phi = 0)
    against key clicks. time_scale = 1 with start_s on the sample grid of a
    rate dividing 16000 takes the exact table path; anything else (a clock
    offset) evaluates the pulse in closed form. Computed in chunks.
    """
    sym = _check_sym(sym, spec)
    n = int(n)
    n0 = round(start_s * fs)
    exact = (time_scale == 1.0 and int(fs) == fs and GRID_FS % int(fs) == 0
             and abs(start_s * fs - n0) < 1e-9)
    lo, hi = keyed_span_pos(spec, lead_in_s)
    ramp = ramp_s / T_SYM
    k_all = window_keying(keying, spec)
    out = np.zeros(n, dtype=np.complex128)
    for i0 in range(0, n, chunk):
        m = min(chunk, n - i0)
        if exact:
            # The 16 kHz grid index is exact; tau comes from it.
            tau = (np.arange(n0 + i0, n0 + i0 + m, dtype=np.int64) * _grid_step(fs)) / T_NUM
        else:
            tau = (start_s + np.arange(i0, i0 + m, dtype=np.float64) / fs) * time_scale / T_SYM
        if tau[-1] < lo - ramp or tau[0] >= hi + ramp:
            continue                             # unkeyed: stays zero
        if exact:
            phi = phase_grid(int(fs), n0 + i0, m, sym, spec)
        else:
            phi, _ = phase_at(tau, sym, spec)
        env = _ramp_envelope(tau, lo, hi, ramp)
        theta, a = cw_phase_turns(tau, spec, k_all, ook)
        out[i0:i0 + m] = (env * a) * _turns_to_phasor(phi / (2 * np.pi) + theta)
    return out


def to_audio(z, fs: int, carrier_hz_num: int, carrier_hz_den: int = 1,
             amplitude: float = 0.5, n0: int = 0) -> np.ndarray:
    """float64 8 kHz audio: amplitude * Re{z e^{j 2 pi f n / 8000}}.

    That is sqrt(2) Re{.} * amplitude/sqrt(2): a unit-power baseband comes
    out as a sine of peak `amplitude`. z is resampled to 8 kHz first if fs is
    lower. The carrier phase is exact integer turns,
    (n * num mod (den * 8000)) / (den * 8000), with n counted from `n0`
    (pass the running sample count when writing a slot in chunks).
    """
    z = np.asarray(z, dtype=np.complex128)
    fs = int(fs)
    if fs != FS:
        g = math.gcd(FS, fs)
        z = resample_poly(z, FS // g, fs // g)
    num, den = int(carrier_hz_num), int(carrier_hz_den)
    if den <= 0:
        raise ValueError("carrier_hz_den must be positive")
    period = den * FS
    nn = (np.arange(len(z), dtype=np.int64) + int(n0)) % period
    turns = ((nn * (num % period)) % period).astype(np.float64) / period
    return amplitude * np.real(z * np.exp(2j * np.pi * turns))


def carrier_fraction(carrier_hz) -> tuple[int, int]:
    """(num, den) of a carrier given as int, float, str or Fraction."""
    f = Fraction(carrier_hz).limit_denominator(1 << 20) if not isinstance(
        carrier_hz, Fraction) else carrier_hz
    return f.numerator, f.denominator


# --- the mean waveform (receiver template) ------------------------------------------------

def mean_template(tau_pos, mu, nu, spec: FrameSpec, keying=None, ook: bool = False,
                  keyed_from_pos=None) -> np.ndarray:
    """E[s(t)] = a * exp(j(phi_mu + 2 pi theta_cw)) * exp(-v/2) at tau_pos.

    mu and nu are per stream symbol: known symbols (mu = +-1, nu = 0),
    unknown data (mu = 0, nu = 1), EM soft data (mu = x_hat, nu = v).
    The keyed span starts at `keyed_from_pos` (default t0 - 8T; pass the
    lead-in's start to include it) and has no ramp.
    """
    tau = np.asarray(tau_pos, dtype=np.float64)
    phi, v = phase_at(tau, mu, spec, var=nu)
    lo = spec.keyed_start_pos if keyed_from_pos is None else float(keyed_from_pos)
    keyed = ((tau >= lo) & (tau < spec.keyed_end_pos)).astype(np.float64)
    theta, a = cw_phase_turns(tau, spec, keying, ook)
    return keyed * a * _turns_to_phasor(phi / (2 * np.pi) + theta) * np.exp(-v / 2)


def _mean_template_grid(fs, n0, n, mu, nu, spec, keying=None, ook=False):
    """`mean_template` on the exact grid (start t0 + n0/fs), via `phase_grid`."""
    tau = (np.arange(n0, n0 + n, dtype=np.int64) * _grid_step(fs)) / T_NUM
    phi = phase_grid(fs, n0, n, mu, spec)
    v = var_grid(fs, n0, n, nu, spec)
    keyed = ((tau >= spec.keyed_start_pos) & (tau < spec.keyed_end_pos)).astype(np.float64)
    theta, a = cw_phase_turns(tau, spec, keying, ook)
    return keyed * a * _turns_to_phasor(phi / (2 * np.pi) + theta) * np.exp(-v / 2)


# --- matched filter and loopback --------------------------------------------------------------

def matched_filter(z, fs: int, n0: int, pos, chunk: int = 2048) -> np.ndarray:
    """m_p = (1/(fs T)) sum_n z[n] p((t_n - pos_p T)/T), t_n = t0 + (n0 + n)/fs.

    The exact symbol matched filter on a grid dividing 16000 (no timing
    error: the genie's). A constant 1 gives G0. Samples outside z count 0.
    """
    z = np.asarray(z)
    L = _grid_step(fs)
    pos = np.asarray(pos, dtype=np.int64)
    width = 2 * HALF // L + 2
    out = np.zeros(len(pos), dtype=np.result_type(z.dtype, np.float64))
    for lo in range(0, len(pos), chunk):
        p = pos[lo:lo + chunk, None]
        first = -((HALF - T_NUM * p) // L)          # ceil((485 p - 3880)/L)
        nn = first + np.arange(width)[None, :]       # absolute sample numbers
        m = L * nn - T_NUM * p
        idx = nn - n0
        ok = (np.abs(m) <= HALF) & (idx >= 0) & (idx < len(z))
        taps = np.where(ok, PULSE_TABLE[np.clip(m + HALF, 0, 2 * HALF)], 0.0)
        out[lo:lo + chunk] = np.sum(z[np.clip(idx, 0, len(z) - 1)] * taps, axis=1)
    return out * (L / T_NUM)


def symbol_classes(spec: FrameSpec) -> dict[str, np.ndarray]:
    """Stream indices of each symbol class: known, header bits, data."""
    lay = layout(spec)
    return {"known": np.asarray(lay.known_idx), "header": np.asarray(lay.hdr_bits),
            "data": np.asarray(lay.data)}


def _loopback_span(spec: FrameSpec, fs: int) -> tuple[int, int]:
    """(n0, n): grid samples covering the keyed frame plus one pulse span each side."""
    L = _grid_step(fs)
    lo, hi = keyed_span_pos(spec)
    n0 = math.floor((lo - SPAN - 1) * T_NUM / L)
    n1 = math.ceil((hi + SPAN + 1) * T_NUM / L)
    return n0, n1 - n0


def loopback_mf(sym, spec: FrameSpec, fs: int = CH_FS, keying=None,
                ook: bool = False) -> np.ndarray:
    """complex128[n_sym]: genie matched-filter output at every stream symbol.

    Noiseless, unit gain, true timing. Im(m)/K estimates the data symbols
    with the per-pass phase-modulation distortion as the only error.
    """
    sym = _check_sym(sym, spec)
    n0, n = _loopback_span(spec, fs)
    s = baseband(sym, spec, fs, n0 / fs, n, keying=keying, ook=ook, ramp_s=0.0)
    return matched_filter(s, fs, n0, layout(spec).pos)


def loopback_stats(sym, spec: FrameSpec, fs: int = CH_FS) -> dict:
    """Noiseless genie loopback of one frame (design 2.4, 6.5, 6.6).

    Synthesises the frame at `fs`, matched-filters it at every stream
    symbol with true timing and unit gain, and compares the output with
    the matched-filtered mean template the receiver uses in round A
    (known symbols mu = +-1, nu = 0; header and data mu = 0, nu = 1).

    Returns:
      gain        Im(m) regressed on the data symbols (about K = 0.581)
      dist_db     per-pass distortion of the data, 10 log10(gain^2 E x^2 / E e^2)
      d_pass      the same as a variance in latent units, mean((Im m/gain - x)^2)/E x^2
      mse_k_db    10 log10(E x^2 / E (Im m / K - x)^2): with the closed-form K
      known_gain  Im(m) regressed on the known +-1 symbols
      classes     per class: n, res_var = E|m - c|^2, self_pred = E(1 - MF{e^-v}/G0),
                  kappa = res_var / self_pred (nan for the known class)
      kappa_self  E|m - c|^2 / E(1 - MF{e^-v}/G0) over header and data together
      carrier     |mean(s)|^2 over the data section (the residual carrier share)
      G0
    """
    sym = _check_sym(sym, spec)
    lay = layout(spec)
    L = _grid_step(fs)
    n0, n = _loopback_span(spec, fs)
    s = baseband(sym, spec, fs, n0 / fs, n, ramp_s=0.0)

    mu = np.zeros(spec.n_sym)
    nu = np.ones(spec.n_sym)
    mu[lay.known_idx] = lay.known_val
    nu[lay.known_idx] = 0.0
    c_wave = _mean_template_grid(fs, n0, n, mu, nu, spec)
    self_wave = np.exp(-var_grid(fs, n0, n, nu, spec))

    pos = np.asarray(lay.pos)
    m = matched_filter(s, fs, n0, pos)
    c = matched_filter(c_wave, fs, n0, pos)
    e_self = 1.0 - matched_filter(self_wave, fs, n0, pos).real / G0
    r2 = np.abs(m - c) ** 2

    cls = symbol_classes(spec)
    classes = {}
    for name, idx in cls.items():
        if len(idx) == 0:
            continue
        rv, sp = float(np.mean(r2[idx])), float(np.mean(e_self[idx]))
        classes[name] = dict(n=int(len(idx)), res_var=rv, self_pred=sp,
                             kappa=(rv / sp if name != "known" and sp > 0 else float("nan")))
    unk = np.concatenate([cls["header"], cls["data"]])
    kappa_self = float(np.mean(r2[unk]) / np.mean(e_self[unk]))

    x = sym[cls["data"]]
    y = m[cls["data"]].imag
    gain = float(np.dot(y, x) / np.dot(x, x))
    err = y - gain * x
    ex2 = float(np.mean(x ** 2))
    dist = gain ** 2 * ex2 / float(np.mean(err ** 2))
    mse_k = float(np.mean((y / K_LIN - x) ** 2))
    kv = sym[cls["known"]]
    known_gain = float(np.dot(m[cls["known"]].imag, kv) / np.dot(kv, kv))

    # Residual carrier over the data section (a time average of s), with
    # the callsign windows and their edges left out: they are pure carrier.
    tau = (np.arange(n0, n0 + n, dtype=np.int64) * L) / T_NUM
    sel = (tau > pos[cls["data"][0]] + SPAN) & (tau < pos[cls["data"][-1]] - SPAN)
    for a_w, b_w in lay.win_pos:
        sel &= ~((tau > a_w - SPAN - 1) & (tau < b_w + SPAN + 1))
    seg = s[sel]
    return dict(gain=gain, dist_db=float(10 * np.log10(dist)),
                d_pass=float(np.mean(err ** 2) / gain ** 2 / ex2),
                mse_k_db=float(10 * np.log10(ex2 / mse_k)),
                known_gain=known_gain, classes=classes, kappa_self=kappa_self,
                carrier=float(np.abs(np.mean(seg)) ** 2), G0=G0)

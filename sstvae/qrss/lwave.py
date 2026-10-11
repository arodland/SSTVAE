"""Waveform L: the pulse, the pilot tone, the clipper, the callsign
windows and lead-in, baseband and audio, the matched filter and the
noiseless loopback (spec sections 2, 2.2, 2.7, 2.8;
docs/qrss/waveform-l.md).

A **format module**: this is what goes on the air for waveform L.

In baseband centred on the pilot tone (the carrier a sender names),

    z0(t) = d(t) e^{-j 2 pi f_p (t - t0)} + pilot(t)
    d(t)  = sum_s a_s p((t - t0)/T - pos(s))
    pilot(t) = e(t) * [A_p (1 - W(t)) + A_id W(t) e^{j 2 pi theta(t)}]

with a the assembled complex symbols (`lframe.assemble`, unit power), p
the unit-energy RRC (alpha 0.15, cut at +-8 symbols, renormalised on the
16 kHz grid at T = 970 steps), f_p = 12207/970 Hz the data centre's
distance below the pilot, A_p^2 = 3/17 (15% of the power before
clipping), e(t) the keying envelope (1 on the keyed span with 50 ms
raised-cosine ramps outside it), theta the callsign FSK in turns (key-down
one cycle per unit below the pilot, Hann-smoothed, as CE's) and W(t) the
weight that raises the pilot to the signal's average power A_id: 1 in a
lead-in and in units 8..183 of each callsign window, raised-cosine ramps
over units 0..7 and 184..191, and down from 1 to 0 over the preamble's
first 8 symbols after a lead-in.

The transmitted signal is z0 through SSTVAE's clipper (spec 2.2): three
passes of z <- z + F{(min(A_c/|z|, 1)^k - 1) z}, k = 1, 1.5, 2, with the
fixed threshold A_c = sqrt(20/17), 0 dB above z0's nominal RMS envelope,
and F a filter to the occupied band [-f_p - EDGE - 1 Hz, +1 Hz] with
1 Hz raised-cosine edges. Only the clipper's *correction* is filtered: z0
itself is band-limited apart from the pilot's keying ramps and the
callsign windows, which pass untouched (a window's FSK reaches 4.6 Hz
above the pilot, past the filter). The clipper runs at CLIP_FS = 250 Hz,
over the whole frame at once, and the result is interpolated to 8 kHz;
it is then scaled by 1/A_id, so the lead-in and the windows have unit
amplitude and the data unit average power.

The clipper is a nonlinearity followed by a whole-frame filter, so unlike
CE this baseband is not a sample-by-sample closed form: two senders agree
on the symbols, the pilot and the clipper's definition, and their
signals agree to the clipper's own rounding.
"""

import math
from dataclasses import dataclass
from fractions import Fraction

import numpy as np
from scipy import fft as sfft
from scipy.signal import firwin, upfirdn, welch

from . import ce
from .constants import (
    CH_FS,
    FS,
    L_CLIP_GUARD_HZ,
    L_CLIP_HEADROOM_DB,
    L_CLIP_OVERSHOOT,
    L_CW_WINDOW,
    L_EDGE_HZ,
    L_PILOT_FRAC,
    L_PILOT_HZ,
    L_T_NUM,
    L_T_SYM,
    LEAD_IN_MAX_S,
    SPAN,
    T_DEN,
)
from .lframe import LFrameSpec, layout

GRID_FS = T_DEN                               # 16 kHz grid of the pulse table
HALF = SPAN * L_T_NUM                         # 7760 grid steps = 8 L symbols
CLIP_FS = CH_FS                               # 250 Hz: the clipper's rate (15.2 samples a symbol)

# --- pulse --------------------------------------------------------------------------

_M = np.arange(-HALF, HALF + 1)
PULSE_NORM = float(np.sqrt(np.sum(ce._rrc(_M / L_T_NUM) ** 2) / L_T_NUM))
PULSE_TABLE = np.where(np.abs(_M) <= HALF, ce._rrc(_M / L_T_NUM) / PULSE_NORM, 0.0)
PULSE_TABLE.setflags(write=False)
G0 = float(PULSE_TABLE.sum() / L_T_NUM)

# --- levels ---------------------------------------------------------------------------

_PF = float(L_PILOT_FRAC)
A_PILOT = math.sqrt(_PF / (1.0 - _PF))        # sqrt(3/17): pilot amplitude, symbols at unit power
A_CLIP = math.sqrt(1.0 + A_PILOT ** 2) * 10 ** (L_CLIP_HEADROOM_DB / 20)   # sqrt(20/17)
# Average power of the clipped signal during the data, before the final
# 1/A_id scaling (measured on Gaussian latents, `loopback_stats`): the
# lead-in and the callsign windows send the pilot at this power (spec 2.7,
# 2.8), so a linear rig sees no step in average power.
L_AVG_POWER = 0.63
A_ID = math.sqrt(L_AVG_POWER)

ID_RAMP_UNITS = 8                             # pilot level ramps over units 0..7 and 184..191
KEY_RAMP_S = 0.05                             # keying ramps outside the keyed span


def _grid_step(fs: int) -> int:
    return ce._grid_step(fs)


# --- the data on a sample grid --------------------------------------------------------

def _grid_sum(fs: int, n0: int, n: int, vals: np.ndarray, table: np.ndarray) -> np.ndarray:
    """sum_p vals[p] * table[G*(n0+i) - 970 p + 7760] for i < n (G = 16000/fs), by upfirdn.

    `ce._grid_sum` with L's symbol (970 grid steps); vals may be complex.
    """
    G = _grid_step(fs)
    n0, n = int(n0), int(n)
    if n <= 0:
        return np.zeros(0, dtype=np.result_type(vals, np.float64))
    n_pos = len(vals)
    p_lo = (G * n0) // L_T_NUM - SPAN - 1
    p_hi = -(-(G * (n0 + n - 1)) // L_T_NUM) + SPAN + 2
    a, b = max(p_lo, 0), min(p_hi, n_pos)
    out_dtype = np.result_type(vals, np.float64)
    if a >= b:
        return np.zeros(n, dtype=out_dtype)
    x = np.zeros(p_hi - p_lo, dtype=out_dtype)
    x[a - p_lo:b - p_lo] = vals[a:b]
    z0 = (L_T_NUM * p_lo - HALF) % G
    h = np.concatenate([np.zeros(z0), table])
    first = (G * n0 - L_T_NUM * p_lo + HALF + z0) // G
    y = upfirdn(h, x, up=L_T_NUM, down=G)
    out = y[first:first + n]
    if len(out) < n:
        out = np.concatenate([out, np.zeros(n - len(out), dtype=out_dtype)])
    return out


def at_positions(sym, spec: LFrameSpec) -> np.ndarray:
    """complex128[n_pos]: stream-order symbols at their positions, 0 in windows."""
    sym = np.asarray(sym, dtype=np.complex128)
    if sym.shape != (spec.n_sym,):
        raise ValueError(f"sym has shape {sym.shape}, expected ({spec.n_sym},)")
    out = np.zeros(spec.n_pos, dtype=np.complex128)
    out[layout(spec).pos] = sym
    return out


def data_grid(fs: int, n0: int, n: int, sym, spec: LFrameSpec) -> np.ndarray:
    """d(t) at t = t0 + (n0 + i)/fs, centred on the data (fs dividing 16000)."""
    return _grid_sum(fs, n0, n, at_positions(sym, spec), PULSE_TABLE)


def mix_turns(fs: int, n0: int, n: int, f_hz: Fraction) -> np.ndarray:
    """f (t - t0) in turns, reduced to [0, 1), at t = t0 + (n0 + i)/fs: exact integers."""
    f = Fraction(f_hz)
    period = f.denominator * int(fs)
    nn = (np.arange(n0, n0 + n, dtype=np.int64)) % period
    return ((nn * (f.numerator % period)) % period).astype(np.float64) / period


def tau_grid(fs: int, n0: int, n: int) -> np.ndarray:
    """(t - t0)/T at the samples, from the exact grid index."""
    return (np.arange(n0, n0 + n, dtype=np.int64) * _grid_step(fs)) / L_T_NUM


# --- pilot: keying, callsign windows and lead-in --------------------------------------

def keyed_span_pos(spec: LFrameSpec, lead_in_s: float = 0.0) -> tuple[float, float]:
    """[start, end) of keying in L symbols from t0 (as `ce.keyed_span_pos`)."""
    lead = float(lead_in_s)
    if not 0.0 <= lead <= LEAD_IN_MAX_S:
        raise ValueError(f"lead-in must be 0 to {LEAD_IN_MAX_S} s, not {lead_in_s}")
    return spec.keyed_start_pos - lead / L_T_SYM, spec.keyed_end_pos


def window_keying(keying, spec: LFrameSpec) -> np.ndarray | None:
    """(n_win, 192) uint8 from None, one (192,) mask, or one mask per window."""
    if keying is None:
        return None
    k = np.asarray(keying)
    if k.ndim == 1:
        k = np.broadcast_to(k, (spec.n_win, len(k)))
    if k.shape != (spec.n_win, L_CW_WINDOW) or np.any((k != 0) & (k != 1)):
        raise ValueError(f"keying must be 0/1 of shape ({L_CW_WINDOW},) or "
                         f"({spec.n_win}, {L_CW_WINDOW}), not {k.shape}")
    return k.astype(np.uint8)


def _raised(y: np.ndarray) -> np.ndarray:
    """0 for y <= 0, 1 for y >= 1, raised cosine between."""
    y = np.clip(y, 0.0, 1.0)
    return 0.5 * (1.0 - np.cos(np.pi * y))


def cw_turns(tau, spec: LFrameSpec, keying) -> np.ndarray:
    """theta in turns at tau (L symbols from t0): the callsign FSK of the pilot.

    CE's definition (`ce.cw_phase_turns`, FSK) with one unit = one L symbol:
    theta = -sum_u k[u] (R(x - u) - R(x - u - 1)) in window w (x in units
    from its start), and after the window the integer -(key-down units so far).
    """
    tau = np.asarray(tau, dtype=np.float64)
    theta = np.zeros(tau.shape)
    k_all = window_keying(keying, spec)
    if k_all is None:
        return theta
    flat_t, flat_th = tau.reshape(-1), theta.reshape(-1)
    for w in range(spec.n_win):
        k = k_all[w]
        total = int(k.sum())
        if total == 0:
            continue
        p0 = spec.win_start_pos[w] - 0.5
        inside = (flat_t >= p0) & (flat_t < p0 + L_CW_WINDOW)
        flat_th[flat_t >= p0 + L_CW_WINDOW] -= total
        x = flat_t[inside] - p0
        f = np.floor(x).astype(np.int64)
        cum = np.concatenate([[0], np.cumsum(k, dtype=np.int64)])
        th = -cum[np.clip(f - 1, 0, L_CW_WINDOW)].astype(np.float64)
        for j in (-1, 0, 1):
            u = f + j
            ok = (u >= 0) & (u < L_CW_WINDOW)
            ku = np.where(ok, k[np.clip(u, 0, L_CW_WINDOW - 1)], 0)
            y = x - u
            th -= ku * (ce._hann_int(y) - ce._hann_int(y - 1))
        flat_th[inside] += th
    return theta


def id_weight(tau, spec: LFrameSpec, lead_in_s: float = 0.0) -> np.ndarray:
    """W(tau): 1 where the pilot is at average power (lead-in, window units 8..183)."""
    tau = np.asarray(tau, dtype=np.float64)
    W = np.zeros(tau.shape)
    r = float(ID_RAMP_UNITS)
    if lead_in_s > 0:
        start = spec.keyed_start_pos                   # the preamble's first pulse
        W = np.where(tau < start, 1.0, 1.0 - _raised((tau - start) / -start))
    for p in spec.win_start_pos:
        x = tau - (p - 0.5)
        up = _raised(x / r)
        down = 1.0 - _raised((x - (L_CW_WINDOW - r)) / r)
        inside = (x >= 0) & (x < L_CW_WINDOW)
        W = np.where(inside, np.minimum(up, down), W)
    return W


def pilot(tau, spec: LFrameSpec, keying=None, lead_in_s: float = 0.0,
          ramp_s: float = KEY_RAMP_S) -> np.ndarray:
    """complex128: the pilot at tau, before the clipper (module docstring)."""
    tau = np.asarray(tau, dtype=np.float64)
    lo, hi = keyed_span_pos(spec, lead_in_s)
    env = ce._ramp_envelope(tau, lo, hi, ramp_s / L_T_SYM)
    W = id_weight(tau, spec, lead_in_s)
    th = cw_turns(tau, spec, keying)
    return env * (A_PILOT * (1.0 - W) + A_ID * W * ce._turns_to_phasor(th))


# --- composite and clipper ------------------------------------------------------------

def composite(sym, spec: LFrameSpec, fs: int, n0: int, n: int, *, keying=None,
              lead_in_s: float = 0.0, ramp_s: float = KEY_RAMP_S) -> np.ndarray:
    """z0 at t0 + (n0 + i)/fs in pilot-centred baseband: data mixed down by f_p, plus pilot."""
    d = data_grid(fs, n0, n, sym, spec)
    mix = ce._turns_to_phasor(-mix_turns(fs, n0, n, L_PILOT_HZ))
    return d * mix + pilot(tau_grid(fs, n0, n), spec, keying, lead_in_s, ramp_s)


def band_mask(freqs: np.ndarray, guard_hz: float = L_CLIP_GUARD_HZ,
              taper_hz: float = 1.0) -> np.ndarray:
    """The clipper's filter: [-f_p - EDGE - guard, +guard] in pilot-centred Hz,
    raised-cosine edges `taper_hz` wide centred on the band edges."""
    lo = -float(L_PILOT_HZ + L_EDGE_HZ) - guard_hz
    hi = guard_hz
    f = np.asarray(freqs, dtype=np.float64)
    t = taper_hz
    rise = _raised((f - (lo - t / 2)) / t)
    fall = 1.0 - _raised((f - (hi - t / 2)) / t)
    return np.minimum(rise, fall)


def clip(z, fs: int = CLIP_FS, threshold: float = A_CLIP,
         overshoot=L_CLIP_OVERSHOOT, guard_hz: float = L_CLIP_GUARD_HZ) -> np.ndarray:
    """SSTVAE's clipper on z (pilot-centred, rate fs), each pass's correction filtered.

    z should be zero for a few seconds at both ends: the filter is applied
    over the whole array at once (an FFT padded against wrap-around).
    """
    z = np.array(z, dtype=np.complex128)
    nfft = sfft.next_fast_len(len(z) + 4 * int(fs))
    mask = band_mask(sfft.fftfreq(nfft, 1.0 / fs), guard_hz)
    for k in overshoot:
        mag = np.abs(z)
        over = mag > threshold
        if not over.any():
            continue
        scale = np.ones(len(z))
        scale[over] = (threshold / mag[over]) ** float(k)
        corr = (scale - 1.0) * z
        z += sfft.ifft(sfft.fft(corr, nfft) * mask)[:len(z)]
    return z


@dataclass(frozen=True)
class Baseband:
    """The clipped frame at `fs`, pilot-centred, scaled to unit average power.

    z[i] is at t0 + (n0 + i)/fs and is zero outside the keyed span and its ramps.
    """
    z: np.ndarray
    n0: int
    fs: int


def frame_baseband(sym, spec: LFrameSpec, *, keying=None, lead_in_s: float = 0.0,
                   fs: int = CLIP_FS, ramp_s: float = KEY_RAMP_S,
                   margin_s: float = 4.0) -> Baseband:
    """The whole frame through the clipper (module docstring), at the clipper's rate."""
    lo, hi = keyed_span_pos(spec, lead_in_s)
    t_lo, t_hi = lo * L_T_SYM - ramp_s, hi * L_T_SYM + ramp_s
    n0 = math.floor((t_lo - margin_s) * fs)
    n1 = math.ceil((t_hi + margin_s) * fs)
    z0 = composite(sym, spec, fs, n0, n1 - n0, keying=keying, lead_in_s=lead_in_s,
                   ramp_s=ramp_s)
    z = clip(z0, fs)
    t = (np.arange(n0, n1, dtype=np.float64)) / fs
    z[(t < t_lo) | (t >= t_hi)] = 0.0
    z /= A_ID
    z.setflags(write=False)
    return Baseband(z=z, n0=n0, fs=fs)


# --- to 8 kHz -------------------------------------------------------------------------------

INTERP_HALF = 240                             # taps either side at 8 kHz (30 ms)


def _interp_taps(up: int) -> np.ndarray:
    """Lowpass for x`up` interpolation: flat to ~60 Hz, >= 110 dB down from 190 Hz."""
    return firwin(2 * INTERP_HALF + 1, 125.0, fs=FS, window=("kaiser", 12.0)) * up


def resample(bb: Baseband, fs_out: int, i0: int, m: int) -> np.ndarray:
    """complex128[m]: `bb` at t0 + (i0 + i)/fs_out, by band-limited interpolation."""
    up = int(fs_out) // bb.fs
    if up * bb.fs != fs_out:
        raise ValueError(f"fs_out must be a multiple of {bb.fs}")
    if up == 1:
        out = np.zeros(m, dtype=np.complex128)
        a, b = max(i0, bb.n0), min(i0 + m, bb.n0 + len(bb.z))
        if a < b:
            out[a - i0:b - i0] = bb.z[a - bb.n0:b - bb.n0]
        return out
    h = _interp_taps(up)
    D = INTERP_HALF
    n_lo = -(-(i0 - D) // up)
    n_hi = (i0 + m - 1 + D) // up + 1
    a, b = max(n_lo, bb.n0), min(n_hi, bb.n0 + len(bb.z))
    out = np.zeros(m, dtype=np.complex128)
    if a >= b:
        return out
    x = bb.z[a - bb.n0:b - bb.n0]
    u = upfirdn(h, x, up=up)                    # u[k] at output sample up*a + k - D
    k0 = i0 - (up * a - D)
    lo_out = max(0, -k0)
    seg = u[max(k0, 0):max(k0, 0) + m - lo_out]
    out[lo_out:lo_out + len(seg)] = seg
    return out


# --- matched filter and loopback ---------------------------------------------------------

def matched_filter(z, fs: int, n0: int, pos, chunk: int = 2048) -> np.ndarray:
    """m_p = (1/(fs T)) sum_n z[n] p((t_n - pos_p T)/T), z centred on the data.

    `ce.matched_filter` with L's pulse. A constant 1 gives G0.
    """
    z = np.asarray(z)
    G = _grid_step(fs)
    pos = np.asarray(pos, dtype=np.int64)
    width = 2 * HALF // G + 2
    out = np.zeros(len(pos), dtype=np.result_type(z.dtype, np.float64))
    for lo in range(0, len(pos), chunk):
        p = pos[lo:lo + chunk, None]
        first = -((HALF - L_T_NUM * p) // G)
        nn = first + np.arange(width)[None, :]
        mm = G * nn - L_T_NUM * p
        idx = nn - n0
        ok = (np.abs(mm) <= HALF) & (idx >= 0) & (idx < len(z))
        taps = np.where(ok, PULSE_TABLE[np.clip(mm + HALF, 0, 2 * HALF)], 0.0)
        out[lo:lo + chunk] = np.sum(z[np.clip(idx, 0, len(z) - 1)] * taps, axis=1)
    return out * (G / L_T_NUM)


def to_data_centre(bb: Baseband) -> np.ndarray:
    """bb.z mixed up by f_p, so the data sit at 0 Hz (the pilot at +f_p)."""
    return bb.z * ce._turns_to_phasor(mix_turns(bb.fs, bb.n0, len(bb.z), L_PILOT_HZ))


def papr_db(z, q: float = 0.9999) -> float:
    """Envelope PAPR, the q quantile of |z|^2 over its mean (SSTVAE's definition)."""
    p = np.abs(np.asarray(z)) ** 2
    return float(10 * np.log10(np.quantile(p, q) / np.mean(p)))


def occupied_hz(z, fs: int, frac: float = 0.99, nperseg: int = 4096):
    """(lo, hi) in Hz: the band holding `frac` of z's power, (1 - frac)/2 cut each side."""
    f, P = welch(z, fs=fs, nperseg=min(nperseg, len(z)), return_onesided=False,
                 detrend=False, scaling="spectrum")
    order = np.argsort(f)
    f, P = f[order], P[order]
    c = np.cumsum(P) / np.sum(P)
    cut = (1.0 - frac) / 2
    return float(np.interp(cut, c, f)), float(np.interp(1.0 - cut, c, f))


def _data_span_mask(spec: LFrameSpec, tau: np.ndarray) -> np.ndarray:
    """Samples inside the data section, away from the windows and their edges."""
    lay = layout(spec)
    first = lay.pos[spec.n_pre + spec.n_hdr]
    sel = (tau > first + SPAN) & (tau < lay.pos[-1] - SPAN)
    for a, b in lay.win_pos:
        sel &= ~((tau > a - SPAN - 1) & (tau < b + SPAN + 1))
    return sel


def loopback_stats(sym, spec: LFrameSpec, *, keying=None, fs: int = CLIP_FS) -> dict:
    """Noiseless genie loopback of one L frame: what the clipper did.

    Returns, with the matched filter at true timing on the clipped signal:
      gain_data, gain_ref, gain_hdr  complex LS gains of the symbol classes
      data_vs_ref     |gain_data| / |gain_ref| (spec 2.2: about 0.88)
      sinr_db         data SINR, 10 log10(|g|^2 E|a|^2 / E|m - g a|^2): per latent too
      sinr_ref_db     the same for the references
      pilot_frac      the pilot tone's share of the power during the data (13.8%)
      papr_db         envelope PAPR over the data section, 99.99% quantile (3.2 dB)
      avg_power       mean |z|^2 over the data section before the 1/A_id scaling
      width_hz        (lo, hi) of 99% of the power during the data, pilot-centred
      split           power shares: data, pilot + references, distortion
    """
    lay = layout(spec)
    bb = frame_baseband(sym, spec, keying=keying, fs=fs)
    zd = to_data_centre(bb)
    m = matched_filter(zd, fs, bb.n0, lay.pos)
    sym = np.asarray(sym, dtype=np.complex128)

    def fit(idx):
        a, y = sym[idx], m[idx]
        g = np.vdot(a, y) / np.vdot(a, a)
        e = y - g * a
        sinr = abs(g) ** 2 * np.mean(abs(a) ** 2) / np.mean(abs(e) ** 2)
        return g, float(10 * np.log10(sinr))

    g_d, s_d = fit(lay.data)
    g_r, s_r = fit(lay.ref)
    g_h = fit(lay.hdr_sym)[0] if len(lay.hdr_sym) else complex("nan")
    tau = tau_grid(fs, bb.n0, len(bb.z))
    sel = _data_span_mask(spec, tau)
    seg = bb.z[sel]
    p_tot = float(np.mean(abs(seg) ** 2))
    p_pilot = float(abs(np.mean(seg)) ** 2)
    data_pow = float(np.mean(abs(sym[lay.data]) ** 2))
    n_ref = len(lay.data_ref)
    n_dat = len(lay.data)
    p_data = abs(g_d) ** 2 * data_pow * n_dat / (n_dat + n_ref)
    p_ref = abs(g_r) ** 2 * n_ref / (n_dat + n_ref)
    return dict(gain_data=complex(g_d), gain_ref=complex(g_r), gain_hdr=complex(g_h),
                data_vs_ref=float(abs(g_d) / abs(g_r)), sinr_db=s_d, sinr_ref_db=s_r,
                pilot_frac=p_pilot / p_tot, papr_db=papr_db(seg),
                avg_power=p_tot * A_ID ** 2, width_hz=occupied_hz(seg, fs),
                split=dict(data=p_data / p_tot, pilot_ref=(p_pilot + p_ref) / p_tot,
                           distortion=1.0 - (p_data + p_pilot + p_ref) / p_tot))

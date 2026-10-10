"""Carrier, timing and complex-gain tracking (design 6.3 V, 6.4, 6.5; WP6).

Not a format module: nothing here goes on the air.

The spec's joint receiver state (frequency, phase, gain, timing) is
estimated in factorised form (decision D16), so every smoother is
linear-Gaussian and nothing can cycle-slip:

1. **Frequency path.** A3's Viterbi path, or A2's line, fitted by a
   smoothing spline with knots every 60 s (`freq_path_spline`); its
   phase is integrated in float64 turns at the CH rate, wrapped, and
   taken out of the channel: z_d = CH * exp(-2 pi j theta_A).
2. **Timing** is an affine model with an optional quadratic, fitted
   over the whole pass (`fit_timing`, decision D17): position p is
   received at CH index t0_index + tau0 + p*T*fs*(1 + ppm 1e-6) +
   gamma*(p/n_pos)^2. A coarse grid over (tau0, ppm) correlates the
   known symbols with Im(z_d * conj(LP(z_d))); once the gain tracker has
   run, `refine_timing` maximises the likelihood of the known symbols
   given u_hat (Newton with numeric derivatives), and its curvature is
   the reported covariance.
3. **Symbol matched filter** at every position of the tracker grid
   (`symbol_mf`): lead-in virtual positions, frame symbols and window
   positions, evaluated directly with the closed-form pulse (a fine
   polyphase table, exact far below the noise) over the ~122 CH samples
   each position spans.
4. **Measurement model.** m_p = u_p c_p + e_p with c_p the matched
   filter of the mean waveform E[s(t)] (`ce` re-modulated exactly as
   sent: known symbols mu = +-1, unknown ones mu = 0, nu = 1, EM soft
   data mu = x_hat, nu = v; callsign windows and the lead-in included),
   and Var(e_p) = N_p + R_self,p, the self-noise of unknown modulation
   R_self = KAPPA * Pu_p * MF{a^2 (1 - e^-v)}/G0. Normalising,
   mu_p = m_p/(c_p psi_p) observes u_p directly with variance R_p.
5. **Complex-gain Kalman filter and RTS smoother** on x = [u, u'],
   constant velocity per position, Q = q Pu [[1/3, 1/2], [1/2, 1]],
   with q chosen by maximum innovation likelihood over a 13-point grid
   of effective Doppler spreads (`kalman_rts`). The forward pass alone
   is the causal live filter.
6. **Re-centring.** The frequency u_hat still rotates at, in 30 s
   windows, is added to the path and steps 3-5 repeat.

`verify` is the single acceptance gate V: the tracker runs with the
known symbols withheld, so u_hat never saw them, and
Z_ref = sum_k c''_k Im(m_k conj(u_hat_k)) / sd is N(0, 1) on noise at a
given timing; a candidate is accepted at Z_ref > 6.

Calibrated constants (measured with `ce.loopback_stats`, pinned here):
KAPPA_SELF scales the self-noise prediction of unknown symbols' carrier
measurements and KAPPA_SELF_KNOWN that of known ones (the prediction is
pessimistic there, the symbol's own term being deterministic).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from fractions import Fraction

import numpy as np
from scipy import interpolate, signal

from sstvae.modem import dsp

from . import ce, frontend
from .constants import (
    CH_FS,
    CW_UNIT,
    FE_CENTER_HZ,
    SPAN,
    T_SYM,
    Z_ACCEPT,
)
from .frame import FrameSpec, layout, spread_symbols
from .types import Detection, FreqPath, Timing, TrackReport

# --- calibrated constants ---------------------------------------------------------------

# E|m - c|^2 / E[MF{1 - e^-v}/G0] over header and data symbols, noiseless genie
# loopback (`ce.loopback_stats`), Gaussian and tanh latents, MEDIUM and FULL,
# 16 frames: 0.935 to 0.948.
KAPPA_SELF = 0.94
# The same ratio at known (preamble, reference, spare) positions: 0.13 to 0.15.
KAPPA_SELF_KNOWN = 0.14

# --- tracker settings ---------------------------------------------------------------------

Q_SPREADS_HZ = tuple(np.round(np.logspace(np.log10(0.01), np.log10(3.0), 13), 5))
LEAD_KEEP_S = 8.0                  # use only the last 8 s of a detected lead-in (spec 2.8)
RECENTER_WIN_S = 30.0
RECENTER_MIN_HZ = 0.02             # a third outer iteration only above this
KNOTS_S = 10.0
CARRIER_LP_HZ = (0.03, 0.3, 1.5, 3.0)   # coarse timing: carrier estimates tried (0.03: weak, steady)
PU_SMOOTH_S = 1.0                  # local |u|^2 smoothing for the self-noise term
C_MIN = 0.1                        # |c_p psi_p| below this: no measurement
PSI_MIN = 0.5                      # blanked share: erase below this (design 6.2)
NOISE_WIN_S = 10.0
_SPS = T_SYM * CH_FS               # CH samples per symbol (7.578)
_MF_RES = 512                      # polyphase resolution of the MF table, per CH sample
_MF_HALF = int(math.ceil(SPAN * _SPS)) + 1


def q_of_spread(spread_hz: float) -> float:
    """Process-noise intensity per position for a 2-sigma Doppler spread.

    A Gaussian-Doppler gain of rms frequency sigma = spread/2 has
    E|u'|^2 = (2 pi sigma T)^2 Pu per position and loses that velocity in
    about 1/(2 pi sigma T) positions, so the white acceleration that
    reproduces it has intensity (2 pi sigma T)^3 Pu.
    """
    return (2 * math.pi * 0.5 * float(spread_hz) * T_SYM) ** 3


def spread_of_q(q: float) -> float:
    """Inverse of `q_of_spread`: the effective 2-sigma spread (Hz) of q."""
    return 2.0 * q ** (1.0 / 3.0) / (2 * math.pi * T_SYM)


Q_GRID = tuple(q_of_spread(b) for b in Q_SPREADS_HZ)
R_FLOOR = 1e-3                   # model-error floor on R, relative to |u|^2 (-30 dB)
_DEBUG_SEG = False
PATH_SEG_S = 20.0                # refine_path: segment length
PATH_FINE_S = 5.0                # refine_path: second-pass segment length
PATH_FINE_SHARE = 0.7            # ...used when this share of its segments find the line
PATH_FINE_RATIO = 40.0           # ...at a median line-to-floor ratio of 16 dB (-25 dB gives 13)
PATH_SPAN_HZ = 2.0               # refine_path: search either side of the last segment
SEG_TIMING_S = 20.0              # segment length of fit_timing's local offsets
V_TIMING_TRIES = 8               # gate V: coarse timing peaks tracked for a blind candidate


# --- the narrow channel -----------------------------------------------------------------------

@dataclass
class ChanCapture:
    """One signal's 250 Hz channel: CH = FE' mixed down by f_mix, decimated.

    t0_index: CH index of the nominal t0 on the receiver's clock.
    f_mix: the channeliser's exact mix, Hz inside FE (audio Hz - 1500).
    keep: the blanker's keep indicator through the same decimation (or None).
    exclude_hz: neighbour frequencies (audio Hz) left out of the noise floor.
    """
    ch: np.ndarray
    fs: int
    t0_index: float
    f_mix: Fraction
    keep: np.ndarray | None = None
    exclude_hz: tuple = ()
    _psd: tuple | None = field(default=None, repr=False)

    @property
    def mix_audio_hz(self) -> float:
        return FE_CENTER_HZ + float(self.f_mix)

    def t_s(self, n) -> np.ndarray:
        """Receiver time (s after nominal t0) of CH index n."""
        return (np.asarray(n, dtype=np.float64) - self.t0_index) / self.fs

    def noise_psd_at(self, t_s) -> np.ndarray:
        """Noise PSD (CH power per Hz) at receiver times t_s, from 10 s windows
        85-120 Hz off the mix frequency (`frontend.noise_psd`), interpolated."""
        if self._psd is None:
            ex = [h - self.mix_audio_hz for h in self.exclude_hz]
            have = np.abs(self.ch) > 0
            w = max(16, int(round(NOISE_WIN_S * self.fs)))
            psd = frontend.noise_psd(self.ch, self.fs, exclude_hz=ex, win_s=NOISE_WIN_S)
            nwin = len(psd)
            cen = np.array([min(i * w, max(len(self.ch) - w, 0)) + w / 2 for i in range(nwin)])
            # windows that are mostly zero padding read low: take them from the rest
            frac = np.array([have[int(max(c - w / 2, 0)):int(c + w / 2)].mean()
                             if len(have) else 0.0 for c in cen])
            good = (frac > 0.9) & (psd > 0)
            if good.any():
                psd = np.where(good, psd, np.median(psd[good]))
            else:
                psd = np.full(nwin, max(float(np.median(psd)), 1e-30))
            self._psd = (self.t_s(cen), psd.astype(np.float64))
        tc, psd = self._psd
        return np.interp(np.asarray(t_s, dtype=np.float64), tc, psd)

    def noise_floor(self) -> float:
        """Median noise PSD over the capture."""
        self.noise_psd_at(0.0)
        return float(np.median(self._psd[1]))


def make_chan(fe, keep, t0_index_fe: float, fs_fe: int, f_audio_hz: float,
              t_lo_s: float, t_hi_s: float, exclude_hz=()) -> ChanCapture:
    """Channelise FE' (and the blanker's keep) about f_audio_hz over
    receiver times [t_lo_s, t_hi_s] after nominal t0; zero-padded where the
    capture does not reach. The CH grid is FE index 16k (aligned with the stream)."""
    fe = np.asarray(fe)
    fs_fe = int(fs_fe)
    decim = fs_fe // CH_FS
    i_a = int(math.floor(t0_index_fe + t_lo_s * fs_fe))
    i_a -= i_a % decim
    i_b = int(math.ceil(t0_index_fe + t_hi_s * fs_fe))
    i_b += (-(i_b - i_a)) % decim
    seg = np.zeros(i_b - i_a, dtype=np.complex64)
    a, b = max(i_a, 0), min(i_b, len(fe))
    if b > a:
        seg[a - i_a:b - i_a] = fe[a:b]
    ch, f_mix = frontend.channelise(seg, fs_fe, frontend.fe_hz(f_audio_hz), CH_FS, n0=i_a)
    kch = None
    if keep is not None:
        ks = np.zeros(i_b - i_a, dtype=np.float32)
        if b > a:
            ks[a - i_a:b - i_a] = keep[a:b]
        kch = frontend.channelise_keep(ks, fs_fe, CH_FS)
    return ChanCapture(ch=ch, fs=CH_FS, t0_index=(t0_index_fe - i_a) / decim, f_mix=f_mix,
                       keep=kch, exclude_hz=tuple(exclude_hz))


# --- frequency path ---------------------------------------------------------------------------

class PathFn:
    """A frequency trajectory f(t) (audio Hz at receiver time t, s after t0).

    A weighted least-squares cubic spline with knots every `knots_s`, or a
    weighted line when the path is too short for one; continued linearly
    outside the fitted span.
    """

    def __init__(self, t, f, w=None, knots_s: float = KNOTS_S):
        t = np.asarray(t, dtype=np.float64)
        f = np.asarray(f, dtype=np.float64)
        w = np.ones(len(t)) if w is None else np.asarray(w, dtype=np.float64)
        order = np.argsort(t)
        t, f, w = t[order], f[order], np.maximum(w[order], 0.0) + 1e-9
        self.lo, self.hi = float(t[0]), float(t[-1])
        self._spl = None
        if len(t) >= 2:
            self._line = np.polyfit(t, f, 1, w=np.sqrt(w))
        else:
            self._line = np.array([0.0, float(f[0])])
        n_int = int((self.hi - self.lo) // knots_s)
        if len(t) >= 8 and n_int >= 1:
            inner = self.lo + knots_s * np.arange(1, n_int + 1)
            inner = inner[inner < self.hi - 0.25 * knots_s]
            knots = np.concatenate([[self.lo] * 4, inner, [self.hi] * 4])
            try:
                self._spl = interpolate.make_lsq_spline(t, f, knots, k=3, w=np.sqrt(w))
            except (ValueError, np.linalg.LinAlgError):
                self._spl = None
            if self._spl is not None and not np.all(np.isfinite(self._spl.c)):
                self._spl = None             # knots without data (Schoenberg-Whitney): a line

    def _inside(self, t):
        if self._spl is None:
            return np.polyval(self._line, t)
        return self._spl(t)

    def __call__(self, t) -> np.ndarray:
        t = np.asarray(t, dtype=np.float64)
        if self._spl is None:
            return np.polyval(self._line, t)
        tc = np.clip(t, self.lo, self.hi)
        out = self._spl(tc)
        lo, hi = t < self.lo, t > self.hi
        if lo.any():
            out = np.where(lo, self._spl(self.lo) + self._spl.derivative()(self.lo) * (t - self.lo),
                           out)
        if hi.any():
            out = np.where(hi, self._spl(self.hi) + self._spl.derivative()(self.hi) * (t - self.hi),
                           out)
        return out


def freq_path_spline(path: FreqPath, knots_s: float = KNOTS_S) -> PathFn:
    """The frequency path f_A(t) of a detection (design 6.5 step 1)."""
    return PathFn(path.t_s, path.f_hz, path.weight, knots_s)


def derotate(chan: ChanCapture, fn) -> np.ndarray:
    """z_d = CH * exp(-2 pi j theta), theta = integral of f(t) - f_mix from t0, in turns."""
    n = len(chan.ch)
    t = chan.t_s(np.arange(n))
    g = (fn(t) - chan.mix_audio_hz) / chan.fs
    cum = np.cumsum(g)
    # theta at t0: interpolate the running sum at the (fractional) t0 index
    k = chan.t0_index
    c0 = float(np.interp(k, np.arange(n), cum)) if n else 0.0
    turns = dsp.wrap_cycles(cum - c0)
    return chan.ch.astype(np.complex128) * np.exp(-2j * np.pi * turns)


# --- timing -----------------------------------------------------------------------------------

def rx_index(timing: Timing, tau, n_pos: int, fs: int = CH_FS) -> np.ndarray:
    """CH index (relative to t0_index) at which sender time tau (symbols) is received."""
    tau = np.asarray(tau, dtype=np.float64)
    sps = T_SYM * fs
    return timing.tau0 + tau * sps * (1 + timing.ppm * 1e-6) + timing.gamma * (tau / n_pos) ** 2


def _mf_table() -> np.ndarray:
    d = np.arange(-_MF_HALF * _MF_RES, _MF_HALF * _MF_RES + 1) / _MF_RES
    return ce.pulse(d / _SPS)


_MF_TAB = _mf_table()


def symbol_mf(z, ch_fs: int, timing: Timing, pos, *, t0_index: float, n_pos: int,
              chunk: int = 4096) -> np.ndarray:
    """m_p = (1/(fs T)) sum_n z(n) p((n - t_p)/(fs T)) at every position p.

    t_p = t0_index + rx_index(timing, p). The pulse comes from a 1/512-sample
    table with linear interpolation (relative error ~1e-7); samples outside
    z count as zero.
    """
    assert int(ch_fs) == CH_FS
    z = np.asarray(z)
    pos = np.asarray(pos, dtype=np.float64)
    tp = t0_index + rx_index(timing, pos, n_pos, ch_fs)
    off = np.arange(-_MF_HALF, _MF_HALF + 1)
    out = np.zeros(len(pos), dtype=np.complex128)
    nz = len(z)
    for lo in range(0, len(pos), chunk):
        t = tp[lo:lo + chunk]
        base = np.floor(t).astype(np.int64)
        nn = base[:, None] + off[None, :]                     # (c, width)
        d = (nn - t[:, None]) * _MF_RES + _MF_HALF * _MF_RES    # table coordinate
        i = np.floor(d).astype(np.int64)
        fr = d - i
        ok = (i >= 0) & (i < len(_MF_TAB) - 1) & (nn >= 0) & (nn < nz)
        ic = np.clip(i, 0, len(_MF_TAB) - 2)
        taps = (_MF_TAB[ic] * (1 - fr) + _MF_TAB[ic + 1] * fr) * ok
        vals = z[np.clip(nn, 0, nz - 1)]
        out[lo:lo + chunk] = np.einsum("ij,ij->i", vals, taps)
    return out / _SPS


def psi_at(keep_ch, timing: Timing, pos, *, t0_index: float, n_pos: int) -> np.ndarray:
    """psi_p = MF{keep}(t_p)/G0, the unblanked share at each position (1 without a blanker)."""
    if keep_ch is None:
        return np.ones(len(np.asarray(pos)))
    return np.clip(symbol_mf(keep_ch, CH_FS, timing, pos, t0_index=t0_index,
                             n_pos=n_pos).real / ce.G0, 0.0, 1.0)


def _lowpass(x, hz: float, fs: int = CH_FS) -> np.ndarray:
    """Zero-phase Hann-windowed-sinc low-pass of cutoff `hz`."""
    n = int(round(4.0 / hz * fs)) | 1
    k = np.arange(n) - n // 2
    h = np.sinc(2 * hz / fs * k) * np.hanning(n + 2)[1:-1]
    h /= h.sum()
    return signal.oaconvolve(x, h, mode="same")


def _mf_taps(fs: int = CH_FS) -> np.ndarray:
    k = np.arange(-_MF_HALF + 1, _MF_HALF)
    return ce.pulse(k / (T_SYM * fs)) / (T_SYM * fs)


@dataclass
class _Known:
    """Known stream symbols as the tracker uses them: positions and values."""
    pos: np.ndarray
    val: np.ndarray


def known_symbols(spec: FrameSpec, hdr_bits=None) -> _Known:
    """Preamble, references and spare (+ the header bits once decoded)."""
    lay = layout(spec)
    idx = np.asarray(lay.known_idx)
    val = np.asarray(lay.known_val, dtype=np.float64)
    if hdr_bits is not None and spec.n_hdr:
        idx = np.concatenate([idx, lay.hdr_bits])
        val = np.concatenate([val, 1.0 - 2.0 * np.asarray(hdr_bits, dtype=np.float64)])
    return _Known(pos=np.asarray(lay.pos)[idx], val=val)


def fit_timing(ch, ch_fs, path, spec: FrameSpec, tmpl_fn=None, tau0_init=None,
               ppm_range: float = 250.0, *, t0_index: float, tau_range=None,
               z_d=None, n_best: int = 1):
    """Coarse timing (design 6.4 steps 1-2): tau0 (CH samples) and ppm.

    ch: the channel (a `ChanCapture`, or its samples with `z_d` given);
    path: a FreqPath or a callable f(t). Im(MF{z_d conj(LP(z_d))}) is
    correlated with the known symbols at every lag and on a grid of clock
    errors (steps of 0.2 T over the frame, +-ppm_range); the carrier
    estimate LP is tried at each of CARRIER_LP_HZ (0.03 Hz for a weak
    signal on a steady path, up to 3 Hz for fast fading) and the best
    segment-refined score kept.
    `tmpl_fn` may give the known symbols (a `_Known`) instead of the
    frame's own. tau0_init (CH samples) narrows the lag search to
    tau_range samples either side; without it the search is +-2.5 s plus
    the largest slip. The peak is refined by parabolas. Timing.z is the
    peak's correlation in units of its noise sd.

    With n_best > 1 a list of up to n_best Timings is returned instead:
    the best peak of each carrier estimate and the runners-up (distinct
    lags more than a symbol apart), strongest first. Gate V tries them in
    turn when the best fails (near the detection floor the coarse
    statistic, a product of two noisy terms, can put the true timing
    second).
    """
    fs = int(ch_fs)
    if z_d is None:
        fn = freq_path_spline(path) if isinstance(path, FreqPath) else path
        z_d = derotate(ch, fn)
    kn = tmpl_fn if isinstance(tmpl_fn, _Known) else known_symbols(spec)
    n_pos = spec.n_pos
    sps = T_SYM * fs
    dur = n_pos * sps
    if tau0_init is None:
        lo, hi = -2.5 * fs - 2, 2.5 * fs + 2
    else:
        r = 3.0 if tau_range is None else float(tau_range)
        lo, hi = tau0_init - r, tau0_init + r
    step = 0.2 / n_pos
    nd = int(math.floor(ppm_range * 1e-6 / step))
    deltas = np.arange(-nd, nd + 1) * step
    # lag j puts position 0 at absolute CH index base + j + frac0, i.e.
    # tau0 = base + j - floor(t0_index)
    base = int(math.floor(t0_index)) + int(math.floor(lo))
    nlag = int(math.ceil(hi)) - int(math.floor(lo)) + 1
    frac0 = t0_index - math.floor(t0_index)
    taps = _mf_taps(fs)
    a = kn.val
    peaks = []
    for lp in CARRIER_LP_HZ:
        uc = _lowpass(z_d, lp, fs)
        x = signal.oaconvolve(z_d * np.conj(uc), taps, mode="same").imag
        sel = slice(max(base, 0), max(min(base + nlag + int(dur) + 1, len(x)), max(base, 0) + 1))
        sig2 = max(float(np.median(x[sel] ** 2)) / 0.4549, 1e-30)
        norm = math.sqrt(sig2 * float(np.sum(a * a)))
        Z = np.full((len(deltas), nlag), -np.inf)
        if nlag <= 64:
            lags = np.arange(nlag)
            for i, d in enumerate(deltas):
                o = kn.pos * sps * (1 + d) + frac0
                idx = base + o[:, None] + lags[None, :]
                i0 = np.floor(idx).astype(np.int64)
                f = idx - i0
                ok = (i0 >= 0) & (i0 + 1 < len(x))
                i0c = np.clip(i0, 0, len(x) - 2)
                v = (x[i0c] * (1 - f) + x[i0c + 1] * f) * ok
                Z[i] = a @ v / norm
        else:
            ng = int(math.ceil(dur * (1 + ppm_range * 1e-6))) + 4
            nfft = 1 << int(math.ceil(math.log2(nlag + ng + 2)))
            xs = np.zeros(nfft)
            s_lo, s_hi = max(base, 0), min(base + nfft, len(x))
            if s_hi > s_lo:
                xs[s_lo - base:s_hi - base] = x[s_lo:s_hi]
            X = np.fft.rfft(xs)
            for i, d in enumerate(deltas):
                o = kn.pos * sps * (1 + d) + frac0
                i0 = np.floor(o).astype(np.int64)
                f = o - i0
                g = np.bincount(i0, a * (1 - f), minlength=nfft)[:nfft] + \
                    np.bincount(i0 + 1, a * f, minlength=nfft)[:nfft]
                c = np.fft.irfft(X * np.conj(np.fft.rfft(g)), nfft)
                Z[i] = c[:nlag] / norm
        Zl = Z.max(axis=0)                     # best clock error per lag
        il = Z.argmax(axis=0)
        for k in range(max(1, int(n_best))):
            j = int(np.argmax(Zl))
            if k and not np.isfinite(Zl[j]):
                break
            i = int(il[j])
            di = dj = 0.0
            if 0 < i < Z.shape[0] - 1:
                di = _parabola(Z[i - 1, j], Z[i, j], Z[i + 1, j])
            if 0 < j < Z.shape[1] - 1:
                dj = _parabola(Z[i, j - 1], Z[i, j], Z[i, j + 1])
            delta = (i - nd + di) * step
            t0abs, delta, score = _segment_timing(x, sig2, kn, base + j + dj + frac0, delta,
                                                  sps, n_pos)
            peaks.append((score, float(Z[i, j]), t0abs, delta))
            Zl[max(j - int(math.ceil(sps)), 0):j + int(math.ceil(sps)) + 1] = -np.inf
    peaks.sort(key=lambda b: -b[0])
    out, seen = [], []
    for _, zmax, t0abs, delta in peaks:
        if any(abs(t0abs - t) < sps for t in seen):
            continue
        seen.append(t0abs)
        out.append(Timing(tau0=float(t0abs - t0_index), ppm=float(delta * 1e6), gamma=0.0,
                          z=zmax, cov=np.diag([0.25, (step * 1e6) ** 2, 0.0])))
    return out[0] if n_best <= 1 else out[:n_best]


def _segment_timing(x, sig2, kn: _Known, t0abs: float, delta: float, sps: float, n_pos: int,
                    seg_s: float = SEG_TIMING_S) -> tuple[float, float]:
    """(t0abs, delta) refined from local timing offsets of segments.

    The whole-frame correlation's clock error is fixed mostly by the
    references (the preamble sits at the start), and on a fading path
    their coherent sum is noisy enough to pick a wrong clock error by a
    symbol or more over the frame. Each `seg_s` stretch of known symbols
    instead gives its own offset, searched +-1 symbol around the line
    fitted to the confident (Z > 4) segments before it, in time order,
    so the fit walks out from the preamble the way a loop would; a final
    weighted line (Z^2) through all of them, outliers dropped, is the
    answer.
    """
    pos = kn.pos.astype(np.float64)
    a = kn.val
    nseg = max(1, int(math.ceil(n_pos * T_SYM / seg_s)))
    edges = np.linspace(pos.min(), pos.max() + 1, nseg + 1)
    lags = np.arange(-sps, sps + 1e-9, 0.1)
    segs = [(pos >= s0) & (pos < s1) for s0, s1 in zip(edges[:-1], edges[1:])]
    segs = [m for m in segs if m.sum() >= 8]

    def local(sel, t0, d):
        o = t0 + pos[sel] * sps * (1 + d)
        idx = o[:, None] + lags[None, :]
        i0 = np.floor(idx).astype(np.int64)
        f = idx - i0
        ok = (i0 >= 0) & (i0 + 1 < len(x))
        i0 = np.clip(i0, 0, len(x) - 2)
        v = (x[i0] * (1 - f) + x[i0 + 1] * f) * ok
        c = a[sel] @ v / math.sqrt(sig2 * float(np.sum(a[sel] ** 2)))
        if not np.all(np.isfinite(c)):
            return None
        k = int(np.argmax(c))
        if c[k] < 4.0:
            return None
        dk = _parabola(c[k - 1], c[k], c[k + 1]) if 0 < k < len(c) - 1 else 0.0
        return lags[k] + dk * 0.1, float(c[k]) ** 2

    def fit(tp, ab, wt, t0_, d_):
        # ab: absolute receive index of each segment's mean position
        tp, ab, wt = np.array(tp), np.array(ab), np.array(wt)
        if len(tp) == 1 or np.ptp(tp) < 300:
            t0 = float(np.average(ab - tp * sps * (1 + delta), weights=wt))
            return t0, delta
        A = np.stack([np.ones_like(tp), tp * sps], axis=1)
        sw = np.sqrt(wt)
        coef, *_ = np.linalg.lstsq(A * sw[:, None], ab * sw, rcond=None)
        res = ab - A @ coef
        keep = np.abs(res) < max(3 * math.sqrt(np.average(res ** 2, weights=wt)), 0.3)
        if keep.sum() >= 2 and not keep.all() and np.ptp(tp[keep]) >= 300:
            coef, *_ = np.linalg.lstsq(A[keep] * sw[keep, None], ab[keep] * sw[keep], rcond=None)
        t0n, dn = float(coef[0]), float(coef[1]) - 1.0
        if not (math.isfinite(t0n) and math.isfinite(dn)) or abs(dn) > 300e-6:
            return t0_, d_
        return t0n, dn

    t0, d = t0abs, delta
    tp, ab, wt = [], [], []
    for sel in segs:
        r = local(sel, t0, d)
        if r is None:
            continue
        pm = float(np.mean(pos[sel]))
        tp.append(pm)
        ab.append(t0 + pm * sps * (1 + d) + r[0])
        wt.append(r[1])
        t0, d = fit(tp, ab, wt, t0, d)
    if not tp:
        return t0abs, delta, 0.0
    # the evidence for the final line: every segment's Z at it (no search)
    score = 0.0
    for sel in segs:
        o = t0 + pos[sel] * sps * (1 + d)
        i0 = np.clip(np.floor(o).astype(np.int64), 0, len(x) - 2)
        f = o - np.floor(o)
        v = x[i0] * (1 - f) + x[i0 + 1] * f
        score += max(float(a[sel] @ v) / math.sqrt(sig2 * float(np.sum(a[sel] ** 2))), 0.0) ** 2
    if _DEBUG_SEG:
        print('SEG', [(round(p_), round(w_ ** .5, 1)) for p_, w_ in zip(tp, wt)], t0, d * 1e6, score)
    return t0, d, score


def _parabola(ym, y0, yp) -> float:
    den = ym - 2 * y0 + yp
    if not np.isfinite(den) or den >= 0:
        return 0.0
    return float(np.clip(0.5 * (ym - yp) / den, -0.5, 0.5))


# --- templates ---------------------------------------------------------------------------------

@dataclass
class Classes:
    """Per stream symbol: template mean and variance, and which are known.

    On a spread-header frame (`FrameSpec.hdr_rho` > 0) `spread` holds the
    header's own stream symbols once the header is known: `track` then
    removes their phase from the channel before anything else, and the
    data templates are the plain ones. Until then the header is a further
    hdr_rho of unknown variance on every data symbol (`spread_rho`).
    """
    mu: np.ndarray
    nu: np.ndarray
    known: np.ndarray                      # bool[n_sym]
    hdr_known: bool = False
    spread: np.ndarray | None = None       # float64[n_sym], known spread header
    spread_rho: float = 0.0                # unknown spread-header variance on the data


def make_classes(spec: FrameSpec, hdr_bits=None, prior=None) -> Classes:
    """Round A (hdr_bits None) or round B classes; `prior` an EmPrior for the data."""
    lay = layout(spec)
    mu = np.zeros(spec.n_sym)
    nu = np.ones(spec.n_sym)
    known = np.zeros(spec.n_sym, dtype=bool)
    mu[lay.known_idx] = lay.known_val
    nu[lay.known_idx] = 0.0
    known[lay.known_idx] = True
    if hdr_bits is not None and spec.n_hdr:
        mu[lay.hdr_bits] = 1.0 - 2.0 * np.asarray(hdr_bits, dtype=np.float64)
        nu[lay.hdr_bits] = 0.0
        known[lay.hdr_bits] = True
    if prior is not None:
        mu[lay.data] = np.asarray(prior.x_hat, dtype=np.float64)
        nu[lay.data] = np.clip(np.asarray(prior.v, dtype=np.float64), 0.0, 1.0)
    spread, rho_u = None, 0.0
    if spec.hdr_rho > 0:
        if hdr_bits is not None:
            spread = spread_symbols(spec, hdr_bits)
        else:
            rho_u = spec.hdr_rho
            nu[lay.data] += rho_u
    return Classes(mu=mu, nu=nu, known=known, hdr_known=hdr_bits is not None,
                   spread=spread, spread_rho=rho_u)


def _tau_of_rx(timing: Timing, rel, n_pos: int) -> np.ndarray:
    """Sender time (symbols) received at CH index `rel` after t0_index: rx_index inverted."""
    rel = np.asarray(rel, dtype=np.float64)
    s = T_SYM * CH_FS * (1 + timing.ppm * 1e-6)
    tau = (rel - timing.tau0) / s
    for _ in range(3):
        tau = (rel - timing.tau0 - timing.gamma * (tau / n_pos) ** 2) / s
    return tau


def remove_spread(z, chan: ChanCapture, spec: FrameSpec, timing: Timing, spread) -> np.ndarray:
    """z * exp(-j phi_h): the known spread header's phase taken off the channel.

    phi_h is the header's part of the sender's phase (`ce.phase_at` of
    its stream symbols), at the sender time each CH sample was sent. CE
    is a pure phase signal, so this leaves exactly the frame without the
    spread header (docs/qrss/spread-header.md).
    """
    if spread is None:
        return z
    z = np.asarray(z, dtype=np.complex128)
    tau = _tau_of_rx(timing, np.arange(len(z)) - chan.t0_index, spec.n_pos)
    near = (tau > -SPAN - 2) & (tau < spec.n_pos + SPAN + 2)
    phi = np.zeros(len(z))
    if near.any():
        phi[near] = ce.phase_at(tau[near], spread, spec)[0]
    return z * np.exp(-1j * phi)


def templates(spec: FrameSpec, grid_pos, mu, nu, keying=None, ook=False,
              keyed_from_pos=None) -> tuple[np.ndarray, np.ndarray]:
    """(c, sf) at integer positions grid_pos: c = MF{E[s]}, sf = MF{a^2 (1 - e^-v)}/G0.

    E[s] = a exp(j(phi_mu + 2 pi theta_cw)) e^{-v/2} on the sender's 250 Hz
    grid (exact table path), keyed from `keyed_from_pos` (default t0 - 8T).
    """
    fs = CH_FS
    L = GRID = ce.GRID_FS // fs
    grid_pos = np.asarray(grid_pos, dtype=np.int64)
    n0 = int(math.floor((grid_pos[0] - SPAN - 2) * 485 / L))
    n1 = int(math.ceil((grid_pos[-1] + SPAN + 2) * 485 / L))
    n = n1 - n0
    tau = (np.arange(n0, n1, dtype=np.int64) * GRID) / 485.0
    phi = ce.phase_grid(fs, n0, n, mu, spec)
    v = ce.var_grid(fs, n0, n, nu, spec)
    lo = spec.keyed_start_pos if keyed_from_pos is None else float(keyed_from_pos)
    keyed = ((tau >= lo) & (tau < spec.keyed_end_pos)).astype(np.float64)
    theta, a = ce.cw_phase_turns(tau, spec, keying, ook)
    turns = dsp.wrap_cycles(phi / (2 * np.pi) + theta)
    amp = keyed * a
    wave = amp * np.exp(2j * np.pi * turns) * np.exp(-v / 2)
    selfw = amp * amp * (1.0 - np.exp(-v))
    c = ce.matched_filter(wave, fs, n0, grid_pos)
    sf = ce.matched_filter(selfw, fs, n0, grid_pos).real / ce.G0
    return c, np.maximum(sf, 0.0)


# --- Kalman filter and RTS smoother -----------------------------------------------------------

def _kf_forward(mu, R, q, Pu, store: bool):
    """Constant-velocity complex-gain Kalman filter; returns loglik (and the pass)."""
    n = len(mu)
    qa, qb, qc = q * Pu / 3.0, q * Pu / 2.0, q * Pu
    meas = np.isfinite(R)
    first = int(np.argmax(meas)) if meas.any() else 0
    x1 = complex(mu[first]) if meas.any() else 0j
    x2 = 0j
    big = 10.0 * Pu + 1e-30
    a, b, c = big, 0.0, big * 1e-2
    ll = 0.0
    mul = mu.tolist()
    Rl = np.where(meas, R, 0.0).tolist()
    ml = meas.tolist()
    if store:
        XA = [0j] * n
        XB = [0j] * n
        PA = [0.0] * n
        PB = [0.0] * n
        PC = [0.0] * n
        FA = [0.0] * n                            # predicted covariances
        FB = [0.0] * n
        FC = [0.0] * n
        FX1 = [0j] * n
        FX2 = [0j] * n
    log = math.log
    for i in range(n):
        if i:
            a = a + 2 * b + c + qa
            b = b + c + qb
            c = c + qc
            x1 = x1 + x2
        if store:
            FA[i] = a
            FB[i] = b
            FC[i] = c
            FX1[i] = x1
            FX2[i] = x2
        if ml[i]:
            r = Rl[i]
            S = a + r
            k1 = a / S
            k2 = b / S
            inn = mul[i] - x1
            ll -= log(S) + (inn.real * inn.real + inn.imag * inn.imag) / S
            x1 += k1 * inn
            x2 += k2 * inn
            c = c - k2 * b
            b = b * (1 - k1)
            a = a * (1 - k1)
        if store:
            XA[i] = x1
            XB[i] = x2
            PA[i] = a
            PB[i] = b
            PC[i] = c
    if not store:
        return ll
    return ll, (XA, XB, PA, PB, PC, FX1, FX2, FA, FB, FC)


def kalman_rts(mu, R, q, Pu, causal: bool = False):
    """(u_hat, P, loglik) of the complex gain (design 6.5 steps 4-5).

    mu: normalised observations of u per grid position (anything where R
    is inf); R: their variance; q: process-noise intensity (`q_of_spread`);
    Pu: the mean signal power. causal=True returns the forward (live)
    filter, else the RTS-smoothed estimate. P is E|u - u_hat|^2.
    """
    mu = np.asarray(mu, dtype=np.complex128)
    R = np.asarray(R, dtype=np.float64)
    ll, (XA, XB, PA, PB, PC, FX1, FX2, FA, FB, FC) = _kf_forward(mu, R, q, Pu, store=True)
    n = len(mu)
    if causal or n == 0:
        return np.array(XA), np.array(PA), ll
    us = [0j] * n
    Ps = [0.0] * n
    s1, s2 = XA[-1], XB[-1]
    sa, sb, sc = PA[-1], PB[-1], PC[-1]
    us[-1] = s1
    Ps[-1] = sa
    for i in range(n - 2, -1, -1):
        # predicted (i+1|i) covariance and the smoother gain G = P F^T Pp^-1
        pa, pb, pc = FA[i + 1], FB[i + 1], FC[i + 1]
        det = pa * pc - pb * pb
        if det <= 0:
            det = 1e-300
        ia, ib, ic = pc / det, -pb / det, pa / det
        fa, fb, fc = PA[i], PB[i], PC[i]
        # P F^T = [[fa + fb, fb], [fb + fc, fc]]
        m11, m12, m21, m22 = fa + fb, fb, fb + fc, fc
        g11 = m11 * ia + m12 * ib
        g12 = m11 * ib + m12 * ic
        g21 = m21 * ia + m22 * ib
        g22 = m21 * ib + m22 * ic
        d1 = s1 - FX1[i + 1]
        d2 = s2 - FX2[i + 1]
        n1 = XA[i] + g11 * d1 + g12 * d2
        n2 = XB[i] + g21 * d1 + g22 * d2
        # Ps = P + G (Ps_next - Pp) G^T
        ea, eb, ec = sa - pa, sb - pb, sc - pc
        t11 = g11 * ea + g12 * eb
        t12 = g11 * eb + g12 * ec
        t21 = g21 * ea + g22 * eb
        t22 = g21 * eb + g22 * ec
        na = fa + t11 * g11 + t12 * g12
        nb = fb + t11 * g21 + t12 * g22
        nc = fc + t21 * g21 + t22 * g22
        s1, s2, sa, sb, sc = n1, n2, na, nb, nc
        us[i] = s1
        Ps[i] = max(sa, 0.0)
    return np.array(us), np.array(Ps), ll


def select_q(mu, R, Pu, grid=Q_GRID) -> tuple[float, float]:
    """(q, loglik): the process noise of greatest innovation likelihood."""
    best = None
    for q in grid:
        ll = _kf_forward(mu, R, q, Pu, store=False)
        if best is None or ll > best[1]:
            best = (q, ll)
    return best


# --- the tracker --------------------------------------------------------------------------------

@dataclass
class TrackResult:
    """Everything one tracker run produced, on its grid of positions."""
    spec: FrameSpec
    chan: ChanCapture
    timing: Timing
    freq: PathFn
    pos: np.ndarray                          # int64 grid positions (lead-in < -8 included)
    z_d: np.ndarray = field(repr=False)
    m: np.ndarray = field(repr=False)
    c: np.ndarray = field(repr=False)
    u: np.ndarray = field(repr=False)
    P: np.ndarray = field(repr=False)
    N: np.ndarray = field(repr=False)
    psi: np.ndarray = field(repr=False)
    meas: np.ndarray = field(repr=False)
    pu: np.ndarray = field(repr=False)
    q: float = 0.0
    loglik: float = 0.0
    classes: Classes | None = None
    keying: np.ndarray | None = None
    ook: bool = False
    lead_in_s: float = 0.0
    z_ref: float = float("nan")
    df_max: float = 0.0
    report: TrackReport | None = None

    @property
    def p_lo(self) -> int:
        return int(self.pos[0])

    def gi(self, p) -> np.ndarray:
        """Grid index of position(s) p."""
        return np.asarray(p, dtype=np.int64) - self.p_lo

    @property
    def doppler_hz(self) -> float:
        return spread_of_q(self.q)

    def t_pos_s(self, p=None) -> np.ndarray:
        """Receiver time (s after t0) of positions p (default the grid)."""
        p = self.pos if p is None else p
        return rx_index(self.timing, p, self.spec.n_pos) / CH_FS


def _window_unknown(spec: FrameSpec, grid_pos, ook_possible=True) -> np.ndarray:
    """Positions whose MF support reaches units that may be keyed (5..183)."""
    out = np.zeros(len(grid_pos), dtype=bool)
    for pw in spec.win_start_pos:
        d = grid_pos - pw
        # units 5..183 are positions 10..367 of the window; the MF reaches 8 more
        lo_u = 5 if ook_possible else 8
        out |= (d >= lo_u * CW_UNIT - SPAN) & (d <= 184 * CW_UNIT - 1 + SPAN)
    return out


def _smooth(x, n: int) -> np.ndarray:
    if n <= 1:
        return np.asarray(x, dtype=np.float64)
    k = np.ones(n) / n
    xp = np.pad(np.asarray(x, dtype=np.float64), (n // 2, n - 1 - n // 2), mode="edge")
    return np.convolve(xp, k, mode="valid")


def _measurements(tr: TrackResult, kappa, sf, Pu_loc, c=None, meas=None):
    """(mu, R) for the Kalman filter from the current m, c, psi, N, Pu."""
    c = tr.c if c is None else c
    meas = tr.meas if meas is None else meas
    cp = c * tr.psi
    ok = meas & (np.abs(cp) >= C_MIN)
    mu = np.zeros(len(cp), dtype=np.complex128)
    R = np.full(len(cp), np.inf)
    mu[ok] = tr.m[ok] / cp[ok]
    R[ok] = (tr.N[ok] + kappa[ok] * Pu_loc[ok] * sf[ok]) / np.abs(cp[ok]) ** 2 + R_FLOOR * Pu_loc[ok]
    return mu, R


def _pu_initial(tr: TrackResult) -> float:
    cp = tr.c * tr.psi
    ok = tr.meas & (np.abs(cp) >= C_MIN)
    if not ok.any():
        return 1e-30
    e = np.abs(tr.m[ok]) ** 2 - tr.N[ok]
    return max(float(np.mean(e / np.abs(cp[ok]) ** 2)), 1e-3 * float(np.median(tr.N)), 1e-30)


def _recenter(tr: TrackResult) -> tuple[PathFn, float]:
    """Add the frequency u_hat still rotates at (30 s windows) to the path."""
    u = tr.u
    frame_ok = tr.pos >= -SPAN
    t = tr.t_pos_s()
    n_w = max(1, int(round(RECENTER_WIN_S / T_SYM)))
    tp, df, wt = [], [], []
    idx = np.nonzero(frame_ok)[0]
    for a in range(idx[0], idx[-1], n_w):
        b = min(a + n_w, len(u) - 1)
        if b - a < n_w // 4:
            continue
        s = np.sum(u[a + 1:b + 1] * np.conj(u[a:b]) * np.abs(u[a:b]) ** 2)
        e = np.sum(np.abs(u[a:b]) ** 4)
        if e <= 0:
            continue
        tp.append(0.5 * (t[a] + t[b]))
        df.append(float(np.angle(s)) / (2 * np.pi * T_SYM))
        wt.append(float(abs(s)))
    if not tp:
        return tr.freq, 0.0
    tp, df, wt = np.array(tp), np.array(df), np.array(wt)
    # re-sample the current path densely so the new spline keeps its shape
    td = np.arange(t[0], t[-1] + 1.0, 2.0)
    fd = tr.freq(td) + np.interp(td, tp, df)
    wd = np.interp(td, tp, wt / max(wt.max(), 1e-300)) + 1e-3
    return PathFn(td, fd, wd), float(np.max(np.abs(df)))


def _keying_tuple(cw_known):
    if cw_known is None:
        return None, False
    k, ook = cw_known
    return (None if k is None else np.asarray(k, dtype=np.uint8)), bool(ook)


def _lead_supported(chan: ChanCapture, z_d, tm: Timing, spec: FrameSpec, lead: float) -> bool:
    """Whether the claimed lead-in carrier is there: the plain carrier's
    MF power over the lead-in, against the frame's first 300 positions
    (whose template power is about 0.55), must be at least a third of
    what a carrier at the frame's level would give. A lead-in claimed
    where there is none would otherwise put a step in the gain."""
    n = int(math.floor(min(lead, LEAD_KEEP_S) / T_SYM))
    if n < 4:
        return True
    pl = np.arange(-SPAN - n, -SPAN - 1)
    pf = np.arange(0, min(300, spec.n_pos))
    ml = symbol_mf(z_d, CH_FS, tm, pl, t0_index=chan.t0_index, n_pos=spec.n_pos)
    mf = symbol_mf(z_d, CH_FS, tm, pf, t0_index=chan.t0_index, n_pos=spec.n_pos)
    t = np.concatenate([pl, pf]) * T_SYM
    N = chan.noise_psd_at(t) / T_SYM
    el = float(np.mean(np.abs(ml) ** 2 - N[:len(pl)]))
    ef = float(np.mean(np.abs(mf) ** 2 - N[len(pl):])) / 0.55
    return el > ef / 3


def track(chan: ChanCapture, spec: FrameSpec, det: Detection, classes: Classes | None = None,
          cw_known=None, prior=None, *, timing: Timing | None = None, freq=None,
          withhold_known: bool = False, outer: int = 2, refine: bool = True,
          lead_in_s: float | None = None, z_d=None) -> TrackResult:
    """One tracker run over a pass (design 6.5).

    `classes` (default round A) sets the template; `cw_known` = (keying,
    ook) once the callsign keying is known (the window positions become
    measurements), else the window's keyable units give none.
    `withhold_known` gives the known symbols no measurement (gate V).
    The timing is det.timing (or `timing`), refined on u_hat when
    `refine`; the frequency path is det.path (or `freq`), re-centred
    `outer` times.
    """
    if classes is None:
        classes = make_classes(spec, prior=prior)
    keying, ook = _keying_tuple(cw_known)
    fn = freq if freq is not None else freq_path_spline(det.path)
    tm = timing if timing is not None else det.timing
    lead = det.lead_in_s if lead_in_s is None else lead_in_s
    lead = float(min(max(lead, 0.0), 10.0))
    if z_d is None:
        z_d = derotate(chan, fn)
    z_d = remove_spread(z_d, chan, spec, tm, classes.spread)
    if lead > 0 and not _lead_supported(chan, z_d, tm, spec, lead):
        lead = 0.0
    n_lead = int(math.floor(min(lead, LEAD_KEEP_S) / T_SYM))
    p_lo = -SPAN - n_lead
    grid = np.arange(p_lo, spec.n_pos, dtype=np.int64)
    lay = layout(spec)
    pos_known = np.asarray(lay.pos)[classes.known]
    is_known = np.zeros(len(grid), dtype=bool)
    is_known[pos_known - p_lo] = True

    keyed_from = spec.keyed_start_pos - lead / T_SYM
    mu_t, nu_t = classes.mu, classes.nu
    if withhold_known:
        mu_t = np.where(classes.known, 0.0, classes.mu)
        nu_t = np.where(classes.known, 1.0, classes.nu)
    c, sf = templates(spec, grid, mu_t, nu_t, keying, ook, keyed_from)
    kappa = np.where(is_known & (not withhold_known), KAPPA_SELF_KNOWN, KAPPA_SELF)

    meas = np.ones(len(grid), dtype=bool)
    if keying is None and spec.n_win:
        meas &= ~_window_unknown(spec, grid)
    if withhold_known:
        meas &= ~is_known
    if n_lead == 0:
        meas &= grid >= -SPAN

    n_pos = spec.n_pos
    tr = TrackResult(spec=spec, chan=chan, timing=tm, freq=fn, pos=grid, z_d=z_d,
                     m=None, c=c, u=None, P=None, N=None, psi=None, meas=meas, pu=None,
                     classes=classes, keying=keying, ook=ook, lead_in_s=lead)

    def measure():
        tr.m = symbol_mf(tr.z_d, CH_FS, tr.timing, grid, t0_index=chan.t0_index, n_pos=n_pos)
        tr.psi = psi_at(chan.keep, tr.timing, grid, t0_index=chan.t0_index, n_pos=n_pos)
        tr.N = chan.noise_psd_at(tr.t_pos_s()) / T_SYM
        tr.meas = meas & (tr.psi >= PSI_MIN)

    def pu_loc():
        if tr.u is None:
            Pu_g = _pu_initial(tr)
            return np.full(len(grid), Pu_g), Pu_g
        Pu_loc = np.maximum(_smooth(np.abs(tr.u) ** 2 + tr.P, int(PU_SMOOTH_S / T_SYM)), 1e-30)
        return Pu_loc, max(float(np.mean(Pu_loc[grid >= -SPAN])), 1e-30)

    def run(q_grid=Q_GRID):
        Pu_loc, Pu_g = pu_loc()
        mu, R = _measurements(tr, kappa, sf, Pu_loc)
        q, _ = select_q(mu, R, Pu_g, q_grid)
        u, P, ll = kalman_rts(mu, R, q, Pu_g)
        tr.u, tr.P, tr.q, tr.loglik, tr.pu = u, P, q, ll, Pu_loc
        return q

    def near(q):
        i = int(np.argmin(np.abs(np.log(np.array(Q_GRID)) - math.log(q))))
        return Q_GRID[max(0, i - 2):i + 3]

    measure()
    run()
    if refine and not withhold_known:
        # the timing is refined against u_hat at the gain agility q chosen
        # *without* the known symbols: chosen with them, a timing error
        # there makes an agile q look likely, and u_hat then follows the
        # error and confirms it
        c_w, sf_w = templates(spec, grid, np.where(classes.known, 0.0, classes.mu),
                              np.where(classes.known, 1.0, classes.nu), keying, ook, keyed_from)
        kap_w = np.full(len(grid), KAPPA_SELF)
        for _ in range(3):
            Pu_loc, Pu_g = pu_loc()
            mw, Rw = _measurements(tr, kap_w, sf_w, Pu_loc, c=c_w, meas=tr.meas & ~is_known)
            qw, _ = select_q(mw, Rw, Pu_g)
            run((qw,))
            new_tm = refine_timing(tr)
            if new_tm is None:
                break
            moved = abs(new_tm.tau0 - tr.timing.tau0) + abs(
                (new_tm.ppm - tr.timing.ppm) * 1e-6 * n_pos * _SPS)
            tr.timing = new_tm
            if classes.spread is not None:
                tr.z_d = remove_spread(derotate(chan, tr.freq), chan, spec, tr.timing,
                                       classes.spread)
            measure()
            if moved < 0.02:
                break
    q = run()
    df_max = 0.0
    for it in range(outer + 1):
        if it == outer and df_max <= RECENTER_MIN_HZ:
            break
        if it > outer:
            break
        fn2, df_max = _recenter(tr)
        if df_max < 0.002:
            break
        tr.freq = fn2
        tr.z_d = remove_spread(derotate(chan, fn2), chan, spec, tr.timing, classes.spread)
        measure()
        q = run(near(q))
    tr.df_max = df_max
    return tr


def refine_timing(tr: TrackResult, iters: int = 3, u_ref=None, P_ref=None) -> Timing | None:
    """Newton steps on the known symbols' likelihood given u_hat (design 6.4 step 3).

    u_ref, P_ref: the gain estimate to use (default the track's own); `track`
    passes one made with the known symbols withheld.

    l(theta) = sum_known |m_k(theta) - u_hat_k c_k psi_k|^2 / Var_k with
    theta = (tau0, slip) in CH samples (slip = clock error accumulated over
    the frame), plus gamma for frames over 10,000 positions when it raises
    the known symbols' Z by more than 1. The inverse Hessian of l is the
    covariance (complex Gaussian: it is the Fisher information).
    """
    spec = tr.spec
    n_pos = spec.n_pos
    sps = _SPS
    gk = tr.gi(known_symbols(spec).pos if tr.classes is None else
               np.asarray(layout(spec).pos)[tr.classes.known])
    gk = gk[(gk >= 0) & (gk < len(tr.pos))]
    gk = gk[tr.meas[gk]]
    if len(gk) < 8:
        return None
    pk = tr.pos[gk].astype(np.float64)
    U = tr.u if u_ref is None else u_ref
    PP = tr.P if P_ref is None else P_ref
    target = U[gk] * tr.c[gk] * tr.psi[gk]
    var = tr.N[gk] + KAPPA_SELF_KNOWN * tr.pu[gk] * 0.1 + np.abs(tr.c[gk]) ** 2 * PP[gk]
    chan = tr.chan

    def timing_of(th):
        tau0, slip, gam = th
        return Timing(tau0=tau0, ppm=slip / (n_pos * sps) * 1e6, gamma=gam, z=0.0,
                      cov=np.zeros((3, 3)))

    def ell(th):
        m = symbol_mf(tr.z_d, CH_FS, timing_of(th), pk, t0_index=chan.t0_index, n_pos=n_pos)
        return float(np.sum(np.abs(m - target) ** 2 / var))

    def zk(th):
        m = symbol_mf(tr.z_d, CH_FS, timing_of(th), pk, t0_index=chan.t0_index, n_pos=n_pos)
        w = tr.c[gk].imag
        num = np.sum(w * np.imag(m * np.conj(U[gk])))
        den = math.sqrt(max(float(np.sum(w * w * np.abs(U[gk]) ** 2 * tr.N[gk] / 2)), 1e-300))
        return num / den

    def newton(th, dims, h=0.3):
        th = np.array(th, dtype=np.float64)
        H = None
        for _ in range(iters):
            f0 = ell(th)
            k = len(dims)
            g = np.zeros(k)
            H = np.zeros((k, k))
            fp, fm = np.zeros(k), np.zeros(k)
            for a, da in enumerate(dims):
                e = np.zeros(3)
                e[da] = h
                fp[a], fm[a] = ell(th + e), ell(th - e)
                g[a] = (fp[a] - fm[a]) / (2 * h)
                H[a, a] = (fp[a] - 2 * f0 + fm[a]) / h ** 2
            for a in range(k):
                for b in range(a + 1, k):
                    e = np.zeros(3)
                    e[dims[a]] = h
                    e[dims[b]] = h
                    fpp = ell(th + e)
                    H[a, b] = H[b, a] = (fpp - fp[a] - fp[b] + f0) / h ** 2
            try:
                w, V = np.linalg.eigh(H)
            except np.linalg.LinAlgError:
                break
            if np.any(w <= 0):
                break
            step = -np.linalg.solve(H, g)
            step = np.clip(step, -1.5, 1.5)
            for a, da in enumerate(dims):
                th[da] += step[a]
            if np.max(np.abs(step)) < 0.01:
                break
        return th, H

    tm = tr.timing
    th0 = np.array([tm.tau0, tm.ppm * 1e-6 * n_pos * sps, tm.gamma])
    th, H = newton(th0, (0, 1))
    if H is None or not np.all(np.isfinite(th)):
        return None
    dims = (0, 1)
    if n_pos > 10000:
        th3, H3 = newton(th, (0, 1, 2))
        if H3 is not None and np.all(np.isfinite(th3)) and zk(th3) > zk(th) + 1.0:
            th, H, dims = th3, H3, (0, 1, 2)
    try:
        cov_s = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return None
    if np.any(np.diag(cov_s) <= 0):
        return None
    J = np.diag([1.0, 1e6 / (n_pos * sps), 1.0])[np.ix_(dims, dims)]
    cov = np.zeros((3, 3))
    cov[np.ix_(dims, dims)] = J @ cov_s @ J.T
    out = timing_of(th)
    out.z = float(zk(th))
    out.cov = cov
    return out


# --- the verification gate ----------------------------------------------------------------------

def z_ref_stat(tr: TrackResult, c_known: np.ndarray) -> float:
    """Z_ref over the known positions: sum_k w_k t_k / sqrt(sum_k w_k^2 V_k).

    t_k = c''_k Im(m_k conj(u_k)), c''_k = Im(c_k) of the template with the
    known symbols known, V_k its variance from the noise of m_k and the
    error P_k of u_hat, and w_k = c''_k^2 |u_k|^2 / V_k its mean over its
    variance (the matched weighting: a known symbol in a stretch the
    tracker could only extrapolate across -- the preamble on a fading
    path -- has a large P_k and counts little). u_hat must not have seen
    m_k (`withhold_known`), and w_k does not depend on m_k, so Z_ref is
    N(0, 1) on noise.
    """
    gk = tr.gi(np.asarray(layout(tr.spec).pos)[tr.classes.known])
    gk = gk[(gk >= 0) & (gk < len(tr.pos))]
    gk = gk[tr.psi[gk] >= PSI_MIN]
    if len(gk) == 0:
        return 0.0
    ci = c_known[gk].imag
    m, u, N, P = tr.m[gk], tr.u[gk], tr.N[gk], tr.P[gk]
    u2 = np.abs(u) ** 2
    t = ci * np.imag(m * np.conj(u))
    V = ci * ci * (N * u2 + np.abs(c_known[gk]) ** 2 * u2 * P + N * P) / 2
    ok = V > 0
    if not ok.any():
        return 0.0
    w = ci[ok] ** 2 * u2[ok] / V[ok]
    den = float(np.sum(w * w * V[ok]))
    return float(np.sum(w * t[ok])) / math.sqrt(max(den, 1e-300))


def verify(ch, ch_fs, spec: FrameSpec, cand: Detection, *, fit: bool = True,
           tau_range=None) -> Detection | None:
    """Gate V (design 6.3): fit the timing, track with the known symbols
    withheld, accept at Z_ref > Z_ACCEPT. Returns the detection with its
    timing and z_ref filled in, or None.

    `ch` is the candidate's `ChanCapture`. With fit=False the candidate's
    own timing is used (Z_ref is then N(0, 1) on noise).
    """
    # the acquisition's drift can be wrong (a preamble's estimate on a fading
    # or warming-up signal): the path is measured on the frame itself
    # first, from the candidate's path and, failing that, its flat version
    tr, path = None, cand.path
    for base in (cand.path, _flat_path(cand.path), None):
        # the last try, a line over the whole frame, is for weak candidates
        pth = refine_path(ch, spec, base) if base is not None else line_path(ch, spec, cand.path)
        alt = verify_track(ch, spec, replace(cand, path=pth), fit=fit, tau_range=tau_range)
        if alt is not None and (tr is None or alt.z_ref > tr.z_ref):
            tr, path = alt, pth
        if tr is not None and tr.z_ref > Z_ACCEPT:
            break
        if base is None and fit and cand.timing is None and V_TIMING_TRIES > 1:
            # a blind (A3) candidate near the floor, on its whole-frame line:
            # the coarse timing statistic may rank the true timing below a
            # noise peak, so the runners-up are tracked too (each Z_ref is
            # N(0, 1) on noise; V_TIMING_TRIES trials move the gate's false
            # alarm rate from about 1e-9 to 1e-8 per candidate)
            fn = freq_path_spline(pth)
            tms = fit_timing(ch, CH_FS, fn, spec, None, None, t0_index=ch.t0_index,
                             z_d=derotate(ch, fn), n_best=V_TIMING_TRIES)
            for tm in tms[1:]:
                alt = verify_track(ch, spec, replace(cand, path=pth, timing=tm), fit=False)
                if alt is not None and (tr is None or alt.z_ref > tr.z_ref):
                    tr, path = alt, pth
                if tr is not None and tr.z_ref > Z_ACCEPT:
                    break
        if tr is not None and tr.z_ref > Z_ACCEPT:
            break
    if tr is None or not tr.z_ref > Z_ACCEPT:
        return None
    return Detection(f_hz=cand.f_hz, path=path, timing=tr.timing, z_ref=tr.z_ref,
                     method=cand.method, lead_in_s=cand.lead_in_s)


def _carrier_lines(z, chan: ChanCapture, spec: FrameSpec, seg_s: float, span_hz: float):
    """(t, f, ratio) of the carrier line in each seg_s segment of z over the
    frame, walking from t0: each searched +-span_hz around the last."""
    fs = float(chan.fs)
    n = int(round(seg_s * fs))
    i0 = max(int(math.ceil(chan.t0_index)), 0)
    i1 = min(int(chan.t0_index + spec.n_pos * T_SYM * fs), len(z))
    nfft = 1 << math.ceil(math.log2(8 * n))
    f = np.fft.fftfreq(nfft, 1.0 / fs)
    win = np.hanning(n)
    tt, ff, ww = [], [], []
    prev = 0.0
    for a in range(i0, max(i1 - n // 2, i0 + 1), n):
        seg = z[a:a + n]
        if len(seg) < n // 2:
            break
        S = np.abs(np.fft.fft(seg * win[:len(seg)], nfft)) ** 2
        near = np.abs(f - prev) <= span_hz
        k = np.nonzero(near)[0][int(np.argmax(S[near]))]
        med = float(np.median(S[np.abs(f) <= 4 * span_hz + 10]))
        if not (med > 0 and S[k] >= 10 * med):     # a blanked (all-zero) segment has med 0
            continue
        d = _parabola(S[(k - 1) % nfft], S[k], S[(k + 1) % nfft])
        fr = f[k] + d * fs / nfft
        tt.append(float(chan.t_s(a + len(seg) / 2)))
        ff.append(fr)
        ww.append(float(S[k] / med))
        prev = fr
    return np.array(tt), np.array(ff), np.array(ww)


def refine_path(chan: ChanCapture, spec: FrameSpec, path: FreqPath, seg_s: float = PATH_SEG_S,
                span_hz: float = PATH_SPAN_HZ) -> FreqPath:
    """The carrier's frequency path, measured on the frame: `path` (fitted
    with knots every 60 s) plus the residual carrier frequency of each
    `seg_s` segment.

    The CE carrier keeps about half the signal's power, so a segment's
    spectrum has a clear line at it. Segments are taken in time order and
    each is searched +-span_hz around the residual the previous one found
    (the first around 0), so the measurement walks along a path whose
    drift the acquisition got wrong (a preamble's drift estimate on a
    fading or warming-up signal) instead of trusting it. Segments whose
    line is under 10 times the spectrum's median are skipped.

    A second pass with PATH_FINE_S segments, +-1 Hz about the first,
    measures the path finely enough for wander with a 10 s correlation
    time; it is used when the carrier is strong enough that at least
    PATH_FINE_SHARE of its segments find their line, at a median ratio of
    PATH_FINE_RATIO (weaker, its noise costs more than wander does).
    """
    fn = PathFn(path.t_s, path.f_hz, path.weight, 60.0)
    z = derotate(chan, fn)
    tt, ff, ww = _carrier_lines(z, chan, spec, seg_s, span_hz)
    if len(tt) < 2:
        return path
    res = PathFn(tt, ff, ww, max(3 * seg_s, 60.0))
    step = seg_s
    ts = np.r_[tt[0] - 10, tt, tt[-1] + 10]
    fn1 = PathFn(ts, fn(ts) + res(ts), None, 60.0)
    z1 = derotate(chan, fn1)
    t2, f2, w2 = _carrier_lines(z1, chan, spec, PATH_FINE_S, 1.0)
    n_fine = spec.n_pos * T_SYM / PATH_FINE_S
    if len(t2) >= max(8, PATH_FINE_SHARE * n_fine) and np.median(w2) >= PATH_FINE_RATIO:
        res2 = PathFn(t2, f2, w2, 2 * PATH_FINE_S)
        t = np.arange(min(t2[0], 0.0) - 10.0, t2[-1] + 2.0 * PATH_FINE_S, 1.0)
        return FreqPath(t_s=t, f_hz=fn1(t) + res2(t), weight=np.ones(len(t)))
    t = np.arange(min(tt[0], 0.0) - 10.0, tt[-1] + 2.0 * step, 2.0)
    return FreqPath(t_s=t, f_hz=fn(t) + res(t), weight=np.ones(len(t)))


LINE_SEG_S = 60.0                # line_path: segment length (1/60 Hz bins)
LINE_SPAN_HZ = 1.0               # ...searched either side of the candidate's line
LINE_DRIFT_HZ_MIN = 0.4          # ...and drift either side of its drift, Hz/min
LINE_DRIFT_STEP = 0.01


def line_path(chan: ChanCapture, spec: FrameSpec, path: FreqPath) -> FreqPath:
    """The best straight frequency line over the whole frame, near `path`'s.

    For a candidate near the detection floor `refine_path`'s 20 s
    segments see the carrier at about 10 dB and can walk off it, and the
    A3 path itself wanders by its 0.25 Hz bins. Over the whole frame the
    carrier is strong: 60 s segments' power spectra (the line about 15 dB
    over its bin at -34 dB), shifted along each drift hypothesis and
    summed, find the line's offset and drift to a few mHz. Drift within
    +-LINE_DRIFT_HZ_MIN of the path's own straight-line fit, offset
    within +-LINE_SPAN_HZ.
    """
    fn = PathFn(path.t_s, path.f_hz, path.weight, 60.0)
    t_frame = np.array([0.0, spec.n_pos * T_SYM])
    f_lin = fn(t_frame)
    d0 = float(f_lin[1] - f_lin[0]) / float(t_frame[1])          # Hz/s
    tm = float(t_frame[1]) / 2
    f_mid = float(f_lin[0]) + d0 * tm
    tl = np.array([-10.0, 0.0, 10.0])
    flat = PathFn(tm + tl, f_mid + d0 * tl)
    z = derotate(chan, flat)                       # the path's line removed
    fs = float(chan.fs)
    n = int(round(LINE_SEG_S * fs))
    i0 = max(int(math.ceil(chan.t0_index)), 0)
    i1 = min(int(chan.t0_index + spec.n_pos * T_SYM * fs), len(z))
    nfft = 1 << math.ceil(math.log2(4 * n))
    win = np.hanning(n)
    S, ts = [], []
    for a in range(i0, i1 - n // 2, n):
        seg = z[a:a + n]
        if len(seg) < n // 2:
            break
        P = np.abs(np.fft.fft(seg * win[:len(seg)], nfft)) ** 2
        S.append(P / max(float(np.median(P)), 1e-300))
        ts.append(float(chan.t_s(a + len(seg) / 2)) - tm)
    if len(S) < 3:
        return path
    S = np.array(S)
    df = fs / nfft
    kspan = int(math.ceil(LINE_SPAN_HZ / df))
    k = np.r_[0:kspan + 1, nfft - kspan:nfft]              # bins within +-span of 0 Hz
    fk = np.where(k > nfft // 2, k - nfft, k) * df
    ts = np.asarray(ts)
    rows = np.arange(len(S))[:, None]

    def search(drifts):
        best = (-np.inf, 0.0, 0.0)
        for dd in drifts:
            sh = np.round(dd * ts / df).astype(np.int64)     # the line's bin in each segment
            tot = S[rows, (k[None, :] + sh[:, None]) % nfft].sum(axis=0)
            j = int(np.argmax(tot))
            if tot[j] > best[0]:
                best = (float(tot[j]), float(fk[j]), float(dd))
        return best

    # coarse drift steps, then steps fine enough that the line moves under
    # a bin across the frame
    _, _, dd = search(np.arange(-LINE_DRIFT_HZ_MIN, LINE_DRIFT_HZ_MIN + 1e-9,
                                LINE_DRIFT_STEP) / 60.0)
    fine = 0.5 * df / max(float(np.max(np.abs(ts))), 1.0)
    _, foff, dd = search(dd + np.arange(-LINE_DRIFT_STEP / 60.0, LINE_DRIFT_STEP / 60.0 + 1e-12,
                                        fine))
    t = np.arange(-LEAD_LINE_S, spec.n_pos * T_SYM + 10.0, 10.0)
    f = f_mid + foff + (d0 + dd) * (t - tm)
    return FreqPath(t_s=t, f_hz=f, weight=np.ones(len(t)))


LEAD_LINE_S = 20.0


def _flat_path(path: FreqPath) -> FreqPath:
    f0 = float(freq_path_spline(path)(np.array([0.0]))[0])
    t = np.array([-10.0, 0.0, 1800.0])
    return FreqPath(t_s=t, f_hz=np.full(3, f0), weight=np.ones(3))


def verify_track(chan: ChanCapture, spec: FrameSpec, cand: Detection, *, fit: bool = True,
                 tau_range=None) -> TrackResult | None:
    """The tracker run behind `verify`, with tr.z_ref set."""
    fn = freq_path_spline(cand.path)
    z_d = derotate(chan, fn)
    tm = cand.timing
    if fit or tm is None:
        tau0 = None if tm is None else tm.tau0
        rng = tau_range
        if tau0 is not None and rng is None:
            sd = math.sqrt(max(float(tm.cov[0, 0]), 0.0)) if tm.cov.size else 1.0
            rng = max(3.0, 4.0 * sd)
        tm = fit_timing(chan, CH_FS, fn, spec, None, tau0, t0_index=chan.t0_index,
                        tau_range=rng, z_d=z_d)
    classes = make_classes(spec)
    tr = track(chan, spec, cand, classes, timing=tm, freq=fn, withhold_known=True, outer=0,
               refine=False, z_d=z_d)
    lay = layout(spec)
    keyed_from = spec.keyed_start_pos - tr.lead_in_s / T_SYM
    c_known, _ = templates(spec, tr.pos, classes.mu, classes.nu, None, False, keyed_from)
    tr.z_ref = z_ref_stat(tr, c_known)
    del lay
    return tr


# --- report -------------------------------------------------------------------------------------

def report(tr: TrackResult, kappa: float = 1.0, suspect: bool = False) -> TrackReport:
    """The pass's TrackReport (design 6.5 step 7)."""
    spec = tr.spec
    frame = (tr.pos >= 0)
    t = tr.t_pos_s()
    tf = t[frame]
    fA = tr.freq(tf)
    line = np.polyfit(tf, fA, 1) if len(tf) > 2 else np.array([0.0, float(tr.freq(0.0))])
    # instantaneous frequency from u_hat, smoothed over 3 s, around the line
    u = tr.u[frame]
    w = np.abs(u[1:]) * np.abs(u[:-1])
    n3 = max(1, int(round(3.0 / T_SYM)))
    num = _smooth((u[1:] * np.conj(u[:-1])).real, n3) + 1j * _smooth(
        (u[1:] * np.conj(u[:-1])).imag, n3)
    finst = tr.freq(tf[:-1]) + np.angle(num) / (2 * np.pi * T_SYM)
    res = finst - np.polyval(line, tf[:-1])
    good = w > 0.25 * np.median(w) if len(w) else np.zeros(0, bool)
    wander = float(np.sqrt(np.mean(res[good] ** 2))) if good.any() else 0.0
    keyed = frame & (tr.pos < spec.keyed_end_pos)
    pu = float(np.mean(np.abs(tr.u[keyed]) ** 2)) if keyed.any() else 0.0
    nf = tr.chan.noise_floor()
    snr = 10 * math.log10(max(pu, 1e-30) / max(nf * 2500.0, 1e-300))
    big = np.abs(tr.u) ** 2 > tr.pu / 4
    d = np.angle(tr.u[1:] * np.conj(tr.u[:-1]))
    slip_free = not bool(np.any((np.abs(d) > np.pi / 2) & big[1:] & big[:-1]))
    return TrackReport(offset_hz=float(tr.freq(0.0)), drift_hz_per_min=float(line[0] * 60.0),
                       wander_hz_rms=wander, doppler_hz=float(tr.doppler_hz),
                       ppm=float(tr.timing.ppm), snr2500_db=float(snr),
                       z_ref=float(tr.z_ref), kappa=float(kappa), slip_free=slip_free,
                       suspect=bool(suspect))


# --- genie (tests and calibration) --------------------------------------------------------------

def genie_track(chan: ChanCapture, spec: FrameSpec, truth, carrier_hz: float,
                classes: Classes | None = None) -> TrackResult:
    """A TrackResult with the channel simulator's true timing and gain.

    `truth` is a `channel.Truth`: u relative to a carrier at the nominal
    carrier_hz with zero phase at t0, and the receive time of every
    position. The front end's overall complex scale (the inverse AGC) is
    fitted once by least squares over the frame; P = 0.
    """
    classes = make_classes(spec) if classes is None else classes
    fn = PathFn(np.array([-30.0, 0.0, 30.0]), np.full(3, float(carrier_hz)))
    z_d = derotate(chan, fn)
    t = np.asarray(truth.t_pos_s, dtype=np.float64)
    p = np.arange(spec.n_pos)
    slope, icpt = np.polyfit(p, t * CH_FS, 1)
    tm = Timing(tau0=float(icpt), ppm=float((slope / _SPS - 1) * 1e6), gamma=0.0, z=0.0,
                cov=np.zeros((3, 3)))
    z_d = remove_spread(z_d, chan, spec, tm, classes.spread)
    grid = np.arange(-SPAN, spec.n_pos, dtype=np.int64)
    c, _ = templates(spec, grid, classes.mu, classes.nu)
    m = symbol_mf(z_d, CH_FS, tm, grid, t0_index=chan.t0_index, n_pos=spec.n_pos)
    ut = np.concatenate([np.full(SPAN, truth.u[0]), np.asarray(truth.u)])
    ref = c * ut
    fit = grid >= 0
    if spec.n_win:
        fit &= ~_window_unknown(spec, grid)      # keyed units are not in the round-A template
    alpha = np.vdot(ref[fit], m[fit]) / np.vdot(ref[fit], ref[fit])
    u = alpha * ut
    tr = TrackResult(spec=spec, chan=chan, timing=tm, freq=fn, pos=grid, z_d=z_d, m=m, c=c,
                     u=u, P=np.zeros(len(grid)), N=None, psi=None,
                     meas=np.ones(len(grid), dtype=bool), pu=np.abs(u) ** 2, classes=classes)
    tr.psi = psi_at(chan.keep, tm, grid, t0_index=chan.t0_index, n_pos=spec.n_pos)
    tr.N = chan.noise_psd_at(tr.t_pos_s()) / T_SYM
    return tr

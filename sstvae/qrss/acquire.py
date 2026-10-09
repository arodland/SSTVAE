"""Acquisition (design 6.3, WP5): A1 carrier lines, A2 preamble search,
A3 whole-slot track-before-detect, and their false-alarm constants.

Not a format module. Every detector here reports a statistic whose
distribution on noise is known -- Gamma for A1 and A2 by construction,
a Gumbel fitted by measurement for A3 -- because the single acceptance
decision is made later, by the verification gate V (`track.verify`,
Z_ref > 6) that every candidate from here goes through. These
detectors only have to keep the candidate list short without losing a
real signal.

Inputs are the blanked, normalised front end FE' (`frontend.blank`,
`frontend.normalise`): complex at FE_FS = 4000 Hz with 0 Hz at 1500 Hz
audio, and t0_index, the FE sample index of the nominal t0 (QH + 1 s)
on the receiver's clock. Frequencies are reported in audio Hz; times
in seconds after t0.

**A1. Carrier lines.** 4 s Hann frames (0.25 Hz bins) at a 2 s hop over
[t0 - 2.5 s, t0 + 22.5 s], each normalised by its median over the
300-2700 Hz bins and then each bin by the local spectral shape (the
median over frames, then over +-40 Hz: `cfar_normalise`), so a noise bin
is Exp(1) however the receiver's passband tilts; summed noncoherently along
linear drift lines of +-3 Hz/min in 0.5 Hz/min steps: a sum of M frames
is Gamma on noise -- Gamma(M), widened slightly for the 50% frame
overlap (`a1_gamma`) -- thresholded at a 1e-4 false-alarm rate per cell.
The whole FE band is searched, which covers +-400 Hz of an uncalibrated
Si5351 about any carrier the sender chose. Every local maximum at least
10 Hz from a stronger one is reported, up to 16: a weak signal beside a
strong one is A2's to judge, not A1's.

**A2. Preamble search** at each A1 candidate: channelise (drift taken
out), notch the carrier (y - LP(y), LP zero-phase +-0.6 Hz: this also
removes any lead-in), matched-filter with the CE pulse, and correlate
with the preamble in ten 66-symbol (2 s) chunks with zero-mean
templates, over tau0 in +-2.5 s and df in +-0.5 Hz. The statistic

    Lambda = sum_c |sum_s (a_s - mean_c a) m(tau0 + s T) e^{-j 2 pi df t_s}|^2
             / (sigma^2 sum_s (a_s - mean_c a)^2)

is Gamma(10, 1) per cell on noise. The chunk templates share one phase
reference, so runs of 2, 5 and 10 chunk sums also add coherently
(`preamble_scales`): 4, 10 and 20 s chunks, Gamma(5), Gamma(2) and
Gamma(1) on noise. A cell passes when its best scale's p, times the four
scales tested, is under 1e-7. The 2 s chunks alone (the design's A2)
detect ~75% of preambles at -28 dB; with the longer runs ~97%, while a
path that wanders still has the 2 s statistic, at 1.2 dB of threshold.
sigma^2 is per chunk: the MF output's median power, raised to the local
mean power over 6 s where that is higher, so the transient a switching
carrier leaves behind the notch (a lead-in ending) reads as noise. The
peak is then refined: df (and a residual drift, when significant) from
a line through the chunk phases, coherent over 20 s, then timing on the
coherent correlation of the whole preamble at fractional offsets (the
MF output is band-limited, so a cubic spline through it is exact). At
-25 dB that timing reaches the Cramer-Rao bound, about T/20 rms.

**Symbol-rate aliases.** A CE signal correlates with the preamble at
f +- k/T as well (the symbol rotation is then whole turns), down only by
the matched filter's response there, so a strong signal passes A2 at
+-33 Hz too. `acquire` drops a hit at a stronger hit's f +- k/T (k = 1, 2,
within 0.5 Hz) with the same timing (within T/2) and a Lambda 3x
smaller.

**Lead-in** (spec 2.8): A1 and A2 never use it (A2 by the notch and
the zero-mean templates); A3 sums it with the rest; `lead_in_span`
measures it for the tracker, 1 s windows of the notch-free MF output
walking back from the keying start, Z > 4 per window.

**A3. Track-before-detect** over the whole stored slot: the same 4 s/2 s
spectrogram from t0 - 10 s (a lead-in adds to it), cut into channels of
+-40 Hz at each A1 candidate and every 50 Hz across the band; a Viterbi
path per channel with |dbin| <= 2 per hop (15 Hz/min), transition cost
0.5 |dbin| (the noise emission's sd is 1) and emission = normalised
power - 1, normalised as A1's (`cfar_normalise`, local spectral shape
included: with a band-wide CFAR alone, a 1 dB passband tilt put 13 of
FULL's 47 channels past p < 1e-3 on noise). Z = sum of emissions / sqrt(n_frames). Z on noise grows like
sqrt(n_frames), so its Gumbel (`TBD_GUMBEL_*`) is tabulated against
n_frames, measured on noise-only captures (`tests/test_qrss_acquire.py`,
A-7) and interpolated in sqrt(n_frames): the slot length is a
`FrameSpec` property and may still change. Candidates pass at p < 1e-3.

The spectrogram is computed once on FE and the channels are bin ranges
of it, which gives the same 0.25 Hz bins as channelising first at a
fraction of the cost.

**Timing convention.** A detection's `Timing.tau0` is in CH samples
(1/250 s) counted from the nominal t0 on the receiver's clock: the
preamble's first symbol was received tau0/250 s after nominal t0.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage, signal, special
from scipy.interpolate import CubicSpline

from . import ce, frontend
from .constants import CH_FS, FE_CENTER_HZ, FE_FS, N_PRE, T_SYM
from .frame import FrameSpec
from .sequences import preamble_ce
from .types import CarrierCandidate, Detection, FreqPath, Timing

# --- shared spectrogram ------------------------------------------------------------------

STFT_FRAME_S = 4.0                       # 0.25 Hz bins
STFT_HOP_S = 2.0
BAND_HZ = (300.0, 2700.0)                # audio band searched (and the CFAR median's)
LN2 = math.log(2.0)
CFAR_LOCAL_HZ = 40.0                     # local CFAR: the spectral shape over +-40 Hz
CFAR_LOCAL_BINS = 2 * int(round(CFAR_LOCAL_HZ * STFT_FRAME_S)) + 1

# --- A1 --------------------------------------------------------------------------------------

A1_SPAN_S = (-2.5, 22.5)                 # window, s after t0
A1_DRIFTS = tuple(np.round(np.arange(-3.0, 3.0001, 0.5), 3))   # Hz/min
A1_PFA = 1e-4                            # per (bin, drift) cell
A1_MAX_CANDS = 16
A1_MIN_SEP_HZ = 10.0
A1_DRIFT_MARGIN = 0.1                    # drift reported only if it adds 10% to the line

# --- A2 --------------------------------------------------------------------------------------

A2_TAU_S = 2.5                           # tau0 searched over +-2.5 s
A2_DF_HZ = tuple(np.round(np.arange(-0.5, 0.5001, 0.05), 3))
A2_CHUNK = 66                            # symbols per coherent chunk (2.0 s)
A2_PFA = 1e-7                            # per (tau0, df) cell
NOTCH_HZ = 0.6                           # carrier notch: y - LP(y), LP zero phase +-0.6 Hz
NOTCH_TAPS = 2001                        # 8 s Hann-windowed sinc at 250 Hz
_A2_MARGIN_S = 6.0                       # FE taken either side of the search, for the filters
A2_DRIFT_SIGMAS = 3.0                    # a residual drift is kept only when this significant
A2_LOCAL_S = 6.0                         # local noise level: mean |m|^2 over 3 chunks
A2_SCALES = (1, 2, 5, 10)                # coherent runs of 2 s chunks: 2, 4, 10 and 20 s
A2_ALIAS_K = (1, 2)                      # a hit at f +- k/T of a stronger one, same timing,
A2_ALIAS_HZ = 0.5                        # is that one seen through the symbol-rate alias
A2_ALIAS_RATIO = 3.0

# --- A3 --------------------------------------------------------------------------------------

TBD_START_S = -10.0                      # first frame starts here, so a lead-in contributes
TBD_END_PAD_S = 2.0                      # frames run to the end of keying plus this
TBD_HALF_HZ = 40.0                       # channel half-width
TBD_GRID_HZ = 50.0                       # channel grid across the band
TBD_MAX_STEP = 2                         # bins per hop (0.5 Hz / 2 s = 15 Hz/min)
TBD_COST = 0.5                           # per bin of step, in noise-emission sd
TBD_PFA = 1e-3
TBD_DEDUP_HZ = 10.0

# Gumbel of Z on noise-only captures, tabulated against n_frames (the
# TINY, SHORT, MEDIUM and FULL frames) and interpolated linearly in
# sqrt(n_frames): Z on noise grows like 1.6 sqrt(n_frames) while its
# spread stays near 0.8. Measured 2026-10-09, after the local CFAR
# (`spectral_shape`) went in, by the procedure of
# tests/test_qrss_acquire.py::test_a7_measure_tbd_gumbel (slow): 2 x 200
# complex white-noise captures per length (seeds 1000 + n and 2000 + n)
# at 500 Hz, 6 disjoint 321-bin (+-40 Hz) channels each, through
# production's spectrogram, CFAR and Viterbi; Gumbel by moments
# (beta = sd sqrt(6)/pi, mu = mean - 0.5772 beta) over the 2400 Z values.
# The true tail is lighter than any Gumbel's (an MLE fit put 10% of its
# nominal share past its own 1% point), so the moment fit, whose 10%
# point is exact, is the tighter of the two and still conservative:
# 8-60% of nominal past its 1% point.
TBD_GUMBEL_N = (31, 132, 331, 896)
TBD_GUMBEL_MU = (9.845, 18.637, 28.924, 47.307)
TBD_GUMBEL_BETA = (0.702, 0.627, 0.624, 0.601)


def _frame_len(fs: float) -> int:
    return int(round(STFT_FRAME_S * fs))


def frame_window(fs: float) -> np.ndarray:
    """Hann window of one 4 s frame, normalised so a noise bin has mean E|x|^2 * L."""
    return signal.windows.hann(_frame_len(fs), sym=False)


def spectrogram(x, fs: float, starts, bins=None, chunk: int = 32) -> np.ndarray:
    """float32[n_frames, n_bins]: |FFT(w * frame)|^2 at frame starts `starts`
    (sample indices into x, rounded; samples outside x count 0), keeping
    the fftfreq bins `bins` (all when None)."""
    x = np.asarray(x)
    L = _frame_len(fs)
    w = frame_window(fs)
    starts = np.round(np.asarray(starts, dtype=np.float64)).astype(np.int64)
    nb = L if bins is None else len(bins)
    out = np.empty((len(starts), nb), dtype=np.float32)
    k = np.arange(L)
    for a in range(0, len(starts), chunk):
        st = starts[a:a + chunk]
        idx = st[:, None] + k[None, :]
        ok = (idx >= 0) & (idx < len(x))
        fr = np.where(ok, x[np.clip(idx, 0, max(len(x) - 1, 0))], 0) * w
        X = np.fft.fft(fr, axis=1)
        if bins is not None:
            X = X[:, bins]
        out[a:a + chunk] = X.real ** 2 + X.imag ** 2
    return out


def band_bins(fs: float, band_hz=BAND_HZ, centre_hz: float = FE_CENTER_HZ):
    """(fftfreq indices, audio Hz) of the 4 s frame's bins inside band_hz, ascending."""
    L = _frame_len(fs)
    f = np.fft.fftfreq(L, 1.0 / fs) + centre_hz
    sel = np.flatnonzero((f >= band_hz[0]) & (f <= band_hz[1]))
    order = np.argsort(f[sel])
    return sel[order], f[sel][order]


def cfar_normalise(P: np.ndarray, local_bins: int | None = CFAR_LOCAL_BINS) -> np.ndarray:
    """A noise bin becomes Exp(1): each frame divided by its median/ln 2
    (which takes out any gain that changes with time), then each bin by
    the local spectral shape (`spectral_shape`), which takes out any
    that changes with frequency: a receiver passband that is not flat,
    a filter's band edge. Without the second step a tilt of b in a
    channel's normalised mean adds b sqrt(n_frames) to A3's Z, about 30b
    on FULL against a Gumbel scale of 0.63. `local_bins` None skips it.
    A frame with zero median (no data) becomes all zero."""
    med = np.median(P, axis=1, keepdims=True) / LN2
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(med > 0, P / med, 0.0)
    if local_bins:
        out = out / spectral_shape(out, local_bins, rows=med[:, 0] > 0)
    return out.astype(np.float32)


def spectral_shape(Pn: np.ndarray, local_bins: int = CFAR_LOCAL_BINS, rows=None) -> np.ndarray:
    """float[n_bins]: the local noise level of frame-normalised power Pn
    (n_frames, n_bins), 1 where the spectrum is flat.

    Per bin the median over frames (`rows`, default all) divided by ln 2,
    then the median of that over `local_bins` bins about it (reflected at
    the ends). The time median ignores anything present in fewer than half
    the frames and the frequency median anything narrower than half the
    window, so a carrier, a drifting line or a lead-in does not lower its
    own emission; a strong signal's data sidebands do raise the level
    about it, which is what a local CFAR is for.
    """
    Pn = np.asarray(Pn)
    if rows is not None:
        Pn = Pn[np.asarray(rows, dtype=bool)]
    if len(Pn) == 0:
        return np.ones(np.shape(Pn)[1])
    lvl = np.median(Pn, axis=0).astype(np.float64) / LN2
    w = min(int(local_bins) | 1, 2 * (len(lvl) // 2) - 1) if len(lvl) > 2 else 1
    sh = ndimage.median_filter(lvl, size=w, mode="reflect")
    return np.where(sh > 0, sh, 1.0)


def _parabola(ym, y0, yp) -> float:
    """Vertex offset in (-1/2, 1/2) of a parabola through (-1, 0, +1)."""
    den = ym - 2 * y0 + yp
    if not np.isfinite(den) or den >= 0:
        return 0.0
    return float(np.clip(0.5 * (ym - yp) / den, -0.5, 0.5))


# --- A1 ---------------------------------------------------------------------------------------

@dataclass
class CarrierStats:
    """A1's statistic: stat[d, b] sums M normalised frames along drift d
    through bin b, b the bin of the frequency *at t0*."""
    stat: np.ndarray                     # float32[n_drift, n_bin]
    f_hz: np.ndarray                     # audio Hz of each bin, at t0
    drifts: np.ndarray                   # Hz/min
    M: int


def carrier_line_stats(fe, t0_index: float, fs: float = FE_FS, *, span_s=A1_SPAN_S,
                       drifts=A1_DRIFTS, band_hz=BAND_HZ) -> CarrierStats:
    """The A1 statistic over the band, every drift hypothesis."""
    L = _frame_len(fs)
    hop = STFT_HOP_S
    M = int((span_s[1] - span_s[0] - STFT_FRAME_S) // hop) + 1
    t_start = span_s[0] + hop * np.arange(M)              # frame starts, s after t0
    t_mid = t_start + STFT_FRAME_S / 2
    df = fs / L
    drifts = np.asarray(drifts, dtype=np.float64)
    # widen the bin range by `pad` each side so drifted paths stay in range
    pad = int(math.ceil(np.max(np.abs(drifts)) / 60.0 * np.max(np.abs(t_mid)) / df)) + 2
    sel, f = band_bins(fs, band_hz)
    lo = int(round((f[0] - FE_CENTER_HZ) / df)) - pad
    hi = int(round((f[-1] - FE_CENTER_HZ) / df)) + pad
    wide = np.arange(lo, hi + 1) % L
    P = spectrogram(fe, fs, t0_index + t_start * fs, wide)
    inner = slice(pad, pad + len(sel))
    med = np.median(P[:, inner], axis=1, keepdims=True) / LN2
    with np.errstate(divide="ignore", invalid="ignore"):
        E = np.where(med > 0, P / med, 0.0)
    E = (E / spectral_shape(E, rows=med[:, 0] > 0)).astype(np.float32)
    stat = np.zeros((len(drifts), len(sel)), dtype=np.float32)
    for i, d in enumerate(drifts):
        shift = np.round(d / 60.0 * t_mid / df).astype(int)
        for k in range(M):
            stat[i] += E[k, pad + shift[k]: pad + shift[k] + len(sel)]
    return CarrierStats(stat=stat, f_hz=f, drifts=drifts, M=M)


def a1_gamma(M: int, fs: float = FE_FS) -> tuple[float, float]:
    """(shape, scale) of the A1 statistic on noise.

    Each normalised bin is Exp(1), so a sum of M frames would be Gamma(M)
    if the frames were independent. At a 50% hop two Hann frames share
    power correlation rho = (sum w(n) w(n+H) / sum w^2)^2 = 1/36, so the
    sum's variance is M + 2(M-1) rho; the Gamma with that mean and
    variance is what the noise follows (measured, A-6). Plain Gamma(M)
    puts 2x too few cells over a 1e-4 threshold.
    """
    w = frame_window(fs)
    H = int(round(STFT_HOP_S * fs))
    rho = (np.sum(w[:-H] * w[H:]) / np.sum(w * w)) ** 2 if H < len(w) else 0.0
    var = M + 2 * (M - 1) * rho
    return M * M / var, var / M


def a1_threshold(M: int, pfa: float = A1_PFA, fs: float = FE_FS) -> float:
    """Per-cell threshold on the A1 statistic at false-alarm probability pfa."""
    k, th = a1_gamma(M, fs)
    return float(special.gammainccinv(k, pfa) * th)


def carrier_lines(fe, t0_index: float, fs: float = FE_FS, *, pfa: float = A1_PFA,
                  max_cands: int = A1_MAX_CANDS, min_sep_hz: float = A1_MIN_SEP_HZ,
                  band_hz=BAND_HZ) -> list[CarrierCandidate]:
    """A1: up to `max_cands` carrier lines above the Gamma CFAR threshold.

    Local maxima at least `min_sep_hz` apart, strongest first. f_hz is
    the line's audio frequency at t0, refined between bins by a
    parabola on the log statistic; `metric` is the A1 statistic (Gamma(M)
    on noise, `a1_gamma`).
    """
    cs = carrier_line_stats(fe, t0_index, fs, band_hz=band_hz)
    thr = a1_threshold(cs.M, pfa, fs)
    best_d = np.argmax(cs.stat, axis=0)
    best = cs.stat[best_d, np.arange(cs.stat.shape[1])]
    df = cs.f_hz[1] - cs.f_hz[0]
    order = np.argsort(best)[::-1]
    out: list[CarrierCandidate] = []
    picked: list[float] = []
    for b in order:
        v = float(best[b])
        if v <= thr or len(out) >= max_cands:
            break
        f0 = float(cs.f_hz[b])
        exc = v - cs.M
        if any(abs(f0 - f) < min_sep_hz for f in picked):
            continue
        d = best_d[b]
        d0 = int(np.argmin(np.abs(cs.drifts)))
        if v - float(cs.stat[d0, b]) < A1_DRIFT_MARGIN * exc:
            d = d0                       # a drifted line must beat the straight one clearly
        row = cs.stat[d]
        off = 0.0
        if 0 < b < len(row) - 1:
            off = _parabola(*np.log(np.maximum(row[b - 1:b + 2], 1e-9)))
        picked.append(f0)
        out.append(CarrierCandidate(f_hz=f0 + off * df,
                                    drift_hz_per_min=float(cs.drifts[d]), metric=v))
    return out


# --- A2 ----------------------------------------------------------------------------------------

def _notch_taps() -> np.ndarray:
    return signal.firwin(NOTCH_TAPS, NOTCH_HZ, fs=CH_FS, window="hann")


def _mf_taps(fs: float = CH_FS) -> np.ndarray:
    """Matched filter at rate fs: p(k/(fs T))/(fs T), |k| within the pulse span."""
    sps = fs * T_SYM
    k = np.arange(-int(8 * sps), int(8 * sps) + 1)
    return ce.pulse(k / sps) / sps


def a2_threshold(pfa: float = A2_PFA, n_chunks: int = N_PRE // A2_CHUNK,
                 n_scales: int = len(A2_SCALES)) -> float:
    """Lambda_k threshold at one scale with k = n_chunks groups: the
    Gamma(k) point at pfa / n_scales (`preamble_p`'s correction)."""
    return float(special.gammainccinv(n_chunks, pfa / n_scales))


@dataclass
class PreambleHit:
    """A2's best cell at one carrier candidate.

    f_hz: audio Hz at t0 (A1 frequency + df). tau0_s: receive time of the
    first preamble symbol minus nominal t0. phase: arg u at t0, relative
    to the channeliser's exact mixing at f_mix (drift chirp and df
    rotation both zero at t0). lam: the 2 s chunks' Lambda at the refined
    cell. cell_lam: Lambda_k of the grid cell the decision was made on,
    at the scale that cell's p came from (n_groups = k, Gamma(k) on
    noise); p_value: that cell's `preamble_p` (per cell, corrected for
    the scales tested but not for the search); passed = p_value < pfa.
    """
    f_hz: float
    drift_hz_per_min: float
    tau0_s: float
    phase: float
    lam: float
    cell_lam: float
    p_value: float
    passed: bool
    n_groups: int = N_PRE // A2_CHUNK
    f_mix: object = None                     # Fraction: channeliser mix, Hz inside FE
    tau0_sd_s: float = float("nan")
    chunk_sums: np.ndarray = field(default=None, repr=False)
    lead_in_s: float = 0.0                   # detected lead-in span (`lead_in_span`)

    def to_detection(self, duration_s: float = 1800.0) -> Detection:
        t = np.array([TBD_START_S, 0.0, duration_s])
        path = FreqPath(t_s=t, f_hz=self.f_hz + self.drift_hz_per_min * t / 60.0,
                        weight=np.ones(3))
        sd = self.tau0_sd_s * CH_FS if np.isfinite(self.tau0_sd_s) else 1.0
        timing = Timing(tau0=self.tau0_s * CH_FS, ppm=0.0, gamma=0.0, z=self.lam,
                        cov=np.diag([sd ** 2, 0.0]))
        return Detection(f_hz=self.f_hz, path=path, timing=timing, z_ref=float("nan"),
                         method="preamble", lead_in_s=self.lead_in_s)


@dataclass
class _A2Channel:
    """The notched, matched-filtered channel A2 searches."""
    m: np.ndarray                            # complex128 MF output of y_hp, at CH rate
    t0_ch: float                             # CH index of nominal t0 inside m
    sigma2: float                            # median |m|^2 / ln 2 over the search span
    f_mix: object
    local: np.ndarray = field(default=None, repr=False)   # max(sigma2, local mean |m|^2)


def preamble_channel(fe, t0_index: float, f_audio_hz: float, drift_hz_per_min: float = 0.0,
                     fs: int = FE_FS) -> _A2Channel:
    """Channelise FE about f_audio_hz (drift removed about t0), notch, MF."""
    decim = int(fs) // CH_FS
    lo_s = -A2_TAU_S - _A2_MARGIN_S
    hi_s = A2_TAU_S + N_PRE * T_SYM + _A2_MARGIN_S
    i_a = int(math.floor(t0_index + lo_s * fs))
    i_a -= i_a % decim                                       # CH grid shared with the stream
    i_b = int(math.ceil(t0_index + hi_s * fs))
    n = i_b - i_a
    fe = np.asarray(fe)
    seg = np.zeros(n, dtype=np.complex64)
    a, b = max(i_a, 0), min(i_b, len(fe))
    if b > a:
        seg[a - i_a:b - i_a] = fe[a:b]
    ch, f_mix = frontend.channelise(seg, fs, frontend.fe_hz(f_audio_hz), CH_FS, n0=i_a)
    t0_ch = (t0_index - i_a) / decim
    y = ch.astype(np.complex128)
    if drift_hz_per_min:
        t = (np.arange(len(y)) - t0_ch) / CH_FS
        y *= np.exp(-2j * np.pi * frontend.dsp.wrap_cycles(drift_hz_per_min / 120.0 * t * t))
    # residual offset of the mixing grid from the requested frequency
    resid = frontend.fe_hz(f_audio_hz) - float(f_mix)
    if resid:
        t = (np.arange(len(y)) - t0_ch) / CH_FS
        y *= np.exp(-2j * np.pi * frontend.dsp.wrap_cycles(resid * t))
    y_hp = y - signal.oaconvolve(y, _notch_taps(), mode="same")
    m = signal.oaconvolve(y_hp, _mf_taps(), mode="same")
    k0 = int(round(t0_ch - A2_TAU_S * CH_FS))
    k1 = int(round(t0_ch + (A2_TAU_S + N_PRE * T_SYM) * CH_FS))
    core = m[max(k0, 0):min(k1, len(m))]
    sigma2 = max(float(np.median(np.abs(core) ** 2) / LN2) if len(core) else 1.0, 1e-30)
    # Local noise level: a carrier that switches (a lead-in ending, a
    # neighbour keying) leaves a transient the notch cannot remove, which a
    # global sigma^2 would read as correlation. Normalising each chunk by
    # the local mean power, never below the global level, turns any such
    # deterministic leftover back into noise-like Lambda and costs a real
    # signal nothing (its chunks are coherent, the leftover is not).
    w = int(round(A2_LOCAL_S * CH_FS))
    loc = signal.oaconvolve(np.abs(m) ** 2, np.ones(w) / w, mode="same")
    return _A2Channel(m=m, t0_ch=t0_ch, sigma2=sigma2, f_mix=f_mix,
                      local=np.maximum(loc, sigma2))


def _templates(df_hz):
    """(b, W, norm): zero-mean chunk templates b (n_chunk, 66), the
    df-rotated weights W (n_df, n_chunk, 66) and sum b^2 per chunk."""
    a = preamble_ce().astype(np.float64)
    nc = len(a) // A2_CHUNK
    b = a[:nc * A2_CHUNK].reshape(nc, A2_CHUNK)
    b = b - b.mean(axis=1, keepdims=True)
    t = (np.arange(nc * A2_CHUNK) * T_SYM).reshape(nc, A2_CHUNK)
    df = np.asarray(df_hz, dtype=np.float64)
    W = b[None] * np.exp(-2j * np.pi * df[:, None, None] * t[None])
    return b, W, np.sum(b * b, axis=1)


def preamble_lambda(chan: _A2Channel, df_hz=A2_DF_HZ, tau_s: float = A2_TAU_S):
    """(Lambda[n_df, n_tau], tau offsets in CH samples from t0, S[n_df, n_tau, n_chunk]).

    Lambda is the 2 s chunks' statistic, Gamma(10) per cell on noise.
    """
    lam, taus, S, _var = _preamble_sums(chan, df_hz, tau_s)
    return lam, taus, S


def _preamble_sums(chan: _A2Channel, df_hz=A2_DF_HZ, tau_s: float = A2_TAU_S):
    """`preamble_lambda`, plus the noise variance of each chunk sum
    var[n_tau, n_chunk] = sigma_c^2 sum b_c^2."""
    b, W, norm = _templates(df_hz)
    nc = b.shape[0]
    off = np.round(np.arange(nc * A2_CHUNK) * T_SYM * CH_FS).astype(np.int64)
    k0 = int(round(chan.t0_ch))
    taus = np.arange(-int(round(tau_s * CH_FS)), int(round(tau_s * CH_FS)) + 1)
    m = chan.m
    idx = np.clip((k0 + taus)[:, None] + off[None, :], 0, len(m) - 1)   # in range by margin
    Mx = m[idx].reshape(len(taus), nc, A2_CHUNK)
    S = np.einsum("fcs,tcs->ftc", W, Mx, optimize=True)
    cidx = (k0 + taus)[:, None] + off.reshape(nc, A2_CHUNK).mean(axis=1).round().astype(
        np.int64)[None, :]
    var = chan.local[np.clip(cidx, 0, len(m) - 1)] * norm[None, :]  # (n_tau, n_chunk)
    lam = np.sum(np.abs(S) ** 2 / var[None], axis=2)
    return lam, taus + (k0 - chan.t0_ch), S, var


def preamble_scales(S: np.ndarray, var: np.ndarray, scales=A2_SCALES):
    """[(k, Lambda_k)]: the statistic with the 2 s chunk sums added
    coherently in runs of g (g in `scales`), k = n_chunk // g groups.

    The chunk templates are rotated by df from the preamble's start, so
    the chunk sums S_c of one cell share a phase reference and a run of
    them adds coherently; each group's sum has noise variance sum var_c,
    so Lambda_k = sum_groups |sum S_c|^2 / sum var_c is Gamma(k, 1) per
    cell on noise. Longer runs gain on a steady path (10 noncoherent
    chunks cost ~2.5 dB against one coherent sum) and lose to wander,
    so all of them are tested (`preamble_p`).

    Each chunk sum is put in units of its own noise sd before the runs
    are added, Lambda_k = sum_groups |sum_c S_c / sqrt(var_c)|^2 / g, which
    is the same Gamma(k) on noise and, the chunks' var_c being nearly
    equal on a real signal, the same statistic there. What it changes is
    a transient confined to one chunk (a lead-in ending, which the notch
    cannot remove): weighted by var_c / sum var it carried half its 2 s
    Lambda into a 20 s sum and passed; weighted 1/g it carries a tenth.
    """
    nc = S.shape[-1]
    Sn = S / np.sqrt(var)[None]
    out = []
    for g in scales:
        k = nc // g
        Sg = Sn[..., :k * g].reshape(Sn.shape[:-1] + (k, g)).sum(axis=-1)
        out.append((k, np.sum(np.abs(Sg) ** 2, axis=-1) / g))
    return out


def preamble_p(S: np.ndarray, var: np.ndarray, scales=A2_SCALES):
    """(p[n_df, n_tau], scale index[n_df, n_tau], [(k, Lambda_k)]): the
    per-cell false-alarm probability of the best scale, Bonferroni-
    corrected for the len(scales) scales tested (so it is <= pfa with
    probability <= pfa on noise)."""
    st = preamble_scales(S, var, scales)
    ps = np.stack([special.gammaincc(k, lam) for k, lam in st])
    best = np.argmin(ps, axis=0)
    p = np.minimum(np.take_along_axis(ps, best[None], axis=0)[0] * len(st), 1.0)
    return p, best, st


def _fine_sums(spline, chan, tau, df, rate=0.0):
    """Chunk sums S_c at fractional tau (CH samples from t0), exact symbol
    offsets, templates rotated by df + rate*t (t from t0, Hz and Hz/s)."""
    a = preamble_ce().astype(np.float64)
    nc = len(a) // A2_CHUNK
    b = a[:nc * A2_CHUNK].reshape(nc, A2_CHUNK)
    b = b - b.mean(axis=1, keepdims=True)
    t = np.arange(nc * A2_CHUNK) * T_SYM
    turns = df * t + 0.5 * rate * t * t
    W = b * np.exp(-2j * np.pi * turns).reshape(nc, A2_CHUNK)
    mv = spline(chan.t0_ch + tau + t * CH_FS).reshape(nc, A2_CHUNK)
    return np.sum(W * mv, axis=1), np.sum(b * b, axis=1)


def _coherent(spline, chan, tau, df, rate) -> float:
    """|sum_c S_c|^2 / (sigma^2 sum b^2): the whole preamble coherently, unit noise."""
    S, norm = _fine_sums(spline, chan, tau, df, rate)
    return float(np.abs(np.sum(S)) ** 2 / (chan.sigma2 * np.sum(norm)))


def _phase_line(S, dt) -> tuple[float, float, float]:
    """(a, b, sd_b): residual frequency a + b (t - tbar) from the chunk
    phases, fitted to the chunk-pair frequencies, weighted by |S_c S_c+1|."""
    pr = S[1:] * np.conj(S[:-1])
    g = np.angle(pr) / (2 * np.pi * dt)
    w = np.abs(pr)
    if w.sum() <= 0:
        return 0.0, 0.0, float("inf")
    t = (np.arange(len(g)) + 1.0) * dt
    tb = np.average(t, weights=w)
    a = float(np.average(g, weights=w))
    sxx = float(np.sum(w * (t - tb) ** 2))
    b = float(np.sum(w * (t - tb) * (g - a)) / sxx)
    res = g - a - b * (t - tb)
    dof = max(len(g) - 2, 1)
    s2 = float(np.sum(w * res ** 2) / np.sum(w)) * len(g) / dof
    sd_b = math.sqrt(s2 * np.sum(w) / sxx) if sxx > 0 else float("inf")
    return a - b * tb, b, sd_b



def preamble_search(fe, t0_index: float, cand: CarrierCandidate, fs: int = FE_FS, *,
                    pfa: float = A2_PFA) -> PreambleHit:
    """A2 at one A1 candidate: the best (tau0, df) cell, refined.

    Always returns the best cell, the one with the least `preamble_p`;
    `passed` says whether that p is under `pfa`. Refinement, after the grid
    parabolas: frequency (and, when significant, a residual drift) from
    a line through the chunk-pair phase differences, coherent across the
    20 s preamble; then timing on the coherent correlation of the whole
    preamble at fractional offsets.
    """
    chan = preamble_channel(fe, t0_index, cand.f_hz, cand.drift_hz_per_min, fs)
    lam2, taus, S, var = _preamble_sums(chan)
    pc, sc, st = preamble_p(S, var)
    if pc.min() > 1e-250:
        i, j = np.unravel_index(int(np.argmin(pc)), pc.shape)
    else:                                    # p underflows on a strong signal
        i, j = np.unravel_index(int(np.argmax(lam2)), lam2.shape)
    k_groups, lam = st[int(sc[i, j])]
    cell = float(lam[i, j])
    df_grid = np.asarray(A2_DF_HZ)
    ddf = df_grid[1] - df_grid[0]
    df = float(df_grid[i])
    if 0 < i < len(df_grid) - 1:
        df += _parabola(lam[i - 1, j], lam[i, j], lam[i + 1, j]) * ddf
    tau = float(taus[j])
    if 0 < j < len(taus) - 1:
        tau += _parabola(lam[i, j - 1], lam[i, j], lam[i, j + 1])

    # m is band-limited to the pulse's +-19 Hz at 250 Hz, so a cubic spline
    # through it is exact to far below the noise at fractional timing.
    lo = int(max(0, math.floor(chan.t0_ch + tau - 80)))
    hi = int(min(len(chan.m), math.ceil(chan.t0_ch + tau + N_PRE * T_SYM * CH_FS + 80)))
    spline = CubicSpline(np.arange(lo, hi), chan.m[lo:hi])
    dt = A2_CHUNK * T_SYM
    rate = 0.0
    for _ in range(2):
        Sc, _n = _fine_sums(spline, chan, tau, df, rate)
        a, b, sd_b = _phase_line(Sc, dt)
        if abs(b) > A2_DRIFT_SIGMAS * sd_b:
            df += a
            rate += b
        else:
            pr = Sc[1:] * np.conj(Sc[:-1])
            df += float(np.angle(np.sum(pr))) / (2 * np.pi * dt)
        for d in (0.25, 0.1, 0.05):
            ym, y0, yp = (_coherent(spline, chan, tau + e, df, rate) for e in (-d, 0.0, d))
            tau += _parabola(ym, y0, yp) * d
    Sc, norm = _fine_sums(spline, chan, tau, df, rate)
    cen = chan.t0_ch + tau + (np.arange(len(norm)) + 0.5) * A2_CHUNK * T_SYM * CH_FS
    sig = chan.local[np.clip(np.round(cen).astype(np.int64), 0, len(chan.local) - 1)]
    lam_f = float(np.sum(np.abs(Sc) ** 2 / (norm * sig)))
    # S ~ u * j K sum b a: arg u = arg sum S - pi/2 (template phases referred to t0)
    phase = float(np.angle(np.sum(Sc)) - np.pi / 2)
    # timing sd from the coherent statistic's curvature: var = 1/(-L'')
    y = [_coherent(spline, chan, tau + e, df, rate) for e in (-0.5, 0.0, 0.5)]
    curv = -(y[0] - 2 * y[1] + y[2]) / 0.25
    sd = math.sqrt(1.0 / curv) / CH_FS if curv > 0 else float("nan")
    p_cell = float(pc[i, j])
    return PreambleHit(
        f_hz=float(cand.f_hz + df),
        drift_hz_per_min=float(cand.drift_hz_per_min + 60.0 * rate),
        tau0_s=tau / CH_FS, phase=phase, lam=lam_f, cell_lam=cell,
        p_value=p_cell, passed=p_cell < pfa, n_groups=k_groups,
        f_mix=chan.f_mix, tau0_sd_s=sd, chunk_sums=Sc)


# --- lead-in ---------------------------------------------------------------------------------

LEAD_WIN_S = 1.0                         # energy-detector window
LEAD_Z = 4.0                             # per window: P(Z > 4 | noise) = e^-8


def lead_in_span(fe, t0_index: float, f_hz: float, drift_hz_per_min: float = 0.0,
                 tau0_s: float = 0.0, fs: int = FE_FS, *, z_min: float = LEAD_Z) -> float:
    """Seconds of lead-in carrier found before the preamble (design 6.3).

    The channel at f_hz (drift removed about t0), matched-filtered with no
    notch, is summed coherently over 1 s windows walking back from the
    keying start (t0 + tau0 - 8T on the receiver's clock); the span is the
    run of windows with Z = |sum| / sqrt(n sigma^2 / 2) > z_min (the sum's
    amplitude in units of its per-quadrature noise sd, so P(Z > 4) = e^-8
    per window on noise, and 1 s of lead-in reaches Z = 4 at about
    -25 dB SNR2500), up to
    LEAD_IN_MAX_S. sigma^2, the MF output's noise power, comes from
    `frontend.noise_psd` (the floor beside the carrier). 0.0 when the
    first window fails.
    """
    from .constants import LEAD_IN_MAX_S, SPAN
    end = tau0_s - SPAN * T_SYM                       # s after nominal t0
    lo_s = end - LEAD_IN_MAX_S - 8.0
    decim = int(fs) // CH_FS
    i_a = int(math.floor(t0_index + lo_s * fs))
    i_a -= i_a % decim
    i_b = int(math.ceil(t0_index + (end + 1.0) * fs))
    fe = np.asarray(fe)
    seg = np.zeros(i_b - i_a, dtype=np.complex64)
    a, b = max(i_a, 0), min(i_b, len(fe))
    if b > a:
        seg[a - i_a:b - i_a] = fe[a:b]
    ch, f_mix = frontend.channelise(seg, fs, frontend.fe_hz(f_hz), CH_FS, n0=i_a)
    t = (np.arange(len(ch)) * decim + i_a - t0_index) / fs          # s after nominal t0
    resid = frontend.fe_hz(f_hz) - float(f_mix)
    turns = resid * t + drift_hz_per_min / 120.0 * t * t
    m = signal.oaconvolve(ch.astype(np.complex128) * np.exp(
        -2j * np.pi * frontend.dsp.wrap_cycles(turns)), _mf_taps(), mode="same")
    # noise at the MF output from the floor 85-120 Hz off the carrier
    # (PSD x the MF's noise bandwidth 1/T): the carrier, lead-in or not,
    # is in every sample near it
    sps = CH_FS * T_SYM
    have = np.abs(ch) > 0
    psd = frontend.noise_psd(ch[have], CH_FS, fs_in=fs) if have.sum() > 2 * CH_FS else [1.0]
    sigma2 = max(float(np.median(psd)) / T_SYM, 1e-30)
    span = 0.0
    for k in range(int(round(LEAD_IN_MAX_S / LEAD_WIN_S))):
        w = (t >= end - (k + 1) * LEAD_WIN_S) & (t < end - k * LEAD_WIN_S)
        n_ind = w.sum() / sps                                       # independent MF samples
        z = abs(np.sum(m[w])) / sps / math.sqrt(max(n_ind, 1) * sigma2 / 2)
        if z <= z_min:
            break
        span = (k + 1) * LEAD_WIN_S
    return span


# --- A3 -----------------------------------------------------------------------------------------

def tbd_frame_starts(spec: FrameSpec, n_fe: int | None = None, t0_index: float = 0.0,
                     fs: float = FE_FS) -> np.ndarray:
    """Frame start times (s after t0) covering [t0 - 10 s, end of keying + 2 s],
    limited to frames lying wholly inside a capture of n_fe samples."""
    end = spec.keyed_end_pos * T_SYM + TBD_END_PAD_S
    t = TBD_START_S + STFT_HOP_S * np.arange(
        int((end - TBD_START_S - STFT_FRAME_S) // STFT_HOP_S) + 1)
    if n_fe is not None:
        i = t0_index + t * fs
        t = t[(i >= 0) & (i + _frame_len(fs) <= n_fe)]
    return t


def tbd_n_frames(spec: FrameSpec) -> int:
    return len(tbd_frame_starts(spec))


def tbd_viterbi(E: np.ndarray, max_step: int = TBD_MAX_STEP, cost: float = TBD_COST):
    """Best path through emissions E (..., n_frames, n_bins).

    Maximises sum E[t, b_t] - cost * sum |b_t - b_{t-1}| with
    |b_t - b_{t-1}| <= max_step, free start and end. Returns (path bins
    (..., n_frames) int, Z = sum of emissions on the path / sqrt(n_frames)).
    """
    E = np.asarray(E, dtype=np.float64)
    lead, (T, B) = E.shape[:-2], E.shape[-2:]
    score = E[..., 0, :].copy()
    back = np.zeros(lead + (T, B), dtype=np.int8)
    steps = range(-max_step, max_step + 1)
    for t in range(1, T):
        best = np.full(lead + (B,), -np.inf)
        arg = np.zeros(lead + (B,), dtype=np.int8)
        for d in steps:                       # arrive at b from b - d
            cand = np.full(lead + (B,), -np.inf)
            if d >= 0:
                cand[..., d:] = score[..., :B - d] - cost * d
            else:
                cand[..., :B + d] = score[..., -d:] - cost * (-d)
            upd = cand > best
            best = np.where(upd, cand, best)
            arg = np.where(upd, np.int8(d), arg)
        back[..., t, :] = arg
        score = best + E[..., t, :]
    path = np.zeros(lead + (T,), dtype=np.int64)
    path[..., -1] = np.argmax(score, axis=-1)
    for t in range(T - 1, 0, -1):
        d = np.take_along_axis(back[..., t, :], path[..., t:t + 1], axis=-1)[..., 0]
        path[..., t - 1] = path[..., t] - d
    em = np.take_along_axis(E, path[..., None], axis=-1)[..., 0]
    return path, em.sum(axis=-1) / math.sqrt(T)


def tbd_gumbel(n_frames: int) -> tuple[float, float]:
    """(mu, beta) of Z on noise for n_frames, from the committed table."""
    r = np.sqrt(np.asarray(TBD_GUMBEL_N, dtype=np.float64))
    x = math.sqrt(n_frames)
    mu = np.asarray(TBD_GUMBEL_MU, dtype=np.float64)
    be = np.asarray(TBD_GUMBEL_BETA, dtype=np.float64)
    if x <= r[0]:
        i = 0
    elif x >= r[-1]:
        i = len(r) - 2
    else:
        i = int(np.searchsorted(r, x)) - 1
    w = (x - r[i]) / (r[i + 1] - r[i])          # linear inside, extrapolated outside
    return float(mu[i] + w * (mu[i + 1] - mu[i])), float(be[i] + w * (be[i + 1] - be[i]))


def gumbel_p(z: float, n_frames: int) -> float:
    """P(Z >= z) on noise: the Gumbel survival function."""
    mu, beta = tbd_gumbel(n_frames)
    return float(-np.expm1(-np.exp(-(z - mu) / beta)))


@dataclass
class TbdHit:
    """A3's best path in one channel: f per frame (audio Hz), Z and its p."""
    path: FreqPath
    z: float
    p_value: float
    n_frames: int
    centre_hz: float                         # the channel searched

    @property
    def f_hz(self) -> float:
        """Frequency at t0 from a straight line through the path.

        Weighted by the emissions, with one pass dropping points more than
        1 Hz off the line: before the signal keys (and in deep fades) the
        path wanders on noise and carries ~no weight.
        """
        return float(np.polyval(self.line(), 0.0))

    def line(self) -> np.ndarray:
        """(slope Hz/s, intercept Hz at t0) of the weighted line fit."""
        t, f = self.path.t_s, self.path.f_hz
        w = np.asarray(self.path.weight, dtype=np.float64) + 1e-9
        keep = np.ones(len(t), dtype=bool)
        for _ in range(2):
            if keep.sum() < 2:
                break
            c = np.polyfit(t[keep], f[keep], 1, w=np.sqrt(w[keep]))
            keep = np.abs(f - np.polyval(c, t)) < 1.0
        return c

    @property
    def passed(self) -> bool:
        return self.p_value < TBD_PFA

    def to_detection(self) -> Detection:
        """A detection with no timing yet (V fits it, design 6.4)."""
        return Detection(f_hz=self.f_hz, path=self.path, timing=None, z_ref=float("nan"),
                         method="tbd", lead_in_s=0.0)


def tbd_channels(cands=(), band_hz=BAND_HZ) -> np.ndarray:
    """Channel centres: every A1 candidate plus a 50 Hz grid, deduplicated
    (a grid centre within 25 Hz of a candidate is dropped), all kept
    +-40 Hz inside the band."""
    lo, hi = band_hz[0] + TBD_HALF_HZ, band_hz[1] - TBD_HALF_HZ
    cf = [min(max(float(c.f_hz), lo), hi) for c in cands]
    grid = np.arange(math.ceil(lo / TBD_GRID_HZ) * TBD_GRID_HZ, hi + 1e-9, TBD_GRID_HZ)
    grid = [g for g in grid if all(abs(g - c) >= TBD_GRID_HZ / 2 for c in cf)]
    return np.array(sorted(cf + list(grid)))


def tbd_emissions(fe, t0_index: float, spec: FrameSpec, fs: float = FE_FS, *,
                  band_hz=BAND_HZ):
    """(E[n_frames, n_band_bins] = normalised power - 1, band audio Hz, frame starts s)."""
    t = tbd_frame_starts(spec, len(fe), t0_index, fs)
    sel, f = band_bins(fs, band_hz)
    P = spectrogram(fe, fs, t0_index + t * fs, sel)
    return cfar_normalise(P) - np.float32(1.0), f, t


def track_before_detect(fe, t0_index: float, spec: FrameSpec, cands=(), fs: float = FE_FS,
                        *, pfa: float = TBD_PFA, band_hz=BAND_HZ,
                        all_hits: bool = False) -> list[TbdHit]:
    """A3 over a stored slot: one Viterbi path per channel; hits at p < pfa
    (all channels with `all_hits`), deduplicated within 10 Hz, best first."""
    E, f, t = tbd_emissions(fe, t0_index, spec, fs, band_hz=band_hz)
    if len(t) < 2:
        return []
    df = f[1] - f[0]
    half = int(round(TBD_HALF_HZ / df))
    centres = tbd_channels(cands, band_hz)
    ci = np.clip(np.round((centres - f[0]) / df).astype(int), half, len(f) - 1 - half)
    stack = np.stack([E[:, c - half:c + half + 1] for c in ci])     # (n_ch, T, 2h+1)
    paths, Z = tbd_viterbi(stack)
    n = len(t)
    hits = []
    for k, c in enumerate(ci):
        bins = c - half + paths[k]
        em = E[np.arange(n), bins]
        # sub-bin refinement per frame where the emission stands out
        fk = f[bins].astype(np.float64)
        inner = (bins > 0) & (bins < len(f) - 1)
        rows = np.arange(n)[inner]
        bb = bins[inner]
        y = np.log(np.maximum(E[rows[:, None], bb[:, None] + np.array([-1, 0, 1])] + 1, 1e-6))
        den = y[:, 0] - 2 * y[:, 1] + y[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            off = np.where(den < 0, np.clip(0.5 * (y[:, 0] - y[:, 2]) / den, -0.5, 0.5), 0.0)
        fk[inner] += off * df
        p = gumbel_p(float(Z[k]), n)
        hits.append(TbdHit(path=FreqPath(t_s=t + STFT_FRAME_S / 2, f_hz=fk,
                                         weight=np.maximum(em, 0.0).astype(np.float64)),
                           z=float(Z[k]), p_value=p, n_frames=n,
                           centre_hz=float(centres[k])))
    hits.sort(key=lambda h: -h.z)
    out: list[TbdHit] = []
    for h in hits:
        if not all_hits and not h.p_value < pfa:
            continue
        if any(abs(h.f_hz - o.f_hz) < TBD_DEDUP_HZ for o in out):
            continue
        out.append(h)
    return out


# --- everything at once ---------------------------------------------------------------------

@dataclass
class Acquisition:
    lines: list                              # A1 CarrierCandidates
    preamble: list                           # A2 PreambleHits that passed
    tbd: list                                # A3 TbdHits that passed

    def detections(self, spec: FrameSpec) -> list[Detection]:
        """Every candidate for the verification gate, preamble hits first."""
        dur = spec.keyed_end_pos * T_SYM
        return ([h.to_detection(dur) for h in self.preamble]
                + [h.to_detection() for h in self.tbd])


def drop_symbol_rate_aliases(hits: list) -> list:
    """Passed A2 hits without those that are a stronger hit seen at
    f +- k/T (k in A2_ALIAS_K, within A2_ALIAS_HZ), with its timing
    (within T/2) and a 2 s Lambda A2_ALIAS_RATIO x smaller. Strongest first.

    At an offset of k/T the preamble's symbol rotation is whole turns, so
    a CE signal correlates there as well as at its own frequency, down only
    by the matched filter's response (Lambda ~70 against ~590 for a
    signal at 0 dB, measured). A real neighbour there would have to share
    the timing to within 15 ms as well to be lost; A3 still offers it.
    """
    out: list = []
    for h in sorted(hits, key=lambda h: -h.lam):
        if any(abs(abs(h.f_hz - g.f_hz) - k / T_SYM) < A2_ALIAS_HZ
               and abs(h.tau0_s - g.tau0_s) < T_SYM / 2
               and g.lam > A2_ALIAS_RATIO * h.lam
               for g in out for k in A2_ALIAS_K):
            continue
        out.append(h)
    return out


def acquire(cap: frontend.Capture, spec: FrameSpec, *, live_only: bool = False,
            preprocessed: bool = False) -> Acquisition:
    """A1 -> A2 at each line (and A3 over the slot unless live_only).

    `cap.fe` is blanked and normalised first unless `preprocessed`.
    """
    fe = cap.fe
    if not preprocessed:
        fe, keep = frontend.blank(fe, fs=cap.fs)
        fe, _ = frontend.normalise(fe, keep, fs=cap.fs)
    lines = carrier_lines(fe, cap.t0_index, cap.fs)
    pre = drop_symbol_rate_aliases(
        [h for h in (preamble_search(fe, cap.t0_index, c, cap.fs) for c in lines) if h.passed])
    for h in pre:
        h.lead_in_s = lead_in_span(fe, cap.t0_index, h.f_hz, h.drift_hz_per_min, h.tau0_s,
                                   cap.fs)
    tbd = [] if live_only else track_before_detect(fe, cap.t0_index, spec, lines, cap.fs)
    return Acquisition(lines=lines, preamble=pre, tbd=tbd)

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

**Waveform L** (`acquire` with a tuple of frames, at most one per
waveform: the joint listener). L's named frequency, its A1 line and
everything reported for it is the pilot; its data band lies 3.10-22.07
Hz below. A2 is per waveform (`PreambleKind`: sequence, symbol, chunk,
where the symbols sit and the phase convention), and one channelisation
and notch at a line (`_a2_input`) serves both: A2-L mixes the data centre
up to 0 Hz (f_p = 12207/970 Hz, from nominal t0) and matched-filters with
L's pulse, then correlates with the QPSK preamble in ten 33-symbol (2 s)
chunks, conj(b) templates, so the same Gamma(10) statistic, scales and
threshold apply. The pilot is notched out with the carrier's notch and
what is left of it lands 12.58 Hz above the data centre, outside L's
+-9.48 Hz filter. Given the pilot line A2-L passes 8/8 SHORT preambles at
-31 dB (Lambda ~43, timing rms 0.08 T_L), but during the preamble the
pilot is only 10.5% of L's power, so A1 sees it to about -26 dB (design
8.1): A2-L also runs at a pilot hypothesis above each A1 line
(`pilot_lines`), where a data-band lump took A1's line. Arbitration: the
symbol-rate aliases per waveform at its own T, then L's band ownership
(`drop_band_owned`, the -1/T_L alias inside its own data band), then one
hit per signal across waveforms (`merge_hits`). A3 runs once, its
emission still the line's (pilot-only for L, so L reaches A3's floor
~6 dB before CE); each hit carries a spectral-shape test (`shape_z`:
L's data are one-sided below its pilot, CE's sidebands symmetric) and,
gated as L, a path re-anchored on the pilot (`l_pilot_bins`), the shape
along which fills in where the hit's own is ambiguous.
`Acquisition.detections` orders the candidates for the gate, each
carrying its waveform.

**Timing convention.** A detection's `Timing.tau0` is in CH samples
(1/250 s) counted from the nominal t0 on the receiver's clock: the
preamble's first symbol was received tau0/250 s after nominal t0.
"""

from __future__ import annotations

import dataclasses
import functools
import math
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage, signal, special
from scipy.interpolate import CubicSpline

from . import ce, frontend, lwave
from .constants import (
    CARRIER_BAND_HZ,
    CH_FS,
    FE_CENTER_HZ,
    FE_FS,
    L_EDGE_HZ,
    L_N_PRE,
    L_PILOT_HZ,
    L_T_SYM,
    N_PRE,
    SPAN,
    T_SYM,
    WAVEFORM_CE,
    WAVEFORM_L,
    WAVEFORM_NAMES,
)
from .frame import FrameSpec
from .sequences import preamble_ce, preamble_l
from .types import CarrierCandidate, Detection, FreqPath, Timing

# --- shared spectrogram ------------------------------------------------------------------

STFT_FRAME_S = 4.0                       # 0.25 Hz bins
STFT_HOP_S = 2.0
BAND_HZ = CARRIER_BAND_HZ                # audio band searched (and the CFAR median's)
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
# Waveform L (the joint acquisition, `acquire` with a tuple of frames).
L_BAND_HZ = (float(L_PILOT_HZ - L_EDGE_HZ), float(L_PILOT_HZ + L_EDGE_HZ))   # 3.10..22.07 Hz:
                                         # L's data band, below its pilot
L_OWN_HZ = (3.1, 22.1)                   # an L hit this far below a stronger one, same timing, is
                                         # it; a pilot may hide this far above an A1 line
L_OWN_T = 3.0                            # "same timing" for that, in L symbols (the -1/T_L alias's
                                         # timing peak is broad: measured -1.9..+0.8 T_L off)
L_HINT_OFFSETS_HZ = (0.0, -1.0, 1.0)     # an L hint A1 cannot see: A2-L at f and f +- 1 Hz
WAVEFORM_MERGE_HZ = 5.0                  # a CE and an L hit this close, same timing: one signal
                                         # (receiver.MERGE_HZ)

# --- A3 --------------------------------------------------------------------------------------

TBD_START_S = -10.0                      # first frame starts here, so a lead-in contributes
HINT_SPAN_HZ = 2.0                       # a hint's line: A1's best bin this close to it (one
                                         # station's two passes were 0.6 Hz apart, 2026-10-11)
TBD_END_PAD_S = 2.0                      # frames run to the end of keying plus this
TBD_HALF_HZ = 40.0                       # channel half-width
TBD_GRID_HZ = 50.0                       # channel grid across the band
TBD_MAX_STEP = 2                         # bins per hop (0.5 Hz / 2 s = 15 Hz/min)
TBD_COST = 0.5                           # per bin of step, in noise-emission sd
TBD_PFA = 1e-3
TBD_DEDUP_HZ = 10.0
# The spectral-shape test on an A3 path (`shape_z`, waveform-aware acquisition):
SHAPE_L_Z = 4.0                          # z1 (below minus above, N(0,1) on noise) over this: L
SHAPE_CE_ABOVE_Z = 3.0                   # CE: power above the line this significant,
SHAPE_CE_MAX_Z1 = 3.0                    # and the two sides this alike
L_ANCHOR_HZ = (-6.0, 26.0)               # an L pilot looked for this far about an A3 path's line

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
                  band_hz=BAND_HZ, stats: CarrierStats | None = None) -> list[CarrierCandidate]:
    """A1: up to `max_cands` carrier lines above the Gamma CFAR threshold.

    Local maxima at least `min_sep_hz` apart, strongest first. f_hz is
    the line's audio frequency at t0, refined between bins by a
    parabola on the log statistic; `metric` is the A1 statistic (Gamma(M)
    on noise, `a1_gamma`). `stats`: `carrier_line_stats` already computed.
    """
    cs = carrier_line_stats(fe, t0_index, fs, band_hz=band_hz) if stats is None else stats
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


def hint_line(cs: CarrierStats, f_hz: float, fs: float = FE_FS, *, pfa: float = A1_PFA,
              span_hz: float = HINT_SPAN_HZ) -> CarrierCandidate:
    """The carrier line for a hint, a frequency where a signal is expected
    (the next pass of a picture already partly received): A1's best (bin,
    drift) within span_hz of f_hz when it clears the threshold, refined as
    `carrier_lines` does, else f_hz itself, straight."""
    near = np.flatnonzero(np.abs(cs.f_hz - f_hz) <= span_hz)
    if not len(near):
        return CarrierCandidate(f_hz=float(f_hz), drift_hz_per_min=0.0, metric=0.0)
    sub = cs.stat[:, near]
    d, k = np.unravel_index(int(np.argmax(sub)), sub.shape)
    b, v = int(near[k]), float(sub[d, k])
    if v <= a1_threshold(cs.M, pfa, fs):
        return CarrierCandidate(f_hz=float(f_hz), drift_hz_per_min=0.0, metric=v)
    return _cell_line(cs, b, d, v)


def _cell_line(cs: CarrierStats, b: int, d: int, v: float) -> CarrierCandidate:
    """A1's cell (bin b, drift d, statistic v) as a line, refined as
    `carrier_lines` does: straight unless the drift adds 10%, then a
    parabola on the log statistic between bins."""
    d0 = int(np.argmin(np.abs(cs.drifts)))
    if v - float(cs.stat[d0, b]) < A1_DRIFT_MARGIN * (v - cs.M):
        d = d0
    row = cs.stat[d]
    off = 0.0
    if 0 < b < len(row) - 1:
        off = _parabola(*np.log(np.maximum(row[b - 1:b + 2], 1e-9)))
    df = cs.f_hz[1] - cs.f_hz[0]
    return CarrierCandidate(f_hz=float(cs.f_hz[b]) + off * df,
                            drift_hz_per_min=float(cs.drifts[d]), metric=v)


def pilot_lines(cs: CarrierStats, lines, *, sep_hz: float = A2_ALIAS_HZ) -> list[CarrierCandidate]:
    """L pilot hypotheses beside A1's lines: for each line f, A1's best
    (bin, drift) cell in (f + 3.1, f + 22.1] Hz, refined as a line.

    During the preamble an L pilot carries 10.5% of the power and its data
    band's 'lumps' (the fixed preamble's periodogram) sit 10-20 Hz below
    it, so a lump can take A1's line and, by the 10 Hz separation, hide
    the pilot above it. The cell need not clear A1's threshold (A2-L
    judges it). A hypothesis within sep_hz of a line or of an earlier
    hypothesis is dropped: A2's df search (+-0.5 Hz) covers it already.
    """
    lo_hz, hi_hz = L_OWN_HZ
    taken = [float(c.f_hz) for c in lines]
    out: list[CarrierCandidate] = []
    for c in lines:
        sel = np.flatnonzero((cs.f_hz > c.f_hz + lo_hz) & (cs.f_hz <= c.f_hz + hi_hz))
        if not len(sel):
            continue
        sub = cs.stat[:, sel]
        d, k = np.unravel_index(int(np.argmax(sub)), sub.shape)
        h = _cell_line(cs, int(sel[k]), int(d), float(sub[d, k]))
        if any(abs(h.f_hz - f) <= sep_hz for f in taken):
            continue
        taken.append(h.f_hz)
        out.append(h)
    return out


# --- A2 ----------------------------------------------------------------------------------------

def _notch_taps() -> np.ndarray:
    return signal.firwin(NOTCH_TAPS, NOTCH_HZ, fs=CH_FS, window="hann")


def _mf_taps(fs: float = CH_FS) -> np.ndarray:
    """Matched filter at rate fs: p(k/(fs T))/(fs T), |k| within the pulse span."""
    sps = fs * T_SYM
    k = np.arange(-int(8 * sps), int(8 * sps) + 1)
    return ce.pulse(k / sps) / sps


def _l_mf_taps(fs: float = CH_FS) -> np.ndarray:
    """L's matched filter at rate fs: L's unit-energy RRC (`lwave.PULSE_NORM`
    on its 16 kHz grid) at sps = fs T_L, cut at +-8 symbols, over sps."""
    sps = fs * L_T_SYM
    k = np.arange(-int(SPAN * sps), int(SPAN * sps) + 1)
    x = k / sps
    return np.where(np.abs(x) <= SPAN, ce._rrc(x) / lwave.PULSE_NORM, 0.0) / sps


@dataclass(frozen=True, eq=False)
class PreambleKind:
    """What A2 needs to know about one waveform's preamble.

    seq: the preamble symbols (CE real +-1, L complex QPSK); t_sym its
    symbol period; chunk symbols per 2 s chunk (10 chunks for both, so
    `a2_threshold` and `preamble_p` carry over); data_offset_hz where the
    preamble's symbols sit relative to the A1 line (0 for CE, whose line
    is the carrier at the centre; -f_p for L, whose line is the pilot);
    phase_offset added to arg sum S to give arg u (CE's linear part is
    j K x: -pi/2); mf_taps the symbol matched filter at CH rate.
    """
    waveform: int
    seq: np.ndarray = field(repr=False)
    t_sym: float
    chunk: int
    data_offset_hz: float
    phase_offset: float
    mf_taps: np.ndarray = field(repr=False)

    @property
    def n_chunks(self) -> int:
        return len(self.seq) // self.chunk

    @property
    def name(self) -> str:
        return WAVEFORM_NAMES[self.waveform]


CE_PRE = PreambleKind(WAVEFORM_CE, preamble_ce().astype(np.float64), T_SYM, A2_CHUNK, 0.0,
                      -np.pi / 2, _mf_taps())
L_PRE = PreambleKind(WAVEFORM_L, np.asarray(preamble_l()), L_T_SYM, A2_CHUNK // 2,
                     -float(L_PILOT_HZ), 0.0, _l_mf_taps())
PREAMBLE_KINDS = {WAVEFORM_CE: CE_PRE, WAVEFORM_L: L_PRE}
assert L_N_PRE == N_PRE // 2 and L_PRE.n_chunks == CE_PRE.n_chunks
for _k in (CE_PRE, L_PRE):
    _k.seq.setflags(write=False)
    _k.mf_taps.setflags(write=False)


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
    waveform: whose preamble (`PreambleKind`) was searched for; for L,
    f_hz is the pilot and phase is the pilot's at t0 (the f_p shift's and
    the pilot's own rotation over tau0 taken out). CE's phase leaves out
    the latter, 2 pi df tau0 (as it always has; nothing reads it but tests).
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
    waveform: int = WAVEFORM_CE

    def to_detection(self, duration_s: float = 1800.0) -> Detection:
        t = np.array([TBD_START_S, 0.0, duration_s])
        path = FreqPath(t_s=t, f_hz=self.f_hz + self.drift_hz_per_min * t / 60.0,
                        weight=np.ones(3))
        sd = self.tau0_sd_s * CH_FS if np.isfinite(self.tau0_sd_s) else 1.0
        timing = Timing(tau0=self.tau0_s * CH_FS, ppm=0.0, gamma=0.0, z=self.lam,
                        cov=np.diag([sd ** 2, 0.0]))
        return Detection(f_hz=self.f_hz, path=path, timing=timing, z_ref=float("nan"),
                         method="preamble", lead_in_s=self.lead_in_s, waveform=self.waveform)


@dataclass
class _A2Input:
    """The channel at one line, drift and mixing residual removed, notched:
    what every waveform's A2 at that line shares (`_a2_mf` finishes it)."""
    y: np.ndarray                            # complex128 at CH rate, 0 Hz = the line
    t0_ch: float                             # CH index of nominal t0 inside y
    f_mix: object


@dataclass
class _A2Channel:
    """The notched, matched-filtered channel A2 searches."""
    m: np.ndarray                            # complex128 MF output of y_hp, at CH rate
    t0_ch: float                             # CH index of nominal t0 inside m
    sigma2: float                            # median |m|^2 / ln 2 over the search span
    f_mix: object
    local: np.ndarray = field(default=None, repr=False)   # max(sigma2, local mean |m|^2)
    kind: PreambleKind = CE_PRE


def preamble_channel(fe, t0_index: float, f_audio_hz: float, drift_hz_per_min: float = 0.0,
                     fs: int = FE_FS, *, kind: PreambleKind = CE_PRE) -> _A2Channel:
    """Channelise FE about f_audio_hz (drift removed about t0), notch, MF
    (`_a2_input` then `_a2_mf` for `kind`)."""
    return _a2_mf(_a2_input(fe, t0_index, f_audio_hz, drift_hz_per_min, fs), kind)


def _a2_input(fe, t0_index: float, f_audio_hz: float, drift_hz_per_min: float = 0.0,
              fs: int = FE_FS) -> _A2Input:
    """Channelise FE about f_audio_hz, drift chirp and mixing residual
    taken out about t0, then the notch y - LP(y) at 0 Hz: CE's carrier or
    L's pilot, either one's lead-in with it. The span is the preamble's,
    the same 20.006 s for both waveforms."""
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
    return _A2Input(y=y_hp, t0_ch=t0_ch, f_mix=f_mix)


def _a2_mf(inp: _A2Input, kind: PreambleKind = CE_PRE) -> _A2Channel:
    """`kind`'s matched filter on the notched channel, with A2's noise levels.

    For L the channel is first mixed up by f_p, exp(+j 2 pi f_p (t - t0))
    with t on the receiver's clock from nominal t0, so the data centre
    sits at 0 Hz: what is left of the pilot after the notch (a lead-in's
    ramp-down, a neighbour's carrier there) lands at +12.58 Hz, outside
    L's matched filter (+-9.48 Hz), which is L's main defence against it.
    """
    y_hp = inp.y
    t0_ch = inp.t0_ch
    if kind.data_offset_hz:
        t = (np.arange(len(y_hp)) - t0_ch) / CH_FS
        y_hp = y_hp * np.exp(2j * np.pi * frontend.dsp.wrap_cycles(-kind.data_offset_hz * t))
    m = signal.oaconvolve(y_hp, kind.mf_taps, mode="same")
    k0 = int(round(t0_ch - A2_TAU_S * CH_FS))
    k1 = int(round(t0_ch + (A2_TAU_S + len(kind.seq) * kind.t_sym) * CH_FS))
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
    return _A2Channel(m=m, t0_ch=t0_ch, sigma2=sigma2, f_mix=inp.f_mix,
                      local=np.maximum(loc, sigma2), kind=kind)


def _chunks(kind: PreambleKind) -> np.ndarray:
    """Zero-mean chunk templates b (n_chunk, chunk) of kind's preamble.

    Zero mean, so a constant at the MF output (CE: the carrier behind the
    notch; L: a carrier at its data centre, a neighbour's) correlates to
    nothing in any chunk. For L it costs ~0.15 dB (its QPSK chunks are not
    zero-mean by themselves) and is kept for that robustness.
    """
    a = kind.seq
    nc = len(a) // kind.chunk
    b = a[:nc * kind.chunk].reshape(nc, kind.chunk)
    return b - b.mean(axis=1, keepdims=True)


def _norm(b: np.ndarray) -> np.ndarray:
    """sum |b|^2 per chunk (b * b exactly for CE's real templates)."""
    return np.sum((b * np.conj(b)).real, axis=1)


def _templates(df_hz, kind: PreambleKind = CE_PRE):
    """(b, W, norm): zero-mean chunk templates b (n_chunk, chunk), the
    df-rotated weights W = conj(b) e^{-j 2 pi df t} (n_df, n_chunk, chunk),
    t from the preamble's first symbol, and sum |b|^2 per chunk."""
    b = _chunks(kind)
    nc, ck = b.shape
    t = (np.arange(nc * ck) * kind.t_sym).reshape(nc, ck)
    df = np.asarray(df_hz, dtype=np.float64)
    W = np.conj(b)[None] * np.exp(-2j * np.pi * df[:, None, None] * t[None])
    return b, W, _norm(b)


def preamble_lambda(chan: _A2Channel, df_hz=A2_DF_HZ, tau_s: float = A2_TAU_S):
    """(Lambda[n_df, n_tau], tau offsets in CH samples from t0, S[n_df, n_tau, n_chunk]).

    Lambda is the 2 s chunks' statistic, Gamma(10) per cell on noise.
    """
    lam, taus, S, _var = _preamble_sums(chan, df_hz, tau_s)
    return lam, taus, S


def _preamble_sums(chan: _A2Channel, df_hz=A2_DF_HZ, tau_s: float = A2_TAU_S):
    """`preamble_lambda`, plus the noise variance of each chunk sum
    var[n_tau, n_chunk] = sigma_c^2 sum |b_c|^2. The preamble is the
    channel's (`chan.kind`)."""
    kind = chan.kind
    b, W, norm = _templates(df_hz, kind)
    nc, ck = b.shape
    off = np.round(np.arange(nc * ck) * kind.t_sym * CH_FS).astype(np.int64)
    k0 = int(round(chan.t0_ch))
    taus = np.arange(-int(round(tau_s * CH_FS)), int(round(tau_s * CH_FS)) + 1)
    m = chan.m
    idx = np.clip((k0 + taus)[:, None] + off[None, :], 0, len(m) - 1)   # in range by margin
    Mx = m[idx].reshape(len(taus), nc, ck)
    S = np.einsum("fcs,tcs->ftc", W, Mx, optimize=True)
    cidx = (k0 + taus)[:, None] + off.reshape(nc, ck).mean(axis=1).round().astype(
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
    offsets, templates rotated by df + rate*t (t from t0, Hz and Hz/s),
    of the channel's preamble (`chan.kind`)."""
    kind = chan.kind
    b = _chunks(kind)
    nc, ck = b.shape
    t = np.arange(nc * ck) * kind.t_sym
    turns = df * t + 0.5 * rate * t * t
    W = np.conj(b) * np.exp(-2j * np.pi * turns).reshape(nc, ck)
    mv = spline(chan.t0_ch + tau + t * CH_FS).reshape(nc, ck)
    return np.sum(W * mv, axis=1), _norm(b)


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
                    pfa: float = A2_PFA, kind: PreambleKind = CE_PRE,
                    inp: _A2Input | None = None) -> PreambleHit:
    """A2 at one A1 candidate: the best (tau0, df) cell, refined.

    Always returns the best cell, the one with the least `preamble_p`;
    `passed` says whether that p is under `pfa`. Refinement, after the grid
    parabolas: frequency (and, when significant, a residual drift) from
    a line through the chunk-pair phase differences, coherent across the
    20 s preamble; then timing on the coherent correlation of the whole
    preamble at fractional offsets.

    kind: the waveform's preamble (CE_PRE, L_PRE); for L, cand is a pilot
    line. inp: `_a2_input` at cand already computed (one channelisation
    serves every kind at a line).
    """
    if inp is None:
        inp = _a2_input(fe, t0_index, cand.f_hz, cand.drift_hz_per_min, fs)
    chan = _a2_mf(inp, kind)
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

    # m is band-limited to the pulse's +-19 Hz (L: +-9.5 Hz) at 250 Hz, so a
    # cubic spline through it is exact to far below the noise at fractional timing.
    lo = int(max(0, math.floor(chan.t0_ch + tau - 80)))
    hi = int(min(len(chan.m), math.ceil(chan.t0_ch + tau + len(kind.seq) * kind.t_sym * CH_FS
                                        + 80)))
    spline = CubicSpline(np.arange(lo, hi), chan.m[lo:hi])
    dt = kind.chunk * kind.t_sym
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
    cen = chan.t0_ch + tau + (np.arange(len(norm)) + 0.5) * kind.chunk * kind.t_sym * CH_FS
    sig = chan.local[np.clip(np.round(cen).astype(np.int64), 0, len(chan.local) - 1)]
    lam_f = float(np.sum(np.abs(Sc) ** 2 / (norm * sig)))
    if not kind.data_offset_hz:
        # S ~ u * j K sum b a: arg u = arg sum S - pi/2 (template phases referred to t0)
        phase = float(np.angle(np.sum(Sc)) + kind.phase_offset)
    else:
        # L: S ~ u e^{j 2 pi (f_p tau0 + phi(tau0))} sum |b|^2. The sender
        # mixes its data down by f_p from its own t0 and the f_p shift here
        # is referred to nominal t0, tau0 earlier, and the templates' df and
        # rate rotation starts at the preamble, not t0, so df is the
        # frequency there and the pilot's phase advanced phi(tau0) = df tau0
        # - rate tau0^2 / 2 since t0; both are taken out, so phase is the
        # pilot's at t0 (u's convention).
        tau_s = tau / CH_FS
        turns = (df - kind.data_offset_hz) * tau_s - 0.5 * rate * tau_s * tau_s
        phase = float(np.angle(np.sum(Sc) * np.exp(
            -2j * np.pi * frontend.dsp.wrap_cycles(turns))) + kind.phase_offset)
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
        f_mix=chan.f_mix, tau0_sd_s=sd, chunk_sums=Sc, waveform=kind.waveform)


# --- lead-in ---------------------------------------------------------------------------------

LEAD_WIN_S = 1.0                         # energy-detector window
LEAD_Z = 4.0                             # per window: P(Z > 4 | noise) = e^-8


def lead_in_span(fe, t0_index: float, f_hz: float, drift_hz_per_min: float = 0.0,
                 tau0_s: float = 0.0, fs: int = FE_FS, *, z_min: float = LEAD_Z,
                 t_sym: float = T_SYM) -> float:
    """Seconds of lead-in carrier found before the preamble (design 6.3).

    The channel at f_hz (drift removed about t0), matched-filtered with no
    notch, is summed coherently over 1 s windows walking back from the
    keying start (t0 + tau0 - 8T on the receiver's clock, T = t_sym, the
    frame's symbol: an L pilot's lead-in ends 8 L symbols before tau0;
    the CE matched filter stays, a tone low-pass here); the span is the
    run of windows with Z = |sum| / sqrt(n sigma^2 / 2) > z_min (the sum's
    amplitude in units of its per-quadrature noise sd, so P(Z > 4) = e^-8
    per window on noise, and 1 s of lead-in reaches Z = 4 at about
    -25 dB SNR2500), up to
    LEAD_IN_MAX_S. sigma^2, the MF output's noise power, comes from
    `frontend.noise_psd` (the floor beside the carrier). 0.0 when the
    first window fails.
    """
    from .constants import LEAD_IN_MAX_S, SPAN
    end = tau0_s - SPAN * t_sym                       # s after nominal t0
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
    psd = (frontend.noise_psd(ch[have], CH_FS, fs_in=fs, audio_hz=FE_CENTER_HZ + float(f_mix))
           if have.sum() > 2 * CH_FS else [1.0])
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
    end = spec.keyed_end_pos * spec.t_sym + TBD_END_PAD_S
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
    """A3's best path in one channel: f per frame (audio Hz), Z and its p.

    shape_z, above_z, waveform: the spectral-shape test along the path
    (`shape_z`, `shape_waveform`) when the acquisition is waveform-aware;
    nan, nan and None otherwise (and None when the shape is ambiguous).
    l_path, l_shape_z, l_above_z: the path re-anchored on an L pilot
    (`l_pilot_bins`), what the hit is gated with as L, and the shape
    along it, when L is enabled (`track_before_detect`).
    """
    path: FreqPath
    z: float
    p_value: float
    n_frames: int
    centre_hz: float                         # the channel searched
    shape_z: float = float("nan")
    above_z: float = float("nan")
    waveform: int | None = None
    l_path: FreqPath | None = field(default=None, repr=False)
    l_shape_z: float = float("nan")
    l_above_z: float = float("nan")

    @functools.cached_property
    def f_hz(self) -> float:
        """Frequency at t0 from a straight line through the path.

        Weighted by the emissions, with one pass dropping points more than
        1 Hz off the line: before the signal keys (and in deep fades) the
        path wanders on noise and carries ~no weight.

        Fitted once per hit: A3's dedup, the arbitration and the
        detections each read it again (measured 8 fits for 3 hits on an
        L SHORT slot, 9 for 3 on a CE FULL one, ~0.16 ms a fit), and a
        hit's path is never changed once it is made (a new path is a new
        hit).
        """
        return float(np.polyval(self.line(), 0.0))

    def line(self) -> np.ndarray:
        """(slope Hz/s, intercept Hz at t0) of the weighted line fit."""
        return _path_line(self.path)

    @property
    def passed(self) -> bool:
        return self.p_value < TBD_PFA

    def to_detection(self, waveform: int | None = None) -> Detection:
        """A detection with no timing yet (V fits it, design 6.4), to be
        gated as `waveform` (default: the shape's, else CE); as L on the
        pilot-anchored path when there is one."""
        if waveform is None:
            waveform = WAVEFORM_CE if self.waveform is None else self.waveform
        path, f_hz = self.path, self.f_hz
        if waveform == WAVEFORM_L and self.l_path is not None:
            path = self.l_path
            f_hz = float(np.polyval(_path_line(path), 0.0))
        return Detection(f_hz=f_hz, path=path, timing=None, z_ref=float("nan"),
                         method="tbd", lead_in_s=0.0, waveform=int(waveform),
                         a3_z=float(self.z))


def _path_line(path: FreqPath) -> np.ndarray:
    """(slope Hz/s, intercept Hz at t0): `TbdHit.line` of any path."""
    t, f = path.t_s, path.f_hz
    w = np.asarray(path.weight, dtype=np.float64) + 1e-9
    keep = np.ones(len(t), dtype=bool)
    for _ in range(2):
        if keep.sum() < 2:
            break
        c = np.polyfit(t[keep], f[keep], 1, w=np.sqrt(w[keep]))
        keep = np.abs(f - np.polyval(c, t)) < 1.0
    return c


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


@functools.lru_cache(maxsize=4)
def _band_cov(n_bins: int, fs: float = FE_FS) -> tuple[float, float]:
    """(v0, v1): on white noise, the variance of a sum of n_bins adjacent
    normalised bins (each Exp(1), var 1) of one Hann frame, and the
    covariance of two such sums in consecutive frames (50% overlap).

    For complex Gaussian bins cov(|Y_i|^2, |Y_j|^2) = |rho_ij|^2, with
    rho the normalised spectrum of w^2 between bins of one frame (Hann:
    -2/3 and 1/6 at 1 and 2 bins) and of w(n) w(n - H) between frames.
    """
    w = frame_window(fs)
    L = len(w)
    H = int(round(STFT_HOP_S * fs))
    e = float(np.sum(w * w))
    r0 = np.abs(np.fft.fft(w * w)) / e
    w1 = np.zeros(L)
    w1[H:] = w[H:] * w[:L - H]
    r1 = np.abs(np.fft.fft(w1)) / e
    k = np.arange(-(n_bins - 1), n_bins)
    mult = n_bins - np.abs(k)
    return float(np.sum(mult * r0[k % L] ** 2)), float(np.sum(mult * r1[k % L] ** 2))


def shape_frames(t_s, specs, frame_s: float = STFT_FRAME_S) -> np.ndarray:
    """bool[n_frames]: the A3 frames (starting at t_s, s after t0) the shape
    test uses: none overlapping a callsign window of any of `specs` (both
    waveforms key down 16.5 Hz below their line there) or starting before
    the keying does (-8 symbols; the later of the specs' starts), which
    leaves out the lead-in."""
    t_s = np.asarray(t_s, dtype=np.float64)
    keep = t_s >= max(s.keyed_start_pos * s.t_sym for s in specs)
    for s in specs:
        for a, b in s.window_spans_s():
            keep &= ~((t_s < b) & (t_s + frame_s > a))
    return keep


def shape_z(E: np.ndarray, bins, keep=None, df_hz: float = 1.0 / STFT_FRAME_S,
            fs: float = FE_FS) -> tuple[float, float]:
    """(z1, z_above): the spectral-shape test along a path (design 3).

    E: normalised power - 1 (n_frames, n_bins), as A3's emissions; bins:
    the path's bin per frame; keep: frames to use (`shape_frames`).
    S_below sums E over L's data band below each frame's path bin
    (3.10..22.07 Hz, 76 bins of 0.25 Hz) and S_above over its mirror;
    z1 = (S_below - S_above) / (sd sqrt 2), z_above = S_above / sd, sd one
    band sum's noise sd (`_band_cov`, frames overlapping by half). Both are
    N(0, 1) on noise. An L signal puts its data below its pilot (z1 large);
    a CE signal is symmetric about its carrier (z1 ~ 0, z_above > 0). The
    difference is what is robust: the local CFAR absorbs part of a strong
    CE's sidebands and pulls an L signal's z_above negative. Frames where
    either band leaves E are not used, nor frames with no data (E = -1 on
    the path: `cfar_normalise` zeroes a frame whose median is 0, a capture
    gap, which would read as a deficit above the line); (nan, nan) when
    none is left.
    """
    E = np.asarray(E)
    bins = np.asarray(bins, dtype=np.int64)
    n, nb = E.shape
    lo = math.ceil(L_BAND_HZ[0] / df_hz - 1e-9)
    hi = math.floor(L_BAND_HZ[1] / df_hz + 1e-9)
    ok = np.ones(n, dtype=bool) if keep is None else np.asarray(keep, dtype=bool).copy()
    ok &= (bins - hi >= 0) & (bins + hi < nb)
    ok &= E[np.arange(n), np.clip(bins, 0, nb - 1)] > -1.0       # 0 power only with no data
    rows = np.flatnonzero(ok)
    if not len(rows):
        return float("nan"), float("nan")
    offs = np.arange(lo, hi + 1)
    b = bins[rows][:, None]
    below = float(np.sum(E[rows[:, None], b - offs], dtype=np.float64))
    above = float(np.sum(E[rows[:, None], b + offs], dtype=np.float64))
    v0, v1 = _band_cov(len(offs), fs)
    var = len(rows) * v0 + 2 * int(np.sum(ok[1:] & ok[:-1])) * v1
    return (below - above) / math.sqrt(2 * var), above / math.sqrt(var)


def shape_waveform(z1: float, z_above: float) -> int | None:
    """The waveform a shape says: L when z1 > SHAPE_L_Z with nothing
    significant above the line (z_above <= SHAPE_CE_ABOVE_Z); CE when
    z_above > SHAPE_CE_ABOVE_Z and |z1| < SHAPE_CE_MAX_Z1; else None
    (ambiguous, gated as both).

    z1's sd is 1 on noise only: a strong signal's noise cross terms widen
    it (a CE's sidebands are mirror images, so its own data cancels, but
    not the noise on them), and the PM's small deterministic asymmetry
    adds to it. On SHORT CE carrier paths z1 read 4.2-5.1 in 2 of 8
    passes at -10 dB and 1 of 8 at -5 and 0 dB, so z1 alone gated a CE
    whose preamble A2 missed only as L. Its power above the line (z_above
    29 at -15 dB, 95 at -10) is what an L signal never has; an L pass with
    a neighbour 3-22 Hz above reads ambiguous instead, gated as both."""
    if z1 > SHAPE_L_Z and not z_above > SHAPE_CE_ABOVE_Z:
        return WAVEFORM_L
    if z_above > SHAPE_CE_ABOVE_Z and abs(z1) < SHAPE_CE_MAX_Z1:
        return WAVEFORM_CE
    return None


def track_before_detect(fe, t0_index: float, spec: FrameSpec, cands=(), fs: float = FE_FS,
                        *, pfa: float = TBD_PFA, band_hz=BAND_HZ,
                        all_hits: bool = False, shape_specs=None) -> list[TbdHit]:
    """A3 over a stored slot: one Viterbi path per channel; hits at p < pfa
    (all channels with `all_hits`), deduplicated within 10 Hz, best first.

    shape_specs: the frames of a waveform-aware acquisition; each hit then
    carries its spectral shape (`shape_z` over `shape_frames(.., shape_specs)`)
    and the waveform it says (`shape_waveform`). With L among them the
    hit also carries its pilot-anchored path (`l_pilot_bins`) and the
    shape along that (`l_shape_z`, `l_above_z`), and the waveform is the
    two shapes' verdict (`_shape_verdict`): a weak L pass's path can run
    in its data band, where its own shape is ambiguous (the pilot then in
    the band above) and might read CE.
    """
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
    keep = None if shape_specs is None else shape_frames(t, tuple(shape_specs))
    hits, path_bins = [], []
    for k, c in enumerate(ci):
        bins = c - half + paths[k]
        p = gumbel_p(float(Z[k]), n)
        hits.append(TbdHit(path=_bins_path(E, f, t, bins), z=float(Z[k]), p_value=p,
                           n_frames=n, centre_hz=float(centres[k])))
        path_bins.append(bins)
    order = sorted(range(len(hits)), key=lambda k: -hits[k].z)
    out: list[TbdHit] = []
    for k in order:
        h = hits[k]
        if not all_hits and not h.p_value < pfa:
            continue
        if any(abs(h.f_hz - o.f_hz) < TBD_DEDUP_HZ for o in out):
            continue
        if keep is not None:
            h.shape_z, h.above_z = shape_z(E, path_bins[k], keep, df, fs)
            h.waveform = shape_waveform(h.shape_z, h.above_z)
            if any(sp.waveform == WAVEFORM_L for sp in shape_specs):
                lb = l_pilot_bins(E, f, t, h.line(), keep, fs)
                h.l_path = _bins_path(E, f, t, lb)
                h.l_shape_z, h.l_above_z = shape_z(E, lb, keep, df, fs)
                h.waveform = _shape_verdict(h.waveform,
                                            shape_waveform(h.l_shape_z, h.l_above_z))
        out.append(h)
    return out


def _shape_verdict(v: int | None, w: int | None) -> int | None:
    """The waveform of an A3 hit from its own path's shape (v) and its
    pilot-anchored path's (w): either one where the other is silent,
    ambiguous (None) where they disagree.

    On a CE carrier `l_pilot_bins` lands on the carrier (the carrier and
    its lower sideband peak there), so w = v. A weak L pass's path can run
    in its data band: L_FULL at -28 and -30 dB gave a second hit 16 Hz
    below the pilot (z1 -5 to -13, the pilot in its band above: ambiguous)
    whose anchored path is on the pilot (z1 14-22, L). Along an anchored
    path z1 is biased on noise (it is chosen for the band below: mean
    +1.1, sd 0.8 over 110 SHORT noise paths, none past 4), which is why it
    only fills in where the path's own shape is silent and can otherwise
    only make the verdict ambiguous, never turn it.
    """
    if v == w or w is None:
        return v
    return w if v is None else None


def _bins_path(E: np.ndarray, f: np.ndarray, t: np.ndarray, bins: np.ndarray) -> FreqPath:
    """The FreqPath of a path through E (one bin per frame starting at t):
    frame centres, the bins' frequencies refined between bins by a
    parabola on the log power where the emission stands out, weights the
    emission (floored at 0)."""
    n = len(t)
    df = f[1] - f[0]
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
    return FreqPath(t_s=t + STFT_FRAME_S / 2, f_hz=fk,
                    weight=np.maximum(em, 0.0).astype(np.float64))


def l_pilot_bins(E: np.ndarray, f: np.ndarray, t: np.ndarray, line, keep=None,
                 fs: float = FE_FS, span_hz=L_ANCHOR_HZ) -> np.ndarray:
    """int[n_frames]: an L pilot's bins near an A3 path, a path through E.

    At a weak L signal A3's Viterbi follows the data band rather than the
    pilot (L_FULL at -28 dB: 6 Hz below it): a path free to wander in a
    19 Hz band of signal collects about as much per frame as the steady
    pilot's bin, the pilot being 14% of the power. The pilot is then 3-22
    Hz above the path. Here a Viterbi in a window following the path's
    line (`span_hz` about it) runs on an L-matched emission, the bin plus
    the one-sided band below it (3.10-22.07 Hz, in units of its own noise
    sd, `_band_cov`), which peaks at the pilot and, varying slowly with
    the bin, gains little from wandering. Frames outside `keep` (callsign
    windows) count nothing. This only places the path; A3's statistic
    and its Gumbel are the pilot-only Z.
    """
    n, nb = E.shape
    df = f[1] - f[0]
    lo = math.ceil(L_BAND_HZ[0] / df - 1e-9)
    hi = math.floor(L_BAND_HZ[1] / df + 1e-9)
    k_lo = math.floor(span_hz[0] / df)
    k_hi = math.ceil(span_hz[1] / df)
    c = np.round((np.polyval(line, t + STFT_FRAME_S / 2) - f[0]) / df).astype(np.int64)
    g = c[:, None] + np.arange(k_lo - hi, k_hi + 1)[None, :]          # window and its band below
    ok = (g >= 0) & (g < nb)
    Eg = np.where(ok, E[np.arange(n)[:, None], np.clip(g, 0, nb - 1)], 0.0).astype(np.float64)
    cs = np.concatenate([np.zeros((n, 1)), np.cumsum(Eg, axis=1)], axis=1)
    j = np.arange(hi, Eg.shape[1])                                   # window columns in g
    band = cs[:, j - lo + 1] - cs[:, j - hi]                          # sum of Eg[:, j-hi .. j-lo]
    em = Eg[:, j] + band / math.sqrt(_band_cov(hi - lo + 1, fs)[0])
    em = np.where(ok[:, j], em, -1e3)
    if keep is not None:
        em[~np.asarray(keep, dtype=bool)] = 0.0
    path, _ = tbd_viterbi(em)
    return np.clip(g[np.arange(n), j[path]], 0, nb - 1)


# --- everything at once ---------------------------------------------------------------------

def frame_map(specs) -> dict:
    """{waveform: frame} of a tuple of frames, at most one per waveform."""
    out: dict = {}
    for s in specs:
        w = int(s.waveform)
        if w in out:
            raise ValueError(f"acquire takes at most one frame per waveform, not "
                             f"{out[w].name!r} and {s.name!r}")
        out[w] = s
    if not out:
        raise ValueError("acquire needs at least one frame")
    return out


def _path_of(f_hz: float, drift_hz_per_min: float, dur: float) -> FreqPath:
    t = np.array([TBD_START_S, 0.0, dur])
    return FreqPath(t_s=t, f_hz=f_hz + drift_hz_per_min * t / 60.0, weight=np.ones(3))


@dataclass
class Acquisition:
    lines: list                              # A1 CarrierCandidates
    preamble: list                           # A2 PreambleHits that passed
    tbd: list                                # A3 TbdHits that passed
    hinted: list = field(default_factory=list)   # CarrierCandidates of hints A2 did not pass
    pilots: list = field(default_factory=list)   # L pilot hypotheses searched (`pilot_lines`)

    def detections(self, spec) -> list[Detection]:
        """Every candidate for the verification gate: preamble hits, then
        hints A2 did not pass (method "hint", no timing), then A3's.

        With a tuple of frames (the waveform-aware acquisition), each
        Detection carries the waveform it is to be gated as, in this order:
        1. preamble hits of every enabled waveform, strongest (Lambda) first;
        2. hints A2 did not pass, in their waveform;
        3. A3 hits in the waveform their shape says, an ambiguous one as CE
           when CE is enabled (only CE is found blind below -34 dB);
        4. second guesses: an L copy of each ambiguous hit, then the hits
           whose shape says a waveform that is not enabled, gated as the
           one that is (a CE signal with a neighbour 3-22 Hz below can read
           as L: A3's lines are reordered, never dropped).
        An A3 detection within TBD_DEDUP_HZ of an earlier A3 one of the same
        waveform is the same gate and is left out (L copies anchor on a
        pilot another hit was already moved to).
        """
        if isinstance(spec, (tuple, list)):
            return self._joint_detections(frame_map(spec))
        dur = spec.keyed_end_pos * spec.t_sym
        t = np.array([TBD_START_S, 0.0, dur])
        hinted = [Detection(f_hz=c.f_hz, path=FreqPath(t_s=t, f_hz=c.f_hz + c.drift_hz_per_min
                                                       * t / 60.0, weight=np.ones(3)),
                            timing=None, z_ref=float("nan"), method="hint", lead_in_s=0.0)
                  for c in self.hinted]
        return ([h.to_detection(dur) for h in self.preamble] + hinted
                + [h.to_detection() for h in self.tbd])

    def _joint_detections(self, specs: dict) -> list[Detection]:
        def dur(w):
            return specs[w].keyed_end_pos * specs[w].t_sym

        pre = [h.to_detection(dur(h.waveform))
               for h in sorted(self.preamble, key=lambda h: -h.lam) if h.waveform in specs]
        hinted = []
        for c in self.hinted:
            for w in ([c.waveform] if c.waveform is not None else list(specs)):
                if w in specs:
                    hinted.append(Detection(
                        f_hz=c.f_hz, path=_path_of(c.f_hz, c.drift_hz_per_min, dur(w)),
                        timing=None, z_ref=float("nan"), method="hint", lead_in_s=0.0,
                        waveform=int(w)))
        first, copies, others = [], [], []
        for h in self.tbd:
            w = h.waveform
            if w is None:
                if WAVEFORM_CE in specs:
                    first.append(h.to_detection(WAVEFORM_CE))
                    if WAVEFORM_L in specs:
                        copies.append(h.to_detection(WAVEFORM_L))
                else:
                    first.append(h.to_detection(WAVEFORM_L))
            elif w in specs:
                first.append(h.to_detection(w))
            else:
                others += [h.to_detection(v) for v in specs]
        # A3's hits are 10 Hz apart, but gated as L each is moved to its
        # pilot, and a weak L pass's data band holds a second hit that
        # anchors on the same one (L_FULL at -28/-30 dB): one gate per
        # (waveform, frequency) is enough, the first in this order.
        tbd: list[Detection] = []
        for d in first + copies + others:
            if not any(g.waveform == d.waveform and abs(g.f_hz - d.f_hz) < TBD_DEDUP_HZ
                       for g in tbd):
                tbd.append(d)
        return pre + hinted + tbd


def drop_symbol_rate_aliases(hits: list, t_sym: float = T_SYM) -> list:
    """Passed A2 hits without those that are a stronger hit seen at
    f +- k/T (k in A2_ALIAS_K, within A2_ALIAS_HZ), with its timing
    (within T/2) and a 2 s Lambda A2_ALIAS_RATIO x smaller. Strongest first.
    T is t_sym, the hits' waveform's symbol (one waveform at a time).

    At an offset of k/T the preamble's symbol rotation is whole turns, so
    a CE signal correlates there as well as at its own frequency, down only
    by the matched filter's response (Lambda ~70 against ~590 for a
    signal at 0 dB, measured). A real neighbour there would have to share
    the timing to within 15 ms as well to be lost; A3 still offers it.
    For L the -1/T alias (16.5 Hz below the pilot, inside its own data
    band) passes with a ratio that falls toward 3 as the signal grows:
    `drop_band_owned` is what removes it.
    """
    out: list = []
    for h in sorted(hits, key=lambda h: -h.lam):
        if any(abs(abs(h.f_hz - g.f_hz) - k / t_sym) < A2_ALIAS_HZ
               and abs(h.tau0_s - g.tau0_s) < t_sym / 2
               and g.lam > A2_ALIAS_RATIO * h.lam
               for g in out for k in A2_ALIAS_K):
            continue
        out.append(h)
    return out


def drop_band_owned(hits: list) -> list:
    """Hits without the L hits whose pilot lies 3.1-22.1 Hz below a
    stronger (Lambda) L hit's, with its timing within L_OWN_T symbols:
    that band is the stronger signal's data, and a hit there is that
    signal. Strongest first.

    The hit this is for is the -1/T_L alias, 16.5 Hz below the pilot: it
    passes A2-L from about -12 dB (3/12 there, 12/12 at -6 and 0 dB,
    measured on SHORT's preamble) with a Lambda only 3-14x under the
    signal's, and it correlates only through the pulses' 2.5 Hz roll-off
    overlap, so its timing peak is broad: 9 of 31 passed aliases were
    more than T_L/2 off the signal's (-1.9 to +0.8 T_L), hence 3 T_L.
    """
    out: list = []
    for h in sorted(hits, key=lambda h: -h.lam):
        if h.waveform == WAVEFORM_L and any(
                g.waveform == WAVEFORM_L
                and g.f_hz - L_OWN_HZ[1] <= h.f_hz <= g.f_hz - L_OWN_HZ[0]
                and abs(h.tau0_s - g.tau0_s) < L_OWN_T * L_T_SYM for g in out):
            continue
        out.append(h)
    return out


def merge_hits(hits: list) -> list:
    """One hit per signal: hits within WAVEFORM_MERGE_HZ and T_L/2 of each
    other are one signal, of whichever waveform has the smaller p (then the
    larger Lambda). Across waveforms, one waveform's A2 can pass on the
    other's signal; within L, a pilot hypothesis or a hint offset can find
    a pilot a line found too. Strongest (Lambda) first."""
    out: list = []
    for h in sorted(hits, key=lambda h: (h.p_value, -h.lam)):
        if any(abs(h.f_hz - g.f_hz) < WAVEFORM_MERGE_HZ
               and abs(h.tau0_s - g.tau0_s) < L_T_SYM / 2 for g in out):
            continue
        out.append(h)
    return sorted(out, key=lambda h: -h.lam)


def live_end_s() -> float:
    """How far past t0 a live (preamble-only) acquisition reads FE: A2's
    timing search and its filters' margin past the preamble (A1 stops sooner)."""
    return A2_TAU_S + N_PRE * T_SYM + _A2_MARGIN_S


def acquire(cap: frontend.Capture, spec, *, live_only: bool = False,
            preprocessed: bool = False, hints=()) -> Acquisition:
    """A1 -> A2 at each line (and A3 over the slot unless live_only).

    `cap.fe` is blanked and normalised first unless `preprocessed`.
    hints: audio frequencies where a signal is expected (`hint_line`).
    Each gets A2 at its own line when no A1 line is that close: one more
    cell set at A2's false-alarm rate, so a weak signal outside A1's
    strongest A1_MAX_CANDS lines still has its preamble searched. One
    that A2 does not pass is kept in `hinted`, for a gate V that looks
    over the whole frame. A hint at a non-finite frequency (nan, +-inf,
    bare or in an (f, waveform) pair) is dropped before A2 (`_finite_hint`).

    spec: one frame -- today's acquisition, CE's, whatever the frame's
    waveform (only the A3 span and the hints' paths follow the frame) --
    or a tuple of frames, at most one per waveform, for the waveform-aware
    acquisition (`_acquire_joint`; `(FULL,)` listens for CE alone,
    `(FULL, L_FULL)` for both).
    """
    if isinstance(spec, (tuple, list)):
        return _acquire_joint(cap, frame_map(spec), live_only=live_only,
                              preprocessed=preprocessed, hints=hints)
    fe = cap.fe
    if not preprocessed:
        fe, keep = frontend.blank(fe, fs=cap.fs)
        fe, _ = frontend.normalise(fe, keep, fs=cap.fs)
    cs = carrier_line_stats(fe, cap.t0_index, cap.fs)
    lines = carrier_lines(fe, cap.t0_index, cap.fs, stats=cs)
    hits = [preamble_search(fe, cap.t0_index, c, cap.fs) for c in lines]
    hinted = []
    for f in hints:
        if not _finite_hint(f):
            continue
        if any(h.passed and abs(h.f_hz - f) <= HINT_SPAN_HZ + 0.5 for h in hits):
            continue
        near = [h for h in hits if abs(h.f_hz - f) <= HINT_SPAN_HZ + 0.5]
        line = hint_line(cs, float(f), cap.fs)
        h = near[0] if near else preamble_search(fe, cap.t0_index, line, cap.fs)
        if h.passed:
            hits.append(h)
        else:
            hinted.append(line)
    pre = drop_symbol_rate_aliases([h for h in hits if h.passed])
    for h in pre:
        h.lead_in_s = lead_in_span(fe, cap.t0_index, h.f_hz, h.drift_hz_per_min, h.tau0_s,
                                   cap.fs)
    tbd = [] if live_only else track_before_detect(fe, cap.t0_index, spec, lines, cap.fs)
    return Acquisition(lines=lines, preamble=pre, tbd=tbd, hinted=hinted)


def _finite_hint(f) -> bool:
    """False for a hint at no frequency (nan, +-inf), which acquire drops
    before A2: it has no line to search, and a history or a caller's
    arithmetic can produce one. Anything that is not a number is kept, to
    fail where it always did."""
    try:
        return math.isfinite(float(f))
    except (TypeError, ValueError):
        return True


def _hint_items(hints, specs: dict) -> list[tuple[float, list[int]]]:
    """(f, waveforms) per hint: a bare frequency means every enabled
    waveform, (f, waveform) that one (nothing when it is not enabled).
    A hint whose f is not finite is read (a malformed one still refuses)
    and dropped (`_finite_hint`)."""
    out = []
    for hnt in hints:
        if isinstance(hnt, (tuple, list)):
            f, w = hnt
            item = (float(f), [int(w)] if int(w) in specs else [])
        else:
            item = (float(hnt), list(specs))
        if math.isfinite(item[0]):
            out.append(item)
    return out


def _acquire_joint(cap: frontend.Capture, specs: dict, *, live_only: bool,
                   preprocessed: bool, hints) -> Acquisition:
    """The waveform-aware acquisition: `acquire` with a tuple of frames.

    A1 once. At each A1 line one channelisation (`_a2_input`) feeds the
    A2 of every enabled waveform; with L enabled, A2-L also runs at each
    line's pilot hypothesis (`pilot_lines`). Hints are searched with
    their waveform's A2 (an L hint A1 has no line for at f and f +- 1 Hz,
    A2's df reaching only +-0.5 Hz). The passed hits are arbitrated per
    waveform (`drop_symbol_rate_aliases` at its own T; for L
    `drop_band_owned`), then across (`merge_hits`); each one's lead-in is
    measured at its own T. A3 runs once over the longest enabled frame,
    each hit carrying the waveform its spectral shape says.
    """
    fe = cap.fe
    if not preprocessed:
        fe, keep = frontend.blank(fe, fs=cap.fs)
        fe, _ = frontend.normalise(fe, keep, fs=cap.fs)
    t0 = cap.t0_index
    cs = carrier_line_stats(fe, t0, cap.fs)
    lines = carrier_lines(fe, t0, cap.fs, stats=cs)
    kinds = [PREAMBLE_KINDS[w] for w in (WAVEFORM_CE, WAVEFORM_L) if w in specs]
    hits = []
    for c in lines:
        inp = _a2_input(fe, t0, c.f_hz, c.drift_hz_per_min, cap.fs)
        hits += [preamble_search(fe, t0, c, cap.fs, kind=k, inp=inp) for k in kinds]
    pilots = pilot_lines(cs, lines) if WAVEFORM_L in specs else []
    hits += [preamble_search(fe, t0, c, cap.fs, kind=L_PRE) for c in pilots]
    thr = a1_threshold(cs.M, A1_PFA, cap.fs)
    hinted = []
    for f, ws in _hint_items(hints, specs):
        for w in ws:
            own = [h for h in hits if h.waveform == w]
            if any(h.passed and abs(h.f_hz - f) <= HINT_SPAN_HZ + 0.5 for h in own):
                continue
            near = [h for h in own if abs(h.f_hz - f) <= HINT_SPAN_HZ + 0.5]
            line = hint_line(cs, f, cap.fs)
            if near:
                h = near[0]
            elif w == WAVEFORM_L and line.metric <= thr:
                tries = [preamble_search(fe, t0, CarrierCandidate(f + o, 0.0, line.metric),
                                         cap.fs, kind=L_PRE) for o in L_HINT_OFFSETS_HZ]
                h = min(tries, key=lambda h: (h.p_value, -h.lam))
            else:
                h = preamble_search(fe, t0, line, cap.fs, kind=PREAMBLE_KINDS[w])
            if h.passed:
                hits.append(h)
            else:
                hinted.append(dataclasses.replace(line, waveform=w))
    passed = [h for h in hits if h.passed]
    per = []
    for w in specs:
        per += drop_symbol_rate_aliases([h for h in passed if h.waveform == w],
                                        PREAMBLE_KINDS[w].t_sym)
    pre = merge_hits(drop_band_owned(per))
    for h in pre:
        h.lead_in_s = lead_in_span(fe, t0, h.f_hz, h.drift_hz_per_min, h.tau0_s, cap.fs,
                                   t_sym=specs[h.waveform].t_sym)
    tbd = []
    if not live_only:
        longest = max(specs.values(), key=lambda s: s.keyed_end_pos * s.t_sym)
        tbd = track_before_detect(fe, t0, longest, lines, cap.fs,
                                  shape_specs=tuple(specs.values()))
    return Acquisition(lines=lines, preamble=pre, tbd=tbd, hinted=hinted, pilots=pilots)

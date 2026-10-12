"""The waveform L tracker: gain, timing and gate V for L frames (the L
counterpart of `track`; docs/qrss/waveform-l.md).

Not a format module: nothing here goes on the air.

L sends a linear complex symbol stream 12.58 Hz below a pilot tone, so
the channel is centred on the pilot (`receiver.channel_for` with the
pilot's path) and `track.derotate` puts the pilot, not the data, at
0 Hz. The model is **one complex gain u**, the data band's reference
gain (a reference symbol a arrives as u a at the data-band matched
filter), fed by two independent measurement streams:

- **Data band (D).** z_dd = z_d exp(+j 2 pi f_p T tau(n)), the data
  mixed up to 0 Hz in *sender* time (tau from the timing model; f_p T
  is exactly 12207/16000 turn, and a receiver-time mix would drift
  2.2 turns over L_FULL at 100 ppm), then L's matched filter at every
  grid position. The template is closed form, exact by Nyquist:
  c = g_class mu (0 at unknown data, windows and the lead-in), with
  variance N + Pu g^2 (nu + D_class), N = PSD/T_L.
- **Pilot (P).** An RRC at 2 T_L on z_d, read at even positions (white
  at that spacing; it lets in 0.9% of the data's power, against 23.7%
  for an RRC at T_L), against `lwave.pilot` through the same filter with
  the clipper's pilot gain per section (`ldemod.PILOT_GAIN_*`). With the
  keying unknown, units 8..183 of every window, plus the filter's reach,
  give no pilot measurement; with it known the FSK is in the template.
  N_pl = PSD/(2 T_L). The pilot stream does not depend on the timing
  beyond where it is read, so it is recomputed only when the path or
  the read-out times move.
- **delta**, the complex ratio of the pilot's gain to the data band's
  (the rig's passband phase, up to 0.15 rad, and a delayed path): the
  pilot measures u delta c_pl. It is fitted once per pass by least
  squares of the data-band measurements at the known symbols on a
  pilot-only track (shrunk towards 1 by its own error), and again in
  the timing refinement, which profiles it out; it is **never** fitted
  inside gate V (delta = 1 there), so Z_ref stays N(0, 1) (or
  narrower, `verify_track_l`) on noise at a given timing. Gate V also
  checks the known symbols' residual against the noise model
  (`resid_ratio`), which is how it refuses a CE signal taken for L.

The two streams' (mu, R) are fused per position by inverse variance
and fed to `track`'s constant-velocity Kalman filter and RTS smoother
unchanged, with q chosen on `track.q_grid(L_T_SYM)`. In round A (data
unknown) the data band measures only the preamble and the references,
1 in 16 of the frame: the pilot carries the track between them, as
CE's carrier does. There is no spread removal: L's spread header copy
is linear and lives in the data template's mean (`make_classes_l`).

Timing: the coarse fit (`fit_timing_l`) correlates the known symbols
with MF_D{z_d e^{+j 2 pi f_p t}} conj(LP z_d), the pilot's low-pass
being the phase reference, over lags and clock errors, with the
receiver-time mix's per-lag phase and per-clock-error rotation in the
complex template; `refine_timing_l` maximises the known symbols'
correlation with the pilot-only gain track, re-mixing z_dd at each
trial timing and profiling delta's phase out in closed form against
its prior (a timing error of e symbols turns the data against the pilot
by f_p T e turns, which a fixed delta would fight and a fitted one
absorb). The cost is that tau0 comes mostly from the pulse envelope,
0.13 CH sample rms on l-short at -12 dB (6 seeds, from a blind gate V),
while a rig's passband phase cannot bias it; a timing error that small
costs the data nothing, delta following it.

`track.erase` and `track.report` work on an LTrackResult unchanged
(erase zeroes the data band's psi, which is what gate V and the
demodulator read); `report_l` is the report with SNR2500 quoted on the
frame's keyed power rather than on the references' |u|^2.

Calibrated constants (noiseless loopbacks through `channel.simulate`,
pinned here): the pilot filter's residual against its template, per
unit of signal power, and the frames' keyed power.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np
from scipy import signal

from . import ce, ldemod, lframe, lwave, sequences
from . import track as T
from .constants import (CH_FS, L_PILOT_HZ, L_T_SYM, SPAN, WAVEFORM_L, Z_ACCEPT)
from .frame import spread_signs
from .types import Detection, FreqPath, Timing, TrackReport
from .track import (C_MIN, LEAD_KEEP_S, PSI_MIN, PU_SMOOTH_S, R_FLOOR, RECENTER_MIN_HZ,
                    V_TIMING_TRIES, ChanCapture, MfTable, PathFn, TrackResult, _flat_path,
                    _keying_tuple, _parabola, _smooth, derotate, freq_path_spline,
                    kalman_rts, select_q)

# --- settings -------------------------------------------------------------------------

CARRIER_LP_HZ_L = (0.03, 0.3, 1.0, 2.0)     # coarse timing's pilot estimates (data from 3.1 Hz)
PILOT_K = 2                                 # pilot filter: an RRC at PILOT_K * T_L
PILOT_REACH = SPAN * PILOT_K                # its reach, L positions either side
_FP_T = float(L_PILOT_HZ) * L_T_SYM         # 12207/16000 turn per L symbol
_FP_NUM, _FP_DEN = 12207, 16000
assert abs(_FP_T - _FP_NUM / _FP_DEN) < 1e-15
DELTA_PRIOR_SD = 0.25                       # delta's prior about 1 (rig phase, tilt, gain model)
SEG_TIMING_S = T.SEG_TIMING_S
# refine_path for the pilot (13.8% of the power, ~5.6 dB under CE's carrier):
# the median is taken one-sided, above the pilot, clear of the data band
PATH_MED_LO_HZ = 1.0

# --- calibrated constants -----------------------------------------------------------------

# Pilot filter output against its template, noiseless: E|m_pl - u c_pl|^2 / Pu
# at even positions (the data's leakage through the filter's skirt plus the
# clipper's distortion in the gap below the pilot), measured per section on
# l-tiny and l-short at the genie's timing and gain: preamble 0.0156, header
# 0.0128, data 0.0174-0.0183, window ramps 0.009, W = 1 (lead-in, window
# interiors with the keying known) 0.000. One figure for every section the
# pilot shares with the symbols; it bounds the pilot's SINR near 15 dB and
# is nothing at the SNRs where L is used.
K_PL_DATA = 0.016                           # W = 0: preamble, header, data
K_PL_ID = 0.0                               # where the pilot is at average power (W = 1)

# The frame's mean power over its keyed span, on the baseband whose
# references arrive at G_REF_ABS (`channel.transmit_source(...).keyed_power`,
# 3-8 pictures per frame, spread +-0.01): what SNR2500 is quoted against,
# where |u|^2 is the references' power (`report_l`).
L_KEYED_POWER = {"l-tiny": 1.050, "l-short": 1.057, "l-medium": 1.025, "l-full": 1.009}


def _l_only(spec, what: str) -> None:
    assert spec.waveform == WAVEFORM_L, \
        f"{what} is L's; {spec.name!r} is waveform {spec.waveform}: use track's"


# --- the filters -------------------------------------------------------------------------

_PILOT_TABLE: list = []


def pilot_table() -> MfTable:
    """The pilot filter: a unit-energy RRC (alpha 0.15, +-8 of its symbols)
    at PILOT_K * T_L, read at positions in L symbols. Its output for a
    constant 1 is g0 (~1). Cached."""
    if not _PILOT_TABLE:
        t2 = PILOT_K * L_T_SYM
        sps = t2 * CH_FS
        half = int(math.ceil(SPAN * sps)) + 1
        m = np.arange(-SPAN * PILOT_K * 970, SPAN * PILOT_K * 970 + 1)
        norm = math.sqrt(float(np.sum(ce._rrc(m / (PILOT_K * 970)) ** 2)) / (PILOT_K * 970))
        d = np.arange(-half * T.MF_RES, half * T.MF_RES + 1) / T.MF_RES
        x = d / sps
        tab = np.where(np.abs(x) <= SPAN, ce._rrc(x) / norm, 0.0)
        tab.setflags(write=False)
        g0 = float(np.sum(np.where(np.abs(m) <= SPAN * PILOT_K * 970,
                                   ce._rrc(m / (PILOT_K * 970)), 0.0)) / norm / (PILOT_K * 970))
        _PILOT_TABLE.append(MfTable(t_sym=L_T_SYM, sps=sps, half=half, tab=tab, g0=g0))
    return _PILOT_TABLE[0]


def _pilot_taps() -> np.ndarray:
    """The pilot filter at integer CH samples, divided by its sps (an FIR)."""
    tb = pilot_table()
    k = np.arange(-tb.half, tb.half + 1)
    return tb.tab[k * T.MF_RES + tb.half * T.MF_RES] / tb.sps


def pilot_filter(x, tp) -> np.ndarray:
    """The pilot filter's output of x (CH samples) at fractional indices tp.

    One convolution at integer samples and linear interpolation: the
    output is band-limited to +-4.7 Hz at 250 Hz, where the interpolation
    is exact to about -55 dB (the closed-form `track.symbol_mf` through
    `pilot_table()` costs ~25x more for the same numbers). Samples
    outside x count as zero.
    """
    x = np.asarray(x)
    y = signal.oaconvolve(x, _pilot_taps(), mode="same")
    tp = np.asarray(tp, dtype=np.float64)
    i0 = np.floor(tp).astype(np.int64)
    f = tp - i0
    n = len(y)
    a = np.where((i0 >= 0) & (i0 < n), y[np.clip(i0, 0, n - 1)], 0)
    b = np.where((i0 + 1 >= 0) & (i0 + 1 < n), y[np.clip(i0 + 1, 0, n - 1)], 0)
    return a * (1 - f) + b * f


def data_mix(z_d, chan: ChanCapture, timing: Timing, n_pos: int, idx=None) -> np.ndarray:
    """z_dd = z_d exp(+j 2 pi f_p T tau(n)) at CH indices idx (default all):
    the data band mixed to 0 Hz in sender time (tau from `timing`)."""
    n = np.arange(len(z_d)) if idx is None else np.asarray(idx)
    tau = T._tau_of_rx(timing, n - chan.t0_index, n_pos, t_sym=L_T_SYM)
    turns = tau * (_FP_NUM / _FP_DEN)
    turns -= np.floor(turns)
    zz = np.asarray(z_d)[n] if idx is not None else np.asarray(z_d)
    return zz * np.exp(2j * np.pi * turns)


def data_mf(z_d, chan: ChanCapture, timing: Timing, pos, n_pos: int) -> np.ndarray:
    """The data-band matched filter at positions `pos`: L's RRC on z_dd,
    re-mixed for this timing (so a trial timing in the refinement sees the
    data exactly where that timing puts them)."""
    tb = T.mf_table(L_T_SYM)
    pos = np.asarray(pos, dtype=np.float64)
    if len(pos) == 0:
        return np.zeros(0, dtype=np.complex128)
    tp = chan.t0_index + T.rx_index(timing, pos, n_pos, CH_FS, t_sym=L_T_SYM)
    lo = max(int(math.floor(tp.min())) - tb.half - 1, 0)
    hi = min(int(math.ceil(tp.max())) + tb.half + 2, len(z_d))
    if hi <= lo:
        return np.zeros(len(pos), dtype=np.complex128)
    if len(pos) * (2 * tb.half + 1) < (hi - lo):
        # a few positions: mix only the samples their filters reach
        idx = np.unique((np.floor(tp)[:, None].astype(np.int64)
                         + np.arange(-tb.half, tb.half + 2)[None, :]).ravel())
        idx = idx[(idx >= 0) & (idx < len(z_d))]
        zz = np.zeros(len(z_d), dtype=np.complex128)
        zz[idx] = data_mix(z_d, chan, timing, n_pos, idx)
    else:
        zz = np.zeros(len(z_d), dtype=np.complex128)
        idx = np.arange(lo, hi)
        zz[lo:hi] = data_mix(z_d, chan, timing, n_pos, idx)
    return T.symbol_mf(zz, CH_FS, timing, pos, t0_index=chan.t0_index, n_pos=n_pos, table=tb)


# --- classes and templates -------------------------------------------------------------------

@dataclass
class LClasses:
    """Per stream symbol: template mean and variance (complex symbol units),
    which are known, and the class gain against the references.

    The spread header copy is linear (`lframe.data_latents`), so it is in
    mu and nu, never removed from the channel (`spread` is always None):
    with the header unknown it is inside the data's unit variance and
    `spread_rho` is the spec's hdr_rho; known, its mean is a further
    reference (spec 5.1, about 3.8% of the data power) and spread_rho 0.
    """
    mu: np.ndarray                         # complex128[n_sym]
    nu: np.ndarray                         # float64[n_sym]: E|a - mu|^2
    known: np.ndarray                      # bool[n_sym]
    gain: np.ndarray                       # float64[n_sym]: class gain (G_REF_L, G_HDR_L, G_DATA_L)
    hdr_bits: np.ndarray | None = None     # uint8[2474] once the header is known
    hdr_known: bool = False
    spread: None = None
    spread_rho: float = 0.0


def make_classes_l(spec, hdr_bits=None, prior=None) -> LClasses:
    """Round A (hdr_bits None) or round B classes; `prior` an EmPrior
    (x_hat, v per latent, in the pass's precoded domain) for the data."""
    _l_only(spec, "ltrack.make_classes_l")
    lay = lframe.layout(spec)
    n = spec.n_sym
    mu = np.zeros(n, dtype=np.complex128)
    nu = np.ones(n)
    known = np.zeros(n, dtype=bool)
    gain = np.full(n, float(ldemod.G_DATA_L))
    mu[lay.known_idx] = lay.known_val
    nu[lay.known_idx] = 0.0
    known[lay.known_idx] = True
    gain[lay.known_idx] = ldemod.G_REF_L
    gain[lay.hdr_sym] = ldemod.G_HDR_L
    bits = None
    if hdr_bits is not None:
        bits = np.asarray(hdr_bits, dtype=np.uint8)
        if spec.n_hdr:
            mu[lay.hdr_sym] = sequences.qpsk(bits)
            nu[lay.hdr_sym] = 0.0
            known[lay.hdr_sym] = True
    rho = float(spec.hdr_rho)
    s = math.sqrt(1.0 + rho)
    h = spread_signs(spec, bits) if (bits is not None and rho > 0) else None
    if prior is not None:
        x = np.asarray(prior.x_hat, dtype=np.float64)
        v = np.clip(np.asarray(prior.v, dtype=np.float64), 0.0, 1.0)
        if h is not None:
            lm, lv = (x + math.sqrt(rho) * h) / s, v / (1.0 + rho)
        else:
            lm, lv = x / s, (v + rho) / (1.0 + rho)
        mu[lay.data] = lframe.pair(lm)
        nu[lay.data] = 0.5 * (lv[0::2] + lv[1::2])
    elif h is not None:
        mu[lay.data] = lframe.pair(math.sqrt(rho) * h / s)
        nu[lay.data] = 1.0 / (1.0 + rho)
    return LClasses(mu=mu, nu=nu, known=known, gain=gain, hdr_bits=bits,
                    hdr_known=bits is not None, spread=None,
                    spread_rho=rho if bits is None else 0.0)


def _distortion(spec) -> np.ndarray:
    """float64[n_sym]: the clipper's distortion at the data MF, relative to
    the symbol's own class gain^2 (`ldemod.D_*`)."""
    lay = lframe.layout(spec)
    d = np.full(spec.n_sym, float(ldemod.D_PASS_L))
    d[lay.hdr_sym] = ldemod.D_HDR_L
    d[lay.known_idx] = ldemod.D_REF_L
    return d


def data_templates(spec, grid, classes: LClasses, withhold_known: bool = False):
    """(c, nu, gain, dist, is_sym) on the grid: the data-band template
    c = g mu (0 off the stream), the variance nu, the class gain and
    distortion per position, and which positions carry a stream symbol."""
    lay = lframe.layout(spec)
    grid = np.asarray(grid, dtype=np.int64)
    p_lo = int(grid[0])
    gi = np.asarray(lay.pos) - p_lo
    c = np.zeros(len(grid), dtype=np.complex128)
    nu = np.zeros(len(grid))
    gain = np.zeros(len(grid))
    dist = np.zeros(len(grid))
    is_sym = np.zeros(len(grid), dtype=bool)
    mu_s, nu_s = classes.mu, classes.nu
    if withhold_known:
        mu_s = np.where(classes.known, 0.0, mu_s)
        nu_s = np.where(classes.known, 1.0, nu_s)
    c[gi] = classes.gain * mu_s
    nu[gi] = nu_s
    gain[gi] = classes.gain
    dist[gi] = _distortion(spec)
    is_sym[gi] = True
    return c, nu, gain, dist, is_sym


def _pilot_gain(tau, spec, lead_in_s: float = 0.0) -> np.ndarray:
    """The clipper's gain on the pilot at sender time tau: per section
    (preamble, header, data) where the pilot shares the clipper with the
    symbols, PILOT_GAIN_ID where it runs at average power (W = 1), and
    W g_ID + (1 - W) g_section on the ramps."""
    lay = lframe.layout(spec)
    tau = np.asarray(tau, dtype=np.float64)
    g = np.full(tau.shape, float(ldemod.PILOT_GAIN_DATA))
    hdr0 = float(lay.pos[spec.n_pre]) - 0.5 if spec.n_sym > spec.n_pre else np.inf
    dat0 = (float(lay.pos[spec.n_pre + spec.n_hdr]) - 0.5
            if spec.n_sym > spec.n_pre + spec.n_hdr else np.inf)
    g = np.where(tau < dat0, ldemod.PILOT_GAIN_HDR, g)
    g = np.where(tau < hdr0, ldemod.PILOT_GAIN_PRE, g)
    W = lwave.id_weight(tau, spec, lead_in_s)
    return W * ldemod.PILOT_GAIN_ID + (1.0 - W) * g


def pilot_wave(tau, spec, keying=None, lead_in_s: float = 0.0) -> np.ndarray:
    """The pilot as it arrives on u's scale (a reference symbol a arrives as
    u a): `lwave.pilot` / A_ID through the clipper's gain, over G_REF_ABS."""
    return (_pilot_gain(tau, spec, lead_in_s) * lwave.pilot(tau, spec, keying, lead_in_s)
            / (lwave.A_ID * ldemod.G_REF_ABS))


# --- the tracker -----------------------------------------------------------------------------

@dataclass
class LTrackResult(TrackResult):
    """`track.TrackResult` for an L pass. The inherited per-position fields
    are the DATA band's: z_d (pilot-derotated), m (data-band MF), c (its
    template), N, psi (L's symbol MF), meas (where the data band may
    measure), u (the reference gain), P, pu. The pilot stream sits beside
    them: m_pl, c_pl (its template on u's scale: the pilot measures
    u delta c_pl), N_pl, psi_pl, meas_pl (even positions it may measure),
    and u_pl, P_pl, the pilot-only track (of u delta) the timing and
    delta were fitted against. resid_ratio is gate V's (`resid_ratio`,
    nan outside it). ook is always False."""
    m_pl: np.ndarray | None = None
    c_pl: np.ndarray | None = None
    N_pl: np.ndarray | None = None
    psi_pl: np.ndarray | None = None
    meas_pl: np.ndarray | None = None
    delta: complex = 1.0 + 0.0j
    u_pl: np.ndarray | None = None
    P_pl: np.ndarray | None = None
    resid_ratio: float = math.nan          # gate V's known-symbol residual (`resid_ratio`)


def _q_near(q, grid) -> tuple:
    i = int(np.argmin(np.abs(np.log(np.array(grid)) - math.log(q))))
    return tuple(grid[max(0, i - 2):i + 3])


class _Model:
    """What one L tracker run holds between measurements: the templates,
    the masks, the noise model, and the per-stream (mu, R)."""

    def __init__(self, tr: LTrackResult, spec, classes: LClasses, withhold_known: bool,
                 keying, lead: float, n_lead: int):
        grid = tr.pos
        self.tr, self.spec, self.classes = tr, spec, classes
        self.keying, self.lead = keying, lead
        c, nu, gain, dist, is_sym = data_templates(spec, grid, classes, withhold_known)
        self.c_all, _, _, _, _ = data_templates(spec, grid, classes, False)
        self.c, self.nu, self.gain, self.dist, self.is_sym = c, nu, gain, dist, is_sym
        lay = lframe.layout(spec)
        known = np.zeros(len(grid), dtype=bool)
        known[np.asarray(lay.pos)[classes.known] - int(grid[0])] = True
        self.is_known = known
        meas_d = is_sym & (np.abs(c) > 0)
        if withhold_known:
            meas_d &= ~known
        self.meas_d = meas_d
        # where the data MF is needed: measurements and the known symbols (Z_ref)
        self.need = is_sym & ((np.abs(c) > 0) | known)
        even = (grid % 2) == 0
        mp = even.copy()
        if keying is None and spec.n_win:
            mp &= ~T._window_unknown(spec, grid, False, reach=PILOT_REACH)
        if n_lead == 0:
            # an unclaimed lead-in's pilot (7x the frame's) would read as a fade
            mp &= grid >= PILOT_REACH - SPAN
        self.meas_p = mp
        self.measured = meas_d | mp
        W = lwave.id_weight(grid.astype(np.float64), spec, lead)
        self.k_pl = K_PL_DATA * (1.0 - W) + K_PL_ID * W
        tr.meas, tr.meas_pl = meas_d, mp

    def streams(self, Pu_loc, delta=1.0 + 0.0j, data=True, pilot=True, data_mask=None):
        """((mu_D, R_D), (mu_P, R_P)): normalised observations of u per position
        (R inf where none), the pilot's divided by delta; no floor yet."""
        tr = self.tr
        n = len(tr.pos)
        out = []
        if data:
            md = self.meas_d if data_mask is None else (self.meas_d & data_mask)
            cp = self.c * tr.psi
            ok = md & (tr.psi >= PSI_MIN) & (np.abs(cp) >= C_MIN)
            mu = np.zeros(n, dtype=np.complex128)
            R = np.full(n, np.inf)
            mu[ok] = tr.m[ok] / cp[ok]
            g2 = self.gain[ok] ** 2
            R[ok] = (tr.N[ok] + Pu_loc[ok] * g2 * (self.nu[ok] + self.dist[ok])) / \
                np.abs(cp[ok]) ** 2
            out.append((mu, R))
        else:
            out.append((np.zeros(n, dtype=np.complex128), np.full(n, np.inf)))
        if pilot:
            cp = tr.c_pl * tr.psi_pl * delta
            ok = self.meas_p & (tr.psi_pl >= PSI_MIN) & (np.abs(cp) >= C_MIN)
            mu = np.zeros(n, dtype=np.complex128)
            R = np.full(n, np.inf)
            mu[ok] = tr.m_pl[ok] / cp[ok]
            R[ok] = (tr.N_pl[ok] + Pu_loc[ok] * self.k_pl[ok]) / np.abs(cp[ok]) ** 2
            out.append((mu, R))
        else:
            out.append((np.zeros(n, dtype=np.complex128), np.full(n, np.inf)))
        return out

    def fused(self, Pu_loc, delta=1.0 + 0.0j, data=True, pilot=True, data_mask=None):
        """(mu, R) for the Kalman filter: the streams by inverse variance, plus
        the model-error floor."""
        (mu_d, R_d), (mu_p, R_p) = self.streams(Pu_loc, delta, data, pilot, data_mask)
        with np.errstate(divide="ignore", invalid="ignore"):
            wd, wp = 1.0 / R_d, 1.0 / R_p
            W = wd + wp
            R = np.where(W > 0, 1.0 / W, np.inf)
            mu = np.where(W > 0, (mu_d * wd + mu_p * wp) * R, 0.0)
        fin = np.isfinite(R)
        R[fin] += R_FLOOR * Pu_loc[fin]
        return mu, R

    def pu_initial(self) -> float:
        tr = self.tr
        num = den = 0.0
        cp = tr.c_pl * tr.psi_pl
        ok = self.meas_p & (tr.psi_pl >= PSI_MIN) & (np.abs(cp) >= C_MIN)
        if ok.any():
            num += float(np.sum(np.abs(tr.m_pl[ok]) ** 2 - tr.N_pl[ok]))
            den += float(np.sum(np.abs(cp[ok]) ** 2 + self.k_pl[ok]))
        cd = self.c * tr.psi
        okd = self.meas_d & (tr.psi >= PSI_MIN) & (np.abs(cd) >= C_MIN)
        if okd.any():
            num += float(np.sum(np.abs(tr.m[okd]) ** 2 - tr.N[okd]))
            den += float(np.sum(np.abs(cd[okd]) ** 2
                                + self.gain[okd] ** 2 * (self.nu[okd] + self.dist[okd])))
        if den <= 0:
            return 1e-30
        floor = 1e-3 * float(np.median(tr.N_pl)) if tr.N_pl is not None else 0.0
        return max(num / den, floor, 1e-30)


def _pu_local(u, P, measured):
    """(Pu_loc, Pu_g): the local signal power (|u|^2 + P over PU_SMOOTH_S)
    and its mean over the positions that measure anything. A callsign
    window with the keying unknown is 12 s without a measurement, and
    one that ends the frame is an extrapolation whose P grows without
    bound: P is capped at the measured mean, so neither reaches the
    noise model of its neighbours or the process noise's scale."""
    raw = np.abs(u) ** 2 + P
    sel = measured if measured.any() else np.ones(len(u), dtype=bool)
    cap = max(float(np.mean(raw[sel])), 1e-30)
    Pu_loc = np.maximum(_smooth(np.abs(u) ** 2 + np.minimum(P, cap),
                                int(PU_SMOOTH_S / L_T_SYM)), 1e-30)
    return Pu_loc, max(float(np.mean(Pu_loc[sel])), 1e-30)


def _hold_ends(u, P, R):
    """Hold the smoothed gain at its first and last measured value outside
    them: the constant-velocity model's extrapolation over a callsign
    window that ends the frame runs away (the gain is not wanted there,
    but `_recenter` and `report` weight by |u|^2)."""
    fin = np.nonzero(np.isfinite(R))[0]
    if len(fin) == 0:
        return u, P
    a, b = int(fin[0]), int(fin[-1])
    u, P = u.copy(), P.copy()
    u[:a], P[:a] = u[a], P[a]
    u[b + 1:], P[b + 1:] = u[b], P[b]
    return u, P


def _lead_supported_l(chan: ChanCapture, z_d, tm: Timing, spec, lead: float) -> bool:
    """Whether a claimed lead-in's pilot is there: its power through the
    pilot filter, against the frame's first positions, must be at least a
    third of what the template says (the lead-in sends the pilot at
    average power, about 8x the preamble's)."""
    n = int(math.floor(min(lead, LEAD_KEEP_S) / L_T_SYM))
    if n < 4 + PILOT_REACH:
        return True
    pl = np.arange(-SPAN - n, -SPAN - PILOT_REACH, 2)
    pf = np.arange(PILOT_REACH, min(300, spec.n_pos), 2)
    p = np.concatenate([pl, pf])
    tp = chan.t0_index + T.rx_index(tm, p, spec.n_pos, CH_FS, t_sym=L_T_SYM)
    m = pilot_filter(z_d, tp)
    c = pilot_filter(pilot_wave(T._tau_of_rx(tm, np.arange(len(z_d)) - chan.t0_index,
                                             spec.n_pos, t_sym=L_T_SYM), spec, None, lead),
                     tp)
    N = chan.noise_psd_at(T.rx_index(tm, p, spec.n_pos, t_sym=L_T_SYM) / CH_FS) / \
        (PILOT_K * L_T_SYM)
    k = len(pl)
    el = float(np.mean(np.abs(m[:k]) ** 2 - N[:k]))
    ef = float(np.mean(np.abs(m[k:]) ** 2 - N[k:]))
    ratio = float(np.mean(np.abs(c[:k]) ** 2)) / max(float(np.mean(np.abs(c[k:]) ** 2)), 1e-30)
    return el > ratio * ef / 3


def _fit_delta(model: _Model, v, Pv, Pu_loc, delta0=1.0 + 0.0j) -> complex:
    """delta from the data band's measurements against a pilot-only track v
    (of u delta): mu_D = v / delta + e, regressed for r = 1/delta, shrunk
    to 1 with the prior DELTA_PRIOR_SD by r's own variance.

    v is noisy, so the regression is diluted where v's error is noise
    added to a steady gain rather than an MMSE shrinkage (the smoother's
    prior is diffuse at low frequencies): |delta| reads high by about
    Pv/|v|^2, 1.09 at -27 dB on l-short steady (12 seeds; 1.015 at -12
    dB, genie 1.01). Taking Pv out of the denominator fixes the steady
    case and overshoots on fading (0.81 at -20 dB on moderate, where the
    smoother does shrink), so it is not done: the scale error this leaves
    in u is far under the noise at the SNRs where it is large."""
    (mu_d, R_d), _ = model.streams(Pu_loc, pilot=False)
    ok = np.isfinite(R_d)
    if not ok.any():
        return complex(delta0)
    r0 = 1.0 / complex(delta0)
    w = 1.0 / (R_d[ok] + abs(r0) ** 2 * Pv[ok] + R_FLOOR * Pu_loc[ok])
    vv = v[ok]
    den = float(np.sum(w * np.abs(vv) ** 2))
    if den <= 0:
        return complex(delta0)
    r_ls = complex(np.sum(w * np.conj(vv) * mu_d[ok])) / den
    s2 = DELTA_PRIOR_SD ** 2
    r = (r_ls * den + 1.0 / s2) / (den + 1.0 / s2)
    if not np.isfinite(r) or abs(r) < 1e-3:
        return complex(delta0)
    return complex(1.0 / r)


def track_l(chan: ChanCapture, spec, det: Detection, classes: LClasses | None = None,
            cw_known=None, prior=None, *, timing: Timing | None = None, freq=None,
            withhold_known: bool = False, outer: int = 2, refine: bool = True,
            lead_in_s: float | None = None, z_d=None) -> LTrackResult:
    """One L tracker run over a pass (`track.track`'s contract).

    `classes` (default round A, `make_classes_l`) sets the data template;
    `cw_known` = (keying, False) once the callsign keying is known (the
    windows' keyed units then measure the pilot, FSK in the template).
    `withhold_known` gives the known symbols no measurement and fixes
    delta at 1 (gate V). The timing is det.timing (or `timing`), refined
    on the pilot-only track when `refine`; the frequency path (the
    pilot's) is det.path (or `freq`), re-centred `outer` times.
    """
    _l_only(spec, "ltrack.track_l")
    if classes is None:
        classes = make_classes_l(spec, prior=prior)
    keying, ook = _keying_tuple(cw_known)
    if ook:
        raise ValueError("waveform L keys its callsign by FSK only: cw_known must say ook=False")
    fn = freq if freq is not None else freq_path_spline(det.path)
    tm = timing if timing is not None else det.timing
    if tm is None:
        raise ValueError("track_l needs a timing (fit_timing_l, or gate V's)")
    lead = det.lead_in_s if lead_in_s is None else lead_in_s
    lead = float(min(max(lead, 0.0), 10.0))
    if z_d is None:
        z_d = derotate(chan, fn)
    if lead > 0 and not _lead_supported_l(chan, z_d, tm, spec, lead):
        lead = 0.0
    n_lead = int(math.floor(min(lead, LEAD_KEEP_S) / L_T_SYM))
    p_lo = -SPAN - n_lead
    grid = np.arange(p_lo, spec.n_pos, dtype=np.int64)
    n_pos = spec.n_pos
    qg = T.q_grid(L_T_SYM)
    tr = LTrackResult(spec=spec, chan=chan, timing=tm, freq=fn, pos=grid, z_d=z_d,
                      m=None, c=None, u=None, P=None, N=None, psi=None, meas=None, pu=None,
                      classes=classes, keying=keying, ook=False, lead_in_s=lead)
    model = _Model(tr, spec, classes, withhold_known, keying, lead, n_lead)
    tr.c = model.c_all if not withhold_known else model.c
    dtab = T.mf_table(L_T_SYM)
    cache = {"psi": (None, None), "pl": (None, None)}

    def measure(full: bool = False):
        key = (tr.timing.tau0, tr.timing.ppm, tr.timing.gamma)
        sel = model.is_sym if full else model.need
        m = np.zeros(len(grid), dtype=np.complex128)
        m[sel] = data_mf(tr.z_d, chan, tr.timing, grid[sel], n_pos)
        tr.m = m
        if cache["psi"][0] != key:
            psi = T.psi_at(chan.keep, tr.timing, grid, t0_index=chan.t0_index, n_pos=n_pos,
                           table=dtab)
            tp = chan.t0_index + T.rx_index(tr.timing, grid, n_pos, CH_FS, t_sym=L_T_SYM)
            psi_pl = (np.ones(len(grid)) if chan.keep is None else
                      np.clip(pilot_filter(np.asarray(chan.keep, dtype=np.float64), tp).real
                              / pilot_table().g0, 0.0, 1.0))
            cache["psi"] = (key, (psi, psi_pl))
        tr.psi, tr.psi_pl = cache["psi"][1]
        t = tr.t_pos_s()
        psd = chan.noise_psd_at(t)
        tr.N = psd / L_T_SYM
        tr.N_pl = psd / (PILOT_K * L_T_SYM)
        pkey = (id(tr.z_d), key)
        if cache["pl"][0] != pkey:
            tp = chan.t0_index + T.rx_index(tr.timing, grid, n_pos, CH_FS, t_sym=L_T_SYM)
            tau = T._tau_of_rx(tr.timing, np.arange(len(tr.z_d)) - chan.t0_index, n_pos,
                               t_sym=L_T_SYM)
            c_pl = pilot_filter(pilot_wave(tau, spec, keying, lead), tp)
            m_pl = pilot_filter(tr.z_d, tp)
            cache["pl"] = (pkey, (m_pl, c_pl))
        tr.m_pl, tr.c_pl = cache["pl"][1]

    def pu_loc():
        if tr.u is None:
            Pu_g = model.pu_initial()
            return np.full(len(grid), Pu_g), Pu_g
        return _pu_local(tr.u, tr.P, model.measured)

    def run(q_grid=qg):
        Pu_loc, Pu_g = pu_loc()
        mu, R = model.fused(Pu_loc, tr.delta)
        q = q_grid[0] if len(q_grid) == 1 else select_q(mu, R, Pu_g, q_grid)[0]
        u, P, ll = kalman_rts(mu, R, q, Pu_g)
        u, P = _hold_ends(u, P, R)
        tr.u, tr.P, tr.q, tr.loglik, tr.pu = u, P, q, ll, Pu_loc
        return q

    def pilot_track(q):
        """The pilot-only track (of u delta) at q, from the current measurements."""
        Pu_loc, Pu_g = pu_loc()
        mu, R = model.fused(Pu_loc, 1.0 + 0.0j, data=False)
        v, Pv, _ = kalman_rts(mu, R, q, Pu_g)
        v, Pv = _hold_ends(v, Pv, R)
        tr.u_pl, tr.P_pl = v, Pv
        return v, Pv, Pu_loc

    def fit_delta(q):
        v, Pv, Pu_loc = pilot_track(q)
        tr.delta = _fit_delta(model, v, Pv, Pu_loc, tr.delta)

    measure()
    q = run()
    if not withhold_known:
        fit_delta(q)
        q = run(_q_near(q, qg))
    if refine and not withhold_known:
        # the timing is refined against the pilot-only track at the gain
        # agility chosen without the known symbols (`track.track`'s reason:
        # chosen with them, a timing error makes an agile q look likely)
        for _ in range(3):
            Pu_loc, Pu_g = pu_loc()
            mw, Rw = model.fused(Pu_loc, tr.delta, data_mask=~model.is_known)
            qw, _ = select_q(mw, Rw, Pu_g, qg)
            v, Pv, _ = pilot_track(qw)
            new_tm = refine_timing_l(tr, u_ref=v, P_ref=Pv)
            if new_tm is None:
                break
            moved = abs(new_tm.tau0 - tr.timing.tau0) + abs(
                (new_tm.ppm - tr.timing.ppm) * 1e-6 * n_pos * (L_T_SYM * CH_FS))
            tr.timing = new_tm
            measure()
            fit_delta(qw)
            if moved < 0.02:
                break
    q = run()
    df_max = 0.0
    for it in range(outer + 1):
        if it == outer and df_max <= RECENTER_MIN_HZ:
            break
        fn2, df_max = T._recenter(tr)
        if df_max < 0.002:
            break
        tr.freq = fn2
        tr.z_d = derotate(chan, fn2)
        measure()
        q = run(_q_near(q, qg))
    tr.df_max = df_max
    if not withhold_known:
        measure(full=True)
    return tr


def refine_timing_l(tr: LTrackResult, iters: int = 3, u_ref=None, P_ref=None) -> Timing | None:
    """Newton steps on the known symbols' likelihood given a gain track
    (`track.refine_timing`'s method, on the data band).

    l(theta) = min_phi [-2 |r| Re(e^{-j phi} B(theta)) + |r e^{j phi} - 1|^2 / s^2],
    B = sum_known conj(U_k c_k psi_k) m_k(theta) / Var_k: the known
    symbols' part of the channel's likelihood at theta, with m the
    data-band MF of z_d re-mixed in sender time at theta, U the pilot-only
    track (default tr.u_pl, else tr.u), theta = (tau0, slip[, gamma]) in
    CH samples, |r| = 1/|delta| and s = DELTA_PRIOR_SD, r's prior about 1.
    It is the correlation alone: sum |m_k(theta)|^2 / Var_k, which the
    residual form sum |m - r U c psi|^2 / Var adds, is the noise at the
    samples theta reads rather than evidence about it (its expectation
    does not move with theta), and at low SNR its fluctuation swamped the
    pulse envelope (from the true timing, 12 seeds of l-short at -27 dB:
    0.105 T rms off and 0.33 dB under the genie's data SINR with it, 0.040
    T and 0.13 dB without; 6 of l-tiny at -24 dB, 0.086 T against 0.027).

    The phase of r = 1/delta is profiled out in closed form at every
    theta, against its prior: a timing error of e symbols turns the data
    against the pilot by f_p T e = 0.763 e turn, and a phase fixed at the
    last fit would pull the timing back to wherever it was fitted (that
    phase's curvature is 3.4x the pulse envelope's). With the data strong
    (|B| >> 1/s^2) tau0 comes from the envelope, as in CE, and a rig's
    passband phase cannot bias it; weak, the prior lends the timing some
    of the phase's information. The slip's phase ramp over the pass counts
    either way. gamma for frames over 303 s when it raises the known
    symbols' Z by more than 1. The inverse Hessian is the covariance.
    """
    spec = tr.spec
    _l_only(spec, "ltrack.refine_timing_l")
    n_pos = spec.n_pos
    sps = L_T_SYM * CH_FS
    lay = lframe.layout(spec)
    known = tr.classes.known if tr.classes is not None else \
        np.isin(np.arange(spec.n_sym), lay.known_idx)
    gk = tr.gi(np.asarray(lay.pos)[known])
    sel = (gk >= 0) & (gk < len(tr.pos))
    gk = gk[sel]
    ok = (tr.psi[gk] >= PSI_MIN) & (np.abs(tr.c[gk]) > 0)
    gk = gk[ok]
    if len(gk) < 8:
        return None
    gain = (tr.classes.gain[known] if tr.classes is not None else
            np.full(int(known.sum()), ldemod.G_REF_L))[sel][ok]
    dist = _distortion(spec)[known][sel][ok]
    pk = tr.pos[gk].astype(np.float64)
    U = (tr.u_pl if tr.u_pl is not None else tr.u) if u_ref is None else u_ref
    PP = (tr.P_pl if tr.u_pl is not None else tr.P) if P_ref is None else P_ref
    c = tr.c[gk]
    tgt = U[gk] * c * tr.psi[gk]
    var = tr.N[gk] + gain ** 2 * dist * tr.pu[gk] + np.abs(c) ** 2 * PP[gk]
    A = float(np.sum(np.abs(tgt) ** 2 / var))
    if not A > 0:
        return None
    # U is of u delta when it is the pilot-only track: r = 1/delta, |r| fixed
    r_abs = 1.0 / abs(tr.delta) if (u_ref is not None or tr.u_pl is not None) else 1.0
    prior = 1.0 / DELTA_PRIOR_SD ** 2
    chan = tr.chan

    def timing_of(th):
        tau0, slip, gam = th
        return Timing(tau0=tau0, ppm=slip / (n_pos * sps) * 1e6, gamma=gam, z=0.0,
                      cov=np.zeros((3, 3)))

    def mf(th):
        return data_mf(tr.z_d, chan, timing_of(th), pk, n_pos)

    def ell(th):
        B = complex(np.sum(np.conj(tgt) * mf(th) / var))
        return r_abs ** 2 * A - 2 * r_abs * abs(B + prior)

    def zk(th):
        m = mf(th)
        B = complex(np.sum(np.conj(tgt) * m / var))
        den = math.sqrt(max(float(np.sum(np.abs(tgt) ** 2 * tr.N[gk] / var ** 2)), 1e-300))
        return abs(B) / den

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
                w = np.linalg.eigvalsh(H)
            except np.linalg.LinAlgError:
                H = None
                break
            if np.any(w <= 0):
                # outside the peak's concave core: a coarse slip several samples
                # off sits on the lobe's shoulder (l-short at -26 dB: 7 samples,
                # where Newton alone stopped and kept it). A gradient step of up
                # to 1.5 samples instead, kept only if it gains; an indefinite H
                # is no covariance
                H = None
                gn = float(np.max(np.abs(g)))
                if not gn > 0:
                    break
                step = -g * (1.5 / gn)
                for _ in range(4):
                    trial = th.copy()
                    trial[list(dims)] += step
                    if ell(trial) < f0:
                        break
                    step = step / 2
                else:
                    break
                th = trial
                continue
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
    if not np.all(np.isfinite(th)):
        return None
    dims = (0, 1)
    if H is not None and n_pos * L_T_SYM > 10000 * T.T_SYM:
        th3, H3 = newton(th, (0, 1, 2))
        if H3 is not None and np.all(np.isfinite(th3)) and zk(th3) > zk(th) + 1.0:
            th, H, dims = th3, H3, (0, 1, 2)
    cov = None
    if H is not None:
        try:
            cov_s = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            cov_s = None
        if cov_s is not None and np.all(np.diag(cov_s) > 0):
            J = np.diag([1.0, 1e6 / (n_pos * sps), 1.0])[np.ix_(dims, dims)]
            cov = np.zeros((3, 3))
            cov[np.ix_(dims, dims)] = J @ cov_s @ J.T
    if cov is None:
        if np.allclose(th, th0):
            return None
        cov = np.array(tm.cov, dtype=np.float64)      # moved by gradient steps only
    out = timing_of(th)
    out.z = float(zk(th))
    out.cov = cov
    return out


def report_l(tr: LTrackResult, kappa: float = 1.0, suspect: bool = False) -> TrackReport:
    """`track.report` for an L pass, with SNR2500 on the frame's keyed power:
    |u|^2 is the references' power, G_REF_ABS^2 / L_KEYED_POWER of the
    signal's (the report alone reads 0.01-0.20 dB low; 1.0 for a frame not
    in the table)."""
    rep = T.report(tr, kappa, suspect)
    p = L_KEYED_POWER.get(tr.spec.name, 1.0)
    return replace(rep, snr2500_db=rep.snr2500_db + 10 * math.log10(p / ldemod.G_REF_ABS ** 2))


# --- genie (tests and calibration) --------------------------------------------------------------

def genie_track_l(chan: ChanCapture, spec, truth, pilot_hz: float,
                  classes: LClasses | None = None) -> LTrackResult:
    """An LTrackResult with the channel simulator's true timing and gain.

    `truth` is a `channel.Truth`; u is its u_data (the data band's gain,
    which a delayed path makes differ from the pilot's), scaled by one
    least-squares complex factor over the known symbols (the front end's
    scale and G_REF_ABS); P = 0. The pilot stream is measured too, and
    delta is its least-squares ratio to u c_pl at the pilot's measurable
    positions.
    """
    _l_only(spec, "ltrack.genie_track_l")
    classes = make_classes_l(spec) if classes is None else classes
    fn = PathFn(np.array([-30.0, 0.0, 30.0]), np.full(3, float(pilot_hz)))
    z_d = derotate(chan, fn)
    t = np.asarray(truth.t_pos_s, dtype=np.float64)
    p = np.arange(spec.n_pos)
    slope, icpt = np.polyfit(p, t * CH_FS, 1)
    sps = L_T_SYM * CH_FS
    tm = Timing(tau0=float(icpt), ppm=float((slope / sps - 1) * 1e6), gamma=0.0, z=0.0,
                cov=np.zeros((3, 3)))
    grid = np.arange(-SPAN, spec.n_pos, dtype=np.int64)
    tr = LTrackResult(spec=spec, chan=chan, timing=tm, freq=fn, pos=grid, z_d=z_d, m=None,
                      c=None, u=None, P=np.zeros(len(grid)), N=None, psi=None, meas=None,
                      pu=None, classes=classes)
    model = _Model(tr, spec, classes, False, None, 0.0, 0)
    tr.c = model.c_all
    m = np.zeros(len(grid), dtype=np.complex128)
    m[model.is_sym] = data_mf(z_d, chan, tm, grid[model.is_sym], spec.n_pos)
    tr.m = m
    ut = np.concatenate([np.full(SPAN, truth.u_data[0]), np.asarray(truth.u_data)])
    ref = model.c * ut
    fit = model.is_known
    alpha = np.vdot(ref[fit], m[fit]) / np.vdot(ref[fit], ref[fit])
    tr.u = alpha * ut
    tr.pu = np.abs(tr.u) ** 2
    tr.psi = T.psi_at(chan.keep, tm, grid, t0_index=chan.t0_index, n_pos=spec.n_pos,
                      table=T.mf_table(L_T_SYM))
    tp = chan.t0_index + T.rx_index(tm, grid, spec.n_pos, CH_FS, t_sym=L_T_SYM)
    tr.psi_pl = (np.ones(len(grid)) if chan.keep is None else
                 np.clip(pilot_filter(np.asarray(chan.keep, dtype=np.float64), tp).real
                         / pilot_table().g0, 0.0, 1.0))
    psd = chan.noise_psd_at(tr.t_pos_s())
    tr.N = psd / L_T_SYM
    tr.N_pl = psd / (PILOT_K * L_T_SYM)
    tau = T._tau_of_rx(tm, np.arange(len(z_d)) - chan.t0_index, spec.n_pos, t_sym=L_T_SYM)
    tr.c_pl = pilot_filter(pilot_wave(tau, spec, None, 0.0), tp)
    tr.m_pl = pilot_filter(z_d, tp)
    ok = model.meas_p
    ref = tr.u[ok] * tr.c_pl[ok]
    tr.delta = complex(np.vdot(ref, tr.m_pl[ok]) / np.vdot(ref, ref)) if ok.any() else 1 + 0j
    tr.u_pl, tr.P_pl = tr.u * tr.delta, np.zeros(len(grid))
    return tr


# --- coarse timing --------------------------------------------------------------------------------

def known_symbols_l(spec, hdr_bits=None) -> T._Known:
    """Preamble and references (+ the header symbols once decoded), complex."""
    _l_only(spec, "ltrack.known_symbols_l")
    lay = lframe.layout(spec)
    idx = np.asarray(lay.known_idx)
    val = np.asarray(lay.known_val, dtype=np.complex128)
    if hdr_bits is not None and spec.n_hdr:
        idx = np.concatenate([idx, lay.hdr_sym])
        val = np.concatenate([val, sequences.qpsk(np.asarray(hdr_bits, dtype=np.uint8))])
    return T._Known(pos=np.asarray(lay.pos)[idx], val=val)


def _lag_phase(lag) -> np.ndarray:
    """exp(-j 2 pi f_p lag / CH_FS): the receiver-time mix's phase at a lag
    (12207/970 Hz at 250 Hz: 12207/242500 turn per CH sample, exactly)."""
    lag = np.asarray(lag, dtype=np.float64)
    turns = lag * (12207.0 / 242500.0)
    return np.exp(-2j * np.pi * (turns - np.floor(turns)))


def _clock_phase(pos, d) -> np.ndarray:
    """exp(-j 2 pi f_p T d pos): the data's turn against the pilot at
    position pos under a clock error d (the receiver-time mix's drift)."""
    turns = np.asarray(pos, dtype=np.float64) * (d * _FP_NUM / _FP_DEN)
    return np.exp(-2j * np.pi * (turns - np.floor(turns)))


def fit_timing_l(ch, ch_fs, path, spec, tmpl_fn=None, tau0_init=None,
                 ppm_range: float = 250.0, *, t0_index: float, tau_range=None,
                 z_d=None, n_best: int = 1):
    """Coarse timing of an L candidate: tau0 (CH samples) and ppm
    (`track.fit_timing`'s contract and return shape).

    x = MF_D{z_d e^{+j 2 pi f_p (n - n_ref)/fs}} conj(LP(z_d)): the data
    band mixed up in receiver time and matched-filtered, against the
    pilot's low-pass as the phase reference (at each of CARRIER_LP_HZ_L,
    all under the data band's 3.1 Hz edge). It is complex, and the
    correlation with the known symbols is coherent: Re(e^{-j 2 pi f_p
    lag/fs} sum_k conj(a_k) e^{-j 2 pi f_p T d pos_k} x(o_k + lag)), the
    first factor the mix's phase at the lag, the second the data's drift
    against the pilot under the clock error d (5.6 turns over L_FULL at
    250 ppm). Its noise sd is sqrt(sigma^2/2 sum|a|^2), sigma^2 =
    median|x|^2 / ln 2. Clock errors in steps of 0.2 T over the frame;
    segments then refine the line as in CE (`_segment_timing_l`).
    """
    _l_only(spec, "ltrack.fit_timing_l")
    fs = int(ch_fs)
    if z_d is None:
        fn = freq_path_spline(path) if isinstance(path, FreqPath) else path
        z_d = derotate(ch, fn)
    kn = tmpl_fn if isinstance(tmpl_fn, T._Known) else known_symbols_l(spec)
    n_pos = spec.n_pos
    sps = L_T_SYM * fs
    dur = n_pos * sps
    if tau0_init is None:
        lo, hi = -2.5 * fs - 2, 2.5 * fs + 2
    else:
        r = 3.0 if tau_range is None else float(tau_range)
        lo, hi = tau0_init - r, tau0_init + r
    step = 0.2 / n_pos
    nd = int(math.floor(ppm_range * 1e-6 / step))
    deltas = np.arange(-nd, nd + 1) * step
    base = int(math.floor(t0_index)) + int(math.floor(lo))
    nlag = int(math.ceil(hi)) - int(math.floor(lo)) + 1
    frac0 = t0_index - math.floor(t0_index)
    n_ref = base + frac0
    taps = T._mf_taps(fs, L_T_SYM)
    a = np.asarray(kn.val, dtype=np.complex128)
    pos = np.asarray(kn.pos, dtype=np.float64)
    z_d = np.asarray(z_d, dtype=np.complex128)
    y = signal.oaconvolve(z_d * np.conj(_lag_phase(np.arange(len(z_d)) - n_ref)), taps,
                          mode="same")
    lagph = _lag_phase(np.arange(nlag))
    peaks = []
    for lp in CARRIER_LP_HZ_L:
        uc = T._lowpass(z_d, lp, fs)
        x = y * np.conj(uc)
        sel = slice(max(base, 0), max(min(base + nlag + int(dur) + 1, len(x)), max(base, 0) + 1))
        sig2 = max(float(np.median(np.abs(x[sel]) ** 2)) / math.log(2.0), 1e-30)
        norm = math.sqrt(sig2 / 2 * float(np.sum(np.abs(a) ** 2)))
        Z = np.full((len(deltas), nlag), -np.inf)
        if nlag <= 64:
            lags = np.arange(nlag)
            for i, d in enumerate(deltas):
                o = pos * sps * (1 + d) + frac0
                idx = base + o[:, None] + lags[None, :]
                i0 = np.floor(idx).astype(np.int64)
                f = idx - i0
                ok = (i0 >= 0) & (i0 + 1 < len(x))
                i0c = np.clip(i0, 0, len(x) - 2)
                v = (x[i0c] * (1 - f) + x[i0c + 1] * f) * ok
                g = np.conj(a) * _clock_phase(pos, d)
                Z[i] = (lagph * (g @ v)).real / norm
        else:
            ng = int(math.ceil(dur * (1 + ppm_range * 1e-6))) + 4
            nfft = 1 << int(math.ceil(math.log2(nlag + ng + 2)))
            xs = np.zeros(nfft, dtype=np.complex128)
            s_lo, s_hi = max(base, 0), min(base + nfft, len(x))
            if s_hi > s_lo:
                xs[s_lo - base:s_hi - base] = x[s_lo:s_hi]
            X = np.fft.fft(xs)
            for i, d in enumerate(deltas):
                o = pos * sps * (1 + d) + frac0
                i0 = np.floor(o).astype(np.int64)
                f = o - i0
                g = np.conj(a) * _clock_phase(pos, d)
                # G(n) at the template's samples: sum_n G(n) x(n + lag) for every lag
                G = (np.bincount(i0, (g * (1 - f)).real, minlength=nfft)[:nfft]
                     + np.bincount(i0 + 1, (g * f).real, minlength=nfft)[:nfft]) + 1j * (
                    np.bincount(i0, (g * (1 - f)).imag, minlength=nfft)[:nfft]
                    + np.bincount(i0 + 1, (g * f).imag, minlength=nfft)[:nfft])
                c = np.fft.ifft(X * np.conj(np.fft.fft(np.conj(G))))
                Z[i] = (lagph * c[:nlag]).real / norm
        Zl = Z.max(axis=0)
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
            t0abs, delta, score = _segment_timing_l(x, sig2, kn, base + j + dj + frac0, delta,
                                                    sps, n_pos, n_ref)
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


def _segment_timing_l(x, sig2, kn: T._Known, t0abs: float, delta: float, sps: float,
                      n_pos: int, n_ref: float, seg_s: float = SEG_TIMING_S):
    """(t0abs, delta, score): `track._segment_timing` on L's complex x, each
    local lag with its mix phase and each position with its clock phase."""
    pos = kn.pos.astype(np.float64)
    a = np.asarray(kn.val, dtype=np.complex128)
    nseg = max(1, int(math.ceil(n_pos * L_T_SYM / seg_s)))
    edges = np.linspace(pos.min(), pos.max() + 1, nseg + 1)
    lags = np.arange(-sps, sps + 1e-9, 0.1)
    segs = [(pos >= s0) & (pos < s1) for s0, s1 in zip(edges[:-1], edges[1:])]
    segs = [m for m in segs if m.sum() >= 8]

    def stat(sel, t0, d, lg):
        o = t0 + pos[sel] * sps * (1 + d)
        idx = o[:, None] + lg[None, :]
        i0 = np.floor(idx).astype(np.int64)
        f = idx - i0
        ok = (i0 >= 0) & (i0 + 1 < len(x))
        i0 = np.clip(i0, 0, len(x) - 2)
        v = (x[i0] * (1 - f) + x[i0 + 1] * f) * ok
        g = np.conj(a[sel]) * _clock_phase(pos[sel], d)
        c = (_lag_phase(t0 + lg - n_ref) * (g @ v)).real
        return c / math.sqrt(sig2 / 2 * float(np.sum(np.abs(a[sel]) ** 2)))

    def local(sel, t0, d):
        c = stat(sel, t0, d, lags)
        if not np.all(np.isfinite(c)):
            return None
        k = int(np.argmax(c))
        if c[k] < 4.0:
            return None
        dk = _parabola(c[k - 1], c[k], c[k + 1]) if 0 < k < len(c) - 1 else 0.0
        return lags[k] + dk * 0.1, float(c[k]) ** 2

    def fit(tp, ab, wt, t0_, d_):
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
    score = 0.0
    zero = np.zeros(1)
    for sel in segs:
        score += max(float(stat(sel, t0, d, zero)[0]), 0.0) ** 2
    return t0, d, score


# --- the verification gate ----------------------------------------------------------------------

def refine_path_l(chan: ChanCapture, spec, path: FreqPath) -> FreqPath:
    """`track.refine_path` on the pilot: the line's median taken above it
    (PATH_MED_LO_HZ to 4 span + 10 Hz), clear of the data band below.

    ratio, fine_ratio and fine_share stay CE's. They are ratios of the
    measured line to its floor, so they ask the same of the pilot as of
    CE's carrier; the pilot being 5.6 dB weaker, the fine 5 s pass turns
    off 5.6 dB sooner (between -12 and -16 dB SNR2500 on l-short, CE's
    between -20 and -24). Not retuned by a sweep: a 1 Hz/min warm-up
    (tau 60 s) is followed to 0.01 Hz rms at -12 and at -20 dB."""
    span = T.PATH_SPAN_HZ
    return T.refine_path(chan, spec, path, T.PATH_SEG_S, span,
                         med_band=(PATH_MED_LO_HZ, 4 * span + 10.0))


def verify_track_l(chan: ChanCapture, spec, cand: Detection, *, fit: bool = True,
                   tau_range=None, erase_s=()) -> LTrackResult | None:
    """The tracker run behind `verify_l`, with tr.z_ref set (over what was
    heard: the symbols in `erase_s` are erased after tracking).

    The known symbols are withheld and delta is 1, so u_hat is the pilot's
    alone and Z_ref = sum w Re(m conj(u d)) / sd over the preamble and
    references (d their template, `track.z_ref_stat`) is N(0, 1) or
    narrower on noise at a given timing (sd about 0.6 on l-tiny, CE's
    TINY 0.5: the variance counts u_hat's error P, as large as u_hat on
    noise); an unfitted rig phase delta costs cos(arg delta) of it (0.99
    at 0.15 rad). At a fitted timing it is the best of the lags searched
    and so shifted up, more than CE's (the data turn against the pilot
    by 0.05 turn per CH sample of lag, so nearby lags try other phases):
    mean 0.7 at +-6 samples, 0.9 over a blind search, max 2.9 in 52
    noise channels; 20 more blind: mean 0.5, sd 1.2, max 3.2 (the tail
    out at Z_ACCEPT is not measured).

    **Not L.** That statistic believes N, the noise floor 85-120 Hz away,
    and a CE signal under an L hypothesis puts its sideband, far above
    N, into L's data band: the known symbols then correlate with it at
    about 1.5 sigma of the true noise, Z_ref 6-12 against CE short at -12
    and -6 dB (the pilot taken on the carrier, at 1.41 s of lag in every
    case, or 10-22 Hz above it on the sideband), growing with the CE
    signal's SNR -- accepted, and on a capture of a strong CE signal with
    L listened for alone, nothing else would refuse it. So Z_ref is
    divided by sqrt(max(1, R)), R the known symbols' residual power over
    the model's (`resid_ratio`): those CE locks read R 2.8-30 and Z_ref
    under 3, while an L signal's R is 0.8-1.5 (Z_ref down 1 dB at most,
    on fading) and noise's 1.05 +- 0.21. Z_ref only shrinks, so its tail
    on noise stays N(0, 1)'s or narrower."""
    _l_only(spec, "ltrack.verify_track_l")
    fn = freq_path_spline(cand.path)
    z_d = derotate(chan, fn)
    tm = cand.timing
    if fit or tm is None:
        tau0 = None if tm is None else tm.tau0
        rng = tau_range
        if tau0 is not None and rng is None:
            sd = math.sqrt(max(float(tm.cov[0, 0]), 0.0)) if tm.cov.size else 1.0
            rng = max(3.0, 4.0 * sd)
        tm = fit_timing_l(chan, CH_FS, fn, spec, None, tau0, t0_index=chan.t0_index,
                          tau_range=rng, z_d=z_d)
    classes = make_classes_l(spec)
    tr = track_l(chan, spec, cand, classes, timing=tm, freq=fn, withhold_known=True, outer=0,
                 refine=False, z_d=z_d)
    T.erase(tr, erase_s)
    c_known, _, gain, dist, _ = data_templates(spec, tr.pos, classes)
    z = T.z_ref_stat(tr, c_known, d_known=c_known)
    tr.resid_ratio = resid_ratio(tr, c_known, gain, dist)
    tr.z_ref = z / math.sqrt(max(tr.resid_ratio, 1.0))
    return tr


def resid_ratio(tr: LTrackResult, c_known, gain, dist) -> float:
    """R: the known symbols' residual power against what the noise model
    says it is, weighted as each symbol's noise weighs in Z_ref.

        R = sum_k om_k |m_k - u_k c_k|^2 / sum_k om_k (N_k + |c_k|^2 P_k + |u_k|^2 g_k^2 D_k)

    om_k = w_k^2 |c_k|^2 |u_k|^2 (`track.z_ref_stat`'s w), over the known
    symbols it reads; c_known, gain, dist from `data_templates` with the
    known symbols known. An L signal, or noise, leaves R about 1 (measured
    on gate V tracks of l-short: 0.77-1.54 steady to disturbed at -6 to
    -25 dB; l-tiny noise 1.05 +- 0.20 at a fixed timing, 40 seeds, 1.13 +-
    0.33 over a blind search, 20, which moves noise's Z_ref by 0.05 on
    average and its maximum not at all). A signal in the data band
    that is not L reads R >> 1: a CE carrier taken for the pilot, with
    its sideband in L's data band, measured 2.8-30 (CE short at -12 and
    -6 dB, pilot hypotheses on the carrier and 10.7, 12.4 and 22 Hz above
    it). 1.0 when no known symbol is usable."""
    lay = lframe.layout(tr.spec)
    gk = tr.gi(np.asarray(lay.pos)[tr.classes.known])
    gk = gk[(gk >= 0) & (gk < len(tr.pos))]
    gk = gk[tr.psi[gk] >= PSI_MIN]
    if len(gk) == 0:
        return 1.0
    m, u, N, P = tr.m[gk], tr.u[gk], tr.N[gk], tr.P[gk]
    c = np.asarray(c_known)[gk]
    u2, c2 = np.abs(u) ** 2, np.abs(c) ** 2
    V = c2 * (N * u2 + c2 * u2 * P + N * P) / 2
    ok = V > 0
    if not ok.any():
        return 1.0
    w = c2[ok] * u2[ok] / V[ok]
    om = w * w * c2[ok] * u2[ok]
    e2 = np.abs(m[ok] - u[ok] * c[ok]) ** 2
    e0 = N[ok] + c2[ok] * P[ok] + u2[ok] * np.asarray(gain)[gk][ok] ** 2 * \
        np.asarray(dist)[gk][ok]
    den = float(np.sum(om * e0))
    return float(np.sum(om * e2)) / den if den > 0 else 1.0


def verify_l(ch, ch_fs, spec, cand: Detection, *, fit: bool = True, tau_range=None,
             erase_s=()) -> Detection | None:
    """Gate V for an L candidate (`track.verify`'s contract and control
    flow): the pilot's path measured on the frame from the candidate's
    path, its flat version and the whole-frame line in turn, the timing
    fitted, the track run with the known symbols withheld, accepted at
    Z_ref > Z_ACCEPT; a blind candidate on its line also tries the coarse
    fit's runners-up. The Detection returned has waveform L."""
    _l_only(spec, "ltrack.verify_l")
    tr, path = None, cand.path
    for base in (cand.path, _flat_path(cand.path), None):
        pth = refine_path_l(ch, spec, base) if base is not None else \
            T.line_path(ch, spec, cand.path)
        alt = verify_track_l(ch, spec, replace(cand, path=pth), fit=fit, tau_range=tau_range,
                             erase_s=erase_s)
        if alt is not None and (tr is None or alt.z_ref > tr.z_ref):
            tr, path = alt, pth
        if tr is not None and tr.z_ref > Z_ACCEPT:
            break
        if base is None and fit and cand.timing is None and V_TIMING_TRIES > 1:
            fn = freq_path_spline(pth)
            tms = fit_timing_l(ch, CH_FS, fn, spec, None, None, t0_index=ch.t0_index,
                               z_d=derotate(ch, fn), n_best=V_TIMING_TRIES)
            for tm in tms[1:]:
                alt = verify_track_l(ch, spec, replace(cand, path=pth, timing=tm), fit=False,
                                     erase_s=erase_s)
                if alt is not None and (tr is None or alt.z_ref > tr.z_ref):
                    tr, path = alt, pth
                if tr is not None and tr.z_ref > Z_ACCEPT:
                    break
        if tr is not None and tr.z_ref > Z_ACCEPT:
            break
    if tr is None or not tr.z_ref > Z_ACCEPT:
        return None
    return Detection(f_hz=cand.f_hz, path=path, timing=tr.timing, z_ref=tr.z_ref,
                     method=cand.method, lead_in_s=cand.lead_in_s, waveform=WAVEFORM_L)

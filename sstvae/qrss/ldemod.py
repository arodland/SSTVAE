"""Symbol extraction, weights and header LLRs for waveform L (the L
counterpart of `demod`; docs/qrss/waveform-l.md).

Not a format module. The tracker (`ltrack`) hands over the data band's
matched filter m_p (L's pulse, the data centre mixed to 0 Hz in sender
time), its gain u_p normalised to the references (a reference symbol a
arrives as u a), u's error variance P_p, the noise N_p (PSD / T_L,
complex: E|n|^2 = N) and the unblanked share psi_p. A stream symbol of
class gain g (`G_*_L`: the clipper's gain on that class against the
references) gives

    r = m conj(u) / (|u|^2 psi g),        E[r | a] = a
    y = lframe.unpair(r)                    two latents (or coded bits) per symbol
    s2n = (N + g^2 P) / (g^2 |u|^2 psi^2)   per latent, the same for I and Q

with no halving (unpair's sqrt(2) puts the complex noise's two halves
back on one latent each) and no carrier term (the pilot does not reach
the data matched filter). A symbol is erased (s2 = inf) where psi < 0.5
or |u|^2 is under `demod.U_MIN_REL` of its median, as in CE.

**The x scale.** Everything after the symbol estimates is CE's, on the
latents the precoder sent: with the spread header (spec 5.1 for L,
`lframe.data_latents`) data symbol k carries (x + sqrt(rho) h)/sqrt(1+rho)
on latents 2k, 2k+1, so

- header unknown: y_x = sqrt(1+rho) y, s2n_x = (1+rho) s2n, and the
  unknown copy is rho of extra variance (CE's structure exactly);
- header known: y_x = sqrt(1+rho) y - sqrt(rho) h, extra 0. A linear
  signal cannot be de-spread on the channel as CE's constant envelope
  is (`track.remove_spread`), so the demodulator subtracts it here.

s2 = kappa s2n_x + D_x + extra (D_x = D_PASS_L (1+rho), `d_pass_x`), and
`demod.block_estimate` does the rest. The header block is scaled by the
header's own class gain (G_HDR_L, about the references', not the data's
0.881): llr = 2 y_h / (kappa s2n_h + D_HDR_L), coded bit 2j on I and 2j+1
on Q of header symbol j, so y_h is already in coded-bit order. While the
header is unknown the spread copy adds `demod.spread_llr(y_x, kappa
s2n_x + D_x)`, which is spec 5.1's L formula. These LLRs mean what CE's
do (log P(0)/P(1)), so L and CE passes of one segment add.

**Calibration** (`calibrate_l`) is `demod.calibrate` with L's residuals:
the references' complex residual r - a (per latent dimension, against
D_REF_L) and the data's unit-RMS contract E y_x^2 = 1 + extra + D_x +
kappa s2n_x; combined, shrunk to 1 and clamped exactly as CE's (copied:
CE's numbers do not move). L has about half CE's references (1,770
against 3,539 on FULL), and they carry much more of u than CE's do, so
the reference branch allows for each measured reference's pull on u
(`kappa_estimates_l`), which CE's form leaves out.
"""

from __future__ import annotations

import math

import numpy as np

from . import lframe
from .constants import N_HDR_BITS, SPREAD_RHO, WAVEFORM_L
from .demod import (KAPPA_PRIOR_SD, KAPPA_RANGE, KAPPA_S2_MAX, KAPPA_S2_REL, U_MIN_REL,
                    block_estimate, spread_llr)
from .frame import spread_signs

# --- pinned constants -----------------------------------------------------------------
#
# Method: noiseless genie loopbacks (`lwave.loopback_stats`: the clipped frame,
# `to_data_centre`, L's matched filter at true timing; LS complex gains per
# symbol class), 24 frames: L_MEDIUM and L_FULL, Gaussian and tanh latents
# (`unit_rms_latents`), precoded as sent (`precoder.precode`, a slot per
# frame), a different random header and callsign per frame, a 4 s lead-in on
# half. Figures are mean +- sd [min, max] over the 24 (2026-10-12). The lead-in
# moves none of them (it changes the pilot only before the preamble's first
# pulse). Precoding matters: the WHT makes tanh latents Gaussian on the air,
# and the unprecoded tanh figures the provisional values came from (pilot
# 0.672-0.681 in the data) are not what a sender transmits.

# Class gains against the references' (the tracker's u is the reference gain).
#   data/ref 0.8846 +- 0.0029 [0.880, 0.891] (Gaussian 0.8839, tanh 0.8854); the
#   picture moves it, and spec 2.2's 0.881 for real v5 latents is inside that
#   spread, so it stays.
#   header/ref 0.9963 +- 0.0044 [0.989, 1.005]; preamble/ref 0.9948 +- 0.0038.
#   The preamble and header are QPSK in a section with no Gaussian data, and
#   clip alike (header/preamble 1.0015 +- 0.0031); the references sit mostly
#   in the data section, at 1.006 of the unit-power baseband against the
#   preamble's 1.0005 in every frame. Angles within +-0.005 rad. G_PRE_L is
#   the preamble's own (`ltrack.make_classes_l` gives it G_REF_L, 0.5% off).
G_REF_L = 1.0
G_HDR_L = 0.996
G_DATA_L = 0.881
G_PRE_L = 0.995

# The pilot's gain through the clipper, against the unclipped pilot
# (`lwave.pilot(tau) / lwave.A_ID`), per section of the frame, each measured
# away from its edges (`lwave._pilot_sections`): preamble 0.6559-0.6560 (the
# same symbols in every frame), header 0.6640 +- 0.0026 [0.659, 0.669] (moves
# with the header's bits), data 0.7030 +- 0.0019 [0.699, 0.707] (with the
# picture); in the lead-in 1.0001 and in window units 8..183 1.0000, where the
# pilot runs alone at average power (W = 1) under the clip threshold. Angles
# against the references within +-0.01 rad.
#
# On the ramps between sections the clipper's gain follows the symbols'
# presence, not the weight W: the clipper only bites where data add to the
# pilot, and its filtered correction reaches about 5 symbols past the last
# pulse. Measured (16 L_MEDIUM frames, each also re-clipped with its symbols
# negated, the two averaged to cancel the data's linear term): at the start
# of window units 0, 1, 2, 3, 4, 5: 0.87, 0.94, 0.97, 0.98, 0.99, 1.00; at the
# end of units 191, 190, 189, 188, 187: 0.96, 0.97, 0.99, 0.99, 1.00; before
# the preamble after a lead-in, at tau = -1, -2, -3, -4, -5, -6: 0.92, 0.95,
# 0.97, 0.98, 0.995, 1.00. W g_ID + (1 - W) g_section reads 0.70 to 0.85 at
# window units 0..4 and 0.67 to 0.83 at tau -1..-4: up to 27% low there.
PILOT_GAIN_PRE = 0.656
PILOT_GAIN_HDR = 0.664
PILOT_GAIN_DATA = 0.703
PILOT_GAIN_ID = 1.0
# The references' own gain on the unit-average-power baseband: 1.0057 +- 0.0038
# [0.994, 1.011]. The data's pilot over it: 0.6991 +- 0.0031.
G_REF_ABS = 1.006

# Clipper distortion at the matched filter, per latent in l units (E|m/g - a|^2
# / E|a|^2 at the class's own LS gain g: `loopback_stats(...)["resid"]`):
#   data 0.0583 +- 0.0011 [0.056, 0.060] (12.34 dB; Gaussian 0.0590, tanh
#   0.0577), the same on I and Q and within 8 symbols of a window or a section
#   edge (0.054 +- 0.010) as away from them; at the fixed G_DATA_L x |gain_ref|
#   instead of the LS gain the picture's gain spread adds about 3e-4;
#   references 0.0236 +- 0.0010 [0.022, 0.025] (mostly in the data section);
#   header bits 0.0201 +- 0.0008 [0.019, 0.022]; preamble 0.0200 (every frame).
D_PASS_L = 0.058
D_REF_L = 0.0236
D_HDR_L = 0.0201
D_PRE_L = 0.0200
# The same on the x scale (y_x = sqrt(1 + rho) y_l) of a frame with the
# spread copy: D_PASS_L (1 + SPREAD_RHO). `d_pass_x(spec)` for any frame.
D_PASS_LX = D_PASS_L * (1.0 + SPREAD_RHO)


def d_pass_x(spec) -> float:
    """The data's clipper distortion on the frame's x scale: D_PASS_L (1 + rho)."""
    return D_PASS_L * (1.0 + float(spec.hdr_rho))


def _l_only(spec, what: str) -> None:
    if spec.waveform != WAVEFORM_L:
        raise AssertionError(f"{what} is L's; {spec.name!r} is waveform {spec.waveform}: "
                             f"use demod's")


# --- symbol estimates -----------------------------------------------------------------

def symbol_estimates_l(tr, stream_idx, gain: float) -> tuple[np.ndarray, np.ndarray]:
    """(r complex128, s2n float64) per stream symbol in `stream_idx`, uncalibrated.

    r = m conj(u) / (|u|^2 psi g), so a symbol a of class gain g arrives as
    r = a + noise; s2n is the noise part per latent dimension (the I and Q
    latents of a symbol share it). Erased symbols: r = 0, s2n = inf.
    """
    lay = lframe.layout(tr.spec)
    p = np.asarray(lay.pos)[np.asarray(stream_idx, dtype=np.int64)]
    gi = tr.gi(p)
    m, u, P, N, psi = (np.asarray(tr.m)[gi], np.asarray(tr.u)[gi], np.asarray(tr.P)[gi],
                       np.asarray(tr.N)[gi], np.asarray(tr.psi)[gi])
    u2 = np.abs(u) ** 2
    uu = np.abs(np.asarray(tr.u)[np.asarray(tr.pos) >= 0]) ** 2
    med = float(np.median(uu)) if len(uu) else 0.0
    ok = ~((psi < 0.5) | (u2 < U_MIN_REL * med) | (u2 <= 0))
    g = float(gain)
    r = np.zeros(len(u2), dtype=np.complex128)
    s2n = np.full(len(u2), np.inf, dtype=np.float64)
    # only the usable symbols are divided by their gain (see demod.symbol_estimates)
    r[ok] = m[ok] * np.conj(u[ok]) / (u2[ok] * psi[ok] * g)
    s2n[ok] = (N[ok] + g * g * P[ok]) / (g * g * u2[ok] * psi[ok] ** 2)
    return r, s2n


def _hdr_bits(tr):
    """The coded header bits the track was run with, or None while unknown."""
    bits = getattr(getattr(tr, "classes", None), "hdr_bits", None)
    return None if bits is None else np.asarray(bits)


def data_estimates_l(tr) -> tuple[np.ndarray, np.ndarray, float]:
    """(y_x, s2n_x, extra) per data latent (the precoded domain, x scale), uncalibrated.

    Header unknown (`tr.classes.hdr_bits` None) on a frame with the spread
    copy: the copy is rho of extra variance. Known: it is subtracted and
    extra is 0 (module docstring).
    """
    spec = tr.spec
    r, s2n = symbol_estimates_l(tr, lframe.layout(spec).data, G_DATA_L)
    y = lframe.unpair(r)
    s2n = np.repeat(s2n, 2)
    rho = float(spec.hdr_rho)
    if rho <= 0:
        return y, s2n, 0.0
    s = math.sqrt(1.0 + rho)
    y, s2n = s * y, (1.0 + rho) * s2n
    bits = _hdr_bits(tr)
    if bits is None:
        return y, s2n, rho
    y = y - math.sqrt(rho) * spread_signs(spec, bits)
    return np.where(np.isfinite(s2n), y, 0.0), s2n, 0.0


def header_estimates_l(tr) -> tuple[np.ndarray, np.ndarray]:
    """(y_h, s2n_h) float64[2474] in coded-bit order (+1 for bit 0), uncalibrated."""
    r, s2n = symbol_estimates_l(tr, lframe.layout(tr.spec).hdr_sym, G_HDR_L)
    return lframe.unpair(r), np.repeat(s2n, 2)


# --- calibration ------------------------------------------------------------------------

def _combine(ests) -> tuple[float, bool]:
    """`demod.calibrate`'s combination: inverse variance, shrunk to 1, clamped."""
    if not ests:
        return 1.0, False
    wsum = sum(1.0 / v for _, v in ests)
    k_hat = sum(k / v for k, v in ests) / wsum
    sd = 1.0 / np.sqrt(wsum)
    w0 = 1.0 / KAPPA_PRIOR_SD ** 2
    kappa = (k_hat * wsum + 1.0 * w0) / (wsum + w0)
    lo, hi = KAPPA_RANGE
    kappa = float(min(max(kappa, lo), hi))
    suspect = (k_hat < lo - 3 * sd) or (k_hat > hi + 3 * sd)
    return kappa, bool(suspect)


def _measured(tr, gi) -> np.ndarray:
    """bool per grid index: the data band's measurement there went into u
    (`tr.meas`, the tracker's mask; none without one, as on a genie)."""
    meas = getattr(tr, "meas", None)
    if meas is None:
        return np.zeros(len(gi), dtype=bool)
    return np.asarray(meas, dtype=bool)[gi]


def kappa_estimates_l(tr) -> list[tuple[float, float]]:
    """The (kappa, variance) estimates `calibrate_l` combines: the
    references' first (when there are 16 usable), then the data's (256).

    - references: e = m conj(u)/(|u|^2 psi) - a_ref, read per latent
      dimension (sqrt(2) Re e, sqrt(2) Im e); the template is the
      reference itself (class gain 1). Against CE's form, a reference the
      tracker measured pulled u towards its own noise: the smoothed
      residual of a measurement that went into u has variance R - P, not
      R + P, so E|e|^2 = kappa N_ + D_REF_L - P_ (N_, P_ the s2n's two
      parts). In L that pull is large (the pilot is 14% of the power
      against CE's 53% carrier, so the references carry much of u). On
      `ltrack.track_l` passes of l-short in moderate fading CE's form
      read 0.2-0.7 where the kappa that makes W honest was 0.9-1.2, and
      overstated W by up to 1.7 dB (11-14% on average over steady, quiet
      and moderate passes at -15 and -6 dB; this form, 0-2%). So
      kappa = (mean |e|^2 - D_REF_L + mean P_m) / mean(N_ + P_u), P_m
      the measured references' P_, P_u the others' (CE's form where
      nothing was measured, as at gate V or on a genie);
    - data: kappa = (mean y_x^2 - 1 - extra - D_x) / mean s2n_x. The data
      pull u by almost nothing: unknown in round A, and in round B only
      the spread copy's mean (5% of a symbol's power) is known. (A prior
      from EM would make them pull as the references do.)

    Both over symbols with s2n under max(KAPPA_S2_MAX, KAPPA_S2_REL x the
    data's median s2n_x), as CE's.
    """
    spec = tr.spec
    _l_only(spec, "ldemod.kappa_estimates_l")
    lay = lframe.layout(spec)
    ests = []
    s2_max = KAPPA_S2_MAX
    yd, s2d, extra = np.zeros(0), np.zeros(0), 0.0
    dx = d_pass_x(spec)
    if spec.n_data:
        yd, s2d, extra = data_estimates_l(tr)
        fin = s2d[np.isfinite(s2d)]
        if len(fin):
            s2_max = max(KAPPA_S2_MAX, KAPPA_S2_REL * float(np.median(fin)))
    ref = np.asarray(lay.ref)
    if len(ref) >= 16:
        r, s2n = symbol_estimates_l(tr, ref, G_REF_L)
        a = np.asarray(lay.known_val)[np.searchsorted(lay.known_idx, ref)]
        ok = np.isfinite(s2n) & (s2n < s2_max)
        if ok.sum() >= 16:
            gi = tr.gi(np.asarray(lay.pos)[ref])[ok]
            P_ = np.asarray(tr.P)[gi] / (np.abs(np.asarray(tr.u)[gi]) ** 2
                                         * np.asarray(tr.psi)[gi] ** 2)
            meas = _measured(tr, gi)
            p_m = float(np.mean(np.where(meas, P_, 0.0)))
            den = float(np.mean(s2n[ok])) - p_m          # mean(N_ + P_u)
            if den > 0:
                k = (float(np.mean(np.abs(r[ok] - a[ok]) ** 2)) - D_REF_L + p_m) / den
                res = max(den - p_m, 0.0) + D_REF_L       # E|e|^2 at kappa 1
                var = 2.0 * res ** 2 / (2 * ok.sum()) / den ** 2
                ests.append((k, var))
    if spec.n_data >= 256:
        ok = np.isfinite(s2d) & (s2d < s2_max)
        if ok.sum() >= 256:
            den = float(np.mean(s2d[ok]))
            if den > 0:
                k = (float(np.mean(yd[ok] ** 2)) - 1.0 - extra - dx) / den
                var = 2.0 * (1.0 + extra + dx + den) ** 2 / ok.sum() / den ** 2
                ests.append((k, var))
    return ests


def calibrate_l(tr) -> tuple[float, bool]:
    """(kappa, suspect): `demod.calibrate` for an L pass, on the two
    estimates of `kappa_estimates_l` (module docstring)."""
    _l_only(tr.spec, "ldemod.calibrate_l")
    return _combine(kappa_estimates_l(tr))


# --- extract ----------------------------------------------------------------------------

def extract_l(m, track, spec, q: int, estimator: str = "joint", *,
              calibrate_kappa: bool = True):
    """(z, w, llr, diag) of one L pass: `demod.extract`'s contract.

    m: the matched-filter outputs (None: the track's own, as demod.extract);
    track: an L track result (`ltrack.LTrackResult`; read: spec, pos, gi, m,
    u, P, N, psi, classes.hdr_bits). z, w float32[n_data] in air order; llr
    float32[2474] (positive means bit 0; length 0 without a header). diag
    holds kappa, suspect, y and s2 of the data per latent in the precoded
    domain (x scale), the header symbols' y (coded-bit order), and the
    block and spread parts of llr (None where absent).
    """
    _l_only(spec, "ldemod.extract_l")
    if m is not None and m is not track.m:
        track.m = np.asarray(m)
    kappa, suspect = calibrate_l(track) if calibrate_kappa else (1.0, False)
    scale = kappa if not suspect else 1.0
    dx = d_pass_x(spec)
    yd, s2n_d, extra = data_estimates_l(track)
    s2d = s2n_d * scale + dx + extra
    z, v = block_estimate(yd, s2d, q, estimator)
    with np.errstate(divide="ignore"):
        w = np.where(np.isfinite(v) & (v > 0), 1.0 / v, 0.0)
    z = np.where(w > 0, z, 0.0)
    llr = np.zeros(N_HDR_BITS if spec.has_header else 0, dtype=np.float64)
    yh = np.zeros(0)
    llr_block = llr_spread = None
    if spec.n_hdr:
        yh, s2n_h = header_estimates_l(track)
        s2h = s2n_h * scale + D_HDR_L
        with np.errstate(divide="ignore", invalid="ignore"):
            llr_block = np.where(np.isfinite(s2h), 2.0 * yh / s2h, 0.0)
        llr = llr + llr_block
    if spec.hdr_rho > 0 and extra > 0:
        # header not known yet: the data symbols carry it too. (Once it is
        # known data_estimates_l has subtracted it, and these y hold none.)
        llr_spread = spread_llr(yd, s2n_d * scale + dx, spec)
        llr = llr + llr_spread
    llr = llr.astype(np.float32)
    diag = dict(kappa=float(kappa), suspect=bool(suspect), y=yd, s2=s2d, y_hdr=yh,
                estimator=estimator, llr_block=llr_block, llr_spread=llr_spread)
    return z.astype(np.float32), w.astype(np.float32), llr, diag

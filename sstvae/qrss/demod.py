"""Symbol extraction, weights, the plain and joint block estimators and
header LLRs (design 6.6, WP6).

Not a format module. For every non-reference stream symbol k at
position p, from the tracker's matched-filter output m_p, smoothed gain
u_hat_p (error variance P_p), noise N_p and unblanked share psi_p:

    y_k  = Im(m_p conj(u_p)) / (K |u_p|^2 psi_p)                 (E[y | x] = x)
    s2_k = [N_p/2 + (A0^2 + K^2) P_p/2] / (K^2 |u_p|^2 psi_p^2) + D_PASS

The halves are there because only the imaginary dimension carries the
symbol. A symbol is erased (s2 = inf) where psi_p < 0.5 or |u_p|^2 is
under 1e-4 of its median. D_PASS is the per-pass phase-modulation
distortion in latent units, measured on Gaussian latents with
`ce.loopback_stats` and pinned below.

**Calibration.** On the reference symbols the residual against the
template's own expectation (Im(c_p)/K: a +-1 symbol comes through the
phase modulator with a larger gain than a Gaussian one) gives
kappa = (mean residual^2 - D_REF) / mean(noise part of s2). With kappa
in [0.5, 2] the *noise part* of every s2 is scaled by it; outside, the
pass is flagged suspect and left alone. (The design scales the whole s2;
the distortion part is a pinned property of the modulator, not of this
pass's noise, and scaling it would make a strong pass's W wrong. That is
the one deviation here.)

**Data estimators**, per precoder block of M in {64, 32, 8} symbols,
c the pass's scrambler signs:

- plain (spec 7): a = c * WHT(y), v = mean(s2) (an erased symbol enters
  as y = 0 with s2 = 1, its prior);
- joint (spec 2.4's 64 x 64 solve, closed form, the default): with
  d = 1/(1 + s2) and g = mean(d), a = c * WHT(d y)/g and
  v = [mean(d^2 s2) + mean((d - g)^2)]/g^2, the second term being the
  crosstalk a non-flat block leaves. Equal to plain on a flat block.

w = 1/v for every latent of the block (float32): W is the per-latent SNR.
Header LLRs are 2 y/s2 on the coded header symbols (positive means bit 0).
"""

from __future__ import annotations

import numpy as np

from .constants import A0, K_LIN
from .frame import FrameSpec, layout
from .precoder import block_index, block_sizes, wht
from .sequences import scrambler

# Per-pass phase-modulation distortion of a data symbol, latent units:
# mean((Im m / gain - x)^2) / E x^2 over noiseless genie loopbacks
# (`ce.loopback_stats`), MEDIUM and FULL frames, Gaussian and tanh latents,
# 24 frames: 0.0428 to 0.0516, mean 0.0466 (13.3 dB below the latents).
D_PASS = 0.0466
# The same for the +-1 reference symbols, after their template expectation.
D_REF = 0.009
KAPPA_RANGE = (0.5, 2.0)
KAPPA_PRIOR_SD = 0.5
KAPPA_S2_MAX = 4.0         # symbols in deep fades (s2n above this, and above
KAPPA_S2_REL = 4.0         # this many times the pass's median) say little about kappa
U_MIN_REL = 1e-4


def symbol_estimates(tr, stream_idx) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(y, s2 noise part, s2 total) for stream symbols `stream_idx`, uncalibrated."""
    spec = tr.spec
    lay = layout(spec)
    p = np.asarray(lay.pos)[np.asarray(stream_idx, dtype=np.int64)]
    g = tr.gi(p)
    m, u, P, N, psi = tr.m[g], tr.u[g], tr.P[g], tr.N[g], tr.psi[g]
    u2 = np.abs(u) ** 2
    med = float(np.median(np.abs(tr.u[tr.pos >= 0]) ** 2)) if len(tr.u) else 0.0
    bad = (psi < 0.5) | (u2 < U_MIN_REL * med) | (u2 <= 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        y = np.imag(m * np.conj(u)) / (K_LIN * u2 * psi)
        s2n = (N / 2 + (A0 ** 2 + K_LIN ** 2) * P / 2) / (K_LIN ** 2 * u2 * psi ** 2)
    y = np.where(bad, 0.0, y)
    s2n = np.where(bad, np.inf, s2n)
    return y, s2n, s2n + D_PASS


def calibrate(tr) -> tuple[float, bool]:
    """(kappa, suspect) from the residuals of the pass's own symbols (design 6.6).

    Two estimates of kappa, combined by inverse variance:

    - references: residual against the template's expectation Im(c_p)/K
      (a +-1 symbol comes through the modulator with a larger gain than a
      Gaussian one), kappa = (mean res^2 - D_REF) / mean(noise part);
    - data: the latents are unit-RMS by contract and the precoder is
      orthonormal, so E y^2 = 1 + D_PASS + kappa s2n, kappa =
      (mean y^2 - 1 - D_PASS) / mean s2n.

    Both over symbols with s2n < max(KAPPA_S2_MAX, KAPPA_S2_REL x the
    pass's median s2n): in a deep fade the gain estimate's own error
    dominates and is not what kappa scales. The relative bound keeps a
    weak pass (below about -23 dB every symbol has s2n > 4) calibrated
    instead of excluding all of it.

    A pass with too few symbols for either estimate keeps kappa = 1 and
    is not suspect: suspect means kappa measured outside KAPPA_RANGE.

    The first is the better one on a strong pass (its spread scales with
    s2n), the second on a weak one (4096+ symbols against ~500 references).
    """
    spec = tr.spec
    lay = layout(spec)
    ests = []
    ref = np.asarray(lay.ref)
    s2_max = KAPPA_S2_MAX
    yd, s2d = np.zeros(0), np.zeros(0)
    if spec.n_data:
        yd, s2d, _ = symbol_estimates(tr, lay.data)
        fin = s2d[np.isfinite(s2d)]
        if len(fin):
            s2_max = max(KAPPA_S2_MAX, KAPPA_S2_REL * float(np.median(fin)))
    if len(ref) >= 16:
        y, s2n, _ = symbol_estimates(tr, ref)
        g = tr.gi(np.asarray(lay.pos)[ref])
        expect = tr.c[g].imag / K_LIN          # what a known +-1 symbol comes through as
        ok = np.isfinite(s2n) & (s2n < s2_max)
        if ok.sum() >= 16:
            den = float(np.mean(s2n[ok]))
            if den > 0:
                k = (float(np.mean((y[ok] - expect[ok]) ** 2)) - D_REF) / den
                var = 2.0 * (den + D_REF) ** 2 / ok.sum() / den ** 2
                ests.append((k, var))
    if spec.n_data >= 256:
        y, s2n = yd, s2d
        ok = np.isfinite(s2n) & (s2n < s2_max)
        if ok.sum() >= 256:
            den = float(np.mean(s2n[ok]))
            if den > 0:
                k = (float(np.mean(y[ok] ** 2)) - 1.0 - D_PASS) / den
                var = 2.0 * (1.0 + D_PASS + den) ** 2 / ok.sum() / den ** 2
                ests.append((k, var))
    if not ests:
        return 1.0, False
    wsum = sum(1.0 / v for _, v in ests)
    k_hat = sum(k / v for k, v in ests) / wsum
    sd = 1.0 / np.sqrt(wsum)
    # shrunk towards 1 (prior sd KAPPA_PRIOR_SD): on a strong pass the noise
    # part is a small share of s2 and its scale is barely observable
    w0 = 1.0 / KAPPA_PRIOR_SD ** 2
    kappa = (k_hat * wsum + 1.0 * w0) / (wsum + w0)
    lo, hi = KAPPA_RANGE
    kappa = float(min(max(kappa, lo), hi))
    suspect = (k_hat < lo - 3 * sd) or (k_hat > hi + 3 * sd)
    return kappa, bool(suspect)


def block_estimate(y, s2, q: int, estimator: str = "joint") -> tuple[np.ndarray, np.ndarray]:
    """(z, v): air-order latents and their variance from precoded-domain y, s2."""
    y = np.asarray(y, dtype=np.float64)
    s2 = np.asarray(s2, dtype=np.float64)
    n = len(y)
    c = scrambler(q, n).astype(np.float64)
    bi = block_index(n)
    sizes = np.array(block_sizes(n), dtype=np.float64)
    erased = ~np.isfinite(s2)
    if estimator == "plain":
        yy = np.where(erased, 0.0, y)
        ss = np.where(erased, 1.0, s2)
        a = c * wht(yy)
        v = np.bincount(bi, ss, minlength=len(sizes)) / sizes
    elif estimator == "joint":
        d = np.where(erased, 0.0, 1.0 / (1.0 + np.where(erased, 0.0, s2)))
        yy = np.where(erased, 0.0, y)
        gam = np.bincount(bi, d, minlength=len(sizes)) / sizes
        ds2 = np.where(erased, 0.0, d * d * np.where(erased, 0.0, s2))
        cross = (d - gam[bi]) ** 2
        num = (np.bincount(bi, ds2, minlength=len(sizes))
               + np.bincount(bi, cross, minlength=len(sizes))) / sizes
        with np.errstate(divide="ignore", invalid="ignore"):
            v = np.where(gam > 0, num / gam ** 2, np.inf)
            a = c * wht(d * yy) / np.where(gam[bi] > 0, gam[bi], 1.0)
        a = np.where(gam[bi] > 0, a, 0.0)
    else:
        raise ValueError(f"estimator must be 'joint' or 'plain', not {estimator!r}")
    return a, v[bi]


def extract(m, track, spec: FrameSpec, q: int, estimator: str = "joint", *,
            calibrate_kappa: bool = True):
    """(z, w, llr, diag) of one pass (design 6.6).

    m: the matched-filter outputs (None: the track's own); track: a
    `track.TrackResult`. z, w float32[n_data] in air order (unscrambled,
    unprecoded); llr float32[2474] (zeros without a header). diag holds
    kappa, suspect, y and s2 of the data in the precoded domain, and the
    header symbols' y.
    """
    if m is not None and m is not track.m:
        track.m = np.asarray(m)
    lay = layout(spec)
    kappa, suspect = calibrate(track) if calibrate_kappa else (1.0, False)
    scale = kappa if not suspect else 1.0
    yd, s2n_d, _ = symbol_estimates(track, lay.data)
    s2d = s2n_d * scale + D_PASS
    z, v = block_estimate(yd, s2d, q, estimator)
    with np.errstate(divide="ignore"):
        w = np.where(np.isfinite(v) & (v > 0), 1.0 / v, 0.0)
    z = np.where(w > 0, z, 0.0)
    llr = np.zeros(len(lay.hdr_bits), dtype=np.float32)
    yh = np.zeros(0)
    if spec.n_hdr:
        yh, s2n_h, _ = symbol_estimates(track, lay.hdr_bits)
        s2h = s2n_h * scale + D_PASS
        with np.errstate(divide="ignore", invalid="ignore"):
            llr = np.where(np.isfinite(s2h), 2.0 * yh / s2h, 0.0).astype(np.float32)
    diag = dict(kappa=float(kappa), suspect=bool(suspect), y=yd, s2=s2d, y_hdr=yh,
                estimator=estimator)
    return z.astype(np.float32), w.astype(np.float32), llr, diag

#!/usr/bin/env python3
"""Is a 2-D LMMSE channel estimate worth it, and is changing the latent
weights to match it worth a stage-2 fine-tune? (Data2G "port back" item.)

Re-equalizes the receiver's own frame pass four ways, a 2x2 of
{Catmull-Rom, LMMSE} channel estimate x {today's |h|/median weights,
per-latent MMSE weights}:

    CR/old   what the modem did before 2026-09-22 (the modem now
             ships a refined LM/old, `_lmmse_channel`)
    LM/old   receiver-only: better h, same weight semantics
    CR/mmse  weights only
    LM/mmse  both

    python scripts/lmmse_study.py snr --trials 32        # no codec
    python scripts/lmmse_study.py psnr --images 12       # current decoder

`snr` scores `latents * weights` as the SNR of the best linear fit,
like scripts/rx_ab.py. The /mmse columns there are a *ceiling* on what
a fine-tune could collect, and latent-domain numbers have overpredicted
end-to-end gains here by 2-2.5x before. `psnr` runs the current decoder,
which was trained on the old weights, so its /mmse column is a floor.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sstvae import hfchannel  # noqa: E402
from sstvae.config import (  # noqa: E402
    FRAME_SAMPLES, FS, MODES, NC, NC_LATENT, SYMS_PER_FRAME,
)
from sstvae.modem import Modem, SyncError, framing  # noqa: E402
from sstvae.modem import modem as modem_mod  # noqa: E402

FRAME_S = FRAME_SAMPLES / FS
BB = modem_mod._BB_FREQS
TIME_TAPS = 4
GAIN = 0.772  # pilot-to-data gain the clipper leaves (CLAUDE.md)
KAPPA = 10 ** (-15.19 / 10)  # clip self-noise relative to signal

CONDS = [
    ("awgn3", 3.0, None), ("mpg6", 6.0, "mpg"), ("mps6", 6.0, "mps"),
    ("mpp8", 8.0, "mpp"), ("mpp3", 3.0, "mpp"), ("mpd8", 8.0, "mpd"),
]


# --- channel estimates -----------------------------------------------------

def cr(hp, rcv):
    """(n_f, 5, NC) Catmull-Rom, with the modem's edge fallbacks."""
    n_f = len(hp)

    def at(i, fb):
        return hp[i] if 0 <= i < n_f and rcv[i] else hp[fb]

    u = (np.arange(1, SYMS_PER_FRAME) / SYMS_PER_FRAME)[:, None]
    h = np.zeros((n_f, SYMS_PER_FRAME - 1, NC), complex)
    for f in np.flatnonzero(rcv):
        p0, p1, p2 = at(f - 1, f), hp[f], at(f + 1, f)
        p3 = at(f + 2, f + 1 if f + 1 < n_f and rcv[f + 1] else f)
        h[f] = 0.5 * (2 * p1 + (p2 - p0) * u + (2 * p0 - 5 * p1 + 4 * p2 - p3) * u**2
                      + (3 * p1 - p0 - 3 * p2 + p3) * u**3)
    return h


def lmmse(hp, rcv):
    """Data2G's equalizer.estimate: projection onto the measured delay
    support across carriers, Wiener interpolation in time with a
    Gaussian Doppler correlation measured from the pilots. Returns h,
    its MSE and the thermal noise per carrier (h units)."""
    n_p = int(rcv.sum())  # frames are a prefix: the buffer ran out
    hp = hp[:n_p]
    d0, d1 = modem_mod._delay_support(hp)
    d = np.arange(d0 - 4, d1 + 5)
    U, s, _ = np.linalg.svd(np.exp(-2j * np.pi * np.outer(BB, d) / FS), full_matrices=False)
    r = int(np.sum(s > s[0] * 1e-2))
    U = U[:, :r]
    hs = (hp @ np.conj(U)) @ U.T
    n0 = float(np.mean(np.abs(hp - hs) ** 2) * NC / max(NC - r, 1))
    n0_s = n0 * r / NC
    p_sig = max(float(np.mean(np.abs(hs) ** 2) - n0_s), 1e-12)
    rho = np.abs(np.mean(hs[1:] * np.conj(hs[:-1]))) / p_sig
    rho = float(np.clip(rho, 1e-3, 0.9999))
    spread = float(np.clip(2 * np.sqrt(-np.log(rho) / 2) / (np.pi * FRAME_S), 0.02, 4.0))

    def corr(dt):
        return np.exp(-2 * (np.pi * spread / 2 * dt) ** 2)

    t_p = np.arange(n_p) * FRAME_S
    offs = np.arange(1, SYMS_PER_FRAME) / SYMS_PER_FRAME * FRAME_S
    h = np.zeros((len(rcv), SYMS_PER_FRAME - 1, NC), complex)
    mse = np.zeros(h.shape)
    k = min(2 * TIME_TAPS, n_p)
    for f in range(n_p):
        lo = max(0, min(f - TIME_TAPS + 1, n_p - k))
        j = np.arange(lo, lo + k)
        Rpp = p_sig * corr(t_p[j, None] - t_p[None, j]) + n0_s * np.eye(k)
        Rdp = p_sig * corr(f * FRAME_S + offs[:, None] - t_p[None, j])
        W = np.linalg.solve(Rpp, Rdp.T).T
        h[f] = W @ hs[j]
        mse[f] = np.maximum(p_sig - np.real(np.sum(W * np.conj(Rdp), axis=1)), 0)[:, None]
    return h, mse, n0


# --- equalize + weight -----------------------------------------------------

def equalize(raw, hp, rcv, spec, h, noise):
    """Latents and weights, canonical order. `noise` None = today's
    min(|h|/median, 1); otherwise per-cu noise variance (h units) for
    per-latent MMSE weights."""
    med = np.median(np.abs(hp[rcv]))
    mag = np.maximum(np.abs(h), max(0.05 * med, 1e-9))
    y = raw[:, 1:] * np.conj(h) / mag**2
    if noise is None:
        w = np.minimum(np.abs(h) / med, 1.0)
    else:
        w = 1.0 / (1 + KAPPA + noise / (GAIN**2 * mag**2))
    w = np.where(rcv[:, None, None], w, 0.0)
    n_f = len(rcv)
    lat = framing.symbols_to_slots(y[:, :, :NC_LATENT]).reshape(n_f, -1)
    wt = np.repeat(w[:, :, :NC_LATENT], 2, axis=-1).reshape(n_f, -1)
    lat = np.clip(lat, -10, 10).reshape(-1)[: spec.n_tx_latents]
    wt = wt.reshape(-1)[: spec.n_tx_latents]
    return framing.deinterleave(lat, spec)[0], framing.deinterleave(wt, spec)[0]


def variants(modem, rx):
    got = {}
    real = modem._demod_frames

    def spy(*a, **k):
        got["out"] = real(*a, **k)
        return got["out"]

    modem._demod_frames = spy
    try:
        r = modem.demodulate(rx)
    finally:
        del modem._demod_frames
    raw, hp, rcv, _ = got["out"]
    spec = r.mode
    h_cr = cr(hp, rcv)
    h_lm, mse, n0 = lmmse(hp, rcv)
    # Catmull-Rom's own error is unmodelled; give it the pilots' noise,
    # the one term both estimates share.
    out = {
        "CR/old": equalize(raw, hp, rcv, spec, h_cr, None),
        "LM/old": equalize(raw, hp, rcv, spec, h_lm, None),
        "CR/mmse": equalize(raw, hp, rcv, spec, h_cr, n0 + np.zeros(h_cr.shape)),
        "LM/mmse": equalize(raw, hp, rcv, spec, h_lm, n0 + mse),
    }
    return spec, out


def fit_snr(t, g):
    rho2 = np.dot(t, g) ** 2 / (np.dot(t, t) * np.dot(g, g) + 1e-30)
    return 10 * np.log10(rho2 / max(1 - rho2, 1e-12))


def report(name, rows):
    keys = list(rows[0])
    m = {k: np.mean([r[k] for r in rows]) for k in keys}
    base = np.array([r["CR/old"] for r in rows])
    line = f"{name:8s} n={len(rows):2d}  CR/old {m['CR/old']:6.2f}"
    for k in keys[1:]:
        d = np.array([r[k] for r in rows]) - base
        line += f"  {k} {d.mean():+.2f} ({(d > 0).sum()}/{len(d)})"
    print(line, flush=True)


def run_snr(a):
    modem, spec = Modem(), MODES[a.mode]
    for name, snr, fade in CONDS:
        rows = []
        for seed in range(a.trials):
            rng = np.random.default_rng(1000 + seed)
            lat = rng.normal(size=spec.n_latents)
            lat /= np.sqrt(np.mean(lat**2))
            tx = modem.modulate(lat, a.mode)
            if seed == 0:
                mask = modem.demodulate(tx).weights[: spec.n_latents] > 0
            rx = hfchannel.apply_channel(tx, snr_db=snr, fading_preset=fade, seed=seed)
            try:
                _, out = variants(modem, rx)
            except SyncError:
                continue
            rows.append({k: fit_snr(lat[mask], (lt * w)[: spec.n_latents][mask])
                         for k, (lt, w) in out.items()})
        report(name, rows)


def run_psnr(a):
    from sstvae.codec import load_codec, pad_to_full, reconstruct
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from snr_sweep import DEFAULT_IMAGES, load_images, psnr

    codec = load_codec(a.model)
    images = load_images(Path(a.source or DEFAULT_IMAGES), a.images)
    modem = Modem()
    for name, snr, fade in CONDS:
        rows = []
        for i, img in enumerate(images):
            lat = codec.encode(img)[: MODES[a.mode].n_latents]
            tx = modem.modulate(lat, a.mode)
            rx = hfchannel.apply_channel(tx, snr_db=snr, fading_preset=fade, seed=i)
            try:
                _, out = variants(modem, rx)
            except SyncError:
                continue
            rows.append({k: psnr(img, reconstruct(codec, pad_to_full(lt), pad_to_full(w)))
                         for k, (lt, w) in out.items()})
        report(name, rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("part", choices=["snr", "psnr"])
    ap.add_argument("--mode", default="A")
    ap.add_argument("--trials", type=int, default=32)
    ap.add_argument("--images", type=int, default=12)
    ap.add_argument("--source")
    ap.add_argument("--model")
    a = ap.parse_args()
    (run_snr if a.part == "snr" else run_psnr)(a)


if __name__ == "__main__":
    main()

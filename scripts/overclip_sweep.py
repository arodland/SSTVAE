#!/usr/bin/env python3
"""CESSB 'more than clipping' applied to the SSTVAE clip-and-filter.

Overshoot the clip correction so the post-filter peak lands nearer the
threshold: out = x + k*(clipped - x), k in [1, 2].  Two forms:

  linear : eff_scale = max(0, 1 - k*(1-scale))   -- CESSB as written
  power  : eff_scale = scale**k                  -- same to first order
           near threshold, but stays positive at deep overshoot, which
           is the role Hershberger's nonlinear gain plays.  The linear
           form inverts the envelope once scale < 1 - 1/k, and at this
           clipper's ~40% first-iteration duty that is a large fraction
           of the waveform.

k is a *schedule*, one value per clip iteration ("1,1.5" = no overshoot
on the first, 1.5 on the second), because the two iterations are not
alike: measured at headroom 0, the bandpass passes 87% of the first
iteration's correction and only 58% of the second's, and overshoot is
only compensating for what the filter removes.

Measures per (headroom, schedule, form), through the real modem on a
NOISELESS channel: post-clip envelope PAPR, and clipping self-noise on
the transmitted latents only -- the dropped latents alone put a 13.8 dB
floor on that and read exactly like a modem impairment.  Combines the
two PEP-fair: on a peak-limited transmitter a lower PAPR is average
power, so each configuration is credited its own PAPR advantage as
channel SNR.

    python scripts/overclip_sweep.py --trials 5 \
        --headroom-db=-1,0,1,2 --schedules 1,1;1,1.5;1.25,1.5
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy import signal

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sstvae import config
from sstvae.config import MODES
from sstvae.modem import framing
from sstvae.modem import modem as M
from sstvae.modem.modem import Modem

FS = config.FS
_TAPS = signal.firwin(201, config.TX_BANDPASS, fs=FS, pass_zero=False)


def tx_condition_k(x, clip_headroom_db, schedule=(1.0, 1.0), form="power"):
    """`dsp.tx_condition` with a per-iteration overshoot factor."""
    power = np.mean(x**2)
    if power == 0:
        return x
    thresh = np.sqrt(2 * power) * 10 ** (clip_headroom_db / 20)
    for k in schedule:
        z = signal.hilbert(x)
        scale = np.minimum(1.0, thresh / np.maximum(np.abs(z), 1e-12))
        if k != 1.0:
            if form == "linear":
                scale = np.maximum(0.0, 1.0 - k * (1.0 - scale))
            elif form == "power":
                scale = scale**k
            else:
                raise ValueError(form)
        x = np.convolve(np.real(z * scale), _TAPS, mode="same")
    return x / np.sqrt(np.mean(x**2))


def papr_db(x):
    env2 = np.abs(signal.hilbert(x)) ** 2
    return 10 * np.log10(np.max(env2) / np.mean(env2))


def latent_snr(ref, rx, idx):
    """Gain-fitted, because a systematic gain is what the confidence
    weights already absorb -- see the 0.794 latent gain in CLAUDE.md."""
    a, b = ref[idx], rx[idx]
    g = float(np.dot(a, b) / np.dot(a, a))
    return 10 * np.log10(np.sum(a**2) / np.sum((b - g * a) ** 2)), g


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", default="A")
    p.add_argument("--headroom-db", default="-1,-0.5,0,0.5,1,1.5,2,2.5,3")
    p.add_argument(
        "--schedules",
        default="1,1;1,1.25;1,1.5;1,1.75;1,2;1.25,1.25;1.25,1.5;1.5,1.5;1.5,1.75",
        help="semicolon-separated per-iteration k lists",
    )
    p.add_argument("--forms", default="power")
    p.add_argument("--report-snr-db", default="0,4,8,12,16")
    p.add_argument("--trials", type=int, default=5)
    p.add_argument("--csv")
    args = p.parse_args()

    spec = MODES[args.mode]
    m = Modem()
    slots = framing.interleave(np.arange(spec.n_latents, dtype=np.float64), spec)
    idx = np.unique(slots[: spec.n_frames * config.LATENTS_PER_FRAME].astype(int))

    rng = np.random.default_rng(0)
    lat = [rng.standard_normal(spec.n_latents) for _ in range(args.trials)]
    lat = [l / np.sqrt(np.mean(l**2)) for l in lat]

    scheds = [tuple(float(v) for v in s.split(",")) for s in args.schedules.split(";")]
    rows = []
    for form in args.forms.split(","):
        for hd in [float(v) for v in args.headroom_db.split(",")]:
            for sch in scheds:
                if all(k == 1.0 for k in sch) and form != args.forms.split(",")[0]:
                    continue
                paprs, snrs, gains = [], [], []
                for l in lat:
                    orig = M.tx_condition
                    M.tx_condition = (
                        lambda x, _h, hd=hd, sch=sch, form=form: tx_condition_k(
                            x, hd, sch, form
                        )
                    )
                    try:
                        wav = m.modulate(l, spec)
                    finally:
                        M.tx_condition = orig
                    paprs.append(papr_db(wav))
                    res = m.demodulate(wav)
                    rx = np.asarray(res.latents) * np.asarray(res.weights)
                    s, g = latent_snr(l, rx, idx)
                    snrs.append(s)
                    gains.append(g)
                rows.append(
                    dict(
                        form=form,
                        hd=hd,
                        sch=sch,
                        papr=float(np.mean(paprs)),
                        papr_sd=float(np.std(paprs)),
                        snr=float(np.mean(snrs)),
                        snr_sd=float(np.std(snrs)),
                        gain=float(np.mean(gains)),
                    )
                )
                print(
                    f"  ...{form} hd={hd} k={sch}: papr={rows[-1]['papr']:.2f} "
                    f"clipSNR={rows[-1]['snr']:.2f}+-{rows[-1]['snr_sd']:.2f} "
                    f"gain={rows[-1]['gain']:.3f}",
                    flush=True,
                )

    ref = [r for r in rows if r["hd"] == 0.0 and all(k == 1.0 for k in r["sch"])][0]
    snrs_report = [float(v) for v in args.report_snr_db.split(",")]

    def eff(r, S):
        chan = S + ref["papr"] - r["papr"]
        return -10 * np.log10(10 ** (-r["snr"] / 10) + 10 ** (-chan / 10))

    print(
        f"\nreference (shipping): hd={ref['hd']} k={ref['sch']} "
        f"papr={ref['papr']:.2f} clipSNR={ref['snr']:.2f} gain={ref['gain']:.3f}"
    )
    print(
        "\nPEP-fair effective SNR delta vs reference, at each nominal SNR\n"
        "(channel SNR = nominal + papr_ref - papr; noise = clip + channel)\n"
    )
    hdr = f"{'form':6} {'hd':>5} {'k':>12} {'PAPR':>6} {'clipSNR':>8} {'gain':>6}"
    hdr += "".join(f"{('@' + str(int(s))):>7}" for s in snrs_report)
    print(hdr)
    rows.sort(key=lambda r: -eff(r, 8.0))
    for r in rows:
        line = (
            f"{r['form']:6} {r['hd']:5.1f} {','.join(f'{k:g}' for k in r['sch']):>12} "
            f"{r['papr']:6.2f} {r['snr']:8.2f} {r['gain']:6.3f}"
        )
        line += "".join(f"{eff(r, s) - eff(ref, s):+7.2f}" for s in snrs_report)
        print(line)

    if args.csv:
        import csv

        with open(args.csv, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["form", "hd", "k", "papr", "papr_sd", "clip_snr", "clip_snr_sd", "gain"]
                       + [f"delta@{s:g}" for s in snrs_report])
            for r in rows:
                w.writerow([r["form"], r["hd"], "|".join(f"{k:g}" for k in r["sch"]),
                            f"{r['papr']:.4f}", f"{r['papr_sd']:.4f}",
                            f"{r['snr']:.4f}", f"{r['snr_sd']:.4f}", f"{r['gain']:.4f}"]
                           + [f"{eff(r, s) - eff(ref, s):.4f}" for s in snrs_report])
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()

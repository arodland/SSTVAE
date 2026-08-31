#!/usr/bin/env python3
"""End-to-end PEP-fair gate for alternative TX clipper settings.

`scripts/overclip_sweep.py` scores clipper candidates on a *proxy* --
latent SNR through a noiseless clip, with Gaussian latents.  This runs
the real thing: real images, the real codec, the real modem, and a real
channel, scored in PSNR.

PEP-fair means each clipper is credited the average power its own PAPR
buys on a peak-limited transmitter, so a candidate with 0.55 dB less
PAPR is run at 0.55 dB more channel SNR.  Without that the comparison
is meaningless -- lower PAPR always costs quality at a fixed *average*
power, and the whole point of clipping is that average is not what the
transmitter is limited by.

A clipper is "HEADROOM:K1,K2,..." -- the envelope clip headroom in dB,
and one overshoot factor per clip-and-filter pass (1.0 = plain clip;
see overclip_sweep.py for what the overshoot is and why it is a power
rather than the linear CESSB form).  The first is the reference.

    python scripts/overclip_e2e.py --clippers "0:1,1;1:1,1.5,2"
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy import signal

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sstvae import hfchannel  # noqa: E402
from sstvae.codec import MODEL_HELP, load_codec  # noqa: E402
from sstvae.config import MODES, SNR_REF_BW_HZ, TX_BANDPASS, FS  # noqa: E402
from sstvae.modem import Modem  # noqa: E402
from sstvae.modem import modem as M  # noqa: E402

from snr_sweep import DEFAULT_IMAGES, load_images, run_point  # noqa: E402

_TAPS = signal.firwin(201, TX_BANDPASS, fs=FS, pass_zero=False)


def tx_condition_k(x, headroom_db, schedule):
    power = np.mean(x**2)
    if power == 0:
        return x
    thresh = np.sqrt(2 * power) * 10 ** (headroom_db / 20)
    for k in schedule:
        z = signal.hilbert(x)
        scale = np.minimum(1.0, thresh / np.maximum(np.abs(z), 1e-12))
        if k != 1.0:
            scale = scale**k
        x = np.convolve(np.real(z * scale), _TAPS, mode="same")
    return x / np.sqrt(np.mean(x**2))


def papr_db(x):
    env2 = np.abs(signal.hilbert(x)) ** 2
    return 10 * np.log10(np.max(env2) / np.mean(env2))


def parse_clipper(text):
    hd, _, ks = text.partition(":")
    return float(hd), tuple(float(v) for v in ks.split(","))


def modulate_all(modem, latents, spec, hd, sched):
    orig = M.tx_condition
    M.tx_condition = lambda x, _h: tx_condition_k(x, hd, sched)
    try:
        return [modem.modulate(lat[: spec.n_latents], spec) for lat in latents]
    finally:
        M.tx_condition = orig


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--model", default=None, help=MODEL_HELP)
    ap.add_argument("--clippers", default="0:1,1;1:1,1.5,2")
    ap.add_argument("--images", type=Path, default=DEFAULT_IMAGES)
    ap.add_argument("--n-images", type=int, default=12)
    ap.add_argument("--modes", default="B")
    ap.add_argument("--conditions", default="awgn,mpp")
    ap.add_argument("--snr-db", default="12,8,4,0")
    ap.add_argument("--seed", type=int, default=1000)
    args = ap.parse_args()

    clippers = [parse_clipper(c) for c in args.clippers.split(";")]
    images = load_images(args.images, args.n_images)
    codec = load_codec(args.model)
    modem = Modem()
    modes = [MODES[m] for m in args.modes]
    conds = [c.strip() for c in args.conditions.split(",") if c.strip()]
    snrs = [float(v) for v in args.snr_db.split(",")]
    print(
        f"{len(images)} images, SNR referenced to {SNR_REF_BW_HZ:.0f} Hz",
        file=sys.stderr,
    )

    latents = [codec.encode(img) for img in images]

    # (clipper, mode) -> waveforms; and the PAPR that sets its PEP credit
    waves, paprs = {}, {}
    for ci, (hd, sched) in enumerate(clippers):
        for spec in modes:
            w = modulate_all(modem, latents, spec, hd, sched)
            waves[ci, spec.name] = w
            paprs[ci, spec.name] = float(np.mean([papr_db(x) for x in w]))
            print(
                f"  clipper {ci} (hd={hd} k={','.join(f'{k:g}' for k in sched)}) "
                f"mode {spec.name}: PAPR {paprs[ci, spec.name]:.2f} dB",
                file=sys.stderr,
                flush=True,
            )

    rows = []
    for spec in modes:
        for cond in conds:
            fading = None if cond == "awgn" else cond
            for snr in snrs:
                out = []
                for ci, (hd, sched) in enumerate(clippers):
                    credit = paprs[0, spec.name] - paprs[ci, spec.name]
                    p, acq, _ = run_point(
                        codec,
                        modem,
                        waves[ci, spec.name],
                        images,
                        snr + credit,
                        fading,
                        args.seed,  # paired: same channel realizations
                    )
                    out.append((p, acq, credit))
                    print(
                        f"    ...{spec.name} {cond} {snr:g} clipper{ci}: "
                        f"psnr={p if p is None else round(p, 2)} acq={acq}",
                        file=sys.stderr,
                        flush=True,
                    )
                rows.append((spec.name, cond, snr, out))

    print(f"\nPEP-fair, mode(s) {args.modes}, {len(images)} images, paired seeds")
    for ci, (hd, sched) in enumerate(clippers):
        print(
            f"  clipper {ci}: headroom {hd} dB, k={','.join(f'{k:g}' for k in sched)}"
            + ("   [reference]" if ci == 0 else "")
        )
    head = f"\n{'mode':4} {'cond':5} {'nominal':>8}"
    head += "".join(f"{('c' + str(i)):>9}" for i in range(len(clippers)))
    head += "".join(f"{('d' + str(i)):>8}" for i in range(1, len(clippers)))
    print(head)
    deltas = {i: [] for i in range(1, len(clippers))}
    for mode, cond, snr, out in rows:
        line = f"{mode:4} {cond:5} {snr:8.0f}"
        for p, acq, _ in out:
            line += f"{'—' if p is None else f'{p:9.2f}'}"
        for i in range(1, len(clippers)):
            a, b = out[0][0], out[i][0]
            if a is None or b is None:
                line += f"{'—':>8}"
            else:
                deltas[i].append(b - a)
                line += f"{b - a:+8.2f}"
        print(line)
    for i in range(1, len(clippers)):
        if deltas[i]:
            d = np.array(deltas[i])
            print(
                f"\nclipper {i} mean delta {d.mean():+.3f} dB PSNR "
                f"(sd {d.std(ddof=1):.3f}, n={len(d)}), "
                f"worst {d.min():+.2f}, best {d.max():+.2f}"
            )


if __name__ == "__main__":
    main()

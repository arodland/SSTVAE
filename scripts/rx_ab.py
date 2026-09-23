#!/usr/bin/env python3
"""Paired-seed receiver A/B: effective latent SNR, acquisition and CFO
error over a fixed grid of channels, for whatever receiver is in the
tree. Run it before and after a receiver change and diff the CSVs.

    python scripts/rx_ab.py --out before.csv
    python scripts/rx_ab.py --out after.csv --compare before.csv

No codec: latents are random unit-RMS draws, and the score is
`latents * weights` against the truth (never bare latents -- see
CLAUDE.md on the 0.77 gain), as the SNR of the best linear fit, which is
what the decoder's input can be rescaled to. That makes this a
*receiver* comparison only. It says nothing about a clipper change, and
end-to-end PSNR (scripts/snr_sweep.py) is still the gate for anything
that ships.
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sstvae import hfchannel  # noqa: E402
from sstvae.config import (  # noqa: E402
    FRAME_SAMPLES, FS, HEADER_SAMPLES, LEADIN_SAMPLES, MODES, PREAMBLE_SAMPLES,
)
from sstvae.modem import Modem, SyncError  # noqa: E402

# name, snr_db, fading preset (or ("2path", delay_ms, spread_hz,
# late_gain_db)), cfo_hz, ppm
CONDITIONS = [
    ("awgn3", 3.0, None, 0.0, 0.0),
    ("mpg6", 6.0, "mpg", 0.0, 0.0),
    ("mpp8", 8.0, "mpp", 0.0, 0.0),
    ("mpp3", 3.0, "mpp", 0.0, 0.0),
    ("mpd8", 8.0, "mpd", 0.0, 0.0),
    ("mpd0", 0.0, "mpd", 0.0, 0.0),
    ("awgn0", 0.0, None, 0.0, 0.0),
    ("mps6", 6.0, "mps", 0.0, 0.0),
    ("mpp8_cfo", 8.0, "mpp", 37.0, 0.0),
    ("mpp8_ppm", 8.0, "mpp", 0.0, 80.0),
    ("mpp8_ppm20", 8.0, "mpp", 0.0, 20.0),
    ("late6db", 8.0, ("2path", 2.0, 1.0, 6.0), 0.0, 0.0),
    ("late6db_3ms", 8.0, ("2path", 3.0, 0.5, 6.0), 0.0, 0.0),
]


def two_path(x, delay_ms, spread_hz, late_gain_db, seed):
    """Watterson two-path with a stronger *late* path, which the stock
    presets (equal mean power) never produce."""
    rng = np.random.default_rng(seed)
    z = hfchannel._analytic(x)
    d = int(round(delay_ms * 1e-3 * FS))
    g1 = hfchannel._gaussian_taps(len(z), spread_hz, rng)
    g2 = hfchannel._gaussian_taps(len(z), spread_hz, rng) * 10 ** (late_gain_db / 20)
    z2 = np.concatenate([np.zeros(d, complex), z[: len(z) - d]])
    y = z * g1 + z2 * g2
    return np.real(y) / np.sqrt(np.mean(np.abs(y) ** 2) / np.mean(np.abs(z) ** 2))


def channel(x, cond, seed):
    _, snr, fade, cfo, ppm = cond
    if isinstance(fade, tuple):
        y = two_path(x, *fade[1:], seed=seed)
        return hfchannel.apply_channel(
            y, snr_db=snr, freq_offset_hz=cfo, ppm=ppm, seed=seed
        )
    return hfchannel.apply_channel(
        x, snr_db=snr, freq_offset_hz=cfo, ppm=ppm, fading_preset=fade, seed=seed
    )


def eff_snr_db(truth, got, mask):
    t, g = truth[mask], got[mask]
    rho2 = np.dot(t, g) ** 2 / (np.dot(t, t) * np.dot(g, g) + 1e-30)
    return 10 * np.log10(rho2 / max(1 - rho2, 1e-12))


def demod(modem, rx, mode, blind, ring=False):
    """(latents * weights, cfo) or None. Blind decodes from 20 frames in,
    preamble and header gone, or with `ring` the whole buffer, and
    returns mode C's container."""
    n = MODES[mode].n_latents
    try:
        if blind:
            r = modem.demodulate_blind(rx if ring else rx[BLIND_CUT:])
            if r.beacon is None:
                return None
        else:
            r = modem.demodulate(rx)
            if r.mode.name != mode:
                return None
    except SyncError:
        return None
    return (r.latents * r.weights)[:n], r.freq_offset


BLIND_CUT = LEADIN_SAMPLES + PREAMBLE_SAMPLES + HEADER_SAMPLES + 20 * FRAME_SAMPLES
RING_S, RING_START_S = 130.0, 40.0  # a full live ring, mostly not the transmission


def in_ring(tx):
    lead = np.zeros(int(RING_START_S * FS))
    return np.concatenate([lead, tx, np.zeros(max(int(RING_S * FS) - len(lead) - len(tx), 0))])


def run(mode, trials, conds, blind=False, ring=False):
    modem = Modem()
    spec = MODES[mode]
    rows = []
    for c in conds:
        for seed in range(trials):
            rng = np.random.default_rng(1000 + seed)
            lat = rng.normal(size=spec.n_latents)
            lat /= np.sqrt(np.mean(lat**2))
            tx = modem.modulate(lat, mode)
            if ring:
                tx = in_ring(tx)
            if seed == 0:
                mask = np.abs(demod(modem, tx, mode, blind, ring)[0]) > 0
            rx = channel(tx, c, seed)
            got = demod(modem, rx, mode, blind, ring)
            if got is not None:
                rows.append((c[0], seed, 1, eff_snr_db(lat, got[0], mask),
                             abs(got[1] - c[3])))
            else:
                rows.append((c[0], seed, 0, float("nan"), float("nan")))
            print(*rows[-1], flush=True, file=sys.stderr)
    return rows


def summarize(rows, base=None):
    by = {}
    for name, seed, ok, snr, cfo in rows:
        by.setdefault(name, {})[int(seed)] = (int(ok), float(snr), float(cfo))
    out = []
    for name, d in by.items():
        acq = np.mean([v[0] for v in d.values()])
        snrs = [v[1] for v in d.values() if v[0]]
        cfos = [v[2] for v in d.values() if v[0]]
        if not snrs:
            out.append(f"{name:12s} acq {acq:.2f}")
            continue
        line = (f"{name:12s} acq {acq:.2f}  eff {np.mean(snrs):6.2f} dB  "
                f"cfo p50 {np.median(cfos):.2f} max {np.max(cfos):.2f} Hz")
        if base and name in base:
            common = [s for s in d if d[s][0] and base[name].get(s, (0,))[0]]
            diff = [d[s][1] - base[name][s][1] for s in common]
            wins = sum(x > 0 for x in diff)
            b_acq = np.mean([v[0] for v in base[name].values()])
            dm = f"{np.mean(diff):+.2f}" if diff else "  n/a"
            line += f"  | d {dm} dB ({wins}/{len(diff)} up)  acq {b_acq:.2f}->{acq:.2f}"
        out.append(line)
    return "\n".join(out), by


def load(path):
    with open(path) as f:
        return summarize(list(csv.reader(f))[1:])[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--mode", default="A")
    ap.add_argument("--trials", type=int, default=16)
    ap.add_argument("--only", help="comma-separated condition names")
    ap.add_argument("--out", required=True)
    ap.add_argument("--compare")
    ap.add_argument("--blind", action="store_true",
                    help="score demodulate_blind on audio with the start cut off")
    ap.add_argument("--ring", action="store_true",
                    help="with --blind: the whole transmission inside a 130 s buffer")
    a = ap.parse_args()
    conds = [c for c in CONDITIONS if not a.only or c[0] in a.only.split(",")]
    rows = run(a.mode, a.trials, conds, a.blind, a.ring)
    with open(a.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cond", "seed", "ok", "eff_snr_db", "cfo_err_hz"])
        w.writerows(rows)
    print(summarize(rows, load(a.compare) if a.compare else None)[0])


if __name__ == "__main__":
    main()

"""Measure the spread-header prototype (docs/qrss/spread-header.md).

Two experiments, each through the real channel simulator and receiver:

header  FULL frames with the spread header (`full-spread`, which also
        keeps the header block, and/or `full-spreadonly`). Each pass is
        received once (round A, no round B), and its header is decoded
        three ways from the same pass: from the block's LLRs alone, from
        the spread copy's alone, and from both summed. `--fade S` takes
        the signal down 40 dB for the first S seconds after t0 (the
        preamble and header block lost). P(decode) per SNR.
data    SHORT frames, `short` against `short-spread` on the same seeds:
        the delivered per-latent SNR and mean weight W after round B
        (header known, so the spread phase has been removed), and after
        round A alone (header not yet known).

genie   No receiver: noiseless CE synthesis of a MEDIUM frame through the
        exact matched filter with the true carrier and timing, the
        header phase either removed (known) or not, for the scheme here
        ("top": header phase added, data keep beta) and the alternative
        ("split": data phase shrunk to make room), at --rho; per-latent
        SNR change at several noise levels, and the spectrum's widths.

Detection is skipped: each pass starts from a detection at the true
carrier with no timing, so gate V fits the timing from the known
symbols as `receive_pass` always does. That keeps a FULL pass to about
a minute and measures the header and the data rather than acquisition.

Usage:
    python scripts/qrss_spread_header.py header --snr -22 -24 -26 --seeds 8
    python scripts/qrss_spread_header.py header --fade 120 --snr -20 -22 --seeds 8
    python scripts/qrss_spread_header.py data --snr -12 -18 --seeds 6
    python scripts/qrss_spread_header.py genie --rho 0.054 0.1
Add --jobs N (default: all CPUs) and --out FILE.json to keep the rows.
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sstvae.qrss import channel as chm  # noqa: E402
from sstvae.qrss import demod, frame, frontend, header, morse  # noqa: E402
from sstvae.qrss import receiver as RX  # noqa: E402
from sstvae.qrss.channel import ChannelConfig  # noqa: E402
from sstvae.qrss.precoder import precode  # noqa: E402
from sstvae.qrss.types import Detection, FreqPath  # noqa: E402

Q = 1954752                                  # the tests' slot (2025-10-01T00:00Z)
CALL = "K1ABC"
HDR = header.HeaderFields(callsign=CALL, grid="FN42", picture_id=0x12345678, mode=0,
                          segment=0, codec_id=0xD1D8)
CARRIER = 1500.0


def latents(n, seed):
    a = np.random.default_rng(seed).standard_normal(n)
    return a / np.sqrt(np.mean(a ** 2))


def simulate(spec, snr, seed, fade_s=0.0):
    """(latents, Sim, FE capture) of one steady pass, faded before t0 + fade_s."""
    a = latents(spec.n_data, seed)
    sym = frame.assemble(spec, header.encode(HDR), precode(a, Q))
    kw = dict(carrier_hz=CARRIER, keying=morse.keying_units(CALL) if spec.n_win else None)
    noisy = chm.simulate(sym, spec, Q, ChannelConfig(snr_db=snr, seed=seed), **kw)
    fe = noisy.fe
    if fade_s > 0:
        clean = chm.simulate(sym, spec, Q, ChannelConfig(snr_db=None, seed=seed), **kw).fe
        t = (np.arange(len(fe)) - noisy.t0_index) / frontend.FE_FS
        fe = clean * np.where(t < fade_s, 0.01, 1.0) + (fe - clean)
    return a, noisy, fe.astype(np.complex64)


def receive(spec, sim, fe, round_b):
    prep = RX.prepare(frontend.Capture(fe, Q, sim.t0_index))
    path = FreqPath(t_s=np.array([-10.0, 0.0, 1800.0]), f_hz=np.full(3, CARRIER),
                    weight=np.ones(3))
    det = Detection(f_hz=CARRIER, path=path, timing=None, z_ref=np.nan, method="genie",
                    lead_in_s=0.0)
    return RX.receive_pass(prep, spec, det, round_b=round_b, return_tracks=True)


def snr_db(z, a):
    return float(10 * np.log10(np.mean(a ** 2) / np.mean((np.asarray(z, float) - a) ** 2)))


def w_db(w):
    return float(10 * np.log10(max(float(np.mean(np.asarray(w, float))), 1e-30)))


def header_trial(args):
    name, snr, seed, fade_s = args
    spec = frame.get(name)
    t = time.time()
    a, sim, fe = simulate(spec, snr, seed, fade_s)
    res = receive(spec, sim, fe, round_b=False)
    row = dict(exp="header", frame=name, snr=snr, seed=seed, fade_s=fade_s)
    if res is None:
        row.update(found=False, block=False, spread=False, both=False)
    else:
        p, trs = res
        _, _, llr, diag = demod.extract(None, trs[0], spec, Q)

        def ok(l):
            return l is not None and header.decode(np.asarray(l, float)) == HDR
        row.update(found=True, both=ok(llr), block=ok(diag["llr_block"]),
                   spread=ok(diag["llr_spread"]), z_ref=float(p.report.z_ref))
        # LLR quality against the true codeword: snr = mu^2/var of llr*sign
        # (= the coded bit's SNR for a consistent LLR), cons = var/(2 mu)
        # (1 for a correctly scaled one)
        c = 1.0 - 2.0 * header.encode(HDR)
        for k in ("llr_block", "llr_spread"):
            if diag[k] is not None:
                v = np.asarray(diag[k], float) * c
                mu, var = float(np.mean(v)), float(np.var(v))
                row[k[4:] + "_snr_db"] = float(10 * np.log10(max(mu * mu / var, 1e-30)))
                row[k[4:] + "_cons"] = var / (2 * mu) if mu > 0 else float("nan")
        # LLR quality against the true codeword: snr = mu^2/var of llr*sign
        # (= the coded bit's SNR for a consistent LLR), cons = var/(2 mu)
        # (1 for a correctly scaled one)
        c = 1.0 - 2.0 * header.encode(HDR)
        for k in ("llr_block", "llr_spread"):
            if diag[k] is not None:
                v = np.asarray(diag[k], float) * c
                mu, var = float(np.mean(v)), float(np.var(v))
                row[k[4:] + "_snr_db"] = float(10 * np.log10(max(mu * mu / var, 1e-30)))
                row[k[4:] + "_cons"] = var / (2 * mu) if mu > 0 else float("nan")
    row["secs"] = round(time.time() - t, 1)
    return row


def data_trial(args):
    snr, seed = args
    out = []
    for name in ("short", "short-spread"):
        spec = frame.get(name)
        a, sim, fe = simulate(spec, snr, seed)
        for rb in (False, True):
            res = receive(spec, sim, fe, round_b=rb)
            row = dict(exp="data", frame=name, snr=snr, seed=seed,
                       round="B" if rb else "A")
            if res is None:
                row.update(found=False)
            else:
                p, trs = res
                row.update(found=True, header=p.header == HDR, rounds=len(trs),
                           lat_db=snr_db(p.z, a), w_db=w_db(p.w))
            out.append(row)
    return out


def genie(rhos):
    from scipy.signal import welch

    from sstvae.qrss import ce
    spec = frame.MEDIUM
    lay = frame.layout(spec)
    fs = 250
    n0, n = ce._loopback_span(spec, fs)
    rng = np.random.default_rng(2)
    data = np.asarray(lay.data)
    pos = np.asarray(lay.pos)[data]
    known = np.zeros(spec.n_sym)
    known[np.asarray(lay.known_idx)] = lay.known_val
    w = (rng.standard_normal(n) + 1j * rng.standard_normal(n)) / np.sqrt(2)
    nv = np.var(np.imag(ce.matched_filter(w, fs, n0, pos)))  # Im(MF) noise per unit sample noise

    def stats(rho, mode, reps=3):
        acc = np.zeros(4)
        for r in range(reps):                           # the same draws for every scheme
            dr = np.random.default_rng(100 + r)
            a = dr.standard_normal(len(data))
            b = dr.choice([-1.0, 1.0], len(data))
            xd, xh = ((np.sqrt(1 - rho) * a, np.sqrt(rho) * b) if mode == "split"
                      else (a, np.sqrt(rho) * b))
            sd = known.copy()
            sd[data] = xd
            sh = np.zeros(spec.n_sym)
            sh[data] = xh
            phid = ce.phase_grid(fs, n0, n, sd, spec)
            phih = ce.phase_grid(fs, n0, n, sh, spec)
            s = np.exp(1j * (phid + phih))
            row = []
            for zz in (s * np.exp(-1j * phih), s):      # header known, unknown
                car = np.mean(zz)
                y = np.imag(ce.matched_filter(zz, fs, n0, pos) * np.conj(car) / abs(car))
                g = np.dot(y, a) / np.dot(a, a)
                row += [g, np.mean((y - g * a) ** 2)]
            acc += np.array(row)
        return acc / reps

    def snr(g, d, s2):
        return 10 * np.log10(g * g / (d + s2 * nv))
    sig = [0.0, 1.0, 10.0, 100.0]
    base = stats(0.0, "split")
    print("per-latent SNR, plain frame:",
          "  ".join(f"{snr(base[0], base[1], s):+.2f}" for s in sig), "dB")
    for mode in ("top", "split"):
        for rho in rhos:
            st = stats(rho, mode)
            k = "  ".join(f"{snr(st[0], st[1], s) - snr(base[0], base[1], s):+.3f}" for s in sig)
            u = "  ".join(f"{snr(st[2], st[3], s) - snr(base[0], base[1], s):+.3f}" for s in sig)
            print(f"{mode:5s} rho {rho:.3f}  known: {k}  | unknown: {u}")
    fs2 = 1000
    n0, n = ce._loopback_span(spec, fs2)
    for rho in [0.0] + list(rhos):
        sym = known.copy()
        sym[data] = np.random.default_rng(7).standard_normal(len(data)) * np.sqrt(1 + rho)
        z = np.exp(1j * ce.phase_grid(fs2, n0, n, sym, spec))
        f, P = welch(z, fs2, nperseg=1 << 14, return_onesided=False)
        o = np.argsort(f)
        f, P = f[o], P[o]
        c = np.cumsum(P) / P.sum()

        def width(q):
            return f[np.searchsorted(c, 1 - (1 - q) / 2)] - f[np.searchsorted(c, (1 - q) / 2)]

        def dens(hz):
            sel = (np.abs(f) > hz - 1) & (np.abs(f) < hz + 1)
            return 10 * np.log10(np.mean(P[sel]) / (P.sum() * (f[1] - f[0])))
        print(f"top rho {rho:.3f}: rms phase {0.8 * np.sqrt(1 + rho):.3f} rad, 99% "
              f"{width(0.99):.1f} Hz, 99.9% {width(0.999):.1f} Hz, density (dB/Hz of total) "
              f"at 30/50/80 Hz {dens(30):.1f}/{dens(50):.1f}/{dens(80):.1f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("exp", choices=["header", "data", "genie"])
    ap.add_argument("--snr", type=float, nargs="+")
    ap.add_argument("--rho", type=float, nargs="+", default=[0.054, 0.10])
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--seed0", type=int, default=1)
    ap.add_argument("--frames", nargs="+", default=["full-spread"],
                    help="header experiment: frames to run (default full-spread)")
    ap.add_argument("--fade", type=float, default=0.0)
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.exp == "genie":
        genie(args.rho)
        return
    if not args.snr:
        ap.error("--snr is required")
    seeds = range(args.seed0, args.seed0 + args.seeds)
    if args.exp == "header":
        work = [(f, s, k, args.fade) for f in args.frames for s in args.snr for k in seeds]
        fn = header_trial
    else:
        work = [(s, k) for s in args.snr for k in seeds]
        fn = data_trial
    rows = []
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        for r in ex.map(fn, work):
            rs = r if isinstance(r, list) else [r]
            for x in rs:
                print(json.dumps(x), flush=True)
            rows.extend(rs)
    print()
    if args.exp == "header":
        for f in args.frames:
            for s in args.snr:
                sel = [r for r in rows if r["frame"] == f and r["snr"] == s]
                n = len(sel)
                frac = {k: sum(r[k] for r in sel) / n for k in ("found", "block", "spread", "both")}
                print(f"{f:16s} {s:+6.1f} dB fade {args.fade:5.0f}s  n={n:2d}  " +
                      "  ".join(f"{k} {v:.2f}" for k, v in frac.items()))
    else:
        for s in args.snr:
            for rnd in ("A", "B"):
                d_lat, d_w = [], []
                for k in seeds:
                    base = [r for r in rows if r["snr"] == s and r["seed"] == k
                            and r["round"] == rnd and r["frame"] == "short"]
                    spr = [r for r in rows if r["snr"] == s and r["seed"] == k
                           and r["round"] == rnd and r["frame"] == "short-spread"]
                    if base and spr and base[0]["found"] and spr[0]["found"]:
                        d_lat.append(spr[0]["lat_db"] - base[0]["lat_db"])
                        d_w.append(spr[0]["w_db"] - base[0]["w_db"])
                if d_lat:
                    print(f"{s:+6.1f} dB round {rnd}: spread - plain  latent SNR "
                          f"{np.mean(d_lat):+.3f} +- {np.std(d_lat) / np.sqrt(len(d_lat)):.3f} dB, "
                          f"W {np.mean(d_w):+.3f} +- {np.std(d_w) / np.sqrt(len(d_w)):.3f} dB "
                          f"(n={len(d_lat)})")
    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()

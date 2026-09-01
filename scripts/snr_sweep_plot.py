#!/usr/bin/env python3
"""Plot SNR vs PSNR from scripts/snr_sweep.py's --csv output.

    python scripts/snr_sweep_plot.py sweep.csv --out sweep.png
    python scripts/snr_sweep_plot.py sweep.csv --combined --model tilt \
        --conditions awgn,mpp --out tilt.png

Separate from the sweep because the sweep costs hours of NumPy
demodulation and the plot costs a second: re-rendering should never mean
re-measuring.

Two layouts. The default is one panel per (condition, mode) with every
model overlaid -- for comparing checkpoints. `--combined` puts every
trace on one pair of axes instead, colour by mode and line style by
condition, which is the shape you want for a single checkpoint's
characteristic curves.

Points below `--min-acq` are dropped. Near threshold the mean PSNR is
taken over whichever images happened to get through, so it stops being a
fair sample of the same population well before it stops being a number,
and a curve that trails off into those points reads as quality falling
gently when it is really the sample changing underneath. The grid
layout still carries the acquisition rate on the right-hand axis, so
what was dropped is visible rather than merely absent.
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

MODES = ["A", "B", "C"]


def plot_combined(series, conds, modes, models, args):
    """Every trace on one axes: colour = mode, line style = condition.

    Deliberately drops the acquisition axis that the grid layout carries
    -- one dotted line per trace is a dozen lines nobody can read. With
    the sub-`--min-acq` points filtered out there is nothing left for it
    to qualify.
    """
    styles = [("-", "o"), ("--", "s"), ("-.", "^"), (":", "v")]
    colors = {m: c for m, c in zip(modes,
              plt.rcParams["axes.prop_cycle"].by_key()["color"])}
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for ci, cond in enumerate(conds):
        ls, mk = styles[ci % len(styles)]
        for mode in modes:
            for model in models:
                pts = sorted(series.get((cond, mode, model), []))
                if not pts:
                    continue
                x = [p[0] for p in pts]
                y = [p[1] for p in pts]
                c = colors[mode]
                lbl = f"{cond} — mode {mode}"
                if len(models) > 1:
                    lbl += f" ({model})"
                ax.plot(x, y, ls, color=c, marker=mk, ms=4.5, lw=1.5,
                        label=lbl, zorder=3)
    if args.min_psnr is not None:
        ax.set_ylim(bottom=args.min_psnr)
    ax.grid(alpha=0.3, zorder=0)
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("PSNR (dB)")
    title = "PSNR vs channel SNR"
    if args.model:
        title += f" — {args.model}"
    ax.set_title(title + f"   (points with <{args.min_acq:.0%} acquisition dropped)",
                 fontsize=10)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", type=Path)
    ap.add_argument("--out", type=Path, default=Path("sweep.png"))
    ap.add_argument("--min-psnr", type=float, default=None,
                    help="clip the y axis here (the sweep runs past the point "
                         "where a 'picture' is noise; default shows everything)")
    ap.add_argument("--no-acq", action="store_true", help="drop the acq-rate axis")
    ap.add_argument("--min-acq", type=float, default=0.5,
                    help="drop points where fewer than this fraction of images "
                         "acquired (default 0.5, the sweep's own stopping rule)")
    ap.add_argument("--model", default=None,
                    help="plot only this model (default: all)")
    ap.add_argument("--conditions", default=None,
                    help="comma-separated subset, in this order")
    ap.add_argument("--combined", action="store_true",
                    help="one pair of axes: colour by mode, line style by "
                         "condition. Needs a single --model to stay readable")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.csv)))
    rows = [r for r in rows
            if r["psnr_db"] != ""
            and int(r["acquired"]) / int(r["n_images"]) >= args.min_acq]
    if not rows:
        raise SystemExit(f"--min-acq {args.min_acq} left no points")
    if args.model:
        rows = [r for r in rows if r["model"] == args.model]
        if not rows:
            raise SystemExit(f"no rows for model {args.model!r}")
    # (condition, mode, model) -> [(snr, psnr, acq_frac)]
    series = defaultdict(list)
    for r in rows:
        if r["psnr_db"] == "":
            continue
        series[(r["channel"], r["mode"], r["model"])].append(
            (float(r["snr_db"]), float(r["psnr_db"]),
             int(r["acquired"]) / int(r["n_images"]))
        )
    conds = list(dict.fromkeys(r["channel"] for r in rows))
    if args.conditions:
        want = [c.strip() for c in args.conditions.split(",") if c.strip()]
        missing = [c for c in want if c not in conds]
        if missing:
            raise SystemExit(f"no rows for condition(s) {missing}; have {conds}")
        conds = want
    models = list(dict.fromkeys(r["model"] for r in rows))
    modes = [m for m in MODES if any(r["mode"] == m for r in rows)]

    if args.combined:
        plot_combined(series, conds, modes, models, args)
        return

    fig, axes = plt.subplots(len(conds), len(modes), figsize=(4.2 * len(modes),
                                                              3.0 * len(conds)),
                             sharex=True, squeeze=False)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for ri, cond in enumerate(conds):
        for ci, mode in enumerate(modes):
            ax = axes[ri][ci]
            axr = None
            for mi, model in enumerate(models):
                pts = sorted(series.get((cond, mode, model), []))
                if not pts:
                    continue
                x = [p[0] for p in pts]
                y = [p[1] for p in pts]
                acq = [p[2] for p in pts]
                c = colors[mi % len(colors)]
                ax.plot(x, y, "-o", ms=4, color=c, label=model, zorder=3)
                if not args.no_acq:
                    axr = axr or ax.twinx()
                    axr.plot(x, [a * 100 for a in acq], ":", color=c, lw=1,
                             alpha=0.55, zorder=2)
            if axr is not None:
                axr.set_ylim(0, 105)
                axr.axhline(50, color="grey", lw=0.6, ls="--", alpha=0.5)
                axr.set_ylabel("acq %", fontsize=7, color="grey")
                axr.tick_params(labelsize=6, colors="grey")
            if args.min_psnr is not None:
                ax.set_ylim(bottom=args.min_psnr)
            ax.grid(alpha=0.3, zorder=0)
            ax.set_title(f"{cond} — mode {mode}", fontsize=9)
            if ci == 0:
                ax.set_ylabel("PSNR (dB)", fontsize=8)
            if ri == len(conds) - 1:
                ax.set_xlabel("SNR (dB)", fontsize=8)
            ax.tick_params(labelsize=7)
            if ri == 0 and ci == 0:
                ax.legend(fontsize=7, loc="lower right")
    fig.suptitle(f"PSNR vs channel SNR (dotted = acq rate; "
                 f"points under {args.min_acq:.0%} acquisition dropped)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

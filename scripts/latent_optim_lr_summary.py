#!/usr/bin/env python3
"""Summarize `latent_optim_lr_sweep.py` output.

Each cell carries a per-step gain trace and full evaluations at the
anchor steps, so the stopping point is chosen here rather than in the
sweep. Prints, over the corpus:

- the mean gain curve by profile at whatever steps `--steps` names;
- the anchor table, with the clean-decode gain beside the channel one
  as the divergence guard the sweep describes;
- a trace-vs-anchor check, since the trace is a 4-draw estimate and the
  anchors are 16-draw -- if they disagree by more than the noise, the
  curve is the thing to distrust. It doubles as an alignment check: the
  raw file indexes *iterates* and everything here counts **updates**,
  off by one, and a shift of one step would show up as a large bias
  here rather than as plausible-looking numbers;
- per-image spread at `--at`, because a 10-picture mean can hide a
  profile that helps eight and hurts two.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def replay_stop(trace_gain, patience=10, min_rel_gain=2e-3):
    """Where `optimize`'s plateau rule would have stopped this run.

    A transcription of the loop in `sstvae/latent_optim.py`, including
    the ordering that is easy to get wrong: `best_step` is tested
    against the *previous* `best_mse`, before it is updated, and the
    break is checked after both. The trace is the MSE sequence up to a
    constant factor and the rule only compares MSEs to each other, so
    replaying from gains loses nothing -- set mse[0] = 1.

    Returns (updates_at_stop, gain_of_best_iterate, fired), where
    `fired` is False if the trace ran out before the rule triggered --
    in which case the run was still improving and the number is a lower
    bound, not a plateau.
    """
    mse = [10.0 ** (-g / 10.0) for g in trace_gain]
    best_mse, best_step, best_i = float("inf"), 0, 0
    for step, m in enumerate(mse, start=1):
        if m < best_mse * (1 - min_rel_gain):
            best_step = step
        if m < best_mse:
            best_mse, best_i = m, step - 1
        if step - best_step >= patience:
            return step - 1, trace_gain[best_i], True
    return len(mse) - 1, trace_gain[best_i], False


def load(shards):
    rows = [json.loads(line)
            for p in shards
            for line in Path(p).read_text().splitlines() if line.strip()]
    return [r for r in rows if r["kind"] == "cell"]


# The sweep records one entry per *iterate*, and `optimize` evaluates
# the objective at the top of the loop -- before that iteration's
# update. So iterate 1 is the encoder's latents with nothing applied,
# and iterate k holds k-1 updates. Everything below counts **updates**,
# which is what "5 steps of optimization" means to a reader: index the
# trace directly (trace[0] = 0 updates) and shift the anchor keys.
def mean_at(cells, updates):
    """Mean trace gain after `updates` updates, over cells that got there."""
    v = [c["trace_gain"][updates] for c in cells
         if len(c["trace_gain"]) > updates]
    return float(np.mean(v)) if v else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("shards", nargs="+")
    ap.add_argument("--steps", type=int, nargs="+",
                    default=[2, 3, 5, 8, 10, 15, 20, 30, 50])
    ap.add_argument("--at", type=int, default=5,
                    help="step for the per-image breakdown")
    ap.add_argument("--patience", type=int, default=10,
                    help="plateau patience to replay (shipped default 10)")
    ap.add_argument("--min-rel-gain", type=float, default=2e-3,
                    help="plateau min_rel_gain to replay (shipped 2e-3)")
    args = ap.parse_args()

    cells = load(args.shards)
    if not cells:
        raise SystemExit("no cells in input")
    profiles = sorted({c["profile"] for c in cells})
    modes = sorted({c["mode"] for c in cells})
    images = sorted({c["image"] for c in cells})
    print(f"{len(cells)} cells: {len(images)} images x {len(modes)} modes "
          f"x {len(profiles)} profiles")
    if len(cells) != len(images) * len(modes) * len(profiles):
        print("  (incomplete -- means are over whatever finished)")

    by = defaultdict(list)
    for c in cells:
        by[c["profile"]].append(c)

    print(f"\n{'=' * 78}\nmean gain (dB) by step -- corpus x modes, "
          f"per-step trace\n{'=' * 78}")
    print(f"{'profile':<13}" + "".join(f"{s:>7}" for s in args.steps))
    for p in profiles:
        print(f"{p:<13}"
              + "".join(f"{mean_at(by[p], s):>7.2f}" for s in args.steps))

    print(f"\n{'=' * 78}\nby mode, at selected steps\n{'=' * 78}")
    show = [s for s in (5, 20, 50) if s in args.steps] or args.steps[-1:]
    print(f"{'profile':<13}"
          + "  ".join(f"{'s=' + str(s):>{5 * len(modes) + 2}}" for s in show))
    print(f"{'':<13}" + "  ".join(
        "".join(f"{m:>5}" for m in modes) + "  " for _ in show))
    for p in profiles:
        cols = []
        for s in show:
            cols.append("".join(
                f"{mean_at([c for c in by[p] if c['mode'] == m], s):>5.2f}"
                for m in modes))
        print(f"{p:<13}" + "    ".join(cols))

    # {updates: raw key}, the raw key being the iterate index on disk.
    anchors = {int(k) - 1: k for c in cells for k in c["anchors"]}
    anchors = dict(sorted(anchors.items()))
    print(f"\n{'=' * 78}\nanchors: 16-draw held-out gain / clean gain (dB)"
          f"\n{'=' * 78}")
    print(f"{'profile':<13}" + "".join(f"{'@' + str(u):>15}" for u in anchors)
          + f"{'worst':>9}{'n<0':>5}")
    for p in profiles:
        cols, worst, neg = [], float("inf"), 0
        for u, key in anchors.items():
            ch = [c["anchors"][key]["gain_ch"] for c in by[p]
                  if key in c["anchors"]]
            cl = [c["anchors"][key]["gain_clean"] for c in by[p]
                  if key in c["anchors"]]
            cols.append(f"{np.mean(ch):>+8.2f}/{np.mean(cl):>+6.2f}"
                        if ch else f"{'-':>15}")
            if u > 0 and ch:
                worst = min(worst, min(ch))
                neg += sum(1 for x in ch if x < 0)
        print(f"{p:<13}" + "".join(cols) + f"{worst:>+9.2f}{neg:>5}")

    # The trace is cheap because its draws are independent of the
    # iterate; that is an argument, not a measurement, so check it.
    diffs = [c["trace_gain"][u] - c["anchors"][key]["gain_ch"]
             for c in cells for u, key in anchors.items()
             if key in c["anchors"] and len(c["trace_gain"]) > u]
    print(f"\ntrace - anchor over {len(diffs)} paired points: "
          f"mean {np.mean(diffs):+.3f} dB, sd {np.std(diffs):.3f}, "
          f"max |{max(abs(d) for d in diffs):.3f}|")

    # Where the shipped plateau rule would have fired, replayed from the
    # traces. Verified against the real loop on 4 cells / 2 rates / 4
    # patience values: identical stop step and gain to 1e-9.
    print(f"\n{'=' * 78}\nplateau rule replayed (patience={args.patience}, "
          f"min_rel_gain={args.min_rel_gain:g})\n{'=' * 78}")
    print(f"{'profile':<13}{'stop@':>8}{'sd':>7}{'range':>12}"
          f"{'gain':>8}{'of final':>10}{'never fired':>13}")
    for p in profiles:
        stops, gains, fracs, unfired = [], [], [], 0
        for c in by[p]:
            u, g, fired = replay_stop(c["trace_gain"], args.patience,
                                      args.min_rel_gain)
            final = max(c["trace_gain"])
            stops.append(u)
            gains.append(g)
            fracs.append(g / final if final > 0 else float("nan"))
            unfired += not fired
        print(f"{p:<13}{np.mean(stops):>8.1f}{np.std(stops):>7.1f}"
              f"{str(min(stops)) + '-' + str(max(stops)):>12}"
              f"{np.mean(gains):>+8.2f}{100 * np.mean(fracs):>9.0f}%"
              f"{unfired:>13}")

    print(f"\n{'=' * 78}\nper-image gain at step {args.at}, over modes"
          f"\n{'=' * 78}")
    order = sorted(profiles, key=lambda p: -mean_at(by[p], args.at))
    print(f"{'image':<18}" + "".join(f"{p[-9:]:>11}" for p in order))
    for img in images:
        print(f"{img:<18}" + "".join(
            f"{mean_at([c for c in by[p] if c['image'] == img], args.at):>11.2f}"
            for p in order))


if __name__ == "__main__":
    main()

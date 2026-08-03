#!/usr/bin/env python3
"""Learning-rate profiles for transmit-time latent optimization.

`sstvae/latent_optim.py` shipped `LEARNING_RATE = 0.02` until
2026-08-02, and that value was never swept -- it was picked in the
prototype and survived. This is the harness that replaced it with 0.05;
see `docs/latent-optimization.md` for the result. Keep it working: the
rate is a property of the encoder's amortization gap, so it has to be
re-measured on a new checkpoint rather than inherited.

**What is measured, and what it is not.** The score here is the
*objective*: image-domain PSNR of the fp32 decoder's output against the
target, through the same additive latent-domain channel the optimizer
optimizes. `docs/latent-optimization.md` records that this overstates
recovered on-air PSNR by roughly 3x, and that a design decision was once
made on it and had to be overturned by the end-to-end round trip. So
this is a *ranking* tool for narrowing candidates cheaply, and whatever
wins here has to be confirmed with `latent_optim_roundtrip.py` before it
becomes a default. Two things guard the obvious trap:

- the evaluation noise draws are **held out** -- a different seed and
  more of them than the four the optimizer averages per step, so a
  profile cannot win by fitting the draws it was shown; and
- a **clean** (noiseless) decode is scored alongside, because the known
  failure of this objective is a latent set that wins under its assumed
  noise and diverges from the picture. A profile whose channel score
  rises while its clean score falls is doing that.

**Every profile is a function of the step number alone.** Nothing here
may know its horizon, and that is a constraint on the candidates rather
than a convenience: the shipping optimizer stops on a plateau or a time
budget, so it does not know how many steps it will get either. A
cosine-annealed-to-50 would score at 5 steps as a policy that could
never have been run, because the real thing cannot be told in advance
that it has 5 steps. That rules out anneals and admits warmups.

**One run per cell, scored at every step.** Each cell runs once to
`--horizon` and records the objective at each step, so any stopping
point can be read off afterwards rather than chosen in advance -- which
matters because how many steps a real station gets is a judgment call
about hardware we do not have, not a number this machine can measure.

The per-step trace is free *and* honest, which is worth stating because
it looks too cheap: `optimize` computes the objective at step N from
noise drawn **after** z_N was determined, so the value it reports is
already an out-of-sample estimate at that iterate. It is a 4-draw
estimate and therefore noisy, which the corpus average absorbs; the
thing it cannot show is the clean decode, so full 16-draw + clean
evaluations still run at `ANCHOR_STEPS`.

    python scripts/latent_optim_lr_sweep.py data/optim_corpus/*.png \\
        --out /tmp/lrsweep.jsonl
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sstvae import checkpoint, latent_optim  # noqa: E402
from sstvae.codec import load_codec, pad_to_full  # noqa: E402
from sstvae.config import CHANNELS_PER_GROUP, MODES  # noqa: E402
from sstvae.images import fit_image, image_to_array  # noqa: E402
from sstvae.latents import flat_to_latents  # noqa: E402

# Held-out evaluation. A different seed from the optimizer's (which is
# 0) and four times as many draws, so the number is about the profile
# rather than about which noise it happened to see.
EVAL_SEED = 20260802
EVAL_DRAWS = 16

# Iterates that get the full held-out + clean evaluation. The trace
# covers every one; these are where it is anchored to a low-noise
# number and where the clean-decode divergence guard is read.
#
# **These are iterate indices, not update counts, and the two differ by
# one.** `optimize` evaluates the objective at the top of its loop,
# before that iteration's update, so iterate 1 is the encoder's latents
# untouched and iterate k carries k-1 updates. That is why iterate 1
# scores exactly +0.00 against the baseline on every profile -- it *is*
# the baseline, which makes it a free check that the two agree.
# `latent_optim_lr_summary.py` converts to updates and is the only
# place that should; anything reading these files raw must subtract one.
ANCHOR_STEPS = (1, 5, 10, 20, 35, 50)


def _const(v: float):
    return lambda step: v


def _halflife(peak: float, half: float, floor: float = 0.002):
    """Exponential decay, `half` steps per halving."""
    return lambda step: max(floor, peak * 0.5 ** ((step - 1) / half))


def _inv(peak: float, scale: float, floor: float = 0.002):
    """1/t decay, down to peak/2 after `scale` steps."""
    return lambda step: max(floor, peak / (1 + (step - 1) / scale))


def _warm(peak: float, warm: int):
    """Linear warmup over `warm` steps, then constant.

    Round 1 said the penalty for a high rate is a first-step overshoot
    rather than anything about the rate itself -- 12 cells at 5 steps
    returned exactly the encoder's latents back, every one of them from
    a profile starting at 0.10 or above. Warming up is horizon-free, so
    unlike an anneal it is a policy the shipping optimizer could
    actually run without knowing how many steps it will get.
    """
    return lambda step: peak * min(1.0, step / warm)


# Constants bracket the shipping 0.02 by half a decade each way. The
# decays all start high and fall, which is the shape worth testing when
# one of the two horizons is 5 steps: a rate tuned to reach a good
# minimum in 50 has barely moved in 5. Their time constants are in
# absolute steps, not fractions of a horizon, per the rule above.
PROFILES = {
    "const-0.01": _const(0.01),
    "const-0.02": _const(0.02),      # ships today
    "const-0.05": _const(0.05),
    "const-0.10": _const(0.10),
    "inv-0.10-t8": _inv(0.10, 8.0),
    "exp-0.10-h15": _halflife(0.10, 15.0),
    "exp-0.20-h10": _halflife(0.20, 10.0),

    # Round 2. The 50-step curve was still monotone at 0.10 with no
    # turnover, so 0.15 and 0.20 look for the edge rather than assume
    # 0.10 is it; 0.07 fills the gap between the two horizons' winners;
    # the warmups test whether 0.10's 5-step penalty is only its first
    # step. Decays are not represented -- round 1 had every one of them
    # losing to the constant it starts from, at both horizons.
    "const-0.07": _const(0.07),
    "const-0.15": _const(0.15),
    "const-0.20": _const(0.20),
    "warm2-0.10": _warm(0.10, 2),
    "warm3-0.15": _warm(0.15, 3),
}

# The candidates actually swept. The decays above are kept for the
# record but dropped from the default set: round 1 had every one of
# them losing to the constant it starts from, at both 5 and 50 steps.
CANDIDATES = ["const-0.01", "const-0.02", "const-0.05", "const-0.07",
              "const-0.10", "const-0.15", "const-0.20",
              "warm2-0.10", "warm3-0.15"]


def psnr_from_mse(mse: float) -> float:
    return -10.0 * math.log10(max(mse, 1e-12))


class Evaluator:
    """Scores a latent vector on held-out channel draws and clean.

    Uses the gradient graph's own MSE output rather than a second
    decoder session: it is the same fp32 decoder the optimizer sees, so
    a difference between two profiles cannot be an artifact of scoring
    them through a different graph than they were optimized through.
    """

    def __init__(self, model: str | None, image: np.ndarray):
        path = checkpoint.resolve_onnx(checkpoint.GRAD_PART, model)
        self.sess = latent_optim._session(path)
        self.in_names = [i.name for i in self.sess.get_inputs()]
        self.out_names = [o.name for o in self.sess.get_outputs()]
        self.target = np.ascontiguousarray(image, dtype=np.float32)[None]
        self.rng = np.random.default_rng(EVAL_SEED)
        self._noise = None

    def _draws(self, shape) -> np.ndarray:
        # Frozen, and shared by every profile and both horizons: the
        # comparison is between latents, never between noise draws.
        if self._noise is None:
            self._noise = self.rng.standard_normal(
                (EVAL_DRAWS, *shape[1:])).astype(np.float32)
        return self._noise

    def _mse(self, z: np.ndarray, weights: np.ndarray) -> float:
        _, _, mse = self.sess.run(
            self.out_names, dict(zip(self.in_names, (z, weights, self.target))))
        return float(mse)

    def score_z(self, z: np.ndarray, weights: np.ndarray) -> dict:
        """Score an iterate as the optimizer holds it: projected, shaped."""
        sigma = float(10.0 ** (-latent_optim.OBJECTIVE_SNR_DB / 20.0))
        noise = self._draws(z.shape)
        vals = [self._mse(z + noise[i] * sigma, weights)
                for i in range(EVAL_DRAWS)]
        return {"psnr_ch": psnr_from_mse(float(np.mean(vals))),
                "psnr_clean": psnr_from_mse(self._mse(z, weights))}

    def score(self, flat: np.ndarray, mode: str) -> dict:
        """Score a flat latent vector, projecting it as `optimize` would."""
        spec = MODES[mode]
        active = spec.groups * CHANNELS_PER_GROUP
        z = flat_to_latents(
            pad_to_full(np.asarray(flat[: spec.n_latents],
                                   dtype=np.float32))[None]).astype(np.float32)
        weights = np.zeros_like(z)
        weights[:, :active] = 1.0
        return self.score_z(latent_optim._project(z * weights, active), weights)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("images", nargs="+")
    ap.add_argument("--model", default=None)
    ap.add_argument("--modes", nargs="+", default=["A", "B", "C"])
    ap.add_argument("--horizon", type=int, default=50)
    ap.add_argument("--anchors", type=int, nargs="+", default=None,
                    metavar="ITERATE",
                    help="iterate indices for the full 16-draw + clean "
                         "evaluation (see ANCHOR_STEPS: these are "
                         "iterates, one more than the update count)")
    ap.add_argument("--profiles", nargs="+", default=CANDIDATES)
    ap.add_argument("--out", required=True, help="JSONL, one row per cell")
    args = ap.parse_args()
    anchor_steps = tuple(args.anchors) if args.anchors else ANCHOR_STEPS

    codec = load_codec(args.model)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("a", buffering=1) as fh:
        for path in args.images:
            name = Path(path).stem
            image = image_to_array(fit_image(Image.open(path)))
            enc = codec.encode(image)
            ev = Evaluator(args.model, image)

            for mode in args.modes:
                base = ev.score(enc, mode)
                fh.write(json.dumps(
                    {"kind": "base", "image": name, "mode": mode, **base}) + "\n")
                print(f"{name} {mode} baseline  ch {base['psnr_ch']:.2f}  "
                      f"clean {base['psnr_clean']:.2f}", flush=True)

                for prof in args.profiles:
                    t0 = time.perf_counter()
                    trace, anchors = [], {}

                    def on_iterate(step, z, weights):
                        if step in anchor_steps:
                            anchors[step] = ev.score_z(z, weights)

                    # +1: the objective reported at a step belongs to
                    # the latents that went *into* it, so N updates need
                    # N+1 evaluated iterates for the last one to count.
                    # Early stopping is off; the whole trace is kept and
                    # the stopping point is chosen at analysis time.
                    latent_optim.optimize(
                        enc, image, mode, model=args.model,
                        max_steps=args.horizon + 1, patience=10**9,
                        time_budget_s=1e9, lr=PROFILES[prof],
                        progress=lambda s, mse, el: trace.append(mse),
                        on_iterate=on_iterate)

                    # Paired against step 1 -- the same projected
                    # encoder latents every profile starts from -- so
                    # the trace is a gain curve rather than an MSE one.
                    row = {
                        "kind": "cell", "image": name, "mode": mode,
                        "profile": prof, "horizon": args.horizon,
                        "trace_gain": [10.0 * math.log10(max(trace[0], 1e-12)
                                                         / max(m, 1e-12))
                                       for m in trace],
                        "anchors": {
                            str(s): {
                                "gain_ch": a["psnr_ch"] - base["psnr_ch"],
                                "gain_clean": (a["psnr_clean"]
                                               - base["psnr_clean"]),
                            } for s, a in sorted(anchors.items())},
                        "seconds": time.perf_counter() - t0,
                    }
                    fh.write(json.dumps(row) + "\n")
                    at = row["anchors"]
                    print(f"  {name} {mode} {prof:<13} "
                          + "  ".join(
                              f"@{s}:{at[str(s)]['gain_ch']:+.2f}"
                              for s in anchor_steps if str(s) in at)
                          + f"  {row['seconds']:.0f}s", flush=True)


if __name__ == "__main__":
    main()

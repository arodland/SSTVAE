"""Transmit-time per-image latent optimization.

The encoder is amortized: one forward pass, trained to minimize loss
averaged over a dataset. For the particular picture in front of it there
is generally a better input to the same frozen decoder, and a
transmission lasts 32-95 s against the encoder's 31 ms -- so the sender
can afford to go looking. Measured worth 1.4-1.8 dB of *recovered*
picture across both test images, all three modes and every channel model
tried; see `docs/latent-optimization.md` for the measurements and for
the two findings that shaped this code.

**Sender-side only.** What comes out is an ordinary unit-RMS latent
vector: same count, same on-air contract, same modem. Every existing
receiver decodes it with no version negotiation, which is what makes
this cheap to ship and is not to be given up lightly.

**Where the gain comes from** (measured 2026-08-04, 5 corpus images,
decomposed clean-vs-channel and by latent group). It is an
*amortization* gain and not a noise-interaction one: the optimized
latents decode **1.8-5.3 dB better on a clean channel**, and the
received gain is almost deterministic in that (r = +0.988) times a
retention factor of 0.64-0.83. So the channel term is a regularizer --
it stops the search finding codes that only work noiselessly -- rather
than the source of the improvement.

Three consequences worth knowing before reasoning about this feature:

* **Encoder quality does not predict per-image gain.** r = +0.078 over
  the 10-image corpus. Two images with the same encoder PSNR (woods
  21.15, xchat1 20.33) differ 2x in headroom (+1.79, +3.47). What
  varies is how much room the *decoder* has left for that picture, not
  how badly the encoder did. Note this is the per-image story only; the
  recorded erosion of the gain across *checkpoints* is a separate and
  compatible effect.
* **Retention shrinks as the gain grows** (r = -0.603): a large clean
  improvement carries fine structure the channel destroys. So a clean
  PSNR gain overstates the delivered one by 20-35% -- a milder version
  of the same trap as `gain_db`'s ~3x, and it applies to any clean-only
  measurement of this feature.
* **The gain is spread uniformly across the three latent groups**
  (within-image spread 0.04-0.17 dB, each third carrying ~a third,
  summing to ~1.27x the whole). No group is special and there is no
  cross-group coordination to preserve.

**No torch.** This runs the published `decoder-grad` ONNX artifact on
the same onnxruntime the codec already uses -- one `Run()` per step,
Adam and the projection in numpy. That is the whole reason the gradient
is exported as a graph rather than computed with autograd.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from . import checkpoint
from .config import CHANNELS_PER_GROUP, MODES
from .latents import flat_to_latents, latents_to_flat

# Optimizing against the *clean* decoder is not a milder version of this
# -- it is harmful, losing on every fading channel measured, because it
# finds latents that decode beautifully in the absence of noise and fall
# apart in its presence. The objective runs through a channel model at
# this SNR instead.
#
# 5 dB is measured, and the optimum is flat: 2.5-7.5 dB are all within
# 0.2 dB of the peak on both test images and on two different
# checkpoints, which is why this is a constant and not an operator
# setting -- a setting would have to be guessed before transmitting, and
# a wrong guess is worse than a fixed compromise. The penalty is
# asymmetric: too *low* costs little, too high degrades toward the clean
# objective. If this ever has to move, move it down.
OBJECTIVE_SNR_DB = 5.0

# The SNR the *fading* profile wants, which is not the flat channel's.
# Swept over the 10-image corpus: +0.200 dB at 5, +0.237 at 7.5, +0.163
# at 10, +/-0.000 at 2.5. Note the asymmetry inverts -- for the flat
# channel too low costs little, here too low is where it dies -- so if
# this has to move, move it *up*.
FADING_OBJECTIVE_SNR_DB = 7.5

# Fraction of a *clean* decode gain that reaches the receiver, for
# turning progress into an approximate on-air figure. A clean PSNR gain
# predicts the delivered one at r = +0.988, so this is the whole
# estimator -- and it is a handwave for a structural reason: over 6
# images x 7 channel cells retention decomposes almost exactly into an
# image term plus a channel term (spreads 0.126 and 0.139, residual
# 0.022), and the sender cannot know the channel term. It runs 0.39 at
# mpp/3 dB to 0.77 at AWGN/12 dB; per cell and image, 0.22-0.90.
#
# 0.50 is mpp at 6 dB, picked to under-promise on a good path rather
# than over-promise on a bad one. Everything is monotone, so it ranks
# correctly even where it is numerically off -- but show it as
# approximate, never as a bare figure.
RETENTION = 0.50

# (probabilities, reported weight, error scale relative to a
# full-confidence latent), binned from a real transmission through the
# demodulator -- `scripts/derive_fading_profile.py`, mpp and mpd at 3/6/12 dB,
# 150k-600k latents per condition. The two are close enough across
# conditions to use one table; mpd is the slightly harsher of the pair
# and is what this is taken from, since the objective should err toward
# the worse channel (`docs/latent-optimization.md`).
#
# **Re-derived across 36 conditions (6 images x {mpp, mpd} x 3/6/12 dB)
# and it is stable**, which is what makes 27 numbers from one
# transmission safe to freeze into two implementations. Preset barely
# matters -- the top bin differs by 0.03 between mpp and mpd and the
# error column by a few percent -- so no per-preset table is needed.
# Nor does SNR: the error column is normalised by the top bin, so an
# SNR change scales every bin together and cancels, and the fraction of
# latents below w=0.2 is 0.029-0.032 across *all* 36 conditions.
#
# These values sit slightly to the conservative side of the pooled mean
# -- every low-confidence bin a few percent heavier, the top bin 0.473
# against a measured 0.484-0.529 -- and are deliberately left there
# rather than recalibrated. The gain below was measured with *this*
# table; replacing it would invalidate that number in order to chase a
# few percent, and the deviation is in the direction the design already
# says to err.
FADING_PROFILES = {
    "measured": (
        np.array([0.009, 0.025, 0.039, 0.053, 0.065, 0.110, 0.115, 0.111, 0.473]),
        np.array([0.05, 0.15, 0.25, 0.35, 0.45, 0.575, 0.725, 0.875, 0.975],
                 dtype=np.float32),
        np.array([10.73, 6.99, 4.44, 3.25, 2.59, 2.03, 1.64, 1.38, 1.00],
                 dtype=np.float32),
    ),
}

# Monte Carlo draws of the channel per step. The gradient is an average
# over noise realizations; 4 was enough for every measurement in the
# doc. Lowering it makes each step cheaper and noisier -- and weakening
# the channel term is exactly what reopens the failure above, so treat
# it as a numerical parameter rather than a performance knob.
CHANNEL_SAMPLES = 4

# --- the objective's channel -------------------------------------------
# `fading_profile` is **on by default as of 2026-08-04** and is what both
# implementations now optimize through; `cvar_frac` was measured and
# rejected. They live here rather than in a forked copy of the loop
# because two copies of an Adam loop diverge, and the thing being
# compared is a two-line difference inside it.
#
# Passing `fading_profile=None` restores the pre-2026-08-04 flat channel,
# which is what every measurement older than that was taken against --
# but `objective_snr_db` must go back to `OBJECTIVE_SNR_DB` with it. The
# two are on different scales and neither is right without the other.
#
# `fading_profile`: the objective's channel is per-latent i.i.d. AWGN at
# one SNR and full confidence. A real fade is deep and correlated in
# (carrier, time) -- but the interleaver's whole job is to scatter that,
# so after deinterleaving what reaches the decoder is i.i.d. *per-latent
# confidence* with the error that goes with it. `FADING_PROFILES` holds
# that joint distribution as measured off the real demodulator; a
# parametric fade was tried first and got both tails wrong (see the
# comment at the draw). Costs one extra draw per step, no extra graph
# run.
#
# **Measured 2026-08-04 and it wins, but only just: +0.198 dB of
# recovered picture over the flat channel across the whole 10-image
# corpus** (5 channel cells x 6 paired seeds each, 60 fixed steps), at
# 1x compute -- same `channel_samples`, same graph runs per step. It
# gains in every cell including AWGN, though the table was taken from
# mpd, and it gains on all 10 images.
#
# Do not quote a larger number from a single picture. A synthetic
# plasma stress test gained +1.45 dB, 7x the corpus mean and 2.5x the
# best corpus image, and nothing measured explains it. Two hypotheses
# for the spread were tested and both failed: the corpus is stratified
# by content (4 photographs, 3 screenshot/text, 3 between) and the
# stratum means are +0.207 / +0.212 / +0.174, i.e. flat, while the
# photo stratum alone spans +0.033 to +0.588. The gain does correlate
# with the encoder's *clean* PSNR on the image (+0.676, n=10) -- the
# opposite direction to the one guessed, and the opposite direction to
# how the headline optimization gain moves with encoder quality -- but
# at n=10 that is a lead, not a finding.
#
# The obvious confound was tested and rejected. The profile carries
# +5.2 dB (amplitude) / +7.4 dB (power) more noise than the flat model
# at the same nominal SNR, and the SNR optimum is known to be flat and
# to err low -- so "it is just a lower effective SNR" had to be ruled
# out. A flat channel at the matching level is far *worse* (-0.63 dB
# amplitude-matched, -1.36 dB power-matched), and the profile still
# gains +0.175 +/- 0.023 on 92% of cells when its nominal SNR is raised
# to 10.22 dB to put mean noise back where the baseline's is. The shape
# is what wins.
#
# **Its own SNR was then swept, and it wants 7.5 dB, not 5.** Full
# corpus, same protocol: +0.237 +/- 0.013 dB over `flat@5`, winning 94%
# of cells, beating it in **all five** channel cells (mpp and AWGN
# alike), and **never negative on any image** (worst +0.048) -- the bar
# the learning rate had to clear. The peak is gentle (+0.200 at 5,
# +0.237 at 7.5, +0.163 at 10), so like the flat channel's constant it
# is flat enough to be a constant rather than a setting.
#
# **But the asymmetry inverts, so do not transplant the rule above.**
# The flat objective's guidance is "if it has to move, move it down";
# for the profile, down is where it dies -- 2.5 dB scores +/-0.000 and
# -0.314 on its worst image. The profile already carries 5-7 dB of
# extra noise, so a low nominal number lands somewhere very deep. If
# this has to move, move it *up*.
#
# **Adopted 2026-08-04**, in both implementations in the same change --
# `native/core/optimize/optimize.hpp` carries the same table and the same
# pair of constants, since two implementations optimizing different
# objectives is the failure this whole exercise was arranged to avoid.
# The table is a property of the *demodulator* under fading, not of the
# codec, so a new checkpoint does not invalidate it; the *gain* should be
# re-measured per checkpoint, like the headline gain. The table itself is a
# property of the *demodulator* under fading, not of the codec, so a new
# checkpoint does not invalidate it -- but the gain should be
# re-measured per checkpoint, like the headline gain, since both are
# properties of the encoder's amortization gap.
#
# `cvar_frac`: reduce over the Monte Carlo draws with the mean of the
# worst `ceil(frac * samples)` instead of the mean of all. 1.0 is the
# current mean; 0.5 with 4 samples optimizes the worst two. This is the
# explicit version of the fragility flattening that currently happens
# only as a side effect.
#
# **Measured 2026-08-04 and it does not work.** 4 corpus images x 5
# channel cells x 6 paired seeds, 60 fixed steps, early stopping off:
# `cvar_frac=0.5` with `channel_samples=8` scored **-0.053 +/- 0.008 dB
# against the plain mean, winning 27% of cells -- while costing twice
# the graph runs. Stacked on `fading_profile` it adds +0.020 +/- 0.04,
# i.e. nothing. No equal-compute arm was needed: it lost at *double*
# compute, and halving its steps can only lower it. Kept as a parameter
# because the negative result is worth being able to reproduce, not
# because it is a pending option.
#
# The tail improvement it was meant to deliver came from modelling the
# channel correctly instead: `fading_profile` moved the worst cell from
# +0.76 to +1.22 dB. Optimizing the tail of a wrong channel model is
# not a substitute for a right one.
#
# **It is also not free, and the cost hides in the wrong parameter.** Lowering
# `cvar_frac` alone keeps the graph runs constant and buys the tail with
# variance instead -- measured gradient CV 0.318 at 0.5 and 0.445 at
# 0.25, against 0.231 for the mean. Restoring that variance means
# raising `channel_samples` to keep the *retained* count at 4 (8 for
# frac 0.5, 16 for 0.25), and `channel_samples` is graph runs per step:
# a 2x or 4x optimizer, in both implementations. So a CVaR variant is
# only worth adopting if it wins at equal *compute* -- half the steps
# for twice the samples -- not at equal steps, which is the comparison
# that makes it look free. If one does win, tilted sampling gets the
# same tail at 4 runs and should be built first; drawing 8 to discard 4
# spends half the budget discovering which draws were adverse, when the
# measured profile lets us construct them.
# -----------------------------------------------------------------------

# Swept 2026-08-02 on a 10-image corpus x 3 modes, against the previous
# 0.02, which was never swept -- it came from the prototype and
# survived. 0.05 is worth **+0.33 dB of recovered picture at 5 steps and
# +0.37 at 20** end to end (mode B, AWGN and mpp at 3/9 dB, 25 paired
# seeds), and it wins at every budget rather than trading short against
# long.
#
# Rates above this are *faster once moving and unstable starting*: Adam's
# first step has magnitude lr exactly (the bias correction cancels), and
# at 0.10 that first step overshoots badly enough that the run spends its
# whole budget recovering -- measured at -1.11 dB after one step, and 8
# of 30 cells still net negative after 10. A 2-step warmup into 0.10
# fixes that and was the sweep's winner on the objective, but it wins end
# to end by under 0.1 dB, which does not buy a schedule in two
# implementations. 0.05 is the largest rate that was never negative
# anywhere: worst cell over 30, +0.20 dB.
#
# Re-measure this on a new checkpoint rather than inheriting it, for the
# same reason the headline gain has to be re-measured: both are
# properties of the encoder's amortization gap, not of the optimizer.
# `scripts/latent_optim_lr_sweep.py` is the harness.
LEARNING_RATE = 0.05


@dataclass
class OptimizeResult:
    latents: np.ndarray          # flat, mode-length, unit RMS
    steps: int
    seconds: float
    stop_reason: str
    mse_start: float
    mse_best: float
    clean_gain_db: float = 0.0      # 0 unless clean_probe_every > 0

    @property
    def estimated_gain_db(self) -> float:
        """Approximate dB the *receiver* gains. See `RETENTION`."""
        return self.clean_gain_db * RETENTION

    @property
    def gain_db(self) -> float:
        """Improvement in the *objective*, which is not the on-air gain.

        Overstates what the receiver sees by roughly 3x (measured), and
        is not comparable across objectives at all -- on one image the
        flat objective reported 5.58 dB for a delivered 1.84 while the
        fading one reported 3.97 for a delivered 3.29, ranking the two
        backwards. Useful as progress, misleading as a headline -- quote
        the end-to-end figures in `docs/latent-optimization.md` instead.

        **What it is not:** an earlier version of this note called it
        latent-domain MSE against a noiseless decode. It is neither.
        The graph's `mse` output is image-domain against the source
        picture; what makes it overstate is that it is measured through
        the *objective's* channel rather than the receiver's. That
        matters because the fix is not to change the domain: a
        noise-free run of the same graph yields clean image PSNR
        directly (agreeing with `codec.decode` to 0.0013 dB), and a
        clean PSNR gain predicts the delivered one at r = +0.988 times
        a retention factor. See the module docstring.
        """
        if self.mse_best <= 0 or self.mse_start <= 0:
            return 0.0
        return 10.0 * math.log10(self.mse_start / self.mse_best)


def _session(path: str):
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 4  # as in codec.py; measured best there
    opts.log_severity_level = 3
    return ort.InferenceSession(path, opts, providers=["CPUExecutionProvider"])


def optimize(
    latents: np.ndarray,
    image: np.ndarray,
    mode: str,
    *,
    model: str | None = None,
    time_budget_s: float = 20.0,
    max_steps: int = 1000,
    patience: int = 10,
    min_rel_gain: float = 2e-3,
    objective_snr_db: float = FADING_OBJECTIVE_SNR_DB,
    channel_samples: int = CHANNEL_SAMPLES,
    fading_profile: tuple | None = FADING_PROFILES["measured"],
    cvar_frac: float = 1.0,
    clean_probe_every: int = 0,
    lr: float | Callable[[int], float] = LEARNING_RATE,
    seed: int = 0,
    progress=None,
    on_iterate=None,
) -> OptimizeResult:
    """Better latents for *this* picture, from the encoder's as a start.

    `latents` is a flat vector of at least this mode's length (the
    encoder's full output is fine; the tail is ignored). `image` is
    (3, H, W) float in [0,1] -- the picture as framed for transmission,
    not the file on disk.

    Stops on whichever comes first: a plateau, `time_budget_s`, or
    `max_steps`. **Not a fixed step count** -- per-step cost varies by
    an order of magnitude across the machines this ships to, so a count
    that is seconds on a desktop is minutes on a small board. The
    plateau test is the one that should normally fire; the budget is
    what makes this safe to run inside a transmit workflow; `max_steps`
    only backstops a loss that never plateaus.

    `progress(step, mse, elapsed)` is called each step if given.

    `clean_probe_every` > 0 additionally runs the graph with the noise
    switched off every N steps and reports the resulting *clean* PSNR
    gain on the result, which is a real picture-quality number rather
    than the objective proxy `gain_db` is. The graph's `mse` output with
    zero noise is image-domain PSNR against the source -- it agrees with
    a full `codec.decode` to 0.0013 dB -- so this needs no decoder
    session and a send-only station never fetches one. It costs one
    extra graph run per probe; at 5 with the default 4 draws, ~5%.

    `on_iterate(step, z, weights)` is a measurement hook, called with
    the *current* iterate before its update. It exists so a sweep can
    score every step of one run instead of re-running to each horizon;
    nothing in the transmit path passes it.

    `lr` is a constant by default; it may also be a callable
    `lr(step) -> float` (1-based) so a schedule can be swept without a
    second copy of this loop. `scripts/latent_optim_lr_sweep.py` is the
    only caller that passes one; until that sweep says otherwise the
    shipping value is the constant, and the C++ port has no schedule.
    """
    lr_at = lr if callable(lr) else (lambda _step: lr)
    spec = MODES[mode]
    active = spec.groups * CHANNELS_PER_GROUP

    grad_path = checkpoint.resolve_onnx(checkpoint.GRAD_PART, model)
    sess = _session(grad_path)
    # By position: the artifact names its inputs after the decoder's
    # convention, and positions are the part that is contractual.
    in_names = [i.name for i in sess.get_inputs()]
    out_names = [o.name for o in sess.get_outputs()]

    # The graph is mode C shaped, always. `flat_to_latents` splits into
    # three groups unconditionally, so a mode A/B vector has to be
    # padded to full length *before* reshaping rather than after -- the
    # short version reshapes without error and puts every coefficient in
    # the wrong place.
    from .codec import pad_to_full

    z = flat_to_latents(
        pad_to_full(np.asarray(latents[: spec.n_latents], dtype=np.float32))[None]
    ).astype(np.float32)

    weights = np.zeros_like(z)
    weights[:, :active] = 1.0
    z *= weights
    z = _project(z, active)

    target = np.ascontiguousarray(image, dtype=np.float32)[None]
    sigma = float(10.0 ** (-objective_snr_db / 20.0))
    rng = np.random.default_rng(seed)

    m = np.zeros_like(z)
    v = np.zeros_like(z)
    b1, b2, eps = 0.9, 0.999, 1e-8

    clean_mse_start = 0.0
    clean_gain_db = 0.0
    if clean_probe_every > 0:
        # Noise-free baseline, from the encoder's own latents, so every
        # later probe is a gain over the picture that would have gone
        # out unaided.
        _, _, m0 = sess.run(out_names, dict(zip(in_names, (z, weights, target))))
        clean_mse_start = float(m0)

    best_mse, best_z, best_step = math.inf, z.copy(), 0
    mse_start = math.inf
    started = time.perf_counter()
    stop = f"max_steps ({max_steps})"
    step = 0

    n_keep = max(1, math.ceil(cvar_frac * channel_samples))
    if n_keep == channel_samples:
        grads = mses = None  # the mean path allocates nothing extra

    for step in range(1, max_steps + 1):
        if n_keep < channel_samples:
            grads, mses = [], []
        grad = np.zeros_like(z)
        total = 0.0
        for _ in range(channel_samples):
            # The channel's Jacobian is the identity, so the gradient
            # the graph returns for the *noisy* latents is already the
            # one for `z`. That is why no channel model has to be
            # ported alongside this: noise in, `weights` back out as
            # the chain-rule factor.
            #
            # A per-latent gain keeps that property: scaling the noise
            # is still an additive perturbation independent of `z`, so
            # nothing about the chain rule changes and no second graph
            # run is needed.
            scale, w_draw = sigma, weights
            if fading_profile is not None:
                # Draw a reported confidence and the error that actually
                # accompanies it, from the *measured* joint -- not from a
                # parametric fade. Two things a shape parameter gets
                # wrong, both measured in `weightcal`:
                #
                #  * the receiver almost never reports near-zero
                #    confidence (under 3.5% of latents below w=0.2 on
                #    mpp and mpd alike, median w ~0.9), so both a
                #    Rayleigh gain and a hard erasure model a population
                #    that barely exists;
                #  * and where it does report low confidence it is
                #    *conservative* -- the pilot floor clamps the
                #    equalizer's amplification, so the error at w<0.1 is
                #    ~0.7x what 1/w predicts, while in the middle of the
                #    range it is 1.1-1.2x, i.e. mildly overconfident.
                #
                # Sampling the table keeps both, and keeps
                # `objective_snr_db` meaning what it did: the scale is
                # relative to a full-confidence latent's error.
                probs, ws, rel = fading_profile
                pick = rng.choice(len(probs), size=z.shape, p=probs)
                w_draw = (weights * ws[pick]).astype(np.float32)
                scale = (sigma * rel[pick]).astype(np.float32)
            noisy = z + rng.standard_normal(z.shape).astype(np.float32) * scale
            _, g, mse = sess.run(
                out_names, dict(zip(in_names, (noisy, w_draw, target))))
            if n_keep < channel_samples:
                grads.append(g)
                mses.append(float(mse))
            else:
                grad += g
            total += float(mse)
        if n_keep < channel_samples:
            # Mean over the worst draws only. Note the *reported* mse
            # stays the CVaR one too, so the plateau test and the
            # returned best iterate are judged on the same objective
            # that produced the gradient -- mixing them would stop the
            # run on a quantity it was not descending.
            worst = np.argsort(mses)[-n_keep:]
            grad = sum(grads[i] for i in worst) / n_keep
            mse = float(np.mean([mses[i] for i in worst]))
        else:
            grad /= channel_samples
            mse = total / channel_samples

        if step == 1:
            mse_start = mse
        if mse < best_mse * (1 - min_rel_gain):
            best_step = step
        if mse < best_mse:
            best_mse, best_z = mse, z.copy()
        if clean_probe_every > 0 and (step == 1 or step % clean_probe_every == 0):
            # The *current* iterate, not the best one: the operator is
            # watching, and a number that tracks the search is easier to
            # read than one that stalls while the loss wanders.
            _, _, mc = sess.run(out_names, dict(zip(in_names, (z, weights, target))))
            if float(mc) > 0.0 and clean_mse_start > 0.0:
                clean_gain_db = 10.0 * math.log10(clean_mse_start / float(mc))
        if progress is not None:
            progress(step, mse, time.perf_counter() - started)
        if on_iterate is not None:
            on_iterate(step, z, weights)

        if step - best_step >= patience:
            stop = "plateau"
            break
        if time.perf_counter() - started >= time_budget_s:
            stop = "time budget"
            break

        m = b1 * m + (1 - b1) * grad
        v = b2 * v + (1 - b2) * grad * grad
        z = z - lr_at(step) * (m / (1 - b1**step)) / (
            np.sqrt(v / (1 - b2**step)) + eps)
        z = _project(z * weights, active)

    # The best iterate, not the last: the loss reported at a step
    # belongs to the latents that went *into* it, so the final update is
    # always unmeasured and returning it would sometimes ship a step
    # past the minimum.
    flat = latents_to_flat(best_z)[0][: spec.n_latents]
    return OptimizeResult(
        latents=flat.astype(np.float32),
        steps=step,
        seconds=time.perf_counter() - started,
        stop_reason=stop,
        mse_start=mse_start,
        mse_best=best_mse,
        clean_gain_db=clean_gain_db,
    )


def _project(z: np.ndarray, active: int) -> np.ndarray:
    """Back onto the unit-RMS shell, over the transmitted groups only.

    That normalization is the on-air contract between encoder, modem and
    training, not a training detail -- and `Modem.modulate` normalizes
    over the *truncated* vector, so a mode A/B optimization that
    normalized over all 132 channels would be solving a different
    problem than the radio poses.
    """
    a = z[:, :active]
    rms = np.sqrt((a * a).mean(axis=(1, 2, 3), keepdims=True))
    return (z / np.maximum(rms, 1e-6)).astype(np.float32)

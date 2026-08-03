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

# Monte Carlo draws of the channel per step. The gradient is an average
# over noise realizations; 4 was enough for every measurement in the
# doc. Lowering it makes each step cheaper and noisier -- and weakening
# the channel term is exactly what reopens the failure above, so treat
# it as a numerical parameter rather than a performance knob.
CHANNEL_SAMPLES = 4

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

    @property
    def gain_db(self) -> float:
        """Improvement in the *objective*, which is not the on-air gain.

        Latent-domain MSE against a noiseless decode overstates what the
        receiver actually sees by roughly 3x (measured). Useful as
        progress, misleading as a headline -- quote the end-to-end
        figures in `docs/latent-optimization.md` instead.
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
    objective_snr_db: float = OBJECTIVE_SNR_DB,
    channel_samples: int = CHANNEL_SAMPLES,
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

    best_mse, best_z, best_step = math.inf, z.copy(), 0
    mse_start = math.inf
    started = time.perf_counter()
    stop = f"max_steps ({max_steps})"
    step = 0

    for step in range(1, max_steps + 1):
        grad = np.zeros_like(z)
        total = 0.0
        for _ in range(channel_samples):
            # The channel's Jacobian is the identity, so the gradient
            # the graph returns for the *noisy* latents is already the
            # one for `z`. That is why no channel model has to be
            # ported alongside this: noise in, `weights` back out as
            # the chain-rule factor.
            noisy = z + rng.standard_normal(z.shape).astype(np.float32) * sigma
            _, g, mse = sess.run(
                out_names, dict(zip(in_names, (noisy, weights, target))))
            grad += g
            total += float(mse)
        grad /= channel_samples
        mse = total / channel_samples

        if step == 1:
            mse_start = mse
        if mse < best_mse * (1 - min_rel_gain):
            best_step = step
        if mse < best_mse:
            best_mse, best_z = mse, z.copy()
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

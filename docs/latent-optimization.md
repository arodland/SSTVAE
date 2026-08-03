# Transmit-time latent optimization

**Status: implemented, end to end.** Measured, published, and shipping
in both implementations as of 2026-07-31:

| | |
|---|---|
| artifact | `v3-decoder-grad-fp32.onnx`, published with the v3 codec |
| Python | `sstvae/latent_optim.py`; `sstvae_encode.py --optimize [SECONDS]` |
| C++ | `native/core/optimize/` (the loop) + `core/codec/grad_session.cpp` (ORT) |
| app | `native/gui/tx_panel.cpp`, behind `transmit.optimize` |
| measurement | `scripts/latent_optim_{prototype,roundtrip,precision}.py` |

**The gain is 1.4–1.8 dB of recovered picture**, in every mode, on
every channel model tested, on both checkpoints and at every decoder
precision — with one condition: the objective must include the channel.
See below; with the clean objective this feature makes pictures *worse*
on every fading channel tested. The one parameter it needs is a
constant.

Not done: a real on-air shakedown. Everything here is simulation and
loopback.

Measurements are from 2026-07-31 on an x86-64 Linux box (24 cores, CPU
execution provider). **The headline tables were re-run against
`sstvae-cc12-epoch438.pt`**, the checkpoint going forward, after an
initial pass on `sstvae-np1-epoch317.pt` (the published v2 codec);
where the two are compared it is labelled. Everything held. The mode B
/ 3 dB figure and the 5 dB optimization SNR are Andrew's.

## The idea

The encoder is *amortized*: one forward pass, trained to minimize loss
averaged over the training set. That is not the same as minimizing loss
for the picture actually in front of it. For any given image there is
generally some other latent vector that the **same frozen decoder**
reconstructs better, and the distance between the two is the
*amortization gap*.

Nothing about the encoder's output is privileged at transmit time. The
decoder is a fixed function; the latents are just its input. So the
sender can spend compute searching for a better input — gradient
descent on the latents, decoder weights frozen — and transmit what it
finds.

The budget is lopsided in our favour. The encoder is 31 ms. A
transmission is 32–95 s. Spending 20 s of that on a search that buys
several dB is a trade a station makes gladly, and it is paid entirely
by the sender, once, before keying.

**The receiver needs no change, and neither does the on-air format.**
Optimized latents are latents: same count, same unit-RMS contract, same
modem, same waveform. Every existing station decodes them, including
the Python one and the native one, with no version negotiation and no
compatibility tier. This is the property that makes the feature cheap
to ship, and it should not be given up without a very good reason.

## The latent-domain numbers, which started this and should not be quoted

**Kept as history, not as evidence.** These are `np1`, measured in the
latent domain through the differentiable channel — i.e. the objective
being optimized, scored on itself. They are the numbers that made the
idea look worth pursuing, and they overstate it by roughly 3×. The
figures to quote are the end-to-end ones in the next section.

Recovered PSNR against the encoder's own output, same decoder, same
image:

| image | mode | objective | clean | @10 dB channel |
|---|---|---|---|---|
| certificate (text, off-distribution) | C | clean | 21.69 → **28.60** (+6.92) | 21.17 → 24.55 (+3.38) |
| certificate | C | channel-aware @10 dB | 21.69 → 27.35 (+5.66) | 21.17 → **25.60** (+4.43) |
| photograph | C | channel-aware @10 dB | 25.25 → **28.85** (+3.60) | 24.79 → **27.83** (+3.04) |
| photograph | A | clean | 24.14 → 27.98 (+3.84) | 23.35 → 25.53 (+2.18) |

Three things this says:

- **The gap is large where the theory says it should be.** The
  certificate is off-distribution for a photograph-trained codec, and
  it gains the most. `docs/onnx.md` already records that this class of
  image is where the codec is weakest (int8 cost 0.002 dB on
  photographs against 1.54 dB on synthetic probes) — the same weakness,
  seen from the other side.
- **It is not only an off-distribution trick.** A real camera
  photograph, in-distribution, still gains +3.6 dB here. Andrew
  measured ~3 dB at mode B on a 3 dB channel, with text legibility
  "greatly improved" — which is the outcome that matters more than the
  number, and which *did* survive the end-to-end check even though the
  dB figure did not.
- **Optimizing against the clean decoder is the wrong objective.** The
  clean-objective row wins by 1.25 dB on a clean decode and *loses* by
  1.05 dB where it counts. Optimizing through the channel model gives
  up some clean-channel PSNR to buy more under noise. The channel is
  not a detail to add later; it changes which latents you pick.
  The end-to-end run below shows this is not a tuning preference — the
  clean objective is *actively harmful* on a real fading channel.

## The end-to-end measurement, which is the one that counts

`scripts/latent_optim_roundtrip.py`: encode → modulate → HF channel →
demodulate → decode, PSNR of the picture that comes out against the
picture that went in. Both payloads go through the **same channel
realization** (same seed), so the comparison is between latents rather
than between noise draws. Mode B, 25 seeds per cell, 2026-07-31.

**This measurement overturned the design, and it is the reason the
latent-domain numbers above are kept but not trusted.** Optimizing MSE
at an assumed SNR and reporting MSE at that same SNR was never
evidence; here is what happens when you ask the radio instead.

Δ PSNR of the recovered picture, optimized vs encoder latents:

| | | clean objective | channel-aware @5 dB |
|---|---|---|---|
| photograph | AWGN 9 dB | +0.21 | **+2.20** |
| | AWGN 3 dB | −1.05 | **+1.69** |
| | mpp 9 dB | −1.21 | **+1.57** |
| | mpp 3 dB | −1.54 | **+1.15** |
| certificate | AWGN 9 dB | −0.41 | **+2.70** |
| | AWGN 3 dB | −2.09 | **+1.68** |
| | mpp 9 dB | −2.24 | **+1.54** |
| | mpp 3 dB | −2.71 | **+0.65** |

Four findings, in order of how much they change the plan:

- **The clean objective is not merely suboptimal, it is harmful.** It
  loses on every fading cell on both checkpoints, up to −2.71 dB. A
  feature shipped on that objective would have made pictures worse for
  any station on a realistic path while reporting a large gain in its
  own logs. This is the whole argument for why the differentiable
  channel belongs *in* the objective.
- **Channel-aware at 5 dB wins every cell**, on both checkpoints, both
  images, all four channel models and 0–12 dB. The 5 dB figure is
  **Andrew's, from a hand sweep** (2026-07-31); it was tried first for
  that reason and did not need adjusting. Note it is deliberately
  *below* most operating points — optimizing for a harder channel than
  the expected one is what buys robustness across the whole range.
- **The honest gain is 1.5–2 dB, roughly a third of what the latent
  domain claimed.** The certificate's +6.9 dB became +1.7 dB at a
  comparable operating point. Quote the end-to-end number; the
  latent-domain one is an objective value, not a result.
- **Acquisition is unaffected.** No-lock counts track the encoder's
  cell for cell (7/7, 8/8, 2/2, …) and every successful reception
  reported 100% of frames. The optimization changes what the carriers
  carry, not whether they are found — worth stating because a
  transmit-side change that cost sync would be far worse than one that
  cost a dB.

Gains grow with SNR and shrink with fading severity, which is the shape
you would expect: both leave less headroom for a better input to
exploit. The weakest cell measured is the certificate at mpp/3 dB,
+0.65 dB — on a poor path near threshold the feature earns less, but it
has never once cost anything with the channel in the objective.

### The gain shrinks as the codec improves, and that is expected

Worth recording because it bears on how long this feature stays worth
its complexity. Going from `np1` to `cc12` (mode B, mean over the four
cells):

| | encoder baseline | optimizer gain |
|---|---|---|
| photograph | 22.32 → 22.42 (+0.10) | +1.69 → +1.65 |
| certificate | 18.49 → 19.06 (**+0.56**) | +1.85 → **+1.64** |

The certificate is the informative row: `cc12` is 0.56 dB better on it
without any optimization — which is what a broader training set was
*for* — and the optimizer's gain falls by 0.21 dB on the same picture.
That is the amortization gap closing from the other side. The station
still comes out ahead overall (18.49 + 1.85 = 20.34 against
19.06 + 1.64 = 20.70), so a better encoder plus optimization beats
either alone.

The forward-looking version: **this feature's value is anti-correlated
with encoder quality on the picture in question**, so expect the
headline number to erode as checkpoints improve, fastest on exactly the
off-distribution content where it looks most impressive today. It is
not a reason to hold off — 1.6 dB now is 1.6 dB — but a re-measurement
belongs in the checklist for any future codec revision, and "it used to
be worth 2 dB" should not be taken on trust.

### The objective SNR is a constant, not a setting

Swept 2026-07-31, mode B, 25 seeds, mean Δ across four channel cells
(AWGN and mpp at 3 and 9 dB):

| objective | photograph | certificate |
|---|---|---|
| clean | −0.90 | −1.86 |
| 0 dB | +1.27 | +1.28 |
| **2.5 dB** | **+1.67** | **+1.66** |
| **5 dB** | **+1.65** | **+1.64** |
| 7.5 dB | +1.48 | +1.52 |
| 10 dB | +1.29 | +1.28 |
| 15 dB | +0.74 | +0.62 |

**The optimum is flat from 2.5 to 5 dB** — those two are within 0.02 dB
of each other on both images, and everything from 2.5 to 7.5 is within
0.2 dB of the peak. So this ships as a constant. It does **not** become
an operator setting, which is the good outcome: a setting for "the
channel you expect" would have to be guessed before transmitting, and a
wrong guess is worse than a fixed compromise.

Which constant does not matter much, and the two checkpoints disagree
about it *within noise* — `np1` peaked at 5 dB (+1.69/+1.85 against
2.5 dB's +1.66/+1.84), `cc12` at 2.5 dB. A parameter whose optimum
moves by one grid step between checkpoints, for two hundredths of a dB,
is a parameter that has been characterized rather than tuned. **5 dB
stays**, on the grounds that nothing here is evidence to change it;
2.5 dB would be equally defensible and marginally more conservative.

**The penalty is asymmetric, and the dangerous direction is
optimistic.** Ten dB too low costs 0.45 dB; ten dB too high costs
0.77 dB and keeps degrading toward the clean objective, which is
actively harmful. If a future change has to err, it should err toward
assuming a *worse* channel than expected.

Per-cell optima do drift the way you would expect — the best channel
(AWGN 9 dB) peaks nearer 7.5 dB and the worst (mpp 3 dB) nearer 2.5 dB
— so a station that genuinely knew its SNR could gain another
0.3–0.7 dB at the extremes. Not worth having: it buys a fraction of a
dB in exchange for a setting that is wrong whenever the operator's
estimate is.

### The learning rate is 0.05, swept 2026-08-02

It was 0.02 until then, and 0.02 was never measured — it came from the
prototype and survived by not being questioned. Swept on a 10-image
corpus (`data/optim_corpus`) across all three modes, one run per cell to
50 steps with the objective recorded at **every** step, so any stopping
point could be read off afterwards rather than chosen in advance:
`scripts/latent_optim_lr_sweep.py`, summarized by
`scripts/latent_optim_lr_summary.py`.

End-to-end confirmation, Δ recovered PSNR against encoder latents, mean
over AWGN and mpp at 3 and 9 dB, 25 paired seeds, mode B, both test
images (`scripts/latent_optim_lr_roundtrip.py`):

| rate | 5 steps | 20 steps | 50 steps |
|---|---|---|---|
| 0.02 (was) | +0.43 | +1.04 | +1.39 |
| **0.05** | **+0.76** | **+1.32** | **+1.54** |
| warm2→0.10 | +0.77 | +1.41 | +1.58 |

**+0.33 dB at 5 steps and +0.37 at 20, free.** It wins at every budget
rather than trading short against long, which matters because *the
budget is not ours to choose*: the shipping loop stops on a plateau or a
20 s clock, and how many steps that buys depends on hardware we do not
have. The measurement box is faster than almost any station's, so the
short-budget column is the one to weight.

**Rates above 0.05 are faster once moving and unstable starting**, and
that is the whole shape of the result. Adam's first step has magnitude
`lr` exactly — the bias corrections cancel — so a large rate takes one
enormous step before any curvature is known. At 0.10 the objective is
**−1.11 dB after one step** and still oscillating at three; at 0.20,
−3.94. They recover and end up level (everything from 0.05 to 0.20 lands
within 0.13 dB at 50 steps, so the curve *flattens* rather than turning
over) — but a station that only got 10 steps would have shipped a worse
picture than the encoder's on 8 of 30 cells at 0.10, and 21 of 30 at
0.15.

So the choice was the largest rate that was **never negative on any cell
measured**, worst case +0.20 dB, rather than the best mean. A two-step
linear warmup into 0.10 removes the instability and won the objective
sweep outright (26–29 of 30 paired cells), but it wins end to end by
under 0.1 dB — inside what two images and one mode can resolve — and
that does not buy a schedule in two implementations plus a parity case.
The safety argument is Andrew's and is the deciding one: 0.05 has passed
every test it has seen, and the corpus is ten pictures, so the worst
first step *available* is not the worst one that exists.

Three notes for whoever re-runs this:

- **Re-measure it on a new checkpoint.** Like the headline gain, this is
  a property of the encoder's amortization gap rather than of the
  optimizer.
- **Decay schedules were tested and all lost** — exponential, 1/t, at
  several time constants — each to the constant it starts from, at both
  budgets. Annealing is not what this needs; the problem is at step 1,
  not at step 50.
- **A schedule must be a function of the step alone.** The optimizer
  never knows how many steps it will get, so a cosine annealed to a
  known horizon would score as a policy that could not have been run.
  That rules out anneals and admits warmups.

### All three modes, and it does not matter which

Mean Δ at the 5 dB objective, same four cells, 25 seeds:

| mode | latents | photograph | certificate |
|---|---|---|---|
| A | 1 group | +1.58 | +1.44 |
| B | 2 groups | +1.65 | +1.64 |
| C | 3 groups | +1.67 | +1.75 |

Near-flat, with a slight monotone trend — more latents, marginally more
gain, most visible on the certificate (+1.44 → +1.75). Small enough
that it changes nothing; every cell in all three modes is positive on
both checkpoints, and mode A is still worth +1.5 dB.

This was checked because it was cheap, not because it was doubted, and
the direction is mildly informative: mode A has a third the
coefficients, and one plausible story had the gain *growing* as the
payload shrank (each coefficient carries more). The opposite happened,
so the gain tracks the size of the search space rather than the
pressure on each value.

### The gain survives every decoder precision, including int8

The sender optimizes against fp32 and cannot know what the receiver
loaded. Optimized once, transmitted, then the *same receptions* decoded
three ways (`scripts/latent_optim_precision.py`, mode B, 25 seeds):

| | encoder | optimized | Δ |
|---|---|---|---|
| photograph fp32 | 22.55 | 24.31 | +1.75 |
| photograph fp16 | 22.55 | 24.31 | +1.75 |
| photograph int8 | 22.52 | 24.29 | **+1.77** |
| certificate fp32 | 19.17 | 21.03 | +1.86 |
| certificate fp16 | 19.17 | 21.03 | +1.86 |
| certificate int8 | 19.15 | 21.01 | **+1.86** |

**Identical to a hundredth of a dB**, int8 included — if anything a
hair better, which is noise. So there is no compatibility tier here and
nothing to warn operators about: a station running the int8 decoder
gets the same benefit as one running fp32.

The reason is visible in the absolute columns. Quantisation costs the
encoder's latents and the optimized latents *the same* ~0.03 dB, so it
is a uniform offset that cancels in the difference. The optimized
latents are not exploiting fine detail in the weights — they are
ordinary unit-RMS latents well inside the decoder's normal operating
regime, which is what the channel-aware objective and the RMS
projection are for. This is worth stating because the opposite was a
reasonable fear: "tuned against one set of weights" sounds exactly like
something quantisation would blunt.

Note the ~0.03 dB int8 penalty here against `docs/onnx.md`'s −0.318 dB.
Not a contradiction — that figure is a clean decode, and under channel
noise the quantisation error is far below the channel's.

### The regularizer is dead weight, and the reason is instructive

Mean Δ across the four cells, L2 pull toward the encoder's latents
swept against both objectives:

| reg | 5 dB objective (photo / cert) | clean objective (photo / cert) |
|---|---|---|
| 0 | +1.65 / +1.84 | −0.85 / −1.53 |
| 1e-4 | +1.70 / +1.83 | −0.74 / −1.43 |
| 1e-3 | +1.69 / +1.86 | −0.12 / −0.89 |
| 1e-2 | +1.34 / +1.73 | +0.79 / +0.26 |
| 1e-1 | +0.73 / +1.10 | +0.66 / +0.78 |

**With the channel in the objective the regularizer does nothing**:
0, 1e-4 and 1e-3 are within 0.05 dB of each other, which is noise. Above
that it is purely a cost — −0.35 dB at 1e-2 and −0.96 dB at 1e-1 on the
photograph. So **the default is now 0** and the term should not be
ported to C++. It is kept as a parameter in the prototypes only so this
measurement stays reproducible.

> This sweep is the one table still measured on `np1`, and at the old
> 1e-3 default. It was not re-run against `cc12`: what it establishes
> is that the term does nothing, and the `cc12` tables above — all
> taken at **reg=0** — hold up unchanged, which is the same conclusion
> arrived at from the other direction. The correction it implies to the
> old `np1` tables was −0.04 dB on the photograph and −0.02 on the
> certificate: noise rather than trend, since 1e-4 scores *above* 1e-3
> on the photograph, so the ordering is not even monotone.

The clean-objective column is why this is worth writing down rather
than just deleting. Regularization *does* work there — it drags the
clean objective from −0.85 to +0.79, turning a harmful configuration
into a mildly useful one. So the two are alternatives, and the
comparison is decisive: the best clean+regularizer result (+0.79) is
less than half the worst channel-aware one (+1.65 at no
regularization).

The mechanism explains the ranking. An L2 pull constrains how far the
latents may move *in every direction at once*, so it limits the damage
and the gain together — a blunt instrument that buys safety by
forbidding the search. The channel term constrains only the directions
that are actually fragile, leaving the rest of the search free. That is
why it dominates, and it is a reason to be suspicious of any future
"cheaper approximation to the channel" that turns back into a norm
penalty: it would be this row, not the one above it.

### PAPR: measured, and it is a non-issue

| | encoder | optimized | Δ |
|---|---|---|---|
| photograph | 4.37 dB | 4.46 dB | **+0.09** |
| certificate | 4.56 dB | 4.38 dB | **−0.18** |

Under a fifth of a dB, in both directions, on the waveform as
transmitted — and it stays there across every objective SNR in the
sweep above and both checkpoints (worst case −0.18 dB), so this is a
property of the optimization rather than of one lucky setting. Note the
sign is not even consistent: the photograph goes up and the certificate
down on `cc12`, and both were within ±0.05 on `np1`. That is a wash
being measured, not an effect.
This confirms Andrew's prediction and its reasoning: the clipper is
effective, post-clip PAPR increase is an order of magnitude below
pre-clip, and stage-2 trained through that same clipper. An earlier
draft of this doc called PAPR the risk that could sink the feature.
It was not; the *objective* was.

## What the optimization actually is

`scripts/latent_optim_prototype.py`. Adam on the latent tensor, decoder
frozen, MSE against the source image, initialized from the encoder's
output. Three constraints keep it honest, and each exists for a reason:

- **Project onto the unit-RMS shell every step.** That normalization is
  the on-air contract between encoder, modem and training, not a
  training detail — and for a truncated mode it is over the
  *transmitted* groups only, matching what `Modem.modulate` actually
  normalizes. An optimizer allowed off that shell is solving a
  different problem than the one the radio will pose.
- ~~**Regularize toward the encoder's output**~~ (small L2). This was
  in the design as the guard against finding an adversarial input that
  decodes beautifully in the simulator and falls apart on air. **It was
  measured and removed** — the channel term does that job, and does it
  better; see below. Default is 0.
- **`--mode` restricts everything to that mode's channel groups**, so
  the optimization, both reported reconstructions, and the WAV all
  describe the same transmission. Optimizing all three groups and then
  sending one would report a gain the receiver never sees.

The channel-aware objective is `apply_latent_channel` in the loop with
a fixed SNR and several Monte Carlo draws per step — the same
differentiable channel `scripts/train.py` trains through, so the
objective and the training objective agree about what a channel is.

The script also writes real transmit audio (`--wav-out`,
`--wav-out-baseline`), through `Modem.modulate` exactly as
`sstvae_encode.py` does, so an A/B pair can be run through
`sstvae_simulate.py`, decoded, or played over a radio.

## Getting it into the native app without torch

This is the part that determines whether the feature can ship, because
`native/` has no torch by design and is not getting any. Gradient
descent needs `d(loss)/d(latents)` through the decoder, which sounds
like it needs an autodiff engine.

It does not. **Export the vector-Jacobian product as an ONNX graph, at
publish time**, and the runtime side becomes arithmetic.

`scripts/decoder_vjp_prototype.py` builds a module whose *forward* is
`(z, weights, target) -> (recon, grad_z, mse)`: it runs the decoder,
recording each primitive's input, then walks the tape backwards using
**hand-derived backward formulas written as ordinary forward tensor
ops** — `conv_transpose2d` for a Conv2d's input-gradient and `conv2d`
for a ConvTranspose2d's, the GroupNorm backward formula spelled out,
elementwise derivatives for SiLU and sigmoid, and a pass-through plus
branch-sum for each ResBlock. Then it exports *that*.

Only the gradient with respect to the decoder's **input** is ever
needed — the weights are frozen — so this is a fixed, finite chain, not
a general autodiff problem.

**Do not implement this by calling `torch.autograd.grad` inside the
module and exporting through it.** That asks the exporter to trace a
double-backward graph and it hits ops with no ONNX symbolic
registered. Written as forward ops, every op involved is one the
exporter already handles at opset 17, and the export is as ordinary as
the decoder's own.

### Verified three ways

A wrong gradient does not announce itself — it optimizes slowly, or to
the wrong place, and still produces a plausible picture. So:

| check | result |
|---|---|
| hand-derived backward vs `torch.autograd.grad`, mode C | `recon` exact; `grad_z` max 2.3e-9 abs, 9.3e-7 rel |
| same, mode A (groups masked off) | `grad_z` max 2.3e-10 abs, 3.4e-7 rel |
| exported ONNX vs the torch module | `recon` 7.1e-6, `grad_z` 5.8e-9 |
| ORT-only optimization loop, certificate | +5.57 dB in 75 steps / 25 s |

The masked check is not redundant: `weights` is both a forward mask and
the chain-rule factor on `grad_z`, and a version that applied it in the
forward pass and forgot it in the backward passes the unmasked test.

The `recon` difference of 7.1e-6 is larger than the decoder's
exact-parity claim in CLAUDE.md, and that is fine — this is a separate
fused graph used only to *drive the optimizer*. The picture the
receiver reconstructs still comes from the published decoder artifact,
so the codec parity claim is untouched. Do not "fix" this by routing
reconstruction through the VJP graph.

The artifact is 18.2 MB at fp32, opset 17, exported through the
existing `dynamo=True` path.

### The runtime side is a loop, not a framework

One ORT session, one `Run()` per step, Adam in about six lines of
elementwise arithmetic, and the unit-RMS projection. No autodiff, no
new dependency, no training-capable onnxruntime build — the same pinned
inference runtime `native/cmake/onnxruntime.cmake` already fetches.
`optimize_through_onnx` in the prototype is that loop with numpy
standing in for the C++.

The channel-aware objective needs **no graph change at all**: noise is
added to `z` before the call, and `weights` comes back as the
chain-rule factor already. A ported `latent_channel.py` is not needed;
Gaussian noise and a mask is a dozen lines.

### Two design decisions that shape the port

**The MSE loss lives inside the graph.** Taking `d(loss)/d(recon)` as an
input would keep the graph loss-agnostic, but no loss of the
reconstruction can be evaluated before the reconstruction exists — so
the caller would run the graph once for `recon` and again for the
gradient, doubling every step. Loss-agnosticism costs exactly 2×.
A different loss is a re-export, not a runtime flag.

**Stopping is plateau / time budget / max steps, whichever fires
first** (Andrew, 2026-07-31), and the loop returns the **best** iterate
rather than the last. Per-step cost varies by an order of magnitude
across the machines this ships to — 314 ms/step here, so 100 steps is
31 s on this box and minutes on a small board — which makes a fixed
step count a hardcoded assumption about the operator's hardware. All
three bounds are kept and they answer different questions: the plateau
test is the one that should normally fire, the time budget is what
makes the feature safe to run inside a transmit workflow, and
`max_steps` only backstops a loss that never plateaus. Best-not-last
because the loss reported at a step belongs to the latents that went
*into* it, so the final update is always unmeasured.

Both stopping paths were driven and observed to fire.

### What a time budget actually buys

Objective gain against elapsed time, mode B, certificate, on the 24-core
box (~1.05 s/step: four channel draws at ~260 ms each):

| elapsed | steps | objective gain | share of achievable |
|---|---|---|---|
| 20 s | 19 | 1.62 dB | 65% |
| 30 s | 30 | 1.92 dB | 77% |
| 53 s | 50 | 2.21 dB | 89% |
| 88 s | 83 | 2.49 dB | 100% (plateau) |

**The shipped default is 20 s** (Andrew, 2026-07-31) — about two thirds
of the gain for a fraction of the wait, with `--optimize 90` available
when the picture is worth it. Worth stating plainly because every
measurement in this document used 300 steps, i.e. the far right of that
curve: the tables above are what the feature can do, not what the
default does.

The curve is steep early and flat late, which is what makes a
smaller default defensible rather than merely cheaper — and the plateau
test means a fast machine stops on its own instead of spending the
budget it was given.

Cutting the Monte Carlo draws from 4 to 2 would double the steps per
second and was **declined** (Andrew, 2026-07-31) pending measurement:
weakening the channel term is exactly what produced the off-manifold
failure, so it is not a free performance knob.

## Work to do

### 1. Measurement — **DONE 2026-07-31**

**The measurement phase is complete, and was re-run in full against
`cc12`.** PAPR, the end-to-end round trip, the objective-SNR sweep, all
three modes and all three decoder precisions are done — see above.
Answered: the feature works (+1.4–1.8 dB), the objective must include
the channel, 5 dB is a constant rather than a setting, the gain is
near-mode-independent, PAPR is unchanged, acquisition is unaffected,
and the gain survives int8. What is left before implementation:

- **fp16 for the gradient artifact.** Gradients have wider dynamic
  range than activations, so fp16 is not free by the same argument that
  made it free for inference. Measure before publishing one. **Skip
  int8 entirely** — differentiating `ConvInteger` is not a well-defined
  operation, and the artifact would be a trap.
Nothing. **All measurement questions are closed** — the last one, the
regularizer, was swept and the term removed.

### 2. Publish the gradient artifact — **DONE 2026-07-31**

`v3-decoder-grad-fp32.onnx` is published alongside the v3 (cc12) codec
at `arodland/sstvae`, 18.2 MB, verified against `torch.autograd.grad` at
rel-RMS 3.6e-05 / cosine 1.000000. `scripts/export_onnx.py` produces it
in the same run as the encoder/decoder pair, from the same checkpoint,
carrying the same `sstvae.source_sha256` stamp. The in-code default
stays at v2 for now.

**fp32 only.** The export attempts fp16 every run and skips it
non-fatally: with `keep_io_types` the converter leaves `target` at fp32
while `recon` becomes fp16 and feeds the same `Sub`, giving a graph
onnxruntime refuses to load. Converting the IO too would push
half-precision buffers into the C++ caller for a 9 MB saving on an
opt-in feature, in the one numerical regime where fp16 is least
obviously safe. Attempted rather than removed so a future converter
that handles it is picked up for free. An **fp32** failure is fatal, by
contrast — that means the hand-derived backward or the export is wrong.

`verify_grad` runs half its probes under a mode-A mask, because
`weights` is both a forward mask and the chain-rule factor on `grad_z`;
an implementation that applied it forward and dropped it backward
passes an unmasked check. It reports cosine alongside relative RMS
because they fail differently: Adam absorbs a scale error and cannot
recover a direction error.

The original plan for this section follows, and still describes the
naming rule.

### 2a. Original notes on publishing

Fold the export into `scripts/export_onnx.py` and publish
`<stem>-decoder-grad-fp32.onnx` beside the existing pair, named for
whichever checkpoint is being published — **not `v1`**: v2
(`sstvae-np1-epoch317.pt`) is the current published codec, and
`sstvae-cc12-epoch438.pt` is the one going forward, so the artifact
should be generated in the same run as the encoder/decoder pair it
belongs to rather than named ahead of time. It must carry the
same `sstvae.source_sha256` stamp and be cross-checked against the
decoder the same way the encoder/decoder pair check each other: a
gradient graph from a *different* checkpoint than the decoder in use
would optimize toward the wrong picture and report improving loss the
whole time. That is the same silent-wrong-picture failure the existing
cross-check exists to prevent, and it deserves the same treatment.

It is a third artifact and a fourth download, so it must be fetched
**only when the feature is used** — per-part laziness is already the
rule in `codec.py` and `checkpoint::resolve_onnx`, and a receive-only
station must never pull 18 MB it has no use for.

### 3. Productionize the Python side — **DONE 2026-07-31**

`sstvae/latent_optim.py`, torch-free: it runs the published gradient
graph on the same onnxruntime the codec uses. `sstvae_encode.py
--optimize [SECONDS]` is the opt-in flag. It deliberately never
appeared in `sstvae/gui/`, which was frozen at the time and deleted on
2026-08-01 — the native app is where it lands.

Two things in `checkpoint.py` earned tests (`test_checkpoint.py`), both
of which fail in ways that still produce a picture:

- **The gradient artifact is fp32 whatever `--precision` says.** It is
  the only precision published, and `--precision` is a statement about
  the *codec*; refusing to optimize because someone picked int8 for
  their decoder would answer a question they did not ask. Verified
  end to end with `--precision int8`, which would fail if the override
  were missing.
- **`-decoder-` is a substring of `-decoder-grad-`.** A containment
  test hands back the gradient graph when the decoder was asked for —
  and it loads, and its first output *is* a reconstruction, so the
  mistake survives to a wrong picture rather than an error. The sibling
  derivation had the same shape of bug in the precision: rebuilding
  `{stem}-{part}-{precision}` is right, substituting into the given
  name is not, because the gradient sibling of an fp16 encoder is fp32.

`GRAD_REVISIONS` existed because the default was still v2, which
predates the artifact. **The default is v3 as of 2026-07-31**, so
`--optimize` needs no `--model` at all; the guard stays for the next
revision that ships without a gradient graph, and keeps that case a
sentence rather than a 404 on a filename the operator has never seen.

### 4. Port the loop to C++ — **DONE 2026-07-31**

Split at the same seam the rx engine uses, and for the same reason:

- **`native/core/optimize/`** is in `sstvae_core` and links no
  onnxruntime. It holds everything that can be *wrong* — the Adam step,
  the unit-RMS projection, the mode mask, which iterate is returned,
  why it stopped — behind a `GradFn` seam. So the loop builds and is
  tested with `--no-codec`, no download, no artifact.
- **`native/core/codec/grad_session.cpp`** is the ORT half: one
  `Run()` per call, target bound at construction because the loss lives
  in the graph.

`tests/test_optimize.cpp` drives it with a stub gradient (21 checks).
The C++ `checkpoint` gained the same three gradient-artifact tests as
the Python one, because both had the same traps.

**Parity, measured.** With a deterministic objective (200 dB, so the
noise term vanishes) and an identical input picture, C++ and Python
agree exactly: mse 0.000256 → 0.000118, +3.35 dB, both. Getting that
comparison honest took two attempts, and the failures are worth
recording because they are *not* optimizer bugs:

- **The test picture must be a 640x480 PNG.** At any other size `fit`
  resizes, and stb's resampler is not PIL's LANCZOS — a visibly
  different target, so a different loss. One test image is 580x375
  and   produced a 10% difference in starting MSE that looked like
  a port bug.
- **JPEG decoding differs too**, more subtly: `wonder_wheel.jpg` is
  already 640x480 and still gave targets differing by ~45 in total sum,
  because stb_image's IDCT is not libjpeg's.

Both are pre-existing and accepted ("above the modem, identical
behaviour is not required"), but they confound any numerical comparison
that goes through an image file. With a PNG at the target size the
encoder is bit-identical, as the codec's parity claim says it should
be, and the optimizer's agreement follows.

What is *not* claimed is bit-identical latents in normal use: the
channel noise comes from `std::mt19937_64` here and numpy's PCG64
there, so the two explore different noise realizations and land ~0.1 dB
apart on the objective. That is the intended contract — the loop's
decisions are the port's claim, not its arithmetic.

**The tests earned their place by mutation, not by passing.** Three
mutants survived the first version of `test_optimize.cpp`, and each one
taught something:

- **Returning the last iterate instead of the best.** The test checked
  `mse_best`, which is tracked separately — so the number was right
  while the vector was wrong. It now compares the returned latents
  against the iterate the gradient function was handed at the minimum.
- **...and the fix exposed a second problem.** With a *uniform* stub
  gradient every iterate is identical, because the unit-RMS projection
  scales the step straight back out. The stub had to be made
  non-uniform before "best" and "last" were even distinguishable.
- **Zeroing the untransmitted tail inside the loop** turned out to be
  unreachable: `weights` is zero there, so the gradient is zero, and
  Adam's update for a coordinate with m = v = 0 is exactly zero. The
  line was removed rather than tested, since no test could reach it.
- **Summing the Monte Carlo draws instead of averaging** is invisible,
  because Adam's update is `m/sqrt(v)` and therefore invariant to a
  constant scaling of the gradient. The original test asserted the
  opposite and was simply wrong. What *is* observable is the reported
  loss, which drives the plateau test — so that is what the test checks
  now.

### 5. App integration — **DONE 2026-07-31**

Transmit-side UX, which is where this stops being a numerical problem:
it is tens of seconds of work between "send" and the first carrier, on
a half-duplex station. It needs progress, a cancel that leaves the
plain encoder's latents ready to send, and a default that is off or
clearly bounded. Any interaction with PTT is a mistake — the
optimization must be entirely finished before the radio is keyed.

**Run it speculatively, before Send** (Andrew, 2026-07-31). When the
optimizer is enabled, start it a short debounce after the operator
stops editing, and let it run to plateau or a generous budget. If Send
arrives first, a **second, shorter budget starts from the click**.

This is the design that makes the feature nearly free. The measured
curve is steep early and flat late (65% of the gain by 20 s, plateau
~90 s), and the expensive part sits in time the operator was spending
anyway — composing the picture, checking the frequency. What they
actually wait for is only the tail, bounded by the post-Send budget.
It also removes the reason the shipped CLI default is 20 s: nothing has
to be traded away when the clock starts before the decision to send.

**No change to `optimize::run` is required**, which is worth stating
because it looks like it needs one. The deadline is not a property the
loop has to own: `ProgressFn` is consulted once per step and returning
false stops it, so the caller applies whichever bound is currently
binding — the generous one before Send, `min(generous, click + short)`
after. `Options::time_budget_s` stays as the outer backstop and
`StopReason::Cancelled` is the same mechanism the Cancel button uses.

**`native/core/optimize/speculative.hpp` implements this** (2026-07-31),
Qt-free and onnxruntime-free: the gradient arrives through a
`GradFactory`, so the debounce, the generation counter, the deadline
policy and the staleness rule are all tested with a stub in a
`--no-codec` build. `tests/test_speculative.cpp` drives it with a latch
rather than a clock — the stub blocks, so the test decides when a run
is mid-flight.

**The GUI is wired** (`gui/tx_panel.cpp`), behind
`transmit.optimize` — one switch and no dials, because the objective
SNR is a measured constant and the budgets only matter through a
plateau test that usually ends the run first. Four seams did the work,
three of which already existed:

- `OverlayEditor::documentChanged` is new, and is emitted per mouse
  move during a drag — the debounce is what makes that affordable.
  Deliberately **not** emitted by `select()`: selection handles are
  painted over the widget rather than into `composed_image()`, so
  choosing a different item changes nothing that would be transmitted.
- `picture_changed` takes a `LatentsFn` rather than a vector, so the
  ~30 ms encode happens on the worker *after* the debounce. A drag that
  produces twenty edits pays for none of them.
- **`TxEngine`'s `Encoder` seam is where the result enters**: with
  optimized latents in hand the "encoder" is a lookup, and with none it
  is the plain codec. There is no second transmit path to keep in step,
  and the empty case is exactly what the app always did.
- Send calls `request_send()` and then *polls* on a `QTimer` rather
  than blocking, because nothing on the GUI thread may block. When the
  feature is off, `ready()` is true immediately and the wait costs
  nothing.

**The setting takes effect on OK, not at the next edit** (Andrew,
2026-07-31). `TransmitPanel::sync_from_config` is called from
`MainWindow::open_settings` beside the receive panel's equivalent:
turning it on starts a run for what is already composed, and turning it
off destroys the optimizer, which *is* the discard — there is nowhere
else a refined latent is held, and `send` then falls through to the
plain encoder exactly as it did before the feature existed. A stale
"Picture refined: +2.4 dB" is cleared at the same time, because it
would otherwise describe something that is no longer what would be
sent.

The same call is made from `on_model_loaded`, since refinement needs a
codec to start from: a run that could not be armed at startup, or after
a checkpoint change, is armed when the model arrives rather than
waiting for the operator to touch something.

Switching it off while a send is waiting on it stops the wait and
returns the button rather than transmitting: the settings dialog is not
somewhere anyone expects to trigger a transmission.

**Send commits to the composition that was on screen when it was
pressed** (Andrew, 2026-07-31), so an edit arriving during the
post-Send wait — or during the transmission — is deferred and applies
to the *next* send. That is the simpler rule, and it happens to remove
the hazard in the alternative: if the generation does not move while a
send is committed, the latents in flight still describe the picture
being sent. Re-arming after an edit mid-wait would either have put the
*old* picture on air (the generation counter's whole purpose) or thrown
away the work and left the operator waiting out the idle budget.

The status line shows `objective_gain_db` while it works **and keeps
the figure once the run ends**, whatever ended it — plateau, either
budget, or Send cutting it short. It is labelled `est.`: it is the
objective's own improvement and overstates what the far end sees by
roughly 3x, so it is a progress figure and not decibels earned. A run
that finished without a single measured step — missing artifact, failed
encode — says "Sending unrefined" instead, because there is no estimate
to report and the operator should know the picture is the plain one.
The settings note says the feature "needs nothing of the receiving
station" rather than "changes nothing at the far end" (Andrew,
2026-07-31) — the latter reads as *no effect*, which is backwards: it
changes the picture, it just does not require anything to decode it.

Four things the implementation has to get right, none of them visible
in the arithmetic:

- **A result belongs to one picture.** Editing invalidates a run in
  progress, so the worker needs a generation counter and a stale
  result must be dropped rather than transmitted. This is the failure
  that would send the *previous* composition on air, which is worse
  than not optimizing at all.
- **The response to Send is one step granular**, ~1 s at 4 channel
  draws. The post-Send budget is therefore a floor of about one step,
  not a precise deadline — fine for a 32–95 s transmission, but it
  means "0 s" is not a meaningful setting.
- **Best-so-far is always available**, so an interrupted run is not a
  wasted one: `run` returns the best iterate it reached, and at worst
  that is the encoder's own latents. There is no path where enabling
  this makes a picture worse than leaving it off.
- **Nothing may block the GUI thread**, the same rule rig control
  already follows. The worker holds the ORT session; the GUI sees
  progress and a result.

**The progress callback carries a live quality estimate** (Andrew,
2026-07-31), because a number that visibly climbs is worth having while
the operator waits and it costs one `log10` a step against four decoder
passes. `Progress::objective_gain_db` is `10*log10(mse_start/best_mse)`.

**It is an objective value and must not be presented as decibels
earned.** It overstates recovered picture quality by roughly 3x, by a
ratio that varies with the image, so a GUI showing "+2.4 dB" would be
making a promise the receiver does not keep. Show it as progress — a
bar, a trend, "refining" — not as a figure of merit. The rule the rest
of this document follows applies to the UI too: latent-domain PSNR is
an objective value, never a result.

**Two of the tests only became real after mutation.** Publishing
results regardless of generation was caught immediately; three others
were not, and each pointed at the test rather than the code:

- **A superseded run that keeps going** still publishes nothing, so
  correctness survives — but the worker runs one job at a time, so it
  blocks its own replacement. The test now gives the first picture a
  loss that improves geometrically forever, so it *can only* end by
  noticing it is stale.
- **`take_result` ignoring the generation** was unreachable, because
  `picture_changed` also cleared the result. Two mechanisms for one
  invariant meant neither was tested; the clear was removed so the
  generation check is the only one, and a test with a *completed*
  result now covers it.
- **Ignoring the post-Send deadline** needed a case where the idle
  budget would not have ended the run anyway.

Deliberately *not* in the design: optimizing during transmission of a
previous picture. It would contend for the CPU with the modem and the
audio callback, and `docs/native-app.md` has a standing rule about what
that costs.

## Risks

- ~~**PAPR regression.**~~ Measured at ±0.05 dB. Closed.
- **Off-manifold latents — this one happened.** The clean objective
  produced exactly the predicted failure: excellent simulated PSNR
  (+6.9 dB) and *worse* pictures on a real fading channel (−1.85 dB).
  The regularizer at 1e-3 did not prevent it; the channel term in the
  objective did. Treat any future change that weakens the channel term
  — a faster approximation, fewer Monte Carlo draws, an optimistic
  default SNR — as reopening this risk, because the symptom appears
  only in an end-to-end test that is far more expensive than the
  latent-domain one it would be tempting to substitute.
- ~~**A tuned objective that flatters itself.**~~ It did flatter
  itself, by roughly 2×, and the round trip caught it. Kept as a
  standing rule rather than an open risk: **latent-domain PSNR is an
  objective value and never a result.**
- **Scope.** This is sender-side and optional, and it must stay that
  way. The moment it requires anything of the receiver, it stops being
  free and starts being a format decision.

# TODO

Known work items that aren't bugs with a clear fix, kept here so the
reasoning doesn't have to be rediscovered.

Completed items are summarized below; the full measurements and
reasoning behind each live in `docs/todo-done.md`.

## Completed: pilot crest factor

**Implemented 2026-08-14, `PROTOCOL_VERSION` 3.** The frozen QPSK pilot
had 7.9 dB envelope PAPR against a clip threshold ~1 dB above the mean,
making it the most heavily clipped symbol in the waveform *and* the
channel-estimate reference, the preamble and the blind template.
Replaced with a minimized crest-factor phase set at 0.99 dB
(`PILOT_PHASE_NUM`, exact integer numerators of a rational turn):
**~+2.5 dB of latent SNR**, acquisition improved, with
`CLIP_HEADROOM_DB` 0.5 → 0.0 and `BLIND_SCORE_THRESHOLD` 4.0 → 9.0 as
part of the same change. **Zadoff-Chu is disqualified** — its
delay-Doppler equivalence makes CFO and timing confusable and this
sequence is the acquisition template — and the 0.794 latent gain the
change introduces is deliberate and must not be corrected. **Still
open:** a stage-2 fine-tune through the new modem (running), and the
image-PSNR confirmation, which needs far more trials than a pilot
comparison first suggests.

## Rejected: tone reservation for PAPR

**Measured 2026-08-14, both sides of the clipper: −0.180 ± 0.024 dB
PSNR** end to end at mode B, full PEP credit given. Structural, not an
implementation limit — the clipper is a compressor at this operating
point (28.7% of samples above threshold), so peak reduction is the wrong
tool, and post-clip reservation is pilot-limited to +0.13 dB. Do not
fine-tune around the dead latents to rescue it: the ceiling with a
decoder that pays nothing for them is +0.10 dB PSNR. Two numbers worth
keeping: **~0.48 dB PSNR per dB of channel SNR** at the 8 dB operating
point, and **clipping self-noise at 10.15 dB** — first-order, and what
made the pilot work above worth doing.

## Rejected: explicit Wiener shrinkage on the received latents

The per-latent confidence weights already are that shrinkage. On
`latents × weights` — the quantity the decoder actually consumes — the
oracle headroom is 0.04 dB at mpp 8. The phantom "+1.5 dB" came from
measuring bare `latents`.

## Completed: preamble detection against a steady-carrier interferer

**Mostly closed 2026-08-13**, as a side effect of the wide-acquisition
false-lock fix: `config.TEMPLATE_SCORE_THRESHOLD` also rejects a steady
tone at every level the original table named — a pure tone reads
metric 1.000 but scores only ~0.2–0.3 against the 24-carrier preamble
template. **Still open:** fix candidate 1 (keep the top-K metric peaks
and let the Golay header arbitrate) is the more direct answer to the
interferer problem and is not implemented; it cannot cost sensitivity,
because its first candidate is exactly today's argmax.

## Completed: wider acquisition search, for a mis-tuned counterpart

**Implemented 2026-08-11, in both implementations**:
`config.ACQUIRE_MAX_BINS` = 12 (±625 Hz) unconditionally,
`config.BLIND_BIN_STEP_HZ` = 12.5 with `sync.refine_cfo`, and the
±625 Hz *blind* range as an opt-in setting (`RxConfig.blind_wide`). A
false lock the widening opened up — real frame data elsewhere in a
mis-tuned transmission clearing the noise-calibrated threshold — was
found against a real recording and closed 2026-08-13 with
`config.TEMPLATE_SCORE_THRESHOLD` (0.40). **Still open:** on the blind
path, `bin_step_hz` could go to 25 Hz for another ~1.7× if the ~1 dB of
worst-case scalloping at the very bottom is ever worth it (50 Hz
collapses).

## Completed: frequency drift *during* a transmission

**Tracking shipped as an option 2026-08-11, in both implementations**:
`drift_track` = `off` | `slow` | `fast` on `Modem.demodulate` /
`demodulate_blind`, off by default (`off` constructs no loop, so the
default path is bit-identical to before); two gain settings because the
loop bandwidth must sit above the drift's spectrum and below the
channel's Doppler spread, and no compiled-in pair serves both. **Still
open:** the blind path's pull-in aliases past ~7 Hz of drift across the
window — the fix is anchoring the loop mid-window and running outward,
pinned as a test but not implemented; a non-causal smoother over the
whole frame sequence has the oracle's 3.8 dB available in every cell
where every causal loop fails; and nobody has measured what real
drift rates look like, which is the first thing to do before any of it.

## Completed: blind acquisition — does longer integration reach weaker signals?

**No; closed 2026-08-06.** The detector's score converges to a value
set by the signal's *per-period* SNR, so no amount of integration
rescues a signal below the floor — what longer windows buy is
reliability near the floor. Multi-timescale accumulation (one decay
timescale per mode, sharing the expensive per-block matched-filter
work) is implemented in both implementations.

## Completed: improve acquisition at large frequency offsets — did not reproduce

**Withdrawn 2026-07-26.** There is no offset effect; the original
result was 6-seed sampling noise, and the 25-seed re-measurement is
flat across ±55 Hz in both acquisition and decode quality. **Do not
re-open without ≥25 seeds per point** — acquisition near threshold
succeeds 40–80% of the time, so single-digit trials per cell will
invent a pattern.

## Completed: range-reduce the phasor arguments

**Done 2026-07-28.** Every phasor argument is reduced before it reaches
`exp()` — exactly, via integer arithmetic, wherever the frequency is an
integer number of Hz — which bought cross-platform determinism (it had
already broken CI) and a 1e-14 parity tolerance. The rule for new DSP:
reduce the argument exactly before any transcendental.

## Completed: quantisation tolerance as a training soft constraint

**Nothing to act on.** The int8 deficit was solved at export by keeping
each part's single most quantisation-sensitive conv layer at fp32;
static calibrated quantisation was tried and is *worse* — do not retry
it without new information. The finding that outlived it — the model
had never seen non-photographic content — became the item below.

## Non-photographic content: evaluate, then train on it

**Andrew, 2026-07-28.** Two related pieces of work, promoted out of the
quantisation section above because they outlived it.

### Why this surfaced

The quantisation work measured the model on smooth synthetic probes for
the first time and found the fully-quantised decoder costing **1.54 dB**
there against 0.10 dB on COCO. Fixing that at export removed the *extra*
penalty, but the underlying fact remains: **the model has never seen
anything that is not a photograph**, and operators routinely send test
cards, charts, callsign graphics, screenshots and text. Nobody has
measured what those cost at fp32.

### 1. Evaluate on non-photographic images

Should join the existing evaluation sweeps (PSNR/LPIPS vs SNR per mode),
as a second content class rather than a replacement. The interesting
number is the *gap* between the two classes at each mode and SNR — that
is what tells you whether this is a training-data problem worth
spending on, and how much.

Expect specific failure modes rather than uniform blur: hard edges,
large flat areas, saturated primaries and small text are all things a
photograph-trained convolutional autoencoder handles differently from
foliage and faces. The README already warns that small text "can come
back subtly wrong rather than merely blurry"; this would quantify it.

**Measured 2026-07-30, codec-only at fp32** (`scripts/gen_nonphoto.py`
→ `scripts/eval_nonphoto.py`; encode → per-mode truncation → decode, no
modem, so the number isolates the model; the generated set is
deterministic — salt "eval", 8 images per class — so these numbers are
re-derivable exactly). PSNR against 25 COCO val images, gap vs COCO in
parentheses:

| class    | mode A         | mode B         | mode C         |
|----------|----------------|----------------|----------------|
| coco     | 26.83          | 27.72          | 27.93          |
| chart    | 26.41 (−0.42)  | 27.65 (−0.08)  | 27.61 (−0.32)  |
| text     | 23.65 (−3.19)  | 25.17 (−2.56)  | 25.73 (−2.21)  |
| lineart  | 23.33 (−3.51)  | 24.05 (−3.68)  | 24.22 (−3.72)  |
| gradient | 23.68 (−3.16)  | 22.12 (−5.60)  | 12.56 (−15.37) |
| callsign | 21.61 (−5.23)  | 23.41 (−4.31)  | 23.61 (−4.33)  |
| testcard | 19.73 (−7.10)  | 20.36 (−7.36)  | 20.62 (−7.32)  |

Three findings, in decreasing order of importance:

- **The operator content classes pay 2–7 dB at fp32**, an order of
  magnitude more than the 0.1 dB the quantisation work was tuned to
  remove. Training-data work is clearly worth spending on. (Charts are
  the exception — mostly white with thin marks, low-energy residual,
  fine as-is.) Qualitatively the reconstructions are still *usable* —
  text stays legible — but flats pick up a grey tint and halos, thin
  lines drop out, and saturated bars bleed at the edges.
- **Smooth gradients are a pathology, not a gap: mode C is *worse*
  than mode A on every gradient image** (systematic across all 8, not
  an outlier — down to 11 dB). The third latent group, fed content with
  no fine detail, emits latents the decoder renders as a full-field
  high-frequency checkerboard. Modes A/B truncate that group away and
  stay smooth. Worth knowing operationally (a longer mode can deliver a
  *worse* picture on smooth content) and a specific thing v2 training
  should fix — it is also a hint that group 2's latent distribution on
  off-distribution input leaves the region the decoder was trained on.
- Full-pipeline SNR sweeps on this set (the original plan above) are
  still worth doing once training-data work starts, but the clean-codec
  gap already answers the go/no-go question and runs in seconds.

`gen_nonphoto.py` is deterministic per (class, index) and is deliberately
the same generator that item 2 below would scale up for training data.

### 2. Find or make training data for v2

Deferred once already, for limited availability and licensing — the
dataset is republished as `arodland/coco640-sstvae`, so anything added
has to be redistributable. Worth looking harder, and there is one angle
that dissolves both objections:

**Generate it.** Test cards, colour bars, grids, gradients, line art,
callsign ID cards, rendered text blocks and plotted charts are all
procedurally generatable, carry no licence at all, and are *exactly* the
content class in question. The project already has the pieces —
`sstvae/images.py` has the font search, `sstvae/overlay/render.py` draws
text and insets with PIL, and matplotlib is already a dependency for
plots. A generator gives unlimited quantity, exact control of the
mixture, and no attribution burden.

Real-world sources worth checking for the part a generator cannot cover
(screenshots, scanned documents, real test-card photographs):

- Wikimedia Commons public-domain test patterns (PM5544, SMPTE bars,
  EBU) — many are PD or simple enough to be uncopyrightable.
- CC0/CC-BY document and figure sets (DocLayNet is CC-BY).
- Openclipart (CC0) for line art.

Check licences individually; "found on the internet" is not a licence,
and this dataset gets redistributed.

**Mixture ratio matters and should be swept, not guessed.** Too much
synthetic content and photographs regress; the point is to widen
coverage, not to move the distribution. Measure both classes at every
candidate ratio, and keep the photograph numbers as a floor.

**The machinery for this landed 2026-07-30.** `sstvae/nonphoto.py` is
the generator (deterministic per (class, index, salt); the salt keeps
"train"/"val"/"eval" disjoint by construction, so no training image can
leak into either measurement set). `data.NonPhotoDataset` generates on
the fly — no dataset to build, host or license — and
`train.py --nonphoto-frac X` mixes it into any base dataset.
`val_psnr_np_clean` / `val_psnr_np_8dB_e20` are logged every epoch on a
fixed salt="val" set *whether or not the mix is on*, so a photo-only
baseline records the same metric the ratio sweep compares against.
**The ratio sweep ran 2026-07-30** (HF Jobs, a10g-large, from-scratch
stage-1, width 128, epoch-size 8192, batch 16, 100 epochs; repos
`arodland/sstvae-s1-640-np{0,01,02,03}`). Means over epochs 90–99,
where every run is a ±0.02 dB plateau:

| frac | photo clean | non-photo clean |
|------|-------------|-----------------|
| 0    | 25.55       | 23.91           |
| 0.1  | 25.42       | **28.85**       |
| 0.2  | 25.19       | 27.61           |

**0.1 buys +4.9 dB non-photo for −0.13 dB photo, and strictly
dominates 0.2** (better on both axes) — more synthetic content past
~10% starts moving the distribution rather than widening it, exactly
the failure mode predicted above. The 0.3 run only reached epoch 16
(quota-delayed start, 1 h limit) and trailed both metrics there;
given 0.2 already loses to 0.1, it was not rerun. Caveats: one seed
per point (runs are independent shuffles, so tenth-of-a-dB differences
are noise — the 1.2 dB and 4.9 dB gaps are not), and short from-scratch
runs, so the *ratio* is the finding, not the absolute PSNRs. **Use
`--nonphoto-frac 0.1` for v2 training.**

**Confirmed at full scale the same day**: `arodland/sstvae-s2-640-np01`
resumed the published stage-2 checkpoint (epoch 217) for 100 epochs
with `--stage2 --nonphoto-frac 0.1 --clip-headroom-db 0.5
--papr-weight 0.001 --lr 1e-4`. The photo baseline recovered in ~40
epochs and finished *above* the base run (val_psnr_clean 26.19 vs
26.06), non-photo val rose from 23-ish to 29.20, and PAPR held at the
base's 4.27 dB. Codec-only eval of the final checkpoint on the held-out
"eval"-salt set, mode C, vs the published model: testcard 20.62→30.08,
callsign 23.61→31.22, gradient 12.56→**35.24** (the mode-C checkerboard
pathology on smooth gradients is gone — verified visually, all modes
smooth), chart 27.61→31.39, text 25.73→27.98, lineart 24.22→26.37, and
COCO itself 27.93→**28.32** — the mixture *helped* photographs, which
matches the regularization guess above. Text and line art remain the
only classes below COCO (≤2 dB): dense fine strokes, a capacity story
more than a distribution one. So a converged model absorbs the mixture
as a cheap fine-tune; nothing needs to be trained from scratch for it.

What remains: real-world sources (screenshots, scanned documents) for
what the generator cannot make; and before publishing this checkpoint
as v2 artifacts, redo the int8 layer-exclusion sweep in
`scripts/export_onnx.py` from scratch — its choices were tuned against
the *old* model's off-distribution behaviour, which this training
specifically changed.

**Quantisation re-assessed on this checkpoint 2026-07-30** (export not
published). The sweep found the same *shape* as v1 — one dominant layer
per part — but had to be re-run to find *which*: encoder keeps
`node_conv2d_2` at fp32 (int8 latent RMS 7.00e-02, 14.4 dB under the
channel), decoder keeps `node_conv2d` (untuned cost 0.98 dB → 0.00 on
the tuning probes). All gates pass; fp16 is ~free as before. Per-class
int8 cost on the eval set: ≤0.17 dB everywhere except **text, +0.44 dB
across all modes** — the tuning probe set (3 photos + 2 smooth
synthetics) contains no rendered text, so the search never saw the one
class that still pays. If that 0.44 dB ever matters, the fix is adding
a text image to the tuning probes, not more exclusions. (Gradients
show +1.2–1.7 dB in modes A/B but from a 40 dB base — quantisation
noise on a near-perfect smooth field, invisible and irrelevant under
any real channel.)

## PE loss: implemented, unmeasured

**Implemented 2026-08-17, default off** (`--pe-alpha 0`), on Andrew's
suggestion. `sstvae/pe_loss.py` is the loss from Li et al., "PE loss:
Perception-enhanced distortion-oriented loss for image restoration",
*Computational Visual Media* 12(3):825–839, 2026
(doi:10.26599/CVM.2025.9450475). It replaces the flat MSE term with a
per-pixel reweighting: the target's Laplacian says where the edges are,
the *sign* of the error says whether this pixel landed on the blurred
side of one, and `W = 1 + alpha·|∇²target|` amplifies it there. Cheap
(one Laplacian, no network), plug-and-play, and **training-only** — no
on-air format change, no receiver change, no artifact compatibility
tier, so the whole cost of being wrong is a training run.

**Nothing here is measured.** The paper's numbers are on
super-resolution, denoising and deblurring; this is neither the task
nor the degradation. What follows is what a sweep has to establish and
the traps it should not have to rediscover.

- **It buys perceptual quality and pays in PSNR** — their Table 1 at
  alpha=2.0: SR LPIPS 0.2479→0.2357 for PSNR 29.14→28.81, deblur LPIPS
  0.0754→0.0684 for 31.19→30.77. That currency matters here because
  PSNR is what `scripts/snr_sweep.py`, `scripts/ab_checkpoint_sweep.py`
  and the wiki's Performance tables all report, and what the v4
  headline (+0.43 dB) was stated in. **LPIPS in the evaluation sweeps
  is a prerequisite for judging this, not a nice-to-have** — see item 1
  of the non-photographic section above, which wants the same thing.
  Pick the decision metric before running it; without one, "looks
  sharper" is unfalsifiable. **`ab_checkpoint_sweep.py` has it as of
  2026-08-17** (VGG to match the training objective's net, whole frames
  rather than train.py's random crop, on by default because a table
  missing the metric looks like one where the metric came out even).
  `snr_sweep.py` and `eval_nonphoto.py` are still PSNR-only.
- **The encouraging trend is the heavy end.** Their PSNR cost *shrinks*
  as the degradation deepens: denoising at σ=75 gives up 0.07 dB for a
  10% LPIPS gain, against 0.37 dB at σ=25. Our operating points are the
  heavy end. Whether that trend continues into a channel that destroys
  information rather than adding noise to it is the open question.
- **Blur here is partly correct, which it is not in their tasks.** SR,
  denoising and deblurring all hand the network the full spatial field;
  the information is present and blur is regression-to-the-mean. A
  faded, erasure-hit mode-A reception genuinely does not contain the
  edge, and hedging toward soft is MMSE-right — the same argument that
  made `--chroma-weight` confidence-scaled ("allowed to hedge toward
  gray instead of hallucinating color speckle"). So `--pe-conf-scale`
  is **on by default** and scales alpha by the per-sample channel
  confidence; `--no-pe-conf-scale` gives the paper's loss unmodified.
  Which of the two is better is itself unmeasured, and the failure mode
  to look for is invented edge structure on the `mpp`/`mpd` cells — a
  crisper picture that is less true.
- **alpha does not transfer from the paper and 2.0 is not a default.**
  The paper never states its intensity scale, and `|∇²|` is in whatever
  units the images carry (ours are [0, 1], where a hard edge reaches
  ~2–4). Their backbones were L1; ours is `p=2`, and W multiplies the
  difference *inside* the norm, so the effective weight on a squared
  error is `W²`. Sweep it — start below 2.0, not at it.
- **Non-photographic content is where this could go either way, and it
  has to be in the sweep** (`scripts/eval_nonphoto.py`). Test cards,
  line art and text are nearly all edge, so the blur factor map is
  close to saturated there and the effective loss is a different loss
  than the photographs get. Text and line art are also the two classes
  still below COCO after the `--nonphoto-frac 0.1` work, and "dense
  fine strokes, a capacity story" is exactly the case a sharpening
  pressure might help — or might turn into halos.

**How to screen it cheaply:** a short stage-2 fine-tune off the current
lineage rather than a from-scratch run, then `ab_checkpoint_sweep.py`
against the unmodified checkpoint — paired per-image deltas on the same
channel seeds, so the difference is attributable to the loss alone —
plus `eval_nonphoto.py` and an LPIPS column. The `--nonphoto-frac`
sweep above is the template.

### The `--pe-alpha 0` baseline (2026-08-17)

`arodland/sstvae-v3-cc12-pe0`: 20 epochs resumed from v4 (epoch 536),
PE off, so it is the null run every PE run is measured against. Means
over epochs 554–556:

| metric | value | metric | value |
|---|---|---|---|
| `val_psnr_wave_mp8` | 25.0995 | `val_lpips_wave_mp8` | 0.1436 |
| `val_psnr_wave_awgn8` | 26.2136 | `val_lpips_wave_awgn8` | 0.1202 |
| `val_psnr_modeB` | 24.2377 | `val_lpips_clean` | 0.0882 |
| `val_psnr_clean` | 27.4705 | `val_lpips_text` | 0.0873 |

Three things it establishes, all of which change how the PE runs must
be read:

- **The training budget itself buys nothing on the channel metrics.**
  20 more epochs of the *same* loss moved `val_psnr_wave_mp8` +0.009 dB
  and `val_psnr_modeB` −0.000. So a PE run's delta is the loss and not
  the extra epochs, which is the whole reason to spend a run on this.
  The clean-channel metrics are *not* equally converged (+0.08 dB), so
  only read those against the paired baseline, never against v4.
- **The window is a full cosine cycle, not a plateau.** `T_max` is this
  invocation's `--epochs`, so a resume restarts the schedule at full
  `--lr`: epoch 537 drops 0.04 dB, wanders ±0.07, and re-converges by
  ~554. **Compare the tail (last ~3 epochs), not the window mean**, and
  give the PE runs an identical `--epochs`, `--lr` and resume point or
  the comparison is between two different schedules.
- **The tail's tightness is within-run, and is not a confidence
  interval.** At LR≈0 the last three epochs agree to ±0.0001 LPIPS and
  ±0.008 dB, which will make a 0.001 LPIPS difference look like 10
  sigma. It is one seed. The spread *across* the cycle (±0.04 dB,
  ±0.0013 LPIPS) is the honest proxy for what a re-run would differ by,
  and `ab_checkpoint_sweep.py`'s paired per-image SEM is the real test.

`train_lpips` at the tail is 0.1275, on the training term's 256 px crop
scale — not comparable to any `val_lpips_*` above, by construction.

### `--pe-alpha 0.5`: no effect, at a dose that was real (2026-08-17)

`arodland/sstvae-v3-cc12-pe0p5`, same 20 epochs from the same
checkpoint, `--pe-conf-scale` on. Tail deltas against the baseline
above, with the across-cycle sd for scale:

| metric | delta | cycle sd |
|---|---|---|
| `val_psnr_wave_mp8` | +0.011 | 0.041 |
| `val_psnr_modeB` | +0.017 | 0.038 |
| `val_psnr_clean` | −0.017 | 0.080 |
| `val_lpips_wave_mp8` | +0.0000 | 0.0013 |
| `val_lpips_clean` | +0.0002 | 0.0016 |

Everything is inside the noise, and LPIPS — the metric this loss exists
to move — is identical to four decimals. **No effect detected.**

**The dose was real, which is the part worth not re-deriving.** The
first instinct is that alpha 0.5 was simply too small to test anything;
it is not. `mean(W^2 d^2)/mean(d^2)` was **1.17x** at the training
channel with conf scaling — 17% more loss mass, redistributed onto
edges — and *the model demonstrably responded to it*: had it not
changed at all, the weighted `recon_loss` would have read 0.0798 against
the baseline's 0.0682, and it came back at 0.0687, recovering ~93% of
the imposed penalty. So the term entered the optimization, the network
moved to satisfy it, and neither PSNR nor LPIPS noticed. That is a more
informative null than "the knob was off".

**Two calibration traps this settled.** The amplification ratio, not
`mean W`, is the measure of dose (mean W at alpha 0.5 is 1.03, which
badly understates it, because the error is concentrated exactly where
`M` is). And **alpha must not be calibrated on a synthetic step edge** —
the guidance originally in `pe_loss.py` said alpha 2 would be an ~80x
weight, from a hard edge's |g| ~ 2-4; on real photographs |g| is median
0.031 and alpha 2 is a 1.9x dose. The paper's own alpha 2.0 is
*moderate* here, not aggressive.

### `--pe-alpha 2 --no-pe-conf-scale`: a real gain, backwards (2026-08-17)

`arodland/sstvae-v3-cc12-pe2p0nc`, the paper's loss exactly, a 3.0x
dose. Same 20 epochs from the same checkpoint. Tail deltas against the
baseline, in units of the across-cycle sd:

| metric | delta | sd | |
|---|---|---|---|
| `val_psnr_wave_mp8` | **+0.178** | 0.041 | 4.3 sd |
| `val_psnr_wave_awgn8` | **+0.180** | 0.050 | 3.6 sd |
| `val_psnr_modeB` | **+0.173** | 0.038 | 4.6 sd |
| `val_psnr_clean` | **+0.150** | 0.080 | 1.9 sd |
| `val_psnr_np_clean` | +0.524 | 0.294 | 1.8 sd |
| `val_lpips_wave_mp8` | +0.0003 | 0.0013 | 0.2 sd |
| `val_lpips_clean` | −0.0000 | 0.0016 | 0.0 sd |

**+0.17 dB of PSNR and nothing at all on LPIPS** — which is the opposite
trade from the one the paper reports, and the opposite of what the Mach
band argument predicts. A loss designed to buy perceptual quality with
distortion bought distortion and no perceptual quality.

**Do not bank this as a PE loss result. The likelier reading is that it
rebalanced the objective**, and there is direct evidence for that rather
than only suspicion. Multiplying the MSE term by ~3x is arithmetically
close to dividing `--lpips-weight`, `--chroma-weight` and
`--papr-weight` by 3 — and **both of the other terms moved in exactly
the direction that predicts**: `papr_db` rose +0.0006 against a cycle sd
of 0.0001 (**4.4 sd**, on a metric otherwise stable to the fourth
decimal), and `train_lpips` rose. The PAPR tell is the sharp one, since
the Mach band mechanism has no route to the crest factor at all. The
model meanwhile absorbed 96% of the imposed penalty (recon_loss 0.0682 →
0.0738 where an unchanged model would read 0.2068), so it did engage
with the reweighting; the question is only whether the *shape* of the
reweighting mattered or just its scale.

**The control that separates them needs no code change**, one run,
identical everything else:

    --pe-alpha 0 --lpips-weight 0.165 --chroma-weight 0.66 \
        --papr-weight 0.00066

That is the same 3.03x relative up-weighting of the reconstruction term
with a *flat* weight map. If it also lands ~+0.17 dB with flat LPIPS,
PE's edge weighting contributed nothing and what this actually found is
that **the loss balance is miscalibrated** — which would be a more
valuable finding than the paper's, and reachable by one scalar instead
of a weight map. If it lands materially short, the edge structure is
carrying the gain and PE loss is worth keeping.

Either way `--papr-weight` must be restored before anything from this
line goes on air: the PAPR term is what holds the crest factor, and this
run moved it measurably.

## `SSTVAE_BRANDING` switch, so a redistributor can build lawfully

**Goal.** One build option that substitutes a freely-licensed placeholder
icon for the licensed artwork, so anyone repackaging this project can
produce a complete, lawful application without editing files.

**Why it is not optional in the long run.** The app icon is licensed to
Andrew for use in this application and that license does not transfer
(NOTICE, and the SPDX sidecars beside each file). Today the *only* way to
comply is to overwrite eight files in place — which works, because
nothing reads them by any other path, but it is a step a packager has to
discover from prose and then redo on every update. Documenting a
restriction is not the same as providing a way to satisfy it.

**Why it is cheap.** The indirection already exists. Every consumer names
the icon by a path that CMake and one shell script control:

- `native/packaging/icons.qrc` → compiled-in window icon
- `native/packaging/sstvae.rc.in` → the Windows executable resource
- `MACOSX_BUNDLE_ICON_FILE` plus the `.icns` as a bundle source
- `tools/package_app.sh`'s freedesktop `hicolor` loop

So the change is a variable holding a directory, plus a placeholder set
generated the same way (`tools/gen_icons.py` already takes one SVG and
emits every format). **Keep the file names identical** between the two
sets — `sstvae.svg`, `sstvae.ico`, `sstvae.icns`, `icons/sstvae-<n>.png`
— and no consumer above needs to change at all; only the directory does.

**Open questions worth deciding rather than defaulting.**

- **Which way it defaults.** `official` is friendlier for the person who
  builds this most (Andrew, on his own machine, wanting his own icon);
  `generic` is the safer default for anyone else, and is what Chromium
  does. Chromium's reasoning applies here too: the branded build is the
  special case, and a default that silently produces a non-redistributable
  binary is a trap. Leaning `generic` **once the placeholder exists** —
  before then it would just mean no icon.
- **Whether the licensed art stays in the repository at all.** It is here
  today. If the artist's license turns out to permit use in the
  application but not distribution as a source file, the switch stops
  being a convenience and becomes the mechanism: `generic` is committed,
  the real artwork lives outside the tree, and the release build is
  handed a path to it. That is a question about the private license text,
  not about this code.
- **A placeholder that is actually good.** A deliberately ugly one gets
  shipped by accident and looks like a bug; an imitation of the licensed
  artwork defeats the purpose. Something simple and clearly different.

**Not urgent.** Nobody is repackaging this yet, and NOTICE plus the
sidecars mean the position is stated correctly in the meantime. It should
land before the first release that invites forks — a switch added after
people have already built from source is one they have to be told about
twice.

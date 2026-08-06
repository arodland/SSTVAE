# TODO

Known work items that aren't bugs with a clear fix, kept here so the
reasoning doesn't have to be rediscovered.

## Wider acquisition search, for a mis-tuned counterpart

**Goal.** Recover a transmission from a station whose dial is off by
more than the current tolerance — hundreds of Hz, not tens. Not a
user-facing "pick your centre frequency" knob; an opt-in wider search
(or an automatic second pass) so a mis-tuned partner still decodes.

**This is achievable, and it is purely an acquisition-side change.**
The demod path has no dependence on absolute centre frequency at all.
Measured: signal placed at each centre, receiver's heterodyne retuned to
match, recovered latent SNR (mode B, no noise, so deterministic):

| centre (Hz) | 900 | 950 | 1000 | 1012.5 | 1250 | 1500 | 1750 | 2000 | 2100 |
|---|---|---|---|---|---|---|---|---|---|
| latent SNR (dB) | 8.73 | 8.73 | 8.73 | **8.63** | 8.73 | 8.73 | 8.73 | 8.73 | 8.73 |

Identical to the last digit across the whole usable passband. Once the
signal is found and its centre known, where it sits is irrelevant.

**What would have to change** (all small, all in acquisition):

- `sync.acquire()`'s `max_bins=2` caps the integer-bin search at
  ±100 Hz. This is the hard limit today.
- `dsp.sync_lowpass()` is `firwin(129, 850, fs=FS)` around the *nominal*
  centre. At a 1000 Hz centre the carriers land at −1075..−475 Hz in
  that filter's frame, so half of them are cut — this is why a
  mis-tuned signal currently shows `NO LOCK` well before `max_bins`
  becomes the binding constraint. Widen or recentre it. Sync-only, so
  the ISI argument that keeps `to_baseband()` unfiltered doesn't apply.
- `dsp.to_baseband()` takes `FCENTER` from config. Note
  `ofdm.BASEBAND_FREQS` and the DFT matrix need *no* change: the wanted
  carriers land at `g - FCENTER` regardless of where the signal is, so
  only the mixing frequency moves.
- `sync.acquire_blind()` already searches CFO directly over
  `max_offset_hz` (default 55.0) at `bin_step_hz` resolution and has no
  ±100 Hz structural limit — widening its range may be the cheaper path
  to a "wide search" mode than reworking `acquire()`.

**Cost.** Acquisition CPU grows with the search range, which is why this
should be opt-in rather than always-on — it's already the dominant cost
of the blind path (see `--blind-search-seconds`).

**Non-issue: the 25 Hz grid.** A mis-tuned radio won't land on any
convenient grid, but it doesn't need to. Working through the mixing
algebra, after the receiver corrects to put carriers on their bins the
heterodyne image sits at −(g + 2ε + FCENTER), where ε is the *true*
mistune — it does not depend on the receiver's choice of mixing
frequency, so no amount of cleverness at the receiver moves it. The
penalty peaks where `2ε mod 50 = 25` and is bounded at about **0.1 dB**
(measured: 8.63 vs 8.73 at a deliberately off-grid centre; same effect
and same size as the ripple documented under the item below). Accept it;
don't engineer around it, and specifically don't try to snap the
correction to a 25 Hz grid — that would leave up to 12.5 Hz of genuine
residual CFO, which is far more damaging than 0.1 dB.

**Practical bound.** The 1150 Hz carrier span has to fit inside a
typical 300–2700 Hz SSB filter, which limits the centre to roughly
900–2100 Hz — exactly the range measured above. Beyond that the
transmitter's own filtering, not the modem, is the limit.

## ~~Blind acquisition: does longer integration reach weaker signals?~~ — no, and multi-timescale is implemented

**Closed 2026-08-06.** `sync.BlindAccumulator` (the incremental,
block-decomposed replacement for `acquire_blind`'s one-shot search — see
`docs/native-app.md`'s history for why it exists) was built on the
assumption that a longer integration window lets it detect weaker
signals, the same way more samples lower a radiometer's noise floor.
**Measured, that assumption is wrong for this detector.**

`result()`'s score is peak-bin-sum / median-of-other-bins-sum, both sums
of matched-filter *power* across periods. Both grow roughly
proportionally with the number of periods folded in, so the ratio
converges to a value set by the signal's *per-period* SNR — a ceiling
integration doesn't push through, unlike coherent (voltage) integration
or a radiometer's magnitude averaging. Measured (mode C frame data,
`mpp` fading, one-shot `acquire_blind` with a growing search window from
~10% to 100% of the real 95 s transmission):

| SNR | window length dependence |
|---|---|
| −6 dB | passes at every length from ~10 s to 95 s, 8/8 seeds, score flat (4.2–5.4) |
| −7 dB | mostly flat too; **one seed of eight missed at a 10 s window and caught at 95 s** |
| −10 dB | fails at every length up to and including the full 95 s, no exceptions, no seed rescued |

So a signal below the floor cannot be rescued by any amount of
integration. What longer integration *does* buy is **reliability near
the floor**: a signal whose true ratio sits just above threshold can
read below it on a short, noisy sample and get missed, and integrating
over more of it converges the estimate and catches it — the −7 dB
result above. That benefit runs out once the window covers the
transmission's own duration; audio *older* than the transmission is
pure dilution (already documented on `BlindAccumulator` itself), not
more signal to integrate.

**That reframes, but doesn't remove, the original motivation**
(Andrew: "we want ~90 s for mode C without giving up on mode A's own
36 s"). The right lever isn't a longer window in general, it's matching
each mode's own duration — long enough to get mode C's full reliability
benefit, short enough that mode A isn't sitting diluted behind 60+
irrelevant seconds a mode-C-tuned window would still weight
non-negligibly. Implemented: `BlindAccumulator.window_s` now accepts
several decay timescales run in parallel off the *same* shared, expensive
per-block matched-filter result (only the cheap decay-and-fold step
repeats per timescale, so N timescales cost barely more than one);
`result()` reports whichever timescale's score is highest.
`rx/engine.py`/`engine.cpp` pass one timescale per `config.MODES`, each
capped at that mode's own `duration_s` by (the now-repurposed)
`blind_search_seconds` — default above every mode's duration, so nothing
is capped unless deliberately lowered. Ported to C++ and cross-checked
against the Python reference via `pytest --native`
(`tests/test_blind_acquisition.py::test_blind_accumulator_multi_timescale_picks_the_better_one`).

**Caveats on the measurement.** Single-digit-to-low-double-digit seed
counts per cell, same order as the acquisition-near-threshold sweeps
this file warns about elsewhere ("40–80% success rate... any sweep with
single-digit trials per cell will invent a pattern") — the −6 dB / −10 dB
rows are each other's sanity check (comfortably-above and clearly-below
the floor, both flat, in opposite directions), which is stronger
evidence than either alone, but the exact floor location and the exact
size of the near-floor benefit were not swept at the 25-seed rigor that
section demands. Good enough to justify the multi-timescale design;
not a citable number for where the floor sits.

## ~~Improve acquisition at large frequency offsets~~ — did not reproduce

**Withdrawn 2026-07-26. There is no offset effect; the original result
was small-sample noise.** Kept rather than deleted because the mechanism
analysis below is still worth having, and because someone will
otherwise re-derive the same false positive from the same impression.

Re-measured with 25 seeds per point instead of 6 (mode C, AWGN, SNR in
2500 Hz — see CLAUDE.md on the convention change):

| SNR | 0 Hz | 25 Hz | 45 Hz | 50 Hz | 55 Hz |
|---|---|---|---|---|---|
| 0 dB | 23/25 | 19/25 | 18/25 | 22/25 | 19/25 |
| −1 dB | 13/25 | 8/25 | 10/25 | 11/25 | 14/25 |

Pooled at −1 dB: offsets ≤25 Hz acquire **21/50 (42%)**, offsets ≥45 Hz
**35/75 (47%)**. The large offsets do marginally *better*, and at 0 dB
the two groups are 84% and 79% — a gap well inside the binomial noise of
these sample sizes. Under `mpp` fading at 6 dB the row is flat as well
(19–21/25 across 0–55 Hz), which is what the original notes said too.

The original claim rested on a single SNR row with 6 seeds per point,
pooled as "2/12 versus 10/12" — and that pooling quietly dropped the
55 Hz column, which at 3/6 already contradicted the trend. Six trials
cannot separate 40% from 80%. The non-monotonicity the notes flagged as
puzzling (45 and 50 Hz worst, 55 Hz recovering) was the tell: it was
never a mechanism, it was variance.

**Do not re-open this without ≥25 seeds per point.** Acquisition near
threshold is a coin flip with a per-point success rate around 40–80%,
so any sweep with single-digit trials per cell will manufacture a
pattern.

The rest of this section is the mechanism analysis done while the effect
was believed real. It is retained only because it records what has been
*ruled out* about `acquire()`, which stays useful if a genuine
acquisition problem turns up later.

`sync.acquire()` estimates CFO in two stages
(`sstvae/modem/sync.py`):

1. A lag-`M` autocorrelation over the periodic preamble gives a
   fractional CFO, unambiguous only over ±FS/(2M) = **±25 Hz**.
2. The remainder is an integer multiple of the 50 Hz carrier spacing,
   resolved by scoring candidate bins (`f_cand = f_frac + m_bin*FS/M`)
   against the known preamble template and taking the best.

That structure predicts the worst performance at the half-bin boundary,
i.e. 25 Hz, where noise most easily pushes the fractional estimate into
the neighbouring bin's basin. The 25 seed-per-point sweep shows no such
dip either (19/25 and 8/25 at 25 Hz, indistinguishable from its
neighbours) — so the two-stage CFO estimator is not leaking a wrong-bin
penalty anywhere across ±55 Hz. That is a genuine, if negative, result:
the estimator behaves.

**Also ruled out: the sync lowpass.** `sync_lowpass()` is
`firwin(129, 850, fs=FS)` on the complex baseband, and the carriers sit
at −550..+600 Hz around `FCENTER` before any offset, so it looked like a
large offset might push them into the skirt. Measured attenuation at the
shifted carrier positions is **flat to within 0.02 dB out to 55 Hz** at
both band edges — the filter is not costing anything here. Don't
re-check this.

If a real acquisition problem does turn up later, these were the
untested candidates — they are ideas, not outstanding work, since there
is currently nothing to explain:

- Instrument `acquire()` on failing cases: log `f_frac`, the chosen
  `m_bin`, the per-candidate scores, and the final `f_hat` versus truth.
  This distinguishes "picked the wrong bin" from "detected nothing".
- Score bin candidates over more than the preamble — fold in the first
  few frames' pilots before committing. The information is already
  demodulated; only the decision is premature.
- Keep the top-2 candidates and let the Golay header arbitrate, turning
  a hard argmax into a cheap 2-way retry.

What the 25-seed sweep *does* establish is the plain AWGN acquisition
curve for mode C, independent of offset: ~80% at 0 dB, ~45% at −1 dB.
That is the threshold region, and it is steep.

**Scope check: this is an acquisition problem, essentially entirely.**
Once a signal is acquired and its centre frequency known, decode quality
is nearly offset-independent. Measured modem-only (random latents,
mode B, *no noise at all*, so the numbers are deterministic), recovered
latent SNR vs offset:

| offset | 0 | 6.25 | 12.5 | 18.75 | 25 | 31.25 | 37.5 | 43.75 | 50 |
|---|---|---|---|---|---|---|---|---|---|
| latent SNR (dB) | 8.73 | 8.69 | 8.68 | 8.70 | 8.73 | 8.67 | 8.66 | 8.70 | 8.73 |

Total spread **0.07 dB** — i.e. nothing. The small structure that is
there is real and
explainable: `to_baseband()` is deliberately unfiltered and relies on
the heterodyne image landing exactly on the 50 Hz bin grid, where the
160-sample demod correlation nulls it. After acquisition corrects by δ,
the image sits **2δ** off that grid, so leakage peaks where `2δ mod 50`
is 25 Hz (δ = 12.5, 37.5 → 8.66–8.68 dB) and vanishes where it is 0
(δ = 0, 25, 50 → 8.73 dB). The ordering follows `2δ mod 50` perfectly,
and the CFO estimate error is constant (−0.021 Hz) across the sweep, so
it isn't residual CFO doing this.

Conclusion: there is no offset sensitivity in the demod/EQ path, and
(per the 25-seed sweep above) none in acquisition either. Frequency
offset within ±55 Hz simply does not cost anything.

**Safety net.** Whatever does cause a missed preamble, it is less
catastrophic than it looks: the beacon gives a fresh sync opportunity
roughly every 10 s for the whole transmission (`sync.acquire_blind` /
`Modem.demodulate_blind`), and `sstvae_listen.py` falls back to it
automatically. A missed preamble costs latency and the header's mode
information, not the image. The preamble path still sets the floor for
`sstvae_decode.py` on a recording.

**Reproducing.** Sync success vs offset, everything else fixed:

```sh
uv run sstvae_encode.py photo.jpg tx.wav --mode C --model checkpoint.pt
for off in 0 25 45 50 55; do
  for seed in $(seq 0 24); do
    uv run sstvae_simulate.py tx.wav rx.wav --snr -1 --freq-offset $off --seed $seed
    uv run sstvae_decode.py rx.wav out.png --model checkpoint.pt
  done
done
```

Count `SyncError: header decode failed` against successes. Three traps:
use AWGN rather than fading (with fading, whether the preamble lands in
a deep fade dominates everything else); stay within about a dB of
threshold, since at 6 dB everything succeeds and there is nothing to
measure; and **use at least 25 seeds per point** — six is what produced
the phantom effect this section used to describe.

## ~~Range-reduce the phasor arguments~~ — done 2026-07-28

**Closed.** Fixed in `sstvae/modem/ofdm.py`, `sstvae/modem/dsp.py`,
`sstvae/hfchannel.py` and `sstvae/waveform_channel.py` before the C++
port went further, so the golden corpus and its tolerances only had to
move once.

Every phasor argument is now reduced before it reaches `exp()`. Where
the frequency is an integer number of Hz — which is all of the OFDM
carriers — the reduction is exact integer arithmetic (`(n*f) % FS`), so
the argument is under one turn and carries no rounding of its own. For
arbitrary frequencies (`freq_correct`, `hfchannel.freq_shift`) the phase
is wrapped to `[0, 1)` cycles, which cannot make the *product* exact but
does remove the large-argument error entirely.

`to_baseband` got the biggest win and the simplest treatment:
`FCENTER/FS = 1500/8000 = 3/16`, so there are only **16 distinct
phasors** and a table lookup replaces 760,000 calls to `exp()`. Derived
from `gcd(FCENTER, FS)` rather than hardcoded, so a config change stays
correct.

Measured against a 70-digit series expansion:

| | before | after |
|---|---|---|
| `MOD_MATRIX` | 2.83e-14 | **8.97e-16** |
| `to_baseband` over a mode C transmission | 7.03e-11 | **1.23e-16** |

The payoff was never the accuracy — 1e-10 rad is 6e-9 degrees, and
nothing on air could notice. It was:

- **Cross-platform determinism.** `sin`/`cos` of a large argument
  disagree between glibc, musl and MSVC, and between x86-64 and Apple
  silicon, because implementations differ in how far they carry argument
  reduction. This had already broken CI: `MOD_MATRIX` and `DEMOD_MATRIX`
  failed byte-comparison on macOS and Windows while passing on Linux.
- **A stronger parity claim.** `PHASOR_TOL` in
  `native/tests/test_golden.cpp` went from 2e-13 to **1e-14**, and the
  sum tolerance from 1e-12 to 1e-13, because they are now sized by one
  ulp of `exp()` rather than by the reference's own error. Measured C++
  against Python: 9.6e-16 on the matrices, exactly 0 on the pilot
  sequence.

Two notes for anyone touching this again. `waveform_channel.py` was
included deliberately even though it is training-only: it is a replica
of the modem, and letting the replica and the original compute the same
matrix differently is exactly the drift the file exists to prevent. The
change is ~1e-14 on a buffer immediately cast to complex64 (eps 1e-7),
so it cannot affect training. And `scripts/precoder_probe.py` still has
the unreduced form; it is a scratch probe, not part of the product.

**The general rule, worth applying to new DSP:** reduce the argument
before a transcendental whenever you can do it exactly. It is free, it
is more accurate, and it is the difference between a result that is a
property of the signal and one that is a property of the machine.

## Quantisation tolerance as a training soft constraint

**Nothing to act on, and the deficit that prompted this is gone.**
Recorded so the reasoning is on the table next time the training recipe
is revised.

The int8 problem was solved at export, not in training: keeping each
part's single most quantisation-sensitive conv layer at fp32 took the
encoder from 5.8 dB to **14.0 dB** under the channel noise, and the
whole pipeline from −0.19 dB to −0.002 dB on photographs and −1.57 dB to
−0.112 dB off-distribution. `docs/onnx.md` has the mechanism
(`ConvInteger` carries one scale per weight *tensor*, so one
outlier-heavy tensor dominates) and the options if more margin is ever
wanted. Static calibrated quantisation was tried and is **worse** — do
not retry it without new information.

**The finding that outlived the quantisation work is about training
data.** The model is trained on COCO photographs; operators send test
cards, charts, callsign graphics, screenshots and text. Quantisation
tuning only removed the *extra* penalty those pictures were paying — a
large one, 1.5 dB, which is why it surfaced here at all. They are still
reconstructed by a model that has never seen anything like them, at
every precision including fp32. Non-photographic content in the training
mix would improve them across the board, and would also make the
model less sensitive to the next perturbation nobody thought to test.

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

# Diversity reception (2026-08-05)

**Hypothesis.** Two receivers on independent antennas, tuned to the
same frequency and no more than a frame apart in time, each demodulate
the *same* over-the-air transmission with independent noise and fading.
Combining the two should recover more of the picture than either
antenna alone, using nothing beyond what the modem already computes:
`Modem.demodulate` already reports a per-latent confidence weight
derived from the pilots (`DemodResult.weights`) alongside a whole-burst
SNR estimate (`DemodResult.snr_db`).

**Result.** Positive, and implemented end to end. `sstvae/modem/
diversity.py` combines two independently-demodulated branches by
maximal-ratio combining (MRC) in the latent domain; measured gain (mode
A, `scripts/diversity_sweep.py`, 20 trials/point) is **+2.9 dB at 6 dB
single-branch AWGN, rising to +5.8 dB at -2 dB**, and **+5.1 to +5.9 dB
under independent `mpp` fading at 3-10 dB** -- see "Measured gain"
below, including a 12-trial/point run confirming the same numbers hold
on modes B and C. No on-air format change: this is a receive-side
combine over ordinary `DemodResult`s, so a diversity-capable station
still transmits and is heard exactly as before.

**Wired in, on both sides of the port.** `sstvae/rx/engine.py`'s
`decode_loop_diversity` and `native/core/rx/engine.cpp`'s C++
counterpart run the two-branch state machine live (`sstvae_listen.py
--device2`, and the native app's Receive settings tab: "Diversity
reception (second antenna)" plus a second input-device picker). Both
also support an optional debug visualization,
`contribution_image`/`modem::diversity::contribution_image` --
`--diversity-debug-image` on the CLI, a checkbox in the native
settings dialog -- a red/blue heatmap (time x latent channel) of which
branch supplied each transmitted latent, written as
`<name>_diversity.png` beside the picture. See "What's not done" for
what this integration deliberately leaves out (blind-fallback
diversity, raw-domain combining).

## Why combine after `demodulate`, not inside it

The obvious design is to fuse the two branches' raw samples or raw OFDM
symbols before equalization, the way a hardware diversity receiver
would. That needs the two branches on a shared sample timebase, which
this scenario doesn't offer -- "within a frame time" leaves plenty of
room for the branches' preambles to land at different sample offsets,
independent CFOs, and independent sample-clock drift.

`Modem.demodulate` already solves exactly that problem, once per
branch: it re-acquires its own preamble, tracks its own clock drift,
and deinterleaves its output back into **canonical latent order**
(`DemodResult.latents`/`.weights`, both sized `mode.n_latents`). Two
independent `demodulate()` calls on two branches of one transmission
therefore produce arrays that are already index-for-index comparable --
latent `k` in branch 1 and latent `k` in branch 2 are estimates of the
*same* picture coefficient, with no shared timebase required to see
that. So `sstvae/modem/diversity.py` is a pure post-processing step over
`DemodResult`, not a change to `modem.py`, `sync.py`, or `framing.py`.
The cost is that each branch must *independently* acquire the preamble
and decode the header -- see "What's not done".

## The combining weight

`DemodResult.weights[k]` (call it `w`) is *relative* confidence: fading
depth against that branch's own median pilot-derived channel gain,
capped at 1 (`modem.py`'s `w = min(|h|/med_h, 1.0)`). It says nothing
about one branch's noise floor relative to another's -- an antenna with
20 dB more noise reports the same 0..1 scale as a quiet one. `snr_db`
(the existing pilot-based whole-burst estimate, `_estimate_snr_db`) is
what puts branches on a common footing.

Model each branch's equalized latent as `latent_i[k] = truth[k] +
noise_i[k]`. Since `y = raw * conj(h) / |h|^2` (zero-forcing
equalization), and `w_i[k] = |h_i[k]| / med_h_i`, the noise variance
scales as

```
Var(noise_i[k]) ∝ 1 / (snr_lin_i * w_i[k]**2)
```

(the branch's own noise floor scaled by `1/snr_lin_i`, amplified by
`1/w_i[k]**2` wherever that carrier faded). The MRC/inverse-variance
combining weight is then `snr_lin_i * w_i[k]**2`, giving

```
combined[k]  = sum_i(snr_lin_i * w_i[k]**2 * latent_i[k]) / sum_i(snr_lin_i * w_i[k]**2)
weight[k]    = min(1, sqrt(sum_i(snr_lin_i * w_i[k]**2) / max_i(snr_lin_i)))
```

The `max_i(snr_lin_i)` reference and the outer `min(1, ...)` are
deliberate: the combined weight **never exceeds 1**, so a confident
multi-branch combine never reports more certainty to the decoder than a
single clean branch did during training -- the decoder (`Decoder.forward`
in `sstvae/models/autoencoder.py`) was only ever trained on `weight in
[0, 1]`, and feeding it `weight > 1` would be out of distribution in a
way nothing has tested. That means the diversity gain is entirely in
the *latent value* having lower variance for the same nominal weight --
never advertised through the weight itself, which is why the frozen
decoder needs no retraining to benefit from it.

`N = 1` reduces to `weight[k] = min(1, w_1[k]) = w_1[k]` exactly --
`combine_demod_results` special-cases it to return the branch unchanged
rather than round-tripping through the arithmetic.

Two failure modes are handled explicitly (`tests/test_diversity.py`):
a latent erased on every branch (`w_i[k] = 0` everywhere) combines to
`latent = 0, weight = 0`, matching single-branch erasure semantics, not
a NaN from `0/0`; and a mode mismatch between branches (different
header decodes -- not looking at the same transmission, or a bad
header) is a hard `ValueError` rather than silently picking one
branch's guess.

The derivation assumes **independent** noise between branches, true for
separate antennas. Feeding it the same recording twice isn't a
diversity scenario (the "noise" is perfectly correlated) and overstates
the combined confidence; the combined latent value is still correct in
that degenerate case since there's nothing to average away, only the
reported weight is optimistic (`test_two_identical_clean_branches_dont_distort_the_signal`).

## API

```python
from sstvae.modem import Modem
from sstvae.modem.diversity import combine_demod_results, demodulate_diversity

modem = Modem()
result = demodulate_diversity(modem, [branch_a_audio, branch_b_audio])
# or, if you already have DemodResults:
result = combine_demod_results([modem.demodulate(a), modem.demodulate(b)])
```

`demodulate_diversity` drops a branch that fails to acquire at all
(`SyncError`) rather than failing the combine -- that's the point of
diversity reception, one antenna losing lock entirely while the other
doesn't. If every branch fails, the first branch's `SyncError`
propagates. `combine_demod_results` generalizes to any number of
branches (`N >= 1`), not just two.

## Measured gain

`scripts/diversity_sweep.py`, mode A, 20 trials/point, latent SNR
(`SNR_REF_BW_HZ`-referenced, same convention as everywhere else in this
codebase). "Single branch" is the *better* of the two independent
branches each trial -- what an operator picking the stronger-looking
antenna would get without combining:

| Channel | Single branch -> combined | Gain |
|---|---|---|
| AWGN 6 dB | 3.8 -> 6.7 dB | +2.9 dB |
| AWGN 3 dB | 0.5 -> 4.8 dB | +4.4 dB |
| AWGN 0 dB | -2.9 -> 2.6 dB | +5.5 dB |
| AWGN -2 dB | -4.6 -> 1.1 dB | +5.8 dB (16/20 both-branch locks) |
| mpp 10 dB | 1.4 -> 6.4 dB | +5.1 dB |
| mpp 6 dB | -1.2 -> 4.6 dB | +5.8 dB (19/20) |
| mpp 3 dB | -3.1 -> 2.8 dB | +5.9 dB (17/20) |

Two things worth noting in that table. First, the gain is **not** flat:
at high single-branch SNR it sits close to the textbook two-equal-branch
MRC prediction of +3 dB (branch SNRs sum in linear terms --
`test_diversity_gain_under_independent_awgn` checks this against the
`10*log10(sum(snr_lin))` prediction directly, within 1.5 dB); it grows
past +5 dB as SNR drops and under fading, because diversity's real
payoff is avoiding *simultaneous* deep fades on both branches, and that
effect gets stronger, not weaker, as the channel gets worse. Second,
`AWGN -4 dB` is missing a row: at that point single-branch acquisition
itself starts failing often enough that too few trials had *both*
branches lock to report a mean (see "What's not done").

A 12-trial/point run across all three modes (`--modes ABC --trials 12`)
reproduces the mode A numbers above to within 0.1 dB on modes B and C as
well, at every AWGN and `mpp` point both reached -- expected, since
latent SNR is a per-latent modem-domain quantity that doesn't depend on
how many frames a mode sends, but worth checking rather than assuming.
That run also got far enough to see `AWGN -4 dB`, at +5.5 dB (mode B)
and +5.6 dB (mode C) -- consistent with the trend above, but from a
single both-branches-locked trial out of 12 each (`docs/todo.md`'s
warning applies: single-digit trials at threshold will show you
whatever pattern it feels like), so treat those two numbers as "a data
point," not a measurement.

Re-run with `python scripts/diversity_sweep.py --modes ABC --trials 20`
for a larger, fuller AWGN/fading grid.

## What's not done

- **Both branches must acquire independently.** `demodulate_diversity`
  runs full preamble acquisition (`sync.acquire`) and header decode per
  branch before any combining happens, so a branch too weak to lock at
  all contributes nothing -- diversity reception classically also helps
  exactly in that regime (one antenna too faded to demod alone, but
  useful once combined with the other's channel estimate). Reaching
  that would mean acquiring on one branch and using its timing to
  *assist* the other's demod (or a true raw-domain multi-branch MRC,
  which was considered and set aside -- see below), not the
  post-`demodulate()` combine this implements.
- **No raw-domain (pre-equalization) combining.** MRC on the raw OFDM
  symbols, weighted by each branch's own noise variance, would in
  principle do slightly better (it doesn't lose anything to each
  branch's independent zero-forcing step first) and could combine
  header soft-bits too, which would help right at the acquisition
  threshold above. It needs per-branch noise-variance estimates *during*
  demod (not just the post-hoc `snr_db`) and duplicating or refactoring
  the frame loop in `modem.py`'s `demodulate()`, which is meaningfully
  more invasive for a gain this experiment didn't need in order to show
  the effect is real and worth the complexity.
- **`decode_loop_diversity` (both languages) is preamble-path only, no
  blind fallback.** It inherits the "both branches must acquire
  independently" limitation above, and additionally gives up
  `decode_loop`'s retrospective mid-stream decode and progress-stall
  detection for a reception that never gets a full header lock on
  either branch. Combining two branches' *blind* results needs matching
  them by the beacon's absolute frame counter rather than by preamble
  position, and a different result shape
  (`BlindDemodResult`/`modem::BlindDemodResult` has no `.mode`) -- real
  additional work this does not attempt. Kept as a separate function
  from `decode_loop` on both sides rather than folded in, since that
  function's state machine is the reference's load-bearing one
  (CLAUDE.md); the two-ring case is therefore some duplication rather
  than a generalization.
- **The waterfall only ever follows the primary branch.** The native
  app's receive panel does not show a second spectrum strip for the
  diversity device, and there is no equivalent live view on the CLI
  side either.
- **No test with genuinely unequal branches** (e.g. one antenna 10 dB
  worse than the other) or with more than two branches -- the formula
  and the tests both generalize to `N > 2`, but it's unmeasured.
- **The GUI side of the native port (`native/gui/rx_panel.{hpp,cpp}`,
  `native/gui/settings_dialog.{hpp,cpp}`) was written against the
  existing Qt patterns but could not be built or screenshot-tested in
  the environment that wrote it** (no Qt6 installed) -- unlike the
  `core/` changes, which have real ctest coverage
  (`native/tests/test_diversity.cpp`, `test_rx_engine.cpp`'s
  `DiversityHarness`) and were verified against a from-scratch offline
  build. Build and exercise the GUI on a machine with Qt before
  trusting it.

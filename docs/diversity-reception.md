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
--device2`, and the native app's Audio settings tab, beside the primary
input device: "Diversity reception (second antenna)" plus a second
input-device picker). Both
also support an optional debug visualization,
`contribution_image`/`modem::diversity::contribution_image` --
`--diversity-debug-image` on the CLI, a checkbox in the native
settings dialog -- a red/blue heatmap of which branch supplied each
transmitted latent *and* how much either had to offer, written as
`<name>_diversity.png` beside the picture: rows are the data carrier
index (frequency order, contiguous), columns are absolute frame index
(time), hue is the fractional split and brightness is the combined
strength relative to this reception's own peak cell -- see
"`contribution_image`: hue and brightness together" below for why both
axes are needed. Each branch also independently
falls back to blind acquisition (`Modem.demodulate_blind`) when it
can't get a header lock, same as `decode_loop` does for a single
receiver -- see "Combining blind-acquired branches". `Progress`/
`SharedState` also publish `branch_a_locked`/`branch_b_locked` --
whichever ring most recently supplied a hit (header or blind) that fed
the last poll's combine, `False` for a branch that never acquired, is
too far from the other's `reception_start` to be treated as the same
transmission, or has dropped out of range since the previous poll (this
is *not* latched for a reception's lifetime; the native receive panel's
"Primary"/"Secondary" lamps track it every poll and can flip back to
unlocked mid-reception, same as the underlying state). See "What's not
done" for what's left (raw-domain combining, a second waterfall,
unequal-branch/N>2 measurements).

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
The cost is that each branch must *independently* acquire -- the
preamble path if it can, `demodulate_blind`'s pilot-periodicity path
if it can't (see "Combining blind-acquired branches" below) -- rather
than one branch's acquisition ever assisting another's.

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

## `contribution_image`: hue and brightness together (2026-08-05)

The first version of this heatmap encoded only `branch_contribution`
-- each branch's *fractional* share of the combine, `mrc_w / sum(mrc_w)`
from the derivation above -- as hue: red for branch 0, blue for branch
1, magenta for an even split. That has a blind spot. Fraction says
nothing about *magnitude*: two branches each contributing a healthy
weight of 0.9 and two branches each contributing a faded 0.05 both
split the latent 50/50, and both drew as the identical saturated
magenta. The image could not tell "combining these two helped a lot
here" from "combining these two barely mattered here, there was almost
nothing to combine" -- exactly the distinction a debug view like this
exists to show.

The fix adds a second channel of information, riding on data the
combine already computes: `weight[k]` from "The combining weight"
above (the branches' *combined*, not fractional, confidence at that
latent -- `min(1, sqrt(sum_i(snr_lin_i * w_i[k]**2) / max_i(snr_lin_i)))`).
Brightness scales by this value, normalized to the *brightest cell this
particular reception ever reached* rather than exposed on its raw
`[0, 1]` scale -- a reception that never got much above 0.3 anywhere
should still show its strongest carriers at full brightness, the same
way the hue itself is relative to what the two branches had between
them rather than to some absolute unit. A carrier that fades on one
branch but stays strong on the other still reads as a saturated, bright
color (that is the case diversity reception exists to rescue); a
carrier that fades on *both* branches goes dark regardless of how
evenly they split what little they had, down to black at the limit
where every branch erased it.

Implementation-wise this is one extra array riding alongside
`branch_contribution`'s fractional-share one, computed from the same
per-latent MRC weights so there is no second pass over the branches:
Python's `_combined_weight` (private; `contribution_image` is the only
caller with a reason to want the un-normalized value) and C++'s
`contribution_data` (an internal struct pairing `frac` with `overall`,
which `branch_contribution` uses just the first half of). Both are
verified directly (`test_contribution_image_darkens_when_both_branches_
fade_together` in both suites): two branches given *identical* weights
throughout, so the hue is an even 50/50 split everywhere by
construction, but pinned to full strength for one frame's latents and a
low baseline for the rest -- the full-strength frame's column comes out
more than 5x brighter than the low-baseline one despite matching hue in
both, and the peak column reaches full brightness (255) by construction
of the normalization. `test_contribution_image_pure_branch_is_
saturated_in_its_color` (renamed `..._is_pure_hue_peaking_at_full_
brightness`) had to change with this: a single live branch against a
dead one is no longer *uniformly* saturated red -- brightness now
tracks the live branch's own per-carrier weight, which even on an
unfaded channel varies slightly carrier to carrier from ordinary pilot
estimation noise -- only its hue (no blue at all) and its peak cell
(near-255) are still asserted.

## Combining blind-acquired branches

A branch too weak for the preamble path isn't necessarily useless --
`Modem.demodulate_blind` locks onto the frame pilot's own periodicity
and, once the beacon's absolute frame counter decodes, places latents
in canonical order without ever having seen a header (see
`sstvae/modem/beacon.py`). `BlindDemodResult.latents`/`.weights` are
always sized to mode C's full range and positioned by that absolute
counter, so two independent blind locks of the *same* transmission land
in the same array positions automatically -- more directly comparable
than two header locks, whose sample positions need the epsilon-based
matching above. That makes blind-acquired branches combine with exactly
the same MRC arithmetic as header-acquired ones; only the branch's
*type* differs, not the math.

`combine_diversity_results` (Python) / `combine_diversity_results`
taking a `Branch = std::variant<DemodResult, BlindDemodResult>` (C++)
handles any mix:

- **Both header-locked** -- delegates to `combine_demod_results`
  unchanged (requires one mode, as before).
- **Both blind-locked** -- delegates to `combine_blind_results`, which
  needs no mode check and no size reconciliation (both already full
  mode-C-sized).
- **Mixed** -- the header-locked branch's mode is authoritative for
  what was actually sent, so the blind branch's arrays are combined at
  full mode-C size, then truncated back down to the header mode's
  range. A header branch's own (mode-sized, not full-C-sized) arrays
  are padded up to match first, using the same zero-fill
  `codec.pad_to_full` already does elsewhere.

`decode_loop_diversity` uses this per-branch preference -- header first,
falling back to blind (`_find_branch_reception` in Python,
`find_branch_reception` in C++) -- mirroring `decode_loop`'s own
single-receiver preference exactly. Both paths report a branch's
position as `reception_start`, expressed at the *preamble's* sample
offset regardless of which path found it (the blind path's
`frame0_start` is one preamble+header later, so it's shifted back by
that much, the same correction `decode_loop`'s blind branch already
applies) -- which is what lets two branches be matched by the same
epsilon criterion no matter how each one locked. For two blind
branches that position agreement is a sanity check ("are these really
the same transmission"), not an alignment requirement, since their
arrays are already aligned by the frame counter regardless of sample
position.

Completion tracking follows the same header-beats-blind preference: if
the combine ended up header-locked, the exact frame count is known and
the reception finishes when it's reached; if every branch was
blind-locked, completion falls back to the progress-stall detection
(`config.end_grace`) `decode_loop` already uses for its own blind path.
`contribution_image` follows the same rule for `n_frames` -- the
header-locked mode's frame count if there is one, else mode C's full
range.

## API

```python
from sstvae.modem import Modem
from sstvae.modem.diversity import (
    combine_demod_results,      # header-locked branches only
    combine_blind_results,      # blind-locked branches only
    combine_diversity_results,  # any mix of the two
    demodulate_diversity,
)

modem = Modem()
result = demodulate_diversity(modem, [branch_a_audio, branch_b_audio])
# or, if you already have DemodResults / BlindDemodResults:
result = combine_demod_results([modem.demodulate(a), modem.demodulate(b)])
result = combine_diversity_results([modem.demodulate(a), modem.demodulate_blind(b)])
```

`demodulate_diversity` drops a branch that fails to acquire at all
(`SyncError`) rather than failing the combine -- that's the point of
diversity reception, one antenna losing lock entirely while the other
doesn't. If every branch fails, the first branch's `SyncError`
propagates; it is header-path only (`Modem.demodulate`) -- a caller
that wants blind-fallback diversity from raw audio combines
`Modem.demodulate`/`demodulate_blind` results itself with
`combine_diversity_results`, which is what `decode_loop_diversity`
does. All three combine functions generalize to any number of branches
(`N >= 1`), not just two -- `contribution_image` is the one exception,
fixed at two (red/blue is a two-way encoding).

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

- **A branch that can't even blind-lock still contributes nothing.**
  Blind acquisition needs `MIN_FRAMES_FOR_SYNC` (~10.5 s) of intact
  frame pilots to see a full beacon superframe and has its own
  SNR/threshold behavior (`sync.acquire_blind`), so a branch degraded
  enough to fail *that* too is still dropped, same as it always was for
  the header path alone. Diversity reception classically also helps in
  that regime (one antenna too faded to demod alone, but useful once
  combined with the other's channel estimate); reaching it would mean
  acquiring on one branch and using its timing to *assist* the other's
  demod, or a true raw-domain multi-branch MRC (see below) -- neither
  attempted here.
- **No raw-domain (pre-equalization) combining.** MRC on the raw OFDM
  symbols, weighted by each branch's own noise variance, would in
  principle do slightly better (it doesn't lose anything to each
  branch's independent zero-forcing step first) and could combine
  header soft-bits too, which would help right at the acquisition
  threshold. It needs per-branch noise-variance estimates *during* demod
  (not just the post-hoc `snr_db`) and duplicating or refactoring the
  frame loop in `modem.py`'s `demodulate()`, which is meaningfully more
  invasive for a gain this experiment didn't need in order to show the
  effect is real and worth the complexity.
- **Cross-branch threshold lowering, considered and not implemented.**
  Once one branch has a *confirmed* preamble lock, the other branches'
  search window for the same transmission could run at a lower
  detection threshold: a real preamble is now known to be in that
  window (from the confirming branch), so the usual false-alarm concern
  a low threshold buys into doesn't apply the same way -- distinct from,
  and not obsoleted by, blind-fallback above. Blind acquisition rescues
  a branch too degraded for the preamble path *at all*, using a longer
  window (~10.5 s) and its own acquisition statistics; threshold
  lowering would instead rescue a branch whose preamble is only
  marginally below the normal threshold, without waiting for that
  longer window or giving up the header path's richer per-frame
  progress tracking. Restricting the search to a narrow window around
  the confirmed branch's position (rather than the whole buffer) already
  suppresses false alarms geometrically on its own, similar to how
  `decode_loop_low_cpu` narrows its search window for a different
  reason -- worth quantifying before picking a lowered threshold.
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
  `DiversityHarness`, including `Progress::branch_a_locked`/
  `branch_b_locked` assertions for the both-locked, single-branch-
  fallback and header+blind-mix cases) and were verified against a
  from-scratch offline build. The receive panel's "Primary"/"Secondary"
  lock lamps (`rx_panel.cpp`'s `set_lock_lamp`, following the
  `ptt_label_`/`QPalette` pattern from `main_window.cpp` -- never a
  stylesheet) are part of that unverified GUI surface. Build and
  exercise the GUI on a machine with Qt before trusting it.

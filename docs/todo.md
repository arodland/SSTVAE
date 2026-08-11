# TODO

Known work items that aren't bugs with a clear fix, kept here so the
reasoning doesn't have to be rediscovered.

## Preamble detection against a steady-carrier interferer

**A pure tone is a perfect-looking preamble to `sync.acquire()`, and
carriers are everywhere on HF.** Found 2026-08-09 while designing the
Android app's VOX leader tone; the leader was the trigger, but the
general case is the reason this is here.

**Mechanism.** `_autocorr_metric` is `|sum of z[n+M]·conj(z[n])| /
energy` over a 480-sample window. For a steady tone the product is
constant, so numerator and denominator are equal and the metric is
**exactly 1.000** — the ceiling, above anything a real preamble reaches
in noise (0.936 at 10 dB, 0.853 at 6 dB, measured below). The metric is
deliberately scale invariant, so a *weak* tone reads 1.000 too. And
`acquire()` takes a hard `argmax` over the whole search window, so if
the tone's window ever out-scores the preamble's there is no second
chance: the template-correlation stage only searches ±200 samples around
the winner, and the real preamble is thousands of samples away.

**Measured** (mode A, AWGN, tone at 1400 Hz — on a carrier — 2 s of
lead-in before the transmission, one seed per cell). "lock err" is the
acquired preamble start minus the true one; latent SNR after the picture
decoded:

| tone vs signal RMS | clean | 20 dB | 10 dB | 6 dB |
|---|---|---|---|---|
| none    | +800, 10.1 dB | +800, 9.7 dB | +800, 6.9 dB | +800, 3.9 dB |
| −20 dB  | **wrong lock** | +800, 8.9 dB | +800, 6.5 dB | +800, 3.8 dB |
| −10 dB  | **wrong lock** | +800, 3.2 dB | +800, 2.5 dB | +800, 1.2 dB |
| −6 dB   | **wrong lock** | +800, 4.5 dB | +800, 3.4 dB | +800, 1.6 dB |
| −3 dB   | **wrong lock** | +800, 5.2 dB | +800, 3.6 dB | +800, 1.2 dB |
| 0 dB    | **wrong lock** | header fails | header fails | header fails |
| +6 dB   | **wrong lock** | **wrong lock** | **wrong lock** | **wrong lock** |

Three things to read out of that, in decreasing order of importance:

- **A noise floor is what saves this today.** In the clean rows the
  lead-in is digital silence, so a tone 20 dB *below* the signal is the
  loudest periodic thing in the window and wins outright. Add any noise
  and the preamble holds its lock up to a tone around −3 dB, which is
  why this has never been seen on air. It also means the failure is a
  property of the *quietest* part of the search window, not of the
  signal-to-interference ratio where the signal is — so the case to
  worry about is a strong carrier plus a weak wanted signal, which is
  exactly the case someone reaches for the receiver in.
- **At tone ≈ signal the lock is still exact and the header is what
  fails.** So acquisition is only half the problem; the other half is
  demod, and the two want different fixes.
- **The quality cost arrives long before the lock does.** A tone 10 dB
  under the signal costs 6.5 dB of latent SNR at 20 dB — a mangled
  picture from an interferer most operators would not think twice about.
  The row is non-monotonic (−10 dB is worse than −3 dB, consistently
  across all three SNRs), which is suspicious in an interesting way: the
  likely explanation is that the per-latent confidence weights notice a
  *strong* interferer and effectively erase the affected carrier, while
  a moderate one gets trusted. Unverified, and worth checking first if
  anyone works on the demod half — if it is right, the fix is making
  that de-weighting kick in earlier rather than anything new.

**Fix candidates, best first.**

1. **Keep the top-K metric peaks and let the header arbitrate.** The
   template-correlation and Golay-header stages already reject a tone
   convincingly (the 0 dB row locks correctly and *still* fails the
   header, i.e. the header is a working discriminator); the only reason
   a tone is fatal is the hard argmax in front of them. Take the K
   highest peaks with a minimum separation, try each in turn, accept the
   first whose header decodes. **This cannot cost sensitivity** — the
   first candidate is exactly today's argmax, so every signal that
   acquires now still acquires — which is the property the other options
   do not have. Cost is K template correlations over a ±200-sample
   segment, small next to the FFT already spent on the metric. Needs
   `acquire()` restructured to return candidates or to take a "does this
   one pass" predicate, since the header lives above it in `demodulate`.
   Noted once before, as speculation, in the withdrawn large-offset
   section below; the tone case makes it concrete.
2. **Discriminate on bandwidth, not periodicity.** A tone is one bin
   wide and the preamble is ~24, so a spectral flatness measure over the
   correlation window separates them cleanly and is a multiplier on the
   metric rather than a replacement. Costs sensitivity in principle
   (anything multiplying the metric can push a marginal preamble under
   threshold) and needs measuring against the acquisition curve, which
   is why it ranks below option 1 despite being the more direct answer.
3. **Notch the interferer for the sync path only.** `sync_lowpass()`
   already has its own filter copy, so a notch there is free of the ISI
   argument that keeps `to_baseband()` unfiltered — but it only helps
   acquisition, and the table above says the quality loss is the bigger
   cost. Notching in the demod path is a real question, not a free one.

**Measurement caveats.** One seed per cell, one tone frequency, one
mode, AWGN only. The −3 dB boundary is an order of magnitude, not a
swept number — and this file's own standing warning applies (≥25 seeds
before quoting any acquisition success rate).

**What was done about it in the meantime.** Nothing in the receiver.
The Android VOX leader is a **chirp**, not a tone, precisely because of
this: a 500 ms sweep across the passband decorrelates at lag M, and
measured, acquisition lands on the true preamble with the same latent
SNR as no leader at all. That is a transmitter-side dodge of one
instance of the problem, not a fix for the general one.

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

**Measured end to end 2026-08-11, and it is smaller than this section
used to think.** The answer to "can we buy range by spending CPU" is
that for the preamble path there is almost no CPU to spend: the range is
set by a *constant*, `max_bins`, and raising it to cover ±600 Hz costs
**about 4 ms on top of a 63 ms acquisition**, no sensitivity and no
false alarms. Two of the three things this section previously listed as
"would have to change" do not have to change, and one of them was
wrong about why.

### `max_bins` is the whole limit, out to ±700 Hz

Mode A, AWGN at 0 dB, 8 seeds per cell, everything else stock. A trial
counts as success only if acquisition lands within ±4 samples of the
true preamble start **and** estimates CFO within 2 Hz — i.e. only if the
header would then decode. `#` is 8/8, `.` is 0/8:

| `max_bins` (reach) | 0 | 200 | 400 | 600 | 700 | 800 | 900 | 1000 | 1200 |
|---|---|---|---|---|---|---|---|---|---|
| 2 (±125 Hz, today) | # | . | . | . | . | . | . | . | . |
| 6 (±325 Hz) | # | # | . | . | . | . | . | . | . |
| 12 (±625 Hz) | # | # | # | # | . | . | . | . | . |
| 16 (±825 Hz) | # | # | # | # | 7 | 2 | . | . | . |
| 26 (±1325 Hz) | # | # | # | # | 7 | 2 | 1 | 1 | . |

Each row holds right up to its own arithmetic reach and then stops, so
up to ±625 Hz the bin count is the *only* thing in the way. Past that,
more bins buy nothing: 16 and 26 are the same row.

**Detection is CFO-blind by construction**, which is why none of this
touches the threshold. `_autocorr_metric` sums `z[n+M]·conj(z[n])`; a
frequency offset multiplies every one of those products by the same
constant `exp(-2πi·f·M/FS)`, and `|·|` removes it. A mis-tuned preamble
therefore produces the same metric as a correctly-tuned one, minus only
whatever the sync filter has taken out of it. Clean, the peak metric
reads 1.000 at every offset from 0 to ±400 Hz and the failure is always
`header decode failed` — never `no preamble found`.

### The sync lowpass is the *next* wall, not the first one

**This section previously said `dsp.sync_lowpass()` is why a mis-tuned
signal shows `NO LOCK` well before `max_bins` binds. That is wrong.**
The stock `firwin(129, 850)` carries acquisition to ±600 Hz at 0 dB
with no change at all (the 8/8 row above): the preamble is 24 carriers,
and losing the third of them that has walked out of the passband still
leaves an overwhelming correlation. Widening the filter does move the
wall, and also costs — same measurement, `max_bins=26`, 0 dB:

| detect cutoff | 600 | 700 | 800 | 900 | 1000 | 1200 |
|---|---|---|---|---|---|---|
| 850 Hz (stock) | 8/8 | 7/8 | 2/8 | 1/8 | 1/8 | 0/8 |
| 1200 Hz | 8/8 | 7/8 | 7/8 | 8/8 | 2/8 | 0/8 |
| 1600 Hz | 8/8 | 7/8 | 6/8 | 7/8 | 7/8 | 2/8 |
| 2200 Hz | 1/8 | 0/8 | 0/8 | 0/8 | 1/8 | 0/8 |

The 2200 Hz row is the trade arriving: the filter sets the noise
bandwidth the metric integrates over, so a wide one admits enough noise
to lose a 0 dB signal that a narrow one holds. **Don't widen it.**
±600 Hz is already the whole physically reachable range (see *Practical
bound* below), the stock filter reaches it, and the peak signal metric
falls monotonically with cutoff — 0.856 → 0.737 at 6 dB going from 850
to 3600 Hz, measured at zero offset, i.e. paid by *every* station
whether mis-tuned or not.

One thing widening does buy, worth recording because it is the opposite
of the intuition: the **noise-only** metric floor *drops* with a wider
filter (max over 60 s of AWGN, 5 seeds: 0.295 at 850 Hz, 0.227 at
1600 Hz, 0.196 at 3600 Hz), because a wider band puts more independent
samples inside the same 480-sample correlation window. So the false-alarm
argument and the sensitivity argument point in opposite directions here,
and sensitivity is the one that loses more.

### It costs no sensitivity, and it cannot cost false alarms

**Sensitivity: none, and not by a narrow margin — the extra candidates
never win.** Zero offset, 25 seeds per cell:

| `max_bins` | 0 dB | −1 dB | −2 dB | −3 dB | −4 dB |
|---|---|---|---|---|---|
| 2 | 25/25 | 25/25 | 24/25 | 16/25 | 11/25 |
| 12 | 25/25 | 25/25 | 24/25 | 16/25 | 11/25 |
| 26 | 25/25 | 25/25 | 24/25 | 16/25 | 11/25 |

Identical rows, not merely similar ones. Checked directly rather than
inferred: over 160 trials at 0, −3, −5 and −7 dB — spanning from "always
acquires" to "never acquires" — `max_bins=2` and `max_bins=12` returned
the **same preamble start and the same CFO in every single case**, 0
disagreements. The correct bin's template-correlation score beats a
wrong bin's by so much that adding candidates changes nothing.

**False alarms: structurally unchanged.** The detection gate is the
metric against `PREAMBLE_THRESHOLD`, and `max_bins` is not in that path
at all, so the *rate* of false detections cannot move. What could have
moved is what happens behind one: `acquire()` emits a single
best-scoring candidate, so more candidates might mean a better-fitting
noise segment reaching the Golay header. Measured by forcing a lock
(threshold 0) on 400 windows of pure noise and running the header
behind it: **1/400 accepted at `max_bins=2`, 0/400 at 12, 0/400 at 26**
— consistent with the header's own 7.2e-4 and with no increase. The
reason there can't be one is that the bin search takes an argmax and
emits one answer, so exactly one header attempt happens regardless of
how many bins were searched. (Contrast the top-K *time* peaks idea in
the steady-carrier section above, which genuinely does multiply header
trials — that is a different gate and needs its own accounting.)

### CPU: the bin search is not where the time goes

Warm, medians:

| buffer | detection stage alone | `max_bins=2` | 12 | 26 |
|---|---|---|---|---|
| 32 s (mode A) | 63.6 ms | 63.4 ms | 67.0 ms | 71.0 ms |
| 130 s (full ring) | ~365 ms | ~350–374 ms | ~371–382 ms | ~367–376 ms |

The `sync_lowpass` + metric FFT over the whole buffer is essentially the
entire cost and it does not depend on the search range; a candidate is a
`freq_correct` plus an FFT convolution over an ~1100-sample segment,
about **0.14 ms each**. Going from ±125 Hz to ±625 Hz is 24 extra
candidates, ~3.4 ms, or **5% of one acquisition on a 32 s buffer**. On a
full ring it is not measurable at all — every cell in that row, the
bin-free detection stage included, lands inside one run-to-run spread of
every other. On Android, where `sync::acquire` was measured at 171 ms
per poll, expect the same 2–4% as the 32 s case.

So this need not be opt-in on the preamble path, and need not be a
second pass. `max_bins=12` as the default is defensible on the
measurements above: it reaches the whole band an SSB filter passes, it
returns bit-identical answers to today's code for every signal today's
code acquires, and it costs 3.4 ms.

### Decode after a wide acquisition

Mode A at 6 dB, 3 seeds, `max_bins=14`, full `demodulate()`:

| offset (Hz) | 0 | 200 | 400 | 600 | −600 |
|---|---|---|---|---|---|
| CFO estimate error (Hz) | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| frames | 220 | 220 | 220 | 220 | 220 |
| latent SNR (dB) | 3.95 | 3.79 | 3.64 | 3.60 | 3.84 |

Every frame, exact CFO, and ≤0.4 dB across the range — consistent with
the retuned-centre table at the top of this section, and small enough at
3 seeds that it is not worth attributing to a mechanism.

### The blind path costs real CPU — but far less than it looks

`sync.acquire_blind` / `BlindAccumulator` search CFO *directly* over
`max_offset_hz` (default 55.0) at `bin_step_hz` (1.7) resolution, so
their cost is linear in the number of bins — and this is already the
dominant per-poll cost of the receive loop. Widening the range naively
is therefore ~linear in the range. Measured on a 20 s window (absolute
times are machine- and load-dependent, ratios are not):

| range | bins | `acquire_blind` | `BlindAccumulator.push` |
|---|---|---|---|
| ±55 Hz (today) | 67 | 453 ms | 263 ms |
| ±125 Hz | 149 | 920 ms | 607 ms |
| ±300 Hz | 355 | 2071 ms | 1356 ms |
| ±625 Hz | 737 | 4435 ms | 2773 ms |

**But the bin count is the wrong thing to scale, because the grid is
~15× finer than the signal can support.** For a fixed lag, the matched
filter as a function of CFO is `|Σ_n (z[j+n]·k*[n]) e^{-2πi f n/FS}|`
over n = 0..159 — a DTFT of a **160-sample** sequence. So `|mf|²` is
exactly band-limited in CFO, with dual support 319 samples, and is fully
determined by samples every **FS/(2·160) = 25 Hz**. The fold is a sum of
such terms and inherits the property. A 1.7 Hz step is not resolution,
it is interpolation done the expensive way.

Measured, and it holds. Detection over ±55 Hz, 12 seeds, signal offset
drawn uniformly so the scalloping between bins is sampled fairly:

| bin step | −3 dB | −6 dB | −8 dB | −10 dB |
|---|---|---|---|---|
| 1.7 Hz (today) | 12/12 | 12/12 | 12/12 | 12/12 |
| 6.8 Hz | 12/12 | 12/12 | 12/12 | 12/12 |
| 12.5 Hz | 12/12 | 12/12 | 12/12 | 12/12 |
| 25 Hz | 12/12 | 12/12 | 12/12 | 11/12 |
| 50 Hz | 12/12 | 12/12 | 12/12 | **3/12** |

12.5 Hz is free; 25 Hz is where the ~1 dB of worst-case scalloping loss
starts to show at the very bottom; 50 Hz is past the sampling limit and
collapses, exactly as predicted rather than gradually.

**The frequency estimate does not have to suffer, and in fact improves.**
A coarse grid's raw argmax is coarse — 3.7 Hz of mean error at a 12.5 Hz
step, which matters, because the drift section below shows ~2 Hz of
uncorrected residual CFO is already costing the picture. But the folded
score is band-limited in CFO, so the peak *between* bins is recoverable;
a parabola through the winning bin and its two neighbours is the cheapest
version of that. Over the full ±625 Hz range at a 12.5 Hz step, 12 seeds,
offset drawn uniformly across the whole range:

| SNR | detects | raw error | interpolated | interpolated worst |
|---|---|---|---|---|
| +6 dB | 12/12 | 3.71 Hz | **0.14 Hz** | 0.45 Hz |
| 0 dB | 12/12 | 3.71 Hz | 0.27 Hz | 0.60 Hz |
| −3 dB | 12/12 | 3.71 Hz | 0.31 Hz | 0.57 Hz |
| −6 dB | 12/12 | 3.71 Hz | 0.46 Hz | 1.02 Hz |
| −8 dB | 12/12 | 3.71 Hz | 0.62 Hz | 1.47 Hz |

That interpolated estimate is **better than today's 1.7 Hz grid gives
raw** (0.56 Hz mean, 1.54 Hz worst, measured over ±55 Hz), from 101 bins
instead of 737.

**Batching and threads are the small win, not the big one.** Doing the
per-bin IFFTs as one batched `ifft(..., axis=1, workers=-1)` and the
periodic fold by reshape-and-sum instead of `np.add.at` is numerically
identical (max relative difference 6.8e-16, same phase bin, same CFO
bin) and worth ~1.1× from the batching, ~1.5× from the threads on four
cores. The IFFTs genuinely are the work; there is no factor hiding in
the loop overhead. Worth taking, but it is the grid that changes the
answer.

Put together, on a 20 s window, best of 3:

| configuration | bins | `push` |
|---|---|---|
| reference, ±55 Hz @ 1.7 Hz (today) | 67 | 273 ms |
| batched, ±55 Hz @ 1.7 Hz | 67 | 215 ms |
| batched + threads, ±55 Hz @ 1.7 Hz | 67 | 161 ms |
| batched + threads, **±55 Hz @ 12.5 Hz** | 11 | **131 ms** |
| batched + threads, **±625 Hz @ 12.5 Hz** | 101 | **516 ms** |
| batched + threads, ±625 Hz @ 25 Hz | 51 | 296 ms |
| reference, ±625 Hz @ 1.7 Hz (the naive widening) | 737 | 2780 ms |

So **11× the frequency range for under 2× today's CPU**, against 10× for
the naive version — or, if the range is left alone, **today's range for
half today's cost**, which is the more interesting number for Android,
where this is the battery item and `decode_loop_low_cpu` exists because
of it.

What widening does *not* cost, measured either way:

- **Sensitivity: none.** Mode A frames, 20 s window, 8 seeds: ±55 Hz and
  ±625 Hz both detect 8/8 at +6, +3, 0 and −3 dB; and the ±625 Hz @
  12.5 Hz configuration detects 12/12 down to −8 dB with the signal
  placed anywhere in the range.
- **False alarms: essentially none.** Peak score on 20 s of pure noise,
  6 seeds, against a threshold of 4.0: **1.34** at ±55 Hz @ 1.7 Hz and
  **1.38** at ±625 Hz @ 12.5 Hz (worst single seed 1.37 → 1.42). Taking
  a max over more cells barely moves it, for the same reason the grid is
  redundant — adjacent bins are heavily correlated.

**What this would touch.** `max_offset_hz` and `bin_step_hz` are already
parameters, so the range and the grid are settings. The parabolic
refinement is new and is where the care goes: it is what makes a coarse
grid safe, and without it a coarse grid quietly hands `demodulate_blind`
a 3.7 Hz error, which the drift section shows is enough to cost the
picture on its own. Batching and threading `push` is a self-contained
change to one loop with a bit-identical result to check against, in both
implementations.

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
900–2100 Hz — exactly the range measured above, and ±600 Hz from the
nominal 1525 Hz centre. Beyond that the transmitter's own filtering, not
the modem, is the limit, which is why the sync filter's ±700 Hz wall
does not need moving.

**What would have to change**, revised:

- `sync.acquire()`'s `max_bins`, from 2 to ~12. This is the change.
- `sync.acquire_blind` / `BlindAccumulator`'s `max_offset_hz`, if the
  blind path is wanted at the same range — a setting, and the only part
  that costs real CPU, though far less than the bin count suggests once
  `bin_step_hz` is coarsened and the peak interpolated (above).
- **Not** `dsp.sync_lowpass()`, and **not** `dsp.to_baseband()`'s
  `FCENTER`. `ofdm.BASEBAND_FREQS` and the DFT matrix were already
  correctly identified as needing no change.
- The C++ port and `NATIVE_SUBSTITUTIONS` carry `acquire`'s defaults, and
  CLAUDE.md already records what a hand-copied stale default costs — a
  new `max_bins` default has to move in both implementations together.

**Reproducing all of the above.** `scripts/freq_range_sweep.py reach
filter-wall sensitivity cpu blind-cost blind-speed` — it carries the parameterized
`acquire_wide` prototype, which is `sync.acquire` with `max_bins` and the
filter cutoff exposed and nothing else changed.

**Measurement caveats.** 8 seeds per cell for the reach tables and 25
for the sensitivity ones; AWGN only, mode A only, one drift-free static
offset per trial. This file's standing warning (≥25 seeds before quoting
an acquisition success rate near threshold) applies to the 7/8 and 2/8
cells in the ±700–800 Hz region, which are in exactly that regime — the
0/8-vs-8/8 cells that carry the argument are not.

## Frequency drift *during* a transmission

**The receiver estimates carrier frequency once, from the preamble, and
never looks again.** `demodulate()` calls `acquire()`, applies one
constant `freq_correct`, and runs 220–660 frames on it. That is fine for
a stable pair of radios and it is the entire story for a drifting one:
measured 2026-08-11, **today's receiver tolerates about ±2 Hz of
residual offset, instant by instant** — not a drift *rate*, and not an
average. For a ramp starting from an acquired zero the residual only
grows, so that budget is spent as total excursion by the end of the
over, and a mode C over lasting three times as long therefore tolerates
a third of the rate. For wander it is the peak excursion that counts and
the over's length does not enter at all. The Ornstein-Uhlenbeck sweep
below is what separates those two readings; the ramp alone cannot.

Mode A (32 s), AWGN at 6 dB, 3 seeds, recovered latent SNR. `ref` is
today's receiver; `oracle` is de-chirped with the true drift rate, i.e.
the upper bound on what any correction could reach; `track` is the
proposed loop (below):

| Hz/s | total Hz | ref | track | oracle |
|---|---|---|---|---|
| 0 | 0.0 | 3.95 | 3.95 | 3.95 |
| 0.05 | 1.6 | 3.61 | 3.88 | 3.90 |
| 0.10 | 3.2 | **−0.05** | 3.67 | 3.69 |
| 0.50 | 16.0 | −6.51 | 3.72 | 3.74 |
| 1.00 | 32.0 | −7.32 | 3.90 | 3.92 |
| 2.00 | 64.0 | −7.17 | 3.76 | 3.86 |
| 5.00 | 160.1 | −7.09 | **−7.03** | 3.89 |

Mode C (95 s), same conditions, 2 seeds — the same cliff, at a third of
the rate:

| Hz/s | total Hz | ref | track | oracle |
|---|---|---|---|---|
| 0 | 0.0 | 3.77 | 3.75 | 3.77 |
| 0.01 | 1.0 | 3.73 | 3.79 | 3.80 |
| 0.02 | 1.9 | 3.39 | 3.78 | 3.79 |
| 0.05 | 4.8 | **−4.63** | 3.81 | 3.83 |
| 1.00 | 95.4 | −7.59 | 3.79 | 3.80 |

So: ~2 Hz of residual is free, ~3 Hz costs the picture, ~5 Hz destroys
it. For a ramp that is **0.06 Hz/s for mode A and 0.02 Hz/s for
mode C** — tight enough that a rig still warming up after key-down, or a
disturbed path with real Doppler on it, can plausibly cross it.

**It fails silently in the worst way.** Every frame is still received
(220/220, 660/660) and the beacon still decodes, so the receiver reports
a complete, healthy reception and hands over a mangled picture. This is
the same signature as the audio-loss and per-chunk-resampling bugs
CLAUDE.md records — "still syncing and reporting every frame received"
is, once again, not evidence of anything.

### Where the damage is, and where it is not

**Not in preamble acquisition.** The preamble is 88 ms long, so even a
violent drift moves it a fraction of a Hz. Mode A, 6 dB, 6 seeds:

| Hz/s | 0 | 0.1 | 1.0 | 5.0 | 20.0 |
|---|---|---|---|---|---|
| peak metric | 0.856 | 0.856 | 0.856 | 0.857 | 0.857 |
| locked | 6/6 | 6/6 | 6/6 | 6/6 | 6/6 |
| CFO estimate (Hz) | 0.14 | 0.16 | 0.30 | 0.89 | 3.04 |

Flat. The estimate itself drifts away from the frames' true frequency
(3 Hz at 20 Hz/s, because the estimate is taken at the preamble and the
frames start ~150 ms later), but detection and timing do not care.

**Not in blind acquisition either.** `BlindAccumulator` folds
matched-filter power into fixed CFO bins over tens of seconds, so a
drifting signal walking across bins looked like it should smear. It
does not: mode A frames, 6 dB, 4 seeds, 20 s window, score against a
threshold of 4.0 — **20.15 at 0 Hz/s, 20.05 at 0.5, 19.46 at 1.0**, 4/4
throughout. The reason is that the pilot matched filter is one 20 ms
symbol, so its frequency response is ~50 Hz wide and the 1.7 Hz bin step
oversamples it ~30×; a 20 Hz walk stays inside a single bin's response.

**All of it is in demod**, and the oracle column proves that nothing
structural stands in the way: de-chirped with the true rate, mode A
recovers 3.75–3.92 dB at *every* rate up to 10 Hz/s, indistinguishable
from no drift at all.

**The mechanism is pilot-rate aliasing, not inter-carrier
interference.** The per-frame pilot EQ *does* remove the frame-to-frame
phase rotation, which is why small residuals cost nothing. But pilots
are one per frame — 6.94 Hz — so the phase sequence they sample aliases
above ±3.47 Hz, and Catmull-Rom interpolation between them is hopeless
well before that: at 3.2 Hz of residual the phase advances **166° per
frame**, and the interpolated channel estimate the data symbols are
divided by is simply wrong. That is why the cliff sits at ~3 Hz of
excursion rather than at the ±25 Hz where a carrier-spacing argument
would put it.

### The fix: a second-order loop on the pilot common phase

Prototyped in full (a ~30-line addition to `demodulate`'s frame loop,
`track` in the tables above). Each frame, the phase **common to all
carriers** between consecutive pilots is a residual-frequency
measurement, `angle(Σ h_pilot[f]·conj(h_pilot[f−1])) / (2π·T_frame)`;
the phase **slope across** carriers is timing and is already tracked by
`tau_ema`. They are orthogonal — a common rotation cancels out of the
slope, and a slope cancels out of the sum — so the new loop and the
existing one do not fight.

Three things the prototype got wrong first, each of which reads as "the
tracker doesn't work" rather than as a bug:

- **The correction must be a continuous phase ramp within the frame, not
  a constant phase per frame.** A per-frame constant removes exactly
  what the pilot EQ already removes and leaves the frequency error
  inside the frame untouched — which is the part that costs the picture.
  With this wrong, the tracker measurably *hurt*: 9.68 dB against the
  reference's 10.05 at 0.02 Hz/s.
- **The measurement is the residual that survived the correction already
  applied, so it must be integrated, not chased.** An EMA of the form
  `f += α(measured − f)` is a closed loop measuring its own output and
  converges to nonsense; `f += α·measured` is right.
- **It has to be second order.** Drift is a *ramp*, and a first-order
  loop leaves a steady-state lag proportional to rate/α — exactly the
  error being removed. A second integrator on the rate (`β`) takes the
  steady-state ramp error to zero, which is why the `track` column lands
  on the oracle rather than near it.

**It costs nothing when there is no drift**, which is the property that
makes it adoptable: 3.95 vs 3.95 dB clean, −2.85 vs −2.88 at 0 dB,
−0.41 vs −0.41 under `mpg`, −1.56 vs −1.60 under `mpp`.

### Limits

**1. Loop bandwidth against the channel's Doppler spread — the real
constraint.** Fading rotates the common pilot phase too, so a fast loop
chases the channel. Measured on mode A at 6 dB, 6 seeds, the cost at
zero drift and the reach at 1 Hz/s pull in opposite directions:

| α, β | `mpd`, 0 Hz/s | `mpd`, 1 Hz/s | `mpp`, 0 Hz/s | `mpp`, 1 Hz/s |
|---|---|---|---|---|
| (reference, no loop) | −3.15 | −7.61 | −1.60 | −7.39 |
| 0.3, 0.05 | **−5.45** | −4.82 | −1.60 | −1.72 |
| **0.1, 0.01** | −3.40 | −3.33 | −1.56 | −1.69 |
| 0.03, 0.002 | −3.21 | **−6.01** | −1.55 | −2.17 |
| 0.01, 0.0005 | −3.15 | **−7.57** | −1.56 | — |
| (oracle) | −3.15 | −3.10 | −1.60 | −1.62 |

α=0.3 costs **2.3 dB** under `mpd` (2 Hz Doppler spread) at zero drift —
the same trap CLAUDE.md already records for the timing tracker ("chasing
it raw wrecked MPP fading performance"), in a different quantity. α=0.01
is safe and tracks nothing. **α=0.1, β=0.01 is the window**: ≤0.25 dB on
the worst channel and within 0.23 dB of the oracle at 1 Hz/s. That is a
comfortable window, not a knife edge, but it is not wide enough to pick
the gains by eye.

**2. The ceiling is a rate, and it moves with the loop bandwidth** —
see also the OU table below, where the gains that maximize this ceiling
are the ones that cost the most under fading. At
α=0.1 the loop holds to **2 Hz/s** (9.95 dB clean, oracle 10.06) and is
gone at 5 Hz/s; at α=0.3 it holds to 5 Hz/s and is gone at 10. The hard
bound behind both is the ±3.47 Hz per-frame ambiguity: the loop must
correct faster than the residual accumulates, and it has 144 ms to do it
in. Either way the ceiling is 30–100× today's ~0.06 Hz/s, and well past
any rate a radio plausibly produces.

**3. Wander is harder than a ramp, and the ceiling is lower.**
Ionospheric Doppler looks more like a sinusoid than a ramp, and a
second-order loop is built for ramps. Mode A, 6 dB, 4 seeds, α=0.1:

| amplitude | period | peak Hz/s | ref | track |
|---|---|---|---|---|
| 0.5 Hz | 30 s | 0.10 | 3.80 | 3.81 |
| 1.0 Hz | 30 s | 0.21 | 2.88 | 3.80 |
| 2.0 Hz | 30 s | 0.42 | **−4.55** | 3.88 |
| 2.0 Hz | 10 s | 1.26 | −4.56 | 3.13 |
| 5.0 Hz | 30 s | 1.05 | −5.44 | 3.71 |
| 5.0 Hz | 10 s | 3.14 | −5.33 | **−6.66** |

The reference falls over at ±1–2 Hz of wander, as the ramp table
predicts. The loop handles everything up to a 30 s period at ±5 Hz, but
a 10 s period at ±5 Hz beats it — and note it ends up *worse* than not
tracking.

**A sinusoid is still too tidy, so the same sweep with the drift drawn
from an Ornstein-Uhlenbeck process** — mean-reverting, no characteristic
phase, wandering back rather than running away, parameterized by
stationary RMS and correlation time, which is the shape a measurement
would actually report. Mode A, 6 dB, 6 seeds; the oracle removes the
*true* frequency path rather than a fitted ramp, so it is the real
ceiling:

| RMS | τ | peak \|f\| | ref | track α=0.1 | track α=0.3 | oracle |
|---|---|---|---|---|---|---|
| 0.5 Hz | 30 s | 0.68 Hz | 3.78 | 3.80 | 3.80 | 3.80 |
| 0.5 Hz | 10 s | 0.87 Hz | 3.71 | 3.78 | 3.78 | 3.78 |
| 0.5 Hz | 3 s | 1.16 Hz | 3.77 | 3.78 | 3.77 | 3.80 |
| 1.0 Hz | 30 s | 1.36 Hz | 3.45 | 3.72 | 3.73 | 3.74 |
| 1.0 Hz | 10 s | 1.75 Hz | 3.19 | 3.77 | 3.77 | 3.79 |
| 1.0 Hz | 3 s | 2.31 Hz | 3.22 | 3.64 | 3.68 | 3.79 |
| 2.0 Hz | 30 s | 2.71 Hz | **0.32** | 3.69 | 3.69 | 3.73 |
| 2.0 Hz | 10 s | 3.49 Hz | −0.64 | 3.53 | 3.66 | 3.78 |
| 2.0 Hz | 3 s | 4.63 Hz | −1.09 | **0.09** | 3.23 | 3.78 |
| 5.0 Hz | 30 s | 6.78 Hz | −4.83 | 2.16 | 3.55 | 3.84 |
| 5.0 Hz | 10 s | 8.73 Hz | −5.14 | −5.34 | **0.17** | 3.87 |
| 5.0 Hz | 3 s | 11.57 Hz | −5.10 | −6.00 | −6.64 | 3.77 |

Four things come out of it.

**It confirms the budget is on the instantaneous residual.** The
reference's cliff tracks `peak |f|` and nothing else: healthy through
1.75 Hz, wobbling at 2.31, gone at 2.71 — the same ~2 Hz as the ramp
table, at a tenth of the drift rate and with no total-excursion story
available. That is why the headline above is stated as a residual rather
than as an excursion.

**The gains that are right for a ramp under fading are wrong for fast
wander, and vice versa.** α=0.3 was the setting that cost 2.3 dB by
chasing `mpd` fading; here it is the setting that rescues 5 Hz RMS
(3.55 against α=0.1's 2.16 at τ=30, and 0.17 against −5.34 at τ=10).
The loop bandwidth has to sit above the drift's spectrum and below the
channel's Doppler spread, and those two can overlap. **A single
compiled-in pair of gains cannot serve both**, which is a stronger
argument for an adaptive bandwidth than the sinusoid alone made — but it
also cannot be settled without knowing which of the two is real on air.

**Everything a causal loop loses, the oracle keeps**: 3.73–3.87 dB in
every cell, including the ones where every tracker fails. So the ceiling
here is the *estimator*, not the waveform. A non-causal smoother over
the whole frame sequence — the pilots are all in memory by the time the
picture is reconstructed, and `demodulate` is not a real-time loop — has
that entire gap available to it, and would be immune to the bandwidth
tension above because it does not have to choose a bandwidth in advance.
That is the more interesting design if wander turns out to matter.

**A measurement trap, recorded because it cost a round.** The first OU
run showed the *oracle* scoring below the tracker, which is impossible
and was the tell. `acquire()` has already removed a constant, so the
oracle must remove the true path **minus `acq.freq_offset`**; removing
the raw path double-corrects by that constant. Invisible for a ramp
starting at zero — the preamble sits at t = 0.1 s, so `acq.freq_offset`
is ~0.01 Hz — and ruinous for wander, where the offset at the preamble
is a full standard deviation. An oracle that can be beaten is not an
oracle.

**4. Total excursion still has to stay inside the radio's filter.** At
10 Hz/s over mode A the signal moves 320 Hz, and at 40 Hz/s it moves
1281 Hz and even the oracle loses 1.9 dB. Below a couple of hundred Hz
of total movement this is not a consideration, which covers everything
in range of limit 2.

**5. A large *initial* offset is a separate problem** — that is the
mis-tuned-counterpart section above, and the two changes are
independent: the wide search finds a station that started off-frequency,
the loop follows one that does not stay there.

### What this would touch

Receiver-only, both halves. No on-air format change, no
`PROTOCOL_VERSION` bump, nothing an existing station would notice — a
tracking receiver and a non-tracking one decode the same waveform, one
of them just better. `demodulate()`'s frame loop is where it goes;
`demodulate_blind()` has pilots and the identical problem (and no
preamble reference, so it starts from a coarser CFO and needs it more),
and `sstvae/waveform_channel.py` does not model drift at all, so
training is unaffected either way. The C++ port would need the same
change in `native/core/modem/`, and `pytest --native` would then be
checking both.

### Reproducing all of the above

`scripts/freq_range_sweep.py --verify drift drift-acquisition
drift-fading drift-ou wander`. **Run `--verify` first and believe nothing without
it**: the tracker prototype is a *fork* of `Modem.demodulate`, and with
tracking off it is required to reproduce the reference bit for bit on
clean audio and under two fading presets. A fork that has quietly drifted
would be measuring itself.

### Caveats

2–6 seeds per cell, mode A except where mode C is named, and three drift
shapes (a linear ramp, a sinusoid, an OU process) none of which is a
measurement of a radio. In particular a rig warming up after key-down is
fastest at the start, which no stationary process models and which is
exactly when the loop has not converged. Nothing here has met a
real radio: `hfchannel` has no drift model at all, so `drift_shift`
was written for this measurement and lives in the sweep script rather
than in the package — deliberately, since a channel-simulator feature
should be added because a channel needs simulating, not because one
investigation wanted it.
**The first thing to do before implementing any of this is to find out
what real drift rates look like** — a recording of a few stations
keying up, measured with the same pilot-phase estimator, would settle in
an afternoon whether the reference's ~2 Hz budget is routinely exceeded
or whether this is a problem nobody has.

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

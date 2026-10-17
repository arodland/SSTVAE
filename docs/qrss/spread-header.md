# Spread header: the header on every data symbol, as extra phase

**Status: on the air as format version 2 (2026-10-10).** Andrew proposed
this as a prototype; Jeff adopted it, and spec rev 11 section 5.1 defines
it. Every preset with a header block (FULL, MEDIUM, SHORT) now carries
the spread copy at ρ = 0.054, in the Python transmitter, the beacon file's
C reference and the receiver. What changed when it went on the air:

- The frames this document calls `full-spread` and `short-spread` are
  now `full` and `short`. The old frames without the spread copy are
  `full-v1`, `medium-v1` and `short-v1` (`frame.LEGACY`), and the frame
  this document calls `short` is now `short-v1`. `full-spreadonly` is
  still a prototype (`frame.PROTOTYPES`).
- The header's version field is 2. Version 1 was sent only in tests.
  A receiver decodes a version 1 header from its block alone and then
  treats the pass as having no spread copy (`receiver.decode_header`).
- The C reference (`qrss_beacon_c/`) adds β·√ρ·h_i to each data symbol
  in its integer phase sum. Its whitener is a fourth SHA-256 sequence,
  and its parity tests (C1-C3) cover it.
- The checks this document left open have been run: acquisition with the
  copy on, header combining across passes, and fading paths. See
  "Checks before air" at the end.

## The problem

Every pass sends its header once, in an 80 s block just after the
preamble (design 2.2, 2.6). Everything that labels a pass needs it: the
callsign, picture ID, segment, mode and codec ID that associate it with
an accumulator. If that block is lost, the rest of the pass carries no
copy:

- a fade in the first 100 s;
- a receiver that starts listening late;
- a capture with a hole at the start.

The latents are still there, and association by latent correlation
(design 8.2, rule 3) can still place the pass, but only into an
accumulator that already exists. A first hearing of a picture whose
header block faded produces nothing usable.

## The scheme

Add the same polar codeword, repeated across the whole data section, as
a small extra phase on top of the latents:

    x_i  ->  x_i + sqrt(rho) * h_i,     h_i = w_i * (1 - 2 * c[i mod 2474])

- x_i is data symbol i, the precoded latent, unchanged with unit variance.
- c is the existing 2,474-bit coded header (`header.encode`, the same
  CA-polar code and INFO_SET).
- w is a whitening sequence, `pm1(sha_bits(b"QRSSTVAE CE spread header", n_data))`
  (`sequences.spread_whitener`). It is not keyed by q: the spread header
  is identical in every pass of a picture, so its LLRs add across passes
  exactly like the block's.
- rho is the header's power relative to a latent. The prototype's
  `frame.SPREAD_RHO = 0.054` gives the spread copy the block's energy on
  a FULL frame: 50,600 × 0.054 = 2,732 symbol-energies against the
  block's 2,474, about 10% more to pay for the latents acting as noise on
  it (below). That accounting left out the modulator's gain. A ±1 block
  symbol gets about 1.4 dB more out of CE than a small term on Gaussian
  data does, so the spread copy alone measures 1.8 dB short of the block
  (see "Why the spread copy is 1.8 dB short").

The phase is linear in the symbol values, so on the transmitter this is
one line in `frame.assemble`. Nothing else on the transmit side changes.

**On top, not instead.** The latents keep their full deviation β = 0.8;
the header adds variance, so the rms phase goes from 0.800 to 0.821 rad.
The point of doing it this way is specific to CE: **a known phase term
can be removed exactly.** Multiplying the received signal by
exp(−j·φ_h(t)) turns the spread frame into precisely the frame without
it. Nothing is lost once the header is known, which is the case for
every pass after the first decode of a picture's header. The costs are
all paid before that:

- the unknown header phase is interference on the data;
- the carrier is down by exp(−β²ρ) = 0.15 dB;
- the spectrum is slightly wider.

Two alternatives were considered and are worse:

- **Split.** Shrink the latents to make room (√(1−ρ)·x + √ρ·h, total
  deviation unchanged). This keeps the spectrum exactly, but loses
  0.08–0.09 dB of data once the header is known, at ρ = 0.054. That is
  cheaper than the 10·log10(1/(1−ρ)) = 0.24 dB the same split costs a
  linear waveform, because β = 0.8 sits near the β²e^(−β²) maximum.
- **Sign.** The header bit sets the sign of each symbol's phase, with the
  latents offset to stay on one side of zero. Keeping every precoded
  value on one side of zero takes 80–95% of the phase variance, which
  costs 5–10 dB. It also leaves a pass whose header fails with no usable
  data.

**Frames.** Two FULL prototypes, plus a SHORT one for tests:

| Frame | Header block | Spread copy | Duration |
|---|---|---|---|
| `full` | yes | no | 1782.7 s |
| `full-spread` | yes | ρ = 0.054 | 1782.7 s |
| `full-spreadonly` | no | ρ = 0.054 | 1702.7 s |
| `short-spread` | yes | ρ = 0.054 | 255.8 s |

`full-spreadonly` moves the callsign windows 2,640 symbols earlier with
the data. Dropping the block shortens the slot; it cannot buy image
quality, because the latent count is fixed.

## The receiver

The changes slot into the existing two rounds (design 6.8).

**Round A: header unknown.**

- *Template.* Every data symbol's template variance is 1 + ρ
  (`track.make_classes`, `Classes.spread_rho`), so the tracker's carrier
  template expects the slightly weaker carrier.
- *Data estimates.* `demod.data_estimates` divides the data estimates by
  the gain the unknown phase costs, exp(−β²ρ/2) = 0.983. It also adds ρ
  to every data symbol's variance, so the weights stay honest, and
  `calibrate` subtracts the same ρ from its data-based κ estimate.
- *Header LLRs.* `demod.spread_llr` computes
  2·√ρ·y_i / (σ²_i + 1) on every data symbol, with the latent's unit
  variance counted as noise beside the measurement's σ². It unwhitens,
  then folds the 50,600 values onto the 2,474 coded bits by summing.
- *Decode.* `extract` adds the block's LLRs and decodes the sum with the
  unchanged `header.decode`. `diag` keeps both halves (`llr_block`,
  `llr_spread`) separately. `PassResult.hdr_llr` holds the sum, so
  soft-header association across passes (design 8.2, rule 2) combines
  both copies with no change.

**Round B: header known.**

- *Phase removal.* `make_classes` with the header bits sets
  `Classes.spread` to the header's stream symbols. `track.track` then
  takes their phase off the channel before anything else, and again after
  every timing refinement and frequency re-centre (`track.remove_spread`).
  That phase is `ce.phase_at` of the spread symbols, at the sender time
  each 250 Hz sample was sent, from `rx_index` inverted.
- *Everything after.* The data templates, extraction and weights are
  those of a frame without a spread header.
- *Where else it applies.* The same path serves EM re-receives and
  template search with a known header (`em.py`), and `genie_track`. An EM
  re-receive keeps the pass's original `hdr_llr`, because after removal
  the data carry no header to read.

**Where `n_hdr` meant "has a header".** Those sites now ask
`spec.has_header` (block or spread): `tx.slot_symbols`, the simulator's
neighbours, `receive_pass`'s decode, and `em.py`.

## Measurements

All simulated, on a steady path with AWGN and Gaussian latents.

`scripts/qrss_spread_header.py` reproduces every number. Receiver trials
skip acquisition: each starts from a detection at the true carrier with
no timing, which gate V then fits. The header experiment receives each
pass once (round A) and decodes the same pass three ways: block LLRs
alone, spread LLRs alone, and both summed. That makes the three columns
a paired comparison.

The header runs are 40 seeds per point (run on the faster machine,
2026-10-10). A fraction of 0.5 has a 95% interval of about ±0.15; at
0.9 or 0.1 it is about ±0.1. The 50% thresholds below come from linear
interpolation between grid points 1.5 dB apart, so they are good to a
few tenths of a dB.

### Data cost (genie: exact synthesis and matched filter, MEDIUM frame)

`qrss_spread_header.py genie --rho 0.054 0.08 0.1` (the 0.2 rows are
from an earlier run with the same draws). These are paired draws: every
scheme sees the same latents. The figure is the change in per-latent SNR
against the plain frame, whose per-latent SNR is +13.4 / +6.2 / −3.0 /
−12.9 dB at the four noise levels:

| Scheme | ρ | Header known | Header unknown |
|---|---|---|---|
| on top (this design) | 0.054 | **0.000 / 0.000 / 0.000 / 0.000** | −3.74 / −1.10 / −0.27 / −0.16 |
| on top | 0.08 | 0.000 at every level | −4.82 / −1.56 / −0.40 / −0.23 |
| on top | 0.10 | 0.000 at every level | −5.52 / −1.88 / −0.50 / −0.29 |
| on top | 0.20 | 0.000 at every level | −7.99 / −3.27 / −0.98 / −0.59 |
| split | 0.054 | +0.49 / +0.01 / −0.08 / −0.09 | −3.65 / −1.13 / −0.36 / −0.25 |
| split | 0.08 | +0.73 / +0.01 / −0.13 / −0.14 | −4.79 / −1.63 / −0.53 / −0.38 |
| split | 0.10 | +0.92 / +0.01 / −0.16 / −0.18 | −5.55 / −2.00 / −0.67 / −0.48 |

On top is exactly free once the header is known.
`tests/test_qrss_spread.py::test_removing_the_known_phase_leaves_the_plain_frame`
pins the reason: the removal reproduces the plain frame's baseband to
1e-9.

The split's gain on noiseless passes is the lower deviation's lower
self-distortion. The "unknown" columns are the price of a pass extracted
before its header is known, and it shrinks at lower SNR.

### Data cost through the real receiver (SHORT, `short` vs `short-spread`)

`qrss_spread_header.py data --snr -12 -18 -24 -28 --seeds 24`. These are
paired seeds, both frames on the same noise. The figures are spread
minus plain, ± standard error:

| SNR₂₅₀₀ | n | Round B (header known), latent SNR | Round B, W | Round A (header unknown), latent SNR | Round A, W |
|---|---|---|---|---|---|
| −12 dB | 24 | +0.010 ± 0.006 | −0.005 ± 0.008 | −0.82 ± 0.01 | −0.78 ± 0.01 |
| −18 dB | 24 | +0.002 ± 0.010 | −0.009 ± 0.012 | −0.39 ± 0.04 | −0.36 ± 0.01 |
| −24 dB | 24 | +0.016 ± 0.012 | +0.003 ± 0.010 | −0.18 ± 0.02 | −0.21 ± 0.01 |
| −28 dB | 18 | −0.007 ± 0.023 | −0.016 ± 0.022 | −0.17 ± 0.02 | −0.17 ± 0.01 |

At −28 dB SHORT's header fails on 6 of the 24 seeds (5 in both frames,
1 in the plain frame only), and those seeds are left out of the pairs.

Round B costs nothing through the whole receiver: tracker, timing
refit, κ calibration and joint estimator included. Round A matches the
genie's "unknown" column at the corresponding per-latent SNR, and W
tracks the measured loss, so the extra ρ in the variance keeps the
weights honest. The round-A loss only applies to a pass whose header
never decodes, and to that pass only until the store knows the header
and an EM re-receive removes it.

### Spectrum

From the same genie run (Welch at 1 kHz, Gaussian latents; the absolute
widths differ from design M5's measurement method, the differences are
the point):

| ρ | rms phase | 99% width | 99.9% width | Density at 30 / 50 / 80 Hz |
|---|---|---|---|---|
| 0 | 0.800 rad | 57.6 Hz | 78.5 Hz | −30.3 / −50.5 / −75.9 dB |
| 0.054 | 0.821 rad | +0.6 Hz | +1.0 Hz | +0.2 / +0.6 / +0.4 dB |
| 0.08 | 0.831 rad | +0.8 Hz | +1.5 Hz | +0.3 / +0.8 / +0.6 dB |
| 0.10 | 0.839 rad | +1.1 Hz | +1.9 Hz | +0.4 / +1.0 / +0.7 dB |

At ρ = 0.054 this is inside M5's ±1.5 Hz width tolerance but uses about
two thirds of it at 99.9%. ρ = 0.08 uses all of it. The 50 Hz density
is the figure to check against the −28 ± 1.5 dB neighbour-leakage budget
(design M5).

### Header decoding (FULL, `full-spread`)

`qrss_spread_header.py header --frames full-spread --snr -25 -26.5 -28 -29.5 -31 --seeds 40`.
Decodes out of 40:

| SNR₂₅₀₀ | Block alone | Spread alone | Both |
|---|---|---|---|
| −25.0 dB | 40 | 39 | 40 |
| −26.5 dB | 39 | 27 | 40 |
| −28.0 dB | 33 | 1 | 40 |
| −29.5 dB | 5 | 0 | 37 |
| −31.0 dB | 0 | 0 | 7 |
| **50% threshold** | **−28.7 dB** | **−26.9 dB** | **−30.3 dB** |

- **Both copies together go 1.6 dB deeper than the block alone.** No
  trial anywhere decoded from one copy and failed on the sum.
- **The spread copy alone is 1.8 dB short of the block.** The LLR
  diagnostics below show this is physics, not calibration.

The block itself does better than the design's single-pass threshold
(−23.5 dB at P ≥ 50%, R17) on this steady path.

**LLR diagnostics.** These are per pass and per copy, medians over the
40 seeds. "SNR" is μ²/var of the LLR times the true sign. "Cons" is
var/2μ, which is 1 for a correctly scaled LLR, and k for an LLR scaled
by k:

| SNR₂₅₀₀ | Block SNR | Block cons | Spread SNR | Spread cons | Gap |
|---|---|---|---|---|---|
| −25.0 dB | −6.78 dB | 0.84 | −8.44 dB | 1.01 | 1.66 dB |
| −26.5 dB | −8.18 dB | 0.85 | −9.86 dB | 1.03 | 1.68 dB |
| −28.0 dB | −9.82 dB | 0.85 | −11.26 dB | 1.03 | 1.44 dB |
| −29.5 dB | −11.38 dB | 0.86 | −12.86 dB | 1.06 | 1.48 dB |
| −31.0 dB | −12.96 dB | 0.86 | −14.88 dB | 1.08 | 1.92 dB |

### Header decoding without a block (`full-spreadonly`)

`qrss_spread_header.py header --frames full-spreadonly --snr -25 -26.5 -28 --seeds 40`:

| SNR₂₅₀₀ | Decodes | Spread SNR (median) | Spread cons |
|---|---|---|---|
| −25.0 dB | 40 / 40 | −8.24 dB | 0.99 |
| −26.5 dB | 34 / 40 | −9.58 dB | 0.99 |
| −28.0 dB | 5 / 40 | −11.09 dB | 1.01 |
| **50% threshold** | **−27.2 dB** | | |

That is 1.5 dB worse than today's `full` frame (the block alone, above),
for an 80 s shorter slot. The spread copy here is about 0.25 dB better
than the same copy on `full-spread` (−9.62 against −9.85 dB mean at
−26.5, −11.10 against −11.36 at −28, standard errors about 0.08). Open
question 4.

### The header block lost (`--fade 120`)

The signal is taken 40 dB down for the first 120 s after t0. That
removes the preamble and the whole header block, which ends 100 s in.
Timing still comes from gate V on the references. Decodes out of 40:

| SNR₂₅₀₀ | Block alone | Spread alone | Both | Spread SNR (median) |
|---|---|---|---|---|
| −23.5 dB | 0 | 40 | 40 | −7.18 dB |
| −26.0 dB | 0 | 34 | 32 | −9.40 dB |
| −28.0 dB | 0 | 1 | 1 | −11.47 dB |
| **50% threshold** | none | **−26.9 dB** | | |

This is what the scheme is for. Today's frame yields no header at any
SNR here. With the spread copy, the header is recovered with at most
0.3 dB of loss against an intact pass's spread copy: losing 120 s of a
1700 s spread costs 10·log10(1700/1580) = 0.3 dB of its energy.

At −26 dB, 2 of the 40 trials decoded from the spread copy alone but
failed on the sum. The faded block's LLRs are not correctly scaled:
their consistency comes out at 1.8 to 2.6 (median), NaN on half the
trials, and they still add noise to the sum. Open question 3.

## Why the spread copy is 1.8 dB short: CE gain

This is resolved, and it is not calibration. The spread LLRs have
consistency 1.01–1.08, so they are correctly scaled. The block's are
0.84–0.86, so they are under-scaled. Read that as the receiver
under-estimating the block symbols' gain by 1/0.85 = 1.18.

A ±1 symbol comes through the phase modulator with more first-order
gain than a small term riding on Gaussian data. For an isolated symbol,
the first-order output is sin β = 0.717 for a ±1 symbol and
β·e^(−β²/2) = 0.581 for Gaussian data: a ratio of 1.23, or 1.8 dB.
Overlapping pulses give the measured 1.18 (1.4 dB).

That gain is what the energy accounting in "The scheme" left out.
Putting it back:

- take the block's measured LLR SNR as r²/σ², with r = 1/cons;
- predict the spread copy's SNR as ρ·(50,600 / 2,474) / (σ² + 1).

The prediction matches the measurement:

| SNR₂₅₀₀ | Predicted spread SNR | Measured |
|---|---|---|
| −25.0 dB | −8.43 dB | −8.44 dB |
| −28.0 dB | −11.08 dB | −11.26 dB |
| −29.5 dB | −12.51 dB | −12.86 dB |

The −31 dB point is 0.9 dB off, with 0 to 7 decodes and long-tailed
LLR statistics. The rule that falls out: **a header bit on top of
Gaussian data is worth about 1.4 dB less per unit of energy than a bit
in the block.** Any energy budget for a spread copy should start from
that.

The under-scaled block LLRs cost almost nothing on their own. In the
sum they under-weight the block by 0.85, which loses about 0.03 dB
against optimal combining at these SNRs. Fixing it means using the ±1
symbols' gain in the block's LLR. That is the same gain `calibrate`
already uses for the references.

## Open questions

1. **ρ.** Matching the block alone with the spread copy alone needs
   +1.7 dB, so ρ ≈ 0.08. That has now been measured. It is still free
   once the header is known. Round A's loss rises from −0.27 / −0.16 to
   −0.40 / −0.23 dB at the two lower SNRs, and the 99.9% width uses all
   of M5's ±1.5 Hz tolerance.

   Recommendation: keep 0.054. At 0.054, what the scheme is for already
   works:
   - the header survives a lost block 1.8 dB above the intact block's
     threshold, where today it does not survive at all;
   - on an intact pass the sum goes 1.6 dB deeper than today's frame.

   0.08 would buy that 1.8 dB back only in the lost-block case, at the
   cost of the whole width tolerance and half as much again of round-A
   loss. This is a judgement call, not a measurement.
2. **The combined gain is short of energy addition.** Adding the two
   LLR SNRs predicts 2.3 dB over the block alone; the thresholds give
   1.6. The block's mis-scaling accounts for 0.03 dB of that. The rest
   is not explained, and some of it may be interpolation on a 1.5 dB
   grid. A −29 / −30 / −30.5 dB sweep would place it.
3. **Faded block LLRs.** A block lost to a fade should contribute
   nothing, but its LLRs come out mis-scaled and occasionally break a
   decode the spread copy alone makes (2 of 40 at −26 dB). Candidates:
   - weight the block by its own measured SNR;
   - drop it when its consistency against the spread copy's hard
     decisions is off.
4. **`full-spreadonly`'s spread copy is about 0.25 dB better** than the
   same copy on `full-spread` (about 3σ). The data and ρ are the same;
   only the block's presence differs. Not explained.
5. **INFO_SET.** It was designed for the block's rate matching (2,474
   coded bits from N = 2,048 by circular repetition). Every coded bit
   gets ~20 repeats here, which preserves the relative pattern, so it is
   reused as-is. The folded per-bit SNR is no longer the design point,
   so a GA construction at the new operating point may gain a little.
6. **Fading paths.** This is where the scheme's time diversity should
   matter most, since the block's 80 s can sit in a single fade.
   Measured since, on the quiet and moderate presets: see "Checks before
   air".

## Commands

The raw results are in `docs/qrss/spread-header-data/` (one JSON row per
trial). These are what produced the tables above (each FULL trial is about
2.5 minutes of one core; `--jobs` defaults to every core). They are
written with today's frame names: `full` was `full-spread` when the
data were recorded, and the `data` run compares `short-v1` with `short`.

```sh
python scripts/qrss_spread_header.py header --frames full \
    --snr -25 -26.5 -28 -29.5 -31 --seeds 40 --out header.json
python scripts/qrss_spread_header.py header --frames full-spreadonly \
    --snr -25 -26.5 -28 --seeds 40 --out spreadonly.json
python scripts/qrss_spread_header.py header --fade 120 --snr -23.5 -26 -28 --seeds 40 --out fade.json
python scripts/qrss_spread_header.py data --snr -12 -18 -24 -28 --seeds 24 --out data.json
python scripts/qrss_spread_header.py genie --rho 0.054 0.08 0.1
```

## Before this could go on the air

These were the steps before air, and all but the last are done
(2026-10-10):

- A `FORMAT_VERSION` bump and a frame identity saying whether a pass
  carries the spread copy: version 2 means it does, on every frame with
  a header block. A receiver applying the wrong round-A model loses
  either the header or 0.2–0.8 dB.
- The beacon file and the Si5351 C reference (`qrss_beacon_c/`): the
  phase generator adds β·√ρ·h_i·p to each data symbol. The beacon file
  is unchanged; it already stores the coded header bits.
- `PRESETS`, the CLIs' `--frame` and the live listener: the presets carry
  the copy, so every CLI and the listener send and receive it.
- Design M5 re-measured with the new deviation: not yet.

## Checks before air (2026-10-10, build thread)

These are the three checks that spec rev 11 section 5.1 listed as open.
All are simulated, in round A, with FULL frames. "Both" means the block
and the spread copy summed.

**Acquisition.** Blind `receiver.detect` over the whole slot, 8 paired
seeds per SNR, `full-v1` against `full`:

| SNR₂₅₀₀ | Found, v1 / v2 | z_ref median, v1 / v2 | Header decoded, v1 / v2 |
|---|---|---|---|
| −28 dB | 8/8 / 8/8 | 19.73 / 19.95 | 6/8 / 8/8 |
| −30 dB | 8/8 / 8/8 | 15.18 / 15.12 | 1/8 / 5/8 |
| −32 dB | 8/8 / 8/8 | 9.89 / 9.65 | 0/8 / 0/8 |

Detection is unaffected and reaches 2 dB below the header's threshold
with both copies.

**Combining across two passes.** 12 passes per SNR, each starting at
the true carrier. Their round-A LLRs were summed over every pair:

| SNR₂₅₀₀ | One pass, both | Two passes, block | Two passes, both |
|---|---|---|---|
| −32.0 dB | 0/12 | 2/66 | 40/66 |
| −33.5 dB | 0/11 | 0/55 | 21/55 |

That puts the two-pass 50% point near −32.7 dB, about 2.4 dB beyond
one pass's −30.3 dB. The ideal is 3 dB, and the pairs are not
independent.

**Fading paths.** Watterson presets, 16 seeds, each pass starting at
the true carrier. 50% thresholds:

| Path | Block alone | Spread alone | Both |
|---|---|---|---|
| quiet (0.1 Hz, 0.5 ms) | −27.0 dB | −26.7 dB | −29.0 dB |
| moderate (0.5 Hz, 1 ms) | −25.2 dB | −24.5 dB | −26.9 dB |

- Both copies together beat the block alone by 2.0 dB on the quiet path
  and 1.7 dB on the moderate one.
- On the moderate path at −28 dB and below, gate V cannot fit the timing
  (14/16 and 16/16 passes not verified), so the tracker is the limit
  there, not the header.
- In none of the 200 trials across the three checks did the block alone
  decode where both copies together failed.

## Files

- `sstvae/qrss/frame.py`: `FrameSpec.hdr_rho` (default `SPREAD_RHO` with
  a header block), `has_header`, `spread_signs`, `spread_symbols`,
  `assemble`, `LEGACY`, `plain`, `PROTOTYPES`.
- `sstvae/qrss/constants.py`: `FORMAT_VERSION` = 2, `FORMAT_VERSIONS`,
  `SPREAD_RHO`; `header.py`: `decode(..., versions=)`.
- `sstvae/qrss/sequences.py`: `spread_whitener`.
- `sstvae/qrss/track.py`: `Classes.spread` / `spread_rho`, `make_classes`,
  `remove_spread`, and its calls in `track` and `genie_track`.
- `sstvae/qrss/demod.py`: `data_estimates`, `spread_llr`, `calibrate`,
  `extract`.
- `sstvae/qrss/receiver.py`: `decode_header` (both copies, then each
  alone, then version 1); `em.py`, `tx.py`, `channel.py`: `has_header`.
- `qrss_beacon_c/`: `qrss_ce_spread`, the spread term in
  `qrss_ce_phase_q22`, `QRSS_SPREAD_Q32` and the presets' `spread` flag
  (`tools/gen_qrss_tables.py`); `tests/test_qrss_c_ref.py` covers it.
- `tests/test_qrss_spread.py`: the format, the LLR fold, exact removal,
  the round-A classes; slow: a block-free frame decodes its header end to
  end and then matches the plain frame's latents to 0.1 dB (measured
  equal, 4.38 dB both, at −12 dB).
- `scripts/qrss_spread_header.py`: the measurements above.

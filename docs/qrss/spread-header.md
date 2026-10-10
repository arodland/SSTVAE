# Spread header: the header on every data symbol, as extra phase

**Status: prototype (2026-10-10).** Implemented behind `FrameSpec.hdr_rho`,
measured in simulation, not part of the on-air format: the prototype
frames live in `frame.PROTOTYPES`, not `frame.PRESETS`, so the C
reference, the beacon file and the CLIs' `--frame` choices are untouched.
Nothing here changes a frame with `hdr_rho = 0`.

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
  it (below).

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

The receiver runs were done on a laptop, 12 seeds per point, so a 0.75
is 9 of 12 with a 95% interval of roughly ±0.25. The `--seeds` and
`--snr` lists below are what to widen on a faster machine.

### Data cost (genie: exact synthesis and matched filter, MEDIUM frame)

`qrss_spread_header.py genie --rho 0.054 0.1 0.2`. These are paired draws:
every scheme sees the same latents. The figure is the change in
per-latent SNR against the plain frame, whose per-latent SNR is
+13.4 / +6.2 / −3.0 / −12.9 dB at the four noise levels:

| Scheme | ρ | Header known | Header unknown |
|---|---|---|---|
| on top (this design) | 0.054 | **0.000 / 0.000 / 0.000 / 0.000** | −3.74 / −1.10 / −0.27 / −0.16 |
| on top | 0.10 | 0.000 at every level | −5.52 / −1.88 / −0.50 / −0.29 |
| on top | 0.20 | 0.000 at every level | −7.99 / −3.27 / −0.98 / −0.59 |
| split | 0.054 | +0.49 / +0.01 / −0.08 / −0.09 | −3.65 / −1.13 / −0.36 / −0.25 |
| split | 0.10 | +0.92 / +0.01 / −0.16 / −0.18 | −5.55 / −2.00 / −0.67 / −0.48 |

On top is exactly free once the header is known.
`tests/test_qrss_spread.py::test_removing_the_known_phase_leaves_the_plain_frame`
pins the reason: the removal reproduces the plain frame's baseband to
1e-9.

The split's gain on noiseless passes is the lower deviation's lower
self-distortion. The "unknown" columns are the price of a pass extracted
before its header is known, and it shrinks at lower SNR.

### Data cost through the real receiver (SHORT, `short` vs `short-spread`)

`qrss_spread_header.py data --snr -12 -18 -24 --seeds 12`. These are
paired seeds, both frames on the same noise. The figures are spread
minus plain, ± standard error:

| SNR₂₅₀₀ | Round B (header known), latent SNR | Round B, W | Round A (header unknown), latent SNR | Round A, W |
|---|---|---|---|---|
| −12 dB | +0.015 ± 0.009 | −0.005 ± 0.011 | −0.81 ± 0.02 | −0.76 ± 0.02 |
| −18 dB | +0.007 ± 0.012 | −0.012 ± 0.015 | −0.35 ± 0.01 | −0.35 ± 0.02 |
| −24 dB | +0.005 ± 0.015 | −0.012 ± 0.014 | −0.20 ± 0.01 | −0.20 ± 0.01 |

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
| 0.10 | 0.839 rad | +1.1 Hz | +1.9 Hz | +0.4 / +1.0 / +0.7 dB |

At ρ = 0.054 this is inside M5's ±1.5 Hz width tolerance but uses about
a third of it. The +0.6 dB at 50 Hz is the figure to check against the
−28 ± 1.5 dB neighbour-leakage budget (design M5).

### Header decoding (FULL, `full-spread`)

`qrss_spread_header.py header --snr -22 -23.5 -25 -26.5 -28 --seeds 12`.
P(decode) per SNR (n = 12):

| SNR₂₅₀₀ | Block alone | Spread alone | Both |
|---|---|---|---|
| −22.0 dB | 1.00 | 1.00 | 1.00 |
| −23.5 dB | 1.00 | 1.00 | 1.00 |
| −25.0 dB | 1.00 | 1.00 | 1.00 |
| −26.5 dB | 1.00 | 0.75 | 1.00 |
| −28.0 dB | 0.67 | 0.00 | 1.00 |

Two findings, one expected and one not.

- **Both copies together decode 12/12 at −28 dB, where the block alone
  manages 8/12.** The combined threshold has not been reached at the
  lowest point run, so the gain is at least what this table shows. The
  sweep should go to −31 dB.
- **The spread copy alone is about 1.5–2 dB short of the block**, where
  the energy accounting above predicted parity. This is open; see "Open
  questions".

The block itself does better than the design's single-pass threshold
(−23.5 dB at P ≥ 50%, R17): 8/12 at −28 dB on this steady path.

### The header block lost (`--fade 120`)

The signal is taken 40 dB down for the first 120 s after t0. That
removes the preamble and the whole header block, which ends 100 s in.
Timing still comes from gate V on the references.

| SNR₂₅₀₀ | Block alone | Spread alone | Both |
|---|---|---|---|
| −20.0 dB | 0.00 | 1.00 | 1.00 |
| −23.5 dB | 0.00 | 1.00 | 1.00 |
| −26.0 dB | 0.00 | 0.75 | 0.75 |

This is what the scheme is for. Today's frame yields no header at any
SNR here. With the spread copy, the header is recovered at the same SNR
as from an intact pass: losing 120 s of a 1700 s spread costs about
0.3 dB of its energy.

## Open questions

1. **Why the spread copy is 1.5–2 dB worse than the block.** Energy says
   +0.43 dB for the spread copy. The latents' self-interference
   (σ² + 1 against σ²) costs about 0.45 dB at these SNRs, and the
   carrier 0.15 dB, which nets to roughly −0.2 dB, not −1.5 to −2.
   Candidates:
   - The two kinds of symbol differ in effective gain. The block's ±1
     symbols sit among ±1 neighbours, with a higher effective gain and
     lower distortion than a small term on Gaussian data; `demod.calibrate`
     already notes that ±1 symbols come through the modulator with a
     larger gain.
   - Round A's tracking is noisier on data with the extra variance.
   - The spread LLRs are mis-scaled. This matters beyond the spread-only
     decode, because a mis-scaled copy also weights the block+spread sum
     wrongly.

   The script now records, per pass and per copy, the LLRs' effective
   SNR (μ²/var of the LLR times the true sign) and their consistency
   (var/2μ, which is 1 for a correctly scaled LLR). The commands below
   split energy from calibration.
2. **The combined threshold.** Not reached at −28 dB; the sweep must go
   lower.
3. **ρ.** 0.054 was set by energy accounting. If the gap in (1) is
   physics rather than calibration, matching the block alone needs about
   ρ = 0.08–0.09. That costs nothing once the header is known, but
   roughly doubles round A's loss and the spectral widening. The choice
   should follow (1) and the neighbour-leakage check.
4. **INFO_SET.** It was designed for the block's rate matching (2,474
   coded bits from N = 2,048 by circular repetition). Every coded bit
   gets ~20 repeats here, which preserves the relative pattern, so it is
   reused as-is. The folded per-bit SNR is no longer the design point,
   so a GA construction at the new operating point may gain a little.
5. **Fading paths.** Not measured. This is where the scheme's time
   diversity should matter most, since the block's 80 s can sit in a
   single fade.

## Commands for a faster machine

Each FULL trial is about 1.5 minutes of one core. These use every core
by default (`--jobs`):

```sh
# header thresholds, block vs spread vs both, with LLR diagnostics (open question 1, 2)
python scripts/qrss_spread_header.py header --frames full-spread \
    --snr -25 -26.5 -28 -29.5 -31 --seeds 40 --out header.json
# the same with the block removed, for the frame that would drop it
python scripts/qrss_spread_header.py header --frames full-spreadonly \
    --snr -25 -26.5 -28 --seeds 40 --out spreadonly.json
# header block lost
python scripts/qrss_spread_header.py header --fade 120 --snr -23.5 -26 -28 --seeds 40 --out fade.json
# rho: rerun the above with a different frame, e.g.
#   FrameSpec("full-spread-09", hdr_rho=0.09) added to frame.PROTOTYPES
python scripts/qrss_spread_header.py data --snr -12 -18 -24 -28 --seeds 24 --out data.json
python scripts/qrss_spread_header.py genie --rho 0.054 0.08 0.1
```

## Before this could go on the air

- A `FORMAT_VERSION` bump and a flag (or a frame identity) saying whether
  a pass carries the spread copy. A receiver applying the wrong round-A
  model loses either the header or 0.2–0.8 dB.
- The beacon file and the Si5351 C reference (`qrss_beacon_c/`): the
  phase generator adds β·√ρ·h_i·p to each data symbol. The beacon file
  already stores the coded header bits.
- `PRESETS`, the CLIs' `--frame`, and the live listener.
- Design M5 re-measured with the new deviation.

## Files

- `sstvae/qrss/frame.py`: `FrameSpec.hdr_rho`, `has_header`,
  `spread_signs`, `spread_symbols`, `assemble`, `SPREAD_RHO`, `PROTOTYPES`.
- `sstvae/qrss/sequences.py`: `spread_whitener`.
- `sstvae/qrss/track.py`: `Classes.spread` / `spread_rho`, `make_classes`,
  `remove_spread`, and its calls in `track` and `genie_track`.
- `sstvae/qrss/demod.py`: `data_estimates`, `spread_llr`, `calibrate`,
  `extract`.
- `sstvae/qrss/receiver.py`, `em.py`, `tx.py`, `channel.py`: `has_header`.
- `tests/test_qrss_spread.py`: the format, the LLR fold, exact removal,
  the round-A classes; slow: a block-free frame decodes its header end to
  end and then matches the plain frame's latents to 0.1 dB (measured
  equal, 4.38 dB both, at −12 dB).
- `scripts/qrss_spread_header.py`: the measurements above.

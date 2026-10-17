# Waveform L: what it adds to CE, what CE taught us, and the cost of hearing both

2026-10-11, Build QRSSTVAE. QRSSTVAE sends SSTVAE's v5 picture latents as
analog values on one narrow carrier for about 30 minutes per pass, and
receivers average repeated passes until a picture appears far below the
noise. It has two waveforms. **CE** (on the air since 2026-10-09) is a
constant-amplitude carrier whose phase carries one latent per symbol,
about 50 Hz wide, which an Si5351 beacon can make directly. **L** is a
linear signal for SSB rigs: one carrier carrying two latents per complex
symbol plus a pilot tone, about 22 Hz wide. The spec
(`qrsstvae-options.md` rev 13, sections 2 to 2.2, 2.7, 2.8, 5.1 and 6) defines L;
this note records what building it settled, what it is for, and what it
costs a receiver to listen for both.

**Status.** The transmitter is built and checked against the spec
(`sstvae/qrss/lframe.py`, `lwave.py`, `qrss_transmit.py --waveform L`),
and since 2026-10-12 the receiver is built too (`acquire.py`, `ltrack.py`,
`ldemod.py`, `receiver.py`, `live.py`): `qrss_receive.py` and the live
listener `qrss_listen.py` hear L beside CE by default, and `--just-ce` or
`--just-l` picks one. The C++ port (`sstvae-qrss`) does both, and the
desktop app sends L by default. **L has been on the air since
2026-10-12** (the first pass was mode A at 14.066 MHz). A listener run
with `--just-ce` (the old CE receiver, and with `--old-edge-noise
--old-gating` bit for bit the one of 2026-10-11) does not know L: it
shows an L signal at most as a weak, headerless CE tile (below,
"Keeping L out of the CE gate"). The measurements in this note come
from the channel simulator, not from on-air passes.

## What L gives over CE

| | L | CE |
|---|---|---|
| 99% width | **21.5 Hz** (measured, data and pilot) | 51.5 Hz |
| Signals in 2.4 kHz | about 80 at 30 Hz spacing | about 48 at 50 Hz |
| Chance two random signals overlap | 1.8% | 4% |
| Power into a neighbour during the data | **-52 dB** (an L signal 30 Hz away, measured) | -28 dB into a CE 50 Hz away; -26 dB into an L 40 Hz from the carrier |
| Into a neighbour during the callsign windows | -28 dB (30 Hz above), -39 dB (below) | -22 dB (OOK) |
| Good picture, one pass, quiet path, **same average power** | **-14.0 dB** | -10.9 dB (3.2 dB worse) |
| The same at the same peak power | -10.8 dB | -10.9 dB (a tie) |
| Morse ID | FSK of an audio tone through an SSB rig (J2A, plain CW in US rules) | FSK of the carrier (F1A, not listed as CW; OOK is the fallback) |
| Receiver work per detected signal | about the same as CE's (measured on a FULL frame: tracking 7.25 s against 6.33 s, gate V 10.9 s against 13.3 s) | |
| Transmitter | any SSB rig and sound card, linear amplifier | also an Si5351 board, class C/D/E amplifiers |

- **Density and cleanliness.** L is 2.4 times narrower, and its clipper's
  correction is filtered to the occupied band, so outside the callsign
  windows it puts essentially nothing into its neighbours. CE's phase
  modulation has skirts it cannot filter (a filter would break the
  constant envelope). On a crowded band, or where uncoordinated stations
  land at random, that is L's main gain.
- **A rig limited by heat, not peak power.** CE is key-down at constant
  power for 30 minutes, so an SSB rig runs it well below its rating (our
  IC-7100 runs 25 to 50 W). L averages 3.2 dB below its peak, so the same
  rig at its full PEP runs at the same average power as CE at half power,
  and at the same average power L needs **3.2 dB less SNR** for the same
  picture. Where the limit is the rig's or the amplifier's heat, that is a
  3 dB gain on every pass. Where the limit is peak power (a legal PEP limit,
  as on 60 m and 630 m, or a rig run at its PEP), they tie for good
  pictures.
- **Station ID** through an SSB rig is plain CW under US rules.
- **The pilot is a QRSS trace** at the signal's top edge, and the data sit
  entirely below it, so the spectrum's shape tells a receiver it is L (used
  below).

**What CE keeps.** Constant envelope (Si5351 beacons, saturated
amplifiers, 630 m). A carrier 9 dB stronger than L's pilot at the same peak
power, which tracks disturbed paths better (0.6 to 2.5 dB at the usable and
recognizable levels after 16 passes), tolerates oscillator wander twice as
well, and is detected blind down to about -38 dB against L's -34 dB
(average power) or -31 dB (same peak); this receiver's blind search finds
L only to about -30 dB so far ("What is left", below). And CE's header
threshold is measured; L's is not yet (an l-short header decoded at
-28 dB on a steady path, one seed).

**Which to send.** L from an SSB rig on a busy band, or whenever the rig is
held back by heat; CE from a beacon, on 630 m, under a peak-power limit
with deep averaging in mind, or on disturbed paths. Passes of both add into
the same picture (same latents, scrambler, precoder and header bits).

## What the implementation pins down

The spec leaves these open; they are now fixed in code and are offered to
"Decoder SNR measurement" for rev 13.

- **The same frame as CE in time.** 330 + 1,320 + 26,987 = 28,637 stream
  symbols, windows after 6,944, 14,175, 21,406 and 28,637 of them (half of
  CE's counts), 1,782.678 s. Window edges fall within 15 ms (half a CE
  symbol) of CE's. One Morse unit is one L symbol.
- **The data section starts mid-period.** 1,320 header symbols are 82.5
  reference periods, so the first data reference is at j = 1,328, not at
  the first data symbol (`lframe._data_symbols`). The spec's counts (83 and
  1,687 references, 25,300 data symbols) come out right.
- **The pilot is exactly 12207/970 Hz (12.5845) above the data centre**:
  the data band's edge (1 + α)·Rs/2 = 920/97 Hz plus 3.1 Hz. A rational keeps
  all mixing in exact integer turns. **The carrier a sender names is the
  pilot**, so `--freq 1500` puts the pilot at 1500 Hz and the data at
  1478 to 1497 Hz.
- **Levels.** Symbols at unit power, pilot amplitude √(3/17) (15% before
  clipping), clip threshold √(20/17) (0 dB above the nominal RMS envelope,
  fixed, so it does not depend on the picture). The clipped signal's
  average power is **0.63** of that scale (0.628 to 0.634 on 21 real
  segments), and the lead-in and the callsign windows send the pilot at
  that average power (`lwave.L_AVG_POWER`). The output is scaled to unit
  average power.
- **The clipper filters only its own correction**: z ← z + F{(min(A/|z|, 1)^k
  - 1)·z}, k = 1, 1.5, 2, F a filter over [-f_p - EDGE - 1 Hz, +1 Hz] with
  1 Hz raised-cosine edges, run at 250 Hz over the whole frame, then
  interpolated to 8 kHz. Filtering the whole signal instead (as
  `sims/clipstudy.py` does) would also filter the windows' FSK, which
  reaches 4.6 Hz above the pilot, and the pilot's keying ramps. Inside the
  band the two forms are the same thing.
- **The pilot's level changes** are raised-cosine ramps over units 0 to 7
  and 184 to 191 of each window (where the spec puts plain pilot) and over
  the preamble's first 8 symbols after a lead-in. The data pulses on either
  side are left untouched (the ramp is added to the pilot, not a
  crossfade).
- **The spread header copy** (Jeff chose it, 2026-10-11) scales the
  latents to make room: latent i is sent as (x_i + √ρ·h_i)/√(1 + ρ), with
  CE's whitener and coded bits indexed by latent, so an L pass's
  spread-copy soft bits add to a CE pass's. It costs 10·log10(1.054) =
  0.23 dB on every latent.

**Measured** (genie loopback: true timing, no noise; `lwave.loopback_stats`):

| | Spec 2.2 (Gaussian) | Gaussian latents | Real v5 latents, 7 pictures, 21 segments |
|---|---|---|---|
| PAPR (99.99%) | 3.2 dB | 3.14 to 3.15 dB | 3.15 dB (3.12 to 3.18), 3.21 dB at 8 kHz |
| Latent SINR, one pass | 12.3 dB | 12.24 to 12.33 dB | 12.2 dB (12.0 to 12.4) |
| Data gain against the references | 0.88 | 0.883 to 0.890 | 0.881 (0.875 to 0.888) |
| Pilot share after clipping | 13.8% | 14.0% | 13.9% |
| Power split: data / pilot and references / distortion | 73.9 / 20.1 / 6% | 73.5 / 20.4 / 6.2% | |
| Distortion that never averages (16 passes, precoder) | 22.2 dB | | 22.5 dB (20.5 to 24.2; screenshots lowest) |

The 8 kHz audio through a genie receiver gives the same symbols as the
250 Hz loopback to 1e-3, and a full pass takes 2.5 s of CPU to write (CE:
2.2 s).

## What the receiver settled

Built 2026-10-12, and measured in simulation.

- **One signal, one identity.** An L signal's frequency is its pilot
  everywhere (detections, passes, tiles), and its data run from 22.07 to
  3.10 Hz below it. Signals are compared at their band centre, f for CE
  and f - 9.5 Hz for L, so a CE and an L reading of one transmission
  within MERGE_HZ (5 Hz) are one signal, and the larger Z_ref says which
  waveform it is. The same centre places an L neighbour in another
  signal's noise floor: the usual 45 Hz guard about it covers 54.5 Hz
  below its pilot to 35.5 Hz above.
- **Acquisition.** A1 is shared. A2 runs both preambles on every A1 line,
  and L's also at each line's strongest cell 3.1 to 22.1 Hz above it,
  because a lump of data band can win A1's 10 Hz separation and hide the
  pilot. L's preamble is correlated in ten 33-symbol chunks, so CE's
  calibration and pass mark carry over. Given the right line, A2-L passed
  8 of 8 l-short preambles at -25, -28 and -31 dB.
- **The tracker** (`ltrack.py`) follows one complex gain, the data
  band's, fed by two measurement streams: the pilot, and the data band's
  known symbols. A complex pilot/data ratio is fitted from the references
  (and held at 1 in gate V, so Z_ref stays N(0,1) on noise). On l-short
  at -12 dB its first round came within 0.07 to 0.17 dB of the genie on a
  steady path, 0.29 to 0.46 dB on a quiet one and 1.2 to 1.8 dB on
  moderate fading, where the loss is noise in the pilot track (the pilot
  is 13.8% of the power).
- **The demodulator** (`ldemod.py`) divides by the spec's 0.881 data
  gain; 24 noiseless loopbacks measured 0.8846 ± 0.0029, which includes
  it. Its kappa allows for the references pulling the gain track, which
  they do far more in L than in CE (the pilot carries less): with CE's
  rule W read 11 to 14% too large on average over 24 to 36 passes, with
  the allowance 0 to 2%.
- **The header** is decoded as format version 2 only (L has no version
  1): from both copies, then the spread copy alone, then the block alone.
  How often the fallbacks rescue a header the sum missed is not measured.
- **Hints** are frequencies or (frequency, waveform) pairs, and a stored
  pass's frame name ("l-..." for L) decides its waveform.

**Measured end to end** (simulated, steady path, SNR in 2500 Hz at L's
average power; latent SNR is against the true latents, and the genie is
the same pass received at the true timing and gain):

| Pass | Latent SNR | Genie | Header, callsign |
|---|---|---|---|
| l-tiny, -12 dB, blind, 4 seeds | 6.46 to 7.73 dB | 6.73 to 7.31 dB | (TINY has neither) |
| l-short, -6 dB, 2 seeds | 9.78, 10.16 dB | 9.92, 10.06 dB | header; callsign read |
| l-short, -12 dB, 3 seeds | 6.71 to 7.04 dB | 6.91 to 6.97 dB | header; callsign read |
| l-short beside a CE signal 60 Hz up, both -12 dB, 2 seeds | 6.67, 6.98 dB | 6.75, 6.98 dB | header; the CE signal received too |
| l-short, -24 dB, 2 seeds | -4.10, -3.63 dB | -3.72, -3.62 dB | header |
| l-short, -28 dB, 1 seed | -7.50 dB | -7.57 dB | header |
| CE short, -12 dB, for comparison, 4 runs | 4.27 to 4.68 dB | | header |
| Noise only (short and tiny) | nothing found, of either waveform | | |

A live listener fed one slot as audio, a CE signal at 1500 Hz and an L
pilot at 1580 Hz, both at -12 dB, gave CE 4.71 and L 6.80 dB on TINY, and
on SHORT CE 4.68 and L 6.57 dB with both headers and both callsigns.

## Lessons from running CE live, applied to L

1. **A stronger neighbour must stay out of the noise estimate** (3c76b3a:
   KC2G missed live beside AG7EW 100 Hz away and 14 dB stronger). L packs
   signals 30 Hz apart and its band is one-sided, so the noise floor should
   come from bins no detected signal of either waveform occupies, each
   neighbour taken with its own band (CE ±25 Hz, L 22 Hz below its pilot).
   Built more simply: an L neighbour is left out about its band centre,
   9.5 Hz below its pilot, with CE's symmetric 45 Hz guard.
2. **A1's 16 lines and 10 Hz separation are too few for L.** A busy band
   already crowds a weak part 1 out of the 16 strongest lines; 80 L
   positions make that common. A2 costs 0.05 s a line (measured below), so
   64 lines cost 3 s a slot. Not built that way: A1 still gives 16 lines,
   and each line's strongest cell 3.1 to 22.1 Hz above it is searched as
   an L pilot as well. Live L acquisition stops at about -26 dB because of
   A1 ("What is left", below).
3. **Hints, re-checks and kept acquisitions** (103dcb3, 63c7eb1) carry
   over as they are, plus the waveform: part k+1 is searched as part k's
   waveform first. Built: a hint is a (frequency, waveform) pair.
4. **Headers get lost** to late joins, restarts and fades. L's header is
   estimated at -27 dB, right at its first recognizable picture (-27.8 dB),
   which is why L carries the spread copy too.
5. **Timing moves between passes** (KC2G 0.72 to 0.82-0.90 s), so no
   timing prior across passes, for L as for CE.
6. **Digital silence, dial changes, one transmission one pass** are
   front-end and store rules and need nothing new.
7. **Aliases, ghosts and merges are per waveform.** CE's symbol-rate
   aliases sit at ±33 Hz; L's would sit at ±16.5 Hz, inside its own data
   band. MERGE_HZ (5 Hz) still holds across waveforms: two signals that
   close overlap whatever they are. Built: aliases per waveform at its own
   symbol time, merges at band centres. L's ghost rule is CE's with its
   timing window in L symbols, and unmeasured.
8. **The tracker is the CPU** (25 s per signal per live refresh, mostly the
   pure-Python Kalman loop over every symbol position). L has half the
   positions, so about half the work per signal, this note expected.
   Measured, it is about the same: L's tracker reads two streams, and its
   first round on a FULL frame took 7.25 s against CE's 6.33 s (gate V
   10.9 s against 13.3 s).
9. **The rig.** L's peaks are 3.2 dB above its average, so the drive must
   be set by the peaks with ALC not engaged; the app's transmit level
   advice needs an L line (the UI thread's).

## The cost of decoding CE and L in one window

Measured on a FULL slot of noise in the cloud (CPU seconds, one thread),
with the gate's costs from the listener's own measurements:

| Stage | What it does | Shared? | Extra cost for both |
|---|---|---|---|
| Front end, passband store | 4 kHz complex, 48 h FLAC | yes | none |
| A1 carrier lines (0.02 s a slot) | strongest lines in the first 25 s | yes: a CE carrier and an L pilot are both lines | none |
| A2 preamble search (0.05 s a line) | correlates each line with a preamble | no | one more search per line: **0.8 s a slot** at 16 lines, 3 s at 64 |
| A3 track-before-detect (2 s a slot) | follows lines over the whole slot | yes | none |
| Gate V on a preamble hit (about 17 s) | tracker over the frame | the preamble says which waveform | none |
| Gate V on a blind candidate (17 s if real, 60 to 80 s if not) | A3 lines, refused hints | **no: the waveform is unknown** | up to one more gate per candidate |
| Receiving a signal (25 s a live refresh) | tracker, EM, decoder | each its own waveform | none; L about the same as CE (measured, above) |

**Everything is shared or nearly free except the blind gate.** A preamble
hit names its waveform, so live reception costs under a second more per
slot. A blind candidate does not, and trying both waveforms doubles the
expensive case: a line that is not a signal costs 60 to 80 s per waveform,
inside the end-of-slot search's 180 s and 12-candidate budget. Doubling it
halves how many blind candidates a busy slot gets through.

**Measured with the receiver** (simulated, on a shared 4-CPU machine): a
joint acquisition costs about twice a single one (TINY 0.8 s against
0.42 s, SHORT 2 to 3 s against 1.1 to 1.3 s; A2-L takes 27 to 31 ms a
line, 18 ms of it the Gamma tail probability). A whole receive costs more
than that suggests: `qrss_receive.py` on a CE SHORT recording took 12.6 s
listening for both against 7.7 s with `--just-ce`, the difference being
L's acquisition and the gates it adds; since the gating rule of
2026-10-12 (below), 7.9 to 8.1 s against 6.8 s. The CE re-check below costs about
0.09 s per CE accept on TINY.

**The spectrum's shape removes most of that.** L's data sit entirely below
its pilot; CE's sidebands are symmetric about its carrier. A3's
spectrogram already holds the slot, so comparing the power 3 to 22 Hz
below a line with the power 3 to 22 Hz above it costs nothing. For an L
signal at its blind-detection floor (-34 dB), the data band is about
-14.6 dB in its 21.5 Hz, and over 1,782 s that asymmetry is about 5 sigma:
L is recognised wherever it can be detected blind at all. A weak CE
signal shows almost nothing either side (its sidebands are under 1 sigma
at -38 dB), so "no band below" says CE or unknown. Gate V then runs once
for a recognised L line and for a strong CE line, and twice only for the
weak ambiguous ones, CE first since only CE reaches below -34 dB. Estimated:
blind-search work grows by 10 to 20%, not 100%. That figure is still
arithmetic; the test itself is now measured (simulated):

- The statistic, z1 = (power below - power above) over its noise sd, is
  N(0,1) on noise: mean +0.01, sd 1.03 on SHORT bands, and mean +0.03,
  sd 1.00 along A3's paths.
- On SHORT at -15, -20 and -25 dB an L signal reads z1 +155, +50 and +16,
  and a CE signal |z1| 2.3 or less. A CE signal at -25 dB is ambiguous in
  2 of 3 passes, and is gated as CE and then as L.
- Along L_FULL's true path z1 is 21.5, 13.3 and 5.0 at -28, -30 and
  -34 dB, so the shape would still say L at -34 dB. A3 itself finds
  L_FULL only at -28 and -30 dB, though: its pilot-only search is about
  6 dB behind CE's.
- As built, a line is L when z1 > 4 and the power above it is not
  significant (z 3 or less: under a strong CE signal the noise cross-terms
  widen z1, which passed 4 in 2 of 8 SHORT passes at -10 dB); CE when the
  power above is significant (z over 3) and the two sides are alike
  (|z1| < 3); and ambiguous otherwise.

**False alarms do not grow measurably.** Two preamble searches double A2's
cells (1e-7 each), and a candidate tried as both waveforms takes two
chances at the gate, but the gate's Z > 6 is what decides, and it does not
pass noise. On the simulated noise-only slots tried (short and tiny),
neither waveform found anything. L's blind gate V on noise reads higher
than CE's (20 seeds: mean 0.5, sd 1.2, highest 3.2), and how often it
would pass 6 is not measured.

**Header soft bits combine across waveforms.** L's QPSK header bits and
CE's ±1 bits are the same 2,474 coded bits, and with the spread copy both
use the same whitener indexed by latent, so their LLRs add. In a genie
test an L pass and a CE pass that each failed alone decoded together.

### Keeping L out of the CE gate

The listener from before L (on Hamlet, and the qrss-live decoding Andrew
runs from the Utah WebSDR) does not know L. This note expected it to
refuse an L signal at the CE gate, 60 to 80 s each, so that three L
signals would use the whole 180 s budget. Measured, it does something
else: CE's gate V passes an L signal (Z_ref 6 to 11 on l-short at -12 and
-6 dB), because L's references correlate with CE's in the data band. So
the old listener shows an L signal as a weak, headerless CE tile with
nothing usable in it. Simulated, at -12 dB: a live TINY slot gave a junk
CE tile at z 7.3 and latent SNR -8.0 dB, a live SHORT slot one at z 7.2,
and `qrss_receive.py` on a recording a SUSPECT CE pass at Z_ref 6.5.

Listening for both keeps L out of the CE gate three ways. L's own
preamble names the waveform; the shape test names it for an A3 line; and
a CE accept is checked again against the residual its known symbols leave
under the noise model (the residual ratio R, about 1 for a real CE pass).
An accept with R > 2 and Z_ref/√R ≤ 6 is refused. Measured R: CE's own
passes 0.93 to 1.48 on TINY, fading CE up to 2.16 (at Z_ref 48, so kept),
an L signal under CE's gate 3.6 to 15.4, and CE's own sideband ghosts
about 12 Hz off its carrier 5.2 to 6.2 (refused, which is no loss).
Since 2026-10-12 those ghosts, and the L copy of a strong CE carrier,
are not gated at all when the CE signal was found on its preamble: a
whole-slot line inside a preamble-found signal's own band (20 Hz either
side of a CE carrier, 22.1 Hz below to 3.1 Hz above an L pilot), no
stronger in A3 than that signal's own line, is skipped, which took the
search on SHORT, CE beside an L signal 60 Hz up, from 19 to 25 s of CPU
to 7 to 8 (`--old-gating` sets it back). A stronger line is still
gated: a station whose preamble faded can outdo one whose preamble was
heard.

`--just-ce` is, by decision, the CE receiver from before L: it makes
none of the joint receiver's calls and has no CE re-check, so an L
signal reaches it as a CE candidate. It does take both 2026-10-12 rules;
with `--old-edge-noise --old-gating` as well it is the listener of
2026-10-11 bit for bit.

## Plan: what was done and what is left

1. **Done:** the L transmitter, frame and tests (`tests/test_qrss_lwave.py`).
   `qrss_transmit.py --waveform L` writes a pass, and the app's schedule
   window can send L (a Waveform choice beside Mode).
2. **Done (2026-10-12):** recognise L in the listener and keep it out of
   the CE gate (above), whenever both waveforms are listened for.
3. **Done (2026-10-12), but for EM:** the L receiver (A2 with the L
   preamble, the tracker on the pilot and the references, demodulation by
   I/Q pairs divided by the 0.881 data gain, the header from QPSK soft bits
   plus the spread copy, tracking on the spread copy as well once the
   header is known, and the Morse reader with one unit per L symbol), and
   the joint listener (both A2s per line, the shape test, per-waveform
   aliases, signals compared at band centres, and the waveform recorded
   per pass and per tile).
4. **Partly done:** on a steady path L's latent SNR is within about 0.4 dB
   of the genie from -6 to -28 dB, and its header decoded at -28 dB (one
   seed). Spec 6.3's thresholds on Watterson paths, and L's header
   threshold, are not measured.
5. **Done:** the app's waveform choice for scheduled sends, and the QRSS
   window reading a tile's `waveform`: an L tile says so, and its marker
   covers its band, from about 22 Hz below its pilot to 3 Hz above it.

**What is left**, in the receiver:

- **An L-shaped A1 statistic.** A1 sees only the pilot, 10.5% of the
  preamble's power, so live acquisition of L stops at about -26 dB:
  simulated on l-short, 6 of 8 at -24 dB, 2 of 8 at -28 and 0 of 8 at
  -31, where A2-L given the right line passes 8 of 8 at -31 and the spec
  says -31 dB. A weaker L signal waits for the end-of-slot search. The
  fix is estimated at 2 to 3 dB.
- **A3 for L**, which finds L_FULL at -28 and -30 dB but not at -34,
  where CE's is found: an L-matched search needs its own Gumbel table.
- **EM, the template search and retro detection for L.** All three are
  CE-only. An L member of an accumulator joins every template, but is
  never re-received or searched for.
- **The L ghost rule, measured.** L's references recur every 16 L
  symbols, so its ghosts would sit at multiples of about 1.03 Hz; none
  appeared in any run, and the rule in use is CE's with its timing window
  in L symbols.
- **Recovery per waveform.** The listener records finished slots by slot
  only, so a slot finished under `--just-ce` is not received again for L
  when the listener restarts listening for both.
- **A strong L signal and the front end's normaliser.** Above about -6 dB
  SNR2500 the 100 ms inverse AGC follows L's envelope (an l-short genie at
  +10 dB read 12.13 dB of latent SNR before it and 9.89 dB after), and a
  strong L signal modulates the others in the band.
- **Done (2026-10-12): the noise floor near the band edges**, which
  also affected CE. Within about 75 Hz of 300 or 2700 Hz a -12 dB signal
  read several dB too strong, and a CE neighbour close by made an L pass
  there much worse. The floor is now measured only where the front end
  is flat (`--old-edge-noise` sets it back). Simulated at -12 dB (TINY,
  both waveforms): an L pilot at 324 to 370 or 2650 to 2690 Hz reported
  -3.9 to -9.5 dB with kappa at its clamp of 2, now -12.1 to -12.6. When
  a neighbour covers the only flat side, the floor is halfway in dB
  between the flat side (too high, by the neighbour) and the skirt
  compensated for the front end (too low, by however much more the
  radio's filter cuts): at 326 Hz beside CE at 400 the pilot reported
  +8.0 dB, suspect, latent SNR -0.24 dB, now -9.0 dB and 5.88 dB; at
  2695 Hz beside CE at 2600, +23.8 dB, suspect, 2.98 dB, now -6.6 dB and
  6.87 dB.

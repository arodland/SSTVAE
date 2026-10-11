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
(`sstvae/qrss/lframe.py`, `lwave.py`, `qrss_transmit.py --waveform L`).
No receiver decodes L yet, and today's listener would waste its blind
search on L signals, so **L should not go on the air until the listener at
least recognises it** (below, "Before L goes on the air").

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
| Receiver work per detected signal | about half of CE's (half the symbol positions; estimated) | |
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
(average power) or -31 dB (same peak). And CE's header is measured; L's is
an estimate.

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

## Lessons from running CE live, applied to L

1. **A stronger neighbour must stay out of the noise estimate** (3c76b3a:
   KC2G missed live beside AG7EW 100 Hz away and 14 dB stronger). L packs
   signals 30 Hz apart and its band is one-sided, so the noise floor should
   come from bins no detected signal of either waveform occupies, each
   neighbour taken with its own band (CE ±25 Hz, L 22 Hz below its pilot).
2. **A1's 16 lines and 10 Hz separation are too few for L.** A busy band
   already crowds a weak part 1 out of the 16 strongest lines; 80 L
   positions make that common. A2 costs 0.05 s a line (measured below), so
   64 lines cost 3 s a slot.
3. **Hints, re-checks and kept acquisitions** (103dcb3, 63c7eb1) carry
   over as they are, plus the waveform: part k+1 is searched as part k's
   waveform first.
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
   close overlap whatever they are.
8. **The tracker is the CPU** (25 s per signal per live refresh, mostly the
   pure-Python Kalman loop over every symbol position). L has half the
   positions, so about half the work per signal.
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
| Receiving a signal (25 s a live refresh) | tracker, EM, decoder | each its own waveform | none; L about half of CE |

**Everything is shared or nearly free except the blind gate.** A preamble
hit names its waveform, so live reception costs under a second more per
slot. A blind candidate does not, and trying both waveforms doubles the
expensive case: a line that is not a signal costs 60 to 80 s per waveform,
inside the end-of-slot search's 180 s and 12-candidate budget. Doubling it
halves how many blind candidates a busy slot gets through.

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
blind-search work grows by 10 to 20%, not 100%. This is arithmetic, not a
measurement; it gets measured when the receiver exists.

**False alarms do not grow measurably.** Two preamble searches double A2's
cells (1e-7 each), and a candidate tried as both waveforms takes two
chances at the gate, but the gate's Z > 6 is what decides, and it does not
pass noise.

**Header soft bits combine across waveforms.** L's QPSK header bits and
CE's ±1 bits are the same 2,474 coded bits, and with the spread copy both
use the same whitener indexed by latent, so their LLRs add.

### Before L goes on the air

Today's listener (on Hamlet, and the qrss-live decoding Andrew runs from
the Utah WebSDR) does not know L. It would see an L pilot as an A1 line,
find no CE preamble there (cheap), then pick the pilot up again in A3 at
the end of the slot and spend a refused CE gate on it, 60 to 80 s each. Three
L signals on the band use the whole 180 s budget, and the weak blind CE
candidates behind them go unchecked that slot. So the first receiver
change, before any L transmission, is to recognise an L line by its shape
and leave it out of the CE gate. That is small and can ship ahead of the
L receiver.

## Plan

1. **Done:** the L transmitter, frame and tests (`tests/test_qrss_lwave.py`).
   `qrss_transmit.py --waveform L` writes a pass; the app runs that script,
   so a Send can carry L once the UI offers the choice.
2. **Recognise L in the listener** and keep it out of the CE gate (above).
3. **The L receiver:** A2 with the L preamble, the tracker on the pilot and
   the references, demodulation by I/Q pairs (divided by the 0.881 data gain),
   EM, the header from QPSK soft bits plus the spread copy, tracking on the
   spread copy as well once the header is known (spec 6.3's L figures assume
   it), and the Morse
   reader with one unit per L symbol. Then the joint listener: both A2s per
   line, the shape test, per-waveform aliases and noise bands, and the
   waveform recorded per pass.
4. **Check the thresholds** of spec 6.3 on Watterson paths, and measure L's
   header (estimated -27 dB from the block, about 1.5 dB deeper with the
   spread copy).
5. **The app:** a waveform choice in the Send editor and the waveform shown
   per pass (UI thread).

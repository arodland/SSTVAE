# Live HF video from autoencoder latents — design draft

Working title: **AETV** (autoencoder television). Alternatives at the end.
Status: **design draft, nothing implemented, no numbers measured** except
where explicitly credited to SSTVAE's existing measurements. Intended to
move to its own repository if pursued; written to be self-contained.

## 1. The reference point: ZL2AFP OFDM NBTV

Con Wassilief ZL2AFP and Murray Greenman ZL1BPU built, in the mid-2000s,
a live analog TV mode for HF voice channels
(https://www.qsl.net/zl1bpu/NBTV/OFDM.htm). The design, from that page:

- **One OFDM carrier per picture line.** 48×48 mode: 48 carriers spaced
  42 Hz; 96×72 mode: 72 carriers spaced 32 Hz. ~2.1 kHz total either way.
- **Pixels are analog, sent as very narrow FM** (a few Hz deviation) on
  each carrier, pixel rate 50 Hz (48-line) or 33 Hz (96-line).
- **No synchronization mechanism at all.** Transmit and receive soundcards
  are assumed to run at the same rate; the frame edge is a dotted line in
  the picture, aligned by eye. A sine-modulated pilot carrier mid-band is
  a *tuning aid for the operator*, not an input to the demodulator.
- **Frame periods**: 1.0 s (48×48 B&W), 3.0 s (48×48 RGB — three
  sequential color fields), 3.0 s (96×72 B&W), 9.0 s (96×72 RGB), 6.0 s
  (96×72 RGGB interlace).
- **Tuning must be within 1 Hz**; a VFO rig cannot be used. On NVIS paths
  the color modes grow "colour stripes" as multipath moves energy between
  the line-carriers.

What it got right is worth keeping: it is *continuous* (motion, not
stills — 30–100× SSTV's frame cadence), it is *analog all the way down*
(no cliff; pictures degrade into the noise instead of failing), and it
fits an ordinary SSB channel. These are exactly the properties RADE and
SSTVAE get from analog latents, arrived at from the other direction.

What dates it is one design decision: **the channel carries raw pixels,
one per analog sample**. Everything painful follows from that. Raw
pixels mean ~2,300 px/s is all a 2.1 kHz channel can do, so resolution ×
frame rate is capped at 48×48×1 fps equivalent. One-carrier-per-line
means carrier identity *is* picture geometry, so mistuning scrambles
lines and 1 Hz accuracy is structural, not incidental. Narrow-FM
discrimination per carrier forgoes coherent combining, and with no
pilots there is no equalization, so multipath paints stripes.

## 2. The proposal in one paragraph

Apply the RADE/SSTVAE recipe to video: a learned **spatiotemporal
autoencoder** turns short blocks of video frames into unit-RMS analog
latents at a fixed rate; the latents ride as analog I/Q values on the
**existing SSTVAE OFDM waveform** (pilot-anchored coherent demod,
per-GOP interleaving, per-latent confidence weights, beacon
side-channel); training runs **through a differentiable replica of the
channel** (SSTVAE's stage-1/stage-2 split, with the fading correlated in
time across the block). The modem, sync, and training machinery already
exist and are measured; the new work is the video codec and the
streaming receive loop. Compression is what moves the needle: at a
conservative ~16–30:1 pixels-per-latent (against NBTV's 1:1), the same
2 kHz that carried 48×48 at 1 fps carries ~96×72 to 160×120, in color,
at ~7–14 fps — and tuning tolerance is set by acquisition DSP (±625 Hz
pull-in in SSTVAE today), not by the operator's dial.

## 3. Channel budget arithmetic

All of this reuses SSTVAE's waveform constants (FS 8000 Hz, 50 Hz
carrier spacing, 160+32-sample symbols = 24 ms, frames of 1 pilot + 5
data symbols = 144 ms, one carrier reserved for the beacon). Each data
carrier then delivers 5 syms × 2 (I,Q) per 144 ms = **69.4 real analog
values/s**; equivalently ~1.39 values/s/Hz after CP and pilot overhead.

Two bandwidth variants:

| Variant | Carriers | Band | Occupied | Latent rate |
|---|---|---|---|---|
| **N** (narrow) | 24 (23 latent + beacon) | 950–2100 Hz | ~1.2 kHz | **1,597/s** |
| **W** (wide) | 45 (44 latent + beacon) | 450–2650 Hz | ~2.25 kHz | **3,056/s** |

N is byte-compatible with SSTVAE's band plan and passes any filter that
passes SSTVAE. W fills a standard 2.4 kHz SSB filter (300–2700 Hz) with
margin at both edges; operators with 2.1 kHz filters use N. At equal
transmit power W spreads it over ~2.7 dB more carriers, so N buys ~3 dB
of per-latent SNR — N is the weak-signal/NVIS variant for two reasons at
once. (SNRs below are in SSTVAE's 2500 Hz reference-bandwidth
convention.)

Context for feasibility: at 10 dB per-latent SNR an analog value carries
≈ ½·log₂(1+SNR) ≈ 1.7 bits equivalent, so W is ~5 kbps-equivalent at
good SNR, degrading smoothly below. Modern learned video codecs produce
acceptable talking-head video in the 3–10 kbps range, and NBTV itself
proves 2,300 *uncompressed* px/s is already watchable. The budget is
tight but not fanciful.

For comparison, NBTV's raw rate is ~2,300 px/s in 2.1 kHz. At 16–44:1
compression the W variant carries a ~20–60× higher effective pixel
rate — which the mode table below spends on resolution, frame rate,
color, and SNR margin in different proportions.

## 4. The codec

### GOP-based spatiotemporal autoencoder

Encode video in independent **GOPs of 8 OFDM frames = 1.152 s** (8
video frames at 6.94 fps, 16 at 13.9 fps — the video cadence inside a
GOP is codec-internal; on-air, a GOP is just a fixed-size block of
latents). The encoder is a 3D-conv (or 2D + temporal attention) network
over the GOP producing a latent block with the same contract as SSTVAE:
unit-RMS tanh latents, decoder consumes `latents × weights` plus weight
planes, erasures/truncation handled by construction.

Sizing sketch, W variant, 96×72 @ 6.94 fps: 3,520 latents/GOP ≈ a
4×12×9 spatiotemporal grid (×8 spatial downsample, ×2 temporal) × 8
channels. 160×120: ×8 spatial → 20×15, ×4 temporal → 2×20×15 × 6
channels ≈ 3,600. Both are ordinary autoencoder shapes.

Why GOPs rather than the two alternatives:

- **Frame-independent 2D coding** (an SSTVAE-per-frame) wastes exactly
  the redundancy that makes video cheap — consecutive webcam frames are
  nearly identical. It would need ~5× the latent rate for the same
  quality. Rejected as the primary design, but it is the natural ablation
  baseline in stage-1 experiments.
- **Recurrent/predictive coding** (condition on previously decoded
  frames) compresses best and fails worst: on a fading channel the
  decoder's state diverges from the encoder's and errors propagate — the
  DPCM death spiral, and the analog cousin of NBTV's color stripes.
  There is a genuinely interesting middle path *because* this system is
  analog: train the conditioning against **channel-corrupted** previous
  latents, so the decoder's state is by construction the state the
  encoder was trained to expect, and fades degrade the conditioning
  smoothly instead of invalidating it. Worth an experiment (§10), not
  the v1 design.

Independent GOPs bound every failure: a deep fade costs one or two soft,
blurry GOPs and the next one starts clean; a late joiner starts at the
next GOP boundary, ~1 s away. This is SSTVAE's "erasures are a training
condition, not an error" philosophy applied in time.

### Interleaving and fading

Interleave each GOP's latents over its full 1.152 s and across all
carriers (same frozen-permutation construction as SSTVAE's per-group
interleaver). Watterson fading coherence is ~400 ms, so a fade inside a
GOP turns into a uniform confidence-weight reduction across the whole
block — every frame gets slightly softer — rather than some frames
dying. The per-latent Wiener-style confidence weights carry over
unchanged and are what makes this graceful.

Note the latency accounting here, because it is easy to misattribute:
**the block interleaver adds no latency on top of the GOP codec.** The
receiver cannot decode until the whole GOP has arrived regardless — the
3D decoder consumes the block — so interleave depth and codec block
length are matched, and deepening one to the length of the other is
free. The latency owner is the *block structure itself*: a frame waits
up to one GOP for its block to close at the encoder, then the block
takes one GOP-duration of air time. That is what the low-latency
profile below changes.

### Low-latency profile

The GOP modes' glass-to-glass latency (~1.2–2.3 s, §7) is fine for
one-way viewing and marginal for conversation. A genuinely low-latency
variant changes the codec from blockwise to **causal**: temporal
context is a bounded window of *past* frames only (causal convolutions
over the last few frames — the bounded form of the recurrent
experiment above, with the same discipline: the decoder's history is
its own channel-corrupted output, and training runs through that), and
each video frame's latents are emitted, transmitted, and decodable as
soon as that frame's air time ends.

Once the codec is causal, interleave depth detaches from the codec and
becomes a pure latency↔smearing knob, from one video frame (72–144 ms;
carriers-only interleaving is free at any depth) up through a
convolutional interleaver over a few hundred ms — the classic broadcast
structure, half the end-to-end delay of a block interleaver at equal
depth, and it decodes continuously rather than in blocks.

What replaces deep interleaving is **concealment, which video has and
stills do not**: a fade that would have been smeared into a whole GOP
instead hits a contiguous run of frames whose confidence weights
collapse, and a causal decoder trained on burst erasures learns to
coast — hold and softly re-converge — through them. Stage-1/2 training
for this profile therefore uses *contiguous* burst erasures matching
un-interleaved fade statistics, not the scattered erasures the
interleaved modes train on. Whether viewers prefer localized 0.3 s
glitches (this profile) to uniform once-a-second quality breathing (the
GOP modes) is a genuinely open subjective question (§12); the guess
recorded here is that for conversation they do.

Latency estimate for the profile (causal codec, one-frame interleave,
13.9 fps): 72 ms frame capture + 72 ms air time + codec ~20 ms + audio
buffering ~100–200 ms ≈ **0.3–0.5 s glass-to-glass** — against ~1.7 s
mean for the GOP modes. The cost is compression efficiency (one-sided,
bounded context; expect worse than the symmetric GOP codec by an amount
stage 1 must measure) and the localized fade artifacts above. On-air,
nothing changes but the interleaver tables and the mode ID: pilot
cadence, acquisition, beacon, and the latent rate are identical, so
this is a codec/profile choice, not a second waveform.

### Ordered latents / graceful truncation

Order latent channels by importance within the GOP (coarse-to-fine,
static-to-dynamic), as SSTVAE orders its three groups, and train with
random truncation. This buys three things: late join mid-GOP renders a
degraded first GOP instead of nothing; a future *nested-bandwidth*
option (W transmission decodable by an N-filter receiver from the
central 24 carriers, at reduced quality) becomes a carrier-mapping
decision rather than a redesign — though it constrains the interleaver
to respect the carrier split, so it is a deliberate trade, not free; and
adaptive quality (drop the tail group when the reported SNR is poor)
needs no new waveform.

### Color

Color is nearly free in a learned codec — chroma is low-entropy and
costs perhaps 10–20% of the latent budget, against NBTV's 3× (sequential
R/G/B fields). All modes below are color; a B&W variant is a training
choice, not a waveform one, and probably not worth a mode slot.

### What the codec must not do

Compression this aggressive on face-heavy training data shades toward
face-reenactment ("avatar") codecs that *animate* a reference face at
<1 kbps. That is explicitly out of scope: the receiver should show what
the camera saw, degraded, not what the prior thinks a face looks like.
Concretely: keep compression in the 16–44:1 regime rather than 300:1,
train and **evaluate on off-distribution content** (text cards, charts,
hands, outdoor scenes — the int8-quantisation lesson from
`docs/onnx.md`: scoring on photographs alone ships the failure), and
treat "test card readability" as a release gate, not a nice-to-have.

## 5. Waveform and sync for continuous streams

### What carries over unchanged

50 Hz carrier spacing on integer multiples (truly cyclic CP), 4 ms CP
(~2 ms delay spread — covers Watterson mpp/mpd), 6-symbol frames with
144 ms pilot spacing (follows ~1 Hz Doppler), envelope clipping at
`CLIP_HEADROOM_DB`, minimized-crest pilot philosophy, Golay-coded
header, beacon side-channel, blind acquisition. The modem stack is the
part of this project that already exists.

Two required adaptations:

- **A new pilot phase set for 45 carriers** (W variant). The 0.99 dB
  crest-factor set is a per-carrier-count optimization; re-run the
  numerical minimization for 45 carriers and freeze the result as exact
  rational turns, same as `PILOT_PHASE_NUM`. The Zadoff-Chu prohibition
  carries over verbatim: the pilot is still the acquisition template, so
  ZC's delay–Doppler equivalence is still disqualifying.
- **Beacon payload gains a mode field.** A continuous stream has no
  "start" a late joiner can rely on, so the mode ID must repeat: add
  ~4 mode bits to the superframe payload (counter + callsign + CRC as
  today). The absolute frame counter (mod 1024) gives GOP alignment
  (GOP = counter mod 8) and interleaver phase; the ~10.5 s worst-case
  superframe window bounds late-join time-to-first-picture.

### Transmission structure

Preamble + header at keying (fast acquisition for a receiver already
waiting), then GOP frames indefinitely — there is no fixed transmission
length and no end-of-image. The receiver runs the same two acquisition
paths as SSTVAE: preamble lock at stream start, blind pilot matched
filter + beacon for a receiver that tunes in mid-stream. **The blind
path and beacon were built in SSTVAE precisely for join-in-progress**,
so the hard part of continuous-stream sync exists and is measured.

### Tuning accuracy — the direct answer to NBTV's 1 Hz

NBTV needs 1 Hz because carrier frequency *is* line identity and the
per-carrier FM discriminators have no common phase reference. Here,
carrier identity comes from acquisition and the pilot, so tuning error
is just CFO to estimate and remove:

- Initial acquisition pulls in **±625 Hz** (SSTVAE's `ACQUIRE_MAX_BINS`,
  measured 2026-08-11), with the template-score gate against false
  locks; the blind path does ±55 Hz by default, ±625 Hz in wide mode.
  Any reasonably tuned dial is inside this without care.
- Residual and slow LO wander are handled by the pilot common-phase
  drift loop (`drift_track`). For SSTVAE's bounded transmissions it is
  optional; for streams that run minutes to hours it is **mandatory**,
  since the ±2 Hz total-excursion budget will be exceeded by almost any
  radio eventually. A continuous stream is actually the loop's easy
  case — it has unlimited time to converge and no end-of-transmission
  deadline. The known open issue carries over: no single (α, β) pair
  serves both fast wander and slow-fade channels
  (SSTVAE `docs/todo.md`), so the slow/fast setting remains exposed.
- Sample-clock offset: EMA drift tracking as today, and the GOP
  structure resets accumulated timing error every 1.152 s anyway.

Net: synthesized rigs need no care at all, and even a drifting VFO is
plausibly usable with `drift_track fast` — the exact rig NBTV
disqualifies. Worth measuring for the announcement value alone.

### PAPR

More carriers raise unclipped PAPR, but the envelope clipper already
sets the transmitted PAPR (~4 dB) and stage-2 trains through it; the
pilot re-optimization above keeps the pilot out of the clipper. No new
mechanism expected; verify with `dsp.papr_db` sweeps when the 45-carrier
variant exists.

## 6. Mode options

All figures below except NBTV's are **design targets, not measurements**.
"Latents/frame" = latent rate ÷ frame rate; px:latent counts luma pixels
only.

| Mode | Band | Resolution | Rate | Latents/frame | px:latent | Intended use |
|---|---|---|---|---|---|---|
| **V0 "NVIS"** | N (1.2 kHz) | 64×48 color | 6.94 fps | 230 | 13:1 | 80 m at night, weak signal; ~3 dB per-latent advantage over W modes |
| **V1 "Classic"** | W (2.25 kHz) | 96×72 color | 6.94 fps | 440 | 16:1 | The NBTV tribute: its best resolution, in color, at ~60× its RGB frame rate |
| **V2 "Motion"** | W | 96×72 color | 13.9 fps | 220 | 31:1 | Conversation/"presence"; smoothness over detail |
| **V3 "Detail"** | W | 160×120 color | 6.94 fps | 440 | 44:1 | Good conditions, quiet bands |
| **V4 "Still+"** | W | 320×240 color | 0.87 fps | 3,520 | 22:1 | Slideshow / SSTV-replacement niche; one GOP per frame |
| **V5 "Convo"** | W | 96×72 color | 13.9 fps | 220 | 31:1 | Two-way conversation: causal codec + shallow interleave (§4 low-latency profile), ~0.3–0.5 s glass-to-glass |

Mode ID travels in the beacon (and header), so a receiver needs no
prior arrangement. V0–V3 share one waveform per band variant; a mode is
purely a codec/latent-layout selection, like SSTVAE's A/B/C.

Expected behavior over SNR (estimates to be measured, stated in the
2500 Hz convention): recognizable talking-head video from roughly
5 dB (W) / 2 dB (N); comfortable at 10 dB; V3 earning its resolution
above ~12 dB. The *floor* should sit near SSTVAE's, because per-latent
SNR mechanics are identical — what falls at low SNR is the quality
ceiling, smoothly, which is the analog premise. Multipath: mpg/mpp
handled by CP + pilots as in SSTVAE; mpd expected to cost quality but
not lock; a >1 s deep fade costs the affected GOPs and nothing after.
V2's smaller per-frame budget makes it the most fade-tolerant W mode
(more temporal redundancy per pixel), V3 the least.

An honest uncertainty, flagged as such: the 44:1 cell (V3) is the one
the stage-1 rehearsal (§10) may kill. V0/V1's 13–16:1 is comfortable
even for near-frame-independent coding; 44:1 leans on temporal
compression working well under channel noise.

## 7. Latency, duplex, and the shape of a "chat"

For the GOP modes, glass-to-glass latency is **~1.2–2.3 s** depending
on where a frame falls in its block (mean ~1.7 s): a frame waits up to
one GOP (1.15 s) for its block to close at the encoder, the block takes
one GOP of air time, and decode/audio buffering add ~0.2 s. Fine for
what those modes are: HF video chat is alternating overs or two
stations watching each other's continuous streams, not a zoom call.
Where latency matters, V5's causal profile (§4) targets **0.3–0.5 s**
by trading compression efficiency and fade-smearing for it. Half duplex is assumed (same rule as SSTVAE:
transmitting suspends receive, fresh ring buffer on resume). True
full-duplex "videophone" = split frequencies or cross-band, out of
scope for the waveform.

Voice: a natural future extension is partitioning carriers between
video latents and a RADE-style speech-latent stream (speech is a few
hundred latents/s), making a genuine A/V mode in one SSB channel. Not
in v1; noted because the carrier-partition mechanism is the same one
the nested-bandwidth option uses, so it should be kept in mind when
freezing the interleaver layout.

## 8. Compute

SSTVAE's measured figures: encoder 31 ms, decoder 50 ms per 640×480
frame on desktop CPU via onnxruntime. V1 frames have 1/44th the pixels
of 640×480; even with 3D-conv overhead and 7–14 fps cadence, codec cost
is a few percent of a core. The DSP side is a continuous version of
SSTVAE's poll loop, which demodulates faster than real time by a wide
margin on desktop and within duty-cycle budget on phones
(`decode_loop_low_cpu`, `max_decode_duty`). Real-time video on the
Android target looks unproblematic *for the codec*; the screen and
rendering path is the thing the Android tier learned to respect.

## 9. Training plan

Direct reuse of SSTVAE's two-stage structure:

- **Stage 1**: differentiable latent channel — AWGN + GOP truncation +
  erasure bursts (time-correlated, matching fade statistics, rather than
  i.i.d.). This is where codec architecture search happens, torch-only,
  no DSP.
- **Stage 2**: the `waveform_channel.py` replica generalized to the
  carrier count, with symbol-domain Watterson fading **correlated across
  the GOP** (the existing implementation is already time-correlated
  within a transmission — verify the coherence structure matches at GOP
  scale) and the envelope clipper in the loop.
- **Data**: talking-head/webcam corpora for the primary distribution
  (VoxCeleb2-class), diluted with generic video and synthetic
  off-distribution probes (text, charts, high-motion) for the §4 gate.
  Dataset licensing needs a pass before anything is published — face
  datasets carry restrictions the COCO pipeline never had to think
  about.
- The latent unit-RMS normalization remains the on-air contract;
  confidence-weight training carries over.

Also inherited: the model checkpoint is part of the on-air contract
(both stations must run the same published revision to interoperate —
the same argument that put the model inside the Android APK), and any
future latent-domain metric is an objective value, not a result
(SSTVAE's latent-PSNR lesson).

## 10. What to build first — cheap kills before expensive commitments

1. **Latent-budget rehearsal (no DSP at all).** Train a small GOP
   autoencoder on webcam video with an AWGN latent channel at exactly
   the table's budgets (230/440 latents per frame at each resolution,
   5/10 dB latent SNR). Look at the output. This is a few GPU-days and
   answers the only real go/no-go: is 16–44:1 spatiotemporal
   compression through an analog channel watchable? Include the
   frame-independent ablation to price the temporal machinery, and a
   causal-context variant to price the low-latency profile's one-sided
   window against the symmetric GOP codec.
2. **45-carrier waveform variant in simulation.** Parameterize carrier
   count/base frequency in a fork of `config.py`, re-run the pilot
   crest minimizer, sweep latent SNR vs. the 24-carrier waveform over
   AWGN/mpg/mpp/mpd. Confirms the ~1.39 values/s/Hz budget and the ~3 dB
   N-vs-W per-latent gap before any codec depends on them.
3. **Streaming sync soak.** 10+ minute continuous transmissions through
   the channel sim with LO drift ramps and wander; verify the drift
   loop + beacon keep lock indefinitely and quantify the VFO-rig claim
   before it gets made in public.
4. **End-to-end loopback prototype** (stage-2-trained codec + streaming
   rx engine), then RF.

Steps 1 and 2 are independent and can run in parallel; each can kill or
resize the project for a few days' effort, which is the point.

## 11. Reuse map from SSTVAE

| Piece | Fate |
|---|---|
| `modem/ofdm.py`, `dsp`, `golay`, `sync`, pilot machinery | Reused; carrier count parameterized; new 45-carrier pilot set |
| `modem/beacon.py` | Reused + mode bits in payload |
| `modem/framing.py` | Same construction; per-GOP permutations, new frozen tables |
| `hfchannel.py`, `waveform_channel.py`, stage-1/2 training harness | Reused; carrier-count generalization; GOP-correlated fading check |
| `sync.acquire_blind` / beacon join, drift loop | Reused — this *is* the continuous-stream sync story |
| `rx/engine.py` | New engine (streaming, no completion state machine), reusing ring buffer, audio layers, watchdog lessons |
| `tx/engine.py`, PTT watchdog, audio stack, `native/core` structure | Reused |
| Autoencoder, codec, ONNX export | New (spatiotemporal); export/laziness/precision playbook reused |
| Frozen-format-constants discipline, golden vectors, `--native` parity | Same methodology from day one |

The honest summary of the delta: **the codec + dataset + training is
the project**; the modem is largely done and measured, and the
streaming receiver is a rewrite of a loop whose failure modes SSTVAE
has already catalogued.

## 12. Open questions

- Does temporal compression survive per-latent noise at 5 dB, or does
  motion smear into mush? (§10 step 1 answers this.)
- GOP length: 1.152 s balances latency, interleaver depth, and late-join
  granularity — but it was chosen for arithmetic convenience (8 OFDM
  frames), not measured. Sweep 0.5–2.5 s in stage 1.
- Is decoder-side temporal conditioning across GOPs (trained on noisy
  state) worth its risk? Potentially large quality win for static
  scenes; the failure mode is exactly NBTV's stripes in modern dress.
  V5's bounded causal context is the contained version of this bet, and
  V5 stands or falls with it.
- Interleaving versus concealment, subjectively: do viewers prefer the
  GOP modes' uniform quality breathing under fades or the causal
  profile's localized glitches? Decides whether a mid-depth
  convolutional interleaver (~0.5 s, ~1 s glass-to-glass) earns a slot
  between them.
- Nested-bandwidth (W stream decodable by N receiver): worth the
  interleaver constraint?
- Does mpd need a denser-pilot variant (1-in-4, −10% capacity), or does
  the 144 ms cadence hold as it does for SSTVAE?
- Aspect ratio / portrait orientation for phones — a codec/training
  question with UI consequences, decide before freezing latent grids.

## 13. Naming and repo notes

Candidate names, no attachment: **AETV**, **LiveAE**, **HFTV**,
**Fuzzyvision** (a nod to the Fuzzy-modes lineage NBTV claims). "RAVE"
is taken (IRCAM's audio VAE).

If this proceeds it gets its own repository (per Andrew). The pragmatic
split: fork the modem/`config`/training layers rather than sharing a
package — the waveform constants diverge immediately (carrier count,
pilot set, beacon payload), and SSTVAE's config module is deliberately
a single frozen truth that two waveforms cannot share. Keep the
methodology, not the dependency: frozen format constants, generated
config headers, golden vectors, and Python-as-normative-oracle from the
first commit, because this project has already paid for those lessons
once.

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

## Completed: acquisition after a previous reception ("locks after a fresh start")

**Fixed 2026-08-31, both implementations.** Operators reported reception
locking reliably after a fresh start (app launch, stop/start listening,
the post-transmit buffer reset) and poorly after a completed reception.
Two mechanisms, both erased by exactly the actions that helped:

- **Stale blind evidence.** `decode_loop` never reset its
  `BlindAccumulator` when a reception was delivered, and a delivered
  transmission's fold holds its lock for a long time — the peak/median
  score does not decay on its own (decay scales a timescale's bins
  uniformly), and its off-phase energy inflates the median under any new
  peak in the same CFO row (the common case: the same station sending
  again). `result()` returns only the single best cell, so every poll
  re-locked and re-demodulated the finished transmission, discarded it
  via `finished_starts`, and a new transmission could not surface. The
  loop now installs a fresh accumulator at delivery with
  `blind_acc_pushed = total` — `= buf_start` would re-fold the
  still-buffered finished transmission straight back in. **The accepted
  cost:** a transmission *overlapping* the delivered one loses its
  accumulated blind evidence and rebuilds from delivery time; using the
  old evidence for the overlapper while rejecting the delivered peak
  would need per-reception subtraction the CPU-friendly accumulator
  cannot do. Its audio is still in the ring, so a rebuilt lock still
  decodes it retrospectively. Pinned (with a mutation check — the test
  times out without the reset) by
  `tests/test_listen_state_machine.py::test_second_blind_transmission_locks_after_the_first_is_delivered`
  and its mirror in `native/tests/test_rx_engine.cpp`.
- **Gate rejections aborted the preamble search.** See the top-K note
  under the steady-carrier item below: a lag-M artefact in the finished
  transmission's still-buffered frame data wins the metric argmax, the
  template gate correctly rejects it, and single-candidate `acquire`
  declared the whole span preamble-free — hiding a genuine
  weaker-metric preamble for as long as the old audio stayed in the
  ring (~130 s).

## Completed: preamble detection against a steady-carrier interferer

**Mostly closed 2026-08-13**, as a side effect of the wide-acquisition
false-lock fix: `config.TEMPLATE_SCORE_THRESHOLD` also rejects a steady
tone at every level the original table named — a pure tone reads
metric 1.000 but scores only ~0.2–0.3 against the 24-carrier preamble
template. **Top-K candidates landed 2026-08-31** (both
implementations), arbitrated by the template gate rather than the Golay
header: `acquire` now masks a gate-rejected metric peak and tries the
next-best, up to `config.ACQUIRE_MAX_CANDIDATES` (5) — so a tone (or a
still-buffered previous transmission's lag-M artefact) no longer steals
the lock from a real preamble sharing the window, and candidate 1 is
exactly the old argmax, so sensitivity is untouched
(`tests/test_preamble.py::test_a_gate_rejected_peak_is_stepped_past_not_fatal`).
The motivating case was post-reception deafness: old frame data in the
ring wins the argmax, the gate correctly rejects it, and the old
single-candidate `acquire` then declared the whole window
preamble-free. **Still open:** header arbitration for a *gate-passing*
impostor — a periodic segment can score above 0.40 at some CFO bin and
still carry no decodable header; today that costs one of
`_find_new_reception`'s demodulate tries rather than the lock.

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

## Android rig control: two USB ports, one device id

Landed 2026-08-22 (`docs/android.md`, "Rig control was said to be
structurally impossible") and taken to hardware 2026-08-23. Most of what
this section used to list is answered:

- **A composite USB device gives the phone audio and serial at once.**
  This was the question that could have sunk the approach, and on an
  Elecraft K4 the answer is yes, with CAT and both control-line keying
  methods working. mik3y/usb-serial-for-android#477 remains an open
  report of a composite device where it does not, so it is a per-device
  answer rather than a general one.
- The Hamlib NDK cross-build, CAT against a real radio, and DTR/RTS
  keying are all exercised. Bluetooth is not, for want of a device.

**What is open is that a K4 works and an IC-9700 and an IC-7100 do
not.** Everything structural was ruled out by reading, and re-deriving
it is the waste this paragraph exists to prevent: the byte path is
length-counted end to end (`SerialBridge.read` → a `jbyteArray` region
copy → `LoopbackBridge`'s `send`), so nothing truncates a binary CI-V
frame at a `0x00`; Hamlib's POSIX `port_read_generic` and `port_write`
have no port-type branch at all, the ones that exist being Win32 serial;
`network_flush` is a `FIONREAD`-guarded drain that cannot block;
`rigs/icom/` contains no port-type conditional; and `serial_defaults`
takes `rig_caps.serial_rate_max`, the same field `rig_init` uses, so a
bridged rig runs at the speed that rig runs at on a desktop.

Nor is it the control lines: **both** `FtdiSerialDriver` and
`Cp21xxSerialDriver` deassert DTR and RTS in `openInt()`, so the K4 is
CAT-ing with them low. That is a difference from a desktop, where the
OS raises them on open, but demonstrably not a fatal one -- and every
Icom in Hamlib declares `RIG_HANDSHAKE_NONE`, so nothing is applying
flow control that could hold the chip's transmitter off.

**Cause found 2026-08-23, awaiting hardware confirmation: an IC-9700
presents two USB serial devices sharing `10c4:ea60` -- CI-V and the USB
serial function -- and `usb:VID:PID` named both.** The picker showed two
identical rows and `findUsbDevice` returned whichever `getDeviceList()`
yielded first, out of a `HashMap`. Ids now carry `@unit` beside
`#port`, which is a different axis: `#port` is one driver's several
UARTs, `@unit` is several devices with one VID:PID. What follows is the
measurement record from before that was found, and it stands -- it is
what ruled the chip's configuration out. `fe fe a2 e0 03 fd` -- address A2, the
IC-9700's default -- written, `read_string_generic(): Timed out 1.001
seconds after 0 chars`, every time, at 115200. `icom_get_usb_echo_off`
therefore returns `-RIG_ETIMEOUT` and `icom_rig_open` gives up with
"is rig on and connected?". What that trace *cannot* say is whether
those six bytes reached the USB endpoint, because Hamlib only ever saw
a successful socket write. `core/rig/trace.hpp` closes that gap: the
bridge now logs each direction's bytes (after the transport accepted
them, never before) and the transport logs which driver and interface
the Android USB layer chose. So the next run distinguishes:

The trace showed `-> rig 6: fe fe a2 e0 03 fd` to a **CP2102N** at
19200 8N1 flow=none, one vendor-class interface, two endpoints,
`Cp21xxSerialDriver` port 0 of 1, and no `<- rig` line, ever -- every
control transfer returning 0 and every bulk write returning its length.
Andrew's A/B settled it: **FT8TW drives the same radio on the same
phone**, and its fork of `Cp21xxSerialDriver` has no
`SILABSER_SET_FLOW_REQUEST_CODE` constant and no `setFlowControl`
method at all. Setting a CP210x's flow control to *none* is not a
no-op; it writes sixteen zero bytes over `ulControlHandshake`,
`ulFlowReplace`, `ulXonLimit` and `ulXoffLimit`. The Linux `cp210x`
driver, which the working `rigctl` goes through, does
`GET_FLOW`/modify/`SET_FLOW` and never zeroes the limits, and carries
erratum **CP2102N_E104** -- firmware <= 0x10004 reads `ulXonLimit` as
`ulFlowReplace`, so a blind write lands one word out of alignment.

That write is now skipped when there is nothing to set, in
`SerialBridge.applyFlowControl` and in the vendored
`Cp21xxSerialDriver.openInt` (both: `openInt` runs inside
`port.open()`) -- the first patch carried against the vendored library,
`third_party/usb-serial-for-android/PATCHES.md`. **It did not fix it**,
and that is the useful result: FT8TW's `setParameters` is behaviourally
identical to ours, so with the write gone the two program the chip the
same way and the Icom still fails. The register writes are not the
difference. The guard stays because both reference implementations
avoid the blind write and the erratum is real.

So the next question is what the chip does with bytes it has accepted,
which it will answer itself. `SerialBridge.describeStatus` issues
`GET_COMM_STATUS` (AN571, 19 bytes) and the transport traces it every
ten consecutive empty reads, with the modem control lines:

**Measured 2026-08-23**: `errors=0x0 hold=0x0 inQueue=0 outQueue=0`,
all six modem lines reading 1, and `reads started=3 returned=2
last=512ms` against the bridge's 500 ms timeout. So the read loop
cycles normally, the chip reports no framing or overrun error and
nothing withholding transmission, and CTS is high so no handshake is
holding the transmitter off. `bridge: <- rig` never appears, so the
radio does not answer.

Two of those fields carry less than they appear to, and it is worth not
re-deriving: `outQueue` is sampled a second after the write and reads
zero whether the chip sent the bytes or dropped them, and `inQueue` is
drained continuously by the read thread. `errors`, `hold` and `CTS` are
the informative ones.

**That exhausts what code inspection can settle.** The chip is
programmed identically to a working implementation, reports itself
healthy, and no software here can see whether the UART put bits on the
pin. The next step is a hardware A/B that separates the chip from the
radio -- a CP210x USB adapter into the K4's RS-232 port, driven from
the same phone and app. If that fails, the fault is in the CP210x path;
if it works, it is the IC-9700 specifically.

One further difference from the Linux driver survives, as the next
place to look if a CP210x still misbehaves: the library writes the
requested baud raw where Linux applies `cp210x_get_actual_rate()` for a
CP2102N -- a no-op at 19200 and a 0.16% shift at 115200.

Radio-side, still worth eliminating if it is not fixed:

1. **The CI-V USB baud rate**, which on these radios is a menu item
   separate from the CI-V port's own and defaults to an "Auto" that is
   not reliable on every model. Set it to a fixed rate and set the same
   rate in Settings. Note that the trace shows **115200**, which is not
   this app's default: `serial_defaults` returns `serial_rate_max`,
   which Hamlib gives as 38400 for the IC-9700 and 19200 for the
   IC-7100. Worth trying Default (or 19200) as well as whatever the
   radio's menu says, since Auto has an easier time at the lower rates.
2. **CI-V USB Echo Back**, which is what `icom_get_usb_echo_off` is
   probing, and which `rig_open` gives exactly one attempt at -- it
   sets `retry` to 0 for the duration.
3. **The CI-V address**, which on these radios is a *separate* setting
   for the USB port when "CI-V USB Port" is "Unlink from [REMOTE]".
   The app sends to A2, the IC-9700's factory default; a changed
   address produces exactly this silence.

Still outstanding, unchanged and unrelated:

- **PTT timing against a physical radio**, which is the *same*
  outstanding item the desktop has and has never had: `ptt_lead_s` is
  0.3 s of guess. A phone into a radio is now a way to measure it.
- **Battery over a multi-hour session with a rig session open.** The
  poll is 10 s rather than the desktop's 5, and the bridge's idle cost
  is meant to be two wakeups a second on one thread; both are claims,
  not measurements.
- **Bluetooth RFCOMM against a real radio.**

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

## Frame-count truncation in the training channel

**Goal.** Replace `latent_channel.py`'s group-prefix truncation with a
truncation to *the latents a contiguous run of frames actually carries*,
using the interleaver's own frozen permutation. Purely a training
distribution change: no on-air format change, no model architecture
change, no receiver change.

**Why the current model is wrong about the commonest case.**
`apply_latent_channel` samples `keep = randint(1, LATENT_GROUPS+1)` — a
prefix of *groups*, i.e. exactly three truncation points, all at group
boundaries — and then applies iid erasure at up to `erasure_rate_max =
0.3`. But the physical event is "the transmission ended after frame f",
and inside the boundary group that is a scattered subset of that group's
latents at *any* fraction from 0 to 1. So every mid-group state is
modelled only as far as 30% erasure will stretch, and the states past
that are out of distribution.

**What this is worth, in order.** The primary goal is unchanged and this
item does not serve it: a correct final decode of a *complete* mode A, B
or C transmission — which is to say at a group boundary, the case the
current channel already trains. A good decode of a transmission
truncated somewhere else is a **nice-to-have**. Better progressive
decoding in the GUI is a **visual nicety that falls out for free** (on
every poll `rx/engine` hands the decoder a partially-filled group it was
never trained on, and this fixes that as a side effect — it is not the
reason to do the work).

So the honest framing is that this buys the second and third of those,
and its **risk is to the first**. Widening the truncation distribution
spends model capacity on states that are not the goal, and the
on-boundary decode can pay for it. That is the thing to measure, and it
is why the gate below is a non-regression rather than an improvement.

**The mechanism is one cached table that already exists.**
`framing.frame_of_latent()` maps canonical latent index → the absolute
frame carrying it, and returns **−1 for the latents each group
permanently drops**. Reshaped to `(LATENT_CHANNELS, LATENT_H,
LATENT_W)` — the same canonical order `flat_to_latents` uses — the mask
for "frames `f0` up to `f1` were received" is one comparison:

```python
tbl = frame_of_latent().reshape(LATENT_CHANNELS, LATENT_H, LATENT_W)
mask = (tbl >= f0) & (tbl < f1)
```

That single expression covers three things the current channel handles
separately or not at all: truncation, late join, and the permanent
`DROPPED_LATENTS_PER_GROUP` (4.2%). The last is worth calling out — those
2200 slots per group are a *specific, known, frozen* set, and today they
are modelled as part of a random erasure rate. Training against the real
set lets the decoder write those exact positions off instead of hedging
everywhere.

**Sample a window `[f0, f1)`, not a prefix `[0, f)`.** Same cost, and it
covers late join, which the current channel never samples at all — it
only ever produces prefixes of groups, so "tuned in after group 0 had
finished" is out of distribution today. Whether the decoder can make
real use of groups 1–2 without group 0 is an open question (the groups
are ordered by construction, so group 1 may be close to meaningless
alone), but it cannot currently even be asked, and the wiki's "losing
all of group 0 costs ~6 dB at a stroke" is measured against a model that
never saw the case.

**The late-join objection does not survive measurement.** The worry is
that training on frame prefixes biases the model against a late joiner's
suffix. Within a group it cannot, because the interleaver makes the two
statistically identical — that is the interleaver's entire purpose.
Measured on group 0, frames 0..109 against frames 110..219 against a
random 50% subset:

| mask | coverage | per-channel range | spatial std |
|---|---|---|---|
| prefix, frames 0..109 | 0.160 | [0.000, 0.503] | 0.0253 |
| suffix, frames 110..219 | 0.160 | [0.000, 0.518] | 0.0253 |
| random 50% of group 0 | 0.159 | [0.000, 0.512] | 0.0244 |

Indistinguishable. The real late-join asymmetry is at *group*
granularity, not frame granularity, and sampling a window rather than a
prefix is what addresses it.

**But land the window separately from the prefix.** A prefix `[0, f)`
already widens the training distribution away from the primary case; a
window `[f0, f1)` widens it again, and toward a case (no group 0) that
may be unrecoverable in principle. Two changes in one run means a
regression in the on-boundary decode cannot be attributed to either.

**Keep the iid erasure, and compose the two.** They model different
physical events and the decomposition is the honest one: the frame
window says **what was transmitted**, the iid erasure says **what was
lost within it** (fades, dropped frames). Composing them also defuses
the one hazard here — a frame window is drawn from a finite set of 660
masks, where iid erasure has unbounded variety, so windows alone could
be overfit to. With erasure layered on top, every mask is still unique.

**`waveform_channel.py` has to follow.** Stage 2 mirrors the same
capacity and erasure accounting, and a stage-1 model trained on frame
windows fine-tuned through a stage-2 channel still truncating by group
would be trained against two different definitions of the same event.

**How to know it worked — and the gate comes first.** The gate is that
**mode A, B and C decodes do not regress**: the same eval as any codec
revision, on the same real-material set. That is the primary goal, this
change puts it at risk, and if it moves down the item is not worth
having whatever the mid-group numbers say. The boundary *inputs* are
identical either way, so any movement there is the widened training
distribution acting on the model, not a different test.

Only then the upside, which a mode A/B/C table cannot show at all: a
sweep over *frame count*, reconstructing at every 10% of each group and
comparing against the current checkpoint on the same frames. Both halves
are needed — one is the gate, the other is the payoff.

## A latent-space denoiser between the channel and the decoder

**Goal.** A small model mapping (received latents, confidence weights) →
cleaner latents, feeding the **existing** decoder unchanged. Trained
against the channel. Sized to stay inside the phone's budget, unlike the
generative decoder below.

**This is the shelved "refiner" again, and the post-mortem is the
argument.** That attempt failed its real-material gate at +0.08 dB and
was shelved for want of a design direction (2026-07-31). The most likely
reason it failed is structural rather than a tuning miss: **a refiner
trained on a regression objective converges to the conditional mean**,
and the conditional mean is precisely the smooth, hedged output the
decoder already produces. Regressing toward it can only recover what the
decoder was already leaving on the table, which is not much — +0.08 dB
is what "correctly computed the average" looks like. A flow or diffusion
objective samples from the posterior instead, which is a different
target, not a better-tuned version of the same one. That is a
hypothesis about the old result, not a measurement of it; it is
falsifiable cheaply, since a regression-objective and a flow-objective
refiner of the same size can be trained against the same frozen codec.

**What is new and specific to this domain: the timestep is measured, not
guessed.** The latents arrive with a *known* noise level — the pilot EQ
reports it per latent, which is where the confidence weights come from.
A flow over latent space starting at the received latents is therefore a
denoiser at a **known** t: initialize the schedule at the timestep
matching the measured SNR rather than at t=1. That is fewer steps, and
the schedule is grounded in a measurement instead of a hyperparameter.
No other part of this project gets to hand a diffusion model its own
noise level.

**Why it is separable, which is the reason to do this one first.** It
trains against a frozen encoder and a frozen decoder, so it is testable
against the current published checkpoint without retraining either, it
changes nothing on the air, and it drops out cleanly if it does not pay.
It is also the cheapest way to find out whether a generative prior buys
anything here at all, before committing to the decoder below.

**Traps, all of them previously paid for.**

- **Score `latents × weights`, never bare `latents`** — the same
  measurement that invented a phantom "+1.5 dB from Wiener shrinkage".
- **Latent-domain PSNR is an objective value, never a result.** It
  flattered latent optimization by ~2×.
- **Gate on real material.** The refiner passed on photographs and died
  on the real-material gate; anything with a generative prior will fail
  in the same direction, so the certificate and the text images decide
  this, not COCO.
- Confidence weights must be an **input**, not just a loss weighting —
  they are the map of where the prior is allowed to act.

## A desktop-only generative decoder

**Goal.** Replace only the decoder with a conditional flow-matching
model, large, GPU-only. Keep the existing encoder, latents, channel,
interleaver and modem exactly as they are.

**The framing that makes this safe: it is a receiver-side-only
change.** Same on-air format, same golden vectors, same
`PROTOCOL_VERSION`, same everything a second station can observe. A
station running the big decoder and a station running the 4.47M conv
decoder receive the same transmission and interoperate completely. It
is the mirror image of transmit-time latent optimization, which was
sender-side-only for the same kind of reason, and it means this can be
built and abandoned without touching the format or asking anybody to
upgrade.

**The conditioning is richer than the tokenizer literature's.** FlexTok
and its relatives condition a generative decoder on a *clean, small*
description and the decoder invents everything unspecified. Here the
conditioning is noisy-but-dense latents **plus a per-latent reliability
map** that the receiver already computes. The confidence weight becomes
a spatially varying "how much am I allowed to invent here" dial: under a
deep fade the conv decoder produces mush, where a conditional flow
decoder can produce plausible sharp content *and* the weights say
exactly where to distrust it. That is a better-posed problem than the
one those papers solve.

**Three things to design in rather than bolt on.**

- **Seed the flow noise from the beacon** (callsign + absolute frame
  counter — both are decoded before reconstruction). Without it the
  decode is stochastic, two receivers get different pictures from one
  transmission, and the whole golden-vector / `pytest --native` /
  bit-exact-codec regime stops meaning anything. With it, decoding is
  reproducible and the parity claim survives in the form it already has.
- **Hallucinated text is the failure mode that matters**, and it is the
  one photographs cannot show. Callsigns, grid squares, certificates and
  QSL card text are real traffic here, and a generative decoder renders
  them confidently wrong. This is the same shape as the int8 result in
  `docs/onnx.md` — 0.10 dB on COCO against 1.54 dB on synthetic probes —
  so the off-distribution set decides it.
- **Decide what the operator is shown.** A picture that is partly
  received and partly invented, with no marking, is a different claim
  about what came over the air than this project has made so far. The
  confidence map is already computed, so rendering the uncertainty is
  available and costs nothing; whether it should be on by default is a
  question for Andrew, not a default to pick quietly.

**Measured 2026-08-30, stage 1 (`scripts/train_flow.py`,
`sstvae/latent_flow.py`).** A latent-space rectified flow against the
frozen v4 codec, trained on `latent_channel.py`, wins **+1.9 dB at
-3 dB and +0.95 dB at 0 dB** with no harm at good SNR (-0.04 dB clean).
The regression twin -- same net, same data, same channel, only the
objective differs -- gets **-1.0 dB**, and degrades further with
training as its latent MSE keeps falling. So the shelved refiner's
+0.08 dB really was the conditional mean being computed correctly, and
a flow objective is a different target rather than a better-tuned one.

**It does not transfer to the real modem**: through the full
encode/modulate/channel/demodulate path it is *worse* everywhere,
-0.6 to -3.5 dB at awgn 0 dB and up to -7.4 dB at mpp 0 dB. Three
mismatches, all in `latent_channel.py` rather than in the flow: the
clipper's ~0.79 latent gain is not modelled (so a sigma estimated from
the unit-RMS contract reads 0 at 3 dB and above, and the flow silently
goes inert), sigma is underestimated when it does fire (true ~1.18 at
0 dB against 0.715), and the real pilot-EQ weights are continuous
(min 0.009, mean 0.864) where training used binary masks. Correcting
the gain at inference makes it *worse*, which indicts the analytic
timestep itself: it needs a sigma the receiver cannot supply. Hence
`--channel wave --conditioned` -- train through `waveform_channel.py`,
with the received latents as conditioning rather than as the flow's
starting point.

**Two later formulations both failed, and the reason names the fix.**
Trained through `waveform_channel.py`: conditioning on (received,
weights) and generating from *noise* sat at -9 to -11 dB for 9000 steps
-- a 3.8M plain conv stack has a receptive field of about +-9 cells on
the 30x40 grid, which is enough to denoise locally and nowhere near
enough to synthesise globally, so generation from noise needs a U-Net or
DiT, i.e. the item below. A *bridge* (transport received -> clean,
`--objective bridge`) started near baseline and degraded with training,
-2.9 dB at 500 steps to -7.6 at 2000, exactly like the regression twin.
That is not tuning: its source is strongly dependent on its target, so
given a point on the path the pair is nearly determined and the
MSE-optimal velocity is the conditional mean -- **the bridge is the
regression twin in expectation**. It is worse than neutral because
conditional-mean shrinkage pulls latent RMS under 1, and unit-RMS is the
contract the decoder was trained against.

**So the +1.9 dB depended on knowing sigma exactly**, which is what let
the received latents be read as a point on a genuine noise-to-data path
(the noise is independent, many pairs share an x_t, and integration
samples). Take sigma away and both fallbacks fail structurally. The
blocker is an *interface* fact rather than a physical one: the modem
estimates SNR (`modem._estimate_snr_db`) and the receiver has it, but it
does not survive `reconstruct(codec, latents, weights)`. An optional
keyword there -- backward compatible, so `rx/engine.py` needs no edit --
restores the condition, and the flow can then be retried on the waveform
channel with a true sigma instead of one estimated from a unit-RMS
assumption the clipper breaks. Also worth profiling the channel's
*output distribution* before designing against it next time: latents are
clamped at +-10 and ~20% arrive with weight < 0.05, which cost two runs.

**Settled 2026-08-30: a latent-space flow is a net negative on this
channel, and the answer is structural.** After calibrating the modem
properly (below) and unfreezing the decoder, the three-way comparison on
the real modem -- base v4, decoder-only fine-tune, and flow+decoder
jointly -- says the flow costs 1-3 dB *on top of* whatever the decoder
does. Joint is worse than decoder-only in **20 of 21** PSNR cells and
HaarPSI (in no objective here, unlike the LPIPS both arms trained
against) is negative in **all 14** flow cells, -0.012 to -0.116, while
decoder-only sits at -0.0004 to -0.0117.

**Why, and it explains the one success too.** The decoder already *is*
the denoiser: it consumes `(latents, weights)` and was trained end to
end on this channel, so it has the optimal mapping. A flow that first
collapses the received latents to a point estimate can only discard
information -- it commits before the decoder gets to weigh the
reliability map. A denoiser pays only when the decoder is mismatched to
its input. That is exactly what the original +1.9 dB was: a v4 decoder
trained through the *waveform* channel being fed *latent_channel*
inputs, where the flow's real job was bridging two channel models. Train
and evaluate on one channel and the gain evaporates. (Best explanation
consistent with every run here, not a separate measurement.)

**The calibration is worth keeping even though the flow is not**
(`latent_flow.channel_calibration`). The real modem does
`r ~= g(snr)*z + n` with `n`'s per-latent std `~= sigma0(snr)/w`. `g` is
Wiener shrinkage the receiver has already applied -- 0.79 at high SNR
but **0.45 at 0 dB and 0.18 at -6 dB**, so treating the documented 0.794
as a constant is wrong by 2.5x exactly where it matters. The 1/w law is
the pilot EQ's and is clean (residual RMS 0.848 at w~1, 2.932 at
1/w=2.72, 6.198 at 7.14). AWGN and multipath agree to ~10% on both, so
one curve indexed by the SNR the receiver already estimates serves both.
Verified on a held-out seed at ratios 0.93-0.99 from -3 to 12 dB.
**Magnitude only** -- it says nothing about the residual's independence
from the signal, which is the likely reason everything degrades further
at `mpd` (delay 4.0 ms against a 4 ms cyclic prefix, so ISI begins, and
ISI is structured interference this model does not describe).
`reconstruct(codec, latents, weights, snr_db=)` is the optional keyword
that carries `DemodResult.snr_db` down to it; `rx/engine.py` needed no
change.

**Two dead ends recorded so they are not retried.** Re-applying the gain
to a flow's output (on the theory that a stage-2-trained decoder wants
its latents shrunk) is **worse in every cell**, by 0.3-2.0 dB -- the
decoder prefers de-shrunk latents. And `--objective bridge` (transport
received -> clean) collapses to the regression twin in expectation: its
source is strongly dependent on its target, so the MSE-optimal velocity
is the conditional mean. It degraded with training exactly as the
regression twin did.

**Running 2026-08-31: a DiT attempt on HF Jobs** (`sstvae/dit.py`,
`scripts/launch_flow_job.sh`). Andrew's call, after the conv results
above. The reasoning worth keeping: the "a denoiser can only discard
information" argument applies to the *analytic-timestep* formulation --
a point estimate handed to a decoder that already weighs reliability --
and does **not** dispose of the conditioned generative one, which failed
for a concrete architectural reason (a conv stack sees ~+-9 cells of a
30x40 grid, enough to denoise locally and nowhere near enough to
synthesise a coherent latent field). Attention removes exactly that
limit, and generating rather than denoising is also the only way to fill
absent groups and beat the mode-A cap.

33.6M params, patch 2 (300 tokens), dim 384 / depth 12 / heads 6, same
`forward(x, t_map, cond)` signature as `FlowNet` so the sampler and
losses are unchanged. Two details are deliberate: **adaLN-zero**, so an
untrained model predicts exactly zero velocity and the sampler is the
identity; and the timestep enters **twice**, as a scalar through adaLN
and as an input plane, because this project's timestep is per-latent and
a scalar cannot say "this group is absent while its neighbour is clean".

**`--decoder-weights` is a confidence floor, `max(received, floor)`**
(Andrew, 2026-08-31), not a set of cases: 0.0 is the old "received"
behaviour, 1.0 the old "ones", and 0.8 says the flow has cleaned
everything while received latents still outrank invented ones. **The
floor is what makes fill-in possible at all** -- the decoder computes
`z*w`, so at floor 0 anything synthesised for an absent group is
multiplied by zero and fill-in cannot beat the mode-A cap however good
it is. Wiring it turned up a harness bug worth remembering: the codec's
decode path fed the decoder the raw received map regardless, so a
decoder *trained* to trust synthesised latents would have been
*evaluated* with a different policy, and the mismatch would have read as
"the model does not work". The policy is stored in the checkpoint and
honoured at inference now.

Two runs in parallel, identical but for the floor (1.0 ->
`arodland/sstvae-flow`, 0.8 -> `arodland/sstvae-flow-p8`), 40000 steps,
batch 12, lr 1e-4, cc12m640 with `--nonphoto-frac 0.1`. Both cold-start
near -3 dB, which is the all-ones penalty the decoder must absorb rather
than the -9 to -13 the conv attempts began at. **The eval batch is 50/50
photographs and procedural now**; synthetic-only answered half the
question and flattered exactly the classes a perceptual objective
overfits.

**Two traps this cost, both the same shape as the earlier ones.** The
first launch died at step 1: `waveform_channel` does complex arithmetic
and `torch.complex` raises on BFloat16, so the channel must run outside
autocast (`scripts/train.py` documents the same split) -- invisible
locally because every local run used `--no-amp` for ROCm. And the flow's
width was saved under `"width"`, the same key `load_torch_model` reads to
size the `SSTVAE`, so a joint checkpoint would not load at all. The
general lesson stands: test the path that will actually run, not the one
that is convenient to run.

**Self-calibrating the channel from the received latents**
(`latent_flow.estimate_channel`, 2026-08-31). This is reusable whatever
happens to the flow, and it replaced two broken assumptions:

- **`DemodResult.snr_db` cannot index a channel calibration.** Measured
  against the nominal SNR it is **+2.9 dB optimistic on AWGN and -8.8 dB
  pessimistic on mpp at 12 dB**, and the sign flips with a fading state
  the receiver cannot detect. Anything that needs "how noisy is this
  reception, really" must not take that number at face value.
- **`waveform_channel` is not the real modem.** At awgn 0 dB the
  replica's latent gain is **0.49 against the modem's 0.70** (sigma0
  0.80 against 0.58), so a table measured on one does not describe the
  other. It correlates >0.98 on waveforms, which is what it was built
  for; that does not make its *latent* statistics interchangeable.

The estimator uses only the received latents and their weights, so
training and inference compute it identically. With `r ~= g*z + n`,
per-latent noise `sigma0/w`, and `E[z^2] = 1` by the unit-RMS contract,
take `S = E[r^2]` and `W = E[1/w^2]` over kept latents and look the pair
up on a 98-point manifold measured offline against the real modem.

Two details are load-bearing. **A free two-parameter fit does not work**:
`E[r^2|w] = g^2 + sigma0^2/w^2` is a straight line, but under AWGN nearly
every weight is ~1, so `1/w^2` has no dynamic range, the regression is
ill-conditioned and the intercept collapses to zero (measured: g
estimated at 0.001 where the truth is 0.70). Constraining to a measured
manifold fixes it. And **`W` is what separates AWGN from fading** at
equal SNR -- `S` alone cannot, since mean weight is pinned near 0.80
across the whole SNR range under fading while the gain moves 0.28 to
0.78.

Recovers `sigma0/g` -- the quantity that sets the flow's timestep -- to
within 1% in 22 of 23 cells including held-out channel seeds. Worth
**+1.6 dB on average and up to +4.0 dB** on the same checkpoint, and it
brought the in-training metric into agreement with the real modem (coco
mpp 0: -1.34 measured against -1.30 reported), which is what makes the
training numbers trustworthy as a deployment proxy at all. Andrew's read
of the pictures: the mis-calibrated version was destroying large chunks
of detail; the corrected one is neutral.

**The apparent "text" side finding is an artifact of the run's recipe,
not a result** (checked 2026-08-30 after Andrew asked why decoder-only
training would find a text gain that encoder+decoder training does not).
Decoder-only fine-tuning through `waveform_channel` measured +1.86 to
+2.44 dB on the `text` class and about -5 dB on `gradient`. Both are
explained by how that run differed from `scripts/train.py`, and neither
survives as evidence about generative decoding:

- **`--nonphoto-frac` was 0.25 against a default of 0.0**, so the
  decoder trained on far more procedural text. That these classes sit
  3-7 dB behind COCO on a photo-only model is already recorded here and
  `--nonphoto-frac` already exists to address it -- a known lever being
  pulled, not a discovery.
- **No chroma term** (`train.py` carries `--chroma-weight 2.0`, which
  exists to counter RGB-MSE's blind spot for desaturation). Measured:
  the `gradient` class comes back **14.2% less saturated** than the base
  decoder's output while every other class is within +-4%. That is the
  class that lost 5 dB.
- **Not PAPR.** The penalty is computed on the transmitted waveform,
  which is a function of the latents, so with the encoder frozen it has
  zero gradient with respect to anything a decoder-only run trains.
  Adding `--papr-weight` there would change nothing.

The general lesson, which cost a wrong claim: diff the run's flags
against `train.py`'s defaults before reading a per-class table as a
finding.

**`--kept-only` is a scaffold, not the endpoint** (Andrew,
2026-08-30). It refuses to touch weight-0 latents, so on a mode-A
reception it is capped at mode A *by construction*. The measurement
behind it was *iid* erasure with all three groups present, which the
decoder interpolates around; mode A is structured truncation and is a
different question. Revisit fill-in once a flow trains through the real
modem.

**And fill-in is worth real dB, in the regime that matters.** Oracle
upper bound, groups 1-2 handed to the decoder exactly on a mode-A
reception, 24 images: COCO **+0.72 dB at 12 dB rising to +3.26 dB at
-3 dB**; nonphoto -1.13 at 12 dB rising to +1.55 at -3 dB. The headroom
*grows as the channel worsens*, and A+fill beats mode C everywhere
(fill-in supplies groups 1-2 clean; mode C receives them noisy). Read
the clean column with care: mode B beats mode C clean on both sets
(COCO 27.85 against 27.35), which is the designed split -- B vs A is
quality, C vs B is robustness -- so at clean the third group is dead
weight and a clean-only reading understates fill-in badly. The bound is
also pessimistic: filled latents enter under all-ones weights, which is
off-distribution for this decoder (nonphoto PSNR *rises* monotonically
with erasure, 28.21 at weights exactly 1.0 to 29.19 at 30% erased), and
that is what makes the nonphoto column negative at high SNR. The frozen
decoder, not the fill-in, is what caps this -- an argument for the
generative decoder below. All PSNR, n=24, one checkpoint.

**Not the first of the three to build.** The latent denoiser above
answers "does a generative prior buy anything on this channel" for far
less work, against the existing checkpoint, and its answer transfers
directly. If it comes back at +0.08 dB again, this one is unlikely to be
worth a GPU-month.

## REPA: align an intermediate decoder representation to a pretrained encoder

**Goal.** An auxiliary training loss that pushes an *early* decoder
layer's features toward the features a frozen pretrained vision encoder
(DINOv2 or similar) produces for the **clean** image. A projection head
plus a cosine loss. Training-only: **zero inference parameters, zero
on-air change, no format impact, and it drops out completely** once
training is done.

**Where it comes from.** REPA (Yu et al. 2024, "Representation Alignment
for Generation") reported large convergence speedups for diffusion
transformers by aligning one intermediate layer to a self-supervised
encoder's features. FlexTok carries it — the config exposes
`intermediate_layers: [1]` and unpacks a `dec_packed_seq_repa_layer`,
i.e. it aligns layer **1 of 28**, very near the input.

**Why it is a different mechanism from the LPIPS/DISTS work, not a
duplicate of it.** Those score the decoder's *output image*, after it has
committed to every pixel. REPA supervises an *internal representation
before that commitment*, which is the point in the network where this
problem is actually hard: the decoder's input is degraded — noise,
erasures, a truncated group — and what it needs is to complete that into
a semantically coherent representation and only then render. Aligning
the early layer against features of the **clean** image, while the
decoder is fed **channel-corrupted** latents, states exactly that as a
training target. That asymmetry is the design decision here, and it is
not the setup the paper used (it aligns against the clean image because
the input is *noise*; here the input is a degraded version of the
answer).

**Why it is worth trying before anything architectural.** It is a loss
change on a branch that is already about losses. It cannot affect the
waveform, the modem, the format, the golden vectors, the ONNX export or
the app, because nothing survives to inference. Failure costs a training
run and leaves no residue.

**Honest status: unproven in this setting.** The published gains are for
diffusion transformers generating from noise. This is a 4.47M
convolutional decoder reconstructing from a degraded description, and
there is no result saying the technique transfers. Treat the speedup
figures in that literature as motivation, not as an expectation.

**Implementation frictions worth knowing before starting.**

- **Resolution mismatch is the main one.** The natural alignment point in
  the current decoder is after the first `ResBlock` pair, at the 30×40
  latent grid, while DINOv2 emits patch features on its own stride at its
  own input size. Something has to interpolate or pool, and picking that
  badly is the most likely way to get a null result that looks like the
  method failing.
- **The frozen encoder forward is the real cost**, paid every training
  step, and this project's training budget is ~3 concurrent HF Jobs. A
  small variant, a reduced-resolution forward, or aligning on a subset of
  steps are all worth pricing before committing to the large one.
- **Which layer.** FlexTok's choice (1 of 28) is very early. The
  proportionate point in a 4-stage conv decoder is right after the first
  stage; it is a hyperparameter and the paper reports sensitivity to it.

**How to measure it, given three pretrained priors are now in play.**
LPIPS, DISTS and REPA all import a pretrained model's notion of
similarity, so redundancy is the likely failure rather than harm. The
test that answers the question is **REPA added to the current
LPIPS+DISTS baseline**, same data, same schedule, same eval — not REPA
against a bare-MSE baseline, which would flatter it by measuring
"perceptual prior helps" a third time. And gate it on real material
(certificate, rendered text), per the standing rule: a semantic
alignment loss is exactly the kind of thing that improves photographs
and does nothing for a page of text.

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

## Completed: preamble detection against a steady-carrier interferer

**Mostly closed 2026-08-13**, as a side effect of the wide-acquisition
false-lock fix: `config.TEMPLATE_SCORE_THRESHOLD` also rejects a steady
tone at every level the original table named — a pure tone reads
metric 1.000 but scores only ~0.2–0.3 against the 24-carrier preamble
template. **Still open:** fix candidate 1 (keep the top-K metric peaks
and let the Golay header arbitrate) is the more direct answer to the
interferer problem and is not implemented; it cannot cost sensitivity,
because its first candidate is exactly today's argmax.

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

## Android rig control: two Icoms do not work

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

**The first trace (2026-08-23) shows correct frames going out and zero
bytes ever coming back.** `fe fe a2 e0 03 fd` -- address A2, the
IC-9700's default -- written, `read_string_generic(): Timed out 1.001
seconds after 0 chars`, every time, at 115200. `icom_get_usb_echo_off`
therefore returns `-RIG_ETIMEOUT` and `icom_rig_open` gives up with
"is rig on and connected?". What that trace *cannot* say is whether
those six bytes reached the USB endpoint, because Hamlib only ever saw
a successful socket write. `core/rig/trace.hpp` closes that gap: the
bridge now logs each direction's bytes (after the transport accepted
them, never before) and the transport logs which driver and interface
the Android USB layer chose. So the next run distinguishes:

* `-> rig 6: fe fe a2 e0 03 fd` with no `<- rig` line -- the bytes left
  and the radio said nothing. Radio-side; see below.
* no `-> rig` line at all -- our write never reached the chip, which
  would be the first evidence of a bug on this side.
* a `<- rig` line that Hamlib did not see -- a fault in the socket half.

Assuming the first, what to check, in order:

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

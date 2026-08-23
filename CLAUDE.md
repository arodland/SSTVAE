# CLAUDE.md

SSTVAE: image transmission over HF radio by sending convolutional
autoencoder latents as analog values on OFDM carriers (RADE-style).
See README.md for the waveform table and usage; the approved design
rationale lives in the plan history.

## Commands

- Run tests: `pytest` (fast, ~10 s; includes full modem end-to-end tests)
- Slow gate: `pytest -m slow` (~2 min) — the listener state machine and
  the app's transmit→receive loopback. Run it after touching `sstvae/rx/`.
- Native port: `tools/build_native.sh --test` (builds `native/`, runs
  `ctest` and `pytest --native`). See "The native port" below.
- Run the app: `tools/build_native.sh` then `native/build/sstvae-gui`
  (the app is C++; there is no Python GUI any more)
- Smoke-train: `python scripts/train.py --smoke --out /tmp/smoke`
- Full pipeline check: `sstvae_encode.py` → `sstvae_simulate.py` → `sstvae_decode.py`

## Testing the live paths without hardware

Both of these exercise the *real* code paths, which is the point — the
audio and rig bugs found so far were all invisible to unit tests.

- **Rig control:** set the app's rig model to **1**, Hamlib's dummy rig,
  and PTT, frequency readback and the whole `RigController` threading
  model can be driven for real without a radio attached. Model **2**
  (NET rigctl) against a `rigctld -m 1 -t <port>` exercises the shared-
  radio path the same way.
- **Audio loopback:** a null sink plus a *remapped* monitor, because Qt
  does not enumerate monitor sources:

  ```sh
  pactl load-module module-null-sink sink_name=null-sink
  pactl load-module module-remap-source source_name=sstvae_loop \
      master=null-sink.monitor channels=1 \
      source_properties=device.description=SSTVAE-Loopback
  ```

  Then play into `null-sink` and capture `SSTVAE-Loopback`. Unload the
  modules by index (`pactl unload-module N`) afterwards. **Pre-resample
  the file to the sink's rate** — `pw-play` converting 44.1k→48k on the
  fly cost ~4 dB of apparent SNR and sent me chasing a phantom.
- **Anything Qt with an event loop: run it under `timeout`.** A headless
  `QApplication` with `app.quit()` called from a worker thread has hung
  this project's runs; `timeout 120 uv run python ...` makes that
  self-limiting. Do not put event-loop tests in the pytest suites.

## Architecture

- `sstvae/config.py` — every constant shared between modem, channel sim,
  and training. **All waveform/latent numbers must agree through this
  module**. One carrier (`BEACON_CARRIER`) is permanently reserved for
  the beacon side-channel, so `LATENTS_PER_FRAME` (23-carrier capacity)
  no longer evenly divides `GROUP_LATENTS` (132ch model contract);
  `FRAMES_PER_GROUP` is pinned to the *pre-beacon* 24-carrier capacity
  instead (so this is a capacity trade, not a time trade — mode
  durations are unchanged), and the `DROPPED_LATENTS_PER_GROUP`
  remainder per group (~4.2%) is a permanent erasure, never transmitted.
- `sstvae/modem/` — NumPy DSP, no torch:
  - `ofdm.py` DFT-matrix mod/demod (24 carriers × 50 Hz at 950–2100 Hz;
    carriers on integer multiples of 50 Hz so the CP is truly cyclic).
    **The pilot is a minimized-crest-factor phase set, 0.99 dB envelope
    PAPR (`PROTOCOL_VERSION` 3, 2026-08-14)** — see "The pilot is three
    things at once" below. It replaced a frozen random QPSK draw at
    7.9 dB, and `CLIP_HEADROOM_DB` moved 0.5 → 0.0 in the same change.
  - `sync.py` preamble detect (lag-160 autocorrelation over a
    480-sample window, energy-floored metric), fractional + integer-bin
    CFO, template timing.
    **The preamble is four repeats, not two, and the correlation window
    is what the length buys** (2026-08-04, `PROTOCOL_VERSION` 2). At two
    repeats the metric's noise floor sat *inside* the threshold: on AWGN
    the median 5-second maximum measured 0.461 against a threshold of
    0.50, i.e. 0.47 crossings per second of noise, and the only gate
    behind it is the header — which admits 3 of 4096 Golay codewords
    (7.2e-4, measured) and **cannot be tightened without more header
    bits**, 3 valid messages in a 12-bit payload being its floor. That
    product is the false lock every few hours a live receiver sees on a
    quiet band. Four repeats at threshold 0.42 give **no crossing at all
    in 3000 s** (24M positions, peak 0.358) *and* mode A acquisition at
    −2 dB of 0.93 against 0.40 — both directions at once, which the old
    preamble could not do: raising *its* threshold to 0.64 cleared the
    false alarms too and cost 0 dB acquisition (0.82 → 0.53) to do it.
    Costs 40 ms on a 32–95 s transmission. The trap is that
    `PREAMBLE_SAMPLES` and `PREAMBLE_CORR_WINDOW` are **one change, not
    two**: a longer preamble read through the old one-symbol window has
    exactly the old noise floor and the old false-alarm rate, with every
    other test still passing, which is what `tests/test_preamble.py`
    exists to catch (through the metric's output length, since a stale
    kernel is invisible in its values). Measured on AWGN — what the
    field false locks actually trigger on is unknown, and an MPP number
    here would be reporting a ±4-sample timing criterion rather than
    acquisition.
    `acquire_blind()` is a separate, preamble-free path: matched-filters
    against the bare pilot symbol at lag-FRAME_SAMPLES, folds energy
    into 1152 phase bins across many periods, and searches CFO bins
    directly (no preamble to give phase-slope CFO) — works on a
    recording that never contains the transmission-start preamble.
  - `framing.py` per-group interleaver, Golay-coded header.
    `_TX_PERMS` truncates each group's permutation to the transmittable
    budget (dropping the beacon carrier's capacity cost); `interleave`/
    `deinterleave` operate over a whole mode's frame range,
    `slot_range_for_frame(abs_frame)` maps a single absolute frame index
    to its canonical latent slice without needing a known mode — used by
    blind decode, which never sees the header.
  - `modem.py` `Modem.modulate/demodulate`; pilot EQ with Catmull-Rom
    interpolation, EMA-smoothed sample-clock drift tracking, per-latent
    confidence weights. `demodulate_blind()` is the preamble-free
    counterpart (via `acquire_blind`): no header, so mode is unknown and
    output is always sized for mode C's full range; frame placement and
    the mode-agnostic image reconstruction both depend on the beacon
    packet decoding (frame position comes from its absolute counter, not
    from where acquisition happened) — no clock-drift tracking (needs a
    preamble phase reference), fine for the bounded windows it targets.
  - `beacon.py` the resync/callsign side-channel carried on
    `BEACON_CARRIER`: a continuously repeating Golay(24,12)-coded
    superframe (Barker-13 sync word + absolute frame counter + 8-char
    callsign + CRC-16). The counter is absolute, not modulo the
    superframe period, so decoding one full copy anywhere gives exact
    position with no dependence on where the transmission started.
    `MIN_FRAMES_FOR_SYNC` (~73 frames, ~10.5 s) is the window size that
    *guarantees* a full copy regardless of phase; shorter windows may
    still get lucky but aren't guaranteed to.
  - `golay.py` Golay(24,12), brute-force soft ML decode.
- `sstvae/hfchannel.py` — channel sim (AWGN in the `SNR_REF_BW_HZ`
  convention,
  Watterson 2-path fading presets mpg/mpp/mpd, freq/clock offset).
- `sstvae/models/autoencoder.py` — encoder (unit-RMS tanh latents,
  132ch in 3 ordered groups of 44) and decoder (takes latents ×
  weights + weight planes; handles erasures/truncation).
- `sstvae/latent_channel.py` — stage-1 differentiable channel
  (AWGN, group truncation, erasures) used by `scripts/train.py`.
- `sstvae/codec.py` — `load_codec` / `reconstruct` / `pad_to_full`.
  These used to live in the top-level `sstvae_encode.py` /
  `sstvae_decode.py` scripts; they are here so package code doesn't
  import a *script*. The scripts re-export them. Always loads on CPU.
  **The runtime backend is ONNX; torch is training-only** (see
  `docs/onnx.md`). `load_codec(path, precision=, backend="auto")` sends
  a `.pt` to `TorchCodec` (the reference implementation) and everything
  else to `OnnxCodec`, so `--model foo.pt` still works. Two things are
  deliberate: `reconstruct(codec, latents, weights)` **keeps its exact
  signature** so `rx/engine.py` needed no edit, and encoder/decoder
  **load lazily and independently** — no CLI needs both, so a
  receive-only station fetches 9 MB rather than the 21 MB pair. That
  laziness is also what lets `--model` accept a single `.onnx`.
  `--model` takes a directory, a single `.onnx`, or a `.pt` — the last
  still works but **needs torch, which the app extras no longer
  install**, so it raises a pointed `SystemExit` rather than a bare
  ImportError. `OnnxCodec` cross-checks the two parts' stamped
  `source_sha256`: an encoder and decoder from different checkpoints
  would run and produce a *silently wrong* picture, which is the worst
  failure available here. Precisions may differ freely; only the
  checkpoint must match.
- `sstvae/latents.py` — `latents_to_flat` / `flat_to_latents` in numpy.
  Same mapping as the torch statics on `SSTVAE`, which stay for
  training; `tests/test_latents.py` asserts they agree **exactly** (both
  are pure reshape, so any tolerance would be hiding something). The
  send/receive path must import this one, never `models`.

## The engines

**`sstvae/gui/` was deleted on 2026-08-01, and `sstvae/rig/` with it.**
The desktop application is `native/` — see "The native port". The
PySide6 GUI was frozen at parity (2026-07-29) and removed once CI was
publishing an installable build for all five platforms, one step ahead
of the signed release `docs/native-app.md` decision 1 originally named:
signing is an external round-trip of unknown length, and the condition
the amendment actually cared about — that an operator can get a working
app without building Qt and Hamlib from source — was already met by the
CI artifacts. `sstvae/rig/` (the rigctld TCP client) went because the
GUI was its only consumer; the native app links libhamlib in-process.

Gone with them: the `gui` extra, the `sstvae-gui` console script, and
nine test modules that only drove widgets. Two test files were
**rewritten rather than deleted**, because they used `sstvae/gui/` as
the *oracle* for the C++ port rather than testing it:
`tests/test_native_settings.py` now drives the C++ reader from a
fixture in which no field holds its default — a dropped field comes
back as its default, and no default appears in the fixture, so it
cannot survive — plus a guard test that fails when a setting is added
to the C++ and not to the fixture; and three audio parity tests in
`tests/test_native_parity.py` restate `bytes_to_mono` and
`match_device` in numpy, which is what the playback-direction test in
that file always did.

What remains is the Qt-free layer the GUI sat on, still live, still
used by the CLIs, and still the reference the native port is checked
against. Nothing in it may import Qt — the native side's equivalent
rule is enforced by `tools/check_layering.py`.

- `sstvae/rx/` — the live reception state machine, extracted from
  `sstvae_listen.py` (which is now just its CLI front end). `engine.py`
  holds `decode_loop` / `decode_loop_low_cpu` **unchanged** from the
  version the slow tests were written against — treat that logic as
  load-bearing and run `pytest -m slow` after touching it. Two seams
  were added: an `RxConfig` in place of the argparse namespace, and a
  `sink` that receives finished receptions. **Saving is the sink's job,
  not the loop's**, because an autosave checkbox may hold a picture for
  a Save button instead of writing it. `ringbuffer.py`
  adds `tail()` (cheap slice for the ~20 fps waterfall; `snapshot()`
  copies all 130 s) and `clear()`.

  **Every completion test runs against a reception retained *across*
  polls** (`_Pending`, and `Pending` in `native/core/rx/engine.cpp`),
  and that is the whole point of the record rather than an
  implementation detail. A reception stops *decoding* long before it
  stops being real: its audio scrolls out of the ring buffer, or its
  blind acquisition score falls back under `BLIND_SCORE_THRESHOLD` as
  the accumulator's evidence decays once the transmission is over.
  Every test used to be evaluated inside the branch that ran only when
  the current poll had produced a decode, so from that moment "is this
  finished?" was never asked again — the loop sat in "receiving"
  indefinitely and the picture it had already decoded was never handed
  to the sink. **The reported hang and "autosave never fires" are one
  bug seen from two ends**, since the sink is the only thing that
  saves. A poll that decodes nothing must therefore *count against* a
  reception, not be skipped over. Three things ended up load-bearing.
  The header path needs a **deadline of its own** (its known mode's
  duration past its start, where blind uses mode C's) — it had none at
  all, `frames_received >= n_frames_expected` and nothing else. The
  stall test **must not be extended to the header path** while it is
  still decoding: a spurious preamble lock in noise goes on decoding
  the same handful of frames for as long as it is looked at, and
  ending *that* on a stall turns a false lock into a saved picture of
  noise (`test_noise_produces_nothing` is what catches it) — so the
  stall clock runs on polls where the reception did not decode, plus,
  blind only, where it decoded and did not advance. And
  `decode_loop_low_cpu`'s "wait until the transmission should have
  arrived" is a wait on **something outside the loop's control**, so it
  is bounded by the audio still missing plus `end_grace`; unbounded, a
  capture that dies mid-reception hangs it forever.
  `tests/test_rx_watchdog.py` is the fast, stubbed-modem pin for all of
  this — the failure under test is *decodes stopping*, and there is no
  way to ask the real modem to stop decoding on cue, which is why the
  slow suite cannot cover it.

  **The progress bar is position, not fill**: the last frame
  successfully received over the frames expected, never latents
  received over latents expected. Only the blind path can tell the two
  apart — the preamble path decodes a contiguous prefix, so its
  `frames_received` count *is* its reach — and there a fill fraction is
  a completion percentage that is not one: the erasures that path lives
  with (a fade, or simply not having heard the start) hold it down
  permanently, so a reception already at the transmission's last frame
  reported 70% and the bar never filled. `_blind_progress` returns both
  numbers because they answer different questions — the confident
  *count* is still what the `--end-grace` stall detector watches, since
  retrospective decoding filling in frames behind the furthest one is
  real progress that the reach deliberately does not move.
  `framing.frame_of_latent()` is what makes the reach computable at all
  (the inverse of `slot_range_for_frame`, as one cached table): the
  interleaver scatters each frame across the whole picture, so a latent
  count answers "how much" and only the frame index answers "how far".
  Mirrored in `native/core/`, where `rx::blind_progress` is public for
  the same reason `poll_wait` is.
- `sstvae/tx/engine.py` — encode → modulate → PTT → play → unkey.
  **The invariant is that PTT always comes back down**: try/finally
  around the keyed region *plus* an independent `_PttWatchdog` thread
  for the case where the transmit path is wedged and its finally will
  never run. `condition_for_output` is a plain peak scale on purpose —
  `Modem.modulate` already did the envelope clipping that sets PAPR,
  and a second clip here would splatter.
- `sstvae/audio.py` — device enumeration and stream opening, both
  directions, with the 8 kHz-rejected → native-rate + `resample_poly`
  fallback. Imports `sounddevice` lazily so the module works with no
  PortAudio installed (a settings UI needs to *report* that).
- `sstvae/overlay/` — `model.py` is the document, `render.py` draws it
  with PIL. Designed so *templates* are a later UI-only change:
  coordinates are normalized 0..1 (resolution-independent) and
  `ImageItem.source` is a late-bound reference (`"last_rx"` or a path)
  rather than a pasted bitmap, so a saved template keeps meaning "the
  most recent received picture". `item_bbox` is shared with the editor
  so selection handles can't drift from what is drawn.

Two rules the deleted GUI established, which the native app inherits
and which are the reason its panels look the way they do: a composition
preview **is** `overlay.render()`'s output rather than a toolkit-drawn
imitation, so what you arrange is what goes on the air by
construction; and half duplex means transmitting suspends receive, with
a **fresh ring buffer** on resume so the tail of our own transmission
isn't decoded back as a reception.

## The native port

`native/` is the C++20 rewrite of the application (`docs/native-app.md`).
**Phases 0–1: the whole modem is ported** — `golay`, `ofdm`, `dsp`,
`framing`, `beacon`, `sync`, `modem` — and the Python suite passes
against it, including `-m slow`. Both interop directions work. Phase 2
(the headless app core) is **complete**: the codec, images, WAV I/O,
settings, the overlay document, the ring buffer, the rx and tx engines,
soundcard audio and rig control. **Python remains the normative
definition of the on-air format** — when the two disagree, Python is
right until proven
otherwise, because that is the only thing that keeps "compatible
implementation" a checkable claim.

**The codec's parity claim is different in kind, and stronger.**
Everywhere else in the port, two implementations of an algorithm agree
to a tolerance. `native/core/codec/` calls the *same* onnxruntime on the
*same* artifact, so the only variables are what we hand it and what we
do with what it returns — and those are required to be **exact**:
the encoder is bit-identical to Python's and the decoder byte-identical
on every subpixel (`tests/test_native_parity.py -m codec`). Two things
buy that, neither of them free:

- **The onnxruntime version is pinned to the Python one** in
  `native/cmake/onnxruntime.cmake`, with a sha256 per platform archive.
  "Identical" is a claim about two builds of one version; two versions
  could differ by a kernel rewrite, both be correct, and deliver a
  picture that is subtly not the one that was sent. Bump it in step with
  `pyproject.toml`, never ahead.
- **The final `* 255` is done in float32**, because numpy's is: NEP 50
  keeps `float32_array * python_int` at float32. Doing it in double
  moved 3 subpixels of 921600 across a round-half-to-even boundary.
  Round with `nearbyint` (half to even, like numpy), never `std::round`
  (half away from zero).

The codec is the only part of `native/` that downloads anything, so it
is a separate library behind `-DSSTVAE_BUILD_CODEC` (`--no-codec` in
`tools/build_native.sh`): the entire modem still builds and tests
offline. Its tests carry a `codec` marker — `-m 'not codec'` for an
isolated run, `SSTVAE_REQUIRE_CODEC=1` to turn their skips into
failures, which is what CI sets after prefetching the artifacts. That
env var exists because these are the suite's strongest checks *and* the
only ones with a downloaded prerequisite, which is exactly the
combination that rots into silently testing nothing.

**Phase 3 (the GUI) is complete**, and its exit criterion was met by
loopback on 2026-07-29 (Andrew's measurement, not mine): a picture sent
and received **native->native, native->Python, and Python->native**. The
cross-implementation pair is the one that matters — it is what makes
"compatible implementation" a checkable claim rather than an assertion,
and it exercises the on-air format in both directions through a real
soundcard rather than through a golden vector. Note what it is *not*:
loopback, not RF. The PTT timing against a physical radio is still
untested, and that is the remaining item.

`native/gui/` is the only place
QtWidgets is allowed; `SSTVAE_BUILD_GUI` is AUTO/ON/OFF like the audio
and overlay switches, but for a different reason — Qt is the app's
toolkit by design, and the switch exists so the modem, the CLI and the
parity module still build on a machine with no GUI stack, which is
every CI job but one and every headless station. It needs the codec,
Qt audio, rig control *and* the overlay renderer, each separately
optional, so ON has to name which piece is missing: "Qt6 not found"
would be a lie when the real problem is `-DSSTVAE_BUILD_CODEC=OFF`.
Panels land one at a time behind placeholders, so the window's
structure and the wiring *between* the panels — half duplex, polling
paused while keyed, last-received picture offered as a transmit inset —
are visible and reviewable before the panels themselves exist.

**The receive and transmit panels are side by side in a splitter, not
in tabs** (2026-08-01). The reference put them in a `QTabWidget` and the
port inherited it, which meant composing a picture was done blind — no
waterfall, no decode progress, no incoming preview, on a mode where you
prepare the next transmission while listening to the current one. Three
consequences worth knowing: the **waterfall became a horizontal strip**
above the picture rather than a column beside it (it was already
frequency-on-x / time-down-y, so this is a layout change and not a
widget one — but history depth follows height, so a strip holds less of
it, which is why the splitter is there); the **half-duplex pause has to
look deliberate** now that the pane stays on screen through an over,
since a stopped waterfall is otherwise the same picture as a wedged
capture; and a **last-reception card** keeps mode, callsign, SNR, frame
count and filename, because the engine wipes all of it from its shared
state two seconds after a reception and the old "Complete — SNR x" line
erased itself while the operator was still looking at the picture.

**Tabs survive as the small-screen layout** (`ui.layout`, View >
Layout, 2026-08-02), because the splitter's floor is a property of the
*arrangement* and only the arrangement can move it: measured on the
real panels with `sstvae-gui-shot --panes`, **1043 px side by side
against 545 px tabbed** *at the time*. The scroll-area idea that was
filed against this treats the symptom.

**Those numbers are long gone, and the conclusion with them**
(re-measured 2026-08-08): side by side now asks **766 x 467** against
tabbed's **342 x 464**. Everything that shrank the panels since —
`Ignored` size policies, `FlowLayout`, retiring the duplicate properties
box — came off the side-by-side figure. At 766 px it fits every screen
anyone will run this on, so `resolve_layout` will not choose tabs on any
real display and the startup log line explaining that choice will never
fire. Worse on the axis tabs were meant to help: at 900 px wide the
tabbed layout measures **49 px taller**, and its
`minimumHeightForWidth` is 755 against the splitter's 518. Tabs are
therefore a *preference* now (View > Layout), not a screen-size
adaptation — kept deliberately (Andrew, 2026-08-07) rather than retired,
but do not reach for the old numbers to justify it. Three things are
settled and worth not re-deciding. **"auto" is resolved once, at startup, against the
screen** — a live breakpoint reads like the obvious implementation and
cannot work, since while side by side is in force the splitter's own
minimum is exactly what stops the window reaching the width that would
trigger a switch away from it, so the downward transition is
unreachable. **Choosing a layout by hand ends "auto"** even when it
picks what auto would have: the next screen may be a different one.
And **the receive status line is mirrored into the status bar while
tabbed**, since tabs give back the one thing side by side buys — seeing
the band while composing.

`gui/pane_container.cpp` owns the switch, and the order in `set_mode`
is load-bearing: build the new container *first* (which reparents both
panels out of the old one), then delete the old one. The obvious
alternative — detach with `setParent(nullptr)`, delete, re-add — marks
both panels explicitly hidden, and a widget hidden that way stays
hidden when it is added to a visible layout. Two related traps, both
caught by `test_pane_container.cpp` on its first run rather than by
reading: **every `addWidget`/`addTab` hides what it reparents**, and
the thing that normally un-hides it is the parent going from hidden to
visible — which never happens on a switch, so the new container needs
an explicit `show()` or the window renders empty with nothing having
failed. And a `QTabWidget` hides its background page with `hide()`,
which is an *explicit* hide that showing an ancestor does not clear —
so rebuilding the splitter has to show both panes by hand.

**`gui/picture_box.cpp` is the receive preview, and `setFixedHeight` is
never how you pin an aspect ratio.** A fixed height is a hard
*minimum*, so a wide pane raises a floor under the whole window that
narrowing it never lowers — a ratchet. It was invisible while the
splitter kept each pane narrow and immediately fatal once a tab gives
one pane the whole width: **a 1400 px window demanded a 1405 px minimum
height**, taller than the laptop panels the tabbed layout exists to
fit. The aspect is now enforced by *geometry* — the label is positioned
by hand, not in a layout, so it imposes nothing upward — with 4:3 as
what the box asks for (`sizeHint`) and caps itself at, never as a
minimum. Given the height the picture is 4:3 and full width, as before;
denied it the picture stays 4:3 and *narrows*, which the old code could
not do at all because it simply forced the window taller instead.
**`OverlayEditor` had the identical construct and it was worse**: the
transmit panel's minimum height ran 611 px at 545 wide, 925 at 1348 and
**1274 at 1900**, so tabs would have traded 498 px of width for
hundreds of pixels of height on exactly the screens `auto` selects them
for. Same fix, and it costs nothing there either, because
`canvas_rect()` already letterboxes in both directions. Both copies are
mutation-tested (`test_picture_box.cpp`, `test_overlay_editor.cpp`) and
both assert on a *container's* minimum rather than the widget's own
size hint — the hint is a constant, so asserting on it is a tautology,
and the effective minimum is what propagates into the window.
`sstvae-gui-shot --panes` reports **both axes** for the same reason: a
width-only number measures the axis this layout improves and stays
silent on the one it can wreck. The
cap is necessarily one pass behind the width (Qt clamps incoming
geometry against the previous maximum), which `updateGeometry` closes —
and `test_picture_box.cpp` drives two passes deliberately rather than
asserting that a two-pass settle is a one-pass settle. Spare height in
the receive pane now goes to the **picture** rather than the waterfall
strip; the old stretch factors were the other way round *because* the
picture was pinned and could not use it.

**The status log is a dock, and error reporting has three tiers**
(2026-08-01). `core/log/` is a Qt-free bounded `StatusLog` plus a
rotating `FileWriter`; `gui/log_pane.cpp` is the view, backfilled from
`snapshot()` so entries logged before it existed still appear. The
tiers exist because everything used to share one overwriting label:
**errors** are sticky (`ErrorBanner`, dismissed by hand) *and* logged,
**state** gets its own indicator (PTT lamp, rig chip with error age,
latched CLIP marker), and **progress** keeps the labels that may
overwrite each other freely. The rule that produced this: `"PTT OFF
FAILED — unkey it manually"` was provably destroyed by the `"Sent"`
that followed it a moment later, on the same label. Errors must never
be written to those one-line labels — beyond losing them, a long
message inflates the layout's minimum width, which is measurable: a
700 px screenshot request came back 1204 px wide.

**The widgets live in `sstvae_gui`, a library, with `sstvae-gui` as
just `main.cpp`** — so a test can drive a widget. Most of what makes a
GUI good needs eyes, but the parts that do not have been wrong before:
the waterfall's scroll must move history *down* (an in-place row copy
is easy to reverse, and the result still looks like a moving display),
a resize must keep the history rather than blank it, and a tone must
paint at the x its frequency says. `tick()` is a slot so a test can
render one frame instead of waiting on the timer — no stopwatch.

**The overlay editor is a painted `QWidget`, not a `QGraphicsView`.**
The design doc assumed the reference's scene graph would port directly,
but the reference's own rule — *the preview is `overlay::render()`'s
output, not a Qt-drawn imitation* — makes a scene pointless here: there
is nothing to put in it but the rendered composite. A plain widget has
no second representation that can drift. Selection handles come from
`overlay::item_bbox`, the same geometry the renderer places items with,
so a handle cannot sit somewhere other than the thing it selects; and a
drag writes *normalized* coordinates, never pixels, which is what keeps
a saved template meaningful at another size.
`tests/test_overlay_editor.cpp` drives synthesized mouse events and
checks the arithmetic — an inverted axis or a dropped letterbox offset
still looks like a working editor until an item will not go where you
put it.

**`sstvae-gui-shot` renders the app's windows to PNG, headless**, so a
layout can be looked at at several sizes without a display or a human.
A tool rather than a ctest, like `sstvae-audio-check`: "is this laid out
well" has no oracle, and a stored-PNG comparison fails on every font and
theme it was not recorded with. It earned its place immediately —
it is how the clipped help text and the zero-padding bug below were both
*seen* rather than guessed at. `MainWindow` is deliberately not one of
its targets: it starts a model load and opens the rig.

**A `QFormLayout` with too little height truncates rather than
compresses**, and what goes first is the wrapped help text at the bottom
of a section. There is no default size that is right on a laptop panel
and on a large monitor, so each settings tab lives in a `QScrollArea`
(horizontal scrolling off — the width is the dialog's, so a horizontal
bar would only ever mean a label refusing to wrap). Clipping is worse
than scrolling in a way that matters: the reader cannot tell whether the
text is cut off or simply ends.

**Never set a Qt stylesheet for styling in this app; use `QPalette`.**
A stylesheet on *any* widget makes Qt wrap the application style in
`QStyleSheetStyle`, whose defaults are not the platform's — most
visibly, padding drops to zero, so every combo, spin box and line edit
gets its text jammed against the left border. One `color:` rule on a
label in the settings dialog did that to the whole window, and the
receive panel's preview background did it from another file entirely.
The symptom appears nowhere near the cause.

**A settings dialog's real bug is a field it displays but forgets to
write back**, so `test_settings_dialog.cpp` round-trips a config in
which *no field holds its default* and requires the output to be
identical. A dropped field comes back as its default, and no default
appears in the fixture, so it cannot survive. Verified by deleting one
`apply_to` line and watching it fail — a round-trip test that passes
trivially is worse than none.

**One rig mapping, in `gui/rig_config.cpp`**, shared by `AppState` and
the dialog's Test CAT / Test PTT. Two copies would let the test button
pass while the app failed, which is worse than having no test button:
it sends the operator to look at their radio instead of at the setting
that differs. The test runs on a worker thread for the same reason
polling does — "nothing on the GUI thread blocks on the rig" has no
exception for a button.

**Probing a rendered widget means knowing what else is painted on it.**
Three of the waterfall's first test failures were the test's fault, not
the widget's: the level meter is the brightest column on the pane, the
band markers are dashed lines down the full height so *no row is ever
entirely black*, and a marker's whole-height column out-totals a tone
that has painted one row. The fixes are in `test_waterfall.cpp` — probe
a single pixel in a column no overlay touches, and use a width narrow
enough that the caption is dropped.

**`core/dsp/spectrum.cpp` is the waterfall's arithmetic, Qt-free.**
`reduce_to_width` is peak-hold when shrinking, not point-sampling: the
carriers are one or two bins wide and about six apart, so taking every
k'th bin drops some outright and leaves a ragged comb — which reads as
a *reception* problem and sends the next person to debug the modem.

**The rig settings broke compatibility with the Python config, on
purpose (2026-07-29), and `CONFIG_VERSION` is 2.** The v1 shape —
`host`, `port`, `spawn_local` — described a *rigctld socket*, the one
part of rig control the native app does not have, since it links
libhamlib in-process. Hamlib model 2 ("NET rigctl") is the rigctld
client, so a remote daemon is now a model number in the same picker
rather than a parallel set of fields. The replacement is modelled on
WSJT-X's Radio tab, because that is the set a real radio needs and the
one operators already know: data bits, stop bits, parity, handshake,
forced DTR/RTS, PTT method (VOX/CAT/DTR/RTS) with **its own port**, and
an optional USB/PKT-USB mode on connect. `"default"` everywhere means
*do not set the token*, leaving the backend's own value — which is why
an unrecognized setting falls back to Default rather than erroring: it
declines to force a wrong value onto a radio. Two migration details
earn their code: v1's dead keys are listed as *known* so an old config
does not read as four typos, and `model` accepts the v1 string as well
as a number — the operator did not do anything wrong, so migrating must
be quiet. The `device` key is reused rather than a new `port`, because
v1's `port` was an integer and reusing that name would make every
migrated file report a type error.

**Hamlib's `rig_set_conf` token names are not guessable and must not be
guessed.** `rig_token_lookup` returns `RIG_CONF_END` for a name it does
not know and `set_conf` then does nothing — a misspelling is silent, so
the radio simply ignores the setting. The authoritative list is
`src/serial_cfg_params.h` and `src/conf.c` in the pinned tarball;
values are case-sensitive combo strings (`"XONXOFF"`, `"Hardware"`,
`"ON"`/`"OFF"`, `ptt_type` of `"RIG"`/`"DTR"`/`"RTS"`/`"None"`). Read
them there, not from memory.

A trap in the ORT C++ API, since it crashes rather than warns:
`GetInputTypeInfo()` returns a `TypeInfo` **by value** and
`GetTensorTypeAndShapeInfo()` is an unowned view into it. Binding only
the view leaves it dangling.

**Above the modem, identical behaviour is not required** (decided
2026-07-27). The on-air format is normative and stays exact; the app
around it may use native idioms and improve on the reference. Two
places do: `native/core/rx/engine.cpp` takes its decoder as a
`std::function` seam where Python imports `codec.reconstruct` directly,
and `SharedState` exposes only `get`/`update` rather than Python's
"here is a mutex, remember to take it".

That seam is load-bearing, not cosmetic. It keeps the decode loop in
`sstvae_core` rather than the codec library, so **the whole receive
state machine builds and is tested with `--no-codec`** — no
onnxruntime, no download — and `native/tests/test_rx_engine.cpp` drives
it with a stub decoder. `pad_to_full` moved to `core/latents/` for the
same reason (it is a memcpy, not an inference); `codec.hpp` re-exports
it, so `codec::pad_to_full` still resolves. The loop's decisions are
where the duplicate-picture and ended-early bugs live and they have no
oracle in the golden vectors, so this is the one part of the port whose
tests had to be written rather than inherited.

**The transmitter's guarantee is doubled on purpose.** `core/tx/` keeps
the reference's rule that PTT always comes back down, by a scope guard
*and* an independent `PttWatchdog` thread. The watchdog is not belt and
braces: the scope guard only runs if control returns, and the failure it
exists for is the one where control does not. It is in the **header**
rather than hidden in the .cpp so it can be tested directly — reaching
it through a `transmit()` that returns normally would be testing the
wrong thing, and its real timeout is lead + duration + tail + 15 s.
`TxConfig::watchdog_margin_s` is a field for the same reason
`Modem::modulate` takes `clip_headroom_db`: the reference's tests patch
a module constant, which a compiled-in one cannot offer.

**Audio is split at the device boundary, and the split is the design.**
`core/audio/audio.hpp` is Qt-free and holds everything with logic in it
— `resample_ratio`, `StreamResampler`, the sample-format conversions,
`match_device` — because *every* audio bug this project has had lived
there rather than in the code talking to the driver, and all of them
were found against a fake device. `core/audio/qt/` is then only
enumeration and moving bytes. It is a **separate library**
(`SSTVAE_BUILD_QTAUDIO`, AUTO/ON/OFF) so the modem, codec and both
engines still build and test on a machine with no Qt at all;
`check_layering.py` enforces that nothing else under `core/` includes Qt
Multimedia. Two departures from the reference: capture runs on **its own
thread with its own event loop** (Python drains from the GUI thread,
which is the same shape as the hazard that cost 5 dB), and the C++ mixes
multichannel float down in double where numpy's `.mean` stays in float32
— a ~3e-8 difference, which is why that one parity test is the only
audio one not held to 1e-12.

**`sstvae-audio-check --loopback` is the soundcard path's only real
test**, and it is a tool rather than a ctest because it needs a device.
The recipe (null sink + *remapped* monitor, since Qt does not enumerate
monitor sources) is in `native/apps/sstvae_audio_check.cpp`. Measured
through it: mode A, 220/220 frames, callsign recovered, 27–29 dB, with
the device at 48 kHz so the capture resampler was in the path. CI has no
audio device, so its `qtaudio` job compiles the layer and runs
enumeration only — with `SSTVAE_BUILD_QTAUDIO=ON`, not AUTO, because a
job whose purpose is to compile that file must fail if it did not.

**The engines are the port's only concurrent code**, so CI runs a
**ThreadSanitizer** job over `rx_engine`, `tx_engine` and `ringbuffer`
(a separate job: TSan and ASan cannot be combined). They make claims
about what may run concurrently — the audio callback never blocks, the
transmitter's `message_` is only written outside the playing window —
and those are the claims that stay true right up until someone adds a
field.

**Build sanitizer jobs at `-O2`, not `-O0`.** The sanitizers are not
what makes an instrumented build slow; the missing optimizer is.
Measured on this suite: **670 s at `-O0` against 90 s at `-O2`**, the
same seven tests, with ASan still reporting a planted
heap-buffer-overflow with a full symbolized stack (upstream recommends
`-O1`/`-O2` with `-fno-omit-frame-pointer`, which `SSTVAE_SANITIZE`
sets). At `-O0` the rx engine's tests did not merely run slowly, they
**timed out** — and the thing that expired was a deadline inside the
test, i.e. a latency assertion that had smuggled itself in as a
watchdog. Two rules came out of that: a watchdog belongs at several
times the measured worst case, never at "about enough"; and the hard
bound on a wedged test is a ctest `TIMEOUT` property, because when the
CI runner kills a job there is no ctest output left to say which test
it was.

**A `printf` is not a diagnostic for a hang — ctest holds a test's
output until the test finishes.** Instrumenting `test_rig_hamlib` with
per-step prints produced exactly as much information as no
instrumentation at all, because a wedged test never reaches the point
where ctest flushes what it captured. What works is a watchdog *inside*
the process (`check::Watchdog` + `check::current_step`): it names the
step and then calls `std::_Exit`, deliberately skipping static
destructors, because unwinding is itself somewhere a wedged library can
hang and a watchdog that can hang is not one. Sized at ~90x the measured
runtime, so expiring can only mean wedged — not "slower than I guessed",
which is the failure the `-O0` episode above records. It and the ctest
`TIMEOUT` answer different questions and both are kept: watchdog fires ⇒
a named step is stuck; ctest timeout *with* the suite's `ok:` line in
the captured output ⇒ everything finished and the wedge is in process
teardown; ctest timeout with nothing at all ⇒ it never reached `main`.

**Never include `<windows.h>` from a widely-included header.** It
defines `min` and `max` as macros, so every later `std::max(a, b)`
becomes a syntax error (C2589) — pulling it into `check.hpp` for one
call to `SetErrorMode` broke two unrelated test files on MSVC and
nothing anywhere else. `NOMINMAX`/`WIN32_LEAN_AND_MEAN` only work until
something includes a Windows header first, which is a constraint on
include order that nothing checks; declaring the one function by hand
has no such requirement. The `SEM_` constants are spelled as literals
for the same reason — redefining those names would break any TU that
*does* include the real header.

**A mingw-w64 cross compiler checks this class locally**, unlike
`check_includes.py`, which only finds missing headers:
`x86_64-w64-mingw32-g++ -std=c++20 -I native/tests -c` over a probe that
includes `<windows.h>` *first*, plus an `#ifdef max` `#error`, proves
both include orders and that no macro leaked. Seconds, against a
Windows CI job's several minutes. **Wine runs the Windows binaries
too** — `wine rigctl.exe -l` and `wine rigctl.exe -m 1 f` exercise the
bundled DLL's load path and the dummy rig from this machine, and
`x86_64-w64-mingw32-gcc` + wine settled the pthread-shim sizes
(`pthread_t` and `pthread_mutex_t` are both 8, matching the shim) by
measurement rather than by reading a header. Wine reimplements the
loader, so a *pass* there is suggestive and not proof; a failure would
have been conclusive.

**A `.lib` in a directory called `gcc` is a MinGW import library, and
MSVC must not be given one.** Hamlib's Windows zip ships
`lib/gcc/libhamlib-4.lib` (a GNU `ar` archive of dlltool stubs) and
`lib/msvc/libhamlib-4.def`. Linking the first from MSVC **succeeds** —
every symbol resolves — but the linker cannot build a valid import
directory out of GNU-convention import members and does not say so, and
the executable then dies at load with `STATUS_DLL_NOT_FOUND`
(0xC0000135). Generate the import library from the `.def` with
`lib.exe /def: /machine:x64 /name:libhamlib-4.dll` instead; `/NAME` is
required because the `.def` has no `LIBRARY` statement. The structural
difference is one `.idata$2` import-descriptor member, which the gcc
archive has none of — checkable from Linux with `llvm-lib` and
`llvm-nm`.

**That failure mode is why the Windows job now runs the rig test once
outside ctest.** Load-time failure happens *before* `main`, so there is
no output on any stream, no test framework has run, and an in-process
watchdog cannot fire — it is byte-for-byte identical in a CI log to a
deadlock, and was diagnosed as one for several rounds. `dumpbin
/dependents` is the tool that shows it: a dependency listed as `(null)`
with `libhamlib-4.dll` absent from the list entirely. Assert the exit
code, and print the dump next to it so the answer arrives with the
failure rather than a round later.

**Windows DLLs go beside the executable, not on `PATH`**
(`sstvae_hamlib_copy_runtime`). Windows always searches the .exe's own
directory first with no environment involved, it is the layout the
installer needs anyway, and the failure mode it retires is the worst
available: an unresolved import stops the process *before* `main`, so
there is no output on any stream, no exit code anyone sees, and nothing
to tell it apart from a deadlock. Copy all of them — upstream's build
carries libusb, libgcc and libwinpthread, and a missing transitive
dependency fails exactly as invisibly as a direct one.

**On Windows a crash and a deadlock look identical in a CI log**, and
that is worth defusing rather than diagnosing twice. An unhandled
exception raises Windows Error Reporting and a CRT assert opens a
message box; on a headless runner both block forever with an empty
stderr, so a crash gets investigated as a hang. `check.hpp`'s
`report_crashes_instead_of_prompting()` routes them to stderr and lets
the process die. No-op on the other two platforms.

**Hamlib's own trace is on for the rig tests** (`SSTVAE_HAMLIB_DEBUG`,
which also exists for operators' bug reports). ctest discards a passing
test's output, so it costs nothing until something fails — and then the
log already says how far `rig_open` got and which CAT command the rig
refused, rather than that needing another CI round to find out. When the
library owns the serial port, "quiet" and "unfalsifiable" are close
together.

**`tools/check_includes.py` catches on Linux what would otherwise only
fail on MSVC**: a `std::` name used without its header. libstdc++ and
libc++ pull in far more than they promise (`<vector>` happens to give
you `std::count_if`), so a missing `#include <algorithm>` builds
cleanly on two of three platforms. That cost two CI rounds before the
check existed, and finding it needs the platform least likely to be in
front of you. It follows project headers, so a .cpp that gets
`<vector>` from its own .hpp is fine — that is a real guarantee, unlike
one standard header happening to include another. Deliberately not
include-what-you-use: no extra dependency, and it only reports the
direction that breaks a build. It is a CI gate, unlike
`freeze_format_constants.py --verify`, because here regenerating *is*
the right fix.

**Five packages, and macOS x86_64 is the awkward one.** CI builds
linux-x86_64, linux-aarch64, macos-arm64, macos-x86_64 and windows-x64.
The Intel slice is **cross-compiled on an Apple-silicon runner** (there
is no Intel runner any more) and **tested through Rosetta**, so ctest
and the packaged-app check run on the artifact that ships. Three things
it needs that nothing else does:

- **onnxruntime has no macOS x86_64 build after 1.22**, so that one
  platform pins 1.22.0 while everything else is on the Python-matched
  version. This is the single accepted departure from "the same runtime
  version, two builds" that the codec parity claim rests on — taken
  deliberately (2026-07-29) because what must match between stations is
  the *model*, which is published and identical, and a runtime
  difference lands as noise under the channel's. arm64 macOS is
  untouched and stays exact. Label that artifact as the lower
  compatibility tier, the way the int8 ones are.
- **The onnxruntime version is per platform now, so anything
  reconstructing the library's filename must use the *resolved* version**,
  not `SSTVAE_ONNXRUNTIME_VERSION`. Getting that wrong downloads the
  right archive and then looks for a file that was never in it. There is
  a glob fallback, which also makes an unpacked `SSTVAE_ONNXRUNTIME_DIR`
  work at any version rather than only the pinned one.
- **`CMAKE_OSX_ARCHITECTURES` means nothing to autotools.** Hamlib needs
  `--host` and `-arch` in CFLAGS or it builds for the runner and the link
  fails with an architecture mismatch far from its cause. It also cannot
  build fat, so `hamlib.cmake` refuses a multi-architecture request
  rather than emitting a thin library inside something that looks
  universal — which is why macOS ships two downloads and not a
  universal2 binary.

**`pytest --native` cannot run on a cross build**: the extension module
would be built for the target while the runner's Python is native. So
the x86_64 slice is not parity-checked against Python — the one real gap
in that platform's coverage, and worth saying out loud rather than
assuming the green tick covers it.

**Pin the packaging tools too, and do not believe a runner ships one.**
appimagetool and NSIS are both fetched and sha256-checked, like
onnxruntime and Hamlib. `makensis` is **not** preinstalled on
`windows-latest` — this script asserted it was, in a comment *and* in its
error message, and one CI round said otherwise. `choco install nsis`
would work and was declined: it makes the version of a packaging tool a
property of a package feed's current contents, which is the thing pinning
exists to prevent. Two extraction traps came with it, both platform
folklore rather than anything a test would find: Git Bash has no `unzip`,
**and its `tar` is msys2 GNU tar, not the bsdtar Windows itself ships**,
so `tar -xf` on a zip fails with "this does not look like a tar archive"
— use `powershell Expand-Archive`.

**The Windows installer is testable from Linux, and was.** `wine
makensis.exe` compiles `installer.nsi`, and the result silently installs,
registers and uninstalls inside a throwaway `WINEPREFIX` — checking
`File /r` recursion into subdirectories, the Start Menu shortcuts, the
Add/Remove Programs block and that an uninstall leaves nothing behind.
Seconds against a Windows CI job's minutes. Same caveat as the rest of
the Wine work here: a pass is suggestive, a failure conclusive.

**Packaging is two scripts, and the split is what makes it usable.**
`tools/package_app.sh` stages a runnable tree (Qt, Hamlib, onnxruntime,
the freedesktop files); `tools/make_installer.sh` wraps that same tree in
an AppImage, a `.dmg` or an NSIS setup `.exe`. Building and *running*
needs none of the second script's tooling, which is the whole reason they
are separate — only whoever produces a download needs `hdiutil`,
`appimagetool` or `makensis`. Both outputs are published per platform:
the portable archive needs no administrator, the installer gives a Start
Menu entry and an uninstaller. **The installer step is not gated on a
tag**, because an installer whose first exercise is the release is three
platform-specific tools with no history of working. Deliberately **not
CPack**, which was the design doc's plan: CPack packages what `install()`
rules install, so adopting it means teaching CMake to install Qt —
capturing `windeployqt`/`macdeployqt` output on two platforms and
reimplementing them on the third — to gain plumbing for containers that
are three lines of shell each.

**One icon source, three attachment mechanisms, and a runtime one that
reaches none of them.** `native/packaging/sstvae.svg` is the only
hand-authored copy; `tools/gen_icons.py` rasterizes the `.ico`, the
`.icns` and the freedesktop PNGs, **generated and committed** so no build
needs image tooling. It is deliberately *not* a CI staleness gate, unlike
`config.hpp` and the golden vectors: the check would be a byte-comparison
of rasterized output, and librsvg's antialiasing is not promised stable
across versions, so the gate would fail on a librsvg upgrade with no icon
having changed. The three mechanisms are not interchangeable — Windows
reads a resource compiled into the `.exe`, macOS reads
`CFBundleIconFile`, Linux reads the `.desktop` file and looks the name up
in the icon theme — and all three are drawn by the OS *without asking the
process*, so `QApplication::setWindowIcon` is a fourth thing, not a
substitute. It is set as well, from a `.qrc`, for the X11 window icon and
the Wayland fallback; `setDesktopFileName` is what lets Wayland match a
window to its launcher instead of showing a second nameless taskbar entry.
Rasterize each size from the vector rather than downscaling one large
PNG: at 16 and 32 pixels a reduction of a 1024px render is a grey blur.

**The icon is licensed artwork and the repository's LICENSE does not
cover it.** Andrew holds a license to use it in this application; it is
**not** sublicensed to anyone who receives the source, so a fork or a
redistributed package must replace it. `NOTICE` at the root is the
exhaustive file list — the SVG plus the seven files generated from it,
which are derivative works and equally restricted — and every one of them
carries a REUSE sidecar (`<file>.license`,
`LicenseRef-SSTVAE-Branding`). `tools/gen_icons.py` **writes the sidecar
beside each file it generates**, deliberately: adding a size later would
otherwise drop an unlabelled non-free file into a tree whose root LICENSE
says Artistic-2.0, and the first thing a packager or a compliance scanner
does is read that. `package_app.sh` ships `LICENSE` and `NOTICE` in all
three packages for the same reason — a package without the NOTICE makes a
claim about the icon that is not true. A `SSTVAE_BRANDING` switch with a
free placeholder, so a redistributor need not edit files at all, is
specified in `docs/todo.md` and not implemented.

**Model artifacts: plain HTTPS to the Hub, and our own cache.** The
design doc said `QNetworkAccessManager`, and that part stands, but the
native app deliberately does **not** share `huggingface_hub`'s cache.
Reading it would be easy; *writing* it means reproducing an
undocumented internal layout — `blobs/` keyed by etag,
`snapshots/<commit>/` symlinked into them (copied on Windows),
`refs/main`, and the locks around it — and a near-miss corrupts a cache
another program owns. The price is that anyone running both the Python
tools and the native app downloads ~9–21 MB twice; worth it to keep the
failure mode "an extra download" rather than "a broken
huggingface_hub". Cache lives at `SSTVAE_MODEL_CACHE`, else the
platform cache dir + `sstvae/models`.

**The Hub's 302 carries the checksum.** `x-linked-etag` on the redirect
is the LFS object's sha256 — verified against the published decoder,
byte for byte. So `qt_fetcher` follows redirects **by hand**, because
Qt's automatic following would hide the response carrying it, and the
artifact is checked against a hash the server stated *before* sending
the bytes. Downloads land as `<name>.part` and are renamed only after
that check: a truncated file left in the cache would be found by
`find_cached` on the next run and handed to onnxruntime, failing a long
way from its cause.

**Only the download needs the network; nothing else does.**
`checkpoint::resolve_onnx` and the cache lookup are path arithmetic in
`sstvae_core`, and the downloader is a `Fetcher` seam in a separate
library — so a build with no Qt still honours `--model` and still uses
a warm cache, which is every case but a first run. Verified end to
end: empty cache → fetches only the *decoder* (per-part laziness
intact) → 220/220 frames; and with the network blocked, dropping the
file into the cache directory by hand decodes identically. **The
offline message is a deliverable, not a nicety** — a fetch failure that
was rethrown unchanged silently dropped it, which is a caught
regression with a test of its own.

**Rig control is a re-derivation, not a port, and the design doc says
why.** The deleted `sstvae/rig/rigctld.py` talked to a `rigctld` child
over a socket because the SWIG Hamlib bindings live in the system
site-packages where a virtualenv cannot see them — a *Python packaging*
constraint with no C++ equivalent. So `native/core/rig/` links
`libhamlib` in-process and the socket client, the redial logic, the
`rigctld` spawner and the `rigctld -l` column parser were never
translated (`rig_list_foreach` gives a struct, which cannot have the
silently-dropped-row bug that parser had). Sharing a
radio with WSJT-X still works: Hamlib **model 2** speaks the rigctld
protocol as a *client*, so it is one more entry in the same picker.
What is given up is crash isolation — a backend segfault now takes the
app down — and that was accepted in `docs/native-app.md`.

**The property that survives is the one that matters**: nothing on the
GUI thread ever blocks on the rig, and keying is never stuck behind a
stale poll. One backend on one worker thread; PTT is priority work, so
worst-case keying latency is *one in-flight operation* rather than a
queue drain (which retires the reference's separate-PTT-socket trick —
that existed to dodge contention the socket layer itself introduced);
polling suspends while transmitting; and **`stop()` detaches rather than
joining**, expressed by the worker co-owning its session through a
`shared_ptr` so the departing thread runs the destructor that closes the
handle. Joining would inherit exactly the timeout being escaped.
`RigController` has no external dependency at all and is tested against
a backend that accepts and never answers, so the part that can be wrong
is covered on a machine with no Hamlib.

**Hamlib is pinned and bundled, not taken from the system**
(`native/cmake/hamlib.cmake`, 4.7.2, sha256 per artifact — same shape as
the onnxruntime pin). Its public API moves between minor releases:
Ubuntu 24.04 ships 4.5.5 where a config token is `token_t`, renamed
`hamlib_token_t` in 4.6, so the backend built locally and failed on CI.
Version `#if`s would have made "which radios work" a per-platform
property. Built from the release tarball on Linux/macOS, taken from
upstream's prebuilt zip on Windows (it ships an MSVC import lib beside
the MinGW DLL). **Dynamically linked** because Hamlib is LGPL-2.1+, the
same reasoning as Qt. `-DSSTVAE_HAMLIB_SYSTEM=ON` for distro packagers,
who then own the >= 4.6 requirement.

**On Windows, nothing may dereference a `RIG*`.** `hamlib/rig.h`
includes `<pthread.h>` unconditionally — upstream's own comment says
"For MSVC install the NuGet pthread package" — and MSVC has none, so
`native/third_party/msvc-pthread/` supplies the two types
(`pthread_t`, `pthread_mutex_t`) that `struct rig_state` needs. Those
sizes are deliberately **not** load-bearing: if they disagreed with the
winpthreads the bundled MinGW-built DLL carries, every field after the
first mutex would sit at the wrong offset, silently. So
`description()` goes through `rig_get_caps_cptr(model, ...)`, which
takes a model number rather than a pointer, and the only struct read
through is `struct rig_caps` — which has no pthread members. The result
is that no struct layout is relied on at all, which is what makes the
shim safe rather than a gamble.

**Rig control on Android: Hamlib over a loopback socket** (2026-08-22).
`core/rig/transport.hpp` is the byte pipe a phone can open, `bridge.*`
presents it to Hamlib as `127.0.0.1:<ephemeral>`, and `bridged.*`
composes the two behind a `BackendFactory` seam -- all three in
`sstvae_core`, so the mechanism is tested with fakes in a `--no-rig`
build. `core/rig/android/` is the JNI half over `SerialBridge.java`
(USB host API + vendored `usb-serial-for-android`; Bluetooth RFCOMM on
the SPP UUID). Six things are settled and worth not re-deriving:

- **The presence of a transport selects a bridge, never the device
  string.** A USB id like `usb:1a86:7523` has no slash and does not
  start with `com`, so Hamlib's own rules read it as a *hostname* -- the
  first draft branched on that and skipped the bridge for exactly the
  devices it exists for. `is_network_device` survives as validation of a
  typed host and matches `parse_hoststr` case for case.
- **DTR/RTS keying cannot go through Hamlib here.** `ser_set_dtr` is a
  `TIOCMSET` ioctl on what is now a socket, so it fails at the moment
  somebody presses transmit. The line is driven on the transport and
  Hamlib is set to `ptt_type` "None" (`PttMethod::Vox`). RFCOMM has no
  modem lines at all, so those methods are not offered over Bluetooth.
- **`Default` line settings stop meaning anything by themselves.**
  Hamlib applies per-rig defaults in `serial_open`, which a network port
  never reaches, so `rig::serial_defaults(model)` reads them out of
  `struct rig_caps` -- `serial_rate_max` included, whose own `rig_init`
  comment says "fastest !" -- and `resolve_serial_params` fills them in.
- **Hamlib's three control-line conflict checks are unreachable** (they
  sit inside `if (rp->type.rig == RIG_PORT_SERIAL)`) and are re-derived,
  or a config a desktop refuses at connect time becomes a radio that
  will not key on a phone.
- **Loopback only, as a security property.** The far end is an
  unauthenticated path to a transmitter; the bind is `127.0.0.1` and the
  port is ephemeral. One thread per direction, so an idle link costs no
  wakeups.
- **Hamlib's trace needs a sink, and Android is why**
  (`rig::set_debug_sink`, 2026-08-23). `SSTVAE_HAMLIB_DEBUG` raises the
  level and the library writes to *stderr*, which a phone discards, so
  the one artifact that answers "how far did `rig_open` get and what
  did the radio refuse" was unreachable on the platform where rig
  control is newest. A `vprintf_cb_t` trampoline formats and splits
  into whole lines (a line is not a call -- `dump_hex` emits one line
  in several); Settings > Rig control > Log rig traffic keeps 600 of
  them and mirrors to logcat, off by default. **The callback is
  registered only while a sink exists**: `rig_debug` writes to stderr
  *or* to a callback, never both, so one left permanently registered
  swallows the trace for every desktop run and every test -- silently,
  a swallowed trace and a quiet library being the same output.
  `test_rig_hamlib.cpp` captures stderr and requires it back, and that
  is the assertion with teeth (mutation-tested; the other two survive
  registering unconditionally). Hamlib's own `#ifdef ANDROID`
  logcat branch is not an alternative: `configure.ac` uses `ANDROID`
  only as an automake conditional, never as a define, and never links
  `-llog`.
- **`core/rig/trace.hpp` is the other half of that trace, and it is
  the half that says whether the bytes left.** Hamlib's view of a
  radio that never answers is "wrote six bytes, read nothing" --
  identical to what a bug in our own bridge would produce. So the
  loopback bridge and the Android transport write to a second,
  Hamlib-free sink in `sstvae_core` (so a `--no-rig` build still tests
  every line of it), and the app installs both into one ring. It logs
  what reached the transport **after** the write and never before (on
  the way in it would say "delivered" for a write that threw --
  `test_rig_bridge.cpp` pins that placement), what came back, the
  resolved line settings, and `SerialBridge.describeLink`'s account of
  which driver `UsbSerialProber` picked and which interface of how many
  it claimed. Off costs one relaxed atomic load, which is why the calls
  live in the byte pump unconditionally.
- **`Default` line states mean *asserted*** (2026-08-23). **Hamlib
  never raises DTR or RTS** -- `src/rig.c` says so outright: *"Needed on Linux
  because the serial port driver sets RTS/DTR on open - only need to
  address the PTT line as we offer config parameters to control the
  other (dtr_state & rts_state)"* -- so it relies on the OS, drops only
  the PTT line, and a desktop presents a radio with both lines high. A
  bridged transport reaches no `serial_open`, and
  `usb-serial-for-android` deasserts both in `openInt()`. So
  `resolve_serial_params` now resolves them too (Default = high;
  explicit `dtr_state`/`rts_state` honoured; the PTT line always low,
  which is `rig_open`'s own single exception), and `openUsb` applies
  them **before** `setFlowControl` -- the CP210x SET_FLOW structure
  encodes each line's mode and the library builds it from the driver's
  current fields, so setting the lines afterwards leaves the chip's
  flow configuration describing the old state. RFCOMM is not offered
  them at all, having no modem lines. This is desktop parity and
  correct on its own terms; it is **not** thought to be the Icom fix
  (Andrew): CI-V has no flow control and an Icom uses those lines only
  for PTT, CW and RTTY keying.
- **A K4 works over USB and two Icoms do not** (reported 2026-08-23,
  **open**). The trace narrowed it to one gap and no further: the app
  writes a correct frame (`-> rig 6: fe fe a2 e0 03 fd`) to a
  **CP2102N** at 19200 8N1 flow=none, one vendor-class interface, two
  endpoints, `Cp21xxSerialDriver` port 0 of 1 -- and nothing ever comes
  back, while `rigctl -m 3081 -r /dev/ttyUSB0 -s 19200` on the same
  cable answers instantly. Every control transfer returns 0 and the
  bulk write returns 6, so the chip is enumerated, configured and
  accepting bytes. **What no software here can see is whether the UART
  actually clocked them out**, which is where code reading stops and a
  hardware A/B has to take over. What was ruled out along the way, so
  it is not re-derived:
  the byte path is length-counted end to end and cannot truncate a
  binary CI-V frame; Hamlib's POSIX `port_read_generic`/`port_write`
  have no port-type branch (the ones that exist are Win32 serial);
  `network_flush` is a `FIONREAD`-guarded drain; `rigs/icom/` has no
  port-type conditional; and `serial_defaults` picks
  `rig_caps.serial_rate_max`, the same field `rig_init` uses, so a
  bridged rig runs at the speed that rig runs at on a desktop; and
  every Icom declares `RIG_HANDSHAKE_NONE`, so no flow control is
  holding the chip's transmitter off. **The one that looked ruled out
  and was not** is the control lines: both `FtdiSerialDriver` and
  `Cp21xxSerialDriver` deassert DTR and RTS, so the K4 demonstrably
  CATs with them low -- but that only shows the *transport* works that
  way, not that this *radio* does, and treating one radio's tolerance
  as a general fact is what kept the real difference hidden for a
  round. Two differences from the Linux `cp210x` driver survive and are
  the shortlist if the A/B points at the chip: the vendored library
  writes `SET_FLOW` as **16 blind zero bytes** where Linux does
  `GET_FLOW`/modify/`SET_FLOW`, preserving `ulXonLimit`/`ulXoffLimit`;
  and it writes the requested baud raw where Linux applies
  `cp210x_get_actual_rate()` for a CP2102N (a no-op at 19200, which
  divides 48 MHz exactly). Linux also carries erratum **CP2102N_E104**
  -- firmware <= 0x10004 reads `ulXonLimit` as `ulFlowReplace`, so it
  declares flow control unsupported on those parts. None of this is
  confirmed to matter here. The two
  radio-side settings to check first are Icom-specific and invisible
  from here: the **CI-V USB baud rate** (a separate menu item,
  defaulting to an Auto that is not reliable on every model) and
  **CI-V USB Echo Back**, which `icom_get_usb_echo_off` probes at open
  and gets exactly one attempt at, `rig_open` having set `retry` to 0.
- **`bridge.cpp`/`bridged.cpp` are not built on Windows** (2026-08-23),
  and the tests follow the source rather than being skipped. Windows has
  COM ports, so nothing there constructs a bridge -- what a Windows
  build compiled was a Winsock translation of a file only POSIX runs,
  whose green test said nothing about the one Android runs. It was also
  wrong: `stop()` depends on `shutdown()` waking a thread blocked in
  `recv()`, which POSIX guarantees and **Winsock does not** -- a blocked
  `recv` there simply does not return, and Microsoft warns against
  `closesocket` concurrently with a blocking call, so a correct port
  means making the data path non-blocking and polling it. `rig_bridge`
  hung for the full 120 s of its own watchdog in CI before that was
  understood. Linux and macOS still build and test it, on the same POSIX
  calls Android uses.
- **`Session` owns the `RigController`, and never destroys it.** An
  over is 32-95 s of committed airtime and the thing that brings PTT
  back down cannot be destroyed by a rotation -- nor by the operator
  switching rig control off mid-over, which is what `stop_rig()`
  destroying the controller used to do: `ptt_function()` captures the
  *controller*, so the engine was left calling into freed memory at
  unkey. Created once, `stop()`ped, never destroyed. That is also what
  makes reconnection safe, since `start()` supersedes a session in place
  and a `Ptt` handed out earlier keys the new backend
  (`test_rig.cpp`, mutation-tested). `TxEngine` needed no change: it
  takes a `Ptt`, the shape `docs/android.md` predicted CAT would arrive
  in.
- **Reconnection is app-level, and "is the device back?" is why.**
  `RigControl::maybe_reconnect` rebuilds the session when the radio
  stops answering and `has_permission(device)` says it is reachable
  again -- false for a device that is not there, so one call answers
  both halves. An absent device does not spend the backoff, so
  replugging is immediate; the backoff is for attempts that reached the
  radio and failed. **A published frequency is the only proof the radio
  answered**: `RigController::running()` means a session is configured
  and said "Connected" with the cable out.

**The Hamlib NDK cross-build and the Android Java have never been
compiled** (`native/cmake/hamlib.cmake`, `SerialBridge.java`) -- written
in a session with no NDK and no reachable `dl.google.com`. Two guards
make a first failure diagnosable: the configure step names the missing
compiler wrapper, and the install step **refuses a versioned SONAME**,
because Android's packager takes only files named exactly `lib*.so` and
would drop `libhamlib.so.4` from the APK without comment, leaving a
`dlopen` failure before `main` -- no output on any stream, and
indistinguishable from a deadlock. `-DSSTVAE_ANDROID_RIG=OFF` drops CAT
and keeps the app; the transport, the rig screen and the settings still
build, so it is one flag rather than a fork.

**Hamlib's own poll thread is turned off** (`poll_interval` = 0).
`rig_open` otherwise starts one, defaulting to 1000 ms, that issues CAT
commands for transceive emulation — which directly contradicts what
`RigController` is for. It exists to keep one command in flight and to
guarantee keying never waits behind a status read, and it can do
neither if the library is talking to the same serial port behind its
back.

**Do not end the process with a rig worker still inside libhamlib.**
`stop()` detaches by design, so at exit a worker may be mid-`rig_close`
— which joins Hamlib's internal threads. On Windows, teardown holds the
loader lock and a thread cannot exit while it is held, so that join can
block forever; Linux and macOS have no equivalent, which is why it
showed up as one platform's test running for minutes. `wait_for_shutdown()`
is for exactly one caller — whatever is about to end the process — and
`stop()` still never waits.

**And never link a Hamlib *data* symbol on Windows.** `hamlib_version2`
is an exported variable, and MSVC cannot import data from a DLL without
`__declspec(dllimport)`, which Hamlib's headers emit only when the
consumer defines `DLL_EXPORT` — a name far too generic to want in a
translation unit. Functions have no such problem, because the import
library thunks them; that is why exactly one symbol failed to link on
Windows while Linux and macOS were clean. `rig_version()` is the
function form and is what `hamlib_version()` calls.

Two traps in the source build, both of which cost time: the tarball's files
share one mtime, so `make` re-runs `aclocal` and fails without the exact
automake the release was rolled with (Hamlib has no
`AM_MAINTAINER_MODE`) — the generated files are re-stamped in dependency
order first. And the install tree lives under `FETCHCONTENT_BASE_DIR`
rather than the build directory, so whatever caches that caches the
*built* library: 26 s cold, 0.26 s warm, against a CI that discards its
build tree every run.

**A CMake `if()` on an unset variable is silently false.** `_want_rig`
was defined *after* `add_subdirectory(tests)`, so the Hamlib test was
not built and `ctest` passed 9/9 while running 8 — green, and testing
nothing, which is the failure mode the `SSTVAE_REQUIRE_CODEC` env var
exists to prevent elsewhere. The optional-dependency blocks now all
precede the tests directory, and `tests/CMakeLists.txt` keys off
`if(TARGET sstvae_rig)` rather than a variable, because a target cannot
exist without having been created.

**Do not assert that noise decodes to nothing.** A preamble-shaped peak
clears the detection threshold every few seed-minutes and the
Golay-coded header behind it occasionally decodes to a plausible mode —
measured at 0 spurious locks in 4 vetted peaks over 12 seeds for
*each* implementation, but the first seed picked for the C++ test
happened to be one that locked. The invariant that does hold, and what
`test_rx_engine.cpp` checks instead, is that noise never *finishes* a
reception: a spurious lock reports a few frames and stops advancing.

**The pilot is three things at once, and it was the most-clipped symbol
in the waveform** (`PROTOCOL_VERSION` 3, 2026-08-14). It is the frame
channel-estimate reference, the preamble (four repeats of it) and the
blind-acquisition template — and the old frozen QPSK draw had **7.9 dB
envelope PAPR against a clip threshold ~1 dB above the mean**, so it
lost power *and* gained distortion exactly where the receiver could
least afford either. `config.PILOT_PHASE_NUM` is now a numerically
minimized crest-factor set at 0.99 dB, worth **~+2.5 dB of latent SNR**
(+3.3 at 4 dB AWGN, +2.5 at multipath 8 dB), with acquisition improving
rather than paying for it. Five things this settled, none of them
obvious:

- **Zadoff-Chu was tried first and must not be tried again.** At 2.70 dB
  with a one-line closed form and ideal periodic autocorrelation it is
  the obvious choice, and its defining property kills it: a ZC frequency
  shift *is* a time shift, and this sequence is the acquisition
  template, so CFO and timing become confusable. Blind acquisition
  locked **55.7 Hz off inside its own ±55 Hz search range**, slipped 7
  frames of timing, and the true-vs-false metric gap collapsed from 14.4
  to **2.07** (the chosen set: 24.5). No threshold survives that. The
  existing suite caught it; the screen that cleared the pilot did not,
  because it ran at **zero frequency offset** — which is precisely where
  the coupling is invisible.
- **Pilot PAPR is a proxy, not the mechanism.** Candidate ordering by
  PAPR is not the ordering by latent SNR. Anything pushing this further
  should optimize against latent SNR and acquisition directly.
- **`CLIP_HEADROOM_DB` and the pilot are one change.** Most of what a
  wider headroom used to buy was relief for a pilot being destroyed by
  the clipper; with a clean pilot the AWGN optimum moves ~3.0 → ~1.0 and
  the multipath optimum to ~0.0. **Do not lower the headroom without the
  new pilot** — at the old one, 0.0 is 0.2 dB *worse* than 0.5.
- **`BLIND_SCORE_THRESHOLD` moved 4.0 → 9.0 with it, also one change.**
  The blind score is scale-invariant in signal level but not in how much
  pilot survives the clipper, so a stronger pilot lifts true and false
  scores together. Calibrated like `TEMPLATE_SCORE_THRESHOLD`, between
  the highest false score (7.33 — an out-of-range *real* transmission,
  not noise, which peaked at 1.43) and the lowest true one kept (10.19).
  Out-of-range false locks went **0.60 → 0.00** and mpp got *better* at
  useful SNRs; it gives up mpp at −6 dB, which no threshold could keep
  (true 6.4 against false 7.3). Four hardcoded copies of the old 4.0
  existed — two Python, two C++, plus a stale one in the parity shim.
- **The 0.794 latent gain is deliberate; do not "fix" it.** The clipper
  compresses high-crest data symbols and leaves the flat pilot alone, so
  the pilot-derived estimate describes a slightly different gain than
  the data saw and latents return 21% small (old pilot: 1.008).
  `native/tests/test_modem_roundtrip` checks it and the Python suite
  structurally cannot. It is benign because the decoder consumes
  `latents × weights`, and the confidence weights **already are** a
  Wiener shrinkage: on that quantity the MMSE-optimal rescale is
  0.91–1.20 and the remaining oracle headroom is 0.04 dB at mpp 8. It is
  stable enough to train against — 0.7933 ± 0.0008 across data,
  identical across modes, unmoved by AWGN, flat across carriers to
  ±1.9% — but it tracks `CLIP_HEADROOM_DB`, so re-measure if that moves.
  **Score `latents × weights`, never bare `latents`**: measuring the
  latter invented a phantom "+1.5 dB from adding Wiener shrinkage".

**Format constants are frozen data, not computations.** The interleaver
permutations (`sstvae/modem/interleaver_perms.npy`) were drawn from
seeded numpy, but nothing re-derives them: doing so would make numpy's
PCG64 part of the waveform, so a future numpy that changed its stream
would silently change what the radio transmits. If numpy ever does
change, the right response is to keep sending the frozen values.
(The pilot used to be in this category and is now strictly better off:
a closed form over exact integers has no generator to depend on. It is
carried as `PILOT_PHASE_NUM`/`PILOT_PHASE_DEN`, an exact rational turn
rather than radians, because `sin`/`cos` of a decimal differ across
libms and architectures — the same argument as `ofdm._phasor`. That is
where any future format constant should aim.)
`tools/freeze_format_constants.py --verify` reports whether numpy still
agrees and **exits 0 either way** — it is deliberately not a CI gate,
because a red build whose obvious fix is "regenerate" would invert the
direction of authority. `tests/test_frozen_format.py` walks the AST of
every module in `sstvae/modem/` and fails on any `default_rng` call.

Three artifacts are **generated and committed**, and CI fails if any is
stale. Committed so a plain `cmake` build needs no Python; generated so
there is only ever one source of truth:

- `native/core/config.hpp` ← `tools/gen_config_header.py` from
  `sstvae/config.py`. Never hand-edit it. Two hand-maintained copies of
  the waveform constants would be the single most likely cause of a
  silent on-air incompatibility.
- `native/tests/golden/` ← `tools/gen_golden_vectors.py`. 22 `.npy`
  files plus a `manifest.json` carrying shape/dtype/sha256 — the
  manifest is the *reviewable* part, so a deliberate regeneration
  produces a diff naming exactly which vectors moved.
- The layering rules are checked by `tools/check_layering.py`, not by
  good intentions: nothing under `core/` includes QtWidgets, only
  `core/overlay/` may include QtGui, and only `bindings/embed/` may link
  libpython.

**`pytest --native` is the point of the whole exercise.** It substitutes
the C++ implementations into the reference modules, so the existing
suite becomes the port's acceptance suite (currently 243 fast + 19 slow
against C++). `tests/test_native_parity.py` is the complement: it holds
both implementations in one process and diffs them, which is what you
need when `--native` fails and you want to know *where*.

- Substitution is by **attribute assignment**, so every binding keeps
  its Python counterpart's exact signature — a shim that "improved" an
  interface would break the mechanism. `from x import y` sites bind at
  import time and are invisible to it, so they are listed explicitly in
  `NATIVE_SUBSTITUTIONS`; a missed one silently keeps testing Python.
  **"Exact signature" includes the defaults**, and a hand-copied one is
  the part that drifts without any import failing: a stale
  `threshold=0.5` in the `acquire` shim outlived the constant it was
  copied from, so the shim asked the C++ a different question than the
  parity test asked it — and the result reads exactly like an
  implementation disagreement, at a single threshold-boundary cell. The
  defaults now come from `config`.
- Without the extension module built, the parity tests **skip** and
  `--native` **errors**. Both are deliberate: a parity suite that
  quietly passes because it tested nothing is worse than no suite.
- **Reduce phase arguments exactly before any transcendental.** Both
  implementations do this now (`ofdm._phasor`, `dsp._HET_TABLE`,
  `dsp.wrap_cycles`, and the C++ `carrier_phasor`), so parity tolerances
  are sized by one ulp of `exp()` rather than by anyone's accumulated
  error: `PHASOR_TOL` is 1e-14 against a measured 9.6e-16.
  **Do the same in new DSP.** Where the frequency is an integer number
  of Hz the reduction `(n*f) % FS` is exact and free; `to_baseband`
  needs only 16 distinct phasors because `FCENTER/FS = 3/16`. This is
  not about accuracy — 1e-10 rad is 6e-9 degrees — it is that `sin`/`cos`
  of a large argument disagree across glibc/musl/MSVC and across
  x86-64/Apple silicon by far more than near zero, so an unreduced
  argument makes a result a property of the machine rather than of the
  signal. It had already broken CI. See `docs/todo-done.md` (closed
  2026-07-28) for the measurements.

## Gotchas learned the hard way

- `dsp.to_baseband` is deliberately **unfiltered**: any FIR selective
  enough to matter smears past the 32-sample CP and causes ISI. The
  160-sample demod correlation already nulls the heterodyne image
  exactly. Only sync filters (its own copy).
- The timing tracker must be heavily smoothed: raw per-frame pilot
  phase slope sees multipath group delay (± many samples), while real
  clock drift is <0.1 sample/frame. Chasing it raw wrecked MPP fading
  performance.
- PAPR is envelope (PEP) based — clip the analytic-signal magnitude,
  not raw samples; measure with `dsp.papr_db`.
- The latent unit-RMS normalization is the on-air contract between
  encoder, modem, and training. Don't renormalize anywhere else.
- Local GPU is ROCm (`torch.cuda.is_available()` is true); never add
  CUDA-only dependencies.
- **Nothing outside `train` touches torch at all**, let alone a GPU. The
  codec runs on onnxruntime — 53 MB installed against torch's 345 MB —
  so `cli`/`listen`/`gui` are ~263 MB installed, down from ~555 MB. This
  deleted the CPU-index pins and shrank `conflicts` to one pair. The
  remaining `[tool.uv.sources]` entry pins **`dev`** to CPU torch,
  because several tests `importorskip` it as the reference
  implementation and there is no CI to notice them silently vanishing;
  the `conflicts` block is still load-bearing for the same old reason
  (uv resolves one torch per lock, so without it the CPU pin wins for
  `train` too). The GPU half of the rule stands on its own measurement:
  the encoder is 31 ms and the decoder 50 ms per 640x480 image against
  ~270 ms of NumPy DSP in the same operation, on a transmission lasting
  32–95 s. Don't add a GPU path to the app, and don't advertise one.
- `sstvae/images.py` holds the geometry (`IMG_W/IMG_H`), `fit_image`,
  `image_to_array` and the font search; `sstvae/data.py` is training
  only and re-exports them. **`images.py` must import without torch** —
  `image_to_tensor` survives for training and imports torch lazily, but
  `load_image` and `image_to_array` return ndarrays. An unconditional
  `import torch` here would pull 345 MB back into every sending station
  no matter what the codec does. Import from `images`, not `data`,
  anywhere in the send/receive path — `data` pulls in torchvision and
  `torch.utils.data`, which is why torchvision is a `train`-only dep.
- **SNR is quoted in a 2500 Hz noise bandwidth** (`config.SNR_REF_BW_HZ`),
  changed from 3000 Hz on 2026-07-26. It is one constant, used by both
  `hfchannel.awgn` (which generates the noise) and
  `modem._estimate_snr_db` (which measures it) — never hardcode a
  bandwidth in either, because a mismatch between them is invisible:
  both keep working and simply disagree about what a number means. The
  same physical channel reads **0.79 dB higher** on the new scale
  (`10log10(3000/2500)`), so any pre-2026-07-26 SNR figure found in old
  notes is 0.79 dB *below* its equivalent today. Note that
  `latent_channel.py` and `waveform_channel.py` add noise per-latent /
  per-carrier against unit-RMS references — those have no noise
  bandwidth and were deliberately left alone; changing them would alter
  training, not relabel it. The README's tables were re-measured on the
  new scale with `scripts/snr_sweep.py`.
- **Nothing may hold the `RingBuffer` lock across a bulk copy.** The
  audio callback calls `write()`, and a blocked audio callback means
  PortAudio *discards input*. `snapshot()` used to copy the whole buffer
  (8 MB at 130 s) under the lock, so the decode loop tore a hole in its
  own audio every `poll_interval`, and the holes **grew** as the buffer
  filled and the copy slowed. Measured against a simultaneous clean
  capture of the same playback: losses of 85 samples rising to 235, one
  every 5.00 s, 1718 samples over 50 s — 0.35% of timing error, which is
  ~4 samples/frame against a drift tracker built for <0.1. Result was
  5 dB of SNR, a failed beacon and a mangled picture, while still
  syncing and reporting every frame received. `write()` now holds the
  lock only to publish two integers, and `snapshot()` copies outside it.
  A microbenchmark of the old code showed writes blocked for **786 ms**
  against a 0.43 ms snapshot; the new one, 0.01 ms.
  `tests/test_rx_ringbuffer.py` guards this on the p95 of write latency,
  self-calibrated against the copy cost.
- **The app defaults to QtMultimedia, not PortAudio**, and the reason is
  a measured bug rather than taste. The dispatch lived in the deleted
  `gui/audio_backend.py` and now lives in `native/core/audio/`;
  `audio.backend` (`"qt"` | `"portaudio"`) is still the config key.
  PortAudio is kept because **Qt does not list PulseAudio/PipeWire
  *monitor* sources**, so a loopback needs `module-remap-source` to be
  visible to Qt while PortAudio sees monitors directly. `sstvae/audio.py`
  is the surviving PortAudio path and is what `sstvae_listen.py` uses,
  so the bullet below still applies to it.
- **A PortAudio callback written in Python sits on the host's realtime
  thread and needs the GIL** — that was the root cause of the worst bug
  found so far. When another thread holds the GIL (converting a 640x480
  preview to a QPixmap and painting it, right after every decode poll),
  the callback cannot run. PulseAudio and PipeWire's own device have a
  big software buffer and absorb it invisibly. **JACK has none**: a
  couple of milliseconds per period with nothing queued, so audio is
  skipped silently, with no status flag. `QAudioSource` is pull-based —
  Qt's C++ backend fills a buffer and we drain it from the event loop —
  so Python leaves the realtime path entirely. Measured on K4 RX A with a
  thread deliberately holding the GIL: **clean through 800 ms of
  blocking** (+211 ppm), where PortAudio on JACK lost 3500 ppm at ~30 ms.
  - Measured on a PipeWire-JACK device: ~200–350 samples lost per decode
    poll, **tracking `poll_interval` exactly** — change it from 5 s to
    11 s and the losses follow — for 5 dB of SNR and a mangled picture,
    while sync succeeded and every frame was reported. The same code was
    clean headless and clean on `pulse`/`pipewire`, which is why it
    looked like a GUI decode bug for several rounds.
  - **Diagnosing this class of bug:** compare two *simultaneous* captures
    of one playback (`scripts/diagnose_capture.py --out` alongside the
    GUI's `receive.save_audio`). Windows that correlate at 1.000 but at a
    drifting lag prove sample loss rather than added noise, and the
    interval between lag steps names the culprit.
  - **PortAudio's blocking API is not an alternative fix**, though it
    would also put C on the realtime path: `stream.read()` **corrupts the
    heap** on the JACK backend (`malloc(): invalid size`) at every
    blocksize and latency tried. Verified, not assumed. That is what
    forced the move to QtMultimedia rather than a PortAudio rework.
    `audio.warn_if_fragile_host` still warns if the PortAudio backend is
    used with a JACK device.
- **Capture opens at the device's *own* rate and resamples in our code,
  never by asking the device for 8 kHz.** Almost nothing is natively
  8 kHz, so requesting it doesn't avoid a resampler — it delegates to
  whichever one the audio stack has, and JACK cannot resample at all
  (a JACK stream only ever runs at the server's rate, whatever you
  asked for).
- **`samplerate` in the audio API is the *ring buffer's* rate, not a
  device setting.** It is fixed at `FS` by the modem, and passing
  anything else fills the ring with wrong-rate audio that decodes to
  nothing. `sstvae_listen.py` used to expose it as `--samplerate`, which
  read like "ask the device for this"; that flag is gone.
- **Capture resampling is stateful — `audio.StreamResampler`, never a
  bare `resample_poly` per callback chunk.** `resample_poly` is an FIR
  polyphase filter, so an isolated chunk is zero-padded at both ends and
  every chunk boundary gets a transient; at 44.1 kHz→8 kHz the filter is
  8821 taps against ~186 output samples per chunk. Per-chunk `ceil`
  rounding also gains samples (684 over 66 s, a 0.13% clock error the
  timing tracker then fights). Measured on a real on-air recording:
  **4.7 dB of SNR** (+2.4 → −2.3 dB) and a badly mangled picture — while
  still syncing and reporting 440/440 frames, which is why it looked
  like a decoder bug. `play()` avoids this by resampling the whole
  waveform up front; capture cannot, hence the class. Only devices that
  *reject* 8 kHz take this path, so the default PulseAudio device never
  shows it — `tests/test_audio.py` now fakes an input device to catch it
  without hardware.
- `wavio.read_wav` must scale integer samples **before** the stereo
  mixdown. `mean` returns float, so a dtype check afterwards skipped
  normalization for every stereo integer file and returned ±32767
  samples. The modem is scale-invariant enough that it decoded anyway.
- Capture and playback need **inverse** resample ratios, and sharing one
  "ratio to the device" helper between them is a silent, hardware-only
  bug: playback decimated 48k→8k instead of interpolating 8k→48k, so a
  32 s transmission went out as 0.9 s of noise. Only devices that
  *reject* 8 kHz take that path (an Elecraft K4's USB codec does;
  PulseAudio's `default` does not), so testing against the default
  device proves nothing. Use `audio.resample_ratio(src, dst)`, which
  names both rates, and see `tests/test_audio.py` for the fake-PortAudio
  harness that catches it without hardware.

- `sstvae/waveform_channel.py` — stage-2 differentiable modem replica
  (torch): OFDM synth, envelope clip/PAPR, symbol-domain fading,
  noisy-pilot Catmull-Rom EQ, burst erasures. Tested to correlate
  >0.98 with the NumPy modem on clean channels. Runs in fp32 outside
  autocast (complex ops); `train.py --stage2` handles that split.

## Docs

- `docs/cyclic-prefix.md` — explainer: what the CP is, why carriers must
  sit on multiples of RS for it to be truly cyclic, why `demod_window`
  throws it away and backs 6 samples into it, and how it divides labor
  with the pilots (CP handles delay spread, pilots handle Doppler).
- `docs/latent-mixer-results.md` — the latent MLP-mixer experiment and
  why no mixer on the latent grid's axes can move PAPR (the interleaver
  scatters the 46 latents that share an OFDM symbol).
- `docs/slot-domain-precoder.md` — design for the mechanism that *can*
  reach PAPR (DFT spreading / learned unitary precoder in slot domain).
  Not implemented.
- `docs/onnx.md` — the ONNX runtime path, **implemented 2026-07-27**:
  onnxruntime is 53 MB installed against torch's 345 MB, fp32 ONNX is
  the same codec to ~2e-06, and both fp16 and int8 are now essentially
  free (int8 −0.002 dB on photographs, −0.112 dB off-distribution, at
  2.7× smaller than fp32). **fp16 remains the default.** Read it before
  assuming quantisation is dangerous here — latents are analog, so it is
  additive noise under the channel's, not a format break. Two traps it
  records, both of which cost real time: `per_channel` is a silent no-op
  because `ConvInteger` is per-tensor only, so int8 accuracy comes from
  leaving the worst layer per part at fp32; and **quantisation must be
  scored on off-distribution pictures**, since the fully-quantised
  decoder measured 0.10 dB on COCO and 1.54 dB on synthetic probes —
  tuning on photographs alone ships the 1.54 dB. Artifacts are exported
  by `scripts/export_onnx.py` and published as
  `v1-{encoder,decoder}-{fp32,fp16,int8}.onnx`.
- `docs/native-app.md` — design for the native C++/Qt 6
  desktop app that **replaced** `sstvae/gui/`. That GUI was frozen at
  parity (2026-07-29) and deleted 2026-08-01, one step ahead of the
  signed release the amended decision named — see "The engines". The
  amendment's reasoning is still the interesting part: between parity
  and packaging the only way to run the native app was to build C++, Qt
  and Hamlib from source, so deleting at parity would have left
  operators with no app at all in order to remove code that costs
  nothing to keep. Freezing is what kept the duplicated-maintenance
  cost bounded meanwhile. Depends on `docs/onnx.md` landing first —
  the app cannot embed torch. Read it before assuming the motivation is
  download size: after ONNX, frozen Python is already in the same size
  class, and the real wins are startup, install robustness, and native
  platform integration. Two load-bearing points: the golden-vector and
  pybind11 parity harness must exist *before* `sync.cpp` is written
  (Python is the oracle, so the riskiest code is also the most
  checkable), and the phases are deliberately sized in lines-displaced
  and what-verifies-them rather than in weeks — the bulk of the code is
  the part that goes quickly, and the hardware/signing/on-air tail is
  the project.
- `docs/latent-optimization.md` — transmit-time per-image latent
  optimization, **implemented end to end 2026-07-31**. The encoder
  is amortized, so for any one picture there are better inputs to the
  same frozen decoder; gradient descent on the latents finds them, and
  a transmission lasts 32–95 s against the encoder's 31 ms. Measured
  +3.6 dB on a photograph and +6.9 dB on off-distribution text, all
  sender-side: optimized latents are ordinary latents, so **no
  receiver, artifact or on-air change is involved**. Read it before
  assuming this needs torch in the app — it does not. The decoder's
  weights are frozen, so only the gradient w.r.t. its *input* is
  wanted, and that VJP exports as a plain opset-17 ONNX graph
  (`scripts/decoder_vjp_prototype.py`, agreeing with autograd to
  2.3e-9), leaving the runtime one ORT session and an Adam loop. Two
  traps it records: the backward must be **hand-derived as forward
  ops**, because exporting through `torch.autograd.grad` traces a
  double-backward and hits unregistered symbolics; and optimizing
  against the *clean* decoder is the wrong objective — it wins 1.25 dB
  clean and loses 1.05 dB under the channel it will actually meet —
  and the **end-to-end round trip (2026-07-31) settled that
  decisively**: with the clean objective the feature is *harmful*,
  losing on every fading cell, while channel-aware at 5 dB (Andrew's
  figure) wins all 40 cells by 0.1–3.0 dB. Two rules came out of it.
  **Latent-domain PSNR is an objective value, never a result** — it
  flattered itself by ~2×. And PAPR, the risk that doc originally led
  with, measured ±0.05 dB: the clipper absorbs it and stage-2 trained
  through that same clipper, so the *objective* was the real risk all
  along.
- `docs/android.md` — design for the Android port, plus the
  implementation notes from building it. **Tier 0 is done (2026-08-08)
  and receives on hardware**: `native/android-app/` decodes complete
  pictures on a Galaxy S25+ over acoustic coupling, no artifacts,
  capture inside ±100 ppm, ~0.5 s of DSP per five-second poll.
  **Tier 1 (transmit) is done too (2026-08-09) and worked on the air
  the same day**: phone into a radio over a USB audio interface,
  received on an Android tablet on another radio, mode B, CW ID on,
  **25 dB SNR** — the first Android-to-Android contact, and the first
  time a phone's USB audio output has been shown to drive a radio.
  **The VOX leader was not exercised**, because that radio has no VOX
  on the USB/data input and it was keyed by hand; the leader is still
  tested only against the preamble detector and the emulator. **A
  coverage gap, not a design signal** (Andrew): that limitation is
  uncommon, plenty of radios do offer VOX on the USB input, and a USB
  soundcard into a radio's *microphone* input keys on VOX regardless —
  so "RX+TX without rig control" remains the right shape for Android
  and this is not an argument for pulling CAT forward. (CAT was pulled
  forward anyway, 2026-08-22, for an unrelated reason — it turned out to
  cost a socket rather than a Hamlib fork. This judgement stands on its
  own terms: VOX is still the shape for a station with no cable to the
  radio.) Four other
  things it
  settled that are not obvious from the design. **No overlay, not even
  an automatic callsign caption** (Andrew): the beacon carrier
  identifies the station to every receiver whether or not the picture
  came through readably, and a CW ID identifies it to a human by ear,
  so text burnt into the pixels only reaches someone who already
  decoded it — the design doc's prediction that `core/overlay/` would
  land in Tier 1 is the one part of it that was wrong, and a callsign
  is **not** required to send, because the app does not know it is
  attached to a radio and does not take responsibility for the
  operator's identification. **The VOX leader must be a swept tone**
  (`core/dsp/leader.hpp`) — a steady one reads 1.000 on the preamble
  detector and steals the lock, measured. **`FindClass` cannot see an
  application class from a thread we created**, which is a hazard for
  every future C++→Java call off the UI thread and was invisible
  through all of Tier 0 because every control call came from Qt's
  thread; `set_java_vm` caches a global reference now. And
  `AudioBridge.java` had become a **hand-synced duplicate** between
  `core/audio/android/java/` and the app's package source dir, because
  androiddeployqt takes only one `QT_ANDROID_PACKAGE_SOURCE_DIR` — it
  is assembled in the build tree from both sources now.
  `native/android-app/README.md` is the working document — build
  commands, the emulator recipe, and the traps; read it before touching
  the app. **`tools/build_android.sh` is how you build it**, and the
  reason it exists is the sharpest lesson of the port: the NDK's Debug
  configuration passes no `-O` flag at all (clang defaults to `-O0`),
  which costs 6–15x in a receive loop that is scalar DSP over a 130 s
  ring, and the alternative (`RelWithDebInfo`) emits an unsigned APK
  that will not install — so everyone picks Debug. The script does
  RelWithDebInfo + zipalign + debug-sign in one command.
  **Back has three behaviours and the split is the design**
  (2026-08-10): it closes the picture viewer if that is open; it
  **backgrounds the app** if a session is running or an over is in
  flight; and only otherwise does it leave. Ending the activity ends
  the *process*, and the process owns the engine, so the most ordinary
  gesture on a phone silently killed a reception and left the shade
  claiming the station was still listening —
  `stopWithTask="false"` does not cover this, since that is about a
  swipe from Recents rather than the activity finishing.
  `moveTaskToBack` is what recorders and media apps do and what the
  ongoing notification already implies. Do **not** extend it to the
  no-session case: hijacking Back unconditionally is the thing this
  avoids. Measuring that bug found a second one — the exit was
  **recorded as a native crash**, a SIGABRT tombstone in Android's own
  `hwuiTask` threads during teardown, which is what Play's vitals
  count — so `main()` now ends with **`std::_Exit`** rather than
  returning. Third instance of the rule that `Session`'s immortality
  and `check::Watchdog` are the other two: at teardown, not running
  code is the reliable option.
  **Three of the port's bugs were bugs in its own instruments**, none
  in the modem, engine or codec, and each presented as a fault
  elsewhere: the `-O0` build read as a slow onnxruntime; the capture
  drift meter timed to `now` while counting samples to the last chunk,
  charging in-flight audio as lost and reporting a steady −4500 ppm
  `DROPPING AUDIO` on a phone whose pictures were perfect; and the
  emulator's default `swiftshader_indirect` renderer tears every
  screenshot, which read as a clipped toolbar. Expect that pattern —
  `native/core/` is covered by golden vectors and `pytest --native`,
  the instruments around it are covered by nothing. Two of them
  corrupted this file's own record before being found, so **re-measure
  before quoting any Android number from before 2026-08-08 evening**.
  Two consequences worth not re-deriving. **The poll cost is DSP, not
  the codec** — `Progress::last_decode_s` is measured *before* the
  codec runs, in both implementations, so it never contained any
  inference; at a full ring it is `sync::acquire` 171 ms +
  `demodulate` 192 ms against a 2 ms `to_baseband`, which is why
  `decode_loop_low_cpu` (no blind accumulator, searches only new
  audio) is the battery lever and ORT thread count is not. And **the
  model fetcher is Java, not `qt_fetcher.cpp`** — Qt for Android ships
  no TLS backend, so transport is `ModelFetcher.java` behind the
  existing `checkpoint::Fetcher` seam, with the sha256 check and the
  `.part` rename kept in C++ because that is the half where a mistake
  silently corrupts a cache. `RxConfig::max_decode_duty` (default 1.0
  = off, unchanged desktop behaviour; Android sets 0.5) is the one
  change this made to shared code.
  **Since 2026-08-10 that fetcher is the fallback, not the normal
  path: the codec ships inside the APK** (+18 MB, 55 against 37). The
  argument is not download size — the model *is* part of the on-air
  contract, since stations must run the same checkpoint to
  interoperate, so "update the model without an app update" is a way
  to desynchronise a station rather than a feature, and what is left is
  that first run has to work on a hilltop with no coverage. Three
  things it settled. It goes in **`assets/`, never a `.qrc`** — a
  resource is compiled into the app library and a bundle carries one
  per ABI, so the weights would ship twice. The bytes are **released
  once the ORT session is built**, and `codec.cpp` sets
  `session.use_ort_model_bytes_directly` to `0` explicitly rather than
  relying on that being the default, because that opt-in is exactly
  what would turn the transient buffer into a use-after-free with no
  diagnostic. And `SSTVAE_ANDROID_BUNDLE_MODELS=OFF` is a supported
  configuration that changes no code path — `assets::model_blob`
  returns nullopt and the codec falls through to `resolve_onnx`, which
  is what keeps that flag from being a fork. **A debug-signed APK is a usable
  beta artifact** — it installs and upgrades in place after the
  warnings (Andrew, 2026-08-08), and two blockers this file previously
  asserted were wrong: Android blocks a *downgrade*, not an equal
  `versionCode`, and the per-machine debug keystore only bites once a
  *second* machine builds a tester APK. What is real is that switching
  to a proper signing key is **not** an upgrade — every tester must
  uninstall, losing settings and saved receptions — so warn them in
  advance and get the real key in early.
  **The Play bundle exists as of 2026-08-10** (`tools/build_android.sh
  --aab`, signed with an upload key under `~/.android-keys/`, outside
  the repo). Read "The Play upload" in `native/android-app/README.md`
  before touching it. Three things it settled. `--aab` deliberately
  shares **no fallback** with the APK path — it refuses rather than
  emit an unsigned or debug-signed bundle, because Play rejects both
  and does so minutes later in a browser, a long way from the build.
  `--version-code` is an **explicit input**, since Play requires it to
  increase forever and no build can infer it. And the gate worth
  knowing about in advance is **16 KB page alignment**, required of
  anything targeting SDK 35+: all 78 libraries pass, onnxruntime's
  prebuilt `.so` included, and it is *not* fixable downstream, because
  a bundle is not zipaligned — alignment belongs to the APKs Play
  generates from it, so it has to be right in the `.so` files.
  The original design, which survived contact almost intact: a Qt Quick
  front end over the existing `native/core/`, starting at Tier 0 (a
  receive-only listener) with later tiers optional, and **native Android
  audio from the beginning rather than QtMultimedia**. The 17.5k Qt-free
  lines under `native/core/` port unchanged and create **no new parity
  surface** — an Android app is a fourth build of the same code, not a
  reimplementation, so the golden vectors and `pytest --native` still
  cover it. onnxruntime publishes an Android AAR at the **exact pinned
  version** (1.28.0, with the C++ headers), so the codec's "same
  version, two builds" basis survives — a better position than macOS
  x86_64. Two points worth not re-deriving. The audio layer is **Java
  `AudioRecord` on a blocking reader thread, not AAudio/Oboe**: there is
  no latency requirement here at all (2 s of buffer, 5 s polls), so
  AAudio's only real benefit is worth nothing, while its `setDeviceId`
  is silently ignored on the OpenSL ES fallback and USB capture has open
  glitch reports — and enumeration needs Java either way. That design is
  also the blocking-read architecture the desktop app wanted and could
  not have, since PortAudio's blocking API corrupts the heap on JACK.
  **Rig control was written up as dropping for a structural reason and
  that was wrong** (corrected 2026-08-22, and the doc keeps the error
  rather than deleting it). The premise held -- Hamlib's serial layer
  opens a path and Android gives an unprivileged app no `/dev/ttyUSB` --
  but the conclusion did not, because **Hamlib does not require a serial
  port**: `rig_open()` puts the pathname through `parse_hoststr()`,
  which rejects `/dev/...` and `COM*` and accepts `host:port`, and on a
  match sets `RIG_PORT_NETWORK` for *any* model. That is the mechanism
  behind `rigctld -m <native model> -r <ser2net host>:4001`, so a socket
  is a first-class transport for every backend Hamlib has. CAT and PTT
  are therefore in (see "Rig control on Android" under "The native
  port"), the radio list
  on a phone is the desktop's, and the two hedges the old section ended
  on both held: `rig::Backend` was a seam and `RigController` ported
  unchanged.
  **The UI is explicitly not a port of the desktop's** (Andrew,
  2026-08-08) — the desktop layout history in this file is a record of
  QtWidgets on a desktop, and reaching for it there would be inheriting
  answers to questions nobody is asking. Three consequences are
  structural rather than cosmetic. **The foreground service owns the
  engine and the UI is a detachable view**, inverting the desktop's
  `AppState`, because a listening session must survive the screen going
  off — which also means rendering stops entirely with no UI attached,
  and that is most of the battery answer. **Reception metadata must be
  persisted beside the picture**, because `rx/engine` wipes it from
  shared state after two seconds and on a phone the operator is usually
  not looking; the desktop's last-reception card was a workaround for
  the same thing. And **the waterfall is the tuning instrument**, not a
  diagnostic, since with no CAT there is no frequency readout at all —
  which is also why `spectrum.cpp`'s peak-hold matters more there, a
  ragged comb from point-sampling being the worst available lie on a
  display whose whole job is "are you tuned right". **CAT since
  2026-08-22 adds a dial readout, and changes none of this**: every
  session without a cable still has only the waterfall, so it keeps its
  height and the readout is one line above it.
- `docs/todo.md` — open work items with the reasoning behind them.
  Completed items keep only a short summary there; the full measurement
  records moved to `docs/todo-done.md` (2026-08-12).
  Two acquisition items are now **implemented** (2026-08-11) and their
  full sections (now in `docs/todo-done.md`) record the reasoning
  rather than outstanding work:
  `config.ACQUIRE_MAX_BINS` is 12 (+-625 Hz) unconditionally,
  `config.BLIND_BIN_STEP_HZ` is 12.5 with `sync.refine_cfo` recovering
  the sub-bin peak, and the +-625 Hz *blind* range plus the off/slow/fast
  drift loop are settings (`RxConfig.blind_wide` / `drift_track`,
  `--blind-wide` / `--drift-track`, Settings > Receive). **The three
  acquisition/blind defaults now live in `config.py` for the reason the
  parity shims already document** -- porting the change turned up four
  hardcoded copies in `native/` (`sync::acquire(z, PREAMBLE_THRESHOLD,
  2, ...)` and `acquire_blind(z, 55.0, 1.7, ...)`), any of which would
  have left the live receive loop on the old range with every test still
  passing. One item remains open on the drift side and is pinned as a
  test: the loop's pull-in is +-`CFO_PULL_HZ` of *initial* residual, and
  on the blind path `acquire_blind` estimates the middle of its window,
  so past ~7 Hz of drift across the window the measurement aliases and
  the loop is worse than off. Anchoring it mid-window and running
  outward is the fix; not implemented.
  Two other items were open; one of them is now mostly closed as a side
  effect of a fix below. **A steady carrier is a perfect-looking
  preamble** (2026-08-09): `_autocorr_metric` is normalized by the
  window's own energy, so any pure tone reads exactly **1.000** — above
  what a real preamble reaches in noise — and `acquire` takes a hard
  argmax with no second chance. A noise floor is what saves this in
  practice (the lock holds up to a tone around −3 dB of the signal),
  but the quality cost arrives far earlier: a tone 10 dB *under* the
  signal costs 6.5 dB of latent SNR. The best fix looks like top-K
  peaks arbitrated by the Golay header, which cannot cost sensitivity
  because candidate 1 is today's argmax. Read that section before
  adding any transmit-side tone — it is why the Android VOX leader is a
  chirp. And a wider acquisition search so a mis-tuned counterpart
  still decodes — measured, the demod path is entirely independent of
  absolute centre frequency (8.73 dB latent SNR from 900 to 2100 Hz), so
  this is acquisition-side only. **Measured end to end 2026-08-11, and
  the answer is one constant**: `sync.acquire`'s `max_bins` is the whole
  limit out to ±700 Hz, where the sync lowpass finally binds — the claim
  that `sync_lowpass` binds *first* was wrong, and the stock 850 Hz
  filter carries acquisition to ±600 Hz at 0 dB unchanged. Detection is
  CFO-blind by construction, so widening cannot move the false-alarm
  rate *at the true preamble's own location*, and `max_bins` 2 vs 12
  returned the identical preamble start and CFO in 160/160 trials from
  0 to −7 dB there. **That measurement did not cover a different
  location — found 2026-08-13, against a real mis-tuned recording, not
  a synthetic one**: real transmission data elsewhere in the same
  buffer can clear `PREAMBLE_THRESHOLD` too (that threshold was
  calibrated against pure noise), and a genuinely off-frequency signal
  has real spectral content near its own true offset even away from
  the preamble — so widening the bin search is more likely to resonate
  with *that* than with unrelated noise, and can win a false lock deep
  in a transmission's own frame data that Golay-decodes a plausible
  header. `config.TEMPLATE_SCORE_THRESHOLD` (0.40) is the fix: a second
  gate on the winning candidate's template-match quality, calibrated
  against the measured false lock (score 0.338) and ~1400 synthetic
  trials at the sensitivity floor (lowest score for a real acquisition:
  0.430). See docs/todo-done.md's "A false lock this widening opened
  up" —
  it also mostly closes the steady-carrier-tone item two paragraphs up,
  as a side effect. It costs 0.14 ms per candidate
  (3.4 ms total, ~5% of one acquisition), so it is not the opt-in this
  file once implied. **The blind path is the opposite case**:
  `acquire_blind`/`BlindAccumulator` search CFO directly, so widening it
  naively is ~linear in the range — but **the 1.7 Hz grid is ~15× finer
  than the signal supports**: for a fixed lag the matched filter is a
  DTFT of a *160-sample* sequence, so `|mf|²` is band-limited in CFO and
  determined by samples every FS/(2·160) = 25 Hz. At a 12.5 Hz step
  detection is unchanged to −10 dB, and a parabola through the winning
  bin and its neighbours returns a *better* frequency estimate than
  today's raw 1.7 Hz argmax (0.14–0.62 Hz against 0.56 Hz). Net: ±625 Hz
  costs 516 ms against today's ±55 Hz at 273 ms — 11× the range for
  under 2× the CPU — or, at today's range, 131 ms, which is the Android
  battery item. Batched threaded IFFTs and a reshape-fold instead of
  `np.add.at` are bit-identical but only ~1.6× together: the IFFTs are
  the work. No sensitivity or false-alarm cost either way (noise-floor
  score 1.34 → 1.38 against a threshold of 4.0). A fourth item is
  **frequency drift during a transmission**, where the receiver corrects
  once from the preamble and never looks again: the budget is ~±2 Hz of
  *total excursion* whatever the mode (0.06 Hz/s for mode A, 0.02 for
  mode C), it fails with every frame received and the beacon decoding,
  and the cause is pilot-rate aliasing rather than ICI — pilots are
  6.94 Hz apart and at 3.2 Hz of residual the phase advances 166° per
  frame, so the interpolated channel estimate is simply wrong. A
  second-order loop on the pilot *common* phase (the slope across
  carriers is timing and is orthogonal) reaches the oracle at every rate
  to 2 Hz/s and costs nothing at zero drift, but its gains are squeezed
  from both sides: α=0.3 chases `mpd` fading for 2.3 dB, and α=0.1 is
  the setting that fails on fast Ornstein-Uhlenbeck wander — the loop
  bandwidth must sit above the drift's spectrum and below the channel's
  Doppler spread, and those can overlap, so **no compiled-in pair of
  gains serves both**. An oracle handed the true frequency path holds
  3.8 dB in every cell where every causal loop fails, so the ceiling is
  the estimator and a non-causal smoother over the whole frame sequence
  has that gap available (nothing here is real-time). Two traps: the
  correction must be a continuous ramp *within* the frame, not one
  constant phase per frame (a constant only redoes what the pilot EQ
  already does), and an oracle must remove the true path **minus
  `acq.freq_offset`** — removing the raw path double-corrects, which is
  invisible for a ramp starting at zero and ruinous for wander. A third item, "acquisition costs
  ~1 dB of threshold at large frequency offset", was **withdrawn
  2026-07-26**: it did not reproduce at 25 seeds per point and was an
  artifact of 6-seed sampling. Acquisition near threshold succeeds
  40–80% of the time, so any sweep with single-digit trials per cell
  will invent a pattern — see the warning kept in that section.

## The wiki

Most of what used to be in the README is now in the **GitHub wiki**, and
the README links to each page by URL. Eight pages: `Home`,
`Command-line-tools`, `Channel-simulator`, `Performance`, `How-it-works`,
`Comparison-with-other-modes`, `Training`, `Development`.

**It is a separate git repository**, `https://github.com/arodland/SSTVAE.wiki.git`
— not a directory in this one. Nothing in the test suite, no CI job and
no staleness gate touches it, so a page contradicting the code is
invisible until a reader hits it. That is the whole reason it is
documented here: it is the project's only prose with no check behind it.

**Cloning is the only mechanism.** There is no wiki API — `gh` has no
wiki commands and the GitHub REST/MCP tools have no wiki endpoints, so
`get_file_contents` and friends cannot see these pages at all. Clone it
(anonymously; it is public) to read or edit:

```sh
git clone https://github.com/arodland/SSTVAE.wiki.git
```

**In a sandboxed session that clone is read-only in practice.** The git
proxy will not inject a credential for `arodland/SSTVAE.wiki` — it is
not in the session's authorized repository set — and `add_repo` cannot
add it, because GitHub does not expose a wiki as a repository. `git
push` returns 403. So: make the edit, commit it in the clone, and say
plainly that it needs pushing from a machine with ordinary GitHub auth
(where it is an unremarkable `git push`). Do not report a wiki change as
published when it is sitting in a scratch clone.

**Renaming a page breaks the README.** GitHub does not redirect a
renamed wiki page and nothing checks the links, so a rename is a
two-repository change: the page and every `README.md` URL naming it, in
the same sitting.

### What each page tracks

Change one of these and the page is stale — the mapping is the point of
this list:

- **Command-line-tools** — the flags of `sstvae_encode.py`,
  `sstvae_decode.py`, `sstvae_listen.py`, the install extras in
  `pyproject.toml` (`cli`/`listen`/`train`, and the installed sizes it
  quotes), and what `--model` accepts.
- **Channel-simulator** — `sstvae_simulate.py`'s options and
  `config.SNR_REF_BW_HZ` (it states the 2.5 kHz convention and the
  0.79 dB offset from pre-2026-07-26 figures).
- **Performance** — the PSNR/SNR tables, the acquisition fractions and
  the late-join table. Regenerate with `scripts/snr_sweep.py` and
  `scripts/late_join_sweep.py` rather than editing numbers by hand; a
  new codec revision moves all of them, including the 1.4–1.8 dB quoted
  for latent optimization (which is anti-correlated with encoder
  quality, so it erodes as checkpoints improve).
- **How-it-works** — the beacon, the nested groups, the interleaver, and
  a **waveform table hand-copied from `sstvae/config.py`**. That is the
  same hazard `native/core/config.hpp` retires with a generator and a CI
  gate, here with neither: currently 24 carriers × 50 Hz at 950–2100 Hz,
  20 ms + 4 ms CP, 230 latents + 5 beacon chips per frame, 32/64/95 s.
  Any `config.py` change to those must be copied over by hand.
- **Comparison-with-other-modes** — the side-by-side table; airtime and
  bandwidth rows come from the same constants, and it embeds
  `docs/images/ota-vs-analog.png` by raw URL, so moving that file breaks
  the image.
- **Training** — `scripts/train.py`'s flags and the two stages,
  `scripts/export_onnx.py`'s artifact set, the Hub dataset name.
- **Development** — `pytest` / `pytest -m slow` / `pytest --native`,
  `tools/build_native.sh`, the `sstvae/` layout, the Qt/CMake build and
  its Linux `dlopen` dependency list, and the three
  generated-and-committed artifacts. This is the page most likely to
  rot, because it names directories.
- **Home** — the page index plus a list of the repo's own `docs/*.md`.
  Adding a doc there means adding it here too.

### Worth reading from here

Not just an obligation — some of it is the measured record and lives
nowhere else. **Performance** has the late-join table (group 0 ends at
32 s; losing part of it is nearly free, losing all of it costs ~6 dB at a
stroke) and the acquisition-vs-quality split that the bracketed
fractions encode. **How-it-works** has the waveform table in one place
and the distinction between "lost the preamble while recording" (beacon
rescues it, full quality) and "tuned in late" (complete picture, lower
fidelity). **Comparison-with-other-modes** is where the on-air analog
comparison and its caveats are written down.

## Status / next steps

Phase 1 (modem) complete; stage-1 training pipeline complete with Hub
dataset (`arodland/coco640-sstvae`, 640×480 — the target resolution was
moved up from 320×240 since mode B/C weren't earning their airtime at
the smaller size; 320×240 is still the minimum accepted input, upscaled)
+ cloud packaging (`scripts/launch_job.sh`); stage-2 channel implemented
and tested.
Beacon carrier (mid-stream resync + callsign) implemented: one reserved
carrier, absolute-frame-counter superframe, and a preamble-free blind
acquisition path (`sync.acquire_blind` / `Modem.demodulate_blind`).
`waveform_channel.py` (stage-2 differentiable replica) mirrors the same
23-carrier capacity/erasure accounting so training stays consistent
with the real modem, but does not simulate/train through beacon content
itself (synthesizes random BPSK there just for realistic PAPR
statistics).

Android: **Tier 0 is built and receiving** (2026-08-08),
`native/android-app/` — a Qt Quick front end over the same
`native/core/`, so no new parity surface. All six Tier 0 items are in
(audio layer, foreground service, three screens, persisted reception
metadata, notifications, model fetch), plus sharing, a technical-detail
switch that is off by default, and keep-screen-on. Verified on a Galaxy
S25+ over acoustic coupling. The MediaStore save to the shared gallery
that used to be listed here as deliberately skipped landed 2026-08-10
(`Gallery.java`), **off by default**, and took `minSdk` from 28 to 29
with it. Three things it settled. It runs in `ListenerService`'s
notification poller, not beside the file write, because that is the
side that already holds a `Context` — doing it from C++ would walk into
the `FindClass` hazard for nothing. The app-private copy stays
canonical and the gallery copy is deliberately provenance-free, since
MediaStore has no column Photos will display and the sidecar is what
answers "who sent this". And `DATE_TAKEN` on the insert **does not
survive** — measured NULL, because the scanner re-derives metadata from
the file once `IS_PENDING` clears and a PNG carries no EXIF date.

**Tier 1 (transmit) is built too** (2026-08-09): a Send screen with
gallery/camera picking and a touch crop, `Session` owning the
`TxEngine` the way it owns the decode loop, the over routed through the
service under a `mediaPlayback` foreground type so it cannot be
truncated, half duplex with a fresh ring buffer on resume, and an
optional swept-tone VOX leader. **Verified over RF** (2026-08-09):
phone → radio over USB → radio → Android tablet, mode B, CW ID, 25 dB,
no issues. `play()` is no longer "written, unexercised": exercising it
found the `FindClass` bug above.

**USB is no longer untried on this app** — it carried that RF contact
in both directions, which retires the caveat this section used to
carry. What is still unmeasured: battery over a multi-hour session,
and the VOX leader against a real VOX circuit (the test radio has none
on its USB input).

**CAT and PTT landed 2026-08-22** and are not Tier 2 after all: they
cost a transport and a socket rather than a fork of Hamlib. Three
connection kinds — USB serial, Bluetooth RFCOMM, and a network host
handed straight to Hamlib (which covers both `rigctld` and a
ser2net-style server, one code path in Hamlib and one here).

**USB works on hardware** (Andrew, 2026-08-23): CAT and both
control-line keying methods, on a phone, over a composite USB
interface — which settles the one question that could have sunk the
whole approach, since the app needs the audio and the serial half of
that device at the same time. **Bluetooth is untested for want of a
device** and goes to beta on the strength of the shared code beneath
it. **An IC-9700 and an IC-7100 do not work, and that is open** — see
the rig-control bullets above for what has been ruled out and the two
radio-side settings to check first. `rig::set_debug_sink` and the "Log
rig traffic" switch landed for it, because until then Hamlib's trace on
a phone went to stderr and therefore nowhere.
Unplug/replug recovery and the "Connected with the cable out"
reading were found the same day and fixed;
`native/android-app/README.md` has the three separate mistakes behind
that one symptom.

Next, in whatever order: further UI work, the rest of Tier 2, or
the Play internal test. **A signed upload bundle exists** as of
2026-08-10 — `tools/build_android.sh --aab`, version code 1, both
ABIs — so the Android half of "store signing" is no longer waiting on
the external verification that still gates the *desktop* installers.
What has not happened is the upload itself, and the thing to tell
existing sideload testers in advance is unchanged: the switch from the
debug key to the upload key forces an uninstall, not an upgrade, and
takes their settings and saved receptions with it.

Desktop app: **one implementation**, `native/` (Phases 0-3), which
reached parity, passed the loopback shakedown in all three directions
including both cross-implementation ones, and replaced the PySide6 GUI
on 2026-08-01 — see "The engines". Overlay *templates* are deliberately
not implemented, but the document format is built for them (see
`sstvae/overlay/` and `native/core/overlay/`).

ONNX runtime path complete: the codec is onnxruntime, torch is
training-only, and `cli`/`listen` install ~263 MB instead of
~555 MB. The published codec is **v4** (2026-08-14), and `DEFAULT_FILE`
/ `DEFAULT_REVISION` point at it in both implementations: six codec
artifacts plus `v4-decoder-grad-fp32.onnx` for the optimizer. The app
fetches what it needs on first run, per part — and a station that never
optimizes never fetches the gradient graph.

**v4 is the cc12 lineage fine-tuned through the `PROTOCOL_VERSION` 3
modem** (epoch 536), and the codec revision and the protocol version are
**different numbers that happen to be adjacent** — do not conflate them.
Measured against v3 through the same channel, mode B: **+0.43 dB AWGN
and +0.42 dB mpp on photographs, +1.14 / +1.27 on non-photo**, and
+0.64 / +0.65 on a real certificate. Roughly a quarter of that is the
pilot and headroom change and three quarters the fine-tune. Bumping the
revision is a **three-place change** — `sstvae/checkpoint.py`'s
`DEFAULT_FILE`, `native/core/checkpoint/checkpoint.hpp`'s
`DEFAULT_REVISION`, and **`GRAD_REVISIONS` in both**, which is the one
that gets missed: omitting the new revision there refuses the
optimizer's gradient fetch on the current codec and reads as an
unpublished artifact rather than a stale list. Both suites now assert
`DEFAULT_REVISION` is in `GRAD_REVISIONS`.

Remaining: run stage-2 fine-tune (start from a good stage-1
checkpoint, `--lr 1e-4`) — note pre-beacon checkpoints remain
architecture-compatible (model channel count unchanged), evaluation
sweeps (PSNR/LPIPS vs SNR per mode), on-air calibration. On the app
side: overlay templates, and a real on-air (not loopback) shakedown of
the PTT timing against a physical radio. For the native app: Phase 4 is
sequenced in five steps and the first three are done — CI builds five
packages and five installers (AppImage, `.dmg`, NSIS setup) on every
push. **Step 4 (signing) is done and green (2026-08-04)**:
`tools/sign.sh <app|installer> <path>` does Developer ID + notarization
+ stapling on macOS and Azure Trusted Signing on Windows, wired into
`ci.yml` around the installer step — macOS reports `source=Notarized
Developer ID` on both slices, Windows signs all three executables and
the NSIS setup. It is a **loud no-op with exit 0 when the credentials
are absent**, so a fork's CI still produces unsigned installers, and
`SSTVAE_REQUIRE_SIGNING=1` turns that skip into a failure — the same
hazard and the same answer as `SSTVAE_REQUIRE_CODEC`. **On a pull
request it is opt-in behind the `sign` label** (quota is monthly and
notarization is a round trip); a push to master, a manual dispatch and
a release all sign, the release naming it explicitly rather than
relying on the input's default, which is *off*. An unsigned release is
the invisible failure — it builds, attaches and publishes exactly as
usual — so `native-build.yml` asserts `release-tag` implies `sign`
right after checkout. Three traps it
cost, all in `docs/native-app.md`: `security import` sniffs the format
from the *file extension*, so the p12 needs a `.p12` name and
`-f pkcs12` or it fails as "Unknown format" and reads like a bad
password; the sign CLI spells all three options `trusted-signing-*`,
certificate profile included; and **a notarization failure can be
Apple's** — one was, and cleared with no change — so check Apple's
system status before editing anything, and use the `notarytool log`
fetch to tell a real rejection from an outage.

Remaining: **step 5, a real release.** There is still no releases page,
so the only downloads are CI artifacts needing a GitHub login — the
README says so plainly rather than implying a release exists. What CI
cannot prove is the part still outstanding: each of the five artifacts
installed and launched on a clean machine. Expect **Windows SmartScreen
to warn anyway at first** — reputation is per certificate and accrues
over downloads and time — which is a thing to tell operators, not a
signing bug to go and fix. See
`docs/native-app.md` for the C++/Qt rewrite design (Phases 0-3 done,
Phase 4 steps 1-3 done) and `docs/todo-done.md` for quantisation
tolerance as a future training constraint.

Transmit-time latent optimization is **implemented end to end**
(`docs/latent-optimization.md`, 2026-07-31): published artifact, both
implementations, and wired into the native app behind
`transmit.optimize`. Not yet shaken down on air.
Go/no-go is a **go**, and the whole measurement set was **re-run
against `sstvae-cc12-epoch438.pt`** and held: +1.4–1.8 dB of recovered
picture across both test images, all four channel models, 0–12 dB and
all three modes, sender-side only,
with acquisition unaffected and PAPR unchanged — provided the objective
optimizes *through* the differentiable channel, which is the condition
the whole result hangs on. The objective SNR is **5 dB and ships as a
constant**: swept, the optimum is flat from 2.5 to 7.5 dB on both
images, so it never becomes an operator setting, and the asymmetry says
err toward assuming a *worse* channel (too optimistic degrades toward
the harmful clean objective). The **L2 regularizer was measured and
removed** (default 0): with the channel in the objective it does
nothing, and above 1e-3 it only costs. It is not a cheaper substitute
for the channel term either — it rescues the clean objective to +0.79
but the channel term reaches +1.69, because a norm penalty restricts
every direction at once while the channel term constrains only the
fragile ones. The gain **survives every decoder
precision identically, int8 included** (Δ within 0.01 dB): quantisation
costs the encoder's and the optimized latents the same, so it cancels
in the difference — no compatibility tier, nothing to warn operators
about. **The gain is anti-correlated with encoder quality**: `cc12` is
0.56 dB better than `np1` on the certificate unaided and the optimizer
earns 0.21 dB less there, so expect the headline to erode as
checkpoints improve — re-measure on every codec revision rather than
trusting "it used to be worth 2 dB". The native app can run it on the
pinned inference onnxruntime via an exported gradient graph rather than
torch. **Measurement, publishing and the Python side are done**
(2026-07-31): a `-decoder-grad-fp32.onnx` ships beside each codec
revision from v3 on, `sstvae/latent_optim.py` runs it torch-free, and
`sstvae_encode.py --optimize [SECONDS]` is the flag. The gradient
artifact is **fp32 whatever `--precision` says** — the only precision
published, since the fp16 converter emits a graph ORT will not load and
int8 is excluded on principle. Watch two name traps that both end in a
picture rather than an error: `-decoder-` is a substring of
`-decoder-grad-`, and a derived sibling must be rebuilt as
`{stem}-{part}-{precision}` rather than substituted into, because the
gradient sibling of an fp16 encoder is fp32. The default budget is 20 s
and buys ~65% of the achievable gain (plateau ~90 s on a fast desktop),
so the doc's tables are what the feature *can* do, not what the default
does. **The C++ port is done too**: `core/optimize/` holds the loop in
`sstvae_core` behind a `GradFn` seam — so it builds and is tested with
`--no-codec` — and `core/codec/grad_session.cpp` is the ORT half. With
a deterministic objective and an identical input the two agree exactly
(+3.35 dB both); in normal use they differ ~0.1 dB because the channel
noise comes from different RNGs, which is the intended contract.
**Comparing the two numerically requires a 640x480 PNG**: at any other
size `fit` resizes and stb's resampler is not PIL's LANCZOS, and even
at the right size a JPEG decodes differently — both accepted
above-the-modem differences that look exactly like a port bug. Three
mutants survived `test_optimize.cpp`'s first draft and each is recorded
in the doc; the sharpest is that Adam's `m/sqrt(v)` is scale-invariant,
so summing the Monte Carlo draws instead of averaging them is
undetectable. Remaining: app integration, whose shape is decided
(Andrew, 2026-07-31) — **run it speculatively**, starting a short
debounce after the operator stops editing and running to plateau or a
generous budget, with a **second shorter budget starting from the Send
click** if that arrives first. The expensive part then sits in time the
operator was spending anyway. This needs **no change to
`optimize::run`**: `ProgressFn` is consulted every step and returning
false stops the loop, so the caller owns the deadline
(`min(generous, click + short)`) while `time_budget_s` stays the outer
backstop. What the implementation must get right is elsewhere — a
generation counter so an edit invalidates a run in flight (transmitting
the *previous* composition is worse than not optimizing), one-step
granularity on the Send response, and nothing blocking the GUI thread.
**`core/optimize/speculative.hpp` implements that**, Qt-free and
ORT-free behind a `GradFactory`, so the whole policy is tested with a
stub in a `--no-codec` build; `test_speculative.cpp` uses a latch
rather than a clock. `Progress::objective_gain_db` rides along for the
GUI to show — free, but it is an **objective** value that overstates
recovered quality ~3x, so present it as progress, never as decibels
earned. **The GUI is wired** behind `transmit.optimize` (one switch, no
dials): `OverlayEditor::documentChanged` drives it — emitted per mouse
move, which the debounce absorbs, and deliberately *not* by `select()`
— and the optimized latents enter through **`TxEngine`'s existing
`Encoder` seam**, so there is no second transmit path and "no result"
is exactly what the app always did. Send polls on a `QTimer` rather
than blocking, and **commits to the composition as it was at the
click**: an edit during the wait or the transmission is deferred to the
next send. `sync_from_config` applies the setting **on OK rather than
at the next edit**, and is also called from `on_model_loaded` because
refinement needs a codec to start from; turning it off destroys the
optimizer, which *is* how refined latents are discarded — nothing else
holds any. That is not just a UX preference — it keeps the generation
still while a send is committed, which is what makes the latents in
flight still describe the picture going out. `transmit.optimize` was
mirrored into the Python GUI's settings module at the time — the two
had to agree about the config file or each read the other's as a typo —
which stopped mattering when that GUI was deleted. What replaced the
check is `tests/test_native_settings.py`'s non-default fixture: a new
setting that the reader knows and the fixture does not now fails
`test_the_fixture_holds_no_default`, so the file's schema still cannot
grow silently.

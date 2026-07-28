# CLAUDE.md

SSTVAE: image transmission over HF radio by sending convolutional
autoencoder latents as analog values on OFDM carriers (RADE-style).
See README.md for the waveform table and usage; the approved design
rationale lives in the plan history.

## Commands

- Run tests: `pytest` (fast, ~10 s; includes full modem end-to-end tests)
- Slow gate: `pytest -m slow` (~2 min) — the listener state machine and
  the app's transmit→receive loopback. Run it after touching `sstvae/rx/`.
- Run the app: `uv run sstvae-gui` (needs `uv sync --extra gui`)
- Smoke-train: `python scripts/train.py --smoke --out /tmp/smoke`
- Full pipeline check: `sstvae_encode.py` → `sstvae_simulate.py` → `sstvae_decode.py`

## Testing the live paths without hardware

Both of these exercise the *real* code paths, which is the point — the
audio and rig bugs found so far were all invisible to unit tests.

- **Rig control:** run a dummy `rigctld` on an ephemeral port
  (`rigctld -m 1 -t <port>`) and point the app's rig settings at it. Model
  1 is Hamlib's dummy rig, so PTT, frequency readback and the whole
  `gui/rig_controller.py` threading model can be driven for real without
  a radio attached.
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
  - `sync.py` preamble detect (lag-160 autocorrelation, energy-floored
    metric), fractional + integer-bin CFO, template timing.
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

## The application

`sstvae/gui/` (PySide6) on top of headless, Qt-free engines. Nothing
below `sstvae/gui/` may import Qt; nothing in `sstvae/overlay/` may
either, so overlays stay renderable from the command line.

- `sstvae/rx/` — the live reception state machine, extracted from
  `sstvae_listen.py` (which is now just its CLI front end). `engine.py`
  holds `decode_loop` / `decode_loop_low_cpu` **unchanged** from the
  version the slow tests were written against — treat that logic as
  load-bearing and run `pytest -m slow` after touching it. Two seams
  were added: an `RxConfig` in place of the argparse namespace, and a
  `sink` that receives finished receptions. **Saving is the sink's job,
  not the loop's**, because the GUI's autosave checkbox may hold a
  picture for the Save button instead of writing it. `ringbuffer.py`
  adds `tail()` (cheap slice for the ~20 fps waterfall; `snapshot()`
  copies all 130 s) and `clear()`.
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
  PortAudio installed (the settings dialog needs to *report* that).
- `sstvae/rig/rigctld.py` — TCP client for Hamlib's `rigctld`. Chosen
  over the SWIG `Hamlib` bindings because those are installed in the
  system site-packages and a virtualenv cannot see them. A Hamlib error
  code raises but keeps the connection; a dead socket redials once.
  Every method is **blocking socket I/O** — see `gui/rig_controller.py`.
  `list_models()` parses `rigctld -l` (~3 ms, 321 rows) to populate the
  settings picker, and is the one place that surfaces "Hamlib isn't
  installed" at configuration time rather than at the first keying.
  It **slices fixed-width columns using offsets taken from the header
  line** — splitting on whitespace runs looks fine and silently drops
  rows, because fields contain single spaces ("N2ADR James Ahlstrom")
  and at least one Model fills its column exactly, leaving a single
  space before Version. Asks `rigctld`, not `rigctl`, so the list comes
  from the same binary `spawn_rigctld` runs.
- `sstvae/gui/rig_controller.py` — all rigctld I/O, on its own thread.
  **Nothing on the GUI thread may call the rig.** A rigctld that is up
  but not answering costs the socket timeout on the recv *and* again on
  the retry, so polling from a `QTimer` froze the window for seconds
  every interval. Three things keep it that way, all regression-tested
  in `tests/test_rig_controller.py` against a server that accepts and
  never replies: the poll loop is a worker thread with exponential
  backoff; `stop()` never joins or closes inline (it calls
  `RigctldClient.interrupt()`, which shuts the socket down *without* the
  lock the stuck worker is holding, then reaps on a throwaway thread);
  and PTT gets a **separate client**, so keying never queues behind a
  poll that is mid-timeout. The worker takes its stop event and client
  as arguments so a superseded one cannot publish stale state.
- `sstvae/overlay/` — `model.py` is the document, `render.py` draws it
  with PIL. Designed so *templates* are a later UI-only change:
  coordinates are normalized 0..1 (resolution-independent) and
  `ImageItem.source` is a late-bound reference (`"last_rx"` or a path)
  rather than a pasted bitmap, so a saved template keeps meaning "the
  most recent received picture". `item_bbox` is shared with the editor
  so selection handles can't drift from what is drawn.
- `sstvae/gui/settings.py` — JSON config (atomic write; unknown keys
  ignored, never fatal). Importable without Qt.
- The editor's preview **is** `overlay.render()`'s output, not a
  Qt-drawn imitation, so composition is WYSIWYG by construction.
- Half duplex: `transmitStarted` suspends receive, and resuming
  allocates a fresh ring buffer so the tail of our own transmission
  isn't decoded back as a reception.

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
- **Audio defaults to QtMultimedia (`gui/qtaudio.py`), not PortAudio**,
  and the reason is a measured bug rather than taste. `gui/audio_backend.py`
  dispatches on `audio.backend` (`"qt"` | `"portaudio"`) for capture,
  playback and device enumeration; PortAudio is kept because **Qt does
  not list PulseAudio/PipeWire *monitor* sources**, so a loopback needs
  `module-remap-source` to be visible to Qt while PortAudio sees monitors
  directly.
- **A PortAudio callback written in Python sits on the host's realtime
  thread and needs the GIL** — that was the root cause of the worst bug
  found so far. When the Qt thread holds the GIL (converting a 640x480
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
- **`pyside6-addons` is now a dependency, for QtMultimedia only.** This
  reverses the earlier deliberate choice of `essentials`: measured
  232 MB → 648 MB installed, of which **195 MB is a copy of Chromium**
  (QtWebEngine) that nothing here loads. Accepted 2026-07-28 — a silently
  mangled picture is worse than a large download. Revisit if PySide6 ever
  ships QtMultimedia without WebEngine.
- **PySide6 cannot marshal `QAudio::State`** into any Python slot in this
  build, not even a `*args` lambda, so `QAudioSource.stateChanged` is
  deliberately not connected; `qtaudio` polls `error()` from the read path
  instead. `QAudioSource` also has no `errorOccurred` signal in PySide6.
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
- `docs/native-app.md` — design (not implemented) for a native C++/Qt 6
  desktop app replacing `sstvae/gui/`, which gets **deleted** when the
  native one reaches parity. Depends on `docs/onnx.md` landing first —
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
- `docs/todo.md` — open work items with the reasoning behind them.
  Currently one: a wider acquisition search so a mis-tuned counterpart
  still decodes — measured, the demod path is entirely independent of
  absolute centre frequency (8.73 dB latent SNR from 900 to 2100 Hz), so
  this is acquisition-side only. The second item, "acquisition costs
  ~1 dB of threshold at large frequency offset", was **withdrawn
  2026-07-26**: it did not reproduce at 25 seeds per point and was an
  artifact of 6-seed sampling. Acquisition near threshold succeeds
  40–80% of the time, so any sweep with single-digit trials per cell
  will invent a pattern — see the warning kept in that section.

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

Desktop app (`sstvae-gui`) implemented: live TX/RX on a soundcard,
rigctld PTT + frequency readback, waterfall, overlay composition,
persistent config. Overlay *templates* are deliberately not implemented
but the document format is built for them (see `sstvae/overlay/`).

ONNX runtime path complete: the codec is onnxruntime, torch is
training-only, and `cli`/`listen`/`gui` install ~263 MB instead of
~555 MB. Six `v1` artifacts are published; the app fetches what it needs
on first run, per part.

Remaining: run stage-2 fine-tune (start from a good stage-1
checkpoint, `--lr 1e-4`) — note pre-beacon checkpoints remain
architecture-compatible (model channel count unchanged), evaluation
sweeps (PSNR/LPIPS vs SNR per mode), on-air calibration. On the app
side: overlay templates, and a real on-air (not loopback) shakedown of
the PTT timing against a physical radio. See `docs/native-app.md` for
the C++/Qt rewrite design (not started) and `docs/todo.md` for
quantisation tolerance as a future training constraint.

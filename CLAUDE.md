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
- `sstvae/codec.py` — `load_model` / `reconstruct` / `pad_to_full`.
  These used to live in the top-level `sstvae_encode.py` /
  `sstvae_decode.py` scripts; they are here so package code doesn't
  import a *script*. The scripts re-export them, so existing imports and
  command lines are unchanged. Always loads on CPU.

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
- **Nothing outside `train` touches a GPU, on purpose.** `codec.load_model`
  is `map_location="cpu"` with no `.to(device)` anywhere. Measured: the
  encoder is 31 ms and the decoder 50 ms per 640x480 image, against
  ~270 ms of NumPy DSP in the same operation, on a transmission lasting
  32–95 s — so GPU offload would save ~70 ms while costing seconds of
  context init. Don't add a GPU path to the app, and don't advertise one.
  The `cli`/`listen`/`gui` extras therefore take torch from the CPU index
  (`[tool.uv.sources]` in pyproject); the `conflicts` block next to it is
  load-bearing, because uv resolves one torch per lock and without it the
  CPU pin silently wins for `train` too. pip can't do index selection, so
  the README tells Linux pip users to install CPU torch first.
- `sstvae/images.py` holds the geometry (`IMG_W/IMG_H`), `fit_image`,
  `image_to_tensor` and the font search; `sstvae/data.py` is training
  only and re-exports them. Import from `images`, not `data`, anywhere in
  the send/receive path — `data` pulls in torchvision and
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

- `docs/latent-mixer-results.md` — the latent MLP-mixer experiment and
  why no mixer on the latent grid's axes can move PAPR (the interleaver
  scatters the 46 latents that share an OFDM symbol).
- `docs/slot-domain-precoder.md` — design for the mechanism that *can*
  reach PAPR (DFT spreading / learned unitary precoder in slot domain).
  Not implemented.
- `docs/onnx.md` — measured (not implemented) ONNX runtime path:
  onnxruntime is 27 MB against torch's 336 MB, fp32 ONNX is the same
  codec to ~1e-6, fp16 is free and int8 costs ~0.15 dB PSNR. Read it
  before assuming quantisation is dangerous here — latents are analog,
  so it is additive noise well under the channel's, not a format break.
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

Remaining: run stage-2 fine-tune (start from a good stage-1
checkpoint, `--lr 1e-4`) — note pre-beacon checkpoints remain
architecture-compatible (model channel count unchanged), evaluation
sweeps (PSNR/LPIPS vs SNR per mode), on-air calibration. On the app
side: overlay templates, and a real on-air (not loopback) shakedown of
the PTT timing against a physical radio.

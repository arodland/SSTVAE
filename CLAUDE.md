# CLAUDE.md

SSTVAE: image transmission over HF radio by sending convolutional
autoencoder latents as analog values on OFDM carriers (RADE-style).
See README.md for the waveform table and usage; the approved design
rationale lives in the plan history.

## Commands

- Run tests: `pytest` (fast, ~2 s; includes full modem end-to-end tests)
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
- `sstvae/hfchannel.py` — channel sim (AWGN in 3 kHz convention,
  Watterson 2-path fading presets mpg/mpp/mpd, freq/clock offset).
- `sstvae/models/autoencoder.py` — encoder (unit-RMS tanh latents,
  132ch in 3 ordered groups of 44) and decoder (takes latents ×
  weights + weight planes; handles erasures/truncation).
- `sstvae/latent_channel.py` — stage-1 differentiable channel
  (AWGN, group truncation, erasures) used by `scripts/train.py`.

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

Remaining: run stage-2 fine-tune (start from a good stage-1
checkpoint, `--lr 1e-4`) — note pre-beacon checkpoints remain
architecture-compatible (model channel count unchanged), evaluation
sweeps (PSNR/LPIPS vs SNR per mode), on-air calibration.

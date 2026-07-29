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

**Phase 3 (the GUI) is under way.** `native/gui/` is the only place
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
why.** `sstvae/rig/rigctld.py` talks to a `rigctld` child over a socket
because the SWIG Hamlib bindings live in the system site-packages where
a virtualenv cannot see them — a *Python packaging* constraint with no
C++ equivalent. So `native/core/rig/` links `libhamlib` in-process and
the socket client, the redial logic, the `rigctld` spawner and the
`rigctld -l` column parser are deleted rather than translated
(`rig_list_foreach` gives a struct, which cannot have the
silently-dropped-row bug that parser is flagged for below). Sharing a
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

**Format constants are frozen data, not computations.** The pilot
quadrants (`config.PILOT_QUADRANTS`) and the interleaver permutations
(`sstvae/modem/interleaver_perms.npy`) were originally drawn from
seeded numpy, but nothing re-derives them: doing so would make numpy's
PCG64 part of the waveform, so a future numpy that changed its stream
would silently change what the radio transmits. If numpy ever does
change, the right response is to keep sending the frozen values.
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
  signal. It had already broken CI. See `docs/todo.md` (closed
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

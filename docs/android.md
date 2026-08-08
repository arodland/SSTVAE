# Android: feasibility and design

Assessed 2026-08-08. The direction below is decided and **Tier 0 is not
written**; no schedule is attached. What does exist is a pre-Tier-0
smoke test (`native/android/`, and the section on it below), which is a
probe rather than the app.

## Decisions (Andrew, 2026-08-08)

1. **Qt Quick front end over the existing `native/core/`.** One
   codebase, one toolchain, a touch-idiomatic UI. Not QtWidgets-as-is
   (desktop hit targets on a phone), not Kotlin/Compose over JNI (a
   second language and an FFI boundary, which is the trade
   `docs/native-app.md` already rejected for the desktop).
2. **Tier 0 first — a receive-only listener.** Later tiers are optional
   and explicitly not committed to.
3. **Native Android audio from the beginning, not QtMultimedia.** The
   audio subsystem is already isolated behind a Qt-free layer, so this
   costs little now and avoids building on a backend whose device
   selection is in doubt. Acoustic coupling works — SSTVAE survives it,
   which is known, not assumed — but USB audio is a large enough jump in
   what the app is worth that it should be in from the start rather than
   retrofitted.

## Verdict

**Feasible, and most of it is not C++ work.** The 17,523 lines under
`native/core/` that are free of Qt — the whole modem, both engines, the
codec wrapper, the optimizer, the ring buffer, settings, images —
compile for Android with no changes anyone has to argue about. CI
already builds and tests that code for `linux-aarch64`, so ARM is not a
new platform for it.

What has to be built is the platform edge: audio, background execution,
storage, and a touch UI. That is real work, and it is the part with no
existing test coverage and no oracle, which is also where every
expensive bug in this project's history has lived.

**One thing does not need re-litigating, and it is the thing that
usually sinks a second implementation.** An Android app built this way
is a fourth build of the *same code*, not a reimplementation of the
waveform. The golden vectors, `pytest --native` and
`tests/test_native_parity.py` cover the shared core exactly as they do
today; the port creates **no new parity surface at all**. Contrast a
Kotlin or Rust reimplementation, which would need the whole harness in
`docs/native-app.md` built a second time.

## What ports for free

- **`native/core/`, minus `audio/qt/` and `overlay/render.cpp`**:
  17,523 lines, no Qt, no platform assumptions beyond `std::filesystem`
  and a handful of `getenv` calls (see "Paths").
- **The seams already exist, and they were built for a different
  reason.** `rx::Decoder`, `tx::Encoder`, `tx::Player`,
  `optimize::GradFn`/`GradFactory`, `rig::Backend`,
  `checkpoint::Fetcher` — every heavyweight or platform dependency in
  the app is behind a `std::function` or an abstract base, because the
  project wanted `--no-codec` builds and hardware-free tests. A second
  platform needs exactly the same seams in exactly the same places, and
  gets them at no cost.
- **onnxruntime has an official Android build at the pinned version.**
  Verified on Maven Central: `com.microsoft.onnxruntime:onnxruntime-android`
  publishes **1.28.0**, the same version `native/cmake/onnxruntime.cmake`
  and `pyproject.toml` pin. The AAR carries `headers/onnxruntime_cxx_api.h`
  — the exact C++ API `core/codec/` uses — and
  `jni/arm64-v8a/libonnxruntime.so` (28.6 MB raw, 10.6 MB compressed).
  It is the **full** build, not the reduced `onnxruntime-mobile`, so the
  published graphs load with no re-export.

  This matters more than it looks. The codec's parity claim rests on
  "the same runtime version, two builds" — Android satisfies that
  exactly, which puts it in a *better* position than macOS x86_64, the
  one platform already accepted as a lower compatibility tier because
  onnxruntime stopped publishing a build for it.
- **pocketfft, stb, dr_libs, easyexif**: header-only or vendored C with
  no platform surface.

## The audio layer

This is the risky piece and the one decision worth spelling out, since
it is where the app is won or lost.

### Why not QtMultimedia

Qt 6's Android audio backend is reported not to honour a `QAudioDevice`
selection for capture — the account is that `QAndroidAudioSource`
compares the device id against a few presets rather than using it. That
report is a forum thread, not a measurement of ours, and it might be
wrong or fixed. It does not need resolving: the app has to reach a USB
interface by name, and building that on a backend whose device selection
is *in question* is a bad trade when the alternative is small and the
layer beneath it is already device-agnostic.

Dropping QtMultimedia also shrinks the Qt module set to Core, Gui, Qml,
Quick and Network, and removes one Android backend from the list of
things that can surprise us.

### Java `AudioRecord`, not AAudio/Oboe

Refining "native audio" one step further, because the obvious NDK answer
is the wrong one here:

- **We have no latency requirement at all.** The capture layer
  deliberately buffers 2 seconds and the decode loop polls every 5. Low
  latency is the entire reason AAudio exists, so its headline benefit is
  worth exactly nothing to this app.
- **`setDeviceId` on AAudio is honoured only when the underlying API is
  AAudio (API 28+), silently ignored on the OpenSL ES fallback** — a
  setting that quietly does nothing on some devices is the shape of bug
  this project has spent the most time on.
- **USB capture through AAudio has open glitch reports.** Since USB is
  the whole point of choosing a native layer, taking the path with
  reported USB trouble to gain latency we do not want is backwards.
- **Enumeration needs Java regardless.**
  `AudioManager.getDevices(GET_DEVICES_INPUTS)` is the only way to see
  what is attached, so a Java layer exists either way.

So: **a Java thread doing blocking `AudioRecord.read()`**, pushing each
chunk across JNI into the existing Qt-free pipeline. Worth noticing what
that architecture *is* — it is the blocking-read design the desktop app
wanted and could not have. PortAudio's blocking API was the right answer
to the GIL bug and had to be abandoned because `stream.read()` corrupts
the heap on its JACK backend. Android has no such obstacle: there is no
interpreter on any thread, and the reader is a plain blocking loop
rather than a realtime callback, so the hazard class that cost 5 dB is
absent by construction rather than by tuning.

### Shape

`core/audio/android/`, a new optional library
(`SSTVAE_BUILD_ANDROIDAUDIO`) mirroring `core/audio/qt/`'s header
**entry point for entry point**, so nothing above it changes:

    input_device_names()   output_device_names()
    default_input_name()   default_output_name()
    class InputStream      play()

`play()` already matches `tx::Player`, so it drops into `TxEngine`'s
seam unchanged and the PTT guarantee is unaffected by which player is in
use. Seven entry points is the whole surface.

**Everything with logic in it stays where it is.** `bytes_to_mono`, the
sample-format conversions, `resample_ratio`, `StreamResampler` and
`match_device` are in `core/audio/audio.hpp`, Qt-free and tested against
a fake device — precisely because every audio bug this project has had
lived there and not in the code talking to the driver. The Android layer
is enumeration and moving bytes, same as the Qt one, and inherits the
three rules that were bought with real losses: open at the device's own
rate and resample in our code, resample *statefully* across blocks, and
never hold the ring buffer's lock across a bulk copy.

Two rules for the JNI boundary, both cheap to keep and expensive to
discover:

- **Java calls into C++, never the reverse on the data path.** The
  reader thread is already attached; a C++ thread calling back into Java
  needs `AttachCurrentThread` and gives nothing in return. Chunks land
  in a direct `ByteBuffer` and cross once per read.
- **Device names must be stable strings**, because `match_device`
  matches against what the config file stored — the same reason the
  desktop stores a human-readable description rather than an opaque id.
  `AudioDeviceInfo.getProductName()` plus the type is the candidate.
  Note it is not unique when two identical interfaces are attached; the
  desktop has the same ambiguity and lives with it.

`tools/check_layering.py` gets one more clause: **only
`core/audio/android/` may include `<jni.h>`.** Same rule and same reason
as Qt Multimedia's — the engines stay drivable with no platform audio at
all, which is what keeps the headless tests possible.

### What still has to be measured on hardware

Everything expensive this project has found in audio was found on real
hardware and was invisible to unit tests: 5 dB from a GIL-starved
callback, 4.7 dB from per-chunk resampling, 0.35% of clock error from a
lock held across a bulk copy. Budget for the same here, and build the
Android equivalent of `sstvae-audio-check --loopback` early — it exists
for exactly this and it is why the desktop soundcard path is trusted.

Open questions, in the order they can hurt:

1. Does a target phone capture from a class-compliant USB interface at
   all, at what rate, and does the existing `StreamResampler` path
   decode what comes out? Note 48 kHz → 8 kHz is an exact 1:6, the
   friendliest case the resampler has.
2. Does a USB audio-class device need a permission prompt of its own?
   Class-compliant audio is routed by the framework rather than claimed
   through the USB host API, so it should not — but "should not" is not
   a measurement, and getting it wrong is a first-run failure.
3. Bus power. A phone feeding an interface over OTG may need a powered
   hub, which is a documentation problem, not a code one.
4. What happens on device disconnect mid-reception, and on an incoming
   call taking the microphone.

## The smoke test (2026-08-08)

`native/android/` is a pre-Tier-0 app whose only job is the question
above: select an audio device, receive a signal. It is **not** the Tier 0
interface and does not follow the UI section — no service, no gallery, no
waterfall, no notification. It is a spinner, a button and an ImageView.

It needs **no Qt at all**, which is the useful part: device selection plus
receive is the Java audio layer, the C++ core and JNI, so a plain Views UI
does it and the Qt Quick decision stays entirely ahead of us. Everything
reusable is reusable: `audio::CapturePipeline` went into the Qt-free layer
with a host test, and the Java `AudioDevices`/`CaptureThread` are close to
what Tier 0 wants.

**What it proved, on an x86_64 emulator (API 36):**

- The whole chain runs on Android: `AudioRecord` → JNI → `CapturePipeline`
  → `RingBuffer` → `decode_loop` → onnxruntime → a picture. Fed a mode A
  transmission at 48 kHz in ragged chunks, it locked the **preamble path,
  reported mode A with a high SNR and correct progress**, and saved a
  correct 640x480 picture — 35.4 dB against a host decode of the same
  transmission (they differ only because the host decoded a separate noisy
  capture of it).
- onnxruntime 1.28.0's Android AAR works against `core/codec/` unmodified.
  The APK carries `libonnxruntime.so` at 28.6 MB for arm64 and 34.6 MB for
  x86_64; the debug APK with both ABIs is 72 MB.
- The `XDG_CONFIG_HOME`/`XDG_CACHE_HOME` trick holds: no path code changed.

**On a real phone it works end to end** (Andrew, 2026-08-08): acoustic
coupling into the built-in microphone, decoded, picture displayed. That
is the core of Tier 0 demonstrated on real hardware — capture, sync,
framing, the beacon, onnxruntime and the display, on a device, off the
air-ish. **The USB path is the remaining unknown** and is being tested
next; everything in "What still has to be measured on hardware" above
that concerns a class-compliant interface is still open.

Worth keeping in proportion: acoustic coupling was described earlier in
this document as the zero-hardware *fallback*, the thing that makes a
first release not depend on the USB question resolving well. It is now
demonstrated rather than assumed, which is exactly the bar that claim
needed.

**The emulator, by contrast, never exercised the microphone path at
all**, because it hands back **zeroed audio**. That diagnosis took two
wrong turns, both worth recording because the second is the kind that
ends an investigation prematurely:

- The first reading was "the emulator's mic is deaf" — from the captured
  RMS being 1/60th of the reference with two thirds of its power out of
  band. **Wrong, and it was wrong in the direction of giving up.** The
  capture had 3 s of silence, then **31.5 s of activity against a 32 s
  transmission**, then silence. Something was getting through; a deaf
  device does not keep time with the signal.
- What it actually is: the byte rate is *correct* (96037 against 96000
  expected, so no underrun and no rate error) while **99.9% of samples
  sit below 16 LSB** with occasional full-scale impulses — kurtosis 175
  against the reference's 2.18. Near-silence with clicks, not attenuated
  audio. `emulator -help` names it exactly: **`-allow-host-audio`**,
  "Allows sending of audio from audio input devices. *Otherwise, zeroes
  out audio.*" Passing that flag alone did not change it, and the
  PulseAudio routing was verified correct at the time (the emulator's
  source-output really was attached to `sstvae_loop`, the same source a
  `parecord` decodes from at **34.6 dB**). So the remaining gap is inside
  the emulator's audio-input plumbing and is not worth further time —
  the real measurement is a real device.

**The one bug the hardware run did find was about error *timing*, and it
generalises.** The first attempt reached the decoder and then reported a
missing model file — a provisioning gap (there is no Hub fetcher without
QtNetwork), not an audio finding.

`OnnxCodec` loads its parts lazily and independently on purpose — a
receive-only station never fetches the encoder — and laziness put that
failure at the worst possible moment: "file doesn't exist" **after the
operator had waited through an entire transmission**, with every step
until then reporting success. A missing prerequisite has to fail when
the session starts. `preload` is in the codec's API for precisely this
("only useful for surfacing a missing-artifact error early") and Tier 0
must call it too, for the model *and* for anything else it needs before
it claims to be listening.

**Our side is exonerated, and by construction rather than by argument.**
The level meter reads `buf.getShort(i)` straight off `AudioRecord` *before*
`Native.push`, so the zeros are upstream of every line of our code. And
the WAV feeder is the control: identical `Native.push` → `CapturePipeline`
→ ring path, ragged chunks, real time — and it decodes a picture.

The general lesson, which is the one this project keeps relearning: **a
capture path that reports the right byte rate can still be delivering
nothing**, and the distribution is what tells you. Mean level cannot: a
quiet path and a silence-plus-clicks path have the same RMS and are not
remotely the same failure. Log the percentiles, not the average.

Two things worth keeping:

- **`Native.dumpAudio` is the diagnostic that settled it**, and it is the
  Android form of the desktop's `receive.save_audio`: dump the ring, pull
  it, decode it on the host. Comparing two captures of one playback is how
  the PortAudio/JACK sample loss was pinned, and it is what separated "the
  emulator's mic is deaf" from "our pipeline is broken" in one step here.
- **After a reception is saved, the display goes wrong** (Andrew's
  observation): the loop keeps running, the blind accumulator re-integrates
  the transmission still sitting in the ring, and mode A at a good SNR is
  replaced by "mode C, −3.2 dB" a few seconds later. That is a live
  instance of exactly why Tier 0 must **persist reception metadata beside
  the picture** rather than read it from shared state — the desktop's
  last-reception card is a workaround for the same thing, and on a phone
  nobody is watching the moment it happens.

## Everything else that has to be written

| Piece | Today | On Android |
|---|---|---|
| Paths (config, model cache) | `getenv` of `HOME`/`XDG_*`/`LOCALAPPDATA` | Two env vars set from JNI at startup. Near-free; see below. |
| Saving received pictures | `std::filesystem` into `received/` | MediaStore or SAF, so the gallery can see them. New. |
| Model download | `core/checkpoint/qt_fetcher.cpp`, QtNetwork, behind a `Fetcher` seam | **Keep it.** QtNetwork is in the module set anyway now that Qt Quick is the UI, so this is free. The manual-redirect requirement is unchanged — the client must not auto-follow, or the `x-linked-etag` checksum on the 302 is lost. |
| Overlay rendering | `core/overlay/render.cpp`, 329 lines, QtGui only | Unchanged. QtGui is in the module set; Tier 1+ only. |
| UI | 6,170 lines of QtWidgets | Rewritten in Qt Quick; see below. |
| Rig control | `core/rig/hamlib.cpp` + libhamlib | Dropped. See below. |

### Paths: cheaper than it looks

`core/settings/settings.cpp` and `core/checkpoint/checkpoint.cpp` both
already honour `XDG_CONFIG_HOME` and `XDG_CACHE_HOME` (and
`SSTVAE_MODEL_CACHE`), on every platform, deliberately. So setting those
two environment variables from the JNI entry point to the app's
`filesDir` and `cacheDir` — before anything reads them — makes the whole
path layer work on Android with **no source change**. Worth doing that
way rather than adding an Android branch: an `#ifdef __ANDROID__` in
those files is a fourth platform's worth of untested code in the one
place where getting it wrong means silently writing a config nobody
reads.

Note what this does *not* cover: a received picture written to
`filesDir` is invisible to the user. Getting pictures into the gallery
is MediaStore work with no desktop counterpart.

### Background execution

A reception is 32–95 s and a listening session is hours. Android needs a
foreground service with `foregroundServiceType="microphone"` and the
`FOREGROUND_SERVICE_MICROPHONE` permission (Android 14+) on top of
`RECORD_AUDIO`; screen-off capture is only allowed under that. Add
battery-optimisation exemption prompts, audio focus, and surviving an
incoming call taking the microphone — the decode loop has to resume
rather than wedge, and what happens to the ring buffer's history across
that gap is a decision someone has to make.

None of this is difficult. All of it is new code with no desktop
counterpart, and it is where an app that "just works" is won or lost.

### Rig control drops for a structural reason, not a scoping one

`sstvae_rig` links libhamlib, which opens `/dev/ttyUSB*`. Android hands
an unprivileged app no such node: USB serial goes through the Java USB
host API, where `usb-serial-for-android` is the standard library
(FTDI/CDC/CP210x/PL2303, **no root**). Hamlib's serial layer opens a
*path* and has no way to be handed an already-open descriptor, so using
it would mean patching Hamlib — a fork of a pinned dependency, in the
one area where "which radios work" is already a per-platform property we
went to some trouble to avoid.

**So: VOX for PTT, no CAT, in any transmitting version.** That is what
"RX+TX without rig control" actually costs — the operator arms VOX and
reads the frequency off the radio's own display.

Two things make this less final than it sounds. `rig::Backend` is a
seam, and `RigController`'s threading design (one worker, PTT as
priority work, `stop()` detaches) has no external dependency and ports
unchanged — so a CAT backend can be added later without restructuring
anything. And **Hamlib model 2 (NET rigctl) needs no USB at all**: a
phone on the same wifi as a station running `rigctld` gets full CAT over
TCP. Since the wire format is trivial ASCII, that is roughly 200 lines
of `rig::Backend` that does not link Hamlib — which on Android is a much
better trade than cross-compiling an autotools tarball for four ABIs.

## The front end

**This is not a port of the desktop UI, and it should not be read as
one** (Andrew, 2026-08-08). The app gets the interface appropriate to a
phone and to the feature set actually shipped. The desktop layout
history in `CLAUDE.md` — splitter versus tabs, the picture box's
minimum-height ratchet, `QFormLayout` truncation, stylesheet versus
`QPalette` — is a record of *QtWidgets on a desktop*, and reaching for
it here would be inheriting the answers to questions nobody is asking.
What survives from it is a short list of rules, at the bottom of this
section.

### The service owns the engine, and the UI is a detachable view

The one consequence that is architectural rather than cosmetic, and it
is forced by the platform rather than chosen.

On the desktop, `AppState` owns the codec and the engines and the window
owns `AppState`: the GUI is the process. On Android the process is the
**foreground service** — it has to be, because a listening session must
survive the screen going off, the app being backgrounded, and the task
switcher. So the ownership inverts: the service holds the ring buffer,
the audio layer, the codec and `decode_loop`, and the UI attaches to it,
reads shared state, and detaches without disturbing anything.

That has a concrete cost if it is discovered late and almost none if it
is designed in: nothing in the UI may own engine state, and every live
display has to be reconstructible from `SharedState` on attach rather
than accumulated by watching. It also has a concrete benefit — **with no
UI attached, rendering stops entirely** while decoding continues, which
is most of the battery answer for a multi-hour session and is not
something the desktop app has any equivalent of.

### The phone can tell you, and that is what it is for

A desktop listener requires you to be at the desk. A phone in a pocket,
cabled to a radio, can put a decoded picture on the lock screen. That is
not a nicety bolted onto a port — it is the reason to want this app at
all, and it should be designed first rather than treated as an Android
obligation to be discharged.

Two surfaces, both with no desktop counterpart:

- **The ongoing notification** is where "listening / receiving, 43% /
  decoded" lives. It is the service's obligation anyway, so the only
  question is whether it is informative or boilerplate.
- **A completed reception posts a picture notification.** Big-picture
  style, so the image itself is on the lock screen.

This also settles something the desktop got wrong twice.
`rx/engine.cpp` wipes mode, callsign, SNR and frame count from shared
state two seconds after a reception; the desktop answer was a
"last reception card" that keeps them on screen. On a phone the operator
is frequently *not looking*, so state that expires is worthless —
**reception metadata has to be persisted beside the picture**, not held
live, and shown in the gallery whenever the picture is opened. The
desktop card was a workaround; persisting is the actual fix, and the
phone is what makes that obvious.

### The waterfall is the tuning instrument, not a diagnostic

With no CAT there is no frequency readout, no rig chip, nothing to say
where the radio is pointed. The waterfall with band markers is the only
tuning feedback the operator has, which makes it *more* important here
than on the desktop, where it competes with a rig panel.

Two things follow. It gets real vertical extent, which portrait suits
better than any desktop arrangement did — the desktop squashed it into a
horizontal strip to fit beside a picture pane, and that constraint is
absent. And `core/dsp/spectrum.cpp`'s peak-hold in `reduce_to_width`
matters more, not less: the carriers are one or two bins wide and about
six apart, so point-sampling leaves a ragged comb that reads as a
*reception* problem — on a display whose entire job is to tell you
whether you are tuned correctly, that is the worst available lie.

### Screens

Three, not panels:

- **Listen** — waterfall, live status, the picture currently arriving.
  The only screen that exists while nothing has been received.
- **Pictures** — a grid of what has been received, each with its
  metadata. This is the app's actual product and the desktop has nothing
  like it; `received/` in a file manager is not a gallery.
- **Settings** — an Android preference list: audio source, model
  precision, save location, keep-screen-on. Not a tabbed dialog.

Tier 0 has no transmit screen, no overlay editor, no crop dialog, no rig
chip, no PTT lamp, and no log dock — `core/log/`'s `FileWriter` still
runs for bug reports, reachable from Settings, but a dock is a desktop
answer to a desktop problem.

Portrait is primary; landscape should work, since the picture is 4:3.
Use the Material style so controls look like the platform's rather than
like Qt's idea of a desktop.

### What carries over

Short, and none of it is layout:

- **Errors must never be written where something else will overwrite
  them.** The desktop's three tiers came from `"PTT OFF FAILED"` being
  destroyed by the `"Sent"` that followed it on the same label. The
  Android trap is the same shape with a different name: a `Snackbar` is
  transient, so it is the wrong home for an error, however convenient.
- **A stopped display must look deliberate.** Half duplex still suspends
  receive with a fresh ring buffer on resume (Tier 1), and a frozen
  waterfall is otherwise indistinguishable from a wedged capture.
- **The settings round-trip discipline.** `core/settings/` is Qt-free
  JSON at `CONFIG_VERSION` 2; Android drops the rig keys and
  `audio.backend` and gains a device selection. Whether that is a
  version 3 or an additive change the desktop ignores is open. What is
  not open is `test_settings_dialog.cpp`'s method — a fixture in which
  no field holds its default — because a field displayed but not written
  back is still the characteristic settings bug, in any toolkit.
- **A preview is `overlay::render()`'s output, never a toolkit-drawn
  imitation** (Tier 1+). The rule is what stops a second representation
  drifting from what goes on the air, and it is toolkit-independent.

**Qt 6.8 is the sensible floor**, which sets `minSdkVersion` to 28
(Android 9). Nothing else we need is above that — `getDevices()` and
`setPreferredDevice()` are API 23 — so Qt is the binding constraint.

## Scoping tiers

### Tier 0 — receive-only listener (the committed one)

USB or mic capture → `RingBuffer` → `decode_loop` → picture. Needs the
audio layer, a foreground service that owns the engine, storage-out with
metadata persisted beside each picture, model fetch, and the three
screens above. Does **not** need the overlay renderer, the editor, the
tx engine, the rig, the optimizer, the crop dialog, or settings for any
of them.

Only the decoder needs fetching — 9 MB, not 21 — because `load_codec`'s
per-part laziness already does that.

Acoustic coupling remains the zero-hardware fallback and should stay
supported, but it is the *fallback*: the app is worth having because it
takes a USB interface.

### Tier 1 — receive and transmit, VOX keying

Adds `core/tx/engine.cpp`, which ports unchanged. Its PTT guarantee
degenerates to "VOX drops when the audio stops" and `PttWatchdog` has
nothing to unkey — keep the state machine anyway, so CAT can be added
later without restructuring the transmit path. Adds audio *output*
routing, which is the same problem as input and is solved by the same
layer. Picture source is the camera or the gallery; `images::fit`
already handles the resize, and 320×240 is still the minimum accepted
input.

A callsign caption will be wanted on a ham mode, so
`core/overlay/render.cpp` comes in here even if the *editor* does not.

### Tier 2 — CAT, overlay editing, the refiner

`core/optimize/` is portable and needs only the fp32 gradient graph as a
third download. The cost is CPU and battery: the desktop default budget
is 20 s for ~65% of the achievable gain, and a phone should be measured
rather than promised. Note `optimize::run` takes its deadline from
`ProgressFn`, so a phone gives it fewer steps — no different
implementation is needed.

The overlay **editor** on touch is a redesign, not a port:
`overlay_editor.cpp` is mouse-pixel arithmetic with small drag handles.
The *document* is unaffected — coordinates are normalized 0..1 already,
which is the same property that was supposed to buy templates.

## Sizes and performance

**APK.** `libonnxruntime.so` is 28.6 MB raw / 10.6 MB compressed for
arm64-v8a. Qt Core/Gui/Qml/Quick/Network adds roughly 25–35 MB
uncompressed, plus our own core. Ballpark **55–75 MB per-ABI**, and an
arm64-only App Bundle keeps the user's download near that instead of
multiplying it by four ABIs. Model artifacts are **not** in the APK —
the 9 MB decoder is fetched on first run, unchanged from desktop.

**CPU.** The desktop figures are `demodulate` at 173 ms and an fp16
decode at 86 ms per five-second poll, on a 24-core x86 box with
`SetIntraOpNumThreads(4)`. A modern phone's big cores should land within
a small multiple of that, comfortably inside the duty cycle — but that
is arithmetic, not a measurement, and the number that actually matters
is battery over a multi-hour session, which does not follow from it.
`decode_loop_low_cpu` exists and is the right starting point.

**int8 may be the right default here, unlike on desktop.**
`docs/onnx.md` records the int8 slowdown as an x86 artifact and notes
dynamic quantisation is often *faster* on ARM — naming small devices as
exactly the case it would be for. Android is that case. If it holds,
int8 is 8.6 MB against fp16's 20.9, at the published accuracy cost
(−0.011 dB on photographs, +0.139 off-distribution). Measure on the
target before concluding anything in either direction, and remember int8
carries a compatibility-tier label already.

`SetIntraOpNumThreads(4)` is hardcoded in `core/codec/codec.cpp` and is
the one number worth revisiting on big.LITTLE.

## Licensing

- **Qt**: LGPLv3, dynamically linked, which is what `androiddeployqt`
  does. Do not static-link without a commercial licence — same rule as
  the desktop build.
- **onnxruntime**: MIT.
- **Hamlib**: LGPL-2.1+, moot now that rig control is dropped.
- **The icon is not free.** `NOTICE` covers `native/packaging/sstvae.svg`
  and the seven files generated from it under
  `LicenseRef-SSTVAE-Branding`; a store listing and a launcher icon both
  carry it. Nothing new, but a Play Store presence makes it more visible
  than a CI artifact does, and a fork publishing its own build must
  replace it. Android also wants its own icon sizes, which means
  `tools/gen_icons.py` grows a target — and it writes the REUSE sidecar
  beside each file it generates for exactly this reason.

## Sources

- [onnxruntime — Build for Android](https://onnxruntime.ai/docs/build/android.html)
- [Qt 6.8 — Android Platform Notes](https://doc.qt.io/qt-6.8/android-platform-notes.html)
- [Qt for Android supported versions guidelines](https://www.qt.io/blog/qt-for-android-supported-versions-guidelines)
- [Qt Forum — Qt6 AudioSource on Android: unable to select non-default source](https://forum.qt.io/topic/157041/qt6-audiosource-on-android-unable-to-select-non-default-source)
- [AAudio | Android NDK](https://developer.android.com/ndk/guides/audio/aaudio/aaudio)
- [Oboe — Full Guide](https://github.com/google/oboe/blob/main/docs/FullGuide.md)
- [Android — USB digital audio](https://source.android.com/docs/core/audio/usb)
- [mik3y/usb-serial-for-android](https://github.com/mik3y/usb-serial-for-android)

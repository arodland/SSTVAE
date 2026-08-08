# Android: feasibility

Assessment only — nothing here is implemented, and no decision has been
taken. Written 2026-08-08 against `native/` at v0.3.1.

## Verdict

**Feasible, and most of it is not C++ work.** The 17,523 lines under
`native/core/` that are free of Qt — the whole modem, both engines, the
codec wrapper, the optimizer, the ring buffer, settings, images —
compile for Android with no changes anyone has to argue about. CI
already builds and tests that code for `linux-aarch64`, so ARM is not a
new platform for it.

What has to be built is the platform edge: audio routing, background
execution, storage, and a touch UI. That is real work, and it is the
part with no existing test coverage and no oracle, which is also where
every expensive bug in this project's history has lived.

The single question that decides whether an Android app is *useful*
rather than merely buildable is **can it get audio to and from a
radio**, and it is answerable in about a week of work. Everything else
in this document is sizing.

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
  and a handful of `getenv` calls (see "Paths" below).
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

## What has to be written

| Piece | Today | On Android |
|---|---|---|
| Audio capture/playback | `core/audio/qt/`, 656 lines | New backend, ~600–900 lines. **The one that decides the project** — see below. |
| Paths (config, model cache) | `getenv` of `HOME`/`XDG_*`/`LOCALAPPDATA` | Two env vars set from JNI at startup. Near-free; see below. |
| Saving received pictures | `std::filesystem` into `received/` | MediaStore or SAF, so the gallery can see them. New. |
| Model download | `core/checkpoint/qt_fetcher.cpp`, QtNetwork, behind a `Fetcher` seam | Keep it if Qt is in; otherwise ~150 lines over `HttpURLConnection`. **The manual-redirect requirement carries over** — whatever client is used must not auto-follow, or the `x-linked-etag` checksum on the 302 is lost. |
| Overlay rendering | `core/overlay/render.cpp`, 329 lines, QtGui only | Works under Qt on Android unchanged. Without Qt it is a re-implementation, and the font handling is the fiddly part, not the drawing. |
| UI | 6,170 lines of QtWidgets | See "The UI decision". |
| Rig control | `core/rig/hamlib.cpp` + libhamlib | Drop, or re-derive behind the existing `rig::Backend`. See below. |

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

## The three genuinely hard parts

### 1. Audio routing to the radio — the real risk

The desktop app's entire audio design assumes you can name a device and
open it. On Android that assumption is shaky in a specific way:

- **Qt 6's Android audio backend reportedly does not honour a
  `QAudioDevice` selection for capture** — the account is that
  `QAndroidAudioSource` compares the device id against a few presets
  rather than using it. If that holds, `audio::match_device` and the
  whole settings picker have nothing to act on under QtMultimedia.
  **This is a forum report, not a measurement of ours** — see "What to
  measure first".
- **Android's own API does support it**:
  `AudioManager.getDevices(GET_DEVICES_INPUTS)` plus
  `AudioRecord.setPreferredDevice()`, and class-compliant USB interfaces
  appear there as `TYPE_USB_DEVICE`/`TYPE_USB_HEADSET`. That is the
  argument for a Java/AAudio audio layer rather than QtMultimedia, and
  it is **independent of the UI decision** — the audio backend and the
  toolkit can be chosen separately, which is precisely what
  `core/audio/audio.hpp` being Qt-free buys.

Station setups, in descending order of how well they will work:

- A class-compliant USB interface (Digirig, SignaLink, or the radio's
  own USB codec) over OTG. Needs the device selection above, and needs
  the phone to supply bus power or a powered hub.
- **Acoustic coupling** — phone mic in front of the radio's speaker.
  Needs no routing at all, and is a perfectly legitimate first target
  for a receive-only app. It is a much lower bar than the desktop app
  ever had to clear on day one.
- A TRRS cable into the headset jack. Fine where the jack still exists.

**Budget for hardware-only bugs.** Everything expensive this project has
found in audio was found on real hardware and was invisible to unit
tests: 5 dB from a GIL-starved callback, 4.7 dB from per-chunk
resampling, 0.35% of clock error from a lock held across a bulk copy.
There is no reason to expect Android to be the exception, and
`sstvae-audio-check --loopback` — which exists because of exactly this —
would need an Android equivalent.

Two structural advantages carry over. The realtime-thread hazard that
cost 5 dB has no analogue here (there is no interpreter to block), and
the rules that were bought with the other two — resample statefully,
open at the device's own rate, never hold the ring lock across a copy —
already live in the Qt-free layer, so any new backend inherits them
rather than rediscovering them.

### 2. Background execution

A reception is 32–95 s and a listening session is hours. Android needs a
foreground service with `foregroundServiceType="microphone"` and the
`FOREGROUND_SERVICE_MICROPHONE` permission (Android 14+) on top of
`RECORD_AUDIO`; screen-off capture is only allowed under that. Add
battery-optimisation exemption prompts, audio focus, and surviving an
incoming call taking the microphone — the decode loop has to resume
rather than wedge, and the ring buffer's history across that gap is a
decision someone has to make.

None of this is difficult. All of it is new code with no desktop
counterpart, and it is where an app that "just works" is won or lost.

### 3. Rig control — drop it, and the reason is structural

`sstvae_rig` links libhamlib, which opens `/dev/ttyUSB*`. Android hands
an unprivileged app no such node: USB serial goes through the Java USB
host API, where `usb-serial-for-android` is the standard library
(FTDI/CDC/CP210x/PL2303, **no root**). Hamlib's serial layer opens a
*path* and has no way to be handed an already-open descriptor, so using
it would mean patching Hamlib — which is a fork of a pinned dependency,
in the one area (`docs/native-app.md`) where "which radios work" is
already a per-platform property we went to some trouble to avoid.

The alternative is a `rig::Backend` that speaks CAT over the Java layer
directly for the handful of operations actually used — PTT, frequency
read, mode set. The seam already exists, and `RigController`'s threading
design (one worker, PTT as priority work, `stop()` detaches) has no
external dependency and ports unchanged. That is a real project, not a
weekend, and it is per-radio rather than per-platform.

**So: VOX for PTT, no CAT, in any first version.** That is what "RX+TX
without rig control" actually costs — the operator arms VOX and reads
the frequency off the radio's own display.

One free consolation: **Hamlib model 2 (NET rigctl) needs no USB at
all.** A phone on the same wifi as a station running `rigctld` gets full
CAT over TCP, and that path is already just a model number in the
existing picker. It would want a real Hamlib to speak the protocol —
or, since the wire format is trivial ASCII, about 200 lines of
`rig::Backend` that does not link Hamlib at all. On Android that is
probably the better trade, given `hamlib.cmake` builds from an autotools
tarball that would need NDK cross-compilation for four ABIs.

## The UI decision

1. **Ship the QtWidgets UI as-is.** Cheapest by a wide margin — the
   6,170 lines compile — and genuinely bad on a phone: desktop hit
   targets, no gestures, dialogs sized for a monitor. Note that
   `picture_box.cpp` and `overlay_editor.cpp` have *already* fought Qt's
   layout over minimum heights on small screens; that work helps a
   tablet and does not make a phone good. Right answer for a spike, not
   for a release.
2. **A Qt Quick front end over the same core.** One codebase, one
   toolchain, touch-idiomatic. The waterfall and the picture box become
   QML items over the same `core/dsp/spectrum.cpp`; `rx_panel` and
   `tx_panel` are largely wiring and translate rather than port. Costs a
   rewrite of most of the 6,170 lines. Qt for Android is LGPLv3 and
   `androiddeployqt` bundles Qt as shared `.so`s, which satisfies the
   relinking obligation the same way the desktop build does.
3. **Kotlin/Compose over a JNI'd core.** The best-feeling app and the
   most work, plus a second language and an FFI boundary — which is
   precisely the trade `docs/native-app.md` rejected for "Rust core + Qt
   front end", and the reasoning transfers intact.

**Recommendation: (2) for a real app, (1) to answer the audio question
first.** The audio question decides the project and is answerable under
either, so there is no reason to spend the UI budget before knowing it.

## Scoping tiers

### Tier 0 — receive-only listener

Mic or USB capture → `RingBuffer` → `decode_loop` → picture. Needs audio
in, a foreground service, storage-out, model fetch, a picture view and a
waterfall. Does **not** need the overlay renderer, the editor, the tx
engine, the rig, the optimizer, the crop dialog, or settings for any of
them.

The appeal is that acoustic coupling makes tier 0 **usable with no cable
at all**, so a first release does not depend on the USB question
resolving well. Only the decoder needs fetching — 9 MB, not 21 — because
`load_codec`'s per-part laziness already does that.

### Tier 1 — receive and transmit, VOX keying

Adds `core/tx/engine.cpp`, which ports unchanged. Its PTT guarantee
degenerates to "VOX drops when the audio stops" and `PttWatchdog` has
nothing to unkey — keep the state machine anyway, so CAT can be added
later without restructuring the transmit path. Adds audio *output*
routing, which on Android is the same problem as input and is solved by
the same layer. Picture source is the camera or the gallery;
`images::fit` already handles the resize, and 320×240 is still the
minimum accepted input.

Skipping the overlay editor means skipping `overlay/render.cpp` too, if
transmissions are the bare picture — but on a ham mode a callsign
caption will be wanted, and under option (2) the renderer comes free.

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
arm64-v8a. Qt Core/Gui/Quick/Multimedia adds roughly 25–40 MB
uncompressed depending on modules, plus our own core. Ballpark
**60–80 MB per-ABI**, and an arm64-only App Bundle keeps the user's
download near that instead of multiplying it by four ABIs. Model
artifacts are **not** in the APK — the 9 MB decoder is fetched on first
run, unchanged from desktop.

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
- **Hamlib**: LGPL-2.1+, moot if dropped.
- **onnxruntime**: MIT.
- **The icon is not free.** `NOTICE` covers `native/packaging/sstvae.svg`
  and the seven files generated from it under
  `LicenseRef-SSTVAE-Branding`; a store listing and a launcher icon both
  carry it. Nothing new, but a Play Store presence makes it more visible
  than a CI artifact does, and a fork publishing its own build must
  replace it.

## What to measure first

In this order, because the first one can end the project:

1. **Can a target phone capture from a class-compliant USB interface at
   all**, at what rate, and does the existing `StreamResampler` path
   decode what comes out? One evening, on tier 0 built as option (1).
2. **Is the Qt Android device-selection limitation real** at current Qt,
   or does `QT_MEDIA_BACKEND=android` or a newer release honour the
   selection? The claim above is a forum report; if it is wrong, the
   audio layer is much cheaper than this document assumes.
3. **Battery per hour of listening**, with the screen off.
4. **int8 against fp16 on the target's CPU**, which `docs/onnx.md`
   explicitly declines to predict from the x86 numbers.

## Sources

- [onnxruntime — Build for Android](https://onnxruntime.ai/docs/build/android.html)
- [Qt 6 — Android Platform Notes](https://doc.qt.io/qt-6/android-platform-notes.html)
- [Qt Forum — Qt6 AudioSource on Android: unable to select non-default source](https://forum.qt.io/topic/157041/qt6-audiosource-on-android-unable-to-select-non-default-source)
- [Android — USB digital audio](https://source.android.com/docs/core/audio/usb)
- [mik3y/usb-serial-for-android](https://github.com/mik3y/usb-serial-for-android)

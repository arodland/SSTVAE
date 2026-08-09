# SSTVAE for Android (Tier 0)

The receive-only listener from `docs/android.md`: a Qt Quick front end
over the **same** `native/core/` the desktop app uses. An Android build
is a fourth build of that code, not a reimplementation, which is what
keeps the golden vectors and `pytest --native` covering it with no new
parity surface.

Not to be confused with **`native/android/`**, the pre-Tier-0 smoke
test — a plain Views probe that answered "can a phone select an audio
device and decode a picture". It can: a Kenwood TH-D75 over USB decodes
at over 24 dB, and acoustic coupling into the built-in mic works too.
This is the app that question was asked on behalf of.

**Status: the audio path is real, the UI is a probe.** Qt Quick,
`sstvae_core` and the onnxruntime AAR build and package together for
both ABIs, and — verified on the emulator 2026-08-08 — the app
enumerates the device's real inputs through JNI, captures through
`core/audio/android/` into a `RingBuffer` at the device's own 48 kHz,
resamples through `CapturePipeline`, and runs `rx::decode_loop` on it:
a clean mode A transmission reported **mode A, callsign KC2G**, and a
low-SNR one reported a beacon-only lock, which is the right answer to
each. None of Tier 0's *screens* exist yet — see the plan below.

Two things that first run settled beyond the audio layer. **The engine
wipes a reception's metadata two seconds later**, so the completed
decode was gone before a screenshot twelve seconds on could catch it —
which is the persistence requirement in `docs/android.md` demonstrated
rather than argued. And the level readout earns its keep: `peak silent
100.0% near-zero` is what a dead capture path looks like, and it is
distinguishable at a glance from merely quiet, which a mean level is
not.

## Building

Needs Qt for Android **matching your host Qt version**, the NDK, and the
SDK. Both kits via [aqtinstall](https://github.com/miurahr/aqtinstall):

```sh
aqt install-qt linux desktop 6.11.1 linux_gcc_64 -O ~/Qt -m qtimageformats
aqt install-qt all_os android 6.11.1 android_arm64_v8a -O ~/Qt \
    -m qtimageformats qtshadertools
aqt install-qt all_os android 6.11.1 android_x86_64 -O ~/Qt \
    -m qtimageformats qtshadertools
```

Both Android ABIs: **arm64-v8a for phones, x86_64 for the emulator.**
176 MB each.

Two things that are easy to get wrong. Qt 6.8+ publishes Android under
the **`all_os`** host, not `linux` — under `linux` the listing stops at
6.7.3 and looks like Android support was dropped. And `qtdeclarative` is
part of the base install now, so asking for it as a module fails the
whole command.

```sh
export QT_ANDROID=$HOME/Qt/6.11.1/android_arm64_v8a
export QT_HOST=$HOME/Qt/6.11.1/gcc_64
export JAVA_HOME=/opt/android-studio/jbr

cmake -S native/android-app -B build-android -G Ninja \
  -DCMAKE_TOOLCHAIN_FILE=$QT_ANDROID/lib/cmake/Qt6/qt.toolchain.cmake \
  -DQT_HOST_PATH=$QT_HOST \
  -DQT_ANDROID_BUILD_ALL_ABIS=ON \
  -DANDROID_SDK_ROOT=$HOME/Android/Sdk \
  -DANDROID_NDK_ROOT=$HOME/Android/Sdk/ndk/28.2.13676358 \
  -DCMAKE_BUILD_TYPE=Debug
cmake --build build-android --target apk
```

This is a **CMake** build, not a Gradle one: `androiddeployqt` generates
the Gradle project. Build `Debug` unless you intend to sign — the
release target emits `-release-unsigned.apk`, which will not install.

```sh
adb install -r build-android/android-build/build/outputs/apk/debug/android-build-debug.apk
```

**Build both ABIs and keep the emulator in the loop** (Andrew,
2026-08-08). It is tempting to go arm64-only on the grounds that the
emulator cannot carry audio — but it can, and in any case that
mistakes the emulator's job. Most of Tier 0's work is *layout*, and
`adb install` + `adb exec-out screencap` is a build-to-picture loop with
no phone in hand, which is the Android equivalent of what
`sstvae-gui-shot` does for the desktop and is there for the same reason:
"is this laid out well" has no oracle and needs eyes, repeatedly.

**The emulator does carry real audio**, through the AVD's *Extended
Controls > Microphone > "Virtual microphone uses host audio input"*
(Andrew found it; it is a per-AVD GUI setting and **not** the same thing
as the `-allow-host-audio` command-line flag, which was the wrong answer
tried first). With it on, the host's default source reaches
`AudioRecord` unchanged: the desktop's own loopback recipe then feeds it,

```sh
pactl load-module module-null-sink sink_name=null-sink
pactl load-module module-remap-source source_name=sstvae_loop \
    master=null-sink.monitor channels=1
pactl move-source-output <qemu-system-x86_64's index> sstvae_loop
pw-play --target=null-sink transmission-48k.wav   # pre-resample!
```

and a whole transmission decodes on the emulator. Two traps. The qemu
capture stream has to be **moved to the loopback** — it opens on the
host's real default source, so without the move the app faithfully
records the room. And **pre-resample the file to 48 kHz**, the same rule
as the desktop: an on-the-fly 44.1k conversion cost ~4 dB there.

The **WAV feeder** (see `native/android/`, `WavFeeder.java`) is still
worth carrying over: it pushes a file through the capture path in ragged
chunks with no host audio stack involved at all, which is reproducible
in a way a live loopback is not.

**Use `-gpu host` for anything you intend to look at.** The default
`swiftshader_indirect` composites a diagonal tear across every
`screencap`, cutting through the system status bar as well as the app,
and it is indistinguishable from a real rendering bug: it cost one
round of investigating a "clipped" toolbar title that was never
clipped. On a machine with a GPU, `-gpu host` gives pixel-clean frames
and boots no slower. Keep swiftshader only where there is no GPU at
all.

A real phone stays the only place the *driver* can be tested.

**The emulator drops audio and a phone does not**, so its SNR figures
are worthless and its drift figures are the point: −7247 ppm measured
there against a phone's clean capture of a whole mode B transmission
(440/440, 17.4 dB) over nothing but acoustic coupling. Use it for
layout and for the state machine; never quote a number off it.

## The technical switch

**Everything numeric is behind Settings > Advanced > Show technical
details, off by default** (`ui/showTechnical` in `QSettings`). Poll
counts, ring depth, capture drift in ppm, peak dBFS, near-zero
fraction, decode cost, the device's sample rate — the readouts that
made several bugs findable at all, and also the first thing an operator
sees on a screen that should look like a radio.

Two rules kept it from becoming a UI with two personalities. **The
plain wording answers a different question, not a smaller one**: the
status line goes from "is the receiver working" (`listening polls 8
ring 45.0 s`) to "is a picture coming and how far along" (`Receiving
from KC2G 43%`), and the level meter's centred number becomes `Level
good` / `Quiet` / `Too loud`, tracking exactly the thresholds the bar is
already coloured by so the word and the colour cannot disagree. And
**anything actionable stays visible at both levels** — the routing
warning, which fires only when the audio is genuinely coming from
somewhere other than was asked for, and the error line.

The switch reaches `Session` as well as the QML, because the ongoing
notification is drawn by the service from native state and would
otherwise keep reporting poll counts with the switch off.

**Qt hex colours are `#AARRGGBB`, not `#RRGGBBAA`.** Written the CSS
way in `LevelMeter.qml`, every alpha landed in the red channel and the
alpha came out `0x00`: background, border and text fully transparent,
and the faint green "good" band painting as pale red. It still looked
like a widget — an offset pink rectangle — which is how it survived a
commit. The meter only draws while listening, so no idle screenshot
shows it.

## Two device-measured behaviours worth knowing

Both from Andrew's 2026-08-08 run, and both now handled rather than
merely observed.

**Capture drift is measured over a sliding 30 s window, not since the
stream opened.** A cumulative figure reports the session average, which
gets the timing wrong in both directions: a startup transient — and
there is one — stays on the meter long after the audio it describes has
been decoded (`DROPPING AUDIO` showed for the first stretch of a
session that then produced two clean pictures), while loss that
*begins* an hour in is diluted by the clean hour in front of it. The
second is the case the meter exists for: the emulator's own failure
looked exactly like that, SNR high early and falling.

**The poll interval backs off on slow devices**
(`RxConfig::max_decode_duty`, 0.5 here, 1.0 — off — everywhere else).
The desktop's decode is ~1% of its 5 s interval and needs nothing; a
mid-range phone's can approach the interval, at which point the device
decodes back to back, the UI starves, and the extra polls buy nothing,
since each one re-decodes a picture that has grown by one interval.
Adaptive rather than a bigger constant because the spread across
devices is the whole problem — a number slow enough for the worst phone
would make the best one needlessly stale. Nothing is lost by waiting:
the audio is still in the ring buffer, so backing off delays a picture
rather than dropping one. The cost that drives it is on the Listen
screen (`decode 2.3 s`), shown always, so the next report of "it feels
slow" arrives as a number from the device it happened on.

**The lever that is *not* available is a faster execution provider.**
The onnxruntime Android AAR exports NNAPI and nothing else — no
XNNPACK — and NNAPI is both deprecated from Android 15 and free to run
a graph in reduced precision on whatever accelerator it finds. That
would trade the codec's "same runtime, same artifact, exact" basis for
an unpredictable amount of speed, which is a bigger decision than a
laggy UI justifies. Thread count (`SetIntraOpNumThreads(4)`, measured
on a desktop) is the untested knob, and the `decode` readout is what
would settle it.

**`ANDROID_HOME` beats `ANDROID_SDK_ROOT` for the emulator's
system-image lookup**, and a profile that exports the first to a
different SDK costs an afternoon: the emulator reports
`/opt/android-sdk/system-images/… is not a valid directory` and exits,
naming a path nothing in the command asked for. Export *both*.

**Never set `QT_ANDROID_PERMISSIONS` by hand** — each element is a
nested `name;<value>` list, so a bare permission string is valid CMake
that emits invalid JSON, and androiddeployqt fails with a byte offset
into a generated file. Use `qt_add_android_permission(target NAME …)`.

## What is left

In roughly the order the doc argues for. The first two are **done**:

1. ~~**`core/audio/android/`**~~ — the permanent audio layer: seven
   entry points mirroring `core/audio/qt/qtaudio.hpp`, so `InputStream`
   and `play()` drop into the existing seams unchanged. Capture is
   verified end to end (see Status); `play()` is written but entirely
   unexercised, which is a Tier 1 concern.
2. ~~**The foreground service**~~ — `ListenerService` plus `Session`,
   the process-wide owner of the ring, the stream and the engine
   thread. Verified with the screen off and the app backgrounded: the
   ongoing notification walked `Listening 19 polls` → `Receiving mode A
   92/220 frames KC2G` → `207/220` → `Listening`, with no UI attached at
   any point.

   Three things in it are load-bearing rather than incidental.
   `stopWithTask="false"` is what makes a reception survive the app
   being swiped away, which on a phone is a normal thing to do while
   waiting. **Only the service starts a session** — the UI asks it, and
   `Session::start` has exactly one caller — because a session started
   from the activity belongs to something Android may destroy
   mid-reception. And `Listener`'s destructor deliberately does
   *nothing*: a `stop()` there reads like tidiness and would end the
   session on every rotation.

   `START_NOT_STICKY`, not sticky: a restarted service arrives with a
   null intent and so no device, and silently opening the *wrong*
   microphone is worse than not restarting.

   **The notification carries a Stop action**, and it is not a
   convenience duplicate of the button on the Listen screen: after the
   app is swiped away — which `stopWithTask="false"` is specifically
   there to survive — it is the *only* control that exists, with the
   session still holding the microphone and no activity to return to.
   The UI needs no wiring for it, because the view already polls
   `Session::running()` rather than tracking its own button; stopping
   from the shade reverts the pane on the next tick, which is what that
   poll was written for. One trap: a `PendingIntent` is identified by
   (context, requestCode, intent-modulo-extras), so the action reuses
   neither the content intent's request code nor its target — 0 and
   `getActivity` against 1 and `getService`. Sharing a code would make
   one of the two silently become the other.
3. **Listen / Pictures / Settings**, with the waterfall over
   `core/dsp/spectrum.cpp` as the *tuning instrument* — with no CAT it
   is the only frequency feedback there is.
4. **Reception metadata persisted beside the picture**, not read from
   shared state, which wipes it two seconds after a reception.
5. **Notifications**, ongoing and per-reception. The phone being able to
   put a decoded picture on the lock screen is the reason to want this
   app, not an obligation to discharge.
6. **Model fetch** — `core/checkpoint/qt_fetcher.cpp` should port as is
   now that QtNetwork is in the module set.

## `core/audio/android/`

The permanent audio layer, entry point for entry point the same surface
as `core/audio/qt/` — so `InputStream` and `play()` drop into the
engines' existing seams unchanged. Built only when targeting Android
(`<jni.h>` and the NDK's `android`/`log` libraries have to exist), and
`tools/check_layering.py` enforces that nothing else under `core/`
includes `<jni.h>`, for the same reason it guards Qt Multimedia: the
engines must stay drivable with no platform audio at all.

Its Java half lives with it, at
`core/audio/android/java/org/cleverdomain/sstvae/AudioBridge.java`,
rather than in an app — an app supplying its own would be free to get
the blocking-read loop subtly wrong, and that loop is what the layer
exists to own. Consumers add that directory to their Gradle
`sourceSets`; `SSTVAE_ANDROID_AUDIO_JAVA_DIR` names it.

**Java calls into C++ on the data path and never the reverse.** The
reader thread is one we own and therefore already attached, so pushing a
chunk is a plain call; C++ calls into Java only to enumerate, open and
close, which is rare, off the audio path, and attaches explicitly.
Backwards, this would put an `AttachCurrentThread` on every buffer.

Three things carried over from what the hardware runs taught:

- **Routing is judged by `getRoutedDevice()` after the stream is live**,
  never by `setPreferredDevice`'s return value — measured against a
  TH-D75 over USB that returned false while routing correctly.
  `routing_warning()` is empty unless the audio is genuinely coming from
  somewhere other than the operator thinks.
- **`peak_level()` and `near_zero_fraction()` are both exposed**, because
  a path that is quiet and a path delivering silence have the same mean
  and are not the same failure.
- **`UNPROCESSED` first, `VOICE_RECOGNITION` as fallback.** AGC or noise
  suppression on an OFDM signal degrades it in a way that reads like a
  bad radio rather than a bad setting.

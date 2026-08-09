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

**Status: Tier 0 is built and has received real pictures.** Qt Quick,
`sstvae_core` and the onnxruntime AAR build and package together for
both ABIs; the app enumerates the device's real inputs through JNI,
captures through `core/audio/android/` into a `RingBuffer` at the
device's own 48 kHz, resamples through `CapturePipeline`, and runs
`rx::decode_loop` on it. On a **Galaxy S25+** (2026-08-08, Andrew's
measurement) it decodes complete pictures over nothing but acoustic
coupling, with no artifacts, a capture rate inside ±100 ppm and ~0.5 s
of DSP per poll. All three screens, the foreground service, the
notifications, sharing and the model fetch are in.

Two things that the first run settled and that shaped everything after.
**The engine wipes a reception's metadata two seconds later**, so a
completed decode was gone before a screenshot twelve seconds on could
catch it — the persistence requirement in `docs/android.md`
demonstrated rather than argued. And the level readout earns its keep:
`peak silent 100.0% near-zero` is what a dead capture path looks like,
and it is distinguishable at a glance from merely quiet, which a mean
level is not.

**Three of this port's bugs were bugs in its own instruments**, and
they are worth reading before trusting a number from here: the drift
meter charging in-flight audio as lost, the `-O0` build that made the
DSP look like a slow codec, and the emulator's renderer tearing a
screenshot that read as a layout fault. Each is written up below at the
place it bites.

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

tools/build_android.sh --install
```

**Use the script, and know what it is for.** The obvious command builds
at `-O0`, and the receive loop is scalar floating-point DSP over a
130 s ring buffer — the shape the optimizer matters most for. Measured
on this code: `sync::acquire` 171 ms → 1513 ms (**8.9x**), the blind
accumulator's full-buffer push 547 ms → 8412 ms (**15x**). On a Galaxy
S25+ an `-O0` build spent **5–8 s per poll with excursions to 20–40 s**
where the real figure is a fraction of a second, and it read as
"onnxruntime is slow on Android" — which it is not; the codec is not
even inside the number that was being looked at (see the technical
switch below). It also starved the capture thread: switching the
optimizer on visibly improved the emulator's measured capture rate,
though how much is no longer separable from the meter's own bias, fixed
later the same day. The figure that stands is the one after both:
**−4 ppm**.

Nothing was misconfigured. The trap is that the two obvious choices are
each half-right: the default configuration signs with the debug
keystore and installs but passes no `-O` flag at all (clang's default
is `-O0`), while `RelWithDebInfo` compiles at `-O2` and then emits
`-release-unsigned.apk`, which will not install. So everyone picks
Debug. The script configures `RelWithDebInfo`, then zipaligns and signs
with the same debug keystore Gradle would have used — installs like a
debug APK, runs at full speed. `--debug` is there for single-stepping
the C++ and for nothing else; every timing figure taken in that
configuration is fiction, and CMake says so at configure time.

Under the covers it is an ordinary CMake build — `androiddeployqt`
generates the Gradle project — so this is equivalent, minus the
signing:

```sh
cmake -S native/android-app -B build-android -G Ninja \
  -DCMAKE_TOOLCHAIN_FILE=$QT_ANDROID/lib/cmake/Qt6/qt.toolchain.cmake \
  -DQT_HOST_PATH=$QT_HOST \
  -DQT_ANDROID_BUILD_ALL_ABIS=ON \
  -DANDROID_SDK_ROOT=$HOME/Android/Sdk \
  -DANDROID_NDK_ROOT=$HOME/Android/Sdk/ndk/28.2.13676358 \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build-android --target apk
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

**The emulator's audio is worse than a phone's, but by much less than
this file used to claim.** The −7247 ppm recorded here earlier was
mostly two artifacts of our own: an `-O0` build starving the capture
thread, and the drift meter's endpoint bias (both below). With the
optimizer on and the meter fixed, the same emulator reads **−4 ppm**.
Treat its *SNR* figures with suspicion still — a phone captured a whole
mode B transmission at 440/440 and 17.4 dB over nothing but acoustic
coupling — but the sample-loss story it seemed to tell was largely our
instrument.

## The launcher icon

The same `native/packaging/sstvae.svg` the desktop uses, rendered by the
same `tools/gen_icons.py` — re-run it when the artwork changes, and look
at the result, which is the only check an icon has.

**An adaptive icon is two layers, not a picture.** Each is 108dp square
and the launcher guarantees only the central 72dp survives its mask,
which the user and the OEM choose (circle, squircle, teardrop). Handing
over the flattened desktop icon would put a hard circular edge inside
that mask and get it clipped into a lens on any non-circular launcher.
Ours splits cleanly because the SVG is a navy disc with the macaw on
top: the disc becomes a flat background colour, the bird becomes the
foreground, inset to 58% so it does not touch the mask edge.
`gen_icons.py` removes exactly one filled `<circle>` and **fails loudly
if it is not there**, because the silent alternative ships the clipped
version.

Two things that bit:

- **Nothing but `*.xml` and `*.png` may exist under `res/`.** The
  Android resource merger fails the build on anything else — which is
  at least loud, but it means the REUSE `.license` sidecars this
  project puts beside every generated icon are impossible here. The
  Android set is covered by `native/android-app/REUSE.toml` instead,
  written by the same generator so a new density cannot slip in
  unlabelled. The icon is licensed artwork and the repository's LICENSE
  does not cover it; see `NOTICE`.
- **`android:icon` is a Qt template token.** The manifest ships
  `-- %%INSERT_APP_ICON%% --`, which androiddeployqt substitutes; write
  the literal `@mipmap/ic_launcher` over it and the substitution simply
  does not happen.

**The notification icon is the same bird as a flat silhouette**, and
flat is the design rather than a shortcut. Android keeps only the alpha
and tints it, so a 24dp white shape is all there is to work with. Two
richer versions were tried on a device and are worse, both
instructively (2026-08-09):

- **Knocking out the dark markings shatters it.** Those lines are the
  outlines of the colour regions, not detail drawn on top of them, so
  removing them leaves disconnected fragments rather than a bird.
- **Knocking out the eye lands on the beak.** The darkest region in the
  head is the beak line; there is no pupil shape in the artwork big
  enough to survive 24dp. The hole ends up on the skull and reads as
  damage.

It reads as a perched parrot in the status bar next to the system
glyphs, and as something closer to a feather in the shade badge, where
it is smaller and tinted into a filled circle. Used for both the
ongoing notification and the picture one (Andrew, 2026-08-09), in place
of the system microphone glyph that was there before.

Note the directory: notification icons go in `drawable-`, **not**
`mipmap-`. mipmap is for launcher icons, which the system deliberately
keeps at densities other than the device's own so a launcher can
display them larger than life.

## The technical switch

**Everything numeric is behind Settings > Advanced > Show technical
details, off by default** (`ui/showTechnical` in `QSettings`). Poll
counts, ring depth, capture drift in ppm, peak dBFS, near-zero
fraction, DSP cost, the device's sample rate — the readouts that
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

**And both ends of the ratio come from chunk arrivals, never from
`now`.** Audio arrives in chunks, so the newest samples the device has
produced have not been handed over yet; timing the elapsed interval to
`now` while counting samples only to the last chunk charges that whole
in-flight gap as lost audio. The bias is −(time since last chunk) /
window — negligible on a desktop, large on a phone, and it read a
steady **−4500 ppm with `DROPPING AUDIO`** on an S25+ whose pictures
were decoding perfectly. The contradiction is what exposed it: 0.45% is
~3400 samples over a mode C transmission, and 1718 samples over 50 s
was enough to mangle a picture in this project's history, so "dropping
badly" and "perfect decodes" could not both be true. Taking both
endpoints from the same two chunk arrivals makes it exact; the emulator
went from −567 ppm to **−4 ppm** on the same session. The one place
`now` is still the honest end point is a *stalled* stream, where
otherwise both halves stop moving together and the meter reports a
serene 0 ppm — the worst answer available, and one neither `peak_level`
nor `near_zero_fraction` catches, since they keep returning whatever
the last chunk held.

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

**The readout says `dsp`, not `decode`, and the rename is the point.**
`Progress::last_decode_s` is measured *before* the codec runs, in both
implementations — it is sync plus demodulation, with no inference in it
at all. Labelled "decode" it sent a slow-app report straight at
onnxruntime while every millisecond of the number was in the DSP, and
the real cause was the build type. The adaptive backoff uses the true
whole-poll cost, measured separately, which does include the codec.

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

## Tier 0, and what is left

**All six of Tier 0's items are done** (2026-08-08), and the app has
received real pictures on a Galaxy S25+ over acoustic coupling with no
visible artifacts. What follows is kept as the record of what each one
turned out to require.

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
3. ~~**Listen / Pictures / Settings**~~ — three screens in a
   `StackLayout` behind a bottom `TabBar`, with the waterfall over
   `core/dsp/spectrum.cpp` as the *tuning instrument* it is: with no CAT
   it is the only frequency feedback there is. Insets come from
   `SafeArea.margins`, which is not optional — edge-to-edge is mandatory
   from targetSdk 35 and the tab bar painted *behind* a 3-button
   navigation bar until it was handled.
4. ~~**Reception metadata persisted beside the picture**~~ — a JSON
   sidecar per PNG, read by the Pictures list, because shared state is
   wiped two seconds after a reception and on a phone nobody is looking.
5. ~~**Notifications**~~ — an ongoing one with a Stop action, and one
   per reception carrying the picture itself as a `BigPictureStyle` on
   its own channel. The phone being able to put a decoded picture on the
   lock screen is the reason to want this app.
6. ~~**Model fetch**~~ — not the Qt fetcher after all. Qt for Android
   ships no TLS backend, so transport is `ModelFetcher.java`
   (`HttpsURLConnection`, redirects followed by hand to read
   `x-linked-etag`) behind the existing `checkpoint::Fetcher` seam,
   while the sha256 check and the `.part` rename stay in C++ — one
   implementation of the part that can silently corrupt a cache.

Not done, and deliberately so: **saving to the shared gallery**
(a MediaStore insert; Share reaches gallery, mail and chat for one
intent and no storage permission, so it went first) and **`play()`**,
which is written, unexercised, and a Tier 1 concern — Tier 0 does not
transmit.

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

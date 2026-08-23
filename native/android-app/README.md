# SSTVAE for Android (Tiers 0 and 1)

The listener from `docs/android.md`, and since 2026-08-09 the
transmitter as well: a Qt Quick front end over the **same**
`native/core/` the desktop app uses. An Android build is a fourth build
of that code, not a reimplementation, which is what keeps the golden
vectors and `pytest --native` covering it with no new parity surface.

Not to be confused with **`native/android/`**, the pre-Tier-0 smoke
test — a plain Views probe that answered "can a phone select an audio
device and decode a picture". It can: a Kenwood TH-D75 over USB decodes
at over 24 dB, and acoustic coupling into the built-in mic works too.
This is the app that question was asked on behalf of.

**Status: Tier 0 receives real pictures; Tier 1 transmits.** Qt Quick,
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

**Rig control is in as of 2026-08-22, and USB works on hardware**
(2026-08-23): CAT and both control-line keying methods, on a phone,
over a composite USB interface — which is the question that could have
sunk the approach, since the app needs the audio and the serial half of
that device at once. `docs/android.md` said this was structurally
impossible; it is not, because Hamlib takes a socket for any backend.
**Bluetooth is untested for want of a device.** See "Rig control" below
before touching any of it: the build-level traps and the runtime ones
are written up there, and every one of them cost a round.

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

## What Back does

Three behaviours, and which one applies is the whole design:

1. **Something is open on top** (the picture viewer) — Back closes it.
2. **A session is running or an over is in flight** — Back sends the
   task to the background. The app keeps running.
3. **Nothing is running** — Back leaves the app, as Android users
   expect.

Case 2 is the fix for a reported bug and case 3 is a second one found
while measuring it (2026-08-10).

**Ending the activity ends the process, and the process owns the
engine.** So the single most ordinary gesture on a phone silently killed
a reception in progress, and the shade went on claiming the station was
listening. `stopWithTask="false"` does not help here: that covers a
swipe from Recents, not the activity finishing. Measured on API 36
before the fix, one Back press at the root:

```
F libc     : FORTIFY: pthread_mutex_lock called on a destroyed mutex
F libc     : Fatal signal 6 (SIGABRT) in tid 28516 (hwuiTask1)
I tombstoned: received crash request for pid 28516
```

So an ordinary exit was **recorded as a native crash** — a tombstone,
which is exactly what Play's vitals count — and the notification's text
froze wherever the poller had last left it.

`moveTaskToBack` is what recorders, navigation and media apps do, and it
is what the ongoing notification already implies. The session
continues, the 2-second poller keeps the notification honest, the task
stays in Recents so the launcher or the notification returns to the
screen that was left, and Stop is still one tap away in the
notification — which is where an operator already looks for it.
Measured after: **23 → 25 polls while backgrounded**, and an over
carried on from 53% to 63% and then finished, stopped its own service
and cleared its own notification with the app still in the background.

**Hijacking Back unconditionally is what this deliberately does not
do.** With nothing to protect it exits, so the gesture keeps its
meaning.

And case 3 is why `main()` ends with **`std::_Exit`** rather than
returning: the tombstone above is Android's own HWUI render threads
outliving the graphics state during teardown, not our thread and not our
mutex, and the reliable answer is not to unwind at all. Nothing is lost
— every `QSettings` write goes through a temporary that syncs when it
dies, and the audio device, PTT and engine threads are the service's to
release on its stop path while the world is still standing. Third
instance of one rule here, after `Session` being deliberately immortal
and `check::Watchdog` calling `std::_Exit`: **at teardown, not running
code is the reliable option.** Verified: zero crash records where there
was a tombstone every time.

## The first-transmit prompt, and the CW ID that cannot work

**Send opens a one-time prompt before the first over** (2026-08-10):
callsign, an offer of CW ID, and an acknowledgement that the operator
holds whatever licence their transmission requires and is responsible
for using the app legally. Only the acknowledgement is required —
declining the callsign is a supported answer, and "Not now" closes
without recording anything, so the prompt returns on the next Send.

**It is not a gate and must not become one.** The app cannot tell
whether it is connected to a radio at all; the service may not be
amateur; the operator may be identifying by voice, on a band with
different rules, or handling the whole question themselves. Refusing to
transmit would be the app claiming an authority it does not have —
the same argument that keeps a callsign optional. What it is is a
roadblock to casual misuse: someone who has not thought about any of it
has now been asked to, once, immediately before the first transmission.
Hence its position *behind* Send rather than at first launch: a station
that only ever listens should never see it.

**The CW ID check is a different kind of thing and does block.** A
message still containing `{callsign}` with no callsign set would key a
partial identification — "SSTVAE DE " and then nothing — so Send is
disabled with the reason on screen, on the transmit pane and again in
Settings where the fix is. It is a setting that cannot do what it says,
and there are three ways out, all of which the message names: set a
callsign, write the identification into the message itself, or turn CW
ID off.

`tx::cw_id_problem` is that predicate, shared. The UI blocks on it and
the engine skips the ID on it, because two implementations of "is this
CW ID sane" would eventually disagree and the direction that
disagreement takes is a station transmitting an ID it was told it had
turned off. The desktop asks the same question in `TransmitPanel::send`.

**Making it shared changed the engine's behaviour, deliberately.** The
guard used to be `!config.callsign.empty()`, so *any* empty callsign
dropped the ID — which meant "write the identification into the message
itself" was an escape the UI could offer and the engine would ignore. A
literal message now goes out with no callsign set;
`test_cw_literal_message_is_sent_with_no_callsign` is that case, and
`test_cw_id_problem_names_only_the_broken_combination` pins all three
escapes.

Two things the on-device pass caught that reading would not have. A
`CheckBox` with a wrapping `contentItem` puts **the indicator in the
middle of the text** — the control centres it against the whole content
height, so a three-line label leaves the box floating over line two; it
is a `CheckBox` beside a `Label` in a `RowLayout` now, which also makes
the sentence a tap target instead of a 24 px box. And the dialog is
**bounded and scrolled** rather than merely tall: a Popup that outgrows
its parent does not compress, it puts the buttons off the bottom where
there is nothing to reach and nothing to say so — the same failure the
desktop's settings tabs have a `QScrollArea` for.

## The models ship inside the APK

**Bundled since 2026-08-10, and the reason is not download size.** The
model *is* part of the on-air contract: the latent space is learned, so
an encoder from one checkpoint only means anything to a decoder from
the same one, and two stations must be running the same artifacts to
talk at all. "Update the model without an app update" therefore reads
as flexibility and behaves as a way to silently desynchronise a station
from the band — a codec revision is a coordinated everyone-at-once
event, which is what an app release already is. So the argument that
usually favours fetching does not apply here, and what is left is the
argument for bundling: **first run has to work with no network.** This
is a radio app, and the moment of need is disproportionately a field
site with no coverage; an operator who installs at home and first opens
the app on a hilltop otherwise has a listener that cannot decode
anything, and the failure is total rather than degraded.

Measured, on the v3 fp16 artifacts: decoder 8.5 MB, encoder 11.3 MB,
**18 MB** added to the APK (55 MB against 37 MB with
`-DSSTVAE_ANDROID_BUNDLE_MODELS=OFF`). Model weights barely compress —
AAPT deflates these to 92% — so that is close to the raw size and there
is no packing trick to be had.

Four decisions inside that:

- **`assets/`, not a Qt resource.** A `.qrc` is compiled into the app
  library, and the bundle carries one library *per ABI*, so 20 MB of
  weights would land in the download twice. Assets live in the bundle's
  base module and are shared by every ABI.
- **Downloaded at configure time and pinned by sha256**, the same shape
  as the onnxruntime and Hamlib pins. These are published immutable
  filenames, so a hash mismatch means the wrong file, never a stale one.
  `SSTVAE_ANDROID_MODEL_DIR` is where they land — pinned by
  `tools/build_android.sh` to the *top* build directory, because a
  multi-ABI build configures this file once per ABI in nested build
  trees and would otherwise fetch 20 MB twice.
- **The bytes are read out and released once the ORT session exists**,
  rather than held for the process. `AAsset_getBuffer` on a compressed
  asset inflates into a block owned by the open asset, which would mean
  ~20 MB resident forever; reading into a `codec::ModelBlob` puts that
  memory somewhere with a lifetime. What makes it safe is stated in
  code rather than assumed: `codec.cpp` sets
  `session.use_ort_model_bytes_directly` to `0` explicitly, because that
  opt-in is the one thing that would make ORT reference the caller's
  buffer, and a change of default would otherwise land as a
  use-after-free with no diagnostic.
- **The fetcher stays, and OFF is a supported configuration.** With no
  assets staged, `assets::model_blob` returns nullopt, the codec falls
  through to `checkpoint::resolve_onnx`, and the app fetches exactly as
  it did before — one code path, two builds. That is what makes the
  switch cheap to flip while developing, where 20 MB per clean build
  tree is not free.

**`assets::init()` runs on the UI thread**, from `Listener`'s
constructor, for the same reason `audio::android::set_java_vm` does: it
needs the Java context. After it, nothing else does — the `AAsset_*`
family is a plain NDK C API with no JNIEnv in it, so the model thread
reads assets with no JNI at all and the `FindClass` hazard cannot
apply.

Verified on the emulator the only way that means anything: **uninstall,
fresh install, airplane mode on, no cache** — model reports ready, and a
mode A transmission decodes 220/220 with the network off for the whole
run.

## The Play upload

```sh
tools/build_android.sh --aab --version-code N   # -> build-android/sstvae-upload.aab
```

A bundle rather than an APK because Play takes nothing else, and it is
a **different job from the sideload build in one respect that decides
the rest**: a sideload APK is signed with the debug key, which is
world-known and per-machine, while a bundle is signed with the *upload
key*, which is an identity. So `--aab` shares no fallback with the APK
path — it refuses to emit anything rather than quietly producing an
unsigned or debug-signed bundle, since Play rejects both and the
rejection arrives minutes later in a browser, a long way from the
build. For the same reason it refuses `--abi` (a single-ABI bundle is a
store listing half the world cannot install), `--debug`, and
`--install` (nothing installs a bundle).

**The upload key lives outside the repository and must be backed up.**
`~/.android-keys/sstvae-upload.jks`, PKCS12, RSA-2048, valid to 2053 —
comfortably past the 2033 floor Play checks — with its password in
`sstvae-upload.pass` beside it at mode 600, which is what the script
reads (`SSTVAE_UPLOAD_KEYSTORE`, `SSTVAE_UPLOAD_KEYSTORE_PASS`,
`SSTVAE_UPLOAD_ALIAS` override all three). A password *file* rather
than an argument or an exported variable, because both of those are
readable in `ps` by every process on the machine. Keeping the password
next to the keystore does mean one compromise gets both; moving it into
a password manager and deleting the file is strictly better, and the
script does not care where it points.

With **Play App Signing** — which is not optional for new apps — Google
holds the actual app signing key and this is only the key you *upload*
with, so losing it is a support request rather than the end of the app.
That is a much softer failure than the pre-2021 arrangement, and it is
still worth backing up: a reset is days.

**`--version-code` is mandatory reading even though it defaults.** Play
requires the code to increase on every upload, forever, and it is the
one number a build cannot infer — `PROJECT_VERSION` moves for its own
reasons, and re-shipping a fixed build of the same release still needs a
new code. Hence an explicit input rather than something derived. Getting
it wrong is tedious, not dangerous: Play refuses the bundle and names
the number it already has.

Four things that were checked on the first bundle and are worth
re-checking only when something below them changes:

- **16 KB page alignment**, which Play requires of anything targeting
  SDK 35+. All 78 native libraries report `0x4000` LOAD alignment,
  onnxruntime's prebuilt `.so` included — NDK 28 does this by default,
  but the prebuilt is the one we do not compile, so it is the one worth
  looking at. `llvm-readelf -l` over `base/lib/*/` is the check.
  It cannot be fixed downstream: the bundle is not zipaligned, because
  alignment is a property of the APKs Play *generates* from it.
- **`jarsigner`, not `apksigner`** — an AAB is a jar and apksigner does
  not handle one. `keytool -printcert -jarfile` on the result shows the
  signing cert, and its SHA256 should equal the keystore's.
- The self-signed / no-timestamp warnings from `jarsigner -verify` are
  **expected and benign**: self-signed is what an upload key *is*, and
  Play does not want a timestamp.
- **armeabi-v7a is not in the bundle**, because only the arm64 and
  x86_64 Qt kits are installed. 32-bit-only devices — essentially
  nothing since ~2019, and nothing anyone drives a radio from — simply
  will not be offered the app. Adding it means a third Qt kit and a
  third slice of build time, and it is a deliberate omission rather than
  an oversight.

**Existing sideload testers must uninstall.** The debug-signed APKs and
this bundle have different signing keys, so Play's copy is not an
upgrade over a sideloaded one; it is a different app as far as Android
is concerned, and the install fails until the old one goes. That costs
testers their settings and saved receptions, which is a thing to say in
advance rather than after.

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

**The GUI toggle has a console equivalent, and it is the one to use.**
`adb emu avd hostmicon` (`hostmicoff` to undo) flips exactly the same
switch, so a scripted, windowed-but-unattended run needs no clicking —
which matters because the extended-controls state is **not persisted per
AVD**, so every launch starts muted. `-allow-host-audio` alone is not
enough: with it the guest gets a stream that is 99.9% zeros with
occasional fragments, which reads on the waterfall as broadband hash and
on the level meter as `peak -13 dBFS 99.9% near-zero` — a pure 1 kHz
tone arrives looking like noise. `hostmicon` turns that into a clean
line at the right frequency.

**Force the qemu capture rate to 48 kHz, or lose ~37 dB.** qemu opens
its host capture stream at **44100 Hz** by default while the guest's
audio HAL runs at 48000, and whatever resamples between them is bad
enough to be the dominant impairment: the same file that decodes at
**35.7 dB** through the host loopback measured **-1 dB** inside the
emulator, all frames received, sync fine, picture mush. The spectrum
looks clean while it happens, because the damage is in-band. Old-style
qemu environment variables fix it, and the emulator still honours them:

```sh
export QEMU_AUDIO_ADC_FIXED_SETTINGS=1 QEMU_AUDIO_ADC_FIXED_FREQ=48000 \
       QEMU_AUDIO_ADC_FIXED_FMT=S16 QEMU_AUDIO_ADC_FIXED_CHANNELS=1
```

**`tools/run_android_emulator.sh` is that whole recipe in one command**,
and is how to start an AVD for anything involving audio:

```sh
tools/run_android_emulator.sh sstvae_phone -no-snapshot
pw-play --target=sstvae-null transmission-48k.wav   # pre-resample!
```

It builds the null-sink + remapped-source loopback itself, points
`PULSE_SOURCE` at it so qemu's capture stream opens there rather than on
the host's real microphone, and unloads both modules again from an EXIT
trap — which is why it does not `exec` the emulator. Export
`PULSE_SOURCE` yourself to use an existing source instead, and it leaves
the audio graph untouched. It sets the four variables, finds the
session's `DISPLAY`/`XAUTHORITY`, passes everything after the AVD name
straight to `emulator`, and runs `adb emu avd hostmicon` once the guest
is up. It also pins
`ANDROID_HOME`/`ANDROID_SDK_ROOT` to `~/Android/Sdk` rather than
inheriting them, because a profile pointing at a distro SDK makes the
emulator hunt for system images under a path nothing asked for and die
with "Broken AVD system path" (`SSTVAE_ANDROID_SDK` overrides).

`pactl list source-outputs short` is the check — the qemu row must read
`s16le 1ch 48000Hz`. With it, the emulator reproduces the file's own
35.7 dB. Set `PULSE_SOURCE=sstvae_loop` in the same environment and the
capture stream opens on the loopback directly, which retires the
move-the-source-output step above.

**Audio only initialises with a window**, so `-no-window` is not an
option for this: headless the log says `pulseaudio: Failed to
initialize PA context` and the guest hears nothing. `-gpu host` also
needs a reachable X display, which on a Wayland desktop means both
`DISPLAY` and `XAUTHORITY` (`/run/user/<uid>/xauth_*`) — without the
latter it fails with `Invalid MIT-MAGIC-COOKIE-1 key` and falls back to
refusing to start.

**And expect run-to-run variance even with all of that right.** Repeats
of one transmission on one AVD measured anywhere from -1 dB to 35.7 dB
with nothing changed, and level makes almost no difference across a
13 dB range — it is the emulator's capture path glitching, not
something to tune. A **fresh boot is worth more than any setting**: the
best runs on each AVD came within the first few receptions after
launch. Take the picture you want and repeat until it is clean rather
than hunting a parameter.

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

## Wide search and drift tracking (2026-08-11)

`RxConfig::blind_wide` and `RxConfig::drift_track` (see docs/todo.md,
"Wider acquisition search" and "Frequency drift during a transmission")
reach the Android app through `Settings > Receive`: a "Wide frequency
search" switch and an off/slow/fast "Track drift" combo, both persisted
in `QSettings` (`receive/blindWide`, `receive/driftTrack`) the same way
`showTechnical` and `saveToGallery` are.

Two things worth not re-deriving. **Both are read once, into the
`RxConfig` `Session::start()` builds**, not consulted continuously the
way `show_technical`/`save_to_gallery` are — there is no running
poller for them to reach, so a change takes effect on the *next* Start,
same as the input device picker, and both controls are disabled while
listening for the same reason. And **`drift_track` is validated through
`modem::drift_track_from_name` rather than trusted as the raw QSettings
string**, on both the read at startup and the write from QML: a stored
value from a different app version is a real possibility here in a way
it mostly isn't on the desktop (Settings dialogs there round-trip a
whole `Config` object with its own version and unknown-key handling),
and an invalid string falling through to the C++ side would either
throw where nothing expects it or silently mean something other than
what the switch shows.

The phone's own use cases are why `drift_track` matters more here than
on the desktop app: VHF FM satellites and EME are exactly where "fast"
earns its cost, while HF with a fixed station is the case the default
(off) is tuned for.

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

**Saving to the shared gallery is in** (2026-08-10, `Gallery.java`),
behind a `Save to gallery` switch that is **off by default**. Share came
first and still earns its place — it reaches gallery, mail and chat for
one intent — but it is a per-picture action, and only a MediaStore row
makes receptions a *collection*: Google Photos builds "On this device"
from `MediaStore.Images` grouped by `BUCKET_DISPLAY_NAME`, so the folder
name is the collection title and app-private storage can never appear
there however it is arranged.

Five things worth not re-deriving.

**The export runs in `ListenerService`, not beside the file write.** The
notification poller already holds a `Context`, which is what a MediaStore
insert needs; doing it from `Session::save_reception` would mean calling
an application class from a thread the engine created — the `FindClass`
hazard that cost the transmit path a run — for no gain. It is one
`Thread` per reception (they arrive 32–95 s apart at best) because the
poller is on the main Looper and the copy is about a megabyte.

**The private copy stays canonical.** The sidecar is what makes a
picture answerable a week later and MediaStore has no column Photos will
show, so the gallery copy is deliberately provenance-free and a failed
export costs an export, not a reception.

**`minSdk` went 28 → 29.** `RELATIVE_PATH` + `IS_PENDING` is API 29 and
needs no permission at all; API 28 wants `WRITE_EXTERNAL_STORAGE`, a
public-directory write and a `MediaScannerConnection` scan — a second
implementation and a runtime prompt for one API level. (Qt still injects
`WRITE_EXTERNAL_STORAGE` into the manifest on its own account; it is
inert at 29+, but it reads badly next to a gallery feature and is worth
capping some time.)

**`IS_PENDING` is not decoration** — without it the scanner can index a
half-written file and Photos shows a truncated picture that never
repairs itself. The logcat trace of a working export is the `.pending-`
name being moved to the final one when it clears.

**`DATE_TAKEN` on the insert does not survive, measured.** Once
`IS_PENDING` clears the scanner re-derives the metadata columns from the
file, and a PNG carries no EXIF date, so the column reads NULL whatever
was written. `date_added`/`date_modified` — the moment the reception
finished — is what Photos ends up sorting on, which is the right answer
anyway. Forcing a timeline position would mean writing metadata *into
the file*.

Verified end to end on the emulator through the host-audio loopback
below: a mode A transmission decoded 220/220, and the row came back
`bucket_display_name=SSTVAE`, `relative_path=Pictures/SSTVAE/`,
`is_pending=0`. With the switch off, the same transmission decodes and
produces **no row and no file** — which is the half worth testing,
since off is the default and the reason it is the default is that a
listener left running overnight otherwise puts whatever arrives on the
band into the operator's camera roll and their photo backup.

## Tier 1: transmitting

**Built 2026-08-09 and on the air the same day.** Pick a picture from
the gallery or the camera, frame it by touch, send it; the session
suspends receiving for the over and resumes by itself afterwards.
`docs/android.md`'s Tier 1 section is the reasoning; this is the shape
of the code.

Verified over RF, not loopback (Andrew, 2026-08-09): transmitted from a
phone into a radio through a **USB audio interface**, received on an
Android tablet on another radio, **mode B, CW ID on, 25 dB SNR, no
issues** — the first Android-to-Android contact. So a phone's USB audio
output drives a radio, and USB is now exercised in both directions on
this app rather than only on the pre-Tier-0 smoke test.

**The VOX leader was not part of that test**, so it is still tested
only against the preamble detector and the emulator. The radio was
keyed by hand because that radio offers no VOX on its USB/data input —
an uncommon limitation, not a reason to doubt VOX keying generally:
many radios do offer it there, and a USB soundcard wired to a radio's
*microphone* input keys on VOX whatever its data port does.

The pieces mirror the receive side deliberately, so there is one
arrangement to understand rather than two. `Session` owns the
`TxEngine` exactly as it owns the decode loop, for the same reason — an
over is 32–95 s of committed airtime and must survive the activity
being destroyed. `Transmitter` is a *view* over it, as `Listener` is
over the receive half. `Composition` holds the picked picture and its
framing, process-wide, so a rotation mid-crop loses nothing.

- **The picture goes out unmodified. No overlay, not even a callsign
  caption.** The station is identified by the beacon carrier and
  optionally a CW ID; burning a callsign into the pixels identifies it
  only to someone who already decoded the picture. So
  `core/overlay/` is not in the Android build at all, and
  `docs/android.md`'s Tier 1 prediction that it would be is the one
  part of that design that was wrong.
- **A callsign is not required to send.** Identifying is required of an
  amateur station, but the app does not know it is attached to a radio
  and does not take responsibility for the operator's identification
  even when it is — voice is a third way it never sees.
- **The crop preview is `images::fit`'s own output**, served through
  the image provider as `compose/<id>`, not a QML `Image` with a clip
  imitating a crop. The desktop's rule and the desktop's reason: a
  second representation is a second thing that can disagree with what
  goes on the air, and this screen exists to decide exactly that.
- **Two `Image`s, swapped on load, and one render in flight.** A single
  `Image` following the id blanks for the duration of every load, so a
  drag becomes a flicker with the picture appearing only when the
  finger stops. Revealing a frame only when it is `Ready` keeps the
  previous one up meanwhile. The throttle is the other half: `fit` is a
  real resize, and one request per touch event queues work faster than
  it drains, so moves during a load collapse into a single repeat.
- **Zoom goes below 1 and letterboxes**, via `images::min_zoom` — the
  desktop's zoom-out work, which the Android slider and pinch clamp
  against so both stop where `fit` does. Below zoom 1 the crop window
  is wider than the source on an axis, so its half-extent exceeds 0.5
  and the centre has to be *pinned* rather than clamped: `std::clamp`
  with an inverted range is undefined behaviour, not a no-op.
- **`Session::stage_transmit` then `start_staged_transmit`**, two calls,
  because the service is the only thing that may start an over and a
  640x480 picture is not something to marshal through an Intent. The
  staged request is *consumed*, so a redelivered intent or a double tap
  cannot put a second copy of the picture on the air. Staging also
  fixes the composition at the moment of the tap: an edit afterwards
  belongs to the next over.

### What Tier 1 cost that Tier 0 did not

- **`FindClass` cannot see an application class from a thread we
  created.** The transmit thread is the first in this app to call into
  Java, and it failed on its first run with `android audio:
  org/cleverdomain/sstvae/AudioBridge not found`. JNI resolves against
  the class loader of a frame on the call stack; a thread attached with
  `AttachCurrentThread` has none, so it falls back to the system loader.
  `set_java_vm` now caches a global reference from the UI thread. This
  was invisible through all of Tier 0 because every control call came
  from Qt's thread and the data path is Java calling us — so it is a
  hazard for *any* new C++→Java call made off the UI thread, not a
  one-off.
- **`AudioBridge.java` was a hand-synced duplicate**, one copy under
  `core/audio/android/java/` and an identical one under
  `android/src/`. androiddeployqt takes exactly one
  `QT_ANDROID_PACKAGE_SOURCE_DIR`, which is why. It is now assembled in
  the build tree from both sources (`sstvae_stage_package_dir`), so the
  layer owns its blocking-read loop the way the CMake comment always
  claimed. Nothing was wrong yet, which is exactly what made it worth
  fixing: the first edit to either copy would have shipped the other.
- **The service gained a transmit action and `mediaPlayback`.** Both
  foreground types are declared for its whole lifetime rather than
  re-declared per transition — except `microphone`, which may not be
  claimed without RECORD_AUDIO and throws from API 34, so a station
  that denied the microphone and only transmits drops it.
- **Settings needed a `ScrollView`.** The transmit settings pushed the
  page past the screen, and a QML column does not compress when it does
  not fit, it truncates — "Model" simply disappeared behind the tab bar.
  Same failure and same fix as the desktop's per-tab `QScrollArea`. The
  trap on this side: the content must be bound to the ScrollView **by
  id**, because a ScrollView reparents its content into a Flickable's
  `contentItem`, so `parent` is neither the ScrollView nor anything with
  its width. The symptom is not a missing scrollbar but text running off
  the right edge with nothing to scroll it back.

### Picking a picture

`ImagePicker.java` is a **transparent activity of our own**, not a call
into Qt's. An activity result comes back to whoever launched it, and
reaching `QtActivity`'s means either subclassing Qt's bindings — and
maintaining that subclass across Qt upgrades — or using its private
`QtAndroidPrivate::startActivity`. An activity that launches an intent
and finishes is smaller than either.

Two things it does that are not obvious. The result is **copied into
app-private storage before any path reaches C++**: what the picker
returns is a `content://` URI whose grant lasts as long as that
activity, so handing it over would produce a path that reads fine while
composing and fails at the moment of transmitting. And the camera's
output file goes in `getExternalCacheDir()`, because Qt's FileProvider
covers that and not the internal cache — declaring a second provider
would collide with Qt's, the same trap `Sharing.java` records.

### Not done

`Composition` does not survive the process being killed, so swiping the
app away loses the picked picture and its framing. A rotation does not,
which is what the process-wide singleton is for. Persisting the source
path and framing in `QSettings` would fix it and is a few lines; it has
not been done because nothing has asked for it yet.

## Editable fields commit per keystroke

Every view object here (`Listener`, `Transmitter`, `RigControl`) has a
single `changed()` signal that a timer emits a few times a second, so
live state redraws without anything having to push it. That is fine
until an editable control is bound to one of those properties, and then
it is a trap with a delay on it:

**A QML binding is not broken by a C++ write.** Only a JavaScript
assignment removes one. So `text: view.someProperty` stays live for the
life of the field, and every tick re-evaluates it. Commit the value on
`editingFinished` and the property sits at its old value while the
operator types — until a tick puts that old value back into the field,
about a second in. It looks exactly like the box deleting itself, which
is how it was reported.

So: **`onTextEdited`, `onValueModified`, `onMoved`, `onToggled`** — the
per-change handlers, never the commit-at-the-end ones. Then the property
tracks the control and the re-evaluation is a no-op. The Send screen's
callsign field already carried this rule as a comment, for the *other*
reason it matters: the back gesture dismisses the keyboard without ever
firing `editingFinished`, so a field committed at the end silently loses
what was typed.

Splitting `changed()` into settings and live halves would also fix it,
and was not done: all three view objects share this design, and one of
them diverging is worse than a convention all of them follow.

The same applies to anything hung off `changed()` that writes to a
control — a `Connections { function onChanged() }` that re-syncs a
`ComboBox` runs at the poll rate, including while the operator has its
popup open. Watch the specific thing that can invalidate it
(`onModelChanged`) instead.

## Rig control

CAT and PTT, added 2026-08-22. `docs/android.md` has the design record
and the reasoning; this is the operational half.

**How it reaches the radio.** Hamlib's `rig_open()` turns any backend
into a network client when its pathname parses as `host:port`, so a
`SerialTransport` (USB or Bluetooth, from Java) is presented to it as a
loopback socket. Nothing is patched and no CAT protocol is
reimplemented, which is why the radio picker here lists the same several
hundred rigs the desktop does.

**Three connection kinds, and the kind is what decides the plumbing —
never the device string.** USB and Bluetooth open a transport and get a
bridge; Network hands the host straight to Hamlib with no bridge at all,
covering both a station PC running `rigctld` (model 2) and a
serial-over-TCP server with a native backend. An earlier draft branched
on the *shape* of the device string and got it exactly backwards: a USB
identifier like `usb:1a86:7523` has no slash and does not start with
`com`, so Hamlib's own `parse_hoststr` rules read it as a hostname, and
the bridge was skipped for precisely the devices it exists to serve.

**Building it.** `-DSSTVAE_ANDROID_RIG=ON` is the default and pulls in
an NDK cross-build of Hamlib's autotools tarball, per ABI.
`-DSSTVAE_ANDROID_RIG=OFF` drops CAT and keeps the app — the rig screen,
the transport and the settings all still build, so it is one flag rather
than a second code path. **That cross-build has never been run**: it was
written from the NDK's documented layout in a session with no NDK and no
reachable `dl.google.com`. Expect to fix something. Two guards make the
first attempt diagnosable rather than mysterious:

- The configure step fails naming the exact compiler wrapper it looked
  for, because the NDK ships one per API level and an unsupported level
  is *missing* rather than wrong. **This is the one that fired on the
  first real run**, and it fired for the wrong reason: the API level was
  being read from `CMAKE_SYSTEM_VERSION`, which CMake defaults to `1`
  when cross-compiling and nothing sets it, so the wrapper it looked for
  was `x86_64-linux-android1-clang`. The guard written to catch exactly
  that was `if(NOT _hl_api)` — and `1` is *true* in CMake, so it never
  ran. A guard whose condition cannot fire is not a guard. The level now
  comes from whichever of `SSTVAE_ANDROID_API`, `CMAKE_ANDROID_API`,
  `ANDROID_NATIVE_API_LEVEL`, `ANDROID_PLATFORM_LEVEL`, `ANDROID_PLATFORM`
  or `CMAKE_SYSTEM_VERSION` first gives a plausible answer, falls back to
  21, says which source it used, and repairs upward to the lowest wrapper
  the NDK actually ships if the chosen level has none.

  `-DSSTVAE_ANDROID_API=<level>` overrides all of it. Building Hamlib at
  a level *below* the app's minSdk is fine and is why 21 is the fallback:
  a library built against an older libc loads on a newer one, never the
  reverse, so guessing low fails on nobody's phone and guessing high
  fails on somebody else's.
- The install step refuses a versioned SONAME. libtool would produce
  `libhamlib.so.4.0.7`; Android's packager takes only files named
  exactly `lib*.so`, so that library is dropped from the APK without
  comment and the app dies at `dlopen` **before `main`** — no output on
  any stream, indistinguishable from a deadlock.

  **`-avoid-version` is a libtool flag, not a linker one**, and putting
  it in configure's `LDFLAGS` is the second thing that went wrong on a
  real NDK build. configure's very first link test runs the compiler
  directly, clang rejects an argument it has never heard of, and the
  build stops at `C compiler cannot create executables` — with the
  actual reason two layers down in `config.log`, which is a long way
  from a flag we chose. It is applied at *make* time now, overriding
  `libhamlib_la_LDFLAGS` (`src/Makefile.am:24`, the one variable
  upstream puts `-version-info` in) and carrying `-no-undefined` over
  from the same line.

  Not plain `LDFLAGS=-avoid-version` at make time, which would also have
  worked as far as libtool is concerned — it copes with `-version-info`
  and `-avoid-version` together and strips the version
  (`build-aux/ltmain.sh:9488`). The reasons are elsewhere: a
  command-line `LDFLAGS` *replaces* the tree's, discarding whatever
  configure computed, and it reaches every link in the tree including
  the fifteen tool executables, where the flag means nothing.

  Verified natively on a desktop, which is possible because the
  mechanism is libtool's and not the NDK's: the same configure and make
  arguments produce a single unversioned `libhamlib.so` whose
  `readelf -d` SONAME is `libhamlib.so`, with the tools still linking.

- **`CC` alone is not enough, and the failure waits until late.**
  Setting only `CC` leaves configure to find C++ on its own, and it
  finds the *host* `g++`. Most of Hamlib is C, so the build gets most of
  the way through before reaching `rotators/androidsensor` — the one C++
  directory — and dying on `-stdlib=libc++`, which configure adds for
  every Android host and the host g++ has never heard of. A host
  compiler quietly standing in for a cross one is the shape of this
  bug; `CXX` is set now even though, after the next point, there should
  be no C++ left to compile.

- **The Android sensor rotator cannot be switched off. Do not try
  again.** It points an antenna using the phone's accelerometer, which
  this app has no use for, and it is the only C++ in the tree we build,
  so it looks like free savings. There is no `--without-androidsensor`;
  upstream gates it on whether `android/sensor.h` exists
  (`configure.ac:171`), and pre-seeding autoconf's cache with
  `ac_cv_header_android_sensor_h=no` does correctly drop the directory
  from `ROT_BACKEND_LIST`.

  It then fails to build, because `src/rot_reg.c` guards the backend's
  two halves on **different conditions**:

  ```
  line  87: #if HAVE_ANDROID_SENSOR                      (the declaration)
  line 141: #if defined(ANDROID) || defined(__ANDROID__) (the table entry)
  ```

  `__ANDROID__` is defined by the compiler and cannot be unset, so the
  table entry is unconditional on Android while the declaration is not.
  They agree only when `HAVE_ANDROID_SENSOR` is true — which upstream is
  entitled to assume, since a real NDK always has `android/sensor.h`.
  Answering "no" leaves the file calling a function nobody declared.

  So it builds, and setting `CXX` is what makes that work rather than a
  precaution. (The cache mechanism itself is sound and is used for the
  two malloc answers — verified against this configure with a proxy
  header: `checking for linux/ppdev.h... (cached) no`, and
  `/* #undef HAVE_LINUX_PPDEV_H */` in the generated `config.h`. The
  problem is specific to this backend's guards.)

  `-lc++` goes into `LDFLAGS` for every Android build regardless
  (`configure.ac:178`). The NDK sysroot ships `libc++.so` as an implicit
  linker script so it resolves, and `libhamlib.so` carries a
  `DT_NEEDED` on `libc++_shared.so` — which Qt for Android packages
  anyway, because Qt itself needs it.

- **A failed Hamlib build used to cement itself.** The "already built"
  stamp was the installed `hamlib/rig.h`, and `SUBDIRS` puts `include`
  second (`Makefile.am:25`) — so `make install` copies the headers long
  before it reaches `src`, and any failure after that left a tree the
  check accepted. The next configure skipped the rebuild and failed with
  "Hamlib library not found", naming a path instead of the compile error
  that actually stopped it. The stamp requires the library now. If you
  are recovering from a build that failed before this landed, delete
  `<build>/_deps/hamlib-install-*` once.

**Testing it without a radio.** Hamlib model 1 is the dummy: it opens,
keys and reports a frequency with nothing attached, so the whole path
above the transport can be exercised on a phone with no cable. The
transport itself needs hardware; the closest thing to a substitute is
the desktop suite, where `test_rig_bridge`, `test_rig_bridged` and
`test_rig_hamlib` cover the bridge, the composition, the PTT routing and
a real Kenwood backend talking through it to a fake radio.

**USB permission is granted per attach, and Android forgets it on
detach.** `UsbAttach` plus `res/xml/device_filter.xml` is what makes the
grant stick: an app that can handle `USB_DEVICE_ATTACHED` is authorised
automatically when the user answers "always open with this app". It is
its own no-display activity rather than a filter on `QtActivity`,
because the filter matches four whole vendor ranges and every CDC-ACM
device — plugging in an Arduino must not open a radio app.

**Bluetooth lists bonded devices only.** Discovery would need
`BLUETOOTH_SCAN` and, before API 31, location permission, which is a
large ask for a picker whose entire content is the radio the operator
already paired in system Settings. Pairing is the system's job.
RFCOMM has no modem control lines, so DTR and RTS keying are not offered
there at all.

**The keying method changes the waveform.** With the rig keyed directly
the VOX leader is skipped — a swept tone into an already-keyed radio
only delays the picture — and the airtime estimate on the Send screen
follows. The Send screen says which one is in force, but only when it is
rig keying: saying "VOX" every time when VOX is the only option trains
the eye to skip the line.

**The platform layer needs an explicit init, and forgetting it looked
like two unrelated bugs.** `rig::android::set_java_vm` and
`SerialBridge.init(Context)` were never called — `init_rig_bridge` in
`rigcontrol.cpp` does it now, from `RigControl`'s constructor, which is
on the UI thread for the `FindClass` reason `core/audio/android/`
records. `listener.cpp`'s `init_audio_bridge` is the same four lines for
the audio layer.

What made it cost a round was not the missing call. The layer threw a
perfectly good message — "set_java_vm() was never called" — and two
things ate it:

- `refreshDevices()` caught every exception and cleared the list, so the
  screen said "Nothing plugged in. Connect the radio…", which was a
  confident lie with a radio attached. **An enumeration that threw is
  not one that came back empty**, and it shows the message now.
  "Nothing connected" and "Bluetooth not granted" are still ordinary
  empty lists with no error.
- `bluetoothReady()` did *not* catch, and it is read from a QML property
  binding — where an exception terminates the process. It crashed on
  switching to Bluetooth, then at every launch, because the connection
  kind is persisted and the binding is evaluated on load. **Nothing
  reaching JNI may throw from a getter.**

The rule that came out of it, now written into `androidrig.cpp`: **a
query answers, an action reports.** `has_permission` returns false when
it cannot ask — the truthful answer to "may the app open this right
now" — while the enumerators throw, because their caller catches and has
somewhere to show it.

**Unplugging the USB cable, and getting back.** Verified on hardware
2026-08-23: CAT and both control-line keying methods work; pulling the
cable used to leave the screen saying "Connected" with no way back.

Three things were wrong and they are worth separating. `running()` means
*a session is configured*, not that the radio is answering — the
desktop's distinction, correct there, but on a phone it was the only
thing the UI showed. `connectionState` is the property to display now,
and the signal it rests on is that **a published frequency is the only
proof the radio answered**: `failed` alone cannot tell "still opening
the port" from "the cable is out".

Reconnection is app-level (`RigControl::maybe_reconnect`, once per 1 Hz
tick) rather than in `RigController`, because the question it has to
answer — *is the device back?* — is a platform one. For USB and
Bluetooth that is `has_permission(device)`, which is false for a device
that is not there, so one cheap call covers both "is it back" and "may
we open it". **A device that is simply absent does not spend the
backoff**, so replugging reconnects on the next tick rather than
somewhere in the next 30 seconds; the 2/4/8/16/32 s backoff is only for
attempts that reached the radio and failed. Never while transmitting: an
over is committed airtime and swapping the link underneath it buys
nothing.

**The rig controller is immortal, and that fixed a bug rather than
tidying one.** `ptt_function()` captures the *controller*, and
`TxEngine` holds that for a whole over — so `stop_rig()` destroying the
controller left the engine calling into freed memory to bring PTT back
down, which is what turning rig control off mid-over used to do. It is
created once and `stop()`ped, never destroyed. The same property makes
reconnection safe: `RigController::start()` supersedes a session in
place, so a `Ptt` handed out before a reconnect still keys the new
backend. `test_rig.cpp` pins it, mutation-tested — binding the lambda to
the session instead of the controller sends the key to the *superseded*
backend and none to the new one, a failure with no symptom until
somebody transmits.

**The VOX leader is sent whatever keys the radio** (Andrew, on
hardware). An earlier version zeroed it whenever PTT was not VOX, on the
theory that a swept tone into an already-keyed radio only delays the
picture. That is the app second-guessing the operator: the leader is
also a settling period for an interface that wants audio flowing before
the radio is properly in transmit, and `ptt_lead_s` is a different
quantity doing a different job. Set it to zero if it is not wanted.

**The bridge is not built on Windows, and the tests follow it.** Windows
has COM ports and never constructs one, so a Windows build was compiling
a Winsock translation of a file only POSIX runs — a second
implementation exercised by nobody, whose green test would have said
nothing about the one a phone runs. It was also broken: `stop()` relies
on `shutdown()` waking a thread blocked in `recv()`, which POSIX
guarantees and Winsock does not, so `rig_bridge` hung for the full 120 s
of its watchdog in CI. Linux and macOS still build and test it, using
the same POSIX calls Android does, so `test_rig_hamlib`'s
real-Kenwood-through-a-real-bridge proof is untouched.

**The composite-device question is settled, and it was the one that
could have sunk the approach.** Most rig USB interfaces present audio
and CDC serial together and this app needs both at once;
mik3y/usb-serial-for-android#477 is an open report of a composite device
where claiming the serial interface fails while the platform holds the
audio ones. On an Elecraft K4 over one composite interface it works:
CAT, DTR keying and RTS keying, with the audio path live (Andrew,
2026-08-23). **Bluetooth is still untested for want of a device** and
goes to beta on the strength of the shared code beneath it.

**Not every radio works, and the trace is how to find out why.** A K4
works over USB; an IC-9700 and an IC-7100 do not (reported 2026-08-23,
unresolved). Everything structural was ruled out by reading before the
logging below was written, and it is worth not re-deriving: the byte
path is length-counted end to end (`SerialBridge.read` → a `jbyteArray`
region copy → `LoopbackBridge`'s `send`), so nothing truncates a binary
CI-V frame at a `0x00` the way a string would; Hamlib's POSIX
`port_read_generic` and `port_write` have no port-type branch at all
(the ones that exist are Win32 serial); `network_flush` is a
`FIONREAD`-guarded drain and cannot block; the port-type conditionals in
`rigs/icom/` are none; and the baud our `serial_defaults` picks is
`rig_caps.serial_rate_max`, which is the same field `rig_init` uses, so
a bridged rig runs at the speed the same rig runs at on a desktop. Nor
is it the control lines: **both** `FtdiSerialDriver` and
`Cp21xxSerialDriver` explicitly deassert DTR and RTS in `openInt()`, so
the K4 is CAT-ing over this transport with them low. That is a
difference from a desktop, where the OS raises them on open, and the
transport now matches the desktop — but it is **not** thought to be the
Icom fix (Andrew): CI-V has no flow control and an Icom uses those
lines only for PTT, CW and RTTY keying. And every Icom in Hamlib declares
`RIG_HANDSHAKE_NONE`, so no flow control is being applied that could
hold the chip's transmitter off.

**Still open, and the chip's configuration is now ruled out
(2026-08-23).** With the blind `SET_FLOW` write below skipped, this app
programs the CP2102N exactly as FT8TW does — its `setParameters` is
behaviourally identical and its older copy of the driver has no flow
control code at all — and FT8TW drives the same radio on the same
phone. So the register writes are not the difference.

What no software above the chip can see is whether the UART clocked the
bytes out: a bulk write returning its length means they reached the
chip. A CP210x will say. `SerialBridge.describeStatus` issues
`GET_COMM_STATUS` (Silicon Labs AN571, 19 bytes) and puts the transmit
queue depth, the receive queue depth, the error mask and the hold
reasons into the trace alongside the modem control lines. Bytes stuck
in `outQueue` mean the chip is holding them and `hold` says why; an
empty `outQueue` with an empty `inQueue` means they went out and the
radio said nothing.

**It is reported from the write path, and where it is reported was a
mistake worth recording.** The first version logged it from the read
loop, every ten consecutive empty reads — and printed nothing at all,
which turned out to be the finding: the read loop was not cycling. A
liveness probe on the thread whose liveness is in question cannot
report. `bulkTransfer` on a CP2102N that has nothing to say blocks for
as long as it likes, where an FTDI returns every ~16 ms because the
chip sends a status packet whether or not there is data — so the K4
never exercised this and the Icom never got past the first read. That
is not itself a fault (a blocked read still delivers data the moment
any arrives) but it is why the instrument was silent.

So the read thread now only *counts* — reads started, reads returned,
and how long the last one took, in relaxed atomics that nothing
synchronises through — and the **write** thread prints them, because
its frames are in the log and it is therefore known to be running. The
probe fires *before* each frame rather than after, so what it describes
is the state the previous frame left behind after Hamlib's full
timeout; probed immediately after a write it would only ever catch a
UART mid-transmission and prove nothing.

**The write that was skipped, and why it stays (2026-08-23).** Setting a
CP210x's flow control to *none* is not a no-op: the library sends the
chip sixteen zero bytes, clobbering `ulControlHandshake`,
`ulFlowReplace`, `ulXonLimit` and `ulXoffLimit` at once. FT8TW drives
the same radio on the same phone with a fork of `Cp21xxSerialDriver`
that has no `SET_FLOW` code in it at all, and the Linux `cp210x`
driver — which is what the working `rigctl` goes through — does
`GET_FLOW`/modify/`SET_FLOW` and never zeroes the limits. The kernel
additionally carries erratum **CP2102N_E104**: firmware ≤ 0x10004 reads
`ulXonLimit` as `ulFlowReplace`, so a blind write lands one word out of
alignment and the chip's own `ulXoffLimit` comes from past the end of
the buffer. The write is now skipped when there is nothing to set —
in `SerialBridge.applyFlowControl` **and** in the vendored
`Cp21xxSerialDriver.openInt`, which does it inside `port.open()` before
any of our code is asked. That is the first patch carried against the
vendored library; see `third_party/usb-serial-for-android/PATCHES.md`.

**How the trace got there:** The
app writes a correct frame — `-> rig 6: fe fe a2 e0 03 fd` — to a
**CP2102N** at 19200 8N1 flow=none, one vendor-class interface, two
endpoints, `Cp21xxSerialDriver` port 0 of 1, and nothing ever comes
back; `rigctl -m 3081 -r /dev/ttyUSB0 -s 19200` on the same cable
answers instantly. Every control transfer the driver makes returns 0
and the bulk write returns 6, so the chip is enumerated, configured and
accepting bytes. **What no software here can see is whether the UART
clocked them out**, which is where reading code stops and a hardware
A/B has to take over — the two worth running are whether FT8CN drives
the same radio on the same phone (it uses the same USB library, so a
pass there means our usage differs and a failure means it is not our
code), and whether a CP210x adapter into the K4's RS-232 port fails the
same way (which would separate the chip from the radio).

Two differences from the Linux `cp210x` driver survive as the shortlist
if the A/B points at the chip: the vendored library writes `SET_FLOW`
as 16 blind zero bytes where Linux does `GET_FLOW`/modify/`SET_FLOW`,
preserving `ulXonLimit`/`ulXoffLimit`; and it writes the requested baud
raw where Linux applies `cp210x_get_actual_rate()` for a CP2102N (a
no-op at 19200, which divides 48 MHz exactly). Linux also carries
erratum **CP2102N_E104** — firmware ≤ 0x10004 reads `ulXonLimit` as
`ulFlowReplace`, so it declares flow control unsupported on those
parts. None of it is confirmed to matter.

**The earlier reading of that trace was:** `fe fe a2 e0 03 fd` — address A2, the IC-9700's
default — written, then `read_string_generic(): Timed out 1.001 seconds
after 0 chars`, every time, at 115200. So `icom_get_usb_echo_off`
returns `-RIG_ETIMEOUT` and `icom_rig_open` gives up with "is rig on and
connected?". What that trace cannot say is whether the six bytes reached
the USB endpoint — Hamlib only ever saw a successful *socket* write —
which is what the second sink below was added to answer.

Three things to check on the radio, all Icom-specific and none visible
from here: **the CI-V USB baud rate** (a menu item separate from the
CI-V port's own, defaulting to an Auto that is not reliable on every
model — note 115200 is not this app's default either, since
`serial_defaults` gives 38400 for an IC-9700 and 19200 for an IC-7100);
**CI-V USB Echo Back**, which is exactly what `icom_get_usb_echo_off`
probes at open and gets one attempt at, `rig_open` having set `retry` to
0 for the duration; and **the CI-V address**, which is a separate
setting for the USB port when "CI-V USB Port" is "Unlink from
[REMOTE]" — the app sends to the factory-default A2, and a changed
address produces exactly this silence.

## Reading the rig trace on a phone

`SSTVAE_HAMLIB_DEBUG` raises Hamlib's level and Hamlib writes to
**stderr**, which a desktop operator can capture and a phone discards.
So on the one platform where rig control is newest, the single artifact
that answers "how far did `rig_open` get, and what did the radio say"
was unreachable. (Hamlib's own `debug.c` has an `#ifdef ANDROID` branch
that would log to logcat — it is dead in any build of ours and probably
in most: `configure.ac` uses `ANDROID` only as an automake conditional,
never as a preprocessor define, and never links `-llog`.)

`rig::set_debug_sink` (`core/rig/hamlib.hpp`) registers a
`vprintf_cb_t` and turns the trace into a stream of whole lines.
**Settings → Rig control → Log rig traffic** installs a sink that keeps
the last 600 lines and mirrors each to logcat; Refresh, Copy and Clear
are under the switch. Off by default — at `RIG_DEBUG_TRACE` it is a line
per CAT frame.

**Hamlib's trace is only half of it, and the missing half is ours.**
For a radio that never answers, Hamlib's view is "I wrote six bytes and
read nothing", repeated — which is byte for byte what a bug in our own
bridge would produce. So `core/rig/trace.hpp` is a second sink, Qt-free
and Hamlib-free in `sstvae_core`, that the loopback bridge and the
Android transport write to; the switch installs both, into one ring, so
the two interleave. It reports what actually reached the transport
(after the write, never before — a line logged on the way in would say
"delivered" for a write that threw), what came back from it, the
resolved line settings, and — from `SerialBridge.describeLink` — the
device's ids, its interface count and classes, which driver
`UsbSerialProber` chose and which of its ports was opened. That last
part is the one nothing else can supply, and it is where an FTDI on its
own and a vendor bridge inside a composite device stop looking alike.
Tracing off costs one relaxed atomic load, which is why the calls sit
in the byte pump unconditionally.

Three things about it are load-bearing rather than incidental:

- **The callback is registered only while a sink exists.** `rig_debug`
  writes to stderr *or* to the callback, never both, so one left
  registered swallows the trace for every later run — silently, since a
  swallowed trace and a quiet library are the same output.
  `test_rig_hamlib.cpp` captures stderr and requires it back, and that
  assertion was mutation-tested: registering unconditionally fails it
  and nothing else.
- **The ring is a process-wide singleton, not a member of `RigControl`.**
  QML destroys and rebuilds that view on every rotation while the rig
  session outlives both, so a per-view buffer would discard the trace of
  the open being diagnosed. It also keeps the sink's captures free of
  `this`, which matters because it is called from the rig worker thread.
- **The text area is refreshed by hand.** The rig screen republishes at
  1 Hz and a bound `text:` would reset the scroll position every second,
  which is unreadable in exactly the situation it is for.

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

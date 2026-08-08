# SSTVAE Android smoke test

**Not the app.** This is a pre-Tier-0 probe with one job: prove that
Android can select an audio input, capture from it, and decode a picture
through the real code path. See `docs/android.md` for the Tier 0 design,
which this deliberately does not follow — there is no foreground service,
no gallery, no waterfall and no notification here.

It needs **no Qt**. Device selection plus receive is the Java audio layer,
`native/core/`, and JNI; a plain Views UI is enough. That keeps the Qt
Quick decision ahead of us and costs nothing, because the pieces worth
keeping are toolkit-independent anyway.

## What is reusable and what is throwaway

Reusable, and already sitting in the right place:

- `audio::CapturePipeline` (`native/core/audio/audio.hpp`) — the
  bytes → mono → resample conversion every backend must do, in the
  Qt-free layer with a host test, so a second backend cannot
  reintroduce the three bugs it encodes.
- `AudioDevices.java`, `CaptureThread.java` — enumeration and the
  blocking `AudioRecord` reader, close to what Tier 0 wants.
- The onnxruntime Android AAR support in `native/cmake/onnxruntime.cmake`.

Throwaway: the JSON status string, the single global session in
`jni_bridge.cpp`, `WavFeeder`, and the entire UI.

## Building

Needs the SDK, NDK 28.2.13676358 and CMake 3.31.6:

```sh
sdkmanager "ndk;28.2.13676358" "cmake;3.31.6" "platforms;android-36"
echo "sdk.dir=$HOME/Android/Sdk" > native/android/local.properties
cd native/android && JAVA_HOME=/opt/android-studio/jbr ./gradlew :app:assembleDebug
```

The Gradle wrapper is pinned to 8.13 on purpose. System Gradle 9.6+ fails
outright — it removed an internal API that AGP 8.x still uses — and
beyond that, letting a build depend on whichever `gradle` is on `PATH` is
the same hazard the project already rejects for `makensis` and
`appimagetool`.

CMake downloads the onnxruntime AAR (sha256-pinned, same shape as the
desktop archives). `-DSSTVAE_BUILD_CODEC=OFF` skips it and still gives a
working app that reports frames, SNR and callsign without making a
picture — available only because `rx::Decoder` is a seam.

## Running

```sh
adb install -r app/build/outputs/apk/debug/app-debug.apk
adb shell pm grant org.cleverdomain.sstvae.smoke android.permission.RECORD_AUDIO
# the decoder, since there is no Hub fetcher here (that is QtNetwork)
adb push ~/.cache/sstvae/models/v3-decoder-fp16.onnx /data/local/tmp/dec.onnx
adb shell "run-as org.cleverdomain.sstvae.smoke sh -c \
  'cat /data/local/tmp/dec.onnx > /data/data/org.cleverdomain.sstvae.smoke/files/models/v3-decoder-fp16.onnx'"
```

Tap **Start** to capture from the selected input.

Two hidden gestures, both diagnostics rather than UI:

- **Long-press Start** feeds `/data/local/tmp/tx48.wav` (16-bit mono,
  any rate) through the same `Native.push` path instead of opening the
  microphone. It pushes in ragged chunks and in real time, so the
  chunk-boundary handling and the 5-second poll cadence are both
  exercised. This is how the decode path gets tested where the
  emulator's virtual mic cannot carry a signal.
- **Long-press the status text** dumps the ring buffer to
  `files/ring.wav`. Pull it and decode it on the host — comparing two
  captures of one playback is the diagnostic that separates "the
  microphone is deaf" from "our pipeline is broken", and it is what did
  so here.

## Emulator caveat

The emulator's virtual microphone **cannot carry the signal**: measured
at 1/60th of the reference amplitude with two thirds of its power outside
the 900–2150 Hz band. Route a host loopback into it if you like — the
recipe in `CLAUDE.md` plus `pactl move-source-output <id> sstvae_loop` —
but what arrives is not the transmission. Use the WAV feeder on the
emulator, and a real device for anything about the driver.

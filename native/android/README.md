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
# The decoder, since there is no Hub fetcher here (that is QtNetwork).
# External files dir: no run-as, no root, and visible over MTP so the
# file can just be dragged across instead.
adb push ~/.cache/sstvae/models/v3-decoder-fp16.onnx \
  /storage/emulated/0/Android/data/org.cleverdomain.sstvae.smoke/files/models/
```

The app searches its internal `files/models`, then that external
directory, then `/sdcard/Download`. If the decoder is in none of them it
says so **when you press Start**, naming the path and the push command —
not when a picture finally arrives. `OnnxCodec` loads its parts lazily on
purpose (a receive-only station never touches the encoder), and on real
hardware that laziness put "file doesn't exist" at the end of a whole
transmission with everything until then reporting success. `preload` is
in the API for exactly this and the smoke test now calls it.

Tap **Start** to capture from the selected input. While capturing, the
status text reports the audio side:

```
in: UNPROCESSED 48000 Hz  ->  USB Device "Digirig"
    peak -18 dBFS   2.1% near-zero
```

That first line is **what actually opened and where it actually routed**,
asked of `AudioRecord` rather than echoed back from the request — the
two can differ, and on a USB test that difference is the entire
question. If the platform refuses the routing, a sticky `!` line says so
rather than a Toast that fades: capturing from the built-in mic while
the UI claims USB is the worst outcome available here.

The level is reported as peak dBFS *and* the share of near-silent
samples, because those answer different questions. "Too quiet" and
"nothing is arriving" have the same mean level and are not the same
problem — which is exactly how the emulator's zeroed audio was
misdiagnosed once already.

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

**Turn on Extended Controls > Microphone > "Virtual microphone uses host
audio input"** before expecting any audio. It is a per-AVD switch, off by
default, and it is *not* the same as the `-allow-host-audio` command-line
flag — that flag alone changes nothing. With the switch on, a host
loopback decodes on the emulator at 33.2 dB.

Without it the emulator hands back zeroed audio and the microphone path
cannot be tested at all. `AudioRecord` returns the correct byte rate (96037/s
against 96000 expected) while 99.9% of samples sit below 16 LSB with
occasional full-scale impulses. `emulator -help` describes the symptom under `-allow-host-audio`
("Otherwise, zeroes out audio"), which is misleading: that flag alone
did not fix it, and the AVD switch alone did.

The WAV feeder remains the quicker path when you only want a signal —
it needs no routing at all — and a real device is still the only place
the driver itself gets tested.

Do not read a low capture level as "the device is deaf" — that was the
first (wrong) conclusion here, and it nearly ended the investigation.
A path delivering silence and a path delivering quiet audio have the
same RMS; the percentiles tell them apart, which is why the capture
thread logs `% below 16 LSB` and peak rather than a mean level.

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

**Status: skeleton.** Qt Quick, `sstvae_core` and the onnxruntime AAR
build and package together for arm64. None of Tier 0's actual behaviour
is here yet — see the plan below.

## Building

Needs Qt for Android **matching your host Qt version**, the NDK, and the
SDK. Both kits via [aqtinstall](https://github.com/miurahr/aqtinstall):

```sh
aqt install-qt linux desktop 6.11.1 linux_gcc_64 -O ~/Qt -m qtimageformats
aqt install-qt all_os android 6.11.1 android_arm64_v8a -O ~/Qt \
    -m qtimageformats qtshadertools
```

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
  -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-28 \
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

arm64-v8a only, deliberately. The emulator is x86_64 and **cannot carry
audio at all** (it hands back zeroed buffers), so an x86_64 kit would
double the download to enable a device that cannot run the one test that
matters. Develop against a real phone.

## What is left

In roughly the order the doc argues for:

1. **`core/audio/android/`** — the permanent audio layer: seven entry
   points mirroring `core/audio/qt/qtaudio.hpp`, so `InputStream` and
   `play()` drop into the existing seams unchanged. The smoke test's
   Java `AudioDevices`/`CaptureThread` are most of it already, and
   `audio::CapturePipeline` is done and host-tested.
2. **The foreground service**, owning the engine, with the UI as a
   detachable view. This inverts the desktop's `AppState` and is cheap
   only if it is designed in: nothing in the UI may own engine state,
   and every live display must be reconstructible from `SharedState` on
   attach.
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

#!/usr/bin/env bash
# Build an installable, *optimized* Android APK.
#
# This script exists for one reason: the obvious command produces an
# -O0 build. `--target apk` with the default configuration signs with
# the debug keystore and installs, but passes no -O flag (clang's
# default is -O0), while RelWithDebInfo compiles at -O2 and then emits
# `-release-unsigned.apk`, which will not install. Faced with that,
# everyone picks Debug -- and the receive loop, which is scalar
# floating-point DSP over a 130 s ring buffer, runs 6-15x slower for
# it. On a Galaxy S25+ that was 5-8 s per poll against a fraction of a
# second, and it read as an onnxruntime problem rather than a build
# one.
#
# So: configure RelWithDebInfo, build, zipalign, and sign with the same
# debug keystore Gradle would have used. The result installs exactly
# like a debug APK and runs at full speed. It is *not* a release
# artifact -- the debug key is world-known and every machine's is
# different -- but nothing here is claiming otherwise.
#
# Usage:
#   tools/build_android.sh [--install] [--debug] [--abi <abi>] [-- <cmake args>]
set -euo pipefail

BUILD_DIR=${BUILD_DIR:-build-android}
BUILD_TYPE=RelWithDebInfo
INSTALL=0
ALL_ABIS=ON
EXTRA=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --install) INSTALL=1; shift ;;
        # Only for single-stepping the C++. Everything timing-related
        # measured in this configuration is fiction.
        --debug) BUILD_TYPE=Debug; shift ;;
        --abi) ALL_ABIS=OFF; EXTRA+=("-DANDROID_ABI=$2"); shift 2 ;;
        --) shift; EXTRA+=("$@"); break ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done

: "${QT_ANDROID:?set QT_ANDROID to e.g. \$HOME/Qt/6.11.1/android_arm64_v8a}"
: "${QT_HOST:?set QT_HOST to e.g. \$HOME/Qt/6.11.1/gcc_64}"
# ANDROID_HOME beats ANDROID_SDK_ROOT for the emulator's system-image
# lookup, and a profile exporting one to a different SDK costs an
# afternoon. Keep them agreeing.
SDK=${ANDROID_HOME:-${ANDROID_SDK_ROOT:-$HOME/Android/Sdk}}
NDK=${ANDROID_NDK_ROOT:-$(ls -d "$SDK"/ndk/* | sort -V | tail -1)}
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

cmake -S "$ROOT/native/android-app" -B "$BUILD_DIR" -G Ninja \
    -DCMAKE_TOOLCHAIN_FILE="$QT_ANDROID/lib/cmake/Qt6/qt.toolchain.cmake" \
    -DQT_HOST_PATH="$QT_HOST" \
    -DQT_ANDROID_BUILD_ALL_ABIS=$ALL_ABIS \
    -DANDROID_SDK_ROOT="$SDK" \
    -DANDROID_NDK_ROOT="$NDK" \
    -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
    "${EXTRA[@]}"
cmake --build "$BUILD_DIR" --target apk

OUT="$BUILD_DIR/sstvae-$BUILD_TYPE.apk"
if [[ $BUILD_TYPE == Debug ]]; then
    cp "$BUILD_DIR"/android-build/build/outputs/apk/debug/*-debug.apk "$OUT"
else
    UNSIGNED=$(echo "$BUILD_DIR"/android-build/build/outputs/apk/release/*-unsigned.apk)
    BT=$(ls -d "$SDK"/build-tools/* | sort -V | tail -1)
    KS=${ANDROID_DEBUG_KEYSTORE:-$HOME/.android/debug.keystore}
    if [[ ! -f $KS ]]; then
        echo "no debug keystore at $KS -- run any Gradle debug build once, or" >&2
        echo "set ANDROID_DEBUG_KEYSTORE to one you have." >&2
        exit 1
    fi
    # zipalign *before* signing: apksigner validates alignment and the
    # other order silently produces something that installs but loads
    # its native libraries by copying them out of the APK.
    "$BT/zipalign" -f 4 "$UNSIGNED" "$OUT"
    "$BT/apksigner" sign --ks "$KS" --ks-pass pass:android \
        --ks-key-alias androiddebugkey --key-pass pass:android "$OUT"
fi

echo "built: $OUT"
[[ $INSTALL == 1 ]] && "$SDK/platform-tools/adb" install -r "$OUT"
exit 0

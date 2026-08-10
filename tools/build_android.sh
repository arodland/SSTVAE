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
# `--aab` builds a Play Store bundle instead, and is a different job in
# one respect that matters: an APK here is signed with the *debug* key,
# which is world-known and fine for a sideload, while a bundle is signed
# with the **upload key** and is an identity. So the two paths do not
# share a fallback -- `--aab` refuses to produce anything rather than
# quietly emitting an unsigned or debug-signed bundle, because Play
# rejects both and the rejection arrives minutes later in a browser,
# a long way from here.
#
# Usage:
#   tools/build_android.sh [--install] [--debug] [--abi <abi>] [-- <cmake args>]
#   tools/build_android.sh --aab [--version-code N] [-- <cmake args>]
#
# Signing an AAB reads the keystore from the environment:
#   SSTVAE_UPLOAD_KEYSTORE       path to the PKCS12 upload keystore
#   SSTVAE_UPLOAD_KEYSTORE_PASS  a *file* holding its password
#   SSTVAE_UPLOAD_ALIAS          key alias (default sstvae-upload)
# A file rather than the password itself, because an argument or an
# exported value is visible in `ps` to every process on the machine.
set -euo pipefail

BUILD_DIR=${BUILD_DIR:-build-android}
BUILD_TYPE=RelWithDebInfo
INSTALL=0
ALL_ABIS=ON
AAB=0
VERSION_CODE=
EXTRA=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --install) INSTALL=1; shift ;;
        # Only for single-stepping the C++. Everything timing-related
        # measured in this configuration is fiction.
        --debug) BUILD_TYPE=Debug; shift ;;
        --abi) ALL_ABIS=OFF; EXTRA+=("-DANDROID_ABI=$2"); shift 2 ;;
        --aab) AAB=1; shift ;;
        --version-code) VERSION_CODE=$2; shift 2 ;;
        --) shift; EXTRA+=("$@"); break ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done

if [[ $AAB == 1 ]]; then
    # A bundle is what Play serves every device from, so a single-ABI
    # one silently ships a store listing that half the phones in the
    # world cannot install. Refuse rather than warn.
    [[ $ALL_ABIS == ON ]] || { echo "--aab cannot be combined with --abi" >&2; exit 2; }
    [[ $BUILD_TYPE == RelWithDebInfo ]] || { echo "--aab cannot be combined with --debug" >&2; exit 2; }
    [[ $INSTALL == 0 ]] || { echo "--aab produces a bundle; adb cannot install one" >&2; exit 2; }

    KS=${SSTVAE_UPLOAD_KEYSTORE:-$HOME/.android-keys/sstvae-upload.jks}
    PASSFILE=${SSTVAE_UPLOAD_KEYSTORE_PASS:-$HOME/.android-keys/sstvae-upload.pass}
    ALIAS=${SSTVAE_UPLOAD_ALIAS:-sstvae-upload}
    # **Resolved here, before cmake is even configured**, because the
    # only thing worse than having no upload key is finding that out
    # after a multi-minute two-ABI build.
    #
    # The same JDK that androiddeployqt runs Gradle with. Not `which
    # jarsigner`: a distro JRE on PATH may have none, and the failure is
    # then "command not found" at the end of that same long build.
    JARSIGNER=${JAVA_HOME:+$JAVA_HOME/bin/jarsigner}
    [[ -x ${JARSIGNER:-} ]] || JARSIGNER=$(command -v jarsigner || true)
    if [[ -z ${JARSIGNER:-} || ! -x $JARSIGNER ]]; then
        echo "no jarsigner -- set JAVA_HOME to a JDK (not a JRE)." >&2
        exit 1
    fi
    if [[ ! -f $KS || ! -f $PASSFILE ]]; then
        echo "no upload keystore ($KS) or password file ($PASSFILE)." >&2
        echo "This is the key Play identifies your uploads by -- see" >&2
        echo "native/android-app/README.md, 'The Play upload'." >&2
        exit 1
    fi
fi
[[ -n $VERSION_CODE ]] && EXTRA+=("-DSSTVAE_ANDROID_VERSION_CODE=$VERSION_CODE")

: "${QT_ANDROID:?set QT_ANDROID to e.g. \$HOME/Qt/6.11.1/android_arm64_v8a}"
: "${QT_HOST:?set QT_HOST to e.g. \$HOME/Qt/6.11.1/gcc_64}"
# ANDROID_HOME beats ANDROID_SDK_ROOT for the emulator's system-image
# lookup, and a profile exporting one to a different SDK costs an
# afternoon. Keep them agreeing.
SDK=${ANDROID_HOME:-${ANDROID_SDK_ROOT:-$HOME/Android/Sdk}}
NDK=${ANDROID_NDK_ROOT:-$(ls -d "$SDK"/ndk/* | sort -V | tail -1)}
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

# The bundled model artifacts, downloaded once and shared by every ABI's
# nested build -- see SSTVAE_ANDROID_MODEL_DIR. An absolute path, since
# the per-ABI builds configure from a different working directory.
MODEL_DIR=${SSTVAE_ANDROID_MODEL_DIR:-$(mkdir -p "$BUILD_DIR" && cd "$BUILD_DIR" && pwd)/sstvae-models}

cmake -S "$ROOT/native/android-app" -B "$BUILD_DIR" -G Ninja \
    -DCMAKE_TOOLCHAIN_FILE="$QT_ANDROID/lib/cmake/Qt6/qt.toolchain.cmake" \
    -DQT_HOST_PATH="$QT_HOST" \
    -DSSTVAE_ANDROID_MODEL_DIR="$MODEL_DIR" \
    -DQT_ANDROID_BUILD_ALL_ABIS=$ALL_ABIS \
    -DANDROID_SDK_ROOT="$SDK" \
    -DANDROID_NDK_ROOT="$NDK" \
    -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
    "${EXTRA[@]}"
if [[ $AAB == 1 ]]; then
    cmake --build "$BUILD_DIR" --target aab
    UNSIGNED=$(echo "$BUILD_DIR"/android-build/build/outputs/bundle/release/*-release.aab)
    OUT="$BUILD_DIR/sstvae-upload.aab"
    # jarsigner, not apksigner: an AAB is a jar and apksigner does not
    # handle one. The bundle is not zipaligned either -- alignment is a
    # property of the APKs Play *generates* from it, which is also why
    # the 16 KB page alignment that Play requires has to be right in the
    # .so files themselves rather than fixable here.
    "$JARSIGNER" -keystore "$KS" \
        -storepass:file "$PASSFILE" -keypass:file "$PASSFILE" \
        -sigalg SHA256withRSA -digestalg SHA-256 \
        -signedjar "$OUT" "$UNSIGNED" "$ALIAS" >/dev/null
    "$JARSIGNER" -verify "$OUT" | head -1
    echo "built: $OUT"
    exit 0
fi

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

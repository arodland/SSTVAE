#!/usr/bin/env bash
# Type-check the app's Android Java without the Android SDK.
#
# **Written because a one-line Java mistake cost a build round on
# hardware.** `native/`'s C++ is checked six ways -- ctest, the golden
# vectors, `pytest --native`, `check_includes.py`, `check_layering.py`, a
# mingw cross-compile -- and its Java was checked by nobody until an APK
# was built. `catch (IOException | UnsupportedOperationException |
# RuntimeException e)` is not a syntax error and no parser would have
# caught it; javac catches it in two seconds, and the only thing missing
# was an `android.jar`.
#
# Robolectric's `android-all` is that jar, on Maven Central, pinned and
# checksummed like onnxruntime, Hamlib and NSIS. It is the real API
# surface rather than a stub set, so it cannot quietly drift from what
# the app compiles against. **This checks types, not behaviour** -- it
# cannot run anything, and Robolectric's implementations are not
# Android's.
#
# Usage: tools/check_android_java.sh [--cache DIR]
set -euo pipefail

# Pinned, with a checksum, for the reason `native/cmake/onnxruntime.cmake`
# records: a version resolved at build time makes a build's success a
# property of somebody else's current contents.
ANDROID_ALL_VERSION="14-robolectric-10818077"
ANDROID_ALL_SHA256="6be2218c6a53fe3c57bc22ebdc723edcb7270a8a6f187545708aa5c0ed813977"
ANDROID_ALL_URL="https://repo1.maven.org/maven2/org/robolectric/android-all/${ANDROID_ALL_VERSION}/android-all-${ANDROID_ALL_VERSION}.jar"

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cache="${SSTVAE_ANDROID_JAR_CACHE:-${TMPDIR:-/tmp}/sstvae-android-jar}"

while [ $# -gt 0 ]; do
    case "$1" in
        --cache) cache="$2"; shift 2 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

if ! command -v javac >/dev/null 2>&1; then
    echo "check_android_java: no javac on PATH; skipping" >&2
    exit 0
fi

mkdir -p "$cache"
jar="$cache/android-all-${ANDROID_ALL_VERSION}.jar"
if [ ! -f "$jar" ]; then
    echo "check_android_java: fetching android-all ${ANDROID_ALL_VERSION} (~130 MB)"
    if ! curl -fsSL --max-time 600 -o "$jar.part" "$ANDROID_ALL_URL"; then
        rm -f "$jar.part"
        echo "check_android_java: could not fetch the Android jar; skipping" >&2
        exit 0
    fi
    # Renamed only after the checksum passes, like `qt_fetcher`: a
    # truncated jar left in the cache would be found on the next run and
    # fail a long way from its cause.
    got="$(sha256sum "$jar.part" | cut -d' ' -f1)"
    if [ "$got" != "$ANDROID_ALL_SHA256" ]; then
        rm -f "$jar.part"
        echo "check_android_java: sha256 mismatch: got $got" >&2
        exit 1
    fi
    mv "$jar.part" "$jar"
fi

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

# Three classes that are neither `android.*` nor ours: two from AndroidX
# and one from Qt. Stubbed rather than fetched, and the trade is worth
# stating -- a stub makes the *import* resolve and pins the *signature we
# use*, so a wrong argument list here still fails; what it cannot catch
# is upstream changing that signature. `IntDef` is `@Retention(SOURCE)`
# and has no runtime meaning at all. `FileProvider.getUriForFile` has had
# this shape for a decade. `QtActivity` is only ever named as a class
# literal, and Qt's own jar is not on Maven Central to fetch.
mkdir -p "$work/stub/androidx/annotation" "$work/stub/androidx/core/content" \
         "$work/stub/org/qtproject/qt/android/bindings"
cat > "$work/stub/androidx/annotation/IntDef.java" <<'EOF'
package androidx.annotation;
import java.lang.annotation.*;
@Retention(RetentionPolicy.SOURCE)
@Target({ElementType.ANNOTATION_TYPE, ElementType.METHOD, ElementType.PARAMETER,
         ElementType.FIELD, ElementType.LOCAL_VARIABLE})
public @interface IntDef {
    int[] value() default {};
    boolean flag() default false;
    boolean open() default false;
}
EOF
cat > "$work/stub/androidx/core/content/FileProvider.java" <<'EOF'
package androidx.core.content;
import android.content.Context;
import android.net.Uri;
import java.io.File;
public class FileProvider {
    public static Uri getUriForFile(Context context, String authority, File file) {
        throw new UnsupportedOperationException("stub");
    }
    public static Uri getUriForFile(Context context, String authority, File file,
                                    String displayName) {
        throw new UnsupportedOperationException("stub");
    }
}
EOF
cat > "$work/stub/org/qtproject/qt/android/bindings/QtActivity.java" <<'EOF'
package org.qtproject.qt.android.bindings;
public class QtActivity extends android.app.Activity {}
EOF

# Every Java source the APK carries, in the same layout
# `native/android-app/CMakeLists.txt` assembles. The vendored library is
# on the sourcepath rather than the file list: it is compiled only as far
# as our code reaches into it, which is what we want to check.
sources="$work/stub"
for d in \
    "$repo_root/native/android-app/android/src" \
    "$repo_root/native/core/audio/android/java" \
    "$repo_root/native/core/rig/android/java" \
    "$repo_root/native/third_party/usb-serial-for-android/java" \
    "$repo_root/native/third_party/usb-serial-for-android/shim"
do
    sources="$sources:$d"
done

mapfile -t files < <(find \
    "$repo_root/native/android-app/android/src" \
    "$repo_root/native/core/audio/android/java" \
    "$repo_root/native/core/rig/android/java" \
    -name '*.java' | sort)

if [ "${#files[@]}" -eq 0 ]; then
    echo "check_android_java: no sources found" >&2
    exit 1
fi

# `-nowarn` because Robolectric's jar is a different API level than the
# app targets and the deprecation noise is not ours to act on; a real
# error still fails.
#
# **javac's exit status, not whether anything was printed.** Piping it
# straight into `grep` and testing that reads correctly and is wrong
# under `pipefail`, which reports the *pipeline's* first failure -- so a
# failing javac made the `if` false and the check passed while printing
# its own errors. Caught on this script's first run, by the compile
# errors it was written to find.
set +e
output="$(javac -nowarn -proc:none -d "$work/out" \
    -cp "$jar" -sourcepath "$sources" "${files[@]}" 2>&1)"
status=$?
set -e

# JAVA_TOOL_OPTIONS is echoed by the JVM when the environment sets it,
# which some sandboxes do; `Note:` lines are deprecation summaries.
filtered="$(printf '%s\n' "$output" | grep -v 'JAVA_TOOL_OPTIONS' | grep -v '^Note:' || true)"

if [ "$status" -ne 0 ]; then
    printf '%s\n' "$filtered" >&2
    echo "check_android_java: FAILED" >&2
    exit 1
fi

echo "android java ok (${#files[@]} source files checked)"

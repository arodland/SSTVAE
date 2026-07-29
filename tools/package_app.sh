#!/usr/bin/env bash
# Stage a runnable copy of the native app, Qt and all.
#
#   tools/package_app.sh [build-dir] [staging-dir]
#
# This is step 2 of the packaging work: an archive somebody can unpack
# and *run*, which is what makes on-air testing possible before any
# installer or signing exists. It is deliberately not an installer --
# no icons, no file associations, no uninstall. Step 3 is where those
# arrive, most likely as CMake install() rules plus CPack, at which
# point this script probably goes away.
#
# One script rather than three CI blocks, because bash is present on all
# three runners and because a packaging step that can only be exercised
# by pushing is one that gets debugged six minutes at a time.
#
# What has to end up beside the executable, and why each is easy to
# forget:
#
#   Qt                 the deploy tools handle it on Windows and macOS;
#                      on Linux there is no such tool in the box, so the
#                      libraries and the platform plugins are copied by
#                      hand and found through a wrapper script.
#   libhamlib          pinned and bundled by us, so it is never on the
#                      target machine.
#   onnxruntime        likewise.
#
# The model artifacts are *not* bundled: they are fetched on first run
# and cached (see docs/onnx.md), which is what keeps this download small
# and lets a receive-only station skip the encoder entirely.

set -euo pipefail

BUILD_DIR="${1:-native/build}"
STAGE_DIR="${2:-dist}"

if [ ! -d "$BUILD_DIR" ]; then
    echo "package_app: no build directory at $BUILD_DIR" >&2
    exit 1
fi

cache="$BUILD_DIR/CMakeCache.txt"
# Read the dependency locations back out of the cache rather than
# guessing them: they are pinned versions under FETCHCONTENT_BASE_DIR,
# and that path is a configure-time choice.
cache_value() { sed -n "s|^$1:[A-Z]*=||p" "$cache" | head -1; }
HAMLIB_RUNTIME_DIR="$(cache_value SSTVAE_HAMLIB_RUNTIME_DIR || true)"
ORT_LIBDIR="$(cache_value SSTVAE_ONNXRUNTIME_LIBDIR || true)"

rm -rf "$STAGE_DIR"
mkdir -p "$STAGE_DIR"

case "$(uname -s)" in
# ---------------------------------------------------------------- Windows
MINGW*|MSYS*|CYGWIN*)
    app="$STAGE_DIR/sstvae"
    mkdir -p "$app"
    cp "$BUILD_DIR/sstvae-gui.exe" "$app/"
    cp "$BUILD_DIR/sstvae-decode.exe" "$app/" 2>/dev/null || true
    # Ours first, so windeployqt sees a complete executable and does not
    # report an import it cannot resolve.
    [ -n "$HAMLIB_RUNTIME_DIR" ] && cp "$HAMLIB_RUNTIME_DIR"/*.dll "$app/"
    [ -n "$ORT_LIBDIR" ] && cp "$ORT_LIBDIR/onnxruntime.dll" "$app/"
    windeployqt --release --no-translations --no-system-d3d-compiler \
        --no-opengl-sw "$app/sstvae-gui.exe"
    ;;

# ------------------------------------------------------------------ macOS
Darwin)
    # CMake built it as a bundle already (MACOSX_BUNDLE), so the layout
    # exists; what it does not have is anything it links from
    # native/.deps, whose absolute paths mean nothing on another Mac.
    cp -R "$BUILD_DIR/sstvae-gui.app" "$STAGE_DIR/"
    app="$STAGE_DIR/sstvae-gui.app"
    mkdir -p "$app/Contents/Frameworks"
    [ -n "$HAMLIB_RUNTIME_DIR" ] && cp "$HAMLIB_RUNTIME_DIR"/libhamlib*.dylib \
        "$app/Contents/Frameworks/" 2>/dev/null || true
    [ -n "$ORT_LIBDIR" ] && cp "$ORT_LIBDIR"/libonnxruntime*.dylib \
        "$app/Contents/Frameworks/" 2>/dev/null || true
    # -libpath so macdeployqt can resolve what it is about to rewrite;
    # it fixes the install names and rpaths for everything it finds.
    macdeployqt "$app" -verbose=1 \
        ${HAMLIB_RUNTIME_DIR:+-libpath="$HAMLIB_RUNTIME_DIR"} \
        ${ORT_LIBDIR:+-libpath="$ORT_LIBDIR"}
    ;;

# ------------------------------------------------------------------ Linux
*)
    # No deploy tool in the box. linuxdeploy would do this, but it is
    # another pinned download for a layout that is twenty lines by hand,
    # and step 3's AppImage will bring its own anyway.
    app="$STAGE_DIR/sstvae"
    mkdir -p "$app/bin" "$app/lib" "$app/plugins"
    cp "$BUILD_DIR/sstvae-gui" "$app/bin/"
    cp "$BUILD_DIR/sstvae-decode" "$app/bin/" 2>/dev/null || true
    [ -n "$HAMLIB_RUNTIME_DIR" ] && cp -P "$HAMLIB_RUNTIME_DIR"/libhamlib.so* "$app/lib/"
    [ -n "$ORT_LIBDIR" ] && cp -P "$ORT_LIBDIR"/libonnxruntime.so* "$app/lib/"

    # Qt: the libraries the binary actually links, plus the plugin
    # directories Qt loads by name at run time. `ldd` gives the first
    # set; the second cannot be discovered that way, because nothing
    # links a plugin -- which is exactly how an app ships fine and then
    # dies with "could not find the Qt platform plugin xcb".
    qt_lib_dir="$(ldd "$BUILD_DIR/sstvae-gui" | sed -n 's|.*=> \(.*/libQt6Core\.so[^ ]*\).*|\1|p' \
                  | head -1 | xargs -r dirname)"
    if [ -n "$qt_lib_dir" ]; then
        ldd "$BUILD_DIR/sstvae-gui" \
            | sed -n 's|.*=> \([^ ]*libQt6[^ ]*\) .*|\1|p' \
            | while read -r lib; do cp -P "$lib"* "$app/lib/" 2>/dev/null || true; done
        # ICU and friends live beside Qt and are not distro packages on
        # every target.
        for extra in libicui18n libicuuc libicudata; do
            cp -P "$qt_lib_dir/$extra.so"* "$app/lib/" 2>/dev/null || true
        done
        # Ask Qt where its plugins are; do not infer it from the library
        # path. aqt puts them at <qt>/plugins and a distro at
        # <libdir>/qt6/plugins, so a relative guess is right on exactly
        # one of the two -- and it fails *silently*, leaving an empty
        # plugins directory and an app that dies at startup with "could
        # not find the Qt platform plugin xcb".
        plugins=""
        for q in qtpaths6 qtpaths qmake6 qmake; do
            if command -v "$q" >/dev/null 2>&1; then
                plugins="$("$q" -query QT_INSTALL_PLUGINS 2>/dev/null || true)"
                [ -n "$plugins" ] && [ -d "$plugins" ] && break
                plugins=""
            fi
        done
        # Last resort, for a Qt with no tools on PATH.
        if [ -z "$plugins" ]; then
            for guess in "$qt_lib_dir/../plugins" "$qt_lib_dir/qt6/plugins"; do
                [ -d "$guess" ] && plugins="$guess" && break
            done
        fi
        for kind in platforms xcbglintegrations imageformats \
                    multimedia tls iconengines platformthemes; do
            [ -d "$plugins/$kind" ] && cp -R "$plugins/$kind" "$app/plugins/"
        done
        # A bundle with no platform plugin cannot start, so refuse to
        # produce one and call it a package.
        if [ ! -d "$app/plugins/platforms" ]; then
            echo "package_app: no Qt platform plugin found (looked in '${plugins:-nowhere}')" >&2
            exit 1
        fi
    fi

    # A wrapper, because the alternative is asking the user to set two
    # environment variables correctly before the app will start.
    cat > "$app/sstvae-gui" <<'LAUNCH'
#!/bin/sh
# Launcher: points the dynamic loader and Qt at the bundled copies.
here="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
export LD_LIBRARY_PATH="$here/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export QT_PLUGIN_PATH="$here/plugins${QT_PLUGIN_PATH:+:$QT_PLUGIN_PATH}"
exec "$here/bin/sstvae-gui" "$@"
LAUNCH
    chmod +x "$app/sstvae-gui"
    ;;
esac

echo "package_app: staged into $STAGE_DIR"
find "$STAGE_DIR" -maxdepth 2 -mindepth 1 | sort | head -20

#!/usr/bin/env bash
# Stage a runnable copy of the native app, Qt and all.
#
#   tools/package_app.sh [build-dir] [staging-dir]
#
# Staging is step 2 of the packaging work: an archive somebody can unpack
# and *run*, which is what makes on-air testing possible before any
# installer or signing exists. `tools/make_installer.sh` then turns this
# tree into the platform's own container -- a .dmg, an AppImage, a setup
# .exe -- so the two halves stay separable: a developer can stage and run
# without any installer tooling present at all.
#
# One script rather than three CI blocks, because bash is present on all
# three runners and because a packaging step that can only be exercised
# by pushing is one that gets debugged six minutes at a time.
#
# **Not CPack**, which was the plan and is worth saying why it was not
# done. CPack packages what `install()` rules install, so adopting it
# means teaching CMake to install Qt -- which on two of three platforms
# means capturing the output of windeployqt/macdeployqt in install rules
# and on the third means reimplementing them. That is a rewrite of the
# part that already works, to gain generator plumbing for containers that
# are three lines of hdiutil, appimagetool and makensis respectively. The
# one thing CPack would genuinely buy is component installs, which this
# application does not have.
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

ROOT="$(realpath "$(dirname "$0")/..")"

# LICENSE and NOTICE travel with every package. NOTICE is the one that
# matters here rather than a formality: it is where the app icon is
# recorded as licensed artwork that the project's own license does not
# cover, so a package that omits it is a package making a claim about the
# icon that is not true.
copy_legal() {
    mkdir -p "$1"
    cp "$ROOT/LICENSE" "$ROOT/NOTICE" "$1/"
}

case "$(uname -s)" in
# ---------------------------------------------------------------- Windows
MINGW*|MSYS*|CYGWIN*)
    app="$STAGE_DIR/sstvae"
    mkdir -p "$app"
    cp "$BUILD_DIR/sstvae-gui.exe" "$app/"
    cp "$BUILD_DIR/sstvae-decode.exe" "$app/" 2>/dev/null || true
    cp "$BUILD_DIR/sstvae-audio-check.exe" "$app/" 2>/dev/null || true
    # Ours first, so windeployqt sees a complete executable and does not
    # report an import it cannot resolve.
    [ -n "$HAMLIB_RUNTIME_DIR" ] && cp "$HAMLIB_RUNTIME_DIR"/*.dll "$app/"
    [ -n "$ORT_LIBDIR" ] && cp "$ORT_LIBDIR/onnxruntime.dll" "$app/"
    windeployqt --release --no-translations --no-system-d3d-compiler \
        --no-opengl-sw "$app/sstvae-gui.exe"
    copy_legal "$app"
    ;;

# ------------------------------------------------------------------ macOS
Darwin)
    # CMake built it as a bundle already (MACOSX_BUNDLE), so the layout
    # exists; what it does not have is anything it links from
    # native/.deps, whose absolute paths mean nothing on another Mac.
    #
    # Renamed on the way in. The bundle *directory*'s name is what Finder
    # draws under the icon and what ends up in /Applications, and
    # `sstvae-gui.app` reads like a build artifact; CFBundleExecutable
    # still names the binary inside, which is unchanged, so the rename is
    # only the user-visible half.
    cp -R "$BUILD_DIR/sstvae-gui.app" "$STAGE_DIR/SSTVAE.app"
    app="$STAGE_DIR/SSTVAE.app"
    # Inside the bundle's MacOS directory, so they share the app's
    # rpaths and the Frameworks macdeployqt is about to populate.
    # Shipped because "the waterfall is black" has several causes and
    # this is what tells them apart on a machine we cannot log in to.
    cp "$BUILD_DIR/sstvae-audio-check" "$app/Contents/MacOS/" 2>/dev/null || true
    cp "$BUILD_DIR/sstvae-decode" "$app/Contents/MacOS/" 2>/dev/null || true
    mkdir -p "$app/Contents/Frameworks"
    [ -n "$HAMLIB_RUNTIME_DIR" ] && cp "$HAMLIB_RUNTIME_DIR"/libhamlib*.dylib \
        "$app/Contents/Frameworks/" 2>/dev/null || true
    [ -n "$ORT_LIBDIR" ] && cp "$ORT_LIBDIR"/libonnxruntime*.dylib \
        "$app/Contents/Frameworks/" 2>/dev/null || true
    copy_legal "$app/Contents/Resources"
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

    # The freedesktop trio, in the layout an AppDir and a system prefix
    # both use. Linux is the one platform where the icon is *not* attached
    # to the binary -- the desktop reads the .desktop file and looks the
    # icon up by name in the theme -- so shipping these is the only way
    # the app has an icon at all here.
    pkg="$(realpath "$(dirname "$0")/../native/packaging")"
    mkdir -p "$app/share/applications" "$app/share/metainfo"
    cp "$pkg/org.cleverdomain.sstvae.desktop" "$app/share/applications/"
    cp "$pkg/org.cleverdomain.sstvae.metainfo.xml" "$app/share/metainfo/"
    copy_legal "$app/share/doc/sstvae"
    for png in "$pkg"/icons/sstvae-*.png; do
        size="${png##*-}"; size="${size%.png}"
        dir="$app/share/icons/hicolor/${size}x${size}/apps"
        mkdir -p "$dir"
        cp "$png" "$dir/org.cleverdomain.sstvae.png"
    done
    # The scalable one too: it is what a HiDPI panel prefers, and it is
    # the source the rest were rasterized from.
    mkdir -p "$app/share/icons/hicolor/scalable/apps"
    cp "$pkg/sstvae.svg" \
       "$app/share/icons/hicolor/scalable/apps/org.cleverdomain.sstvae.svg"

    cp "$BUILD_DIR/sstvae-gui" "$app/bin/"
    cp "$BUILD_DIR/sstvae-decode" "$app/bin/" 2>/dev/null || true
    cp "$BUILD_DIR/sstvae-audio-check" "$app/bin/" 2>/dev/null || true
    [ -n "$HAMLIB_RUNTIME_DIR" ] && cp -P "$HAMLIB_RUNTIME_DIR"/libhamlib.so* "$app/lib/"
    [ -n "$ORT_LIBDIR" ] && cp -P "$ORT_LIBDIR"/libonnxruntime.so* "$app/lib/"

    # Qt: seeded from what the executable links, then completed by
    # following the *plugins*. Nothing links a plugin, so an ldd of the
    # binary cannot see a single one of a plugin's dependencies -- which
    # is exactly how the xcb platform plugin shipped without
    # libQt6XcbQpa and then reported itself "found" but unloadable.
    qt_lib_dir="$(ldd "$BUILD_DIR/sstvae-gui" | sed -n 's|.*=> \(.*/libQt6Core\.so[^ ]*\).*|\1|p' \
                  | head -1 | xargs -r dirname)"
    if [ -n "$qt_lib_dir" ]; then
        ldd "$BUILD_DIR/sstvae-gui" \
            | sed -n 's|.*=> \([^ ]*libQt6[^ ]*\) .*|\1|p' \
            | while read -r lib; do cp -P "$lib"* "$app/lib/" 2>/dev/null || true; done

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
        for kind in platforms xcbglintegrations imageformats styles \
                    multimedia tls iconengines platformthemes wayland-shell-integration \
                    wayland-decoration-client wayland-graphics-integration-client; do
            [ -d "$plugins/$kind" ] && cp -R "$plugins/$kind" "$app/plugins/"
        done
        # A bundle with no platform plugin cannot start, so refuse to
        # produce one and call it a package.
        if [ ! -d "$app/plugins/platforms" ]; then
            echo "package_app: no Qt platform plugin found (looked in '${plugins:-nowhere}')" >&2
            exit 1
        fi
    fi

    # Which libraries we take responsibility for.
    #
    # The rule is "whatever Qt itself shipped": if a dependency resolves
    # inside Qt's own lib directory it is part of the Qt we built
    # against, and the target has no reason to have that exact build.
    # That is what picks up the media backend's FFmpeg -- libraries with
    # no `Qt` in the name at all, which a name-based allowlist missed,
    # leaving the plugin unloadable and QtMultimedia reporting "no
    # backends found".
    #
    # Plus the libxcb-* leaf helpers, which Qt's xcb platform plugin
    # needs and a minimal desktop very often lacks. xcb-cursor is the one
    # Qt's own error names, and being newer it is missing most often.
    #
    # Deliberately *not* libxcb.so.1, libX11, libwayland-*, GTK, GStreamer
    # or anything GL: those talk to the running display server, the
    # graphics driver or the desktop, and a bundled copy that disagrees
    # with the host is worse than not bundling at all.
    #
    # The Qt-tree rule only holds when Qt lives in its own prefix. A
    # distro Qt has its libraries in /usr/lib beside libc and libGL, and
    # "everything in Qt's lib dir" would then mean the entire system.
    case "$qt_lib_dir" in
        /usr/lib|/usr/lib64|/lib|/lib64|/usr/lib/*-linux-gnu) qt_own_prefix=0 ;;
        "") qt_own_prefix=0 ;;
        *) qt_own_prefix=1 ;;
    esac

    bundle_worthy() {
        # $1 = basename, $2 = full path
        case "$1" in
            libxcb.so.*) return 1 ;;
            libxcb-*)    return 0 ;;
        esac
        if [ "$qt_own_prefix" -eq 1 ]; then
            case "$2" in "$qt_lib_dir"/*) return 0 ;; esac
            return 1
        fi
        # Distro Qt: name-matched instead, including the media backend's
        # codec libraries, which are the ones the tree rule exists for.
        #
        # The image codecs are here for the same reason: the app loads
        # pictures through Qt's decoders, and the tiff and webp plugins
        # are what make that more than PNG and JPEG. Qt's own binary
        # builds compile those libraries into the plugin, so on the tree
        # rule above there is nothing to copy; a distro's plugins link
        # them, and a missing one is a plugin that silently fails to
        # load -- no error, just a format the file dialog offers and the
        # loader refuses. Leaf codecs with no display, driver or desktop
        # coupling, which is the line the paragraph above draws.
        case "$1" in
            libQt6*|libicu*|libav*|libsw*) return 0 ;;
            libwebp*|libsharpyuv*|libtiff*|libjpeg*|libjbig*|libmng*) return 0 ;;
            *) return 1 ;;
        esac
    }

    # A few rounds, because the answer is transitive: the xcb plugin
    # needs libQt6XcbQpa, which needs more Qt than the executable does.
    deps="$(mktemp)"
    for _round in 1 2 3 4 5; do
        find "$app/bin" "$app/lib" "$app/plugins" -type f \
             \( -name '*.so*' -o -perm -u+x \) -print0 2>/dev/null \
            | xargs -0 -r -n1 ldd 2>/dev/null \
            | sed -n 's|.*=> \(/[^ ]*\) .*|\1|p' | sort -u > "$deps"
        while IFS= read -r dep; do
            base="$(basename "$dep")"
            bundle_worthy "$base" "$dep" || continue
            [ -e "$app/lib/$base" ] && continue
            # The glob matters: `cp -P` on its own copies the *symlink*
            # (libFoo.so.6) and not the file it names (libFoo.so.6.11.1),
            # leaving a dangling link -- which `[ -e ]` then reports as
            # absent, so it is copied again every round and is still
            # broken at the end. Taking the whole family copies the link
            # and its target together.
            cp -P "$dep"* "$app/lib/" 2>/dev/null || true
        done < "$deps"
    done
    rm -f "$deps"

    # Report what is still unresolved; delete only a platform plugin's
    # worth of trouble.
    #
    # **Do not drop a plugin merely because this machine cannot satisfy
    # it.** Qt skips an unloadable optional plugin by itself, and the
    # host is a different computer: `libqgtk3.so` needs GTK, which we
    # deliberately do not bundle and which most desktops have -- an
    # earlier version of this script deleted it because the CI runner
    # did not, and the result was an app with none of the native file
    # dialogs the user's own machine could have given it.
    #
    # A *platform* plugin is different, because the app cannot start
    # without one, so an unresolved xcb or wayland plugin is fatal here
    # rather than a surprise on someone else's desktop.
    unresolved() {
        LD_LIBRARY_PATH="$app/lib" ldd "$1" 2>/dev/null \
            | sed -n 's|^[[:space:]]*\([^ ]*\) => not found.*|\1|p'
    }

    fatal=0
    for obj in "$app"/plugins/*/*.so; do
        [ -e "$obj" ] || continue
        libs="$(unresolved "$obj" | tr '\n' ' ')"
        [ -z "$libs" ] && continue
        case "$obj" in
            */platforms/*)
                echo "package_app: platform plugin $(basename "$obj") needs $libs" >&2
                fatal=1
                ;;
            *)
                echo "package_app: note: $(basename "$obj") wants $libs" \
                     "(kept; the target may have it)" >&2
                ;;
        esac
    done
    libs="$(unresolved "$app/bin/sstvae-gui" | tr '\n' ' ')"
    if [ -n "$libs" ]; then
        echo "package_app: sstvae-gui needs $libs" >&2
        fatal=1
    fi
    [ "$fatal" -eq 0 ] || exit 1

    # A wrapper, because the alternative is asking the user to set two
    # environment variables correctly before the app will start.
    cat > "$app/sstvae-gui" <<'LAUNCH'
#!/bin/sh
# Launcher: points the dynamic loader and Qt at the bundled copies.
here="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
export LD_LIBRARY_PATH="$here/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export QT_PLUGIN_PATH="$here/plugins${QT_PLUGIN_PATH:+:$QT_PLUGIN_PATH}"

# Ask for native dialogs through the desktop portal.
#
# Qt chooses a platform theme by detecting the desktop, and on KDE it
# looks for that desktop's own plugin -- which belongs to the
# distribution and cannot be bundled. Finding nothing it falls back to
# the generic theme, and the operator gets Qt's plain file chooser
# instead of the one the rest of their desktop uses. The portal theme
# *is* bundled and works on GNOME and KDE alike, because it asks the
# desktop over D-Bus rather than linking any of it -- but it is never
# selected unless it is named.
#
# Only when the caller has not chosen, and only if the plugin is
# actually here; with no portal running Qt falls back to exactly the
# behaviour we already had.
if [ -z "${QT_QPA_PLATFORMTHEME:-}" ] \
   && [ -e "$here/plugins/platformthemes/libqxdgdesktopportal.so" ]; then
    export QT_QPA_PLATFORMTHEME=xdgdesktopportal
fi
exec "$here/bin/sstvae-gui" "$@"
LAUNCH
    chmod +x "$app/sstvae-gui"
    ;;
esac

echo "package_app: staged into $STAGE_DIR"
find "$STAGE_DIR" -maxdepth 2 -mindepth 1 | sort | head -20

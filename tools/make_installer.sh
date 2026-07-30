#!/usr/bin/env bash
# Turn a staged tree into the platform's own installable container.
#
#   tools/make_installer.sh [staging-dir] [output-basename]
#
# Step 3 of the packaging work, and deliberately a *second* script rather
# than more of package_app.sh. The split is the useful part: staging needs
# nothing but a compiler and Qt, so a developer can build, stage and run
# the result with none of the tooling below installed. Only the person
# producing a download needs hdiutil, appimagetool or makensis.
#
# What each platform gets, and why that one:
#
#   macOS     .dmg. The expected shape: a disk image with the bundle and
#             a symlink to /Applications, so installing is a drag. A .pkg
#             would be an installer for an application that has nothing
#             to install beyond copying itself.
#   Linux     AppImage. The one format that runs on distributions we have
#             not heard of, which matters because the Qt this bundles is
#             newer than several current distributions ship. A .deb and an
#             .rpm would each cover one family and need a build per
#             release; a Flatpak needs a runtime we would not control.
#   Windows   NSIS setup .exe. Start Menu entry, an uninstaller and the
#             "Apps & features" registration -- the three things a zip
#             cannot give. The zip is still published beside it, because
#             the installer needs administrator rights and a portable
#             copy on a stick does not.
#
# Nothing here signs anything. That is step 4, and it lands as an extra
# call between staging and packing on macOS and Windows -- notarization in
# particular has to happen on the *finished* container.

set -euo pipefail

STAGE_DIR="${1:-dist}"
OUT_BASE="${2:-sstvae}"
DEPS_DIR="${SSTVAE_DEPS_DIR:-native/.deps}"

if [ ! -d "$STAGE_DIR" ]; then
    echo "make_installer: no staged tree at $STAGE_DIR" \
         "(run tools/package_app.sh first)" >&2
    exit 1
fi

# The version, from the one place it is declared. A packaging script with
# its own copy is a packaging script that ships the wrong number.
VERSION="$(sed -n 's/^project(sstvae_native VERSION \([0-9.]*\).*/\1/p' \
           native/CMakeLists.txt | head -1)"
if [ -z "$VERSION" ]; then
    echo "make_installer: could not read the version from native/CMakeLists.txt" >&2
    exit 1
fi

need() {
    command -v "$1" >/dev/null 2>&1 && return 0
    echo "make_installer: $1 not found -- ${2:-install it and try again}" >&2
    exit 1
}

case "$(uname -s)" in
# ---------------------------------------------------------------- Windows
MINGW*|MSYS*|CYGWIN*)
    # NSIS, pinned by sha256 -- the same treatment appimagetool,
    # onnxruntime and Hamlib get, and for the same reason: a packaging
    # tool that changes underneath us changes what we ship.
    #
    # It is **not** preinstalled on GitHub's windows runners. This script
    # asserted that it was, in a comment and in an error message, and the
    # first CI run on a real runner said otherwise. `choco install nsis`
    # would have worked and is the obvious fix, but it makes the version
    # of the tool a property of a package feed's current contents, which
    # is exactly what pinning exists to prevent.
    nsis_version=3.11
    nsis_sha=c7d27f780ddb6cffb4730138cd1591e841f4b7edb155856901cdf5f214394fa1
    nsis_dir="$DEPS_DIR/nsis-$nsis_version"
    if command -v makensis >/dev/null 2>&1; then
        # A local NSIS install wins, so a developer with one is not made
        # to download a second copy.
        makensis=makensis
    else
        if [ ! -x "$nsis_dir/makensis.exe" ]; then
            mkdir -p "$DEPS_DIR"
            zip="$DEPS_DIR/nsis-$nsis_version.zip"
            url="https://downloads.sourceforge.net/project/nsis/NSIS%203"
            url="$url/$nsis_version/nsis-$nsis_version.zip"
            echo "make_installer: fetching NSIS $nsis_version"
            curl -fsSL -o "$zip.part" "$url"
            echo "$nsis_sha  $zip.part" | sha256sum -c - >/dev/null
            mv "$zip.part" "$zip"
            # PowerShell, because neither of the obvious two works here:
            # Git Bash has no `unzip`, and its `tar` is GNU tar (msys2),
            # not the bsdtar that Windows itself ships -- so `tar -xf` on
            # a zip fails with "this does not look like a tar archive".
            # Expand-Archive is present on every supported Windows.
            powershell -NoProfile -NonInteractive -Command \
                "Expand-Archive -LiteralPath '$(cygpath -w "$zip")' \
                 -DestinationPath '$(cygpath -w "$DEPS_DIR")' -Force"
            rm -f "$zip"
        fi
        # The zip's top-level makensis.exe, not Bin/makensis.exe: the
        # former is the portable-distribution stub that locates the
        # Include and Stubs directories relative to itself. Called
        # directly, the one in Bin has to guess.
        makensis="$nsis_dir/makensis.exe"
    fi
    # Named in the log on the way past. If a future NSIS breaks the
    # script, the version that did it is already in the failure output
    # rather than a CI round away.
    "$makensis" -VERSION
    out="$OUT_BASE-setup.exe"
    # NSIS wants Windows paths, and its /D defines are not quoted the way
    # a shell would: pass them through cygpath so a path with a drive
    # letter survives.
    src="$(cygpath -w "$(cd "$STAGE_DIR/sstvae" && pwd)")"
    nsi="$(cygpath -w "$(pwd)/native/packaging/installer.nsi")"
    lic="$(cygpath -w "$(pwd)/LICENSE")"
    "$makensis" -V2 "-DVERSION=$VERSION" "-DSRCDIR=$src" "-DLICENSEFILE=$lic" \
                "-DOUTFILE=$(cygpath -w "$(pwd)/$out")" "$nsi"
    ;;

# ------------------------------------------------------------------ macOS
Darwin)
    need hdiutil
    out="$OUT_BASE.dmg"
    # A staging copy, because the /Applications symlink belongs in the
    # image and not in the tree the caller staged (which is also what
    # gets tarred up as the portable download).
    dmg_root="$(mktemp -d)"
    trap 'rm -rf "$dmg_root"' EXIT
    cp -R "$STAGE_DIR/SSTVAE.app" "$dmg_root/"
    ln -s /Applications "$dmg_root/Applications"
    rm -f "$out"
    # UDZO: compressed and read-only, which is what a download should be.
    # An uncompressed image is roughly three times the size and a
    # read-write one invites the user to modify the app in place.
    hdiutil create -volname "SSTVAE $VERSION" -srcfolder "$dmg_root" \
                   -fs HFS+ -format UDZO -ov "$out" >/dev/null
    ;;

# ------------------------------------------------------------------ Linux
*)
    out="$OUT_BASE.AppImage"
    case "$(uname -m)" in
        x86_64)          arch=x86_64 ;;
        aarch64|arm64)   arch=aarch64 ;;
        *) echo "make_installer: no appimagetool pinned for $(uname -m)" >&2
           exit 1 ;;
    esac

    # Pinned by sha256, the same shape as onnxruntime and Hamlib: a
    # packaging tool that changes underneath us changes what we ship.
    ait_version=1.9.1
    case "$arch" in
        x86_64)  ait_sha=ed4ce84f0d9caff66f50bcca6ff6f35aae54ce8135408b3fa33abfc3cb384eb0 ;;
        aarch64) ait_sha=f0837e7448a0c1e4e650a93bb3e85802546e60654ef287576f46c71c126a9158 ;;
    esac
    ait="$DEPS_DIR/appimagetool-$ait_version-$arch"
    if [ ! -x "$ait" ]; then
        mkdir -p "$DEPS_DIR"
        url="https://github.com/AppImage/appimagetool/releases/download"
        url="$url/$ait_version/appimagetool-$arch.AppImage"
        echo "make_installer: fetching appimagetool $ait_version ($arch)"
        # Download to .part and rename only after the hash matches, so a
        # truncated file cannot be found and used on the next run.
        curl -fsSL -o "$ait.part" "$url"
        echo "$ait_sha  $ait.part" | sha256sum -c - >/dev/null
        chmod +x "$ait.part"
        mv "$ait.part" "$ait"
    fi

    # An AppDir is the staged tree with three additions at its root: a
    # launcher called AppRun, the .desktop file, and the icon named by
    # that file's Icon= key.
    appdir="$(mktemp -d)/SSTVAE.AppDir"
    trap 'rm -rf "$(dirname "$appdir")"' EXIT
    mkdir -p "$appdir/usr"
    cp -a "$STAGE_DIR/sstvae/bin" "$STAGE_DIR/sstvae/lib" \
          "$STAGE_DIR/sstvae/plugins" "$STAGE_DIR/sstvae/share" "$appdir/usr/"

    # The staged launcher already resolves everything relative to itself,
    # so it works unmodified as AppRun -- given the three symlinks that
    # make the AppDir root look like the staged directory it expects.
    # Reusing it rather than writing a second launcher is deliberate: two
    # launchers would mean the AppImage and the tarball could differ in
    # how they set up Qt, and only one of them gets tested.
    cp "$STAGE_DIR/sstvae/sstvae-gui" "$appdir/AppRun"
    chmod +x "$appdir/AppRun"
    ln -s usr/bin "$appdir/bin"
    ln -s usr/lib "$appdir/lib"
    ln -s usr/plugins "$appdir/plugins"

    cp "$appdir/usr/share/applications/org.cleverdomain.sstvae.desktop" "$appdir/"
    cp "$appdir/usr/share/icons/hicolor/256x256/apps/org.cleverdomain.sstvae.png" \
       "$appdir/org.cleverdomain.sstvae.png"
    # .DirIcon is what a file manager reads to draw the AppImage file
    # itself, as opposed to the launcher it installs.
    cp "$appdir/org.cleverdomain.sstvae.png" "$appdir/.DirIcon"

    # Validate the two metadata files here rather than leaving it to
    # appimagetool's --appstream pass, which needs appstreamcli present
    # and *fails the build* when it is not. Doing it ourselves means the
    # check runs wherever the tools exist and is skipped, loudly, where
    # they do not -- rather than making a packaging run depend on which
    # validator the runner image happens to ship.
    if command -v desktop-file-validate >/dev/null 2>&1; then
        desktop-file-validate "$appdir/org.cleverdomain.sstvae.desktop"
    else
        echo "make_installer: note: desktop-file-validate absent, not checked" >&2
    fi
    if command -v appstreamcli >/dev/null 2>&1; then
        appstreamcli validate --no-net \
            "$appdir/usr/share/metainfo/org.cleverdomain.sstvae.metainfo.xml"
    else
        echo "make_installer: note: appstreamcli absent, metainfo not checked" >&2
    fi

    rm -f "$out"
    # APPIMAGE_EXTRACT_AND_RUN because appimagetool is itself an AppImage
    # and CI runners have no FUSE; without it the tool cannot mount
    # *itself* and exits before doing anything.
    APPIMAGE_EXTRACT_AND_RUN=1 ARCH="$arch" "$ait" \
        --no-appstream "$appdir" "$out"
    ;;
esac

echo "make_installer: wrote $out"
ls -l "$out"

# Hand the name back to CI rather than making the workflow reconstruct
# it. The suffix is this script's business -- a second copy of "dmg on
# macOS, AppImage on Linux, -setup.exe on Windows" in a YAML file is a
# copy that goes stale the first time one of them changes.
if [ -n "${GITHUB_ENV:-}" ]; then
    echo "installer=$out" >> "$GITHUB_ENV"
fi

#!/usr/bin/env bash
# Sign what tools/package_app.sh staged, and what tools/make_installer.sh
# wrapped it in.
#
#   tools/sign.sh app        <staging-dir>
#   tools/sign.sh installer  <installer-file>
#
# Step 4 of the packaging work, and the reason it is two modes of one
# script rather than a third script is that the credential handling is
# the whole job. Both modes need the same keychain on macOS and the same
# Azure service principal on Windows; splitting them would mean importing
# a certificate twice and having two places where that can be got wrong.
#
# The two modes exist because the platforms disagree about *when*
# signing happens:
#
#   macOS    the bundle is signed inside-out before it goes into the
#            .dmg, and then the finished .dmg is signed, notarized and
#            stapled. Notarization has to see the container a user will
#            actually download, so it cannot happen any earlier.
#   Windows  our executables are signed inside the staged tree, and the
#            NSIS setup .exe is signed after makensis has built it. A
#            setup signed before it packs unsigned payloads is a setup
#            that SmartScreen trusts and whose contents nobody vouched
#            for.
#   Linux    nothing. An AppImage can carry a detached GPG signature and
#            essentially no desktop verifies one, so it would be effort
#            spent on a check that never runs. AppImages are trusted by
#            their download source, which is why the release page's
#            checksums are the thing that matters there.
#
# **With no credentials this script does nothing and says so, exiting 0.**
# CI builds installers on every push, including from a fork whose runs
# cannot see repository secrets, and an unsigned installer is still worth
# having -- it is what the whole of steps 1-3 produced. But a signing
# step that quietly does nothing is the same hazard the codec tests have
# (see SSTVAE_REQUIRE_CODEC): the strongest thing in the pipeline is also
# the one with an external prerequisite, which is exactly the combination
# that rots into a green tick over nothing. So **SSTVAE_REQUIRE_SIGNING=1
# turns the skip into a failure**, and CI sets it on pushes to this
# repository, where the secrets are known to be present.

set -euo pipefail

MODE="${1:-}"
TARGET="${2:-}"
DEPS_DIR="${SSTVAE_DEPS_DIR:-native/.deps}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

case "$MODE" in
    app|installer) ;;
    *) echo "usage: tools/sign.sh <app|installer> <path>" >&2; exit 2 ;;
esac
if [ -z "$TARGET" ] || [ ! -e "$TARGET" ]; then
    echo "sign: no such $MODE to sign: '${TARGET:-<none>}'" >&2
    exit 1
fi

# Skip, or refuse to skip. Called with a one-line reason naming the
# credential that was missing -- "no signing credentials" sends the
# reader to look at all six.
skip() {
    if [ "${SSTVAE_REQUIRE_SIGNING:-}" = "1" ]; then
        echo "sign: $1, and SSTVAE_REQUIRE_SIGNING=1" >&2
        exit 1
    fi
    echo "sign: $1 -- leaving $TARGET unsigned" >&2
    exit 0
}

# All of these, or none of them: a half-configured signing setup should
# fail at the first step rather than after a certificate import.
require_env() {
    local missing=""
    for v in "$@"; do
        [ -n "${!v:-}" ] || missing="$missing $v"
    done
    [ -z "$missing" ] || skip "unset:$missing"
}

case "$(uname -s)" in

# ================================================================= macOS
Darwin)
require_env BUILD_CERTIFICATE_BASE64 P12_PASSWORD KEYCHAIN_PASSWORD

# --- the keychain ----------------------------------------------------
#
# A throwaway keychain rather than the login one, because a CI runner has
# no login keychain unlocked and because this must not leave a private
# key anywhere that outlives the job. Created once and reused by the
# second invocation: the path is derived, not passed, so the two modes
# cannot disagree about which keychain holds the identity.
KEYCHAIN="${RUNNER_TEMP:-${TMPDIR:-/tmp}}/sstvae-signing.keychain-db"

if [ ! -f "$KEYCHAIN" ]; then
    echo "sign: creating $KEYCHAIN"
    security create-keychain -p "$KEYCHAIN_PASSWORD" "$KEYCHAIN"
    # 6 hours, and -u so it does not lock on sleep. The default is to
    # lock after 5 minutes of inactivity, and `notarytool --wait` is
    # comfortably longer than that -- so the app signs, the wait
    # succeeds, and stapling then fails on a locked keychain twenty
    # minutes after the thing that caused it.
    security set-keychain-settings -lut 21600 "$KEYCHAIN"
    security unlock-keychain -p "$KEYCHAIN_PASSWORD" "$KEYCHAIN"

    # **The file needs a .p12 name and `-f pkcs12`, and neither is
    # optional.** `security import` sniffs the format from the file's
    # extension when it is not told one, so a `mktemp` file -- which has
    # no extension at all -- fails with `SecKeychainItemImport: Unknown
    # format in import.` That message names neither the file nor the
    # format it guessed, so it reads exactly like a corrupt secret or a
    # wrong password, which is where a CI round gets spent. Belt and
    # braces here: the extension is what the tool looks at, `-f` is what
    # makes it not have to.
    p12dir="$(mktemp -d)"
    p12="$p12dir/certificate.p12"
    trap 'rm -rf "$p12dir"' EXIT
    # `tr -d` because a secret pasted from `base64 cert.p12` without
    # `-w0` carries line breaks, and `printf` rather than `echo` because
    # a leading `-` in the payload would be eaten as an option.
    printf '%s' "$BUILD_CERTIFICATE_BASE64" | tr -d '[:space:]' \
        | base64 --decode > "$p12" 2>/dev/null || true

    # Say which of the three things went wrong, rather than leaving all
    # three to one opaque message from `security`. A PKCS#12 file is DER:
    # it begins with a SEQUENCE, 0x30, and a long-form length byte, 0x82.
    # If that holds, the bytes are a certificate bundle and any later
    # failure is about the *password* -- which is a different secret to
    # go and look at.
    if [ ! -s "$p12" ]; then
        echo "sign: BUILD_CERTIFICATE_BASE64 did not decode to anything." >&2
        echo "sign: it should be the output of: base64 -i Certificates.p12" >&2
        exit 1
    fi
    magic="$(od -An -tx1 -N2 "$p12" | tr -d ' ')"
    if [ "$magic" != "3082" ]; then
        echo "sign: the decoded BUILD_CERTIFICATE_BASE64 is not a PKCS#12 file" >&2
        echo "sign: (first two bytes are 0x$magic, expected 0x3082)." >&2
        echo "sign: the secret is probably the base64 of something else," >&2
        echo "sign: or was double-encoded." >&2
        exit 1
    fi

    security import "$p12" -k "$KEYCHAIN" -P "$P12_PASSWORD" -f pkcs12 \
        -T /usr/bin/codesign -T /usr/bin/security
    rm -rf "$p12dir"
    rm -f "$p12"

    # **Without this, codesign blocks forever.** `security import -T`
    # names the tools allowed to use the key, but the key's partition
    # list still defaults to asking the user for confirmation -- and on
    # a runner there is nobody to click Allow, so codesign sits at a
    # dialog that is not drawn on any screen. It is indistinguishable
    # from a hang, and the job dies at its timeout with no output.
    security set-key-partition-list -S apple-tool:,apple:,codesign: \
        -s -k "$KEYCHAIN_PASSWORD" "$KEYCHAIN" >/dev/null

    # Put it on the search list *without* dropping the system roots:
    # `list-keychains -s` replaces the list outright, so naming only ours
    # would remove the Apple intermediates and every signature would fail
    # to build a chain.
    others="$(security list-keychains -d user | tr -d '"' | tr -s ' ')"
    # shellcheck disable=SC2086
    security list-keychains -d user -s "$KEYCHAIN" $others
fi
security unlock-keychain -p "$KEYCHAIN_PASSWORD" "$KEYCHAIN"

# INSTALLER_CERTIFICATE_BASE64 is deliberately not imported. It is a
# Developer ID *Installer* certificate, which signs a .pkg; this project
# ships a .dmg, which is signed with the Application certificate like any
# other bundle. Importing it would put an unused private key on the
# runner and leave the next reader looking for the productsign call that
# uses it.

# --- the identity, and the team ID that comes with it ----------------
#
# Read out of the certificate rather than configured. `find-identity`
# prints `Developer ID Application: Name (TEAMID)`, and TEAMID is exactly
# what notarytool wants -- so there is no seventh secret to keep in step
# with the six, and no way for a certificate and a team ID to name
# different accounts.
IDENTITY="$(security find-identity -v -p codesigning "$KEYCHAIN" \
            | sed -n 's/.*"\(Developer ID Application: [^"]*\)".*/\1/p' | head -1)"
if [ -z "$IDENTITY" ]; then
    echo "sign: no Developer ID Application identity in the imported certificate." >&2
    echo "sign: identities found were:" >&2
    security find-identity -v -p codesigning "$KEYCHAIN" >&2
    exit 1
fi
TEAM_ID="$(printf '%s' "$IDENTITY" | sed -n 's/.*(\([A-Z0-9]*\))$/\1/p')"
[ -n "$TEAM_ID" ] || { echo "sign: no team ID in identity '$IDENTITY'" >&2; exit 1; }
echo "sign: identity '$IDENTITY', team $TEAM_ID"

ENTITLEMENTS="$ROOT/native/packaging/sstvae.entitlements"

# --timestamp because a signature without one dies when the certificate
# expires, rather than remaining valid for what it signed at the time --
# and notarization rejects an un-timestamped signature outright.
# --options runtime is the hardened runtime, which notarization requires;
# it is what makes the entitlements file mean anything.
codesign_it() {  # $1 = path, rest = extra flags
    local path="$1"; shift
    codesign --force --timestamp --options runtime \
             --sign "$IDENTITY" --keychain "$KEYCHAIN" "$@" "$path"
}

if [ "$MODE" = app ]; then
    # Either the bundle or the staging directory holding it. Tested for
    # by *name*, not by `[ -d ]`: a staging directory is a directory too,
    # so the obvious "is it already the bundle?" check accepts `dist`
    # itself and then signs a folder with no Contents in it -- which
    # codesign does without complaint, producing a package whose app is
    # untouched and whose staging directory carries a stray signature.
    case "$TARGET" in
        *.app|*.app/) app="${TARGET%/}" ;;
        *)            app="${TARGET%/}/SSTVAE.app" ;;
    esac
    [ -d "$app/Contents" ] || { echo "sign: no app bundle at $app" >&2; exit 1; }

    # Inside-out, by hand. `codesign --deep` would do this in one call
    # and is deprecated; more to the point it signs nested code with the
    # *outer* target's flags, so a framework would get the app's
    # entitlements. Signing each piece with what that piece should have
    # is the only way the entitlement list above describes anything.

    # 1. Loose dylibs -- onnxruntime, libhamlib -- and every Qt plugin
    #    macdeployqt put under PlugIns. Not inside a .framework: those
    #    are sealed by signing the framework, and signing the binary
    #    within one separately invalidates the framework's own seal.
    while IFS= read -r lib; do
        case "$lib" in *.framework/*) continue ;; esac
        echo "sign:   $(basename "$lib")"
        codesign_it "$lib"
    done < <(find "$app/Contents" -type f \( -name '*.dylib' -o -name '*.so' \) | sort -r)

    # 2. Qt's frameworks, at Versions/A rather than at the bundle root.
    #    A framework's signature lives in the versioned directory, and
    #    signing the top-level symlinked path produces the "bundle format
    #    unrecognized, invalid, or unsuitable" error that sends people to
    #    rebuild Qt.
    for fw in "$app"/Contents/Frameworks/*.framework; do
        [ -d "$fw" ] || continue
        echo "sign:   $(basename "$fw")"
        if [ -d "$fw/Versions/A" ]; then
            codesign_it "$fw/Versions/A"
        else
            codesign_it "$fw"
        fi
    done

    # 3. The helper executables beside the main one. They get the
    #    entitlements too: sstvae-audio-check opens a capture stream, and
    #    without audio-input it would run, report a working device and
    #    measure silence -- which is the exact failure it exists to
    #    diagnose.
    for helper in "$app"/Contents/MacOS/*; do
        [ -f "$helper" ] || continue
        case "$(basename "$helper")" in sstvae-gui) continue ;; esac
        echo "sign:   $(basename "$helper")"
        codesign_it "$helper" --entitlements "$ENTITLEMENTS"
    done

    # 4. The bundle itself, last, which seals everything above.
    echo "sign:   SSTVAE.app"
    codesign_it "$app" --entitlements "$ENTITLEMENTS"

    # --deep is deprecated for *signing* and is exactly right for
    # verifying: it is the recursive check the notary service will do.
    codesign --verify --deep --strict --verbose=2 "$app"

    # And assert the hardened runtime is actually on. A bundle signed
    # without it verifies fine here and is rejected by the notary service
    # ten minutes later, with an error naming a nested binary rather than
    # the missing flag.
    if ! codesign --display --verbose=2 "$app" 2>&1 | grep -q 'flags=.*runtime'; then
        echo "sign: SSTVAE.app is signed without the hardened runtime" >&2
        exit 1
    fi
    echo "sign: signed $app"

else
    # --- the .dmg: sign, notarize, staple ----------------------------
    require_env APPLE_ID APPLE_PASSWORD
    dmg="$TARGET"

    # No entitlements and no runtime flag on a disk image: it is a
    # container, not code. codesign_it passes --options runtime anyway
    # and it is ignored for this file type; keeping one signing function
    # is worth more than the special case.
    codesign_it "$dmg"

    echo "sign: submitting $dmg to the notary service (this takes minutes)"
    out="$(mktemp)"
    set +e
    xcrun notarytool submit "$dmg" \
        --apple-id "$APPLE_ID" --password "$APPLE_PASSWORD" \
        --team-id "$TEAM_ID" --wait --timeout 30m \
        --output-format json > "$out" 2>&1
    rc=$?
    set -e
    cat "$out"
    status="$(sed -n 's/.*"status":"\([^"]*\)".*/\1/p' "$out" | tail -1)"
    subid="$(sed -n 's/.*"id":"\([^"]*\)".*/\1/p' "$out" | head -1)"

    if [ "$rc" -ne 0 ] || [ "$status" != "Accepted" ]; then
        # **Fetch the log before failing.** A rejection's own output says
        # only "Invalid"; the reasons -- an unsigned nested dylib, a
        # missing timestamp, no hardened runtime -- are in a separate
        # document that is unreachable once the job has ended. Getting it
        # here is the difference between one round and three.
        echo "sign: notarization did not succeed (status='${status:-none}')" >&2
        if [ -n "$subid" ]; then
            echo "sign: notary log for $subid:" >&2
            xcrun notarytool log "$subid" \
                --apple-id "$APPLE_ID" --password "$APPLE_PASSWORD" \
                --team-id "$TEAM_ID" >&2 || true
        fi
        exit 1
    fi

    # Staple the ticket into the image, so Gatekeeper can approve it
    # without asking Apple -- which is what makes the download work for
    # an operator whose shack PC is not on the internet.
    xcrun stapler staple "$dmg"
    # The check a first-launch actually performs. `-t open --context
    # context:primary-signature` is the assessment for a downloaded
    # document rather than for an installed application, which is what a
    # .dmg is at this point.
    spctl --assess -t open --context context:primary-signature -v "$dmg"
    echo "sign: signed, notarized and stapled $dmg"
fi
;;

# =============================================================== Windows
MINGW*|MSYS*|CYGWIN*)
require_env AZURE_ENDPOINT AZURE_CODE_SIGNING_NAME AZURE_CERT_PROFILE_NAME \
            AZURE_TENANT_ID AZURE_CLIENT_ID AZURE_CLIENT_SECRET

# Microsoft's `sign` CLI, pinned by version like NSIS, appimagetool,
# onnxruntime and Hamlib -- and for the stronger version of the same
# reason: this one decides what a signature says.
#
# Installed to a tool path under the dependency directory rather than
# --global, so it lands where the existing native/.deps cache already
# covers and the script calls it by an absolute path instead of hoping
# for a PATH the shell has picked up mid-job.
#
# The credentials are not passed as arguments. The tool authenticates
# with Azure's DefaultAzureCredential, which reads AZURE_TENANT_ID,
# AZURE_CLIENT_ID and AZURE_CLIENT_SECRET from the environment -- which
# is why those three secrets have no corresponding flag below.
SIGN_VERSION="${SSTVAE_SIGN_CLI_VERSION:-0.9.1-beta.25278.1}"
SIGN_DIR="$DEPS_DIR/sign-$SIGN_VERSION"
if [ ! -x "$SIGN_DIR/sign.exe" ]; then
    echo "sign: installing the sign CLI $SIGN_VERSION"
    dotnet tool install sign --version "$SIGN_VERSION" --tool-path "$SIGN_DIR"
fi
SIGN="$SIGN_DIR/sign.exe"

# Named in the log on the way past, like makensis -VERSION: if a future
# release breaks this call, the version that did it is already in the
# failure output rather than a CI round away.
"$SIGN" --version

# The three Trusted Signing options are all spelled `trusted-signing-*`,
# including the certificate profile. `--certificate-profile` is what the
# option is *called* in the Azure portal and in every other tool's
# configuration, and it is not what this CLI accepts -- it rejected it as
# an unknown option and then reported the real name as missing, which is
# a good error and still cost a CI round. The short forms are `-tse`,
# `-tsa` and `-tscp`; the long ones are used here because a signing
# command is read far more often than it is typed.
#
# Not passed, because the defaults are already right: `--timestamp-url`
# defaults to Trusted Signing's own RFC 3161 server, and `--file-digest`
# to SHA256.
sign_files() {  # $@ = Windows-relative paths under $dir, dir in $1
    local dir="$1"; shift
    "$SIGN" code trusted-signing \
        --trusted-signing-endpoint "$AZURE_ENDPOINT" \
        --trusted-signing-account "$AZURE_CODE_SIGNING_NAME" \
        --trusted-signing-certificate-profile "$AZURE_CERT_PROFILE_NAME" \
        --description "SSTVAE" \
        --description-url "https://github.com/arodland/SSTVAE" \
        --base-directory "$(cygpath -w "$dir")" \
        --verbosity Information \
        "$@"
}

if [ "$MODE" = app ]; then
    dir="$TARGET/sstvae"
    [ -d "$dir" ] || { echo "sign: no staged tree at $dir" >&2; exit 1; }
    dir="$(cd "$dir" && pwd)"

    # **Only our own executables.** The temptation is `**/*.dll`, and it
    # is wrong twice. Trusted Signing bills against a monthly signature
    # quota and a staged tree is forty-odd Qt DLLs, on every push; and
    # signing somebody else's library with our certificate makes a claim
    # about code we did not build. Windows has no notarization pass that
    # would look inside, and SmartScreen's reputation attaches to the
    # executable the user launches -- which is this list.
    found=""
    for exe in sstvae-gui.exe sstvae-decode.exe sstvae-audio-check.exe; do
        [ -f "$dir/$exe" ] && found="$found $exe"
    done
    [ -n "$found" ] || { echo "sign: no executables found in $dir" >&2; exit 1; }
    # shellcheck disable=SC2086
    sign_files "$dir" $found

    # Not signed, and worth saying why rather than leaving it to be
    # noticed: NSIS generates uninstall.exe at *install* time by
    # extracting it from the setup executable, so signing it needs the
    # two-pass trick -- build the installer, run it to shed the
    # uninstaller, sign that, build the installer again around it. Two
    # builds and a signature for a binary Windows never shows a
    # SmartScreen prompt for, because it is launched by the installed
    # application's own uninstall entry.
    echo "sign: signed$found in $dir"
else
    dir="$(cd "$(dirname "$TARGET")" && pwd)"
    sign_files "$dir" "$(basename "$TARGET")"
    echo "sign: signed $TARGET"
fi
;;

# ================================================================= Linux
*)
    echo "sign: nothing to sign on Linux -- see the comment at the top of $0"
    ;;
esac

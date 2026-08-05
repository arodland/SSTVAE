#!/usr/bin/env bash
# Set the application version everywhere it is written down.
#
#   tools/bump_version.sh <x.y.z> [--dry-run] [--force]
#
# The version lives in three files and `.github/workflows/release.yml`
# checks two of them against the tag before it will build anything, so a
# release with a mismatch fails early rather than shipping a download
# called 0.2.0 whose About box says 0.1.0. That gate is the reason this
# script exists: it turns "remember to edit three files" into one
# command, and the alternative -- a single source of truth generated
# into the other two -- was not taken, because CMake and PEP 621 both
# want a literal there and the generated-file machinery this project
# already has (`gen_config_header.py`) is for constants a human never
# reads, not for a number a packager greps for.
#
# **The current version is read with the release workflow's own sed
# expressions**, and the result is verified by reading it back the same
# way. A bump that edits a line the gate does not match is exactly as
# broken as no bump at all, and would be found a tag later.
#
# `sstvae/__init__.py` is bumped as well even though nothing checks it.
# That is *why* -- it is the copy free to drift, and it is what
# `sstvae.__version__` reports to anyone debugging a Python-side
# station.
#
# Linux/bash only, by request: it is a developer's pre-release step, not
# something CI or an operator runs.

set -euo pipefail

usage() {
    cat >&2 <<'EOF'
usage: tools/bump_version.sh <x.y.z> [--dry-run] [--force]

  --dry-run  report what would change and write nothing
  --force    allow a version that is not greater than the current one,
             and allow a version whose git tag already exists
EOF
    exit 2
}

version=""
dry_run=0
force=0
for arg in "$@"; do
    case "$arg" in
        --dry-run) dry_run=1 ;;
        --force)   force=1 ;;
        -h|--help) usage ;;
        -*)        echo "unknown option: $arg" >&2; usage ;;
        *)
            [ -z "$version" ] || { echo "give exactly one version" >&2; usage; }
            version="$arg"
            ;;
    esac
done
[ -n "$version" ] || usage

# Strict numeric triple, and not merely out of tidiness: NSIS builds
# `VIProductVersion "${VERSION}.0"`, which must be four integers, and the
# release workflow extracts the version with `[0-9.]*` -- so `1.2.0-rc1`
# would pass here, truncate to `1.2.0` at the gate, compare equal, and
# then fail inside makensis at the far end of a five-platform build.
if ! [[ "$version" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    echo "version must be x.y.z with numeric parts (got '$version')" >&2
    exit 1
fi

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo"

cmake_file="native/CMakeLists.txt"
pyproject_file="pyproject.toml"
init_file="sstvae/__init__.py"

for f in "$cmake_file" "$pyproject_file" "$init_file"; do
    [ -f "$f" ] || { echo "missing $f -- is $repo the right tree?" >&2; exit 1; }
done

# The first two expressions are copied verbatim from release.yml's check.
read_cmake()     { sed -n 's/^project(sstvae_native VERSION \([0-9.]*\).*/\1/p' "$cmake_file" | head -1; }
read_pyproject() { sed -n 's/^version = "\([0-9.]*\)".*/\1/p' "$pyproject_file" | head -1; }
read_init()      { sed -n 's/^__version__ = "\([0-9.]*\)".*/\1/p' "$init_file" | head -1; }

cmake_now="$(read_cmake)"
pyproject_now="$(read_pyproject)"
init_now="$(read_init)"

for pair in "$cmake_file:$cmake_now" "$pyproject_file:$pyproject_now" "$init_file:$init_now"; do
    if [ -z "${pair#*:}" ]; then
        echo "could not read a version from ${pair%%:*} -- has the line moved?" >&2
        exit 1
    fi
done

echo "current: $cmake_file=$cmake_now  $pyproject_file=$pyproject_now  $init_file=$init_now"
echo "new:     $version"

if [ "$cmake_now" != "$pyproject_now" ] || [ "$cmake_now" != "$init_now" ]; then
    echo "note: the files disagree today; this run makes them agree." >&2
fi

# Ordering check against the highest of the three, so a half-finished
# previous bump cannot make a real increase look like a decrease. `sort
# -V` rather than a hand-rolled field compare: 0.10.0 > 0.9.0.
highest="$(printf '%s\n' "$cmake_now" "$pyproject_now" "$init_now" | sort -V | tail -1)"
if [ "$version" = "$highest" ]; then
    if [ "$force" -eq 0 ]; then
        echo "already at $version (use --force to rewrite anyway)" >&2
        exit 1
    fi
elif [ "$(printf '%s\n%s\n' "$version" "$highest" | sort -V | head -1)" = "$version" ]; then
    if [ "$force" -eq 0 ]; then
        echo "$version is older than $highest (use --force if that is deliberate)" >&2
        exit 1
    fi
fi

# A tag that already exists is a release that was already built, and the
# workflow will happily rebuild it from the tagged commit -- which is not
# the commit this bump is heading for.
if git rev-parse --verify -q "refs/tags/v$version" >/dev/null 2>&1; then
    if [ "$force" -eq 0 ]; then
        echo "tag v$version already exists (use --force if you mean to reuse it)" >&2
        exit 1
    fi
    echo "note: tag v$version already exists; continuing because --force." >&2
fi

# Each edit is anchored to the one line that the reader above matches,
# and each file must contain exactly one such line: a second copy would
# mean half the file bumped and half not, which is the failure this
# script is supposed to retire.
bump() {
    local file="$1" match="$2" replace="$3" n
    n="$(grep -cE -- "$match" "$file" || true)"
    if [ "$n" != "1" ]; then
        echo "expected exactly 1 version line in $file, found $n" >&2
        exit 1
    fi
    if [ "$dry_run" -eq 1 ]; then
        printf '  would edit %-24s %s\n' "$file" "$(grep -E -- "$match" "$file")"
        return
    fi
    sed -i -E "s|$match|$replace|" "$file"
}

bump "$cmake_file" \
     '^project\(sstvae_native VERSION [0-9]+\.[0-9]+\.[0-9]+' \
     "project(sstvae_native VERSION $version"
bump "$pyproject_file" \
     '^version = "[0-9]+\.[0-9]+\.[0-9]+"' \
     "version = \"$version\""
bump "$init_file" \
     '^__version__ = "[0-9]+\.[0-9]+\.[0-9]+"' \
     "__version__ = \"$version\""

if [ "$dry_run" -eq 1 ]; then
    echo "dry run: nothing written"
    exit 0
fi

# Read back through the gate's expressions rather than trusting sed's
# exit status, which is 0 whether or not anything matched.
ok=1
for pair in "$cmake_file:$(read_cmake)" \
            "$pyproject_file:$(read_pyproject)" \
            "$init_file:$(read_init)"; do
    got="${pair#*:}"
    if [ "$got" != "$version" ]; then
        echo "${pair%%:*} still reads '$got' after the edit" >&2
        ok=0
    fi
done
[ "$ok" -eq 1 ] || exit 1

echo "bumped to $version in:"
printf '  %s\n' "$cmake_file" "$pyproject_file" "$init_file"
cat <<EOF

Next: review the diff, commit, then tag:

  git diff -- $cmake_file $pyproject_file $init_file
  git tag -a v$version -m 'SSTVAE $version' && git push origin v$version
EOF

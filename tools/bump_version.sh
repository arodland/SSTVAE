#!/usr/bin/env bash
# Set the application version everywhere it is written down, and commit
# it as "Release vX.Y.Z".
#
#   tools/bump_version.sh <x.y.z> [--dry-run] [--force] [--no-commit]
#
# The version is written down in four places -- native/CMakeLists.txt,
# pyproject.toml, sstvae/__init__.py and uv.lock -- and
# `.github/workflows/release.yml` checks the first two against the tag
# before it will build anything, so a release with a mismatch fails
# early rather than shipping a download called 0.2.0 whose About box
# says 0.1.0. That gate is the reason this script exists: it turns
# "remember to edit four files" into one command, and the alternative --
# a single source of truth generated into the rest -- was not taken,
# because CMake and PEP 621 both want a literal there and the
# generated-file machinery this project already has
# (`gen_config_header.py`) is for constants a human never reads, not for
# a number a packager greps for.
#
# **The current version is read with the release workflow's own sed
# expressions**, and the result is verified by reading it back the same
# way. A bump that edits a line the gate does not match is exactly as
# broken as no bump at all, and would be found a tag later.
#
# **The two copies nothing checks are the ones that drift**, which is
# why they are here rather than left to a habit: `sstvae/__init__.py` is
# what `sstvae.__version__` reports to anyone debugging a Python-side
# station, and `uv.lock` records the project's own version in its
# `[[package]]` entry. v0.1.1 shipped with the lock left at 0.1.0.
#
# Linux/bash only, by request: it is a developer's pre-release step, not
# something CI or an operator runs.

set -euo pipefail

usage() {
    cat >&2 <<'EOF'
usage: tools/bump_version.sh <x.y.z> [--dry-run] [--force] [--no-commit]

  --dry-run    report what would change and write nothing
  --force      allow a version that is not greater than the current one,
               and allow a version whose git tag already exists
  --no-commit  edit the files and leave them in the working tree
EOF
    exit 2
}

version=""
dry_run=0
force=0
commit=1
for arg in "$@"; do
    case "$arg" in
        --dry-run)   dry_run=1 ;;
        --force)     force=1 ;;
        --no-commit) commit=0 ;;
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
lock_file="uv.lock"

for f in "$cmake_file" "$pyproject_file" "$init_file" "$lock_file"; do
    [ -f "$f" ] || { echo "missing $f -- is $repo the right tree?" >&2; exit 1; }
done

# --------------------------------------------------------------- git state
#
# Checked before anything is written, so a refusal leaves the tree
# exactly as it was found rather than half-bumped.
#
# **The index must be empty.** The release commit this script makes has
# to contain the version bump and nothing else: a tag is a claim about
# what is in the build, and anything already staged would ride along
# under a message saying "Release", where it is invisible to the person
# reading the tag later and to anyone bisecting. Refusing is also the
# reversible choice -- `git stash`/`git reset` are one command, whereas
# unpicking a commit that has been tagged and pushed is not.
#
# This is a *whole-index* check on purpose, not a check of the three
# files. `git commit` without a pathspec takes everything staged, so
# restricting the check to the version files would let the rest through.
if ! git rev-parse --git-dir >/dev/null 2>&1; then
    echo "not a git repository: $repo" >&2
    exit 1
fi

# Checked here rather than beside the `uv lock` call it guards, because
# that call comes *after* the three text files are written -- so failing
# there would leave a half-bumped tree, which is precisely what doing
# these checks up front is for. A missing `uv` is a hard error and not a
# skip: a bump that quietly left the lock behind is the failure that
# step exists to fix, and it is invisible, since nothing in CI reads
# uv.lock and it surfaces only as the *next* person's `uv run` rewriting
# the file into an unrelated commit.
if [ "$dry_run" -eq 0 ] && ! command -v uv >/dev/null 2>&1; then
    echo "uv is not installed, so $lock_file cannot be regenerated." >&2
    echo "install uv (this project's dev tool) and re-run" >&2
    exit 1
fi
if ! git diff --cached --quiet; then
    echo "there are staged changes:" >&2
    git diff --cached --name-only | sed 's/^/  /' >&2
    echo "commit or unstage them first (a release commit must be the bump alone)" >&2
    exit 1
fi

# The complement, and a narrow one: this tree normally has unrelated
# working-tree edits, which are fine -- they are never staged here. What
# is not fine is an unstaged edit to a file this script is about to
# `git add`, because that edit would be swept into the release commit
# just as invisibly as a staged one.
if [ "$commit" -eq 1 ]; then
    dirty="$(git diff --name-only -- "$cmake_file" "$pyproject_file" "$init_file" "$lock_file")"
    if [ -n "$dirty" ]; then
        echo "uncommitted changes in files this script commits:" >&2
        printf '  %s\n' $dirty >&2
        echo "commit or revert them first, or pass --no-commit" >&2
        exit 1
    fi
fi

# The first two expressions are copied verbatim from release.yml's check.
read_cmake()     { sed -n 's/^project(sstvae_native VERSION \([0-9.]*\).*/\1/p' "$cmake_file" | head -1; }
read_pyproject() { sed -n 's/^version = "\([0-9.]*\)".*/\1/p' "$pyproject_file" | head -1; }
read_init()      { sed -n 's/^__version__ = "\([0-9.]*\)".*/\1/p' "$init_file" | head -1; }

# uv.lock carries the project's own version in its `[[package]]` entry,
# so it needs the *entry's* version and not the file's first `version =`
# line -- there are 117 other packages in there, and matching the wrong
# one would report a dependency's number as the app's.
read_lock() {
    awk '/^name = "sstvae"$/ { f = 1; next }
         f && /^version = "/  { sub(/^version = "/, ""); sub(/".*$/, ""); print; exit }' \
        "$lock_file"
}

cmake_now="$(read_cmake)"
pyproject_now="$(read_pyproject)"
init_now="$(read_init)"
lock_now="$(read_lock)"

for pair in "$cmake_file:$cmake_now" "$pyproject_file:$pyproject_now" \
            "$init_file:$init_now" "$lock_file:$lock_now"; do
    if [ -z "${pair#*:}" ]; then
        echo "could not read a version from ${pair%%:*} -- has the line moved?" >&2
        exit 1
    fi
done

echo "current: $cmake_file=$cmake_now  $pyproject_file=$pyproject_now"
echo "         $init_file=$init_now  $lock_file=$lock_now"
echo "new:     $version"

if [ "$cmake_now" != "$pyproject_now" ] || [ "$cmake_now" != "$init_now" ] \
   || [ "$cmake_now" != "$lock_now" ]; then
    echo "note: the files disagree today; this run makes them agree." >&2
fi

# Ordering check against the highest of the three, so a half-finished
# previous bump cannot make a real increase look like a decrease. `sort
# -V` rather than a hand-rolled field compare: 0.10.0 > 0.9.0.
highest="$(printf '%s\n' "$cmake_now" "$pyproject_now" "$init_now" "$lock_now" \
           | sort -V | tail -1)"
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
    echo "  would run  uv lock                 ($lock_file: $lock_now -> $version)"
    echo "dry run: nothing written"
    exit 0
fi

# uv.lock records the project's own version, so it goes stale on every
# bump. **Regenerated with `uv lock`, never sed'd**: the three files
# above are hand-written and their version is a line a human put there,
# whereas this one is a generated artifact whose format is uv's to
# change. Measured on this project it is a 0.26 s resolve from cache
# that touches exactly the one line -- no network and no dependency
# churn, because `uv lock` without `--upgrade` keeps every existing pin
# and only re-resolves what actually changed.
#
uv lock

# Read back through the gate's expressions rather than trusting sed's
# exit status, which is 0 whether or not anything matched.
ok=1
for pair in "$cmake_file:$(read_cmake)" \
            "$pyproject_file:$(read_pyproject)" \
            "$init_file:$(read_init)" \
            "$lock_file:$(read_lock)"; do
    got="${pair#*:}"
    if [ "$got" != "$version" ]; then
        echo "${pair%%:*} still reads '$got' after the edit" >&2
        ok=0
    fi
done
[ "$ok" -eq 1 ] || exit 1

echo "bumped to $version in:"
printf '  %s\n' "$cmake_file" "$pyproject_file" "$init_file" "$lock_file"

if [ "$commit" -eq 0 ]; then
    cat <<EOF

Left in the working tree (--no-commit). Next:

  git diff -- $cmake_file $pyproject_file $init_file $lock_file
EOF
    exit 0
fi

# The index was empty at the top and these files had no unstaged edits,
# so what is staged here is exactly what this script wrote.
git add -- "$cmake_file" "$pyproject_file" "$init_file" "$lock_file"
git commit -q -m "Release v$version"
echo
git --no-pager show --stat --oneline HEAD

cat <<EOF

Next: check that commit, then tag it:

  git tag -a v$version -m 'SSTVAE $version' && git push origin v$version
EOF

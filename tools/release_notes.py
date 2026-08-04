#!/usr/bin/env python3
"""Append a downloads table to a release's notes.

    tools/release_notes.py <release.json> <tag> <owner/repo> > body.md

where release.json is the output of

    gh release view <tag> --json body,assets

Writes the whole new body to stdout: the notes as written by hand, then
a generated section listing every asset on the release.

Three things this does deliberately.

**The table is built from the assets that are actually there**, not from
a filename pattern. ``tools/make_installer.sh`` owns the rule that an
installer is a ``.dmg`` on macOS, an AppImage on Linux and a
``-setup.exe`` on Windows, and its closing comment says so; a second
copy of that rule here is one that goes stale the first time a suffix
changes, and the symptom is a release page full of dead links. Anything
unrecognised still gets listed rather than dropped -- a new platform
should show up as an ugly row, not as a missing one.

**It is idempotent.** Everything from the marker comment onward is
replaced, so re-running the workflow against a release that already has
a table gives the same result rather than a second copy. That matters
because the whole point of the ``workflow_dispatch`` entry point is
re-running a release that half-failed.

**It is a script rather than fifteen lines of shell in the workflow**,
because a release page's download links are seen by more people than
anything else this project produces and there is exactly one chance to
get them right per release. Here they can be tested; see
``tests/test_release_notes.py``.
"""

from __future__ import annotations

import json
import sys

# Everything from here down is generated. Kept as an HTML comment so it
# is invisible on the rendered page but findable in the raw markdown.
MARKER = "<!-- generated-downloads: do not edit below this line -->"

# label fragment -> (sort key, human name). The label is what CI names a
# package after -- the *target*, not the runner image -- so these are the
# same strings that appear in ci.yml's matrix.
PLATFORMS = [
    ("windows-x64", "Windows (64-bit)"),
    ("macos-arm64", "macOS (Apple Silicon)"),
    ("macos-x86_64", "macOS (Intel)"),
    ("linux-x86_64", "Linux (x86-64)"),
    ("linux-aarch64", "Linux (arm64)"),
]

# Which of a platform's two downloads is which. An installer registers
# itself with the desktop; a portable copy unpacks and runs and needs no
# administrator, which is the shape a shack PC often wants.
INSTALLER_SUFFIXES = (".dmg", ".appimage", "-setup.exe")


def classify(name: str) -> str:
    low = name.lower()
    if low.endswith(INSTALLER_SUFFIXES):
        return "installer"
    return "portable"


def platform_of(name: str) -> str | None:
    # Longest fragment first, so `macos-x86_64` is not matched by a
    # hypothetical `macos` entry, and `linux-x86_64` never collides with
    # `linux-aarch64`.
    for frag, human in sorted(PLATFORMS, key=lambda p: -len(p[0])):
        if frag in name:
            return human
    return None


# What to call a download in the link text. The full filename is
# accurate and unreadable -- `sstvae-0.2.0-macos-arm64.dmg` repeated ten
# times makes a table nobody can scan, and the version and platform are
# already the release and the row. So the link says what *kind* of file
# it is, which is the only thing the two cells in a row differ by.
FORMATS = [
    ("-setup.exe", "Setup .exe"),
    ("-portable.zip", ".zip"),
    (".appimage", ".AppImage"),
    (".tar.gz", ".tar.gz"),
    (".dmg", ".dmg"),
    (".zip", ".zip"),
    (".exe", ".exe"),
]


def format_label(name: str) -> str:
    low = name.lower()
    for suffix, label in FORMATS:
        if low.endswith(suffix):
            return label
    return name


def build_table(assets: list[dict], tag: str, repo: str) -> str:
    base = f"https://github.com/{repo}/releases/download/{tag}"

    def link(asset: dict, text: str | None = None) -> str:
        name = asset["name"]
        size = asset.get("size") or 0
        mb = f" ({size / 1_000_000:.0f} MB)" if size else ""
        return f"[{text or format_label(name)}{mb}]({base}/{name})"

    rows: dict[str, dict[str, list[str]]] = {}
    unknown: list[dict] = []
    for a in assets:
        human = platform_of(a["name"])
        if human is None:
            unknown.append(a)
            continue
        rows.setdefault(human, {}).setdefault(classify(a["name"]), []).append(link(a))

    out = [MARKER, "", "## Downloads", ""]
    if rows:
        out += ["| Platform | Installer | Portable |", "|---|---|---|"]
        for _frag, human in PLATFORMS:
            if human not in rows:
                continue
            cell = rows[human]
            inst = "<br>".join(cell.get("installer", [])) or "—"
            port = "<br>".join(cell.get("portable", [])) or "—"
            out.append(f"| {human} | {inst} | {port} |")
        out.append("")

    if unknown:
        # Named in full here, since there is no row to say what they are.
        out += ["Other files:", ""]
        out += [f"- {link(a, a['name'])}" for a in unknown]
        out.append("")

    # The two things an operator hits on first launch, said once here so
    # they are not a support question. macOS is genuinely clean now --
    # the notarization ticket is stapled, so Gatekeeper approves without
    # a network round trip. Windows is not, and saying so plainly is
    # better than letting it look like a broken download.
    out += [
        "The installer is the easy path; the portable archive needs no "
        "administrator and runs from a stick.",
        "",
        "macOS builds are signed with a Developer ID and notarized by "
        "Apple. Windows builds are signed with Azure Trusted Signing — "
        "SmartScreen may still warn on the first downloads until the "
        "certificate earns a reputation; choose **More info** → **Run "
        "anyway**.",
        "",
        "The image model is fetched on first use and cached, so there is "
        "nothing else to download.",
    ]
    return "\n".join(out)


def main() -> int:
    if len(sys.argv) != 4:
        print(__doc__.strip().splitlines()[2].strip(), file=sys.stderr)
        return 2
    # **Force UTF-8 out, whatever the platform thinks.** Markdown for a
    # release page is UTF-8 by definition, but Python picks stdout's
    # encoding from the locale -- which on Windows is cp1252, and cp1252
    # has an em dash but no arrow. So the "More info -> Run anyway" line
    # raised UnicodeEncodeError on one platform out of three, and the
    # script exited 1 having written most of a release note. Setting it
    # here rather than dropping the character: the text is aimed at
    # operators reading a release page, and it should not be rationed to
    # the intersection of every legacy code page.
    sys.stdout.reconfigure(encoding="utf-8", newline="\n")

    payload, tag, repo = sys.argv[1], sys.argv[2], sys.argv[3]
    with open(payload, encoding="utf-8") as fh:
        data = json.load(fh)

    body = (data.get("body") or "").rstrip()
    assets = data.get("assets") or []

    # Drop any previously generated section before appending a new one.
    head = body.split(MARKER)[0].rstrip()

    table = build_table(assets, tag, repo)
    print(f"{head}\n\n{table}" if head else table)
    return 0


if __name__ == "__main__":
    sys.exit(main())

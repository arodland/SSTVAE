"""The release page's download links.

More people see a release page than anything else this project
publishes, and there is one chance per release to get it right -- so the
generator is a script with tests rather than shell inside a workflow,
where the only way to exercise it is to cut a release.

The cases here are the ones that would produce a plausible-looking page
that is wrong: a dead link, a platform silently missing, or a second copy
of the table stacked on the first.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "release_notes.py"
TAG = "v0.2.0"
REPO = "arodland/SSTVAE"

# What CI actually attaches: an installer and a portable copy for each of
# the five targets, named by tools/make_installer.sh's own suffix rule.
ASSETS = [
    "sstvae-0.2.0-linux-x86_64.AppImage",
    "sstvae-0.2.0-linux-x86_64.tar.gz",
    "sstvae-0.2.0-linux-aarch64.AppImage",
    "sstvae-0.2.0-linux-aarch64.tar.gz",
    "sstvae-0.2.0-macos-arm64.dmg",
    "sstvae-0.2.0-macos-arm64.tar.gz",
    "sstvae-0.2.0-macos-x86_64.dmg",
    "sstvae-0.2.0-macos-x86_64.tar.gz",
    "sstvae-0.2.0-windows-x64-setup.exe",
    "sstvae-0.2.0-windows-x64-portable.zip",
]


def run(body: str, names=ASSETS, tmp_path=None) -> str:
    payload = {
        "body": body,
        "assets": [{"name": n, "size": 40_000_000} for n in names],
    }
    p = tmp_path / "release.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    # `check=False` plus an explicit assertion, because `check=True`
    # raises a CalledProcessError whose message is the command line and
    # the exit status and nothing else -- so the Windows-only
    # UnicodeEncodeError this file now guards against showed up in CI as
    # nine identical "returned non-zero exit status 1" and no traceback.
    # encoding= is named for the same reason: without it the decode side
    # follows the platform locale too, which is how a test for an
    # encoding bug acquires one of its own.
    out = subprocess.run(
        [sys.executable, str(SCRIPT), str(p), TAG, REPO],
        capture_output=True, text=True, encoding="utf-8", check=False,
    )
    assert out.returncode == 0, (
        f"release_notes.py exited {out.returncode}\n"
        f"--- stderr ---\n{out.stderr}"
    )
    return out.stdout


def test_hand_written_notes_are_kept_and_come_first(tmp_path):
    out = run("## What's new\n\nBeacon carrier fixes.", tmp_path=tmp_path)
    assert "Beacon carrier fixes." in out
    assert out.index("Beacon carrier fixes.") < out.index("## Downloads")


def test_every_asset_is_linked_and_no_link_is_invented(tmp_path):
    out = run("notes", tmp_path=tmp_path)
    base = f"https://github.com/{REPO}/releases/download/{TAG}"
    for name in ASSETS:
        assert f"({base}/{name})" in out, f"{name} missing from the table"
    # The converse: every link in the output names a real asset. A typo
    # in the URL builder would still produce a page that looks complete.
    import re
    linked = set(re.findall(rf"{re.escape(base)}/([^)]+)\)", out))
    assert linked == set(ASSETS)


def test_all_five_platforms_appear(tmp_path):
    out = run("notes", tmp_path=tmp_path)
    for human in [
        "Windows (64-bit)", "macOS (Apple Silicon)", "macOS (Intel)",
        "Linux (x86-64)", "Linux (arm64)",
    ]:
        assert human in out


def test_intel_and_arm_are_not_confused(tmp_path):
    """`macos-x86_64` must not be swallowed by a shorter `macos` match,
    and linux-x86_64 must not land in the aarch64 row."""
    out = run("notes", tmp_path=tmp_path)
    rows = {line.split("|")[1].strip(): line
            for line in out.splitlines() if line.startswith("| ")}
    assert "macos-x86_64.dmg" in rows["macOS (Intel)"]
    assert "macos-arm64.dmg" in rows["macOS (Apple Silicon)"]
    assert "macos-x86_64" not in rows["macOS (Apple Silicon)"]
    assert "linux-aarch64" not in rows["Linux (x86-64)"]


def test_installer_and_portable_land_in_the_right_columns(tmp_path):
    out = run("notes", tmp_path=tmp_path)
    rows = {line.split("|")[1].strip(): line.split("|")
            for line in out.splitlines() if line.startswith("| ")}
    win = rows["Windows (64-bit)"]
    assert "setup.exe" in win[2] and "portable.zip" in win[3]
    mac = rows["macOS (Apple Silicon)"]
    assert ".dmg" in mac[2] and ".tar.gz" in mac[3]
    lin = rows["Linux (x86-64)"]
    assert ".AppImage" in lin[2] and ".tar.gz" in lin[3]


def test_rerunning_replaces_the_table_rather_than_stacking_it(tmp_path):
    """The workflow_dispatch entry point exists to re-run a release that
    half-failed, so generating twice must be a no-op."""
    once = run("hand-written notes", tmp_path=tmp_path)
    twice = run(once, tmp_path=tmp_path)
    assert twice == once
    assert once.count("## Downloads") == 1


def test_an_unrecognised_asset_is_listed_not_dropped(tmp_path):
    """A sixth platform must show up as an ugly row rather than vanish --
    a missing download is invisible, an ugly one gets fixed."""
    out = run("notes", names=ASSETS + ["sstvae-0.2.0-freebsd-x86_64.tar.gz"],
              tmp_path=tmp_path)
    assert "freebsd" in out


def test_empty_notes_still_produce_a_table(tmp_path):
    out = run("", tmp_path=tmp_path)
    assert "## Downloads" in out
    assert not out.startswith("\n")


def test_output_survives_a_legacy_code_page(tmp_path, monkeypatch):
    """The notes contain characters cp1252 cannot represent, and Windows
    picks stdout's encoding from the locale.

    All nine tests above passed on Linux and macOS and every one of them
    failed on Windows, because `->` in the SmartScreen line is U+2192 and
    cp1252 has no such character. Forcing the child's IO encoding here
    reproduces that on any platform, so the guard does not depend on
    being run on the one that breaks.
    """
    monkeypatch.setenv("PYTHONIOENCODING", "cp1252")
    out = run("notes", tmp_path=tmp_path)
    assert "Downloads" in out
    # And the character that caused it is still there -- the fix is to
    # emit UTF-8, not to ration the text to what cp1252 can spell.
    assert "→" in out


def test_no_carriage_returns(tmp_path):
    """Markdown for the release API, written on whichever runner got the
    job. A CRLF here would make the idempotency check above compare
    unequal on Windows only."""
    out = run("notes", tmp_path=tmp_path)
    assert "\r" not in out


def test_windows_smartscreen_is_mentioned(tmp_path):
    """Operators will hit it, and an unexplained warning on a signed
    download reads as a broken download."""
    out = run("notes", tmp_path=tmp_path)
    assert "SmartScreen" in out

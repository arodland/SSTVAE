#!/usr/bin/env python3
"""Reject `--` inside an XML comment.

XML forbids a double hyphen inside `<!-- -->`, which is a rule nobody
remembers until they write an em-dash as two hyphens in a comment they
were adding for someone else's benefit.

**This exists for the same reason `check_includes.py` does**: it catches
on Linux, in a second, something that otherwise only fails on the
platform least likely to be in front of you. Here that platform is an
Android Gradle build -- the slowest feedback loop in this project -- and
aapt reports the failure against a *copy* of the file under
`build/intermediates/packaged_res/`, so the path in the error is not a
path you can edit.

`native/android-app/android/AndroidManifest.xml` already carried a
comment warning about this. That was not enough: the same mistake landed
in `res/xml/device_filter.xml` two files away, written by the same
person who had put the warning there. A note tells whoever reads that
file; a check tells whoever does not.

Comments only. A `--` *outside* a comment is perfectly legal and appears
in this tree for real -- Qt's manifest template carries
`android:versionCode="-- %%INSERT_VERSION_CODE%% --"` -- so scanning for
the characters alone would report the placeholders forever and be
switched off within a week.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def problems(text: str) -> list[tuple[int, int, str]]:
    """(line, column, message) for each malformed comment."""
    out: list[tuple[int, int, str]] = []
    at = 0
    while True:
        start = text.find("<!--", at)
        if start < 0:
            return out
        end = text.find("-->", start + 4)
        if end < 0:
            line = text.count("\n", 0, start) + 1
            col = start - (text.rfind("\n", 0, start) + 1) + 1
            out.append((line, col, "comment is never closed"))
            return out

        body = text[start + 4 : end]
        # Report the first offending `--`, which is the one to fix.
        bad = body.find("--")
        if bad >= 0:
            pos = start + 4 + bad
            line = text.count("\n", 0, pos) + 1
            col = pos - (text.rfind("\n", 0, pos) + 1) + 1
            out.append((line, col, "'--' is not permitted inside a comment"))
        at = end + 3


def main() -> int:
    files = [
        p
        for p in ROOT.rglob("*.xml")
        if "third_party" not in p.parts
        and "build" not in p.parts
        and not any(part.startswith(".") for part in p.relative_to(ROOT).parts)
    ]

    failures = 0
    for path in sorted(files):
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            print(f"{path.relative_to(ROOT)}: could not read ({exc})")
            failures += 1
            continue
        for line, col, message in problems(text):
            print(f"{path.relative_to(ROOT)}:{line}:{col}: {message}")
            failures += 1

    if failures:
        print(
            f"\n{failures} malformed XML comment(s). "
            "Use a colon, a single hyphen, or a real em-dash instead."
        )
        return 1
    print(f"xml comments ok ({len(files)} file(s) checked)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

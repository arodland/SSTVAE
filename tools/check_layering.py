#!/usr/bin/env python3
"""Enforce the native tree's layering rules.

docs/native-app.md: *"Enforced by a CI grep, not by good intentions."*
This is that grep, made specific enough to be worth running.

The rules mirror the Python package's, for the same reasons:

* **Nothing under `core/` may include QtWidgets.** The Python rule is
  "nothing below sstvae/gui/ may import Qt"; the point is that the
  engines stay drivable without a GUI, which is what makes headless
  tests and CLI tools possible at all.
* **`core/overlay/` may include QtGui, but not QtWidgets.** Overlay
  rendering has to work under QGuiApplication with an offscreen
  platform, so an overlay stays renderable from the command line. This
  is the C++ restatement of "nothing in sstvae/overlay/ may import Qt".
* **Nothing outside `bindings/embed/` may link libpython.** The
  dev-only build that embeds the Python modem is the single exception,
  and it is never shipped.
* **`config.hpp` is generated**, so nothing may be hand-edited into it.
  Checked by regenerating; see tools/gen_config_header.py --check.

Run with no arguments to check the whole tree.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NATIVE = ROOT / "native"

SOURCE_SUFFIXES = {".cpp", ".hpp", ".h", ".cc", ".cxx"}

# (description, path predicate, forbidden #include regex)
RULES = [
    (
        "core/ must not depend on QtWidgets (it has to stay headless)",
        lambda p: p.parts[0] == "core",
        re.compile(r'^\s*#\s*include\s*[<"]QtWidgets', re.M),
    ),
    (
        "core/ outside overlay/ must not depend on QtGui either",
        lambda p: p.parts[0] == "core" and (len(p.parts) < 2 or p.parts[1] != "overlay"),
        re.compile(r'^\s*#\s*include\s*[<"]QtGui', re.M),
    ),
    (
        "only bindings/embed/ may link libpython",
        lambda p: not (p.parts[:2] == ("bindings", "embed")),
        re.compile(r'^\s*#\s*include\s*[<"](Python\.h|pybind11/embed\.h)', re.M),
    ),
]

# The pybind11 *module* is a test fixture, not application code, and of
# course includes pybind11 headers. It does not embed an interpreter --
# that is the distinction the libpython rule is really drawing.
EXEMPT = {("bindings", "module", "module.cpp")}


def sources() -> list[Path]:
    out = []
    for path in sorted(NATIVE.rglob("*")):
        if path.suffix not in SOURCE_SUFFIXES or not path.is_file():
            continue
        rel = path.relative_to(NATIVE)
        if rel.parts[0] == "build" or rel.parts[0].startswith("build-"):
            continue
        # Vendored code is not ours to lay out. Scanning it inflates the
        # reported count into implying coverage we do not have, and a
        # third-party file that merely *mentions* QtGui would fail a rule
        # written about this project's structure.
        if rel.parts[0] == "third_party":
            continue
        out.append(path)
    return out


def main() -> int:
    violations: list[str] = []
    files = sources()
    for path in files:
        rel = path.relative_to(NATIVE)
        if tuple(rel.parts) in EXEMPT:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for description, applies, pattern in RULES:
            if not applies(rel):
                continue
            for match in pattern.finditer(text):
                line = text[: match.start()].count("\n") + 1
                violations.append(
                    f"native/{rel}:{line}: {description}\n"
                    f"    {match.group(0).strip()}"
                )

    generated = NATIVE / "core" / "config.hpp"
    if generated.exists():
        head = generated.read_text(encoding="utf-8")[:200]
        if "GENERATED FILE" not in head:
            violations.append(
                f"native/core/config.hpp has lost its generated-file banner; "
                "it must come from tools/gen_config_header.py"
            )

    if violations:
        print("Layering violations:\n", file=sys.stderr)
        for v in violations:
            print(f"  {v}", file=sys.stderr)
        print(f"\n{len(violations)} violation(s). See docs/native-app.md.",
              file=sys.stderr)
        return 1

    print(f"layering ok ({len(files)} source files checked)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

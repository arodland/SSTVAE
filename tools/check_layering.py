#!/usr/bin/env python3
"""Enforce the native tree's layering rules.

docs/native-app.md: *"Enforced by a CI grep, not by good intentions."*
This is that grep, made specific enough to be worth running.

The rules mirror the Python package's, for the same reasons:

* **Nothing under `core/` may include QtWidgets.** The Python package's
  rule was "nothing below sstvae/gui/ may import Qt"; that GUI is gone,
  but the point it protected is not -- the engines stay drivable without
  a GUI, which is what makes headless tests and CLI tools possible at
  all, and `sstvae/rx/`, `sstvae/tx/` and `sstvae/overlay/` are still
  Qt-free for exactly that reason.
* **`core/overlay/` may include QtGui, but not QtWidgets.** Overlay
  rendering has to work under QGuiApplication with an offscreen
  platform, so an overlay stays renderable from the command line. This
  is the C++ restatement of "nothing in sstvae/overlay/ may import Qt".
* **Only `core/audio/android/` and `core/rig/android/` may include
  `<jni.h>`.** Same rule and same reason as Qt Multimedia's below: the
  engines stay drivable with no platform audio and no platform serial
  port at all, which is what keeps the headless tests and the CLI tools
  possible. A JNI include elsewhere under `core/` would quietly make the
  core un-buildable off Android.

  Both are `<subsystem>/android/` and both are separate libraries, which
  is the convention -- but this is an allowlist rather than a pattern on
  purpose. A third one should be a deliberate edit here, not something
  that starts passing because of where a directory was put.
* **Only `core/audio/qt/` may include Qt Multimedia.** Soundcard I/O is
  the one place in `core/` that needs a Qt module, and it is a separate
  library (`SSTVAE_BUILD_QTAUDIO`) so the modem, the codec and both
  engines still build and test on a machine with no Qt at all. The
  engines take their player and their decoder as seams to keep that
  true; a Qt include anywhere else in `core/` would quietly undo it.
* **Only `core/checkpoint/` may include Qt Network.** The Hub download
  is the one thing in `core/` that needs a network stack, and it is a
  separate library so that resolution and the cache lookup -- which is
  every case but a first run -- stay dependency-free.
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

# The C++ counterparts of sstvae/qrss's format modules (what goes on air).
QRSS_FORMAT_MODULES = {
    "sequences", "frame", "precoder", "morse", "polar", "header", "picture",
    "beaconfile", "pulse", "ce", "lwave", "tx", "config", "tables",
}

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
        "only core/audio/qt/ may depend on Qt Multimedia",
        lambda p: p.parts[0] == "core" and p.parts[:3] != ("core", "audio", "qt"),
        re.compile(r'^\s*#\s*include\s*[<"](QtMultimedia|QAudio|QMediaDevices)', re.M),
    ),
    (
        "only core/audio/android/ and core/rig/android/ may depend on JNI",
        lambda p: p.parts[0] == "core"
        and p.parts[:3] not in (("core", "audio", "android"), ("core", "rig", "android")),
        re.compile(r'^\s*#\s*include\s*[<"]jni\.h', re.M),
    ),
    (
        "only core/checkpoint/ may depend on Qt Network",
        lambda p: p.parts[0] == "core" and p.parts[:2] != ("core", "checkpoint"),
        re.compile(r'^\s*#\s*include\s*[<"](QtNetwork|QNetwork)', re.M),
    ),
    (
        "core/qrss/ and core/math/ must not depend on Qt at all (QtCore included): "
        "they link no Qt",
        lambda p: p.parts[:2] in (("core", "qrss"), ("core", "math")),
        re.compile(r'^\s*#\s*include\s*[<"](Qt[A-Za-z]*/|Q[A-Z][A-Za-z]*>|Qt[A-Z][A-Za-z]*>)', re.M),
    ),
    (
        "QRSSTVAE format modules must not use a random generator "
        "(what goes on air is closed form; tests/test_qrss_frozen.py is the Python rule)",
        lambda p: p.parts[:2] == ("core", "qrss") and p.stem in QRSS_FORMAT_MODULES,
        re.compile(r'(^\s*#\s*include\s*<random>|\b(s?rand|mt19937(_64)?|default_random_engine)\b)', re.M),
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
        # Dot-directories are tooling, not source: `.deps` is where
        # FetchContent unpacks onnxruntime, and scanning its headers
        # inflated this tool's reported count from 32 files to 51 while
        # checking nothing anyone here wrote.
        if rel.parts[0].startswith("."):
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

    for rel, tool in (("core/config.hpp", "tools/gen_config_header.py"),
                      ("core/qrss/qrss_config.hpp", "tools/gen_qrss_config_header.py"),
                      ("core/qrss/tables.hpp", "tools/gen_qrss_config_header.py"),
                      ("core/qrss/tables.cpp", "tools/gen_qrss_config_header.py"),
                      ("core/qrss/rx_config.hpp", "tools/gen_qrss_rx_config.py")):
        generated = NATIVE / rel
        if generated.exists():
            head = generated.read_text(encoding="utf-8")[:200]
            if "GENERATED FILE" not in head:
                violations.append(
                    f"native/{rel} has lost its generated-file banner; "
                    f"it must come from {tool}"
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

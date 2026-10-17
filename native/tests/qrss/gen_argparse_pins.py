#!/usr/bin/env python3
"""The pins of native/tests/qrss/test_argparse.cpp, from CPython's own argparse.

Writes native/tests/qrss/argparse_pins.hpp, which is committed so the ctest
needs no Python. Re-run it from the repository root if PARSERS or CASES
change:

    python native/tests/qrss/gen_argparse_pins.py

sstvae-qrss's argparse (native/apps/sstvae_qrss/argparse.hpp) models the part
of argparse the qrss_*.py scripts use; these pins hold its mutually exclusive
groups to Python's: the usage line (one bracket for a group whose options were
added together, "(a | b)" for a required one, bare for a required group of
one, each option bracketed alone when the group's are not together), "not
allowed with" raised at the second option once its value has passed type and
choices, a required group's "one of the arguments ... is required" after the
missing-arguments refusal and before "unrecognized arguments".

Each parser is built on both sides from the same table: its arguments in
order (a flag, an option with a type, or a positional) and its groups (the
flags they hold, required or not). Python adds an argument to its group when
it is added, the groups created first in table order.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import os
import sys
from pathlib import Path

OUT = Path(__file__).with_name("argparse_pins.hpp")
PROG = "prog"

# (kind, name, type): kind "flag" (store_true), "option" (one value), "positional"
PARSERS = [
    {   # the receive shape: a plain group of two flags, among other options
        "args": [("option", "--slot", "str"), ("flag", "--a", ""), ("flag", "--b", ""),
                 ("option", "--n", "int"), ("positional", "x", "")],
        "groups": [(["--a", "--b"], False)],
    },
    {   # a required group of two options with values
        "args": [("option", "--a", "int"), ("option", "--b", "float"), ("flag", "--c", ""),
                 ("positional", "x", "")],
        "groups": [(["--a", "--b"], True)],
    },
    {   # a required group whose options are not next to each other
        "args": [("flag", "--a", ""), ("flag", "--b", ""), ("option", "--c", "str")],
        "groups": [(["--a", "--c"], True)],
    },
    {   # a required group of one, and a plain group of one
        "args": [("flag", "--a", ""), ("flag", "--b", ""), ("positional", "x", "")],
        "groups": [(["--a"], True), (["--b"], False)],
    },
    {   # two groups side by side, one of three
        "args": [("flag", "--a", ""), ("flag", "--b", ""), ("flag", "--c", ""),
                 ("option", "--d", "str"), ("flag", "--e", ""), ("positional", "x", "")],
        "groups": [(["--a", "--b", "--c"], False), (["--d", "--e"], False)],
    },
    {   # a plain group whose options are not next to each other
        "args": [("flag", "--a", ""), ("option", "--m", "str"), ("flag", "--b", ""),
                 ("positional", "x", ""), ("positional", "y", "")],
        "groups": [(["--a", "--b"], False)],
    },
]

CASES = [
    (0, []), (0, ["v"]), (0, ["v", "--a"]), (0, ["v", "--a", "--b"]), (0, ["v", "--b", "--a"]),
    (0, ["--a", "--b"]), (0, ["v", "--a", "--a"]), (0, ["--b", "v", "--a"]),
    (0, ["v", "--a", "--b=1"]), (0, ["v", "--a", "--b", "--zz"]), (0, ["v", "--a", "--b", "w"]),
    (0, ["v", "--a", "--n", "q", "--b"]), (0, ["v", "--a", "--b", "--n", "q"]),
    (0, ["v", "--a", "--n"]), (0, ["v", "--a", "--", "--b"]), (0, ["v", "--b", "--a", "--b"]),
    (0, ["v", "--a", "--slot", "--b"]),
    (1, []), (1, ["v"]), (1, ["v", "--a", "1"]), (1, ["v", "--b", "2.5"]), (1, ["v", "--a", "1", "--b", "2"]),
    (1, ["v", "--a", "1", "--b", "z"]), (1, ["v", "--a", "z", "--b", "2"]), (1, ["v", "--b=2", "--a=1"]),
    (1, ["v", "--c"]), (1, ["--c"]), (1, ["v", "--zz"]), (1, ["v", "--a", "1", "--a", "2"]),
    (1, ["v", "--a", "1", "--b"]),
    (2, []), (2, ["--a"]), (2, ["--c", "q"]), (2, ["--a", "--c", "q"]), (2, ["--c", "q", "--a"]),
    (2, ["--b"]), (2, ["--b", "--zz"]), (2, ["--a", "--b", "--c=q"]),
    (3, []), (3, ["v"]), (3, ["v", "--a"]), (3, ["v", "--a", "--b"]), (3, ["v", "--b", "--a", "--b"]),
    (4, ["v"]), (4, ["v", "--a", "--c"]), (4, ["v", "--c", "--b"]), (4, ["v", "--a", "--d", "q"]),
    (4, ["v", "--a", "--e", "--d", "q"]), (4, ["v", "--d", "q", "--b", "--e"]), (4, ["v", "--b", "--c", "--e"]),
    (5, ["v", "w"]), (5, ["v", "w", "--a", "--b"]), (5, ["v", "--a", "--m", "q", "--b", "w"]),
    (5, ["v", "--b", "--m", "q"]),
]


def build(spec: dict) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog=PROG, allow_abbrev=True)
    owner = {}
    for flags, required in spec["groups"]:
        g = ap.add_mutually_exclusive_group(required=required)
        for f in flags:
            owner[f] = g
    for kind, name, typ in spec["args"]:
        target = owner.get(name, ap)
        if kind == "flag":
            target.add_argument(name, action="store_true")
        elif kind == "option":
            target.add_argument(name, type={"str": str, "int": int, "float": float}[typ])
        else:
            target.add_argument(name)
    return ap


def run(ap: argparse.ArgumentParser, argv: list[str]) -> tuple[int, str]:
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        try:
            ap.parse_args(argv)
        except SystemExit as e:
            return int(e.code), err.getvalue()
    return 0, err.getvalue()


def cstr(s: str) -> str:
    out = []
    for b in s.encode():
        if b == 0x22 or b == 0x5C:
            out.append("\\" + chr(b))
        elif 0x20 <= b < 0x7F:
            out.append(chr(b))
        else:
            out.append("\\%03o" % b)
    return '"' + "".join(out) + '"'


def main() -> None:
    os.environ["COLUMNS"] = "10000"   # one usage line, as sstvae-qrss prints it
    lines = [
        "// GENERATED by native/tests/qrss/gen_argparse_pins.py from CPython's argparse",
        f"// (CPython {sys.version.split()[0]}). Do not edit.",
        "",
        "#pragma once",
        "",
        "namespace argparse_pins {",
        "",
        'constexpr const char* PROG = ' + cstr(PROG) + ";",
        "",
        "// kind: 'f' a store_true flag, 'o' an option taking one value of `type`",
        "// ('s' str, 'i' int, 'f' float), 'p' a positional",
        "struct Arg {",
        "    int parser;",
        "    char kind;",
        "    const char* name;",
        "    char type;",
        "};",
        "// flags space-separated, in the group's order",
        "struct Group {",
        "    int parser;",
        "    const char* flags;",
        "    bool required;",
        "};",
        "struct Usage {",
        "    int parser;",
        "    const char* usage;",
        "};",
        "// argv newline-separated; status 0 (parsed) or 2, and what reached stderr",
        "struct Case {",
        "    int parser;",
        "    const char* argv;",
        "    int status;",
        "    const char* err;",
        "};",
        "",
        "constexpr Arg ARGS[] = {",
    ]
    for p, spec in enumerate(PARSERS):
        for kind, name, typ in spec["args"]:
            t = {"": "s", "str": "s", "int": "i", "float": "f"}[typ]
            lines.append(f"    {{{p}, '{kind[0]}', {cstr(name)}, '{t}'}},")
    lines += ["};", "", "constexpr Group GROUPS[] = {"]
    for p, spec in enumerate(PARSERS):
        for flags, required in spec["groups"]:
            lines.append(f"    {{{p}, {cstr(' '.join(flags))}, {'true' if required else 'false'}}},")
    lines += ["};", "", "constexpr Usage USAGES[] = {"]
    for p, spec in enumerate(PARSERS):
        usage = build(spec).format_usage().rstrip("\n")
        assert "\n" not in usage, usage
        lines.append(f"    {{{p}, {cstr(usage)}}},")
    lines += ["};", "", "constexpr Case CASES[] = {"]
    for p, argv in CASES:
        status, err = run(build(PARSERS[p]), argv)
        lines.append(f"    {{{p}, {cstr(chr(10).join(argv))}, {status}, {cstr(err)}}},")
    lines += ["};", "", "}  // namespace argparse_pins", ""]
    OUT.write_text("\n".join(lines))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()

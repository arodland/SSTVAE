#!/usr/bin/env python3
"""The pins of native/tests/qrss/test_argparse_nargs.cpp, from CPython's argparse.

Writes native/tests/qrss/argparse_nargs_pins.hpp, which is committed so the
ctest needs no Python. Re-run it from the repository root if PARSERS or
CASES change:

    python native/tests/qrss/gen_argparse_nargs_pins.py

sstvae-qrss's argparse (native/apps/sstvae_qrss/argparse.hpp) takes one
optional positional (add_argument(name, nargs="?")) as the last positional,
the shape of qrss_decode.py's OUT.png. These pins hold it to Python's: the
usage line ("[name]", after the options), the status and stderr of each
command line, and what each argument holds afterwards -- in particular
where "--" goes (argparse removes the first "--" only where it falls inside
a positional's match, so a second one, or one after the positional was
filled, is a value or an unrecognized argument) and that an optional
positional matched by nothing before an option waits for a later value
(CPython 3.13's _match_arguments_partial).

Each parser is built on both sides from the same table: its arguments in
order (a flag, an option with a type, a positional, or an optional
positional) and its mutually exclusive groups.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import os
import sys
from pathlib import Path

OUT = Path(__file__).with_name("argparse_nargs_pins.hpp")
PROG = "prog"

# (kind, name, type): kind "flag" (store_true), "option" (one value),
# "positional", "optional" (a positional with nargs="?")
PARSERS = [
    {   # qrss_decode.py's shape: the optional positional first, a group of two options and a flag
        "args": [("optional", "output", ""), ("option", "--key", "str"), ("option", "--pass", "str"),
                 ("flag", "--list", ""), ("option", "--store", "str"), ("option", "--em", "int"),
                 ("flag", "--retro", "")],
        "groups": [(["--key", "--pass", "--list"], False)],
    },
    {   # a required positional, then the optional one
        "args": [("positional", "x", ""), ("flag", "--a", ""), ("option", "--n", "int"),
                 ("optional", "y", "")],
        "groups": [],
    },
    {   # the optional positional alone
        "args": [("optional", "out", ""), ("flag", "--a", ""), ("option", "--o", "str")],
        "groups": [],
    },
]

CASES = [
    (0, []), (0, ["out"]), (0, ["out", "--key", "K"]), (0, ["--key", "K", "out"]), (0, ["--key", "K"]),
    (0, ["a", "b"]), (0, ["--"]), (0, ["--", "--list"]), (0, ["out", "--"]), (0, ["a", "--", "b"]),
    (0, ["--key", "K", "--", "--", "x"]), (0, ["-", "--list"]), (0, ["--list", "--", "x"]),
    (0, ["--key", "K", "--"]), (0, ["out", "--key", "K", "--"]), (0, ["--list", "out"]),
    (0, ["out", "--list", "--key", "K"]), (0, ["--em", "3", "out"]), (0, ["--em", "x"]), (0, ["--key"]),
    (0, ["--key", "--list"]), (0, ["-x"]), (0, ["out", "-x"]), (0, ["--store=s", "--list"]), (0, ["--", "-x"]),
    (0, ["out", "--em", "-1"]), (0, ["--em", "-1", "out"]), (0, ["-1"]), (0, ["--key", "K", "-1"]),
    (0, ["--retro", "--", "--", "--"]), (0, ["--list", "--", "--"]), (0, ["--", "a", "b"]),
    (0, ["--key", "K", "--store", "s", "--", "o"]), (0, ["--li"]), (0, ["--re", "o"]), (0, ["o", "--key=K=1"]),
    (0, ["--key", "-x"]), (0, ["--pass", "p", "--list"]), (0, ["o", "--", "--", "--"]), (0, [""]),
    (1, []), (1, ["v"]), (1, ["v", "w"]), (1, ["v", "w", "z"]), (1, ["v", "--a", "w"]), (1, ["--a", "v", "w"]),
    (1, ["v", "--", "w"]), (1, ["--", "v", "w"]), (1, ["v", "--n", "1", "w"]), (1, ["--", "--a"]),
    (1, ["v", "--", "--a"]), (1, ["v", "w", "--"]), (1, ["v", "--"]), (1, ["--n", "1"]),
    (1, ["v", "--a", "--", "w", "z"]), (1, ["--a", "v"]), (1, ["v", "--a"]), (1, ["--", "--", "w"]),
    (2, []), (2, ["o"]), (2, ["--a"]), (2, ["--a", "o"]), (2, ["o", "--a"]), (2, ["--o", "x"]),
    (2, ["--o", "x", "o"]), (2, ["o", "p"]), (2, ["--", "--a"]), (2, ["--a", "--", "--a"]), (2, ["--o", "--"]),
    (2, ["--o=--", "--"]), (2, ["--", "--", "--"]),
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
            target.add_argument(name, type={"str": str, "int": int}[typ])
        elif kind == "optional":
            target.add_argument(name, nargs="?")
        else:
            target.add_argument(name)
    return ap


def dest(name: str) -> str:
    return name.lstrip("-").replace("-", "_")


def run(spec: dict, argv: list[str]) -> tuple[int, str, str]:
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        try:
            ns = build(spec).parse_args(argv)
        except SystemExit as e:
            return int(e.code), err.getvalue(), ""
    values = []
    for kind, name, _ in spec["args"]:
        v = getattr(ns, dest(name))
        values.append(f"{dest(name)}={'None' if v is None else v}")
    return 0, err.getvalue(), "\n".join(values)


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
        "// GENERATED by native/tests/qrss/gen_argparse_nargs_pins.py from CPython's argparse",
        f"// (CPython {sys.version.split()[0]}). Do not edit.",
        "",
        "#pragma once",
        "",
        "namespace argparse_nargs_pins {",
        "",
        'constexpr const char* PROG = ' + cstr(PROG) + ";",
        "",
        "// kind: 'f' a store_true flag, 'o' an option taking one value of `type`",
        "// ('s' str, 'i' int), 'p' a positional, 'q' a positional with nargs='?'",
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
        "// argv newline-separated (an empty argument is an empty line); status 0",
        "// (parsed) or 2, what reached stderr, and on 0 each argument's value as",
        "// \"dest=value\" lines in table order (None, True / False, an int in decimal)",
        "struct Case {",
        "    int parser;",
        "    int argc;",
        "    const char* argv;",
        "    int status;",
        "    const char* err;",
        "    const char* values;",
        "};",
        "",
        "constexpr Arg ARGS[] = {",
    ]
    for p, spec in enumerate(PARSERS):
        for kind, name, typ in spec["args"]:
            k = {"flag": "f", "option": "o", "positional": "p", "optional": "q"}[kind]
            t = {"": "s", "str": "s", "int": "i"}[typ]
            lines.append(f"    {{{p}, '{k}', {cstr(name)}, '{t}'}},")
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
        status, err, values = run(PARSERS[p], argv)
        lines.append(f"    {{{p}, {len(argv)}, {cstr(chr(10).join(argv))}, {status}, {cstr(err)}, {cstr(values)}}},")
    lines += ["};", "", "}  // namespace argparse_nargs_pins", ""]
    OUT.write_text("\n".join(lines))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()

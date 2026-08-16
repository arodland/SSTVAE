#!/usr/bin/env python3
"""Every std:: name a native source uses must have its header included.

libstdc++ and libc++ pull in far more than they promise -- `<vector>`
happens to give you `std::count_if`, `<string>` happens to give you
`std::size_t` -- so a missing `#include <algorithm>` compiles cleanly on
Linux and macOS and fails only on MSVC, which is stricter. That has now
cost two CI rounds, and it is exactly the kind of error that is trivial
to fix and expensive to find, because the feedback comes from the one
platform a developer here is least likely to be running.

So this checks it directly, on any platform. It is deliberately *not*
include-what-you-use: no extra dependency, no build database, and it
only reports the direction that actually breaks a build (a name used
without its header), never an unused include.

Project headers are followed, so a .cpp that gets `<vector>` from its
own .hpp is fine -- that is a real guarantee, unlike a standard header
that happens to include another today.

Exits non-zero on the first file with a problem. Run it directly or via
CI; `tools/check_layering.py` is its sibling.
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NATIVE = ROOT / "native"
# Where a #include "..." is resolved from, in order.
SEARCH_ROOTS = [NATIVE / "core", NATIVE / "tests", NATIVE]

# std:: name -> the header that is *required* to provide it. Only
# unambiguous cases belong here: a name that two headers may legitimately
# provide would produce false positives, which would get this turned off.
SYMBOLS = {
    "algorithm": [
        "all_of", "any_of", "none_of", "count", "count_if", "find", "find_if",
        "copy", "copy_n", "fill", "sort", "stable_sort", "nth_element",
        "max_element", "min_element", "max", "min", "clamp", "reverse",
        "transform", "equal", "lower_bound", "upper_bound", "unique",
        "remove_if", "accumulate_if",
    ],
    "atomic": ["atomic", "atomic_flag", "memory_order"],
    "array": ["array"],
    "chrono": ["chrono"],
    "cmath": ["sqrt", "abs", "fabs", "pow", "exp", "log", "log10", "sin", "cos",
              "atan2", "hypot", "isnan", "isinf", "nearbyint", "nearbyintf",
              "lround", "llround", "floor", "ceil", "fmod", "nan"],
    "complex": ["complex", "conj", "norm", "arg", "polar"],
    "condition_variable": ["condition_variable", "condition_variable_any"],
    "cstddef": ["size_t", "ptrdiff_t", "byte"],
    "cstdint": ["int8_t", "uint8_t", "int16_t", "uint16_t", "int32_t",
                "uint32_t", "int64_t", "uint64_t", "intptr_t", "uintptr_t"],
    "cstdio": ["printf", "fprintf", "snprintf", "sprintf", "fopen", "fclose",
               "fread", "fwrite", "fflush", "FILE"],
    "cstdlib": ["abort", "exit", "getenv", "llabs", "labs", "strtod", "strtol"],
    "cstring": ["memcpy", "memset", "memcmp", "strlen", "strcmp"],
    "ctime": ["time_t", "tm", "strftime", "localtime", "mktime"],
    "deque": ["deque"],
    "filesystem": ["filesystem"],
    "functional": ["function", "bind", "ref", "cref", "hash"],
    "limits": ["numeric_limits"],
    "map": ["map", "multimap"],
    "memory": ["unique_ptr", "shared_ptr", "make_unique", "make_shared",
               "weak_ptr", "enable_shared_from_this"],
    "mutex": ["mutex", "lock_guard", "unique_lock", "scoped_lock", "recursive_mutex"],
    "numeric": ["accumulate", "iota", "inner_product", "partial_sum"],
    "optional": ["optional", "nullopt", "make_optional"],
    "ostream": ["ostream"],
    "set": ["set", "multiset"],
    "span": ["span"],
    "sstream": ["ostringstream", "istringstream", "stringstream"],
    "stdexcept": ["runtime_error", "logic_error", "invalid_argument",
                  "out_of_range", "length_error", "domain_error"],
    "string": ["string", "to_string", "stod", "stoi", "stoll", "getline"],
    "string_view": ["string_view"],
    "thread": ["thread", "this_thread", "jthread"],
    "unordered_map": ["unordered_map"],
    "utility": ["move", "forward", "pair", "make_pair", "swap", "exchange"],
    "variant": ["variant", "get_if", "holds_alternative", "visit"],
    "vector": ["vector"],
}

# Reverse it, and note which names are provided by more than one header
# so those can be treated as satisfied by any of them.
PROVIDERS: dict[str, set[str]] = {}
for header, names in SYMBOLS.items():
    for name in names:
        PROVIDERS.setdefault(name, set()).add(header)

STD_USE = re.compile(r"\bstd::([A-Za-z_][A-Za-z_0-9]*)")
SYS_INCLUDE = re.compile(r'^\s*#\s*include\s*<([^>]+)>', re.M)
LOCAL_INCLUDE = re.compile(r'^\s*#\s*include\s*"([^"]+)"', re.M)
# Comments are stripped before scanning: prose mentioning std::vector is
# not a use, and this file's own docstring should not fail it.
BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.S)
LINE_COMMENT = re.compile(r"//[^\n]*")
STRING_LITERAL = re.compile(r'"(?:[^"\\\n]|\\.)*"')


def strip(text: str) -> str:
    text = BLOCK_COMMENT.sub(" ", text)
    text = LINE_COMMENT.sub(" ", text)
    return STRING_LITERAL.sub('""', text)


def resolve(include: str, origin: Path) -> Path | None:
    for root in [origin.parent, *SEARCH_ROOTS]:
        candidate = root / include
        if candidate.is_file():
            return candidate.resolve()
    return None


def system_includes(path: Path, seen: set[Path] | None = None) -> set[str]:
    """Every <header> this file gets, following project headers."""
    seen = seen if seen is not None else set()
    path = path.resolve()
    if path in seen:
        return set()
    seen.add(path)

    text = path.read_text(encoding="utf-8", errors="replace")
    headers = set(SYS_INCLUDE.findall(text))
    for local in LOCAL_INCLUDE.findall(text):
        target = resolve(local, path)
        if target is not None:
            headers |= system_includes(target, seen)
    return headers


def check(path: Path) -> list[str]:
    body = strip(path.read_text(encoding="utf-8", errors="replace"))
    available = system_includes(path)
    problems = []
    for name in sorted(set(STD_USE.findall(body))):
        providers = PROVIDERS.get(name)
        if providers and not (providers & available):
            want = " or ".join(f"<{h}>" for h in sorted(providers))
            problems.append(f"uses std::{name} without {want}")
    return problems


def main() -> int:
    # Vendored third-party code is not ours to fix.
    files = [
        p for p in NATIVE.rglob("*")
        if p.suffix in {".cpp", ".hpp", ".h"}
        and "third_party" not in p.parts
        and not any(part.startswith(".") for part in p.relative_to(NATIVE).parts)
        # `build-asan`, `build-gui`, `build-android` and friends as well as
        # plain `build`: a FetchContent tree lands *inside* the build
        # directory, so a name this misses turns onnxruntime's and
        # Hamlib's headers into seven failures in code nobody here wrote.
        # check_layering.py has always excluded the prefix; this did not.
        and not any(part == "build" or part.startswith("build-") for part in p.parts)
    ]

    failures = 0
    for path in sorted(files):
        for problem in check(path):
            print(f"{path.relative_to(ROOT)}: {problem}")
            failures += 1

    if failures:
        print(f"\n{failures} missing include(s). "
              "libstdc++ forgives these; MSVC does not.")
        return 1
    print(f"includes ok ({len(files)} source files checked)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

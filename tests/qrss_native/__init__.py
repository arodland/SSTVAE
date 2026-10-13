"""`pytest --native-qrss=STAGE`: the QRSS suites against the C++ port.

The same mechanism as SSTVAE's `--native` (tests/conftest.py), kept
separate so that switch keeps its exact meaning: the C++ implementations
in `sstvae_native.qrss` are assigned over the reference's attributes in
`sstvae.qrss`, and the existing `tests/test_qrss_*` suites then test the
port. Python stays the reference (CLAUDE.md, "The native port").

One file per ported module, `tests/qrss_native/<module>.py`, so modules
ported in parallel never edit a shared table. Each defines

    STAGE = "tx" | "rx" | "live"     # the port stage it belongs to
    def rows(q) -> dict[(target, attr), replacement]

where `q` is `sstvae_native.qrss`, `target` a module name (or
"module.Class") and `replacement` the native callable. `--native-qrss=rx`
applies every row of stage tx and rx.

Two rules, both from what went wrong with the SSTVAE port:

- **A row naming something that does not exist is an error, never a
  skip**, on either side: a missing reference attribute means the table
  is stale; a missing native attribute means a stale or partial build.
  Either would otherwise leave a test exercising Python while reporting
  itself as a native run.
- **From-import sites are found, not listed.** `from .sequences import
  scrambler` binds the function at import time, so patching
  `sequences.scrambler` does not reach `precoder.scrambler`. After every
  `sstvae.qrss` module is imported, each name in each of them still bound
  to a replaced object is replaced too. What this cannot reach: values
  computed at import from the reference (module-level tables), default
  arguments and closures. A module whose tables matter substitutes the
  tables as rows of their own.
"""

from __future__ import annotations

import importlib
import pkgutil
from dataclasses import dataclass, field
from pathlib import Path

STAGES = ("tx", "rx", "live")


@dataclass
class Report:
    stage: str
    modules: list[str] = field(default_factory=list)
    rows: list[tuple[str, str]] = field(default_factory=list)
    extra_sites: int = 0


def _row_modules():
    here = Path(__file__).resolve().parent
    for info in sorted(pkgutil.iter_modules([str(here)]), key=lambda m: m.name):
        if info.name.startswith("_"):
            continue
        yield info.name, importlib.import_module(f"qrss_native.{info.name}")


def _owner(target: str):
    try:
        return importlib.import_module(target)
    except ImportError:
        mod_name, _, cls_name = target.rpartition(".")
        return getattr(importlib.import_module(mod_name), cls_name)


def _import_reference():
    """Every sstvae.qrss module, so from-import sites exist to be found."""
    import sstvae.qrss as pkg

    mods = [pkg]
    for info in pkgutil.iter_modules(pkg.__path__):
        mods.append(importlib.import_module(f"sstvae.qrss.{info.name}"))
    return mods


def qrss(native):
    """`native.qrss`, or LookupError naming the missing build."""
    q = getattr(native, "qrss", None)
    if q is None:
        raise LookupError(
            f"{native.__file__} has no qrss submodule; build with SSTVAE_BUILD_QRSS=ON")
    return q


# Set by apply(): the reference is then partly the C++ itself, so a parity
# test (tests/test_qrss_native_*.py) would compare native with native.
ACTIVE: str | None = None


def require(native, module: str):
    """`native.qrss.<module>` for a parity test, skipping when not built,
    and under --native-qrss (where the reference is no longer Python)."""
    import pytest

    if ACTIVE is not None:
        pytest.skip(f"parity tests compare with the Python reference; "
                    f"--native-qrss={ACTIVE} has substituted it")
    q = getattr(native, "qrss", None)
    if q is None or not hasattr(q, module):
        pytest.skip(f"sstvae_native.qrss.{module} not built")
    return getattr(q, module)


def apply(native, stage: str) -> Report:
    if stage not in STAGES:
        raise LookupError(f"unknown stage {stage!r}; one of {', '.join(STAGES)}")
    global ACTIVE
    q = qrss(native)
    ACTIVE = stage
    upto = STAGES[: STAGES.index(stage) + 1]
    report = Report(stage=stage)
    reference = _import_reference()
    replaced: dict[int, object] = {}
    for name, mod in _row_modules():
        if getattr(mod, "STAGE", None) not in STAGES:
            raise LookupError(f"tests/qrss_native/{name}.py has no valid STAGE")
        if mod.STAGE not in upto:
            continue
        rows = mod.rows(q)
        report.modules.append(name)
        for (target, attr), repl in rows.items():
            owner = _owner(target)
            if not hasattr(owner, attr):
                raise LookupError(
                    f"{target}.{attr} does not exist; tests/qrss_native/{name}.py is "
                    "out of date with the reference")
            if repl is None:
                raise LookupError(f"tests/qrss_native/{name}.py: no native {target}.{attr}")
            original = getattr(owner, attr)
            setattr(owner, attr, repl)
            report.rows.append((target, attr))
            if callable(original) or hasattr(original, "__array__"):
                replaced[id(original)] = (original, repl)
    # From-import sites: names in other modules still bound to a replaced object.
    for m in reference:
        for key, value in list(vars(m).items()):
            hit = replaced.get(id(value))
            if hit is not None and hit[0] is value:
                setattr(m, key, hit[1])
                report.extra_sites += 1
    return report

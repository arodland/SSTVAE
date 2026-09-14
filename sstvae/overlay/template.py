"""Templates: an overlay document with holes in it.

The reference implementation of ``docs/overlay-templates.md``. A template
is an ordinary :class:`OverlayDoc` whose text items contain placeholders,
and this module is the pure string processing that fills them in. The
renderer never sees a placeholder: :func:`substitute` returns a plain
document and the file the template came from is never mutated.

Placeholders are ``{name}`` inside ``TextItem.text``:

- the **built-in** names in :data:`BUILTIN_FIELDS`, every one of which
  the app fills in without the operator typing anything except
  ``theircall``;
- ``{field Label}``, a **custom** field presented to the operator as a
  text box with that label. The same label twice is one field.

Three rules, and they are the whole contract:

1. An unknown placeholder is left literally in the text, so a typo is
   visible in the preview rather than silently deleted from the air.
   ``{{`` and ``}}`` are literal braces.
2. A line whose placeholders are all empty is dropped, literal text and
   all -- ``SNR {snr}`` vanishes whole when there is no SNR, since a
   label with nothing after it is worse than no line. Only whole lines,
   never partial text; a line with no placeholders is never touched.
3. The stored template is never mutated. Substitution works on a copy.

Nothing here imports Qt or PIL; it is data in, data out, and the C++
counterpart in ``native/core/overlay/template.cpp`` is held to the same
outputs by ``tests/test_native_overlay.py``.
"""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field
from pathlib import Path

from .model import OverlayDoc, TextItem

# The closed set of built-in placeholder names. Adding one here is a
# format change for every template file in the field: a name that an
# older build does not know renders literally there (rule 1), which is
# the intended forward-compatibility behaviour, but it is still worth
# knowing that is what happens.
BUILTIN_FIELDS = ("mycall", "grid", "name", "theircall", "snr", "utc", "date", "mode")

# Where the shipped templates live. A package resource so the CLI and
# the tests find them without an install step; the apps bundle the same
# files (Android in assets/, the desktop in its data directory).
BUILTIN_DIR = Path(__file__).resolve().parent / "templates"

_BUILTIN_RE = re.compile(r"^[a-z]+$")
_CUSTOM_RE = re.compile(r"^field\s+(\S.*?)\s*$")


@dataclass
class Fields:
    """What to fill the holes with.

    ``builtin`` is keyed by placeholder name, ``custom`` by label.
    Anything missing is treated as empty, so a caller need only supply
    what it has.
    """

    builtin: dict = field(default_factory=dict)
    custom: dict = field(default_factory=dict)


@dataclass
class Placeholders:
    """What a template asks for, in order of first appearance and
    without repeats. This is what derives the per-over form: no schema,
    nothing to keep in sync with the text."""

    builtin: list = field(default_factory=list)
    custom: list = field(default_factory=list)


def normalize_label(label: str) -> str:
    """Trim and collapse internal whitespace, so ``{field  Comment}``
    and ``{field Comment}`` are one field."""
    return " ".join(label.split())


# --- tokenizing ---------------------------------------------------------
#
# A token is (kind, value): kind is "text" (value is literal, escapes
# already resolved), "builtin" (value is the name) or "custom" (value is
# the normalized label). Unknown placeholders come back as "text" with
# the braces intact, which is rule 1.


def _tokenize_line(line: str) -> list:
    tokens = []
    literal = []
    i = 0
    n = len(line)
    while i < n:
        c = line[i]
        if c == "{":
            if i + 1 < n and line[i + 1] == "{":
                literal.append("{")
                i += 2
                continue
            end = line.find("}", i + 1)
            inner = line[i + 1 : end] if end != -1 else None
            if inner is not None and "{" not in inner:
                kind, value = _classify(inner)
                if kind != "text":
                    if literal:
                        tokens.append(("text", "".join(literal)))
                        literal = []
                    tokens.append((kind, value))
                    i = end + 1
                    continue
            # No closing brace, or not a placeholder we know: literal.
            literal.append("{")
            i += 1
            continue
        if c == "}":
            if i + 1 < n and line[i + 1] == "}":
                literal.append("}")
                i += 2
                continue
            literal.append("}")
            i += 1
            continue
        literal.append(c)
        i += 1
    if literal:
        tokens.append(("text", "".join(literal)))
    return tokens


def _classify(inner: str):
    if _BUILTIN_RE.match(inner) and inner in BUILTIN_FIELDS:
        return "builtin", inner
    m = _CUSTOM_RE.match(inner)
    if m:
        return "custom", normalize_label(m.group(1))
    return "text", None


# --- the public functions -----------------------------------------------


def placeholders(doc: OverlayDoc) -> Placeholders:
    """Which built-ins and which custom fields ``doc`` uses."""
    out = Placeholders()
    for item in doc.items:
        if not isinstance(item, TextItem):
            continue
        for line in item.text.split("\n"):
            for kind, value in _tokenize_line(line):
                if kind == "builtin" and value not in out.builtin:
                    out.builtin.append(value)
                elif kind == "custom" and value not in out.custom:
                    out.custom.append(value)
    return out


def _is_empty(value) -> bool:
    # Whitespace-only counts as empty: a stray space in a comment box is
    # not a comment, and dropping the line is what the operator meant.
    return value is None or not str(value).strip()


def substitute_text(text: str, fields: Fields) -> str:
    """Rules 1 and 2 on one text item's content."""
    kept = []
    for line in text.split("\n"):
        tokens = _tokenize_line(line)
        holes = [t for t in tokens if t[0] != "text"]
        parts = []
        any_filled = False
        for kind, value in tokens:
            if kind == "text":
                parts.append(value)
                continue
            source = fields.builtin if kind == "builtin" else fields.custom
            filled = source.get(value)
            if _is_empty(filled):
                continue
            any_filled = True
            parts.append(str(filled))
        if holes and not any_filled:
            continue  # rule 2: every placeholder empty -> the line goes
        kept.append("".join(parts))
    return "\n".join(kept)


def substitute(doc: OverlayDoc, fields: Fields) -> OverlayDoc:
    """A copy of ``doc`` with every hole filled (rule 3: ``doc`` itself
    is untouched). Image items pass through unchanged."""
    out = copy.deepcopy(doc)
    for item in out.items:
        if isinstance(item, TextItem):
            item.text = substitute_text(item.text, fields)
    return out


def format_snr(snr_db) -> str:
    """The ``{snr}`` value: an integer with its unit, ``12 dB``; empty
    when there is nothing measured. Rounds half to even, like numpy and
    like the C++ ``nearbyint``, so the two implementations print the
    same figure for the same sidecar."""
    if snr_db is None:
        return ""
    return f"{int(round(float(snr_db)))} dB"


def load_template(text: str) -> OverlayDoc:
    """A template file is an overlay document, optionally with a name;
    the model's reader already understands both."""
    return OverlayDoc.from_json(text)


def builtin_templates() -> list:
    """The shipped templates, in the order the UI lists them. ``None``
    (send the picture unmodified) is the absence of a template and is
    not a file."""
    order = ["cq", "reply", "reply-picture"]
    return [load_template((BUILTIN_DIR / f"{stem}.json").read_text()) for stem in order]

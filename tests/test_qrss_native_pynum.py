"""Parity of native/core/qrss/pynum (sstvae_native.qrss.pynum) with int()
and float() on text holding characters past ASCII: CPython reads a str for
both through _PyUnicode_TransformDecimalAndSpaceToASCII, so a Unicode space
counts as a space and a Unicode decimal digit as its digit, and anything
else past ASCII makes the number invalid. test_qrss_native_cli.py holds the
ASCII rules.

The contract (the bar), and in brackets the worst measured (2026-10-14):
every code point (surrogates included, which reach the C++ as their
surrogatepass bytes) before, after and inside an ASCII number, as int()
base 10 and 16 and as float() read it: the same value (str(int), the same
float bit for bit) or the same refusal [exact, all 1,114,112].
"""

import pytest

from qrss_native import require


@pytest.fixture
def n(native):
    return require(native, "pynum")


def _int(t, base):
    try:
        return str(int(t, base))
    except ValueError:
        return None


def _float(t):
    try:
        return float(t).hex()
    except ValueError:
        return None


def _native_int(n, t, base):
    r = n.py_int(t, base)
    return None if r is None else r[0]


def _native_float(n, t):
    r = n.py_float(t)
    return None if r is None else r.hex()


@pytest.mark.parametrize("block", range(17), ids=lambda b: f"plane{b}")
def test_every_code_point(n, block):
    """chr(c) + "5", "5" + chr(c), "1" + chr(c) + "2", chr(c) alone and as
    an exponent's digit: int() base 10 and 16 and float(), plane by plane."""
    bad = []
    for c in range(block << 16, (block + 1) << 16):
        ch = chr(c)
        for t in (ch + "5", "5" + ch, "1" + ch + "2", ch, "1e" + ch):
            if _native_float(n, t) != _float(t):
                bad.append(("float", hex(c), t))
            if _native_int(n, t, 10) != _int(t, 10):
                bad.append(("int10", hex(c), t))
        t = "0x" + ch + "f"
        if _native_int(n, t, 16) != _int(t, 16):
            bad.append(("int16", hex(c), t))
    assert bad == [], bad[:20]


@pytest.mark.parametrize("text", [
    "١٢٣", "１_２", "　 -༣.༤e༢ ", " inf ",
    " nan", "๑๐" * 20, "۱." + "۹" * 30, "1²", "Ⅻ", "\x7f",
], ids=repr)
def test_mixed(n, text):
    assert _native_float(n, text) == _float(text)
    for base in (10, 16):
        assert _native_int(n, text, base) == _int(text, base)
        if _int(text, base) is None:
            try:
                int(text, base)
            except ValueError as e:
                assert n.invalid_int_literal(text, base) == str(e)

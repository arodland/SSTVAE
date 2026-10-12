"""C++ Morse keying (native/core/qrss/morse) against sstvae.qrss.morse.

Exact: integer keying, and refusals compared by their full text. A
callsign crosses into C++ as UTF-8 (lone surrogates as "surrogatepass"),
and qrss/pytext reproduces Python's character count, so the same text is
accepted and refused. The binding words a refusal with Python's own
repr() of the caller's object (a str subclass such as np.str_ quotes
itself differently), so its text is exact; the core's own text, which a
C++ caller sees, is pytext's and is compared too (`_core_refusal`). Skips
when the extension is not built; `pytest --native-qrss=tx` runs the rest
of the QRSS suites against the same code.
"""

import unicodedata

import numpy as np
import pytest

from qrss_native import require
from sstvae.qrss import morse as ref


@pytest.fixture(scope="module")
def nat(native, request):
    if request.config.getoption("--native-qrss", default=None):
        # The reference is then (partly) the C++ itself -- at stage tx even
        # precoder.unprecode's own scrambler is -- so there is nothing to
        # compare against. Run these without --native-qrss.
        pytest.skip("--native-qrss replaces the reference this file compares against")
    return require(native, "morse")


def _outcome(fn, arg):
    """('ok', result) or (exception type, message)."""
    try:
        out = fn(arg)
    except (ValueError, TypeError) as e:
        return type(e), str(e)
    if isinstance(out, np.ndarray):
        return "ok", out.dtype.str, out.shape, out.tobytes()
    return "ok", type(out), out


def test_table(nat):
    assert nat.MORSE == ref.MORSE
    assert list(nat.MORSE) == list(ref.MORSE)
    assert nat.MAX_CALL_CHARS == ref.MAX_CALL_CHARS


@pytest.mark.parametrize("ch", list(ref.MORSE))
def test_every_character(nat, ch):
    for fn in ("morse_keying", "keying_units", "check_callsign"):
        assert _outcome(getattr(nat, fn), ch) == _outcome(getattr(ref, fn), ch), fn
    k = nat.keying_units(ch)
    assert k.dtype == np.uint8 and k.shape == (192,) and k.flags.writeable


def test_pinned_calls(nat):
    import hashlib

    assert hashlib.sha256(nat.keying_units("K1ABC/P").tobytes()).hexdigest() == (
        "46c4fb4542fce939dffea24afa991d130d650e7bfca58a7e1735023750a83089")
    assert len(nat.morse_keying("00000000")) == 173
    assert nat.keying_units("00000000")[8 + 172] == 1
    for call in ("K1ABC", "K1ABC/P", "00000000", "E", "T", "Q/0", "VK2/G4XY", "W1AW   "):
        for fn in ("morse_keying", "keying_units", "check_callsign"):
            assert _outcome(getattr(nat, fn), call) == _outcome(getattr(ref, fn), call), (fn, call)


def _fuzz_alphabet():
    ascii_bits = "ABCXYZ0189/" + "akz" + " " + "-.'\"\\" + "\t\n\x00\x1f\x7f"
    other = [0xC4, 0xA0, 0xAD, 0x85, 0x130, 0x2028, 0x200B, 0xFEFF, 0x3000, 0xE000,
             0x1F600, 0xE0001, 0xDCFF, 0xD800, 0x378, 0x10FFFF]
    return list(ascii_bits) + [chr(c) for c in other]


def _unassigned_unescaped(outcome, call):
    """Python's refusal as pytext writes it: identical, except that an
    unassigned code point (category Cn) is printed as itself where Python's
    repr escapes it -- the one simplification in qrss/pytext.hpp."""
    if outcome[0] != ValueError:
        return outcome
    msg = outcome[1]
    for c in set(call):
        if unicodedata.category(c) == "Cn":
            msg = msg.replace(repr(c)[1:-1], c)
    return outcome[0], msg


def test_fuzzed_accept_and_reject_agree(nat):
    rng = np.random.default_rng(2026)
    alphabet = _fuzz_alphabet()
    valid = list(ref.MORSE)
    calls = ["", " ", "        ", "K1ABC/P", "k1abc", "K1 BC", "K1ABCDEFG", "K1-AB", "K1.AB",
             " K1ABC", "K1ABC" + " " * 20, "ABCDEFGH", "ABCDEFGHI", "/" * 8]
    for _ in range(4000):
        n = int(rng.integers(0, 11))
        pool = valid if rng.random() < 0.5 else alphabet
        s = "".join(pool[int(i)] for i in rng.integers(0, len(pool), n))
        if rng.random() < 0.2:
            s += " " * int(rng.integers(1, 4))
        if rng.random() < 0.1 and s:
            s = s.lower()
        calls.append(s)
    accepted = 0
    for call in calls:
        for fn in ("check_callsign", "morse_keying", "keying_units"):
            got, want = _outcome(getattr(nat, fn), call), _outcome(getattr(ref, fn), call)
            assert got == want, (fn, call)
        want = _outcome(ref.keying_units, call)
        core = nat._core_refusal(call)
        if want[0] == "ok":
            assert core is None, call
        else:
            assert (ValueError, core) == _unassigned_unescaped(want, call), call
        accepted += _outcome(ref.check_callsign, call)[0] == "ok"
    assert 500 < accepted < len(calls) - 500    # both sides of the boundary were exercised


@pytest.mark.parametrize("bad", [b"K1ABC", None, 5, ["K1ABC"], 1.5])
def test_non_str_is_refused_as_python_does(nat, bad):
    for fn in ("check_callsign", "morse_keying", "keying_units"):
        got, want = _outcome(getattr(nat, fn), bad), _outcome(getattr(ref, fn), bad)
        assert got == want, (fn, bad)
        assert got[0] is ValueError and "callsign must be a str" in got[1]


def test_str_subclasses_are_accepted(nat):
    class Call(str):
        pass

    for call in (np.str_("K1ABC"), Call("W1AW  ")):
        for fn in ("check_callsign", "keying_units"):
            assert _outcome(getattr(nat, fn), call) == _outcome(getattr(ref, fn), call)


def test_str_subclasses_are_refused_in_their_own_words(nat):
    """morse.py quotes `{callsign!r}`: repr of the caller's object, which
    np.str_ (numpy 2: "np.str_('k1abc')") and any str subclass with its
    own __repr__ word differently from a plain str."""
    class Call(str):
        def __repr__(self):
            return "<Call>"

    class Plain(str):
        pass

    calls = [np.str_("k1abc"), np.str_("K1ABCDEFGH"), np.str_(""), np.str_("K\u0378"),
             Call("k1"), Call(""), Call("K1ABCDEFGHI"), Plain("k1 "), Plain("\udcff")]
    for call in calls:
        for fn in ("check_callsign", "morse_keying", "keying_units"):
            got, want = _outcome(getattr(nat, fn), call), _outcome(getattr(ref, fn), call)
            assert want[0] is ValueError
            assert got == want, (fn, call)


def test_refusals_of_unassigned_code_points_are_exact(nat):
    """Python's repr escapes an unassigned code point; the binding words
    the refusal with that repr, so pytext's simplification never shows."""
    for call, escaped in (("K\u0378", "\\u0378"), ("\U000e0080AB", "\\U000e0080"),
                          ("A\U0010ffff", "\\U0010ffff"), ("K1\u0378 ", "\\u0378")):
        for fn in ("check_callsign", "keying_units"):
            got, want = _outcome(getattr(nat, fn), call), _outcome(getattr(ref, fn), call)
            assert want[0] is ValueError and escaped in want[1], (fn, call, want)
            assert got == want, (fn, call, got)


def test_repr_matches_python(nat):
    rng = np.random.default_rng(5)
    alphabet = _fuzz_alphabet() + ["'", '"', "\r", "\\", chr(0xFF), chr(0x100), chr(0xFFFF), chr(0x10000)]
    samples = ["", "'", '"', "'\"", "it's", 'say "hi"', "\\", "\r\n\t"]
    for _ in range(3000):
        n = int(rng.integers(0, 8))
        samples.append("".join(alphabet[int(i)] for i in rng.integers(0, len(alphabet), n)))
    for s in samples:
        if any(unicodedata.category(c) == "Cn" for c in s):
            continue   # unassigned: Python escapes them, pytext (by design) does not
        assert nat._repr(s) == repr(s), s


def test_isprintable_matches_python_on_every_assigned_code_point(nat):
    cps = np.arange(0x110000, dtype=np.uint32)
    got = nat._isprintable(cps)
    want = np.array([chr(c).isprintable() for c in range(0x110000)])
    assigned = np.array([unicodedata.category(chr(c)) != "Cn" for c in range(0x110000)])
    assert np.array_equal(got[assigned], want[assigned])
    assert got[~assigned].all()   # the stated simplification: unassigned reads as printable

"""C++ slot clock (native/core/qrss/slot) against sstvae.qrss.sequences.

parse_utc is datetime.fromisoformat underneath, so the C++ is a port of
CPython 3.13's C parser (Modules/_datetimemodule.c) and is compared with
the running interpreter's. Two of its corners moved between 3.12 and 3.13
-- a fraction mark with no digit after it ("06:00:00.Z") and an offset's
microseconds, which 3.12 drops -- so those rows, and the fuzz that could
reach them, compare only on 3.13 (the interpreter the port was written
against); native/tests/qrss/test_slot.cpp pins 3.13's answers either way.
"""

import random
import sys
from datetime import datetime, timedelta, timezone

import pytest

from qrss_native import require
from sstvae.qrss import sequences as ref

Q_TEST = 1954752
PY313 = sys.version_info[:2] == (3, 13)


@pytest.fixture(scope="module")
def nat(native):
    return require(native, "slot")


CASES = [
    # read
    "2026-10-09T06:00Z", "2026-10-09T06:00z", "2026-10-09t06:00Z", "2026-10-09 06:00Z",
    " 2026-10-09T06:00Z\n", "　2026-10-09T06:00Z\x85", "\t2026-10-09T06:00Z\x1c",
    "2026-10-09T06:00:00Z", "2026-10-09T06:00:00.000Z", "2026-10-09T06:00:00,000Z",
    "2026-10-09T06:00+00:00", "2026-10-09T06:00-00:00", "2026-10-09T08:00+02:00",
    "2026-10-09T01:30-04:30", "2026-10-09T06:00+0000", "2026-10-09T08:00+02",
    "20261009T0600Z", "2026-10-09T0600Z", "2026-10-09T06Z", "2026-W41-5T06:00Z",
    "2026W415T0600Z", "2026-W41T06:00Z", "2026W41T06Z", "2026-10-09T06:00+05:45",
    "2026-10-09é06:00Z", "2026-10-09€06:00Z", "2026-10-09\U0001f60006:00Z",
    "2026-10-09T06:00:00:00Z", "2026-10-09T06:00 Z", "2025-10-01T00:00Z", "1970-01-01T00:00Z",
    "1969-12-31T23:45Z", "0001-01-01T00:00+01:00", "9999-12-31T23:45-01:00",
    "2024-02-29T00:00Z", "2026-W53-1T00:00Z", "2020-W53-1T00:00Z", "2026-10-09T06:00:00.000000000Z",
    # off the quarter hour
    "2026-10-09T06:07Z", "2026-10-09T06:00:30Z", "2026-10-09T06:00:00.5Z",
    "2026-10-09T06:00+05:30:15", "2026-10-09T06:00:00.1234567Z", "2026-10-09T06:00+00:99",
    "2026-10-09T06.5Z", "2026-10-09T06:00+23:59:59.999999", "2026-10-09T06:00-05:00:30",
    "1969-12-31T23:59:59.999999Z", "2026-10-09T06:00:00.000001Z",
    # no zone
    "2026-10-09T06:00", "2026-10-09", "2026-10-09Z", "2025-10-01 00:00", "2026W41",
    # not ISO 8601
    "", "   ", "x", "2026-10-09T24:00Z", "2026-13-01T00:00Z", "2026-02-29T00:00Z",
    "2026-10-09T06:00+24:00", "2026-10-09T06:00ZZ", "2026-10-09T6:00Z", "0000-01-01T00:00Z",
    "2026-10-09T06:00Zjunk", "2026-10-9T06:00Z", "2026-1009T06:00Z", "2026-W54-1T00:00Z",
    "2026-W41-8T00:00Z", "2026-10-09T06:60Z", "2026-10-09T06:00:61Z", "2026-10-09T06:00+05:3",
    "2026-10-09T06:00+", "2026-10-09T06:00Z+", "２０２６-10-09T06:00Z", "2026-10-09T06:00:00.+00:00",
    "2026-10-09T٠٦:00Z",
    # lone surrogates: Python's own argv decoding (surrogateescape) of bad UTF-8
    "2026-10-09\udcff06:00Z", "2026-10-09T06:00Z\udcff", "2026-10\udcff09T06:00Z",
    "2026-10-09T06:00\ud800Z", "\udcff\udcff\udcff\udcff\udcff\udcff\udcff\udcff",
    "2026-10\udcff\udcffT06:00Z",
]

# Rows whose answer moved between CPython 3.12 and 3.13 (see the docstring).
VERSION_DEPENDENT = {
    "2026-10-09T06:00:00.+00:00",
    "2026-10-09T06:00:00+00:00:00.000001",
    "2026-10-09T06:00-00:00:00.000001",
    "2026-10-09T06:00:00.Z",
}
CASES += sorted(VERSION_DEPENDENT - set(CASES))


def outcome(f, *a):
    try:
        return ("ok", f(*a))
    except ValueError as e:
        return ("ValueError", str(e))


def same_datetime(a, b):
    return (a == b and a.isoformat() == b.isoformat() and type(a) is type(b)
            and a.utcoffset() == b.utcoffset() and (a.tzinfo is timezone.utc) == (b.tzinfo is timezone.utc))


@pytest.mark.parametrize("text", CASES, ids=ascii)
def test_parse_slot(nat, text):
    if text in VERSION_DEPENDENT and not PY313:
        pytest.skip("CPython 3.12 parses this differently; the port follows 3.13")
    assert outcome(nat.parse_slot, text) == outcome(ref.parse_slot, text)


@pytest.mark.parametrize("text", CASES, ids=ascii)
def test_parse_utc(nat, text):
    if text in VERSION_DEPENDENT and not PY313:
        pytest.skip("CPython 3.12 parses this differently; the port follows 3.13")
    got, want = outcome(nat.parse_utc, text), outcome(ref.parse_utc, text)
    assert got[0] == want[0], (got, want)
    if want[0] == "ok":
        assert same_datetime(got[1], want[1]), (got[1], want[1])
    else:
        assert got == want


def _fields(d):
    off = d.utcoffset()
    us = None if off is None else (off.days * 86400 + off.seconds) * 1_000_000 + off.microseconds
    return (d.year, d.month, d.day, d.hour, d.minute, d.second, d.microsecond, us)


def _reference_fromisoformat(s):
    try:
        return _fields(datetime.fromisoformat(s))
    except ValueError:
        return None


PIECES = ["2026", "-", "10", "09", "T", "t", " ", "06", ":", "00", ".", ",", "5", "123456789",
          "+", "-", "05", "30", "Z", "W", "41", "1", "0", "9", "é", "\udcff", "　", "24",
          "99", "1234", "+00:00", "-00:00:00.000001", "\U0001f600", "\x00"]
SEEDS = ["2026-10-09T06:00:00.123456+05:30", "2026-W41-5T06:00Z", "20261009T060000,5-0430",
         "2026-10-09 06:00", "0001-01-01T00:00:00+23:59:59.999999", "9999-12-31T23:59:59.999999-01:00"]


def _fuzz_strings(n, seed=20261013):
    rng = random.Random(seed)
    out = []
    for i in range(n):
        if i % 2:
            out.append("".join(rng.choice(PIECES) for _ in range(rng.randint(0, 12))))
            continue
        s = list(rng.choice(SEEDS))
        for _ in range(rng.randint(1, 3)):
            k = rng.randrange(len(s) + 1)
            op = rng.randrange(3)
            if op == 0 and s and k < len(s):
                del s[k]
            elif op == 1:
                s.insert(k, rng.choice(PIECES))
            elif s and k < len(s):
                s[k] = rng.choice("0123456789-:T.,+Z Wé")
        out.append("".join(s))
    return out


@pytest.mark.skipif(not PY313, reason="the port is CPython 3.13's parser; other versions differ in corners")
def test_fromisoformat_fuzz(nat):
    bad = []
    for s in _fuzz_strings(20000):
        want = _reference_fromisoformat(s)
        got = nat.fromisoformat(s)
        if got != want:
            bad.append((s, got, want))
    assert not bad, bad[:10]


def test_quarter_hour_count(nat):
    t = datetime(2025, 10, 1, tzinfo=timezone.utc)
    cases = [t, t + timedelta(minutes=15), t - timedelta(minutes=15),
             datetime(2025, 10, 1, 2, tzinfo=timezone(timedelta(hours=2))),
             datetime(2025, 9, 30, 19, 15, tzinfo=timezone(timedelta(hours=-4, minutes=-45))),
             datetime(1969, 12, 31, 23, 45, tzinfo=timezone.utc),
             datetime(1, 1, 1, 1, tzinfo=timezone(timedelta(hours=1))),
             datetime(9999, 12, 31, 23, 45, tzinfo=timezone.utc),
             t + timedelta(seconds=1), t + timedelta(minutes=7), t + timedelta(microseconds=1),
             datetime(2025, 10, 1, tzinfo=timezone(timedelta(microseconds=1))),
             datetime(2025, 10, 1), datetime(2025, 10, 1, 6, 0, 0, 1)]
    for d in cases:
        assert outcome(nat.quarter_hour_count, d) == outcome(ref.quarter_hour_count, d), d


def test_quarter_hour_start(nat):
    for q in (Q_TEST, 0, -1, 1990584, -69039552, 281558111, Q_TEST + 0.9, True):
        got, want = nat.quarter_hour_start(q), ref.quarter_hour_start(q)
        assert same_datetime(got, want), q
        assert nat.quarter_hour_count(got) == ref.quarter_hour_count(want)
    # Past years 1..9999 the refusal's text depends on how far: the date,
    # then timedelta's day limit, then its days past a C int.
    big = 2**31 * 96
    for q in (281558112, -69039553, 95999999999, 96000000095, -95999999999, -96000000000, big - 1, -big,
              big, -big - 1, 2**63 - 1, -2**63, 2**63, -2**63 - 1, 10**30, 1.5e300):
        want, got = outcome_any(ref.quarter_hour_start, q), outcome_any(nat.quarter_hour_start, q)
        assert want[0] == "OverflowError" and got == want, (q, got, want)


def outcome_any(f, *a):
    try:
        return ("ok", f(*a))
    except Exception as e:
        return (type(e).__name__, str(e))


def test_slot_iso(nat):
    for q in (Q_TEST, 0, -1, 1990585, -69039552, 281558111):
        assert nat.slot_iso(q) == ref.quarter_hour_start(q).isoformat()[:16] + "Z"
        assert nat.parse_slot(nat.slot_iso(q)) == q

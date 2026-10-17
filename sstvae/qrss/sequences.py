"""SHA-256 bit streams: the CE and L preambles, their reference symbols,
the per-pass scrambler (design section 2.3) and the spread header's
whitener (spec 5.1).

A **format module**. Every sequence here is closed form over integers --
SHA-256 of an ASCII domain string, an optional prefix and a big-endian
block counter -- so two stations on different numpy versions (or a
microcontroller with a 200-line SHA-256) cannot disagree about it. That
is the rule SSTVAE learned with its interleaver (`config.py`, "frozen
format constants"): no seeded generator ever defines what goes on air.

Bit conventions (decision D5): block b is SHA256(domain || prefix ||
uint32_be(b)); bit i is bit (7 - i%8) of byte (i%256)//8 of block i//256,
i.e. MSB first; and bit 0 maps to +1, bit 1 to -1. Waveform L takes the
bits in pairs, the first to I and the second to Q, each scaled by 1/sqrt(2)
(spec 2.3): `qpsk`.
"""

import functools
import hashlib
import struct
from datetime import datetime, timedelta, timezone

import numpy as np

from .constants import L_N_PRE, N_PRE

PREAMBLE_DOMAIN = b"QRSSTVAE CE preamble"
REFERENCE_DOMAIN = b"QRSSTVAE CE reference"
SCRAMBLE_DOMAIN = b"QRSSTVAE scramble"
SPREAD_DOMAIN = b"QRSSTVAE CE spread header"   # spec 5.1, format version 2
L_PREAMBLE_DOMAIN = b"QRSSTVAE L preamble"
L_REFERENCE_DOMAIN = b"QRSSTVAE L reference"

_BITS_PER_BLOCK = 256


def sha_bits(domain: bytes, n: int, prefix: bytes = b"") -> np.ndarray:
    """The first `n` bits of the stream SHA256(domain || prefix || u32be(b)).

    Returns uint8 0/1, MSB first within each byte.
    """
    if n < 0:
        raise ValueError(f"n must be >= 0, not {n}")
    n_blocks = -(-n // _BITS_PER_BLOCK)
    head = bytes(domain) + bytes(prefix)
    digest = b"".join(hashlib.sha256(head + struct.pack(">I", b)).digest()
                      for b in range(n_blocks))
    bits = np.unpackbits(np.frombuffer(digest, dtype=np.uint8))  # MSB first
    return bits[:n].copy()


def pm1(bits) -> np.ndarray:
    """0 -> +1, 1 -> -1, as int8."""
    return (1 - 2 * np.asarray(bits, dtype=np.int8)).astype(np.int8)


@functools.lru_cache(maxsize=None)
def preamble_ce() -> np.ndarray:
    """The CE preamble, int8 +-1, N_PRE = 660 values. Read-only."""
    out = pm1(sha_bits(PREAMBLE_DOMAIN, N_PRE))
    out.setflags(write=False)
    return out


@functools.lru_cache(maxsize=8)
def _references(n_ref: int) -> np.ndarray:
    out = pm1(sha_bits(REFERENCE_DOMAIN, n_ref))
    out.setflags(write=False)
    return out


def references_ce(n_ref: int) -> np.ndarray:
    """Reference symbols, int8 +-1; reference m (in time order) takes bit m.

    One stream for the whole frame, header references first, so a
    shorter frame's references are a prefix of a longer one's.
    """
    return _references(int(n_ref))


def qpsk(bits) -> np.ndarray:
    """complex128: bits 2i and 2i+1 to I and Q of symbol i, 0 -> +1, scaled by 1/sqrt(2)."""
    b = np.asarray(bits)
    if b.ndim != 1 or len(b) % 2:
        raise ValueError(f"qpsk takes an even number of bits, not shape {b.shape}")
    v = pm1(b).astype(np.float64) / np.sqrt(2.0)
    return v[0::2] + 1j * v[1::2]


@functools.lru_cache(maxsize=None)
def preamble_l() -> np.ndarray:
    """The L preamble, L_N_PRE = 330 QPSK symbols (complex128, unit power). Read-only."""
    out = qpsk(sha_bits(L_PREAMBLE_DOMAIN, 2 * L_N_PRE))
    out.setflags(write=False)
    return out


@functools.lru_cache(maxsize=8)
def _references_l(n_ref: int) -> np.ndarray:
    out = qpsk(sha_bits(L_REFERENCE_DOMAIN, 2 * n_ref))
    out.setflags(write=False)
    return out


def references_l(n_ref: int) -> np.ndarray:
    """L reference symbols, complex QPSK; reference m takes bits 2m and 2m+1.

    One stream for the whole frame, header references first, as in CE.
    """
    return _references_l(int(n_ref))


@functools.lru_cache(maxsize=8)
def _spread_whitener(n: int) -> np.ndarray:
    out = pm1(sha_bits(SPREAD_DOMAIN, n))
    out.setflags(write=False)
    return out


def spread_whitener(n: int) -> np.ndarray:
    """Sign flips of the spread header (spec 5.1, docs/qrss/spread-header.md),
    int8 +-1, indexed by data symbol. Not keyed by q: the spread header is
    the same in every pass, so its LLRs add across passes."""
    return _spread_whitener(int(n))


def scrambler(q: int, n: int) -> np.ndarray:
    """Per-pass sign flips c_q, int8 +-1, indexed by data latent in air order.

    Keyed by the slot's quarter-hour count q as uint64_be, so every pass
    of the same picture sends different signs (spec 2.2) while sender
    and receiver need nothing but the clock. FULL uses 198 hash blocks.
    """
    q = int(q)
    if not 0 <= q < 1 << 64:
        raise ValueError(f"q must fit in uint64, not {q}")
    return pm1(sha_bits(SCRAMBLE_DOMAIN, n, struct.pack(">Q", q)))


_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


def quarter_hour_count(utc: datetime) -> int:
    """q = floor(unix(QH)/900) for an aware datetime exactly on a quarter hour.

    Raises ValueError for a naive datetime or one off the quarter hour,
    because a slot that does not start on one is not a slot. Unix time
    ignores leap seconds, and so does q.
    """
    if utc.tzinfo is None or utc.tzinfo.utcoffset(utc) is None:
        raise ValueError("quarter_hour_count needs an aware datetime (UTC)")
    delta = utc - _EPOCH
    if delta % timedelta(seconds=900):
        raise ValueError(f"{utc.isoformat()} is not on a quarter hour")
    return delta // timedelta(seconds=900)


def parse_utc(s: str) -> datetime:
    """An aware datetime from ISO 8601 text with a zone, e.g. 2026-10-09T06:00Z.

    A time without Z or an offset is refused (ValueError) rather than read
    as local time.
    """
    try:
        dt = datetime.fromisoformat(s.strip().replace("Z", "+00:00").replace("z", "+00:00"))
    except ValueError:
        raise ValueError("not an ISO 8601 time; expected e.g. 2026-10-09T06:00Z") from None
    if dt.tzinfo is None or dt.tzinfo.utcoffset(dt) is None:
        raise ValueError("give the time zone: end it with Z for UTC, e.g. 2026-10-09T06:00Z")
    return dt


def parse_slot(s: str) -> int:
    """q of a quarter hour given as ISO 8601 UTC text, e.g. 2026-10-09T06:00Z (ValueError).

    A quarter hour before 1970 (q < 0) is no slot: the scrambler's seed and
    the frame clock count from the epoch, and a receive refuses it anyway,
    after the work of reading the capture. It is refused here, by its UTC
    time ("slot 1969-12-31T23:45Z is before 1970-01-01T00:00Z"), or by the
    time as read (its isoformat()) where that UTC time is before year 1.
    """
    dt = parse_utc(s)
    q = quarter_hour_count(dt)
    if q < 0:
        try:
            when = quarter_hour_start(q).isoformat()[:16] + "Z"
        except OverflowError:
            when = dt.isoformat()
        raise ValueError(f"slot {when} is before 1970-01-01T00:00Z")
    return q


def quarter_hour_start(q: int) -> datetime:
    """The UTC datetime of quarter hour `q` (inverse of quarter_hour_count)."""
    return _EPOCH + timedelta(seconds=900 * int(q))

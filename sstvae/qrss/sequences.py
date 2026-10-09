"""SHA-256 bit streams: the CE preamble, the reference symbols and the
per-pass scrambler (design section 2.3).

A **format module**. Every sequence here is closed form over integers --
SHA-256 of an ASCII domain string, an optional prefix and a big-endian
block counter -- so two stations on different numpy versions (or a
microcontroller with a 200-line SHA-256) cannot disagree about it. That
is the rule SSTVAE learned with its interleaver (`config.py`, "frozen
format constants"): no seeded generator ever defines what goes on air.

Bit conventions (decision D5): block b is SHA256(domain || prefix ||
uint32_be(b)); bit i is bit (7 - i%8) of byte (i%256)//8 of block i//256,
i.e. MSB first; and bit 0 maps to +1, bit 1 to -1.
"""

import functools
import hashlib
import struct
from datetime import datetime, timedelta, timezone

import numpy as np

from .constants import N_PRE

PREAMBLE_DOMAIN = b"QRSSTVAE CE preamble"
REFERENCE_DOMAIN = b"QRSSTVAE CE reference"
SCRAMBLE_DOMAIN = b"QRSSTVAE scramble"
L_PREAMBLE_DOMAIN = b"QRSSTVAE L preamble"   # reserved for waveform L

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


def quarter_hour_start(q: int) -> datetime:
    """The UTC datetime of quarter hour `q` (inverse of quarter_hour_count)."""
    return _EPOCH + timedelta(seconds=900 * int(q))

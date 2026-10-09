"""The QRSSTVAE header: 142 bits, polar coded to 2474 (design section 2.6).

This is a **format module**. The header is identical in every pass of a
segment, so a receiver can add the header LLRs of passes it could not
decode alone and try again (soft combining; `decode(sum_of_llrs)`).

Fields, in table order, each MSB first:

    bits   0-47   callsign      8 x 6 bits, SSTVAE's beacon alphabet, space padded
    bits  48-62   grid          ((F1*18 + F2)*10 + D1)*10 + D2; 32767 = none
    bits  63-94   picture ID    32 bits
    bits  95-96   mode          A = 0, B = 1, C = 2
    bits  97-98   segment       0..mode, the latent group
    bits  99-114  codec ID      16 bits
    bits 115-118  version       FORMAT_VERSION (1)
    bits 119-125  reserved      0
    bits 126-141  CRC           `crc16(bits[0:126])`

**The CRC is SSTVAE's beacon CRC bit for bit, and it is not
CRC-16/CCITT-FALSE** despite what `beacon._crc16`'s docstring and the
spec say: it shifts the message bits into the register rather than
XORing them into the top, so its check value over ASCII "123456789" is
0xA69D, where CCITT-FALSE gives 0x29B1 (decision D8). Reusing the exact
function is the point, so it is wrapped here, once, rather than
re-implemented.

QRSS callsigns are restricted to [A-Z0-9/], 1 to 8 characters, and
anything else raises instead of being mapped to a space as SSTVAE's
beacon does: every character must also have a Morse code for the
callsign windows (spec 2.7).

A decode is accepted only when the CRC matches **and** version = 1,
reserved = 0, mode <= 2, segment <= mode, the grid code is valid and the
callsign is a valid QRSS call. Eight list paths against a 16-bit CRC
would otherwise accept noise about once in 8,000 attempts (decision D9).
"""

import re
from dataclasses import dataclass

import numpy as np

from ..modem import beacon as _beacon
from .constants import FORMAT_VERSION, N_INFO_BITS
from .polar import INFO_SET, polar_decode, polar_encode  # noqa: F401  (re-exported)

# Field widths in table order; offsets follow.
_FIELDS = (("callsign", 48), ("grid", 15), ("picture_id", 32), ("mode", 2),
           ("segment", 2), ("codec_id", 16), ("version", 4), ("reserved", 7),
           ("crc", 16))
_OFFSET = {}
_o = 0
for _name, _w in _FIELDS:
    _OFFSET[_name] = (_o, _o + _w)
    _o += _w
assert _o == N_INFO_BITS == 142 and _OFFSET["crc"] == (126, 142)
del _o, _name, _w

CALL_CHARS = 8
GRID_NONE = 32767
MAX_MODE = 2
_CALL_RE = re.compile(r"[A-Z0-9/]{1,8}")
_GRID_RE = re.compile(r"[A-R]{2}[0-9]{2}")


# --- SSTVAE private names, wrapped once ----------------------------------------


def _int_to_bits(value: int, width: int) -> np.ndarray:
    return _beacon._int_to_bits(int(value), width).astype(np.uint8)


def _bits_to_int(bits: np.ndarray) -> int:
    return _beacon._bits_to_int(bits)


def crc16(bits: np.ndarray) -> np.ndarray:
    """SSTVAE's beacon CRC over a 0/1 bit array -> 16 bits, uint8, MSB first."""
    return _beacon._crc16(np.asarray(bits)).astype(np.uint8)


_ALPHABET = _beacon._ALPHABET


# --- fields ---------------------------------------------------------------------


@dataclass(frozen=True)
class HeaderFields:
    callsign: str
    grid: str | None
    picture_id: int
    mode: int
    segment: int
    codec_id: int
    version: int = FORMAT_VERSION
    reserved: int = 0


def grid_encode(g: str | None) -> int:
    """4-character Maidenhead locator -> 15-bit code; None -> 32767."""
    if g is None:
        return GRID_NONE
    if not isinstance(g, str) or not _GRID_RE.fullmatch(g.upper()):
        raise ValueError(f"grid must be a 4-character Maidenhead locator, got {g!r}")
    g = g.upper()
    f1, f2 = ord(g[0]) - ord("A"), ord(g[1]) - ord("A")
    return ((f1 * 18 + f2) * 10 + int(g[2])) * 10 + int(g[3])


def grid_decode(v: int) -> str | None:
    """15-bit code -> locator; 32767 -> None. ValueError for the unused codes."""
    v = int(v)
    if v == GRID_NONE:
        return None
    if not 0 <= v < 18 * 18 * 100:
        raise ValueError(f"invalid grid code {v}")
    f, d = divmod(v, 100)
    return chr(ord("A") + f // 18) + chr(ord("A") + f % 18) + f"{d // 10}{d % 10}"


def _check_fields(h: HeaderFields) -> None:
    if not isinstance(h.callsign, str) or not _CALL_RE.fullmatch(h.callsign):
        raise ValueError(f"callsign must be 1-8 characters of [A-Z0-9/], got {h.callsign!r}")
    for name, hi in (("picture_id", 1 << 32), ("codec_id", 1 << 16)):
        if not 0 <= int(getattr(h, name)) < hi:
            raise ValueError(f"{name} out of range: {getattr(h, name)}")
    if not 0 <= h.mode <= MAX_MODE:
        raise ValueError(f"mode must be 0..{MAX_MODE}, got {h.mode}")
    if not 0 <= h.segment <= h.mode:
        raise ValueError(f"segment must be 0..mode ({h.mode}), got {h.segment}")
    if h.version != FORMAT_VERSION:
        raise ValueError(f"version must be {FORMAT_VERSION}, got {h.version}")
    if h.reserved != 0:
        raise ValueError(f"reserved must be 0, got {h.reserved}")


def pack(h: HeaderFields) -> np.ndarray:
    """HeaderFields -> 142 bits (uint8, table order, CRC last).

    Refuses (ValueError) anything `unpack` would reject, so a transmitter
    can never send a header that receivers throw away.
    """
    _check_fields(h)
    codes = _beacon.callsign_to_codes(h.callsign)
    parts = [_int_to_bits(c, 6) for c in codes]
    parts += [_int_to_bits(grid_encode(h.grid), 15),
              _int_to_bits(h.picture_id, 32),
              _int_to_bits(h.mode, 2),
              _int_to_bits(h.segment, 2),
              _int_to_bits(h.codec_id, 16),
              _int_to_bits(h.version, 4),
              _int_to_bits(h.reserved, 7)]
    body = np.concatenate(parts)
    return np.concatenate([body, crc16(body)])


def _field(bits: np.ndarray, name: str) -> int:
    a, b = _OFFSET[name]
    return _bits_to_int(bits[a:b])


def unpack(bits: np.ndarray) -> HeaderFields | None:
    """142 bits -> HeaderFields, or None if the CRC or any field check fails."""
    bits = np.asarray(bits)
    if bits.shape != (N_INFO_BITS,) or np.any((bits != 0) & (bits != 1)):
        return None
    bits = bits.astype(np.uint8)
    if not np.array_equal(crc16(bits[:126]), bits[126:]):
        return None
    version, reserved = _field(bits, "version"), _field(bits, "reserved")
    mode, segment = _field(bits, "mode"), _field(bits, "segment")
    if version != FORMAT_VERSION or reserved != 0 or mode > MAX_MODE or segment > mode:
        return None
    a, _ = _OFFSET["callsign"]
    raw = "".join(_ALPHABET[_bits_to_int(bits[a + 6 * i:a + 6 * i + 6])]
                  for i in range(CALL_CHARS))
    call = raw.rstrip(" ")
    if not _CALL_RE.fullmatch(call):        # also rejects an internal space
        return None
    try:
        grid = grid_decode(_field(bits, "grid"))
    except ValueError:
        return None
    return HeaderFields(callsign=call, grid=grid,
                        picture_id=_field(bits, "picture_id"), mode=mode,
                        segment=segment, codec_id=_field(bits, "codec_id"),
                        version=version, reserved=reserved)


# --- coded header ---------------------------------------------------------------


def encode(h: HeaderFields) -> np.ndarray:
    """HeaderFields -> 2474 coded bits (uint8). Bit e rides header symbol e as +1 for 0."""
    return polar_encode(pack(h))


def decode(llr: np.ndarray, list_size: int = 8) -> HeaderFields | None:
    """2474 coded-bit LLRs (log P(0)/P(1)) -> HeaderFields, or None.

    For soft combining, pass the sum of several passes' LLRs: the header
    is identical in every pass, so the sum is one pass at N times the SNR.
    """
    bits = polar_decode(llr, list_size)
    return None if bits is None else unpack(bits)

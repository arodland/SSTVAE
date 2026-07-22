"""Extended Golay (24,12) code, systematic, with brute-force soft decoding.

Encoding uses the cyclic (23,12) generator polynomial 0xC75
(x^11+x^10+x^6+x^5+x^4+x^2+1) plus an overall parity bit. The decoder
correlates soft values against all 4096 codewords, which is trivially
fast and gives true maximum-likelihood performance.
"""

import numpy as np

_POLY = 0xC75  # degree 11


def _mod2div_remainder(value: int) -> int:
    """Remainder of value / _POLY over GF(2)."""
    for shift in range(value.bit_length() - 12, -1, -1):
        if value & (1 << (shift + 11)):
            value ^= _POLY << shift
    return value


def encode(data12: int) -> int:
    """12 info bits -> 24-bit codeword (data in high bits, parity last)."""
    assert 0 <= data12 < 4096
    shifted = data12 << 11
    cw23 = shifted | _mod2div_remainder(shifted)
    parity = bin(cw23).count("1") & 1
    return (cw23 << 1) | parity


_CODEWORDS = np.array([encode(m) for m in range(4096)], dtype=np.int64)
# (4096, 24) matrix of +/-1 signs, bit 23 (MSB) first
_SIGNS = 1.0 - 2.0 * (
    (_CODEWORDS[:, None] >> np.arange(23, -1, -1)[None, :]) & 1
).astype(np.float64)


def codeword_bits(data12: int) -> np.ndarray:
    """24-bit codeword as an array of 0/1, MSB first."""
    cw = encode(data12)
    return (cw >> np.arange(23, -1, -1)) & 1


def decode_soft(soft: np.ndarray) -> int:
    """ML-decode 24 soft values (positive => bit 0) to the 12 info bits."""
    scores = _SIGNS @ soft
    return int(np.argmax(scores))


def min_distance() -> int:
    """Minimum distance of the code (should be 8). Used by tests."""
    weights = [bin(int(c)).count("1") for c in _CODEWORDS[1:]]
    return min(weights)

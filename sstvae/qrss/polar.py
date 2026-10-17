"""Polar (2048, 142) code for the QRSSTVAE header (design section 2.6).

This is a **format module**: the encoder and `INFO_SET` define what goes
on the air. The decoder is not on-air format, but lives here so the two
are tested against each other.

    mother code   x = u . F^(x)11 (mod 2), F = [[1, 0], [1, 1]],
                  natural order, no bit reversal:  x_j = XOR_{i superset of j} u_i
    info bits     header bit t -> u[INFO_SET[t]]; every other u is frozen at 0
    rate matching circular repetition, coded bit e = x[e mod 2048], e < 2474,
                  so mother bits 0..425 go out twice

**`INFO_SET` is committed literal data, and the literal is the
definition.** It came from Gaussian-approximation density evolution at
Es/N0 = -9.4 dB per coded bit (SNR_2500 = -23.5 dB, the spec's header
threshold), following the exact repetition pattern; `tools/gen_qrss_polar.py`
regenerates it and `--check`s it against this tuple. Should that tool
ever disagree on some platform, the tool is wrong, not the format.

Decoding is CRC-aided successive-cancellation list decoding (CA-SCL):

1. Fold the coded LLRs onto the mother code, L_i = sum_{e = i mod 2048} L_e.
2. SCL with list size L (8), min-sum f, exact g, and the LLR-based path
   metric PM += log(1 + exp(-(1 - 2u) lambda)), vectorized over the list.
   A subtree whose leaves are all frozen (rate 0) is not descended: its
   decisions are all zero, and its metric is the same softplus summed over
   the subtree's input LLRs -- the likelihood that all its outputs are 0.
   Half the mother code (indices below 1015) is one such subtree.
3. Return the most likely surviving path that the caller's acceptance test
   passes (by default `header.unpack`: CRC plus the field checks).

LLR convention throughout: llr = log P(b=0)/P(b=1), so a positive value
means bit 0, which header symbol e sends as +1.
"""

import hashlib
from functools import lru_cache

import numpy as np

from .constants import N_HDR_BITS, N_INFO_BITS, POLAR_N

N = POLAR_N                 # 2048 mother-code length
K = N_INFO_BITS             # 142 information bits, CRC included
E = N_HDR_BITS              # 2474 coded bits on the air
N_STAGES = N.bit_length() - 1   # 11
assert 1 << N_STAGES == N

INFO_SET = (1015, 1019, 1021, 1022, 1023, 1503, 1519, 1527, 1530, 1531, 1532, 1533, 1534, 1535, 1663,
 1727, 1759, 1771, 1773, 1774, 1775, 1779, 1781, 1782, 1783, 1785, 1786, 1787, 1788, 1789, 1790, 1791,
 1853, 1854, 1855, 1879, 1883, 1885, 1886, 1887, 1895, 1899, 1901, 1902, 1903, 1907, 1909, 1910, 1911,
 1912, 1913, 1914, 1915, 1916, 1917, 1918, 1919, 1935, 1943, 1947, 1949, 1950, 1951, 1959, 1961, 1962,
 1963, 1964, 1965, 1966, 1967, 1969, 1970, 1971, 1972, 1973, 1974, 1975, 1976, 1977, 1978, 1979, 1980,
 1981, 1982, 1983, 1989, 1990, 1991, 1993, 1994, 1995, 1996, 1997, 1998, 1999, 2001, 2002, 2003, 2004,
 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2017, 2018, 2019, 2020, 2021, 2022,
 2023, 2024, 2025, 2026, 2027, 2028, 2029, 2030, 2031, 2032, 2033, 2034, 2035, 2036, 2037, 2038, 2039,
 2040, 2041, 2042, 2043, 2044, 2045, 2046, 2047)

# sha256 of the literal as big-endian uint16, as printed in the design.
INFO_SET_SHA256 = "4e064eee9b1df68ee4d044970b51f05a16700d51599f857f86e7d15d782b97bb"

_INFO = np.array(INFO_SET, dtype=np.intp)
assert len(INFO_SET) == K and len(set(INFO_SET)) == K
assert list(INFO_SET) == sorted(INFO_SET) and 0 <= _INFO[0] and _INFO[-1] < N
assert hashlib.sha256(np.array(INFO_SET, ">u2").tobytes()).hexdigest() == INFO_SET_SHA256

# Coded bit e carries mother bit _RM[e].
_RM = np.arange(E, dtype=np.intp) % N


# --- encoder -------------------------------------------------------------------


def polar_transform(v: np.ndarray) -> np.ndarray:
    """v . F^(x)n (mod 2) over the last axis, natural order. Self-inverse.

    F^(x)n = [[G, 0], [G, G]] with G = F^(x)(n-1), so [a, b] maps to
    [aG + bG, bG]: at each stage the first half of every block takes the
    XOR of the second. The stages commute, so their order is free.
    """
    x = np.array(v, dtype=np.uint8, copy=True)
    n = x.shape[-1]
    lead = x.shape[:-1]
    h = n // 2
    while h >= 1:
        y = x.reshape(lead + (n // (2 * h), 2, h))
        y[..., 0, :] ^= y[..., 1, :]
        h //= 2
    return x


def rate_match(x: np.ndarray) -> np.ndarray:
    """Mother codeword (..., 2048) -> coded bits (..., 2474) by circular repetition."""
    return np.asarray(x)[..., _RM]


def fold(llr: np.ndarray) -> np.ndarray:
    """Coded LLRs (2474,) -> mother LLRs (2048,): repeated bits add."""
    llr = np.asarray(llr, dtype=np.float64)
    if llr.shape != (E,):
        raise ValueError(f"expected {E} coded LLRs, got shape {llr.shape}")
    return np.bincount(_RM, weights=llr, minlength=N)


def polar_encode(info: np.ndarray) -> np.ndarray:
    """142 header bits (table order) -> 2474 coded bits, uint8 0/1."""
    info = np.asarray(info)
    if info.shape != (K,):
        raise ValueError(f"expected {K} information bits, got shape {info.shape}")
    if np.any((info != 0) & (info != 1)):
        raise ValueError("information bits must be 0 or 1")
    u = np.zeros(N, dtype=np.uint8)
    u[_INFO] = info
    return rate_match(polar_transform(u))


# --- CA-SCL decoder ------------------------------------------------------------


@lru_cache(maxsize=1)
def _rate0_nodes() -> frozenset:
    """(offset, size) of every decoding-tree node whose leaves are all frozen."""
    info = np.zeros(N + 1, dtype=np.int64)
    info[1:] = np.cumsum(np.isin(np.arange(N), _INFO))
    out = set()
    size = N
    while size >= 1:
        for lo in range(0, N, size):
            if info[lo + size] == info[lo]:
                out.add((lo, size))
        size //= 2
    return frozenset(out)


class _List:
    """Mutable state shared down the recursion: path metrics and list size."""

    __slots__ = ("pm", "size", "rate0")

    def __init__(self, size: int):
        self.size = size
        self.pm = np.full(size, np.inf)
        self.pm[0] = 0.0                # one live path; the rest are placeholders
        self.rate0 = _rate0_nodes()


def _softplus_neg(a: np.ndarray) -> np.ndarray:
    """log(1 + exp(-a)), stable: the metric cost of deciding 0 against LLR a."""
    return np.logaddexp(0.0, -a)


def _scl(alpha: np.ndarray, lo: int, st: _List) -> tuple[np.ndarray, np.ndarray]:
    """Decode the subtree at mother indices [lo, lo + n) from its LLRs.

    alpha: (list, n) LLRs at this node's input. Returns (beta, origin):
    beta (list, n) uint8 is the node's re-encoded output bits for each
    surviving path, and origin[l] is the index, in the ordering at entry,
    of the path that surviving path l descends from.
    """
    L, n = alpha.shape
    if (lo, n) in st.rate0:
        st.pm = st.pm + _softplus_neg(alpha).sum(axis=1)
        return np.zeros((L, n), dtype=np.uint8), np.arange(L)
    if n == 1:
        lam = alpha[:, 0]
        cand = np.concatenate([st.pm + _softplus_neg(lam), st.pm + _softplus_neg(-lam)])
        keep = np.argsort(cand, kind="stable")[:L]
        st.pm = cand[keep]
        return (keep // L).astype(np.uint8)[:, None], keep % L
    h = n // 2
    a1, a2 = alpha[:, :h], alpha[:, h:]
    f = np.sign(a1) * np.sign(a2) * np.minimum(np.abs(a1), np.abs(a2))
    beta_a, org_a = _scl(f, lo, st)
    a1, a2 = a1[org_a], a2[org_a]
    g = a2 + (1.0 - 2.0 * beta_a) * a1
    beta_b, org_b = _scl(g, lo + h, st)
    beta_a = beta_a[org_b]
    return np.concatenate([beta_a ^ beta_b, beta_b], axis=1), org_a[org_b]


def decode_list(llr: np.ndarray, list_size: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """SCL over coded LLRs (2474,). Returns (info (m, 142) uint8, pm (m,)).

    The m <= list_size surviving paths come back most likely first (lowest
    path metric). No CRC or field check is applied here.
    """
    if list_size < 1:
        raise ValueError("list_size must be >= 1")
    mother = fold(llr)
    if not np.all(np.isfinite(mother)):
        raise ValueError("LLRs must be finite")
    st = _List(list_size)
    alpha = np.broadcast_to(mother, (list_size, N))
    beta, _ = _scl(alpha, 0, st)
    u = polar_transform(beta)                       # G is its own inverse
    order = np.argsort(st.pm, kind="stable")
    order = order[np.isfinite(st.pm[order])]
    return u[order][:, _INFO], st.pm[order]


def polar_decode(llr: np.ndarray, list_size: int = 8, accept=None) -> np.ndarray | None:
    """Coded LLRs (2474,) -> 142 header bits, or None.

    Returns the most likely list path for which `accept(bits)` is true.
    The default acceptance is the header's: CRC, version, reserved, mode,
    segment and callsign checks (`header.unpack(bits) is not None`).
    """
    if accept is None:
        from . import header   # header imports this module
        accept = lambda bits: header.unpack(bits) is not None  # noqa: E731
    cands, _ = decode_list(llr, list_size)
    for bits in cands:
        if accept(bits):
            return bits
    return None

"""The per-pass precoder: sign flips, then a blockwise Walsh-Hadamard
transform (spec 2.4, design 2.5).

A **format module**. A segment's data latents in air order are split into
blocks -- 64 at a time, then a binary decomposition of the remainder
(50,600 = 790 x 64 + 32 + 8, decision D6) -- and each block goes out as
H.(c_q * a), with H the orthonormal Sylvester Hadamard matrix of the
block's own size, H[i, j] = (-1)^popcount(i & j) / sqrt(M). H is
symmetric and orthonormal, so it is its own inverse, and the receiver
undoes the precoder with the same transform followed by the same signs.

Without the transform, the part of CE's phase-modulation distortion that
depends on a latent's own value would come back identical in every pass
and never average away (spec 2.4's table: 13.4 dB -> 23.1 dB).
"""

import functools

import numpy as np

from .sequences import scrambler

BLOCK = 64


def block_sizes(n: int) -> list[int]:
    """[64] * (n // 64) plus the binary decomposition of n % 64, descending.

    50600 -> 790 x 64, 32, 8;  16384 -> 256 x 64;  4096 -> 64 x 64.
    """
    return list(_block_sizes(int(n)))


@functools.lru_cache(maxsize=64)
def _block_sizes(n: int) -> tuple[int, ...]:
    if n < 0:
        raise ValueError(f"n must be >= 0, not {n}")
    rem = n % BLOCK
    tail = [1 << k for k in range(BLOCK.bit_length() - 1, -1, -1) if rem & (1 << k)]
    return tuple([BLOCK] * (n // BLOCK) + tail)


@functools.lru_cache(maxsize=64)
def _runs(n: int) -> tuple[tuple[int, int, int], ...]:
    """(start, block size, block count) for each run of equal-size blocks."""
    sizes = _block_sizes(n)
    out, start = [], 0
    for m in sorted(set(sizes), reverse=True):
        count = sizes.count(m)
        out.append((start, m, count))
        start += m * count
    return tuple(out)


def _fwht_inplace(a: np.ndarray) -> None:
    """Unnormalised Sylvester-order butterflies along the last axis."""
    m = a.shape[-1]
    h = 1
    while h < m:
        v = a.reshape(a.shape[:-1] + (m // (2 * h), 2, h))
        x = v[..., 0, :].copy()
        y = v[..., 1, :]
        v[..., 0, :] += y
        v[..., 1, :] = x - y
        h *= 2


def wht(v) -> np.ndarray:
    """Blockwise orthonormal WHT over the last axis (self-inverse).

    Leading axes are batch axes. Returns a new float64 array.
    """
    out = np.array(v, dtype=np.float64, copy=True)
    n = out.shape[-1]
    for start, m, count in _runs(n):
        seg = out[..., start:start + m * count].reshape(out.shape[:-1] + (count, m))
        seg = np.ascontiguousarray(seg)
        _fwht_inplace(seg)
        out[..., start:start + m * count] = (seg / np.sqrt(m)).reshape(
            out.shape[:-1] + (m * count,))
    return out


def precode(a_air, q: int) -> np.ndarray:
    """x = WHT(c_q * a): air-order latents -> precoded data symbols."""
    a = np.asarray(a_air, dtype=np.float64)
    return wht(scrambler(q, a.shape[-1]) * a)


def unprecode(y, q: int) -> np.ndarray:
    """a = c_q * WHT(y): received data symbols -> air-order latents."""
    y = np.asarray(y, dtype=np.float64)
    return scrambler(q, y.shape[-1]) * wht(y)


def block_mean(v) -> np.ndarray:
    """Per-precoder-block mean over the last axis, broadcast back to (..., n)."""
    v = np.asarray(v, dtype=np.float64)
    out = np.empty_like(v)
    for start, m, count in _runs(v.shape[-1]):
        seg = v[..., start:start + m * count].reshape(v.shape[:-1] + (count, m))
        out[..., start:start + m * count] = np.repeat(seg.mean(axis=-1), m, axis=-1)
    return out


def block_index(n: int) -> np.ndarray:
    """(n,) int64: the precoder block each data symbol belongs to."""
    sizes = _block_sizes(int(n))
    return np.repeat(np.arange(len(sizes)), sizes)

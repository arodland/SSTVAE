"""Frame layout: latent interleaving and the Golay-coded header.

Latents are ordered by channel group (G0 first) so that early-stopping a
transmission keeps whole coarse groups. Within each group, a fixed
pseudo-random permutation spreads latents across that group's frames, so
a lost frame becomes diffuse noise over the whole image rather than a
missing region.
"""

import numpy as np

from ..config import (
    NC,
    DATA_SYMS_PER_FRAME,
    LATENTS_PER_FRAME,
    GROUP_LATENTS,
    FRAMES_PER_GROUP,
    INTERLEAVER_SEED,
    PROTOCOL_VERSION,
    MODES_BY_INDEX,
    ModeSpec,
)
from . import golay

_PERMS = [
    np.random.default_rng(INTERLEAVER_SEED + g).permutation(GROUP_LATENTS)
    for g in range(3)
]
_INV_PERMS = [np.argsort(p) for p in _PERMS]


def interleave(latents: np.ndarray, mode: ModeSpec) -> np.ndarray:
    """Canonical latent vector -> on-air slot order (same length)."""
    assert len(latents) == mode.n_latents
    out = np.empty_like(latents)
    for g in range(mode.groups):
        lo, hi = g * GROUP_LATENTS, (g + 1) * GROUP_LATENTS
        out[lo:hi] = latents[lo:hi][_PERMS[g]]
    return out


def deinterleave(slots: np.ndarray, mode: ModeSpec) -> np.ndarray:
    assert len(slots) == mode.n_latents
    out = np.empty_like(slots)
    for g in range(mode.groups):
        lo, hi = g * GROUP_LATENTS, (g + 1) * GROUP_LATENTS
        out[lo:hi] = slots[lo:hi][_INV_PERMS[g]]
    return out


def slots_to_symbols(frame_slots: np.ndarray) -> np.ndarray:
    """528 real slot values -> (DATA_SYMS_PER_FRAME, NC) complex symbols.

    Pairs of consecutive slots map to I/Q of one carrier; 1/sqrt(2)
    keeps unit-RMS latents at unit symbol power.
    """
    s = frame_slots.reshape(DATA_SYMS_PER_FRAME, NC, 2)
    return (s[..., 0] + 1j * s[..., 1]) / np.sqrt(2)


def symbols_to_slots(symbols: np.ndarray) -> np.ndarray:
    """Inverse of slots_to_symbols; accepts any (n_sym, NC) shape."""
    s = np.empty(symbols.shape + (2,))
    s[..., 0] = np.real(symbols) * np.sqrt(2)
    s[..., 1] = np.imag(symbols) * np.sqrt(2)
    return s.reshape(-1)


# --- header ----------------------------------------------------------------


def _check_nibble(mode_idx: int, version: int) -> int:
    return (mode_idx ^ version ^ 0xA) & 0xF


def header_bits(mode: ModeSpec) -> np.ndarray:
    """24 codeword bits (0/1) for the header BPSK symbol."""
    data = (mode.index & 0xF) | ((PROTOCOL_VERSION & 0xF) << 4)
    data |= _check_nibble(mode.index, PROTOCOL_VERSION) << 8
    return golay.codeword_bits(data)


def header_symbol(mode: ModeSpec) -> np.ndarray:
    """Header as (NC,) complex BPSK carrier amplitudes."""
    return (1.0 - 2.0 * header_bits(mode)).astype(np.complex128)


def decode_header(soft: np.ndarray) -> ModeSpec | None:
    """Soft values (one per carrier, summed over header repeats) -> mode."""
    data = golay.decode_soft(soft)
    mode_idx = data & 0xF
    version = (data >> 4) & 0xF
    check = (data >> 8) & 0xF
    if version != PROTOCOL_VERSION or check != _check_nibble(mode_idx, version):
        return None
    return MODES_BY_INDEX.get(mode_idx)

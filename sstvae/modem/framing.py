"""Frame layout: latent interleaving and the Golay-coded header.

Latents are ordered by channel group (G0 first) so that early-stopping a
transmission keeps whole coarse groups. Within each group, a fixed
pseudo-random permutation spreads latents across that group's frames, so
a lost frame becomes diffuse noise over the whole image rather than a
missing region.

Each group has more canonical latents (GROUP_LATENTS) than on-air slots
(TRANSMIT_LATENTS_PER_GROUP), because one carrier per frame is reserved
for the beacon side-channel (see config.py). Only the first
TRANSMIT_LATENTS_PER_GROUP entries of each group's permutation get a
slot; the rest are permanently erased (weight 0), never transmitted.
"""

import numpy as np

from ..config import (
    NC,
    NC_LATENT,
    DATA_SYMS_PER_FRAME,
    LATENTS_PER_FRAME,
    GROUP_LATENTS,
    TRANSMIT_LATENTS_PER_GROUP,
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
# Only the first TRANSMIT_LATENTS_PER_GROUP permuted indices ever reach a
# slot; the remainder (_PERMS[g][TRANSMIT_LATENTS_PER_GROUP:]) are the
# canonical latents this group always drops.
_TX_PERMS = [p[:TRANSMIT_LATENTS_PER_GROUP] for p in _PERMS]


def slot_range_for_frame(abs_frame: int) -> tuple[int, np.ndarray]:
    """Absolute frame index (0..3*FRAMES_PER_GROUP-1, i.e. mode C's full
    frame range — every mode is a prefix of it) -> (group, canonical
    latent indices for that one frame's LATENTS_PER_FRAME slots, in slot
    order). Lets a single frame be placed into canonical latent space
    without knowing which mode (A/B/C) produced the transmission — the
    situation a blind, no-header resync lands in (see sync.acquire_blind
    and Modem.demodulate_blind)."""
    g, fg = divmod(abs_frame, FRAMES_PER_GROUP)
    slo = fg * LATENTS_PER_FRAME
    shi = slo + LATENTS_PER_FRAME
    return g, g * GROUP_LATENTS + _TX_PERMS[g][slo:shi]


def interleave(latents: np.ndarray, mode: ModeSpec) -> np.ndarray:
    """Canonical latent vector -> on-air slot order (mode.n_tx_latents)."""
    assert len(latents) == mode.n_latents
    out = np.empty(mode.n_tx_latents, dtype=latents.dtype)
    for g in range(mode.groups):
        lo, hi = g * GROUP_LATENTS, (g + 1) * GROUP_LATENTS
        slo, shi = g * TRANSMIT_LATENTS_PER_GROUP, (g + 1) * TRANSMIT_LATENTS_PER_GROUP
        out[slo:shi] = latents[lo:hi][_TX_PERMS[g]]
    return out


def deinterleave(slots: np.ndarray, mode: ModeSpec) -> tuple[np.ndarray, np.ndarray]:
    """On-air slot order -> (canonical latent vector, weight mask).

    Weight is 1 everywhere a slot maps to (fill it in from `slots`); the
    latents each group permanently drops (never given a slot) come back
    as 0 with weight 0, same contract as a channel erasure.
    """
    assert len(slots) == mode.n_tx_latents
    out = np.zeros(mode.n_latents, dtype=slots.dtype)
    weight = np.zeros(mode.n_latents, dtype=slots.dtype)
    for g in range(mode.groups):
        lo = g * GROUP_LATENTS
        slo, shi = g * TRANSMIT_LATENTS_PER_GROUP, (g + 1) * TRANSMIT_LATENTS_PER_GROUP
        idx = lo + _TX_PERMS[g]
        out[idx] = slots[slo:shi]
        weight[idx] = 1
    return out, weight


def slots_to_symbols(frame_slots: np.ndarray) -> np.ndarray:
    """One frame's real slot values -> (DATA_SYMS_PER_FRAME, NC_LATENT)
    complex symbols, covering only the 23 latent-carrying carriers (the
    beacon carrier is handled separately by the caller).

    Pairs of consecutive slots map to I/Q of one carrier; 1/sqrt(2)
    keeps unit-RMS latents at unit symbol power.
    """
    s = frame_slots.reshape(DATA_SYMS_PER_FRAME, NC_LATENT, 2)
    return (s[..., 0] + 1j * s[..., 1]) / np.sqrt(2)


def symbols_to_slots(symbols: np.ndarray) -> np.ndarray:
    """Inverse of slots_to_symbols; accepts any (n_sym, NC_LATENT) shape."""
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

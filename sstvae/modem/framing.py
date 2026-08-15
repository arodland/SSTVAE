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

import functools
from pathlib import Path

import numpy as np

from ..config import (
    NC,
    NC_LATENT,
    DATA_SYMS_PER_FRAME,
    LATENT_GROUPS,
    LATENTS_PER_FRAME,
    GROUP_LATENTS,
    TRANSMIT_LATENTS_PER_GROUP,
    FRAMES_PER_GROUP,
    PROTOCOL_VERSION,
    MODES_BY_INDEX,
    ModeSpec,
)
from . import golay

# The interleaver permutations are **frozen data, not a computation**.
#
# They were originally drawn with
# `np.random.default_rng(INTERLEAVER_SEED + g).permutation(GROUP_LATENTS)`,
# and re-deriving them at import would make numpy's PCG64, its
# bounded-integer draw and its shuffle loop part of the on-air format.
# numpy commits to stream stability, but if it ever did change, the
# correct behaviour is to keep interleaving exactly as before -- two
# stations that disagree here produce noise, with no error to say why.
# That is only possible if the permutation is written down, so it is.
#
# Only the transmittable prefix is stored. The remainder of each
# permutation is the set of latents that group permanently drops
# (weight 0); it is defined by its absence, so storing it would be
# storing something nothing reads.
#
# `tools/freeze_format_constants.py --verify` re-derives these from the
# seed and reports whether the current numpy still agrees. That is
# information, not a gate: see the note in config.py.
_PERMS_PATH = Path(__file__).with_name("interleaver_perms.npy")
try:
    # Stored as uint16 (indices are < GROUP_LATENTS = 52,800) to keep the
    # committed file small, but widened on load: callers add a group
    # offset of up to 2*GROUP_LATENTS = 105,600, and under NEP 50 a
    # Python int added to a uint16 array stays uint16. numpy 2 raises on
    # that, but an older numpy would wrap silently -- which would
    # scramble the interleave for groups 1 and 2 and produce a picture
    # that decoded to noise with nothing to indicate why.
    _TX_PERMS = np.load(_PERMS_PATH).astype(np.intp)
except FileNotFoundError as e:  # pragma: no cover - packaging error
    raise RuntimeError(
        f"{_PERMS_PATH.name} is missing. It carries the interleaver "
        "permutations, which are part of the on-air format and are shipped "
        "as package data rather than recomputed. Reinstall the package, or "
        "regenerate with tools/freeze_format_constants.py."
    ) from e

assert _TX_PERMS.shape == (3, TRANSMIT_LATENTS_PER_GROUP), (
    f"interleaver_perms.npy has shape {_TX_PERMS.shape}, expected "
    f"(3, {TRANSMIT_LATENTS_PER_GROUP}); it does not match this config.py"
)


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


@functools.lru_cache(maxsize=1)
def frame_of_latent() -> np.ndarray:
    """Canonical latent index -> the absolute frame index that carries
    it, over mode C's full range; -1 for the latents each group
    permanently drops (never given a slot, see the module docstring).

    The inverse of `slot_range_for_frame`, as one array. A receiver that
    knows only *which latents* arrived -- the blind path, which has no
    header and so no mode -- needs this to say how far into the
    transmission it has got: a latent count answers "how much", and the
    interleaver scatters each frame across the whole picture, so only
    the frame index answers "how far"."""
    out = np.full(LATENT_GROUPS * GROUP_LATENTS, -1, dtype=np.int32)
    for f in range(LATENT_GROUPS * FRAMES_PER_GROUP):
        _, idx = slot_range_for_frame(f)
        out[idx] = f
    out.setflags(write=False)
    return out


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

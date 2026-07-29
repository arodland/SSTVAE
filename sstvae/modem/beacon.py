"""Beacon side-channel: a continuously repeating, self-describing packet
carried as BPSK chips on the one reserved carrier (config.BEACON_CARRIER),
CHIPS_PER_FRAME chips per frame, for the entire transmission.

Each repetition ("superframe") is:

    sync word (Barker-13, unmodulated marker)
    | Golay(24,12)-coded payload (7 codewords = 84 padded payload bits)

Payload = 10-bit absolute frame counter (index of the frame whose data
symbols carry the sync word's first chip) + 48-bit callsign (8 chars x
6 bits) + 16-bit CRC over counter+callsign, zero-padded to 84 bits for
clean 12-bit Golay chunking.

Because the counter is absolute (not modulo the superframe period), a
receiver needs no prior knowledge of where the transmission started: any
window containing one full, correctly-decoded superframe reveals exactly
which frame index it landed on, which — combined with frame-boundary
timing recovered independently (preamble, or blind pilot-lag lock, see
sync.acquire_blind) — reconstructs the whole transmission's sample
alignment retroactively, even from audio recorded before the receiver
"noticed" the signal.
"""

from dataclasses import dataclass

import numpy as np

from ..config import (
    BEACON_SYNC,
    BEACON_COUNTER_BITS,
    BEACON_CALLSIGN_CHARS,
    BEACON_CALLSIGN_CHAR_BITS,
    BEACON_CALLSIGN_BITS,
    BEACON_CRC_BITS,
    CHIPS_PER_FRAME,
)
from . import golay

SYNC = np.array(BEACON_SYNC, dtype=np.float64)
SYNC_LEN = len(SYNC)

# 64-symbol alphabet for 6-bit callsign characters. Amateur callsigns are
# uppercase letters/digits/slash; the rest of the code space is filled
# with harmless extra punctuation so any 6-bit value round-trips to a
# printable character.
_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789/-. " + "?!@#$%^&*()_+=~[]{}<>:;,"
assert len(_ALPHABET) == 64
_CHAR_TO_CODE = {c: i for i, c in enumerate(_ALPHABET)}

_PAYLOAD_BITS = BEACON_COUNTER_BITS + BEACON_CALLSIGN_BITS + BEACON_CRC_BITS  # 74
N_CHUNKS = -(-_PAYLOAD_BITS // 12)  # 7
PADDED_PAYLOAD_BITS = N_CHUNKS * 12  # 84
CODED_LEN = N_CHUNKS * 24  # 168 chips
SUPERFRAME_LEN = SYNC_LEN + CODED_LEN  # 181 chips
MAX_FRAME_COUNTER = (1 << BEACON_COUNTER_BITS) - 1

# A window needs at least 2*SUPERFRAME_LEN-1 chips to *guarantee* a full,
# uncut superframe repetition regardless of phase (worst case: the one
# repetition that would fit starts just past the window's first chip).
# Below this, decode() may still succeed if you get lucky on phase, but
# isn't guaranteed to.
MIN_FRAMES_FOR_SYNC = -(-(2 * SUPERFRAME_LEN - 1) // CHIPS_PER_FRAME)  # 73


@dataclass
class BeaconResult:
    chip_offset: int  # index into the input chip array where sync starts
    frame_index: int  # absolute frame index of that chip (from the counter)
    callsign: str


# --- callsign <-> 6-bit codes -----------------------------------------------


def callsign_to_codes(callsign: str) -> np.ndarray:
    """String -> BEACON_CALLSIGN_CHARS 6-bit codes, space-padded/truncated."""
    s = callsign.upper()[:BEACON_CALLSIGN_CHARS]
    s = s.ljust(BEACON_CALLSIGN_CHARS)
    return np.array([_CHAR_TO_CODE.get(c, _CHAR_TO_CODE[" "]) for c in s])


def codes_to_callsign(codes: np.ndarray) -> str:
    return "".join(_ALPHABET[int(c) & 0x3F] for c in codes).rstrip()


# --- bit packing -------------------------------------------------------------


def _int_to_bits(value: int, width: int) -> np.ndarray:
    return ((value >> np.arange(width - 1, -1, -1)) & 1).astype(np.int64)


def _bits_to_int(bits: np.ndarray) -> int:
    v = 0
    for b in bits:
        v = (v << 1) | int(b)
    return v


def _crc16(bits: np.ndarray) -> np.ndarray:
    """CRC-16/CCITT-FALSE over a 0/1 bit array, MSB-first, init 0xFFFF."""
    reg = 0xFFFF
    for b in bits:
        top = (reg >> 15) & 1
        reg = ((reg << 1) & 0xFFFF) | int(b)
        if top:
            reg ^= 0x1021
    return _int_to_bits(reg, 16)


def _payload_bits(frame_index: int, callsign: str) -> np.ndarray:
    if not (0 <= frame_index <= MAX_FRAME_COUNTER):
        raise ValueError(
            f"frame_index {frame_index} exceeds {BEACON_COUNTER_BITS}-bit counter "
            f"range (max {MAX_FRAME_COUNTER})"
        )
    counter_bits = _int_to_bits(frame_index, BEACON_COUNTER_BITS)
    codes = callsign_to_codes(callsign)
    callsign_bits = np.concatenate(
        [_int_to_bits(int(c), BEACON_CALLSIGN_CHAR_BITS) for c in codes]
    )
    body = np.concatenate([counter_bits, callsign_bits])
    crc_bits = _crc16(body)
    return np.concatenate([body, crc_bits])


# --- superframe encode/decode ------------------------------------------------


def encode_chips(frame_index: int, callsign: str) -> np.ndarray:
    """One superframe repetition -> SUPERFRAME_LEN chips in {-1, +1}."""
    payload = _payload_bits(frame_index, callsign)
    padded = np.concatenate([payload, np.zeros(PADDED_PAYLOAD_BITS - len(payload), dtype=np.int64)])
    coded_bits = np.concatenate(
        [golay.codeword_bits(_bits_to_int(padded[i : i + 12])) for i in range(0, PADDED_PAYLOAD_BITS, 12)]
    )
    coded_chips = 1.0 - 2.0 * coded_bits.astype(np.float64)
    return np.concatenate([SYNC, coded_chips])


def chip_stream(start_frame: int, n_frames: int, callsign: str) -> np.ndarray:
    """Continuous beacon chip stream, CHIPS_PER_FRAME chips per frame,
    covering frames [start_frame, start_frame + n_frames): superframes
    back-to-back (truncating the last one if it doesn't fit exactly),
    each correctly labeled with the absolute frame index its sync word
    lands on."""
    n_chips = n_frames * CHIPS_PER_FRAME
    out = np.empty(n_chips)
    pos = 0
    while pos < n_chips:
        frame_index = start_frame + pos // CHIPS_PER_FRAME
        sf = encode_chips(frame_index, callsign)
        take = min(len(sf), n_chips - pos)
        out[pos : pos + take] = sf[:take]
        pos += take
    return out


def _decode_payload(coded_chips: np.ndarray) -> tuple[int, str] | None:
    """Soft chip values (168,) -> (frame_index, callsign), or None on CRC fail."""
    bits = []
    for i in range(N_CHUNKS):
        soft = coded_chips[i * 24 : (i + 1) * 24]
        data = golay.decode_soft(soft)
        bits.append(_int_to_bits(data, 12))
    padded = np.concatenate(bits)
    payload = padded[:_PAYLOAD_BITS]
    body, crc_bits = payload[:-BEACON_CRC_BITS], payload[-BEACON_CRC_BITS:]
    if not np.array_equal(_crc16(body), crc_bits):
        return None
    counter_bits, callsign_bits = body[:BEACON_COUNTER_BITS], body[BEACON_COUNTER_BITS:]
    frame_index = _bits_to_int(counter_bits)
    codes = [
        _bits_to_int(callsign_bits[i : i + BEACON_CALLSIGN_CHAR_BITS])
        for i in range(0, BEACON_CALLSIGN_BITS, BEACON_CALLSIGN_CHAR_BITS)
    ]
    return frame_index, codes_to_callsign(np.array(codes))


def find_sync(chips: np.ndarray, threshold: float = 0.6, max_candidates: int = 8) -> list[int]:
    """Offsets in `chips` where the Barker-13 sync word plausibly starts,
    best-correlation first. Normalized so results are ~comparable across
    signal levels."""
    if len(chips) < SYNC_LEN:
        return []
    corr = np.correlate(chips, SYNC, mode="valid")
    energy = np.sqrt(
        np.convolve(chips**2, np.ones(SYNC_LEN), mode="valid") * np.sum(SYNC**2)
    ) + 1e-12
    score = corr / energy
    # Best first, ties by lowest offset.
    #
    # `np.argsort(score)[::-1]` was not deterministic here: a clean
    # stream has one perfectly-correlating position per superframe, so
    # exact ties are the normal case rather than a freak one, and
    # argsort's default is an *unstable* sort. Which of the tied
    # candidates decode() returned therefore depended on numpy's sort
    # internals — it picked [0, 543, 362, 181] out of four equal scores.
    # Every one of them is a valid superframe, so the answer was never
    # wrong, but it was arbitrary, and it made the result unreproducible
    # across implementations for no benefit.
    #
    # kind="stable" on the negated score gives descending score with
    # ties in ascending offset order, which is both deterministic and
    # the sensible choice: prefer the earliest complete superframe.
    order = np.argsort(-score, kind="stable")
    out = []
    for i in order[: max_candidates * 4]:
        if score[i] >= threshold:
            out.append(int(i))
        if len(out) >= max_candidates:
            break
    return out


def decode(chips: np.ndarray, threshold: float = 0.6) -> BeaconResult | None:
    """Find and decode one beacon superframe anywhere in `chips` (soft
    values, any length >= SUPERFRAME_LEN). Tries the best-correlated sync
    candidates in order and returns the first one whose CRC checks out."""
    for off in find_sync(chips, threshold=threshold):
        end = off + SYNC_LEN + CODED_LEN
        if end > len(chips):
            continue
        result = _decode_payload(chips[off + SYNC_LEN : end])
        if result is not None:
            frame_index, callsign = result
            return BeaconResult(chip_offset=off, frame_index=frame_index, callsign=callsign)
    return None

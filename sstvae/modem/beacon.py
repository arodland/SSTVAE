"""Beacon side-channel: a continuously repeating, self-describing packet
carried as BPSK chips on the one reserved carrier (config.BEACON_CARRIER),
CHIPS_PER_FRAME chips per frame, for the entire transmission.

Each repetition ("superframe") is:

    sync word (Barker-13, unmodulated marker)
    | Golay(24,12)-coded payload (7 codewords = 84 payload bits)

Payload = 10-bit absolute frame counter (index of the frame whose data
symbols carry the sync word's first chip) + 48-bit callsign (8 chars x
6 bits) + 2-bit mode index + 8 reserved bits (transmitted as
config.BEACON_RESERVED_VALUE, ignored on decode) + 16-bit CRC over
everything before it. Exactly 84 bits -- the same 7 Golay chunks the
pre-mode-field format padded out to, so the superframe length and chip
cadence are unchanged from PROTOCOL_VERSION 3.

The mode field is what lets a late joiner -- who has no header and
never will -- know the transmission's real frame count: without it the
blind path had to assume mode C's range, and every frame demodulated
past the real transmission's end was post-transmission noise entering
the reconstruction at nonzero weight (see Modem.demodulate_blind).

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
    BEACON_MODE_BITS,
    BEACON_RESERVED_BITS,
    BEACON_RESERVED_VALUE,
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

_PAYLOAD_BITS = (
    BEACON_COUNTER_BITS + BEACON_CALLSIGN_BITS + BEACON_MODE_BITS
    + BEACON_RESERVED_BITS + BEACON_CRC_BITS
)  # 84
N_CHUNKS = -(-_PAYLOAD_BITS // 12)  # 7
PADDED_PAYLOAD_BITS = N_CHUNKS * 12  # 84: the payload fills every chunk exactly
CODED_LEN = N_CHUNKS * 24  # 168 chips
SUPERFRAME_LEN = SYNC_LEN + CODED_LEN  # 181 chips
MAX_FRAME_COUNTER = (1 << BEACON_COUNTER_BITS) - 1
MAX_MODE_INDEX = (1 << BEACON_MODE_BITS) - 1

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
    # Transmitted mode index, raw, and deliberately without a default: a
    # constructor that forgot it would otherwise silently claim mode A
    # and truncate every blind reception to one group. Look it up via
    # config.MODES_BY_INDEX; a value with no entry there (3 today) is a
    # mode this receiver does not know, and consumers must fall back to
    # the old "assume mode C" behaviour rather than reject the packet --
    # that is what lets a future mode be added without breaking fielded
    # receivers.
    mode_index: int


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


def _payload_bits(frame_index: int, callsign: str, mode_index: int) -> np.ndarray:
    if not (0 <= frame_index <= MAX_FRAME_COUNTER):
        raise ValueError(
            f"frame_index {frame_index} exceeds {BEACON_COUNTER_BITS}-bit counter "
            f"range (max {MAX_FRAME_COUNTER})"
        )
    if not (0 <= mode_index <= MAX_MODE_INDEX):
        raise ValueError(
            f"mode_index {mode_index} exceeds {BEACON_MODE_BITS}-bit mode field "
            f"range (max {MAX_MODE_INDEX})"
        )
    counter_bits = _int_to_bits(frame_index, BEACON_COUNTER_BITS)
    codes = callsign_to_codes(callsign)
    callsign_bits = np.concatenate(
        [_int_to_bits(int(c), BEACON_CALLSIGN_CHAR_BITS) for c in codes]
    )
    body = np.concatenate([
        counter_bits,
        callsign_bits,
        _int_to_bits(mode_index, BEACON_MODE_BITS),
        _int_to_bits(BEACON_RESERVED_VALUE, BEACON_RESERVED_BITS),
    ])
    crc_bits = _crc16(body)
    return np.concatenate([body, crc_bits])


# --- superframe encode/decode ------------------------------------------------


def encode_chips(frame_index: int, callsign: str, mode_index: int) -> np.ndarray:
    """One superframe repetition -> SUPERFRAME_LEN chips in {-1, +1}."""
    payload = _payload_bits(frame_index, callsign, mode_index)
    assert len(payload) == PADDED_PAYLOAD_BITS  # exactly fills the chunks
    coded_bits = np.concatenate(
        [golay.codeword_bits(_bits_to_int(payload[i : i + 12])) for i in range(0, PADDED_PAYLOAD_BITS, 12)]
    )
    coded_chips = 1.0 - 2.0 * coded_bits.astype(np.float64)
    return np.concatenate([SYNC, coded_chips])


def chip_stream(start_frame: int, n_frames: int, callsign: str, mode_index: int) -> np.ndarray:
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
        sf = encode_chips(frame_index, callsign, mode_index)
        take = min(len(sf), n_chips - pos)
        out[pos : pos + take] = sf[:take]
        pos += take
    return out


def _decode_payload(coded_chips: np.ndarray) -> tuple[int, str, int] | None:
    """Soft chip values (168,) -> (frame_index, callsign, mode_index), or
    None on CRC fail. The reserved bits are CRC-covered but their *value*
    is deliberately not checked: a future sender that assigns them still
    decodes here (the CRC covers whatever was actually sent), which is
    what "reserved" has to mean for fielded receivers."""
    bits = []
    for i in range(N_CHUNKS):
        soft = coded_chips[i * 24 : (i + 1) * 24]
        data = golay.decode_soft(soft)
        bits.append(_int_to_bits(data, 12))
    payload = np.concatenate(bits)[:_PAYLOAD_BITS]
    body, crc_bits = payload[:-BEACON_CRC_BITS], payload[-BEACON_CRC_BITS:]
    if not np.array_equal(_crc16(body), crc_bits):
        return None
    counter_bits = body[:BEACON_COUNTER_BITS]
    callsign_bits = body[BEACON_COUNTER_BITS : BEACON_COUNTER_BITS + BEACON_CALLSIGN_BITS]
    mode_bits = body[
        BEACON_COUNTER_BITS + BEACON_CALLSIGN_BITS :
        BEACON_COUNTER_BITS + BEACON_CALLSIGN_BITS + BEACON_MODE_BITS
    ]
    frame_index = _bits_to_int(counter_bits)
    codes = [
        _bits_to_int(callsign_bits[i : i + BEACON_CALLSIGN_CHAR_BITS])
        for i in range(0, BEACON_CALLSIGN_BITS, BEACON_CALLSIGN_CHAR_BITS)
    ]
    return frame_index, codes_to_callsign(np.array(codes)), _bits_to_int(mode_bits)


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
    candidates in order and returns the first one whose CRC checks out;
    falls back to combining evidence across every repetition in `chips`
    (see `_decode_combined`) if no single one decodes alone."""
    candidates = find_sync(chips, threshold=threshold)
    for off in candidates:
        end = off + SYNC_LEN + CODED_LEN
        if end > len(chips):
            continue
        result = _decode_payload(chips[off + SYNC_LEN : end])
        if result is not None:
            frame_index, callsign, mode_index = result
            return BeaconResult(chip_offset=off, frame_index=frame_index,
                                callsign=callsign, mode_index=mode_index)
    for off in candidates:
        result = _decode_combined(chips, off)
        if result is not None:
            return result
    for off in _folded_sync_phases(chips):
        result = _decode_combined(chips, off)
        if result is not None:
            return result
    return None


# --- multi-repetition combining ----------------------------------------------
#
# The superframe repeats continuously through the whole transmission
# (~5.2 s apart -- up to ~18 times in a mode C transmission), but
# `decode()` above only ever looks at ONE of them: whichever sync
# candidate's own single-shot Golay+CRC decode happens to succeed. That
# throws away the other repetitions' evidence entirely, even though two
# things are guaranteed about them within one transmission:
#
# - The callsign is the same in every repetition.
# - The frame counter is not the same, but it is an *exact*, fully
#   predictable function of chip position: `chip_stream()` lays
#   superframes back-to-back with no gaps, and every decoder-side chip
#   array indexes frames uniformly (`beacon_soft[f * CHIPS_PER_FRAME +
#   ...]`), so knowing any one repetition's true counter value fixes
#   every other repetition's counter exactly -- no estimation needed,
#   just `(chip_delta // CHIPS_PER_FRAME)` arithmetic.
#
# `_decode_combined` is a limited (not fully maximum-likelihood) way to
# use that, in two tiers of decreasing simplicity:
#
# - Chunks that fall entirely inside the invariant callsign+mode fields
#   are identical, chip for chip, in every repetition, so their *signs*
#   (see below) can simply be summed across repetitions before
#   Golay-decoding once -- genuine coherent combining, and the cheap
#   case. The PROTOCOL_VERSION 4 layout (counter | callsign | mode |
#   reserved | CRC, 84 bits exactly) was chosen so the mode field lands
#   here: chunks 1-4 are pure callsign+mode.
# - The chunk carrying the counter's own low bit varies by design (a
#   different value is genuinely correct at every repetition), so it
#   can't be summed the same way -- but the exact value at every
#   repetition is a known function of one hypothesis for the anchor's
#   own value, so `_search_counter_chunk` evaluates every hypothesis by
#   regenerating each repetition's expected codeword and summing its
#   correlation, combining evidence across repetitions *before*
#   choosing rather than voting on independent per-repetition guesses.
#   Voting was tried first and measured useless here: 18 repetitions,
#   18 different guesses, no repeat at all, because at the SNRs this is
#   meant to help with individual repetitions rarely decode correctly
#   on their own.
#
# The pre-v4 layout had a third tier -- a chunk mixing leftover callsign
# bits with the per-repetition CRC, which needed its own brute-force
# search (`_search_crc_mixed_chunk`). The v4 layout retired it: with the
# reserved field pinned to a known constant and the CRC pushed to the
# very end, no chunk mixes unknown bits with CRC bits, so there is
# nothing left to search -- chunks 5-6 (reserved+CRC, CRC) are fully
# *predictable* from a (counter, callsign, mode) hypothesis instead,
# which is exactly what makes them usable as verification below.
#
# Summing/correlating raw chip values, throughout -- coherent
# (maximal-ratio) combining, so a repetition heard well counts for more
# than one heard badly.
#
# This summed *signs* until 2026-08-25, and that was correct for what
# it was given: the chips then came off the zero-forcing equalizer, so
# a repetition whose channel estimate sat near a fade null contributed
# an enormous, essentially random magnitude rather than a merely noisy
# one, and a single such repetition dominated a raw sum and wrecked
# it. Equal-gain combining bought immunity to that at the cost of
# throwing away every repetition's reliability.
#
# The chips are now `real(raw * conj(h))` (see modem.py), so a faded
# repetition arrives *small* instead of huge, which is exactly what
# coherent combining wants -- and the ordering inverts. Measured with
# scripts/beacon_combine_sweep.py, mode B, 150-frame window (3-4
# repetitions -- the regime where this code decides anything), 40
# trials/cell, bit-exact beacon decode:
#
#     chips           equalized  equalized      MRC      MRC
#     combining            sign        raw     sign      raw
#     awgn -4.0            0.45       0.20     0.50     0.65
#     mpg   0.0            0.72       0.30     0.75     0.82
#     mpp   0.0            0.57       0.23     0.72     0.93
#     mpd   0.0            0.40       0.15     0.53     0.90
#
# Note the two middle columns: raw summing really was ruinous on the
# old chips, in every cell, not just the one case originally measured.
# The two changes compound -- mpd at 0 dB goes 0.40 to 0.90 -- and
# neither is safe without the other. **Do not revert the chip metric
# without reverting this too.**
#
# A final correlation check against every repetition's full superframe
# (sync word included) guards against returning a wrong assembly as if
# it were a real decode.


def _folded_sync_phases(chips: np.ndarray, max_candidates: int = 4) -> list[int]:
    """Candidate anchor phases (mod SUPERFRAME_LEN) for the combining
    fallback, found by folding the sync-word correlation across every
    period instead of trusting any single repetition's own (noisy,
    13-chip) peak -- the same "fold across periods" idea
    BlindAccumulator uses for the pilot, applied here to the Barker-13
    word. A single repetition's find_sync() candidate can be the wrong
    phase under fading (any one 13-chip correlation has its own noise-
    driven false peaks, and the strongest one isn't always the true
    one); summing the same signed correlation across every repetition
    in the buffer before picking a phase is far more robust, since a
    real periodic sync word reinforces at one phase while noise at the
    wrong phases mostly cancels."""
    if len(chips) < SYNC_LEN + SUPERFRAME_LEN:
        return []
    corr = np.correlate(chips, SYNC, mode="valid")
    folded = np.array([corr[p::SUPERFRAME_LEN].sum() for p in range(SUPERFRAME_LEN)])
    order = np.argsort(-folded)[:max_candidates]
    return [int(p) for p in order]


def _chunk_bit_range(chunk_idx: int) -> tuple[int, int]:
    return chunk_idx * 12, (chunk_idx + 1) * 12


def _decode_chunk_bits(chip_slice: np.ndarray) -> np.ndarray:
    return _int_to_bits(golay.decode_soft(chip_slice), 12)


def _repetition_grid(n_chips: int, anchor_off: int) -> list[int]:
    """Every chip offset spaced an exact multiple of SUPERFRAME_LEN from
    `anchor_off` that has a full superframe's worth of chips available
    -- the repetition positions are *known* exactly once the anchor is
    fixed, so this doesn't depend on find_sync() having independently
    flagged each one (weak repetitions it missed are included anyway).
    Checks each direction's fit explicitly rather than assuming the
    anchor's own position (k=0) fits -- it doesn't always, e.g. an
    anchor near the very end of a short buffer."""
    def fits(pos: int) -> bool:
        return 0 <= pos and pos + SYNC_LEN + CODED_LEN <= n_chips

    grid = [anchor_off] if fits(anchor_off) else []
    k = 1
    while fits(anchor_off + k * SUPERFRAME_LEN):
        grid.append(anchor_off + k * SUPERFRAME_LEN)
        k += 1
    k = -1
    while fits(anchor_off + k * SUPERFRAME_LEN):
        grid.append(anchor_off + k * SUPERFRAME_LEN)
        k -= 1
    return sorted(grid)


def _search_counter_chunk(
    chips: np.ndarray, grid: list[int], anchor_off: int, chunk_idx: int, n_counter_bits: int
) -> tuple[int, int] | None:
    """Joint search over the chunk carrying the counter's low bit 0 --
    the one chunk voting can't help, since a *different* 12-bit value
    is genuinely correct at every repetition (counter, not callsign),
    so independently hard-decoding each repetition and voting on the
    (delta-normalized) results only works if a real plurality of
    individual repetitions already decode correctly on their own. Under
    fading, at exactly the SNRs where combining would matter most, they
    routinely don't (measured: 18 repetitions, 18 different normalized
    guesses, no repeat at all).

    Instead this evaluates *every* possible value of this chunk's
    remaining unknown bits (assumed to be `12 - n_counter_bits` extra
    callsign bits after the counter, matching chunk 0's layout) at the
    anchor, regenerates what each repetition's *own* counter-shifted
    12-bit value and Golay codeword would be for that hypothesis (the
    counter delta between repetitions is exact, see the module
    docstring), and scores each hypothesis by summing its predicted
    codeword's correlation against every repetition's actual
    chips -- combining evidence across repetitions *before* choosing,
    the same principle as the coherent chunk combining above, just
    applied to a field that varies (predictably) instead of one that
    doesn't. Cheap: (2**n_counter_bits) * n_extra_hyp * n_grid * 24
    multiply-adds, done once in the fallback path only.
    """
    clo, chi = chunk_idx * 24, (chunk_idx + 1) * 24
    anchor_frame = anchor_off // CHIPS_PER_FRAME
    received = np.array(
        [chips[pos + SYNC_LEN + clo : pos + SYNC_LEN + chi] for pos in grid]
    )  # (n_grid, 24)
    frame_deltas = np.array([pos // CHIPS_PER_FRAME - anchor_frame for pos in grid])

    n_extra = 12 - n_counter_bits
    counter_mask = (1 << n_counter_bits) - 1
    counters = np.arange(1 << n_counter_bits)
    best_score, best = -np.inf, None
    for extra in range(1 << n_extra):
        counter_vals = (counters[:, None] + frame_deltas[None, :]) & counter_mask
        data12 = (counter_vals << n_extra) | extra  # (2**n_counter_bits, n_grid)
        signs = golay._SIGNS[data12]  # (2**n_counter_bits, n_grid, 24)
        scores = np.einsum("hgc,gc->h", signs, received)
        h = int(np.argmax(scores))
        if scores[h] > best_score:
            best_score = float(scores[h])
            best = (int(counters[h]), extra)
    return best


def _decode_combined(chips: np.ndarray, anchor_off: int) -> BeaconResult | None:
    n = len(chips)
    grid = _repetition_grid(n, anchor_off)
    if len(grid) < 3:  # too few repetitions for voting to mean anything
        return None

    counter_lo = 0
    # The invariant region: every payload bit that is identical in every
    # repetition *and* unknown a priori -- callsign plus mode. (The
    # reserved field is invariant too, but its value is a protocol
    # constant, so it belongs to the verify chunks below rather than to
    # anything that needs recovering.)
    inv_lo = BEACON_COUNTER_BITS
    inv_hi = inv_lo + BEACON_CALLSIGN_BITS + BEACON_MODE_BITS

    invariant_chunks = [
        c for c in range(N_CHUNKS)
        if inv_lo <= _chunk_bit_range(c)[0] and _chunk_bit_range(c)[1] <= inv_hi
    ]
    variant_chunks = [
        c for c in range(N_CHUNKS)
        if c not in invariant_chunks and _chunk_bit_range(c)[0] < inv_hi
    ]

    # Recovered callsign+mode bits, indexed relative to inv_lo.
    inv_bits = np.full(inv_hi - inv_lo, -1, dtype=np.int64)

    # Coherent case: sum this chunk's coded chips across every
    # repetition (identical by construction), decode once. Sums the raw
    # chip values -- coherent combining, valid because the chips are
    # maximal-ratio rather than equalized; see the module docstring for
    # why this summed signs until 2026-08-25 and what changed.
    for c in invariant_chunks:
        clo, chi = c * 24, (c + 1) * 24
        summed = np.zeros(24)
        for pos in grid:
            summed += chips[pos + SYNC_LEN + clo : pos + SYNC_LEN + chi]
        bits = _decode_chunk_bits(summed)
        blo, bhi = _chunk_bit_range(c)
        inv_bits[blo - inv_lo : bhi - inv_lo] = bits

    # The chunk carrying the counter's own low bit 0 can't be summed
    # like the others -- a genuinely *different* 12-bit value is
    # correct at every repetition, not the same one imperfectly
    # received, so joint-search it instead (see _search_counter_chunk).
    counter_chunk = next(c for c in variant_chunks if _chunk_bit_range(c)[0] <= counter_lo)
    result = _search_counter_chunk(chips, grid, anchor_off, counter_chunk, BEACON_COUNTER_BITS)
    if result is None:
        return None
    frame_index, extra_bits_value = result
    if not (0 <= frame_index <= MAX_FRAME_COUNTER):
        return None
    n_extra = 12 - BEACON_COUNTER_BITS
    blo, bhi = _chunk_bit_range(counter_chunk)
    lo, hi = max(blo, inv_lo), min(bhi, inv_hi)
    if lo < hi:  # the counter chunk's tail bits are callsign, not counter
        inv_bits[lo - inv_lo : hi - inv_lo] = _int_to_bits(
            extra_bits_value, n_extra
        )[lo - (bhi - n_extra) :]

    if np.any(inv_bits < 0):
        return None  # a fragment nobody resolved (shouldn't happen given the layout)

    callsign_bits = inv_bits[:BEACON_CALLSIGN_BITS]
    mode_index = _bits_to_int(inv_bits[BEACON_CALLSIGN_BITS:])
    codes = [
        _bits_to_int(callsign_bits[i : i + BEACON_CALLSIGN_CHAR_BITS])
        for i in range(0, BEACON_CALLSIGN_BITS, BEACON_CALLSIGN_CHAR_BITS)
    ]
    callsign = codes_to_callsign(np.array(codes))

    # This assembly came from combined evidence, not a single verified
    # CRC, and needs a real check before it's trusted -- not a
    # correlation against the whole superframe, which is the wrong
    # thing to check against.
    #
    # Chunks 0-4 were *used* to build (frame_index, callsign, mode): the
    # searches above pick whichever hypothesis best matches the noisy
    # chips there, so a wrong hypothesis can still score deceptively
    # well on them -- it's not really testing anything independent, and
    # correlating against it measures how good a curve-fit the search
    # found, not whether the fit is *right*. Diluted into a whole-
    # superframe correlation this was silently catastrophic: measured on
    # pure noise long enough to hold this many repetitions, 40 of 40
    # trials produced a confident, fully-assembled, completely fake
    # BeaconResult before this fix, at what looked like a comfortable
    # multiple of a "3-sigma" bar -- exhaustively searching thousands of
    # hypotheses and keeping the best one produces exactly the inflated,
    # not-actually-3-sigma correlations extreme-value statistics predict.
    # Restricting the correlation to just the chunks nothing was
    # searched against (the two reserved/CRC chunks) fixed the false
    # locks, but a correlation *threshold* over so few chips (48, next
    # to the whole superframe's 181) turned out to cost most of the
    # measured gain: fading routinely knocks a real repetition's own
    # agreement down far enough that no threshold cleanly separates a
    # correct combine from a wrong one on 48 chips alone.
    #
    # A bit-exact check does, and needs no threshold to calibrate: Golay
    # -decode the verify chunks at each repetition (a plain, unsearched
    # decode, no hypothesis-fitting involved) and compare against what
    # `_payload_bits` says they should be for that repetition's own
    # counter -- if even *one* repetition matches exactly, accept. A
    # wrong hypothesis's chance of an exact coincidental match this way
    # is about (1/4096) per verify chunk (a Golay decode of noise is
    # effectively a uniform draw over its 4096 codewords), roughly
    # 6e-8 for both chunks in this layout at any one repetition -- comparable
    # to or better than the false-accept rate the original single-
    # superframe CRC check itself already relies on (1/65536 per
    # attempt), and unlike a correlation threshold it doesn't get
    # weaker just because fading make the chips noisier.
    #
    # In the v4 layout the verify chunks are 5 and 6: reserved+CRC and
    # pure CRC. The reserved field is predicted at its protocol constant
    # (BEACON_RESERVED_VALUE, via _payload_bits) -- which means a future
    # sender that assigns those bits loses only this combining fallback
    # on today's receivers, never the single-shot CRC path, exactly the
    # graceful-degradation trade recorded in config.py.
    verify_chunks = [c for c in range(N_CHUNKS) if _chunk_bit_range(c)[0] >= inv_hi]
    if not verify_chunks:
        return None  # layout has no untouched chunk to verify against -- refuse rather than guess
    verified = False
    for pos in grid:
        fi = frame_index + (pos // CHIPS_PER_FRAME - anchor_off // CHIPS_PER_FRAME)
        if not (0 <= fi <= MAX_FRAME_COUNTER):
            continue
        payload = _payload_bits(fi, callsign, mode_index)
        if all(
            np.array_equal(
                _decode_chunk_bits(chips[pos + SYNC_LEN + c * 24 : pos + SYNC_LEN + (c + 1) * 24]),
                payload[_chunk_bit_range(c)[0] : _chunk_bit_range(c)[1]],
            )
            for c in verify_chunks
        ):
            verified = True
            break
    if not verified:
        return None

    return BeaconResult(chip_offset=anchor_off, frame_index=frame_index,
                        callsign=callsign, mode_index=mode_index)

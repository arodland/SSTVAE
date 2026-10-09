"""Morse table and callsign-window keying units (spec 2.7, design 2.7).

A **format module**. Each callsign window is CW_UNITS = 192 Morse units
of CW_UNIT = 2 CE symbols (60.625 ms, 19.8 wpm). Units 0..7 are plain
carrier, the call starts at unit 8, and everything after it to unit 191
is plain carrier again; the call has CW_ROOM_UNITS = 176 units of room.

Standard international timing: dot 1 unit, dash 3, 1 unit between
elements, 3 between characters, no trailing gap. The alphabet is A-Z,
0-9 and '/', exactly the table of the spec's `sims/cwid/cwid.py`, and
nothing else is accepted -- a header callsign must be sendable in Morse
(decision D10), so an unsendable character is an error rather than a
silent substitution.
"""

import numpy as np

from .constants import CW_LEAD_UNITS, CW_ROOM_UNITS, CW_UNITS

MORSE: dict[str, str] = {
    "A": ".-", "B": "-...", "C": "-.-.", "D": "-..", "E": ".", "F": "..-.",
    "G": "--.", "H": "....", "I": "..", "J": ".---", "K": "-.-", "L": ".-..",
    "M": "--", "N": "-.", "O": "---", "P": ".--.", "Q": "--.-", "R": ".-.",
    "S": "...", "T": "-", "U": "..-", "V": "...-", "W": ".--", "X": "-..-",
    "Y": "-.--", "Z": "--..",
    "0": "-----", "1": ".----", "2": "..---", "3": "...--", "4": "....-",
    "5": ".....", "6": "-....", "7": "--...", "8": "---..", "9": "----.",
    "/": "-..-.",
}

MAX_CALL_CHARS = 8


def check_callsign(callsign: str) -> str:
    """The callsign with trailing spaces stripped, or ValueError.

    Accepts 1 to 8 characters from [A-Z0-9/]. Lowercase is rejected, not
    folded: the header carries what the operator typed, and a receiver's
    Morse must spell the same thing.
    """
    if not isinstance(callsign, str):
        raise ValueError(f"callsign must be a str, not {type(callsign).__name__}")
    call = callsign.rstrip(" ")
    if not 1 <= len(call) <= MAX_CALL_CHARS:
        raise ValueError(
            f"callsign {callsign!r} must be 1 to {MAX_CALL_CHARS} characters")
    bad = sorted({c for c in call if c not in MORSE})
    if bad:
        raise ValueError(
            f"callsign {callsign!r} has characters outside [A-Z0-9/]: {bad}")
    return call


def morse_keying(callsign: str) -> np.ndarray:
    """uint8 key-down (1) / key-up (0) per unit for the call itself."""
    call = check_callsign(callsign)
    out: list[int] = []
    for i, ch in enumerate(call):
        if i:
            out += [0, 0, 0]
        for j, el in enumerate(MORSE[ch]):
            if j:
                out.append(0)
            out += [1] * (1 if el == "." else 3)
    return np.array(out, dtype=np.uint8)


def keying_units(callsign: str) -> np.ndarray:
    """uint8[192]: k[u] = 1 for key-down, the call in units 8 .. 8+len-1.

    ValueError if the call does not fit its 176 units (it always does for
    a valid call: the longest, "00000000", takes 173).
    """
    k = morse_keying(callsign)
    if len(k) > CW_ROOM_UNITS:
        raise ValueError(
            f"callsign {callsign!r} needs {len(k)} Morse units, "
            f"more than the window's {CW_ROOM_UNITS}")
    out = np.zeros(CW_UNITS, dtype=np.uint8)
    out[CW_LEAD_UNITS:CW_LEAD_UNITS + len(k)] = k
    return out

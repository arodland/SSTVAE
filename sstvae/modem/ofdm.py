"""DFT-matrix OFDM: complex carrier amplitudes <-> waveform samples.

Passband modulation generates the real transmit waveform directly.
Demodulation operates on the complex baseband signal produced by
dsp.to_baseband, where carrier k sits at bin (k - 11) * RS Hz.
"""

import numpy as np

from ..config import (
    FS, RS, NC, M, NCP, NSYM, CARRIER0, FCENTER, PILOT_QUADRANTS,
)

CARRIER_FREQS = CARRIER0 + RS * np.arange(NC)  # passband, Hz
BASEBAND_FREQS = CARRIER_FREQS - FCENTER  # multiples of RS

# Every frequency here is an integer number of Hz and every sample index
# is an integer, so `n*f/FS` has an exact integer remainder and the
# phasor depends only on `(n*f) mod FS`. Reducing first, in integer
# arithmetic, is exact and keeps the argument to exp() under one turn.
#
# Without it |theta| reaches 262 rad, where one ulp is 5.7e-14, so the
# phasors carried ~3e-14 of error and *which* entries rounded which way
# was an accident of numpy's complex-array arithmetic. Two reasons that
# was worth fixing, neither of them the accuracy itself:
#
#   * sin/cos of a large argument disagree between libms -- and between
#     x86-64 and Apple silicon -- by far more than they do near zero,
#     because implementations differ in how far they carry argument
#     reduction. This made the tables non-reproducible across platforms.
#   * The C++ port checks itself against these values, and a tolerance
#     sized by the reference's error rather than the port's is a much
#     weaker statement. See docs/native-app.md.
def _phasor(cycles_num: np.ndarray, sign: int = 1) -> np.ndarray:
    """exp(sign * 2j*pi * cycles_num / FS) for integer `cycles_num`."""
    return np.exp(sign * 2j * np.pi * (np.asarray(cycles_num) % FS) / FS)


# Passband modulation matrix: symbol samples n=0..NSYM-1, phase reference
# at the start of the useful part (n=NCP). Carriers are multiples of RS,
# so the first NCP samples are a true cyclic prefix.
_n_sym = np.arange(NSYM) - NCP
MOD_MATRIX = _phasor(np.outer(_n_sym, CARRIER_FREQS))  # (NSYM, NC)

# Baseband demod matrix over one useful window (M samples).
_n_use = np.arange(M)
DEMOD_MATRIX = _phasor(np.outer(BASEBAND_FREQS, _n_use), -1)  # (NC, M)


def modulate_symbols(symbols: np.ndarray) -> np.ndarray:
    """(n_sym, NC) complex -> real waveform (n_sym * NSYM,)."""
    x = np.real(MOD_MATRIX @ symbols.T)  # (NSYM, n_sym)
    return x.T.reshape(-1)


def demod_window(z: np.ndarray, start: int, backoff: int = 0) -> np.ndarray:
    """Demodulate one useful window of baseband signal starting at `start`
    (nominal index of the first useful sample). `backoff` shifts the window
    earlier into the cyclic prefix; the resulting linear phase slope is
    absorbed by pilot equalization as long as it is applied consistently.
    Factor 2 undoes the amplitude halving of the real->analytic conversion.
    """
    s = start - backoff
    win = z[s : s + M]
    if len(win) < M:
        win = np.pad(win, (0, M - len(win)))
    return (2.0 / M) * (DEMOD_MATRIX @ win)


def pilot_sequence() -> np.ndarray:
    """Fixed unit-magnitude QPSK sequence used for preamble and frame pilots.

    Built from the frozen `config.PILOT_QUADRANTS` rather than re-drawn
    from `np.random.default_rng(PILOT_SEED)`. This sequence is part of
    the on-air format: if a future numpy changed its generator stream,
    the right behaviour is to keep transmitting the same pilots, not to
    follow numpy. See the note in config.py.
    """
    phases = np.pi / 4 + np.pi / 2 * np.asarray(PILOT_QUADRANTS)
    return np.exp(1j * phases)


def preamble_waveform() -> np.ndarray:
    """Real passband preamble: pilot symbol, periodic with M over the whole
    block (double-length CP + two periods)."""
    from ..config import PREAMBLE_CP, PREAMBLE_SAMPLES

    p = pilot_sequence()
    n = np.arange(PREAMBLE_SAMPLES) - PREAMBLE_CP
    e = _phasor(np.outer(n, CARRIER_FREQS))
    return np.real(e @ p)


def preamble_template() -> np.ndarray:
    """Complex baseband replica of the preamble (for timing correlation)."""
    from ..config import PREAMBLE_CP, PREAMBLE_SAMPLES

    p = pilot_sequence()
    n = np.arange(PREAMBLE_SAMPLES) - PREAMBLE_CP
    e = _phasor(np.outer(n, BASEBAND_FREQS))
    return 0.5 * (e @ p)


def pilot_template() -> np.ndarray:
    """Complex baseband replica of one bare frame-pilot symbol's useful
    window (no CP) — used by sync.acquire_blind() to find frame timing
    purely from the pilot's own per-frame periodicity, without needing
    the (non-repeating) transmission-start preamble."""
    p = pilot_sequence()
    n = np.arange(M)
    e = _phasor(np.outer(n, BASEBAND_FREQS))
    return 0.5 * (e @ p)

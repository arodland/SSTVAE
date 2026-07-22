"""DFT-matrix OFDM: complex carrier amplitudes <-> waveform samples.

Passband modulation generates the real transmit waveform directly.
Demodulation operates on the complex baseband signal produced by
dsp.to_baseband, where carrier k sits at bin (k - 11) * RS Hz.
"""

import numpy as np

from ..config import FS, RS, NC, M, NCP, NSYM, CARRIER0, FCENTER, PILOT_SEED

CARRIER_FREQS = CARRIER0 + RS * np.arange(NC)  # passband, Hz
BASEBAND_FREQS = CARRIER_FREQS - FCENTER  # multiples of RS

# Passband modulation matrix: symbol samples n=0..NSYM-1, phase reference
# at the start of the useful part (n=NCP). Carriers are multiples of RS,
# so the first NCP samples are a true cyclic prefix.
_n_sym = np.arange(NSYM) - NCP
MOD_MATRIX = np.exp(2j * np.pi * np.outer(_n_sym, CARRIER_FREQS) / FS)  # (NSYM, NC)

# Baseband demod matrix over one useful window (M samples).
_n_use = np.arange(M)
DEMOD_MATRIX = np.exp(-2j * np.pi * np.outer(BASEBAND_FREQS, _n_use) / FS)  # (NC, M)


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
    """Fixed unit-magnitude QPSK sequence used for preamble and frame pilots."""
    rng = np.random.default_rng(PILOT_SEED)
    phases = np.pi / 4 + np.pi / 2 * rng.integers(0, 4, NC)
    return np.exp(1j * phases)


def preamble_waveform() -> np.ndarray:
    """Real passband preamble: pilot symbol, periodic with M over the whole
    block (double-length CP + two periods)."""
    from ..config import PREAMBLE_CP, PREAMBLE_SAMPLES

    p = pilot_sequence()
    n = np.arange(PREAMBLE_SAMPLES) - PREAMBLE_CP
    e = np.exp(2j * np.pi * np.outer(n, CARRIER_FREQS) / FS)
    return np.real(e @ p)


def preamble_template() -> np.ndarray:
    """Complex baseband replica of the preamble (for timing correlation)."""
    from ..config import PREAMBLE_CP, PREAMBLE_SAMPLES

    p = pilot_sequence()
    n = np.arange(PREAMBLE_SAMPLES) - PREAMBLE_CP
    e = np.exp(2j * np.pi * np.outer(n, BASEBAND_FREQS) / FS)
    return 0.5 * (e @ p)

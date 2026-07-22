"""Small DSP helpers shared by the modem."""

import numpy as np
from scipy import signal

from ..config import FS, FCENTER, TX_BANDPASS


def to_baseband(x: np.ndarray) -> np.ndarray:
    """Real passband -> complex baseband (pure heterodyne by FCENTER).

    Deliberately unfiltered: any FIR long enough to be selective smears
    beyond the 32-sample cyclic prefix and causes ISI, while the
    160-sample demod correlation already nulls the -f-FCENTER heterodyne
    image exactly (all image spacings are carrier-spacing multiples) and
    provides per-carrier noise selectivity. Sync filters its own copy.
    """
    n = np.arange(len(x))
    return x.astype(np.float64) * np.exp(-2j * np.pi * FCENTER * n / FS)


def sync_lowpass(z: np.ndarray) -> np.ndarray:
    """Selective lowpass used only for preamble detection, where FIR
    smearing is harmless and out-of-band noise would degrade the
    autocorrelation metric."""
    taps = signal.firwin(129, 850.0, fs=FS)
    return np.convolve(z, taps, mode="same")


def freq_correct(z: np.ndarray, f_hz: float) -> np.ndarray:
    n = np.arange(len(z))
    return z * np.exp(-2j * np.pi * f_hz * n / FS)


def tx_condition(x: np.ndarray, clip_headroom_db: float, iterations: int = 2) -> np.ndarray:
    """Envelope clip-and-filter for PAPR (PEP) control.

    SSB transmitters are limited by envelope peak power, so clipping acts
    on the analytic-signal magnitude, not raw samples. Iterated because
    the bandpass regrows peaks after each clip. Returns unit-RMS floats.
    """
    power = np.mean(x**2)
    if power == 0:
        return x
    # mean envelope power is 2x mean real power
    thresh = np.sqrt(2 * power) * 10 ** (clip_headroom_db / 20)
    taps = signal.firwin(201, TX_BANDPASS, fs=FS, pass_zero=False)
    for _ in range(iterations):
        z = signal.hilbert(x)
        mag = np.abs(z)
        scale = np.minimum(1.0, thresh / np.maximum(mag, 1e-12))
        x = np.real(z * scale)
        x = np.convolve(x, taps, mode="same")
    return x / np.sqrt(np.mean(x**2))


def papr_db(x: np.ndarray) -> float:
    """Envelope (PEP) peak-to-average power ratio in dB."""
    env2 = np.abs(signal.hilbert(x)) ** 2
    return 10 * np.log10(np.max(env2) / np.mean(env2))


def to_int16(x: np.ndarray, peak: float = 0.95) -> np.ndarray:
    return np.round(x / np.max(np.abs(x)) * peak * 32767).astype(np.int16)

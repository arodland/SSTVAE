"""8 kHz mono WAV read/write helpers."""

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly

from .config import FS


def read_wav(path: str) -> np.ndarray:
    """Read a WAV file as float64 mono at FS, resampling if needed."""
    rate, data = wavfile.read(path)
    data = np.asarray(data)
    # Scale *before* mixing down: `mean` returns float, so testing the
    # dtype afterwards silently skipped normalization for every stereo
    # integer file and handed back samples in the +-32767 range. The
    # modem is scale-invariant enough that this decoded anyway, which is
    # why it went unnoticed.
    if np.issubdtype(data.dtype, np.integer):
        data = data / float(np.iinfo(data.dtype).max)
    data = np.asarray(data, dtype=np.float64)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if rate != FS:
        from math import gcd

        g = gcd(FS, rate)
        data = resample_poly(data, FS // g, rate // g)
    return data


def write_wav_float(path: str, x: np.ndarray) -> None:
    """Write float32 at FS with **no normalization and no quantization**.

    For diagnostics, where the question is "what exactly did we
    capture?" -- `write_wav` rescales to a fixed peak and rounds to
    int16, both of which destroy the evidence. `read_wav` returns these
    unchanged, so a dump round-trips exactly.
    """
    wavfile.write(path, FS, np.asarray(x, dtype=np.float32))


def write_wav(path: str, x: np.ndarray, peak: float = 0.95) -> None:
    m = np.max(np.abs(x))
    if m > 0:
        x = x / m * peak
    wavfile.write(path, FS, np.round(x * 32767).astype(np.int16))

"""8 kHz mono WAV read/write helpers."""

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly

from .config import FS


def read_wav(path: str) -> np.ndarray:
    """Read a WAV file as float64 mono at FS, resampling if needed."""
    rate, data = wavfile.read(path)
    data = np.asarray(data)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if np.issubdtype(data.dtype, np.integer):
        data = data / float(np.iinfo(data.dtype).max)
    data = data.astype(np.float64)
    if rate != FS:
        from math import gcd

        g = gcd(FS, rate)
        data = resample_poly(data, FS // g, rate // g)
    return data


def write_wav(path: str, x: np.ndarray, peak: float = 0.95) -> None:
    m = np.max(np.abs(x))
    if m > 0:
        x = x / m * peak
    wavfile.write(path, FS, np.round(x * 32767).astype(np.int16))

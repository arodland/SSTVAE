"""NumPy HF channel simulator: AWGN, Watterson-style fading, frequency
offset, sample-clock error. Operates on real passband audio at FS.

SNR follows the FreeDV convention: signal power relative to the noise
power falling in a 3000 Hz bandwidth.
"""

from dataclasses import dataclass

import numpy as np
from scipy import signal

from .config import FS


@dataclass(frozen=True)
class FadingPreset:
    name: str
    doppler_hz: float  # two-sided Doppler spread
    delay_ms: float  # second-path delay


FADING_PRESETS = {
    "mpg": FadingPreset("mpg", 0.1, 0.5),  # good
    "mpp": FadingPreset("mpp", 1.0, 2.0),  # poor (CCIR)
    "mpd": FadingPreset("mpd", 2.0, 4.0),  # disturbed
}


def _analytic(x: np.ndarray) -> np.ndarray:
    return signal.hilbert(x)


def freq_shift(x: np.ndarray, df_hz: float) -> np.ndarray:
    n = np.arange(len(x))
    return np.real(_analytic(x) * np.exp(2j * np.pi * df_hz * n / FS))


def sample_clock_offset(x: np.ndarray, ppm: float) -> np.ndarray:
    """Resample as if the far-end clock ran (1 + ppm*1e-6) fast."""
    t_out = np.arange(len(x)) * (1 + ppm * 1e-6)
    t_out = t_out[t_out <= len(x) - 1]
    return np.interp(t_out, np.arange(len(x)), x)


def _rayleigh_taps(n: int, doppler_hz: float, rng: np.random.Generator) -> np.ndarray:
    """Complex Gaussian tap gains with ~Gaussian Doppler spectrum, unit power."""
    lowrate = max(8 * doppler_hz, 1.0)
    n_low = int(np.ceil(n * lowrate / FS)) + 8
    g = rng.normal(size=n_low) + 1j * rng.normal(size=n_low)
    b, a = signal.butter(2, min(doppler_hz / (lowrate / 2), 0.99))
    g = signal.lfilter(b, a, g)
    g = g[4:]  # drop filter transient
    t_low = np.arange(len(g)) * (FS / lowrate)
    t = np.arange(n)
    tap = np.interp(t, t_low, g.real) + 1j * np.interp(t, t_low, g.imag)
    return tap / np.sqrt(np.mean(np.abs(tap) ** 2))


def fading(x: np.ndarray, preset: str | FadingPreset, seed: int = 0) -> np.ndarray:
    """Two independent equal-power Rayleigh paths (Watterson model)."""
    p = FADING_PRESETS[preset] if isinstance(preset, str) else preset
    rng = np.random.default_rng(seed)
    z = _analytic(x)
    delay = int(round(p.delay_ms * 1e-3 * FS))
    g1 = _rayleigh_taps(len(z), p.doppler_hz, rng)
    g2 = _rayleigh_taps(len(z), p.doppler_hz, rng)
    z2 = np.concatenate([np.zeros(delay, dtype=complex), z[: len(z) - delay]])
    return np.real((z * g1 + z2 * g2) / np.sqrt(2))


def awgn(x: np.ndarray, snr_db: float, seed: int = 0) -> np.ndarray:
    """Add white noise for the given SNR in a 3000 Hz noise bandwidth.

    Signal power is measured over the active portion (envelope above 10%
    of the overall RMS) so lead-in/out silence doesn't skew it.
    """
    rng = np.random.default_rng(seed)
    env = np.abs(_analytic(x))
    active = env > 0.1 * np.sqrt(np.mean(x**2))
    s_power = np.mean(x[active] ** 2) if active.any() else np.mean(x**2)
    # White noise over FS/2 Hz with total power sigma^2 puts
    # sigma^2 * 3000 / (FS/2) into 3000 Hz.
    sigma2 = s_power * (FS / 2) / 3000.0 / 10 ** (snr_db / 10)
    return x + rng.normal(scale=np.sqrt(sigma2), size=len(x))


def zero_spans(x: np.ndarray, spans_s: list[tuple[float, float]]) -> np.ndarray:
    """Blank out time spans (seconds) — simulates lost/blocked frames."""
    y = x.copy()
    for a, b in spans_s:
        y[int(a * FS) : int(b * FS)] = 0.0
    return y


def apply_channel(
    x: np.ndarray,
    snr_db: float | None = None,
    freq_offset_hz: float = 0.0,
    ppm: float = 0.0,
    fading_preset: str | None = None,
    spans: list[tuple[float, float]] | None = None,
    seed: int = 0,
) -> np.ndarray:
    y = x.astype(np.float64)
    if ppm:
        y = sample_clock_offset(y, ppm)
    if freq_offset_hz:
        y = freq_shift(y, freq_offset_hz)
    if fading_preset:
        y = fading(y, fading_preset, seed=seed)
    if spans:
        y = zero_spans(y, spans)
    if snr_db is not None:
        y = awgn(y, snr_db, seed=seed + 1)
    return y

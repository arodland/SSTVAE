"""NumPy HF channel simulator: AWGN, Watterson-style fading, frequency
offset, sample-clock error. Operates on real passband audio at FS.

SNR is signal power relative to the noise power falling in
`config.SNR_REF_BW_HZ`.
"""

from dataclasses import dataclass

import numpy as np
from scipy import signal

from .config import (
    FS,
    FRAME_SAMPLES,
    LEADIN_SAMPLES,
    PREAMBLE_SAMPLES,
    HEADER_SAMPLES,
    MODES,
    SNR_REF_BW_HZ,
    ModeSpec,
)


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
    """Add white noise for the given SNR in a `SNR_REF_BW_HZ` bandwidth.

    Signal power is measured over the active portion (envelope above 10%
    of the overall RMS) so lead-in/out silence doesn't skew it.
    """
    rng = np.random.default_rng(seed)
    env = np.abs(_analytic(x))
    active = env > 0.1 * np.sqrt(np.mean(x**2))
    s_power = np.mean(x[active] ** 2) if active.any() else np.mean(x**2)
    # White noise over FS/2 Hz with total power sigma^2 puts
    # sigma^2 * SNR_REF_BW_HZ / (FS/2) into the reference bandwidth.
    sigma2 = s_power * (FS / 2) / SNR_REF_BW_HZ / 10 ** (snr_db / 10)
    return x + rng.normal(scale=np.sqrt(sigma2), size=len(x))


def zero_spans(x: np.ndarray, spans_s: list[tuple[float, float]]) -> np.ndarray:
    """Blank out time spans (seconds) — simulates lost/blocked frames."""
    y = x.copy()
    for a, b in spans_s:
        y[int(a * FS) : int(b * FS)] = 0.0
    return y


def detect_mode_by_length(x: np.ndarray, tol_samples: int = 4) -> ModeSpec:
    """Identify which mode produced this waveform purely from its sample
    count — exact for audio straight out of sstvae_encode.py, which is
    the only case data_sample_mask()/apply_channel_data_only() support
    (they rely on TX's fixed, known layout, not on re-acquiring sync)."""
    n = len(x)
    for spec in MODES.values():
        if abs(n - round(spec.duration_s * FS)) <= tol_samples:
            return spec
    raise ValueError(
        f"{n} samples doesn't match any mode's expected length "
        f"(A={round(MODES['A'].duration_s*FS)}, "
        f"B={round(MODES['B'].duration_s*FS)}, "
        f"C={round(MODES['C'].duration_s*FS)}); pass mode explicitly "
        "if this wasn't produced by sstvae_encode.py as-is"
    )


def data_sample_mask(mode: ModeSpec, n_samples: int) -> np.ndarray:
    """Boolean mask over a TX waveform from sstvae_encode.py: True over
    every frame (pilot symbols AND data together), False over lead-in/
    out, the preamble, and the header.

    Only the preamble and header are protected — what acquisition and
    header decode actually need to always succeed. Per-frame pilots are
    corrupted right along with their frame's data on purpose: the
    demodulator estimates each frame's channel from its pilot and uses
    that to equalize the neighboring data, so a clean pilot next to
    corrupted data would make it confidently apply the *wrong*
    correction (implying the channel is clean when it wasn't) — a
    self-inflicted decode failure, not a realistic one. Letting fading/
    noise affect pilots and data together keeps equalization physically
    consistent; it can still legitimately struggle under extreme
    fading, but that's real degradation, not an artifact of this mask.
    """
    mask = np.zeros(n_samples, dtype=bool)
    start = LEADIN_SAMPLES + PREAMBLE_SAMPLES + HEADER_SAMPLES
    end = start + mode.n_frames * FRAME_SAMPLES
    mask[start:end] = True
    return mask[:n_samples]


def apply_channel_data_only(
    x: np.ndarray,
    mode: ModeSpec | None = None,
    snr_db: float | None = None,
    freq_offset_hz: float = 0.0,
    fading_preset: str | None = None,
    spans: list[tuple[float, float]] | None = None,
    seed: int = 0,
) -> np.ndarray:
    """Like apply_channel, but the preamble and header are spliced back
    in clean afterward, so acquisition and header decode always succeed
    no matter how extreme snr_db/fading_preset are. Per-frame pilots are
    NOT protected — they're corrupted along with their frame's data on
    purpose, since the demodulator equalizes data using its frame's own
    pilot; protecting pilots but not data would make the equalizer
    confidently apply a wrong correction (see data_sample_mask's
    docstring). This is for visualizing worst-case data corruption
    while guaranteeing a lock, with equalization still behaving
    physically consistently — not a fully realistic channel (a real one
    can't protect the preamble/header this way either), but not
    self-defeating like protecting pilots would be.

    freq_offset_hz is applied globally to the whole composite signal
    afterward (a stable LO offset realistically affects everything and
    is what the pilots/header are there to estimate and correct, so it
    isn't isolated like the noise/fading terms).

    No ppm support: sample-clock resampling shifts alignment, which
    breaks the fixed-layout assumption this function depends on.
    """
    if mode is None:
        mode = detect_mode_by_length(x)
    dirty = apply_channel(
        x, snr_db=snr_db, fading_preset=fading_preset, spans=spans, seed=seed
    )
    mask = data_sample_mask(mode, len(x))
    y = np.where(mask, dirty, x)
    if freq_offset_hz:
        y = freq_shift(y, freq_offset_hz)
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

"""Acquisition: preamble detection, timing, and carrier frequency offset.

The preamble is periodic with M samples, so an autocorrelation at lag M
gives detection plus a fractional CFO estimate that is unambiguous over
+/- FS/(2M) = +/-25 Hz. The remaining offset is a multiple of the 50 Hz
carrier spacing, resolved by trying integer-bin candidates against the
known preamble template. Net tolerance comfortably exceeds +/-50 Hz.
"""

from dataclasses import dataclass

import numpy as np
from scipy import signal

from ..config import FS, M, PREAMBLE_CP, PREAMBLE_SAMPLES
from .dsp import freq_correct, sync_lowpass
from .ofdm import preamble_template


class SyncError(Exception):
    pass


@dataclass
class Acquisition:
    preamble_start: int  # index of first preamble sample (CP start)
    freq_offset: float  # Hz
    metric: float  # detection confidence, ~0..1


def _autocorr_metric(z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sliding lag-M autocorrelation over an M-sample window."""
    prod = z[M:] * np.conj(z[:-M])
    power = np.abs(z) ** 2
    kernel = np.ones(M)
    a = signal.fftconvolve(prod, kernel, mode="valid")  # A[n] over window n..n+M
    e1 = signal.fftconvolve(power[:-M], kernel, mode="valid")
    e2 = signal.fftconvolve(power[M:], kernel, mode="valid")
    # Floor the energies at a fraction of the typical window energy so
    # near-silent regions (filter ringing) can't produce inflated metrics.
    floor = 1e-3 * M * np.mean(power)
    energy = np.sqrt(np.maximum(e1, floor) * np.maximum(e2, floor)) + 1e-12
    return np.abs(a) / energy, a


def acquire(
    z: np.ndarray,
    threshold: float = 0.5,
    max_bins: int = 2,
    search: tuple[int, int] | None = None,
) -> Acquisition:
    """Find the preamble in baseband signal z.

    `search` optionally restricts the preamble hunt to a [start, end)
    sample range (the rest of the signal is still used for frames).
    """
    if len(z) < PREAMBLE_SAMPLES + 2 * M:
        raise SyncError("signal too short")

    z = sync_lowpass(z)
    metric, a = _autocorr_metric(z)
    if search is not None:
        s0 = max(0, int(search[0]))
        s1 = min(len(metric), int(search[1]))
        if s1 - s0 < 1:
            raise SyncError(f"empty search window {search}")
        masked = np.full_like(metric, -1.0)
        masked[s0:s1] = metric[s0:s1]
        metric = masked
    n_star = int(np.argmax(metric))
    if metric[n_star] < threshold:
        raise SyncError(f"no preamble found (peak metric {metric[n_star]:.2f})")

    f_frac = np.angle(a[n_star]) / (2 * np.pi * M / FS)

    # Integer-bin CFO search + fine timing via template correlation.
    template = preamble_template()
    t_norm = np.sqrt(np.sum(np.abs(template) ** 2))
    lo = max(0, n_star - PREAMBLE_CP - 200)
    hi = min(len(z) - PREAMBLE_SAMPLES, n_star + 200)
    if hi <= lo:
        raise SyncError("preamble at signal edge")
    seg = z[lo : hi + PREAMBLE_SAMPLES]

    best = None
    for m_bin in range(-max_bins, max_bins + 1):
        f_cand = f_frac + m_bin * FS / M
        seg_c = freq_correct(seg, f_cand)
        corr = signal.fftconvolve(seg_c, np.conj(template[::-1]), mode="valid")
        peak = int(np.argmax(np.abs(corr)))
        seg_energy = np.sqrt(
            np.sum(np.abs(seg_c[peak : peak + PREAMBLE_SAMPLES]) ** 2)
        )
        score = np.abs(corr[peak]) / (t_norm * seg_energy + 1e-12)
        if best is None or score > best[0]:
            best = (score, lo + peak, f_cand)

    _, p0, f_hat = best

    # Refine CFO from the phase between the two preamble periods at the
    # now-known timing (same lag-M estimate, but noise-averaged at the
    # exact alignment).
    u0 = p0 + PREAMBLE_CP
    zc = freq_correct(z[u0 : u0 + 2 * M], f_hat)
    if len(zc) == 2 * M:
        d = np.sum(zc[M:] * np.conj(zc[:M]))
        if np.abs(d) > 0:
            f_hat += np.angle(d) / (2 * np.pi * M / FS)

    return Acquisition(preamble_start=p0, freq_offset=f_hat, metric=float(metric[n_star]))

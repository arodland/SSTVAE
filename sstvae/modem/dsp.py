"""Small DSP helpers shared by the modem."""

from math import gcd

import numpy as np
from scipy import signal

from ..config import FS, FCENTER, TX_BANDPASS

# The heterodyne is exactly periodic: FCENTER/FS reduces to 3/16, so
# there are only 16 distinct phasors, and `n` never has to reach exp()
# at all.
#
# This is the same range-reduction argument as in ofdm.py but it matters
# far more here, because `n` runs over a whole recording rather than one
# symbol. Measured over a mode C transmission (760k samples), the old
# form reached |theta| = 895,000 rad -- where one ulp is 1.16e-10 --
# and accumulated 1.47e-10 rad of phase error, ~5000x the error in the
# OFDM matrices. Still nothing on air (1e-10 rad is 6e-9 degrees), but
# it was the largest numerical defect in the receiver and the one most
# exposed to differences between platforms' sin/cos.
#
# Derived from the config rather than hardcoded as 3/16, so a change to
# FCENTER or FS stays correct instead of silently producing a table for
# the wrong frequency.
_HET_G = gcd(FCENTER, FS)
_HET_PERIOD = FS // _HET_G  # 16
_HET_STEP = FCENTER // _HET_G  # 3
# k/period is exact for a power-of-two period, so these 16 values are as
# accurate as exp() can be.
_HET_TABLE = np.exp(-2j * np.pi * np.arange(_HET_PERIOD) / _HET_PERIOD)


def to_baseband(x: np.ndarray) -> np.ndarray:
    """Real passband -> complex baseband (pure heterodyne by FCENTER).

    Deliberately unfiltered: any FIR long enough to be selective smears
    beyond the 32-sample cyclic prefix and causes ISI, while the
    160-sample demod correlation already nulls the -f-FCENTER heterodyne
    image exactly (all image spacings are carrier-spacing multiples) and
    provides per-carrier noise selectivity. Sync filters its own copy.
    """
    n = np.arange(len(x))
    return x.astype(np.float64) * _HET_TABLE[(_HET_STEP * n) % _HET_PERIOD]


def to_baseband_at(x: np.ndarray, start_sample: int) -> np.ndarray:
    """`to_baseband(x)`, but as if `x` were a slice of a longer signal
    starting at absolute sample index `start_sample` rather than at 0 --
    i.e. exactly `to_baseband(full)[start_sample : start_sample + len(x)]`
    for whatever longer `full` array `x` actually came from.

    `to_baseband` always treats its own input's first sample as the
    heterodyne's local n=0, which only matches the true carrier phase
    when `start_sample` happens to be 0. A caller that baseband-converts
    one long recording as a series of independent chunks (rather than
    the whole thing in one call) needs this instead, or consecutive
    chunks disagree about the carrier phase by an amount that depends on
    where the chunk boundary fell -- invisible to code that only ever
    looks within one chunk, and it matters only to code that carries
    state *across* chunks the way sync.BlindAccumulator does.

    Cheap because the heterodyne is periodic in 16 samples (see
    to_baseband's own docstring): the correction needed is one constant
    phasor for the whole chunk (`_HET_TABLE` composes multiplicatively
    under addition of its index), not something that varies through it.
    """
    correction = _HET_TABLE[(_HET_STEP * start_sample) % _HET_PERIOD]
    return correction * to_baseband(x)


def sync_lowpass(z: np.ndarray) -> np.ndarray:
    """Selective lowpass used only for preamble detection, where FIR
    smearing is harmless and out-of-band noise would degrade the
    autocorrelation metric.

    FFT-based rather than a direct sum: `acquire()` runs this over the
    whole ring-buffer snapshot (up to the full ~130 s capacity) on every
    poll, not just a bounded search window, and the direct convolution
    was measured as the single largest item in a live decode-loop
    profile -- a per-poll cost that scaled with total buffer duration
    rather than with anything acquisition actually needs. `convolve_same`
    stays a direct sum for its other caller (`tx_condition`'s clip
    filter), which runs once per transmit, not every poll."""
    taps = signal.firwin(129, 850.0, fs=FS)
    return signal.fftconvolve(z, taps, mode="same")


def wrap_cycles(cycles: np.ndarray) -> np.ndarray:
    """Fractional part of a phase expressed in cycles, in [0, 1).

    For arbitrary (non-integer) frequencies the product cannot be made
    exact the way the integer cases above can, but reducing it before it
    reaches exp() still removes the large-argument error entirely and
    leaves only the rounding already present in the product itself.
    """
    return cycles - np.floor(cycles)


def freq_correct(z: np.ndarray, f_hz: float) -> np.ndarray:
    n = np.arange(len(z))
    return z * np.exp(-2j * np.pi * wrap_cycles(f_hz * n / FS))


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

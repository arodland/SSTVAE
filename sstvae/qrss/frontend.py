"""Receiver front end (design 6.2, WP5): audio <-> FE, impulse blanker,
inverse AGC, channeliser, noise floor, and the 48 h passband store.

Not a format module: nothing here goes on the air, and nothing here
draws random numbers either.

Rates (design 6.1):

    8 kHz audio --audio_to_fe--> FE 4000 Hz complex, 0 Hz = 1500 Hz audio
    FE --blank, normalise--> FE'          (impulses out, receive AGC undone)
    FE' --channelise(f_mix)--> CH 250 Hz complex, +-125 Hz about one signal

**Blanker** (spec 7: "blanked sample by sample before anything else").
nu(t) is the running median of |fe|^2 over 1 s windows at a 50 ms hop,
interpolated; a sample is blanked where |fe|^2 > k^2 nu (k = 5, 14 dB),
with a 2 ms guard each side and a 1 ms raised-cosine taper beyond that.
A constant-envelope signal raises the median exactly as much as it
raises every sample, so CE itself is never blanked however strong it is.

**Inverse AGC.** A fast receive AGC multiplies signal and noise by one
g(t). At the SNRs this mode lives at the wideband power is noise, so
dividing FE' by the square root of its local median power (100 ms
windows, blanked samples left out) removes g(t) and leaves the noise
flat; the signal's residual gain change is common to every symbol in a
window and the tracker follows it.

**Channeliser.** Mixing is exact integer turn arithmetic on a 1/64 Hz
grid, (n f_num mod 64 fs_in)/(64 fs_in) (design 0, rule 5), then
`resample_poly` down by 16. CH sample k is FE sample 16k: the channel
keeps the FE time origin.

**PassbandStore** (spec 7, design 6.9) keeps the whole FE stream for
48 h as int16 I/Q in hourly files, so that passes too weak to detect
when they arrived can be found later by template search.
"""

from __future__ import annotations

import functools
import json
import math
import os
import re
import threading
import time
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy import ndimage, signal

from sstvae import config as _c
from sstvae.modem import dsp

from . import rules
from .constants import CH_FS, FE_CENTER_HZ, FE_FS, FS

_FE_DECIM = FS // FE_FS                       # 2
_FE_TAPS = signal.firwin(255, 1250.0, fs=FS, window=("kaiser", 8.0))
# `noise_psd` takes noise only from bins the front end's low-pass leaves
# within this of white (it is -0.10 dB at 310 Hz, -0.18 at 305, -0.29 at
# 300 and -6.0 at 250, mirrored about 1500 Hz)
FE_FLAT_DB = 0.1
# ...unless neighbours leave fewer than FE_MIN_HZ of those clear: then
# the skirt's clear bins down to FE_SKIRT_DB, compensated for the
# low-pass (`_fe_response`), are weighed against them
FE_MIN_HZ = 3.0
FE_SKIRT_DB = 20.0
F_MIX_DEN = 64                                # channeliser mixing grid: 1/64 Hz


# --- audio <-> FE ----------------------------------------------------------------------

def audio_to_fe(x8k) -> np.ndarray:
    """8 kHz real audio -> complex64 FE at 4 kHz, 0 Hz = 1500 Hz audio.

    x*sqrt(2)*exp(-j 2 pi 1500 n/8000) through `dsp.to_baseband`'s exact
    16-phasor table, a zero-phase FIR low-pass at 1250 Hz (255 taps,
    Kaiser 8) and decimation by 2. A real sine of power P comes out as a
    complex tone of power P: power is preserved.
    """
    assert _c.FCENTER == FE_CENTER_HZ
    z = np.sqrt(2.0) * dsp.to_baseband(np.asarray(x8k, dtype=np.float64))
    z = signal.oaconvolve(z, _FE_TAPS, mode="same")
    return z[::_FE_DECIM].astype(np.complex64)


def fe_to_audio(fe) -> np.ndarray:
    """Inverse of `audio_to_fe`: upsample by 2, mix up to 1500 Hz, sqrt(2)*Re."""
    fe = np.asarray(fe, dtype=np.complex128)
    up = np.zeros(len(fe) * _FE_DECIM, dtype=np.complex128)
    up[::_FE_DECIM] = fe * _FE_DECIM
    up = signal.oaconvolve(up, _FE_TAPS, mode="same")
    # mix up: the conjugate of to_baseband's exact 16-phasor table
    return np.sqrt(2.0) * np.real(np.conj(dsp.to_baseband(np.ones(len(up)))) * up)


def fe_hz(audio_hz: float) -> float:
    """Audio frequency -> frequency inside the FE stream (0 Hz = 1500 Hz)."""
    return float(audio_hz) - FE_CENTER_HZ


@dataclass
class Capture:
    """A front-end capture of one slot.

    t0_index: FE sample index of the nominal t0 (QH + 1 s) on the
    receiver's clock; fractional values are allowed.
    """
    fe: np.ndarray                           # complex64
    q: int
    t0_index: float
    fs: int = FE_FS


# --- blanker and inverse AGC -------------------------------------------------------------

def running_median(p, fs: float, win_s: float = 1.0, hop_s: float = 0.05,
                   chunk: int = 256) -> tuple[np.ndarray, np.ndarray]:
    """(centres, medians): median of p over win_s windows every hop_s.

    Windows that would run off either end are shifted inward, so every
    median is over a full window (or over all of p when it is shorter).
    """
    p = np.asarray(p, dtype=np.float32)
    n = len(p)
    hop = max(1, int(round(hop_s * fs)))
    w = max(1, int(round(win_s * fs)))
    centres = np.arange(hop // 2, max(n, 1), hop)
    if n <= w:
        return centres, np.full(len(centres), np.median(p) if n else 0.0, np.float32)
    starts = np.clip(centres - w // 2, 0, n - w)
    view = np.lib.stride_tricks.sliding_window_view(p, w)
    med = np.empty(len(centres), dtype=np.float32)
    for a in range(0, len(starts), chunk):
        med[a:a + chunk] = np.median(view[starts[a:a + chunk]], axis=1)
    return centres, med


def _interp_to_samples(n: int, centres, vals, chunk: int = 1 << 20) -> np.ndarray:
    out = np.empty(n, dtype=np.float32)
    for a in range(0, n, chunk):
        out[a:a + chunk] = np.interp(np.arange(a, min(n, a + chunk)), centres, vals)
    return out


def blank(fe, k_sigma: float = 5.0, guard_ms: float = 2.0, win_s: float = 1.0,
          hop_s: float = 0.05, *, fs: float = FE_FS, taper_ms: float = 1.0
          ) -> tuple[np.ndarray, np.ndarray]:
    """(fe * keep, keep): impulses blanked sample by sample (design 6.2).

    keep in [0, 1] is 0 where |fe|^2 > k_sigma^2 * nu (nu the running
    median power) and within guard_ms of such a sample, rises over a
    raised-cosine taper of taper_ms beyond the guard, and is 1 elsewhere.
    """
    fe = np.asarray(fe, dtype=np.complex64)
    n = len(fe)
    p = fe.real.astype(np.float32) ** 2 + fe.imag.astype(np.float32) ** 2
    keep = np.ones(n, dtype=np.float32)
    if n == 0:
        return fe.copy(), keep
    centres, med = running_median(p, fs, win_s, hop_s)
    nu = _interp_to_samples(n, centres, med)
    bad = p > np.float32(k_sigma ** 2) * nu
    if not bad.any():
        return fe.copy(), keep
    g = int(round(guard_ms * 1e-3 * fs))
    dead = ndimage.maximum_filter1d(bad.view(np.uint8), 2 * g + 1).astype(bool)
    L = int(round(taper_ms * 1e-3 * fs))
    for k in range(L, 0, -1):               # nearer samples overwrite farther ones
        near = ndimage.maximum_filter1d(dead.view(np.uint8), 2 * k + 1).astype(bool)
        keep[near] = 0.5 * (1 - math.cos(math.pi * k / (L + 1)))
    keep[dead] = 0.0
    return (fe * keep).astype(np.complex64), keep


def normalise(fe, keep=None, win_ms: float = 100.0, *, fs: float = FE_FS,
              hop_ms: float = 10.0, min_kept: float = 0.25,
              chunk: int = 512) -> tuple[np.ndarray, np.ndarray]:
    """(fe / sqrt(level), level): the inverse AGC (design 6.2).

    level is the median |fe|^2 over the fully kept samples (keep == 1) of
    a win_ms window sliding at hop_ms, interpolated between window
    centres. Sliding (rather than abutting) windows put a receive AGC's
    gain step within a hop of where it happened: a centred median
    switches level as soon as more of its window lies past the step. A
    window with less than `min_kept` of its samples kept borrows from
    its neighbours. A noise-only stream comes out with median power 1,
    i.e. E|n|^2 = 1/ln 2.
    """
    fe = np.asarray(fe, dtype=np.complex64)
    n = len(fe)
    if n == 0:
        return fe.copy(), np.ones(0, dtype=np.float32)
    w = min(n, max(1, int(round(win_ms * 1e-3 * fs))))
    hop = max(1, int(round(hop_ms * 1e-3 * fs)))
    p = fe.real.astype(np.float32) ** 2 + fe.imag.astype(np.float32) ** 2
    if keep is not None:
        p[np.asarray(keep) < 1.0] = np.nan
    centres = np.arange(hop // 2, n, hop)
    starts = np.clip(centres - w // 2, 0, n - w)
    view = np.lib.stride_tricks.sliding_window_view(p, w)
    med = np.empty(len(starts), dtype=np.float32)
    cnt = np.empty(len(starts), dtype=np.int64)
    for a in range(0, len(starts), chunk):
        blk = view[starts[a:a + chunk]]
        bad = np.isnan(blk)
        cnt[a:a + chunk] = w - bad.sum(axis=1)
        # median of the kept samples: unkept ones sort to the top as +inf,
        # then pick the middle of the kept count per row
        srt = np.sort(np.where(bad, np.inf, blk), axis=1)
        c = np.maximum(cnt[a:a + chunk], 1)
        r = np.arange(len(c))
        med[a:a + chunk] = 0.5 * (srt[r, (c - 1) // 2] + srt[r, c // 2])
    ok = cnt >= max(1, int(min_kept * w))
    if not ok.any():
        lev = np.ones(n, dtype=np.float32)
    else:
        lev = _interp_to_samples(n, centres[ok], np.maximum(med[ok], np.float32(1e-30)))
    return (fe / np.sqrt(lev)).astype(np.complex64), lev


# --- channeliser ---------------------------------------------------------------------------

def quantise_mix(f_mix_hz) -> Fraction:
    """f_mix on the channeliser's 1/64 Hz grid."""
    return Fraction(int(round(float(f_mix_hz) * F_MIX_DEN)), F_MIX_DEN)


def mix_phasor(n0: int, n: int, f_mix: Fraction, fs_in: int) -> np.ndarray:
    """exp(-j 2 pi f_mix (n0 + i)/fs_in), i < n, from exact integer turns.

    f_mix must lie on the 1/64 Hz grid: turns = ((n0+i) f_num mod D)/D
    with D = 64 fs_in, so the phase never depends on how far into a
    stream the sample is.
    """
    f_mix = Fraction(f_mix)
    num = int(f_mix * F_MIX_DEN)
    if Fraction(num, F_MIX_DEN) != f_mix:
        raise ValueError(f"f_mix {f_mix} is not on the 1/{F_MIX_DEN} Hz grid")
    D = F_MIX_DEN * int(fs_in)
    k = num % D
    # The phase repeats every P = D / gcd(k, D) samples, so one period is
    # computed and repeated: the same arithmetic per value as computing
    # every sample (so the same result, bit for bit), at a fraction of the
    # cost over a whole slot.
    P = D // math.gcd(k, D)
    idx = np.arange(P, dtype=np.int64)
    turns = ((idx * k) % D).astype(np.float64) / D
    ang = 2 * np.pi * turns
    tab = np.empty(P, dtype=np.complex128)
    tab.real = np.cos(ang)
    tab.imag = -np.sin(ang)
    return np.tile(np.roll(tab, -(int(n0) % P)), -(-n // P))[:n]


def channelise(fe, fs_in: int, f_mix_hz, fs_out: int = CH_FS, *, n0: int = 0
               ) -> tuple[np.ndarray, Fraction]:
    """(ch complex64, f_mix): fe mixed down by f_mix and decimated to fs_out.

    f_mix_hz is a frequency *inside* fe (for FE: audio Hz - 1500), and is
    quantised to 1/64 Hz; the quantised value is returned. Sample i of
    fe is stream sample n0 + i for the mixing phase. Decimation is
    `resample_poly(., 1, fs_in/fs_out, window=("kaiser", 10))`, which is
    zero-phase: ch[k] is aligned with fe[k*decim].
    """
    fs_in, fs_out = int(fs_in), int(fs_out)
    if fs_in % fs_out:
        raise ValueError(f"fs_out {fs_out} must divide fs_in {fs_in}")
    f_mix = quantise_mix(f_mix_hz)
    fe = np.asarray(fe)
    y = fe.astype(np.complex128) * mix_phasor(n0, len(fe), f_mix, fs_in)
    decim = fs_in // fs_out
    if decim > 1:
        y = signal.resample_poly(y, 1, decim, window=("kaiser", 10.0))
    return y.astype(np.complex64), f_mix


def channelise_keep(keep, fs_in: int = FE_FS, fs_out: int = CH_FS) -> np.ndarray:
    """float32: the blanker's keep indicator through the channeliser's
    decimation (no mixing), aligned with `channelise`'s output. The
    blanked share at position p is then psi_p = MF{keep_ch}(t_p)/G0
    (design 6.2), and a position with psi_p < 0.5 is erased."""
    fs_in, fs_out = int(fs_in), int(fs_out)
    k = np.asarray(keep, dtype=np.float64)
    decim = fs_in // fs_out
    if decim > 1:
        k = signal.resample_poly(k, 1, decim, window=("kaiser", 10.0))
    return k.astype(np.float32)


@functools.lru_cache(maxsize=8)
def _noise_response(fs_in: int, fs_out: int, nfft: int) -> np.ndarray:
    """Aliased power response sum_k |H(f + k fs_out)|^2 of the channeliser's
    decimation filter, on the fftfreq(nfft, 1/fs_out) grid: what white FE
    noise looks like in CH."""
    decim = fs_in // fs_out
    if decim == 1:
        return np.ones(nfft)
    # the filter resample_poly(., 1, decim, window=("kaiser", 10)) designs
    h = signal.firwin(2 * 10 * decim + 1, 1.0 / decim, window=("kaiser", 10.0))
    f = np.fft.fftfreq(nfft, 1.0 / fs_out)
    r = np.zeros(nfft)
    for k in range(-(decim // 2), decim - decim // 2):   # each alias once (freqz is fs_in-periodic)
        _, H = signal.freqz(h, worN=(f + k * fs_out), fs=fs_in)
        r += np.abs(H) ** 2
    return r


@functools.lru_cache(maxsize=8)
def _fold_power(fs_in: int, fs_out: int, nfft: int, band_hz: tuple) -> tuple:
    """(idx, fk, P) for `_fe_flat`: the band's bins idx on the
    fftfreq(nfft, 1/fs_out) grid, the FE frequencies fk[k] each bin's
    k'th fold comes from (f + k fs_out), and the decimation filter's
    power |H(fk)|^2 there, `_noise_response`'s terms."""
    decim = fs_in // fs_out
    f = np.fft.fftfreq(nfft, 1.0 / fs_out)
    idx = np.flatnonzero((np.abs(f) >= band_hz[0]) & (np.abs(f) <= band_hz[1]))
    if decim == 1:
        return idx, f[idx][None, :], np.ones((1, len(idx)))
    h = signal.firwin(2 * 10 * decim + 1, 1.0 / decim, window=("kaiser", 10.0))
    fk = np.array([f[idx] + k * fs_out for k in range(-(decim // 2), decim - decim // 2)])
    P = np.array([np.abs(signal.freqz(h, worN=row, fs=fs_in)[1]) ** 2 for row in fk])
    return idx, fk, P


@functools.lru_cache(maxsize=64)
def _fe_power(fs_in: int, fs_out: int, nfft: int, audio_hz: float, band_hz: tuple) -> tuple:
    """(idx, num, den) for `_fe_flat` and `_fe_response`: the band's bins
    idx, the noise power white FE-input noise leaves in each through the
    front end's low-pass and the decimation (num), and through the
    decimation alone (den, `_noise_response`'s value there)."""
    idx, fk, P = _fold_power(fs_in, fs_out, nfft, tuple(band_hz))
    _, G = signal.freqz(_FE_TAPS, worN=((audio_hz + fk - FE_CENTER_HZ) % FS).ravel(), fs=FS)
    return idx, np.sum(P * np.abs(G.reshape(fk.shape)) ** 2, axis=0), np.sum(P, axis=0)


@functools.lru_cache(maxsize=64)
def _fe_flat(fs_in: int, fs_out: int, nfft: int, audio_hz: float,
             band_hz: tuple = (85.0, 120.0)) -> np.ndarray:
    """bool[nfft] on the fftfreq(nfft, 1/fs_out) grid of a channel whose
    0 Hz is audio_hz: the bins of band_hz (either side) whose noise the
    front end's low-pass (`audio_to_fe`) leaves within FE_FLAT_DB of
    white, counting what the decimation folds in from k fs_out away
    (`_noise_response`'s sum), so a bin near +-fs_out/2 whose fold lies
    on the skirt is not flat either. False outside band_hz."""
    idx, num, den = _fe_power(fs_in, fs_out, nfft, audio_hz, tuple(band_hz))
    flat = np.zeros(nfft, dtype=bool)
    flat[idx] = num >= 10.0 ** (-FE_FLAT_DB / 10.0) * den
    flat.flags.writeable = False             # (cached)
    return flat


@functools.lru_cache(maxsize=64)
def _fe_response(fs_in: int, fs_out: int, nfft: int, audio_hz: float,
                 band_hz: tuple = (85.0, 120.0)) -> np.ndarray:
    """float[nfft], `_fe_flat`'s grid: the fraction of white noise's power
    the front end's low-pass leaves in each bin of band_hz, folds
    counted; 1 outside band_hz."""
    idx, num, den = _fe_power(fs_in, fs_out, nfft, audio_hz, tuple(band_hz))
    r = np.ones(nfft)
    r[idx] = num / den
    r.flags.writeable = False                # (cached)
    return r


def noise_psd(ch, fs: int = CH_FS, exclude_hz=(), win_s: float = 10.0, *,
              band_hz=(85.0, 120.0), guard_hz: float = 45.0,
              fs_in: int = FE_FS, audio_hz: float | None = None) -> np.ndarray:
    """float32[n_win]: noise PSD (CH power per Hz) in win_s windows.

    For each window, a Hann periodogram of the whole window; bins between
    band_hz[0] and band_hz[1] from the carrier (0 Hz in ch), either side,
    leaving out bins within guard_hz of any frequency in `exclude_hz`
    (detected neighbours, Hz in ch) or of its alias one fs away; the
    45 Hz default covers a CE neighbour's 99.9% width (+-36 Hz) plus the
    decimation filter's skirt, through which its sideband folds in. When
    neighbours leave no bin either side (signals about 100 Hz apart on
    both sides), each window takes the quieter side, neighbours and all:
    a neighbour only adds power. Each bin is divided by the
    channeliser's aliased noise response there (the band sits on the
    decimation filter's skirt), and the median over bins is divided by
    ln 2, the median of an exponential. White FE noise of variance s2 at
    fs_in gives s2/fs_in, which is the PSD of the flat passband.

    `audio_hz`, the audio frequency of ch's 0 Hz, leaves out the bins the
    front end's own low-pass has taken down by more than FE_FLAT_DB
    (`_fe_flat`): near 300 and 2700 Hz half the band sits on that skirt,
    so the floor read low and the SNR high (9.1 dB at 305 Hz, 7.0 at 326,
    2.5 at 360 Hz, measured on white noise; within about 0.5 dB now).
    When neighbours leave fewer than FE_MIN_HZ of flat bins clear, no
    bin is both flat and clear, and each kind errs one way: the flat
    bins, neighbour and all, can only read high, and the skirt's clear
    bins (down to FE_SKIRT_DB), each divided by the low-pass's response
    there (`_fe_response`), only low, as a radio's filter narrower than
    this front end takes the skirt down further. The floor is the
    geometric mean of their medians, its error at most half the spread
    in dB (2026-10-12 review: from the flat bins alone a weak pass beside
    a stronger neighbour was lost, and from the skirt alone, beside an
    equal one and through a radio as narrow as this front end, it read
    some 16 dB strong). With no such skirt bin, the flat sides are the
    fallback above. A channel whose bins are all flat (audio_hz from
    about 430 to 2570 Hz) is not changed at all, nor one with no flat
    bin. `rules.EDGE_NOISE` set back ignores audio_hz.

    Window i covers samples [i*w, i*w + w), the last shifted inward to
    stay whole; there are ceil(len(ch)/w) windows.
    """
    ch = np.asarray(ch, dtype=np.complex64)
    w = max(16, int(round(win_s * fs)))
    n = len(ch)
    if n < w:
        w = n
    nwin = max(1, -(-n // w))
    f = np.fft.fftfreq(w, 1.0 / fs)
    band = (np.abs(f) >= band_hz[0]) & (np.abs(f) <= band_hz[1])
    clear = np.ones(w, dtype=bool)
    for fx in exclude_hz:
        for alias in (-fs, 0, fs):             # its skirt folds back across +-fs/2
            clear &= np.abs(f - float(fx) - alias) > guard_hz
    skirt = None                               # the fallback's: (bins, the low-pass's response)
    if audio_hz is not None and not rules.old(rules.EDGE_NOISE) and fs_in % fs == 0:
        key = (int(fs_in), int(fs), w, float(audio_hz), (float(band_hz[0]), float(band_hz[1])))
        flat = band & _fe_flat(*key)
        if flat.any() and not np.array_equal(flat, band):
            if np.count_nonzero(flat & clear) < FE_MIN_HZ * w / fs:
                r = _fe_response(*key)
                sk = band & ~flat & clear & (r >= 10.0 ** (-FE_SKIRT_DB / 10.0))
                if sk.any():
                    skirt = (sk, r[sk])
            band = flat
    sel = band & clear
    sides = [sel] if sel.any() and skirt is None else [band & (f < 0), band & (f > 0)]
    if not any(s.any() for s in sides):
        raise ValueError("noise_psd: no bin in the band")
    resp = (_noise_response(int(fs_in), int(fs), w) if fs_in % fs == 0
            else np.ones(w))
    win = signal.windows.hann(w, sym=False)
    scale = 1.0 / (fs * np.sum(win ** 2))
    out = np.empty(nwin, dtype=np.float32)
    for i in range(nwin):
        a = min(i * w, n - w)
        X = np.fft.fft(ch[a:a + w] * win)
        p = (X.real ** 2 + X.imag ** 2) * scale / resp
        if skirt is None:
            out[i] = min(np.median(p[s]) for s in sides if s.any()) / math.log(2)
        else:
            # the flat bins, neighbours and all, can only read high; the
            # skirt, compensated, only low (a radio's filter): between them
            hi = np.median(p[band])
            lo = np.median(p[skirt[0]] / skirt[1])
            out[i] = math.sqrt(hi * lo) / math.log(2)
    return out


# --- the 48 h passband store ---------------------------------------------------------------

_HOUR_RE = re.compile(r"^fe_([0-9]+)\.(?:i16|flac)\Z")
PB_PACK_AFTER_S = 120.0          # a finished hour is packed (FLAC) this long after it ends
PB_LEAD_S = 15.0                 # a slot read back starts this long before t0 (lead-in, search margin)
PB_TAIL_S = 5.0                  # and ends this long after the frame


def slot_t0(q: int) -> int:
    """Unix time of slot q's nominal t0: the quarter hour plus 1 s."""
    return 900 * int(q) + 1


def _is_temporary(name: str) -> bool:
    """A temporary the passband store writes and renames: fe_<hour>.flac /
    .i16 / .json or meta.json, then .<pid>[.<thread id>].tmp."""
    return re.match(r"(?:fe_[0-9]+\.(?:i16|flac|json)|meta\.json)\.[0-9]+(?:\.[0-9]+)?\.tmp\Z", name) is not None


def _soundfile():
    """The soundfile module (FLAC), or None when it is not installed."""
    try:
        import soundfile
    except (ImportError, OSError):     # (OSError: the package without its libsndfile)
        return None
    return soundfile


class PassbandStore:
    """The whole FE stream on disk for `keep_hours` (spec 7, design 6.9).

    One file per UTC hour, `fe_<hour>.i16` with hour = floor(t/3600):
    a fixed-size raw array of interleaved int16 I/Q, sample k of the hour
    at unix time 3600*hour + k/fs (memory-mapped, so unwritten stretches
    cost no disk on a sparse-file system). Beside each, `fe_<hour>.json`
    lists the written sample ranges, so a gap reads back as zeros with a
    mask that says so. `meta.json` at the root pins fs and the int16
    scale, so a restart re-opens the files exactly as written.

    With `pack` (the live listener's store) and the soundfile package
    installed, an hour is packed into `fe_<hour>.flac` (I and Q as two
    16-bit channels at fs, lossless) PB_PACK_AFTER_S after it ends, as
    the stream's own time goes, on a thread of its own, and checked
    sample for sample before the raw file goes: 47-68% of the raw size on
    band noise, against 72-89% for zstd, because FLAC's linear prediction
    sees that the FE only fills part of its band. Reads take either kind
    of file, and a write into a packed hour unpacks it first; without
    soundfile hours stay raw (and a packed one cannot be read).

    Times are unix seconds (float); sample n of the stream is at n/fs,
    and a write or read at time t starts at sample round(t*fs). A raw
    hour shorter than an hour (a crash between its creation and its
    sizing) reads as unwritten past its end, and the next write into it
    grows it to size. fs must be positive (ValueError).
    """

    def __init__(self, root, fs: int = FE_FS, scale: float = 8192.0,
                 keep_hours: float = 48.0, pack: bool = False):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        meta = self.root / "meta.json"
        fresh = not meta.exists()
        if not fresh:
            m = json.loads(meta.read_text())
            fs, scale = int(m["fs"]), float(m["scale"])
        if int(fs) <= 0:            # (an hour of no samples: read() would never advance)
            raise ValueError(f"the passband store's fs must be positive, not {int(fs)}")
        if fresh:
            _atomic_write(meta, json.dumps({"fs": int(fs), "scale": float(scale)}))
        self.fs = int(fs)
        self.scale = float(scale)               # int16 counts per unit amplitude
        self.keep_hours = float(keep_hours)
        self.per_hour = 3600 * self.fs
        self.pack = bool(pack) and _soundfile() is not None
        self._lock = threading.RLock()          # a write and a pack's swap of one hour
        self._pack_lock = threading.Lock()      # one pack_hour at a time
        self._gen: dict[int, int] = {}          # writes per hour, this process (under _lock)
        self._packer: threading.Thread | None = None
        self._pack_check = 0.0

    # --- files ---
    def _paths(self, hour: int) -> tuple[Path, Path]:
        return self.root / f"fe_{hour}.i16", self.root / f"fe_{hour}.json"

    def _flac(self, hour: int) -> Path:
        return self.root / f"fe_{hour}.flac"

    def _map(self, hour: int, write: bool):
        data, _ = self._paths(hour)
        if write and not data.exists() and self._flac(hour).exists():
            self._unpack(hour)
        if not data.exists():
            if not write:
                return None
            return np.memmap(data, dtype=np.int16, mode="w+", shape=(self.per_hour, 2))
        return np.memmap(data, dtype=np.int16, mode="r+" if write else "r",
                         shape=(self.per_hour, 2))

    def _read_hour(self, hour: int, lo: int, hi: int) -> np.ndarray | None:
        """int16 (n, 2) of hour's samples [lo, lo + n), raw or packed; None
        without a file. n is hi - lo, or less where the file ends first (a
        raw hour left short, which a write grows to size): never written."""
        for _ in range(2):           # (a pack may swap the files between the two looks)
            try:
                data, _ = self._paths(hour)
                if data.exists():
                    y = np.fromfile(data, dtype=np.int16, count=2 * (hi - lo), offset=4 * lo)
                    return y[:len(y) // 2 * 2].reshape(-1, 2)
                path = self._flac(hour)
                if not path.exists():
                    return None
                sf = _soundfile()
                if sf is None:
                    raise RuntimeError(f"{path} is FLAC: reading it needs the soundfile package")
                y, _ = sf.read(str(path), start=lo, stop=hi, dtype="int16", always_2d=True)
                return y
            except FileNotFoundError:
                continue
        return None

    def raw_hours(self) -> list[int]:
        """Hours still in raw files, ascending."""
        return sorted(h for h in self.hours() if self._paths(h)[0].exists())

    def pack_hour(self, hour: int) -> bool:
        """Pack raw hour `hour` into FLAC, checked sample for sample before
        the raw file goes; False (and the raw file kept) if it could not be.

        One at a time (a lock), into a temporary named for this process and
        thread; the FLAC is synced to the device before it replaces the raw
        hour. A write meanwhile -- seen by the raw file's mtime, or, where
        the file system's clock is too coarse for that (FAT's 2 s), by this
        store's count of its own writes -- leaves the hour raw."""
        sf = _soundfile()
        data, _ = self._paths(hour)
        if sf is None or not data.exists():
            return False
        with self._pack_lock:
            tmp = self.root / f"fe_{hour}.flac.{os.getpid()}.{threading.get_ident()}.tmp"
            try:
                before = data.stat().st_mtime_ns
                with self._lock:
                    gen = self._gen.get(hour, 0)
                mm = np.memmap(data, dtype=np.int16, mode="r", shape=(self.per_hour, 2))
                sf.write(str(tmp), mm, self.fs, format="FLAC", subtype="PCM_16")
                with open(tmp, "r+b") as f:
                    os.fsync(f.fileno())
                y, _ = sf.read(str(tmp), dtype="int16", always_2d=True)
                same = y.shape == mm.shape and np.array_equal(y, mm)
                del mm, y
                with self._lock:
                    if (not same or data.stat().st_mtime_ns != before
                            or self._gen.get(hour, 0) != gen):
                        return False           # (written to meanwhile: packed again later)
                    replace(tmp, self._flac(hour))
                    data.unlink()
                return True
            except (OSError, RuntimeError, ValueError):
                return False
            finally:
                tmp.unlink(missing_ok=True)

    def _unpack(self, hour: int) -> None:
        """Turn packed hour `hour` back into a raw file (for a write into it)."""
        sf = _soundfile()
        path = self._flac(hour)
        if sf is None:
            raise RuntimeError(f"{path} is FLAC: writing into it needs the soundfile package")
        y, _ = sf.read(str(path), dtype="int16", always_2d=True)
        data, _ = self._paths(hour)
        tmp = self.root / f"fe_{hour}.i16.{os.getpid()}.{threading.get_ident()}.tmp"
        mm = np.memmap(tmp, dtype=np.int16, mode="w+", shape=(self.per_hour, 2))
        mm[:len(y)] = y[:self.per_hour]
        mm.flush()
        del mm
        replace(tmp, data)
        path.unlink()

    def _maybe_pack(self, now: float) -> None:
        """Start packing, on a thread, the raw hours that ended PB_PACK_AFTER_S ago."""
        if not self.pack or now < self._pack_check or (
                self._packer is not None and self._packer.is_alive()):
            return
        self._pack_check = now + 60.0
        due = [h for h in self.raw_hours() if (h + 1) * 3600 + PB_PACK_AFTER_S <= now]
        if due:
            self._packer = threading.Thread(target=lambda: [self.pack_hour(h) for h in due],
                                            name="passband-pack", daemon=True)
            self._packer.start()

    def _ranges(self, hour: int) -> list[list[int]]:
        _, idx = self._paths(hour)
        return json.loads(idx.read_text()) if idx.exists() else []

    def hours(self) -> list[int]:
        """Hours (floor(t/3600)) with a file on disk, ascending."""
        out = set()
        for p in self.root.iterdir():
            m = _HOUR_RE.match(p.name)
            if m:
                out.add(int(m.group(1)))
        return sorted(out)

    # --- write / read ---
    def write(self, t0: float, x, expire: bool = True) -> None:
        """Store complex samples x, the first at unix time t0.

        Values are clipped to the int16 range at `scale`. With `expire`,
        files older than keep_hours before the end of x are deleted.
        """
        self._write_at(int(round(float(t0) * self.fs)), x, expire)

    def _write_at(self, n0: int, x, expire: bool) -> None:
        x = np.asarray(x, dtype=np.complex64)
        i = 0
        while i < len(x):
            n = n0 + i
            hour, k = divmod(n, self.per_hour)
            m = min(len(x) - i, self.per_hour - k)
            seg = x[i:i + m]
            q = np.empty((m, 2), dtype=np.int16)
            q[:, 0] = np.clip(np.rint(seg.real * self.scale), -32767, 32767)
            q[:, 1] = np.clip(np.rint(seg.imag * self.scale), -32767, 32767)
            with self._lock:
                mm = self._map(hour, write=True)
                mm[k:k + m] = q
                mm.flush()
                del mm
                self._gen[hour] = self._gen.get(hour, 0) + 1
                _, idx = self._paths(hour)
                _atomic_write(idx, json.dumps(_merge_ranges(self._ranges(hour) + [[k, k + m]])))
            i += m
        if expire and len(x):
            self.expire(now=(n0 + len(x)) / self.fs)
            self._maybe_pack((n0 + len(x)) / self.fs)

    def read(self, t_start: float, t_end: float, return_mask: bool = False):
        """complex64 samples [round(t_start fs), round(t_end fs)); zeros in gaps.

        With return_mask, also a bool array, True where samples were written.
        """
        a = int(round(float(t_start) * self.fs))
        b = int(round(float(t_end) * self.fs))
        n = max(0, b - a)
        out = np.zeros(n, dtype=np.complex64)
        mask = np.zeros(n, dtype=bool)
        i = 0
        while i < n:
            hour, k = divmod(a + i, self.per_hour)
            m = min(n - i, self.per_hour - k)
            for lo, hi in self._ranges(hour):
                lo, hi = max(lo, k), min(hi, k + m)
                if lo < hi:
                    y = self._read_hour(hour, lo, hi)
                    if y is None:
                        continue
                    q = np.asarray(y, dtype=np.float32) / self.scale
                    out[i + lo - k:i + lo - k + len(q)] = q[:, 0] + 1j * q[:, 1]
                    mask[i + lo - k:i + lo - k + len(q)] = True
            i += m
        return (out, mask) if return_mask else out

    # --- slots ---
    def write_capture(self, cap: Capture, expire: bool = True) -> tuple[float, float]:
        """Store a slot capture's raw FE stream (before the blanker); returns
        (unix time of its first sample, gain it was stored with).

        The capture's first sample is at t0 - t0_index/fs, t0 = 900 q + 1
        (a fractional t0_index is rounded to the nearest sample, under
        125 us at 4 kHz, which the timing search absorbs). A capture whose
        peak would use more than half the int16 range is stored attenuated
        by a power of 2 rather than clipped: the receiver undoes any
        constant gain (inverse AGC), and audio from a WAV never needs it.
        The peak is the finite samples'; a non-finite one is stored as write
        stores it (NaN as 0, an infinity clipped).
        """
        if int(cap.fs) != self.fs:
            raise ValueError(f"the store holds {self.fs} Hz FE, not a {cap.fs} Hz capture")
        x = np.asarray(cap.fe, dtype=np.complex64)
        peak = float(max(np.max(np.abs(x.real), initial=0.0, where=np.isfinite(x.real)),
                         np.max(np.abs(x.imag), initial=0.0, where=np.isfinite(x.imag)))) * self.scale
        gain = 1.0
        if peak > 16383.0:
            gain = 2.0 ** -math.ceil(math.log2(peak / 16383.0))
            x = x * np.float32(gain)
        n0 = int(slot_t0(cap.q)) * self.fs - int(round(float(cap.t0_index)))
        self._write_at(n0, x, expire)
        return n0 / self.fs, gain

    def capture(self, q: int, dur_s: float, lead_s: float = PB_LEAD_S,
                tail_s: float = PB_TAIL_S) -> tuple[Capture | None, float]:
        """(Capture, coverage) of slot q read back: t0 - lead_s to t0 + dur_s + tail_s.

        `coverage` is the fraction of [t0, t0 + dur_s) that was written;
        unwritten samples read as zeros. The Capture is None when nothing
        in the span was written.
        """
        t0 = slot_t0(q)
        a = int(t0) * self.fs - int(round(lead_s * self.fs))
        b = int(t0) * self.fs + int(round((dur_s + tail_s) * self.fs))
        x, mask = self.read(a / self.fs, b / self.fs, return_mask=True)
        if not mask.any():
            return None, 0.0
        k0 = int(t0) * self.fs - a
        k1 = min(len(mask), k0 + int(round(dur_s * self.fs)))
        cov = float(np.mean(mask[k0:k1])) if k1 > k0 else 0.0
        # unwritten stretches at either end are dropped (inside, gaps stay zeros)
        on = np.flatnonzero(mask)
        i, j = int(on[0]), int(on[-1]) + 1
        return Capture(fe=x[i:j], q=int(q), t0_index=float(k0 - i), fs=self.fs), cov

    def slots(self, dur_s: float) -> list[int]:
        """Every q whose frame [t0, t0 + dur_s) overlaps a stored hour, ascending."""
        out = set()
        for hour in self.hours():
            lo = hour * 3600 - dur_s - 1.0
            hi = (hour + 1) * 3600
            for q in range(int(math.floor(lo / 900.0)), int(math.ceil(hi / 900.0)) + 1):
                if slot_t0(q) + dur_s > hour * 3600 and slot_t0(q) < hi:
                    out.add(q)
        return sorted(out)

    def expire(self, now: float | None = None) -> list[int]:
        """Delete hours that ended more than keep_hours before `now`; return
        them. Also deletes the store's temporaries whose mtime is more than
        an hour old by the wall clock (not `now`, a stream's time): left by
        a process killed mid-pack or mid-write, since none takes that long.
        That sweep is housekeeping: a temporary it cannot stat or remove (a
        directory of that name, a file another process holds open on
        Windows) is left for the next expire, never raised out of the write
        that called it."""
        now = time.time() if now is None else float(now)
        gone = []
        for hour in self.hours():
            if (hour + 1) * 3600 <= now - self.keep_hours * 3600:
                for p in (*self._paths(hour), self._flac(hour)):
                    p.unlink(missing_ok=True)
                with self._lock:
                    self._gen.pop(hour, None)
                gone.append(hour)
        old = time.time() - 3600.0
        for p in self.root.iterdir():
            if _is_temporary(p.name):
                try:
                    if p.stat().st_mtime < old:
                        p.unlink()
                except OSError:
                    pass
        return gone


def _merge_ranges(rs) -> list[list[int]]:
    out: list[list[int]] = []
    for lo, hi in sorted(rs):
        if out and lo <= out[-1][1]:
            out[-1][1] = max(out[-1][1], int(hi))
        else:
            out.append([int(lo), int(hi)])
    return out


def replace(src, dst) -> None:
    """os.replace(src, dst), retried for about a second on Windows while it
    raises PermissionError: there a target held open for a moment (a reader
    of state.json, another thread's replace of the same file, a virus
    scanner) refuses a replace that succeeds a few milliseconds later. As
    native/core/util/fileio.hpp's replace_file does."""
    if os.name != "nt":
        os.replace(src, dst)
        return
    for k in range(21):                     # 20 waits of 50 ms
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if k == 20:
                raise
            time.sleep(0.05)


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_text(text)
    replace(tmp, path)

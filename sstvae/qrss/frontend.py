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
import time
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy import ndimage, signal

from sstvae import config as _c
from sstvae.modem import dsp

from .constants import CH_FS, FE_CENTER_HZ, FE_FS, FS

_FE_DECIM = FS // FE_FS                       # 2
_FE_TAPS = signal.firwin(255, 1250.0, fs=FS, window=("kaiser", 8.0))
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
    idx = (np.arange(n, dtype=np.int64) + int(n0)) % D
    turns = ((idx * (num % D)) % D).astype(np.float64) / D
    ang = 2 * np.pi * turns
    out = np.empty(n, dtype=np.complex128)
    out.real = np.cos(ang)
    out.imag = -np.sin(ang)
    return out


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


def noise_psd(ch, fs: int = CH_FS, exclude_hz=(), win_s: float = 10.0, *,
              band_hz=(85.0, 120.0), guard_hz: float = 45.0,
              fs_in: int = FE_FS) -> np.ndarray:
    """float32[n_win]: noise PSD (CH power per Hz) in win_s windows.

    For each window, a Hann periodogram of the whole window; bins between
    band_hz[0] and band_hz[1] from the carrier (0 Hz in ch), either side,
    leaving out bins within guard_hz of any frequency in `exclude_hz`
    (detected neighbours, Hz in ch) or of its alias one fs away; the
    45 Hz default covers a CE neighbour's 99.9% width (+-36 Hz) plus the
    decimation filter's skirt, through which its sideband folds in. Each bin is divided by the
    channeliser's aliased noise response there (the band sits on the
    decimation filter's skirt), and the median over bins is divided by
    ln 2, the median of an exponential. White FE noise of variance s2 at
    fs_in gives s2/fs_in, which is the PSD of the flat passband.

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
    sel = (np.abs(f) >= band_hz[0]) & (np.abs(f) <= band_hz[1])
    for fx in exclude_hz:
        for alias in (-fs, 0, fs):             # its skirt folds back across +-fs/2
            sel &= np.abs(f - float(fx) - alias) > guard_hz
    if not sel.any():
        raise ValueError("noise_psd: every bin in the band is excluded")
    resp = _noise_response(int(fs_in), int(fs), w)[sel] if fs_in % fs == 0 else 1.0
    win = signal.windows.hann(w, sym=False)
    scale = 1.0 / (fs * np.sum(win ** 2))
    out = np.empty(nwin, dtype=np.float32)
    for i in range(nwin):
        a = min(i * w, n - w)
        X = np.fft.fft(ch[a:a + w] * win)
        pw = (X.real[sel] ** 2 + X.imag[sel] ** 2) * scale / resp
        out[i] = np.median(pw) / math.log(2)
    return out


# --- the 48 h passband store ---------------------------------------------------------------

_HOUR_RE = re.compile(r"^fe_(\d+)\.i16$")


class PassbandStore:
    """The whole FE stream on disk for `keep_hours` (spec 7, design 6.9).

    One file per UTC hour, `fe_<hour>.i16` with hour = floor(t/3600):
    a fixed-size raw array of interleaved int16 I/Q, sample k of the hour
    at unix time 3600*hour + k/fs (memory-mapped, so unwritten stretches
    cost no disk on a sparse-file system). Beside each, `fe_<hour>.json`
    lists the written sample ranges, so a gap reads back as zeros with a
    mask that says so. `meta.json` at the root pins fs and the int16
    scale, so a restart re-opens the files exactly as written.

    Times are unix seconds (float); sample n of the stream is at n/fs,
    and a write or read at time t starts at sample round(t*fs).
    """

    def __init__(self, root, fs: int = FE_FS, scale: float = 8192.0,
                 keep_hours: float = 48.0):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        meta = self.root / "meta.json"
        if meta.exists():
            m = json.loads(meta.read_text())
            fs, scale = int(m["fs"]), float(m["scale"])
        else:
            _atomic_write(meta, json.dumps({"fs": int(fs), "scale": float(scale)}))
        self.fs = int(fs)
        self.scale = float(scale)               # int16 counts per unit amplitude
        self.keep_hours = float(keep_hours)
        self.per_hour = 3600 * self.fs

    # --- files ---
    def _paths(self, hour: int) -> tuple[Path, Path]:
        return self.root / f"fe_{hour}.i16", self.root / f"fe_{hour}.json"

    def _map(self, hour: int, write: bool):
        data, _ = self._paths(hour)
        if not data.exists():
            if not write:
                return None
            return np.memmap(data, dtype=np.int16, mode="w+", shape=(self.per_hour, 2))
        return np.memmap(data, dtype=np.int16, mode="r+" if write else "r",
                         shape=(self.per_hour, 2))

    def _ranges(self, hour: int) -> list[list[int]]:
        _, idx = self._paths(hour)
        return json.loads(idx.read_text()) if idx.exists() else []

    def hours(self) -> list[int]:
        """Hours (floor(t/3600)) with a file on disk, ascending."""
        out = []
        for p in self.root.iterdir():
            m = _HOUR_RE.match(p.name)
            if m:
                out.append(int(m.group(1)))
        return sorted(out)

    # --- write / read ---
    def write(self, t0: float, x, expire: bool = True) -> None:
        """Store complex samples x, the first at unix time t0.

        Values are clipped to the int16 range at `scale`. With `expire`,
        files older than keep_hours before the end of x are deleted.
        """
        x = np.asarray(x, dtype=np.complex64)
        n0 = int(round(float(t0) * self.fs))
        i = 0
        while i < len(x):
            n = n0 + i
            hour, k = divmod(n, self.per_hour)
            m = min(len(x) - i, self.per_hour - k)
            mm = self._map(hour, write=True)
            seg = x[i:i + m]
            q = np.empty((m, 2), dtype=np.int16)
            q[:, 0] = np.clip(np.rint(seg.real * self.scale), -32767, 32767)
            q[:, 1] = np.clip(np.rint(seg.imag * self.scale), -32767, 32767)
            mm[k:k + m] = q
            mm.flush()
            del mm
            _, idx = self._paths(hour)
            _atomic_write(idx, json.dumps(_merge_ranges(self._ranges(hour) + [[k, k + m]])))
            i += m
        if expire and len(x):
            self.expire(now=(n0 + len(x)) / self.fs)

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
            mm = self._map(hour, write=False)
            if mm is not None:
                for lo, hi in self._ranges(hour):
                    lo, hi = max(lo, k), min(hi, k + m)
                    if lo < hi:
                        q = np.asarray(mm[lo:hi], dtype=np.float32) / self.scale
                        out[i + lo - k:i + hi - k] = q[:, 0] + 1j * q[:, 1]
                        mask[i + lo - k:i + hi - k] = True
                del mm
            i += m
        return (out, mask) if return_mask else out

    def expire(self, now: float | None = None) -> list[int]:
        """Delete hours that ended more than keep_hours before `now`; return them."""
        now = time.time() if now is None else float(now)
        gone = []
        for hour in self.hours():
            if (hour + 1) * 3600 <= now - self.keep_hours * 3600:
                for p in self._paths(hour):
                    p.unlink(missing_ok=True)
                gone.append(hour)
        return gone


def _merge_ranges(rs) -> list[list[int]]:
    out: list[list[int]] = []
    for lo, hi in sorted(rs):
        if out and lo <= out[-1][1]:
            out[-1][1] = max(out[-1][1], hi)
        else:
            out.append([int(lo), int(hi)])
    return out


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)

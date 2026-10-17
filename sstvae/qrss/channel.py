"""QRSSTVAE channel simulator (design section 4, WP4).

Works in the receiver's front-end format: complex at FE_FS = 4000 Hz
with 0 Hz at FE_CENTER_HZ = 1500 Hz audio, so a full slot is about
7.2 M samples and nothing passes through 8 kHz real audio unless a WAV
is wanted (`fe_to_audio`).

Not a format module: this is the one place in `sstvae/qrss/` (with the
CLIs and tests) allowed to draw from `default_rng`. Every random choice
comes from `ChannelConfig.seed` (or a neighbour's own seed), so a
simulation is reproducible.

The steps are applied in the order a real link sees them:

1. **Clocks.** The transmit baseband is evaluated directly on the
   receiver's sample grid with time_scale = (1 + tx_ppm)/(1 + rx_ppm)
   (both clocks agree at t0), which is exact and avoids the circular
   FFT resampling of `hfchannel.sample_clock_offset`. The audio
   carrier is scaled by the same factor, so a sound-card-generated
   carrier at carrier_hz lands at carrier_hz*time_scale.
2. **Frequency trajectory.** offset + drift (linear, or warm-up
   settling) + Ornstein-Uhlenbeck wander. Its phase is integrated in
   closed form -- exactly, for the piecewise-linear wander too -- in
   float64 turns and wrapped before `exp`.
3. **Watterson fading.** Two equal-power Gaussian-Doppler Rayleigh
   paths, the second delayed; "steady" is a unit gain.
4. **Neighbours.** Other stations, CE or L, with their own random latents.
5. **AWGN** in the `SNR_REF_BW_HZ` convention of `hfchannel.awgn`,
   against the wanted signal's power over its keyed samples.
6. **Impulses and crashes.**
7. **AGC.** A peak detector with attack and decay on the smoothed
   envelope of the whole front end, dividing it out.

`Truth` records what a receiver should find: the complex gain u (fading
times the trajectory phasor, relative to a carrier at the nominal
carrier_hz with zero phase at t0), the carrier's audio frequency and
the receive time of every position.

`transmit_source` supplies the transmitted signal, built once per
transmission. For CE it is `ce.baseband` (WP3), a closed form evaluated
chunk by chunk at any time; until that module exists an internal stub
stands in: a bare carrier keyed over the same span with the same ramps,
enough for every channel property tested here (`baseband_is_stub()`
says which one is in use).

**Waveform L** (`lframe.LFrameSpec`, `lwave`) goes through the same
steps. Its named frequency (carrier_hz, a neighbour's df_hz, truth.f_hz)
is the pilot, and its baseband is pilot-centred, so nothing after the
source changes: evaluating it at t*scale scales the data's -f_p offset
too, and the clock's share of the carrier error scales the pilot, so
the data land at (pilot - f_p)*scale as they should. Three things do
differ. The clipper runs over the whole frame (`lwave.frame_baseband`),
so the source builds it once and serves chunks from it; recomputing it
per chunk would cost a FULL slot 30 s and more. The data centre sits
f_p = 12.58 Hz below the pilot, so on a delayed second path its gain is
not the pilot's: `Truth.u_data` holds it (0.3-1.9 dB of genie error on
the presets otherwise). And the signal power P is taken over the keying
span itself: L's envelope has nulls, which CE's amplitude rule would
drop (about 0.5% of the samples, 0.02 dB of P).
"""

from __future__ import annotations

import importlib
import math
from dataclasses import dataclass, field
from fractions import Fraction

import numpy as np
from scipy import signal

from sstvae import config as _c
from sstvae.modem import dsp

from . import frame as _frame
from . import lframe, lwave
from .ce import check_carrier
from .constants import FE_CENTER_HZ, FE_FS, FS, L_PILOT_HZ, WAVEFORM_CE, WAVEFORM_L
from .frame import FrameSpec
from .lframe import LFrameSpec

WANDER_RATE_HZ = 20.0                 # OU process sample rate before interpolation
AGC_FLOOR = 1e-9                      # never divide by less than this envelope
CHUNK = 1 << 16                       # samples per processing chunk (keeps temporaries in cache)


# --- configuration --------------------------------------------------------------

@dataclass(frozen=True)
class Neighbour:
    """Another station sharing the slot.

    Its named frequency (a CE carrier, an L pilot) is the wanted one's
    nominal named frequency plus df_hz, at rel_db relative to the wanted
    signal's power (each over its own keyed span), with its own random
    latents, header and callsign keying drawn from `seed`. Its frame is
    the wanted frame's twin in its own `waveform` ("short" <-> "l-short",
    `lframe.twin`).
    """
    df_hz: float
    rel_db: float
    lead_in_s: float = 0.0
    drift_hz_per_min: float = 0.0
    seed: int = 1
    waveform: int = WAVEFORM_CE


@dataclass(frozen=True)
class Impulses:
    """Clicks (Poisson, rate_hz) and crashes (Poisson, crash_per_min).

    Levels are peak power in dB above the wanted signal's rms. A click is
    a complex Gaussian burst decaying with tau_ms at a log-normal level
    amp_db +- amp_sd_db. A crash is crash_strokes (inclusive range)
    such bursts spread over crash_len_s, at a level uniform in crash_db
    and decay times uniform in crash_tau_ms.
    """
    rate_hz: float = 0.0
    amp_db: float = 30.0
    amp_sd_db: float = 6.0
    tau_ms: float = 1.0
    crash_per_min: float = 0.0
    crash_db: tuple = (30.0, 60.0)
    crash_strokes: tuple = (5, 20)
    crash_len_s: float = 0.5
    crash_tau_ms: tuple = (20.0, 100.0)


@dataclass(frozen=True)
class Agc:
    """Receive AGC: attack/decay of the peak detector, envelope smoothing."""
    attack_ms: float = 2.0
    decay_ms: float = 200.0
    smooth_ms: float = 5.0


@dataclass(frozen=True)
class ChannelConfig:
    snr_db: float | None = None                      # in SNR_REF_BW_HZ (2500), on keyed samples
    preset: str | None = None                        # key of PRESETS, or None with doppler/delay below
    doppler_hz: float = 0.0                          # Watterson 2-sigma spread
    delay_ms: float = 0.0                            # two equal paths if delay > 0 (or doppler > 0)
    freq_offset_hz: float = 0.0
    drift_hz_per_min: float = 0.0
    drift_tau_s: float | None = None                 # None: linear; else warm-up settling
    wander_rms_hz: float = 0.0                       # Ornstein-Uhlenbeck
    wander_tau_s: float = 10.0
    tx_ppm: float = 0.0
    rx_ppm: float = 0.0
    impulses: Impulses | None = None
    agc: Agc | None = None
    neighbours: tuple[Neighbour, ...] = ()
    seed: int = 0

    def __post_init__(self):
        if self.preset is not None and self.preset not in PRESETS:
            raise ValueError(f"unknown preset {self.preset!r}; one of {', '.join(PRESETS)}")
        object.__setattr__(self, "neighbours", tuple(self.neighbours))

    @property
    def fading(self) -> tuple[float, float] | None:
        """(doppler_hz, delay_ms), or None for a unit-gain (steady) path."""
        d, tau = PRESETS[self.preset] if self.preset is not None else \
            (self.doppler_hz, self.delay_ms)
        if d <= 0 and tau <= 0:
            return None
        return float(d), float(tau)

    @property
    def time_scale(self) -> float:
        """Transmit seconds per receive second, both counted from t0."""
        return (1 + self.tx_ppm * 1e-6) / (1 + self.rx_ppm * 1e-6)


# (doppler_hz, delay_ms), decision D24
PRESETS = {"steady": (0.0, 0.0), "quiet": (0.1, 0.5), "moderate": (0.5, 1.0),
           "disturbed": (1.0, 2.0), "mps": (0.15, 2.0)}


@dataclass
class Truth:
    """What the channel did, for scoring a receiver.

    Arrays over positions are on the receiver's clock: t_pos_s is the
    time (s after t0) at which position p is centred; f_hz the wanted
    carrier's audio frequency there (trajectory only, no fading); u the
    complex gain relative to a carrier at the nominal carrier_hz with
    zero phase at t0, so the received wanted signal near position p is
    u_p * s(t) * exp(2 pi j (carrier_hz - 1500)(t - t0)).
    """
    u: np.ndarray                            # complex128[n_pos]
    f_hz: np.ndarray                         # float64[n_pos]
    t_pos_s: np.ndarray                      # float64[n_pos]
    noise_var_fe: float                      # E|n|^2 of the added white noise (pre-AGC)
    impulse_mask: np.ndarray = field(repr=False)        # bool[n_fe]: impulse above signal + noise
    agc_gain: np.ndarray | None = field(repr=False, default=None)   # float32[n_fe]
    signal_power: float = 1.0                # P, wanted power over keyed samples, pre-AGC
    time_scale: float = 1.0                  # transmit s per receive s
    # L only (None for CE): u at the data centre, pilot - f_p*scale, in u's
    # convention otherwise -- the gain the data band sees, which a delayed
    # second path makes differ from the pilot's.
    u_data: np.ndarray | None = field(repr=False, default=None)    # complex128[n_pos]


@dataclass
class Sim:
    fe: np.ndarray                           # complex64[n]
    fs: int
    t0_index: float                          # FE sample index of t0 (receiver's clock)
    q: int
    spec: FrameSpec | LFrameSpec
    truth: Truth


# --- transmit signal -----------------------------------------------------------------

def _ce_baseband():
    """`ce.baseband`, or None while WP3's module does not exist yet."""
    try:
        ce = importlib.import_module(f"{__package__}.ce")
    except ModuleNotFoundError as e:
        if e.name != f"{__package__}.ce":
            raise                            # ce exists but is broken: say so
        return None
    return getattr(ce, "baseband", None)


def baseband_is_stub() -> bool:
    """True while the internal carrier stub stands in for `ce.baseband`."""
    return _ce_baseband() is None


def _raised_cosine_gate(t, on, off, ramp):
    """1 on [on, off), raised-cosine ramps of `ramp` s outside it, else 0."""
    a = ((t >= on) & (t < off)).astype(np.float64)
    if ramp > 0:
        up = (t >= on - ramp) & (t < on)
        a[up] = 0.5 * (1 - np.cos(np.pi * (t[up] - (on - ramp)) / ramp))
        dn = (t >= off) & (t < off + ramp)
        a[dn] = 0.5 * (1 + np.cos(np.pi * (t[dn] - off) / ramp))
    return a


def _tone_stub(sym, spec, fs, start_s, n, *, keying=None, ook=False, lead_in_s=0.0,
               ramp_s=0.05, time_scale=1.0):
    """Stand-in for `ce.baseband`: the bare carrier (phi = 0) over the keyed span."""
    t = (start_s + np.arange(n) / fs) * time_scale
    on = spec.keyed_start_pos * spec.t_sym - lead_in_s
    off = spec.keyed_end_pos * spec.t_sym
    return _raised_cosine_gate(t, on, off, ramp_s).astype(np.complex128)


# lwave.resample's interpolation filter is designed at 8 kHz whatever the
# rate: right at FS, fine at FE_FS (-60 dB against FS), wrong elsewhere.
_L_EXACT_FS = (FE_FS, FS)


def _l_source(bb: lwave.Baseband, fs, start_s, n, time_scale=1.0) -> np.ndarray:
    """An L frame's clipped baseband at t0 + (start_s + i/fs)*time_scale.

    On an exact grid (no clock error, start on a sample of FE_FS or FS)
    this is `lwave.resample`; otherwise the frame is resampled at FS over
    the span the times cover and interpolated linearly: -96 dB against
    a band-limited delay of the FS resample (measured on L_TINY and
    L_SHORT), far below that filter's own -61 dB at FE_FS.
    """
    n = int(n)
    x = start_s * fs
    i0 = round(x)
    if time_scale == 1.0 and fs in _L_EXACT_FS and abs(x - i0) < 1e-6:
        return lwave.resample(bb, fs, i0, n)
    if n <= 0:
        return np.zeros(0, dtype=np.complex128)
    k = (start_s + np.arange(n) / fs) * time_scale * FS        # transmit time in FS samples
    lo = math.floor(min(k[0], k[-1])) - 2
    hi = math.ceil(max(k[0], k[-1])) + 3
    g = lwave.resample(bb, FS, lo, hi - lo)
    x = k - lo
    j = x.astype(np.int64)
    gj = g[j]
    return gj + (g[j + 1] - gj) * (x - j)


def transmit_source(sym, spec, *, keying=None, ook=False, lead_in_s=0.0):
    """The wanted signal, built once: src(fs, start_s, n, time_scale=1.0) gives
    complex128[n] at t0 + (start_s + i/fs)*time_scale.

    CE: `ce.baseband` evaluated per call, as `transmit_baseband` always did
    (its closed form costs nothing extra per chunk). L: the whole frame
    through the clipper once (`lwave.frame_baseband`), then `_l_source`
    per call; on-off keying is CE's and is refused. `src.keyed_power` is
    the mean power over the keying span, for scaling a neighbour (1 for
    CE's unit envelope, which is never rescaled).
    """
    spec = lframe.get_any(spec)
    if spec.waveform == WAVEFORM_L:
        if ook:
            raise ValueError("waveform L keys its callsign windows by FSK of its pilot; "
                             "on-off keying is waveform CE's option (spec 2.7)")
        bb = lwave.frame_baseband(sym, spec, keying=keying, lead_in_s=lead_in_s)
        lo, hi = lwave.keyed_span_pos(spec, lead_in_s)
        t = np.arange(bb.n0, bb.n0 + len(bb.z)) / bb.fs
        on = (t >= lo * spec.t_sym) & (t < hi * spec.t_sym)

        def l_src(fs, start_s, n, time_scale=1.0):
            return _l_source(bb, fs, start_s, n, time_scale)
        l_src.keyed_power = float(np.mean(np.abs(bb.z[on]) ** 2))
        return l_src
    fn = _ce_baseband() or _tone_stub

    def ce_src(fs, start_s, n, time_scale=1.0):
        return np.asarray(fn(sym, spec, fs, start_s, n, keying=keying, ook=ook,
                             lead_in_s=lead_in_s, time_scale=time_scale),
                          dtype=np.complex128)
    ce_src.keyed_power = 1.0
    return ce_src


def transmit_baseband(sym, spec, fs, start_s, n, *, keying=None, ook=False,
                      lead_in_s=0.0, time_scale=1.0) -> np.ndarray:
    """complex128[n]: the wanted signal at t0 + (start_s + i/fs)*time_scale.

    One call of `transmit_source`: free for CE, but an L frame runs its
    whole-frame clipper on every call, so anything chunked builds a
    source once instead.
    """
    return transmit_source(sym, spec, keying=keying, ook=ook,
                           lead_in_s=lead_in_s)(fs, start_s, n, time_scale)


def _check_freq(spec, carrier_hz) -> float:
    """The named frequency in Hz: a CE carrier (`ce.check_carrier`), or an L
    pilot whose data, 22.07 Hz below it, must stay in band (`tx.check_pilot`)."""
    if spec.waveform == WAVEFORM_L:
        from .tx import check_pilot
        return check_pilot(carrier_hz)
    return check_carrier(carrier_hz)


# --- front-end conversion ------------------------------------------------------------
# The front end proper (blanker, normaliser, channeliser) is WP5's
# `frontend.py`; these two conversions are needed here already, for the
# WAV paths and test C-1, and follow design 6.2 exactly. `frontend`'s
# versions are used once that module provides them.

_FE_DECIM = FS // FE_FS                       # 2
_FE_TAPS = signal.firwin(255, 1250.0, fs=FS, window=("kaiser", 8.0))


def _frontend_fn(name):
    try:
        frontend = importlib.import_module(f"{__package__}.frontend")
    except ModuleNotFoundError as e:
        if e.name != f"{__package__}.frontend":
            raise
        return None
    return getattr(frontend, name, None)


def audio_to_fe(x8k) -> np.ndarray:
    """8 kHz real audio -> complex64 FE at 4 kHz, 0 Hz = 1500 Hz. Power preserved."""
    fn = _frontend_fn("audio_to_fe")
    if fn is not None:
        return fn(x8k)
    assert _c.FCENTER == FE_CENTER_HZ
    z = np.sqrt(2.0) * dsp.to_baseband(np.asarray(x8k, dtype=np.float64))
    z = signal.fftconvolve(z, _FE_TAPS, mode="same")
    return z[::_FE_DECIM].astype(np.complex64)


def fe_to_audio(fe) -> np.ndarray:
    """Inverse of `audio_to_fe`: upsample by 2, mix up to 1500 Hz, sqrt(2)*Re."""
    fn = _frontend_fn("fe_to_audio")
    if fn is not None:
        return fn(fe)
    fe = np.asarray(fe, dtype=np.complex128)
    up = np.zeros(len(fe) * _FE_DECIM, dtype=np.complex128)
    up[::_FE_DECIM] = fe * _FE_DECIM
    up = signal.fftconvolve(up, _FE_TAPS, mode="same")
    # mix up: the conjugate of to_baseband's exact 16-phasor table
    return np.sqrt(2.0) * np.real(np.conj(dsp.to_baseband(np.ones(len(up)))) * up)


# --- frequency trajectory --------------------------------------------------------------

def _wrap(turns):
    return dsp.wrap_cycles(turns)


def nominal_turns(f_hz: float, n, fs: int = FE_FS, n0: float = 0.0) -> np.ndarray:
    """Phase in turns, in [0, 1), of a tone at f_hz at samples n, zero at n0.

    Exact integer turn arithmetic, (n*f_num mod f_den*fs)/(f_den*fs),
    whenever f_hz is a short rational (any integer or 1/64-Hz frequency);
    a float product reduced before exp otherwise. A fractional n0 only
    costs one constant, reduced the same way.
    """
    n = np.asarray(n, dtype=np.int64)
    fr = Fraction(f_hz).limit_denominator(1 << 16)
    if float(fr) == float(f_hz):
        num, den = fr.numerator, fr.denominator
        mod = den * fs
        turns = ((n % mod) * (num % mod) % mod) / mod
    else:
        turns = _wrap(f_hz * (n / fs))
    if n0:
        turns = turns - float(_wrap(np.float64(f_hz) * (n0 / fs)))
    return _wrap(turns)


class Trajectory:
    """f(t) = offset + drift + OU wander; theta(t) its integral in turns.

    Times are seconds after t0 on the receiver's clock and theta(0) = 0.
    Drift is drift_hz_per_min * t/60 when linear; with a settling time tau
    it is (D/60)*tau*(1 - exp(-t_k/tau)) with t_k counted from key-up
    (`t_key`, lead-in included), so D is the initial rate. Wander is an
    OU process sampled at WANDER_RATE_HZ over [t_start, t_end] and
    linearly interpolated; its phase is the exact integral of that
    piecewise-linear frequency.
    """

    def __init__(self, offset_hz=0.0, drift_hz_per_min=0.0, drift_tau_s=None,
                 wander_rms_hz=0.0, wander_tau_s=10.0, *, t_start=0.0, t_end=0.0,
                 t_key=0.0, rng=None):
        self.offset = float(offset_hz)
        self.drift = float(drift_hz_per_min) / 60.0          # Hz/s
        self.tau = drift_tau_s
        self.t_key = float(t_key)
        self.wander = None
        if wander_rms_hz > 0:
            dt = 1.0 / WANDER_RATE_HZ
            k = int(math.ceil((t_end - t_start) / dt)) + 2
            a = math.exp(-dt / wander_tau_s)
            xi = rng.standard_normal(k)
            f = np.empty(k)
            f[0] = wander_rms_hz * xi[0]                     # stationary start
            # f[n+1] = a f[n] + sqrt(1-a^2) sigma xi: one linear recursion
            f[1:] = math.sqrt(1 - a * a) * wander_rms_hz * xi[1:]
            f = signal.lfilter([1.0], [1.0, -a], f)
            # exact integral at the knots (trapezoid of a linear segment)
            W = np.concatenate([[0.0], np.cumsum(0.5 * (f[1:] + f[:-1]) * dt)])
            self.wander = (float(t_start), dt, f, W)
            self._w0 = 0.0
            self._w0 = float(self._wander_integral(np.zeros(1))[0])

    def _drift_f(self, t):
        if self.tau is None:
            return self.drift * t
        return self.drift * self.tau * (1 - np.exp(-(t - self.t_key) / self.tau))

    def _drift_theta(self, t):
        if self.tau is None:
            return 0.5 * self.drift * t * t
        tau = self.tau

        def F(s):
            return self.drift * tau * (s + tau * np.exp(-s / tau))
        return F(t - self.t_key) - F(0.0 - self.t_key)

    def _wander_pos(self, t):
        """(knot index k, offset into the segment in units of dt)."""
        t0, dt, f, _ = self.wander
        x = (np.asarray(t, dtype=np.float64) - t0) * (1.0 / dt)
        k = x.astype(np.int64)                         # t >= t0 by construction: floor
        np.clip(k, 0, len(f) - 2, out=k)
        return k, x - k

    def _wander_f(self, t):
        _, _, f, _ = self.wander
        k, u = self._wander_pos(t)
        fk = f[k]
        return fk + (f[k + 1] - fk) * u

    def _wander_integral(self, t):
        _, dt, f, W = self.wander
        k, u = self._wander_pos(t)
        fk = f[k]
        return W[k] + dt * u * (fk + 0.5 * (f[k + 1] - fk) * u) - self._w0

    def f(self, t) -> np.ndarray:
        """Frequency deviation in Hz at times t."""
        t = np.asarray(t, dtype=np.float64)
        out = self.offset + self._drift_f(t)
        if self.wander is not None:
            out = out + self._wander_f(t)
        return out

    def theta(self, t) -> np.ndarray:
        """Integral of f from t0 to t, in turns (unwrapped, float64)."""
        t = np.asarray(t, dtype=np.float64)
        out = self.offset * t + self._drift_theta(t)
        if self.wander is not None:
            out = out + self._wander_integral(t)
        return out

    def wander_f(self, t) -> np.ndarray:
        t = np.asarray(t, dtype=np.float64)
        return np.zeros_like(t) if self.wander is None else self._wander_f(t)


# --- fading -----------------------------------------------------------------------------

class _Taps:
    """One unit-power Gaussian-Doppler Rayleigh tap, evaluated on demand.

    The `hfchannel._gaussian_taps` recipe at a general rate: shaped in
    the frequency domain at max(64*spread, 8) Hz (circular, so no
    transient) and linearly interpolated, normalised so the n samples at
    fs have unit mean power. A zero spread is a constant tap of random
    phase. Evaluated chunk by chunk (`__call__`) or at fractional sample
    indices (`at`) on the same piecewise-linear curve.
    """

    def __init__(self, n: int, spread_hz: float, fs: float, rng):
        self.const = None
        if spread_hz <= 0:
            g = rng.normal() + 1j * rng.normal()
            self.const = g / abs(g)
            return
        lowrate = max(64 * spread_hz, 8.0)
        n_low = int(np.ceil(n * lowrate / fs)) + 2
        g = np.fft.fft(rng.normal(size=n_low) + 1j * rng.normal(size=n_low))
        f = np.fft.fftfreq(n_low, 1 / lowrate)
        self.g = np.fft.ifft(g * np.exp(-(f ** 2) / (4 * (spread_hz / 2) ** 2)))
        self.r = lowrate / fs                  # low-rate index per sample
        self.scale = 1.0
        p = sum(float(np.sum(np.abs(self(a, min(n, a + CHUNK))) ** 2))
                for a in range(0, n, CHUNK))
        self.scale = 1.0 / math.sqrt(p / n)

    def at(self, idx) -> np.ndarray:
        """The tap at (fractional) sample indices idx."""
        idx = np.asarray(idx, dtype=np.float64)
        if self.const is not None:
            return np.full(idx.shape, self.const)
        x = np.clip(idx * self.r, 0.0, len(self.g) - 1.000001)
        k = x.astype(np.int64)
        gk = self.g[k]
        return (gk + (self.g[k + 1] - gk) * (x - k)) * self.scale

    def __call__(self, a: int, b: int) -> np.ndarray:
        if self.const is not None:
            return np.full(b - a, self.const)
        x = np.arange(a, b) * self.r
        k = x.astype(np.int64)
        gk = self.g[k]
        return (gk + (self.g[k + 1] - gk) * (x - k)) * self.scale


def gaussian_taps(n: int, spread_hz: float, fs: float, rng) -> np.ndarray:
    """complex128[n]: unit-power Rayleigh tap, Gaussian Doppler with 2 sigma = spread.

    `hfchannel._gaussian_taps` re-implemented at a general rate (the
    private function is not imported).
    """
    return _Taps(n, spread_hz, fs, rng)(0, n)


class _Fading:
    """Two equal-power paths, the second delayed by d samples, summed / sqrt(2)."""

    def __init__(self, n, doppler_hz, delay_ms, fs, rng):
        self.d = int(round(delay_ms * 1e-3 * fs))
        self.g1 = _Taps(n, doppler_hz, fs, rng)
        self.g2 = _Taps(n, doppler_hz, fs, rng)

    def apply(self, y, a, b, prev_tail):
        """Faded chunk [a, b) of y (chunk values), given the d samples before it."""
        y2 = np.concatenate([prev_tail, y])[:len(y)] if self.d else y
        return (self.g1(a, b) * y + self.g2(a, b) * y2) / math.sqrt(2)

    def next_tail(self, prev_tail, y):
        """The last d unfaded samples up to the end of this chunk."""
        return np.concatenate([prev_tail, y])[len(prev_tail) + len(y) - self.d:]


# --- impulses ---------------------------------------------------------------------------

def _bursts(out, starts, amps, taus_s, fs, rng):
    """Add complex Gaussian bursts amp*xi[k]*exp(-k/(tau fs)) at sample starts."""
    n = len(out)
    for s, a, tau in zip(starts, amps, taus_s):
        L = min(int(math.ceil(8 * tau * fs)) + 1, n - s)
        if L <= 0:
            continue
        k = np.arange(L)
        xi = (rng.standard_normal(L) + 1j * rng.standard_normal(L)) / math.sqrt(2)
        out[s:s + L] += a * xi * np.exp(-k / (tau * fs))


def impulse_noise(n: int, fs: float, rms: float, imp: Impulses, rng):
    """(complex128[n] impulse waveform, events) for clicks and crashes.

    `events` is a list of (t_s, level_db, kind) with kind "click" or
    "crash"; level_db is peak power above rms**2 (a crash's own level;
    its strokes scatter +-3 dB about it).
    """
    out = np.zeros(n, dtype=np.complex128)     # lazily zeroed: only bursts touch pages
    events = []
    dur = n / fs
    if imp.rate_hz > 0:
        m = rng.poisson(imp.rate_hz * dur)
        t = np.sort(rng.uniform(0, dur, m))
        lev = rng.normal(imp.amp_db, imp.amp_sd_db, m)
        _bursts(out, (t * fs).astype(np.int64), rms * 10 ** (lev / 20),
                np.full(m, imp.tau_ms * 1e-3), fs, rng)
        events += [(float(a), float(b), "click") for a, b in zip(t, lev)]
    if imp.crash_per_min > 0:
        m = rng.poisson(imp.crash_per_min / 60.0 * dur)
        for tc in np.sort(rng.uniform(0, dur, m)):
            lev = rng.uniform(*imp.crash_db)
            k = rng.integers(imp.crash_strokes[0], imp.crash_strokes[1] + 1)
            ts = tc + rng.uniform(0, imp.crash_len_s, k)
            ts = ts[ts < dur]
            sl = lev + rng.normal(0, 3.0, len(ts))
            taus = rng.uniform(*imp.crash_tau_ms, len(ts)) * 1e-3
            _bursts(out, (ts * fs).astype(np.int64), rms * 10 ** (sl / 20), taus, fs, rng)
            events.append((float(tc), float(lev), "crash"))
    return out, events


# --- AGC ---------------------------------------------------------------------------------

class _AgcDetector:
    """Stateful peak detector: feed |x| chunk by chunk, get the envelope.

    Causal boxcar of smooth_ms, then a decay-hold max(a_k d^(n-k))
    computed in the log domain as a running maximum (no Python loop over
    samples), then a one-pole attack smoother, which leaves the much
    slower decay essentially untouched.
    """

    def __init__(self, fs: float, agc: Agc):
        self.m = max(1, int(round(agc.smooth_ms * 1e-3 * fs)))
        self.rate = 1.0 / (agc.decay_ms * 1e-3 * fs)       # -log d per sample
        self.al = 1 - math.exp(-1.0 / (agc.attack_ms * 1e-3 * fs))
        self.tail = np.zeros(0)
        self.carry = -np.inf                                 # log held peak before the chunk
        self.zi = None

    def __call__(self, a) -> np.ndarray:
        a = np.asarray(a, dtype=np.float64)
        if self.m > 1:
            h = np.concatenate([self.tail, a])
            c = np.concatenate([[0.0], np.cumsum(h)])
            i = np.arange(len(self.tail), len(h)) + 1
            lo = np.maximum(i - self.m, 0)
            sm = (c[i] - c[lo]) / (i - lo)
            self.tail = h[-(self.m - 1):]
        else:
            sm = a
        k = np.arange(len(sm))
        r = self.rate
        with np.errstate(divide="ignore"):
            lh = np.maximum(np.maximum.accumulate(np.log(sm) + k * r) - k * r,
                            self.carry - (k + 1) * r)
        self.carry = lh[-1]
        held = np.exp(lh)
        if self.zi is None:
            self.zi = [(1 - self.al) * held[0]]
        env, self.zi = signal.lfilter([self.al], [1.0, -(1 - self.al)], held, zi=self.zi)
        return env


def agc_envelope(x, fs: float, agc: Agc) -> np.ndarray:
    """The AGC's envelope of x (its gain is 1/max(env, AGC_FLOOR))."""
    det = _AgcDetector(fs, agc)
    a = np.abs(np.asarray(x))
    return np.concatenate([det(a[s:s + CHUNK]) for s in range(0, len(a), CHUNK)])


# --- the channel --------------------------------------------------------------------------

def _neighbour_symbols(nb: Neighbour, spec):
    """(sym, keying) of a neighbour: random latents and header bits, its own call.

    `spec` is the neighbour's own frame (CE or L), so the draws are the
    same for either and only the assembly differs.
    """
    from .morse import keying_units

    rng = np.random.default_rng(nb.seed)
    hdr = rng.integers(0, 2, _frame.N_HDR_BITS, dtype=np.uint8) if spec.has_header else None
    assemble = lframe.assemble if spec.waveform == WAVEFORM_L else _frame.assemble
    sym = assemble(spec, hdr, rng.standard_normal(spec.n_data))
    keying = keying_units(f"N{nb.seed % 10}NBR") if spec.n_win else None
    return sym, keying


def _channel(src, n, t_start, spec, q, cfg: ChannelConfig, *, carrier_hz, lead_in_s,
             mix_nominal, rng, pos_t=None, keyed_s=None) -> Sim:
    """Steps 2-7 on the clean wanted signal; `src(a, b)` gives its samples [a, b).

    Sample i is at receive time t_start + i/FE_FS (s after t0).
    `mix_nominal`: the source is at 0 Hz and still needs mixing to
    carrier_hz (the synthesis path), rather than already carrying its
    carrier and its clock scaling (the WAV path). `keyed_s`: the keying
    span [on, off) in receive time, over which P is measured (L); None
    keeps CE's rule, samples above a tenth of the rms amplitude.

    Processed in chunks of CHUNK samples so no full-length temporaries
    are made: pass 1 forms the faded wanted signal and measures its
    power P, pass 2 adds neighbours, noise and impulses and runs the AGC.
    """
    fs = FE_FS
    scale = cfg.time_scale
    c_off = carrier_hz * (scale - 1)                   # clock error on the audio carrier
    n0 = -t_start * fs                                 # t0's sample index
    traj = Trajectory(cfg.freq_offset_hz + c_off, cfg.drift_hz_per_min, cfg.drift_tau_s,
                      cfg.wander_rms_hz, cfg.wander_tau_s, t_start=t_start - 1.0,
                      t_end=t_start + n / fs + 1.0,
                      t_key=spec.keyed_start_pos * spec.t_sym - lead_in_s, rng=rng)
    fad = cfg.fading
    fading = None if fad is None else _Fading(n, fad[0], fad[1], fs, rng)

    # pass 1: wanted signal -- clock (in src), trajectory, fading
    out = np.empty(n, dtype=np.complex64)
    amp = np.empty(n, dtype=np.float32)                # |clean| for the keyed rule
    tail = np.zeros(fading.d if fading else 0, dtype=np.complex128)
    for a in range(0, n, CHUNK):
        b = min(n, a + CHUNK)
        x = src(a, b)
        amp[a:b] = np.abs(x)
        t = t_start + np.arange(a, b) / fs
        th = traj.theta(t)
        if not mix_nominal:
            th = th - c_off * t                        # the resampling already did it
        turns = _wrap(th)
        if mix_nominal:
            turns = turns + nominal_turns(carrier_hz - FE_CENTER_HZ, np.arange(a, b), fs, n0)
        y = x * np.exp(2j * np.pi * _wrap(turns))
        if fading is not None:
            y, tail = fading.apply(y, a, b, tail), fading.next_tail(tail, y)
        out[a:b] = y
    if keyed_s is None:
        keyed = amp > 0.1 * math.sqrt(float(np.mean(amp.astype(np.float64) ** 2)))
    else:
        keyed = np.zeros(n, dtype=bool)
        i_on, i_off = (min(n, max(0, math.ceil((s - t_start) * fs))) for s in keyed_s)
        keyed[i_on:i_off] = True
    if not keyed.any():
        keyed[:] = True
    P = sum(float(np.sum(np.abs(out[a:a + CHUNK][keyed[a:a + CHUNK]].astype(np.complex128))
                         ** 2)) for a in range(0, n, CHUNK)) / int(keyed.sum())

    # truth at positions
    t_pos = (np.arange(spec.n_pos) * spec.t_sym / scale if pos_t is None
             else np.asarray(pos_t, dtype=np.float64))
    u = np.exp(2j * np.pi * _wrap(traj.theta(t_pos)))
    f_true = carrier_hz + traj.f(t_pos)
    u_data = None
    if spec.waveform == WAVEFORM_L:
        f_data = f_true - float(L_PILOT_HZ) * scale        # the data centre, clock-scaled
        u_data = u.copy()
    if fading is not None:
        # composite narrowband gain: path 2 lags by d samples of carrier phase
        idx = (t_pos - t_start) * fs
        g1, g2 = fading.g1.at(idx), fading.g2.at(idx)
        if u_data is not None:
            lag = np.exp(-2j * np.pi * _wrap((f_data - FE_CENTER_HZ) * fading.d / fs))
            u_data = u_data * (g1 + g2 * lag) / math.sqrt(2)
        lag = np.exp(-2j * np.pi * _wrap((f_true - FE_CENTER_HZ) * fading.d / fs))
        u = u * (g1 + g2 * lag) / math.sqrt(2)

    # pass 2: neighbours, AWGN, impulses, AGC
    nbs = []
    for nb in cfg.neighbours:
        nspec = lframe.twin(spec, nb.waveform)
        sym_n, key_n = _neighbour_symbols(nb, nspec)
        nsrc = transmit_source(sym_n, nspec, keying=key_n, lead_in_s=nb.lead_in_s)
        nrng = np.random.default_rng([cfg.seed, nb.seed, 7])
        nts = 1.0 / (1 + cfg.rx_ppm * 1e-6)
        f_nb = (carrier_hz + nb.df_hz) * nts - FE_CENTER_HZ
        ntraj = Trajectory(0.0, nb.drift_hz_per_min,
                           t_key=nspec.keyed_start_pos * nspec.t_sym - nb.lead_in_s)
        nfad = None if fad is None else _Fading(n, fad[0], fad[1], fs, nrng)
        p_nb = P * 10 ** (nb.rel_db / 10)
        if nspec.waveform == WAVEFORM_L:
            p_nb = p_nb / nsrc.keyed_power                 # CE's unit envelope needs none
        gain = math.sqrt(p_nb) * np.exp(2j * np.pi * nrng.uniform())
        nbs.append([nsrc, nb, nts, f_nb, ntraj, nfad, gain,
                    np.zeros(nfad.d if nfad else 0, dtype=np.complex128)])
    noise_var = 0.0
    if cfg.snr_db is not None:
        noise_var = P * fs / (_c.SNR_REF_BW_HZ * 10 ** (cfg.snr_db / 10))
    imp = None
    mask = np.zeros(n, dtype=bool)
    if cfg.impulses is not None:
        imp, _ = impulse_noise(n, fs, math.sqrt(P), cfg.impulses, rng)
    det = None if cfg.agc is None else _AgcDetector(fs, cfg.agc)
    agc_gain = None if det is None else np.empty(n, dtype=np.float32)
    for a in range(0, n, CHUNK):
        b = min(n, a + CHUNK)
        y = out[a:b].astype(np.complex128)
        for nbx in nbs:
            nsrc, nb, nts, f_nb, ntraj, nfad, gain, ntail = nbx
            t = t_start + np.arange(a, b) / fs
            z = nsrc(fs, t_start + a / fs, b - a, nts)
            z = z * np.exp(2j * np.pi * _wrap(
                nominal_turns(f_nb, np.arange(a, b), fs, n0) + _wrap(ntraj.theta(t))))
            if nfad is not None:
                z, nbx[7] = nfad.apply(z, a, b, ntail), nfad.next_tail(ntail, z)
            y += gain * z
        if noise_var > 0:
            y += math.sqrt(noise_var / 2) * rng.standard_normal(2 * (b - a)).view(np.complex128)
        if imp is not None:
            ic = imp[a:b]
            mask[a:b] = ic.real ** 2 + ic.imag ** 2 > P + noise_var
            y += ic
        if det is not None:
            g = 1.0 / np.maximum(det(np.abs(y)), AGC_FLOOR)
            agc_gain[a:b] = g
            y *= g
        out[a:b] = y

    truth = Truth(u=u, f_hz=f_true, t_pos_s=t_pos, noise_var_fe=float(noise_var),
                  impulse_mask=mask, agc_gain=agc_gain, signal_power=P, time_scale=scale,
                  u_data=u_data)
    return Sim(fe=out, fs=fs, t0_index=n0, q=int(q), spec=spec, truth=truth)


def _l_keyed_tx_s(spec, lead_in_s) -> tuple[float, float] | None:
    """An L frame's keying span [on, off) in transmit seconds after t0; None for CE."""
    if spec.waveform != WAVEFORM_L:
        return None
    lo, hi = lwave.keyed_span_pos(spec, lead_in_s)
    return lo * spec.t_sym, hi * spec.t_sym


def simulate(sym, spec, q, cfg: ChannelConfig, *, carrier_hz=1500.0, lead_in_s=0.0,
             keying=None, ook=False, pre_s=12.0, post_s=3.0) -> Sim:
    """Synthesise a slot of `sym` (stream symbols) through the channel `cfg`.

    The FE capture starts at t0 - pre_s on the receiver's clock and runs
    post_s past the end of keying. `keying` (uint8[192] from
    `morse.keying_units`, the header callsign's) and `ook` select the
    callsign windows' keying and go straight to the source
    (`transmit_source`); a transmit path must always give them (spec
    2.7). An L frame (`lframe.LFrameSpec`) takes complex symbols
    (`lframe.assemble`), carrier_hz is its pilot, and ook is refused.
    """
    spec = lframe.get_any(spec)
    _check_freq(spec, carrier_hz)
    if spec.n_win and keying is None:
        raise ValueError(f"frame {spec.name!r} has callsign windows: a transmission must "
                         "key the callsign (spec 2.7); pass keying=")
    if lead_in_s - spec.keyed_start_pos * spec.t_sym + 0.05 > pre_s:
        raise ValueError(f"pre_s = {pre_s} s does not cover the keying start with a "
                         f"{lead_in_s} s lead-in")
    scale = cfg.time_scale
    end_s = spec.keyed_end_pos * spec.t_sym / scale
    n = int(round((pre_s + end_s + post_s) * FE_FS))
    source = transmit_source(sym, spec, keying=keying, ook=ook, lead_in_s=lead_in_s)
    span = _l_keyed_tx_s(spec, lead_in_s)

    def src(a, b):
        return source(FE_FS, -pre_s + a / FE_FS, b - a, scale)

    return _channel(src, n, -pre_s, spec, q, cfg, carrier_hz=carrier_hz,
                    lead_in_s=lead_in_s, mix_nominal=True,
                    rng=np.random.default_rng(cfg.seed),
                    keyed_s=None if span is None else (span[0] / scale, span[1] / scale))


def simulate_audio(x8k, spec, q, cfg: ChannelConfig, *, carrier_hz=1500.0, lead_in_s=0.0,
                   pre_s=12.0) -> Sim:
    """The channel applied to transmit audio whose sample 0 is t0 - pre_s.

    Clock error is applied by `hfchannel.sample_clock_offset` on the
    audio (so time is scaled about sample 0, not t0, and truth.t_pos_s
    follows that); the trajectory then shifts the whole front end.
    truth.u's phase is relative to whatever carrier phase the audio had
    at t0, and carrier_hz is only used for truth.f_hz and the clock's
    share of the frequency error. For an L frame carrier_hz is the pilot
    and lead_in_s places the keying span P is measured over; truth.u
    excludes the audio's own amplitude, as for CE.
    """
    from sstvae import hfchannel

    spec = lframe.get_any(spec)
    _check_freq(spec, carrier_hz)
    x8k = np.asarray(x8k, dtype=np.float64)
    scale = cfg.time_scale
    if scale != 1.0:
        x8k = hfchannel.sample_clock_offset(x8k, (scale - 1) * 1e6)
    x = audio_to_fe(x8k)
    t_pos = (pre_s + np.arange(spec.n_pos) * spec.t_sym) / scale - pre_s
    span = _l_keyed_tx_s(spec, lead_in_s)
    return _channel(lambda a, b: x[a:b].astype(np.complex128), len(x), -pre_s, spec, q, cfg,
                    carrier_hz=carrier_hz, lead_in_s=lead_in_s, mix_nominal=False,
                    rng=np.random.default_rng(cfg.seed), pos_t=t_pos,
                    keyed_s=None if span is None else
                    tuple((pre_s + s) / scale - pre_s for s in span))


# --- files -------------------------------------------------------------------------------

_TRUTH_ARRAYS = ("u", "f_hz", "t_pos_s", "impulse_mask")


def save_npz(path, sim: Sim) -> None:
    """fe (c8), fs, t0_index, q, frame and the truth, as one .npz.

    The frame is stored by name, which names its waveform too ("l-..." is L).
    """
    tr = sim.truth
    arrays = {f"truth_{k}": getattr(tr, k) for k in _TRUTH_ARRAYS}
    if tr.agc_gain is not None:
        arrays["truth_agc_gain"] = tr.agc_gain
    if tr.u_data is not None:
        arrays["truth_u_data"] = tr.u_data
    np.savez(path, fe=sim.fe.astype(np.complex64), fs=sim.fs, t0_index=sim.t0_index,
             q=sim.q, frame=sim.spec.name, truth_noise_var_fe=tr.noise_var_fe,
             truth_signal_power=tr.signal_power, truth_time_scale=tr.time_scale, **arrays)


def load_npz(path) -> Sim:
    """The `Sim` written by `save_npz` (frame by preset name, either waveform).

    What save_npz never writes is refused as it is read (ValueError naming
    the member), not left for the receiver to trip over: fe other than
    complex64 (n,), fs <= 0, q < 0 (a quarter hour before 1970, which no
    slot is). The texts are the native reader's (native/core/qrss/simfile).
    """
    with np.load(path, allow_pickle=False) as d:
        truth = Truth(
            **{k: d[f"truth_{k}"] for k in _TRUTH_ARRAYS},
            noise_var_fe=float(d["truth_noise_var_fe"]),
            agc_gain=d["truth_agc_gain"] if "truth_agc_gain" in d else None,
            signal_power=float(d["truth_signal_power"]),
            time_scale=float(d["truth_time_scale"]),
            u_data=d["truth_u_data"] if "truth_u_data" in d else None)
        fe = d["fe"]
        # complex64 in either byte order: '>c8' is what save_npz writes on a
        # big-endian host, and np.load reads it anywhere (the native reader
        # refuses every big-endian member, test_qrss_native_infra).
        if fe.dtype.newbyteorder("=") != np.complex64 or fe.ndim != 1:
            raise ValueError(f"simfile: fe is {fe.dtype.str} of {fe.ndim} dimensions, not complex64 "
                             "(n,) as qrss_simulate.py writes it")
        fe = fe.astype(np.complex64, copy=False)
        fs = int(d["fs"])
        if fs <= 0:
            raise ValueError(f"simfile: fs {fs} is not a sample rate")
        t0_index = float(d["t0_index"])
        q = int(d["q"])
        if q < 0:
            raise ValueError(f"simfile: q {q} is a quarter hour before 1970-01-01T00:00Z")
        return Sim(fe=fe, fs=fs, t0_index=t0_index, q=q, spec=lframe.get_any(str(d["frame"])),
                   truth=truth)

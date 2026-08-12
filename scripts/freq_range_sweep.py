#!/usr/bin/env python3
"""Acquisition frequency range, and frequency drift during a transmission.

Reproduces the tables in docs/todo.md under "Wider acquisition search,
for a mis-tuned counterpart" and "Frequency drift *during* a
transmission". Neither question has a fixture to point at, and both
answers turned on measurements that are cheap to re-run and easy to get
wrong by reasoning, so the harness is kept rather than the numbers alone.

Two prototypes live here, deliberately outside the package because
neither is a decision that has been taken:

  `acquire_wide`         -- sync.acquire with max_bins and the sync
                            filter as parameters.
  `demodulate_tracked`   -- Modem.demodulate plus a second-order loop on
                            the pilots' *common* phase, which is residual
                            carrier frequency (the slope *across*
                            carriers is timing, and modem.py already
                            tracks that).

`demodulate_tracked` is a fork of `Modem.demodulate`, which is exactly
the kind of copy that rots. `--verify` is the answer: with tracking off
it must reproduce the reference bit for bit, on clean audio and under
two fading presets. Run it before believing any number this script
prints.

  python scripts/freq_range_sweep.py --verify
  python scripts/freq_range_sweep.py reach sensitivity cpu
  python scripts/freq_range_sweep.py drift drift-fading
"""
import argparse
import sys
import time

import numpy as np
from scipy import signal as sig
from scipy.fft import fft, ifft, next_fast_len

from sstvae import config, hfchannel
from sstvae.config import (
    FS, RS, NC, NC_LATENT, BEACON_CARRIER, CHIPS_PER_FRAME, M, NCP, NSYM,
    SYMS_PER_FRAME, FRAME_SAMPLES, LATENTS_PER_FRAME, LEADIN_SAMPLES,
    PREAMBLE_CP, PREAMBLE_REPEATS, PREAMBLE_SAMPLES, HEADER_SAMPLES,
    DEMOD_BACKOFF, PREAMBLE_THRESHOLD,
)
from sstvae.modem import beacon, dsp, framing, ofdm, sync
from sstvae.modem.modem import DemodResult, Modem, _estimate_snr_db

TRUE_START = LEADIN_SAMPLES
FRAME_S = FRAME_SAMPLES / FS
NARROW = sig.firwin(129, 850.0, fs=FS)


# --- signals ---------------------------------------------------------------

_TX = {}


def make_tx(mode="A", seed=0):
    if (mode, seed) not in _TX:
        spec = config.MODES[mode]
        rng = np.random.default_rng(seed)
        lat = rng.standard_normal(spec.n_latents)
        lat /= np.sqrt(np.mean(lat**2))
        _TX[(mode, seed)] = (spec, lat, Modem().modulate(lat, spec, callsign="TEST"))
    return _TX[(mode, seed)]


def latent_snr_db(tx_lat, rx_lat, weights=None):
    idx = np.flatnonzero(weights) if weights is not None else np.arange(len(tx_lat))
    if len(idx) == 0:
        return float("-inf")
    err = rx_lat[idx] - tx_lat[idx]
    return 10 * np.log10(np.mean(tx_lat[idx] ** 2) / max(np.mean(err**2), 1e-30))


def drift_shift(x, rate_hz_s, f0_hz=0.0, sinus=None):
    """Frequency offset f0 plus a linear drift of `rate` Hz/s, and
    optionally a sinusoidal term (amplitude Hz, period s). Phase is
    reduced to one turn before exp(), for the reason in dsp.wrap_cycles:
    over a whole transmission the unreduced argument reaches tens of
    thousands of radians."""
    t = np.arange(len(x)) / FS
    cycles = f0_hz * t + 0.5 * rate_hz_s * t**2
    if sinus is not None:
        amp, period = sinus
        cycles = cycles + amp * period / (2 * np.pi) * np.sin(2 * np.pi * t / period)
    return np.real(sig.hilbert(x) * np.exp(2j * np.pi * dsp.wrap_cycles(cycles)))


def ou_shift(x, rms_hz, tau_s, seed=0):
    """Frequency wander drawn from an Ornstein-Uhlenbeck process, which
    is what a mean-reverting random walk of the carrier looks like --
    closer to ionospheric Doppler than either a ramp or a sinusoid,
    because it has no characteristic phase and wanders back rather than
    running away.

    Parameterized the way a measurement would be: `rms_hz` is the
    stationary standard deviation of the offset, `tau_s` its correlation
    time. Exact AR(1) discretization (an OU process sampled at any
    spacing is exactly AR(1), so this is not an approximation), run
    through lfilter rather than a Python loop, and started from the
    stationary distribution so there is no warm-up transient sitting
    inside the transmission.

    Returns (drifted signal, true frequency path in Hz).
    """
    rng = np.random.default_rng(seed)
    n = len(x)
    a = np.exp(-1.0 / (tau_s * FS))
    c = rms_hz * np.sqrt(1.0 - a * a)
    xi = rng.standard_normal(n)
    f = sig.lfilter([c], [1.0, -a], xi, zi=np.array([rms_hz * rng.standard_normal()]))[0]
    cycles = np.cumsum(f) / FS
    y = np.real(sig.hilbert(x) * np.exp(2j * np.pi * dsp.wrap_cycles(cycles)))
    return y, f


# --- prototype 1: a widened acquisition ------------------------------------

def acquire_wide(z, threshold=PREAMBLE_THRESHOLD, max_bins=2, detect_cutoff=850.0):
    """sync.acquire() with the integer-bin count and the sync filter's
    cutoff exposed. Everything else is the reference, deliberately."""
    if len(z) < PREAMBLE_SAMPLES + 2 * M:
        raise sync.SyncError("signal too short")

    taps = NARROW if detect_cutoff == 850.0 else sig.firwin(129, detect_cutoff, fs=FS)
    zf = sig.fftconvolve(z, taps, mode="same")
    metric, a = sync._autocorr_metric(zf)
    n_star = int(np.argmax(metric))
    if metric[n_star] < threshold:
        raise sync.SyncError(f"no preamble found (peak metric {metric[n_star]:.2f})")

    f_frac = np.angle(a[n_star]) / (2 * np.pi * M / FS)
    template = ofdm.preamble_template()
    t_norm = np.sqrt(np.sum(np.abs(template) ** 2))
    lo = max(0, n_star - PREAMBLE_CP - 200)
    hi = min(len(z) - PREAMBLE_SAMPLES, n_star + 200)
    if hi <= lo:
        raise sync.SyncError("preamble at signal edge")
    seg = zf[lo : hi + PREAMBLE_SAMPLES]

    best = None
    for m_bin in range(-max_bins, max_bins + 1):
        f_cand = f_frac + m_bin * FS / M
        seg_c = dsp.freq_correct(seg, f_cand)
        corr = sig.fftconvolve(seg_c, np.conj(template[::-1]), mode="valid")
        peak = int(np.argmax(np.abs(corr)))
        energy = np.sqrt(np.sum(np.abs(seg_c[peak : peak + PREAMBLE_SAMPLES]) ** 2))
        score = np.abs(corr[peak]) / (t_norm * energy + 1e-12)
        if best is None or score > best[0]:
            best = (score, lo + peak, f_cand)

    _, p0, f_hat = best
    n_pre = PREAMBLE_REPEATS * M
    zc = dsp.freq_correct(z[p0 + PREAMBLE_CP : p0 + PREAMBLE_CP + n_pre], f_hat)
    if len(zc) == n_pre:
        d = np.sum(zc[M:] * np.conj(zc[:-M]))
        if np.abs(d) > 0:
            f_hat += np.angle(d) / (2 * np.pi * M / FS)
    return sync.Acquisition(p0, f_hat, float(metric[n_star]))


def acq_ok(a, true_offset):
    """Success = the header would have decoded: right place, right
    frequency. A lock that is 40 samples out is not a partial success."""
    return abs(a.preamble_start - TRUE_START) <= 4 and abs(a.freq_offset - true_offset) <= 2.0


# --- prototype 2: demod with residual-CFO (drift) tracking -----------------

def demodulate_tracked(m, x, track=True, alpha=0.1, beta=0.01, oracle_rate=None,
                       oracle_freq=None):
    """Modem.demodulate with an optional drift tracker.

    track=False must reproduce the reference exactly -- see --verify.
    oracle_rate de-chirps with a known true rate, and oracle_freq with a
    known per-sample frequency path (the general case, for drift that is
    not a ramp). Either is the upper bound on what any tracker could
    reach, since it is handed the answer.

    Second order (alpha on frequency, beta on rate) because drift is a
    *ramp*: a first-order loop leaves a steady-state lag proportional to
    rate/alpha, and that lag is the error being removed. The measurement
    is the residual that survived the correction already applied, so
    both terms integrate it rather than chasing it.
    """
    z = dsp.to_baseband(np.asarray(x, dtype=np.float64))
    acq = sync.acquire(z)
    z = dsp.freq_correct(z, acq.freq_offset)

    if oracle_rate:
        t = (np.arange(len(z)) - acq.preamble_start) / FS
        z = z * np.exp(-2j * np.pi * dsp.wrap_cycles(0.5 * oracle_rate * t**2))
    if oracle_freq is not None:
        # What is left to remove is the true path *minus what
        # acquisition already took out*, anchored at the preamble.
        # Removing the raw path double-corrects by acq.freq_offset --
        # invisible for a ramp starting at zero (the preamble sits at
        # t=0.1 s, so acq.freq_offset is ~0.01 Hz) and ruinous for
        # wander, where the offset at the preamble is a full standard
        # deviation. The tell was an "oracle" scoring below the tracker.
        resid = np.asarray(oracle_freq[: len(z)]) - acq.freq_offset
        cyc = np.cumsum(resid) / FS
        z = z * np.exp(-2j * np.pi * dsp.wrap_cycles(cyc - cyc[acq.preamble_start]))

    u0 = acq.preamble_start + PREAMBLE_CP
    h_pre = sum(
        ofdm.demod_window(z, u0 + r * M, DEMOD_BACKOFF) for r in range(PREAMBLE_REPEATS)
    ) / (PREAMBLE_REPEATS * m.pilot)

    h0 = acq.preamble_start + PREAMBLE_SAMPLES
    soft = np.zeros(NC)
    for s in range(2):
        y = ofdm.demod_window(z, h0 + s * NSYM + NCP, DEMOD_BACKOFF)
        soft += np.real(y * np.conj(h_pre))
    spec = framing.decode_header(soft)
    if spec is None:
        raise sync.SyncError("header decode failed")

    n_f = spec.n_frames
    raw = np.zeros((n_f, SYMS_PER_FRAME, NC), dtype=np.complex128)
    h_pilot = np.zeros((n_f, NC), dtype=np.complex128)
    received = np.zeros(n_f, dtype=bool)
    phi_ref = m._bin_phase_step(h_pre)
    pilot_powers = []
    tau_ema = 0.0
    f_est = 0.0      # residual CFO estimate, Hz
    r_est = 0.0      # drift rate estimate, Hz/s
    phase_acc = 0.0  # accumulated de-rotation, cycles
    nloc = np.arange(FRAME_SAMPLES)

    p = h0 + HEADER_SAMPLES
    for f in range(n_f):
        if p + FRAME_SAMPLES > len(z):
            break
        if track:
            # A *continuous* ramp across the frame, not one constant
            # phase per frame. A constant only removes the frame-to-frame
            # step, which the pilot EQ already removes, and leaves the
            # frequency error inside the frame -- which is the part that
            # costs the picture (pilot-to-data rotation, and ICI).
            zz = z[p : p + FRAME_SAMPLES] * np.exp(
                -2j * np.pi * dsp.wrap_cycles(phase_acc + f_est * nloc / FS))
            for s in range(SYMS_PER_FRAME):
                raw[f, s] = ofdm.demod_window(zz, s * NSYM + NCP, DEMOD_BACKOFF)
        else:
            for s in range(SYMS_PER_FRAME):
                raw[f, s] = ofdm.demod_window(z, p + s * NSYM + NCP, DEMOD_BACKOFF)
        h_pilot[f] = raw[f, 0] / m.pilot
        received[f] = True

        power = float(np.mean(np.abs(raw[f, 0]) ** 2))
        pilot_powers.append(power)
        healthy = power > 0.1 * np.median(pilot_powers)

        if track:
            if f > 0 and healthy and received[f - 1]:
                # Phase common to all carriers between consecutive pilots
                # is residual frequency; the slope across them is timing,
                # tracked separately below, and cancels out of this sum.
                d = np.sum(h_pilot[f] * np.conj(h_pilot[f - 1]))
                if np.abs(d) > 0:
                    err = np.angle(d) / (2 * np.pi * FRAME_S)  # Hz, +-3.47
                    f_est += alpha * err
                    r_est += beta * err / FRAME_S
            phase_acc += f_est * FRAME_S   # carry phase, then step frequency
            f_est += r_est * FRAME_S

        p += FRAME_SAMPLES
        if healthy:
            phi = m._bin_phase_step(h_pilot[f])
            d = np.angle(np.exp(1j * (phi - phi_ref)))
            tau = -d * FS / (2 * np.pi * RS)
            tau_ema += 0.02 * (tau - tau_ema)
            if abs(tau_ema) >= 2:
                step = int(np.clip(round(tau_ema), -2, 2))
                p += step
                tau_ema -= step

    latents = np.zeros(spec.n_tx_latents)
    weights = np.zeros(spec.n_tx_latents)
    med_h = np.median(np.abs(h_pilot[received])) if received.any() else 1.0
    floor = max(0.05 * med_h, 1e-9)

    def pilot_at(i, fallback):
        if 0 <= i < n_f and received[i]:
            return h_pilot[i]
        return h_pilot[fallback]

    beacon_soft = np.zeros(n_f * CHIPS_PER_FRAME)
    for f in range(n_f):
        if not received[f]:
            continue
        p0, p1 = pilot_at(f - 1, f), h_pilot[f]
        p2 = pilot_at(f + 1, f)
        p3 = pilot_at(f + 2, f + 1 if f + 1 < n_f and received[f + 1] else f)
        frame_slots = np.zeros(LATENTS_PER_FRAME)
        frame_w = np.zeros(LATENTS_PER_FRAME)
        for s in range(1, SYMS_PER_FRAME):
            u = s / SYMS_PER_FRAME
            h = 0.5 * (2 * p1 + (p2 - p0) * u + (2 * p0 - 5 * p1 + 4 * p2 - p3) * u**2
                       + (3 * p1 - p0 - 3 * p2 + p3) * u**3)
            mag = np.maximum(np.abs(h), floor)
            y = raw[f, s] * np.conj(h) / mag**2
            w = np.minimum(np.abs(h) / med_h, 1.0)
            i0 = (s - 1) * NC_LATENT * 2
            frame_slots[i0 : i0 + NC_LATENT * 2] = framing.symbols_to_slots(
                y[:NC_LATENT][None, :])
            frame_w[i0 : i0 + NC_LATENT * 2] = np.repeat(w[:NC_LATENT], 2)
            beacon_soft[f * CHIPS_PER_FRAME + (s - 1)] = np.real(y[BEACON_CARRIER])
        lo = f * LATENTS_PER_FRAME
        latents[lo : lo + LATENTS_PER_FRAME] = frame_slots
        weights[lo : lo + LATENTS_PER_FRAME] = frame_w

    latents = np.clip(latents, -10, 10)
    lat_full, _ = framing.deinterleave(latents, spec)
    w_full, _ = framing.deinterleave(weights, spec)
    br = beacon.decode(beacon_soft)
    return DemodResult(
        latents=lat_full, weights=w_full, mode=spec, freq_offset=acq.freq_offset,
        sync_metric=acq.metric, frames_received=int(received.sum()), beacon=br,
        callsign=br.callsign if br else "", preamble_start=acq.preamble_start,
        snr_db=_estimate_snr_db(h_pilot, received),
    )


# --- prototype 3: a cheaper blind search -----------------------------------

class FastBlindAccumulator:
    """sync.BlindAccumulator with the same result and less work.

    Two independent changes, and only one of them matters much:

    * **Batched, threaded IFFTs.** One `ifft(..., axis=1, workers=-1)`
      over a chunk of CFO bins rather than one call per bin, and the
      periodic fold by reshape-and-sum rather than `np.add.at` (the
      unbuffered `ufunc.at` path). Numerically identical -- the shift is
      still a circular shift of the block spectrum, expressed as a
      gather. Worth ~1.1x from batching and ~1.5x from the threads;
      the IFFTs really are the work, so there is no large win here.

    * **A coarser CFO grid, which is the actual win.** For a fixed lag,
      the matched filter as a function of CFO is a DTFT of a
      160-sample sequence, so |mf|**2 is exactly band-limited and
      determined by samples every FS/(2*160) = 25 Hz. The default
      1.7 Hz step oversamples that ~15x. Detection holds to a 12.5 Hz
      step at every SNR measured, and `refine_cfo` recovers a *better*
      frequency estimate than the 1.7 Hz grid's raw argmax.

    `bin_chunk` bounds peak memory: the shifted spectra are
    (chunk x block) complex, which at 737 bins in one go would be ~100 MB.
    """

    def __init__(self, max_offset_hz=55.0, bin_step_hz=1.7, min_periods=8,
                 threshold=4.0, window_s=25.0, bin_chunk=128, workers=-1):
        template = ofdm.pilot_template()
        kernel = np.conj(template[::-1])
        m = len(kernel)
        self._m = m
        self._min_periods = min_periods
        self._threshold = threshold
        self._workers = workers
        self._bin_chunk = bin_chunk

        self._block = next_fast_len(max(int(np.ceil(FS / bin_step_hz)), 4 * m))
        self._step = self._block - (m - 1)
        self._bin_hz = FS / self._block

        n_bins = int(np.ceil(max_offset_hz / bin_step_hz))
        shift_bins = np.array(
            [int(round(k * bin_step_hz / self._bin_hz)) for k in range(-n_bins, n_bins + 1)])
        self._shift_bins = shift_bins
        self._freqs = shift_bins * self._bin_hz
        B = self._block
        self._kernel_f = fft(np.concatenate([kernel, np.zeros(B - m, dtype=complex)]))
        # np.roll(v, -s)[j] == v[(j + s) % B], precomputed as a gather.
        self._take = ((np.arange(B)[None, :] + shift_bins[:, None]) % B).astype(np.int32)

        wl = list(window_s) if isinstance(window_s, (list, tuple)) else [window_s]
        self._decay_per_block = np.array(
            [1.0 if w is None else float(np.exp(-self._step / (w * FS))) for w in wl])
        self._folded = np.zeros((len(wl), len(shift_bins), FRAME_SAMPLES))
        self._n_valid = 0
        self._buf = np.zeros(0, dtype=complex)
        self._buf_start = None
        self._K = int(np.ceil(self._step / FRAME_SAMPLES))
        self._pad = self._K * FRAME_SAMPLES - self._step

    def push(self, z, start_sample):
        if self._buf_start is None:
            self._buf_start = start_sample
        elif start_sample != self._buf_start + self._buf.size:
            raise ValueError("non-contiguous push")
        buf = np.concatenate([self._buf, z]) if self._buf.size else np.asarray(z, dtype=complex)

        B, m, step, F = self._block, self._m, self._step, FRAME_SAMPLES
        n_bins = len(self._shift_bins)
        pos = 0
        while pos + B <= buf.size:
            block_f = fft(buf[pos : pos + B], workers=self._workers)
            phase0 = (self._buf_start + pos) % F
            local = np.empty((n_bins, F))
            for lo in range(0, n_bins, self._bin_chunk):
                hi = min(lo + self._bin_chunk, n_bins)
                mf = ifft(block_f[self._take[lo:hi]] * self._kernel_f, axis=1,
                          workers=self._workers)
                p2 = np.abs(mf[:, m - 1 : B]) ** 2
                if self._pad:
                    p2 = np.concatenate([p2, np.zeros((hi - lo, self._pad))], axis=1)
                local[lo:hi] = p2.reshape(hi - lo, self._K, F).sum(axis=1)
            self._folded *= self._decay_per_block[:, None, None]
            self._folded += np.roll(local, phase0, axis=1)[None, :, :]
            self._n_valid += step
            pos += step

        self._buf = buf[pos:]
        self._buf_start += pos

    def _scores(self):
        return self._folded.max(axis=2) / (np.median(self._folded, axis=2) + 1e-12)

    def result(self):
        if self._n_valid < FRAME_SAMPLES * self._min_periods:
            raise sync.SyncError("window too short for blind acquisition")
        sc = self._scores()
        t, i = (int(v) for v in np.unravel_index(np.argmax(sc), sc.shape))
        score = float(sc[t, i])
        if score < self._threshold:
            raise sync.SyncError(f"no periodic pilot found (peak prominence {score:.3g})")
        return sync.BlindAcquisition(int(np.argmax(self._folded[t, i])),
                                     float(self._freqs[i]), score)


def refine_cfo(acc):
    """Sub-bin CFO by fitting a parabola through the winning bin and its
    two neighbours. The folded score is band-limited in CFO (see
    FastBlindAccumulator), so the peak between samples is recoverable
    rather than lost -- which is what lets the grid be coarse."""
    sc = acc._scores()
    t, i = (int(v) for v in np.unravel_index(np.argmax(sc), sc.shape))
    raw = float(acc._freqs[i])
    if 0 < i < sc.shape[1] - 1:
        y0, y1, y2 = sc[t, i - 1], sc[t, i], sc[t, i + 1]
        denom = y0 - 2 * y1 + y2
        if denom != 0:
            delta = float(np.clip(0.5 * (y0 - y2) / denom, -1.0, 1.0))
            return raw + delta * (acc._freqs[i + 1] - acc._freqs[i])
    return raw


# --- measurements ----------------------------------------------------------

def verify():
    """The fork check. demodulate_tracked(track=False) is a copy of
    Modem.demodulate; if it has drifted, every drift number here is
    measuring the copy rather than the receiver."""
    spec, lat, x = make_tx("A")
    m = Modem()
    ok = True
    for fade in [None, "mpg", "mpp"]:
        y = hfchannel.apply_channel(x, snr_db=6.0, fading_preset=fade, seed=0)
        a = m.demodulate(y)
        b = demodulate_tracked(m, y, track=False)
        same = (np.array_equal(a.latents, b.latents)
                and np.array_equal(a.weights, b.weights)
                and a.frames_received == b.frames_received)
        ok &= same
        print(f"  fading={str(fade):5} frames {a.frames_received:4d}/{b.frames_received:<4d} "
              f"latents identical: {same}")
    print("VERIFY", "OK" if ok else "FAILED -- the fork has drifted from Modem.demodulate")
    return 0 if ok else 1


def reach(seeds=8, snr=0.0):
    print(f"reach vs max_bins (mode A, {snr:+.0f} dB, {seeds} seeds; "
          f"'#'={seeds}/{seeds}, '.'=0)")
    spec, lat, x = make_tx("A")
    offs = [0, 200, 400, 600, 700, 800, 900, 1000, 1200]
    print("  max_bins (reach) |" + "".join(f"{o:>6d}" for o in offs))
    for mb in [2, 6, 12, 16, 26]:
        row = f"  {mb:3d} (+-{25 + 50 * mb:5.0f} Hz) |"
        for off in offs:
            n = 0
            for s in range(seeds):
                y = hfchannel.apply_channel(x, snr_db=snr, freq_offset_hz=off, seed=s)
                try:
                    n += acq_ok(acquire_wide(dsp.to_baseband(y), max_bins=mb), off)
                except sync.SyncError:
                    pass
            row += f"{'#' if n == seeds else ('.' if n == 0 else str(n)):>6}"
        print(row)


def filter_wall(seeds=8, snr=0.0):
    print(f"\nthe wall past max_bins: sync filter cutoff (max_bins=26, {snr:+.0f} dB)")
    spec, lat, x = make_tx("A")
    offs = [600, 700, 800, 900, 1000, 1200]
    print("  cutoff |" + "".join(f"{o:>7d}" for o in offs))
    for cutoff in [850.0, 1200.0, 1600.0, 2200.0]:
        row = f"  {cutoff:6.0f} |"
        for off in offs:
            n = 0
            for s in range(seeds):
                y = hfchannel.apply_channel(x, snr_db=snr, freq_offset_hz=off, seed=s)
                try:
                    n += acq_ok(acquire_wide(dsp.to_baseband(y), max_bins=26,
                                             detect_cutoff=cutoff), off)
                except sync.SyncError:
                    pass
            row += f"{n:>4d}/{seeds:<2d}"
        print(row)


def sensitivity(seeds=25):
    print(f"\nsensitivity cost of a wider search (mode A, {seeds} seeds)")
    spec, lat, x = make_tx("A")
    snrs = [0.0, -1.0, -2.0, -3.0, -4.0]
    for off in [0.0, 300.0]:
        print(f"  offset {off:.0f} Hz -- max_bins |" + "".join(f"{s:>8.0f} dB" for s in snrs))
        for mb in [2, 12, 26]:
            row = f"  {'':>17}{mb:3d} |"
            for snr in snrs:
                n = 0
                for s in range(seeds):
                    y = hfchannel.apply_channel(x, snr_db=snr, freq_offset_hz=off, seed=s)
                    try:
                        n += acq_ok(acquire_wide(dsp.to_baseband(y), max_bins=mb), off)
                    except sync.SyncError:
                        pass
                row += f"  {n:3d}/{seeds:<4d}"
            print(row)

    print("\n  do the extra candidates ever win? (identical answer required)")
    print(f"  {'SNR':>6} {'trials':>7} {'differ':>7} {'both detected':>14}")
    for snr in [0.0, -3.0, -5.0, -7.0]:
        n = differ = both = 0
        for s in range(40):
            y = hfchannel.apply_channel(x, snr_db=snr, freq_offset_hz=0.0, seed=s)
            z = dsp.to_baseband(y)
            r = {}
            for mb in (2, 12):
                try:
                    a = acquire_wide(z, max_bins=mb)
                    r[mb] = (a.preamble_start, round(a.freq_offset, 6))
                except sync.SyncError:
                    r[mb] = None
            n += 1
            if r[2] is not None and r[12] is not None:
                both += 1
                differ += r[2] != r[12]
        print(f"  {snr:6.0f} {n:7d} {differ:7d} {both:14d}")


def cpu():
    print("\nCPU: where acquisition's time actually goes")
    spec, lat, x = make_tx("A")
    z32 = dsp.to_baseband(hfchannel.apply_channel(x, snr_db=6.0, freq_offset_hz=300.0))
    z130 = dsp.to_baseband(np.concatenate([
        hfchannel.apply_channel(x, snr_db=6.0, freq_offset_hz=300.0),
        np.random.default_rng(0).normal(scale=0.1, size=98 * FS)]))
    acquire_wide(z32, max_bins=2)  # warm the FFT plans; the first call is not a measurement
    for name, buf in [("32 s", z32), ("130 s (full ring)", z130)]:
        t0 = time.perf_counter()
        for _ in range(3):
            sync._autocorr_metric(dsp.sync_lowpass(buf))
        print(f"  {name:>18}  detection stage alone  {(time.perf_counter()-t0)/3*1000:7.1f} ms")
        for mb in [2, 12, 26]:
            t0 = time.perf_counter()
            for _ in range(3):
                try:
                    acquire_wide(buf, max_bins=mb)
                except sync.SyncError:
                    pass
            print(f"  {name:>18}  max_bins={mb:<3d}            "
                  f"{(time.perf_counter()-t0)/3*1000:7.1f} ms")


def blind_cost():
    print("\nthe blind path: CPU is linear in the range, and nothing else moves")
    spec, lat, x = make_tx("A")
    fr0 = LEADIN_SAMPLES + PREAMBLE_SAMPLES + HEADER_SAMPLES
    win = x[fr0 : fr0 + 20 * FS]
    z = dsp.to_baseband(hfchannel.apply_channel(win, snr_db=6.0, seed=0))
    print(f"  {'range':>9} {'bins':>6} {'push':>10} {'noise floor':>13} {'detects':>9}"
          "   (threshold 4.0)")
    for mo in [55.0, 125.0, 300.0, 625.0]:
        n_bins = 2 * int(np.ceil(mo / 1.7)) + 1
        acc = sync.BlindAccumulator(max_offset_hz=mo, window_s=25.0)
        t0 = time.perf_counter()
        acc.push(z, 0)
        t_push = time.perf_counter() - t0

        peaks = []
        for seed in range(4):
            noise = np.random.default_rng(500 + seed).normal(size=20 * FS)
            a = sync.BlindAccumulator(max_offset_hz=mo, threshold=0.0, window_s=None)
            a.push(dsp.to_baseband(noise), 0)
            peaks.append(a.result().metric)

        hits = 0
        for seed in range(4):
            y = hfchannel.apply_channel(win, snr_db=-3.0, seed=seed)
            a = sync.BlindAccumulator(max_offset_hz=mo, window_s=25.0)
            a.push(dsp.to_baseband(y), 0)
            try:
                hits += abs(a.result().freq_offset) <= 3.0
            except sync.SyncError:
                pass
        print(f"  +-{mo:6.0f} {n_bins:6d} {t_push*1000:8.0f} ms {np.mean(peaks):13.2f}"
              f" {hits:7d}/4   (at -3 dB)")


def _drift_row(mode, rate, snr, seeds, fading=None, sinus=None, alpha=0.1, beta=0.01):
    spec, lat, x = make_tx(mode)
    out = {k: [] for k in ("ref", "track", "oracle")}
    frames = []
    for seed in range(seeds):
        y = drift_shift(x, rate, sinus=sinus)
        if fading:
            y = hfchannel.fading(y, fading, seed=seed)
        if snr is not None:
            y = hfchannel.awgn(y, snr, seed=seed + 1)
        for name, kw in [("ref", dict(track=False)),
                         ("track", dict(track=True, alpha=alpha, beta=beta)),
                         ("oracle", dict(track=False, oracle_rate=rate))]:
            try:
                r = demodulate_tracked(Modem(), y, **kw)
                out[name].append(latent_snr_db(lat, r.latents, r.weights))
                if name == "ref":
                    frames.append(r.frames_received)
            except sync.SyncError:
                out[name].append(float("nan"))
    return out, (np.mean(frames) if frames else 0)


def drift(mode="A", snr=6.0, fading=None, rates=None, seeds=3, alpha=0.1, beta=0.01):
    spec = config.MODES[mode]
    rates = rates if rates is not None else [0, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
    print(f"\ndrift, mode {mode} ({spec.duration_s:.0f} s), snr={snr}, fading={fading}, "
          f"alpha={alpha}, beta={beta}")
    print(f"  {'Hz/s':>7} {'total Hz':>9} {'frames':>7} |{'ref':>10} |{'track':>10} "
          f"|{'oracle':>10}")
    for rate in rates:
        out, fr = _drift_row(mode, rate, snr, seeds, fading=fading, alpha=alpha, beta=beta)
        row = f"  {rate:7.3f} {rate*spec.duration_s:9.1f} {fr:7.0f} |"
        for k in ("ref", "track", "oracle"):
            row += f"{np.nanmean(out[k]):8.2f} dB|"
        print(row)


def drift_acquisition(seeds=6):
    """Drift is a demod problem, not an acquisition one -- both
    acquisition paths are shown here to be untouched by it."""
    spec, lat, x = make_tx("A")
    print("\npreamble detection vs drift rate (mode A, 6 dB)")
    print(f"  {'Hz/s':>7} {'metric':>8} {'locked':>9} {'CFO estimate Hz':>17}")
    for rate in [0.0, 0.1, 1.0, 5.0, 20.0]:
        mets, oks, fs = [], 0, []
        for seed in range(seeds):
            y = hfchannel.awgn(drift_shift(x, rate), 6.0, seed=seed + 1)
            try:
                a = sync.acquire(dsp.to_baseband(y))
            except sync.SyncError:
                continue
            mets.append(a.metric)
            oks += abs(a.preamble_start - TRUE_START) <= 4
            fs.append(a.freq_offset)
        print(f"  {rate:7.3f} {np.mean(mets):8.3f} {oks:6d}/{seeds:<2d} {np.mean(fs):17.3f}")

    fr0 = LEADIN_SAMPLES + PREAMBLE_SAMPLES + HEADER_SAMPLES
    win = x[fr0 : fr0 + 20 * FS]
    print("\nblind acquisition vs drift rate (mode A frames, 20 s, 6 dB)")
    print(f"  {'Hz/s':>7} {'Hz over 20 s':>13} {'score':>8} {'detects':>9}"
          "   (threshold 4.0)")
    for rate in [0.0, 0.1, 0.5, 1.0]:
        scores, oks = [], 0
        for seed in range(4):
            y = hfchannel.awgn(drift_shift(win, rate), 6.0, seed=seed + 1)
            acc = sync.BlindAccumulator(window_s=25.0, threshold=0.0)
            acc.push(dsp.to_baseband(y), 0)
            r = acc.result()
            scores.append(r.metric)
            oks += r.metric >= 4.0
        print(f"  {rate:7.3f} {rate*20:13.1f} {np.mean(scores):8.2f} {oks:7d}/4")


def drift_fading(seeds=6):
    """The loop's gains are bounded from both sides: too fast and it
    chases fading, too slow and it cannot follow a ramp."""
    print("\nloop bandwidth against Doppler spread (mode A, 6 dB)")
    for fade in ["mpd", "mpp"]:
        print(f"  {fade}: {'alpha, beta':>14} |{'0 Hz/s':>10} |{'1 Hz/s':>10}")
        for alpha, beta in [(None, None), (0.3, 0.05), (0.1, 0.01), (0.03, 0.002)]:
            cells = []
            for rate in [0.0, 1.0]:
                out, _ = _drift_row("A", rate, 6.0, seeds, fading=fade,
                                    alpha=alpha or 0.1, beta=beta or 0.01)
                cells.append(np.nanmean(out["ref" if alpha is None else "track"]))
            label = "no loop (ref)" if alpha is None else f"{alpha}, {beta}"
            print(f"  {'':>6}{label:>14} |{cells[0]:8.2f} dB|{cells[1]:8.2f} dB")


def blind_speed(seeds=12):
    """How much of the wide blind search's cost is actually necessary."""
    spec, lat, x = make_tx("A")
    fr0 = LEADIN_SAMPLES + PREAMBLE_SAMPLES + HEADER_SAMPLES
    win = x[fr0 : fr0 + 20 * FS]

    def z_at(snr, off, seed):
        return dsp.to_baseband(hfchannel.apply_channel(
            win, snr_db=snr, freq_offset_hz=off, seed=seed))

    z = z_at(6.0, 0.0, 0)

    print("(a) the batched form is the same computation, not an approximation")
    for mo in [55.0, 300.0]:
        a = sync.BlindAccumulator(max_offset_hz=mo, window_s=25.0)
        b = FastBlindAccumulator(max_offset_hz=mo, window_s=25.0)
        a.push(z, 0)
        b.push(z, 0)
        rel = np.max(np.abs(a._folded - b._folded)) / np.max(np.abs(a._folded))
        ra, rb = a.result(), b.result()
        print(f"  +-{mo:5.0f} Hz  max rel diff {rel:.1e}  same phase {ra.frame_start == rb.frame_start}"
              f"  same bin {ra.freq_offset == rb.freq_offset}")

    print("\n(b) one push over a 20 s window, best of 3")
    print(f"  {'configuration':>28} {'bins':>6} {'push':>9}")
    for label, mo, sh, kind in [
        ("reference   +-55 Hz @1.7", 55.0, 1.7, "ref"),
        ("batched     +-55 Hz @1.7", 55.0, 1.7, "w1"),
        ("batched+MT  +-55 Hz @1.7", 55.0, 1.7, "fast"),
        ("batched+MT  +-55 Hz @12.5", 55.0, 12.5, "fast"),
        ("batched+MT +-625 Hz @12.5", 625.0, 12.5, "fast"),
        ("batched+MT +-625 Hz @25", 625.0, 25.0, "fast"),
        ("reference  +-625 Hz @1.7", 625.0, 1.7, "ref"),
    ]:
        best = np.inf
        for _ in range(3):
            if kind == "ref":
                acc = sync.BlindAccumulator(max_offset_hz=mo, bin_step_hz=sh, window_s=25.0)
            else:
                acc = FastBlindAccumulator(max_offset_hz=mo, bin_step_hz=sh, window_s=25.0,
                                           workers=1 if kind == "w1" else -1)
            t0 = time.perf_counter()
            acc.push(z, 0)
            best = min(best, time.perf_counter() - t0)
        print(f"  {label:>28} {2*int(np.ceil(mo/sh))+1:6d} {best*1000:7.0f} ms")

    print("\n(c) does a coarser grid cost detection? (+-55 Hz, offset drawn")
    print("    uniformly so scalloping between bins is sampled fairly)")
    snrs = [-3.0, -6.0, -8.0, -10.0]
    print(f"  {'bin step':>9} |" + "".join(f"{s:>8.0f} dB" for s in snrs))
    for sh in [1.7, 6.8, 12.5, 25.0, 50.0]:
        row = f"  {sh:8.1f} |"
        for snr in snrs:
            hits = 0
            for seed in range(seeds):
                off = float(np.random.default_rng(900 + seed).uniform(-40, 40))
                acc = FastBlindAccumulator(max_offset_hz=55.0, bin_step_hz=sh,
                                           window_s=25.0, threshold=0.0)
                acc.push(z_at(snr, off, seed), 0)
                hits += acc.result().metric >= 4.0
            row += f"  {hits:3d}/{seeds:<3d} "
        print(row)

    print("\n(d) full +-625 Hz range at a 12.5 Hz step, signal placed anywhere in it")
    print(f"  {'SNR':>5} {'detects':>10} {'raw err Hz':>11} {'interp err':>11} {'interp worst':>13}")
    for snr in [6.0, 0.0, -3.0, -6.0, -8.0]:
        hits, raws, ints = 0, [], []
        for seed in range(seeds):
            off = float(np.random.default_rng(300 + seed).uniform(-600, 600))
            acc = FastBlindAccumulator(max_offset_hz=625.0, bin_step_hz=12.5,
                                       window_s=25.0, threshold=0.0)
            acc.push(z_at(snr, off, seed), 0)
            r = acc.result()
            hits += r.metric >= 4.0
            raws.append(abs(r.freq_offset - off))
            ints.append(abs(refine_cfo(acc) - off))
        print(f"  {snr:5.0f} {hits:8d}/{seeds:<3d} {np.mean(raws):11.2f} {np.mean(ints):11.2f}"
              f" {np.max(ints):13.2f}")

    print("\n(e) noise-only score -- more cells taken a max over (threshold 4.0)")
    for mo, sh in [(55.0, 1.7), (625.0, 12.5), (625.0, 25.0)]:
        peaks = []
        for seed in range(6):
            noise = np.random.default_rng(500 + seed).normal(size=20 * FS)
            a = FastBlindAccumulator(max_offset_hz=mo, bin_step_hz=sh, threshold=0.0,
                                     window_s=None)
            a.push(dsp.to_baseband(noise), 0)
            peaks.append(a.result().metric)
        print(f"  +-{mo:5.0f} Hz @{sh:5.1f}  {2*int(np.ceil(mo/sh))+1:4d} bins   "
              f"mean {np.mean(peaks):.2f}  worst {np.max(peaks):.2f}")


def drift_ou(seeds=6):
    """Drift as an Ornstein-Uhlenbeck process: mean-reverting, no
    characteristic phase, parameterized by stationary RMS and
    correlation time. The oracle here removes the *true* frequency path
    rather than a fitted ramp, so it is the real ceiling."""
    spec, lat, x = make_tx("A")
    print("\nOrnstein-Uhlenbeck frequency wander (mode A, 6 dB)")
    print(f"  {'RMS Hz':>7} {'tau s':>6} {'peak |f|':>9} {'RMS Hz/s':>9} |"
          f"{'ref':>10} |{'track .1':>10} |{'track .3':>10} |{'oracle':>10}")
    for rms_hz in [0.5, 1.0, 2.0, 5.0]:
        for tau in [30.0, 10.0, 3.0]:
            cells = {k: [] for k in ("ref", "t1", "t3", "oracle")}
            peaks, rates = [], []
            for seed in range(seeds):
                y, f = ou_shift(x, rms_hz, tau, seed=seed)
                y = hfchannel.awgn(y, 6.0, seed=seed + 1)
                peaks.append(np.max(np.abs(f)))
                # the rate the loop actually faces, measured over the
                # pilot spacing rather than per sample
                rates.append(np.std(np.diff(f[::FRAME_SAMPLES])) / FRAME_S)
                for k, kw in [("ref", dict(track=False)),
                              ("t1", dict(track=True, alpha=0.1, beta=0.01)),
                              ("t3", dict(track=True, alpha=0.3, beta=0.05)),
                              ("oracle", dict(track=False, oracle_freq=f))]:
                    try:
                        r = demodulate_tracked(Modem(), y, **kw)
                        cells[k].append(latent_snr_db(lat, r.latents, r.weights))
                    except sync.SyncError:
                        cells[k].append(float("nan"))
            print(f"  {rms_hz:7.1f} {tau:6.0f} {np.mean(peaks):9.2f} {np.mean(rates):9.3f} |"
                  + "|".join(f"{np.nanmean(cells[k]):8.2f} dB"
                             for k in ("ref", "t1", "t3", "oracle")))


def wander(seeds=4):
    print("\nsinusoidal wander (mode A, 6 dB) -- Doppler looks more like this than a ramp")
    print(f"  {'amp Hz':>7} {'period s':>9} {'peak Hz/s':>10} |{'ref':>10} |{'track':>10}")
    for amp, per in [(0.5, 30), (1.0, 30), (2.0, 30), (2.0, 10), (5.0, 30), (5.0, 10)]:
        out, _ = _drift_row("A", 0.0, 6.0, seeds, sinus=(amp, per))
        print(f"  {amp:7.1f} {per:9.0f} {2*np.pi*amp/per:10.2f} |"
              f"{np.nanmean(out['ref']):8.2f} dB|{np.nanmean(out['track']):8.2f} dB")


STEPS = {
    "reach": reach,
    "filter-wall": filter_wall,
    "sensitivity": sensitivity,
    "cpu": cpu,
    "blind-cost": blind_cost,
    "blind-speed": blind_speed,
    "drift": lambda: (drift("A", 6.0), drift("C", 6.0, rates=[0, 0.01, 0.02, 0.05, 0.1, 1.0],
                                             seeds=2)),
    "drift-acquisition": drift_acquisition,
    "drift-fading": drift_fading,
    "drift-ou": drift_ou,
    "wander": wander,
}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("steps", nargs="*", metavar="STEP",
                    help="one or more of: " + ", ".join(sorted(STEPS)))
    ap.add_argument("--verify", action="store_true",
                    help="check the forked demodulator still matches Modem.demodulate")
    args = ap.parse_args()
    for s in args.steps:
        if s not in STEPS:
            ap.error(f"unknown step {s!r}; choose from {', '.join(sorted(STEPS))}")
    if args.verify:
        rc = verify()
        if rc or not args.steps:
            return rc
    if not args.steps:
        ap.error("give at least one STEP, or --verify")
    for s in args.steps:
        STEPS[s]()
    return 0


if __name__ == "__main__":
    sys.exit(main())

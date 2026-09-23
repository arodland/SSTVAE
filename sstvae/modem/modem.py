"""Top-level modem: latent vector <-> passband audio samples.

TX layout:  silence | preamble | 2x header symbol | N frames | silence
Frame:      1 pilot symbol + 5 data symbols (230 real latents + 5 beacon
chips on the one carrier reserved for resync/callsign, see beacon.py).

RX equalizes each data symbol against per-carrier gains interpolated
between the surrounding frame pilots, tracks sample-clock drift from the
pilot phase slope across carriers, and reports per-latent confidence
weights (0 for frames that never arrived) so a decoder can treat missing
or faded latents as erasures.
"""

from dataclasses import dataclass

import numpy as np

from ..config import (
    DRIFT_TRACK_MODES,
    drift_gains,
    FS,
    RS,
    NC,
    NC_LATENT,
    BEACON_CARRIER,
    CHIPS_PER_FRAME,
    M,
    NCP,
    NSYM,
    SYMS_PER_FRAME,
    DATA_SYMS_PER_FRAME,
    FRAME_SAMPLES,
    FRAMES_PER_GROUP,
    LATENTS_PER_FRAME,
    PREAMBLE_CP,
    PREAMBLE_REPEATS,
    PREAMBLE_SAMPLES,
    SNR_REF_BW_HZ,
    HEADER_SAMPLES,
    LEADIN_SAMPLES,
    LEADOUT_SAMPLES,
    CLIP_HEADROOM_DB,
    DEMOD_BACKOFF,
    LATENT_GROUPS,
    MODES,
    MODES_BY_INDEX,
    ModeSpec,
)
from . import beacon, framing, ofdm
from .beacon import BeaconResult
from .dsp import to_baseband, freq_correct, tx_condition, wrap_cycles
from .sync import acquire, acquire_blind, BlindAcquisition, SyncError

__all__ = ["Modem", "DemodResult", "BlindDemodResult", "SyncError"]


@dataclass
class DemodResult:
    latents: np.ndarray  # canonical order, zeros where not received
    weights: np.ndarray  # per-latent confidence 0..1 (0 = erased)
    mode: ModeSpec
    freq_offset: float
    sync_metric: float
    frames_received: int
    beacon: BeaconResult | None = None  # decoded resync/callsign packet
    callsign: str = ""
    preamble_start: int = 0  # sample index (into the demodulated buffer) of the preamble
    snr_db: float = float("nan")  # pilot-based radio SNR estimate, see _estimate_snr_db


@dataclass
class BlindDemodResult:
    """Result of demodulate_blind: no preamble/header was needed.
    latents/weights are always sized for mode C's full canonical range
    (every mode is a prefix of it, so it is the one container that fits
    whatever the beacon turns out to say) and only populated where a
    demodulated frame actually landed, via the beacon's recovered
    absolute frame index. The transmission's real mode is
    `beacon.mode_index` (PROTOCOL_VERSION 4) — look it up via
    config.MODES_BY_INDEX, falling back to mode C's range when the
    index is one this receiver doesn't know."""

    latents: np.ndarray
    weights: np.ndarray
    freq_offset: float
    beacon: BeaconResult | None
    callsign: str
    frame_offset: int | None  # absolute index of this buffer's first frame
    n_frames: int  # local frames demodulated (may exceed what beacon covers)
    frame0_start: int | None = None  # sample index (into the demodulated
    # buffer) the transmitter's own absolute frame 0 would fall at --
    # only known once the beacon gives frame_offset; a stable identifier
    # for "which transmission is this" across repeated blind decodes of
    # a buffer that hasn't advanced past it yet. May be negative if the
    # buffer starts mid-transmission (frame 0 is then a virtual position
    # before the buffer). Note this is the start of *frame 0*, one
    # preamble+header later than a preamble-path DemodResult.preamble_start
    # for the same transmission -- convert before comparing the two.
    snr_db: float = float("nan")  # pilot-based radio SNR estimate, see _estimate_snr_db


def _estimate_snr_db(h_pilot: np.ndarray, received: np.ndarray | None = None) -> float:
    """Pilot-based radio SNR estimate, in the same "dB SNR in a
    `SNR_REF_BW_HZ` noise bandwidth" convention used elsewhere
    (hfchannel.awgn) -- so it's directly comparable to those numbers,
    not an ad-hoc scale.

    Treats the frame-to-frame difference of each carrier's pilot-derived
    channel gain as a noise proxy (real fading is assumed to move much
    more slowly than one frame; a fast fade will therefore read as
    extra "noise" and understate SNR a bit -- fine for a status
    display, not a calibration instrument). That gives a per-carrier
    SNR in a ~RS-wide (50 Hz) noise bandwidth (the DFT correlator's
    matched-filter bandwidth), which is then scaled to the reference
    bandwidth assuming roughly even power across the NC
    carriers (measured spread is well under 1 dB in practice -- see
    scripts/diagnose_carrier_power.py).
    """
    if received is not None:
        idx = np.flatnonzero(received)
    else:
        idx = np.arange(len(h_pilot))
    if len(idx) < 2:
        return float("nan")
    h = h_pilot[idx]
    adjacent = np.diff(idx) == 1
    if not adjacent.any():
        return float("nan")
    diffs = np.diff(h, axis=0)[adjacent]
    noise_var = 0.5 * float(np.mean(np.abs(diffs) ** 2))
    signal_var = float(np.mean(np.abs(h) ** 2))
    if noise_var <= 0:
        return float("inf")
    if signal_var <= 0:
        return float("-inf")
    snr_50hz_linear = signal_var / noise_var
    snr_ref_linear = snr_50hz_linear * (NC * RS / SNR_REF_BW_HZ)
    return 10 * np.log10(snr_ref_linear)


FRAME_S = FRAME_SAMPLES / FS
# The residual a pilot-rate estimator can measure without ambiguity:
# one pilot per frame is a 6.94 Hz sampling rate for the phase.
CFO_PULL_HZ = 1.0 / (2 * FRAME_S)


# --- Receive-side channel measurements over the whole transmission ------
_BB_FREQS = ofdm.BASEBAND_FREQS  # integer Hz, so phasors reduce exactly
# Delay grid for the support search, in samples of apparent delay past
# the demod window's start. Covers every delay the CP could hold and
# margin either side, since acquisition may have timed on a later path.
_DELAYS = np.arange(-2 * NCP, 2 * NCP + 1)
_STEER = ofdm._phasor(np.outer(_BB_FREQS, _DELAYS), -1)  # (NC, D)


def _time_shift_phase(shift) -> np.ndarray:
    """Per-carrier phasor undoing a demod-window move of `shift` whole
    samples (scalar or per-frame array): a window moved later by s
    multiplies carrier k by exp(+2j*pi*f_k*s/FS)."""
    s = np.asarray(shift, dtype=np.int64)
    return ofdm._phasor(np.multiply.outer(s, _BB_FREQS), -1)


def _shift_phasor(shift: np.ndarray) -> np.ndarray:
    """_time_shift_phase for fractional shifts (n, NC): not exact, but
    reduced before exp() as dsp.wrap_cycles requires."""
    return np.exp(-2j * np.pi * wrap_cycles(np.multiply.outer(shift, _BB_FREQS) / FS))


def _residual_cfo(h_pilot: np.ndarray, received: np.ndarray) -> float:
    """Hz: the pilots' common phase rotation frame to frame, summed over
    every adjacent received pair, so fading's random FM averages out
    instead of landing whole on the preamble's 80 ms estimate.
    Unambiguous within +-CFO_PULL_HZ."""
    both = received[1:] & received[:-1]
    d = np.sum(h_pilot[1:][both] * np.conj(h_pilot[:-1][both]))
    return float(np.angle(d) / (2 * np.pi * FRAME_S)) if np.abs(d) > 0 else 0.0


def _delay_support(h_pilot: np.ndarray, floor_db: float = -15.0) -> tuple[int, int]:
    """(first, last) apparent delay holding power within `floor_db` of
    the strongest, from the transmission-averaged power delay profile
    (matched filter, Hann-tapered across carriers to keep sidelobes out
    of the support). Paths are local maxima, since a lone path's
    mainlobe is ~10 samples wide and must not read as spread."""
    w = np.hanning(NC + 2)[1:-1]
    prof = np.mean(np.abs((h_pilot * w) @ np.conj(_STEER)) ** 2, axis=0)
    # Noise puts a flat floor under the whole profile, and most of the
    # grid is nothing but that floor, so its median is the floor's level.
    # At 0 dB the floor sits near -15 dB and its ripples read as paths
    # across the whole grid: placement then moved the window by up to 28
    # samples on mpd.
    thr = max(prof.max() * 10 ** (floor_db / 10), 2 * np.median(prof))
    peaks = [
        i for i in range(1, len(prof) - 1)
        if prof[i] >= thr and prof[i] >= prof[i - 1] and prof[i] >= prof[i + 1]
    ] or [int(np.argmax(prof))]
    return int(_DELAYS[peaks[0]]), int(_DELAYS[peaks[-1]])


def _window_shift(support: tuple[int, int]) -> int:
    """Samples to move the demod window later so every path in `support`
    lies inside the cyclic prefix, centred so drift has margin both ways.
    A path of apparent delay a is interference-free after a shift s when
    0 <= a - s <= NCP; the centre of what every path allows is
    s = (a_first + a_last - NCP) / 2. Past the CP no shift is clean and
    the centre is still the least bad."""
    return int(round((support[0] + support[1] - NCP) / 2))


# Time-interpolation window of the channel estimate: this many pilots
# either side of the frame.
LMMSE_TIME_TAPS = 4
# Doppler spread assumed when too few frame pairs exist to measure one.
LMMSE_DEFAULT_SPREAD_HZ = 2.0


# Pilot coherence with a neighbouring frame that marks a frame as the
# locked transmission. Measured in a 130 s ring: its frames 0.56-0.84
# at the 5th percentile (AWGN to mpd), noise frames 0.23 median and
# 0.46 at the 99th.
TX_COHERENCE = 0.5


def _transmission_frames(h_pilot: np.ndarray) -> np.ndarray:
    """Frames of a blind buffer that look like the locked transmission.

    Power cannot say it: another transmission still in the ring, read at
    this one's timing, is junk at whatever power it arrived with, and a
    stronger one then sets every statistic (the second of two blind
    receptions, 12 dB weaker than the first, was never delivered). So a
    frame must first be coherent with a neighbour -- the channel moves
    slowly and noise or a misaligned symbol does not -- and then within
    10 dB of the strongest such frame."""
    pw = np.mean(np.abs(h_pilot) ** 2, axis=1)
    ok = _coherent_frames(h_pilot)
    return ok & (pw > 0.1 * pw[ok].max())


def _coherent_frames(h_pilot: np.ndarray) -> np.ndarray:
    """Frames whose pilot correlates with a neighbour's above
    TX_COHERENCE; all of them if none does."""
    n = len(h_pilot)
    c = np.abs(np.sum(h_pilot[1:] * np.conj(h_pilot[:-1]), axis=1)) / np.sqrt(
        np.sum(np.abs(h_pilot[1:]) ** 2, axis=1) * np.sum(np.abs(h_pilot[:-1]) ** 2, axis=1)
        + 1e-30
    )
    coh = np.zeros(n)
    coh[1:] = c
    coh[:-1] = np.maximum(coh[:-1], c)
    ok = coh > TX_COHERENCE
    return ok if ok.any() else np.ones(n, dtype=bool)


def _lmmse_channel(
    h_pilot: np.ndarray, steps: np.ndarray | None = None,
    plausible: np.ndarray | None = None,
) -> np.ndarray:
    """(n, 5, NC) channel at every data symbol of n consecutive frames,
    from their pilots (n, NC). 2-D LMMSE in the robust form of Li,
    Cimini and Sollenberger (via Data2G's equalizer): across carriers a
    projection onto the measured delay support, in time Wiener
    interpolation over the nearest 2*LMMSE_TIME_TAPS pilots with a
    Gaussian (ITU-R F.1487) Doppler correlation whose spread is
    measured from the pilots and centred on their measured rotation.

    `plausible` is for a buffer that may be mostly not the transmission
    (the blind path; see `_transmission_frames`): the delay support and
    drift then come from those frames, and power and spread from the
    coherent frames whose pilot power is over twice the noise floor.
    The noise floor itself comes from every frame, since the projection
    residual measures it on signal and noise frames alike. Gating power
    and spread on the 10 dB rule alone dropped a third of a fading
    transmission's own frames and cost ~1 dB on a late-path channel.

    Two things the pilots do between frames that a textbook estimator
    does not expect, each measured as a loss before it was handled:

    * **Clock drift.** `steps` are the timing steps `_demod_frames`
      undid. Undone, the path delays drift with the clock across the
      whole transmission (~20 samples at 80 ppm over mode A), which
      smears the delay support and cost 3.4 dB there. So the
      projection runs on pilots with the drift taken out -- the steps,
      plus a line fitted through each frame's pilot phase slope for
      whatever the steps did not take (all of it on the blind path,
      which has no timing loop: 80 ppm cost it 4 dB) -- and the drift
      is put back after.
    * **Residual frequency.** The blind path carries 0.3 to 4 Hz of
      it, and a Doppler spectrum centred on zero averages rotating
      pilots away (-2 dB on a late-path channel). The spectrum is
      centred on the pilots' frame-to-frame rotation instead, which
      is what Catmull-Rom tracked implicitly. Like Catmull-Rom it
      cannot see a rotation past +-CFO_PULL_HZ.

    Replaced Catmull-Rom, which interpolated raw per-carrier pilots and
    so passed every pilot's noise straight into the equalizer: +0.21 to
    +0.56 dB PSNR through the v5 decoder, every image in all 12 cells
    measured (modes A and B, AWGN to mpd). The latent weights keep their
    |h|/median meaning on purpose -- the decoder was trained on it.

    The projector comes from eigh(B B^H), not an SVD of B: the same
    subspace (singular values above 1e-2 of the largest are
    eigenvalues above 1e-4), and a 24x24 Hermitian problem is what the
    C++ port solves too.
    """
    n = len(h_pilot)
    steps = np.zeros(n) if steps is None else steps
    pw = np.mean(np.abs(h_pilot) ** 2, axis=1)
    strong = plausible if plausible is not None else np.ones(n, dtype=bool)
    stepped = h_pilot * np.conj(_time_shift_phase(steps))
    tau = -np.angle(np.sum(stepped[:, 1:] * np.conj(stepped[:, :-1]), axis=1)) * FS / (2 * np.pi * RS)
    f_idx = np.arange(n, dtype=np.float64)
    wts = pw * strong
    drift = np.zeros(n)
    if np.count_nonzero(wts) >= 2:
        fm = np.sum(wts * f_idx) / np.sum(wts)
        var = np.sum(wts * (f_idx - fm) ** 2)
        if var > 0:
            rate = np.sum(wts * (f_idx - fm) * (tau - np.sum(wts * tau) / np.sum(wts))) / var
            drift = rate * (f_idx - fm)
    undo = _shift_phasor(steps + drift)
    aligned = h_pilot * np.conj(undo)
    d0, d1 = _delay_support(aligned[strong])
    B = ofdm._phasor(np.outer(_BB_FREQS, np.arange(d0 - 4, d1 + 5)), -1)
    lam, V = np.linalg.eigh(B @ B.conj().T)
    U = V[:, lam > lam[-1] * 1e-4]
    r = U.shape[1]
    hs = (aligned @ (U @ U.conj().T).T) * undo
    n0 = float(np.mean(np.abs(h_pilot - hs) ** 2)) * NC / max(NC - r, 1)
    n0_s = n0 * r / NC  # what is left on a projected pilot
    if plausible is None:
        stats = strong
    else:
        coherent = _coherent_frames(h_pilot)
        stats = (coherent & (pw > 2 * n0)) | strong
    p_sig = max(float(np.mean(np.abs(hs[stats]) ** 2)) - n0_s, 1e-12)
    pairs = stats[1:] & stats[:-1]
    lag1 = np.mean(hs[1:][pairs] * np.conj(hs[:-1][pairs])) if pairs.any() else 0.0
    rot = float(np.angle(lag1)) / (2 * np.pi)  # cycles per frame
    if pairs.sum() >= 8:
        rho = float(np.clip(np.abs(lag1) / p_sig, 1e-3, 0.9999))
        spread = float(np.clip(2 * np.sqrt(-np.log(rho) / 2) / (np.pi * FRAME_S), 0.02, 4.0))
    else:
        spread = LMMSE_DEFAULT_SPREAD_HZ

    def turn(t):  # the measured rotation at time t, in frames
        return np.exp(2j * np.pi * wrap_cycles(rot * np.asarray(t, dtype=np.float64)))

    def corr(dt):  # in frames
        return np.exp(-2 * (np.pi * spread / 2 * dt * FRAME_S) ** 2)

    k = min(2 * LMMSE_TIME_TAPS, n)
    offs = np.arange(1, SYMS_PER_FRAME) / SYMS_PER_FRAME
    tj = np.arange(k)
    Rpp = p_sig * corr(tj[:, None] - tj[None, :]) + n0_s * np.eye(k)
    hd = hs * np.conj(turn(np.arange(n)))[:, None]  # rotation removed
    h = np.zeros((n, SYMS_PER_FRAME - 1, NC), dtype=np.complex128)
    cache: dict[int, np.ndarray] = {}
    for f in range(n):
        lo = max(0, min(f - LMMSE_TIME_TAPS + 1, n - k))
        if f - lo not in cache:  # evenly spaced pilots: few distinct W
            Rdp = p_sig * corr((f - lo) + offs[:, None] - tj[None, :])
            cache[f - lo] = np.linalg.solve(Rpp, Rdp.T).T
        h[f] = (cache[f - lo] @ hd[lo : lo + k]) * turn(f + offs)[:, None]
    return h


class _DriftTracker:
    """Second-order loop on the pilots' *common* phase, which is residual
    carrier frequency. Off unless `drift_track` says otherwise, and when
    off it is not merely a no-op but is never consulted, so the default
    path is bit-identical to the receiver that had no tracker at all.

    The phase **common to all carriers** between consecutive pilots is
    frequency; the phase **slope across** carriers is timing, which
    `_bin_phase_step` already tracks. They are orthogonal -- a common
    rotation cancels out of the slope and a slope cancels out of the sum
    -- so the two loops do not fight.

    Three things here are load-bearing and each was wrong first:

    * **The correction is a continuous ramp within the frame, not one
      constant phase per frame.** A per-frame constant removes only the
      frame-to-frame step, which the pilot equalizer already removes,
      and leaves the frequency error *inside* the frame -- which is the
      part that costs the picture (pilot-to-data rotation and ICI). With
      a constant the tracker measurably hurt.
    * **The measurement is the residual that survived the correction
      already applied**, so both terms integrate it. Chasing it, EMA
      style (`f += a*(measured - f)`), is a loop measuring its own
      output.
    * **Second order.** Drift is a ramp, and a first-order loop leaves a
      steady-state lag proportional to rate/alpha -- exactly the error
      being removed. The second integrator takes that to zero.

    **Pull-in is +-CFO_PULL_HZ of residual, and outside it the loop does
    not merely fail to help.** One pilot per frame samples the phase at
    6.94 Hz, so a larger residual *aliases*: it is measured as a small
    error rather than a large one, and the loop confidently locks to the
    wrong frequency. That cannot be caught by testing the measurement's
    magnitude, because the magnitude is what the aliasing makes small.
    It does not arise on the preamble path -- acquisition leaves ~0.1 Hz
    and drift only grows from there, so the loop is already locked when
    the residual gets big -- but it does on the blind path, whose CFO
    estimate describes the middle of its window and so starts out around
    half the window's total drift away. See demodulate_blind.

    Beyond that, the drift *rate* the loop can follow is bounded by the
    same number over one frame; see docs/todo.md for the measured
    ceiling and for why alpha is a user-facing choice rather than a
    constant.
    """

    def __init__(self, alpha: float, beta: float):
        self.alpha = alpha
        self.beta = beta
        self.f_est = 0.0  # residual CFO estimate, Hz
        self.r_est = 0.0  # drift rate estimate, Hz/s
        self.phase_acc = 0.0  # accumulated de-rotation, cycles
        self._n = np.arange(FRAME_SAMPLES)

    def frame(self, z: np.ndarray, p: int) -> np.ndarray:
        """This frame's samples, de-rotated by the running estimate.
        Zero where the frame hangs off the buffer: a placed window can
        start a frame's CP before the buffer does, and the demod windows
        never read that part."""
        seg = z[max(p, 0) : p + FRAME_SAMPLES]
        seg = np.pad(seg, (max(-p, 0), FRAME_SAMPLES - len(seg) - max(-p, 0)))
        return seg * np.exp(
            -2j * np.pi * wrap_cycles(self.phase_acc + self.f_est * self._n / FS)
        )

    def update(self, h_cur: np.ndarray, h_prev: np.ndarray | None) -> None:
        if h_prev is not None:
            d = np.sum(h_cur * np.conj(h_prev))
            if np.abs(d) > 0:
                err = np.angle(d) / (2 * np.pi * FRAME_S)  # Hz, +-CFO_PULL_HZ
                self.f_est += self.alpha * err
                self.r_est += self.beta * err / FRAME_S
        # Carry absolute phase across the boundary *before* stepping the
        # frequency, so the correction stays continuous frame to frame.
        self.phase_acc += self.f_est * FRAME_S
        self.f_est += self.r_est * FRAME_S


def _make_tracker(drift_track: str) -> _DriftTracker | None:
    if drift_track not in DRIFT_TRACK_MODES:
        raise ValueError(
            f"drift_track must be one of {DRIFT_TRACK_MODES}, got {drift_track!r}"
        )
    alpha, beta = drift_gains(drift_track)
    return _DriftTracker(alpha, beta) if alpha else None


class Modem:
    def __init__(self):
        self.pilot = ofdm.pilot_sequence()

    # --- transmit ----------------------------------------------------------

    def modulate(
        self,
        latents: np.ndarray,
        mode: str | ModeSpec,
        normalize: bool = True,
        callsign: str = "",
    ) -> np.ndarray:
        """Latent vector -> unit-RMS float waveform at FS.

        The on-air contract is unit-RMS latents; `normalize` enforces it.
        `callsign` (up to 8 chars) rides the reserved beacon carrier along
        with a resync frame counter on every frame; leave blank to send
        just the resync counter.
        """
        spec = MODES[mode] if isinstance(mode, str) else mode
        latents = np.asarray(latents, dtype=np.float64)
        if latents.shape != (spec.n_latents,):
            raise ValueError(
                f"mode {spec.name} needs {spec.n_latents} latents, got {latents.shape}"
            )
        if normalize:
            rms = np.sqrt(np.mean(latents**2))
            if rms > 0:
                latents = latents / rms

        slots = framing.interleave(latents, spec)
        n_f = spec.n_frames
        beacon_chips = beacon.chip_stream(0, n_f, callsign, spec.index)
        symbols = np.empty((n_f * SYMS_PER_FRAME, NC), dtype=np.complex128)
        for f in range(n_f):
            sl = slots[f * LATENTS_PER_FRAME : (f + 1) * LATENTS_PER_FRAME]
            symbols[f * SYMS_PER_FRAME] = self.pilot
            data_syms = np.empty((DATA_SYMS_PER_FRAME, NC), dtype=np.complex128)
            data_syms[:, :NC_LATENT] = framing.slots_to_symbols(sl)
            data_syms[:, BEACON_CARRIER] = beacon_chips[
                f * CHIPS_PER_FRAME : (f + 1) * CHIPS_PER_FRAME
            ]
            symbols[f * SYMS_PER_FRAME + 1 : (f + 1) * SYMS_PER_FRAME] = data_syms

        hdr = framing.header_symbol(spec)
        x = np.concatenate(
            [
                np.zeros(LEADIN_SAMPLES),
                ofdm.preamble_waveform(),
                ofdm.modulate_symbols(np.stack([hdr, hdr])),
                ofdm.modulate_symbols(symbols),
                np.zeros(LEADOUT_SAMPLES),
            ]
        )
        return tx_condition(x, CLIP_HEADROOM_DB)

    # --- receive -----------------------------------------------------------

    def demodulate(
        self, x: np.ndarray, search_s: tuple[float, float] | None = None,
        drift_track: str = "off",
    ) -> DemodResult:
        """`search_s` restricts preamble acquisition to a time window
        (seconds); frames are still demodulated past its end.

        `drift_track` ("off" | "slow" | "fast") follows a carrier that
        moves during the transmission -- see `_DriftTracker`. Off by
        default because acquisition already removes a static offset and
        the remaining budget (~+-2 Hz of residual) is not usually
        threatened on HF by a modern radio; the settings exist for the
        cases where it is."""
        z = to_baseband(np.asarray(x, dtype=np.float64))
        search = None
        if search_s is not None:
            search = (int(search_s[0] * FS), int(search_s[1] * FS))
        acq = acquire(z, search=search)
        z = freq_correct(z, acq.freq_offset)

        # Channel reference from the preamble, averaged over every
        # repeat. Backing DEMOD_BACKOFF samples into the *previous*
        # repeat is safe for the same reason it is safe into the CP:
        # the block is periodic with M throughout.
        u0 = acq.preamble_start + PREAMBLE_CP
        h_pre = sum(
            ofdm.demod_window(z, u0 + r * M, DEMOD_BACKOFF)
            for r in range(PREAMBLE_REPEATS)
        ) / (PREAMBLE_REPEATS * self.pilot)

        # Header: two identical BPSK symbols, soft-combined.
        h0 = acq.preamble_start + PREAMBLE_SAMPLES
        # Matched-filter combining: faded carriers contribute little
        # instead of amplifying noise as zero-forcing would.
        soft = np.zeros(NC)
        for s in range(2):
            y = ofdm.demod_window(z, h0 + s * NSYM + NCP, DEMOD_BACKOFF)
            soft += np.real(y * np.conj(h_pre))
        spec = framing.decode_header(soft)
        if spec is None:
            raise SyncError("header decode failed")

        # Demodulate frames, tracking sample-clock drift via the phase
        # slope of the pilot across carriers (relative to the preamble).
        n_f = spec.n_frames
        phi_ref = self._bin_phase_step(h_pre)
        p_frames = h0 + HEADER_SAMPLES
        # Pass 1 at acquisition timing only measures what the whole
        # transmission says about residual frequency (measured on mpd:
        # the preamble's estimate reaches 2.1 Hz off, the pilots' 0.17)
        # and about delay spread, from which the window is placed so
        # every path sits inside the CP (+1.1 dB latent SNR on mpd).
        _, hp, rcv, _ = self._demod_frames(z, p_frames, n_f, phi_ref, 0, None, False)
        cfo_res = _residual_cfo(hp, rcv)
        z = freq_correct(z, cfo_res)
        _, hp, rcv, _ = self._demod_frames(z, p_frames, n_f, phi_ref, 0, None, False)
        shift = _window_shift(_delay_support(hp[rcv])) if rcv.any() else 0
        raw, h_pilot, received, steps = self._demod_frames(
            z, p_frames, n_f, phi_ref, shift, _make_tracker(drift_track)
        )
        # Equalize data symbols with pilots interpolated across the frame.
        latents = np.zeros(spec.n_tx_latents)
        weights = np.zeros(spec.n_tx_latents)
        med_h = np.median(np.abs(h_pilot[received])) if received.any() else 1.0
        floor = max(0.05 * med_h, 1e-9)
        # Frames present are a prefix: the buffer can only run out.
        n_rx = int(received.sum())
        h_est = np.zeros((n_f, SYMS_PER_FRAME - 1, NC), dtype=np.complex128)
        if n_rx:
            h_est[:n_rx] = _lmmse_channel(h_pilot[:n_rx], steps[:n_rx])

        beacon_soft = np.zeros(n_f * CHIPS_PER_FRAME)
        for f in range(n_f):
            if not received[f]:
                continue
            frame_slots = np.zeros(LATENTS_PER_FRAME)
            frame_w = np.zeros(LATENTS_PER_FRAME)
            for s in range(1, SYMS_PER_FRAME):
                h = h_est[f, s - 1]
                mag = np.maximum(np.abs(h), floor)
                y = raw[f, s] * np.conj(h) / mag**2
                w = np.minimum(np.abs(h) / med_h, 1.0)
                i0 = (s - 1) * NC_LATENT * 2
                sl = framing.symbols_to_slots(y[:NC_LATENT][None, :])
                frame_slots[i0 : i0 + NC_LATENT * 2] = sl
                frame_w[i0 : i0 + NC_LATENT * 2] = np.repeat(w[:NC_LATENT], 2)
                # Maximal-ratio, not the equalized value: `y` divides by
                # the channel estimate, so a beacon carrier in a fade
                # null returns amplified noise with a large magnitude,
                # and Golay's soft ML decode reads magnitude as
                # confidence -- one nulled chip then outvotes the four
                # good ones in the same codeword. Weighting by |h|^2
                # (equivalently, skipping the equalization) is the
                # correct soft metric for BPSK and needs no `floor`.
                beacon_soft[f * CHIPS_PER_FRAME + (s - 1)] = np.real(
                    raw[f, s, BEACON_CARRIER] * np.conj(h[BEACON_CARRIER])
                )
            lo = f * LATENTS_PER_FRAME
            latents[lo : lo + LATENTS_PER_FRAME] = frame_slots
            weights[lo : lo + LATENTS_PER_FRAME] = frame_w

        latents = np.clip(latents, -10, 10)
        latents_full, _ = framing.deinterleave(latents, spec)
        weights_full, _ = framing.deinterleave(weights, spec)
        beacon_result = beacon.decode(beacon_soft)
        return DemodResult(
            latents=latents_full,
            weights=weights_full,
            mode=spec,
            freq_offset=acq.freq_offset + cfo_res,
            sync_metric=acq.metric,
            frames_received=int(received.sum()),
            beacon=beacon_result,
            callsign=beacon_result.callsign if beacon_result else "",
            preamble_start=acq.preamble_start,
            snr_db=_estimate_snr_db(h_pilot, received),
        )

    def demodulate_blind(
        self, x: np.ndarray, search_s: tuple[float, float] | None = None,
        acquisition: BlindAcquisition | None = None,
        drift_track: str = "off",
    ) -> BlindDemodResult:
        """Recover frame timing purely from the pilot's own periodicity
        (sync.acquire_blind) — no preamble or header needed, so this
        works on a recording that starts mid-transmission. Once the
        beacon carrier's superframe decodes, every demodulated frame's
        absolute index is known, which places its latents in the right
        canonical (group-aware) slot without ever having seen the
        header, and reconstructs where the transmission's frame 0 fell
        in sample time — the "retrospective decode" case.

        No sample-clock drift tracking (that needs a preamble-phase
        reference); fine for the bounded windows this is meant for.
        `drift_track` is the *carrier* drift loop, which needs no such
        reference -- it works off the pilots, which this path has. It has
        one limit here that it does not have on the preamble path, and
        it is sharp rather than gradual: `acquire_blind` estimates one
        frequency for its whole window, so on a drifting signal that
        estimate describes the *middle* of the window and the residual
        at the first frame is about half the window's total drift. Once
        that exceeds `CFO_PULL_HZ` the per-frame measurement aliases and
        the loop locks to the wrong frequency -- measured, at 0.5 Hz/s
        over a 30 s window (7.7 Hz of initial residual) it takes the
        beacon down, where leaving it off decodes. Roughly: helpful
        while the total drift across the window stays under ~7 Hz,
        harmful past it. Fixing that properly means anchoring the loop
        at the window's middle and running it outward in both
        directions, which is not implemented.

        `acquisition`, if given, skips the internal acquire_blind call
        and demodulates at that position instead -- for a caller (e.g.
        rx/engine.py) that already found it via a persistent
        sync.BlindAccumulator rather than a fresh bounded-window search.
        The rest of this method is unaffected: it still demodulates
        every frame the *whole* of `x` can hold, using `acquisition`
        only to place frame 0.
        """
        z = to_baseband(np.asarray(x, dtype=np.float64))
        if acquisition is not None:
            ba = acquisition
        else:
            search = None
            if search_s is not None:
                search = (int(search_s[0] * FS), int(search_s[1] * FS))
            ba = acquire_blind(z, search=search)
        z = freq_correct(z, ba.freq_offset)

        p0 = ba.frame_start - NCP  # CP-start of local frame 0
        L_lo = int(np.ceil(-p0 / FRAME_SAMPLES))
        L_hi = int(np.floor((len(z) - FRAME_SAMPLES - p0) / FRAME_SAMPLES))
        if L_lo > L_hi:
            raise SyncError("blind lock too close to buffer edge to demod any full frame")
        n_f = L_hi - L_lo + 1
        p_start = p0 + L_lo * FRAME_SAMPLES

        def frames(p: int) -> tuple[np.ndarray, np.ndarray]:
            raw = np.zeros((n_f, SYMS_PER_FRAME, NC), dtype=np.complex128)
            h_pilot = np.zeros((n_f, NC), dtype=np.complex128)
            tracker = _make_tracker(drift_track)
            pilot_powers: list[float] = []
            for f in range(n_f):
                if tracker is None:
                    for s in range(SYMS_PER_FRAME):
                        raw[f, s] = ofdm.demod_window(z, p + s * NSYM + NCP, DEMOD_BACKOFF)
                else:
                    zz = tracker.frame(z, p)
                    for s in range(SYMS_PER_FRAME):
                        raw[f, s] = ofdm.demod_window(zz, s * NSYM + NCP, DEMOD_BACKOFF)
                h_pilot[f] = raw[f, 0] / self.pilot
                if tracker is not None:
                    # Most of this range is usually not the transmission at
                    # all (silence or noise before it starts, or accumulating
                    # after it ends -- see the med_h comment below), so the
                    # loop must not integrate phase out of noise frames. Same
                    # health test the preamble path uses.
                    power = float(np.mean(np.abs(raw[f, 0]) ** 2))
                    pilot_powers.append(power)
                    healthy = power > 0.1 * np.median(pilot_powers)
                    tracker.update(h_pilot[f], h_pilot[f - 1] if (f > 0 and healthy) else None)
                p += FRAME_SAMPLES
            return raw, h_pilot

        raw, h_pilot = frames(p_start)
        if np.any(h_pilot):
            # Same placement as the preamble path, from the frames that
            # are plausibly the transmission rather than the noise
            # around it. A frame the moved window would take past either
            # end of the buffer is dropped rather than the placement
            # skipped: blind acquisition times on the stronger path, so
            # a lock that needs moving earlier is the common case, and
            # skipping it within a few samples of the buffer start cost
            # a late-path channel a dB. Moving p_start by whole frames
            # leaves frame0_start where it was, since the beacon's frame
            # offset moves with it.
            shift = _window_shift(_delay_support(h_pilot[_transmission_frames(h_pilot)]))
            lo, n = p_start, n_f
            if lo + shift + NCP - DEMOD_BACKOFF < 0:  # its first demod window
                lo, n = lo + FRAME_SAMPLES, n - 1
            if lo + shift + n * FRAME_SAMPLES > len(z):
                n -= 1
            if shift and n > 0:
                p_start, n_f = lo, n
                raw, h_pilot = frames(p_start + shift)

        # Blind demod always covers every frame the *whole current
        # buffer* can hold, since the transmission's true length is
        # unknown until the beacon resolves it -- unlike the preamble
        # path (demodulate() above), which restricts this same
        # computation to the header's known real frame count. Most of
        # that range is often not the real transmission at all (silence
        # or noise before it starts, or accumulating after it ends,
        # while the loop waits to see whether a longer mode is still
        # arriving) -- a straight median over the *whole* range
        # describes "typical", which is the noise floor whenever noise
        # frames are the numerical majority, and noise then reads as
        # fully trustworthy (weight ~1) right alongside real frames
        # instead of being down-weighted, feeding reconstruct() latents
        # that are mostly garbage at full confidence. Anchoring instead
        # on frames within an order of magnitude of the strongest ones
        # seen needs only a few genuinely real frames to set the right
        # reference, regardless of how much silence surrounds them; a
        # real (even faded) frame is never excluded by this on its own
        # account, since a *minority* of low-|h| frames barely moves a
        # median in the first place.
        h_mag = np.abs(h_pilot)
        peak_h = np.max(h_mag) if h_mag.size else 0.0
        plausible = h_mag > 0.1 * peak_h
        med_h = np.median(h_mag[plausible]) if np.any(plausible) else 1.0
        floor = max(0.05 * med_h, 1e-9)

        # Channel statistics from the frames that are plausibly the
        # transmission, for the same reason as med_h above; the estimate
        # still covers every frame, and noise frames get what their
        # small |h| earns them in the weights.
        h_est = (
            _lmmse_channel(h_pilot, plausible=_transmission_frames(h_pilot))
            if np.any(h_pilot)
            else np.zeros((n_f, SYMS_PER_FRAME - 1, NC), dtype=np.complex128)
        )

        beacon_soft = np.zeros(n_f * CHIPS_PER_FRAME)
        slot_values = np.zeros((n_f, LATENTS_PER_FRAME))
        slot_weights = np.zeros((n_f, LATENTS_PER_FRAME))
        for f in range(n_f):
            for s in range(1, SYMS_PER_FRAME):
                h = h_est[f, s - 1]
                mag = np.maximum(np.abs(h), floor)
                y = raw[f, s] * np.conj(h) / mag**2
                w = np.minimum(np.abs(h) / med_h, 1.0)
                i0 = (s - 1) * NC_LATENT * 2
                sl = framing.symbols_to_slots(y[:NC_LATENT][None, :])
                slot_values[f, i0 : i0 + NC_LATENT * 2] = sl
                slot_weights[f, i0 : i0 + NC_LATENT * 2] = np.repeat(w[:NC_LATENT], 2)
                beacon_soft[f * CHIPS_PER_FRAME + (s - 1)] = np.real(
                    raw[f, s, BEACON_CARRIER] * np.conj(h[BEACON_CARRIER])
                )

        beacon_result = beacon.decode(beacon_soft)
        latents_full = np.zeros(MODES["C"].n_latents)
        weights_full = np.zeros(MODES["C"].n_latents)
        frame_offset = None
        if beacon_result is not None:
            frame_offset = (
                beacon_result.frame_index - beacon_result.chip_offset // CHIPS_PER_FRAME
            )
            # The beacon's mode field bounds which absolute frames can be
            # real: everything the buffer holds past the transmission's
            # actual last frame is post-transmission noise, and placing it
            # would hand reconstruct() garbage latents at nonzero weight
            # where a true erasure (weight 0) is what the decoder was
            # trained for. An unknown mode index (a future mode) falls
            # back to mode C's full range -- the pre-mode-field behaviour
            # -- rather than rejecting the reception.
            tx_spec = MODES_BY_INDEX.get(beacon_result.mode_index)
            n_frames_limit = (
                tx_spec.n_frames if tx_spec is not None
                else LATENT_GROUPS * FRAMES_PER_GROUP
            )
            for f in range(n_f):
                abs_frame = frame_offset + f
                if 0 <= abs_frame < n_frames_limit:
                    _, idx = framing.slot_range_for_frame(abs_frame)
                    latents_full[idx] = np.clip(slot_values[f], -10, 10)
                    weights_full[idx] = slot_weights[f]

        return BlindDemodResult(
            latents=latents_full,
            weights=weights_full,
            freq_offset=ba.freq_offset,
            beacon=beacon_result,
            callsign=beacon_result.callsign if beacon_result else "",
            frame_offset=frame_offset,
            n_frames=n_f,
            # Anchor on p_start, not p0: the demod loop (and so the beacon
            # chip stream frame_offset indexes) starts at p_start, which is
            # L_lo frames away from p0 whenever the blind lock isn't already
            # at the buffer start. Using p0 here put absolute frame 0 off by
            # L_lo frames -- tens of seconds for a mid-stream lock.
            frame0_start=(
                p_start - frame_offset * FRAME_SAMPLES
                if frame_offset is not None else None
            ),
            snr_db=_estimate_snr_db(h_pilot),
        )

    def _demod_frames(
        self, z: np.ndarray, p: int, n_f: int, phi_ref: float, shift: int,
        tracker: "_DriftTracker | None", unstep: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Demodulate up to `n_f` frames from `p`, the window moved
        `shift` samples later than nominal. Returns raw (n_f, 6, NC),
        pilot gains (n_f, NC), which frames were in the buffer, and each
        frame's accumulated timing step.

        Sample-clock drift is followed by the pilot phase slope across
        carriers against `phi_ref`, through a slow EMA and +-2 sample
        window steps. The raw per-frame timing estimate also sees the
        channel's group delay, which swings by many samples as
        multipath taps fade; real clock drift is < 0.1 samples/frame.
        Each step's own phase is then undone, so every frame shares one
        timing reference and the pilot interpolator never straddles a
        step. `unstep=False` leaves them in: that is what the delays
        look like to the frames as demodulated, which is what window
        placement must be measured on (measured: placing against the
        unstepped profile cost 0.6 dB on mpd).
        """
        # The shift's own slope, or the loop would read it as drift and
        # walk the window straight back.
        phi_ref = phi_ref + 2 * np.pi * RS * shift / FS
        p = p + shift
        raw = np.zeros((n_f, SYMS_PER_FRAME, NC), dtype=np.complex128)
        received = np.zeros(n_f, dtype=bool)
        steps = np.zeros(n_f, dtype=np.int64)
        pilot_powers: list[float] = []
        tau_ema, total = 0.0, 0
        h_prev = None
        for f in range(n_f):
            if p + FRAME_SAMPLES > len(z):
                break
            if tracker is None:
                for s in range(SYMS_PER_FRAME):
                    raw[f, s] = ofdm.demod_window(z, p + s * NSYM + NCP, DEMOD_BACKOFF)
            else:
                zz = tracker.frame(z, p)
                for s in range(SYMS_PER_FRAME):
                    raw[f, s] = ofdm.demod_window(zz, s * NSYM + NCP, DEMOD_BACKOFF)
            h = raw[f, 0] / self.pilot
            received[f] = True
            steps[f] = total
            p += FRAME_SAMPLES

            power = float(np.mean(np.abs(h) ** 2))
            pilot_powers.append(power)
            healthy = power > 0.1 * np.median(pilot_powers)
            if tracker is not None:
                # A faded frame's pilot phase is noise; feed the loop
                # nothing rather than a bad measurement, but still let it
                # coast forward on its rate estimate. Its phase sees the
                # step as well, so compare like with like.
                if unstep and h_prev is not None:
                    h_prev = h_prev * _time_shift_phase(steps[f - 1] - steps[f])
                tracker.update(h, h_prev if healthy else None)
            h_prev = h
            if healthy:
                phi = self._bin_phase_step(h)
                d = np.angle(np.exp(1j * (phi - phi_ref)))
                tau = -d * FS / (2 * np.pi * RS)
                tau_ema += 0.02 * (tau - tau_ema)
                if abs(tau_ema) >= 2:
                    step = int(np.clip(round(tau_ema), -2, 2))
                    p += step
                    total += step
                    tau_ema -= step
        if unstep:
            raw *= _time_shift_phase(steps)[:, None, :]
        return raw, raw[:, 0] / self.pilot, received, steps

    @staticmethod
    def _bin_phase_step(h: np.ndarray) -> float:
        """Mean per-carrier phase increment of a gain vector (timing proxy)."""
        return float(np.angle(np.sum(h[1:] * np.conj(h[:-1]))))

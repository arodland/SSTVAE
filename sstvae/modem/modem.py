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
    """Result of demodulate_blind: no preamble/header was needed, so
    unlike DemodResult there's no known `mode` — latents/weights are
    always sized for mode C's full canonical range (every mode is a
    prefix of it) and only populated where a demodulated frame actually
    landed, via the beacon's recovered absolute frame index."""

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
        """This frame's samples, de-rotated by the running estimate."""
        return z[p : p + FRAME_SAMPLES] * np.exp(
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
        beacon_chips = beacon.chip_stream(0, n_f, callsign)
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
        raw = np.zeros((n_f, SYMS_PER_FRAME, NC), dtype=np.complex128)
        h_pilot = np.zeros((n_f, NC), dtype=np.complex128)
        received = np.zeros(n_f, dtype=bool)
        phi_ref = self._bin_phase_step(h_pre)
        pilot_powers: list[float] = []

        # Sample-clock drift tracking. The raw per-frame timing estimate
        # also sees the channel's group delay, which swings by many
        # samples as multipath taps fade; real clock drift is < 0.1
        # samples/frame. A slow EMA keeps the fading wiggle out while
        # following the drift ramp; shifts are small and incremental.
        tau_ema = 0.0
        tracker = _make_tracker(drift_track)
        p = h0 + HEADER_SAMPLES
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
            h_pilot[f] = raw[f, 0] / self.pilot
            received[f] = True
            p += FRAME_SAMPLES

            power = float(np.mean(np.abs(raw[f, 0]) ** 2))
            pilot_powers.append(power)
            healthy = power > 0.1 * np.median(pilot_powers)
            if tracker is not None:
                # A faded frame's pilot phase is noise; feed the loop
                # nothing rather than a bad measurement, but still let it
                # coast forward on its rate estimate.
                prev = h_pilot[f - 1] if (f > 0 and healthy and received[f - 1]) else None
                tracker.update(h_pilot[f], prev)
            if healthy:
                phi = self._bin_phase_step(h_pilot[f])
                d = np.angle(np.exp(1j * (phi - phi_ref)))
                tau = -d * FS / (2 * np.pi * RS)
                tau_ema += 0.02 * (tau - tau_ema)
                if abs(tau_ema) >= 2:
                    step = int(np.clip(round(tau_ema), -2, 2))
                    p += step
                    tau_ema -= step

        # Equalize data symbols with pilots interpolated across the frame.
        latents = np.zeros(spec.n_tx_latents)
        weights = np.zeros(spec.n_tx_latents)
        med_h = np.median(np.abs(h_pilot[received])) if received.any() else 1.0
        floor = max(0.05 * med_h, 1e-9)
        def pilot_at(i: int, fallback: int) -> np.ndarray:
            if 0 <= i < n_f and received[i]:
                return h_pilot[i]
            return h_pilot[fallback]

        beacon_soft = np.zeros(n_f * CHIPS_PER_FRAME)
        for f in range(n_f):
            if not received[f]:
                continue
            # Catmull-Rom interpolation over four surrounding pilots: the
            # 6.9 Hz pilot rate oversamples even 2 Hz Doppler fading, but
            # linear interpolation alone loses ~14 dB tracking it.
            p0, p1 = pilot_at(f - 1, f), h_pilot[f]
            p2 = pilot_at(f + 1, f)
            p3 = pilot_at(f + 2, f + 1 if f + 1 < n_f and received[f + 1] else f)
            frame_slots = np.zeros(LATENTS_PER_FRAME)
            frame_w = np.zeros(LATENTS_PER_FRAME)
            for s in range(1, SYMS_PER_FRAME):
                u = s / SYMS_PER_FRAME
                h = 0.5 * (
                    2 * p1
                    + (p2 - p0) * u
                    + (2 * p0 - 5 * p1 + 4 * p2 - p3) * u**2
                    + (3 * p1 - p0 - 3 * p2 + p3) * u**3
                )
                mag = np.maximum(np.abs(h), floor)
                y = raw[f, s] * np.conj(h) / mag**2
                w = np.minimum(np.abs(h) / med_h, 1.0)
                i0 = (s - 1) * NC_LATENT * 2
                sl = framing.symbols_to_slots(y[:NC_LATENT][None, :])
                frame_slots[i0 : i0 + NC_LATENT * 2] = sl
                frame_w[i0 : i0 + NC_LATENT * 2] = np.repeat(w[:NC_LATENT], 2)
                beacon_soft[f * CHIPS_PER_FRAME + (s - 1)] = np.real(y[BEACON_CARRIER])
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
            freq_offset=acq.freq_offset,
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

        raw = np.zeros((n_f, SYMS_PER_FRAME, NC), dtype=np.complex128)
        h_pilot = np.zeros((n_f, NC), dtype=np.complex128)
        tracker = _make_tracker(drift_track)
        pilot_powers: list[float] = []
        p = p_start
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

        def pilot_at(i: int) -> np.ndarray:
            return h_pilot[int(np.clip(i, 0, n_f - 1))]

        beacon_soft = np.zeros(n_f * CHIPS_PER_FRAME)
        slot_values = np.zeros((n_f, LATENTS_PER_FRAME))
        slot_weights = np.zeros((n_f, LATENTS_PER_FRAME))
        for f in range(n_f):
            p0_, p1_, p2_, p3_ = pilot_at(f - 1), h_pilot[f], pilot_at(f + 1), pilot_at(f + 2)
            for s in range(1, SYMS_PER_FRAME):
                u = s / SYMS_PER_FRAME
                h = 0.5 * (
                    2 * p1_
                    + (p2_ - p0_) * u
                    + (2 * p0_ - 5 * p1_ + 4 * p2_ - p3_) * u**2
                    + (3 * p1_ - p0_ - 3 * p2_ + p3_) * u**3
                )
                mag = np.maximum(np.abs(h), floor)
                y = raw[f, s] * np.conj(h) / mag**2
                w = np.minimum(np.abs(h) / med_h, 1.0)
                i0 = (s - 1) * NC_LATENT * 2
                sl = framing.symbols_to_slots(y[:NC_LATENT][None, :])
                slot_values[f, i0 : i0 + NC_LATENT * 2] = sl
                slot_weights[f, i0 : i0 + NC_LATENT * 2] = np.repeat(w[:NC_LATENT], 2)
                beacon_soft[f * CHIPS_PER_FRAME + (s - 1)] = np.real(y[BEACON_CARRIER])

        beacon_result = beacon.decode(beacon_soft)
        latents_full = np.zeros(MODES["C"].n_latents)
        weights_full = np.zeros(MODES["C"].n_latents)
        frame_offset = None
        if beacon_result is not None:
            frame_offset = (
                beacon_result.frame_index - beacon_result.chip_offset // CHIPS_PER_FRAME
            )
            for f in range(n_f):
                abs_frame = frame_offset + f
                if 0 <= abs_frame < LATENT_GROUPS * FRAMES_PER_GROUP:
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

    @staticmethod
    def _bin_phase_step(h: np.ndarray) -> float:
        """Mean per-carrier phase increment of a gain vector (timing proxy)."""
        return float(np.angle(np.sum(h[1:] * np.conj(h[:-1]))))

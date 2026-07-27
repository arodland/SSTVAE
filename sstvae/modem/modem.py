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
from .dsp import to_baseband, freq_correct, tx_condition
from .sync import acquire, acquire_blind, SyncError

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
        self, x: np.ndarray, search_s: tuple[float, float] | None = None
    ) -> DemodResult:
        """`search_s` restricts preamble acquisition to a time window
        (seconds); frames are still demodulated past its end."""
        z = to_baseband(np.asarray(x, dtype=np.float64))
        search = None
        if search_s is not None:
            search = (int(search_s[0] * FS), int(search_s[1] * FS))
        acq = acquire(z, search=search)
        z = freq_correct(z, acq.freq_offset)

        u0 = acq.preamble_start + PREAMBLE_CP
        h_pre = (
            ofdm.demod_window(z, u0, DEMOD_BACKOFF)
            + ofdm.demod_window(z, u0 + M, DEMOD_BACKOFF)
        ) / (2 * self.pilot)

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
        p = h0 + HEADER_SAMPLES
        for f in range(n_f):
            if p + FRAME_SAMPLES > len(z):
                break
            for s in range(SYMS_PER_FRAME):
                raw[f, s] = ofdm.demod_window(z, p + s * NSYM + NCP, DEMOD_BACKOFF)
            h_pilot[f] = raw[f, 0] / self.pilot
            received[f] = True
            p += FRAME_SAMPLES

            power = float(np.mean(np.abs(raw[f, 0]) ** 2))
            pilot_powers.append(power)
            healthy = power > 0.1 * np.median(pilot_powers)
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
        self, x: np.ndarray, search_s: tuple[float, float] | None = None
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
        """
        z = to_baseband(np.asarray(x, dtype=np.float64))
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
        p = p_start
        for f in range(n_f):
            for s in range(SYMS_PER_FRAME):
                raw[f, s] = ofdm.demod_window(z, p + s * NSYM + NCP, DEMOD_BACKOFF)
            h_pilot[f] = raw[f, 0] / self.pilot
            p += FRAME_SAMPLES

        med_h = np.median(np.abs(h_pilot))
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

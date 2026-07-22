"""Top-level modem: latent vector <-> passband audio samples.

TX layout:  silence | preamble | 2x header symbol | N frames | silence
Frame:      1 pilot symbol + 11 data symbols (528 real latents).

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
    M,
    NCP,
    NSYM,
    SYMS_PER_FRAME,
    FRAME_SAMPLES,
    LATENTS_PER_FRAME,
    PREAMBLE_CP,
    PREAMBLE_SAMPLES,
    HEADER_SAMPLES,
    LEADIN_SAMPLES,
    LEADOUT_SAMPLES,
    CLIP_HEADROOM_DB,
    DEMOD_BACKOFF,
    MODES,
    ModeSpec,
)
from . import framing, ofdm
from .dsp import to_baseband, freq_correct, tx_condition
from .sync import acquire, SyncError

__all__ = ["Modem", "DemodResult", "SyncError"]


@dataclass
class DemodResult:
    latents: np.ndarray  # canonical order, zeros where not received
    weights: np.ndarray  # per-latent confidence 0..1 (0 = erased)
    mode: ModeSpec
    freq_offset: float
    sync_metric: float
    frames_received: int


class Modem:
    def __init__(self):
        self.pilot = ofdm.pilot_sequence()

    # --- transmit ----------------------------------------------------------

    def modulate(
        self, latents: np.ndarray, mode: str | ModeSpec, normalize: bool = True
    ) -> np.ndarray:
        """Latent vector -> unit-RMS float waveform at FS.

        The on-air contract is unit-RMS latents; `normalize` enforces it.
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
        symbols = np.empty((n_f * SYMS_PER_FRAME, NC), dtype=np.complex128)
        for f in range(n_f):
            sl = slots[f * LATENTS_PER_FRAME : (f + 1) * LATENTS_PER_FRAME]
            symbols[f * SYMS_PER_FRAME] = self.pilot
            symbols[f * SYMS_PER_FRAME + 1 : (f + 1) * SYMS_PER_FRAME] = (
                framing.slots_to_symbols(sl)
            )

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

    def demodulate(self, x: np.ndarray) -> DemodResult:
        z = to_baseband(np.asarray(x, dtype=np.float64))
        acq = acquire(z)
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
        latents = np.zeros(spec.n_latents)
        weights = np.zeros(spec.n_latents)
        med_h = np.median(np.abs(h_pilot[received])) if received.any() else 1.0
        floor = max(0.05 * med_h, 1e-9)
        def pilot_at(i: int, fallback: int) -> np.ndarray:
            if 0 <= i < n_f and received[i]:
                return h_pilot[i]
            return h_pilot[fallback]

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
                i0 = (s - 1) * NC * 2
                sl = framing.symbols_to_slots(y[None, :])
                frame_slots[i0 : i0 + NC * 2] = sl
                frame_w[i0 : i0 + NC * 2] = np.repeat(w, 2)
            lo = f * LATENTS_PER_FRAME
            latents[lo : lo + LATENTS_PER_FRAME] = frame_slots
            weights[lo : lo + LATENTS_PER_FRAME] = frame_w

        latents = np.clip(latents, -10, 10)
        return DemodResult(
            latents=framing.deinterleave(latents, spec),
            weights=framing.deinterleave(weights, spec),
            mode=spec,
            freq_offset=acq.freq_offset,
            sync_metric=acq.metric,
            frames_received=int(received.sum()),
        )

    @staticmethod
    def _bin_phase_step(h: np.ndarray) -> float:
        """Mean per-carrier phase increment of a gain vector (timing proxy)."""
        return float(np.angle(np.sum(h[1:] * np.conj(h[:-1]))))

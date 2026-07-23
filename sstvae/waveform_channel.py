"""Stage-2 differentiable channel: the real modem chain in torch.

Latents ride through OFDM synthesis, envelope clip-and-filter (the same
PAPR conditioning as the TX), symbol-domain Watterson fading, AWGN,
noisy-pilot equalization with Catmull-Rom interpolation, and burst
frame erasures — everything the NumPy modem does except sync, which is
assumed acquired. Gradients flow end to end, and the synthesized
envelope PAPR is returned for use as a loss term.

Always simulates the full mode C stream; group truncation is expressed
through the weights (untransmitted groups = weight 0), matching how the
decoder sees a real early-stopped reception.
"""

from dataclasses import dataclass

import numpy as np
import torch

from .config import (
    FS,
    NC,
    M,
    NCP,
    NSYM,
    SYMS_PER_FRAME,
    DATA_SYMS_PER_FRAME,
    LATENTS_PER_FRAME,
    GROUP_LATENTS,
    FRAMES_PER_GROUP,
    LATENT_GROUPS,
    CLIP_HEADROOM_DB,
    DEMOD_BACKOFF,
    TX_BANDPASS,
    MODES,
)
from .modem import framing, ofdm


@dataclass
class Stage2Config:
    snr_db_range: tuple[float, float] = (-2.0, 22.0)
    p_fading: float = 0.7  # else AWGN-only
    doppler_range_hz: tuple[float, float] = (0.1, 2.0)
    delay_range_ms: tuple[float, float] = (0.5, 4.0)
    p_truncate: float = 0.5
    erasure_bursts_mean: float = 1.0  # Poisson mean per transmission
    erasure_burst_frames: tuple[int, int] = (1, 20)
    clip_headroom_db: float = CLIP_HEADROOM_DB


class WaveformChannel(torch.nn.Module):
    N_FRAMES = LATENT_GROUPS * FRAMES_PER_GROUP  # 660
    N_SYMS = N_FRAMES * SYMS_PER_FRAME
    N_LATENTS = MODES["C"].n_latents

    def __init__(self, cfg: Stage2Config | None = None):
        super().__init__()
        self.cfg = cfg or Stage2Config()

        perm = np.concatenate(
            [framing._PERMS[g] + g * GROUP_LATENTS for g in range(LATENT_GROUPS)]
        )
        self.register_buffer("perm", torch.from_numpy(perm).long())
        self.register_buffer("inv_perm", torch.from_numpy(np.argsort(perm)).long())

        self.register_buffer(
            "mod_mat", torch.from_numpy(ofdm.MOD_MATRIX).to(torch.complex64)
        )
        n = np.arange(M)
        demod = np.exp(-2j * np.pi * np.outer(ofdm.CARRIER_FREQS, n) / FS)
        self.register_buffer("demod_mat", torch.from_numpy(demod).to(torch.complex64))
        self.register_buffer(
            "pilot", torch.from_numpy(ofdm.pilot_sequence()).to(torch.complex64)
        )
        from scipy.signal import firwin

        taps = firwin(201, TX_BANDPASS, fs=FS, pass_zero=False)
        self.register_buffer("bp_taps", torch.from_numpy(taps).float()[None, None, :])
        self.register_buffer(
            "carrier_freqs", torch.from_numpy(ofdm.CARRIER_FREQS.astype(np.float32))
        )

        sym_idx = torch.arange(self.N_SYMS)
        self.register_buffer("is_pilot", (sym_idx % SYMS_PER_FRAME) == 0)

    # --- TX ------------------------------------------------------------

    def _to_symbols(self, latents: torch.Tensor) -> torch.Tensor:
        b = latents.shape[0]
        slots = latents[:, self.perm]
        s = slots.view(b, self.N_FRAMES, DATA_SYMS_PER_FRAME, NC, 2)
        data = torch.complex(s[..., 0], s[..., 1]).to(torch.complex64) / np.sqrt(2)
        syms = torch.empty(
            (b, self.N_FRAMES, SYMS_PER_FRAME, NC),
            dtype=torch.complex64,
            device=latents.device,
        )
        syms[:, :, 0, :] = self.pilot
        syms[:, :, 1:, :] = data
        return syms.view(b, self.N_SYMS, NC)

    def _synthesize(self, syms: torch.Tensor) -> torch.Tensor:
        x = torch.einsum("bsc,nc->bsn", syms, self.mod_mat).real
        return x.reshape(x.shape[0], -1)

    def _clip_filter(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Envelope clip + bandpass (2 iterations); returns (x, papr_db)."""
        thresh = (
            torch.sqrt(2 * x.pow(2).mean(dim=1, keepdim=True))
            * 10 ** (self.cfg.clip_headroom_db / 20)
        )
        for _ in range(2):
            z = self._analytic(x)
            mag = z.abs().clamp_min(1e-9)
            x = (z * torch.clamp(thresh / mag, max=1.0)).real
            x = torch.nn.functional.conv1d(x[:, None, :], self.bp_taps, padding=100)[
                :, 0, :
            ]
        env2 = self._analytic(x).abs().pow(2)
        papr_db = 10 * torch.log10(
            torch.quantile(env2, 0.9999, dim=1) / env2.mean(dim=1).clamp_min(1e-12)
        )
        return x, papr_db

    @staticmethod
    def _analytic(x: torch.Tensor) -> torch.Tensor:
        n = x.shape[-1]
        X = torch.fft.fft(x.to(torch.float32), dim=-1)
        h = torch.zeros(n, device=x.device)
        h[0] = 1
        h[1 : (n + 1) // 2] = 2
        if n % 2 == 0:
            h[n // 2] = 1
        return torch.fft.ifft(X * h, dim=-1)

    def _demodulate(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        xs = x.view(b, self.N_SYMS, NSYM)
        start = NCP - DEMOD_BACKOFF
        win = xs[:, :, start : start + M].to(torch.complex64)
        return (2.0 / M) * torch.einsum("bsn,cn->bsc", win, self.demod_mat)

    # --- channel ---------------------------------------------------------

    def _smooth_gains(self, b: int, doppler: torch.Tensor, device) -> torch.Tensor:
        """(b, N_SYMS) complex tap gains, ~Gaussian Doppler spectrum."""
        sym_rate = FS / NSYM
        g = torch.complex(
            torch.randn(b, self.N_SYMS, device=device),
            torch.randn(b, self.N_SYMS, device=device),
        )
        # Per-sample Gaussian smoothing kernel sized by Doppler.
        sigma_syms = (sym_rate / (2 * np.pi * doppler)).clamp(1.0, self.N_SYMS / 4)
        out = torch.empty_like(g)
        half = 3 * int(sigma_syms.max().item())
        t = torch.arange(-half, half + 1, device=device).float()
        for i in range(b):
            k = torch.exp(-0.5 * (t / sigma_syms[i]) ** 2)
            k = (k / k.sum())[None, None, :]
            gr = torch.nn.functional.conv1d(
                g[i].real[None, None, :], k, padding=half
            )[0, 0, : self.N_SYMS]
            gi = torch.nn.functional.conv1d(
                g[i].imag[None, None, :], k, padding=half
            )[0, 0, : self.N_SYMS]
            out[i] = torch.complex(gr, gi)
        return out / out.abs().pow(2).mean(dim=1, keepdim=True).sqrt().clamp_min(1e-9)

    def _fading(self, b: int, device) -> torch.Tensor:
        """(b, N_SYMS, NC) complex channel gains."""
        cfg = self.cfg
        lo, hi = cfg.doppler_range_hz
        doppler = lo + torch.rand(b, device=device) * (hi - lo)
        g1 = self._smooth_gains(b, doppler, device)
        g2 = self._smooth_gains(b, doppler, device)
        dlo, dhi = cfg.delay_range_ms
        tau_s = (dlo + torch.rand(b, device=device) * (dhi - dlo)) * 1e-3
        phase = -2 * np.pi * tau_s[:, None] * self.carrier_freqs[None, :]
        rot = torch.polar(torch.ones_like(phase), phase).to(torch.complex64)
        h = (g1[:, :, None] + g2[:, :, None] * rot[:, None, :]) / np.sqrt(2)
        flat = torch.rand(b, device=device) >= cfg.p_fading
        h[flat] = 1.0 + 0j
        return h

    # --- RX -------------------------------------------------------------

    def _equalize(self, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Noisy-pilot EQ with Catmull-Rom interpolation, like the modem."""
        b = y.shape[0]
        yf = y.view(b, self.N_FRAMES, SYMS_PER_FRAME, NC)
        h_pilot = yf[:, :, 0, :] / self.pilot  # (b, F, NC)

        def pilot(idx):
            return h_pilot[:, idx.clamp(0, self.N_FRAMES - 1), :]

        f = torch.arange(self.N_FRAMES, device=y.device)
        p0, p1, p2, p3 = pilot(f - 1), pilot(f), pilot(f + 1), pilot(f + 2)
        u = (
            torch.arange(1, SYMS_PER_FRAME, device=y.device).float() / SYMS_PER_FRAME
        )[None, None, :, None]
        p0, p1, p2, p3 = (p[:, :, None, :] for p in (p0, p1, p2, p3))
        hi = 0.5 * (
            2 * p1
            + (p2 - p0) * u
            + (2 * p0 - 5 * p1 + 4 * p2 - p3) * u**2
            + (3 * p1 - p0 - 3 * p2 + p3) * u**3
        )  # (b, F, D, NC)

        med = h_pilot.abs().flatten(1).median(dim=1).values[:, None, None, None]
        med = med.clamp_min(1e-6)
        mag = torch.maximum(hi.abs(), 0.05 * med)
        data = yf[:, :, 1:, :]
        eq = data * hi.conj() / mag.pow(2)
        w = (hi.abs() / med).clamp(0.0, 1.0)
        return eq, w

    def _burst_erasures(self, b: int, device) -> torch.Tensor:
        keep = torch.ones(b, self.N_FRAMES, device=device)
        n_bursts = torch.poisson(
            torch.full((b,), self.cfg.erasure_bursts_mean, device=device)
        )
        blo, bhi = self.cfg.erasure_burst_frames
        for i in range(b):
            for _ in range(int(n_bursts[i])):
                ln = int(torch.randint(blo, bhi + 1, (1,)))
                st = int(torch.randint(0, self.N_FRAMES - 1, (1,)))
                keep[i, st : st + ln] = 0.0
        return keep

    # --- full chain -------------------------------------------------------

    def forward(
        self, latents: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(B, N_LATENTS) unit-RMS -> (noisy latents, weights, papr_db)."""
        b, dev = latents.shape[0], latents.device
        cfg = self.cfg

        syms = self._to_symbols(latents)
        x = self._synthesize(syms)
        x, papr_db = self._clip_filter(x)
        y = self._demodulate(x)

        h = self._fading(b, dev)
        lo, hi_snr = cfg.snr_db_range
        snr = lo + torch.rand(b, 1, 1, device=dev) * (hi_snr - lo)
        sigma = (10 ** (-snr / 20)) / np.sqrt(2)
        noise = torch.complex(
            torch.randn(b, self.N_SYMS, NC, device=dev),
            torch.randn(b, self.N_SYMS, NC, device=dev),
        ) * sigma
        y = y * h + noise

        eq, w = self._equalize(y)

        keep = self._burst_erasures(b, dev)
        n_groups = torch.full((b,), LATENT_GROUPS, device=dev, dtype=torch.long)
        trunc = torch.rand(b, device=dev) < cfg.p_truncate
        n_groups[trunc] = torch.randint(
            1, LATENT_GROUPS + 1, (int(trunc.sum()),), device=dev
        )
        frame_group = (
            torch.arange(self.N_FRAMES, device=dev) // FRAMES_PER_GROUP
        )[None, :]
        keep = keep * (frame_group < n_groups[:, None])
        w = w * keep[:, :, None, None]

        eqw = eq * (w > 0)
        sl_pairs = torch.stack(
            [eqw.real * np.sqrt(2), eqw.imag * np.sqrt(2)], dim=-1
        )
        sl = sl_pairs.reshape(b, -1)[:, self.inv_perm]
        wl = (
            torch.stack([w, w], dim=-1).reshape(b, -1)[:, self.inv_perm]
        )
        sl = sl.clamp(-10, 10)
        return sl, wl, papr_db

"""Stage-1 differentiable channel model applied to latents during training.

Approximates what the modem+HF channel does to latents:
  - AWGN at a per-sample random SNR,
  - truncation to a random number of channel groups (progressive TX),
  - random erasures approximating lost frames (the interleaver spreads a
    frame across the whole image, so iid masks are a fair stand-in),
  - correlated amplitude weighting approximating fading (weights known
    to the decoder, as the pilot EQ reports them).

Stage 2 replaces this with a model matched to the real OFDM waveform.
"""

from dataclasses import dataclass

import torch

from .config import LATENT_GROUPS, CHANNELS_PER_GROUP


@dataclass
class ChannelConfig:
    snr_db_range: tuple[float, float] = (0.0, 22.0)
    erasure_rate_max: float = 0.3
    p_truncate: float = 0.5  # probability of dropping trailing groups


def apply_latent_channel(
    z: torch.Tensor, cfg: ChannelConfig, generator: torch.Generator | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """z: (B, C, H, W) unit-RMS latents -> (noisy latents, weights)."""
    b, c, h, w = z.shape
    dev = z.device

    def rand(*shape):
        return torch.rand(*shape, device=dev, generator=generator)

    weights = torch.ones_like(z)

    # Group truncation: keep a random prefix of groups per sample.
    keep = torch.full((b,), LATENT_GROUPS, device=dev, dtype=torch.long)
    trunc = rand(b) < cfg.p_truncate
    keep[trunc] = torch.randint(
        1, LATENT_GROUPS + 1, (int(trunc.sum()),), device=dev, generator=generator
    )
    group_idx = torch.arange(c, device=dev) // CHANNELS_PER_GROUP
    weights = weights * (group_idx[None, :, None, None] < keep[:, None, None, None])

    # Random erasures (lost frames, deep fades).
    rate = rand(b, 1, 1, 1) * cfg.erasure_rate_max
    weights = weights * (rand(b, c, h, w) >= rate)

    # AWGN on surviving latents.
    lo, hi = cfg.snr_db_range
    snr_db = lo + rand(b, 1, 1, 1) * (hi - lo)
    sigma = (10 ** (-snr_db / 20)).to(z.dtype)
    noise = torch.randn(z.shape, device=dev, generator=generator) * sigma
    noisy = (z + noise) * weights
    return noisy, weights.to(z.dtype)

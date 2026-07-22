"""Convolutional autoencoder whose latents ride the OFDM waveform.

Not a strict-ELBO VAE: as in RADE, the channel noise injected during
training acts as the regularizer. The encoder emits a bounded,
unit-RMS latent tensor (the modem's on-air contract); the decoder
receives noisy latents plus a per-latent confidence weight and can run
on any prefix of the channel groups (progressive / early-stopped
reception).

Latent tensor layout: (B, LATENT_CHANNELS, LATENT_H, LATENT_W), with
channels [0:44) = group 0 (coarse), [44:88) = group 1, [88:132) = group 2.
The canonical flat latent vector fed to the modem is groups concatenated,
each flattened C-order — matching sstvae.modem framing.
"""

import torch
import torch.nn as nn

from ..config import (
    LATENT_CHANNELS,
    LATENT_GROUPS,
    CHANNELS_PER_GROUP,
    LATENT_H,
    LATENT_W,
)


class ResBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(8, ch),
            nn.SiLU(),
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.GroupNorm(8, ch),
            nn.SiLU(),
            nn.Conv2d(ch, ch, 3, padding=1),
        )

    def forward(self, x):
        return x + self.block(x)


class Encoder(nn.Module):
    """(B, 3, 240, 320) image in [0,1] -> unit-RMS latents."""

    def __init__(self, width: int = 128):
        super().__init__()
        w = width
        self.net = nn.Sequential(
            nn.Conv2d(3, w // 2, 4, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(w // 2, w, 4, stride=2, padding=1),
            ResBlock(w),
            nn.SiLU(),
            nn.Conv2d(w, w * 2, 4, stride=2, padding=1),
            ResBlock(w * 2),
            ResBlock(w * 2),
            nn.GroupNorm(8, w * 2),
            nn.SiLU(),
            nn.Conv2d(w * 2, LATENT_CHANNELS, 3, padding=1),
        )

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        z = torch.tanh(self.net(img * 2 - 1))
        # Unit RMS over each image's whole latent tensor: the modem
        # transmits latents at unit power, so train with the same scale.
        rms = z.flatten(1).pow(2).mean(dim=1, keepdim=True).sqrt().clamp_min(1e-6)
        return z / rms[:, :, None, None]


class Decoder(nn.Module):
    """Noisy latents + per-latent weights -> (B, 3, 240, 320) in [0,1].

    Erased latents must be zeroed and their weights 0; the weight planes
    let the network discount unreliable coefficients.
    """

    def __init__(self, width: int = 128):
        super().__init__()
        w = width
        self.net = nn.Sequential(
            nn.Conv2d(2 * LATENT_CHANNELS, w * 2, 3, padding=1),
            ResBlock(w * 2),
            ResBlock(w * 2),
            nn.SiLU(),
            nn.ConvTranspose2d(w * 2, w, 4, stride=2, padding=1),
            ResBlock(w),
            ResBlock(w),
            nn.SiLU(),
            nn.ConvTranspose2d(w, w // 2, 4, stride=2, padding=1),
            ResBlock(w // 2),
            nn.SiLU(),
            nn.ConvTranspose2d(w // 2, w // 4, 4, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(w // 4, 3, 3, padding=1),
        )

    def forward(self, z: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        x = torch.cat([z * weights, weights], dim=1)
        return torch.sigmoid(self.net(x))


class SSTVAE(nn.Module):
    def __init__(self, width: int = 128):
        super().__init__()
        self.encoder = Encoder(width)
        self.decoder = Decoder(width)
        self.width = width

    @staticmethod
    def latents_to_flat(z: torch.Tensor) -> torch.Tensor:
        """(B, C, H, W) -> (B, n_latents) canonical modem order."""
        b = z.shape[0]
        groups = z.split(CHANNELS_PER_GROUP, dim=1)
        return torch.cat([g.reshape(b, -1) for g in groups], dim=1)

    @staticmethod
    def flat_to_latents(flat: torch.Tensor) -> torch.Tensor:
        b = flat.shape[0]
        gs = flat.chunk(LATENT_GROUPS, dim=1)
        return torch.cat(
            [g.reshape(b, CHANNELS_PER_GROUP, LATENT_H, LATENT_W) for g in gs],
            dim=1,
        )

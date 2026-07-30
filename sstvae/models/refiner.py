"""Optional post-decoder refiner: reconstruction -> closer-to-reference.

The autoencoder trains to its own objective and never sees this model;
the refiner is trained afterwards against a *frozen* codec, on decoder
outputs degraded through the stage-1 latent channel. That makes it a
pure receive-side option — no on-air change, no new encoder, and a
station that skips it (CPU, download size, taste) loses nothing it had
before. On a progressive reception it only needs to run once, on the
final decode.

It is deliberately *conditioned* rather than blind: the receiver already
knows which latents were erased or never transmitted (the same weight
planes the decoder consumes) and how good the channel was, so the
refiner is told instead of left to infer damage from pixels. The
conditioning planes are the per-group mean of the latent weights,
bilinearly upsampled to image resolution (a spatial erasure-density map
per group, which also encodes mode truncation as an all-zero plane),
plus one plane of scalar channel confidence.

At inference the confidence scalar must be computed exactly as training
computed it: `confidence_from_snr_db` reuses `latent_channel`'s sigmoid
anchors so the two cannot drift apart.

The final conv is zero-initialized, so an untrained refiner is exactly
the identity — it can only ever be trained *into* helping, and a
mismatched or corrupt checkpoint degrades toward "no refinement", not
toward garbage.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import (
    LATENT_CHANNELS,
    LATENT_GROUPS,
    CHANNELS_PER_GROUP,
    LATENT_H,
    LATENT_W,
)
from ..latent_channel import SNR_CONF_MIDPOINT, SNR_CONF_SCALE
from .autoencoder import ResBlock


def confidence_from_snr_db(snr_db: torch.Tensor) -> torch.Tensor:
    """Map an estimated channel SNR (dB) to the confidence scalar the
    refiner was conditioned on during training. Same anchors as
    `latent_channel.apply_latent_channel`; erasures are visible to the
    refiner through the weight planes, so unlike training-time
    confidence this deliberately does not fold in an erasure factor —
    folding it in twice would double-count the damage."""
    return torch.sigmoid((snr_db - SNR_CONF_MIDPOINT) / SNR_CONF_SCALE)


class Refiner(nn.Module):
    """Decoder output + channel conditioning -> refined image, [0,1].

    Residual UNet, two downsampling levels. Kept narrow at full
    resolution on purpose: this runs on CPU in the app (the codec is
    onnxruntime; nothing outside train touches torch), where the
    decoder is ~50 ms per picture — the refiner should stay in that
    class, not in a GPU class.
    """

    COND_CHANNELS = LATENT_GROUPS + 1  # group weight-density planes + confidence

    def __init__(self, width: int = 32):
        super().__init__()
        w = width
        self.width = width
        self.stem = nn.Conv2d(3 + self.COND_CHANNELS, w, 3, padding=1)
        self.down1 = nn.Sequential(
            nn.SiLU(),
            nn.Conv2d(w, w * 2, 4, stride=2, padding=1),
            ResBlock(w * 2),
        )
        self.down2 = nn.Sequential(
            nn.SiLU(),
            nn.Conv2d(w * 2, w * 4, 4, stride=2, padding=1),
            ResBlock(w * 4),
            ResBlock(w * 4),
        )
        self.up1 = nn.Sequential(
            nn.SiLU(),
            nn.ConvTranspose2d(w * 4, w * 2, 4, stride=2, padding=1),
        )
        self.fuse1 = nn.Sequential(
            nn.Conv2d(w * 4, w * 2, 3, padding=1),
            ResBlock(w * 2),
        )
        self.up2 = nn.Sequential(
            nn.SiLU(),
            nn.ConvTranspose2d(w * 2, w, 4, stride=2, padding=1),
        )
        self.fuse2 = nn.Sequential(
            nn.Conv2d(w * 2, w, 3, padding=1),
            ResBlock(w),
        )
        self.tail = nn.Sequential(
            nn.GroupNorm(8, w),
            nn.SiLU(),
            nn.Conv2d(w, 3, 3, padding=1),
        )
        # Identity at init: see module docstring.
        nn.init.zeros_(self.tail[-1].weight)
        nn.init.zeros_(self.tail[-1].bias)

    @staticmethod
    def cond_planes(
        weights: torch.Tensor, confidence: torch.Tensor, out_hw: tuple[int, int]
    ) -> torch.Tensor:
        """(B, LATENT_CHANNELS, LATENT_H, LATENT_W) weights + (B,)
        confidence -> (B, COND_CHANNELS, H, W) conditioning planes."""
        b = weights.shape[0]
        g = weights.view(
            b, LATENT_GROUPS, CHANNELS_PER_GROUP, LATENT_H, LATENT_W
        ).mean(dim=2)
        g = F.interpolate(g, size=out_hw, mode="bilinear", align_corners=False)
        conf = confidence.view(b, 1, 1, 1).expand(b, 1, *out_hw)
        return torch.cat([g, conf.to(g.dtype)], dim=1)

    def forward(
        self, recon: torch.Tensor, weights: torch.Tensor, confidence: torch.Tensor
    ) -> torch.Tensor:
        """recon: (B, 3, H, W) decoder output in [0,1]; weights: the same
        (B, LATENT_CHANNELS, LATENT_H, LATENT_W) planes the decoder was
        given; confidence: (B,) in [0,1] (`confidence_from_snr_db`)."""
        assert weights.shape[1] == LATENT_CHANNELS
        cond = self.cond_planes(weights, confidence, recon.shape[-2:])
        x0 = self.stem(torch.cat([recon * 2 - 1, cond], dim=1))
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        y1 = self.fuse1(torch.cat([self.up1(x2), x1], dim=1))
        y0 = self.fuse2(torch.cat([self.up2(y1), x0], dim=1))
        return (recon + self.tail(y0)).clamp(0.0, 1.0)

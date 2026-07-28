"""The latent tensor <-> flat modem vector mapping, in numpy.

`SSTVAE.latents_to_flat` / `flat_to_latents` are the same functions in
torch. They live there because training needs them differentiable; they
live *here* because the send/receive path must not import torch. The
codec runs on ONNX (see `docs/onnx.md`), and a numpy reshape that
smuggled torch back in would defeat the point of that entirely.

Pure reshape and concatenate -- no arithmetic -- so the two
implementations agree exactly rather than approximately, and
`tests/test_latents.py` asserts it.

The layout is the on-air contract: groups concatenated in order, each
flattened C-order. See `sstvae/models/autoencoder.py` for what the
groups mean.
"""

import numpy as np

from .config import CHANNELS_PER_GROUP, LATENT_GROUPS, LATENT_H, LATENT_W


def latents_to_flat(z: np.ndarray) -> np.ndarray:
    """(B, C, H, W) -> (B, n_latents) in canonical modem order."""
    b = z.shape[0]
    groups = np.split(z, LATENT_GROUPS, axis=1)
    return np.concatenate([g.reshape(b, -1) for g in groups], axis=1)


def flat_to_latents(flat: np.ndarray) -> np.ndarray:
    """(B, n_latents) -> (B, C, H, W). Inverse of `latents_to_flat`."""
    b = flat.shape[0]
    groups = np.split(flat, LATENT_GROUPS, axis=1)
    return np.concatenate(
        [g.reshape(b, CHANNELS_PER_GROUP, LATENT_H, LATENT_W) for g in groups],
        axis=1,
    )

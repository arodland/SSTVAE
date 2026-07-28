"""The numpy latent mapping must equal the torch one, exactly.

`sstvae/latents.py` exists so the send/receive path doesn't import
torch. That is only safe if it is the *same* mapping -- a disagreement
here would scramble every latent's slot assignment and put garbage on
the air, while both implementations kept working on their own terms.

Both are pure reshape and concatenate, so "exactly" is the right bar:
any tolerance at all would be hiding something.
"""

import numpy as np
import pytest

from sstvae import latents
from sstvae.config import LATENT_CHANNELS, LATENT_H, LATENT_W

torch = pytest.importorskip("torch", reason="reference side needs torch")

from sstvae.models import SSTVAE  # noqa: E402


def _sample(batch: int = 2, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal(
        (batch, LATENT_CHANNELS, LATENT_H, LATENT_W)
    ).astype(np.float32)


def test_latents_to_flat_matches_torch():
    z = _sample()
    got = latents.latents_to_flat(z)
    want = SSTVAE.latents_to_flat(torch.from_numpy(z)).numpy()
    assert got.shape == want.shape
    assert np.array_equal(got, want)


def test_flat_to_latents_matches_torch():
    z = _sample()
    flat = latents.latents_to_flat(z)
    got = latents.flat_to_latents(flat)
    want = SSTVAE.flat_to_latents(torch.from_numpy(flat)).numpy()
    assert got.shape == want.shape
    assert np.array_equal(got, want)


def test_round_trip_is_identity():
    z = _sample(batch=3, seed=1)
    assert np.array_equal(latents.flat_to_latents(latents.latents_to_flat(z)), z)


def test_group_order_is_preserved():
    """Group g must occupy the g'th contiguous block of the flat vector.

    This is the property the modem's framing depends on; a transposed
    reshape would still round-trip while sending the groups in the
    wrong order.
    """
    z = np.zeros((1, LATENT_CHANNELS, LATENT_H, LATENT_W), dtype=np.float32)
    per_group = LATENT_CHANNELS // 3 * LATENT_H * LATENT_W
    z[0, 0, 0, 0] = 1.0            # first channel of group 0
    z[0, LATENT_CHANNELS // 3, 0, 0] = 2.0   # first channel of group 1

    flat = latents.latents_to_flat(z)[0]
    assert flat[0] == 1.0
    assert flat[per_group] == 2.0

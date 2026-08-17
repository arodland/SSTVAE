"""PE loss, whose whole content is a sign convention.

The magnitudes are one Laplacian and a multiply; what can actually be
wrong is the *direction* — flag the blurred pixels or the sharp ones —
and inverting it produces a loss that still trains, still falls, and
optimizes for smoothing. So the load-bearing test is a blurred and a
sharpened reconstruction with **identical** pixel error, which MSE
cannot tell apart by construction and PE loss must.
"""

import pytest

torch = pytest.importorskip("torch")

import torch.nn.functional as F  # noqa: E402

from sstvae.pe_loss import blur_factor_map, laplacian, pe_loss  # noqa: E402


def _step_edge(h=8, w=8):
    """A hard vertical edge — the paper's own worked example (Eq. 1)."""
    img = torch.zeros(1, 1, h, w)
    img[..., w // 2 :] = 1.0
    return img


def _smoothed(img, c=0.1):
    """One explicit diffusion step: blur, by exactly `c * laplacian`."""
    return img + c * laplacian(img)


def _sharpened(img, c=0.1):
    """The same perturbation with the opposite sign: unsharp mask."""
    return img - c * laplacian(img)


def test_alpha_zero_is_mse_exactly():
    # The default has to be a true no-op, or every baseline comparison
    # silently measures the knob's plumbing instead of the knob.
    g = torch.Generator().manual_seed(0)
    recon = torch.rand(2, 3, 16, 16, generator=g)
    target = torch.rand(2, 3, 16, 16, generator=g)
    assert pe_loss(recon, target, 0.0) == F.mse_loss(recon, target)
    assert pe_loss(recon, target, 0.0, p=1.0) == F.l1_loss(recon, target)


def test_laplacian_border_is_replicate_not_zero():
    # Zero padding would fabricate a maximal edge around every image, so
    # the border would be permanently "blurred" whatever the network did.
    flat = torch.full((1, 1, 5, 5), 0.7)
    assert torch.allclose(laplacian(flat), torch.zeros(1, 1, 5, 5))


def test_blur_is_flagged_and_sharpening_is_not():
    """The sign convention, stated as the thing it has to distinguish."""
    target = _step_edge()
    blurred, sharp = _smoothed(target), _sharpened(target)

    # Same pixel error, to the last bit — only the sign differs.
    assert F.mse_loss(blurred, target) == F.mse_loss(sharp, target)

    # Blur fills the dip on the dark side of the edge (positive
    # Laplacian, positive error) and shaves the bright side: signs agree
    # everywhere the second-order gradient is nonzero.
    edge = laplacian(target).abs() > 0
    assert torch.equal(blur_factor_map(blurred, target)[edge],
                       laplacian(target).abs()[edge])
    # Overshoot puts the error on the other side of the target, so no
    # pixel is penalized and the weight map is flat 1.
    assert torch.count_nonzero(blur_factor_map(sharp, target)) == 0

    alpha = 2.0
    assert pe_loss(blurred, target, alpha) > pe_loss(sharp, target, alpha)
    assert pe_loss(sharp, target, alpha) == F.mse_loss(sharp, target)


def test_weight_is_one_plus_alpha_m_squared_at_p2():
    # W multiplies the difference *inside* the norm (Eq. 10), so at p=2
    # the effective weight on a squared error is W**2, not W. That is why
    # the paper's alpha=2.0 (chosen against L1 backbones) does not carry
    # over, and it is worth pinning rather than rediscovering.
    target = _step_edge()
    recon = _smoothed(target)
    alpha = 3.0
    w = 1.0 + alpha * blur_factor_map(recon, target)
    expected = (w.square() * (recon - target).square()).mean()
    assert torch.allclose(pe_loss(recon, target, alpha), expected)


def test_alpha_broadcasts_per_sample():
    # How --pe-conf-scale gets applied: alpha scaled by each sample's
    # channel confidence, so a badly faded picture is allowed to hedge.
    target = torch.cat([_step_edge(), _step_edge()])
    recon = _smoothed(target)
    alpha = torch.tensor([0.0, 4.0]).view(-1, 1, 1, 1)
    both = pe_loss(recon, target, alpha)
    off = pe_loss(recon[:1], target[:1], 0.0)
    on = pe_loss(recon[1:], target[1:], 4.0)
    assert torch.allclose(both, (off + on) / 2)


def test_gradient_flows_only_through_the_error():
    # The weight map is a function of the target's edges and the *sign*
    # of the error; treating it as differentiable would let the network
    # reduce the loss by nudging pixels across a sign boundary rather
    # than by reconstructing them.
    target = _step_edge()
    recon = _smoothed(target).requires_grad_(True)
    alpha = 2.0
    pe_loss(recon, target, alpha).backward()
    w = 1.0 + alpha * blur_factor_map(recon.detach(), target)
    n = target.numel()
    expected = 2.0 * w.square() * (recon.detach() - target) / n
    assert torch.allclose(recon.grad, expected)

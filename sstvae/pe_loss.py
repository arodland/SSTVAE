"""PE loss: perception-enhanced distortion-oriented loss. Training-only.

Li, Wang, Zhong, Zhang & Liu, "PE loss: Perception-enhanced
distortion-oriented loss for image restoration", *Computational Visual
Media* 12(3):825-839, 2026 (doi:10.26599/CVM.2025.9450475).

A pixel-space reweighting of an existing L_p distortion loss, not a new
metric: the target's second-order gradient says where the edges are, the
*sign* of the reconstruction error says whether this pixel fell on the
blurred side of one, and the error there is amplified. Motivated by the
Mach band effect -- the HVS enhances edges, so a blurred edge reads as
worse quality than its pixel error alone says.

    g = laplacian(target)                      (paper Eq. 3)
    d = recon - target                         (Eq. 7)
    M = |g| where sign(g) == sign(d), else 0   (Eq. 8)
    W = 1 + alpha * M                          (Eq. 9)
    L = mean(|W * d| ** p)                     (Eq. 10)

**Sign agreement means blurred**, and getting that backwards is the trap
this module exists to make testable: it inverts the whole loss into one
that penalizes *sharp* pixels and rewards smoothing. On the dark side of
an edge the Laplacian is positive (a local minimum), and blur fills that
dip in, so `recon > target` -- signs agree. A reconstruction that
overshoots instead pushes the same pixel further down, so the signs
differ and the pixel gets weight 1. `tests/test_pe_loss.py` asserts both
directions on a synthetic step edge rather than trusting the algebra.

**alpha is not transferable from the paper**, which never states its
intensity scale -- `M` is `|laplacian|` in whatever units the images
carry, and ours are floats in [0, 1]. It also interacts with `p`: W
multiplies the difference *inside* the norm, so the effective weight on
a squared error is `W**2`.

**Do not calibrate alpha on a synthetic step edge.** That reaches
|g| ~ 2-4 and suggests alpha = 2 is a huge weight; on real photographs
(measured, 12 coco640 val images) |g| is median 0.031, p99 0.74, max
2.78, so alpha = 2 gives mean W 1.14 and max 6.6. The number that
actually says how hard the loss is pulling is the **amplification
ratio** `mean(W^2 d^2) / mean(d^2)` -- PE loss *is* MSE at 1.0 -- and it
is far above what mean W suggests, because reconstruction error lives
exactly where the Laplacian is large. Measured against v4 through the
stage-2 channel, with the `--pe-conf-scale` factor (conf averages 0.52)
in parentheses:

    alpha  0.5 -> 1.36x (1.17x)      alpha  5 ->  9.1x (3.98x)
    alpha  2.0 -> 3.03x (1.86x)      alpha 10 -> 27.5x (9.74x)

Quote a dose in those terms, not in alpha.

**What it buys, in the paper's own numbers, is perceptual quality paid
for in PSNR** (their Table 1, alpha = 2.0): super-resolution LPIPS
0.2479 -> 0.2357 for PSNR 29.14 -> 28.81 dB; deblurring LPIPS 0.0754 ->
0.0684 for 31.19 -> 30.77 dB. The trend worth knowing here is that the
PSNR cost *shrinks* as the degradation gets heavier -- denoising at
sigma=75 gives up 0.07 dB for a 10% LPIPS improvement against 0.37 dB at
sigma=25 -- and this project's operating points are the heavy end.
"""

import torch


def laplacian(img: torch.Tensor) -> torch.Tensor:
    """4-neighbour discrete Laplacian, replicate-padded, per channel.

    Written as slice arithmetic rather than a `conv2d`, for two reasons:
    it stays in the caller's dtype (autocast would run a convolution in
    bf16, and this is differencing near-equal neighbours), and zero
    padding would fabricate a maximal edge all the way around the border.
    """
    x = torch.nn.functional.pad(img, (1, 1, 1, 1), mode="replicate")
    return (
        x[..., :-2, 1:-1]
        + x[..., 2:, 1:-1]
        + x[..., 1:-1, :-2]
        + x[..., 1:-1, 2:]
        - 4.0 * img
    )


def blur_factor_map(recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Paper Eq. 8: `|laplacian(target)|` on pixels whose error has the
    same sign as it (blurred), 0 elsewhere.

    Constant with respect to `recon` by construction -- the magnitude
    comes from the target alone and only the *sign* of the error selects
    it -- so it is detached. Gradient reaches the network through `d` in
    `pe_loss`, never through the weight.
    """
    g = laplacian(target).detach()
    d = (recon - target).detach()
    blurred = ((g > 0) & (d > 0)) | ((g <= 0) & (d <= 0))
    return torch.where(blurred, g.abs(), torch.zeros_like(g))


def pe_loss(
    recon: torch.Tensor,
    target: torch.Tensor,
    alpha,
    p: float = 2.0,
) -> torch.Tensor:
    """Paper Eq. 10. `alpha` is a float or a tensor broadcastable against
    the image (e.g. `[B, 1, 1, 1]` to scale the penalty per sample).

    At `alpha = 0` this is exactly `F.mse_loss` for `p = 2` and
    `F.l1_loss` for `p = 1` -- the same expression, not an approximation
    of it, which is what makes the knob's default a true no-op.
    """
    d = recon - target
    if isinstance(alpha, torch.Tensor) or alpha:
        w = 1.0 + alpha * blur_factor_map(recon, target)
        d = w * d
    return d.abs().pow(p).mean() if p != 2.0 else d.square().mean()

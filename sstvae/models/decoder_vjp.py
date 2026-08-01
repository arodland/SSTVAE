"""The decoder's vector-Jacobian product, as a forward-only module.

**Export-time only, like the rest of this package: torch.** Nothing in
the send/receive path imports it. What ships is the ONNX graph exported
*from* it, which the transmit-time latent optimizer
(`docs/latent-optimization.md`) runs on the same pinned inference
onnxruntime as the codec — no autodiff engine, no training-capable ORT
build, no torch on the operator's machine.

The decoder's weights are frozen, so the only gradient anyone wants is
the one with respect to its *input*. That makes this a fixed, finite
chain rather than a general autodiff problem: each primitive's backward
is written out as ordinary forward tensor ops, so the exporter sees
nothing but Conv, ConvTranspose and elementwise arithmetic.

**Do not reimplement this by calling `torch.autograd.grad` inside
`forward` and exporting through it.** That asks the exporter to trace a
double-backward graph and it hits ops with no ONNX symbolic registered.

The loss lives in the graph on purpose -- see `DecoderVJP`.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import LATENT_CHANNELS
from ..images import IMG_H, IMG_W
from .autoencoder import ResBlock


def _fwd(mod: nn.Module, x: torch.Tensor, tape: list) -> torch.Tensor:
    """Run `mod`, recording each primitive's input for the backward pass.

    The tape is nested rather than flat so a ResBlock's two paths stay
    explicit: `y = x + block(x)` means the incoming gradient reaches `x`
    twice, once unchanged.
    """
    if isinstance(mod, nn.Sequential):
        for m in mod:
            x = _fwd(m, x, tape)
        return x
    if isinstance(mod, ResBlock):
        sub: list = []
        y = _fwd(mod.block, x, sub)
        tape.append(("res", mod, x, sub))
        return x + y
    if isinstance(mod, (nn.Conv2d, nn.ConvTranspose2d, nn.GroupNorm)):
        kind = {nn.Conv2d: "conv", nn.ConvTranspose2d: "convT",
                nn.GroupNorm: "gn"}[type(mod)]
        tape.append((kind, mod, x, None))
        return mod(x)
    if isinstance(mod, nn.SiLU):
        tape.append(("silu", mod, x, None))
        return F.silu(x)
    raise TypeError(f"no backward rule for {type(mod).__name__}")


def _gn_backward(mod: nn.GroupNorm, x: torch.Tensor,
                 g: torch.Tensor) -> torch.Tensor:
    """GroupNorm's input-gradient.

    The fiddly one: mu and sigma are functions of x, so every element's
    gradient carries two reduction terms over its own group. Variance is
    spelled out rather than `torch.var` to keep the exported graph to
    ops with an unambiguous ONNX mapping.
    """
    b, c, h, w = x.shape
    xg = x.reshape(b, mod.num_groups, -1)
    mu = xg.mean(dim=2, keepdim=True)
    centered = xg - mu
    var = (centered * centered).mean(dim=2, keepdim=True)
    inv = torch.rsqrt(var + mod.eps)
    xhat = centered * inv

    gg = (g * mod.weight.view(1, c, 1, 1)).reshape(b, mod.num_groups, -1)
    gx = inv * (gg - gg.mean(dim=2, keepdim=True)
                - xhat * (gg * xhat).mean(dim=2, keepdim=True))
    return gx.reshape(b, c, h, w)


def _bwd(tape: list, g: torch.Tensor) -> torch.Tensor:
    for kind, mod, x, sub in reversed(tape):
        if kind == "res":
            # Both paths see the same incoming gradient; the identity
            # branch passes it through untouched.
            g = g + _bwd(sub, g)
        elif kind == "conv":
            g = F.conv_transpose2d(g, mod.weight, None, mod.stride, mod.padding)
        elif kind == "convT":
            # A transposed convolution's input-gradient is a plain
            # convolution, and ConvTranspose2d's weight layout
            # (in, out, kh, kw) is already what conv2d wants here.
            g = F.conv2d(g, mod.weight, None, mod.stride, mod.padding)
        elif kind == "gn":
            g = _gn_backward(mod, x, g)
        elif kind == "silu":
            s = torch.sigmoid(x)
            g = g * s * (1 + x * (1 - s))
    return g


class DecoderVJP(nn.Module):
    """`(z, weights, target) -> (recon, grad_z, mse)`.

    **The loss lives inside the graph, deliberately.** Taking
    `d(loss)/d(recon)` as an input would keep the graph loss-agnostic,
    but any loss of the reconstruction needs the reconstruction first --
    so the caller would run the graph once to get `recon` and again to
    get the gradient, doubling the cost of every optimizer step. MSE is
    what the optimization uses and what training used; a different loss
    means re-exporting rather than a runtime flag.
    """

    def __init__(self, decoder: nn.Module):
        super().__init__()
        self.decoder = decoder

    def forward(self, z, weights, target):
        x = torch.cat([z * weights, weights], dim=1)
        tape: list = []
        logits = _fwd(self.decoder.net, x, tape)
        recon = torch.sigmoid(logits)

        diff = recon - target
        mse = (diff * diff).mean()
        grad_recon = diff * (2.0 / (3 * IMG_H * IMG_W))

        g = grad_recon * recon * (1 - recon)
        gx = _bwd(tape, g)
        # `x` is cat([z * weights, weights]): the weights half is a
        # constant here, and the product rule leaves the mask behind.
        grad_z = gx[:, :LATENT_CHANNELS] * weights
        return recon, grad_z, mse

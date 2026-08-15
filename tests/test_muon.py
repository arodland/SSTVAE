"""Our Muon against torch.optim.Muon, which is the oracle for 2D.

torch's implementation refuses `ndim != 2`, so the only thing we add is
the conv reshape. Holding the 2D path to bit-level agreement is what
keeps "we extended it" honest rather than "we wrote another one".
"""

import pytest

torch = pytest.importorskip("torch")

from sstvae.muon import (  # noqa: E402
    Muon, adjust_lr, build_param_groups, newton_schulz,
)

torch_muon = pytest.importorskip("torch.optim._muon")


def _quadratic_steps(opt, params, n=6, seed=0):
    """Drive `opt` on a fixed deterministic objective, return final params."""
    g = torch.Generator().manual_seed(seed)
    targets = [torch.randn(p.shape, generator=g) for p in params]
    for _ in range(n):
        opt.zero_grad()
        loss = sum(((p - t) ** 2).sum() for p, t in zip(params, targets))
        loss.backward()
        opt.step()
    return [p.detach().clone() for p in params]


def _clone(shapes, seed=1):
    g = torch.Generator().manual_seed(seed)
    return [torch.nn.Parameter(torch.randn(s, generator=g)) for s in shapes]


def test_newton_schulz_orthogonalizes():
    """Singular values should collapse toward 1."""
    torch.manual_seed(0)
    g = torch.randn(64, 128)
    s = torch.linalg.svdvals(newton_schulz(g).float())
    assert s.min() > 0.4 and s.max() < 1.6


def test_newton_schulz_rejects_non_2d():
    with pytest.raises(ValueError):
        newton_schulz(torch.randn(4, 8, 3, 3))


def test_adjust_lr_formulas():
    assert adjust_lr(1.0, "none", (10, 4)) == 1.0
    assert adjust_lr(1.0, "original", (10, 4)) == pytest.approx((10 / 4) ** 0.5)
    assert adjust_lr(1.0, "original", (4, 10)) == pytest.approx(1.0)
    assert adjust_lr(1.0, "match_rms_adamw", (4, 10)) == pytest.approx(
        0.2 * 10 ** 0.5)
    with pytest.raises(ValueError):
        adjust_lr(1.0, "bogus", (4, 4))


@pytest.mark.parametrize("adjust_lr_fn", ["original", "match_rms_adamw"])
@pytest.mark.parametrize("nesterov", [True, False])
def test_2d_matches_torch_muon(adjust_lr_fn, nesterov):
    """The oracle test: identical arithmetic to torch on 2D parameters."""
    shapes = [(32, 64), (64, 32), (48, 48)]
    ours_p, theirs_p = _clone(shapes), _clone(shapes)
    kw = dict(lr=0.02, weight_decay=0.01, momentum=0.95, nesterov=nesterov,
              adjust_lr_fn=adjust_lr_fn)
    ours = Muon([{"params": ours_p, "use_muon": True, "lr": kw["lr"],
                  "weight_decay": kw["weight_decay"]}], **kw)
    theirs = torch.optim.Muon(theirs_p, **kw)
    a = _quadratic_steps(ours, ours_p)
    b = _quadratic_steps(theirs, theirs_p)
    for x, y in zip(a, b):
        assert torch.equal(x, y), (x - y).abs().max()


def test_adamw_path_matches_torch_adamw():
    """An all-AdamW configuration must be a no-op change."""
    shapes = [(16,), (32,), (8, 8)]
    ours_p, theirs_p = _clone(shapes), _clone(shapes)
    ours = Muon([{"params": ours_p, "use_muon": False, "lr": 2e-3,
                  "weight_decay": 0.01}], lr=2e-3, weight_decay=0.01)
    theirs = torch.optim.AdamW(theirs_p, lr=2e-3, weight_decay=0.01)
    a = _quadratic_steps(ours, ours_p)
    b = _quadratic_steps(theirs, theirs_p)
    for x, y in zip(a, b):
        assert torch.allclose(x, y, atol=1e-7), (x - y).abs().max()


def test_conv_4d_equals_reshaped_2d():
    """The extension itself: a 4D filter behaves as its (out, in*kh*kw) matrix.

    Oracle is torch.optim.Muon driven on a genuinely 2D parameter of the
    reshaped shape, with the same objective reshaped to match.
    """
    shape4, shape2 = (16, 8, 3, 3), (16, 72)
    g = torch.Generator().manual_seed(3)
    init = torch.randn(shape4, generator=g)
    p4 = torch.nn.Parameter(init.clone())
    p2 = torch.nn.Parameter(init.reshape(shape2).clone())

    kw = dict(lr=0.02, weight_decay=0.01, momentum=0.95, nesterov=True,
              adjust_lr_fn="match_rms_adamw")
    ours = Muon([{"params": [p4], "use_muon": True, "lr": kw["lr"],
                  "weight_decay": kw["weight_decay"]}], **kw)
    theirs = torch.optim.Muon([p2], **kw)

    tg = torch.Generator().manual_seed(7)
    target = torch.randn(shape4, generator=tg)
    for _ in range(5):
        ours.zero_grad()
        ((p4 - target) ** 2).sum().backward()
        ours.step()
        theirs.zero_grad()
        ((p2 - target.reshape(shape2)) ** 2).sum().backward()
        theirs.step()
    assert torch.equal(p4.detach().reshape(shape2), p2.detach())


def test_build_param_groups_splits_by_rank():
    from sstvae.models import SSTVAE

    model = SSTVAE(width=32)
    groups = build_param_groups(model, lr=2e-4)
    assert [g["use_muon"] for g in groups] == [True, False]
    assert all(p.ndim >= 2 for p in groups[0]["params"])
    assert all(p.ndim < 2 for p in groups[1]["params"])
    n = sum(len(g["params"]) for g in groups)
    assert n == len([p for p in model.parameters() if p.requires_grad])
    # Conv/ConvTranspose weights are the bulk of the model by parameter
    # count; if that ever stops being true, Muon is optimizing a corner.
    muon_numel = sum(p.numel() for p in groups[0]["params"])
    total = sum(p.numel() for p in model.parameters())
    assert muon_numel / total > 0.95


def test_step_actually_moves_a_conv_model():
    from sstvae.models import SSTVAE

    torch.manual_seed(0)
    model = SSTVAE(width=32)
    opt = Muon(build_param_groups(model, lr=2e-4))
    before = [p.detach().clone() for p in model.parameters()]
    img = torch.rand(1, 3, 480, 640)
    z = model.encoder(img)
    out = model.decoder(z, torch.ones_like(z))
    out.mean().backward()
    opt.step()
    moved = sum(not torch.equal(a, b)
                for a, b in zip(before, model.parameters()))
    assert moved == len(before)

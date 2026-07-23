import numpy as np
import pytest

torch = pytest.importorskip("torch")

from sstvae.config import MODES
from sstvae.waveform_channel import Stage2Config, WaveformChannel


def _clean_cfg(**kw):
    base = dict(
        snr_db_range=(60.0, 60.0),
        p_fading=0.0,
        p_truncate=0.0,
        erasure_bursts_mean=0.0,
    )
    base.update(kw)
    return Stage2Config(**base)


def _unit(b, seed=0):
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(b, MODES["C"].n_latents, generator=g)
    return z / z.pow(2).mean(dim=1, keepdim=True).sqrt()


def test_clean_loopback_hits_clip_floor():
    ch = WaveformChannel(_clean_cfg())
    z = _unit(2)
    out, w, papr_pre, papr_post, conf = ch(z)
    err = (out - z).pow(2).mean()
    snr = 10 * torch.log10(z.pow(2).mean() / err)
    assert snr > 17, f"latent SNR {snr:.1f} dB"
    assert w.min() > 0.5
    assert 5.0 < papr_post.min() and papr_post.max() < 9.0, f"PAPR {papr_post}"
    # Random unshaped latents are peaky pre-clip; clipping should reduce
    # (or at worst leave roughly equal) the measured PAPR.
    assert (papr_post <= papr_pre + 0.5).all()


def test_matches_numpy_modem():
    """Torch chain and the real NumPy modem should agree closely on the
    same latents over a clean channel."""
    from sstvae.modem import Modem

    ch = WaveformChannel(_clean_cfg())
    z = _unit(1, seed=3)
    out_t, _, _, _, _ = ch(z)
    r = Modem().demodulate(Modem().modulate(z[0].numpy().astype(np.float64), "C"))
    # Both go through clip-and-filter + EQ; residuals differ (sync path,
    # real acquisition) but recovered latents should correlate ~1 with
    # each other and with the input.
    a, b = out_t[0].numpy(), r.latents
    corr = np.corrcoef(a, b)[0, 1]
    assert corr > 0.98, f"torch/numpy latent correlation {corr:.3f}"


def test_gradients_flow():
    ch = WaveformChannel(_clean_cfg())
    z = _unit(1)
    z.requires_grad_(True)
    out, w, papr_pre, papr_post, conf = ch(z)
    (out.pow(2).mean() + papr_pre.mean() * 0.01 + papr_post.mean() * 0.01).backward()
    g = z.grad
    assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0


def test_papr_pre_clip_has_informative_gradient():
    """This is the actual bug being fixed: post-clip PAPR alone gives
    ~zero gradient once clipping is active (crest factor is
    scale-invariant, so clipping pins the post-clip envelope near a
    fixed level regardless of how peaky the input was). Pre-clip PAPR
    must respond to genuine changes in the input's crest factor."""
    ch = WaveformChannel(_clean_cfg())
    torch.manual_seed(0)
    z_peaky = _unit(1, seed=10)
    # Sharpen a few latents to create a peakier pre-clip waveform.
    z_peaky = z_peaky.clone()
    z_peaky[:, ::37] *= 4.0
    z_flat = _unit(1, seed=10)

    _, _, papr_pre_peaky, papr_post_peaky, _ = ch(z_peaky)
    _, _, papr_pre_flat, papr_post_flat, _ = ch(z_flat)

    assert papr_pre_peaky.item() > papr_pre_flat.item() + 0.5, (
        f"pre-clip PAPR should clearly track input peakiness: "
        f"{papr_pre_peaky.item():.2f} vs {papr_pre_flat.item():.2f}"
    )


def test_confidence_independent_of_truncation():
    """A clean channel with heavy truncation should report the same
    confidence as a clean channel with no truncation — truncation is
    not channel degradation."""
    torch.manual_seed(0)
    ch_notrunc = WaveformChannel(_clean_cfg(p_truncate=0.0))
    ch_trunc = WaveformChannel(_clean_cfg(p_truncate=1.0))
    z = _unit(8, seed=5)
    _, w1, _, _, conf1 = ch_notrunc(z)
    _, w2, _, _, conf2 = ch_trunc(z)
    assert (w1 == 0).float().mean() < 0.05
    assert (w2 == 0).float().mean() > 0.3  # truncation actually happened
    assert conf1.mean() > 0.9 and conf2.mean() > 0.9
    assert (conf1 - conf2).abs().max() < 0.05


def test_confidence_tracks_snr():
    ch_hi = WaveformChannel(_clean_cfg(snr_db_range=(30.0, 30.0)))
    ch_lo = WaveformChannel(_clean_cfg(snr_db_range=(-5.0, -5.0)))
    z = _unit(4, seed=6)
    _, _, _, _, conf_hi = ch_hi(z)
    _, _, _, _, conf_lo = ch_lo(z)
    assert conf_hi.mean() > conf_lo.mean()


def test_fading_and_erasures_shape_weights():
    ch = WaveformChannel(
        Stage2Config(
            snr_db_range=(12.0, 12.0),
            p_fading=1.0,
            p_truncate=1.0,
            erasure_bursts_mean=3.0,
        )
    )
    torch.manual_seed(0)
    z = _unit(4, seed=4)
    out, w, papr_pre, papr_post, conf = ch(z)
    assert ((w == 0).float().mean() > 0.05)  # truncation/erasures present
    good = w > 0.7
    if good.any():
        err = (out[good] - z[good]).pow(2).mean()
        snr = 10 * torch.log10(z[good].pow(2).mean() / err)
        assert snr > 5, f"faded latent SNR {snr:.1f} dB"

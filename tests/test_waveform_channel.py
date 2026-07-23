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
    out, w, papr = ch(z)
    err = (out - z).pow(2).mean()
    snr = 10 * torch.log10(z.pow(2).mean() / err)
    assert snr > 17, f"latent SNR {snr:.1f} dB"
    assert w.min() > 0.5
    assert 5.0 < papr.min() and papr.max() < 9.0, f"PAPR {papr}"


def test_matches_numpy_modem():
    """Torch chain and the real NumPy modem should agree closely on the
    same latents over a clean channel."""
    from sstvae.modem import Modem

    ch = WaveformChannel(_clean_cfg())
    z = _unit(1, seed=3)
    out_t, _, _ = ch(z)
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
    out, w, papr = ch(z)
    (out.pow(2).mean() + papr.mean() * 0.01).backward()
    g = z.grad
    assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0


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
    out, w, papr = ch(z)
    assert ((w == 0).float().mean() > 0.05)  # truncation/erasures present
    good = w > 0.7
    if good.any():
        err = (out[good] - z[good]).pow(2).mean()
        snr = 10 * torch.log10(z[good].pow(2).mean() / err)
        assert snr > 5, f"faded latent SNR {snr:.1f} dB"

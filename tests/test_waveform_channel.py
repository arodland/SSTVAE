import numpy as np
import pytest

torch = pytest.importorskip("torch")

from sstvae.config import MODES
from sstvae.waveform_channel import Stage2Config, WaveformChannel

from conftest import snr_floor_db

# SNR this scenario reaches with clipping disabled (see conftest).
FADED_ERASED_ONLY_DB = 8.7


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


def test_clean_loopback_hits_clip_floor(clip_floor_db):
    cfg = _clean_cfg()
    ch = WaveformChannel(cfg)
    z = _unit(2)
    out, w, papr_pre, papr_post, conf = ch(z)
    # Exclude the small per-group fraction permanently dropped for the
    # beacon carrier (weight 0 by design, not a channel error).
    mask = w > 0
    err = (out[mask] - z[mask]).pow(2).mean()
    snr = 10 * torch.log10(z[mask].pow(2).mean() / err)
    # The torch replica should land on the same clip floor as the real
    # NumPy modem, whatever headroom is configured.
    assert snr > snr_floor_db(clip_floor_db), f"latent SNR {snr:.1f} dB"
    assert w[mask].min() > 0.5
    # The clipper only engages when the threshold sits below the
    # waveform's own PAPR. When it does, it pins the envelope near the
    # threshold and bandpass regrowth adds some back, so the transmitted
    # PAPR lands above the threshold; when it doesn't, the waveform
    # passes through untouched.
    engaged = cfg.clip_headroom_db < papr_pre
    assert torch.where(
        engaged, papr_post > cfg.clip_headroom_db, papr_post >= papr_pre - 0.5
    ).all(), f"PAPR pre {papr_pre} post {papr_post} headroom {cfg.clip_headroom_db}"
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


def test_fading_and_erasures_shape_weights(clip_floor_db):
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
        # Looser margin than elsewhere: the w > 0.7 mask keeps a
        # different subset of latents as clipping worsens, so the
        # independent-noise model under-predicts the damage here by up
        # to ~2 dB (checked over headroom 5.0 down to -3.0).
        floor = snr_floor_db(clip_floor_db, FADED_ERASED_ONLY_DB, margin_db=3.0)
        assert snr > floor, f"faded latent SNR {snr:.1f} dB, floor {floor:.1f} dB"

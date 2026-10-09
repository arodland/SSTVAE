"""QRSSTVAE carrier and gain tracker, timing and gate V (design 6.3 V,
6.4, 6.5; acceptance R1, R7 timing, R13's Z_ref distribution).

- R1: the genie (true timing and gain, `track.genie_track`) delivers
  per-latent SNR = SNR2500 + 17.09 dB in parallel with the pinned
  distortion D_PASS, within 0.2 dB (at -15, -20 and -25 dB, slow).
- The Kalman smoother on a synthetic Gaussian-Doppler gain: q chosen near
  the true spread, NEES near 1, the causal filter worse than the smoother.
- The matched filter and template: noiseless, m = u c on the known
  symbols, unbiased, up to their neighbours' self-noise.
- Timing: +-100 ppm clocks recovered within 0.02 T rms (R7's timing half).
- `refine_path` follows a warming-up oscillator the acquisition's
  straight-line drift misses.
- Gate V's Z_ref on noise is N(0, 1) (KS p > 0.01) at fixed timing.
- A claimed lead-in with no carrier in it is not used.
- KAPPA_SELF and D_PASS: re-derived from `ce.loopback_stats` and checked
  against the pinned constants.

Every check here that receives a SHORT pass is `slow` (10-15 s each,
see `test_qrss_receiver.py`), as are R1 and gate V's KS test (40 TINY
channels); the default run keeps the smoother, the noiseless
measurement model and KAPPA_SELF / D_PASS.
"""

import math

import numpy as np
import pytest
from scipy import stats

from sstvae.qrss import demod, frame, frontend
from sstvae.qrss import receiver as RX
from sstvae.qrss import track as T
from sstvae.qrss.types import Detection, FreqPath, Timing

from qrss_helpers import Q_TEST, latent_snr_db
from test_qrss_receiver import G_CE_DB, _cfg_key, genie, received, scenario


def _theory_db(snr_db):
    return -10 * math.log10(10 ** (-(snr_db + G_CE_DB) / 10) + demod.D_PASS)


def _pooled_genie_db(snr_db, seeds):
    num = den = 0.0
    for s in seeds:
        a, _ = scenario(snr=snr_db, seed=s)
        z, _ = genie(snr=snr_db, seed=s)
        num += float(np.sum(a ** 2))
        den += float(np.sum((z - a) ** 2))
    return 10 * math.log10(num / den)


@pytest.mark.slow
def test_r1_genie_matches_theory_at_minus_15():
    got = _pooled_genie_db(-15.0, (1, 2, 3))
    assert abs(got - _theory_db(-15.0)) <= 0.2, (got, _theory_db(-15.0))


@pytest.mark.slow
@pytest.mark.parametrize("snr", [-25.0, -20.0])
def test_r1_genie_matches_theory_slow(snr):
    got = _pooled_genie_db(snr, (1, 2, 3, 4))
    assert abs(got - _theory_db(snr)) <= 0.2, (got, _theory_db(snr))


def _gauss_fading(n, B, seed):
    """Unit-power complex gain with a Gaussian Doppler PSD of 2-sigma spread B, at 1/T."""
    rng = np.random.default_rng(seed)
    f = np.fft.fftfreq(n, T.T_SYM)
    S = np.exp(-f ** 2 / (2 * (B / 2) ** 2))
    x = np.fft.ifft(np.sqrt(S) * (rng.standard_normal(n) + 1j * rng.standard_normal(n)))
    return x / np.sqrt(np.mean(np.abs(x) ** 2))


@pytest.mark.parametrize("B", [0.1, 1.0])
def test_kalman_smoother_on_gaussian_doppler(B):
    n = 6000
    u = _gauss_fading(n, B, 3)
    rng = np.random.default_rng(4)
    R = np.full(n, 0.5)
    mu = u + np.sqrt(R / 2) * (rng.standard_normal(n) + 1j * rng.standard_normal(n))
    q, _ = T.select_q(mu, R, 1.0)
    assert 0.4 * B <= T.spread_of_q(q) <= 2.5 * B
    uh, P, _ = T.kalman_rts(mu, R, q, 1.0)
    err = np.abs(uh - u) ** 2
    assert 0.7 <= np.mean(err) / np.mean(P) <= 1.4
    uc, Pc, _ = T.kalman_rts(mu, R, q, 1.0, causal=True)
    assert np.mean(np.abs(uc - u) ** 2) > np.mean(err)
    assert np.mean(err) < 0.1


def test_noiseless_measurement_model():
    """m = u c at every measured frame position, to -25 dB, through the genie."""
    spec = frame.SHORT
    a, sim = scenario(snr=60.0, seed=2)
    prep = RX.prepare(frontend.Capture(sim.fe, Q_TEST, sim.t0_index))
    flat = FreqPath(t_s=np.array([-10.0, 0.0, 1800.0]), f_hz=np.full(3, 1500.0),
                    weight=np.ones(3))
    det = Detection(f_hz=1500.0, path=flat, timing=None, z_ref=np.nan, method="genie",
                    lead_in_s=0.0)
    chan = RX.channel_for(prep, spec, det)
    g = T.genie_track(chan, spec, sim.truth, 1500.0)
    lay = frame.layout(spec)
    gk = g.gi(np.asarray(lay.pos)[lay.known_idx])
    uc = g.u[gk] * g.c[gk]
    r = g.m[gk] - uc
    # unbiased: the template is the mean of what arrives
    assert abs(np.vdot(uc, r)) / np.vdot(uc, uc).real < 0.01
    # what is left is the neighbouring data's self-noise (KAPPA_SELF_KNOWN), ~-21 dB
    assert 10 * np.log10(np.mean(np.abs(r) ** 2) / np.mean(np.abs(g.m[gk]) ** 2)) < -18


@pytest.mark.slow
def test_clock_errors_recovered():
    """R7 timing on SHORT: tx +100 / rx -100 ppm, timing within 0.02 T rms."""
    p, _, _, sim, _ = received(cfg=_cfg_key(dict(tx_ppm=100.0, rx_ppm=-100.0)))
    spec = frame.SHORT
    tp = T.rx_index(p.timing, np.arange(spec.n_pos), spec.n_pos) / 250.0
    assert np.sqrt(np.mean((tp - sim.truth.t_pos_s) ** 2)) / T.T_SYM <= 0.02
    assert abs(p.report.ppm + 200) < 20


@pytest.mark.slow
def test_refine_path_follows_warmup():
    """A 1 Hz/min oscillator settling with tau = 60 s: the acquisition fits a
    line; the measured path follows the truth within 0.05 Hz."""
    cfg = _cfg_key(dict(drift_hz_per_min=1.0, drift_tau_s=60.0))
    p, trs, a, sim, _ = received(cfg=cfg)
    tr = trs[-1]
    t = np.asarray(sim.truth.t_pos_s)
    err = tr.freq(t) - np.asarray(sim.truth.f_hz)
    assert np.sqrt(np.mean(err ** 2)) < 0.05
    zg, _ = genie(cfg=cfg)
    assert latent_snr_db(zg, a) - latent_snr_db(p.z, a) <= 0.2


@pytest.mark.slow
def test_report_drift():
    p = received(cfg=_cfg_key(dict(drift_hz_per_min=1.0)))[0]
    assert abs(p.report.drift_hz_per_min - 1.0) < 0.1
    assert abs(p.report.offset_hz - 1500.0) < 0.05


@pytest.mark.slow
def test_z_ref_on_noise_is_standard_normal():
    """Gate V's statistic at a fixed timing on noise-only TINY channels."""
    spec = frame.TINY
    zs = []
    for seed in range(40):
        rng = np.random.default_rng(seed)
        n = int((12 + spec.n_pos * T.T_SYM + 3) * 4000)
        fe = ((rng.standard_normal(n) + 1j * rng.standard_normal(n)) / np.sqrt(2)).astype(
            np.complex64)
        prep = RX.prepare(frontend.Capture(fe, Q_TEST, 12 * 4000.0))
        f0 = 1400.0 + 5.0 * seed
        path = FreqPath(t_s=np.array([-10.0, 0.0, 100.0]), f_hz=np.full(3, f0),
                        weight=np.ones(3))
        det = Detection(f_hz=f0, path=path, timing=Timing(0.0, 0.0, 0.0, 0.0, np.eye(3)),
                        z_ref=np.nan, method="preamble", lead_in_s=0.0)
        chan = RX.channel_for(prep, spec, det)
        zs.append(T.verify_track(chan, spec, det, fit=False).z_ref)
    p = stats.kstest(zs, "norm").pvalue
    print(f"Z_ref on noise: mean {np.mean(zs):.2f}, sd {np.std(zs):.2f}, KS p {p:.3f}")
    assert p > 0.01


@pytest.mark.slow
def test_false_lead_in_is_dropped():
    """A detection claiming 6 s of lead-in on a frame sent without one: the
    tracker checks the carrier is there and tracks as if it were not claimed."""
    p0, trs, _, _, _ = received()
    chan = trs[0].chan
    det = Detection(f_hz=p0.f_hz, path=FreqPath(np.array([-10.0, 0.0, 300.0]),
                                                np.full(3, p0.f_hz), np.ones(3)),
                    timing=p0.timing, z_ref=50.0, method="preamble", lead_in_s=6.0)
    tr = T.track(chan, frame.SHORT, det, refine=False, outer=0)
    assert tr.lead_in_s == 0.0 and tr.pos[0] == -T.SPAN


@pytest.mark.parametrize("name", ["short", "medium"])
def test_kappa_self_and_d_pass_rederived(name):
    """KAPPA_SELF and D_PASS re-measured with `ce.loopback_stats` (noiseless
    genie loopback, Gaussian latents, two seeds) agree with the pinned
    values: KAPPA_SELF within 0.04 (measured 0.935-0.97), D_PASS within
    15% per frame and 10% on the mean (measured 0.043-0.052)."""
    from sstvae.qrss import ce, header
    from sstvae.qrss.precoder import precode

    from qrss_helpers import unit_rms_latents
    from test_qrss_receiver import HDR

    spec = frame.get(name)
    ks, ds = [], []
    for seed in (1, 2):
        a = unit_rms_latents(spec.n_data, seed)
        sym = frame.assemble(spec, header.encode(HDR), precode(a, Q_TEST))
        st = ce.loopback_stats(sym, spec)
        ks.append(st["kappa_self"])
        ds.append(st["d_pass"])
    print(f"{name}: kappa_self {np.round(ks, 3)}, d_pass {np.round(ds, 4)}")
    assert all(abs(k - T.KAPPA_SELF) <= 0.04 for k in ks)
    assert all(abs(d / demod.D_PASS - 1) <= 0.15 for d in ds)
    assert abs(np.mean(ds) / demod.D_PASS - 1) <= 0.10

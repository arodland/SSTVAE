"""The waveform L tracker (`sstvae/qrss/ltrack.py`; L receiver design 4).

- The Kalman smoother's q grid at T_L: q chosen near the true spread of a
  synthetic Gaussian-Doppler gain sampled at 1/T_L, NEES near 1.
- The classes: round A, round B (the spread copy's mean is a reference),
  an EM prior folded in.
- The pilot filter (one convolution plus interpolation) is the
  closed-form `track.symbol_mf` through `pilot_table()`, and its noise is
  white at the even positions the tracker reads.
- Noiseless genie (`genie_track_l`) on l-tiny: m = u c at the known
  symbols, the data's class gain, delta ~ 1, and the pilot's residual
  against its template is K_PL_DATA; on l-short, with the keying known, a
  callsign window's FSK pilot is in the template exactly, with it unknown
  the window and the filter's reach go unmeasured, and the pilot is its
  template in each section (the clipper's per-section pilot gain).
- track_l on l-tiny at -12 dB from the true timing: |u_hat - u_genie|
  small and NEES near 1; from a coarse fit the data SINR is the genie's;
  the sparse data MF is the full one.
- verify_l accepts that pass from a detection without timing, Z_ref >> 6,
  and at the true timing Z_ref is the complex known symbols' matched
  statistic; Z_ref on noise-only captures at a given timing is N(0, 1) or
  narrower (a handful of seeds here, 40 in slow, plus fitted and blind
  timings). A CE signal taken for L is refused: gate V's known-symbol
  residual (`resid_ratio`) reads it, where Z_ref alone would pass it.
- A 0.15 rad pilot/data phase is recovered in delta, u_hat following the
  data, the timing and the data SINR unmoved.
- A lead-in claimed, unclaimed and falsely claimed; both streams' noise
  model at -6 dB; a blanked hole; the timing refinement at -24 dB.
- track.erase and track.report work on an LTrackResult unchanged;
  `report_l` quotes SNR2500 on the frame's keyed power.

Slow: Z_ref's distribution on noise, +-100 ppm clocks on l-short (the
coarse fit and the tracker), round B against round A, the tracker against
the genie on l-short (steady, quiet, moderate), the pilot's path on a
warming-up oscillator.
"""

import math
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import stats

from sstvae.qrss import channel as chm
from sstvae.qrss import frame, frontend, header, lframe, ldemod, sequences
from sstvae.qrss import ltrack as LT
from sstvae.qrss import receiver as RX
from sstvae.qrss import track as T
from sstvae.qrss.constants import CH_FS, L_T_SYM, SPAN, WAVEFORM_L, Z_ACCEPT
from sstvae.qrss.precoder import precode
from sstvae.qrss.types import Detection, FreqPath, Timing

from qrss_fakes import L_HDR, l_keying, l_scenario, l_symbols
from qrss_helpers import Q_TEST, unit_rms_latents

F0 = 1500.0
FLAT = FreqPath(t_s=np.array([-10.0, 0.0, 1800.0]), f_hz=np.full(3, F0), weight=np.ones(3))


def _det(timing=None, path=FLAT, f0=F0, method="tbd", lead=0.0):
    return Detection(f_hz=f0, path=path, timing=timing, z_ref=np.nan, method=method,
                     lead_in_s=lead, waveform=WAVEFORM_L)


def _channel(name="l-tiny", snr=-12.0, seed=1, cfg=(), lead=0.0):
    """(spec, latents, sim, chan) of one simulated L pass, channelised at the pilot."""
    spec = lframe.get(name)
    a, sim = l_scenario(name, snr, seed, lead_in=lead, cfg=cfg)
    if snr is None:
        prep = RX.Prepared(fe=sim.fe, keep=None, q=sim.q, t0_index=sim.t0_index)
    else:
        prep = RX.prepare(frontend.Capture(sim.fe, Q_TEST, sim.t0_index))
    return spec, a, sim, RX.channel_for(prep, spec, _det())


def _data_sinr_db(tr, spec, a) -> float:
    """SINR (dB) of the data-band MF at the data symbols, against u_hat times
    the sent symbols, after one complex least-squares gain."""
    lay = lframe.layout(spec)
    sym = l_symbols(spec, a)
    gd = tr.gi(np.asarray(lay.pos)[lay.data])
    us = tr.u[gd] * sym[lay.data]
    m = tr.m[gd]
    g = np.vdot(us, m) / np.vdot(us, us)
    return float(10 * np.log10(np.sum(np.abs(g * us) ** 2) / np.sum(np.abs(m - g * us) ** 2)))


def _timing_err(timing, spec, sim) -> float:
    """rms timing error over the frame's positions, in T_L."""
    tp = T.rx_index(timing, np.arange(spec.n_pos), spec.n_pos, CH_FS, t_sym=L_T_SYM)
    return float(np.sqrt(np.mean((tp / CH_FS - sim.truth.t_pos_s) ** 2)) / L_T_SYM)


def _noise_channel(spec, seed):
    """A noise-only capture of l-tiny's length, channelised at a seed's frequency."""
    rng = np.random.default_rng(seed)
    n = int((12 + spec.n_pos * spec.t_sym + 3) * 4000)
    fe = ((rng.standard_normal(n) + 1j * rng.standard_normal(n)) / np.sqrt(2)).astype(
        np.complex64)
    prep = RX.prepare(frontend.Capture(fe, Q_TEST, 12 * 4000.0))
    f0 = 1400.0 + 5.0 * seed
    path = FreqPath(t_s=np.array([-10.0, 0.0, 100.0]), f_hz=np.full(3, f0), weight=np.ones(3))
    det = _det(Timing(0.0, 0.0, 0.0, 0.0, np.eye(3)), path=path, f0=f0, method="preamble")
    return RX.channel_for(prep, spec, det), det


def _z_ref_noise(seeds, mode="fixed"):
    spec = lframe.L_TINY
    zs = []
    for seed in seeds:
        chan, det = _noise_channel(spec, seed)
        if mode == "fixed":
            tr = LT.verify_track_l(chan, spec, det, fit=False)
        elif mode == "fit":
            tr = LT.verify_track_l(chan, spec, det, fit=True, tau_range=6.0)
        else:
            tr = LT.verify_track_l(chan, spec, _det(None, path=det.path, f0=det.f_hz), fit=True)
        zs.append(tr.z_ref)
    return np.array(zs)


def _inject_phase(z_d, phi):
    """z_d with everything below -1.5 Hz (the data band, pilot at 0 Hz) turned by phi."""
    Z = np.fft.fft(z_d)
    Z[np.fft.fftfreq(len(z_d), 1 / CH_FS) < -1.5] *= np.exp(1j * phi)
    return np.fft.ifft(Z)


# --- the smoother at T_L --------------------------------------------------------------------

def _gauss_fading(n, B, seed):
    """Unit-power complex gain with a Gaussian Doppler PSD of 2-sigma spread B, at 1/T_L."""
    rng = np.random.default_rng(seed)
    f = np.fft.fftfreq(n, L_T_SYM)
    S = np.exp(-f ** 2 / (2 * (B / 2) ** 2))
    x = np.fft.ifft(np.sqrt(S) * (rng.standard_normal(n) + 1j * rng.standard_normal(n)))
    return x / np.sqrt(np.mean(np.abs(x) ** 2))


@pytest.mark.parametrize("B", [0.1, 1.0])
def test_q_grid_recovers_the_spread_at_t_l(B):
    n = 4000
    u = _gauss_fading(n, B, 3)
    rng = np.random.default_rng(4)
    R = np.full(n, 0.5)
    mu = u + np.sqrt(R / 2) * (rng.standard_normal(n) + 1j * rng.standard_normal(n))
    q, _ = T.select_q(mu, R, 1.0, T.q_grid(L_T_SYM))
    assert 0.4 * B <= T.spread_of_q(q, L_T_SYM) <= 2.5 * B
    uh, P, _ = T.kalman_rts(mu, R, q, 1.0)
    err = np.abs(uh - u) ** 2
    assert 0.7 <= np.mean(err) / np.mean(P) <= 1.4
    assert np.mean(err) < 0.1


# --- classes and filters ----------------------------------------------------------------------

def test_classes():
    spec = lframe.L_SHORT
    lay = lframe.layout(spec)
    rho = spec.hdr_rho
    a = LT.make_classes_l(spec)
    assert a.known.sum() == len(lay.known_idx) and not a.hdr_known
    assert a.spread is None and a.spread_rho == rho
    assert np.all(a.mu[lay.data] == 0) and np.all(a.nu[lay.data] == 1)
    assert np.allclose(a.mu[lay.known_idx], lay.known_val) and np.all(a.nu[lay.known_idx] == 0)
    assert np.all(a.gain[lay.known_idx] == ldemod.G_REF_L)
    assert np.all(a.gain[lay.hdr_sym] == ldemod.G_HDR_L)
    assert np.all(a.gain[lay.data] == ldemod.G_DATA_L)
    bits = header.encode(L_HDR)
    b = LT.make_classes_l(spec, bits)
    assert b.hdr_known and b.spread_rho == 0.0
    assert b.known.sum() == len(lay.known_idx) + len(lay.hdr_sym)
    assert np.allclose(b.mu[lay.hdr_sym], sequences.qpsk(bits))
    h = frame.spread_signs(spec, bits)
    assert np.allclose(b.mu[lay.data], lframe.pair(math.sqrt(rho) * h / math.sqrt(1 + rho)))
    assert np.allclose(b.nu[lay.data], 1 / (1 + rho))
    # an EM prior: its mean and variance, with the copy's mean once the header is known
    rng = np.random.default_rng(0)
    x = rng.standard_normal(spec.n_data) * 0.5
    v = rng.uniform(0.1, 0.9, spec.n_data)
    prior = SimpleNamespace(x_hat=x, v=v)
    c = LT.make_classes_l(spec, bits, prior)
    assert np.allclose(c.mu[lay.data], lframe.pair((x + math.sqrt(rho) * h) / math.sqrt(1 + rho)))
    assert np.allclose(c.nu[lay.data], 0.5 * (v[0::2] + v[1::2]) / (1 + rho))
    d = LT.make_classes_l(spec, None, prior)
    assert np.allclose(d.mu[lay.data], lframe.pair(x / math.sqrt(1 + rho)))
    vv = (v + rho) / (1 + rho)
    assert np.allclose(d.nu[lay.data], 0.5 * (vv[0::2] + vv[1::2]))


def test_pilot_filter_is_the_closed_form():
    rng = np.random.default_rng(5)
    x = rng.standard_normal(3000) + 1j * rng.standard_normal(3000)
    tm = Timing(tau0=400.3, ppm=37.0, gamma=0.0, z=0.0, cov=np.zeros((3, 3)))
    pos = np.arange(0, 120)
    tb = LT.pilot_table()
    assert abs(tb.g0 - 1.0) < 0.01
    ref = T.symbol_mf(x, CH_FS, tm, pos, t0_index=0.0, n_pos=120, table=tb)
    got = LT.pilot_filter(x, T.rx_index(tm, pos, 120, CH_FS, t_sym=L_T_SYM))
    # white input is the worst case for the interpolation (the pilot is narrow)
    assert 10 * np.log10(np.mean(np.abs(got - ref) ** 2) / np.mean(np.abs(ref) ** 2)) < -40
    # its noise is white at the even positions the tracker reads (the RRC's own
    # spacing, 2 T_L) and far from it at T_L: reading every position would
    # count the pilot's information about twice
    y = LT.pilot_filter(rng.standard_normal(64000) + 1j * rng.standard_normal(64000),
                        1000 + np.arange(4000) * (L_T_SYM * CH_FS))
    r1, r2 = (abs(np.vdot(y[:-k], y[k:])) / np.vdot(y, y).real for k in (1, 2))
    assert r1 > 0.5 and r2 < 0.08


def test_refusals():
    spec, _, sim, chan = _channel()
    g = LT.genie_track_l(chan, spec, sim.truth, F0)
    with pytest.raises(ValueError):
        LT.track_l(chan, spec, _det())                       # no timing
    with pytest.raises(ValueError):
        LT.track_l(chan, spec, _det(g.timing), cw_known=(None, True))
    with pytest.raises(AssertionError):
        LT.make_classes_l(frame.TINY)


# --- the genie, noiseless --------------------------------------------------------------------

def test_genie_noiseless_measurement_model():
    spec, a, sim, chan = _channel(snr=None)
    g = LT.genie_track_l(chan, spec, sim.truth, F0)
    lay = lframe.layout(spec)
    gk = g.gi(np.asarray(lay.pos)[lay.known_idx])
    uc = g.u[gk] * g.c[gk]
    r = g.m[gk] - uc
    assert abs(np.vdot(uc, r)) / np.vdot(uc, uc).real < 0.01
    assert 10 * np.log10(np.mean(np.abs(r) ** 2) / np.mean(np.abs(uc) ** 2)) < -15
    # the data arrive at their class gain
    sym = l_symbols(spec, a)
    gd = g.gi(np.asarray(lay.pos)[lay.data])
    y = g.m[gd] / g.u[gd]
    gain = np.vdot(sym[lay.data], y) / np.vdot(sym[lay.data], sym[lay.data])
    assert abs(gain / ldemod.G_DATA_L - 1) < 0.03
    # the pilot is its template on u's scale: delta ~ 1, residual ~ K_PL_DATA
    assert abs(g.delta - 1) < 0.03
    dat0 = lay.pos[spec.n_pre + spec.n_hdr]
    sel = g.meas_pl & (g.pos > dat0 + 16) & (g.pos < spec.n_pos - 30)
    e = g.m_pl[sel] - g.u_pl[sel] * g.c_pl[sel]
    k = np.mean(np.abs(e) ** 2) / np.mean(np.abs(g.u[sel]) ** 2)
    assert 0.5 * LT.K_PL_DATA < k < 1.5 * LT.K_PL_DATA


def test_window_pilot_with_known_keying_is_in_the_template():
    """l-short noiseless: inside a callsign window the pilot runs at average
    power with the callsign's FSK; with the keying known it is the template."""
    spec, _, sim, chan = _channel("l-short", snr=None)
    g = LT.genie_track_l(chan, spec, sim.truth, F0)
    tau = T._tau_of_rx(g.timing, np.arange(len(g.z_d)) - chan.t0_index, spec.n_pos,
                       t_sym=L_T_SYM)
    tp = chan.t0_index + T.rx_index(g.timing, g.pos, spec.n_pos, CH_FS, t_sym=L_T_SYM)
    c = LT.pilot_filter(LT.pilot_wave(tau, spec, l_keying(spec), 0.0), tp)
    inner = np.zeros(len(g.pos), dtype=bool)
    for pw in spec.win_start_pos:
        inner |= (g.pos >= pw + 8 + 16) & (g.pos < pw + 184 - 16)
    inner &= (g.pos % 2) == 0
    e = g.m_pl[inner] - g.u_pl[inner] * c[inner]
    assert np.mean(np.abs(e) ** 2) / np.mean(np.abs(g.u[inner] * c[inner]) ** 2) < 1e-3
    # and with it unknown, no pilot measurement wherever the pilot filter
    # (+-PILOT_REACH positions) reaches units 8..183: window positions -8..199
    assert not g.meas_pl[inner].any()
    for pw in spec.win_start_pos:
        d = g.pos - pw
        assert not g.meas_pl[(d >= -8) & (d <= 199)].any()
        assert g.meas_pl[d == -10].all() and g.meas_pl[(d == 200) & (g.pos < spec.n_pos)].all()
    # the clipper's pilot gain per section: the pilot is its template, on u's
    # scale, in the preamble, the header and the data alike
    lay = lframe.layout(spec)
    hdr0, dat0 = lay.pos[spec.n_pre], lay.pos[spec.n_pre + spec.n_hdr]
    for lo, hi in ((16, hdr0 - 16), (hdr0 + 16, dat0 - 16),
                   (dat0 + 16, spec.win_start_pos[0] - 16)):
        s = g.meas_pl & (g.pos >= lo) & (g.pos < hi)
        ref = g.u[s] * g.c_pl[s]
        r = np.vdot(ref, g.m_pl[s]) / np.vdot(ref, ref)
        assert abs(abs(r) - 1) < 0.01 and abs(np.angle(r)) < 0.02, (lo, r)


# --- the tracker at -12 dB -------------------------------------------------------------------

def test_track_l_at_true_timing():
    """refine and re-centring off: u_hat against the genie's gain."""
    spec, a, sim, chan = _channel()
    g = LT.genie_track_l(chan, spec, sim.truth, F0)
    tr = LT.track_l(chan, spec, _det(g.timing), refine=False, outer=0)
    assert tr.ook is False and isinstance(tr, LT.LTrackResult)
    fr = tr.pos >= 0
    e = np.abs(tr.u - g.u)[fr] ** 2
    assert 10 * np.log10(np.mean(e) / np.mean(np.abs(g.u[fr]) ** 2)) < -15
    assert 0.4 < np.mean(e) / np.mean(tr.P[fr]) < 2.5
    assert abs(tr.delta - 1) < 0.1
    assert T.spread_of_q(tr.q, L_T_SYM) < 0.1                  # a steady path
    assert tr.meas_pl.any() and not tr.meas_pl[tr.pos % 2 == 1].any()
    # the data MF at a few positions (mixing only what their filters reach)
    # is the data MF of the whole channel
    p = np.array([3, 200, 401, 650, spec.n_pos - 1])
    full = LT.data_mf(tr.z_d, chan, tr.timing, np.arange(spec.n_pos), spec.n_pos)
    assert np.allclose(LT.data_mf(tr.z_d, chan, tr.timing, p, spec.n_pos), full[p],
                       rtol=0, atol=1e-12 * np.max(np.abs(full)))
    # the full tracker from the coarse fit: the genie's data SINR
    z_d = T.derotate(chan, T.freq_path_spline(FLAT))
    tm = LT.fit_timing_l(chan, CH_FS, None, spec, None, None, t0_index=chan.t0_index, z_d=z_d)
    tr2 = LT.track_l(chan, spec, _det(tm))
    assert _data_sinr_db(tr2, spec, a) > _data_sinr_db(g, spec, a) - 0.3
    assert _timing_err(tr2.timing, spec, sim) < 0.03
    # erase and report work unchanged on an L track; report_l quotes the SNR
    rep = T.report(tr2)
    assert abs(rep.snr2500_db + 12.0) < 0.5 and abs(rep.offset_hz - F0) < 0.05
    rep_l = LT.report_l(tr2)
    assert abs(rep_l.snr2500_db + 12.0) < 0.3
    assert 0.1 < rep_l.snr2500_db - rep.snr2500_db < 0.25 and rep_l.ppm == rep.ppm
    psi0 = tr2.psi.copy()
    T.erase(tr2, [(10.0, 12.0)])
    t = tr2.t_pos_s()
    gone = (t > 10.0 - L_T_SYM) & (t < 12.0 + L_T_SYM)
    assert np.all(tr2.psi[gone] == 0) and np.array_equal(tr2.psi[~gone], psi0[~gone])


def test_verify_l_from_a_detection_without_timing():
    spec, _, sim, chan = _channel()
    g = LT.genie_track_l(chan, spec, sim.truth, F0)
    v = LT.verify_l(chan, CH_FS, spec, _det())
    assert v is not None and v.waveform == WAVEFORM_L
    assert v.z_ref > 5 * Z_ACCEPT
    assert abs(v.timing.tau0 - g.timing.tau0) < 1.0
    # at the true timing, unfitted: Z_ref is the matched statistic over the
    # complex known symbols, sqrt(sum 2 |u c|^2 / N) less what u_hat's error
    # costs (a CE-style Im-only statistic would read 0.71 of it)
    tr = LT.verify_track_l(chan, spec, _det(g.timing, method="preamble"), fit=False)
    assert tr.z_ref > 5 * Z_ACCEPT and tr.delta == 1.0
    assert 0.8 < tr.resid_ratio < 1.25          # an L signal is what the model says
    gk = g.gi(np.asarray(lframe.layout(spec).pos)[tr.classes.known])
    z_full = np.sqrt(np.sum(2 * np.abs(g.u[gk] * g.c[gk]) ** 2 / g.N[gk]))
    assert 0.9 * z_full < tr.z_ref < 1.05 * z_full


def test_z_ref_on_noise_few_seeds():
    zs = _z_ref_noise(range(5))
    assert np.all(np.abs(zs) < 3.5) and abs(np.mean(zs)) < 1.0, zs


def test_a_ce_signal_is_not_l():
    """CE TINY at -6 dB gated as L, the pilot taken on its carrier and 22 Hz
    above it (CE's sideband in L's data band): Z_ref on the noise model
    alone passes both (7.1 and 10.8 here), the known symbols' residual
    reads 9-28 times the model's, and the gate refuses it."""
    spec = frame.TINY
    a = unit_rms_latents(spec.n_data, 1)
    sim = chm.simulate(frame.assemble(spec, None, precode(a, Q_TEST)), spec, Q_TEST,
                       chm.ChannelConfig(snr_db=-6.0, seed=1))
    prep = RX.prepare(frontend.Capture(sim.fe, Q_TEST, sim.t0_index))
    lt = lframe.L_TINY
    for off in (0.0, 22.0):
        f0 = F0 + off
        path = FreqPath(t_s=np.array([-10.0, 0.0, 300.0]), f_hz=np.full(3, f0),
                        weight=np.ones(3))
        det = _det(path=path, f0=f0)
        chan = RX.channel_for(prep, lt, det)
        tr = LT.verify_track_l(chan, lt, det)
        assert tr.resid_ratio > 5.0, (off, tr.resid_ratio)
        assert tr.z_ref * math.sqrt(tr.resid_ratio) > Z_ACCEPT > 1.5 * tr.z_ref, \
            (off, tr.z_ref, tr.resid_ratio)
        if off == 0.0:
            assert LT.verify_l(chan, CH_FS, lt, det) is None


def test_pilot_data_phase_is_recovered():
    """A 0.15 rad turn of the data band against the pilot (a rig's passband
    phase): delta takes it, u_hat follows the data's phase rather than the
    pilot's, and the timing and the data SINR do not move. The timing moves
    by 0.02 CH samples (0.0013 T_L): the refinement's prior on delta, about
    1, lends it a little of the phase's information (the phase turns 0.05
    turn per sample of tau0), and 0.15 rad of rig phase is what that costs."""
    spec, a, sim, chan = _channel()
    z0 = T.derotate(chan, T.freq_path_spline(FLAT))
    lay = lframe.layout(spec)
    sym = l_symbols(spec, a)
    out = []
    for phi in (0.0, 0.15):
        z_d = _inject_phase(z0, phi)
        tm = LT.fit_timing_l(chan, CH_FS, None, spec, None, None, t0_index=chan.t0_index,
                             z_d=z_d)
        tr = LT.track_l(chan, spec, _det(tm), outer=0, z_d=z_d)
        gd = tr.gi(np.asarray(lay.pos)[lay.data])
        us = tr.u[gd] * sym[lay.data]
        out.append((np.angle(tr.delta), tr.timing.tau0, _data_sinr_db(tr, spec, a),
                    np.angle(np.vdot(us, tr.m[gd]))))
    (d0, t0, s0, g0), (d1, t1, s1, g1) = out
    assert abs((d1 - d0) + 0.15) < 0.01
    assert abs(g1 - g0) < 0.03
    assert abs(t1 - t0) < 0.05 and abs(s1 - s0) < 0.05


def test_lead_in():
    """l-tiny at -12 dB with 5 s of lead-in (the pilot alone, at average
    power): claimed, the grid starts 82 positions early and the pilot is
    measured there; unclaimed, no pilot measurement whose filter reaches
    before the frame (the lead-in's pilot is not in the template); claimed
    where there is none, the claim is refused."""
    spec, a, sim, chan = _channel(lead=5.0)
    g = LT.genie_track_l(chan, spec, sim.truth, F0)
    n_lead = int(5.0 / L_T_SYM)
    tr = LT.track_l(chan, spec, _det(g.timing, lead=5.0), refine=False, outer=0)
    assert tr.lead_in_s == 5.0 and tr.pos[0] == -SPAN - n_lead
    assert tr.meas_pl[tr.pos < -SPAN].sum() > n_lead // 2 - 2
    fr = tr.pos >= -SPAN
    e = np.abs(tr.u[fr] - g.u) ** 2
    assert 10 * np.log10(np.mean(e) / np.mean(np.abs(g.u) ** 2)) < -20
    assert _data_sinr_db(tr, spec, a) > _data_sinr_db(g, spec, a) - 0.3
    tr0 = LT.track_l(chan, spec, _det(g.timing), refine=False, outer=0)
    assert tr0.lead_in_s == 0.0 and tr0.pos[0] == -SPAN
    assert not tr0.meas_pl[tr0.pos < LT.PILOT_REACH - SPAN].any()
    spec, a, sim, chan = _channel()
    g = LT.genie_track_l(chan, spec, sim.truth, F0)
    trf = LT.track_l(chan, spec, _det(g.timing, lead=5.0), refine=False, outer=0)
    assert trf.lead_in_s == 0.0 and trf.pos[0] == -SPAN


def test_noise_model_of_both_streams():
    """At -6 dB (l-tiny, two seeds) the two streams' observations of the
    true gain, as the tracker hands them to the Kalman filter, have the
    variance the model gives them: data band at the known symbols N plus
    the references' clipper distortion, pilot N_pl = PSD/(2 T_L) plus
    K_PL_DATA (at -6 dB each is about half of its stream's variance)."""
    rd, rp = [], []
    for seed in (1, 2):
        spec, a, sim, chan = _channel(snr=-6.0, seed=seed)
        g = LT.genie_track_l(chan, spec, sim.truth, F0)
        tr = LT.track_l(chan, spec, _det(g.timing), refine=False, outer=0)
        model = LT._Model(tr, spec, tr.classes, False, None, 0.0, 0)
        (mu_d, R_d), (mu_p, R_p) = model.streams(np.abs(g.u) ** 2, g.delta)
        kn = model.is_known & np.isfinite(R_d)
        rd.append(np.abs(mu_d[kn] - g.u[kn]) ** 2 / R_d[kn])
        pl = np.isfinite(R_p)
        rp.append(np.abs(mu_p[pl] - g.u[pl]) ** 2 / R_p[pl])
    rd, rp = np.concatenate(rd), np.concatenate(rp)
    assert len(rd) > 600 and len(rp) > 800
    assert 0.75 < np.mean(rd) < 1.1, np.mean(rd)
    assert 0.9 < np.mean(rp) < 1.3, np.mean(rp)


def test_blanked_hole():
    """1.5 s blanked out of l-tiny at -12 dB: the pilot's psi (the keep
    indicator through the pilot filter) is 0 inside and 1 beyond its reach,
    and the gain is carried across the hole."""
    spec = lframe.get("l-tiny")
    a, sim = l_scenario("l-tiny", -12.0, 1)
    prep = RX.prepare(frontend.Capture(sim.fe, Q_TEST, sim.t0_index))
    fe, keep = prep.fe.copy(), prep.keep.copy()
    hole = slice(int(prep.t0_index + 20.0 * prep.fs), int(prep.t0_index + 21.5 * prep.fs))
    fe[hole], keep[hole] = 0, 0
    chan = RX.channel_for(RX.Prepared(fe=fe, keep=keep, q=prep.q, t0_index=prep.t0_index,
                                      fs=prep.fs), spec, _det())
    g = LT.genie_track_l(chan, spec, sim.truth, F0)
    tr = LT.track_l(chan, spec, _det(g.timing), refine=False, outer=0)
    t = tr.t_pos_s()
    reach = (LT.PILOT_REACH + 2) * L_T_SYM
    assert np.all(tr.psi_pl[(t > 20.3) & (t < 21.2)] < 0.05)
    assert np.all(tr.psi[(t > 20.3) & (t < 21.2)] < 0.05)
    far = (t < 20.0 - reach) | (t > 21.5 + reach)
    assert np.all(np.abs(tr.psi_pl[far] - 1) < 0.01)
    near = (t > 19.5) & (t < 22.0)
    e = np.abs(tr.u[near] - g.u[near]) ** 2
    assert 10 * np.log10(np.mean(e) / np.mean(np.abs(g.u) ** 2)) < -20


def test_refinement_at_minus_24_db():
    """From the true timing, l-tiny at -24 dB: the refinement stays within
    0.05 T_L rms (a likelihood carrying sum |m(theta)|^2 / Var, the noise at
    the samples theta reads, walked 0.11 T rms off here)."""
    errs = []
    for seed in (1, 2, 3):
        spec, a, sim, chan = _channel(snr=-24.0, seed=seed)
        g = LT.genie_track_l(chan, spec, sim.truth, F0)
        tr = LT.track_l(chan, spec, _det(g.timing), outer=0)
        errs.append(_timing_err(tr.timing, spec, sim))
    assert np.sqrt(np.mean(np.square(errs))) < 0.05, errs


# --- slow --------------------------------------------------------------------------------------

@pytest.mark.slow
def test_z_ref_on_noise_is_standard_normal_or_narrower():
    """Gate V at a fixed timing on 40 noise-only l-tiny channels: N(0, 1) by
    KS, sd at most 1 (track.z_ref_stat's variance counts the error P of
    u_hat, which on noise is as large as u_hat: sd about 0.6, as CE's 0.5).
    At a fitted timing Z_ref is the best of the lags searched: shifted up
    (more than CE's, the data/pilot phase turning 0.05 turn per CH sample
    of lag), and still far below Z_ACCEPT."""
    zs = _z_ref_noise(range(40))
    assert stats.kstest(zs, "norm").pvalue > 0.01
    assert abs(np.mean(zs)) < 0.3 and np.std(zs) < 1.1
    zf = _z_ref_noise(range(40, 52), "fit")
    zb = _z_ref_noise(range(60, 72), "blind")
    assert np.max(zf) < 4.0 and np.mean(zf) < 1.5
    assert np.max(zb) < 4.5 and np.mean(zb) < 2.0


@pytest.mark.slow
def test_clock_errors_recovered():
    """tx +100 / rx -100 ppm on l-short at -12 dB, from a blind coarse fit:
    over two seeds the coarse fit (gate V's timing) within 0.02 T_L rms and
    15 ppm of -200 (0.010 T, -194 and -200 ppm as built; with its clock
    phase's sign turned 0.21 T, -170 and -88; the segments' clock phase
    dropped 0.031 T); then the tracker within 0.02 T_L rms, ppm within 20
    of -200."""
    cfg = (("rx_ppm", -100.0), ("tx_ppm", 100.0))
    coarse = []
    for seed in (2, 1):
        spec, a, sim, chan = _channel("l-short", cfg=cfg, seed=seed)
        v = LT.verify_l(chan, CH_FS, spec, _det())
        assert v is not None
        coarse.append(_timing_err(v.timing, spec, sim))
        assert abs(v.timing.ppm + 200) < 15, (seed, v.timing.ppm)
    assert np.sqrt(np.mean(np.square(coarse))) < 0.02, coarse
    tr = LT.track_l(chan, spec, v)
    assert _timing_err(tr.timing, spec, sim) <= 0.02
    assert abs(tr.timing.ppm + 200) < 20
    g = LT.genie_track_l(chan, spec, sim.truth, F0)
    assert _data_sinr_db(tr, spec, a) > _data_sinr_db(g, spec, a) - 0.5


@pytest.mark.slow
def test_refinement_from_a_coarse_slip_on_the_shoulder():
    """l-short at -26 dB, seed 3: the blind coarse fit is 0.22 T_L rms off, a
    slip several samples wrong where the likelihood is not yet concave;
    gradient steps carry the refinement to the peak, within 0.05 T_L rms
    and 0.5 dB of the genie's data SINR (Newton alone stopped there, 2.7 dB
    under)."""
    spec, a, sim, chan = _channel("l-short", snr=-26.0, seed=3)
    v = LT.verify_l(chan, CH_FS, spec, _det())
    assert v is not None
    tr = LT.track_l(chan, spec, v)
    assert _timing_err(tr.timing, spec, sim) < 0.05
    g = LT.genie_track_l(chan, spec, sim.truth, F0)
    assert _data_sinr_db(tr, spec, a) > _data_sinr_db(g, spec, a) - 0.5


def _round_ab(preset, seed=1):
    spec, a, sim, chan = _channel("l-short", cfg=(("preset", preset),), seed=seed)
    v = LT.verify_l(chan, CH_FS, spec, _det())
    assert v is not None
    tra = LT.track_l(chan, spec, v)
    cb = LT.make_classes_l(spec, header.encode(L_HDR))
    trb = LT.track_l(chan, spec, v, cb, cw_known=(l_keying(spec), False), outer=1)
    g = LT.genie_track_l(chan, spec, sim.truth, F0)
    return (_data_sinr_db(tra, spec, a), _data_sinr_db(trb, spec, a),
            _data_sinr_db(g, spec, a))


@pytest.mark.slow
@pytest.mark.parametrize("preset,loss_db", [("steady", 0.2), ("quiet", 0.7), ("moderate", 1.8)])
def test_tracker_against_the_genie_on_l_short(preset, loss_db):
    """Round A within loss_db of the genie (true timing and gain); round B
    (header and keying known) never worse than round A, and better where
    the gain moves."""
    a_db, b_db, g_db = _round_ab(preset)
    assert a_db > g_db - loss_db, (a_db, g_db)
    assert b_db > a_db - 0.05, (a_db, b_db)
    if preset == "moderate":
        assert b_db > a_db + 0.05, (a_db, b_db)


@pytest.mark.slow
def test_pilot_path_follows_warmup():
    """A 1 Hz/min oscillator settling with tau = 60 s, l-short at -12 dB, from
    the acquisition's straight line: the pilot's measured path within 0.05 Hz."""
    cfg = (("drift_hz_per_min", 1.0), ("drift_tau_s", 60.0))
    spec, a, sim, _ = _channel("l-short", cfg=cfg)
    prep = RX.prepare(frontend.Capture(sim.fe, Q_TEST, sim.t0_index))
    f = np.asarray(sim.truth.f_hz)
    t = np.asarray(sim.truth.t_pos_s)
    d0 = (f[50] - f[0]) / (t[50] - t[0])
    ts = np.array([-10.0, 0.0, 600.0])
    line = FreqPath(t_s=ts, f_hz=f[0] + d0 * ts, weight=np.ones(3))
    det = _det(path=line, f0=float(f[0]))
    chan = RX.channel_for(prep, spec, det)
    v = LT.verify_l(chan, CH_FS, spec, det)
    assert v is not None
    tr = LT.track_l(chan, spec, v)
    assert np.sqrt(np.mean((tr.freq(t) - f) ** 2)) < 0.05

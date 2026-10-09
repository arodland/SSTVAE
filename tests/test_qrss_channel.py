"""QRSSTVAE channel simulator, design section 4, acceptance C-1 to C-4.

- C-1: AWGN in the `hfchannel.awgn` convention -- going out to audio,
  through `hfchannel.awgn` and back gives the same measured SNR as
  `simulate(snr)`.
- C-2: Watterson taps -- Gaussian Doppler of 2 sigma = spread, unit power;
  truth.u is the gain actually applied.
- C-3: the frequency trajectory (offset, drift, warm-up, OU wander) and
  its integrated phase; the clock time scale (slip and direction, and,
  slow, against `hfchannel.sample_clock_offset`).
- C-4: AGC step response, impulse and crash statistics, full-slot speed
  (slow).
"""

import math
import subprocess
import sys
import time
from fractions import Fraction

import numpy as np
import pytest

from sstvae import config, hfchannel, wavio
from sstvae.qrss import channel as ch
from sstvae.qrss import frame
from sstvae.qrss.channel import Agc, ChannelConfig, Impulses, Neighbour, Trajectory
from sstvae.qrss.constants import FE_FS, T_SYM
from sstvae.qrss.morse import keying_units

from qrss_helpers import Q_TEST, REPO_ROOT

TINY = frame.TINY
CARRIER = 1700.0                       # 200 Hz into the front end


def carrier_sym(spec=TINY):
    """All-zero symbols: CE sends its bare carrier (phi = 0)."""
    return np.zeros(spec.n_sym)


def keyed_slice(sim, margin_s=1.0):
    """Sample range well inside the keyed span (receiver's clock)."""
    sc = sim.truth.time_scale
    a = int(sim.t0_index + margin_s * FE_FS)
    b = int(sim.t0_index + (sim.spec.keyed_end_pos * T_SYM / sc - margin_s) * FE_FS)
    return slice(a, b)


def snr2500_on_carrier(z, f_fe, fs=FE_FS):
    """SNR in 2500 Hz of a pure carrier at f_fe in complex z (rectangular FFT).

    The carrier's bin holds its power; the noise PSD is the mean bin
    power more than 20 Hz from the carrier and within +-1000 Hz.
    """
    n = len(z) - len(z) % fs                    # whole seconds: whole-Hz carriers on a bin
    X = np.fft.fft(z[:n]) / n
    f = np.fft.fftfreq(n, 1 / fs)
    p = np.abs(X) ** 2
    kc = int(np.argmin(np.abs(f - f_fe)))
    noise = (np.abs(f - f_fe) > 20) & (np.abs(f) < 1000)
    nb = np.mean(p[noise])                      # per bin = sigma^2 / n
    s = p[kc] - nb
    return 10 * np.log10(s / (nb * n * 2500 / fs))


# --- C-1 ---------------------------------------------------------------------------------

def test_c1_snr_matches_hfchannel_awgn():
    snr = -3.0
    a = ch.simulate(carrier_sym(), TINY, Q_TEST, ChannelConfig(snr_db=snr, seed=3),
                    carrier_hz=CARRIER)
    clean = ch.simulate(carrier_sym(), TINY, Q_TEST, ChannelConfig(), carrier_hz=CARRIER)
    audio = hfchannel.awgn(ch.fe_to_audio(clean.fe), snr, seed=5)
    b = ch.audio_to_fe(audio)
    sl = keyed_slice(a)
    snr_a = snr2500_on_carrier(a.fe[sl], CARRIER - 1500)
    snr_b = snr2500_on_carrier(b[sl], CARRIER - 1500)
    assert abs(snr_a - snr) < 0.1, snr_a
    assert abs(snr_b - snr_a) < 0.1, (snr_a, snr_b)
    # and the recorded noise is what was added
    assert a.truth.signal_power == pytest.approx(1.0, abs=1e-3)
    assert a.truth.noise_var_fe == pytest.approx(
        FE_FS / config.SNR_REF_BW_HZ * 10 ** (-snr / 10), rel=2e-3)
    resid = a.fe[sl] - clean.fe[sl]
    assert np.mean(np.abs(resid) ** 2) == pytest.approx(a.truth.noise_var_fe, rel=0.02)


def test_c1_frontend_round_trip_preserves_power():
    clean = ch.simulate(carrier_sym(), TINY, Q_TEST, ChannelConfig(), carrier_hz=CARRIER)
    audio = ch.fe_to_audio(clean.fe)
    assert len(audio) == 2 * len(clean.fe)
    sl = keyed_slice(clean)
    assert np.mean(audio[2 * sl.start:2 * sl.stop] ** 2) == pytest.approx(1.0, abs=0.01)
    back = ch.audio_to_fe(audio)
    err = back[sl] - clean.fe[sl]
    assert 10 * np.log10(np.mean(np.abs(err) ** 2)) < -50


# --- C-2 ---------------------------------------------------------------------------------

@pytest.mark.parametrize("spread,fs,dur", [(0.1, 50, 20000.0), (1.0, FE_FS, 600.0),
                                           (2.0, FE_FS, 300.0)])
def test_c2_tap_doppler_is_gaussian(spread, fs, dur):
    rng = np.random.default_rng(11)
    g = ch.gaussian_taps(int(fs * dur), spread, fs, rng)
    assert np.mean(np.abs(g) ** 2) == pytest.approx(1.0, abs=0.05)
    S = np.abs(np.fft.fft(g)) ** 2
    f = np.fft.fftfreq(len(g), 1 / fs)
    rms_bw = math.sqrt(np.sum(f ** 2 * S) / np.sum(S))
    assert rms_bw == pytest.approx(spread / 2, rel=0.10)     # Gaussian: rms = sigma = spread/2
    # Gaussian shape, not just its second moment: power within +-sigma
    inside = np.sum(S[np.abs(f) <= spread / 2]) / np.sum(S)
    assert inside == pytest.approx(0.6827, abs=0.07)


def test_c2_truth_u_is_the_applied_gain():
    """Noiseless pure carrier: fe at a position's sample equals u * nominal carrier."""
    cfg = ChannelConfig(preset="disturbed", freq_offset_hz=7.0, drift_hz_per_min=2.0,
                        wander_rms_hz=0.3, seed=4)
    sim = ch.simulate(carrier_sym(), TINY, Q_TEST, cfg, carrier_hz=CARRIER)
    assert sim.t0_index == int(sim.t0_index)
    # every 4th position is centred on an FE sample (T = 121.25 samples)
    p = np.arange(40, TINY.n_pos - 40, 4)
    n = (sim.t0_index + np.rint(p * T_SYM * FE_FS)).astype(np.int64)
    nominal = np.exp(2j * np.pi * ch.nominal_turns(CARRIER - 1500, n, FE_FS, sim.t0_index))
    got = sim.fe[n] / nominal
    err = np.abs(got - sim.truth.u[p])
    assert np.max(err) < 2e-4 * np.max(np.abs(sim.truth.u)), np.max(err)
    pw = np.mean(np.abs(sim.truth.u) ** 2)
    assert 0.5 < pw < 1.5                       # one short fading realisation
    assert sim.truth.signal_power == pytest.approx(
        np.mean(np.abs(sim.fe[keyed_slice(sim, 0.2)]) ** 2), rel=0.02)


def test_c2_steady_is_unit_gain():
    cfg = ChannelConfig(preset="steady", freq_offset_hz=-3.25, seed=1)
    sim = ch.simulate(carrier_sym(), TINY, Q_TEST, cfg, carrier_hz=CARRIER)
    np.testing.assert_allclose(np.abs(sim.truth.u), 1.0, atol=1e-12)
    sl = keyed_slice(sim)
    np.testing.assert_allclose(np.abs(sim.fe[sl]), 1.0, atol=1e-6)


# --- C-3 ---------------------------------------------------------------------------------

def test_c3_offset_and_linear_drift():
    cfg = ChannelConfig(freq_offset_hz=-123.5, drift_hz_per_min=1.5)
    sim = ch.simulate(carrier_sym(), TINY, Q_TEST, cfg, carrier_hz=CARRIER)
    t = sim.truth.t_pos_s
    np.testing.assert_allclose(sim.truth.f_hz, CARRIER - 123.5 + 1.5 * t / 60, atol=1e-9)
    np.testing.assert_allclose(t, np.arange(TINY.n_pos) * T_SYM, atol=1e-12)
    # integrated phase against exact rational arithmetic, at 30 minutes
    tr = Trajectory(-123.5, 1.5)
    for ts in (0.03125, 600.0, 1800.0, -9.5):
        exact = Fraction(-123.5) * Fraction(ts) + Fraction(3, 2) / 120 * Fraction(ts) ** 2
        assert abs(float(tr.theta(ts)) - float(exact)) < 1e-9


def test_c3_warm_up_drift():
    tau, D, lead = 300.0, 1.0, 10.0
    tr = Trajectory(0.0, D, tau, t_key=-8 * T_SYM - lead)
    t = np.array([-8 * T_SYM - lead, 0.0, 300.0, 3000.0])
    f = tr.f(t)
    assert f[0] == pytest.approx(0.0, abs=1e-12)                     # from key-up
    assert f[-1] == pytest.approx(D / 60 * tau, rel=1e-4)            # settles
    # initial rate is D Hz/min
    assert (tr.f(t[0] + 1e-3) - f[0]) / 1e-3 == pytest.approx(D / 60, rel=1e-3)
    # theta is the integral of f
    tt = np.linspace(-20.0, 1800.0, 2_000_001)
    th = tr.theta(tt)
    num = np.concatenate([[0], np.cumsum(0.5 * (tr.f(tt)[1:] + tr.f(tt)[:-1]) * np.diff(tt))])
    num -= np.interp(0.0, tt, num)
    assert np.max(np.abs(th - num)) < 1e-6
    assert tr.theta(0.0) == 0.0


def test_c3_ou_wander_statistics():
    sigma, tau = 0.3, 10.0
    tr = Trajectory(wander_rms_hz=sigma, wander_tau_s=tau, t_start=0.0, t_end=40000.0,
                    rng=np.random.default_rng(2))
    f = tr.wander[2]
    dt = tr.wander[1]
    assert np.std(f) == pytest.approx(sigma, rel=0.10)
    lag = int(round(tau / 2 / dt))
    rho = np.corrcoef(f[:-lag], f[lag:])[0, 1]
    assert -lag * dt / math.log(rho) == pytest.approx(tau, rel=0.10)


def test_c3_wander_phase_is_the_exact_integral():
    """theta of the piecewise-linear wander equals the trapezoid sum on a
    grid containing the knots (which is exact for it) to 1e-9 turns."""
    tr = Trajectory(wander_rms_hz=0.5, wander_tau_s=3.0, t_start=-12.0, t_end=620.0,
                    rng=np.random.default_rng(5))
    t = -12.0 + np.arange(int(620 * FE_FS)) / FE_FS     # 200 samples per knot
    f = tr.f(t)
    trap = np.concatenate([[0.0], np.cumsum(0.5 * (f[1:] + f[:-1]) / FE_FS)])
    i0 = int(12 * FE_FS)                                # t = 0
    trap -= trap[i0]
    assert np.max(np.abs(tr.theta(t) - trap)) < 1e-9


def test_c3_signal_follows_trajectory():
    """The front end's carrier frequency is the trajectory's (its phase
    against truth.u is checked sample-exactly in C-2)."""
    cfg = ChannelConfig(freq_offset_hz=41.0, drift_hz_per_min=-3.0, wander_rms_hz=0.5,
                        wander_tau_s=3.0, seed=8)
    sim = ch.simulate(carrier_sym(), TINY, Q_TEST, cfg, carrier_hz=CARRIER)
    sl = keyed_slice(sim)
    n = np.arange(sl.start, sl.stop)
    nominal = np.exp(2j * np.pi * ch.nominal_turns(CARRIER - 1500, n, FE_FS, sim.t0_index))
    ph = np.unwrap(np.angle(sim.fe[sl] / nominal)) / (2 * np.pi)
    t_n = (n - sim.t0_index) / FE_FS
    # instantaneous frequency
    finst = np.diff(ph) * FE_FS
    fref = np.interp(t_n[:-1] + 0.5 / FE_FS, sim.truth.t_pos_s, sim.truth.f_hz) - CARRIER
    # linear interpolation of f between positions (30 ms) against 20 Hz OU knots
    assert np.max(np.abs(finst - fref)) < 0.1
    assert np.mean(np.abs(finst - fref)) < 0.01


def _ramp_end_crossing(sim):
    """Receive time (s after t0) where |fe| falls through 0.5 at the end of keying."""
    a = np.abs(sim.fe).astype(np.float64)
    i = int(np.nonzero(a > 0.5)[0][-1])
    frac = (a[i] - 0.5) / (a[i] - a[i + 1])
    return (i + frac - sim.t0_index) / FE_FS


@pytest.mark.parametrize("tx_ppm,rx_ppm", [(100.0, -100.0), (-100.0, 100.0), (100.0, 100.0),
                                           (250.0, 0.0)])
def test_c3_clock_time_scale(tx_ppm, rx_ppm):
    cfg = ChannelConfig(tx_ppm=tx_ppm, rx_ppm=rx_ppm)
    sim = ch.simulate(carrier_sym(), TINY, Q_TEST, cfg, carrier_hz=CARRIER)
    scale = (1 + tx_ppm * 1e-6) / (1 + rx_ppm * 1e-6)
    assert sim.truth.time_scale == pytest.approx(scale, rel=1e-15)
    end_tx = TINY.keyed_end_pos * T_SYM + 0.025          # middle of the 50 ms ramp
    t_meas = _ramp_end_crossing(sim)
    assert abs(t_meas - end_tx / scale) < 1e-6           # ppm * duration of slip, to 1 us
    if scale != 1.0:
        assert abs(end_tx / scale - end_tx) > 2e-3        # and it is a real slip
    # a fast transmitter arrives early and high: carrier scales the same way
    np.testing.assert_allclose(sim.truth.t_pos_s, np.arange(TINY.n_pos) * T_SYM / scale,
                               rtol=1e-14)
    np.testing.assert_allclose(sim.truth.f_hz, CARRIER * scale, rtol=1e-12)


def test_ppm_direction_matches_hfchannel():
    """sample_clock_offset(x, +ppm) shortens x: a fast far-end clock is tx_ppm > 0."""
    x = np.zeros(80000)
    assert len(hfchannel.sample_clock_offset(x, 100.0)) < len(x)
    assert ChannelConfig(tx_ppm=100.0).time_scale > 1
    assert ChannelConfig(rx_ppm=100.0).time_scale < 1


@pytest.mark.slow
def test_c3_slow_time_scale_against_sample_clock_offset():
    """One ~650 s pass: simulate's exact time scale equals band-limited
    resampling of the clean audio (after removing the different scaling
    origin: sample 0 there, t0 here)."""
    spec = frame.MEDIUM
    rng = np.random.default_rng(1)
    sym = frame.assemble(spec, rng.integers(0, 2, 2474, dtype=np.uint8),
                         rng.standard_normal(spec.n_data))
    key = keying_units("K1ABC")
    ppm = 100.0
    s1 = ch.simulate(sym, spec, Q_TEST, ChannelConfig(tx_ppm=ppm), keying=key,
                     carrier_hz=CARRIER)
    s0 = ch.simulate(sym, spec, Q_TEST, ChannelConfig(), keying=key, carrier_hz=CARRIER,
                     post_s=10.0)
    # pad so len/(1 + ppm) is a whole number: sample_clock_offset rounds the
    # output length, which would otherwise shift its ratio by ~0.1 ppm
    clean = ch.fe_to_audio(s0.fe)
    clean = np.concatenate([clean, np.zeros(-len(clean) % 10001)])
    audio = hfchannel.sample_clock_offset(clean, ppm)
    assert len(audio) * 10001 == len(clean) * 10000
    ref = ch.audio_to_fe(audio).astype(np.complex128)
    # ref sample i is at tx time -pre + i*scale/fs; simulate's j at (-pre + j/fs)*scale,
    # so ref[i] is simulate's j = i + pre*(scale - 1)*fs/scale: delay ref by that.
    scale = 1 + ppm * 1e-6
    delta = 12.0 * (scale - 1) * FE_FS / scale
    F = np.fft.fft(ref)
    f = np.fft.fftfreq(len(ref))
    ref = np.fft.ifft(F * np.exp(-2j * np.pi * f * delta))[:len(s1.fe)]
    sl = keyed_slice(s1, 5.0)
    # one constant carrier phase: the reference scales the carrier about sample 0
    ph = np.angle(np.vdot(ref[sl], s1.fe[sl]))
    err = s1.fe[sl] - ref[sl] * np.exp(1j * ph)
    assert 10 * np.log10(np.mean(np.abs(err) ** 2)) < -40


# --- C-4 ---------------------------------------------------------------------------------

def test_c4_agc_step_response():
    fs = FE_FS
    agc = Agc(attack_ms=2.0, decay_ms=200.0, smooth_ms=0.25)   # one-sample smoothing
    x = np.concatenate([np.ones(fs), 10 * np.ones(fs), np.ones(2 * fs)])
    env = ch.agc_envelope(x, fs, agc)
    # attack: 63% of the step after attack_ms
    rise = (env[fs:fs + 200] - 1) / 9
    t63 = np.interp(1 - math.exp(-1), rise, np.arange(200) / fs)
    assert t63 == pytest.approx(agc.attack_ms * 1e-3, rel=0.15)
    # decay: exponential with decay_ms
    d = env[2 * fs:2 * fs + int(0.3 * fs)]
    tt = np.arange(len(d)) / fs
    keep = (tt > 0.02) & (d > 1.5)
    slope = np.polyfit(tt[keep], np.log(d[keep]), 1)[0]
    assert -1 / slope == pytest.approx(agc.decay_ms * 1e-3, rel=0.05)
    assert env[-1] == pytest.approx(1.0, rel=1e-3)
    # chunking does not change it
    det = ch._AgcDetector(fs, agc)
    pieces = np.concatenate([det(x[i:i + 777]) for i in range(0, len(x), 777)])
    np.testing.assert_allclose(pieces, env, rtol=1e-9)


def test_c4_agc_in_simulate_is_a_recorded_gain():
    base = ChannelConfig(snr_db=0.0, impulses=Impulses(crash_per_min=6.0), seed=9)
    from dataclasses import replace
    a = ch.simulate(carrier_sym(), TINY, Q_TEST, base, carrier_hz=CARRIER)
    b = ch.simulate(carrier_sym(), TINY, Q_TEST, replace(base, agc=Agc()), carrier_hz=CARRIER)
    assert a.truth.agc_gain is None and b.truth.agc_gain is not None
    np.testing.assert_allclose(b.fe, a.fe * b.truth.agc_gain, rtol=1e-5, atol=1e-6)
    assert np.all(np.isfinite(b.fe))
    # the gain dips during the crashes: what makes a fast AGC harmful
    g = b.truth.agc_gain
    assert np.min(g) < 0.1 * np.median(g)


def test_c4_impulse_and_crash_statistics():
    fs, dur = FE_FS, 600.0
    imp = Impulses(rate_hz=10.0, amp_db=30.0, amp_sd_db=6.0, tau_ms=1.0, crash_per_min=2.0)
    x, ev = ch.impulse_noise(int(fs * dur), fs, 1.0, imp, np.random.default_rng(4))
    clicks = np.array([lv for _, lv, k in ev if k == "click"])
    crashes = np.array([lv for _, lv, k in ev if k == "crash"])
    lam = imp.rate_hz * dur
    assert abs(len(clicks) - lam) < 4 * math.sqrt(lam)
    lam_c = imp.crash_per_min * dur / 60
    assert abs(len(crashes) - lam_c) < 4 * math.sqrt(lam_c) + 1
    assert np.mean(clicks) == pytest.approx(imp.amp_db, abs=0.3)
    assert np.std(clicks) == pytest.approx(imp.amp_sd_db, rel=0.05)
    assert np.all((crashes >= 30) & (crashes <= 60))
    # the waveform carries the configured energy: clicks only
    xc, evc = ch.impulse_noise(int(fs * dur), fs, 1.0, Impulses(rate_hz=10.0),
                               np.random.default_rng(6))
    per = 1 / (1 - math.exp(-2 / (1e-3 * fs)))           # sum of exp(-2k/(tau fs))
    want = sum(10 ** (lv / 10) for _, lv, _ in evc) * per
    got = np.sum(np.abs(xc) ** 2)
    assert got / want == pytest.approx(1.0, rel=0.15)


def test_impulses_in_simulate_are_masked():
    cfg = ChannelConfig(snr_db=-10.0, impulses=Impulses(rate_hz=5.0, amp_db=40.0), seed=2)
    sim = ch.simulate(carrier_sym(), TINY, Q_TEST, cfg, carrier_hz=CARRIER)
    m = sim.truth.impulse_mask
    assert 0 < m.mean() < 0.05
    p = np.abs(sim.fe) ** 2
    assert np.median(p[m]) > 30 * np.median(p[~m])
    assert np.all(np.isfinite(sim.fe))


# --- neighbours, lead-in, callsign windows ---------------------------------------------------

def test_neighbour_is_added_at_its_level_and_offset():
    nb = Neighbour(df_hz=50.0, rel_db=10.0, seed=3)
    cfg = ChannelConfig(neighbours=(nb,), seed=1)
    sim = ch.simulate(carrier_sym(), TINY, Q_TEST, cfg, carrier_hz=CARRIER)
    alone = ch.simulate(carrier_sym(), TINY, Q_TEST, ChannelConfig(seed=1), carrier_hz=CARRIER)
    assert sim.truth.signal_power == pytest.approx(alone.truth.signal_power)
    sl = keyed_slice(sim)
    z = sim.fe[sl] - alone.fe[sl]
    assert np.mean(np.abs(z) ** 2) == pytest.approx(10.0, rel=0.02)
    # a CE signal: its carrier line sits at +50 Hz
    n = len(z) - len(z) % 400
    S = np.abs(np.fft.fft(z[:n])) ** 2
    f = np.fft.fftfreq(n, 1 / FE_FS)
    assert f[np.argmax(S)] == pytest.approx(CARRIER - 1500 + 50.0, abs=0.05)


def test_lead_in_and_callsign_windows_pass_through():
    spec = frame.SHORT
    rng = np.random.default_rng(0)
    sym = frame.assemble(spec, rng.integers(0, 2, 2474, dtype=np.uint8),
                         rng.standard_normal(spec.n_data))
    with pytest.raises(ValueError, match="callsign"):
        ch.simulate(sym, spec, Q_TEST, ChannelConfig())
    key = keying_units("K1ABC")
    sim = ch.simulate(sym, spec, Q_TEST, ChannelConfig(), keying=key, lead_in_s=10.0)
    a = np.abs(sim.fe)
    t = (np.arange(len(a)) - sim.t0_index) / FE_FS
    assert np.all(a[(t > -9.9) & (t < -0.3)] > 0.999)   # lead-in carrier is keyed
    assert np.all(a[t < -10.4] == 0)
    # the windows are there (FSK: constant envelope, frequency 16.5 Hz lower)
    p_w = spec.win_start_pos[0]
    t_w = (p_w - 0.5 + 2 * (8 + 0.5)) * T_SYM            # middle of the first key-down unit
    i = int(round(sim.t0_index + t_w * FE_FS))
    dph = np.angle(sim.fe[i + 1] * np.conj(sim.fe[i])) * FE_FS / (2 * np.pi)
    assert dph == pytest.approx(-1 / (2 * T_SYM), abs=1.0)
    with pytest.raises(ValueError, match="pre_s"):
        ch.simulate(sym, spec, Q_TEST, ChannelConfig(), keying=key, lead_in_s=10.0, pre_s=5.0)


def test_simulate_audio_path():
    clean = ch.simulate(carrier_sym(), TINY, Q_TEST, ChannelConfig(), carrier_hz=CARRIER)
    audio = ch.fe_to_audio(clean.fe)
    sim = ch.simulate_audio(audio, TINY, Q_TEST, ChannelConfig(snr_db=-3.0, freq_offset_hz=5.0,
                                                               seed=2), carrier_hz=CARRIER)
    sl = keyed_slice(sim)
    assert snr2500_on_carrier(sim.fe[sl], CARRIER + 5 - 1500) == pytest.approx(-3.0, abs=0.1)
    np.testing.assert_allclose(sim.truth.f_hz, CARRIER + 5.0)


def test_npz_round_trip(tmp_path):
    cfg = ChannelConfig(snr_db=0.0, agc=Agc(), impulses=Impulses(rate_hz=1.0), seed=3)
    sim = ch.simulate(carrier_sym(), TINY, Q_TEST, cfg, carrier_hz=CARRIER)
    p = tmp_path / "s.npz"
    ch.save_npz(p, sim)
    back = ch.load_npz(p)
    assert back.fs == sim.fs and back.q == sim.q and back.spec == sim.spec
    assert back.t0_index == sim.t0_index
    np.testing.assert_array_equal(back.fe, sim.fe)
    for k in ("u", "f_hz", "t_pos_s", "impulse_mask", "agc_gain"):
        np.testing.assert_array_equal(getattr(back.truth, k), getattr(sim.truth, k))
    assert back.truth.noise_var_fe == sim.truth.noise_var_fe


def test_reproducible_from_seed():
    cfg = ChannelConfig(snr_db=-5.0, preset="moderate", wander_rms_hz=0.2,
                        impulses=Impulses(rate_hz=2.0), neighbours=(Neighbour(-60, 3),), seed=7)
    a = ch.simulate(carrier_sym(), TINY, Q_TEST, cfg, carrier_hz=CARRIER)
    b = ch.simulate(carrier_sym(), TINY, Q_TEST, cfg, carrier_hz=CARRIER)
    np.testing.assert_array_equal(a.fe, b.fe)


def test_nominal_turns_exact():
    n = np.array([0, 1, 7_200_000, 123_456_789])
    t = ch.nominal_turns(200.0, n, FE_FS)
    np.testing.assert_array_equal(t, (n * 200 % 4000) / 4000)
    t = ch.nominal_turns(Fraction(1, 64) + 13, n, FE_FS)
    np.testing.assert_array_equal(t, ((n * (13 * 64 + 1)) % (64 * 4000)) / (64 * 4000))


def run_cli(monkeypatch, *argv):
    """qrss_simulate.main() in-process (an interpreter start costs ~1 s)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("qrss_simulate",
                                                  REPO_ROOT / "qrss_simulate.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(sys, "argv", ["qrss_simulate.py", *argv])
    mod.main()


def test_cli(tmp_path):
    clean = ch.simulate(carrier_sym(), TINY, Q_TEST, ChannelConfig(), carrier_hz=CARRIER)
    wav = tmp_path / "tx.wav"
    wavio.write_wav_float(str(wav), ch.fe_to_audio(clean.fe))
    out = tmp_path / "rx.npz"
    r = subprocess.run([sys.executable, str(REPO_ROOT / "qrss_simulate.py"), str(wav),
                        str(out), "--slot", "2025-10-01T00:00Z", "--frame", "tiny",
                        "--snr", "-3", "--offset", "2", "--freq", str(CARRIER)],
                       capture_output=True, text=True, cwd=REPO_ROOT)
    assert r.returncode == 0, r.stderr
    sim = ch.load_npz(out)
    assert sim.q == Q_TEST and sim.spec is TINY
    assert snr2500_on_carrier(sim.fe[keyed_slice(sim)], CARRIER + 2 - 1500) == \
        pytest.approx(-3.0, abs=0.15)


def test_cli_picture_and_beacon_inputs(tmp_path, monkeypatch):
    """.qrsp and .bin are synthesised at baseband with their callsign windows."""
    from sstvae.qrss import beaconfile, picture, tx
    from sstvae.qrss.header import HeaderFields
    from qrss_helpers import synthetic_full_latents

    sp = picture.StoredPicture.from_latents(synthetic_full_latents(3), 0xD1D8, 0)
    qrsp = tmp_path / "p.qrsp"
    picture.save_qrsp(qrsp, sp)
    h = HeaderFields(callsign="K1ABC", grid="FN42", picture_id=sp.picture_id, mode=0,
                     segment=0, codec_id=sp.codec_id)
    binf = tmp_path / "p.bin"
    beaconfile.write(binf, beaconfile.from_picture(sp, 0, h, ook=True))
    common = ["--slot", "2025-10-01T00:00Z", "--frame", "short", "--preset", "quiet",
              "--offset", "3", "--neighbour", "50:-3"]
    run_cli(monkeypatch, str(qrsp), str(tmp_path / "a.npz"), "--callsign", "K1ABC",
            "--grid", "FN42", *common)
    with pytest.raises(SystemExit, match="callsign"):              # windows need a call
        run_cli(monkeypatch, str(qrsp), str(tmp_path / "x.npz"), *common)
    run_cli(monkeypatch, str(binf), str(tmp_path / "b.wav"), "--lead-in", "5", *common)
    a = ch.load_npz(tmp_path / "a.npz")
    assert a.spec is frame.SHORT
    # noiseless and unfaded relative to its own truth: the .qrsp pass is
    # exactly simulate() of tx.slot_symbols with the header's keying
    sym = tx.slot_symbols(sp, Q_TEST, frame.SHORT, segment=0, header=h)
    nb = Neighbour(50.0, -3.0, seed=1)
    ref = ch.simulate(sym, frame.SHORT, Q_TEST,
                      ChannelConfig(preset="quiet", freq_offset_hz=3.0, neighbours=(nb,)),
                      keying=keying_units("K1ABC"))
    np.testing.assert_array_equal(a.fe, ref.fe)
    x = wavio.read_wav(str(tmp_path / "b.wav"))
    assert len(x) == 2 * len(a.fe)                    # same slot span, 8 kHz audio


# --- speed (C-4) -------------------------------------------------------------------------

@pytest.mark.slow
def test_c4_full_slot_under_8_s():
    spec = frame.FULL
    rng = np.random.default_rng(0)
    sym = frame.assemble(spec, rng.integers(0, 2, 2474, dtype=np.uint8),
                         rng.standard_normal(spec.n_data))
    cfg = ChannelConfig(snr_db=-8.0, preset="quiet", freq_offset_hz=250.0,
                        drift_hz_per_min=1.0, wander_rms_hz=0.1, tx_ppm=100.0, rx_ppm=-100.0,
                        impulses=Impulses(rate_hz=10.0), agc=Agc(), seed=1)
    t = time.perf_counter()
    sim = ch.simulate(sym, spec, Q_TEST, cfg, keying=keying_units("K1ABC"), lead_in_s=10.0)
    dt = time.perf_counter() - t
    assert len(sim.fe) > 7_000_000
    assert dt < 8.0, dt


@pytest.mark.parametrize("f", [3400.0, 5000.0, -5.0, 100.0, float("nan")])
def test_carrier_outside_the_band_is_refused(f):
    """Integration review: a carrier receivers do not search (or one past
    Nyquist, which aliased silently) is refused by the simulator and the
    transmitter, not quietly sent somewhere else."""
    from sstvae.qrss import acquire, ce, tx
    from sstvae.qrss.constants import CARRIER_BAND_HZ

    assert acquire.BAND_HZ == CARRIER_BAND_HZ
    with pytest.raises(ValueError, match="audio band"):
        ch.simulate(carrier_sym(), TINY, Q_TEST, ChannelConfig(), carrier_hz=f)
    with pytest.raises(ValueError, match="audio band"):
        ch.simulate_audio(np.zeros(8000), TINY, Q_TEST, ChannelConfig(), carrier_hz=f)
    with pytest.raises(ValueError, match="audio band"):
        tx.transmit_audio(None, Q_TEST, f, spec=TINY)
    for ok in CARRIER_BAND_HZ:
        assert ce.check_carrier(ok) == ok

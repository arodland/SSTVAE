"""The channel simulator on waveform L (design L-receiver 2.2).

`channel.simulate` and `simulate_audio` take an `lframe.LFrameSpec` as
they take a CE frame: the named frequency is the pilot, the clipped
baseband is built once per transmission (`transmit_source`) and served
chunk by chunk, positions are L symbols (`spec.t_sym`), and
`Truth.u_data` is the gain at the data centre, which a delayed second
path makes differ from the pilot's.

- The noiseless pass is `lwave.resample(frame_baseband)` sample for
  sample; transmit audio through `simulate_audio` agrees with it; the
  off-grid source (a clock error) is a band-limited delay to -90 dB.
- SNR in 2500 Hz on a pilot-only frame (a pure tone); P's keying span is
  in receive time on both paths; the clock ramp ends at keyed_end_pos *
  L_T_SYM / scale; a settling drift counts from L's key-up.
- u is the pilot's gain, u_data the data band's (and a genie divides by
  the right one).
- Neighbours of the other waveform: an L neighbour beside CE has its
  one-sided spectrum, a CE one beside L its symmetric one, each at
  rel_db over its own keyed span.
- npz round trip; `qrss_simulate.py --waveform L` in process (from
  transmit audio, and its refusals; from a .qrsp in slow).
- The genie latent SNR on L_SHORT at -12 dB against theory.

Slow: a FULL L slot with every impairment in under 8 s, and
`qrss_simulate.py --waveform L` from a .qrsp exactly `simulate()` of its
symbols (moved from the default tier for time, 2026-10-12).
"""

import importlib.util
import math
import sys
import time

import numpy as np
import pytest

from sstvae import wavio
from sstvae.qrss import channel as ch
from sstvae.qrss import ce, frame, header, lframe, lwave, picture, tx
from sstvae.qrss.channel import Agc, ChannelConfig, Impulses, Neighbour
from sstvae.qrss.constants import FE_FS, L_T_SYM, WAVEFORM_CE, WAVEFORM_L
from sstvae.qrss.precoder import precode

from qrss_fakes import (L_HDR, l_genie, l_keying, l_scenario, l_symbols, make_header,
                        synthetic_picture)
from qrss_helpers import Q_TEST, REPO_ROOT, latent_snr_db, synthetic_full_latents

L_TINY, L_SHORT = lframe.L_TINY, lframe.L_SHORT
PILOT = 1700.0                          # 200 Hz into the front end
A_TONE = lwave.A_PILOT / lwave.A_ID     # a pilot-only frame's amplitude (no clipping)


def pilot_only(spec=L_TINY):
    """All-zero symbols: L sends its bare pilot, a pure tone of amplitude A_TONE."""
    return np.zeros(spec.n_sym, dtype=np.complex128)


def snr2500_on_tone(z, f_fe, fs=FE_FS):
    """SNR in 2500 Hz of a pure tone at f_fe in complex z (as test_qrss_channel's C-1)."""
    n = len(z) - len(z) % fs                    # whole seconds: whole-Hz tones on a bin
    X = np.fft.fft(z[:n]) / n
    f = np.fft.fftfreq(n, 1 / fs)
    p = np.abs(X) ** 2
    kc = int(np.argmin(np.abs(f - f_fe)))
    noise = (np.abs(f - f_fe) > 20) & (np.abs(f) < 1000)
    nb = np.mean(p[noise])
    return 10 * np.log10((p[kc] - nb) / (nb * n * 2500 / fs))


def keyed(sim, margin_s=1.0, spec=None):
    """Sample range inside `spec`'s keyed span (default the sim's), receiver's clock."""
    spec = sim.spec if spec is None else spec
    sc = sim.truth.time_scale
    a = int(sim.t0_index + margin_s * FE_FS)
    b = int(sim.t0_index + (spec.keyed_end_pos * spec.t_sym / sc - margin_s) * FE_FS)
    return slice(a, b)


def genie_sinr_db(sim, a, gain=None):
    """Per-latent SINR of the genie's data symbols against the assembled latents
    (spread copy included), after one LS complex gain; and that gain."""
    spec = sim.spec
    lay = lframe.layout(spec)
    x = lframe.data_latents(spec, precode(a, Q_TEST),
                            header.encode(L_HDR) if spec.has_header else None)
    s = lframe.pair(x)
    y = l_genie(sim, gain=gain)[lay.data]
    g = np.vdot(s, y) / np.vdot(s, s)
    return latent_snr_db(lframe.unpair(y / g), x), g


# --- the source ------------------------------------------------------------------------

def test_noiseless_pass_is_the_resampled_frame():
    """Steady, noiseless, at 1500 Hz: the front end is exactly
    lwave.resample(frame_baseband) on the FE grid, windows and lead-in
    included; the truth is unit gain at L symbol times."""
    sym = l_symbols(L_SHORT, np.random.default_rng(2).standard_normal(L_SHORT.n_data))
    key = l_keying(L_SHORT)
    sim = ch.simulate(sym, L_SHORT, Q_TEST, ChannelConfig(), keying=key, lead_in_s=5.0)
    bb = lwave.frame_baseband(sym, L_SHORT, keying=key, lead_in_s=5.0)
    ref = lwave.resample(bb, FE_FS, -int(sim.t0_index), len(sim.fe))
    np.testing.assert_array_equal(sim.fe, ref.astype(np.complex64))
    assert len(sim.fe) == round((12.0 + L_SHORT.keyed_end_pos * L_T_SYM + 3.0) * FE_FS)
    tr = sim.truth
    np.testing.assert_allclose(tr.t_pos_s, np.arange(L_SHORT.n_pos) * L_T_SYM, rtol=1e-15)
    np.testing.assert_allclose(tr.f_hz, 1500.0)
    np.testing.assert_allclose(tr.u, 1.0)
    np.testing.assert_array_equal(tr.u_data, tr.u)
    # and at another pilot, the same baseband mixed up
    sym = l_symbols(L_TINY, np.random.default_rng(3).standard_normal(L_TINY.n_data))
    sim2 = ch.simulate(sym, L_TINY, Q_TEST, ChannelConfig(), carrier_hz=PILOT)
    ref = lwave.resample(lwave.frame_baseband(sym, L_TINY), FE_FS, -48000, len(sim2.fe))
    n = np.arange(len(sim2.fe))
    nominal = np.exp(2j * np.pi * ch.nominal_turns(PILOT - 1500, n, FE_FS, sim2.t0_index))
    np.testing.assert_allclose(sim2.fe, ref * nominal, rtol=0, atol=1e-6)


def test_simulate_agrees_with_simulate_audio():
    """tx.transmit_audio of an L frame through simulate_audio is simulate of
    tx.slot_symbols, to the 4 kHz resampling filter's -60 dB, scaled by the
    audio amplitude (0.5/sqrt 2 in the front end)."""
    sp = synthetic_picture(0)
    audio = tx.transmit_audio(sp, Q_TEST, 1500, spec=L_TINY)
    sa = ch.simulate_audio(audio, L_TINY, Q_TEST, ChannelConfig())
    sd = ch.simulate(tx.slot_symbols(sp, Q_TEST, L_TINY), L_TINY, Q_TEST, ChannelConfig())
    assert sa.spec is L_TINY and len(sa.fe) == len(sd.fe) and sa.t0_index == sd.t0_index
    sl = keyed(sd, 0.5)
    A, B = sa.fe[sl].astype(np.complex128), sd.fe[sl].astype(np.complex128)
    g = np.vdot(B, A) / np.vdot(B, B)
    assert g == pytest.approx(0.5 / math.sqrt(2), abs=2e-5)
    assert 10 * np.log10(np.mean(np.abs(A - g * B) ** 2) / np.mean(np.abs(A) ** 2)) < -55
    assert sa.truth.signal_power / sd.truth.signal_power == pytest.approx(abs(g) ** 2, rel=1e-3)
    np.testing.assert_allclose(sa.truth.t_pos_s, sd.truth.t_pos_s, atol=1e-12)
    assert sa.truth.u_data is not None


@pytest.mark.parametrize("d", [0.25, 0.731])
def test_off_grid_source_is_a_band_limited_delay(d):
    """The source between FE samples (the clock-error path) equals an FFT
    delay of the frame's 8 kHz resample: linear interpolation at 8 kHz,
    -96 dB measured. On the grid it is lwave.resample itself."""
    sym = l_symbols(L_TINY, np.random.default_rng(1).standard_normal(L_TINY.n_data))
    src = ch.transmit_source(sym, L_TINY)
    bb = lwave.frame_baseband(sym, L_TINY)
    n = int((L_TINY.duration_s + 16) * FE_FS)
    np.testing.assert_array_equal(src(FE_FS, -12.0, n), lwave.resample(bb, FE_FS, -48000, n))
    r8 = lwave.resample(bb, 8000, -96000, 2 * n)
    f = np.fft.fftfreq(2 * n)
    ref = np.fft.ifft(np.fft.fft(r8) * np.exp(2j * np.pi * f * 2 * d))[::2]
    off = src(FE_FS, -12.0 + d / FE_FS, n)
    assert 10 * np.log10(np.mean(np.abs(off - ref) ** 2) / np.mean(np.abs(ref) ** 2)) < -90


def test_the_clipper_runs_once_per_transmission(monkeypatch):
    """The wanted signal and each L neighbour build their clipped frame once,
    not once per chunk (a FULL slot is ~110 chunks of a 0.25 s clipper)."""
    calls = []
    real = lwave.frame_baseband

    def counting(*args, **kw):
        calls.append(args[1].name)
        return real(*args, **kw)
    monkeypatch.setattr(ch.lwave, "frame_baseband", counting)
    nbs = (Neighbour(40.0, -3.0, seed=2, waveform=WAVEFORM_L), Neighbour(-60.0, 0.0, seed=3))
    sim = ch.simulate(pilot_only(), L_TINY, Q_TEST, ChannelConfig(rx_ppm=50.0, neighbours=nbs))
    assert len(sim.fe) > 4 * ch.CHUNK
    assert calls == ["l-tiny", "l-tiny"]


def test_refusals():
    sym = pilot_only(L_SHORT)
    key = l_keying(L_SHORT)
    with pytest.raises(ValueError, match="on-off keying"):
        ch.simulate(sym, L_SHORT, Q_TEST, ChannelConfig(), keying=key, ook=True)
    with pytest.raises(ValueError, match="callsign"):
        ch.simulate(sym, L_SHORT, Q_TEST, ChannelConfig())
    # the data reach 22 Hz below the pilot: 310 Hz puts them under 300
    with pytest.raises(ValueError, match="below"):
        ch.simulate(pilot_only(), L_TINY, Q_TEST, ChannelConfig(), carrier_hz=310.0)
    with pytest.raises(ValueError, match="below"):
        ch.simulate_audio(np.zeros(8000), L_TINY, Q_TEST, ChannelConfig(), carrier_hz=310.0)
    ch.simulate(np.zeros(frame.TINY.n_sym), frame.TINY, Q_TEST, ChannelConfig(),
                carrier_hz=310.0)                                  # CE may
    with pytest.raises(ValueError, match="pre_s"):
        ch.simulate(sym, L_SHORT, Q_TEST, ChannelConfig(), keying=key, lead_in_s=10.0,
                    pre_s=5.0)


# --- levels and clocks ----------------------------------------------------------------------

def test_snr_calibration_on_a_pilot_only_frame():
    """A pilot-only frame is a pure tone: the measured SNR in 2500 Hz is the
    configured one, P is the tone's power, and the noise is what was added."""
    snr = -3.0
    sim = ch.simulate(pilot_only(), L_TINY, Q_TEST, ChannelConfig(snr_db=snr, seed=3),
                      carrier_hz=PILOT)
    clean = ch.simulate(pilot_only(), L_TINY, Q_TEST, ChannelConfig(), carrier_hz=PILOT)
    sl = keyed(sim)
    assert snr2500_on_tone(sim.fe[sl], PILOT - 1500) == pytest.approx(snr, abs=0.1)
    assert sim.truth.signal_power == pytest.approx(A_TONE ** 2, rel=1e-4)
    assert sim.truth.noise_var_fe == pytest.approx(
        A_TONE ** 2 * FE_FS / 2500 * 10 ** (-snr / 10), rel=1e-4)
    resid = sim.fe[sl] - clean.fe[sl]
    assert np.mean(np.abs(resid) ** 2) == pytest.approx(sim.truth.noise_var_fe, rel=0.02)


def test_signal_power_is_over_the_keying_span():
    """P is the clean signal's mean power over [keyed start, keyed end): L's
    envelope nulls count (CE's amplitude rule would drop them). l-tiny's
    hotter preamble puts it about 4-5% above the data's unit power."""
    a, sim = l_scenario("l-tiny", None, 1)
    lo, hi = lwave.keyed_span_pos(L_TINY)
    t = (np.arange(len(sim.fe)) - sim.t0_index) / FE_FS
    on = (t >= lo * L_T_SYM) & (t < hi * L_T_SYM)
    assert sim.truth.signal_power == pytest.approx(
        np.mean(np.abs(sim.fe[on].astype(np.complex128)) ** 2), rel=1e-6)
    assert 1.03 < sim.truth.signal_power < 1.07
    lay = lframe.layout(L_TINY)
    data = (t > lay.pos[L_TINY.n_pre] * L_T_SYM + 1) & (t < hi * L_T_SYM - 1)
    assert np.mean(np.abs(sim.fe[data]) ** 2) == pytest.approx(1.0, abs=0.03)


@pytest.mark.parametrize("audio", [False, True])
def test_signal_power_span_follows_the_clock(audio):
    """With a clock error P's keying span is in receive time on both paths:
    s/scale for synthesis, (pre + s)/scale - pre for transmit audio (which
    is resampled whole from t0 - pre, as truth.t_pos_s is)."""
    sp = synthetic_picture(0)
    cfg = ChannelConfig(tx_ppm=250.0)
    if audio:
        sim = ch.simulate_audio(tx.transmit_audio(sp, Q_TEST, 1500, spec=L_TINY), L_TINY,
                                Q_TEST, cfg)
    else:
        sim = ch.simulate(tx.slot_symbols(sp, Q_TEST, L_TINY), L_TINY, Q_TEST, cfg)
    sc, pre = sim.truth.time_scale, sim.t0_index / FE_FS
    lo, hi = (s * L_T_SYM for s in lwave.keyed_span_pos(L_TINY))
    rx = (lambda s: (pre + s) / sc - pre) if audio else (lambda s: s / sc)
    t = (np.arange(len(sim.fe)) - sim.t0_index) / FE_FS
    on = (t >= rx(lo)) & (t < rx(hi))
    assert sim.truth.signal_power == pytest.approx(
        np.mean(np.abs(sim.fe[on].astype(np.complex128)) ** 2), rel=1e-6)


def test_drift_settles_from_the_l_keying_start():
    """A settling drift counts from key-up, 8 L symbols and the lead-in before
    t0 (CE's 8 symbols are half as long): f = D tau (1 - exp(-(t - t_key)/tau))."""
    d, tau, lead = 3.0, 20.0, 2.0
    sim = ch.simulate(pilot_only(), L_TINY, Q_TEST,
                      ChannelConfig(drift_hz_per_min=d, drift_tau_s=tau), carrier_hz=PILOT,
                      lead_in_s=lead)
    t_key = L_TINY.keyed_start_pos * L_T_SYM - lead
    t = sim.truth.t_pos_s
    np.testing.assert_allclose(sim.truth.f_hz,
                               PILOT + d / 60 * tau * (1 - np.exp(-(t - t_key) / tau)),
                               rtol=0, atol=1e-9)


def _ramp_end_crossing(sim, level):
    """Receive time (s after t0) where |fe| falls through `level` at the end of keying."""
    a = np.abs(sim.fe).astype(np.float64)
    i = int(np.nonzero(a > level)[0][-1])
    frac = (a[i] - level) / (a[i] - a[i + 1])
    return (i + frac - sim.t0_index) / FE_FS


@pytest.mark.parametrize("tx_ppm,rx_ppm", [(100.0, -100.0), (-100.0, 100.0), (100.0, 100.0),
                                           (250.0, 0.0)])
def test_clock_ramp_end(tx_ppm, rx_ppm):
    """The keying's end ramp is crossed at keyed_end_pos * L_T_SYM / scale (to
    1 us), and the truth's times and pilot frequency scale with the clock."""
    cfg = ChannelConfig(tx_ppm=tx_ppm, rx_ppm=rx_ppm)
    sim = ch.simulate(pilot_only(), L_TINY, Q_TEST, cfg, carrier_hz=PILOT)
    scale = (1 + tx_ppm * 1e-6) / (1 + rx_ppm * 1e-6)
    assert sim.truth.time_scale == pytest.approx(scale, rel=1e-15)
    end_tx = L_TINY.keyed_end_pos * L_T_SYM + lwave.KEY_RAMP_S / 2   # the ramp's middle
    assert abs(_ramp_end_crossing(sim, A_TONE / 2) - end_tx / scale) < 1e-6
    np.testing.assert_allclose(sim.truth.t_pos_s, np.arange(L_TINY.n_pos) * L_T_SYM / scale,
                               rtol=1e-14)
    np.testing.assert_allclose(sim.truth.f_hz, PILOT * scale, rtol=1e-12)
    assert len(sim.fe) == round((12.0 + L_TINY.keyed_end_pos * L_T_SYM / scale + 3.0) * FE_FS)


# --- the truth: u at the pilot, u_data at the data centre ------------------------------------

def test_u_is_the_pilot_gain():
    """A pilot-only frame on a faded, drifting channel: the front end at a
    position's sample is A_TONE * u * the nominal pilot (as C-2 for CE)."""
    cfg = ChannelConfig(preset="disturbed", freq_offset_hz=7.0, drift_hz_per_min=2.0,
                        wander_rms_hz=0.3, seed=4)
    sim = ch.simulate(pilot_only(), L_TINY, Q_TEST, cfg, carrier_hz=PILOT)
    p = np.arange(20, L_TINY.n_pos - 20, 2)          # even positions sit on FE samples
    n = (sim.t0_index + np.rint(p * L_T_SYM * FE_FS)).astype(np.int64)
    nominal = np.exp(2j * np.pi * ch.nominal_turns(PILOT - 1500, n, FE_FS, sim.t0_index))
    err = np.abs(sim.fe[n] / nominal - A_TONE * sim.truth.u[p])
    assert np.max(err) < 1e-3 * A_TONE * np.max(np.abs(sim.truth.u))


def test_u_data_is_the_data_band_gain():
    """Two fixed paths 2 ms apart (no Doppler): the data band's LS gain against
    the unfaded pass is u_data, and is 0.11 away from the pilot's u (the
    second path's phase differs by 2 pi f_p 2 ms = 0.16 rad between them)."""
    for seed in (1, 2):
        _, s0 = l_scenario("l-tiny", None, seed)
        _, s1 = l_scenario("l-tiny", None, seed, cfg=(("delay_ms", 2.0),))
        data = lframe.layout(L_TINY).data
        one = np.ones(L_TINY.n_pos)
        m0, m1 = l_genie(s0, gain=one)[data], l_genie(s1, gain=one)[data]
        r = np.vdot(m0, m1) / np.vdot(m0, m0)
        ud, u = s1.truth.u_data, s1.truth.u
        assert np.ptp(np.abs(ud)) < 1e-12               # constant taps
        assert abs(r - ud[0]) < 0.01, (r, ud[0])
        assert abs(r - u[0]) > 0.08, (r, u[0])
    np.testing.assert_array_equal(s0.truth.u_data, s0.truth.u)   # steady: one gain


def test_a_genie_needs_u_data_on_disturbed():
    """Noiseless on 'disturbed': dividing by u_data keeps the clipper's SINR
    (~12 dB) to within 1 dB; dividing by the pilot's u loses 3 dB here."""
    a, sim = l_scenario("l-tiny", None, 2, cfg=(("preset", "disturbed"),))
    s_data, _ = genie_sinr_db(sim, a)
    s_u, _ = genie_sinr_db(sim, a, gain=sim.truth.u)
    assert s_data > 10.8, s_data
    assert s_data > s_u + 2.0, (s_data, s_u)


def test_ce_truth_has_no_u_data():
    sim = ch.simulate(np.zeros(frame.TINY.n_sym), frame.TINY, Q_TEST,
                      ChannelConfig(preset="quiet"), carrier_hz=PILOT)
    assert sim.truth.u_data is None


# --- neighbours -------------------------------------------------------------------------------

def _band_split_db(z, f0):
    """10 log10 of the power 3-22 Hz below f0 over the power 3-22 Hz above,
    and the strongest bin's frequency."""
    n = len(z) - len(z) % FE_FS
    S = np.abs(np.fft.fft(z[:n])) ** 2
    f = np.fft.fftfreq(n, 1 / FE_FS)
    below = S[(f >= f0 - 22) & (f <= f0 - 3)].sum()
    above = S[(f >= f0 + 3) & (f <= f0 + 22)].sum()
    return 10 * np.log10(below / above), f[np.argmax(S)]


def _neighbour_alone(sim, alone, nb_spec):
    """The neighbour's samples over its own keying span (no clock error)."""
    lo, hi = (lwave if nb_spec.waveform == WAVEFORM_L else ce).keyed_span_pos(nb_spec)
    i0 = int(sim.t0_index)
    sl = slice(i0 + math.ceil(lo * nb_spec.t_sym * FE_FS),
               i0 + math.ceil(hi * nb_spec.t_sym * FE_FS))
    return (sim.fe[sl] - alone.fe[sl]).astype(np.complex128)


def test_an_l_neighbour_beside_ce():
    """An L neighbour of a CE pass uses CE's twin frame ("tiny" -> "l-tiny"):
    its pilot line sits at df, its data only below it, and it has rel_db of
    the wanted power over its own keyed span."""
    cs = np.zeros(frame.TINY.n_sym)
    nb = Neighbour(60.0, 3.0, seed=3, waveform=WAVEFORM_L)
    alone = ch.simulate(cs, frame.TINY, Q_TEST, ChannelConfig(seed=1), carrier_hz=PILOT)
    sim = ch.simulate(cs, frame.TINY, Q_TEST, ChannelConfig(seed=1, neighbours=(nb,)),
                      carrier_hz=PILOT)
    assert sim.truth.signal_power == alone.truth.signal_power
    assert sim.truth.u_data is None                              # the wanted is CE
    z = _neighbour_alone(sim, alone, lframe.twin(frame.TINY, WAVEFORM_L))
    assert np.mean(np.abs(z) ** 2) == pytest.approx(
        sim.truth.signal_power * 10 ** 0.3, rel=0.01)
    split, f_peak = _band_split_db(z, PILOT - 1500 + 60)
    assert f_peak == pytest.approx(PILOT - 1500 + 60, abs=0.05)
    assert split > 30, split


def test_a_ce_neighbour_beside_l():
    """And the other way: a CE neighbour of an L pass is symmetric about its
    carrier, at rel_db, while the L pass itself is one-sided."""
    nb = Neighbour(-60.0, 0.0, seed=3, waveform=WAVEFORM_CE)
    _, alone = l_scenario("l-tiny", None, 1, PILOT)
    _, sim = l_scenario("l-tiny", None, 1, PILOT, cfg=(("neighbours", (nb,)),))
    z = _neighbour_alone(sim, alone, frame.TINY)
    assert np.mean(np.abs(z) ** 2) == pytest.approx(sim.truth.signal_power, rel=0.01)
    split, f_peak = _band_split_db(z, PILOT - 1500 - 60)
    assert f_peak == pytest.approx(PILOT - 1500 - 60, abs=0.05)
    assert abs(split) < 1.0, split
    own, _ = _band_split_db(alone.fe[keyed(alone)].astype(np.complex128), PILOT - 1500)
    assert own > 30, own


# --- files and the CLI ------------------------------------------------------------------------

def test_npz_round_trip(tmp_path):
    _, sim = l_scenario("l-tiny", -6.0, 3, cfg=(("agc", Agc()), ("preset", "quiet")))
    p = tmp_path / "l.npz"
    ch.save_npz(p, sim)
    back = ch.load_npz(p)
    assert back.spec is L_TINY and back.q == sim.q and back.t0_index == sim.t0_index
    np.testing.assert_array_equal(back.fe, sim.fe)
    for k in ("u", "u_data", "f_hz", "t_pos_s", "impulse_mask", "agc_gain"):
        np.testing.assert_array_equal(getattr(back.truth, k), getattr(sim.truth, k))
    assert back.truth.signal_power == sim.truth.signal_power
    # a CE file has no u_data and loads without one
    ce = ch.simulate(np.zeros(frame.TINY.n_sym), frame.TINY, Q_TEST, ChannelConfig())
    ch.save_npz(tmp_path / "ce.npz", ce)
    assert ch.load_npz(tmp_path / "ce.npz").truth.u_data is None


def _load_cli():
    spec = importlib.util.spec_from_file_location("qrss_simulate",
                                                  REPO_ROOT / "qrss_simulate.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_cli(monkeypatch, *argv):
    """qrss_simulate.main() in process (an interpreter start costs ~1 s)."""
    mod = _load_cli()
    monkeypatch.setattr(sys, "argv", ["qrss_simulate.py", *argv])
    mod.main()


@pytest.mark.slow
def test_cli_waveform_l(tmp_path, monkeypatch):
    """--waveform L: a .qrsp is exactly simulate() of tx.slot_symbols with the
    header's keying, and the neighbour suffix picks its waveform."""
    sp = picture.StoredPicture.from_latents(synthetic_full_latents(3), 0xD1D8, 0)
    qrsp = tmp_path / "p.qrsp"
    picture.save_qrsp(qrsp, sp)
    h = make_header(sp)
    common = ["--slot", "2025-10-01T00:00Z", "--frame", "short", "--waveform", "L",
              "--offset", "3", "--neighbour", "50:-3:CE"]
    run_cli(monkeypatch, str(qrsp), str(tmp_path / "a.npz"), "--callsign", h.callsign,
            "--grid", h.grid, *common)
    a = ch.load_npz(tmp_path / "a.npz")
    assert a.spec is L_SHORT and a.truth.u_data is not None
    sym = tx.slot_symbols(sp, Q_TEST, L_SHORT, segment=0, header=h)
    nb = Neighbour(50.0, -3.0, seed=1, waveform=WAVEFORM_CE)
    ref = ch.simulate(sym, L_SHORT, Q_TEST,
                      ChannelConfig(freq_offset_hz=3.0, neighbours=(nb,)),
                      keying=l_keying(L_SHORT, h.callsign))
    np.testing.assert_array_equal(a.fe, ref.fe)


def test_cli_waveform_l_from_transmit_audio(tmp_path, monkeypatch):
    """An L transmit WAV is read as an L frame, its pilot at --freq."""
    sp = synthetic_picture(0)
    wav = tmp_path / "tx.wav"
    wavio.write_wav_float(str(wav), tx.transmit_audio(sp, Q_TEST, PILOT, spec=L_TINY))
    run_cli(monkeypatch, str(wav), str(tmp_path / "b.npz"), "--slot", "2025-10-01T00:00Z",
            "--frame", "tiny", "--waveform", "L", "--freq", str(PILOT), "--snr", "-3")
    b = ch.load_npz(tmp_path / "b.npz")
    assert b.spec is L_TINY
    np.testing.assert_allclose(b.truth.f_hz, PILOT)


def test_cli_refusals_and_neighbour_syntax(tmp_path, monkeypatch):
    from sstvae.qrss import beaconfile

    sp = synthetic_picture(1)
    binf = tmp_path / "p.bin"
    beaconfile.write(binf, beaconfile.from_picture(sp, 0, make_header(sp)))
    args = ["--slot", "2025-10-01T00:00Z", "--frame", "tiny", "--waveform", "L"]
    with pytest.raises(SystemExit, match="beacon file"):
        run_cli(monkeypatch, str(binf), str(tmp_path / "x.npz"), *args)
    with pytest.raises(SystemExit, match="--ook"):
        run_cli(monkeypatch, str(binf), str(tmp_path / "x.npz"), *args, "--ook")
    mod = _load_cli()
    assert mod.parse_neighbour("50:-3") == ((50.0, -3.0), None)
    assert mod.parse_neighbour("50:-3:L") == ((50.0, -3.0), WAVEFORM_L)
    assert mod.parse_neighbour("1:2:3:4:ce") == ((1.0, 2.0, 3.0, 4.0), WAVEFORM_CE)
    for bad in ("50", "50:L", "a:b", "1:2:3:4:5", "50:-3:X"):
        with pytest.raises(Exception, match="DF:DB"):
            mod.parse_neighbour(bad)


# --- the genie ------------------------------------------------------------------------------

G_L_DB = 20.71                          # 10 log10(0.881^2 * 2500 * L_T_SYM), map_sim
SINR_CLIP_DB = 12.2                     # the clipper's own SINR (noiseless genie)


def test_genie_latent_snr_on_l_short_at_minus_12_db():
    """True timing and the true data-band gain on L_SHORT at -12 dB: the
    per-latent SNR against the assembled latents is
    -10 log10(10^(-(S + 20.71)/10) + 10^(-12.2/10)) to 0.5 dB, and the data
    arrive at a gain of about 0.89 (the clipper's, spec 2.2)."""
    snr = -12.0
    a, sim = l_scenario("l-short", snr, 1)
    s, g = genie_sinr_db(sim, a)
    theory = -10 * np.log10(10 ** (-(snr + G_L_DB) / 10) + 10 ** (-SINR_CLIP_DB / 10))
    assert s == pytest.approx(theory, abs=0.5), (s, theory)
    assert abs(g) == pytest.approx(0.89, abs=0.02)
    assert abs(np.angle(g)) < 0.03


# --- slow ---------------------------------------------------------------------------------

@pytest.mark.slow
def test_l_full_slot_under_8_s():
    """A FULL L slot with every impairment, as test_qrss_channel's C-4 for CE."""
    spec = lframe.L_FULL
    sym = l_symbols(spec, np.random.default_rng(0).standard_normal(spec.n_data))
    cfg = ChannelConfig(snr_db=-8.0, preset="quiet", freq_offset_hz=250.0,
                        drift_hz_per_min=1.0, wander_rms_hz=0.1, tx_ppm=100.0, rx_ppm=-100.0,
                        impulses=Impulses(rate_hz=10.0), agc=Agc(), seed=1)
    t = time.perf_counter()
    sim = ch.simulate(sym, spec, Q_TEST, cfg, keying=l_keying(spec), lead_in_s=10.0)
    dt = time.perf_counter() - t
    assert len(sim.fe) > 7_000_000
    assert dt < 8.0, dt

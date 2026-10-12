"""The receiver's shared pieces in either waveform's units (L receiver, F1).

Positions are in the frame's own symbols (`spec.t_sym`: L's is twice
CE's), a Morse unit is `spec.cw_window // CW_UNITS` of them, and CE's
model refuses an L frame. Each check here is one a CE constant would
fail silently with plausible numbers:

- durations: `live.frame_seconds`, A3's frame starts, a hint's path and
  `receiver.channel_for` span L_FULL's 1782.65 s (CE's T would give 891);
- `rx_index` / `_tau_of_rx` are inverses at L's period, and
  `TrackResult.t_pos_s` / `doppler_hz` follow the spec's;
- `mf_table`: CE's is exactly the old module table, L's is L's pulse
  (unit energy, `lwave.G0`, and `symbol_mf` through it agrees with
  `lwave.matched_filter`); complex taps are applied as they are;
- `q_grid(L_T_SYM)` is 8x CE's q per position for the same spread;
- `_window_unknown` on L: units 8..183, FSK only, the given reach;
- `z_ref_stat` with CE's signature d = j Im(c) given explicitly equals
  CE's own form, and works on an L layout;
- the carrier-line search (`_carrier_lines`, `refine_path`, with
  `ratio` and `med_band`) and `line_path` cover the whole L frame;
  `erase`'s margin, `_recenter`'s and `report`'s frequencies are in L's
  period; `acquire.lead_in_span`'s first window ends 8 L symbols before
  tau0;
- `cwid` on a synthetic L window (the pilot keyed as `lwave.pilot`
  keys it, one unit per L symbol, in a fake track): `read` gets the
  call, `keying_type` says "fsk" without computing, and the FSK
  amplitude ignores the window's plain units;
- every CE-only entry point raises on an L frame.

Fast: about a second in all.
"""

import math
from fractions import Fraction
from types import SimpleNamespace

import numpy as np
import pytest

from sstvae.qrss import acquire, ce, cwid, demod, frame, lframe, live, lwave
from sstvae.qrss import receiver as RX
from sstvae.qrss import track as T
from sstvae.qrss.constants import (CH_FS, CW_UNITS, FE_FS, L_T_SYM, LEAD_IN_MAX_S, SPAN,
                                   T_SYM, WAVEFORM_L)
from sstvae.qrss.morse import keying_units
from sstvae.qrss.types import CarrierCandidate, FreqPath, Timing

CALL = "K1ABC"


def _timing(tau0=0.0, ppm=0.0, gamma=0.0):
    return Timing(tau0=tau0, ppm=ppm, gamma=gamma, z=0.0, cov=np.zeros((3, 3)))


# --- durations ----------------------------------------------------------------------------

def test_frame_seconds_l_full():
    s = live.frame_seconds(lframe.L_FULL)
    assert s == lframe.L_FULL.keyed_end_pos * L_T_SYM
    assert abs(s - 1782.65) < 0.01
    assert abs(s - live.frame_seconds(frame.FULL)) < 0.05      # the two frames agree in time


def test_slot_spans_in_l_time():
    end_l = acquire.tbd_frame_starts(lframe.L_FULL)[-1]
    assert abs(end_l - acquire.tbd_frame_starts(frame.FULL)[-1]) < 0.5
    acq = acquire.Acquisition(lines=[], preamble=[], tbd=[],
                              hinted=[CarrierCandidate(f_hz=1500.0, drift_hz_per_min=0.0,
                                                       metric=1.0)])
    d = acq.detections(lframe.L_FULL)[0]
    assert d.path.t_s[-1] == lframe.L_FULL.keyed_end_pos * L_T_SYM
    spec = lframe.L_TINY
    prep = RX.Prepared(fe=np.zeros(70 * FE_FS, np.complex64), keep=None, q=0,
                       t0_index=12.0 * FE_FS)
    ch = RX.channel_for(prep, spec, d)
    assert ch.t_s(len(ch.ch) - 1) > spec.keyed_end_pos * L_T_SYM + RX.MARGIN_S
    assert ch.t_s(0) < -(LEAD_IN_MAX_S + SPAN * L_T_SYM) - RX.MARGIN_S


# --- timing -------------------------------------------------------------------------------

def test_rx_index_inverse_at_l():
    n_pos = lframe.L_SHORT.n_pos
    tm = _timing(tau0=37.3, ppm=-80.0, gamma=4.0)
    tau = np.linspace(-30.0, n_pos + 10.0, 2001)
    rel = T.rx_index(tm, tau, n_pos, t_sym=L_T_SYM)
    assert np.max(np.abs(T._tau_of_rx(tm, rel, n_pos, t_sym=L_T_SYM) - tau)) < 1e-9
    # twice CE's samples per position
    one = T.rx_index(_timing(), 100.0, n_pos, t_sym=L_T_SYM)
    assert one == pytest.approx(2 * T.rx_index(_timing(), 100.0, n_pos), rel=1e-15)
    tr = T.TrackResult(spec=lframe.L_SHORT, chan=None, timing=tm, freq=None,
                       pos=np.arange(-SPAN, n_pos), z_d=None, m=None, c=None, u=None, P=None,
                       N=None, psi=None, meas=None, pu=None)
    assert np.array_equal(tr.t_pos_s(), T.rx_index(tm, tr.pos, n_pos, t_sym=L_T_SYM) / CH_FS)


# --- the matched filter -------------------------------------------------------------------

def test_mf_table_ce_is_the_old_table():
    tb = T.mf_table()
    sps = T_SYM * CH_FS
    half = int(math.ceil(SPAN * sps)) + 1
    d = np.arange(-half * 512, half * 512 + 1) / 512
    assert T.mf_table(T_SYM) is tb and tb.t_sym == T_SYM
    assert tb.sps == sps and tb.half == half == 62 and tb.g0 == ce.G0
    assert np.array_equal(tb.tab, ce.pulse(d / sps))


def test_mf_table_l():
    tb = T.mf_table(L_T_SYM)
    assert tb.t_sym == L_T_SYM and tb.sps == L_T_SYM * CH_FS == 15.15625
    assert tb.half == 123 and tb.g0 == lwave.G0
    for ph in (0, 100, 256, 511):             # any sampling phase at the CH rate
        x = tb.tab[ph::T.MF_RES]
        assert abs(np.sum(x * x) / tb.sps - 1.0) < 1e-4       # unit energy
        assert abs(np.sum(x) / tb.sps - tb.g0) < 1e-3          # DC gain
    # the table is L's pulse: symbol_mf at the true timing is lwave's exact MF
    spec = lframe.L_TINY
    rng = np.random.default_rng(1)
    sym = np.exp(2j * np.pi * (rng.integers(0, 4, spec.n_sym) + 0.5) / 4)
    n0 = -400
    z = lwave.data_grid(CH_FS, n0, spec.n_pos * 16 + 800, sym, spec)
    pos = np.arange(0, spec.n_pos, 7)
    m = T.symbol_mf(z, CH_FS, _timing(), pos, t0_index=-n0, n_pos=spec.n_pos, table=tb)
    ref = lwave.matched_filter(z, CH_FS, n0, pos)
    assert np.max(np.abs(m - ref)) < 1e-5
    assert np.max(np.abs(m - lwave.at_positions(sym, spec)[pos])) < 0.02     # Nyquist
    # complex taps are applied as they are (nothing takes a real part)
    tc = T.MfTable(t_sym=tb.t_sym, sps=tb.sps, half=tb.half, tab=(1 + 2j) * tb.tab, g0=tb.g0)
    mc = T.symbol_mf(z, CH_FS, _timing(), pos, t0_index=-n0, n_pos=spec.n_pos, table=tc)
    assert np.allclose(mc, (1 + 2j) * m, rtol=1e-12, atol=1e-14)
    psi = T.psi_at(np.ones(len(z)), _timing(), pos[5:-5], t0_index=-n0, n_pos=spec.n_pos,
                   table=tb)
    assert np.all(psi > 0.999)
    with pytest.raises(ValueError):
        T.MfTable(t_sym=tb.t_sym, sps=tb.sps, half=tb.half, tab=tb.tab[:-1], g0=tb.g0)
    with pytest.raises(ValueError):
        T.mf_table(0.05)


# --- process noise ------------------------------------------------------------------------

def test_q_grid_l_is_8x_ce_per_position():
    gl, gc = T.q_grid(L_T_SYM), T.q_grid()
    assert gc is T.Q_GRID or gc == T.Q_GRID
    assert len(gl) == len(gc) == len(T.Q_SPREADS_HZ)
    for ql, qc, b in zip(gl, gc, T.Q_SPREADS_HZ):
        assert ql == pytest.approx(8 * qc, rel=1e-12)
        assert T.spread_of_q(ql, L_T_SYM) == pytest.approx(b, rel=1e-12)
    tr = T.TrackResult(spec=lframe.L_TINY, chan=None, timing=_timing(), freq=None,
                       pos=np.arange(3), z_d=None, m=None, c=None, u=None, P=None, N=None,
                       psi=None, meas=None, pu=None, q=gl[6])
    assert tr.doppler_hz == pytest.approx(T.Q_SPREADS_HZ[6], rel=1e-12)


def test_window_unknown_l():
    spec = lframe.L_SHORT
    grid = np.arange(-SPAN, spec.n_pos)
    pw = spec.win_start_pos[0]
    for ook, reach in ((True, SPAN), (False, 16)):
        mask = T._window_unknown(spec, grid, ook_possible=ook, reach=reach)
        d = grid - pw
        near = (d > -spec.cw_window) & (d < 2 * spec.cw_window)
        want = (d >= 8 - reach) & (d <= 183 + reach)
        assert np.array_equal(mask[near], want[near])


# --- gate V's statistic -------------------------------------------------------------------

def _zref_fake(spec, known, rng, sig):
    n = spec.n_pos + SPAN
    pos = np.arange(-SPAN, spec.n_pos)
    u = 0.7 * np.exp(1j * np.linspace(0, 3, n))
    c = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    N = np.full(n, 0.5)
    m = sig * u * c + math.sqrt(0.25) * (rng.standard_normal(n) + 1j * rng.standard_normal(n))
    return T.TrackResult(spec=spec, chan=None, timing=_timing(), freq=None, pos=pos, z_d=None,
                         m=m, c=c, u=u, P=np.full(n, 0.01), N=N, psi=np.ones(n), meas=None,
                         pu=None, classes=SimpleNamespace(known=known)), c


def test_z_ref_stat_signature():
    rng = np.random.default_rng(3)
    spec = frame.TINY
    known = np.zeros(spec.n_sym, bool)
    known[frame.layout(spec).known_idx] = True
    tr, c = _zref_fake(spec, known, rng, 1.0)
    z0 = T.z_ref_stat(tr, c)
    assert z0 > 20
    assert T.z_ref_stat(tr, c, 1j * c.imag) == pytest.approx(z0, rel=1e-12)
    # L: its own layout; the signature is the whole template here
    spec = lframe.L_TINY
    known = np.zeros(spec.n_sym, bool)
    known[lframe.layout(spec).known_idx] = True
    tr, c = _zref_fake(spec, known, rng, 1.0)
    assert T.z_ref_stat(tr, c, c) > 20
    tr, c = _zref_fake(spec, known, rng, 0.0)
    assert abs(T.z_ref_stat(tr, c, c)) < 5


# --- paths, erasure, report: the frame in L time -----------------------------------------

PRE_S = 20.0                                   # CH capture before t0


def _tone_chan(spec, theta, sigma2=1.0, band=None, seed=0):
    """A ChanCapture of a unit tone exp(2 pi j theta(t)) (t s after t0; mixed at
    1500 Hz, so 0 Hz in CH) in complex noise of variance sigma2, from PRE_S
    before t0 to 30 s past the frame; band = (lo, hi, var) adds noise there."""
    rng = np.random.default_rng(seed)
    n = int((PRE_S + spec.n_pos * spec.t_sym + 30.0) * CH_FS)
    t = np.arange(n) / CH_FS - PRE_S
    z = np.exp(2j * np.pi * theta(t))
    z = z + math.sqrt(sigma2 / 2) * (rng.standard_normal(n) + 1j * rng.standard_normal(n))
    if band is not None:
        lo, hi, var = band
        w = rng.standard_normal(n) + 1j * rng.standard_normal(n)
        f = np.fft.fftfreq(n, 1.0 / CH_FS)
        W = np.fft.fft(w) * ((f >= lo) & (f <= hi))
        z = z + np.fft.ifft(W) * math.sqrt(var / np.mean(np.abs(np.fft.ifft(W)) ** 2))
    return T.ChanCapture(ch=z.astype(np.complex64), fs=CH_FS, t0_index=PRE_S * CH_FS,
                         f_mix=Fraction(0))


def _flat(f_hz, spec):
    t = np.array([-PRE_S, 0.0, spec.n_pos * spec.t_sym + 10.0])
    return FreqPath(t_s=t, f_hz=np.full(3, float(f_hz)), weight=np.ones(3))


def test_carrier_lines_and_refine_path_span_the_l_frame():
    """The carrier search walks over the whole L frame (CE's T would stop it
    half-way), keeps lines by `ratio` against the median over `med_band`."""
    spec = lframe.L_SHORT
    dur = spec.n_pos * spec.t_sym
    chan = _tone_chan(spec, lambda t: 0.4 * t, band=(50.0, 100.0, 1e4))
    z = T.derotate(chan, T.freq_path_spline(_flat(1500.0, spec)))
    tt, ff, ww = T._carrier_lines(z, chan, spec, 20.0, 2.0)
    assert len(tt) == 13 and tt[-1] > 0.9 * dur              # 20 s segments to 256 s
    np.testing.assert_allclose(ff, 0.4, atol=0.01)
    assert len(T._carrier_lines(z, chan, spec, 20.0, 2.0, ratio=1e6)[0]) == 0
    # the median over the noisy 50..100 Hz band buries the line; 1..18 Hz does not
    assert len(T._carrier_lines(z, chan, spec, 20.0, 2.0, med_band=(50.0, 100.0))[0]) == 0
    assert len(T._carrier_lines(z, chan, spec, 20.0, 2.0, med_band=(1.0, 18.0))[0]) == 13
    p = T.refine_path(chan, spec, _flat(1500.0, spec), med_band=(1.0, 18.0))
    t_end = np.array([0.0, 0.5 * dur, dur - 10.0])
    np.testing.assert_allclose(T.freq_path_spline(p)(t_end), 1500.4, atol=0.01)
    flat = _flat(1500.0, spec)                       # no line found: the path comes back
    assert T.refine_path(chan, spec, flat, med_band=(50.0, 100.0)) is flat


def test_line_path_spans_the_l_frame():
    spec = lframe.L_SHORT
    dur = spec.n_pos * spec.t_sym
    d = 0.2 / 60.0                                            # 0.2 Hz/min
    chan = _tone_chan(spec, lambda t: 0.3 * t + 0.5 * d * t * t)
    p = T.line_path(chan, spec, _flat(1500.3, spec))
    assert p.t_s[-1] >= dur
    t = np.array([0.0, dur / 2, dur])
    np.testing.assert_allclose(np.interp(t, p.t_s, p.f_hz), 1500.3 + d * t, atol=0.01)


def _l_track(spec, delta_hz, chan=None):
    """A fake L TrackResult on a flat 1500 Hz path whose gain rotates at delta_hz."""
    pos = np.arange(-SPAN, spec.n_pos)
    tm = _timing(tau0=3.0)
    t = T.rx_index(tm, pos, spec.n_pos, t_sym=L_T_SYM) / CH_FS
    u = np.exp(2j * np.pi * delta_hz * t)
    n = len(pos)
    return T.TrackResult(spec=spec, chan=chan, timing=tm,
                         freq=T.PathFn(np.array([-30.0, 0.0, 30.0]), np.full(3, 1500.0)),
                         pos=pos, z_d=None, m=None, c=None, u=u, P=np.zeros(n),
                         N=np.ones(n), psi=np.ones(n), meas=np.ones(n, bool), pu=np.ones(n),
                         q=T.q_grid(L_T_SYM)[3], z_ref=9.0)


def test_erase_recenter_report_in_l_time():
    spec = lframe.L_SHORT
    tr = _l_track(spec, 0.05, chan=_tone_chan(spec, lambda t: 0.0 * t))
    t = tr.t_pos_s()
    T.erase(tr, [(10.0, 20.0)])
    np.testing.assert_array_equal(tr.psi == 0, (t > 10.0 - L_T_SYM) & (t < 20.0 + L_T_SYM))
    # the gain's rotation, in Hz at L's period
    _, df = T._recenter(tr)
    assert df == pytest.approx(0.05, rel=1e-6)
    r = T.report(tr)
    assert r.offset_hz == pytest.approx(1500.0) and r.drift_hz_per_min == pytest.approx(0.0)
    assert r.wander_hz_rms == pytest.approx(0.05, rel=1e-3)
    assert r.doppler_hz == pytest.approx(T.Q_SPREADS_HZ[3], rel=1e-12)
    assert r.slip_free and r.z_ref == 9.0


def test_lead_in_span_ends_8_l_symbols_before_tau0():
    """A tone over the second before an L frame's keying start (8 L symbols
    before tau0) is a 1 s lead-in at L's period; at CE's, the 1 s window
    ending 8 CE symbols before tau0 would find another 0.24 s of it."""
    rng = np.random.default_rng(4)
    t0 = 12 * FE_FS
    n = 20 * FE_FS
    t = (np.arange(n) - t0) / FE_FS
    end = -SPAN * L_T_SYM
    sigma2 = FE_FS / 2500 * 10 ** 0.6                        # -6 dB SNR2500
    fe = ((t >= end - 1.0) & (t < end)).astype(np.complex128)
    fe += math.sqrt(sigma2 / 2) * (rng.standard_normal(n) + 1j * rng.standard_normal(n))
    assert acquire.lead_in_span(fe, t0, 1500.0, t_sym=L_T_SYM) == 1.0
    assert acquire.lead_in_span(fe, t0, 1500.0) == 2.0      # CE's period: the window is late


# --- callsign windows ---------------------------------------------------------------------

def _l_window_track(spec, keying, sigma2, rng, junk=0.0):
    """A pilot-derotated L capture (`lwave.pilot`, keyed) in complex noise, as a fake
    track: key-up at 0 Hz, key-down one cycle per L symbol below. `junk` adds that
    amplitude at 0 Hz in units 0..7 and 184..191 of every window."""
    tm = _timing(tau0=37.3, ppm=60.0)
    t0 = 50.5
    n = int(t0 + T.rx_index(tm, spec.n_pos + 20, spec.n_pos, t_sym=L_T_SYM)) + 64
    tau = T._tau_of_rx(tm, np.arange(n) - t0, spec.n_pos, t_sym=L_T_SYM)
    z = lwave.pilot(tau, spec, keying)
    z = z + math.sqrt(sigma2 / 2) * (rng.standard_normal(n) + 1j * rng.standard_normal(n))
    if junk:
        for pw in spec.win_start_pos:
            x = tau - (pw - 0.5)
            z = z + junk * (((x >= 0) & (x < 8)) | ((x >= 184) & (x < 192)))
    psd = sigma2 / CH_FS
    chan = SimpleNamespace(t0_index=t0, fs=CH_FS,
                           t_s=lambda k: (np.asarray(k, dtype=np.float64) - t0) / CH_FS,
                           noise_psd_at=lambda t: np.full(np.shape(t), psd))
    return SimpleNamespace(z_d=z, chan=chan, timing=tm)


def test_cwid_reads_an_l_window():
    spec = lframe.L_SHORT
    assert spec.n_win == 2 and spec.cw_window // CW_UNITS == 1
    rng = np.random.default_rng(7)
    tr = _l_window_track(spec, keying_units(CALL), 0.5, rng)     # 12.8 dB per unit
    llr = cwid.window_llr(None, CH_FS, tr, spec)
    assert llr.shape == (2, cwid.CW_ROOM_UNITS)
    assert cwid.read(llr) == CALL
    assert cwid.read(llr[0]) == CALL and cwid.read(llr[1]) == CALL
    assert cwid.match(cwid.window_soft(None, CH_FS, tr, spec), CALL) > 6
    assert cwid.keying_type(None, tr, spec) == "fsk"
    assert cwid.keying_type(None, SimpleNamespace(), spec) == "fsk"     # nothing computed
    assert cwid.keying_type(None, SimpleNamespace(), spec, CALL) == "fsk"
    with pytest.raises(ValueError):
        cwid.window_llr(None, CH_FS, tr, spec, ook=True)


def test_cwid_l_amplitude_from_call_units():
    """The FSK tone amplitude of an L window comes from units 8..183: energy in
    the plain units (a ramping pilot, data tails) changes no call-unit LLR."""
    spec = lframe.L_SHORT
    a = _l_window_track(spec, keying_units(CALL), 1.0, np.random.default_rng(5))
    b = _l_window_track(spec, keying_units(CALL), 1.0, np.random.default_rng(5), junk=3.0)
    la = cwid.window_llr(None, CH_FS, a, spec)
    lb = cwid.window_llr(None, CH_FS, b, spec)
    assert np.array_equal(la, lb)


# --- CE only ------------------------------------------------------------------------------

CE_ONLY = [
    ("track.track", lambda s: T.track(None, s, None)),
    ("track.fit_timing", lambda s: T.fit_timing(None, CH_FS, None, s, t0_index=0.0)),
    ("track.make_classes", lambda s: T.make_classes(s)),
    ("track.known_symbols", lambda s: T.known_symbols(s)),
    ("track.templates", lambda s: T.templates(s, np.arange(4), None, None)),
    ("track.remove_spread", lambda s: T.remove_spread(np.zeros(4), None, s, None, None)),
    ("track.verify", lambda s: T.verify(None, CH_FS, s, None)),
    ("track.verify_track", lambda s: T.verify_track(None, s, None)),
    ("track.genie_track", lambda s: T.genie_track(None, s, None, 1500.0)),
    ("track.refine_timing", lambda s: T.refine_timing(SimpleNamespace(spec=s))),
    ("demod.extract", lambda s: demod.extract(None, SimpleNamespace(spec=s), s, 0)),
    ("demod.calibrate", lambda s: demod.calibrate(SimpleNamespace(spec=s))),
]


@pytest.mark.parametrize("name,call", CE_ONLY, ids=[n for n, _ in CE_ONLY])
def test_ce_only_refuses_l(name, call):
    spec = lframe.L_SHORT
    assert spec.waveform == WAVEFORM_L
    with pytest.raises(AssertionError, match=name.replace(".", r"\.") + " is CE's"):
        call(spec)


def test_l_counterparts_named():
    with pytest.raises(AssertionError, match="ltrack.track_l"):
        T.track(None, lframe.L_TINY, None)
    with pytest.raises(AssertionError, match="ldemod.extract_l"):
        demod.extract(None, None, lframe.L_TINY, 0)

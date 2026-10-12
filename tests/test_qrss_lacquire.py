"""QRSSTVAE acquisition of waveform L, and the joint (CE + L) acquisition
(design section 3: `acquire` with a tuple of frames).

- A2-L (`acquire.L_PRE`): finds an L_TINY pilot at -20 dB, df < 0.05 Hz,
  timing within T_L/20, phase the pilot's at t0 whatever the receiver's
  t0; its statistic on noise is A2's Gamma (complex templates); a lead-in
  pilot with nothing after it, or a pilot with no data, gives no hit.
- Arbitration: the -1/T_L alias passes A2-L and is dropped by band
  ownership; CE's A2 on an L signal and L's on a CE one pass nowhere at
  0 and +6 dB; `merge_hits` keeps one hit per signal; the joint
  acquisition wires all of it, per waveform (A1 and A2 stubbed).
- A3's spectral shape: z1 is N(0, 1) on noise and tells L from CE at
  -15 dB (L needing nothing above its line); an L path is re-anchored on
  the pilot (`l_pilot_bins`), and the shape along it fills in.
- `acquire` with a tuple: a mixed CE + L slot end to end, hints with a
  waveform, `Acquisition.detections`' order, and the frame tuples.

Slow: A2-L's rate at -28 and -31 dB (10 SHORT preambles each, at the true
pilot), the joint live acquisition's (A1-limited) rate at -24 dB, A2-L's
tail over 1e6 noise cells, the shape z along A3's own paths on noise, and
the pilot re-anchoring on L_FULL at -28 dB.
CE through the single-frame `acquire` is pinned bit for bit by the CE
baseline check, not here.

L signals are kept at or below -6 dB (the inverse AGC follows L's
envelope above that), except the cross-waveform terms, which need
strength to show what does *not* pass (0 and +6 dB, design section 3).
"""

import functools
import math

import numpy as np
import pytest
from qrss_fakes import l_keying, l_scenario, l_symbols
from qrss_helpers import Q_TEST, unit_rms_latents
from scipy import special

from sstvae.qrss import acquire as A
from sstvae.qrss import ce, frame, frontend, lframe, lwave
from sstvae.qrss import channel as ch
from sstvae.qrss.channel import ChannelConfig, Neighbour
from sstvae.qrss.constants import FE_FS, L_PILOT_HZ, L_T_SYM, T_SYM, WAVEFORM_CE, WAVEFORM_L
from sstvae.qrss.morse import keying_units
from sstvae.qrss.sequences import preamble_ce, preamble_l
from sstvae.qrss.types import CarrierCandidate, FreqPath

PRE_S = 12.0                              # captures start at t0 - 12 s
F_P = float(L_PILOT_HZ)                   # data centre below the pilot
JOINT = (frame.TINY, lframe.L_TINY)


@functools.lru_cache(maxsize=8)
def _l_clean(spec_name="l-tiny", dur_s=40.0, lead_in_s=0.0, seed=0, zero=False):
    """L baseband (pilot-centred, unit power) from t0 - 12 s at FE rate."""
    spec = lframe.get(spec_name)
    sym = (np.zeros(spec.n_sym, dtype=complex) if zero
           else l_symbols(spec, unit_rms_latents(spec.n_data, seed)))
    bb = lwave.frame_baseband(sym, spec, keying=l_keying(spec), lead_in_s=lead_in_s)
    z = lwave.resample(bb, FE_FS, int(-PRE_S * FE_FS), int(dur_s * FE_FS))
    z.setflags(write=False)
    return z


@functools.lru_cache(maxsize=2)
def _ce_clean(dur_s=40.0):
    """CE SHORT baseband from t0 - 12 s at FE rate."""
    spec = frame.SHORT
    rng = np.random.default_rng(0)
    sym = frame.assemble(spec, rng.integers(0, 2, 2474).astype(np.uint8),
                         rng.standard_normal(spec.n_data))
    z = ce.baseband(sym, spec, FE_FS, -PRE_S, int(dur_s * FE_FS), keying=keying_units("K1ABC"))
    z.setflags(write=False)
    return z


def _capture(z, f_audio, snr_db, shift_s=0.0, seed=0):
    """(FE', nominal t0 index): z with its named frequency (CE carrier, L
    pilot) at f_audio, arriving shift_s late against the nominal t0, in
    AWGN at SNR2500 = snr_db, blanked and normalised as the receiver does."""
    n = len(z)
    turns = ch.nominal_turns(f_audio - 1500.0, np.arange(n), FE_FS, PRE_S * FE_FS)
    rng = np.random.default_rng(seed)
    var = FE_FS / (2500 * 10 ** (snr_db / 10))
    x = z * np.exp(2j * np.pi * (turns - np.floor(turns))) + math.sqrt(var / 2) * (
        rng.standard_normal(n) + 1j * rng.standard_normal(n))
    x, keep = frontend.blank(x.astype(np.complex64))
    x, _ = frontend.normalise(x, keep)
    return x, PRE_S * FE_FS - shift_s * FE_FS


def _noise_fe(n, rng):
    return ((rng.standard_normal(n) + 1j * rng.standard_normal(n)) / math.sqrt(2)).astype(
        np.complex64)


def _line(f):
    return CarrierCandidate(f, 0.0, 0.0)


def _hit(f, tau, lam, p, waveform):
    return A.PreambleHit(f_hz=f, drift_hz_per_min=0.0, tau0_s=tau, phase=0.0, lam=lam,
                         cell_lam=lam, p_value=p, passed=p < A.A2_PFA, waveform=waveform)


# --- the preamble kinds --------------------------------------------------------------------

def test_preamble_kinds():
    """CE's record is today's A2 (sequence, chunk, taps); L's holds its QPSK
    preamble in ten 2 s chunks, so both share one threshold; L's filter is
    its unit-energy pulse and passes nothing of the pilot left at +f_p."""
    assert np.array_equal(A.CE_PRE.seq, preamble_ce().astype(np.float64))
    assert np.array_equal(A.CE_PRE.mf_taps, A._mf_taps())
    assert A.CE_PRE.chunk == A.A2_CHUNK and A.CE_PRE.data_offset_hz == 0.0
    assert np.array_equal(A.L_PRE.seq, preamble_l())
    assert A.L_PRE.chunk == 33 and A.L_PRE.n_chunks == A.CE_PRE.n_chunks == 10
    assert A.L_PRE.data_offset_hz == -F_P and A.L_PRE.phase_offset == 0.0
    assert len(A.L_PRE.seq) * A.L_PRE.t_sym == pytest.approx(len(A.CE_PRE.seq) * T_SYM)
    taps, sps = A.L_PRE.mf_taps, 250 * L_T_SYM
    assert np.sum(taps ** 2) * sps == pytest.approx(1.0, rel=2e-3)     # unit energy
    assert np.sum(taps) == pytest.approx(lwave.G0, rel=2e-3)            # a constant gives G0

    def gain(f):
        k = np.arange(len(taps)) - len(taps) // 2
        return abs(np.sum(taps * np.exp(-2j * np.pi * f * k / 250)))
    assert gain(F_P) < 1e-3 * gain(0.0)
    assert gain(F_P - 2.0) < 1e-2 * gain(0.0)               # a lead-in's +-2 Hz ramp too
    b, W, norm = A._templates(A.A2_DF_HZ, A.L_PRE)
    assert b.shape == (10, 33) and W.dtype == complex
    assert np.abs(b.sum(axis=1)).max() < 1e-9 * norm.min()           # zero mean, as CE's
    assert np.allclose(norm, np.sum(np.abs(b) ** 2, axis=1))


# --- A2-L ------------------------------------------------------------------------------------

@pytest.mark.parametrize("f,shift", [(1500.0, 0.0), (1234.56, 1.3), (1777.7, -1.9),
                                     (2611.1, 0.4)])
def test_a2l_finds_an_l_tiny_pilot(f, shift):
    """A2-L at the pilot's line, L_TINY at -20 dB: passed, df < 0.05 Hz,
    timing within T_L/20, an L hit whose Detection is L's; CE's A2 at the
    same line does not pass."""
    x, t0 = _capture(_l_clean(), f, -20.0, shift, seed=int(f))
    h = A.preamble_search(x, t0, _line(f), kind=A.L_PRE)
    assert h.passed and h.waveform == WAVEFORM_L and h.n_groups in (1, 2, 5, 10)
    assert abs(h.f_hz - f) < 0.05
    assert abs(h.tau0_s - shift) < L_T_SYM / 20
    assert h.tau0_sd_s < L_T_SYM / 10
    d = h.to_detection(lframe.L_TINY.duration_s)
    assert d.waveform == WAVEFORM_L and d.method == "preamble"
    assert d.timing.tau0 == pytest.approx(shift * 250, abs=L_T_SYM * 250 / 20)
    assert not A.preamble_search(x, t0, _line(f), kind=A.CE_PRE).passed


def test_a2l_phase_is_the_pilots():
    """The line is at 1500 Hz (the channeliser does not mix) and the pilot
    0.3 Hz above it, with phase 0 at the sender's t0, which the receiver's
    nominal t0 precedes by `shift`: A2-L's phase is the pilot's there,
    -2 pi 0.3 shift, so both the f_p shift's rotation over tau0 (16 turns
    at 1.3 s) and the pilot's own (0.39 turns) are taken out. A rotation of
    the capture moves it by the same angle. (The phase is extrapolated
    from the preamble to t0, so df's error sets its scatter: sd 0.12-0.13
    rad at -20 dB over 12-16 seeds, mean 0.03-0.04.)"""
    for k, shift in enumerate((-1.9, 0.0, 1.3)):
        x, t0 = _capture(_l_clean(), 1500.3, -20.0, shift, seed=40 + k)
        h = A.preamble_search(x, t0, _line(1500.0), kind=A.L_PRE)
        want = -2 * np.pi * 0.3 * shift
        assert abs(np.angle(np.exp(1j * (h.phase - want)))) < 0.45, (shift, h.phase, want)
        h1 = A.preamble_search(x * np.exp(1j * 1.0), t0, _line(1500.0), kind=A.L_PRE)
        assert np.angle(np.exp(1j * (h1.phase - h.phase - 1.0))) == pytest.approx(0, abs=0.02)


def _l_noise_stats(n_caps, seed):
    rng = np.random.default_rng(seed)
    lams, ps = {}, []
    for _ in range(n_caps):
        x = _noise_fe(40 * FE_FS, rng)
        chan = A.preamble_channel(x, 8 * FE_FS, 1500 + rng.uniform(-900, 900), kind=A.L_PRE)
        lam, _, S, var = A._preamble_sums(chan)
        p, _, st = A.preamble_p(S, var)
        assert np.allclose(st[0][1], lam)
        for k, lk in st:
            lams.setdefault(k, []).append(lk.ravel())
        ps.append(p.ravel())
    return {k: np.concatenate(v) for k, v in lams.items()}, np.concatenate(ps)


def test_a2l_tail_on_noise():
    """A2-L on noise (8 captures, 2e5 cells per scale): every scale's Lambda
    against its Gamma at 1e-2 under 1.5x nominal and over 1/3 of it, and the
    combined p under nominal by no more than the 4 scales allow. (The cells
    are strongly correlated, so 8 captures resolve only 1e-2: CE's own
    statistic reads 0.53-1.05 of nominal there over two seeds, L's
    0.48-0.80. The slow test has 1e6 cells.)"""
    lams, ps = _l_noise_stats(8, 31)
    assert sorted(lams) == [1, 2, 5, 10]
    for k, v in lams.items():
        r = np.mean(v > special.gammainccinv(k, 1e-2)) / 1e-2
        assert 1 / 3 < r < 1.5, (k, r)
    r = np.mean(ps < 1e-2) / 1e-2
    assert 1 / len(A.A2_SCALES) < r < 1.2, r


@pytest.mark.slow
def test_a2l_tail_on_noise_1e6_cells():
    """As A-6 for CE, ~1e6 cells per scale: every scale's Lambda has its
    Gamma's mean to 5% and is under 2x nominal at 1e-3 and 1e-4. It reads
    light, 0.4-0.7 of nominal at 1e-4 over two seeds (CE: 0.5-0.85 on the
    same noise): the local noise floor, never below the global one, lifts
    the normaliser 3.7% on noise against CE's 2.6% (L's 2 s chunk holds
    half the symbols), conservative by ~0.05 dB. Hence 1/4 below."""
    lams, ps = _l_noise_stats(40, 32)
    for k, v in lams.items():
        assert len(v) > 1e6
        assert abs(np.mean(v) / k - 1) < 0.05, (k, np.mean(v))
        for p in (1e-3, 1e-4):
            r = np.mean(v > special.gammainccinv(k, p)) / p
            assert 0.25 < r < 2.0, (k, p, r)
    for p in (1e-2, 1e-3, 1e-4):
        r = np.mean(ps < p) / p
        assert 1 / len(A.A2_SCALES) < r < 1.2, (p, r)


def test_a2l_lead_in_only_pilot_gives_no_preamble():
    """An L lead-in (the pilot at average power) ending at t0 - 8 T_L with
    nothing after it, and a pilot with no data under it at all: A1 sees
    the line, and neither A2 passes at any A1 line; at the lead-in's own
    line A2-L finds nothing at timing -1.9 and +1.7 s either, at -6 and
    -10 dB."""
    lead = _l_clean(lead_in_s=10.0)
    t = np.arange(len(lead)) / FE_FS - PRE_S
    only = np.where(t < -8 * L_T_SYM, lead, 0)
    for z in (only, _l_clean(lead_in_s=10.0, zero=True)):
        x, t0 = _capture(z, 1637.3, -6.0, 0.0, seed=13)
        lines = A.carrier_lines(x, t0)
        assert any(abs(c.f_hz - 1637.3) < 0.25 for c in lines)
        for c in lines[:4]:
            inp = A._a2_input(x, t0, c.f_hz, c.drift_hz_per_min)
            for kind in (A.L_PRE, A.CE_PRE):
                h = A.preamble_search(x, t0, c, kind=kind, inp=inp)
                assert not h.passed, (kind.name, c.f_hz, h.p_value)
    for seed, snr, shift in ((14, -10.0, 1.7), (15, -6.0, -1.9)):
        x, t0 = _capture(only, 1637.3, snr, shift, seed=seed)
        h = A.preamble_search(x, t0, _line(1637.3), kind=A.L_PRE)
        assert not h.passed, (seed, snr, shift, h.p_value)


# --- arbitration -----------------------------------------------------------------------------

def test_l_alias_is_band_owned():
    """At -6 dB L's -1/T_L alias (16.5 Hz below the pilot, in its data band)
    passes A2-L; band ownership drops it and keeps the signal, while a
    CE hit there, or an L hit with its own timing, stays."""
    x, t0 = _capture(_l_clean(), 1500.0, -6.0, 0.7, seed=1)
    main = A.preamble_search(x, t0, _line(1500.0), kind=A.L_PRE)
    alias = A.preamble_search(x, t0, _line(1500.0 - 1 / L_T_SYM), kind=A.L_PRE)
    assert main.passed and alias.passed and main.lam > 3 * alias.lam
    assert A.drop_band_owned([alias, main]) == [main]
    other = _hit(alias.f_hz, alias.tau0_s + 0.5, alias.lam, alias.p_value, WAVEFORM_L)
    ce_hit = _hit(alias.f_hz, alias.tau0_s, alias.lam, alias.p_value, WAVEFORM_CE)
    assert len(A.drop_band_owned([other, main])) == 2
    assert len(A.drop_band_owned([ce_hit, main])) == 2
    above = _hit(1500.0 + 10.0, main.tau0_s, alias.lam, alias.p_value, WAVEFORM_L)
    assert len(A.drop_band_owned([above, main])) == 2         # the band is below the pilot
    # per waveform, aliases at L's own 1/T
    a2 = _hit(1500.0 + 1 / L_T_SYM, main.tau0_s, main.lam / 4, 1e-9, WAVEFORM_L)
    assert A.drop_symbol_rate_aliases([a2, main], L_T_SYM) == [main]
    assert len(A.drop_symbol_rate_aliases([a2, main])) == 2   # CE's T: not an alias


@pytest.mark.parametrize("snr_db", [0.0, 6.0])
def test_no_cross_waveform_pass(snr_db):
    """CE's A2 on an L signal (at its pilot, its data centre and its -1/T_L
    line) and L's A2 on a CE signal (at its carrier, f_p above it and its
    +-1/T lines) pass nowhere."""
    x, t0 = _capture(_l_clean(), 1500.0, snr_db, 0.7, seed=2)
    for f in (1500.0, 1500.0 - F_P, 1500.0 - 1 / L_T_SYM):
        h = A.preamble_search(x, t0, _line(f), kind=A.CE_PRE)
        assert not h.passed and h.p_value > 1e-5, (f, h.p_value)
    x, t0 = _capture(_ce_clean(), 1500.0, snr_db, 0.7, seed=3)
    for f in (1500.0, 1500.0 + F_P, 1500.0 + 1 / T_SYM, 1500.0 - 1 / T_SYM):
        h = A.preamble_search(x, t0, _line(f), kind=A.L_PRE)
        assert not h.passed and h.p_value > 1e-6, (f, h.p_value)


def test_merge_hits():
    """One hit per signal: a CE and an L hit 2 Hz and 10 ms apart are one,
    the smaller p kept (then the larger Lambda); with timings 0.5 s apart,
    or 6 Hz apart, two."""
    c = _hit(1502.0, 0.31, 60.0, 1e-12, WAVEFORM_CE)
    l_ = _hit(1500.0, 0.32, 40.0, 1e-20, WAVEFORM_L)
    assert A.merge_hits([c, l_]) == [l_]
    assert A.merge_hits([_hit(1502.0, 0.31, 60.0, 0.0, WAVEFORM_CE),
                         _hit(1500.0, 0.32, 40.0, 0.0, WAVEFORM_L)])[0].waveform == WAVEFORM_CE
    assert len(A.merge_hits([_hit(1502.0, 0.81, 60.0, 1e-12, WAVEFORM_CE), l_])) == 2
    assert len(A.merge_hits([_hit(1506.0, 0.31, 60.0, 1e-12, WAVEFORM_CE), l_])) == 2
    out = A.merge_hits([_hit(1300.0, 0.0, 20.0, 1e-8, WAVEFORM_L), c,
                        _hit(1800.0, 0.0, 90.0, 1e-30, WAVEFORM_L)])
    assert [h.lam for h in out] == [90.0, 60.0, 20.0]          # strongest first


def test_pilot_lines_offer_the_cell_above_a_line():
    """For each A1 line, A1's best cell in (f + 3.1, f + 22.1] Hz: here a
    lump took the line at 1488 Hz and the pilot sits 12 Hz above it; a
    hypothesis on top of a line is not offered twice."""
    f = 300.0 + 0.25 * np.arange(9601)
    stat = np.full((3, len(f)), 11.0, dtype=np.float32)
    b0 = int(np.argmin(np.abs(f - 1488.0)))
    stat[1, b0] = 80.0                                   # the lump: A1's line
    stat[1, b0 + 48] = 50.0                              # the pilot, 12 Hz above
    stat[1, b0 + 47], stat[1, b0 + 49] = 30.0, 40.0
    stat[1, b0 + 100] = 70.0                             # 25 Hz above: outside
    cs = A.CarrierStats(stat=stat, f_hz=f, drifts=np.array([-0.5, 0.0, 0.5]), M=11)
    lines = [CarrierCandidate(1488.0, 0.0, 80.0)]
    hyp = A.pilot_lines(cs, lines)
    assert len(hyp) == 1 and 1500.0 < hyp[0].f_hz < 1500.25
    assert hyp[0].drift_hz_per_min == 0.0 and hyp[0].metric == 50.0
    assert A.pilot_lines(cs, lines + [CarrierCandidate(1500.1, 0.0, 50.0)])[0].f_hz != \
        pytest.approx(hyp[0].f_hz)                        # covered by the line at 1500.1


# --- the spectral shape -------------------------------------------------------------------

def test_shape_frames():
    """The shape test leaves out frames overlapping either waveform's
    callsign windows (both key down 16.5 Hz below the line) and the
    lead-in (before -8 symbols)."""
    specs = (frame.SHORT, lframe.L_SHORT)
    t = A.tbd_frame_starts(frame.SHORT)
    keep = A.shape_frames(t, specs)
    assert not keep[t < -0.2425].any() and keep[t == 0.0].all()
    for spec in specs:
        for a, b in spec.window_spans_s():
            assert not keep[(t < b) & (t + A.STFT_FRAME_S > a)].any()
    assert keep.sum() == 113 and len(t) == 132


def _shape_noise(n_caps, seed, spec=frame.SHORT, tbd=False):
    """z1, z_above along straight paths 45 Hz apart (independent bands) on
    noise-only slots, and A3's own z1 when tbd."""
    specs = (spec, lframe.L_SHORT)
    rng = np.random.default_rng(seed)
    n = int((PRE_S + spec.keyed_end_pos * spec.t_sym + 18) * FE_FS)
    z1, za, zv = [], [], []
    for _ in range(n_caps):
        x = _noise_fe(n, rng)
        E, f, t = A.tbd_emissions(x, PRE_S * FE_FS, spec)
        kf = A.shape_frames(t, specs)
        for b in range(100, len(f) - 100, 180):
            a, c = A.shape_z(E, np.full(len(t), b), kf)
            z1.append(a)
            za.append(c)
        if tbd:
            zv += [h.shape_z for h in A.track_before_detect(
                x, PRE_S * FE_FS, spec, all_hits=True, shape_specs=specs)]
    zv = np.array(zv)
    return np.array(z1), np.array(za), zv[np.isfinite(zv)]


def test_shape_z_is_standard_normal_on_noise():
    """z1 and z_above on noise-only SHORT slots (3 slots, 159 bands): mean
    ~0, sd ~1, nothing past the L threshold; no usable frame gives nan."""
    z1, za, _ = _shape_noise(3, 41)
    assert len(z1) > 150
    for v in (z1, za):
        assert abs(np.mean(v)) < 0.25 and 0.8 < np.std(v) < 1.2, (np.mean(v), np.std(v))
    assert np.abs(z1).max() < A.SHAPE_L_Z
    E = np.zeros((5, 400))
    assert np.isnan(A.shape_z(E, np.full(5, 200), np.zeros(5, bool))[0])
    # a frame with no data (a capture gap: E = -1 everywhere) is left out
    E = np.random.default_rng(3).exponential(1.0, (40, 400)) - 1.0
    gap = E.copy()
    gap[10:14] = -1.0
    keep = np.ones(40, bool)
    keep[10:14] = False
    assert A.shape_z(gap, np.full(40, 200)) == A.shape_z(E, np.full(40, 200), keep)


@pytest.mark.slow
def test_shape_z_on_a3_paths_on_noise():
    """z1 along A3's own paths on noise (SHORT, 6 slots, every hit A3
    keeps with all_hits): mean ~0, sd ~1, nothing past 4. The bands beside
    a Viterbi path read ~0.3 sd low on *both* sides, the path's own
    medians lifting the local CFAR; z1 is their difference and is not
    moved."""
    _, _, zv = _shape_noise(6, 42, tbd=True)
    assert len(zv) > 200
    assert abs(np.mean(zv)) < 0.25 and 0.8 < np.std(zv) < 1.2, (np.mean(zv), np.std(zv))
    assert np.abs(zv).max() < A.SHAPE_L_Z


def test_shape_band_covariance():
    """`_band_cov`: Hann's adjacent-bin power correlations 4/9 and 1/36
    within a frame, 1/36 at one bin between frames half overlapped."""
    v0, v1 = A._band_cov(1)
    assert v0 == pytest.approx(1.0) and v1 == pytest.approx(1 / 36)
    v0, _ = A._band_cov(76)
    assert v0 == pytest.approx(76 + 2 * 75 * 4 / 9 + 2 * 74 / 36, rel=1e-6)


def test_shape_tells_l_from_ce(monkeypatch):
    """A3 on an L_TINY and a TINY slot at -15 dB: the L hit's z1 is far over
    4 (its data below the pilot), the CE hit's |z1| under 3 with its power
    above the carrier significant; each says its waveform, along its own
    path and along the pilot-anchored one (on the carrier, for CE), over
    the frames `shape_frames` keeps. L needs nothing significant above
    the line: a strong CE's z1 can pass 4 (2 of 8 SHORT passes at -10 dB),
    its z_above never fails; two paths that disagree are ambiguous."""
    specs = (frame.TINY, lframe.L_TINY)
    seen, real = [], A.shape_frames

    def frames(t, sp, *a, **k):
        seen.append(sp)
        return real(t, sp, *a, **k)
    monkeypatch.setattr(A, "shape_frames", frames)
    _, sim = l_scenario("l-tiny", -15.0, 7, pilot=1421.7)
    spec = frame.TINY
    rng = np.random.default_rng(8)
    sym = frame.assemble(spec, None, rng.standard_normal(spec.n_data))
    sim_ce = ch.simulate(sym, spec, Q_TEST, ChannelConfig(snr_db=-15.0, seed=8),
                         carrier_hz=1421.7)
    for s, wf in ((sim, WAVEFORM_L), (sim_ce, WAVEFORM_CE)):
        x, keep = frontend.blank(s.fe)
        x, _ = frontend.normalise(x, keep)
        hits = A.track_before_detect(x, s.t0_index, spec, shape_specs=specs)
        h = next(h for h in hits if abs(h.f_hz - 1421.7) < 0.5)
        assert h.waveform == wf, (wf, h.shape_z, h.above_z)
        assert A.shape_waveform(h.l_shape_z, h.l_above_z) == wf, (h.l_shape_z, h.l_above_z)
        if wf == WAVEFORM_L:
            assert h.shape_z > 10 * A.SHAPE_L_Z
        else:
            assert abs(h.shape_z) < A.SHAPE_CE_MAX_Z1 and h.above_z > 2 * A.SHAPE_CE_ABOVE_Z
            assert abs(h.to_detection(WAVEFORM_L).f_hz - 1421.7) < 0.1      # anchored: the carrier
    assert seen == [specs, specs]
    assert A.shape_waveform(4.5, -3.0) == WAVEFORM_L
    assert A.shape_waveform(1.0, 3.5) == WAVEFORM_CE
    assert A.shape_waveform(3.5, 6.0) is None and A.shape_waveform(1.0, 2.0) is None
    assert A.shape_waveform(5.0, 95.0) is None                        # a strong CE: not L
    CE, L = WAVEFORM_CE, WAVEFORM_L
    assert [A._shape_verdict(*p) for p in ((L, L), (L, None), (None, L), (None, CE),
                                           (CE, L), (L, CE), (None, None))] == [
        L, L, L, CE, None, None, None]


def test_l_pilot_bins_re_anchor_a_data_band_path():
    """At a weak L signal A3's path can follow the data band. On planted
    emissions -- a pilot line, a band of data 3.1-22 Hz below it, noise --
    a path along the band is re-anchored on the pilot, and it is the band
    below the bin that keeps it there: on the bin alone a path wandering
    in a band this strong (a third of the pilot's power per bin) collects
    more. Frames outside `keep` count nothing, so a strong line beside the
    pilot there (a key-down) does not pull it."""
    rng = np.random.default_rng(9)
    n, nb = 300, 800
    f = 1000.0 + 0.25 * np.arange(nb)
    E = rng.exponential(1.0, (n, nb)) - 1.0
    p0 = 500                                                   # the pilot: 1125 Hz
    E[:, p0] += 1.5 * rng.exponential(1.0, n)
    E[:, p0 - 88:p0 - 12] += 0.5 * rng.exponential(1.0, (n, 76))
    t = -10.0 + 2.0 * np.arange(n)
    line = np.array([0.0, f[p0 - 40] - 0.0])                   # a path 10 Hz below
    bins = A.l_pilot_bins(E, f, t, line)
    d = np.abs(bins - p0)
    assert np.median(d) <= 1 and np.mean(d <= 3) > 0.9 and (bins > p0 - 12).all()
    keep = np.ones(n, dtype=bool)
    keep[100:130] = False
    E[100:130, p0 + 6] += 40.0                                 # 1.5 Hz above, outside keep
    bins = A.l_pilot_bins(E, f, t, line, keep)
    assert np.median(bins) == p0 and np.abs(bins[100:130] - p0).max() <= 3


# --- acquire with a tuple ---------------------------------------------------------------------

def test_joint_acquire_on_a_mixed_slot():
    """An L_TINY pass at 1611.3 Hz with a 5 s lead-in and a TINY CE
    neighbour at 1187.5 Hz, both at -20 dB: one preamble hit each, in its
    own waveform, at its frequency and timing, the L one's lead-in measured
    at L's symbol; A3 finds both and its shape says which is which; the
    gate gets the preamble hits first, each as its waveform. With CE alone
    enabled the L signal is still offered, as CE, last."""
    _, sim = l_scenario("l-tiny", -20.0, 4, pilot=1611.3, lead_in=5.0, cfg=(
        ("neighbours", (Neighbour(df_hz=-423.8, rel_db=0.0, seed=3, waveform=WAVEFORM_CE),)),))
    cap = frontend.Capture(sim.fe, sim.q, sim.t0_index)
    acq = A.acquire(cap, JOINT)
    pre = {h.waveform: h for h in acq.preamble}
    assert sorted(pre) == [WAVEFORM_CE, WAVEFORM_L] and len(acq.preamble) == 2
    assert abs(pre[WAVEFORM_L].f_hz - 1611.3) < 0.05 and abs(pre[WAVEFORM_CE].f_hz - 1187.5) < 0.05
    assert abs(pre[WAVEFORM_L].tau0_s) < L_T_SYM / 20 and abs(pre[WAVEFORM_CE].tau0_s) < T_SYM / 4
    assert pre[WAVEFORM_L].lead_in_s == 5.0 and pre[WAVEFORM_CE].lead_in_s == 0.0
    tbd = {h.waveform: h for h in acq.tbd}
    assert abs(tbd[WAVEFORM_L].f_hz - 1611.3) < 0.3 and abs(tbd[WAVEFORM_CE].f_hz - 1187.5) < 0.3
    dets = acq.detections(JOINT)
    assert [(d.method, d.waveform) for d in dets[:2]] == [
        ("preamble", pre[w].waveform) for w in sorted(pre, key=lambda w: -pre[w].lam)]
    assert {(d.method, d.waveform, round(d.f_hz)) for d in dets} >= {
        ("tbd", WAVEFORM_L, 1611), ("tbd", WAVEFORM_CE, 1187)}
    ce_only = acq.detections((frame.TINY,))
    assert [(d.method, round(d.f_hz)) for d in ce_only] == [
        ("preamble", 1187), ("tbd", 1187), ("tbd", 1611)]
    assert all(d.waveform == WAVEFORM_CE for d in ce_only)
    live = A.acquire(cap, JOINT, live_only=True)
    assert live.tbd == [] and [h.f_hz for h in live.preamble] == [h.f_hz for h in acq.preamble]


def test_joint_hints(monkeypatch):
    """Hints in the joint acquisition: (f, waveform) is searched with that
    waveform's A2, a bare f with every enabled one's. An L pass at -29 dB,
    below A1 (no line offered at all here), hinted 0.8 Hz off: A2-L at f
    and f +- 1 Hz finds it. A bare hint at nothing is offered to the gate
    once per waveform, in its waveform; a hint for a waveform that is not
    enabled is ignored."""
    _, sim = l_scenario("l-tiny", -29.0, 5, pilot=1611.3)
    cap = frontend.Capture(sim.fe, sim.q, sim.t0_index)
    monkeypatch.setattr(A, "carrier_lines", lambda *a, **k: [])
    acq = A.acquire(cap, JOINT, live_only=True, hints=((1612.1, WAVEFORM_L), 2000.0))
    assert len(acq.preamble) == 1 and acq.preamble[0].waveform == WAVEFORM_L
    assert abs(acq.preamble[0].f_hz - 1611.3) < 0.05
    assert sorted((round(c.f_hz), c.waveform) for c in acq.hinted) == [
        (2000, WAVEFORM_CE), (2000, WAVEFORM_L)]
    dets = acq.detections(JOINT)
    assert [(d.method, d.waveform) for d in dets] == [
        ("preamble", WAVEFORM_L), ("hint", WAVEFORM_CE), ("hint", WAVEFORM_L)]
    assert dets[1].timing is None and abs(dets[2].f_hz - 2000.0) < 2.5
    none = A.acquire(cap, (frame.TINY,), live_only=True, hints=((1612.1, WAVEFORM_L),))
    assert none.preamble == [] and none.hinted == []


def test_joint_arbitration(monkeypatch):
    """`acquire` with a tuple, A1 and A2 stubbed: CE's A2 runs at A1's lines,
    L's at the lines and at their pilot hypotheses (only with L enabled).
    The passed hits are arbitrated per waveform at its own symbol (CE's
    +1/T alias at 33 Hz, L's +1/T_L at 16.5 Hz), L's -1/T_L alias 0.8 T_L
    off the pilot's timing (outside the alias rule's T_L/2) goes by band
    ownership, an L hit on the CE carrier merges into the CE one (the
    smaller p), and each lead-in is measured at its waveform's symbol."""
    CE, L = WAVEFORM_CE, WAVEFORM_L
    passed = {(CE, 1000.0): (500.0, 1e-200, 0.30), (CE, 1033.0): (60.0, 1e-9, 0.30),
              (L, 1000.0): (20.0, 1e-8, 0.30), (L, 1500.0): (200.0, 1e-90, 0.70),
              (L, 1483.5): (40.0, 1e-12, 0.75), (L, 1516.5): (30.0, 1e-10, 0.70)}
    calls, leads = [], {}

    def search(fe, t0, cand, fs=FE_FS, *, pfa=A.A2_PFA, kind=A.CE_PRE, inp=None):
        calls.append((kind.waveform, cand.f_hz))
        lam, p, tau = passed.get((kind.waveform, cand.f_hz), (5.0, 0.3, 0.0))
        return _hit(cand.f_hz, tau, lam, p, kind.waveform)

    def lead_in(fe, t0, f, drift=0.0, tau0_s=0.0, fs=FE_FS, *, t_sym=T_SYM, **k):
        leads[round(f, 1)] = t_sym
        return 0.0
    monkeypatch.setattr(A, "carrier_lines", lambda *a, **k: [
        _line(1000.0), _line(1033.0), _line(1483.5), _line(1516.5)])
    monkeypatch.setattr(A, "pilot_lines", lambda cs, lines: [_line(1500.0)])
    monkeypatch.setattr(A, "_a2_input", lambda *a, **k: None)
    monkeypatch.setattr(A, "preamble_search", search)
    monkeypatch.setattr(A, "lead_in_span", lead_in)
    cap = frontend.Capture(np.zeros(40 * FE_FS, np.complex64), Q_TEST, PRE_S * FE_FS)
    acq = A.acquire(cap, JOINT, live_only=True, preprocessed=True)
    assert [(h.waveform, h.f_hz) for h in acq.preamble] == [(CE, 1000.0), (L, 1500.0)]
    assert {f for w, f in calls if w == CE} == {1000.0, 1033.0, 1483.5, 1516.5}
    assert {f for w, f in calls if w == L} == {1000.0, 1033.0, 1483.5, 1516.5, 1500.0}
    assert leads == {1000.0: T_SYM, 1500.0: L_T_SYM}
    assert [c.f_hz for c in acq.pilots] == [1500.0]
    calls.clear()
    acq = A.acquire(cap, (frame.TINY,), live_only=True, preprocessed=True)
    assert [(h.waveform, h.f_hz) for h in acq.preamble] == [(CE, 1000.0)]
    assert {w for w, _ in calls} == {CE} and acq.pilots == []
    calls.clear()
    acq = A.acquire(cap, (lframe.L_TINY,), live_only=True, preprocessed=True)
    assert [(h.waveform, h.f_hz) for h in acq.preamble] == [(L, 1500.0), (L, 1000.0)]
    assert {w for w, _ in calls} == {L}


def test_detections_order():
    """`Acquisition.detections` with a tuple: preamble hits (strongest
    first), hints, A3 hits in their shape's waveform (ambiguous as CE),
    then the L copies of ambiguous ones, then hits whose waveform is not
    enabled, gated as the one that is. An L detection from A3 takes the
    pilot-anchored path, and one landing within 10 Hz of an earlier A3 L
    detection (a weak L pass's second hit, in its data band) is the same
    gate, left out."""
    def tbd(f, w, l_f=None):
        t = np.array([0.0, 100.0])
        path = FreqPath(t_s=t, f_hz=np.full(2, f), weight=np.ones(2))
        lp = None if l_f is None else FreqPath(t_s=t, f_hz=np.full(2, l_f), weight=np.ones(2))
        return A.TbdHit(path=path, z=50.0, p_value=1e-9, n_frames=31, centre_hz=f,
                        waveform=w, l_path=lp)
    acq = A.Acquisition(
        lines=[], preamble=[_hit(1000.0, 0.0, 50.0, 1e-12, WAVEFORM_CE),
                            _hit(1100.0, 0.0, 80.0, 1e-20, WAVEFORM_L)],
        tbd=[tbd(1200.0, WAVEFORM_CE), tbd(1300.0, None, l_f=1306.0), tbd(1400.0, WAVEFORM_L),
             tbd(1385.0, None, l_f=1404.0)],
        hinted=[CarrierCandidate(1500.0, 0.0, 0.0, waveform=WAVEFORM_L)])

    def order(specs):
        return [(d.method, d.waveform, round(d.f_hz)) for d in acq.detections(specs)]
    CE, L = WAVEFORM_CE, WAVEFORM_L
    assert order(JOINT) == [("preamble", L, 1100), ("preamble", CE, 1000), ("hint", L, 1500),
                            ("tbd", CE, 1200), ("tbd", CE, 1300), ("tbd", L, 1400),
                            ("tbd", CE, 1385), ("tbd", L, 1306)]
    assert order((frame.TINY,)) == [("preamble", CE, 1000), ("tbd", CE, 1200),
                                    ("tbd", CE, 1300), ("tbd", CE, 1385), ("tbd", CE, 1400)]
    assert order((lframe.L_TINY,)) == [("preamble", L, 1100), ("hint", L, 1500),
                                       ("tbd", L, 1306), ("tbd", L, 1400), ("tbd", L, 1200)]
    dets = acq.detections(JOINT)
    assert dets[0].timing is not None and dets[0].path.t_s[-1] == pytest.approx(
        lframe.L_TINY.keyed_end_pos * L_T_SYM)
    # a single frame keeps today's order: preamble as stored, hints, A3
    assert [d.method for d in acq.detections(frame.TINY)] == (
        ["preamble"] * 2 + ["hint"] + ["tbd"] * 4)


def test_frame_tuples():
    """At most one frame per waveform, and at least one. A single frame,
    whatever its waveform, is today's acquisition (CE's, with the frame's
    own span), which callers not yet waveform-aware rely on; the same
    frame in a tuple is the waveform-aware one."""
    cap = frontend.Capture(np.zeros(40 * FE_FS, np.complex64), Q_TEST, PRE_S * FE_FS)
    with pytest.raises(ValueError, match="one frame per waveform"):
        A.acquire(cap, (frame.TINY, frame.SHORT))
    with pytest.raises(ValueError, match="at least one"):
        A.acquire(cap, ())
    acq = A.Acquisition([], [], [], hinted=[CarrierCandidate(1500.0, 0.0, 1.0)])
    end = lframe.L_TINY.keyed_end_pos * L_T_SYM
    (d,) = acq.detections(lframe.L_TINY)
    assert d.waveform == WAVEFORM_CE and d.path.t_s[-1] == end
    (d,) = acq.detections((lframe.L_TINY,))
    assert d.waveform == WAVEFORM_L and d.path.t_s[-1] == end


# --- slow ---------------------------------------------------------------------------------------

@pytest.mark.slow
def test_a2l_detects_at_minus_28_and_31_db():
    """SHORT L preambles through the channel simulator, A2-L at the true
    pilot line: at -28 dB 10/10 (f within 0.05 Hz, timing within T_L/4),
    timing rms under T_L/10 and its reported sd honest; at -31 dB >= 9/10.
    (Measured 2026-10-12: 8/8 at both, Lambda median 67 and 43, timing rms
    0.057 and 0.083 T_L.)"""
    for snr, need in ((-28.0, 10), (-31.0, 9)):
        found, errs, sds = 0, [], []
        for s in range(10):
            f = 900.0 + 97.3 * s + 0.37
            _, sim = l_scenario("l-short", snr, 100 + s, pilot=f)
            x, keep = frontend.blank(sim.fe)
            x, _ = frontend.normalise(x, keep)
            h = A.preamble_search(x, sim.t0_index, _line(f), kind=A.L_PRE)
            found += h.passed and abs(h.f_hz - f) < 0.05 and abs(h.tau0_s) < L_T_SYM / 4
            errs.append(h.tau0_s / L_T_SYM)
            sds.append(h.tau0_sd_s / L_T_SYM)
        assert found >= need, (snr, found)
        if snr == -28.0:
            rms = np.sqrt(np.mean(np.square(errs)))
            assert rms < 1 / 10, errs
            assert 0.5 < rms / np.mean(sds) < 1.6


@pytest.mark.slow
def test_joint_live_acquisition_rate():
    """The live chain (A1 lines and their pilot hypotheses, then A2-L) on
    SHORT L passes at -24 dB: >= 6/10 found (measured 8/10, 2 of them only
    through a pilot hypothesis). A1 is what limits it (design 8.1): at
    -28 dB A2-L given the line passes 10/10 and the chain ~1/10."""
    specs = (frame.SHORT, lframe.L_SHORT)
    found = 0
    for s in range(10):
        f = 900.0 + 97.3 * s + 0.37
        _, sim = l_scenario("l-short", -24.0, 200 + s, pilot=f)
        acq = A.acquire(frontend.Capture(sim.fe, sim.q, sim.t0_index), specs, live_only=True)
        found += any(p.waveform == WAVEFORM_L and abs(p.f_hz - f) < 0.25
                     and abs(p.tau0_s) < L_T_SYM / 4 for p in acq.preamble)
    assert found >= 6, found


@pytest.mark.slow
def test_a3_l_full_is_anchored_on_its_pilot():
    """L_FULL at -28 dB: A3's best path near the signal can run in its data
    band, several Hz below the pilot; its shape says L, and gated as L
    the detection is on the pilot (within 0.25 Hz). Measured 2026-10-12:
    hits at 994.2 Hz (z1 16.1) and 984.0 Hz (z1 -13.4, z_above 18.2; on
    its anchored path z1 21.8), both anchored on 1000.46 Hz."""
    lspec = lframe.L_FULL
    specs = (frame.FULL, lspec)
    f = 1000.4
    a = unit_rms_latents(lspec.n_data, 300)
    sim = ch.simulate(l_symbols(lspec, a), lspec, Q_TEST, ChannelConfig(snr_db=-28.0, seed=300),
                      carrier_hz=f, keying=l_keying(lspec))
    x, keep = frontend.blank(sim.fe)
    x, _ = frontend.normalise(x, keep)
    hits = A.track_before_detect(x, sim.t0_index, lspec, shape_specs=specs)
    near = [h for h in hits if abs(h.f_hz - f) < 25]
    h = min(near, key=lambda h: abs(h.f_hz - f))
    assert h.waveform == WAVEFORM_L and h.shape_z > 10
    d = h.to_detection()
    assert d.waveform == WAVEFORM_L and abs(d.f_hz - f) < 0.25, (h.f_hz, d.f_hz)
    # the second hit, 16 Hz below the pilot in the data band, is ambiguous
    # on its own path (the pilot above it) and L on its anchored one, so
    # the gate gets one L detection for the pass, not a CE one too
    assert len(near) == 2 and all(g.waveform == WAVEFORM_L for g in near)
    assert any(A.shape_waveform(g.shape_z, g.above_z) is None for g in near)
    dets = [(d.waveform, d.method) for d in A.Acquisition([], [], hits).detections(specs)
            if abs(d.f_hz - f) < 25]
    assert dets == [(WAVEFORM_L, "tbd")]

"""QRSSTVAE acquisition, design 6.3, acceptance A-4 to A-7.

- A-4: A1 finds the carrier at -400/-150/0/+150/+400 Hz about 1500 Hz and
  at the band edges (340, 2660 Hz), at SNR2500 = -25 dB, within 0.25 Hz.
- A-5: A2 on SHORT at -25 dB, the same frequencies, timing offsets
  -1.9/0/+1.7 s: detection 100%; df < 0.05 Hz in every case; timing
  rms < T/20 (see the test for why rms rather than every case).
- A-6: A2's Lambda (1e6 cells, every scale) and A1's bins on noise follow
  their Gamma within a factor 2 at 1e-3 and 1e-4, and A2's combined p is
  calibrated; a lead-in-only carrier at +10 dB gives no A2 peak above
  threshold, at any timing, and each of A2's lead-in defences is pinned.
- A-7: the committed TBD_GUMBEL_* constants: re-measured on 200 noise-only
  captures per frame length (slow), and a fast 20-capture check.
- A3 finds and follows a drifting signal, lead-in included; `acquire`
  end to end through the channel simulator.
"""

import functools
import math

import numpy as np
import pytest
from scipy import special

from sstvae.qrss import acquire as A
from sstvae.qrss import ce, frame, frontend
from sstvae.qrss import channel as ch
from sstvae.qrss.channel import ChannelConfig
from sstvae.qrss.constants import FE_FS, T_SYM
from sstvae.qrss.morse import keying_units
from sstvae.qrss.types import CarrierCandidate, Detection

from qrss_helpers import Q_TEST

KEY = keying_units("K1ABC")
PRE_S = 12.0                              # capture starts at t0 - 12 s
FREQS = (1100.0, 1350.0, 1500.0, 1650.0, 1900.0, 340.0, 2660.0)   # A-4's offsets and edges
SHIFTS = (-1.9, 0.0, 1.7)


@functools.lru_cache(maxsize=4)
def _clean(spec_name="short", dur_s=40.0, lead_in_s=0.0):
    """CE baseband of a frame from t0 - 12 s, at FE rate, unit power."""
    spec = frame.PRESETS[spec_name]
    rng = np.random.default_rng(0)
    hdr = rng.integers(0, 2, 2474).astype(np.uint8) if spec.n_hdr else None
    sym = frame.assemble(spec, hdr, rng.standard_normal(spec.n_data))
    z = ce.baseband(sym, spec, FE_FS, -PRE_S, int(dur_s * FE_FS),
                    keying=KEY if spec.n_win else None, lead_in_s=lead_in_s)
    z.setflags(write=False)
    return z


def _capture(f_audio, snr_db, shift_s=0.0, seed=0, drift=0.0, z=None, preprocess=True):
    """(FE', nominal t0 index): the signal at f_audio (Hz at t0), drifting
    `drift` Hz/min, arriving shift_s late against the nominal t0, in AWGN
    at SNR2500 = snr_db; blanked and normalised like the receiver does."""
    z = _clean() if z is None else z
    n = len(z)
    t = np.arange(n) / FE_FS - PRE_S                      # s after the true t0
    turns = ch.nominal_turns(f_audio - 1500.0, np.arange(n), FE_FS, PRE_S * FE_FS)
    turns = turns + drift / 120.0 * t * t
    rng = np.random.default_rng(seed)
    var = FE_FS / (2500 * 10 ** (snr_db / 10))
    x = z * np.exp(2j * np.pi * (turns - np.floor(turns))) + math.sqrt(var / 2) * (
        rng.standard_normal(n) + 1j * rng.standard_normal(n))
    x = x.astype(np.complex64)
    if preprocess:
        x, keep = frontend.blank(x)
        x, _ = frontend.normalise(x, keep)
    return x, PRE_S * FE_FS - shift_s * FE_FS


def _best_near(cands, f, tol=1.0):
    near = [c for c in cands if abs(c.f_hz - f) < tol]
    return near[0] if near else None


# --- A-4 --------------------------------------------------------------------------------------

@pytest.mark.parametrize("f_audio", FREQS)
def test_a4_carrier_lines(f_audio):
    """A1 at -25 dB: the strongest line is the carrier, within 0.25 Hz."""
    x, t0 = _capture(f_audio + 0.37, -25.0, seed=int(f_audio))
    cands = A.carrier_lines(x, t0)
    assert 1 <= len(cands) <= A.A1_MAX_CANDS
    assert all(isinstance(c, CarrierCandidate) for c in cands)
    assert cands[0].f_hz == pytest.approx(f_audio + 0.37, abs=0.25)
    assert cands[0].metric > A.a1_threshold(11)
    fs = sorted(c.f_hz for c in cands)
    assert np.all(np.diff(fs) >= A.A1_MIN_SEP_HZ)


def test_a4_carrier_line_drift():
    """A drifting carrier (+2.5 Hz/min) is summed along its drift and
    reported at its frequency at t0; a steady one reports no drift."""
    x, t0 = _capture(1234.5, -25.0, seed=3, drift=2.5)
    c = A.carrier_lines(x, t0)[0]
    assert c.f_hz == pytest.approx(1234.5, abs=0.25)
    assert c.drift_hz_per_min == pytest.approx(2.5, abs=0.75)
    cs = A.carrier_line_stats(x, t0)
    b = int(np.argmin(np.abs(cs.f_hz - 1234.5)))
    d0 = int(np.argmin(np.abs(cs.drifts)))
    assert cs.stat[:, b].max() > 1.3 * cs.stat[d0, b]
    x, t0 = _capture(1234.5, -25.0, seed=4)
    assert A.carrier_lines(x, t0)[0].drift_hz_per_min == 0.0


# --- A-5 --------------------------------------------------------------------------------------

def test_a5_preamble_search_at_minus_25_db():
    """A2 on SHORT at -25 dB, every A-4 frequency x timing -1.9/0/+1.7 s:
    detected in every case, df < 0.05 Hz in every case, timing rms < T/20.

    Timing is held as an rms, not per case: at -25 dB the preamble's
    known linear part carries SNR ~ 17 dB in all, which bounds the timing
    sd at ~0.05 T (Cramer-Rao for this pulse); the estimate reaches it
    (rms 0.046 T measured over 30 seeds) and reports its own sd from the
    curvature, checked here too. Every case must still be within 3 T/20.
    """
    errs, sds = [], []
    for k, f in enumerate(FREQS):
        for j, shift in enumerate(SHIFTS):
            f_true = f + 0.21 * j
            x, t0 = _capture(f_true, -25.0, shift, seed=100 + 3 * k + j)
            cand = A.carrier_lines(x, t0)[0]
            hit = A.preamble_search(x, t0, cand)
            assert hit.passed, (f, shift, hit.cell_lam)
            assert hit.cell_lam > A.a2_threshold(n_chunks=hit.n_groups)
            assert abs(hit.f_hz - f_true) < 0.05, (f, shift, hit.f_hz)
            errs.append((hit.tau0_s - shift) / T_SYM)
            sds.append(hit.tau0_sd_s / T_SYM)
    errs = np.array(errs)
    assert np.all(np.abs(errs) < 3 / 20), errs
    assert np.sqrt(np.mean(errs ** 2)) < 1 / 20, errs
    assert abs(np.mean(errs)) < 0.025
    assert 0.6 < np.sqrt(np.mean(errs ** 2)) / np.mean(sds) < 1.6


def test_a5_preamble_search_at_minus_15_db():
    """10 dB stronger every case is within T/20, and the phase is u's."""
    for k, (f, shift) in enumerate([(1100.0, -1.9), (1900.0, 1.7), (340.0, 0.0),
                                    (2660.0, 0.4), (1500.0, -0.8)]):
        x, t0 = _capture(f, -15.0, shift, seed=200 + k)
        hit = A.preamble_search(x, t0, A.carrier_lines(x, t0)[0])
        assert hit.passed
        assert abs(hit.tau0_s - shift) < T_SYM / 20
        assert abs(hit.f_hz - f) < 0.01
    # the phase: rotate the whole capture by a known angle
    x, t0 = _capture(1500.0, -15.0, 0.0, seed=300)
    c = A.carrier_lines(x, t0)[0]
    p0 = A.preamble_search(x, t0, c).phase
    p1 = A.preamble_search(x * np.exp(1j * 1.0), t0, c).phase
    assert np.angle(np.exp(1j * (p1 - p0 - 1.0))) == pytest.approx(0, abs=0.05)


def test_a5_drift_through_the_preamble():
    """1.5 Hz/min (0.5 Hz across the preamble): found, f at t0 within
    0.05 Hz, the drift recovered from the chunk phases."""
    x, t0 = _capture(1650.0, -20.0, 0.9, seed=7, drift=1.5)
    hit = A.preamble_search(x, t0, A.carrier_lines(x, t0)[0])
    assert hit.passed
    assert hit.f_hz == pytest.approx(1650.0, abs=0.05)
    assert hit.drift_hz_per_min == pytest.approx(1.5, abs=0.5)
    assert abs(hit.tau0_s - 0.9) < T_SYM / 10


def test_a5_detection_record():
    x, t0 = _capture(1777.0, -20.0, 1.2, seed=8)
    hit = A.preamble_search(x, t0, A.carrier_lines(x, t0)[0])
    det = hit.to_detection(frame.SHORT.duration_s)
    assert isinstance(det, Detection) and det.method == "preamble"
    assert det.timing.tau0 == pytest.approx(1.2 * 250, abs=0.2)
    assert det.f_hz == hit.f_hz and math.isnan(det.z_ref)
    assert det.path.f_hz[1] == pytest.approx(hit.f_hz)


def test_a2_detects_at_minus_28_db():
    """R14's A2 point, on A2's side: SHORT preambles at -28 dB on a steady
    path, 30 frequencies, found by A1 and passed by A2 (f within 0.25 Hz,
    timing within T/4) in >= 90%. The 2 s chunks alone (the design's A2)
    pass ~75% here (measured 59/80 at the true frequency); the coherent
    4/10/20 s runs bring it to ~97% (78/80). At -31 dB: 21/40 against 4/40."""
    found = only_2s = 0
    for s in range(30):
        f = 1100.0 + 37.3 * s
        x, t0 = _capture(f, -28.0, 0.0, seed=900 + s)
        near = _best_near(A.carrier_lines(x, t0), f, 0.5)
        assert near is not None
        hit = A.preamble_search(x, t0, near)
        found += hit.passed and abs(hit.f_hz - f) < 0.25 and abs(hit.tau0_s) < T_SYM / 4
        chan = A.preamble_channel(x, t0, near.f_hz, near.drift_hz_per_min)
        only_2s += A.preamble_p(*A._preamble_sums(chan)[2:], scales=(1,))[0].min() < A.A2_PFA
    assert found >= 27, (found, only_2s)
    assert only_2s <= found - 3, (found, only_2s)


def test_a2_symbol_rate_alias_is_dropped():
    """A strong CE signal also passes A2 at f + 1/T (whole turns per
    symbol); `drop_symbol_rate_aliases` removes that hit and keeps the
    signal's own."""
    x, t0 = _capture(1500.0, 0.0, 0.7, seed=1)
    main = A.preamble_search(x, t0, CarrierCandidate(1500.0, 0.0, 0.0))
    alias = A.preamble_search(x, t0, CarrierCandidate(1500.0 + 1 / T_SYM, 0.0, 0.0))
    assert main.passed and alias.passed
    kept = A.drop_symbol_rate_aliases([alias, main])
    assert kept == [main]
    # a neighbour at 1/T with its own timing is not an alias
    other = A.preamble_search(x, t0, CarrierCandidate(1500.0 + 1 / T_SYM, 0.0, 0.0))
    other.tau0_s += 0.5
    assert len(A.drop_symbol_rate_aliases([other, main])) == 2


def _two_signals(df_hz, strong_db, weak_db, seed):
    """A strong CE signal at 1500 Hz and a weaker one df_hz above it (0.5 s
    later), in AWGN, blanked and normalised."""
    z = _clean()
    n = len(z)

    def tone(f, snr_db, shift_s):
        zz = np.roll(z, int(shift_s * FE_FS))
        turns = ch.nominal_turns(f - 1500.0, np.arange(n), FE_FS, PRE_S * FE_FS)
        return zz * np.exp(2j * np.pi * (turns - np.floor(turns))) * math.sqrt(
            FE_FS / 2500 * 10 ** (snr_db / 10))

    rng = np.random.default_rng(seed)
    x = (tone(1500.0, strong_db, 0.0) + tone(1500.0 + df_hz, weak_db, 0.5)
         + (rng.standard_normal(n) + 1j * rng.standard_normal(n)) / math.sqrt(2))
    x, keep = frontend.blank(x.astype(np.complex64))
    x, _ = frontend.normalise(x, keep)
    return x, PRE_S * FE_FS


@pytest.mark.parametrize("df_hz,strong_db,weak_db", [(25.0, -5.0, -22.0), (30.0, -5.0, -22.0),
                                                     (35.0, -8.0, -24.0)])
def test_a1_weak_neighbour_of_a_strong_signal(df_hz, strong_db, weak_db):
    """A weaker signal 25-35 Hz from a strong one is A1's line and A2's hit:
    A1 reports every local maximum, and the local CFAR keeps the strong
    signal's far sidebands under threshold instead."""
    x, t0 = _two_signals(df_hz, strong_db, weak_db, seed=5)
    lines = A.carrier_lines(x, t0)
    assert len(lines) < A.A1_MAX_CANDS
    weak = _best_near(lines, 1500.0 + df_hz, 0.5)
    assert weak is not None, [round(c.f_hz, 1) for c in lines]
    hits = A.drop_symbol_rate_aliases([h for h in (A.preamble_search(x, t0, c) for c in lines)
                                       if h.passed])
    assert _best_near(hits, 1500.0, 0.1) is not None
    w = _best_near(hits, 1500.0 + df_hz, 0.1)
    assert w is not None and abs(w.tau0_s - 0.5) < T_SYM / 4
    # no line 38-50 Hz out: the strong signal's sidebands (A1 used to report them)
    assert not [c for c in lines if 38 < abs(c.f_hz - 1500.0) < 50]


def _tilted_noise(n, tilt_db, seed):
    """Complex white noise with a linear dB tilt across FE (+-tilt/2 at
    +-1200 Hz from the centre), as a receiver passband that is not flat."""
    rng = np.random.default_rng(seed)
    X = np.fft.fft(rng.standard_normal(n) + 1j * rng.standard_normal(n))
    f = np.fft.fftfreq(n, 1 / FE_FS)
    g_db = tilt_db * np.clip(f / 1200, -1, 1) / 2
    return np.fft.ifft(X * 10 ** (g_db / 20)).astype(np.complex64)


def test_local_cfar_on_tilted_noise():
    """A 3 dB tilt across the band: `cfar_normalise` leaves every 100 Hz of
    it at mean 1 (within 3%), A3 finds nothing at p < 1e-3 in any channel
    and A1 no more lines than on flat noise. With the band-wide median
    alone this put 12 of 47 channels past p < 1e-3 (p down to 4e-10) and
    filled A1's 16."""
    spec = frame.SHORT
    n = int((PRE_S + spec.keyed_end_pos * T_SYM + 18) * FE_FS)
    x = _tilted_noise(n, 3.0, 1)
    x, keep = frontend.blank(x)
    x, _ = frontend.normalise(x, keep)
    t0 = PRE_S * FE_FS
    E, f, _ = A.tbd_emissions(x, t0, spec)
    means = (E[:, :9600] + 1).reshape(len(E), 24, 400).mean(axis=(0, 2))
    assert np.all(np.abs(means - 1) < 0.03), means
    raw = A.cfar_normalise(A.spectrogram(x, FE_FS, t0 + np.arange(40) * 2 * FE_FS,
                                         A.band_bins(FE_FS)[0]), local_bins=None)
    lo, hi = raw[:, :2000].mean(), raw[:, -2000:].mean()
    assert hi / lo > 1.4                         # the tilt is there to be removed
    hits = A.track_before_detect(x, t0, spec, all_hits=True)
    assert min(h.p_value for h in hits) > A.TBD_PFA
    assert len(A.carrier_lines(x, t0)) <= 12


# --- A-6 --------------------------------------------------------------------------------------

def _noise_fe(n, rng):
    return ((rng.standard_normal(n) + 1j * rng.standard_normal(n)) / math.sqrt(2)).astype(
        np.complex64)


def test_a6_a2_tail_on_noise():
    """~1e6 (tau0, df) cells of A2 on noise: at every scale (2, 4, 10 and
    20 s runs, Gamma(10), (5), (2), (1)) the fraction over the 1e-3 and
    1e-4 points is within a factor 2 of nominal; the combined, corrected
    p is under nominal (Bonferroni) by no more than the 4 scales allow
    (measured ~0.7 of nominal: the scales are correlated)."""
    rng = np.random.default_rng(11)
    lams, ps = {}, []
    for _ in range(40):
        x = _noise_fe(40 * FE_FS, rng)
        chan = A.preamble_channel(x, 8 * FE_FS, 1500 + rng.uniform(-900, 900))
        lam, _, S, var = A._preamble_sums(chan)
        p, _, st = A.preamble_p(S, var)
        assert np.allclose(st[0][1], lam)              # the 2 s scale is preamble_lambda's
        for k, lk in st:
            lams.setdefault(k, []).append(lk.ravel())
        ps.append(p.ravel())
    assert sorted(lams) == [1, 2, 5, 10]
    for k, v in lams.items():
        v = np.concatenate(v)
        assert len(v) > 1e6
        for p in (1e-3, 1e-4):
            r = np.mean(v > special.gammainccinv(k, p)) / p
            assert 0.5 < r < 2.0, (k, p, r)
    ps = np.concatenate(ps)
    for p in (1e-2, 1e-3, 1e-4):
        r = np.mean(ps < p) / p
        assert 1 / len(A.A2_SCALES) < r < 1.2, (p, r)


def test_a6_a1_tail_on_noise():
    """A1's statistic on noise (2.5e6 cells, every drift) against its Gamma:
    within a factor 2 at 1e-3 and 1e-4. Plain Gamma(M) is 2x light at 1e-4
    because the Hann frames overlap by half; `a1_gamma` widens it."""
    rng = np.random.default_rng(12)
    st = np.concatenate([A.carrier_line_stats(_noise_fe(30 * FE_FS, rng), 3 * FE_FS).stat.ravel()
                         for _ in range(20)])
    M = 11
    for p in (1e-3, 1e-4):
        r = np.mean(st > A.a1_threshold(M, p)) / p
        assert 0.5 < r < 2.0, (p, r)
    k, th = A.a1_gamma(M)
    assert k < M and th > 1 and k * th == pytest.approx(M)


def _lead_in_only(seed, snr_db, shift_s):
    """A 10 s lead-in ending at t0 - 8T with nothing after it, at 1637.3 Hz."""
    lead = _clean(lead_in_s=10.0)
    t = np.arange(len(lead)) / FE_FS - PRE_S
    return _capture(1637.3, snr_db, shift_s, seed=seed, z=np.where(t < -8 * T_SYM, lead, 0))


def test_a6_lead_in_only_carrier_gives_no_preamble(snr_db=10.0):
    """A 10 s lead-in that ends at t0 - 8T with nothing after it (and a plain
    carrier throughout), far above the noise: A1 sees the line, A2 finds
    nothing above threshold at any candidate; and at the lead-in's own
    line nothing at timing -1.9 and +1.7 s either, or at +30 dB."""
    spec = frame.SHORT
    n = 40 * FE_FS
    carrier = ce.baseband(np.zeros(spec.n_sym), spec, FE_FS, -PRE_S, n, keying=KEY,
                          lead_in_s=10.0)
    for x, t0 in (_lead_in_only(13, snr_db, 0.0), _capture(1637.3, snr_db, 0.0, seed=13,
                                                           z=carrier)):
        cands = A.carrier_lines(x, t0)
        assert _best_near(cands, 1637.3, 0.25) is not None
        hits = [A.preamble_search(x, t0, c) for c in cands]
        assert not any(h.passed for h in hits), [h.cell_lam for h in hits]
    for seed, snr, shift in ((13, 10.0, 1.7), (14, 10.0, -1.9), (15, 30.0, 1.7),
                             (16, 30.0, 0.0)):
        x, t0 = _lead_in_only(seed, snr, shift)
        h = A.preamble_search(x, t0, CarrierCandidate(1637.3, 0.0, 0.0))   # A1 may not see it
        assert not h.passed, (seed, snr, shift, h.p_value)


def test_a6_local_sigma_is_what_stops_a_lead_in_ending():
    """The lead-in's end is a step the notch cannot remove; with one global
    sigma^2 its transient passes A2 (p 1e-8 to 1e-11 at +1.7 s, measured),
    with the local level it reads as noise."""
    import dataclasses
    for seed in (13, 14):
        x, t0 = _lead_in_only(seed, 10.0, 1.7)
        chan = A.preamble_channel(x, t0, 1637.3)
        flat = dataclasses.replace(chan, local=np.full_like(chan.local, chan.sigma2))
        p_local = A.preamble_p(*A._preamble_sums(chan)[2:])[0].min()
        p_flat = A.preamble_p(*A._preamble_sums(flat)[2:])[0].min()
        assert p_local > A.A2_PFA and p_flat < A.A2_PFA, (seed, p_local, p_flat)


def test_a6_notch_and_zero_mean_templates():
    """The other two lead-in defences, each on its own: the notch takes a
    steady +10 dB carrier out of the MF output (its median power is the
    noise's), and every chunk template is zero-mean, so whatever carrier
    is left correlates to nothing at df = 0."""
    x, t0 = _capture(1637.3, 10.0, 0.0, seed=17, z=np.full(40 * FE_FS, 1.0 + 0j),
                     preprocess=False)
    n, _ = _capture(1637.3, 10.0, 0.0, seed=17, z=np.zeros(40 * FE_FS, complex),
                    preprocess=False)
    with_carrier = A.preamble_channel(x, t0, 1637.3).sigma2
    noise_only = A.preamble_channel(n, t0, 1637.3).sigma2
    assert with_carrier == pytest.approx(noise_only, rel=0.1)
    b, W, norm = A._templates(A.A2_DF_HZ)
    assert np.abs(b.sum(axis=1)).max() < 1e-9 * norm.min()
    i0 = int(np.argmin(np.abs(A.A2_DF_HZ)))
    assert np.abs(W[i0].sum(axis=1)).max() < 1e-9 * norm.min()


def test_lead_in_span_on_noise():
    """`lead_in_span` on noise: 0.0 in 300 of 300 noise captures (its
    first window's false alarm is e^-8 = 3.4e-4), and the per-window Z it
    thresholds is calibrated: with z_min = 2, P = e^-2 = 0.135."""
    rng = np.random.default_rng(18)
    spans, low = [], []
    for k in range(300):
        x = _noise_fe(30 * FE_FS, rng)
        f = 400.0 + 6.37 * k
        spans.append(A.lead_in_span(x, 20 * FE_FS, f))
        if k < 150:
            low.append(A.lead_in_span(x, 20 * FE_FS, f, z_min=2.0) > 0)
    assert max(spans) == 0.0
    assert 0.07 < np.mean(low) < 0.22


def test_a6_lead_in_does_not_move_timing():
    """With a 10 s lead-in at the signal's own level, A2's timing is the
    preamble's, not the lead-in's."""
    z = _clean(lead_in_s=10.0)
    for shift in (-1.9, 1.7):
        x, t0 = _capture(1500.0, -15.0, shift, seed=14, z=z)
        hit = A.preamble_search(x, t0, A.carrier_lines(x, t0)[0])
        assert hit.passed and abs(hit.tau0_s - shift) < T_SYM / 20


def test_lead_in_span():
    """The lead-in energy detector: 0, 3 and 10 s of lead-in at -15 dB read
    back to the second; at -20 dB the 10 s one still reads 10 s."""
    for lead, snr in ((0.0, -15.0), (3.0, -15.0), (10.0, -15.0), (10.0, -20.0)):
        x, t0 = _capture(1450.0, snr, 0.7, seed=3, z=_clean(lead_in_s=lead))
        h = A.preamble_search(x, t0, A.carrier_lines(x, t0)[0])
        assert A.lead_in_span(x, t0, h.f_hz, h.drift_hz_per_min, h.tau0_s) == lead


# --- A3 and A-7 ---------------------------------------------------------------------------------

def test_tbd_viterbi_follows_a_planted_path():
    rng = np.random.default_rng(15)
    T, B = 60, 81
    E = rng.exponential(1.0, (T, B)) - 1
    true = np.clip(40 + np.cumsum(rng.integers(-1, 2, T)), 0, B - 1)
    E[np.arange(T), true] += 3.0
    path, Z = A.tbd_viterbi(E)
    assert np.mean(path == true) > 0.9
    assert Z == pytest.approx(E[np.arange(T), path].sum() / math.sqrt(T))
    assert np.all(np.abs(np.diff(path)) <= A.TBD_MAX_STEP)
    # batched over leading axes gives the same answer
    p2, Z2 = A.tbd_viterbi(np.stack([E, E[:, ::-1]]))
    assert np.array_equal(p2[0], path) and Z2[0] == pytest.approx(Z)
    assert np.array_equal(p2[1], B - 1 - path)


def test_tbd_gumbel_table():
    for n, mu, be in zip(A.TBD_GUMBEL_N, A.TBD_GUMBEL_MU, A.TBD_GUMBEL_BETA):
        assert A.tbd_gumbel(n) == pytest.approx((mu, be))
    mu, _ = A.tbd_gumbel(500)
    assert A.TBD_GUMBEL_MU[2] < mu < A.TBD_GUMBEL_MU[3]
    assert A.gumbel_p(A.tbd_gumbel(132)[0] - 5, 132) > 0.999
    n = A.tbd_n_frames(frame.FULL)
    mu, be = A.tbd_gumbel(n)
    assert A.gumbel_p(mu + be * 6.907, n) == pytest.approx(1e-3, rel=0.01)
    # frame counts come from FrameSpec: the table's lengths are the presets'
    assert tuple(A.tbd_n_frames(frame.PRESETS[k])
                 for k in ("tiny", "short", "medium", "full")) == A.TBD_GUMBEL_N


CAL_FS = 500                         # calibration captures: 0.25 Hz bins, +-250 Hz


def _noise_tbd_z(n_frames, n_caps, seed, n_ch=6):
    """Z of the best path in n_ch disjoint 321-bin channels of n_caps
    complex white-noise captures, through production's spectrogram, CFAR
    and Viterbi (the CFAR median over all 2000 bins of the capture)."""
    rng = np.random.default_rng(seed)
    L = A._frame_len(CAL_FS)
    hop = int(A.STFT_HOP_S * CAL_FS)
    n = (n_frames - 1) * hop + L
    w = 2 * int(round(A.TBD_HALF_HZ / 0.25)) + 1
    out = []
    for _ in range(n_caps):
        x = (rng.standard_normal(n) + 1j * rng.standard_normal(n)) / math.sqrt(2)
        P = np.fft.fftshift(A.spectrogram(x, CAL_FS, np.arange(n_frames) * hop), axes=1)
        E = A.cfar_normalise(P) - 1
        st = np.stack([E[:, 37 + k * w:37 + (k + 1) * w] for k in range(n_ch)])
        out.extend(A.tbd_viterbi(st)[1])
    return np.array(out)


def _gumbel_moments(Z):
    be = np.std(Z) * math.sqrt(6) / math.pi
    return float(np.mean(Z) - np.euler_gamma * be), float(be)


def _gumbel_point(mu, be, p):
    """z with Gumbel survival p."""
    return mu - be * math.log(-math.log1p(-p))


def test_a7_tbd_gumbel_fast_subsample():
    """20 noise captures at SHORT length (120 Z values) agree with the
    committed Gumbel: mean within 0.2, sd within 25%, and its 5% point is
    not exceeded by more than 10% of them (the fit is conservative)."""
    n = A.tbd_n_frames(frame.SHORT)
    Z = _noise_tbd_z(n, 20, seed=21)
    mu, be = A.tbd_gumbel(n)
    assert len(Z) == 120
    assert np.mean(Z) == pytest.approx(mu + np.euler_gamma * be, abs=0.2)
    assert np.std(Z) == pytest.approx(be * math.pi / math.sqrt(6), rel=0.25)
    assert np.mean(Z > _gumbel_point(mu, be, 0.05)) <= 0.10


def test_a7_production_tbd_matches_calibration_on_noise():
    """Production A3 on a noise-only FE slot (the band-wide CFAR, every
    channel) gives Z from the same Gumbel; no hit at p < 1e-3."""
    spec = frame.SHORT
    rng = np.random.default_rng(22)
    n = int((PRE_S + spec.keyed_end_pos * T_SYM + 3) * FE_FS)
    zs, ps = [], []
    for _ in range(2):
        hits = A.track_before_detect(_noise_fe(n, rng), PRE_S * FE_FS, spec, all_hits=True)
        zs += [h.z for h in hits]
        ps += [h.p_value for h in hits]
        assert all(h.n_frames == A.tbd_n_frames(spec) for h in hits)
    mu, be = A.tbd_gumbel(A.tbd_n_frames(spec))
    assert len(zs) > 60
    assert np.mean(zs) == pytest.approx(mu + np.euler_gamma * be, abs=0.4)
    assert sum(p < A.TBD_PFA for p in ps) <= 1


@pytest.mark.slow
def test_a7_measure_tbd_gumbel():
    """The measurement behind TBD_GUMBEL_*: 200 noise-only captures per
    frame length, 6 channels each, Gumbel by moments. Re-measured with
    other seeds the committed values hold to 0.1 (mu) and 10% (beta), and
    the fit stays conservative: no more than its share past its 10% and
    1% points."""
    for n, mu_c, be_c in zip(A.TBD_GUMBEL_N, A.TBD_GUMBEL_MU, A.TBD_GUMBEL_BETA):
        Z = _noise_tbd_z(n, 200, seed=1000 + n)
        mu, be = _gumbel_moments(Z)
        tail = [float(np.mean(Z > _gumbel_point(mu_c, be_c, p)) / p) for p in (0.1, 0.01)]
        print(f"n_frames={n}: mu={mu:.3f} beta={be:.3f} (committed {mu_c}, {be_c}); "
              f"exceedance/nominal at 10%, 1%: {tail[0]:.2f}, {tail[1]:.2f}")
        assert mu == pytest.approx(mu_c, abs=0.1)
        assert be == pytest.approx(be_c, rel=0.1)
        assert tail[0] < 1.25 and tail[1] < 1.0


def test_a3_finds_and_follows_a_drifting_signal():
    """SHORT at -30 dB, +233.3 Hz, 1 Hz/min, 10 s lead-in, through the
    channel simulator: A3 detects it at p < 1e-3, f at t0 within 0.2 Hz,
    and the path stays on the carrier (rms < 0.5 Hz while keyed)."""
    spec = frame.SHORT
    rng = np.random.default_rng(2)
    sym = frame.assemble(spec, rng.integers(0, 2, 2474).astype(np.uint8),
                         rng.standard_normal(spec.n_data))
    sim = ch.simulate(sym, spec, Q_TEST, ChannelConfig(
        snr_db=-30.0, freq_offset_hz=233.3, drift_hz_per_min=1.0, seed=4),
        keying=KEY, lead_in_s=10.0)
    x, keep = frontend.blank(sim.fe)
    x, _ = frontend.normalise(x, keep)
    hits = A.track_before_detect(x, sim.t0_index, spec)
    assert hits and hits[0].passed and hits[0].p_value < 1e-6
    h = hits[0]
    assert h.f_hz == pytest.approx(1733.3, abs=0.2)
    keyed = (h.path.t_s > -9.0) & (h.path.t_s < spec.keyed_end_pos * T_SYM - 2)
    f_true = np.interp(h.path.t_s, sim.truth.t_pos_s, sim.truth.f_hz)
    assert np.sqrt(np.mean((h.path.f_hz - f_true)[keyed] ** 2)) < 0.5
    assert h.line()[0] * 60 == pytest.approx(1.0, abs=0.1)        # Hz/min
    det = h.to_detection()
    assert det.method == "tbd" and det.timing is None


def test_acquire_end_to_end():
    """`acquire` on a simulated SHORT slot at -20 dB, -312.5 Hz, 10 s
    lead-in: one preamble hit at the right frequency and timing, the A3
    hit at the same frequency, and both offered to the gate."""
    spec = frame.SHORT
    rng = np.random.default_rng(5)
    sym = frame.assemble(spec, rng.integers(0, 2, 2474).astype(np.uint8),
                         rng.standard_normal(spec.n_data))
    sim = ch.simulate(sym, spec, Q_TEST, ChannelConfig(
        snr_db=-20.0, freq_offset_hz=-312.5, seed=6), keying=KEY, lead_in_s=10.0)
    cap = frontend.Capture(fe=sim.fe, q=sim.q, t0_index=sim.t0_index)
    acq = A.acquire(cap, spec)
    near = [h for h in acq.preamble if abs(h.f_hz - 1187.5) < 1]
    assert near and abs(near[0].f_hz - 1187.5) < 0.05
    assert abs(near[0].tau0_s) < T_SYM / 20
    assert any(abs(h.f_hz - 1187.5) < 0.3 for h in acq.tbd)
    dets = acq.detections(spec)
    assert dets[0].method == "preamble" and any(d.method == "tbd" for d in dets)
    assert dets[0].lead_in_s == 10.0
    live = A.acquire(cap, spec, live_only=True)
    assert live.tbd == [] and len(live.preamble) == len(acq.preamble)

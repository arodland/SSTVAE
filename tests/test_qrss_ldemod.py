"""Waveform L's demodulator (`sstvae.qrss.ldemod`; design L-receiver 5).

All on one genie: an L_MEDIUM frame through the real transmitter
(`lwave.frame_baseband`, clipper included), `to_data_centre` and L's
matched filter at every true position, then complex AWGN of a stated N
at the matched-filter output, with the tracker's contract filled in by
hand: u the references' gain, P = 0, psi = 1 (`GenieTrack`). Noise is
stated as s2n, the data's per-latent noise in l units.

- the symbol estimates' formula, P and psi included (the genie has
  neither);
- latent SINR against the genie theory to 0.3 dB, with the header
  unknown (the copy as rho of extra variance) and known (subtracted), and
  W honest;
- the header from the block alone and from the spread copy alone, both
  LLRs consistent (scaled as CE's);
- a known header leaves no copy in y and E y_x^2 back at 1 + D_x + s2n_x;
- kappa near 1 on a correctly scaled genie, following a noise 1.5x the
  stated one (strong and weak passes, and through extract_l), a suspect
  pass left unscaled; each of the two estimates unbiased over noise draws
  with its stated variance; a reference the tracker measured allowed its
  pull on u (an L tracker leans on its references: CE's form read kappa
  0.2-0.6 on fading passes);
- an L pass and a CE pass (`qrss_fakes.noisy_llr`) too weak alone decode
  together;
- erasures, a frame without a header, CE frames refused, and a given m
  used in place of the track's;
- the pinned constants against this frame's own loopback.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from qrss_fakes import L_CALL, L_HDR, noisy_llr
from qrss_helpers import Q_TEST, latent_snr_db, unit_rms_latents
from sstvae.qrss import demod, frame, header, ldemod, lframe, lwave, morse
from sstvae.qrss.constants import N_HDR_BITS
from sstvae.qrss.precoder import precode

SPEC = lframe.L_MEDIUM
RHO = SPEC.hdr_rho


class GenieTrack:
    """The fields `extract_l` reads of an `ltrack.LTrackResult`."""

    def __init__(self, spec, pos, m, u, N, hdr_bits=None, psi=None):
        n = len(pos)
        self.spec, self.pos, self.m = spec, pos, m
        self.u = np.full(n, u, dtype=np.complex128) if np.ndim(u) == 0 else u
        self.P = np.zeros(n)
        self.N = np.full(n, float(N))
        self.psi = np.ones(n) if psi is None else psi
        self.classes = SimpleNamespace(hdr_bits=hdr_bits)

    def gi(self, p):
        return np.asarray(p, dtype=np.int64) - int(self.pos[0])


class Genie:
    """One noiseless L frame at the matched filter, and noisy tracks of it."""

    def __init__(self, spec, seed=3, keying=None):
        self.spec = spec
        self.lay = lframe.layout(spec)
        self.a = unit_rms_latents(spec.n_data, seed)
        self.bits = header.encode(L_HDR) if spec.has_header else None
        self.sym = lframe.assemble(spec, self.bits, precode(self.a, Q_TEST))
        bb = lwave.frame_baseband(self.sym, spec, keying=keying)
        self.pos = np.arange(spec.n_pos, dtype=np.int64)
        self.m0 = lwave.matched_filter(lwave.to_data_centre(bb), bb.fs, bb.n0, self.pos)
        ref = self.lay.ref
        m_ref = self.m0[self.lay.pos[ref]]
        self.u = complex(np.vdot(self.sym[ref], m_ref) / np.vdot(self.sym[ref], self.sym[ref]))

    def track(self, s2n, seed=0, known=False, noise=1.0, **kw) -> GenieTrack:
        """Noise N giving the data s2n per latent (l units); `noise` scales
        the actual variance against the stated N."""
        N = s2n * ldemod.G_DATA_L ** 2 * abs(self.u) ** 2
        rng = np.random.default_rng(seed)
        n = len(self.pos)
        w = np.sqrt(N * noise / 2) * (rng.standard_normal(n) + 1j * rng.standard_normal(n))
        return GenieTrack(self.spec, self.pos, self.m0 + w, self.u, N,
                          hdr_bits=self.bits if known else None, **kw)


@pytest.fixture(scope="module")
def genie():
    return Genie(SPEC, keying=morse.keying_units(L_CALL))


def consistency(llr, bits) -> float:
    """var(s llr) / (2 mean(s llr)), s = +-1 the true bit's sign: 1 for an
    LLR scaled as log P(0)/P(1) of a Gaussian observation."""
    s = (1.0 - 2.0 * np.asarray(bits, dtype=np.float64)) * np.asarray(llr, dtype=np.float64)
    return float(np.var(s) / (2 * np.mean(s)))


def decodes(llr) -> bool:
    return header.decode(np.asarray(llr, dtype=np.float64)) == L_HDR


def cnormal(rng, var, n):
    """Circular complex Gaussian, E|x|^2 = var."""
    return np.sqrt(np.asarray(var) / 2) * (rng.standard_normal(n) + 1j * rng.standard_normal(n))


# --- symbol estimates -------------------------------------------------------------------

def test_symbol_estimates_follow_the_formula():
    """r = m conj(u) / (|u|^2 psi g), s2n = (N + g^2 P) / (g^2 |u|^2 psi^2),
    from every field of the track (the genies below have P = 0 and psi = 1),
    and erased (r 0, s2n inf) where psi < 0.5 or |u|^2 is under U_MIN_REL of
    its median over positions >= 0 (a lead-in grid included)."""
    spec = lframe.L_TINY
    lay = lframe.layout(spec)
    rng = np.random.default_rng(21)
    pos = np.arange(-12, spec.n_pos, dtype=np.int64)
    n = len(pos)
    u = rng.uniform(0.5, 2.0, n) * np.exp(2j * np.pi * rng.uniform(size=n))
    tr = GenieTrack(spec, pos, cnormal(rng, 1.0, n), u, 1.0, psi=rng.uniform(0.3, 1.0, n))
    tr.P = rng.uniform(0.0, 0.5, n)
    tr.N = rng.uniform(0.1, 1.0, n)
    tr.u[tr.gi(lay.pos[lay.data[7]])] = 1e-3          # |u|^2 1e-6 of the median
    g = 0.7
    r, s2n = ldemod.symbol_estimates_l(tr, lay.data, g)
    gi = tr.gi(lay.pos[lay.data])
    m, uu, P, N, psi = tr.m[gi], tr.u[gi], tr.P[gi], tr.N[gi], tr.psi[gi]
    gone = (psi < 0.5) | (np.arange(len(gi)) == 7)
    assert 0 < gone.sum() < len(gi)
    u2 = np.abs(uu) ** 2
    np.testing.assert_allclose(r[~gone], (m * np.conj(uu) / (u2 * psi * g))[~gone], rtol=1e-12)
    np.testing.assert_allclose(s2n[~gone], ((N + g * g * P) / (g * g * u2 * psi ** 2))[~gone],
                               rtol=1e-12)
    assert np.all(r[gone] == 0) and np.all(np.isinf(s2n[gone]))


# --- latents ----------------------------------------------------------------------------

@pytest.mark.parametrize("s2n", [0.02, 0.3])
@pytest.mark.parametrize("known", [False, True])
def test_latent_sinr_and_w_on_the_genie(genie, s2n, known):
    """SINR of z against the air-order latents within 0.3 dB of
    -10 log10((1 + rho)(s2n + D_PASS_L) + extra), extra = rho while the
    header is unknown, and W honest (mean (z - a)^2 w near 1)."""
    tr = genie.track(s2n, seed=1, known=known)
    z, w, llr, diag = ldemod.extract_l(None, tr, SPEC, Q_TEST)
    assert z.dtype == w.dtype == llr.dtype == np.float32
    assert z.shape == w.shape == (SPEC.n_data,) and llr.shape == (N_HDR_BITS,)
    extra = 0.0 if known else RHO
    theory = -10 * np.log10((1 + RHO) * (s2n + ldemod.D_PASS_L) + extra)
    assert latent_snr_db(z, genie.a) == pytest.approx(theory, abs=0.3)
    assert np.mean((z - genie.a) ** 2 * w) == pytest.approx(1.0, abs=0.08)
    assert diag["s2"] == pytest.approx(diag["kappa"] * (1 + RHO) * s2n
                                       + ldemod.d_pass_x(SPEC) + extra)


def test_a_known_header_leaves_no_copy_in_y(genie):
    """With the header known the demodulator subtracts sqrt(rho) h (the
    tracker cannot remove a linear copy): y_x's projection on h is 0, not
    sqrt(rho), E y_x^2 is back at 1 + D_x + s2n_x, and there is no spread
    LLR."""
    s2n = 0.01
    h = frame.spread_signs(SPEC, genie.bits)
    for known, coef in ((False, np.sqrt(RHO)), (True, 0.0)):
        z, w, llr, diag = ldemod.extract_l(None, genie.track(s2n, seed=2, known=known),
                                           SPEC, Q_TEST)
        y = diag["y"]
        assert np.dot(y, h) / np.dot(h, h) == pytest.approx(coef, abs=0.01)
        assert (diag["llr_spread"] is None) == known
    assert np.mean(y ** 2) == pytest.approx(1 + (1 + RHO) * (s2n + ldemod.D_PASS_L), abs=0.025)
    assert decodes(diag["llr_block"])


# --- the header -------------------------------------------------------------------------

def test_the_header_decodes_from_each_copy_alone(genie):
    """Header unknown: the block, the spread copy and their sum each decode,
    and both parts' LLRs are consistent (they mean what CE's do)."""
    z, w, llr, diag = ldemod.extract_l(None, genie.track(1.0, seed=3), SPEC, Q_TEST)
    block, spread = diag["llr_block"], diag["llr_spread"]
    assert decodes(block) and decodes(spread) and decodes(llr)
    np.testing.assert_allclose(llr, (block + spread).astype(np.float32), rtol=1e-6)
    # the spread copy's noise is the data's s2 less the copy itself (rho)
    np.testing.assert_allclose(spread, demod.spread_llr(diag["y"], diag["s2"] - RHO, SPEC),
                               rtol=1e-12)
    assert consistency(block, genie.bits) == pytest.approx(1.0, abs=0.12)
    assert consistency(spread, genie.bits) == pytest.approx(1.0, abs=0.12)


def test_the_header_block_is_scaled_by_its_own_gain(genie):
    """On a strong pass the header symbols come through at G_HDR_L of the
    references (not the data's 0.881: that would over-scale the block's
    LLRs against the spread copy and CE's), and the block's LLRs are
    consistent with D_HDR_L as their distortion."""
    z, w, llr, diag = ldemod.extract_l(None, genie.track(0.02, seed=9), SPEC, Q_TEST)
    s = 1.0 - 2.0 * genie.bits
    assert np.mean(s * diag["y_hdr"]) == pytest.approx(1.0, abs=0.03)
    assert consistency(diag["llr_block"], genie.bits) == pytest.approx(1.0, abs=0.1)


def test_l_and_ce_header_llrs_add(genie):
    """An L pass and a CE pass of one segment, each too weak to decode its
    header, decode it together: the LLRs share CE's scale."""
    z, w, llr_l, _ = ldemod.extract_l(None, genie.track(25.0, seed=200), SPEC, Q_TEST)
    llr_ce = noisy_llr(L_HDR, -14.5, np.random.default_rng(300))
    assert consistency(llr_l, genie.bits) == pytest.approx(1.0, abs=0.12)
    assert not decodes(llr_l) and not decodes(llr_ce)
    assert decodes(llr_l.astype(np.float64) + llr_ce)


# --- calibration ------------------------------------------------------------------------

@pytest.mark.parametrize("s2n", [0.02, 0.3])
def test_kappa_near_one_on_a_scaled_genie(genie, s2n):
    kappa, suspect = ldemod.calibrate_l(genie.track(s2n, seed=4))
    assert not suspect and kappa == pytest.approx(1.0, abs=0.15)


def test_kappa_follows_the_noise(genie):
    """Noise 1.5x the stated variance: kappa 1.5, on a weak pass too (every
    s2n over KAPPA_S2_MAX: the relative bound keeps it calibrated), and
    extract_l's W and header LLRs honest with it; 3x: clamped and suspect."""
    kappa, suspect = ldemod.calibrate_l(genie.track(0.3, seed=5, noise=1.5))
    assert not suspect and kappa == pytest.approx(1.5, abs=0.1)
    weak = genie.track(8.0, seed=5, noise=1.5)
    assert np.all(ldemod.data_estimates_l(weak)[1] > ldemod.KAPPA_S2_MAX)
    kappa, suspect = ldemod.calibrate_l(weak)
    assert not suspect and kappa == pytest.approx(1.5, abs=0.05)
    z, w, llr, diag = ldemod.extract_l(None, genie.track(0.3, seed=6, noise=1.5), SPEC, Q_TEST)
    assert np.mean((z - genie.a) ** 2 * w) == pytest.approx(1.0, abs=0.08)
    assert consistency(diag["llr_block"], genie.bits) == pytest.approx(1.0, abs=0.12)
    kappa, suspect = ldemod.calibrate_l(genie.track(0.3, seed=5, noise=3.0))
    assert suspect and kappa == ldemod.KAPPA_RANGE[1]


def test_a_suspect_pass_is_left_unscaled(genie):
    """As CE's: a pass whose kappa measured outside KAPPA_RANGE keeps its
    stated noise (scale 1), and reports the clamped kappa."""
    tr = genie.track(0.3, seed=5, noise=3.0)
    z, w, llr, diag = ldemod.extract_l(None, tr, SPEC, Q_TEST)
    assert diag["suspect"] and diag["kappa"] == ldemod.KAPPA_RANGE[1]
    y, s2n, extra = ldemod.data_estimates_l(tr)
    np.testing.assert_allclose(diag["s2"], s2n + ldemod.d_pass_x(SPEC) + extra, rtol=1e-12)


def test_the_kappa_estimates_are_unbiased_with_honest_variances(genie):
    """Over 120 noise draws at s2n 1: each of calibrate_l's two estimates
    averages 1 and spreads as its stated variance says (the data's a little
    less: these latents are unit-RMS exactly, so their own square adds no
    spread), which is what their inverse-variance combination relies on."""
    est = np.array([[x for kv in ldemod.kappa_estimates_l(genie.track(1.0, seed=500 + i))
                     for x in kv] for i in range(120)])
    k_ref, v_ref, k_dat, v_dat = est.T
    assert k_ref.mean() == pytest.approx(1.0, abs=0.03)
    assert k_dat.mean() == pytest.approx(1.0, abs=0.03)
    assert 0.85 < k_ref.std(ddof=1) / np.sqrt(v_ref.mean()) < 1.25
    assert 0.6 < k_dat.std(ddof=1) / np.sqrt(v_dat.mean()) < 1.25


def test_a_measured_reference_is_allowed_its_pull_on_u(genie):
    """A tracker's gain at a reference it measured leans towards that
    reference's own noise (leverage h = P0/(P0 + R), posterior P = h R), so
    the residual's variance is R - P, not R + P. Built so exactly (a prior
    of variance N, so h is about 0.46 and P about half N): with `tr.meas`
    saying so, the reference estimate reads 1; without it (CE's form)
    about 0.3. A gain independent of the references' noise (error variance
    P, not measured there) reads 1 in CE's form."""
    lay = genie.lay
    rng = np.random.default_rng(12)
    tr = genie.track(0.2, seed=11)
    gi = tr.gi(lay.pos[lay.ref])
    a = genie.sym[lay.ref]
    R = tr.N[gi] + abs(genie.u) ** 2 * ldemod.D_REF_L
    P0 = tr.N[gi]
    prior = genie.u + cnormal(rng, P0, len(gi))
    h = P0 / (P0 + R)
    tr.u[gi] = prior + h * (tr.m[gi] / a - prior)
    tr.P[gi] = h * R
    tr.meas = np.zeros(len(tr.pos), dtype=bool)
    tr.meas[gi] = True
    assert ldemod.kappa_estimates_l(tr)[0][0] == pytest.approx(1.0, abs=0.1)
    tr.meas = None
    assert ldemod.kappa_estimates_l(tr)[0][0] == pytest.approx(0.3, abs=0.1)
    tr.u[gi] = genie.u + cnormal(rng, h * R, len(gi))
    assert ldemod.kappa_estimates_l(tr)[0][0] == pytest.approx(1.0, abs=0.1)


def test_the_combination_is_ces():
    """`demod.calibrate`'s rules: inverse variance, shrunk to 1 with prior sd
    KAPPA_PRIOR_SD, clamped to KAPPA_RANGE, suspect 3 sd outside it."""
    assert ldemod._combine([]) == (1.0, False)
    w0 = 1.0 / ldemod.KAPPA_PRIOR_SD ** 2
    kappa, suspect = ldemod._combine([(1.5, 0.25), (1.2, 0.5)])
    wsum = 1 / 0.25 + 1 / 0.5
    assert kappa == pytest.approx((1.5 * 4 + 1.2 * 2 + w0) / (wsum + w0)) and not suspect
    assert ldemod._combine([(3.0, 0.01)]) == (ldemod.KAPPA_RANGE[1], True)
    assert ldemod._combine([(0.1, 0.01)]) == (ldemod.KAPPA_RANGE[0], True)
    assert ldemod._combine([(2.2, 0.01)]) == (ldemod.KAPPA_RANGE[1], False)


# --- erasures, frames -------------------------------------------------------------------

def test_erasures(genie):
    """psi < 0.5 or a vanishing gain erases the symbol: both of its latents
    get s2 = inf and y = 0, and a block of nothing but erasures has w = 0."""
    tr = genie.track(0.3, seed=6)
    data_pos = genie.lay.pos[genie.lay.data]
    tr.psi[data_pos[:32]] = 0.3                    # data symbols 0..31: latents 0..63
    tr.u[data_pos[100]] = 1e-3 * genie.u           # data symbol 100: latents 200, 201
    z, w, llr, diag = ldemod.extract_l(None, tr, SPEC, Q_TEST)
    gone = np.zeros(SPEC.n_data, dtype=bool)
    gone[:64] = gone[200:202] = True
    assert np.all(np.isinf(diag["s2"][gone])) and np.all(np.isfinite(diag["s2"][~gone]))
    assert np.all(diag["y"][gone] == 0)
    assert np.all(w[:64] == 0) and np.all(z[:64] == 0)    # the first precoder block
    assert np.all(w[64:] > 0)
    # with the header known an erased latent is not left holding -sqrt(rho) h
    tr = genie.track(0.3, seed=6, known=True)
    tr.psi[data_pos[:32]] = 0.3
    z, w, llr, diag = ldemod.extract_l(None, tr, SPEC, Q_TEST)
    assert np.all(diag["y"][:64] == 0) and np.all(np.isinf(diag["s2"][:64]))


def test_a_given_m_replaces_the_tracks():
    """As demod.extract: a non-None m is used (and kept) in place of the
    track's own matched filter."""
    g = Genie(lframe.L_TINY, seed=7)
    _, _, _, d1 = ldemod.extract_l(None, g.track(0.1, seed=8), lframe.L_TINY, Q_TEST)
    tr = g.track(0.1, seed=8)
    m2 = 2.0 * tr.m
    _, _, _, d2 = ldemod.extract_l(m2, tr, lframe.L_TINY, Q_TEST, calibrate_kappa=False)
    np.testing.assert_allclose(d2["y"], 2.0 * d1["y"], rtol=1e-12)
    np.testing.assert_array_equal(tr.m, m2)


def test_a_frame_without_a_header():
    """L_TINY: no header block, no spread copy (rho 0): y_x is the plain
    unpaired estimate and the LLRs are empty, as CE's."""
    g = Genie(lframe.L_TINY, seed=7)
    tr = g.track(0.1, seed=8)
    z, w, llr, diag = ldemod.extract_l(None, tr, lframe.L_TINY, Q_TEST)
    assert llr.shape == (0,) and diag["llr_block"] is None and diag["llr_spread"] is None
    r = tr.m[g.lay.pos[g.lay.data]] / (tr.u[0] * ldemod.G_DATA_L)
    np.testing.assert_allclose(diag["y"], lframe.unpair(r), rtol=1e-12)
    assert latent_snr_db(z, g.a) == pytest.approx(
        -10 * np.log10(0.1 + ldemod.D_PASS_L), abs=0.5)


def test_ce_frames_are_refused(genie):
    tr = genie.track(0.3)
    tr.spec = frame.SHORT
    with pytest.raises(AssertionError, match="L's"):
        ldemod.calibrate_l(tr)
    with pytest.raises(AssertionError, match="L's"):
        ldemod.kappa_estimates_l(tr)
    with pytest.raises(AssertionError, match="L's"):
        ldemod.extract_l(None, tr, frame.SHORT, Q_TEST)


# --- the pinned constants ---------------------------------------------------------------

def test_the_constants_against_the_loopback(genie):
    """This frame's own clipper figures (`lwave.loopback_stats`, with a
    lead-in) sit inside the spreads the constants were pinned from."""
    st = lwave.loopback_stats(genie.sym, SPEC, keying=morse.keying_units(L_CALL),
                              lead_in_s=2.0)
    gr = st["gain_ref"]
    assert st["data_vs_ref"] == pytest.approx(ldemod.G_DATA_L, rel=0.015)
    assert abs(st["gain_hdr"] / gr) == pytest.approx(ldemod.G_HDR_L, rel=0.015)
    assert abs(st["gain_pre"] / gr) == pytest.approx(ldemod.G_PRE_L, rel=0.015)
    assert abs(gr) == pytest.approx(ldemod.G_REF_ABS, rel=0.015)
    res = st["resid"]
    assert res["data"] == pytest.approx(ldemod.D_PASS_L, rel=0.06)
    assert res["ref"] == pytest.approx(ldemod.D_REF_L, rel=0.1)
    assert res["hdr"] == pytest.approx(ldemod.D_HDR_L, rel=0.1)
    assert res["pre"] == pytest.approx(ldemod.D_PRE_L, rel=0.02)
    gp = {k: abs(v) for k, v in st["gain_pilot"].items()}
    assert set(gp) == {"lead", "pre", "hdr", "data", "id"}
    assert gp["pre"] == pytest.approx(ldemod.PILOT_GAIN_PRE, rel=0.003)
    assert gp["hdr"] == pytest.approx(ldemod.PILOT_GAIN_HDR, rel=0.015)
    assert gp["data"] == pytest.approx(ldemod.PILOT_GAIN_DATA, rel=0.015)
    assert gp["id"] == pytest.approx(ldemod.PILOT_GAIN_ID, abs=0.002)
    assert gp["lead"] == pytest.approx(ldemod.PILOT_GAIN_ID, abs=0.002)

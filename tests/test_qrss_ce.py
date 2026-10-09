"""Waveform CE: pulse, phase synthesis, callsign windows, lead-in, spectrum
and the genie loopback (design 10.2: M1-M5, M8, M9).

All fast except M4 (slow, codec). Tolerances are the design's; where a
measured value is pinned (the per-pass distortion D_PASS), the pinned
number is stated next to the assertion so WP6 can quote it.
"""

import numpy as np
import pytest
from scipy.signal import welch

from qrss_helpers import (
    Q_TEST, REPO_ROOT, latent_snr_db, load_codec, require_coco, unit_rms_latents,
)
from sstvae.qrss import ce, frame, morse, precoder, sequences
from sstvae.qrss.constants import (
    A0, BETA, CARRIER_FRAC, CW_UNIT, CW_UNITS, K_LIN, T_SYM, USEFUL_FRAC,
)

BETA_F = float(BETA)
CALL = "K1ABC/P"


def _header_bits(seed=11):
    return np.random.default_rng(seed).integers(0, 2, 2474).astype(np.uint8)


def _frame(spec=frame.SHORT, seed=0, q=Q_TEST, dist="gauss"):
    """(sym, air latents) of a frame with synthetic unit-RMS latents."""
    a = unit_rms_latents(spec.n_data, seed=seed, dist=dist)
    x = precoder.precode(a, q)
    hdr = _header_bits() if spec.n_hdr else None
    return frame.assemble(spec, hdr, x), a


# --- pulse ---------------------------------------------------------------------------

def test_pulse_constants():
    """Design 2.4's checked values: c, p(0), G0; the table is the pulse on m/485."""
    assert ce.PULSE_NORM == pytest.approx(0.9999137, abs=1e-7)
    assert ce.pulse(0.0) == pytest.approx(1.041076, abs=1e-6)
    assert ce.G0 == pytest.approx(1.001665, abs=1e-6)
    m = np.arange(-3880, 3881)
    assert np.array_equal(ce.PULSE_TABLE, ce.pulse(m / 485))
    assert np.sum(ce.PULSE_TABLE ** 2) / 485 == pytest.approx(1.0, abs=1e-14)
    assert ce.pulse(8.0001) == 0.0 and ce.pulse(-8.5) == 0.0
    # The removable singularity at 1/(4 alpha) is continuous.
    t = 5 / 3
    assert ce.pulse(t) == pytest.approx(ce.pulse(t + 1e-6), abs=1e-5)
    # Nyquist-ish: RRC * RRC is a raised cosine with zeros at nonzero integers.
    rc = np.convolve(ce.PULSE_TABLE, ce.PULSE_TABLE) / 485
    centre = len(rc) // 2
    assert rc[centre] == pytest.approx(1.0, abs=1e-12)
    assert np.max(np.abs(rc[centre + 485 * np.arange(1, 8)])) < 2e-3


def test_matched_filter_of_a_constant_is_g0():
    """Exactly G0 on the 16 kHz grid; within 1e-3 on the 250 Hz grid (the cut
    pulse is not strictly band-limited, so the 7.6-sample Riemann sum ripples)."""
    m = ce.matched_filter(np.ones(16000 * 2), 16000, -16000, np.arange(-5, 5))
    assert np.allclose(m, ce.G0, atol=1e-12)
    m = ce.matched_filter(np.ones(4000), 250, -1000, np.arange(0, 200))
    assert np.allclose(m, ce.G0, rtol=1e-3)


# --- M1: exact grid against closed form -------------------------------------------------

@pytest.mark.parametrize("fs", [16000, 8000, 4000, 250])
def test_m1_phase_grid_equals_phase_at(fs):
    """M1: phase_grid at 16000/8000/4000/250 Hz equals phase_at to < 1e-12 rad.

    Checked over the frame's first 30 s. phase_at takes tau as a float, whose
    own rounding grows with tau (one ulp at tau = 50,000 is 7e-12 symbols),
    so later in the frame the agreement is bounded by that input rounding:
    checked across the first callsign window against 4 rad/symbol x ulp(tau).
    """
    spec = frame.SHORT
    sym, _ = _frame(spec)
    L = 16000 // fs
    n0, n = -fs, 30 * fs
    g = ce.phase_grid(fs, n0, n, sym, spec)
    tau = np.arange(n0, n0 + n, dtype=np.int64) * L / 485
    a, _ = ce.phase_at(tau, sym, spec)
    assert np.max(np.abs(g - a)) < 1e-12

    p_w = spec.win_start_pos[0]
    n0 = int((p_w - 40) * 485 / L)
    n = int(460 * 485 / L)
    g = ce.phase_grid(fs, n0, n, sym, spec)
    tau = np.arange(n0, n0 + n, dtype=np.int64) * L / 485
    a, _ = ce.phase_at(tau, sym, spec)
    assert np.max(np.abs(g - a)) < 4 * np.spacing(tau.max()) * 4


def test_phase_variance_and_var_grid():
    """phi has time-averaged variance ~beta^2; var_grid is the E[phi^2] of unit-variance data."""
    spec = frame.SHORT
    sym, _ = _frame(spec)
    lay = frame.layout(spec)
    p0 = lay.pos[lay.data[0]] + 20
    fs = 250
    n0, n = int(p0 * 485 / 64), 40 * fs          # data only, before the first window
    assert (n0 + n) * 64 / 485 < spec.win_start_pos[0] - 8
    phi = ce.phase_grid(fs, n0, n, sym, spec)
    assert np.var(phi) == pytest.approx(BETA_F ** 2, rel=0.05)
    v = ce.var_grid(fs, n0, n, np.ones(spec.n_sym), spec)
    assert np.mean(v) == pytest.approx(BETA_F ** 2, rel=0.01)
    tau = np.arange(n0, n0 + 50, dtype=np.int64) * 64 / 485
    _, v_at = ce.phase_at(tau, sym, spec, var=np.ones(spec.n_sym))
    assert np.allclose(v_at, v[:50], atol=1e-12)


def test_baseband_clock_offset_path_matches_grid():
    """time_scale != 1 evaluates in closed form; at 1 + 1e-15 it is the same signal.

    Except exactly at whole symbols: the pulse is cut at |tau| <= 8, a step of
    p(8) = 0.006, so a tau that lands on an integer on the grid and 1e-12 off
    it in floating point sees a pulse 8 symbols away on one side of its cut
    and not on the other. Those samples are left out.
    """
    spec = frame.SHORT
    sym, _ = _frame(spec)
    k = morse.keying_units(CALL)
    fs = 250
    start = 150.0
    a = ce.baseband(sym, spec, fs, start, 40 * fs, keying=k)
    b = ce.baseband(sym, spec, fs, start, 40 * fs, keying=k, time_scale=1 + 1e-15)
    tau = (start + np.arange(40 * fs) / fs) / T_SYM
    off_edge = np.abs(tau - np.round(tau)) > 1e-9
    assert np.sum(~off_edge) <= 21              # every 64 symbols at 250 Hz
    assert np.max(np.abs(a - b)[off_edge]) < 1e-9


# --- M2: power split -------------------------------------------------------------------

def test_m2_power_split():
    """M2: carrier |E e^{j phi}|^2 = 0.527 +- 0.005 and linear part 0.337 +- 0.005
    on 50,600 Gaussian latents.

    The carrier is the time average of cos(phi) over the data (its sine part
    averages to zero by symmetry, and cos has the smaller variance). The
    linear part is what the receiver regresses: the matched-filter gain on
    the data symbols, squared (its power is gain^2 E x^2 with a unit-energy
    pulse).
    """
    spec = frame.FULL
    sym, _ = _frame(spec)
    lay = frame.layout(spec)
    fs = 250
    n0, n = ce._loopback_span(spec, fs)
    phi = ce.phase_grid(fs, n0, n, sym, spec)
    tau = np.arange(n0, n0 + n, dtype=np.int64) * 64 / 485
    sel = (tau > lay.pos[lay.data[0]] + 8) & (tau < lay.pos[lay.data[-1]] - 8)
    for a_w, b_w in lay.win_pos:
        sel &= ~((tau > a_w - 9) & (tau < b_w + 9))
    carrier = np.mean(np.cos(phi[sel])) ** 2
    assert carrier == pytest.approx(CARRIER_FRAC, abs=0.005)
    st = ce.loopback_stats(sym, spec)
    assert st["carrier"] == pytest.approx(CARRIER_FRAC, abs=0.005)
    x = sym[lay.data]
    linear = st["gain"] ** 2 * np.mean(x ** 2)
    assert linear == pytest.approx(USEFUL_FRAC, abs=0.005)


# --- M3: genie loopback ------------------------------------------------------------------

# Pinned from this implementation (FULL frame, Gaussian latents, seed 0):
# gain 0.5788, per-pass distortion 13.16 dB, D_PASS = 0.0483 in latent units.
PINNED_GAIN = 0.5788
PINNED_DIST_DB = 13.16


def test_m3_genie_loopback_gaussian():
    """M3: gain 0.581 +- 0.003; per-pass distortion 13.2 +- 0.4 dB (pinned 13.16)."""
    spec = frame.FULL
    sym, _ = _frame(spec)
    st = ce.loopback_stats(sym, spec)
    assert st["gain"] == pytest.approx(K_LIN, abs=0.003)
    assert st["dist_db"] == pytest.approx(13.2, abs=0.4)
    assert st["gain"] == pytest.approx(PINNED_GAIN, abs=5e-4)
    assert st["dist_db"] == pytest.approx(PINNED_DIST_DB, abs=0.05)
    assert st["d_pass"] == pytest.approx(10 ** (-st["dist_db"] / 10), rel=1e-9)
    # Known +-1 symbols: the MF sees the linear part plus a residual carrier
    # tilt; the receiver's template handles both, so just sanity-check it.
    assert 0.6 < st["known_gain"] < 0.75
    # The template's self-noise model (KAPPA_SELF) is within 10% of the truth.
    assert st["kappa_self"] == pytest.approx(1.0, abs=0.1)
    assert st["classes"]["known"]["res_var"] < 0.2 * st["classes"]["data"]["res_var"]


def test_m3_latent_domain_distortion_matches_symbol_domain():
    """The precoder is orthonormal: unprecoding Im(m)/gain gives the same SNR."""
    spec = frame.SHORT
    sym, a = _frame(spec)
    lay = frame.layout(spec)
    m = ce.loopback_mf(sym, spec)
    st = ce.loopback_stats(sym, spec)
    a_hat = precoder.unprecode(m[lay.data].imag / st["gain"], Q_TEST)
    snr = latent_snr_db(a_hat, a)
    assert snr == pytest.approx(st["dist_db"], abs=0.3)


@pytest.mark.codec
def test_m3_genie_loopback_v5_latents():
    """M3 (codec): real v5 latents of wonder_wheel.jpg, group 0: 0.581 +- 0.003, 13.2 +- 0.4."""
    from sstvae.images import load_image
    from sstvae.qrss import picture

    codec = load_codec("fp16")
    img = REPO_ROOT / "wonder_wheel.jpg"
    if not img.exists():
        pytest.skip("wonder_wheel.jpg not in the repo")
    sp = picture.StoredPicture.from_latents(codec.encode(load_image(img)), 0xD1D8, 0)
    a = picture.air_values(sp.segs[0], 0)
    spec = frame.FULL
    sym = frame.assemble(spec, _header_bits(), precoder.precode(a, Q_TEST))
    st = ce.loopback_stats(sym, spec)
    assert st["gain"] == pytest.approx(K_LIN, abs=0.003)
    assert st["dist_db"] == pytest.approx(13.2, abs=0.4)


def _coco_latents(n_pics):
    import io

    import pyarrow.parquet as pq
    from PIL import Image

    from sstvae.images import fit_image, image_to_array
    from sstvae.qrss import picture

    path = require_coco()
    codec = load_codec("fp16")
    rows = pq.ParquetFile(path).read_row_group(0).slice(0, n_pics).to_pylist()
    out = []
    for r in rows:
        img = image_to_array(fit_image(Image.open(io.BytesIO(r["image"]["bytes"]))))
        sp = picture.StoredPicture.from_latents(codec.encode(img), 0xD1D8, 0)
        out.append(picture.air_values(sp.segs[0], 0))
    return out


def _fixed_part_db(a, passes, precode=True, spec=frame.FULL):
    """(SINR of the average of `passes` genie passes, extrapolated fixed part), dB.

    The first is the spec's number (`sims/ce/det_test.py`: "F = average of 32
    noiseless passes estimates the deterministic response"), which still holds
    1/32 of the per-pass random part. The second fits 1/S_P = D + R/P over
    P = 8, 16, 32 and reports 1/D, the part that truly never averages.
    """
    lay = frame.layout(spec)
    acc = np.zeros(len(a))
    snr = {}
    for p in range(1, passes + 1):
        q = Q_TEST + 1000 * p
        if precode:
            x = precoder.precode(a, q)
        else:
            x = sequences.scrambler(q, len(a)) * a
        sym = frame.assemble(spec, _header_bits(p), x)
        y = ce.loopback_mf(sym, spec)[lay.data].imag / K_LIN
        acc += precoder.unprecode(y, q) if precode else sequences.scrambler(q, len(a)) * y
        if p in (8, 16, 32):
            est = acc / p
            c = np.dot(est, a) / np.dot(a, a)
            snr[p] = c * c * np.mean(a * a) / np.mean((est - c * a) ** 2)
    P = np.array(sorted(snr))
    A = np.vstack([np.ones(len(P)), 1 / P]).T
    D, _ = np.linalg.lstsq(A, np.array([1 / snr[k] for k in P]), rcond=None)[0]
    return 10 * np.log10(snr[passes]), 10 * np.log10(1 / max(D, 1e-9))


@pytest.mark.slow
@pytest.mark.codec
def test_m4_never_averaging_part_with_and_without_precoder():
    """M4: 32 genie passes with independent q on 5 COCO pictures: the part that
    never averages is 23.1 +- 1 dB with the precoder, 13.4 +- 0.6 dB without."""
    lats = _coco_latents(5)
    with_pc = np.mean([_fixed_part_db(a, 32, True) for a in lats], axis=0)
    without = np.mean([_fixed_part_db(a, 32, False) for a in lats], axis=0)
    print(f"M4: 32-pass average {with_pc[0]:.2f} dB with the precoder (fixed part "
          f"{with_pc[1]:.2f} dB extrapolated), {without[0]:.2f} dB without "
          f"({without[1]:.2f} dB)")
    assert with_pc[0] == pytest.approx(23.1, abs=1.0)
    assert without[0] == pytest.approx(13.4, abs=0.6)


# --- M5: spectrum -------------------------------------------------------------------------

def _psd(z, fs):
    f, S = welch(z, fs=fs, nperseg=fs * 4, return_onesided=False, detrend=False)
    o = np.argsort(f)
    return f[o], S[o]


def test_m5_spectrum():
    """M5: 99% / 99.9% widths, sideband density and leakage into a CE neighbour
    50 Hz away, by Welch on 200 s of data at the FE rate (the spec's
    `pm_spectrum_synth` measures, on the data section of a precoded frame)."""
    spec = frame.FULL
    sym, _ = _frame(spec, seed=3)
    fs = 4000
    z = ce.baseband(sym, spec, fs, 110.0, 200 * fs)
    f, S = _psd(z, fs)
    df = f[1] - f[0]
    tot = S.sum() * df

    def width(q):
        for B in np.arange(1, 400, 0.5):
            if S[np.abs(f) <= B / 2].sum() * df / tot >= q:
                return B

    w99, w999 = width(0.99), width(0.999)
    f2, S2 = _psd(z - z.mean(), fs)
    ref = np.median(S2[np.abs(f2) < 15])
    prof = {off: 10 * np.log10(np.mean(S2[(np.abs(f2) > off - 1) & (np.abs(f2) < off + 1)]) / ref)
            for off in (19, 25, 30, 40, 50, 60, 80)}
    band = (f2 > 50 - 19) & (f2 < 50 + 19)
    leak = 10 * np.log10(S2[band].sum() * df / tot)
    print(f"M5: 99% {w99} Hz, 99.9% {w999} Hz, profile "
          + " ".join(f"{k}:{v:.1f}" for k, v in prof.items()) + f", leak {leak:.1f} dB")
    assert w99 == pytest.approx(51.5, abs=1.5)
    assert w999 == pytest.approx(72.5, abs=3.0)
    want = {19: -8, 25: -11, 30: -15, 40: -24, 50: -34, 60: -44}
    for off, db in want.items():
        assert prof[off] == pytest.approx(db, abs=2.0), off
    assert leak == pytest.approx(-28, abs=1.5)
    # 80 Hz: see test_m5_far_skirt_at_80_hz. Pinned at the measured -60.4 dB.
    assert prof[80] == pytest.approx(-60.4, abs=2.0)


@pytest.mark.xfail(strict=True, reason=(
    "design M5 asks -65 +- 4 dB at 80 Hz, taken from sims/ce/pm_spectrum_synth.py, "
    "which cuts the pulse at +-16 symbols; the normative pulse (design D2, spec 2.5) "
    "is cut at +-8, whose step of p(8) = 0.006 lifts the 80 Hz skirt to -60.4 dB"))
def test_m5_far_skirt_at_80_hz():
    spec = frame.FULL
    sym, _ = _frame(spec, seed=3)
    fs = 4000
    z = ce.baseband(sym, spec, fs, 110.0, 200 * fs)
    f2, S2 = _psd(z - z.mean(), fs)
    ref = np.median(S2[np.abs(f2) < 15])
    p80 = 10 * np.log10(np.mean(S2[(np.abs(f2) > 79) & (np.abs(f2) < 81)]) / ref)
    assert p80 == pytest.approx(-65, abs=4.0)


# --- M8: lead-in ---------------------------------------------------------------------------

@pytest.mark.parametrize("lead", [0.0, 3.0, 10.0])
def test_m8_lead_in(lead):
    """M8: phi = 0 before t0 - 8T; the lead-in runs phase-continuously into the
    preamble; |s| = 1 across the keyed span of an FSK frame."""
    spec = frame.SHORT
    sym, _ = _frame(spec)
    k = morse.keying_units(CALL)
    fs = 8000
    start = -12.0
    n = int((12 + spec.keyed_end_pos * T_SYM + 1) * fs)
    s = ce.baseband(sym, spec, fs, start, n, keying=k, lead_in_s=lead)
    t = start + np.arange(n) / fs
    tau = t / T_SYM
    lo, hi = ce.keyed_span_pos(spec, lead)
    assert lo == pytest.approx(-8 - lead / T_SYM)
    keyed = (tau >= lo) & (tau < hi)
    assert np.max(np.abs(np.abs(s[keyed]) - 1)) < 1e-12
    pre = keyed & (tau < -8)
    if lead:
        # Lead-in: plain carrier, exactly phase 0, for lead seconds.
        assert np.sum(pre) == pytest.approx(lead * fs, abs=2)
        assert np.max(np.abs(s[pre] - 1)) < 1e-15
    phi = np.angle(s[keyed])
    # Continuity: the largest sample-to-sample step is the peak frequency's.
    assert np.max(np.abs(np.angle(s[keyed][1:] / s[keyed][:-1]))) < 2 * np.pi * 60 / fs
    assert np.max(np.abs(phi[:fs // 10])) < 1e-12 or not lead
    # Ramp outside the keyed span: amplitude only, at phase 0, then silence.
    ramp = (tau >= lo - 0.05 / T_SYM) & (tau < lo)
    assert np.all(np.abs(np.angle(s[ramp][np.abs(s[ramp]) > 0])) < 1e-15)
    assert np.all(np.diff(np.abs(s[ramp])) >= -1e-15)
    assert np.all(s[tau < lo - 0.05 / T_SYM - 1e-9] == 0)
    assert np.all(s[tau >= hi + 0.05 / T_SYM + 1e-9] == 0)


def test_lead_in_longer_than_ten_seconds_is_refused():
    with pytest.raises(ValueError):
        ce.keyed_span_pos(frame.SHORT, 10.5)
    with pytest.raises(ValueError):
        ce.keyed_span_pos(frame.SHORT, -1)


def test_frame_end_keying():
    """The FULL and SHORT frames end with their last window; TINY 8 symbols after
    its last pulse (design 2.2)."""
    assert frame.FULL.keyed_end_pos == frame.FULL.n_pos - 0.5
    assert frame.FULL.keyed_end_pos * T_SYM == pytest.approx(1782.678125 - T_SYM / 2)
    tiny = frame.TINY
    assert tiny.keyed_end_pos == tiny.pos_of(tiny.n_sym - 1) + 8


# --- M9: callsign windows -----------------------------------------------------------------

def test_m9_fsk_window_phase():
    """M9: theta_cw is an integer at the window end (to 1e-12 turn), continuous
    across the window, and the frequency is one cycle per unit below the
    carrier while key-down."""
    spec = frame.SHORT
    k = morse.keying_units(CALL)
    for w, p_w in enumerate(spec.win_start_pos):
        end = p_w - 0.5 + 384
        th, a = ce.cw_phase_turns(np.array([p_w - 0.5, end, end + 50]), spec, k)
        assert th[0] == -w * int(k.sum())
        assert abs(th[1] - (-(w + 1) * int(k.sum()))) < 1e-12
        assert th[2] == -(w + 1) * int(k.sum())
        assert np.all(a == 1)
    # Continuous, with frequency in [-16.5, 0] Hz throughout.
    p_w = spec.win_start_pos[0]
    tau = np.arange(p_w - 10, p_w + 400, 1 / 64)
    th, _ = ce.cw_phase_turns(tau, spec, k)
    f_hz = np.diff(th) / (np.diff(tau) * T_SYM)
    assert f_hz.min() >= ce.FSK_SHIFT_HZ - 1e-9 and f_hz.max() <= 1e-9
    # Key-down units sit at the full shift at their centre; key-up at 0.
    x = (tau[:-1] - (p_w - 0.5)) / CW_UNIT
    u = np.floor(x).astype(int)
    centre = (np.abs(x - u - 0.5) < 0.02) & (u >= 0) & (u < CW_UNITS)
    want = np.where(k[u[centre]] == 1, ce.FSK_SHIFT_HZ, 0.0)
    # Isolated units (a dot between gaps) reach the shift exactly at their centre.
    assert np.max(np.abs(f_hz[centre] - want)) < 0.6
    assert ce.FSK_SHIFT_HZ == pytest.approx(-16.495, abs=1e-3)


def test_m9_data_tails_die_inside_the_plain_units():
    """M9: phi in units 4..187 of a window is below 1e-3 rad (exactly 0 with the
    pulse cut at 8 symbols), so the Morse sees a clean carrier."""
    spec = frame.SHORT
    sym, _ = _frame(spec)
    for p_w in spec.win_start_pos:
        tau = np.linspace(p_w - 0.5 + 4 * CW_UNIT, p_w - 0.5 + 188 * CW_UNIT, 5000)
        phi, _ = ce.phase_at(tau, sym, spec)
        assert np.max(np.abs(phi)) < 1e-3
        # ... and the data does reach into units 0..3 (the tails are real).
        tau = np.linspace(p_w - 0.5, p_w - 0.5 + 3 * CW_UNIT, 500)
        assert np.max(np.abs(ce.phase_at(tau, sym, spec)[0])) > 1e-3


def test_m9_fsk_occupied_band():
    """M9: FSK 99% of the window's power within -20.7 to +4.6 Hz of the carrier
    (+-1 Hz), measured as `sims/cwid/cwid_final.py` does: one isolated window,
    averaged over its eight calls, 0.5% and 99.5% points of the power."""
    spec = frame.SHORT
    calls = ["K1ABC", "KB1ABC", "VE3ABC", "DL1ABC", "K1ABC/P", "JH0QQQ/P", "W1AW", "G4ABC"]
    fs = 64 / (CW_UNIT * T_SYM)                  # 64 samples per unit
    p_w = spec.win_start_pos[0]
    tau = p_w - 0.5 + (np.arange(CW_UNITS * 64) + 0.5) / 64 * CW_UNIT
    nfft = 1 << 18
    P = 0
    for c in calls:
        th, _ = ce.cw_phase_turns(tau, spec, morse.keying_units(c))
        X = np.fft.fftshift(np.fft.fft(np.exp(2j * np.pi * th), nfft))
        p = np.abs(X) ** 2
        P = P + p / p.sum()
    f = np.fft.fftshift(np.fft.fftfreq(nfft, 1 / fs))
    cdf = np.cumsum(P / P.sum())
    lo, hi = f[np.searchsorted(cdf, 0.005)], f[np.searchsorted(cdf, 0.995)]
    print(f"M9: FSK 99% band {lo:.2f} to {hi:.2f} Hz")
    assert lo == pytest.approx(-20.7, abs=1.0)
    assert hi == pytest.approx(4.6, abs=1.0)


def test_ook_window_keying():
    """On-off keying: on in units 0..4 and 184..191, off in 5..7, Morse in 8..183,
    no phase (spec 2.7, design D14)."""
    spec = frame.SHORT
    k = morse.keying_units(CALL)
    p_w = spec.win_start_pos[1]
    u = np.arange(CW_UNITS)
    tau = p_w - 0.5 + (u + 0.5) * CW_UNIT
    th, a = ce.cw_phase_turns(tau, spec, k, ook=True)
    assert np.all(th == 0)
    want = np.ones(CW_UNITS)
    want[5:184] = k[5:184]
    assert np.array_equal(a, want)
    assert np.all(a[5:8] == 0) and np.all(a[:5] == 1) and np.all(a[184:] == 1)
    # In the baseband: zero where keyed off, unit magnitude elsewhere.
    sym, _ = _frame(spec)
    fs = 8000
    start = (p_w - 20) * T_SYM
    n = int(420 * T_SYM * fs)
    s = ce.baseband(sym, spec, fs, start, n, keying=k, ook=True)
    t_tau = (start + np.arange(n) / fs) / T_SYM
    x = (t_tau - (p_w - 0.5)) / CW_UNIT
    uu = np.floor(x).astype(int)
    inwin = (uu >= 0) & (uu < CW_UNITS)
    assert np.allclose(np.abs(s[inwin]), want[uu[inwin]], atol=1e-12)


def test_window_keying_shapes():
    spec = frame.SHORT
    k = morse.keying_units(CALL)
    assert ce.window_keying(k, spec).shape == (2, CW_UNITS)
    assert ce.window_keying([k, k], spec).shape == (2, CW_UNITS)
    assert ce.window_keying(None, spec) is None
    with pytest.raises(ValueError):
        ce.window_keying(k[:100], spec)
    with pytest.raises(ValueError):
        ce.window_keying([k, k, k], spec)


# --- mean template, audio ---------------------------------------------------------------------

def test_mean_template_on_known_stretch_is_the_signal():
    """With every symbol known (nu = 0) the mean template is the signal itself."""
    spec = frame.SHORT
    sym, _ = _frame(spec)
    k = morse.keying_units(CALL)
    fs = 250
    tau = (np.arange(-500, 6000) * 64) / 485
    s = ce.baseband(sym, spec, fs, -2.0, 6500, keying=k, ramp_s=0.0)
    tmpl = ce.mean_template(tau, sym, np.zeros(spec.n_sym), spec, keying=k)
    assert np.max(np.abs(s - tmpl)) < 1e-12
    # Unknown data: the template shrinks by exp(-v/2); on the data, ~A0.
    lay = frame.layout(spec)
    mu = np.zeros(spec.n_sym)
    nu = np.ones(spec.n_sym)
    mu[lay.known_idx] = lay.known_val
    nu[lay.known_idx] = 0
    p0 = lay.pos[lay.data[100]]
    tau = np.linspace(p0, p0 + 300, 3000)
    c = ce.mean_template(tau, mu, nu, spec)
    assert np.mean(np.abs(c)) == pytest.approx(A0, rel=0.1)


def test_to_audio_power_and_exact_mixing():
    z = np.exp(2j * np.pi * 0.01 * np.arange(8000))
    y = ce.to_audio(z, 8000, 1500, amplitude=np.sqrt(2))
    assert np.mean(y ** 2) == pytest.approx(1.0, rel=1e-3)
    # Chunked mixing equals one-shot mixing (integer-turn carrier).
    y1 = ce.to_audio(z[:3000], 8000, 3001, 2, n0=0)
    y2 = ce.to_audio(z[3000:], 8000, 3001, 2, n0=3000)
    assert np.array_equal(np.concatenate([y1, y2]), ce.to_audio(z, 8000, 3001, 2))
    # A carrier at 1500 Hz lands at 1500 Hz.
    Y = np.abs(np.fft.rfft(ce.to_audio(np.ones(8000), 8000, 1500)))
    assert np.argmax(Y) == 1500
    # Lower-rate baseband is resampled up.
    assert len(ce.to_audio(np.ones(250), 250, 1500)) == 8000

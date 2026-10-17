"""Waveform L, the transmitter (spec 2, 2.2, 2.7, 2.8; docs/qrss/waveform-l.md).

The frame's counts and times against CE's, the pinned L sequences, the
symbol mapping, the clipper's numbers against the spec's table (PAPR,
SINR, the data's gain against the references, the pilot's share), the
callsign windows and lead-in, and the 8 kHz audio through a genie
receiver. All fast: the frames here take well under a second each.
"""

import dataclasses
import hashlib
import subprocess
import sys
from fractions import Fraction

import numpy as np
import pytest

from qrss_fakes import make_header, synthetic_picture
from qrss_helpers import Q_TEST, REPO_ROOT
from sstvae.qrss import constants as C
from sstvae.qrss import frame, lframe, lwave, morse, picture, sequences, tx

CALL = "AG7EW"


def _gauss_frame(spec, seed=1):
    rng = np.random.default_rng(seed)
    bits = rng.integers(0, 2, C.N_HDR_BITS).astype(np.uint8) if spec.has_header else None
    x = rng.normal(size=spec.n_data)
    return lframe.assemble(spec, bits, x), bits, x


# --- the frame ----------------------------------------------------------------------


def test_full_frame_counts_match_the_spec():
    f = lframe.L_FULL
    lay = lframe.layout(f)
    assert (f.n_sym, f.n_pos, f.n_ref) == (28637, 29405, 1770)
    assert (len(lay.pre), len(lay.hdr_ref), len(lay.hdr_sym), len(lay.data_ref),
            len(lay.data)) == (330, 83, 1237, 1687, 25300)
    assert f.cw_after == (6944, 14175, 21406, 28637)
    assert f.keyed_end_pos == f.n_pos - 0.5
    # The same frame in time as CE's: same length, windows within half a CE symbol.
    assert f.duration_s == frame.FULL.duration_s
    for (a, b), (c, d) in zip(f.window_spans_s(), frame.FULL.window_spans_s()):
        assert abs(a - c) <= C.T_SYM / 2 + 1e-9 and abs(b - d) <= C.T_SYM / 2 + 1e-9


def test_the_data_section_does_not_start_on_a_reference():
    """1,320 header symbols are 82.5 reference periods: the first data
    reference is at j = 1328 (counting from the first header symbol)."""
    lay = lframe.layout(lframe.L_FULL)
    j = lay.ref - lframe.L_FULL.n_pre
    assert np.all(j % 16 == 0) and j[82] == 1312 and j[83] == 1328
    assert lframe.L_FULL.n_pre + lframe.L_FULL.n_hdr == 1650
    assert lay.data[0] == 1650


@pytest.mark.parametrize("spec", list(lframe.PRESETS.values()), ids=lambda s: s.name)
def test_every_preset_lays_out(spec):
    lay = lframe.layout(spec)
    assert len(lay.data) == spec.n_data // 2
    assert spec.n_sym == spec.n_pre + spec.n_hdr + spec.n_data_sym
    assert sorted(np.concatenate([lay.known_idx, lay.hdr_sym, lay.data]).tolist()) == \
        list(range(spec.n_sym))


@pytest.mark.parametrize("kwargs", [dict(n_pre=660), dict(n_hdr=2640), dict(n_data=101),
                                    dict(cw_after=(10,)), dict(hdr_rho=1.0)])
def test_invalid_l_frames_raise(kwargs):
    with pytest.raises(ValueError):
        lframe.LFrameSpec(**kwargs)


def test_the_waveform_is_a_property_of_the_frame():
    assert frame.FULL.waveform == C.WAVEFORM_CE and frame.FULL.t_sym == C.T_SYM
    assert lframe.L_FULL.waveform == C.WAVEFORM_L and lframe.L_FULL.t_sym == C.L_T_SYM
    assert C.L_PILOT_HZ == Fraction(12207, 970)
    assert C.L_PILOT_HZ - C.L_EDGE_HZ == Fraction(31, 10)


# --- sequences ----------------------------------------------------------------------

PINNED_L = {
    "preamble_bits": "51917ffc62ac971a2051f5a7385eb750296ba388856524ed765d32936582916a",
    "references_1770_bits": "1d49063667ddbaffe4f0f5037a73ca07852f09b26420da893ec2d98a7b3aca83",
}


def test_l_sequences_are_pinned():
    pre = sequences.sha_bits(b"QRSSTVAE L preamble", 660)
    ref = sequences.sha_bits(b"QRSSTVAE L reference", 2 * 1770)
    assert hashlib.sha256(pre.tobytes()).hexdigest() == PINNED_L["preamble_bits"]
    assert hashlib.sha256(ref.tobytes()).hexdigest() == PINNED_L["references_1770_bits"]


def test_qpsk_takes_bit_pairs_i_then_q():
    """Spec 2.3: the first bit of a pair to I, the second to Q, 0 -> +1, / sqrt(2)."""
    r = 1 / np.sqrt(2)
    assert np.allclose(sequences.qpsk([0, 0, 0, 1, 1, 0, 1, 1]),
                       [r + 1j * r, r - 1j * r, -r + 1j * r, -r - 1j * r])
    bits = sequences.sha_bits(b"QRSSTVAE L preamble", 8)
    assert np.allclose(sequences.preamble_l()[:4], sequences.qpsk(bits))
    ref = sequences.references_l(1770)
    assert np.array_equal(sequences.references_l(83), ref[:83])     # a prefix, as CE's
    assert len(sequences.preamble_l()) == 330 and np.allclose(abs(ref), 1)


# --- symbols ------------------------------------------------------------------------


def test_assemble_places_header_bits_and_latent_pairs():
    spec = dataclasses.replace(lframe.L_SHORT, hdr_rho=0.0)
    sym, bits, x = _gauss_frame(spec)
    lay = lframe.layout(spec)
    assert np.allclose(sym[lay.known_idx], lay.known_val)
    assert np.allclose(sym[lay.hdr_sym], sequences.qpsk(bits))
    assert np.allclose(sym[lay.data[:3]], (x[0:6:2] + 1j * x[1:6:2]) / np.sqrt(2))
    h, d = lframe.disassemble(sym, spec)
    assert np.allclose(h, 1 - 2.0 * bits) and np.allclose(d, x)


def test_the_spread_copy_scales_the_latents_to_make_room():
    """With rho > 0 latent i is (x_i + sqrt(rho) h_i)/sqrt(1 + rho), h being CE's."""
    spec = lframe.L_SHORT
    assert spec.hdr_rho == C.SPREAD_RHO
    sym, bits, x = _gauss_frame(spec)
    _, d = lframe.disassemble(sym, spec)
    h = frame.spread_signs(spec, bits)
    assert np.allclose(d, (x + np.sqrt(C.SPREAD_RHO) * h) / np.sqrt(1 + C.SPREAD_RHO))
    assert lframe.L_TINY.hdr_rho == 0.0


# --- the clipper ----------------------------------------------------------------------


@pytest.fixture(scope="module")
def short_stats():
    sym, _, _ = _gauss_frame(lframe.L_SHORT)
    return lwave.loopback_stats(sym, lframe.L_SHORT, keying=morse.keying_units(CALL))


def test_the_clipper_matches_the_spec_table(short_stats):
    """Spec 2.2 at 0 dB, three overshoot passes: PAPR 3.2 dB, SINR 12.3 dB,
    data 0.88 of the references, pilot 13.8% of the power after clipping."""
    st = short_stats
    assert 2.9 < st["papr_db"] < 3.5
    assert 11.6 < st["sinr_db"] < 13.0
    assert 0.86 < st["data_vs_ref"] < 0.90
    assert 0.13 < st["pilot_frac"] < 0.15
    assert abs(st["avg_power"] - lwave.L_AVG_POWER) < 0.02
    lo, hi = st["width_hz"]
    assert -22.5 < lo < -20.5 and -0.2 < hi < 0.5         # 99% in 21.5 Hz, ending at the pilot


def test_without_clipping_the_genie_is_exact():
    """A threshold nothing reaches leaves the linear signal, and the matched
    filter returns the symbols: the pulse's cut at +-8 symbols leaves
    intersymbol interference 45 dB down, and the pilot leaks in 74 dB below
    itself."""
    spec = lframe.L_TINY
    sym, _, _ = _gauss_frame(spec)
    lay = lframe.layout(spec)
    fs = lwave.CLIP_FS
    n0, n = -10 * fs, round(spec.duration_s * fs) + 20 * fs
    z0 = lwave.composite(sym, spec, fs, n0, n)
    z = lwave.clip(z0, fs, threshold=1e6)
    assert np.array_equal(z, z0)
    bb = lwave.Baseband(z=z, n0=n0, fs=fs)
    m = lwave.matched_filter(lwave.to_data_centre(bb), fs, n0, lay.pos)
    err = abs(m - sym) ** 2
    assert 10 * np.log10(np.mean(err)) < -43 and np.max(err) < 1e-3


def test_the_clipper_filters_only_its_correction():
    """Below the threshold nothing changes; above it the change is in band."""
    fs = lwave.CLIP_FS
    t = np.arange(40 * fs) / fs
    z = np.where((t > 10) & (t < 30), 3.0, 0.5).astype(np.complex128)
    out = lwave.clip(z, fs)
    assert np.allclose(out[t < 5], 0.5, atol=1e-3)            # far from the clipped stretch
    corr = np.fft.fft(out - z)
    f = np.fft.fftfreq(len(z), 1 / fs)
    band = lwave.band_mask(f) > 0
    assert np.sum(abs(corr[~band]) ** 2) < 1e-10 * np.sum(abs(corr) ** 2)   # -100 dB


# --- callsign windows and lead-in ------------------------------------------------------


@pytest.fixture(scope="module")
def short_bb():
    sym, _, _ = _gauss_frame(lframe.L_SHORT)
    return lwave.frame_baseband(sym, lframe.L_SHORT, keying=morse.keying_units(CALL),
                                lead_in_s=5.0)


def test_windows_send_the_pilot_at_average_power_by_fsk(short_bb):
    bb, spec = short_bb, lframe.L_SHORT
    tau = lwave.tau_grid(bb.fs, bb.n0, len(bb.z))
    data = lwave._data_span_mask(spec, tau)
    assert abs(np.mean(abs(bb.z[data]) ** 2) - 1.0) < 0.03     # unit average power
    rs = 1 / C.L_T_SYM
    for a, b in lframe.layout(spec).win_pos:
        inner = (tau > a + 9) & (tau < b - 9)
        z = bb.z[inner]
        assert np.all(abs(abs(z) - 1.0) < 0.03)
        f = np.diff(np.unwrap(np.angle(z))) * bb.fs / (2 * np.pi)
        assert abs(f.min() + rs) < 0.3 and abs(f.max()) < 0.3   # key-down one Rs below
        # The pilot keeps its phase across the window (spec 2.7).
        if b == spec.n_pos - 1:
            continue                                            # the last one ends the frame
        before = bb.z[(tau > a - 300) & (tau < a - 12)].mean()
        after = bb.z[(tau > b + 12) & (tau < b + 300)].mean()
        assert abs(np.angle(after / before)) < 0.15 and abs(abs(after / before) - 1) < 0.1


def test_lead_in_is_the_pilot_at_average_power(short_bb):
    bb = short_bb
    tau = lwave.tau_grid(bb.fs, bb.n0, len(bb.z))
    start = -8 - 5.0 / C.L_T_SYM
    lead = (tau > start + 2) & (tau < -9)
    assert np.all(abs(abs(bb.z[lead]) - 1.0) < 0.01)
    assert np.all(abs(np.angle(bb.z[lead])) < 0.01)            # the pilot's own phase
    assert not bb.z[tau < start - (lwave.KEY_RAMP_S / C.L_T_SYM) - 1e-9].any()


def test_window_keying_must_match_the_frame():
    with pytest.raises(ValueError):
        lwave.window_keying(np.ones(384, np.uint8), lframe.L_SHORT)


# --- audio ----------------------------------------------------------------------------------


def test_audio_through_a_genie_receiver_matches_the_loopback():
    """transmit_audio's 8 kHz signal, mixed down by the pilot and decimated,
    gives the clipper's own symbols back: the interpolation adds nothing."""
    from scipy.signal import resample_poly
    spec = lframe.L_SHORT
    sp = synthetic_picture(0)
    h = make_header(sp, callsign=CALL)
    f_pilot = Fraction(30003, 20)
    audio = tx.transmit_audio(sp, Q_TEST, f_pilot, spec=spec, header=h, lead_in_s=2.0)
    assert len(audio) == tx.slot_samples(spec)
    n = np.arange(len(audio), dtype=np.int64)
    period = f_pilot.denominator * C.FS
    turns = ((n % period) * (f_pilot.numerator % period) % period) / period
    bb8 = 2 * audio * np.exp(-2j * np.pi * turns) / 0.5
    z = resample_poly(bb8, 1, 32, window=("kaiser", 12.0))
    bb = lwave.Baseband(z=z, n0=-12 * 250, fs=250)
    lay = lframe.layout(spec)
    m = lwave.matched_filter(lwave.to_data_centre(bb), 250, bb.n0, lay.pos)
    sym = tx.slot_symbols(sp, Q_TEST, spec, header=h)
    ref = lwave.frame_baseband(sym, spec, keying=tx.slot_keying(sp, h), lead_in_s=2.0)
    m_ref = lwave.matched_filter(lwave.to_data_centre(ref), 250, ref.n0, lay.pos)
    assert np.max(abs(m - m_ref)) < 1e-3
    # Silence before keying (the lead-in plus the ramp) and after the frame.
    t = n / C.FS - 12.0
    keyed = (-8 * C.L_T_SYM - 2.0 - 0.05, spec.keyed_end_pos * C.L_T_SYM + 0.05)
    quiet = (t < keyed[0] - 0.1) | (t > keyed[1] + 0.1)
    assert np.max(abs(audio[quiet])) < 1e-4 * np.max(abs(audio))


def test_l_refuses_on_off_keying_and_a_pilot_too_low():
    sp = synthetic_picture(0)
    h = make_header(sp, callsign=CALL)
    with pytest.raises(ValueError, match="FSK"):
        tx.transmit_audio(sp, Q_TEST, 1500, spec=lframe.L_SHORT, header=h, ook=True)
    with pytest.raises(ValueError, match="below"):
        tx.transmit_audio(sp, Q_TEST, 310, spec=lframe.L_SHORT, header=h)
    with pytest.raises(ValueError, match="callsign"):
        tx.transmit_audio(sp, Q_TEST, 1500, spec=lframe.L_SHORT)


def test_qrss_transmit_writes_an_l_pass(tmp_path):
    qrsp = tmp_path / "pic.qrsp"
    picture.save_qrsp(qrsp, synthetic_picture(0))
    wav = tmp_path / "l.wav"
    r = subprocess.run([sys.executable, "qrss_transmit.py", str(qrsp), str(wav), "--slot",
                        "2025-10-01T00:00Z", "--callsign", CALL, "--waveform", "L",
                        "--frame", "tiny", "--float"], cwd=REPO_ROOT, capture_output=True,
                       text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    assert "waveform L" in r.stdout and wav.stat().st_size > 0
    bad = subprocess.run([sys.executable, "qrss_transmit.py", str(qrsp), str(wav), "--slot",
                          "2025-10-01T00:00Z", "--callsign", CALL, "--waveform", "L",
                          "--ook"], cwd=REPO_ROOT, capture_output=True, text=True,
                         timeout=120)
    assert bad.returncode != 0 and "FSK" in bad.stderr

"""Beacon file, transmit pipeline, Si5351 model and the WP3 CLIs
(design 10.2: M6, M7, M10; section 3).

All fast except the encoder CLI, which needs the codec.
"""

import subprocess
import sys
from fractions import Fraction

import numpy as np
import pytest
from scipy.signal import resample_poly

from qrss_helpers import Q_TEST, REPO_ROOT, resolve_model, synthetic_full_latents, \
    unit_rms_latents
from sstvae import wavio
from sstvae.qrss import beaconfile, ce, frame, morse, picture, precoder, si5351, tx
from sstvae.qrss.constants import K_LIN, T_SYM
from sstvae.qrss.header import HeaderFields

CALL = "K1ABC/P"


@pytest.fixture(scope="module")
def sp():
    return picture.StoredPicture.from_latents(synthetic_full_latents(3), 0xD1D8, 1)


def _header(sp, segment=0, call=CALL, grid="FN42"):
    return HeaderFields(callsign=call, grid=grid, picture_id=sp.picture_id, mode=sp.mode,
                        segment=segment, codec_id=sp.codec_id)


@pytest.fixture(scope="module")
def bf(sp):
    return beaconfile.from_picture(sp, 1, _header(sp, 1))


# --- M7: int8 storage ------------------------------------------------------------------

def test_m7_int8_storage_error():
    """M7: the beacon's int8 latents are 37 +- 0.5 dB below unit-RMS latents."""
    a = unit_rms_latents(50600, seed=4, dist="tanh")
    err = picture.from_int8(picture.to_int8(a)) - a
    snr = 10 * np.log10(np.mean(a ** 2) / np.mean(err ** 2))
    assert snr == pytest.approx(37.0, abs=0.5)


# --- M10: beacon file ------------------------------------------------------------------

def test_m10_round_trip_and_size(bf, tmp_path):
    path = tmp_path / "seg1.bin"
    beaconfile.write(path, bf)
    assert path.stat().st_size == 50970 == beaconfile.file_size()
    back = beaconfile.read(path)
    for f in ("waveform", "mode", "segment", "picture_id", "codec_id", "callsign",
              "grid", "ook"):
        assert getattr(back, f) == getattr(bf, f), f
    for f in ("keying", "hdr_bits", "latents_i8"):
        assert np.array_equal(getattr(back, f), getattr(bf, f)), f
    assert back.latents_i8.dtype == np.int8
    assert back.to_bytes() == path.read_bytes()


def test_m10_layout_offsets(bf):
    """The byte layout of design 2.9, field by field."""
    data = bf.to_bytes()
    assert data[:4] == b"QRSB" and data[4] == 1 and data[5] == 0
    assert data[6] == bf.mode and data[7] == bf.segment
    assert int.from_bytes(data[8:12], "little") == bf.picture_id
    assert int.from_bytes(data[12:14], "little") == bf.codec_id
    assert data[14:22] == CALL.ljust(8).encode()
    assert int.from_bytes(data[24:28], "little") == 50600
    assert int.from_bytes(data[28:30], "little") == 20
    assert data[30] == 0 and data[31] == 0
    assert np.array_equal(np.unpackbits(np.frombuffer(data[32:56], np.uint8)),
                          morse.keying_units(CALL))
    hdr = np.unpackbits(np.frombuffer(data[56:366], np.uint8))
    assert np.array_equal(hdr[:2474], bf.hdr_bits) and not hdr[2474:].any()
    assert np.array_equal(np.frombuffer(data[366:366 + 50600], np.int8), bf.latents_i8)


def test_m10_crc_detects_a_flipped_byte(bf):
    data = bytearray(bf.to_bytes())
    for i in (0 + 10, 40, 400, 30000, len(data) - 5):
        bad = bytearray(data)
        bad[i] ^= 0x10
        with pytest.raises(ValueError):
            beaconfile.BeaconFile.from_bytes(bytes(bad))


def _recrc(body: bytes) -> bytes:
    import struct
    import zlib
    return body + struct.pack("<I", zlib.crc32(body) & 0xFFFFFFFF)


def test_m10_keying_mismatch_raises(bf):
    data = bytearray(bf.to_bytes()[:-4])
    data[33] ^= 0x01                         # one keying unit, CRC recomputed
    with pytest.raises(ValueError, match="keying"):
        beaconfile.BeaconFile.from_bytes(_recrc(bytes(data)))
    data = bytearray(bf.to_bytes()[:-4])
    data[100] ^= 0x01                        # one header coded bit
    with pytest.raises(ValueError, match="header"):
        beaconfile.BeaconFile.from_bytes(_recrc(bytes(data)))
    data = bytearray(bf.to_bytes()[:-4])
    data[0:4] = b"QRSX"
    with pytest.raises(ValueError, match="magic"):
        beaconfile.BeaconFile.from_bytes(_recrc(bytes(data)))


def test_m10_ook_flag_and_bad_inputs(sp):
    b = beaconfile.from_picture(sp, 0, _header(sp, 0), ook=True)
    assert b.to_bytes()[30] == 1
    assert beaconfile.BeaconFile.from_bytes(b.to_bytes()).ook
    with pytest.raises(ValueError):                 # header for another segment
        beaconfile.from_picture(sp, 0, _header(sp, 1))
    with pytest.raises(ValueError):                 # unsendable callsign
        beaconfile.from_picture(sp, 0, _header(sp, 0, call="K1ABC?"))


def test_m10_slot_symbols_bin_equals_qrsp_within_int8(sp, bf):
    """M10: symbols from the .bin equal those from the .qrsp within the int8 error,
    and exactly equal the .qrsp's with use_int8=True."""
    h = _header(sp, 1)
    spec = frame.FULL
    s_bin = tx.slot_symbols(bf, Q_TEST, spec)
    s_q = tx.slot_symbols(sp, Q_TEST, spec, segment=1, header=h)
    s_q8 = tx.slot_symbols(sp, Q_TEST, spec, segment=1, header=h, use_int8=True)
    assert np.array_equal(s_bin, s_q8)
    lay = frame.layout(spec)
    known = np.asarray(lay.known_idx)
    assert np.array_equal(s_bin[known], s_q[known])
    assert np.array_equal(s_bin[lay.hdr_bits], s_q[lay.hdr_bits])
    err = s_bin[lay.data] - s_q[lay.data]
    snr = 10 * np.log10(np.mean(s_q[lay.data] ** 2) / np.mean(err ** 2))
    assert snr == pytest.approx(37.0, abs=0.7)


# --- slot symbols and transmit audio ------------------------------------------------------

def test_slot_symbols_structure(sp):
    spec = frame.SHORT
    h = _header(sp, 0)
    sym = tx.slot_symbols(sp, Q_TEST, spec, segment=0, header=h)
    lay = frame.layout(spec)
    assert sym.shape == (spec.n_sym,)
    assert np.array_equal(sym[lay.known_idx], lay.known_val)
    from sstvae.qrss.header import encode
    assert np.array_equal(sym[lay.hdr_bits], 1.0 - 2.0 * encode(h))
    air = picture.air_values(sp.segs[0], 0)[:spec.n_data]
    assert np.allclose(precoder.unprecode(sym[lay.data], Q_TEST), air, atol=1e-12)
    # The scrambler follows q.
    other = tx.slot_symbols(sp, Q_TEST + 1, spec, segment=0, header=h)
    assert not np.allclose(other[lay.data], sym[lay.data])
    # A .qrsp needs a header whenever the frame has one; TINY has none.
    with pytest.raises(ValueError):
        tx.slot_symbols(sp, Q_TEST, spec, segment=0)
    assert tx.slot_symbols(sp, Q_TEST, frame.TINY).shape == (frame.TINY.n_sym,)
    with pytest.raises(ValueError):                 # header for a different picture
        tx.slot_symbols(sp, Q_TEST, spec, header=HeaderFields(
            CALL, None, sp.picture_id ^ 1, sp.mode, 0, sp.codec_id))


def _demod(audio, carrier_hz, pre_s, amplitude=0.5):
    """8 kHz audio -> 250 Hz complex baseband starting at t0 - pre_s."""
    n = np.arange(len(audio))
    z = audio * np.exp(-2j * np.pi * carrier_hz * n / 8000) * (2 / amplitude)
    return resample_poly(z, 1, 32)


@pytest.mark.parametrize("ook", [False, True])
def test_transmit_audio_is_the_baseband_with_windows(sp, ook):
    """The WAV carries ce.baseband: sample 0 at t0 - 12 s, windows keyed with the
    header's callsign, the carrier where asked, and power amplitude^2/2."""
    spec = frame.SHORT
    h = _header(sp, 0)
    audio = tx.transmit_audio(sp, Q_TEST, 1500, spec=spec, header=h, segment=0,
                              ook=ook, lead_in_s=5.0)
    assert len(audio) == tx.slot_samples(spec)
    assert len(audio) == int(np.ceil((12 + spec.keyed_end_pos * T_SYM + 3) * 8000))
    sym = tx.slot_symbols(sp, Q_TEST, spec, segment=0, header=h)
    k = morse.keying_units(CALL)
    z = _demod(audio, 1500, 12.0)
    ref = ce.baseband(sym, spec, 250, -12.0, len(z), keying=k, ook=ook, lead_in_s=5.0)
    mid = slice(250 * 5, len(z) - 250 * 5)
    err = np.mean(np.abs(z[mid] - ref[mid]) ** 2) / np.mean(np.abs(ref[mid]) ** 2)
    assert 10 * np.log10(err) < -30
    body = audio[30 * 8000:140 * 8000]          # keyed, before the first window
    assert np.mean(body ** 2) == pytest.approx(0.125, rel=0.01)
    # The lead-in starts 5 s + 8T before t0, i.e. 12 - 5.24 s into the file.
    first = np.argmax(np.abs(audio) > 0.01) / 8000
    assert first == pytest.approx(12 - 5 - 8 * T_SYM - 0.05, abs=0.06)


def test_transmit_audio_refuses_to_skip_the_callsign(sp):
    with pytest.raises(ValueError, match="callsign"):
        tx.transmit_audio(sp, Q_TEST, spec=frame.SHORT)
    with pytest.raises(ValueError):
        tx.transmit_audio(sp, Q_TEST, spec=frame.SHORT, header=_header(sp), lead_in_s=11)
    with pytest.raises(ValueError):
        tx.transmit_audio(sp, Q_TEST, spec=frame.SHORT, header=_header(sp), pre_s=5.0,
                          lead_in_s=10.0)


def test_transmit_audio_from_bin_uses_its_keying(bf):
    """A beacon file's own keying flag is used unless overridden."""
    spec = frame.SHORT
    b = beaconfile.BeaconFile.from_bytes(bf.to_bytes())
    b.ook = True
    a1 = tx.transmit_audio(b, Q_TEST, spec=spec)
    a2 = tx.transmit_audio(b, Q_TEST, spec=spec, ook=True)
    a3 = tx.transmit_audio(b, Q_TEST, spec=spec, ook=False)
    assert np.array_equal(a1, a2) and not np.array_equal(a1, a3)


def test_slot_timing_follows_the_frame_spec():
    """Nothing hard-codes a slot length: a custom FrameSpec moves the WAV length."""
    custom = frame.FrameSpec("custom", n_data=4096, cw_after=(5000,))
    assert tx.slot_samples(custom) == int(np.ceil(
        (12 + custom.keyed_end_pos * T_SYM + 3) * 8000))
    assert tx.slot_samples(frame.FULL) == int(np.ceil((12 + 1782.678125 - T_SYM / 2 + 3) * 8000))


# --- M6: Si5351 -----------------------------------------------------------------------------

def _si_frame():
    spec = frame.SHORT
    a = unit_rms_latents(spec.n_data, seed=2)
    hdr = np.random.default_rng(5).integers(0, 2, 2474).astype(np.uint8)
    return spec, frame.assemble(spec, hdr, precoder.precode(a, Q_TEST))


def _si_error_db(spec, sym, k, fu, step_hz=0.4, dur=120.0, fs=2000):
    st = si5351.si5351_steps(sym, spec, Q_TEST, Fraction(fu), step_hz, int(dur * fu), keying=k)
    n0 = -485                                   # t0 - 8T at 2 kHz
    n = int(dur * fs)
    ph_s = si5351.synth_phase(st, Fraction(fu), step_hz, fs, n,
                              phase0=si5351.start_phase(sym, spec, k))
    tau = np.arange(n0, n0 + n, dtype=np.int64) * 8 / 485
    ph_i = ce.phase_grid(fs, n0, n, sym, spec) + 2 * np.pi * ce.cw_phase_turns(tau, spec, k)[0]
    lay = frame.layout(spec)
    sel = lay.data[lay.pos[lay.data] < (dur - 10) / T_SYM]
    m = ce.matched_filter(np.exp(1j * ph_i) - np.exp(1j * ph_s), fs, n0, lay.pos[sel])
    snr = 10 * np.log10(K_LIN ** 2 * np.mean(sym[sel] ** 2) / np.mean(m.imag ** 2))
    return snr, st * step_hz


# Measured here: 67.3 / 44.7 / 33.0 dB (7.6 Hz rms, 27 Hz max). The design's 43 / 41 dB come from
# pm_spectrum_synth.synth, whose phase lags the ideal by one fine sample
# (1/240 symbol, ~43 dB on its own); without that lag 990 and 250 Hz are
# limited only by the 0.4 Hz steps and the interpolation, and are better.
@pytest.mark.parametrize("fu, lo, hi", [(990, 41, 75), (250, 39, 48), (125, 31, 35)])
def test_m6_si5351_error_after_matched_filter(fu, lo, hi):
    """M6: synthesis error after the matched filter, 0.4 Hz steps: at least the
    design's 43 / 41 / 33 dB (-2) at 990 / 250 / 125 Hz updates; 125 Hz within +-2."""
    spec, sym = _si_frame()
    snr, f_hz = _si_error_db(spec, sym, morse.keying_units(CALL), fu)
    print(f"M6: {fu} Hz updates: {snr:.1f} dB, rms {np.sqrt(np.mean(f_hz ** 2)):.2f} Hz, "
          f"max {np.max(np.abs(f_hz)):.1f} Hz")
    assert lo <= snr <= hi


def test_m6_si5351_frequency_swing():
    """M6: the frequency swings 7.7 +- 0.5 Hz rms and at most 40 Hz."""
    spec, sym = _si_frame()
    _, f_hz = _si_error_db(spec, sym, morse.keying_units(CALL), 990)
    assert np.sqrt(np.mean(f_hz ** 2)) == pytest.approx(7.7, abs=0.5)
    assert np.max(np.abs(f_hz)) <= 40


def test_si5351_error_feedback_does_not_accumulate():
    """With coarse 5 Hz steps the reached phase still stays within half a step
    of the target at every update, through an FSK callsign window."""
    spec, sym = _si_frame()
    k = morse.keying_units(CALL)
    fu = Fraction(500)
    p_w = spec.win_start_pos[0]
    n_upd = int((p_w + 400) * T_SYM * 500)
    st = si5351.si5351_steps(sym, spec, Q_TEST, fu, 5.0, n_upd, keying=k)
    start = si5351.target_phase(np.array([-8.0]), sym, spec, k)[0]
    reached = start + np.cumsum(st * 5.0 * 2 * np.pi / 500)
    tgt = si5351.target_phase(si5351.update_times_pos(fu, n_upd + 1)[1:], sym, spec, k)
    assert np.max(np.abs(reached - tgt)) <= np.pi * 5.0 / 500 + 1e-9
    # Through the window the FSK shift shows up as the -16.5 Hz steps.
    assert np.min(st * 5.0) <= -15
    # A custom phase function drives the same loop (the C port's seam).
    st2 = si5351.si5351_steps(sym, spec, Q_TEST, fu, 5.0, 2000, keying=k,
                              phase_fn=lambda t: si5351.target_phase(t, sym, spec, k))
    assert np.array_equal(st2, st[:2000])


def test_synth_phase_is_piecewise_linear():
    steps = np.array([10, -3, 0, 7], dtype=np.int32)
    ph = si5351.synth_phase(steps, Fraction(100), 0.5, 1000, 60)
    # 10 samples per update; slope 2 pi f / fs per sample.
    assert ph[0] == 0
    assert ph[10] == pytest.approx(2 * np.pi * 5.0 / 100)
    assert np.allclose(np.diff(ph[:10]), 2 * np.pi * 5.0 / 1000)
    assert np.allclose(ph[40:], ph[40])


# --- CLIs -------------------------------------------------------------------------------------

def _run(*args):
    r = subprocess.run([sys.executable, *map(str, args)], cwd=REPO_ROOT,
                       capture_output=True, text=True, timeout=120)
    return r


def test_cli_beacon_and_transmit(sp, tmp_path):
    """qrss_beacon.py writes the beacon file; qrss_transmit.py turns it (or the
    .qrsp with --callsign) into the same audio as tx.transmit_audio."""
    qrsp = tmp_path / "pic.qrsp"
    picture.save_qrsp(qrsp, sp)
    binf = tmp_path / "seg0.bin"
    r = _run(REPO_ROOT / "qrss_beacon.py", qrsp, binf, "--callsign", CALL, "--grid", "FN42")
    assert r.returncode == 0, r.stderr
    assert binf.stat().st_size == 50970
    bf = beaconfile.read(binf)
    assert bf.callsign == CALL and bf.grid == "FN42" and not bf.ook

    wav = tmp_path / "tx.wav"
    r = _run(REPO_ROOT / "qrss_transmit.py", binf, wav, "--slot", "2025-10-01T00:00Z",
             "--frame", "short", "--float", "--freq", "1200", "--lead-in", "2")
    assert r.returncode == 0, r.stderr
    got = wavio.read_wav(str(wav))
    want = tx.transmit_audio(bf, Q_TEST, 1200, spec=frame.SHORT, lead_in_s=2.0)
    assert np.allclose(got, want.astype(np.float32), atol=1e-7)

    wav2 = tmp_path / "tx2.wav"
    r = _run(REPO_ROOT / "qrss_transmit.py", qrsp, wav2, "--slot", "2025-10-01T00:00Z",
             "--frame", "short", "--float", "--callsign", CALL, "--grid", "FN42")
    assert r.returncode == 0, r.stderr
    want2 = tx.transmit_audio(sp, Q_TEST, 1500, spec=frame.SHORT, segment=0,
                              header=_header(sp, 0))
    assert np.allclose(wavio.read_wav(str(wav2)), want2.astype(np.float32), atol=1e-7)


def test_cli_refusals(sp, tmp_path):
    qrsp = tmp_path / "pic.qrsp"
    picture.save_qrsp(qrsp, sp)
    r = _run(REPO_ROOT / "qrss_transmit.py", qrsp, tmp_path / "x.wav",
             "--slot", "2025-10-01T00:00Z", "--frame", "tiny")
    assert r.returncode != 0 and "callsign" in r.stderr
    r = _run(REPO_ROOT / "qrss_transmit.py", qrsp, tmp_path / "x.wav",
             "--slot", "2025-10-01T00:07Z", "--callsign", CALL, "--frame", "tiny")
    assert r.returncode != 0 and "quarter hour" in r.stderr
    r = _run(REPO_ROOT / "qrss_beacon.py", qrsp, tmp_path / "x.bin", "--callsign", "k1abc")
    assert r.returncode != 0


@pytest.mark.codec
def test_cli_encode(tmp_path):
    """qrss_encode.py: v5 encoder, codec ID 0xD1D8 from the metadata, picture ID
    printed and stored."""
    enc = resolve_model("encoder", "fp16")
    img = REPO_ROOT / "wonder_wheel.jpg"
    if not img.exists():
        pytest.skip("wonder_wheel.jpg not in the repo")
    from pathlib import Path
    out = tmp_path / "ww.qrsp"
    r = _run(REPO_ROOT / "qrss_encode.py", img, out, "--model", Path(enc).parent,
             "--mode", "A")
    assert r.returncode == 0, r.stderr
    sp = picture.load_qrsp(out)
    assert sp.codec_id == 0xD1D8 and sp.mode == 0
    assert sp.id_hex in r.stdout

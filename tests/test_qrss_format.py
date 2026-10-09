"""QRSSTVAE core format: frame layout, precoder, slot count, picture
store and Morse (design 10.1: F1, F2, F4, F5, F7), plus the shared types.

All fast. The codec parts of F5 are marked `codec` and skip cleanly
when the v5 models are absent (see `qrss_helpers.resolve_model`).
"""

import hashlib
import importlib.util
import struct
from datetime import datetime, timedelta, timezone
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
import scipy.linalg

from qrss_helpers import (
    Q_TEST, REPO_ROOT, latent_snr_db, load_codec, resolve_model,
    synthetic_full_latents, unit_rms_latents,
)
from sstvae import config
from sstvae.qrss import constants as C
from sstvae.qrss import frame, morse, picture, precoder, sequences
from sstvae.qrss import types as qt

# --- F1: frame counts, positions and timing ---------------------------------------


def test_full_frame_counts():
    f = frame.FULL
    lay = frame.layout(f)
    assert f.n_data_sym == 53974
    assert f.n_sym == 57274
    assert f.n_pos == 58810
    assert (len(lay.hdr_ref), len(lay.hdr_bits), len(lay.hdr_spare),
            len(lay.data_ref), len(lay.data)) == (165, 2474, 1, 3374, 50600)
    assert f.n_data == config.TRANSMIT_LATENTS_PER_GROUP
    assert f.win_start_pos == (13888, 28734, 43580, 58426)
    assert f.duration_s == 1782.678125
    assert f.n_pos * C.T_NUM * C.FS == 14261425 * C.T_DEN  # exact samples at 8 kHz
    assert f.cw_after[-1] == f.n_sym                         # the last window ends it
    assert f.keyed_end_pos == f.n_pos - 0.5


def test_full_window_times():
    """Windows start 420.96, 870.98, 1321.00 and 1771.02 s after t0; the
    frame ends 1782.68 s after t0, 29:43.7 past the quarter hour."""
    spans = frame.FULL.window_spans_s()
    assert [round(a, 2) for a, _ in spans] == [420.96, 870.98, 1321.00, 1771.02]
    assert all(abs((b - a) - 384 * C.T_SYM) < 1e-9 for a, b in spans)
    end = 1.0 + spans[-1][1]
    assert abs(end - (29 * 60 + 43.7)) < 0.05


@pytest.mark.parametrize("name,n_sym,n_pos,dur", [
    ("medium", 20777, 21545, 653.1),
    ("short", 7670, 8438, 255.8),
    ("tiny", 1753, 1753, 53.1),
])
def test_preset_counts(name, n_sym, n_pos, dur):
    f = frame.PRESETS[name]
    assert f.n_sym == n_sym and f.n_pos == n_pos
    assert round(f.duration_s, 1) == dur
    lay = frame.layout(f)
    assert len(lay.data) == f.n_data
    assert lay.win_pos.shape == (f.n_win, 2)


def test_tiny_has_no_header_and_no_windows():
    lay = frame.layout(frame.TINY)
    assert len(lay.hdr_bits) == len(lay.hdr_ref) == len(lay.hdr_spare) == 0
    assert frame.TINY.n_win == 0
    assert np.array_equal(lay.pos, np.arange(frame.TINY.n_sym))
    # last window not at the end -> keyed until the last pulse tail
    assert frame.TINY.keyed_end_pos == frame.TINY.n_sym - 1 + C.SPAN


@pytest.mark.parametrize("kwargs", [
    dict(cw_after=(5000, 4000)),               # not increasing
    dict(cw_after=(5000, 5000)),               # not strictly increasing
    dict(cw_after=(3300,)),                    # inside the header (must be > n_pre+n_hdr)
    dict(cw_after=(7671,)),                    # past the end
    dict(n_hdr=1000),
    dict(n_pre=600),
    dict(n_data=0),
])
def test_invalid_frame_specs_raise(kwargs):
    base = dict(name="bad", n_data=4096, cw_after=(5000, 7670))
    base.update(kwargs)
    with pytest.raises(ValueError):
        frame.FrameSpec(**base)


def test_window_may_follow_the_first_data_symbol():
    f = frame.FrameSpec("edge", n_data=4096, cw_after=(3301, 7670))
    assert f.win_start_pos == (3301, 8054)


def test_positions_skip_the_windows():
    f = frame.FULL
    lay = frame.layout(f)
    pos = lay.pos
    assert pos[0] == 0
    # strictly increasing, with a 384 jump exactly where each window sits
    jumps = np.diff(pos)
    assert set(np.unique(jumps)) == {1, 1 + C.CW_WINDOW}
    assert np.flatnonzero(jumps > 1).tolist() == [c - 1 for c in f.cw_after[:-1]]
    for w, (p0, p1) in enumerate(lay.win_pos):
        assert p1 - p0 == C.CW_WINDOW - 1
        assert p0 == f.cw_after[w] + C.CW_WINDOW * w
        assert not np.any((pos >= p0) & (pos <= p1))      # no symbol inside a window
    assert f.pos_of(13887) == 13887 and f.pos_of(13888) == 13888 + 384
    assert np.array_equal(f.pos_of(np.arange(f.n_sym)), pos)
    assert pos[-1] == f.n_pos - C.CW_WINDOW - 1


def test_reference_grid_and_known_symbols():
    f = frame.FULL
    lay = frame.layout(f)
    j = lay.ref - f.n_pre
    assert np.all(j % 16 == 0) and j[0] == 0
    assert np.array_equal(lay.ref, np.concatenate([lay.hdr_ref, lay.data_ref]))
    # data references fall at data-local index = 0 mod 16 too
    assert np.all((lay.data_ref - f.n_pre - f.n_hdr) % 16 == 0)
    # every symbol is in exactly one class
    classes = np.concatenate([lay.pre, lay.ref, lay.hdr_bits, lay.hdr_spare, lay.data])
    assert np.array_equal(np.sort(classes), np.arange(f.n_sym))
    assert lay.hdr_spare.tolist() == [f.n_pre + f.n_hdr - 1]
    assert len(lay.known_idx) == f.n_pre + f.n_ref + 1
    assert np.all(np.diff(lay.known_idx) > 0)
    assert set(np.unique(lay.known_val)) <= {-1, 1}


def test_assemble_places_every_class():
    f = frame.SHORT
    lay = frame.layout(f)
    bits = (np.arange(C.N_HDR_BITS) % 3 == 0).astype(np.uint8)
    x = unit_rms_latents(f.n_data, seed=3)
    sym = frame.assemble(f, bits, x)
    assert sym.shape == (f.n_sym,)
    assert np.array_equal(sym[lay.pre], sequences.preamble_ce())
    assert np.array_equal(sym[lay.ref], sequences.references_ce(f.n_ref))
    assert sym[lay.hdr_spare].tolist() == [1.0]
    assert np.array_equal(sym[lay.hdr_bits], 1.0 - 2.0 * bits)
    assert np.array_equal(sym[lay.data], x)
    h, d = frame.disassemble(sym, f)
    assert np.array_equal(h, 1.0 - 2.0 * bits) and np.array_equal(d, x)
    # batch axes pass through disassemble
    h2, d2 = frame.disassemble(np.stack([sym, -sym]), f)
    assert d2.shape == (2, f.n_data) and np.array_equal(d2[1], -x)


def test_reference_counter_is_shared_across_frames():
    """Reference m takes bit m of one stream, header references first."""
    for f in (frame.SHORT, frame.MEDIUM):
        sym = frame.assemble(f, np.zeros(C.N_HDR_BITS, np.uint8), np.zeros(f.n_data))
        assert np.array_equal(sym[frame.layout(f).ref],
                              sequences.references_ce(frame.FULL.n_ref)[:f.n_ref])


def test_assemble_rejects_bad_inputs():
    f = frame.SHORT
    with pytest.raises(ValueError):
        frame.assemble(f, None, np.zeros(f.n_data))
    with pytest.raises(ValueError):
        frame.assemble(f, np.zeros(10, np.uint8), np.zeros(f.n_data))
    with pytest.raises(ValueError):
        frame.assemble(f, np.full(C.N_HDR_BITS, 2, np.uint8), np.zeros(f.n_data))
    with pytest.raises(ValueError):
        frame.assemble(f, np.zeros(C.N_HDR_BITS, np.uint8), np.zeros(f.n_data + 1))
    t = frame.TINY
    sym = frame.assemble(t, None, np.zeros(t.n_data))
    assert sym.shape == (t.n_sym,)
    with pytest.raises(ValueError):
        frame.assemble(t, np.zeros(C.N_HDR_BITS, np.uint8), np.zeros(t.n_data))


def test_layout_is_cached_and_read_only():
    a, b = frame.layout(frame.FULL), frame.layout(frame.FrameSpec())
    assert a is b
    with pytest.raises(ValueError):
        a.data[0] = 0
    assert frame.get("short") is frame.SHORT and frame.get(frame.TINY) is frame.TINY
    with pytest.raises(ValueError):
        frame.get("huge")


# --- F2: precoder ---------------------------------------------------------------------


def test_block_sizes():
    assert precoder.block_sizes(50600) == [64] * 790 + [32, 8]
    assert precoder.block_sizes(16384) == [64] * 256
    assert precoder.block_sizes(4096) == [64] * 64
    assert precoder.block_sizes(1024) == [64] * 16
    assert precoder.block_sizes(63) == [32, 16, 8, 4, 2, 1]
    for n in (0, 1, 50600, 777):
        assert sum(precoder.block_sizes(n)) == n


@pytest.mark.parametrize("m", [8, 32, 64])
def test_fast_wht_equals_the_sylvester_matrix(m):
    v = np.cos(np.arange(5 * m) * 0.37).reshape(5, m)
    expect = v @ (scipy.linalg.hadamard(m) / np.sqrt(m)).T
    assert np.max(np.abs(precoder.wht(v) - expect)) < 1e-12


def test_blockwise_wht_on_a_segment():
    a = unit_rms_latents(50600, seed=1)
    y = precoder.wht(a)
    for start, m in ((0, 64), (64 * 790, 32), (64 * 790 + 32, 8)):
        blk = a[start:start + m]
        assert np.allclose(y[start:start + m],
                           scipy.linalg.hadamard(m) @ blk / np.sqrt(m), atol=1e-12)
    assert np.max(np.abs(precoder.wht(y) - a)) < 1e-12        # self-inverse
    assert abs(np.sum(y ** 2) - np.sum(a ** 2)) < 1e-8         # orthonormal


def test_precode_round_trip():
    a = unit_rms_latents(50600, seed=2)
    x = precoder.precode(a, Q_TEST)
    assert np.max(np.abs(precoder.unprecode(x, Q_TEST) - a)) < 1e-12
    # a different slot's precoding does not undo this one
    assert latent_snr_db(precoder.unprecode(x, Q_TEST + 1), a) < 1.0
    # explicit definition: x = WHT(c_q * a)
    c = sequences.scrambler(Q_TEST, 50600)
    assert np.array_equal(x, precoder.wht(c * a))


def test_block_mean_and_index():
    v = np.arange(50600, dtype=float)
    bm = precoder.block_mean(v)
    assert bm[0] == 31.5 and bm[-1] == np.mean(v[-8:]) and bm[-9] == np.mean(v[-40:-8])
    idx = precoder.block_index(50600)
    assert idx[-1] == 791 and np.bincount(idx)[-2:].tolist() == [32, 8]


# --- F4: quarter-hour count ----------------------------------------------------------


def test_quarter_hour_count():
    t = datetime(2025, 10, 1, tzinfo=timezone.utc)
    assert sequences.quarter_hour_count(t) == Q_TEST == 1954752
    assert sequences.quarter_hour_count(t + timedelta(minutes=15)) == Q_TEST + 1
    t_other_tz = datetime(2025, 10, 1, 2, tzinfo=timezone(timedelta(hours=2)))
    assert sequences.quarter_hour_count(t_other_tz) == Q_TEST
    assert sequences.quarter_hour_start(Q_TEST) == t
    for bad in (t + timedelta(seconds=1), t + timedelta(minutes=7),
                t + timedelta(microseconds=1)):
        with pytest.raises(ValueError):
            sequences.quarter_hour_count(bad)
    with pytest.raises(ValueError):
        sequences.quarter_hour_count(datetime(2025, 10, 1))      # naive


# --- F5: picture store and IDs -------------------------------------------------------


def _closed_form_full():
    g = np.arange(config.LATENT_GROUPS * config.GROUP_LATENTS, dtype=np.float64)
    return np.sin(0.001 * g * g % 6.283) + 0.3 * np.cos(g)


def test_prepare_segment_unit_rms_and_never_sent_zero():
    full = synthetic_full_latents(4) * 1.7
    for g in range(3):
        seg = picture.prepare_segment(full, g)
        assert seg.dtype == np.float16 and seg.shape == (52800,)
        sent = picture.air_to_canonical(g)
        assert np.all(seg[picture.never_sent(g)] == 0)
        assert picture.never_sent(g).sum() == config.DROPPED_LATENTS_PER_GROUP
        rms = np.sqrt(np.mean(seg[sent].astype(np.float64) ** 2))
        assert abs(rms - 1) < 1e-3                       # fp16 rounding only
        # the scaling was computed on the sent values only, in float64
        src = full[g * 52800:(g + 1) * 52800]
        expect = (src / np.sqrt(np.mean(src[sent] ** 2)))
        expect[picture.never_sent(g)] = 0
        assert np.array_equal(seg, expect.astype(np.float16))


def test_picture_id_by_hand():
    """The ID is the first 4 bytes of SHA-256 over the documented bytes."""
    full = _closed_form_full()
    segs = [picture.prepare_segment(full, g) for g in range(2)]
    head = struct.pack(">H", 0xD1D8) + bytes([1])
    body = b"".join(np.asarray(s, dtype="<f2").tobytes() for s in segs)
    by_hand = int.from_bytes(hashlib.sha256(head + body).digest()[:4], "big")
    assert picture.picture_id(0xD1D8, 1, segs) == by_hand
    # pinned for mode A on the closed-form tensor
    sp = picture.StoredPicture.from_latents(full, 0xD1D8, 0)
    assert sp.picture_id == 0xD349C671
    assert sp.id_hex == "d349c671"
    # every input changes it
    assert picture.picture_id(0xD1D9, 0, segs[:1]) != sp.picture_id
    assert picture.picture_id(0xD1D8, 1, segs) != sp.picture_id
    s2 = segs[0].copy()
    s2[picture.air_to_canonical(0)[0]] += np.float16(0.01)
    assert picture.picture_id(0xD1D8, 0, [s2]) != sp.picture_id
    with pytest.raises(ValueError):
        picture.picture_id(0xD1D8, 2, segs)              # mode C needs 3 groups
    with pytest.raises(ValueError):
        picture.picture_id(1 << 16, 0, segs[:1])


def test_air_canonical_maps():
    for g in range(3):
        perm = picture.air_to_canonical(g)
        assert perm.dtype == np.intp and perm.shape == (50600,)
        assert len(np.unique(perm)) == 50600
        seg = np.arange(52800, dtype=np.float16) / np.float16(4096)
        air = picture.air_values(seg, g)
        assert air.dtype == np.float64
        back = picture.canonical_from_air(air, g)
        assert np.array_equal(back[perm], air)
        assert np.all(back[picture.never_sent(g)] == 0)
    z = np.ones((2, 50600), np.float32)
    assert picture.canonical_from_air(z, 1).dtype == np.float32
    assert picture.canonical_from_air(z, 1).shape == (2, 52800)
    with pytest.raises(ValueError):
        picture.air_to_canonical(3)


def test_int8_storage():
    a = np.array([0.0, 0.125, 0.375, -0.125, 6.35, 6.4, -7.0, 1.0, 0.124])
    q = picture.to_int8(a)
    assert q.dtype == np.int8
    assert q.tolist() == [0, 2, 8, -2, 127, 127, -127, 20, 2]    # half to even; +-127
    assert np.array_equal(picture.from_int8(q), q / 20.0)
    lat = unit_rms_latents(50600, seed=5, dist="tanh")
    snr = latent_snr_db(picture.from_int8(picture.to_int8(lat)), lat)
    assert 36.5 < snr < 37.5                                     # spec: 37 dB


def test_qrsp_round_trip(tmp_path):
    full = synthetic_full_latents(6)
    sp = picture.StoredPicture.from_latents(full, 0xD1D8, 2)
    assert len(sp.segs) == 3
    path = tmp_path / "pic.qrsp"
    picture.save_qrsp(path, sp)
    assert path.exists() and not (tmp_path / "pic.qrsp.npz").exists()
    back = picture.load_qrsp(path)
    assert (back.mode, back.codec_id, back.picture_id) == (2, 0xD1D8, sp.picture_id)
    assert all(np.array_equal(a, b) and b.dtype == np.float16
               for a, b in zip(sp.segs, back.segs))


def test_qrsp_rejects_a_wrong_id(tmp_path):
    sp = picture.StoredPicture.from_latents(synthetic_full_latents(7), 0x1234, 0)
    bad = picture.StoredPicture(sp.mode, sp.codec_id, sp.segs, sp.picture_id ^ 1)
    with pytest.raises(ValueError):
        picture.save_qrsp(tmp_path / "x.qrsp", bad)
    path = tmp_path / "y.qrsp"
    with open(path, "wb") as f:
        np.savez(f, version=1, mode=0, codec_id=0x1234,
                 segs=np.stack(sp.segs), picture_id=sp.picture_id ^ 1)
    with pytest.raises(ValueError):
        picture.load_qrsp(path)


@pytest.mark.codec
def test_codec_id_from_v5_onnx():
    enc = resolve_model("encoder", "fp16")
    assert picture.codec_id_from_onnx(enc) == 0xD1D8
    assert picture.codec_id_from_onnx(str(Path(enc).parent)) == 0xD1D8


@pytest.mark.codec
def test_v5_picture_id_is_stable_through_the_store(tmp_path):
    codec = load_codec("fp16")
    img_path = REPO_ROOT / "wonder_wheel.jpg"
    if not img_path.exists():
        pytest.skip("wonder_wheel.jpg not in the repo")
    from sstvae import images
    img = images.load_image(img_path)
    sp1 = picture.StoredPicture.from_latents(codec.encode(img), 0xD1D8, 0)
    sp2 = picture.StoredPicture.from_latents(codec.encode(img), 0xD1D8, 0)
    assert sp1.picture_id == sp2.picture_id
    picture.save_qrsp(tmp_path / "ww.qrsp", sp1)
    picture.save_qrsp(tmp_path / "ww2.qrsp", picture.load_qrsp(tmp_path / "ww.qrsp"))
    assert picture.load_qrsp(tmp_path / "ww2.qrsp").picture_id == sp1.picture_id


# --- F7: Morse -------------------------------------------------------------------------

# K -.-   1 .----   A .-   B -...   C -.-.  (dot 1, dash 3, gaps 1 and 3)
_K1ABC = ("111010111" "000" "10111011101110111" "000" "10111" "000"
          "111010101" "000" "11101011101")


def _cwid_sim():
    path = Path("/mnt/project-files/qrss-spec/sims/cwid/cwid.py")
    if not path.exists():
        return None
    spec = importlib.util.spec_from_file_location("_qrss_cwid_sim", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_morse_lengths_and_pattern():
    assert len(morse.morse_keying("00000000")) == 173
    k = morse.keying_units("K1ABC")
    assert k.dtype == np.uint8 and k.shape == (192,)
    assert not k[:8].any()
    call = "".join(map(str, morse.morse_keying("K1ABC")))
    assert call == _K1ABC
    assert "".join(map(str, k[8:8 + len(_K1ABC)])) == _K1ABC
    assert not k[8 + len(_K1ABC):].any()
    assert k[8] == 1 and morse.keying_units("00000000")[8 + 172] == 1


def test_morse_table_matches_the_spec_sim():
    sim = _cwid_sim()
    if sim is None:
        pytest.skip("spec sims not mounted")
    assert sim.MORSE == morse.MORSE
    for call in ("K1ABC", "00000000", "VK2/G4XYZ"[:8], "E", "Q/0"):
        assert morse.morse_keying(call).tolist() == sim.keying(call).astype(int).tolist()


def test_trailing_spaces_are_stripped():
    assert np.array_equal(morse.keying_units("K1ABC   "), morse.keying_units("K1ABC"))


@pytest.mark.parametrize("bad", ["K1ABCDEFG", "k1abc", "K1 BC", "K1?BC", "", "   ",
                                 " K1ABC", "K1-AB"])
def test_unsendable_callsigns_raise(bad):
    with pytest.raises(ValueError):
        morse.keying_units(bad)


# --- shared types --------------------------------------------------------------------


def _fake_pass(header=None):
    n = 4096
    rng = np.random.default_rng(11)
    z = rng.standard_normal(n).astype(np.float32)
    return qt.PassResult(
        uid=qt.pass_uid(Q_TEST, 1500.25, z), q=Q_TEST, frame="short", waveform=0,
        f_hz=1500.25,
        timing=qt.Timing(tau0=3000.5, ppm=12.0, gamma=0.0, z=40.0, cov=np.eye(2) * 1e-3),
        report=qt.TrackReport(offset_hz=0.25, drift_hz_per_min=0.1, wander_hz_rms=0.02,
                              doppler_hz=0.1, ppm=12.0, snr2500_db=-12.0, z_ref=40.0,
                              kappa=1.02, slip_free=True, suspect=False),
        z=z, w=np.full(n, 3.0, np.float32), hdr_llr=np.zeros(2474, np.float32),
        header=header,
        cw=qt.CwIdResult(text="K1ABC", z_match=9.5, keying="fsk", agrees=True),
        ch=(rng.standard_normal(1000) + 1j).astype(np.complex64), ch_fs=C.CH_FS,
        ch_t0_index=3000.5, f_mix_hz=Fraction(96017, 64),
        psi=np.ones(frame.SHORT.n_pos, np.float32), estimator="joint",
    )


def _assert_same_pass(a, b):
    for k in ("uid", "q", "frame", "waveform", "f_hz", "ch_fs", "ch_t0_index",
              "f_mix_hz", "estimator", "em_round", "report", "cw", "header"):
        assert getattr(a, k) == getattr(b, k), k
    for k in ("z", "w", "hdr_llr", "ch", "psi"):
        assert np.array_equal(getattr(a, k), getattr(b, k)) and \
            getattr(a, k).dtype == getattr(b, k).dtype, k
    assert np.array_equal(a.timing.cov, b.timing.cov)
    assert (a.timing.tau0, a.timing.ppm, a.timing.z) == (b.timing.tau0, b.timing.ppm, b.timing.z)


def test_pass_result_round_trip(tmp_path):
    p = _fake_pass()
    assert p.uid.startswith(f"{Q_TEST}_1500250_") and len(p.uid.split("_")[2]) == 8
    path = tmp_path / f"{p.uid}.npz"
    p.save(path)
    _assert_same_pass(p, qt.PassResult.load(path))


def test_pass_result_round_trip_with_header(tmp_path):
    header = pytest.importorskip("sstvae.qrss.header")
    h = header.HeaderFields(callsign="K1ABC", grid="FN42", picture_id=0xD349C671,
                            mode=0, segment=0, codec_id=0xD1D8)
    p = _fake_pass(h)
    p.save(tmp_path / "p.npz")
    _assert_same_pass(p, qt.PassResult.load(tmp_path / "p.npz"))

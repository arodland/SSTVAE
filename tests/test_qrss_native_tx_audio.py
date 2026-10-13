"""C++ pass audio (native/core/qrss/tx, wav) against sstvae.qrss.tx.transmit_audio
and sstvae.wavio.

The whole pass, for CE and L: tiny and short (and l-tiny, l-short) at
carriers 1200, 1500, 2650 and 1500.3 Hz, lead-in 0 and 5 s, FSK and OOK on
CE, from a .qrsp with a header and from a beacon file; one FULL and one
l-full pass. CE is bit for bit when numpy's float64 trig is the C
library's and its complex multiply the fused one (probed as in
test_qrss_native_ce.py; true of the reference platform), else within
1e-12; L goes through the clipper's FFTs and is held to 1e-9 (measured
~1e-15). The streaming PassAudio gives the same samples in any block size
(1, 7, 4096, 1e5, the whole), as float64 or float32, and every refusal is
compared by type and text. The WAV files -- peak-normalised int16 and
float32 -- are byte-identical to wavio's (scipy's) for the same samples.
"""

import math
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from qrss_helpers import Q_TEST, synthetic_full_latents
from qrss_native import require
from qrss_native import tx as rows_mod
from sstvae import wavio
from sstvae.qrss import beaconfile, frame, lframe, picture
from sstvae.qrss import tx as ref
from sstvae.qrss.constants import WAVEFORM_L
from sstvae.qrss.header import HeaderFields

CALL = "K1ABC/P"
CARRIERS = (1200, 1500, 2650, 1500.3)
TINY, SHORT = frame.TINY, frame.SHORT
LTINY, LSHORT = lframe.get("l-tiny"), lframe.get("l-short")
L_TOL = 1e-9
CE_TOL = 1e-12


def _libm_trig() -> bool:
    """numpy's float64 sin/cos are the C library's (what the C++ calls)."""
    x = np.concatenate([np.linspace(-60.0, 60.0, 20011), np.random.default_rng(0).uniform(0, 7, 5000)])
    return (np.array_equal(np.cos(x), [math.cos(v) for v in x])
            and np.array_equal(np.sin(x), [math.sin(v) for v in x]))


def _fused_cmul() -> bool:
    """numpy's complex multiply forms Re as fma(ar, br, -(ai*bi)), as the C++ does."""
    rng = np.random.default_rng(1)
    a = rng.standard_normal(4096) + 1j * rng.standard_normal(4096)
    b = np.exp(1j * rng.uniform(0, 2 * np.pi, 4096))
    want = [math.fma(p.real, q.real, -(p.imag * q.imag)) for p, q in zip(a, b)]
    return np.array_equal((a * b).real, want)


EXACT_CE = _libm_trig() and _fused_cmul()


@pytest.fixture(scope="module")
def nat(native):
    return require(native, "tx")


@pytest.fixture(scope="module")
def ad(native):
    """The --native-qrss adapters by attribute name."""
    require(native, "tx")
    return {attr: f for (_, attr), f in rows_mod.rows(native.qrss).items()}


def _header(sp, segment=1, call=CALL, grid="FN42"):
    return HeaderFields(callsign=call, grid=grid, picture_id=sp.picture_id, mode=sp.mode,
                        segment=segment, codec_id=sp.codec_id)


@pytest.fixture(scope="module")
def src(tmp_path_factory):
    """A .qrsp read back from disk, its header, and an OOK and an FSK beacon file."""
    d = tmp_path_factory.mktemp("qrss_tx_audio")
    sp0 = picture.StoredPicture.from_latents(synthetic_full_latents(3), 0xD1D8, 1)
    picture.save_qrsp(d / "p.qrsp", sp0)
    sp = picture.load_qrsp(d / "p.qrsp")
    h = _header(sp, 1)
    beaconfile.write(d / "ook.bin", beaconfile.from_picture(sp, 1, h, ook=True))
    beaconfile.write(d / "fsk.bin", beaconfile.from_picture(sp, 1, h))
    return {"sp": sp, "h": h, "ook": beaconfile.read(d / "ook.bin"),
            "fsk": beaconfile.read(d / "fsk.bin"), "dir": d}


def _check(got, want, tol, exact, what=""):
    assert isinstance(got, np.ndarray) and got.dtype == want.dtype, what
    assert got.shape == want.shape, (what, got.shape, want.shape)
    if exact:
        assert got.tobytes() == want.tobytes(), (what, float(np.max(np.abs(got - want))))
    elif got.size:
        err = float(np.max(np.abs(got - want)))
        assert err <= tol, (what, err)


def _outcome(f, *args, **kw):
    try:
        return ("ok", f(*args, **kw))
    except Exception as e:      # noqa: BLE001 -- any refusal is compared
        return ("raise", type(e), str(e))


def _same_refusal(f_ref, f_nat, *args, **kw):
    """Both refuse alike, or both succeed (the audio is compared elsewhere)."""
    want, got = _outcome(f_ref, *args, **kw), _outcome(f_nat, *args, **kw)
    assert got[0] == want[0], (want, got)
    if want[0] == "raise":
        assert got[1:] == want[1:], (want, got)
    else:
        assert len(got[1]) == len(want[1])
    return want


def _ce_cases(src):
    """(id, source, kwargs): CE tiny and short, every carrier, both sources,
    FSK and OOK (a .qrsp asked for OOK, an OOK file left alone or asked
    for FSK), lead-in alternating 0 and 5 s. Every combination on tiny;
    on short (four times as long) half of them, each carrier with one
    .qrsp and one beacon file, of opposite keying."""
    variants = (("qrsp-fsk", src["sp"], dict(segment=1, header=src["h"])),
                ("bin-ook", src["ook"], {}),
                ("qrsp-ook", src["sp"], dict(segment=1, header=src["h"], ook=True)),
                ("bin-fsk", src["ook"], dict(ook=False)))
    out = []
    for spec in (TINY, SHORT):
        for i, car in enumerate(CARRIERS):
            lead = 5.0 if i % 2 else 0.0
            these = variants if spec is TINY else variants[2 * (i % 2):2 * (i % 2) + 2]
            for name, s, kw in these:
                out.append((f"{spec.name}-{car}-{name}-lead{lead:g}", s,
                            dict(carrier_hz=car, spec=spec, lead_in_s=lead, **kw)))
    return out


def _l_cases(src):
    out = []
    for spec in (LTINY, LSHORT):
        for i, car in enumerate(CARRIERS):
            lead = 5.0 if i % 2 else 0.0
            for name, s, kw in (("qrsp", src["sp"], dict(segment=1, header=src["h"])),
                                ("bin-fsk", src["fsk"], {})):
                out.append((f"{spec.name}-{car}-{name}-lead{lead:g}", s,
                            dict(carrier_hz=car, spec=spec, lead_in_s=lead, **kw)))
    return out


# --- the whole pass ----------------------------------------------------------------------

def test_ce_passes(nat, src):
    for what, s, kw in _ce_cases(src):
        want = ref.transmit_audio(s, Q_TEST, **kw)
        got = nat.transmit_audio(s, Q_TEST, **kw)
        assert len(got) == ref.slot_samples(kw["spec"]), what
        _check(got, want, CE_TOL, EXACT_CE, what)


def test_l_passes(nat, src):
    for what, s, kw in _l_cases(src):
        want = ref.transmit_audio(s, Q_TEST, **kw)
        got = nat.transmit_audio(s, Q_TEST, **kw)
        assert len(got) == ref.slot_samples(kw["spec"]), what
        _check(got, want, L_TOL, False, what)


def test_l_entry_point(nat, ad, src):
    """_transmit_audio_l directly, all eleven arguments positional."""
    sp, h = src["sp"], src["h"]
    args = (sp, Q_TEST + 1, Fraction(30003, 20), LTINY, 1, h, 2.0, 12.0, 3.0, 0.4, 7.3)
    want = ref._transmit_audio_l(*args)
    _check(ad["_transmit_audio_l"](*args), want, L_TOL, False)
    _check(nat._transmit_audio_l(*args), want, L_TOL, False)


def test_full_passes(nat, src):
    """One FULL CE pass and one l-full pass, compared once."""
    sp, h = src["sp"], src["h"]
    kw = dict(segment=1, header=h, lead_in_s=5.0)
    want = ref.transmit_audio(sp, Q_TEST, 1500.3, spec=frame.FULL, **kw)
    got = nat.transmit_audio(sp, Q_TEST, 1500.3, spec=frame.FULL, **kw)
    assert len(got) == ref.slot_samples(frame.FULL)
    _check(got, want, CE_TOL, EXACT_CE, "full")
    lfull = lframe.get("l-full")
    want = ref.transmit_audio(sp, Q_TEST, 1200, spec=lfull, **kw)
    got = nat.transmit_audio(sp, Q_TEST, 1200, spec=lfull, **kw)
    assert len(got) == ref.slot_samples(lfull)
    _check(got, want, L_TOL, False, "l-full")


def test_argument_forms(nat, src):
    """Carriers as float, int, str, numpy scalar and Fraction; q, segment,
    lead-in and pre_s as numpy and int; ook by truthiness; chunk_s
    shorter than a pass, longer, and not a whole number of samples;
    amplitude and pre/post other than the defaults."""
    sp, h, bf = src["sp"], src["h"], src["ook"]
    cases = [
        (sp, Q_TEST, "1500.3", dict(spec=TINY, segment=1, header=h)),
        (sp, Q_TEST, np.float64(1500.3), dict(spec=TINY, segment=1, header=h)),
        (sp, Q_TEST, Fraction(30003, 20), dict(spec=TINY, segment=1, header=h)),
        (sp, np.uint64(Q_TEST), np.int32(1500), dict(spec=TINY, segment=np.int64(1), header=h)),
        (sp, Q_TEST + 7, 1500, dict(spec=TINY, segment=1, header=h, ook=1, lead_in_s=3)),
        (bf, Q_TEST, 1500, dict(spec=TINY, ook="")),
        (bf, Q_TEST, 1500, dict(spec=TINY, ook=np.bool_(False), pre_s=np.float64(13.5))),
        (bf, (1 << 40) + 12345, 1700, dict(spec=TINY, chunk_s=7.3, pre_s=20, post_s=0.25)),
        (bf, Q_TEST, 1500, dict(spec=TINY, chunk_s=1e9, amplitude=0.9)),
        # short passes (the reference loops per chunk in Python): step 1 from
        # max(1, int(...)) of a tiny chunk_s and of one under a sample, and 7
        (bf, Q_TEST, 1500, dict(spec=TINY, chunk_s=1e-9, pre_s=0.5, post_s=-52.75)),
        (bf, Q_TEST, 1500.3, dict(spec=TINY, chunk_s=0.000123, pre_s=0.5, post_s=-52.75)),
        (sp, Q_TEST, 2650, dict(spec=TINY, segment=1, chunk_s=0.000875, pre_s=0.5, post_s=-50.0)),
        (bf, Q_TEST, 1500, dict(spec=TINY, post_s=-(ref.slot_samples(TINY) / 8000) + 3.0)),
        (src["fsk"], Q_TEST, 2200, dict(spec=LTINY, chunk_s=0.5, amplitude=0.25, post_s=1.0)),
        (src["fsk"], Q_TEST, "1500.3", dict(spec=LTINY, ook=0, chunk_s=1e9)),
    ]
    for s, q, car, kw in cases:
        what = (q, car, kw.get("chunk_s"), kw.get("pre_s"))
        want = ref.transmit_audio(s, q, car, **kw)
        got = nat.transmit_audio(s, q, car, **kw)
        is_l = kw["spec"].waveform == WAVEFORM_L
        _check(got, want, L_TOL if is_l else CE_TOL, EXACT_CE and not is_l, what)


def test_refusals(nat, ad, src):
    """Every refusal: the same exception type and text, from transmit_audio,
    the --native-qrss adapter and the PassAudio constructor alike."""
    sp, h, bf, fsk = src["sp"], src["h"], src["ook"], src["fsk"]
    nan, inf = float("nan"), float("inf")
    cases = [
        ((sp, Q_TEST, 1500), dict(spec=LSHORT, header=h, segment=1, ook=True)),     # L + OOK
        ((sp, Q_TEST, 1500), dict(spec=LSHORT, header=h, segment=1, ook="yes")),
        ((bf, Q_TEST, 1500), dict(spec=LSHORT)),                                      # OOK file on L
        ((bf, Q_TEST, 1500), dict(spec=LSHORT, ook=False)),
        ((sp, Q_TEST, 310), dict(spec=LSHORT, header=h, segment=1)),                  # check_pilot
        ((sp, Q_TEST, 322.0), dict(spec=LSHORT, header=h, segment=1)),
        ((fsk, Q_TEST, 2800), dict(spec=LTINY)),
        ((sp, Q_TEST, 2800), dict(spec=SHORT, header=h, segment=1)),                  # check_carrier
        ((sp, Q_TEST, 299.99), dict(spec=SHORT, header=h, segment=1)),
        ((sp, Q_TEST, nan), dict(spec=SHORT, header=h, segment=1)),
        ((sp, Q_TEST, inf), dict(spec=TINY, header=h, segment=1)),
        ((sp, Q_TEST, "abc"), dict(spec=TINY, header=h, segment=1)),
        ((sp, Q_TEST, None), dict(spec=TINY, header=h, segment=1)),
        ((sp, Q_TEST), dict(spec=SHORT, segment=1)),                                  # no callsign
        ((sp, Q_TEST), dict(spec=LSHORT, segment=1)),
        ((sp, Q_TEST), dict(spec=SHORT, header=h, segment=1, pre_s=5.0, lead_in_s=10.0)),
        ((sp, Q_TEST), dict(spec=LSHORT, header=h, segment=1, pre_s=0.5)),            # pre_s cover
        ((sp, Q_TEST), dict(spec=SHORT, header=h, segment=1, pre_s=12.00001)),        # not whole
        ((fsk, Q_TEST), dict(spec=LTINY, pre_s=12.00001)),
        ((sp, Q_TEST), dict(spec=SHORT, header=h, segment=1, pre_s=nan)),
        ((sp, Q_TEST), dict(spec=SHORT, header=h, segment=1, pre_s=inf)),
        ((sp, Q_TEST), dict(spec=SHORT, header=h, segment=1, lead_in_s=11.0)),        # lead-in
        ((sp, Q_TEST), dict(spec=LSHORT, header=h, segment=1, lead_in_s=-1.0)),
        ((sp, Q_TEST), dict(spec=TINY, header=h, segment=1, lead_in_s=nan)),
        ((bf, Q_TEST), dict(spec=TINY, post_s=-100.0)),                               # negative length
        ((fsk, Q_TEST), dict(spec=LTINY, post_s=-100.0)),
        ((bf, Q_TEST), dict(spec=TINY, post_s=nan)),
        ((bf, Q_TEST), dict(spec=TINY, chunk_s=nan)),
        ((bf, Q_TEST), dict(spec=TINY, chunk_s=inf)),
        ((fsk, Q_TEST), dict(spec=LTINY, chunk_s=nan)),
        ((bf, -1), dict(spec=TINY)),                                                  # q
        ((bf, 1 << 64), dict(spec=TINY)),
        ((bf, 1.5), dict(spec=TINY)),
        ((bf, Q_TEST), dict(spec=TINY, segment=0)),                                   # segments
        ((sp, Q_TEST), dict(spec=TINY, header=h)),
        ((sp, Q_TEST), dict(spec=TINY, segment=5, header=h)),
        ((sp, Q_TEST), dict(spec=SHORT, segment=0, header=h)),
        ((None, Q_TEST), dict(spec=TINY)),                                            # source
        ((sp, Q_TEST), dict(spec=TINY, segment=1)),                                   # tiny: no windows
    ]
    for args, kw in cases:
        want = _same_refusal(ref.transmit_audio, nat.transmit_audio, *args, **kw)
        _same_refusal(ref.transmit_audio, ad["transmit_audio"], *args, **kw)
        got = _outcome(nat.PassAudio, *args, **kw)
        assert got[0] == want[0], (args, kw, want, got)
        if want[0] == "raise":
            assert got[1:] == want[1:], (args, kw, want, got)
        else:
            assert len(got[1]) == len(want[1])


# --- PassAudio: streaming ----------------------------------------------------------------

def _read_all(p, size, dtype="float64"):
    parts = []
    while True:
        b = p.read(size, dtype)
        if not len(b):
            break
        assert len(b) == min(size, len(p) - p.position + len(b))
        parts.append(b)
    return np.concatenate(parts) if parts else np.zeros(0, dtype)


@pytest.mark.parametrize("spec", [TINY, LTINY], ids=lambda s: s.name)
def test_pass_audio_blocks(nat, src, spec):
    """Blocks of 1, 7, 4096 and 1e5 samples and the whole: the one-shot audio,
    bit for bit (CE and L alike, whatever the block size)."""
    sp, h = src["sp"], src["h"]
    kw = dict(spec=spec, segment=1, header=h, lead_in_s=5.0)
    whole = nat.transmit_audio(sp, Q_TEST, 1500.3, **kw)
    p = nat.PassAudio(sp, Q_TEST, 1500.3, **kw)
    assert len(p) == p.size == len(whole) == ref.slot_samples(spec)
    assert p.n_pre == 96000 and p.position == 0 and p.waveform_l == (spec.waveform == WAVEFORM_L)
    assert p.carrier == (15003, 10)       # Fraction(1500.3).limit_denominator(2**20)
    for size in (1, 7, 4096, 100_000, len(whole)):
        p.seek(0)
        got = _read_all(p, size)
        assert got.tobytes() == whole.tobytes(), size
        assert p.position == len(p) and len(p.read(5)) == 0


def test_pass_audio_short(nat, src):
    """A short CE (OOK) and l-short pass streamed in 4096 and 1e5 blocks
    against the reference, plus render at chunk boundaries and float32."""
    sp, h = src["sp"], src["h"]
    for spec, kw in ((SHORT, dict(ook=True, chunk_s=7.3)), (LSHORT, dict(chunk_s=7.3))):
        kw = dict(spec=spec, segment=1, header=h, lead_in_s=5.0, **kw)
        want = ref.transmit_audio(sp, Q_TEST, 1500.3, **kw)
        p = nat.PassAudio(sp, Q_TEST, 1500.3, **kw)
        assert p.chunk == 58400
        is_l = spec.waveform == WAVEFORM_L
        exact = EXACT_CE and not is_l
        for size in (4096, 100_000):
            p.seek(0)
            _check(_read_all(p, size), want, L_TOL if is_l else CE_TOL, exact, (spec.name, size))
        whole = p.render(0, len(p))
        for i0 in (0, 58399, 58400, 96000 - 3, 262143, 262144, 1_000_001, len(p) - 9):
            m = min(17, len(p) - i0)
            assert p.render(i0, m).tobytes() == whole[i0:i0 + m].tobytes(), i0
            # one sample at a time across the boundary
            ones = np.array([p.render(i, 1)[0] for i in range(i0, i0 + m)])
            assert ones.tobytes() == whole[i0:i0 + m].tobytes(), i0
        f32 = p.render(0, len(p), "float32")
        assert f32.dtype == np.float32 and f32.tobytes() == whole.astype(np.float32).tobytes()
        p.seek(10)
        assert _read_all(p, 99_999, np.float32).tobytes() == f32[10:].tobytes()


def test_pass_audio_bounds(nat, src):
    p = nat.PassAudio(src["ook"], Q_TEST, spec=TINY)
    n = len(p)
    assert len(p.render(n, 0)) == 0 and len(p.render(0, 0)) == 0
    for i0, m in ((-1, 1), (0, n + 1), (n, 1), (n - 1, 2)):
        with pytest.raises(IndexError):
            p.render(i0, m)
    with pytest.raises(ValueError):
        p.render(0, -1)
    with pytest.raises(ValueError):
        p.read(-1)
    for i in (-1, n + 1):
        with pytest.raises(IndexError):
            p.seek(i)
    p.seek(n)
    assert len(p.read(10)) == 0
    with pytest.raises(ValueError):
        p.render(0, 4, "int16")


def test_long_frame_closed_form(nat, src):
    """A frame longer than 2048 s, chunked so one chunk starts off the 16 kHz
    grid (more than 16384005 samples from t0): the reference's closed-form
    path, which no preset reaches."""
    sp, h = src["sp"], src["h"]
    spec = frame.FrameSpec("long", n_data=50600, cw_after=tuple(range(3500, 57000, 1500)))
    step = 16480005                      # chunk 2 starts at t0 + 16384005 samples
    kw = dict(spec=spec, segment=1, header=h, chunk_s=step / 8000)
    want = ref.transmit_audio(sp, Q_TEST, 1500.3, **kw)
    got = nat.transmit_audio(sp, Q_TEST, 1500.3, **kw)
    _check(got, want, CE_TOL, EXACT_CE, "long")
    p = nat.PassAudio(sp, Q_TEST, 1500.3, **kw)
    for i0 in (step - 5, step + 262139, len(p) - 3000):
        assert p.render(i0, 3000).tobytes() == got[i0:i0 + 3000].tobytes(), i0


# --- WAV ---------------------------------------------------------------------------------

def _bytes(path):
    return Path(path).read_bytes()


def test_wav_bytes(nat, src, tmp_path):
    """qrss_transmit.py's two files (wavio over scipy) against the native
    writers, for an array and streamed from a PassAudio."""
    sp, h = src["sp"], src["h"]
    for spec, car in ((TINY, 1500.3), (LTINY, 1200)):
        kw = dict(spec=spec, segment=1, header=h, lead_in_s=5.0)
        x = ref.transmit_audio(sp, Q_TEST, car, **kw)
        p = nat.PassAudio(sp, Q_TEST, car, **kw)
        wavio.write_wav(tmp_path / "ref.wav", x)
        wavio.write_wav_float(tmp_path / "ref_f.wav", x)
        y = p.render(0, len(p))
        nat.write_wav(tmp_path / "a.wav", y)
        nat.write_wav_float(str(tmp_path / "a_f.wav"), y)
        p.write_wav(tmp_path / "p.wav")
        p.write_wav(str(tmp_path / "p_f.wav"), float=True)
        wavio.write_wav(tmp_path / "ref_y.wav", y)
        wavio.write_wav_float(tmp_path / "ref_yf.wav", y)
        for a, b in (("a.wav", "ref_y.wav"), ("p.wav", "ref_y.wav"),
                     ("a_f.wav", "ref_yf.wav"), ("p_f.wav", "ref_yf.wav")):
            assert _bytes(tmp_path / a) == _bytes(tmp_path / b), (spec.name, a)
        if EXACT_CE and spec.waveform != WAVEFORM_L:
            assert _bytes(tmp_path / "p.wav") == _bytes(tmp_path / "ref.wav")
            assert _bytes(tmp_path / "p_f.wav") == _bytes(tmp_path / "ref_f.wav")
        # the header the CLI's file is read back by
        r = wavio.read_wav(str(tmp_path / "p_f.wav"))
        assert r.astype(np.float32).tobytes() == y.astype(np.float32).tobytes()


def test_wav_arrays(nat, tmp_path):
    """Edge arrays: silence (no scaling), a peak other than 0.95, values on
    rounding ties, a lone sample, odd lengths; empty is refused alike for
    int16 and written for float32."""
    rng = np.random.default_rng(5)
    ties = (np.arange(-40, 41) + 0.5) / 32767
    arrays = [np.zeros(9), np.array([0.25]), rng.standard_normal(10_001) * 3, ties,
              np.array([-1.0, 1.0, -0.5, 0.5, -0.0]), np.linspace(-2, 2, 4097)]
    for i, x in enumerate(arrays):
        for peak in (0.95, 0.5, 1.0):
            wavio.write_wav(tmp_path / "r.wav", x, peak)
            nat.write_wav(tmp_path / "n.wav", x, peak)
            assert _bytes(tmp_path / "r.wav") == _bytes(tmp_path / "n.wav"), (i, peak)
        wavio.write_wav(tmp_path / "r.wav", x)
        nat.write_wav(tmp_path / "n.wav", x)
        assert _bytes(tmp_path / "r.wav") == _bytes(tmp_path / "n.wav"), i
        wavio.write_wav_float(tmp_path / "r.wav", x)
        nat.write_wav_float(tmp_path / "n.wav", x)
        assert _bytes(tmp_path / "r.wav") == _bytes(tmp_path / "n.wav"), i
    empty = np.zeros(0)
    want = _outcome(wavio.write_wav, tmp_path / "e1.wav", empty)
    got = _outcome(nat.write_wav, tmp_path / "e2.wav", empty)
    assert want[0] == got[0] == "raise" and want[1:] == got[1:]
    assert not (tmp_path / "e1.wav").exists() and not (tmp_path / "e2.wav").exists()
    wavio.write_wav_float(tmp_path / "e1.wav", empty)
    nat.write_wav_float(tmp_path / "e2.wav", empty)
    assert _bytes(tmp_path / "e1.wav") == _bytes(tmp_path / "e2.wav")
    # a directory that is not there: the same OSError
    bad = tmp_path / "missing" / "x.wav"
    want = _outcome(wavio.write_wav, bad, arrays[1])
    got = _outcome(nat.write_wav, bad, arrays[1])
    assert want[0] == got[0] == "raise" and want[1:] == got[1:], (want, got)
    got = _outcome(nat.write_wav_float, str(bad), arrays[1])
    assert got[1:] == want[1:], (want, got)

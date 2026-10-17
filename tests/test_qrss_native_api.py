"""The in-process QRSSTVAE transmit API (native/core/qrss/api.hpp) against
qrss_transmit.py and qrss_encode.py.

The API is what the desktop app sends with, in place of running the two
scripts; `sstvae-qrss transmit` is built on it. It is not a port of a
Python module, so the reference here is the scripts themselves:

- qrss_transmit.py is run in-process (its main(), with sys.argv set) on
  the same input file and arguments as api.problem / api.pass_audio, for
  a .qrsp and for beacon files keyed FSK and OOK. A refusal must be the
  script's SystemExit text, in its order; a pass's audio must be the
  samples of the script's --float WAV (float32): bit for bit for CE where
  numpy's trig and complex multiply are the C++'s (EXACT_CE), else within
  1e-12, and within 1e-9 for L, plus one float32 spacing either way. The
  source may be the file's path or the dataclass read from it, with the
  same result.
  A slot before 1970 is the exception: the script's --slot parser
  refuses it (sequences.parse_slot) before it looks at anything else,
  while the API takes q as a number. Its order is the script's from
  before that parser refused, every other refusal first and q last, in
  the words of sequences.scrambler, which the script reached with it.
- The constants (PASS_S, AUDIO_LEAD_S, AUDIO_TAIL_S, T0_OFFSET_S, the
  carrier band, ...) are compared with where the reference keeps them,
  and the slot helpers with sequences/frontend/tx.
- Under the `codec` marker, end to end: a 640x480 picture encoded through
  api.open_encoder is the reference's encode of the same pixels as a PNG,
  bit for bit; a SHORT pass made from it by api.pass_audio, through the
  channel simulator at -6 dB, is received by Python's receiver with the
  callsign, the grid and that picture ID in its header. SHORT and not
  TINY, because a TINY frame carries no header and so no picture ID.
"""

import contextlib
import io
import runpy
import sys
from pathlib import Path

import numpy as np
import pytest

from qrss_helpers import Q_TEST, synthetic_full_latents
from qrss_native import require
from sstvae import wavio
from sstvae.qrss import beaconfile, frame, frontend, lframe, picture, sequences
from sstvae.qrss import tx as ref_tx
from sstvae.qrss.constants import CARRIER_BAND_HZ, LEAD_IN_MAX_S
from sstvae.qrss.header import HeaderFields
from test_qrss_native_cli import decimal_comma_locale  # noqa: F401 (a fixture)
from test_qrss_native_tx_audio import EXACT_CE

ROOT = Path(__file__).resolve().parent.parent
CALL = "K1ABC/P"


@pytest.fixture(scope="module")
def api(native):
    return require(native, "api")


@pytest.fixture(scope="module")
def script():
    """qrss_transmit.py's main, loaded once."""
    return runpy.run_path(str(ROOT / "qrss_transmit.py"), run_name="qrss_transmit_under_test")["main"]


@pytest.fixture(scope="module")
def inputs(tmp_path_factory):
    """A mode-B .qrsp and its segment-1 beacon files, FSK and OOK."""
    d = tmp_path_factory.mktemp("api-in")
    sp = picture.StoredPicture.from_latents(synthetic_full_latents(7), 0xD1D8, 1)
    picture.save_qrsp(d / "pic.qrsp", sp)
    h = HeaderFields(callsign=CALL, grid="FN42", picture_id=sp.picture_id, mode=1, segment=1, codec_id=0xD1D8)
    beaconfile.write(d / "fsk.bin", beaconfile.from_picture(sp, 1, h, ook=False))
    beaconfile.write(d / "ook.bin", beaconfile.from_picture(sp, 1, h, ook=True))
    return {k: str(d / f) for k, f in (("qrsp", "pic.qrsp"), ("fsk", "fsk.bin"), ("ook", "ook.bin"))}


def _argv(src: str, out: str, q: int, kw: dict) -> list[str]:
    """qrss_transmit.py's command line for api's keywords."""
    argv = [src, out, "--slot", _iso(q), "--float"]
    for key, flag in (("waveform", "--waveform"), ("callsign", "--callsign"), ("grid", "--grid"),
                      ("frame", "--frame")):
        if kw.get(key) is not None:
            argv += [flag, kw[key]]
    if "segment" in kw:
        argv += ["--segment", str(kw["segment"])]
    if "freq_hz" in kw:
        argv += ["--freq", repr(float(kw["freq_hz"]))]
    if "lead_in_s" in kw:
        argv += ["--lead-in", repr(float(kw["lead_in_s"]))]
    if kw.get("ook"):
        argv.append("--ook")
    return argv


def _iso(q: int) -> str:
    """Slot q as --slot text, e.g. 2025-10-01T00:00Z."""
    import datetime as dt
    return (dt.datetime(1970, 1, 1, tzinfo=dt.timezone.utc)
            + dt.timedelta(seconds=900 * q)).strftime("%Y-%m-%dT%H:%MZ")


def _codec_api(api):
    if not hasattr(api, "open_encoder"):
        pytest.skip("sstvae_native built without the codec")
    return api


def _run_script(script, argv: list[str]):
    """("ok", float32 samples, stdout) or ("refused", text) from qrss_transmit.py."""
    out = Path(argv[1])
    out.unlink(missing_ok=True)
    saved = sys.argv
    sys.argv = ["qrss_transmit.py", *argv]
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            script()
    except SystemExit as e:
        assert not out.exists(), "the script wrote a file for a refusal"
        return ("refused", str(e.code))
    finally:
        sys.argv = saved
    x = wavio.read_wav(str(out))
    return ("ok", np.asarray(x, dtype=np.float32), buf.getvalue())


def _same_audio(waveform: str, got: np.ndarray, want: np.ndarray):
    assert got.dtype == np.float32 and got.shape == want.shape
    if waveform == "CE" and EXACT_CE:
        assert got.tobytes() == want.tobytes()
        return
    tol = (1e-12 if waveform == "CE" else 1e-9) + np.spacing(np.abs(want)).astype(np.float64)
    assert np.all(np.abs(got.astype(np.float64) - want) <= tol)


# Each case: (input, q, the request's keywords). TINY frames unless named,
# so that each pass is ~68 s of audio.
CASES = [
    ("qrsp", Q_TEST, dict(callsign="K1ABC", frame="tiny")),
    ("qrsp", Q_TEST, dict(callsign=CALL, grid="FN42", segment=1, frame="tiny", freq_hz=1234.5, ook=True)),
    ("qrsp", Q_TEST + 3, dict(callsign="K1ABC", frame="tiny", waveform="L", lead_in_s=2.5, freq_hz=2650)),
    ("qrsp", Q_TEST, dict(callsign="K1ABC", frame="tiny", lead_in_s=10.0, freq_hz=300.0)),
    ("qrsp", Q_TEST, dict(callsign="K1ABC", frame="short", waveform="L", grid="FN42")),
    ("fsk", Q_TEST, dict(frame="tiny")),
    ("fsk", Q_TEST, dict(frame="tiny", callsign=CALL, grid="XX99", segment=9)),   # not used for a .bin
    ("ook", Q_TEST, dict(frame="tiny", ook=True, waveform="CE")),
    ("fsk", Q_TEST, dict(frame="tiny", waveform="L", freq_hz=1800.5)),
    # Refusals, in the script's order.
    ("ook", Q_TEST, dict(frame="tiny", waveform="L")),
    ("qrsp", Q_TEST, dict(frame="tiny")),
    ("qrsp", Q_TEST, dict(frame="tiny", waveform="L", ook=True)),
    ("qrsp", Q_TEST, dict(frame="tiny", waveform="L", ook=True, callsign="K1ABC", freq_hz=100)),
    ("fsk", Q_TEST, dict(frame="tiny", callsign="K1ABC")),
    ("fsk", Q_TEST, dict(frame="tiny", ook=True)),
    ("fsk", Q_TEST, dict(frame="tiny", ook=True, callsign="W1AW")),
    ("qrsp", Q_TEST, dict(callsign="K1ABC", frame="tiny", segment=2)),
    ("qrsp", Q_TEST, dict(callsign="K1ABC", frame="tiny", segment=-1)),
    ("qrsp", Q_TEST, dict(callsign="K1ABC", frame="tiny", freq_hz=100)),
    ("qrsp", Q_TEST, dict(callsign="K1ABC", frame="tiny", freq_hz=2700.5)),
    ("qrsp", Q_TEST, dict(callsign="K1ABC", frame="tiny", waveform="L", freq_hz=310)),
    ("qrsp", Q_TEST, dict(callsign="K1ABC!", frame="tiny")),
    ("qrsp", Q_TEST, dict(callsign="", frame="tiny")),
    ("qrsp", Q_TEST, dict(callsign="K1ABC", grid="XX", frame="tiny")),
    ("qrsp", Q_TEST, dict(callsign="K1ABC", frame="tiny", lead_in_s=11)),
    ("qrsp", Q_TEST, dict(callsign="K1ABC", frame="tiny", lead_in_s=-0.5)),
    ("qrsp", Q_TEST, dict(callsign="K1ABC", frame="tiny", segment=5, freq_hz=100)),
    ("qrsp", -1, dict(callsign="K1ABC", frame="tiny")),
    ("qrsp", -1, dict(callsign="K1ABC", frame="tiny", freq_hz=100)),
    ("fsk", -1, dict(frame="tiny")),
]


def _id(case):
    src, q, kw = case
    return f"{src}-q{q - Q_TEST if q > 0 else q}-" + "-".join(f"{k}={v}" for k, v in kw.items())


@pytest.fixture(scope="module")
def outcomes(script, inputs, tmp_path_factory):
    """The script's outcome for every case, run once, and for a slot
    before 1970 also for the same request at Q_TEST (see `_want`)."""
    d = tmp_path_factory.mktemp("api-out")
    cases = CASES + [(s, Q_TEST, kw) for s, q, kw in CASES if q < 0]
    return {_id(c): _run_script(script, _argv(inputs[c[0]], str(d / "out.wav"), c[1], c[2])) for c in cases}


def _want(outcomes, case):
    """What the API must say for a case: the script's outcome, except that
    for q < 0 the script's --slot parser refuses the slot before anything
    else is looked at (`test_the_script_refuses_a_slot_before_1970`). The
    API keeps the order the script had before that: the request's own
    refusal if it has one, else sequences.scrambler's for the q."""
    src, q, kw = case
    if q >= 0:
        return outcomes[_id(case)]
    at = outcomes[_id((src, Q_TEST, kw))]
    if at[0] == "refused":
        return at
    with pytest.raises(ValueError) as e:
        sequences.scrambler(q, 1)
    return ("refused", str(e.value))


def test_the_script_refuses_a_slot_before_1970(outcomes):
    for case in CASES:
        if case[1] < 0:
            iso = _iso(case[1])
            assert outcomes[_id(case)] == ("refused", f"--slot {iso!r}: slot {iso} is before 1970-01-01T00:00Z")


def _sources(inputs, name):
    """The input as a path and as the dataclass the script reads from it."""
    path = inputs[name]
    obj = beaconfile.read(path) if name != "qrsp" else picture.load_qrsp(path)
    return [path, Path(path), obj]


@pytest.mark.parametrize("case", CASES, ids=_id)
def test_problem_and_audio_match_the_script(api, inputs, outcomes, case):
    src_name, q, kw = case
    want = _want(outcomes, case)
    for src in _sources(inputs, src_name):
        text = api.problem(src, q, **kw)
        if want[0] == "refused":
            assert text == want[1]
            with pytest.raises(ValueError) as e:
                api.pass_audio(src, q, **kw)
            assert str(e.value) == want[1]
        else:
            assert text == ""
            _same_audio(kw.get("waveform", "CE"), api.pass_audio(src, q, **kw), want[1])


def test_the_cases_cover_each_ending(outcomes):
    """Every case reached the end it was written for (the refusals are distinct)."""
    wants = [_want(outcomes, c) for c in CASES]
    ok = [w for w in wants if w[0] == "ok"]
    refused = {w[1] for w in wants if w[0] == "refused"}
    assert len(ok) == 9
    assert len(refused) >= 15, refused
    assert "q must fit in uint64, not -1" in refused


def test_the_audio_is_transmit_audio(api, inputs):
    """pass_audio is tx.transmit_audio's float64, as float32 (no file in between)."""
    sp = picture.load_qrsp(inputs["qrsp"])
    h = HeaderFields(callsign="K1ABC", grid="FN42", picture_id=sp.picture_id, mode=1, segment=1, codec_id=0xD1D8)
    want = ref_tx.transmit_audio(sp, Q_TEST, 1777.25, spec=frame.TINY, segment=1, header=h, lead_in_s=1.5,
                                 ook=False)
    got = api.pass_audio(sp, Q_TEST, callsign="K1ABC", grid="FN42", segment=1, frame="tiny", freq_hz=1777.25,
                         lead_in_s=1.5)
    _same_audio("CE", got, want.astype(np.float32))
    assert len(got) == ref_tx.slot_samples(frame.TINY) == api.pass_samples("CE", "tiny")


def test_a_frame_the_script_does_not_offer_is_refused(api, inputs):
    want = "frame must be one of full, medium, short, tiny, not 'huge'"
    assert api.problem(inputs["qrsp"], Q_TEST, callsign="K1ABC", frame="huge") == want
    assert api.problem(inputs["qrsp"], Q_TEST, callsign="K1ABC", frame="l-tiny") == \
        "frame must be one of full, medium, short, tiny, not 'l-tiny'"
    with pytest.raises(ValueError, match="frame must be one of"):
        api.pass_audio(inputs["fsk"], Q_TEST, frame="huge")
    with pytest.raises(ValueError, match="waveform must be 'CE' or 'L', not 'X'"):
        api.problem(inputs["fsk"], Q_TEST, waveform="X")


# --- constants and slots -------------------------------------------------------------------

def test_constants_are_the_references(api):
    import inspect

    params = inspect.signature(ref_tx.transmit_audio).parameters
    assert api.SAMPLE_RATE == frontend.FS == 8000
    assert api.SLOT_S == frontend.slot_t0(1) - frontend.slot_t0(0) == 900
    assert api.T0_OFFSET_S == frontend.slot_t0(0) == 1
    assert api.AUDIO_LEAD_S == params["pre_s"].default == 12.0
    assert api.AUDIO_TAIL_S == params["post_s"].default == 3.0
    assert api.CARRIER_DEFAULT_HZ == params["carrier_hz"].default == 1500
    assert api.PASS_S == frame.FULL.duration_s == lframe.get("l-full").duration_s
    assert (api.CARRIER_MIN_HZ, api.CARRIER_MAX_HZ) == tuple(CARRIER_BAND_HZ)
    assert api.LEAD_IN_MAX_S == LEAD_IN_MAX_S
    assert api.FRAMES == tuple(sorted(frame.PRESETS))
    for name in api.FRAMES:
        assert api.frame_name("CE", name) == frame.get(name).name
        assert api.frame_name("L", name) == lframe.get(f"l-{name}").name
    assert [api.mode_index(m) for m in picture.MODE_NAMES] == [0, 1, 2]
    assert api.mode_index("D") is None and api.mode_index("b") is None
    assert [api.passes_for(m) for m in (*picture.MODE_NAMES, "Z")] == [1, 2, 3, 0]


@pytest.mark.parametrize("frame_name", ["full", "medium", "short", "tiny"])
def test_pass_length(api, frame_name):
    for wf, spec in (("CE", frame.get(frame_name)), ("L", lframe.get(f"l-{frame_name}"))):
        n = ref_tx.slot_samples(spec)
        assert api.pass_samples(wf, frame_name) == n
        assert api.audio_start_s(Q_TEST) == frontend.slot_t0(Q_TEST) - 12.0
        assert api.audio_end_s(Q_TEST, wf, frame_name) == frontend.slot_t0(Q_TEST) - 12.0 + n / 8000


@pytest.mark.parametrize("text", ["2025-10-01T00:00Z", "2026-10-09T06:15:00+00:00", "2026-10-09T08:45+02:00",
                                  "2025-10-01T00:05Z", "nope", "2025-10-01T00:00", "1969-12-31T23:45Z",
                                  "1970-01-01T00:00Z"])
def test_parse_slot_is_sequences(api, text):
    try:
        want = ("ok", sequences.parse_slot(text))
    except ValueError as e:
        want = ("refused", str(e))
    try:
        got = ("ok", api.parse_slot(text))
    except ValueError as e:
        got = ("refused", str(e))
    assert got == want
    if want[0] == "ok":
        assert api.slot_iso(want[1]) == _iso(want[1])
        assert api.slot_of(900.0 * want[1]) == want[1]


def test_slot_of_refuses_what_is_not_a_quarter_hour(api):
    for t in (900.5, 450.0, float("inf"), float("nan"), 1e30):
        with pytest.raises(ValueError, match="QRSS passes start on a quarter hour"):
            api.slot_of(t)
    assert api.slot_of(0.0) == 0 and api.slot_of(-900.0) == -1


def test_refusals_ignore_a_decimal_comma_locale(api, inputs, outcomes, decimal_comma_locale, monkeypatch):
    """The app runs under QApplication, whose setlocale(LC_ALL, "") gives
    printf the environment's decimal point; the script's texts never use
    the locale. Before the fix an L pilot refusal read "287,9 Hz" in de_DE."""
    import locale

    monkeypatch.setenv("LOCPATH", str(decimal_comma_locale))
    old = locale.setlocale(locale.LC_NUMERIC)
    try:
        locale.setlocale(locale.LC_NUMERIC, "de_DE.UTF-8")
        assert locale.localeconv()["decimal_point"] == ","
        sp = picture.load_qrsp(inputs["qrsp"])
        assert api.problem(sp, Q_TEST, callsign="K1ABC", frame="tiny", waveform="L", freq_hz=310.0) == (
            "an L pilot at 310.0 Hz puts the data down to 287.9 Hz, below the 300 Hz receivers search")
        for case in CASES:
            src_name, q, kw = case
            want = _want(outcomes, case)
            if want[0] == "refused":
                assert api.problem(inputs[src_name], q, **kw) == want[1], case
    finally:
        locale.setlocale(locale.LC_NUMERIC, old)


def test_read_source_says_why_a_file_cannot_be_read(api, tmp_path):
    """open(path, "rb")'s OSError text, as the script's traceback ends."""
    for path in (tmp_path / "missing.qrsp", tmp_path, tmp_path / "it's here.bin"):
        with pytest.raises(OSError) as want:
            open(path, "rb").read()
        with pytest.raises(RuntimeError) as got:
            api.read_source(str(path))
        assert str(got.value) == str(want.value)


def test_read_source_is_the_scripts(api, inputs):
    sp = picture.load_qrsp(inputs["qrsp"])
    got = api.read_source(inputs["qrsp"])
    assert (got["mode"], got["codec_id"], got["picture_id"]) == (sp.mode, sp.codec_id, sp.picture_id)
    for a, b in zip(got["segs"], sp.segs):
        assert np.asarray(a).tobytes() == np.asarray(b).tobytes()
    bf = beaconfile.read(inputs["ook"])
    got = api.read_source(inputs["ook"])
    assert (got["callsign"], got["grid"], got["ook"], got["segment"]) == (bf.callsign, bf.grid, True, 1)


# --- the encode, and a pass to Python's receiver ----------------------------------------------

def _same_picture(got: dict, want: picture.StoredPicture):
    assert (got["mode"], got["codec_id"], got["picture_id"]) == (want.mode, want.codec_id, want.picture_id)
    assert len(got["segs"]) == len(want.segs)
    for a, b in zip(got["segs"], want.segs):
        assert np.asarray(a).dtype == np.float16 and np.asarray(a).tobytes() == np.asarray(b).tobytes()


def test_encode_picture_with_a_python_encoder(api):
    """The fit and the scaling around the encoder: a 640x480 picture reaches
    the encoder as images.image_to_array gives it, and its latents become
    from_latents' picture."""
    from qrss_codec_fixtures import scene
    from sstvae import images

    rgb = scene()
    seen = []
    lat = synthetic_full_latents(3)

    def enc(img):
        seen.append(np.array(img))
        return lat

    got = api.encode_picture(rgb, 2, 0x1234, enc)
    _same_picture(got, picture.StoredPicture.from_latents(lat, 0x1234, 2))
    from PIL import Image
    want = images.image_to_array(Image.fromarray(rgb))
    assert seen[0].dtype == np.float32 and seen[0].shape == (3, 480, 640)
    assert np.array_equal(seen[0], want)
    with pytest.raises(ValueError):
        api.encode_picture(rgb, 3, 0x1234, enc)
    with pytest.raises(ValueError):
        api.encode_picture(rgb, 1, 1 << 16, enc)
    assert len(seen) == 1, "a refusal must come before the encode"


@pytest.mark.codec
def test_api_picture_reaches_pythons_receiver(api, tmp_path):
    import qrss_codec_fixtures as fixtures
    from PIL import Image

    from qrss_codec_fixtures import scene
    from sstvae import codec, images
    from sstvae.qrss import channel as chm
    from sstvae.qrss import receiver as RX
    from sstvae.qrss.channel import ChannelConfig

    api = _codec_api(api)
    v5 = str(fixtures.v5_dir())
    rgb = scene(seed=21)
    png = tmp_path / "scene.png"
    Image.fromarray(rgb).save(png)

    enc = api.open_encoder(v5)
    assert enc.codec_id == 0xD1D8 and enc.onnx_path.endswith("-encoder-fp16.onnx")
    got = enc.encode_picture(rgb, 1)
    want = picture.StoredPicture.from_latents(
        codec.load_codec(v5, precision="fp16").encode(images.load_image(str(png))), 0xD1D8, 1)
    _same_picture(got, want)
    _same_picture(enc.encode_file(str(png), 1), want)
    sp = picture.StoredPicture(**got)

    assert api.problem(sp, Q_TEST, callsign="K1ABC", grid="FN42", frame="short") == ""
    audio = api.pass_audio(sp, Q_TEST, callsign="K1ABC", grid="FN42", frame="short")
    sim = chm.simulate_audio(audio.astype(np.float64), frame.SHORT, Q_TEST, ChannelConfig(snr_db=-6.0, seed=5))
    passes = RX.receive_slot(frontend.Capture(sim.fe, Q_TEST, sim.t0_index), frame.SHORT)
    assert len(passes) == 1, [p.f_hz for p in passes]
    assert passes[0].header == HeaderFields(callsign="K1ABC", grid="FN42", picture_id=want.picture_id, mode=1,
                                            segment=0, codec_id=0xD1D8)


def test_open_encoder_refusals(api, tmp_path):
    api = _codec_api(api)
    nowhere = str(tmp_path / "nowhere")
    with pytest.raises(SystemExit) as e:
        api.open_encoder(nowhere)
    assert str(e.value.code) == (f"--model {nowhere}: expected a .pt checkpoint, a .onnx artifact, or a "
                                 "directory containing the exported .onnx files")

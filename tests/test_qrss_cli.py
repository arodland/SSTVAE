"""QRSSTVAE command-line chain (design 10.7 E3) and `qrss_decode.py`.

E3 runs the CLIs as subprocesses on a SHORT frame: `qrss_encode.py`
(the v5 codec when its models are present, else a synthetic `.qrsp`
written directly) -> `qrss_beacon.py` -> `qrss_transmit.py` ->
`qrss_simulate.py --snr -10` -> `qrss_receive.py --store`; the store
then holds one pass with the decoded header, and `qrss_decode.py`
lists it (and renders it when the codec is there). E3 takes ~20 s and is
`slow`, as is `--retro` (a SHORT template search); the default run keeps
the argument and error handling and an accumulator render, and
`test_qrss_smoke.py` runs the chain in-process on a TINY slot.
"""

import subprocess
import sys

import numpy as np
import pytest

from qrss_fakes import make_header, make_pass_result, synthetic_picture
from qrss_helpers import REPO_ROOT, load_codec, model_dir
from sstvae.qrss import picture
from sstvae.qrss.store import Store

SLOT = "2025-10-01T00:00Z"
E3_EXTRA: list = []                     # accumulator sizes seen by the E3 run
CALL = "K1ABC"


def run(*args, ok=True, timeout=300):
    r = subprocess.run([sys.executable, *map(str, args)], cwd=REPO_ROOT, capture_output=True,
                       text=True, timeout=timeout)
    if ok and r.returncode != 0:
        raise AssertionError(f"{args[0]} failed ({r.returncode}):\n{r.stdout}\n{r.stderr}")
    return r


def model_args() -> list:
    d = model_dir()
    return ["--model", d] if d else []


def _have_codec() -> bool:
    try:
        from qrss_helpers import resolve_model
        resolve_model("encoder", "fp16")
        resolve_model("decoder", "fp16")
        return True
    except BaseException:                       # a skip is an exception too
        return False


@pytest.mark.slow
def test_e3_cli_chain_short(tmp_path):
    """E3: encode -> beacon -> transmit -> simulate -10 dB -> receive -> store -> decode."""
    qrsp = tmp_path / "pic.qrsp"
    codec = _have_codec()
    if codec:
        img = REPO_ROOT / "wonder_wheel.jpg"
        r = run("qrss_encode.py", img, qrsp, *model_args(), "--precision", "fp16")
        assert "picture ID" in r.stdout
    else:
        picture.save_qrsp(qrsp, synthetic_picture(0))
    sp = picture.load_qrsp(qrsp)
    run("qrss_beacon.py", qrsp, tmp_path / "pic.bin", "--callsign", CALL, "--grid", "FN42")
    run("qrss_transmit.py", tmp_path / "pic.bin", tmp_path / "tx.wav", "--slot", SLOT,
        "--frame", "short")
    run("qrss_simulate.py", tmp_path / "tx.wav", tmp_path / "rx.npz", "--slot", SLOT,
        "--frame", "short", "--snr", "-10", "--seed", "3")
    store = tmp_path / "store"
    r = run("qrss_receive.py", tmp_path / "rx.npz", "--store", store)
    assert f"header     {CALL}" in r.stdout and "stored" in r.stdout
    st = Store(store)
    passes = [st.load_pass(u) for u in st.pass_uids()]
    with_hdr = [p for p in passes if p.header is not None]
    assert len(with_hdr) == 1
    p = with_hdr[0]
    assert p.header.callsign == CALL and p.cw is not None and p.cw.agrees
    assert p.header.picture_id == sp.picture_id and p.frame == "short"
    key = (CALL, sp.picture_id)
    assert st.open_keys() == [key]
    members = st.accumulator(key).uids
    assert p.uid in members
    # Open (WP6/WP7): with the wonder_wheel picture the receiver also returns
    # a second, spurious pass of this one transmitter (9.5 Hz off, +237 ppm,
    # Z_ref 8, kappa flagged suspect), and since it is a distorted copy of
    # the same signal its latents correlate and rule 3 attaches it: one
    # transmission counted twice. Recorded by test_e3_one_transmission_one_member.
    for o in passes:
        if o is not p:
            print(f"E3: extra pass {o.uid}: Z_ref {o.report.z_ref:.1f}, "
                  f"suspect {o.report.suspect}, member {o.uid in members}")
    E3_EXTRA.append(len(members))
    r = run("qrss_decode.py", "--list", "--store", store)
    assert f"{sp.picture_id:08x}" in r.stdout and p.uid in r.stdout
    if codec:
        png = tmp_path / "out.png"
        run("qrss_decode.py", png, "--key", f"{CALL}:{sp.picture_id:08x}", "--store", store,
            *model_args(), "--precision", "fp16")
        assert png.exists() and png.stat().st_size > 1000


@pytest.mark.slow
@pytest.mark.xfail(reason="open (WP6/WP7): a spurious second pass of one transmission is "
                          "returned by receive_slot and attached by correlation", strict=False)
def test_e3_one_transmission_one_member():
    """E3's single transmission gave the accumulator exactly one member."""
    if not E3_EXTRA:
        pytest.skip("runs after test_e3_cli_chain_short")
    assert E3_EXTRA[-1] == 1


def test_decode_cli_errors(tmp_path):
    """--key parsing, a missing picture, and a pass without a header are refused."""
    sys.path.insert(0, str(REPO_ROOT))
    import qrss_decode

    assert qrss_decode.parse_key("k1abc/p:00ff00ff") == ("K1ABC/P", 0x00FF00FF)
    with pytest.raises(SystemExit):
        qrss_decode.parse_key("K1ABC")
    with pytest.raises(SystemExit):
        qrss_decode.parse_key("K1ABC:xyz")
    r = run("qrss_decode.py", tmp_path / "x.png", "--key", "K1ABC:12345678", "--store",
            tmp_path / "s", ok=False)
    assert r.returncode != 0 and "no picture" in (r.stdout + r.stderr)
    sp = synthetic_picture(0)
    p = make_pass_result(picture.air_values(sp.segs[0], 0), 1.0)
    p.save(tmp_path / "p.npz")
    r = run("qrss_decode.py", tmp_path / "x.png", "--pass", tmp_path / "p.npz", ok=False)
    assert r.returncode != 0 and "no decoded header" in (r.stdout + r.stderr)
    r = run("qrss_decode.py", "--list", "--store", tmp_path / "s")
    assert "no pictures" in r.stdout


@pytest.mark.codec
def test_decode_cli_renders_an_accumulator(tmp_path):
    """qrss_decode.py --key renders the accumulator (synthetic passes at +5 dB)."""
    load_codec("fp16")
    sp = synthetic_picture(0)
    h = make_header(sp, callsign=CALL)
    a = picture.air_values(sp.segs[0], 0)
    st = Store(tmp_path / "s")
    for seed in range(2):
        p = make_pass_result(a, 10 ** 0.5 / 2, seed=seed, hdr=h, decoded=True)
        st.add_pass(p)
        st.attach((CALL, sp.picture_id), p.uid, 0, "header", mode=0, codec_id=sp.codec_id)
    png = tmp_path / "out.png"
    r = run("qrss_decode.py", png, "--key", f"{CALL}:{sp.picture_id:08x}", "--store",
            tmp_path / "s", "--em", "1", *model_args(), "--precision", "fp16")
    assert "EM: mean W" in r.stdout and "segment 0: effective SNR +5.0" in r.stdout
    from PIL import Image
    im = np.asarray(Image.open(png))
    assert im.shape == (480, 640, 3)
    # the single-pass route places a header-decoded pass
    p.save(tmp_path / "p.npz")
    run("qrss_decode.py", tmp_path / "one.png", "--pass", tmp_path / "p.npz",
        *model_args(), "--precision", "fp16")
    assert (tmp_path / "one.png").exists()


class _FakeOnnxCodec:
    """Just enough of `codec.OnnxCodec` for `render.decoder_codec_id`."""
    backend = "onnx"

    def __init__(self, sha):
        self._sources = {"decoder": sha} if sha else {}

    def _session(self, part):
        return None


def test_codec_id_is_checked_before_decoding():
    """Integration review regression: a picture of another codec is not decoded
    silently by the v5 decoder. A mismatch refuses (or, forced, warns); a
    decoder without a codec ID warns that nothing could be checked."""
    from sstvae.qrss import render

    v5 = _FakeOnnxCodec("d1d8" + "0" * 60)
    assert render.decoder_codec_id(v5) == 0xD1D8
    assert render.check_codec_id(v5, 0xD1D8) is None
    assert render.check_codec_id(v5, None) is None
    with pytest.raises(render.CodecMismatch, match="codec 1234.*codec d1d8"):
        render.check_codec_id(v5, 0x1234)
    assert "decoding anyway" in render.check_codec_id(v5, 0x1234, force=True)
    assert "cannot be checked" in render.check_codec_id(_FakeOnnxCodec(None), 0x1234)


@pytest.mark.slow
@pytest.mark.codec
def test_decode_cli_refuses_another_codec(tmp_path):
    """qrss_decode.py --key on an accumulator of codec 1234 with the v5 decoder:
    refused with the two IDs named, and decoded (with a warning) only on --any-codec."""
    load_codec("fp16")
    sp = synthetic_picture(0)
    h = make_header(sp, callsign=CALL)
    st = Store(tmp_path / "s")
    p = make_pass_result(picture.air_values(sp.segs[0], 0), 3.0, seed=0, hdr=h, decoded=True)
    st.add_pass(p)
    st.attach((CALL, sp.picture_id), p.uid, 0, "header", mode=0, codec_id=0x1234)
    png = tmp_path / "out.png"
    argv = ["qrss_decode.py", png, "--key", f"{CALL}:{sp.picture_id:08x}", "--store",
            tmp_path / "s", *model_args(), "--precision", "fp16"]
    r = run(*argv, ok=False)
    assert r.returncode != 0 and not png.exists()
    assert "codec 1234" in r.stderr and "codec d1d8" in r.stderr, r.stderr
    r = run(*argv, "--any-codec")
    assert png.exists() and "decoding anyway" in r.stdout


@pytest.mark.slow
@pytest.mark.codec
def test_decode_cli_retro(tmp_path):
    """Integration review regression: `qrss_decode.py --retro` searches the
    passband store and attaches a -30 dB SHORT pass that was never received."""
    load_codec("fp16")
    from qrss_helpers import Q_TEST
    from sstvae.qrss import channel as chm
    from sstvae.qrss import frame, frontend, morse, tx
    from sstvae.qrss.channel import ChannelConfig

    sp = synthetic_picture(0)
    h = make_header(sp, callsign=CALL)
    st = Store(tmp_path / "s")
    p = make_pass_result(tx.slot_air_latents(sp, 0), 10 ** 0.5, seed=1, hdr=h, decoded=True)
    st.add_pass(p)
    st.attach((CALL, sp.picture_id), p.uid, 0, "header", mode=0, codec_id=sp.codec_id)
    q = Q_TEST + 11
    sym = tx.slot_symbols(sp, q, frame.SHORT, header=h)
    sim = chm.simulate(sym, frame.SHORT, q, ChannelConfig(snr_db=-30.0, preset="quiet", seed=7),
                       keying=morse.keying_units(CALL))
    frontend.PassbandStore(tmp_path / "s" / "passband").write_capture(
        frontend.Capture(sim.fe, q, sim.t0_index), expire=False)
    png = tmp_path / "out.png"
    r = run("qrss_decode.py", png, "--key", f"{CALL}:{sp.picture_id:08x}", "--store",
            tmp_path / "s", "--retro", "--frame", "short", *model_args(), "--precision", "fp16")
    assert "retro: 1 pass found" in r.stdout and "by corr" in r.stdout, r.stdout
    assert "2 passes" in r.stdout and png.exists()

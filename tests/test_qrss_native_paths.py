"""File names that are not ASCII -- and, on POSIX, not UTF-8 at all -- through
every native binding that takes a path and through sstvae-qrss, held to what
Python does with the same names.

The native code takes a file name as UTF-8 bytes (native/core/util/fileio.hpp;
on Windows those become UTF-16 for _wfopen) and the bindings make them as
open() does (native/bindings/module/qrss/fspath.hpp): os.fspath, then
os.fsencode, so a str, bytes or os.PathLike all work, and a str holding the
surrogate escapes of a name that is not UTF-8 means that name's own bytes.
Each binding is compared with its reference on three forms of each name
(str, bytes, Path): the same result, or the same exception -- class, errno,
text and filename. Where the reference does something else with a path, the
binding does that: capture_from_wav and receive_wav take str(path) first (a
bytes path becomes "b'...'" there), codec_id_from_onnx and the model paths
are made a Path (bytes are pathlib's TypeError), load_qrsp names the file as
f"{path}" does.

sstvae-qrss receives a WAV and an .npz, and transmits to a WAV, at such
names, side by side with qrss_receive.py and qrss_transmit.py. The scripts'
stdout prints a name that is not UTF-8 as its bytes ("surrogateescape") under
the C locale or in UTF-8 mode, and under any other locale is strict, so
qrss_transmit.py writes the file and then dies printing "wrote ...". They
are run here with PYTHONIOENCODING=utf-8:surrogateescape, so the comparison
does not depend on this machine's locale; the native prints the bytes. Their
stderr is "backslashreplace" whatever the locale (each such byte as the
six characters of a backslash-u escape of U+DCXX), and so is the native's.
"""

import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath

import numpy as np
import pytest

from qrss_codec_fixtures import scene
from qrss_helpers import Q_TEST, REPO_ROOT, synthetic_full_latents
from qrss_native import require
from sstvae import wavio
from sstvae.qrss import beaconfile, frame, picture
from sstvae.qrss import receiver as RX
from sstvae.qrss import types as qt
from sstvae.qrss.header import HeaderFields

NON_ASCII = "café — привет 日本 \U0001f600"
UTF8_FS = sys.getfilesystemencoding().lower() in ("utf-8", "utf8")
# A name that is no UTF-8 at all, as Python holds it: surrogate escapes.
NOT_UTF8 = os.fsdecode(b"caf\xe9-\xff\xfe") if os.name == "posix" and UTF8_FS else None
NAMES = [NON_ASCII] + ([NOT_UTF8] if NOT_UTF8 else [])
SLOT = "2025-10-01T00:00Z"


def _ids(names):
    return ["non-ascii" if n == NON_ASCII else "not-utf8" for n in names]


def forms(path: Path) -> list:
    """What open() takes for one file: a str, bytes and a Path."""
    return [str(path), os.fsencode(path), path]


def outcome(fn, *args, **kw):
    """("ok", result) or the exception as a comparable tuple."""
    try:
        return ("ok", fn(*args, **kw))
    except (Exception, SystemExit) as e:  # noqa: BLE001 -- compared, not swallowed
        return ("raise", type(e), str(e), getattr(e, "errno", None), getattr(e, "filename", None))


def same_raise(a, b, where):
    assert a[0] == b[0] == "raise", (where, a, b)
    assert a == b, (where, a, b)


@pytest.fixture(scope="module", params=NAMES, ids=_ids(NAMES))
def name(request):
    return request.param


@pytest.fixture
def d(tmp_path, name):
    """A directory whose own name is the name too, so every component is."""
    p = tmp_path / name
    p.mkdir()
    return p


# --- the WAV reader -----------------------------------------------------------------------------

def _wav(path: Path, seconds=1.0, seed=3):
    x = np.random.default_rng(seed).standard_normal(int(8000 * seconds)) * 0.1
    wavio.write_wav(path, x)


def test_read_wav(native, d, name):
    rs = require(native, "receiver_slot")
    p = d / f"{name}.wav"
    _wav(p)
    want = wavio.read_wav(str(p))
    for f in forms(p):
        got = rs._read_wav(f)
        assert got.dtype == want.dtype and got.tobytes() == want.tobytes(), type(f)
    for f in forms(d / f"{name}-missing.wav"):
        same_raise(outcome(rs._read_wav, f), outcome(wavio.read_wav, f), type(f))


def test_capture_from_wav(native, d, name):
    """capture_from_wav as the reference: str(path) first, so a bytes path
    is its repr, missing alike; typed (the C++ reads) and not."""
    rs = require(native, "receiver_slot")
    p = d / f"{name}.wav"
    _wav(p)
    ref = RX.capture_from_wav(p, Q_TEST)
    for typed in (True, False):
        for f in forms(p):
            want = outcome(RX.capture_from_wav, f, Q_TEST)
            got = outcome(rs.capture_from_wav, f, Q_TEST, None, typed)
            if isinstance(f, bytes):
                same_raise(got, want, (typed, "bytes"))
                assert got[1] is FileNotFoundError and got[4] == str(f), got
                continue
            assert got[0] == want[0] == "ok", (typed, f, got)
            c = got[1]
            assert (c.q, c.t0_index, c.fs) == (ref.q, ref.t0_index, ref.fs)
            assert c.fe.shape == ref.fe.shape and np.max(np.abs(c.fe - ref.fe)) <= 1e-5 * np.max(np.abs(ref.fe))
        for f in forms(d / f"{name}-missing.wav"):
            same_raise(outcome(rs.capture_from_wav, f, Q_TEST, None, typed),
                       outcome(RX.capture_from_wav, f, Q_TEST), (typed, type(f)))


def test_receive_wav(native, d, name):
    """receive_wav reads the file as capture_from_wav does: a noise WAV at
    such a name receives nothing, as the reference; a missing one is the
    reference's FileNotFoundError, naming str(path)."""
    rs = require(native, "receiver_slot")
    p = d / f"{name}.wav"
    _wav(p, seconds=2.0)
    assert RX.receive_wav(p, Q_TEST, spec=frame.TINY) == []
    for f in (str(p), p):
        assert rs.receive_wav(f, Q_TEST, None, frame.TINY, "joint") == []
    for f in forms(d / f"{name}-missing.wav"):
        same_raise(outcome(rs.receive_wav, f, Q_TEST, None, frame.TINY, "joint"),
                   outcome(RX.receive_wav, f, Q_TEST, spec=frame.TINY), type(f))


# --- .npy, .npz and a pass ----------------------------------------------------------------------

def test_npz_files(native, d, name):
    nat = require(native, "infra")
    arrays = {"version": np.int64(1), "z": np.linspace(0, 1, 7, dtype=np.float32)}
    for i, f in enumerate(forms(d / f"{name}.npz")):
        nat.save_npz(f, arrays)
        with np.load(f, allow_pickle=False) as z:
            assert z["z"].tobytes() == arrays["z"].tobytes() and int(z["version"]) == 1
        got = nat.load_npz(f)
        assert got["z"].tobytes() == arrays["z"].tobytes(), i
    for f in forms(d / f"{name}.npy"):
        nat.save_npy(f, arrays["z"])
        assert np.load(f).tobytes() == arrays["z"].tobytes()
        assert nat.load_npy(f).tobytes() == arrays["z"].tobytes()
    missing = d / f"{name}-missing.npz"
    shown = os.fsdecode(missing)
    for f in forms(missing):
        for fn in (nat.load_npz, nat.load_npy):
            r = outcome(fn, f)
            assert r[:3] == ("raise", RuntimeError, "cannot open " + shown), r
    nowhere = d / "no-such-dir" / f"{name}.npz"
    r = outcome(nat.save_npz, os.fsencode(nowhere), arrays)
    assert r[:3] == ("raise", RuntimeError, "cannot write " + os.fsdecode(nowhere)), r


def test_pass_files(native, d, name):
    from test_qrss_native_types import fake_pass

    nt = require(native, "types")
    p = fake_pass()
    ref_file = d / f"{name}-py.npz"
    p.save(ref_file)
    for f in forms(d / f"{name}.npz"):
        nt.save_pass(p, f)
        assert Path(os.fsdecode(f)).read_bytes() == ref_file.read_bytes(), type(f)
        d2 = nt.load_pass(f)
        assert d2["uid"] == p.uid
    r = outcome(nt.load_pass, os.fsencode(d / "missing.npz"))
    assert r[:3] == ("raise", RuntimeError, "cannot open " + os.fsdecode(d / "missing.npz")), r


def test_bad_path_arguments(native):
    """open()'s own refusals, where it makes them: a NUL, and something that
    is no path at all."""
    nat = require(native, "infra")
    rs = require(native, "receiver_slot")
    for fn, ref in ((nat.load_npz, np.load), (rs._read_wav, wavio.read_wav)):
        a, b = outcome(fn, "a\0b"), outcome(ref, "a\0b")
        assert a[:3] == b[:3] == ("raise", ValueError, "embedded null byte"), (a, b)
        a, b = outcome(fn, 3.5), outcome(open, 3.5)
        assert a[:3] == b[:3] == ("raise", TypeError, "expected str, bytes or os.PathLike object, not float")


# --- the transmit side --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def stored():
    sp = picture.StoredPicture.from_latents(synthetic_full_latents(3), 0xD1D8, 1)
    h = HeaderFields(callsign="K1ABC", grid="FN42", picture_id=sp.picture_id, mode=sp.mode, segment=1,
                     codec_id=sp.codec_id)
    return sp, h


def test_wav_writers(native, d, name, stored):
    nat = require(native, "tx")
    sp, h = stored
    pa = nat.PassAudio(sp, Q_TEST, 1500.0, spec=frame.TINY, segment=1, header=h)
    y = pa.render(0, 8000 * 3)
    plain = d.parent / "plain"
    plain.mkdir(exist_ok=True)
    wavio.write_wav(plain / "ref.wav", y)
    wavio.write_wav_float(plain / "ref_f.wav", y)
    pa.write_wav(plain / "pass.wav")
    for i, f in enumerate(forms(d / f"{name}.wav")):
        nat.write_wav(f, y)
        assert Path(os.fsdecode(f)).read_bytes() == (plain / "ref.wav").read_bytes(), i
        nat.write_wav_float(f, y)
        assert Path(os.fsdecode(f)).read_bytes() == (plain / "ref_f.wav").read_bytes(), i
        pa.write_wav(f)
        assert Path(os.fsdecode(f)).read_bytes() == (plain / "pass.wav").read_bytes(), i
    for f in forms(d / "no-such-dir" / f"{name}.wav"):
        want = outcome(wavio.write_wav, f, y)
        same_raise(outcome(nat.write_wav, f, y), want, type(f))
        same_raise(outcome(pa.write_wav, f), want, type(f))


def test_sources(native, d, name, stored):
    """api.read_source, picture.load_qrsp and beaconfile at such names."""
    api = require(native, "api")
    npic = require(native, "picture")
    nbf = require(native, "beaconfile")
    sp, h = stored
    qrsp = d / f"{name}.qrsp"
    picture.save_qrsp(qrsp, sp)
    for f in forms(qrsp):
        got = api.read_source(f)
        assert (got["mode"], got["codec_id"], got["picture_id"]) == (sp.mode, sp.codec_id, sp.picture_id)
        assert npic.load_qrsp(f)["picture_id"] == sp.picture_id
    for f in forms(d / f"{name}-missing.qrsp"):
        r = outcome(api.read_source, f)
        try:
            open(f, "rb")
        except OSError as e:
            assert r[:3] == ("raise", RuntimeError, str(e)), r
    # load_qrsp's refusals name the file as f"{path}" does
    bad = d / f"{name}-bad.qrsp"
    with open(bad, "wb") as fh:
        np.savez(fh, version=np.int64(99), mode=np.int64(0), codec_id=np.int64(1), picture_id=np.int64(0),
                 segs=np.zeros((1, 4), np.float16))
    for f in forms(bad):
        same_raise(outcome(npic.load_qrsp, f), outcome(picture.load_qrsp, f), type(f))
    # beacon files: Python's own open() on both sides
    bf = beaconfile.from_picture(sp, 1, h)
    ref = d / f"{name}-ref.bin"
    beaconfile.write(ref, bf)
    for f in forms(d / f"{name}.bin"):
        nbf.write(f, bf)
        assert open(f, "rb").read() == ref.read_bytes()
        assert nbf.read(f)["callsign"] == "K1ABC"


# --- the codec glue (with the codec only) ---------------------------------------------------------

def test_codec_paths(native, d, name):
    """The model paths (checkpoint.resolve_onnx, codec_id_from_onnx: a str
    or a PathLike, made a Path; bytes refused as pathlib refuses them), the
    pathlib spelling of such a name, and images.load_image (PIL's open:
    str, bytes or PathLike) at such names."""
    from PIL import Image

    from sstvae import images
    from sstvae.checkpoint import resolve_onnx

    nc = require(native, "codec")
    npic = require(native, "picture")
    for part in ("encoder", "decoder"):
        (d / f"m-{part}-fp16.onnx").write_bytes(b"not a model")
    for f in (str(d), d):
        assert nc.resolve_onnx("encoder", f, "fp16") == str(resolve_onnx("encoder", f, "fp16"))
    for f in (str(d / "nothing-here"), d / "nothing-here"):
        want = outcome(resolve_onnx, "encoder", f, "fp16")
        assert want[1] is SystemExit
        same_raise(outcome(nc.resolve_onnx, "encoder", f, "fp16"), want, type(f))
    a = outcome(npic.codec_id_from_onnx, os.fsencode(d))
    b = outcome(picture.codec_id_from_onnx, os.fsencode(d))
    assert a[:2] == b[:2] == ("raise", TypeError), (a, b)
    for t in (f"{d}//x/./{name}.onnx/", f"{name}.tar.{name}"):
        p = PurePosixPath(t)
        assert (nc._pathlib_str(t), nc._pathlib_suffix(t)) == (str(p), p.suffix), t
    png = d / f"{name}.png"
    Image.fromarray(scene()).save(png)
    want = images.load_image(str(png))
    for f in forms(png):
        assert np.array_equal(nc.load_image(f), images.load_image(f)), type(f)
        assert np.array_equal(nc.load_image(f), want), type(f)
    # the image layer's refusal names the file as Python decodes names
    r = outcome(nc.load_image, d / f"{name}-missing.png")
    assert r[1] is RuntimeError and os.fsdecode(d / f"{name}-missing.png") in r[2], r


@pytest.mark.codec
def test_codec_model_paths(native, d, name):
    """The v5 model in a directory at such a name (links to the files):
    codec_id_from_onnx, the encoder found there (qrss/codec's
    encode_picture, api.open_encoder and its encode_file, the image at such
    a name too) and the decoder (codec_render.decoder_codec_id), each what
    the same files give at a plain name. (test_qrss_native_codec.py holds
    those to the reference; here the reference cannot follow everywhere.)"""
    import qrss_codec_fixtures as fixtures
    from PIL import Image

    from sstvae.checkpoint import resolve_onnx

    nc = require(native, "codec")
    npic = require(native, "picture")
    api = require(native, "api")
    v5 = fixtures.v5_dir()
    m = d / f"{name}-model"
    m.mkdir()
    for part in ("encoder", "decoder"):
        src = Path(resolve_onnx(part, v5, "fp16"))
        (m / src.name).symlink_to(src)
    plain = d.parent / "scene.png"
    Image.fromarray(scene()).save(plain)
    png = d / f"{name}.png"
    png.write_bytes(plain.read_bytes())
    cid = npic.codec_id_from_onnx(str(v5))
    for f in (str(m), m):
        assert npic.codec_id_from_onnx(f) == cid
        # The reference's own onnxruntime session takes the file name as a
        # strict str (its pybind11 binding), so a name that is not UTF-8 is
        # its TypeError there; the native opens it.
        ref = outcome(picture.codec_id_from_onnx, f)
        assert ref == ("ok", cid) if name == NON_ASCII else ref[1] is TypeError, ref
    want = nc.encode_picture(str(plain), 0, cid, model=str(v5))
    for img in forms(png):
        for model in (str(m), m):
            got = nc.encode_picture(img, 0, cid, model=model)
            assert [np.asarray(s).tobytes() for s in got["segs"]] == [np.asarray(s).tobytes() for s in want["segs"]]
            assert got["picture_id"] == want["picture_id"]
    enc = api.open_encoder(m)
    assert enc.onnx_path == str(resolve_onnx("encoder", m, "fp16")) and enc.codec_id == cid
    for img in forms(png):
        assert enc.encode_file(img, 0)["picture_id"] == want["picture_id"]
    cr = native.qrss.codec_render
    assert cr.decoder_codec_id(m, "fp16") == cr.decoder_codec_id(str(v5), "fp16") == cid


# --- sstvae-qrss ----------------------------------------------------------------------------------

def _module_dir() -> Path:
    env = os.environ.get("SSTVAE_NATIVE_MODULE_DIR")
    return Path(env).resolve() if env else REPO_ROOT / "native" / "build" / "python"


@pytest.fixture(scope="module")
def cli():
    env = os.environ.get("SSTVAE_QRSS_CLI")
    p = Path(env) if env else _module_dir().parent / "sstvae-qrss"
    if not (p.is_file() and os.access(p, os.X_OK)):
        pytest.skip("sstvae-qrss not built (set SSTVAE_QRSS_CLI, or build beside the module)")
    return p


def _run(cmd: list, cwd: Path):
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env["COLUMNS"] = "200"
    env["PYTHONIOENCODING"] = "utf-8:surrogateescape"
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, timeout=180, env=env)
    return r.returncode, r.stdout, r.stderr


def _side_by_side(cli: Path, sub: str, args: list, cwd: Path):
    """The script and the subcommand on the same argv (bytes on POSIX, so
    a name that is not UTF-8 reaches both as itself), in parallel."""
    with ThreadPoolExecutor(max_workers=2) as pool:
        py = pool.submit(_run, [sys.executable, str(REPO_ROOT / f"qrss_{sub}.py"), *args], cwd)
        nat = pool.submit(_run, [str(cli), sub, *args], cwd)
        return py.result(), nat.result()


def _arg(text: str):
    return os.fsencode(text) if os.name == "posix" else text


@pytest.fixture(scope="module")
def tiny_inputs(tmp_path_factory):
    """A stored picture, a TINY CE .npz at -? dB (qrss_simulate.py) and a
    noise WAV, each at an ASCII name; the cases copy them under theirs."""
    root = tmp_path_factory.mktemp("paths-cli")
    sp = picture.StoredPicture.from_latents(synthetic_full_latents(3), 0xD1D8, 1)
    picture.save_qrsp(root / "pic.qrsp", sp)
    r = subprocess.run([sys.executable, str(REPO_ROOT / "qrss_simulate.py"), "pic.qrsp", "tiny.npz", "--slot", SLOT,
                        "--callsign", "K1ABC", "--frame", "tiny", "--snr", "10", "--seed", "1"],
                       cwd=root, capture_output=True, text=True, timeout=180)
    assert r.returncode == 0, r.stderr
    _wav(root / "noise.wav", seconds=4.0)
    return root


def test_cli_receive(cli, tiny_inputs, tmp_path, name):
    """sstvae-qrss receive of an .npz (a TINY pass) and a WAV at such a
    name, and of a missing one: stdout, stderr and status as the script's."""
    d = tmp_path / name
    d.mkdir()
    npz, wav = d / f"{name}.npz", d / f"{name}.wav"
    npz.write_bytes((tiny_inputs / "tiny.npz").read_bytes())
    wav.write_bytes((tiny_inputs / "noise.wav").read_bytes())
    cases = [
        ([_arg(os.fspath(npz)), "--no-store"], 0),
        ([_arg(f"{name}/{name}.npz"), "--no-store"], 0),
        ([_arg(os.fspath(wav)), "--slot", SLOT, "--frame", "tiny", "--no-store"], 1),
        ([_arg(f"./{name}//{name}-missing.wav"), "--slot", SLOT, "--no-store"], 1),
    ]
    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs = [pool.submit(_side_by_side, cli, "receive", args, tmp_path) for args, _ in cases]
        results = [j.result() for j in jobs]
    for (args, rc), (py, nat) in zip(cases, results):
        assert py[0] == rc, (args, py)
        assert nat == py, (args, py, nat)
    # the pass was received, and a missing file named as pathlib prints it,
    # through stderr's "backslashreplace" ("\udcXX" for a byte not UTF-8)
    assert b"   CE tiny\n" in results[0][1][1]
    assert results[3][1][2] == f"{name}/{name}-missing.wav: no such file\n".encode("utf-8", "backslashreplace")


def test_cli_transmit(cli, tiny_inputs, tmp_path, name):
    """sstvae-qrss transmit to a WAV at such a name: "wrote <name>: ..." as
    the script prints it, and the file the script writes (the CE TINY
    pass, bit for bit or to the CLI test's bar)."""
    from test_qrss_native_cli import _same_wav

    (tmp_path / "pic.qrsp").write_bytes((tiny_inputs / "pic.qrsp").read_bytes())
    (tmp_path / "py").mkdir()
    (tmp_path / "nat").mkdir()
    out = f"{name}.wav"
    args = ["../pic.qrsp", _arg(out), "--slot", SLOT, "--callsign", "K1ABC", "--frame", "tiny"]
    with ThreadPoolExecutor(max_workers=2) as pool:
        py = pool.submit(_run, [sys.executable, str(REPO_ROOT / "qrss_transmit.py"), *args], tmp_path / "py")
        nat = pool.submit(_run, [str(cli), "transmit", *args], tmp_path / "nat")
        py, nat = py.result(), nat.result()
    assert py[0] == nat[0] == 0, (py, nat)
    assert nat[1:] == py[1:], (py, nat)
    assert nat[1].startswith(b"wrote " + _arg(out) + b": ")
    _same_wav("ce", (tmp_path / "py" / out).read_bytes(), (tmp_path / "nat" / out).read_bytes())

"""C++ codec glue (native/core/qrss/codec) against the reference.

What qrss_encode.py does between its arguments and its output, piece by
piece: `picture.codec_id_from_onnx`, `checkpoint.resolve_onnx` with an
explicit --model, `images.load_image` and the encode into a
StoredPicture. Built only with SSTVAE's codec, so everything here skips
when `sstvae_native.qrss.codec` is absent.

- The codec ID is compared on hand-built ONNX models (an Identity graph,
  a few hundred bytes, qrss_codec_fixtures.tiny_onnx) carrying whatever
  `sstvae.source_sha256` a case needs -- absent, empty, short, signed,
  underscored, not hex -- so the hex reading is pinned without the
  published artifacts; on a directory, which part is preferred and what
  an ambiguous or empty one does; and on files onnxruntime refuses, by
  exception class and text. Then on the v5 artifacts (`codec` marker):
  0xD1D8 for each of the four files and for their directory, fp16 and
  fp32.
- resolve_onnx and pathlib's spelling of a path are compared exactly,
  refusal texts (SystemExit) included.
- load_image on 640x480 PNGs (RGB, grey, RGBA, palette) is bit-exact. At
  any other size the resize is stb's, not PIL's LANCZOS -- the accepted
  difference CLAUDE.md records for sstvae-encode -- and is not compared.
- encode_picture is bit-exact against the reference's encode of the same
  PNG through the v5 encoder: the fp16 segments byte for byte and the
  picture ID, modes A, B and C at fp16 and mode B at fp32.

The onnxruntime refusals are compared for the errors whose wording both
runtime versions share (the module links the pinned 1.28.0, the Python
side imports whatever the environment has). Two are not: an empty .onnx
(1.28 says FAIL "ModelProto does not have a graph", 1.31 INVALID_ARGUMENT
"No graph was found in the protobuf") and a directory where the model
should be (1.28 points at its model-package API, 1.31 fails to parse it).
Both are onnxruntime's exceptions on both sides, which is what is checked
there.
"""

from pathlib import Path, PurePosixPath

import numpy as np
import pytest

import qrss_codec_fixtures as fixtures
from qrss_codec_fixtures import scene, tiny_onnx
from qrss_helpers import synthetic_full_latents
from qrss_native import codec as rows_mod
from qrss_native import require
from sstvae import checkpoint
from sstvae.qrss import picture

V5_CODEC_ID = 0xD1D8


@pytest.fixture(scope="module")
def nat(native):
    return require(native, "codec")


@pytest.fixture(scope="module")
def npic(native):
    require(native, "codec")
    return native.qrss.picture


def _write(path: Path, data: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return str(path)


def _outcome(f, *args, **kw):
    try:
        return ("ok", f(*args, **kw))
    except SystemExit as e:
        return ("exit", str(e.code))
    except Exception as e:      # noqa: BLE001 -- any refusal is compared
        return ("raise", type(e).__module__, type(e).__name__, str(e))


def _same(f_ref, f_nat, *args, **kw):
    want, got = _outcome(f_ref, *args, **kw), _outcome(f_nat, *args, **kw)
    assert got == want
    if want[0] == "ok":
        assert type(got[1]) is type(want[1])
    return want


# --- the rows -------------------------------------------------------------------------

def test_rows_name_the_native_codec_id(native):
    require(native, "codec")
    rows = rows_mod.rows(native.qrss)
    assert rows == {("sstvae.qrss.picture", "codec_id_from_onnx"): native.qrss.picture.codec_id_from_onnx}
    assert rows_mod.STAGE == "tx"


def test_rows_are_empty_without_the_codec():
    class NoCodec:
        picture = object()

    assert rows_mod.rows(NoCodec()) == {}


# --- the codec ID ------------------------------------------------------------------------

SHAS = [
    None, "", "a", "abc", "d1d8", "D1D8" + "0" * 60, "d1d8" + "f" * 60, "0000", "ffff",
    "0x12ab", "0X12", " 1a2", "\t1a2", "1a2 ", "1_23", "_123", "12_3", "1__2", "0x_1",
    "-1ab", "+1ab", "--12", "zz12", "12zz", "g000", "    ", "1 2 3",
]


@pytest.mark.parametrize("sha", SHAS, ids=repr)
def test_codec_id_of_sha_matches_the_reference(nat, npic, tmp_path, sha):
    path = _write(tmp_path / "m-encoder-fp16.onnx", tiny_onnx(sha))
    want = _same(picture.codec_id_from_onnx, npic.codec_id_from_onnx, path)
    assert want[0] == "ok"
    assert nat.source_sha256(path) == sha
    if sha is not None:
        assert nat.codec_id_of_sha(sha) == want[1]


def test_codec_id_of_sha_pins(nat):
    assert nat.codec_id_of_sha("d1d8") == V5_CODEC_ID
    assert nat.codec_id_of_sha("0x12ab") == 0x12
    assert nat.codec_id_of_sha("-1ab") == -0x1AB
    assert nat.codec_id_of_sha("1_23") == 0x123
    assert nat.codec_id_of_sha("zz12") is None
    assert nat.codec_id_of_sha("abc") is None


def test_codec_id_accepts_a_path_object_and_a_precision(npic, tmp_path):
    path = _write(tmp_path / "m-decoder-fp32.onnx", tiny_onnx("beef"))
    assert npic.codec_id_from_onnx(Path(path)) == 0xBEEF
    assert npic.codec_id_from_onnx(path, "fp32") == 0xBEEF
    assert npic.codec_id_from_onnx(path_or_dir=path, precision="int8") == 0xBEEF


def _tree(root: Path) -> dict[str, str]:
    """Directories of tiny models, each part carrying a different hash."""
    files = {
        "both/v9-encoder-fp16.onnx": "e1e1",
        "both/v9-decoder-fp16.onnx": "d0d0",
        "both/v9-decoder-fp32.onnx": "f32d",
        "dec/v9-decoder-fp16.onnx": "dec0",
        "dec/v9-decoder-grad-fp32.onnx": "9999",
        "amb/a-encoder-fp16.onnx": "aaaa",
        "amb/b-encoder-fp16.onnx": "bbbb",
        "amb/a-decoder-fp16.onnx": "abab",
        "ambboth/a-encoder-fp16.onnx": "aaaa",
        "ambboth/b-encoder-fp16.onnx": "bbbb",
        "ambboth/a-decoder-fp16.onnx": "abab",
        "ambboth/b-decoder-fp16.onnx": "baba",
        "bare/-encoder-fp16.onnx": "0e0e",
        "hidden/.v-encoder-fp16.onnx": "1d1d",
        "nometa/v9-encoder-fp16.onnx": None,
        "nometa/v9-decoder-fp16.onnx": "c0de",
        "int8only/v9-encoder-int8.onnx": "1818",
        "other/readme.txt": None,
    }
    for rel, sha in files.items():
        _write(root / rel, tiny_onnx(sha) if rel.endswith(".onnx") else b"text\n")
    (root / "empty").mkdir()
    (root / "subdir" / "v9-encoder-fp16.onnx").mkdir(parents=True)     # a directory so named
    _write(root / "subdir" / "v9-decoder-fp16.onnx", tiny_onnx("5d5d"))
    return files


DIRS = ["both", "dec", "amb", "ambboth", "bare", "hidden", "nometa", "int8only", "other", "empty"]


@pytest.mark.parametrize("precision", ["fp16", "fp32", "int8"])
@pytest.mark.parametrize("d", DIRS)
def test_codec_id_of_a_directory_matches_the_reference(npic, tmp_path, d, precision):
    _tree(tmp_path)
    for spelling in (str(tmp_path / d), str(tmp_path / d) + "/", str(tmp_path) + "//" + d + "/./"):
        _same(picture.codec_id_from_onnx, npic.codec_id_from_onnx, spelling, precision)


def test_codec_id_of_a_directory_prefers_the_encoder(npic, tmp_path):
    _tree(tmp_path)
    assert npic.codec_id_from_onnx(str(tmp_path / "both")) == 0xE1E1
    assert npic.codec_id_from_onnx(str(tmp_path / "both"), "fp32") == 0xF32D
    assert npic.codec_id_from_onnx(str(tmp_path / "dec")) == 0xDEC0
    assert npic.codec_id_from_onnx(str(tmp_path / "amb")) == 0xABAB       # ambiguous encoder: the decoder
    assert npic.codec_id_from_onnx(str(tmp_path / "nometa")) is None      # the encoder, without metadata
    with pytest.raises(FileNotFoundError, match=r"^no \*-encoder-fp16.onnx or \*-decoder-fp16.onnx in "):
        npic.codec_id_from_onnx(str(tmp_path / "ambboth"))


def _ort_refusal(f, path):
    """onnxruntime's exception with its "[ONNXRuntimeError] : N : NAME :" form."""
    got = _outcome(f, path)
    assert got[0] == "raise" and got[1] == "onnxruntime.capi.onnxruntime_pybind11_state", got
    code = got[3].split(" : ")[1]
    assert got[3].startswith(f"[ONNXRuntimeError] : {code} : "), got
    return got


def test_codec_id_of_a_directory_named_like_a_model(npic, tmp_path):
    """glob sees directories too: the encoder slot is taken by a directory,
    which onnxruntime then refuses, as it does for the reference. Its words
    for that are the runtime version's (1.28 points at its model-package
    API, 1.31 reports a protobuf parse failure), so only the kind is pinned."""
    _tree(tmp_path)
    for f in (picture.codec_id_from_onnx, npic.codec_id_from_onnx):
        _ort_refusal(f, str(tmp_path / "subdir"))


def test_onnxruntime_refusals_match_the_reference(npic, tmp_path):
    garbage = _write(tmp_path / "g-encoder-fp16.onnx", b"garbage bytes, not a protobuf")
    for path in (garbage, str(tmp_path / "missing-encoder-fp16.onnx"), str(tmp_path / "notes.txt"),
                 str(tmp_path) + "//./g-encoder-fp16.onnx"):
        want = _same(picture.codec_id_from_onnx, npic.codec_id_from_onnx, path)
        assert want[0] == "raise" and want[1] == "onnxruntime.capi.onnxruntime_pybind11_state"


def test_an_empty_model_is_refused_by_onnxruntime(npic, tmp_path):
    empty = _write(tmp_path / "e-encoder-fp16.onnx", b"")
    for f in (picture.codec_id_from_onnx, npic.codec_id_from_onnx):
        assert f"Load model from {empty} failed:" in _ort_refusal(f, empty)[3]


def test_codec_id_refuses_what_pathlib_refuses(npic):
    for bad in (None, 3, b"bytes.onnx"):
        want, got = _outcome(picture.codec_id_from_onnx, bad), _outcome(npic.codec_id_from_onnx, bad)
        assert got[0] == want[0] == "raise" and got[2] == want[2] == "TypeError"


# --- resolve_onnx and pathlib ------------------------------------------------------------

PARTS = ["encoder", "decoder", "decoder-grad"]


def _model_paths(root: Path) -> list[str]:
    r = str(root)
    out = [r + "/" + d for d in DIRS] + [r + "/both/", r + "//both/./", r + "/nowhere", r + "/nowhere/"]
    out += [
        r + "/both/v9-encoder-fp16.onnx", r + "/both/v9-decoder-fp16.onnx",
        r + "/both/v9-decoder-fp32.onnx", r + "/dec/v9-decoder-grad-fp32.onnx",
        r + "/dec/v9-decoder-fp16.onnx", r + "/both/x.onnx", r + "/both/.onnx",
        r + "/both/v9-encoder-fp16.onnx/", r + "/both/v9-encoder-int8.onnx",
        r + "/both/model.pt", r + "/both/weights.bin", r + "/other/readme.txt",
        r + "/both/v9-encoderish-fp16.onnx", r + "/dec/v9-decoder-grad-fp16.onnx",
    ]
    return out


@pytest.mark.parametrize("precision", ["fp16", "fp32", "int8"])
@pytest.mark.parametrize("part", PARTS)
def test_resolve_onnx_matches_the_reference(nat, tmp_path, part, precision):
    _tree(tmp_path)
    for path in _model_paths(tmp_path):
        _same(checkpoint.resolve_onnx, nat.resolve_onnx, part, path, precision)


def test_resolve_onnx_refusal_texts(nat, tmp_path):
    _tree(tmp_path)
    r = str(tmp_path)
    assert _outcome(nat.resolve_onnx, "encoder", r + "/empty") == (
        "exit", f"no *-encoder-fp16.onnx in {r}/empty\nthat directory holds no .onnx files at all")
    assert _outcome(nat.resolve_onnx, "encoder", r + "/amb") == (
        "exit", f"ambiguous: 2 candidates for encoder/fp16 in {r}/amb (a-encoder-fp16.onnx, b-encoder-fp16.onnx)")
    assert _outcome(nat.resolve_onnx, "decoder-grad", r + "/both/v9-encoder-fp16.onnx") == (
        "exit", "need the decoder-grad as well as the encoder, but v9-decoder-grad-fp32.onnx is not next to "
                "v9-encoder-fp16.onnx")


PATHS = [
    "", ".", "./", "/", "//", "///", "////a", "//a", "//a//b", "a//b/", "./a/./b", "a/../b", "/./x",
    "x/.", "~/x", "a/b/", "a.", ".onnx", "a.b.", "x/.a.pt", "a..pt", "foo.PT", "a.pt/", "a.tar.gz",
    "a/b.c/d", "..", "a/..", "...", "..a", "a/.", "é/ñ.onnx", "müde.pt", "sp ace/x y.onnx",
]


@pytest.mark.parametrize("text", PATHS, ids=repr)
def test_pathlib_spelling_matches(nat, text):
    p = PurePosixPath(text)
    assert nat._pathlib_str(text) == str(p)
    assert nat._pathlib_suffix(text) == p.suffix


# --- images -----------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def scene_png(tmp_path_factory):
    from PIL import Image

    path = tmp_path_factory.mktemp("scene") / "scene.png"
    Image.fromarray(scene()).save(path)
    return str(path)


@pytest.mark.parametrize("mode", ["RGB", "L", "RGBA", "P"])
def test_load_image_is_bit_exact_at_640x480(nat, tmp_path, mode):
    from PIL import Image

    from sstvae import images

    img = Image.fromarray(scene(seed=11))
    img = img.convert(mode) if mode != "RGBA" else Image.fromarray(
        np.dstack([scene(seed=11), scene(seed=12)[..., 0]]), "RGBA")
    path = tmp_path / f"p-{mode}.png"
    img.save(path)
    want = images.load_image(str(path))
    got = nat.load_image(str(path))
    assert got.dtype == want.dtype == np.float32 and got.shape == want.shape == (3, 480, 640)
    assert np.array_equal(got, want)


# --- the encode -------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def v5_dir():
    return str(fixtures.v5_dir())


@pytest.mark.codec
@pytest.mark.parametrize("precision", ["fp16", "fp32"])
def test_v5_codec_id_is_d1d8(npic, v5_dir, precision):
    for part in ("encoder", "decoder"):
        f = checkpoint.resolve_onnx(part, v5_dir, precision)
        assert picture.codec_id_from_onnx(f, precision) == V5_CODEC_ID
        assert npic.codec_id_from_onnx(f, precision) == V5_CODEC_ID
        assert npic.codec_id_from_onnx(f) == V5_CODEC_ID
    assert npic.codec_id_from_onnx(v5_dir, precision) == V5_CODEC_ID
    assert npic.codec_id_from_onnx(v5_dir + "/", precision) == V5_CODEC_ID


@pytest.fixture(scope="module")
def ref_latents(v5_dir, scene_png):
    """The reference's encode of the scene, per precision (load once)."""
    from sstvae import codec, images

    img = images.load_image(scene_png)
    return {prec: codec.load_codec(v5_dir, precision=prec).encode(img) for prec in ("fp16", "fp32")}


def _same_picture(got: dict, want: picture.StoredPicture):
    assert got["mode"] == want.mode and got["codec_id"] == want.codec_id
    assert got["picture_id"] == want.picture_id
    assert len(got["segs"]) == len(want.segs) == want.mode + 1
    for a, b in zip(got["segs"], want.segs):
        a, b = np.asarray(a), np.asarray(b)
        assert a.dtype == b.dtype == np.float16 and a.shape == b.shape
        assert a.tobytes() == b.tobytes()


@pytest.mark.codec
@pytest.mark.parametrize("precision,mode", [("fp16", 0), ("fp16", 1), ("fp16", 2), ("fp32", 1)])
def test_encode_picture_is_bit_exact(nat, v5_dir, scene_png, ref_latents, precision, mode):
    want = picture.StoredPicture.from_latents(ref_latents[precision], V5_CODEC_ID, mode)
    got = nat.encode_picture(scene_png, mode, V5_CODEC_ID, model=v5_dir, precision=precision)
    _same_picture(got, want)
    # A single .onnx names the model too (the encoder found beside a decoder).
    if (precision, mode) == ("fp16", 0):
        dec = checkpoint.resolve_onnx("decoder", v5_dir, precision)
        _same_picture(nat.encode_picture(scene_png, mode, V5_CODEC_ID, dec, precision), want)


def test_encode_picture_refusals(nat, tmp_path, scene_png):
    """Mode and codec ID are refused before any model is looked for."""
    nowhere = str(tmp_path / "nowhere")
    for mode, cid in ((3, 1), (-1, 1), (0, 1 << 16), (0, -1)):
        want = _outcome(picture.StoredPicture.from_latents, synthetic_full_latents(1), cid, mode)
        got = _outcome(nat.encode_picture, scene_png, mode, cid, nowhere)
        assert want[0] == "raise" and got == want
    got = _outcome(nat.encode_picture, scene_png, 0, 1, nowhere)
    assert got == ("exit", f"--model {nowhere}: expected a .pt checkpoint, a .onnx artifact, or a directory "
                           "containing the exported .onnx files")


def test_load_image_refuses_what_is_not_a_picture(nat, tmp_path):
    bad = _write(tmp_path / "x.png", b"not a picture")
    with pytest.raises(RuntimeError, match="cannot read"):
        nat.load_image(bad)

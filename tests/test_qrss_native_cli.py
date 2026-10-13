"""`sstvae-qrss encode` and `beacon` (native/apps/sstvae_qrss) against
qrss_encode.py and qrss_beacon.py, run side by side.

Each case runs the script and the subcommand with the same arguments, in
two directories holding the same input files, and compares:

- the exit status;
- stdout, exactly (the "wrote ..." line);
- the output file, byte for byte, or that neither run left one;
- stderr by how the script ended. A refusal (SystemExit) exactly. A usage
  error by its message, the text after "error: " on argparse's last line
  ("the following arguments are required: ...") -- the usage line above it
  names a different program and argparse wraps it at the terminal width.
  An exception the script does not catch by the traceback's last line,
  which is all the native prints ("FileNotFoundError: [Errno 2] ...");
  where only the class can agree (a corrupt zip: zipfile's message is
  zipfile's), by the class.

Also int() and float() on text (native/core/qrss/pynum), which decide what
argparse's type=int and qrss_encode.py's int(text, 16) accept.

The binary is $SSTVAE_QRSS_CLI, else `sstvae-qrss` beside the module's
build directory ($SSTVAE_NATIVE_MODULE_DIR/.., native/build by default);
everything skips without it. The encode cases need the codec (a build
without it says so and exits 2, which is checked instead) and the v5
models (`codec` marker). The Python runs start in parallel: each costs a
second of imports.
"""

import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest

import qrss_codec_fixtures as fixtures
from qrss_helpers import REPO_ROOT, synthetic_full_latents
from qrss_native import require
from sstvae.qrss import picture


def _module_dir() -> Path:
    env = os.environ.get("SSTVAE_NATIVE_MODULE_DIR")
    return Path(env).resolve() if env else REPO_ROOT / "native" / "build" / "python"


def _cli() -> Path | None:
    env = os.environ.get("SSTVAE_QRSS_CLI")
    p = Path(env) if env else _module_dir().parent / "sstvae-qrss"
    return p if p.is_file() and os.access(p, os.X_OK) else None


@pytest.fixture(scope="module")
def cli():
    p = _cli()
    if p is None:
        pytest.skip("sstvae-qrss not built (set SSTVAE_QRSS_CLI, or build beside the module)")
    return p


def _has_encode(cli: Path) -> bool:
    r = subprocess.run([cli, "--help"], capture_output=True, text=True, timeout=60)
    return any(line.split()[:1] == ["encode"] for line in r.stdout.splitlines())


# --- running a case -------------------------------------------------------------------------

@dataclass
class Case:
    name: str
    script: str                      # "qrss_beacon.py"
    sub: str                         # "beacon"
    args: list[str]
    files: dict[str, Path] = field(default_factory=dict)   # name in the run dir -> source
    dirs: list[str] = field(default_factory=list)          # empty directories to make
    output: str | None = None        # the file a success writes
    only_class: bool = False         # an uncaught exception's class, not its text


@dataclass
class Run:
    rc: int
    out: str
    err: str
    output: bytes | None


def _run(cmd: list, cwd: Path, output: str | None) -> Run:
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env["COLUMNS"] = "200"
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=180, env=env)
    data = (cwd / output).read_bytes() if output and (cwd / output).is_file() else None
    return Run(r.returncode, r.stdout, r.stderr, data)


def _stage(case: Case, root: Path) -> tuple[Path, Path]:
    dirs = []
    for side in ("py", "nat"):
        d = root / case.name / side
        d.mkdir(parents=True)
        for name, src in case.files.items():
            (d / name).parent.mkdir(parents=True, exist_ok=True)
            os.symlink(Path(src).resolve(), d / name)
        for sub in case.dirs:
            (d / sub).mkdir(parents=True, exist_ok=True)
        dirs.append(d)
    return dirs[0], dirs[1]


def run_cases(cases: list[Case], root: Path, cli: Path) -> dict[str, tuple[Run, Run]]:
    jobs = {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        for c in cases:
            py_dir, nat_dir = _stage(c, root)
            jobs[c.name] = (
                pool.submit(_run, [sys.executable, str(REPO_ROOT / c.script), *c.args], py_dir, c.output),
                pool.submit(_run, [str(cli), c.sub, *c.args], nat_dir, c.output),
            )
        return {name: (a.result(), b.result()) for name, (a, b) in jobs.items()}


def _last_line(text: str) -> str:
    lines = [ln for ln in text.splitlines() if ln.strip()]
    return lines[-1] if lines else ""


def check(case: Case, py: Run, nat: Run):
    assert nat.rc == py.rc, (py, nat)
    assert nat.out == py.out, (py, nat)
    assert nat.output == py.output, "output files differ"
    assert "Traceback" not in nat.err
    if py.rc == 0:
        assert nat.err == py.err
    elif py.rc == 2:
        prog = f"{case.script}: error: "
        assert py.err.startswith("usage: ") and _last_line(py.err).startswith(prog), py.err
        assert nat.err.startswith("usage: sstvae-qrss " + case.sub), nat.err
        assert _last_line(nat.err) == f"sstvae-qrss {case.sub}: error: " + _last_line(py.err)[len(prog):]
    elif "Traceback (most recent call last)" in py.err:
        want, got = _last_line(py.err), _last_line(nat.err)
        if case.only_class:
            assert got.split(":")[0] == want.split(":")[0], (want, got)
        else:
            assert got == want
        assert nat.err == got + "\n"
    else:
        assert nat.err == py.err


# --- beacon ---------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def beacon_inputs(tmp_path_factory):
    d = tmp_path_factory.mktemp("beacon-in")
    f = {}
    for mode in (0, 1):
        sp = picture.StoredPicture.from_latents(synthetic_full_latents(3 + mode), 0xD1D8, mode)
        picture.save_qrsp(d / f"m{mode}.qrsp", sp)
        f[f"m{mode}"] = d / f"m{mode}.qrsp"
    (d / "empty.qrsp").write_bytes(b"")
    f["empty"] = d / "empty.qrsp"
    (d / "garbage.qrsp").write_bytes(b"this is not a stored picture\n")
    f["garbage"] = d / "garbage.qrsp"
    np.save(d / "array.npy", np.arange(4))
    f["npy"] = d / "array.npy"
    with open(d / "nosegs.qrsp", "wb") as fh:
        np.savez(fh, version=np.int64(picture.QRSP_VERSION), mode=np.int64(0), codec_id=np.int64(1),
                 picture_id=np.int64(0))
    f["nosegs"] = d / "nosegs.qrsp"
    (d / "badzip.qrsp").write_bytes(b"PK\x03\x04" + bytes(range(256)) * 4)
    f["badzip"] = d / "badzip.qrsp"
    sp = picture.StoredPicture.from_latents(synthetic_full_latents(9), 0x1234, 0)
    with open(d / "wrongid.qrsp", "wb") as fh:
        np.savez(fh, version=np.int64(picture.QRSP_VERSION), mode=np.int64(0), codec_id=np.int64(0x1234),
                 segs=np.stack(sp.segs).astype(np.float16), picture_id=np.int64(sp.picture_id ^ 1))
    f["wrongid"] = d / "wrongid.qrsp"
    return f


def _beacon_cases(f) -> list[Case]:
    pic = {"in.qrsp": f["m1"]}

    def ok(name, *args, files=None):
        return Case(name, "qrss_beacon.py", "beacon", list(args), files or pic, output="out.bin")

    def bad_input(name, key, only_class=False):
        return Case(name, "qrss_beacon.py", "beacon", ["in.qrsp", "out.bin", "--callsign", "K1ABC"],
                    {"in.qrsp": f[key]}, output="out.bin", only_class=only_class)

    return [
        ok("seg0-grid", "in.qrsp", "out.bin", "--callsign", "K1ABC", "--grid", "FN42"),
        ok("seg1-ook", "in.qrsp", "out.bin", "--callsign", "K1ABC/P", "--segment", "1", "--ook"),
        ok("no-grid", "in.qrsp", "out.bin", "--callsign", "W1AW"),
        ok("mode-a", "in.qrsp", "out.bin", "--callsign", "K1ABC", files={"in.qrsp": f["m0"]}),
        ok("abbrev", "--call", "K1ABC", "--seg", "1", "--gr", "FN42", "--o", "in.qrsp", "out.bin"),
        ok("equals", "in.qrsp", "out.bin", "--callsign=K1ABC", "--segment=1", "--grid=FN42"),
        ok("dashdash", "--callsign", "K1ABC", "--", "in.qrsp", "out.bin"),
        ok("seg-space", "in.qrsp", "out.bin", "--callsign", "K1ABC", "--segment", " 1 "),
        ok("seg-plus", "in.qrsp", "out.bin", "--callsign", "K1ABC", "--segment", "+1"),
        ok("repeat", "in.qrsp", "out.bin", "--callsign", "AA1", "--callsign", "K1ABC", "--ook", "--ook"),
        ok("lowercase", "in.qrsp", "out.bin", "--callsign", "k1abc", "--grid", "fn42"),
        ok("seg-2", "in.qrsp", "out.bin", "--callsign", "K1ABC", "--segment", "2"),
        ok("seg-neg", "in.qrsp", "out.bin", "--callsign", "K1ABC", "--segment", "-1"),
        ok("seg-neg-eq", "in.qrsp", "out.bin", "--callsign", "K1ABC", "--segment=-1"),
        ok("seg-under", "in.qrsp", "out.bin", "--callsign", "K1ABC", "--segment", "1_0"),
        ok("seg-huge", "in.qrsp", "out.bin", "--callsign", "K1ABC", "--segment", "99999999999999999999999"),
        ok("seg-hugeneg", "in.qrsp", "out.bin", "--callsign", "K1ABC", "--segment", "-99999999999999999999"),
        ok("seg-x", "in.qrsp", "out.bin", "--callsign", "K1ABC", "--segment", "x"),
        ok("seg-float", "in.qrsp", "out.bin", "--callsign", "K1ABC", "--segment", "-1.5"),
        ok("seg-missing", "in.qrsp", "out.bin", "--callsign", "K1ABC", "--segment"),
        ok("call-bad", "in.qrsp", "out.bin", "--callsign", "K1ABC!"),
        ok("call-long", "in.qrsp", "out.bin", "--callsign", "K1ABCDEFG"),
        ok("call-empty", "in.qrsp", "out.bin", "--callsign", ""),
        ok("call-space", "in.qrsp", "out.bin", "--callsign", "K1 ABC"),
        ok("call-dash", "in.qrsp", "out.bin", "--callsign", "-K1"),
        ok("grid-bad", "in.qrsp", "out.bin", "--callsign", "K1ABC", "--grid", "XX"),
        ok("grid-empty", "in.qrsp", "out.bin", "--callsign", "K1ABC", "--grid", ""),
        ok("no-callsign", "in.qrsp", "out.bin"),
        ok("no-args"),
        ok("one-arg", "in.qrsp", "--callsign", "K1ABC"),
        ok("ook-value", "in.qrsp", "out.bin", "--callsign", "K1ABC", "--ook=yes"),
        ok("ook-empty", "in.qrsp", "out.bin", "--callsign", "K1ABC", "--ook="),
        ok("call-eq-empty", "in.qrsp", "out.bin", "--callsign="),
        ok("help-eq", "in.qrsp", "-h=x"),
        ok("help-dash", "-h-x", "in.qrsp"),
        ok("dashdash-twice", "--callsign", "K1ABC", "--", "in.qrsp", "--", "out.bin"),
        ok("value-after-dashdash", "in.qrsp", "--callsign", "--", "K1ABC", "out.bin"),
        ok("unknown", "in.qrsp", "out.bin", "--callsign", "K1ABC", "--bogus"),
        ok("unknown-eq", "in.qrsp", "out.bin", "--callsign", "K1ABC", "--bogus=1", "-x"),
        ok("surplus", "in.qrsp", "out.bin", "extra", "--callsign", "K1ABC"),
        ok("missing-and-unknown", "in.qrsp", "--bogus"),
        ok("out-nodir", "in.qrsp", "nodir/out.bin", "--callsign", "K1ABC"),
        Case("out-isdir", "qrss_beacon.py", "beacon", ["in.qrsp", "outdir", "--callsign", "K1ABC"], pic,
             dirs=["outdir"]),
        ok("in-missing", "nope.qrsp", "out.bin", "--callsign", "K1ABC"),
        bad_input("in-empty", "empty"),
        bad_input("in-garbage", "garbage"),
        bad_input("in-npy", "npy"),
        bad_input("in-nosegs", "nosegs"),
        bad_input("in-badzip", "badzip", only_class=True),
        bad_input("in-wrongid", "wrongid"),
    ]


@pytest.fixture(scope="module")
def beacon_runs(cli, beacon_inputs, tmp_path_factory):
    cases = _beacon_cases(beacon_inputs)
    return {c.name: c for c in cases}, run_cases(cases, tmp_path_factory.mktemp("beacon-runs"), cli)


@pytest.mark.parametrize("name", [c.name for c in _beacon_cases({k: Path("x") for k in (
    "m0", "m1", "empty", "garbage", "npy", "nosegs", "badzip", "wrongid")})])
def test_beacon_matches_the_script(beacon_runs, name):
    cases, runs = beacon_runs
    py, nat = runs[name]
    check(cases[name], py, nat)


def test_beacon_cases_cover_each_ending(beacon_runs):
    """The comparison above is only as good as its cases: every kind of
    ending must be among them."""
    _, runs = beacon_runs
    rcs = {name: py.rc for name, (py, _) in runs.items()}
    assert rcs["seg0-grid"] == 0 and runs["seg0-grid"][0].output is not None
    assert runs["seg0-grid"][0].out.startswith("wrote out.bin: picture ")
    assert rcs["no-args"] == 2 and rcs["seg-x"] == 2
    assert rcs["seg-2"] == 1 and "Traceback" not in runs["seg-2"][0].err
    assert rcs["in-missing"] == 1 and "FileNotFoundError" in runs["in-missing"][0].err


def test_beacon_help(cli):
    r = subprocess.run([cli, "beacon", "--help"], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0 and r.stderr == ""
    assert r.stdout.startswith("usage: sstvae-qrss beacon [-h] --callsign CALLSIGN [--grid GRID] "
                               "[--segment SEGMENT] [--ook] input output\n")
    for flag in ("--callsign", "--grid", "--segment", "--ook", "input", "output"):
        assert flag in r.stdout


def test_dispatch(cli):
    r = subprocess.run([cli], capture_output=True, text=True, timeout=60)
    assert r.returncode == 2 and r.stdout == "" and "beacon" in r.stderr
    r = subprocess.run([cli, "--help"], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0 and "beacon" in r.stdout and "qrss_beacon.py" in r.stdout
    r = subprocess.run([cli, "frobnicate"], capture_output=True, text=True, timeout=60)
    assert r.returncode == 2 and r.stderr.startswith("sstvae-qrss: unknown command 'frobnicate'\n")
    r = subprocess.run([cli, "transmit"], capture_output=True, text=True, timeout=60)
    if r.returncode == 2 and "not in this build yet" in r.stderr:
        assert r.stderr == "sstvae-qrss transmit: not in this build yet; use qrss_transmit.py\n"


# --- encode ---------------------------------------------------------------------------------

def test_encode_without_the_codec_says_so(cli):
    if _has_encode(cli):
        pytest.skip("this build has the codec")
    r = subprocess.run([cli, "encode", "a.png", "b.qrsp"], capture_output=True, text=True, timeout=60)
    assert r.returncode == 2 and r.stdout == ""
    assert r.stderr == ("sstvae-qrss encode: built without the codec (SSTVAE_BUILD_CODEC=OFF); use "
                        "qrss_encode.py, or rebuild with the codec\n")


@pytest.fixture(scope="module")
def encode_inputs(tmp_path_factory):
    from PIL import Image

    v5 = fixtures.v5_dir()
    d = tmp_path_factory.mktemp("encode-in")
    Image.fromarray(fixtures.scene()).save(d / "scene.png")
    (d / "bad.png").write_bytes(b"not a picture")
    (d / "x-encoder-fp16.onnx").write_bytes(fixtures.tiny_onnx(None))
    (d / "g-encoder-fp16.onnx").write_bytes(b"garbage, not a protobuf")
    files = {"scene.png": d / "scene.png", "bad.png": d / "bad.png"}
    for prec in ("fp16", "fp32"):
        for part in ("encoder", "decoder"):
            src = next(v5.glob(f"*-{part}-{prec}.onnx"))
            files[f"m/{src.name}"] = src
    files["dir.pt/v5-encoder-fp16.onnx"] = files["m/v5-encoder-fp16.onnx"]
    files["nometa/x-encoder-fp16.onnx"] = d / "x-encoder-fp16.onnx"
    files["garb/g-encoder-fp16.onnx"] = d / "g-encoder-fp16.onnx"
    return files


def _encode_cases(f) -> list[Case]:
    def c(name, *args, only_class=False, dirs=()):
        return Case(name, "qrss_encode.py", "encode", list(args), f, list(dirs), output="out.qrsp",
                    only_class=only_class)

    return [
        c("mode-a", "scene.png", "out.qrsp", "--model", "m"),
        c("mode-b", "scene.png", "out.qrsp", "--model", "m", "--mode", "B"),
        c("mode-c-fp32", "scene.png", "out.qrsp", "--model", "m/", "--mode", "C", "--precision", "fp32"),
        c("onnx-file", "scene.png", "out.qrsp", "--model", "m/v5-decoder-fp16.onnx", "--mode=B"),
        c("codec-id", "scene.png", "out.qrsp", "--model", "m", "--codec-id", "BEEF"),
        c("codec-id-0x", "scene.png", "out.qrsp", "--model", "m", "--codec-id", "0x_1_2"),
        c("codec-id-bad", "scene.png", "out.qrsp", "--model", "m", "--codec-id", "xyz"),
        c("codec-id-big", "scene.png", "out.qrsp", "--model", "m", "--codec-id", "10000"),
        c("codec-id-neg", "scene.png", "out.qrsp", "--model", "m", "--codec-id", "-1"),
        c("codec-id-empty", "scene.png", "out.qrsp", "--model", "m", "--codec-id", ""),
        c("ambiguous", "scene.png", "out.qrsp", "--mod", "m"),
        c("ambiguous-eq", "scene.png", "out.qrsp", "--m=B"),
        c("prefix", "scene.png", "out.qrsp", "--mode", "B", "--prec", "fp16", "--model", "m", "--co", "d1d8"),
        c("precision-bad", "scene.png", "out.qrsp", "--precision", "fp8"),
        c("mode-bad", "scene.png", "out.qrsp", "--mode", "D"),
        c("no-output", "scene.png"),
        c("image-missing", "nope.png", "out.qrsp", "--model", "m"),
        c("image-dir", "m", "out.qrsp", "--model", "m"),
        c("model-nowhere", "scene.png", "out.qrsp", "--model", "nowhere"),
        c("model-empty", "scene.png", "out.qrsp", "--model", "emptydir", dirs=["emptydir"]),
        c("model-int8", "scene.png", "out.qrsp", "--model", "m", "--precision", "int8"),
        c("model-nometa", "scene.png", "out.qrsp", "--model", "nometa"),
        c("model-garbage", "scene.png", "out.qrsp", "--model", "garb/g-encoder-fp16.onnx"),
        c("model-pt", "scene.png", "out.qrsp", "--model", "x.pt"),
        c("model-pt-id", "scene.png", "out.qrsp", "--model", "x.pt", "--codec-id", "d1d8"),
        c("model-pt-dir", "scene.png", "out.qrsp", "--model", "dir.pt"),
        c("out-nodir", "scene.png", "nodir/out.qrsp", "--model", "m"),
    ]


@pytest.fixture(scope="module")
def encode_runs(cli, encode_inputs, tmp_path_factory):
    if not _has_encode(cli):
        pytest.skip("sstvae-qrss built without the codec")
    cases = _encode_cases(encode_inputs)
    return {c.name: c for c in cases}, run_cases(cases, tmp_path_factory.mktemp("encode-runs"), cli)


@pytest.mark.codec
@pytest.mark.parametrize("name", [c.name for c in _encode_cases({})])
def test_encode_matches_the_script(encode_runs, name):
    cases, runs = encode_runs
    py, nat = runs[name]
    check(cases[name], py, nat)


@pytest.mark.codec
def test_encode_cases_cover_each_ending(encode_runs):
    _, runs = encode_runs
    for name in ("mode-a", "mode-b", "mode-c-fp32", "onnx-file", "codec-id"):
        py, _ = runs[name]
        assert py.rc == 0 and py.output is not None, (name, py)
    assert runs["mode-a"][0].out.endswith(", codec d1d8, picture ID " + runs["mode-a"][0].out[-9:])
    assert "(3 segments)" in runs["mode-c-fp32"][0].out
    assert runs["codec-id-bad"][0].rc == 1 and "ValueError: invalid literal" in runs["codec-id-bad"][0].err
    assert runs["ambiguous"][0].rc == 2 and "could match --model, --mode" in runs["ambiguous"][0].err
    assert runs["model-nometa"][0].rc == 1 and "carries no sstvae.source_sha256" in runs["model-nometa"][0].err


@pytest.mark.codec
def test_encode_an_unreadable_picture(cli, encode_inputs, tmp_path):
    """Both refuse it with "<image>: <the image library's reason>" and exit
    1; the reasons are PIL's and the native image layer's, so only the
    form is compared."""
    if not _has_encode(cli):
        pytest.skip("sstvae-qrss built without the codec")
    case = Case("bad-image", "qrss_encode.py", "encode", ["bad.png", "out.qrsp", "--model", "m"],
                encode_inputs, output="out.qrsp")
    py, nat = run_cases([case], tmp_path, cli)["bad-image"]
    assert py.rc == nat.rc == 1 and py.out == nat.out == ""
    assert py.output is None and nat.output is None
    assert py.err.startswith("bad.png: ") and nat.err.startswith("bad.png: ") and "Traceback" not in nat.err


# --- int() and float() ---------------------------------------------------------------------

INTS = [
    "", " ", "0", "-0", "+5", "  12  ", "\t\n7\x0b\x0c\r", "\x1c5\x1f", "1_000", "1__0", "_1", "1_",
    "0x1f", "0X_1F", "0x", "0x_", "0b1", "1e3", "007", "0_0", "+", "-", "+-1", "1 2", "1.0", "ff", "FF",
    "-0x10", "+0x_f", "0x__f", "9223372036854775807", "9223372036854775808", "-9223372036854775808",
    "-9223372036854775809", "99999999999999999999999999999999", "0xffffffffffffffffffff", "x" * 300,
    "'quoted'", 'a"b', "it's", "back\\slash", "tab\there",
]


@pytest.mark.parametrize("base", [10, 16])
@pytest.mark.parametrize("text", INTS, ids=repr)
def test_py_int_matches_int(native, text, base):
    n = require(native, "pynum")
    try:
        want = int(text, base)
    except ValueError as e:
        assert n.py_int(text, base) is None
        assert n.invalid_int_literal(text, base) == str(e)
        return
    decimal, value = n.py_int(text, base)
    assert decimal == str(want)
    assert value == (want if -(1 << 63) <= want < (1 << 63) else None)


FLOATS = [
    "", " ", "0", "-0", "+1.5", " 2.5e3 ", "1_0.5e1", "1__0", "_1", "1_", "1._5", "1_.5", "1e_5",
    "1e5_0", ".5", "5.", ".", "e5", "1e", "1e+", "1e-3", "inf", "-Infinity", "+INF", "nan", "-NaN",
    "infinit", "nan(1)", "0x1p3", "1,5", "1.5.2", "1e400", "-1e400", "1e-400", "4.9e-324",
    "2.2250738585072011e-308", "0.1", "123456789012345678901234567890", "\t3\n", "1 0", "\x1c1.5",
    "1.5\x1f", "\x0b1.5\x0c",
]


@pytest.mark.parametrize("text", FLOATS, ids=repr)
def test_py_float_matches_float(native, text):
    n = require(native, "pynum")
    try:
        want = float(text)
    except ValueError:
        assert n.py_float(text) is None
        return
    got = n.py_float(text)
    assert got is not None
    if np.isnan(want):
        assert np.isnan(got) and np.signbit(got) == np.signbit(want)
    else:
        assert got == want and np.signbit(got) == np.signbit(want)

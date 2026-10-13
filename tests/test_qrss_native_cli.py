"""`sstvae-qrss encode`, `beacon` and `transmit` (native/apps/sstvae_qrss)
against qrss_encode.py, qrss_beacon.py and qrss_transmit.py, run side by
side, and the C++ chain into Python's receiver (C++ encode and transmit,
then sstvae.qrss.receiver: plan item 19).

Each case runs the script and the subcommand with the same arguments, in
two directories holding the same input files, and compares:

- the exit status;
- stdout, exactly (the "wrote ..." line);
- the output file, byte for byte, or that neither run left one (a
  transmit WAV: its header byte for byte and its samples as the tx
  parity tests hold them -- see _same_wav);
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
    audio: str | None = None         # a transmit WAV: "ce" or "l" (see _same_wav)


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
    if case.audio is not None:
        _same_wav(case.audio, py.output, nat.output)
    else:
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
    # Scalars int() cannot read: numpy's TypeError, OverflowError and
    # Python's TypeError for a complex, as the tracebacks name them.
    good = picture.StoredPicture.from_latents(synthetic_full_latents(4), 0xD1D8, 1)
    fields = dict(version=np.int64(picture.QRSP_VERSION), mode=np.int64(1), codec_id=np.int64(0xD1D8),
                  segs=np.stack(good.segs).astype(np.float16), picture_id=np.int64(good.picture_id))
    for key, field_name, value in (("v1d", "version", np.array([1, 1])), ("vinf", "version", np.float64("inf")),
                                   ("vcplx", "version", np.complex128(1)), ("mode1d", "mode", np.array([1])),
                                   ("vfloat", "version", np.float64(1.5))):
        with open(d / f"{key}.qrsp", "wb") as fh:
            np.savez(fh, **{**fields, field_name: value})
        f[key] = d / f"{key}.qrsp"
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
        bad_input("in-version-1d", "v1d"),
        bad_input("in-version-inf", "vinf"),
        bad_input("in-version-complex", "vcplx"),
        bad_input("in-mode-1d", "mode1d"),
        bad_input("in-version-float", "vfloat"),
        Case("in-isdir", "qrss_beacon.py", "beacon", ["indir", "out.bin", "--callsign", "K1ABC"], {},
             dirs=["indir"], output="out.bin"),
        # argparse drops '--' only where a positional takes it.
        ok("dashdash-trailing", "in.qrsp", "out.bin", "--callsign", "K1ABC", "--"),
        ok("dashdash-trailing-value", "in.qrsp", "out.bin", "--callsign", "K1ABC", "--", "x"),
        ok("dashdash-before-last", "in.qrsp", "--callsign", "K1ABC", "--", "out.bin"),
        ok("dashdash-after-last", "--callsign", "K1ABC", "in.qrsp", "out.bin", "--"),
        ok("dashdash-only-after-option", "in.qrsp", "--callsign", "K1ABC", "--"),
        ok("dash-equals", "in.qrsp", "out.bin", "--callsign", "K1ABC", "-=x"),
        ok("seg-newline", "in.qrsp", "out.bin", "--callsign", "K1ABC", "--segment", "-1\n"),
        ok("call-unicode", "in.qrsp", "out.bin", "--callsign", "K\u00c4B"),
        ok("call-trailing-space", "in.qrsp", "out.bin", "--callsign", "K1ABC "),
    ]


@pytest.fixture(scope="module")
def beacon_runs(cli, beacon_inputs, tmp_path_factory):
    cases = _beacon_cases(beacon_inputs)
    return {c.name: c for c in cases}, run_cases(cases, tmp_path_factory.mktemp("beacon-runs"), cli)


@pytest.mark.parametrize("name", [c.name for c in _beacon_cases({k: Path("x") for k in (
    "m0", "m1", "empty", "garbage", "npy", "nosegs", "badzip", "wrongid", "v1d", "vinf", "vcplx", "mode1d",
    "vfloat")})])
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
    assert rcs["dashdash-trailing"] == 2 and runs["dashdash-trailing"][0].err.endswith("arguments: --\n")
    assert rcs["dashdash-after-last"] == 0 and rcs["dashdash-before-last"] == 0
    assert "TypeError: only 0-dimensional" in runs["in-version-1d"][0].err
    assert "OverflowError" in runs["in-version-inf"][0].err


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
    r = subprocess.run([cli, "--help"], capture_output=True, text=True, timeout=60)
    assert any(line.split()[:1] == ["transmit"] and "qrss_transmit.py" in line for line in r.stdout.splitlines())
    r = subprocess.run([cli, "receive"], capture_output=True, text=True, timeout=60)
    assert r.returncode == 2 and r.stderr == "sstvae-qrss receive: not in this build yet; use qrss_receive.py\n"


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
        c("model-missing-encoder", "scene.png", "out.qrsp", "--model", "nope-encoder-fp16.onnx"),
        c("model-missing-sibling", "scene.png", "out.qrsp", "--model", "nope-decoder-fp16.onnx"),
        c("model-decoder-fp32", "scene.png", "out.qrsp", "--model", "m/v5-decoder-fp16.onnx", "--precision", "fp32"),
        c("model-not-an-encoder", "scene.png", "out.qrsp", "--model", "nometa", "--codec-id", "1234"),
        c("model-garbage-id", "scene.png", "out.qrsp", "--model", "garb", "--codec-id", "12"),
        c("codec-id-spaces", "scene.png", "out.qrsp", "--model", "m", "--codec-id", " +0x_d1d8 "),
        c("image-slash", "scene.png/", "out.qrsp", "--model", "m"),
        c("dashdash-trailing", "scene.png", "out.qrsp", "--model", "m", "--"),
        # The glob takes a directory so named: codec_id_from_onnx then looks
        # inside it and raises FileNotFoundError (the native aborted here).
        c("model-encoder-is-dir", "scene.png", "out.qrsp", "--model", "dd", dirs=["dd/v9-encoder-fp16.onnx"]),
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


# --- transmit -------------------------------------------------------------------------------
#
# The WAVs: the header byte for byte, and the samples as the tx parity
# tests hold the audio (tests/test_qrss_native_tx_audio.py) -- waveform CE
# bit for bit where numpy's trig and complex multiply are the C library's
# and fused (this platform; EXACT_CE), else within 1e-12; L within 1e-9,
# the clipper's FFT rounding. So a --float file is byte-identical, or (L,
# or CE elsewhere) differs by at most the float64 tolerance plus one
# float32 rounding; an int16 file is normalised by the pass's own peak and
# rounded, so it may differ by one LSB where a sample sits on a rounding
# boundary. Measured here: every CE file byte-identical; L int16
# byte-identical; L --float within one float32 ulp on a few samples.

from test_qrss_native_tx_audio import EXACT_CE  # noqa: E402

SLOT = "2025-10-01T00:00Z"   # Q_TEST


def _wav_parts(data: bytes) -> tuple[bytes, np.ndarray]:
    """(the bytes before the samples, the samples) of scipy's mono WAV."""
    i = data.index(b"data") + 8
    tag = int.from_bytes(data[20:22], "little")
    return data[:i], np.frombuffer(data[i:], np.int16 if tag == 1 else np.float32)


def _same_wav(kind: str, py: bytes | None, nat: bytes | None):
    assert (py is None) == (nat is None), "only one run wrote the WAV"
    if py is None or py == nat:
        return
    assert not (kind == "ce" and EXACT_CE), "a CE WAV differs where the audio is bit for bit"
    hp, xp = _wav_parts(py)
    hn, xn = _wav_parts(nat)
    assert hn == hp and xn.shape == xp.shape, "WAV headers differ"
    if xp.dtype == np.int16:
        assert np.max(np.abs(xn.astype(np.int32) - xp)) <= 1
    else:
        tol = (1e-12 if kind == "ce" else 1e-9) + np.spacing(np.abs(xp)).astype(np.float64)
        assert np.all(np.abs(xn.astype(np.float64) - xp) <= tol)


@pytest.fixture(scope="module")
def transmit_inputs(beacon_inputs, tmp_path_factory):
    from sstvae.qrss import beaconfile
    from sstvae.qrss.header import HeaderFields

    d = tmp_path_factory.mktemp("transmit-in")
    f = dict(beacon_inputs)
    sp = picture.load_qrsp(beacon_inputs["m1"])
    h = HeaderFields(callsign="K1ABC", grid="FN42", picture_id=sp.picture_id, mode=sp.mode, segment=1,
                     codec_id=sp.codec_id)
    for key, ook in (("fsk", False), ("ook", True)):
        beaconfile.write(d / f"{key}.bin", beaconfile.from_picture(sp, 1, h, ook=ook))
        f[key] = d / f"{key}.bin"
    data = bytearray((d / "fsk.bin").read_bytes())
    data[-1] ^= 0xFF
    (d / "badcrc.bin").write_bytes(bytes(data))
    f["badcrc"] = d / "badcrc.bin"
    (d / "short.bin").write_bytes(beaconfile.MAGIC + bytes(12))
    f["shortbin"] = d / "short.bin"
    (d / "three.bin").write_bytes(beaconfile.MAGIC[:3])
    f["three"] = d / "three.bin"
    return f


def _transmit_cases(f) -> list[Case]:
    pic = {"in.qrsp": f["m1"]}

    def tx(name, *args, files=None, audio="ce", only_class=False, dirs=()):
        return Case(name, "qrss_transmit.py", "transmit", list(args), pic if files is None else files, list(dirs),
                    output="out.wav", only_class=only_class, audio=audio)

    def q(name, *extra, audio="ce", files=None, slot=SLOT):
        """in.qrsp out.wav --slot SLOT --callsign K1ABC, then `extra`."""
        return tx(name, "in.qrsp", "out.wav", "--slot", slot, "--callsign", "K1ABC", *extra, audio=audio,
                  files=files)

    def b(name, key, *extra, audio="ce"):
        return tx(name, "in.bin", "out.wav", "--slot", SLOT, *extra, files={"in.bin": f[key]}, audio=audio)

    def bad_input(name, key, only_class=False):
        return tx(name, "in.qrsp", "out.wav", "--slot", SLOT, "--callsign", "K1ABC", "--frame", "tiny",
                  files={"in.qrsp": f[key]}, only_class=only_class)

    return [
        # what gets written
        q("ce-tiny-float", "--frame", "tiny", "--float"),
        q("ce-tiny-int16", "--frame", "tiny"),
        q("l-tiny-float", "--frame", "tiny", "--waveform", "L", "--float", audio="l"),
        q("l-tiny-int16", "--frame", "tiny", "--waveform", "L", audio="l"),
        q("ce-short-seg1", "--frame", "short", "--grid", "FN42", "--segment", "1", "--float"),
        q("ce-short-ook", "--frame", "short", "--ook"),
        q("l-short-float", "--frame", "short", "--waveform", "L", "--grid", "fn42", "--float", audio="l"),
        q("ce-medium", "--frame", "medium", "--float"),
        q("ce-full-int16"),
        q("l-full-float", "--waveform", "L", "--float", audio="l"),
        q("mode-a", "--frame", "tiny", "--float", files={"in.qrsp": f["m0"]}),
        b("bin-fsk", "fsk", "--frame", "short", "--float"),
        b("bin-ook", "ook", "--frame", "short"),
        b("bin-same-call", "fsk", "--callsign", "K1ABC", "--frame", "tiny", "--float"),
        b("bin-segment-ignored", "fsk", "--segment", "7", "--grid", "XX", "--frame", "tiny", "--float"),
        b("bin-l", "fsk", "--waveform", "L", "--frame", "short", "--float", audio="l"),
        b("bin-ook-flag", "ook", "--ook", "--frame", "tiny", "--float"),
        q("lead-in", "--frame", "tiny", "--lead-in", "2.5", "--float"),
        q("lead-in-l", "--frame", "tiny", "--lead-in", "10", "--waveform", "L", "--float", audio="l"),
        q("lead-in-negzero", "--frame", "tiny", "--lead-in=-0", "--float"),
        q("lead-in-tiny", "--frame", "tiny", "--lead-in", "1e-7", "--float"),
        q("freq-decimal", "--frame", "tiny", "--freq", "1500.3", "--float"),
        q("freq-exp", "--frame", "tiny", "--freq", "1.2345e3", "--float"),
        q("freq-l", "--frame", "tiny", "--freq", "700", "--waveform", "L", "--float", audio="l"),
        q("freq-edge", "--frame", "tiny", "--freq", "2700", "--float"),
        q("slot-offset", "--frame", "tiny", "--float", slot="2025-10-01T02:00+02:00"),
        q("slot-lower", "--frame", "tiny", "--float", slot=" 2025-10-01t00:00z "),
        q("slot-week", "--frame", "tiny", "--float", slot="2025-W40-3T00:00Z"),
        q("slot-badbyte-sep", "--frame", "tiny", "--float", slot="2025-10-01\udcff00:00Z"),
        tx("abbrev", "--wave", "L", "--fra", "tiny", "--call", "K1ABC", "--sl", SLOT, "--fl", "in.qrsp", "out.wav",
           audio="l"),
        tx("ambiguous", "--fr", "tiny", "--callsign", "K1ABC", "--slot", SLOT, "in.qrsp", "out.wav"),
        tx("equals", "in.qrsp", "out.wav", "--slot=" + SLOT, "--callsign=K1ABC", "--frame=tiny", "--float"),
        # refusals: --slot
        q("slot-off", "--frame", "tiny", slot="2025-10-01T00:07Z"),
        q("slot-seconds", "--frame", "tiny", slot="2025-10-01T00:00:00.5Z"),
        q("slot-naive", "--frame", "tiny", slot="2025-10-01T00:00"),
        q("slot-garbage", "--frame", "tiny", slot="tomorrow"),
        q("slot-quote", "--frame", "tiny", slot="it's"),
        q("slot-badbyte-end", "--frame", "tiny", slot="2025-10-01T00:00Z\udcff"),
        q("slot-1969", "--frame", "tiny", slot="1969-12-31T23:45Z"),
        tx("slot-1969-callsign", "in.qrsp", "out.wav", "--slot", "1969-12-31T23:45Z", "--callsign", "K1ABC!"),
        tx("slot-1969-bin-l", "in.bin", "out.wav", "--slot", "1969-12-31T23:45Z", "--waveform", "L",
           files={"in.bin": f["ook"]}),
        tx("slot-missing", "in.qrsp", "out.wav", "--callsign", "K1ABC"),
        tx("slot-before-file", "nope.qrsp", "out.wav", "--slot", "x"),
        # --ook with L, before the input is opened
        q("l-ook", "--waveform", "L", "--ook"),
        tx("l-ook-no-input", "nope.qrsp", "out.wav", "--slot", SLOT, "--waveform", "L", "--ook"),
        # a stored picture's header
        tx("no-callsign", "in.qrsp", "out.wav", "--slot", SLOT),
        q("grid-bad", "--frame", "tiny", "--grid", "XX"),
        q("grid-empty", "--frame", "tiny", "--grid", ""),
        tx("call-bad", "in.qrsp", "out.wav", "--slot", SLOT, "--callsign", "K1ABC!"),
        tx("call-long", "in.qrsp", "out.wav", "--slot", SLOT, "--callsign", "K1ABCDEFG"),
        tx("call-lower", "in.qrsp", "out.wav", "--slot", SLOT, "--callsign", "k1abc", "--frame", "tiny"),
        tx("call-empty", "in.qrsp", "out.wav", "--slot", SLOT, "--callsign", ""),
        tx("call-unicode", "in.qrsp", "out.wav", "--slot", SLOT, "--callsign", "KÄB"),
        q("seg-2", "--segment", "2"),
        q("seg-neg", "--segment=-1"),
        q("seg-huge", "--segment", "99999999999999999999999"),
        q("seg-huge-freq", "--segment", "99999999999999999999999", "--freq", "100"),
        q("seg-huge-l-pilot", "--segment", "-99999999999999999999999", "--freq", "310", "--waveform", "L"),
        tx("seg-huge-no-call", "in.qrsp", "out.wav", "--slot", SLOT, "--segment", "99999999999999999999999"),
        q("seg-x", "--segment", "x"),
        # the carrier and the lead-in
        q("freq-low", "--freq", "100"),
        q("freq-high", "--freq", "2700.5"),
        q("freq-nan", "--freq", "nan"),
        q("freq-inf", "--freq", "-inf"),
        q("freq-l-low", "--freq", "310", "--waveform", "L"),
        q("freq-text", "--freq", "1500Hz"),
        q("lead-in-11", "--lead-in", "11"),
        q("lead-in-neg", "--lead-in=-1"),
        q("lead-in-nan", "--lead-in", "nan"),
        b("bin-freq-low", "fsk", "--freq", "299.9"),
        # a beacon file
        b("bin-other-call", "fsk", "--callsign", "W1AW"),
        b("bin-empty-call", "fsk", "--callsign", ""),
        b("bin-ook-fsk", "fsk", "--ook"),
        b("bin-ook-l", "ook", "--waveform", "L", "--frame", "tiny"),
        b("bin-badcrc", "badcrc"),
        b("bin-short", "shortbin"),
        b("bin-three", "three"),
        # the command line
        q("frame-bad", "--frame", "huge"),
        q("waveform-bad", "--waveform", "ce"),
        tx("no-args"),
        q("unknown", "--bogus"),
        # the input and the output
        tx("in-missing", "nope.qrsp", "out.wav", "--slot", SLOT, "--callsign", "K1ABC"),
        tx("in-isdir", "indir", "out.wav", "--slot", SLOT, "--callsign", "K1ABC", files={}, dirs=["indir"]),
        Case("out-nodir", "qrss_transmit.py", "transmit",
             ["in.qrsp", "nodir/out.wav", "--slot", SLOT, "--callsign", "K1ABC", "--frame", "tiny"], pic,
             output="nodir/out.wav"),
        Case("out-isdir", "qrss_transmit.py", "transmit",
             ["in.qrsp", "outdir", "--slot", SLOT, "--callsign", "K1ABC", "--frame", "tiny"], pic, dirs=["outdir"]),
        bad_input("in-empty", "empty"),
        bad_input("in-garbage", "garbage"),
        bad_input("in-npy", "npy"),
        bad_input("in-nosegs", "nosegs"),
        bad_input("in-badzip", "badzip", only_class=True),
        bad_input("in-wrongid", "wrongid"),
        bad_input("in-version-1d", "v1d"),
        bad_input("in-version-inf", "vinf"),
        bad_input("in-version-complex", "vcplx"),
        bad_input("in-mode-1d", "mode1d"),
        bad_input("in-version-float", "vfloat"),
    ]


_TRANSMIT_KEYS = ("m0", "m1", "empty", "garbage", "npy", "nosegs", "badzip", "wrongid", "v1d", "vinf", "vcplx",
                  "mode1d", "vfloat", "fsk", "ook", "badcrc", "shortbin", "three")


@pytest.fixture(scope="module")
def transmit_runs(cli, transmit_inputs, tmp_path_factory):
    cases = _transmit_cases(transmit_inputs)
    return {c.name: c for c in cases}, run_cases(cases, tmp_path_factory.mktemp("transmit-runs"), cli)


@pytest.mark.parametrize("name", [c.name for c in _transmit_cases({k: Path("x") for k in _TRANSMIT_KEYS})])
def test_transmit_matches_the_script(transmit_runs, name):
    cases, runs = transmit_runs
    py, nat = runs[name]
    check(cases[name], py, nat)


def test_transmit_cases_cover_each_ending(transmit_runs):
    _, runs = transmit_runs
    for name in ("ce-tiny-float", "l-tiny-int16", "ce-full-int16", "l-full-float", "bin-fsk", "bin-l",
                 "slot-badbyte-sep", "abbrev"):
        py, _ = runs[name]
        assert py.rc == 0 and py.output is not None and py.out.startswith("wrote out.wav: slot q="), (name, py)
    assert "2.5 s lead-in" in runs["lead-in"][0].out and "lead-in" not in runs["lead-in-negzero"][0].out
    assert "(OOK callsign windows)" in runs["bin-ook"][0].out and "pilot 700 Hz" in runs["freq-l"][0].out
    assert "carrier 1500.3 Hz" in runs["freq-decimal"][0].out
    refused = {"slot-off": "is not on a quarter hour", "slot-naive": "give the time zone",
               "slot-1969": "q must fit in uint64, not -1", "l-ook": "--ook is waveform CE's",
               "no-callsign": "a .qrsp needs --callsign", "seg-huge": "segment must be 0..1, not 9999",
               "seg-huge-freq": "carrier 100.0 Hz is outside", "freq-l-low": "an L pilot at 310.0 Hz",
               "lead-in-11": "lead", "bin-other-call": "the beacon file is for K1ABC, not W1AW",
               "bin-ook-fsk": "keys its callsign windows FSK", "bin-ook-l": "on and off",
               "bin-badcrc": "CRC-32 mismatch", "in-wrongid": "", "l-ook-no-input": "--ook is waveform CE's"}
    for name, text in refused.items():
        py, _ = runs[name]
        assert py.rc == 1 and py.output is None and "Traceback" not in py.err and text in py.err, (name, py)
    for name in ("in-missing", "out-nodir", "in-empty", "in-version-1d"):
        py, _ = runs[name]
        assert py.rc == 1 and "Traceback" in py.err and runs[name][0].output is None, (name, py)
    assert runs["frame-bad"][0].rc == 2 and runs["slot-missing"][0].rc == 2


def test_transmit_help(cli):
    r = subprocess.run([cli, "transmit", "--help"], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0 and r.stderr == ""
    assert r.stdout.startswith("usage: sstvae-qrss transmit [-h] --slot SLOT [--callsign CALLSIGN] [--grid GRID] "
                               "[--segment SEGMENT] [--freq FREQ] [--waveform {CE,L}] [--lead-in LEAD_IN] [--ook] "
                               "[--frame {full,medium,short,tiny}] [--float] input output\n")
    for flag in ("--slot", "--callsign", "--grid", "--segment", "--freq", "--waveform", "--lead-in", "--ook",
                 "--frame", "--float", "(0-10)"):
        assert flag in r.stdout


# --- the C++ chain into Python's receiver (plan item 19) ------------------------------------
#
# sstvae-qrss encode makes the stored picture and sstvae-qrss transmit the
# passes; Python reads both files and receives the audio through its
# channel simulator at -6 dB (what test_qrss_smoke.py does with its own
# transmitter's audio: a noiseless recording sends the receiver after every
# grating lobe). A TINY pass has no header, so the header comes from SHORT
# passes; the picture from what the passes carry, decoded by the v5 codec.

CHAIN_SNR_DB = -6.0


def _receive(wav: Path, spec, q: int, seed: int):
    from sstvae import wavio
    from sstvae.qrss import channel as chm
    from sstvae.qrss import frontend
    from sstvae.qrss import receiver as RX
    from sstvae.qrss.channel import ChannelConfig

    audio = wavio.read_wav(str(wav))
    sim = chm.simulate_audio(audio, spec, q, ChannelConfig(snr_db=CHAIN_SNR_DB, seed=seed))
    return RX.receive_slot(frontend.Capture(sim.fe, q, sim.t0_index), spec)


@pytest.fixture(scope="module")
def chain(cli, encode_inputs, tmp_path_factory):
    """The C++ CLI's stored picture and its passes, CE and L, TINY and SHORT."""
    if not _has_encode(cli):
        pytest.skip("sstvae-qrss built without the codec")
    d = tmp_path_factory.mktemp("chain")
    for name, src in encode_inputs.items():
        (d / name).parent.mkdir(parents=True, exist_ok=True)
        os.symlink(Path(src).resolve(), d / name)

    def run(*args):
        r = subprocess.run([str(cli), *args], cwd=d, capture_output=True, text=True, timeout=180)
        assert r.returncode == 0 and r.stderr == "", r
        return r.stdout

    enc = run("encode", "scene.png", "pic.qrsp", "--model", "m", "--mode", "B")
    wavs = {}
    for wf, frame_name, extra in (("CE", "tiny", ()), ("L", "tiny", ()), ("CE", "short", ("--grid", "FN42")),
                                  ("L", "short", ("--grid", "FN42", "--segment", "1"))):
        out = f"{wf}-{frame_name}.wav"
        run("transmit", "pic.qrsp", out, "--slot", SLOT, "--callsign", "K1ABC", "--waveform", wf, "--frame",
            frame_name, "--float", *extra)
        wavs[wf, frame_name] = d / out
    return d, enc, picture.load_qrsp(d / "pic.qrsp"), wavs


@pytest.mark.codec
@pytest.mark.parametrize("wf", ["CE", "L"])
def test_chain_header(chain, wf):
    """A SHORT pass from the C++ transmitter: Python's receiver reads its
    header (callsign, grid, the C++ encoder's picture ID) and callsign windows."""
    from qrss_helpers import Q_TEST
    from sstvae.qrss import frame, lframe
    from sstvae.qrss.header import HeaderFields

    _, enc, sp, wavs = chain
    assert enc.endswith(f", picture ID {sp.id_hex}\n")
    segment = 1 if wf == "L" else 0
    spec = lframe.get("l-short") if wf == "L" else frame.SHORT
    passes = _receive(wavs[wf, "short"], spec, Q_TEST, seed=3)
    assert len(passes) == 1, [p.f_hz for p in passes]
    p = passes[0]
    assert p.header == HeaderFields(callsign="K1ABC", grid="FN42", picture_id=sp.picture_id, mode=sp.mode,
                                    segment=segment, codec_id=sp.codec_id)
    assert p.cw is not None and p.cw.text == "K1ABC" and p.cw.agrees


@pytest.mark.codec
@pytest.mark.parametrize("wf", ["CE", "L"])
def test_chain_picture(chain, wf, tmp_path):
    """A TINY pass from the C++ transmitter: Python's receiver hears the
    C++ encoder's latents, and the v5 decoder draws from them what it draws
    from the latents themselves."""
    from qrss_helpers import Q_TEST, latent_snr_db, load_codec
    from sstvae.qrss import frame, lframe, render, tx
    from sstvae.qrss.store import Store

    _, _, sp, wavs = chain
    spec = lframe.get("l-tiny") if wf == "L" else frame.TINY
    passes = _receive(wavs[wf, "tiny"], spec, Q_TEST, seed=4)
    assert len(passes) == 1, [p.f_hz for p in passes]
    p = passes[0]
    sent = tx.slot_air_latents(sp, 0)[:spec.n_data]
    snr = latent_snr_db(p.z, sent)
    assert snr > 6.0, snr   # measured 8.7 (CE) and 9.9 dB (L) at -6 dB

    st = Store(tmp_path / "store")
    st.add_pass(p)
    acc = st.attach(("K1ABC", int(sp.picture_id)), p.uid, 0, "test", mode=sp.mode, codec_id=sp.codec_id)
    codec = load_codec("fp16")
    got = np.asarray(render.render(codec, acc.S, acc.W, acc.mode), dtype=np.float64)
    # The same latents, noiseless: what a perfect channel would have drawn.
    S = np.zeros_like(acc.S)
    W = np.zeros_like(acc.W)
    canon = picture.air_to_canonical(0)[:spec.n_data]
    W[0, canon] = 1e6
    S[0, canon] = 1e6 * sent
    want = np.asarray(render.render(codec, S, W, acc.mode), dtype=np.float64)
    # A TINY pass is 1024 of the group's 52,800 latents, so both pictures
    # are mostly what the decoder draws from nothing: compare what the
    # latents add to that.
    blank = np.asarray(render.render(codec, np.zeros_like(S), np.zeros_like(W), acc.mode), dtype=np.float64)
    rho = np.corrcoef((got - blank).ravel(), (want - blank).ravel())[0, 1]
    psnr = 10 * np.log10(255.0 ** 2 / np.mean((got - want) ** 2))
    print(f"chain {wf}: latent SNR {snr:.1f} dB, picture {psnr:.1f} dB from the noiseless one, "
          f"what the latents add correlates {rho:.3f}")
    assert got.shape == (480, 640, 3) and rho > 0.9


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


@pytest.fixture(scope="module")
def decimal_comma_locale(tmp_path_factory):
    """A de_DE.UTF-8 locale compiled into a temporary LOCPATH (glibc), or a
    skip: its LC_NUMERIC writes 1.5 as "1,5"."""
    import shutil

    if shutil.which("localedef") is None or not Path("/usr/share/i18n/locales/de_DE").is_file():
        pytest.skip("no localedef or no de_DE locale source")
    d = tmp_path_factory.mktemp("locale")
    r = subprocess.run(["localedef", "-i", "de_DE", "-f", "UTF-8", str(d / "de_DE.UTF-8")],
                       capture_output=True, text=True, timeout=120)
    if not (d / "de_DE.UTF-8").exists():
        pytest.skip(f"localedef failed: {r.stderr.strip()}")
    return d


def test_py_float_ignores_a_decimal_comma_locale(native, decimal_comma_locale, monkeypatch):
    """sstvae-qrss runs under QCoreApplication, whose setlocale(LC_ALL, "")
    hands strtod the environment's decimal point; float() always reads '.'.
    Before the fix "1500.3" read as 1500.0 under de_DE."""
    import locale

    n = require(native, "pynum")
    monkeypatch.setenv("LOCPATH", str(decimal_comma_locale))
    old = locale.setlocale(locale.LC_NUMERIC)
    try:
        locale.setlocale(locale.LC_NUMERIC, "de_DE.UTF-8")
        assert locale.localeconv()["decimal_point"] == ","
        for text in ("1500.3", "-0.5e1", ".25", "5.", "1_0.5", "1e3", "inf"):
            assert n.py_float(text) == float(text), text
        assert n.py_float("1,5") is None
    finally:
        locale.setlocale(locale.LC_NUMERIC, old)


def test_the_cli_runs_under_a_decimal_comma_locale(cli, beacon_inputs, decimal_comma_locale, tmp_path):
    """Under a decimal-comma locale the beacon file and its line are what
    the C locale gives. A smoke test only: neither encode nor beacon reads
    or prints a float, so main.cpp's LC_NUMERIC reset (after Qt's
    setlocale) is exercised here but only a float-taking subcommand
    (transmit's --freq) can show it doing anything."""
    outs = []
    for i, env_extra in enumerate(({}, {"LOCPATH": str(decimal_comma_locale), "LC_ALL": "de_DE.UTF-8"})):
        out = tmp_path / f"b{i}.bin"
        r = subprocess.run([str(cli), "beacon", str(beacon_inputs["m1"]), str(out), "--callsign", "K1ABC"],
                           capture_output=True, text=True, timeout=60, env={**os.environ, **env_extra})
        assert r.returncode == 0 and r.stderr == "", r
        outs.append((r.stdout.replace(str(out), "OUT"), out.read_bytes()))
    assert outs[0] == outs[1]

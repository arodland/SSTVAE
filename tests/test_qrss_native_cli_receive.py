"""`sstvae-qrss receive` (native/apps/sstvae_qrss/receive.cpp) against
qrss_receive.py --no-store, run side by side, and Python's transmitter
into the C++ receiver (the reverse of test_qrss_native_cli.py's chain).

Each case runs the script and the subcommand with the same arguments in
two directories holding the same input files, and compares:

- the exit status;
- stdout line by line: every character but the decimal numbers exactly
  (so the header line -- callsign, grid, picture ID, mode, segment,
  codec -- and the callsign windows' text, keying and verdict are exact),
  each decimal number within one unit of its last printed digit, and a
  pass's uid by its slot and frequency parts (the third part hashes z's
  float32 bytes, which agree only to the receiver's tolerance);
- stderr by how the script ended, as test_qrss_native_cli.py does: a
  SystemExit exactly, a usage error by its message, an uncaught exception
  by the traceback's last line (or its class, where only the class can
  agree).

The fixtures are made by the reference's own tools: qrss_simulate.py
(.qrsp -> .npz at -6 dB) for the .npz inputs, and qrss_transmit.py then
qrss_simulate.py (WAV -> WAV at -6 dB) for the WAV inputs. The binary is
$SSTVAE_QRSS_CLI, else `sstvae-qrss` beside the module's build directory
($SSTVAE_NATIVE_MODULE_DIR/..); everything skips without it.

--image: the C++ renders only when the codec half is compiled in
(apps/sstvae_qrss/codec/receive_image.cpp); without it the subcommand
refuses --image, which is checked instead. The picture's decoded pixels
are the reference's for the same latents (test_render_is_the_reference);
the slow FULL test compares the two programs' PNGs.
"""

import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest

from qrss_helpers import Q_TEST, REPO_ROOT, latent_snr_db, synthetic_full_latents
from qrss_native import require
from sstvae.qrss import picture

SLOT = "2025-10-01T00:00Z"
SNR_DB = "-6"
STORE_REFUSAL = ("the multi-pass store is not ported to sstvae-qrss yet (plan stage 3): run with --no-store "
                 "(and without --store, --passband or --no-passband), or use qrss_receive.py\n")
NO_RENDERER = ("--image: this sstvae-qrss renders no pictures (the codec half of receive is not compiled in: it "
               "needs SSTVAE_BUILD_CODEC and apps/sstvae_qrss/codec/receive_image.cpp); use qrss_receive.py "
               "--image\n")


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
    r = subprocess.run([p, "--help"], capture_output=True, text=True, timeout=60)
    if not any(line.split()[:1] == ["receive"] for line in r.stdout.splitlines()):
        pytest.skip("this sstvae-qrss has no receive subcommand")
    return p


def _renders(cli: Path) -> bool:
    """Whether the subcommand has the codec half (--image), asked without
    reading anything: a renderer refuses the missing input, none refuses
    --image first."""
    r = subprocess.run([cli, "receive", "nope.npz", "--no-store", "--image", "a.png"], capture_output=True,
                       text=True, timeout=60)
    assert r.returncode == 1, r
    assert r.stderr in (NO_RENDERER, "nope.npz: no such file\n"), r.stderr
    return r.stderr != NO_RENDERER


# --- running a case -------------------------------------------------------------------------

@dataclass
class Case:
    name: str
    args: list[str]
    files: dict[str, Path] = field(default_factory=dict)   # name in the run dir -> source
    dirs: list[str] = field(default_factory=list)          # empty directories to make
    only_class: bool = False         # an uncaught exception's class, not its text
    store: bool = False              # do not add --no-store


@dataclass
class Run:
    rc: int
    out: str
    err: str


def _run(cmd: list, cwd: Path, env_extra: dict | None = None) -> Run:
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env["COLUMNS"] = "200"
    env.update(env_extra or {})
    r = subprocess.run([str(c) for c in cmd], cwd=cwd, capture_output=True, text=True, timeout=180, env=env)
    return Run(r.returncode, r.stdout, r.stderr)


def _stage(case: Case, root: Path) -> tuple[Path, Path]:
    dirs = []
    for side in ("py", "nat"):
        d = root / case.name / side
        d.mkdir(parents=True)
        for name, src in case.files.items():
            os.symlink(Path(src).resolve(), d / name)
        for sub in case.dirs:
            (d / sub).mkdir(parents=True, exist_ok=True)
        dirs.append(d)
    return dirs[0], dirs[1]


def _argv(case: Case) -> list[str]:
    return case.args if case.store else [*case.args, "--no-store"]


def run_cases(cases: list[Case], root: Path, cli: Path, workers: int = 4) -> dict[str, tuple[Run, Run]]:
    jobs = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for c in cases:
            py_dir, nat_dir = _stage(c, root)
            jobs[c.name] = (
                pool.submit(_run, [sys.executable, REPO_ROOT / "qrss_receive.py", *_argv(c)], py_dir),
                pool.submit(_run, [cli, "receive", *_argv(c)], nat_dir),
            )
        return {name: (a.result(), b.result()) for name, (a, b) in jobs.items()}


def _last_line(text: str) -> str:
    lines = [ln for ln in text.splitlines() if ln.strip()]
    return lines[-1] if lines else ""


_DECIMAL = re.compile(r"[+-]?\d+\.\d+")
_UID = re.compile(r"^pass (\d+)_(-?\d+)_([0-9a-f]{8})(   .*)$")


def _same_line(want: str, got: str, where: str):
    """One report line: text exact, decimals within one unit of the last
    digit, a pass's uid by its slot and frequency."""
    mw, mg = _UID.match(want), _UID.match(got)
    if mw or mg:
        assert mw and mg, (where, want, got)
        assert mg.group(1) == mw.group(1) and mg.group(4) == mw.group(4), (where, want, got)
        assert abs(int(mg.group(2)) - int(mw.group(2))) <= 1, (where, want, got)
        return
    assert _DECIMAL.split(got) == _DECIMAL.split(want), (where, want, got)
    nw, ng = _DECIMAL.findall(want), _DECIMAL.findall(got)
    assert len(nw) == len(ng), (where, want, got)
    for a, b in zip(nw, ng):
        unit = 10.0 ** -len(a.split(".")[1])
        assert abs(float(a) - float(b)) <= unit * 1.0001, (where, a, b, want, got)


def same_report(want: str, got: str, where: str = ""):
    w, g = want.splitlines(), got.splitlines()
    assert len(w) == len(g), (where, want, got)
    for a, b in zip(w, g):
        _same_line(a, b, where)


def check(case: Case, py: Run, nat: Run):
    assert nat.rc == py.rc, (case.name, py, nat)
    same_report(py.out, nat.out, case.name)
    assert "Traceback" not in nat.err
    if py.rc in (0, 1) and "Traceback" not in py.err and py.err == "":
        assert nat.err == "", (case.name, nat.err)
    elif py.rc == 2:
        prog = "qrss_receive.py: error: "
        assert py.err.startswith("usage: ") and _last_line(py.err).startswith(prog), py.err
        assert nat.err.startswith("usage: sstvae-qrss receive"), nat.err
        assert _last_line(nat.err) == "sstvae-qrss receive: error: " + _last_line(py.err)[len(prog):]
    elif "Traceback (most recent call last)" in py.err:
        want, got = _last_line(py.err), _last_line(nat.err)
        if case.only_class:
            assert got.split(":")[0] == want.split(":")[0], (want, got)
        else:
            assert got == want
        assert nat.err == got + "\n"
    else:
        assert nat.err == py.err, (case.name, py.err, nat.err)


# --- inputs ---------------------------------------------------------------------------------

def _tool(script: str, *args, cwd: Path):
    r = subprocess.run([sys.executable, str(REPO_ROOT / script), *map(str, args)], cwd=cwd, capture_output=True,
                       text=True, timeout=180)
    assert r.returncode == 0, (script, args, r.stdout, r.stderr)


def _resave(src: Path, dst: Path, **changes):
    """An .npz with some members replaced, written by np.savez (as save_npz does)."""
    with np.load(src, allow_pickle=False) as d:
        members = {k: d[k] for k in d.files}
    members.update(changes)
    with open(dst, "wb") as fh:
        np.savez(fh, **members)


@pytest.fixture(scope="module")
def inputs(tmp_path_factory):
    """The stored picture, CE and L SHORT .npz files (qrss_simulate.py at
    -6 dB) and CE and L SHORT WAVs (qrss_transmit.py, then qrss_simulate.py
    WAV -> WAV at -6 dB: int16 for CE, float32 for L), plus the broken and
    noise inputs the refusal cases read."""
    d = tmp_path_factory.mktemp("receive-in")
    sp = picture.StoredPicture.from_latents(synthetic_full_latents(3), 0xD1D8, 1)
    picture.save_qrsp(d / "pic.qrsp", sp)
    common = ("--slot", SLOT, "--callsign", "K1ABC", "--grid", "FN42", "--frame", "short")

    def npz(name, *extra):
        _tool("qrss_simulate.py", "pic.qrsp", name, *common, "--snr", SNR_DB, *extra, cwd=d)

    def wav(wf, seg, seed, *sim_extra):
        tx = f"{wf.lower()}-tx.wav"
        _tool("qrss_transmit.py", "pic.qrsp", tx, *common, "--waveform", wf, "--segment", seg, "--float", cwd=d)
        _tool("qrss_simulate.py", tx, f"{wf.lower()}-rx.wav", "--slot", SLOT, "--frame", "short",
              "--waveform", wf, "--snr", SNR_DB, "--seed", seed, *sim_extra, cwd=d)

    with ThreadPoolExecutor(max_workers=4) as pool:
        jobs = [pool.submit(npz, "ce.npz", "--seed", "1"),
                pool.submit(npz, "l.npz", "--waveform", "L", "--segment", "1", "--seed", "2"),
                pool.submit(wav, "CE", "0", "5"),
                pool.submit(wav, "L", "1", "6", "--float")]
        for j in jobs:
            j.result()

    f = {k: d / k for k in ("pic.qrsp", "ce.npz", "l.npz", "ce-rx.wav", "l-rx.wav")}
    (d / "empty.npz").write_bytes(b"")
    (d / "garbage.npz").write_bytes(b"hello, this is not an archive\n")
    (d / "text.wav").write_text("hello, this is not a WAV file at all")
    np.save(d / "array.npy", np.arange(4))
    os.replace(d / "array.npy", d / "array.npz")
    _resave(f["ce.npz"], d / "bogus-frame.npz", frame=np.str_("bogus"))
    _resave(f["ce.npz"], d / "v1.npz", frame=np.str_("short-v1"))
    with np.load(f["ce.npz"]) as z:
        fe = z["fe"]
    rng = np.random.default_rng(7)
    noise = ((rng.standard_normal(fe.size) + 1j * rng.standard_normal(fe.size)) * np.std(fe) / np.sqrt(2))
    _resave(f["ce.npz"], d / "noise.npz", fe=noise.astype(np.complex64))
    (d / "not-zip.npz").write_bytes(b"PK\x03\x04" + bytes(range(64)))
    data = bytearray(f["ce.npz"].read_bytes())
    data[data.index(fe[:4].tobytes())] ^= 0x01        # a bit of fe's data: zipfile's CRC check
    (d / "bad-crc.npz").write_bytes(bytes(data))
    for k in ("empty.npz", "garbage.npz", "text.wav", "array.npz", "bogus-frame.npz", "v1.npz", "noise.npz",
              "not-zip.npz", "bad-crc.npz"):
        f[k] = d / k
    return sp, f


# --- side by side ---------------------------------------------------------------------------

def _receive_cases(f, image: bool) -> list[Case]:
    img = ["--image", "out/pic.png"] if image else []
    return [
        Case("ce-npz", ["ce.npz", "--slot", SLOT], {"ce.npz": f["ce.npz"]}),
        Case("ce-wav", ["ce-rx.wav", "--slot", SLOT, "--frame", "short", *img], {"ce-rx.wav": f["ce-rx.wav"]}),
    ]


def _slow_receive_cases(f) -> list[Case]:
    return [
        Case("l-npz", ["l.npz"], {"l.npz": f["l.npz"]}),
        Case("l-wav", ["l-rx.wav", "--slot", SLOT, "--frame", "l-short", "--just-l"], {"l-rx.wav": f["l-rx.wav"]}),
        # The receiver of 2026-10-11 (both rules set back, CE alone), the
        # plain estimator, and a waveform named by the other's frame.
        Case("ce-old", ["ce.npz", "--just-ce", "--old-edge-noise", "--old-gating", "--estimator", "plain"],
             {"ce.npz": f["ce.npz"]}),
        Case("ce-lframe", ["ce.npz", "--frame", "l-short", "--start", "2025-10-01T00:00:00Z"],
             {"ce.npz": f["ce.npz"]}),
        Case("noise", ["noise.npz"], {"noise.npz": f["noise.npz"]}),
        Case("noise-just-ce", ["noise.npz", "--frame", "l-short", "--just-ce"], {"noise.npz": f["noise.npz"]}),
    ]


def _expect_pass(out: str, wf: str, frame_name: str, segment: int):
    assert f"   {wf} {frame_name}\n" in out, out
    assert f"  header     K1ABC FN42 picture ab2dea7f mode B segment {segment} codec d1d8\n" in out, out
    assert "  callsign   K1ABC   match Z " in out and "keying fsk   agrees\n" in out, out
    assert "Hz at t0" in out and "Z_ref" in out and "mean W" in out


def test_receive_side_by_side(cli, inputs, tmp_path):
    """A CE pass from an .npz and from a WAV: the same report, to the
    receiver's tolerance, header and callsign windows exact. With the
    codec half the WAV case asks for --image too: a SHORT pass is not a
    picture segment, and the reference's line says so after the report
    (no model is loaded for it)."""
    sp, f = inputs
    assert sp.id_hex == "ab2dea7f"
    image = _renders(cli)
    cases = _receive_cases(f, image)
    results = run_cases(cases, tmp_path, cli)
    for c in cases:
        py, nat = results[c.name]
        check(c, py, nat)
        assert py.rc == 0, py
        _expect_pass(nat.out, "CE", "short", 0)
        assert nat.out.count("\npass ") + nat.out.startswith("pass ") == 1, nat.out
    if image:
        last = _last_line(results["ce-wav"][1].out)
        assert last.startswith("image: a short frame holds ") and last.endswith(" latents, not a picture segment")


@pytest.mark.slow
def test_receive_side_by_side_slow(cli, inputs, tmp_path):
    """L passes (.npz and WAV), the old rules with the plain estimator, a
    CE .npz under an L frame name, and noise: "no CE or L signal found"
    and exit 1 from both."""
    _, f = inputs
    cases = _slow_receive_cases(f)
    results = run_cases(cases, tmp_path, cli)
    for c in cases:
        check(c, *results[c.name])
    _expect_pass(results["l-npz"][1].out, "L", "l-short", 1)
    _expect_pass(results["l-wav"][1].out, "L", "l-short", 1)
    _expect_pass(results["ce-old"][1].out, "CE", "short", 0)
    _expect_pass(results["ce-lframe"][1].out, "CE", "short", 0)
    assert results["noise"][1] == Run(1, "no CE or L signal found\n", "")
    assert results["noise-just-ce"][1] == Run(1, "no CE signal found\n", "")


# --- refusals -------------------------------------------------------------------------------

def _refusal_cases(f) -> list[Case]:
    ce = {"ce.npz": f["ce.npz"]}
    wav = {"rx.wav": f["ce-rx.wav"]}

    def c(name, *args, files=None, **kw):
        return Case(name, list(args), ce if files is None else files, **kw)

    return [
        c("slot-off-quarter", "ce.npz", "--slot", "2025-10-01T00:07Z"),
        c("slot-garbage", "ce.npz", "--slot", "yesterday"),
        c("start-garbage", "rx.wav", "--slot", SLOT, "--start", "noon", files=wav),
        c("missing", "nope.wav", "--slot", SLOT, files={}),
        c("directory", "d.npz", files={}, dirs=["d.npz"]),
        c("wav-no-slot", "rx.wav", files=wav),
        c("wav-empty-slot", "rx.wav", "--slot", "", files=wav),
        c("npz-excluded", "ce.npz", "--just-l"),
        c("both-just", "ce.npz", "--just-ce", "--just-l"),
        c("frame-invalid", "ce.npz", "--frame", "bogus"),
        c("estimator-invalid", "ce.npz", "--estimator", "fast"),
        c("precision-invalid", "ce.npz", "--precision", "fp8"),
        c("no-args", files={}),
        c("unknown-flag", "ce.npz", "--bogus"),
        c("npz-empty", "x.npz", files={"x.npz": f["empty.npz"]}),
        c("npz-garbage", "x.npz", files={"x.npz": f["garbage.npz"]}),
        c("npz-npy", "x.npz", files={"x.npz": f["array.npz"]}),
        c("npz-qrsp", "x.npz", files={"x.npz": f["pic.qrsp"]}),
        c("npz-bogus-frame", "x.npz", files={"x.npz": f["bogus-frame.npz"]}),
        c("npz-no-twin", "x.npz", files={"x.npz": f["v1.npz"]}),
        c("npz-no-twin-just-l", "x.npz", "--just-l", files={"x.npz": f["v1.npz"]}),
        c("npz-not-zip", "x.npz", files={"x.npz": f["not-zip.npz"]}),
        c("npz-bad-crc", "x.npz", files={"x.npz": f["bad-crc.npz"]}),
        c("wav-text", "x.wav", "--slot", SLOT, files={"x.wav": f["text.wav"]}, only_class=True),
    ]


def test_refusals(cli, inputs, tmp_path):
    """Every refusal a user can hit before the receive: the same exit
    status and message (a usage error's by its last line)."""
    _, f = inputs
    cases = _refusal_cases(f)
    results = run_cases(cases, tmp_path, cli)
    for c in cases:
        py, nat = results[c.name]
        check(c, py, nat)
        assert py.rc != 0, (c.name, py)
    assert results["missing"][1].err == "nope.wav: no such file\n"
    assert results["wav-no-slot"][1].err == "--slot is required for a WAV input\n"
    assert results["npz-excluded"][1].err == ("the .npz holds frame short, waveform CE, which --just-l "
                                              "does not listen for\n")
    assert results["npz-no-twin"][1].err == "frame short-v1 has no L twin; add --just-ce\n"
    assert _last_line(results["both-just"][1].err) == ("sstvae-qrss receive: error: argument --just-l: "
                                                       "not allowed with argument --just-ce")


def test_store_is_refused(cli, inputs, tmp_path):
    """Stage 2 receives with --no-store only: the store's flags, and their
    absence (the reference's default stores), are refused before anything
    is read."""
    _, f = inputs
    for args in (["ce.npz"], ["ce.npz", "--store", "st"], ["ce.npz", "--no-store", "--store", "st"],
                 ["ce.npz", "--no-store", "--passband", "pb"], ["ce.npz", "--no-store", "--no-passband"],
                 ["nope.npz"], ["ce.npz", "--no-store", "--no-passband", "--image", "a.png"]):
        d = tmp_path / str(len(list(tmp_path.iterdir())))
        d.mkdir()
        os.symlink(f["ce.npz"], d / "ce.npz")
        r = _run([cli, "receive", *args], d)
        assert r == Run(1, "", STORE_REFUSAL), (args, r)
        assert sorted(p.name for p in d.iterdir()) == ["ce.npz"], args


def test_help(cli):
    r = _run([cli, "receive", "--help"], REPO_ROOT)
    assert r.rc == 0 and r.err == ""
    assert r.out.startswith(
        "usage: sstvae-qrss receive [-h] [--slot SLOT] [--start START] "
        "[--frame {full,medium,short,tiny,l-full,l-medium,l-short,l-tiny}] [--just-ce | --just-l] "
        "[--store STORE] [--no-store] [--passband PASSBAND] [--no-passband] [--estimator {joint,plain}] "
        "[--image IMAGE] [--model MODEL] [--precision {fp32,fp16,int8}] [--old-edge-noise] [--old-gating] "
        "input\n")
    for flag in ("--slot", "--start", "--frame", "--just-ce", "--just-l", "--estimator", "--image", "--model",
                 "--precision", "--old-edge-noise", "--old-gating", "--no-store", "not ported to sstvae-qrss yet"):
        assert flag in r.out, flag
    r = _run([cli, "--help"], REPO_ROOT)
    assert any(line.split()[:1] == ["receive"] and "qrss_receive.py" in line for line in r.out.splitlines())


# --- --image --------------------------------------------------------------------------------

def test_image_without_the_codec_half(cli, tmp_path):
    """Without the codec half --image is refused, before anything is read
    (with it, test_receive_side_by_side runs --image)."""
    if _renders(cli):
        pytest.skip("this sstvae-qrss renders --image")
    r = _run([cli, "receive", "nope.npz", "--no-store", "--image", "out/pic.png"], tmp_path)
    assert r == Run(1, "", NO_RENDERER)


def test_image_names_png_only(cli, tmp_path):
    """The native image layer writes PNG: another extension is refused up
    front rather than written as PNG under a name that says otherwise."""
    if not _renders(cli):
        pytest.skip("this sstvae-qrss has no --image")
    for name, ok in (("a.png", True), ("A.PNG", True), ("x/b.apng", True), ("a.jpg", False), ("png", False),
                     (".png", False), ("a.png.bmp", False)):
        r = _run([cli, "receive", "nope.npz", "--no-store", "--image", name], tmp_path)
        want = "nope.npz: no such file\n" if ok else (
            f"--image {name}: sstvae-qrss writes PNG only (a .png name); qrss_receive.py --image writes "
            "whatever format Pillow names by the extension\n")
        assert r == Run(1, "", want), (name, r)


@pytest.mark.codec
def test_render_is_the_reference(native):
    """render.render's pixels, byte for byte, for the same planes (fp16, a
    mode B and a mode C segment), and receive's placement of a pass."""
    import qrss_codec_fixtures as fixtures
    from sstvae.qrss import render

    n = require(native, "codec_render")
    models = fixtures.v5_dir()
    lat = synthetic_full_latents(5)
    rng = np.random.default_rng(2)
    codec = render.load("fp16", str(models), codec_id=0xD1D8)
    assert n.decoder_codec_id(str(models), "fp16") == render.decoder_codec_id(codec) == 0xD1D8
    for seg, mode in ((0, 1), (2, 2)):
        idx = picture.air_to_canonical(seg)
        z = lat.reshape(3, -1)[seg][idx].astype(np.float32)
        w = rng.uniform(0.1, 3.0, z.size).astype(np.float32)
        S = np.zeros((picture.N_GROUPS, picture.GROUP_LATENTS))
        W = np.zeros_like(S)
        S[seg, idx] = w.astype(np.float64) * z.astype(np.float64)
        W[seg, idx] = w
        Sn, Wn = n.pass_planes(z, w, seg)
        assert np.array_equal(Sn, S) and np.array_equal(Wn, W)
        want = np.asarray(render.render(codec, S, W, mode))
        got = n.render(S, W, mode, str(models), "fp16")
        assert got.dtype == np.uint8 and got.shape == want.shape == (480, 640, 3)
        assert np.array_equal(got, want), (seg, mode, np.count_nonzero(got != want))


def test_check_codec_id_is_the_reference(native):
    """check_codec_id: the same warning, None, or CodecMismatch with the
    same text, for every (decoder's ID, picture's ID, force)."""
    from sstvae.qrss import render

    n = require(native, "codec_render")

    class Fake:
        backend = "onnx"

        def __init__(self, have):
            self._sources = {"decoder": None if have is None else f"{have:04x}" + "0" * 60}

        def _session(self, part):
            pass

    for have in (None, 0xD1D8, 0x1234):
        for cid in (None, 0xD1D8, 0x0001):
            for force in (False, True):
                want = got = None
                try:
                    want = ("ok", render.check_codec_id(Fake(have), cid, force))
                except render.CodecMismatch as e:
                    want = ("mismatch", str(e))
                try:
                    got = ("ok", n.check_codec_id(have, cid, force))
                except render.CodecMismatch as e:
                    got = ("mismatch", str(e))
                assert got == want, (have, cid, force)


# --- the reverse chain: Python transmits, the C++ receives ---------------------------------

_HEADER = re.compile(r"^  header     (\S+) (\S+) picture ([0-9a-f]{8}) mode ([ABC]) segment (\d) codec ([0-9a-f]{4})$",
                     re.M)


@pytest.mark.parametrize("wf", ["CE", "L"])
def test_reverse_chain(cli, native, inputs, wf):
    """qrss_transmit.py's SHORT pass through the channel at -6 dB into
    sstvae-qrss receive: the header (callsign, grid, the stored picture's
    ID, mode, segment, codec) and the callsign windows; and the picture's
    latents, from the same C++ receive in process (the CLI prints no
    latents: receiver_slot.receive_wav, whose pass the CLI's uid names,
    hash included)."""
    from qrss_native._rx import to_ref
    from sstvae.qrss import frame, lframe, tx
    from sstvae.qrss.types import pass_uid

    sp, f = inputs
    segment = 1 if wf == "L" else 0
    spec = lframe.get("l-short") if wf == "L" else frame.SHORT
    wav = f[f"{wf.lower()}-rx.wav"]
    flag = "--just-l" if wf == "L" else "--just-ce"
    n = require(native, "receiver_slot")
    with ThreadPoolExecutor(max_workers=1) as pool:
        job = pool.submit(_run, [cli, "receive", wav, "--slot", SLOT, "--frame", spec.name, flag, "--no-store"],
                          REPO_ROOT)
        passes = to_ref(n.receive_wav(str(wav), Q_TEST, None, spec, "joint"))
        r = job.result()
    assert r.rc == 0 and r.err == "", r
    heads = _HEADER.findall(r.out)
    assert heads == [("K1ABC", "FN42", sp.id_hex, "ABC"[sp.mode], str(segment), f"{sp.codec_id:04x}")], r.out
    assert "  callsign   K1ABC   match Z " in r.out and "keying fsk   agrees\n" in r.out, r.out

    assert len(passes) == 1
    p = passes[0]
    assert r.out.startswith(f"pass {pass_uid(p.q, p.f_hz, p.z)}   {wf} {spec.name}\n"), (r.out, p.uid)
    sent = tx.slot_air_latents(sp, segment)[:spec.n_data]
    snr = latent_snr_db(p.z, sent)
    print(f"reverse chain {wf}: latent SNR {snr:.1f} dB at {SNR_DB} dB")
    assert snr > 6.0, snr


# --- locale ---------------------------------------------------------------------------------

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


def test_report_ignores_a_decimal_comma_locale(cli, inputs, decimal_comma_locale):
    """The report, under de_DE (decimal comma), is the C locale's byte for
    byte: every number is formatted by frame::py_format, and --start's
    seconds are read by the slot parser, not strtod."""
    _, f = inputs
    args = [cli, "receive", f["ce.npz"], "--no-store", "--start", "2025-10-01T00:00:00.5Z"]
    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs = [pool.submit(_run, args, REPO_ROOT),
                pool.submit(_run, args, REPO_ROOT, {"LOCPATH": str(decimal_comma_locale), "LC_ALL": "de_DE.UTF-8"})]
        plain, comma = (j.result() for j in jobs)
    assert plain.rc == 0 and plain.err == "", plain
    assert comma == plain
    assert "," not in plain.out
    _expect_pass(plain.out, "CE", "short", 0)


# --- FULL, with the picture -----------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.codec
@pytest.mark.parametrize("wf", ["CE", "L"])
def test_full_image(cli, native, tmp_path, wf):
    """A FULL pass at -3 dB from both programs, --image (fp16).

    The report is the reference's, and so is the picture, two ways. Exactly:
    the C++ CLI's PNG is, pixel for pixel, what qrss_receive.render_pass
    draws from the C++ receiver's own pass (receiver_slot.receive_slot in
    process, the pass the CLI's uid names, hash included). And to the
    receivers' tolerance against qrss_receive.py's own PNG: the two
    receivers' z agree to rounding, so a few subpixels of 921,600 land one
    level apart (measured 18 for CE and 22 for L), every other one equal."""
    import importlib.util

    import qrss_codec_fixtures as fixtures
    from PIL import Image
    from qrss_native._rx import to_ref
    from sstvae.qrss import channel, frontend, lframe
    from sstvae.qrss.constants import WAVEFORM_CE, WAVEFORM_L

    if not _renders(cli):
        pytest.skip("this sstvae-qrss has no --image")
    n = require(native, "receiver_slot")
    models = fixtures.v5_dir()
    d = tmp_path / "in"
    d.mkdir()
    sp = picture.StoredPicture.from_latents(synthetic_full_latents(3), 0xD1D8, 1)
    picture.save_qrsp(d / "pic.qrsp", sp)
    extra = ("--waveform", "L", "--segment", "1", "--seed", "4") if wf == "L" else ("--seed", "3")
    _tool("qrss_simulate.py", "pic.qrsp", "full.npz", "--slot", SLOT, "--callsign", "K1ABC", "--grid", "FN42",
          "--frame", "full", "--snr", "-3", *extra, cwd=d)
    case = Case(f"full-{wf}", ["full.npz", "--image", "pic.png", "--model", str(models), "--precision", "fp16"],
                {"full.npz": d / "full.npz"})
    sim = channel.load_npz(d / "full.npz")
    cap = frontend.Capture(fe=sim.fe, q=sim.q, t0_index=sim.t0_index, fs=sim.fs)
    spec = tuple(lframe.twin(sim.spec, w) for w in (WAVEFORM_CE, WAVEFORM_L))
    with ThreadPoolExecutor(max_workers=1) as pool:
        job = pool.submit(lambda: to_ref(n.receive_slot(cap, spec, False, "joint", (), True, None, None, None, (),
                                                        True)))
        py, nat = run_cases([case], tmp_path, cli)[case.name]
        mine = job.result()
    check(case, py, nat)
    assert nat.rc == 0 and _last_line(nat.out) == "image: pic.png", nat.out
    _expect_pass(nat.out, wf, "l-full" if wf == "L" else "full", 1 if wf == "L" else 0)
    assert len(mine) == 1 and nat.out.startswith(f"pass {mine[0].uid}   "), (nat.out, [p.uid for p in mine])

    mod_spec = importlib.util.spec_from_file_location("qrss_receive_ref", REPO_ROOT / "qrss_receive.py")
    ref = importlib.util.module_from_spec(mod_spec)
    mod_spec.loader.exec_module(ref)
    swap = tmp_path / "swap.png"
    assert ref.render_pass(mine[0], swap, str(models), "fp16") == f"image: {swap}"

    def rgb(path):
        return np.asarray(Image.open(path).convert("RGB"), dtype=np.int16)

    b = rgb(tmp_path / case.name / "nat" / "pic.png")
    assert np.array_equal(b, rgb(swap))
    a = rgb(tmp_path / case.name / "py" / "pic.png")
    assert a.shape == b.shape == (480, 640, 3)
    diff = np.abs(a - b)
    print(f"full {wf}: {np.count_nonzero(diff)} subpixels differ from qrss_receive.py's, at most {diff.max()}")
    assert diff.max() <= 1 and np.count_nonzero(diff) <= 200

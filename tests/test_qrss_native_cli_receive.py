"""`sstvae-qrss receive` (native/apps/sstvae_qrss/receive.cpp) against
qrss_receive.py, run side by side, and Python's transmitter into the C++
receiver (the reverse of test_qrss_native_cli.py's chain).

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
  agree);
- with the multi-pass store (the cases without --no-store), everything
  the two leave in their directories (same_tree): the same files, with
  the pass uids' hash part taken out of their names and texts (it hashes
  z's float32 bytes); the JSON (the accumulators', the clusters', the
  passband store's index and meta.json, log.jsonl) as parsed, floats
  within 1e-6 relative, wall-clock stamps within the test's run; every
  .npz member by member (dtype, shape; floats within 1e-6 of the array's
  scale, the rest exact; a pass's `meta` as JSON); and every other file
  (the passband store's raw hours, a lock file) byte for byte.

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

import json
import os
import re
import subprocess
import sys
import time
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
    texts: dict[str, str] = field(default_factory=dict)    # files to write in the run dir
    locks: list[str] = field(default_factory=list)         # directories this process locks (live.dir_lock)
    json_error: bool = False         # the script's JSONDecodeError is the C++'s ValueError


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
        for name, text in case.texts.items():
            (d / name).parent.mkdir(parents=True, exist_ok=True)
            (d / name).write_text(text)
        dirs.append(d)
    return dirs[0], dirs[1]


def _argv(case: Case) -> list[str]:
    return case.args if case.store else [*case.args, "--no-store"]


def _home(d: Path) -> dict:
    """The default store's root ($QRSSTVAE_HOME) inside the run directory,
    for every case: a store without --store never reaches the real one."""
    return {"QRSSTVAE_HOME": str(d / "home")}


def run_cases(cases: list[Case], root: Path, cli: Path, workers: int = 4) -> dict[str, tuple[Run, Run]]:
    from sstvae.qrss import live

    jobs, held = {}, []
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for c in cases:
                py_dir, nat_dir = _stage(c, root)
                for d in (py_dir, nat_dir):
                    for sub in c.locks:
                        held.append(live.dir_lock(d / sub))
                        assert held[-1] is not None, d / sub
                jobs[c.name] = (
                    pool.submit(_run, [sys.executable, REPO_ROOT / "qrss_receive.py", *_argv(c)], py_dir,
                                _home(py_dir)),
                    pool.submit(_run, [cli, "receive", *_argv(c)], nat_dir, _home(nat_dir)),
                )
            return {name: (a.result(), b.result()) for name, (a, b) in jobs.items()}
    finally:
        for f in held:
            f.close()


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
        elif case.json_error:
            # json.decoder.JSONDecodeError is a ValueError, which the C++
            # store raises with json's message (receive.cpp's header)
            prefix = "json.decoder.JSONDecodeError: "
            assert want.startswith(prefix), want
            assert got == "ValueError: " + want[len(prefix):], (want, got)
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
    WAV -> WAV at -6 dB: int16 for CE, float32 for L), two headerless CE
    TINY .npz files of the same picture at 0 dB (1800 and 1000 Hz, so
    neither is the SHORT pass's transmission again), plus the broken and
    noise inputs the refusal and store cases read."""
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

    def tiny(name, freq, seed):
        _tool("qrss_simulate.py", "pic.qrsp", name, "--slot", SLOT, "--callsign", "K1ABC", "--frame", "tiny",
              "--snr", "0", "--freq", freq, "--seed", seed, cwd=d)

    with ThreadPoolExecutor(max_workers=4) as pool:
        jobs = [pool.submit(npz, "ce.npz", "--seed", "1"),
                pool.submit(npz, "l.npz", "--waveform", "L", "--segment", "1", "--seed", "2"),
                pool.submit(wav, "CE", "0", "5"),
                pool.submit(wav, "L", "1", "6", "--float"),
                pool.submit(tiny, "tiny-1800.npz", "1800", "3"),
                pool.submit(tiny, "tiny-1000.npz", "1000", "4")]
        for j in jobs:
            j.result()

    f = {k: d / k for k in ("pic.qrsp", "ce.npz", "l.npz", "ce-rx.wav", "l-rx.wav", "tiny-1800.npz",
                            "tiny-1000.npz")}
    (d / "empty.npz").write_bytes(b"")
    (d / "garbage.npz").write_bytes(b"hello, this is not an archive\n")
    (d / "text.wav").write_text("hello, this is not a WAV file at all")
    np.save(d / "array.npy", np.arange(4))
    os.replace(d / "array.npy", d / "array.npz")
    _resave(f["ce.npz"], d / "bogus-frame.npz", frame=np.str_("bogus"))
    _resave(f["ce.npz"], d / "v1.npz", frame=np.str_("short-v1"))
    _resave(f["ce.npz"], d / "q-negative.npz", q=np.int64(-5))
    _resave(f["ce.npz"], d / "fs-zero.npz", fs=np.int64(0))
    with np.load(f["ce.npz"]) as z:
        fe = z["fe"]
    rng = np.random.default_rng(7)
    noise = ((rng.standard_normal(fe.size) + 1j * rng.standard_normal(fe.size)) * np.std(fe) / np.sqrt(2))
    _resave(f["ce.npz"], d / "noise.npz", fe=noise.astype(np.complex64))
    # noise at twice the front end's rate: the passband store is not written
    _resave(f["ce.npz"], d / "noise-8k.npz", fe=noise.astype(np.complex64), fs=np.int64(8000))
    _resave(f["ce.npz"], d / "fe-c16.npz", fe=fe.astype(np.complex128))
    (d / "not-zip.npz").write_bytes(b"PK\x03\x04" + bytes(range(64)))
    data = bytearray(f["ce.npz"].read_bytes())
    data[data.index(fe[:4].tobytes())] ^= 0x01        # a bit of fe's data: zipfile's CRC check
    (d / "bad-crc.npz").write_bytes(bytes(data))
    for k in ("empty.npz", "garbage.npz", "text.wav", "array.npz", "bogus-frame.npz", "v1.npz", "noise.npz",
              "not-zip.npz", "bad-crc.npz", "q-negative.npz", "fs-zero.npz", "fe-c16.npz", "noise-8k.npz"):
        f[k] = d / k
    return sp, f


# --- side by side ---------------------------------------------------------------------------

def _receive_cases(f, image: bool) -> list[Case]:
    img = ["--image", "out/pic.png"] if image else []
    return [
        Case("ce-npz", ["ce.npz", "--slot", SLOT], {"ce.npz": f["ce.npz"]}),
        Case("ce-wav", ["ce-rx.wav", "--slot", SLOT, "--frame", "short", *img], {"ce-rx.wav": f["ce-rx.wav"]},
             dirs=["out"]),
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
        # A slot before 1970 (q < 0), from --slot and from the .npz, refused
        # where it is read (parse_slot and load_npz), as are an .npz's other
        # members save_npz never writes.
        c("slot-before-1970", "ce.npz", "--slot", "1969-12-31T23:45Z"),
        c("npz-q-negative", "x.npz", files={"x.npz": f["q-negative.npz"]}),
        c("npz-fs-zero", "x.npz", files={"x.npz": f["fs-zero.npz"]}),
        c("npz-fe-c16", "x.npz", files={"x.npz": f["fe-c16.npz"]}),
        # --image's directory, checked before anything is read (with or
        # without the codec half: the directory comes first)
        c("image-no-dir", "ce.npz", "--image", "nodir/pic.png"),
        c("image-no-dir-missing-input", "nope.npz", "--image", "a/b/pic.png", files={}, dirs=["a"]),
        c("image-no-dir-bad-slot", "ce.npz", "--slot", "yesterday", "--image", "nodir/pic.png"),
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
    assert results["slot-before-1970"][1] == Run(
        1, "", "--slot '1969-12-31T23:45Z': slot 1969-12-31T23:45Z is before 1970-01-01T00:00Z\n")
    assert results["npz-q-negative"][1] == Run(
        1, "", "ValueError: simfile: q -5 is a quarter hour before 1970-01-01T00:00Z\n")
    assert results["npz-fs-zero"][1] == Run(1, "", "ValueError: simfile: fs 0 is not a sample rate\n")
    assert results["npz-fe-c16"][1] == Run(
        1, "", "ValueError: simfile: fe is <c16 of 1 dimensions, not complex64 (n,) as qrss_simulate.py "
        "writes it\n")
    for name, image in (("image-no-dir", "nodir/pic.png"), ("image-no-dir-missing-input", "a/b/pic.png"),
                        ("image-no-dir-bad-slot", "nodir/pic.png")):
        parent = image.rsplit("/", 1)[0]
        assert results[name][1] == Run(1, "", f"--image {image}: {parent}: no such directory\n"), name


# --- the multi-pass store ------------------------------------------------------------------

_UID_LINE = re.compile(r"^pass (\S+)   ", re.M)
_STAMPS = {"created", "updated", "time", "merged"}     # wall-clock stamps (time.time())
_REL = 1e-6                                            # floats, relative (measured worst: 3e-8)


def _uid_map(*outs: str) -> dict:
    """{uid: "UID<k>"} by the order the passes were printed (one side's
    stdout, run after run): the uid's hash part is z's float32 bytes."""
    m = {}
    for out in outs:
        for uid in _UID_LINE.findall(out):
            m.setdefault(uid, f"UID{len(m)}")
    return m


def _norm(x, uids: dict):
    if isinstance(x, str):
        for u, v in uids.items():
            x = x.replace(u, v)
        return x
    if isinstance(x, dict):
        return {_norm(k, uids): _norm(v, uids) for k, v in x.items()}
    if isinstance(x, list):
        return [_norm(v, uids) for v in x]
    return x


class _Worst:
    rel = 0.0


def _close(a, b, where: str, window: tuple[float, float]):
    """Parsed JSON: the same structure and values, floats within _REL
    relative (and 1e-9 absolute, for values near zero), stamps equal or
    both within the run's window."""
    if isinstance(a, dict):
        assert isinstance(b, dict) and sorted(a) == sorted(b), (where, a, b)
        for k in a:
            if k in _STAMPS and isinstance(a[k], float) and a[k] != b[k]:
                for v in (a[k], b[k]):
                    assert isinstance(v, float) and window[0] <= v <= window[1], (where, k, v, window)
            else:
                _close(a[k], b[k], f"{where}.{k}", window)
    elif isinstance(a, list):
        assert isinstance(b, list) and len(a) == len(b), (where, a, b)
        for k, (u, v) in enumerate(zip(a, b)):
            _close(u, v, f"{where}[{k}]", window)
    elif isinstance(a, float):
        assert isinstance(b, float), (where, a, b)
        if a != b and not (np.isnan(a) and np.isnan(b)):
            d = abs(a - b)
            assert d <= _REL * max(abs(a), abs(b)) + 1e-9, (where, a, b)
            _Worst.rel = max(_Worst.rel, d / max(abs(a), abs(b)))
    else:
        assert type(a) is type(b) and a == b, (where, a, b)


def _same_bytes(a: Path, b: Path) -> bool:
    if a.stat().st_size != b.stat().st_size:
        return False
    with open(a, "rb") as fa, open(b, "rb") as fb:
        while True:
            x, y = fa.read(1 << 20), fb.read(1 << 20)
            if x != y:
                return False
            if not x:
                return True


def _same_npz(a: Path, b: Path, uids: tuple[dict, dict], where: str, window):
    with np.load(a, allow_pickle=False) as da, np.load(b, allow_pickle=False) as db:
        assert da.files == db.files, (where, da.files, db.files)
        for k in da.files:
            x, y = da[k], db[k]
            if k == "meta" and x.dtype == y.dtype == np.uint8:     # PassResult.save's JSON (any length)
                _close(_norm(json.loads(x.tobytes()), uids[0]), _norm(json.loads(y.tobytes()), uids[1]),
                       f"{where}:meta", window)
                continue
            assert x.dtype == y.dtype and x.shape == y.shape, (where, k, x.dtype, y.dtype, x.shape, y.shape)
            if np.issubdtype(x.dtype, np.inexact) and x.size:
                scale = float(np.max(np.abs(x)))
                assert np.allclose(x, y, rtol=_REL, atol=_REL * scale, equal_nan=True), (where, k)
                d = float(np.max(np.abs(x.astype(np.complex128) - y.astype(np.complex128))))
                _Worst.rel = max(_Worst.rel, d / scale if scale else 0.0)
            else:
                assert np.array_equal(x, y), (where, k)


def _parses(path: Path) -> bool:
    try:
        json.loads(path.read_text())
    except ValueError:
        return False                  # (a broken file a case planted: compared byte for byte)
    return True


def same_tree(py_dir: Path, nat_dir: Path, uids: tuple[dict, dict], window: tuple[float, float]) -> list[str]:
    """What the two runs left in their directories (the staged inputs, symlinks, aside): the same
    names, uids' hash parts taken out; JSON as parsed, .npz member by member, the rest byte for
    byte (module docstring). Returns the names."""
    def names(d: Path, m: dict) -> dict:
        out = {}
        for q in sorted(d.rglob("*")):
            if q.is_symlink():
                continue
            out[_norm(q.relative_to(d).as_posix(), m) + ("/" if q.is_dir() else "")] = q
        return out

    a, b = names(py_dir, uids[0]), names(nat_dir, uids[1])
    assert sorted(a) == sorted(b), sorted(set(a) ^ set(b))
    for name, pa in a.items():
        pb = b[name]
        if name.endswith("/"):
            continue
        if name.endswith(".json") and _parses(pa):
            _close(_norm(json.loads(pa.read_text()), uids[0]), _norm(json.loads(pb.read_text()), uids[1]),
                   name, window)
        elif name.endswith(".jsonl"):
            la, lb = pa.read_text().splitlines(), pb.read_text().splitlines()
            assert len(la) == len(lb), (name, la, lb)
            for k, (x, y) in enumerate(zip(la, lb)):
                _close(_norm(json.loads(x), uids[0]), _norm(json.loads(y), uids[1]), f"{name}:{k + 1}", window)
        elif name.endswith(".npz"):
            _same_npz(pa, pb, uids, name, window)
        else:
            assert _same_bytes(pa, pb), name
    return sorted(a)


_ACC = "acc/K1ABC_ab2dea7f"
_ACC_JSON = json.dumps({"callsign": "K1ABC", "codec_id": 0xD1D8, "created": 0.0, "foreign": [], "members": [],
                        "mode": 1, "picture_id": 0xAB2DEA7F, "updated": 0.0})


def _store_cases(f) -> list[Case]:
    def c(name, *args, files=None, **kw):
        return Case(name, list(args), {k: f[k] for k in (files or [args[0]])}, store=True, **kw)

    meta = "pb/meta.json"
    tiny = "tiny-1800.npz"      # (the refusals come before the receive: the smaller file)
    return [
        # (the default store, $QRSSTVAE_HOME, is test_store_sequence's)
        # --store, from a WAV, with a listener's locks held on both stores
        # (the script takes none: it stores beside a listener)
        c("store-wav", "ce-rx.wav", "--slot", SLOT, "--frame", "short", "--store", "st",
          locks=["st", "st/passband"]),
        # --passband elsewhere, an L pass
        c("passband-l", "l.npz", "--store", "st/", "--passband", "./pb"),
        # nothing found: the slot kept (an empty --passband is the default)...
        c("nothing", "noise.npz", "--store", "st", "--passband", ""),
        # ... and not, with --no-passband
        c("nothing-no-passband", "noise.npz", "--store", "st", "--no-passband"),
        # a capture at another rate: no passband store written, though the
        # line says the slot is kept (the script's condition)
        c("nothing-8k", "noise-8k.npz", "--store", "st"),
        # --store "": Path("") is the current directory
        c("store-empty", "noise-8k.npz", "--store", ""),
        # --passband and --no-passband with --no-store: ignored
        c("no-store-passband", "noise.npz", "--no-store", "--passband", "pb", "--no-passband"),
        # the store's and the passband store's refusals, uncaught in the script
        c("store-file", tiny, "--store", "f", texts={"f": "x"}),
        c("store-under-file", tiny, "--store", "f/st", texts={"f": "x"}),
        c("passband-file", tiny, "--store", "st", "--passband", "f", texts={"f": "x"}),
        c("passband-meta-dir", tiny, "--store", "st", "--passband", "pb", dirs=[meta]),
        c("passband-meta-garbage", tiny, "--store", "st", "--passband", "pb", texts={meta: "not json"},
          json_error=True),
        c("passband-meta-no-fs", tiny, "--store", "st", "--passband", "pb", texts={meta: '{"scale": 1.0}'}),
        c("passband-meta-fs-0", tiny, "--store", "st", "--passband", "pb",
          texts={meta: '{"fs": 0, "scale": 8192.0}'}),
        c("passband-meta-8k", tiny, "--store", "st", "--passband", "pb",
          texts={meta: '{"fs": 8000, "scale": 8192.0}'}),
        # an association that fails, after the pass's lines: a picture's
        # index without its arrays (read by rule 3, the TINY pass having
        # no header)
        c("acc-npz-missing", "tiny-1800.npz", "--store", "st", "--no-passband",
          texts={f"st/{_ACC}.json": _ACC_JSON}),
    ]


def test_store_side_by_side(cli, inputs, tmp_path):
    """Receives that store, side by side: the same report (with its
    "stored" line), exit status and stderr, and the same trees left behind
    (same_tree) -- the multi-pass store, the passband store, and nothing
    where nothing is to be written."""
    _, f = inputs
    cases = _store_cases(f)
    t0 = time.time()
    results = run_cases(cases, tmp_path, cli)
    window = (t0, time.time())
    trees = {}
    for c in cases:
        py, nat = results[c.name]
        check(c, py, nat)
        py_dir, nat_dir = tmp_path / c.name / "py", tmp_path / c.name / "nat"
        trees[c.name] = same_tree(py_dir, nat_dir, (_uid_map(py.out), _uid_map(nat.out)), window)
    print(f"store trees: worst relative difference {_Worst.rel:.2e}")

    def out(name):
        return results[name][1].out

    def err(name):
        return results[name][1].err

    stored = "  stored     K1ABC ab2dea7f by header\n"
    passband = ["", "fe_488687.i16", "fe_488687.json", "fe_488688.i16", "fe_488688.json", "meta.json"]
    empty = ["", "acc/", "passes/", "prov/"]
    store = empty + [f"{_ACC}.json", f"{_ACC}.npz", "passes/UID0.npz"]

    def under(d, names):
        return [f"{d}/{n}" for n in names]

    for name in ("store-wav", "passband-l"):
        assert out(name).endswith(stored) and out(name).count("\npass ") + out(name).startswith("pass ") == 1
    assert trees["store-wav"] == sorted(under("st", store + [".lock"]) + under("st/passband", passband + [".lock"]))
    assert trees["passband-l"] == sorted(under("pb", passband) + under("st", store))
    kept = "no CE or L signal found (slot kept for a later --retro search)\n"
    assert results["nothing"][1] == Run(1, kept, "")
    assert trees["nothing"] == sorted(under("st", empty) + under("st/passband", passband))
    assert results["nothing-no-passband"][1] == Run(1, "no CE or L signal found\n", "")
    assert trees["nothing-no-passband"] == under("st", empty)
    assert results["nothing-8k"][1] == Run(1, kept, "")          # (and nothing kept)
    assert trees["nothing-8k"] == under("st", empty)
    assert results["store-empty"][1] == Run(1, kept, "")
    assert trees["store-empty"] == empty[1:]
    assert results["no-store-passband"][1] == Run(1, "no CE or L signal found\n", "")
    assert trees["no-store-passband"] == []
    assert err("store-file") == "NotADirectoryError: [Errno 20] Not a directory: 'f/passes'\n"
    assert err("store-under-file") == "NotADirectoryError: [Errno 20] Not a directory: 'f/st/passes'\n"
    assert err("passband-file") == "FileExistsError: [Errno 17] File exists: 'f'\n"
    assert err("passband-meta-dir") == "IsADirectoryError: [Errno 21] Is a directory: 'pb/meta.json'\n"
    assert err("passband-meta-garbage") == "ValueError: Expecting value: line 1 column 1 (char 0)\n"
    assert err("passband-meta-no-fs") == "KeyError: 'fs'\n"
    assert err("passband-meta-fs-0") == "ValueError: the passband store's fs must be positive, not 0\n"
    assert err("passband-meta-8k") == "ValueError: the store holds 8000 Hz FE, not a 4000 Hz capture\n"
    for name in ("passband-file", "passband-meta-dir", "passband-meta-garbage", "passband-meta-no-fs",
                 "passband-meta-fs-0", "passband-meta-8k"):
        assert out(name) == "" and "st/passes/" in trees[name], name     # the store first
    assert out("acc-npz-missing").startswith("pass ") and "  mean W     " in out("acc-npz-missing")
    assert "stored" not in out("acc-npz-missing")
    assert err("acc-npz-missing") == f"FileNotFoundError: [Errno 2] No such file or directory: 'st/{_ACC}.npz'\n"
    assert "st/passes/UID0.npz" in trees["acc-npz-missing"]      # the pass was stored first


def test_store_sequence(cli, inputs, tmp_path):
    """Four receives into the default store ($QRSSTVAE_HOME, with its
    passband store), one per implementation, the trees compared after each
    receive: a headerless TINY pass with nothing to join (provisional); the
    SHORT pass with the picture's header (header, and the TINY pass's
    cluster then joins the picture by correlation); a second TINY pass, at
    another frequency (corr); the first again (already a member)."""
    _, f = inputs
    steps = [("tiny-1800.npz", "provisional (prov)"), ("ce.npz", "K1ABC ab2dea7f by header"),
             ("tiny-1000.npz", "K1ABC ab2dea7f by corr"), ("tiny-1800.npz", "K1ABC ab2dea7f by existing membership")]
    sides = {side: tmp_path / side for side in ("py", "nat")}
    for side, d in sides.items():
        d.mkdir()
        for name, _ in steps:
            if not (d / name).exists():
                os.symlink(f[name], d / name)
    outs = {"py": [], "nat": []}
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=2) as pool:
        for k, (name, want) in enumerate(steps):
            case = Case(f"step{k}", [name], store=True)
            jobs = (pool.submit(_run, [sys.executable, REPO_ROOT / "qrss_receive.py", *case.args], sides["py"],
                                _home(sides["py"])),
                    pool.submit(_run, [cli, "receive", *case.args], sides["nat"], _home(sides["nat"])))
            py, nat = (j.result() for j in jobs)
            check(case, py, nat)
            assert py.rc == 0 and _last_line(nat.out) == f"  stored     {want}", (k, py, nat)
            outs["py"].append(py.out)
            outs["nat"].append(nat.out)
            tree = same_tree(sides["py"], sides["nat"], (_uid_map(*outs["py"]), _uid_map(*outs["nat"])),
                             (t0, time.time()))
            if k == 0:
                assert "home/prov/c_UID0.json" in tree, tree
    acc = json.loads((sides["nat"] / "home" / f"{_ACC}.json").read_text())
    assert [(m["uid"], m["method"]) for m in _norm(acc["members"], _uid_map(*outs["nat"]))] == [
        ("UID1", "header"), ("UID0", "corr"), ("UID2", "corr")]
    hours = ["fe_488687.i16", "fe_488687.json", "fe_488688.i16", "fe_488688.json", "meta.json"]
    assert tree == sorted(["home/", "home/acc/", f"home/{_ACC}.json", f"home/{_ACC}.npz", "home/passband/",
                           *(f"home/passband/{h}" for h in hours), "home/passes/",
                           *(f"home/passes/UID{k}.npz" for k in range(3)), "home/prov/"])
    print(f"store sequence: worst relative difference {_Worst.rel:.2e}")


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
                 "--precision", "--old-edge-noise", "--old-gating", "--store", "--no-store", "--passband",
                 "--no-passband"):
        assert flag in r.out, flag
    # the docstring's store paragraph, the script's words
    assert ("Passes are stored and associated in the multi-pass store\n(--store DIR, default $QRSSTVAE_HOME or "
            "~/.local/share/qrsstvae)\nunless --no-store.") in r.out
    assert "not ported" not in r.out
    r = _run([cli, "--help"], REPO_ROOT)
    assert any(line.split()[:1] == ["receive"] and "qrss_receive.py" in line and "--no-store" not in line
               for line in r.out.splitlines())


# --- --image --------------------------------------------------------------------------------

def test_image_without_the_codec_half(cli, tmp_path):
    """Without the codec half --image is refused, before anything is read
    (with it, test_receive_side_by_side runs --image)."""
    if _renders(cli):
        pytest.skip("this sstvae-qrss renders --image")
    (tmp_path / "out").mkdir()
    r = _run([cli, "receive", "nope.npz", "--no-store", "--image", "out/pic.png"], tmp_path)
    assert r == Run(1, "", NO_RENDERER)


def test_image_names_png_only(cli, tmp_path):
    """The native image layer writes PNG: another extension is refused up
    front rather than written as PNG under a name that says otherwise."""
    if not _renders(cli):
        pytest.skip("this sstvae-qrss has no --image")
    (tmp_path / "x").mkdir()
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
    seconds are read by the slot parser, not strtod. Only a build with the
    model fetcher adopts the environment's locale (QCoreApplication runs
    setlocale(LC_ALL, "")); without it the program never leaves "C" and
    this holds by construction. Even there it pins the behaviour, not
    either layer: main puts LC_NUMERIC back to "C" and py_format maps the
    locale's point to '.', and taking out the reset alone still passes
    (measured), as would a plain snprintf behind the reset."""
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

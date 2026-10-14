"""`sstvae-qrss listen` (native/apps/sstvae_qrss/listen.cpp, on the
in-process API native/core/qrss/listen_api.hpp) against qrss_listen.py,
run side by side.

- --help: the script's, modulo the program's name, argparse's wrapping and
  the description's few sentences about this port (test_help);
- every refusal a user can hit, run through both in two directories: the
  exit status (1 for a SystemExit, 2 for argparse's), stderr (a SystemExit
  exactly, argparse's by its message, an uncaught exception by the
  traceback's last line) and the directory left behind (test_refusals);
- a TINY CE pass (1200 Hz) and a TINY L pass (2000 Hz) in one slot, each
  about -10 dB, replayed with --wav --slot --frame tiny through both into
  separate directories (test_tiny_replay_side_by_side): the same tiles
  (f_hz within 0.1 Hz, snr_db within 0.3 dB, z_ref and mean W close, the
  rest -- id, status, callsign, picture ID, segment, mode, waveform,
  frame, progress... -- equal), listener.log and state.json equal but for
  wall-clock stamps and receive durations, the store trees equal but for
  the pass uids (whose hash covers z's float32 bytes), the passband store
  byte for byte; with the codec compiled in, the tile pictures (PSNR);
- a SHORT CE pass with a header and callsign, the same way, and the
  stores it leaves read by qrss_decode.py and by `sstvae-qrss decode`
  (--list, --key) each on the other's store (test_short_replay_and_decode);
- stdin, f32 and s16, live: listening while audio flows; at its end "audio
  ended", state.json not listening, exit (test_stdin);
- the end of stdin with an end-of-slot receive in flight (a slot recovered
  from the passband store): exit within about 2 s, the receive abandoned
  (test_eof_with_a_receive_in_flight);
- recovery across implementations, each over the other's FLAC passband
  store (packed by its own pack_hour) (test_recover_across);
- the locks both ways, between running listeners of the two
  implementations: the same --state refused, the same store run without
  and said so, the same passband store likewise, and each lock free again
  after a clean stop (test_locks_both_ways);
- no libpython: not linked, and the C++ side of the TINY replay runs with
  no python on PATH and PYTHONHOME/PYTHONPATH unset (test_no_python).

The binary is $SSTVAE_QRSS_CLI, else `sstvae-qrss` beside the module's
build directory ($SSTVAE_NATIVE_MODULE_DIR/..); everything skips without
it, or when it has no listen subcommand. The script runs in a plain Python
subprocess, so --native-qrss does not reach it.
"""

import json
import math
import os
import pty
import re
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from qrss_helpers import Q_TEST, REPO_ROOT, codec_skip, model_dir

SLOT = "2025-10-01T00:00Z"
SLOT_UTC = "2025-10-01T00:00:00Z"
PY = [sys.executable, str(REPO_ROOT / "qrss_listen.py")]
NO_PICTURES = ("this build renders no pictures (the codec is not compiled in); the tiles have none, "
               "as with --no-pictures")
STOP_BOUND_S = 1.5


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
    r = subprocess.run([p, "--help"], capture_output=True, text=True, timeout=60, check=False)
    if not any(line.split()[:1] == ["listen"] for line in r.stdout.splitlines()):
        pytest.skip("this sstvae-qrss has no listen subcommand")
    return p


@dataclass
class Run:
    rc: int
    out: str
    err: str
    wall: float = 0.0
    cpu: float = 0.0
    rss_mb: float = 0.0


def _env(extra: dict | None = None) -> dict:
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env["COLUMNS"] = "200"
    for k in ("QRSS_OLD_EDGE_NOISE", "QRSS_OLD_GATING"):
        env.pop(k, None)
    env.update(extra or {})
    return env


def _no_python_env(tmp: Path) -> dict:
    """An environment with no python anywhere on PATH and no PYTHONHOME or
    PYTHONPATH."""
    empty = tmp / "empty-path"
    empty.mkdir(exist_ok=True)
    env = {k: v for k, v in os.environ.items() if not k.startswith("PYTHON") and k != "VIRTUAL_ENV"}
    env["PATH"] = str(empty)
    return env


def _run(cmd: list, cwd: Path, env: dict | None = None, stdin=subprocess.DEVNULL, timeout=180) -> Run:
    """Run to the end, with its wall time, CPU time (user + system) and
    peak RSS (os.wait4's rusage)."""
    cwd.mkdir(parents=True, exist_ok=True)
    out_f, err_f = cwd.parent / f".{cwd.name}.out", cwd.parent / f".{cwd.name}.err"
    t = time.monotonic()
    with open(out_f, "wb") as o, open(err_f, "wb") as e:
        p = subprocess.Popen([str(c) for c in cmd], cwd=cwd, env=env if env is not None else _env(), stdin=stdin,
                             stdout=o, stderr=e)
        deadline = time.monotonic() + timeout
        while True:
            pid, status, ru = os.wait4(p.pid, os.WNOHANG)
            if pid:
                break
            if time.monotonic() > deadline:
                p.kill()
                os.wait4(p.pid, 0)
                raise AssertionError(f"{cmd} did not end within {timeout} s")
            time.sleep(0.01)
        p.returncode = os.waitstatus_to_exitcode(status)
    wall = time.monotonic() - t
    return Run(p.returncode, out_f.read_text(errors="replace"), err_f.read_text(errors="replace"), wall,
               ru.ru_utime + ru.ru_stime, ru.ru_maxrss / 1024.0)


def _both(py_cmd, nat_cmd, py_dir: Path, nat_dir: Path, py_env=None, nat_env=None) -> tuple[Run, Run]:
    with ThreadPoolExecutor(2) as ex:
        a = ex.submit(_run, py_cmd, py_dir, py_env)
        b = ex.submit(_run, nat_cmd, nat_dir, nat_env)
        return a.result(), b.result()


def _last_line(text: str) -> str:
    lines = [ln for ln in text.splitlines() if ln.strip()]
    return lines[-1] if lines else ""


def _tree(d: Path) -> list[str]:
    if not d.exists():
        return []
    return sorted(p.relative_to(d).as_posix() + ("/" if p.is_dir() else "") for p in d.rglob("*")
                  if not p.name.startswith(".") or p.name == ".lock")


def _state(d: Path) -> dict | None:
    try:
        return json.loads((d / "state.json").read_text())
    except (OSError, ValueError):
        return None


def _log(d: Path) -> list[str]:
    return (d / "listener.log").read_text().splitlines()


def _norm_line(line: str) -> str:
    """A log line without a receive's duration, and without the stamp of
    "listening:" (the clock's, taken before any audio)."""
    m = re.match(r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ (listening: .*)$", line)
    if m:
        line = "<utc> " + m.group(1)
    return re.sub(r" \((\d+) s\)$", " (_ s)", line)


def _msg(line: str) -> str:
    """A log line's message: without its stamp, and a duration normalised."""
    return _norm_line(line).split(" ", 1)[1]


class Live:
    """A listener on a pipe: stderr read line by line on a thread."""

    def __init__(self, cmd: list, cwd: Path, env: dict | None = None):
        cwd.mkdir(parents=True, exist_ok=True)
        self.p = subprocess.Popen([str(c) for c in cmd], cwd=cwd, env=env if env is not None else _env(),
                                  stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        self.lines: list[str] = []
        self._eof = False
        self._cv = threading.Condition()
        self._t = threading.Thread(target=self._read, daemon=True)
        self._t.start()

    def _read(self):
        for raw in self.p.stderr:
            with self._cv:
                self.lines.append(raw.decode("utf-8", "replace").rstrip("\n"))
                self._cv.notify_all()
        with self._cv:
            self._eof = True
            self._cv.notify_all()

    def wait_line(self, pred, timeout: float) -> bool:
        with self._cv:
            self._cv.wait_for(lambda: self._eof or any(pred(ln) for ln in self.lines), timeout)
            return any(pred(ln) for ln in self.lines)

    def write(self, data: bytes):
        self.p.stdin.write(data)
        self.p.stdin.flush()

    def end(self, timeout: float = 120.0) -> tuple[int, float]:
        """Close stdin: the exit status and the seconds to exit."""
        t = time.monotonic()
        self.p.stdin.close()
        rc = self.p.wait(timeout)
        took = time.monotonic() - t
        self._t.join(10)
        return rc, took

    def kill(self):
        if self.p.poll() is None:
            self.p.kill()
            self.p.wait(30)


def _wait_for(pred, timeout: float, step: float = 0.02) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if pred():
            return True
        time.sleep(step)
    return pred()


# --- --help ---------------------------------------------------------------------------------

def test_help(cli, tmp_path):
    """The script's help, word for word, modulo the program's name, the
    wrapping, and the sentences about this port: its examples name
    `sstvae-qrss listen`, its end-of-slot receive is a thread, it packs FLAC
    with its own encoder, and a last paragraph says what Ctrl-C, SIGTERM
    and a codec-less build do."""
    py = _run([*PY, "--help"], tmp_path / "py")
    nat = _run([cli, "listen", "--help"], tmp_path / "nat")
    assert py.rc == nat.rc == 0 and nat.err == "" and py.err == ""
    want = " ".join(py.out.split())
    for a, b in (("usage: qrss_listen.py ", "usage: sstvae-qrss listen "),
                 ("| python qrss_listen.py", "| sstvae-qrss listen"),
                 ("python qrss_listen.py --wav", "sstvae-qrss listen --wav"),
                 ("in a second process,", "on a second thread,"),
                 (("With the soundfile package installed (`uv pip install soundfile`; a `uv sync` removes it "
                   "again), each finished hour is packed into FLAC."),
                  "Each finished hour is packed into FLAC (sstvae-qrss's own encoder: nothing to install).")):
        assert a in want, a
        want = want.replace(a, b)
    extra = ("sstvae-qrss: Ctrl-C and SIGTERM both stop it as Ctrl-C stops the script (\"stopped\" in the "
             "log, state.json not listening). Without the codec compiled in (SSTVAE_BUILD_CODEC), the "
             "tiles have no pictures, as with --no-pictures, and it says so once. ")
    got = " ".join(nat.out.split())
    assert extra in got
    assert got.replace(extra, "") == want
    r = _run([cli, "--help"], tmp_path / "top")
    assert any(line.split()[:1] == ["listen"] and "qrss_listen.py" in line for line in r.out.splitlines())


# --- refusals -------------------------------------------------------------------------------

_HOLDER = """
import sys
sys.path[:0] = [{repo!r}]
from sstvae.qrss.live import dir_lock
f = dir_lock(sys.argv[1])
print("held" if f is not None else "refused", flush=True)
sys.stdin.readline()
"""


def _holder(path: Path):
    p = subprocess.Popen([sys.executable, "-c", _HOLDER.format(repo=str(REPO_ROOT)), str(path)],
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    assert p.stdout.readline().strip() == "held"
    return p


def _release(p):
    p.stdin.write("\n")
    p.stdin.flush()
    assert p.wait(30) == 0


ST = ["--no-store", "--state", "st"]
# (name, arguments, how it ends: "exit" a SystemExit, "usage" argparse's, "raise" an uncaught exception)
REFUSALS = [
    ("frame-bad", ["--frame", "x", *ST], "exit"),
    ("frame-empty", ["--frame", "", *ST], "exit"),
    ("frame-ce-only", ["--frame", "full-v1", *ST], "exit"),
    ("frame-ce-only-l", ["--frame", "medium-v1", "--just-l", *ST], "exit"),
    ("both-just", ["--just-ce", "--just-l", *ST], "usage"),
    ("refresh-0", ["--refresh", "0", *ST], "exit"),
    ("refresh-neg", ["--refresh", "-2.5", *ST], "exit"),
    ("refresh-text", ["--refresh", "soon", *ST], "usage"),
    ("format-bad", ["--format", "f64", *ST], "usage"),
    ("precision-bad", ["--precision", "fp64", *ST], "usage"),
    ("unknown-flag", ["--bogus", *ST], "usage"),
    ("positional", ["extra", *ST], "usage"),
    ("wav-no-time", ["--wav", "x.wav", *ST], "exit"),
    ("start-bad", ["--wav", "x.wav", "--start", "yesterday", *ST], "exit"),
    ("start-wins", ["--wav", "nope.wav", "--start", "2025-10-01T00:00:00Z", "--slot", "bad", *ST], "exit"),
    ("slot-bad", ["--wav", "x.wav", "--slot", "nope", *ST], "exit"),
    ("slot-before-1970", ["--wav", "x.wav", "--slot", "1969-12-31T23:45Z", *ST], "exit"),
    ("wav-missing", ["--wav", "nope.wav", "--slot", SLOT, *ST], "exit"),
    ("wav-dir", ["--wav", "adir", "--slot", SLOT, *ST], "exit"),
    ("wav-not-riff", ["--wav", "text.wav", "--slot", SLOT, *ST], "raise"),
    ("held-state", ["--frame", "tiny", "--store", "store", "--state", "st"], "exit"),
    ("tty", ["--frame", "tiny", *ST], "exit"),
]


def _refusal_run(cmd: list, d: Path, name: str) -> Run:
    d.mkdir(parents=True)
    (d / "adir").mkdir()
    (d / "text.wav").write_text("hello, this is not a WAV file\n")
    holder = _holder(d / "st") if name == "held-state" else None
    try:
        if name == "tty":
            master, slave = pty.openpty()
            try:
                return _run(cmd, d, stdin=slave, timeout=60)
            finally:
                os.close(slave)
                os.close(master)
        return _run(cmd, d, timeout=60)
    finally:
        if holder is not None:
            _release(holder)


def test_refusals(cli, tmp_path):
    """Each refusal through both: the same status, the same message (and
    for argparse's, the same usage line), the same directory left."""
    jobs = {}
    with ThreadPoolExecutor(4) as ex:
        for name, args, _ in REFUSALS:
            jobs[name] = (ex.submit(_refusal_run, [*PY, *args], tmp_path / name / "py", name),
                          ex.submit(_refusal_run, [cli, "listen", *args], tmp_path / name / "nat", name))
    for name, args, how in REFUSALS:
        py, nat = (j.result() for j in jobs[name])
        assert py.rc == nat.rc == (2 if how == "usage" else 1), (name, py, nat)
        assert nat.out == py.out == "", name
        if how == "exit":
            assert nat.err == py.err, (name, py.err, nat.err)
        elif how == "usage":
            pl, nl = _last_line(py.err), _last_line(nat.err)
            assert pl.startswith("qrss_listen.py: error: ") and nl.startswith("sstvae-qrss listen: error: ")
            assert nl.split(": error: ", 1)[1] == pl.split(": error: ", 1)[1], (name, pl, nl)
            assert " ".join(nat.err.split("\n")[:-2]).split() == " ".join(
                py.err.replace("usage: qrss_listen.py", "usage: sstvae-qrss listen").split("\n")[:-2]).split(), name
        else:
            assert _last_line(nat.err) == _last_line(py.err), (name, py.err, nat.err)
        # the directory each leaves (the tile directory and its lock, tiles/ ...)
        assert _tree(tmp_path / name / "nat") == _tree(tmp_path / name / "py"), name
    by = {name: jobs[name][1].result() for name, _, _ in REFUSALS}
    assert by["frame-bad"].err == ("--frame 'x': one of full, medium, short, tiny, l-full, l-medium, l-short, "
                                   "l-tiny\n")
    assert by["frame-ce-only"].err == (
        "--frame 'full-v1' is CE only: waveform L has no frame of that shape (L frames: l-full, l-medium, "
        "l-short, l-tiny); add --just-ce to receive it\n")
    assert by["refresh-0"].err == by["refresh-neg"].err == "--refresh must be positive\n"
    assert by["wav-no-time"].err == "--wav needs --start or --slot\n"
    assert by["wav-missing"].err == "nope.wav: no such file\n"
    assert by["wav-dir"].err == "adir: no such file\n"
    assert by["start-wins"].err == "nope.wav: no such file\n"
    assert by["start-bad"].err.startswith("--start 'yesterday': ")
    assert by["slot-bad"].err.startswith("--slot 'nope': ")
    assert by["tty"].err == "pipe 8 kHz mono audio into stdin (see --help), or use --wav\n"
    assert by["held-state"].err == ("another listener is writing tiles to st; stop it, or give this one its "
                                    "own --state DIR\n")
    assert _last_line(by["both-just"].err) == ("sstvae-qrss listen: error: argument --just-l: not allowed with "
                                               "argument --just-ce")


# --- the replays ----------------------------------------------------------------------------

def _tiny_wav(path: Path) -> None:
    """A TINY CE pass at 1200 Hz and a TINY L pass (pilot) at 2000 Hz in
    slot Q_TEST, from 12 s before it, with white noise putting each at
    about -10 dB in 2500 Hz: weak enough that their missing headers do not
    get them dropped as a carrier (HEADERLESS_MAX_SNR_DB)."""
    from qrss_fakes import synthetic_picture

    from sstvae import wavio
    from sstvae.qrss import frame, lframe, tx

    ce = tx.transmit_audio(synthetic_picture(0), Q_TEST, 1200, spec=frame.TINY)
    el = tx.transmit_audio(synthetic_picture(1), Q_TEST, 2000, spec=lframe.L_TINY)
    x = np.zeros(max(len(ce), len(el)))
    x[:len(ce)] += ce
    x[:len(el)] += el
    x += 1.4 * np.random.default_rng(5).standard_normal(len(x))
    wavio.write_wav(str(path), x)


def _short_wav(path: Path) -> None:
    """A SHORT CE pass at 1500 Hz with a header and callsign windows
    (K1ABC), about -6 dB."""
    from qrss_fakes import make_header, synthetic_picture

    from sstvae import wavio
    from sstvae.qrss import frame, tx

    sp = synthetic_picture(0)
    x = tx.transmit_audio(sp, Q_TEST, 1500, spec=frame.SHORT, header=make_header(sp, callsign="K1ABC"))
    x = x + 0.9 * np.random.default_rng(7).standard_normal(len(x))
    wavio.write_wav(str(path), x)


def _renders(cli: Path, tmp: Path) -> bool:
    """Whether this sstvae-qrss draws pictures: one without the codec says
    so as it starts (here on an empty stdin, which ends it at once)."""
    r = _run([cli, "listen", "--frame", "tiny", "--no-store", "--state", "st"], tmp / "probe", timeout=60)
    assert r.rc == 0, r
    return NO_PICTURES not in r.err.splitlines()


@dataclass
class Replay:
    base: Path
    py: Run
    nat: Run
    pictures: bool


def _replay(cli, base: Path, wav: Path, frame_args: list, nat_env: dict | None) -> Replay:
    pictures = _renders(cli, base)
    extra = {}
    if pictures:
        md = model_dir()
        if md is None or not Path(md).is_dir():
            codec_skip("the v5 models are not available ($QRSSTVAE_MODEL_DIR)")
        extra = {"QRSSTVAE_MODEL_DIR": md}
        if nat_env is not None:
            nat_env = {**nat_env, **extra}
    args = ["--wav", str(wav), "--slot", SLOT, *frame_args, *([] if pictures else ["--no-pictures"]),
            "--store", "store", "--state", "state"]
    py, nat = _both([*PY, *args], [cli, "listen", *args], base / "py", base / "nat", _env(extra),
                    nat_env if nat_env is not None else _env(extra))
    print(f"\n{base.name}: qrss_listen.py {py.wall:.1f} s wall, {py.cpu:.1f} s CPU, {py.rss_mb:.0f} MB peak; "
          f"sstvae-qrss listen {nat.wall:.1f} s wall, {nat.cpu:.1f} s CPU, {nat.rss_mb:.0f} MB peak")
    assert py.rc == 0 and nat.rc == 0, (py, nat)
    return Replay(base, py, nat, pictures)


@pytest.fixture(scope="module")
def tiny(cli, tmp_path_factory):
    base = tmp_path_factory.mktemp("listen-tiny")
    wav = base / "tiny.wav"
    _tiny_wav(wav)
    # the C++ with no python to be found (test_no_python)
    return _replay(cli, base, wav, ["--frame", "tiny"], _no_python_env(base))


@pytest.fixture(scope="module")
def short(cli, tmp_path_factory):
    base = tmp_path_factory.mktemp("listen-short")
    wav = base / "short.wav"
    _short_wav(wav)
    return _replay(cli, base, wav, ["--frame", "short", "--just-ce"], None)


TOL = {"f_hz": 0.1, "snr_db": 0.3, "mean_w_db": 0.05}


def _compare_tiles(a: list[dict], b: list[dict], ignore=()) -> dict:
    """The worst differences of the toleranced fields; the rest equal
    (but those in `ignore`)."""
    assert len(a) == len(b) > 0
    worst = {k: 0.0 for k in (*TOL, "z_ref_rel", "confidence")}
    for x, y in zip(a, b):
        assert sorted(x) == sorted(y)
        for k in x:
            if k in ignore:
                continue
            if k in TOL:
                d = abs(x[k] - y[k])
                worst[k] = max(worst[k], d)
                assert d < TOL[k], (x["id"], k, x[k], y[k])
            elif k == "z_ref":
                d = abs(x[k] - y[k]) / abs(x[k])
                worst["z_ref_rel"] = max(worst["z_ref_rel"], d)
                assert d < 1e-3, (x["id"], k, x[k], y[k])
            elif k == "confidence":
                assert len(x[k]) == len(y[k])
                for u, v in zip(x[k], y[k]):
                    assert (u is None) == (v is None), (x["id"], k)
                    if u is not None:
                        worst[k] = max(worst[k], abs(u - v))
                        assert abs(u - v) < 0.05, (x["id"], k, u, v)
            elif k == "uid":
                # <q>_<mHz>_<sha8>: the hash covers z's float32 bytes
                assert x[k].split("_")[:1] == y[k].split("_")[:1], (x["id"], x[k], y[k])
            else:
                assert x[k] == y[k], (x["id"], k, x[k], y[k])
    return worst


def _uids(d: Path) -> dict:
    return {t["uid"]: f"UID{k}" for k, t in enumerate(_state(d / "state")["tiles"]) if t.get("uid")}


def _store_tree(d: Path) -> list[str]:
    names = _tree(d / "store")
    for u, v in _uids(d).items():
        names = [n.replace(u, v) for n in names]
    return sorted(names)


def _racy(line: str) -> bool:
    """The final live refresh's line ("N signal(s) at 100%"): it runs on
    the live worker beside the end-of-slot receive on the finish worker, and
    is dropped if the finish ends first (in both implementations, so either
    may log it or not)."""
    return " signal(s) at 100% (" in line


def _lines(xs: list[str]) -> list[str]:
    return [_norm_line(x) for x in xs if not _racy(x)]


def _same_run(r: Replay):
    """state.json, listener.log, the stores and the tiles of the two."""
    py, nat = r.base / "py", r.base / "nat"
    sp, sn = _state(py / "state"), _state(nat / "state")
    assert sorted(sp) == sorted(sn)
    for k in sp:
        if k == "updated":
            continue
        if k == "log":
            assert _lines(sn[k]) == _lines(sp[k])
        elif k != "tiles":
            assert sn[k] == sp[k], (k, sp[k], sn[k])
    assert sp["listening"] is False and sp["finished"] == [Q_TEST]
    worst = _compare_tiles(sp["tiles"], sn["tiles"])
    print(f"{r.base.name}: {len(sp['tiles'])} tiles; worst |d f_hz| {worst['f_hz']:.2e} Hz, "
          f"|d snr_db| {worst['snr_db']:.2e} dB, |d mean_w_db| {worst['mean_w_db']:.2e} dB, "
          f"z_ref {worst['z_ref_rel']:.2e} relative, confidence {worst['confidence']:.2e}")
    lp, ln = _log(py / "state"), _log(nat / "state")
    assert _lines(ln) == _lines(lp)
    assert ln[-1].endswith(" audio ended")
    assert _store_tree(nat) == _store_tree(py)
    # the passband store, byte for byte (the same front end, the same int16
    # quantisation); an hour packed during the replay sample for sample (two
    # FLAC encoders: soundfile's libFLAC and the C++'s own)
    pb_p, pb_n = py / "store" / "passband", nat / "store" / "passband"
    assert _tree(pb_n) == _tree(pb_p)
    for f in _tree(pb_p):
        if f.endswith(".flac"):
            import soundfile

            a, fa = soundfile.read(str(pb_p / f), dtype="int16", always_2d=True)
            b, fb = soundfile.read(str(pb_n / f), dtype="int16", always_2d=True)
            assert fa == fb and a.shape == b.shape and np.array_equal(a, b), f
            print(f"{r.base.name}: {f} packed by both, {a.shape[0]} samples equal "
                  f"({(pb_p / f).stat().st_size} bytes by soundfile, {(pb_n / f).stat().st_size} by the C++)")
        elif not f.endswith("/") and f != ".lock":
            assert (pb_n / f).read_bytes() == (pb_p / f).read_bytes(), f
    # stderr: "tiles in state", then the log's lines
    assert _lines(r.nat.err.splitlines()) == _lines(r.py.err.splitlines())
    if r.pictures:
        from PIL import Image

        for t in sp["tiles"]:
            if t.get("image"):
                a = np.asarray(Image.open(py / "state" / t["image"]), dtype=np.float64)
                b = np.asarray(Image.open(nat / "state" / t["image"]), dtype=np.float64)
                mse = float(np.mean((a - b) ** 2))
                psnr = math.inf if mse == 0 else 10 * math.log10(255.0 ** 2 / mse)
                print(f"{r.base.name}: tile {t['id']} picture PSNR {psnr:.1f} dB (C++ vs Python)")
                assert psnr > 30.0, (t["id"], psnr)
    return sp


def test_tiny_replay_side_by_side(tiny):
    sp = _same_run(tiny)
    by = {t["id"]: t for t in sp["tiles"]}
    assert sorted(by) == ["1954752-1200", "1954752-2000"]
    assert by["1954752-1200"]["waveform"] == "CE" and by["1954752-2000"]["waveform"] == "L"
    assert all(t["status"] == "complete" and t["frame"] in ("tiny", "l-tiny") for t in sp["tiles"])
    for side in ("py", "nat"):
        log = map(_norm_line, _log(tiny.base / side / "state"))
        assert any(ln.endswith("frame over, 2 pass(es) (_ s)") for ln in log)
    if not tiny.pictures:
        assert NO_PICTURES not in tiny.nat.err     # (--no-pictures asked for none)


@pytest.mark.slow          # (a SHORT pass: about 22 s with both replays)
def test_short_replay_and_decode(cli, short, tmp_path):
    """The header pass: its tile, then the stores read by both decoders."""
    sp = _same_run(short)
    (t,) = sp["tiles"]
    assert t["callsign"] == "K1ABC" and t["status"] == "complete" and t["mode"] == "A"
    key = f"K1ABC:{t['picture_id']}"
    env = _env({"QRSSTVAE_MODEL_DIR": str(tmp_path / "no-models")})
    outs = {}
    for store in ("py", "nat"):
        root = short.base / store / "store"
        for name, args in (("list", ["--list"]), ("key", ["out.png", "--key", key])):
            py = _run([sys.executable, str(REPO_ROOT / "qrss_decode.py"), *args, "--store", root],
                      tmp_path / f"{store}-{name}-py", env)
            nat = _run([cli, "decode", *args, "--store", root], tmp_path / f"{store}-{name}-nat", env)
            # the same reading of one store, whichever wrote it
            assert nat.out == py.out and nat.rc == py.rc, (store, name, py, nat)
            if name == "list":
                assert py.rc == 0 and py.out.startswith("picture K1ABC "), py
            else:
                assert py.rc == 1 and "1 passes" in py.out, py         # (no models: the picture is refused)
            outs[store, name] = py.out
    # and each store reads as the other: the same summary but for the pass's uid
    for name in ("list", "key"):
        a, b = outs["py", name], outs["nat", name]
        for u, v in _uids(short.base / "py").items():
            a = a.replace(u, v)
        for u, v in _uids(short.base / "nat").items():
            b = b.replace(u, v)
        assert a == b, (name, a, b)


def test_no_python(cli, tiny):
    """The C++ links no libpython, and its side of the TINY replay ran with
    no python on PATH and no PYTHONHOME or PYTHONPATH (the fixture)."""
    for tool, args in (("readelf", ["-d"]), ("ldd", [])):
        exe = shutil.which(tool)
        if exe is None:
            continue
        r = subprocess.run([exe, *args, str(cli)], capture_output=True, text=True, timeout=60, check=False)
        assert r.returncode == 0, r
        assert "libpython" not in r.stdout, r.stdout
    assert tiny.nat.rc == 0 and len(_state(tiny.base / "nat" / "state")["tiles"]) == 2


# --- stdin ----------------------------------------------------------------------------------

def _noise(n: int, fmt: str, seed: int) -> bytes:
    x = 0.05 * np.random.default_rng(seed).standard_normal(n)
    if fmt == "s16":
        return np.round(x * 32767).astype("<i2").tobytes()
    return x.astype("<f4").tobytes()


@pytest.mark.parametrize("fmt", ["f32", "s16"])
def test_stdin(cli, tmp_path, fmt):
    """Live audio on stdin, paced at real time: listening while it flows,
    the source named as the script names it; at its end "audio ended",
    state.json not listening, exit 0 (the C++ within about 2 s)."""
    args = ["--format", fmt, "--frame", "tiny", "--no-pictures", "--no-store", "--state", "st", "--recover-hours", "0"]
    procs = {"py": Live([*PY, *args], tmp_path / "py"), "nat": Live([cli, "listen", *args], tmp_path / "nat")}
    try:
        for p in procs.values():
            assert p.wait_line(lambda ln: ln.startswith("tiles in "), 60), p.lines
        # 0.1 s chunks at real time, until both have said they listen, and a second more
        listening = {}
        t = time.monotonic()
        k = 0
        while time.monotonic() - t < 20:
            data = _noise(800, fmt, k)
            for p in procs.values():
                p.write(data)
            k += 1
            time.sleep(0.1)
            for name in procs:
                s = _state(tmp_path / name / "st")
                if s is not None and s["listening"] and name not in listening:
                    listening[name] = k
            if len(listening) == 2 and k >= max(listening.values()) + 10:
                break
        assert sorted(listening) == ["nat", "py"], procs["nat"].lines
        states = {name: _state(tmp_path / name / "st") for name in procs}
        assert states["nat"]["source"] == states["py"]["source"] == f"stdin ({fmt}, 8 kHz)"
        assert states["nat"]["listening"] and states["py"]["listening"]
        ended = {name: p.end() for name, p in procs.items()}
    finally:
        for p in procs.values():
            p.kill()
    print(f"\nstdin {fmt}: exit {ended['nat'][1]:.2f} s after the end of the audio (C++), "
          f"{ended['py'][1]:.2f} s (Python)")
    for name, p in procs.items():
        rc, _ = ended[name]
        assert rc == 0, (name, p.lines)
        s = _state(tmp_path / name / "st")
        assert s["listening"] is False
        assert _log(tmp_path / name / "st")[-1].endswith(" audio ended")
        assert p.lines[-1].endswith(" audio ended")
    assert ended["nat"][1] < 2.0
    # the same lines (but those of the current slot, which depend on the wall clock)
    said = {name: [_msg(x) for x in p.lines if not x.startswith("tiles in ") and not _msg(x).startswith("slot ")]
            for name, p in procs.items()}
    assert said["nat"] == said["py"] and said["nat"][0].startswith("listening: stdin ")


@pytest.mark.parametrize("sig", ["SIGINT", "SIGTERM"])
def test_a_signal_stops_it(cli, tmp_path, sig):
    """Ctrl-C (SIGINT) stops either implementation alike: "stopped" last, in
    listener.log too, state.json not listening, exit 0, the three locks
    free. SIGTERM stops the C++ the same way (the script dies on SIGTERM
    with nothing written: not compared).

    The script then aborts at interpreter shutdown (SIGABRT, "Fatal Python
    error: _enter_buffered_busy: could not acquire lock for
    <_io.BufferedReader name='<stdin>'>"): StdinSource's daemon thread is
    blocked in sys.stdin.buffer.read1 holding the reader's lock, which the
    interpreter's finalisation then needs. Its exit status is allowed to be
    that, and only its lines up to "stopped" are compared."""
    import signal

    from sstvae.qrss.live import dir_lock

    kinds = {"nat": [cli, "listen"], "py": PY} if sig == "SIGINT" else {"nat": [cli, "listen"]}
    args = ["--frame", "tiny", "--no-pictures", "--store", "store", "--state", "st", "--recover-hours", "0"]
    procs = {name: Live([*cmd, *args], tmp_path / name) for name, cmd in kinds.items()}
    try:
        for name, p in procs.items():
            assert p.wait_line(lambda ln: " listening: stdin " in ln, 60), (name, p.lines)
        t = time.monotonic()
        for p in procs.values():
            p.p.send_signal(getattr(signal, sig))
        rcs = {name: p.p.wait(30) for name, p in procs.items()}
        took = time.monotonic() - t
        for p in procs.values():
            p._t.join(10)
    finally:
        for p in procs.values():
            p.kill()
    print(f"\n{sig}: exited after {took:.2f} s")
    lines = {}
    for name, p in procs.items():
        d = tmp_path / name
        stop = next((k for k, ln in enumerate(p.lines) if ln.endswith(" stopped")), None)
        assert stop is not None, (name, p.lines)
        lines[name] = p.lines[:stop + 1]
        if name == "py" and rcs[name] == -signal.SIGABRT:
            assert p.lines[stop + 1].startswith("Fatal Python error: _enter_buffered_busy: "), p.lines
        else:
            assert rcs[name] == 0 and stop == len(p.lines) - 1, (name, rcs[name], p.lines)
        assert _log(d / "st")[-1].endswith(" stopped"), name
        assert _state(d / "st")["listening"] is False, name
        for lock in (d / "st", d / "store", d / "store" / "passband"):
            f = dir_lock(lock)
            assert f is not None, (name, lock)
            f.close()
    said = {name: [_msg(x) for x in xs if not x.startswith("tiles in ") and not _msg(x).startswith("slot ")]
            for name, xs in lines.items()}
    assert said["nat"][-2:] == ["listening: stdin (f32, 8 kHz), frame tiny, waveforms CE (tiny) and L (l-tiny)",
                                "stopped"], said["nat"]
    if "py" in said:
        assert said["nat"] == said["py"]


def _hours_back() -> str:
    """--recover-hours reaching back to slot Q_TEST from now."""
    return f"{(time.time() - Q_TEST * 900.0) / 3600.0 + 2.0:.0f}"


def _recovering(store_from: Path, root: Path) -> Path:
    """A fresh store holding only `store_from`'s passband store."""
    shutil.copytree(store_from, root / "store" / "passband", ignore=shutil.ignore_patterns(".lock"))
    return root / "store"


def test_eof_with_a_receive_in_flight(cli, tiny, tmp_path):
    """The end of stdin while the end-of-slot receive of a slot recovered
    from the passband store runs (a few seconds of TINY CE + L receive):
    exit within about 2 s, "audio ended" last, state.json not listening,
    the receive abandoned (its slot not finished), the locks free."""
    store = _recovering(tiny.base / "nat" / "store" / "passband", tmp_path)
    p = Live([cli, "listen", "--frame", "tiny", "--no-pictures", "--store", store, "--state", "st",
              "--recover-hours", _hours_back()], tmp_path)
    try:
        assert p.wait_line(lambda ln: ln.endswith("receiving it from the passband store"), 60), p.lines
        assert p.wait_line(lambda ln: " listening: stdin " in ln, 60), p.lines
        # run() steps after a second without audio, which starts the end-of-slot
        # receive (about 3 s of work); the end of stdin comes a moment into it
        busy = _wait_for(lambda: ((_state(tmp_path / "st") or {}).get("busy") or "").startswith("finish"), 1.5)
        time.sleep(0.2 if busy else 0.0)
        rc, took = p.end()
    finally:
        p.kill()
    print(f"\nEOF with an end-of-slot receive in flight: exit after {took:.2f} s")
    assert rc == 0, p.lines
    assert took < 2.0
    assert not any("frame over" in ln for ln in p.lines), "the receive finished: nothing was in flight"
    s = _state(tmp_path / "st")
    assert s["listening"] is False and Q_TEST not in s["finished"]
    assert _log(tmp_path / "st")[-1].endswith(" audio ended") and p.lines[-1].endswith(" audio ended")
    from sstvae.qrss.live import dir_lock

    for d in (tmp_path / "st", store, store / "passband"):
        f = dir_lock(d)
        assert f is not None, d
        f.close()


_PACK = """
import sys
sys.path[:0] = [{repo!r}]
from sstvae.qrss.frontend import PassbandStore
pb = PassbandStore(sys.argv[1], pack=True)
assert pb.pack, "soundfile is needed to pack"
for h in pb.raw_hours():
    assert pb.pack_hour(h), h
print(sorted(p.name for p in pb.root.iterdir()))
"""


def _retired(s: dict | None) -> bool:
    """state.json says slot Q_TEST's receive is over: the slot retired, its
    tiles complete."""
    return (s is not None and not any(x["q"] == Q_TEST for x in s["slots"]) and len(s["tiles"]) > 0
            and all(t["status"] == "complete" for t in s["tiles"]))


def test_recover_across(cli, native, tiny, tmp_path):
    """Each implementation recovers the TINY slot from the other's FLAC
    passband store: the Python-written one packed by Python's pack_hour
    (soundfile), recovered by the C++ (dr_flac); the C++-written one packed
    by the C++ (its own encoder), recovered by Python (soundfile). The two
    stores hold the same int16 samples (test_tiny_replay_side_by_side), so
    the two recoveries agree as the replays do."""
    pbmod = getattr(getattr(native, "qrss", None), "passband", None)
    if pbmod is None:
        pytest.skip("sstvae_native.qrss.passband not built")
    a = _recovering(tiny.base / "py" / "store" / "passband", tmp_path / "a")
    r = subprocess.run([sys.executable, "-c", _PACK.format(repo=str(REPO_ROOT)), str(a / "passband")],
                       capture_output=True, text=True, timeout=120, env=_env(), check=False)
    assert r.returncode == 0, r
    b = _recovering(tiny.base / "nat" / "store" / "passband", tmp_path / "b")
    st = pbmod.PassbandStore(str(b / "passband"), pack=True)
    for h in st.raw_hours():
        assert st.pack_hour(h), h
    del st
    for d in (a, b):
        names = sorted(p.name for p in (d / "passband").iterdir())
        assert any(n.endswith(".flac") for n in names) and not any(n.endswith(".i16") for n in names), names

    common = ["--frame", "tiny", "--no-pictures", "--state", "st", "--recover-hours", _hours_back()]
    procs = {"nat": Live([cli, "listen", "--store", a, *common], tmp_path / "a"),
             "py": Live([*PY, "--store", b, *common], tmp_path / "b")}
    try:
        for name, p in procs.items():
            assert p.wait_line(lambda ln: "frame over" in ln, 120), (name, p.lines)
            # retired at the next step (a second later, with no audio), and written
            d = tmp_path / ("a" if name == "nat" else "b") / "st"
            assert _wait_for(lambda d=d: _retired(_state(d)), 30), (name, _state(d))
        ended = {name: p.end() for name, p in procs.items()}
    finally:
        for p in procs.values():
            p.kill()
    for name, (rc, _) in ended.items():
        assert rc == 0, (name, procs[name].lines)
    lines = {name: [_msg(x) for x in p.lines if not x.startswith("tiles in ")] for name, p in procs.items()}
    assert lines["nat"] == lines["py"]
    assert f"slot {SLOT_UTC}: ended while no listener finished it; receiving it from the passband store" in \
        lines["nat"]
    assert f"slot {SLOT_UTC}: frame over, 2 pass(es), from the passband store (_ s)" in lines["nat"]
    sa, sb = _state(tmp_path / "a" / "st"), _state(tmp_path / "b" / "st")
    worst = _compare_tiles(sb["tiles"], sa["tiles"], ignore=("updated",))
    print(f"\nrecovered across: worst |d f_hz| {worst['f_hz']:.2e} Hz, |d snr_db| {worst['snr_db']:.2e} dB")
    assert len(sa["tiles"]) == 2 and all(t["status"] == "complete" for t in sa["tiles"])
    # (state.json lists only the last 48 h of finished slots, and Q_TEST is older)
    assert sa["finished"] == sb["finished"] == []


# --- the locks ------------------------------------------------------------------------------

def test_locks_both_ways(cli, tmp_path):
    """A running listener of one implementation holds its three directories
    against the other's: the same --state refused (exit 1), the same store
    run without (and said so, and then no passband store either), the same
    passband store run without (said so); all three free again after the
    holder's clean stop."""
    for holder_kind in ("nat", "py"):
        d = tmp_path / holder_kind
        d.mkdir()
        hold_cmd = [cli, "listen"] if holder_kind == "nat" else PY
        other = PY if holder_kind == "nat" else [cli, "listen"]
        st, store, pb = d / "st", d / "store", d / "store" / "passband"
        store_notice = (f"another listener is using the store in {store}; this one runs without a store (pass "
                        "--store DIR for one of its own)")
        pb_notice = f"another listener is writing the passband store in {pb}; this one keeps none"
        tiny_np = ["--frame", "tiny", "--no-pictures"]
        holder = Live([*hold_cmd, *tiny_np, "--store", store, "--state", st, "--recover-hours", "0"], d)
        try:
            assert holder.wait_line(lambda ln: ln.startswith("tiles in "), 60), holder.lines
            with ThreadPoolExecutor(3) as ex:
                same_state = ex.submit(_run, [*other, *tiny_np, "--store", d / "s1", "--state", st], d / "o1")
                same_store = ex.submit(_run, [*other, *tiny_np, "--store", store, "--state", d / "st2"], d / "o2")
                same_pb = ex.submit(_run, [*other, *tiny_np, "--store", d / "s3", "--passband", pb,
                                           "--state", d / "st3"], d / "o3")
                r1, r2, r3 = same_state.result(), same_store.result(), same_pb.result()
            assert holder.p.poll() is None, holder.lines
        finally:
            rc, _ = holder.end()
            holder.kill()
        assert rc == 0, holder.lines
        assert r1.rc == 1 and r1.out == "", r1
        assert r1.err == f"another listener is writing tiles to {st}; stop it, or give this one its own --state DIR\n"
        assert r2.rc == 0 and r2.err.splitlines()[0] == store_notice, r2
        assert "passband" not in r2.err          # (no store: no passband store, and nothing said of one)
        assert r3.rc == 0 and r3.err.splitlines()[0] == pb_notice, r3
        assert not (d / "s1").exists()           # (refused before the store)
        # free after the holder's clean stop: the other takes all three
        again = _run([*other, *tiny_np, "--store", store, "--state", st], d / "o4")
        assert again.rc == 0 and "another listener" not in again.err, again
        assert again.err.splitlines()[0] == f"tiles in {st}", again

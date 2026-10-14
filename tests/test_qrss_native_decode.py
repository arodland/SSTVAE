"""`sstvae-qrss decode` (native/apps/sstvae_qrss/decode.cpp) against
qrss_decode.py, run side by side on multi-pass stores written by the
reference's Store and by the C++ store (sstvae_native.qrss.store, through
tests/qrss_native/_store.py's facade).

Each case copies the store (and any extra input) into two directories,
runs the script in one and the subcommand in the other with the same
arguments, and compares:

- the exit status;
- stdout, exactly: --list's and --key's accumulator summary (members,
  waveforms and frames, "?" for a pass file that is gone, merged foreign
  data, segments with no data), --pass's line, "wrote ...";
- stderr by how the script ended, as test_qrss_native_cli.py does: a
  SystemExit exactly, a usage error by its message (the text after
  "error: "), an uncaught exception by the traceback's last line (or its
  class, where only the class can agree);
- the directory afterwards, file by file (Store() makes passes/, acc/
  and prov/; decode writes nothing else but its picture);
- the picture, pixel for pixel (its bytes are stb's PNG encoder's, not
  Pillow's).

Without the codec half (apps/sstvae_qrss/codec/decode_image.cpp) the
subcommand refuses the picture where the script loads the codec, after
everything the script prints before it. The script is then pointed at a
model directory that does not exist ($QRSSTVAE_MODEL_DIR), so it ends at
that same step (resolve_onnx's refusal) without rendering anything, and
the two outputs agree up to it exactly. With the codec half both load the
v5 decoder from $QRSSTVAE_MODEL_DIR (qrss_helpers.model_dir) and every
case is compared in full.

--em and --retro are not ported: the subcommand refuses them, as an
argparse error, where the script would begin them (on --key, once the
picture is found); on --pass both say they are ignored. That refusal is
checked by itself (test_em_and_retro_are_refused).

The binary is $SSTVAE_QRSS_CLI, else `sstvae-qrss` beside the module's
build directory ($SSTVAE_NATIVE_MODULE_DIR/..); everything skips without
it. The Python runs start in parallel: each costs a second of imports.
"""

import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest

from qrss_fakes import make_header, make_pass_result, synthetic_picture
from qrss_helpers import Q_TEST, REPO_ROOT, codec_skip, model_dir
from qrss_native import _store as H
from qrss_native import require
from sstvae.qrss import frame, lframe, picture
from sstvae.qrss.constants import WAVEFORM_L
from sstvae.qrss.store import canonical_index

EM_RETRO = "--em and --retro are not in this build yet; use qrss_decode.py"
NO_MODELS = ": expected a .pt checkpoint, a .onnx artifact, or a directory containing the exported .onnx files\n"


def _no_renderer(output: str) -> str:
    return (f"{output}: this sstvae-qrss renders no pictures (the codec half of decode is not compiled in: it "
            "needs SSTVAE_BUILD_CODEC and apps/sstvae_qrss/codec/decode_image.cpp); use qrss_decode.py\n")


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
    if not any(line.split()[:1] == ["decode"] for line in r.stdout.splitlines()):
        pytest.skip("this sstvae-qrss has no decode subcommand")
    return p


def _renders(cli: Path, tmp: Path) -> bool:
    """Whether the subcommand has the codec half, asked of a pass: without
    it the picture is refused once the pass is read, with it the model
    directory is."""
    p = make_pass_result(np.zeros(frame.SHORT.n_data), 1.0, seed=3,
                         hdr=make_header(synthetic_picture(3)), decoded=True)
    d = tmp / "renders"
    d.mkdir()
    p.save(d / "p.npz")
    env = dict(os.environ, QRSSTVAE_MODEL_DIR=str(d / "no-models"))
    r = subprocess.run([cli, "decode", "out.png", "--pass", "p.npz"], cwd=d, capture_output=True, text=True,
                       timeout=60, env=env)
    assert r.returncode == 1, r
    if r.stderr == _no_renderer("out.png"):
        return False
    assert r.stderr == f"--model {d / 'no-models'}{NO_MODELS}", r.stderr
    return True


# --- the stores -----------------------------------------------------------------------------

CLOCK = 1.76e9
SHORT = frame.SHORT.n_data
L_SHORT = lframe.L_SHORT.n_data


def _build(Store, root: Path) -> dict:
    """The store every case reads, written through `Store` (the reference's
    or the C++). Five pictures and a pass in no picture:

    - A, "K1ABC/P", mode C: a short CE member per segment (the last one an
      EM round's, Z_ref 12.5) and an l-short L member in segment 1;
    - F, "FULL", mode A: one full CE pass, and a foreign (S, W) merged in;
    - B, "W1AW", mode B: two members in segment 0 -- the second one's pass
      file deleted after it was attached -- and nothing in segment 1;
    - X, "N0CALL", mode A, codec 0007 (not the decoder's: a codec mismatch);
    - a short pass with no decoded header, stored but in no accumulator.
    """
    st = Store(root, clock=lambda: CLOCK)
    out = {}

    spA = synthetic_picture(41, mode=2)
    hA = [make_header(spA, segment=g, callsign="K1ABC/P") for g in range(3)]
    A = [make_pass_result(picture.air_values(spA.segs[g], g)[:SHORT], 0.5 + g, seed=1 + g, hdr=hA[g],
                          decoded=True, q=Q_TEST + 2 * g, z_ref=12.5 if g == 2 else 10.0, em_round=int(g == 2))
         for g in range(3)]
    AL = make_pass_result(picture.air_values(spA.segs[1], 1)[:L_SHORT], 1.3, seed=7, hdr=hA[1], decoded=True,
                          q=Q_TEST + 9, f_hz=1250.0, waveform=WAVEFORM_L)
    for p in A + [AL]:
        st.add_pass(p)
    keyA = ("K1ABC/P", spA.picture_id)
    st.attach(keyA, A[0].uid, 0, "header", mode=2)
    st.attach(keyA, A[1].uid, 1, "header")
    st.attach(keyA, AL.uid, 1, "header")
    st.attach(keyA, A[2].uid, 2, "correlation")
    out.update(pidA=spA.picture_id, short=A[0].uid, em_member=A[2].uid, lpass=AL.uid)

    spF = synthetic_picture(46)
    hF = make_header(spF, callsign="FULL")
    F0 = make_pass_result(picture.air_values(spF.segs[0], 0), 0.8, seed=15, q=Q_TEST + 60, f_hz=987.6, hdr=hF,
                          decoded=True)
    st.add_pass(F0)
    keyF = ("FULL", spF.picture_id)
    st.attach(keyF, F0.uid, 0, "header", mode=0)
    F1 = make_pass_result(picture.air_values(spF.segs[0], 0), 0.6, seed=16, q=Q_TEST + 61, hdr=hF, decoded=True)
    S = np.zeros((picture.N_GROUPS, picture.GROUP_LATENTS))
    W = np.zeros_like(S)
    idx = canonical_index(0, F1.z.size)
    S[0, idx] = F1.w.astype(np.float64) * F1.z.astype(np.float64)
    W[0, idx] = F1.w
    assert st.merge(keyF, S, W, source="remote")
    out.update(pidF=spF.picture_id, full=F0.uid)

    spB = synthetic_picture(47, mode=1)
    hB = make_header(spB, callsign="W1AW")
    B = [make_pass_result(picture.air_values(spB.segs[0], 0)[:SHORT], s, seed=21 + k, hdr=hB, decoded=True,
                          q=Q_TEST + 20 + 2 * k) for k, s in enumerate((1.0, 2.0))]
    for p in B:
        st.add_pass(p)
        st.attach(("W1AW", spB.picture_id), p.uid, 0, "header")
    (Path(root) / "passes" / f"{B[1].uid}.npz").unlink()
    out.update(pidB=spB.picture_id)

    spX = synthetic_picture(48, mode=0, codec_id=7)
    X = make_pass_result(picture.air_values(spX.segs[0], 0)[:SHORT], 1.5, seed=31, q=Q_TEST + 30,
                         hdr=make_header(spX, callsign="N0CALL"), decoded=True)
    st.add_pass(X)
    st.attach(("N0CALL", spX.picture_id), X.uid, 0, "header")
    out.update(pidX=spX.picture_id)

    U = make_pass_result(picture.air_values(spA.segs[0], 0)[:SHORT], 1.0, seed=41, q=Q_TEST + 40)
    st.add_pass(U)
    out.update(nohdr=U.uid)
    return out


def _tree(d: Path) -> list[str]:
    return sorted(str(p.relative_to(d)) + ("/" if p.is_dir() else "") for p in d.rglob("*"))


@pytest.fixture(scope="module")
def stores(native, tmp_path_factory):
    """{"py": (root, info), "cpp": (root, info)}: one store per writer, and
    the edited copies (_EDITS) of the reference's."""
    require(native, "store")
    base = tmp_path_factory.mktemp("decode-stores")
    from sstvae.qrss.store import Store as PyStore

    out = {}
    for name, cls in (("py", PyStore), ("cpp", H.store_class(native.qrss))):
        root = base / name
        out[name] = (root, _build(cls, root))
    # the two writers wrote the same files (the bytes are the store tests')
    assert _tree(out["py"][0]) == _tree(out["cpp"][0])
    assert out["py"][1] == out["cpp"][1]
    for name, edit in _EDITS.items():
        root = base / f"edit-{name}"
        shutil.copytree(out["py"][0], root)
        info = out["py"][1]
        j = root / "acc" / f"K1ABC_P_{info['pidA']:08x}.json"
        meta = json.loads(j.read_text())
        edit(meta)
        j.write_text(json.dumps(meta, indent=1))
        out[f"edit-{name}"] = (root, info)
    return out


def _member(field_name, value):
    def edit(meta):
        if value is _DELETE:
            del meta["members"][0][field_name]
        else:
            meta["members"][0][field_name] = value
    return edit


_DELETE = object()

# Accumulator indexes no receiver writes, which the script reads as JSON
# gives them: an f-string of whatever type is there, its formats' refusals,
# MODE_NAMES[mode] (negative modes index from the end), member_wave on a
# uid that is no string ("?": the store refuses it as a bad uid).
_EDITS = {
    "mode5": lambda m: m.update(mode=5),
    "mode-1": lambda m: m.update(mode=-1),
    "mode-4": lambda m: m.update(mode=-4),
    "uid-int": _member("uid", 5),
    "uid-null": _member("uid", None),
    "uid-list": _member("uid", [1]),
    "uid-dict": _member("uid", {"a": 1}),
    "uid-gone": _member("uid", "1_2_3"),
    "uid-slash": _member("uid", "a/b"),
    "uid-dot": _member("uid", ".hidden"),
    "zref-int": _member("z_ref", 7),
    "zref-bool": _member("z_ref", True),
    "zref-str": _member("z_ref", "x"),
    "zref-null": _member("z_ref", None),
    "zref-list": _member("z_ref", [1.0]),
    "zref-dict": _member("z_ref", {}),
    "zref-gone": _member("z_ref", _DELETE),
    "meanw-str": _member("mean_w_db", "1"),
    "meanw-big": _member("mean_w_db", 1e300),
    "segment-str": _member("segment", "s"),
    "segment-gone": _member("segment", _DELETE),
    "method-num": _member("method", 3),
    "emround-float": _member("em_round", 1.5),
    "emround-gone": _member("em_round", _DELETE),
}


# --- running a case -------------------------------------------------------------------------

@dataclass
class Case:
    name: str
    args: list
    store: str = "py"                # which store is "st": "py", "cpp" or "edit-<name>"
    files: dict[str, bytes] = field(default_factory=dict)   # extra files in the run dir
    dirs: list[str] = field(default_factory=list)          # extra directories
    output: str | None = "out.png"   # the picture's name as given (None: no picture asked for)
    only_class: bool = False


@dataclass
class Run:
    rc: int
    out: str
    err: str


def _run(cmd: list, cwd: Path, env: dict) -> Run:
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, timeout=180, env=env)
    dec = (lambda b: b.decode("utf-8", "backslashreplace"))
    return Run(r.returncode, dec(r.stdout), dec(r.stderr))


def _stage(case: Case, root: Path, stores: dict) -> tuple[Path, Path]:
    dirs = []
    for side in ("py", "nat"):
        d = root / case.name / side
        d.mkdir(parents=True)
        shutil.copytree(stores[case.store][0], d / "st")
        for name, data in case.files.items():
            (d / name).parent.mkdir(parents=True, exist_ok=True)
            (d / name).write_bytes(data)
        for sub in case.dirs:
            (d / sub).mkdir(parents=True, exist_ok=True)
        dirs.append(d)
    return dirs[0], dirs[1]


def run_cases(cases: list[Case], root: Path, cli: Path, stores: dict, models: str) -> dict:
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env["COLUMNS"] = "200"
    env["QRSSTVAE_MODEL_DIR"] = models
    names = [c.name for c in cases]
    assert len(set(names)) == len(names)
    jobs = {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        for c in cases:
            py_dir, nat_dir = _stage(c, root, stores)
            jobs[c.name] = (
                pool.submit(_run, [sys.executable, str(REPO_ROOT / "qrss_decode.py"), *c.args], py_dir, env),
                pool.submit(_run, [str(cli), "decode", *c.args], nat_dir, env),
                py_dir, nat_dir,
            )
        return {name: (a.result(), b.result(), pd, nd) for name, (a, b, pd, nd) in jobs.items()}


def _last_line(text: str) -> str:
    lines = [ln for ln in text.splitlines() if ln.strip()]
    return lines[-1] if lines else ""


def _pixels(path: Path) -> np.ndarray:
    from PIL import Image

    with Image.open(path) as im:
        assert im.format == "PNG" and im.mode == "RGB", (path, im.format, im.mode)
        return np.asarray(im)


def check(case: Case, py: Run, nat: Run, py_dir: Path, nat_dir: Path, renders: bool):
    assert "Traceback" not in nat.err
    assert _tree(nat_dir) == _tree(py_dir)
    if not renders and py.rc == 1 and (py.err.endswith(NO_MODELS) or py.err.startswith("no *-decoder-")):
        # the script reached render.load (resolve_onnx refused the model
        # directory); the subcommand refuses the picture there
        assert nat == Run(1, py.out, _no_renderer(case.output)), (py, nat)
        return
    assert nat.rc == py.rc, (py, nat)
    assert nat.out == py.out, (py, nat)
    if py.rc == 0:
        assert nat.err == py.err
    elif py.rc == 2:
        prog = "qrss_decode.py: error: "
        assert py.err.startswith("usage: ") and _last_line(py.err).startswith(prog), py.err
        assert nat.err.startswith("usage: sstvae-qrss decode"), nat.err
        assert _last_line(nat.err) == "sstvae-qrss decode: error: " + _last_line(py.err)[len(prog):]
    elif "Traceback (most recent call last)" in py.err:
        want, got = _last_line(py.err), _last_line(nat.err)
        if case.only_class:
            assert got.split(":")[0] == want.split(":")[0], (want, got)
        else:
            assert got == want
        assert nat.err == got + "\n"
    else:
        assert nat.err == py.err
    if case.output is not None and (py_dir / case.output).is_file():
        assert np.array_equal(_pixels(nat_dir / case.output), _pixels(py_dir / case.output)), case.name


def _key(call: str, pid: int) -> str:
    return f"{call}:{pid:08x}"


def _cases(info: dict, store: str) -> list[Case]:
    """The cases that read the store, on the store `store` wrote."""
    def c(name, *args, **kw):
        return Case(f"{store}-{name}", list(args), store=store, **kw)

    kA, kF, kB, kX = (_key("K1ABC/P", info["pidA"]), _key("FULL", info["pidF"]), _key("W1AW", info["pidB"]),
                      _key("N0CALL", info["pidX"]))
    passes = "st/passes/"
    return [
        c("list", "--list", "--store", "st", output=None),
        c("list-out", "out.png", "--list", "--store", "st"),          # output is ignored with --list
        c("list-em", "--list", "--store", "st", "--em", "3", "--retro", output=None),
        c("key-a", "out.png", "--key", kA.lower(), "--store", "st"),
        c("key-a-0x", "out.png", "--key", f"K1ABC/P:0x{info['pidA']:X}", "--store", "st", "--precision", "fp16"),
        c("key-a-em-0", "out.png", "--key", kA, "--store", "st", "--em", "0"),
        c("key-a-em-neg", "out.png", "--key", kA, "--store", "st", "--em", "-1", "--old-edge-noise"),
        c("key-full", "out.png", "--key", kF, "--store", "st"),
        c("key-b", "out.png", "--key", kB, "--store", "st"),
        c("key-x", "out.png", "--key", kX, "--store", "st"),
        c("key-x-any", "out.png", "--key", kX, "--store", "st", "--any-codec"),
        c("pass-short", "out.png", "--pass", passes + info["short"] + ".npz"),
        c("pass-em-member", "out.png", "--pass", passes + info["em_member"] + ".npz", "--em", "3"),
        c("pass-l", "out.png", "--pass", passes + info["lpass"] + ".npz", "--retro"),
        c("pass-full", "out.png", "--pass", passes + info["full"] + ".npz", "--precision", "fp16"),
        c("pass-no-header", "out.png", "--pass", passes + info["nohdr"] + ".npz"),
        c("key-missing", "out.png", "--key", "K1ABC/P:1", "--store", "st", "--em", "2", "--retro"),
        # the render step's own refusals: the model, the output's directory
        c("pass-no-model", "out.png", "--pass", passes + info["short"] + ".npz", "--model", "nomodels"),
        c("key-no-model", "out.png", "--key", kF, "--store", "st", "--model", "st/acc"),
        c("key-no-dir", "nodir/out.png", "--key", kF, "--store", "st", "--precision", "fp16",
          output="nodir/out.png"),
        c("pass-into-dir", "d.png", "--pass", passes + info["short"] + ".npz", "--precision", "fp16",
          dirs=["d.png"], output="d.png"),
    ]


def _shared_cases() -> list[Case]:
    """Cases that end before the store matters (or read the reference's)."""
    def c(name, *args, **kw):
        return Case(name, list(args), **kw)

    npz_zip = b"PK\x03\x04" + bytes(range(256)) * 4
    return [
        # parse_key
        c("key-no-colon", "out.png", "--key", "K1ABC", "--store", "st"),
        c("key-empty-call", "out.png", "--key", ":12", "--store", "st"),
        c("key-not-hex", "out.png", "--key", "K1:zz", "--store", "st"),
        c("key-empty-id", "out.png", "--key", "K1:", "--store", "st"),
        c("key-quote", "out.png", "--key", "K'1\"x", "--store", "st"),
        c("key-quote-id", "out.png", "--key", "K1:'", "--store", "st"),
        c("key-spaces", "out.png", "--key", "k1: 1f ", "--store", "st"),
        c("key-underscore", "out.png", "--key", "K1:1_f", "--store", "st"),
        c("key-0x-underscore", "out.png", "--key", "K1:0x_1_f", "--store", "st"),
        c("key-double-underscore", "out.png", "--key", "K1:1__f", "--store", "st"),
        c("key-negative", "out.png", "--key", "K1:-1", "--store", "st"),
        c("key-plus", "out.png", "--key", "K1:+abc", "--store", "st"),
        c("key-huge", "out.png", "--key", "K1:" + "f" * 40, "--store", "st"),
        c("key-huge-neg", "out.png", "--key", "K1:-" + "f" * 40, "--store", "st"),
        c("key-colons", "out.png", "--key", "a:b:c", "--store", "st"),
        c("key-sharp-s", "out.png", "--key", "straße:1", "--store", "st"),
        c("key-ligature", "out.png", "--key", "ﬃ:1", "--store", "st"),
        c("key-greek", "out.png", "--key", "ΐᾀ:1", "--store", "st"),
        c("key-fullwidth-id", "out.png", "--key", "K1:１ａ", "--store", "st"),
        c("key-arabic-id", "out.png", "--key", "K1:١٢", "--store", "st"),
        c("key-nbsp-id", "out.png", "--key", "K1: 1 ", "--store", "st"),
        c("key-escaped-byte", "out.png", "--key", b"K\xff1:1", "--store", "st"),
        c("key-escaped-id", "out.png", "--key", b"K1:1\xff", "--store", "st"),
        c("key-every-upper", "out.png", "--key", "".join(_UPPERS) + ":1", "--store", "st"),
        c("key-slash", "out.png", "--key", "a/b/c:1", "--store", "st"),
        # --pass
        c("pass-missing", "out.png", "--pass", "nope.npz"),
        c("pass-dir", "out.png", "--pass", "st"),
        c("pass-dir-em", "out.png", "--pass", "st", "--em", "1"),
        c("pass-empty", "out.png", "--pass", "e.npz", files={"e.npz": b""}),
        c("pass-text", "out.png", "--pass", "t.npz", files={"t.npz": b"not a pass\n"}),
        c("pass-npy", "out.png", "--pass", "a.npy", files={"a.npy": _npy_bytes()}),
        c("pass-badzip", "out.png", "--pass", "z.npz", files={"z.npz": npz_zip}, only_class=True),
        c("pass-no-meta", "out.png", "--pass", "m.npz", files={"m.npz": _npz_bytes(x=np.arange(3))}),
        # the store
        c("list-new-store", "--list", "--store", "new/st", output=None),
        c("list-store-is-file", "--list", "--store", "f", files={"f": b"x"}, output=None),
        c("key-store-is-file", "out.png", "--key", "A:1", "--store", "f", files={"f": b"x"}),
        c("key-store-under-file", "out.png", "--key", "A:1", "--store", "f/st", files={"f": b"x"}),
        # argparse
        c("no-args", output=None),
        c("output-only", "out.png"),
        c("key-only", "--key", "A:1"),
        c("exclusive", "out.png", "--key", "A:1", "--pass", "p.npz"),
        c("exclusive-list", "out.png", "--list", "--key", "A:1"),
        c("two-outputs", "a.png", "b.png", "--list"),
        c("output-after-dashes", "--key", "A:1", "--", "--list", "--store", "st"),
        c("dashes-then-two", "--list", "--", "x", "y"),
        c("dashes-only", "--key", "A:1", "--"),
        c("dashes-twice", "--key", "A:1", "--", "--", "x"),
        c("abbreviated", "--pa", "x.npz", "out.png"),
        c("ambiguous", "--p", "x", "out.png"),
        c("bad-precision", "out.png", "--key", "A:1", "--precision", "fp64"),
        c("bad-em", "out.png", "--key", "A:1", "--em", "x"),
        c("em-unicode", "out.png", "--key", "K1:1", "--em", "２", "--store", "st"),
        c("missing-value", "out.png", "--key"),
        c("unknown", "out.png", "--key", "A:1", "--frobnicate"),
        c("eq-form", "--list", "--store=st", output=None),
    ]


def _npy_bytes() -> bytes:
    import io

    buf = io.BytesIO()
    np.save(buf, np.arange(4))
    return buf.getvalue()


def _npz_bytes(**arrays) -> bytes:
    import io

    buf = io.BytesIO()
    np.savez(buf, **arrays)
    return buf.getvalue()


def _uppers() -> list[str]:
    """Every code point str.upper() changes, in order (the key is one long
    name: ENAMETOOLONG from the stat, which pathlib passes on)."""
    return [chr(c) for c in range(0x110000) if not 0xD800 <= c < 0xE000 and chr(c).upper() != chr(c)]


_UPPERS = _uppers()


@pytest.fixture(scope="module")
def renders(cli, tmp_path_factory):
    return _renders(cli, tmp_path_factory.mktemp("decode-renders"))


@pytest.fixture(scope="module")
def models(renders, tmp_path_factory):
    """The model directory the runs see: the v5 decoder with the codec
    half, else one that does not exist (the script then stops at
    render.load, where the subcommand refuses the picture)."""
    if not renders:
        return str(tmp_path_factory.mktemp("decode-models") / "no-models")
    d = model_dir()
    if d is None or not list(Path(d).glob("*-decoder-fp32.onnx")) or not list(Path(d).glob("*-decoder-fp16.onnx")):
        codec_skip("no model directory with the v5 decoder at fp32 and fp16")
    return d


def _run_and_check(cases, tmp_path, cli, stores, models, renders):
    results = run_cases(cases, tmp_path, cli, stores, models)
    for case in cases:
        py, nat, py_dir, nat_dir = results[case.name]
        check(case, py, nat, py_dir, nat_dir, renders)
    return results


@pytest.mark.parametrize("writer", ["py", "cpp"])
def test_decode_side_by_side(cli, stores, models, renders, tmp_path, writer):
    """Every case that reads the store, on the store each implementation
    wrote; the two writers' stores decode alike."""
    cases = _cases(stores[writer][1], writer)
    results = _run_and_check(cases, tmp_path, cli, stores, models, renders)
    out = {c.name.split("-", 1)[1]: results[c.name][0].out for c in cases}
    # what the cases are for: every line kind appears
    assert "  segment 1: no data\n" in out["key-b"] and "  ?  segment 0" in out["key-b"], out["key-b"]
    assert "  L l-short  segment 1  by header" in out["key-a"], out["key-a"]
    assert "Z_ref 12.5" in out["key-a"] and "EM round 1" in out["key-a"] and "by correlation" in out["key-a"]
    assert "1 passes, 1 merged" in out["key-full"] and "effective SNR" in out["key-full"], out["key-full"]
    assert out["list"].count("picture ") == 4, out["list"]
    assert out["pass-l"].startswith("--em and --retro need an accumulator (--key); ignored for a single pass\n"
                                    f"pass {stores[writer][1]['lpass']}: L l-short frame, mean W ")
    if renders:
        assert out["key-a"].endswith("wrote out.png\n") and out["key-x-any"].endswith("wrote out.png\n")
        assert results[f"{writer}-key-x"][0].rc == 1
        err = {name: results[f"{writer}-{name}"][0].err for name in ("key-no-dir", "pass-into-dir", "pass-no-model")}
        assert err["key-no-dir"].endswith("FileNotFoundError: [Errno 2] No such file or directory: 'nodir/out.png'\n")
        assert err["pass-into-dir"].endswith("IsADirectoryError: [Errno 21] Is a directory: 'd.png'\n")
        assert err["pass-no-model"] == "--model nomodels" + NO_MODELS


def test_decode_refusals(cli, stores, models, renders, tmp_path):
    """Keys, pass files, stores and arguments the script refuses, and the
    tracebacks it ends with; the stores' own as the reference wrote them."""
    cases = _shared_cases()
    results = _run_and_check(cases, tmp_path, cli, stores, models, renders)
    errs = {c.name: results[c.name][0].err for c in cases}
    assert errs["key-every-upper"].rstrip().endswith(".json'"), errs["key-every-upper"][-300:]
    assert "File name too long" in errs["key-every-upper"]
    assert errs["key-sharp-s"] == "no picture STRASSE 00000001 in st\n", errs["key-sharp-s"]
    assert errs["key-negative"] == "no picture K1 -0000001 in st\n"


def test_decode_edited_accumulators(cli, stores, models, renders, tmp_path):
    """Accumulator indexes holding what no receiver writes: --list and --key
    end alike (a line of mixed types, or the same exception)."""
    info = stores["py"][1]
    kA = _key("K1ABC/P", info["pidA"])
    cases = []
    for name in _EDITS:
        cases.append(Case(f"{name}-list", ["--list", "--store", "st"], store=f"edit-{name}", output=None))
    # --key prints the same acc_lines, then renders: a few, for the modes
    # the codec is handed and the refusals that end before it
    for name in ("mode-1", "mode-4", "uid-gone", "zref-str", "zref-gone"):
        cases.append(Case(f"{name}-key", ["out.png", "--key", kA, "--store", "st", "--precision", "fp16"],
                          store=f"edit-{name}"))
    results = _run_and_check(cases, tmp_path, cli, stores, models, renders)
    assert results["mode5-list"][0].err.endswith("IndexError: tuple index out of range\n")
    for name, shown in (("uid-int", "5"), ("uid-null", "None"), ("uid-list", "[1]"), ("uid-dict", "{'a': 1}")):
        py = results[f"{name}-list"][0]
        assert py.rc == 0 and f"\n    {shown}  ?  segment " in py.out, (name, py)
    assert results["zref-str-list"][0].err.endswith(
        "ValueError: Unknown format code 'f' for object of type 'str'\n")


def test_em_and_retro_are_refused(cli, stores, tmp_path):
    """--em ROUNDS > 0 or --retro on a picture that exists: an argparse
    error naming qrss_decode.py, before the accumulator is read (the
    script would print the EM or retro lines first); --em 0 or negative
    decodes as without. On --pass both are ignored, as the script says."""
    root, info = stores["cpp"]
    kA = _key("K1ABC/P", info["pidA"])
    env = dict(os.environ, COLUMNS="200")
    for extra in (["--em", "1"], ["--em", "+2"], ["--retro"], ["--em", "3", "--retro"],
                  ["--retro", "--passband", "nope", "--frame", "l-short"], ["--em", "1_0"]):
        r = _run([cli, "decode", "out.png", "--key", kA, "--store", str(root), *extra], tmp_path, env)
        assert r.rc == 2 and r.out == "", (extra, r)
        assert r.err.startswith("usage: sstvae-qrss decode [-h] [--key KEY | --pass PASS_FILE | --list] "), r.err
        assert r.err.endswith(f"\nsstvae-qrss decode: error: {EM_RETRO}\n"), r.err
    assert not (tmp_path / "out.png").exists()


def test_png_only(cli, stores, models, renders, tmp_path):
    """The native image layer writes PNG: another extension is refused
    where the codec would be loaded (after the summary), where Pillow would
    have written that format."""
    if not renders:
        pytest.skip("this sstvae-qrss renders no pictures")
    root, info = stores["cpp"]
    kF = _key("FULL", info["pidF"])
    env = dict(os.environ, COLUMNS="200", QRSSTVAE_MODEL_DIR=models)
    for name in ("out.jpg", "out.PNG.bmp", "out", ".png", "dir.png/x", "out.webp"):
        r = _run([cli, "decode", name, "--key", kF, "--store", str(root)], tmp_path, env)
        assert r.rc == 1 and r.out.startswith("picture FULL "), (name, r)
        assert r.err == (f"{name}: sstvae-qrss decode writes PNG only (a .png name); qrss_decode.py writes "
                         "whatever format Pillow names by the extension\n"), (name, r.err)
    for name in ("out.Png", "out.apng"):
        r = _run([cli, "decode", name, "--key", kF, "--store", str(root), "--precision", "fp16"], tmp_path, env)
        assert r.rc == 0 and r.out.endswith(f"wrote {name}\n"), (name, r)
        assert _pixels(tmp_path / name).shape == (480, 640, 3)


def test_help(cli):
    """The usage is the script's, word for word (argparse wraps it at the
    terminal's width, the native at its own), and every flag is listed."""
    env = dict(os.environ, COLUMNS="200")
    env.pop("PYTHONPATH", None)
    py = _run([sys.executable, str(REPO_ROOT / "qrss_decode.py"), "--help"], REPO_ROOT, env)
    nat = _run([str(cli), "decode", "--help"], REPO_ROOT, env)
    assert py.rc == nat.rc == 0 and nat.err == ""

    def usage(text):
        return " ".join(text.split("\n\n", 1)[0].split())

    assert usage(nat.out) == usage(py.out).replace("usage: qrss_decode.py ", "usage: sstvae-qrss decode ")
    for flag in ("--key KEY", "--pass PASS_FILE", "--list", "--store STORE", "--em ROUNDS", "--retro",
                 "--passband PASSBAND", "--frame FRAME", "--any-codec", "--model MODEL",
                 "--precision {fp32,fp16,int8}", "--old-edge-noise", "output"):
        assert f"  {flag}" in nat.out, flag

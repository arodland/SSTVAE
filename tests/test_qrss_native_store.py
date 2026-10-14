"""Parity of native/core/qrss/store (sstvae_native.qrss.store) with
sstvae/qrss/store.py. tests/qrss_native/store.py substitutes it under
--native-qrss=rx, its Store row being tests/qrss_native/_store.py's facade
(the native class with store.py's os.replace and ZW_CACHE as seams and
load_pass returning the reference's PassResult).

The contract (the bar), and in brackets what was measured (2026-10-14):

- **the files**: a store the C++ wrote is the reference's byte for byte --
  every pass .npz, accumulator .npz and .json, provisional cluster and
  log.jsonl line -- save one field: a member's mean_w_db, 10 log10 of its
  pass's mean weight, which the reference takes from numpy's log10 and the
  C++ from the C library's. On this machine numpy's own (SIMD) log10
  differs from glibc's in the last place for a few percent of arguments;
  each such difference is checked to be exactly that, the file otherwise
  the same text (`_store.compare_trees`) [the scripted sequence: two
  members' mean_w_db in one accumulator, 2.05436353280837 against
  2.0543635328083694 and 2.0382447587105825 against 2.038244758710582;
  every other byte equal];
- **every return value** of a scripted sequence of 142 operations (each
  method of Store and Accumulator, twins CE and L, header modes, merges
  accepted, duplicated, replaced, anonymous, uncorrelated, without overlap,
  non-finite, negative and misshapen, provisional clusters including a
  new one whose name c_<uid> is taken (c_<uid>_2), logs of numpy scalars
  and odd keys, due with an accumulator whose later member's pass is gone,
  expiry): the same values of the same types (np.float64 where the
  reference returns one), the same arrays bit for bit, the same dicts in
  the same key order, the same errors with the same text (paths aside),
  mean_w_db to 12 significant digits (the line above) [all equal];
- **interop both ways**: a store begun by either implementation and
  continued by either -- reading the other's accumulators, rebuilding them
  (S and W bit for bit what the other wrote), merging over its foreign
  parts, attaching, updating, expiring -- leaves the same files and returns
  the same values;
- **the seams**: the clock (called for every stamp, its exception passed
  on with nothing written), the facade's os.replace (a failing replace
  leaves what the reference's leaves, temporary files removed) and
  ZW_CACHE (the same cache, in the same order, at any size, 0 included:
  the pass returned and nothing cached);
- **threads**: eight threads hammering one C++ Store (attach, detach,
  merge, the provisional clusters, due, reads) raise nothing, and every
  accumulator they leave is what its members and parts sum to (the
  reference's rebuild of it, bit for bit); two C++ Stores open on one
  directory (spelled two ways) in two threads share its lock -- no
  temporary file collides, no attach is lost;
- **expiry at the cutoff**: an accumulator, a cluster and an orphan pass
  exactly at it are kept and a ulp older dropped, as in the reference;
- **em.py's store use, whole**: em_prior, known_header, em_refine (a
  deterministic re-receive stub) and template_search, every row of this
  lane in place with counting spies installed before the rows: the same
  priors, history, log, result and files as on the reference's store,
  with Store and canonical_index reached.

The module functions (default_root, key_name, slot_time, mean_w_db,
canonical_index) and Accumulator's construction, repr and copies are
compared on their own. Without Python, native/tests/qrss/test_store.cpp
replays a session pinned by gen_store_pins.py (the SHA-256 of every file
at four checkpoints), the seams and three threads on one Store.
Fast tier ~15 s; nothing is slow.
"""

import copy
import dataclasses
import functools
import hashlib
import json
import math
import os
import random
import shutil
import threading
from pathlib import Path

import numpy as np
import pytest
from qrss_fakes import make_header, make_pass_result, synthetic_picture
from qrss_helpers import Q_TEST
from qrss_native import _store as H
from qrss_native import require

from sstvae.qrss import frame, picture
from sstvae.qrss import store as S
from sstvae.qrss.constants import WAVEFORM_L

N = frame.SHORT.n_data


@pytest.fixture(scope="module")
def ns(native):
    return require(native, "store")


@pytest.fixture(scope="module")
def NStore(native, ns):
    return H.store_class(native.qrss)


class Clock:
    def __init__(self, t=1.76e9):
        self.t = float(t)

    def __call__(self):
        return self.t


# --- normalising results ---------------------------------------------------------------------------

def _sha(a) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:20]


def _is_acc(x) -> bool:
    return hasattr(x, "rebuild") and hasattr(x, "S_ext")


def norm(x, root: str):
    """A comparable rendering of a result: types kept, arrays by dtype,
    shape and content, the store's root spelled <root>, mean_w_db to 12
    digits (the module docstring)."""
    if isinstance(x, BaseException):
        return ("raise", type(x).__name__, norm(str(x), root))
    if isinstance(x, np.ndarray):
        return ("nd", x.dtype.str, x.shape, _sha(x))
    if _is_acc(x):
        return ("Accumulator", norm(x.key, root), norm(x.mode, root), norm(x.codec_id, root),
                [norm(a, root) for a in (x.S, x.W, x.S_ext, x.W_ext)],
                norm(x.members, root), norm(x.foreign, root),
                norm(dict(x.parts), root), norm(x.created, root), norm(x.updated, root),
                norm(x.uids, root))
    if dataclasses.is_dataclass(x) and not isinstance(x, type):
        return (type(x).__name__, [(f.name, norm(getattr(x, f.name), root))
                                   for f in dataclasses.fields(x)])
    if isinstance(x, Path):
        return ("Path", norm(str(x), root))
    if isinstance(x, dict):
        return (type(x).__name__ if type(x) is not dict else "dict",
                [(norm(k, root), ("float12", f"{v:.12g}") if k == "mean_w_db" and isinstance(v, float)
                  else norm(v, root)) for k, v in x.items()])
    if isinstance(x, (list, tuple)):
        return (type(x).__name__, [norm(v, root) for v in x])
    if isinstance(x, str):
        return x.replace(root, "<root>")
    if isinstance(x, (bool, int, float, np.generic)) or x is None:
        return (type(x).__name__, repr(x))
    return ("?", type(x).__name__, repr(x))


class Rec:
    """Calls an operation and records its outcome, normalised."""

    def __init__(self, root):
        self.root, self.out = str(root), []

    def __call__(self, name, fn, *a, **k):
        try:
            res = fn(*a, **k)
            self.out.append((name, "ok", norm(res, self.root)))
            return res
        except Exception as e:  # noqa: BLE001 - the reference's own type and text
            self.out.append((name, "err", norm(e, self.root)))
            return None


def assert_same(a, b):
    assert len(a) == len(b)
    for x, y in zip(a, b):
        assert x == y, f"{x[0]}:\n  reference {x[1:]}\n  C++       {y[1:]}"


# --- the passes --------------------------------------------------------------------------------------

def _air(sp, g):
    return picture.air_values(sp.segs[g], g)[:N]


@functools.lru_cache(maxsize=1)
def _passes():
    spA, spB = synthetic_picture(41, mode=2), synthetic_picture(42)
    spC, spD, spE = synthetic_picture(43, mode=1), synthetic_picture(44), synthetic_picture(45, mode=2)
    spF = synthetic_picture(46)
    hA = [make_header(spA, segment=g, callsign="K1ABC/P") for g in range(3)]
    P = {
        "A0": make_pass_result(_air(spA, 0), 0.5, seed=1, hdr=hA[0], decoded=True, q=Q_TEST),
        "A1": make_pass_result(_air(spA, 1), 1.5, seed=2, hdr=hA[1], decoded=True, q=Q_TEST + 2),
        "A2": make_pass_result(_air(spA, 2), 2.5, seed=3, hdr=hA[2], decoded=True, q=Q_TEST + 4),
        "A0s": make_pass_result(_air(spA, 0), 2.0, seed=4, q=Q_TEST, f_hz=1502.0),
        "A0w": make_pass_result(_air(spA, 0), 0.1, seed=5, q=Q_TEST, f_hz=1498.0),
        # an L pass whose band centre (pilot - 9.5 Hz) is 3 Hz from A1's
        "A1L": make_pass_result(_air(spA, 1), 3.0, seed=6, q=Q_TEST + 2, f_hz=1512.5,
                                waveform=WAVEFORM_L),
        "B1": make_pass_result(_air(spB, 0), 1.0, seed=7, q=Q_TEST + 30, f_hz=1700.0),
        "B2": make_pass_result(_air(spB, 0), 1.0, seed=8, q=Q_TEST + 40, f_hz=1700.0),
        "C0": make_pass_result(_air(spC, 0), 1.0, seed=9, hdr=make_header(spC, callsign="G4XYZ"),
                               decoded=True, q=Q_TEST + 6, f_hz=1100.25),
        "E0": make_pass_result(_air(spE, 0), 1.0, seed=13, q=Q_TEST + 20, f_hz=1234.5),
        "E1": make_pass_result(_air(spE, 1), 1.0, seed=14, q=Q_TEST + 22, f_hz=1234.5),
        # a FULL pass: the median over a group's 50,600 sent latents is not 0
        "F0": make_pass_result(picture.air_values(spF.segs[0], 0), 0.8, seed=15, q=Q_TEST + 60,
                               f_hz=987.6),
    }
    for i in range(3):
        P[f"D{i + 1}"] = make_pass_result(_air(spD, 0), 0.075, seed=10 + i, q=Q_TEST + 50 + i)
    for i in range(8):                     # weights whose means are not round numbers
        w = np.random.default_rng(84 + i).uniform(0.2, 3.0, N)
        P[f"R{i}"] = make_pass_result(_air(spF, 0), w=w, seed=80 + i, q=Q_TEST + 70 + i, f_hz=1800.0)
    keys = {"A": ("K1ABC/P", spA.picture_id), "B": ("W1AW", spB.picture_id),
            "C": ("G4XYZ", spC.picture_id), "D": ("DL1ABC", spD.picture_id),
            "E": ("VK2XX", spE.picture_id), "F": ("FULL/1", spF.picture_id),
            "R": ("R1ND", spF.picture_id ^ 1)}
    # foreign accumulators: other receivers' passes of A (and one of B)
    extra = {}
    for i, (sp, g) in enumerate([(spA, 0), (spA, 1), (spA, 2), (spA, 0), (spA, 1), (spB, 0)]):
        extra[i] = (make_pass_result(_air(sp, g), 1.0, seed=60 + i, q=Q_TEST + 7), g)
    return P, keys, extra


def _foreign(*passes):
    Sf = np.zeros((3, 52800))
    Wf = np.zeros((3, 52800))
    for p, g in passes:
        idx = picture.air_to_canonical(g)[:p.z.size]
        w = p.w.astype(np.float64)
        Sf[g, idx] += w * p.z
        Wf[g, idx] += w
    return Sf.astype(np.float32), Wf.astype(np.float32)


# --- the module functions ------------------------------------------------------------------------------

def _outcome(fn, *a):
    try:
        r = fn(*a)
    except Exception as e:  # noqa: BLE001
        return ("raise", type(e).__name__, str(e))
    if isinstance(r, np.ndarray):
        return ("nd", r.dtype.str, r.shape, r.flags.writeable, _sha(r))
    return (type(r).__name__, repr(r))


_FUNCS = {
    "key_name": [(("K1ABC/P", 5),), (("A", -10),), (("A", 2**40),), (("A", "12"),), (("A", 1.7),),
                 (("A", True),), ((1, 2),), (("A",),), (("A", 1, 2),), ("ab",), (("A", None),),
                 (("A", np.int64(7)),), (("A", np.float32(3.9)),), ([b"A", 1],), (("Ä/x", 1),),
                 (("A", "0x10"),), (("A", " 12 "),), (("A", "1_000"),), (5,)],
    "slot_time": [(5,), (-3,), (2**40,), (1.9,), ("7",), (np.int64(4),), (True,), (None,), (-1.5,)],
    "mean_w_db": [(np.ones(4),), ([],), (np.zeros(3),), ([1, 2, 3],), (np.float32([0.5, 0.25]),),
                  (-np.ones(2),), (2.0,), (np.array([np.inf]),), (np.array([np.nan]),),
                  ([[1, 2], [3, 4]],), (np.array([1e-300, 1e-300]),), ([True, False],), (0,), ("x",),
                  (np.full(7, 1e-3),)],
    "canonical_index": [(0, 1), (1, 50600), (2, 50601), (0, 0), (3, 5), (-1, 5), (2, 4096), (0.5, 3),
                        (1, 2.7), (1, 50600.5), ("1", 3), (1, "3"), (np.int64(1), np.int64(9)),
                        (True, 2), (-4, 4), (1, None), (5, 2.5)],
}


def test_module_functions(ns):
    """key_name, slot_time, mean_w_db and canonical_index on edge inputs:
    the same value of the same type, or the same error with the same text.
    Integers past int64 are the one deviation (OverflowError here)."""
    for name, cases in _FUNCS.items():
        for args in cases:
            ref = _outcome(getattr(S, name), *args)
            got = _outcome(getattr(ns, name), *args)
            assert got == ref, (name, args)
    ref = _outcome(S.key_name, iter(["Z/1", 3]))          # any iterable of two
    assert _outcome(ns.key_name, iter(["Z/1", 3])) == ref
    for name in ("key_name", "slot_time"):           # the deviation, pinned
        arg = ("A", 2**70) if name == "key_name" else 2**70
        with pytest.raises(OverflowError):
            getattr(ns, name)(arg)
    for k in ("SENT", "N_GROUPS", "GROUP_LATENTS", "DAY_S", "SLOT_S", "ZW_CACHE", "SAME_PASS_HZ"):
        assert getattr(ns, k) == getattr(S, k) and type(getattr(ns, k)) is type(getattr(S, k)), k


def test_default_root(ns, monkeypatch, tmp_path):
    """$QRSSTVAE_HOME, else ~/.local/share/qrsstvae, as pathlib.Path; an
    empty $QRSSTVAE_HOME is unset, and so is an empty $HOME (the password
    database's home, not expanduser's '/')."""
    for env, home in [("x/y", None), (str(tmp_path / "q"), None), ("", str(tmp_path)),
                      (None, str(tmp_path)), (None, ""), (None, "/"), ("~/z", "/h")]:
        if env is None:
            monkeypatch.delenv("QRSSTVAE_HOME", raising=False)
        else:
            monkeypatch.setenv("QRSSTVAE_HOME", env)
        monkeypatch.setenv("HOME", home if home is not None else str(tmp_path))
        a, b = S.default_root(), ns.default_root()
        assert type(b) is type(a) and b == a, (env, home)


def test_accumulator_construction_and_copies(ns):
    """Accumulator(...) and .empty: the reference's fields, types and repr;
    copies are independent; arrays are float32 (3, 52800) views a rebuild
    writes into (the reference's rebuild replaces its arrays instead)."""
    rng = np.random.default_rng(0)
    arrs = [rng.standard_normal((3, 52800)).astype(np.float32) for _ in range(4)]
    for cls in (S.Accumulator, ns.Accumulator):
        e = cls.empty(("K1ABC/P", np.int64(5)), np.int32(2), 7.0, 1.5)
        assert (e.key, e.mode, e.codec_id, e.created, e.updated) == (("K1ABC/P", 5), 2, 7, 1.5, 1.5)
        assert [type(v) for v in (e.key[1], e.mode, e.codec_id)] == [int, int, int]
        assert e.members == [] and e.foreign == [] and e.parts == {} and e.uids == []
        assert all(a.dtype == np.float32 and a.shape == (3, 52800) and not a.any()
                   for a in (e.S, e.W, e.S_ext, e.W_ext))
    a = S.Accumulator(("A", 1), 0, 3, *arrs, members=[{"uid": "u", "segment": 0}],
                      foreign=[{"sha": "s"}], parts={"s": (arrs[0], arrs[1])}, created=2.0, updated=3.0)
    b = ns.Accumulator(("A", 1), 0, 3, *arrs, members=[{"uid": "u", "segment": 0}],
                       foreign=[{"sha": "s"}], parts={"s": (arrs[0], arrs[1])}, created=2.0, updated=3.0)
    assert repr(b) == repr(a)
    assert norm(b, "/") == norm(a, "/")
    assert b.member("u") == a.member("u") and b.member("v") is None
    assert b.has_data(0) == a.has_data(0) and b.median_w(1) == a.median_w(1)
    assert b.seg_snr_db() == a.seg_snr_db() and b.meta() == a.meta()
    c = copy.copy(b)
    d = copy.deepcopy(b)
    c.S = np.zeros((3, 52800), np.float32)
    d.members = []
    assert b.S.any() and b.members and norm(d.S, "/") == norm(a.S, "/")
    b.S = arrs[2].ravel()                          # any 158,400 values
    assert np.array_equal(b.S, arrs[2])
    with pytest.raises(ValueError, match="cannot reshape array of size 5 into shape"):
        b.W = np.zeros(5)


# --- the scripted sequence ---------------------------------------------------------------------------

def _script(cls, root):
    """Some 110 operations on a fresh store of class `cls` (module docstring)."""
    P, K, X = _passes()
    clk = Clock()
    st = cls(root, clock=clk)
    r = Rec(root)

    def tick(s=1.25):
        clk.t += s

    r("root", lambda: st.root)
    r("paths", lambda: (st._pass_path("u1"), st._acc_paths(K["A"]), st._prov_path("c_x")))
    for bad in ("a/b", ".hidden", "", "x.y"):
        r("_pass_path", st._pass_path, bad)
    r("has_pass none", st.has_pass, P["A0"].uid)
    r("accumulator none", st.accumulator, K["A"])
    r("open_keys empty", st.open_keys)
    r("prov empty", st.prov_clusters)
    r("due empty", st.due, Q_TEST)
    for name in ("A0", "A1", "A2", "A0s", "A0w", "A1L", "B1", "B2", "C0", "D1", "D2", "D3", "E0", "E1",
                 "F0"):
        r("add_pass " + name, st.add_pass, P[name])
    r("pass_uids", st.pass_uids)
    r("has_pass", st.has_pass, P["A0"].uid)
    r("pass_info", st.pass_info, P["A1L"].uid)
    r("pass_info header", st.pass_info, P["C0"].uid)
    r("pass_info missing", st.pass_info, "nope")
    r("pass_zw", st.pass_zw, P["A0"].uid)
    r("load_pass", st.load_pass, P["C0"].uid)
    r("load_pass L", st.load_pass, P["A1L"].uid)
    r("_zw", lambda: list(st._zw))
    tick()
    r("attach A0", st.attach, K["A"], P["A0"].uid, 0, "header", mode=2)
    tick()
    r("attach A1", st.attach, K["A"], P["A1"].uid, 1, "header")
    r("attach A2", st.attach, K["A"], P["A2"].uid, np.int64(2), "header", codec_id=7)
    tick()
    r("twin stronger", st.attach, K["A"], P["A0s"].uid, 0, "corr")
    r("twin weaker", st.attach, K["A"], P["A0w"].uid, 0, "corr")
    r("twin L", st.attach, K["A"], P["A1L"].uid, 1, "corr")
    r("reattach", st.attach, K["A"], P["A2"].uid, 2, "soft-header")
    r("attach missing pass", st.attach, K["A"], "nope", 0, "x")
    r("attach bad segment", st.attach, K["A"], P["B1"].uid, 3, "x")
    r("attach headerless", st.attach, K["B"], P["B1"].uid, 0, "corr")
    r("attach header mode", st.attach, K["C"], P["C0"].uid, 0, "header")
    tick()
    r("attach B2", st.attach, K["B"], P["B2"].uid, 0, "corr")
    r("move B2", st.attach, K["A"], P["B2"].uid, 0, "corr")
    r("attach E0", st.attach, K["E"], P["E0"].uid, 0, "corr", mode=2)
    r("attach E1", st.attach, K["E"], P["E1"].uid, 1, "corr")
    r("_zw", lambda: list(st._zw))
    r("create existing", st.create, K["A"], 0, 0)
    r("create new", st.create, ("N0CALL", 0xFFFFFFFF), 1, 9)
    r("open_keys", st.open_keys)
    r("has_accumulator", lambda: [st.has_accumulator(k) for k in K.values()])
    r("find_member", st.find_member, P["A0s"].uid)
    r("find_member none", st.find_member, "nope")
    acc = r("accumulator A", st.accumulator, K["A"])
    r("uids", lambda: acc.uids)
    r("member", acc.member, P["A2"].uid)
    r("member none", acc.member, "x")
    r("loo", acc.loo, P["A2"].uid, st)
    r("loo none", acc.loo, "x", st)
    r("_sums", acc._sums, st)
    r("_sums skip", acc._sums, st, skip=P["A1L"].uid, ext=False)
    r("has_data", lambda: [acc.has_data(g) for g in range(3)])
    r("median_w", lambda: [acc.median_w(g) for g in range(3)])
    r("seg_snr_db", acc.seg_snr_db)
    r("meta", acc.meta)
    for i in range(8):
        st.add_pass(P[f"R{i}"])
        r(f"attach R{i}", st.attach, K["R"], P[f"R{i}"].uid, 0, "corr", mode=0)
    accF = r("attach FULL", st.attach, K["F"], P["F0"].uid, 0, "header", mode=0)
    r("median_w FULL", accF.median_w, 0)
    r("has_data FULL", lambda: [accF.has_data(g) for g in range(3)])
    # 10 log10 of the median: numpy's log10 against the C library's (module docstring)
    r("seg_snr_db FULL", lambda: [None if x is None else f"{x:.12g}" for x in accF.seg_snr_db()])
    r("twin", st.twin, acc, Q_TEST, 1504.9, 0)
    r("twin uid", st.twin, acc, Q_TEST, 1502.0, 0, P["A0s"].uid)
    r("twin L L", st.twin, acc, Q_TEST + 2, 1512.0, 1, None, WAVEFORM_L)
    r("twin CE L", st.twin, acc, Q_TEST + 2, 1503.0, 1, None, 0)
    r("twin CE L far", st.twin, acc, Q_TEST + 2, 1508.0, 1, None, 0)
    r("twin none", st.twin, acc, Q_TEST + 1, 1502.0, 0)
    r("due", lambda: [st.due(q) for q in range(Q_TEST - 2, Q_TEST + 30)])
    tick()
    new_b1 = dataclasses.replace(P["B1"], z=(P["B1"].z * 0.5).astype(np.float32),
                                 w=(P["B1"].w * 3).astype(np.float32), em_round=2)
    r("update_pass", st.update_pass, new_b1)
    r("update_pass loose", st.update_pass, P["D1"])
    r("_zw", lambda: list(st._zw))
    tick()
    F1 = _foreign(X[0], X[1])
    r("merge", st.merge, K["A"], *F1, "rx2")
    r("merge dup", st.merge, K["A"], *F1, "rx2")
    r("merge dup other source", st.merge, K["A"], *F1, "rx3")
    tick()
    r("merge replaces", st.merge, K["A"], *_foreign(X[2], X[3]), "rx2")
    r("merge anon", st.merge, K["A"], *_foreign(X[4]))
    r("merge anon 2", st.merge, K["A"], *_foreign(X[0], X[4]), "")
    F5 = _foreign(X[1], X[2])
    r("merge float64 flat", st.merge, K["A"], F5[0].astype(np.float64).ravel(),
      F5[1].astype(np.float64).ravel(), "rx4")
    r("merge uncorrelated", st.merge, K["A"], *_foreign(X[5]), "rx9")
    bad = F1[0].copy()
    bad[1, 7] = np.nan
    r("merge nan", st.merge, K["A"], bad, F1[1], "x")
    neg = F1[1].copy()
    neg[0, 3] = -1e-3
    r("merge negative", st.merge, K["A"], F1[0], neg, "x")
    r("merge wrong size", st.merge, K["A"], np.zeros(10), np.zeros(10))
    r("merge no acc", st.merge, ("ZZ", 1), *F1, "x")
    r("merge nan no acc", st.merge, ("ZZ", 1), bad, F1[1])
    Fg = _foreign(X[2])                    # group 2 only, where B has nothing
    r("merge no overlap", st.merge, K["B"], *Fg, "rx5")
    r("accumulator A merged", st.accumulator, K["A"])
    tick()
    r("detach", st.detach, K["A"], P["B2"].uid)
    r("detach non-member", st.detach, K["A"], "nope")
    r("detach no acc", st.detach, ("ZZ", 1), "nope")
    r("delete", st.delete_accumulator, ("N0CALL", 0xFFFFFFFF))
    r("delete missing", st.delete_accumulator, ("N0CALL", 0xFFFFFFFF))
    tick()
    c1 = r("prov_add D1", st.prov_add, P["D1"].uid)
    r("prov_add D2", st.prov_add, P["D2"].uid, c1)
    r("prov_add D2 again", st.prov_add, P["D2"].uid, c1)
    tick()
    c3 = r("prov_add D3", st.prov_add, P["D3"].uid)
    r("prov_clusters", st.prov_clusters)
    r("prov_cluster_of", st.prov_cluster_of, P["D2"].uid)
    r("prov_cluster_of none", st.prov_cluster_of, "nope")
    r("prov_add missing cluster", st.prov_add, P["D3"].uid, "c_missing")
    r("prov_remove", st.prov_remove, P["D2"].uid)
    r("prov_remove again", st.prov_remove, P["D2"].uid)
    tick()
    r("prov_add D2 to c3", st.prov_add, P["D2"].uid, c3)
    r("prov_add D3 anew", st.prov_add, P["D3"].uid)        # c_<D3> holds D2: c_<D3>_2
    r("prov after", st.prov_clusters)
    r("attach provisional", st.attach, K["D"], P["D1"].uid, 0, "corr", mode=0)
    r("prov end", st.prov_clusters)
    tick()
    r("log", st.log, {"event": "x", "i": np.int64(3), "f": np.float32(0.1), "b": np.bool_(True),
                      "d": np.float64(2.5), "l": [1, (2, 3.5), None, True], "s": "ünï \"q\"\n",
                      "n": float("nan"), "inf": -math.inf, "g": {"z": np.int32(-4), "e": []},
                      "u": np.uint8(200), "neg": -(2**63)})
    r("log override", st.log, {"time": "t", "x": 1})
    r("log keys", st.log, {5: 1, 2.5: 2, True: 3, None: 4, "s": 5, -0.0: 6, float("inf"): 7})
    r("log bad value", st.log, {"x": object()})
    r("log bad key", st.log, {(1, 2): 3})
    r("log list", st.log, [("a", 1)])
    r("log empty", st.log, {})
    os.remove(Path(root) / "passes" / f"{P['E1'].uid}.npz")
    st2 = cls(root, clock=clk)
    r("due partial", lambda: [st2.due(q) for q in range(Q_TEST + 18, Q_TEST + 27)])
    r("fresh accumulator", st2.accumulator, K["A"])
    r("fresh _zw", lambda: list(st2._zw))
    shutil.copytree(root, f"{root}-mid")           # the files before expiry removes them
    clk.t += 3 * 86400
    r("expire", st.expire)
    r("expire days", st.expire, None, 2.0)
    r("open_keys after", st.open_keys)
    r("pass_uids after", st.pass_uids)
    r("expire later", st.expire, clk.t + 8 * 86400)
    r("_zw end", lambda: list(st._zw))
    return r.out


def test_scripted_sequence(NStore, tmp_path):
    """The reference's and the C++'s results and files for the scripted
    sequence (module docstring)."""
    a = _script(S.Store, tmp_path / "py")
    b = _script(NStore, tmp_path / "cpp")
    assert len(a) > 100
    assert_same(a, b)
    justified = H.compare_trees(tmp_path / "py-mid", tmp_path / "cpp-mid")
    H.compare_trees(tmp_path / "py", tmp_path / "cpp")
    print(f"scripted sequence: {len(a)} operations, mean_w_db last-place differences {justified}")


# --- dotted callsigns, malformed accumulators, the cache signature ---------------------------------------

def _malformed(uid):
    """Accumulator JSONs whose values are not the types they hold (a hand edit)."""
    return {
        "members=5": {"callsign": "X", "picture_id": 1, "mode": 2, "members": 5},
        "meta=[1,2]": [1, 2],
        "meta=str": "str",
        "meta=null": None,
        "members=dict": {"callsign": "X", "picture_id": 1, "mode": 2, "members": {"segment": 0}},
        "member=int": {"callsign": "X", "picture_id": 1, "mode": 2, "members": [5]},
        "uid=int": {"callsign": "X", "picture_id": 1, "mode": 2, "members": [{"segment": 0, "uid": 5}]},
        "mode=null": {"callsign": "X", "picture_id": 1, "mode": None,
                      "members": [{"segment": 0, "uid": uid}]},
        "segment=null": {"callsign": "X", "picture_id": 1, "mode": 2,
                         "members": [{"segment": None, "uid": uid}]},
    }


def _odd_files(cls, root):
    P, K, X = _passes()
    st = cls(root, clock=Clock())
    r = Rec(root)
    acc_dir = Path(root) / "acc"
    for key in (("K1.AB", 1), ("K1.CD", 1), ("K1.AB/P", 2), ("W1.X.Y", 0xABC), ("dot.", 3)):
        r("paths " + key[0], st._acc_paths, key)
    r("create K1.AB", st.create, ("K1.AB", 1), 0, 0)
    r("has K1.CD", st.has_accumulator, ("K1.CD", 1))
    r("create K1.CD", st.create, ("K1.CD", 1), 2, 0)
    r("create W1.X.Y", st.create, ("W1.X.Y", 0xABC), 1, 7)
    r("accumulator K1.AB", st.accumulator, ("K1.AB", 1))
    r("files", lambda: sorted(p.name for p in acc_dir.iterdir()))
    r("open_keys", st.open_keys)
    r("add_pass", st.add_pass, P["A0"])
    r("attach", st.attach, K["A"], P["A0"].uid, 0, "header", mode=2)
    r("due good", lambda: [st.due(q) for q in range(Q_TEST, Q_TEST + 6)])
    for case, meta in _malformed(P["A0"].uid).items():
        (acc_dir / "AAA_00000001.json").write_text(json.dumps(meta))
        r("due " + case, lambda: [st.due(q) for q in range(Q_TEST, Q_TEST + 6)])
    (acc_dir / "AAA_00000001.json").unlink()
    # a pass rewritten with the same size and its mtime put back
    path = Path(root) / "passes" / f"{P['A0'].uid}.npz"
    r("pass_zw", lambda: st.pass_zw(P["A0"].uid)[1][:4].tolist())
    before = os.stat(path)
    cls(root, clock=Clock()).add_pass(dataclasses.replace(P["A0"], w=(P["A0"].w * 2).astype(np.float32),
                                                          em_round=1))
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    after = os.stat(path)
    assert (after.st_size, after.st_mtime_ns) == (before.st_size, before.st_mtime_ns)
    r("pass_zw rewritten", lambda: st.pass_zw(P["A0"].uid)[1][:4].tolist())
    r("pass_info rewritten", lambda: st.pass_info(P["A0"].uid)["em_round"])
    return r.out


def test_dotted_callsigns_malformed_accumulators_and_rewrites(NStore, tmp_path):
    """Both implementations name a dotted callsign's accumulator files
    <key_name>.json/.npz (store.py's with_suffix cut "K1.AB_00000001" to
    "K1"), give due rows past an accumulator JSON of the wrong types (a
    TypeError escaped store.py's due, a null mode or segment the C++'s), and
    read again a pass rewritten with the same size and mtime (the inode and
    change time in the signature); the same results and the same files."""
    a = _odd_files(S.Store, tmp_path / "py")
    b = _odd_files(NStore, tmp_path / "cpp")
    assert_same(a, b)
    files = dict((x[0], x[2]) for x in a)["files"]
    assert files == ("list", ["K1.AB_00000001.json", "K1.AB_00000001.npz", "K1.CD_00000001.json",
                              "K1.CD_00000001.npz", "W1.X.Y_00000abc.json", "W1.X.Y_00000abc.npz"])
    assert all(x[1] == "ok" for x in a if x[0].startswith("due")), [x for x in a if x[1] != "ok"]
    due = dict((x[0], x[2]) for x in a)
    assert all(due["due " + c] == due["due good"] for c in _malformed("u")), "a malformed one hides only itself"
    assert dict((x[0], x[2]) for x in a)["pass_info rewritten"] == ("int", "1")
    H.compare_trees(tmp_path / "py", tmp_path / "cpp")


# --- interop ---------------------------------------------------------------------------------------------

def _begin(cls, root):
    P, K, X = _passes()
    st = cls(root, clock=Clock())
    for name in ("A0", "A1", "A2", "A1L", "B1", "C0", "D1", "D2", "E0"):
        st.add_pass(P[name])
    st.attach(K["A"], P["A0"].uid, 0, "header", mode=2)
    st.attach(K["A"], P["A1"].uid, 1, "header")
    st.attach(K["B"], P["B1"].uid, 0, "corr")
    st.attach(K["C"], P["C0"].uid, 0, "header")
    assert st.merge(K["A"], *_foreign(X[0], X[1]), "rx2")
    assert st.merge(K["A"], *_foreign(X[4]))
    c = st.prov_add(P["D1"].uid)
    st.prov_add(P["D2"].uid, c)
    st.log({"event": "begun", "n": np.int64(4)})


def _continue(cls, root):
    P, K, X = _passes()
    clk = Clock(1.76e9 + 100.0)
    st = cls(root, clock=clk)
    r = Rec(root)
    for key in st.open_keys():
        acc = st.accumulator(key)
        S0, W0 = acc.S.copy(), acc.W.copy()
        acc.rebuild(st)
        assert np.array_equal(acc.S, S0) and np.array_equal(acc.W, W0), key   # bit for bit
        r("read " + S.key_name(key), lambda: acc)
    r("prov", st.prov_clusters)
    r("pass_info", st.pass_info, P["C0"].uid)
    r("load_pass", st.load_pass, P["A1L"].uid)
    clk.t += 1.0
    r("merge replaces", st.merge, K["A"], *_foreign(X[2], X[3]), "rx2")
    r("merge dup of the other's", st.merge, K["A"], *_foreign(X[4]))
    r("merge anon", st.merge, K["A"], *_foreign(X[1]), "")
    r("attach", st.attach, K["A"], P["A2"].uid, 2, "header")
    r("twin L", st.attach, K["A"], P["A1L"].uid, 1, "corr")
    r("update_pass", st.update_pass, dataclasses.replace(P["B1"], w=(P["B1"].w * 2).astype(np.float32),
                                                         em_round=1))
    r("prov_add", st.prov_add, P["E0"].uid, st.prov_cluster_of(P["D1"].uid)["cluster"])
    r("due", lambda: [st.due(q) for q in range(Q_TEST, Q_TEST + 10)])
    r("accumulator", st.accumulator, K["A"])
    shutil.copytree(root, f"{root}-mid")
    clk.t += 2 * 86400
    r("expire", st.expire, None, 1.0)
    return r.out


def test_interop_both_ways(NStore, tmp_path):
    """A store begun by either implementation, continued by either: the
    same results and the same files, the other's accumulators rebuilt bit
    for bit and its foreign parts kept and merged over."""
    for first, a in ((S.Store, "py"), (NStore, "cpp")):
        base = tmp_path / f"{a}-base"
        _begin(first, base)
        outs = {}
        for second, b in ((S.Store, "py"), (NStore, "cpp")):
            root = tmp_path / f"{a}-then-{b}"
            shutil.copytree(base, root)
            outs[b] = _continue(second, root)
        assert_same(outs["py"], outs["cpp"])
        H.compare_trees(tmp_path / f"{a}-then-py-mid", tmp_path / f"{a}-then-cpp-mid")
        H.compare_trees(tmp_path / f"{a}-then-py", tmp_path / f"{a}-then-cpp")
    H.compare_trees(tmp_path / "py-base", tmp_path / "cpp-base")


def _expiry(cls, root):
    """expire() at each of its three cutoffs exactly (kept: the reference
    drops only what is older) and a ulp past it (dropped)."""
    P, K, _ = _passes()
    clk = Clock(1.76e9)
    st = cls(root, clock=clk)
    r = Rec(root)
    for n in ("A0", "B1", "D1", "E0"):
        st.add_pass(P[n])
    st.attach(K["A"], P["A0"].uid, 0, "header", mode=2)        # updated 1.76e9
    clk.t += 50.0
    st.prov_add(P["D1"].uid)                                     # updated 1.76e9 + 50
    week = 7 * S.DAY_S
    edges = sorted({1.76e9, 1.76e9 + 50.0, S.slot_time(P["E0"].q), S.slot_time(P["B1"].q)})
    for e in edges:
        r(f"expire at {e}", st.expire, e + week)
        r(f"expire past {e}", st.expire, float(np.nextafter(e + week, np.inf)))
        r(f"left {e}", lambda: (st.open_keys(), st.pass_uids(), [c["cluster"] for c in st.prov_clusters()]))
    r("expire days", st.expire, 1.76e9 + 50.0 + 0.5 * S.DAY_S, 0.5)
    return r.out


def test_expire_at_the_cutoff(NStore, tmp_path):
    """An accumulator, a cluster and an orphan pass each exactly at the
    cutoff are kept and a ulp older is dropped, as in the reference."""
    a = _expiry(S.Store, tmp_path / "py")
    b = _expiry(NStore, tmp_path / "cpp")
    assert_same(a, b)
    nothing = norm({"acc": [], "prov": [], "passes": []}, str(tmp_path / "py"))
    assert all(x[1:] == ("ok", nothing) for x in a if x[0].startswith("expire at"))
    assert all(x[1] == "ok" and x[2] != nothing for x in a if x[0].startswith("expire past"))
    H.compare_trees(tmp_path / "py", tmp_path / "cpp")


# --- the seams -------------------------------------------------------------------------------------------

def test_clock_seam(NStore, tmp_path):
    """The clock is called for every stamp; one that raises stops the
    operation with nothing written, as in the reference. An int clock is
    written as a float here (the reference writes the int): a deviation."""
    P, K, _ = _passes()
    for cls, name in ((S.Store, "py"), (NStore, "cpp")):
        calls = []

        def clock(calls=calls):
            calls.append(1)
            if len(calls) == 3:
                raise RuntimeError("the clock stopped")
            return 1.7e9 + len(calls)

        st = cls(tmp_path / name, clock=clock)
        st.add_pass(P["B1"])
        st.prov_add(P["B1"].uid)                       # 1
        st.create(K["B"], 0, 0)                        # 2
        with pytest.raises(RuntimeError, match="the clock stopped"):
            st.attach(K["B"], P["B1"].uid, 0, "corr")  # 3: the commit's (after prov_remove)
        st.log({"x": 1})
        assert len(calls) == 4
    H.compare_trees(tmp_path / "py", tmp_path / "cpp")
    st = NStore(tmp_path / "int", clock=lambda: 1700000000)
    st.create(K["B"], 0, 0)
    meta = (tmp_path / "int" / "acc" / f"{S.key_name(K['B'])}.json").read_text()
    assert '"created": 1700000000.0' in meta
    st.clock = lambda: 5.0
    assert st.clock() == 5.0 and st.create(("X", 1), 0, 0).created == 5.0


def test_replace_seam(NStore, tmp_path, monkeypatch):
    """os.replace through the facade: a replace that fails mid-attach
    leaves the reference's files (the new .npz, the old .json, no
    temporary file) and raises the same error."""
    P, K, _ = _passes()
    real = os.replace
    for cls, name in ((S.Store, "py"), (NStore, "cpp")):
        st = cls(tmp_path / name, clock=Clock())
        for p in ("A0", "A1"):
            st.add_pass(P[p])
        st.attach(K["A"], P["A0"].uid, 0, "header", mode=2)
        n = []

        def boom(src, dst, n=n):
            n.append(dst)
            if len(n) == 2:
                raise OSError(28, "No space left on device")
            real(src, dst)

        monkeypatch.setattr(os, "replace", boom)
        with pytest.raises(OSError, match="No space left"):
            st.attach(K["A"], P["A1"].uid, 1, "header")
        monkeypatch.setattr(os, "replace", real)
        assert [Path(x).name for x in n] == [f"{S.key_name(K['A'])}.npz", f"{S.key_name(K['A'])}.json"]
    a, b = H.tree(tmp_path / "py"), H.tree(tmp_path / "cpp")
    assert a == b and not any(k.endswith(".tmp") for k in a)


def test_zw_cache_seam(NStore, tmp_path, monkeypatch):
    """store.ZW_CACHE read at every pass_zw: the same cache in the same
    order at sizes 0 (the pass returned, nothing cached) to 4."""
    P, K, _ = _passes()
    uids = [P[k].uid for k in ("A0", "A1", "A2", "B1", "C0")]
    seq = [0, 1, 2, 0, 3, 4, 1, 0, 2, 2, 4]
    for size in (0, 1, 2, 4, S.ZW_CACHE):
        monkeypatch.setattr(S, "ZW_CACHE", size)
        orders = []
        for cls, name in ((S.Store, "py"), (NStore, "cpp")):
            st = cls(tmp_path / f"{name}{size}", clock=Clock())
            for k in ("A0", "A1", "A2", "B1", "C0"):
                st.add_pass(P[k])
            got = []
            for i in seq:
                z, w = st.pass_zw(uids[i])
                got.append((_sha(z), _sha(w), list(st._zw)))
            orders.append(got)
        assert orders[0] == orders[1], size
        if size == 0:
            assert all(o[2] == [] for o in orders[0])


# --- threads ---------------------------------------------------------------------------------------------

def test_threads_hammer_one_store(NStore, tmp_path):
    """Eight threads on one C++ Store: nothing raises, and each accumulator
    left is exactly its members and parts summed (the reference's rebuild)."""
    P, K, X = _passes()
    st = NStore(tmp_path / "s", clock=Clock())
    names = ["A0", "A1", "A2", "B1", "B2", "C0", "D1", "D2", "D3", "E0"]
    for n in names:
        st.add_pass(P[n])
    errors = []
    F = [_foreign(X[0], X[1]), _foreign(X[2]), _foreign(X[4])]

    def work(seed):
        rng = random.Random(seed)
        try:
            for i in range(25):
                n = rng.choice(names)
                key = rng.choice([K["A"], K["B"], ("T", seed)])
                op = rng.randrange(8)
                if op == 0:
                    st.attach(key, P[n].uid, 0, "corr", mode=2)
                elif op == 1 and st.has_accumulator(key):
                    st.detach(key, P[n].uid)
                elif op == 2 and st.has_accumulator(key):
                    st.merge(key, *F[rng.randrange(3)], f"rx{seed}")
                elif op == 3:
                    try:
                        st.prov_add(P[n].uid, None if rng.random() < 0.5 else f"c_{P[n].uid}")
                    except FileNotFoundError:
                        pass
                elif op == 4:
                    st.prov_remove(P[n].uid)
                elif op == 5:
                    st.due(Q_TEST + rng.randrange(10))
                    st.open_keys()
                    st.find_member(P[n].uid)
                elif op == 6 and st.has_accumulator(key):
                    acc = st.accumulator(key)
                    acc.loo(acc.uids[0], st) if acc.uids else acc.meta()
                else:
                    st.pass_zw(P[n].uid)
                    st.pass_info(P[n].uid)
                    st.log({"thread": seed, "i": i})
        except Exception as e:  # noqa: BLE001
            errors.append(repr(e))

    threads = [threading.Thread(target=work, args=(s,)) for s in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    ref = S.Store(tmp_path / "s", clock=Clock())
    keys = ref.open_keys()
    assert keys
    for key in keys:
        acc = ref.accumulator(key)
        S0, W0 = acc.S.copy(), acc.W.copy()
        acc.rebuild(ref)
        assert np.array_equal(acc.S, S0) and np.array_equal(acc.W, W0), key
        assert len(acc.foreign) == len(acc.parts)
    held = [u for k in keys for u in ref.accumulator(k).uids]
    assert len(held) == len(set(held))                 # a pass belongs to one accumulator
    lines = (tmp_path / "s" / "log.jsonl").read_text().splitlines()
    assert all(line.startswith('{"time": ') and line.endswith("}") for line in lines)


def test_two_stores_one_root(NStore, tmp_path):
    """Two C++ Stores open on one directory (spelled two ways) used from two
    threads: they share the root's lock, so neither trips over the other's
    temporary file or loses the other's update of the accumulator both
    attach to. (Two reference Stores would; it claims no thread safety.)"""
    P, K, _ = _passes()
    root = tmp_path / "s"
    (tmp_path / "x").mkdir()
    a = NStore(root, clock=Clock())
    b = NStore(tmp_path / "x" / ".." / "s", clock=Clock())
    names = [["A0", "A1", "A2"], ["B1", "B2", "C0"]]
    for n in names[0] + names[1]:
        a.add_pass(P[n])
    a.create(K["A"], 2, 0)
    errors = []

    def work(st, mine):
        try:
            for _ in range(10):
                for n in mine:
                    st.attach(K["A"], P[n].uid, 0, "corr")
                for n in mine:
                    st.detach(K["A"], P[n].uid)
            for n in mine:
                st.attach(K["A"], P[n].uid, 0, "corr")
        except Exception as e:  # noqa: BLE001
            errors.append(repr(e))

    threads = [threading.Thread(target=work, args=(st, mine)) for st, mine in zip((a, b), names)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    ref = S.Store(root, clock=Clock())
    acc = ref.accumulator(K["A"])
    assert sorted(acc.uids) == sorted(P[n].uid for n in names[0] + names[1])
    S0, W0 = acc.S.copy(), acc.W.copy()
    acc.rebuild(ref)
    assert np.array_equal(acc.S, S0) and np.array_equal(acc.W, W0)
    assert not [f for d in ("acc", "passes", "prov") for f in os.listdir(root / d) if f.startswith(".")]


# --- em.py's store use, whole ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=1)
def _noise_sim():
    """test_qrss_em's noise-only SHORT slot (its template_search null test's)."""
    from test_qrss_em import simulate_pass

    return simulate_pass(frame.SHORT, -30.0, Q_TEST + 5, seed=3, signal=False)


def _em_run(tmp, monkeypatch):
    """em_prior, known_header, em_refine (a deterministic re-receive) and
    template_search, on a store built through sstvae.qrss.store.Store as
    the module holds it (the row, when the rows are in place)."""
    from sstvae.qrss import em, frontend, tx

    spec = frame.SHORT
    sp = synthetic_picture(0)
    h = make_header(sp, callsign="K1ABC")
    key = (h.callsign, int(h.picture_id))
    a = tx.slot_air_latents(sp, 0)[:spec.n_data]
    ch = np.exp(1j * np.arange(16) / 3.0).astype(np.complex64)
    ps = [dataclasses.replace(make_pass_result(a, 0.5, seed=s, hdr=h, hdr_es_n0_db=-16.0), ch=ch)
          for s in range(4)]
    st = S.Store(tmp / "store", clock=Clock())
    for p in ps:
        st.add_pass(p)
        st.attach(key, p.uid, 0, "test", mode=h.mode, codec_id=h.codec_id)
    out = []
    acc = st.accumulator(key)
    pr = em.em_prior(acc, ps[1].uid, st, Q_TEST, spec)
    out.append(("prior", _sha(pr.x_hat), _sha(pr.v)))
    out.append(("known_header", repr(em.known_header(st, acc, 0))))
    calls = []

    def rereceive(p, prior, hdr, cw, estimator=None, r=1, path=None, return_path=False):
        calls.append((p.uid, r, _sha(prior.x_hat), _sha(prior.v), repr(hdr), repr(cw)))
        new = dataclasses.replace(p, z=(p.z * 0.9 + 0.01 * r).astype(np.float32),
                                  w=(p.w * (1.0 + float(np.mean(prior.v)))).astype(np.float32),
                                  em_round=r)
        return (new, None) if return_path else new

    monkeypatch.setattr(em, "rereceive", rereceive)
    lines = []
    hist = em.em_refine(st, key, rounds=3, log=lines.append)
    assert len(calls) >= 4 and len(hist) >= 2
    out += [("history", [repr(x) for x in hist]), ("calls", calls), ("log", lines)]
    sim = _noise_sim()
    det, Z, info = em.template_search(st, key, frontend.Capture(sim.fe, Q_TEST + 5, sim.t0_index),
                                      spec, return_stats=True)
    out.append(("template", repr(det), repr(Z), repr(sorted(info.items()))))
    return out


def test_em_whole_caller(native, NStore, tmp_path, monkeypatch):
    """em.py on the C++ store, every row of this lane in place with
    counting spies before them: what it does on the reference's store,
    with the Store and canonical_index rows reached."""
    with monkeypatch.context() as mp:
        ref = _em_run(tmp_path / "py", mp)
    used = {}
    with monkeypatch.context() as mp:
        H.apply_store_rows(native, mp, wrap=H.counting(used))
        got = _em_run(tmp_path / "cpp", mp)
    assert got == ref
    assert used.get("Store", 0) >= 1 and used.get("canonical_index", 0) >= 1, used
    H.compare_trees(tmp_path / "py" / "store", tmp_path / "cpp" / "store")

"""Parity of native/core/qrss/passband, flac and dirlock
(sstvae_native.qrss.passband) with frontend.PassbandStore, _merge_ranges,
slot_t0 and live.dir_lock. tests/qrss_native/passband.py substitutes them
under --native-qrss=rx.

The contract (the bar):

- the store's files: the same sequence of writes, reads, captures, packs
  and expiries, drawn at random (hour boundaries, negative hours, gaps,
  clipping, NaN and infinite samples, ties at half an LSB, zero lengths,
  attenuated captures, non-finite ones included), through both stores
  into separate roots, leaves the same files with the same bytes -- every
  fe_<hour>.i16, every fe_<hour>.json, meta.json -- and no temporary file;
  a short raw hour reads alike (what it holds, unwritten past its end);
  expire() deletes the same stale temporaries; a packed hour is the
  same samples (the two FLAC encoders are not byte-identical: each picks
  its own subframes; tests/test_qrss_native_flac.py holds the C++ one to
  libFLAC, dr_flac and ffmpeg). Every result on the way is bitwise the reference's: read's
  samples and mask, capture's Capture (its fe, q, t0_index, fs) and
  coverage, write_capture's (time, gain), slots, hours, raw_hours,
  expire, pack_hour; and the refusals, type and text;
- across implementations: an hour the reference wrote, raw or packed by
  soundfile, reads back through the C++ exactly as through the reference,
  at arbitrary offsets inside and across hours; an hour the C++ packed
  reads back through the reference (soundfile) and through dr_flac as it
  was written raw; a write into an hour the other implementation packed
  unpacks it to the same bytes as the reference unpacking its own;
- the FLAC writer and reader on an edge corpus (lengths about the block
  size, mono and stereo, full scale, -32768, constants, alternations,
  sines, noise; sample rates from 1 Hz to 655350 Hz, standard and not;
  block sizes 16 to 65535; the stereo modes off and on), read back by
  soundfile and by dr_flac sample for sample, the STREAMINFO MD5 the one
  libFLAC computes; and soundfile's own files read by the C++ reader,
  partial reads at arbitrary offsets included;
- the lock: a Python listener's lock refuses a C++ one and the reverse,
  across processes and inside one, and either is free again once its
  holder closes it or exits; the lock file holds "<pid>\\n";
- the whole caller, with counting spies installed before the rows (the
  counters are what the module attributes and from-import sites hold):
  a LiveListener writing the store as it hears audio across an hour
  boundary, a second one backfilling its ring and recovering a slot from
  it (the end-of-slot receive stubbed: it records what it was handed),
  em._from_passband and em.retro_detect reading slots out of it
  (template_search stubbed), and qrss_listen.py's main taking its three
  locks and opening the store -- the reference's outcomes, files and
  logs, with every row reached in one configuration or the other
  (_merge_ranges has no caller once the store is native: it is reached
  under a Python store).

Measured (2026-10-14, this machine; not asserted): the C++ packs a
simulated passband hour (channel.py transmissions over band noise through
live.StreamFE) in 0.4-1.5 s of one core against soundfile's 0.4-0.8 s, to
0.90-0.97x its size: LPC to order 12 with partial_tukey windows, where
libFLAC level 5 stops at 8 with tukey(0.5) alone (the FIXED-only encoder
before it: 1.21-1.55x). See tests/test_qrss_native_flac.py.
"""

import filecmp
import hashlib
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from qrss_helpers import Q_TEST, REPO_ROOT
from qrss_native import require

import qrss_native
from sstvae.qrss import em, frame, frontend, live
from sstvae.qrss.constants import FE_FS

FRONTEND = "sstvae.qrss.frontend"
PY_DIR_LOCK = live.dir_lock          # the reference's, whatever a whole-caller test assigns over it


@pytest.fixture(scope="module")
def P(native):
    return require(native, "passband")


def _sf():
    return pytest.importorskip("soundfile")


# --- the free functions -----------------------------------------------------------------------------

def test_constants_and_free_functions(P):
    for name in ("PB_PACK_AFTER_S", "PB_LEAD_S", "PB_TAIL_S"):
        assert getattr(P, name) == getattr(frontend, name), name
    for q in (0, 1, -1, Q_TEST, -Q_TEST, 2 ** 40, np.int64(17), np.int32(-3), 3.7, -3.7, True, "12"):
        assert P.slot_t0(q) == frontend.slot_t0(q) and type(P.slot_t0(q)) is int, q
    rng = np.random.default_rng(1)
    cases = [[], [[0, 0]], [[5, 9], [0, 3], [3, 4], [8, 12]], [[1, 2], [1, 2]], [(3, 5), (5, 6)],
             [[np.int64(4), np.int32(9)], [0, 4]], [[10, 1], [0, 2]], [[-5, -1], [-1, 3]]]
    for _ in range(300):
        n = int(rng.integers(0, 12))
        lo = rng.integers(-50, 50, n)
        cases.append([[int(a), int(a + rng.integers(0, 20))] for a in lo])
    for rs in cases:
        got, ref = P._merge_ranges(rs), frontend._merge_ranges(rs)
        assert got == ref and all(type(v) is int for r in got for v in r), rs
    names = ["fe_12.i16", "fe_12.flac", "fe_0012.i16", "fe_12.i16\n", "fe_.i16", "fe_12.wav", "fe_-1.i16",
             "fe_12.json", "xfe_12.i16", "fe_12.i16.tmp", "fe_12.flac.77.tmp", "fe_12.I16", "fe_99999999.flac",
             "fe_12.i16\n\n", "fe_1 2.i16", "meta.json", ""]
    for name in names:
        m = frontend._HOUR_RE.match(name)
        assert P._hour_of_name(name) == (int(m.group(1)) if m else None), name
    # (ASCII digits only, in both: \d took the Arabic-Indic ones before 2026-10-14's fix)
    assert not frontend._HOUR_RE.match("fe_١٢.i16") and P._hour_of_name("fe_١٢.i16") is None


# --- the random sequences ---------------------------------------------------------------------------

def _samples(rng, n, scale):
    """complex128 samples: noise at a random level, with the edges sprinkled in."""
    lvl = float(rng.choice([0.01, 0.3, 2.0, 5.0])) * 8192.0 / scale
    x = (rng.standard_normal(n) + 1j * rng.standard_normal(n)) * lvl
    if n:
        k = rng.integers(0, n, max(1, n // 50))
        special = np.array([np.nan, np.inf, -np.inf, 4.0, -4.0, 1e39, -1e39, 0.5 / scale, 1.5 / scale,
                            -2.5 / scale, 32767.5 / scale, 0.0], dtype=np.float64)
        x.real[k] = rng.choice(special, len(k))
        x.imag[k] = rng.choice(special, len(k))
        if rng.random() < 0.2:
            x[:] = complex(rng.integers(-3, 4)) / scale      # a constant (a silent hour, packed)
    return x


def _length(rng, fs, long_s):
    return int(rng.choice([0, 1, int(rng.integers(1, fs * 60)), int(rng.integers(1, fs * long_s))]))


def _call(fn, *a, **k):
    with np.errstate(all="ignore"):
        try:
            return ("ok", fn(*a, **k))
        except Exception as e:      # noqa: BLE001 -- the refusal is the result compared
            return ("raise", type(e), str(e))


def _same_result(got, ref, what):
    if ref[0] == "raise":
        assert got[0] == "raise" and got[1] is ref[1] and got[2] == ref[2], (what, got, ref)
        return
    assert got[0] == "ok", (what, got)
    g, r = got[1], ref[1]
    if isinstance(r, tuple) and len(r) == 2 and isinstance(r[0], np.ndarray):    # read with its mask
        _same_array(g[0], r[0], what)
        _same_array(g[1], r[1], what)
    elif isinstance(r, np.ndarray):
        _same_array(g, r, what)
    elif isinstance(r, tuple) and len(r) == 2 and (r[0] is None or isinstance(r[0], frontend.Capture)):
        assert (g[0] is None) == (r[0] is None) and g[1] == r[1] and type(g[1]) is type(r[1]), what
        if r[0] is not None:
            assert type(g[0]) is type(r[0]), what
            _same_array(g[0].fe, r[0].fe, what)
            for f in ("q", "t0_index", "fs"):
                assert getattr(g[0], f) == getattr(r[0], f), (what, f)
                assert type(getattr(g[0], f)) is type(getattr(r[0], f)), (what, f)
    else:
        assert g == r and type(g) is type(r), (what, g, r)
        if isinstance(r, (tuple, list)):
            assert all(type(a) is type(b) for a, b in zip(g, r)), what


def _same_array(a, b, what):
    a, b = np.asarray(a), np.asarray(b)
    assert a.dtype == b.dtype and a.shape == b.shape, (what, a.dtype, b.dtype, a.shape, b.shape)
    assert a.tobytes() == b.tobytes(), what


def _flac_samples(path):
    y, _ = _sf().read(str(path), dtype="int16", always_2d=True)
    return y


def _same_tree(a: Path, b: Path, what=""):
    """The same file names, the same bytes; a packed hour the same samples."""
    na = sorted(p.name for p in a.iterdir() if p.name != ".lock")
    nb = sorted(p.name for p in b.iterdir() if p.name != ".lock")
    assert na == nb, (what, na, nb)
    assert not [n for n in na if n.endswith(".tmp")], what
    for name in na:
        if name.endswith(".flac"):
            assert np.array_equal(_flac_samples(a / name), _flac_samples(b / name)), (what, name)
        else:
            assert filecmp.cmp(a / name, b / name, shallow=False), (what, name)


def _run(P, tmp_path, seed, fs, h0, n_ops, scale, keep_hours, long_s=5400.0):
    rng = np.random.default_rng(seed)
    ra, rb = tmp_path / f"py{seed}", tmp_path / f"cc{seed}"
    ref = frontend.PassbandStore(ra, fs=fs, scale=scale, keep_hours=keep_hours)
    got = P.PassbandStore(rb, fs=fs, scale=scale, keep_hours=keep_hours)
    for f in ("fs", "scale", "keep_hours", "per_hour", "pack"):
        assert getattr(got, f) == getattr(ref, f) and type(getattr(got, f)) is type(getattr(ref, f)), f
    assert got.root == rb and isinstance(got.root, Path) and got._packer is None
    base = h0 * 3600.0
    span = (base - 1800.0, base + 3 * 3600.0)
    counts: dict = {}
    for i in range(n_ops):
        op = str(rng.choice(["write", "write", "write_at", "write_capture", "read", "read", "capture",
                             "slots", "hours", "pack", "expire"],
                            p=[0.16, 0.12, 0.08, 0.1, 0.14, 0.08, 0.12, 0.05, 0.05, 0.06, 0.04]))
        counts[op] = counts.get(op, 0) + 1
        what = (seed, i, op)
        if op == "write":
            t = float(rng.uniform(*span))
            if rng.random() < 0.3:          # about an hour boundary
                t = (h0 + int(rng.integers(0, 3))) * 3600.0 - float(rng.uniform(0, min(long_s, 600.0)))
            if rng.random() < 0.3:
                t = round(t * fs) / fs + float(rng.choice([0.0, 0.5, -0.5, 0.25])) / fs
            x = _samples(rng, _length(rng, fs, long_s), scale)
            expire = bool(rng.random() < 0.3)
            res = [_call(s.write, t, x, expire=expire) for s in (got, ref)]
        elif op == "write_at":
            n0 = int(rng.uniform(*span) * fs)
            with np.errstate(over="ignore"):
                x = _samples(rng, _length(rng, fs, long_s), scale).astype(np.complex64)
            expire = bool(rng.random() < 0.3)
            res = [_call(s._write_at, n0, x, expire) for s in (got, ref)]
        elif op == "write_capture":
            q = int(rng.integers(int(span[0] // 900), int(span[1] // 900) + 1))
            x = _samples(rng, _length(rng, fs, long_s), scale)
            if rng.random() < 0.5 and len(x):
                x = np.where(np.isfinite(x), x, 0) * float(rng.choice([1.0, 3.0, 9.5, 1e3]))
            t0i = float(rng.choice([0.0, rng.uniform(0, 30 * fs), int(rng.integers(0, 30 * fs)) + 0.5]))
            with np.errstate(over="ignore"):
                cap = frontend.Capture(fe=x.astype(np.complex64), q=q, t0_index=t0i, fs=fs)
            expire = bool(rng.random() < 0.3)
            res = [_call(s.write_capture, cap, expire=expire) for s in (got, ref)]
        elif op == "read":
            a = float(rng.uniform(*span))
            b = a + float(rng.choice([-5.0, 0.0, 1.0 / fs, rng.uniform(0, 600), rng.uniform(0, long_s)]))
            mask = bool(rng.random() < 0.7)
            res = [_call(s.read, a, b, return_mask=mask) for s in (got, ref)]
        elif op == "capture":
            q = int(rng.integers(int(span[0] // 900) - 1, int(span[1] // 900) + 2))
            dur = float(rng.choice([1.0, 120.0, 330.0, 800.0, 3600.0]))
            kw = {} if rng.random() < 0.6 else dict(lead_s=float(rng.choice([0.0, 2.5, 40.0])),
                                                     tail_s=float(rng.choice([0.0, 0.1, 30.0])))
            res = [_call(s.capture, q, dur, **kw) for s in (got, ref)]
        elif op == "slots":
            dur = float(rng.choice([0.0, 60.0, 330.0, 899.0, 900.0, 1801.0]))
            res = [_call(s.slots, dur) for s in (got, ref)]
        elif op == "hours":
            res = [(_call(s.hours), _call(s.raw_hours)) for s in (got, ref)]
            _same_result(res[0][0], res[1][0], what)
            _same_result(res[0][1], res[1][1], what)
            continue
        elif op == "pack":
            hours = ref.hours() or [h0]
            h = int(rng.choice(hours + [h0 - 7]))
            res = [_call(s.pack_hour, h) for s in (got, ref)]
        else:
            now = float(rng.uniform(base, base + (keep_hours + 4) * 3600.0))
            res = [_call(s.expire, now) for s in (got, ref)]
        _same_result(res[0], res[1], what)
        if rng.random() < 0.1:
            _same_tree(ra, rb, what)
    _same_tree(ra, rb, (seed, "end"))
    # a restart on each root: meta.json wins over the arguments, and reads the same
    again = P.PassbandStore(rb, fs=fs + 1, scale=scale * 2)
    ref_again = frontend.PassbandStore(ra, fs=fs + 1, scale=scale * 2)
    assert (again.fs, again.scale) == (ref_again.fs, ref_again.scale) == (fs, scale)
    _same_result(_call(again.read, *span, return_mask=True), _call(ref_again.read, *span, return_mask=True), seed)
    return counts


SEQUENCES = {
    # seed: (fs, first hour, ops, scale, keep_hours)
    1: (40, 490_000, 140, 8192.0, 3.0),
    2: (40, 0, 140, 1.0, 1.5),          # hours around 0, negative ones included
    3: (25, 490_000, 140, 1000.5, 48.0),
    4: (7, -3, 140, 32767.0, 2.0),
}


@pytest.mark.parametrize("seed", list(SEQUENCES))
def test_random_sequences(P, tmp_path, seed):
    """The same operations through both stores: the same results, bit for
    bit, and the same files. (A small fs keeps an hour's file small; the
    code is the same at FE_FS, which the cross-implementation tests use.)"""
    _sf()
    counts = _run(P, tmp_path, seed, *SEQUENCES[seed])
    assert all(counts.get(k, 0) for k in ("write", "write_at", "write_capture", "read", "capture",
                                          "slots", "hours", "pack", "expire")), counts


def test_a_sequence_at_the_fe_rate(P, tmp_path):
    _sf()
    _run(P, tmp_path, 5, FE_FS, 490_000, 30, 8192.0, 48.0, long_s=400.0)


def test_refusals(P, tmp_path):
    """The reference's exceptions, type and text (an OSError's errno and file name too)."""
    (tmp_path / "file").write_text("x")
    for root in (tmp_path / "file", tmp_path / "file" / "below", tmp_path / "file" / "below" / "deeper"):
        g = _call(P.PassbandStore, root)
        r = _call(frontend.PassbandStore, root)
        assert g[0] == r[0] == "raise" and g[1] is r[1] and g[2] == r[2], (g, r)
    for meta, exc in (('{"scale": 2.0}', KeyError), ('{"fs": 40}', KeyError), ("{", ValueError),
                      ('{"fs": "x", "scale": 1}', ValueError)):
        d = tmp_path / f"m{abs(hash(meta))}"
        d.mkdir()
        (d / "meta.json").write_text(meta)
        g, r = _call(P.PassbandStore, d), _call(frontend.PassbandStore, d)
        assert g[0] == r[0] == "raise" and issubclass(g[1], exc) and issubclass(r[1], exc), (meta, g, r)
        if exc is KeyError:
            assert g[1] is r[1] and g[2] == r[2], (meta, g, r)
    # meta.json's values as int(m["fs"]) and float(m["scale"]) take them (a
    # numeric string is a number), and the TypeError of a value they refuse,
    # or of a meta.json that is no dict, type and text
    for k, meta in enumerate(('{"fs": " 4_000 ", "scale": "8192"}', '{"fs": "\\u0664\\u0660", "scale": 1}',
                              '{"fs": "4000.0", "scale": 1}', '{"fs": 40, "scale": "x"}',
                              '{"fs": null, "scale": 1}', '{"fs": [40], "scale": 1}', '{"fs": 40, "scale": {}}',
                              "[40]", '"fs"', "40", "null")):
        d = tmp_path / f"meta{k}"
        d.mkdir()
        (d / "meta.json").write_text(meta)
        g, r = _call(P.PassbandStore, d), _call(frontend.PassbandStore, d)
        if r[0] == "ok":
            assert g[0] == "ok" and (g[1].fs, g[1].scale) == (r[1].fs, r[1].scale), (meta, g, r)
        else:
            _same_result(g, r, meta)
    # fs <= 0, given or in meta.json: the same ValueError, and no meta.json written
    for fs in (0, -40, -1):
        g = _call(P.PassbandStore, tmp_path / f"cfs{fs}", fs=fs)
        r = _call(frontend.PassbandStore, tmp_path / f"pfs{fs}", fs=fs)
        assert g[0] == r[0] == "raise" and g[1] is r[1] is ValueError and g[2] == r[2], (fs, g, r)
        assert not (tmp_path / f"cfs{fs}" / "meta.json").exists() and not (tmp_path / f"pfs{fs}" / "meta.json").exists()
    d = tmp_path / "mfs0"
    d.mkdir()
    (d / "meta.json").write_text('{"fs": 0, "scale": 8192.0}')
    g, r = _call(P.PassbandStore, d, fs=40), _call(frontend.PassbandStore, d, fs=40)
    assert g[0] == r[0] == "raise" and g[1] is r[1] is ValueError and g[2] == r[2], (g, r)
    ref = frontend.PassbandStore(tmp_path / "py", fs=40)
    got = P.PassbandStore(tmp_path / "cc", fs=40)
    x = np.ones(10, np.complex64)
    for args in ((math.nan, x), (math.inf, x), (-math.inf, x), ("1000", x)):
        _same_result(_call(got.write, *args, expire=False), _call(ref.write, *args, expire=False), args)
    for args in ((math.nan, 1.0), (0.0, math.inf), (1.0, 2.0)):
        _same_result(_call(got.read, *args), _call(ref.read, *args), args)
    cap = frontend.Capture(fe=x, q=Q_TEST, t0_index=0.0, fs=4000)
    _same_result(_call(got.write_capture, cap), _call(ref.write_capture, cap), "fs")
    cap = frontend.Capture(fe=x, q=Q_TEST, t0_index=math.nan, fs=40)
    _same_result(_call(got.write_capture, cap), _call(ref.write_capture, cap), "t0_index")
    _same_result(_call(got.capture, Q_TEST, math.nan), _call(ref.capture, Q_TEST, math.nan), "dur")
    _same_result(_call(got.expire, math.nan), _call(ref.expire, math.nan), "expire")
    _same_tree(tmp_path / "py", tmp_path / "cc")


def test_a_short_raw_hour(P, tmp_path):
    """A raw hour shorter than an hour (a crash between its creation and its
    sizing): reads give what it holds and the rest as unwritten (a final
    partial pair too; they raised ValueError before 2026-10-14's fix), a
    pack refuses it, a write grows it back to size (numpy's memmap "r+"),
    keeping what was there, and the two stores are then the same files."""
    _sf()
    ra, rb = tmp_path / "py", tmp_path / "cc"
    ref, got = frontend.PassbandStore(ra, fs=40), P.PassbandStore(rb, fs=40)
    h = 490_000
    for size in (0, 4 * 7 + 2, 4 * 3000):
        for r in (ra, rb):
            (r / f"fe_{h}.i16").write_bytes(bytes(range(256)) * (size // 256) + bytes(range(size % 256)))
            (r / f"fe_{h}.json").write_text("[[0, 7]]")
        span = (h * 3600.0, h * 3600.0 + 2.0)
        _same_result(_call(got.read, *span, return_mask=True), _call(ref.read, *span, return_mask=True), size)
        _same_result(_call(got.capture, h * 4, 1.0), _call(ref.capture, h * 4, 1.0), size)
        _same_result(_call(got.pack_hour, h), _call(ref.pack_hour, h), ("short", size))
        x = np.full(5, 0.25 - 0.5j, np.complex64)
        _same_result(_call(got.write, h * 3600.0 + 1.0, x, expire=False),
                     _call(ref.write, h * 3600.0 + 1.0, x, expire=False), size)
        _same_tree(ra, rb, size)
        _same_result(_call(got.read, *span, return_mask=True), _call(ref.read, *span, return_mask=True), size)
        _same_result(_call(got.pack_hour, h), _call(ref.pack_hour, h), size)
        _same_tree(ra, rb, size)
        for r in (ra, rb):
            for p in r.glob("fe_*"):
                p.unlink()


def test_stale_temporaries(P, tmp_path):
    """expire() deletes the store's temporaries whose mtime is over an hour
    old by the wall clock (whatever `now` says), and nothing else: the same
    files left in both roots. Every kind (.flac, .i16, fe_<h>.json,
    meta.json; one id or two) is there both stale and fresh, and two stale
    temporaries are directories, which the sweep leaves without raising."""
    import time

    ra, rb = tmp_path / "py", tmp_path / "cc"
    ref, got = frontend.PassbandStore(ra, fs=40), P.PassbandStore(rb, fs=40)
    stale = {"fe_1.flac.123.tmp": 7200.0, "fe_1.i16.5.6.tmp": 3700.0, "meta.json.7.8.tmp": 7200.0,
             "fe_2.json.9.10.tmp": 3700.0, "fe_12.i16.0.tmp": 7200.0, "fe_4.json.1.tmp": 7200.0,
             "meta.json.1.tmp": 3700.0, "fe_9.flac.1.2.tmp": 3601.0}
    fresh = {"fe_3.flac.11.12.tmp": 3500.0, "fe_5.flac.6.tmp": 10.0, "fe_6.i16.1.2.tmp": 3500.0,
             "fe_7.i16.3.tmp": 10.0, "meta.json.3.4.tmp": 3500.0, "meta.json.5.tmp": 10.0,
             "fe_7.json.5.tmp": 3500.0, "fe_8.json.6.7.tmp": 10.0}
    other = {n: 7200.0 for n in ("notes.1.tmp", "fe_1.wav.1.tmp", "fe_1.flac.tmp", "fe_1.flac.x.tmp",
                                  "fe_1.flac.1.2.3.tmp", "fe_.flac.1.tmp", "meta.json.1.tmp.tmp",
                                  "fe_\u0663.flac.1.tmp", "fe_1.json.1.tmp.x", "xfe_1.flac.1.tmp",
                                  "fe_1.i16.1.TMP")}
    dirs = {"fe_10.flac.1.2.tmp": 7200.0, "meta.json.9.tmp": 7200.0}
    now = time.time()
    for r in (ra, rb):
        for n, age in {**stale, **fresh, **other}.items():
            (r / n).write_bytes(b"x")
            os.utime(r / n, (now - age, now - age))
        for n, age in dirs.items():
            (r / n).mkdir()
            if n.startswith("fe_"):
                (r / n / "x").write_bytes(b"x")
            os.utime(r / n, (now - age, now - age))
    for when in (0.0, 1e12):
        assert got.expire(when) == ref.expire(when) == []
        assert sorted(p.name for p in ra.iterdir()) == sorted(p.name for p in rb.iterdir())
    assert sorted(p.name for p in ra.iterdir()) == sorted({"meta.json", *fresh, *other, *dirs})
    assert (ra / "fe_10.flac.1.2.tmp" / "x").exists() and (rb / "fe_10.flac.1.2.tmp" / "x").exists()


# --- across implementations -------------------------------------------------------------------------

def _fill(store, h0, rng):
    """Two stretches of band-like noise across the h0 / h0 + 1 boundary, a gap between."""
    x = (rng.standard_normal(200 * FE_FS) + 1j * rng.standard_normal(200 * FE_FS)) * 0.1
    store.write((h0 + 1) * 3600.0 - 150.0, x, expire=False)
    store.write((h0 + 1) * 3600.0 + 100.0, x[:30 * FE_FS] * 3, expire=False)
    store.write(h0 * 3600.0 + 7.3, x[:5 * FE_FS], expire=False)


def _spans(rng, h0, n):
    lo, hi = h0 * 3600.0 - 20.0, (h0 + 1) * 3600.0 + 200.0
    out = [(h0 * 3600.0 + 3450.0, (h0 + 1) * 3600.0 + 130.0), (h0 * 3600.0, (h0 + 2) * 3600.0)]
    for _ in range(n):
        a = float(rng.uniform(lo, hi))
        out.append((a, a + float(rng.choice([1 / FE_FS, rng.uniform(0, 30), rng.uniform(0, 400)]))))
    return out


def test_reads_across_implementations(P, tmp_path):
    """The reference writes and packs (soundfile); the C++ reads the same.
    The C++ packs; the reference (soundfile) and dr_flac read the same.
    Each writes into an hour the other packed: the same bytes as the
    reference unpacking its own."""
    sf = _sf()
    rng = np.random.default_rng(11)
    h0 = 490_000
    a, b = tmp_path / "a", tmp_path / "b"
    ref = frontend.PassbandStore(a)
    _fill(ref, h0, rng)
    raw_h1 = np.fromfile(a / f"fe_{h0 + 1}.i16", np.int16).reshape(-1, 2)
    assert ref.pack_hour(h0) and ref.raw_hours() == [h0 + 1]
    spans = _spans(rng, h0, 40)
    got = P.PassbandStore(a)
    want = [ref.read(s, e, return_mask=True) for s, e in spans]
    for (s, e), w in zip(spans, want):
        _same_result(("ok", got.read(s, e, return_mask=True)), ("ok", w), (s, e))
    for q in range(h0 * 4 - 1, h0 * 4 + 6):
        _same_result(("ok", got.capture(q, 330.0)), ("ok", ref.capture(q, 330.0)), q)
    assert got.slots(330.0) == ref.slots(330.0)
    # the C++ packs hour h0 + 1 of a copy; both read it as it was
    shutil.copytree(a, b)
    got_b = P.PassbandStore(b)
    assert got_b.pack_hour(h0 + 1) and got_b.raw_hours() == [] and got_b.hours() == [h0, h0 + 1]
    flac = b / f"fe_{h0 + 1}.flac"
    y, fs = sf.read(str(flac), dtype="int16", always_2d=True)
    assert fs == FE_FS and np.array_equal(y, raw_h1)
    assert np.array_equal(P._flac_read(flac), raw_h1)
    for k0 in (0, 1, 4095, 4096, 4097, 3_000_000, len(raw_h1) - 1):      # partial reads, anywhere
        k1 = min(len(raw_h1), k0 + int(rng.integers(1, 100_000)))
        assert np.array_equal(P._flac_read(flac, k0, k1), raw_h1[k0:k1]), k0
        assert np.array_equal(sf.read(str(flac), start=k0, stop=k1, dtype="int16", always_2d=True)[0],
                              raw_h1[k0:k1]), k0
    ref_b = frontend.PassbandStore(b)
    for (s, e), w in zip(spans, want):
        _same_result(("ok", ref_b.read(s, e, return_mask=True)), ("ok", w), ("ref reads C++ flac", s, e))
        _same_result(("ok", got_b.read(s, e, return_mask=True)), ("ok", w), ("C++ reads C++ flac", s, e))
    # writes into packed hours: h0 packed by soundfile (the C++ unpacks it in b, the
    # reference in a), h0 + 1 packed by the C++ (the reference unpacks it in b; a's is raw)
    x = (rng.standard_normal(2 * FE_FS) + 1j * rng.standard_normal(2 * FE_FS)) * 0.2
    for t in (h0 * 3600.0 + 1000.0, (h0 + 1) * 3600.0 + 1000.0):
        ref.write(t, x, expire=False)
    got_b.write(h0 * 3600.0 + 1000.0, x, expire=False)
    ref_b.write((h0 + 1) * 3600.0 + 1000.0, x, expire=False)
    _same_tree(a, b)
    assert sorted(p.name for p in a.iterdir()) == sorted(["meta.json", f"fe_{h0}.i16", f"fe_{h0}.json",
                                                          f"fe_{h0 + 1}.i16", f"fe_{h0 + 1}.json"])


def test_pack_unpack_and_expire_round_trip(P, tmp_path):
    """The C++ alone: packed, read, written into (unpacked), packed again,
    expired -- the reference's files at each step."""
    _sf()
    rng = np.random.default_rng(12)
    h0 = 490_001
    ra, rb = tmp_path / "py", tmp_path / "cc"
    stores = (frontend.PassbandStore(ra, fs=50), P.PassbandStore(rb, fs=50))
    x = (rng.standard_normal(9000) + 1j * rng.standard_normal(9000)) * 0.05
    for s in stores:
        s.write(h0 * 3600.0 + 3500.0, x, expire=False)
        assert s.pack_hour(h0) and s.pack_hour(h0 + 1) and not s.pack_hour(h0)   # (packed: no raw file)
        assert s.raw_hours() == [] and s.hours() == [h0, h0 + 1]
    _same_tree(ra, rb)
    for s in stores:
        s.write(h0 * 3600.0 + 10.0, x[:100], expire=False)
    _same_tree(ra, rb)
    assert (rb / f"fe_{h0}.i16").exists() and (rb / f"fe_{h0 + 1}.flac").exists()
    _same_result(_call(stores[1].read, h0 * 3600.0, h0 * 3600.0 + 7200.0, return_mask=True),
                 _call(stores[0].read, h0 * 3600.0, h0 * 3600.0 + 7200.0, return_mask=True), "read")
    for s in stores:
        assert s.pack_hour(h0)
        assert s.expire((h0 + 1) * 3600.0 + 48 * 3600.0) == [h0]
        assert s.expire((h0 + 2) * 3600.0 + 48 * 3600.0) == [h0 + 1]
    _same_tree(ra, rb)
    assert sorted(p.name for p in rb.iterdir()) == ["meta.json"]


def test_the_packer_thread(P, tmp_path):
    """pack=True: an hour is packed on a thread of its own PB_PACK_AFTER_S
    after it ends by the stream's time, checked at most once a minute; the
    hour being written stays raw. The same files as the reference's."""
    _sf()
    ra, rb = tmp_path / "py", tmp_path / "cc"
    ref, got = frontend.PassbandStore(ra, pack=True), P.PassbandStore(rb, pack=True)
    assert got.pack and ref.pack
    h0 = 490_000
    x = np.full(10 * FE_FS, 0.1 + 0.2j, dtype=np.complex64)
    for s in (ref, got):
        s.write(h0 * 3600 + 3590.0, x, expire=False)
        s.write((h0 + 1) * 3600 + 60.0, x)
        assert s._packer is None
        s.write((h0 + 1) * 3600 + frontend.PB_PACK_AFTER_S + 70.0, x)
        assert s._packer is not None and s._packer.name == "passband-pack" and s._packer.daemon
        s._packer.join(30.0)
        assert not s._packer.is_alive()
        assert s.raw_hours() == [h0 + 1] and s.hours() == [h0, h0 + 1]
        s.write((h0 + 2) * 3600 + 30.0, x)       # (hour h0 + 1 is not due yet)
        assert s.raw_hours() == [h0 + 1, h0 + 2]
    _same_tree(ra, rb)


# --- the FLAC edge corpus ---------------------------------------------------------------------------

def _corpus():
    rng = np.random.default_rng(13)
    out = []
    for n in (1, 2, 5, 15, 16, 17, 1151, 4095, 4096, 4097, 3 * 4096 + 5, 70_000):
        t = np.arange(n)
        sig = {
            "zeros": np.zeros(n),
            "const": np.full(n, -7),
            "full+": np.full(n, 32767),
            "min": np.full(n, -32768),
            "alt": np.where(t % 2, 32767, -32768),
            "ramp": (t * 37) % 65536 - 32768,
            "sine": np.round(30000 * np.sin(t * 0.013)),
            "noise": rng.integers(-32768, 32768, n),
            "quiet": rng.integers(-3, 4, n),
            "spike": np.where(t == n // 2, -32768, 0),
        }
        for name, s in sig.items():
            if n > 5000 and name not in ("noise", "sine", "spike"):
                continue
            mono = np.asarray(s, np.int64).astype(np.int16)
            other = {"same": mono, "neg": (-(mono.astype(np.int32))).clip(-32768, 32767).astype(np.int16),
                     "rand": rng.integers(-32768, 32768, n).astype(np.int16)}
            out.append((f"{name}-{n}-mono", mono[:, None]))
            for k, o in other.items():
                out.append((f"{name}-{n}-{k}", np.stack([mono, o], 1)))
    return out


CORPUS = _corpus()
RATES = (1, 8, 4000, 8000, 11025, 22050, 44100, 48000, 65535, 65537, 96000, 192000, 655350)


def _md5(q):
    return hashlib.md5(np.ascontiguousarray(q).astype("<i2").tobytes()).digest()


def test_flac_edge_corpus(P, tmp_path):
    """Every corpus array through the C++ writer and back through soundfile
    and dr_flac, at assorted sample rates, block sizes and options."""
    sf = _sf()
    rng = np.random.default_rng(14)
    path = tmp_path / "x.flac"
    for i, (name, q) in enumerate(CORPUS):
        fs = RATES[i % len(RATES)]
        opts = dict(blocksize=int(rng.choice([16, 192, 1152, 4096, 4096, 4608, 65535])),
                    max_fixed_order=int(rng.integers(0, 5)), max_partition_order=int(rng.integers(0, 9)),
                    stereo=bool(rng.random() < 0.8), seek_every=int(rng.choice([0, 1, 16])))
        st = P._flac_encode(path, q, fs, **opts)
        assert st["frames"] == -(-len(q) // opts["blocksize"]) and st["md5"] == _md5(q), name
        assert st["bytes"] == path.stat().st_size, name
        info = P._flac_info(path)
        assert info == dict(channels=q.shape[1], sample_rate=fs, bits_per_sample=16, frames=len(q), md5=_md5(q))
        assert path.read_bytes()[26:42] == _md5(q), name     # STREAMINFO's MD5 field
        y, f = sf.read(str(path), dtype="int16", always_2d=True)
        assert f == fs and np.array_equal(y, q), (name, fs, opts)
        assert np.array_equal(P._flac_read(path), q), (name, fs, opts)
        k0 = int(rng.integers(0, len(q)))
        k1 = int(rng.integers(k0, len(q) + 1))
        assert np.array_equal(P._flac_read(path, k0, k1), q[k0:k1]), (name, k0, k1)
    # an empty stream: valid FLAC, which dr_flac reads (libsndfile refuses one)
    P._flac_encode(path, np.zeros((0, 2), np.int16), 4000)
    assert P._flac_read(path).shape == (0, 2) and P._flac_info(path)["frames"] == 0
    # soundfile's own files: what libFLAC writes, read by the C++ reader at any offset
    for i, (name, q) in enumerate(CORPUS[::3]):
        fs = (4000, 8000, 44100, 96000)[i % 4]
        sf.write(str(path), q, fs, format="FLAC", subtype="PCM_16")
        assert path.read_bytes()[26:42] == _md5(q), name     # (the MD5 above is libFLAC's)
        assert np.array_equal(P._flac_read(path), q), name
        assert P._flac_info(path) == dict(channels=q.shape[1], sample_rate=fs, bits_per_sample=16,
                                          frames=len(q), md5=_md5(q))
        for _ in range(3):
            k0 = int(rng.integers(0, len(q) + 2))
            k1 = int(rng.integers(0, len(q) + 3))
            want = sf.read(str(path), start=k0, stop=k1, dtype="int16", always_2d=True)[0]
            assert np.array_equal(P._flac_read(path, k0, k1), want), (name, k0, k1)


def test_flac_refusals(P, tmp_path):
    for bad in (b"", b"fLaC", b"RIFF" + bytes(40), bytes(100)):
        p = tmp_path / "bad.flac"
        p.write_bytes(bad)
        with pytest.raises(RuntimeError):
            P._flac_read(p)
    with pytest.raises(FileNotFoundError):
        P._flac_read(tmp_path / "missing.flac")
    q = np.zeros((10, 2), np.int16)
    for kw in (dict(fs=0), dict(fs=700000), dict(fs=4000, blocksize=15), dict(fs=4000, max_fixed_order=5),
               dict(fs=4000, max_partition_order=16)):
        with pytest.raises(ValueError):
            P._flac_encode(tmp_path / "x.flac", q, **kw)
    with pytest.raises(ValueError):
        P._flac_encode(tmp_path / "x.flac", np.zeros((10, 3), np.int16), 4000)


# --- the lock ---------------------------------------------------------------------------------------

_HOLDER = """
import sys
sys.path[:0] = [{repo!r}, {mod!r}]
impl, path = sys.argv[1], sys.argv[2]
if impl == "py":
    from sstvae.qrss.live import dir_lock
else:
    import sstvae_native
    dir_lock = sstvae_native.qrss.passband.dir_lock
f = dir_lock(path)
print("held" if f is not None else "refused", flush=True)
sys.stdin.readline()
"""


def _holder(native, impl, path):
    code = _HOLDER.format(repo=str(REPO_ROOT), mod=str(Path(native.__file__).resolve().parent))
    p = subprocess.Popen([sys.executable, "-c", code, impl, str(path)], stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE, text=True)
    return p, p.stdout.readline().strip()


def _release(p):
    p.stdin.write("\n")
    p.stdin.flush()
    assert p.wait(30) == 0


@pytest.mark.parametrize("holder", ["py", "native"])
def test_the_lock_across_processes(native, P, tmp_path, holder):
    """A listener in another process holds the directory: both
    implementations are refused; once it exits, either may take it."""
    p, said = _holder(native, holder, tmp_path / "d")
    try:
        assert said == "held"
        assert (tmp_path / "d" / ".lock").read_text() == f"{p.pid}\n"
        assert P.dir_lock(tmp_path / "d") is None
        assert PY_DIR_LOCK(tmp_path / "d") is None
    finally:
        _release(p)
    # and the other way: this process holds it, the other implementation's
    f = P.dir_lock(tmp_path / "d") if holder == "py" else PY_DIR_LOCK(tmp_path / "d")
    assert f is not None
    assert (tmp_path / "d" / ".lock").read_text() == f"{os.getpid()}\n"
    q, said = _holder(native, holder, tmp_path / "d")
    _release(q)
    assert said == "refused"
    f.close()
    q, said = _holder(native, holder, tmp_path / "d")
    _release(q)
    assert said == "held"


def test_the_lock_in_one_process(P, tmp_path):
    """Two locks in one process are two open files: they exclude each other
    too, whichever implementation took each. close() releases; so does
    dropping the object."""
    d = tmp_path / "a" / "b"            # (made, parents and all)
    f = P.dir_lock(d)
    assert f is not None and not f.closed and f.mode == "a+" and f.name == str(d / ".lock")
    assert isinstance(f.fileno(), int) and (d / ".lock").read_text() == f"{os.getpid()}\n"
    assert PY_DIR_LOCK(d) is None and P.dir_lock(d) is None
    f.close()
    assert f.closed
    f.close()                                # (twice is fine)
    g = PY_DIR_LOCK(d)
    assert g is not None and P.dir_lock(d) is None
    g.close()
    with P.dir_lock(d) as h:
        assert PY_DIR_LOCK(d) is None
    assert h.closed and PY_DIR_LOCK(d) is not None    # (that one dropped at once: free again)
    h = P.dir_lock(d)
    del h
    assert PY_DIR_LOCK(d) is not None
    (tmp_path / "file").write_text("")
    for bad in (tmp_path / "file", tmp_path / "file" / "x", tmp_path / "file" / "x" / "y"):
        g, r = _call(P.dir_lock, bad), _call(PY_DIR_LOCK, bad)
        assert g[0] == r[0] == "raise" and g[1] is r[1] and g[2] == r[2], (g, r)


# --- the whole caller -------------------------------------------------------------------------------

def _rows(native):
    from qrss_native.passband import rows

    return rows(native.qrss)


def _spy(key, repl, used):
    if isinstance(repl, type):
        class Spy(repl):
            def __init__(self, *a, **k):
                used[key[1]] = used.get(key[1], 0) + 1
                super().__init__(*a, **k)
        Spy.__name__ = Spy.__qualname__ = repl.__name__
        return Spy

    def f(*a, **k):
        used[key[1]] = used.get(key[1], 0) + 1
        return repl(*a, **k)
    return f


def _apply(native, mp, used, skip=()):
    """This file's rows over their targets and every from-import site, each a counting spy."""
    reference = qrss_native._import_reference()
    replaced = {}
    for key, repl in _rows(native).items():
        if key in skip:
            continue
        spy = _spy(key, repl, used)
        owner = qrss_native._owner(key[0])
        orig = getattr(owner, key[1])
        mp.setattr(owner, key[1], spy)
        replaced[id(orig)] = (orig, spy)
    for m in reference:
        for k, v in list(vars(m).items()):
            hit = replaced.get(id(v))
            if hit is not None and hit[0] is v:
                mp.setattr(m, k, hit[1])


def _hiss(n, seed):
    return np.random.default_rng(seed).standard_normal(n) * 0.05


def _listener_run(root: Path, mp):
    """A listener hears audio across an hour boundary into the store; the
    slot's frame is filed straight into it; a second listener backfills
    its ring from it and recovers the slot (finish_receive recording)."""
    out = SimpleNamespace(lines=[], handed=[])
    q = Q_TEST + 4                                     # 2025-10-01T01:00Z: its lead-in starts at 00:59:46
    t0 = frontend.slot_t0(q)
    dur = live.frame_seconds(frame.TINY)
    pb = frontend.PassbandStore(root / "passband")
    cfg = live.LiveConfig(spec=frame.TINY, render=False)
    L = live.LiveListener(root / "a", cfg, passband=pb, log=out.lines.append)
    t = t0 - 11.0                                      # 10 s before the hour, for 20 s
    audio = _hiss(20 * live.FS, 1)
    for k in range(0, len(audio), 2000):
        L.feed(audio[k:k + 2000], t + (k + 2000) / live.FS)
    n = int((dur + 20.0) * FE_FS)
    fe = (np.random.default_rng(2).standard_normal(n)
          + 1j * np.random.default_rng(3).standard_normal(n)).astype(np.complex64) * 0.3
    pb.write(t0 - 5.0, fe[:n // 3], expire=False)          # the frame from 5 s before its t0, a gap,
    pb.write(t0 - 5.0 + (n // 3) / FE_FS + 2.0, fe[n // 3:], expire=False)   # of 2 s, the rest

    def finish_receive(x, ok, origin, qq, spec, estimator, dets, hints=()):
        out.handed.append((x.copy(), ok.copy(), origin, qq, spec, estimator, list(dets), list(hints)))
        return [], float(np.mean(ok[origin:origin + int(dur * FE_FS)]))

    mp.setattr(live, "finish_receive", finish_receive)
    now = t0 + dur + frontend.PB_TAIL_S + 120.0
    L2 = live.LiveListener(root / "b", cfg, passband=pb, log=out.lines.append)
    out.backfill = L2.backfill(now)
    out.ring = L2.ring.read(int(now * FE_FS) - L2.ring.n, int(now * FE_FS))
    out.recover = L2.recover(now)
    L2.step(now)
    out.finished = sorted(L2.finished)
    # (each line starts with the wall-clock time; one ends with how long a receive took)
    out.lines = [s.split(" ", 1)[1].split(" (")[0] for s in out.lines]
    out.pb_type = type(pb).__name__
    return out


def _em_run(pb_root: Path, mp):
    out = SimpleNamespace(handed=[], said=[])
    pb = frontend.PassbandStore(pb_root)
    spec = frame.TINY
    q = Q_TEST + 4
    c = em._from_passband(pb, q, spec)
    out.cap = (c.fe.tobytes(), c.q, c.t0_index, c.fs)
    out.errs = []
    for qq in (None, q + 400):
        try:
            em._from_passband(pb, qq, spec)
        except ValueError as e:
            out.errs.append(str(e))
    out.passthrough = em._from_passband("not a store", q, spec)

    def template_search(store, key, cap, spec, return_stats=False, **search):
        out.handed.append((cap.fe.tobytes(), cap.q, cap.t0_index, cap.fs))
        return None, 1.5, {}

    mp.setattr(em, "template_search", template_search)
    store = SimpleNamespace(accumulator=lambda key: SimpleNamespace(uids=[]), pass_uids=lambda: [],
                            pass_info=lambda u: None)
    out.retro = em.retro_detect(store, ("AG7EW", 1), pb, spec, log=out.said.append)
    return out


def _listen_run(root: Path, mp, held_by=None):
    """qrss_listen.py's main up to the replay's missing file: its three
    locks, the store and the passband store (pack on) it opens. With
    `held_by`, another listener's lock on the passband directory."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("qrss_listen", REPO_ROOT / "qrss_listen.py")
    ql = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ql)
    seen = {}

    class Listener:
        def __init__(self, state, cfg, store=None, passband=None):
            seen.update(state=state, store=store, passband=passband)

    mp.setattr(live, "LiveListener", Listener)
    other = held_by(root / "store" / "passband") if held_by is not None else None
    try:
        with pytest.raises(SystemExit) as e:
            ql.main(["--store", str(root / "store"), "--state", str(root / "state"), "--wav",
                     str(root / "missing.wav"), "--start", "2026-10-09T05:59:49Z"])
    finally:
        if other is not None:
            other.close()
    pb = seen["passband"]
    return SimpleNamespace(
        exit=str(e.value), state=seen["state"], store=seen["store"] is not None,
        pb=None if pb is None else (type(pb).__name__, pb.root, pb.pack, pb.fs, pb.scale, pb.keep_hours),
        locks=sorted(p.relative_to(root).as_posix() for p in root.rglob(".lock")),
        meta=(root / "store" / "passband" / "meta.json").read_text() if pb is not None else None)


def _same_listener(got, ref):
    assert got.lines == ref.lines
    assert got.backfill == ref.backfill > 0 and got.recover == ref.recover != [] and got.finished == ref.finished
    _same_array(got.ring[0], ref.ring[0], "ring")
    _same_array(got.ring[1], ref.ring[1], "ring")
    assert len(got.handed) == len(ref.handed) == 1
    for a, b in zip(got.handed[0], ref.handed[0]):
        if isinstance(b, np.ndarray):
            _same_array(a, b, "handed")
        else:
            assert a == b and type(a) is type(b)


# configuration: (the rows kept the reference's, the rows the callers must reach)
CONFIGS = {
    "every row": ((), {"PassbandStore", "slot_t0", "dir_lock"}),
    "PassbandStore kept": (((FRONTEND, "PassbandStore"),), {"_merge_ranges", "slot_t0", "dir_lock"}),
}


def test_configs_reach_every_row(native, P):
    assert set().union(*(m for _k, m in CONFIGS.values())) == {k[1] for k in _rows(native)}


@pytest.mark.parametrize("config", list(CONFIGS))
def test_whole_caller(native, P, tmp_path, config):
    """The listener, em and qrss_listen.py over the store, the reference's
    against the rows' (spies installed before them): the same files, the
    same logs, the same captures handed on, every row of the configuration
    reached."""
    _sf()
    with pytest.MonkeyPatch.context() as mp:
        ref = _listener_run(tmp_path / "ref", mp)
        ref_em = _em_run(tmp_path / "ref" / "passband", mp)
        ref_listen = _listen_run(tmp_path / "ref_l", mp)
        ref_held = _listen_run(tmp_path / "ref_h", mp, held_by=P.dir_lock)
    keep, must = CONFIGS[config]
    used: dict = {}
    with pytest.MonkeyPatch.context() as mp:
        _apply(native, mp, used, skip=keep)
        got = _listener_run(tmp_path / "got", mp)
        got_em = _em_run(tmp_path / "got" / "passband", mp)
        got_listen = _listen_run(tmp_path / "got_l", mp)
        got_held = _listen_run(tmp_path / "got_h", mp, held_by=PY_DIR_LOCK)
    assert must <= set(used), (config, sorted(must - set(used)), used)
    assert got.pb_type == ref.pb_type == "PassbandStore"
    _same_listener(got, ref)
    _same_tree(tmp_path / "ref" / "passband", tmp_path / "got" / "passband")
    assert got_em.cap == ref_em.cap and got_em.errs == ref_em.errs and len(got_em.errs) == 2
    assert got_em.passthrough == ref_em.passthrough == "not a store"
    assert got_em.handed == ref_em.handed != [] and got_em.retro == ref_em.retro == []
    assert got_em.said == ref_em.said
    for g, r, base in ((got_listen, ref_listen, "_l"), (got_held, ref_held, "_h")):
        assert g.exit == r.exit.replace("ref" + base, "got" + base) and "no such file" in g.exit
        assert g.store == r.store and g.locks == r.locks
        assert g.state == tmp_path / ("got" + base) / "state"
        if r.pb is None:
            assert g.pb is None
        else:
            assert g.pb[0] == r.pb[0] and g.pb[2:] == r.pb[2:] and g.pb[1] == tmp_path / ("got" + base) / \
                "store" / "passband" and g.meta == r.meta
    assert ref_listen.pb is not None and ref_listen.pb[2] is True and ref_held.pb is None
    assert ref_listen.locks == ["state/.lock", "store/.lock", "store/passband/.lock"]

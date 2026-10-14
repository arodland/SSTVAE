"""Parity of native/core/qrss/live (sstvae_native.qrss.live) with
sstvae/qrss/live.py up to LiveListener: utc_iso, frame_seconds, in_band,
centre_of, _waveform_id, _due_waveform, FeRing, _spans, capture_read,
capture_prepare, slot_capture, finish_receive, DialLog, same_dial and
restrict, and the C++ twins of Tile and LiveConfig (which stay the
reference's dataclasses in Python). tests/qrss_native/live.py substitutes
them under --native-qrss=rx, and LiveListener, which stays the
reference's, runs on them.

The contract (the bar), and in brackets the worst measured (2026-10-14):

- the small things: the reference's value and type, or its error's type
  and text: utc_iso over the microsecond rounding, the year's range, time_t
  and gmtime's (OSError) edges, ints past 2^53 and past int64, bools and
  numpy scalars; frame_seconds of every preset of both waveforms, tuples,
  lists, nested and empty ones, duck-typed specs (ints and floats, max()
  keeping the first of equals) and native specs; in_band, centre_of,
  _waveform_id and _due_waveform over their edges and odd types (in_band
  answers a Python bool where numpy input made the reference's an np.bool_);
- FeRing (native class): bit for bit the reference's ring -- every read,
  `end`, `n` and the storage itself -- over 4000 random writes, clears and
  reads on rings of 1 to 400 samples: wraps, writes longer than the ring,
  holes past the end, writes behind the window (which land on newer
  samples' slots, as the reference's do), empty writes, negative indices,
  reads over unwritten and evicted spans, and the errors;
- _spans: the same spans (np.float64) on 3000 random masks;
- captures from rings with holes, written zeros (digital silence written
  as heard), an unheard future, evicted starts and nothing heard:
  capture_read the same (x, written, origin) bit for bit, typed (a native
  ring read in C++, a Python ring through its read) and over the Python
  objects; capture_prepare and slot_capture bit for bit with numpy's noise
  handed in, typed and not, seeds and the reference's refusal of a
  negative one; with the C++ noise (`_default_noise`), the heard samples
  and keep bit for bit and the noise of the stated distribution (KS and
  moments on 400k samples);
- DialLog (native class) on a file written as the app writes it, changed,
  with a partial last line completed later, rewritten in place, its
  signature kept (no re-read, as the reference), removed, a directory;
  bad lines of every kind (not JSON, not a dict, keys missing, strings,
  bools, NaN, non-positive and half-way dials, CRLF, \\v, \\x1c, U+2028,
  a BOM, invalid UTF-8, duplicate keys, big JSON ints) skipped or taken
  exactly as the reference; spans, by_dial, dial_at and span_at over a
  time grid with DIAL_GUARD_S changed at run time; the two refusals that
  differ (an infinite dial, which the reference raises OverflowError out
  of refresh for, and a dial past int64) pinned as deviations;
- restrict: the same mask (and `ok` itself for an unbounded span) around
  every dial's guard spans, for bool and int8 masks, and the errors;
  same_dial the same value and type;
- Tile: to_json is json.dumps(asdict(tile)) byte for byte, fields in the
  reference's order; NaN and infinities null (never a NaN token);
  from_json reads back what to_json wrote, ignores unknown keys, refuses
  what the reference's Tile(**...) would raise for; LiveConfig: the
  defaults, specs, selection and joint, and the errors;
- finish_receive's decisions, the receivers stubbed: on 300 random
  captures (lengths, origins outside the capture, masks, dets, hints, CE
  and joint) the same calls with the same arguments (the prepared capture
  bit for bit) and the same result, the share heard included (NaN for the
  empty slice the reference averages); typed, nothing is drawn for a slot
  with nothing to receive (a negative q passes, as in the reference);
- finish_receive on the real receivers, typed (asserted by a spy on the
  binding, and no per-pass binding reached through Python), against the
  reference: CE TINY and L l-tiny heard whole (the whole-slot search) and
  heard for the preamble only (the preamble's detections plus a hint):
  receive_pass's bars (test_qrss_native_receiver_slot.py), Z_ref heard in
  part to 1e-5 (_erased_same), the share heard exactly; CE SHORT and L
  l-short whole in the slow tier, timed (the numbers are printed with
  QRSS_PARITY_WORST=1 -s). [report.z_ref 7.9e-08 (ce-tiny whole), 2.4e-09
  (ce-short); z 6.1e-08; f_hz 2.3e-13 Hz; tau0 2.8e-13 CH samples; heard
  in part: z_ref 0, f_hz 6.8e-13 Hz; w, ch, psi and hdr_llr exact. C++
  against Python, one run each: ce-tiny whole 0.89 s / 1.72 s, part
  1.31 s / 2.6-3.2 s; l-tiny whole 0.75 s / 2.18 s, part 1.11 s / 3.29 s;
  ce-short 1.79 s / 4.97 s; l-short 1.68 s / 5.48 s: 1.9x to 3.3x];
- the whole caller: LiveListener (CE and L, TINY) fed audio with a
  digital silence and a stall across a slot whose dial changes and comes
  back (the dial file appended to, a partial line completed, rewritten in
  place), with hints from a store and from an earlier slot's tile, the
  receivers stubbed and the end-of-slot arguments pickled: every receiver
  call (the prepared capture bit for bit), the log, the tiles and
  state.json the reference's, with every tx and rx row in place and the
  live rows counted (every row the listener reaches reached), and again
  with spies on the binding installed before the rows (the captures typed,
  each reached row counted);
- threads: readers of a native FeRing while it is written never see a
  torn write.

The fast tier keeps all of this but the SHORT slots, the L l-tiny finish
and CE TINY heard in part (the slow tier's, `-m slow`). [fast tier: 24
tests, about 16 s.] native/tests/qrss/test_live.cpp pins the same values
without Python.
"""

import dataclasses
import functools
import hashlib
import json
import math
import os
import pickle
import re
import threading
import time
import types
import warnings
from types import SimpleNamespace

import numpy as np
import pytest
from qrss_helpers import Q_TEST
from qrss_native import _live, _rx, require

import test_qrss_native_receiver as P
import test_qrss_native_receiver_slot as S
from sstvae.qrss import frame, lframe, live
from sstvae.qrss import receiver as RX
from sstvae.qrss.constants import FE_FS, FS, WAVEFORM_CE, WAVEFORM_L
from sstvae.qrss.frontend import PB_LEAD_S, PB_TAIL_S, slot_t0

CE, L_ = WAVEFORM_CE, WAVEFORM_L
TINY, L_TINY = frame.TINY, lframe.L_TINY
JOINT = (TINY, L_TINY)
TIMES: dict = {}


@pytest.fixture(scope="module", autouse=True)
def _print_times():
    yield
    if os.environ.get("QRSS_PARITY_WORST"):
        for k in sorted(P.WORST):
            if k.startswith("slot"):
                print(f"worst {k}: {P.WORST[k]:.3g}")
        for k, (ref, cpp) in sorted(TIMES.items()):
            print(f"finish_receive {k}: Python {ref:.2f} s, C++ {cpp:.2f} s ({ref / cpp:.1f}x)")


@pytest.fixture(scope="module")
def Lv(native):
    return require(native, "live")


@pytest.fixture(scope="module")
def row(native, Lv):
    from qrss_native.live import rows

    return {name: fn for (_t, name), fn in rows(native.qrss).items()}


def _outcome(fn, *a, **k):
    """("ok", type, repr) of fn's value, or (error type, text)."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            v = fn(*a, **k)
    except Exception as e:  # noqa: BLE001 - the reference's own type and text
        return type(e), str(e)
    return "ok", type(v), repr(v)


def _digest(a):
    a = np.ascontiguousarray(a)
    return str(a.dtype), a.shape, hashlib.sha256(a.tobytes()).hexdigest()


def _same(a, b, what):
    a, b = np.asarray(a), np.asarray(b)
    assert a.dtype == b.dtype and a.shape == b.shape, (what, a.dtype, b.dtype, a.shape, b.shape)
    assert a.tobytes() == b.tobytes(), what


# --- the small things --------------------------------------------------------------------------

UTC = (0, 1, -1, 1.5, 0.5, 2.5, 0.9999995, 0.99999949, 1.0000005, -0.5, -0.0000005, -0.0000015, 1e-7,
       1760000000.4999995, 1760000000.5000005, 253402300799, 253402300799.9999995, 253402300799.99999,
       253402300800, 253402300800.0, -62135596800, -62135596800.0, -62135596801, -62135596800.5, 1e15, -1e15,
       1e17, -1e17, 1e19, 1e20, -1e20, 2 ** 53 + 1, 2 ** 62, 2 ** 63 - 1, 2 ** 63, -2 ** 63, -2 ** 63 - 1,
       math.nan, math.inf, -math.inf, True, False, np.float64(1.5), np.int64(7), np.int32(-7), np.float32(1.5),
       np.bool_(True), "1", None, 1 + 0j)


def test_utc_iso(Lv):
    for t in UTC:
        assert _outcome(Lv.utc_iso, t) == _outcome(live.utc_iso, t), t
    # the int path, through the window, and every second of a leap-year-ish walk
    rng = np.random.default_rng(3)
    for t in [*rng.integers(-62135596800, 253402300800, 300).tolist(), 951782400, 951868800, 4107542400]:
        assert Lv._utc_iso_int(t) == live.utc_iso(t) == Lv.utc_iso(float(t)), t
    for t in rng.uniform(-1e10, 1e10, 300).tolist() + (rng.integers(0, 2 ** 31, 50) + 0.5).tolist():
        assert Lv.utc_iso(t) == live.utc_iso(t), t


def _frame_cases(native):
    nf, nl = native.qrss.frame, native.qrss.lframe
    ns = SimpleNamespace
    return [TINY, frame.SHORT, frame.FULL, frame.FULL_V1, L_TINY, lframe.L_SHORT, lframe.L_FULL, JOINT,
            [L_TINY, TINY], (TINY,), (), [], ((TINY,), L_TINY), (frame.FULL, lframe.L_FULL),
            ns(keyed_end_pos=3, t_sym=0.5), ns(keyed_end_pos=np.int64(3), t_sym=np.float32(0.5)),
            (ns(keyed_end_pos=3, t_sym=1), ns(keyed_end_pos=3.0, t_sym=1)),
            (ns(keyed_end_pos=3.0, t_sym=1), ns(keyed_end_pos=3, t_sym=1)),
            ns(keyed_end_pos=math.nan, t_sym=1.0), (ns(keyed_end_pos=math.nan, t_sym=1.0), TINY),
            (TINY, ns(keyed_end_pos=math.nan, t_sym=1.0)), object(), ns(keyed_end_pos=3),
            ns(keyed_end_pos="ab", t_sym=2), nf.get("tiny"), nl.get("l-full"), (nl.get("l-tiny"), TINY),
            [nf.get("short"), nl.get("l-short")]]


def test_frame_seconds(native, Lv):
    for spec in _frame_cases(native):
        assert _outcome(Lv.frame_seconds, spec) == _outcome(live.frame_seconds, spec), spec


def test_in_band_centre_of_waveform_ids(Lv):
    fs = (299.9, 300.0, 300, 322.0, 322.06, 322.069072164948455, 322.07, 322.1, 1500.0, 2700.0, 2700, 2700.1,
          -1.0, math.nan, math.inf, -math.inf, np.float64(322.1), np.float64(299.0), np.float32(322.1),
          np.float32(322.07), np.float32(300.0), np.float32(2700.5), np.int64(2700), True)
    ws = (CE, L_, 1.0, True, "L", "CE", 2, None, np.int64(1), np.float32(1.0))
    for f in fs:
        for w in ws:
            assert _outcome(Lv.in_band, f, w) == _outcome(live.in_band, f, w), (f, w)
        assert _outcome(Lv.in_band, f) == _outcome(live.in_band, f), f
    for f in (1509.5, 1509, 0, -9.5, math.nan, math.inf, np.float32(1509.5), np.float64(1509.5), True, "x",
              None):
        for w in ("L", L_, CE, "CE", None, 1.0, True, np.int64(1), "l", 2):
            assert _outcome(Lv.centre_of, f, w) == _outcome(live.centre_of, f, w), (f, w)
            if w in (CE, "CE", None, "l", 2):
                assert Lv.centre_of(f, w) is f
    for w in ("CE", "L", "?", "", "ce", "1", 0, 1, 2, -1, 1.7, -0.5, True, np.int64(1), np.float32(1.9), None,
              math.nan, math.inf, b"L", b"1", [1], 2 ** 70):
        assert _outcome(Lv._waveform_id, w) == _outcome(live._waveform_id, w), w
    dues = [{"frame": "l-tiny"}, {"frame": "tiny"}, {"frame": "l-full"}, {"frame": "full-v1"},
            {"frame": "no-such"}, {"frame": "l-no-such"}, {"frame": "L-TINY"}, {"frame": ""},
            {"frame": "l-tiny", "waveform": 0}, {"frame": "tiny", "waveform": 1}, {"waveform": "L"},
            {"waveform": 1}, {"waveform": 1.9}, {"waveform": True}, {}, {"frame": None}, {"frame": 3},
            {"frame": ["tiny"]}, {"frame": {"a": 1}}, {"frame": "no-such", "waveform": None},
            {"frame": "no-such", "waveform": "x"}, {"frame": "no-such", "waveform": math.nan},
            {"frame": TINY}, {"frame": L_TINY}, types.MappingProxyType({"frame": "l-tiny"}),
            types.MappingProxyType({"waveform": "L"}), [("frame", "tiny")], None, "tiny"]
    for d in dues:
        assert _outcome(Lv._due_waveform, d) == _outcome(live._due_waveform, d), d
        try:
            text = json.dumps(d)
        except TypeError:
            continue
        if isinstance(d, dict) and _outcome(live._due_waveform, d)[0] == "ok":
            assert Lv._due_waveform_json(text) == live._due_waveform(d), d
    for w in ("CE", "L", "?", 0, 1, 1.9, True, 2 ** 40):
        assert Lv._waveform_id_json(json.dumps(w)) == live._waveform_id(w), w
    for w in (None, [1], math.nan, math.inf):
        assert type(_outcome(Lv._waveform_id_json, json.dumps(w))[0]) is type, w    # (raises, as int() does)


# --- FeRing and _spans --------------------------------------------------------------------------

def _ring_same(a, b, what):
    assert (a.n, a.end) == (b.n, b.end), what
    _same(a.x, b.x, (what, "x"))
    _same(a.ok, b.ok, (what, "ok"))


def test_fe_ring(Lv):
    """The ring under random writes, clears and reads: every read, end, n and
    the storage the reference's, bit for bit."""
    rng = np.random.default_rng(7)
    for span in (0.0, -1.0, math.nan, math.inf, -math.inf, 2.5e-4, 1e-3):
        assert _outcome(lambda s: Lv.FeRing(s).n, span) == _outcome(lambda s: live.FeRing(s).n, span), span
    assert Lv.FeRing().n == live.FeRing().n == int(live.RING_S * FE_FS)
    a, b = live.FeRing(0.0), Lv.FeRing(0.0)
    _same(a.read(-3, 5)[0], b.read(-3, 5)[0], "n=0")
    ops = 0
    for trial in range(40):
        n = int(rng.choice([1, 2, 7, 40, 100, 400]))
        ref, got = live.FeRing(n / FE_FS), Lv.FeRing(n / FE_FS)
        assert ref.n == got.n == n
        base = int(rng.integers(-3 * n, 1_000_000))
        for _ in range(100):
            ops += 1
            k = rng.random()
            if k < 0.55:
                n0 = base + int(rng.integers(-3 * n, 4 * n))
                m = int(rng.choice([0, 1, 2, n - 1, n, n + 1, n + 7, 3 * n, int(rng.integers(0, 2 * n + 2))]))
                m = max(m, 0)
                x = (rng.standard_normal(m) + 1j * rng.standard_normal(m)).astype(np.complex64)
                if m and rng.random() < 0.3:
                    x[rng.random(m) < 0.5] = 0           # written zeros: heard
                if rng.random() < 0.1:
                    x = x.astype(np.complex128)
                if rng.random() < 0.1:
                    n0 = np.int64(n0)
                ref.write(n0, x)
                got.write(n0, x)
                base = max(base, int(n0) + m) if rng.random() < 0.8 else base
            elif k < 0.62:
                lo = base + int(rng.integers(-3 * n, 3 * n))
                hi = lo + int(rng.integers(-2, 3 * n))
                ref._clear(lo, hi)
                got._clear(lo, hi)
            else:
                lo = base + int(rng.integers(-4 * n, 2 * n))
                hi = lo + int(rng.choice([0, 1, n, 2 * n, int(rng.integers(0, 4 * n + 1))]))
                if rng.random() < 0.05:
                    hi = lo - 1
                r, g = _outcome(ref.read, lo, hi), _outcome(got.read, lo, hi)
                if r[0] != "ok":
                    assert g == r, (trial, lo, hi)
                    continue
                rx, rok = ref.read(lo, hi)
                gx, gok = got.read(lo, hi)
                _same(gx, rx, (trial, lo, hi, "x"))
                _same(gok, rok, (trial, lo, hi, "ok"))
            _ring_same(got, ref, (trial, ops))
    assert ops >= 4000


def test_spans(Lv):
    rng = np.random.default_rng(8)
    for k in range(3000):
        n = int(rng.integers(0, 60))
        mask = rng.random(n) < rng.choice([0.0, 0.2, 0.5, 0.9, 1.0])
        origin = [0, 5, -7, 13.5, np.int64(9), np.float64(2.25), n][int(rng.integers(7))]
        fs = [FE_FS, 3.0, 1, np.float32(7.5)][int(rng.integers(4))]
        ref, got = live._spans(mask, fs, origin), Lv._spans(mask, fs, origin)
        assert got == ref, (k, mask, origin, fs)
        assert [type(v) for s in got for v in s] == [type(v) for s in ref for v in s], k
    assert Lv._spans(np.array([], bool), FE_FS, 0) == []


# --- captures ------------------------------------------------------------------------------------

def _rings(kind, n):
    """A reference and a native ring with the same writes: slot Q_TEST with
    holes, written zeros and an unheard future (kind)."""
    rng = np.random.default_rng(CAPTURE_KINDS.index(kind))
    t0 = slot_t0(Q_TEST)
    span = live.frame_seconds(JOINT) + PB_LEAD_S + PB_TAIL_S + (300.0 if kind != "evicted" else -30.0)
    ref, got = live.FeRing(span), n.FeRing(span)
    start = (t0 - 15) * FE_FS
    end = {"whole": (t0 + 60) * FE_FS, "future": (t0 + 30) * FE_FS, "evicted": (t0 + 60) * FE_FS,
           "nothing": start}[kind]
    k = start
    while k < end:
        m = int(rng.integers(1000, 30000))
        x = (0.3 * (rng.standard_normal(m) + 1j * rng.standard_normal(m))).astype(np.complex64)
        if rng.random() < 0.2:
            x[: int(rng.integers(1, m))] = 0                # digital silence, written as heard
        if rng.random() < 0.15:
            k += int(rng.integers(100, 9000))                # a hole: never written
        ref.write(k, x[: max(0, min(m, end - k))])
        got.write(k, x[: max(0, min(m, end - k))])
        k += m
    return ref, got


CAPTURE_KINDS = ("whole", "future", "evicted", "nothing")


@pytest.mark.parametrize("kind", CAPTURE_KINDS)
def test_captures(Lv, row, kind):
    """capture_read, capture_prepare and slot_capture on rings with holes,
    written zeros, an unheard future, an evicted start, nothing heard:
    typed and over the Python objects, bit for bit with numpy's noise."""
    ref_ring, ring = _rings(kind, Lv)
    for spec in (TINY, JOINT):
        for lead, tail in ((PB_LEAD_S, PB_TAIL_S), (0.0, 0.0), (3.33333, 1.25)):
            r = live.capture_read(ref_ring, Q_TEST, spec, lead, tail)
            reads = [row["capture_read"](ring, Q_TEST, spec, lead, tail),
                     Lv.capture_read(ring, Q_TEST, spec, lead, tail, False),
                     Lv.capture_read(ref_ring, Q_TEST, spec, lead, tail, True)]
            for i, g in enumerate(reads):
                _same(g[0], r[0], (kind, i, "x"))
                _same(g[1], r[1], (kind, i, "ok"))
                assert g[2] == r[2] and type(g[2]) is type(r[2]) is int, (kind, i)
        x, ok, origin = r
        if kind == "whole":
            assert ok.any() and not ok.all() and (x[ok] == 0).any()
        for seed in ((None,) if spec is TINY else (None, 5)):
            ref = live.capture_prepare(x, ok, origin, Q_TEST, spec, seed)
            gots = [row["capture_prepare"](x, ok, origin, Q_TEST, spec, seed),
                    _rx.to_ref(Lv.capture_prepare(x, ok, origin, Q_TEST, spec, seed, _live.numpy_noise, True)),
                    _rx.to_ref(Lv.capture_prepare(x, ok, origin, Q_TEST, spec, seed, _live.numpy_noise, False))]
            for i, g in enumerate(gots):
                _capture_same(g, ref, (kind, spec, seed, i))
            if seed is not (None if spec is TINY else 5):
                continue
            ref = live.slot_capture(ref_ring, Q_TEST, spec, seed)
            gots = [row["slot_capture"](ring, Q_TEST, spec, seed),           # (no rows: over the Python objects)
                    _rx.to_ref(Lv.slot_capture(ring, Q_TEST, spec, seed, PB_LEAD_S, PB_TAIL_S,
                                               _live.numpy_noise, True)),
                    _rx.to_ref(Lv.slot_capture(ref_ring, Q_TEST, spec, seed, PB_LEAD_S, PB_TAIL_S,
                                               _live.numpy_noise, True))]
            for i, g in enumerate(gots):
                _capture_same(g, ref, (kind, spec, seed, "slot", i))
        # the reference's refusal of a negative seed, whether or not anything is missing
        for q in (-3,):
            assert _outcome(Lv.capture_prepare, x, ok, origin, q, spec, None, _live.numpy_noise, True)[:2] == \
                _outcome(live.capture_prepare, x, ok, origin, q, spec)[:2]


def _capture_same(g, r, what):
    gp, rp = g[0], r[0]
    assert type(gp) is type(rp) is RX.Prepared, what
    _same(gp.fe, rp.fe, (what, "fe"))
    _same(gp.keep, rp.keep, (what, "keep"))
    for f in ("q", "t0_index", "fs"):
        assert getattr(gp, f) == getattr(rp, f) and type(getattr(gp, f)) is type(getattr(rp, f)), (what, f)
    assert g[1] == r[1] and [type(v) for s in g[1] for v in s] == [type(v) for s in r[1] for v in s], what
    assert repr(g[2]) == repr(r[2]) and type(g[2]) is type(r[2]) is float, what


def test_capture_prepare_cpp_noise(Lv):
    """With the C++ generator: the heard samples and keep bit for bit, the
    noise where nothing was heard of the stated distribution."""
    from scipy import stats

    ref_ring, ring = _rings("future", Lv)
    x, ok, origin = live.capture_read(ref_ring, Q_TEST, JOINT)
    ref = live.capture_prepare(x, ok, origin, Q_TEST, JOINT)
    got = _rx.to_ref(Lv.capture_prepare(x, ok, origin, Q_TEST, JOINT, None, None, True))
    _same(got[0].keep, ref[0].keep, "keep")
    _same(got[0].fe[ok], ref[0].fe[ok], "heard")
    assert got[1] == ref[1] and got[2] == ref[2]
    assert not np.array_equal(got[0].fe[~ok], ref[0].fe[~ok])
    s = math.sqrt(0.5 / math.log(2.0))
    for seed in (1, 2, Q_TEST):
        v = Lv._default_noise(seed, 400_000)
        assert v.dtype == np.complex64 and len(v) == 400_000
        _same(v, Lv._default_noise(seed, 400_000), "deterministic")
        _same(v[:1000], Lv._default_noise(seed, 1000), "a prefix")
        re_, im = v.real.astype(np.float64), v.imag.astype(np.float64)
        sig = s / math.sqrt(len(v))
        assert abs(re_.mean()) < 5 * sig and abs(im.mean()) < 5 * sig, seed
        assert abs(re_.var() / s ** 2 - 1) < 0.01 and abs(im.var() / s ** 2 - 1) < 0.01, seed
        assert abs(np.corrcoef(re_, im)[0, 1]) < 0.01, seed
        assert abs(np.median(np.abs(v.astype(np.complex128)) ** 2) - 1.0) < 0.01, seed
        assert stats.kstest(re_ / s, "norm").pvalue > 1e-3 and stats.kstest(im / s, "norm").pvalue > 1e-3, seed
    assert not np.array_equal(Lv._default_noise(1, 100), Lv._default_noise(2, 100))
    assert len(Lv._default_noise(1, 0)) == 0


# --- the dial --------------------------------------------------------------------------------------

def _dial_line(t, hz, **kw):
    return json.dumps({"t": t, "dial_hz": hz, **kw})


BAD_AND_ODD = [
    "not json", "[1, 2]", '"x"', "null", "{}", '{"t": 1}', '{"dial_hz": 5}', '{"t": "x", "dial_hz": 1}',
    '{"t": "12.5", "dial_hz": "14070000"}', '{"t": " 13 ", "dial_hz": "1_400"}', '{"t": true, "dial_hz": 14070000}',
    '{"t": 14, "dial_hz": false}', '{"t": NaN, "dial_hz": 1}', '{"t": Infinity, "dial_hz": 1}',
    '{"t": 15, "dial_hz": NaN}', '{"t": 16, "dial_hz": -3}', '{"t": 17, "dial_hz": 0.4}', '{"t": 18, "dial_hz": 0.5}',
    '{"t": 19, "dial_hz": 2.5}', '{"t": 20, "dial_hz": 3.5}', '{"t": 21, "dial_hz": 0}',
    '{"t": 22, "dial_hz": 14070000, "x": [1, {"y": null}]}', '{"t": 1e400, "dial_hz": 3}',
    '{"t": "inf", "dial_hz": 3}', '{"t": "nan", "dial_hz": 3}', '{"t": 23, "dial_hz": "nan"}',
    '{"t": 1, "t": 24, "dial_hz": 5}', '{"t": 1E3, "dial_hz": 7}', '{"t": -0.0, "dial_hz": 7}',
    '   {"t": 25, "dial_hz": 9}   ', "", "  ", '{"t": 26, "dial_hz": 9.0}', '{"t": 26, "dial_hz": 10}',
    '{"t": 123456789012345678901234567890, "dial_hz": 11}', '{"t": 27, "dial_hz": 123456789012}',
    '{"t": [27], "dial_hz": 5}', '{"t": {"a": 1}, "dial_hz": 5}', '{"t": 28, "dial_hz": "0x10"}',
    '{"t": 29, "dial_hz": "1e3"}', '{"t": "\\u0031\\u0032", "dial_hz": 5}', '{"t": 30, "dial_hz": 6',
    '{"t": 31, "dial_hz": 12, "seq": 123456789012345678901234567890}',
    '{"t": -123456789012345678901234567890, "dial_hz": 13}', '{"t": 32, "dial_hz": 14, "s": "-99999999999999999999"}',
    '{"t": 33, "dial_hz": 15, "a": [9223372036854775807, -9223372036854775808, 9223372036854775808]}',
    '{"t": 34.5e1, "dial_hz": 16, "b": -0, "c": 1.5e+3}',
]


def _dial_bytes(lines, sep="\n", end="\n"):
    return (sep.join(lines) + end).encode()


DIAL_FILES = {
    "lines": _dial_bytes([_dial_line(100.0, 10_141_000), *BAD_AND_ODD, _dial_line(200, 10_140_900),
                          _dial_line(150.5, 10_141_000), _dial_line(150.5, 14_070_000)]),
    "crlf": _dial_bytes([_dial_line(1, 5), _dial_line(2, 6), _dial_line(3, 6), _dial_line(4, 5)], sep="\r\n"),
    "separators": "﻿{\"t\": 1, \"dial_hz\": 5}\x0b{\"t\": 2, \"dial_hz\": 6}\x0c{\"t\": 3, \"dial_hz\": 7}\x1c"
                  "{\"t\": 4, \"dial_hz\": 8}\x1d{\"t\": 5, \"dial_hz\": 9}\x1e{\"t\": 6, \"dial_hz\": 10}\x85"
                  "{\"t\": 7, \"dial_hz\": 11} {\"t\": 8, \"dial_hz\": 12} {\"t\": 9, \"dial_hz\": 13}\r"
                  "{\"t\": 10, \"dial_hz\": 14}".encode(),
    "bad-utf8": b'{"t": 1, "dial_hz": 5}\n\xff\xfe{"t": 2, "dial_hz": 6}\n{"t": 3, "dial_hz": 7, "c": "\xe9"}\n'
                b'{"t": 4, "dial_hz": 8, "c": "\xed\xa0\x80"}\n{"t": 5, "dial\xc0\x80": 9}\n{"t": 6, "dial_hz": 10}',
    "partial": _dial_bytes([_dial_line(1, 5), _dial_line(2, 6)]) + b'{"t": 3, "dial_',
    "unsorted": _dial_bytes([_dial_line(t, hz) for t, hz in ((5, 1), (3, 2), (5, 3), (4, 2), (1, 2), (5, 1))]),
    "empty": b"",
}


def _dial_same(got, ref, what):
    assert got.points == ref.points, what
    assert [tuple(map(type, p)) for p in got.points] == [tuple(map(type, p)) for p in ref.points], what


def _queries_same(got, ref, what):
    pts = [t for t, _ in ref.points]
    grid = {-math.inf, math.inf, 0.0}
    for t in pts:
        for d in (-50.5, -50.0, -2.0, -1e-9, 0.0, 1e-9, 1.0, 2.0, 49.9, 50.0, 50.1, 300.0):
            grid.add(t + d)
    grid = sorted(grid)
    for t in grid:
        for f in ("dial_at", "span_at"):
            assert _outcome(getattr(got, f), t) == _outcome(getattr(ref, f), t), (what, f, t)
    for i, a in enumerate(grid[::4]):
        for b in grid[i::7]:
            for f in ("spans", "by_dial"):
                assert _outcome(getattr(got, f), a, b) == _outcome(getattr(ref, f), a, b), (what, f, a, b)


@pytest.mark.parametrize("guard", [None, (0.0, 0.0), (10.5, 3)], ids=["default", "none", "odd"])
def test_dial_log_files(Lv, tmp_path, monkeypatch, guard):
    """Every file above parsed as the reference parses it, and every query
    over the points it gives the same, with DIAL_GUARD_S as set at call time."""
    if guard is not None:
        monkeypatch.setattr(live, "DIAL_GUARD_S", guard)
    for name, data in DIAL_FILES.items():
        p = tmp_path / f"{name}.jsonl"
        p.write_bytes(data)
        ref, got = live.DialLog(p), Lv.DialLog(p)
        assert got.path == ref.path
        ref.refresh()
        got.refresh()
        _dial_same(got, ref, name)
        _queries_same(got, ref, name)
        assert [h for _, h in got.points] == [h for _, h in Lv._parse_dial_text(data)], name
    # points set by hand
    ref, got = live.DialLog(tmp_path / "none"), Lv.DialLog(tmp_path / "none")
    for pts in ([], [(5.0, 7)], [(5.0, 7), (60.0, 8)], [(-math.inf, 3), (0.0, 4), (1e9, 5)]):
        ref.points, got.points = list(pts), list(pts)
        _queries_same(got, ref, pts)


def test_dial_log_refreshes_on_its_signature(Lv, tmp_path):
    """The file as the app writes it: appended to, a partial last line
    completed, rewritten in place, its signature kept (not re-read, as the
    reference does not), removed, replaced by a directory."""
    p = tmp_path / live.DIAL_FILE
    ref, got = live.DialLog(p), Lv.DialLog(p)
    mtime = [1_700_000_000]

    def put(data, keep_mtime=False):
        p.write_bytes(data)
        if not keep_mtime:
            mtime[0] += 1
        os.utime(p, ns=(mtime[0] * 10 ** 9, mtime[0] * 10 ** 9))

    def check(what, want=None):
        ref.refresh()
        got.refresh()
        _dial_same(got, ref, what)
        _queries_same(got, ref, what)
        if want is not None:
            assert got.points == want, what

    check("no file", [])
    put(_dial_bytes([_dial_line(100.0, 7_040_000)]) + b'{"t": 160.0, "dial')
    check("a partial last line", [(100.0, 7_040_000)])
    put(_dial_bytes([_dial_line(100.0, 7_040_000), _dial_line(160.0, 7_041_000)]))
    check("completed", [(100.0, 7_040_000), (160.0, 7_041_000)])
    put(_dial_bytes([_dial_line(100.0, 7_040_000), _dial_line(160.0, 7_041_000), _dial_line(170.0, 7_040_000)]))
    check("appended: back to the first dial", [(100.0, 7_040_000), (160.0, 7_041_000), (170.0, 7_040_000)])
    put(_dial_bytes([_dial_line(100.0, 7_040_000), _dial_line(160.0, 7_042_000), _dial_line(170.0, 7_040_000)]))
    check("rewritten in place, the same size", [(100.0, 7_040_000), (160.0, 7_042_000), (170.0, 7_040_000)])
    put(_dial_bytes([_dial_line(100.0, 7_040_000), _dial_line(160.0, 7_043_000), _dial_line(170.0, 7_040_000)]),
        keep_mtime=True)
    check("the same signature: not read again", [(100.0, 7_040_000), (160.0, 7_042_000), (170.0, 7_040_000)])
    put(_dial_bytes([_dial_line(100.0, 7_040_000), _dial_line(160.0, 7_043_000), _dial_line(170.0, 7_040_000)]))
    check("a new mtime: read", [(100.0, 7_040_000), (160.0, 7_043_000), (170.0, 7_040_000)])
    p.unlink()
    check("removed", [])
    put(_dial_bytes([_dial_line(1.0, 5)]))
    check("back", [(1.0, 5)])
    p.unlink()
    p.mkdir()
    check("a directory: what was read is kept", [(1.0, 5)])
    p.rmdir()
    put(_dial_bytes([_dial_line(2.0, 6)]))
    check("a file again", [(2.0, 6)])
    put(_dial_bytes([_dial_line(3.0, 7)]))
    p.chmod(0)
    try:
        if not os.access(p, os.R_OK):        # (root reads it anyway)
            check("unreadable: what was read is kept", [(2.0, 6)])
    finally:
        p.chmod(0o644)


def test_dial_log_deviations(Lv, tmp_path):
    """Lines the reference does not skip, pinned: an infinite dial (or a
    time past float's range) raises OverflowError out of the reference's
    refresh (round(inf), float(10**400)), and a dial past int64 is a Python
    int there; the port skips each of those lines."""
    p = tmp_path / "dial.jsonl"
    for line, ref_points in (('{"t": 5, "dial_hz": 1e400}', OverflowError),
                             ('{"t": 5, "dial_hz": "-Infinity"}', OverflowError),
                             ('{"t": 1' + "0" * 400 + ', "dial_hz": 5}', OverflowError),
                             ('{"t": 5, "dial_hz": 1e30}', [(1.0, 3), (5.0, int(round(1e30)))]),
                             ('{"t": 5, "dial_hz": 123456789012345678901234567890}',
                              [(1.0, 3), (5.0, int(round(float(123456789012345678901234567890))))]),
                             ('{"t": 5, "dial_hz": 99999999999999999999.5}', [(1.0, 3), (5.0, 10 ** 20)])):
        p.write_bytes(_dial_bytes([_dial_line(1.0, 3), line]))
        ref, got = live.DialLog(p), Lv.DialLog(p)
        if ref_points is OverflowError:
            with pytest.raises(OverflowError):
                ref.refresh()
        else:
            ref.refresh()
            assert ref.points == ref_points
        got.refresh()
        assert got.points == [(1.0, 3)], line


# --- restrict and same_dial ------------------------------------------------------------------------

def test_restrict(Lv, tmp_path, monkeypatch):
    """restrict around every dial's guard spans (DialLog.by_dial on a slot
    whose dial changes and comes back), random spans, both mask types."""
    t0 = slot_t0(Q_TEST)
    monkeypatch.setattr(live, "DIAL_GUARD_S", (2.0, 5.0))
    p = tmp_path / live.DIAL_FILE
    p.write_bytes(_dial_bytes([_dial_line(t0 - 600.0, 5), _dial_line(t0 + 20.0, 6), _dial_line(t0 + 40.0, 5),
                               _dial_line(t0 + 60.0, 7), _dial_line(t0 + 60.1, 8)]))
    d = live.DialLog(p)
    d.refresh()
    rng = np.random.default_rng(9)
    n = int((live.frame_seconds(JOINT) + PB_LEAD_S + PB_TAIL_S) * FE_FS)
    origin = int(PB_LEAD_S * FE_FS)
    ok = rng.random(n) < 0.9
    span_sets = [s for _, s in d.by_dial(t0 - PB_LEAD_S, t0 + 80.0)]
    span_sets += [[], [(-math.inf, math.inf)], [(t0, t0 + 1), (-math.inf, math.inf)], [(-math.inf, t0 + 3.3)],
                  [(t0 + 10.00001, math.inf)], [(t0 + 5, t0 + 2)], [(t0 - 100, t0 - 50)], [(t0 + 100, t0 + 200)],
                  [(t0 + 1, t0 + 9), (t0 + 5, t0 + 15), (t0 - 20, t0 - 14.9999)], [(np.float64(t0 + 1), t0 + 2)],
                  [(t0 + 1, t0 + 2)], [[t0 + 3, t0 + 4]], ((t0 + 3.00025, t0 + 4.000125),)]
    for _ in range(200):
        k = int(rng.integers(1, 4))
        lo = t0 + rng.uniform(-30, 70, k)
        span_sets.append([(float(a), float(a + rng.uniform(-2, 30))) for a in lo])
    assert len(span_sets[0]) == 2          # the first dial left and come back to: two spans
    for spans in span_sets:
        for mask in (ok, ok.astype(np.int8)):
            r, g = live.restrict(mask, origin, Q_TEST, spans), Lv.restrict(mask, origin, Q_TEST, spans)
            if r is mask:
                assert g is mask, spans
            _same(g, r, spans)
    for spans in ([(math.nan, t0)], [(t0, math.nan)], [(math.inf, t0)], [(t0, -math.inf)], [(t0,)], [(t0, 1, 2)],
                  [None], 5, [(-math.inf, math.inf, 3)], [(t0, t0 + 1), (-math.inf, math.inf, 3)]):
        assert _outcome(Lv.restrict, ok, origin, Q_TEST, spans) == _outcome(live.restrict, ok, origin, Q_TEST, spans), \
            spans
    for spans in ([("a", "b")], [(t0, "b")]):
        assert _outcome(Lv.restrict, ok, origin, Q_TEST, spans)[0] is _outcome(live.restrict, ok, origin, Q_TEST,
                                                                                spans)[0] is TypeError, spans
    # (the reference iterates `spans` twice: a generator is spent by its any(),
    # and the mask comes back empty; the listener hands it lists, and the port
    # takes the spans as they come)
    assert not live.restrict(ok, origin, Q_TEST, (s for s in [(t0, t0 + 1)])).any()
    _same(Lv.restrict(ok, origin, Q_TEST, (s for s in [(t0, t0 + 1)])), live.restrict(ok, origin, Q_TEST,
                                                                                      [(t0, t0 + 1)]), "generator")


def test_same_dial(Lv):
    cases = [(None, 5), (5, None), (None, None), (5, 5), (5, 6), (2 ** 70, 2 ** 70), (2 ** 70, 2 ** 70 + 1),
             (5, 5.0), (np.int64(5), 5), (5, np.int64(6)), (True, 1), ("a", "a"), (-(2 ** 63), -(2 ** 63))]
    for a, b in cases:
        assert _outcome(Lv.same_dial, a, b) == _outcome(live.same_dial, a, b), (a, b)


# --- Tile, LiveConfig --------------------------------------------------------------------------------

def _tiles():
    t = live.Tile
    full = t(id="1954752-1500", q=Q_TEST, slot_utc="2025-10-01T00:00:00Z", frame="l-tiny", f_hz=1509.5,
             dial_hz=10_140_000, status="complete", snr_db=-12.25, z_ref=17.5, callsign="K1ABC/P",
             grid="FN42", picture_id="1234abcd", mode="B", segment=1, cw_text="K1ABC", progress=1.0, heard=0.875,
             received=0.5, mean_w_db=-3.125, confidence=[None, -1.5, 2.0, None, 1e-300, 0.1], passes=2,
             image="tiles/x.png", image_rev=3, updated=1759276800.123, note="stored (corr)", uid="u_1",
             waveform="L")
    yield t(id="a", q=1, slot_utc="", frame="tiny", f_hz=1500.0)
    yield full
    yield dataclasses.replace(full, note='quote " backslash \\ tab \t nl \n ctl \x01 \x7f é ü 日本 \U0001f600',
                              callsign="", grid=None, dial_hz=None, q=-5, f_hz=-0.0, updated=0.1 + 0.2,
                              progress=1 / 3, heard=2 ** -1074, received=1e16, mean_w_db=1e-7, snr_db=123456789.0)
    yield dataclasses.replace(full, dial_hz=2 ** 62, segment=0, passes=0, image_rev=2 ** 53 + 1, confidence=[])


def _nonfinite_tiles():
    full = list(_tiles())[1]
    yield dataclasses.replace(full, snr_db=math.nan, z_ref=math.inf)
    yield dataclasses.replace(full, f_hz=math.nan, progress=-math.inf, heard=math.inf, received=math.nan,
                              mean_w_db=-math.inf, updated=math.nan, confidence=[math.nan, None, math.inf, 1.0])


def _nulled(v):
    if isinstance(v, float) and not math.isfinite(v):
        return None
    if isinstance(v, list):
        return [_nulled(x) for x in v]
    if isinstance(v, dict):
        return {k: _nulled(x) for k, x in v.items()}
    return v


def test_tile_json(Lv):
    names = [f.name for f in dataclasses.fields(live.Tile)]
    for t in _tiles():
        text = Lv._tile_json(t)
        assert text == json.dumps(t.to_json()), t
        assert list(json.loads(text)) == names
        assert Lv._tile_roundtrip(text) == text
        assert live.Tile(**json.loads(text)) == t
        assert Lv._tile_roundtrip(json.dumps({**t.to_json(), "extra": [1, 2], "zz": None})) == text
        d = t.to_json()
        assert Lv._tile_roundtrip(json.dumps({k: d[k] for k in ("id", "q", "slot_utc", "frame", "f_hz")})) == \
            json.dumps(live.Tile(**{k: d[k] for k in ("id", "q", "slot_utc", "frame", "f_hz")}).to_json())
        for k in ("id", "q", "slot_utc", "frame", "f_hz"):
            short = {x: v for x, v in d.items() if x != k}
            with pytest.raises(TypeError):
                live.Tile(**short)
            assert Lv._tile_roundtrip(json.dumps(short)) is None, k
        for k, bad in (("q", "1"), ("f_hz", "1500"), ("confidence", 3), ("status", 5), ("dial_hz", 1.5),
                       ("passes", None)):
            assert Lv._tile_roundtrip(json.dumps({**d, k: bad})) is None, k
        assert json.loads(Lv._tile_roundtrip(json.dumps({**d, "f_hz": 1500, "progress": 1}))) == \
            {**d, "f_hz": 1500.0, "progress": 1.0}
    for t in _nonfinite_tiles():
        text = Lv._tile_json(t)
        assert "NaN" not in text and "Infinity" not in text, text
        assert json.loads(text) == _nulled(t.to_json())
        assert list(json.loads(text)) == names
        ref = json.dumps(t.to_json())
        assert "NaN" in ref or "Infinity" in ref       # (the reference's own would carry the tokens)


def test_live_config(native, Lv):
    defaults = dataclasses.asdict(live.LiveConfig())
    defaults["spec"] = live.LiveConfig().spec.name
    assert Lv._live_config_defaults() == defaults
    assert set(Lv._live_config_defaults()) == {f.name for f in dataclasses.fields(live.LiveConfig)}
    specs = (TINY, frame.SHORT, frame.FULL, frame.FULL_V1, L_TINY, lframe.L_FULL)
    for spec in specs:
        for ws in (None, (CE,), (L_,), (CE, L_), (L_, CE), (L_, L_, CE), (), (2,), (CE, 2), (-1,), [L_], (True,)):
            c = live.LiveConfig(spec=spec, waveforms=ws)

            def ref(c=c):
                sel = c.selection
                return ([s.name for s in c.specs], [s.name for s in (sel if isinstance(sel, tuple) else (sel,))],
                        RX.is_joint(sel))

            r, g = _outcome(ref), _outcome(Lv._live_config, spec, ws)
            if isinstance(ws, list) and r[0] != "ok":
                assert g[0] is r[0], (spec, ws)        # (the port shows the waveforms as a tuple)
            else:
                assert g == r, (spec.name, ws)


# --- finish_receive: its decisions, the receivers stubbed ----------------------------------------

class _Det:
    def __init__(self, f, waveform=CE, z=10.0, method="preamble"):
        self.f_hz, self.waveform, self.z_ref, self.method = f, waveform, z, method

    def __repr__(self):
        return f"_Det({self.f_hz!r}, {self.waveform!r}, {self.z_ref!r}, {self.method!r})"


class _Report:
    def __init__(self, z, snr=-12.0):
        self.snr2500_db, self.z_ref, self.suspect = snr, z, False


class _Pass:
    def __init__(self, f, waveform=CE, z=20.0, header=None, snr=-12.0):
        self.f_hz, self.waveform, self.report = f, waveform, _Report(z, snr)
        self.frame = "l-tiny" if waveform == L_ else "tiny"
        self.header, self.cw, self.uid = header, None, f"u{f}"
        self.q = Q_TEST
        self.w = np.ones(TINY.n_data, dtype=np.float32)
        self.z = np.zeros(TINY.n_data, dtype=np.float32)

    def __repr__(self):
        return f"_Pass({self.f_hz!r}, {self.waveform!r}, {self.report.z_ref!r})"


def _prep_key(prep):
    return (type(prep).__name__, _digest(prep.fe), _digest(prep.keep), prep.q, type(prep.q).__name__,
            repr(prep.t0_index), type(prep.t0_index).__name__, prep.fs)


def _spec_key(spec):
    return tuple(s.name for s in spec) if isinstance(spec, tuple) else spec.name


def _val(v):
    """A comparable record of an argument a stub was handed."""
    if isinstance(v, (list, tuple)):
        return type(v).__name__, tuple(_val(x) for x in v)
    if isinstance(v, (_Det, _Pass)):
        return type(v).__name__, repr(v.f_hz), v.waveform
    if isinstance(v, np.ndarray):
        return _digest(v)
    return type(v).__name__, repr(v)


class _Stubs:
    """Receivers that record what they are handed and answer from it."""

    def __init__(self):
        self.calls = []

    def candidates(self, prep, spec, live_only=False, hints=()):
        self.calls.append(("candidates", _prep_key(prep), _spec_key(spec), live_only, _val(hints)))
        return [_Det(1500.0, CE, 9.0), _Det(1709.5, L_, 12.0), _Det(2004.0, CE, 3.0, "hint")]

    def detect(self, prep, spec, live_only=False, **kw):
        self.calls.append(("detect", _prep_key(prep), _spec_key(spec), live_only,
                           tuple(sorted((k, _val(v)) for k, v in kw.items()))))
        cands = kw.get("cands")
        if cands is not None:
            return [c for c in cands if c.f_hz < 1900.0]
        return [_Det(float(h[0] if isinstance(h, tuple) else h) + 0.25, CE, 4.0, "hint") for h in kw.get("hints", ())]

    def receive_slot(self, prep, spec, live_only=False, estimator="joint", **kw):
        self.calls.append(("receive_slot", _prep_key(prep), _spec_key(spec), live_only, estimator,
                           tuple(sorted((k, _val(v)) for k, v in kw.items()))))
        dets = kw.get("dets")
        if dets is None:
            dets = [_Det(1500.0), _Det(1709.5, L_)]
        out = []
        for i, d in enumerate(dets):
            hdr = None
            if d.waveform == L_:
                hdr = SimpleNamespace(callsign="K1ABC", grid="FN42", picture_id=0x1234ABCD, mode=1, segment=0)
            out.append(_Pass(d.f_hz + 0.125, d.waveform, z=20.0 + i + len(self.calls), header=hdr))
        return out

    def confidence_db(self, p, spec, elapsed_s=None, n=RX.CONF_BINS):
        self.calls.append(("confidence_db", p.f_hz, _spec_key(spec), repr(elapsed_s)))
        return [None] * (n - 2) + [1.5, -0.25]

    def install(self, mp, names=("candidates", "detect", "receive_slot", "confidence_db")):
        for name in names:
            mp.setattr(RX, name, getattr(self, name))


def _finish_cases():
    rng = np.random.default_rng(12)
    hint_sets = ((), [], [(1560.0, CE)], [1560.0, 1800.25], [(1600.0, L_), (1400.0, CE)], ((1560.0, CE),))
    det_sets = ((), [], [_Det(1500.0)], [_Det(1509.5, L_), _Det(1700.0)])
    for k in range(300):
        n = int(rng.choice([0, 1, 50, 2000, 4000, 9000]))
        origin = int(rng.choice([0, 5, n // 2, n, n + 3, -3, -n - 5, int(rng.integers(-n - 10, n + 10))]))
        frac = rng.choice([0.0, 0.1, 0.49, 0.5, 0.51, 0.9, 1.0])
        ok = rng.random(n) < frac
        if rng.random() < 0.3 and n:
            ok[: int(rng.integers(0, n))] = rng.random() < 0.5
        x = (rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(np.complex64)
        spec = JOINT if rng.random() < 0.5 else TINY
        dets = det_sets[int(rng.integers(len(det_sets)))]
        hints = hint_sets[int(rng.integers(len(hint_sets)))]
        yield k, x, ok, origin, spec, dets, hints


def test_finish_receive_stubbed(Lv, row, monkeypatch):
    """The reference's calls, arguments and result on 300 random captures,
    the receivers stubbed (so the row goes over the Python objects): the
    gate, the share heard, and which receive runs with what."""
    seen = {"search": 0, "known": 0, "detect": 0, "nothing": 0, "nan": 0}
    for k, x, ok, origin, spec, dets, hints in _finish_cases():
        out = []
        for fn in (live.finish_receive, row["finish_receive"],
                   lambda *a: _rx.to_ref(Lv.finish_receive(*a, _live.numpy_noise, False))):
            stubs = _Stubs()
            with pytest.MonkeyPatch.context() as mp:
                stubs.install(mp, ("detect", "receive_slot"))
                r = _outcome(fn, x, ok, origin, Q_TEST, spec, "joint", dets, hints)
            out.append((r, stubs.calls))
        assert out[1] == out[0] and out[2] == out[0], (k, origin, len(x), dets, hints, out)
        names = [c[0] for c in out[0][1]]
        if "max_candidates" in str(out[0][1][-1:]) and names == ["receive_slot"]:
            seen["search"] += 1
        elif names:
            seen["known"] += 1
            seen["detect"] += "detect" in names
        else:
            seen["nothing"] += 1
            seen["nan"] += out[0][0][0] == "ok" and "nan" in out[0][0][2]
    assert min(seen.values()) >= 5, seen


def test_finish_receive_typed_draws_no_noise_for_nothing(Lv):
    """Typed, the noise is drawn only for a slot it receives: with nothing
    heard a negative q passes, as in the reference (whose generator is
    made in capture_prepare)."""
    n = 30000
    x = np.zeros(n, np.complex64)
    for ok, hints in ((np.zeros(n, bool), [1500.0]), (np.r_[np.ones(4100, bool), np.zeros(n - 4100, bool)], [])):
        ref = live.finish_receive(x, ok, 4000, -5, TINY, "joint", [], hints)
        got = Lv.finish_receive(x, ok, 4000, -5, TINY, "joint", [], hints, _live.numpy_noise, True)
        assert got == tuple(ref), (got, ref)
    with pytest.raises(ValueError):           # ...and a slot it receives draws, as the reference does
        live.finish_receive(x, np.ones(n, bool), 4000, -5, TINY, "joint", [], [])


# --- finish_receive on the real receivers ------------------------------------------------------------

FINISH = {"ce-tiny": TINY, "l-tiny": L_TINY, "ce-short": frame.SHORT, "l-short": lframe.L_SHORT}


@functools.lru_cache(maxsize=None)
def _finish_ref(label, part):
    """The reference's finish_receive on the capture heard whole, or heard
    for the preamble's first 20 s with the preamble's detections and a hint
    (call it before any row is applied)."""
    cap = S._cap(label)
    x = np.asarray(cap.fe, np.complex64)
    origin = int(cap.t0_index)
    spec = FINISH[label]
    ok = np.ones(len(x), bool)
    dets, hints = [], []
    if part:
        ok[origin + 20 * FE_FS:] = False
        dets = list(S._reference(label).detect[2])
        assert dets, label
        hints = [dets[0].f_hz + 60.0] if spec.waveform == CE else [(dets[0].f_hz + 60.0, L_)]
    t = time.perf_counter()
    passes, heard = live.finish_receive(x, ok, origin, int(cap.q), spec, "joint", dets, hints)
    return SimpleNamespace(x=x, ok=ok, origin=origin, q=int(cap.q), spec=spec, dets=dets, hints=hints,
                           passes=passes, heard=heard, t=time.perf_counter() - t)


def _finish_typed(native, label, part):
    r = _finish_ref(label, part)
    seen, below = [], []
    with pytest.MonkeyPatch.context() as mp:
        P._apply_rows(native, mp)
        n = native.qrss.live
        orig = n.finish_receive

        def spy(*a):
            seen.append(a[-1])
            return orig(*a)
        mp.setattr(n, "finish_receive", spy)
        for mod, name in ((native.qrss.receiver_slot, "receive_slot"), (native.qrss.receiver_slot, "detect"),
                          (native.qrss.receiver, "prepare"), (native.qrss.receiver, "receive_pass")):
            o = getattr(mod, name)

            def spy2(*a, _o=o, _n=name, **k):
                below.append(_n)
                return _o(*a, **k)
            mp.setattr(mod, name, spy2)
        t = time.perf_counter()
        passes, heard = live.finish_receive(r.x, r.ok, r.origin, r.q, r.spec, "joint", r.dets, r.hints)
        dt = time.perf_counter() - t
    assert seen == [True] and not below, (label, part, seen, below)
    assert heard == r.heard and type(heard) is float
    if part:
        assert 0.0 < heard < live.FINISH_SEARCH_MIN_HEARD
        S._erased_same(passes, r.passes, (label, "part"))
    else:
        assert heard == 1.0
        S._slot_close(passes, r.passes, (label, "whole"))
    assert r.passes, (label, part)
    TIMES[f"{label} {'part' if part else 'whole'}"] = (r.t, dt)
    return passes


@pytest.mark.parametrize("label,part", [("ce-tiny", False), pytest.param("ce-tiny", True, marks=pytest.mark.slow),
                                        pytest.param("l-tiny", False, marks=pytest.mark.slow),
                                        pytest.param("l-tiny", True, marks=pytest.mark.slow),
                                        pytest.param("ce-short", False, marks=pytest.mark.slow),
                                        pytest.param("l-short", False, marks=pytest.mark.slow)])
def test_finish_receive_typed(native, Lv, label, part):
    """finish_receive in C++ whole (typed, asserted) on the real receivers:
    the reference's passes to receive_pass's bars and its share heard."""
    _finish_typed(native, label, part)


# --- the whole caller: LiveListener -------------------------------------------------------------------

class _InProcess:
    """`_FinishProcess` without the process: the arguments go through pickle."""

    def call(self, args):
        return live.finish_receive(*pickle.loads(pickle.dumps(args)))

    def close(self):
        pass


class _Store:
    """store.due's records, and nothing else (associate's calls fail, and
    the tile notes it, as the listener does with a store that fails)."""

    def __init__(self, rows):
        self.rows = rows

    def due(self, q):
        return [dict(r) for r in self.rows]


LIVE_ROWS = ("utc_iso", "frame_seconds", "in_band", "centre_of", "_waveform_id", "_due_waveform", "FeRing",
             "capture_read", "capture_prepare", "finish_receive", "DialLog", "same_dial", "restrict")
_TIMING = re.compile(r"\(\d+ s\)")


def _hiss(n, seed):
    return 1e-3 * np.random.default_rng(seed).standard_normal(n)


def _run_listener(tmp, stubs, mp):
    """A TINY slot heard as audio across its end, CE and L listened for:
    the dial changes at t0+20 and comes back at t0+40 (DIAL_GUARD_S 2 s
    and 5 s), the file growing as the app writes it; 1.5 s of digital
    silence at t0+8 and a 2.5 s stall at t0+28; hints from the store and
    from a tile of slot q-2. Returns what the listener did."""
    q = Q_TEST
    t0 = slot_t0(q)
    A, B = 10_141_000, 10_140_900
    tmp.mkdir(parents=True, exist_ok=True)
    mp.setattr(live, "DIAL_GUARD_S", (2.0, 5.0))
    dial = tmp / live.DIAL_FILE
    stamp = [1_759_000_000]

    def put(lines, tail=b""):
        dial.write_bytes(_dial_bytes(lines) + tail)
        stamp[0] += 1
        os.utime(dial, ns=(stamp[0] * 10 ** 9, stamp[0] * 10 ** 9))

    put([_dial_line(t0 - 600.0, A)], b'{"t": ' + str(t0 + 20.0).encode())
    store = _Store([dict(f_hz=1205.47, q=q - 2, waveform=L_, frame="l-tiny"), dict(f_hz=1700.0, q=q - 2),
                    dict(f_hz=2210.0, q=q - 2, frame="tiny", waveform="L"), dict(f_hz=320.0, q=q, waveform=L_)])
    cfg = live.LiveConfig(spec=TINY, waveforms=(CE, L_), render=False, refresh_s=10.0)
    stubs.install(mp)
    lst = live.LiveListener(tmp, cfg, store=store, log=lambda m: None)
    lst._proc = _InProcess()
    lst.done_tiles.append(live.Tile(id="old", q=q - 2, slot_utc="", frame="l-tiny", f_hz=2509.5,
                                    status="complete", mode="B", segment=0, waveform="L", dial_hz=A,
                                    updated=t0 - 1800.0, callsign="K9XYZ"))
    lst.now = t0 - 15.0
    end = t0 + lst.dur + PB_TAIL_S + 0.5
    k = 0

    def feed_until(t):
        nonlocal k
        start = t0 - 15.0
        while start + (k + 1) * 2.0 <= t:
            a = start + k * 2.0
            x = _hiss(int(2.0 * FS), k)
            if t0 + 8.0 <= a < t0 + 10.0:
                x[2000:14000] = 0.0                 # 1.5 s of digital silence
            if not (t0 + 27.0 <= a < t0 + 29.0):    # a stall: 2 s never arrive
                lst.feed(x, a + 2.0 - 1 / FS)
            k += 1

    for t, write in ((t0 + 25.0, lambda: put([_dial_line(t0 - 600.0, A), _dial_line(t0 + 20.0, B)],
                                             b'{"t": ' + str(t0 + 40.0).encode() + b', "dial_hz": ')),
                     (t0 + 35.0, None),
                     (t0 + 45.0, lambda: put([_dial_line(t0 - 600.0, A), _dial_line(t0 + 20.0, B),
                                              _dial_line(t0 + 40.0, A)])),
                     (t0 + 52.0, lambda: put([_dial_line(t0 - 600.0, A), _dial_line(t0 + 20.0, B + 1),
                                              _dial_line(t0 + 40.0, A)])),
                     (end, None), (end + 30.0, None)):
        feed_until(t)
        if write is not None:
            write()
        lst.step(t)
    lst.write_state()
    st = json.loads((tmp / "state.json").read_text())
    st.pop("updated")
    st["log"] = [_TIMING.sub("(_ s)", s) for s in st["log"]]
    lines = [_TIMING.sub("(_ s)", s) for s in lst.lines]
    return SimpleNamespace(calls=stubs.calls, lines=lines, state=st,
                           tiles=[dataclasses.asdict(t) for t in lst.tiles()])


def _listener_same(got, ref, what):
    assert got.lines == ref.lines, what
    assert len(got.calls) == len(ref.calls), what
    for i, (g, r) in enumerate(zip(got.calls, ref.calls)):
        assert g == r, (what, i, g, r)
    assert got.tiles == ref.tiles, what
    assert got.state == ref.state, what


def test_whole_caller_live_listener(native, Lv, tmp_path):
    """LiveListener on the rows (module docstring): the reference's every
    receiver call, log line, tile and state.json, with every tx and rx row
    in place and the live rows counted (each one the listener reaches
    reached), then with spies on the binding installed before the rows
    (captures typed; each reached row counted)."""
    with pytest.MonkeyPatch.context() as mp:
        ref = _run_listener(tmp_path / "ref", _Stubs(), mp)
    names = [c[0] for c in ref.calls]
    assert {"candidates", "detect", "receive_slot", "confidence_db"} <= set(names), names
    assert sum(1 for c in ref.calls if c[0] == "receive_slot" and "max_candidates" not in str(c)) >= 3
    assert any("digital silence" in s for s in ref.lines) and any("missing" in s for s in ref.lines)
    assert any("a later pass is due" in s for s in ref.lines), ref.lines
    assert ref.state["tiles"] and any(t["status"] == "complete" for t in ref.state["tiles"])
    # 1. every row in place, the live rows counted (which sends the captures
    # and the finish over the Python objects)
    used: dict = {}

    def wrap(key, fn):
        if key[0] != "sstvae.qrss.live":
            return fn

        def counted(*a, **k):
            used[key[1]] = used.get(key[1], 0) + 1
            return fn(*a, **k)
        return counted
    with pytest.MonkeyPatch.context() as mp:
        P._apply_rows(native, mp, wrap=wrap)
        got = _run_listener(tmp_path / "counted", _Stubs(), mp)
    _listener_same(got, ref, "counted")
    assert all(used.get(k) for k in LIVE_ROWS), {k: used.get(k) for k in LIVE_ROWS}
    # 2. spies on the binding installed before the rows: they are what the
    # rows hold; the captures run typed, the finish over the Python objects
    # (its receivers are stubs)
    calls: dict = {}
    flags: dict = {}
    with pytest.MonkeyPatch.context() as mp:
        n = native.qrss.live
        for name in (*LIVE_ROWS, "_spans", "slot_capture"):
            orig = getattr(n, name)

            def spy(*a, _o=orig, _n=name, **k):
                calls[_n] = calls.get(_n, 0) + 1
                if _n in ("capture_read", "capture_prepare", "finish_receive"):
                    flags.setdefault(_n, set()).add(a[-1])
                return _o(*a, **k)
            mp.setattr(n, name, spy)
        P._apply_rows(native, mp)
        got = _run_listener(tmp_path / "spied", _Stubs(), mp)
    _listener_same(got, ref, "spied")
    assert all(calls.get(k) for k in LIVE_ROWS), {k: calls.get(k) for k in LIVE_ROWS}
    assert flags == {"capture_read": {True}, "capture_prepare": {True}, "finish_receive": {False}}, flags


# --- threads ------------------------------------------------------------------------------------------

def test_fe_ring_threads(Lv):
    """Readers of a native ring while one thread writes it: each read sees
    whole writes only (the written prefix a multiple of the chunk, every
    sample its chunk's value)."""
    chunk, chunks = 1000, 400
    ring = Lv.FeRing(chunk * chunks / FE_FS)
    stop = threading.Event()
    bad = []

    def reader():
        while not stop.is_set():
            x, ok = ring.read(0, chunk * chunks)
            k = int(ok.sum())
            if k % chunk or not ok[:k].all() or ok[k:].any():
                bad.append(("torn", k))
            elif k and not np.array_equal(x[:k].real, np.repeat(np.arange(1, k // chunk + 1, dtype=np.float32), chunk)):
                bad.append(("values", k))

    threads = [threading.Thread(target=reader) for _ in range(3)]
    for t in threads:
        t.start()
    for j in range(chunks):
        ring.write(j * chunk, np.full(chunk, j + 1, np.complex64))
    stop.set()
    for t in threads:
        t.join()
    assert not bad, bad[:5]
    assert ring.end == chunk * chunks

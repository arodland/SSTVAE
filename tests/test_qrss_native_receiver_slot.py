"""Parity of native/core/qrss/receiver_slot (sstvae_native.qrss.receiver_slot)
with the slot half of sstvae/qrss/receiver.py: candidates, detect,
_detect_joint, _gate, _ce_resid_ratio, receive_slot, _receive_slot_joint,
capture_from_wav, receive_wav, and the rules they apply (_aliases,
_ghosts, _ghost_of, _suspect_neighbour, _suspect_beside, _a3_lines,
_owned). tests/qrss_native/receiver_slot.py substitutes them under
--native-qrss=rx.

The orchestration runs two ways (the rows file's docstring): in C++ whole
("typed") when every stage it reaches is the native row, and otherwise
over the Python objects, every stage resolved through receiver's globals.
Both are tested: the scenarios typed (asserted, by a spy on the binding),
and the stubbed fuzz and the whole callers the other way, their spies
being what sends them there.

The contract (the bar), and in brackets the worst measured over every
scenario, slow tier included (2026-10-13; QRSS_PARITY_WORST=1 with -s
prints them):

- the rules: exactly the reference's -- the same objects in the same
  order (_aliases, _ghosts: by identity, ties in Z_ref kept in input
  order), the same booleans (_ghost_of, _owned with QRSS_OLD_GATING set
  and not, _suspect_neighbour, _suspect_beside) and tuples (_a3_lines),
  and the same error type and text where a detection has no timing --
  on 600 random sets of detections around one frequency (k/T aliases at
  either waveform's symbol, ghosts' windows, the L band's edges, Z_ref
  ties, nan and infinite a3_z);
- detect and _detect_joint (typed), as each scenario's receive_slot
  called it: the same detections in the same order, method and waveform
  exactly, f_hz to 1e-6 Hz [0], tau0 to 1e-6 s [4.6e-12 CH samples],
  z_ref, ppm and the timing Z to 1e-6 relative [z_ref 1.1e-7, ppm
  6.0e-10], lead_in_s to 1e-9 s [0], a3_z the same (nan with nan), the
  frequency path to 1e-6 of its scale [3e-16];
- receive_slot and _receive_slot_joint (typed): the same passes in the
  same order, each to receive_pass's bars (test_qrss_native_receiver.py:
  header exact, cw exact with z_match to 1e-6 [0], f_hz to 1e-6 Hz
  [1.8e-12], tau0 to 1e-6 s [2.4e-12 CH samples], z and w to 1e-6 of the
  scale and element by element [z 9.4e-10 and 7.1e-8, w 0], the report to
  1e-6 [z_ref 1.1e-7, the rest under 1e-10]). Scenarios: CE TINY, L
  l-tiny, joint (an L l-tiny with a CE TINY 60 Hz above), CE beside L
  (that capture received as CE alone), a symbol-rate alias (a -22 dB
  neighbour 1/T above a +10 dB TINY: the alias rule drops it), a weak
  neighbour (-10 dB, 60 Hz above), noise, the old gating rule (a -15 dB
  neighbour 12 Hz above, QRSS_OLD_GATING set: more candidates verified,
  the same signals found), the old edge noise rule (a TINY at 350 Hz,
  QRSS_OLD_EDGE_NOISE set: the SNR it reports moves by 0.17 dB, and the
  C++ moves with it), max_candidates, erase_s, the plain estimator
  without round B, detections handed in (one of them twice); in the slow
  tier CE SHORT, L l-short and joint (SHORT, L_SHORT), headers and
  callsigns read, and live_only, a joint max_candidates and hints;
- the two ways agree bit for bit: with a pass-through spy on track.verify
  (which sends the binding over the Python objects) the joint, the old
  gating and the joint SHORT scenarios give the typed run's passes and
  detections exactly;
- the stubbed orchestration (2000 seeds, 49,000 stage calls): every stage
  the slot receiver reaches a deterministic stub (acquire, frame_map,
  lead_in_span, twin, channel_for, verify, verify_l, verify_track,
  _ce_resid_ratio, prepare, receive_pass, the clock, rules.old) and every
  receiver function it calls logged (candidates, detect, _detect_joint,
  _gate, _receive_slot_joint, frames, is_joint, _single, centre_hz,
  _centres, _ce_hints): the same calls in the same order with the same
  arguments (float and np.float64 alike: the C++ hands numbers on as
  float), the same result objects in the same order, the same errors --
  with max_candidates and budget_s (a fake clock) at their edges, owned
  lines, merges, aliases, ghosts, suspect neighbours, twins, waveforms
  not listened for, a detection listed twice, QRSS_OLD_GATING both ways;
  every rule decides against something at least 8 times;
- the whole callers, every row of every tx and rx rows file in place, the
  receiver's rows counted (the spies installed before the rows: they are
  what the module attributes hold): qrss_receive.py's main on a joint
  .npz and a CE WAV with --no-store, and live.finish_receive heard whole
  and in part, print and return what the reference does (pass uids but
  their hash), to the bars above but Z_ref heard in part, 1e-5 [1.06e-6:
  _erased_same says why], and every slot row is reached; the same callers
  typed (no spies) likewise;
- capture_from_wav (typed: the C++ reads the WAV): unsigned 8-bit, 16,
  24 and 32-bit PCM, float32 and float64, 1, 2, 3, 8 and 9 channels,
  WAVE_FORMAT_EXTENSIBLE with a LIST chunk, at 8000, 11025, 44100 and
  48000 Hz: bit for bit what wavio.read_wav then the native front end
  make (1e-12 of the scale allowed when resampled) [0 at every rate];
  the reference's capture to the front end's bar (1e-5); q, t0_index and
  fs exactly; FileNotFoundError and IsADirectoryError with the
  reference's errno and text; receive_wav's pass, typed throughout;
- threads: four slots received at once, typed, each what it gives alone.
"""

import copy
import dataclasses
import functools
import math
import os
import random
import runpy
import struct
import time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from qrss_demod_tracks import CALL, HDR
from qrss_fakes import l_scenario, synthetic_picture
from qrss_helpers import Q_TEST, unit_rms_latents
from qrss_native import require

import test_qrss_native_receiver as P
from sstvae import wavio
from sstvae.qrss import acquire as A
from sstvae.qrss import channel as chm
from sstvae.qrss import frame, frontend, header, lframe, live, morse, rules, tx
from sstvae.qrss import receiver as RX
from sstvae.qrss import track as T
from sstvae.qrss.channel import ChannelConfig, Neighbour
from sstvae.qrss.constants import CH_FS, FE_FS, T_SYM, WAVEFORM_CE, WAVEFORM_L
from sstvae.qrss.precoder import precode
from sstvae.qrss.types import Detection, FreqPath, Timing

ROOT = Path(__file__).resolve().parent.parent
RECEIVER = "sstvae.qrss.receiver"
T_L = lframe.L_TINY.t_sym


@pytest.fixture(scope="module", autouse=True)
def _print_worst():
    yield
    if os.environ.get("QRSS_PARITY_WORST"):
        for k in sorted(P.WORST):
            if k.startswith(("slot", "det", "wav")):
                print(f"worst {k}: {P.WORST[k]:.3g}")


@pytest.fixture(scope="module")
def Rs(native):
    return require(native, "receiver_slot")


@pytest.fixture(scope="module")
def row(native, Rs):
    from qrss_native.receiver_slot import rows

    return {name: fn for (_t, name), fn in rows(native.qrss).items()}


def _outcome(fn, *a, **k):
    try:
        return "ok", fn(*a, **k)
    except Exception as e:  # noqa: BLE001 - the reference's own type and text
        return type(e), str(e)


@contextmanager
def _env(name):
    """QRSS_OLD_* `name` set for the block (None: nothing)."""
    if name is None:
        yield
        return
    old = os.environ.get(name)
    os.environ[name] = "1"
    try:
        yield
    finally:
        if old is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = old


# --- the rules -----------------------------------------------------------------------------------

_OFFSETS = (0.0, 0.2, 0.49, 0.51, 2.0, 4.9, 5.0, 5.1, 9.5, -9.5, 12.0, -12.0, 19.9, 20.0, 20.1, 22.1, -22.1,
            -22.2, 3.1, 3.2, 1 / T_SYM, -1 / T_SYM, 1 / T_SYM + 0.4, 2 / T_SYM, -2 / T_SYM + 0.3, 1 / T_L,
            2 / T_L, -1 / T_L - 0.45, 60.0, 100.0)
_TAUS = (0.0, 1.0, 5.0, 7.5, 11.36, 11.37, 15.0, 22.7, 22.8, 34.1, 45.4, 45.5, 60.0, 200.0)
_ZS = (3.0, 6.0, 9.0, 18.0, 27.0, 27.0, 50.0, 100.0, 150.0, 400.0)


def _rand_det(rng, timing=True):
    tau = rng.choice(_TAUS) + rng.choice((0.0, 0.25, -0.25))
    method = rng.choice(("preamble", "tbd", "tbd", "hint", "template"))
    a3 = rng.choice((math.nan, 3.0, 9.0, 18.0, 27.0, math.inf, -math.inf)) if method == "tbd" or \
        rng.random() < 0.1 else math.nan
    f = 1500.0 + rng.choice(_OFFSETS) + rng.choice((0.0, 0.0, 0.01))
    tm = Timing(tau0=tau, ppm=0.0, gamma=0.0, z=1.0, cov=np.eye(2)) if timing else None
    d = Detection(f_hz=f, path=None, timing=tm, z_ref=rng.choice(_ZS), method=method,
                  lead_in_s=0.0, waveform=rng.choice((WAVEFORM_CE, WAVEFORM_L)), a3_z=a3)
    if rng.random() < 0.08:          # duck-typed: no waveform, no a3_z
        d = SimpleNamespace(f_hz=d.f_hz, timing=d.timing, z_ref=d.z_ref, method=d.method, lead_in_s=0.0)
    return d


def _rand_pass(rng):
    return SimpleNamespace(f_hz=1500.0 + rng.choice(_OFFSETS), waveform=rng.choice((WAVEFORM_CE, WAVEFORM_L)),
                           report=SimpleNamespace(suspect=rng.random() < 0.5, z_ref=rng.choice(_ZS)),
                           timing=SimpleNamespace(tau0=rng.choice(_TAUS)))


def _ids(r):
    return ("ok", [id(x) for x in r[1]]) if r[0] == "ok" else r


def test_rules_fuzz(row, monkeypatch):
    """Every rule against the reference's on 600 random sets (module
    docstring): _aliases and _ghosts hand back the caller's own objects in
    the reference's order, the rest the same values, and a detection
    without timing fails as the reference fails, where it fails."""
    n = {"alias": 0, "ghost": 0, "owned": 0, "suspect": 0, "beside": 0, "err": 0}
    for seed in range(600):
        rng = random.Random(seed)
        dets = [_rand_det(rng, timing=rng.random() > 0.04) for _ in range(rng.randrange(0, 9))]
        t_sym = rng.choice((T_SYM, T_L))
        for name in ("_aliases", "_ghosts"):
            ref = _outcome(getattr(RX, name), dets, t_sym)
            got = _outcome(row[name], dets, t_sym)
            assert _ids(got) == _ids(ref), (seed, name, ref, got)
            if ref[0] == "ok":
                n["alias" if name == "_aliases" else "ghost"] += len(dets) - len(ref[1])
            else:
                n["err"] += 1
        if rng.random() < 0.2:      # the default symbol
            assert _ids(_outcome(row["_aliases"], dets)) == _ids(_outcome(RX._aliases, dets)), seed
            assert _ids(_outcome(row["_ghosts"], dets)) == _ids(_outcome(RX._ghosts, dets)), seed
        for d in dets:
            for g in dets:
                if d.timing is None or g.timing is None:
                    continue
                a = (d.f_hz, d.timing.tau0, g.f_hz, g.timing.tau0)
                r = RX._ghost_of(*a, t_sym)
                assert row["_ghost_of"](*a, t_sym) is r, (seed, a)
                assert row["_ghost_of"](*a) is RX._ghost_of(*a), (seed, a)
        a3 = RX._a3_lines(dets)
        assert row["_a3_lines"](dets) == a3, seed
        if rng.random() < 0.3:
            a3 = [(1500.0 + rng.choice(_OFFSETS), 1500.0 + rng.choice(_OFFSETS), rng.choice(_ZS))
                  for _ in range(rng.randrange(0, 4))]
        found = dets[: rng.randrange(0, len(dets) + 1)]
        for old in (False, True):
            if old:
                monkeypatch.setenv(rules.GATING, "1")
            else:
                monkeypatch.delenv(rules.GATING, raising=False)
            for d in dets:
                r = RX._owned(d, found, a3)
                assert row["_owned"](d, found, a3) is r, (seed, old)
                n["owned"] += r
        passes = [_rand_pass(rng) for _ in range(rng.randrange(0, 6))]
        for p in passes:
            for name, key in (("_suspect_neighbour", "suspect"), ("_suspect_beside", "beside")):
                r = getattr(RX, name)(p, passes)
                assert row[name](p, passes) is r, (seed, name)
                n[key] += r
    assert all(v >= 20 for v in n.values()), n


def test_rules_refusals(row):
    """_owned's a3 entries unpack as the reference's do (where the
    reference unpacks them: a candidate in a found signal's band)."""
    d = Detection(f_hz=1500.0, path=None, timing=None, z_ref=9.0, method="tbd", lead_in_s=0.0, a3_z=3.0)
    g = Detection(f_hz=1500.0, path=None, timing=None, z_ref=90.0, method="preamble", lead_in_s=0.0)
    for a3 in ([(1500.0, 1500.0)], [(1500.0, 1500.0, 9.0, 1.0)]):
        assert _outcome(row["_owned"], d, [g], a3)[0] is _outcome(RX._owned, d, [g], a3)[0] is ValueError
    assert row["_owned"](d, [g], [(1500.0, 1500.0, 9.0)]) is RX._owned(d, [g], [(1500.0, 1500.0, 9.0)]) is True


def test_ce_resid_ratio(row):
    """_ce_resid_ratio on real CE gate-V tracks of the joint capture (the
    reference's tracks): the CE signal (R near 1) and the L signal gated as
    CE (R well over CE_RESID_MIN), each to 1e-12 relative of the
    reference's; 1.0 with every known symbol erased, as the reference."""
    cap = _cap("mixed")
    prep = RX.prepare(cap)
    cands = RX.candidates(prep, JOINT)       # the L one CE's gate passes is its A3 line (Z 6.7, R 4.0)
    rs = []
    for c in cands:
        d = dataclasses.replace(c, waveform=WAVEFORM_CE)
        chan = RX.channel_for(prep, frame.TINY, d, ())
        v = T.verify(chan, CH_FS, frame.TINY, d)
        tr = T.verify_track(chan, frame.TINY, v, fit=False) if v is not None else None
        if tr is None:
            continue
        r = RX._ce_resid_ratio(tr)
        g = row["_ce_resid_ratio"](tr)
        assert type(g) is float and abs(g - r) <= 1e-12 * abs(r), (c.f_hz, g, r)
        rs.append((c.waveform, r))
        tr.psi = np.zeros_like(tr.psi)
        assert row["_ce_resid_ratio"](tr) == RX._ce_resid_ratio(tr) == 1.0
    by = dict(rs)
    assert by[WAVEFORM_CE] < RX.CE_RESID_MIN < by[WAVEFORM_L], rs


# --- the scenarios -------------------------------------------------------------------------------

def _neigh(df, rel):
    return (("neighbours", (Neighbour(df, rel, seed=7),)),)


def _ce_cap(name, snr, seed, cfg=(), carrier=1500.0):
    spec = frame.get(name)
    a = unit_rms_latents(spec.n_data, seed)
    sym = frame.assemble(spec, header.encode(HDR) if spec.n_hdr else None, precode(a, Q_TEST))
    sim = chm.simulate(sym, spec, Q_TEST, ChannelConfig(snr_db=snr, seed=seed, **dict(cfg)), carrier_hz=carrier,
                       keying=morse.keying_units(CALL) if spec.n_win else None)
    return frontend.Capture(sim.fe, Q_TEST, sim.t0_index)


def _l_cap(name, snr, seed):
    _, sim = l_scenario(name, snr, seed)
    return frontend.Capture(sim.fe, sim.q, sim.t0_index)


def _noise_cap():
    rng = np.random.default_rng(4)
    n = int((frame.TINY.keyed_end_pos * frame.TINY.t_sym + 40) * FE_FS)
    fe = ((rng.standard_normal(n) + 1j * rng.standard_normal(n)) / math.sqrt(2)).astype(np.complex64)
    return frontend.Capture(fe, Q_TEST, FE_FS * 20.0)


CAPTURES = {
    "ce-tiny": lambda: _ce_cap("tiny", -12.0, 2),
    "l-tiny": lambda: _l_cap("l-tiny", -6.0, 1),
    "mixed": lambda: P._mixed("l-tiny"),                 # L at -12 dB, CE 60 Hz above at 0 dB relative
    "alias": lambda: _ce_cap("tiny", 10.0, 3, _neigh(32.99, -22.0)),   # 1/T above
    "weak": lambda: _ce_cap("tiny", 0.0, 3, _neigh(60.0, -10.0)),
    "near": lambda: _ce_cap("tiny", 0.0, 3, _neigh(12.0, -15.0)),
    "edge": lambda: _ce_cap("tiny", -6.0, 1, carrier=350.0),
    "noise": _noise_cap,
    "ce-short": lambda: _ce_cap("short", -6.0, 1),
    "l-short": lambda: _l_cap("l-short", -6.0, 1),
    "mixed-short": lambda: P._mixed("l-short", -6.0),
}


@functools.lru_cache(maxsize=None)
def _cap(name):
    return CAPTURES[name]()


JOINT = (frame.TINY, lframe.L_TINY)
# label: (capture, selection, the QRSS_OLD_* rule set, receive_slot's keywords)
SCENARIOS = {
    "ce-tiny": ("ce-tiny", frame.TINY, None, ()),
    "l-tiny": ("l-tiny", lframe.L_TINY, None, ()),
    "joint": ("mixed", JOINT, None, ()),
    "ce-beside-l": ("mixed", frame.TINY, None, ()),
    "alias": ("alias", frame.TINY, None, ()),
    "weak-neighbour": ("weak", frame.TINY, None, ()),
    "noise": ("noise", JOINT, None, ()),
    "old-gating": ("near", frame.TINY, rules.GATING, ()),
    "old-edge-noise": ("edge", frame.TINY, rules.EDGE_NOISE, ()),
    "live-only": ("mixed", JOINT, None, (("live_only", True),)),
    "max-candidates": ("weak", frame.TINY, None, (("max_candidates", 1),)),
    "joint-max-candidates": ("mixed", JOINT, None, (("max_candidates", 1),)),
    "erase": ("ce-tiny", frame.TINY, None, (("erase_s", ((5.0, 12.0),)),)),
    "hints": ("mixed", JOINT, None, (("hints", ((1560.0, WAVEFORM_CE), 1700.0)), ("live_only", True))),
    "plain": ("l-tiny", lframe.L_TINY, None, (("estimator", "plain"), ("round_b", False))),
    "ce-short": ("ce-short", frame.SHORT, None, ()),
    "l-short": ("l-short", lframe.L_SHORT, None, ()),
    "joint-short": ("mixed-short", (frame.SHORT, lframe.L_SHORT), None, ()),
}
SLOW = {"ce-short", "l-short", "joint-short", "live-only", "joint-max-candidates", "hints"}


def _labels():
    return [pytest.param(k, marks=pytest.mark.slow) if k in SLOW else k for k in SCENARIOS]


@functools.lru_cache(maxsize=None)
def _reference(label):
    """The reference's receive_slot on the scenario, with the detect call
    it made and what that returned, and how often the alias rule dropped a
    detection (call it before any row is applied: it is the reference's)."""
    capname, spec, env, kw = SCENARIOS[label]
    cap = _cap(capname)
    kw = dict(kw)
    calls, hits = [], {}
    detect, aliases = RX.detect, RX._aliases

    def spy(*a, **k):
        out = detect(*a, **k)
        calls.append((a, k, out))
        return out

    def alias_spy(dets, *a):
        out = aliases(dets, *a)
        hits["_aliases"] = hits.get("_aliases", 0) + len(dets) - len(out)
        return out
    with pytest.MonkeyPatch.context() as mp, _env(env):
        mp.setattr(RX, "detect", spy)
        mp.setattr(RX, "_aliases", alias_spy)
        t = time.perf_counter()
        passes = RX.receive_slot(cap, spec, **kw)
        t_slot = time.perf_counter() - t
    assert len(calls) <= 1, label
    return SimpleNamespace(cap=cap, spec=spec, env=env, kw=kw, passes=passes, detect=calls[0] if calls else None,
                           hits=hits, t=t_slot)


def _spy_typed(native, mp, seen, names=("receive_slot", "_receive_slot_joint", "detect", "_detect_joint")):
    """Record the `typed` flag the rows hand the binding."""
    m = native.qrss.receiver_slot
    for name in names:
        orig = getattr(m, name)

        def spy(*a, _orig=orig, _name=name):
            seen.append((_name, a[-1]))
            return _orig(*a)
        mp.setattr(m, name, spy)


def _run(native, label, python_mode=False, dets=None):
    """receive_slot on the scenario with every row in place, and detect
    as the reference's receive_slot called it, typed (asserted) or, with
    python_mode, over the Python objects (a pass-through spy on
    track.verify sends it there)."""
    r = _reference(label)
    seen = []
    got = None
    with pytest.MonkeyPatch.context() as mp:
        P._apply_rows(native, mp)
        _spy_typed(native, mp, seen)
        if python_mode:
            verify = T.verify
            mp.setattr(T, "verify", lambda *a, **k: verify(*a, **k))
        kw = dict(r.kw, dets=dets) if dets is not None else r.kw
        with _env(r.env):
            t = time.perf_counter()
            passes = RX.receive_slot(r.cap, r.spec, **kw)
            t_slot = time.perf_counter() - t
            if dets is None and r.detect is not None:
                a, k, _ = r.detect
                got = RX.detect(*a, **k)
    assert seen and all(flag is not python_mode for _n, flag in seen), (label, seen)
    return SimpleNamespace(passes=passes, dets=got, t=t_slot)


def _det_close(g, r, what, key="det"):
    assert type(g) is type(r), what
    assert (g.method, g.waveform) == (r.method, r.waveform), what
    P._abs(g.f_hz, r.f_hz, 1e-6, (what, "f_hz"), f"{key} f_hz (Hz, abs)")
    P._abs(g.lead_in_s, r.lead_in_s, 1e-9, (what, "lead_in_s"), f"{key} lead_in_s (s, abs)")
    P._num(g.z_ref, r.z_ref, 1e-6, (what, "z_ref"), f"{key} z_ref")
    assert (math.isnan(g.a3_z) and math.isnan(r.a3_z)) or g.a3_z == r.a3_z, what
    assert (g.timing is None) == (r.timing is None), what
    if r.timing is not None:
        P._abs(g.timing.tau0, r.timing.tau0, 1e-6 * CH_FS, (what, "tau0"), f"{key} tau0 (CH samples, abs)")
        for f in ("ppm", "z"):
            P._num(getattr(g.timing, f), getattr(r.timing, f), 1e-6, (what, f), f"{key} timing.{f}")
    for f in ("t_s", "f_hz", "weight"):
        e = P._rel(getattr(g.path, f), getattr(r.path, f), (what, "path", f))
        P._worst(f"{key} path.{f}", e)
        assert e <= 1e-6, (what, f, e)


def _dets_close(got, ref, what):
    assert len(got) == len(ref), (what, [d.f_hz for d in got], [d.f_hz for d in ref])
    for i, (g, r) in enumerate(zip(got, ref)):
        _det_close(g, r, (what, i))


def _slot_close(got, ref, what):
    assert len(got) == len(ref), (what, [p.f_hz for p in got], [p.f_hz for p in ref])
    for i, (g, r) in enumerate(zip(got, ref)):
        P._pass_close(g, r, (what, i), key="slot")


# the scenarios also run over the Python objects, against the typed run
PYTHON_MODE = {"joint", "old-gating", "joint-short"}


@pytest.mark.parametrize("label", _labels())
def test_receive_slot(native, Rs, label):
    """receive_slot, and detect as it called it, on the scenario, typed,
    against the reference (module docstring); and (PYTHON_MODE) over the
    Python objects, the typed run's bit for bit."""
    r = _reference(label)
    g = _run(native, label)
    _slot_close(g.passes, r.passes, label)
    _dets_close(g.dets, r.detect[2], label)
    if label == "noise":
        assert r.passes == [] and r.detect[2] == [], label
    else:
        assert r.passes, label
    if label not in PYTHON_MODE:
        return
    h = _run(native, label, python_mode=True)
    assert len(h.passes) == len(g.passes) and len(h.dets) == len(g.dets), label
    for a, b in zip(h.passes, g.passes):
        P._bitwise_pass((a, []), (b, []), label)
    for a, b in zip(h.dets, g.dets):
        _det_same(a, b, label)


def _det_same(a, b, what):
    for f in ("f_hz", "z_ref", "method", "lead_in_s", "waveform"):
        assert getattr(a, f) == getattr(b, f), (what, f)
    assert a.a3_z == b.a3_z or (math.isnan(a.a3_z) and math.isnan(b.a3_z)), what
    for f in ("tau0", "ppm", "gamma", "z"):
        assert getattr(a.timing, f) == getattr(b.timing, f), (what, f)
    P._same(a.timing.cov, b.timing.cov, what)
    for f in ("t_s", "f_hz", "weight"):
        P._same(getattr(a.path, f), getattr(b.path, f), (what, f))


def test_scenarios_reach_the_rules():
    """The scenarios do what their names say, on the reference: the alias
    rule drops the -22 dB neighbour 1/T above, the joint capture holds an L
    and a CE pass, the weak neighbour is a second pass, the gating rule
    changes which candidates are verified (not what is found), and the
    edge rule moves the SNR a 350 Hz pass reports."""
    assert _reference("alias").hits == {"_aliases": 1}
    assert sorted(p.waveform for p in _reference("joint").passes) == [WAVEFORM_CE, WAVEFORM_L]
    assert [round(p.f_hz) for p in _reference("weak-neighbour").passes] == [1500, 1560]
    gating = _reference("old-gating")
    n = {}
    for env in (None, gating.env):
        with pytest.MonkeyPatch.context() as mp, _env(env):
            verify = T.verify
            mp.setattr(T, "verify", lambda *a, _e=env, **k: n.__setitem__(_e, n.get(_e, 0) + 1) or verify(*a, **k))
            RX.detect(*gating.detect[0], **gating.detect[1])
    assert n[gating.env] > n[None], n
    edge = _reference("old-edge-noise")
    new = RX.receive_slot(edge.cap, edge.spec)
    assert len(new) == len(edge.passes) == 1
    assert new[0].report.snr2500_db > edge.passes[0].report.snr2500_db + 0.1


def test_receive_slot_dets_given(native, Rs):
    """receive_slot with `dets` (the live view's preamble detections, the
    reference's objects), CE and joint: typed, the reference's passes; one
    listed twice goes over the Python objects and is received twice, as the
    reference's is."""
    for label in ("weak-neighbour", "joint"):
        r = _reference(label)
        dets = list(r.detect[2])
        ref = RX.receive_slot(r.cap, r.spec, dets=dets)
        g = _run(native, label, dets=dets)
        _slot_close(g.passes, ref, (label, "dets"))
        twice = dets + dets[:1]
        ref2 = RX.receive_slot(r.cap, r.spec, dets=twice)
        seen = []
        with pytest.MonkeyPatch.context() as mp:
            P._apply_rows(native, mp)
            _spy_typed(native, mp, seen)
            got2 = RX.receive_slot(r.cap, r.spec, dets=twice)
        _slot_close(got2, ref2, (label, "dets twice"))
        # the row was typed, the binding fell through to the Python objects
        # (the joint one then calls _receive_slot_joint's row, typed again)
        assert seen[0] == ("receive_slot", True) and len(ref2) == len(ref) == 2, (label, seen)


# --- the stubbed orchestration ---------------------------------------------------------------------

def _canon(x):
    """A comparable rendering of a stage's argument or result: stub objects
    by label, numbers by value (float and np.float64 alike, ints apart),
    containers by type and content."""
    lab = getattr(x, "_label", None)
    if isinstance(lab, str):
        return lab
    if x is None or isinstance(x, str):
        return x
    if isinstance(x, (bool, np.bool_)):
        return ("b", bool(x))
    if isinstance(x, (int, np.integer)):
        return ("i", int(x))
    if isinstance(x, (float, np.floating)):
        return ("f", float(x).hex())
    if hasattr(x, "t_sym") and hasattr(x, "name"):
        return ("spec", type(x).__name__, x.name)
    if isinstance(x, np.ndarray):
        return ("arr", x.dtype.str, x.shape, x.tobytes())
    if isinstance(x, (list, tuple)):
        return (type(x).__name__, tuple(_canon(v) for v in x))
    if isinstance(x, dict):
        return ("dict", tuple((_canon(k), _canon(v)) for k, v in x.items()))
    if dataclasses.is_dataclass(x):
        return (type(x).__name__,) + tuple((f.name, _canon(getattr(x, f.name))) for f in dataclasses.fields(x))
    if isinstance(x, SimpleNamespace):
        return ("ns",) + tuple(sorted((k, _canon(v)) for k, v in vars(x).items()))
    return ("obj", type(x).__name__, repr(x))


_SINGLE = (frame.TINY, frame.SHORT, (frame.TINY,), [frame.SHORT])
_JOINTS = ((frame.TINY, lframe.L_TINY), (lframe.L_TINY, frame.TINY), [frame.SHORT, lframe.L_SHORT],
           lframe.L_TINY, (lframe.L_SHORT,))
_BAD = ((frame.TINY, frame.SHORT),)
_F = (1500.0, 1500.3, 1504.9, 1505.2, 1512.0, 1488.0, 1490.5, 1519.9, 1521.0, 1500.0 + 1 / T_SYM,
      1500.0 - 1 / T_SYM + 0.2, 1500.0 + 2 / T_SYM, 1500.0 + 1 / T_L, 1478.0, 1477.8, 1503.1, 1503.2, 1560.0,
      1561.0, 1600.0, 1700.0)

# a strong signal's neighbourhood: its own line, merges, ghosts, aliases (CE's 1/T, L's 1/T_L)
_F_NEAR = (1500.0, 1500.0, 1500.3, 1503.1, 1504.9, 1505.2, 1512.0, 1488.0, 1500.0 + 1 / T_SYM,
           1500.0 + 1 / T_SYM + 0.3, 1500.0 + 1 / T_L, 1500.0 - 2 / T_SYM)


class _World:
    """Deterministic stubs for every stage the slot receiver reaches, each
    logging its call; the same seed gives the same answers to the same
    calls in the same order."""

    def __init__(self, seed, joint):
        self.rng = rng = random.Random(seed)
        self.log = []
        self.n = 0
        self.clock = 0.0
        self.step = rng.choice((0.5, 1.0, 1.25))
        self.cands = [self._cand(joint) for _ in range(rng.randrange(0, 11))]
        if rng.random() < 0.35:     # a strong signal and its k/T aliases at its timing
            self._aliases_of(joint)
        if rng.random() < 0.8:      # acquisition's order: preamble hits, hints, then A3's lines
            self.cands.sort(key=lambda d: ("preamble", "hint", "tbd").index(d.method))
        if self.cands and rng.random() < 0.15:
            self.cands.append(self.cands[rng.randrange(len(self.cands))])   # one listed twice

    def label(self, kind):
        self.n += 1
        return f"{kind}{self.n}"

    def _cand(self, joint):
        rng = self.rng
        method = rng.choice(("preamble", "preamble", "tbd", "tbd", "tbd", "hint"))
        a3 = rng.choice((3.0, 9.0, 27.0, 81.0, math.nan, math.inf)) if method == "tbd" else math.nan
        wf = rng.choice((WAVEFORM_CE, WAVEFORM_L)) if joint or rng.random() < 0.1 else WAVEFORM_CE
        if joint and rng.random() < 0.04:
            wf = 7                                   # a waveform not listened for
        f = rng.choice(_F if rng.random() < 0.5 else _F_NEAR) + rng.choice((0.0, 0.0, 0.004))
        if rng.random() < 0.1:
            f = np.float64(f)
        tm = None if rng.random() < 0.03 else SimpleNamespace(tau0=rng.choice((0.0, 5.0, 10.0, 40.0, 100.0)))
        lead = rng.choice((0.0, 0.0, 0, 1.75))
        if rng.random() < 0.1:
            d = SimpleNamespace(f_hz=f, timing=tm, z_ref=rng.choice(_ZS), method=method, lead_in_s=lead)
        else:
            d = Detection(f_hz=f, path=None, timing=tm, z_ref=rng.choice(_ZS), method=method, lead_in_s=lead,
                          waveform=wf, a3_z=a3)
        d._label = self.label("c")
        return d

    def _aliases_of(self, joint):
        rng = self.rng
        wf = rng.choice((WAVEFORM_CE, WAVEFORM_L)) if joint else WAVEFORM_CE
        t = T_L if wf == WAVEFORM_L else T_SYM
        tau = rng.choice((0.0, 40.0))
        f0 = rng.choice((1500.0, 1700.0))
        strong = Detection(f_hz=f0, path=None, timing=SimpleNamespace(tau0=tau), z_ref=400.0, method="preamble",
                           lead_in_s=0.0, waveform=wf)
        strong._label = self.label("c")
        self.cands.append(strong)
        for _ in range(rng.randrange(1, 4)):
            k = rng.choice((1, 2, -1, -2))
            d = Detection(f_hz=f0 + k / t + rng.choice((0.0, 0.3, -0.45, 0.6)), path=None,
                          timing=SimpleNamespace(tau0=tau + rng.choice((0.0, 5.0, 20.0, 40.0))),
                          z_ref=rng.choice((20.0, 60.0, 150.0)), method=rng.choice(("tbd", "hint")),
                          lead_in_s=0.0, waveform=wf, a3_z=rng.choice((3.0, math.nan)))
            d._label = self.label("c")
            self.cands.append(d)

    def rec(self, name, *a, **k):
        self.log.append((name, _canon(a), tuple(sorted((key, _canon(v)) for key, v in k.items()))))

    def maybe_fail(self, name, p=0.01):
        if self.rng.random() < p:
            raise RuntimeError(f"{name} failed ({self.n})")

    # the stages
    def monotonic(self):
        self.rec("monotonic")
        self.clock += self.step
        return self.clock

    def old(self, rule):
        self.rec("rules.old", rule)
        return rules.old(rule)

    def acquire(self, cap, x, live_only=False, preprocessed=False, hints=()):
        self.rec("acquire", cap, x, live_only=live_only, preprocessed=preprocessed, hints=hints)
        self.maybe_fail("acquire")
        world = self

        class Acq:
            def detections(self, specs):
                world.rec("detections", specs)
                return list(world.cands)
        return Acq()

    def frame_map(self, specs):
        self.rec("frame_map", specs)
        return A.frame_map(specs)

    def lead_in_span(self, fe, t0, f, t_start, tau_s, fs, **k):
        self.rec("lead_in_span", fe, t0, f, t_start, tau_s, fs, **k)
        return self.rng.choice((0.0, 1.5, 3.25))

    def twin(self, spec, w):
        self.rec("twin", spec, w)
        return lframe.twin(spec, w)

    def channel_for(self, prep, spec, d, others):
        self.rec("channel_for", prep, spec, d, others)
        self.maybe_fail("channel_for")
        return SimpleNamespace(_label=self.label("chan"))

    def _verified(self, d):
        rng = self.rng
        if rng.random() < 0.3:
            return None
        v = copy.copy(d)
        v.f_hz = float(d.f_hz) + rng.choice((0.0, 0.0, 0.0, 0.01, -0.2, 0.6))
        v.z_ref = (rng.choice((4.0, 5.9, 6.0, 8.0, 12.0, 30.0, 80.0)) if rng.random() < 0.4
                   else d.z_ref * rng.choice((0.5, 1.0, 2.0)))
        if d.timing is not None:
            v.timing = SimpleNamespace(tau0=d.timing.tau0 + rng.choice((0.0, 0.0, 2.0, 30.0)))
        if rng.random() < 0.1:
            v.method = "preamble"
        v._label = self.label("v")
        return v

    def verify(self, chan, fs, spec, d, erase_s=()):
        self.rec("verify", chan, fs, spec, d, erase_s=erase_s)
        self.maybe_fail("verify")
        return self._verified(d)

    def verify_l(self, chan, fs, spec, d, erase_s=()):
        self.rec("verify_l", chan, fs, spec, d, erase_s=erase_s)
        return self._verified(d)

    def verify_track(self, chan, spec, v, fit=True, erase_s=()):
        self.rec("verify_track", chan, spec, v, fit=fit, erase_s=erase_s)
        return None if self.rng.random() < 0.2 else SimpleNamespace(_label=self.label("tr"))

    def ce_resid_ratio(self, tr):
        self.rec("_ce_resid_ratio", tr)
        return self.rng.choice((0.5, 1.0, 2.0, 2.0001, 3.0, 4.0, 16.0, 100.0))

    def prepare(self, cap):
        self.rec("prepare", cap)
        return SimpleNamespace(_label="prep", fe=np.zeros(4, np.complex64), keep=None, q=Q_TEST,
                               t0_index=80000.0, fs=FE_FS)

    def receive_pass(self, prep, spec, d, estimator="joint", exclude_hz=(), round_b=True, erase_s=()):
        self.rec("receive_pass", prep, spec, d, estimator=estimator, exclude_hz=exclude_hz, round_b=round_b,
                 erase_s=erase_s)
        rng = self.rng
        self.maybe_fail("receive_pass")
        if rng.random() < 0.2:
            return None
        tau = d.timing.tau0 if d.timing is not None else 0.0
        return SimpleNamespace(
            _label=self.label("p"), f_hz=float(d.f_hz) + rng.choice((0.0, 0.5, 3.0, 4.99, 6.0, 15.0, 25.0)),
            waveform=spec.waveform, timing=SimpleNamespace(tau0=tau + rng.choice((0.0, 10.0, 30.0, 80.0, 200.0))),
            report=SimpleNamespace(z_ref=rng.choice((5.0, 9.0, 50.0, 100.0, 1000.0)), suspect=rng.random() < 0.5))


# the receiver functions the slot code calls and the stubs stand under
_LOGGED = ("candidates", "detect", "_detect_joint", "_gate", "receive_slot", "_receive_slot_joint", "frames",
           "is_joint", "_single", "centre_hz", "_centres", "_ce_hints")


def _install(world, mp, slot_fns):
    """The world's stubs over receiver's globals, and every function of
    _LOGGED (slot_fns's for the slot rows, the reference's otherwise)
    logged."""
    mp.setattr(RX, "acquire", SimpleNamespace(acquire=world.acquire, frame_map=world.frame_map,
                                              lead_in_span=world.lead_in_span))
    mp.setattr(RX, "track", SimpleNamespace(verify=world.verify, verify_track=world.verify_track))
    mp.setattr(RX, "ltrack", SimpleNamespace(verify_l=world.verify_l))
    mp.setattr(RX, "lframe", SimpleNamespace(twin=world.twin))
    mp.setattr(RX, "time", SimpleNamespace(monotonic=world.monotonic))
    mp.setattr(RX, "rules", SimpleNamespace(old=world.old, GATING=rules.GATING))
    mp.setattr(RX, "channel_for", world.channel_for)
    mp.setattr(RX, "_ce_resid_ratio", world.ce_resid_ratio)
    mp.setattr(RX, "prepare", world.prepare)
    mp.setattr(RX, "receive_pass", world.receive_pass)
    for name in _LOGGED:
        fn = slot_fns.get(name, getattr(RX, name))

        def logged(*a, _fn=fn, _name=name, **k):
            world.rec(_name, *a, **k)
            return _fn(*a, **k)
        mp.setattr(RX, name, logged)


def _stub_call(seed):
    """(joint, entry, args, kwargs) of one fuzz case, drawn from `seed`."""
    rng = random.Random(10_000 + seed)
    kind = rng.choice(("single", "single", "joint", "joint", "joint", "bad"))
    spec = rng.choice({"single": _SINGLE, "joint": _JOINTS, "bad": _BAD}[kind])
    joint = kind != "single"
    entry = rng.choice(("receive_slot",) * 5 + ("detect",) * 4 + ("_gate", "candidates"))
    maxc = rng.choice((None, None, 0, 1, 2, 3, 5, np.int64(2), 2.5))
    budget = rng.choice((None, None, None, 0, 0.5, 2, 3.5, 6.0, np.float64(2.0)))
    erase = rng.choice(((), ((1.0, 2.0),), [(0, 5)]))
    hints = rng.choice(((), [1600.0], [(1600.0, WAVEFORM_L), 1550.0], [(1500.0, WAVEFORM_CE)], ((1700.0, 0),)))
    live_only = rng.choice((False, True, 0, 1))
    skip = rng.choice(((), [1500.0], [1512.0, 1700], np.array([1505.0]),
                       [SimpleNamespace(f_hz=1509.5, waveform=WAVEFORM_L), 1560.0]))
    if not joint and not isinstance(skip, np.ndarray) and any(isinstance(s, SimpleNamespace) for s in skip):
        skip = (1504.0,)
    return SimpleNamespace(rng=rng, joint=joint, spec=spec, entry=entry, maxc=maxc, budget=budget, erase=erase,
                           hints=hints, live_only=live_only, skip=skip)


def _rule_hits(mp, hits):
    """Count, in the reference's run, what each rule decided against a
    detection or pass (the C++ applies the rules itself: nothing to count
    there, and the logs say it decided alike)."""
    def wrap(name, fn):
        def f(*a, **k):
            r = fn(*a, **k)
            hits[name] = hits.get(name, 0) + (len(a[0]) - len(r) if name in ("_aliases", "_ghosts") else bool(r))
            return r
        return f
    for name in ("_aliases", "_ghosts", "_owned", "_ghost_of", "_suspect_neighbour", "_suspect_beside"):
        mp.setattr(RX, name, wrap(name, getattr(RX, name)))


def _stub_case(seed, mp, slot_fns, hits=None):
    """One fuzz case: (outcome, log)."""
    c = _stub_call(seed)
    world = _World(seed, c.joint)
    if random.Random(20_000 + seed).random() < 0.25:
        mp.setenv(rules.GATING, "1")
    else:
        mp.delenv(rules.GATING, raising=False)
    _install(world, mp, slot_fns)
    if hits is not None:
        _rule_hits(mp, hits)
    prep = world.prepare(None)
    world.log.clear()
    rng = c.rng
    if c.entry == "receive_slot":
        dets = None
        if rng.random() < 0.3:
            dets = [world._verified(d) or d for d in world.cands]
            if dets and rng.random() < 0.3:
                dets.append(dets[0])
        cap = frontend.Capture(np.zeros(4, np.complex64), Q_TEST, 80000.0)
        out = _outcome(RX.receive_slot, cap, c.spec, c.live_only, rng.choice(("joint", "plain")),
                       erase_s=c.erase, round_b=rng.choice((True, False)), dets=dets, max_candidates=c.maxc,
                       budget_s=c.budget, hints=c.hints)
    elif c.entry == "detect":
        cands = list(world.cands) if rng.random() < 0.3 else None
        out = _outcome(RX.detect, prep, c.spec, c.live_only, c.maxc, c.budget, erase_s=c.erase, cands=cands,
                       skip_hz=c.skip, hints=c.hints)
    elif c.entry == "_gate":
        sp = rng.choice((frame.TINY, frame.SHORT, lframe.L_TINY, lframe.L_SHORT))
        d = world.cands[0] if world.cands else world._cand(True)
        others = tuple(sorted(rng.sample(_F, rng.randrange(0, 4))))
        out = _outcome(RX._gate, prep, sp, d, others, c.erase)
    else:
        out = _outcome(RX.candidates, prep, c.spec, c.live_only, c.hints)
    if out[0] == "ok":
        out = ("ok", _canon(out[1]))
    return out, world.log


def test_slot_orchestration_stubbed(native, row, monkeypatch):
    """The slot receiver with every stage a deterministic stub (module
    docstring), over the Python objects (the logged functions are what
    send it there): the reference's calls, in its order, with its
    arguments, and its results and errors, on 2000 seeds; every rule
    decides against something."""
    slot_rows = {k: row[k] for k in ("candidates", "detect", "_detect_joint", "_gate", "receive_slot",
                                     "_receive_slot_joint")}
    seen = []
    _spy_typed(native, monkeypatch, seen)
    n = {"calls": 0, "verify": 0, "verify_l": 0, "resid": 0, "lead_in": 0, "twin": 0, "err": 0, "passes": 0,
         "budget": 0}
    hits: dict = {}
    for seed in range(2000):
        with monkeypatch.context() as mp:
            ro, rl = _stub_case(seed, mp, {}, hits)
        with monkeypatch.context() as mp:
            go, gl = _stub_case(seed, mp, slot_rows)
        assert len(rl) == len(gl), (seed, len(rl), len(gl), _first_diff(rl, gl))
        for i, (a, b) in enumerate(zip(rl, gl)):
            assert a == b, (seed, i, a, b)
        assert ro == go, (seed, ro, go)
        n["calls"] += len(rl)
        n["err"] += ro[0] != "ok"
        for name, _a, _k in rl:
            if name in ("verify", "verify_l", "lead_in_span", "twin", "_ce_resid_ratio"):
                n[{"lead_in_span": "lead_in", "_ce_resid_ratio": "resid"}.get(name, name)] += 1
            if name == "receive_pass":
                n["passes"] += 1
            if name == "monotonic":
                n["budget"] += 1
    assert seen and not any(flag for _n, flag in seen), seen
    print(n, hits)
    assert n["verify"] >= 800 and n["verify_l"] >= 400 and n["resid"] >= 150 and n["lead_in"] >= 250, n
    assert n["twin"] >= 200 and n["err"] >= 300 and n["passes"] >= 1500 and n["budget"] >= 1800, n
    assert all(hits.get(k, 0) >= 8 for k in ("_aliases", "_ghosts", "_owned", "_ghost_of", "_suspect_neighbour",
                                              "_suspect_beside")), hits


def _first_diff(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i, x, y
    return min(len(a), len(b)), a[len(b):len(b) + 1], b[len(a):len(a) + 1]


def test_budget_stops_at_the_reference_candidate(native, row, monkeypatch):
    """budget_s on a fake clock: the gate stops after the same candidate
    as the reference, for budgets that land just before and just after
    each clock reading."""
    slot_rows = {k: row[k] for k in ("candidates", "detect", "_detect_joint", "_gate", "receive_slot",
                                     "_receive_slot_joint")}
    for seed in range(3):
        for budget in (0, 0.99, 1.0, 1.01, 2.5, 3.0, 4.75):
            outs = []
            for fns in ({}, slot_rows):
                with monkeypatch.context() as mp:
                    world = _World(seed + 7, joint=bool(seed % 2))
                    world.step = 1.0
                    world.cands = [d for d in world.cands if d.timing is not None]
                    _install(world, mp, fns)
                    spec = JOINT if seed % 2 else frame.TINY
                    outs.append((_outcome(RX.detect, world.prepare(None), spec, budget_s=budget), world.log))
            (ro, rl), (go, gl) = outs
            assert rl == gl and _canon(ro[1]) == _canon(go[1]), (seed, budget)


# --- the whole callers ---------------------------------------------------------------------------

_SLOT_ROWS = {"receive_slot", "_receive_slot_joint", "detect", "_detect_joint", "candidates", "_gate",
              "_ce_resid_ratio", "capture_from_wav"}


def _counted(used):
    """test_qrss_native_receiver's counting wrap, the receiver's constant
    rows (GHOST_HZ, ...) left as they are: live reads receiver.GHOST_HZ."""
    inner = P._counted(None, used)
    return lambda key, fn: inner(key, fn) if callable(fn) else fn


def _strip_uid(lines):
    out = []
    for line in lines:
        if line.startswith("pass "):
            uid, rest = line[5:].split(" ", 1)
            line = f"pass {uid.rsplit('_', 1)[0]} {rest}"
        out.append(line)
    return out


@functools.lru_cache(maxsize=None)
def _wav_path(tmpdir):
    """A TINY at 1500 Hz in 8 kHz audio with noise, as qrss_transmit.py
    writes a slot (12 s before t0), int16."""
    from scipy.io import wavfile

    sp = synthetic_picture(0)
    audio = tx.transmit_audio(sp, Q_TEST, 1500, spec=frame.TINY)
    rng = np.random.default_rng(11)
    x = audio + 0.4 * rng.standard_normal(len(audio))
    path = Path(tmpdir) / "rx.wav"
    wavfile.write(path, 8000, np.clip(np.round(x * 12000), -32767, 32767).astype(np.int16))
    return path


def _main_lines(argv, capsys, monkeypatch):
    main = runpy.run_path(str(ROOT / "qrss_receive.py"), run_name="qrss_receive_under_test")["main"]
    monkeypatch.setattr("sys.argv", ["qrss_receive.py", *map(str, argv)])
    capsys.readouterr()
    try:
        main()
        code = 0
    except SystemExit as e:
        code = e.code
    return code, _strip_uid(capsys.readouterr().out.splitlines())


def test_whole_caller_qrss_receive(native, Rs, tmp_path_factory, capsys, monkeypatch):
    """qrss_receive.py's main with --no-store on a joint .npz (an L l-tiny
    at -12 dB, a CE 60 Hz above) and on a CE WAV (--just-ce): the
    reference's lines, with every row in place and the receiver's rows
    counted, and typed; every slot row reached."""
    tmp = tmp_path_factory.mktemp("rx")
    _, sim = l_scenario("l-tiny", -12.0, 1, cfg=(("neighbours", (Neighbour(60.0, 0.0, seed=7,
                                                                           waveform=WAVEFORM_CE),)),))
    npz = tmp / "rx.npz"
    chm.save_npz(npz, sim)
    wav = _wav_path(str(tmp))
    runs = ((npz, "--no-store"), (wav, "--slot", "2025-10-01T00:00Z", "--frame", "tiny", "--just-ce",
                                  "--no-store"))
    ref = [_main_lines(a, capsys, monkeypatch) for a in runs]
    assert ref[0][0] == ref[1][0] == 0, ref
    assert sum(line.startswith("pass ") for line in ref[0][1]) == 2, ref[0]
    assert sum(line.startswith("pass ") for line in ref[1][1]) == 1, ref[1]
    used: dict = {}
    with pytest.MonkeyPatch.context() as mp:
        P._apply_rows(native, mp, wrap=_counted(used))
        got = [_main_lines(a, capsys, monkeypatch) for a in runs]
    assert got == ref
    assert _SLOT_ROWS <= set(used), sorted(_SLOT_ROWS - set(used))
    seen = []
    with pytest.MonkeyPatch.context() as mp:
        P._apply_rows(native, mp)
        _spy_typed(native, mp, seen, names=("receive_slot", "capture_from_wav"))
        typed = [_main_lines(a, capsys, monkeypatch) for a in runs]
    assert typed == ref
    assert seen and all(flag for _n, flag in seen), seen


def _finish_case():
    _, sim = l_scenario("l-tiny", -12.0, 1, cfg=(("neighbours", (Neighbour(60.0, 0.0, seed=7,
                                                                           waveform=WAVEFORM_CE),)),))
    x = np.asarray(sim.fe, np.complex64)
    origin = int(sim.t0_index)
    whole = np.ones(len(x), bool)
    part = np.zeros(len(x), bool)
    part[: origin + 20 * FE_FS] = True
    return x, origin, whole, part


def _erased_same(got, ref, what):
    """_slot_same for passes heard in part: Z_ref to 1e-5 relative. With
    63% of the slot erased the known symbols' weights lean on the
    tracker's covariance P, which the tracker lanes hold to 1e-4 (its
    conditioning: a 1-ulp change of an input moves the reference's own P
    by ~1e-6), and P's 4.9e-5 here moves Z_ref by 1.06e-6 (measured
    2026-10-13, the CE pass: the gate's Z_ref, and with it the pass's)."""
    assert len(got) == len(ref), what
    for g, r in zip(got, ref):
        P._num(g.report.z_ref, r.report.z_ref, 1e-5, (what, "z_ref"), "slot erased report.z_ref")
        g = dataclasses.replace(g, report=dataclasses.replace(g.report, z_ref=r.report.z_ref))
        P._pass_close(g, r, (what, r.frame), key="slot erased")


def test_whole_caller_finish_receive(native, Rs):
    """live.finish_receive on the joint capture, heard whole (the search
    over the slot, with live's candidate and time budgets) and heard for
    the preamble's first 20 s only (the known detection plus a hint,
    detected live-only with the known signal skipped by its centre):
    every row in place and the receiver's rows counted, then typed, the
    reference's passes and share heard (heard in part, Z_ref to 1e-5:
    _erased_same)."""
    x, origin, whole, part = _finish_case()
    sel = JOINT
    ref_whole = live.finish_receive(x, whole, origin, Q_TEST, sel, "joint", [])
    assert len(ref_whole[0]) == 2 and ref_whole[1] == 1.0
    known = [d for d in _reference("joint").detect[2] if d.waveform == WAVEFORM_L]
    hints = [(1560.0, WAVEFORM_CE)]
    ref_part = live.finish_receive(x, part, origin, Q_TEST, sel, "joint", known, hints)
    assert ref_part[0] and 0.0 < ref_part[1] < live.FINISH_SEARCH_MIN_HEARD
    used: dict = {}
    for wrap in (_counted(used), None):
        with pytest.MonkeyPatch.context() as mp:
            P._apply_rows(native, mp, wrap=wrap)
            got_whole = live.finish_receive(x, whole, origin, Q_TEST, sel, "joint", [])
            got_part = live.finish_receive(x, part, origin, Q_TEST, sel, "joint", known, hints)
        assert got_whole[1] == ref_whole[1] and got_part[1] == ref_part[1]
        P._slot_same(got_whole[0], ref_whole[0], ("whole", wrap is None))
        _erased_same(got_part[0], ref_part[0], ("part", wrap is None))
    assert {"receive_slot", "_receive_slot_joint", "detect", "_detect_joint", "candidates", "_gate"} <= set(used), \
        used


# --- capture_from_wav ------------------------------------------------------------------------------

def _wav_bytes(data, rate, fmt, bits, extensible=False, extra_chunk=False):
    """A WAV file's bytes: data (frames, channels) already in its sample
    type, written little-endian as `bits` per sample (24 packed)."""
    frames, ch = data.shape
    if bits == 24:
        v = data.astype("<i4").reshape(-1)
        raw = b"".join(int(s).to_bytes(4, "little", signed=True)[:3] for s in v)
    else:
        raw = data.astype(data.dtype.newbyteorder("<")).tobytes()
    block = ch * bits // 8
    if extensible:
        guid = struct.pack("<HHHH", fmt, 0, 0, 0x0010) + b"\x80\x00\x00\xaa\x00\x38\x9b\x71"
        fmt_body = struct.pack("<HHIIHHHHI", 0xFFFE, ch, rate, rate * block, block, bits, 22, bits,
                               (1 << ch) - 1) + guid
    else:
        fmt_body = struct.pack("<HHIIHH", fmt, ch, rate, rate * block, block, bits)
    chunks = b"fmt " + struct.pack("<I", len(fmt_body)) + fmt_body
    if extra_chunk:
        info = b"INFOISFT\x06\x00\x00\x00sstvae"
        chunks += b"LIST" + struct.pack("<I", len(info)) + info
    chunks += b"data" + struct.pack("<I", len(raw)) + raw + (b"\x00" if len(raw) % 2 else b"")
    return b"RIFF" + struct.pack("<I", 4 + len(chunks)) + b"WAVE" + chunks


def _wav_cases():
    rng = np.random.default_rng(5)
    n = 8000 * 3

    def pcm(dtype, ch, lo, hi):
        return rng.integers(lo, hi, size=(n, ch)).astype(dtype)

    def flt(dtype, ch):
        return (0.3 * rng.standard_normal((n, ch))).astype(dtype)

    return {
        "u8": (pcm(np.uint8, 1, 0, 256), 8000, 1, 8, False, False),
        "i16": (pcm(np.int16, 1, -32768, 32768), 8000, 1, 16, False, False),
        "i16-stereo": (pcm(np.int16, 2, -32768, 32768), 8000, 1, 16, False, False),
        "i24": (pcm(np.int32, 1, -(1 << 23), 1 << 23), 8000, 1, 24, False, False),
        "i32-ext-list": (pcm(np.int32, 2, -(1 << 31), 1 << 31), 8000, 1, 32, True, True),
        "f32": (flt(np.float32, 1), 8000, 3, 32, False, False),
        "f64-3ch": (flt(np.float64, 3), 8000, 3, 64, False, False),
        "i16-8ch": (pcm(np.int16, 8, -32768, 32768), 8000, 1, 16, False, False),
        "f32-9ch": (flt(np.float32, 9), 8000, 3, 32, True, False),
        "i16-11025": (pcm(np.int16, 1, -32768, 32768), 11025, 1, 16, False, False),
        "f32-44100": (flt(np.float32, 2), 44100, 3, 32, False, False),
        "i16-48000": (pcm(np.int16, 1, -32768, 32768), 48000, 1, 16, False, False),
    }


def test_capture_from_wav(native, row, tmp_path):
    """capture_from_wav, typed (the C++ reads the file), on every format
    and rate the module docstring lists: bit for bit wavio.read_wav then
    the native front end at 8 kHz, to 1e-12 of the scale resampled; the
    reference's capture to the front end's bar; q, t0_index and fs
    exactly, with start_unix given and not."""
    fe_native = native.qrss.frontend.audio_to_fe
    for name, (data, rate, fmt, bits, ext, extra) in _wav_cases().items():
        path = tmp_path / f"{name}.wav"
        path.write_bytes(_wav_bytes(data, rate, fmt, bits, ext, extra))
        x = wavio.read_wav(str(path))
        want = np.asarray(fe_native(x))
        for start in (None, 900.0 * Q_TEST + 1.0 - 9.5, np.float64(900.0 * Q_TEST - 3.0)):
            ref = RX.capture_from_wav(path, Q_TEST, start)
            got = row["capture_from_wav"](path, Q_TEST, start)
            assert type(got) is type(ref), name
            assert (got.q, got.t0_index, got.fs) == (ref.q, ref.t0_index, ref.fs), (name, start)
            assert type(got.q) is int and type(got.t0_index) is float and type(got.fs) is int, name
            assert got.fe.dtype == ref.fe.dtype == np.complex64 and got.fe.shape == ref.fe.shape, name
            e = P._rel(got.fe, want, name)
            P._worst("wav fe (C++ read vs read_wav)", e)
            assert (e == 0.0) if rate == 8000 else (e <= 1e-12), (name, e)
            assert P._rel(got.fe, ref.fe, name) <= 1e-5, name
    # str or Path, and a q that is a numpy integer
    path = tmp_path / "i16.wav"
    a = row["capture_from_wav"](str(path), np.int64(Q_TEST))
    assert a.q == Q_TEST and type(a.q) is int


def test_capture_from_wav_refusals(row, tmp_path):
    """A missing file and a directory: the reference's OSError subclass,
    errno and text; a file that is no WAV: ValueError, as scipy's (the
    text is the C++'s own)."""
    for path in (tmp_path / "nope.wav", tmp_path):
        ref = _outcome(RX.capture_from_wav, path, Q_TEST)
        got = _outcome(row["capture_from_wav"], path, Q_TEST)
        assert ref[0] in (FileNotFoundError, IsADirectoryError) and got == ref, (ref, got)
        try:
            row["capture_from_wav"](path, Q_TEST)
        except OSError as e:
            assert e.errno == (2 if path.name == "nope.wav" else 21) and e.filename == str(path)
    txt = tmp_path / "text.wav"
    txt.write_text("hello, this is not a WAV file at all")
    assert _outcome(RX.capture_from_wav, txt, Q_TEST)[0] is ValueError
    assert _outcome(row["capture_from_wav"], txt, Q_TEST)[0] is ValueError


def test_receive_wav(native, Rs, tmp_path_factory):
    """receive_wav on the TINY WAV, every row in place: the C++ throughout
    (asserted), the reference's pass."""
    path = _wav_path(str(tmp_path_factory.mktemp("wav")))
    for spec in (frame.TINY,):
        ref = RX.receive_wav(path, Q_TEST, spec=spec)
        assert len(ref) == 1, spec
        seen = []
        with pytest.MonkeyPatch.context() as mp:
            P._apply_rows(native, mp)
            m = native.qrss.receiver_slot
            orig = m.receive_wav
            mp.setattr(m, "receive_wav", lambda *a: seen.append(a) or orig(*a))
            got = RX.receive_wav(path, Q_TEST, spec=spec)
        assert len(seen) == 1, spec
        _slot_close(got, ref, ("receive_wav", spec))


# --- threads ---------------------------------------------------------------------------------------

def test_threads(native, Rs):
    """Four slots received at once, typed (the GIL released): each the
    slot alone gives."""
    from concurrent.futures import ThreadPoolExecutor

    labels = ["ce-tiny", "l-tiny", "joint", "weak-neighbour"]
    refs = {k: _reference(k) for k in labels}
    with pytest.MonkeyPatch.context() as mp:
        P._apply_rows(native, mp)
        want = {k: RX.receive_slot(refs[k].cap, refs[k].spec) for k in labels}
        with ThreadPoolExecutor(4) as ex:
            got = list(ex.map(lambda k: RX.receive_slot(refs[k].cap, refs[k].spec), labels))
    for k, g in zip(labels, got):
        assert [p.uid for p in g] == [p.uid for p in want[k]], k

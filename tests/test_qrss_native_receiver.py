"""Parity of native/core/qrss/receiver (sstvae_native.qrss.receiver) with
the per-pass half of sstvae/qrss/receiver.py: prepare, frames, is_joint,
_single, centre_hz, _centres, _ce_hints, channel_for, _psi_frame,
_cw_result, receive_pass (with _receive_pass_l, which the C++ folds into
it), decode_header, _decode_header_l, detection_from_pass,
capture_from_pass, mean_w_db and confidence_db.
tests/qrss_native/receiver.py substitutes them under --native-qrss=rx.

The passes are simulated ones (channel.simulate): CE TINY and SHORT, L
l-tiny and l-short, at -6 and -12 dB, with a 1 Hz/min drift, +100 ppm
(transmitter) and -100 ppm (receiver) clocks, a 10 s lead-in and (SHORT)
OOK callsign keying, each through receive_pass at a flat 1500 Hz
candidate -- gate V first, as the EM and live callers may hand it an
unverified detection. The C++ is called on its own (no seams) and set
against the reference run with every primitive Python. Every SHORT pass
against the pure reference is in the slow tier (CLAUDE.md), as are half
the TINY variants: run them by node id (-m slow -k test_receive_pass).

The contract (the bar), and in brackets the worst measured over every
scenario, slow tier included (2026-10-13; QRSS_PARITY_WORST=1 with -s
prints them):

- receive_pass: the same pass or none, the same header and frame, the
  callsign reading's text, keying and verdict exactly and z_match to 1e-6
  relative [0: bit-equal]; f_hz to 1e-6 Hz [1.4e-12 Hz]; tau0 to 1e-6 s
  [5.6e-12 CH samples]; ppm to 1e-6 [3.3e-10 relative]; z and w to 1e-6
  of the array's scale [z 3.4e-8, w 0] and element by element relative
  [z 5.7e-7, w 0 -- z's is 5 to 10 float32 ulps of one element, the
  margin under this bar is less than 2x; the scale-relative one is the
  contract]; hdr_llr, psi and ch to 1e-6 of the array's scale [0]; every
  report field to 1e-6 relative [z_ref 3.2e-8 (7.9e-8 through the
  callers), the rest under 1e-11] and
  its flags exactly; ch_t0_index, f_mix_hz, q, frame, waveform, estimator
  exactly (uid is z's float32 hash: its q and frequency fields only).
  With return_tracks the same tracks, round A's and B's, each with the
  same q, keying, positions and decisions, and every array to the
  tracker lanes' own bars (TRACK_BAR) [u 1.7e-10, P 1.4e-15]. FULL passes
  (CE and L at -12 dB, not a test: 30 s of Python each) met the same bars
  [z 2.2e-8, w 6.8e-8, f_hz 4.5e-13 Hz];
- the swap test (test_receive_pass_composition): the reference's
  receive_pass with the native rows of what it calls assigned over the
  names it calls is the C++ receive_pass bit for bit, every field of the
  pass and of its tracks -- once with every receiver function the
  reference's (their glue too) over the native primitives, once with
  only receive_pass the reference's over the receiver's rows; on SHORT
  the header decodes and round B runs with the callsign keying known, and
  once with round_b off. So every difference above is a primitive's,
  measured in that primitive's own parity test;
- decode_header and _decode_header_l: bitwise -- the header, the LLRs kept
  (the caller's own array, or the block's in its dtype) and the frame
  gone on with (the caller's own object, or frame.plain) -- on noisy LLR
  vectors over every kind of frame, with the version 1 fallback reached:
  against the pure-Python reference on 60 (2,000 in the slow tier) and
  against the reference's glue over the native polar decoder on 300
  (2,000 in the slow tier);
- confidence_db and mean_w_db: to the bit (the bar is 1e-12) on random
  passes of every frame, elapsed times and bin counts, with psi of the
  wrong length and absent, and numpy's refusals' texts for n <= 0;
- prepare, channel_for, _psi_frame, _cw_result, detection_from_pass,
  capture_from_pass, frames, is_joint, _single, centre_hz, _centres and
  _ce_hints on the receiver's own inputs, the objects they hand back
  included (a Prepared passes through, frames returns the caller's
  frames, capture_from_pass the pass's own arrays);
- the seams: a receiver-level name holding a test's spy is called, with
  the reference's arguments, by the native receive_pass;
- the whole caller, every row of every tx and rx rows file over its
  target and every from-import site, the receiver's rows counted (the
  counters are what the module attributes hold): receive_slot on a CE
  and a joint capture (a CE neighbour 60 Hz above an L signal), an L one
  and a joint (SHORT, L_SHORT) one in the slow tier, em.rereceive and
  live's tile update; the same passes, and every row of this file
  reached in one configuration or another;
- the reference's refusals, type and text.
"""

import dataclasses
import functools
import math
import os
import sys
import time
from fractions import Fraction
from types import SimpleNamespace

import numpy as np
import pytest
from qrss_demod_tracks import CALL, HDR, flat_det
from qrss_fakes import l_scenario, make_pass_result, noisy_llr
from qrss_helpers import Q_TEST, unit_rms_latents
from qrss_native import _rx, require

import qrss_native
from sstvae.qrss import channel as chm
from sstvae.qrss import em, frame, frontend, header, lframe, live, morse
from sstvae.qrss import ltrack as LT
from sstvae.qrss import receiver as RX
from sstvae.qrss import track as T
from sstvae.qrss.channel import ChannelConfig, Neighbour
from sstvae.qrss.constants import CH_FS, WAVEFORM_CE, WAVEFORM_L
from sstvae.qrss.precoder import precode
from sstvae.qrss.types import EmPrior

RECEIVER = "sstvae.qrss.receiver"
WORST: dict = {}

# the tracker lanes' bars (tests/test_qrss_native_track.py, ltrack.py): the
# arrays of a track to 1e-6 of their scale, P to 1e-4 (the Kalman/RTS
# conditioning: a 1-ulp change of an input moves the reference's own P by
# ~1e-6 relative, 2e-5 on a drifting pass)
TRACK_BAR = {"z_d": 1e-6, "m": 1e-6, "c": 1e-6, "N": 1e-6, "psi": 1e-6, "u": 1e-6, "pu": 1e-6, "P": 1e-4,
             "m_pl": 1e-6, "c_pl": 1e-6, "N_pl": 1e-6, "psi_pl": 1e-6, "u_pl": 1e-6, "P_pl": 1e-4}
PASS_BAR = 1e-6


def _worst(key, err):
    WORST[key] = max(WORST.get(key, 0.0), float(err))


@pytest.fixture(scope="module", autouse=True)
def _print_worst():
    yield
    if os.environ.get("QRSS_PARITY_WORST"):
        for k in sorted(WORST):
            print(f"worst {k}: {WORST[k]:.3g}")


@pytest.fixture(scope="module")
def Rc(native):
    return require(native, "receiver")


@pytest.fixture(scope="module")
def row(native, Rc):
    from qrss_native.receiver import rows

    return {name: fn for (_t, name), fn in rows(native.qrss).items()}


# --- the scenarios ----------------------------------------------------------------------------------

# label: (waveform, frame, SNR dB, seed, further ChannelConfig fields, lead-in s, ook)
SCENARIOS = {
    "tiny-6": (WAVEFORM_CE, "tiny", -6.0, 1, (), 0.0, False),
    "tiny-12": (WAVEFORM_CE, "tiny", -12.0, 2, (), 0.0, False),
    "short-6": (WAVEFORM_CE, "short", -6.0, 1, (), 0.0, False),
    "short-12": (WAVEFORM_CE, "short", -12.0, 3, (), 0.0, False),
    "short-ook": (WAVEFORM_CE, "short", -6.0, 9, (), 0.0, True),
    "drift": (WAVEFORM_CE, "tiny", -6.0, 4, (("drift_hz_per_min", 1.0),), 0.0, False),
    "ppm+100": (WAVEFORM_CE, "tiny", -9.0, 5, (("tx_ppm", 100.0),), 0.0, False),
    "ppm-100": (WAVEFORM_CE, "tiny", -9.0, 6, (("rx_ppm", -100.0),), 0.0, False),
    "lead": (WAVEFORM_CE, "tiny", -6.0, 8, (), 10.0, False),
    "l-tiny-6": (WAVEFORM_L, "l-tiny", -6.0, 1, (), 0.0, False),
    "l-tiny-12": (WAVEFORM_L, "l-tiny", -12.0, 2, (), 0.0, False),
    "l-short-6": (WAVEFORM_L, "l-short", -6.0, 1, (), 0.0, False),
    "l-short-12": (WAVEFORM_L, "l-short", -12.0, 3, (), 0.0, False),
    "l-drift": (WAVEFORM_L, "l-tiny", -6.0, 4, (("drift_hz_per_min", 1.0),), 0.0, False),
    "l-ppm+100": (WAVEFORM_L, "l-tiny", -9.0, 5, (("tx_ppm", 100.0),), 0.0, False),
    "l-ppm-100": (WAVEFORM_L, "l-tiny", -9.0, 6, (("rx_ppm", -100.0),), 0.0, False),
    "l-lead": (WAVEFORM_L, "l-tiny", -6.0, 8, (), 10.0, False),
}
# SHORT passes are the slow tier's (CLAUDE.md: test_receive_pass_composition runs short-6 and
# l-short-6, header, callsign windows and round B, bit for bit against the reference's glue in the
# fast tier), and so are half the TINY variants, to keep this file's fast tier near 40 s
SLOW = {"short-6", "short-12", "short-ook", "l-short-6", "l-short-12", "tiny-12", "l-tiny-12", "ppm-100",
        "l-ppm+100", "l-drift"}


def _labels(only=None):
    out = []
    for label in SCENARIOS:
        if only is not None and label not in only:
            continue
        out.append(pytest.param(label, marks=pytest.mark.slow) if label in SLOW else label)
    return out


@functools.lru_cache(maxsize=None)
def _ce_sim(name, snr, seed, cfg, lead_in, ook):
    spec = frame.get(name)
    a = unit_rms_latents(spec.n_data, seed)
    sym = frame.assemble(spec, header.encode(HDR) if spec.n_hdr else None, precode(a, Q_TEST))
    sim = chm.simulate(sym, spec, Q_TEST, ChannelConfig(snr_db=snr, seed=seed, **dict(cfg)), carrier_hz=1500.0,
                       lead_in_s=lead_in, ook=ook, keying=morse.keying_units(CALL) if spec.n_win else None)
    return spec, sim


def _sim(label):
    wf, name, snr, seed, cfg, lead, ook = SCENARIOS[label]
    if wf == WAVEFORM_CE:
        return _ce_sim(name, snr, seed, cfg, lead, ook)
    _, sim = l_scenario(name, snr, seed, lead_in=lead, cfg=cfg)
    return lframe.get(name), sim


@functools.lru_cache(maxsize=None)
def _capture(label):
    """The scenario's capture prepared by the reference and its flat
    candidate (build it before any row is applied: it is the reference's)."""
    wf, lead = SCENARIOS[label][0], SCENARIOS[label][5]
    spec, sim = _sim(label)
    cap = frontend.Capture(sim.fe, Q_TEST, sim.t0_index)
    prep = RX.prepare(cap)
    det = dataclasses.replace(flat_det(1500.0, wf), lead_in_s=lead)
    return SimpleNamespace(spec=spec, sim=sim, cap=cap, prep=prep, det=det)


@functools.lru_cache(maxsize=None)
def _case(label):
    """_capture's, and the reference's receive_pass on it (with its tracks)."""
    c = _capture(label)
    t = time.perf_counter()
    ref = RX.receive_pass(c.prep, c.spec, c.det, return_tracks=True)
    return SimpleNamespace(**vars(c), ref=ref, t_ref=time.perf_counter() - t)


@functools.lru_cache(maxsize=None)
def _native_case(label, Rc):
    """The C++ receive_pass (no seams) on _capture's, with its tracks: for
    the tests that need a pass or a tracker, not the reference's run."""
    c = _capture(label)
    return _native_pass(Rc, c.prep, c.spec, c.det, return_tracks=True)


def _native_pass(Rc, cap, spec, det, **kw):
    """The C++ receive_pass on its own (no seams), as the row returns it."""
    r = Rc.receive_pass(cap, spec, det, **kw)
    if r is None or not kw.get("return_tracks"):
        return _rx.to_ref(r)
    p, trs = r
    chan = _rx.to_ref(trs[0]["chan"])
    tracks = []
    for d in trs:
        d["chan"] = chan
        tracks.append(_rx.to_ref(d))
    pr = _rx.to_ref(p)
    pr.timing = tracks[-1].timing
    return pr, tracks


# --- comparison --------------------------------------------------------------------------------

def _rel(a, b, what):
    """max |a - b| over max(1, max |b|), the same shape and dtype; NaN and
    infinities only where the other has them."""
    a, b = np.asarray(a), np.asarray(b)
    assert a.shape == b.shape, (what, a.shape, b.shape)
    assert a.dtype == b.dtype, (what, a.dtype, b.dtype)
    if a.size == 0:
        return 0.0
    if a.dtype == bool or a.dtype.kind in "iu":
        assert np.array_equal(a, b), what
        return 0.0
    ac, bc = a.astype(np.complex128), b.astype(np.complex128)
    same = (ac == bc) | (np.isnan(ac) & np.isnan(bc))
    if same.all():
        return 0.0
    assert np.isfinite(ac[~same]).all() and np.isfinite(bc[~same]).all(), what
    fin = np.isfinite(bc)
    scale = max(1.0, float(np.max(np.abs(bc[fin])))) if fin.any() else 1.0
    return float(np.max(np.abs(ac[~same] - bc[~same]))) / scale


def _elementwise(a, b):
    """max |a - b| / |b| over the elements (equal ones, zeros included, 0)."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    d = a != b
    if not d.any():
        return 0.0
    return float(np.max(np.abs(a[d] - b[d]) / np.abs(b[d])))


def _num(x, y, tol, what, key=None):
    """|x - y| <= tol max(1, |y|) (NaN with NaN)."""
    x, y = float(x), float(y)
    if math.isnan(y) or math.isnan(x):
        assert math.isnan(x) and math.isnan(y), (what, x, y)
        return
    err = abs(x - y) / max(1.0, abs(y))
    if key:
        _worst(key, err)
    assert err <= tol, (what, x, y)


def _abs(x, y, tol, what, key):
    """|x - y| <= tol, absolute."""
    err = abs(float(x) - float(y))
    _worst(key, err)
    assert err <= tol, (what, x, y)


def _timing_close(g, r, what, key):
    # tau0 to 1e-6 s, absolute (it is in CH samples)
    _abs(g.tau0, r.tau0, 1e-6 * CH_FS, (what, "tau0"), f"{key} timing.tau0 (CH samples, abs)")
    for f in ("ppm", "gamma", "z"):
        _num(getattr(g, f), getattr(r, f), PASS_BAR, (what, f), f"{key} timing.{f}")
    e = _rel(np.asarray(g.cov, np.float64), np.asarray(r.cov, np.float64), (what, "cov"))
    _worst(f"{key} timing.cov", e)
    assert e <= 1e-4, (what, "cov", e)


def _report_close(g, r, what, key):
    assert (g is None) == (r is None), what
    if r is None:
        return
    for f in ("offset_hz", "drift_hz_per_min", "wander_hz_rms", "doppler_hz", "ppm", "snr2500_db", "z_ref",
              "kappa"):
        _num(getattr(g, f), getattr(r, f), PASS_BAR, (what, f), f"{key} report.{f}")
    assert (g.slip_free, g.suspect) == (r.slip_free, r.suspect), what


def _pass_close(g, r, what, key="pass", bar=PASS_BAR):
    assert (g is None) == (r is None), what
    if r is None:
        return
    assert type(g) is type(r), what
    assert (g.q, g.frame, g.waveform, g.estimator, g.em_round, g.ch_fs) == \
        (r.q, r.frame, r.waveform, r.estimator, r.em_round, r.ch_fs), what
    assert type(g.q) is int and type(g.ch_fs) is int and type(g.waveform) is int, what
    assert g.uid.rsplit("_", 1)[0] == r.uid.rsplit("_", 1)[0], what
    assert g.header == r.header, what
    assert (g.cw is None) == (r.cw is None), what
    if r.cw is not None:
        assert (g.cw.text, g.cw.keying, g.cw.agrees) == (r.cw.text, r.cw.keying, r.cw.agrees), what
        _num(g.cw.z_match, r.cw.z_match, bar, (what, "z_match"), f"{key} cw.z_match")
    _abs(g.f_hz, r.f_hz, 1e-6, (what, "f_hz"), f"{key} f_hz (Hz, abs)")
    _timing_close(g.timing, r.timing, what, key)
    _report_close(g.report, r.report, what, key)
    for f in ("z", "w", "hdr_llr", "psi", "ch"):
        e = _rel(getattr(g, f), getattr(r, f), (what, f))
        _worst(f"{key} {f}", e)
        assert e <= bar, (what, f, e)
    # the latents and their weights element by element, relative (float32)
    for f in ("z", "w"):
        e = _elementwise(getattr(g, f), getattr(r, f))
        _worst(f"{key} {f} (elementwise)", e)
        assert e <= bar, (what, f, "elementwise", e)
    assert g.ch_t0_index == r.ch_t0_index and type(g.ch_t0_index) is float, what
    assert g.f_mix_hz == r.f_mix_hz and isinstance(g.f_mix_hz, Fraction), what


def _tracks_close(gt, rt, what, key="track"):
    assert len(gt) == len(rt), what
    for k, (a, b) in enumerate(zip(gt, rt)):
        w = (what, "AB"[k] if k < 2 else k)
        assert type(a) is type(b), w
        assert a.spec == b.spec, w
        assert np.array_equal(a.pos, b.pos) and a.pos.dtype == b.pos.dtype, w
        assert a.q == b.q and a.ook == b.ook and a.lead_in_s == b.lead_in_s, w
        assert (a.keying is None) == (b.keying is None), w
        if b.keying is not None:
            assert np.array_equal(a.keying, b.keying), w
        assert np.array_equal(a.meas, b.meas), w
        _num(a.z_ref, b.z_ref, PASS_BAR, (w, "z_ref"), f"{key} z_ref")
        _num(a.loglik, b.loglik, PASS_BAR, (w, "loglik"), f"{key} loglik")
        _num(a.df_max, b.df_max, PASS_BAR, (w, "df_max"), f"{key} df_max")
        _timing_close(a.timing, b.timing, w, key)
        t = np.linspace(-10.0, float(b.t_pos_s(b.pos[-1:])[0]), 64)
        e = float(np.max(np.abs(np.asarray(a.freq(t)) - np.asarray(b.freq(t)))))
        _worst(f"{key} freq", e)
        assert e <= PASS_BAR, (w, "freq", e)
        for f, bar in TRACK_BAR.items():
            if not hasattr(b, f):
                continue
            x, y = getattr(a, f), getattr(b, f)
            assert (x is None) == (y is None), (w, f)
            if y is None:
                continue
            e = _rel(x, y, (w, f))
            _worst(f"{key} {f}", e)
            assert e <= bar, (w, f, e)
        if isinstance(b, LT.LTrackResult):
            assert np.array_equal(a.meas_pl, b.meas_pl), w
            assert abs(complex(a.delta) - complex(b.delta)) <= PASS_BAR, w
            _num(a.resid_ratio, b.resid_ratio, PASS_BAR, (w, "resid_ratio"))
        assert (a.report is None) == (b.report is None), w
        assert (a.classes is None) == (b.classes is None), w
    for a in gt[1:]:
        assert a.chan is gt[0].chan, what


# --- receive_pass ------------------------------------------------------------------------------

@pytest.mark.parametrize("label", _labels())
def test_receive_pass(Rc, label):
    """The C++ receive_pass against the reference's on the scenario, with
    its tracks."""
    c = _case(label)
    t = time.perf_counter()
    got = _native_pass(Rc, c.prep, c.spec, c.det, return_tracks=True)
    t_cpp = time.perf_counter() - t
    if os.environ.get("QRSS_PARITY_WORST"):
        print(f"time {label}: reference {c.t_ref:.2f} s, C++ {t_cpp:.2f} s")
    ref = c.ref
    assert (got is None) == (ref is None), label
    if ref is None:
        return
    _pass_close(got[0], ref[0], label)
    _tracks_close(got[1], ref[1], label)
    assert got[0].timing is got[1][-1].timing
    if c.spec.has_header and SCENARIOS[label][2] > -7:
        assert ref[0].header is not None and len(ref[1]) == 2, label   # round B reached


@pytest.mark.parametrize("label", ["tiny-6", "l-tiny-6"])
def test_receive_pass_options(Rc, label):
    """The keyword arguments: estimator, exclude_hz, erase_s (a span, two,
    and one past the end), a prior, round_b (moot on a frame without a
    header: test_receive_pass_composition runs SHORT with it off), and
    return_tracks off (the pass alone, the same)."""
    c = _case(label)
    n = c.spec.n_data
    prior = EmPrior(x_hat=np.where(np.arange(n) % 3 == 0, 0.5, -0.25), v=np.full(n, 0.75))
    for kw in (dict(estimator="plain", exclude_hz=(1530.0, 1471.5), erase_s=((10.0, 20.0), (40.0, 41.0))),
               dict(exclude_hz=[1525.0], erase_s=[(30.0, 1e6)], prior=prior, round_b=False)):
        r = RX.receive_pass(c.prep, c.spec, c.det, return_tracks=True, **kw)
        g = _native_pass(Rc, c.prep, c.spec, c.det, return_tracks=True, **kw)
        what = (label, tuple(kw))
        assert (g is None) == (r is None), what
        if r is not None:
            _pass_close(g[0], r[0], what)
            _tracks_close(g[1], r[1], what)
    _pass_close(_native_pass(Rc, c.prep, c.spec, c.det), c.ref[0], (label, "no tracks"))


@pytest.mark.slow
def test_receive_pass_round_a_only_short(Rc):
    c = _case("short-6")
    _pass_close(_native_pass(Rc, c.prep, c.spec, c.det, round_b=False),
                RX.receive_pass(c.prep, c.spec, c.det, round_b=False), "short round A")


@pytest.mark.parametrize("label", ["tiny-6", "l-tiny-6"])
def test_receive_pass_raw_capture(Rc, label):
    """A raw frontend.Capture: the C++ prepares it (blank, normalise) on the way."""
    c = _case(label)
    _pass_close(_native_pass(Rc, c.cap, c.spec, c.det), c.ref[0], label)


@pytest.mark.parametrize("label", ["tiny-6", "l-tiny-6", pytest.param("short-6", marks=pytest.mark.slow),
                                   pytest.param("l-short-6", marks=pytest.mark.slow)])
def test_receive_pass_stored(Rc, label):
    """EM's re-receive: the pass's own CH capture (capture_from_pass) at its
    own detection (detection_from_pass, verified: no gate V) and f_mix_hz;
    and the same detection unverified (gate V on the CH capture)."""
    c = _case(label)
    p = c.ref[0]
    prep, det = RX.capture_from_pass(p), RX.detection_from_pass(p)
    for d in (det, dataclasses.replace(det, timing=None, z_ref=math.nan)):
        r = RX.receive_pass(prep, c.spec, d, f_mix_hz=p.f_mix_hz, return_tracks=True)
        g = _native_pass(Rc, prep, c.spec, d, f_mix_hz=p.f_mix_hz, return_tracks=True)
        what = (label, d.timing is None)
        assert (g is None) == (r is None), what
        if r is not None:
            _pass_close(g[0], r[0], what)
            _tracks_close(g[1], r[1], what)
    # the default mix at the CH rate: the detection's frequency on the mix grid
    _pass_close(_native_pass(Rc, prep, c.spec, det), RX.receive_pass(prep, c.spec, det), (label, "default mix"))


def _apply_rows(native, mp, skip=(), wrap=None):
    """Every row of every tx and rx rows file over its target and every
    from-import site in sstvae.qrss, but those in `skip` ((target, name)
    pairs); `wrap(key, fn)` may replace each row first."""
    q = native.qrss
    reference = qrss_native._import_reference()
    replaced = {}
    for _name, mod in qrss_native._row_modules():
        if mod.STAGE not in ("tx", "rx"):
            continue
        for key, repl in mod.rows(q).items():
            if key in skip:
                continue
            if wrap is not None:
                repl = wrap(key, repl)
            owner = qrss_native._owner(key[0])
            orig = getattr(owner, key[1])
            mp.setattr(owner, key[1], repl)
            if callable(orig) or hasattr(orig, "__array__"):
                replaced[id(orig)] = (orig, repl)
    for m in reference:
        for k, v in list(vars(m).items()):
            hit = replaced.get(id(v))
            if hit is not None and hit[0] is v:
                mp.setattr(m, k, hit[1])


def _same(a, b, what):
    a, b = np.asarray(a), np.asarray(b)
    assert a.dtype == b.dtype and a.shape == b.shape, what
    assert a.tobytes() == b.tobytes(), what


def _bitwise_pass(g, h, what):
    """The C++ pass and tracks (g) against the reference's (h), bit for bit."""
    assert (g is None) == (h is None), what
    if h is None:
        return
    gp, hp = g[0], h[0]
    for f in ("uid", "q", "frame", "waveform", "f_hz", "header", "cw", "ch_fs", "ch_t0_index", "f_mix_hz",
              "estimator", "em_round", "report"):
        assert getattr(gp, f) == getattr(hp, f), (what, f)
    for f in ("z", "w", "hdr_llr", "ch", "psi"):
        _same(getattr(gp, f), getattr(hp, f), (what, f))
    for f in ("tau0", "ppm", "gamma", "z", "cov"):
        _same(getattr(gp.timing, f), getattr(hp.timing, f), (what, "timing", f))
    assert len(g[1]) == len(h[1]), what
    for a, b in zip(g[1], h[1]):
        for f in ("pos", "z_d", "m", "c", "u", "P", "N", "psi", "meas", "pu", "keying"):
            assert (getattr(a, f) is None) == (getattr(b, f) is None), (what, f)
            if getattr(b, f) is not None:
                _same(getattr(a, f), getattr(b, f), (what, f))
        for f in ("q", "loglik", "ook", "lead_in_s", "df_max", "report", "spec"):
            assert getattr(a, f) == getattr(b, f), (what, f)
        assert a.z_ref == b.z_ref or (math.isnan(a.z_ref) and math.isnan(b.z_ref)), what
        _same(a.freq(np.linspace(-10.0, 2000.0, 41)), b.freq(np.linspace(-10.0, 2000.0, 41)), (what, "freq"))
        if isinstance(b, LT.LTrackResult):
            for f in ("m_pl", "c_pl", "N_pl", "psi_pl", "meas_pl", "u_pl", "P_pl"):
                _same(getattr(a, f), getattr(b, f), (what, f))
            assert a.delta == b.delta, what


def _receiver_rows(native):
    from qrss_native.receiver import rows

    return {k for k in rows(native.qrss)}


# label: the variants run -- (which receiver rows the reference keeps, its capture, kwargs)
COMPOSITION = {
    "tiny-6": (("all", "prep", {}), ("receive_pass", "cap", {})),
    "drift": (("all", "prep", {}),),
    "short-6": (("all", "prep", {}), ("receive_pass", "prep", dict(round_b=False))),
    "l-tiny-6": (("all", "prep", {}), ("receive_pass", "cap", {})),
    "l-ppm+100": (("all", "prep", {}),),
    "l-short-6": (("all", "prep", {}), ("receive_pass", "prep", dict(round_b=False))),
}


@pytest.mark.parametrize("label", list(COMPOSITION))
def test_receive_pass_composition(native, Rc, label):
    """The swap test. The reference's receive_pass (and _receive_pass_l)
    with the native rows of what it calls assigned over the names it calls
    is the C++ receive_pass bit for bit, every field of the pass and of its
    tracks. Two ways: "all" keeps every receiver function the reference's
    (prepare, channel_for, decode_header, _cw_result, _psi_frame: their
    glue too) over the native primitives (track, ltrack, demod, cwid,
    header, frontend, ...); "receive_pass" keeps only receive_pass, over
    the receiver's own rows. On SHORT the header decodes and round B runs
    with the callsign keying known (asserted), and once with round_b off."""
    c = _capture(label)
    for keep, which, kw in COMPOSITION[label]:
        what = (label, keep, which, tuple(kw))
        skip = _receiver_rows(native) if keep == "all" else {(RECEIVER, "receive_pass")}
        cap = c.prep if which == "prep" else c.cap
        with pytest.MonkeyPatch.context() as mp:
            _apply_rows(native, mp, skip=skip)
            h = RX.receive_pass(cap, c.spec, c.det, return_tracks=True, **kw)
        g = _native_pass(Rc, cap, c.spec, c.det, return_tracks=True, **kw)
        _bitwise_pass(g, h, what)
        if c.spec.has_header and kw.get("round_b", True):
            # round B ran, with the header and the callsign keying known
            assert h[0].header is not None and len(h[1]) == 2 and h[1][1].keying is not None, what
            assert h[0].cw is not None and h[0].cw.agrees, what


# --- seams -------------------------------------------------------------------------------------

def _counter(fn, calls, name):
    def spy(*a, **k):
        calls[name] = calls.get(name, 0) + 1
        return fn(*a, **k)
    return spy


@pytest.mark.parametrize("label", ["tiny-6", "l-tiny-6"])
def test_seams_reach_spies(row, label, monkeypatch):
    """With the reference's (spied) prepare, channel_for, verify / verify_l
    and noise_psd in the module attributes, the row's receive_pass calls
    each, with the reference's arguments, and makes the reference's pass of
    what they return: the seams are resolved at each call."""
    c = _case(label)
    calls: dict = {}
    import sstvae.qrss.frontend as F

    monkeypatch.setattr(RX, "prepare", _counter(RX.prepare, calls, "prepare"))
    monkeypatch.setattr(RX, "channel_for", _counter(RX.channel_for, calls, "channel_for"))
    monkeypatch.setattr(T, "verify", _counter(T.verify, calls, "verify"))
    monkeypatch.setattr(LT, "verify_l", _counter(LT.verify_l, calls, "verify_l"))
    monkeypatch.setattr(F, "noise_psd", _counter(F.noise_psd, calls, "noise_psd"))
    got = row["receive_pass"](c.cap, c.spec, c.det, return_tracks=True)
    L = c.spec.waveform == WAVEFORM_L
    assert calls.get("prepare") == 1 and calls.get("channel_for") == 1, calls
    assert calls.get("verify_l" if L else "verify") == 1, calls
    assert ("verify" if L else "verify_l") not in calls, calls
    assert calls.get("noise_psd", 0) >= 1, calls
    _pass_close(got[0], c.ref[0], label)
    _tracks_close(got[1], c.ref[1], label)
    # a seam's channel is the reference's ChanCapture; the tracks share it
    assert isinstance(got[1][0].chan, T.ChanCapture) and got[1][0].chan._psd is not None
    # a gate that refuses: None
    monkeypatch.setattr(LT if L else T, "verify_l" if L else "verify", lambda *a, **k: None)
    assert row["receive_pass"](c.prep, c.spec, c.det) is None


def test_seam_prepare_replaced(row, monkeypatch):
    """A test's prepare returning a Prepared of its own: the native
    receive_pass receives that capture."""
    c = _case("tiny-6")
    seen = []

    def prep(cap):
        seen.append(cap)
        return c.prep

    monkeypatch.setattr(RX, "prepare", prep)
    got = row["receive_pass"](c.cap, c.spec, c.det)
    assert seen == [c.cap]
    _pass_close(got, c.ref[0], "prepare replaced")


# --- decode_header ------------------------------------------------------------------------------

DH_FRAMES = ("short", "full", "short-v1", "full-v1", "full-spreadonly", "tiny", "l-short", "l-full", "l-tiny")
CALLS = ("K1ABC", "W1AW", "VK2XYZ/P", "G4ABC")
GRIDS = ("FN42", None, "IO91")


def _random_header(rng, version):
    mode = int(rng.integers(0, 3))
    return header.HeaderFields(callsign=str(rng.choice(CALLS)), grid=GRIDS[int(rng.integers(0, 3))],
                               picture_id=int(rng.integers(0, 2 ** 32)), mode=mode,
                               segment=int(rng.integers(0, mode + 1)), codec_id=int(rng.integers(0, 2 ** 16)),
                               version=version)


def _header_cases(n, seed):
    """n (llr, diag, spec) triples: every kind of frame; a sum that decodes,
    that is noise, or that is too weak; the spread copy and the block each
    absent, None, noise or a header of either version at an SNR that
    mostly decodes -- weighted so the fallbacks are reached."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        spec = lframe.get_any(str(rng.choice(DH_FRAMES)))
        h2, h1 = _random_header(rng, 2), _random_header(rng, 1)
        u = rng.random()
        if u < 0.3:
            llr = noisy_llr(h1 if rng.random() < 0.3 else h2, float(rng.uniform(-3.0, 3.0)), rng)
        elif u < 0.6:
            llr = noisy_llr(h2, float(rng.uniform(-14.0, -8.0)), rng)
        else:
            llr = noisy_llr(None, None, rng) + rng.standard_normal(2474).astype(np.float32)
        if rng.random() < 0.3:
            llr = llr.astype(np.float64)
        diag = {}
        for key in ("llr_spread", "llr_block"):
            v = rng.random()
            if v < 0.15:
                continue
            if v < 0.25:
                diag[key] = None
            elif v < 0.45:
                diag[key] = rng.standard_normal(2474) * 2.0
            else:
                h = h1 if rng.random() < 0.5 else h2
                part = noisy_llr(h, float(rng.uniform(-2.0, 3.0)), rng)
                diag[key] = part if rng.random() < 0.5 else part.astype(np.float64)
        out.append((llr, diag, spec))
    return out


def _decode_same(row, cases):
    """The row's decode_header and _decode_header_l against the module's
    (the reference, or the reference over the native polar decoder);
    returns how many took each path."""
    paths = {"sum": 0, "fallback": 0, "v1 block": 0, "none": 0}
    for i, (llr, diag, spec) in enumerate(cases):
        r = RX.decode_header(llr, diag, spec)
        g = row["decode_header"](llr, diag, spec)
        what = (i, spec.name)
        assert g[0] == r[0], what
        if r[1] is llr:
            assert g[1] is llr, what
        else:
            assert g[1].dtype == r[1].dtype and np.array_equal(g[1], r[1]), what
            assert (g[1] is diag["llr_block"]) == (r[1] is diag["llr_block"]), what
        if r[2] is spec:
            assert g[2] is spec, what
        else:
            assert g[2] == r[2] and type(g[2]) is type(r[2]), what
            paths["v1 block"] += 1
        if spec.waveform == WAVEFORM_L:
            assert row["_decode_header_l"](llr, diag, spec) == RX._decode_header_l(llr, diag, spec), what
        if r[0] is None:
            paths["none"] += 1
        elif header.decode(np.asarray(llr, np.float64), versions=(1, 2)) is not None:
            paths["sum"] += 1
        else:
            paths["fallback"] += 1
    return paths


def test_decode_header_reference(row):
    """60 cases against the pure-Python reference."""
    paths = _decode_same(row, _header_cases(60, 11))
    assert paths["v1 block"] >= 1 and paths["fallback"] >= 3 and paths["sum"] >= 9, paths


@pytest.mark.slow
def test_decode_header_reference_2000(row):
    paths = _decode_same(row, _header_cases(2000, 12))
    assert paths["v1 block"] >= 30, paths


def _decode_native_polar(native, row, n, seed):
    """n cases against the reference's glue over the native polar decoder
    (header.decode's row, whose own parity test holds it to the reference
    bit for bit): the decision, the copies tried and their order, the LLRs
    and frame handed back."""
    from qrss_native.header_rx import rows as h_rows

    with pytest.MonkeyPatch.context() as mp:
        for (t, name), fn in h_rows(native.qrss).items():
            mp.setattr(qrss_native._owner(t), name, fn)
        return _decode_same(row, _header_cases(n, seed))


def test_decode_header_300(native, row):
    paths = _decode_native_polar(native, row, 300, 14)
    assert paths["v1 block"] >= 5 and paths["fallback"] >= 15 and paths["sum"] >= 45, paths


@pytest.mark.slow
def test_decode_header_2000(native, row):
    paths = _decode_native_polar(native, row, 2000, 13)
    assert paths["v1 block"] >= 30 and paths["fallback"] >= 100 and paths["sum"] >= 300, paths


def test_decode_header_diag_reads(row):
    """diag is read only past a failed sum, by .get, as the reference reads
    it; a frame without a header never reads llr."""
    class Diag(dict):
        reads: list = []

        def get(self, k, d=None):
            self.reads.append(k)
            return super().get(k, d)

    h = header.HeaderFields(callsign="K1ABC", grid="FN42", picture_id=1, mode=0, segment=0, codec_id=7)
    ok = noisy_llr(h, 3.0, np.random.default_rng(0))
    for spec in (frame.SHORT, lframe.L_SHORT):
        d = Diag(llr_spread=np.zeros(2474), llr_block=np.zeros(2474))
        d.reads = []
        assert row["decode_header"](ok, d, spec)[0] == h and d.reads == []
    sentinel = object()
    assert row["decode_header"](sentinel, None, frame.TINY) == (None, sentinel, frame.TINY)
    assert RX.decode_header(sentinel, None, frame.TINY) == (None, sentinel, frame.TINY)


# --- confidence_db, mean_w_db ---------------------------------------------------------------------

CONF_FRAMES = ("tiny", "short", "medium", "full", "l-tiny", "l-short", "l-full")


def _conf_cases(n, seed):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        spec = lframe.get_any(str(rng.choice(CONF_FRAMES)))
        nd = spec.n_data
        w = rng.exponential(2.0, nd) * (rng.random(nd) > 0.1)
        if rng.random() < 0.1:
            w = np.zeros(nd)
        p = make_pass_result(np.zeros(nd), w=w, seed=int(rng.integers(0, 1000)), waveform=spec.waveform,
                             hdr=HDR, hdr_es_n0_db=float(rng.uniform(-6, 6)))
        p.frame = spec.name
        v = rng.random()
        if v < 0.15:
            p.psi = None
        elif v < 0.25:
            p.psi = np.ones(spec.n_pos + 3, np.float32)
        elif v < 0.7:
            k = np.arange(spec.n_pos)
            p.psi = np.where((k // int(rng.integers(10, 400))) % int(rng.integers(2, 6)) == 1, 0.0,
                             rng.random(spec.n_pos)).astype(np.float32)
        if rng.random() < 0.1:
            p.hdr_llr = None
        elif rng.random() < 0.1:
            p.hdr_llr = p.hdr_llr[:100]
        el = [None, float(rng.uniform(0, spec.keyed_end_pos * spec.t_sym * 1.1)), 0.0, 1e12][int(rng.integers(0, 4))]
        nb = [1, 2, 7, 120, 500, 120][int(rng.integers(0, 6))]
        sp = [None, spec, lframe.get_any("short" if spec.waveform == WAVEFORM_L else "l-short")][int(rng.integers(0, 3))]
        out.append((p, sp, el, nb))
    return out


def test_confidence_db(row):
    for i, (p, sp, el, nb) in enumerate(_conf_cases(250, 21)):
        r = RX.confidence_db(p, sp, el, nb)
        g = row["confidence_db"](p, sp, el, nb)
        assert len(g) == len(r), i
        for x, y in zip(g, r):
            assert (x is None) == (y is None), i
            if y is not None:
                assert type(x) is float and abs(x - y) <= 1e-12, (i, x, y)
        assert row["mean_w_db"](p) == RX.mean_w_db(p), i


def test_confidence_db_duck_typed(row):
    """live's form: a namespace with w, hdr_llr and psi and nothing else,
    and the spec given; the default bin count."""
    spec = frame.SHORT
    rng = np.random.default_rng(3)
    ns = SimpleNamespace(w=rng.random(spec.n_data).astype(np.float32),
                         hdr_llr=rng.standard_normal(2474).astype(np.float32),
                         psi=np.ones(spec.n_pos, np.float32))
    assert row["confidence_db"](ns, spec) == RX.confidence_db(ns, spec)
    assert row["confidence_db"](ns, spec, 120.0) == RX.confidence_db(ns, spec, 120.0)
    ns.psi = None
    assert row["confidence_db"](ns, spec, n=33) == RX.confidence_db(ns, spec, n=33)


def _outcome(fn, *a, **k):
    try:
        return "ok", fn(*a, **k)
    except Exception as e:  # noqa: BLE001 - the reference's own type and text
        return type(e), str(e)


def test_confidence_db_refusals(row):
    p = make_pass_result(np.zeros(frame.SHORT.n_data), w=np.ones(frame.SHORT.n_data), waveform=WAVEFORM_CE)
    for nb in (0, -1, -2, -5):
        r, g = _outcome(RX.confidence_db, p, None, None, nb), _outcome(row["confidence_db"], p, None, None, nb)
        assert r[0] is ValueError and g == r, (nb, r, g)
    # nothing to place at n = 0: an empty list, no refusal
    q = make_pass_result(np.zeros(frame.SHORT.n_data), w=np.zeros(frame.SHORT.n_data), waveform=WAVEFORM_CE)
    q.psi = np.zeros(frame.SHORT.n_pos, np.float32)
    assert row["confidence_db"](q, None, None, 0) == RX.confidence_db(q, None, None, 0) == []


def test_mean_w_db(row):
    rng = np.random.default_rng(5)
    for w in (np.zeros(10), np.ones(1), rng.exponential(1.0, 50600), np.array([1e-31, 3.0, 0.5]),
              np.array([np.inf, 1.0]), np.array([np.nan, 1.0]), rng.random(4096).astype(np.float32),
              np.array([-1.0, 0.5])):
        p = SimpleNamespace(w=w)
        r, g = RX.mean_w_db(p), row["mean_w_db"](p)
        assert (math.isnan(r) and math.isnan(g)) or r == g, (w[:3], r, g)


# --- the small functions -------------------------------------------------------------------------

def test_frames(row):
    for sel in (frame.SHORT, lframe.L_SHORT, (frame.SHORT,), [lframe.L_TINY], (lframe.L_SHORT, frame.SHORT),
                [frame.TINY, lframe.L_TINY]):
        r = RX.frames(sel)
        g = row["frames"](sel)
        assert type(g) is tuple and len(g) == len(r) and all(a is b for a, b in zip(g, r)), sel
        assert row["is_joint"](sel) is RX.is_joint(sel), sel
        assert row["_single"](sel) is RX._single(sel) if not RX.is_joint(sel) else True, sel
    for bad in ((frame.SHORT, frame.TINY), (), [lframe.L_SHORT, lframe.L_TINY]):
        r, g = _outcome(RX.frames, bad), _outcome(row["frames"], bad)
        assert r[0] is ValueError and g == r, (bad, r, g)
        r, g = _outcome(RX.is_joint, bad), _outcome(row["is_joint"], bad)
        assert g == r, bad


def test_centres(row):
    c = _case("tiny-6")
    lc = _case("l-tiny-6")
    for obj in (c.det, lc.det, c.ref[0], lc.ref[0], SimpleNamespace(f_hz=1500.25),
                SimpleNamespace(f_hz=np.float32(1499.5), waveform=WAVEFORM_L)):
        assert row["centre_hz"](obj) == RX.centre_hz(obj), obj
    skip = [1500, 1490.5, np.float64(1480.25), np.int64(1400), np.float32(1300.5), c.det, lc.det, lc.ref[0]]
    assert row["_centres"](skip) == RX._centres(skip)
    assert row["_centres"](()) == RX._centres(()) == []


def test_ce_hints(row):
    for hints in ((), [1500.0, 1520.5], (1500.0,), [(1500.0, WAVEFORM_CE), (1520.0, WAVEFORM_L), 1530.5],
                  [[1490.0, WAVEFORM_L]], ((1500, 0), (1510.5, np.int64(0)))):
        r, g = RX._ce_hints(hints), row["_ce_hints"](hints)
        if r is hints:
            assert g is hints, hints
        else:
            assert g == r and all(type(a) is type(b) for a, b in zip(g, r)), (hints, g, r)


def test_prepare(row):
    for label in ("tiny-6", "l-tiny-12", "lead"):
        c = _capture(label)
        g = row["prepare"](c.cap)
        assert isinstance(g, RX.Prepared)
        assert g.q is c.cap.q and g.t0_index is c.cap.t0_index and g.fs is c.cap.fs, label
        for f in ("fe", "keep"):
            e = _rel(getattr(g, f), getattr(c.prep, f), (label, f))
            _worst(f"prepare {f}", e)
            assert e <= 1e-6, (label, f, e)
    p = c.prep
    assert row["prepare"](p) is p
    ch = frontend.Capture(np.zeros(100, np.complex64), Q_TEST, 2.5, fs=CH_FS)
    g, r = row["prepare"](ch), RX.prepare(ch)
    assert g.fe is ch.fe and r.fe is ch.fe and g.keep is None and g.fs == CH_FS and g.t0_index is ch.t0_index


def test_channel_for(row):
    for label in ("tiny-6", "l-tiny-6", "drift"):
        c = _case(label)
        p = c.ref[0]
        det_v = RX.detection_from_pass(p)
        for det, ex in ((c.det, ()), (det_v, (1530.0, 1460.5)), (det_v, [1525.0])):
            r = RX.channel_for(c.prep, c.spec, det, ex)
            g = row["channel_for"](c.prep, c.spec, det, ex)
            what = (label, det.timing is None, ex)
            assert isinstance(g, T.ChanCapture)
            assert (g.fs, g.t0_index, g.f_mix, g.exclude_hz) == (r.fs, r.t0_index, r.f_mix, r.exclude_hz), what
            assert type(g.exclude_hz) is tuple and (g.keep is None) == (r.keep is None), what
            for f in ("ch", "keep"):
                if getattr(r, f) is None:
                    continue
                e = _rel(getattr(g, f), getattr(r, f), (what, f))
                _worst(f"channel_for {f}", e)
                assert e <= 1e-6, (what, f, e)
        # a stored pass's CH capture: as it is, mixed at f_mix_hz or the detection's frequency
        prep = RX.capture_from_pass(p)
        for fm in (None, p.f_mix_hz, 12.5, Fraction(-3, 7)):
            r = RX.channel_for(prep, c.spec, det_v, (), fm)
            g = row["channel_for"](prep, c.spec, det_v, (), fm)
            assert g.ch is prep.fe and r.ch is prep.fe
            assert (g.fs, g.t0_index, g.f_mix, g.exclude_hz) == (r.fs, r.t0_index, r.f_mix, r.exclude_hz), fm
            assert type(g.f_mix) is Fraction


def test_psi_frame_and_cw_result(Rc, row):
    for label in ("short-6", "l-short-6", "tiny-6"):
        c = _capture(label)
        p, trs = _native_case(label, Rc)
        for tr in trs:
            g, r = row["_psi_frame"](tr), RX._psi_frame(tr)
            assert g.dtype == r.dtype == np.float32 and np.array_equal(g, r), label
        tr = trs[0]
        hdrs = (p.header, None, dataclasses.replace(HDR, callsign="W1AW"),
                dataclasses.replace(HDR, callsign="K1ABC   "), dataclasses.replace(HDR, callsign="1"))
        for h in hdrs:
            r = RX._cw_result(tr, c.spec, h)
            g = row["_cw_result"](tr, c.spec, h)
            what = (label, None if h is None else h.callsign)
            assert (g[0] is None) == (r[0] is None), what
            if r[0] is not None:
                assert (g[0].text, g[0].keying, g[0].agrees) == (r[0].text, r[0].keying, r[0].agrees), what
                _num(g[0].z_match, r[0].z_match, 1e-6, what, "_cw_result z_match")
            assert (g[1] is None) == (r[1] is None), what
            if r[1] is not None:
                assert np.array_equal(g[1][0], r[1][0]) and g[1][0].dtype == r[1][0].dtype, what
                assert g[1][1] == r[1][1], what


def test_stored_pass(row):
    c = _case("tiny-6")
    p = c.ref[0]
    for lead in (0.0, 3.5):
        g, r = row["detection_from_pass"](p, lead), RX.detection_from_pass(p, lead)
        assert type(g) is type(r)
        assert g.timing is p.timing and g.f_hz == r.f_hz and g.z_ref == r.z_ref and g.method == r.method
        assert g.lead_in_s == lead and g.waveform == r.waveform == WAVEFORM_CE
        assert np.array_equal(g.path.f_hz, r.path.f_hz) and np.array_equal(g.path.t_s, r.path.t_s)
        assert np.array_equal(g.path.weight, r.path.weight)
    lp = _case("l-tiny-6").ref[0]
    assert row["detection_from_pass"](lp).waveform == WAVEFORM_L
    g, r = row["capture_from_pass"](p), RX.capture_from_pass(p)
    assert isinstance(g, RX.Prepared) and g.fe is p.ch and r.fe is p.ch
    assert (g.keep, g.q, g.t0_index, g.fs) == (r.keep, r.q, r.t0_index, r.fs)


def test_refusals(row):
    """The waveform refusal (before the capture is touched), malformed
    erase spans and an unknown estimator: the reference's type and text."""
    for spec, wf in ((lframe.L_TINY, WAVEFORM_CE), (frame.TINY, WAVEFORM_L), (frame.SHORT, 5)):
        det = flat_det(1500.0, wf)
        r, g = _outcome(RX.receive_pass, None, spec, det), _outcome(row["receive_pass"], None, spec, det)
        assert r[0] is ValueError and g == r, (r, g)
    c = _case("tiny-6")
    v = RX.detection_from_pass(c.ref[0])
    for kw in (dict(erase_s=[(1.0, 2.0, 3.0)]), dict(erase_s=[(1.0,)]), dict(erase_s=[1.0]),
               dict(estimator="bogus")):
        r = _outcome(RX.receive_pass, c.prep, c.spec, v, **kw)
        g = _outcome(row["receive_pass"], c.prep, c.spec, v, **kw)
        assert r[0] != "ok" and g == r, (kw, r, g)


# --- the whole caller ------------------------------------------------------------------------------

def _mixed(name, snr=-12.0):
    cfg = (("neighbours", (Neighbour(60.0, 0.0, seed=7, waveform=WAVEFORM_CE),)),)
    _, sim = l_scenario(name, snr, 1, cfg=cfg)
    return frontend.Capture(sim.fe, sim.q, sim.t0_index)


def _tile(p):
    t = live.Tile(id="t", q=int(p.q), slot_utc="", frame=p.frame, f_hz=0.0)
    me = SimpleNamespace(joint=True, cfg=SimpleNamespace(spec=None, render=False), _headers={}, now=0.0,
                         dur=600.0, _spec_of=lambda pp: lframe.get_any(pp.frame))
    live.LiveListener._update_tile(me, t, p, 1.0, progress=0.5)
    return (t.mean_w_db, t.confidence, t.callsign, t.received)


@functools.lru_cache(maxsize=None)
def _callers(Rc):
    """The callers' reference outcomes: receive_slot on a CE TINY, an L
    l-tiny and a joint capture, em.rereceive of a CE pass, and live's tile
    update on a CE and an L SHORT pass (a header's LLRs in its confidence)."""
    def raw(label):
        sim = _capture(label).sim
        return frontend.Capture(sim.fe, Q_TEST, sim.t0_index)

    out = SimpleNamespace(ce=raw("tiny-12"), joint=_mixed("l-tiny"))
    out.slot_ce = RX.receive_slot(out.ce, frame.TINY)
    out.slot_joint = RX.receive_slot(out.joint, (frame.TINY, lframe.L_TINY))
    assert out.slot_ce and len(out.slot_joint) == 2
    out.p_ce = _case("tiny-6").ref[0]
    out.re = em.rereceive(out.p_ce, None)
    out.tile_passes = (_native_case("short-6", Rc)[0], _native_case("l-short-6", Rc)[0])
    assert all(p.header is not None for p in out.tile_passes)
    out.tiles = [_tile(p) for p in out.tile_passes]
    return out


def _counted(table, used):
    def wrap(key, fn):
        if key[0] != RECEIVER:
            return fn

        def f(*a, **k):
            used[key[1]] = used.get(key[1], 0) + 1
            return fn(*a, **k)
        for attr in ("__qrss_native__",):
            if hasattr(fn, attr):
                setattr(f, attr, getattr(fn, attr))
        return f
    return wrap


def _slot_same(got, ref, what):
    assert len(got) == len(ref), what
    for g, r in zip(got, ref):
        _pass_close(g, r, (what, r.frame), key="caller")


ALL_ROWS = {"prepare", "frames", "is_joint", "_single", "centre_hz", "_centres", "_ce_hints", "channel_for",
            "_psi_frame", "_cw_result", "receive_pass", "decode_header", "_decode_header_l", "detection_from_pass",
            "capture_from_pass", "mean_w_db", "confidence_db"}
# configuration: (the receiver functions kept the reference's, what the callers must reach)
CONFIGS = {
    "every row": ((), {"prepare", "frames", "is_joint", "_single", "centre_hz", "_centres", "_ce_hints",
                       "channel_for", "receive_pass", "detection_from_pass", "capture_from_pass", "mean_w_db",
                       "confidence_db"}),
    "receive_pass kept": (("receive_pass",), {"prepare", "channel_for", "decode_header", "_cw_result",
                                              "_psi_frame"}),
    "decode_header kept": (("receive_pass", "decode_header"), {"_decode_header_l", "_cw_result"}),
}


def test_configs_reach_every_row(row):
    assert set().union(*(m for _k, m in CONFIGS.values())) == ALL_ROWS == set(row)


@pytest.mark.parametrize("config", list(CONFIGS))
def test_whole_caller(native, Rc, config):
    """Every row of every tx and rx rows file over its target and its
    from-import sites, the receiver's counted, but the receiver functions
    the configuration keeps. "every row": receive_slot (CE, L, joint),
    em.rereceive and live's tile update give the reference's outcomes.
    "receive_pass kept": the reference's receive_pass under the joint
    receive_slot and em.rereceive, over the receiver's rows. "decode_header
    kept": the reference's receive_pass and decode_header on a CE and an L
    SHORT pass, bit for bit the C++ receive_pass. Every row the
    configuration is for is reached from them (the counters were installed
    before the rows: they are what the module attributes and from-import
    sites hold)."""
    ref = _callers(Rc)
    keep, must = CONFIGS[config]
    used: dict = {}
    with pytest.MonkeyPatch.context() as mp:
        _apply_rows(native, mp, skip={(RECEIVER, k) for k in keep}, wrap=_counted(None, used))
        if config == "decode_header kept":
            for label in ("short-6", "l-short-6"):
                c = _capture(label)
                got = RX.receive_pass(c.prep, c.spec, c.det, return_tracks=True)
                _bitwise_pass(_native_case(label, Rc), got, (config, label))
        else:
            if config == "every row":
                _slot_same(RX.receive_slot(ref.ce, frame.TINY), ref.slot_ce, (config, "CE"))
                for p, t in zip(ref.tile_passes, ref.tiles):
                    assert _tile(p) == t, config
            _slot_same(RX.receive_slot(ref.joint, (frame.TINY, lframe.L_TINY)), ref.slot_joint, (config, "joint"))
            _pass_close(em.rereceive(ref.p_ce, None), ref.re, (config, "rereceive"), key="caller")
    assert must <= set(used), (config, sorted(must - set(used)), sorted(used))


@pytest.mark.slow
def test_whole_caller_l_alone(native, Rc):
    """receive_slot(cap, L_TINY) on an l-tiny capture at -12 dB, every row
    applied, the receiver's counted."""
    sim = _capture("l-tiny-12").sim
    cap = frontend.Capture(sim.fe, Q_TEST, sim.t0_index)
    ref = RX.receive_slot(cap, lframe.L_TINY)
    assert len(ref) == 1 and ref[0].waveform == WAVEFORM_L
    used: dict = {}
    with pytest.MonkeyPatch.context() as mp:
        _apply_rows(native, mp, wrap=_counted(None, used))
        got = RX.receive_slot(cap, lframe.L_TINY)
    _slot_same(got, ref, "L alone")
    assert {"prepare", "receive_pass", "channel_for", "centre_hz", "is_joint", "frames"} <= set(used), used


@pytest.mark.slow
def test_whole_caller_joint_short(native, Rc):
    """receive_slot(cap, (SHORT, L_SHORT)) on an l-short capture at -6 dB
    with a CE SHORT neighbour 60 Hz above: the L pass with its header, the
    CE one with the neighbour's callsign reading (its header is drawn at
    random and does not decode, at 0 dB either), every row applied (slow:
    two SHORT passes)."""
    cap = _mixed("l-short", -6.0)
    ref = RX.receive_slot(cap, (frame.SHORT, lframe.L_SHORT))
    assert sorted((p.waveform, round(p.f_hz)) for p in ref) == [(WAVEFORM_CE, 1560), (WAVEFORM_L, 1500)]
    by = {p.waveform: p for p in ref}
    assert by[WAVEFORM_L].header is not None and by[WAVEFORM_L].cw.text == by[WAVEFORM_L].header.callsign
    assert by[WAVEFORM_CE].cw is not None and by[WAVEFORM_CE].cw.text == "N7NBR"
    used: dict = {}
    with pytest.MonkeyPatch.context() as mp:
        _apply_rows(native, mp, wrap=_counted(None, used))
        got = RX.receive_slot(cap, (frame.SHORT, lframe.L_SHORT))
    _slot_same(got, ref, "joint SHORT")
    assert {"prepare", "receive_pass", "channel_for", "centre_hz", "_centres", "is_joint", "frames"} <= set(used), used


# --- threads -------------------------------------------------------------------------------------

def test_threads(Rc):
    """Four passes at once (the GIL released while each is received): each
    the pass alone gives."""
    from concurrent.futures import ThreadPoolExecutor

    labels = ["tiny-6", "l-tiny-6", "tiny-12", "l-tiny-12"]
    want = {k: _native_pass(Rc, _capture(k).prep, _capture(k).spec, _capture(k).det) for k in labels}
    with ThreadPoolExecutor(4) as ex:
        got = list(ex.map(lambda k: _native_pass(Rc, _capture(k).prep, _capture(k).spec, _capture(k).det), labels))
    for k, g in zip(labels, got):
        assert g.uid == want[k].uid and np.array_equal(g.z, want[k].z), k

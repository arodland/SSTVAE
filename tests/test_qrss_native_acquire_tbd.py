"""Parity of native/core/qrss/acquire_tbd (sstvae_native.qrss.acquire_tbd)
with A3 of sstvae/qrss/acquire.py (track-before-detect), `frame_map` and
`_path_of`, held side by side in one process.

What is exact (dtype and shape included) and what is not:

- Frame starts, the Viterbi paths *and* Z (numpy's float64 widening and
  its pairwise sum of the path's emissions), the channels, the emissions
  (spectrogram and CFAR are acquire_stats', bit for bit), the shape-test
  frames, `shape_z` (z1 and z_above), `shape_waveform` / `_shape_verdict`,
  `_bins_path` (numpy's float32 SIMD log is reproduced), `l_pilot_bins`,
  `frame_map`, `_path_of`: equal to the bit.
- A3's hits: every field but p_value equal to the bit, in the same order;
  p_value is `gumbel_p`'s (numpy's SIMD exp/expm1), a few 1e-16.
- `TbdHit.line` / `f_hz`, and a detection's f_hz: `np.polyfit`, whose
  least squares is LAPACK's SVD in the reference and Householder QR here
  (core/math/npstats): 1e-9 relative in the contract, measured ~1e-15 on
  the intercept; the slope is near 0, so it is held to 1e-9 relative
  plus 1e-12 Hz/s.

The callers: `track_before_detect` and `acquire` (one frame and a CE + L
tuple) on CE and L SHORT captures and on noise, with the rows of
tests/qrss_native/acquire_tbd.py and acquire_stats.py assigned over
acquire's names -- the same Acquisition, field for field.

Review probes (end of file): `track_before_detect` on 14 random captures
(TINY, L_TINY, SHORT; complex64 and complex128; fractional t0; tones,
drifting tones and L-like pilots with sidebands; random candidates,
bands, pfa, all_hits and shape specs) against its row, hits field for
field; `shape_z`, `l_pilot_bins`, `_bins_path` and the Viterbi (one
channel and a stacked three) at FULL size (896 and 2500 frames, float32
and float64), bit for bit; a whole FULL slot through
`track_before_detect` (slow, ~12 s); and a spy that the joint `acquire`
A/B really ran the C++.
"""

import dataclasses
import math

import numpy as np
import pytest

from qrss_native import _rx, require
from sstvae.qrss import acquire as A
from sstvae.qrss import frame, lframe
from sstvae.qrss.constants import FE_FS, WAVEFORM_CE, WAVEFORM_L
from sstvae.qrss.types import CarrierCandidate, Detection, FreqPath

SPECS = (frame.TINY, frame.SHORT, frame.MEDIUM, frame.FULL, lframe.L_TINY, lframe.L_SHORT,
         lframe.L_FULL)


@pytest.fixture(scope="module")
def N(native):
    return require(native, "acquire_tbd")


def _same(a, b, what=""):
    a, b = np.asarray(a), np.asarray(b)
    assert a.dtype == b.dtype, (what, a.dtype, b.dtype)
    assert a.shape == b.shape, (what, a.shape, b.shape)
    if a.dtype.kind in "fc":
        an, bn = np.isnan(a), np.isnan(b)
    else:
        an = bn = np.zeros(a.shape, bool)
    assert np.array_equal(an, bn), what
    assert a[~an].tobytes() == b[~bn].tobytes(), (what, np.count_nonzero(a[~an] != b[~bn]))


def _same_path(r, g, what=""):
    g = _rx.to_ref(g)                          # the C++ returns the "__type__" dict
    assert type(g) is type(r) is FreqPath, (what, type(r), type(g))
    for f in ("t_s", "f_hz", "weight"):
        _same(getattr(g, f), getattr(r, f), f"{what}.{f}")


def _line_close(r, g, what=""):
    r, g = np.asarray(r), np.asarray(g)
    assert g.dtype == r.dtype and g.shape == r.shape == (2,), (what, r, g)
    assert abs(g[0] - r[0]) <= 1e-9 * abs(r[0]) + 1e-12, (what, r, g)
    assert abs(g[1] - r[1]) <= 1e-9 * abs(r[1]), (what, r, g)


def _outcome(fn, *args, **kw):
    try:
        return "ok", fn(*args, **kw)
    except (ValueError, TypeError, IndexError, ZeroDivisionError, UnboundLocalError, OverflowError,
            NotImplementedError) as e:
        return type(e), str(e)


# --- frames, Viterbi, channels -------------------------------------------------------------

def test_frame_starts(N):
    for spec in SPECS:
        _same(N.tbd_frame_starts(spec), A.tbd_frame_starts(spec), spec.name)
        assert N.tbd_n_frames(spec) == A.tbd_n_frames(spec)
        for n_fe in (0, 1, 16000, 400_000, 10 ** 7):
            for t0 in (0.0, 48000.0, 12345.5, -500.25):
                for fs in (FE_FS, 500, 250.0):
                    _same(N.tbd_frame_starts(spec, n_fe, t0, fs), A.tbd_frame_starts(spec, n_fe, t0, fs),
                          f"{spec.name} {n_fe} {t0} {fs}")


def test_viterbi(N):
    rng = np.random.default_rng(1)
    E = rng.exponential(1.0, (120, 81)) - 1.0
    cases = [E, E.astype(np.float32), np.stack([E, E[:, ::-1], E[::-1]]),
             np.stack([E, E]).astype(np.float32)[None], np.round(E * 2),          # ties
             np.round(E).astype(np.int64), E[:1], E[:, :1], E[:2, :3], E[:, :2]]
    nanE = E.copy()
    nanE[7, 13] = np.nan
    infE = E.copy()
    infE[3, :40] = -np.inf
    cases += [nanE, infE]
    for i, X in enumerate(cases):
        for max_step, cost in ((2, 0.5), (0, 0.5), (1, 0.0), (3, 2.0), (2, -0.25)):
            if max_step >= X.shape[-1]:
                continue
            (rp, rz), (gp, gz) = A.tbd_viterbi(X, max_step, cost), N.tbd_viterbi(X, max_step, cost)
            _same(gp, rp, f"path {i} {max_step} {cost}")
            assert type(gz) is type(rz), (i, type(rz), type(gz))
            _same(gz, rz, f"Z {i} {max_step} {cost}")
    for X, ms in ((E[:0], 2), (E[:, :0], 2), (E[:, :3], 4), (E, 200)):
        r, g = _outcome(A.tbd_viterbi, X, ms), _outcome(N.tbd_viterbi, X, ms)
        if r[0] == "ok":
            _same(g[1][0], r[1][0])
        else:
            assert g[0] in (r[0], NotImplementedError), (X.shape, ms, r, g)
            if g[0] is not NotImplementedError:
                assert g == r, (X.shape, ms, r, g)


def test_channels(N):
    rng = np.random.default_rng(2)
    _same(N.tbd_channels(), A.tbd_channels())
    for _ in range(200):
        cands = [CarrierCandidate(float(f), 0.0, 0.0) for f in rng.uniform(250, 2750, rng.integers(0, 12))]
        for band in (A.BAND_HZ, (900.0, 1300.0), (1000, 1100), (1000.0, 1050.0)):
            _same(N.tbd_channels(cands, band), A.tbd_channels(cands, band), f"{band}")
    for cands in ([CarrierCandidate(1525.0, 0.0, 0.0)], [CarrierCandidate(1500.0, 0, 0)] * 3,
                  [CarrierCandidate(np.float32(1234.5), 0.0, 0.0)]):
        _same(N.tbd_channels(cands), A.tbd_channels(cands))


def test_path_line_and_path_of(N):
    rng = np.random.default_rng(3)
    for i in range(300):
        n = int(rng.integers(2, 200))
        t = -8.0 + 2.0 * np.arange(n)
        f = 1500.0 + rng.uniform(-5, 5) + rng.uniform(-0.01, 0.01) * t + rng.standard_normal(n) * 0.3
        f[rng.integers(0, n, n // 5)] += rng.uniform(-20, 20, n // 5)
        w = np.maximum(rng.standard_normal(n), 0.0)
        if i % 10 == 0:
            w[:] = 0.0
        p = FreqPath(t_s=t, f_hz=f, weight=w)
        _line_close(A._path_line(p), N._path_line(p), f"line {i}")
    for n in (0, 1):
        p = FreqPath(t_s=np.arange(n, dtype=float), f_hz=np.ones(n), weight=np.ones(n))
        r, g = _outcome(A._path_line, p), _outcome(N._path_line, p)
        assert r == g and r[0] is UnboundLocalError, (r, g)
    for f, d, dur in ((1500.0, 0.0, 300.0), (1234.5, -3.3, 1800.0), (0.0, 1e-3, 0.0)):
        _same_path(A._path_of(f, d, dur), N._path_of(f, d, dur))


def test_frame_map(N):
    for specs in ((frame.SHORT,), (frame.SHORT, lframe.L_SHORT), [lframe.L_FULL, frame.TINY],
                  (frame.SHORT, frame.FULL), (lframe.L_TINY, lframe.L_SHORT), ()):
        r, g = _outcome(A.frame_map, specs), _outcome(N.frame_map, specs)
        assert r[0] == g[0], (specs, r, g)
        if r[0] == "ok":
            assert list(r[1].items()) == list(g[1].items()) and all(
                g[1][w] is specs[i] for i, w in enumerate(g[1])), (r, g)
        else:
            assert r[1] == g[1], (r, g)


# --- the shape test ------------------------------------------------------------------------

def test_shape_frames(N):
    for spec in (frame.SHORT, frame.FULL, lframe.L_SHORT):
        t = A.tbd_frame_starts(spec)
        for specs in ((frame.SHORT, lframe.L_SHORT), (frame.FULL,), (lframe.L_FULL, frame.MEDIUM),
                      (frame.TINY,)):
            for fs in (A.STFT_FRAME_S, 2.0):
                _same(N.shape_frames(t, specs, fs), A.shape_frames(t, specs, fs), f"{spec.name} {specs}")
        _same(N.shape_frames(list(t), (frame.SHORT,)), A.shape_frames(list(t), (frame.SHORT,)))


def _grid(rng, n, nb, dtype):
    return (rng.exponential(1.0, (n, nb)) - 1.0).astype(dtype)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_shape_z(N, dtype):
    rng = np.random.default_rng(4)
    for i in range(120):
        n, nb = int(rng.integers(1, 60)), int(rng.integers(150, 500))
        E = _grid(rng, n, nb, dtype)
        if i % 3 == 0:
            E[rng.integers(0, n, max(1, n // 6))] = -1.0          # capture gaps
        bins = rng.integers(-20, nb + 20, n)
        if i % 2:
            bins = np.clip(bins, 90, nb - 91)
        keep = None if i % 4 == 0 else rng.random(n) < 0.8
        for df, fs in ((0.25, FE_FS), (0.5, 500.0)):
            r, g = A.shape_z(E, bins, keep, df, fs), N.shape_z(E, bins, keep, df, fs)
            assert type(g) is tuple and type(g[0]) is type(r[0]) is float, (r, g)
            _same(np.array(g), np.array(r), f"shape_z {i} {df}")
    E = _grid(rng, 5, 400, dtype)
    _same(np.array(N.shape_z(E, np.full(5, 200), np.zeros(5, bool))),
          np.array(A.shape_z(E, np.full(5, 200), np.zeros(5, bool))))
    _same(np.array(N.shape_z(E, [200] * 5)), np.array(A.shape_z(E, [200] * 5)))


def test_shape_waveform_and_verdict(N):
    vals = (-np.inf, -5.0, 0.0, 2.9999, 3.0, 3.5, 4.0, 4.0001, 10.0, np.inf, np.nan)
    for a in vals:
        for b in vals:
            assert N.shape_waveform(a, b) == A.shape_waveform(a, b), (a, b)
    ws = (None, WAVEFORM_CE, WAVEFORM_L)
    for v in ws:
        for w in ws:
            assert N._shape_verdict(v, w) == A._shape_verdict(v, w), (v, w)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_bins_path(N, dtype):
    rng = np.random.default_rng(5)
    for i in range(60):
        n, nb = int(rng.integers(2, 80)), int(rng.integers(3, 400))
        E = _grid(rng, n, nb, dtype)
        E[rng.random(E.shape) < 0.05] = -1.0                     # log of a floored 0
        f = 1000.0 + 0.25 * np.arange(nb)
        t = -10.0 + 2.0 * np.arange(n)
        bins = rng.integers(0, nb, n)
        bins[0], bins[-1] = 0, nb - 1
        _same_path(A._bins_path(E, f, t, bins), N._bins_path(E, f, t, bins), f"{i}")


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_l_pilot_bins(N, dtype):
    """The planted pilot of tests/test_qrss_lacquire.py, random lines across
    the grid (and past its edges), keep, the default and a narrower span."""
    rng = np.random.default_rng(9)
    n, nb = 300, 800
    f = 1000.0 + 0.25 * np.arange(nb)
    E = rng.exponential(1.0, (n, nb)) - 1.0
    p0 = 500
    E[:, p0] += 1.5 * rng.exponential(1.0, n)
    E[:, p0 - 88:p0 - 12] += 0.5 * rng.exponential(1.0, (n, 76))
    E = E.astype(dtype)
    t = -10.0 + 2.0 * np.arange(n)
    keep = np.ones(n, dtype=bool)
    keep[100:130] = False
    for line in ([0.0, f[p0 - 40]], [0.001, 1100.0], [-0.002, 1010.0], [0.0, 1195.0], [0.0, 900.0]):
        for kp in (None, keep):
            for span in (A.L_ANCHOR_HZ, (-2.0, 4.0)):
                r = A.l_pilot_bins(E, f, t, np.array(line), kp, FE_FS, span)
                g = N.l_pilot_bins(E, f, t, np.array(line), kp, FE_FS, span)
                _same(g, r, f"{line} {span}")


@pytest.mark.parametrize("cdtype", [np.complex64, np.complex128])
def test_emissions(N, cdtype):
    rng = np.random.default_rng(6)
    fs = FE_FS
    t0 = 12 * fs + 0.37
    t = A.tbd_frame_starts(frame.TINY, None, t0, fs)
    n = int(t0 + (t[-1] + A.STFT_FRAME_S) * fs) + 10
    x = ((rng.standard_normal(n) + 1j * rng.standard_normal(n)) / np.sqrt(2)).astype(cdtype)
    for band in (A.BAND_HZ, (1200.0, 1500.0)):
        r, g = A.tbd_emissions(x, t0, frame.TINY, band_hz=band), N.tbd_emissions(x, t0, frame.TINY, band_hz=band)
        for a, b, what in zip(r, g, "Eft"):
            _same(b, a, what)
    r, g = A.tbd_emissions(x[:5000], t0, frame.TINY), N.tbd_emissions(x[:5000], t0, frame.TINY)
    for a, b, what in zip(r, g, "Eft"):
        _same(b, a, what + " short capture")


# --- the hits ------------------------------------------------------------------------------

def _hit(rng, l_path, waveform):
    n = 40
    t = -8.0 + 2.0 * np.arange(n)

    def path():
        return FreqPath(t_s=t, f_hz=1500.0 + rng.uniform(-30, 30) + 0.002 * t + 0.1 * rng.standard_normal(n),
                        weight=np.maximum(rng.standard_normal(n), 0.0))
    return A.TbdHit(path=path(), z=float(rng.uniform(5, 30)), p_value=float(rng.uniform(0, 2e-3)),
                    n_frames=n, centre_hz=1500.0, waveform=waveform,
                    l_path=path() if l_path else None)


def test_hit_methods(N):
    """f_hz, line, passed and to_detection on hits built in Python, as the
    tests build them: the C++ reads the hit's fields."""
    rng = np.random.default_rng(7)
    for i in range(60):
        h = _hit(rng, i % 2 == 0, (None, WAVEFORM_CE, WAVEFORM_L)[i % 3])
        assert N._hit_passed(h) == h.passed
        _line_close(h.line(), N._hit_line(h))
        assert abs(N._hit_f_hz(h) - h.f_hz) <= 1e-12 * abs(h.f_hz)
        for wf in (None, WAVEFORM_CE, WAVEFORM_L):
            r = h.to_detection(wf)
            g = N._hit_to_detection(h, wf)
            assert g["__type__"] == "Detection"
            assert abs(g["f_hz"] - r.f_hz) <= 1e-12 * abs(r.f_hz)
            assert (g["method"], g["waveform"], g["a3_z"], g["lead_in_s"], g["timing"]) == (
                r.method, r.waveform, r.a3_z, r.lead_in_s, r.timing)
            assert math.isnan(g["z_ref"]) and math.isnan(r.z_ref)
            for k in ("t_s", "f_hz", "weight"):
                _same(g["path"][k], getattr(r.path, k))


# --- the callers ---------------------------------------------------------------------------

_LINE_FIELDS = {"f_hz"}             # polyfit's, in a Detection: tolerance


def _deep(r, g, what, worst):
    if dataclasses.is_dataclass(r) and not isinstance(r, type):
        assert type(g) is type(r), (what, type(r), type(g))
        for f in dataclasses.fields(r):
            a, b = getattr(r, f.name), getattr(g, f.name)
            if type(r) is A.TbdHit and f.name == "p_value":
                assert type(b) is float and abs(b - a) <= 4e-16 * abs(a), (what, a, b)
                worst["p_value"] = max(worst["p_value"], abs(b - a) / abs(a) if a else 0.0)
                continue
            if type(r) is Detection and f.name in _LINE_FIELDS and r.method == "tbd":
                assert type(b) is float and abs(b - a) <= 1e-12 * abs(a), (what, a, b)
                worst["f_hz"] = max(worst["f_hz"], abs(b - a) / abs(a))
                continue
            _deep(a, b, f"{what}.{f.name}", worst)
    elif isinstance(r, dict):
        assert type(g) is dict and list(r) == list(g), (what, r, g)
        for k in r:
            _deep(r[k], g[k], f"{what}[{k!r}]", worst)
    elif isinstance(r, (list, tuple)):
        assert type(g) is type(r) and len(g) == len(r), (what, len(r), len(g))
        for i, (a, b) in enumerate(zip(r, g)):
            _deep(a, b, f"{what}[{i}]", worst)
    elif isinstance(r, np.ndarray):
        _same(g, r, what)
    elif isinstance(r, float):
        assert type(g) is type(r) and (r == g or (np.isnan(r) and np.isnan(g))), (what, r, g)
    else:
        assert type(g) is type(r) and r == g, (what, r, g)


@pytest.fixture(scope="module")
def captures():
    import test_qrss_acquire as TA
    import test_qrss_lacquire as TL
    from qrss_fakes import l_scenario
    from sstvae.qrss import frontend

    out = {}
    x, t0 = TA._capture(1650.0, -15.0, shift_s=0.7, seed=3)
    out["ce"] = (frontend.Capture(fe=x, q=0, t0_index=t0), frame.SHORT)
    _, sim = l_scenario("l-short", -15.0, 7, pilot=1421.7)
    x, keep = frontend.blank(sim.fe)
    x, _ = frontend.normalise(x, keep)
    out["l"] = (frontend.Capture(fe=x, q=0, t0_index=sim.t0_index), lframe.L_SHORT)
    out["noise"] = (frontend.Capture(fe=TL._noise_fe(len(x), np.random.default_rng(16)), q=0,
                                     t0_index=sim.t0_index), frame.SHORT)
    return out


def _apply(native, monkeypatch):
    from qrss_native import _owner
    from qrss_native.acquire_stats import rows as stats_rows
    from qrss_native.acquire_tbd import rows as tbd_rows

    table = dict(stats_rows(native.qrss))
    table.update(tbd_rows(native.qrss))
    for (target, name), fn in table.items():
        assert target in ("sstvae.qrss.acquire", "sstvae.qrss.acquire.TbdHit"), target
        monkeypatch.setattr(_owner(target), name, fn)


def test_track_before_detect_on_the_cpp(N, native, captures, monkeypatch):
    """track_before_detect on CE and L SHORT captures and noise -- every
    channel (all_hits) and the default, with and without A1's lines, the
    joint shape test (CE + L, so the L re-anchoring too) -- and each hit's
    f_hz, line, passed and detections: the same hits, in the same order."""
    specs = (frame.SHORT, lframe.L_SHORT)

    def run():
        out = []
        for cap, spec in captures.values():
            fe, t0 = cap.fe, cap.t0_index
            cands = A.carrier_lines(fe, t0)
            for kw in ({"all_hits": True}, {"shape_specs": specs}, {"all_hits": True, "shape_specs": specs}):
                hits = A.track_before_detect(fe, t0, spec, cands, **kw)
                out.append(hits)
                out.append([h.passed for h in hits])
                out.append([h.to_detection(w) for h in hits for w in (None, WAVEFORM_CE, WAVEFORM_L)])
                out.append({"lines": [h.line() for h in hits], "f": [h.f_hz for h in hits]})
            out.append(A.track_before_detect(fe, t0, spec, (), FE_FS, pfa=0.5, band_hz=(1000.0, 1900.0)))
        return out

    ref = run()
    _apply(native, monkeypatch)
    got = run()
    worst = {"p_value": 0.0, "f_hz": 0.0}
    for i, (r, g) in enumerate(zip(ref, got)):
        if isinstance(r, dict):
            for a, b in zip(r["lines"], g["lines"]):
                _line_close(a, b, f"run[{i}].line")
            for a, b in zip(r["f"], g["f"]):
                assert type(b) is float and abs(b - a) <= 1e-12 * abs(a), (a, b)
            continue
        _deep(r, g, f"run[{i}]", worst)
    assert sum(len(r) for r in ref if isinstance(r, list) and r and isinstance(r[0], A.TbdHit)) > 200
    print("worst", worst)


def test_acquire_on_the_cpp(N, native, captures, monkeypatch):
    """acquire() -- A1, A2, A3 and the arbitration around them -- with one
    frame and with the CE + L tuple, on the CE and L captures: the same
    Acquisition, and the same detections in the same order."""
    def run():
        out = []
        for key in ("ce", "l"):
            cap, spec = captures[key]
            acq = A.acquire(cap, spec, preprocessed=True)
            out += [acq, acq.detections(spec)]
            joint = A.acquire(cap, (frame.SHORT, lframe.L_SHORT), preprocessed=True)
            out += [joint, joint.detections((frame.SHORT, lframe.L_SHORT))]
        return out

    ref = run()
    _apply(native, monkeypatch)
    got = run()
    worst = {"p_value": 0.0, "f_hz": 0.0}
    _deep(ref, got, "acquire", worst)
    assert ref[0].tbd and ref[0].preamble and ref[2].tbd and ref[4].tbd
    print("worst", worst)


# --- review probes (2026-10-13): sizes, layouts and callers the tests above miss -------------

def _synthetic_capture(spec, t0, rng, tones=(), l_signals=(), noise=1.0, dtype=np.complex64):
    """A capture covering every A3 frame of `spec` (plus a little), complex
    noise of variance `noise`, steady tones (f_audio, amplitude, drift Hz/min)
    and L-like signals (pilot f_audio, pilot amplitude, data amplitude: a
    pilot tone with band-limited noise 3.1-22 Hz below it)."""
    from sstvae.qrss.constants import FE_CENTER_HZ

    t = A.tbd_frame_starts(spec, None, t0, FE_FS)
    n = int(t0 + (t[-1] + A.STFT_FRAME_S) * FE_FS) + 37
    x = math.sqrt(noise / 2) * (rng.standard_normal(n) + 1j * rng.standard_normal(n))
    k = np.arange(n)
    ts = (k - t0) / FE_FS
    def phasor(turns):                                    # np.exp of complex is slow at 7 M
        ph = 2 * np.pi * (turns - np.floor(turns))
        return np.cos(ph) + 1j * np.sin(ph)

    for f, a, drift in tones:
        x += a * phasor((f - FE_CENTER_HZ) * ts + drift / 120.0 * ts * ts)
    m = 1 << int(n - 1).bit_length()                      # one power-of-two FFT per signal
    fr = np.fft.fftfreq(m, 1.0 / FE_FS) + FE_CENTER_HZ
    for f, a_pilot, a_data in l_signals:
        x += a_pilot * phasor((f - FE_CENTER_HZ) * ts)
        band = np.flatnonzero((fr > f - 22.0) & (fr < f - 3.1))
        X = np.zeros(m, dtype=np.complex128)
        X[band] = rng.standard_normal(len(band)) + 1j * rng.standard_normal(len(band))
        d = np.fft.ifft(X)[:n]
        x += a_data * d / np.sqrt(np.mean(np.abs(d) ** 2))
    return x.astype(dtype)


def _hits_same(r, g, what):
    """Every field equal (p_value to gumbel_p's few 1e-16), in order."""
    worst = {"p_value": 0.0, "f_hz": 0.0}
    assert len(g) == len(r), (what, len(r), len(g))
    _deep(r, g, what, worst)
    return worst


def test_track_before_detect_random_captures(N, native):
    """A3 on random TINY- and SHORT-length captures: fractional t0, tones
    and L-like signals anywhere in (and past) the band, A1-like candidates
    in and out of it, other bands and pfas, every shape_specs a caller
    passes (none, CE, L, both): the reference's hits, field for field and
    in order, with the C++ row's TbdHits."""
    from qrss_native.acquire_tbd import rows

    tbd = rows(native.qrss)[("sstvae.qrss.acquire", "track_before_detect")]
    rng = np.random.default_rng(21)
    shapes = (None, (frame.TINY,), (lframe.L_TINY,), (frame.TINY, lframe.L_TINY),
              (lframe.L_SHORT, frame.SHORT))
    n_hits = n_l = n_wf = 0
    for i in range(14):
        spec = (frame.TINY, lframe.L_TINY, frame.SHORT)[i % 3] if i % 5 else frame.TINY
        t0 = float(rng.uniform(0, 20)) * FE_FS + float(rng.uniform(0, 1))
        tones = [(float(rng.uniform(200, 2800)), float(rng.uniform(0.02, 0.3)),
                  float(rng.uniform(-2, 2))) for _ in range(int(rng.integers(0, 4)))]
        ls = [(float(rng.uniform(400, 2600)), float(rng.uniform(0.05, 0.3)),
               float(rng.uniform(0.1, 0.6))) for _ in range(int(rng.integers(0, 3)))]
        x = _synthetic_capture(spec, t0, rng, tones, ls,
                               dtype=np.complex64 if i % 4 else np.complex128)
        cands = [CarrierCandidate(float(f), 0.0, 0.0)
                 for f in rng.uniform(150, 2850, int(rng.integers(0, 6)))]
        cands += [CarrierCandidate(f, 0.0, 0.0) for f, _, _ in tones[:1]]
        band = (A.BAND_HZ, (700.0, 2100.0), (1000, 1301))[i % 3]
        kw = {"all_hits": bool(i % 2), "pfa": (A.TBD_PFA, 0.5, 1e-6)[i % 3], "band_hz": band}
        sh = shapes[i % len(shapes)]
        if sh is not None:
            kw["shape_specs"] = sh
        r = A.track_before_detect(x, t0, spec, cands, **kw)
        g = tbd(x, t0, spec, cands, **kw)
        _hits_same(r, g, f"case {i}")
        n_hits += len(r)
        n_l += sum(h.l_path is not None for h in r)
        n_wf += sum(h.waveform is not None for h in r)
    assert n_hits > 60 and n_l > 10 and n_wf > 5, (n_hits, n_l, n_wf)


def test_full_size_pieces(N):
    """FULL-pass sizes (896 frames, and past), where numpy's float64 sum of
    float32 emissions runs over several 8192-element buffers: shape_z,
    l_pilot_bins, _bins_path and the Viterbi (one channel and a stack)
    stay bit-exact."""
    rng = np.random.default_rng(22)
    for dtype in (np.float32, np.float64):
        for n in (896, 2500):
            nb = 700
            E = _grid(rng, n, nb, dtype)
            E[:, 350] += dtype(1.0)
            E[:, 262:338] += dtype(0.3)
            E[rng.integers(0, n, n // 20)] = -1.0
            bins = np.clip(350 + rng.integers(-3, 4, n), 0, nb - 1)
            keep = rng.random(n) < 0.95
            for kp in (None, keep):
                r, g = A.shape_z(E, bins, kp), N.shape_z(E, bins, kp)
                _same(np.array(g), np.array(r), f"shape_z {dtype} {n}")
            f = 1000.0 + 0.25 * np.arange(nb)
            t = -10.0 + 2.0 * np.arange(n)
            _same(N.l_pilot_bins(E, f, t, np.array([1e-4, 1080.0]), keep),
                  A.l_pilot_bins(E, f, t, np.array([1e-4, 1080.0]), keep), f"l_pilot_bins {n}")
            _same_path(A._bins_path(E, f, t, bins), N._bins_path(E, f, t, bins), f"_bins_path {n}")
            (rp, rz), (gp, gz) = A.tbd_viterbi(E[:, 190:511]), N.tbd_viterbi(E[:, 190:511])
            _same(gp, rp, "path")
            _same(gz, rz, "Z")
            st = np.stack([E[:, c - 160:c + 161] for c in (160, 350, 539)])
            (rp, rz), (gp, gz) = A.tbd_viterbi(st), N.tbd_viterbi(st)
            _same(gp, rp, "stacked paths")
            _same(gz, rz, "stacked Z")


@pytest.mark.slow
def test_track_before_detect_full_slot(N, native):
    """A3 over a whole FULL slot (896 frames, 7 M FE samples) with the
    joint shape test (FULL + L_FULL): a drifting carrier, an L-like signal
    and noise, every channel's hit -- the same hits as the reference's.
    Slow (about 12 s, most of it building the capture and the reference's
    4.5 s); `test_full_size_pieces` holds the same sizes in the fast tier."""
    from qrss_native.acquire_tbd import rows

    tbd = rows(native.qrss)[("sstvae.qrss.acquire", "track_before_detect")]
    rng = np.random.default_rng(23)
    t0 = 12 * FE_FS + 0.25
    x = _synthetic_capture(frame.FULL, t0, rng, tones=[(1234.5, 0.03, 0.8)],
                           l_signals=[(1777.0, 0.03, 0.06)])
    kw = {"all_hits": True, "shape_specs": (frame.FULL, lframe.L_FULL)}
    r = A.track_before_detect(x, t0, frame.FULL, [CarrierCandidate(1234.5, 0.0, 0.0)], **kw)
    g = tbd(x, t0, frame.FULL, [CarrierCandidate(1234.5, 0.0, 0.0)], **kw)
    _hits_same(r, g, "FULL")
    assert r[0].n_frames == 896 and len(r) > 30
    assert any(abs(h.f_hz - 1234.5) < 1.0 for h in r)                 # a bare carrier: no shape
    assert any(abs(h.f_hz - 1777.0) < 1.0 and h.waveform == WAVEFORM_L for h in r)


def test_the_a_b_ran_the_cpp(N, native, captures, monkeypatch):
    """The whole-caller A/B above compares acquire() with itself unless the
    rows really replace the reference: count the native calls it makes."""
    calls = {}
    a = native.qrss.acquire_tbd
    for name in ("track_before_detect", "_hit_to_detection", "frame_map"):
        fn = getattr(a, name)

        def spy(*args, _fn=fn, _name=name, **kw):
            calls[_name] = calls.get(_name, 0) + 1
            return _fn(*args, **kw)
        monkeypatch.setattr(a, name, spy)
    _apply(native, monkeypatch)
    cap, spec = captures["l"]
    joint = A.acquire(cap, (frame.SHORT, lframe.L_SHORT), preprocessed=True)
    joint.detections((frame.SHORT, lframe.L_SHORT))
    assert joint.tbd
    for name in ("track_before_detect", "_hit_to_detection", "frame_map"):
        assert calls.get(name, 0) > 0, (name, calls)          # TbdHit.f_hz: inside the C++

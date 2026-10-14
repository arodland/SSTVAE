"""Parity of native/core/qrss/track_timing (sstvae_native.qrss.track_timing)
with the rest of sstvae/qrss/track.py: the coarse timing fit (fit_timing,
_segment_timing, _parabola), gate V (verify, verify_track), the frequency
path measured on the frame (_carrier_lines, refine_path, line_path) and
_flat_path. tests/qrss_native/track_timing.py substitutes them under
--native-qrss=rx.

The inputs are the receiver's own: simulated CE passes through
`receiver.prepare` / `channel_for` at a flat 1500 Hz candidate (what
`receive_pass` gates), TINY and SHORT at -6 and -12 dB, a 1 Hz/min drift,
+100 ppm and -100 ppm clocks, and a capture with no signal in it (-80 dB)
that gate V must refuse. Spies on the reference record what its
fit_timing computed on the way (its grids, through `np.full`, and every
`_segment_timing` call).

The contract, and what was measured (2026-10-13, these scenarios;
QRSS_PARITY_WORST=1 with -s prints the worst of each this run):

- fit_timing: the same argmax cell and runners-up (every peak's
  _segment_timing input to 1e-9: measured 9e-13 samples, its clock error
  4e-12 samples of slip over the frame), the same timings in the same
  order, tau0 to 1e-6 CH samples (measured 8e-12) and ppm to 1e-6 samples
  of slip (5e-11), Timing.z to 1e-8 relative (5e-12). The C++ correlates
  by the direct gather where the reference's blind search transforms (588
  FFT pairs on FULL): its grid is the reference's FFT grid to 1e-12 of
  the grid's scale (measured 1.1e-13) and the reference's direct path (a
  local search; edge terms and non-finite edge samples included) to 1e-13
  (4.4e-15). The statistic x to 1e-12 of its scale (9.8e-16: scipy's FFT
  convolutions against the C++'s), sig2 to 1e-12 relative (2.3e-15);
- _segment_timing on the reference's own inputs: t0 to 1e-6 samples
  (8e-12), the clock error to 1e-6 samples of slip (6e-11), the score to
  1e-8 relative (2.3e-13);
- _carrier_lines: to the bit (asserted bitwise); refine_path and line_path: the
  same branch (unchanged, the 1 s fine grid or the 2 s coarse one) and the
  same grid to the bit, f to 1e-7 Hz (measured 4.5e-13), the caller's own
  path back when nothing was measured; _flat_path to 1e-9 Hz (4.5e-13);
- _fit_timing_tries (verify's blind whole-frame line takes its best
  timing and the runners-up from one coarse grid, 2026-10-14): in each
  implementation exactly fit_timing's n_best 1 and n_best V answers, to
  the bit; the C++ pair against the reference's to fit_timing's bars;
- verify: the same accept/reject in every scenario (the drift, both
  clocks, noise refused after the runners-up), the same path (so the same
  base and segment decisions), the timing to 1e-6 samples, Z_ref to 1e-7
  relative (measured 4.5e-8);
- verify_track: Z_ref to 1e-7 (2.5e-8) and receiver._ce_resid_ratio to
  1e-6 (1.1e-7) relative; z_d, m, c, N and psi to 1e-9 (5e-11); u, P and
  pu to the tracker core's bars, 1e-8, 1e-5 and 1e-6 (measured 1.1e-9,
  1.6e-6, 1.2e-7). Those are tests/test_qrss_native_track's numbers and
  its mechanism, not this module's: the reference's own Z_ref moves by
  1.4e-8 to 2.2e-8 when it is handed the C++ timing, 1e-12 samples from
  its own (TINY -12 dB, the drift), and the Kalman smoother's diffuse
  start keeps about seven digits of P. On the noise capture, q at the
  grid's floor, the reference moves its Z_ref by 4.8e-5 for a 4.5e-13
  sample change of timing: there Z_ref (N(0, 1)) is held to 1e-4
  absolute (measured 6.1e-6) and P to 1e-3 (1.3e-4);
- the composition is exact: the reference's verify_track and verify with
  the C++ primitives assigned over the ones they call are the C++
  verify_track and verify bit for bit, every scenario
  (test_gate_v_composition), so the tolerances above are the primitives';
- receive_pass with both rows files (track.py's and this one) over
  track's names and ltrack's from-imports, on CE TINY, CE SHORT and an L
  TINY pass: the same passes and headers, f_hz to 1e-6 Hz, latents to
  1e-6; and the other callers of gate V's pieces on the same captures,
  receiver._gate (verify, then the verify_track row for _ce_resid_ratio)
  and em.pass_path (line_path, refine_path, verify_track with fit=False on
  a stored pass's CH-rate capture): the same detections and paths;
- the reference's refusals, type and text: an L frame, a malformed
  erase span;
- probes (review, 2026-10-13): fit_timing with randomised arguments on the
  scenarios' own z_d (fractional t0_index moves, local searches on both
  sides of the reference's 64-lag FFT switch, a one-row clock-error grid,
  n_best -1 to 11, known-symbol subsets in no order), the path
  measurements on random captures (frames starting before or far into
  them, blanked thirds, segment lengths 0.5 to 60 s, median bands, all
  four refine/line branches), _segment_timing on planted pulses (every
  branch of its walk), and gate V with every known symbol erased and with
  an A2-like candidate (2x2 covariance, a 120-sample local search);
- a real signal with q at the grid's floor (TINY -15 dB, no fading): the
  scenarios above hold q at grid index 1-2, and there the 1e-7 Z_ref bar
  is honest; at the floor it is not, because the reference itself moves
  Z_ref by 7.8e-7 and P by 8.4e-5 relative for a 1e-13-sample nudge of
  its own timing. Same decision, timing, path, z_d/m/N/psi to the bit;
  Z_ref held to 1e-5 relative (measured 3.7e-7 through verify, 2.9e-7
  through verify_track), u to 1e-7 (4.3e-9), P to 1e-3 (4.6e-5).
"""

import dataclasses
import math
import os
from fractions import Fraction
from types import SimpleNamespace

import numpy as np
import pytest

from qrss_demod_tracks import CALL, HDR, flat_det, rel
from qrss_helpers import Q_TEST, unit_rms_latents
from qrss_native import _rx, require
from sstvae.qrss import channel as chm
from sstvae.qrss import frame, frontend, header, lframe, morse
from sstvae.qrss import ltrack as LT
from sstvae.qrss import receiver as RX
from sstvae.qrss import track as T
from sstvae.qrss.channel import ChannelConfig
from sstvae.qrss.constants import CH_FS, WAVEFORM_L
from sstvae.qrss.precoder import precode
from sstvae.qrss.types import FreqPath, Timing

TRACK = "sstvae.qrss.track"

# label: (frame, SNR dB, seed, further ChannelConfig fields)
SCENARIOS = {
    "tiny-12": ("tiny", -12.0, 1, {}),
    "tiny-6": ("tiny", -6.0, 2, {}),
    "short-6": ("short", -6.0, 1, {}),
    "short-12": ("short", -12.0, 3, {}),
    "drift": ("tiny", -6.0, 4, {"drift_hz_per_min": 1.0}),
    "ppm+100": ("tiny", -9.0, 5, {"tx_ppm": 100.0}),
    "ppm-100": ("tiny", -9.0, 6, {"rx_ppm": -100.0}),
    "noise": ("tiny", -80.0, 7, {}),
}
FROM_IMPORTS = ("derotate", "kalman_rts", "select_q", "_smooth", "_keying_tuple", "_parabola",
                "_flat_path")


@pytest.fixture(scope="module")
def Tt(native):
    return require(native, "track_timing")


@pytest.fixture(scope="module")
def row(native, Tt):
    from qrss_native.track_timing import rows

    return {name: fn for (_t, name), fn in rows(native.qrss).items()}


def _sim(name, snr, seed, cfg):
    spec = frame.get(name)
    a = unit_rms_latents(spec.n_data, seed)
    sym = frame.assemble(spec, header.encode(HDR) if spec.n_hdr else None, precode(a, Q_TEST))
    sim = chm.simulate(sym, spec, Q_TEST, ChannelConfig(snr_db=snr, seed=seed, **cfg),
                       carrier_hz=1500.0, keying=morse.keying_units(CALL) if spec.n_win else None)
    return spec, sim


@pytest.fixture(scope="module")
def scen(Tt):
    """{label: scenario}: the capture prepared, the flat candidate's channel
    and the reference's gate V on it."""
    out = {}
    for label, (name, snr, seed, cfg) in SCENARIOS.items():
        spec, sim = _sim(name, snr, seed, cfg)
        prep = RX.prepare(frontend.Capture(sim.fe, Q_TEST, sim.t0_index))
        det = flat_det(1500.0)
        chan = RX.channel_for(prep, spec, det)
        chan.noise_psd_at(0.0)
        tracked = []                         # fit=... of each verify_track gate V ran
        orig = T.verify_track

        def spy(*a, _orig=orig, _calls=tracked, **k):
            _calls.append(k.get("fit", True))
            return _orig(*a, **k)

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(T, "verify_track", spy)
            v = T.verify(chan, CH_FS, spec, det)
        out[label] = SimpleNamespace(spec=spec, sim=sim, prep=prep, det=det, chan=chan, v=v,
                                     tracked=tracked)
    assert out["noise"].v is None
    assert all(s.v is not None for k, s in out.items() if k != "noise"), \
        [k for k, s in out.items() if s.v is None]
    return out


class _NpSpy:
    """numpy, recording the 2-D -inf grids fit_timing allocates (and then
    fills in place)."""

    def __init__(self, grids):
        self._grids = grids

    def __getattr__(self, name):
        return getattr(np, name)

    def full(self, shape, fill_value, *a, **k):
        out = np.full(shape, fill_value, *a, **k)
        if isinstance(shape, tuple) and len(shape) == 2 and fill_value == -np.inf:
            self._grids.append(out)
        return out


def _ref_fit(*args, **kw):
    """The reference's fit_timing, with its grids and _segment_timing calls."""
    grids, segs = [], []
    orig = T._segment_timing

    def spy(x, sig2, kn, t0abs, delta, sps, n_pos, *a, **k):
        r = orig(x, sig2, kn, t0abs, delta, sps, n_pos, *a, **k)
        segs.append(SimpleNamespace(x=x, sig2=sig2, kn=kn, t0abs=t0abs, delta=delta, sps=sps,
                                    n_pos=n_pos, out=r))
        return r

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(T, "np", _NpSpy(grids))
        mp.setattr(T, "_segment_timing", spy)
        out = T.fit_timing(*args, **kw)
    return out, grids, segs


def _cells(Z, n_best, sps):
    """The reference's peak cells on its own grid (fit_timing's loop)."""
    Zl, il = Z.max(axis=0), Z.argmax(axis=0)
    out = []
    for k in range(max(1, int(n_best))):
        j = int(np.argmax(Zl))
        if k and not np.isfinite(Zl[j]):
            break
        out.append((int(il[j]), j))
        Zl[max(j - int(math.ceil(sps)), 0):j + int(math.ceil(sps)) + 1] = -np.inf
    return out


# Worst value of each compared quantity this run: QRSS_PARITY_WORST=1 prints them
# (pytest -s), which is how the figures in the module docstring were taken.
_WORST = {}


@pytest.fixture(scope="module", autouse=True)
def _print_worst():
    yield
    if os.environ.get("QRSS_PARITY_WORST"):
        for k in sorted(_WORST):
            print(f"{k:28s} {_WORST[k]:.3g}")


def _le(key, value, bar, what=""):
    """value <= bar, the worst value per key kept for the record."""
    value = float(value)
    _WORST[key] = max(_WORST.get(key, 0.0), value)
    assert value <= bar, (key, what, value, bar)


def _r(g, r, floor=1e-300):
    return abs(g - r) / max(abs(r), floor)


def _timing_close(g, r, spec, what, z_tol=1e-8):
    sps = spec.t_sym * CH_FS
    _le("timing tau0", abs(g.tau0 - r.tau0), 1e-6, what)
    _le("timing slip", abs((g.ppm - r.ppm) * 1e-6 * spec.n_pos * sps), 1e-6, what)
    assert g.gamma == r.gamma, what
    _le("timing z", abs(g.z - r.z) / max(abs(r.z), 1.0), z_tol, what)
    assert g.cov.shape == r.cov.shape and np.allclose(g.cov, r.cov, rtol=1e-12, atol=0.0), what


def _path_close(g, r, what, tol=1e-7):
    assert g.t_s.shape == r.t_s.shape, (what, g.t_s.shape, r.t_s.shape)
    assert np.array_equal(g.t_s, r.t_s), what
    _le("path f_hz", np.max(np.abs(g.f_hz - r.f_hz)), tol, what)
    assert np.array_equal(g.weight, r.weight), what


def _outcome(fn, *args, **kw):
    try:
        out = fn(*args, **kw)
    except (ValueError, TypeError, IndexError, ZeroDivisionError, AssertionError, AttributeError) as e:
        return type(e), str(e)
    return "ok", out


# --- the coarse timing fit ----------------------------------------------------------------------

@pytest.fixture(scope="module")
def blind(Tt, scen):
    """{label: (path fn, z_d, reference (out, grids, segs), C++ (out, trace))}: the
    blind search with n_best = V_TIMING_TRIES, on gate V's path (the flat one
    where gate V refused)."""
    out = {}
    for label, s in scen.items():
        fn = T.freq_path_spline((s.v or s.det).path)
        z_d = T.derotate(s.chan, fn)
        kw = dict(t0_index=s.chan.t0_index, z_d=z_d, n_best=T.V_TIMING_TRIES)
        ref = _ref_fit(s.chan, CH_FS, fn, s.spec, **kw)
        got = Tt._fit_timing_trace(s.chan, CH_FS, fn, s.spec, **kw)
        out[label] = SimpleNamespace(fn=fn, z_d=z_d, ref=ref, got=got)
    return out


def test_fit_timing_blind(Tt, row, scen, blind):
    for label, b in blind.items():
        spec = scen[label].spec
        sps = spec.t_sym * CH_FS
        (r_out, grids, segs), (g_out, tr) = b.ref, b.got
        assert len(grids) == len(T.CARRIER_LP_HZ) == len(tr["sig2"]), label
        # the same peaks, in the same order: each _segment_timing call's inputs and result
        peaks = tr["peaks"]
        assert len(peaks) == len(segs), (label, len(peaks), len(segs))
        lps = []
        for k, (p, s) in enumerate(zip(peaks, segs)):
            what = f"{label} peak {k}"
            lps.append(p["lp"])
            _le("peak t0abs_in", abs(p["t0abs_in"] - s.t0abs), 1e-9, what)
            _le("peak delta_in slip", abs((p["delta_in"] - s.delta) * spec.n_pos * sps), 1e-9, what)
            t0, d, score = s.out
            _le("peak t0abs", abs(p["t0abs"] - t0), 1e-6, what)
            _le("peak delta slip", abs((p["delta"] - d) * spec.n_pos * sps), 1e-6, what)
            _le("peak score", abs(p["score"] - score) / max(abs(score), 1e-4), 1e-8, what)
            _le("sig2", _r(tr["sig2"][p["lp"]], s.sig2), 1e-12, what)
        assert lps == sorted(lps), label
        # the grid, the statistic and the cells, per carrier estimate
        kn = T.known_symbols(spec)
        deltas = np.arange(-tr["nd"], tr["nd"] + 1) * tr["step"]
        for lp_i, (lp, Z) in enumerate(zip(T.CARRIER_LP_HZ, grids)):
            s = next(s for p, s in zip(peaks, segs) if p["lp"] == lp_i)
            x = Tt._fit_statistic(b.z_d, lp)
            assert x.shape == s.x.shape and x.dtype == s.x.dtype
            _le("statistic x", np.max(np.abs(x - s.x)) / np.max(np.abs(s.x)), 1e-12, (label, lp))
            assert Z.shape == (len(deltas), tr["nlag"]), label
            norm = math.sqrt(s.sig2 * float(np.sum(kn.val * kn.val)))
            G = Tt._fit_grid(s.x, kn, tr["base"], tr["nlag"], tr["frac0"], deltas, sps, norm)
            assert G.shape == Z.shape
            _le("grid vs FFT", np.max(np.abs(G - Z)) / np.max(np.abs(Z)), 1e-12, (label, lp))
            mine = [(p["i"], p["j"]) for p in peaks if p["lp"] == lp_i]
            assert mine == _cells(Z, T.V_TIMING_TRIES, sps), (label, lp)
        # the timings returned
        assert len(g_out) == len(r_out), label
        for k, (g, r) in enumerate(zip(g_out, r_out)):
            g = _rx.to_ref(g)
            _timing_close(g, r, spec, f"{label} timing {k}")
        # through the row (n_best 1: one Timing, not a list)
        r1 = T.fit_timing(scen[label].chan, CH_FS, b.fn, spec, t0_index=scen[label].chan.t0_index,
                          z_d=b.z_d)
        g1 = row["fit_timing"](scen[label].chan, CH_FS, b.fn, spec, t0_index=scen[label].chan.t0_index,
                               z_d=b.z_d)
        _timing_close(g1, r1, spec, f"{label} n_best 1")
        _timing_close(g1, r_out[0], spec, f"{label} n_best 1 is the best")


def _timing_same(g, r, what):
    """Two timings to the bit, type for type."""
    for f in ("tau0", "ppm", "gamma", "z"):
        a, b = getattr(r, f), getattr(g, f)
        assert type(b) is type(a) and (a == b or (a != a and b != b)), (what, f, a, b)
    assert g.cov.dtype == r.cov.dtype and np.array_equal(g.cov, r.cov), what


def test_fit_timing_tries(row, scen, blind):
    """verify's blind whole-frame line: one coarse grid gives the best
    timing and the runners-up. In each implementation _fit_timing_tries
    is (fit_timing n_best 1, fit_timing n_best n) to the bit; the C++ pair
    is the reference's to fit_timing's bars (test_fit_timing_blind)."""
    V = T.V_TIMING_TRIES
    assert V > 1
    for label, b in blind.items():
        s = scen[label]
        spec = s.spec
        for n_best, kw in ((V, {}), (3, {}), (2, dict(ppm_range=50.0)), (1, {})):
            kw = dict(t0_index=s.chan.t0_index, z_d=b.z_d, **kw)
            what = (label, n_best, kw.get("ppm_range"))
            for name, fit_timing, tries in (("reference", T.fit_timing, T._fit_timing_tries),
                                            ("native", row["fit_timing"], row["_fit_timing_tries"])):
                best, tms = tries(s.chan, CH_FS, b.fn, spec, None, None, n_best=n_best, **kw)
                _timing_same(best, fit_timing(s.chan, CH_FS, b.fn, spec, n_best=1, **kw), (name, what))
                want = fit_timing(s.chan, CH_FS, b.fn, spec, n_best=n_best, **kw)
                if n_best == 1:
                    _timing_same(tms, want, (name, what))
                else:
                    assert type(tms) is list and len(tms) == len(want) >= 1, (name, what)
                    for k, (g, r) in enumerate(zip(tms, want)):
                        _timing_same(g, r, (name, what, k))
                    # not necessarily tms[0]: a carrier estimate's runner-up can
                    # outscore every estimate's first peak once refined (the L
                    # noise scenario), and verify still skips tms[0], as it did
                if name == "reference":
                    ref = (best, tms)
                else:
                    _timing_close(best, ref[0], spec, (what, "best"))
                    pairs = zip(tms, ref[1]) if n_best > 1 else [(tms, ref[1])]
                    for k, (g, r) in enumerate(pairs):
                        _timing_close(g, r, spec, (what, k))
    # its refusal is fit_timing's
    r = _outcome(T._fit_timing_tries, None, CH_FS, None, lframe.L_TINY, t0_index=0.0)
    g = _outcome(row["_fit_timing_tries"], None, CH_FS, None, lframe.L_TINY, t0_index=0.0)
    assert r[0] is AssertionError and g[:2] == r[:2], (r, g)
    assert _outcome(T.fit_timing, None, CH_FS, None, lframe.L_TINY, t0_index=0.0)[:2] == r[:2]


def test_fit_timing_local(Tt, row, scen, blind):
    """tau0_init: the reference's direct path (nlag <= 64), grids compared."""
    for label in ("tiny-12", "short-6", "ppm+100", "noise"):
        s, b = scen[label], blind[label]
        spec = s.spec
        sps = spec.t_sym * CH_FS
        best = b.ref[0][0]
        for kw in (dict(tau0_init=best.tau0 + 0.7),
                   dict(tau0_init=best.tau0 - 1.3, tau_range=10.0, n_best=3),
                   dict(tau0_init=best.tau0, tau_range=0.4, ppm_range=60.0),
                   dict(tau0_init=best.tau0 + 2.0, tmpl_fn=T.known_symbols(spec)),
                   dict(tau0_init=best.tau0, tmpl_fn=lambda *a: None)):
            kw = dict(kw, t0_index=s.chan.t0_index, z_d=b.z_d)
            ref, grids, segs = _ref_fit(s.chan, CH_FS, b.fn, spec, **kw)
            g, tr = Tt._fit_timing_trace(s.chan, CH_FS, b.fn, spec, **kw)
            assert tr["nlag"] <= 64 and len(tr["peaks"]) == len(segs), (label, kw)
            kn = T.known_symbols(spec)
            deltas = np.arange(-tr["nd"], tr["nd"] + 1) * tr["step"]
            for p, sg in zip(tr["peaks"], segs):
                _le("peak t0abs_in", abs(p["t0abs_in"] - sg.t0abs), 1e-9, (label, kw))
            for lp_i, Z in enumerate(grids):
                sg = next(sg for p, sg in zip(tr["peaks"], segs) if p["lp"] == lp_i)
                norm = math.sqrt(sg.sig2 * float(np.sum(kn.val * kn.val)))
                G = Tt._fit_grid(sg.x, kn, tr["base"], tr["nlag"], tr["frac0"], deltas, sps, norm)
                _le("grid vs direct", np.max(np.abs(G - Z)) / np.max(np.abs(Z)), 1e-13, (label, kw))
            got = row["fit_timing"](s.chan, CH_FS, b.fn, spec, **kw)
            if kw.get("n_best", 1) <= 1:
                ref, got = [ref], [got]
            assert len(got) == len(ref), (label, kw)
            for gg, rr in zip(got, ref):
                _timing_close(gg, rr, spec, f"{label} {kw}")


def test_fit_timing_without_z_d(row, scen):
    """z_d None: derotated from the path (a FreqPath or a callable)."""
    s = scen["tiny-6"]
    for path in (s.v.path, T.freq_path_spline(s.v.path)):
        r = T.fit_timing(s.chan, CH_FS, path, s.spec, None, s.v.timing.tau0 + 0.5, t0_index=s.chan.t0_index)
        g = row["fit_timing"](s.chan, CH_FS, path, s.spec, None, s.v.timing.tau0 + 0.5,
                              t0_index=s.chan.t0_index)
        _timing_close(g, r, s.spec, "z_d None")


def test_fit_grid_edges(Tt):
    """Terms whose interpolation pair leaves x, and non-finite edge samples:
    the reference's direct path (`(...) * ok`, so inf * 0 is NaN), worked
    here in numpy as the reference writes it."""
    rng = np.random.default_rng(5)
    kn = T.known_symbols(frame.TINY)
    sps = frame.TINY.t_sym * CH_FS
    span = int(kn.pos.max() * sps) + 1
    for nx, base, nlag, edge in ((span + 40, -30, 50, None), (span + 40, -30, 50, "inf"),
                                 (span + 300, -20, 200, "nan"), (span + 300, 12, 200, None),
                                 (span - 100, -5, 90, None)):
        x = rng.standard_normal(nx)
        if edge == "inf":
            x[-1] = np.inf
        elif edge == "nan":
            x[0] = np.nan
        frac0 = 0.37
        deltas = np.arange(-3, 4) * 2.1e-5
        norm = 3.5
        ref = np.full((len(deltas), nlag), -np.inf)
        lags = np.arange(nlag)
        for i, d in enumerate(deltas):
            o = kn.pos * sps * (1 + d) + frac0
            idx = base + o[:, None] + lags[None, :]
            i0 = np.floor(idx).astype(np.int64)
            f = idx - i0
            ok = (i0 >= 0) & (i0 + 1 < len(x))
            i0c = np.clip(i0, 0, len(x) - 2)
            with np.errstate(invalid="ignore"):
                v = (x[i0c] * (1 - f) + x[i0c + 1] * f) * ok
            ref[i] = kn.val @ v / norm
        got = Tt._fit_grid(x, kn, base, nlag, frac0, deltas, sps, norm)
        assert np.array_equal(np.isnan(got), np.isnan(ref)), (nx, base, edge)
        fin = np.isfinite(ref)
        assert np.array_equal(fin, np.isfinite(got))
        assert np.array_equal(got[np.isinf(ref)], ref[np.isinf(ref)])
        _le("grid edges", np.max(np.abs(got[fin] - ref[fin])) / np.max(np.abs(ref[fin])), 1e-13,
            (nx, base, edge))


def test_segment_timing(row, scen, blind):
    for label, b in blind.items():
        spec = scen[label].spec
        sps = spec.t_sym * CH_FS
        s = b.ref[2][0]
        for t0abs, delta, seg in ((s.t0abs, s.delta, T.SEG_TIMING_S), (s.t0abs, s.delta, 7.0),
                                  (s.t0abs + 1.5, s.delta, 40.0), (s.t0abs, s.delta + 5e-5, T.SEG_TIMING_S),
                                  (s.t0abs - 0.4, s.delta + 1e-3, T.SEG_TIMING_S),
                                  (s.t0abs, s.delta, 1000.0)):
            r = T._segment_timing(s.x, s.sig2, s.kn, t0abs, delta, sps, spec.n_pos, seg)
            g = row["_segment_timing"](s.x, s.sig2, s.kn, t0abs, delta, sps, spec.n_pos, seg)
            what = (label, t0abs - s.t0abs, delta - s.delta, seg)
            _le("segment t0", abs(g[0] - r[0]), 1e-6, what)
            _le("segment slip", abs((g[1] - r[1]) * spec.n_pos * sps), 1e-6, what)
            _le("segment score", abs(g[2] - r[2]) / max(abs(r[2]), 1e-4), 1e-8, what)


# --- the frequency path ---------------------------------------------------------------------------

def test_carrier_lines(row, scen):
    for label, s in scen.items():
        p = s.det.path
        z = T.derotate(s.chan, T.PathFn(p.t_s, p.f_hz, p.weight, 60.0))
        for seg_s, span in ((T.PATH_SEG_S, T.PATH_SPAN_HZ), (T.PATH_FINE_S, 1.0), (13.0, 3.0)):
            for kw in ({}, dict(med_band=(-30.0, -10.0)), dict(ratio=4.0)):
                r = T._carrier_lines(z, s.chan, s.spec, seg_s, span, **kw)
                g = row["_carrier_lines"](z, s.chan, s.spec, seg_s, span, **kw)
                what = (label, seg_s, span, kw)
                # to the bit: the same z gives the same spectra (pocketfft, the
                # Hann window through libm cos, numpy's median)
                for a, b in zip(g, r):
                    assert a.dtype == b.dtype and np.array_equal(a, b), what


def test_refine_and_line_path(row, scen):
    branches = set()
    for label, s in scen.items():
        bases = [s.det.path, T._flat_path(s.det.path)] + ([s.v.path] if s.v is not None else [])
        for k, base in enumerate(bases):
            kws = ({}, dict(seg_s=10.0, span_hz=1.0), dict(med_band=(-30.0, -10.0)),
                   dict(fine_ratio=1e9), dict(fine_share=0.0, fine_ratio=0.0), dict(ratio=1e6))
            for kw in kws if k == 0 else ({},):
                r = T.refine_path(s.chan, s.spec, base, **kw)
                g = row["refine_path"](s.chan, s.spec, base, **kw)
                what = (label, k, kw)
                assert (g is base) == (r is base), what
                if r is base:
                    branches.add("unchanged")
                    continue
                branches.add("fine" if np.allclose(np.diff(r.t_s), 1.0) else "coarse")
                _path_close(g, r, what)
            r, g = T.line_path(s.chan, s.spec, base), row["line_path"](s.chan, s.spec, base)
            assert (g is base) == (r is base), (label, k)
            if r is not base:
                _path_close(g, r, (label, k, "line"))
            r, g = T._flat_path(base), row["_flat_path"](base)
            assert np.array_equal(g.t_s, r.t_s) and np.array_equal(g.weight, r.weight)
            _le("flat_path f", np.max(np.abs(g.f_hz - r.f_hz)), 1e-9, (label, k))
    assert branches == {"unchanged", "fine", "coarse"}, branches
    # captures cut short: under three line segments (line_path hands the
    # path back) and under two path segments (so does refine_path)
    s = scen["short-6"]
    for secs, unchanged in ((100, {"line_path"}), (15, {"line_path", "refine_path"})):
        cut = dataclasses.replace(s.chan, ch=s.chan.ch[:int(s.chan.t0_index) + secs * CH_FS])
        for fn in ("line_path", "refine_path"):
            r, g = getattr(T, fn)(cut, s.spec, s.det.path), row[fn](cut, s.spec, s.det.path)
            assert (r is s.det.path) == (fn in unchanged) == (g is s.det.path), (secs, fn)
            if r is not s.det.path:
                _path_close(g, r, (secs, fn))


# --- gate V ----------------------------------------------------------------------------------------

def _det_close(g, r, spec, what):
    assert (g is None) == (r is None), (what, g, r)
    if r is None:
        return
    assert (g.f_hz, g.method, g.lead_in_s, g.waveform) == (r.f_hz, r.method, r.lead_in_s, r.waveform), what
    _le("verify z_ref", _r(g.z_ref, r.z_ref), 1e-7, what)
    _timing_close(g.timing, r.timing, spec, what)
    _path_close(g.path, r.path, what)


def test_verify(row, scen):
    """Blind on the flat candidate (every scenario, the reference's own
    gate V from the fixture), then with the timing known, a narrower
    search and a span erased."""
    for label, s in scen.items():
        _det_close(row["verify"](s.chan, CH_FS, s.spec, s.det), s.v, s.spec, label)
    for label, kw in (("tiny-12", dict(fit=False)), ("tiny-12", dict(tau_range=6.0)),
                      ("tiny-12", dict(erase_s=[(10.0, 20.0)])), ("short-6", dict(fit=False)),
                      ("drift", dict(fit=False, erase_s=((5.0, 6.0), (30.0, 31.0))))):
        s = scen[label]
        cand = dataclasses.replace(s.det, timing=s.v.timing)
        r = T.verify(s.chan, CH_FS, s.spec, cand, **kw)
        g = row["verify"](s.chan, CH_FS, s.spec, cand, **kw)
        _det_close(g, r, s.spec, (label, kw))


def test_verify_noise_tries_the_runners_up(row, scen):
    """The noise capture is refused only after every base and the coarse
    statistic's runners-up were tracked (the reference's calls, counted in
    the fixture); a signal is accepted on the first. The two refined bases
    fit their timing (fit=True); the whole-frame line's best timing and
    its runners-up come from one grid (_fit_timing_tries) and are tracked
    as given (fit=False)."""
    calls = scen["noise"].tracked
    assert len(calls) == 2 + T.V_TIMING_TRIES and calls[:2] == [True, True], calls
    assert calls.count(False) == len(calls) - 2, calls
    assert scen["tiny-6"].tracked == [True]
    s = scen["noise"]
    assert row["verify"](s.chan, CH_FS, s.spec, s.det) is None


def _track_close(g, r, what, noise=False):
    """A C++ gate-V track against the reference's. On a signal: Z_ref and R
    to the tracker's conditioning, z_d..psi to 1e-9, u, P, pu to the
    tracker core's bars. On noise (q at the grid's floor) the reference's
    own smoother is that much worse conditioned: Z_ref (N(0, 1) there) to
    1e-4 absolute, P to 1e-3 (module docstring)."""
    assert (g.q, g.lead_in_s, g.ook) == (r.q, r.lead_in_s, r.ook), what
    assert np.array_equal(g.pos, r.pos), what
    tag = "noise " if noise else ""
    if noise:
        _le("noise track z_ref (abs)", abs(g.z_ref - r.z_ref), 1e-4, what)
    else:
        _le("track z_ref", _r(g.z_ref, r.z_ref), 1e-7, what)
        _le("track resid_ratio", _r(RX._ce_resid_ratio(g), RX._ce_resid_ratio(r)), 1e-6, what)
    for f in ("z_d", "m", "c", "N", "psi"):
        _le(f"{tag}track {f}", rel(getattr(g, f), getattr(r, f)), 1e-9, (what, f))
    for f, tol in (("u", 1e-7 if noise else 1e-8), ("P", 1e-3 if noise else 1e-5), ("pu", 1e-6)):
        _le(f"{tag}track {f}", rel(getattr(g, f), getattr(r, f)), tol, (what, f))
    assert g.chan is r.chan and g.spec is r.spec, what


def _z_ref_of(tr):
    """The reference's Z_ref statistic over a track's own arrays (verify_track's last step)."""
    spec, classes = tr.spec, tr.classes
    keyed_from = spec.keyed_start_pos - tr.lead_in_s / spec.t_sym
    c_known, _ = T.templates(spec, tr.pos, classes.mu, classes.nu, None, False, keyed_from)
    return T.z_ref_stat(tr, c_known)


def test_verify_track(row, scen):
    """Every scenario on gate V's own candidate (timing known, fit=False:
    identical inputs), and the fits: blind, local, a narrower local and a
    span erased. Z_ref is the reference's statistic of the C++ track's own
    arrays to 1e-12; against the reference's track it carries the
    tracker's conditioning (module docstring)."""
    runs = [(label, s.v, dict(fit=False)) for label, s in scen.items() if s.v is not None]
    for label in ("tiny-12", "short-12", "drift", "noise"):
        s = scen[label]
        runs.append((label, dataclasses.replace(s.det, path=(s.v or s.det).path), {}))
    for label in ("tiny-12", "short-6"):
        s = scen[label]
        runs += [(label, s.v, {}), (label, s.v, dict(tau_range=2.0)),
                 (label, s.v, dict(fit=False, erase_s=[(10.0, 20.0)]))]
    for label, cand, kw in runs:
        s = scen[label]
        r = T.verify_track(s.chan, s.spec, cand, **kw)
        g = row["verify_track"](s.chan, s.spec, cand, **kw)
        what = (label, cand.timing is None, kw)
        _track_close(g, r, what, noise=label == "noise")
        _timing_close(g.timing, r.timing, s.spec, what)
        assert (g.timing is cand.timing) == (r.timing is cand.timing), what
        _le("track z_ref own arrays", _r(g.z_ref, _z_ref_of(g)), 1e-12, what)


def _native_primitives(native, mp, *, gate=False):
    """Assign the C++ primitives that verify_track composes over the
    reference's names (the path spline, derotate, the coarse fit, the
    tracker, its classes, templates and statistics); with gate=True also
    those verify composes (verify_track itself and the path measurements)."""
    from qrss_native.track import rows as core_rows
    from qrss_native.trackbase import rows as base_rows
    from qrss_native.track_timing import rows as timing_rows

    table = {**base_rows(native.qrss), **core_rows(native.qrss), **timing_rows(native.qrss)}
    names = ["PathFn", "freq_path_spline", "derotate", "fit_timing", "make_classes", "track",
             "templates", "z_ref_stat", "erase"]
    if gate:
        names += ["verify_track", "refine_path", "line_path", "_flat_path", "_fit_timing_tries"]
    for name in names:
        mp.setattr(T, name, table[(TRACK, name)])


def test_gate_v_composition(native, row, scen):
    """The reference's verify_track and verify with the C++ primitives they
    compose assigned over their own are the C++ verify_track and verify bit
    for bit (Z_ref, the timing, every array, the path): the glue -- the
    local search's range, the classes, the tracker's arguments, the erasure,
    keyed_from, the bases tried in order, the runners-up, the choice and the
    acceptance -- is the reference's exactly, and every difference above is
    a primitive's, measured in that primitive's own parity test."""
    fields = ("pos", "z_d", "m", "c", "u", "P", "N", "psi", "pu")
    for label, s in scen.items():
        cands = [(dataclasses.replace(s.det, path=(s.v or s.det).path), {})]
        if s.v is not None:
            cands += [(s.v, dict(fit=False)), (s.v, dict(tau_range=2.0, erase_s=[(10.0, 20.0)]))]
        for cand, kw in cands:
            g = row["verify_track"](s.chan, s.spec, cand, **kw)
            with pytest.MonkeyPatch.context() as mp:
                _native_primitives(native, mp)
                h = T.verify_track(s.chan, s.spec, cand, **kw)
            what = (label, kw)
            for f in fields:
                assert np.array_equal(getattr(g, f), getattr(h, f)), (what, f)
            assert (g.q, g.loglik, g.z_ref) == (h.q, h.loglik, h.z_ref), what
            for f in ("tau0", "ppm", "gamma", "z"):
                assert getattr(g.timing, f) == getattr(h.timing, f), (what, f)
        g = row["verify"](s.chan, CH_FS, s.spec, s.det)
        with pytest.MonkeyPatch.context() as mp:
            _native_primitives(native, mp, gate=True)
            h = T.verify(s.chan, CH_FS, s.spec, s.det)
        assert (g is None) == (h is None), label
        if h is not None:
            assert g.z_ref == h.z_ref and g.timing.tau0 == h.timing.tau0 and g.timing.ppm == h.timing.ppm
            assert np.array_equal(g.path.t_s, h.path.t_s) and np.array_equal(g.path.f_hz, h.path.f_hz)


def test_refusals(row):
    spec = lframe.L_TINY
    for fn, args, kw in (("fit_timing", (None, CH_FS, None, spec), dict(t0_index=0.0)),
                         ("verify", (None, CH_FS, spec, None), {}),
                         ("verify_track", (None, spec, None), {})):
        r, g = _outcome(getattr(T, fn), *args, **kw), _outcome(row[fn], *args, **kw)
        assert r[0] is AssertionError and g == r, (fn, r, g)


def test_erase_span_refusals(row, scen):
    s = scen["tiny-6"]
    for bad in ([(1.0, 2.0, 3.0)], [(1.0,)], [1.0]):
        for fn, args in (("verify_track", (s.chan, s.spec, s.v)), ("verify", (s.chan, CH_FS, s.spec, s.v))):
            r = _outcome(getattr(T, fn), *args, fit=False, erase_s=bad)
            g = _outcome(row[fn], *args, fit=False, erase_s=bad)
            assert r[0] != "ok" and g == r, (fn, bad, r, g)


# --- probes: randomised arguments and edges (review, 2026-10-13) ------------------------------------

def test_probe_fit_timing_random(row, scen, blind):
    """fit_timing on the scenarios' own z_d with randomised arguments, each
    scenario through each kind: t0_index moved by up to half a sample; a
    local search wide enough for the reference's FFT path (40 to 160
    samples either side, so its grid is a transform and the C++'s a
    gather); a blind search over 20 to 400 ppm with n_best from -1 to 11; a
    random half of the known symbols, in no order (tmpl_fn); and a narrow
    search whose clock-error grid can be one row. The same timings in the
    same order (on the noise capture: the same noise peaks)."""
    rng = np.random.default_rng(20261013)
    labels = ("tiny-12", "ppm-100", "drift", "noise")
    for trial in range(16):
        label, kind = labels[trial % 4], trial // 4
        s, b = scen[label], blind[label]
        spec, best = s.spec, b.ref[0][0]
        kw = dict(t0_index=s.chan.t0_index + float(rng.uniform(-0.5, 0.5)), z_d=b.z_d)
        if kind == 0:
            kw.update(tau0_init=best.tau0 + float(rng.uniform(-4, 4)), tau_range=float(rng.uniform(40, 160)))
        elif kind == 1:
            kw.update(ppm_range=float(rng.uniform(20, 400)), n_best=int(rng.integers(-1, 12)))
        elif kind == 2:
            kn = T.known_symbols(spec)
            sub = rng.permutation(len(kn.pos))[:len(kn.pos) // 2]
            kw.update(tmpl_fn=T._Known(pos=kn.pos[sub], val=kn.val[sub]),
                      tau0_init=best.tau0 + float(rng.uniform(-2, 2)), tau_range=float(rng.uniform(2, 20)))
        else:
            kw.update(tau0_init=best.tau0 + float(rng.uniform(-1, 1)), tau_range=float(rng.uniform(0, 3)),
                      ppm_range=float(rng.uniform(0, 40)), n_best=2)
        r = T.fit_timing(s.chan, CH_FS, b.fn, spec, **kw)
        g = row["fit_timing"](s.chan, CH_FS, b.fn, spec, **kw)
        what = (trial, label, {k: v for k, v in kw.items() if k not in ("z_d", "tmpl_fn")})
        assert type(g) is type(r), what
        rl, gl = (r, g) if isinstance(r, list) else ([r], [g])
        assert len(gl) == len(rl), what
        for k, (gg, rr) in enumerate(zip(gl, rl)):
            _timing_close(gg, rr, spec, (what, k))


def _probe_chan(rng, spec):
    """A capture holding a carrier (possibly none) at a random offset and
    drift in unit noise, its frame starting anywhere from before the
    capture to far into it, sometimes with its first third blanked (zeros)."""
    dur = spec.n_pos * spec.t_sym
    t0 = float(rng.choice([-300.5, 0.0, 37.25, 1000.0, 3000.7, rng.uniform(-500, 4000)]))
    n = int(max(t0, 0) + dur * CH_FS * rng.uniform(0.2, 1.3)) + int(rng.integers(0, 2000))
    t = np.arange(n)
    f0, drift = rng.uniform(-3, 3), rng.uniform(-0.05, 0.05)
    turns = np.cumsum(f0 + drift * (t - t0) / CH_FS) / CH_FS
    z = rng.choice([0.0, 0.1, 1.0, 5.0]) * np.exp(2j * np.pi * (turns % 1.0))
    z = z + rng.standard_normal(n) + 1j * rng.standard_normal(n)
    if rng.uniform() < 0.15:
        z[:n // 3] = 0
    return T.ChanCapture(ch=z.astype(np.complex64), fs=CH_FS, t0_index=t0, f_mix=Fraction(0)), f0


def test_probe_paths_random(row):
    """_carrier_lines, refine_path, line_path and _flat_path on random
    captures (`_probe_chan`): segment lengths from 0.5 to 60 s (odd and
    even sample counts), spans, a median band, ratios, and paths with
    random weights. _carrier_lines to the bit; the paths to the same
    branch (the caller's own object back or not) and grid, f to 1e-7 Hz."""
    rng = np.random.default_rng(31)
    branches = set()
    for trial in range(16):
        spec = frame.TINY if trial % 3 else frame.get("short")
        chan, f0 = _probe_chan(rng, spec)
        z = chan.ch.astype(np.complex128)
        seg_s = float(rng.choice([20.0, 5.0, 3.3, 7.1, 60.0, 0.5]))
        span = float(rng.choice([2.0, 1.0, 0.3, 5.0]))
        kw = {}
        if trial % 3 == 1:
            kw["med_band"] = (float(rng.uniform(-40, 0)), float(rng.uniform(0, 40)))
        if trial % 4 == 2:
            kw["ratio"] = float(rng.uniform(1, 30))
        what = (trial, chan.t0_index, len(z), seg_s, span, kw)
        r = T._carrier_lines(z, chan, spec, seg_s, span, **kw)
        g = row["_carrier_lines"](z, chan, spec, seg_s, span, **kw)
        for a, b in zip(g, r):
            assert a.dtype == b.dtype and np.array_equal(a, b), what
        pt = np.array([-10.0, 0.0, 30.0, 60.0, 100.0, 200.0])
        path = FreqPath(t_s=pt, f_hz=1500.0 + f0 + rng.uniform(-0.3, 0.3, len(pt)),
                        weight=rng.uniform(0, 2, len(pt)))
        for name in ("refine_path", "line_path"):
            r, g = getattr(T, name)(chan, spec, path), row[name](chan, spec, path)
            assert (r is path) == (g is path), (what, name)
            branches.add((name, r is path))
            if r is not path:
                _path_close(g, r, (what, name))
        r, g = T._flat_path(path), row["_flat_path"](path)
        assert np.array_equal(g.t_s, r.t_s) and np.array_equal(g.weight, r.weight), what
        _le("flat_path f", np.max(np.abs(g.f_hz - r.f_hz)), 1e-9, what)
    assert branches == {(n, u) for n in ("refine_path", "line_path") for u in (True, False)}, branches


def test_probe_segment_timing_planted(row):
    """_segment_timing on statistics with known-symbol pulses planted along
    a line with a wobble, at noise levels from clean to buried and with half
    the frame attenuated: every branch of the reference's walk (segments
    under Z 4 skipped, one segment, the weighted line, the outlier refit,
    the fit refused past 300 ppm, nothing found)."""
    rng = np.random.default_rng(7)
    for trial in range(14):
        spec = frame.TINY if trial < 10 else frame.get("short")
        kn = T.known_symbols(spec)
        sps = spec.t_sym * CH_FS
        n = int(spec.n_pos * sps) + 4000
        t0, d = 1000 + rng.uniform(0, 50), rng.uniform(-150e-6, 150e-6)
        wob = rng.uniform(0, 3) * np.sin(2 * np.pi * kn.pos / spec.n_pos * rng.uniform(0.5, 3))
        x = rng.standard_normal(n) * rng.choice([0.3, 1.0, 3.0, 8.0])
        for c, a in zip(t0 + kn.pos * sps * (1 + d) + wob, kn.val):
            k = np.arange(int(c) - 8, int(c) + 9)
            x[k] += a * np.maximum(0, 1 - np.abs(k - c) / sps)
        if trial % 5 == 4:
            x[:n // 2] *= 0.01
        sig2 = float(rng.choice([0.5, 1.0, 2.0])) * float(np.median(x ** 2)) / 0.4549
        t0in, din = t0 + rng.uniform(-2, 2), d + rng.uniform(-3e-5, 3e-5)
        seg = float(rng.choice([20.0, 5.0, 11.3, 60.0, 2.0]))
        r = T._segment_timing(x, sig2, kn, t0in, din, sps, spec.n_pos, seg)
        g = row["_segment_timing"](x, sig2, kn, t0in, din, sps, spec.n_pos, seg)
        what = (trial, seg)
        _le("segment t0", abs(g[0] - r[0]), 1e-6, what)
        _le("segment slip", abs((g[1] - r[1]) * spec.n_pos * sps), 1e-6, what)
        _le("segment score", abs(g[2] - r[2]) / max(abs(r[2]), 1e-4), 1e-8, what)


def test_probe_gate_edges(row, scen):
    """Gate V where the receiver can take it: every known symbol erased (a
    pass joined late: Z_ref 0, refused), and an A2-like candidate whose
    timing is 7 samples off with a 2x2 covariance from a 30-sample tau0 sd
    (the local search then spans 120 samples either side: the reference's
    FFT path)."""
    s = scen["tiny-6"]
    gone = [(-1e3, 1e4)]
    r = T.verify_track(s.chan, s.spec, s.v, fit=False, erase_s=gone)
    g = row["verify_track"](s.chan, s.spec, s.v, fit=False, erase_s=gone)
    assert r.z_ref == g.z_ref == 0.0 and not np.any(g.psi) and not np.any(r.psi)
    assert T.verify(s.chan, CH_FS, s.spec, s.v, fit=False, erase_s=gone) is None
    assert row["verify"](s.chan, CH_FS, s.spec, s.v, fit=False, erase_s=gone) is None
    tm = Timing(tau0=s.v.timing.tau0 + 7.0, ppm=0.0, gamma=0.0, z=9.0, cov=np.diag([30.0 ** 2, 0.0]))
    cand = dataclasses.replace(s.det, path=s.v.path, timing=tm, method="preamble")
    r, g = T.verify_track(s.chan, s.spec, cand), row["verify_track"](s.chan, s.spec, cand)
    _track_close(g, r, "A2-like")
    _timing_close(g.timing, r.timing, s.spec, "A2-like")
    _det_close(row["verify"](s.chan, CH_FS, s.spec, cand), T.verify(s.chan, CH_FS, s.spec, cand), s.spec,
               "A2-like")


def test_probe_steady_weak_q_floor(row):
    """A weak steady pass (TINY at -15 dB, no fading) on which the tracker's
    q sits at its grid's floor, as it does for a quiet unfaded QRSS signal.
    There the reference does not reproduce itself to the module's signal
    bars: nudging its own timing by 1e-13 samples moves its Z_ref by 7.8e-7
    and P by 8.4e-5 (measured, recorded below). The C++ on the reference's
    own candidate (identical m, bit for bit) lands inside that: Z_ref 2.9e-7,
    P 4.6e-5. So: the same accept, path and timing, and Z_ref to 1e-5, u to
    1e-7, P to 1e-3 (the noise capture's bars)."""
    spec = frame.TINY
    a = unit_rms_latents(spec.n_data, 11)
    sym = frame.assemble(spec, None, precode(a, Q_TEST))
    sim = chm.simulate(sym, spec, Q_TEST, ChannelConfig(snr_db=-15.0, seed=11), carrier_hz=1500.0)
    prep = RX.prepare(frontend.Capture(sim.fe, Q_TEST, sim.t0_index))
    det = flat_det(1500.0)
    chan = RX.channel_for(prep, spec, det)
    chan.noise_psd_at(0.0)
    r, g = T.verify(chan, CH_FS, spec, det), row["verify"](chan, CH_FS, spec, det)
    assert r is not None and g is not None
    _timing_close(g.timing, r.timing, spec, "q floor")
    _path_close(g.path, r.path, "q floor")
    _le("q floor verify z_ref", _r(g.z_ref, r.z_ref), 1e-5)
    rt, gt = T.verify_track(chan, spec, r, fit=False), row["verify_track"](chan, spec, r, fit=False)
    assert rt.q == gt.q == T.Q_GRID[0], (rt.q, gt.q)
    for f in ("z_d", "m", "N", "psi"):
        assert np.array_equal(getattr(gt, f), getattr(rt, f)), f
    _le("q floor track z_ref", _r(gt.z_ref, rt.z_ref), 1e-5)
    _le("q floor track u", rel(gt.u, rt.u), 1e-7)
    _le("q floor track P", rel(gt.P, rt.P), 1e-3)
    nudged = dataclasses.replace(r, timing=dataclasses.replace(r.timing, tau0=r.timing.tau0 + 1e-13))
    rn = T.verify_track(chan, spec, nudged, fit=False)
    _le("q floor reference's own z_ref (1e-13 nudge)", _r(rn.z_ref, rt.z_ref), 1e-4)
    _le("q floor reference's own P (1e-13 nudge)", rel(rn.P, rt.P), 1e-2)


# --- the whole caller ------------------------------------------------------------------------------

def _patched(native, monkeypatch):
    """Every row of both track rows files over track's names and ltrack's
    from-imports of them; {name: calls} counts what was reached."""
    from qrss_native.track import rows as core_rows
    from qrss_native.track_timing import rows as timing_rows

    core, timing = core_rows(native.qrss), timing_rows(native.qrss)
    assert not set(core) & set(timing)
    table = {**core, **timing}
    assert all(target == TRACK for target, _ in table)
    used = {}

    def counted(name, fn):
        def f(*a, **k):
            used[name] = used.get(name, 0) + 1
            return fn(*a, **k)
        return f

    wrapped = {name: counted(name, fn) for (_t, name), fn in table.items()}
    for name in FROM_IMPORTS:                    # ltrack's own bindings are track's functions
        assert getattr(LT, name) is getattr(T, name), name
    for name, fn in wrapped.items():
        monkeypatch.setattr(T, name, fn)
    for name in FROM_IMPORTS:
        monkeypatch.setattr(LT, name, wrapped[name])
    return used


def _pass_same(g, r, what):
    assert g.uid.rsplit("_", 1)[0] == r.uid.rsplit("_", 1)[0], what
    assert (g.q, g.frame, g.waveform, g.estimator) == (r.q, r.frame, r.waveform, r.estimator), what
    assert g.header == r.header and g.cw == r.cw, what
    assert abs(g.f_hz - r.f_hz) <= 1e-6, what
    for f, tol in (("z", 1e-6), ("w", 1e-6), ("hdr_llr", 1e-4), ("psi", 1e-6)):
        a, b = np.asarray(getattr(g, f)), np.asarray(getattr(r, f))
        assert a.shape == b.shape and a.dtype == b.dtype, (what, f)
        assert a.size == 0 or np.max(np.abs(a.astype(np.float64) - b)) <= tol, (what, f)
    assert abs(g.timing.tau0 - r.timing.tau0) <= 1e-6 and abs(g.timing.ppm - r.timing.ppm) <= 1e-6, what
    for f in ("offset_hz", "doppler_hz", "snr2500_db", "z_ref", "kappa"):
        assert abs(getattr(g.report, f) - getattr(r.report, f)) <= \
            1e-6 * max(abs(getattr(r.report, f)), 1.0), (what, f)
    assert (g.report.slip_free, g.report.suspect) == (r.report.slip_free, r.report.suspect), what


def test_receive_pass_on_the_cpp(native, Tt, scen, monkeypatch):
    """receiver.receive_pass on CE TINY (-12 dB) and SHORT (-6 dB) with every
    row of both files: gate V, the tracker and everything under them in
    C++. The same passes and headers, f_hz to 1e-6 Hz, latents to 1e-6.

    And the two other callers of gate V's pieces, on the same captures:
    receiver._gate on TINY (verify, then the verify_track row's track for
    _ce_resid_ratio, and the lead-in of a non-preamble candidate) and
    em.pass_path on both passes received (line_path, refine_path and the
    verify_track row with fit=False, on a stored pass's CH-rate capture):
    the same detections and paths."""
    from sstvae.qrss import em

    refs, gates, paths = {}, {}, {}
    for label in ("tiny-12", "short-6"):
        s = scen[label]
        refs[label] = RX.receive_pass(s.prep, s.spec, s.det, return_tracks=True)
        assert refs[label] is not None, label
        if label == "tiny-12":
            gates[label] = RX._gate(s.prep, s.spec, s.det, (), ())
            assert gates[label] is not None, label
        paths[label] = em.pass_path(refs[label][0])
    used = _patched(native, monkeypatch)
    for label, ref in refs.items():
        s = scen[label]
        got = RX.receive_pass(s.prep, s.spec, s.det, return_tracks=True)
        assert got is not None, label
        _pass_same(got[0], ref[0], label)
        assert len(got[1]) == len(ref[1]), label
    # gate V's own tracks run inside the C++ verify; rounds A and B through the row
    assert used.get("verify", 0) == 2, used
    assert used.get("track", 0) == sum(len(r[1]) for r in refs.values()), used
    for label in refs:
        s = scen[label]
        if label in gates:
            _det_close(RX._gate(s.prep, s.spec, s.det, (), ()), gates[label], s.spec, (label, "_gate"))
        g = em.pass_path(refs[label][0])
        _path_close(g, paths[label], (label, "pass_path"))
    assert used.get("verify", 0) == 2 + len(gates), used
    assert used.get("verify_track", 0) == len(gates) + 2 * len(refs), used
    assert used.get("line_path", 0) == used.get("refine_path", 0) == len(refs), used


def test_l_receive_pass_on_the_cpp(native, Tt, monkeypatch):
    """An L TINY pass at -12 dB: ltrack's gate V reaches refine_path,
    _flat_path and _parabola (its from-imports) through the rows."""
    from qrss_fakes import l_scenario

    spec = lframe.get("l-tiny")
    _, sim = l_scenario("l-tiny", -12.0, 1)
    prep = RX.prepare(frontend.Capture(sim.fe, Q_TEST, sim.t0_index))
    det = flat_det(1500.0, WAVEFORM_L)
    ref = RX.receive_pass(prep, spec, det, return_tracks=True)
    assert ref is not None
    used = _patched(native, monkeypatch)
    got = RX.receive_pass(prep, spec, det, return_tracks=True)
    assert got is not None
    assert {"refine_path", "_flat_path", "_parabola"} <= set(used), sorted(used)
    _pass_same(got[0], ref[0], "l-tiny")
    assert len(got[1]) == len(ref[1])

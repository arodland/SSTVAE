"""Parity of native/core/qrss/ltrack_timing (sstvae_native.qrss.ltrack_timing)
with the rest of sstvae/qrss/ltrack.py: the receiver-time mix's phases
(_lag_phase, _clock_phase), L's coarse timing fit (fit_timing_l,
_segment_timing_l), the pilot's path measured on the frame
(refine_path_l), the residual check (resid_ratio) and gate V
(verify_track_l, verify_l). tests/qrss_native/ltrack_timing.py
substitutes them under --native-qrss=rx.

The inputs are the receiver's own: simulated L passes (qrss_fakes'
l_scenario) through `receiver.prepare` / `channel_for` at a flat 1500 Hz
candidate (what `receive_pass` gates), l-tiny and l-short at -6 and -12
dB, a 1 Hz/min drift, +100 ppm and -100 ppm clocks, a weak steady pass
whose tracker sits at the bottom of its q grid (l-tiny -18 dB) and a
capture with no signal in it (-80 dB) that gate V must refuse. Spies on
the reference record what its fit_timing_l computed on the way (its
grids, through `np.full`, and every `_segment_timing_l` call).

The contract (the bar), and in brackets the worst measured on these
scenarios (2026-10-13; QRSS_PARITY_WORST=1 with -s prints them):

- _lag_phase, _clock_phase: to the bit, shapes, signed zeros and numpy
  scalars alike;
- fit_timing_l: the same argmax cell and runners-up, every peak's
  _segment_timing_l input to 1e-9 samples [4.6e-13 in t0, 1.7e-12 of
  slip]; the same timings in the same order, tau0 to 1e-6 CH samples
  [6.4e-12], ppm to 1e-6 samples of slip [4.3e-11], Timing.z to 1e-8
  relative [5.9e-13]. The C++ correlates by the direct gather where the
  reference's blind search transforms: its grid is the reference's FFT
  grid to 1e-12 of the grid's scale [7.2e-14] and the reference's direct
  path (a local search) to 1e-13 [2.0e-15]; the statistic x and sig2 to
  1e-12 [1.1e-15, 2.6e-15] (scipy's FFT convolutions against the C++'s);
- _segment_timing_l on the reference's own inputs: t0 and the slip to
  1e-6 samples [1.4e-11, 1.1e-10], the score to 1e-8 relative [1.9e-13];
- refine_path_l: the same branch, the same grid, f to 1e-7 Hz [6.8e-13],
  the caller's own path back when nothing was measured;
- resid_ratio: to the bit on the same track;
- verify_l: the same accept/reject in every scenario (noise refused after
  the runners-up), the same path, the timing as above, Z_ref to 1e-8
  relative [1.6e-12];
- verify_track_l: Z_ref and R to 1e-8 relative [1.1e-11, 3.0e-12]; every
  array to the L tracker's own bar, 1e-9 [1.6e-10: z_d, u and m of the
  weak pass]. The weak pass needs no wider bar: a 1e-13-sample nudge of
  the reference's own timing moves its Z_ref and P by ~1e-13 there
  (test_probe_q_floor);
- the composition is exact: the reference's verify_track_l and verify_l
  with the C++ primitives assigned over the ones they call are the C++
  verify_track_l and verify_l bit for bit, every scenario
  (test_gate_v_composition), so the tolerances above are the primitives';
- the whole caller, every row of tests/qrss_native/{trackbase,track,
  track_timing,ltrack,ltrack_timing}.py over its target and every
  from-import site of it, counting spies installed before the rows: receiver.receive_pass
  and receiver._gate on L passes, and the joint receiver
  (receive_slot) on a mixed capture, L with a CE neighbour 60 Hz above:
  the same passes and headers, f_hz to 1e-6 Hz, latents to 1e-6, every
  row of this file reached from the caller in one configuration or
  another (the reference's verify_l, verify_track_l and fit_timing_l
  kept in turn so that what they call is the C++);
- the reference's refusals, type and text: a CE frame, a malformed erase
  span.

The review probes (2026-10-13) go where the scenarios do not: random
fit_grid_l inputs against the reference's direct path [1.3e-15], a capture
starting inside the blind search (the straddle at x's start, which the
receiver's padding rules out: test_receiver_channel_holds_every_lag),
random _segment_timing_l cuts [7.3e-12 in t0, 4.3e-11 of slip], a CE
signal heard under an L hypothesis (the resid_ratio cut), erasures, four
threads, the preamble detection's 2x2 cov, and a candidate whose timing
holds NaN: verify_track_l(fit=False) must hand back the candidate's own
Timing object, as the reference does, which the binding's float `==`
missed for NaN until this review (it now compares bits).
"""

import dataclasses
import math
import os
from types import SimpleNamespace

import numpy as np
import pytest
from qrss_demod_tracks import flat_det, rel
from qrss_fakes import l_scenario
from qrss_helpers import Q_TEST
from qrss_native import _rx, require

from sstvae.qrss import frame, frontend, lframe
from sstvae.qrss import ltrack as LT
from sstvae.qrss import receiver as RX
from sstvae.qrss import track as T
from sstvae.qrss.channel import Neighbour
from sstvae.qrss.constants import CH_FS, L_T_SYM, WAVEFORM_CE, WAVEFORM_L

TRACK = "sstvae.qrss.track"
LTRACK = "sstvae.qrss.ltrack"

# label: (frame, SNR dB, seed, further ChannelConfig fields)
SCENARIOS = {
    "tiny-6": ("l-tiny", -6.0, 1, ()),
    "tiny-12": ("l-tiny", -12.0, 2, ()),
    "short-6": ("l-short", -6.0, 1, ()),
    "short-12": ("l-short", -12.0, 3, ()),
    "drift": ("l-tiny", -6.0, 4, (("drift_hz_per_min", 1.0),)),
    "ppm+100": ("l-tiny", -9.0, 5, (("tx_ppm", 100.0),)),
    "ppm-100": ("l-tiny", -9.0, 6, (("rx_ppm", -100.0),)),
    "qfloor": ("l-tiny", -18.0, 11, ()),
    "noise": ("l-tiny", -80.0, 7, ()),
}
SPS = L_T_SYM * CH_FS
ARRAYS = ("z_d", "m", "c", "N", "psi", "u", "P", "pu", "m_pl", "c_pl", "N_pl", "psi_pl", "u_pl", "P_pl")


@pytest.fixture(scope="module")
def Lt(native):
    return require(native, "ltrack_timing")


@pytest.fixture(scope="module")
def row(native, Lt):
    from qrss_native.ltrack_timing import rows

    return {name: fn for (_t, name), fn in rows(native.qrss).items()}


@pytest.fixture(scope="module")
def scen(Lt):
    """{label: scenario}: the capture prepared, the flat candidate's channel
    and the reference's gate V on it (with the fit flag of every
    verify_track_l it ran)."""
    out = {}
    for label, (name, snr, seed, cfg) in SCENARIOS.items():
        spec = lframe.get(name)
        _, sim = l_scenario(name, snr, seed, cfg=cfg)
        prep = RX.prepare(frontend.Capture(sim.fe, Q_TEST, sim.t0_index))
        det = flat_det(1500.0, WAVEFORM_L)
        chan = RX.channel_for(prep, spec, det)
        chan.noise_psd_at(0.0)
        tracked = []
        orig = LT.verify_track_l

        def spy(*a, _orig=orig, _calls=tracked, **k):
            _calls.append(k.get("fit", True))
            return _orig(*a, **k)

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(LT, "verify_track_l", spy)
            v = LT.verify_l(chan, CH_FS, spec, det)
        out[label] = SimpleNamespace(spec=spec, sim=sim, prep=prep, det=det, chan=chan, v=v,
                                     tracked=tracked)
    assert out["noise"].v is None
    assert all(s.v is not None for k, s in out.items() if k != "noise"), \
        [k for k, s in out.items() if s.v is None]
    return out


class _NpSpy:
    """numpy, recording the 2-D -inf grids fit_timing_l allocates (and then
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
    """The reference's fit_timing_l, with its grids and _segment_timing_l calls."""
    grids, segs = [], []
    orig = LT._segment_timing_l

    def spy(x, sig2, kn, t0abs, delta, sps, n_pos, n_ref, *a, **k):
        r = orig(x, sig2, kn, t0abs, delta, sps, n_pos, n_ref, *a, **k)
        segs.append(SimpleNamespace(x=x, sig2=sig2, kn=kn, t0abs=t0abs, delta=delta, sps=sps,
                                    n_pos=n_pos, n_ref=n_ref, out=r))
        return r

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(LT, "np", _NpSpy(grids))
        mp.setattr(LT, "_segment_timing_l", spy)
        out = LT.fit_timing_l(*args, **kw)
    return out, grids, segs


def _cells(Z, n_best, sps):
    """The reference's peak cells on its own grid (fit_timing_l's loop)."""
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
            print(f"{k:34s} {_WORST[k]:.3g}")


def _le(key, value, bar, what=""):
    """value <= bar, the worst value per key kept for the record."""
    value = float(value)
    _WORST[key] = max(_WORST.get(key, 0.0), value)
    assert value <= bar, (key, what, value, bar)


def _r(g, r, floor=1e-300):
    return abs(g - r) / max(abs(r), floor)


def _slip(d, spec):
    """A clock error as samples of slip over the frame."""
    return abs(d * spec.n_pos * SPS)


def _timing_close(g, r, spec, what, z_tol=1e-8):
    _le("timing tau0", abs(g.tau0 - r.tau0), 1e-6, what)
    _le("timing slip", _slip((g.ppm - r.ppm) * 1e-6, spec), 1e-6, what)
    assert g.gamma == r.gamma, what
    _le("timing z", abs(g.z - r.z) / max(abs(r.z), 1.0), z_tol, what)
    assert g.cov.shape == r.cov.shape and np.allclose(g.cov, r.cov, rtol=1e-12, atol=0.0), what


def _path_close(g, r, what, tol=1e-7):
    assert g.t_s.shape == r.t_s.shape, (what, g.t_s.shape, r.t_s.shape)
    assert np.array_equal(g.t_s, r.t_s), what
    _le("path f_hz", np.max(np.abs(g.f_hz - r.f_hz)), tol, what)
    assert np.array_equal(g.weight, r.weight), what


def _same(a, b, what=""):
    """Bit for bit (NaN where NaN), dtype and shape alike."""
    if a is None or b is None:
        assert a is None and b is None, what
        return
    a, b = np.asarray(a), np.asarray(b)
    assert a.dtype == b.dtype and a.shape == b.shape, (what, a.dtype, b.dtype, a.shape, b.shape)
    if a.dtype.kind in "fc":
        a, b = np.atleast_1d(a), np.atleast_1d(b)
        av = a.view(a.real.dtype) if a.dtype.kind == "c" else a
        bv = b.view(b.real.dtype) if b.dtype.kind == "c" else b
        an, bn = np.isnan(av), np.isnan(bv)
        assert np.array_equal(an, bn), what
        assert av[~an].tobytes() == bv[~bn].tobytes(), what
    else:
        assert np.array_equal(a, b), what


def _outcome(fn, *args, **kw):
    try:
        out = fn(*args, **kw)
    except (ValueError, TypeError, IndexError, ZeroDivisionError, AssertionError, AttributeError) as e:
        return type(e), str(e)
    return "ok", out


# --- the mix's phases -------------------------------------------------------------------------------

def test_mix_phases(row):
    """_lag_phase and _clock_phase to the bit: np.exp of a purely imaginary
    argument is (cos, sin) through libm, the turns reduced first; any
    shape, a 0-d argument a numpy scalar back."""
    rng = np.random.default_rng(3)
    lags = [rng.uniform(-1e6, 1e6, 500), np.arange(-300, 300) - 0.37, rng.uniform(-5, 5, (3, 7)),
            np.array([0.0, -0.0, 1e300, -1e300, np.inf, -np.inf, np.nan, 5e-324]), 12.5, -3,
            np.float64(2.0 ** 52 + 1)]
    for lag in lags:
        r, g = LT._lag_phase(lag), row["_lag_phase"](lag)
        assert type(g) is type(r)
        _same(g, r, ("lag", lag if np.ndim(lag) == 0 else np.shape(lag)))
    pos = [np.arange(30000, dtype=np.int64), rng.uniform(0, 3e4, (4, 5)), 17, np.array([np.nan, np.inf])]
    for p in pos:
        for d in (0.0, 2.5e-4, -1.7e-4, np.float64(3.3e-6), 1.0, np.nan):
            r, g = LT._clock_phase(p, d), row["_clock_phase"](p, d)
            assert type(g) is type(r)
            _same(g, r, ("clock", np.shape(p), d))


# --- the coarse timing fit ----------------------------------------------------------------------

@pytest.fixture(scope="module")
def blind(Lt, scen):
    """{label: (path fn, z_d, reference (out, grids, segs), C++ (out, trace))}: the
    blind search with n_best = V_TIMING_TRIES, on gate V's path (the flat one
    where gate V refused)."""
    out = {}
    for label, s in scen.items():
        fn = T.freq_path_spline((s.v or s.det).path)
        z_d = T.derotate(s.chan, fn)
        kw = dict(t0_index=s.chan.t0_index, z_d=z_d, n_best=T.V_TIMING_TRIES)
        ref = _ref_fit(s.chan, CH_FS, fn, s.spec, **kw)
        got = Lt._fit_timing_l_trace(s.chan, CH_FS, fn, s.spec, **kw)
        out[label] = SimpleNamespace(fn=fn, z_d=z_d, ref=ref, got=got)
    return out


def _peaks_close(tr, segs, spec, what):
    peaks = tr["peaks"]
    assert len(peaks) == len(segs), (what, len(peaks), len(segs))
    for k, (p, s) in enumerate(zip(peaks, segs)):
        w = f"{what} peak {k}"
        _le("peak t0abs_in", abs(p["t0abs_in"] - s.t0abs), 1e-9, w)
        _le("peak delta_in slip", _slip(p["delta_in"] - s.delta, spec), 1e-9, w)
        t0, d, score = s.out
        _le("peak t0abs", abs(p["t0abs"] - t0), 1e-6, w)
        _le("peak delta slip", _slip(p["delta"] - d, spec), 1e-6, w)
        _le("peak score", abs(p["score"] - score) / max(abs(score), 1e-4), 1e-8, w)
        _le("sig2", _r(tr["sig2"][p["lp"]], s.sig2), 1e-12, w)
        assert s.n_ref == tr["base"] + tr["frac0"], w


def _norm(sig2, kn):
    return math.sqrt(sig2 / 2 * float(np.sum(np.abs(np.asarray(kn.val, dtype=np.complex128)) ** 2)))


def test_fit_timing_l_blind(Lt, row, scen, blind):
    for label, b in blind.items():
        spec = scen[label].spec
        (r_out, grids, segs), (g_out, tr) = b.ref, b.got
        assert len(grids) == len(LT.CARRIER_LP_HZ_L) == len(tr["sig2"]), label
        _peaks_close(tr, segs, spec, label)
        lps = [p["lp"] for p in tr["peaks"]]
        assert lps == sorted(lps), label
        # the grid, the statistic and the cells, per carrier estimate
        kn = LT.known_symbols_l(spec)
        deltas = np.arange(-tr["nd"], tr["nd"] + 1) * tr["step"]
        n_ref = tr["base"] + tr["frac0"]
        for lp_i, (lp, Z) in enumerate(zip(LT.CARRIER_LP_HZ_L, grids)):
            s = next(s for p, s in zip(tr["peaks"], segs) if p["lp"] == lp_i)
            x = Lt._fit_statistic_l(b.z_d, n_ref, lp)
            assert x.shape == s.x.shape and x.dtype == s.x.dtype
            _le("statistic x", np.max(np.abs(x - s.x)) / np.max(np.abs(s.x)), 1e-12, (label, lp))
            assert Z.shape == (len(deltas), tr["nlag"]), label
            G = Lt._fit_grid_l(s.x, kn, tr["base"], tr["nlag"], tr["frac0"], deltas, SPS, _norm(s.sig2, kn))
            assert G.shape == Z.shape
            _le("grid vs FFT", np.max(np.abs(G - Z)) / np.max(np.abs(Z)), 1e-12, (label, lp))
            mine = [(p["i"], p["j"]) for p in tr["peaks"] if p["lp"] == lp_i]
            assert mine == _cells(Z, T.V_TIMING_TRIES, SPS), (label, lp)
        # the timings returned
        assert len(g_out) == len(r_out), label
        for k, (g, r) in enumerate(zip(g_out, r_out)):
            _timing_close(_rx.to_ref(g), r, spec, f"{label} timing {k}")
        # through the row (n_best 1: one Timing, not a list)
        c = scen[label].chan
        r1 = LT.fit_timing_l(c, CH_FS, b.fn, spec, t0_index=c.t0_index, z_d=b.z_d)
        g1 = row["fit_timing_l"](c, CH_FS, b.fn, spec, t0_index=c.t0_index, z_d=b.z_d)
        _timing_close(g1, r1, spec, f"{label} n_best 1")


def test_fit_timing_l_local(Lt, row, scen, blind):
    """tau0_init: the reference's direct path (nlag <= 64), grids compared;
    a known-symbol set given as tmpl_fn (complex, and real), and a tmpl_fn
    that is not one (ignored)."""
    for label in ("tiny-12", "short-6", "ppm+100", "qfloor", "noise"):
        s, b = scen[label], blind[label]
        spec = s.spec
        best = b.ref[0][0]
        kn_l = LT.known_symbols_l(spec)
        kn_real = T._Known(pos=kn_l.pos[::3], val=np.sign(kn_l.val.real[::3]) + 0.0)
        for kw in (dict(tau0_init=best.tau0 + 0.7),
                   dict(tau0_init=best.tau0 - 1.3, tau_range=10.0, n_best=3),
                   dict(tau0_init=best.tau0, tau_range=0.4, ppm_range=60.0),
                   dict(tau0_init=best.tau0 + 2.0, tmpl_fn=kn_l),
                   dict(tau0_init=best.tau0 + 0.2, tmpl_fn=kn_real),
                   dict(tau0_init=best.tau0, tmpl_fn=lambda *a: None)):
            kw = dict(kw, t0_index=s.chan.t0_index, z_d=b.z_d)
            ref, grids, segs = _ref_fit(s.chan, CH_FS, b.fn, spec, **kw)
            _, tr = Lt._fit_timing_l_trace(s.chan, CH_FS, b.fn, spec, **kw)
            what = (label, {k: v for k, v in kw.items() if k != "z_d"})
            assert tr["nlag"] <= 64, what
            _peaks_close(tr, segs, spec, what)
            kn = kw["tmpl_fn"] if isinstance(kw.get("tmpl_fn"), T._Known) else kn_l
            deltas = np.arange(-tr["nd"], tr["nd"] + 1) * tr["step"]
            for lp_i, Z in enumerate(grids):
                sg = next(sg for p, sg in zip(tr["peaks"], segs) if p["lp"] == lp_i)
                G = Lt._fit_grid_l(sg.x, kn, tr["base"], tr["nlag"], tr["frac0"], deltas, SPS, _norm(sg.sig2, kn))
                _le("grid vs direct", np.max(np.abs(G - Z)) / np.max(np.abs(Z)), 1e-13, what)
            got = row["fit_timing_l"](s.chan, CH_FS, b.fn, spec, **kw)
            if kw.get("n_best", 1) <= 1:
                ref, got = [ref], [got]
            assert len(got) == len(ref), what
            for gg, rr in zip(got, ref):
                _timing_close(gg, rr, spec, what)


def test_fit_timing_l_without_z_d(row, scen):
    """z_d None: derotated from the path (a FreqPath or a callable)."""
    s = scen["tiny-6"]
    for path in (s.v.path, T.freq_path_spline(s.v.path)):
        r = LT.fit_timing_l(s.chan, CH_FS, path, s.spec, None, s.v.timing.tau0 + 0.5, t0_index=s.chan.t0_index)
        g = row["fit_timing_l"](s.chan, CH_FS, path, s.spec, None, s.v.timing.tau0 + 0.5,
                                t0_index=s.chan.t0_index)
        _timing_close(g, r, s.spec, "z_d None")


def test_fit_grid_l_edges(Lt):
    """Terms whose interpolation pair leaves x (finite x): the reference's
    direct path, worked here in numpy as the reference writes it."""
    rng = np.random.default_rng(5)
    kn = LT.known_symbols_l(lframe.L_TINY)
    span = int(kn.pos.max() * SPS) + 1
    for nx, base, nlag in ((span + 40, -30, 50), (span + 300, 12, 200), (span - 100, -5, 90), (span, -40, 7)):
        x = rng.standard_normal(nx) + 1j * rng.standard_normal(nx)
        frac0 = 0.37
        deltas = np.arange(-3, 4) * 2.1e-5
        norm = 3.5
        lagph = LT._lag_phase(np.arange(nlag))
        ref = np.full((len(deltas), nlag), -np.inf)
        lags = np.arange(nlag)
        pos = kn.pos.astype(np.float64)
        for i, d in enumerate(deltas):
            o = pos * SPS * (1 + d) + frac0
            idx = base + o[:, None] + lags[None, :]
            i0 = np.floor(idx).astype(np.int64)
            f = idx - i0
            ok = (i0 >= 0) & (i0 + 1 < len(x))
            i0c = np.clip(i0, 0, len(x) - 2)
            v = (x[i0c] * (1 - f) + x[i0c + 1] * f) * ok
            g = np.conj(kn.val) * LT._clock_phase(pos, d)
            ref[i] = (lagph * (g @ v)).real / norm
        got = Lt._fit_grid_l(x, kn, base, nlag, frac0, deltas, SPS, norm)
        assert got.shape == ref.shape and np.all(np.isfinite(got))
        _le("grid edges", np.max(np.abs(got - ref)) / np.max(np.abs(ref)), 1e-13, (nx, base))


def test_segment_timing_l(row, scen, blind):
    for label, b in blind.items():
        spec = scen[label].spec
        s = b.ref[2][0]
        for t0abs, delta, seg in ((s.t0abs, s.delta, LT.SEG_TIMING_S), (s.t0abs, s.delta, 7.0),
                                  (s.t0abs + 1.5, s.delta, 40.0), (s.t0abs, s.delta + 5e-5, LT.SEG_TIMING_S),
                                  (s.t0abs - 0.4, s.delta + 1e-3, LT.SEG_TIMING_S),
                                  (s.t0abs, s.delta, 1000.0)):
            r = LT._segment_timing_l(s.x, s.sig2, s.kn, t0abs, delta, SPS, spec.n_pos, s.n_ref, seg)
            g = row["_segment_timing_l"](s.x, s.sig2, s.kn, t0abs, delta, SPS, spec.n_pos, s.n_ref, seg)
            what = (label, t0abs - s.t0abs, delta - s.delta, seg)
            _le("segment t0", abs(g[0] - r[0]), 1e-6, what)
            _le("segment slip", _slip(g[1] - r[1], spec), 1e-6, what)
            _le("segment score", abs(g[2] - r[2]) / max(abs(r[2]), 1e-4), 1e-8, what)


# --- the path, gate V ------------------------------------------------------------------------------

def test_refine_path_l(row, scen):
    branches = set()
    for label, s in scen.items():
        bases = [s.det.path, T._flat_path(s.det.path)] + ([s.v.path] if s.v is not None else [])
        for k, base in enumerate(bases):
            r = LT.refine_path_l(s.chan, s.spec, base)
            g = row["refine_path_l"](s.chan, s.spec, base)
            assert (g is base) == (r is base), (label, k)
            if r is base:
                branches.add("unchanged")
                continue
            branches.add("fine" if np.allclose(np.diff(r.t_s), 1.0) else "coarse")
            _path_close(g, r, (label, k))
    assert {"unchanged", "coarse"} <= branches, branches
    # a capture cut short of two path segments: the path back
    s = scen["short-6"]
    cut = dataclasses.replace(s.chan, ch=s.chan.ch[:int(s.chan.t0_index) + 15 * CH_FS])
    r, g = LT.refine_path_l(cut, s.spec, s.det.path), row["refine_path_l"](cut, s.spec, s.det.path)
    assert r is s.det.path and g is s.det.path


def _templates(tr):
    c_known, _, gain, dist, _ = LT.data_templates(tr.spec, tr.pos, tr.classes)
    return c_known, gain, dist


def test_resid_ratio(row, scen):
    """R to the bit on the reference's own gate-V tracks; with every known
    symbol erased, and with the measurements scrambled (psi under PSI_MIN,
    N and P zero somewhere, so that V <= 0 drops a symbol)."""
    rng = np.random.default_rng(8)
    seen = set()
    for label, s in scen.items():
        cand = s.v if s.v is not None else dataclasses.replace(s.det, timing=None)
        tr = LT.verify_track_l(s.chan, s.spec, cand, fit=s.v is None)
        c, gain, dist = _templates(tr)
        r, g = LT.resid_ratio(tr, c, gain, dist), row["resid_ratio"](tr, c, gain, dist)
        assert type(g) is float and g == r, (label, g, r)
        assert r == tr.resid_ratio, label
        seen.add(r == 1.0)
        e = dataclasses.replace(tr, psi=np.zeros_like(tr.psi))
        assert LT.resid_ratio(e, c, gain, dist) == row["resid_ratio"](e, c, gain, dist) == 1.0
        e = dataclasses.replace(tr, psi=rng.uniform(0.0, 1.0, tr.psi.shape),
                                N=np.where(rng.random(tr.N.shape) < 0.3, 0.0, tr.N * rng.uniform(0.5, 2.0)),
                                P=np.where(rng.random(tr.P.shape) < 0.3, 0.0, tr.P))
        r, g = LT.resid_ratio(e, c, gain, dist), row["resid_ratio"](e, c, gain, dist)
        assert g == r, (label, "scrambled", g, r)
    assert False in seen


def _det_close(g, r, spec, what, z_tol=1e-8):
    assert (g is None) == (r is None), (what, g, r)
    if r is None:
        return
    assert (g.f_hz, g.method, g.lead_in_s, g.waveform) == (r.f_hz, r.method, r.lead_in_s, r.waveform), what
    assert g.waveform == WAVEFORM_L
    _le("verify z_ref", _r(g.z_ref, r.z_ref), z_tol, what)
    _timing_close(g.timing, r.timing, spec, what)
    _path_close(g.path, r.path, what)


def test_verify_l(row, scen):
    """Blind on the flat candidate (every scenario, the reference's own
    gate V from the fixture), then with the timing known, a narrower
    search and spans erased."""
    for label, s in scen.items():
        _det_close(row["verify_l"](s.chan, CH_FS, s.spec, s.det), s.v, s.spec, label)
    for label, kw in (("tiny-12", dict(fit=False)), ("tiny-12", dict(tau_range=6.0)),
                      ("tiny-12", dict(erase_s=[(10.0, 20.0)])), ("short-6", dict(fit=False)),
                      ("drift", dict(fit=False, erase_s=((5.0, 6.0), (30.0, 31.0))))):
        s = scen[label]
        cand = dataclasses.replace(s.det, timing=s.v.timing)
        r = LT.verify_l(s.chan, CH_FS, s.spec, cand, **kw)
        g = row["verify_l"](s.chan, CH_FS, s.spec, cand, **kw)
        _det_close(g, r, s.spec, (label, kw))


def test_verify_l_noise_tries_the_runners_up(row, scen):
    """The noise capture is refused only after every base and the coarse
    statistic's runners-up were tracked (the reference's calls, counted in
    the fixture); a signal is accepted on the first."""
    calls = scen["noise"].tracked
    assert len(calls) > 3 and calls.count(False) == len(calls) - 3, calls
    assert scen["tiny-6"].tracked == [True]
    s = scen["noise"]
    assert row["verify_l"](s.chan, CH_FS, s.spec, s.det) is None


def _track_close(g, r, what):
    """A C++ gate-V track against the reference's: every decision alike,
    Z_ref and R to 1e-8 relative, the arrays to the L tracker's own bar
    (tests/test_qrss_native_ltrack's 1e-9)."""
    assert (g.q, g.lead_in_s, g.ook) == (r.q, r.lead_in_s, r.ook), what
    assert np.array_equal(g.pos, r.pos), what
    _le("track z_ref", abs(g.z_ref - r.z_ref) / max(abs(r.z_ref), 1.0), 1e-8, what)
    _le("track resid_ratio", _r(g.resid_ratio, r.resid_ratio), 1e-8, what)
    for f in ARRAYS:
        a, b = getattr(g, f), getattr(r, f)
        assert (a is None) == (b is None), (what, f)
        if b is not None:
            _le(f"track {f}", rel(a, b), 1e-9, (what, f))
    _le("track delta", abs(g.delta - r.delta) / abs(r.delta), 1e-9, what)
    assert g.chan is r.chan and g.spec is r.spec, what


def test_verify_track_l(row, scen):
    """Every scenario on gate V's own candidate (timing known, fit=False:
    identical inputs), and the fits: blind, local, a narrower local and a
    span erased."""
    runs = [(label, s.v, dict(fit=False)) for label, s in scen.items() if s.v is not None]
    for label in ("tiny-12", "short-12", "drift", "noise", "qfloor"):
        s = scen[label]
        runs.append((label, dataclasses.replace(s.det, path=(s.v or s.det).path), {}))
    for label in ("tiny-12", "short-6"):
        s = scen[label]
        runs += [(label, s.v, {}), (label, s.v, dict(tau_range=2.0)),
                 (label, s.v, dict(fit=False, erase_s=[(10.0, 20.0)]))]
    for label, cand, kw in runs:
        s = scen[label]
        r = LT.verify_track_l(s.chan, s.spec, cand, **kw)
        g = row["verify_track_l"](s.chan, s.spec, cand, **kw)
        what = (label, cand.timing is None, kw)
        _track_close(g, r, what)
        _timing_close(g.timing, r.timing, s.spec, what)
        assert (g.timing is cand.timing) == (r.timing is cand.timing), what
        c, gain, dist = _templates(g)
        assert LT.resid_ratio(g, c, gain, dist) == g.resid_ratio, what
        z = T.z_ref_stat(g, c, d_known=c) / math.sqrt(max(g.resid_ratio, 1.0))
        _le("track z_ref own arrays", _r(g.z_ref, z), 1e-12, what)


def test_probe_q_floor(row, scen):
    """The weak steady pass (l-tiny -18 dB, no fading): its tracker's q sits
    at the bottom of its grid, where a smoother can be ill-conditioned. It
    is not here: a 1e-13-sample nudge of the reference's own timing moves
    its Z_ref and P by ~1e-13, so this scenario is held to the signal bars
    (_track_close, every test above) and the C++ track to 1e-9."""
    s = scen["qfloor"]
    rt = LT.verify_track_l(s.chan, s.spec, s.v, fit=False)
    assert rt.q == T.q_grid(L_T_SYM)[0], rt.q
    gt = row["verify_track_l"](s.chan, s.spec, s.v, fit=False)
    assert gt.q == rt.q
    _track_close(gt, rt, "q floor")
    nudged = dataclasses.replace(s.v, timing=dataclasses.replace(s.v.timing, tau0=s.v.timing.tau0 + 1e-13))
    rn = LT.verify_track_l(s.chan, s.spec, nudged, fit=False)
    _le("q floor reference's own z_ref (1e-13 nudge)", _r(rn.z_ref, rt.z_ref), 1e-9)
    _le("q floor reference's own P (1e-13 nudge)", rel(rn.P, rt.P), 1e-9)


def _native_primitives(native, mp, *, gate=False):
    """Assign the C++ primitives that verify_track_l composes over the
    reference's names (the path spline, derotate, the coarse fit, the
    tracker, its classes, templates and statistics); with gate=True also
    those verify_l composes (verify_track_l itself and the path
    measurements)."""
    from qrss_native.ltrack import rows as l_rows
    from qrss_native.ltrack_timing import rows as lt_rows
    from qrss_native.track import rows as core_rows
    from qrss_native.track_timing import rows as timing_rows
    from qrss_native.trackbase import rows as base_rows

    q = native.qrss
    table = {**base_rows(q), **core_rows(q), **timing_rows(q), **l_rows(q), **lt_rows(q)}
    for name in ("erase", "z_ref_stat"):
        mp.setattr(T, name, table[(TRACK, name)])
    for name in ("freq_path_spline", "derotate"):
        mp.setattr(LT, name, table[(TRACK, name)])
    for name in ("fit_timing_l", "make_classes_l", "track_l", "data_templates", "resid_ratio"):
        mp.setattr(LT, name, table[(LTRACK, name)])
    if gate:
        mp.setattr(T, "line_path", table[(TRACK, "line_path")])
        mp.setattr(LT, "_flat_path", table[(TRACK, "_flat_path")])
        for name in ("verify_track_l", "refine_path_l"):
            mp.setattr(LT, name, table[(LTRACK, name)])


def test_gate_v_composition(native, row, scen):
    """The reference's verify_track_l and verify_l with the C++ primitives
    they compose assigned over their own are the C++ verify_track_l and
    verify_l bit for bit (Z_ref, R, the timing, every array, the path): the
    glue -- the local search's range, the classes, the tracker's arguments,
    the erasure, the templates, R's scaling, the bases tried in order, the
    runners-up, the choice and the acceptance -- is the reference's
    exactly, and every difference above is a primitive's, measured in that
    primitive's own parity test."""
    for label, s in scen.items():
        cands = [(dataclasses.replace(s.det, path=(s.v or s.det).path), {})]
        if s.v is not None:
            cands += [(s.v, dict(fit=False)), (s.v, dict(tau_range=2.0, erase_s=[(10.0, 20.0)]))]
        for cand, kw in cands:
            g = row["verify_track_l"](s.chan, s.spec, cand, **kw)
            with pytest.MonkeyPatch.context() as mp:
                _native_primitives(native, mp)
                h = LT.verify_track_l(s.chan, s.spec, cand, **kw)
            what = (label, kw)
            for f in ("pos",) + ARRAYS:
                _same(getattr(g, f), getattr(h, f), (what, f))
            assert (g.q, g.loglik, g.z_ref, g.resid_ratio, g.delta) == \
                (h.q, h.loglik, h.z_ref, h.resid_ratio, h.delta), what
            for f in ("tau0", "ppm", "gamma", "z"):
                assert getattr(g.timing, f) == getattr(h.timing, f), (what, f)
        g = row["verify_l"](s.chan, CH_FS, s.spec, s.det)
        with pytest.MonkeyPatch.context() as mp:
            _native_primitives(native, mp, gate=True)
            h = LT.verify_l(s.chan, CH_FS, s.spec, s.det)
        assert (g is None) == (h is None), label
        if h is not None:
            assert g.z_ref == h.z_ref and g.timing.tau0 == h.timing.tau0 and g.timing.ppm == h.timing.ppm
            assert np.array_equal(g.path.t_s, h.path.t_s) and np.array_equal(g.path.f_hz, h.path.f_hz)
            assert g.waveform == h.waveform == WAVEFORM_L


def test_refusals(row, scen):
    """A CE frame: AssertionError from _l_only (fit_timing_l, verify_track_l,
    verify_l) or lframe.layout's error (resid_ratio), the reference's text."""
    spec = frame.TINY
    for fn, args, kw in (("fit_timing_l", (None, CH_FS, None, spec), dict(t0_index=0.0)),
                         ("verify_l", (None, CH_FS, spec, None), {}),
                         ("verify_track_l", (None, spec, None), {})):
        r, g = _outcome(getattr(LT, fn), *args, **kw), _outcome(row[fn], *args, **kw)
        assert r[0] is AssertionError and g == r, (fn, r, g)
    s = scen["tiny-6"]
    tr = LT.verify_track_l(s.chan, s.spec, s.v, fit=False)
    c, gain, dist = _templates(tr)
    for ce in (frame.TINY, frame.SHORT):
        bad = dataclasses.replace(tr, spec=ce)
        r, g = _outcome(LT.resid_ratio, bad, c, gain, dist), _outcome(row["resid_ratio"], bad, c, gain, dist)
        assert r[0] != "ok" and g == r, (ce.name, r, g)


def test_erase_span_refusals(row, scen):
    s = scen["tiny-6"]
    for bad in ([(1.0, 2.0, 3.0)], [(1.0,)], [1.0]):
        for fn, args in (("verify_track_l", (s.chan, s.spec, s.v)), ("verify_l", (s.chan, CH_FS, s.spec, s.v))):
            r = _outcome(getattr(LT, fn), *args, fit=False, erase_s=bad)
            g = _outcome(row[fn], *args, fit=False, erase_s=bad)
            assert r[0] != "ok" and g == r, (fn, bad, r, g)


# --- review probes ------------------------------------------------------------------------------

def test_receiver_channel_holds_every_lag(scen, blind):
    """The C++ correlates by the direct gather, the reference's blind search
    by an FFT, and the two differ only for a term whose interpolation pair
    straddles an end of x (the FFT path keeps the half inside; probed in
    test_probe_fit_timing_l_unpadded). The receiver's L channel
    (`channel_for`) is zero-padded past every lag the blind search tries
    at every clock error, with whole samples to spare, so no such term
    arises there: this pins that precondition."""
    for label, b in blind.items():
        tr = b.got[1]
        kn = LT.known_symbols_l(scen[label].spec)
        reach = float(kn.pos.max()) * SPS * (1 + tr["nd"] * tr["step"]) + tr["frac0"]
        first = tr["base"] + float(kn.pos.min()) * SPS * (1 - tr["nd"] * tr["step"])
        last = tr["base"] + tr["nlag"] - 1 + reach + 1      # the pair's second sample
        assert first >= 1.0, (label, first)
        assert last <= len(b.z_d) - 2, (label, last, len(b.z_d))


def test_probe_fit_timing_l_unpadded(Lt, row):
    """A capture whose frame starts inside the blind search (t0_index -50.5):
    the reference's FFT path keeps half of each term straddling x[0] and its
    direct path drops it, as the C++ does. The C++ grid is the reference's
    direct path to rounding there (the structural difference in
    ltrack_timing.hpp, which the receiver's padding keeps away); with the
    capture padded the two paths and the C++ agree, timings too."""
    rng = np.random.default_rng(2)
    spec = lframe.L_TINY
    z = rng.standard_normal(30000) + 1j * rng.standard_normal(30000)
    kn = LT.known_symbols_l(spec)
    for pad in (0, 2000):
        t0 = -50.5 + pad
        zp = np.concatenate([np.zeros(pad, dtype=np.complex128), z])
        ref, grids, segs = _ref_fit(None, CH_FS, None, spec, t0_index=t0, z_d=zp)
        g, tr = Lt._fit_timing_l_trace(None, CH_FS, None, spec, t0_index=t0, z_d=zp)
        assert tr["base"] == (-678 if not pad else 2000 - 678)
        deltas = np.arange(-tr["nd"], tr["nd"] + 1) * tr["step"]
        for lp_i, Z in enumerate(grids):
            sg = next(sg for p, sg in zip(tr["peaks"], segs) if p["lp"] == lp_i)
            norm = _norm(sg.sig2, kn)
            G = Lt._fit_grid_l(sg.x, kn, tr["base"], tr["nlag"], tr["frac0"], deltas, SPS, norm)
            D = _direct_grid(sg.x, kn, tr["base"], tr["nlag"], tr["frac0"], deltas, norm)
            _le("unpadded grid vs direct", np.max(np.abs(G - D)) / np.max(np.abs(D)), 1e-13, (pad, lp_i))
            gap = np.max(np.abs(Z - D)) / np.max(np.abs(D))
            if pad:
                _le("padded grid vs FFT", np.max(np.abs(G - Z)) / np.max(np.abs(Z)), 1e-12, (pad, lp_i))
            else:
                assert gap > 1e-3, gap                  # the straddling terms: structural
        if pad:
            _peaks_close(tr, segs, spec, "padded")
            _timing_close(_rx.to_ref(g[0]), ref, spec, "padded")


def _direct_grid(x, kn, base, nlag, frac0, deltas, norm):
    """fit_timing_l's direct path (nlag <= 64) restated for any nlag."""
    pos = np.asarray(kn.pos, dtype=np.float64)
    a = np.asarray(kn.val, dtype=np.complex128)
    lags = np.arange(nlag)
    lagph = LT._lag_phase(lags)
    out = np.empty((len(deltas), nlag))
    for i, d in enumerate(deltas):
        o = pos * SPS * (1 + d) + frac0
        idx = base + o[:, None] + lags[None, :]
        i0 = np.floor(idx).astype(np.int64)
        f = idx - i0
        ok = (i0 >= 0) & (i0 + 1 < len(x))
        i0c = np.clip(i0, 0, len(x) - 2)
        v = (x[i0c] * (1 - f) + x[i0c + 1] * f) * ok
        out[i] = (lagph * ((np.conj(a) * LT._clock_phase(pos, d)) @ v)).real / norm
    return out


def test_probe_fit_grid_l_random(Lt):
    """The grid on random inputs against the reference's direct path: x of
    random length (shorter than the template's span too), a base before x
    or past it, random fractions and clock errors, complex and real known
    symbols, one symbol, none."""
    rng = np.random.default_rng(11)
    kn_l = LT.known_symbols_l(lframe.L_TINY)
    span = int(kn_l.pos.max() * SPS) + 1
    for trial in range(24):
        k = int(rng.integers(0, 4))
        if k == 0:
            kn = kn_l
        elif k == 1:
            sel = np.sort(rng.choice(len(kn_l.pos), int(rng.integers(1, 40)), replace=False))
            kn = T._Known(pos=kn_l.pos[sel], val=kn_l.val[sel])
        elif k == 2:
            kn = T._Known(pos=kn_l.pos[::4], val=np.sign(kn_l.val.real[::4]) + 0.0)
        else:
            kn = T._Known(pos=kn_l.pos[:1], val=kn_l.val[:1])
        nx = int(rng.integers(span // 2, span + 400))
        x = rng.standard_normal(nx) + 1j * rng.standard_normal(nx)
        base = int(rng.integers(-80, 300))
        nlag = int(rng.integers(1, 120))
        frac0 = float(rng.uniform(0, 1))
        nd = int(rng.integers(0, 4))
        deltas = np.arange(-nd, nd + 1) * float(rng.uniform(1e-5, 3e-4))
        norm = float(rng.uniform(0.5, 4.0))
        ref = _direct_grid(x, kn, base, nlag, frac0, deltas, norm)
        got = Lt._fit_grid_l(x, kn, base, nlag, frac0, deltas, SPS, norm)
        assert got.shape == ref.shape, trial
        scale = max(np.max(np.abs(ref)), 1e-300)
        _le("grid random", np.max(np.abs(got - ref)) / scale, 1e-13, (trial, k, nx, base, nlag))
    # no known symbol: every cell 0
    empty = T._Known(pos=np.zeros(0, dtype=np.int64), val=np.zeros(0, dtype=np.complex128))
    got = Lt._fit_grid_l(x, empty, 0, 5, 0.25, np.zeros(1), SPS, 1.0)
    assert got.shape == (1, 5) and not np.any(got)


def test_probe_segment_timing_l_random(row, scen, blind):
    """_segment_timing_l on the reference's own statistic, perturbed: x cut
    short (late segments fall off its end), cut at the front (early ones
    before it), extra noise, the start and slip moved, real known symbols,
    segment lengths from 4 s to 300 s."""
    rng = np.random.default_rng(7)
    for label in ("tiny-12", "short-12"):
        spec, s = scen[label].spec, blind[label].ref[2][0]
        for trial in range(10):
            mode = trial % 4
            x, cut = s.x, 0
            if mode == 1:
                x = s.x[:int(len(s.x) * rng.uniform(0.4, 0.95))]
            elif mode == 2:
                cut = int(s.t0abs + rng.uniform(-200, 600))
                x = s.x[cut:]
            elif mode == 3:
                x = s.x + (rng.standard_normal(len(s.x)) + 1j * rng.standard_normal(len(s.x))) * \
                    math.sqrt(s.sig2) * rng.uniform(0.2, 1.0)
            kn = s.kn if trial % 5 != 4 else T._Known(pos=s.kn.pos, val=np.sign(s.kn.val.real) + 0.0)
            args = (x, s.sig2, kn, s.t0abs + rng.uniform(-4, 4) - cut, s.delta + rng.uniform(-1e-4, 1e-4),
                    SPS, spec.n_pos, s.n_ref - cut, float(rng.choice([4.0, 7.0, 13.0, 20.0, 33.0, 300.0])))
            r, g = LT._segment_timing_l(*args), row["_segment_timing_l"](*args)
            what = (label, trial, mode)
            _le("segment random t0", abs(g[0] - r[0]), 1e-6, what)
            _le("segment random slip", _slip(g[1] - r[1], spec), 1e-6, what)
            _le("segment random score", abs(g[2] - r[2]) / max(abs(r[2]), 1e-4), 1e-8, what)


def test_probe_nan_timing_keeps_its_identity(row, scen):
    """verify_track_l(fit=False) hands back cand.timing itself, whatever it
    holds: a NaN z or covariance entry too (the binding compared timings
    with ==, which a NaN fails, so the row returned a copy)."""
    s = scen["tiny-6"]
    for tm in (dataclasses.replace(s.v.timing, z=float("nan")),
               dataclasses.replace(s.v.timing, cov=np.diag([np.nan, 0.0, 0.0])),
               dataclasses.replace(s.v.timing, cov=np.diag([4.0, 0.0]))):
        cand = dataclasses.replace(s.v, timing=tm)
        r = LT.verify_track_l(s.chan, s.spec, cand, fit=False)
        g = row["verify_track_l"](s.chan, s.spec, cand, fit=False)
        assert r.timing is tm and g.timing is tm
        _track_close(g, r, ("nan timing", tm.z, tm.cov.shape))
        g = row["verify_track_l"](s.chan, s.spec, cand)           # fitted: a new timing
        assert g.timing is not tm


def test_probe_erasures(row, scen):
    """Gate V with spans erased: everything (Z_ref 0, R 1, refused), the
    preamble only, the lead and the frame's first seconds."""
    s = scen["tiny-6"]
    cand = dataclasses.replace(s.det, timing=s.v.timing)
    for es in ([(-100.0, 1e4)], [(0.0, 5.0)], [(-30.0, 0.5)], [(20.0, 21.0), (19.0, 30.0)]):
        r = LT.verify_track_l(s.chan, s.spec, s.v, fit=False, erase_s=es)
        g = row["verify_track_l"](s.chan, s.spec, s.v, fit=False, erase_s=es)
        _track_close(g, r, ("erase", es))
        if es[0][1] > 1e3:
            assert r.z_ref == g.z_ref == 0.0 and r.resid_ratio == g.resid_ratio == 1.0
        _det_close(row["verify_l"](s.chan, CH_FS, s.spec, cand, fit=False, erase_s=es),
                   LT.verify_l(s.chan, CH_FS, s.spec, cand, fit=False, erase_s=es), s.spec, ("erase", es))
    assert LT.verify_l(s.chan, CH_FS, s.spec, cand, fit=False, erase_s=[(-100.0, 1e4)]) is None


def test_probe_threads(native, row, scen):
    """The row releases the GIL: gate V from four threads at once is the
    serial result bit for bit."""
    import threading

    s = scen["tiny-12"]
    want = row["verify_track_l"](s.chan, s.spec, s.v, fit=False)
    out, errs = [None] * 4, []

    def run(k):
        try:
            out[k] = row["verify_track_l"](s.chan, s.spec, s.v, fit=False)
        except Exception as e:                  # pragma: no cover - reported below
            errs.append(e)

    th = [threading.Thread(target=run, args=(k,)) for k in range(4)]
    for t in th:
        t.start()
    for t in th:
        t.join()
    assert not errs, errs
    for g in out:
        assert g.z_ref == want.z_ref and g.resid_ratio == want.resid_ratio
        for f in ARRAYS:
            _same(getattr(g, f), getattr(want, f), f)


def test_probe_ce_signal_under_l(row):
    """A CE TINY signal at -6 dB gated as L (the slow test_a_ce_signal_is_not_l
    of test_qrss_ltrack, against the C++): the pilot taken on its carrier
    and 22 Hz above it, R well over 1 and the deciding factor (Z_ref times
    sqrt(R) over Z_ACCEPT, Z_ref under it), the same in both; verify_l
    refuses it at the carrier after every base and runner-up."""
    from qrss_demod_tracks import _ce_sim

    spec = lframe.L_TINY
    _, sim = _ce_sim("tiny", -6.0, 1)
    prep = RX.prepare(frontend.Capture(sim.fe, Q_TEST, sim.t0_index))
    for off in (0.0, 22.0):
        det = flat_det(1500.0 + off, WAVEFORM_L)
        chan = RX.channel_for(prep, spec, det)
        chan.noise_psd_at(0.0)
        r = LT.verify_track_l(chan, spec, det)
        g = row["verify_track_l"](chan, spec, det)
        _track_close(g, r, ("CE under L", off))
        _timing_close(g.timing, r.timing, spec, ("CE under L", off))
        assert r.resid_ratio > 5.0, r.resid_ratio
        for tr in (r, g):
            assert tr.z_ref * math.sqrt(tr.resid_ratio) > 6.0 > 1.5 * tr.z_ref, (off, tr.z_ref)
        if off == 0.0:
            assert LT.verify_l(chan, CH_FS, spec, det) is None
            assert row["verify_l"](chan, CH_FS, spec, det) is None


@pytest.mark.slow
def test_probe_preamble_candidates(row):
    """Acquisition's own L candidates (receiver.candidates: a preamble one,
    whose Timing carries A2's 2x2 covariance and lambda as z, and a blind
    one) through gate V, and the preamble one with a 40-sample tau0 sd, so
    that gate V's local search spans 321 lags and the reference correlates
    it by FFT."""
    for name, snr, seed in (("l-tiny", -9.0, 21), ("l-short", -9.0, 3)):
        spec = lframe.get(name)
        _, sim = l_scenario(name, snr, seed)
        prep = RX.prepare(frontend.Capture(sim.fe, Q_TEST, sim.t0_index))
        dets = RX.candidates(prep, spec)
        assert {d.method for d in dets} >= {"preamble"}, [d.method for d in dets]
        for d in dets[:3]:
            chan = RX.channel_for(prep, spec, d)
            chan.noise_psd_at(0.0)
            cands = [d]
            if d.timing is not None:
                assert d.timing.cov.shape == (2, 2)
                cands.append(dataclasses.replace(d, timing=dataclasses.replace(d.timing,
                                                                               cov=np.diag([1600.0, 0.0]))))
            for c in cands:
                what = (name, c.method, None if c.timing is None else c.timing.cov[0, 0])
                _det_close(row["verify_l"](chan, CH_FS, spec, c), LT.verify_l(chan, CH_FS, spec, c), spec, what)


# --- the whole caller ------------------------------------------------------------------------------

def _counted(table, used):
    def wrap(name, fn):
        if isinstance(fn, type):
            class Counted(fn):
                def __init__(self, *a, **k):
                    used[name] = used.get(name, 0) + 1
                    super().__init__(*a, **k)
            return Counted

        def f(*a, **k):
            used[name] = used.get(name, 0) + 1
            return fn(*a, **k)
        return f
    return {key: wrap(key[1], fn) for key, fn in table.items()}


def _patched(native, monkeypatch, keep=()):
    """Every row of tests/qrss_native/{trackbase,track,track_timing,ltrack,
    ltrack_timing}.py (every row over track's and ltrack's names) over its
    target and over every from-import site of it in sstvae.qrss (counting
    spies installed before the rows), but those named in `keep`, which stay
    the reference's. Returns {name: calls}."""
    import sys

    from qrss_native.ltrack import rows as l_rows
    from qrss_native.ltrack_timing import rows as lt_rows
    from qrss_native.track import rows as core_rows
    from qrss_native.track_timing import rows as timing_rows
    from qrss_native.trackbase import rows as base_rows

    q = native.qrss
    table = {}
    for f in (base_rows(q), core_rows(q), timing_rows(q), l_rows(q), lt_rows(q)):
        assert not set(f) & set(table)
        table.update(f)
    assert all(t in (TRACK, LTRACK) for t, _ in table)
    assert set(keep) <= {n for _t, n in table}, keep
    table = {k: v for k, v in table.items() if k[1] not in keep}
    used: dict = {}
    wrapped = _counted(table, used)
    owners = {TRACK: T, LTRACK: LT}
    # the reference object each row replaces -> its row
    by_obj = {id(getattr(owners[t], n)): fn for (t, n), fn in wrapped.items()}
    sites = []
    for mname, mod in list(sys.modules.items()):
        if not mname.startswith("sstvae.qrss.") or mod is T or mod is None:
            continue
        for name, val in list(vars(mod).items()):
            if mod is LT and (LTRACK, name) in wrapped:
                continue                       # its own row, set below
            if id(val) in by_obj:
                sites.append((mod, name, by_obj[id(val)]))
    # ltrack's from-imports of track's rows are among the sites
    for name in ("_flat_path", "derotate", "freq_path_spline"):
        if (TRACK, name) in wrapped:
            assert any(m is LT and n == name for m, n, _ in sites), name
    for (t, n), fn in wrapped.items():
        monkeypatch.setattr(owners[t], n, fn)
    for mod, name, fn in sites:
        monkeypatch.setattr(mod, name, fn)
    return used


def _pass_same(g, r, what):
    assert g is not None, what
    assert g.uid.rsplit("_", 1)[0] == r.uid.rsplit("_", 1)[0], what
    assert (g.q, g.frame, g.waveform, g.estimator) == (r.q, r.frame, r.waveform, r.estimator), what
    assert g.header == r.header and (g.cw is None) == (r.cw is None), what
    if r.cw is not None:
        assert (g.cw.text, g.cw.keying, g.cw.agrees) == (r.cw.text, r.cw.keying, r.cw.agrees), what
    assert abs(g.f_hz - r.f_hz) <= 1e-6, what
    for f, tol in (("z", 1e-6), ("w", 1e-6), ("hdr_llr", 1e-4), ("psi", 1e-6)):
        a, b = np.asarray(getattr(g, f)), np.asarray(getattr(r, f))
        assert a.shape == b.shape and a.dtype == b.dtype, (what, f)
        assert a.size == 0 or np.max(np.abs(a.astype(np.float64) - b)) <= tol, (what, f)
    assert abs(g.timing.tau0 - r.timing.tau0) <= 1e-6 and abs(g.timing.ppm - r.timing.ppm) <= 1e-6, what
    for f in ("offset_hz", "doppler_hz", "snr2500_db", "z_ref", "kappa"):
        x, y = getattr(g.report, f), getattr(r.report, f)
        assert abs(x - y) <= 1e-6 * max(abs(y), 1.0), (what, f)
    assert (g.report.slip_free, g.report.suspect) == (r.report.slip_free, r.report.suspect), what


# which reference functions each configuration keeps, and which of this
# file's rows the caller must then reach
CONFIGS = {
    "every row": ((), {"verify_l"}),
    "verify_l kept": (("verify_l",), {"refine_path_l", "verify_track_l"}),
    "verify_track_l kept": (("verify_l", "verify_track_l"), {"refine_path_l", "fit_timing_l", "resid_ratio"}),
    "fit_timing_l kept": (("verify_l", "verify_track_l", "fit_timing_l"),
                          {"_lag_phase", "_clock_phase", "_segment_timing_l"}),
}


@pytest.fixture(scope="module")
def l_tiny_pass():
    """l-tiny at -12 dB through the reference's receive_pass and _gate, and
    the noise capture's _gate (refused after the runners-up)."""
    _, sim = l_scenario("l-tiny", -12.0, 1)
    prep = RX.prepare(frontend.Capture(sim.fe, Q_TEST, sim.t0_index))
    spec, det = lframe.L_TINY, flat_det(1500.0, WAVEFORM_L)
    ref = RX.receive_pass(prep, spec, det, return_tracks=True)
    assert ref is not None
    gate = RX._gate(prep, spec, det, (), ())
    assert gate is not None
    _, nsim = l_scenario("l-tiny", -80.0, 7)
    nprep = RX.prepare(frontend.Capture(nsim.fe, Q_TEST, nsim.t0_index))
    assert RX._gate(nprep, spec, det, (), ()) is None
    return SimpleNamespace(prep=prep, spec=spec, det=det, ref=ref, gate=gate, nprep=nprep)


@pytest.mark.parametrize("config", list(CONFIGS))
def test_l_receive_pass_on_the_cpp(native, Lt, l_tiny_pass, monkeypatch, config):
    """receiver.receive_pass and receiver._gate on an l-tiny pass at -12 dB,
    and _gate on the noise capture, with every row of the five rows files
    but the reference functions the configuration keeps: the same pass,
    detection and refusal, and every row of this file the configuration is
    for reached from the caller (the runners-up of a blind candidate on its
    whole-frame line reached through _gate on noise)."""
    p = l_tiny_pass
    keep, must = CONFIGS[config]
    used = _patched(native, monkeypatch, keep)
    got = RX.receive_pass(p.prep, p.spec, p.det, return_tracks=True)
    _pass_same(got[0], p.ref[0], config)
    assert len(got[1]) == len(p.ref[1]), config
    _det_close(RX._gate(p.prep, p.spec, p.det, (), ()), p.gate, p.spec, (config, "_gate"))
    assert RX._gate(p.nprep, p.spec, p.det, (), ()) is None, config
    assert must <= set(used), (config, sorted(used))
    assert {"track_l", "make_classes_l", "report_l"} <= set(used), (config, sorted(used))
    if not keep:
        # gate V's own tracks run inside the C++ verify_l; rounds A and B through the row
        assert used.get("verify_l", 0) == 3, used
        assert used.get("track_l", 0) == len(p.ref[1]), used
    if keep == ("verify_l",):
        # the reference's verify_l: the noise capture's blind candidate has its
        # runners-up tracked too, through the verify_track_l row
        assert used["verify_track_l"] > used.get("refine_path_l", 0), used


@pytest.mark.slow
def test_l_short_receive_pass_on_the_cpp(native, Lt, monkeypatch):
    """l-short at -6 dB (header, callsign windows) through receive_pass with
    every row: the same pass (slow: an l-short pass)."""
    _, sim = l_scenario("l-short", -6.0, 1)
    prep = RX.prepare(frontend.Capture(sim.fe, Q_TEST, sim.t0_index))
    det = flat_det(1500.0, WAVEFORM_L)
    ref = RX.receive_pass(prep, lframe.L_SHORT, det, return_tracks=True)
    assert ref is not None and ref[0].header is not None
    used = _patched(native, monkeypatch)
    got = RX.receive_pass(prep, lframe.L_SHORT, det, return_tracks=True)
    _pass_same(got[0], ref[0], "l-short")
    assert len(got[1]) == len(ref[1]) == 2
    assert used.get("verify_l", 0) == 1 and used.get("track_l", 0) == 2, used


def _mixed(name):
    cfg = (("neighbours", (Neighbour(60.0, 0.0, seed=7, waveform=WAVEFORM_CE),)),)
    _, sim = l_scenario(name, -12.0, 1, cfg=cfg)
    return frontend.Capture(sim.fe, sim.q, sim.t0_index)


def _joint(native, monkeypatch, name, frames):
    cap = _mixed(name)
    ref = RX.receive_slot(cap, frames)
    assert sorted((p.waveform, round(p.f_hz)) for p in ref) == [(WAVEFORM_CE, 1560), (WAVEFORM_L, 1500)]
    used = _patched(native, monkeypatch)
    got = RX.receive_slot(cap, frames)
    assert len(got) == len(ref)
    for g, r in zip(got, ref):
        _pass_same(g, r, (name, r.waveform))
    # both gates (CE's verify with its verify_track for R, L's verify_l), both trackers
    assert used.get("verify_l", 0) >= 1 and used.get("verify", 0) >= 1, used
    assert used.get("track_l", 0) >= 1 and used.get("track", 0) >= 1, used
    return used


def test_joint_receiver_on_a_mixed_capture(native, Lt, monkeypatch):
    """The joint receiver, receive_slot(cap, (TINY, L_TINY)), on l-tiny at
    -12 dB with a CE TINY signal of the same power 60 Hz above: both
    received, each as its own waveform, as the reference receives them."""
    _joint(native, monkeypatch, "l-tiny", (frame.TINY, lframe.L_TINY))


@pytest.mark.slow
def test_joint_receiver_on_a_mixed_short_capture(native, Lt, monkeypatch):
    """The same with (SHORT, L_SHORT) on l-short (slow: two SHORT passes)."""
    _joint(native, monkeypatch, "l-short", (frame.SHORT, lframe.L_SHORT))

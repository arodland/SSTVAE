"""The A-7 false-alarm recheck on the C++ statistic (cpp-port-plan.md, stage 2
exit criteria; docs/qrss/design.md 10.4, A-6 and A-7).

TBD_GUMBEL_*, A2's Gamma thresholds and A1's frozen threshold were
calibrated on the Python statistic. A normalisation difference in the C++
(a median taken over other bins, a CFAR divided in another precision, a
Viterbi summing its path in another order) would move every false-alarm
rate built on them with no test failing, since a calibration is only ever
checked on noise. So each is measured again here through the C++, on the
reference's own procedure and noise (numpy's generators: the same seeds,
the same captures), and against the reference on the same captures.

- A-7, the procedure behind TBD_GUMBEL_* (test_qrss_acquire.py's
  `_noise_tbd_z`: complex white-noise captures at 500 Hz, 6 disjoint
  321-bin channels each, through acquire's spectrogram, CFAR and Viterbi;
  Gumbel by moments), run by the reference's own function with the rows of
  tests/qrss_native/acquire_stats.py and acquire_tbd.py over acquire's
  names (the native entry points counted: every capture reaches each).
  The fast test takes 20 captures at each of the four lengths, the slow
  one per length the 200 captures of test_a7_measure_tbd_gumbel (seeds
  1000 + n) on both sides and the other 200 the constants were measured
  on (seeds 2000 + n) on the C++.
- A3 as production runs it (`track_before_detect` on FE noise slots, the
  band-wide CFAR, every channel): the C++ hits against the reference's.
- A-6: A1's statistic against its frozen threshold (20 captures, 2.5e6
  cells) and, slow, A2's Lambda against its Gamma tails and A2's combined
  p (40 captures, 1e6 cells per scale), C++ against the reference.
- End to end, slow: noise-only slots through the C++ receive_slot (joint
  CE + L, budget_s None, typed: asserted by spies), 60 at SHORT length and
  40 at TINY length, and the reference on a subsample: passes, A2 passes
  and A3 hits at p < TBD_PFA counted, each capture's A3 statistic compared.

The bars, and in brackets what was measured (2026-10-14):

- Z per capture, C++ against the reference: 1e-9 absolute [0: bit for bit,
  every capture of every length, 1,600 captures and 9,600 Z values];
- the C++ Gumbel on the committed procedure: test_a7_measure_tbd_gumbel's
  own (mu within 0.1, beta within 10%, exceedance at the committed 10% and
  1% points under 1.25 and 1.0 of nominal), and over both seed sets (the
  2,400 Z values the constants were fitted to) mu within 0.002 and beta
  within 0.002 of the committed constants [see MEASURED below];
- A3 hits on production noise slots: every field the reference's,
  p_value to 1e-12 relative (gumbel_p's numpy SIMD exp) [Z, paths, f_hz
  bit for bit; p_value 0];
- A1: the statistic within one float32 ulp in 1e-5 of cells (the plan's
  "a few 1-ulp flips") and the same exceedance counts at 1e-3 and 1e-4
  [see MEASURED];
- A2: Lambda at every scale to 1e-9 relative, p to 1e-9 relative, the
  same exceedance counts [see MEASURED];
- noise slots: no pass on either side, and per capture the same A2 passes,
  A3 hits and A3 statistic (Z of every channel bit for bit) as the
  reference's.

MEASURED holds the numbers (`pytest -s` prints them again).
"""

import importlib
import math
import time

import numpy as np
import pytest

import test_qrss_acquire as TA
import test_qrss_native_receiver as P
from qrss_helpers import Q_TEST
from qrss_native import _owner, require
from sstvae.qrss import acquire as A
from sstvae.qrss import frame, frontend, lframe
from sstvae.qrss import receiver as RX
from sstvae.qrss.constants import FE_FS, T_SYM

LENGTHS = A.TBD_GUMBEL_N                 # A3's frames on TINY, SHORT, MEDIUM, FULL
Z_TOL = 1e-9                             # per-capture Z, C++ against the reference


@pytest.fixture(scope="module")
def Q(native):
    for name in ("acquire_stats", "acquire_tbd", "acquire_a12", "acquire", "receiver_slot"):
        require(native, name)
    return native.qrss


# --- harness ------------------------------------------------------------------------------------

def _count(mod, names, calls, mp):
    """Count the calls into the native entry points `names` of `mod` (set
    before any row is built, since a row may hold the function itself)."""
    for name in names:
        fn = getattr(mod, name)

        def spy(*a, _fn=fn, _name=name, **k):
            calls[_name] = calls.get(_name, 0) + 1
            return _fn(*a, **k)
        mp.setattr(mod, name, spy)


def _rows(q, mp, files=("acquire_stats", "acquire_tbd", "acquire_a12")):
    """The rows of tests/qrss_native/<files>.py over acquire's names."""
    for f in files:
        for (target, name), fn in importlib.import_module(f"qrss_native.{f}").rows(q).items():
            assert target.startswith("sstvae.qrss.acquire"), target
            mp.setattr(_owner(target), name, fn)


_STAT = {"acquire_stats": ("spectrogram", "cfar_normalise"), "acquire_tbd": ("tbd_viterbi",)}


def _cpp_noise_tbd_z(q, n_frames, n_caps, seed):
    """test_qrss_acquire._noise_tbd_z, the reference's own function, with the
    C++ spectrogram, CFAR and Viterbi under it; each reached once per capture."""
    calls = {}
    with pytest.MonkeyPatch.context() as mp:
        for mod, names in _STAT.items():
            _count(getattr(q, mod), names, calls, mp)
        _rows(q, mp, ("acquire_stats", "acquire_tbd"))
        Z = TA._noise_tbd_z(n_frames, n_caps, seed)
    assert calls == {"spectrogram": n_caps, "cfar_normalise": n_caps, "tbd_viterbi": n_caps}, calls
    return Z


def _dz(got, ref, what):
    got, ref = np.asarray(got), np.asarray(ref)
    assert got.shape == ref.shape and got.dtype == ref.dtype, (what, got.shape, ref.shape)
    d = float(np.max(np.abs(got - ref))) if got.size else 0.0
    assert d <= Z_TOL, (what, d)
    return d, got.tobytes() == ref.tobytes()


def _gumbel_ok(Z, n, mean_abs, sd_rel, p5):
    """test_a7_tbd_gumbel_fast_subsample's check of a subsample against the
    committed Gumbel."""
    mu, be = A.tbd_gumbel(n)
    assert np.mean(Z) == pytest.approx(mu + np.euler_gamma * be, abs=mean_abs), (n, np.mean(Z))
    assert np.std(Z) == pytest.approx(be * math.pi / math.sqrt(6), rel=sd_rel), (n, np.std(Z))
    assert np.mean(Z > TA._gumbel_point(mu, be, 0.05)) <= p5, n


def _tails(Z, n):
    """Exceedance over nominal at the committed Gumbel's 10% and 1% points."""
    i = LENGTHS.index(n)
    mu, be = A.TBD_GUMBEL_MU[i], A.TBD_GUMBEL_BETA[i]
    return [float(np.mean(Z > TA._gumbel_point(mu, be, p)) / p) for p in (0.1, 0.01)]


# --- A-7: the Gumbel procedure --------------------------------------------------------------------

def test_a7_subsample_on_the_cpp(Q):
    """20 noise captures per length (120 Z values): the C++ Z is the
    reference's on the same captures, and holds the committed Gumbel as
    test_a7_tbd_gumbel_fast_subsample holds the reference's (mean within
    0.2, sd within 25%, at most 10% past its 5% point)."""
    for n in LENGTHS:
        seed = 21 if n == A.tbd_n_frames(frame.SHORT) else 21 + n
        ref = TA._noise_tbd_z(n, 20, seed=seed)
        got = _cpp_noise_tbd_z(Q, n, 20, seed)
        assert len(got) == 120
        d, same = _dz(got, ref, n)
        _gumbel_ok(got, n, 0.2, 0.25, 0.10)
        mu, be = TA._gumbel_moments(got)
        print(f"A-7 subsample n={n}: max|dZ| {d:.3g} (bitwise {same}); mu={mu:.3f} beta={be:.3f}")


@pytest.mark.slow
@pytest.mark.parametrize("n", LENGTHS)
def test_a7_measure_on_the_cpp(Q, n):
    """The measurement behind TBD_GUMBEL_* through the C++: 200 captures
    (seeds 1000 + n) on both sides, compared capture by capture, the C++
    held to test_a7_measure_tbd_gumbel's criteria; and the 400 captures the
    constants were fitted to (seeds 1000 + n and 2000 + n) on the C++,
    whose moments are held to the committed values within 0.002."""
    i = LENGTHS.index(n)
    mu_c, be_c = A.TBD_GUMBEL_MU[i], A.TBD_GUMBEL_BETA[i]
    t = time.perf_counter()
    ref = TA._noise_tbd_z(n, 200, seed=1000 + n)
    t_py = time.perf_counter() - t
    t = time.perf_counter()
    got = _cpp_noise_tbd_z(Q, n, 200, 1000 + n)
    t_cpp = time.perf_counter() - t
    d, same = _dz(got, ref, n)
    mu, be = TA._gumbel_moments(got)
    mu_r, be_r = TA._gumbel_moments(ref)
    tail = _tails(got, n)
    assert mu == pytest.approx(mu_c, abs=0.1) and be == pytest.approx(be_c, rel=0.1), (mu, be)
    assert tail[0] < 1.25 and tail[1] < 1.0, tail
    both = np.concatenate([got, _cpp_noise_tbd_z(Q, n, 200, 2000 + n)])
    mu2, be2 = TA._gumbel_moments(both)
    tail2 = _tails(both, n)
    assert abs(mu2 - mu_c) <= 0.002 and abs(be2 - be_c) <= 0.002, (mu2, be2)
    assert tail2[0] < 1.25 and tail2[1] < 1.0, tail2
    print(f"A-7 n={n}: 200 captures, max|dZ| {d:.3g} (bitwise {same}); C++ mu={mu:.4f} "
          f"beta={be:.4f}, reference mu={mu_r:.4f} beta={be_r:.4f}; exceedance/nominal at 10%, 1%: "
          f"{tail[0]:.2f}, {tail[1]:.2f}. Over 400 (2400 Z): mu={mu2:.4f} beta={be2:.4f} "
          f"(committed {mu_c}, {be_c}), exceedance {tail2[0]:.2f}, {tail2[1]:.2f}. "
          f"Time per capture: reference {t_py / 200 * 1e3:.1f} ms, C++ {t_cpp / 200 * 1e3:.1f} ms")


# --- A3 as production runs it ----------------------------------------------------------------------

def _hits_same(got, ref, what):
    assert len(got) == len(ref), (what, len(got), len(ref))
    worst = 0.0
    for k, (g, r) in enumerate(zip(got, ref)):
        assert type(g) is type(r) is A.TbdHit, what
        assert (g.z, g.n_frames, g.centre_hz) == (r.z, r.n_frames, r.centre_hz), (what, k)
        for f in ("t_s", "f_hz", "weight"):
            assert getattr(g.path, f).tobytes() == getattr(r.path, f).tobytes(), (what, k, f)
        e = abs(g.p_value - r.p_value) / max(abs(r.p_value), 1e-300)
        assert e <= 1e-12, (what, k, g.p_value, r.p_value)
        worst = max(worst, e)
        assert g.passed == r.passed, (what, k)
    return worst


def test_a7_production_a3_on_the_cpp(Q):
    """test_a7_production_tbd_matches_calibration_on_noise's two SHORT noise
    slots through the C++ track_before_detect (the band-wide CFAR, every
    channel): the reference's hits, and its Gumbel."""
    spec = frame.SHORT
    rng = np.random.default_rng(22)
    n = int((TA.PRE_S + spec.keyed_end_pos * T_SYM + 3) * FE_FS)
    xs = [TA._noise_fe(n, rng) for _ in range(2)]
    t0 = TA.PRE_S * FE_FS
    ref = [A.track_before_detect(x, t0, spec, all_hits=True) for x in xs]
    calls = {}
    with pytest.MonkeyPatch.context() as mp:
        _count(Q.acquire_tbd, ("track_before_detect",), calls, mp)
        _rows(Q, mp)
        got = [A.track_before_detect(x, t0, spec, all_hits=True) for x in xs]
    assert calls == {"track_before_detect": 2}, calls
    worst = max(_hits_same(g, r, k) for k, (g, r) in enumerate(zip(got, ref)))
    zs = [h.z for hs in got for h in hs]
    mu, be = A.tbd_gumbel(A.tbd_n_frames(spec))
    assert len(zs) > 60
    assert np.mean(zs) == pytest.approx(mu + np.euler_gamma * be, abs=0.4)
    n_hit = sum(h.p_value < A.TBD_PFA for hs in got for h in hs)
    assert n_hit <= 1
    print(f"A3 production, 2 SHORT noise slots: {len(zs)} channels, mean Z {np.mean(zs):.3f} "
          f"(Gumbel mean {mu + np.euler_gamma * be:.3f}), hits at p < TBD_PFA {n_hit}; "
          f"p_value worst relative {worst:.3g}")


# --- A-6: A1 and A2 on noise ---------------------------------------------------------------------

def test_a6_a1_tail_on_the_cpp(Q):
    """test_a6_a1_tail_on_noise's 20 captures (2.5e6 cells) through the C++
    carrier_line_stats: the statistic the reference's (a float32 flip in at
    most 1e-5 of cells), the same exceedances of the frozen threshold at
    1e-3 and 1e-4, each within a factor 2 of nominal."""
    rng = np.random.default_rng(12)
    xs = [TA._noise_fe(30 * FE_FS, rng) for _ in range(20)]
    M = 11
    ref = np.concatenate([A.carrier_line_stats(x, 3 * FE_FS).stat.ravel() for x in xs])
    th_ref = [A.a1_threshold(M, p) for p in (1e-3, 1e-4)]
    calls = {}
    with pytest.MonkeyPatch.context() as mp:
        _count(Q.acquire_a12, ("carrier_line_stats",), calls, mp)
        _count(Q.acquire_stats, ("a1_threshold",), calls, mp)
        _rows(Q, mp)
        cs = [A.carrier_line_stats(x, 3 * FE_FS) for x in xs]
        th = [A.a1_threshold(M, p) for p in (1e-3, 1e-4)]
    assert calls == {"carrier_line_stats": 20, "a1_threshold": 2}, calls
    assert all(c.M == M for c in cs)
    got = np.concatenate([c.stat.ravel() for c in cs])
    assert got.dtype == ref.dtype == np.float32 and got.shape == ref.shape
    assert th == th_ref
    diff = got != ref
    ulps = np.abs(got.view(np.int32).astype(np.int64) - ref.view(np.int32).astype(np.int64))
    assert int(ulps.max()) <= 1 and diff.mean() <= 1e-5, (int(ulps.max()), int(diff.sum()))
    out = []
    for p, t in zip((1e-3, 1e-4), th):
        cg, cr = int(np.sum(got > t)), int(np.sum(ref > t))
        assert cg == cr, (p, cg, cr)
        r = cg / len(got) / p
        assert 0.5 < r < 2.0, (p, r)
        out.append(f"{p:g}: {cg} cells, {r:.2f} of nominal")
    print(f"A-6 A1, {len(got)} cells: {int(diff.sum())} differ (max {int(ulps.max())} ulp); "
          + "; ".join(out))


@pytest.mark.slow
def test_a6_a2_tail_on_the_cpp(Q):
    """test_a6_a2_tail_on_noise's 40 captures (over 1e6 cells per scale)
    through the C++ preamble_channel, _preamble_sums and preamble_p: Lambda
    at every scale and the combined p to 1e-9 relative of the reference's,
    the same exceedances of each scale's Gamma point at 1e-3 and 1e-4 and of
    the combined p at 1e-2 to 1e-4, and the test's own criteria."""
    from scipy import special

    rng = np.random.default_rng(11)
    inputs = []
    for _ in range(40):
        x = TA._noise_fe(40 * FE_FS, rng)
        inputs.append((x, 1500 + rng.uniform(-900, 900)))

    def run():
        lams, ps = {}, []
        for x, f in inputs:
            chan = A.preamble_channel(x, 8 * FE_FS, f)
            _, _, S, var = A._preamble_sums(chan)
            p, _, st = A.preamble_p(S, var)
            for k, lk in st:
                lams.setdefault(k, []).append(np.asarray(lk).ravel())
            ps.append(np.asarray(p).ravel())
        return {k: np.concatenate(v) for k, v in lams.items()}, np.concatenate(ps)

    t = time.perf_counter()
    lam_r, p_r = run()
    t_py = time.perf_counter() - t
    calls = {}
    with pytest.MonkeyPatch.context() as mp:
        _count(Q.acquire_a12, ("preamble_channel", "_preamble_sums"), calls, mp)
        _count(Q.acquire_stats, ("preamble_p",), calls, mp)
        _rows(Q, mp)
        t = time.perf_counter()
        lam_g, p_g = run()
        t_cpp = time.perf_counter() - t
    assert calls == {"preamble_channel": 40, "_preamble_sums": 40, "preamble_p": 40}, calls
    assert sorted(lam_g) == sorted(lam_r) == [1, 2, 5, 10]
    out = []
    for k in sorted(lam_r):
        g, r = lam_g[k], lam_r[k]
        assert g.shape == r.shape and len(r) > 1e6
        e = float(np.max(np.abs(g - r) / np.abs(r)))
        assert e <= 1e-9, (k, e)
        rs = []
        for p in (1e-3, 1e-4):
            th = special.gammainccinv(k, p)
            cg, cr = int(np.sum(g > th)), int(np.sum(r > th))
            assert cg == cr, (k, p, cg, cr)
            rs.append(cg / len(g) / p)
            assert 0.5 < rs[-1] < 2.0, (k, p, rs[-1])
        out.append(f"k={k}: rel {e:.2g}, {rs[0]:.2f} / {rs[1]:.2f} of nominal")
    ep = float(np.max(np.abs(p_g - p_r) / p_r))
    assert ep <= 1e-9, ep
    pr = []
    for p in (1e-2, 1e-3, 1e-4):
        cg, cr = int(np.sum(p_g < p)), int(np.sum(p_r < p))
        assert cg == cr, (p, cg, cr)
        pr.append(cg / len(p_g) / p)
        assert 1 / len(A.A2_SCALES) < pr[-1] < 1.2, (p, pr[-1])
    print(f"A-6 A2, {len(p_g)} cells per scale: " + "; ".join(out)
          + f"; combined p rel {ep:.2g}, at 1e-2/1e-3/1e-4: {pr[0]:.2f} / {pr[1]:.2f} / {pr[2]:.2f} "
          f"of nominal. Time: reference {t_py:.1f} s, C++ {t_cpp:.1f} s")


# --- end to end: noise-only slots through the C++ receive_slot ----------------------------------

def _noise_slot(specs, seed):
    """A noise-only slot capture: complex64 white noise from t0 - 12 s to 3 s
    past the longest frame (test_qrss_receiver's `_tiny_noise` for TINY)."""
    rng = np.random.default_rng(seed)
    n = int((12 + max(s.n_pos * s.t_sym for s in specs) + 3) * FE_FS)
    fe = ((rng.standard_normal(n) + 1j * rng.standard_normal(n)) / np.sqrt(2)).astype(np.complex64)
    return frontend.Capture(fe, Q_TEST, 12 * FE_FS)


def _listen(cap, specs):
    """receive_slot (budget_s None) and the acquisition behind it: passes,
    A2 passes, A3 hits at p < TBD_PFA, gate-V candidates, and A3's Z on
    every channel (the production statistic, all_hits)."""
    passes = RX.receive_slot(cap, specs, budget_s=None)
    prep = RX.prepare(cap)
    pc = frontend.Capture(prep.fe, prep.q, prep.t0_index, prep.fs)
    acq = A.acquire(pc, specs, preprocessed=True)
    longest = max(specs, key=lambda s: s.keyed_end_pos * s.t_sym)
    every = A.track_before_detect(prep.fe, prep.t0_index, longest, acq.lines, prep.fs, all_hits=True)
    return {"passes": [(p.f_hz, p.waveform) for p in passes], "a2": len(acq.preamble),
            "a3": len(acq.tbd), "a3_p": [h.p_value for h in acq.tbd],
            "cands": len(acq.detections(specs)), "z": np.array([h.z for h in every]),
            "n_frames": every[0].n_frames}


def _py_listen(seeds, specs):
    t = time.perf_counter()
    out = [_listen(_noise_slot(specs, s), specs) for s in seeds]
    return out, time.perf_counter() - t


def _cpp_listen(native, seeds, specs):
    """_listen with every tx and rx row in place: the slot received in C++
    whole (the binding's typed flag spied, and the per-pass bindings never
    reached through Python: only _listen's own prepare is), the
    acquisition and A3 the C++'s (counted)."""
    q = native.qrss
    seen, below, calls = [], [], {}
    with pytest.MonkeyPatch.context() as mp:
        for name in ("receive_slot", "_receive_slot_joint"):
            orig = getattr(q.receiver_slot, name)

            def spy(*a, _orig=orig, _name=name):
                seen.append((_name, a[-1]))
                return _orig(*a)
            mp.setattr(q.receiver_slot, name, spy)
        for name in ("prepare", "receive_pass"):
            orig = getattr(q.receiver, name)

            def spy2(*a, _orig=orig, _name=name, **k):
                below.append(_name)
                return _orig(*a, **k)
            mp.setattr(q.receiver, name, spy2)
        _count(q.acquire, ("acquire", "_acquire_joint"), calls, mp)
        _count(q.acquire_tbd, ("track_before_detect",), calls, mp)
        P._apply_rows(native, mp)
        t = time.perf_counter()
        out = [_listen(_noise_slot(specs, s), specs) for s in seeds]
        dt = time.perf_counter() - t
    assert len(seen) == len(seeds) and all(flag is True for _n, flag in seen), seen
    assert below == ["prepare"] * len(seeds), below
    assert calls.get("track_before_detect", 0) == len(seeds), calls
    assert calls.get("acquire", 0) + calls.get("_acquire_joint", 0) == len(seeds), calls
    return out, dt


def _summary(rs):
    z = np.concatenate([r["z"] for r in rs])
    return {"slots": len(rs), "passes": sum(len(r["passes"]) for r in rs),
            "a2": sum(r["a2"] for r in rs), "a3": sum(r["a3"] for r in rs),
            "cands": sum(r["cands"] for r in rs), "channels": len(z), "z": z}


# frames, C++ slots, reference slots (a subsample of the same seeds), first seed
SLOTS = {
    "short": ((frame.SHORT, lframe.L_SHORT), 60, 6, 5000),
    "tiny": ((frame.TINY, lframe.L_TINY), 40, 10, 9000),
}


@pytest.mark.slow
@pytest.mark.parametrize("name", list(SLOTS))
def test_noise_slots_on_the_cpp(native, Q, name):
    """Noise-only slots through the C++ receive_slot, CE and L listened for
    (budget_s None): no pass; A2 passes and A3 hits counted; and on the
    reference's subsample the same, capture by capture, with A3's Z on
    every channel bit for bit. A3's channels hold the committed Gumbel."""
    specs, n_cpp, n_py, seed0 = SLOTS[name]
    seeds = list(range(seed0, seed0 + n_cpp))
    got, t_cpp = _cpp_listen(native, seeds, specs)
    ref, t_py = _py_listen(seeds[:n_py], specs)
    for s, g, r in zip(seeds, got, ref):
        assert g["passes"] == r["passes"], s
        assert (g["a2"], g["a3"], g["cands"], g["n_frames"]) == \
            (r["a2"], r["a3"], r["cands"], r["n_frames"]), s
        assert np.allclose(g["a3_p"], r["a3_p"], rtol=1e-12, atol=0), s
        _dz(g["z"], r["z"], s)
    sg, sr = _summary(got), _summary(ref)
    assert sg["passes"] == 0 and sr["passes"] == 0, (sg["passes"], sr["passes"])
    n = got[0]["n_frames"]
    mu, be = A.tbd_gumbel(n)
    z = sg["z"]
    assert np.mean(z) == pytest.approx(mu + np.euler_gamma * be, abs=0.4), np.mean(z)
    p_ch = float(np.mean([A.gumbel_p(v, n) < A.TBD_PFA for v in z]))
    assert sg["a3"] <= max(3, 3 * A.TBD_PFA * sg["channels"]), sg["a3"]
    print(f"noise slots {name} ({'+'.join(s.name for s in specs)}), A3 over {n} frames: C++ "
          f"{sg['slots']} slots in {t_cpp:.1f} s: {sg['passes']} passes, {sg['a2']} A2 passes, "
          f"{sg['a3']} A3 hits at p < {A.TBD_PFA:g}, {sg['cands']} gate-V candidates; "
          f"{sg['channels']} A3 channels, mean Z {np.mean(z):.3f} "
          f"(Gumbel {mu + np.euler_gamma * be:.3f}), {p_ch:.2g} of channels at p < {A.TBD_PFA:g}. "
          f"Reference {sr['slots']} slots in {t_py:.1f} s: {sr['passes']} passes, {sr['a2']} A2 "
          f"passes, {sr['a3']} A3 hits, {sr['cands']} candidates")

"""Parity of native/core/qrss/demod (sstvae_native.qrss.demod) with
sstvae/qrss/demod.py, on the Python tracker's own TrackResults.

The tracks come from `receiver.receive_pass(..., return_tracks=True)` on
simulated CE passes (tests/qrss_demod_tracks.py): TINY at -12 dB, TINY with
10 s of its data erased, SHORT at -6 dB in both rounds (spread header
unknown, then known) and a genie SHORT track -- plus the deterministic stub
of the ctests and the edge cases the tracker does not produce.

The contract, and what was measured (2026-10-13, these tracks):

- extract and block_estimate: z, w, llr within one ulp and >= 99.9%
  bit-identical; measured bit-identical (0 ulps, 100%) on every track,
  both estimators, calibrated and not;
- symbol_estimates, data_estimates, spread_llr and extract's diag: 1e-12
  relative; measured 0 (bit-identical) -- numpy's fused complex multiply
  and pairwise sums are emulated, and the one transcendental (data's
  exp(-beta^2 rho / 2)) agrees with numpy's at the rho in use;
- calibrate: the same branch (no estimate / clamped / suspect) and kappa
  to 1e-12; measured equal to the bit;
- track.m replaced by a given m exactly as the reference does, and the
  reference's refusals with their texts.
"""

import copy
from types import SimpleNamespace

import numpy as np
import pytest

import qrss_demod_probes as P
import qrss_demod_tracks as D
from qrss_native import require
from sstvae.qrss import demod, frame, lframe

Q = 0x1234_5678_9ABC
CE_LABELS = ("tiny/A", "tiny-erased/A", "short/A", "short/B", "short/genie")


@pytest.fixture(scope="module")
def dm(native):
    return require(native, "demod")


@pytest.fixture(scope="module")
def tracks(dm):
    return D.ce_tracks()


def _f32(a, b, what=""):
    u, frac = D.ulps(a, b)
    assert u <= 1 and frac >= 0.999, (what, u, frac)


def _rel(a, b, what="", tol=1e-12):
    r = D.rel(a, b)
    assert r <= tol, (what, r)


def test_constants(dm):
    for name in ("D_PASS", "D_REF", "KAPPA_RANGE", "KAPPA_PRIOR_SD", "KAPPA_S2_MAX",
                 "KAPPA_S2_REL", "U_MIN_REL"):
        assert getattr(dm, name) == getattr(demod, name), name


def test_the_tracks_cover_what_they_claim(tracks):
    """The fixtures are the cases the docstring names (a guard on the guard)."""
    assert set(tracks) == set(CE_LABELS)
    assert np.isinf(demod.data_estimates(tracks["tiny-erased/A"])[1]).sum() > 100
    assert demod.data_estimates(tracks["short/A"])[2] > 0       # header unknown
    assert demod.data_estimates(tracks["short/B"])[2] == 0      # known


@pytest.mark.parametrize("label", CE_LABELS)
def test_symbol_and_data_estimates(dm, tracks, label):
    tr = tracks[label]
    lay = frame.layout(tr.spec)
    for which in ("data", "ref", "hdr_bits", "known_idx"):
        idx = np.asarray(getattr(lay, which))
        if not len(idx):
            continue
        a = demod.symbol_estimates(tr, idx)
        b = dm.symbol_estimates(tr, idx)
        assert len(b) == 3
        for x, y, part in zip(a, b, ("y", "s2n", "s2")):
            _rel(y, x, f"{which} {part}")
    a = demod.data_estimates(tr)
    b = dm.data_estimates(tr)
    _rel(b[0], a[0], "y")
    _rel(b[1], a[1], "s2n")
    assert b[2] == a[2] and type(b[2]) is float


@pytest.mark.parametrize("label", CE_LABELS)
def test_calibrate(dm, tracks, label):
    ka, sa = demod.calibrate(tracks[label])
    kb, sb = dm.calibrate(tracks[label])
    assert sb is sa
    assert abs(kb - ka) <= 1e-12 * abs(ka)
    lo, hi = demod.KAPPA_RANGE
    assert (kb in (lo, hi)) == (ka in (lo, hi)), "the same branch"


@pytest.mark.parametrize("label", ("short/A", "short/genie"))
def test_spread_llr(dm, tracks, label):
    tr = tracks[label]
    y, s2n, _ = demod.data_estimates(tr)
    for s2 in (s2n + demod.D_PASS, np.where(np.arange(len(s2n)) % 5 == 0, np.inf, s2n)):
        _rel(dm.spread_llr(y, s2, tr.spec), demod.spread_llr(y, s2, tr.spec))


@pytest.mark.parametrize("label", CE_LABELS)
@pytest.mark.parametrize("est", ("joint", "plain"))
def test_block_estimate(dm, tracks, label, est):
    y, s2n, extra = demod.data_estimates(tracks[label])
    s2 = s2n * 0.9 + demod.D_PASS + extra
    za, va = demod.block_estimate(y, s2, Q, est)
    zb, vb = dm.block_estimate(y, s2, Q, est)
    _f32(zb, za, "z")
    _f32(vb, va, "v")


def _same_extract(a, b):
    for k, part in enumerate(("z", "w", "llr")):
        assert b[k].dtype == np.float32
        _f32(b[k], a[k], part)
    da, db = a[3], b[3]
    assert set(db) == set(da)
    assert abs(db["kappa"] - da["kappa"]) <= 1e-12 * abs(da["kappa"])
    assert db["suspect"] is da["suspect"] and db["estimator"] == da["estimator"]
    for key in ("y", "s2", "y_hdr"):
        _rel(db[key], da[key], key)
    for key in ("llr_block", "llr_spread"):
        assert (db[key] is None) == (da[key] is None), key
        if da[key] is not None:
            _rel(db[key], da[key], key)


@pytest.mark.parametrize("label", CE_LABELS)
@pytest.mark.parametrize("est", ("joint", "plain"))
@pytest.mark.parametrize("cal", (True, False))
def test_extract(dm, tracks, label, est, cal):
    tr = tracks[label]
    _same_extract(demod.extract(None, tr, tr.spec, Q, est, calibrate_kappa=cal),
                  dm.extract(None, tr, tr.spec, Q, est, calibrate_kappa=cal))


def test_extract_with_given_m_replaces_the_tracks(dm, tracks):
    tr = tracks["short/A"]
    m = tr.m * (0.75 - 0.25j)
    ta, tb = copy.copy(tr), copy.copy(tr)
    _same_extract(demod.extract(m, ta, tr.spec, Q), dm.extract(m, tb, tr.spec, Q))
    assert np.array_equal(tb.m, ta.m) and np.array_equal(tb.m, m)
    # a list is taken through np.asarray, as the reference does
    ta, tb = copy.copy(tr), copy.copy(tr)
    _same_extract(demod.extract(list(m[:]), ta, tr.spec, Q), dm.extract(list(m[:]), tb, tr.spec, Q))
    assert isinstance(tb.m, np.ndarray) and np.array_equal(tb.m, ta.m)
    # the track's own m: left as the same object
    ta = copy.copy(tr)
    own = ta.m
    dm.extract(own, ta, tr.spec, Q)
    assert ta.m is own


@pytest.mark.parametrize("name,rho", [("tiny", 0.0), ("short", 0.054), ("short", 0.0)])
def test_stub_tracks(dm, name, rho):
    """The ctests' stub: kappa clamped low and suspect; header unknown and known."""
    tr = D.synth(frame.get(name), rho)
    assert dm.calibrate(tr) == demod.calibrate(tr)
    for est in ("joint", "plain"):
        _same_extract(demod.extract(None, tr, tr.spec, Q, est), dm.extract(None, tr, tr.spec, Q, est))


def test_no_estimate_keeps_kappa_one(dm):
    """Every symbol erased: neither estimate, kappa 1, not suspect, w 0."""
    tr = D.synth(frame.get("short"), 0.054)
    tr.psi = np.zeros_like(tr.psi)
    assert dm.calibrate(tr) == demod.calibrate(tr) == (1.0, False)
    a, b = demod.extract(None, tr, tr.spec, Q), dm.extract(None, tr, tr.spec, Q)
    _same_extract(a, b)
    assert not b[1].any()


def test_spread_rho_read_as_the_reference_reads_it(dm):
    tr = D.synth(frame.get("short"), 0.054)
    for rho in (None, 0, 0.0, -1.0, 0.03):
        tr.classes = SimpleNamespace(spread_rho=rho)
        a, b = demod.data_estimates(tr), dm.data_estimates(tr)
        _rel(b[0], a[0])
        assert b[2] == a[2]
    tr.classes = SimpleNamespace()                       # getattr's default
    assert dm.data_estimates(tr)[2] == demod.data_estimates(tr)[2] == 0.0
    del tr.classes                                       # tr.classes itself is strict
    with pytest.raises(AttributeError):
        demod.data_estimates(tr)
    with pytest.raises(AttributeError):
        dm.data_estimates(tr)


def _raises_alike(f_py, f_nat, exc):
    with pytest.raises(exc) as ea:
        f_py()
    with pytest.raises(exc) as eb:
        f_nat()
    assert str(eb.value) == str(ea.value)
    return str(ea.value)


def test_refusals(dm):
    l_tr = D.synth(lframe.get("l-short"))
    text = _raises_alike(lambda: demod.calibrate(l_tr), lambda: dm.calibrate(l_tr), AssertionError)
    assert "use ldemod.calibrate_l" in text
    _raises_alike(lambda: demod.extract(None, l_tr, l_tr.spec, Q),
                  lambda: dm.extract(None, l_tr, l_tr.spec, Q), AssertionError)
    y, s2 = np.full(10, 0.5), np.ones(10)
    for est in ("bad", None, 3):
        _raises_alike(lambda: demod.block_estimate(y, s2, Q, est),
                      lambda: dm.block_estimate(y, s2, Q, est), ValueError)
    _raises_alike(lambda: demod.block_estimate(y, s2[:9], Q),
                  lambda: dm.block_estimate(y, s2[:9], Q), ValueError)
    spec = frame.get("short")
    _raises_alike(lambda: demod.spread_llr(y, s2[:9], spec),
                  lambda: dm.spread_llr(y, s2[:9], spec), ValueError)
    short_grid = D.synth(frame.get("tiny"))
    short_grid.m = short_grid.m[:5]
    lay = frame.layout(short_grid.spec)
    _raises_alike(lambda: demod.symbol_estimates(short_grid, lay.data),
                  lambda: dm.symbol_estimates(short_grid, lay.data), IndexError)


def test_all_erased_block(dm):
    y, s2 = np.full(10, 0.5), np.full(10, np.inf)
    for est in ("joint", "plain"):
        a, b = demod.block_estimate(y, s2, Q, est), dm.block_estimate(y, s2, Q, est)
        for x, z in zip(a, b):
            assert np.array_equal(z, x)


# ---- adversarial probes (review stage) ----------------------------------------------
# Inputs the tracker never produces, to try to break the port: random stub
# tracks on every preset (lead-ins, extra trailing positions, exact-zero and
# denormal spectra), non-finite grids, block lengths around every pairwise
# and chunk boundary, and the argument forms the bindings must accept as
# the reference does. Measured bit-identical throughout (2026-10-13); the
# assertions are the bars.

CE_PRESETS = ("tiny", "short", "medium", "full", "short-v1", "full-spreadonly")


@pytest.mark.parametrize("name", CE_PRESETS)
def test_probe_random_tracks(dm, name):
    spec = frame.get(name)
    rng = np.random.default_rng(sum(map(ord, name)))
    for rho in (0.0, spec.hdr_rho):
        for nasty in (False, True):
            tr = P.random_track(spec, rng, lead=int(rng.integers(0, 300)), extra=int(rng.integers(0, 50)),
                                spread_rho=rho, nasty=nasty)
            ka, kb = P.outcome(lambda: demod.calibrate(tr)), P.outcome(lambda: dm.calibrate(tr))
            assert kb[1] is ka[1]
            P.same_kappa(kb[0], ka[0])
            for est in ("joint", "plain"):
                P.same_extract(P.outcome(lambda: demod.extract(None, tr, spec, 77, est)),
                               P.outcome(lambda: dm.extract(None, tr, spec, 77, est)))


@pytest.mark.parametrize("what", P.EDGES)
def test_probe_edge_tracks(dm, what):
    """NaN/inf in u, psi, P, N, m; no position at or after t0; all-zero u or psi."""
    spec = frame.get("short")
    tr = P.edge_track(spec, np.random.default_rng(len(what)), what)
    a = P.outcome(lambda: demod.extract(None, tr, spec, 3))
    b = P.outcome(lambda: dm.extract(None, tr, spec, 3))
    if not P.same_refusal(a, b):
        P.same_extract(a, b)
    a, b = P.outcome(lambda: demod.calibrate(tr)), P.outcome(lambda: dm.calibrate(tr))
    if not P.same_refusal(a, b):
        assert b[1] is a[1]
        P.same_kappa(b[0], a[0])
    a, b = P.outcome(lambda: demod.data_estimates(tr)), P.outcome(lambda: dm.data_estimates(tr))
    if not P.same_refusal(a, b):
        P.rel(b[0], a[0])
        P.rel(b[1], a[1])
        assert b[2] == a[2]


def test_probe_block_estimate_lengths(dm):
    """Every length to 69 and around 128/256/4096, a FULL pass's 50600; s2 with
    erasures, NaN, negative and 1e300 entries."""
    rng = np.random.default_rng(99)
    for n in list(range(70)) + [127, 128, 129, 255, 256, 257, 1000, 4095, 4096, 4097, 50600]:
        y = rng.standard_normal(n) * 3
        s2 = P.nasty_s2(rng, n)
        for est in ("joint", "plain"):
            a = P.outcome(lambda: demod.block_estimate(y, s2, 5, est))
            b = P.outcome(lambda: dm.block_estimate(y, s2, 5, est))
            if not P.same_refusal(a, b):
                P.f32(b[0], a[0], ("z", n, est))
                P.f32(b[1], a[1], ("v", n, est))


def test_probe_argument_forms(dm):
    rng = np.random.default_rng(3)
    spec = frame.get("short")
    y = rng.standard_normal(spec.n_data)
    s2 = rng.exponential(1, spec.n_data)
    lspec = lframe.get("l-short")
    yl, s2l = rng.standard_normal(lspec.n_data), rng.exponential(1, lspec.n_data)
    calls = [
        ("np.str_ estimator", lambda m: m.block_estimate(y, s2, 5, np.str_("plain"))),
        ("q np.uint64", lambda m: m.block_estimate(y, s2, np.uint64(2**63 + 5))),
        ("q 2**64-1", lambda m: m.block_estimate(y, s2, 2**64 - 1)),
        ("q -1", lambda m: m.block_estimate(y, s2, -1)),
        ("q 2**64", lambda m: m.block_estimate(y, s2, 2**64)),
        ("lists", lambda m: m.block_estimate(list(y[:64]), list(s2[:64]), 5)),
        ("empty", lambda m: m.block_estimate([], [], 5)),
        ("one", lambda m: m.block_estimate([0.3], [0.5], 5)),
        ("float32", lambda m: m.block_estimate(y.astype(np.float32), s2.astype(np.float32), 5)),
        ("spread_llr on an L spec", lambda m: m.spread_llr(yl, s2l, lspec)),
    ]
    tr = P.random_track(spec, rng, spread_rho=0.054)
    calls += [
        ("symbol_estimates list idx", lambda m: m.symbol_estimates(tr, [0, 5, -1, 300])),
        ("symbol_estimates empty idx", lambda m: m.symbol_estimates(tr, [])),
        ("symbol_estimates past the end", lambda m: m.symbol_estimates(tr, [spec.n_sym])),
        ("extract keywords", lambda m: m.extract(None, tr, spec, 5, estimator="plain", calibrate_kappa=0)),
    ]
    for label, call in calls:
        a, b = P.outcome(lambda: call(demod)), P.outcome(lambda: call(dm))
        if P.same_refusal(a, b):
            continue
        if isinstance(a, tuple) and len(a) == 4 and isinstance(a[3], dict):
            P.same_extract(a, b)
            continue
        for x, z in zip(a, b):
            assert np.asarray(z).dtype == np.asarray(x).dtype, label
            assert np.asarray(z).tobytes() == np.asarray(x).tobytes(), label


@pytest.mark.parametrize("name", ("tiny", "short"))
def test_probe_template_c_read_as_calibrate_reads_it(dm, name):
    """tr.c is read by calibrate alone, as `tr.c[g].imag`: absent is an
    AttributeError, None and a list TypeErrors (the port had read all three
    as an empty template and failed with an IndexError), a real or short
    array as numpy indexes it; extract without calibration never reads it."""
    spec = frame.get(name)
    for label in ("absent", "None", "list", "tuple", "dict", "short", "empty", "real"):
        tr = D.synth(spec, spec.hdr_rho)
        if label == "absent":
            del tr.c
        else:
            tr.c = {"None": None, "list": list(tr.c), "tuple": tuple(tr.c), "dict": {}, "short": tr.c[:100],
                    "empty": np.zeros(0, complex), "real": tr.c.real.copy()}[label]
        for call in (lambda m: m.calibrate(tr), lambda m: m.extract(None, tr, spec, 5),
                     lambda m: m.extract(None, tr, spec, 5, calibrate_kappa=False)):
            a, b = P.outcome(lambda: call(demod)), P.outcome(lambda: call(dm))
            if P.same_refusal(a, b):
                continue
            if isinstance(a, tuple) and len(a) == 2:
                assert b == a, label
            else:
                P.same_extract(a, b)

"""Parity of native/core/qrss/ldemod (sstvae_native.qrss.ldemod) with
sstvae/qrss/ldemod.py, on the Python tracker's own LTrackResults.

The tracks come from `receiver.receive_pass(..., return_tracks=True)` on
simulated L passes (tests/qrss_demod_tracks.py): l-tiny at -12 dB and its
genie track, and l-short at -6 dB in both rounds -- round A with the
header unknown (the spread copy as extra variance), round B with it known
(hdr_bits subtracted) and the tracker's measured-reference mask -- plus
the ctests' stub with and without `meas` and `classes.hdr_bits`.

The contract, and what was measured (2026-10-13, these tracks):

- extract_l: z, w, llr within one ulp and >= 99.9% bit-identical;
  measured bit-identical on every track, both estimators, calibrated and
  not;
- symbol_estimates_l, data_estimates_l, header_estimates_l and extract_l's
  diag: 1e-12 relative; measured 0;
- kappa_estimates_l: the same estimates in the same order, each to 1e-12;
  calibrate_l: the same branch and kappa to 1e-12; measured equal to the
  bit;
- the refusals (a CE frame, header bits of the wrong length) with the
  reference's texts.
"""

import copy
from types import SimpleNamespace

import numpy as np
import pytest

import qrss_demod_probes as P
import qrss_demod_tracks as D
from qrss_native import require
from sstvae.qrss import frame, ldemod, lframe

Q = 0x1234_5678_9ABC
L_LABELS = ("l-tiny/A", "l-tiny/genie", "l-short/A", "l-short/B")


@pytest.fixture(scope="module")
def ld(native):
    return require(native, "ldemod")


@pytest.fixture(scope="module")
def tracks(ld):
    return D.l_tracks()


def _f32(a, b, what=""):
    u, frac = D.ulps(a, b)
    assert u <= 1 and frac >= 0.999, (what, u, frac)


def _rel(a, b, what="", tol=1e-12):
    r = D.rel(a, b)
    assert r <= tol, (what, r)


def test_constants(ld):
    for name in ("G_REF_L", "G_HDR_L", "G_DATA_L", "G_PRE_L", "D_PASS_L", "D_REF_L", "D_HDR_L",
                 "D_PRE_L", "D_PASS_LX"):
        assert getattr(ld, name) == getattr(ldemod, name), name
    for name in lframe.PRESETS:
        spec = lframe.get(name)
        assert ld.d_pass_x(spec) == ldemod.d_pass_x(spec), name


def test_the_tracks_cover_what_they_claim(tracks):
    assert set(tracks) == set(L_LABELS)
    a, b = tracks["l-short/A"], tracks["l-short/B"]
    assert getattr(a.classes, "hdr_bits", None) is None and a.classes.spread_rho > 0
    assert b.classes.hdr_bits is not None and b.meas is not None and b.meas.sum() > 1000
    assert ldemod.data_estimates_l(a)[2] > 0 and ldemod.data_estimates_l(b)[2] == 0


@pytest.mark.parametrize("label", L_LABELS)
def test_estimates(ld, tracks, label):
    tr = tracks[label]
    lay = lframe.layout(tr.spec)
    for which, gain in (("data", ldemod.G_DATA_L), ("ref", ldemod.G_REF_L),
                        ("hdr_sym", ldemod.G_HDR_L), ("known_idx", 0.5)):
        idx = np.asarray(getattr(lay, which))
        if not len(idx):
            continue
        a = ldemod.symbol_estimates_l(tr, idx, gain)
        b = ld.symbol_estimates_l(tr, idx, gain)
        assert b[0].dtype == np.complex128
        _rel(b[0], a[0], f"{which} r")
        _rel(b[1], a[1], f"{which} s2n")
    a, b = ldemod.data_estimates_l(tr), ld.data_estimates_l(tr)
    _rel(b[0], a[0], "y")
    _rel(b[1], a[1], "s2n")
    assert b[2] == a[2]
    if tr.spec.n_hdr:
        a, b = ldemod.header_estimates_l(tr), ld.header_estimates_l(tr)
        _rel(b[0], a[0], "y_h")
        _rel(b[1], a[1], "s2n_h")


@pytest.mark.parametrize("label", L_LABELS)
def test_kappa_and_calibrate(ld, tracks, label):
    tr = tracks[label]
    ea, eb = ldemod.kappa_estimates_l(tr), ld.kappa_estimates_l(tr)
    assert len(eb) == len(ea) > 0
    for (k1, v1), (k2, v2) in zip(ea, eb):
        assert abs(k2 - k1) <= 1e-12 * abs(k1) and abs(v2 - v1) <= 1e-12 * abs(v1)
    ka, sa = ldemod.calibrate_l(tr)
    kb, sb = ld.calibrate_l(tr)
    assert sb is sa and abs(kb - ka) <= 1e-12 * abs(ka)
    assert ld._combine(ea) == ldemod._combine(ea)


def test_combine(ld):
    cases = [[], [(1.1, 0.01), (0.9, np.float64(0.02))], [(1.3, 0.004)], [(3.0, 0.01), (2.5, 0.5)],
             [(0.2, 0.001)], [(1.0, 1e-9), (5.0, 1e9)], ((0.7, 0.03), (1.4, 0.05))]
    for ests in cases:
        assert ld._combine(ests) == ldemod._combine(ests), ests


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


@pytest.mark.parametrize("label", L_LABELS)
@pytest.mark.parametrize("est", ("joint", "plain"))
@pytest.mark.parametrize("cal", (True, False))
def test_extract_l(ld, tracks, label, est, cal):
    tr = tracks[label]
    _same_extract(ldemod.extract_l(None, tr, tr.spec, Q, est, calibrate_kappa=cal),
                  ld.extract_l(None, tr, tr.spec, Q, est, calibrate_kappa=cal))


def test_extract_l_with_given_m_replaces_the_tracks(ld, tracks):
    tr = tracks["l-short/B"]
    m = tr.m * (1.25 + 0.5j)
    ta, tb = copy.copy(tr), copy.copy(tr)
    _same_extract(ldemod.extract_l(m, ta, tr.spec, Q), ld.extract_l(m, tb, tr.spec, Q))
    assert np.array_equal(tb.m, ta.m) and np.array_equal(tb.m, m)
    ta = copy.copy(tr)
    own = ta.m
    ld.extract_l(own, ta, tr.spec, Q)
    assert ta.m is own


@pytest.mark.parametrize("known,meas", [(False, False), (True, True), (False, True), (True, False)])
def test_stub_tracks(ld, known, meas):
    spec = lframe.get("l-short")
    tr = D.synth(spec, 0.054, D.known_bits() if known else None, meas)
    assert ld.kappa_estimates_l(tr) == ldemod.kappa_estimates_l(tr)
    assert ld.calibrate_l(tr) == ldemod.calibrate_l(tr)
    for est in ("joint", "plain"):
        _same_extract(ldemod.extract_l(None, tr, spec, Q, est), ld.extract_l(None, tr, spec, Q, est))


def test_a_genie_shaped_track(ld):
    """Only the attributes a GenieTrack has: no meas, classes without hdr_bits."""
    spec = lframe.get("l-short")
    tr = D.synth(spec)
    del tr.meas
    tr.classes = SimpleNamespace()
    assert ld.kappa_estimates_l(tr) == ldemod.kappa_estimates_l(tr)
    _same_extract(ldemod.extract_l(None, tr, spec, Q), ld.extract_l(None, tr, spec, Q))
    del tr.classes                                        # getattr(tr, "classes", None)
    _same_extract(ldemod.extract_l(None, tr, spec, Q), ld.extract_l(None, tr, spec, Q))


def test_l_tiny_stub(ld):
    spec = lframe.get("l-tiny")
    tr = D.synth(spec)
    for est in ("joint", "plain"):
        _same_extract(ldemod.extract_l(None, tr, spec, Q, est), ld.extract_l(None, tr, spec, Q, est))


def _raises_alike(f_py, f_nat, exc):
    with pytest.raises(exc) as ea:
        f_py()
    with pytest.raises(exc) as eb:
        f_nat()
    assert str(eb.value) == str(ea.value)
    return str(ea.value)


def test_refusals(ld):
    ce = D.synth(frame.get("short"))
    text = _raises_alike(lambda: ldemod.calibrate_l(ce), lambda: ld.calibrate_l(ce), AssertionError)
    assert text.startswith("ldemod.calibrate_l is L's")
    _raises_alike(lambda: ldemod.kappa_estimates_l(ce), lambda: ld.kappa_estimates_l(ce), AssertionError)
    _raises_alike(lambda: ldemod.extract_l(None, ce, ce.spec, Q),
                  lambda: ld.extract_l(None, ce, ce.spec, Q), AssertionError)
    tr = D.synth(lframe.get("l-short"), 0.054, D.known_bits()[:5])
    _raises_alike(lambda: ldemod.data_estimates_l(tr), lambda: ld.data_estimates_l(tr), ValueError)
    tr = D.synth(lframe.get("l-short"))
    _raises_alike(lambda: ldemod.extract_l(None, tr, tr.spec, Q, "bad"),
                  lambda: ld.extract_l(None, tr, tr.spec, Q, "bad"), ValueError)


# ---- adversarial probes (review stage) ----------------------------------------------
# What the tracker never produces: random stub tracks on every L preset
# (lead-ins, extra trailing positions, exact-zero and denormal spectra),
# the header known and unknown, the measured-reference mask on and off;
# non-finite grids; and the forms of classes.hdr_bits and tr.meas the
# bindings must read as the reference does. Measured bit-identical
# (2026-10-13); the assertions are the bars.


def _same_kappa_estimates(eb, ea):
    assert len(eb) == len(ea)
    for (k1, v1), (k2, v2) in zip(ea, eb):
        P.same_kappa(k2, k1)
        P.same_kappa(v2, v1)


@pytest.mark.parametrize("name", lframe.PRESETS)
def test_probe_random_tracks(ld, name):
    spec = lframe.get(name)
    rng = np.random.default_rng(sum(map(ord, name)))
    for known in (False, True):
        for nasty, meas in ((False, True), (True, False), (True, True)):
            bits = (rng.random(2474) < 0.5).astype(np.uint8) if known else None
            tr = P.random_track(spec, rng, lead=int(rng.integers(0, 300)), extra=int(rng.integers(0, 50)),
                                spread_rho=0.0 if known else spec.hdr_rho, hdr_bits=bits, meas=meas,
                                nasty=nasty)
            _same_kappa_estimates(P.outcome(lambda: ld.kappa_estimates_l(tr)),
                                  P.outcome(lambda: ldemod.kappa_estimates_l(tr)))
            for est in ("joint", "plain"):
                P.same_extract(P.outcome(lambda: ldemod.extract_l(None, tr, spec, 99, est)),
                               P.outcome(lambda: ld.extract_l(None, tr, spec, 99, est)))


@pytest.mark.parametrize("what", P.EDGES)
def test_probe_edge_tracks(ld, what):
    spec = lframe.get("l-short")
    tr = P.edge_track(spec, np.random.default_rng(len(what)), what)
    a = P.outcome(lambda: ldemod.extract_l(None, tr, spec, 3))
    b = P.outcome(lambda: ld.extract_l(None, tr, spec, 3))
    if not P.same_refusal(a, b):
        P.same_extract(a, b)
    a, b = P.outcome(lambda: ldemod.calibrate_l(tr)), P.outcome(lambda: ld.calibrate_l(tr))
    if not P.same_refusal(a, b):
        assert b[1] is a[1]
        P.same_kappa(b[0], a[0])
    a, b = P.outcome(lambda: ldemod.kappa_estimates_l(tr)), P.outcome(lambda: ld.kappa_estimates_l(tr))
    if not P.same_refusal(a, b):
        _same_kappa_estimates(b, a)


def _hdr_forms():
    kb = D.known_bits()
    return {"uint8": kb, "list": list(map(int, kb)), "bool": kb.astype(bool), "float": kb.astype(float),
            "int64": kb.astype(np.int64), "tuple": tuple(kb), "2-D": kb.reshape(2, -1), "short": kb[:10],
            "str": "0" * 2474, "scalar": 1}


@pytest.mark.parametrize("name", ("l-short", "l-tiny"))
def test_probe_hdr_bits_forms(ld, name):
    """Any dtype of 2474 values 0/1 is taken exactly; another shape is refused
    where the reference reads the bits (with its text), and ignored where it
    does not (l-tiny has no spread copy). A 2-D array of 2474 bits used to
    be flattened and accepted."""
    spec = lframe.get(name)
    for label, bits in _hdr_forms().items():
        tr = D.synth(spec, spec.hdr_rho, bits, True)
        for fn in ("data_estimates_l", "kappa_estimates_l", "calibrate_l"):
            a = P.outcome(lambda: getattr(ldemod, fn)(tr))
            b = P.outcome(lambda: getattr(ld, fn)(tr))
            if P.same_refusal(a, b):
                continue
            if fn == "data_estimates_l":
                P.rel(b[0], a[0], label)
                P.rel(b[1], a[1], label)
                assert b[2] == a[2], label
            else:
                assert b == a, (label, fn)
        a = P.outcome(lambda: ldemod.extract_l(None, tr, spec, 5))
        b = P.outcome(lambda: ld.extract_l(None, tr, spec, 5))
        if not P.same_refusal(a, b):
            P.same_extract(a, b)


def test_probe_hdr_bits_that_are_not_0_1_are_refused(ld):
    """The documented deviation: the reference computes 1 - 2b from any value;
    the port refuses, with the reference's own text, rather than give a
    different number (a uint8 cast had read -1 as 255 and 0.5 as 0)."""
    spec = lframe.get("l-short")
    kb = D.known_bits()
    for bits in (np.where(kb == 1, 2, 0), -kb.astype(np.int8), kb * 0.5, np.where(kb == 1, np.nan, 0.0)):
        tr = D.synth(spec, spec.hdr_rho, bits, True)
        ldemod.data_estimates_l(tr)                     # the reference computes ...
        with pytest.raises(ValueError, match="hdr_bits must be 2474 values of 0/1"):
            ld.data_estimates_l(tr)                     # ... the port refuses


def test_probe_meas_forms(ld):
    """meas read lazily by np.asarray(meas, bool)[gi]: None reads as unmeasured,
    an empty mask (not None) and a short one fail their first index, a long
    one, ints and floats are read as numpy reads them."""
    spec = lframe.get("l-short")
    base = D.synth(spec, spec.hdr_rho, D.known_bits(), True)
    m = np.asarray(base.meas)
    n = len(base.pos)
    for label, meas in (("None", None), ("short", m[: n // 2]), ("long", np.concatenate([m, m])),
                        ("ints", m.astype(int) * 3), ("floats", m * 0.5), ("list", list(m)),
                        ("empty", np.zeros(0, bool))):
        tr = copy.copy(base)
        tr.meas = meas
        for fn in ("kappa_estimates_l", "calibrate_l"):
            a = P.outcome(lambda: getattr(ldemod, fn)(tr))
            b = P.outcome(lambda: getattr(ld, fn)(tr))
            if not P.same_refusal(a, b):
                assert b == a, (label, fn)
    for meas in (np.stack([m, m]), True):               # other shapes: refused, as by numpy
        tr = copy.copy(base)
        tr.meas = meas
        with pytest.raises(IndexError):
            ldemod.calibrate_l(tr)
        with pytest.raises(IndexError):
            ld.calibrate_l(tr)


def test_probe_argument_forms(ld):
    spec = lframe.get("l-short")
    rng = np.random.default_rng(8)
    tr = P.random_track(spec, rng, spread_rho=0.054)
    lay = lframe.layout(spec)
    for gain in (np.float32(0.881), 1, True, np.float64(0.5)):
        a, b = ldemod.symbol_estimates_l(tr, lay.ref, gain), ld.symbol_estimates_l(tr, lay.ref, gain)
        assert b[0].tobytes() == a[0].tobytes() and b[1].tobytes() == a[1].tobytes()
    for idx in ([0, 5, -1, 300], [], np.array([spec.n_sym])):
        a = P.outcome(lambda: ldemod.symbol_estimates_l(tr, idx, 0.5))
        b = P.outcome(lambda: ld.symbol_estimates_l(tr, idx, 0.5))
        if not P.same_refusal(a, b):
            assert b[0].tobytes() == a[0].tobytes() and b[1].tobytes() == a[1].tobytes()
    P.same_extract(ldemod.extract_l(None, tr, spec, 5, estimator=np.str_("plain"), calibrate_kappa=0),
                   ld.extract_l(None, tr, spec, 5, estimator=np.str_("plain"), calibrate_kappa=0))

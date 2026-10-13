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

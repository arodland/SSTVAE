"""Adversarial inputs for the demod lane's parity tests (the review stage).

Not a test module (no `test_` prefix). tests/qrss_demod_tracks.py holds the
tracks the receiver really produces; this holds what it does not -- random
stub tracks on every preset with a lead-in, extra trailing positions and
"nasty" spectra (exact zeros, denormal |u|^2, tiny gains), the
measured-reference mask on or off, and non-finite or degenerate grids --
plus the comparisons the parity bars need when NaN and inf are legitimate
outputs on both sides.

The stubs are GenieTrack-shaped (attributes only) with gi = p - pos[0],
which is what the bindings read (demod_rx.hpp).
"""

from __future__ import annotations

import warnings
from types import SimpleNamespace

import numpy as np

import qrss_demod_tracks as D

# Non-finite or degenerate grids the tracker never hands over.
EDGES = ("u-nan", "psi-nan", "P-nan", "N-inf", "m-inf", "u-inf", "all-neg", "u-allzero", "psi-allzero")


def random_track(spec, rng, *, lead=0, extra=0, spread_rho=0.0, hdr_bits=None, meas=True, nasty=False):
    """A random track on `spec`'s grid, starting `lead` positions before -8."""
    p0 = -8 - lead
    n = spec.n_pos - p0 + extra
    pos = np.arange(p0, p0 + n, dtype=np.int64)
    u = (rng.standard_normal(n) + 1j * rng.standard_normal(n)) * rng.uniform(0.01, 2, n)
    if nasty:
        u[rng.integers(0, n, n // 20)] = 0.0
        u[rng.integers(0, n, n // 20)] *= 1e-160      # |u|^2 denormal or zero
        u[rng.integers(0, n, n // 50)] *= 1e-3
    ns = SimpleNamespace(
        spec=spec, pos=pos, u=u,
        m=rng.standard_normal(n) + 1j * rng.standard_normal(n),
        psi=np.clip(rng.uniform(-0.2, 1.3, n), 0, 1),     # exact 0s and 1s too
        P=rng.exponential(0.01, n), N=rng.exponential(0.5, n),
        c=rng.standard_normal(n) + 1j * rng.standard_normal(n),
        classes=SimpleNamespace(spread_rho=spread_rho, hdr_bits=hdr_bits),
        meas=(rng.random(n) < 0.4) if meas else None)
    ns.gi = lambda p, p0=p0: np.asarray(p, dtype=np.int64) - p0
    return ns


def edge_track(spec, rng, what):
    """random_track with one of EDGES applied."""
    tr = random_track(spec, rng, spread_rho=spec.hdr_rho)
    k = rng.integers(0, len(tr.pos), 30)
    if what == "u-nan":
        tr.u[k] = np.nan
    elif what == "psi-nan":
        tr.psi[k] = np.nan
    elif what == "P-nan":
        tr.P[k] = np.nan
    elif what == "N-inf":
        tr.N[k] = np.inf
    elif what == "m-inf":
        tr.m[k] = np.inf
    elif what == "u-inf":
        tr.u[k] = np.inf + 1j
    elif what == "all-neg":                       # no position at or after t0
        tr.pos = tr.pos - (spec.n_pos + 100)
        tr.gi = lambda p, p0=int(tr.pos[0]): np.asarray(p, dtype=np.int64) - p0
    elif what == "u-allzero":
        tr.u[:] = 0
    elif what == "psi-allzero":
        tr.psi[:] = 0
    else:
        raise ValueError(what)
    return tr


def outcome(f):
    """f()'s value, or the exception it raised (warnings silenced)."""
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        try:
            return f()
        except Exception as e:  # noqa: BLE001 -- compared, not swallowed
            return e


def same_refusal(a, b):
    """Both exceptions of one type with one text, or neither an exception."""
    if isinstance(a, Exception) or isinstance(b, Exception):
        assert type(b) is type(a) and str(b) == str(a), (repr(a), repr(b))
        return True
    return False


def f32(b, a, what=""):
    """The float32 bar: one ulp, >= 99.9% bit-identical (NaN/inf identical)."""
    u, frac = D.ulps(b, a)
    assert u <= 1 and frac >= 0.999, (what, u, frac)


def rel(b, a, what="", tol=1e-12):
    r = D.rel(b, a)
    assert r <= tol, (what, r)


def same_kappa(kb, ka):
    assert (np.isnan(ka) and np.isnan(kb)) or abs(kb - ka) <= 1e-12 * abs(ka), (ka, kb)


def same_extract(a, b):
    """extract / extract_l: the float32 bar on z, w, llr; diag to 1e-12."""
    for k, part in enumerate(("z", "w", "llr")):
        assert b[k].dtype == np.float32 and b[k].shape == a[k].shape, part
        f32(b[k], a[k], part)
    da, db = a[3], b[3]
    assert set(db) == set(da)
    same_kappa(db["kappa"], da["kappa"])
    assert db["suspect"] is da["suspect"] and db["estimator"] == da["estimator"]
    for key in ("y", "s2", "y_hdr"):
        rel(db[key], da[key], key)
    for key in ("llr_block", "llr_spread"):
        assert (db[key] is None) == (da[key] is None), key
        if da[key] is not None:
            rel(db[key], da[key], key)


def nasty_s2(rng, n):
    """Variances with erasures (inf), NaN, negative and huge entries."""
    s2 = rng.exponential(1.0, n)
    s2[rng.random(n) < 0.1] = np.inf
    s2[rng.random(n) < 0.02] = np.nan
    s2[rng.random(n) < 0.02] = -1.0
    s2[rng.random(n) < 0.02] = 1e300
    return s2


# ---- the matched filters' argument forms (ce.matched_filter, lwave.matched_filter) ----

def mf_cases(seed=1):
    """(label, args) for matched_filter(z, fs, n0, pos[, chunk]): the forms of
    n0 (whose float the reference refuses as an index, never truncates),
    pos (0-d, 2-D), z (0-d, 2-D, empty, odd dtypes) and chunk, each with
    and without positions -- the reference reads n0 and z only when its
    loop body runs."""
    from fractions import Fraction
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(500)
    zc = z + 1j * z[::-1]
    cases = []
    for n0 in (3, -7, True, np.int8(3), np.uint64(3), np.array(3), 3.0, np.float64(3.0), 1.5, None, "3",
               2**70, Fraction(3), 1j):
        for pos in ([0, 1, 2], []):
            cases.append((f"n0={n0!r} pos={pos}", (zc, 250, n0, pos)))
    cases += [
        ("scalar pos", (z, 250, 0, 5)),
        ("2-D pos, empty", (z, 250, 0, np.zeros((0, 3)))),
        ("0-d z", (np.float64(1.0), 250, 0, [1])),
        ("0-d z, no positions", (np.float64(1.0), 250, 0, [])),
        ("2-D z, no positions", (z.reshape(2, 250), 250, 0, [])),
        ("empty z", (z[:0], 250, 0, [1])),
        ("z of one sample", (z[:1], 250, 0, [0, 1, 2])),
        ("bool z", (np.ones(500, bool), 250, 0, [0, 1, 2])),
        ("int32 z", (np.arange(500, dtype=np.int32), 250, 0, [0, 1, 2])),
        ("float16 z", (np.arange(500, dtype=np.float16), 250, 0, [0, 1, 2])),
        ("z with NaN at the clipped edge", (np.where(np.arange(500) == 0, np.nan, 1.0), 250, 0, [-3, 0, 1, 300])),
        ("z with inf, complex", (np.where(np.arange(500) == 499, np.inf, 1.0) + 0j, 250, 0, [-3, 0, 300, 400])),
        ("float pos", (z, 250, 0, np.array([0.7, 1.2, 5.9]))),
        ("list z", (list(z), 250, 0, [0, 1, 2])),
        ("fs np.int64", (z, np.int64(250), 0, [0, 1, 2])),
        ("fs float", (z, 250.0, 0, [0, 1, 2])),
        ("fs 0", (z, 0, 0, [0, 1, 2])),
        ("fs -250", (z, -250, 0, [0, 1, 2])),
        ("fs refused before a 0-d pos", (z, 7, 0, 5)),
        ("chunk 0, no positions", (z, 250, 0, [], 0)),
        ("chunk -1", (zc, 250, 0, [1, 2], -1)),
        ("chunk float", (z, 250, 0, [1], 2.0)),
        ("chunk np.int64", (z, 250, 0, [1, 2, 3], np.int64(2))),
        ("chunk huge", (z, 250, 0, np.arange(-20, 200), 10**6)),
    ]
    return cases


def check_mf_case(f_py, f_nat, args):
    a, b = outcome(lambda: f_py(*args)), outcome(lambda: f_nat(*args))
    if not same_refusal(a, b):
        assert b.dtype == a.dtype and b.shape == a.shape and b.tobytes() == a.tobytes()

"""Parity of native/core/qrss/acquire_a12 (sstvae_native.qrss.acquire_a12)
with sstvae/qrss/acquire.py's A1, A2 and lead-in, held side by side in one
process:

- A1 bit for bit: the CarrierStats statistic (float32), its bins, drifts
  and M; carrier_lines' candidates field for field and type for type (an
  np.float64 frequency where the reference's arithmetic makes one), in the
  same order; hint_line, _cell_line and pilot_lines likewise; numpy's own
  float32 log, which the parabola there runs on;
- the matched and notch filters and both preamble kinds to the bit;
- A2 to rounding, with the measured worst errors pinned: the channel
  (_a2_input, _a2_mf), the Lambda grid (rtol 1e-9, the same winning cell),
  the chunk sums, the refined hit (f_hz within 1e-7 Hz, tau0_s within
  1e-6 s, the same pass/fail and the same NaN timing sd), and the p-value
  the C++ takes from its own arrays exactly as numpy computes it on them;
- lead_in_span's span, and its noise_psd seam called as the reference
  calls it;
- the reference's errors, type and text;
- the whole callers on the C++: acquire() (A1, A2 of both kinds, hints,
  lead-in, A3) and receiver.receive_pass on a CE TINY pass, with this
  module's and acquire_stats' rows assigned over acquire's names.

tests/qrss_native/acquire_a12.py substitutes these under --native-qrss=rx.
"""

import dataclasses
import math

import numpy as np
import pytest

import test_qrss_acquire as TA
import test_qrss_lacquire as TL
from qrss_native import _owner, require
from sstvae.qrss import acquire as A
from sstvae.qrss import frame, frontend, lframe
from sstvae.qrss.constants import CH_FS, FE_FS, WAVEFORM_CE, WAVEFORM_L
from sstvae.qrss.types import CarrierCandidate


@pytest.fixture(scope="module")
def N(native):
    return require(native, "acquire_a12")


@pytest.fixture(scope="module")
def caps():
    """CE SHORT and L SHORT at -15 dB, and noise: (fe, t0_index)."""
    x, t0 = TA._capture(1650.0, -15.0, shift_s=0.7, seed=3)
    xl, t0l = TL._capture(TL._l_clean("l-short"), 1203.4, -15.0, shift_s=-0.4, seed=5)
    noise = TA._noise_fe(len(x), np.random.default_rng(16))
    return {"ce": (x, t0), "l": (xl, t0l), "noise": (noise, t0)}


def _same(a, b, what=""):
    a, b = np.asarray(a), np.asarray(b)
    assert a.dtype == b.dtype, (what, a.dtype, b.dtype)
    assert a.shape == b.shape, (what, a.shape, b.shape)
    if a.dtype.kind in "fc":
        an, bn = np.isnan(a), np.isnan(b)
        assert np.array_equal(an, bn), what
        assert a[~an].tobytes() == b[~bn].tobytes(), (what, np.count_nonzero(a[~an] != b[~bn]))
    else:
        assert np.array_equal(a, b), what


def _close(r, g, rtol, what=""):
    r, g = np.asarray(r), np.asarray(g)
    assert r.shape == g.shape and r.dtype == g.dtype, (what, r.shape, g.shape, r.dtype, g.dtype)
    rn = np.isnan(r)
    assert np.array_equal(rn, np.isnan(g)), what
    err = np.abs(g[~rn] - r[~rn])
    scale = np.abs(r[~rn])
    worst = float(np.max(err / np.where(scale > 0, scale, 1.0))) if err.size else 0.0
    assert worst <= rtol, (what, worst)
    return worst


def _close_max(r, g, tol, what=""):
    """max |g - r| / max |r|: for arrays whose small entries are rounding
    residue (a notch's output, a sum of noise), where an elementwise
    relative error means nothing."""
    r, g = np.asarray(r), np.asarray(g)
    assert r.shape == g.shape and r.dtype == g.dtype, (what, r.shape, g.shape, r.dtype, g.dtype)
    if not r.size:
        return 0.0
    scale = float(np.max(np.abs(r)))
    worst = float(np.max(np.abs(g - r))) / (scale if scale > 0 else 1.0)
    assert worst <= tol, (what, worst)
    return worst


def _cand_same(r, g, what=""):
    assert type(g) is type(r) is CarrierCandidate, (what, type(r), type(g))
    for f in dataclasses.fields(r):
        a, b = getattr(r, f.name), getattr(g, f.name)
        assert type(b) is type(a), (what, f.name, type(a), type(b))
        assert a == b or (a != a and b != b), (what, f.name, a, b)


def _cands_same(r, g, what=""):
    assert len(r) == len(g), (what, r, g)
    for i, (a, b) in enumerate(zip(r, g)):
        _cand_same(a, b, f"{what}[{i}]")


def _outcome(fn, *args, **kw):
    try:
        out = fn(*args, **kw)
    except (ValueError, TypeError, IndexError, ZeroDivisionError, OverflowError) as e:
        return type(e), str(e)
    return "ok", out


def _stats(d):
    return A.CarrierStats(**{k: v for k, v in d.items() if k != "__type__"})


def _hit(d):
    return A.PreambleHit(**{k: v for k, v in d.items() if k != "__type__"})


def _cand(d):
    return CarrierCandidate(**{k: v for k, v in d.items() if k != "__type__"})


# --- tables -------------------------------------------------------------------------------------

def test_filters_and_kinds(N):
    _same(N._notch_taps(), A._notch_taps(), "notch")
    for fs in (CH_FS, 125, 500, 1000.0, 333.3):
        _same(N._mf_taps(fs), A._mf_taps(fs), f"mf {fs}")
        _same(N._l_mf_taps(fs), A._l_mf_taps(fs), f"l_mf {fs}")
    _same(N._mf_taps(), A._mf_taps(), "mf default")
    _same(N._l_mf_taps(), A._l_mf_taps(), "l_mf default")
    for nk, rk in ((N.CE_PRE, A.CE_PRE), (N.L_PRE, A.L_PRE)):
        _same(nk.seq, rk.seq, f"seq {rk.name}")
        _same(nk.mf_taps, rk.mf_taps, f"mf_taps {rk.name}")
        assert not nk.seq.flags.writeable and not nk.mf_taps.flags.writeable
        for f in ("waveform", "t_sym", "chunk", "data_offset_hz", "phase_offset", "n_chunks", "name"):
            a, b = getattr(rk, f), getattr(nk, f)
            assert type(a) is type(b) and a == b, (rk.name, f, a, b)
    assert list(N.PREAMBLE_KINDS) == list(A.PREAMBLE_KINDS) == [WAVEFORM_CE, WAVEFORM_L]
    assert N.PREAMBLE_KINDS[WAVEFORM_CE] is N.CE_PRE and N.PREAMBLE_KINDS[WAVEFORM_L] is N.L_PRE
    assert N.LEAD_WIN_S == A.LEAD_WIN_S and N.LEAD_Z == A.LEAD_Z
    for kind in (A.CE_PRE, A.L_PRE, N.CE_PRE, N.L_PRE):
        rk = A.CE_PRE if kind.waveform == WAVEFORM_CE else A.L_PRE
        _same(N._chunks(kind), A._chunks(rk), f"chunks {rk.name}")
        _same(N._norm(A._chunks(rk)), A._norm(A._chunks(rk)), f"norm {rk.name}")
        for df in (A.A2_DF_HZ, (0.0,), (-1.3, 0.25, 7.0)):
            for got, ref in zip(N._templates(df, kind), A._templates(df, rk)):
                _same(got, ref, f"templates {rk.name} {df}")
    for got, ref in zip(N._templates(A.A2_DF_HZ), A._templates(A.A2_DF_HZ)):
        _same(got, ref, "templates default kind")


def test_numpys_float32_log(N):
    """numpy computes a float32 log with its own SIMD kernel wherever the
    CPU has AVX2+FMA3 or AVX-512F (a scalar included), not libm's logf;
    acquire's vertex parabola runs on it."""
    feats = np._core._multiarray_umath.__cpu_features__
    if not (feats.get("AVX512F") or (feats.get("AVX2") and feats.get("FMA3"))):
        pytest.skip("numpy runs libm's logf on this CPU")
    rng = np.random.default_rng(3)
    bits = rng.integers(0, 2**32, 1 << 21, dtype=np.uint64).astype(np.uint32)
    x = bits.view(np.float32)
    x = np.concatenate([x, np.abs(x), np.array([0.0, -0.0, np.inf, -np.inf, np.nan, 1.0, 1e-45, 1e-38,
                                                3.4e38, 0.70710677, 0.7071068, 1e-9],
                                               dtype=np.float32),
                        np.linspace(0.5, 2.0, 1 << 18, dtype=np.float32),
                        rng.gamma(11, 1.0, 1 << 18).astype(np.float32)])
    with np.errstate(all="ignore"):
        ref = np.log(x)
    got = N._log_f32(x)
    assert got.dtype == np.float32
    same = (got.view(np.uint32) == ref.view(np.uint32)) | (np.isnan(got) & np.isnan(ref))
    assert same.all(), (np.count_nonzero(~same), x[~same][:5])
    # the reference's call: one float32 at a time
    for v in (np.float32(0.3), np.float32(17.25), np.float32(1e-9)):
        assert N._log_f32(np.array([v]))[0] == np.log(v)


# --- A1 ------------------------------------------------------------------------------------------

def _a1_cases(caps):
    x, t0 = caps["ce"]
    xl, t0l = caps["l"]
    noise, _ = caps["noise"]
    yield "ce", x, t0, {}
    yield "l", xl, t0l, {}
    yield "noise", noise, t0, {}
    yield "ce c128", x.astype(np.complex128), t0, {}
    yield "ce early", x, -50.3 * FE_FS, {}
    yield "ce late", x, len(x) + 3.7, {}
    yield "ce frac", x, t0 + 0.37, {}
    yield "ce span", x, t0, {"span_s": (-1.0, 10.0)}
    yield "ce drifts", x, t0, {"drifts": (0.0, 1.0, -1.0, 2.5)}
    yield "ce one drift", x, t0, {"drifts": (0.25,)}
    yield "ce band", x, t0, {"band_hz": (1000.0, 2000.0)}
    yield "ce band edge", x, t0, {"band_hz": (1650.0, 1652.0)}


def test_carrier_line_stats(N, caps):
    for what, fe, t0, kw in _a1_cases(caps):
        ref = A.carrier_line_stats(fe, t0, **kw)
        got = _stats(N.carrier_line_stats(fe, t0, **kw))
        _same(got.stat, ref.stat, f"stat {what}")      # 0 flips measured (the bar was a few 1-ulp)
        _same(got.f_hz, ref.f_hz, f"f_hz {what}")
        _same(got.drifts, ref.drifts, f"drifts {what}")
        assert type(got.M) is type(ref.M) is int and got.M == ref.M, what
    x, t0 = caps["ce"]
    ref, got = A.carrier_line_stats(x[:4 * FE_FS], 2.0 * FE_FS, FE_FS), N.carrier_line_stats(x[:4 * FE_FS], 2.0 * FE_FS)
    _same(got["stat"], ref.stat, "short capture")


def test_carrier_lines(N, caps):
    for what, fe, t0, kw in _a1_cases(caps):
        stats_kw = {k: v for k, v in kw.items() if k in ("band_hz",)}
        ref = A.carrier_lines(fe, t0, **stats_kw)
        got = [_cand(d) for d in N.carrier_lines(fe, t0, **stats_kw)]
        _cands_same(ref, got, what)
        cs = A.carrier_line_stats(fe, t0, **kw)
        for opts in ({}, {"pfa": 1e-2}, {"max_cands": 3}, {"min_sep_hz": 0.5}, {"max_cands": 0},
                     {"pfa": 0.9, "max_cands": 64, "min_sep_hz": 2.0}):
            ref = A.carrier_lines(fe, t0, stats=cs, **opts)
            got = [_cand(d) for d in N.carrier_lines(fe, t0, stats=cs, **opts)]
            _cands_same(ref, got, f"{what} {opts}")


def test_hint_cell_and_pilot_lines(N, caps):
    for what in ("ce", "l", "noise"):
        fe, t0 = caps[what]
        cs = A.carrier_line_stats(fe, t0)
        lines = A.carrier_lines(fe, t0, stats=cs)
        for f in (1650.0, 1651.5, 1203.4, 1216.0, 1190.0, 300.0, 2700.0, 150.0, 5000.0, float("nan"),
                  1649.9999, 2000.0):
            for kw in ({}, {"pfa": 0.5}, {"span_hz": 0.0}, {"span_hz": 30.0}, {"pfa": 1e-12}):
                _cand_same(A.hint_line(cs, f, **kw), _cand(N.hint_line(cs, f, **kw)), f"hint {what} {f} {kw}")
        n_d, n_b = cs.stat.shape
        for b in (0, 1, n_b // 2, n_b - 1, -1, -n_b):
            for d in (0, n_d // 2, n_d - 1, -2):
                v = float(cs.stat[d, b])
                for vv in (v, v + 50.0, 0.0):
                    _cand_same(A._cell_line(cs, b, d, vv), _cand(N._cell_line(cs, b, d, vv)),
                               f"cell {what} {b} {d} {vv}")
        probe = lines + [CarrierCandidate(1180.0, 0.0, 0.0), CarrierCandidate(2690.0, 0.0, 0.0),
                         CarrierCandidate(1650.0, 0.0, 0.0)]
        for sep in (A.A2_ALIAS_HZ, 0.0, 5.0):
            ref = A.pilot_lines(cs, probe, sep_hz=sep)
            got = [_cand(d) for d in N.pilot_lines(cs, probe, sep_hz=sep)]
            _cands_same(ref, got, f"pilots {what} {sep}")
        _cands_same(A.pilot_lines(cs, []), [_cand(d) for d in N.pilot_lines(cs, [])], "no lines")


def test_a1_refusals_and_errors(N, caps):
    x, t0 = caps["ce"]
    cs = A.carrier_line_stats(x, t0)
    n_d, n_b = cs.stat.shape
    for b, d in ((n_b, 0), (0, n_d), (-n_b - 1, 0)):
        r = _outcome(A._cell_line, cs, b, d, 10.0)
        g = _outcome(lambda *a: _cand(N._cell_line(*a)), cs, b, d, 10.0)
        assert r == g, (b, d, r, g)
    r = _outcome(A.carrier_line_stats, x, t0, drifts=())
    g = _outcome(N.carrier_line_stats, x, t0, drifts=())
    assert r == g, (r, g)
    bad = dataclasses.replace(cs, stat=cs.stat.astype(np.float64))
    with pytest.raises(NotImplementedError):          # a statistic carrier_line_stats never makes
        N.carrier_lines(x, t0, stats=bad)


# --- A2 ------------------------------------------------------------------------------------------

def _channel_close(ref, got, what):
    assert type(got["t0_ch"]) is type(ref.t0_ch) and got["t0_ch"] == ref.t0_ch, what
    assert got["f_mix"] == ref.f_mix and type(got["f_mix"]) is type(ref.f_mix), what
    worst = 0.0
    if "y" in got:
        worst = max(worst, _close_max(ref.y, got["y"], 1e-12, f"y {what}"))
    else:
        worst = max(worst, _close_max(ref.m, got["m"], 1e-12, f"m {what}"))
        assert type(got["sigma2"]) is float
        worst = max(worst, _close(ref.sigma2, got["sigma2"], 1e-12, f"sigma2 {what}"))
        worst = max(worst, _close(ref.local, got["local"], 1e-12, f"local {what}"))
    return worst


def test_a2_channel(N, caps):
    worst = 0.0
    for what in ("ce", "l", "noise"):
        fe, t0 = caps[what]
        for f, drift in ((1650.0, 0.0), (1203.4, 0.0), (1650.0017, 1.5), (1180.25, -2.5), (340.0, 0.0)):
            for t0x in (t0, t0 + 0.6, -5.0 * FE_FS, len(fe) - 30.0 * FE_FS):
                ref = A._a2_input(fe, t0x, f, drift)
                got = N._a2_input(fe, t0x, f, drift)
                worst = max(worst, _channel_close(ref, got, f"input {what} {f} {t0x}"))
                for rk, nk in ((A.CE_PRE, N.CE_PRE), (A.L_PRE, N.L_PRE)):
                    rc = A._a2_mf(ref, rk)
                    gc = N._a2_mf(ref, nk)
                    worst = max(worst, _channel_close(rc, gc, f"mf {what} {f} {t0x} {rk.name}"))
                    assert gc["kind"] is nk
    assert worst < 1e-12
    # the whole channel, from FE
    fe, t0 = caps["ce"]
    rc = A.preamble_channel(fe, t0, 1650.0, 0.0, kind=A.L_PRE)
    gc = N.preamble_channel(fe, t0, 1650.0, 0.0, kind=A.L_PRE)
    _channel_close(rc, gc, "preamble_channel")
    assert gc["kind"] is A.L_PRE
    assert N.preamble_channel(fe, t0, 1650.0)["kind"] is N.CE_PRE


def _cell(lam, p):
    """preamble_search's grid cell: least p, or the largest Lambda when p underflows."""
    if p.min() > 1e-250:
        return np.unravel_index(int(np.argmin(p)), p.shape)
    return np.unravel_index(int(np.argmax(lam)), lam.shape)


def test_lambda_grid(N, caps, native):
    """_preamble_sums on the reference's channel (S to rounding: numpy 2.5's
    einsum is a batched BLAS zgemm), Lambda within rtol 1e-9 and the same
    winning cell; the C++'s preamble_p on its own S equal to numpy's on it."""
    Q = native.qrss.acquire_stats
    worst = {"lam": 0.0, "S": 0.0}
    for what in ("ce", "l", "noise"):
        fe, t0 = caps[what]
        for f in (1650.0, 1203.4, 1216.0):
            for rk in (A.CE_PRE, A.L_PRE):
                chan = A.preamble_channel(fe, t0, f, 0.0, kind=rk)
                lam, taus, S, var = A._preamble_sums(chan)
                glam, gtaus, gS, gvar = N._preamble_sums(chan)
                _same(gtaus, taus, "taus")
                _same(gvar, var, "var")                   # local[...] * norm, exact
                assert gS.flags.f_contiguous and S.flags.f_contiguous
                assert glam.flags.f_contiguous == lam.flags.f_contiguous
                worst["S"] = max(worst["S"], _close_max(S, gS, 1e-12, f"S {what} {f} {rk.name}"))
                worst["lam"] = max(worst["lam"], _close(lam, glam, 1e-9, f"lam {what} {f} {rk.name}"))
                p = A.preamble_p(S, var)[0]
                gp = A.preamble_p(gS, gvar)[0]
                assert _cell(lam, p) == _cell(glam, gp), (what, f, rk.name)
                # numpy's preamble_p on the C++'s arrays is the C++'s preamble_p on them
                qp = Q.preamble_p(gS, gvar)
                _same(qp[0], gp, "p on native S")
                lam3, taus3, S3 = N.preamble_lambda(chan)
                _same(lam3, glam, "preamble_lambda lam")
                _same(S3, gS, "preamble_lambda S")
                for df, tau_s in (((0.0,), 1.0), ((-0.2, 0.3), 0.0)):
                    r = A._preamble_sums(chan, df, tau_s)
                    g = N._preamble_sums(chan, df, tau_s)
                    _same(g[1], r[1], "taus custom")
                    _close(r[0], g[0], 1e-9, "lam custom")
    assert worst["lam"] < 1e-12 and worst["S"] < 1e-12, worst


def _hit_close(r, g, what):
    """f_hz to 1e-7 Hz, tau0_s to 1e-6 s, the decision the same; returns the
    worst |difference| per field."""
    assert type(g) is A.PreambleHit
    errs = {}
    for f in ("f_hz", "drift_hz_per_min", "tau0_s", "phase", "lam", "cell_lam", "p_value", "tau0_sd_s",
              "lead_in_s"):
        a, b = getattr(r, f), getattr(g, f)
        assert type(b) is type(a) is float, (what, f, type(a), type(b))
        assert math.isnan(a) == math.isnan(b), (what, f, a, b)
        errs[f] = 0.0 if math.isnan(a) else abs(b - a)
    for f in ("passed", "n_groups", "f_mix", "waveform"):
        a, b = getattr(r, f), getattr(g, f)
        assert type(b) is type(a) and a == b, (what, f, a, b)
    assert errs["f_hz"] <= 1e-7 and errs["tau0_s"] <= 1e-6, (what, errs)
    assert errs["drift_hz_per_min"] <= 1e-6, (what, errs)
    for f in ("lam", "cell_lam"):
        assert errs[f] <= 1e-9 * abs(getattr(r, f)), (what, f, errs)
    assert errs["p_value"] <= 1e-6 * r.p_value, (what, errs)
    assert errs["phase"] <= 1e-6, (what, errs)
    if not math.isnan(r.tau0_sd_s):
        assert errs["tau0_sd_s"] <= 1e-6 * r.tau0_sd_s, (what, errs)
    _close_max(r.chunk_sums, g.chunk_sums, 1e-6, f"chunk_sums {what}")
    return errs


def test_preamble_search(N, caps):
    worst: dict = {}
    for what in ("ce", "l", "noise"):
        fe, t0 = caps[what]
        cands = A.carrier_lines(fe, t0)
        probe = list(cands[:2]) + [CarrierCandidate(1650.0, 0.0, 0.0), CarrierCandidate(1180.3, 1.5, 9.0)]
        for c in probe:
            inp = A._a2_input(fe, t0, c.f_hz, c.drift_hz_per_min)
            for rk, nk in ((A.CE_PRE, N.CE_PRE), (A.L_PRE, A.L_PRE)):
                ref = A.preamble_search(fe, t0, c, kind=rk)
                got = _hit(N.preamble_search(fe, t0, c, kind=nk))
                for k, v in _hit_close(ref, got, f"{what} {c.f_hz} {rk.name}").items():
                    worst[k] = max(worst.get(k, 0.0), v)
                got2 = _hit(N.preamble_search(None, t0, c, kind=nk, inp=inp))
                _hit_close(ref, got2, f"inp {what} {c.f_hz} {rk.name}")
            ref = A.preamble_search(fe, t0, c, pfa=1e-300)
            got = _hit(N.preamble_search(fe, t0, c, pfa=1e-300))
            _hit_close(ref, got, f"pfa {what} {c.f_hz}")
    # measured (this machine): f_hz ~1e-12 Hz, tau0_s ~1e-13 s
    assert worst["f_hz"] < 1e-9 and worst["tau0_s"] < 1e-9, worst


def test_preamble_search_p_is_numpys_on_its_own_arrays(N, caps):
    """The hit's p_value and cell_lam are what numpy's preamble_p gives on
    the C++'s own S and var, at the cell numpy would pick from them."""
    for what in ("ce", "l", "noise"):
        fe, t0 = caps[what]
        for f in (1650.0, 1203.4):
            for nk in (N.CE_PRE, N.L_PRE):
                chan = N.preamble_channel(fe, t0, f, 0.0, kind=nk)
                lam2, _taus, S, var = N._preamble_sums(chan)
                pc, sc, st = A.preamble_p(S, var)
                i, j = _cell(lam2, pc)
                hit = N.preamble_search(fe, t0, CarrierCandidate(f, 0.0, 0.0), kind=nk)
                assert hit["p_value"] == float(pc[i, j]), (what, f, nk.name)
                k, lam = st[int(sc[i, j])]
                assert hit["cell_lam"] == float(lam[i, j]) and hit["n_groups"] == k


def test_fine_sums_coherent_phase_line(N, caps, native):
    from scipy.interpolate import CubicSpline

    fe, t0 = caps["ce"]
    for rk in (A.CE_PRE, A.L_PRE):
        chan = A.preamble_channel(fe, t0, 1650.0, 0.0, kind=rk)
        lo, hi = int(chan.t0_ch) - 700, int(chan.t0_ch) + 6000
        x = np.arange(lo, hi)
        for spline in (CubicSpline(x, chan.m[lo:hi]),
                       native.qrss.cubic_spline.CubicSpline(x, chan.m[lo:hi])):
            for tau, df, rate in ((175.0, 0.0, 0.0), (175.3, 0.012, 0.0), (-3.25, -0.3, 0.004)):
                rS, rn = A._fine_sums(spline, chan, tau, df, rate)
                gS, gn = N._fine_sums(spline, chan, tau, df, rate)
                _same(gn, rn, "fine norm")
                _close_max(rS, gS, 1e-12, f"fine S {rk.name} {tau}")
                rc, gc = A._coherent(spline, chan, tau, df, rate), N._coherent(spline, chan, tau, df, rate)
                assert type(gc) is float
                _close(rc, gc, 1e-12, "coherent")
                for dt in (rk.chunk * rk.t_sym, 0.5):
                    r, g = A._phase_line(rS, dt), N._phase_line(rS, dt)
                    assert [type(v) for v in g] == [type(v) for v in r], (r, g)
                    for a, b in zip(r, g):
                        assert abs(b - a) <= 1e-9 * max(abs(a), 1e-3), (r, g)
    rng = np.random.default_rng(4)
    for S in (np.zeros(10, complex), np.zeros(1, complex), np.zeros(0, complex),
              rng.standard_normal(10) + 1j * rng.standard_normal(10), np.ones(2, complex),
              np.array([1, 1j, -1, -1j, 1]), np.array([1, np.nan, 1, 1])):
        r, g = _outcome(A._phase_line, S, 0.6), _outcome(N._phase_line, S, 0.6)
        assert r[0] == g[0]
        if r[0] == "ok":
            assert [type(v) for v in g[1]] == [type(v) for v in r[1]], (S, r, g)
            np.testing.assert_allclose(g[1], r[1], rtol=1e-12, atol=1e-15)


# --- lead-in -------------------------------------------------------------------------------------

def test_lead_in_span(N, monkeypatch):
    """Spans on CE and L lead-ins of several lengths, at the carrier and
    beside it, through both noise_psd paths (the C++'s own; the
    reference's frontend.noise_psd called back with its arguments)."""
    cases = []
    for lead in (0.0, 3.0, 10.0):
        z = TA._clean("short", 40.0, lead) if lead else TA._clean()
        x, t0 = TA._capture(1650.0, -15.0, shift_s=0.3, seed=7, z=z)
        cases.append(("ce", lead, x, t0, 1650.0, 0.3, A.T_SYM))
        z = TL._l_clean(lead_in_s=lead)
        x, t0 = TL._capture(z, 1203.4, -15.0, shift_s=-0.2, seed=8)
        cases.append(("l", lead, x, t0, 1203.4, -0.2, A.L_T_SYM))
    spans = []
    for what, lead, x, t0, f, tau0, t_sym in cases:
        for df, drift in ((0.0, 0.0), (0.02, 0.0), (40.0, 0.0), (0.0, 0.5)):
            for kw in ({"t_sym": t_sym}, {"t_sym": t_sym, "z_min": 2.0}, {}):
                ref = A.lead_in_span(x, t0, f + df, drift, tau0, **kw)
                got = N.lead_in_span(x, t0, f + df, drift, tau0, **kw)
                assert type(got) is type(ref) is float and got == ref, (what, lead, df, drift, kw, ref, got)
                spans.append(ref)
                cb = N.lead_in_span(x, t0, f + df, drift, tau0, _noise_psd=frontend.noise_psd, **kw)
                assert cb == ref
    assert min(spans) == 0.0 and max(spans) >= 9.0           # both outcomes exercised
    # the seam: called as the reference calls frontend.noise_psd
    x, t0 = cases[0][2], cases[0][3]
    seen = []

    def spy(ch, fs, **kw):
        seen.append((ch.dtype, len(ch), fs, kw))
        return np.full(7, 0.5, np.float32)
    monkeypatch.setattr(frontend, "noise_psd", spy)
    ref = A.lead_in_span(x, t0, 1650.0)
    want = seen.pop()
    got = N.lead_in_span(x, t0, 1650.0, _noise_psd=spy)
    assert got == ref and seen.pop() == want
    # edges: a capture that does not reach the window, empty, too short for a noise floor
    monkeypatch.undo()
    for fe, t0x in ((x, -1e6), (x, len(x) + 1e6), (x[:0], 0.0), (x[:FE_FS], 0.5 * FE_FS)):
        r = _outcome(A.lead_in_span, fe, t0x, 1650.0)
        g = _outcome(N.lead_in_span, fe, t0x, 1650.0)
        assert r == g, (r, g)


def test_to_detection(N, caps):
    fe, t0 = caps["ce"]
    hit = A.preamble_search(fe, t0, CarrierCandidate(1650.0, 0.0, 0.0))
    hit.lead_in_s = 3.0
    for h in (hit, dataclasses.replace(hit, tau0_sd_s=float("nan"), drift_hz_per_min=-2.5,
                                       waveform=WAVEFORM_L)):
        for dur in (1800.0, 75.0):
            from qrss_native import _rx

            ref, got = h.to_detection(dur), _rx.to_ref(N.to_detection(h, dur))
            assert type(got) is type(ref)
            for f in dataclasses.fields(ref):
                a, b = getattr(ref, f.name), getattr(got, f.name)
                if dataclasses.is_dataclass(a):
                    for ff in dataclasses.fields(a):
                        _same(np.asarray(getattr(b, ff.name)), np.asarray(getattr(a, ff.name)), ff.name)
                else:
                    assert type(b) is type(a) and (a == b or (a != a and b != b)), (f.name, a, b)


# --- the callers, on the C++ ---------------------------------------------------------------------

def _apply_rows(native, monkeypatch):
    from qrss_native import acquire_a12, acquire_stats

    for mod in (acquire_stats, acquire_a12):
        for (target, attr), fn in mod.rows(native.qrss).items():
            monkeypatch.setattr(_owner(target), attr, fn)


def _acq_close(r, g, what):
    _cands_same(r.lines, g.lines, f"{what} lines")
    _cands_same(r.pilots, g.pilots, f"{what} pilots")
    _cands_same(r.hinted, g.hinted, f"{what} hinted")
    assert len(r.preamble) == len(g.preamble), (what, r.preamble, g.preamble)
    for a, b in zip(r.preamble, g.preamble):
        _hit_close(a, b, f"{what} preamble")
    assert len(r.tbd) == len(g.tbd), what
    for a, b in zip(r.tbd, g.tbd):            # A3 sees only A1's lines, which are exact
        assert type(b) is type(a)
        for f in dataclasses.fields(a):
            x, y = getattr(a, f.name), getattr(b, f.name)
            if dataclasses.is_dataclass(x):
                for ff in dataclasses.fields(x):
                    _same(np.asarray(getattr(y, ff.name)), np.asarray(getattr(x, ff.name)), ff.name)
            elif x is not None:
                assert x == y or (x != x and y != y), (what, f.name, x, y)


def test_acquire_end_to_end_on_the_cpp(N, caps, native, monkeypatch):
    """acquire() -- A1, A2 at every line in both kinds, L's pilot lines, the
    hints, the arbitration, each lead-in, A3 -- on CE and L SHORT at -15 dB
    and on noise, with this module's rows and acquire_stats' assigned over
    acquire's names: the same lines, hits, hints and A3 hits."""
    joint = (frame.SHORT, lframe.L_SHORT)
    hints = (1650.0, (1203.4, WAVEFORM_L), 2100.0)
    runs = [("ce", joint, hints, False), ("l", joint, hints, True), ("noise", joint, hints, True),
            ("ce", frame.SHORT, (1650.5, 1203.4), True)]

    def run():
        out = []
        for what, spec_set, hnt, live_only in runs:
            fe, t0 = caps[what]
            cap = frontend.Capture(fe, 0, t0)
            out.append(A.acquire(cap, spec_set, preprocessed=True, hints=hnt, live_only=live_only))
        return out

    ref = run()
    _apply_rows(native, monkeypatch)
    assert A.carrier_line_stats is not ref and A.CE_PRE.mf_taps is not None
    got = run()
    n_hits = 0
    for (what, spec_set, _h, _l), r, g in zip(runs, ref, got):
        _acq_close(r, g, f"{what} {getattr(spec_set, 'name', 'joint')}")
        n_hits += len(r.preamble)
    assert n_hits >= 3                         # passed hits were compared, not only empty lists


def test_receive_pass_on_the_cpp(native, monkeypatch):
    """receiver.detect and receive_pass on a CE TINY pass with the rows
    assigned: the same detections and the same pass."""
    import test_qrss_receiver as TR
    from sstvae.qrss import receiver as RX
    from qrss_helpers import Q_TEST

    _a, sim = TR.scenario("tiny", -6.0, 3)
    spec = frame.get("tiny")

    def run():
        prep = RX.prepare(frontend.Capture(sim.fe, Q_TEST, sim.t0_index))
        dets = RX.detect(prep, spec)
        return dets, (RX.receive_pass(prep, spec, dets[0]) if dets else None)

    rdets, rp = run()
    _apply_rows(native, monkeypatch)
    gdets, gp = run()
    assert len(rdets) == len(gdets) >= 1
    for a, b in zip(rdets, gdets):
        assert (a.method, a.waveform) == (b.method, b.waveform)
        assert abs(a.f_hz - b.f_hz) <= 1e-7 and a.lead_in_s == b.lead_in_s
        if a.timing is not None:
            assert abs(a.timing.tau0 - b.timing.tau0) <= 1e-6 * CH_FS
    assert (rp is None) == (gp is None) and rp is not None
    assert rp.header == gp.header and rp.cw == gp.cw            # TINY: no header, no windows
    assert (rp.uid, rp.frame, rp.waveform, rp.estimator, rp.em_round, rp.f_mix_hz) == \
        (gp.uid, gp.frame, gp.waveform, gp.estimator, gp.em_round, gp.f_mix_hz)
    assert abs(gp.f_hz - rp.f_hz) <= 1e-7 and abs(gp.timing.tau0 - rp.timing.tau0) <= 1e-6 * CH_FS
    for f in ("z", "w", "hdr_llr", "psi"):
        a, b = getattr(rp, f), getattr(gp, f)
        assert a.dtype == b.dtype and a.shape == b.shape, f
        if a.size:
            err = float(np.max(np.abs(b.astype(np.float64) - a)))
            assert err <= 1e-3 * max(float(np.max(np.abs(a))), 1.0), (f, err)

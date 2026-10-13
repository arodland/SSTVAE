"""Parity of native/core/qrss/cwid (sstvae_native.qrss.cwid) with
sstvae/qrss/cwid.py, held side by side in one process.

- The reader: `read_ml` (text equal, Z and the margin to 1e-9) and `read`
  on 1800 seeded per-unit LLR stacks built as tests/test_qrss_cwid.py
  builds them (+-mu keying of a call in unit noise, 1-4 rows, 1-D too),
  from signal far over the read threshold down to noise (180 of them with
  the margin, a programme per character read); its dynamic programme
  `_dp` / `_trace` and the alphabet `_patterns` exactly.
- The windows: `_unit_sums` (1e-10), `window_soft`, `window_llr` (both
  keyings for CE; the bar's 1e-10 before the float32 cast read as at
  most one float32 ulp in at most 1e-4 of the values) and `keying_type`
  (equal) on the Python tracker's own TrackResults of CE and L passes
  (tests/qrss_demod_tracks.py), and with a channel given in place of the
  track's z_d.
- `match` to 1e-10, `decode_units` exactly (calls, random flags, every
  refusal), `_ln_i0` (see its test), `_combine` (row-major stacks of up
  to 200 rows exactly, and a column-major one refused, since numpy sums
  that in another order), and the reference's errors.
- The caller: `receiver._cw_result` (keying_type, window_soft, read of
  window_llr, match) on the SHORT tracks with cwid's rows assigned over
  the reference's names, every one of them seen to run: the same
  CwIdResult. And `receiver.receive_pass` of a SHORT CE pass and an
  L_SHORT one the same way (slow): the same cw, round B's tracking from
  its cw_known, and the same latents.

Measured (2026-10-13): everything but `_ln_i0` bit-identical -- read_ml's
Z and margin, the unit sums, every window value on every track (0 ulps),
and match, whose np.dot is OpenBLAS's SkylakeX ddot here (its order and
fused multiply-adds are reproduced; another BLAS kernel would leave an
ulp, which the 1e-10 allows).
"""

import math
from types import SimpleNamespace

import numpy as np
import pytest

import qrss_demod_tracks as D
from qrss_native import require
from sstvae.qrss import cwid, frame, lframe
from sstvae.qrss.morse import keying_units

CALLS = ("K1ABC", "VE3XYZ7", "G4/W1AW", "M0A", "W1AW", "KH6/N0CL", "E", "T", "5", "/", "Q0QQ")


@pytest.fixture(scope="module")
def C(native):
    return require(native, "cwid")


def _soft(call, mu, nrow, rng):
    k = keying_units(call)[cwid.CALL_UNITS].astype(np.float64) * 2 - 1
    return mu * k[None, :] + rng.standard_normal((nrow, len(k)))


def _llr(call, mu, nrow, rng):
    return 2 * mu * _soft(call, mu, nrow, rng)


def _outcome(fn, *args, **kw):
    try:
        return "ok", fn(*args, **kw)
    except (ValueError, TypeError, IndexError, ZeroDivisionError, KeyError, NotImplementedError) as e:
        return type(e), str(e)


def _rel(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    if a.size == 0:
        return 0.0
    same = (a == b) | (np.isnan(a) & np.isnan(b))
    if same.all():
        return 0.0
    d = np.abs(a - b)[~same]
    return float(np.max(d / np.maximum(np.abs(b[~same]), 1e-300)))


def _ulps32(a, b):
    a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
    ia, ib = a.view(np.int32).astype(np.int64), b.view(np.int32).astype(np.int64)
    ia = np.where(ia < 0, -(2 ** 31) - ia, ia)
    ib = np.where(ib < 0, -(2 ** 31) - ib, ib)
    return int(np.max(np.abs(ia - ib))) if a.size else 0


def _random_call(rng):
    alpha = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789/"
    for _ in range(100):
        s = "".join(alpha[i] for i in rng.integers(0, len(alpha), rng.integers(1, 9)))
        try:
            keying_units(s)
            return s
        except ValueError:
            continue
    return "K1ABC"


# --- the reader --------------------------------------------------------------------------------

def test_patterns(C):
    chars, pats = C._patterns()
    assert chars == cwid._CHARS
    assert len(pats) == len(cwid._PATS)
    for a, b in zip(pats, cwid._PATS):
        assert a.dtype == b.dtype and np.array_equal(a, b)


def _stacks(n, seed):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        call = CALLS[i % len(CALLS)] if i % 3 else _random_call(rng)
        mu = (0.0, 0.3, 0.6, 0.9, 1.3, 2.0, 3.0)[rng.integers(0, 7)]
        nrow = int(rng.integers(1, 5))
        L = _llr(call, mu, nrow, rng)
        if i % 7 == 0:
            L = L.sum(axis=0)                         # 1-D: one row
        out.append(L)
    return out


def test_read_ml(C):
    """1800 stacks: the read and Z (margin=False, one programme); 180 of
    them with the margin too (a programme per character read), and `read`,
    which is the margin's verdict. Text equal, Z and the margin to 1e-9
    relative (worst measured below)."""
    worst_z = worst_m = 0.0
    n_text = n_read = 0
    for i, L in enumerate(_stacks(1800, 1)):
        rt, rz = cwid.read_ml(L)
        gt, gz = C.read_ml(L)
        assert gt == rt, (rt, gt)
        assert type(gz) is type(rz), (type(rz), type(gz))
        worst_z = max(worst_z, _rel(gz, rz))
        n_text += rt is not None
        if i % 10 == 0:
            rt, rz, rm = cwid.read_ml(L, margin=True)
            gt, gz, gm = C.read_ml(L, margin=True)
            assert gt == rt and type(gm) is type(rm), (rt, gt, type(rm), type(gm))
            worst_z, worst_m = max(worst_z, _rel(gz, rz)), max(worst_m, _rel(gm, rm))
            r_read = None if rt is None or rm < cwid.READ_LLR_MIN else rt   # cwid.read, as written
            assert C.read(L) == r_read and C.read(L, True) == r_read
            n_read += r_read is not None
    assert n_text > 1500 and 40 < n_read < 175, (n_text, n_read)
    assert worst_z <= 1e-9 and worst_m <= 1e-9, (worst_z, worst_m)
    print(f"read_ml worst rel: Z {worst_z:.3g}, margin {worst_m:.3g}")


def test_read_ml_options(C):
    rng = np.random.default_rng(2)
    for i in range(150):
        L = _llr(CALLS[i % len(CALLS)], 1.5, 2, rng)
        for max_chars in (1, 2, 3, 5):
            r, g = cwid.read_ml(L, max_chars), C.read_ml(L, max_chars)
            assert g[0] == r[0] and _rel(g[1], r[1]) <= 1e-9, (max_chars, r, g)
        r, g = cwid.read_ml(L), C.read_ml(L)
        assert len(g) == len(r) == 2 and g[0] == r[0] and _rel(g[1], r[1]) <= 1e-9
    # short and odd inputs: the DP over fewer units than a call needs
    for n in (0, 1, 2, 3, 4, 5, 13, 40):
        L = rng.standard_normal(n) * 3
        for margin in (False, True):
            r, g = _outcome(cwid.read_ml, L, margin=margin), _outcome(C.read_ml, L, margin=margin)
            assert r[0] == g[0], (n, r, g)
            if r[0] == "ok":
                assert g[1][0] == r[1][0] and _rel(g[1][1:], r[1][1:]) <= 1e-9, (n, r, g)
            else:
                assert r[1] == g[1], (n, r, g)
    for L in (np.zeros(176), np.full(176, -5.0), np.full((2, 176), 3.0), [1.0] * 176):
        r, g = cwid.read_ml(L, margin=True), C.read_ml(L, margin=True)
        assert g[0] == r[0] and _rel(g[1:], r[1:]) <= 1e-9, (r, g)
    for bad in (np.zeros((2, 3, 176)), np.float64(1.0), np.zeros(176), np.zeros((0, 176))):
        r, g = _outcome(cwid.read_ml, bad), _outcome(C.read_ml, bad)
        if np.ndim(bad) > 2:                       # numpy's broadcast error; not ported
            assert r[0] is ValueError and g[0] is NotImplementedError, (r, g)
        else:
            assert r == g, (np.shape(bad), r, g)
    for bad in (np.float64(1.0), np.zeros((2, 3, 176))):
        r, g = _outcome(cwid.match, bad, "K1ABC"), _outcome(C.match, bad, "K1ABC")
        assert r == g or (np.ndim(bad) > 2 and g[0] is NotImplementedError), (r, g)
    for mc in (0, -1):
        r, g = _outcome(cwid.read_ml, np.ones(176), mc), _outcome(C.read_ml, np.ones(176), mc)
        assert r == g, (mc, r, g)


def test_dp_and_trace(C):
    rng = np.random.default_rng(3)
    lens = np.array([len(p) for p in cwid._PATS], dtype=np.int64)
    for trial in range(30):
        n = int(rng.integers(5, 60))
        gains = []
        for pat in cwid._PATS:
            g = np.full(n + 1, -np.inf)
            if len(pat) <= n:
                g[:n - len(pat) + 1] = rng.standard_normal(n - len(pat) + 1) * 4
            gains.append(g)
        for forbid in (None, (int(rng.integers(0, len(lens))), int(rng.integers(0, n)))):
            for mc in (1, 3, 8):
                rb, rk = cwid._dp(gains, lens, n, mc, forbid)
                gb, gk = C._dp(gains, lens, n, mc, forbid)
                assert gb.dtype == rb.dtype and gk.dtype == rk.dtype
                assert np.array_equal(gb, rb) and np.array_equal(gk, rk), (trial, forbid, mc)
                r, g = cwid._trace(rb, rk), C._trace(gb, gk)
                assert r == g, (trial, r, g)


def test_match(C):
    rng = np.random.default_rng(4)
    worst = 0.0
    for i in range(400):
        call = CALLS[i % len(CALLS)] if i % 2 else _random_call(rng)
        soft = _soft(call if i % 3 else "K1ABC", 1.0, int(rng.integers(1, 4)), rng)
        if i % 5 == 0:
            soft = soft[0]
        r, g = cwid.match(soft, call), C.match(soft, call)
        assert type(g) is type(r) is float
        worst = max(worst, _rel(g, r))
    assert worst <= 1e-10, worst                  # measured 0: np.dot's order and FMAs, emulated
    s = rng.standard_normal((2, 176))
    for call in ("", "k1abc", "K1ABC!", "ABCDEFGHI", "0000000", 5):
        r, g = _outcome(cwid.match, s, call), _outcome(C.match, s, call)
        assert r == g, (call, r, g)
    for bad in (np.zeros(175), np.zeros((2, 10))):
        r, g = _outcome(cwid.match, bad, "K1ABC"), _outcome(C.match, bad, "K1ABC")
        assert r == g, (bad.shape, r, g)


def test_decode_units(C):
    for call in CALLS:
        k = keying_units(call)[cwid.CALL_UNITS]
        assert C.decode_units(k) == cwid.decode_units(k) == call
        assert C.decode_units(list(k)) == call
        assert C.decode_units(k.astype(bool)) == call
        assert C.decode_units(np.concatenate([[0, 0, 0], k])) == cwid.decode_units(np.concatenate([[0, 0, 0], k]))
    rng = np.random.default_rng(5)
    n_ok = 0
    for i in range(3000):
        call = CALLS[i % len(CALLS)]
        k = keying_units(call)[cwid.CALL_UNITS].astype(np.uint8)
        flips = rng.integers(0, 176, rng.integers(0, 4))
        k[flips] ^= 1
        if i % 4 == 0:
            k = (rng.random(int(rng.integers(0, 200))) < 0.5).astype(np.uint8)
        r, g = cwid.decode_units(k), C.decode_units(k)
        assert r == g, (k.tolist(), r, g)
        n_ok += r is not None
    assert n_ok > 500
    for k in ([], [0] * 10, [1], [1, 0, 0, 0, 1], [1] * 6, [1, 0, 0, 0, 0, 0, 0, 0, 1], [0.5, 0, 0],
              np.array([2, 0, -1])):
        assert C.decode_units(k) == cwid.decode_units(k), k


def test_ln_i0_and_combine(C):
    """_ln_i0 = log(i0e(x)) + x: numpy's float64 log is a SIMD kernel, not
    glibc's, so a last-ulp difference in the log survives the
    cancellation against x at small x: measured 5e-11 relative there
    (where the value is ~x^2/4, so ~1e-26 absolute), 1e-14 from x = 1
    up, 99.7% bit-identical."""
    rng = np.random.default_rng(16)
    x = np.concatenate([np.linspace(0, 50, 1001), np.exp(rng.uniform(-30, 7, 20000)),
                        [0.0, 1e-300, 1e-8, 700.0, 1e5, 1e300, np.inf, -1.0, -700.0, np.nan]])
    r, g = cwid._ln_i0(x), C._ln_i0(x)
    assert g.dtype == r.dtype and g.shape == r.shape
    assert _rel(g, r) <= 1e-10, _rel(g, r)
    big = x >= 1.0
    assert _rel(g[big], r[big]) <= 1e-14, _rel(g[big], r[big])
    assert np.mean(g == r) > 0.99
    r, g = cwid._ln_i0(3.5), C._ln_i0(3.5)
    assert type(g) is type(r) and _rel(g, r) <= 1e-14
    m = x[:600].reshape(3, 200)
    assert C._ln_i0(m).shape == (3, 200) and _rel(C._ln_i0(m), cwid._ln_i0(m)) <= 1e-14
    rng = np.random.default_rng(6)
    for s in (rng.standard_normal(176), rng.standard_normal((3, 176)), rng.standard_normal((1, 5)),
              [1.0, 2.0], rng.standard_normal((3, 176)).astype(np.float32)):
        (rs, rn), (gs, gn) = cwid._combine(s), C._combine(s)
        assert rn == gn and type(gn) is type(rn)
        assert gs.dtype == rs.dtype and np.array_equal(gs, rs)


# --- the windows -------------------------------------------------------------------------------

def _spec_of(label):
    return {"tiny": frame.TINY, "tiny-erased": frame.TINY, "short": frame.SHORT,
            "l-tiny": lframe.L_TINY, "l-short": lframe.L_SHORT}[label.split("/")[0]]


@pytest.fixture(scope="module")
def tracks(C):
    out = dict(D.ce_tracks())
    out.update(D.l_tracks())
    return out


def test_windows_on_tracks(C, tracks):
    """window_soft / window_llr / _unit_sums / keying_type on every CE and
    L track: float32 outputs within one ulp, in at most 1e-4 of the
    values (1e-10 relative before the cast can move a value across a
    float32 rounding boundary, rarely), _unit_sums to 1e-10, keying_type
    equal. Measured here: all of it bit-identical."""
    worst = {"sums": 0.0, "ulps": 0, "units": 0, "n": 0}
    for label, tr in tracks.items():
        spec = _spec_of(label)
        for ook in ((False, True) if spec.waveform == 0 else (False,)):
            for name in ("window_soft", "window_llr"):
                r = getattr(cwid, name)(None, 250, tr, spec, ook=ook)
                g = getattr(C, name)(None, 250, tr, spec, ook=ook)
                assert g.dtype == r.dtype == np.float32 and g.shape == r.shape, (label, name)
                u = _ulps32(g, r)
                worst["ulps"] = max(worst["ulps"], u)
                worst["units"] += int(np.count_nonzero(g != r))
                worst["n"] += g.size
                assert u <= 1, (label, name, ook, u)
        if spec.n_win:
            for rv, gv in zip(cwid._unit_sums(tr, spec), C._unit_sums(tr, spec)):
                assert gv.dtype == rv.dtype and gv.shape == rv.shape
                worst["sums"] = max(worst["sums"], _rel(gv.real, rv.real), _rel(gv.imag, rv.imag))
        for call in (None, D.CALL, "VE3XYZ7"):
            assert C.keying_type(None, tr, spec, call) == cwid.keying_type(None, tr, spec, call), (label, call)
    assert worst["sums"] <= 1e-10, worst
    assert worst["units"] <= 1e-4 * worst["n"], worst
    print("windows", worst)


def test_windows_with_a_channel(C, tracks):
    """`ch` given: the windows read it in place of tr.z_d (a scaled copy
    and noise of the same length), and the reference's refusals: L keyed
    by OOK; a channel the C++ does not read (complex64) is
    NotImplementedError, not a different answer."""
    rng = np.random.default_rng(7)
    for label in ("short/A", "l-short/A"):
        tr, spec = tracks[label], _spec_of(label)
        n = len(tr.z_d)
        for z in (tr.z_d * (0.5 - 0.25j), (rng.standard_normal(n) + 1j * rng.standard_normal(n)) * 0.1):
            for name in ("window_soft", "window_llr"):
                r = getattr(cwid, name)(z, 250, tr, spec)
                g = getattr(C, name)(z, 250, tr, spec)
                assert _ulps32(g, r) <= 1, (label, name)
            for call in (None, D.CALL):
                assert C.keying_type(z, tr, spec, call) == cwid.keying_type(z, tr, spec, call)
        short = tr.z_d[: n // 3]
        r = cwid.window_soft(short, 250, tr, spec)
        assert _ulps32(C.window_soft(short, 250, tr, spec), r) <= 1, label
    tr = tracks["l-short/A"]
    for name in ("window_soft", "window_llr"):
        r = _outcome(getattr(cwid, name), None, 250, tr, lframe.L_SHORT, ook=True)
        g = _outcome(getattr(C, name), None, 250, tr, lframe.L_SHORT, ook=True)
        assert r == g and r[0] is ValueError, (r, g)
    assert _outcome(C._fsk_only, lframe.L_SHORT, True) == _outcome(cwid._fsk_only, lframe.L_SHORT, True)
    assert C._fsk_only(frame.SHORT, True) is None
    assert C.keying_type(None, SimpleNamespace(), lframe.L_SHORT) == "fsk"
    assert C.keying_type(None, SimpleNamespace(), frame.TINY, D.CALL) == "fsk"
    with pytest.raises(NotImplementedError):
        C.window_soft(tracks["short/A"].z_d.astype(np.complex64), 250, tracks["short/A"], frame.SHORT)


def test_cw_result_on_the_cpp(C, native, tracks, monkeypatch):
    """receiver._cw_result -- keying_type, window_soft, read(window_llr),
    match -- on the SHORT tracks, header call known, unknown and invalid,
    with every row of tests/qrss_native/cwid.py over cwid's names: the
    same CwIdResult (z_match to 1e-10) and round B's cw_known."""
    from qrss_native.cwid import rows
    from sstvae.qrss import receiver as RX

    labels = ("short/A", "short/B", "short/genie", "l-short/A", "l-short/B")
    hdrs = (None, SimpleNamespace(callsign="K1ABC   "), SimpleNamespace(callsign="VE3XYZ7"),
            SimpleNamespace(callsign="k1abc"))

    def run():
        return [RX._cw_result(tracks[lb], _spec_of(lb), h) for lb in labels for h in hdrs]

    ref = run()
    calls = {}

    def counted(name, fn):
        def f(*a, **k):
            calls[name] = calls.get(name, 0) + 1
            return fn(*a, **k)
        return f

    for (target, name), fn in rows(native.qrss).items():
        assert target == "sstvae.qrss.cwid"
        monkeypatch.setattr(cwid, name, counted(name, fn))
    got = run()
    # every function _cw_result calls ran on the C++ (read_ml only inside
    # the C++ read, never back through Python)
    assert {n: calls.get(n, 0) > 0 for n in ("keying_type", "window_soft", "window_llr", "read", "match")} \
        == dict.fromkeys(("keying_type", "window_soft", "window_llr", "read", "match"), True), calls
    assert "read_ml" not in calls and "_unit_sums" not in calls, calls
    for (r, rk), (g, gk) in zip(ref, got):
        assert (g.text, g.keying, g.agrees) == (r.text, r.keying, r.agrees), (r, g)
        assert type(g.z_match) is float and _rel(g.z_match, r.z_match) <= 1e-10, (r, g)
        assert (rk is None) == (gk is None)
        if rk is not None:
            assert np.array_equal(gk[0], rk[0]) and gk[1] == rk[1]
    assert any(r.text == D.CALL for r, _ in ref)


# --- review probes -----------------------------------------------------------------------------

def test_combine_many_rows_and_layouts(C):
    """_combine's s.sum(axis=0) over stacks of 9 to 200 rows, where
    numpy's order starts to matter: a row-major stack (float32 or
    float64, reversed, strided or a broadcast row) is summed row by row
    from row 0 as the C++ does -- equal bytes, and read_ml / read / match
    of it equal too. A column-major stack numpy sums pairwise down each
    column, which is another number from 8 rows up: the C++ refuses it
    (NotImplementedError) rather than answer differently."""
    rng = np.random.default_rng(31)
    n_diff = 0
    for n in (2, 9, 17, 40, 200):
        a = rng.standard_normal((n, 176)) * 10.0 ** rng.uniform(-3, 3, (n, 176))
        mu = 0.3 * a.std()
        a = a + mu * (keying_units("K1ABC")[cwid.CALL_UNITS] * 2.0 - 1.0)
        row_major = {"C": a, "C32": a.astype(np.float32), "rev": a[::-1], "revcols": a[:, ::-1],
                     "step": np.repeat(a, 2, axis=0)[::2], "bcast": np.broadcast_to(a[0], (n, 176))}
        for k, x in row_major.items():
            (rs, rn), (gs, gn) = cwid._combine(x), C._combine(x)
            assert gn == rn == n and gs.dtype == rs.dtype and np.array_equal(gs, rs), (n, k)
            assert C.read_ml(x, margin=True) == cwid.read_ml(x, margin=True), (n, k)
            assert C.read(x) == cwid.read(x) and C.match(x, "K1ABC") == cwid.match(x, "K1ABC"), (n, k)
        for k, x in {"F": np.asfortranarray(a), "F32": np.asfortranarray(a.astype(np.float32)),
                     "T": np.ascontiguousarray(a.T).T}.items():
            for fn in (C._combine, C.read_ml, C.read, lambda v: C.match(v, "K1ABC")):
                with pytest.raises(NotImplementedError, match="column-major"):
                    fn(x)
            n_diff += not np.array_equal(cwid._combine(x)[0], cwid._combine(np.ascontiguousarray(x))[0])
    assert n_diff >= 6, n_diff                    # the refusal is not hiding an equal answer


@pytest.mark.slow
@pytest.mark.parametrize("which", ["ce-short", "l-short"])
def test_receive_pass_on_the_cpp(C, native, monkeypatch, which):
    """receiver.receive_pass of a SHORT CE pass (-9 dB) and an L_SHORT one
    (-6 dB), once on the reference and once with every cwid row assigned
    over the reference's names: the same cw (text, keying, agrees,
    z_match to 1e-10), and -- since round B tracks with the keying
    cw_known hands it -- the same header, latents and weights. ~10 s."""
    from qrss_native.cwid import rows
    from sstvae.qrss import frontend
    from sstvae.qrss import receiver as RX
    from qrss_fakes import L_CALL, l_scenario
    from sstvae.qrss.constants import WAVEFORM_L

    if which == "ce-short":
        spec, sim = D._ce_sim("short", -9.0, 3)
        det, call = D.flat_det(1500.0), D.CALL
    else:
        spec = lframe.L_SHORT
        _, sim = l_scenario("l-short", -6.0, 2)
        det, call = D.flat_det(1500.0, WAVEFORM_L), L_CALL
    prep = RX.prepare(frontend.Capture(sim.fe, D.Q_TEST, sim.t0_index))

    ref = RX.receive_pass(prep, spec, det)
    calls = {}
    for (_, name), fn in rows(native.qrss).items():
        def counted(*a, _n=name, _f=fn, **k):
            calls[_n] = calls.get(_n, 0) + 1
            return _f(*a, **k)
        monkeypatch.setattr(cwid, name, counted)
    got = RX.receive_pass(prep, spec, det)
    assert calls.get("window_soft") and calls.get("window_llr") and calls.get("read"), calls
    assert ref is not None and got is not None and ref.cw is not None
    r, g = ref.cw, got.cw
    assert (g.text, g.keying, g.agrees) == (r.text, r.keying, r.agrees), (r, g)
    assert type(g.z_match) is float and _rel(g.z_match, r.z_match) <= 1e-10, (r, g)
    assert g.text == call and g.agrees, g
    assert got.header == ref.header
    for f in ("z", "w", "hdr_llr", "psi"):
        a, b = getattr(got, f), getattr(ref, f)
        assert np.asarray(a).dtype == np.asarray(b).dtype and np.array_equal(a, b, equal_nan=True), f
    assert got.uid == ref.uid and got.f_hz == ref.f_hz

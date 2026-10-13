"""Parity of native/core/qrss/chan (sstvae_native.qrss.chan) with
sstvae/qrss/track.py's make_chan, ChanCapture.noise_psd_at and
ChanCapture.noise_floor.

- make_chan: the channel complex64 within rtol 1e-6 (in practice: equal),
  the identical Fraction, t0_index, keep and the zero padding to the bit,
  inside, across and past either end of the capture; the refusals;
- noise_psd_at / noise_floor within rtol 1e-9 (noise_psd's bar) under both
  rules, with windows of zero padding taking the others' median, an
  all-zero channel's floor, a channel shorter than a window; the table
  cached in `_psd` exactly as the reference caches it (and used as it is
  when already set), a dict form, scalar and shaped t_s;
- the noise_psd seam called with the reference's arguments.
"""

import copy
from fractions import Fraction

import numpy as np
import pytest

from qrss_native import _rx, require
from sstvae.qrss import frontend, rules, track
from sstvae.qrss.constants import CH_FS, FE_FS


@pytest.fixture(scope="module")
def nc(native):
    return require(native, "chan")


@pytest.fixture(scope="module")
def capture():
    rng = np.random.default_rng(7)
    n = FE_FS * 120
    fe = ((rng.standard_normal(n) + 1j * rng.standard_normal(n)) / np.sqrt(2)).astype(np.complex64)
    t = np.arange(n) / FE_FS
    fe += (0.2 * np.exp(2j * np.pi * (812.4 - 1500.0) * t)).astype(np.complex64)
    fe, keep = frontend.blank(fe)
    fe, _ = frontend.normalise(fe, keep)
    return fe, keep


def _same(a, b):
    a, b = np.asarray(a), np.asarray(b)
    assert a.dtype == b.dtype and a.shape == b.shape, (a.dtype, b.dtype, a.shape, b.shape)
    an = np.isnan(a) if a.dtype.kind in "fc" else np.zeros(a.shape, bool)
    assert np.array_equal(an, np.isnan(b) if b.dtype.kind in "fc" else an)
    assert a[~an].tobytes() == b[~an].tobytes()


def _chan_same(got, ref):
    assert isinstance(got, track.ChanCapture)
    assert got.ch.dtype == ref.ch.dtype == np.complex64 and got.ch.shape == ref.ch.shape
    if ref.ch.size:
        np.testing.assert_allclose(got.ch, ref.ch, rtol=1e-6, atol=0)   # sample by sample, no floor
    assert got.fs == ref.fs and isinstance(got.f_mix, Fraction) and got.f_mix == ref.f_mix
    assert got.t0_index == ref.t0_index
    assert (got.keep is None) == (ref.keep is None)
    if ref.keep is not None:
        _same(got.keep, ref.keep)
    assert got.exclude_hz == tuple(ref.exclude_hz)
    assert got._psd is None


def _make(nc, *args, **kw):
    return _rx.to_ref(nc.make_chan(*args, **kw))


_SPANS = [(-20.0, 30.0), (-80.0, 10.0), (90.0, 200.0), (-200.0, -150.0), (0.0, 0.0), (5.0, 5.0001),
          (-10.0, 100.0)]


@pytest.mark.parametrize("t0", [50.0 * FE_FS, 50.0 * FE_FS + 0.37, 1234.5, -3000.25])
def test_make_chan(nc, capture, t0):
    fe, keep = capture
    for lo, hi in _SPANS:
        for f, k, ex in ((812.4, keep, (700.0, 900.5)), (330.25, None, ()), (2690.0, keep, [1500])):
            ref = track.make_chan(fe, k, t0, FE_FS, f, lo, hi, ex)
            _chan_same(_make(nc, fe, k, t0, FE_FS, f, lo, hi, ex), ref)


def test_make_chan_types_and_refusals(nc, capture):
    fe, keep = capture
    small = fe[:FE_FS * 5]
    for x, k, t0, fs in ((small.astype(np.complex128), keep[:FE_FS * 5].astype(np.float64), 100, FE_FS),
                         (small, None, 100, float(FE_FS)), (small.real, None, 100.0, FE_FS),
                         (small, None, 0, 8000), (small, None, 0, CH_FS)):
        _chan_same(_make(nc, x, k, t0, fs, 900.0, -1.0, 2.0), track.make_chan(x, k, t0, fs, 900.0, -1.0, 2.0))
    for args in ((small, None, 0, 100, 900.0, 0.0, 1.0),                    # fs_fe < CH_FS: modulo by zero
                 (small, None, 0, FE_FS, 900.0, 1.0, -1.0),                 # a negative span
                 (small, None, float("nan"), FE_FS, 900.0, 0.0, 1.0),
                 (small, None, 0, FE_FS, 900.0, -float("inf"), 1.0),
                 (small, None, 0, FE_FS, float("nan"), 0.0, 1.0),
                 (small, None, 0, 4100, 900.0, 0.0, 1.0)):                  # 250 does not divide 4100
        with pytest.raises(Exception) as ref:
            track.make_chan(*args)
        with pytest.raises(Exception) as got:
            nc.make_chan(*args)
        assert type(got.value) is type(ref.value) and str(got.value) == str(ref.value), (args, ref.value)


@pytest.fixture(scope="module")
def chans(capture):
    """Reference ChanCaptures: whole, half padded, mostly padded, short,
    all zero, at the band's edges and its middle."""
    fe, keep = capture
    t0 = 50.0 * FE_FS
    out = [track.make_chan(fe, keep, t0, FE_FS, 812.4, -40.0, 60.0, (700.0, 900.5)),
           track.make_chan(fe, keep, t0, FE_FS, 330.25, -40.0, 120.0, (400.0,)),
           track.make_chan(fe, keep, t0, FE_FS, 2690.0, 60.0, 95.0, ()),
           track.make_chan(fe, None, t0, FE_FS, 1500.0, -60.0, -48.0, (1580.0,)),
           track.make_chan(fe, None, t0, FE_FS, 1200.0, -10.0, -4.0, ()),
           track.make_chan(fe, None, t0, FE_FS, 1200.0, 300.0, 330.0, ())]
    return out


@pytest.mark.parametrize("old", [False, True])
def test_noise_psd_at(nc, chans, old, monkeypatch):
    if old:
        monkeypatch.setenv(rules.EDGE_NOISE, "1")
    else:
        monkeypatch.delenv(rules.EDGE_NOISE, raising=False)
    t = np.linspace(-70.0, 140.0, 333)
    for ref_chan in chans:
        ref, got = copy.deepcopy(ref_chan), copy.deepcopy(ref_chan)
        want = ref.noise_psd_at(t)
        have = nc.noise_psd_at(got, t)
        assert have.dtype == want.dtype == np.float64 and have.shape == want.shape
        np.testing.assert_allclose(have, want, rtol=1e-9, atol=0)
        # the cache: written back as the reference writes it
        assert got._psd is not None and len(got._psd) == 2
        _same(got._psd[0], ref._psd[0])
        np.testing.assert_allclose(got._psd[1], ref._psd[1], rtol=1e-9, atol=0)
        assert got._psd[1].dtype == np.float64
        # noise_floor on a fresh copy and on the cached one
        fresh = copy.deepcopy(ref_chan)
        assert nc.noise_floor(fresh) == pytest.approx(ref.noise_floor(), rel=1e-9, abs=0)
        assert nc.noise_floor(got) == pytest.approx(ref.noise_floor(), rel=1e-9, abs=0)


def test_noise_psd_at_shapes_and_the_cache(nc, chans):
    ref = copy.deepcopy(chans[0])
    x = ref.noise_psd_at(3.0)
    got = copy.deepcopy(chans[0])
    y = nc.noise_psd_at(got, 3.0)
    assert type(y) is type(x) is np.float64 and y == pytest.approx(x, rel=1e-9)
    for t in ([0.0, 1.0], np.zeros((2, 3)), np.float32(7.5), 4):
        a, b = nc.noise_psd_at(got, t), ref.noise_psd_at(t)
        assert np.shape(a) == np.shape(b) and type(a) is type(b)
    # a table already in _psd is used as it is
    planted = copy.deepcopy(chans[0])
    planted._psd = (np.array([0.0, 10.0]), np.array([1.0, 3.0]))
    assert nc.noise_psd_at(planted, 5.0) == 2.0 and nc.noise_floor(planted) == 2.0
    # the dict form (what make_chan returns before to_ref) caches in the dict
    d = nc.make_chan(np.ones(FE_FS * 20, np.complex64), None, 0.0, FE_FS, 1000.0, 0.0, 20.0)
    assert d["_psd"] is None
    nc.noise_psd_at(d, 1.0)
    assert d["_psd"] is not None and d["_psd"][0].dtype == np.float64


def test_noise_psd_at_with_no_window(nc):
    """An empty channel: no window, so np.interp refuses -- after caching
    the empty table, as the reference does."""
    for impl in ("ref", "nat"):
        cap = track.ChanCapture(ch=np.zeros(0, np.complex64), fs=CH_FS, t0_index=0.0, f_mix=Fraction(0))
        with pytest.raises(Exception) as e:
            if impl == "ref":
                cap.noise_psd_at(0.0)
            else:
                nc.noise_psd_at(cap, 0.0)
        assert isinstance(e.value, ZeroDivisionError), e.value   # noise_psd's own refusal first


def test_the_seam_gets_the_references_arguments(nc):
    seen = []

    def spy(ch, fs, **kw):
        seen.append((ch, fs, kw))
        return np.array([2.0, 0.0, 4.0], np.float32)

    ch = np.ones(CH_FS * 25, np.complex64)
    ch[: CH_FS * 3] = 0                                    # the first window mostly padding
    cap = track.ChanCapture(ch=ch, fs=CH_FS, t0_index=10.0, f_mix=Fraction(-4679, 4), exclude_hz=(400.0, 250))
    ref = copy.deepcopy(cap)
    got = nc.noise_psd_at(cap, [0.0, 50.0], spy)
    seen_ref = []

    def spy_ref(ch_, fs_, **kw):
        seen_ref.append((ch_, fs_, kw))
        return np.array([2.0, 0.0, 4.0], np.float32)

    orig = frontend.noise_psd
    frontend.noise_psd = spy_ref
    try:
        want = ref.noise_psd_at([0.0, 50.0])
    finally:
        frontend.noise_psd = orig
    np.testing.assert_array_equal(got, want)
    np.testing.assert_array_equal(cap._psd[1], ref._psd[1])
    _same(cap._psd[0], ref._psd[0])
    (ch_n, fs_n, kw_n), (ch_r, fs_r, kw_r) = seen[0], seen_ref[0]
    assert ch_n is cap.ch and fs_n == fs_r == CH_FS
    assert kw_n == kw_r and kw_n["audio_hz"] == 330.25 and kw_n["exclude_hz"] == [400.0 - 330.25, 250 - 330.25]
    assert list(kw_n) == list(kw_r)


def test_noise_floor_of_silence(nc):
    """No good window: every window is max(median, 1e-30)."""
    cap = track.ChanCapture(ch=np.zeros(CH_FS * 30, np.complex64), fs=CH_FS, t0_index=0.0, f_mix=Fraction(0))
    ref = copy.deepcopy(cap)
    assert nc.noise_floor(cap) == ref.noise_floor() == 1e-30
    _same(cap._psd[1], ref._psd[1])


# --- review probes ------------------------------------------------------------------------------

def test_padding_fraction_counts_nan_as_the_reference_does(nc):
    """`np.abs(ch) > 0` is False for a NaN sample, so a window of NaNs is
    "mostly padding" and takes the others' median -- which a plain nonzero
    test would not do (seen only through a noise_psd that reads finite
    values off such a channel, as a replaced one can)."""
    def spy(ch, fs, **kw):
        return np.array([2.0, 3.0, 4.0], np.float32)

    ch = np.ones(CH_FS * 25, np.complex64)
    ch[:2400] = np.complex64(complex(np.nan, 0.0))         # window 0: 4% numbers
    ch[2600:2800] = np.complex64(complex(0.0, np.nan))     # window 1: 92% numbers
    cap = track.ChanCapture(ch=ch, fs=CH_FS, t0_index=0.0, f_mix=Fraction(-4679, 4))
    ref = copy.deepcopy(cap)
    got = nc.noise_psd_at(cap, [5.0, 15.0, 22.0], spy)
    orig = frontend.noise_psd
    frontend.noise_psd = spy
    try:
        want = ref.noise_psd_at([5.0, 15.0, 22.0])
    finally:
        frontend.noise_psd = orig
    np.testing.assert_array_equal(got, want)
    np.testing.assert_array_equal(cap._psd[1], ref._psd[1])
    assert ref._psd[1][0] == 3.5                           # the median of the good windows


@pytest.mark.parametrize("seed", range(8))
def test_make_chan_and_noise_floor_random(nc, capture, seed, monkeypatch):
    """make_chan over random spans, offsets, frequencies and neighbours
    (inside, across and off either end of the capture), then the noise
    table, under both rules."""
    fe, keep = capture
    rng = np.random.default_rng(4000 + seed)
    for _ in range(6):
        t0 = float(rng.uniform(-len(fe) / 2, 1.5 * len(fe)))
        lo = float(rng.uniform(-60, 60))
        hi = lo + float(rng.uniform(0, 90))
        f = float(rng.uniform(280, 2720))
        ex = tuple(float(v) for v in rng.uniform(200, 2800, int(rng.integers(0, 4))))
        k = keep if rng.random() < 0.7 else None
        ref = track.make_chan(fe, k, t0, FE_FS, f, lo, hi, ex)
        got = _make(nc, fe, k, t0, FE_FS, f, lo, hi, ex)
        _chan_same(got, ref)
        if not ref.ch.size:
            continue
        if rng.random() < 0.5:
            monkeypatch.setenv(rules.EDGE_NOISE, "1")
        else:
            monkeypatch.delenv(rules.EDGE_NOISE, raising=False)
        t = rng.uniform(lo - 5, hi + 5, 9)
        a, b = copy.deepcopy(ref), copy.deepcopy(ref)
        np.testing.assert_allclose(nc.noise_psd_at(b, t), a.noise_psd_at(t), rtol=1e-9, atol=0)
        assert nc.noise_floor(b) == pytest.approx(a.noise_floor(), rel=1e-9, abs=0)

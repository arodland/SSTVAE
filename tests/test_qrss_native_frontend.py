"""Parity of native/core/qrss/frontend (sstvae_native.qrss.frontend) with
sstvae/qrss/frontend.py (without the passband store) and live.StreamFE,
and of native/core/math/convolve (sstvae_native.qrss.convolve) with the
scipy calls behind them.

- audio_to_fe and StreamFE: complex64 within 1e-5 of the samples' scale,
  no sample more than 1 ulp apart (in practice: equal);
- fe_to_audio within 1e-12 of the scale;
- running_median, _interp_to_samples, blank, normalise, quantise_mix,
  mix_phasor, channelise_keep to the bit; channelise complex64 within
  rtol 1e-6 (in practice: equal) with the identical Fraction;
- the noise floor's tables (_noise_response, _fold_power, _fe_power,
  _fe_flat, _fe_response) to the bit; _fe_power, and with it the flat
  masks, at every 1/4 Hz of 300-440 and 2560-2700 Hz (every 1/64 Hz in
  the slow tier); noise_psd within rtol 1e-9 under both rules, through
  every branch (sides, flat, skirt fallback, quieter side);
- the refusals' types and texts;
- oaconvolve/fftconvolve against scipy within 1e-13 of the scale with
  scipy's block sizes, freqz to the bit, maximum_filter1d exactly.
"""

import threading
from fractions import Fraction

import numpy as np
import pytest
import scipy
from scipy import ndimage, signal

from qrss_native import require
from sstvae.qrss import frontend as F
from sstvae.qrss import live, rules
from sstvae.qrss.constants import CH_FS, FE_FS, FS


@pytest.fixture(scope="module")
def nf(native):
    return require(native, "frontend")


@pytest.fixture(scope="module")
def cv(native):
    return require(native, "convolve")


def _splitmix(seed, n):
    s = np.uint64(seed) + np.arange(1, n + 1, dtype=np.uint64) * np.uint64(0x9E3779B97F4A7C15)
    z = (s ^ (s >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return z ^ (z >> np.uint64(31))


def _noise(seed, n, complex_=False):
    rng = np.random.default_rng(seed)
    if complex_:
        return (rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(np.complex64)
    return rng.standard_normal(n)


def _same(a, b):
    a, b = np.asarray(a), np.asarray(b)
    assert a.dtype == b.dtype, (a.dtype, b.dtype)
    assert a.shape == b.shape, (a.shape, b.shape)
    if a.dtype.kind in "fc":
        an, bn = np.isnan(a), np.isnan(b)
        assert np.array_equal(an, bn)
        assert a[~an].tobytes() == b[~bn].tobytes(), np.flatnonzero(a != b)[:10]
    else:
        assert a.tobytes() == b.tobytes()


def _ulps32(a, b, floor):
    """Largest distance in float32 ulps between two complex64 arrays' parts,
    over the parts at least `floor` in magnitude (below it an ulp is far
    under the arithmetic's own rounding at the signal's scale, which the
    absolute bound covers)."""
    def ordered(x):
        i = x.view(np.int32).astype(np.int64)
        return np.where(i < 0, -(i & 0x7FFFFFFF), i)
    ra, rb = a.view(np.float32), b.view(np.float32)
    big = np.abs(rb) >= floor
    return int(np.max(np.abs(ordered(ra[big]) - ordered(rb[big])))) if big.any() else 0


def _fe_close(a, b):
    """The plan's bar for audio_to_fe/StreamFE: complex64, at most 1e-5 of
    the samples (rounded up) not bit-identical, and those one ulp apart
    (parts far below the signal's scale, where an ulp is under the
    arithmetic's own rounding, within one ulp of the scale instead)."""
    assert a.dtype == b.dtype == np.complex64 and a.shape == b.shape
    if a.size:
        scale = max(float(np.max(np.abs(a))), 1e-30)
        differ = int(np.count_nonzero(a.view(np.uint64) != b.view(np.uint64)))
        assert differ <= int(np.ceil(1e-5 * a.size)), (differ, a.size)
        assert float(np.max(np.abs(a.astype(np.complex128) - b))) <= 2.0**-23 * scale
        assert _ulps32(a, b, 1e-6 * scale) <= 1


def _raises_same(fa, fb):
    """The reference and the port refuse alike: same type, same text."""
    with pytest.raises(Exception) as ea:
        fa()
    with pytest.raises(Exception) as eb:
        fb()
    assert type(eb.value) is type(ea.value), (ea.value, eb.value)
    assert str(eb.value) == str(ea.value)


# --- constants -------------------------------------------------------------------------------

def test_constants(nf):
    for name in ("FE_FLAT_DB", "FE_MIN_HZ", "FE_SKIRT_DB", "F_MIX_DEN", "_FE_DECIM"):
        assert getattr(nf, name) == getattr(F, name), name
    assert (nf.ALIGN, nf.HALO) == (live.ALIGN, live.HALO)
    _same(nf._FE_TAPS, F._FE_TAPS)
    _same(nf._FE_TAPS, signal.firwin(255, 1250.0, fs=FS, window=("kaiser", 8.0)))
    assert not nf._FE_TAPS.flags.writeable


# --- audio <-> FE ------------------------------------------------------------------------------

@pytest.mark.parametrize("n", [0, 1, 2, 3, 16, 100, 255, 256, 257, 511, 1000, 2420, 8001, 40000])
def test_audio_to_fe(nf, n):
    x = _noise(n + 7, n) * 3.0
    _fe_close(nf.audio_to_fe(x), F.audio_to_fe(x))


def test_audio_to_fe_tone_and_dtypes(nf):
    t = np.arange(16000) / FS
    x = 0.5 * np.sin(2 * np.pi * 1712.25 * t)
    _fe_close(nf.audio_to_fe(x), F.audio_to_fe(x))
    for x in (x.astype(np.float32), np.arange(300, dtype=np.int16), list(range(50))):
        _fe_close(nf.audio_to_fe(x), F.audio_to_fe(x))


@pytest.mark.parametrize("n", [0, 1, 2, 5, 128, 129, 1210, 20000])
def test_fe_to_audio(nf, n):
    fe = _noise(n + 3, n, complex_=True)
    a, b = nf.fe_to_audio(fe), F.fe_to_audio(fe)
    assert a.dtype == b.dtype == np.float64 and a.shape == b.shape
    if n:
        assert np.max(np.abs(a - b)) <= 1e-12 * max(np.max(np.abs(b)), 1e-300)


def test_fe_hz(nf):
    for f in (0.0, 330.25, 1500, 2699.99, Fraction(1, 3)):
        assert nf.fe_hz(f) == F.fe_hz(f)


# --- StreamFE ----------------------------------------------------------------------------------

def _drive(stream, pushes):
    out = []
    for a, x in pushes:
        out.append([(int(i), np.asarray(fe)) for i, fe in stream.push(a, x)])
    return out


@pytest.mark.parametrize("block", [8000, 16, 1000, 7, 0])
def test_stream_fe(nf, block):
    rng = np.random.default_rng(block + 11)
    x = rng.standard_normal(40000)
    # contiguous pushes of uneven sizes, starting off the ALIGN grid, then a
    # gap (a restart) and a push shorter than the realignment skip
    pushes, a = [], 5
    for n in (3, 700, 1, 4000, 255, 9000, 2):
        pushes.append((a, x[a:a + n]))
        a += n
    pushes.append((a + 101, x[a + 101:a + 101 + 5000]))
    pushes.append((30001, x[30001:30010]))
    pushes.append((30010, x[30010:39999]))
    ref, nat = live.StreamFE(block), nf.StreamFE(block)
    assert nat.block == ref.block
    for (a, xs), r_out, n_out in zip(pushes, _drive(ref, pushes), _drive(nat, pushes)):
        assert [i for i, _ in n_out] == [i for i, _ in r_out], a
        for (_, fr), (_, fn) in zip(r_out, n_out):
            _fe_close(fn, fr)
    assert nat.a0 == ref.a0 and nat.next_index == ref.next_index
    _same(nat.pend, ref.pend)


def test_stream_fe_matches_the_batch(nf):
    """Away from the stream's first and last HALO samples, StreamFE is
    audio_to_fe of the whole: the C++ against itself and the reference."""
    x = _noise(5, 24000)
    s = nf.StreamFE(4000)
    got = {}
    for a in range(0, 24000, 3000):
        for i, fe in s.push(a, x[a:a + 3000]):
            for k, v in enumerate(fe):
                got[i + k] = v
    batch = F.audio_to_fe(x)
    idx = np.array(sorted(got))
    assert idx[0] == nf.HALO // 2
    vals = np.array([got[i] for i in idx], dtype=np.complex64)
    inner = (idx >= nf.HALO) & (idx < len(batch) - nf.HALO)
    assert np.max(np.abs(vals[inner] - batch[idx[inner]])) <= 1e-5 * np.max(np.abs(batch))


def test_stream_fe_fresh(nf):
    s = nf.StreamFE()
    assert s.a0 is None and s.next_index is None and s.pend.size == 0 and s.block == 8000


# --- blanker and inverse AGC ----------------------------------------------------------------

@pytest.mark.parametrize("n,fs,win,hop", [(0, 4000, 1.0, 0.05), (1, 4000, 1.0, 0.05), (4000, 4000, 1.0, 0.05),
                                          (4001, 4000, 1.0, 0.05), (12345, 4000, 1.0, 0.05),
                                          (5000, 250.0, 2.0, 0.3), (777, 1000, 0.1, 0.0001),
                                          (3000, 4000, 0.00001, 0.05)])
def test_running_median(nf, n, fs, win, hop):
    p = (np.abs(_noise(n, n, complex_=True)) ** 2).astype(np.float32)
    p[::17] = np.round(p[::17])                    # ties
    c1, m1 = F.running_median(p, fs, win, hop)
    c2, m2 = nf.running_median(p, fs, win, hop)
    _same(c2, c1)
    _same(m2, m1)


def test_running_median_nan_windows(nf):
    """NaN entering and leaving the sliding windows (the C++ keeps them
    sorted across hops), signed zeros, and a window wider than its hop."""
    p = (np.abs(_noise(8, 20000, complex_=True)) ** 2).astype(np.float32)
    p[[50, 3000, 3001, 9999, 19999]] = np.nan
    p[100:140] = 0.0
    p[200:220] = -0.0
    for fs, win, hop in ((4000, 1.0, 0.05), (4000, 0.1, 0.01), (1000, 0.5, 0.001), (4000, 0.01, 0.05)):
        c1, m1 = F.running_median(p, fs, win, hop)
        c2, m2 = nf.running_median(p, fs, win, hop)
        _same(c2, c1)
        _same(m2, m1)


def test_interp_to_samples(nf):
    for n, c, v in ((10, [0, 4, 9], [1.0, 2.0, 7.0]), (0, [], []), (5, [2], [3.5]),
                    (2_100_000, np.arange(100, 2_100_000, 200), np.linspace(1, 9, 10500))):
        _same(nf._interp_to_samples(n, np.asarray(c), np.asarray(v, dtype=np.float32)),
              F._interp_to_samples(n, np.asarray(c), np.asarray(v, dtype=np.float32)))
    _raises_same(lambda: F._interp_to_samples(3, np.array([]), np.array([])),
                 lambda: nf._interp_to_samples(3, np.array([]), np.array([])))
    _raises_same(lambda: F._interp_to_samples(3, np.array([1.0, 2.0]), np.array([1.0])),
                 lambda: nf._interp_to_samples(3, np.array([1.0, 2.0]), np.array([1.0])))
    _raises_same(lambda: F._interp_to_samples(-1, np.array([1.0]), np.array([1.0])),
                 lambda: nf._interp_to_samples(-1, np.array([1.0]), np.array([1.0])))
    _raises_same(lambda: F._interp_to_samples(3, np.array([1.0]), np.array([1.0]), chunk=0),
                 lambda: nf._interp_to_samples(3, np.array([1.0]), np.array([1.0]), chunk=0))


def _impulsive(seed, n):
    fe = _noise(seed, n, complex_=True)
    rng = np.random.default_rng(seed + 1)
    for i in rng.integers(0, n, max(1, n // 3000)):
        fe[i] *= 40.0                              # isolated impulses
    if n > 20000:
        fe[20000:20030] *= 25.0                    # and a burst
        fe[-3:] *= 60.0                            # at the very end
        fe[:2] *= 60.0                             # and the very start
    return fe


@pytest.mark.parametrize("kw", [{}, dict(guard_ms=0.0), dict(taper_ms=0.0), dict(guard_ms=5.0, taper_ms=3.0),
                                dict(k_sigma=3.0, win_s=0.5, hop_s=0.01), dict(fs=250, guard_ms=20.0),
                                dict(taper_ms=-2.0)])
def test_blank(nf, kw):
    fe = _impulsive(3, 60000)
    a, b = nf.blank(fe, **kw), F.blank(fe, **kw)
    _same(a[0], b[0])
    _same(a[1], b[1])


@pytest.mark.parametrize("n", [0, 101, 102, 4000, 4001, 9000])
def test_blank_short_streams(nf, n):
    fe = _impulsive(n, n) if n else np.zeros(0, np.complex64)
    if n:
        fe[n // 2] = 100.0
    a, b = nf.blank(fe), F.blank(fe)
    _same(a[0], b[0])
    _same(a[1], b[1])


@pytest.mark.parametrize("n", [1, 50, 100])
def test_blank_refuses_what_the_reference_does(nf, n):
    """1 to 100 samples at FE_FS: running_median gives no centre and
    np.interp refuses (the reference's own behaviour, kept)."""
    fe = _noise(n, n, complex_=True)
    _raises_same(lambda: F.blank(fe), lambda: nf.blank(fe))


def test_blank_refusals(nf):
    fe = _impulsive(1, 5000)
    _raises_same(lambda: F.blank(fe, guard_ms=-1.0), lambda: nf.blank(fe, guard_ms=-1.0))
    _raises_same(lambda: F.blank(fe, win_s=float("nan")), lambda: nf.blank(fe, win_s=float("nan")))


def test_blank_no_impulse_and_nan(nf):
    fe = _noise(9, 30000, complex_=True)
    for x in (fe, np.where(np.arange(30000) == 7, np.complex64(np.nan), fe).astype(np.complex64)):
        a, b = nf.blank(x), F.blank(x)
        _same(a[0], b[0])
        _same(a[1], b[1])


@pytest.mark.parametrize("kw", [{}, dict(win_ms=20.0, hop_ms=3.0), dict(min_kept=0.9), dict(min_kept=0.0),
                                dict(fs=250, win_ms=1000.0, hop_ms=100.0), dict(win_ms=1e6, hop_ms=1000.0)])
def test_normalise(nf, kw):
    fe, keep = F.blank(_impulsive(4, 50000))
    fe = fe * np.where(np.arange(50000) < 25000, 1.0, 30.0).astype(np.float32)   # an AGC step
    for k in (keep, None):
        a, b = nf.normalise(fe, k, **kw), F.normalise(fe, k, **kw)
        _same(a[0], b[0])
        _same(a[1], b[1])


def test_normalise_keep_dtypes_and_edges(nf):
    fe = _noise(5, 9000, complex_=True)
    keep64 = np.ones(9000)
    keep64[100:400] = 1.0 - 1e-12                  # < 1 in float64, == 1 as float32
    keep64[5000:5003] = np.nan
    for k in (keep64, keep64.astype(np.float32), (keep64 > 0.5).astype(int), np.zeros(9000)):
        a, b = nf.normalise(fe, k), F.normalise(fe, k)
        _same(a[0], b[0])
        _same(a[1], b[1])
    for x in (np.zeros(0, np.complex64), np.zeros(10, np.complex64), np.full(500, np.nan, np.complex64)):
        a, b = nf.normalise(x), F.normalise(x)
        _same(a[0], b[0])
        _same(a[1], b[1])
    _raises_same(lambda: F.normalise(fe, np.ones(10)), lambda: nf.normalise(fe, np.ones(10)))
    _raises_same(lambda: F.normalise(fe, chunk=0), lambda: nf.normalise(fe, chunk=0))
    _same(nf.normalise(np.zeros(0, np.complex64), np.ones(3))[1], F.normalise(np.zeros(0, np.complex64), np.ones(3))[1])


# --- channeliser --------------------------------------------------------------------------

def test_quantise_mix(nf):
    for f in (0.0, -1169.75, 0.5 / 64, 1.5 / 64, -2.5 / 64, 1e6 + 1 / 128, Fraction(7, 3), 3, "12.3"):
        a, b = nf.quantise_mix(f), F.quantise_mix(f)
        assert isinstance(a, Fraction) and a == b
    for f in (float("nan"), float("inf"), -float("inf")):
        _raises_same(lambda: F.quantise_mix(f), lambda: nf.quantise_mix(f))


@pytest.mark.parametrize("fs_in", [4000, 250, 8000, 1])
def test_mix_phasor(nf, fs_in):
    for f_mix in (Fraction(0), Fraction(-4679, 4), Fraction(1, 64), Fraction(-3, 64), 125, Fraction(7, 1),
                  -1169.75, Fraction(1999, 1)):
        for n0 in (0, 1, -5, 12345678901, -987654321):
            for n in (0, 1, 7, 1000):
                _same(nf.mix_phasor(n0, n, f_mix, fs_in), F.mix_phasor(n0, n, f_mix, fs_in))
    _raises_same(lambda: F.mix_phasor(0, 10, Fraction(1, 128), fs_in),
                 lambda: nf.mix_phasor(0, 10, Fraction(1, 128), fs_in))
    _raises_same(lambda: F.mix_phasor(0, 10, 0.3, fs_in), lambda: nf.mix_phasor(0, 10, 0.3, fs_in))


def test_mix_phasor_edges(nf):
    for n in (-1, -63, -64, -100):
        try:
            ref = F.mix_phasor(0, n, Fraction(1, 1), 4000)
        except Exception:
            _raises_same(lambda: F.mix_phasor(0, n, Fraction(1, 1), 4000),
                         lambda: nf.mix_phasor(0, n, Fraction(1, 1), 4000))
        else:
            _same(nf.mix_phasor(0, n, Fraction(1, 1), 4000), ref)
    _raises_same(lambda: F.mix_phasor(0, 5, Fraction(1, 1), 0), lambda: nf.mix_phasor(0, 5, Fraction(1, 1), 0))


def _ch_close(a, b):
    assert a[0].dtype == b[0].dtype == np.complex64 and a[0].shape == b[0].shape
    assert isinstance(a[1], Fraction) and a[1] == b[1]
    if b[0].size:   # rtol 1e-6 sample by sample (no absolute floor: it would excuse small samples)
        np.testing.assert_allclose(a[0], b[0], rtol=1e-6, atol=0)


@pytest.mark.parametrize("fs_in,fs_out", [(4000, 250), (4000, 4000), (8000, 250), (4000, 500), (250, 250)])
def test_channelise(nf, fs_in, fs_out):
    fe = _noise(fs_in + fs_out, 30001, complex_=True)
    for f in (-1169.75, 0.0, 400.3, 1199.99, -fs_in / 2):
        for n0 in (0, 16 * 12345, -77):
            _ch_close(nf.channelise(fe, fs_in, f, fs_out, n0=n0), F.channelise(fe, fs_in, f, fs_out, n0=n0))


def test_channelise_dtypes_and_refusals(nf):
    fe = _noise(2, 5000, complex_=True)
    for x in (fe.astype(np.complex128) * (1 + 1e-9), fe.real.astype(np.float64), fe.real, np.zeros(0, np.complex64),
              np.ones(3, np.complex64)):
        _ch_close(nf.channelise(x, 4000, -200.0), F.channelise(x, 4000, -200.0))
    _ch_close(nf.channelise(fe, 4000.0, -200.0, 250.0), F.channelise(fe, 4000.0, -200.0, 250.0))
    _raises_same(lambda: F.channelise(fe, 4000, 1.0, 300), lambda: nf.channelise(fe, 4000, 1.0, 300))
    _raises_same(lambda: F.channelise(fe, 4000, 1.0, 0), lambda: nf.channelise(fe, 4000, 1.0, 0))
    _raises_same(lambda: F.channelise(fe, 4000, float("nan")), lambda: nf.channelise(fe, 4000, float("nan")))
    _raises_same(lambda: F.channelise(fe, 4000, 1.0, 300, n0=0), lambda: nf.channelise(fe, 4000, "x", 300, n0=0))


@pytest.mark.parametrize("fs_in,fs_out", [(4000, 250), (4000, 4000), (8000, 250), (250, 4000)])
def test_channelise_keep(nf, fs_in, fs_out):
    _, keep = F.blank(_impulsive(6, 40000))
    for k in (keep, keep.astype(np.float64), np.zeros(0, np.float32), np.ones(5)):
        _same(nf.channelise_keep(k, fs_in, fs_out), F.channelise_keep(k, fs_in, fs_out))
    _raises_same(lambda: F.channelise_keep(keep, 4000, 0), lambda: nf.channelise_keep(keep, 4000, 0))


# --- the noise floor's tables --------------------------------------------------------------

_RATES = [(4000, 250, 2500), (4000, 250, 1000), (4000, 250, 16), (4000, 250, 17), (250, 250, 2500),
          (8000, 250, 2500), (4000, 500, 999)]


@pytest.mark.parametrize("fs_in,fs_out,nfft", _RATES)
def test_noise_response_and_folds(nf, fs_in, fs_out, nfft):
    _same(nf._noise_response(fs_in, fs_out, nfft), F._noise_response(fs_in, fs_out, nfft))
    for band in ((85.0, 120.0), (0.0, 125.0), (200.0, 300.0)):
        for a, b in zip(nf._fold_power(fs_in, fs_out, nfft, band), F._fold_power(fs_in, fs_out, nfft, band)):
            _same(a, b)
        for audio in (330.0, 1500.0, 2697.5):
            for a, b in zip(nf._fe_power(fs_in, fs_out, nfft, audio, band),
                            F._fe_power(fs_in, fs_out, nfft, audio, band)):
                _same(a, b)
            _same(nf._fe_flat(fs_in, fs_out, nfft, audio, band), F._fe_flat(fs_in, fs_out, nfft, audio, band))
            _same(nf._fe_response(fs_in, fs_out, nfft, audio, band),
                  F._fe_response(fs_in, fs_out, nfft, audio, band))
    assert not nf._fe_flat(fs_in, fs_out, nfft, 330.0).flags.writeable
    assert not nf._fe_response(fs_in, fs_out, nfft, 330.0).flags.writeable


def test_table_refusals(nf):
    _raises_same(lambda: F._noise_response(250, 4000, 100), lambda: nf._noise_response(250, 4000, 100))
    _raises_same(lambda: F._fold_power(250, 4000, 100, (85.0, 120.0)),
                 lambda: nf._fold_power(250, 4000, 100, (85.0, 120.0)))
    _raises_same(lambda: F._fe_flat.__wrapped__(4000, 250, 0, 330.0), lambda: nf._fe_flat(4000, 250, 0, 330.0))


def _edge_grid(step):
    return np.concatenate([np.arange(300 * 64, 440 * 64 + 1, step), np.arange(2560 * 64, 2700 * 64 + 1, step)]) / 64.0


def _sweep(nf, step):
    w = 10 * CH_FS
    for a in _edge_grid(step):
        a = float(a)
        for x, y in zip(nf._fe_power(FE_FS, CH_FS, w, a, (85.0, 120.0)),
                        F._fe_power.__wrapped__(FE_FS, CH_FS, w, a, (85.0, 120.0))):
            _same(x, y)
        _same(nf._fe_flat(FE_FS, CH_FS, w, a), F._fe_flat(FE_FS, CH_FS, w, a))


def test_flat_masks_at_every_quarter_hz_of_the_edges(nf):
    _sweep(nf, 16)


@pytest.mark.slow
def test_flat_masks_at_every_64th_hz_of_the_edges(nf):
    _sweep(nf, 1)


# --- noise_psd ---------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def chans():
    """Channels of white FE noise mixed down to edge and mid-band audio
    frequencies, plus neighbours: (audio_hz, ch) pairs."""
    rng = np.random.default_rng(42)
    fe = ((rng.standard_normal(FE_FS * 25) + 1j * rng.standard_normal(FE_FS * 25)) / np.sqrt(2)).astype(np.complex64)
    t = np.arange(len(fe)) / FE_FS
    out = []
    for audio in (305.0, 330.25, 360.0, 412.5, 1500.0, 2569.0, 2650.0, 2695.0):
        x = fe + 0.3 * np.exp(2j * np.pi * (audio + 100.0 - 1500.0) * t).astype(np.complex64)
        ch, f_mix = F.channelise(x, FE_FS, F.fe_hz(audio), CH_FS)
        out.append((1500.0 + float(f_mix), ch))
    return out


_EXCLUDES = [(), (60.0,), (100.0,), (-100.0, 100.0), (100.0, -100.0, 30.0), (-60.0,), (110.0, -110.0)]


@pytest.mark.parametrize("old", [False, True])
def test_noise_psd(nf, chans, old, monkeypatch):
    if old:
        monkeypatch.setenv(rules.EDGE_NOISE, "1")
    else:
        monkeypatch.delenv(rules.EDGE_NOISE, raising=False)
    for audio, ch in chans:
        for ex in _EXCLUDES:
            for kw in (dict(audio_hz=audio), {}, dict(audio_hz=audio, win_s=3.0), dict(audio_hz=audio, guard_hz=20.0),
                       dict(audio_hz=audio, band_hz=(80.0, 124.0))):
                try:
                    ref = F.noise_psd(ch, CH_FS, ex, **kw)
                except ValueError:
                    _raises_same(lambda: F.noise_psd(ch, CH_FS, ex, **kw), lambda: nf.noise_psd(ch, CH_FS, ex, **kw))
                    continue
                got = nf.noise_psd(ch, CH_FS, ex, **kw)
                assert got.dtype == ref.dtype == np.float32 and got.shape == ref.shape
                np.testing.assert_allclose(got, ref, rtol=1e-9, atol=0)


def test_noise_psd_shapes_and_rates(nf, chans, monkeypatch):
    monkeypatch.delenv(rules.EDGE_NOISE, raising=False)
    audio, ch = chans[1]
    for x in (ch[:17], ch[:2500], ch[:2501], ch[:7499], ch[:16]):
        np.testing.assert_allclose(nf.noise_psd(x, audio_hz=audio), F.noise_psd(x, audio_hz=audio), rtol=1e-9)
    # FE itself as the channel (fs_in == fs), a rate that does not divide fs_in, a float rate
    fe = _noise(8, 8000, complex_=True)
    np.testing.assert_allclose(nf.noise_psd(fe, 4000, win_s=1.0, band_hz=(300.0, 900.0), audio_hz=1500.0),
                               F.noise_psd(fe, 4000, win_s=1.0, band_hz=(300.0, 900.0), audio_hz=1500.0), rtol=1e-9)
    np.testing.assert_allclose(nf.noise_psd(ch, 300, audio_hz=audio), F.noise_psd(ch, 300, audio_hz=audio), rtol=1e-9)
    np.testing.assert_allclose(nf.noise_psd(ch, 250.0, fs_in=4000.0, audio_hz=audio),
                               F.noise_psd(ch, 250.0, fs_in=4000.0, audio_hz=audio), rtol=1e-9)


def test_noise_psd_refusals(nf, chans):
    _, ch = chans[0]
    _raises_same(lambda: F.noise_psd(ch, band_hz=(200.0, 300.0)), lambda: nf.noise_psd(ch, band_hz=(200.0, 300.0)))
    _raises_same(lambda: F.noise_psd(ch[:0]), lambda: nf.noise_psd(ch[:0]))


def test_rules_reach_the_native_noise_floor(nf, chans, monkeypatch):
    """noise_psd reads rules.EDGE_NOISE as the reference does, at call time."""
    audio, ch = chans[0]
    monkeypatch.delenv(rules.EDGE_NOISE, raising=False)
    new = nf.noise_psd(ch, audio_hz=audio)
    monkeypatch.setenv(rules.EDGE_NOISE, "1")
    old = nf.noise_psd(ch, audio_hz=audio)
    monkeypatch.setenv(rules.EDGE_NOISE, "0")
    np.testing.assert_array_equal(nf.noise_psd(ch, audio_hz=audio), new)
    assert not np.array_equal(new, old)
    np.testing.assert_array_equal(old, nf.noise_psd(ch))


def test_threads(nf, chans):
    """The tables' caches and the GIL-free noise floor from several threads."""
    audio, ch = chans[2]
    want = nf.noise_psd(ch, audio_hz=audio)
    masks = {a: F._fe_flat(FE_FS, CH_FS, 2500, a) for a in (301.0, 302.5, 2601.0, 2690.0)}
    errors = []

    def work():
        try:
            for _ in range(4):
                np.testing.assert_array_equal(nf.noise_psd(ch, audio_hz=audio), want)
                for a, m in masks.items():
                    np.testing.assert_array_equal(nf._fe_flat(FE_FS, CH_FS, 2500, a), m)
        except Exception as e:          # pragma: no cover - reported below
            errors.append(e)

    threads = [threading.Thread(target=work) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, errors


# --- the scipy calls (native/core/math/convolve) --------------------------------------------

def test_scipy_version():
    assert scipy.__version__.startswith("1.18"), scipy.__version__


def test_next_fast_len_and_blocks(cv):
    from scipy.signal._signaltools import _calc_oa_lens

    for n in list(range(1, 400)) + [2408, 1000001, 999983, 2**20 + 1]:
        assert cv.next_fast_len(n) == scipy.fft.next_fast_len(n), n
        assert cv.next_fast_len(n, True) == scipy.fft.next_fast_len(n, True), n
    for s1, s2 in ((2_000_000, 255), (40000, 121), (5000, 243), (7000, 161), (300000, 2001), (9000, 481),
                   (7000, 321), (100, 255), (256, 255), (255, 255), (1, 255), (255, 1), (3, 2), (17, 16)):
        got = cv.calc_oa_lens(s1, s2)
        assert got == tuple(_calc_oa_lens(s1, s2)), (s1, s2, got)
    from scipy.special import lambertw

    for x in (-1 / np.e + 1e-12, -0.3, -0.1, -1e-3, -1e-9, -1e-200):
        assert abs(cv.lambertw_m1(x) - lambertw(x, -1).real) <= 1e-14 * abs(lambertw(x, -1).real)


@pytest.mark.parametrize("n1,n2", [(1000, 255), (255, 1000), (5000, 121), (300, 300), (1, 50), (50, 1),
                                   (2500, 2001), (1, 1), (7, 3), (40000, 481)])
@pytest.mark.parametrize("kinds", ["cr", "cc", "rc", "rr"])
@pytest.mark.parametrize("mode", ["full", "same"])
def test_convolve(cv, n1, n2, kinds, mode):
    a = _noise(n1, n1, complex_=kinds[0] == "c").astype(np.complex128 if kinds[0] == "c" else np.float64)
    b = _noise(n2 + 1, n2, complex_=kinds[1] == "c").astype(np.complex128 if kinds[1] == "c" else np.float64)
    for ours, theirs in ((cv.oaconvolve, signal.oaconvolve), (cv.fftconvolve, signal.fftconvolve)):
        x, y = ours(a, b, mode), theirs(a, b, mode=mode)
        assert x.dtype == y.dtype and x.shape == y.shape
        assert np.max(np.abs(x - y)) <= 1e-13 * max(np.max(np.abs(y)), 1e-300)


def test_convolve_edges(cv):
    e = np.zeros(0)
    assert cv.oaconvolve(e, np.ones(3)).size == 0 and signal.oaconvolve(e, np.ones(3)).size == 0
    with pytest.raises(NotImplementedError):
        cv.oaconvolve(np.ones(5), np.ones(3), "valid")
    with pytest.raises(NotImplementedError):
        cv.oaconvolve(np.ones((2, 5)), np.ones(3))


def test_freqz(cv):
    rng = np.random.default_rng(3)
    for nb in (1, 2, 41, 255, 321):
        b = rng.standard_normal(nb)
        for f in (np.linspace(-9000, 9000, 777), np.array([0.0, 125.0, -0.0, 1e9]), np.array([np.nan, np.inf]),
                  rng.uniform(0, 4000, 200)):
            with np.errstate(invalid="ignore"):       # exp(-1j * nan), exp(-1j * inf)
                w1, h1 = signal.freqz(b, worN=f, fs=8000)
            w2, h2 = cv.freqz(b, f, 8000.0)
            _same(w2, w1)
            _same(h2, h1)


def test_maximum_filter1d(cv):
    rng = np.random.default_rng(4)
    for n in (1, 2, 3, 7, 50, 1000):
        x = rng.random(n) < 0.05
        for size in (1, 2, 3, 4, 5, 9, 2 * n + 3, 3 * n + 1):
            _same(cv.maximum_filter1d(x, size), ndimage.maximum_filter1d(x, size))
            u = rng.integers(0, 255, n).astype(np.uint8)
            _same(cv.maximum_filter1d(u, size), ndimage.maximum_filter1d(u, size))
    _raises_same(lambda: ndimage.maximum_filter1d(np.ones(3, np.uint8), 0), lambda: cv.maximum_filter1d(np.ones(3, np.uint8), 0))


# --- review probes (randomised and long inputs) ----------------------------------------------

def test_audio_to_fe_a_full_slot(nf):
    """A FULL pass's worth of audio (15 min 5 s at 8 kHz, 3.6M FE samples):
    thousands of oaconvolve blocks, batched in the C++ (a batch boundary
    carries the overlap), against the bar -- at most 1e-5 of the samples
    one ulp apart, the rest identical."""
    n = FS * 905
    x = _noise(905, n) * 0.25 + 0.05 * np.sin(2 * np.pi * 812.4 * np.arange(n) / FS)
    _fe_close(nf.audio_to_fe(x), F.audio_to_fe(x))


@pytest.mark.parametrize("seed", range(12))
def test_front_end_chain_random(nf, seed):
    """blank -> normalise -> channelise / channelise_keep on random streams:
    impulses and bursts at random places, zero stretches, an inf or a NaN,
    random blanker and AGC settings, a keep with fractional values."""
    rng = np.random.default_rng(1000 + seed)
    n = int(rng.choice([rng.integers(101, 3000), rng.integers(3000, 60000)]))
    fe = ((rng.standard_normal(n) + 1j * rng.standard_normal(n)) * rng.uniform(0.01, 30)).astype(np.complex64)
    for _ in range(int(rng.integers(0, 8))):
        i = int(rng.integers(0, n))
        fe[i:i + int(rng.integers(1, 60))] *= np.float32(rng.uniform(5, 200))
    if rng.random() < 0.3:
        fe[: int(rng.integers(0, n))] = 0
    if rng.random() < 0.2:
        fe[int(rng.integers(0, n))] = np.complex64(complex(np.inf, 0.0))
    if rng.random() < 0.2:
        fe[int(rng.integers(0, n))] = np.complex64(complex(np.nan, 1.0))
    kw = dict(k_sigma=float(rng.uniform(2, 8)), guard_ms=float(rng.choice([0.0, 0.25, 2.0, 7.3])),
              taper_ms=float(rng.choice([0.0, 0.25, 1.0, 3.3])))
    if rng.random() < 0.5:
        kw.update(win_s=float(rng.uniform(0.05, 2.0)), hop_s=float(rng.uniform(0.001, 0.2)))
    with np.errstate(invalid="ignore", over="ignore"):
        ref = F.blank(fe, **kw)
    got = nf.blank(fe, **kw)
    _same(got[0], ref[0])
    _same(got[1], ref[1])
    bfe, keep = ref
    if rng.random() < 0.3:
        keep = keep.copy()
        keep[rng.random(n) < 0.2] = np.float32(rng.uniform(0, 1))
    kw2 = dict(win_ms=float(rng.uniform(1, 400)), hop_ms=float(rng.uniform(0.25, 50)),
               min_kept=float(rng.uniform(0, 1)))
    for k in (keep, None):
        with np.errstate(invalid="ignore", over="ignore"):
            ref = F.normalise(bfe, k, **kw2)
        got = nf.normalise(bfe, k, **kw2)
        _same(got[0], ref[0])
        _same(got[1], ref[1])
    nfe = ref[0]
    f, n0 = float(rng.uniform(-1250, 1250)), int(rng.integers(-10**9, 10**9))
    a, b = nf.channelise(nfe, FE_FS, f, CH_FS, n0=n0), F.channelise(nfe, FE_FS, f, CH_FS, n0=n0)
    assert a[1] == b[1]
    _same(a[0], b[0])                              # equal in practice, inside the rtol 1e-6 bar
    _same(nf.channelise_keep(keep), F.channelise_keep(keep))


@pytest.mark.parametrize("seed", range(4))
def test_noise_psd_random(nf, seed, monkeypatch):
    """noise_psd over random channels, rates, neighbours, bands, guards and
    windows, under both rules: within rtol 1e-9 (equal in practice), and
    refusing alike."""
    rng = np.random.default_rng(2000 + seed)
    for _ in range(60):
        n = int(rng.choice([rng.integers(1, 3000), rng.integers(3000, 12000)]))
        fs = int(rng.choice([250, 250, 250, 500, 300, 4000]))
        ch = ((rng.standard_normal(n) + 1j * rng.standard_normal(n)) * 0.7).astype(np.complex64)
        if rng.random() < 0.4:
            ch += (3 * np.exp(2j * np.pi * rng.uniform(-125, 125) * np.arange(n) / fs)).astype(np.complex64)
        ex = rng.uniform(-130, 130, int(rng.integers(0, 4)))
        if rng.random() < 0.5:
            ex = tuple(float(v) for v in ex)
        kw = {}
        if rng.random() < 0.8:
            kw["audio_hz"] = float(rng.choice([rng.uniform(280, 460), rng.uniform(2540, 2720), rng.uniform(300, 2700)]))
        if rng.random() < 0.3:
            kw["guard_hz"] = float(rng.uniform(5, 60))
        if rng.random() < 0.3:
            kw["band_hz"] = tuple(sorted(float(v) for v in rng.uniform(50, 125, 2)))
        if rng.random() < 0.3:
            kw["win_s"] = float(rng.uniform(0.01, 12))
        if rng.random() < 0.5:
            monkeypatch.setenv(rules.EDGE_NOISE, "1")
        else:
            monkeypatch.delenv(rules.EDGE_NOISE, raising=False)
        try:
            ref = F.noise_psd(ch, fs, ex, **kw)
        except ValueError:
            _raises_same(lambda: F.noise_psd(ch, fs, ex, **kw), lambda: nf.noise_psd(ch, fs, ex, **kw))
            continue
        got = nf.noise_psd(ch, fs, ex, **kw)
        assert got.dtype == ref.dtype == np.float32 and got.shape == ref.shape
        np.testing.assert_allclose(got, ref, rtol=1e-9, atol=0)


@pytest.mark.parametrize("seed", range(6))
def test_stream_fe_random(nf, seed):
    """Random push sizes, gaps and restarts (backwards too), negative
    starts and odd block sizes: chunk indices, samples and state."""
    rng = np.random.default_rng(3000 + seed)
    block = int(rng.choice([8000, 16, 17, 1000, 333, -5, 0, 100000]))
    x = rng.standard_normal(150000)
    ref, nat = live.StreamFE(block), nf.StreamFE(block)
    assert nat.block == ref.block
    pos = int(rng.integers(-1000, 1000))
    for step in range(30):
        n = int(rng.choice([0, 1, 5, 15, 16, 17, 255, 256, 257, 271, 3000, 8000, 9000]))
        if rng.random() < 0.15:
            pos += int(rng.integers(-50, 300))
        xs = x[(step * 4999) % 100000:(step * 4999) % 100000 + n]
        r_out, n_out = ref.push(pos, xs), nat.push(pos, xs)
        assert [i for i, _ in n_out] == [i for i, _ in r_out], (block, pos, n)
        for (_, fr), (_, fn) in zip(r_out, n_out):
            _fe_close(np.asarray(fn), fr)
        assert (nat.a0, nat.next_index) == (ref.a0, ref.next_index)
        _same(nat.pend, ref.pend)
        pos += n


def test_calc_oa_lens_sweep(cv):
    """scipy's block arithmetic for every overlap the receiver's filters
    could have (lambertw to the bit, so ceil() never lands differently)."""
    from scipy.signal._signaltools import _calc_oa_lens

    for s2 in range(2, 3001):
        for s1 in (10**7, 4 * s2 + 1, 2 * s2 + 1):
            assert cv.calc_oa_lens(s1, s2) == tuple(_calc_oa_lens(s1, s2)), (s1, s2)
            assert cv.calc_oa_lens(s2, s1) == tuple(_calc_oa_lens(s2, s1)), (s2, s1)


@pytest.mark.parametrize("n1,n2", [(300000, 255), (255, 300000), (400000, 121), (200000, 2001)])
@pytest.mark.parametrize("kinds", ["cr", "rc", "cc", "rr"])
def test_convolve_many_blocks(cv, n1, n2, kinds):
    """More blocks than one batch of the C++'s (the overlap carried across
    batches), the long operand either side."""
    a = _noise(n1 + 5, n1, complex_=kinds[0] == "c").astype(np.complex128 if kinds[0] == "c" else np.float64)
    b = _noise(n2 + 6, n2, complex_=kinds[1] == "c").astype(np.complex128 if kinds[1] == "c" else np.float64)
    for mode in ("same", "full"):
        x, y = cv.oaconvolve(a, b, mode), signal.oaconvolve(a, b, mode=mode)
        assert x.dtype == y.dtype and x.shape == y.shape
        assert np.max(np.abs(x - y)) <= 1e-13 * np.max(np.abs(y))


def test_maximum_filter1d_random(cv):
    rng = np.random.default_rng(5)
    for _ in range(40):
        n = int(rng.integers(1, 5000))
        x = rng.random(n) < rng.uniform(0, 0.3)
        size = int(rng.choice([1, 2, 3, 9, 17, 33, int(rng.integers(1, 3 * n + 2))]))
        _same(cv.maximum_filter1d(x, size), ndimage.maximum_filter1d(x, size))
        u = rng.integers(0, 256, n).astype(np.uint8)
        _same(cv.maximum_filter1d(u, size), ndimage.maximum_filter1d(u, size))

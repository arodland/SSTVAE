"""C++ infrastructure (native/core/qrss/{half,npsum,crc32,rational,npz})
against the Python libraries it mirrors: numpy's float16 casts and
pairwise sums, zlib.crc32, fractions.Fraction, and np.save / np.savez /
np.load -- plus the two file formats sstvae/qrss writes with them (.qrsp
and a PassResult .npz), read and written across the two implementations.

Everything here is compared bit for bit. None of it replaces a
sstvae.qrss attribute, so `--native-qrss` has no rows for it; the ctests
(native/tests/qrss/test_{half,npsum,crc32,rational,npz}.cpp) pin the
same behaviour by digest without Python. Skips when the extension is not
built.
"""

import io
import json
import math
import sys
import zipfile
import zlib
from fractions import Fraction

import numpy as np
import pytest

from qrss_helpers import Q_TEST, synthetic_full_latents
from qrss_native import require
from sstvae.qrss import constants as C
from sstvae.qrss import frame, picture
from sstvae.qrss import types as qt


@pytest.fixture(scope="module")
def nat(native):
    return require(native, "infra")


def splitmix(seed, n):
    """splitmix64 outputs 1..n (native/tests/qrss/infra_testgen.hpp)."""
    with np.errstate(over="ignore"):
        s = np.uint64(seed) + np.arange(1, n + 1, dtype=np.uint64) * np.uint64(0x9E3779B97F4A7C15)
        z = (s ^ (s >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return z ^ (z >> np.uint64(31))


def unit(z):
    return (z >> np.uint64(11)).astype(np.float64) * 2.0 ** -53


def same_bits(a, b) -> bool:
    a, b = np.asarray(a), np.asarray(b)
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


# --- binary16 ------------------------------------------------------------------------------


def test_every_half_pattern_both_ways(nat):
    h = np.arange(65536, dtype=np.uint16)
    want64 = h.view(np.float16).astype(np.float64)
    want32 = h.view(np.float16).astype(np.float32)
    assert same_bits(nat.half_to_double(h), want64)
    assert same_bits(nat.half_to_float(h), want32)
    # Back again: the identity, NaN payloads included, and what numpy says.
    assert same_bits(nat.half_from_double(want64), h)
    assert same_bits(nat.half_from_double(want64), want64.astype(np.float16).view(np.uint16))
    assert same_bits(nat.half_from_float(want32), h)


def _half_inputs():
    """~1e6 doubles: every half, the ties between neighbours and the doubles
    either side of each tie (which a float32 detour rounds to the tie and
    then the wrong way), negated; raw 64-bit patterns (NaN payloads,
    subnormal and huge doubles); and values over 2^-30..2^33."""
    pos = np.arange(0x7C00, dtype=np.uint16)
    lo = pos.view(np.float16).astype(np.float64)
    hi = np.append(lo[1:], 65536.0)
    mid = (lo + hi) / 2
    a = np.stack([lo, mid, np.nextafter(mid, 0.0), np.nextafter(mid, np.inf)], 1).reshape(-1)
    a = np.concatenate([a, -a])
    raw = splitmix(1, 500000).view(np.float64)
    z = splitmix(2, 250000)
    k = (z & np.uint64(63)).astype(np.int64) - 30
    special = np.array([0.0, -0.0, np.inf, -np.inf, np.nan, 65504.0, 65519.99, 65520.0, 2.0 ** -24,
                        2.0 ** -25, np.nextafter(2.0 ** -25, 1.0), 3 * 2.0 ** -25, 2.0 ** -14,
                        np.nextafter(2.0 ** -14, 0.0), 1 + 2.0 ** -11, 1 + 2.0 ** -11 + 2.0 ** -52,
                        5e-324, -5e-324, 1e300, -1e300, np.finfo(np.float64).max])
    return np.concatenate([a, raw, np.ldexp(unit(z) * 2.0 - 1.0, k), special])


def test_half_from_double_matches_numpy(nat):
    x = _half_inputs()
    assert x.size > 1_000_000
    with np.errstate(over="ignore"):
        want = x.astype(np.float16).view(np.uint16)
    got = nat.half_from_double(x)
    bad = np.flatnonzero(got != want)
    assert bad.size == 0, [(float(x[i]).hex(), hex(got[i]), hex(want[i])) for i in bad[:5]]


def test_half_from_double_is_not_a_float32_detour(nat):
    # One double ulp above the tie between 1 and 1 + 2^-10: rounds up from
    # the double, but to the tie (and then down, to even) through float32.
    x = np.array([1 + 2.0 ** -11 + 2.0 ** -52])
    assert x.astype(np.float16).view(np.uint16)[0] == 0x3C01
    assert x.astype(np.float32).astype(np.float16).view(np.uint16)[0] == 0x3C00
    assert nat.half_from_double(x)[0] == 0x3C01


def test_half_from_float_matches_numpy(nat):
    x = _half_inputs()
    with np.errstate(over="ignore", invalid="ignore"):
        x32 = x.astype(np.float32)
        x32 = x32[~np.isnan(x32)]   # numpy's float32 cast may use F16C, whose NaN payloads differ
        want = x32.astype(np.float16).view(np.uint16)
    got = nat.half_from_float(x32)
    bad = np.flatnonzero(got != want)
    assert bad.size == 0, [(float(x32[i]).hex(), hex(got[i]), hex(want[i])) for i in bad[:5]]


def test_half_keeps_shape(nat):
    x = np.arange(24, dtype=np.float64).reshape(2, 3, 4) / 7
    got = nat.half_from_double(x)
    assert got.shape == x.shape and same_bits(got, x.astype(np.float16).view(np.uint16))


# --- numpy's pairwise sums -----------------------------------------------------------------

LENGTHS = list(range(1, 301)) + [50600]
CAST_LENGTHS = list(range(1, 301)) + [8191, 8192, 8193, 16385, 50600, 100000]


def _values(n, seed):
    z = splitmix(seed, n)
    v = unit(z) * 2000.0 - 1000.0
    k = (z & np.uint64(63)).astype(np.int64) - 32
    return v, np.ldexp(v, k)   # one tame, one spanning 2^-32..2^31, where the order shows


def test_sum_and_mean_float64(nat):
    for n in LENGTHS:
        for x in _values(n, 1000 + n):
            assert same_bits(nat.np_sum(x), np.asarray(np.sum(x))), n
            assert same_bits(nat.np_mean(x), np.asarray(np.mean(x))), n
    rng = np.random.default_rng(5)
    x = rng.standard_normal(50600) * 10.0 ** rng.uniform(-6, 6, 50600)
    assert same_bits(nat.np_sum(x), np.asarray(np.sum(x)))
    assert same_bits(nat.np_mean(x ** 2), np.asarray(np.mean(x ** 2)))


def test_sum_and_mean_float32(nat):
    for n in LENGTHS:
        for x in _values(n, 2000 + n):
            x32 = x.astype(np.float32)
            assert same_bits(nat.np_sum(x32), np.asarray(np.sum(x32))), n
            assert same_bits(nat.np_mean(x32), np.asarray(np.mean(x32))), n


def test_sum_complex(nat):
    for n in LENGTHS:
        v, w = _values(n, 3000 + n)
        c = v + 1j * w[::-1]
        assert same_bits(nat.np_sum(c), np.asarray(np.sum(c))), n
        c64 = c.astype(np.complex64)
        assert same_bits(nat.np_sum(c64), np.asarray(np.sum(c64))), n


def test_sum_and_mean_through_a_cast(nat):
    # np.add.reduce casts in buffers of 8192 and restarts the pairwise
    # tree in each, so these differ from summing the cast array.
    for n in CAST_LENGTHS:
        v, w = _values(n, 4000 + n)
        w32 = w.astype(np.float32)
        assert same_bits(nat.np_sum_as(w32, "float64"), np.asarray(np.sum(w32, dtype=np.float64))), n
        assert same_bits(nat.np_mean_as_double(w32), np.asarray(np.mean(w32, dtype=np.float64))), n
        assert same_bits(nat.np_sum_as(w, "float32"), np.asarray(np.sum(w, dtype=np.float32))), n
        mask = (splitmix(5000 + n, n) & np.uint64(1)).astype(bool)
        assert same_bits(nat.np_mean(mask), np.asarray(np.mean(mask))), n
        assert same_bits(nat.np_mean(mask.astype(np.uint8)), np.asarray(np.mean(mask.astype(np.uint8)))), n
        assert same_bits(nat.np_sum_as(mask, "float64"), np.asarray(np.sum(mask, dtype=np.float64))), n
        ints = (splitmix(6000 + n, n) >> np.uint64(1)).astype(np.int64)
        assert same_bits(nat.np_mean(ints), np.asarray(np.mean(ints))), n
        assert same_bits(nat.np_sum_as(ints, "float64"), np.asarray(np.sum(ints, dtype=np.float64))), n


def test_the_buffer_boundary_matters(nat):
    f = np.ones(8193, np.float32)
    f[0], f[8192] = 2.0 ** 60, -(2.0 ** 60)
    assert np.sum(f, dtype=np.float64) == 7936.0 == nat.np_sum_as(f, "float64")
    assert np.sum(f.astype(np.float64)) == 7808.0 == nat.np_sum(f.astype(np.float64))


def test_sum_of_strided_input(nat):
    x = _values(301, 7)[1]
    view = x[::3]
    assert same_bits(nat.np_sum(view), np.asarray(np.sum(view)))


def test_prepare_segment_reproduced(nat):
    # picture.prepare_segment's arithmetic, done with the port's sum and
    # float16: the stored, hashed, transmitted values.
    full = synthetic_full_latents(3)
    for g in range(3):
        idx = picture.air_to_canonical(g)
        seg = full[g * picture.GROUP_LATENTS:(g + 1) * picture.GROUP_LATENTS].copy()
        seg[picture.never_sent(g)] = 0.0
        rms = np.sqrt(nat.np_mean(seg[idx] ** 2))
        got = nat.half_from_double(seg / rms)
        assert same_bits(got, picture.prepare_segment(full, g).view(np.uint16)), g


# --- zlib.crc32 ----------------------------------------------------------------------------


def test_crc32_matches_zlib(nat):
    rng = np.random.default_rng(9)
    for n in [0, 1, 2, 3, 7, 8, 15, 16, 255, 256, 1000, 4096, 100003]:
        data = rng.integers(0, 256, n, dtype=np.uint8).tobytes()
        for start in (0, 1, 0xDEADBEEF, 0xFFFFFFFF):
            assert nat.crc32(data, start) == zlib.crc32(data, start), (n, start)
        cut = n // 3
        assert nat.crc32(data[cut:], nat.crc32(data[:cut])) == zlib.crc32(data)
    # zlib masks the start value to 32 bits, negative or wide.
    for start in (-1, -12345, 2 ** 40 + 5):
        assert nat.crc32(b"QRSS", start) == zlib.crc32(b"QRSS", start)


# --- fractions.Fraction ---------------------------------------------------------------------


def _fr(r):
    return Fraction(r.numerator, r.denominator)


def test_limit_denominator_over_the_band(nat):
    rng = np.random.default_rng(13)
    xs = np.concatenate([rng.uniform(300.0, 2700.0, 100000), np.round(rng.uniform(300, 2700, 2000), 1),
                         np.round(rng.uniform(300, 2700, 2000), 4), [1500.3, 1234.5678, 300.0, 2700.0]])
    for max_den in (1 << 20,):
        for x in xs.tolist():
            want = Fraction(x).limit_denominator(max_den)
            r = nat.Rational.from_float(x).limit_denominator(max_den)
            assert (r.numerator, r.denominator) == (want.numerator, want.denominator), x
            r2 = nat.Rational.from_float_limited(x, max_den)
            assert (r2.numerator, r2.denominator) == (want.numerator, want.denominator), x
    for max_den in (1 << 16, 1000000, 1, 7, 1000):
        for x in xs[:5000].tolist():
            want = Fraction(x).limit_denominator(max_den)
            r = nat.Rational.from_float_limited(x, max_den)
            assert (r.numerator, r.denominator) == (want.numerator, want.denominator), (x, max_den)
    assert str(nat.Rational.from_float(1500.3).limit_denominator(1 << 20)) == "15003/10"
    assert str(nat.Rational.from_float(1234.5678).limit_denominator(1 << 20)) == "6172839/5000"


def test_limit_denominator_of_small_and_negative_floats(nat):
    # Exact ratios wider than int64 (denominator 2^63 and up): CPython's
    # loop on wide integers, ties included.
    z = splitmix(21, 20000)
    k = (z % np.uint64(1085)).astype(np.int64) - 1074
    sign = np.where((z >> np.uint64(63)) == 1, -1.0, 1.0)
    xs = np.concatenate([sign * np.ldexp(unit(z) + 0.5, k), [2.0 ** -63, -(2.0 ** -63), 5e-324, -5e-324,
                                                              3 * 2.0 ** -70, 7e-4, -7e-4, 1e-10]])
    dens = [1, 2, 7, 1 << 16, 1 << 20, 10 ** 6, 2 ** 62, 2 ** 63 - 1]
    for i, x in enumerate(xs.tolist()):
        for max_den in (dens if i >= len(xs) - 8 else [dens[i % len(dens)]]):
            want = Fraction(x).limit_denominator(max_den)
            r = nat.Rational.from_float_limited(x, max_den)
            assert (r.numerator, r.denominator) == (want.numerator, want.denominator), (x.hex(), max_den)


def test_limit_denominator_errors(nat):
    with pytest.raises(ValueError, match="max_denominator should be at least 1"):
        Fraction(1, 3).limit_denominator(0)
    with pytest.raises(ValueError, match="max_denominator should be at least 1"):
        nat.Rational(1, 3).limit_denominator(0)
    with pytest.raises(ValueError, match="max_denominator should be at least 1"):
        nat.Rational.from_float_limited(1e-300, 0)
    for f in (Fraction, nat.Rational.from_float, lambda x: nat.Rational.from_float_limited(x, 0)):
        with pytest.raises(ValueError, match="cannot convert NaN to integer ratio"):
            f(float("nan"))
        with pytest.raises(OverflowError, match="cannot convert Infinity to integer ratio"):
            f(float("-inf"))


def test_from_float_is_exact(nat):
    for x in [1500.3, 0.1, -2.5, 0.0, -0.0, 1e18, 2.0 ** 62, 2.0 ** -62, 300.123456789, -2699.999]:
        assert _fr(nat.Rational.from_float(x)) == Fraction(x), x
    with pytest.raises(OverflowError):
        nat.Rational.from_float(5e-324)   # 2^-1074: past int64, refused rather than wrong


PARSE_OK = ["3/2", " -35/4 ", "3.1415", "-47e-2", "1_000", "1.e5", ".5", "+.5e-3", "0.00000095367431640625",
            "1E+0_1", "0e-99999", "1500.300000000000000000000000", "-0", "007/0014", "\t96017/64\n",
            "1 /2", "1/ 2", " 1 / 2 ", "12207/970", "1500.25", "-1500.25e0", "9223372036854775807",
            "-9223372036854775807", "0.000", "5e18", "1/9223372036854775807",
            "\u00a01/2\u3000", "\u20021.5\u2028", "\x1c3\x1f", "\x85-7\x85"]
PARSE_BAD = ["1._5", "1__2", "1_", "1e5/2", "1.5/2", "nan", "inf", "1 . 5", "", " ", "/2", ".", "e5", "1/",
             "1/-2", "--1", "0x10", "1/2/3", "'a'", "\x01\t", "1/2 3", "+", "1e", "1e+", "1.5e-_3",
             "_1", "1\u200b", "\xa01/2/3", "1/2\u2028x", "\U000e0001x", "\u00e91", "1'\"",
             "\x85\x9f", "1\ufeff2", "\u30001\u3000\u3000/"]


@pytest.mark.parametrize("text", PARSE_OK)
def test_parse_matches_fraction(nat, text):
    want = Fraction(text)
    r = nat.Rational.parse(text)
    assert (r.numerator, r.denominator) == (want.numerator, want.denominator)
    assert str(r) == str(want)


@pytest.mark.parametrize("text", PARSE_BAD)
def test_parse_refuses_as_fraction(nat, text):
    with pytest.raises(ValueError) as want:
        Fraction(text)
    with pytest.raises(ValueError) as got:
        nat.Rational.parse(text)
    assert str(got.value) == str(want.value)


@pytest.mark.parametrize("text", ["5/0", "-5/0", "0/0", " 7 / 0 "])
def test_parse_zero_denominator(nat, text):
    with pytest.raises(ZeroDivisionError) as want:
        Fraction(text)
    with pytest.raises(ZeroDivisionError) as got:
        nat.Rational.parse(text)
    assert str(got.value) == str(want.value)


def test_parse_beyond_int64_is_refused(nat):
    for text in ["1e30", "92233720368547758070", "1/92233720368547758070", "1e-30"]:
        with pytest.raises(OverflowError):
            nat.Rational.parse(text)


def test_str_parse_round_trip_like_passresult(nat):
    # PassResult writes str(Fraction(f_mix_hz)) and reads Fraction(text).
    for f in [Fraction(96017, 64), Fraction(-3, 7), Fraction(5), Fraction(0), Fraction(12207, 970)]:
        r = nat.Rational(f.numerator, f.denominator)
        assert str(r) == str(f)
        assert _fr(nat.Rational.parse(str(f))) == Fraction(str(f))


def _pairs(seed, n, lim):
    rng = np.random.default_rng(seed)
    nums = rng.integers(-lim, lim, (n, 2))
    dens = rng.integers(1, lim, (n, 2))
    return [(Fraction(int(a), int(b)), Fraction(int(c), int(d)))
            for (a, c), (b, d) in zip(nums.tolist(), dens.tolist())]


def test_arithmetic_and_comparison_match_fraction(nat):
    for a, b in _pairs(17, 20000, 10 ** 9) + _pairs(18, 2000, 50):
        ra, rb = nat.Rational(a.numerator, a.denominator), nat.Rational(b.numerator, b.denominator)
        assert _fr(ra + rb) == a + b
        assert _fr(ra - rb) == a - b
        assert _fr(ra * rb) == a * b
        if b:
            assert _fr(ra / rb) == a / b
        assert _fr(-ra) == -a
        assert ((ra < rb), (ra <= rb), (ra == rb), (ra > rb), (ra >= rb)) == \
            ((a < b), (a <= b), (a == b), (a > b), (a >= b))
        assert str(ra) == str(a) and float(ra) == float(a)
        assert (int(ra), math.floor(ra), math.ceil(ra), round(ra)) == (int(a), math.floor(a), math.ceil(a), round(a))
        assert hash(ra) == hash(a)


def test_float_is_correctly_rounded(nat):
    rng = np.random.default_rng(19)
    nums = rng.integers(-(2 ** 63) + 1, 2 ** 63 - 1, 20000, dtype=np.int64, endpoint=True)
    dens = rng.integers(1, 2 ** 63 - 1, 20000, dtype=np.int64, endpoint=True)
    shift = rng.integers(0, 63, 20000)
    for n, d, s in zip(nums.tolist(), dens.tolist(), shift.tolist()):
        d = max(1, d >> s)
        assert float(nat.Rational(n, d)) == float(Fraction(n, d)), (n, d)
    for n, d in [(9007199254740993, 1), (9007199254740995, 1), (2 ** 63 - 1, 2 ** 62 + 1), (1, 3), (-2, 3)]:
        assert float(nat.Rational(n, d)) == float(Fraction(n, d))


def test_rounding_ties(nat):
    for n in range(-21, 22):
        for d in (1, 2, 3, 4, 6):
            f, r = Fraction(n, d), nat.Rational(n, d)
            assert (int(r), math.floor(r), math.ceil(r), round(r)) == \
                (int(f), math.floor(f), math.ceil(f), round(f)), (n, d)


def test_zero_division_and_range(nat):
    for make in (Fraction, nat.Rational):
        with pytest.raises(ZeroDivisionError, match=r"^Fraction\(1, 0\)$"):
            make(1, 0)
    with pytest.raises(ZeroDivisionError) as want:
        Fraction(96017, 64) / Fraction(0, 7)
    with pytest.raises(ZeroDivisionError) as got:
        nat.Rational(96017, 64) / nat.Rational(0, 7)
    assert str(got.value) == str(want.value)
    with pytest.raises(OverflowError):
        nat.Rational(2 ** 63, 1)
    big = 2 ** 63 - 1
    with pytest.raises(OverflowError):
        nat.Rational(big, 3) * nat.Rational(big, 5)
    assert (nat.Rational(big - 1, big) > nat.Rational(big - 2, big - 1)) == \
        (Fraction(big - 1, big) > Fraction(big - 2, big - 1))


# --- np.save / np.savez / np.load -----------------------------------------------------------

DTYPES = ["<f2", "<f4", "<f8", "<c8", "<c16", "|i1", "|u1", "<i2", "<u2", "<i4", "<u4", "<i8", "<u8", "|b1"]
SHAPES = [(), (0,), (1,), (5,), (2, 3), (2, 3, 4), (3, 0, 2)]


def _sample(dt, shape, seed):
    rng = np.random.default_rng(seed)
    n = int(np.prod(shape, dtype=np.int64))
    dtype = np.dtype(dt)
    if dtype.kind == "b":
        a = rng.integers(0, 2, n).astype(bool)
    else:
        # Every bit pattern, NaNs and infinities among them for the floats.
        a = np.frombuffer(rng.integers(0, 256, n * dtype.itemsize, dtype=np.uint8).tobytes(), dtype=dtype)
    return a.reshape(shape).copy()


def _arrays(dt):
    return {f"a{i}": _sample(dt, shape, 100 * i + len(dt)) for i, shape in enumerate(SHAPES)}


def _savez(arrays) -> bytes:
    buf = io.BytesIO()
    np.savez(buf, **arrays)
    return buf.getvalue()


def _same_arrays(got: dict, want: dict):
    assert list(got) == list(want)
    for k in want:
        g, w = np.asarray(got[k]), np.asarray(want[k])
        assert g.dtype.str == w.dtype.str and g.shape == w.shape and g.tobytes() == w.tobytes(), k


@pytest.mark.parametrize("dt", DTYPES)
def test_numpy_writes_native_reads(nat, dt):
    arrays = _arrays(dt)
    _same_arrays(nat.parse_npz(_savez(arrays)), arrays)
    for a in arrays.values():
        buf = io.BytesIO()
        np.save(buf, a)
        assert same_bits(nat.parse_npy(buf.getvalue()), a)


@pytest.mark.parametrize("dt", DTYPES)
def test_native_writes_numpy_reads(nat, dt):
    arrays = _arrays(dt)
    data = nat.npz_bytes(arrays)
    with np.load(io.BytesIO(data), allow_pickle=False) as z:
        _same_arrays({k: z[k] for k in z.files}, arrays)
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        assert zf.testzip() is None
    for a in arrays.values():
        assert same_bits(np.load(io.BytesIO(nat.npy_bytes(a)), allow_pickle=False), a)


@pytest.mark.skipif(sys.platform == "win32", reason="zipfile records create_system 0 on Windows, the port 3")
@pytest.mark.parametrize("dt", DTYPES)
def test_native_writes_numpys_bytes(nat, dt):
    # zipfile stamps a member opened for writing 1980-01-01 00:00, so
    # np.savez is reproducible, and the port writes the same file.
    arrays = _arrays(dt)
    assert nat.npz_bytes(arrays) == _savez(arrays)
    for a in arrays.values():
        buf = io.BytesIO()
        np.save(buf, a)
        assert nat.npy_bytes(a) == buf.getvalue()


def test_fortran_order_is_read_in_c_order(nat):
    a = np.asfortranarray(np.arange(24, dtype=np.int32).reshape(2, 3, 4))
    buf = io.BytesIO()
    np.save(buf, a)
    assert b"'fortran_order': True" in buf.getvalue()[:128]
    got = nat.parse_npy(buf.getvalue())
    assert got.shape == a.shape and np.array_equal(got, a) and got.flags.c_contiguous
    # And written from Fortran order: the logical array, in C order.
    assert np.array_equal(np.load(io.BytesIO(nat.npy_bytes(a))), a)


def test_long_header_and_shapes(nat):
    a = np.zeros((1, 2, 3), np.float32)
    for shape in [(10 ** 9, 0), (0, 10 ** 12), (7,) * 3]:
        b = np.zeros(shape, np.uint8)
        buf = io.BytesIO()
        np.save(buf, b)
        assert nat.npy_bytes(b) == buf.getvalue()
        assert same_bits(nat.parse_npy(buf.getvalue()), b)
    assert same_bits(nat.parse_npy(nat.npy_bytes(a)), a)


def test_refusals(nat):
    buf = io.BytesIO()
    np.save(buf, np.arange(3, dtype=">f8"))
    with pytest.raises(ValueError, match="not supported"):
        nat.parse_npy(buf.getvalue())
    buf = io.BytesIO()
    np.savez_compressed(buf, a=np.arange(3, dtype=np.int16))
    with pytest.raises(ValueError, match="compressed"):
        nat.parse_npz(buf.getvalue())
    good = _savez({"a": np.arange(10, dtype=np.int64)})
    bad = bytearray(good)
    bad[good.index(b"\x93NUMPY") + 130] ^= 1
    with pytest.raises(ValueError, match="bad CRC-32"):
        nat.parse_npz(bytes(bad))
    with pytest.raises(zipfile.BadZipFile, match="Bad CRC-32"):
        np.load(io.BytesIO(bytes(bad)))["a"]
    with pytest.raises(ValueError, match="not a zip file"):
        nat.parse_npz(good[:-1])
    with pytest.raises(ValueError, match="magic string"):
        nat.parse_npy(b"\x93NUMPX" + good[6:])
    with pytest.raises(ValueError):
        nat.npz_bytes({"a": np.zeros(2, dtype=[("x", "<f4")])})
    with pytest.raises(ValueError):
        nat.npy_bytes(np.array(["text"]))


def test_files_both_ways(nat, tmp_path):
    arrays = {"version": np.int64(1), "z": np.linspace(0, 1, 7, dtype=np.float32), "flag": np.bool_(True)}
    np.savez(tmp_path / "py.npz", **arrays)
    _same_arrays(nat.load_npz(str(tmp_path / "py.npz")), {k: np.asarray(v) for k, v in arrays.items()})
    nat.save_npz(str(tmp_path / "cpp.npz"), arrays)
    with np.load(tmp_path / "cpp.npz", allow_pickle=False) as z:
        _same_arrays({k: z[k] for k in z.files}, {k: np.asarray(v) for k, v in arrays.items()})
    np.save(tmp_path / "py.npy", arrays["z"])
    assert same_bits(nat.load_npy(str(tmp_path / "py.npy")), arrays["z"])
    nat.save_npy(str(tmp_path / "cpp.npy"), arrays["z"])
    assert same_bits(np.load(tmp_path / "cpp.npy"), arrays["z"])
    assert nat.npz_to_int64(arrays["version"]) == 1
    assert nat.npz_to_double(np.arange(3, dtype=np.float16)).tolist() == [0.0, 1.0, 2.0]


# --- the two QRSS formats -------------------------------------------------------------------


@pytest.mark.parametrize("mode", [0, 2])
def test_qrsp_both_ways(nat, tmp_path, mode):
    sp = picture.StoredPicture.from_latents(synthetic_full_latents(mode), 0xD1D8, mode)
    picture.save_qrsp(tmp_path / "py.qrsp", sp)
    got = nat.load_npz(str(tmp_path / "py.qrsp"))
    assert list(got) == ["version", "mode", "codec_id", "segs", "picture_id"]
    for k, v in [("version", picture.QRSP_VERSION), ("mode", mode), ("codec_id", 0xD1D8),
                 ("picture_id", sp.picture_id)]:
        assert got[k].dtype == np.int64 and got[k].shape == () and nat.npz_to_int64(got[k]) == v, k
    assert got["segs"].dtype == np.float16 and got["segs"].shape == (mode + 1, picture.GROUP_LATENTS)
    assert np.array_equal(got["segs"].view(np.uint16), np.stack(sp.segs).view(np.uint16))
    # Written by the port, the reference reads it and the picture ID checks.
    nat.save_npz(str(tmp_path / "cpp.qrsp"), got)
    back = picture.load_qrsp(tmp_path / "cpp.qrsp")
    assert back.picture_id == sp.picture_id and back.mode == mode
    if sys.platform != "win32":
        assert (tmp_path / "cpp.qrsp").read_bytes() == (tmp_path / "py.qrsp").read_bytes()


def test_qrsp_like_uint16_file(nat):
    segs = (splitmix(31, 2 * 528) & np.uint64(0xFFFF)).astype(np.uint16).reshape(2, 528)
    arrays = {"version": np.int64(1), "mode": np.int64(1), "codec_id": np.int64(0xD1D8), "segs": segs,
              "picture_id": np.int64(0xD349C671)}
    _same_arrays(nat.parse_npz(_savez(arrays)), {k: np.asarray(v) for k, v in arrays.items()})
    with np.load(io.BytesIO(nat.npz_bytes(arrays)), allow_pickle=False) as z:
        assert int(z["picture_id"]) == 0xD349C671 and np.array_equal(z["segs"], segs)


def _fake_pass():
    # As tests/test_qrss_format.py builds one.
    n = 4096
    rng = np.random.default_rng(11)
    z = rng.standard_normal(n).astype(np.float32)
    return qt.PassResult(
        uid=qt.pass_uid(Q_TEST, 1500.25, z), q=Q_TEST, frame="short", waveform=0, f_hz=1500.25,
        timing=qt.Timing(tau0=3000.5, ppm=12.0, gamma=0.0, z=40.0, cov=np.eye(2) * 1e-3),
        report=qt.TrackReport(offset_hz=0.25, drift_hz_per_min=0.1, wander_hz_rms=0.02, doppler_hz=0.1,
                              ppm=12.0, snr2500_db=-12.0, z_ref=40.0, kappa=1.02, slip_free=True,
                              suspect=False),
        z=z, w=np.full(n, 3.0, np.float32), hdr_llr=np.zeros(2474, np.float32), header=None,
        cw=qt.CwIdResult(text="K1ABC", z_match=9.5, keying="fsk", agrees=True),
        ch=(rng.standard_normal(1000) + 1j).astype(np.complex64), ch_fs=C.CH_FS, ch_t0_index=3000.5,
        f_mix_hz=Fraction(96017, 64), psi=np.ones(frame.SHORT.n_pos, np.float32), estimator="joint",
    )


def test_pass_result_both_ways(nat, tmp_path):
    p = _fake_pass()
    p.save(tmp_path / "py.npz")
    got = nat.load_npz(str(tmp_path / "py.npz"))
    assert list(got) == ["meta", "z", "w", "hdr_llr", "ch", "psi", "timing_cov"]
    assert got["meta"].dtype == np.uint8
    meta = json.loads(got["meta"].tobytes().decode())
    assert meta["uid"] == p.uid and meta["f_mix_hz"] == "96017/64"
    assert str(nat.Rational.parse(meta["f_mix_hz"])) == "96017/64"
    for k, want in [("z", p.z), ("w", p.w), ("hdr_llr", p.hdr_llr), ("ch", p.ch), ("psi", p.psi),
                    ("timing_cov", p.timing.cov)]:
        assert same_bits(got[k], np.asarray(want)), k
    nat.save_npz(str(tmp_path / "cpp.npz"), got)
    back = qt.PassResult.load(tmp_path / "cpp.npz")
    assert back.uid == p.uid and back.f_mix_hz == p.f_mix_hz and back.report == p.report
    for k in ("z", "w", "hdr_llr", "ch", "psi"):
        assert same_bits(getattr(back, k), getattr(p, k)), k
    if sys.platform != "win32":
        assert (tmp_path / "cpp.npz").read_bytes() == (tmp_path / "py.npz").read_bytes()

"""C++ precoder (native/core/qrss/precoder) against sstvae.qrss.precoder.

Bit-exact: the C++ runs numpy's butterflies in numpy's order (h ascending,
x + y then x - y, then a division by sqrt(M)), so every output is compared
by its bit pattern, signed zeros included. On fp16-grid latents every step
is exact anyway; on int8/20 latents and on arbitrary doubles only the
same order gives the same roundings. Skips when the extension is not
built; `pytest --native-qrss=tx` (and `=rx`, for the receive side) runs
the rest of the QRSS suites against the same code.
"""

import numpy as np
import pytest

from qrss_native import require
from sstvae.qrss import precoder as ref

Q_TEST = 1954752
LENGTHS = list(range(1, 201)) + [1024, 4096, 16384, 50600]


class _Weird(int):
    """An int whose str() is not its value: int() of it is a plain int."""

    def __str__(self):
        return "weird"

    __repr__ = __str__


@pytest.fixture(scope="module")
def nat(native):
    return require(native, "precoder")


def _bits_equal(got, want):
    """Same shape and dtype, NaN where NaN, and identical bits elsewhere."""
    got, want = np.asarray(got), np.asarray(want)
    if got.shape != want.shape or got.dtype != want.dtype:
        return False
    gn, wn = np.isnan(got), np.isnan(want)
    if not np.array_equal(gn, wn):
        return False
    return np.array_equal(got[~gn].view(np.int64), want[~wn].view(np.int64))


def _inputs(n, seed):
    """fp16-grid, int8/20 (picture.from_int8) and arbitrary float64 vectors."""
    rng = np.random.default_rng(seed)
    fp16 = rng.standard_normal(n).astype(np.float16).astype(np.float64)
    int8 = rng.integers(-127, 128, n).astype(np.float64) / 20
    wide = rng.standard_normal(n) * 10.0 ** rng.uniform(-3, 3, n)
    return {"fp16": fp16, "int8": int8, "wide": wide}


def test_block_constant(nat):
    assert nat.BLOCK == ref.BLOCK


@pytest.mark.parametrize("n", list(range(0, 300)) + [777, 1024, 4096, 16384, 50600, 50601, 10**6 + 3])
def test_block_sizes_and_runs(nat, n):
    assert nat.block_sizes(n) == ref.block_sizes(n)
    assert nat._runs(n) == ref._runs(n)


def test_block_sizes_spec_values(nat):
    assert nat.block_sizes(50600) == [64] * 790 + [32, 8]
    assert nat.block_sizes(16384) == [64] * 256
    assert nat.block_sizes(4096) == [64] * 64
    assert nat.block_sizes(63) == [32, 16, 8, 4, 2, 1]


def test_count_arguments_are_read_as_int_does(nat):
    for n in (np.int64(777), np.uint16(100), 64.9, True, "130"):
        assert nat.block_sizes(n) == ref.block_sizes(n), repr(n)
        assert np.array_equal(nat.block_index(n), ref.block_index(n)), repr(n)
    for bad in (-1, np.int64(-5), -0.5 - 1, _Weird(-3)):
        with pytest.raises(ValueError, match="n must be >= 0") as want:
            ref.block_sizes(bad)
        with pytest.raises(ValueError, match="n must be >= 0") as got:
            nat.block_sizes(bad)
        assert str(got.value) == str(want.value)
        with pytest.raises(ValueError, match="n must be >= 0"):
            nat.block_index(bad)
    for bad in (None, "x", [3]):
        with pytest.raises((TypeError, ValueError)) as want:
            ref.block_sizes(bad)
        with pytest.raises(want.type):
            nat.block_sizes(bad)


def test_block_sizes_overflow_is_refused_as_python_does(nat):
    with pytest.raises(OverflowError):
        ref.block_sizes(2**80)
    with pytest.raises(OverflowError):
        nat.block_sizes(2**80)


@pytest.mark.parametrize("n", LENGTHS)
def test_wht_precode_unprecode_bitwise(nat, n):
    for kind, x in _inputs(n, seed=n).items():
        assert _bits_equal(nat.wht(x), ref.wht(x)), (kind, n)
        assert _bits_equal(nat.precode(x, Q_TEST), ref.precode(x, Q_TEST)), (kind, n)
        assert _bits_equal(nat.unprecode(x, Q_TEST), ref.unprecode(x, Q_TEST)), (kind, n)


@pytest.mark.parametrize("n", [1, 7, 8, 40, 64, 100, 777, 4096, 50600])
def test_block_mean_and_index_bitwise(nat, n):
    for kind, x in _inputs(n, seed=1000 + n).items():
        assert _bits_equal(nat.block_mean(x), ref.block_mean(x)), (kind, n)
    got, want = nat.block_index(n), ref.block_index(n)
    assert got.dtype == want.dtype == np.int64
    assert np.array_equal(got, want)


def test_block_mean_ramp(nat):
    v = np.arange(50600, dtype=float)
    assert _bits_equal(nat.block_mean(v), ref.block_mean(v))
    assert nat.block_mean(v)[0] == 31.5


@pytest.mark.parametrize("shape", [(3, 4096), (2, 3, 777), (5, 50600), (1, 1), (4, 0), (0, 64), (0,)])
def test_batch_rows(nat, shape):
    rng = np.random.default_rng(sum(shape) + len(shape))
    x = rng.standard_normal(shape)
    for fn in ("wht", "block_mean"):
        assert _bits_equal(getattr(nat, fn)(x), getattr(ref, fn)(x)), fn
    for q in (0, Q_TEST, 2**40 + 12345, 2**64 - 1):
        assert _bits_equal(nat.precode(x, q), ref.precode(x, q)), q
        assert _bits_equal(nat.unprecode(x, q), ref.unprecode(x, q)), q


def test_input_dtypes_and_layouts(nat):
    rng = np.random.default_rng(7)
    base = rng.standard_normal((3, 200))
    fortran = np.asfortranarray(base)
    cases = [
        base.astype(np.float32), base.astype(np.float16), np.clip(base * 40, -127, 127).astype(np.int8),
        (base * 1e6).astype(np.int64), base > 0, fortran, base[:, ::2], base[::-1], base[:, ::-1],
        base.tolist(), base[0].tolist(), list(range(70)),
    ]
    ro = base.copy()
    ro.setflags(write=False)
    cases.append(ro)
    for x in cases:
        assert _bits_equal(nat.wht(x), ref.wht(x))
        assert _bits_equal(nat.precode(x, Q_TEST), ref.precode(x, Q_TEST))
        assert _bits_equal(nat.unprecode(x, Q_TEST), ref.unprecode(x, Q_TEST))
        if x is not fortran:
            assert _bits_equal(nat.block_mean(x), ref.block_mean(x))
    assert not ro.flags.writeable and np.array_equal(ro, base)   # inputs untouched
    y = base.copy()
    nat.wht(y)
    assert np.array_equal(y, base)


@pytest.mark.parametrize("x", [np.asfortranarray(np.random.default_rng(8).standard_normal((3, 200))),
                               np.random.default_rng(9).standard_normal((64, 5)).T,
                               np.asfortranarray(np.random.default_rng(10).standard_normal((2, 3, 777)))])
def test_block_mean_of_a_batch_laid_out_inside_out(nat, x):
    """The one stated deviation. When a batch's leading axes are laid out
    inside its last one (Fortran order, a transpose), numpy's reduction
    iterates the batch axis innermost and sums each block left to right
    instead of pairwise, so the reference's last bit depends on memory
    layout; the C++ sums pairwise, as numpy does for every C-ordered input.
    No caller passes such a batch (EM's is 1-D)."""
    got, want = nat.block_mean(x), ref.block_mean(x)
    assert got.shape == want.shape
    assert np.allclose(got, want, rtol=1e-14, atol=1e-15)
    assert _bits_equal(got, ref.block_mean(np.ascontiguousarray(x)))


@pytest.mark.filterwarnings("ignore:overflow encountered", "ignore:invalid value encountered")
def test_special_values(nat):
    # numpy warns on the overflows below; the C++ computes the same IEEE
    # results without a warning.
    x = np.array([0.0, -0.0, np.inf, -np.inf, np.nan, 1e308, -1e308, 5e-324] * 8)
    for v in (x, -x, x[::-1], np.full(64, -0.0), np.full(8, 1e308)):
        assert _bits_equal(nat.wht(v), ref.wht(v))
        assert _bits_equal(nat.precode(v, Q_TEST), ref.precode(v, Q_TEST))
        assert _bits_equal(nat.unprecode(v, Q_TEST), ref.unprecode(v, Q_TEST))
        assert _bits_equal(nat.block_mean(v), ref.block_mean(v))


def test_slot_counts_are_read_as_int_does(nat):
    x = _inputs(100, 3)["wide"]
    for q in (np.int64(Q_TEST), np.uint64(2**64 - 1), float(Q_TEST) + 0.75, True, _Weird(5), "77"):
        assert _bits_equal(nat.precode(x, q), ref.precode(x, q)), repr(q)
        assert _bits_equal(nat.unprecode(x, q), ref.unprecode(x, q)), repr(q)


@pytest.mark.parametrize("q", [-1, 2**64, -0.5 - 1, _Weird(-1), np.int64(-7)])
def test_q_outside_uint64_is_refused(nat, q):
    x = np.ones(8)
    for fn in ("precode", "unprecode"):
        with pytest.raises(ValueError, match="q must fit in uint64") as want:
            getattr(ref, fn)(x, q)
        with pytest.raises(ValueError, match="q must fit in uint64") as got:
            getattr(nat, fn)(x, q)
        assert str(got.value) == str(want.value)


def test_zero_dimensional_input_is_refused(nat):
    for fn, args in (("wht", ()), ("precode", (Q_TEST,)), ("unprecode", (Q_TEST,)), ("block_mean", ())):
        with pytest.raises(IndexError):
            getattr(ref, fn)(np.float64(1.0), *args)
        with pytest.raises(IndexError):
            getattr(nat, fn)(np.float64(1.0), *args)


def test_definitions_hold(nat):
    """x = WHT(c_q * a), a = c_q * WHT(x), and the WHT is its own inverse."""
    from sstvae.qrss.sequences import scrambler

    a = _inputs(50600, 11)["wide"]
    x = nat.precode(a, Q_TEST)
    assert np.array_equal(x, nat.wht(scrambler(Q_TEST, 50600) * a))
    assert np.max(np.abs(nat.unprecode(x, Q_TEST) - a)) < 1e-12
    assert np.max(np.abs(nat.wht(nat.wht(a)) - a)) < 1e-12

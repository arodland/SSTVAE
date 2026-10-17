"""C++ polar code (native/core/qrss/polar) against sstvae.qrss.polar.

The encoder is integer arithmetic and must agree bit for bit. The CA-SCL
decoder repeats the reference's floating-point operations in the same
order (min-sum f, exact g, numpy's logaddexp and pairwise row sum, a
stable sort with NaN last), so on one machine its path metrics come out
bit-identical; the bar asserted is identical decisions -- list paths, their
order, accept/reject -- and path metrics within 1e-9 relative. Skips when
the extension is not built; `pytest --native-qrss=rx` runs the header
suites against the same code.
"""

import numpy as np
import pytest

from qrss_native import require
from sstvae.qrss import header as ref_header
from sstvae.qrss import polar as ref
from sstvae.qrss.header import HeaderFields

H = HeaderFields("K1ABC/P", "FN42", 0xDEADBEEF, 2, 1, 0xD1D8)


@pytest.fixture(scope="module")
def nat(native):
    return require(native, "polar")


def _outcome(f, *args, **kw):
    """What a call gives: ("ok", result) or ("raise", type, text)."""
    try:
        return ("ok", f(*args, **kw))
    except Exception as e:      # noqa: BLE001 -- any refusal is compared
        return ("raise", type(e), str(e))


def _same_outcome(f_ref, f_nat, *args, **kw):
    """The same refusal (type and text), or results equal array by array."""
    want, got = _outcome(f_ref, *args, **kw), _outcome(f_nat, *args, **kw)
    assert got[0] == want[0], (want, got)
    if want[0] == "raise":
        assert got[1:] == want[1:]
        return want
    w, g = want[1], got[1]
    if w is None or g is None:
        assert w is None and g is None
        return want
    for a, b in zip(w if isinstance(w, tuple) else (w,), g if isinstance(g, tuple) else (g,)):
        assert a.dtype == b.dtype and a.shape == b.shape and np.array_equal(a, b)
    return want


def _same_error(f_ref, f_nat, *args, **kw):
    """Both raise the same exception type with the same text."""
    with pytest.raises(Exception) as want:
        f_ref(*args, **kw)
    with pytest.raises(type(want.value)) as got:
        f_nat(*args, **kw)
    assert str(got.value) == str(want.value)


def test_constants(nat):
    assert (nat.N, nat.K, nat.E, nat.N_STAGES) == (ref.N, ref.K, ref.E, ref.N_STAGES)
    assert nat.INFO_SET == ref.INFO_SET
    assert nat.info_set_ok()


# --- encoder ------------------------------------------------------------------


@pytest.mark.parametrize("shape", [(2048,), (3, 2048), (2, 2, 2048), (6,), (12,), (1,), (0,), (4, 0), (64,),
                                   (0, 10), (0, 3), (3, 0, 5), (0, 2048)])
def test_polar_transform(nat, shape):
    rng = np.random.default_rng(1)
    for high in (2, 256):                     # bits, and any byte (XORed)
        u = rng.integers(0, high, shape).astype(np.uint8)
        got, want = nat.polar_transform(u), ref.polar_transform(u)
        assert got.dtype == want.dtype == np.uint8 and got.shape == want.shape
        assert np.array_equal(got, want)


def test_polar_transform_conversions(nat):
    rng = np.random.default_rng(2)
    u = rng.integers(0, 2, (3, 2048))
    for v in (u, u.astype(bool), u.astype(np.float32), np.asfortranarray(u), u.tolist(), u[:, ::-1]):
        assert np.array_equal(nat.polar_transform(v), ref.polar_transform(v))
    x = u.astype(np.uint8)
    before = x.copy()
    nat.polar_transform(x)
    assert np.array_equal(x, before), "the input is not modified"


@pytest.mark.parametrize("bad", [np.zeros(5, np.uint8), np.zeros((3, 10), np.uint8), np.zeros(3, np.uint8),
                                 np.uint8(3), [-1]])
def test_polar_transform_refusals(nat, bad):
    _same_error(ref.polar_transform, nat.polar_transform, bad)


@pytest.mark.parametrize("dtype", [np.uint8, np.int64, np.float64, np.complex128, np.float16, bool, "U3", object])
def test_rate_match(nat, dtype):
    rng = np.random.default_rng(3)
    for shape in ((2048,), (2, 2048), (3, 1, 2050)):
        x = rng.integers(0, 7, shape).astype(dtype)
        got, want = nat.rate_match(x), ref.rate_match(x)
        assert got.dtype == want.dtype and got.shape == want.shape
        assert np.array_equal(got, want)


@pytest.mark.parametrize("bad", [np.zeros(10), np.zeros((2, 100)), np.float64(1.0), np.zeros(2047)])
def test_rate_match_refusals(nat, bad):
    _same_error(ref.rate_match, nat.rate_match, bad)


def test_fold(nat):
    rng = np.random.default_rng(4)
    for scale in (1e-3, 1.0, 1e3, 1e300):
        llr = rng.normal(0.0, scale, 2474)
        got, want = nat.fold(llr), ref.fold(llr)
        assert got.dtype == want.dtype == np.float64
        assert np.array_equal(got.view(np.int64), want.view(np.int64))
    for llr in (rng.normal(0, 1, 2474).astype(np.float32), list(range(2474)), np.arange(2474)):
        assert np.array_equal(nat.fold(llr), ref.fold(llr))
    neg0 = np.zeros(2474)
    neg0[[0, 2048, 3]] = -0.0
    assert np.array_equal(np.signbit(nat.fold(neg0)), np.signbit(ref.fold(neg0)))


@pytest.mark.parametrize("bad", [np.zeros(2048), np.zeros((2, 2474)), np.float64(0.0), np.zeros(2475), []])
def test_fold_refusals(nat, bad):
    _same_error(ref.fold, nat.fold, bad)


def test_polar_encode_random(nat):
    rng = np.random.default_rng(5)
    info = rng.integers(0, 2, (10_000, 142)).astype(np.uint8)
    for row in info:
        got, want = nat.polar_encode(row), ref.polar_encode(row)
        assert got.dtype == want.dtype == np.uint8
        assert np.array_equal(got, want)


def test_polar_encode_conversions(nat):
    rng = np.random.default_rng(6)
    info = rng.integers(0, 2, 142)
    for v in (info, info.astype(bool), info.astype(np.float64), info.astype(np.int8), info.tolist(),
              info.astype(np.complex128)):
        assert np.array_equal(nat.polar_encode(v), ref.polar_encode(v))


def _bad_info(i):
    v = np.zeros(142)
    v[i] = (2, -1, 0.5, np.nan, np.inf, 1 + 1e-15)[i]
    return v


@pytest.mark.parametrize("bad", [np.zeros(141, np.uint8), np.zeros((2, 142), np.uint8), np.uint8(0)]
                         + [_bad_info(i) for i in range(6)] + [np.full(142, 255, np.uint8)])
def test_polar_encode_refusals(nat, bad):
    _same_error(ref.polar_encode, nat.polar_encode, bad)


# --- CA-SCL decoder -----------------------------------------------------------------


def _llr(coded, esn0_db, rng):
    """BPSK on AWGN, as tests/test_qrss_header.py: llr = 2 y / sigma^2."""
    s = 1.0 - 2.0 * coded
    var = 1.0 / (2.0 * 10 ** (esn0_db / 10))
    return 2.0 * (s + rng.normal(0.0, np.sqrt(var), s.size)) / var


def _random_header(rng):
    mode = int(rng.integers(0, 3))
    calls = ("K1ABC", "G", "VK2/W1AW", "00000000", "W1AW/7", "JA1XYZ")
    return HeaderFields(calls[int(rng.integers(0, len(calls)))],
                        None if rng.random() < 0.2 else ref_header.grid_decode(int(rng.integers(0, 32400))),
                        int(rng.integers(0, 1 << 32)), mode, int(rng.integers(0, mode + 1)),
                        int(rng.integers(0, 1 << 16)), version=int(rng.choice([1, 2])))


def _compare_list(nat, llr, list_size, stats):
    want_bits, want_pm = ref.decode_list(llr, list_size)
    got_bits, got_pm = nat.decode_list(llr, list_size)
    assert got_bits.dtype == want_bits.dtype == np.uint8
    assert got_bits.shape == want_bits.shape and got_pm.shape == want_pm.shape
    assert np.array_equal(got_bits, want_bits)
    np.testing.assert_allclose(got_pm, want_pm, rtol=1e-9, atol=0)
    stats["bitwise"] += np.array_equal(got_pm.view(np.int64), want_pm.view(np.int64))
    stats["n"] += 1
    # polar_decode is "first path the acceptance passes": the reference's
    # default acceptance applied to the reference's own list.
    want = next((b for b in want_bits if ref_header.unpack(b) is not None), None)
    got = nat.polar_decode(llr, list_size)
    assert (got is None) == (want is None)
    if want is not None:
        assert got.dtype == np.uint8 and np.array_equal(got, want)
        stats["accepted"] += 1
    return want_bits


@pytest.mark.parametrize("esn0_db", [-14.0, -12.0, -10.4, -9.4, -8.0, -5.0])
def test_decoder_on_noisy_codewords(nat, esn0_db):
    """350 vectors per SNR (2100 in all), a different header each."""
    rng = np.random.default_rng(int(1000 - 10 * esn0_db))
    stats = {"n": 0, "bitwise": 0, "accepted": 0}
    for _ in range(350):
        llr = _llr(ref_header.encode(_random_header(rng)), esn0_db, rng)
        _compare_list(nat, llr, 8, stats)
    assert stats["n"] == 350
    if esn0_db >= -9.4:     # about half the headers are version 2, which the default accepts
        assert stats["accepted"] > 100, stats


def test_decoder_on_noise(nat):
    """Weak, threshold-like and strong noise, single and summed 'passes'."""
    rng = np.random.default_rng(50)
    stats = {"n": 0, "bitwise": 0, "accepted": 0}
    for t in range(150):
        scale = (0.5, 2.0, 8.0)[t % 3]
        llr = rng.normal(0.0, scale, 2474)
        if t % 5 == 4:
            llr = llr + rng.normal(0.0, scale, (3, 2474)).sum(axis=0)
        _compare_list(nat, llr, 8, stats)
    assert stats["accepted"] == 0


@pytest.mark.parametrize("list_size", [1, 2, 3, 4, 16, 32])
def test_decoder_list_sizes(nat, list_size):
    rng = np.random.default_rng(60 + list_size)
    stats = {"n": 0, "bitwise": 0, "accepted": 0}
    for t in range(40):
        llr = _llr(ref_header.encode(_random_header(rng)), (-12.0, -9.4, -6.0)[t % 3], rng)
        _compare_list(nat, llr, list_size, stats)
    _compare_list(nat, llr, np.int64(list_size), stats)        # a numpy int list size


def test_decoder_ties_and_extremes(nat):
    """Exact ties everywhere (zero LLRs), quantised LLRs with many ties,
    erased repeats, huge but finite LLRs whose g overflows."""
    rng = np.random.default_rng(70)
    stats = {"n": 0, "bitwise": 0, "accepted": 0}
    coded = ref_header.encode(H)
    cases = [np.zeros(2474), np.full(2474, 1.0), np.full(2474, -3.0),
             np.round(_llr(coded, -9.4, rng)), np.sign(_llr(coded, -6.0, rng)),
             np.where(rng.random(2474) < 0.5, 0.0, _llr(coded, -6.0, rng)),
             8.0 * (1.0 - 2.0 * coded), 1e300 * (1.0 - 2.0 * coded),
             1e307 * rng.normal(0, 1, 2474), 1e-300 * rng.normal(0, 1, 2474)]
    erased = 6.0 * (1.0 - 2.0 * coded)
    erased[2048:] = 0.0
    cases.append(erased)
    for llr in cases:
        for list_size in (1, 8):
            _compare_list(nat, llr, list_size, stats)


def test_decoder_refusals(nat):
    for bad in (0, -1):
        _same_error(ref.decode_list, nat.decode_list, np.zeros(2474), bad)
        _same_error(ref.polar_decode, nat.polar_decode, np.zeros(2474), bad)
    for bad in (np.zeros(2048), np.zeros((2, 2474)), np.float64(1.0)):
        _same_error(ref.decode_list, nat.decode_list, bad)
        _same_error(ref.polar_decode, nat.polar_decode, bad)
    for v in (np.inf, -np.inf, np.nan):
        llr = np.zeros(2474)
        llr[5] = v
        _same_error(ref.decode_list, nat.decode_list, llr)
    big = np.zeros(2474)
    big[[3, 3 + 2048]] = 1e308                  # finite, but the fold overflows
    _same_error(ref.decode_list, nat.decode_list, big)


def test_polar_decode_accept(nat):
    rng = np.random.default_rng(80)
    v1 = HeaderFields(**{**H.__dict__, "version": 1})
    for _ in range(30):
        llr = _llr(ref_header.encode(v1), -8.0, rng)

        def accept(b):
            return ref_header.unpack(b, (1, 2)) is not None

        want, got = ref.polar_decode(llr, 8, accept), nat.polar_decode(llr, 8, accept)
        assert (got is None) == (want is None)
        if want is not None:
            assert np.array_equal(got, want)
        assert nat.polar_decode(llr) is None and ref.polar_decode(llr) is None   # version 1: not by default
    seen = []
    assert nat.polar_decode(llr, 4, lambda b: seen.append(b.copy()) or False) is None
    assert len(seen) == 4 and all(s.shape == (142,) and s.dtype == np.uint8 for s in seen)
    assert np.array_equal(np.array(seen), ref.decode_list(llr, 4)[0])

    def boom(b):
        raise RuntimeError("from the acceptance test")

    with pytest.raises(RuntimeError, match="from the acceptance test"):
        nat.polar_decode(llr, 8, boom)
    # Truthiness, as the reference's `if accept(bits)`.
    assert np.array_equal(nat.polar_decode(llr, 8, lambda b: np.int64(1)), ref.decode_list(llr, 8)[0][0])


# --- probes: the reference's own argument semantics ---------------------------------


class _LessIsFalse:
    """`< 1` is False and the index is 3: numpy, not the comparison, decides."""

    def __lt__(self, other):
        return False

    def __index__(self):
        return 3


@pytest.mark.parametrize("list_size", [0.5, 8.0, np.float64(8.0), np.float32(2.0), True, False, None, "8", [3],
                                       np.array(3), np.array([3]), np.array([3, 4]), np.array([[3]]), np.int8(3),
                                       np.uint64(2), 2**70, -(2**70), _LessIsFalse(), 3])
def test_decoder_list_size_like_the_reference(nat, list_size):
    """`list_size < 1` first, then the LLRs, then numpy's own conversion of
    list_size to a shape (np.full / np.broadcast_to): so a bool or a float
    is refused with numpy's text, and only after the LLRs are checked."""
    rng = np.random.default_rng(90)
    llr = _llr(ref_header.encode(H), -6.0, rng)
    _same_outcome(ref.decode_list, nat.decode_list, llr, list_size)
    _same_outcome(ref.polar_decode, nat.polar_decode, llr, list_size)
    _same_outcome(ref.decode_list, nat.decode_list, np.zeros(10), list_size)        # which refusal wins
    bad = np.zeros(2474)
    bad[0] = np.nan
    _same_outcome(ref.decode_list, nat.decode_list, bad, list_size)


def test_polar_decode_returns_the_row_it_handed_over(nat):
    """The reference returns the very array `accept` saw, so what `accept`
    did to it shows in the result."""
    rng = np.random.default_rng(91)
    llr = _llr(ref_header.encode(H), -6.0, rng)
    seen = []

    def flip(b):
        b[0] ^= 1
        seen.append(b)
        return len(seen) == 3

    want = ref.polar_decode(llr, 8, flip)
    ref_seen, seen = seen, []
    got = nat.polar_decode(llr, 8, flip)
    assert got is seen[-1] and np.array_equal(got, want)
    assert all(np.array_equal(a, b) for a, b in zip(seen, ref_seen))


def test_polar_transform_probe(nat):
    """Random shapes, byte values and dtypes, refusals included."""
    rng = np.random.default_rng(92)
    for _ in range(300):
        nd = int(rng.integers(1, 4))
        shape = tuple(int(x) for x in rng.integers(0, 4, nd - 1)) + (int(rng.integers(0, 70)),)
        x = rng.integers(-300, 300, shape)
        for v in (x, x.astype(np.uint8), (x % 2).astype(bool), (x % 7).astype(np.float64)):
            _same_outcome(ref.polar_transform, nat.polar_transform, v)


def test_rate_match_probe(nat):
    rng = np.random.default_rng(93)
    for _ in range(100):
        nd = int(rng.integers(1, 4))
        shape = tuple(int(x) for x in rng.integers(0, 3, nd - 1)) + (int(rng.choice([0, 1, 2047, 2048, 2049, 2500])),)
        x = rng.normal(0, 1, shape)
        for v in (x, x.astype(np.float32), (x > 0), x.astype(">f8"), x + 1j * x):
            _same_outcome(ref.rate_match, nat.rate_match, v)


def test_fold_and_encode_probe(nat):
    rng = np.random.default_rng(94)
    for _ in range(200):
        llr = rng.normal(0, 1, 2474) * 10.0 ** rng.integers(-310, 300, 2474)
        llr[rng.random(2474) < 0.05] = rng.choice([0.0, -0.0, np.inf, -np.inf, np.nan])
        got, want = nat.fold(llr), ref.fold(llr)
        assert np.array_equal(got.view(np.int64), want.view(np.int64)) or np.array_equal(got, want, equal_nan=True)
        info = rng.integers(0, 2, 142)
        if rng.random() < 0.3:
            info = info.astype(object)
            info[int(rng.integers(0, 142))] = rng.choice([2, -1, 0.5, True, 1.0, None, "1"])
        _same_outcome(ref.polar_encode, nat.polar_encode, info)

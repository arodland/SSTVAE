"""C++ header (native/core/qrss/header) against sstvae.qrss.header.

Bit-exact by construction: integer fields, SSTVAE's beacon CRC and
alphabet, the polar encoder. The refusal texts are compared character for
character. HeaderFields stays the reference's dataclass, converted by the
adapters in tests/qrss_native/header.py and header_rx.py, which are what
`--native-qrss` installs; they are exercised here directly.
"""

import numpy as np
import pytest

from qrss_native import header as rows_tx
from qrss_native import header_rx as rows_rx
from qrss_native import require
from sstvae.modem import beacon
from sstvae.qrss import header as ref
from sstvae.qrss.header import HeaderFields

H = HeaderFields("K1ABC/P", "FN42", 0xDEADBEEF, 2, 1, 0xD1D8)
HEADERS = [
    H,
    HeaderFields("K1ABC", None, 0, 0, 0, 0),
    HeaderFields("G", "AA00", 1, 1, 0, 1),
    HeaderFields("VK2/W1AW", "RR99", 0xFFFFFFFF, 2, 2, 0xFFFF),
    HeaderFields("00000000", "JO62", 0x12345678, 1, 1, 0xD1D8),
    HeaderFields("K1ABC", "FN42", 7, 0, 0, 1, version=1),
]


@pytest.fixture(scope="module")
def nat(native):
    return require(native, "header")


@pytest.fixture(scope="module")
def tx(native):
    q = require(native, "header") and native.qrss
    return {attr: f for (_, attr), f in rows_tx.rows(q).items()}


@pytest.fixture(scope="module")
def rx(native):
    q = require(native, "header") and native.qrss
    return {attr: f for (_, attr), f in rows_rx.rows(q).items()}


def _same_error(f_ref, f_nat, *args, **kw):
    with pytest.raises(Exception) as want:
        f_ref(*args, **kw)
    with pytest.raises(type(want.value)) as got:
        f_nat(*args, **kw)
    assert str(got.value) == str(want.value)


def _random_header(rng, version=None):
    mode = int(rng.integers(0, 3))
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789/"
    call = "".join(alphabet[int(i)] for i in rng.integers(0, len(alphabet), int(rng.integers(1, 9))))
    return HeaderFields(call, None if rng.random() < 0.2 else ref.grid_decode(int(rng.integers(0, 32400))),
                        int(rng.integers(0, 1 << 32)), mode, int(rng.integers(0, mode + 1)),
                        int(rng.integers(0, 1 << 16)),
                        version=int(rng.choice([1, 2])) if version is None else version)


def test_constants(nat):
    assert (nat.CALL_CHARS, nat.GRID_NONE, nat.MAX_MODE) == (ref.CALL_CHARS, ref.GRID_NONE, ref.MAX_MODE)
    assert nat.OFFSET == ref._OFFSET


# --- CRC -------------------------------------------------------------------------


def test_crc16(nat):
    bits = np.array([int(b) for c in b"123456789" for b in format(c, "08b")], dtype=np.uint8)
    assert np.array_equal(nat.crc16(bits), ref.crc16(bits))
    assert int("".join(map(str, nat.crc16(bits))), 2) == 0xA69D
    rng = np.random.default_rng(10)
    for n in (0, 1, 16, 126, 300):
        b = rng.integers(0, 2, n)
        for v in (b, b.astype(bool), b.astype(np.uint8), b.astype(float), b.tolist()):
            got, want = nat.crc16(v), ref.crc16(v)
            assert got.dtype == want.dtype == np.uint8 and np.array_equal(got, want)
            assert np.array_equal(got, beacon._crc16(b))


# --- grid ---------------------------------------------------------------------------


def test_grid_every_code(nat):
    for v in range(18 * 18 * 100):
        g = ref.grid_decode(v)
        assert nat.grid_decode(v) == g
        assert nat.grid_encode(g) == ref.grid_encode(g) == v
        assert nat.grid_encode(g.lower()) == v
    assert nat.grid_decode(32767) is None and nat.grid_encode(None) == 32767
    for v in (np.int64(1234), np.uint16(32767), True, 1234.9, "1234"):     # v = int(v)
        assert nat.grid_decode(v) == ref.grid_decode(v)


@pytest.mark.parametrize("bad", [32400, 32766, 32768, -1, 2**70, -(2**70), 2**63, 2**63 - 1])
def test_grid_decode_refusals(nat, bad):
    _same_error(ref.grid_decode, nat.grid_decode, bad)


@pytest.mark.parametrize("bad", ["FN4", "FN421", "SN42", "F142", "AAAA", "", "FN 42", "FN4\n", "fn4'",
                                 "K1'\"", "ÄN42", "FN42\x00", 42, b"FN42", 4.2, ["FN42"]])
def test_grid_encode_refusals(nat, bad):
    _same_error(ref.grid_encode, nat.grid_encode, bad)


def test_grid_encode_unicode_case_mapping(nat):
    """str.upper() can map a non-ASCII letter into A-R: the reference
    accepts it, and the binding hands the C++ Python's upper()."""
    for g in ("ıN42", "ﬀ42", "ßN42"):
        try:
            want = ref.grid_encode(g)
        except ValueError:
            _same_error(ref.grid_encode, nat.grid_encode, g)
        else:
            assert nat.grid_encode(g) == want


# --- pack / encode ------------------------------------------------------------------


@pytest.mark.parametrize("h", HEADERS)
def test_pack_and_encode(tx, h):
    for name in ("pack", "encode"):
        got, want = tx[name](h), getattr(ref, name)(h)
        assert got.dtype == want.dtype == np.uint8 and got.shape == want.shape
        assert np.array_equal(got, want)


def test_pack_and_encode_random_headers(tx):
    rng = np.random.default_rng(20)
    for _ in range(10_000):
        h = _random_header(rng)
        assert np.array_equal(tx["pack"](h), ref.pack(h))
        assert np.array_equal(tx["encode"](h), ref.encode(h))


def test_pack_takes_numpy_and_bool_integers(tx):
    h = HeaderFields("K1ABC", "FN42", np.uint32(0xDEADBEEF), np.int64(2), np.int8(1), np.uint16(0xD1D8),
                     version=np.int64(2), reserved=False)
    assert np.array_equal(tx["pack"](h), ref.pack(h))
    h = HeaderFields("K1ABC", "fn42", True, True, False, 7, version=True)
    assert np.array_equal(tx["pack"](h), ref.pack(h))


@pytest.mark.parametrize("h", [
    HeaderFields("k1abc", None, 0, 0, 0, 0),
    HeaderFields("K1ABCDEFG", None, 0, 0, 0, 0),
    HeaderFields("K1 AB", None, 0, 0, 0, 0),
    HeaderFields("K1AB?", None, 0, 0, 0, 0),
    HeaderFields("", None, 0, 0, 0, 0),
    HeaderFields("K1'B", None, 0, 0, 0, 0),
    HeaderFields("K1\"'B", None, 0, 0, 0, 0),
    HeaderFields("K1\\B\n\t\x01\x7f", None, 0, 0, 0, 0),
    HeaderFields("K1ÄBC", None, 0, 0, 0, 0),
    HeaderFields("K1ABC\ud800", None, 0, 0, 0, 0),
    HeaderFields(None, None, 0, 0, 0, 0),
    HeaderFields(12345, None, 0, 0, 0, 0),
    HeaderFields(b"K1ABC", None, 0, 0, 0, 0),
    HeaderFields("K1ABC", "XX99", 0, 0, 0, 0),
    HeaderFields("K1ABC", 42, 0, 0, 0, 0),
    HeaderFields("K1ABC", None, 1 << 32, 0, 0, 0),
    HeaderFields("K1ABC", None, -1, 0, 0, 0),
    HeaderFields("K1ABC", None, 2**70, 0, 0, 0),
    HeaderFields("K1ABC", None, -(2**70), 0, 0, 0),
    HeaderFields("K1ABC", None, np.int64(-5), 0, 0, 0),
    HeaderFields("K1ABC", None, 0, 3, 0, 0),
    HeaderFields("K1ABC", None, 0, -1, 0, 0),
    HeaderFields("K1ABC", None, 0, 2**64, 0, 0),
    HeaderFields("K1ABC", None, 0, 1, 2, 0),
    HeaderFields("K1ABC", None, 0, 1, -1, 0),
    HeaderFields("K1ABC", None, 0, 2, 2**65, 0),
    HeaderFields("K1ABC", None, 0, 0, 0, 1 << 16),
    HeaderFields("K1ABC", None, 0, 0, 0, -(2**80)),
    HeaderFields("K1ABC", None, 0, 0, 0, 0, version=0),
    HeaderFields("K1ABC", None, 0, 0, 0, 0, version=3),
    HeaderFields("K1ABC", None, 0, 0, 0, 0, version=2**64 + 2),
    HeaderFields("K1ABC", None, 0, 0, 0, 0, reserved=1),
    HeaderFields("K1ABC", None, 0, 0, 0, 0, reserved=True),
    HeaderFields("K1ABC", None, 0, 0, 0, 0, reserved=-(2**90)),
    HeaderFields("k1abc", "XX99", -1, 3, 9, 1 << 16, version=7, reserved=1),   # first check wins
    HeaderFields("K1ABC", "XX99", 0, 3, 0, 0),
    HeaderFields("K1ABC", "XX99", 0, 0, 0, 0, version=1),
])
def test_pack_refusals(tx, h):
    for name in ("pack", "encode"):
        _same_error(getattr(ref, name), tx[name], h)


# --- unpack ------------------------------------------------------------------------


def _check_unpack(tx, bits, versions=None):
    if versions is None:
        got, want = tx["unpack"](bits), ref.unpack(bits)
    else:
        got, want = tx["unpack"](bits, versions), ref.unpack(bits, versions)
    assert got == want
    return want


def test_unpack_random_vectors(tx):
    """1e5 random 142-bit vectors: plain random (CRC fails), random bodies
    under a valid CRC (field checks decide), and valid headers with one
    field overwritten under a valid CRC."""
    rng = np.random.default_rng(30)
    accepted = 0
    plain = rng.integers(0, 2, (40_000, 142)).astype(np.uint8)
    for b in plain:
        accepted += _check_unpack(tx, b) is not None
    bodies = rng.integers(0, 2, (30_000, 126)).astype(np.uint8)
    for body in bodies:
        _check_unpack(tx, np.concatenate([body, ref.crc16(body)]))
    for _ in range(30_000):
        body = ref.pack(_random_header(rng))[:126].copy()
        a = int(rng.integers(0, 126))
        b = min(126, a + int(rng.integers(1, 16)))
        body[a:b] = rng.integers(0, 2, b - a)
        bits = np.concatenate([body, ref.crc16(body)])
        for versions in ((2,), (1, 2)):
            accepted += _check_unpack(tx, bits, versions) is not None
    assert accepted > 1000


@pytest.mark.parametrize("h", HEADERS)
def test_unpack_every_single_bit_flip(tx, h):
    bits = ref.pack(h)
    for versions in ((2,), (1, 2), (1,)):
        _check_unpack(tx, bits, versions)
        for i in range(142):
            flipped = bits.copy()
            flipped[i] ^= 1
            _check_unpack(tx, flipped, versions)


def test_unpack_bad_fields_under_a_valid_crc(tx):
    for (a, b), value in [((115, 119), 1), ((115, 119), 3), ((115, 119), 0), ((119, 126), 1),
                          ((95, 97), 3), ((97, 99), 3), ((48, 63), 32400), ((48, 63), 32766),
                          ((48, 63), 32767)]:
        body = ref.pack(H)[:126].copy()
        body[a:b] = [int(c) for c in format(value, f"0{b - a}b")]
        for versions in ((2,), (1, 2), (0, 1, 2, 3)):
            _check_unpack(tx, np.concatenate([body, ref.crc16(body)]), versions)
    for call in ("K1-AB", "K1 AB", "", "        ", "K1AB?", " K1AB", "K1AB   ", "A", "12345678"):
        body = ref.pack(H)[:126].copy()
        codes = beacon.callsign_to_codes(call)
        body[:48] = np.concatenate([[int(c) for c in format(int(x), "06b")] for x in codes])
        _check_unpack(tx, np.concatenate([body, ref.crc16(body)]))


def test_unpack_versions_containers(tx):
    v1 = ref.pack(HEADERS[-1])
    for versions in ((1, 2), [1], {1}, range(3), (2.0, 1.0), (np.int64(1),), (), ("1",), (1, "x")):
        _check_unpack(tx, v1, versions)
        _check_unpack(tx, ref.pack(H), versions)


def test_unpack_versions_is_consulted_lazily(tx):
    """Only once the CRC passes, and by Python's own `in`."""
    good, bad = ref.pack(H), ref.pack(H)
    bad[130] ^= 1
    for make in (lambda: (v for v in (1, 2)), lambda: iter([2]), lambda: iter([1])):
        assert tx["unpack"](good, make()) == ref.unpack(good, make())
    for versions in (2, None, 2.5):
        assert tx["unpack"](bad, versions) is None and ref.unpack(bad, versions) is None
        _same_error(ref.unpack, tx["unpack"], good, versions)


def test_unpack_malformed_input(tx):
    bits = ref.pack(H)
    for bad in (np.zeros(141, np.uint8), np.full(142, 2, np.uint8), bits[None, :], bits.astype(float) * 0.5,
                np.where(np.arange(142) == 3, np.nan, bits.astype(float)), bits.astype(bool),
                bits.astype(float), bits.astype(np.int64) - 2 * (np.arange(142) == 0), bits.tolist(),
                bits.astype(np.complex128), 0, []):
        _check_unpack(tx, bad)


# --- decode ---------------------------------------------------------------------------


def _llr(coded, esn0_db, rng):
    s = 1.0 - 2.0 * coded
    var = 1.0 / (2.0 * 10 ** (esn0_db / 10))
    return 2.0 * (s + rng.normal(0.0, np.sqrt(var), s.size)) / var


@pytest.mark.parametrize("h", HEADERS)
def test_decode_noiseless(rx, h):
    llr = 10.0 * (1.0 - 2.0 * ref.encode(h))
    for kw in ({}, {"list_size": 1}, {"versions": (1, 2)}, {"versions": [1]}, {"list_size": 3, "versions": (2,)}):
        assert rx["decode"](llr, **kw) == ref.decode(llr, **kw)


def test_decode_version_fallback(rx):
    v1 = HEADERS[-1]
    llr = 8.0 * (1.0 - 2.0 * ref.encode(v1))
    assert rx["decode"](llr) is None and ref.decode(llr) is None
    assert rx["decode"](llr, versions=(1, 2)) == ref.decode(llr, versions=(1, 2)) == v1


@pytest.mark.parametrize("esn0_db", [-13.4, -11.0, -9.4, -7.0])
def test_decode_noisy(rx, esn0_db):
    rng = np.random.default_rng(int(500 - 10 * esn0_db))
    for t in range(60):
        h = _random_header(rng)
        llr = _llr(ref.encode(h), esn0_db, rng)
        if t % 4 == 3:                                  # soft combining of two passes
            llr = llr + _llr(ref.encode(h), esn0_db, rng)
        versions = ((2,), (1, 2))[t % 2]
        assert rx["decode"](llr, versions=versions) == ref.decode(llr, versions=versions)
        if t % 10 == 0:
            assert rx["decode"](llr.astype(np.float32), 2) == ref.decode(llr.astype(np.float32), 2)


def test_decode_refusals(rx):
    _same_error(ref.decode, rx["decode"], np.zeros(2474), 0)
    _same_error(ref.decode, rx["decode"], np.zeros(2048))
    _same_error(ref.decode, rx["decode"], None)
    llr = np.zeros(2474)
    llr[0] = np.nan
    _same_error(ref.decode, rx["decode"], llr)
    _same_error(ref.decode, rx["decode"], np.zeros(2474), 8, 2)          # versions not iterable

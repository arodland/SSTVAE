"""C++ sequences (native/core/qrss/sequences) against sstvae.qrss.sequences.

Bit-exact by construction: SHA-256 over integers. Skips when the
extension is not built; `pytest --native-qrss=tx` runs the rest of the
QRSS suites against the same code.
"""

import numpy as np
import pytest

from qrss_native import require
from sstvae.qrss import sequences as ref

Q_TEST = 1954752


@pytest.fixture(scope="module")
def nat(native):
    return require(native, "sequences")


@pytest.mark.parametrize("domain", [ref.PREAMBLE_DOMAIN, ref.REFERENCE_DOMAIN,
                                    ref.SCRAMBLE_DOMAIN, ref.SPREAD_DOMAIN,
                                    ref.L_PREAMBLE_DOMAIN, ref.L_REFERENCE_DOMAIN])
@pytest.mark.parametrize("n", [0, 1, 7, 255, 256, 257, 660, 3539, 50600])
def test_sha_bits(nat, domain, n):
    want = ref.sha_bits(domain, n)
    got = nat.sha_bits(domain, n)
    assert got.dtype == want.dtype == np.uint8
    assert np.array_equal(got, want)


@pytest.mark.parametrize("prefix", [b"", b"\x01\x02", bytes(range(55)), bytes(range(64))])
def test_sha_bits_prefix(nat, prefix):
    assert np.array_equal(nat.sha_bits(b"QRSSTVAE CE preamble", 1000, prefix),
                          ref.sha_bits(b"QRSSTVAE CE preamble", 1000, prefix))


@pytest.mark.parametrize("q", [0, Q_TEST, Q_TEST + 1, 2**40 + 12345, 2**64 - 1])
def test_scrambler(nat, q):
    got, want = nat.scrambler(q, 50600), ref.scrambler(q, 50600)
    assert got.dtype == want.dtype == np.int8
    assert np.array_equal(got, want)


@pytest.mark.parametrize("q", [-1, 2**64])
def test_scrambler_refuses_q_outside_uint64(nat, q):
    with pytest.raises(ValueError, match="q must fit in uint64"):
        ref.scrambler(q, 8)
    with pytest.raises(ValueError, match="q must fit in uint64"):
        nat.scrambler(q, 8)


def test_cached_streams(nat):
    for name in ("preamble_ce", "preamble_l"):
        got, want = getattr(nat, name)(), getattr(ref, name)()
        assert got.dtype == want.dtype and np.array_equal(got, want), name
        assert not got.flags.writeable, name
    for n in (0, 165, 3539):
        assert np.array_equal(nat.references_ce(n), ref.references_ce(n))
    for n in (0, 83, 1770):
        assert np.array_equal(nat.references_l(n), ref.references_l(n))
    assert np.array_equal(nat.spread_whitener(50600), ref.spread_whitener(50600))


def test_pm1_and_qpsk(nat):
    rng = np.random.default_rng(1)
    b = rng.integers(0, 2, size=(3, 40)).astype(np.uint8)
    assert np.array_equal(nat.pm1(b), ref.pm1(b)) and nat.pm1(b).shape == b.shape
    v = b[0]
    assert np.array_equal(nat.qpsk(v), ref.qpsk(v))   # exact: one division by sqrt(2)
    for bad in ([0, 1, 1], np.zeros((2, 2), dtype=np.uint8)):
        with pytest.raises(ValueError, match="even number of bits"):
            ref.qpsk(bad)
        with pytest.raises(ValueError, match="even number of bits"):
            nat.qpsk(bad)


def test_negative_count_is_refused(nat):
    with pytest.raises(ValueError, match="n must be >= 0"):
        ref.sha_bits(b"x", -1)
    with pytest.raises(ValueError, match="n must be >= 0"):
        nat.sha_bits(b"x", -1)

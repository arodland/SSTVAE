"""QRSSTVAE header and polar FEC: acceptance tests H1-H4 (design section 10.1).

The channel here is ideal BPSK on AWGN: header coded bit e rides a symbol
of +1 for 0 and -1 for 1, y = s + n with n ~ N(0, sigma^2), and the LLR is
2y/sigma^2. Es/N0 is per coded bit, so sigma^2 = 1/(2 Es/N0); the design
threshold is Es/N0 = -9.4 dB (SNR_2500 = -23.5 dB for waveform CE). Every
statistical test uses a fixed seed and states its tolerance.
"""

import hashlib
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from sstvae.modem import beacon
from sstvae.qrss import header, polar
from sstvae.qrss.header import HeaderFields

ROOT = Path(__file__).resolve().parent.parent

H = HeaderFields("K1ABC/P", "FN42", 0xDEADBEEF, 2, 1, 0xD1D8)
HEADERS = [
    H,
    HeaderFields("K1ABC", None, 0, 0, 0, 0),
    HeaderFields("G", "AA00", 1, 1, 0, 1),
    HeaderFields("VK2/W1AW", "RR99", 0xFFFFFFFF, 2, 2, 0xFFFF),
    HeaderFields("00000000", "JO62", 0x12345678, 1, 1, 0xD1D8),
]


def _bits_of(data: bytes) -> np.ndarray:
    return np.array([int(b) for c in data for b in format(c, "08b")], dtype=np.uint8)


def _to_int(bits) -> int:
    return int("".join(str(int(b)) for b in bits), 2)


def _llr(coded: np.ndarray, esn0_db: float, rng) -> np.ndarray:
    s = 1.0 - 2.0 * coded
    var = 1.0 / (2.0 * 10 ** (esn0_db / 10))
    return 2.0 * (s + rng.normal(0.0, np.sqrt(var), s.size)) / var


def _with_crc(body: np.ndarray) -> np.ndarray:
    return np.concatenate([body, header.crc16(body)])


# --- H1: CRC, fields, pack/unpack --------------------------------------------


def test_crc_check_value_is_a69d_not_ccitt_false():
    """The SSTVAE beacon CRC over ASCII "123456789" is 0xA69D. True
    CRC-16/CCITT-FALSE gives 0x29B1; the header uses the former (D8)."""
    assert _to_int(header.crc16(_bits_of(b"123456789"))) == 0xA69D


def test_crc_is_beacon_crc_bit_for_bit():
    rng = np.random.default_rng(10)
    for n in (1, 16, 126, 300):
        bits = rng.integers(0, 2, n)
        np.testing.assert_array_equal(header.crc16(bits), beacon._crc16(bits))


@pytest.mark.parametrize("h", HEADERS)
def test_pack_unpack_round_trip(h):
    bits = header.pack(h)
    assert bits.shape == (142,) and bits.dtype == np.uint8
    assert header.unpack(bits) == h


def test_field_layout_is_table_order_msb_first():
    bits = header.pack(H)
    assert beacon.codes_to_callsign(
        [_to_int(bits[6 * i:6 * i + 6]) for i in range(8)]) == "K1ABC/P"
    assert _to_int(bits[48:63]) == header.grid_encode("FN42") == ((5 * 18 + 13) * 10 + 4) * 10 + 2
    assert _to_int(bits[63:95]) == 0xDEADBEEF
    assert _to_int(bits[95:97]) == 2 and _to_int(bits[97:99]) == 1
    assert _to_int(bits[99:115]) == 0xD1D8
    assert _to_int(bits[115:119]) == 1 and _to_int(bits[119:126]) == 0
    np.testing.assert_array_equal(bits[126:], beacon._crc16(bits[:126]))
    assert _to_int(header.pack(HEADERS[1])[48:63]) == 32767       # grid none


def test_every_single_bit_flip_is_rejected():
    bits = header.pack(H)
    for i in range(142):
        flipped = bits.copy()
        flipped[i] ^= 1
        assert header.unpack(flipped) is None, f"bit {i}"


def test_grid_codec_round_trips_every_locator():
    for v in range(18 * 18 * 100):
        g = header.grid_decode(v)
        assert header.grid_encode(g) == v
    assert header.grid_decode(32767) is None and header.grid_encode(None) == 32767
    assert header.grid_encode("fn42") == header.grid_encode("FN42")
    for bad in ("FN4", "FN421", "SN42", "F142", "AAAA"):
        with pytest.raises(ValueError):
            header.grid_encode(bad)
    with pytest.raises(ValueError):
        header.grid_decode(32400)


@pytest.mark.parametrize("field,value", [
    ((115, 119), 2),          # version
    ((119, 126), 1),          # reserved
    ((95, 97), 3),            # mode
    ((97, 99), 3),            # segment > mode (mode 2)
    ((48, 63), 32400),        # unused grid code
])
def test_crc_valid_header_with_bad_field_is_rejected(field, value):
    body = header.pack(H)[:126].copy()
    a, b = field
    body[a:b] = [int(c) for c in format(value, f"0{b - a}b")]
    assert header.unpack(_with_crc(body)) is None


def test_crc_valid_header_with_segment_above_mode_is_rejected():
    body = header.pack(HeaderFields("K1ABC", None, 7, 0, 0, 1))[:126].copy()
    body[97:99] = [0, 1]                                   # segment 1, mode 0
    assert header.unpack(_with_crc(body)) is None


@pytest.mark.parametrize("call", ["K1-AB", "K1 AB", "", "        ", "K1AB?"])
def test_crc_valid_header_with_bad_callsign_is_rejected(call):
    """Only [A-Z0-9/] with trailing padding passes; '-', '?', an internal
    space or an empty call do not, though SSTVAE's alphabet can carry them."""
    body = header.pack(H)[:126].copy()
    codes = beacon.callsign_to_codes(call)                 # beacon maps '?' itself
    body[:48] = np.concatenate([[int(c) for c in format(int(x), "06b")] for x in codes])
    assert header.unpack(_with_crc(body)) is None


@pytest.mark.parametrize("h", [
    HeaderFields("k1abc", None, 0, 0, 0, 0),
    HeaderFields("K1ABCDEFG", None, 0, 0, 0, 0),
    HeaderFields("K1 AB", None, 0, 0, 0, 0),
    HeaderFields("K1AB?", None, 0, 0, 0, 0),
    HeaderFields("", None, 0, 0, 0, 0),
    HeaderFields("K1ABC", "XX99", 0, 0, 0, 0),
    HeaderFields("K1ABC", None, 1 << 32, 0, 0, 0),
    HeaderFields("K1ABC", None, 0, 3, 0, 0),
    HeaderFields("K1ABC", None, 0, 1, 2, 0),
    HeaderFields("K1ABC", None, 0, 0, 0, 1 << 16),
    HeaderFields("K1ABC", None, 0, 0, 0, 0, version=2),
    HeaderFields("K1ABC", None, 0, 0, 0, 0, reserved=1),
])
def test_pack_refuses_what_unpack_would_reject(h):
    with pytest.raises(ValueError):
        header.pack(h)


def test_unpack_rejects_malformed_input():
    assert header.unpack(np.zeros(141, np.uint8)) is None
    assert header.unpack(np.full(142, 2, np.uint8)) is None


# --- H2: INFO_SET, encoder, rate matching, noiseless decode ----------------------


def test_info_set_literal_hash():
    assert len(polar.INFO_SET) == 142
    assert hashlib.sha256(np.array(polar.INFO_SET, ">u2").tobytes()).hexdigest() == (
        "4e064eee9b1df68ee4d044970b51f05a16700d51599f857f86e7d15d782b97bb")
    assert header.INFO_SET is polar.INFO_SET


def test_gen_qrss_polar_check_passes():
    out = subprocess.run([sys.executable, str(ROOT / "tools" / "gen_qrss_polar.py"), "--check"],
                         capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stdout + out.stderr
    assert "check ok" in out.stdout


def test_polar_transform_is_the_natural_order_kronecker_power():
    """x_j = XOR over i containing j of u_i, i.e. x = u F^(x)11 with
    F^(x)11[i, j] = 1 iff the bits of j are a subset of those of i."""
    i = np.arange(polar.N)
    g = ((i[:, None] & i[None, :]) == i[None, :]).astype(np.int64)
    rng = np.random.default_rng(20)
    u = rng.integers(0, 2, (3, polar.N)).astype(np.uint8)
    np.testing.assert_array_equal(polar.polar_transform(u), (u @ g) % 2)
    np.testing.assert_array_equal(polar.polar_transform(polar.polar_transform(u)), u)


def test_rate_matching_and_fold():
    x = np.arange(polar.N) % 2
    coded = polar.rate_match(x)
    assert coded.shape == (2474,)
    np.testing.assert_array_equal(coded[:2048], x)
    np.testing.assert_array_equal(coded[2048:], x[:426])
    folded = polar.fold(np.ones(2474))
    assert np.all(folded[:426] == 2) and np.all(folded[426:] == 1)
    llr = np.arange(2474, dtype=np.float64)
    f = polar.fold(llr)
    assert f[0] == 0 + 2048 and f[425] == 425 + 2473 and f[426] == 426
    with pytest.raises(ValueError):
        polar.fold(np.zeros(2048))


def test_encoder_puts_info_on_info_set_and_freezes_the_rest():
    rng = np.random.default_rng(21)
    info = rng.integers(0, 2, 142).astype(np.uint8)
    coded = polar.polar_encode(info)
    assert coded.shape == (2474,) and coded.dtype == np.uint8
    u = polar.polar_transform(coded[:2048])
    np.testing.assert_array_equal(u[list(polar.INFO_SET)], info)
    frozen = np.setdiff1d(np.arange(2048), polar.INFO_SET)
    assert not u[frozen].any()
    np.testing.assert_array_equal(coded[2048:], coded[:426])


@pytest.mark.parametrize("h", HEADERS)
def test_noiseless_encode_decode(h):
    coded = header.encode(h)
    assert header.decode(10.0 * (1.0 - 2.0 * coded)) == h
    assert header.decode(10.0 * (1.0 - 2.0 * coded), list_size=1) == h


def test_decode_list_returns_the_codeword_first():
    info = header.pack(H)
    cands, pm = polar.decode_list(4.0 * (1.0 - 2.0 * polar.polar_encode(info)))
    np.testing.assert_array_equal(cands[0], info)
    assert np.all(np.diff(pm) >= 0) and len(cands) == 8


def test_repeated_copies_may_be_erased():
    """Zero LLRs on the 426 repeated coded bits leave the full mother code."""
    llr = 6.0 * (1.0 - 2.0 * header.encode(H))
    llr[2048:] = 0.0
    assert header.decode(llr) == H


# --- H3: CA-SCL on BPSK AWGN -------------------------------------------------------


def test_decodes_30_of_30_at_minus_8_5_db():
    rng = np.random.default_rng(30)
    coded = header.encode(H)
    ok = sum(header.decode(_llr(coded, -8.5, rng)) == H for _ in range(30))
    assert ok == 30


def test_block_error_rate_at_design_threshold_fast():
    """Measured BLER of the real L = 8 list decoder at Es/N0 = -9.4 dB.
    The GA SC bound there is 2.3e-4 and CA-SCL only improves on it, so
    100 trials should all decode; one failure is tolerated."""
    rng = np.random.default_rng(31)
    coded = header.encode(H)
    fails = sum(header.decode(_llr(coded, -9.4, rng)) != H for _ in range(100))
    assert fails <= 1


@pytest.mark.slow
def test_block_error_rate_at_design_threshold_slow():
    """H3 slow: BLER <= 1e-2 over 500 trials at Es/N0 = -9.4 dB, with a
    different header per trial."""
    rng = np.random.default_rng(32)
    fails = 0
    for t in range(500):
        h = HeaderFields("K1ABC", "FN42", int(rng.integers(0, 1 << 32)), 2,
                         int(rng.integers(0, 3)), 0xD1D8)
        fails += header.decode(_llr(header.encode(h), -9.4, rng)) != h
    assert fails / 500 <= 1e-2


def test_list_decoding_beats_successive_cancellation():
    """At -12 dB, 2.6 dB under threshold, L = 8 still decodes nearly every
    block (197/200 measured) where plain SC (L = 1) loses about a third."""
    rng = np.random.default_rng(33)
    coded = header.encode(H)
    ok = {1: 0, 8: 0}
    for _ in range(60):
        llr = _llr(coded, -12.0, rng)
        for size in ok:
            ok[size] += header.decode(llr, list_size=size) == H
    assert ok[8] >= 55
    assert ok[1] <= ok[8] - 10


def test_decode_time_under_half_a_second():
    rng = np.random.default_rng(34)
    llr = _llr(header.encode(H), -9.4, rng)
    best = np.inf
    for _ in range(3):
        t = time.perf_counter()
        header.decode(llr)
        best = min(best, time.perf_counter() - t)
    assert best < 0.5


# --- soft combining across passes ------------------------------------------------


@pytest.mark.parametrize("n_pass,esn0_db", [(2, -13.4), (4, -15.4)])
def test_soft_combining_passes(n_pass, esn0_db):
    """N passes each 10 log10(N) dB under -10.4 / -9.4 dB: one pass alone
    mostly fails (measured 22/60 at -13.4, 0/60 at -15.4) while the sum of
    the N passes' LLRs decodes every time."""
    rng = np.random.default_rng(40 + n_pass)
    coded = header.encode(H)
    single = combined = 0
    trials = 20
    for _ in range(trials):
        llrs = [_llr(coded, esn0_db, rng) for _ in range(n_pass)]
        single += header.decode(llrs[0]) == H
        combined += header.decode(np.sum(llrs, axis=0)) == H
    assert combined == trials
    assert single <= trials // 2


# --- H4: noise-only LLRs are never accepted -------------------------------------


def _noise_trials(n: int, seed: int) -> int:
    rng = np.random.default_rng(seed)
    accepted = 0
    for t in range(n):
        scale = (0.5, 2.0, 8.0)[t % 3]          # weak, threshold-like and strong LLRs
        llr = rng.normal(0.0, scale, 2474)
        if t % 5 == 4:                          # and a sum of four noise "passes"
            llr = llr + rng.normal(0.0, scale, (3, 2474)).sum(axis=0)
        accepted += header.decode(llr) is not None
    return accepted


def test_noise_is_never_accepted_fast():
    assert _noise_trials(30, 50) == 0


@pytest.mark.slow
def test_noise_is_never_accepted_slow():
    assert _noise_trials(300, 51) == 0

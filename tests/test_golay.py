import numpy as np

from sstvae.modem import golay


def test_min_distance_is_8():
    assert golay.min_distance() == 8


def test_roundtrip_all_messages_sample():
    for m in [0, 1, 0x7FF, 0xFFF, 0xABC]:
        bits = golay.codeword_bits(m)
        soft = 1.0 - 2.0 * bits
        assert golay.decode_soft(soft) == m


def test_corrects_three_hard_errors():
    rng = np.random.default_rng(7)
    for _ in range(50):
        m = int(rng.integers(0, 4096))
        bits = golay.codeword_bits(m).copy()
        flip = rng.choice(24, size=3, replace=False)
        bits[flip] ^= 1
        soft = 1.0 - 2.0 * bits
        assert golay.decode_soft(soft) == m


def test_soft_decode_with_noise():
    rng = np.random.default_rng(8)
    ok = 0
    trials = 200
    for _ in range(trials):
        m = int(rng.integers(0, 4096))
        soft = 1.0 - 2.0 * golay.codeword_bits(m) + rng.normal(scale=0.7, size=24)
        ok += golay.decode_soft(soft) == m
    assert ok / trials > 0.95

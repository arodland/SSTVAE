import numpy as np

from sstvae.modem.dsp import to_baseband, to_baseband_at


def test_to_baseband_at_matches_a_slice_of_the_whole_signal():
    """to_baseband_at(x, g) must equal to_baseband(full)[g:g+len(x)] for
    whatever longer `full` array `x` is a slice of -- including start
    offsets that are not multiples of the heterodyne's 16-sample period,
    which is exactly the case a naive to_baseband(x) gets wrong."""
    rng = np.random.default_rng(0)
    full = rng.normal(size=5000)

    for start in [0, 1, 7, 15, 16, 17, 31, 32, 33, 999, 4321]:
        for n in [1, 5, 37, 400]:
            if start + n > len(full):
                continue
            chunk = full[start : start + n]
            want = to_baseband(full)[start : start + n]
            got = to_baseband_at(chunk, start)
            np.testing.assert_allclose(got, want, atol=1e-12)


def test_to_baseband_at_zero_start_matches_to_baseband():
    rng = np.random.default_rng(1)
    x = rng.normal(size=257)
    np.testing.assert_allclose(to_baseband_at(x, 0), to_baseband(x), atol=1e-12)

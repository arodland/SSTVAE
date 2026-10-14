"""Shared by the live rows (tests/qrss_native/live.py) and the live parity
tests: the noise that fills what a slot capture did not hear.

Not a row module (the leading underscore keeps `apply` from loading it).

`live.capture_prepare` fills every sample not heard with numpy's
`default_rng(seed)` normals. The C++ carries no numpy generator: its
default (`live.hpp`'s default_noise) is the same distribution from another
generator, so a capture it prepares is the reference's in everything but
those samples. The rows therefore hand the binding numpy's own noise
(`numpy_noise`), which makes every capture the reference's bit for bit and
lets the suites compare whole pipelines; `QRSS_LIVE_NOISE=cpp` hands it
None instead, so the suites run on the C++ generator too.
"""

from __future__ import annotations

import math
import os

import numpy as np


def numpy_noise(seed, n):
    """The reference's fill, exactly as `live.capture_prepare` draws it:
    n complex64 samples, s * (re + 1j im) with s = sqrt(0.5 / ln 2), from
    np.random.default_rng(seed) (the real parts drawn first)."""
    rng = np.random.default_rng(seed)
    s = math.sqrt(0.5 / math.log(2.0))
    return (s * (rng.standard_normal(n) + 1j * rng.standard_normal(n))).astype(np.complex64)


def noise():
    """What the rows hand the binding: `numpy_noise`, or None (the C++
    generator) when QRSS_LIVE_NOISE=cpp."""
    return None if os.environ.get("QRSS_LIVE_NOISE") == "cpp" else numpy_noise

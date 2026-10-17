#!/usr/bin/env python3
"""Regenerate the QRSSTVAE header's polar `INFO_SET` by Gaussian approximation.

    tools/gen_qrss_polar.py                 # print the set, its hash and SC bound
    tools/gen_qrss_polar.py --check         # exit 1 unless it equals polar.INFO_SET
    tools/gen_qrss_polar.py --esn0 -10      # another design point (prints the diff)

**The committed literal in `sstvae/qrss/polar.py` is the definition, not
this script.** The set is on-air format: a transmitter and a receiver
that disagree about one index cannot talk. So this tool exists to show
where the literal came from and to `--check` that it still reproduces;
should it ever disagree on some platform or numpy version, that is a bug
in the tool, and the right response is to keep the literal.

Method (design section 2.6): Gaussian-approximation density evolution for
BPSK on AWGN at Es/N0 per coded bit (default -9.4 dB, SNR_2500 = -23.5 dB,
the spec's header threshold).

- A coded bit's LLR is Gaussian with mean m0 = 4 Es/N0 (variance 2 m0).
  Circular repetition sends mother bits 0..425 twice, so their folded LLR
  has mean 2 m0; the rest have m0.
- The code is x = u F^(x)11 in natural order, so the top split of the
  decoding tree is chosen by an index's **MSB**: the first half of the
  mother bits is decoded through f(x_first, x_second) and the second half
  through g = x_first + x_second, recursively. (Applying the operations
  LSB-first, as one earlier design did, gives a set whose SC block error
  rate is 1.0.)
- f maps means by m = phi^-1(1 - (1 - phi(m1))(1 - phi(m2))) with Chung's
  phi; g adds means.
- The 142 bit channels with the largest mean are the information set.
  The SC block-error upper bound is the sum of Q(sqrt(m/2)) over them.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
from scipy.special import erfc

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sstvae.qrss import polar  # noqa: E402

DESIGN_ESN0_DB = -9.4


def phi(x: np.ndarray) -> np.ndarray:
    """Chung's approximation of the GA phi function (phi(0) = 1, decreasing)."""
    x = np.asarray(x, dtype=np.float64)
    out = np.ones_like(x)
    lo = (x > 0) & (x < 10)
    hi = x >= 10
    out[lo] = np.exp(-0.4527 * x[lo] ** 0.86 + 0.0218)
    xh = x[hi]
    out[hi] = np.sqrt(np.pi / xh) * np.exp(-xh / 4) * (1 - 10 / (7 * xh))
    return out


def phi_inv(y: np.ndarray, iters: int = 200) -> np.ndarray:
    """Inverse of `phi` by bisection on [0, 1e4]."""
    y = np.asarray(y, dtype=np.float64)
    lo = np.zeros_like(y)
    hi = np.full_like(y, 1e4)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        above = phi(mid) > y
        lo = np.where(above, mid, lo)
        hi = np.where(above, hi, mid)
    return 0.5 * (lo + hi)


def bit_channel_means(esn0_db: float) -> np.ndarray:
    """GA LLR mean of every mother bit channel u_i, natural order."""
    n, e = polar.N, polar.E
    reps = np.bincount(np.arange(e) % n, minlength=n)
    m = reps * 4.0 * 10 ** (esn0_db / 10)
    m = m[None, :]                          # (blocks, size): one block at the root
    while m.shape[1] > 1:
        h = m.shape[1] // 2
        a, b = m[:, :h], m[:, h:]
        f = phi_inv(1 - (1 - phi(a)) * (1 - phi(b)))
        m = np.stack([f, a + b], axis=1).reshape(2 * m.shape[0], h)
    return m[:, 0]


def construct(esn0_db: float = DESIGN_ESN0_DB, k: int = polar.K):
    """-> (info set as a sorted tuple, bit-channel means)."""
    means = bit_channel_means(esn0_db)
    order = np.argsort(-means, kind="stable")
    return tuple(sorted(int(i) for i in order[:k])), means


def sc_bound(info, means) -> float:
    """Union bound on the SC block error rate: sum of Q(sqrt(m/2))."""
    m = means[list(info)]
    return float(np.sum(0.5 * erfc(np.sqrt(m / 4))))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--esn0", type=float, default=DESIGN_ESN0_DB,
                    help="design point, Es/N0 per coded bit in dB (default %(default)s)")
    ap.add_argument("--check", action="store_true",
                    help="exit 1 unless the regenerated set equals polar.INFO_SET")
    args = ap.parse_args(argv)

    info, means = construct(args.esn0)
    sha = hashlib.sha256(np.array(info, ">u2").tobytes()).hexdigest()
    ranked = np.sort(means)[::-1]
    margin_db = 10 * np.log10(ranked[polar.K - 1] / ranked[polar.K])
    diff = sorted(set(info) ^ set(polar.INFO_SET))
    print(f"Es/N0 {args.esn0:+.2f} dB: K = {len(info)}, sha256 {sha}")
    print(f"  SC block-error bound {sc_bound(info, means):.3g} "
          f"(committed set: {sc_bound(polar.INFO_SET, means):.3g})")
    print(f"  margin between channel {polar.K} and {polar.K + 1}: {margin_db:.3f} dB")
    print(f"  differs from polar.INFO_SET at {diff if diff else 'no index'}")
    if args.check:
        if info != polar.INFO_SET or sha != polar.INFO_SET_SHA256:
            print("CHECK FAILED: the GA construction does not reproduce INFO_SET", file=sys.stderr)
            return 1
        print("check ok")
        return 0
    if not diff:
        print("INFO_SET = (" + ", ".join(str(i) for i in info) + ")")
    return 0


if __name__ == "__main__":
    sys.exit(main())

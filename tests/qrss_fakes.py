"""Synthetic passes for the multi-pass tests (design 1, WP7).

Not a test module (no `test_` prefix). Builds `PassResult`s directly
from known air-order latents, with the receiver replaced by its
contract (design 7): z is unbiased with variance 1/w, w is the
per-latent SNR, and the header LLRs are BPSK at a stated Es/N0 per coded
bit. This is the "genie" single-pass receiver, so multi-pass tests
measure the store and association arithmetic alone, and WP9 can use the
same fakes before the real receiver exists.

- `synthetic_picture`: a `StoredPicture` from synthetic latents.
- `make_header`: the header a picture's segment is sent with.
- `noisy_llr`: header LLRs at an Es/N0 per coded bit (None: no header energy).
- `make_pass_result`: one pass.
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np

from qrss_helpers import Q_TEST, synthetic_full_latents
from sstvae.qrss import frame, header, picture
from sstvae.qrss.constants import N_HDR_BITS
from sstvae.qrss.types import PassResult, Timing, TrackReport, pass_uid


def synthetic_picture(seed: int = 0, mode: int = 0, codec_id: int = 0xD1D8):
    """A StoredPicture of synthetic (tanh-shaped) latents."""
    return picture.StoredPicture.from_latents(synthetic_full_latents(seed), codec_id, mode)


def make_header(sp, segment: int = 0, callsign: str = "K1ABC",
                grid: str | None = "FN42") -> header.HeaderFields:
    return header.HeaderFields(callsign=callsign, grid=grid,
                               picture_id=sp.picture_id, mode=sp.mode,
                               segment=segment, codec_id=sp.codec_id)


def noisy_llr(h: header.HeaderFields | None, es_n0_db: float | None, rng) -> np.ndarray:
    """float32[2474]: BPSK LLRs of h's coded bits at Es/N0 per coded bit.

    +1 for bit 0, y = s + n, sigma^2 = 1/(2 Es/N0), llr = 2y/sigma^2 (the
    convention of `tests/test_qrss_header.py`). With h or es_n0_db None,
    all zeros (a pass whose header carried nothing).
    """
    if h is None or es_n0_db is None:
        return np.zeros(N_HDR_BITS, dtype=np.float32)
    s = 1.0 - 2.0 * header.encode(h).astype(np.float64)
    sigma2 = 1.0 / (2.0 * 10 ** (es_n0_db / 10))
    y = s + np.sqrt(sigma2) * rng.standard_normal(s.size)
    return (2.0 * y / sigma2).astype(np.float32)


def _frame_name(n: int) -> str:
    for spec in frame.PRESETS.values():
        if spec.n_data == n:
            return spec.name
    return "full"


def make_pass_result(a_air, s=1.0, *, w=None, seed: int = 0, q: int = Q_TEST,
                     f_hz: float = 1500.0, hdr: header.HeaderFields | None = None,
                     decoded: bool = False, hdr_es_n0_db: float | None = None,
                     erase_after: int | None = None, z_ref: float = 10.0,
                     em_round: int = 0) -> PassResult:
    """One genie pass of the air-order latents `a_air`.

    s: per-latent SNR (linear), or pass `w` (per-latent weights) instead.
    z = a + N(0, 1/w) where w > 0; erased latents (w = 0, and every
    latent from `erase_after` on) carry z = 0. `hdr` is the header the
    pass was sent with: its LLRs are drawn at `hdr_es_n0_db`, and with
    `decoded` it is also reported as decoded on this pass. The uid
    follows `types.pass_uid`, so passes differing only in noise differ
    in uid.
    """
    rng = np.random.default_rng(seed)
    a = np.asarray(a_air, dtype=np.float64)
    n = a.size
    w = (np.full(n, float(s)) if w is None else np.asarray(w, dtype=np.float64)).copy()
    if erase_after is not None:
        w[erase_after:] = 0.0
    z = np.zeros(n)
    on = w > 0
    z[on] = a[on] + rng.standard_normal(int(on.sum())) / np.sqrt(w[on])
    z = z.astype(np.float32)
    llr = noisy_llr(hdr, hdr_es_n0_db, rng)
    spec = frame.get(_frame_name(n))
    timing = Timing(tau0=0.0, ppm=0.0, gamma=0.0, z=z_ref, cov=np.eye(2))
    report = TrackReport(offset_hz=f_hz, drift_hz_per_min=0.0, wander_hz_rms=0.0,
                         doppler_hz=0.0, ppm=0.0, snr2500_db=0.0, z_ref=z_ref,
                         kappa=1.0, slip_free=True, suspect=False)
    return PassResult(
        uid=pass_uid(q, f_hz, z), q=int(q), frame=spec.name, waveform=0,
        f_hz=float(f_hz), timing=timing, report=report,
        z=z, w=w.astype(np.float32), hdr_llr=llr,
        header=hdr if decoded else None, cw=None,
        ch=np.zeros(0, dtype=np.complex64), ch_fs=250, ch_t0_index=0.0,
        f_mix_hz=Fraction(1500), psi=np.ones(spec.n_pos, dtype=np.float32),
        estimator="joint", em_round=em_round)

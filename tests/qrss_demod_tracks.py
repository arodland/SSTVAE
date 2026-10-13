"""Tracker results for the demod lane's parity tests
(tests/test_qrss_native_demod.py and test_qrss_native_ldemod.py).

Not a test module (no `test_` prefix). Each run is the *Python* tracker
on a simulated pass, through `receiver.receive_pass(..., return_tracks=
True)` as the receiver itself drives it, so the C++ demodulators are fed
exactly what the reference's own are fed:

- `ce_tracks()`: CE TINY at -12 dB, TINY with 10 s of its data erased (psi = 0, the
  natural erasure fixture), SHORT at -6 dB (round A with the header
  unknown and its spread copy as extra variance, round B with it known),
  and a genie SHORT track.
- `l_tracks()`: L l-tiny at -12 dB, l-short at -6 dB (round A unknown with
  spread_rho, round B with hdr_bits and the measured-reference mask), and
  a genie l-tiny track.
- `synth()`: a deterministic stub track (the ctests' demod_synth.hpp),
  GenieTrack-shaped (attributes only, gi = p - pos[0]).

Cached per process; treat the results as read-only (copy.copy before
anything that replaces `m`).
"""

from __future__ import annotations

import functools
from types import SimpleNamespace

import numpy as np

from qrss_fakes import l_scenario
from qrss_helpers import Q_TEST, unit_rms_latents
from sstvae.qrss import channel as chm
from sstvae.qrss import frame, frontend, header, lframe, ltrack, morse
from sstvae.qrss import receiver as RX
from sstvae.qrss import track as T
from sstvae.qrss.channel import ChannelConfig
from sstvae.qrss.constants import WAVEFORM_CE, WAVEFORM_L
from sstvae.qrss.precoder import precode
from sstvae.qrss.types import Detection, FreqPath

CALL = "K1ABC"
HDR = header.HeaderFields(callsign=CALL, grid="FN42", picture_id=0x12345678, mode=0,
                          segment=0, codec_id=0xD1D8)


def flat_det(f0: float, wf: int = WAVEFORM_CE) -> Detection:
    flat = FreqPath(t_s=np.array([-10.0, 0.0, 1800.0]), f_hz=np.full(3, f0), weight=np.ones(3))
    return Detection(f_hz=f0, path=flat, timing=None, z_ref=np.nan, method="tbd", lead_in_s=0.0,
                     waveform=wf)


def _ce_sim(name: str, snr: float, seed: int):
    spec = frame.get(name)
    a = unit_rms_latents(spec.n_data, seed)
    sym = frame.assemble(spec, header.encode(HDR) if spec.n_hdr else None, precode(a, Q_TEST))
    sim = chm.simulate(sym, spec, Q_TEST, ChannelConfig(snr_db=snr, seed=seed), carrier_hz=1500.0,
                       keying=morse.keying_units(CALL) if spec.n_win else None)
    return spec, sim


@functools.cache
def ce_tracks() -> dict:
    """{label: TrackResult} of CE passes (module docstring)."""
    out = {}
    for label, name, snr, erase in (("tiny", "tiny", -12.0, ()),
                                    ("tiny-erased", "tiny", -12.0, ((25.0, 35.0),)),
                                    ("short", "short", -6.0, ())):
        spec, sim = _ce_sim(name, snr, 1)
        prep = RX.prepare(frontend.Capture(sim.fe, Q_TEST, sim.t0_index))
        res = RX.receive_pass(prep, spec, flat_det(1500.0), return_tracks=True, erase_s=erase)
        assert res is not None, label
        for k, tr in enumerate(res[1]):
            out[f"{label}/{'AB'[k]}"] = tr
        if label == "short":
            chan = RX.channel_for(prep, spec, flat_det(1500.0))
            out["short/genie"] = T.genie_track(chan, spec, sim.truth, 1500.0)
    return out


@functools.cache
def l_tracks() -> dict:
    """{label: LTrackResult} of L passes (module docstring)."""
    out = {}
    for name, snr in (("l-tiny", -12.0), ("l-short", -6.0)):
        spec = lframe.get(name)
        _, sim = l_scenario(name, snr, 1)
        prep = RX.prepare(frontend.Capture(sim.fe, Q_TEST, sim.t0_index))
        det = flat_det(1500.0, WAVEFORM_L)
        res = RX.receive_pass(prep, spec, det, return_tracks=True)
        assert res is not None, name
        for k, tr in enumerate(res[1]):
            out[f"{name}/{'AB'[k]}"] = tr
        if name == "l-tiny":
            chan = RX.channel_for(prep, spec, det)
            out["l-tiny/genie"] = ltrack.genie_track_l(chan, spec, sim.truth, 1500.0)
    return out


def synth(spec, spread_rho: float = 0.0, hdr_bits=None, with_meas: bool = False):
    """The deterministic stub of native/tests/qrss/demod_synth.hpp."""
    n = spec.n_pos + 40
    i = np.arange(n)
    u = (0.75 + (i % 3) * 0.25) + 1j * (((i % 4) - 1.5) * 0.125)
    m = (((i * 7) % 13 - 6) * 0.125) + 1j * (((i * 5) % 11 - 5) * 0.0625)
    ns = SimpleNamespace(
        spec=spec, pos=np.arange(-40, spec.n_pos, dtype=np.int64), m=m, u=u,
        P=0.001 * (1 + i % 5), N=0.25 + 0.0625 * (i % 7),
        psi=np.where(i % 97 == 0, 0.25, 1.0), c=0.5 + 1j * (((i % 2) * 2 - 1) * 0.375),
        classes=SimpleNamespace(spread_rho=spread_rho, hdr_bits=hdr_bits),
        meas=(i % 3 == 0) if with_meas else None)
    ns.gi = lambda p: np.asarray(p, dtype=np.int64) + 40
    return ns


def known_bits() -> np.ndarray:
    """uint8[2474]: the stub's header bits (demod_synth.hpp's)."""
    return ((np.arange(2474) * 7 + 3) % 5 < 2).astype(np.uint8)


# ---- comparisons ---------------------------------------------------------------------

def rel(a, b) -> float:
    """Worst elementwise |a - b| / |b| (inf and nan where both are)."""
    a, b = np.asarray(a), np.asarray(b)
    assert a.shape == b.shape, (a.shape, b.shape)
    assert a.dtype == b.dtype, (a.dtype, b.dtype)
    if a.size == 0:
        return 0.0
    fa, fb = np.isfinite(a), np.isfinite(b)
    assert np.array_equal(fa, fb)
    assert np.array_equal(a[~fa], b[~fb], equal_nan=True)
    if not fa.any():
        return 0.0
    d = np.abs(a[fa] - b[fb])
    return float(np.max(d / np.maximum(np.abs(b[fb]), 1e-300)))


def ulps(a, b) -> tuple[int, float]:
    """(worst distance in ulps, fraction bit-identical) of two float arrays."""
    a, b = np.asarray(a), np.asarray(b)
    assert a.shape == b.shape, (a.shape, b.shape)
    assert a.dtype == b.dtype, (a.dtype, b.dtype)
    if a.dtype.kind == "c":
        a, b = a.view(a.real.dtype), b.view(b.real.dtype)
    if a.size == 0:
        return 0, 1.0
    it = {4: np.int32, 8: np.int64}[a.dtype.itemsize]
    same = (a.view(it) == b.view(it)) | (np.isnan(a) & np.isnan(b))
    fin = np.isfinite(a) & np.isfinite(b)
    assert not ((~np.isfinite(a) | ~np.isfinite(b)) & ~same).any(), "non-finite values differ"

    def ordered(x):
        i = x.view(it).astype(np.int64)
        return np.where(i < 0, np.iinfo(it).min - i, i)
    d = np.abs(ordered(a[fin]) - ordered(b[fin]))
    return int(d.max()) if d.size else 0, float(same.mean())

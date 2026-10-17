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
- `make_pass_result`: one pass (either waveform's frame names).

Waveform L passes through the real channel simulator, kept apart from
CE's `test_qrss_receiver.scenario` (own cache, L frame names):

- `l_symbols`, `l_keying`: an L pass of given latents, as a sender assembles it.
- `l_scenario`: (latents, Sim) of one simulated L pass, cached.
- `l_genie`: the genie L demodulator at true timing and true data-band gain.
"""

from __future__ import annotations

import functools
from fractions import Fraction

import numpy as np

from qrss_helpers import Q_TEST, synthetic_full_latents, unit_rms_latents
from sstvae.qrss import channel, frame, frontend, header, lframe, lwave, picture
from sstvae.qrss.channel import ChannelConfig
from sstvae.qrss.constants import (CH_FS, FE_CENTER_HZ, FE_FS, N_HDR_BITS, WAVEFORM_CE,
                                   WAVEFORM_L)
from sstvae.qrss.morse import keying_units
from sstvae.qrss.precoder import precode
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


def _frame_name(n: int, waveform: int = WAVEFORM_CE) -> str:
    presets = lframe.PRESETS if waveform == WAVEFORM_L else frame.PRESETS
    for spec in presets.values():
        if spec.n_data == n:
            return spec.name
    return "l-full" if waveform == WAVEFORM_L else "full"


def make_pass_result(a_air, s=1.0, *, w=None, seed: int = 0, q: int | None = None,
                     f_hz: float = 1500.0, hdr: header.HeaderFields | None = None,
                     decoded: bool = False, hdr_es_n0_db: float | None = None,
                     erase_after: int | None = None, z_ref: float = 10.0,
                     em_round: int = 0, waveform: int = WAVEFORM_CE) -> PassResult:
    """One genie pass of the air-order latents `a_air`.

    s: per-latent SNR (linear), or pass `w` (per-latent weights) instead.
    z = a + N(0, 1/w) where w > 0; erased latents (w = 0, and every
    latent from `erase_after` on) carry z = 0. `hdr` is the header the
    pass was sent with: its LLRs are drawn at `hdr_es_n0_db`, and with
    `decoded` it is also reported as decoded on this pass. The uid
    follows `types.pass_uid`, so passes differing only in noise differ
    in uid. q defaults to Q_TEST - seed: passes of one picture at one
    frequency are different transmissions only in different slots (the
    store keeps one member per transmission, `Store.twin`). `waveform`
    picks the frame of that waveform holding n latents ("short" or
    "l-short"; f_hz is then an L pilot).
    """
    if q is None:
        q = Q_TEST - int(seed)
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
    spec = lframe.get_any(_frame_name(n, waveform))
    timing = Timing(tau0=0.0, ppm=0.0, gamma=0.0, z=z_ref, cov=np.eye(2))
    report = TrackReport(offset_hz=f_hz, drift_hz_per_min=0.0, wander_hz_rms=0.0,
                         doppler_hz=0.0, ppm=0.0, snr2500_db=0.0, z_ref=z_ref,
                         kappa=1.0, slip_free=True, suspect=False)
    return PassResult(
        uid=pass_uid(q, f_hz, z), q=int(q), frame=spec.name, waveform=int(waveform),
        f_hz=float(f_hz), timing=timing, report=report,
        z=z, w=w.astype(np.float32), hdr_llr=llr,
        header=hdr if decoded else None, cw=None,
        ch=np.zeros(0, dtype=np.complex64), ch_fs=250, ch_t0_index=0.0,
        f_mix_hz=Fraction(1500), psi=np.ones(spec.n_pos, dtype=np.float32),
        estimator="joint", em_round=em_round)


# --- waveform L through the channel simulator -----------------------------------------

L_CALL = "K1ABC"
# `test_qrss_receiver.HDR`'s fields, so an L and a CE pass can carry one header.
L_HDR = header.HeaderFields(callsign=L_CALL, grid="FN42", picture_id=0x12345678, mode=0,
                            segment=0, codec_id=0xD1D8)


def l_symbols(spec, a, hdr: header.HeaderFields | None = L_HDR, q: int = Q_TEST) -> np.ndarray:
    """complex128[n_sym]: an L pass of the air-order latents `a` (precoded in
    slot q), with `hdr`'s coded bits where the frame has a header."""
    spec = lframe.get(spec)
    bits = header.encode(hdr) if spec.has_header else None
    return lframe.assemble(spec, bits, precode(np.asarray(a)[:spec.n_data], q))


def l_keying(spec, call: str = L_CALL):
    """The callsign keying an L frame with windows must send, else None."""
    return keying_units(call) if lframe.get(spec).n_win else None


@functools.lru_cache(maxsize=32)
def l_scenario(spec_name: str = "l-short", snr: float | None = -12.0, seed: int = 1,
               pilot: float = 1500.0, lead_in: float = 0.0, cfg: tuple = ()):
    """(latents, Sim) of one simulated L pass, cached: treat both as read-only.

    Unit-RMS Gaussian latents drawn from `seed` (also the channel's seed),
    L_HDR's header where the frame has one and L_CALL's callsign windows,
    the pilot at `pilot` Hz; cfg is further ChannelConfig kwargs as a
    sorted tuple (neighbours, presets and the rest).
    """
    spec = lframe.get(spec_name)
    a = unit_rms_latents(spec.n_data, seed)
    sim = channel.simulate(l_symbols(spec, a), spec, Q_TEST,
                           ChannelConfig(snr_db=snr, seed=seed, **dict(cfg)),
                           carrier_hz=pilot, lead_in_s=lead_in, keying=l_keying(spec))
    return a, sim


def l_genie(sim, pilot_hz: float = 1500.0, gain=None) -> np.ndarray:
    """complex128[n_sym]: the genie L demodulator, in stream order.

    The front end channelised at the nominal pilot (`pilot_hz`, as given
    to `channel.simulate`; on the channeliser's 1/64 Hz grid) to CH_FS,
    mixed up by f_p to the data centre (`lwave.to_data_centre`), then L's
    matched filter (`lwave.matched_filter`) at every stream symbol's true
    position, divided by the true gain there: `gain` per position, default
    `truth.u_data`. A symbol of class gain g (about 0.89 for data, 1 for
    references) arrives as g a plus noise. True timing on the grid only:
    no clock error, and t0_index a whole number of CH samples (simulate's
    default pre_s).
    """
    spec, tr = sim.spec, sim.truth
    if spec.waveform != WAVEFORM_L:
        raise ValueError(f"l_genie takes an L pass, not frame {spec.name!r}")
    decim = FE_FS // CH_FS
    if tr.time_scale != 1.0 or sim.fs != FE_FS or sim.t0_index % decim:
        raise ValueError("l_genie needs true timing on the grid: no clock error and "
                         "t0_index a whole number of CH samples")
    f_mix = pilot_hz - FE_CENTER_HZ
    ch, q = frontend.channelise(sim.fe, FE_FS, f_mix, CH_FS, n0=-int(sim.t0_index))
    if float(q) != f_mix:
        raise ValueError(f"pilot {pilot_hz} Hz is not on the channeliser's 1/64 Hz grid")
    n0 = -int(sim.t0_index) // decim
    zd = lwave.to_data_centre(lwave.Baseband(z=ch.astype(np.complex128), n0=n0, fs=CH_FS))
    pos = lframe.layout(spec).pos
    m = lwave.matched_filter(zd, CH_FS, n0, pos)
    g = tr.u_data if gain is None else np.asarray(gain)
    return m / g[pos]

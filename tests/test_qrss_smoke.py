"""QRSSTVAE fast end-to-end smoke tests.

The receiver-level tests elsewhere run one SHORT pass each, and a SHORT
pass at -12 dB takes 10-15 s to receive (almost all of it in `detect`'s
gate-V verification of the grating-lobe candidates), so most of them are
`slow`. These two keep the whole chain in the default run:

- TINY: a synthetic picture through `tx.transmit_audio` (8 kHz audio),
  `channel.simulate_audio`, blind `receiver.receive_slot`, the store
  (`add_pass` / `attach`) and `render.decoder_planes`; then the v5
  decoder renders it when the models are present (without them only
  that last step is left out, unless SSTVAE_REQUIRE_CODEC is set, which
  makes it a failure). About 4 s.
- SHORT at -6 dB (where `detect` has one candidate to verify, so a pass
  is about 5 s): header decoded through round B, callsign windows read
  and matched, the pass report, timing against the simulator's truth.
  The same checks at -15 dB (and the R-table around them) are in
  `test_qrss_receiver.py` and `test_qrss_cwid.py`, `slow`.
"""

import numpy as np
import pytest

from qrss_fakes import make_header, synthetic_picture
from qrss_helpers import Q_TEST, latent_snr_db
from sstvae.qrss import channel as chm
from sstvae.qrss import cwid, demod, frame, frontend, picture, render, tx
from sstvae.qrss import receiver as RX
from sstvae.qrss.channel import ChannelConfig
from sstvae.qrss.store import Store

from test_qrss_receiver import CALL, G_CE_DB, HDR, received, truth_timing_err_T

TINY = frame.TINY


def _theory_db(snr_db: float) -> float:
    return -10 * np.log10(10 ** (-(snr_db + G_CE_DB) / 10) + demod.D_PASS)


def _codec_or_none():
    """The v5 codec, or None when its models are absent (a failure instead
    when SSTVAE_REQUIRE_CODEC is set, via `qrss_helpers.codec_skip`)."""
    import os

    from qrss_helpers import load_codec
    if os.environ.get("SSTVAE_REQUIRE_CODEC"):
        return load_codec("fp16")
    try:
        return load_codec("fp16")
    except pytest.skip.Exception:
        return None


def test_tiny_transmit_channel_receive_decode(tmp_path):
    """One TINY slot at -12 dB, transmit audio to rendered picture."""
    sp = synthetic_picture(0)
    h = make_header(sp, callsign=CALL)
    a = tx.slot_air_latents(sp, 0)[:TINY.n_data]

    audio = tx.transmit_audio(sp, Q_TEST, 1500, spec=TINY)
    sim = chm.simulate_audio(audio, TINY, Q_TEST, ChannelConfig(snr_db=-12.0, seed=1))
    passes = RX.receive_slot(frontend.Capture(sim.fe, Q_TEST, sim.t0_index), TINY)

    assert len(passes) == 1, [p.f_hz for p in passes]
    p = passes[0]
    assert p.frame == "tiny" and p.q == Q_TEST and p.header is None and p.cw is None
    assert abs(p.f_hz - 1500.0) < 0.1
    assert p.z.shape == (TINY.n_data,) and p.z.dtype == np.float32
    assert np.all(np.isfinite(p.z)) and np.all(p.w > 0)
    got = latent_snr_db(p.z, a)
    print(f"TINY smoke: latent SNR {got:.2f} dB (theory {_theory_db(-12.0):.2f}), "
          f"mean W {10 * np.log10(np.mean(p.w)):.2f} dB, Z_ref {p.report.z_ref:.1f}")
    assert got >= _theory_db(-12.0) - 1.5
    assert abs(got - 10 * np.log10(np.mean(p.w))) <= 1.5      # W is about honest

    st = Store(tmp_path / "store")
    st.add_pass(p)
    key = (CALL, int(sp.picture_id))
    acc = st.attach(key, p.uid, 0, "test", mode=h.mode, codec_id=h.codec_id)
    assert acc.uids == [p.uid] and st.open_keys() == [key]
    lat, wt = render.decoder_planes(acc.S, acc.W, acc.mode)
    canon = picture.air_to_canonical(0)[:TINY.n_data]
    heard = np.flatnonzero(wt > 0)
    assert np.array_equal(heard, np.sort(canon))          # exactly the sent latents
    # ... carrying the transmitted values: the shrunk latents are a scaled
    # noisy copy, so their correlation is sqrt(S/(1 + S)) at latent SNR S
    rho = np.corrcoef(lat[canon], a)[0, 1]
    s_lin = 10 ** (got / 10)
    print(f"TINY smoke: decoder-plane correlation {rho:.3f} "
          f"(sqrt(S/(1+S)) = {np.sqrt(s_lin / (1 + s_lin)):.3f})")
    assert rho >= np.sqrt(s_lin / (1 + s_lin)) - 0.05

    codec = _codec_or_none()
    if codec is None:
        print("TINY smoke: v5 models absent, picture not rendered")
        return
    im = np.asarray(render.render(codec, acc.S, acc.W, acc.mode))
    assert im.shape == (480, 640, 3) and im.dtype == np.uint8


def test_short_pass_through_the_receiver():
    """One SHORT pass at -6 dB: header, callsign windows, report and timing."""
    p, trs, a, sim, ndet = received(snr=-6.0)
    assert ndet == 1 and p is not None and len(trs) == 2      # rounds A and B
    assert p.header == HDR
    assert p.cw is not None and p.cw.agrees and p.cw.keying == "fsk" and p.cw.text == CALL
    spec = frame.SHORT
    tr = trs[0]
    assert cwid.keying_type(None, tr, spec, CALL) == "fsk"
    soft = cwid.window_soft(None, 250, tr, spec)
    assert soft.shape == (spec.n_win, 176) and cwid.match(soft, CALL) > 20
    assert cwid.read(cwid.window_llr(None, 250, tr, spec)) == CALL
    got = latent_snr_db(p.z, a)
    print(f"SHORT smoke at -6 dB: latent SNR {got:.2f} dB (theory {_theory_db(-6.0):.2f}), "
          f"report {p.report.snr2500_db:.2f} dB")
    assert got >= _theory_db(-6.0) - 0.5
    assert abs(p.report.snr2500_db + 6) < 1.0
    assert p.report.slip_free and not p.report.suspect
    assert abs(p.f_hz - 1500.0) < 0.05
    assert truth_timing_err_T(p, sim, spec) < 0.02
    assert p.z.dtype == np.float32 and p.w.dtype == np.float32 and len(p.hdr_llr) == 2474

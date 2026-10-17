"""The spread header (spec 5.1, format version 2; docs/qrss/spread-header.md).

Fast: the format (a spread frame is the plain frame plus sqrt(rho) h on
the data; every frame with a header block carries it, and the version 1
frames are the same shapes without it), the folded LLRs, the version
fallback, and the property the whole scheme rests on -- removing the
known header phase from a spread frame's signal leaves exactly the plain
frame's signal.
Slow: a frame whose only header is the spread one, received end to end,
decodes it, and once it is known delivers the plain frame's latents.
"""

import numpy as np
import pytest

from qrss_helpers import Q_TEST, latent_snr_db, unit_rms_latents
from sstvae.qrss import ce, demod, frame, header, precoder
from sstvae.qrss import track as T
from sstvae.qrss.constants import CH_FS
from sstvae.qrss.types import Timing

HDR = header.HeaderFields(callsign="K1ABC", grid="FN42", picture_id=0x12345678, mode=0,
                          segment=0, codec_id=0xD1D8)
# a frame long enough for every coded bit to be sent 3+ times, no block, no windows
SPREAD_ONLY = frame.FrameSpec("test-spreadonly", n_hdr=0, n_data=8192, cw_after=(),
                              hdr_rho=0.3)


def _x(spec, seed=0):
    return precoder.precode(unit_rms_latents(spec.n_data, seed), Q_TEST)


def test_spread_frame_is_plain_frame_plus_header():
    bits = header.encode(HDR)
    x = _x(frame.SHORT)
    plain = frame.assemble(frame.SHORT_V1, bits, x)
    spread = frame.assemble(frame.SHORT, bits, x)
    lay = frame.layout(frame.SHORT)
    d = spread - plain
    np.testing.assert_array_equal(np.delete(d, lay.data), 0.0)
    h = frame.spread_signs(frame.SHORT, bits)
    np.testing.assert_allclose(d[lay.data], np.sqrt(frame.SPREAD_RHO) * h, atol=1e-15)
    np.testing.assert_allclose(frame.spread_symbols(frame.SHORT, bits), d, atol=1e-15)
    # the same codeword on every repeat, under the whitener
    c = 1.0 - 2.0 * bits
    np.testing.assert_array_equal(h * frame.spread_whitener(frame.SHORT.n_data),
                                  c[np.arange(frame.SHORT.n_data) % 2474])


def test_every_header_block_carries_the_spread_copy():
    """Format version 2: a frame with the header block has the spread copy
    unless it says otherwise; the version 1 frames are the same shapes
    without it."""
    for name, spec in frame.PRESETS.items():
        assert spec.hdr_rho == (frame.SPREAD_RHO if spec.n_hdr else 0.0), name
    assert frame.SPREAD_RHO == 0.054 and header.FORMAT_VERSION == 2
    assert frame.FrameSpec("custom", n_data=2000, cw_after=()).hdr_rho == frame.SPREAD_RHO
    for v1 in frame.LEGACY.values():
        spec = frame.get(v1.name.removesuffix("-v1"))
        assert v1.hdr_rho == 0.0 and frame.plain(spec) is v1 and frame.get(v1.name) is v1
        assert frame.layout(v1).data.tolist() == frame.layout(spec).data.tolist()
    assert frame.plain(frame.TINY) is frame.TINY


def test_rho_zero_and_header_requirement():
    assert not frame.TINY.has_header and SPREAD_ONLY.has_header
    with pytest.raises(ValueError):
        frame.assemble(SPREAD_ONLY, None, _x(SPREAD_ONLY))
    with pytest.raises(ValueError):
        frame.FrameSpec("bad", hdr_rho=1.0)
    assert frame.FULL_SPREADONLY.n_pos == frame.FULL.n_pos - 2640


def test_spread_llr_folds_to_the_codeword():
    spec = SPREAD_ONLY
    bits = header.encode(HDR)
    rng = np.random.default_rng(3)
    y = np.sqrt(spec.hdr_rho) * frame.spread_signs(spec, bits) + rng.standard_normal(spec.n_data)
    llr = demod.spread_llr(y, np.zeros(spec.n_data), spec)
    assert llr.shape == (2474,)
    assert header.decode(llr) == HDR
    # an erased symbol contributes nothing
    s2 = np.zeros(spec.n_data)
    s2[:100] = np.inf
    assert np.all(np.isfinite(demod.spread_llr(y, s2, spec)))


def test_removing_the_known_phase_leaves_the_plain_frame():
    """CE is a pure phase signal: exp(-j phi_h) applied at the receiver's
    timing turns the spread frame's baseband into the plain frame's."""
    spec = frame.SHORT
    bits = header.encode(HDR)
    x = _x(spec)
    plain = frame.assemble(frame.SHORT_V1, bits, x)
    spread = frame.assemble(spec, bits, x)
    n0, n = ce._loopback_span(spec, CH_FS)
    zs = ce.baseband(spread, spec, CH_FS, n0 / CH_FS, n, ramp_s=0.0)
    zp = ce.baseband(plain, frame.SHORT_V1, CH_FS, n0 / CH_FS, n, ramp_s=0.0)

    class Chan:                               # sample i is at t0 + (n0 + i)/fs
        t0_index = -n0
    tm = Timing(tau0=0.0, ppm=0.0, gamma=0.0, z=0.0, cov=np.zeros((3, 3)))
    out = T.remove_spread(zs, Chan, spec, tm, frame.spread_symbols(spec, bits))
    assert np.max(np.abs(out - zp)) < 1e-9
    assert np.max(np.abs(zs - zp)) > 0.1      # and there was something to remove


def test_round_a_classes_carry_the_unknown_header_variance():
    spec = frame.SHORT
    lay = frame.layout(spec)
    a = T.make_classes(spec)
    assert a.spread is None and a.spread_rho == spec.hdr_rho
    np.testing.assert_allclose(a.nu[lay.data], 1.0 + spec.hdr_rho)
    b = T.make_classes(spec, header.encode(HDR))
    assert b.spread is not None and b.spread_rho == 0.0
    np.testing.assert_allclose(b.nu[lay.data], 1.0)
    plain = T.make_classes(frame.SHORT_V1)
    assert plain.spread is None and plain.spread_rho == 0.0


def _llr(bits, gain, rng):
    """LLRs of a coded header at `gain` (0: pure noise)."""
    c = 1.0 - 2.0 * np.asarray(bits, float)
    return 4.0 * gain * c + 2.0 * np.sqrt(2.0 * gain + 1.0) * rng.standard_normal(len(c))


def test_header_versions():
    """Version 2 is what a header says and what a decode accepts; version 1
    decodes only when asked for (the block-alone fallback)."""
    v1 = header.HeaderFields(**{**HDR.__dict__, "version": 1})
    assert HDR.version == 2
    llr = 8.0 * (1.0 - 2.0 * header.encode(v1))
    assert header.decode(llr) is None
    assert header.decode(llr, versions=(1, 2)) == v1


def test_a_version_1_pass_falls_back_to_its_block():
    """Round A on a version 1 recording: the spread LLRs are noise, so the
    sum may fail; the block alone decodes it, says version 1, and the pass
    goes on as the frame without the spread copy (round B removes
    nothing). A version 2 pass whose block faded decodes from the spread
    copy alone."""
    from sstvae.qrss.receiver import decode_header
    rng = np.random.default_rng(7)
    v1 = header.HeaderFields(**{**HDR.__dict__, "version": 1})
    block = _llr(header.encode(v1), 0.6, rng)
    spread = 30.0 * rng.standard_normal(2474)          # loud noise: the sum fails
    diag = dict(llr_block=block, llr_spread=spread)
    assert header.decode(block + spread) is None
    hdr, llr, spec = decode_header(block + spread, diag, frame.FULL)
    assert hdr == v1 and spec is frame.FULL_V1
    np.testing.assert_array_equal(llr, block)
    # version 2, block faded: noise there, the spread copy decodes
    spread = _llr(header.encode(HDR), 0.6, rng)
    block = 30.0 * rng.standard_normal(2474)
    hdr, llr, spec = decode_header(block + spread, dict(llr_block=block, llr_spread=spread),
                                    frame.FULL)
    assert hdr == HDR and spec is frame.FULL
    # a frame named as version 1 (an old recording) takes either version
    hdr, _, spec = decode_header(_llr(header.encode(v1), 0.6, rng), {}, frame.FULL_V1)
    assert hdr == v1 and spec is frame.FULL_V1
    hdr, _, _ = decode_header(_llr(header.encode(HDR), 0.6, rng), {}, frame.FULL_V1)
    assert hdr == HDR
    # and a frame without a header has nothing to decode
    assert decode_header(np.zeros(2474), {}, frame.TINY)[0] is None


@pytest.mark.slow
def test_spread_only_header_received():
    """A frame with no header block: the spread copy alone decodes the
    header (round A), and round B, with the header phase removed, gives the
    plain frame's latents to 0.1 dB on the same noise."""
    from sstvae.qrss import channel as chm
    from sstvae.qrss import frontend
    from sstvae.qrss import receiver as RX
    from sstvae.qrss.channel import ChannelConfig
    from sstvae.qrss.types import Detection, FreqPath

    bits = header.encode(HDR)
    a = unit_rms_latents(SPREAD_ONLY.n_data, 5)
    x = precoder.precode(a, Q_TEST)
    plain_spec = frame.FrameSpec("test-plain", n_hdr=0, n_data=8192, cw_after=())
    out = {}
    for spec, b in ((SPREAD_ONLY, bits), (plain_spec, None)):
        sim = chm.simulate(frame.assemble(spec, b, x), spec, Q_TEST,
                           ChannelConfig(snr_db=-12.0, seed=5), carrier_hz=1500.0)
        prep = RX.prepare(frontend.Capture(sim.fe, Q_TEST, sim.t0_index))
        path = FreqPath(t_s=np.array([-10.0, 0.0, 1800.0]), f_hz=np.full(3, 1500.0),
                        weight=np.ones(3))
        det = Detection(f_hz=1500.0, path=path, timing=None, z_ref=np.nan, method="genie",
                        lead_in_s=0.0)
        p, trs = RX.receive_pass(prep, spec, det, return_tracks=True)
        out[spec.name] = (p, trs)
    p, trs = out["test-spreadonly"]
    assert p.header == HDR and len(trs) == 2
    lat_spread = latent_snr_db(p.z, a)
    lat_plain = latent_snr_db(out["test-plain"][0].z, a)
    print(f"spread-only round B {lat_spread:.2f} dB, plain {lat_plain:.2f} dB")
    assert abs(lat_spread - lat_plain) < 0.1


def test_a_stored_version_1_pass_is_re_received_without_the_copy():
    """A pass stored before format version 2 says frame "full", which now
    names the frame with the spread copy; its header's version says which
    frame it was sent in, so EM re-receives it without the copy."""
    from types import SimpleNamespace

    from sstvae.qrss import em
    v1 = header.HeaderFields(**{**HDR.__dict__, "version": 1})
    assert em.pass_spec(SimpleNamespace(frame="full", header=v1)) is frame.FULL_V1
    assert em.pass_spec(SimpleNamespace(frame="full-v1", header=v1)) is frame.FULL_V1
    assert em.pass_spec(SimpleNamespace(frame="full", header=HDR)) is frame.FULL
    assert em.pass_spec(SimpleNamespace(frame="short", header=None)) is frame.SHORT

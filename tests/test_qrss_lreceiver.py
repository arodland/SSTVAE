"""The receiver with waveform L, and with both waveforms at once.

`receiver.receive_slot` / `detect` / `candidates` take a frame or a
tuple of frames: CE alone is the receiver's original path, call for
call (test_qrss_receiver.py covers it), and anything holding an L frame
the joint one (`receiver.is_joint`).

Fast: the frame selection and band centres, the L header decode
(version 2 only, from the sum or either copy), a detection of the wrong
waveform refused, the joint detect and receive merging by centre with
aliases and ghosts per waveform at its own symbol and the
suspect-neighbour rule across waveforms (stub gates and receives), a
CE selection taking CE's own acquisition and gate (stubs), the CE
accept re-checked against an L signal (stub), a lead-in measured at L's
symbol, L confidence over time, the store's waveform and frame and its
one member per transmission by centre, headerless passes filed side by
side, EM and the retro search leaving L alone, the two command lines (in
process, the receiver stubbed), and one blind l-tiny pass at -12 dB
through the real joint receiver. Slow: l-short at -6 and -24 dB, noise,
an L and a CE signal side by side either way round, the residual
re-check on real TINY gate-V tracks at -6 dB and on a real l-short
signal, and qrss_receive.py on an L recording.
Latent SNR floors sit about 0.5 dB under the chain's measurements of
2026-10-12, each within 0.3 dB of the genie.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest

from qrss_fakes import L_CALL, L_HDR, l_scenario, make_pass_result, noisy_llr, synthetic_picture
from qrss_helpers import Q_TEST, REPO_ROOT, latent_snr_db, unit_rms_latents
from sstvae.qrss import associate as asc
from sstvae.qrss import channel as chm
from sstvae.qrss import em, frame, frontend, header, lframe, ltrack, picture, rules, track
from sstvae.qrss import receiver as RX
from sstvae.qrss.channel import Neighbour
from sstvae.qrss.constants import CH_FS, WAVEFORM_CE, WAVEFORM_L
from sstvae.qrss.morse import keying_units
from sstvae.qrss.store import Store
from sstvae.qrss.types import CwIdResult, Detection, FreqPath, Timing

CE, L = WAVEFORM_CE, WAVEFORM_L
SHORT, TINY, FULL = frame.SHORT, frame.TINY, frame.FULL
L_SHORT, L_TINY, L_FULL = lframe.L_SHORT, lframe.L_TINY, lframe.L_FULL
OFF = RX.L_EXCLUDE_OFFSET_HZ


def _script(name):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _det(f_hz, waveform=CE, method="tbd", z_ref=float("nan"), tau0=0.0):
    t = np.array([-10.0, 0.0, 1800.0])
    timing = None if not np.isfinite(z_ref) else Timing(tau0=tau0, ppm=0.0, gamma=0.0,
                                                        z=z_ref, cov=np.eye(2))
    return Detection(f_hz=float(f_hz), path=FreqPath(t, np.full(3, float(f_hz)), np.ones(3)),
                     timing=timing, z_ref=z_ref, method=method, lead_in_s=0.0,
                     waveform=waveform)


def _cap(sim):
    return frontend.Capture(sim.fe, Q_TEST, sim.t0_index)


# --- the selection and the band centre ----------------------------------------------------------

def test_frames_and_joint():
    assert RX.frames(SHORT) == (SHORT,) and RX.frames((SHORT,)) == (SHORT,)
    assert RX.frames(L_SHORT) == (L_SHORT,)
    assert RX.frames((L_SHORT, SHORT)) == (SHORT, L_SHORT)        # CE first
    assert not RX.is_joint(SHORT) and not RX.is_joint((SHORT,))
    assert RX.is_joint(L_SHORT) and RX.is_joint((L_SHORT,)) and RX.is_joint((SHORT, L_SHORT))
    with pytest.raises(ValueError):
        RX.frames((SHORT, TINY))                                  # two CE frames
    with pytest.raises(ValueError):
        RX.frames(())


def test_the_centre_of_a_signal():
    assert OFF == -9.5
    assert RX.centre_hz(_det(1509.5, L)) == 1500.0
    assert RX.centre_hz(_det(1500.0, CE)) == 1500.0
    assert RX.centre_hz(SimpleNamespace(f_hz=1500.0)) == 1500.0   # no waveform: CE
    p = make_pass_result(unit_rms_latents(L_SHORT.n_data, 0), 1.0, f_hz=1509.5, waveform=L)
    assert RX.centre_hz(p) == 1500.0


# --- the L header -------------------------------------------------------------------------------

def test_l_header_is_version_2_from_the_sum_or_either_copy():
    rng = np.random.default_rng(5)
    good = noisy_llr(L_HDR, 4.0, rng)
    nothing = np.zeros_like(good)
    h, llr, spec = RX.decode_header(good, {}, L_SHORT)
    assert h == L_HDR and llr is good and spec is L_SHORT
    for key in ("llr_spread", "llr_block"):
        diag = {"llr_spread": nothing, "llr_block": nothing, key: good}
        h, llr, spec = RX.decode_header(nothing, diag, L_SHORT)
        assert h == L_HDR, key
        assert llr is nothing and spec is L_SHORT                # the sum is kept, frame as is
    # version 1 is never accepted from an L frame, in any copy; CE's block takes it
    v1 = noisy_llr(dataclasses.replace(L_HDR, version=1), 4.0, rng)
    assert RX.decode_header(v1, {"llr_spread": v1, "llr_block": v1}, L_SHORT)[0] is None
    h, _, spec = RX.decode_header(nothing, {"llr_spread": nothing, "llr_block": v1}, SHORT)
    assert h.version == 1 and spec.hdr_rho == 0
    # a frame without a header
    assert RX.decode_header(good, {}, L_TINY) == (None, good, L_TINY)


def test_receive_pass_refuses_a_detection_of_the_other_waveform():
    with pytest.raises(ValueError, match="twin"):
        RX.receive_pass(None, L_SHORT, _det(1500.0, CE))
    with pytest.raises(ValueError, match="twin"):
        RX.receive_pass(None, SHORT, _det(1509.5, L))


def test_receive_pass_l_refuses_a_frame_that_is_not_l():
    """_receive_pass_l, handed a CE frame (or a detection of the other
    waveform), refuses with receive_pass's own message -- not an error from
    deep inside ltrack's layout."""
    kw = dict(exclude_hz=(), f_mix_hz=None, round_b=True, return_tracks=False, erase_s=())
    for spec, d, as_wf in ((SHORT, _det(1509.5, L), L), (SHORT, _det(1500.0, CE), L),
                           (L_SHORT, _det(1500.0, CE), CE)):
        with pytest.raises(ValueError) as want:
            RX.receive_pass(None, spec, _det(1500.0, as_wf))
        with pytest.raises(ValueError) as got:
            RX._receive_pass_l(None, spec, d, None, "joint", **kw)
        assert str(got.value) == str(want.value) and "twin" in str(got.value), (spec.name, d.waveform)


# --- the joint detect and receive (stubs) ------------------------------------------------------

def test_joint_detect_merges_by_centre_and_ghosts_per_waveform(monkeypatch):
    # gated as before 2026-10-12: now the CE line at 1507 is in the L
    # signal's own band and is not gated at all (test_qrss_rules.py)
    monkeypatch.setenv(rules.GATING, "1")
    calls = []
    out = {  # candidate f -> what gate V makes of it (f, Z_ref, tau0)
        1509.5: (1509.5, 30.0, 0.0),   # L, centre 1500
        1507.0: (1503.0, 20.0, 0.0),   # CE moved to 1503: the L's signal, weaker
        1527.5: (1527.5, 8.0, 0.0),    # L, centre 1518: a ghost of the L at 1509.5
        1525.0: (1525.0, 7.0, 0.0),    # CE 15.5 Hz from the L pilot: no L's ghost
    }

    def gate(prep, spec, d, others, erase_s):
        calls.append((d.f_hz, spec.name, others))
        f, z, tau = out[d.f_hz]
        return _det(f, d.waveform, d.method, z, tau)

    monkeypatch.setattr(RX, "_gate", gate)
    cands = [_det(1509.5, L, "preamble"), _det(1507.0, CE), _det(1527.5, L), _det(1525.0, CE),
             _det(1700.0, CE), _det(1810.0, L)]
    skip = [_det(1709.5, L, z_ref=9.0), 1801.0]                # centres 1700 and 1801
    dets = RX.detect(None, (SHORT, L_SHORT), cands=cands, skip_hz=skip)
    assert [c[:2] for c in calls] == [(1509.5, "l-short"), (1507.0, "short"),
                                     (1527.5, "l-short"), (1525.0, "short")]
    assert calls[0][2] == (1700.0, 1801.0)
    assert calls[1][2] == (1500.0, 1700.0, 1801.0)   # the L's centre is out of CE's noise
    assert [(d.waveform, d.f_hz) for d in dets] == [(L, 1509.5), (CE, 1525.0)]
    # L alone: a CE candidate is not gated
    calls.clear()
    dets = RX.detect(None, L_SHORT, cands=cands[:2])
    assert [c[:2] for c in calls] == [(1509.5, "l-short")]
    assert [(d.waveform, d.f_hz) for d in dets] == [(L, 1509.5)]


def _fake_pass(spec, d, z_ref, f_hz=None, suspect=False):
    f = d.f_hz if f_hz is None else f_hz
    tau0 = 0.0 if d.timing is None else d.timing.tau0
    return SimpleNamespace(waveform=spec.waveform, frame=spec.name, f_hz=f,
                           timing=SimpleNamespace(tau0=tau0),
                           report=SimpleNamespace(z_ref=z_ref, suspect=suspect))


def test_joint_receive_slot_takes_each_frame_and_merges_by_centre(monkeypatch):
    calls, z = [], {1509.5: 30.0, 1502.0: 40.0, 1800.0: 12.0, 1525.0: 2.0}
    monkeypatch.setattr(RX, "prepare", lambda cap: cap)

    def receive_pass(prep, spec, d, **kw):
        calls.append((d.f_hz, spec.name, kw["exclude_hz"]))
        return _fake_pass(spec, d, z[d.f_hz], suspect=d.f_hz == 1525.0)

    monkeypatch.setattr(RX, "receive_pass", receive_pass)
    dets = [_det(1509.5, L, z_ref=30.0), _det(1502.0, CE, z_ref=40.0),
            _det(1800.0, CE, z_ref=12.0)]
    ps = RX.receive_slot(None, (SHORT, L_SHORT), dets=dets)
    assert calls == [(1509.5, "l-short", (1502.0, 1800.0)),
                     (1502.0, "short", (1500.0, 1800.0)),
                     (1800.0, "short", (1500.0, 1502.0))]
    assert [(p.waveform, p.f_hz) for p in ps] == [(CE, 1502.0), (CE, 1800.0)]
    # a waveform not in the selection is received with the selection's twin
    calls.clear()
    RX.receive_slot(None, L_TINY, dets=[_det(1800.0, CE, z_ref=12.0)])
    assert calls == [(1800.0, "tiny", ())]
    # the suspect-neighbour rule holds across waveforms, by centre: a weak
    # SUSPECT CE 12 Hz from the L's centre goes (21.5 Hz from its pilot), one
    # 25 Hz from it stays (15.5 from the pilot); within CE as before, a weak
    # SUSPECT 10 Hz from a sound CE goes when its timing makes it no ghost
    calls.clear()
    z.update({1488.0: 2.0, 1810.0: 2.0, 1800.0: 30.0})
    sus = {1525.0, 1488.0, 1810.0}
    monkeypatch.setattr(RX, "receive_pass", lambda prep, spec, d, **kw: _fake_pass(
        spec, d, z[d.f_hz], suspect=d.f_hz in sus))
    ps = RX.receive_slot(None, (SHORT, L_SHORT),
                         dets=[_det(1509.5, L, z_ref=30.0), _det(1525.0, CE, z_ref=2.0),
                               _det(1488.0, CE, z_ref=2.0), _det(1800.0, CE, z_ref=30.0),
                               _det(1810.0, CE, z_ref=2.0, tau0=100.0)])
    assert [(p.waveform, p.f_hz) for p in ps] == [(L, 1509.5), (CE, 1525.0), (CE, 1800.0)]
    # after tracking, a ghost within a waveform at its own symbol: an L 10 Hz
    # from a stronger L with tau0 30 CH samples off (3 T_L is 45.5, CE's 3 T
    # 22.7) is that L; a CE that far from it is another signal
    z.update({1719.5: 8.0, 1720.0: 8.0, 1709.5: 30.0})
    ps = RX.receive_slot(None, (SHORT, L_SHORT),
                         dets=[_det(1709.5, L, z_ref=30.0), _det(1719.5, L, z_ref=8.0, tau0=30.0),
                               _det(1720.0, CE, z_ref=8.0, tau0=30.0)])
    assert [(p.waveform, p.f_hz) for p in ps] == [(L, 1709.5), (CE, 1720.0)]


def test_a_ce_selection_is_the_ce_receiver(monkeypatch):
    """D1: CE alone, bare or in a tuple, is the original receiver: CE's own
    acquisition (a bare frame, the hints as frequencies, pairs of another
    waveform left out), and no residual re-check at the gate; anything
    holding L gets the tuple, the hints as given, and the re-check."""
    got = []

    def acq(cap, spec, live_only=False, preprocessed=False, hints=()):
        got.append((spec, list(hints)))
        return SimpleNamespace(detections=lambda s: [_det(1500.0, CE, "preamble")])

    monkeypatch.setattr(RX.acquire, "acquire", acq)
    prep = SimpleNamespace(fe=np.zeros(8, np.complex64), q=Q_TEST, t0_index=0.0, fs=4000,
                           keep=None)
    hints = [(1500.0, CE), (1600.0, L), 1700.0, (1800.0, CE)]
    for sel in (SHORT, (SHORT,), [SHORT]):
        RX.candidates(prep, sel, hints=hints)
        assert got[-1][0] is SHORT and got[-1][1] == [1500.0, 1700.0, 1800.0]
    RX.candidates(prep, SHORT, hints=[1500.0, 1600.0])
    assert got[-1] == (SHORT, [1500.0, 1600.0])
    RX.candidates(prep, (L_SHORT, SHORT), hints=hints)
    assert got[-1] == ((SHORT, L_SHORT), hints)
    RX.candidates(prep, L_SHORT, hints=hints)
    assert got[-1] == ((L_SHORT,), hints)
    # gate V: an accept the residual ratio would refuse (an L signal under CE's gate)
    monkeypatch.setattr(RX, "channel_for", lambda *a, **k: None)
    monkeypatch.setattr(track, "verify", lambda chan, fs, spec, d, erase_s=(): _det(
        d.f_hz, CE, "preamble", 6.41))
    monkeypatch.setattr(track, "verify_track", lambda *a, **k: object())
    monkeypatch.setattr(RX, "_ce_resid_ratio", lambda tr: 4.26)
    cands = [_det(1500.0, CE, "preamble")]
    for sel in (SHORT, (SHORT,)):
        assert [d.z_ref for d in RX.detect(prep, sel, cands=cands)] == [6.41]
    assert RX.detect(prep, (SHORT, L_SHORT), cands=cands) == []
    # receive_slot: CE's detect with the bare frame, every detection as SHORT
    seen = []
    monkeypatch.setattr(RX, "prepare", lambda cap: cap)
    monkeypatch.setattr(RX, "detect", lambda prep, spec, *a, **k: seen.append(spec) or [])
    RX.receive_slot(prep, (SHORT,))
    RX.receive_slot(prep, L_SHORT)
    assert seen[0] is SHORT and seen[1] == (L_SHORT,)
    seen.clear()
    monkeypatch.setattr(RX, "receive_pass", lambda prep, spec, d, **kw: seen.append(
        (spec, kw["exclude_hz"])) or _fake_pass(spec, d, d.z_ref))
    ps = RX.receive_slot(prep, (SHORT,), dets=[_det(1500.0, CE, z_ref=20.0),
                                                _det(1509.0, CE, z_ref=10.0, tau0=100.0)])
    assert seen == [(SHORT, (1509.0,)), (SHORT, (1500.0,))]
    assert [p.f_hz for p in ps] == [1500.0, 1509.0]


def test_joint_aliases_and_ghosts_at_each_waveform_symbol(monkeypatch):
    """After the gate, aliases and ghosts are dropped per waveform with
    that waveform's symbol: L's 2/T_L alias (32.99 Hz, CE's 1/T) within
    2.5 T_L, and its ghost window 3 T_L (45.5 CH samples, CE's 22.7)."""
    res = {  # candidate f -> gate V's (Z_ref, tau0 in CH samples)
        1509.5: (40.0, 0.0), 1542.49: (10.0, 20.0),       # L and its alias
        1709.5: (40.0, 0.0), 1724.5: (8.0, 30.0),          # L and its ghost
        1300.0: (40.0, 0.0), 1315.0: (8.0, 30.0),          # CE and no ghost of it
        1324.0: (40.0, 0.0), 1357.0: (10.0, 20.0),         # L and no alias at CE's 1/T
    }
    wf = {1509.5: L, 1542.49: L, 1709.5: L, 1724.5: L, 1300.0: CE, 1315.0: CE,
          1324.0: CE, 1357.0: CE}

    def gate(prep, spec, d, others, erase_s):
        zr, tau = res[d.f_hz]
        return _det(d.f_hz, d.waveform, d.method, zr, tau)

    monkeypatch.setattr(RX, "_gate", gate)
    cands = [_det(f, w) for f, w in wf.items()]
    dets = RX.detect(None, (SHORT, L_SHORT), cands=cands)
    assert sorted((d.waveform, d.f_hz) for d in dets) == [
        (CE, 1300.0), (CE, 1315.0), (CE, 1324.0), (CE, 1357.0), (L, 1509.5), (L, 1709.5)]


def test_the_joint_gate_measures_a_lead_in_at_its_own_symbol(monkeypatch):
    seen = []
    monkeypatch.setattr(RX, "channel_for", lambda *a, **k: None)
    monkeypatch.setattr(ltrack, "verify_l", lambda chan, fs, spec, d, erase_s=(): _det(
        d.f_hz, L, d.method, 9.0, 250.0))

    def span(fe, t0_index, f_hz, drift, tau0_s, fs, *, t_sym):
        seen.append((f_hz, tau0_s, t_sym))
        return 4.0

    monkeypatch.setattr(RX.acquire, "lead_in_span", span)
    prep = SimpleNamespace(fe=None, t0_index=0.0, fs=4000)
    assert RX._gate(prep, L_SHORT, _det(1509.5, L), (), ()).lead_in_s == 4.0
    assert seen == [(1509.5, 1.0, L_SHORT.t_sym)]
    assert RX._gate(prep, L_SHORT, _det(1509.5, L, "preamble"), (), ()).lead_in_s == 0.0
    assert len(seen) == 1                                    # a preamble hit has its own


def test_a_ce_accept_is_rechecked_for_an_l_signal(monkeypatch):
    """D3: in the joint receiver a CE gate-V accept whose known symbols fit
    R > CE_RESID_MIN times worse than they should is rejected unless its
    Z_ref still clears Z_ACCEPT after dividing by sqrt(R) (an L signal
    under CE's gate read Z_ref 6.41 at R 4.26; CE itself reads 1.02-1.35)."""
    case = {}
    monkeypatch.setattr(RX, "channel_for", lambda *a, **k: None)
    monkeypatch.setattr(track, "verify",
                        lambda chan, fs, spec, d, erase_s=(): _det(d.f_hz, CE, "preamble",
                                                                    case["z"]))
    monkeypatch.setattr(track, "verify_track", lambda *a, **k: object())
    monkeypatch.setattr(RX, "_ce_resid_ratio", lambda tr: case["R"])
    for z, R, kept in ((6.41, 4.26, False), (20.0, 4.26, True), (6.41, RX.CE_RESID_MIN, True),
                       (6.41, 1.5, True), (12.0, 4.0, False), (12.1, 4.0, True)):
        case.update(z=z, R=R)
        v = RX._gate(None, SHORT, _det(1500.0, CE), (), ())
        assert (v is not None) == kept, (z, R)
    # an L accept is not re-checked
    monkeypatch.setattr(ltrack, "verify_l", lambda chan, fs, spec, d, erase_s=(): _det(
        d.f_hz, L, "preamble", 6.5))
    monkeypatch.setattr(RX, "_ce_resid_ratio", lambda tr: pytest.fail("L re-checked"))
    assert RX._gate(None, L_SHORT, _det(1509.5, L), (), ()) is not None


@pytest.mark.slow
def test_the_residual_recheck_on_real_signals():
    """D3 on real gate-V tracks of TINY frames from blind candidates at
    1500 Hz: at -6 dB CE's gate alone accepts an l-tiny signal (Z_ref 9.2,
    seed 1) and the residual ratio (11.3) refuses it, while L's gate takes
    it; a CE signal reads 1.5 and its accept stands, Z_ref unchanged; and
    weak CE passes on fading paths keep theirs."""
    from test_qrss_receiver import scenario

    for wave in (L, CE):
        _, sim = l_scenario("l-tiny", -6.0, 1) if wave == L else scenario("tiny", -6.0, 1)
        prep = RX.prepare(_cap(sim))
        chan = RX.channel_for(prep, TINY, _det(1500.0, CE), ())
        v = track.verify(chan, CH_FS, TINY, _det(1500.0, CE))
        assert v is not None and v.z_ref > 6.0
        R = RX._ce_resid_ratio(track.verify_track(chan, TINY, v, fit=False))
        g = RX._gate(prep, TINY, _det(1500.0, CE), (), ())
        if wave == L:
            assert R > 5.0 and g is None
            assert RX._gate(prep, L_TINY, _det(1500.0, L), (), ()).waveform == L
        else:
            assert R < RX.CE_RESID_MIN and g is not None and g.z_ref == v.z_ref
    # weak CE on fading paths keeps its accept: R 1.30 and 1.19 (Z_ref 6.57
    # and 9.84), where leaving u's error P out of the model would read 2.43
    # and 3.22 and refuse both
    for snr, preset, seed in ((-18.0, "mps", 1), (-12.0, "disturbed", 3)):
        _, sim = scenario("tiny", snr, seed, cfg=(("preset", preset),))
        prep = RX.prepare(_cap(sim))
        g = RX._gate(prep, TINY, _det(1500.0, CE), (), ())
        assert g is not None and g.waveform == CE, preset


# --- confidence over time -----------------------------------------------------------------------

def test_l_confidence_over_time():
    p = make_pass_result(unit_rms_latents(L_SHORT.n_data, 1), 2.0, f_hz=1509.5, waveform=L)
    assert p.frame == "l-short"
    c = RX.confidence_db(p)
    assert len(c) == RX.CONF_BINS
    assert RX.confidence_db(p, L_SHORT) == c == RX.confidence_db(p, SHORT)   # its own frame
    assert c[0] is None                                          # the preamble
    vals = [v for v in c if v is not None]
    assert len(vals) > 0.8 * len(c)
    assert max(vals) == pytest.approx(10 * np.log10(2.0), abs=0.06)     # data alone: W
    end = L_SHORT.keyed_end_pos * L_SHORT.t_sym
    half = RX.confidence_db(p, elapsed_s=end / 2)
    assert all(v is None for v in half[RX.CONF_BINS // 2 + 1:])
    assert half[:RX.CONF_BINS // 2] == c[:RX.CONF_BINS // 2]
    # data symbol k carries latents 2k, 2k+1: erasing the last half of the
    # latents empties the last half of the data symbols' bins
    w = np.asarray(p.w).copy()
    w[L_SHORT.n_data // 2:] = 0.0
    p2 = dataclasses.replace(p, w=w)
    lay = lframe.layout(L_SHORT)
    t_mid = lay.pos[lay.data[L_SHORT.n_data // 4]] * L_SHORT.t_sym
    c2 = RX.confidence_db(p2)
    b = int(t_mid / end * RX.CONF_BINS) + 2
    assert all(v is None or v <= RX.CONF_FLOOR_DB + 1e-9 for v in c2[b:])


# --- the store, EM and the command lines ---------------------------------------------------------

def _air(sp, g, n):
    return picture.air_values(sp.segs[g], g)[:n]


def _l_and_ce_store(tmp_path, mode=0):
    """A store with one picture heard once as L (segment 0, l-short) and
    once as CE (short), both by header."""
    st = Store(tmp_path / "qrss")
    sp = synthetic_picture(31, mode=mode)
    h = dataclasses.replace(L_HDR, picture_id=sp.picture_id, mode=sp.mode,
                            codec_id=sp.codec_id)
    pl = make_pass_result(_air(sp, 0, L_SHORT.n_data), 2.0, seed=1, q=Q_TEST, f_hz=1509.5,
                          hdr=h, decoded=True, waveform=L)
    pl = dataclasses.replace(pl, ch=np.ones(64, np.complex64),
                             cw=CwIdResult(text=L_CALL, z_match=40.0, keying="ook",
                                           agrees=True))
    pc = make_pass_result(_air(sp, 0, SHORT.n_data), 2.0, seed=2, q=Q_TEST - 4, f_hz=1500.0,
                          hdr=h, decoded=True)
    key = (L_CALL, sp.picture_id)
    assert asc.associate(pl, st) == (key, "header")
    assert asc.associate(pc, st) == (key, "header")
    return st, key, pl, pc


def test_the_store_knows_each_pass_waveform_and_frame(tmp_path):
    st, _, pl, pc = _l_and_ce_store(tmp_path, mode=1)
    il, ic = st.pass_info(pl.uid), st.pass_info(pc.uid)
    assert (il["waveform"], il["frame"], il["f_hz"]) == (L, "l-short", 1509.5)
    assert (ic["waveform"], ic["frame"]) == (CE, "short")
    due = st.due(Q_TEST + 2)
    assert [(d["segment"], d["waveform"], d["frame"], d["f_hz"]) for d in due] == [
        (1, L, "l-short", 1509.5)]
    # a pass written before waveforms had names reads as CE
    path = st._pass_path(pc.uid)
    with np.load(path, allow_pickle=False) as d:
        arrays = {k: d[k] for k in d.files}
    meta = json.loads(arrays.pop("meta").tobytes().decode())
    del meta["waveform"], meta["frame"]
    with open(path, "wb") as f:
        np.savez(f, meta=np.frombuffer(json.dumps(meta).encode(), np.uint8), **arrays)
    old = Store(tmp_path / "qrss").pass_info(pc.uid)
    assert old["waveform"] == CE and old["frame"] is None


def test_one_transmission_is_one_member_whichever_waveform(tmp_path):
    """`Store.twin` with a waveform compares a member of the other waveform
    by band centre: the L member (pilot 1509.5 Hz, centre 1500) is the CE
    reading at 1500 Hz of that slot, not the one at 1509.5."""
    st, key, pl, pc = _l_and_ce_store(tmp_path)
    acc = st.accumulator(key)
    uid = lambda m: None if m is None else m["uid"]
    assert uid(st.twin(acc, Q_TEST, 1500.0, 0, waveform=CE)) == pl.uid
    assert uid(st.twin(acc, Q_TEST, 1509.5, 0, waveform=CE)) is None
    assert uid(st.twin(acc, Q_TEST, 1509.5, 0)) == pl.uid          # no waveform: as before
    assert uid(st.twin(acc, Q_TEST - 4, 1509.5, 0, waveform=L)) == pc.uid
    assert uid(st.twin(acc, Q_TEST - 4, 1509.5, 1, waveform=L)) is None
    # a weaker CE reading of the L member's transmission is left out
    sp_h = st.load_pass(pl.uid).header
    again = make_pass_result(np.asarray(pc.z, np.float64), 1.0, seed=9, q=Q_TEST, f_hz=1500.5,
                             hdr=sp_h, decoded=True)
    assert asc.associate(again, st) == (key, "")
    assert st.accumulator(key).uids == [pl.uid, pc.uid]


def test_headerless_passes_are_filed_beside_each_other(tmp_path):
    """Rule 2 sums header LLRs, and a headerless frame (TINY, l-tiny) keeps
    none: a second such pass in the store is filed provisional, not
    refused over an empty sum, and a pass with a header beside them still
    soft-combines (found 2026-10-12: the joint listener's two tiny tiles
    stored one pass and "store failed" the other)."""
    st = Store(tmp_path / "qrss")
    a = unit_rms_latents(TINY.n_data, 3)
    for k, (f, w) in enumerate([(1500.0, CE), (1580.0, L), (1700.0, CE)]):
        p = make_pass_result(a, 0.01, seed=k, f_hz=f, waveform=w)
        p = dataclasses.replace(p, hdr_llr=np.zeros(0, np.float32))
        assert asc.associate(p, st) == (None, "prov")
        assert st.pass_info(p.uid)["frame"] == ("l-tiny" if w == L else "tiny")
    sp = synthetic_picture(5)
    h = dataclasses.replace(L_HDR, picture_id=sp.picture_id, mode=sp.mode, codec_id=sp.codec_id)
    one = [make_pass_result(_air(sp, 0, SHORT.n_data), 0.5, seed=10 + k, f_hz=1500.0, hdr=h,
                            hdr_es_n0_db=-14.0) for k in range(2)]
    assert all(header.decode(np.asarray(p.hdr_llr, np.float64)) is None for p in one)
    assert asc.associate(one[0], st) == (None, "prov")
    assert asc.associate(one[1], st) == ((L_CALL, sp.picture_id), "soft-header")


def test_em_leaves_l_members_as_received(tmp_path):
    st, key, pl, pc = _l_and_ce_store(tmp_path)
    p = st.load_pass(pl.uid)
    assert em.pass_spec(p) is L_SHORT and em.is_l(p)
    v1 = dataclasses.replace(p, header=dataclasses.replace(p.header, version=1))
    assert em.pass_spec(v1) is L_SHORT                           # L has no version 1
    assert em.pass_spec(st.load_pass(pc.uid)) is SHORT
    log = []
    hist = em.em_refine(st, key, rounds=2, log=log.append)
    assert len(hist) == 2 and hist[0] == hist[1]                 # nothing moved: one round
    assert log == [f"pass {pl.uid}: waveform L, not re-received (EM is CE-only)"]
    assert em.rereceive(p, None) is None
    assert em.rereceive(p, None, return_path=True) == (None, None)
    for call in (lambda: em.pass_path(p),
                 lambda: em.template_search(st, key, None, L_SHORT),
                 lambda: em.receive_template(st, key, None, L_SHORT),
                 lambda: em.retro_detect(st, key, None, L_SHORT)):
        with pytest.raises(ValueError, match="CE-only"):
            call()
    # the callsign keying: each waveform's members vote alone, and L is never OOK
    units, ook = em.known_keying(st, st.accumulator(key), L_CALL, L)
    assert np.array_equal(units, keying_units(L_CALL)) and ook is False
    assert em.known_keying(st, st.accumulator(key), L_CALL) is None
    assert em._candidates(st, st.accumulator(key)) == [(1500.0, 0.0, 0.0)]


def test_retro_takes_an_l_pass_of_the_slot_by_its_centre(tmp_path, monkeypatch):
    """A retro find (CE) at an L pass's band centre in its slot is that pass."""
    st, key, _, _ = _l_and_ce_store(tmp_path)
    q = Q_TEST + 8
    st.add_pass(make_pass_result(unit_rms_latents(L_SHORT.n_data, 5), 1.0, seed=5, q=q,
                                 f_hz=1509.5, waveform=L))
    monkeypatch.setattr(em, "template_search", lambda *a, **k: (
        _det(1500.4, CE, "template", 9.0), 9.0, {"segment": 0}))
    monkeypatch.setattr(em, "receive_template", lambda *a, **k: pytest.fail("received again"))
    log = []
    pb = SimpleNamespace(capture=lambda q, dur: (object(), 1.0))
    assert em.retro_detect(st, key, pb, SHORT, slots=[q], log=log.append) == []
    assert log == [f"slot q={q}: Z 9.0 at 1500.400 Hz, already stored"]


def test_qrss_decode_names_each_member_waveform(tmp_path, monkeypatch):
    st, key, pl, pc = _l_and_ce_store(tmp_path)
    qd = _script("qrss_decode")
    lines = qd.acc_lines(st.accumulator(key), st)
    assert any(pl.uid in s and "L l-short" in s for s in lines)
    assert any(pc.uid in s and "CE short" in s for s in lines)
    assert all("CE" not in s and " L " not in s for s in qd.acc_lines(st.accumulator(key))[1:]
               if "segment " in s and "effective" in s)
    # --retro searches for CE passes only
    (st.root / "passband").mkdir()
    monkeypatch.setattr(sys, "argv", ["qrss_decode.py", str(tmp_path / "x.png"), "--key",
                                      f"{key[0]}:{key[1]:08x}", "--store", str(st.root),
                                      "--retro", "--frame", "l-short"])
    with pytest.raises(SystemExit, match="CE passes only"):
        qd.main()


def test_qrss_receive_listens_for_both_unless_told(monkeypatch):
    qr = _script("qrss_receive")
    sel = lambda *argv, npz=None: qr.select_frames(qr.parse_args(["x.npz", *argv]), npz)
    assert sel() == (FULL, L_FULL)
    assert sel("--frame", "short") == (SHORT, L_SHORT)
    assert sel("--frame", "l-tiny") == (TINY, L_TINY)                # either name
    assert sel("--just-ce", "--frame", "short") is SHORT              # CE's own path
    assert sel("--just-l", "--frame", "short") is L_SHORT
    assert sel("--just-l", "--frame", "l-short") is L_SHORT
    assert sel(npz="l-tiny") == (TINY, L_TINY)                      # the recording's shape
    assert sel("--frame", "short", npz="l-tiny") == (SHORT, L_SHORT)
    assert sel("--just-ce", npz="full-v1").name == "full-v1"
    for argv, npz, msg in ((("--just-ce",), "l-short", "--just-ce"),
                           (("--just-l",), "short", "--just-l"),
                           ((), "full-v1", "no L twin; add --just-ce")):
        with pytest.raises(SystemExit, match=msg):
            sel(*argv, npz=npz)
    with pytest.raises(SystemExit):
        qr.parse_args(["x.npz", "--just-ce", "--just-l"])
    p = make_pass_result(unit_rms_latents(L_SHORT.n_data, 2), 1.0, f_hz=1509.5, waveform=L)
    lines = qr.pass_lines(p)
    assert lines[0].endswith("L l-short") and lines[1].split()[:3] == ["pilot", "1509.500", "Hz"]


def test_qrss_receive_hands_the_selection_to_the_receiver(tmp_path, monkeypatch, capsys):
    qr = _script("qrss_receive")
    path = tmp_path / "rx.npz"
    chm.save_npz(path, l_scenario("l-tiny", -12.0, 1)[1])
    seen = []
    monkeypatch.setattr(RX, "receive_slot", lambda cap, spec, estimator="joint": seen.append(
        spec) or [])
    for argv, sel, names in (((), (TINY, L_TINY), "CE or L"), (("--just-l",), L_TINY, "L"),
                             (("--frame", "short"), (SHORT, L_SHORT), "CE or L")):
        monkeypatch.setattr(sys, "argv", ["qrss_receive.py", str(path), "--no-store", *argv])
        with pytest.raises(SystemExit) as e:
            qr.main()
        assert e.value.code == 1 and seen[-1] == sel, argv
        assert f"no {names} signal found" in capsys.readouterr().out


# --- the real receiver -------------------------------------------------------------------------

def test_a_blind_l_tiny_pass_at_minus_12():
    """l-tiny at -12 dB, both waveforms listened for: one L pass at the
    pilot, latent SNR 6.46 dB measured (genie 6.73), no CE pass."""
    a, sim = l_scenario("l-tiny", -12.0, 1)
    t = time.perf_counter()
    ps = RX.receive_slot(_cap(sim), (TINY, L_TINY))
    dt = time.perf_counter() - t
    assert len(ps) == 1
    p = ps[0]
    assert (p.waveform, p.frame, p.header, p.cw) == (L, "l-tiny", None, None)
    # every field a sidecar or the store writes, as plain JSON
    for k in ("uid", "q", "frame", "waveform", "f_hz", "ch_fs", "ch_t0_index", "estimator", "em_round"):
        json.dumps(getattr(p, k))
    json.dumps(dataclasses.asdict(p.report))
    json.dumps({k: getattr(p.timing, k) for k in ("tau0", "ppm", "gamma", "z")})
    assert abs(p.f_hz - 1500.0) < 0.05
    snr = latent_snr_db(p.z, a)
    print(f"l-tiny -12 dB: latent {snr:.2f} dB, W {RX.mean_w_db(p):+.2f} dB, "
          f"Z_ref {p.report.z_ref:.1f}, {dt:.1f} s")
    assert snr > 6.0
    assert abs(RX.mean_w_db(p) - snr) < 0.5
    c = RX.confidence_db(p)
    assert c[0] is None and sum(v is not None for v in c) > len(c) // 2
    # the pass round-trips through a detection of its own waveform, and the
    # report quotes SNR2500 on L's keyed power (`ltrack.report_l`)
    d = RX.detection_from_pass(p)
    assert d.waveform == L
    p2, trs = RX.receive_pass(_cap(sim), L_TINY, d, return_tracks=True)
    assert p2.report == ltrack.report_l(trs[-1], p2.report.kappa, p2.report.suspect)
    assert p2.report != track.report(trs[-1], p2.report.kappa, p2.report.suspect)


@pytest.mark.slow
@pytest.mark.parametrize("snr, floor, call", [(-6.0, 9.7, L_CALL), (-24.0, -4.2, None)],
                         ids=["-6", "-24"])
def test_l_short(snr, floor, call):
    """l-short through the joint receiver: header, the callsign windows at
    -6 dB, latent SNR 10.16 / -3.63 dB measured (genie 10.06 / -3.62)."""
    a, sim = l_scenario("l-short", snr, 1)
    ps = RX.receive_slot(_cap(sim), (SHORT, L_SHORT))
    assert [(p.waveform, p.frame) for p in ps] == [(L, "l-short")]
    p = ps[0]
    s = latent_snr_db(p.z, a)
    print(f"l-short {snr} dB: latent {s:.2f} dB, W {RX.mean_w_db(p):+.2f} dB, "
          f"Z_ref {p.report.z_ref:.1f}, call {p.cw.text if p.cw else None}")
    assert p.header == L_HDR
    assert s > floor
    if call is not None:
        assert p.cw is not None and p.cw.text == call and p.cw.agrees
        assert p.cw.keying == "fsk"
        # round B tracks with the header's callsign keying known, FSK
        _, trs = RX.receive_pass(_cap(sim), L_SHORT, RX.detection_from_pass(p),
                                 return_tracks=True)
        assert len(trs) == 2 and trs[0].keying is None and trs[1].ook is False
        assert np.array_equal(trs[1].keying, keying_units(call))


@pytest.mark.slow
def test_noise_holds_neither_waveform():
    rng = np.random.default_rng(11)
    n = round((12 + L_SHORT.keyed_end_pos * L_SHORT.t_sym + 3) * 4000)
    fe = ((rng.standard_normal(n) + 1j * rng.standard_normal(n)) / np.sqrt(2)).astype(
        np.complex64)
    assert RX.receive_slot(frontend.Capture(fe, Q_TEST, 12 * 4000.0), (SHORT, L_SHORT)) == []


@pytest.mark.slow
def test_l_beside_ce():
    """An L pass at -12 dB with a CE signal of the same power 60 Hz above:
    both received, each as its own waveform (L 6.98 dB latent measured)."""
    cfg = (("neighbours", (Neighbour(60.0, 0.0, seed=7, waveform=CE),)),)
    a, sim = l_scenario("l-short", -12.0, 1, cfg=cfg)
    ps = RX.receive_slot(_cap(sim), (SHORT, L_SHORT))
    got = sorted((p.waveform, round(p.f_hz)) for p in ps)
    assert got == [(CE, 1560), (L, 1500)]
    pl = next(p for p in ps if p.waveform == L)
    assert pl.header == L_HDR and latent_snr_db(pl.z, a) > 6.4


@pytest.mark.slow
def test_ce_beside_l():
    """A CE pass at -12 dB with an L signal 60 Hz above: both received (CE
    4.57 dB latent measured, as CE alone)."""
    from test_qrss_receiver import HDR, scenario

    cfg = (("neighbours", (Neighbour(60.0, 0.0, seed=7, waveform=L),)),)
    a, sim = scenario("short", -12.0, 1, cfg=cfg)
    ps = RX.receive_slot(_cap(sim), (SHORT, L_SHORT))
    got = sorted((p.waveform, round(p.f_hz)) for p in ps)
    assert got == [(CE, 1500), (L, 1560)]
    pc = next(p for p in ps if p.waveform == CE)
    assert pc.header == HDR and latent_snr_db(pc.z, a) > 4.0


@pytest.mark.slow
def test_ce_gate_on_a_real_l_signal_is_rejected():
    """l-short at -12 dB under CE's gate: track.verify accepts it (Z_ref
    6.41), the residual ratio (4.26) rejects it."""
    _, sim = l_scenario("l-short", -12.0, 1)
    prep = RX.prepare(_cap(sim))
    c = next(c for c in RX.candidates(prep, SHORT) if abs(c.f_hz - 1500.0) < 1.0)
    v = track.verify(RX.channel_for(prep, SHORT, c, ()), CH_FS, SHORT, c)
    assert v is not None and v.z_ref < 10.0
    tr = track.verify_track(RX.channel_for(prep, SHORT, v, ()), SHORT, v, fit=False)
    assert RX._ce_resid_ratio(tr) > 3.0
    assert RX._gate(prep, SHORT, c, (), ()) is None


@pytest.mark.slow
def test_qrss_receive_on_an_l_recording(tmp_path):
    _, sim = l_scenario("l-tiny", -12.0, 1)
    path = tmp_path / "rx.npz"
    chm.save_npz(path, sim)
    run = lambda *argv: subprocess.run([sys.executable, str(REPO_ROOT / "qrss_receive.py"),
                                        str(path), "--no-store", *argv],
                                       capture_output=True, text=True, timeout=300,
                                       cwd=REPO_ROOT, check=False)
    out = run()
    assert out.returncode == 0, out.stderr
    assert "L l-tiny" in out.stdout and "pilot" in out.stdout
    out = run("--just-ce")
    assert out.returncode != 0 and "--just-ce does not listen" in out.stderr

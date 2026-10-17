"""The receiver rules a command line can set back (`sstvae.qrss.rules`),
and the two rules of 2026-10-12 they restore.

- The switches: set and cleared through the environment, an unknown one
  refused, a spawned process (the listener's end-of-slot receive)
  following its parent, and the flags of qrss_listen.py and
  qrss_receive.py setting them (qrss_decode.py: --old-edge-noise only).
- EDGE_NOISE (`frontend.noise_psd` with audio_hz): on white audio through
  the real front end the floor near 300 and 2700 Hz reads within 0.75 dB
  of the true one, where the old rule read it 6-12 dB low, also with a
  tone excluded on the flat side; with a wide neighbour there, halfway
  in dB between the flat bins' reading and the compensated skirt's; a
  channel whose bins are all flat (about 430-2570 Hz) is not changed at
  all; both callers (`track.ChanCapture`, `acquire.lead_in_span`) pass
  the channel's audio frequency.
- GATING (`receiver._owned`): a whole-slot line inside a preamble-found
  signal's own band and no stronger in A3 than that signal's own line
  is not gated, in the CE receiver and in the joint one (stub gates),
  nor counted against max_candidates; a preamble hit, a hint, a
  stronger line and a line beside a signal with no A3 line of its own
  are, and a signal found only by the whole-slot search or a hint owns
  nothing.
"""

from __future__ import annotations

import importlib.util
import math
import multiprocessing
import os
import sys
from fractions import Fraction
from types import SimpleNamespace

import numpy as np
import pytest

from qrss_helpers import REPO_ROOT
from sstvae.qrss import acquire, frame, frontend, lframe, rules, track
from sstvae.qrss import receiver as RX
from sstvae.qrss.constants import CH_FS, FE_CENTER_HZ, FE_FS, WAVEFORM_CE, WAVEFORM_L
from sstvae.qrss.types import Detection, FreqPath, Timing

CE, L = WAVEFORM_CE, WAVEFORM_L


@pytest.fixture(autouse=True)
def _current_rules():
    """Every test starts on the current rules and leaves the environment as
    it was. (Not monkeypatch.delenv: on an unset variable it records
    nothing, so `rules.set_old` writing os.environ afterwards would leak
    into every later module.)"""
    saved = {r: os.environ.pop(r, None) for r in rules.RULES}
    yield
    for r, v in saved.items():
        if v is None:
            os.environ.pop(r, None)
        else:
            os.environ[r] = v


def _script(name):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --- the switches -------------------------------------------------------------------------------

def test_a_rule_is_set_back_through_the_environment(monkeypatch):
    for r in rules.RULES:
        assert not rules.old(r)
        rules.set_old(r)
        assert rules.old(r)
        rules.set_old(r, False)
        assert not rules.old(r)
    monkeypatch.setenv(rules.GATING, "0")
    assert not rules.old(rules.GATING)
    monkeypatch.setenv(rules.GATING, "yes")
    assert rules.old(rules.GATING)
    with pytest.raises(ValueError, match="unknown rule"):
        rules.set_old("QRSS_OLD_SOMETHING")


def test_a_spawned_process_follows_its_parent():
    """The listener's end-of-slot receive runs in a spawned process."""
    rules.set_old(rules.GATING)
    with multiprocessing.get_context("spawn").Pool(1) as pool:
        assert pool.apply(rules.old, (rules.GATING,))
        assert not pool.apply(rules.old, (rules.EDGE_NOISE,))


def test_the_command_lines_set_the_rules_back(tmp_path, monkeypatch):
    import argparse

    ap = argparse.ArgumentParser()
    rules.add_arguments(ap)
    rules.apply(ap.parse_args([]))
    assert not any(rules.old(r) for r in rules.RULES)
    rules.apply(ap.parse_args(["--old-edge-noise"]))
    assert rules.old(rules.EDGE_NOISE) and not rules.old(rules.GATING)
    # qrss_listen.py: parsing alone sets nothing (build_config is side-effect
    # free); main sets them before it starts anything
    ql = _script("qrss_listen")
    monkeypatch.delenv(rules.EDGE_NOISE)
    args, _ = ql.build_config(["--old-gating", "--old-edge-noise", "--state", str(tmp_path)])
    assert args.old_gating and args.old_edge_noise and not rules.old(rules.GATING)

    class Started(Exception):
        pass

    def lock(path):
        assert rules.old(rules.GATING) and rules.old(rules.EDGE_NOISE)
        raise Started

    monkeypatch.setattr("sstvae.qrss.live.dir_lock", lock)
    with pytest.raises(Started):
        ql.main(["--old-gating", "--old-edge-noise", "--state", str(tmp_path)])
    # qrss_receive.py: set before the receive
    for r in rules.RULES:
        monkeypatch.delenv(r)
    qr = _script("qrss_receive")
    seen = []
    monkeypatch.setattr(RX, "receive_slot", lambda cap, spec, estimator="joint": seen.append(
        [rules.old(r) for r in rules.RULES]) or [])
    monkeypatch.setattr(qr, "load_capture", lambda path, q, start: (
        frontend.Capture(fe=np.zeros(4, np.complex64), q=0, t0_index=0.0), "tiny"))
    monkeypatch.setattr(sys, "argv", ["qrss_receive.py", "x.npz", "--no-store", "--old-gating"])
    with pytest.raises(SystemExit):
        qr.main()
    assert seen == [[False, True]]
    # qrss_decode.py: the edge rule only (--em and --retro gate no candidates)
    monkeypatch.delenv(rules.GATING)
    qd = _script("qrss_decode")
    monkeypatch.setattr(sys, "argv", ["qrss_decode.py", "--list", "--store", str(tmp_path),
                                      "--old-edge-noise"])
    qd.main()
    assert rules.old(rules.EDGE_NOISE) and not rules.old(rules.GATING)
    monkeypatch.setattr(sys, "argv", ["qrss_decode.py", "--list", "--store", str(tmp_path),
                                      "--old-gating"])
    with pytest.raises(SystemExit):
        qd.main()


# --- EDGE_NOISE ---------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def white_fe():
    """60 s of white audio of variance 1 through the real front end: its
    floor is 1/FE_FS per Hz wherever the front end is flat."""
    return frontend.audio_to_fe(np.random.default_rng(0).standard_normal(8000 * 60))


def _floor_db(fe, audio_hz, exclude_audio_hz=(), **kw):
    """The median noise_psd of the channel at audio_hz, in dB against the true floor."""
    ch, fm = frontend.channelise(fe, FE_FS, frontend.fe_hz(audio_hz), CH_FS, n0=0)
    a = frontend.FE_CENTER_HZ + float(fm)
    psd = frontend.noise_psd(ch, CH_FS, exclude_hz=[h - a for h in exclude_audio_hz],
                             audio_hz=a, **kw)
    return 10 * math.log10(float(np.median(psd)) * FE_FS), ch, a


@pytest.mark.parametrize("audio_hz", [305.0, 326.0, 2695.0])
def test_the_floor_near_the_band_edges_is_the_true_one(white_fe, audio_hz, monkeypatch):
    new, _, _ = _floor_db(white_fe, audio_hz)
    assert abs(new) < 0.75
    monkeypatch.setenv(rules.EDGE_NOISE, "1")
    old, _, _ = _floor_db(white_fe, audio_hz)
    assert old < -6.0                      # (measured -7.0 to -9.6 dB)


@pytest.mark.parametrize("audio_hz, tone_hz", [(326.0, 400.0), (2695.0, 2600.0)])
def test_a_neighbour_on_the_flat_side_leaves_the_floor_true(monkeypatch, audio_hz, tone_hz):
    """A strong tone excluded on the flat side leaves under FE_MIN_HZ of
    flat bins clear (about 1 Hz at 326 Hz, none at 2695): the floor is
    then the geometric mean of the flat bins, tone and all (one bin of
    hundreds, so true), and the skirt compensated for the low-pass (true
    on white audio). Raw from the skirt, the old rule's, it reads low."""
    rng = np.random.default_rng(1)
    t = np.arange(8000 * 60) / 8000
    x = rng.standard_normal(len(t)) + math.sqrt(2) * 3.0 * np.cos(2 * np.pi * tone_hz * t)
    fe = frontend.audio_to_fe(x)
    new, _, _ = _floor_db(fe, audio_hz, (tone_hz,))
    assert abs(new) < 0.75
    monkeypatch.setenv(rules.EDGE_NOISE, "1")
    old, _, _ = _floor_db(fe, audio_hz, (tone_hz,))
    assert old < -6.0                      # (measured -12.5 dB at 326 Hz)


def test_with_no_clear_flat_bin_the_floor_is_halfway(monkeypatch):
    """2695 Hz beside 50 Hz of noise at 2600 Hz, 20 dB over the floor:
    the flat bins alone read the neighbour, and the floor (on white
    audio, where the compensated skirt is true) is their reading halved
    in dB, where the old rule's raw skirt reads low."""
    from scipy import signal as sg

    rng = np.random.default_rng(3)
    n = 8000 * 60
    t = np.arange(n) / 8000
    nb = sg.lfilter(sg.firwin(801, 25.0, fs=8000), 1.0,
                    rng.standard_normal(n) + 1j * rng.standard_normal(n))
    nb *= math.sqrt(2.0 * 50.0 * 100.0 / 4000.0 / np.mean(np.abs(nb) ** 2))
    fe = frontend.audio_to_fe(rng.standard_normal(n) + np.real(nb * np.exp(2j * np.pi * 2600.0 * t)))
    hi, _, _ = _floor_db(fe, 2695.0)                # no neighbour excluded: the flat bins, filled
    assert hi > 12.0
    new, _, _ = _floor_db(fe, 2695.0, (2600.0,))
    assert abs(new - hi / 2) < 1.0
    monkeypatch.setenv(rules.EDGE_NOISE, "1")
    old, _, _ = _floor_db(fe, 2695.0, (2600.0,))
    assert old < -6.0


@pytest.mark.parametrize("audio_hz", [1500.0, 430.0, 2570.0])
def test_mid_band_the_floor_is_unchanged(white_fe, audio_hz, monkeypatch):
    _, ch, a = _floor_db(white_fe, audio_hz)
    new = frontend.noise_psd(ch, CH_FS, exclude_hz=[60.0], audio_hz=a)
    bare = frontend.noise_psd(ch, CH_FS, exclude_hz=[60.0])
    monkeypatch.setenv(rules.EDGE_NOISE, "1")
    old = frontend.noise_psd(ch, CH_FS, exclude_hz=[60.0], audio_hz=a)
    assert np.array_equal(new, bare) and np.array_equal(old, bare)


def test_the_front_end_is_flat_from_430_to_2570_hz():
    """`_fe_flat` on a 10 s window: every bin flat across the middle of the
    band, counting the decimation's folds, and not past it."""
    w = 10 * CH_FS
    band = np.abs(np.fft.fftfreq(w, 1.0 / CH_FS))
    band = (band >= 85.0) & (band <= 120.0)
    for a in np.arange(430.0, 2570.5, 10.0):
        assert frontend._fe_flat(FE_FS, CH_FS, w, float(a))[band].all(), a
    for a in (300.0, 420.0, 2580.0, 2700.0):
        assert not frontend._fe_flat(FE_FS, CH_FS, w, a)[band].all(), a
    # at 330 Hz no bin below is flat, and above, those up to 111 Hz: past
    # that the decimation folds in noise from 250 Hz further down (about 200 Hz)
    f = np.fft.fftfreq(w, 1.0 / CH_FS)
    flat = frontend._fe_flat(FE_FS, CH_FS, w, 330.0)
    assert not flat[band & (f < 0)].any()
    assert flat[band & (f > 0) & (f <= 111.0)].all() and not flat[band & (f > 111.5)].any()


def test_the_channel_capture_measures_its_noise_where_the_front_end_is_flat(monkeypatch):
    seen = []
    monkeypatch.setattr(frontend, "noise_psd", lambda ch, fs, **kw: seen.append(kw) or
                        np.ones(1, np.float32))
    cap = track.ChanCapture(ch=np.ones(CH_FS * 20, np.complex64), fs=CH_FS, t0_index=0.0,
                            f_mix=Fraction(-4679, 4), exclude_hz=(400.0,))
    cap.noise_psd_at([0.0])
    assert seen[0]["audio_hz"] == 330.25 and seen[0]["exclude_hz"] == [400.0 - 330.25]


def test_the_lead_in_measures_its_noise_where_the_front_end_is_flat(monkeypatch):
    """`acquire.lead_in_span` passes its channel's audio frequency too."""
    seen = []
    monkeypatch.setattr(frontend, "noise_psd", lambda ch, fs, **kw: seen.append(kw) or
                        np.ones(1, np.float32))
    fe = (np.random.default_rng(2).standard_normal((FE_FS * 60, 2)) @ [1, 1j]).astype(np.complex64)
    acquire.lead_in_span(fe, 50.0 * FE_FS, 330.3, fs=FE_FS)
    f_mix = frontend.quantise_mix(frontend.fe_hz(330.3))
    assert seen and seen[0]["audio_hz"] == FE_CENTER_HZ + float(f_mix)
    assert abs(seen[0]["audio_hz"] - 330.3) < 0.1


# --- GATING -------------------------------------------------------------------------------------

def _det(f_hz, waveform=CE, method="tbd", z_ref=float("nan"), a3_z=float("nan")):
    t = np.array([-10.0, 0.0, 1800.0])
    timing = None if not np.isfinite(z_ref) else Timing(tau0=0.0, ppm=0.0, gamma=0.0,
                                                        z=z_ref, cov=np.eye(2))
    return Detection(f_hz=float(f_hz), path=FreqPath(t, np.full(3, float(f_hz)), np.ones(3)),
                     timing=timing, z_ref=z_ref, method=method, lead_in_s=0.0,
                     waveform=waveform, a3_z=a3_z)


def test_a_whole_slot_candidate_carries_its_lines_z():
    hit = acquire.TbdHit(path=FreqPath(np.array([0.0, 1.0]), np.array([1500.0, 1500.0]),
                                       np.ones(2)), z=17.5, p_value=1e-6, n_frames=2,
                         centre_hz=1500.0)
    assert hit.to_detection().a3_z == 17.5 and hit.to_detection(L).a3_z == 17.5


def test_the_own_band_of_a_preamble_found_signal(monkeypatch):
    ce = _det(1500.0, CE, "preamble", 50.0)
    l_ = _det(1709.5, L, "preamble", 50.0)                 # pilot 1709.5: band 1687.4-1712.6
    found = [ce, l_]
    # their own A3 lines: the CE's carrier, and the L's by its centre (a
    # weak L's path can run in its data band, gated as CE)
    a3 = RX._a3_lines([_det(1500.3, CE, a3_z=30.0), _det(1700.2, CE, a3_z=20.0)])

    def owned(f, w=CE, m="tbd", z=10.0, fnd=found, lines=a3):
        return RX._owned(_det(f, w, m, a3_z=z), fnd, lines)

    for f in (1480.5, 1487.6, 1500.0, 1512.4, 1519.5):    # within GHOST_HZ of the carrier
        assert owned(f) and owned(f, L) and owned(f, z=30.0), f
    for f in (1480.0, 1520.0, 1450.0):
        assert not owned(f), f
    for f in (1687.5, 1700.0, 1709.5, 1712.5):             # in the L's band, either waveform
        assert owned(f) and owned(f, L) and owned(f, z=20.0), f
    for f in (1687.3, 1712.7, 1730.0):
        assert not owned(f), f
    # a line stronger in A3 than the owner's own is gated: the owner's
    # preamble says nothing of the rest of the slot
    assert not owned(1512.4, z=30.5) and not owned(1690.0, L, z=21.0)
    # so is one beside a signal with no A3 line of its own, a hint, a
    # preamble hit, and anything beside a whole-slot or hint detection
    assert not owned(1512.4, lines=[]) and not owned(1512.4, lines=RX._a3_lines(
        [_det(1506.0, CE, a3_z=30.0)]))
    assert not owned(1512.4, m="hint", z=float("nan")) and not owned(1512.4, m="hint")
    assert not owned(1500.0, m="preamble") and not owned(1709.5, L, "preamble")
    weak = [_det(1500.0, CE, "tbd", 9.0), _det(1709.5, L, "hint", 9.0)]
    assert not any(owned(f, fnd=weak) for f in (1500.0, 1512.4, 1700.0))
    monkeypatch.setenv(rules.GATING, "1")
    assert not owned(1512.4) and not owned(1700.0)


def test_the_ce_receiver_does_not_gate_a_found_signals_band(monkeypatch):
    """CE alone: ghosts of two preamble hits (1487.6 beside 1500's line,
    1587.6 beside 1600's) go ungated and uncounted; a hint in 1500's band,
    a line stronger than 1600's own (1615: its preamble faded), and lines
    beside a whole-slot detection are gated. Set back, every one is."""
    calls = []

    def verify(chan, fs, spec, d, erase_s=()):
        calls.append(d.f_hz)
        return _det(d.f_hz, CE, d.method, 40.0 if d.method == "preamble" else 8.0)

    monkeypatch.setattr(RX, "channel_for", lambda *a, **k: None)
    monkeypatch.setattr(track, "verify", verify)
    monkeypatch.setattr(RX.acquire, "lead_in_span", lambda *a, **k: 0.0)
    prep = SimpleNamespace(fe=None, t0_index=0.0, fs=FE_FS)
    cands = [_det(1500.0, CE, "preamble"), _det(1600.0, CE, "preamble"),
             _det(1512.0, CE, "hint"),
             _det(1500.4, a3_z=30.0), _det(1487.6, a3_z=6.0), _det(1512.4, a3_z=6.0),
             _det(1615.0, a3_z=40.0), _det(1600.2, a3_z=12.0), _det(1587.6, a3_z=5.0),
             _det(1800.0, a3_z=9.0), _det(1810.0, a3_z=7.0)]
    RX.detect(prep, frame.SHORT, cands=cands)
    assert calls == [1500.0, 1600.0, 1512.0, 1615.0, 1800.0, 1810.0]
    # max_candidates counts what is gated or merged, not the lines skipped
    # (the owners' own lines among them)
    calls.clear()
    RX.detect(prep, frame.SHORT, cands=cands, max_candidates=5)
    assert calls == [1500.0, 1600.0, 1512.0, 1615.0, 1800.0]
    calls.clear()
    monkeypatch.setenv(rules.GATING, "1")
    RX.detect(prep, frame.SHORT, cands=cands)
    assert calls == [1500.0, 1600.0, 1512.0, 1487.6, 1615.0, 1587.6, 1800.0, 1810.0]
    calls.clear()
    RX.detect(prep, frame.SHORT, cands=cands, max_candidates=5)
    assert calls == [1500.0, 1600.0, 1512.0, 1487.6]


def test_the_joint_receiver_does_not_gate_a_found_signals_band(monkeypatch):
    """Both waveforms: the L copy of a CE carrier found on its preamble
    (pilot at the carrier, centre 9.5 Hz below) and its ghost go ungated,
    by the candidate's line and the found signal's waveform; a CE whose
    preamble faded inside an L signal's band, stronger in A3 than the
    L's own line, is gated, and so is its L copy; outside those bands,
    and beside a whole-slot detection, all are."""
    calls = []

    def gate(prep, spec, d, others, erase_s):
        calls.append((d.waveform, d.f_hz))
        return _det(d.f_hz, d.waveform, d.method, 40.0 if d.method == "preamble" else 8.0)

    monkeypatch.setattr(RX, "_gate", gate)
    cands = [_det(1509.5, L, "preamble"), _det(1700.0, CE, "preamble"),
             _det(1600.0, L, "hint"),
             _det(1509.7, L, a3_z=20.0), _det(1489.0, CE, a3_z=45.0), _det(1489.0, L, a3_z=45.0),
             _det(1470.0, CE, a3_z=8.0), _det(1700.2, CE, a3_z=30.0), _det(1700.0, L, a3_z=30.0),
             _det(1712.4, CE, a3_z=9.0), _det(1720.0, CE, a3_z=9.0),
             _det(1900.0, CE, a3_z=12.0), _det(1912.4, CE, a3_z=5.0), _det(1915.5, L, a3_z=5.0)]
    RX.detect(None, (frame.SHORT, lframe.L_SHORT), cands=cands)
    assert calls == [(L, 1509.5), (CE, 1700.0), (L, 1600.0), (CE, 1489.0), (L, 1489.0),
                     (CE, 1470.0), (CE, 1720.0), (CE, 1900.0), (CE, 1912.4), (L, 1915.5)]
    calls.clear()
    monkeypatch.setenv(rules.GATING, "1")
    RX.detect(None, (frame.SHORT, lframe.L_SHORT), cands=cands)
    assert calls == [(L, 1509.5), (CE, 1700.0), (L, 1600.0), (CE, 1489.0), (L, 1489.0),
                     (CE, 1470.0), (L, 1700.0), (CE, 1712.4), (CE, 1720.0),
                     (CE, 1900.0), (CE, 1912.4), (L, 1915.5)]

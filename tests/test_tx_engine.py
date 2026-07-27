"""Transmit sequencing, with the emphasis on never leaving PTT keyed.

No soundcard and no model: the player is injected, and the encode step
is bypassed by handing the engine a stub model, so these run in
milliseconds and exercise the sequencing rather than the DSP.
"""

import threading
import time

import numpy as np
import pytest

from sstvae.config import FS
from sstvae.tx.engine import TxConfig, TxEngine, TxPhase, condition_for_output


class FakePtt:
    """Records every key/unkey in order."""

    def __init__(self, fail_on=None):
        self.events = []
        self.fail_on = fail_on  # True/False to raise on that transition
        self.lock = threading.Lock()

    def set_ptt(self, on: bool):
        with self.lock:
            if self.fail_on is not None and on is self.fail_on:
                self.events.append(("fail", on))
                raise RuntimeError("serial port went away")
            self.events.append(on)

    @property
    def keyed(self):
        real = [e for e in self.events if isinstance(e, bool)]
        return bool(real) and real[-1] is True


def instant_player(device, wave, samplerate=FS, on_progress=None,
                   should_stop=None, on_error=None):
    if on_progress:
        on_progress(1.0)
    return True


def make_engine(ptt=None, player=instant_player, **kw):
    engine = TxEngine(ptt=ptt, player=player, model=object(), **kw)
    # prepare() is exercised separately; here we only care about keying,
    # so hand _keyed_send a ready-made waveform.
    return engine


def short_wave(seconds=0.5):
    return np.zeros(int(seconds * FS))


def test_ptt_brackets_the_transmission():
    ptt = FakePtt()
    engine = make_engine(ptt)
    cfg = TxConfig(ptt_lead_s=0.0, ptt_tail_s=0.0)

    assert engine._keyed_send(short_wave(), cfg) is True

    assert ptt.events == [True, False], "PTT must go up before audio and down after"
    assert engine.state.phase is TxPhase.DONE


def test_ptt_drops_when_the_player_raises():
    def exploding_player(*a, **kw):
        raise RuntimeError("USB audio device disappeared")

    ptt = FakePtt()
    errors = []
    engine = make_engine(ptt, player=exploding_player, on_error=errors.append)

    assert engine._keyed_send(short_wave(), TxConfig(ptt_lead_s=0.0, ptt_tail_s=0.0)) is False

    assert ptt.keyed is False, "a crash mid-transmission must still unkey"
    assert engine.state.phase is TxPhase.FAILED
    assert any("USB audio" in e for e in errors)


def test_ptt_drops_when_cancelled_mid_playback():
    started = threading.Event()

    def slow_player(device, wave, samplerate=FS, on_progress=None,
                    should_stop=None, on_error=None):
        started.set()
        for _ in range(200):
            if should_stop and should_stop():
                return False
            time.sleep(0.005)
        return True

    ptt = FakePtt()
    engine = make_engine(ptt, player=slow_player)
    cfg = TxConfig(ptt_lead_s=0.0, ptt_tail_s=0.0)

    result = {}
    th = threading.Thread(target=lambda: result.update(ok=engine._keyed_send(short_wave(), cfg)))
    th.start()
    assert started.wait(2.0)
    engine.cancel()
    th.join(timeout=5.0)

    assert result["ok"] is False
    assert ptt.keyed is False
    assert engine.state.phase is TxPhase.CANCELLED


def test_cancel_during_the_ptt_lead_delay_still_unkeys():
    ptt = FakePtt()
    engine = make_engine(ptt)
    engine.cancel()  # cancelled before it even starts

    assert engine._keyed_send(short_wave(), TxConfig(ptt_lead_s=5.0)) is False
    assert ptt.keyed is False
    assert engine.state.phase is TxPhase.CANCELLED


def test_failure_to_unkey_is_reported_loudly():
    ptt = FakePtt(fail_on=False)
    errors = []
    engine = make_engine(ptt, on_error=errors.append)

    engine._keyed_send(short_wave(), TxConfig(ptt_lead_s=0.0, ptt_tail_s=0.0))

    assert any("PTT OFF FAILED" in e for e in errors), (
        "the operator has to be told the rig may still be transmitting"
    )


def test_watchdog_unkeys_a_wedged_transmission():
    """The transmit path's own finally never runs if the player hangs
    forever; the watchdog is the independent backstop."""
    from sstvae.tx.engine import _PttWatchdog

    ptt = FakePtt()
    ptt.set_ptt(True)
    fired = threading.Event()
    wd = _PttWatchdog(ptt, timeout_s=0.05, on_fire=fired.set)
    wd.start()

    assert fired.wait(2.0), "watchdog did not fire"
    assert ptt.keyed is False


def test_watchdog_stays_quiet_on_a_normal_transmission():
    from sstvae.tx.engine import _PttWatchdog

    ptt = FakePtt()
    fired = threading.Event()
    wd = _PttWatchdog(ptt, timeout_s=0.5, on_fire=fired.set)
    wd.start()
    wd.cancel()
    assert not fired.wait(0.8)
    assert ptt.events == []


def test_no_rig_configured_is_not_an_error():
    engine = make_engine(ptt=None)
    assert engine._keyed_send(short_wave(), TxConfig(ptt_lead_s=0.0, ptt_tail_s=0.0)) is True


def test_output_conditioning_scales_to_the_requested_peak():
    x = np.array([0.0, 0.25, -0.5, 0.1])
    out = condition_for_output(x, level=0.9)
    assert np.max(np.abs(out)) == pytest.approx(0.9)
    # Shape preserved -- it is a scale, not a clip: the modem already did
    # the envelope clipping that sets PAPR.
    assert np.allclose(out / np.max(np.abs(out)), x / np.max(np.abs(x)))


def test_output_conditioning_survives_digital_silence():
    assert np.all(condition_for_output(np.zeros(10), 0.9) == 0)


def test_progress_reaches_the_callback():
    seen = []
    engine = make_engine(FakePtt(), on_state=lambda s: seen.append(s.progress))
    engine._keyed_send(short_wave(), TxConfig(ptt_lead_s=0.0, ptt_tail_s=0.0))
    assert 1.0 in seen

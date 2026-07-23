import numpy as np
import pytest

from sstvae import hfchannel
from sstvae.config import (
    FS,
    MODES,
    LEADIN_SAMPLES,
    PREAMBLE_SAMPLES,
    HEADER_SAMPLES,
    FRAME_SAMPLES,
)
from sstvae.modem import Modem, beacon
from sstvae.modem.sync import acquire_blind, SyncError


def _tx(seed=0, callsign="N0CALL"):
    modem = Modem()
    lat = np.random.default_rng(seed).normal(size=MODES["C"].n_latents)
    lat /= np.sqrt(np.mean(lat**2))
    x = modem.modulate(lat, "C", callsign=callsign)
    frames_start = LEADIN_SAMPLES + PREAMBLE_SAMPLES + HEADER_SAMPLES
    return modem, lat, x, frames_start


def _frames_slice(x, frames_start, start_frame, n_frames):
    lo = frames_start + start_frame * FRAME_SAMPLES
    return x[lo : lo + n_frames * FRAME_SAMPLES]


def test_acquire_blind_no_preamble_present_at_all():
    """The window contains *only* frame data — no lead-in, preamble, or
    header anywhere in it — yet timing still locks from pilot
    periodicity alone."""
    _, _, x, frames_start = _tx()
    win = _frames_slice(x, frames_start, 300, beacon.MIN_FRAMES_FOR_SYNC + 5)
    from sstvae.config import NCP
    from sstvae.modem.dsp import to_baseband

    ba = acquire_blind(to_baseband(win))
    # window starts exactly on a frame boundary, so the useful-window
    # start of the first pilot found should land right after its CP
    assert ba.frame_start % FRAME_SAMPLES == NCP
    assert abs(ba.freq_offset) < 2.0


def test_acquire_blind_rejects_pure_noise():
    rng = np.random.default_rng(0)
    junk = rng.normal(size=FRAME_SAMPLES * 20)
    with pytest.raises(SyncError):
        acquire_blind(junk.astype(np.float64) * 0 + rng.normal(size=junk.shape))


def test_demodulate_blind_recovers_position_and_callsign():
    modem, lat, x, frames_start = _tx(seed=1, callsign="K6ABC/P")
    win = _frames_slice(x, frames_start, 300, 90)
    r = modem.demodulate_blind(win)
    assert r.frame_offset == 300
    assert r.callsign == "K6ABC/P"
    good = r.weights > 0.5
    assert good.sum() > 0.5 * lat.size * (90 / MODES["C"].n_frames)
    err = np.mean((lat[good] - r.latents[good]) ** 2)
    snr = 10 * np.log10(np.mean(lat[good] ** 2) / err)
    assert snr > 18


def test_demodulate_blind_survives_awgn_and_cfo():
    modem, lat, x, frames_start = _tx(seed=2, callsign="W1AW")
    win = _frames_slice(x, frames_start, 150, 90)
    y = hfchannel.apply_channel(win, snr_db=20.0, freq_offset_hz=15.0)
    r = modem.demodulate_blind(y)
    assert r.frame_offset == 150
    assert r.callsign == "W1AW"
    assert abs(r.freq_offset - 15.0) < 2.0


def test_retrospective_decode_using_a_late_lock_window():
    """The core scenario: the receiver only searches/locks using the
    tail of a recorded buffer (simulating 'noticed the signal late'),
    but the whole buffer — including frames recorded before the lock
    point — still comes back at the correct absolute position."""
    modem, lat, x, frames_start = _tx(seed=3, callsign="N0CALL")
    buf_start_frame = 200
    buf = _frames_slice(x, frames_start, buf_start_frame, 300)
    search_s = ((len(buf) - 100 * FRAME_SAMPLES) / FS, len(buf) / FS)
    r = modem.demodulate_blind(buf, search_s=search_s)
    assert r.frame_offset == buf_start_frame
    assert r.callsign == "N0CALL"
    # frames from *before* the search window (retrospective) also decoded
    good = r.weights > 0.5
    assert good.sum() > 0
    err = np.mean((lat[good] - r.latents[good]) ** 2)
    snr = 10 * np.log10(np.mean(lat[good] ** 2) / err)
    assert snr > 18


def test_window_shorter_than_min_frames_for_sync_may_fail_gracefully():
    """Below beacon.MIN_FRAMES_FOR_SYNC there's no guarantee of a full
    superframe fitting; demodulate_blind must not report a wrong
    position — either it decodes correctly, or beacon is None."""
    modem, lat, x, frames_start = _tx(seed=4, callsign="N0CALL")
    for start_frame in range(0, MODES["C"].n_frames - 40, 40):
        win = _frames_slice(x, frames_start, start_frame, 30)
        try:
            r = modem.demodulate_blind(win)
        except SyncError:
            continue
        if r.beacon is not None:
            assert r.frame_offset == start_frame

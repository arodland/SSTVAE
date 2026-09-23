"""The receiver's whole-transmission measurements (2026-09-22, ported
from Data2G): step-phase undo, residual CFO, window placement. Placement
itself is pinned in test_first_path.py."""

import numpy as np

from sstvae import hfchannel
from sstvae.config import DEMOD_BACKOFF, MODES, NC, NCP
from sstvae.modem import Modem, ofdm
from sstvae.modem.dsp import to_baseband
from sstvae.modem.modem import _time_shift_phase


def _tx(mode="A", seed=0):
    lat = np.random.default_rng(seed).normal(size=MODES[mode].n_latents)
    lat /= np.sqrt(np.mean(lat**2))
    return Modem().modulate(lat, mode)


def test_step_phase_undo_has_the_right_sign():
    """A window moved later by s, times _time_shift_phase(s), is the
    unmoved window. Inside the CP both windows see the same symbol, so
    the match is exact; a wrong sign doubles the ramp instead."""
    rng = np.random.default_rng(0)
    sym = np.exp(2j * np.pi * rng.random((1, NC)))
    z = to_baseband(np.concatenate([ofdm.modulate_symbols(sym), np.zeros(64)]))
    ref = ofdm.demod_window(z, NCP, DEMOD_BACKOFF)
    for s in (-2, 2):
        moved = ofdm.demod_window(z, NCP + s, DEMOD_BACKOFF)
        assert np.allclose(moved * _time_shift_phase(s), ref, atol=1e-12)


def test_frequency_comes_from_the_whole_transmission():
    """Fading's random FM makes the preamble's 80 ms estimate heavy-
    tailed: measured on mpd at 8 dB, p50 0.32 Hz and max 2.1 Hz over 48
    seeds. The pilots over the whole transmission hold 0.17 Hz there."""
    modem, x = Modem(), _tx()
    errs = [
        abs(modem.demodulate(hfchannel.apply_channel(
            x, snr_db=8.0, freq_offset_hz=21.0, fading_preset="mpd", seed=s
        )).freq_offset - 21.0)
        for s in range(8)
    ]
    assert max(errs) < 0.3, errs

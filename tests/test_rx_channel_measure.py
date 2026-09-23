"""The receiver's whole-transmission measurements (2026-09-22, ported
from Data2G)."""

import numpy as np

from sstvae.config import DEMOD_BACKOFF, NC, NCP
from sstvae.modem import ofdm
from sstvae.modem.dsp import to_baseband
from sstvae.modem.modem import _time_shift_phase


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


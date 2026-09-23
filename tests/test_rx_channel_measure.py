"""The receiver's whole-transmission measurements (2026-09-22, ported
from Data2G): step-phase undo, residual CFO, window placement. Placement
itself is pinned in test_first_path.py."""

import numpy as np

from sstvae import hfchannel
from sstvae.config import (
    DEMOD_BACKOFF, FRAME_SAMPLES, HEADER_SAMPLES, LEADIN_SAMPLES, MODES, NC, NCP,
    PREAMBLE_SAMPLES,
)
from sstvae.modem import Modem, ofdm
from sstvae.modem.dsp import to_baseband
from sstvae.modem.modem import _time_shift_phase


def _lat(mode="A", seed=0):
    lat = np.random.default_rng(seed).normal(size=MODES[mode].n_latents)
    return lat / np.sqrt(np.mean(lat**2))


def _tx(mode="A", seed=0):
    return Modem().modulate(_lat(mode, seed), mode)


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


# --- the 2-D LMMSE channel estimate ------------------------------------------

def _eff_snr_db(truth, got):
    """Best linear fit of latents*weights to the truth, over what was placed."""
    placed = got != 0
    t, g = truth[placed], got[placed]
    rho2 = np.dot(t, g) ** 2 / (np.dot(t, t) * np.dot(g, g))
    return 10 * np.log10(rho2 / (1 - rho2))


def _rx_snr(x, lat, blind=False, acquisition=None, **channel):
    rx = hfchannel.apply_channel(x, **channel)
    m = Modem()
    if blind:
        cut = LEADIN_SAMPLES + PREAMBLE_SAMPLES + HEADER_SAMPLES + 20 * FRAME_SAMPLES
        r = m.demodulate_blind(rx[cut:], acquisition=acquisition)
    else:
        r = m.demodulate(rx)
    return _eff_snr_db(lat, (r.latents * r.weights)[: len(lat)])


def test_the_estimate_smooths_the_pilot_noise():
    """Catmull-Rom passed every pilot's noise straight to the equalizer:
    3.56 dB of latent SNR at 3 dB AWGN over 48 seeds, against 4.82 for
    the LMMSE estimate. The bar sits between the two."""
    snrs = [_rx_snr(_tx(seed=s), _lat(seed=s), snr_db=3.0, seed=s) for s in range(4)]
    assert np.mean(snrs) > 4.3, snrs


def test_clock_drift_does_not_smear_the_delay_support():
    """Undone, the timing steps let the path delays drift ~20 samples
    across mode A at 80 ppm, and projecting onto that smeared support
    cost 3.4 dB. Projected with the drift taken out, the clock costs
    nothing measurable."""
    loss = [
        _rx_snr(_tx(seed=s), _lat(seed=s), snr_db=8.0, fading_preset="mpp", seed=s)
        - _rx_snr(_tx(seed=s), _lat(seed=s), snr_db=8.0, fading_preset="mpp", ppm=80.0, seed=s)
        for s in range(3)
    ]
    assert np.mean(loss) < 0.4, loss


def test_the_doppler_model_follows_a_residual_frequency():
    """The blind path carries a few tenths of a Hz to a few Hz of
    residual frequency, and a Doppler spectrum centred on zero averages
    rotating pilots away. Centred on the pilots' own rotation, a 1.5 Hz
    error in the acquisition costs almost nothing."""
    from dataclasses import replace

    from sstvae.modem.sync import acquire_blind

    x, lat = _tx(seed=5), _lat(seed=5)
    cut = LEADIN_SAMPLES + PREAMBLE_SAMPLES + HEADER_SAMPLES + 20 * FRAME_SAMPLES
    rx = hfchannel.apply_channel(x, snr_db=8.0, fading_preset="mpp", seed=5)
    acq = acquire_blind(to_baseband(rx[cut:]))
    exact = _rx_snr(x, lat, blind=True, acquisition=replace(acq, freq_offset=0.0),
                    snr_db=8.0, fading_preset="mpp", seed=5)
    off = _rx_snr(x, lat, blind=True, acquisition=replace(acq, freq_offset=1.5),
                  snr_db=8.0, fading_preset="mpp", seed=5)
    assert exact - off < 0.3, (exact, off)

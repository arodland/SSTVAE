"""Acquisition frequency range, and the optional carrier-drift loop.

Three changes landed together (2026-08-11, measured in docs/todo.md) and
each has a distinct way of going quietly wrong:

* the preamble search got wider, and the thing to protect is that it
  cost nothing -- a wider search that quietly changed the answer for
  an on-frequency station would be a regression dressed as a feature;
* the blind search got a much coarser CFO grid, which is only safe
  because the sub-bin peak is interpolated. Drop `refine_cfo` and every
  existing test still passes while the receiver hands the demodulator
  several Hz of error;
* drift tracking is off by default, and "off" has to mean *nothing
  happens*, not "a loop runs with small gains".
"""

import numpy as np
import pytest

from sstvae import config, hfchannel
from sstvae.config import ACQUIRE_MAX_BINS, FS, LEADIN_SAMPLES, MODES
from sstvae.modem import dsp, sync
from sstvae.modem.modem import CFO_PULL_HZ, Modem, _make_tracker

from conftest import latent_snr_db as _latent_snr_db, unit_latents as _unit_latents


@pytest.fixture(scope="module")
def modem():
    return Modem()


@pytest.fixture(scope="module")
def mode_a(modem):
    spec = MODES["A"]
    latents = _unit_latents("A", seed=3)
    return spec, latents, modem.modulate(latents, spec, callsign="TEST")


# --- the preamble search ---------------------------------------------------

def test_the_search_reaches_where_the_arithmetic_says_it_does(modem, mode_a):
    """max_bins covers +-(25 + 50*max_bins) Hz. 300 Hz is inside the new
    default and far outside the old one, so this fails against a
    reverted ACQUIRE_MAX_BINS rather than merely getting slower."""
    spec, latents, wave = mode_a
    assert 300.0 < 25 + 50 * ACQUIRE_MAX_BINS
    y = hfchannel.apply_channel(wave, snr_db=6.0, freq_offset_hz=300.0, seed=1)
    result = modem.demodulate(y)
    assert result.mode.name == "A"
    assert result.frames_received == spec.n_frames
    assert abs(result.freq_offset - 300.0) < 2.0
    assert _latent_snr_db(latents, result.latents, result.weights) > 0.0


def test_a_wider_search_returns_the_identical_answer_on_frequency(mode_a):
    """The property that made this free rather than a trade: the extra
    candidates never win, so a station that acquired before acquires
    identically -- not merely comparably."""
    _, _, wave = mode_a
    for seed in range(4):
        y = hfchannel.apply_channel(wave, snr_db=0.0, freq_offset_hz=0.0, seed=seed)
        z = dsp.to_baseband(y)
        narrow = sync.acquire(z, max_bins=2)
        wide = sync.acquire(z, max_bins=ACQUIRE_MAX_BINS)
        assert narrow.preamble_start == wide.preamble_start
        assert narrow.freq_offset == wide.freq_offset
        assert narrow.metric == wide.metric


def test_detection_is_blind_to_frequency_offset(mode_a):
    """Why widening cannot cost false alarms: a CFO multiplies every
    lag-M product by one constant phasor, which |.| removes. So the
    detection metric -- the only thing the threshold sees -- does not
    depend on the offset at all, up to what the sync filter removes."""
    _, _, wave = mode_a
    clean = hfchannel.apply_channel(wave, snr_db=None, freq_offset_hz=0.0)
    metric0 = sync._autocorr_metric(dsp.sync_lowpass(dsp.to_baseband(clean)))[0].max()
    for offset in (100.0, 300.0):
        shifted = hfchannel.apply_channel(wave, snr_db=None, freq_offset_hz=offset)
        metric = sync._autocorr_metric(dsp.sync_lowpass(dsp.to_baseband(shifted)))[0].max()
        assert metric == pytest.approx(metric0, abs=0.05)


# --- the blind search's coarse grid ----------------------------------------

def test_the_grid_is_coarse_but_the_block_is_not():
    """These are two different numbers and were one. Deriving the block
    from the search grid collapses it to a few hundred samples the
    moment the grid is coarsened, which hands back most of the saving
    while every other test still passes.

    Reference-only: the block size is an internal of the accumulator, and
    under --native `sync.BlindAccumulator` is a wrapper around the C++
    object that does not expose it. tests/test_native_parity.py is where
    the two implementations are held to the same *behaviour*; this one is
    about how the reference gets there."""
    acc = sync.BlindAccumulator()
    if not hasattr(acc, "_block"):
        pytest.skip("native accumulator does not expose its block size")
    assert acc._block >= FS / config.BLIND_BLOCK_RES_HZ
    # The grid must be far coarser than the block's own resolution --
    # that gap is the whole saving.
    assert config.BLIND_BIN_STEP_HZ > 5 * acc._bin_hz


def test_refine_cfo_finds_a_peak_between_bins():
    """A parabola through three points, on deliberately *non-uniform*
    abscissae -- the real grid is non-uniform, because a shift must be a
    whole number of block-FFT bins."""
    freqs = np.array([-13.5, 0.0, 11.9])
    peak = 3.0
    scores = -((freqs - peak) ** 2) + 100.0
    assert sync.refine_cfo(freqs, scores, 1) == pytest.approx(peak, abs=1e-9)
    # An edge bin has no bracket, so it must report itself rather than
    # extrapolate off the end of the grid.
    assert sync.refine_cfo(freqs, scores, 0) == freqs[0]
    assert sync.refine_cfo(freqs, scores, 2) == freqs[2]


def test_refine_cfo_never_leaves_the_bracketing_bins():
    """Three noisy points can put a parabola's vertex anywhere; outside
    the bracket it is an extrapolation, not a peak."""
    freqs = np.array([-12.5, 0.0, 12.5])
    for scores in ([1.0, 1.0000001, 100.0], [100.0, 1.0, 1.0], [5.0, 5.0, 5.0]):
        out = sync.refine_cfo(freqs, np.array(scores), 1)
        assert freqs[0] <= out <= freqs[2]


def test_the_coarse_grid_still_estimates_frequency_well(modem, mode_a):
    """The grid is ~7x coarser than the estimate it has to produce, so
    this is the assertion that interpolation is doing its job. The bound
    is well inside the ~2 Hz the demodulator can absorb, and well
    outside half a bin (6.25 Hz), which is what a raw argmax gives."""
    _, _, wave = mode_a
    frames = wave[LEADIN_SAMPLES + config.PREAMBLE_SAMPLES + config.HEADER_SAMPLES :]
    for offset in (-31.0, -7.0, 4.5, 22.0):
        y = hfchannel.apply_channel(frames[: 20 * FS], snr_db=6.0,
                                    freq_offset_hz=offset, seed=2)
        acq = sync.acquire_blind(dsp.to_baseband(y))
        assert abs(acq.freq_offset - offset) < 2.0


def test_the_wide_blind_range_reaches_it(modem, mode_a):
    """The opt-in range. 300 Hz is far outside the narrow default, so a
    narrow accumulator must *not* find it -- otherwise this test would
    pass with the option doing nothing."""
    _, _, wave = mode_a
    frames = wave[LEADIN_SAMPLES + config.PREAMBLE_SAMPLES + config.HEADER_SAMPLES :]
    y = hfchannel.apply_channel(frames[: 20 * FS], snr_db=6.0,
                                freq_offset_hz=300.0, seed=2)
    z = dsp.to_baseband(y)

    wide = sync.BlindAccumulator(max_offset_hz=config.BLIND_WIDE_MAX_OFFSET_HZ)
    wide.push(z, 0)
    assert abs(wide.result().freq_offset - 300.0) < 2.0

    narrow = sync.BlindAccumulator()
    narrow.push(z, 0)
    with pytest.raises(sync.SyncError):
        narrow.result()


# --- drift tracking --------------------------------------------------------

def _drifted(wave, rate_hz_s):
    """A linear carrier drift of `rate` Hz/s. Phase reduced to one turn
    before exp(), as dsp.wrap_cycles requires."""
    from scipy import signal

    t = np.arange(len(wave)) / FS
    cycles = dsp.wrap_cycles(0.5 * rate_hz_s * t**2)
    return np.real(signal.hilbert(wave) * np.exp(2j * np.pi * cycles))


def test_off_means_no_loop_at_all():
    """Not "a loop with small gains" -- the default path must not run the
    tracker, so it cannot cost anything it was not asked for."""
    assert _make_tracker("off") is None
    assert _make_tracker("slow") is not None
    assert _make_tracker("fast") is not None
    with pytest.raises(ValueError):
        _make_tracker("medium")


def test_off_is_bit_identical_to_not_asking(modem, mode_a):
    _, latents, wave = mode_a
    y = hfchannel.apply_channel(wave, snr_db=6.0, seed=5)
    a = modem.demodulate(y)
    b = modem.demodulate(y, drift_track="off")
    assert np.array_equal(a.latents, b.latents)
    assert np.array_equal(a.weights, b.weights)


def test_tracking_rescues_a_drifting_carrier(modem, mode_a):
    """0.5 Hz/s is 16 Hz over a mode A over -- far past the ~2 Hz the
    untracked receiver can absorb, and well inside what the loop
    follows. Both settings must handle a plain ramp; they differ on
    fading and on fast wander, not here."""
    _, latents, wave = mode_a
    y = hfchannel.awgn(_drifted(wave, 0.5), 6.0, seed=6)

    off = modem.demodulate(y, drift_track="off")
    assert _latent_snr_db(latents, off.latents, off.weights) < 0.0

    for setting in ("slow", "fast"):
        tracked = modem.demodulate(y, drift_track=setting)
        # Every frame still arrives either way -- the failure is silent,
        # which is why this asserts on quality rather than frame count.
        assert tracked.frames_received == off.frames_received
        assert _latent_snr_db(latents, tracked.latents, tracked.weights) > 3.0


def test_tracking_costs_nothing_when_there_is_no_drift(modem, mode_a):
    """The condition for shipping it as an option rather than not at
    all. On a clean, static carrier a loop that has nothing to do must
    not damage what it is tracking."""
    _, latents, wave = mode_a
    y = hfchannel.apply_channel(wave, snr_db=6.0, seed=7)
    base = _latent_snr_db(latents, *_lw(modem.demodulate(y, drift_track="off")))
    for setting in ("slow", "fast"):
        got = _latent_snr_db(latents, *_lw(modem.demodulate(y, drift_track=setting)))
        assert got > base - 0.2


def _lw(result):
    return result.latents, result.weights


def test_blind_demod_takes_the_same_setting(modem, mode_a):
    """demodulate_blind has pilots and so has the identical problem; the
    loop needs no preamble reference, unlike the sample-clock tracker.
    0.2 Hz/s over 30 s is 6 Hz of drift, whose ~3 Hz of initial residual
    is inside the loop's pull-in -- see the next test for what happens
    outside it."""
    _, _, wave = mode_a
    frames = wave[LEADIN_SAMPLES + config.PREAMBLE_SAMPLES + config.HEADER_SAMPLES :]
    y = hfchannel.awgn(_drifted(frames[: 30 * FS], 0.2), 6.0, seed=8)
    plain = modem.demodulate_blind(y)
    tracked = modem.demodulate_blind(y, drift_track="slow")
    assert _good_frac(tracked) > _good_frac(plain)
    with pytest.raises(ValueError):
        modem.demodulate_blind(y, drift_track="medium")


def test_the_blind_pull_in_limit_is_real_and_is_a_cliff(modem, mode_a):
    """Pinned deliberately, as a known limit rather than a surprise.

    A pilot-rate estimator is unambiguous only over +-CFO_PULL_HZ, and
    `acquire_blind` hands the loop a frequency describing the *middle*
    of its window -- so at 0.5 Hz/s over 30 s the first frame is 7.7 Hz
    out, the measurement aliases to a small wrong value, and the loop
    locks to it. The failure is not graceful: the beacon stops decoding,
    so nothing is placed at all.

    If someone anchors the loop at the window's middle and runs it
    outward (the fix named in demodulate_blind's docstring), this test
    should start failing -- that is the point of writing it down."""
    _, _, wave = mode_a
    frames = wave[LEADIN_SAMPLES + config.PREAMBLE_SAMPLES + config.HEADER_SAMPLES :]
    y = hfchannel.awgn(_drifted(frames[: 30 * FS], 0.5), 6.0, seed=8)

    acq = sync.acquire_blind(dsp.to_baseband(y))
    assert abs(acq.freq_offset) > CFO_PULL_HZ  # the residual the loop must start from

    assert _good_frac(modem.demodulate_blind(y)) > 0.2
    assert _good_frac(modem.demodulate_blind(y, drift_track="slow")) == 0.0


def _good_frac(result):
    """Fraction of latents the demodulator was confident about. The blind
    path places nothing at all unless the beacon decodes, so this is 0
    exactly when the reception failed outright."""
    return float(np.mean(result.weights > 0.5))

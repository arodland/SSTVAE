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
from sstvae.modem.dsp import to_baseband
from sstvae.modem.sync import BlindAccumulator, acquire_blind, SyncError

from conftest import snr_floor_db

# Blind decode has no preamble phase reference and so no clock-drift
# tracking; it normally lands within ~0.15 dB of the clean-loopback
# clip floor, but accumulated timing error over a long buffer
# interacts with clip distortion and it dipped 2.5 dB at one
# headroom setting (3.0 dB) out of ten sampled from 8.0 down to
# -5.0. Margin covers that rather than tracking the floor tightly.
BLIND_MARGIN_DB = 3.5


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


def test_demodulate_blind_recovers_position_and_callsign(clip_floor_db):
    modem, lat, x, frames_start = _tx(seed=1, callsign="K6ABC/P")
    win = _frames_slice(x, frames_start, 300, 90)
    r = modem.demodulate_blind(win)
    assert r.frame_offset == 300
    assert r.callsign == "K6ABC/P"
    good = r.weights > 0.5
    assert good.sum() > 0.5 * lat.size * (90 / MODES["C"].n_frames)
    err = np.mean((lat[good] - r.latents[good]) ** 2)
    snr = 10 * np.log10(np.mean(lat[good] ** 2) / err)
    assert snr > snr_floor_db(clip_floor_db, margin_db=BLIND_MARGIN_DB)


def test_demodulate_blind_survives_awgn_and_cfo():
    modem, lat, x, frames_start = _tx(seed=2, callsign="W1AW")
    win = _frames_slice(x, frames_start, 150, 90)
    y = hfchannel.apply_channel(win, snr_db=20.0, freq_offset_hz=15.0)
    r = modem.demodulate_blind(y)
    assert r.frame_offset == 150
    assert r.callsign == "W1AW"
    assert abs(r.freq_offset - 15.0) < 2.0


def test_retrospective_decode_using_a_late_lock_window(clip_floor_db):
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
    assert snr > snr_floor_db(clip_floor_db, margin_db=BLIND_MARGIN_DB)


def test_frame0_start_locates_absolute_frame_zero_after_a_late_lock():
    """frame0_start must point at absolute frame 0 no matter where in the
    buffer the blind lock landed.

    It was anchored on p0 (the CP-start of the frame the lock found)
    rather than p_start (where the demod loop, and so the beacon chip
    stream that frame_offset indexes, actually begins). The two differ by
    L_lo frames, so a lock late in a long recording reported absolute
    frame 0 tens of seconds away from the truth. The latents still landed
    in the right slots -- only the reported position was wrong -- so
    nothing downstream of the image caught it, but a caller using
    frame0_start to identify *which* transmission this is (sstvae_listen's
    dedup) saw one transmission as two.
    """
    modem, _, x, frames_start = _tx(seed=5, callsign="N0CALL")
    # Whole transmission in the buffer, but only the tail is searched --
    # forces a large negative L_lo.
    search_s = ((len(x) - 60 * FRAME_SAMPLES) / FS, len(x) / FS)
    r = modem.demodulate_blind(x, search_s=search_s)
    assert r.beacon is not None and r.frame0_start is not None
    err = abs(r.frame0_start - frames_start)
    assert err < FRAME_SAMPLES // 2, (
        f"frame0_start off by {err} samples ({err / FRAME_SAMPLES:.2f} frames); "
        f"got {r.frame0_start}, true frame 0 at {frames_start}"
    )


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


def test_blind_accumulator_chunking_is_invariant():
    """The whole point of BlindAccumulator is that a caller can feed it
    audio in whatever pieces arrive off the ring buffer -- so the
    accumulated result must not depend on how the same total signal was
    sliced into push() calls. Uses ragged, mutually-prime-ish chunk
    sizes deliberately, so a bug that only shows up when a chunk
    boundary lands mid-block (rather than conveniently on a block
    boundary) has somewhere to hide."""
    _, _, x, frames_start = _tx(seed=6)
    win = _frames_slice(x, frames_start, 100, beacon.MIN_FRAMES_FOR_SYNC + 20)
    z = to_baseband(win)

    whole = BlindAccumulator()
    whole.push(z, 0)

    chunked = BlindAccumulator()
    chunk_sizes = [4001, 1500, 9999, 2000, 12345]
    pos = 0
    i = 0
    while pos < len(z):
        n = min(chunk_sizes[i % len(chunk_sizes)], len(z) - pos)
        chunked.push(z[pos : pos + n], pos)
        pos += n
        i += 1

    # Internal-state check: precise, but only meaningful against the
    # Python reference -- the native adapter wraps an opaque C++ object
    # with no equivalent attributes, and pytest --native substitutes
    # BlindAccumulator with that adapter.
    if hasattr(whole, "_folded"):
        np.testing.assert_allclose(chunked._folded, whole._folded, rtol=1e-9, atol=1e-6)
        assert chunked._n_valid == whole._n_valid

    r_whole = whole.result()
    r_chunked = chunked.result()
    assert r_chunked.frame_start == r_whole.frame_start
    assert r_chunked.freq_offset == pytest.approx(r_whole.freq_offset)
    assert r_chunked.metric == pytest.approx(r_whole.metric, rel=1e-6)


def test_blind_accumulator_matches_acquire_blind_one_shot():
    """Cross-checks the streaming, block-decomposed accumulator against
    the existing one-shot acquire_blind (one huge FFT over the whole
    window) on the same signal. The two use different-sized FFTs and so
    a different, independently-rounded grid of candidate CFO bins --
    exact frequency/score agreement isn't expected -- but they must
    agree on the thing that actually matters: which pilot phase wins.

    window_s=None: acquire_blind weights every period equally, so this
    is a check on the block-decomposition math, not on the (separate,
    separately tested) decay feature."""
    _, _, x, frames_start = _tx(seed=7)
    win = _frames_slice(x, frames_start, 300, beacon.MIN_FRAMES_FOR_SYNC + 5)
    z = to_baseband(win)

    one_shot = acquire_blind(z)

    acc = BlindAccumulator(window_s=None)
    acc.push(z, 0)
    streamed = acc.result()

    assert streamed.frame_start == one_shot.frame_start
    assert abs(streamed.freq_offset - one_shot.freq_offset) < 2.0
    assert streamed.metric > 4.0


def test_blind_accumulator_decay_forgets_stale_history():
    """Without decay, a real (but short) transmission preceded by a long
    stretch of unrelated noise gets diluted: the peak bin's fold sums
    matched-filter power over *every* period pushed, real signal or not,
    so the score (peak / median) drifts toward 1 as the irrelevant
    fraction of history grows -- measured (not asserted here, to keep
    this test fast): 30 s of noise prefix barely dents it (9.3 vs. an
    undiluted ~18), 90 s roughly halves it to right at the default
    threshold (4.6), and 180 s pushes it below threshold entirely (3.0,
    a SyncError) even though the same signal alone locks cleanly. A
    bounded-window search of just the recent audio wouldn't see any of
    that dilution, which is the whole reason acquire_blind's caller
    bounds its search window in the first place.

    Rather than reproduce the slow (~180 s of noise) threshold-crossing
    case, compare metrics directly: decay should recover most of the
    undiluted score regardless of how much stale noise came before it."""
    rng = np.random.default_rng(9)
    _, _, x, frames_start = _tx(seed=9)
    win = _frames_slice(x, frames_start, 300, beacon.MIN_FRAMES_FOR_SYNC + 5)
    z_signal = to_baseband(win)

    noise_s = 90.0
    z_noise = rng.normal(size=int(noise_s * FS)) + 1j * rng.normal(size=int(noise_s * FS))

    undecayed = BlindAccumulator(window_s=None)
    undecayed.push(z_noise, 0)
    undecayed.push(z_signal, z_noise.size)
    undiluted_metric = undecayed.result().metric

    decayed = BlindAccumulator(window_s=10.0)
    decayed.push(z_noise, 0)
    decayed.push(z_signal, z_noise.size)
    decayed_metric = decayed.result().metric

    assert decayed_metric > 2.0 * undiluted_metric

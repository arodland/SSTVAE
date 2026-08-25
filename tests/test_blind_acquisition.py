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
    BLIND_SCORE_THRESHOLD,
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


def test_demodulate_blind_does_not_trust_silence_as_signal():
    """demodulate_blind's per-latent confidence weight comes from how
    each frame's pilot response compares to a "typical" |h| — but unlike
    demodulate() (which only ever looks at the header's known real frame
    count), the blind range is whatever the whole buffer holds, and a
    short real transmission sitting in a lot of leading/trailing silence
    or low-level noise made that "typical" value describe the noise
    floor instead of the signal: a plain median() over the whole range
    is dominated by however many silent frames surround the real ones,
    so noise reads back as fully trustworthy (weight ~1) right alongside
    the real signal, feeding reconstruct() latents that are mostly
    garbage at full confidence -- the reported symptom was a
    near-black decoded image when acquisition locked onto a buffer that
    was mostly pre- or post-transmission silence.

    Regression: real transmission is a small fraction of a much longer
    buffer of low-level ambient noise (not exact digital silence, which
    is the more realistic case and also exercises the >0 floor). Every
    canonical latent slot the real transmission's own frames could not
    have written to must come back at ~0 weight.
    """
    from sstvae.modem import framing

    mode = "A"
    modem = Modem()
    lat = np.random.default_rng(3).normal(size=MODES[mode].n_latents)
    lat /= np.sqrt(np.mean(lat**2))
    x = modem.modulate(lat, mode, callsign="TEST")
    frames_start = LEADIN_SAMPLES + PREAMBLE_SAMPLES + HEADER_SAMPLES
    sig = x[frames_start:]
    n_real_frames = MODES[mode].n_frames

    rng = np.random.default_rng(1)
    pre = rng.normal(scale=0.01, size=int(100.0 * FS))
    audio = np.concatenate([pre, sig])

    r = modem.demodulate_blind(audio)
    assert r.beacon is not None

    real_indices = np.zeros(MODES["C"].n_latents, dtype=bool)
    for abs_frame in range(n_real_frames):
        _, idx = framing.slot_range_for_frame(abs_frame)
        real_indices[idx] = True

    spurious = r.weights[~real_indices]
    assert np.all(spurious < 0.3), (
        f"{np.count_nonzero(spurious >= 0.3)} canonical latent slots outside "
        "the real transmission's own frames came back with high confidence "
        f"(max spurious weight {spurious.max():.3f}) -- silence is being "
        "trusted as signal"
    )


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
    assert streamed.metric > BLIND_SCORE_THRESHOLD


def test_blind_accumulator_result_origin_rebases_the_phase():
    """The fold lives in absolute (push start_sample) coordinates, but
    the caller uses the phase against its own buffer -- result(origin)
    is the bridge. The two coordinates agree only while the buffer
    starts at 0 mod FRAME_SAMPLES, which held in every test (sessions
    shorter than the ring buffer, buf_start pinned at 0) and silently
    stopped holding on real hardware once a listening session outlived
    the ring: the demod grid was then off by buf_start mod
    FRAME_SAMPLES, so blind acquisition locked with a healthy score
    while the pilot -- and with it the beacon -- read garbage. "Blind
    RX works for the first couple of minutes of a session, then almost
    never" was this."""
    from sstvae.config import FRAME_SAMPLES
    from sstvae.modem.dsp import to_baseband_at

    _, _, x, frames_start = _tx(seed=9)
    win = _frames_slice(x, frames_start, 300, beacon.MIN_FRAMES_FOR_SYNC + 5)

    # The engine's situation after the ring wraps: the buffer's first
    # sample sits at an absolute position that is NOT 0 mod
    # FRAME_SAMPLES.
    buf_start = 7 * FRAME_SAMPLES + 500

    ref = BlindAccumulator()
    ref.push(to_baseband(win), 0)
    r_ref = ref.result()

    acc = BlindAccumulator()
    acc.push(to_baseband_at(win, buf_start), buf_start)
    r_abs = acc.result()                  # absolute-coordinate phase
    r_rel = acc.result(origin=buf_start)  # what the engine must use

    # Rebased, the phase points at the same pilot the zero-based
    # reference found; unrebased it is off by exactly buf_start's
    # residue -- the bug's magnitude, pinned so a silent revert to the
    # old behaviour cannot look like a harmless refactor.
    assert r_rel.frame_start == r_ref.frame_start
    assert (r_abs.frame_start - r_rel.frame_start) % FRAME_SAMPLES == \
        buf_start % FRAME_SAMPLES
    assert r_rel.metric == pytest.approx(r_ref.metric, rel=1e-6)


def test_blind_accumulator_best_score_is_observable_below_threshold():
    """best_score() is the live loop's diagnostic: the same prominence
    result() gates on, visible whether or not it clears the threshold.
    Above threshold the two must agree exactly (same statistic, same
    fold); below it, where result() raises and the loop used to go
    silent, best_score() must still report the number -- that silence
    is what made a field receiver's failed acquisitions unfalsifiable."""
    _, _, x, frames_start = _tx(seed=8)
    win = _frames_slice(x, frames_start, 300, beacon.MIN_FRAMES_FOR_SYNC + 5)

    acc = BlindAccumulator()
    acc.push(to_baseband(win), 0)
    assert acc.best_score() == pytest.approx(acc.result().metric, rel=1e-9)

    noise = BlindAccumulator()
    noise.push(to_baseband(np.random.default_rng(0).normal(size=len(win))), 0)
    with pytest.raises(SyncError):
        noise.result()
    score = noise.best_score()
    assert 0.0 < score < BLIND_SCORE_THRESHOLD

    # Too little pushed to say anything at all: 0.0, not an exception.
    assert BlindAccumulator().best_score() == 0.0


def test_blind_accumulator_decay_forgets_stale_history():
    """Without decay, a real (but short) transmission preceded by a long
    stretch of unrelated noise gets diluted: the peak bin's fold sums
    matched-filter power over *every* period pushed, real signal or not,
    so the score (peak / median) drifts toward 1 as the irrelevant
    fraction of history grows -- measured (not asserted here, to keep
    this test fast, and re-measured for the PROTOCOL_VERSION 3 pilot):
    against an undiluted 38.8, a 30 s noise prefix gives 14.1, 90 s
    gives 6.7 and 180 s gives 4.1 -- the last two below
    BLIND_SCORE_THRESHOLD, i.e. a SyncError, even though the same signal
    alone locks cleanly. A
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

    # threshold=0 on both: this compares score *magnitudes*, and at 90 s
    # the undecayed score is legitimately below the gate -- which is the
    # dilution being demonstrated, not a failure to reproduce it.
    undecayed = BlindAccumulator(window_s=None, threshold=0.0)
    undecayed.push(z_noise, 0)
    undecayed.push(z_signal, z_noise.size)
    undiluted_metric = undecayed.result().metric

    decayed = BlindAccumulator(window_s=10.0, threshold=0.0)
    decayed.push(z_noise, 0)
    decayed.push(z_signal, z_noise.size)
    decayed_metric = decayed.result().metric

    assert decayed_metric > 2.0 * undiluted_metric


def test_blind_accumulator_multi_timescale_picks_the_better_one():
    """A single decay constant can't serve every mode well: short enough
    to not dilute a short transmission behind unrelated history means a
    long transmission's own real duration doesn't get folded in fully,
    and vice versa. window_s accepts several timescales run in parallel
    off the same (shared, expensive) per-block matched-filter result, and
    result() reports whichever scores best -- so it should never do worse
    than the single timescale that happens to suit the situation, on
    either a short-signal-behind-noise case (short timescale wins) or a
    long-signal case (long timescale wins, since it gets more of the
    real transmission's own duration folded in)."""
    rng = np.random.default_rng(11)

    def noisy_prefix(seconds):
        n = int(seconds * FS)
        return rng.normal(size=n) + 1j * rng.normal(size=n)

    # Case 1: short real signal behind a long stretch of unrelated noise
    # -- the short timescale alone should win, and multi-scale should
    # match it (not get dragged down by the long timescale's dilution).
    _, _, x, frames_start = _tx(seed=12)
    win = _frames_slice(x, frames_start, 300, beacon.MIN_FRAMES_FOR_SYNC + 5)
    z_short_signal = to_baseband(win)
    z_noise = noisy_prefix(90.0)

    short_only = BlindAccumulator(window_s=10.0)
    long_only = BlindAccumulator(window_s=90.0)
    multi = BlindAccumulator(window_s=[10.0, 90.0])
    for acc in (short_only, long_only, multi):
        acc.push(z_noise, 0)
        acc.push(z_short_signal, z_noise.size)

    short_metric = short_only.result().metric
    long_metric = long_only.result().metric
    multi_metric = multi.result().metric
    assert short_metric > long_metric  # the dilution effect this case is built to show
    assert multi_metric == pytest.approx(short_metric, rel=1e-9)

    # Case 2: a long, continuous, genuinely marginal signal, where the
    # long timescale gets more of the transmission's own duration folded
    # in and catches what the short one misses.
    #
    # **Slow fading, not fast.** This case used to be an `mpp` seed hunt,
    # and after the PROTOCOL_VERSION 3 pilot it stopped reproducing there
    # at any SNR from -1 to -18 dB -- with a pilot that survives the
    # clipper, 10 s already builds a strong enough peak that mpp's 1 Hz
    # Doppler decorrelates the signal faster than 90 s of integration can
    # accumulate it, so the short window ties or wins everywhere. The
    # long-window advantage lives in *slow* fading (mpg, 0.1 Hz), where
    # the signal stays coherent long enough for the extra duration to be
    # worth folding in -- which is a mechanism rather than a lucky seed.
    from sstvae import hfchannel
    from sstvae.config import MODES

    modem = Modem()
    lat = np.random.default_rng(4).normal(size=MODES["C"].n_latents)
    lat /= np.sqrt(np.mean(lat**2))
    tx_wave = modem.modulate(lat, "C", callsign="N0CALL")
    clean = tx_wave[frames_start:]  # the whole mode C frame region, ~95 s
    noisy = hfchannel.apply_channel(clean, snr_db=-5.0, fading_preset="mpg", seed=2)
    z_long_signal = to_baseband(noisy)

    short_only2 = BlindAccumulator(window_s=10.0)
    long_only2 = BlindAccumulator(window_s=90.0)
    multi2 = BlindAccumulator(window_s=[10.0, 90.0])
    for acc in (short_only2, long_only2, multi2):
        acc.push(z_long_signal, 0)

    with pytest.raises(SyncError):
        short_only2.result()
    assert long_only2.result().metric > BLIND_SCORE_THRESHOLD
    assert multi2.result().metric > BLIND_SCORE_THRESHOLD

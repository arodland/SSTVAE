"""The two reception numbers: how far the transmission got, and what
decoded.

The progress bar is `frames elapsed since the transmission's first
frame / frames expected` (`_frames_elapsed`) -- pure arithmetic on
buffer positions, so it climbs with the clock whatever the decoder
manages, starts part-way up on a late join, and can always reach 100%.
Beside it, and never as it, is `frames_decoded`: how many frames
carried confident data. That one is a *fill*, and a fill reads as a
completion percentage without being one, because the erasures the blind
path lives with (a fade, or simply not having heard the start) hold it
down permanently.

The third number is the one nothing may be substituted for: the
confident-latent *count* is what the --end-grace stall clock watches.
It is neither of the two above on purpose -- a position does not move
when retrospective decoding fills in frames behind it, and anything
positional climbs on buffer growth alone, so a reception fed either one
would either never stall or never stop stalling.
"""

import numpy as np

from sstvae.config import (
    DROPPED_LATENTS_PER_GROUP,
    FRAME_SAMPLES,
    HEADER_SAMPLES,
    LATENT_GROUPS,
    MODES,
    PREAMBLE_SAMPLES,
)
from sstvae.modem import framing
from sstvae.rx.engine import (
    PROGRESS_WEIGHT_THRESHOLD,
    _decode_progress,
    _frames_elapsed,
)

TOTAL_FRAMES = MODES["C"].n_frames
# Where a transmission starting at absolute 0 puts its first frame.
FRAMES_START = PREAMBLE_SAMPLES + HEADER_SAMPLES


def _weights_for_frames(frames, weight=1.0):
    """Canonical weight vector with exactly `frames` received."""
    w = np.zeros(MODES["C"].n_latents)
    for f in frames:
        _, idx = framing.slot_range_for_frame(f)
        w[idx] = weight
    return w


def test_frame_of_latent_inverts_slot_range_for_frame():
    table = framing.frame_of_latent()
    assert table.shape == (MODES["C"].n_latents,)
    for f in (0, 1, 219, 220, 437, 659):
        _, idx = framing.slot_range_for_frame(f)
        assert np.all(table[idx] == f)


def test_frame_of_latent_marks_exactly_the_dropped_latents():
    # The latents each group never gets a slot for (the beacon carrier's
    # capacity cost) belong to no frame at all. Getting this wrong by
    # defaulting to 0 instead of -1 would silently claim frame 0 for
    # thousands of latents that are never transmitted.
    table = framing.frame_of_latent()
    assert np.count_nonzero(table < 0) == LATENT_GROUPS * DROPPED_LATENTS_PER_GROUP


def test_no_confident_latents_is_nothing_decoded():
    assert _decode_progress(np.zeros(MODES["C"].n_latents)) == (0, 0)


def test_weights_at_the_threshold_do_not_count():
    w = _weights_for_frames([0], weight=PROGRESS_WEIGHT_THRESHOLD)
    assert _decode_progress(w) == (0, 0)


def test_frames_decoded_counts_frames_not_latents():
    # Every other frame of mode B's range: half the frames, and a frame
    # counts once whatever it carries. The interleaver is why the two
    # answers differ at all -- each frame's latents are scattered across
    # the whole picture.
    frames = list(range(0, MODES["B"].n_frames, 2))
    w = _weights_for_frames(frames)
    metric, decoded = _decode_progress(w)
    assert decoded == len(frames)
    assert metric == np.count_nonzero(w > PROGRESS_WEIGHT_THRESHOLD)
    assert metric > decoded  # latents per frame, not the same question


def test_the_stall_metric_is_the_latent_count_and_moves_on_backfill():
    # The two properties that make the count the only usable stall
    # signal, and that the display numbers do not have.
    reached = _weights_for_frames([0, 300])
    backfilled = _weights_for_frames(range(0, 301))

    metric_reached, decoded_reached = _decode_progress(reached)
    metric_backfilled, decoded_backfilled = _decode_progress(backfilled)

    assert metric_reached == np.count_nonzero(reached > PROGRESS_WEIGHT_THRESHOLD)
    # Retrospective decoding filling in frames *behind* the furthest one
    # is real progress, and the stall clock has to see it as progress or
    # it ends a reception that is still improving.
    assert metric_backfilled > metric_reached
    assert decoded_backfilled > decoded_reached


def test_the_bar_counts_frames_that_have_gone_by():
    n = MODES["A"].n_frames
    end = FRAMES_START + n * FRAME_SAMPLES

    assert _frames_elapsed(0, 0, n) == 0
    assert _frames_elapsed(0, FRAMES_START, n) == 0
    assert _frames_elapsed(0, FRAMES_START + FRAME_SAMPLES, n) == 1
    assert _frames_elapsed(0, end, n) == n
    # Audio past the transmission's end is not more of it.
    assert _frames_elapsed(0, end + 100 * FRAME_SAMPLES, n) == n


def test_the_bar_climbs_with_the_clock_and_nothing_else():
    # The property the whole change is for: no weights are involved, so
    # a fade cannot hold it back the way it holds back frames_decoded.
    n = MODES["A"].n_frames
    got = [
        _frames_elapsed(0, FRAMES_START + k * FRAME_SAMPLES, n)
        for k in [*range(0, n, 7), n]
    ]
    assert got == sorted(got)
    assert got[0] == 0 and got[-1] == n


def test_a_late_join_starts_part_way_up_and_still_reaches_the_top():
    # Tuned in at frame 400 of mode C: those frames' audio is gone for
    # good, so the bar starts at 400/660 -- the transmission is 400
    # frames into its schedule the moment we join -- and fills to
    # 660/660 as the rest arrives. The frames the join missed appear as
    # the gap against frames_decoded, exactly like a fade's. Counting
    # only captured audio instead starts the bar at zero and caps it at
    # 260/660 forever, on the display whose job is to say how far along
    # the transmission is.
    n = TOTAL_FRAMES
    joined = FRAMES_START + 400 * FRAME_SAMPLES
    assert _frames_elapsed(0, joined, n) == 400
    end = FRAMES_START + n * FRAME_SAMPLES
    assert _frames_elapsed(0, end, n) == n


def test_a_transmission_that_starts_after_the_buffer_does():
    # The ordinary case: the preamble arrived while we were listening,
    # so `start` sits inside the buffer and nothing is missing.
    n = MODES["A"].n_frames
    start = 5 * FRAME_SAMPLES
    total = start + FRAMES_START + n * FRAME_SAMPLES
    assert _frames_elapsed(start, total, n) == n
    # And before that transmission's first frame, nothing of it yet.
    assert _frames_elapsed(start, start + FRAMES_START, n) == 0


def test_dropped_latent_slots_never_count_as_a_decoded_frame():
    # frame_of_latent() is -1 for the never-transmitted dropped slots,
    # and numpy indexing would wrap -1 to the *last* frame -- a phantom
    # "frame 659 decoded" from a latent that never went on the air. The
    # C++ guards f >= 0; this pins the Python mirror to it. The metric
    # still counts every confident latent, matching the C++ exactly.
    table = framing.frame_of_latent()
    w = np.zeros(MODES["C"].n_latents)
    w[table < 0] = 0.9
    metric, decoded = _decode_progress(w)
    assert decoded == 0
    assert metric == np.count_nonzero(table < 0)

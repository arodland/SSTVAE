"""The reception progress bar: position, not fill.

The bar is `last frame successfully received / frames expected`, never
`latents received / latents expected`. The blind path is the one where
the two differ -- it decodes whatever frames it can place, with holes
where the signal faded or where the transmission started before the
buffer did -- and a fill fraction there is a completion percentage that
is not one: a reception already at the transmission's last frame reports
70% and never fills.
"""

import numpy as np

from sstvae.config import (
    DROPPED_LATENTS_PER_GROUP,
    LATENT_GROUPS,
    MODES,
)
from sstvae.modem import framing
from sstvae.rx.engine import PROGRESS_WEIGHT_THRESHOLD, _blind_progress

TOTAL_FRAMES = MODES["C"].n_frames


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


def test_no_confident_latents_is_zero_progress():
    assert _blind_progress(np.zeros(MODES["C"].n_latents), TOTAL_FRAMES) == (0, 0.0, 0)


def test_weights_at_the_threshold_do_not_count():
    w = _weights_for_frames([0], weight=PROGRESS_WEIGHT_THRESHOLD)
    assert _blind_progress(w, TOTAL_FRAMES) == (0, 0.0, 0)


def test_progress_is_the_last_frame_reached_not_the_count():
    # Every other frame of mode B's range: half the latents, but the
    # transmission has been followed all the way to its end. The old
    # count-based fraction reported ~50% here.
    reach = MODES["B"].n_frames
    w = _weights_for_frames([*range(0, reach, 2), reach - 1])
    metric, frac, got_reach = _blind_progress(w, TOTAL_FRAMES)
    assert frac == reach / TOTAL_FRAMES
    # The reach is the fraction's numerator, reported as the blind
    # path's frames_received so the status line's frame counter and its
    # percentage advance together instead of one freezing.
    assert got_reach == reach
    # The stall metric is still the count -- a different question, and
    # the one --end-grace watches.
    assert metric == np.count_nonzero(w > PROGRESS_WEIGHT_THRESHOLD)


def test_a_known_mode_makes_the_bar_fill_at_that_modes_end():
    # The beacon's mode field (PROTOCOL_VERSION 4) gives the blind path
    # the real frame count: a complete mode B reception reads 100%, not
    # the 2/3 that dividing by mode C's count reported before the field
    # existed.
    reach = MODES["B"].n_frames
    w = _weights_for_frames([*range(0, reach, 2), reach - 1])
    _, frac, _ = _blind_progress(w, reach)
    assert frac == 1.0


def test_a_late_join_reports_where_it_is_not_how_much_it_has():
    # Tuned in at frame 400 of mode C and heard the rest: two thirds of
    # the latents are gone for good, and the bar must still read 100%.
    w = _weights_for_frames(range(400, TOTAL_FRAMES))
    _, frac, _ = _blind_progress(w, TOTAL_FRAMES)
    assert frac == 1.0


def test_progress_advances_with_reach_and_ignores_backfill():
    reached = _weights_for_frames([0, 300])
    _, frac_reached, reach_reached = _blind_progress(reached, TOTAL_FRAMES)
    assert frac_reached == 301 / TOTAL_FRAMES
    assert reach_reached == 301

    # Retrospective decoding filling in frames *behind* the furthest one
    # is real progress in quality but not in position, and the bar must
    # not jump backwards or forwards for it.
    backfilled = _weights_for_frames(range(0, 301))
    _, frac_backfilled, reach_backfilled = _blind_progress(backfilled, TOTAL_FRAMES)
    assert frac_backfilled == frac_reached
    assert reach_backfilled == reach_reached

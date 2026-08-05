"""The preamble's detection margin, and the coupling that creates it.

The live receiver's false-lock problem was never that the threshold was
set wrong -- it was that the lag-M metric's *noise floor* sat right
underneath it, because the correlation window was one symbol long. The
fix was to lengthen the preamble and widen the window together, and the
half of that pair which can be undone silently is the window: a longer
preamble read through a one-symbol window detects exactly as badly as
before, with every other test still passing.

So these tests are about the coupling, not about the numbers. They are
deliberately not statistical -- see the note in CLAUDE.md about not
asserting that noise decodes to nothing, which is precisely the shape
of assertion a margin test invites and which a seeded sweep has already
been shown to falsify every few seed-minutes.
"""

import numpy as np
import pytest

from sstvae.config import (
    M,
    PREAMBLE_CORR_WINDOW,
    PREAMBLE_CP,
    PREAMBLE_REPEATS,
    PREAMBLE_SAMPLES,
    PREAMBLE_THRESHOLD,
)
from sstvae.modem import ofdm, sync


def test_the_window_spans_every_repeat_but_one():
    """With R repeats there are exactly (R-1)*M sample pairs M apart that
    both land inside the preamble. Integrating fewer wastes the length;
    integrating more reaches past the end into the header."""
    assert PREAMBLE_CORR_WINDOW == (PREAMBLE_REPEATS - 1) * M
    assert PREAMBLE_SAMPLES == PREAMBLE_CP + PREAMBLE_REPEATS * M


def test_the_detector_actually_integrates_that_window():
    """The constant above is inert unless `_autocorr_metric` uses it.

    Checked through the output length rather than by reading the source:
    a lag-M correlation over a W-sample window turns n samples into
    n - M - W + 1 positions, so a stale `kernel = np.ones(M)` shows up
    here as a longer array. This is the assertion that fails if someone
    lengthens the preamble and forgets the window.
    """
    z = np.exp(1j * np.linspace(0, 7.0, 5000))
    metric, a = sync._autocorr_metric(z)
    assert len(metric) == len(z) - M - PREAMBLE_CORR_WINDOW + 1
    assert len(a) == len(metric)


def test_the_preamble_is_periodic_with_M_throughout():
    """Every repeat must be the same symbol, including the double-length
    CP, or the four demod windows the receiver averages for its channel
    reference are not averaging copies of one thing -- and backing
    DEMOD_BACKOFF samples into the previous repeat stops being free."""
    w = ofdm.preamble_waveform()
    assert len(w) == PREAMBLE_SAMPLES
    body = w[PREAMBLE_CP:]
    first = body[:M]
    for r in range(1, PREAMBLE_REPEATS):
        np.testing.assert_allclose(body[r * M : (r + 1) * M], first, atol=1e-9)
    # The CP is the tail of the symbol, which is what makes the whole
    # block periodic rather than merely repetitive.
    np.testing.assert_allclose(w[:PREAMBLE_CP], first[-PREAMBLE_CP:], atol=1e-9)


def test_a_clean_preamble_scores_far_above_threshold():
    """The margin has to exist in the direction that matters too: a real
    preamble with no noise at all must not be near the threshold."""
    z = ofdm.preamble_template()
    pad = np.zeros(M, dtype=complex)
    metric, _ = sync._autocorr_metric(np.concatenate([z, pad]))
    assert metric.max() > 0.99
    assert PREAMBLE_THRESHOLD < 0.9 * metric.max()


@pytest.mark.parametrize("offset_hz", [0.0, 12.5, -37.5])
def test_acquire_finds_a_noiseless_preamble_exactly(offset_hz):
    """End to end through the shipped `acquire`, including the template
    correlation and the CFO refinement that now averages over every
    repeat rather than one pair."""
    from sstvae.hfchannel import freq_shift
    from sstvae.modem.dsp import to_baseband

    lead = 500
    x = np.concatenate([np.zeros(lead), ofdm.preamble_waveform(), np.zeros(2 * M)])
    if offset_hz:
        x = freq_shift(x, offset_hz)
    acq = sync.acquire(to_baseband(x))
    assert abs(acq.preamble_start - lead) <= 1
    assert abs(acq.freq_offset - offset_hz) < 0.5
    assert acq.metric > PREAMBLE_THRESHOLD

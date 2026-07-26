"""RingBuffer, in particular the accessors the display added."""

import numpy as np

from sstvae.rx import RingBuffer


def test_snapshot_before_wrapping_returns_only_what_was_written():
    r = RingBuffer(1.0, fs=100)
    r.write(np.arange(10, dtype=float))
    data, total = r.snapshot()
    assert total == 10
    assert np.array_equal(data, np.arange(10))


def test_snapshot_after_wrapping_is_chronological():
    r = RingBuffer(1.0, fs=100)  # 100 samples
    r.write(np.arange(150, dtype=float))
    data, total = r.snapshot()
    assert total == 150
    assert len(data) == 100
    assert np.array_equal(data, np.arange(50, 150))


def test_tail_returns_the_most_recent_samples():
    r = RingBuffer(1.0, fs=100)
    r.write(np.arange(60, dtype=float))
    assert np.array_equal(r.tail(5), np.arange(55, 60))


def test_tail_across_the_wrap_point():
    r = RingBuffer(1.0, fs=100)
    r.write(np.arange(140, dtype=float))  # write_pos == 40
    assert np.array_equal(r.tail(60), np.arange(80, 140))


def test_tail_is_clamped_to_what_exists():
    r = RingBuffer(1.0, fs=100)
    assert len(r.tail(50)) == 0
    r.write(np.arange(7, dtype=float))
    assert np.array_equal(r.tail(50), np.arange(7))


def test_tail_never_exceeds_the_buffer():
    r = RingBuffer(1.0, fs=100)
    r.write(np.arange(500, dtype=float))
    assert len(r.tail(10_000)) == 100


def test_clear_drops_audio_but_keeps_absolute_positions():
    """The decode loop records reception positions in `total_written`
    coordinates; clearing after a transmission must not invalidate
    them."""
    r = RingBuffer(1.0, fs=100)
    r.write(np.arange(250, dtype=float))
    before = r.total_written
    r.clear()
    data, total = r.snapshot()
    assert total == before
    assert not np.any(data)
    # And the buffer keeps working afterwards.
    r.write(np.ones(10))
    assert np.array_equal(r.tail(10), np.ones(10))


def test_a_write_longer_than_the_buffer_keeps_the_newest_part():
    r = RingBuffer(1.0, fs=100)
    r.write(np.arange(250, dtype=float))
    data, _ = r.snapshot()
    assert np.array_equal(data, np.arange(150, 250))

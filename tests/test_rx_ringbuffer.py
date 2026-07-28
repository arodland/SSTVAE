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


# --- the audio callback must never block ---------------------------------

def test_snapshot_does_not_stall_the_writer():
    """The regression that cost 5 dB of on-air SNR.

    `snapshot` used to hold the lock while copying the whole buffer.
    The decode loop calls it every poll_interval, so `write` -- which
    runs in the PortAudio callback -- blocked for the length of an 8 MB
    copy, several times a minute. PortAudio's answer to a callback that
    does not return promptly is to discard input, so the receiver
    punched a hole in its own audio every 5 seconds, growing as the
    buffer filled.

    Self-calibrating rather than a fixed millisecond bound: measure what
    a full copy costs on this machine, then require that a concurrent
    write costs a small fraction of it.

    Judged on the 95th percentile, not the maximum. The old code blocked
    writes *systematically* -- the reader holds the lock nearly
    continuously, so almost every write waits -- whereas a lone
    scheduling hiccup on a loaded machine can inflate a maximum without
    meaning anything. Measured on the old code the p95 was ~700x the
    copy cost, so there is no ambiguity to split.
    """
    import threading
    import time

    ring = RingBuffer(130.0)
    chunk = np.random.default_rng(0).standard_normal(400)
    for _ in range(2000):          # fill it enough that a copy is expensive
        ring.write(chunk)

    t = time.perf_counter()
    for _ in range(5):
        ring.snapshot()
    copy_cost = (time.perf_counter() - t) / 5
    assert copy_cost > 0

    stop = threading.Event()

    def reader():
        while not stop.is_set():
            ring.snapshot()

    th = threading.Thread(target=reader, daemon=True)
    th.start()
    try:
        times = []
        for _ in range(300):
            t = time.perf_counter()
            ring.write(chunk)
            times.append(time.perf_counter() - t)
    finally:
        stop.set()
        th.join(timeout=5)

    p95 = float(np.percentile(times, 95))
    assert p95 < copy_cost / 4, (
        f"write p95 was {p95 * 1e3:.2f} ms against a {copy_cost * 1e3:.2f} ms "
        "snapshot -- the audio callback is being stalled by the decode loop again"
    )


def test_concurrent_snapshots_still_return_consistent_audio():
    """Copying outside the lock must not corrupt what a reader sees.

    Writers only ever advance into the region a reader has already
    passed, so a snapshot taken while writing is still a valid stretch
    of audio -- verified here on a ramp, where any tear or reordering
    shows up as a discontinuity.
    """
    import threading

    ring = RingBuffer(4.0)          # small, so it wraps constantly
    stop = threading.Event()
    bad = []

    def reader():
        while not stop.is_set():
            snap, total = ring.snapshot()
            if total >= ring.n and len(snap) != ring.n:
                bad.append(("length", len(snap), total))

    th = threading.Thread(target=reader, daemon=True)
    th.start()
    try:
        for i in range(500):
            ring.write(np.full(200, float(i)))
    finally:
        stop.set()
        th.join(timeout=5)

    assert not bad, bad[:3]
    snap, total = ring.snapshot()
    assert total == 500 * 200
    assert len(snap) == ring.n

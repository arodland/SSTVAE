"""The watchdog that stops a reception sitting in "receiving" forever.

These are fast tests: the modem is replaced by a stub, so nothing here
does any DSP at all. That is deliberate, and it is what the slow tests in
`test_listen_state_machine.py` cannot do — the failure under test is
*decodes stopping*, and there is no way to ask the real modem to stop
decoding on cue. The state machine's decisions are the thing being
checked, so the state machine is the only thing left in the loop.

The failure they pin: every completion test used to be evaluated inside
the branch that ran only when the current poll had produced a decode.
A reception stops producing a decode long before it stops being real —
its audio scrolls out of the ring buffer, or its blind acquisition score
falls back under BLIND_SCORE_THRESHOLD as the accumulator's evidence
decays once the transmission is over — and from that poll onward "is
this finished?" was simply never asked again. The loop stayed in
"receiving" indefinitely, the sink was never called, and the picture
that had already been decoded was never delivered. The reported hang and
"autosave never fires" are the same bug seen from two ends.

`decode_loop_diversity` is pinned here as well, and for the same reason
these tests exist at all rather than living in the slow suite: it is a
separate state machine (deliberately, so the single-receiver one stays
byte-for-byte what the slow tests were written against), so it can
regress on its own, and every existing diversity test does real DSP and
therefore cannot ask decodes to stop. It has two independent ways to
stop decoding rather than one, since a poll produces nothing whenever
*neither* branch locks.
"""

import threading
import time

import numpy as np
import pytest
from PIL import Image

from sstvae.config import FS, HEADER_SAMPLES, MODES, PREAMBLE_SAMPLES
from sstvae.modem import SyncError
from sstvae.modem.modem import DemodResult
from sstvae.modem.sync import Acquisition
from sstvae.rx import engine as rx_engine
from sstvae.rx.engine import (
    Reception,
    RxConfig,
    SharedState,
    decode_loop,
    decode_loop_diversity,
)

MODE_A = MODES["A"]


class _CollectingSink:
    def __init__(self):
        self.receptions: list[Reception] = []

    def on_reception(self, rec):
        self.receptions.append(rec)
        return "/dev/null/fake.png"


def _demod_result(frames_received: int) -> DemodResult:
    """A partial mode-A decode: the preamble and header were heard, and
    `frames_received` of the mode's frames landed."""
    n = MODE_A.n_latents
    return DemodResult(
        latents=np.zeros(n),
        weights=np.full(n, 0.9),
        mode=MODE_A,
        freq_offset=0.0,
        sync_metric=0.9,
        frames_received=frames_received,
        beacon=None,
        callsign="TEST",
        preamble_start=0,
        snr_db=12.0,
    )


class _StubModem:
    """Decodes for the first `good_polls` calls, then stops — the shape
    of every real way a reception's decodes end (its audio ages out of
    the ring buffer, its blind lock decays below threshold, the signal
    was never really there). Never reports a complete transmission, so
    the only thing that can finish it is the watchdog."""

    def __init__(self, good_polls: int, frames_received: int):
        self.good_polls = good_polls
        self.frames_received = frames_received
        self.calls = 0
        self.calls_after_stop = 0

    def demodulate(self, samples, search_s=None, drift_track="off"):
        self.calls += 1
        if self.calls > self.good_polls:
            self.calls_after_stop += 1
            raise SyncError("decodes have stopped")
        return _demod_result(self.frames_received)

    def demodulate_blind(self, x, search_s=None, acquisition=None, drift_track="off"):
        raise SyncError("no blind lock either")


@pytest.fixture
def stub_modem(monkeypatch):
    """Put the loop's whole signal-processing surface under control:
    `Modem`, the module-level `sync_acquire` that `_find_new_reception`
    vets peaks with, and `reconstruct` (so no checkpoint is needed).
    `to_baseband` is left real — it is a cheap pure function of the
    samples and faking it would only add a way for the test to lie."""

    def install(modem):
        monkeypatch.setattr(rx_engine, "Modem", lambda: modem)
        monkeypatch.setattr(
            rx_engine, "sync_acquire",
            lambda z, **kw: Acquisition(preamble_start=0, freq_offset=0.0, metric=0.9),
        )
        monkeypatch.setattr(
            rx_engine, "reconstruct",
            lambda model, latents, weights: Image.new("RGB", (8, 8)),
        )
        return modem

    return install


def _run(config, sink, seconds_of_audio=6.0, timeout_s=20.0, feed_more=None):
    """Run decode_loop against a ring buffer holding `seconds_of_audio`
    of noise, until the sink sees a reception or `timeout_s` elapses.

    The audio's *content* is irrelevant here — the stub modem ignores it
    — but its length is not: the loop refuses to attempt a decode below
    MIN_SECONDS_BEFORE_ATTEMPT, and the ring buffer's fill is what the
    sample-position deadline is measured against."""
    ring = rx_engine.RingBuffer(200.0)
    ring.write(np.zeros(int(seconds_of_audio * FS)))

    state = SharedState()
    stop = threading.Event()
    th = threading.Thread(
        target=decode_loop, args=(ring, None, state, config, stop, sink), daemon=True
    )
    th.start()
    try:
        if feed_more is not None:
            feed_more(ring)
        deadline = time.time() + timeout_s
        while time.time() < deadline and not sink.receptions and th.is_alive():
            time.sleep(0.02)
    finally:
        stop.set()
        th.join(timeout=10.0)
    return state


def _run_diversity(config, sink, seconds_of_audio=6.0, timeout_s=20.0, feed_more=None):
    """`_run`'s two-branch counterpart, for `decode_loop_diversity`.

    Both rings are fed identically: the branches are two receivers
    hearing one transmission, and the stub modem ignores the samples
    anyway, so what this exercises is the loop's own bookkeeping."""
    rings = [rx_engine.RingBuffer(200.0), rx_engine.RingBuffer(200.0)]
    for ring in rings:
        ring.write(np.zeros(int(seconds_of_audio * FS)))

    state = SharedState()
    stop = threading.Event()
    th = threading.Thread(
        target=decode_loop_diversity,
        args=(rings, None, state, config, stop, sink),
        daemon=True,
    )
    th.start()
    try:
        if feed_more is not None:
            feed_more(rings)
        deadline = time.time() + timeout_s
        while time.time() < deadline and not sink.receptions and th.is_alive():
            time.sleep(0.02)
    finally:
        stop.set()
        th.join(timeout=10.0)
    return state


def test_a_reception_whose_decodes_stop_is_still_delivered(stub_modem):
    """The core regression. A partial reception is decoded, then the
    decodes stop. Nothing else can finish it: it never reports all its
    frames, and the buffer never reaches its sample-position deadline.
    Only the stall test can, and only if it is still evaluated on polls
    that decoded nothing."""
    modem = stub_modem(_StubModem(good_polls=2, frames_received=MODE_A.n_frames // 2))
    sink = _CollectingSink()
    config = RxConfig(poll_interval=0.05, end_grace=0.5)

    state = _run(config, sink, timeout_s=20.0)

    assert modem.calls_after_stop > 0, (
        "the stub never got past its good polls — the test isn't exercising "
        "the failure at all"
    )
    assert len(sink.receptions) == 1, (
        "a reception whose decodes stopped was never handed to the sink: it is "
        "still sitting in 'receiving' with its picture undelivered, which is "
        "both the hang and the autosave-never-fires report"
    )
    rec = sink.receptions[0]
    assert rec.frames_received == MODE_A.n_frames // 2
    assert rec.mode_name == "A"
    assert rec.callsign == "TEST"
    assert state.status == "listening", (
        f"status stuck at {state.status!r} after the reception ended"
    )


def test_the_preamble_path_has_a_deadline_too(stub_modem):
    """A reception that goes on decoding, always partially, must still
    end once the buffer holds audio past the last point one of its
    frames could have arrived — mode A's own duration past its start.

    end_grace is set out of reach so this isolates the sample-position
    deadline from the stall test above. Before the fix the header path
    had no deadline of any kind: `done` was `frames_received >=
    n_frames_expected` and nothing else, so a transmission that was cut
    short, or faded before its end, was re-decoded to the same partial
    result on every poll forever."""
    modem = stub_modem(_StubModem(good_polls=10**9, frames_received=3))
    sink = _CollectingSink()
    config = RxConfig(poll_interval=0.05, end_grace=1e9)

    # The stub reports preamble_start 0, so the deadline sits one whole
    # mode-A transmission from the buffer's start. Start below it, then
    # feed past it, so a delivery can only be the deadline firing.
    need = PREAMBLE_SAMPLES + HEADER_SAMPLES + MODE_A.n_frames * rx_engine.FRAME_SAMPLES

    def feed_more(ring):
        time.sleep(0.5)
        assert not sink.receptions, (
            "finished before its deadline could be reached — end_grace=1e9 was "
            "supposed to make that impossible, so this isn't isolated"
        )
        ring.write(np.zeros(need))

    _run(config, sink, seconds_of_audio=6.0, timeout_s=20.0, feed_more=feed_more)

    assert len(sink.receptions) == 1, (
        "a partial header-path reception never ended: past its own mode's "
        "duration there is provably no more of it left to arrive"
    )
    assert sink.receptions[0].frames_received == 3


def test_a_complete_reception_still_finishes_on_its_frame_count(stub_modem):
    """The ordinary path, unchanged: all frames received ends it at
    once, without waiting on end_grace or on any deadline."""
    stub_modem(_StubModem(good_polls=10**9, frames_received=MODE_A.n_frames))
    sink = _CollectingSink()
    config = RxConfig(poll_interval=0.05, end_grace=1e9)

    t0 = time.time()
    _run(config, sink, timeout_s=20.0)

    assert len(sink.receptions) == 1
    assert time.time() - t0 < 5.0, "waited on a timer for an already-complete reception"
    assert sink.receptions[0].frames_received == MODE_A.n_frames


def test_nothing_is_delivered_when_nothing_ever_decoded(stub_modem):
    """The watchdog must not invent receptions. A loop that never
    decodes anything has nothing pending, so no deadline is running and
    the sink is never called."""
    stub_modem(_StubModem(good_polls=0, frames_received=0))
    sink = _CollectingSink()
    config = RxConfig(poll_interval=0.05, end_grace=0.2)

    state = _run(config, sink, timeout_s=3.0)

    assert sink.receptions == []
    assert state.status == "listening"


def test_a_diversity_reception_whose_decodes_stop_is_still_delivered(stub_modem):
    """The core regression again, against the two-branch loop.

    Both branches decode for a while and then neither does — which is
    how a two-receiver reception ordinarily ends, both antennas hearing
    the same transmission stop at the same time. Nothing else can finish
    it: the combine never reports all its frames, and the buffer never
    reaches the reception's sample-position deadline."""
    # Each poll decodes once per branch, so four good calls is two full
    # polls with both branches locked — enough to establish the
    # reception, after which no branch acquires on any poll.
    modem = stub_modem(_StubModem(good_polls=4, frames_received=MODE_A.n_frames // 2))
    sink = _CollectingSink()
    config = RxConfig(poll_interval=0.05, end_grace=0.5)

    state = _run_diversity(config, sink, timeout_s=20.0)

    assert modem.calls_after_stop > 0, (
        "the stub never got past its good polls — the test isn't exercising "
        "the failure at all"
    )
    assert len(sink.receptions) == 1, (
        "a diversity reception whose branches both stopped decoding was never "
        "handed to the sink: the loop is still in 'receiving' with its picture "
        "undelivered"
    )
    rec = sink.receptions[0]
    assert rec.frames_received == MODE_A.n_frames // 2
    assert rec.mode_name == "A"
    assert rec.callsign == "TEST"
    assert state.status == "listening", (
        f"status stuck at {state.status!r} after the reception ended"
    )


def test_the_diversity_path_has_a_deadline_too(stub_modem):
    """And the deadline, against the two-branch loop: a combine that
    goes on decoding the same partial result must still end once the
    buffer holds audio past the last point one of its frames could have
    arrived.

    end_grace is out of reach so this isolates the sample-position
    deadline from the stall test above."""
    stub_modem(_StubModem(good_polls=10**9, frames_received=3))
    sink = _CollectingSink()
    config = RxConfig(poll_interval=0.05, end_grace=1e9)

    need = PREAMBLE_SAMPLES + HEADER_SAMPLES + MODE_A.n_frames * rx_engine.FRAME_SAMPLES

    def feed_more(rings):
        time.sleep(0.5)
        assert not sink.receptions, (
            "finished before its deadline could be reached — end_grace=1e9 was "
            "supposed to make that impossible, so this isn't isolated"
        )
        for ring in rings:
            ring.write(np.zeros(need))

    _run_diversity(config, sink, seconds_of_audio=6.0, timeout_s=20.0,
                   feed_more=feed_more)

    assert len(sink.receptions) == 1, (
        "a partial two-branch reception never ended: past its own mode's "
        "duration there is provably no more of it left to arrive"
    )
    assert sink.receptions[0].frames_received == 3


def test_a_delivered_reception_carries_its_latents_and_start_time(stub_modem):
    """The sink gets the demodulator output itself, not only the picture.

    Decoding is lossy and throws the per-latent confidence away, so a
    station that wants its reception *combined* with another station's
    needs `result`. `utc_start` is what lets the two be recognized as
    the same transmission at all -- see docs/reception-aggregation.md.
    """
    modem = stub_modem(_StubModem(good_polls=2, frames_received=MODE_A.n_frames // 2))
    sink = _CollectingSink()
    config = RxConfig(poll_interval=0.05, end_grace=0.5)

    before = time.time()
    _run(config, sink, timeout_s=20.0)
    after = time.time()

    assert len(sink.receptions) == 1
    rec = sink.receptions[0]
    assert isinstance(rec.result, DemodResult), (
        "the sink was handed a picture with no demodulator output behind it, "
        "so the latents and their confidence weights are unrecoverable"
    )
    assert rec.result.frames_received == MODE_A.n_frames // 2
    assert len(rec.result.latents) == MODE_A.n_latents, (
        "the *unpadded* result is wanted: its arrays are the mode's own size"
    )
    # The audio was written just before the loop started, so the
    # reception's start dates to around then -- and inside the window
    # this test itself ran in, which a construction-time epoch on a
    # long-lived ring would not guarantee.
    assert rec.utc_start is not None
    assert before - 30.0 <= rec.utc_start <= after

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
from sstvae.rx.engine import Reception, RxConfig, SharedState, decode_loop

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


def test_blind_reach_at_the_last_frame_does_not_complete_the_reception(stub_modem, monkeypatch):
    """The blind path's frames_received is a *reach* (the furthest frame
    decoded), reported so the status line's frame counter advances in
    step with its percentage -- not a contiguous count. A reach at the
    transmission's last frame is exactly when retrospective backfill is
    still filling erasures behind it, so the header path's
    "frames_received >= n_frames_expected ends it now" test must not
    apply: the reception here reaches mode A's last frame on its very
    first decode and must still be ended by the stall clock (end_grace),
    not on the spot."""
    from sstvae.modem import framing
    from sstvae.modem.beacon import BeaconResult
    from sstvae.modem.modem import BlindDemodResult

    n = MODES["C"].n_latents
    weights = np.zeros(n)
    for f in range(MODE_A.n_frames):  # full reach, half the weights left 0
        _, idx = framing.slot_range_for_frame(f)
        weights[idx[::2]] = 0.9
    result = BlindDemodResult(
        latents=np.zeros(n), weights=weights, freq_offset=0.0,
        beacon=BeaconResult(chip_offset=0, frame_index=0, callsign="TEST",
                            mode_index=MODE_A.index),
        callsign="TEST", frame_offset=0, n_frames=MODE_A.n_frames,
        frame0_start=0, snr_db=10.0,
    )

    class _BlindOnlyModem:
        def demodulate(self, samples, search_s=None, drift_track="off"):
            raise SyncError("no preamble")

        def demodulate_blind(self, x, search_s=None, acquisition=None,
                             drift_track="off"):
            return result

    class _FakeAccumulator:
        def __init__(self, *a, **kw):
            pass

        def push(self, z, start_sample):
            pass

        def best_score(self):
            return 12.0  # anything; the loop only reports it

        def result(self, origin=0):
            return object()  # non-None: the loop only forwards it

    modem = stub_modem(_BlindOnlyModem())
    # The preamble vet must fail too, or _find_new_reception decodes.
    monkeypatch.setattr(
        rx_engine, "sync_acquire",
        lambda z, **kw: (_ for _ in ()).throw(SyncError("no preamble")),
    )
    monkeypatch.setattr(rx_engine, "BlindAccumulator", _FakeAccumulator)

    sink = _CollectingSink()
    config = RxConfig(poll_interval=0.05, end_grace=1.5)

    t0 = time.time()
    _run(config, sink, timeout_s=20.0)
    elapsed = time.time() - t0

    assert len(sink.receptions) == 1, "the stall clock should still end it"
    rec = sink.receptions[0]
    assert rec.mode_name == "A"
    assert rec.frames_received == MODE_A.n_frames  # the reach, on display
    assert rec.n_frames_expected == MODE_A.n_frames
    assert elapsed >= config.end_grace, (
        f"delivered after {elapsed:.2f}s < end_grace={config.end_grace}s: the "
        "reception was completed on its reach, cutting off retrospective backfill"
    )


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

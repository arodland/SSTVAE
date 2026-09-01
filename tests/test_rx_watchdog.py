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


def _demod_result(frames_received: int, preamble_start: int = 0) -> DemodResult:
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
        preamble_start=preamble_start,
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


def _run(config, sink, seconds_of_audio=6.0, timeout_s=20.0, feed_more=None,
         until=None):
    """Run decode_loop against a ring buffer holding `seconds_of_audio`
    of noise, until `until(state, sink)` holds -- by default, until the
    sink sees a reception -- or `timeout_s` elapses.

    `until` is a parameter because delivering and retiring are separate
    events now: a stalled reception is handed to the sink and stays
    tracked, so "the sink saw something" no longer means the loop is
    finished with it.

    The audio's *content* is irrelevant here — the stub modem ignores it
    — but its length is not: the loop refuses to attempt a decode below
    MIN_SECONDS_BEFORE_ATTEMPT, and the ring buffer's fill is what the
    sample-position deadline is measured against."""
    ring = rx_engine.RingBuffer(200.0)
    ring.write(np.zeros(int(seconds_of_audio * FS)))

    if until is None:
        def until(state, sink):
            return bool(sink.receptions)

    state = SharedState()
    stop = threading.Event()
    th = threading.Thread(
        target=decode_loop, args=(ring, None, state, config, stop, sink), daemon=True
    )
    th.start()
    try:
        if feed_more is not None:
            feed_more(ring, state)
        deadline = time.time() + timeout_s
        while time.time() < deadline and not until(state, sink) and th.is_alive():
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
    that decoded nothing.

    Losing sync delivers the reception but no longer retires it: it goes
    dormant ("waiting"), still tracked, until the buffer reaches the
    point where its last frame could have arrived. So this also pins the
    two halves apart -- delivered promptly on the stall, retired on the
    deadline, exactly one picture out of the pair."""
    modem = stub_modem(_StubModem(good_polls=2, frames_received=MODE_A.n_frames // 2))
    sink = _CollectingSink()
    config = RxConfig(poll_interval=0.05, end_grace=0.5)

    need = PREAMBLE_SAMPLES + HEADER_SAMPLES + MODE_A.n_frames * rx_engine.FRAME_SAMPLES

    def feed_more(ring, state):
        deadline = time.time() + 10.0
        while time.time() < deadline and not sink.receptions:
            time.sleep(0.02)
        assert sink.receptions, (
            "a reception whose decodes stopped was never handed to the sink: it is "
            "still sitting in 'receiving' with its picture undelivered, which is "
            "both the hang and the autosave-never-fires report"
        )
        assert state.status == "waiting", (
            f"status {state.status!r} after losing sync: a delivered reception is "
            "still open for the rest of its frames until its scheduled end"
        )
        ring.write(np.zeros(need))

    state = _run(
        config, sink, timeout_s=20.0, feed_more=feed_more,
        until=lambda state, sink: state.status == "listening",
    )

    assert modem.calls_after_stop > 0, (
        "the stub never got past its good polls — the test isn't exercising "
        "the failure at all"
    )
    assert len(sink.receptions) == 1, (
        "delivered twice: a dormant reception that never improved has nothing "
        "new for the sink when its deadline retires it"
    )
    rec = sink.receptions[0]
    assert rec.frames_received == MODE_A.n_frames // 2
    assert rec.mode_name == "A"
    assert rec.callsign == "TEST"
    assert rec.saved_path is None, "the first delivery of a reception replaces nothing"
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

    def feed_more(ring, state):
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


def test_a_header_reception_whose_buffer_stopped_growing_is_delivered(stub_modem):
    """Capture died mid-reception, and the header path had no test that
    could fire.

    It goes on decoding the same partial result from the audio already
    in the buffer, so "it stopped decoding" is false; it never reaches
    its frame count; and the sample-position deadline is measured
    against a `total` that has stopped, so it is unreachable by
    construction. The reception sat in "receiving" indefinitely with its
    picture undelivered -- the same hang this record exists to close,
    left open on one path because the no-advance test used to be
    blind-only.

    The stall metric is the confident-latent count on both paths: a
    frozen buffer decodes the same audio to the same latents forever, so
    the metric never advances and the stall clock is what fires."""
    modem = stub_modem(_StubModem(good_polls=10**9, frames_received=3))
    sink = _CollectingSink()
    config = RxConfig(poll_interval=0.05, end_grace=0.5)

    # Nothing is ever fed: the buffer holds its 6 s and stops, well
    # short of mode A's own duration, so the deadline cannot fire.
    state = _run(config, sink, timeout_s=20.0)

    assert modem.calls > 4, (
        "the stub never decoded repeatedly -- not the case under test"
    )
    assert len(sink.receptions) == 1, (
        "a header-path reception whose audio stopped arriving was never "
        "delivered: it decodes the same frames forever, so only progress "
        "having stopped can end it"
    )
    assert sink.receptions[0].frames_received == 3
    assert state.status == "waiting", (
        f"status {state.status!r}: the reception is delivered but still open "
        "until its scheduled end"
    )


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


def test_the_bar_counts_frames_that_arrived_not_frames_that_decoded(stub_modem, monkeypatch):
    """The two numbers the UI shows, and why they are two.

    `frames_received` is what the progress bar fills with: the frames of
    the transmission whose audio has arrived, which climbs with the
    clock whatever the decoder manages. `frames_decoded` is how many of
    them carried confident data -- a *fill*, held down permanently by
    the erasures this path lives with, which is exactly why it is shown
    beside the bar and never as it.

    Here they are pulled as far apart as a stub can pull them: every one
    of mode A's frames decoded, against a buffer holding only the first
    few seconds of it. And neither ends the reception -- the blind path
    has no contiguous count to finish on, so the stall clock (end_grace)
    is what delivers it."""
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
    assert rec.frames_decoded == MODE_A.n_frames, "every frame carried data"
    assert rec.n_frames_expected == MODE_A.n_frames
    assert 0 < rec.frames_received < MODE_A.n_frames, (
        f"frames_received is {rec.frames_received}: the bar must count the "
        "frames whose audio has arrived, and only a few seconds of this "
        "transmission is in the buffer"
    )
    assert elapsed >= config.end_grace, (
        f"delivered after {elapsed:.2f}s < end_grace={config.end_grace}s: the "
        "reception was completed on a decoded count, cutting off retrospective "
        "backfill"
    )


def _blind_result(frames_present, mode=MODE_A):
    """A blind decode holding every latent of the frames in
    `frames_present` and nothing else -- so its progress metric is the
    number of latents those frames carry."""
    from sstvae.modem import framing
    from sstvae.modem.beacon import BeaconResult
    from sstvae.modem.modem import BlindDemodResult

    weights = np.zeros(MODES["C"].n_latents)
    for f in frames_present:
        _, idx = framing.slot_range_for_frame(f)
        weights[idx] = 0.9
    return BlindDemodResult(
        latents=np.zeros(MODES["C"].n_latents), weights=weights, freq_offset=0.0,
        beacon=BeaconResult(chip_offset=0, frame_index=0, callsign="TEST",
                            mode_index=mode.index),
        callsign="TEST", frame_offset=0, n_frames=mode.n_frames,
        frame0_start=PREAMBLE_SAMPLES + HEADER_SAMPLES, snr_db=10.0,
    )


class _FakeAccumulator:
    """Stands in for BlindAccumulator: the loop only pushes into it,
    reports its score, and forwards whatever `result` returns."""

    def __init__(self, *a, **kw):
        pass

    def push(self, z, start_sample):
        pass

    def best_score(self):
        return 12.0

    def result(self, origin=0):
        return object()


@pytest.fixture
def blind_only(stub_modem, monkeypatch):
    """Install a modem whose blind decodes come from `results`, one per
    poll (the last repeating), where None means "no lock this poll".
    The preamble path is made to fail outright, so the blind path is the
    only thing driving the loop."""

    def install(results):
        class _BlindOnlyModem:
            def __init__(self):
                self.polls = 0

            def demodulate(self, samples, search_s=None, drift_track="off"):
                raise SyncError("no preamble")

            def demodulate_blind(self, x, search_s=None, acquisition=None,
                                 drift_track="off"):
                r = results[min(self.polls, len(results) - 1)]
                self.polls += 1
                if r is None:
                    raise SyncError("lock decayed")
                return r

        modem = stub_modem(_BlindOnlyModem())
        monkeypatch.setattr(
            rx_engine, "sync_acquire",
            lambda z, **kw: (_ for _ in ()).throw(SyncError("no preamble")),
        )
        monkeypatch.setattr(rx_engine, "BlindAccumulator", _FakeAccumulator)
        return modem

    return install


def test_a_faded_reception_resumes_and_replaces_its_picture(blind_only):
    """The reason the stall no longer ends a reception.

    A fade longer than end_grace is indistinguishable from a transmitter
    that stopped, so the picture is delivered at once -- but the
    reception stays tracked until its scheduled end, and a lock
    recovered before then must be routed back into it. The old loop
    recorded the start as finished at that first delivery and
    `_already_finished` then dropped every re-acquisition, so the rest
    of a picture still being heard was refused for as long as the
    transmission lasted.

    The second delivery replaces the first: one transmission, one
    picture, not two files with the better one arriving second."""
    half = list(range(MODE_A.n_frames // 2))
    whole = list(range(MODE_A.n_frames))
    # Decode half of it, lose the lock for longer than end_grace, come
    # back with all of it.
    modem = blind_only([_blind_result(half)] * 2 + [None] * 30 + [_blind_result(whole)])
    sink = _CollectingSink()
    config = RxConfig(poll_interval=0.05, end_grace=0.5)

    seen = []

    def watch(state, sink):
        seen.append(state.status)
        return len(sink.receptions) >= 2

    state = _run(config, sink, timeout_s=20.0, until=watch)

    assert modem.polls > 32, "the fade never ended -- the test isn't exercising resume"
    assert len(sink.receptions) == 2, (
        "a reception delivered when it lost sync never took the rest of its "
        "frames: re-acquiring it before its scheduled end must contribute to "
        "the picture, not be dropped as already handled"
    )
    first, second = sink.receptions
    assert first.saved_path is None
    assert second.saved_path == "/dev/null/fake.png", (
        "the improved picture was delivered as a new reception rather than as a "
        "replacement for the one already saved"
    )
    # The engine-side flag, independent of whether the sink saved: a GUI
    # with autosave off gets no path back and still must not read a
    # redelivery as a second reception.
    assert not first.redelivery
    assert second.redelivery, (
        "a second delivery of the same reception must say so itself -- "
        "saved_path cannot, when the sink declined to save"
    )
    # On frames_decoded, deliberately: frames_received is positional and
    # climbs with the buffer whether or not anything was decoded, so it
    # cannot tell a resumed reception from an abandoned one.
    assert second.frames_decoded > first.frames_decoded
    assert second.frames_decoded == MODE_A.n_frames
    assert "waiting" in seen, "losing sync must be visible as its own state"


def test_a_new_reception_does_not_discard_an_undelivered_one(stub_modem):
    """Taking over the tracked slot delivers what is being replaced.

    Dropping it was survivable while a stall retired receptions within
    end_grace; now that one is kept until its scheduled end, a second
    transmission arriving over the top of it is routine, and the picture
    already decoded would go out with the record.

    end_grace and the deadline are both out of reach here, so the only
    thing that can deliver the first reception is the takeover."""
    class _TwoTransmissions:
        def __init__(self):
            self.polls = 0

        def demodulate(self, samples, search_s=None, drift_track="off"):
            self.polls += 1
            if self.polls <= 2:
                return _demod_result(3)
            return _demod_result(7, preamble_start=int(5 * FS))

        def demodulate_blind(self, x, search_s=None, acquisition=None,
                             drift_track="off"):
            raise SyncError("no blind lock")

    stub_modem(_TwoTransmissions())
    sink = _CollectingSink()
    config = RxConfig(poll_interval=0.05, end_grace=1e9)

    _run(config, sink, timeout_s=20.0)

    assert sink.receptions, (
        "the first reception was discarded when the second took over the "
        "tracked slot: its picture had already decoded and nothing else can "
        "deliver it"
    )
    assert sink.receptions[0].frames_received == 3


def test_a_frozen_buffer_still_retires_on_the_wall_clock(stub_modem):
    """Capture died mid-reception, close to the transmission's end. The
    stall delivers the picture, but retirement's sample-position
    deadline is measured against `total`, and `total` has frozen just
    short of it: unreachable by construction, so the record -- and
    --once, which only exits on retirement -- sat in "waiting" forever
    with the picture already delivered. The wall-clock shadow of the
    deadline is what retires it; while capture is alive it trails the
    buffer deadline by end_grace and never fires.

    The buffer stops half a second short of the scheduled end so this
    test's own wait is bounded by about that plus end_grace -- what is
    asserted is that the wait is finite, not how long it takes."""
    stub_modem(_StubModem(good_polls=10**9, frames_received=3))
    sink = _CollectingSink()
    config = RxConfig(poll_interval=0.05, end_grace=0.4, once=True)

    need_s = (
        PREAMBLE_SAMPLES + HEADER_SAMPLES + MODE_A.n_frames * rx_engine.FRAME_SAMPLES
    ) / FS
    state = _run(
        config, sink, seconds_of_audio=need_s - 0.5, timeout_s=20.0,
        until=lambda state, sink: state.status == "done",
    )

    assert len(sink.receptions) == 1, (
        "a reception whose buffer froze short of its deadline was never "
        "retired: the sample-position deadline is unreachable by construction "
        "there, and only the wall clock can end it"
    )
    assert state.status == "done", (
        f"status {state.status!r}: still waiting on a buffer deadline that a "
        "frozen buffer can never reach"
    )


def _mode_a_weights_for_frames(frames):
    """Mode-A-sized weights with exactly `frames` confidently carried."""
    from sstvae.modem import framing

    w = np.zeros(MODES["C"].n_latents)
    for f in frames:
        _, idx = framing.slot_range_for_frame(f)
        w[idx] = 0.9
    # Mode A's frames live entirely in the first group, so slicing to the
    # mode's latent count loses nothing -- assert it, since a silent
    # truncation here would weaken every test built on this helper.
    assert not w[MODE_A.n_latents:].any()
    return w[:MODE_A.n_latents]


def _demod_result_for_frames(frames):
    n = MODE_A.n_latents
    return DemodResult(
        latents=np.zeros(n),
        weights=_mode_a_weights_for_frames(frames),
        mode=MODE_A,
        freq_offset=0.0,
        sync_metric=0.9,
        frames_received=len(frames),
        beacon=None,
        callsign="TEST",
        preamble_start=0,
        snr_db=12.0,
    )


def test_a_header_reception_resumes_over_its_own_preamble(stub_modem):
    """After a stall-delivery the reception's start is in
    finished_starts, so the preamble search steps over it -- correctly
    for *new* receptions, but the only other way back in used to be the
    blind path, which needs the beacon to decode. With the beacon
    unreadable (narrowband interference on its carrier, a
    pre-PROTOCOL_VERSION-4 sender), a header reception that faded once
    was refused for the rest of the transmission with its preamble
    sitting decodable in the buffer. The engine now aims one demodulate
    at the tracked reception's own preamble when the search finds
    nothing new.

    The stub refuses any search window that excludes its preamble at
    position 0, which is what makes the carve-out visible to it: after
    the delivery only the targeted resume ever offers a window the stub
    accepts."""
    half = _demod_result_for_frames(range(MODE_A.n_frames // 2))
    full = _demod_result_for_frames(range(MODE_A.n_frames))

    class _ResumingHeaderModem:
        def __init__(self):
            self.calls = 0

        def demodulate(self, samples, search_s=None, drift_track="off"):
            if search_s is not None and search_s[0] > 0:
                raise SyncError("the preamble is not in this window")
            self.calls += 1
            if self.calls <= 2:
                return half
            if self.calls <= 12:
                raise SyncError("faded out")
            return full

        def demodulate_blind(self, x, search_s=None, acquisition=None,
                             drift_track="off"):
            raise SyncError("the beacon never decodes")

    modem = stub_modem(_ResumingHeaderModem())
    sink = _CollectingSink()
    config = RxConfig(poll_interval=0.05, end_grace=0.3)

    _run(config, sink, timeout_s=20.0,
         until=lambda state, sink: len(sink.receptions) >= 2)

    assert modem.calls > 12, (
        "the modem never got a decodable window again after the stall "
        "delivery: the header path cannot resume its own reception"
    )
    assert len(sink.receptions) == 2, (
        "a header reception that recovered from a fade, with the beacon "
        "unreadable, never redelivered: only the blind path could resume it"
    )
    first, second = sink.receptions
    assert first.frames_decoded == MODE_A.n_frames // 2
    assert second.frames_decoded == MODE_A.n_frames
    assert second.redelivery and second.saved_path == "/dev/null/fake.png"


def test_an_inferior_blind_resume_does_not_replace_a_better_picture(
        stub_modem, monkeypatch):
    """The stall/delivery metric is one unit -- the confident-latent
    count -- on both paths, because a reception can stall on the header
    path and resume on the blind one. When the header path fed its
    frame count in instead, a blind resume carrying ~two frames of
    content outranked a delivered half-transmission (a frame count
    against a latent count) and overwrote the saved file with the worse
    decode."""
    blind_scrap = _blind_result([0, 1])

    class _HeaderThenScrapModem:
        def __init__(self):
            self.calls = 0
            self.blind_calls = 0

        def demodulate(self, samples, search_s=None, drift_track="off"):
            self.calls += 1
            if self.calls <= 2:
                return _demod_result_for_frames(range(MODE_A.n_frames // 2))
            raise SyncError("gone")

        def demodulate_blind(self, x, search_s=None, acquisition=None,
                             drift_track="off"):
            self.blind_calls += 1
            return blind_scrap

    modem = stub_modem(_HeaderThenScrapModem())
    monkeypatch.setattr(rx_engine, "BlindAccumulator", _FakeAccumulator)
    sink = _CollectingSink()
    config = RxConfig(poll_interval=0.05, end_grace=0.3)

    _run(config, sink, timeout_s=2.5,
         until=lambda state, sink: len(sink.receptions) >= 2)

    assert modem.blind_calls > 5, (
        "the blind path never resumed the reception -- not the case under test"
    )
    assert len(sink.receptions) == 1, (
        "a two-frame blind decode was redelivered over a half-transmission "
        "picture: the paths' metrics are being compared in different units"
    )
    assert sink.receptions[0].frames_decoded == MODE_A.n_frames // 2, (
        "the delivered picture no longer carries the header path's decode"
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

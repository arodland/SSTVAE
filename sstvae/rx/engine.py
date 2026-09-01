"""Continuous live reception: rolling buffer in, decoded pictures out.

Every --poll-interval seconds the loop tries to demodulate the whole
buffer: first the normal preamble-based path (`Modem.demodulate`, works
if the buffer happens to contain the transmission's start), then falling
back to the preamble-free blind path (`Modem.demodulate_blind`, works
from any long-enough mid-transmission excerpt via the beacon
side-channel -- see sstvae/modem/beacon.py). Because the buffer holds
history from before sync was acquired, a mid-stream lock still decodes
the frames that arrived before it -- retrospective decoding, not just
from-here-on.

A reception is finished when a fully-synced decode reports all its
frames received, or when the buffer holds audio past the point where the
last frame of that transmission could possibly still be arriving --
whichever comes first. Losing sync for --end-grace seconds does not
finish it: it *delivers* it, so autosave never waits on a signal that
may not come back, and leaves it tracked until its scheduled end so a
fade it recovers from still contributes. Every one of those tests runs
against `_Pending`, the tracked reception retained *across* polls, so
that none of them depends on the current poll having produced a decode
at all; see `_Pending` for why that is the whole ballgame, and
`_decode_progress` for what "progress" counts as -- which is not what
the progress bar shows, deliberately (`_frames_elapsed`).

`decode_loop_low_cpu` is a cheaper variant that drops the blind fallback
(and with it retrospective mid-stream decoding); see its docstring.

This module is headless and knows nothing about audio devices or any
particular UI: it reads a `RingBuffer` somebody else is filling, pushes
live status into a `SharedState`, and hands finished receptions to a
`sink`. Deciding whether a finished reception gets written to disk is
the sink's job, not the loop's -- the CLI always saves, while the GUI
has an autosave checkbox and may hold the picture for a Save button.
"""

import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

from ..codec import pad_to_full, reconstruct
from ..config import (
    BLIND_MAX_OFFSET_HZ,
    BLIND_WIDE_MAX_OFFSET_HZ,
    FS,
    FRAME_SAMPLES,
    HEADER_SAMPLES,
    MODES,
    MODES_BY_INDEX,
    PREAMBLE_SAMPLES,
)
from ..modem import Modem, SyncError, framing
from ..modem.dsp import to_baseband, to_baseband_at
from ..modem.sync import acquire as sync_acquire
from ..modem.sync import BlindAccumulator
from .ringbuffer import RingBuffer

__all__ = [
    "RxConfig",
    "SharedState",
    "Reception",
    "SaveToDirSink",
    "decode_loop",
    "decode_loop_low_cpu",
    "RingBuffer",
]

MIN_SECONDS_BEFORE_ATTEMPT = 3.0
# How close two acquisitions' transmission-start sample position must be
# to be treated as "the same reception" rather than a new one.
SAME_RECEPTION_EPSILON_S = 1.0
# Weight a blind-path latent must clear to count as "confidently
# received" for the stall-detection progress metric -- see its use
# below. Matches the "good" cutoff tests already judge latents by.
PROGRESS_WEIGHT_THRESHOLD = 0.5


def _decode_progress(weights_full: np.ndarray) -> tuple[int, int]:
    """(stall metric, frames decoded) for one decode's weights.

    The **metric** is the count of confidently-received latents, and
    confidence is what makes it usable as a stall signal:
    demodulate_blind assigns *some* nonzero weight to essentially every
    legal abs_frame slot its ever-growing search range touches, real
    signal or not (just small for noise, after the med_h fix in
    modem.py), so a nonzero count keeps climbing every poll purely from
    the buffer growing -- independent of whether any new real data has
    arrived -- until buffer growth has mapped the *entire* legal
    abs_frame range, which can take a whole mode's duration after the
    real transmission already ended. That read as "stuck receiving
    forever": the stall detector never saw a stable value to end_grace
    against. PROGRESS_WEIGHT_THRESHOLD matches the "good" cutoff already
    used to judge latents elsewhere (see
    tests/test_blind_acquisition.py); only real frames clear it, so the
    count stops climbing exactly when the real data does. **This is the
    number the stall clock watches, and nothing else may be substituted
    for it** -- every alternative here either climbs on buffer growth
    alone (so a reception never stalls) or is a position rather than an
    amount (so retrospective backfill reads as no progress).

    **Frames decoded** is the same mask counted the other way: how many
    of the transmission's frames carried confident data at all. It is a
    *fill*, and it is what the UI shows beside the progress bar rather
    than as it -- a fill fraction reads as a completion percentage and
    is not one, since the erasures both paths live with (a fade, or
    simply not having heard the start) hold it down permanently. What
    fills is `_frames_elapsed`. The interleaver is why the two differ
    at all: each frame's latents are scattered across the whole picture,
    so a latent count answers "how much" and only a frame index answers
    "how far".
    """
    good = weights_full > PROGRESS_WEIGHT_THRESHOLD
    frame_of = framing.frame_of_latent()
    seen = np.zeros(MODES["C"].n_frames, dtype=bool)
    # frame_of is -1 for the never-transmitted dropped latents, and numpy
    # would happily wrap that to the *last* frame. The C++ mirror guards
    # `f >= 0`; the metric still counts every confident latent, dropped
    # slot or not, exactly as the C++ does.
    seen[frame_of[good & (frame_of >= 0)]] = True
    return int(np.count_nonzero(good)), int(np.count_nonzero(seen))


def _frames_elapsed(start: int, total: int, n_frames: int) -> int:
    """How far through its schedule a transmission has got: the progress
    bar's numerator, and pure arithmetic on buffer positions.

    This is what the bar should show, because it is the only one of the
    available numbers that climbs with the clock rather than with the
    decoder's luck. The header path already reported nearly this --
    `DemodResult.frames_received` is `received.sum()`, and `received[f]`
    is set for every frame whose samples are in the buffer, signal or
    noise -- so this is that number generalized to the blind path, which
    had been showing how far its furthest *decoded* frame reached and so
    stalled on the erasures that are its normal state.

    It counts from the transmission's own first frame, not from the
    audio we happened to capture: joining a transmission late leaves its
    early frames permanently unavailable, and a bar that starts at 60%
    and fills to 100% is honest where one that starts at 0 and can never
    reach 100 is not. The frames the join missed show up instead as the
    gap against `frames_decoded`, exactly like a fade's.
    """
    frames_start = start + PREAMBLE_SAMPLES + HEADER_SAMPLES
    elapsed = (total - frames_start) // FRAME_SAMPLES
    return int(min(max(elapsed, 0), n_frames))


@dataclass
class RxConfig:
    """Everything the decode loops need to know. Field names match the
    `sstvae_listen.py` command-line options they came from."""

    out_dir: str = "received"
    poll_interval: float = 5.0
    end_grace: float = 8.0
    size: str | None = None  # "320x240" to downscale saved images
    once: bool = False
    # A cap, not a fixed timescale: BlindAccumulator runs one decay
    # timescale per mode (config.MODES), each capped at
    # min(mode.duration_s, blind_search_seconds) -- see decode_loop.
    # Default is above every mode's own duration, so nothing is capped:
    # there is no reliability reason to raise it further, since there is
    # no more real signal beyond a mode's own duration to integrate (see
    # BlindAccumulator's docstring). Only useful to *shrink* below a
    # mode's own duration.
    blind_search_seconds: float = MODES["C"].duration_s

    # Widen the preamble-free search to config.BLIND_WIDE_MAX_OFFSET_HZ,
    # for a counterpart whose dial is off by hundreds of Hz. Opt-in
    # because unlike the preamble path -- which searches frequency for
    # free and so is always wide -- this one searches CFO directly and
    # its cost is linear in the number of bins: measured, ~1.6x a poll.
    blind_wide: bool = False

    # "off" | "slow" | "fast": follow a carrier that moves during the
    # transmission. Off by default; on HF with a modern radio the
    # receiver's ~+-2 Hz budget is not usually threatened, and the two
    # gains suit different things -- see config.drift_gains and
    # docs/todo.md.
    drift_track: str = "off"


@dataclass
class _Pending:
    """The reception `decode_loop` is currently tracking, retained from
    one poll to the next.

    This exists so the three "is it finished?" tests can fire on a poll
    that decoded *nothing*, which is the whole reason it is a record
    rather than a handful of locals recomputed per poll. A reception
    stops producing a decode long before it stops being real: its audio
    scrolls out of the ring buffer, or -- much sooner -- its blind
    acquisition score falls back under BLIND_SCORE_THRESHOLD as the
    accumulator's evidence decays once the transmission is over. If the
    completion tests are only asked on polls that produced a decode,
    then in exactly the case that matters they are never asked again:
    the loop sits in "receiving" forever, the sink is never called, and
    the picture that *was* decoded is never delivered or saved. Both
    reported symptoms -- an indefinite hang, and autosave never firing
    -- are that one bug, so a poll that decodes nothing must go on
    counting against the reception rather than being skipped over.

    `image` and the fields beside it are the last good decode, held so
    the reception can still be delivered after its decodes have stopped.
    `deadline_abs` is a buffer position, not a time: past it, no real
    frame of this transmission can still be arriving (see decode_loop).

    The record also outlives its own delivery. Delivering and retiring
    are separate events: a stall hands the picture to the sink straight
    away (so autosave is never delayed) but leaves the reception
    *dormant* -- still tracked, still able to take contributions -- until
    its scheduled end. That is what makes a fade survivable. A fade
    longer than end_grace is indistinguishable from a transmitter that
    stopped, and the old behaviour retired the reception on the spot and
    recorded its start as finished, so the blind path's re-acquisition a
    few seconds later was dropped as already-handled and the rest of a
    picture still being heard was refused. Since PROTOCOL_VERSION 4 the
    beacon names the mode, so `deadline_abs` is exact on the blind path
    too, and the scheduled end -- not the stall -- is what ends a
    reception.
    """

    start: int  # absolute (ring-buffer-coordinate) preamble-start position
    deadline_abs: int
    # `deadline_abs` translated to wall time when it was computed, plus
    # end_grace. The buffer-position deadline is the transmitter's clock
    # and is right whenever capture is alive -- but it is measured
    # against `total`, so if capture dies mid-reception `total` freezes
    # below it and it becomes unreachable by construction: the loop
    # would sit in "waiting" forever with the picture already delivered,
    # the same indefinite-hang shape all of this exists to close. The
    # wall clock keeps running when the buffer does not. Anchored when
    # `deadline_abs` is set (or tightened by a later-learned mode),
    # never re-anchored per poll -- recomputing it from `now` every
    # decoded poll would push it forever into the future in exactly the
    # frozen-buffer case it exists for.
    deadline_wall: float | None = None
    # Whether the *most recent* decode of this reception came from the
    # blind path. Per-poll, not per-reception: a reception first found
    # blind can later decode over the preamble path (or vice versa).
    # Only `complete` consults it -- the header path's frames_received
    # is a contiguous decoded count that genuinely finishes at the
    # total, where the blind path's is positional with erasures behind
    # it. The stall clock watches the same confident-latent metric on
    # both paths.
    blind: bool = False
    metric: int = 0  # best progress metric seen; see _decode_progress
    stable_since: float | None = None  # when the metric last stopped advancing
    image: object = None
    mode_name: str | None = None
    frames_received: int | None = None
    frames_decoded: int | None = None
    n_frames_expected: int | None = None
    callsign: str = ""
    snr_db: float = float("nan")
    # `metric` as of the last delivery, and where that delivery went.
    # Together they are the whole dormancy bookkeeping: a delivery only
    # happens when the reception has improved on what the sink already
    # has, and a second one replaces the first in place rather than
    # producing a second picture of one transmission.
    delivered_metric: int = 0
    saved_path: str | None = None


@dataclass
class Reception:
    """A reception the loop considers finished."""

    image: Image.Image
    # None only when the mode is unknown: a blind reception whose beacon
    # carried a mode index this receiver doesn't recognize (a future
    # mode). Ordinarily the blind path knows the mode too, from the
    # beacon's mode field.
    mode_name: str | None
    callsign: str
    snr_db: float
    # How far through its schedule the transmission got, and how much of
    # it actually decoded. The first is the progress bar's numerator and
    # climbs with the clock; the second is a fill, and the gap between
    # them is what a fade -- or joining late -- costs. See
    # _frames_elapsed and _decode_progress.
    frames_received: int | None
    frames_decoded: int | None
    n_frames_expected: int | None
    # Set when this same reception has already been delivered once, to
    # the path named here: a fade ended it early, it recovered before its
    # scheduled end, and this is the better decode. **Replace what is
    # there rather than adding a second picture** -- one transmission is
    # one file, one gallery entry, one notification.
    saved_path: str | None = None
    # True on every delivery after the first, whether or not the sink
    # chose to save. `saved_path` cannot carry this by itself: a sink
    # that declined to save (a GUI with autosave off) returns no path,
    # and a redelivery would then read as a brand-new reception -- two
    # "reception complete" records for one transmission. `saved_path`
    # says where to write; this says whether it is the same reception
    # again.
    redelivery: bool = False


@dataclass
class SharedState:
    # listening | receiving | waiting | done.
    #
    # "waiting" is a reception that lost sync and has already been
    # delivered, but whose scheduled end has not arrived: it is neither
    # receiving a signal (there isn't one) nor idle (a picture is still
    # open for the rest of its frames). See _Pending.
    status: str = "listening"
    mode_name: str | None = None
    # frames_received is how far the transmission has got (what the bar
    # shows); frames_decoded is how many frames carried confident data
    # (shown beside it, never as it). See _frames_elapsed.
    frames_received: int | None = None
    frames_decoded: int | None = None
    n_frames_expected: int | None = None
    progress_frac: float = 0.0
    callsign: str = ""
    snr_db: float = float("nan")
    image: object = None
    saved_path: str | None = None
    seconds_captured: float = 0.0
    last_decode_s: float = 0.0
    # Blind-path observability, refreshed every poll the blind branch
    # runs (NaN / False when it didn't). The score is the accumulator's
    # best prominence whether or not it clears BLIND_SCORE_THRESHOLD,
    # because a below-threshold score is otherwise invisible in live
    # operation and a receiver that fails to acquire on real hardware
    # gives no number to compare against the threshold's calibration.
    # blind_locked distinguishes the two ways the blind path can be
    # silently stuck: score below threshold (not locked), and locked
    # with the beacon not decoding -- which the UI otherwise cannot
    # tell apart, and which mean opposite things (the second is a
    # payload/format problem, e.g. a pre-PROTOCOL_VERSION-4 sender,
    # not a weak signal).
    blind_score: float = float("nan")
    blind_locked: bool = False

    def __post_init__(self):
        self.lock = threading.Lock()


def fmt_snr(snr_db: float) -> str:
    if snr_db != snr_db:  # NaN
        return ""
    return f"  SNR {snr_db:.1f}dB"


def parse_size(size: str | None) -> tuple[int, int] | None:
    if not size:
        return None
    w, h = size.lower().split("x")
    return int(w), int(h)


def timestamped_path(out_dir: str) -> Path:
    """Unique output path. Millisecond resolution because two short-mode
    receptions can be finished within the same second (both already sat
    complete in the buffer), and a second-resolution name would have the
    later one silently overwrite the earlier."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    return Path(out_dir) / f"rx_{ts}.png"


class SaveToDirSink:
    """Default sink: write every finished reception into a directory and
    report it on stdout. This is exactly what the CLI listener did before
    saving became the sink's job."""

    def __init__(self, out_dir: str, size: str | None = None, verbose: bool = True):
        self.out_dir = out_dir
        self.size = parse_size(size)
        self.verbose = verbose

    def on_reception(self, rec: Reception) -> str | None:
        # A reception that recovered after a fade comes back with the
        # path its first delivery went to: overwrite that, so one
        # transmission stays one file.
        out_path = (
            Path(rec.saved_path) if rec.saved_path else timestamped_path(self.out_dir)
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        img = rec.image
        if self.size is not None:
            img = img.resize(self.size)
        img.save(out_path)
        if self.verbose:
            verb = "updated" if rec.saved_path else "saved"
            print(
                f"{verb} {out_path} (mode={rec.mode_name or 'unknown, blind sync'}, "
                f"callsign={rec.callsign or '(none)'}{fmt_snr(rec.snr_db)})"
            )
        return str(out_path)


def _deliver(pending: "_Pending", sink, state: "SharedState", finished_starts) -> bool:
    """Hand `pending` to the sink if it has improved on what the sink
    already has. Returns whether anything was delivered.

    Called both when a reception stalls and when it retires, which is the
    whole point: the stall delivers so autosave is prompt, retirement
    delivers whatever arrived afterwards. `delivered_metric` is what
    keeps that from turning into a delivery per poll, and `saved_path`
    is what makes the second delivery a replacement rather than a second
    picture of the same transmission.

    The first delivery is also what records the start in
    `finished_starts` -- from then on the preamble search steps over it,
    so a transmission that was cut off early cannot go on hiding a later
    one for the rest of its own scheduled duration. The reception stays
    resumable anyway, on both paths: the blind path checks the tracked
    reception before that list, and the header path aims one demodulate
    at its own preamble when the search finds nothing new (see
    decode_loop).
    """
    if pending.metric <= pending.delivered_metric or pending.image is None:
        return False
    first = pending.delivered_metric == 0
    saved_path = sink.on_reception(
        Reception(
            image=pending.image,
            mode_name=pending.mode_name,
            callsign=pending.callsign,
            snr_db=pending.snr_db,
            frames_received=pending.frames_received,
            frames_decoded=pending.frames_decoded,
            n_frames_expected=pending.n_frames_expected,
            saved_path=pending.saved_path,
            redelivery=not first,
        )
    )
    pending.delivered_metric = pending.metric
    if saved_path is not None:
        pending.saved_path = saved_path
    if first:
        # Bookkeeping, not disk: this reception has been *handled*, so it
        # must never be rediscovered as a new one while its audio is
        # still in the buffer -- whether or not the sink chose to save.
        finished_starts.append(pending.start)
    with state.lock:
        state.saved_path = saved_path
    return True


def _already_finished(pos: float, finished_starts, epsilon_samples: float) -> bool:
    return any(abs(pos - k) <= epsilon_samples for k in finished_starts)


def _free_spans(n: int, buf_start: int, finished_starts, epsilon_samples: int):
    """Local [lo, hi) spans of the buffer with already-saved receptions'
    preambles carved out.

    Only the preamble region of a finished reception is excluded, not its
    whole duration: two transmissions can overlap in time, and blanking a
    finished one's full extent would hide an overlapping neighbour's
    preamble along with it.
    """
    blocked = []
    for p in finished_starts:
        lo = int(p - buf_start - epsilon_samples)
        hi = int(p - buf_start + PREAMBLE_SAMPLES + epsilon_samples)
        if hi > 0 and lo < n:
            blocked.append((max(0, lo), min(n, hi)))
    blocked.sort()
    spans, cur = [], 0
    for lo, hi in blocked:
        if lo > cur:
            spans.append((cur, lo))
        cur = max(cur, hi)
    if cur < n:
        spans.append((cur, n))
    return spans


def _new_blind_accumulator(config: RxConfig) -> BlindAccumulator:
    """A fresh accumulator with one decay timescale per mode, each capped
    at that mode's own duration (see RxConfig.blind_search_seconds)."""
    timescales = [min(m.duration_s, config.blind_search_seconds) for m in MODES.values()]
    return BlindAccumulator(
        max_offset_hz=(
            BLIND_WIDE_MAX_OFFSET_HZ if config.blind_wide else BLIND_MAX_OFFSET_HZ
        ),
        window_s=timescales,
    )


def _find_new_reception(modem, samples, z, buf_start, finished_starts,
                        epsilon_samples, max_tries=4, drift_track="off"):
    """Decode the strongest preamble that is neither already saved nor a
    spurious peak. Returns (DemodResult, reception_start) or (None, None).

    sync_acquire returns a single global argmax inside its window, so an
    already-decoded transmission still sitting in the buffer can outrank
    and hide a second one. Searching only *forward* of finished hits
    doesn't fix that either -- the strongest peak is often the later
    transmission, and stepping past it buries every earlier one. So
    search each still-unclaimed span, and within a span step past any
    peak whose header won't decode (a correlation artefact inside a
    transmission's own frames rather than a real preamble).

    demodulate gets the same window the hit came from. It runs its own
    acquisition, and left to scan the whole buffer it can lock a
    different preamble than the one just vetted -- which is how an
    already-saved reception ends up decoded and written out a second
    time while the bookkeeping records some other position.
    """
    n = len(samples)
    tries = 0
    for span_lo, span_hi in _free_spans(n, buf_start, finished_starts,
                                        epsilon_samples):
        lo = span_lo
        while lo < span_hi and tries < max_tries:
            try:
                acq = sync_acquire(z, search=(lo, span_hi))
            except SyncError:
                break
            tries += 1
            try:
                r = modem.demodulate(
                    samples, search_s=(lo / FS, span_hi / FS),
                    drift_track=drift_track,
                )
                return r, buf_start + r.preamble_start
            except SyncError:
                lo = acq.preamble_start + PREAMBLE_SAMPLES
    return None, None


def decode_loop(ring: RingBuffer, model, state: SharedState, config: RxConfig,
                stop_event: threading.Event, sink=None):
    if sink is None:
        sink = SaveToDirSink(config.out_dir, config.size)
    modem = Modem()
    epsilon_samples = SAME_RECEPTION_EPSILON_S * FS
    # Absolute (ring-buffer-coordinate) transmission-start sample position
    # of every reception already saved this run. The ring buffer keeps
    # holding a finished reception's audio for up to --buffer-seconds
    # afterward, so without this a still-buffered transmission would be
    # rediscovered and re-decoded/re-saved on every following poll.
    finished_starts = deque(maxlen=50)
    # The reception being tracked, or None while listening. Progress is
    # per-reception: carrying a previous transmission's metric across to
    # the next one can make a brand-new reception look like it has
    # already stalled and end it early, so a different reception gets a
    # fresh record rather than an updated one.
    pending: _Pending | None = None
    # Blind acquisition's persistent search state: a BlindAccumulator
    # folds in only the audio that's new since the last poll (see
    # sync.BlindAccumulator), so unlike the preamble path it carries
    # state across iterations of this loop instead of re-deriving
    # everything from the current snapshot each time. blind_acc_pushed
    # is the absolute (ring-buffer-coordinate) sample count already
    # folded in; None means "nothing pushed yet, or the last push's
    # position fell out of the ring buffer's retained window" -- either
    # way there is a gap push() cannot bridge contiguously, so the right
    # response is a fresh accumulator over what is available now rather
    # than guessing at what filled it.
    blind_acc = None
    blind_acc_pushed = None

    while not stop_event.is_set():
        stop_event.wait(config.poll_interval)
        if stop_event.is_set():
            break

        samples, total = ring.snapshot()
        seconds_captured = total / FS
        with state.lock:
            state.seconds_captured = seconds_captured
        if len(samples) < MIN_SECONDS_BEFORE_ATTEMPT * FS:
            continue
        buf_start = total - len(samples)

        t0 = time.time()
        latents_full = weights_full = None
        mode_name = n_frames_expected = frames_received = None
        frames_decoded = None
        callsign = ""
        snr_db = float("nan")
        progress_frac = 0.0
        progress_metric = 0
        reception_start = None
        blind_score = float("nan")
        blind_locked = False

        # Preamble path first: find and decode the strongest reception
        # that hasn't already been saved. Falls through to the blind path
        # below if nothing there decodes (corrupted header, or the only
        # hits are spurious correlation peaks).
        r, reception_start = _find_new_reception(
            modem, samples, to_baseband(samples), buf_start,
            finished_starts, epsilon_samples, drift_track=config.drift_track,
        )
        # A delivered-but-still-open reception is in finished_starts, so
        # the search above steps over its preamble -- correctly, for new
        # receptions (a transmission cut off early must not hide a later
        # one), and fatally for resuming *this* one over the header path:
        # the blind branch has its own resume (the `ours` check below)
        # but needs the beacon to decode, and "blind-locked with the
        # beacon not decoding" is a real field condition. So when nothing
        # new was found, aim one demodulate at the tracked reception's
        # own preamble; if the signal came back, this decode is
        # retrospective over the whole ring and picks up everything.
        if r is None and pending is not None and _already_finished(
            pending.start, finished_starts, epsilon_samples
        ):
            lo = max(0, int(pending.start - buf_start - epsilon_samples))
            hi = min(
                len(samples),
                int(pending.start - buf_start + PREAMBLE_SAMPLES + epsilon_samples),
            )
            if hi - lo >= PREAMBLE_SAMPLES:
                try:
                    r = modem.demodulate(
                        samples, search_s=(lo / FS, hi / FS),
                        drift_track=config.drift_track,
                    )
                    reception_start = buf_start + r.preamble_start
                except SyncError:
                    pass
        full_ok = r is not None

        if full_ok:
            latents_full = pad_to_full(r.latents)
            weights_full = pad_to_full(r.weights)
            mode_name = r.mode.name
            n_frames_expected = r.mode.n_frames
            # `received.sum()`: the frames whose samples are in the
            # buffer, which is the bar's numerator here (see
            # _frames_elapsed -- on this path the two agree while
            # capture is alive, since the preamble was heard).
            frames_received = r.frames_received
            callsign = r.callsign
            snr_db = r.snr_db
            progress_frac = frames_received / n_frames_expected
            # The stall/delivery metric is the confident-latent count on
            # *both* paths -- one unit, because `pending.metric` and
            # `delivered_metric` compare across polls and a reception can
            # switch paths between them (a header reception that stalls
            # and resumes blind, or the reverse). The header path used to
            # feed frames_received here, and a frame count against a
            # latent count let a ~14-frame blind resume outrank a
            # 300-frame delivered picture. It also gates delivery on
            # decoded content rather than buffer coverage, which is what
            # keeps a spurious lock in noise from being *delivered* as a
            # picture of noise at its stall or deadline.
            progress_metric, frames_decoded = _decode_progress(weights_full)
        else:
            # Fold whatever's new since the last poll into the running
            # accumulator -- O(new samples), not O(window) -- which is
            # what lets this run one decay timescale per mode (see
            # RxConfig.blind_search_seconds) rather than a single
            # one-size-fits-all window. The retrospective decode below
            # still covers the whole current buffer once locked, exactly
            # as before.
            if blind_acc is None or blind_acc_pushed is None or blind_acc_pushed < buf_start:
                blind_acc = _new_blind_accumulator(config)
                blind_acc_pushed = buf_start
            new_lo = blind_acc_pushed - buf_start
            if new_lo < len(samples):
                new_chunk = to_baseband_at(samples[new_lo:], blind_acc_pushed)
                blind_acc.push(new_chunk, blind_acc_pushed)
                blind_acc_pushed = total

            blind_score = blind_acc.best_score()
            try:
                # origin=buf_start: the accumulator folds in absolute
                # (ring-coordinate) phase, but the phase is about to be
                # used against `samples`, which starts at buf_start.
                # Without this the two coordinates agree only while the
                # session is younger than the ring buffer -- see
                # BlindAccumulator.result for the field failure that
                # found it.
                ba = blind_acc.result(origin=buf_start)
            except SyncError:
                ba = None
            blind_locked = ba is not None
            try:
                rb = (
                    modem.demodulate_blind(
                        samples, acquisition=ba, drift_track=config.drift_track
                    )
                    if ba is not None
                    else None
                )
            except SyncError:
                rb = None
            if rb is not None and rb.beacon is not None and rb.frame0_start is not None:
                # Record every reception in the same coordinate -- the
                # preamble start -- so finished_starts is homogeneous.
                # The blind path locates absolute frame 0, which sits one
                # preamble+header later; without this the two paths label
                # one transmission with two positions 768 samples apart,
                # and _free_spans blocks the wrong region.
                reception_start = (
                    buf_start + rb.frame0_start - PREAMBLE_SAMPLES - HEADER_SAMPLES
                )
                # The tracked reception is checked *before* the
                # finished list, and that is the whole resume mechanism:
                # a reception delivered early on a stall is in that list
                # (so the preamble search steps over it) while still
                # being the one we are tracking, and dropping its
                # re-acquisition here is exactly what used to refuse the
                # rest of a picture that was still being heard.
                #
                # Anything else already handled: treat it as nothing
                # decoded rather than skipping the rest of the poll, so
                # that whatever reception is *currently* pending still
                # gets its completion tests run this time round.
                ours = (
                    pending is not None
                    and abs(reception_start - pending.start) <= epsilon_samples
                )
                if not ours and _already_finished(
                    reception_start, finished_starts, epsilon_samples
                ):
                    reception_start = None
                else:
                    latents_full = rb.latents
                    weights_full = rb.weights
                    callsign = rb.callsign
                    snr_db = rb.snr_db
                    # The beacon's mode field (PROTOCOL_VERSION 4) names
                    # the transmission's real mode, so the blind path no
                    # longer has to assume mode C: the deadline, the
                    # progress denominator and the reported mode are all
                    # exact. An unknown index (a future mode) keeps the
                    # old assume-mode-C behaviour.
                    blind_spec = MODES_BY_INDEX.get(rb.beacon.mode_index)
                    if blind_spec is not None:
                        mode_name = blind_spec.name
                        n_frames_expected = blind_spec.n_frames
                    # The stall metric stays the confident-latent count
                    # -- the amount decoded, which is the only thing
                    # that stops climbing when the signal does. The bar
                    # is the positional count instead, so a fade holds
                    # back the *decoded* figure beside it rather than
                    # freezing the bar.
                    progress_metric, frames_decoded = _decode_progress(weights_full)
                    n_display = (
                        n_frames_expected if n_frames_expected is not None
                        else MODES["C"].n_frames
                    )
                    frames_received = _frames_elapsed(
                        reception_start, total, n_display
                    )
                    progress_frac = frames_received / n_display
            else:
                reception_start = None

        decode_s = time.time() - t0

        with state.lock:
            state.last_decode_s = decode_s
            state.blind_score = blind_score
            state.blind_locked = blind_locked

        advanced = False
        decoded = latents_full is not None and reception_start is not None
        if decoded:
            # A different reception than the one being tracked: it takes
            # over, with a fresh record. Its very first poll must not
            # inherit the previous one's stall clock, or it can be
            # mistaken for a reception that has already stopped
            # advancing and be ended immediately.
            #
            # The outgoing record is delivered first. Taking over used to
            # discard it -- picture, metric and all -- which was survivable
            # only because a stall retired receptions quickly; now that one
            # is kept until its scheduled end, the overlap is routine and
            # dropping it would lose a picture that had already decoded.
            # No "done" flash here: the new reception publishes "receiving"
            # a few lines below, and a status the operator cannot see is
            # not worth a two-second pause in the loop.
            if pending is None or abs(reception_start - pending.start) > epsilon_samples:
                if pending is not None:
                    _deliver(pending, sink, state, finished_starts)
                pending = _Pending(start=reception_start, deadline_abs=0)
                # Nothing has been saved for *this* reception yet, and a
                # status line that named the previous one's file would
                # be attributing it to the picture now on screen.
                with state.lock:
                    state.saved_path = None
            # The deterministic backstop. The transmission's start is
            # known exactly -- from the header path's own acquisition,
            # or from the beacon on the blind path, both in the same
            # "preamble start" coordinate -- so the latest a real
            # frame of it can possibly still be arriving is its own
            # duration after that point, fixed the moment the start
            # is. The mode comes from the header, or from the beacon's
            # mode field on the blind path; only a beacon whose mode
            # index this receiver doesn't know (a future mode) falls
            # back to mode C's duration, the longest. That is unlike
            # "progress stopped advancing" below, which rides on the
            # decoder's own noise floor and is not guaranteed to ever
            # settle. Once the buffer holds audio past this deadline
            # there is provably no more real signal left to arrive
            # for this reception, done or not. Recomputed every decoded
            # poll (from pending.start, which is stable across polls)
            # so a mode learned on a later poll tightens it.
            n_deadline_frames = (
                n_frames_expected if n_frames_expected is not None
                else MODES["C"].n_frames
            )
            deadline_abs = (
                pending.start + PREAMBLE_SAMPLES + HEADER_SAMPLES
                + n_deadline_frames * FRAME_SAMPLES
            )
            if deadline_abs != pending.deadline_abs:
                pending.deadline_abs = deadline_abs
                # Its wall-clock shadow, anchored here and only here --
                # see _Pending.deadline_wall for why re-anchoring it
                # every poll would defeat it.
                pending.deadline_wall = (
                    time.time() + (deadline_abs - total) / FS + config.end_grace
                )
            pending.blind = not full_ok
            if progress_metric > pending.metric:
                pending.metric = progress_metric
                advanced = True
            # Only an at-least-as-good decode replaces the held picture.
            # A reception now lives until its scheduled end, so it sees
            # decodes taken during a fade -- which are real decodes with
            # fewer frames in them, and used to overwrite a better
            # picture wholesale while `metric`, the only monotone field,
            # went on saying progress had been made.
            if progress_metric >= pending.metric:
                pending.image = reconstruct(model, latents_full, weights_full)
                pending.mode_name = mode_name
                pending.frames_received = frames_received
                pending.frames_decoded = frames_decoded
                pending.n_frames_expected = n_frames_expected
                pending.callsign = callsign
                pending.snr_db = snr_db
            with state.lock:
                state.status = "receiving"
                state.mode_name = mode_name
                state.frames_received = frames_received
                state.frames_decoded = frames_decoded
                state.n_frames_expected = n_frames_expected
                state.progress_frac = min(progress_frac, 1.0)
                state.callsign = callsign
                state.snr_db = snr_db
                state.image = pending.image

        if pending is None:
            with state.lock:
                state.status = "listening"
            continue

        # Two things interrupt a reception short of its own frame count,
        # and both are timed against end_grace. Since the beacon's mode
        # field made the deadline exact on both paths, neither of them
        # *ends* a reception any more: they deliver it, and it stays
        # tracked until its scheduled end. A fade longer than end_grace
        # looks exactly like a transmitter that stopped, and only the
        # transmitter's own clock can tell the difference -- so the
        # picture goes to the sink now (autosave must not wait on a
        # signal that may never come back) and the reception stays open
        # for the rest of it (nothing is lost if it does).
        #
        # It stopped decoding. That is how a reception actually ends in
        # the field: its audio scrolls out of the ring buffer, or its
        # blind acquisition score falls back under
        # BLIND_SCORE_THRESHOLD as the accumulator's evidence decays
        # once the transmission is over. A poll that decoded nothing is
        # not a neutral event -- it is the absence of progress, and is
        # timed as such. Resetting the clock there instead, which is
        # what the loop used to do from the branch that skipped the rest
        # of the poll entirely, is why a reception in that state could
        # never satisfy any completion test however long it sat.
        #
        # Or its decoded progress stopped advancing. The metric is the
        # confident-latent count on both paths (see _decode_progress),
        # so on the header path this fires on a fade -- delivering
        # promptly, exactly as the blind path does -- and on a capture
        # that died mid-reception, where the same audio decodes to the
        # same latents forever, the count never reaches anything, and
        # `total` has stopped so the buffer deadline is unreachable (the
        # wall-clock shadow is what retires it). This used to be
        # blind-only, on the grounds that a spurious preamble-shaped
        # lock in noise goes on decoding the same handful of frames for
        # as long as it is looked at, and that ending *that* on a stall
        # turns a false lock into an autosaved picture of noise. What
        # protects against that is not the stall's reach but the
        # delivery gate: a stall (or a deadline) delivers nothing unless
        # confident latents were decoded, and a false lock in noise has
        # essentially none -- where the buffer-coverage count the header
        # path used to feed this clock climbed for noise exactly as for
        # signal, and would have *delivered* the noise at the deadline.
        no_progress = not decoded or not advanced
        if no_progress:
            if pending.stable_since is None:
                pending.stable_since = time.time()
        else:
            pending.stable_since = None

        # Header path only: its frames_received is a contiguous decoded
        # count, so reaching the total genuinely means everything
        # arrived. The blind path's frames_received is a *reach* -- the
        # furthest frame decoded, with erasures routinely behind it --
        # and a reach at the last frame is exactly when retrospective
        # backfill is still improving the picture, so ending on it would
        # trade picture quality for nothing (the deadline already bounds
        # the wait).
        complete = (
            not pending.blind
            and pending.n_frames_expected is not None
            and pending.frames_received is not None
            and pending.frames_received >= pending.n_frames_expected
        )
        stalled = (
            pending.stable_since is not None
            and (time.time() - pending.stable_since) >= config.end_grace
        )
        # Retirement is the transmitter's own clock, not ours: its frame
        # count is in, or the buffer holds audio past the point where
        # its last frame could still be arriving. The wall-clock shadow
        # is for the one case where the buffer's clock has stopped --
        # capture died mid-reception, so `total` froze below the
        # deadline and would hold the record (and `--once`) in "waiting"
        # forever. While capture is alive the buffer deadline fires
        # first by construction (the shadow trails it by end_grace).
        # Nothing else drops the record -- see _Pending.
        retire = (
            complete
            or total >= pending.deadline_abs
            or (pending.deadline_wall is not None
                and time.time() >= pending.deadline_wall)
        )

        if not retire:
            if stalled:
                # Delivered, but still tracked: dormant. _deliver is a
                # no-op unless it improved on what the sink already has,
                # so a reception that stays quiet is handed over exactly
                # once however many polls it sits through -- while the
                # status says "waiting" throughout, including for a
                # reception that never had a confident latent to deliver
                # at all.
                _deliver(pending, sink, state, finished_starts)
                with state.lock:
                    state.status = "waiting"
            continue

        # Deliver only what has something in it. A reception that ended
        # with no confidently-received latent at all is not a picture,
        # and is deliberately *not* recorded in finished_starts either:
        # there is nothing to protect against re-saving, and a real
        # transmission that was merely weak on its first polls must stay
        # findable rather than be blocked out for as long as its audio
        # is in the buffer.
        _deliver(pending, sink, state, finished_starts)
        delivered = pending.delivered_metric > 0
        if delivered:
            # Retire the blind evidence with the reception. The delivered
            # transmission's fold otherwise keeps the accumulator's
            # argmax for a long time -- its peak/median score does not
            # decay on its own (decay scales a timescale's bins equally;
            # see BlindAccumulator) and its off-phase energy inflates the
            # median under any new peak in the same CFO row, so a new
            # transmission could not lock blind until minutes of noise
            # had diluted the old evidence, and every poll meanwhile
            # re-ran demodulate_blind on the finished transmission only
            # to discard it via finished_starts. That was the "locks
            # after a fresh start, not after a reception" report.
            # blind_acc_pushed = total, NOT None: None (or leaving it)
            # would make the next poll fold the still-buffered finished
            # transmission straight back in, rebuilding exactly the
            # evidence being discarded. The known cost, accepted
            # deliberately: a second transmission *overlapping* the
            # delivered one loses whatever blind evidence it had already
            # accumulated and rebuilds from delivery time -- the same
            # position a stop/start gives, and its audio is still in the
            # ring, so a rebuilt lock still decodes it retrospectively.
            # Using the old evidence for the overlapper while rejecting
            # the delivered peak would need per-reception subtraction
            # the CPU-friendly accumulator cannot do.
            blind_acc = _new_blind_accumulator(config)
            blind_acc_pushed = total
            with state.lock:
                state.status = "done"

        pending = None

        if delivered and config.once:
            stop_event.set()
            break

        if delivered:
            stop_event.wait(2.0)
        with state.lock:
            state.status = "listening"
            state.mode_name = None
            state.frames_received = None
            state.frames_decoded = None
            state.n_frames_expected = None
            state.progress_frac = 0.0
            state.callsign = ""
            state.snr_db = float("nan")


def decode_loop_low_cpu(ring: RingBuffer, model, state: SharedState, config: RxConfig,
                        stop_event: threading.Event, sink=None):
    """Header-only variant: no blind fallback, no retrospective decode.
    While idle, only searches the newly-arrived slice of audio each poll
    (not the whole buffer) for the preamble. Once it locks, it does no
    further signal processing at all until enough audio has been
    captured for the whole transmission, then decodes and saves once."""
    if sink is None:
        sink = SaveToDirSink(config.out_dir, config.size)
    modem = Modem()
    search_overlap_s = 2.0  # margin so a preamble can't be missed by
    # straddling the boundary between one poll's search window and the next
    last_search_pos = 0  # abs sample position (ring.total_written coord)

    while not stop_event.is_set():
        stop_event.wait(config.poll_interval)
        if stop_event.is_set():
            break

        samples, total = ring.snapshot()
        seconds_captured = total / FS
        if len(samples) < MIN_SECONDS_BEFORE_ATTEMPT * FS:
            continue
        buf_start = total - len(samples)

        search_from_abs = max(buf_start, last_search_pos - int(search_overlap_s * FS))
        search = (search_from_abs - buf_start, len(samples))

        with state.lock:
            state.seconds_captured = seconds_captured
            if state.status != "receiving":
                state.status = "listening"

        try:
            acq = sync_acquire(to_baseband(samples), search=search)
        except SyncError:
            last_search_pos = total
            continue

        try:
            # Same window the hit came from, so demodulate can't lock a
            # different (older, already-saved) preamble than the one just
            # found -- see _find_new_reception's docstring.
            r = modem.demodulate(
                samples, search_s=(search[0] / FS, search[1] / FS),
                drift_track=config.drift_track,
            )
        except SyncError:
            last_search_pos = total
            continue  # spurious preamble-shaped hit; keep listening

        reception_start = buf_start + acq.preamble_start
        frames_end_abs = (
            reception_start + PREAMBLE_SAMPLES + HEADER_SAMPLES + r.mode.n_frames * FRAME_SAMPLES
        )

        with state.lock:
            state.status = "receiving"
            state.mode_name = r.mode.name
            state.frames_received = r.frames_received
            state.n_frames_expected = r.mode.n_frames
            state.progress_frac = min(r.frames_received / r.mode.n_frames, 1.0)
            state.callsign = r.callsign
            state.snr_db = r.snr_db

        # No further DSP until the whole transmission should have
        # arrived -- just wait, updating the status text cheaply.
        #
        # Bounded, because this waits on something outside the loop's
        # control: if capture stops -- the device is unplugged, the
        # stream dies, the host stops delivering callbacks -- total_now
        # stops advancing, and an unbounded wait here holds the receiver
        # in "receiving" forever waiting for audio that will never come,
        # with no picture ever handed to the sink. The bound is how long
        # the audio still missing should take to arrive, plus end_grace;
        # past it, decode whatever did arrive.
        wait_deadline = time.time() + (frames_end_abs - total) / FS + config.end_grace
        while not stop_event.is_set():
            _, total_now = ring.snapshot()
            if total_now >= frames_end_abs or time.time() >= wait_deadline:
                break
            with state.lock:
                state.seconds_captured = total_now / FS
            stop_event.wait(min(1.0, config.poll_interval))
        if stop_event.is_set():
            break

        samples, total = ring.snapshot()
        # Never look for this reception's preamble again: without this the
        # next poll resumes from where the search stood *before* waiting
        # out the transmission, re-finds the preamble that is still in the
        # buffer, and decodes and saves the same image a second time.
        last_search_pos = frames_end_abs
        # Re-anchor the window on this reception in the grown buffer's
        # coordinates, so the final decode can't lock a different preamble.
        lo = max(0, reception_start - (total - len(samples)))
        try:
            r = modem.demodulate(
                samples, search_s=(lo / FS, len(samples) / FS),
                drift_track=config.drift_track,
            )
        except SyncError:
            # transmission was cut short / corrupted after all; go back
            # to listening rather than crash the loop
            with state.lock:
                state.status = "listening"
            continue

        weights_full = pad_to_full(r.weights)
        img = reconstruct(model, pad_to_full(r.latents), weights_full)
        _, frames_decoded = _decode_progress(weights_full)
        saved_path = sink.on_reception(
            Reception(
                image=img,
                mode_name=r.mode.name,
                callsign=r.callsign,
                snr_db=r.snr_db,
                frames_received=r.frames_received,
                frames_decoded=frames_decoded,
                n_frames_expected=r.mode.n_frames,
            )
        )
        with state.lock:
            state.status = "done"
            state.image = img
            state.frames_received = r.frames_received
            state.frames_decoded = frames_decoded
            state.progress_frac = min(r.frames_received / r.mode.n_frames, 1.0)
            state.snr_db = r.snr_db
            state.saved_path = saved_path

        if config.once:
            stop_event.set()
            break

        stop_event.wait(2.0)
        with state.lock:
            state.status = "listening"
            state.mode_name = None
            state.frames_received = None
            state.frames_decoded = None
            state.n_frames_expected = None
            state.progress_frac = 0.0
            state.callsign = ""
            state.snr_db = float("nan")

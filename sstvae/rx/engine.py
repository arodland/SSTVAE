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

Reception is considered finished either when a fully-synced decode
reports all its frames received, or (blind case, true mode/duration
unknown) when decoded progress stops advancing for --end-grace seconds
-- backstopped by a hard deadline, mode C's own duration (the longest
mode) past the transmission's start as the beacon already reported it,
so a reception can't sit in "receiving" indefinitely if the stall
detector's metric never settles (see PROGRESS_WEIGHT_THRESHOLD and the
deadline computation in decode_loop's blind branch).

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
from ..config import FS, FRAME_SAMPLES, HEADER_SAMPLES, MODES, PREAMBLE_SAMPLES
from ..modem import Modem, SyncError
from ..modem.diversity import combine_diversity_results, contribution_image
from ..modem.dsp import to_baseband, to_baseband_at
from ..modem.modem import DemodResult
from ..modem.sync import acquire as sync_acquire
from ..modem.sync import BlindAccumulator
from .ringbuffer import RingBuffer

__all__ = [
    "RxConfig",
    "SharedState",
    "Reception",
    "SaveToDirSink",
    "SaveDebugImageToDirSink",
    "decode_loop",
    "decode_loop_low_cpu",
    "decode_loop_diversity",
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


@dataclass
class Reception:
    """A reception the loop considers finished."""

    image: Image.Image
    mode_name: str | None  # None when the mode is unknown (blind sync)
    callsign: str
    snr_db: float
    frames_received: int | None
    n_frames_expected: int | None


@dataclass
class SharedState:
    status: str = "listening"  # listening | receiving | done
    mode_name: str | None = None
    frames_received: int | None = None
    n_frames_expected: int | None = None
    progress_frac: float = 0.0
    callsign: str = ""
    snr_db: float = float("nan")
    image: object = None
    saved_path: str | None = None
    seconds_captured: float = 0.0
    last_decode_s: float = 0.0
    # Per-branch lock state, published only by decode_loop_diversity --
    # both stay False for the single-receiver loops. Each reflects
    # whichever ring most recently supplied a hit that fed the last
    # poll's combine, so a branch that acquires and later drops out of
    # range goes False again on the next poll that doesn't see it, not
    # just at reception end.
    branch_a_locked: bool = False
    branch_b_locked: bool = False

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
        out_path = timestamped_path(self.out_dir)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        img = rec.image
        if self.size is not None:
            img = img.resize(self.size)
        img.save(out_path)
        if self.verbose:
            print(
                f"saved {out_path} (mode={rec.mode_name or 'unknown, blind sync'}, "
                f"callsign={rec.callsign or '(none)'}{fmt_snr(rec.snr_db)})"
            )
        return str(out_path)


class SaveDebugImageToDirSink:
    """Optional companion to a picture sink, for `decode_loop_diversity`'s
    per-latent branch-contribution image (`sstvae.modem.diversity.
    contribution_image`) -- which branch supplied each transmitted
    latent, red vs blue. Not a `Reception` sink itself (there is no
    picture here, just a diagnostic), so it takes the image and the
    already-saved picture's path directly and writes alongside it."""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose

    def on_contribution_image(self, image: Image.Image, saved_path: str | None) -> str | None:
        if saved_path is None:
            return None
        p = Path(saved_path)
        out_path = p.with_name(p.stem + "_diversity" + p.suffix)
        image.save(out_path)
        if self.verbose:
            print(f"saved {out_path} (diversity branch-contribution map)")
        return str(out_path)


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


def _find_new_reception(modem, samples, z, buf_start, finished_starts,
                        epsilon_samples, max_tries=4):
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
                r = modem.demodulate(samples, search_s=(lo / FS, span_hi / FS))
                return r, buf_start + r.preamble_start
            except SyncError:
                lo = acq.preamble_start + PREAMBLE_SAMPLES
    return None, None


def _find_branch_reception(modem, samples, buf_start, finished_starts,
                           epsilon_samples, blind_search_seconds):
    """One diversity branch's `decode_loop`-style preference: header path
    first, falling back to blind. Used only by `decode_loop_diversity`
    -- `decode_loop` itself keeps its own inline version of this same
    preference, since it is explicitly load-bearing (CLAUDE.md) and this
    keeps it byte-for-byte unchanged.

    Returns `(result, reception_start)` or `(None, None)`. `result` is a
    `DemodResult` or a `BlindDemodResult`; `reception_start` is always
    expressed in the *preamble's* sample position (ring-buffer
    coordinates), whichever path found it -- the blind path's
    `frame0_start` is one preamble+header later, so it is shifted back
    by that much, same correction `decode_loop`'s own blind branch
    applies. That gives every branch, however it locked, one directly
    comparable position -- which is what lets `decode_loop_diversity`
    match branches (and dedupe against `finished_starts`) the same way
    regardless of acquisition path, and it is exactly the field
    `decode_loop` already uses for its own bookkeeping.
    """
    z = to_baseband(samples)
    r, start = _find_new_reception(
        modem, samples, z, buf_start, finished_starts, epsilon_samples,
    )
    if r is not None:
        return r, start

    n = len(samples)
    blind_span = int(blind_search_seconds * FS)
    blind_search = None if n <= blind_span else ((n - blind_span) / FS, n / FS)
    try:
        rb = modem.demodulate_blind(samples, search_s=blind_search)
    except SyncError:
        return None, None
    if rb.beacon is None or rb.frame0_start is None:
        return None, None
    reception_start = buf_start + rb.frame0_start - PREAMBLE_SAMPLES - HEADER_SAMPLES
    if _already_finished(reception_start, finished_starts, epsilon_samples):
        return None, None
    return rb, reception_start


def _progress_frac(r) -> float:
    """Comparable progress fraction across a header-locked or
    blind-locked result, for picking "whichever branch is furthest
    along" when only one is usable this poll."""
    if isinstance(r, DemodResult):
        return r.frames_received / r.mode.n_frames
    return int(np.count_nonzero(r.weights)) / MODES["C"].n_latents


def decode_loop(ring: RingBuffer, model, state: SharedState, config: RxConfig,
                stop_event: threading.Event, sink=None):
    if sink is None:
        sink = SaveToDirSink(config.out_dir, config.size)
    modem = Modem()
    total_c_latents = MODES["C"].n_latents
    epsilon_samples = SAME_RECEPTION_EPSILON_S * FS
    # Absolute (ring-buffer-coordinate) transmission-start sample position
    # of every reception already saved this run. The ring buffer keeps
    # holding a finished reception's audio for up to --buffer-seconds
    # afterward, so without this a still-buffered transmission would be
    # rediscovered and re-decoded/re-saved on every following poll.
    finished_starts = deque(maxlen=50)
    last_progress_metric = -1
    stable_since = None
    current_reception_start = None
    # Which reception the progress counters above describe. Progress is
    # per-reception: carrying a previous transmission's metric across to
    # the next one can make a brand-new reception look like it has
    # already stalled and end it early.
    tracked_reception_start = None
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
        callsign = ""
        snr_db = float("nan")
        progress_frac = 0.0
        progress_metric = 0
        reception_start = None

        # Preamble path first: find and decode the strongest reception
        # that hasn't already been saved. Falls through to the blind path
        # below if nothing there decodes (corrupted header, or the only
        # hits are spurious correlation peaks).
        r, reception_start = _find_new_reception(
            modem, samples, to_baseband(samples), buf_start,
            finished_starts, epsilon_samples,
        )
        full_ok = r is not None

        if full_ok:
            latents_full = pad_to_full(r.latents)
            weights_full = pad_to_full(r.weights)
            mode_name = r.mode.name
            n_frames_expected = r.mode.n_frames
            frames_received = r.frames_received
            callsign = r.callsign
            snr_db = r.snr_db
            progress_frac = frames_received / n_frames_expected
            progress_metric = frames_received
        else:
            # Fold whatever's new since the last poll into the running
            # accumulator -- O(new samples), not O(window) -- which is
            # what lets this run one decay timescale per mode (see
            # RxConfig.blind_search_seconds) rather than a single
            # one-size-fits-all window. The retrospective decode below
            # still covers the whole current buffer once locked, exactly
            # as before.
            if blind_acc is None or blind_acc_pushed is None or blind_acc_pushed < buf_start:
                timescales = [min(m.duration_s, config.blind_search_seconds) for m in MODES.values()]
                blind_acc = BlindAccumulator(window_s=timescales)
                blind_acc_pushed = buf_start
            new_lo = blind_acc_pushed - buf_start
            if new_lo < len(samples):
                new_chunk = to_baseband_at(samples[new_lo:], blind_acc_pushed)
                blind_acc.push(new_chunk, blind_acc_pushed)
                blind_acc_pushed = total

            try:
                ba = blind_acc.result()
            except SyncError:
                ba = None
            try:
                rb = modem.demodulate_blind(samples, acquisition=ba) if ba is not None else None
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
                if _already_finished(reception_start, finished_starts, epsilon_samples):
                    continue
                latents_full = rb.latents
                weights_full = rb.weights
                callsign = rb.callsign
                snr_db = rb.snr_db
                # Confidently-received latents only, not a bare nonzero
                # count: demodulate_blind assigns *some* nonzero weight
                # to essentially every legal abs_frame slot its ever-
                # growing search range touches, real signal or not (just
                # small for noise, after the med_h fix above), so a
                # nonzero count keeps climbing every poll purely from
                # the buffer growing -- independent of whether any new
                # real data has arrived -- until buffer growth has
                # mapped the *entire* legal abs_frame range, which can
                # take up to a whole mode's duration after the real
                # transmission already ended. That read as "stuck
                # receiving forever": the stall detector below never saw
                # a stable metric to end_grace against. PROGRESS_WEIGHT_
                # THRESHOLD matches the "good" cutoff already used to
                # judge latents elsewhere (see tests/test_blind_
                # acquisition.py); only real frames clear it, so the
                # count stops climbing exactly when the real data does.
                progress_metric = int(np.count_nonzero(weights_full > PROGRESS_WEIGHT_THRESHOLD))
                progress_frac = progress_metric / total_c_latents
            else:
                reception_start = None

        decode_s = time.time() - t0

        with state.lock:
            state.last_decode_s = decode_s

        if latents_full is None:
            with state.lock:
                if state.status != "receiving":
                    state.status = "listening"
            last_progress_metric = -1
            stable_since = None
            current_reception_start = None
            continue

        # A different reception than the one the progress counters
        # describe: start its progress history fresh, or its very first
        # poll can be mistaken for a stalled (and so finished) one.
        if (
            tracked_reception_start is None
            or reception_start is None
            or abs(reception_start - tracked_reception_start) > epsilon_samples
        ):
            tracked_reception_start = reception_start
            last_progress_metric = -1
            stable_since = None

        current_reception_start = reception_start
        img = reconstruct(model, latents_full, weights_full)
        with state.lock:
            state.status = "receiving"
            state.mode_name = mode_name
            state.frames_received = frames_received
            state.n_frames_expected = n_frames_expected
            state.progress_frac = min(progress_frac, 1.0)
            state.callsign = callsign
            state.snr_db = snr_db
            state.image = img

        if n_frames_expected is not None:
            done = frames_received >= n_frames_expected
        else:
            # Deterministic backstop: the beacon already told us exactly
            # where this transmission's frame 0 sits (reception_start,
            # in the same "preamble start" coordinate the header path
            # uses), so the latest a real frame of it can possibly still
            # be arriving is mode C's own duration -- the longest mode --
            # after that point, fixed the moment reception_start is
            # known. That is unlike "progress stopped changing" below,
            # which rides on demodulate_blind's own noise floor and is
            # not guaranteed to ever settle: a stuck reception was
            # observed sitting in "receiving" for many minutes, far past
            # any mode's duration, because the buffer kept growing and
            # touching more of the blind search's legal frame range
            # (see PROGRESS_WEIGHT_THRESHOLD above) or some other reason
            # the metric kept moving. Once the buffer holds audio past
            # this deadline there is provably no more real signal left
            # to arrive for this reception, done or not.
            deadline_abs = (
                current_reception_start + PREAMBLE_SAMPLES + HEADER_SAMPLES
                + MODES["C"].n_frames * FRAME_SAMPLES
            )
            if progress_metric > 0 and progress_metric == last_progress_metric:
                stable_since = stable_since or time.time()
                done = (time.time() - stable_since) >= config.end_grace
            else:
                stable_since = None
                done = False
            done = done or total >= deadline_abs
        last_progress_metric = progress_metric

        if done and progress_metric > 0:
            saved_path = sink.on_reception(
                Reception(
                    image=img,
                    mode_name=mode_name,
                    callsign=callsign,
                    snr_db=snr_db,
                    frames_received=frames_received,
                    n_frames_expected=n_frames_expected,
                )
            )
            # Bookkeeping, not disk: this reception has been *handled*,
            # so it must never be rediscovered while its audio is still
            # in the buffer -- whether or not the sink chose to save it.
            if current_reception_start is not None:
                finished_starts.append(current_reception_start)
            with state.lock:
                state.status = "done"
                state.saved_path = saved_path

            if config.once:
                stop_event.set()
                break

            last_progress_metric = -1
            stable_since = None
            current_reception_start = None
            tracked_reception_start = None
            stop_event.wait(2.0)
            with state.lock:
                state.status = "listening"
                state.mode_name = None
                state.frames_received = None
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
                samples, search_s=(search[0] / FS, search[1] / FS)
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
        while not stop_event.is_set():
            _, total_now = ring.snapshot()
            if total_now >= frames_end_abs:
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
            r = modem.demodulate(samples, search_s=(lo / FS, len(samples) / FS))
        except SyncError:
            # transmission was cut short / corrupted after all; go back
            # to listening rather than crash the loop
            with state.lock:
                state.status = "listening"
            continue

        img = reconstruct(model, pad_to_full(r.latents), pad_to_full(r.weights))
        saved_path = sink.on_reception(
            Reception(
                image=img,
                mode_name=r.mode.name,
                callsign=r.callsign,
                snr_db=r.snr_db,
                frames_received=r.frames_received,
                n_frames_expected=r.mode.n_frames,
            )
        )
        with state.lock:
            state.status = "done"
            state.image = img
            state.frames_received = r.frames_received
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
            state.n_frames_expected = None
            state.progress_frac = 0.0
            state.callsign = ""
            state.snr_db = float("nan")


def decode_loop_diversity(rings, model, state: SharedState, config: RxConfig,
                          stop_event: threading.Event, sink=None, debug_sink=None):
    """Two-branch counterpart of `decode_loop`: independently finds and
    demodulates the same transmission from two `RingBuffer`s -- different
    antennas/audio devices, independent noise and fading, no assumption
    of phase lock between them -- then maximal-ratio combines the
    branches (`sstvae.modem.diversity.combine_diversity_results`) before
    reconstructing. `rings` is a 2-element sequence; the two are assumed
    to start filling at the same wall-clock moment (the caller opens
    both input streams together), so a reception position recorded in
    one ring's `total_written` coordinate is directly comparable to the
    other's without any shared sample timebase -- `SAME_RECEPTION_
    EPSILON_S` covers device startup jitter and the "within a frame
    time" independent-clock drift this is designed for (see
    docs/diversity-reception.md).

    Kept as its own function rather than folded into `decode_loop`:
    that function's state machine is explicitly load-bearing (CLAUDE.md)
    for the single-device slow tests, and this keeps it byte-for-byte
    unchanged for that case at the cost of some duplication here.

    Each branch independently prefers a header lock and falls back to a
    blind one, same as `decode_loop` does for a single receiver
    (`_find_branch_reception`) -- so a branch too weak for the preamble
    path can still contribute once enough audio has accumulated for a
    beacon superframe. Branches are matched by their *position*
    (`reception_start`, always expressed at the preamble's sample offset
    whichever path found it) within `SAME_RECEPTION_EPSILON_S`, the same
    criterion `decode_loop` itself uses -- for two blind locks this
    position agreement is a sanity check ("are these really the same
    transmission"), not an alignment requirement, since
    `BlindDemodResult.latents`/`.weights` are already aligned by the
    beacon's absolute frame counter regardless of sample position (see
    `combine_diversity_results`). If only one branch locks (the other
    never acquires at all, or its lock is more than
    `SAME_RECEPTION_EPSILON_S` away -- most likely a spurious hit),
    that branch is used alone, same erasure/weight semantics as
    `combine_diversity_results` given a single branch.

    Progress/completion tracking mirrors `decode_loop`'s own preference:
    a header-locked combine (the common case) reports an exact frame
    count and finishes when it is reached; an all-blind combine (true
    duration unknown) finishes when progress stops advancing for
    `config.end_grace` seconds, the same stall detection `decode_loop`
    uses for its own blind path.

    `debug_sink`, if given (a `SaveDebugImageToDirSink`), is handed
    `sstvae.modem.diversity.contribution_image([branch_a, branch_b])`
    for every finished reception where both branches actually
    contributed -- skipped when only one branch locked, since there is
    nothing to compare.
    """
    if len(rings) != 2:
        raise ValueError("decode_loop_diversity needs exactly two ring buffers")
    if sink is None:
        sink = SaveToDirSink(config.out_dir, config.size)
    modem = Modem()
    total_c_latents = MODES["C"].n_latents
    epsilon_samples = SAME_RECEPTION_EPSILON_S * FS
    finished_starts = deque(maxlen=50)
    last_progress_metric = -1
    stable_since = None
    current_reception_start = None
    # Which reception the progress counters above describe -- see
    # decode_loop's identical field for why this is tracked separately
    # from current_reception_start.
    tracked_reception_start = None

    while not stop_event.is_set():
        stop_event.wait(config.poll_interval)
        if stop_event.is_set():
            break

        snaps = [ring.snapshot() for ring in rings]
        seconds_captured = min(total for _, total in snaps) / FS
        with state.lock:
            state.seconds_captured = seconds_captured
        if any(len(samples) < MIN_SECONDS_BEFORE_ATTEMPT * FS for samples, _ in snaps):
            continue

        found_by_branch = [None, None]  # (result, reception_start) per ring, indexed
        for i, (samples, total) in enumerate(snaps):
            buf_start = total - len(samples)
            r, start = _find_branch_reception(
                modem, samples, buf_start, finished_starts, epsilon_samples,
                config.blind_search_seconds,
            )
            if r is not None:
                found_by_branch[i] = (r, start)

        combined = reception_start = None
        branch_results = []
        branch_a_locked = branch_b_locked = False
        if (found_by_branch[0] is not None and found_by_branch[1] is not None
                and abs(found_by_branch[0][1] - found_by_branch[1][1]) <= epsilon_samples):
            branch_results = [found_by_branch[0][0], found_by_branch[1][0]]
            combined = combine_diversity_results(branch_results)
            reception_start = found_by_branch[0][1]
            branch_a_locked = branch_b_locked = True
        elif found_by_branch[0] is not None or found_by_branch[1] is not None:
            # Only one branch locked, or the two locks are further apart
            # than a same-transmission match tolerates (a spurious hit on
            # one branch, most likely). Use whichever branch is furthest
            # along, same as an operator picking the stronger antenna.
            found = [(i, hit) for i, hit in enumerate(found_by_branch) if hit is not None]
            best_i, (combined, reception_start) = max(
                found, key=lambda x: _progress_frac(x[1][0])
            )
            branch_results = [combined]
            branch_a_locked = best_i == 0
            branch_b_locked = best_i == 1

        with state.lock:
            state.branch_a_locked = branch_a_locked
            state.branch_b_locked = branch_b_locked

        if combined is None:
            with state.lock:
                if state.status != "receiving":
                    state.status = "listening"
            last_progress_metric = -1
            stable_since = None
            current_reception_start = None
            continue

        headered = isinstance(combined, DemodResult)
        if headered:
            latents_full = pad_to_full(combined.latents)
            weights_full = pad_to_full(combined.weights)
            mode_name = combined.mode.name
            n_frames_expected = combined.mode.n_frames
            frames_received = combined.frames_received
            progress_frac = frames_received / n_frames_expected
            progress_metric = frames_received
        else:
            latents_full = combined.latents
            weights_full = combined.weights
            mode_name = None
            n_frames_expected = None
            frames_received = None
            progress_metric = int(np.count_nonzero(weights_full))
            progress_frac = progress_metric / total_c_latents
        callsign = combined.callsign
        snr_db = combined.snr_db

        if (tracked_reception_start is None or reception_start is None
                or abs(reception_start - tracked_reception_start) > epsilon_samples):
            tracked_reception_start = reception_start
            last_progress_metric = -1
            stable_since = None

        current_reception_start = reception_start
        img = reconstruct(model, latents_full, weights_full)
        with state.lock:
            state.status = "receiving"
            state.mode_name = mode_name
            state.frames_received = frames_received
            state.n_frames_expected = n_frames_expected
            state.progress_frac = min(progress_frac, 1.0)
            state.callsign = callsign
            state.snr_db = snr_db
            state.image = img

        if n_frames_expected is not None:
            done = frames_received >= n_frames_expected
        else:
            if progress_metric > 0 and progress_metric == last_progress_metric:
                stable_since = stable_since or time.time()
                done = (time.time() - stable_since) >= config.end_grace
            else:
                stable_since = None
                done = False
        last_progress_metric = progress_metric

        if not (done and progress_metric > 0):
            continue

        saved_path = sink.on_reception(
            Reception(
                image=img,
                mode_name=mode_name,
                callsign=callsign,
                snr_db=snr_db,
                frames_received=frames_received,
                n_frames_expected=n_frames_expected,
            )
        )
        if debug_sink is not None and len(branch_results) == 2:
            debug_sink.on_contribution_image(
                contribution_image(branch_results), saved_path
            )
        if current_reception_start is not None:
            finished_starts.append(current_reception_start)
        with state.lock:
            state.status = "done"
            state.saved_path = saved_path

        if config.once:
            stop_event.set()
            break

        last_progress_metric = -1
        stable_since = None
        current_reception_start = None
        tracked_reception_start = None
        stop_event.wait(2.0)
        with state.lock:
            state.status = "listening"
            state.mode_name = None
            state.frames_received = None
            state.n_frames_expected = None
            state.progress_frac = 0.0
            state.callsign = ""
            state.snr_db = float("nan")

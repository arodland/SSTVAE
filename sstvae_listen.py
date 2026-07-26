#!/usr/bin/env python3
"""Continuously listen on an audio input device for an SSTVAE
transmission and decode it, including the case where listening starts
mid-transmission.

    python sstvae_listen.py --out-dir received
    python sstvae_listen.py --list-devices
    python sstvae_listen.py --device pulse --no-gui

Keeps a rolling buffer of the last --buffer-seconds of audio (long
enough to cover a full mode-C transmission). Every --poll-interval
seconds it tries to demodulate the whole buffer: first the normal
preamble-based path (Modem.demodulate, works if the buffer happens to
contain the transmission's start), then falling back to the
preamble-free blind path (Modem.demodulate_blind, works from any
long-enough mid-transmission excerpt via the beacon side-channel --
see sstvae/modem/beacon.py). Because the buffer holds history from
before sync was acquired, a mid-stream lock still decodes the frames
that arrived before it -- retrospective decoding, not just
from-here-on.

Reception is considered finished either when a fully-synced decode
reports all its frames received, or (blind case, true mode/duration
unknown) when decoded progress stops advancing for --end-grace
seconds. The image is saved and the listener goes back to waiting for
the next transmission.

--low-cpu drops the blind fallback (and with it, retrospective
mid-stream decoding) in exchange for much lower idle CPU use: it only
ever looks for the preamble, restricts that search to the audio that's
newly arrived since the last poll (instead of rescanning the whole
buffer), and once the header locks it just sleeps until the whole
transmission has been captured and decodes it once, rather than
repeatedly re-decoding for progress updates.

Requires sounddevice (PortAudio): pip install -e .[listen]
"""

import argparse
import queue
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from sstvae.config import FS, FRAME_SAMPLES, HEADER_SAMPLES, LATENTS_PER_FRAME, MODES, PREAMBLE_SAMPLES
from sstvae.modem import Modem, SyncError
from sstvae.modem.dsp import to_baseband
from sstvae.modem.sync import acquire as sync_acquire
from sstvae_decode import pad_to_full, reconstruct
from sstvae_encode import MODEL_HELP, load_model

MIN_SECONDS_BEFORE_ATTEMPT = 3.0
# How close two acquisitions' transmission-start sample position must be
# to be treated as "the same reception" rather than a new one.
SAME_RECEPTION_EPSILON_S = 1.0


class RingBuffer:
    """Fixed-length circular float64 audio buffer, thread-safe."""

    def __init__(self, seconds: float, fs: int = FS):
        self.n = int(seconds * fs)
        self.buf = np.zeros(self.n, dtype=np.float64)
        self.write_pos = 0
        self.total_written = 0
        self.lock = threading.Lock()

    def write(self, chunk: np.ndarray) -> None:
        chunk = np.asarray(chunk, dtype=np.float64).reshape(-1)
        with self.lock:
            n = len(chunk)
            if n >= self.n:
                self.buf[:] = chunk[-self.n :]
                self.write_pos = 0
            else:
                end = self.write_pos + n
                if end <= self.n:
                    self.buf[self.write_pos : end] = chunk
                else:
                    k = self.n - self.write_pos
                    self.buf[self.write_pos :] = chunk[:k]
                    self.buf[: end - self.n] = chunk[k:]
                self.write_pos = end % self.n
            self.total_written += n

    def snapshot(self) -> tuple[np.ndarray, int]:
        """Chronological copy of everything currently held (oldest
        first), and the total sample count ever written (for display)."""
        with self.lock:
            if self.total_written < self.n:
                valid = self.buf[: self.total_written].copy()
            else:
                valid = np.concatenate([self.buf[self.write_pos :], self.buf[: self.write_pos]])
            total = self.total_written
        return valid, total


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

    def __post_init__(self):
        self.lock = threading.Lock()


def open_input_stream(device, samplerate, ring: RingBuffer):
    """Open a sounddevice InputStream feeding `ring`. Tries the
    requested sample rate directly first (PulseAudio/most backends
    resample transparently); falls back to the device's default rate
    with polyphase resampling per-chunk if that's rejected."""
    import sounddevice as sd

    def make_callback(resample_fn=None):
        def callback(indata, frames, time_info, status):
            if status:
                print(f"[audio] {status}", file=sys.stderr)
            mono = indata[:, 0] if indata.ndim > 1 else indata
            ring.write(resample_fn(mono) if resample_fn else mono)

        return callback

    try:
        stream = sd.InputStream(
            samplerate=samplerate, channels=1, dtype="float32",
            device=device, callback=make_callback(),
        )
        stream.start()
        return stream, samplerate
    except Exception as e:
        dev_info = sd.query_devices(device, "input")
        native = int(round(dev_info["default_samplerate"]))
        print(
            f"[audio] {samplerate} Hz rejected ({e}); falling back to "
            f"device default {native} Hz with resampling",
            file=sys.stderr,
        )
        from math import gcd

        from scipy.signal import resample_poly

        g = gcd(samplerate, native)
        up, down = samplerate // g, native // g

        def resample_fn(x):
            return resample_poly(x, up, down)

        stream = sd.InputStream(
            samplerate=native, channels=1, dtype="float32",
            device=device, callback=make_callback(resample_fn),
        )
        stream.start()
        return stream, native


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


def _timestamped_path(out_dir: str) -> Path:
    """Unique output path. Millisecond resolution because two short-mode
    receptions can be finished within the same second (both already sat
    complete in the buffer), and a second-resolution name would have the
    later one silently overwrite the earlier."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    return Path(out_dir) / f"rx_{ts}.png"


def decode_loop(ring: RingBuffer, model, state: SharedState, args, stop_event: threading.Event):
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

    while not stop_event.is_set():
        stop_event.wait(args.poll_interval)
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
            # Bound where blind acquisition searches (the dominant CPU
            # cost of this path) to the most recent slice of the buffer;
            # the retrospective decode still covers everything once
            # locked. Whole buffer if it's shorter than the window.
            blind_span = int(args.blind_search_seconds * FS)
            blind_search = (
                None
                if len(samples) <= blind_span
                else ((len(samples) - blind_span) / FS, len(samples) / FS)
            )
            try:
                rb = modem.demodulate_blind(samples, search_s=blind_search)
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
                progress_metric = int(np.count_nonzero(weights_full))
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
            if progress_metric > 0 and progress_metric == last_progress_metric:
                stable_since = stable_since or time.time()
                done = (time.time() - stable_since) >= args.end_grace
            else:
                stable_since = None
                done = False
        last_progress_metric = progress_metric

        if done and progress_metric > 0:
            out_path = _timestamped_path(args.out_dir)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            saved = img
            if args.size:
                w, h = args.size.lower().split("x")
                saved = saved.resize((int(w), int(h)))
            saved.save(out_path)
            if current_reception_start is not None:
                finished_starts.append(current_reception_start)
            print(
                f"saved {out_path} (mode={mode_name or 'unknown, blind sync'}, "
                f"callsign={callsign or '(none)'}{_fmt_snr(snr_db)})"
            )
            with state.lock:
                state.status = "done"
                state.saved_path = str(out_path)

            if args.once:
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


def decode_loop_low_cpu(
    ring: RingBuffer, model, state: SharedState, args, stop_event: threading.Event
):
    """Header-only variant: no blind fallback, no retrospective decode.
    While idle, only searches the newly-arrived slice of audio each poll
    (not the whole buffer) for the preamble. Once it locks, it does no
    further signal processing at all until enough audio has been
    captured for the whole transmission, then decodes and saves once."""
    modem = Modem()
    search_overlap_s = 2.0  # margin so a preamble can't be missed by
    # straddling the boundary between one poll's search window and the next
    last_search_pos = 0  # abs sample position (ring.total_written coord)

    while not stop_event.is_set():
        stop_event.wait(args.poll_interval)
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
            # found -- see _acquire_unfinished's docstring.
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
            stop_event.wait(min(1.0, args.poll_interval))
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
        out_path = _timestamped_path(args.out_dir)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        saved = img
        if args.size:
            w, h = args.size.lower().split("x")
            saved = saved.resize((int(w), int(h)))
        saved.save(out_path)
        print(
            f"saved {out_path} (mode={r.mode.name}, frames={r.frames_received}/"
            f"{r.mode.n_frames}, callsign={r.callsign or '(none)'}{_fmt_snr(r.snr_db)})"
        )
        with state.lock:
            state.status = "done"
            state.image = img
            state.frames_received = r.frames_received
            state.progress_frac = min(r.frames_received / r.mode.n_frames, 1.0)
            state.snr_db = r.snr_db
            state.saved_path = str(out_path)

        if args.once:
            stop_event.set()
            break

        time.sleep(2.0)
        with state.lock:
            state.status = "listening"
            state.mode_name = None
            state.frames_received = None
            state.n_frames_expected = None
            state.progress_frac = 0.0
            state.callsign = ""
            state.snr_db = float("nan")


def _fmt_snr(snr_db: float) -> str:
    if snr_db != snr_db:  # NaN
        return ""
    return f"  SNR {snr_db:.1f}dB"


def run_gui(state: SharedState, stop_event: threading.Event):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    fig, ax = plt.subplots(figsize=(6, 5))
    placeholder = np.zeros((30, 40, 3))
    im = ax.imshow(placeholder)
    ax.axis("off")
    title = ax.set_title("listening...")
    fig.canvas.mpl_connect("close_event", lambda e: stop_event.set())

    def update(_frame):
        with state.lock:
            status = state.status
            img = state.image
            mode_name = state.mode_name
            frames_received = state.frames_received
            n_frames_expected = state.n_frames_expected
            progress_frac = state.progress_frac
            callsign = state.callsign
            snr_db = state.snr_db
            saved_path = state.saved_path
            seconds_captured = state.seconds_captured

        if img is not None:
            im.set_data(np.asarray(img))
            im.set_extent((0, img.width, img.height, 0))

        if status == "listening":
            txt = f"listening... ({seconds_captured:.0f}s captured)"
        elif status == "receiving":
            if n_frames_expected is not None:
                txt = (
                    f"receiving mode {mode_name}: frame "
                    f"{frames_received}/{n_frames_expected} "
                    f"({100 * progress_frac:.0f}%)"
                )
            else:
                txt = f"receiving (blind sync): {100 * progress_frac:.0f}% of latents"
            txt += _fmt_snr(snr_db)
            if callsign:
                txt += f"  de {callsign}"
        else:
            txt = f"done -- saved {saved_path}" + _fmt_snr(snr_db)
        title.set_text(txt)

        if stop_event.is_set():
            plt.close(fig)
        return im, title

    anim = FuncAnimation(fig, update, interval=500, cache_frame_data=False)
    plt.show()
    stop_event.set()


def run_console(state: SharedState, stop_event: threading.Event):
    last_status = None
    while not stop_event.is_set():
        with state.lock:
            status = state.status
            mode_name = state.mode_name
            frames_received = state.frames_received
            n_frames_expected = state.n_frames_expected
            progress_frac = state.progress_frac
            callsign = state.callsign
            snr_db = state.snr_db
            seconds_captured = state.seconds_captured
            saved_path = state.saved_path

        if status == "listening":
            line = f"listening... ({seconds_captured:.0f}s captured)"
        elif status == "receiving":
            if n_frames_expected is not None:
                line = (
                    f"receiving mode {mode_name}: frame "
                    f"{frames_received}/{n_frames_expected} ({100 * progress_frac:.0f}%)"
                )
            else:
                line = f"receiving (blind sync): {100 * progress_frac:.0f}% of latents"
            line += _fmt_snr(snr_db)
            if callsign:
                line += f"  de {callsign}"
        else:
            line = f"done -- saved {saved_path}" + _fmt_snr(snr_db)

        if line != last_status:
            print(line)
            last_status = line
        stop_event.wait(1.0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=None, help=MODEL_HELP)
    ap.add_argument("--out-dir", default="received", help="directory for saved images")
    ap.add_argument("--device", default=None, help="input device name/index (see --list-devices)")
    ap.add_argument("--samplerate", type=int, default=FS, help="capture sample rate to request")
    ap.add_argument(
        "--buffer-seconds", type=float, default=130.0,
        help="rolling audio buffer length; must exceed the longest mode duration "
        "(mode C is ~95s) with margin for retrospective decode",
    )
    ap.add_argument("--poll-interval", type=float, default=5.0, help="seconds between decode attempts")
    ap.add_argument(
        "--blind-search-seconds", type=float, default=25.0,
        help="how much of the buffer's most recent audio the blind CFO/timing "
        "search scans, rather than the whole --buffer-seconds window. Must "
        "exceed MIN_FRAMES_FOR_SYNC's ~10.5s with margin; the retrospective "
        "decode itself still covers the full buffer once locked, this only "
        "bounds where acquisition looks (the dominant CPU cost of the blind "
        "path).",
    )
    ap.add_argument(
        "--end-grace", type=float, default=8.0,
        help="seconds of no further progress (blind-sync case only, true length unknown) "
        "before a reception is considered finished",
    )
    ap.add_argument("--size", default=None, help="resize saved image, e.g. 320x240")
    ap.add_argument("--no-gui", action="store_true", help="print status instead of a matplotlib window")
    ap.add_argument("--once", action="store_true", help="exit after the first successful reception")
    ap.add_argument("--list-devices", action="store_true", help="list audio devices and exit")
    ap.add_argument(
        "--low-cpu", action="store_true",
        help="header-sync only: no blind fallback, no retrospective mid-stream "
        "decode. Searches only newly-arrived audio each poll instead of the "
        "whole buffer, and decodes once at the end of a locked reception "
        "instead of repeatedly for progress updates.",
    )
    args = ap.parse_args()

    if args.list_devices:
        import sounddevice as sd

        print(sd.query_devices())
        return

    model = load_model(args.model)
    ring = RingBuffer(args.buffer_seconds)
    state = SharedState()
    stop_event = threading.Event()

    stream, actual_rate = open_input_stream(args.device, args.samplerate, ring)
    print(f"listening at {actual_rate} Hz, buffer {args.buffer_seconds:.0f}s -- Ctrl+C to stop")

    target = decode_loop_low_cpu if args.low_cpu else decode_loop
    worker = threading.Thread(target=target, args=(ring, model, state, args, stop_event), daemon=True)
    worker.start()

    try:
        if args.no_gui:
            run_console(state, stop_event)
        else:
            run_gui(state, stop_event)
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        stream.stop()
        stream.close()
        worker.join(timeout=2.0)


if __name__ == "__main__":
    main()

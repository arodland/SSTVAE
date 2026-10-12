"""The live QRSSTVAE listener: audio in, a tile per signal out (spec 9, build step 6).

Not a format module. `qrss_listen.py` is its command line, and the
desktop app's QRSS window is one of its front ends: the app pipes its
capture audio in and shows the tiles this writes.

    8 kHz audio chunks --StreamFE--> FE 4 kHz --> FeRing (in memory, ~33 min)
                                              \\-> PassbandStore (48 h, optional)
    every refresh_s (every empty_refresh_s for a slot where nothing has
    been found by detect_until_s), for each slot whose frame is on the air:
        slot_capture: the slot's FE so far, blanked and normalised, the
                      rest (future, and any holes) noise, erased
        detect on the preamble (until detect_until_s; after it, until
        recheck_until_s, only the preamble hits gate V refused), then
        receive_pass, round A only, with the unheard spans erased
        --> tile: header, SNR, share received, picture so far
    when a slot's frame has ended (in a process of its own, beside the live ones):
        receive_slot over the whole slot (round B, whole-slot search),
        associate into the multi-pass store, render the accumulator
        --> the tile goes "complete"
    (the tile reads "finishing" while that runs, and one more live
    receive of the whole frame brings it to the pass's end meanwhile)

The tiles are a directory: `state.json` (rewritten atomically after
every change) and `tiles/<id>.png`. Anything can show them; nothing in
here knows about a GUI.

**Both waveforms.** `LiveConfig.waveforms` says which are listened for,
each in the shape of `LiveConfig.spec` (`lframe.twin`: "full" and
"l-full"). With L among them the receiver is handed the tuple of frames
(`LiveConfig.selection`), and acquires, gates and receives every signal
as its own waveform; the slot is read, blanked and normalised once for
both, and lasts as long as the longer frame. One signal is one tile
whichever waveform it is read as: detections, passes, tiles and hints
are the same signal when their band centres (`receiver.centre_hz`; an L
pilot sits 9.5 Hz above its band's) are within MERGE_HZ, and where both
waveforms lock on one signal the larger Z_ref says which it is. With CE
alone (no `waveforms`, or just CE) the receiver is handed the CE frame
itself and every call is the one made before L existed.

**The picture so far.** A pass in progress is received like a finished
one with every symbol not yet heard erased (`receive_pass(erase_s=)`),
so its W is honest about what arrived, and the tile is the store's
accumulator for that picture (earlier passes) plus this pass. Latents
are interleaved over the whole frame, so the whole picture sharpens at
once. v5 was not trained for a fraction of a group: a recognizable
picture a fifth of the way through is luck (spec 11, 12).

**Clock.** Audio has no timestamps here: a chunk is stamped when it is
read, its last sample at "now". Samples are numbered on the absolute
8 kHz grid round(t * 8000), so FE sample n is unix time n / 4000 (the
passband store's convention). Small differences between the sound
card's clock and the computer's are absorbed; when the two disagree by
more than RESYNC_S the other way -- samples arriving later than their
count says, which is audio that was never sent (the app stopped feeding
while it transmitted, or a capture stalled) -- the stream restarts at
the new time and the missing stretch stays a hole, erased in every pass
it touches. Digital silence (SILENCE_S of samples under SILENCE_LEVEL,
which is what a WebSDR sends once it has dropped the session) is a hole
too: nothing was heard there. Samples arriving early (a burst after a stall) are only
believed after EARLY_WINDOW_S of nothing but: a pipe loses nothing, so
early audio is the sound card's clock running fast, which is slow. A
replayed file (`WavSource`) is stamped from its start time instead, so a
recording can be replayed faster than real time.
"""

from __future__ import annotations

import collections
import copy
import json
import math
import os
import queue
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import acquire, frontend, lframe, picture, receiver
from .constants import (CARRIER_BAND_HZ, FE_FS, FS, L_EDGE_HZ, L_PILOT_HZ, WAVEFORM_CE,
                        WAVEFORM_L, WAVEFORM_NAMES)
from .frame import FULL, FrameSpec
from .frontend import PB_LEAD_S, PB_TAIL_S, slot_t0
from .types import Detection, PassResult

FE_DECIM = FS // FE_FS            # 2
ALIGN = 16                        # audio blocks start on multiples of 16 (to_baseband's table)
HALO = 128                        # audio samples each side of a block (the FE FIR is 255 taps)
RESYNC_S = 1.0                    # clock disagreement that restarts the stream
STAMP_HOLD_S = 5.0                # stdin audio arriving late is held this long (backlog or gap?)
EARLY_WINDOW_S = 30.0             # ...when audio has run ahead of the clock this long
RING_S = 2100.0                   # FE kept in memory: a FULL slot, its margins and slack
# Audio this quiet this long is a source sending nothing (a WebSDR that
# has dropped the session, a muted stream), not a quiet band: no sound
# card hears below a 24-bit step. It is kept as unheard.
SILENCE_LEVEL = 2.0 ** -24
SILENCE_S = 1.0
MERGE_HZ = receiver.MERGE_HZ      # a detection this close to a tile is that tile
PREAMBLE_S = 20                   # the preamble's span after t0 (660 symbols, 20.0 s)
CAPTURE_MARGIN_S = 2.0            # blanker and AGC run this far past the newest audio
FINISH_SEARCH_MIN_HEARD = 0.5     # the end-of-slot search over the whole slot needs this much heard
FINISH_MAX_CANDIDATES = 12        # ...and verifies at most this many candidates
FINISH_BUDGET_S = 180.0           # ...for at most this long (a false one costs ~1 min)
HEADERLESS_MAX_SNR_DB = -6.0      # a pass this strong whose header did not decode is not QRSSTVAE
RECOVER_HOURS = 3.0               # a restarted listener finishes slots that ended this recently
GUESS_NOTE = "no header yet: picture assumes the first pass of a mode A send"
DIAL_FILE = "dial_changes.jsonl"  # the radio's dial over time, in the state directory
LOG_FILE = "listener.log"         # every log line, in the state directory...
LOG_FILE_BYTES = 2_000_000        # ...moved to listener.log.1 past this size (one kept)
# unheard before and after a dial change: settling, and the app's optional
# tune 5 s after it, which takes up to 45 s on Jeff's tuner (rarely)
DIAL_GUARD_S = (2.0, 50.0)
L_BELOW_HZ = float(L_PILOT_HZ + L_EDGE_HZ)   # an L signal's data reach this far below its pilot
_WAVEFORM_OF = {name: w for w, name in WAVEFORM_NAMES.items()}


def utc_iso(t: float) -> str:
    return datetime.fromtimestamp(t, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def frame_seconds(spec) -> float:
    """Seconds from t0 to the end of the frame's last keyed symbol (either
    waveform: positions are in the spec's own symbols); for a tuple of
    frames the longest's, which is how long the listener's slot lasts."""
    if isinstance(spec, (tuple, list)):
        return max(frame_seconds(s) for s in spec)
    return spec.keyed_end_pos * spec.t_sym


def in_band(f_hz: float, waveform: int = WAVEFORM_CE) -> bool:
    """Whether a signal named at f_hz lies inside CARRIER_BAND_HZ: a CE
    carrier, or an L pilot and the data L_BELOW_HZ under it (`tx.check_pilot`)."""
    lo = f_hz - L_BELOW_HZ if waveform == WAVEFORM_L else f_hz
    return CARRIER_BAND_HZ[0] <= lo and f_hz <= CARRIER_BAND_HZ[1]


def centre_of(f_hz: float, waveform) -> float:
    """`receiver.centre_hz` for what is not a Detection or a pass (a tile,
    a hint): the band centre of a signal named at f_hz in `waveform` (the
    number, or its name in WAVEFORM_NAMES)."""
    return f_hz + receiver.L_EXCLUDE_OFFSET_HZ if waveform in (WAVEFORM_L, "L") else f_hz


def _waveform_id(w) -> int:
    """A waveform's number from the number or its name (a tile's, or a stored
    pass's); a name that is not one is CE, which is what a tile without one is."""
    return _WAVEFORM_OF.get(w, WAVEFORM_CE) if isinstance(w, str) else int(w)


def _due_waveform(d: dict) -> int:
    """The waveform of a pass `store.due` names: its frame's ("l-..." is L),
    else its "waveform", else CE (a pass stored before L)."""
    try:
        return int(lframe.get_any(d["frame"]).waveform)
    except (KeyError, ValueError, TypeError, AttributeError):
        return _waveform_id(d.get("waveform", WAVEFORM_CE))


# --- audio -> FE, streaming ----------------------------------------------------------------

class StreamFE:
    """`frontend.audio_to_fe` over a stream of chunks, sample for sample.

    Audio sample a (absolute, on the 8 kHz grid) becomes FE sample a / 2.
    A block is converted with HALO samples of context each side, so the
    output equals the batch conversion away from the stream's first and
    last HALO samples, and blocks start on multiples of ALIGN so the
    heterodyne's phase table runs on unbroken. `push` returns the FE
    produced, as (first FE index, samples); a chunk that does not follow
    on from the last one restarts the stream.
    """

    def __init__(self, block: int = 8000):
        self.block = max(ALIGN, block // ALIGN * ALIGN)
        self.a0 = None                   # absolute audio index of pend[0]
        self.pend = np.zeros(0, dtype=np.float64)

    @property
    def next_index(self) -> int | None:
        """The audio index the next chunk must start at to follow on."""
        return None if self.a0 is None else self.a0 + len(self.pend)

    def push(self, a: int, x) -> list[tuple[int, np.ndarray]]:
        x = np.asarray(x, dtype=np.float64)
        if self.a0 is None or a != self.next_index:
            # (re)start on the grid: drop the samples before the next multiple of ALIGN
            skip = (-a) % ALIGN
            self.a0 = a + skip
            self.pend = x[skip:].copy()
        else:
            self.pend = np.concatenate([self.pend, x])
        out = []
        while len(self.pend) >= 2 * HALO + ALIGN:
            n = min(self.block, (len(self.pend) - 2 * HALO) // ALIGN * ALIGN)
            seg = self.pend[:n + 2 * HALO]
            fe = frontend.audio_to_fe(seg)
            k0 = HALO // FE_DECIM
            out.append(((self.a0 + HALO) // FE_DECIM, fe[k0:k0 + n // FE_DECIM]))
            self.pend = self.pend[n:]
            self.a0 += n
        return out


class FeRing:
    """The last `span_s` of FE in memory, addressed by absolute FE index (t * 4000)."""

    def __init__(self, span_s: float = RING_S):
        self.n = int(span_s * FE_FS)
        self.x = np.zeros(self.n, dtype=np.complex64)
        self.ok = np.zeros(self.n, dtype=bool)
        self.end = None                  # one past the newest index written

    def write(self, n0: int, x) -> None:
        x = np.asarray(x, dtype=np.complex64)
        if len(x) > self.n:
            n0, x = n0 + len(x) - self.n, x[-self.n:]
        if self.end is not None and n0 > self.end:
            self._clear(self.end, n0)     # a hole: what was there is older than the ring
        i = np.arange(n0, n0 + len(x)) % self.n
        self.x[i] = x
        self.ok[i] = True
        self.end = max(self.end or 0, n0 + len(x))

    def _clear(self, a: int, b: int) -> None:
        b = min(b, a + self.n)
        i = np.arange(a, b) % self.n
        self.x[i] = 0
        self.ok[i] = False

    def read(self, a: int, b: int) -> tuple[np.ndarray, np.ndarray]:
        """(x, written) for FE indices [a, b); anything not held reads as unwritten."""
        x = np.zeros(b - a, dtype=np.complex64)
        ok = np.zeros(b - a, dtype=bool)
        if self.end is None:
            return x, ok
        lo, hi = max(a, self.end - self.n), min(b, self.end)
        if hi > lo:
            i = np.arange(lo, hi) % self.n
            x[lo - a:hi - a] = self.x[i]
            ok[lo - a:hi - a] = self.ok[i]
        return x, ok


def _spans(mask: np.ndarray, fs: float, origin: float) -> list[tuple[float, float]]:
    """(start, end) times of the False runs of mask, sample k at (k - origin) / fs."""
    if mask.all():
        return []
    m = np.r_[True, mask, True].astype(np.int8)
    d = np.diff(m)
    starts = np.flatnonzero(d == -1)
    ends = np.flatnonzero(d == 1)
    return [((s - origin) / fs, (e - origin) / fs) for s, e in zip(starts, ends)]


def slot_capture(ring: FeRing, q: int, spec, seed: int | None = None,
                 lead_s: float = PB_LEAD_S, tail_s: float = PB_TAIL_S):
    """(Prepared, erase spans, share of the frame heard) for slot q from the ring.

    `spec` is a frame, or a tuple of frames (`LiveConfig.selection`): the
    slot lasts as long as the longest (`frame_seconds`).

    The heard samples are blanked and normalised as `receiver.prepare`
    would; everything not heard (the future, and holes) is complex noise
    at the normalised level (median power 1) and is returned as spans to
    erase, in seconds after t0. It is `capture_read` (quick: the part a
    worker does holding the listener's lock) then `capture_prepare`.
    """
    return capture_prepare(*capture_read(ring, q, spec, lead_s, tail_s), q, spec, seed)


def capture_read(ring, q: int, spec, lead_s: float = PB_LEAD_S,
                 tail_s: float = PB_TAIL_S) -> tuple[np.ndarray, np.ndarray, int]:
    """(FE, written, index of t0 in them): slot q as read from the ring (or
    anything with its `read`, such as the passband store)."""
    t0 = slot_t0(q)
    a = t0 * FE_FS - int(round(lead_s * FE_FS))
    b = t0 * FE_FS + int(round((frame_seconds(spec) + tail_s) * FE_FS))
    x, ok = ring.read(a, b)
    return x, ok, t0 * FE_FS - a


def capture_prepare(x, ok, origin: int, q: int, spec, seed: int | None = None):
    """`slot_capture` from what `capture_read` returned."""
    # The blanker and the inverse AGC look at most about 0.6 s either
    # side of a sample, and everything after the newest heard sample is
    # replaced by noise below, so they run only up to CAPTURE_MARGIN_S past
    # it: the same capture, bit for bit, without working through the
    # unheard rest of the slot at every refresh.
    h = int(np.flatnonzero(ok)[-1]) + 1 if ok.any() else 0
    m = min(len(x), h + int(CAPTURE_MARGIN_S * FE_FS))
    fe = np.zeros(len(x), dtype=np.complex64)
    keep = np.ones(len(x), dtype=np.float32)
    fe[:m], keep[:m] = frontend.blank(x[:m])
    keep = keep * ok
    fe[:m], _ = frontend.normalise(fe[:m], keep[:m])
    rng = np.random.default_rng(q if seed is None else seed)
    miss = ~ok
    nm = int(miss.sum())
    if nm:
        s = math.sqrt(0.5 / math.log(2.0))   # E|n|^2 = 1/ln 2: median power 1, like normalise
        fe[miss] = (s * (rng.standard_normal(nm) + 1j * rng.standard_normal(nm))).astype(np.complex64)
        keep = np.where(miss, 1.0, keep).astype(np.float32)
    prep = receiver.Prepared(fe=fe.astype(np.complex64), keep=keep.astype(np.float32), q=int(q),
                             t0_index=float(origin), fs=FE_FS)
    k1 = min(len(ok), origin + int(round(frame_seconds(spec) * FE_FS)))
    heard = float(np.mean(ok[origin:k1])) if k1 > origin else 0.0
    return prep, _spans(ok, FE_FS, origin), heard


def finish_receive(x, ok, origin: int, q: int, spec, estimator: str,
                   dets: list, hints=()) -> tuple[list[PassResult], float]:
    """(passes, share heard): the end-of-slot receive of slot q from its
    `capture_read`.

    The search over the whole slot (for signals too weak to see on the
    preamble) runs only when most of the slot was heard, and verifies at
    most FINISH_MAX_CANDIDATES candidates for FINISH_BUDGET_S, strongest
    first: on a phone band, voices and carriers make candidates that each
    cost about a minute to reject, and the search ran for longer than a
    slot. With less heard, only `dets`, the signals already found on the
    preamble, are received, and whatever is found at the `hints`
    (`LiveListener._hints`), which are among the candidates either way.

    `spec` is what the listener hands the receiver (`LiveConfig.selection`):
    the CE frame, or a tuple of frames, whose candidates the receiver puts
    in one order across both waveforms (preamble hits, then hints, then
    the whole-slot search's lines by their shape), so one budget serves
    both; the signals known already are then skipped by their band centres.
    """
    k1 = min(len(ok), origin + int(round(frame_seconds(spec) * FE_FS)))
    if not ok[origin:k1].any() or (not dets and not hints
                                   and np.mean(ok[origin:k1]) < FINISH_SEARCH_MIN_HEARD):
        return [], float(np.mean(ok[origin:k1])) if k1 > origin else 0.0   # nothing to receive
    prep, erase, heard = capture_prepare(x, ok, origin, q, spec)
    passes = []
    if heard >= FINISH_SEARCH_MIN_HEARD:
        passes = receiver.receive_slot(prep, spec, estimator=estimator, erase_s=erase,
                                       max_candidates=FINISH_MAX_CANDIDATES,
                                       budget_s=FINISH_BUDGET_S, hints=hints)
    elif (dets or hints) and heard > 0.0:
        more = []
        if hints:
            known = ([receiver.centre_hz(d) for d in dets] if receiver.is_joint(spec)
                     else [d.f_hz for d in dets])
            more = receiver.detect(prep, spec, live_only=True, erase_s=erase, hints=hints,
                                   skip_hz=known,
                                   max_candidates=FINISH_MAX_CANDIDATES,
                                   budget_s=FINISH_BUDGET_S)
        passes = receiver.receive_slot(prep, spec, estimator=estimator, erase_s=erase,
                                       dets=list(dets) + more)
    return passes, heard


class _FinishProcess:
    """A child process that runs `finish_receive`, so the end-of-slot
    receive (minutes, on a busy band) runs beside the live receives
    instead of holding them up.

    A process rather than a thread because the receiver's tracker is a
    Python loop, which two threads cannot run at once. It is started on
    first use (spawned: none of this process's threads or locks come
    with it), started again if it dies, runs at a lower priority than the
    live receives, and exits by itself when this process does.
    """

    def __init__(self):
        import multiprocessing

        self._ctx = multiprocessing.get_context("spawn")
        self._proc = None
        self._conn = None

    def call(self, args: tuple):
        if self._proc is None or not self._proc.is_alive():
            self.close()
            ours, theirs = self._ctx.Pipe()
            self._proc = self._ctx.Process(target=_finish_process_main, args=(theirs,),
                                           name="qrss-finish", daemon=True)
            self._proc.start()
            theirs.close()
            self._conn = ours
        try:
            self._conn.send(args)
            ok, val = self._conn.recv()
        except (EOFError, OSError) as e:
            self.close()
            raise RuntimeError("the end-of-slot receive's process stopped") from e
        if not ok:
            raise RuntimeError(val)
        return val

    def close(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            except OSError:
                pass
        if self._proc is not None and self._proc.is_alive():
            self._proc.terminate()
            self._proc.join(5.0)
        self._proc = self._conn = None


def _finish_process_main(conn) -> None:
    """The loop of `_FinishProcess`'s child: args in, (True, result) or
    (False, error text) out, until the pipe closes or the parent goes."""
    import multiprocessing

    parent = multiprocessing.parent_process()
    if parent is not None:
        def watch():
            while parent.is_alive():
                time.sleep(2.0)
            os._exit(0)              # the listener is gone: so is the reason to finish

        threading.Thread(target=watch, daemon=True).start()
    if hasattr(os, "nice"):
        try:
            os.nice(5)               # the live receives come first
        except OSError:
            pass
    while True:
        try:
            args = conn.recv()
        except (EOFError, OSError):
            return
        try:
            out = (True, finish_receive(*args))
        except Exception as e:       # reported to the listener, which logs it
            out = (False, f"{type(e).__name__}: {e}")
        conn.send(out)


# --- the radio's dial ----------------------------------------------------------------------

class DialLog:
    """The radio's dial frequency over time, from DIAL_FILE in the state
    directory, which the app writes: one {"t": unix seconds, "dial_hz": Hz}
    per line, appended at every change it makes or sees on the rig readout
    and once when rig control connects (extra keys are ignored, and so is
    a line that does not parse, such as one still being written).

    A change splits a slot's audio into dial epochs: what was heard on one
    dial says nothing about a signal heard on another, and a signal
    tracked across a change would read whatever sits at its audio
    frequency on the new band as its own picture numbers. Each epoch is
    received on its own (`spans`), DIAL_GUARD_S around every change
    counting as unheard. With no file (no rig control) the dial is
    unknown and nothing is split. The first line's dial is taken to have
    been in force before it: connecting is not a change.
    """

    def __init__(self, path):
        self.path = Path(path)
        self._sig = None
        self.points: list[tuple[float, int]] = []    # (time, dial) at the start of each epoch

    def refresh(self) -> None:
        """Re-read the file if it changed."""
        try:
            st = self.path.stat()
        except OSError:
            self._sig, self.points = None, []
            return
        sig = (st.st_mtime_ns, st.st_size)
        if sig == self._sig:
            return
        try:
            text = self.path.read_text(errors="replace")
        except OSError:
            return
        self._sig = sig
        pts = []
        for line in text.splitlines():
            try:
                d = json.loads(line)
                t, hz = float(d["t"]), int(round(float(d["dial_hz"])))
            except (ValueError, KeyError, TypeError, IndexError):
                continue
            if math.isfinite(t) and hz > 0:
                pts.append((t, hz))
        pts.sort(key=lambda p: p[0])
        out: list[tuple[float, int]] = []
        for t, hz in pts:
            if not out or out[-1][1] != hz:
                out.append((t, hz))
        self.points = out

    def spans(self, a: float, b: float) -> list[tuple[float, float, int | None]]:
        """(lo, hi, dial) for each dial in force at some time in [a, b), in
        time order: heard on that dial from lo to hi, unix seconds (-inf or
        inf where no change bounds it), DIAL_GUARD_S already taken off
        around each change; [(-inf, inf, None)] when the dial is unknown."""
        pts = self.points
        if not pts:
            return [(-math.inf, math.inf, None)]
        out = []
        for i, (t, hz) in enumerate(pts):
            start = -math.inf if i == 0 else t
            end = pts[i + 1][0] if i + 1 < len(pts) else math.inf
            if end <= a or start >= b:
                continue
            lo = start + DIAL_GUARD_S[1] if i else -math.inf
            hi = end - DIAL_GUARD_S[0] if i + 1 < len(pts) else math.inf
            if hi > lo:
                out.append((lo, hi, hz))
        return out

    def by_dial(self, a: float, b: float) -> list[tuple[int | None, list[tuple[float, float]]]]:
        """`spans` gathered by dial, in order of first use: a dial left and
        come back to is one capture with a hole in it, not two passes."""
        out: dict = {}
        for lo, hi, dial in self.spans(a, b):
            out.setdefault(dial, []).append((lo, hi))
        return list(out.items())

    def dial_at(self, t: float) -> int | None:
        """The dial in force at time t (None: unknown)."""
        return self.span_at(t)[2]

    def span_at(self, t: float) -> tuple[float, float, int | None]:
        """`spans` for the dial in force at time t (its guard included, so lo
        can lie after t when t is just after a change)."""
        pts = self.points
        if not pts:
            return (-math.inf, math.inf, None)
        i = max(0, int(np.searchsorted([p[0] for p in pts], t, side="right")) - 1)
        lo = pts[i][0] + DIAL_GUARD_S[1] if i else -math.inf
        hi = pts[i + 1][0] - DIAL_GUARD_S[0] if i + 1 < len(pts) else math.inf
        return (lo, hi, pts[i][1])


def same_dial(a: int | None, b: int | None) -> bool:
    """Whether two dials can be the same one: equal, or either unknown (a
    tile found before rig control connected, or with none)."""
    return a is None or b is None or a == b


def restrict(ok: np.ndarray, origin: int, q: int, spans) -> np.ndarray:
    """`capture_read`'s written mask, less everything outside the spans
    [(lo, hi), ...] (unix s, -inf/inf for unbounded)."""
    if any(lo == -math.inf and hi == math.inf for lo, hi in spans):
        return ok
    t0 = slot_t0(q)
    out = np.zeros_like(ok)
    for lo, hi in spans:
        k_lo = 0 if lo == -math.inf else origin + int(math.ceil((lo - t0) * FE_FS))
        k_hi = len(ok) if hi == math.inf else origin + int(math.ceil((hi - t0) * FE_FS))
        a, b = max(0, k_lo), min(len(ok), k_hi)
        if b > a:
            out[a:b] = ok[a:b]
    return out


# --- tiles -----------------------------------------------------------------------------------

@dataclass
class Tile:
    """What the window shows for one signal in one slot."""
    id: str
    q: int
    slot_utc: str
    frame: str
    f_hz: float                     # audio Hz at t0
    dial_hz: int | None = None      # the radio's dial while it was heard (None: unknown)
    status: str = "receiving"       # receiving | finishing | complete | lost
    snr_db: float | None = None     # SNR in 2500 Hz
    z_ref: float | None = None
    callsign: str | None = None
    grid: str | None = None
    picture_id: str | None = None   # hexadecimal
    mode: str | None = None         # A, B, C
    segment: int | None = None
    cw_text: str | None = None
    progress: float = 0.0           # share of the frame's time elapsed when its audio was read
    heard: float = 0.0              # share of the frame's audio heard
    received: float = 0.0           # share of the data latents with W > 0
    mean_w_db: float | None = None
    # This pass's per-latent SNR (dB) over time, receiver.CONF_BINS bins
    # from t0 to the frame's end; None where there is nothing to say.
    confidence: list = field(default_factory=list)
    passes: int = 1                 # passes of this picture in the picture shown
    image: str | None = None        # path relative to the state directory
    image_rev: int = 0              # bumped every time the picture changes
    updated: float = 0.0
    note: str = ""
    uid: str | None = None          # the pass as filed in the multi-pass store
    # "CE" or "L" (WAVEFORM_NAMES): for L, f_hz is the pilot and the
    # signal reaches L_BELOW_HZ under it; a tile from before L is CE
    waveform: str = "CE"

    def to_json(self) -> dict:
        return asdict(self)


@dataclass
class SlotState:
    q: int
    dets: list = field(default_factory=list)
    tiles: dict = field(default_factory=dict)     # id -> Tile
    last_refresh: float | None = None
    done: bool = False
    finishing: bool = False         # the end-of-slot receive is queued or running
    reader: object = None           # where its audio is read from (None: the ring)
    cands: list | None = None       # its acquisition candidates, once they can no longer change
    hints: tuple = ()               # ...and the hints they were found with (`_hints`)


@dataclass
class LiveConfig:
    spec: FrameSpec = FULL
    # The waveforms listened for (WAVEFORM_CE, WAVEFORM_L), each in `spec`'s
    # shape (`specs`). None is CE alone, the listener as it was before L,
    # and so is (WAVEFORM_CE,); qrss_listen.py asks for both unless told
    # --just-ce or --just-l.
    waveforms: tuple[int, ...] | None = None
    refresh_s: float = 120.0          # time between live receives of a slot
    first_s: float = 25.0             # first receive this long after t0 (the preamble is 20 s)
    detect_until_s: float = 300.0     # acquisition re-runs at each refresh until then
    # ...and after it, until this, a preamble hit that gate V refused is
    # checked again at each refresh (about 14 s each, measured): a weak
    # signal's Z_ref grows with the audio, and refused at 205 s for a
    # wandering frequency estimate, KC2G was never checked again
    recheck_until_s: float = 900.0
    # ...and after it, a slot where nothing has been found yet is searched
    # again only this often: on a quiet band that search, over two slots
    # at a time, was most of the listener's work, every minute, for nothing
    empty_refresh_s: float = 300.0
    keep_done_s: float = 6 * 3600.0   # a finished tile stays listed this long
    estimator: str = "joint"
    precision: str = "fp32"
    model: str | None = None
    render: bool = True
    # Receive on worker threads, so the loop keeps taking audio and
    # writing state.json while a receive runs (minutes on a busy band):
    # one for the live receives, one for the end-of-slot ones, so a slot
    # ending does not stop the others' tiles. Off for tests, which step
    # the listener synchronously.
    background: bool = False
    # With background: the end-of-slot receives run in a child process
    # (`_FinishProcess`), on a core of their own. The command line turns
    # it on; tests that replace the receiver in this process leave it off.
    finish_process: bool = False

    @property
    def specs(self) -> tuple:
        """The frames listened for, CE's first: `spec` with no `waveforms`,
        otherwise its twin in each (ValueError for a shape the other
        waveform has no frame of, such as full-v1)."""
        if self.waveforms is None:
            return (self.spec,)
        ws = sorted({int(w) for w in self.waveforms})
        if not ws or any(w not in WAVEFORM_NAMES for w in ws):
            raise ValueError(f"waveforms {self.waveforms!r}: one or more of "
                             f"{', '.join(map(str, WAVEFORM_NAMES))}")
        return tuple(lframe.twin(self.spec, w) for w in ws)

    @property
    def selection(self):
        """What the listener hands the receiver: the tuple of `specs` when
        it holds an L frame (`receiver.is_joint`), and otherwise the CE
        frame itself, so CE alone makes exactly the calls, with the same
        object, it made before L."""
        sp = self.specs
        return sp if receiver.is_joint(sp) else sp[0]


def dir_lock(path):
    """Hold an exclusive lock on directory `path` for this process's life.

    Returns the open lock file (keep it referenced), or None when another
    process holds it. One writer per directory: two listeners on one
    store or one tile directory overwrite each other's files and, before
    this lock, crashed on each other's temporary files.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    f = open(path / ".lock", "a+")
    try:
        if os.name == "nt":
            import msvcrt
            f.seek(0)
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        f.close()
        return None
    f.seek(0)
    f.truncate()
    f.write(f"{os.getpid()}\n")
    f.flush()
    return f


class _PassbandReader:
    """A PassbandStore read like an FeRing: (x, written) for FE indices [a, b)."""

    def __init__(self, passband):
        self.passband = passband

    def read(self, a: int, b: int):
        return self.passband.read(a / FE_FS, b / FE_FS, return_mask=True)


class LiveListener:
    """Feed it audio (`feed`), call `step(now)`; it keeps `state_dir` current."""

    def __init__(self, state_dir, cfg: LiveConfig | None = None, store=None, passband=None,
                 log=None):
        self.cfg = cfg or LiveConfig()
        self.dir = Path(state_dir)
        (self.dir / "tiles").mkdir(parents=True, exist_ok=True)
        self.store = store
        self.passband = passband
        # what is listened for, and what the receiver is handed for it
        self.specs = self.cfg.specs
        self.selection = self.cfg.selection
        self.joint = receiver.is_joint(self.selection)
        self.dur = frame_seconds(self.specs)          # a slot's frame, the longer of the two
        self.ring = FeRing(self.dur + PB_LEAD_S + PB_TAIL_S + 300.0)
        self.dial = DialLog(self.dir / DIAL_FILE)
        self.fe = StreamFE()
        self._early: collections.deque = collections.deque()
        self._quiet: tuple[int, int] | None = None    # the run of silence ending the last chunk
        self._silent: list[list[int]] = []            # audio spans of digital silence, unheard
        self._silent_from: int | None = None          # the silence going on now began here
        self.slots: dict[int, SlotState] = {}
        self.finished: set[int] = set()
        self.done_tiles: list[Tile] = []
        self._recover: list[SlotState] = []   # ended slots to finish from the passband store
        self._headers: dict = {}               # tile id -> the header its signal decoded
        self.codec = None
        self.codec_error: str | None = None
        self.now = 0.0
        self.audio_end = None           # unix time of the newest audio sample
        self.started = time.time()
        self.source = ""
        self.lines: list[str] = []
        self._log = log or (lambda m: print(m, file=sys.stderr, flush=True))
        # Everything but the receivers' own arithmetic happens under this
        # lock; a worker job takes it to copy its capture and to apply what
        # it found, never while it computes.
        self.lock = threading.RLock()
        self._busy = {"live": None, "finish": None}   # what each worker is doing
        self._jobs: queue.Queue | None = None          # live receives
        self._finish_jobs: queue.Queue | None = None   # end-of-slot receives
        self._proc = None
        if self.cfg.background:
            self._jobs = queue.Queue()
            self._finish_jobs = queue.Queue()
            for name, jobs in (("live", self._jobs), ("finish", self._finish_jobs)):
                threading.Thread(target=self._worker, args=(name, jobs), daemon=True).start()
            if self.cfg.finish_process:
                self._proc = _FinishProcess()
        self._load_previous()

    @property
    def busy(self) -> str | None:
        """What the workers are doing, for state.json (None when nothing)."""
        return "; ".join(b for b in (self._busy["live"], self._busy["finish"]) if b) or None

    def close(self) -> None:
        """Stop the end-of-slot process, if there is one."""
        if self._proc is not None:
            self._proc.close()

    # -- logging and state ------------------------------------------------------------

    def log(self, msg: str) -> None:
        line = f"{utc_iso(self.now or time.time())} {msg}"
        self.lines = (self.lines + [line])[-50:]
        self._log(line)
        # and to a file: state.json keeps the last 20 lines and the app's
        # panel lives as long as the app, while a question about last night
        # needs the whole night (a field test, 2026-10-11)
        path = self.dir / LOG_FILE
        try:
            if path.exists() and path.stat().st_size > LOG_FILE_BYTES:
                os.replace(path, path.with_name(LOG_FILE + ".1"))
            with open(path, "a") as f:
                f.write(line + "\n")
        except OSError:
            pass                          # a log that cannot be written must not stop the listener

    def _load_previous(self) -> None:
        """Finished tiles from an earlier run stay listed (until keep_done_s)."""
        try:
            st = json.loads((self.dir / "state.json").read_text())
        except (OSError, ValueError):
            return
        self.finished |= {int(q) for q in st.get("finished", [])}
        for t in st.get("tiles", []):
            if t.get("status") in ("complete", "lost") and "q" in t:
                self.finished.add(int(t["q"]))       # (a state file from before "finished")
            if t.get("status") not in ("receiving", "finishing"):
                if (not t.get("callsign") and not t.get("cw_text")
                        and (t.get("snr_db") or -99.0) > HEADERLESS_MAX_SNR_DB):
                    continue           # kept by an older listener; see _plausible
                try:
                    self.done_tiles.append(Tile(**{k: t[k] for k in Tile.__dataclass_fields__
                                                   if k in t}))
                except TypeError:
                    pass

    def tiles(self) -> list[Tile]:
        live = [t for s in self.slots.values() for t in s.tiles.values()]
        return sorted(live + self.done_tiles, key=lambda t: (-t.q, t.f_hz))

    def write_state(self, listening: bool = True) -> None:
        st = dict(
            version=1, updated=time.time(), now=self.now, now_utc=utc_iso(self.now or time.time()),
            listening=listening, source=self.source, frame=self.cfg.spec.name,
            waveforms=[WAVEFORM_NAMES[s.waveform] for s in self.specs],
            audio_utc=utc_iso(self.audio_end) if self.audio_end else None,
            slots=[dict(q=q, slot_utc=utc_iso(slot_t0(q) - 1),
                        progress=self._progress(q), signals=len(s.tiles))
                   for q, s in sorted(self.slots.items())],
            codec_error=self.codec_error, busy=self.busy,
            tiles=[t.to_json() for t in self.tiles()],
            log=self.lines[-20:],
            finished=sorted(q for q in self.finished
                            if slot_t0(q) > (self.now or time.time()) - 48 * 3600))
        tmp = self.dir / f"state.json.{os.getpid()}.tmp"
        tmp.write_text(json.dumps(st, indent=1))
        os.replace(tmp, self.dir / "state.json")

    # -- audio --------------------------------------------------------------------------

    def feed(self, x, t_end: float) -> None:
        """Audio chunk x (8 kHz) whose last sample is at unix time t_end."""
        x = np.asarray(x, dtype=np.float64)
        if not len(x):
            return
        a = int(round(t_end * FS)) - len(x) + 1
        nxt = self.fe.next_index
        if nxt is not None:
            d = (a - nxt) / FS            # > 0: later than the samples say
            if d > RESYNC_S:
                self.log(f"{d:.1f} s of audio missing; the stream restarts there")
                self._early.clear()
            else:
                # Audio only ever arrives late (a stall, a burst after one),
                # never early, so the stream follows on unless *every* chunk
                # for EARLY_WINDOW_S came early: then the sound card's clock
                # has run ahead of the computer's, and the stream steps back.
                self._early.append((t_end, d))
                while self._early and t_end - self._early[0][0] > EARLY_WINDOW_S:
                    self._early.popleft()
                if (t_end - self._early[0][0] >= 0.9 * EARLY_WINDOW_S
                        and max(e for _, e in self._early) < -RESYNC_S):
                    self.log(f"audio running {-d:.1f} s ahead of the clock; "
                             "the stream restarts at the clock")
                    self._early.clear()
                else:
                    a = nxt
        self._note_silence(a, x)
        for n0, fe in self.fe.push(a, x):
            for lo, hi in self._heard(n0, len(fe)):
                self.ring.write(n0 + lo, fe[lo:hi])
                if self.passband is not None:
                    self.passband.write((n0 + lo) / FE_FS, fe[lo:hi])
        self.audio_end = (a + len(x) - 1) / FS

    def _note_silence(self, a: int, x: np.ndarray) -> None:
        """Add x's runs of digital silence (x from audio index a) to `_silent`.

        A run is counted once it has lasted SILENCE_S, so the first part of
        one spanning several chunks may already be written as heard: at
        most SILENCE_S of it, once per silence.
        """
        n_min = int(round(SILENCE_S * FS))
        run = self._quiet if self._quiet is not None and self._quiet[1] == a else None
        self._quiet = None
        d = np.diff(np.r_[0, (np.abs(x) < SILENCE_LEVEL).astype(np.int8), 0])
        starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
        if self._silent_from is not None and (not len(starts) or starts[0]):
            self._sound_back(a)                   # the silence ended where this chunk began
        for s, e in zip(starts, ends):
            lo = run[0] if s == 0 and run is not None else a + int(s)
            hi = a + int(e)
            if hi - lo >= n_min:
                if self._silent and lo <= self._silent[-1][1]:
                    self._silent[-1][1] = max(self._silent[-1][1], hi)
                else:
                    self._silent.append([lo, hi])
                if self._silent_from is None:
                    self._silent_from = lo
                    self.log(f"the audio is digital silence from {utc_iso(lo / FS)}: "
                             "not counted as heard")
            if e == len(x):
                self._quiet = (lo, hi)            # it may go on in the next chunk
            elif self._silent_from is not None:
                self._sound_back(hi)

    def _sound_back(self, a: int) -> None:
        self.log(f"audio back after {(a - self._silent_from) / FS:.0f} s of digital silence")
        self._silent_from = None

    def _heard(self, n0: int, n: int) -> list[tuple[int, int]]:
        """The runs [lo, hi) of FE samples n0 .. n0+n-1 that are not digital silence."""
        ok = np.ones(n, dtype=bool)
        keep = []
        for lo, hi in self._silent:
            ka, kb = -(-lo // FE_DECIM) - n0, -(-hi // FE_DECIM) - n0
            if kb > -10 * FE_FS:                  # (kept a while: the stream can step back)
                keep.append([lo, hi])
            ok[max(ka, 0):max(min(kb, n), 0)] = False
        self._silent = keep
        d = np.diff(np.r_[0, ok.astype(np.int8), 0])
        return list(zip(np.flatnonzero(d == 1).tolist(), np.flatnonzero(d == -1).tolist()))

    def backfill(self, now: float) -> float:
        """Refill the ring from the passband store: seconds of audio read back.

        A listener restarted mid-slot (the app restarting, receive stopping
        for a transmission) otherwise starts with an empty ring, so every
        slot already under way has lost its preamble, and with it the live
        view, until it ends. The passband store holds the same front-end
        stream, so whatever an earlier listener heard is put back. Call it
        once, before the first `feed`.
        """
        if self.passband is None:
            return 0.0
        a = int(round(now * FE_FS)) - self.ring.n
        x, m = self.passband.read(a / FE_FS, a / FE_FS + self.ring.n / FE_FS, return_mask=True)
        if not m.any():
            return 0.0
        edges = np.flatnonzero(np.diff(np.concatenate(([0], m.astype(np.int8), [0]))))
        for lo, hi in zip(edges[::2], edges[1::2]):
            self.ring.write(a + int(lo), x[lo:hi])
        got = float(m.sum()) / FE_FS
        self.log(f"{got:.0f} s of earlier audio read back from the passband store")
        return got

    def recover(self, now: float, hours: float = RECOVER_HOURS) -> list[int]:
        """Queue the end-of-slot receive of every slot that ended in the last
        `hours` while no listener finished it, from the passband store.

        Stopping a listener near the end of a frame (just before a quarter
        hour) would otherwise lose the slot: its latents are stored only by
        that receive. The slots run newest first, one at a time, each when
        no other end-of-slot receive is waiting (see `step`). Returns the
        slots queued.
        """
        if self.passband is None or hours <= 0:
            return []
        dur = self.dur
        reader = _PassbandReader(self.passband)
        # A slot whose passes the store already holds was finished by the
        # earlier listener, which stopped before it recorded that: receiving
        # it again would file a second copy of each pass (the passband
        # store's audio is not bit for bit the ring's, so the copy is a new
        # pass) and count one slot as two passes of its picture.
        stored = set()
        if self.store is not None:
            for u in self.store.pass_uids():       # pass_uid: "<q>_<mHz>_<sha8>"
                head = u.split("_", 1)[0]
                if head.lstrip("-").isdigit():
                    stored.add(int(head))
        out = []
        for q in range(int(math.floor(now / 900.0)), int(math.floor(
                (now - hours * 3600.0 - dur) / 900.0)) - 1, -1):
            end = slot_t0(q) + dur + PB_TAIL_S
            if (not (now - hours * 3600.0 <= end <= now) or q in self.finished
                    or q in self.slots or any(r.q == q for r in self._recover)):
                continue
            t0 = slot_t0(q)
            if q in stored:
                self.finished.add(q)
                self.log(f"slot {utc_iso(t0 - 1)}: its passes are in the store already; "
                         "not received again")
                continue
            _, ok = reader.read(t0 * FE_FS, int(round((t0 + dur) * FE_FS)))
            if ok.mean() < FINISH_SEARCH_MIN_HEARD:
                continue                 # not heard (enough) by an earlier listener
            self._recover.append(SlotState(q, finishing=True, reader=reader))
            out.append(q)
            self.log(f"slot {utc_iso(t0 - 1)}: ended while no listener finished it; "
                     "receiving it from the passband store")
        return out

    # -- slots ----------------------------------------------------------------------------

    def _progress(self, q: int) -> float:
        return float(min(max((self.now - slot_t0(q)) / self.dur, 0.0), 1.0))

    def active_slots(self, now: float) -> list[int]:
        """Slots whose frame has started and is not yet finished at `now`."""
        dur = self.dur
        lo = int(math.floor((now - dur - PB_TAIL_S - 1.0) / 900.0))
        return [q for q in range(lo, int(math.floor((now - 1.0) / 900.0)) + 1)
                if slot_t0(q) <= now < slot_t0(q) + dur + PB_TAIL_S]

    def step(self, now: float) -> bool:
        """Do whatever is due at `now` (unix seconds); True if anything changed."""
        self.now = now
        changed = False
        dur = self.dur
        for q in self.active_slots(now):
            if q not in self.slots and q not in self.finished:
                # a slot whose start we never heard is still received live
                self.slots[q] = SlotState(q)
                self.log(f"slot {utc_iso(slot_t0(q) - 1)}: frame started")
                changed = True
        for q, s in sorted(self.slots.items()):
            t0 = slot_t0(q)
            if s.done:
                continue
            if now >= t0 + dur + PB_TAIL_S:
                if not s.finishing:
                    s.finishing = True
                    for t in s.tiles.values():
                        t.status = "finishing"
                    # The end-of-slot receive takes minutes (3-5 on a desktop).
                    # One more live receive, of the whole frame, shows the
                    # pass to its end meanwhile instead of where the last
                    # refresh, up to refresh_s before the end, left it.
                    if self._jobs is not None and s.dets and (
                            s.last_refresh is None or s.last_refresh < t0 + dur):
                        s.last_refresh = now
                        self._submit(self._refresh, s)
                    self._submit(self._finish, s, finish=True)
                    changed = True
            elif now >= t0 + self.cfg.first_s and (
                    s.last_refresh is None or now - s.last_refresh >= self._interval(s, now)):
                if self._jobs is None or (self._busy["live"] is None and self._jobs.empty()):
                    s.last_refresh = now     # (when busy, it is still due next time)
                    self._submit(self._refresh, s)
                    changed = True
        if self._recover and self._finish_idle():
            s = self._recover.pop(0)
            self.slots[s.q] = s
            self._submit(self._finish, s, finish=True)
            changed = True
        for q in [q for q, s in self.slots.items() if s.done]:
            for tid in self.slots[q].tiles:
                self._headers.pop(tid, None)
            self.done_tiles.extend(self.slots.pop(q).tiles.values())
            self.finished.add(q)
        keep = [t for t in self.done_tiles if now - t.updated < self.cfg.keep_done_s]
        changed |= len(keep) != len(self.done_tiles)
        self.done_tiles = keep
        return changed

    def _interval(self, s: SlotState, now: float) -> float:
        """Seconds between live receives of slot s: refresh_s, or
        empty_refresh_s once detect_until_s has passed with nothing found."""
        if not s.dets and now - slot_t0(s.q) >= self.cfg.detect_until_s:
            return max(self.cfg.refresh_s, self.cfg.empty_refresh_s)
        return self.cfg.refresh_s

    # -- the worker -----------------------------------------------------------------------

    def _submit(self, job, s: SlotState, finish: bool = False) -> None:
        if self._jobs is None:
            job(s)
        else:
            (self._finish_jobs if finish else self._jobs).put((job, s))

    def _worker(self, name: str, jobs: queue.Queue) -> None:
        while True:
            job, s = jobs.get()
            self._busy[name] = f"{job.__name__.strip('_')} {utc_iso(slot_t0(s.q) - 1)}"
            try:
                job(s)
            except Exception as e:       # one bad slot must not stop the listener
                with self.lock:
                    self.log(f"slot {utc_iso(slot_t0(s.q) - 1)}: {type(e).__name__}: {e}")
                    s.done = True
            finally:
                self._busy[name] = None

    def _finish_idle(self) -> bool:
        """No end-of-slot receive queued or running."""
        return self._finish_jobs is None or (self._busy["finish"] is None
                                             and self._finish_jobs.empty())

    def idle(self) -> bool:
        """No receive queued or running."""
        return self._finish_idle() and (self._jobs is None or (self._busy["live"] is None
                                                               and self._jobs.empty()))

    # -- which signal is which ---------------------------------------------------------

    def _centre(self, x) -> float:
        """Where a detection or pass sits for "the same signal": its band's
        centre (`receiver.centre_hz`) when both waveforms may be heard, its
        frequency when only CE is, as before L."""
        return receiver.centre_hz(x) if self.joint else x.f_hz

    def _waveform(self, p) -> int:
        return int(p.waveform) if self.joint else WAVEFORM_CE

    def _kind(self, p) -> tuple[str, str]:
        """(waveform, frame) for pass p's tile: the pass's own on the joint
        path, CE and the configured frame on CE's alone."""
        if not self.joint:
            return "CE", self.cfg.spec.name
        return WAVEFORM_NAMES[self._waveform(p)], p.frame

    def _spec_of(self, p):
        """The frame pass p was received with, which its frame name says."""
        return lframe.get_any(p.frame) if self.joint else self.cfg.spec

    def _match(self, s: SlotState, f_hz: float, dial: int | None = None,
               waveform: str = "CE") -> Tile | None:
        """The tile of slot s (of either waveform) at the band centre of a
        signal at audio f_hz in `waveform`, on the same dial (never one
        heard on another)."""
        c = centre_of(f_hz, waveform)
        best, gap = None, MERGE_HZ
        for t in s.tiles.values():
            d = abs(centre_of(t.f_hz, t.waveform) - c)
            if same_dial(t.dial_hz, dial) and d < gap:
                best, gap = t, d
        return best

    def _tile_for(self, s: SlotState, f_hz: float, dial: int | None = None,
                  waveform: str = "CE", frame: str | None = None) -> Tile:
        t = self._match(s, f_hz, dial, waveform)
        if t is None:
            tid = f"{s.q}-{int(round(f_hz))}"
            while tid in s.tiles:
                tid += "b"
            t = Tile(id=tid, q=s.q, slot_utc=utc_iso(slot_t0(s.q) - 1),
                     frame=frame or self.cfg.spec.name, f_hz=float(f_hz), dial_hz=dial,
                     updated=self.now, waveform=waveform)
            s.tiles[tid] = t
            on = "" if dial is None else f" (dial {dial / 1e6:.6f} MHz)"
            what = f", waveform {waveform}" if self.joint else ""
            self.log(f"slot {t.slot_utc}: signal at {f_hz:.1f} Hz{what}{on}")
        elif t.dial_hz is None:
            t.dial_hz = dial          # found before the dial was known (rig control connected)
        return t

    def _takes(self, t: Tile, p) -> bool:
        """Whether pass p may update tile t (`_tile_for`'s, matched by band
        centre). A pass of the tile's waveform does; one of the other, both
        waveforms having locked on one signal, only with the larger Z_ref,
        and the tile then becomes that waveform."""
        wf, fr = self._kind(p)
        if t.waveform != wf:
            z = float(p.report.z_ref)
            if t.z_ref is not None and not z > t.z_ref:
                return False
            was = f" against {t.z_ref:.1f} as {t.waveform}" if t.z_ref is not None else ""
            self.log(f"slot {t.slot_utc}: the signal at {t.f_hz:.1f} Hz is waveform {wf} "
                     f"(Z_ref {z:.1f}{was})")
            t.waveform = wf
        t.frame = fr
        return True

    def _keep(self, s: SlotState, d) -> None:
        """Detection d stands for its signal in s.dets: in place of any at its
        band centre, the newest, unless that one is the other waveform and
        at least as strong."""
        c = self._centre(d)
        if self.joint and any(abs(self._centre(g) - c) < MERGE_HZ and g.waveform != d.waveform
                              and not d.z_ref > g.z_ref for g in s.dets):
            return
        s.dets = [g for g in s.dets if abs(self._centre(g) - c) >= MERGE_HZ] + [d]

    def _candidates(self, s: SlotState, prep, ok, origin: int) -> list:
        """Slot s's live acquisition candidates. Acquisition reads only the
        preamble and a few seconds around it, so once the audio read is
        past that (with the blanker's reach) its answer is final: kept,
        rather than found again at every refresh (unless a new hint turns
        up). A hint A2 did not pass is left to the end of the slot: with
        no timing, gate V costs about a minute."""
        hints = tuple(self._hints(s.q, self.dial.dial_at(slot_t0(s.q))))
        if s.cands is None or not set(hints) <= set(s.hints):
            if hints and set(hints) != set(s.hints):
                where = (", ".join(f"{f:.1f} Hz ({WAVEFORM_NAMES[w]})" for f, w in hints)
                         if self.joint else ", ".join(f"{f:.1f}" for f in hints) + " Hz")
                self.log(f"slot {utc_iso(slot_t0(s.q) - 1)}: a later pass is due at {where}")
            cands = receiver.candidates(prep, self.selection, live_only=True, hints=hints)
            heard = np.flatnonzero(ok)
            final = len(heard) and (heard[-1] + 1 - origin) / FE_FS > acquire.live_end_s() + 1.0
            s.cands, s.hints = (cands if final else None), hints
        else:
            cands = s.cands
        return [c for c in cands if getattr(c, "method", None) != "hint"]

    def _hints(self, q: int, dial: int | None) -> list:
        """Audio frequencies (on `dial`) where a later pass of a picture
        partly received is due in slot q: from the multi-pass store, and
        from the tiles of earlier slots (whose end-of-slot receive may not
        have filed them yet). A pass heard on another dial is moved by the
        difference (USB), and dropped if that leaves the band. On the joint
        path each is (f, waveform), the waveform its pass was; with CE
        alone a bare f, as before L. A pass of a waveform not listened for
        gives none, and one signal (one band centre) gives one."""
        due = []                                       # (f_hz, waveform, the dial it was heard on)
        if self.store is not None:
            try:
                due = [(d["f_hz"], _due_waveform(d), self.dial.dial_at(slot_t0(d["q"])))
                       for d in self.store.due(q)]
            except Exception as e:                     # never lose the slot over the store
                self.log(f"hints: {e}")
        for t in self.tiles():
            if t.mode in ("A", "B", "C") and t.segment is not None and t.q < q:
                n = "ABC".index(t.mode) + 1
                if any(t.q + 2 * (g - t.segment) == q for g in range(t.segment + 1, n)):
                    due.append((t.f_hz, _waveform_id(t.waveform), t.dial_hz if t.dial_hz is not None
                                else self.dial.dial_at(slot_t0(t.q))))
        heard_as = {sp.waveform for sp in self.specs}
        out: list[tuple[float, int]] = []
        for f, w, was in due:
            if w not in heard_as:
                continue
            if was is not None and dial is not None:
                f += was - dial
            if in_band(f, w) and all(abs(centre_of(f, w) - centre_of(g, v)) >= MERGE_HZ
                                     for g, v in out):
                out.append((round(float(f), 2), w))
        return sorted(out) if self.joint else sorted(f for f, _ in out)

    def _refresh(self, s: SlotState) -> None:
        if s.done:
            return              # (a final refresh queued behind a finish that has run)
        spec = self.selection
        t_start = time.time()
        t0 = slot_t0(s.q)
        with self.lock:
            x, ok, origin = capture_read(self.ring, s.q, spec)
            elapsed = self.now - t0
            # the tile's progress is when its audio was read, not when the
            # receive returned: the audio in between is not in this receive
            # and must not read as a hole
            progress = self._progress(s.q)
            self.dial.refresh()
            dial = self.dial.dial_at(t0)
            heard = dict(self.dial.by_dial(t0 - origin / FE_FS, t0 + len(x) / FE_FS))
        # live, a slot is received on the dial it started on: anything heard
        # on another counts as not heard
        ok = restrict(ok, origin, s.q, heard.get(dial, []))
        if ok[origin:origin + PREAMBLE_S * FE_FS].mean() < 0.5:
            return          # joined after its preamble: only the end-of-slot search can find it
        prep, erase, heard = capture_prepare(x, ok, origin, s.q, spec)
        found = []
        if elapsed < self.cfg.detect_until_s or not s.dets:
            found = receiver.detect(prep, spec, live_only=True, erase_s=erase,
                                    cands=self._candidates(s, prep, ok, origin))
        elif elapsed < self.cfg.recheck_until_s:
            known = [self._centre(d) for d in s.dets]
            cands = self._candidates(s, prep, ok, origin)
            if any(all(abs(self._centre(c) - f) >= MERGE_HZ for f in known) for c in cands):
                found = receiver.detect(prep, spec, live_only=True, erase_s=erase,
                                        cands=cands, skip_hz=known)
        for d in found:          # keep one detection per signal
            self._keep(s, d)
        if not s.dets:
            return
        passes = receiver.receive_slot(prep, spec, estimator=self.cfg.estimator, erase_s=erase,
                                       round_b=False, dets=list(s.dets))
        with self.lock:
            if s.done:
                return
            for p in passes:
                p = self._known_header(s, p, dial)
                if not self._plausible(s, p, dial):
                    continue
                t = self._tile_for(s, p.f_hz, dial, *self._kind(p))
                if not self._takes(t, p):
                    continue
                self._update_tile(t, p, heard, progress=progress)
                t.status = "finishing" if s.finishing else "receiving"
                self.log(f"slot {utc_iso(slot_t0(s.q) - 1)}: {len(passes)} signal(s) at "
                     f"{100 * self._progress(s.q):.0f}% ({time.time() - t_start:.0f} s)")

    def _finish(self, s: SlotState) -> None:
        """The frame is over: receive the whole slot (`finish_receive`, in
        the end-of-slot process when there is one), store, render the
        accumulator."""
        spec = self.selection
        t_start = time.time()
        t0 = slot_t0(s.q)
        with self.lock:
            x, ok, origin = capture_read(s.reader or self.ring, s.q, spec)
            dets = list(s.dets)
            self.dial.refresh()
            dials = self.dial.by_dial(t0 - origin / FE_FS, t0 + len(x) / FE_FS)
            first = self.dial.dial_at(t0)
            hints = self._hints(s.q, first)
        results = []
        # one receive per dial the slot was heard on, each with the others
        # erased; the preamble's detections belong to the dial it started on
        for dial, spans in dials:
            args = (x, restrict(ok, origin, s.q, spans), origin, s.q, spec, self.cfg.estimator,
                    dets if dial == first else [], hints if dial == first else [])
            if self._proc is not None:
                passes, heard = self._proc.call(args)
            else:
                passes, heard = finish_receive(*args)
            results.append((dial, passes, heard))
        with self.lock:
            self._finish_apply(s, results)
            n = sum(len(p) for _, p, _ in results)
            heard = max((h for _, _, h in results), default=0.0)
            what = "" if heard >= FINISH_SEARCH_MIN_HEARD else \
                f", {100 * heard:.0f}% heard: no whole-slot search"
            if len(results) > 1:
                what += f", on {len(results)} dials"
            if s.reader is not None:
                what += ", from the passband store"
            self.log(f"slot {utc_iso(slot_t0(s.q) - 1)}: frame over, {n} pass(es)"
                     f"{what} ({time.time() - t_start:.0f} s)")

    def _finish_apply(self, s: SlotState, results) -> None:
        """results: (dial, passes, share heard) for each dial the slot was heard on."""
        s.done = True
        seen = set()
        for dial, passes, heard in results:
            passes = [self._known_header(s, p, dial) for p in passes]
            passes = [p for p in passes if self._plausible(s, p, dial)]
            for p in passes:
                t = self._tile_for(s, p.f_hz, dial, *self._kind(p))
                if not self._takes(t, p):
                    self.log(f"slot {t.slot_utc}: the pass at {p.f_hz:.1f} Hz as waveform "
                             f"{self._kind(p)[0]} is not kept: the signal read stronger as "
                             f"{t.waveform}")
                    continue
                seen.add(t.id)
                key = None
                if self.store is not None:
                    from .associate import associate

                    try:
                        key, rule = associate(p, self.store)
                        t.note = (f"stored ({rule or 'already a member'})" if key
                                  else "stored (provisional)")
                        t.uid = p.uid
                    except Exception as e:         # never lose the tile over the store
                        t.note = f"store failed: {e}"
                self._update_tile(t, p, heard, key=key)
                t.status = "complete"
        for tid, t in s.tiles.items():
            if tid not in seen:
                t.status = "lost"
                t.updated = self.now
        if seen:
            self._identify_provisional()

    def _identify_provisional(self) -> None:
        """Finished tiles filed provisional whose pass the store has since
        filed with its picture: a later pass's header (or the contents of
        a picture stored since) identified it. They take the picture's
        callsign, ID and mode, so the app hides them behind the newer
        tile that combines them; their own picture is not redrawn."""
        todo = [t for t in self.done_tiles
                if t.uid and t.status == "complete" and t.note.startswith("stored (provisional)")]
        if not todo or self.store is None:
            return
        for t in todo:
            try:
                key = self.store.find_member(t.uid)
                if key is None:
                    continue
                acc = self.store.accumulator(key)
                m = acc.member(t.uid) or {}
            except Exception:          # a store problem must not stop the finish
                continue
            t.callsign, t.picture_id = str(key[0]), f"{int(key[1]):08x}"
            t.mode = "ABC"[int(acc.mode)]
            if m.get("segment") is not None:
                t.segment = int(m["segment"])
            t.note = f"stored ({m.get('method', 'corr')})"
            t.updated = self.now
            self.log(f"slot {t.slot_utc}: the pass at {t.f_hz:.1f} Hz is "
                     f"{t.callsign} {t.picture_id}")

    def _known_header(self, s: SlotState, p: PassResult, dial: int | None = None) -> PassResult:
        """p, with the header its signal decoded at an earlier receive when
        this one did not decode it.

        Each live receive decodes the header afresh from what has arrived,
        and near its threshold one receive can miss what an earlier one
        found. The header of a signal does not change during its pass, and
        a decoded one is checked to well under 1e-8 false accepts, so it
        stands; without this a tile fell back to the headerless guess.
        """
        if p.header is not None:
            return p
        t = self._match(s, p.f_hz, dial, self._kind(p)[0])
        h = self._headers.get(t.id) if t is not None else None
        if h is None:
            return p
        p = copy.copy(p)                  # (the receiver's result stays as it was)
        p.header = h
        return p

    def _plausible(self, s: SlotState, p: PassResult, dial: int | None = None) -> bool:
        """Whether a pass is worth a tile (and the store).

        A pass needs a finite frequency inside the band receivers search
        (for L, its whole signal: `in_band`).
        And with continuous audio every slot's capture holds half of each
        neighbouring slot's transmissions (FULL frames last two quarter
        hours), whose carriers the whole-slot search can lock onto at the
        wrong timing: a headerless pass within GHOST_HZ of a tile of
        another slot that did decode its header (band centres, whichever
        waveform each is) is taken to be that
        signal, and dropped; only the two neighbouring slots overlap this
        one, so a decoded tile of any other slot (the same picture sent
        again half an hour later, say) never drops it. A headerless pass
        stronger than HEADERLESS_MAX_SNR_DB is dropped too, unless its
        callsign windows read a callsign: a pass joined after its header
        (a listener started mid-slot) is headerless however strong.
        """
        f = float(p.f_hz)
        if not (math.isfinite(f) and in_band(f, self._waveform(p))):
            return False
        if p.header is not None:
            return True
        # A header decodes far below this SNR, so a pass this strong
        # without one is something else locked on (a carrier, a birdie, a
        # voice on a phone band), not a QRSSTVAE signal.
        snr = float(p.report.snr2500_db)
        called = p.cw is not None and bool(p.cw.text)
        if not called and (not math.isfinite(snr) or snr > HEADERLESS_MAX_SNR_DB):
            return False
        others = [t for q, o in self.slots.items() if abs(q - s.q) == 1 for t in o.tiles.values()]
        others += [t for t in self.done_tiles if abs(t.q - s.q) == 1]
        c = float(self._centre(p))
        return not any(t.callsign and same_dial(t.dial_hz, dial)
                       and abs(centre_of(t.f_hz, t.waveform) - c) < receiver.GHOST_HZ
                       for t in others)

    def _update_tile(self, t: Tile, p: PassResult, heard: float, key=None,
                     progress: float | None = None) -> None:
        r = p.report
        t.f_hz = float(p.f_hz)
        t.snr_db = float(r.snr2500_db) if np.isfinite(r.snr2500_db) else None
        t.z_ref = float(r.z_ref) if np.isfinite(r.z_ref) else None
        t.progress = self._progress(t.q) if progress is None else progress
        t.heard = float(heard)
        w = np.asarray(p.w, dtype=np.float64)
        t.received = float(np.mean(w > 0)) if len(w) else 0.0
        t.mean_w_db = receiver.mean_w_db(p) if t.received > 0 else None
        # (elapsed in seconds: progress is a share of the slot's frame, the
        # longer one when both waveforms are heard, not of the pass's own)
        t.confidence = receiver.confidence_db(p, self._spec_of(p), t.progress * self.dur)
        h = p.header
        if h is not None:
            self._headers[t.id] = h
            t.callsign, t.grid = h.callsign, h.grid
            t.picture_id = f"{int(h.picture_id):08x}"
            t.mode, t.segment = "ABC"[h.mode], int(h.segment)
        if p.cw is not None and p.cw.text:
            t.cw_text = p.cw.text
        t.updated = self.now
        if self.cfg.render:
            self._render(t, p, key)

    # -- pictures -------------------------------------------------------------------------

    def _codec(self, codec_id: int):
        if self.codec is None and self.codec_error is None:
            from . import render

            try:
                self.codec = render.load(self.cfg.precision, self.cfg.model, codec_id=codec_id,
                                         warn=self.log)
            except BaseException as e:      # CodecMismatch is a SystemExit
                self.codec_error = str(e) or type(e).__name__
                self.log(f"no pictures: {self.codec_error}")
        return self.codec

    def _render(self, t: Tile, p: PassResult, key=None) -> None:
        """Draw the tile's picture: this pass, plus the store's accumulator.

        Without a header (not decoded yet, or never) the picture is a
        guess: segment 0 of a mode A send with the codec loaded, which is
        right for every mode A send and the first pass of B and C, and
        noise otherwise. It is marked GUESS_NOTE, never combined with the
        store, and replaced once a header decodes. A headerless pass the
        store matched to a picture (`key`, by a combined header or by its
        latents, spec section 9) is drawn as that picture instead.
        """
        from . import render

        h = p.header
        if len(p.z) != picture.SENT:
            return                       # nowhere to place the latents
        matched = None
        if h is None and key is not None and self.store is not None \
                and self.store.has_accumulator(key):
            matched = self.store.accumulator(key)
        codec_id = h.codec_id if h is not None else matched.codec_id if matched else None
        codec = self._codec(codec_id)
        if codec is None:
            return
        if matched is not None:
            seg, mode = None, int(matched.mode)      # the pass is a member already
            t.callsign, t.picture_id = str(key[0]), f"{int(key[1]):08x}"
            t.mode = "ABC"[mode]
        if h is None and matched is None:
            seg, mode = 0, 0
            if GUESS_NOTE not in t.note:
                t.note = f"{t.note}; {GUESS_NOTE}" if t.note else GUESS_NOTE
        else:
            if h is not None:
                seg, mode = int(h.segment), int(h.mode)
            try:
                render.check_codec_id(codec, codec_id)
            except SystemExit as e:      # CodecMismatch: a silently wrong picture
                t.note = str(e)
                return
            t.note = t.note.replace(f"; {GUESS_NOTE}", "").replace(GUESS_NOTE, "")
        S = np.zeros((picture.N_GROUPS, picture.GROUP_LATENTS), dtype=np.float64)
        W = np.zeros_like(S)
        n = 1
        k = key if key is not None else None if h is None else (h.callsign, int(h.picture_id))
        member = False
        if k is not None and self.store is not None and self.store.has_accumulator(k):
            acc = self.store.accumulator(k)
            S += acc.S
            W += acc.W
            # (a copy of a transmission the store holds already is that member;
            # on the joint path compared by band centre, as the store files it)
            wf ={"waveform": self._waveform(p)} if self.joint else {}
            member = p.uid in acc.uids or (
                seg is not None and self.store.twin(acc, p.q, p.f_hz, seg, **wf) is not None)
            n = len(acc.members) + (0 if member else 1)
        if not member and seg is not None:
            idx = picture.air_to_canonical(seg)
            w = np.asarray(p.w, dtype=np.float64)
            S[seg, idx] += np.where(w > 0, w * np.asarray(p.z, dtype=np.float64), 0.0)
            W[seg, idx] += w
        if not np.any(W > 0):
            return
        try:
            img = render.render(codec, S, W, mode)
        except Exception as e:
            t.note = f"render failed: {e}"
            return
        rel = f"tiles/{t.id}.png"
        tmp = self.dir / f"tiles/.{t.id}.{os.getpid()}.png.tmp"
        img.save(tmp, format="PNG")
        os.replace(tmp, self.dir / rel)
        t.image = rel
        t.image_rev += 1
        t.passes = n


# --- audio sources -------------------------------------------------------------------------

class Stamper:
    """Unix times for audio read from a pipe, which says when it was read,
    not when it was heard.

    Audio only ever arrives late: a chunk read at time t ended at t or
    before. Steadily it arrives as it is heard, but after a stall (the
    app's audio pump held up, this process busy, a listener starting with
    audio queued for it) several seconds come at once, and stamping each
    piece with its read time put the first piece seconds after the audio
    before it: the listener took a burst for missing audio, left a hole,
    and ran the rest of the burst seconds late until it stepped back.

    So a chunk that reads as late is held, with the chunks after it,
    until either one of them reads as following on (a backlog, now
    drained: nothing was missing) or hold_s has passed (audio really was
    missing). Held chunks go out as one, timed by the chunk that arrived
    least late: if the stream is contiguous, the start it implies is the
    latest start possible. The stream's first audio is held hold_s too,
    since a listener starting up has audio queued for it.
    """

    def __init__(self, hold_s: float = STAMP_HOLD_S):
        self.hold_s = float(hold_s)
        self._held: list[tuple[np.ndarray, float]] = []
        self._next: int | None = None       # 8 kHz index after the last sample put out

    def push(self, x: np.ndarray, t: float) -> list[tuple[np.ndarray, float]]:
        """Chunk x, read at unix time t: the (x, time of its last sample) ready to go out."""
        self._held.append((x, t))
        start = self._start()
        if t - self._held[0][1] < self.hold_s and (
                self._next is None or start - self._next > RESYNC_S * FS):
            return []
        return self.flush()

    def flush(self) -> list[tuple[np.ndarray, float]]:
        """Everything held, as one chunk."""
        if not self._held:
            return []
        start = self._start()
        x = np.concatenate([c for c, _ in self._held])
        self._held = []
        self._next = start + len(x)
        return [(x, (start + len(x) - 1) / FS)]

    def _start(self) -> int:
        n, best = 0, None
        for c, t in self._held:
            n += len(c)
            a = int(round(t * FS)) - n + 1
            best = a if best is None else min(best, a)
        return best


class StdinSource:
    """Raw mono 8 kHz audio on stdin (float32 or int16, little-endian), read on a thread.

    Each chunk is stamped with the time it was read (see `Stamper` for
    what happens after a stall). The reading thread never waits on the
    receiver, so a receive that takes a minute loses no audio (the queue
    holds it).
    """

    def __init__(self, fmt: str = "f32", chunk_s: float = 0.25, stream=None,
                 hold_s: float = STAMP_HOLD_S):
        self.dtype = np.dtype("<f4") if fmt == "f32" else np.dtype("<i2")
        self.scale = 1.0 if fmt == "f32" else 1.0 / 32768.0
        self.nbytes = int(chunk_s * FS) * self.dtype.itemsize
        self.stream = stream if stream is not None else sys.stdin.buffer
        self.stamper = Stamper(hold_s)
        self.q: queue.Queue = queue.Queue()
        self.eof = False
        self.name = f"stdin ({fmt}, 8 kHz)"
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self) -> None:
        rest = b""
        while True:
            b = self.stream.read1(self.nbytes) if hasattr(self.stream, "read1") else \
                self.stream.read(self.nbytes)
            if not b:
                break
            t = time.time()
            b = rest + b
            k = len(b) // self.dtype.itemsize * self.dtype.itemsize
            rest = b[k:]
            if k:
                x = np.frombuffer(b[:k], self.dtype).astype(np.float64) * self.scale
                for out in self.stamper.push(x, t):
                    self.q.put(out)
        for out in self.stamper.flush():
            self.q.put(out)
        self.q.put(None)

    def chunks(self, timeout: float = 1.0):
        """Yield (x, t_end) as they arrive, or (None, None) after `timeout` with none."""
        while True:
            try:
                item = self.q.get(timeout=timeout)
            except queue.Empty:
                yield None, None
                continue
            if item is None:
                self.eof = True
                return
            yield item


class WavSource:
    """A recording replayed from `start_unix`, stamped from its own sample count.

    speed 0 replays as fast as the receiver goes; otherwise that many
    times real time.
    """

    def __init__(self, path, start_unix: float, speed: float = 0.0, chunk_s: float = 1.0):
        from sstvae import wavio

        self.x = wavio.read_wav(str(path))
        self.start = float(start_unix)
        self.speed = float(speed)
        self.chunk = int(chunk_s * FS)
        self.eof = False
        self.name = f"{path} (replay from {utc_iso(start_unix)})"

    def chunks(self, timeout: float = 1.0):
        wall0 = time.time()
        for a in range(0, len(self.x), self.chunk):
            x = self.x[a:a + self.chunk]
            t_end = self.start + (a + len(x) - 1) / FS
            if self.speed > 0:
                lag = (t_end - self.start) / self.speed - (time.time() - wall0)
                if lag > 0:
                    time.sleep(lag)
            yield x, t_end
        self.eof = True


def run(listener: LiveListener, source, *, realtime: bool, stop=None) -> None:
    """Feed `source` into `listener` until it ends (or `stop` is set).

    realtime: "now" is the computer's clock (live audio); otherwise it is
    the time of the newest sample (a replay), and at the end of a replay
    time runs on until every slot it touched has finished.
    """
    listener.source = source.name
    names = [f"{WAVEFORM_NAMES[s.waveform]} ({s.name})" for s in listener.specs]
    listener.log(f"listening: {source.name}, frame {listener.cfg.spec.name}, "
                 f"waveform{'s' if len(names) > 1 else ''} {' and '.join(names)}")
    listener.write_state()
    last_write = 0.0
    for x, t_end in source.chunks():
        if stop is not None and stop.is_set():
            break
        if not realtime:
            while not listener.idle():      # a replay waits for its receives
                time.sleep(0.05)
        with listener.lock:
            if x is not None:
                listener.feed(x, t_end)
            now = time.time() if realtime else (listener.audio_end or 0.0)
            if now and (listener.step(now) or time.time() - last_write > 2.0):
                listener.write_state()
                last_write = time.time()
    if not realtime and listener.audio_end is not None:
        while not listener.idle():
            time.sleep(0.05)
        with listener.lock:
            ends = [slot_t0(q) + listener.dur + PB_TAIL_S + 1.0 for q in listener.slots]
            if ends:
                listener.step(max(max(ends), listener.audio_end))
        while not listener.idle():
            time.sleep(0.05)
        with listener.lock:
            listener.step(listener.now)     # retire the slots the worker finished
    with listener.lock:
        listener.log("audio ended")
        listener.write_state(listening=False)


__all__ = ["StreamFE", "FeRing", "slot_capture", "capture_read", "capture_prepare",
           "finish_receive", "DialLog", "same_dial", "restrict", "Tile", "LiveConfig", "LiveListener",
           "Stamper", "StdinSource", "WavSource", "run", "frame_seconds", "in_band", "centre_of",
           "utc_iso"]

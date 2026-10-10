"""The live QRSSTVAE listener: audio in, a tile per signal out (spec 9, build step 6).

Not a format module. `qrss_listen.py` is its command line, and the
desktop app's QRSS window is one of its front ends: the app pipes its
capture audio in and shows the tiles this writes.

    8 kHz audio chunks --StreamFE--> FE 4 kHz --> FeRing (in memory, ~33 min)
                                              \\-> PassbandStore (48 h, optional)
    every refresh_s, for each slot whose frame is on the air:
        slot_capture: the slot's FE so far, blanked and normalised, the
                      rest (future, and any holes) noise, erased
        detect on the preamble (until detect_until_s), then
        receive_pass, round A only, with the unheard spans erased
        --> tile: header, SNR, share received, picture so far
    when a slot's frame has ended:
        receive_slot over the whole slot (round B, whole-slot search),
        associate into the multi-pass store, render the accumulator
        --> the tile goes "complete"

The tiles are a directory: `state.json` (rewritten atomically after
every change) and `tiles/<id>.png`. Anything can show them; nothing in
here knows about a GUI.

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
it touches. Samples arriving early (a burst after a stall) are only
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

from . import frontend, picture, receiver
from .constants import CARRIER_BAND_HZ, FE_FS, FS, T_SYM
from .frame import FULL, FrameSpec
from .frontend import PB_LEAD_S, PB_TAIL_S, slot_t0
from .types import Detection, PassResult

FE_DECIM = FS // FE_FS            # 2
ALIGN = 16                        # audio blocks start on multiples of 16 (to_baseband's table)
HALO = 128                        # audio samples each side of a block (the FE FIR is 255 taps)
RESYNC_S = 1.0                    # clock disagreement that restarts the stream
EARLY_WINDOW_S = 30.0             # ...when audio has run ahead of the clock this long
RING_S = 2100.0                   # FE kept in memory: a FULL slot, its margins and slack
MERGE_HZ = receiver.MERGE_HZ      # a detection this close to a tile is that tile
PREAMBLE_S = 20                   # the preamble's span after t0 (660 symbols, 20.0 s)
FINISH_SEARCH_MIN_HEARD = 0.5     # the end-of-slot search over the whole slot needs this much heard
FINISH_MAX_CANDIDATES = 12        # ...and verifies at most this many candidates
FINISH_BUDGET_S = 180.0           # ...for at most this long (a false one costs ~1 min)
HEADERLESS_MAX_SNR_DB = -6.0      # a pass this strong whose header did not decode is not QRSSTVAE
RECOVER_HOURS = 3.0               # a restarted listener finishes slots that ended this recently
GUESS_NOTE = "no header yet: picture assumes the first pass of a mode A send"


def utc_iso(t: float) -> str:
    return datetime.fromtimestamp(t, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def frame_seconds(spec: FrameSpec) -> float:
    """Seconds from t0 to the end of the frame's last keyed symbol."""
    return spec.keyed_end_pos * T_SYM


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


def slot_capture(ring: FeRing, q: int, spec: FrameSpec, seed: int | None = None,
                 lead_s: float = PB_LEAD_S, tail_s: float = PB_TAIL_S):
    """(Prepared, erase spans, share of the frame heard) for slot q from the ring.

    The heard samples are blanked and normalised as `receiver.prepare`
    would; everything not heard (the future, and holes) is complex noise
    at the normalised level (median power 1) and is returned as spans to
    erase, in seconds after t0.
    """
    t0 = slot_t0(q)
    a = t0 * FE_FS - int(round(lead_s * FE_FS))
    b = t0 * FE_FS + int(round((frame_seconds(spec) + tail_s) * FE_FS))
    x, ok = ring.read(a, b)
    origin = t0 * FE_FS - a
    fe, keep = frontend.blank(x)
    keep = keep * ok
    fe, _ = frontend.normalise(fe, keep)
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


# --- tiles -----------------------------------------------------------------------------------

@dataclass
class Tile:
    """What the window shows for one signal in one slot."""
    id: str
    q: int
    slot_utc: str
    frame: str
    f_hz: float                     # audio Hz at t0
    status: str = "receiving"       # receiving | complete | lost
    snr_db: float | None = None     # SNR in 2500 Hz
    z_ref: float | None = None
    callsign: str | None = None
    grid: str | None = None
    picture_id: str | None = None   # hexadecimal
    mode: str | None = None         # A, B, C
    segment: int | None = None
    cw_text: str | None = None
    progress: float = 0.0           # share of the frame's time elapsed
    heard: float = 0.0              # share of the frame's audio heard
    received: float = 0.0           # share of the data latents with W > 0
    mean_w_db: float | None = None
    passes: int = 1                 # passes of this picture in the picture shown
    image: str | None = None        # path relative to the state directory
    image_rev: int = 0              # bumped every time the picture changes
    updated: float = 0.0
    note: str = ""

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


@dataclass
class LiveConfig:
    spec: FrameSpec = FULL
    refresh_s: float = 60.0           # time between live receives of a slot
    first_s: float = 25.0             # first receive this long after t0 (the preamble is 20 s)
    detect_until_s: float = 300.0     # acquisition re-runs at each refresh until then
    keep_done_s: float = 6 * 3600.0   # a finished tile stays listed this long
    estimator: str = "joint"
    precision: str = "fp32"
    model: str | None = None
    render: bool = True
    # Receive on a worker thread, so the loop keeps taking audio and
    # writing state.json while a receive runs (minutes on a busy band).
    # Off for tests, which step the listener synchronously.
    background: bool = False


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
        self.ring = FeRing(frame_seconds(self.cfg.spec) + PB_LEAD_S + PB_TAIL_S + 300.0)
        self.fe = StreamFE()
        self._early: collections.deque = collections.deque()
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
        self.busy: str | None = None       # what the worker is doing, for state.json
        self._jobs: queue.Queue | None = None
        if self.cfg.background:
            self._jobs = queue.Queue()
            threading.Thread(target=self._worker, daemon=True).start()
        self._load_previous()

    # -- logging and state ------------------------------------------------------------

    def log(self, msg: str) -> None:
        line = f"{utc_iso(self.now or time.time())} {msg}"
        self.lines = (self.lines + [line])[-50:]
        self._log(line)

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
            if t.get("status") != "receiving":
                if not t.get("callsign") and (t.get("snr_db") or -99.0) > HEADERLESS_MAX_SNR_DB:
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
        for n0, fe in self.fe.push(a, x):
            self.ring.write(n0, fe)
            if self.passband is not None:
                self.passband.write(n0 / FE_FS, fe)
        self.audio_end = (a + len(x) - 1) / FS

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
        that receive. The slots run newest first, each only when nothing
        live is waiting (see `step`). Returns the slots queued.
        """
        if self.passband is None or hours <= 0:
            return []
        dur = frame_seconds(self.cfg.spec)
        reader = _PassbandReader(self.passband)
        out = []
        for q in range(int(math.floor(now / 900.0)), int(math.floor(
                (now - hours * 3600.0 - dur) / 900.0)) - 1, -1):
            end = slot_t0(q) + dur + PB_TAIL_S
            if (not (now - hours * 3600.0 <= end <= now) or q in self.finished
                    or q in self.slots or any(r.q == q for r in self._recover)):
                continue
            t0 = slot_t0(q)
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
        return float(min(max((self.now - slot_t0(q)) / frame_seconds(self.cfg.spec), 0.0), 1.0))

    def active_slots(self, now: float) -> list[int]:
        """Slots whose frame has started and is not yet finished at `now`."""
        dur = frame_seconds(self.cfg.spec)
        lo = int(math.floor((now - dur - PB_TAIL_S - 1.0) / 900.0))
        return [q for q in range(lo, int(math.floor((now - 1.0) / 900.0)) + 1)
                if slot_t0(q) <= now < slot_t0(q) + dur + PB_TAIL_S]

    def step(self, now: float) -> bool:
        """Do whatever is due at `now` (unix seconds); True if anything changed."""
        self.now = now
        changed = False
        dur = frame_seconds(self.cfg.spec)
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
                    self._submit(self._finish, s)
                    changed = True
            elif now >= t0 + self.cfg.first_s and (
                    s.last_refresh is None or now - s.last_refresh >= self.cfg.refresh_s):
                if self._jobs is None or (self.busy is None and self._jobs.empty()):
                    s.last_refresh = now     # (when busy, it is still due next time)
                    self._submit(self._refresh, s)
                    changed = True
        if self._recover and self.idle():
            s = self._recover.pop(0)
            self.slots[s.q] = s
            self._submit(self._finish, s)
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

    # -- the worker -----------------------------------------------------------------------

    def _submit(self, job, s: SlotState) -> None:
        if self._jobs is None:
            job(s)
        else:
            self._jobs.put((job, s))

    def _worker(self) -> None:
        while True:
            job, s = self._jobs.get()
            self.busy = f"{job.__name__.strip('_')} {utc_iso(slot_t0(s.q) - 1)}"
            try:
                job(s)
            except Exception as e:       # one bad slot must not stop the listener
                with self.lock:
                    self.log(f"slot {utc_iso(slot_t0(s.q) - 1)}: {type(e).__name__}: {e}")
                    s.done = True
            finally:
                self.busy = None

    def idle(self) -> bool:
        """No receive queued or running."""
        return self._jobs is None or (self.busy is None and self._jobs.empty())

    def _match(self, s: SlotState, f_hz: float) -> Tile | None:
        best = None
        for t in s.tiles.values():
            if abs(t.f_hz - f_hz) < MERGE_HZ and (best is None or
                                                 abs(t.f_hz - f_hz) < abs(best.f_hz - f_hz)):
                best = t
        return best

    def _tile_for(self, s: SlotState, f_hz: float) -> Tile:
        t = self._match(s, f_hz)
        if t is None:
            tid = f"{s.q}-{int(round(f_hz))}"
            while tid in s.tiles:
                tid += "b"
            t = Tile(id=tid, q=s.q, slot_utc=utc_iso(slot_t0(s.q) - 1), frame=self.cfg.spec.name,
                     f_hz=float(f_hz), updated=self.now)
            s.tiles[tid] = t
            self.log(f"slot {t.slot_utc}: signal at {f_hz:.1f} Hz")
        return t

    def _refresh(self, s: SlotState) -> None:
        spec = self.cfg.spec
        t_start = time.time()
        t0 = slot_t0(s.q)
        with self.lock:
            _, ok = self.ring.read(t0 * FE_FS, (t0 + PREAMBLE_S) * FE_FS)
            if ok.mean() < 0.5:
                return      # joined after its preamble: only the end-of-slot search can find it
            prep, erase, heard = slot_capture(self.ring, s.q, spec)
            elapsed = self.now - t0
        if elapsed < self.cfg.detect_until_s or not s.dets:
            found = receiver.detect(prep, spec, live_only=True)
            for d in found:      # keep one detection per signal, the newest
                s.dets = [g for g in s.dets if abs(g.f_hz - d.f_hz) >= MERGE_HZ] + [d]
        if not s.dets:
            return
        passes = receiver.receive_slot(prep, spec, estimator=self.cfg.estimator, erase_s=erase,
                                       round_b=False, dets=list(s.dets))
        with self.lock:
            if s.done:
                return
            for p in passes:
                p = self._known_header(s, p)
                if not self._plausible(s, p):
                    continue
                t = self._tile_for(s, p.f_hz)
                self._update_tile(t, p, heard)
                t.status = "receiving"
                self.log(f"slot {utc_iso(slot_t0(s.q) - 1)}: {len(passes)} signal(s) at "
                     f"{100 * self._progress(s.q):.0f}% ({time.time() - t_start:.0f} s)")

    def _finish(self, s: SlotState) -> None:
        """The frame is over: receive the whole slot, store, render the accumulator.

        The search over the whole slot (for signals too weak to see on the
        preamble) runs only when most of the slot was heard, and verifies
        at most FINISH_MAX_CANDIDATES candidates for FINISH_BUDGET_S,
        strongest first: on a phone band, voices and carriers make
        candidates that each cost about a minute to reject, and the search
        ran for longer than a slot. With less heard, only the
        signals already found on the preamble are received.
        """
        spec = self.cfg.spec
        t_start = time.time()
        with self.lock:
            prep, erase, heard = slot_capture(s.reader or self.ring, s.q, spec)
            dets = list(s.dets)
        passes = []
        if heard >= FINISH_SEARCH_MIN_HEARD:
            passes = receiver.receive_slot(prep, spec, estimator=self.cfg.estimator,
                                           erase_s=erase, max_candidates=FINISH_MAX_CANDIDATES,
                                           budget_s=FINISH_BUDGET_S)
        elif dets and heard > 0.0:
            passes = receiver.receive_slot(prep, spec, estimator=self.cfg.estimator,
                                           erase_s=erase, dets=dets)
        with self.lock:
            self._finish_apply(s, passes, heard)
            what = "" if heard >= FINISH_SEARCH_MIN_HEARD else \
                f", {100 * heard:.0f}% heard: no whole-slot search"
            if s.reader is not None:
                what += ", from the passband store"
            self.log(f"slot {utc_iso(slot_t0(s.q) - 1)}: frame over, {len(passes)} pass(es)"
                     f"{what} ({time.time() - t_start:.0f} s)")

    def _finish_apply(self, s: SlotState, passes, heard: float) -> None:
        s.done = True
        passes = [self._known_header(s, p) for p in passes]
        passes = [p for p in passes if self._plausible(s, p)]
        seen = set()
        for p in passes:
            t = self._tile_for(s, p.f_hz)
            seen.add(t.id)
            key = None
            if self.store is not None:
                from .associate import associate

                try:
                    key, rule = associate(p, self.store)
                    t.note = f"stored ({rule or 'already a member'})" if key else "stored (provisional)"
                except Exception as e:         # never lose the tile over the store
                    t.note = f"store failed: {e}"
            self._update_tile(t, p, heard, key=key)
            t.status = "complete"
        for tid, t in s.tiles.items():
            if tid not in seen:
                t.status = "lost"
                t.updated = self.now

    def _known_header(self, s: SlotState, p: PassResult) -> PassResult:
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
        t = self._match(s, p.f_hz)
        h = self._headers.get(t.id) if t is not None else None
        if h is None:
            return p
        p = copy.copy(p)                  # (the receiver's result stays as it was)
        p.header = h
        return p

    def _plausible(self, s: SlotState, p: PassResult) -> bool:
        """Whether a pass is worth a tile (and the store).

        A pass needs a finite frequency inside the band receivers search.
        And with continuous audio every slot's capture holds half of each
        neighbouring slot's transmissions (FULL frames last two quarter
        hours), whose carriers the whole-slot search can lock onto at the
        wrong timing: a headerless pass within GHOST_HZ of a tile of
        another slot that did decode its header is taken to be that
        signal, and dropped. A headerless pass stronger than
        HEADERLESS_MAX_SNR_DB is dropped too.
        """
        f = float(p.f_hz)
        if not (math.isfinite(f) and CARRIER_BAND_HZ[0] <= f <= CARRIER_BAND_HZ[1]):
            return False
        if p.header is not None:
            return True
        # A header decodes far below this SNR, so a pass this strong
        # without one is something else locked on (a carrier, a birdie, a
        # voice on a phone band), not a QRSSTVAE signal.
        snr = float(p.report.snr2500_db)
        if not math.isfinite(snr) or snr > HEADERLESS_MAX_SNR_DB:
            return False
        others = [t for q, o in self.slots.items() if q != s.q for t in o.tiles.values()]
        others += [t for t in self.done_tiles if t.q != s.q]
        return not any(t.callsign and abs(t.f_hz - f) < receiver.GHOST_HZ for t in others)

    def _update_tile(self, t: Tile, p: PassResult, heard: float, key=None) -> None:
        r = p.report
        t.f_hz = float(p.f_hz)
        t.snr_db = float(r.snr2500_db) if np.isfinite(r.snr2500_db) else None
        t.z_ref = float(r.z_ref) if np.isfinite(r.z_ref) else None
        t.progress = self._progress(t.q)
        t.heard = float(heard)
        w = np.asarray(p.w, dtype=np.float64)
        t.received = float(np.mean(w > 0)) if len(w) else 0.0
        t.mean_w_db = receiver.mean_w_db(p) if t.received > 0 else None
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
            member = p.uid in acc.uids
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

class StdinSource:
    """Raw mono 8 kHz audio on stdin (float32 or int16, little-endian), read on a thread.

    Each chunk is stamped with the time it was read. The reading thread
    never waits on the receiver, so a receive that takes a minute loses
    no audio (the queue holds it).
    """

    def __init__(self, fmt: str = "f32", chunk_s: float = 0.25, stream=None):
        self.dtype = np.dtype("<f4") if fmt == "f32" else np.dtype("<i2")
        self.scale = 1.0 if fmt == "f32" else 1.0 / 32768.0
        self.nbytes = int(chunk_s * FS) * self.dtype.itemsize
        self.stream = stream if stream is not None else sys.stdin.buffer
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
                self.q.put((np.frombuffer(b[:k], self.dtype).astype(np.float64) * self.scale, t))
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
    listener.log(f"listening: {source.name}, frame {listener.cfg.spec.name}")
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
            ends = [slot_t0(q) + frame_seconds(listener.cfg.spec) + PB_TAIL_S + 1.0
                    for q in listener.slots]
            if ends:
                listener.step(max(max(ends), listener.audio_end))
        while not listener.idle():
            time.sleep(0.05)
        with listener.lock:
            listener.step(listener.now)     # retire the slots the worker finished
    with listener.lock:
        listener.log("audio ended")
        listener.write_state(listening=False)


__all__ = ["StreamFE", "FeRing", "slot_capture", "Tile", "LiveConfig", "LiveListener",
           "StdinSource", "WavSource", "run", "frame_seconds", "utc_iso"]

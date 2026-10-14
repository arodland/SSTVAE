"""Parity of native/core/qrss/sources (sstvae_native.qrss.sources) with
live.py's audio sources: Stamper, StdinSource (the C++ StreamSource) and
WavSource. tests/qrss_native/sources.py substitutes the Stamper under
--native-qrss=rx (test_qrss_live.py's stall, gap and start-up tests then
run on it, and the reference's StdinSource stamps through it).

The contract (the bar), and in brackets the worst measured (2026-10-14):

- Stamper: the same chunks (samples bit for bit) at the same times (the
  same float) on 400 random streams -- steady flow, stalls released as a
  burst, backlogs, a start-up queue, real gaps, read-time jitter, chunks of
  1 to 4000 samples and empty ones, hold_s from 0 to 8 s, times at the
  half-sample boundaries round() splits to even and start - next exactly
  at RESYNC_S * FS; a NaN or infinite time refused with the reference's
  exception and text, and the chunk kept held after it as the reference
  keeps it [exact everywhere];
- StdinSource over a pipe with the same scripted writes and the same
  injected clock (time.time patched on the reference's side): the same
  chunks at the same times and the same number of clock readings, f32 and
  s16, a sample split across two reads (every byte offset), an odd byte
  left at the end (dropped), a backlog already in the pipe read in
  chunk_s pieces, an empty input; the name; eof only after the end
  [exact];
- WavSource on a file written by scipy: the same chunks (samples bit for
  bit), times, clock readings and sleeps (speed 0, 2.5 and 1), x, start,
  chunk and name, and the reference's refusals (a missing file, a file that
  is not a WAV) [exact];
- beyond the reference: close() stops a reader blocked on a pipe that
  never writes; a read error (a directory) ends the stream with an OSError
  where the reference's reader dies and its chunks() never ends.
"""

from __future__ import annotations

import math
import os
import random
import sys
import threading
import time

import numpy as np
import pytest

from qrss_native import require
from sstvae.qrss import live

FS = 8000


@pytest.fixture
def S(native):
    return require(native, "sources")


# --- Stamper ---------------------------------------------------------------------------------

def _same_chunks(got, want, what=""):
    assert len(got) == len(want), what
    for (gx, gt), (wx, wt) in zip(got, want):
        assert np.asarray(gx).dtype == np.float64
        assert np.asarray(gx).tobytes() == np.asarray(wx, dtype=np.float64).tobytes(), what
        assert type(gt) is float and gt == wt and repr(gt) == repr(wt), (what, gt, wt)


def _stream(rng: random.Random):
    """(n, t) pushes: audio heard from T on, read with stalls, gaps, backlogs."""
    T = rng.choice([0.0, 1_000_000.0, 1_760_000_123.4375, rng.uniform(1e9, 2e9)])
    heard = T                        # end of the audio heard so far
    out = []
    if rng.random() < 0.4:           # a start-up queue: seconds read at once
        q = rng.uniform(0.5, 9.0)
        while q > 0:
            n = rng.randint(1, 4000)
            heard += n / FS
            q -= n / FS
            out.append((n, T + 9.0 + 1e-4 * len(out)))
    for _ in range(rng.randint(5, 80)):
        kind = rng.random()
        n = rng.choice([0, 1, 2, rng.randint(1, 4000), 2000, 2000])
        if kind < 0.6:               # steady, a little jitter
            heard += n / FS
            out.append((n, heard + rng.choice([0.0, 0.0, rng.uniform(0, 0.05)])))
        elif kind < 0.75:            # a stall: k chunks read at once, late
            late = rng.uniform(0.1, 7.0)
            for _ in range(rng.randint(1, 15)):
                m = rng.randint(1, 4000)
                heard += m / FS
                out.append((m, heard + late + rng.uniform(0, 1e-3)))
        elif kind < 0.85:            # a real gap
            heard += rng.uniform(0.5, 20.0)
            heard += n / FS
            out.append((n, heard))
        elif kind < 0.92:            # a time on a half-sample boundary
            heard += n / FS
            k = math.floor(heard * FS)
            out.append((n, (k + 0.5) / FS))
        else:                        # read early (a clock step back)
            heard += n / FS
            out.append((n, heard - rng.uniform(0, 3.0)))
    return out


def test_stamper_fuzz(S):
    rng = random.Random(20261014)
    for case in range(400):
        hold = rng.choice([live.STAMP_HOLD_S, 0.0, 0.25, rng.uniform(0, 8.0)])
        a, b = live.Stamper(hold), S.Stamper(hold)
        assert b.hold_s == a.hold_s
        drng = np.random.default_rng(case)
        for k, (n, t) in enumerate(_stream(rng)):
            x = drng.standard_normal(n)
            _same_chunks(b.push(x, t), a.push(x, t), f"case {case} push {k}")
            if rng.random() < 0.02:
                _same_chunks(b.flush(), a.flush(), f"case {case} flush {k}")
        _same_chunks(b.flush(), a.flush(), f"case {case} final flush")
        assert b.flush() == [] == a.flush()


def test_stamper_resync_boundary(S):
    """start - next exactly RESYNC_S * FS (held) and one more (held), one
    less (out): Python's exact int-against-float comparison."""
    for d in (-1, 0, 1, int(live.RESYNC_S * FS) - 1, int(live.RESYNC_S * FS),
              int(live.RESYNC_S * FS) + 1):
        a, b = live.Stamper(), S.Stamper()
        x = np.arange(100.0)
        for st in (a, b):
            st.push(x, 1000.0)
        _same_chunks(b.push(x, 1006.0), a.push(x, 1006.0))          # out: hold passed
        t = (1006.0 * FS + d + 100) / FS
        _same_chunks(b.push(x, t), a.push(x, t), f"d={d}")
        _same_chunks(b.flush(), a.flush())


@pytest.mark.parametrize("bad, exc, text", [
    (math.nan, ValueError, "cannot convert float NaN to integer"),
    (math.inf, OverflowError, "cannot convert float infinity to integer"),
    (-math.inf, OverflowError, "cannot convert float infinity to integer"),
])
def test_stamper_refusals(S, bad, exc, text):
    """A time round() refuses: the reference's error, and the chunk stays
    held (so every later push and the flush refuse too)."""
    for st in (live.Stamper(), S.Stamper()):
        st.push(np.ones(3), 100.0)
        with pytest.raises(exc, match=text):
            st.push(np.ones(2), bad)
        with pytest.raises(exc, match=text):
            st.push(np.ones(2), 101.0)
        with pytest.raises(exc, match=text):
            st.flush()


# --- StdinSource -----------------------------------------------------------------------------

PIPE_BUF = 4096


class _Script:
    """Writes `blobs` to a pipe one at a time, each once the reader has
    stamped the read before it (so every read is exactly one blob), the
    clock answering `times` in order. `pre` is written before the source
    exists (a backlog: read in chunk-sized pieces, `pre_reads` of them)."""

    def __init__(self, blobs, times, pre=b"", pre_reads=0):
        self.r, self.w = os.pipe()
        if pre:
            assert os.write(self.w, pre) == len(pre)
        self.blobs, self.times, self.pre_reads = blobs, list(times), pre_reads
        self.calls = 0
        self.stamped = threading.Semaphore(0)
        self.failed = None

    def clock(self):
        t = self.times[self.calls]
        self.calls += 1
        self.stamped.release()
        return t

    def start(self):
        threading.Thread(target=self._write, daemon=True).start()

    def _write(self):
        try:
            for _ in range(self.pre_reads):
                if not self.stamped.acquire(timeout=20):
                    raise TimeoutError("no read of the backlog")
            for b in self.blobs:
                os.write(self.w, b)
                if not self.stamped.acquire(timeout=20):
                    raise TimeoutError("no read")
        except BaseException as e:      # noqa: BLE001 -- reported by the test
            self.failed = e
        finally:
            os.close(self.w)


def _drain(src, deadline_s=60.0):
    out, t0 = [], time.monotonic()
    for x, t in src.chunks(timeout=0.2):
        if x is not None:
            out.append((x, t))
        assert time.monotonic() - t0 < deadline_s, "the source never ended"
    assert src.eof
    return out


def _run_python(monkeypatch, fmt, chunk_s, blobs, times, pre=b"", pre_reads=0):
    sc = _Script(blobs, times, pre, pre_reads)
    stream = os.fdopen(sc.r, "rb")
    with monkeypatch.context() as mp:
        mp.setattr(live.time, "time", sc.clock)
        src = live.StdinSource(fmt, chunk_s, stream=stream)
        assert not src.eof
        sc.start()
        out = _drain(src)
    stream.close()
    assert sc.failed is None, sc.failed
    return out, sc.calls, src.name


def _run_native(S, fmt, chunk_s, blobs, times, pre=b"", pre_reads=0):
    sc = _Script(blobs, times, pre, pre_reads)
    stream = os.fdopen(sc.r, "rb")
    src = S.StdinSource(fmt, chunk_s, stream=stream, clock=sc.clock)
    assert not src.eof
    sc.start()
    out = _drain(src)
    assert src.error is None
    src.close()
    stream.close()
    assert sc.failed is None, sc.failed
    return out, sc.calls, src.name


def _times(rng, nbytes_list, itemsize, T=1_700_000_000.0):
    """Read times for a stream of writes: steady, with a stall and a gap."""
    heard, out = T, []
    for k, nb in enumerate(nbytes_list):
        heard += nb / itemsize / FS
        late = 3.0 if 10 <= k < 14 else 0.0          # a stall
        if k == 30:
            heard += 12.0                             # a gap
        out.append(heard + late + rng.uniform(0, 2e-3))
    return out


def _payload(rng, n, fmt):
    a = np.random.default_rng(rng.randint(0, 2**31)).standard_normal(n)
    if fmt == "f32":
        a = a.astype("<f4")
        a[::97] = np.float32(np.nan)
        a[1::89] = np.float32(np.inf)
        return a.tobytes()
    return (a * 9000).clip(-32768, 32767).astype("<i2").tobytes()


@pytest.mark.parametrize("fmt", ["f32", "s16"])
def test_stdin_source_scripted(S, monkeypatch, fmt):
    """Random write sizes (samples split across reads at every offset), a
    stall and a gap, an odd byte at the end: the same chunks and times."""
    rng = random.Random(7 if fmt == "f32" else 8)
    itemsize = 4 if fmt == "f32" else 2
    chunk_s = 0.25
    cap = min(int(chunk_s * FS) * itemsize, PIPE_BUF)
    sizes = [rng.choice([1, 2, 3, itemsize + 1, rng.randint(1, cap), cap]) for _ in range(60)]
    sizes += [1]                                     # the odd byte left at the end
    data = _payload(rng, sum(sizes) // itemsize + 2, fmt)
    blobs, at = [], 0
    for s in sizes:
        blobs.append(data[at:at + s])
        at += s
    times = _times(rng, sizes, itemsize)
    ref, ref_calls, ref_name = _run_python(monkeypatch, fmt, chunk_s, blobs, times)
    got, got_calls, got_name = _run_native(S, fmt, chunk_s, blobs, times)
    assert got_calls == ref_calls == len(blobs)
    assert got_name == ref_name == f"stdin ({fmt}, 8 kHz)"
    _same_chunks(got, ref)
    assert sum(len(x) for x, _ in ref) == at // itemsize


@pytest.mark.filterwarnings("ignore:invalid value encountered in cast:RuntimeWarning")
@pytest.mark.parametrize("fmt, chunk_s", [("f32", 0.25), ("s16", 0.1), ("f32", 0.01)])
def test_stdin_source_backlog(S, monkeypatch, fmt, chunk_s):
    """Seconds already in the pipe: read chunk_s at a time (read1's one raw
    read), then audio as it arrives."""
    rng = random.Random(11)
    itemsize = 4 if fmt == "f32" else 2
    nbytes = int(chunk_s * FS) * itemsize
    pre = _payload(rng, 4 * nbytes // itemsize, fmt)[: 3 * nbytes + 5]
    pre_reads = -(-len(pre) // nbytes)
    cap = min(nbytes, PIPE_BUF)
    sizes = [rng.randint(1, cap) for _ in range(20)]
    data = _payload(rng, sum(sizes) // itemsize + 2, fmt)
    blobs, at = [], 0
    for s in sizes:
        blobs.append(data[at:at + s])
        at += s
    times = [5000.0 + 1e-4 * k for k in range(pre_reads)] + _times(rng, sizes, itemsize, 5003.0)
    ref, ref_calls, _ = _run_python(monkeypatch, fmt, chunk_s, blobs, times, pre, pre_reads)
    got, got_calls, _ = _run_native(S, fmt, chunk_s, blobs, times, pre, pre_reads)
    assert got_calls == ref_calls == pre_reads + len(blobs)
    _same_chunks(got, ref)


def test_stdin_source_empty(S, monkeypatch):
    """Nothing on the input: no chunks, no clock reading, the end."""
    ref, ref_calls, _ = _run_python(monkeypatch, "f32", 0.25, [], [])
    got, got_calls, _ = _run_native(S, "f32", 0.25, [], [])
    assert ref == got == [] and ref_calls == got_calls == 0


def test_stdin_source_timeout_and_close(S):
    """A pipe nobody writes: (None, None) after each timeout, as the
    reference's chunks() yields; close() stops the reader (the stream then
    ends); a negative timeout is queue.get's refusal."""
    r, w = os.pipe()
    try:
        src = S.StdinSource("f32", 0.25, stream=r)
        it = src.chunks(timeout=0.05)
        t0 = time.monotonic()
        assert next(it) == (None, None)
        assert time.monotonic() - t0 >= 0.04
        assert src.next(0.0) is None and not src.eof
        with pytest.raises(ValueError, match="'timeout' must be a non-negative number"):
            src.next(-1.0)
        os.write(w, np.arange(3, dtype="<f4").tobytes())
        src.close()                      # what was read stays queued, then the end
        rest = [c for c in it]
        assert src.eof and src.error is None
        assert sum(len(x) for x, _ in rest if x is not None) in (0, 3)
    finally:
        os.close(r)
        os.close(w)


@pytest.mark.skipif(sys.platform == "win32", reason="a directory as a descriptor")
def test_stdin_source_read_error_ends_the_stream(S, tmp_path):
    """A read error ends the stream (the reference's reader thread dies
    and its chunks() never ends), the error kept."""
    fd = os.open(tmp_path, os.O_RDONLY)
    try:
        src = S.StdinSource("f32", stream=fd)
        assert [c for c in src.chunks(timeout=1.0) if c[0] is not None] == []
        assert src.eof
        assert isinstance(src.error, OSError) and src.error.errno == 21   # EISDIR
    finally:
        os.close(fd)


# --- WavSource -------------------------------------------------------------------------------

class _Clock:
    def __init__(self, times):
        self.times, self.calls, self.sleeps = list(times), 0, []

    def time(self):
        t = self.times[self.calls]
        self.calls += 1
        return t

    def sleep(self, s):
        self.sleeps.append(s)


def _wav(tmp_path, n, rate=8000, name="rec.wav"):
    from scipy.io import wavfile

    p = tmp_path / name
    x = np.random.default_rng(n).integers(-32768, 32767, n, dtype=np.int16)
    wavfile.write(p, rate, x)
    return p


@pytest.mark.parametrize("n, speed, chunk_s, start", [
    (8000 * 7 + 123, 0.0, 1.0, 1_760_000_000.0),
    (8000 * 3 + 1, 2.5, 1.0, 1_760_000_900.75),
    (5000, 1.0, 0.3, 1.5),
    (0, 1.0, 1.0, 1_760_000_000.0),
])
def test_wav_source(S, tmp_path, monkeypatch, n, speed, chunk_s, start):
    p = _wav(tmp_path, n)
    times = [100.0 + 0.37 * k * k for k in range(64)]
    ca, cb = _Clock(times), _Clock(times)
    with monkeypatch.context() as mp:
        mp.setattr(live.time, "time", ca.time)
        mp.setattr(live.time, "sleep", ca.sleep)
        a = live.WavSource(p, start, speed=speed, chunk_s=chunk_s)
        ref = list(a.chunks())
    b = S.WavSource(p, start, speed=speed, chunk_s=chunk_s, clock=cb.time, sleep=cb.sleep)
    assert not b.eof
    got = list(b.chunks())
    assert a.eof and b.eof
    _same_chunks(got, ref)
    assert (ca.calls, ca.sleeps) == (cb.calls, cb.sleeps)
    assert b.x.tobytes() == a.x.tobytes() and b.start == a.start and b.chunk == a.chunk
    assert b.speed == a.speed and b.name == a.name


def test_wav_source_resampled_and_named(S, tmp_path):
    """A 44.1 kHz file (read_wav resamples) under a str path: the same
    name, and the samples to 1e-12 of the scale (wavread.hpp's resampler,
    as test_qrss_native_receiver_slot.py holds it)."""
    p = _wav(tmp_path, 44100 * 2 + 17, rate=44100, name="r 44k.wav")
    a = live.WavSource(str(p), 1e9)
    b = S.WavSource(str(p), 1e9, clock=lambda: 0.0)
    assert b.x.shape == a.x.shape and b.name == a.name
    err = float(np.max(np.abs(b.x - a.x)))
    assert err <= 1e-12 * float(np.max(np.abs(a.x))), err


def test_wav_source_refusals(S, tmp_path):
    missing = tmp_path / "nope.wav"
    for make in (live.WavSource, S.WavSource):
        with pytest.raises(FileNotFoundError) as e:
            make(missing, 0.0)
        assert e.value.filename == str(missing)
    bad = tmp_path / "bad.wav"
    bad.write_bytes(b"not a wav file at all")
    errs = []
    for make in (live.WavSource, S.WavSource):
        with pytest.raises(Exception) as e:
            make(bad, 0.0)
        errs.append((type(e.value), str(e.value)))
    assert errs[0] == errs[1]

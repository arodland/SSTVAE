"""The rolling capture buffer shared by the decode loop and the display.

Long enough to hold a whole mode-C transmission plus margin, so a
mid-stream lock can still decode the frames that arrived *before* sync
was acquired (see engine.py).
"""

import threading

import numpy as np

from ..config import FS


class RingBuffer:
    """Fixed-length circular float64 audio buffer, thread-safe."""

    def __init__(self, seconds: float, fs: int = FS):
        self.n = int(seconds * fs)
        self.buf = np.zeros(self.n, dtype=np.float64)
        self.write_pos = 0
        self.total_written = 0
        self.lock = threading.Lock()

    def write(self, chunk: np.ndarray) -> None:
        """Append captured audio. Called from the audio callback, which
        **must never block** -- see the note on `snapshot`.

        There is exactly one writer, so where the data goes depends only
        on state this thread owns. The lock is therefore held just long
        enough to publish two integers, not for the array copy.
        """
        chunk = np.asarray(chunk, dtype=np.float64).reshape(-1)
        n = len(chunk)
        if n == 0:
            return
        pos = self.write_pos
        if n >= self.n:
            self.buf[:] = chunk[-self.n :]
            new_pos = 0
        else:
            end = pos + n
            if end <= self.n:
                self.buf[pos:end] = chunk
            else:
                k = self.n - pos
                self.buf[pos:] = chunk[:k]
                self.buf[: end - self.n] = chunk[k:]
            new_pos = end % self.n
        # Publish only after the samples are in place.
        with self.lock:
            self.write_pos = new_pos
            self.total_written += n

    def snapshot(self) -> tuple[np.ndarray, int]:
        """Chronological copy of everything currently held (oldest
        first), and the total sample count ever written (for display).

        **The copy happens outside the lock, on purpose.** Holding it
        across a copy of the whole buffer -- 8 MB at the default 130 s --
        blocks the audio callback's `write`, and a blocked audio callback
        means PortAudio discards input. That was a real bug, and a
        vicious one: the decode loop calls this every `poll_interval`, so
        the receiver tore a hole in its own audio every 5 seconds, and
        the holes *grew* as the buffer filled and the copy got slower.
        Measured against a simultaneous clean capture of the same
        playback: losses of 85 samples rising to 235, one per poll,
        1718 samples over 50 s. Enough to put the picture 5 dB down and
        break the beacon, while still syncing and reporting every frame.

        The tradeoff is that the writer may overwrite the oldest samples
        while they are being copied, once the buffer has wrapped. That
        costs at most the few hundred samples produced during the copy,
        at the far end of a 130-second history, and the decoder is
        reconstructing from weighted latents that tolerate it. Losing
        *live* audio does not tolerate anything.
        """
        with self.lock:
            write_pos = self.write_pos
            total = self.total_written
        if total < self.n:
            return self.buf[:total].copy(), total
        return np.concatenate([self.buf[write_pos:], self.buf[:write_pos]]), total

    def tail(self, n: int) -> np.ndarray:
        """The most recent `n` samples, oldest first (shorter if less has
        been captured).

        A spectrum display wants a few thousand samples many times a
        second, and `snapshot()` copies the entire buffer -- ~8 MB for
        the default 130 s -- which is fine at the decode loop's 5 s poll
        and ruinous at 20 fps.
        """
        n = min(n, self.n)
        with self.lock:
            n = min(n, self.total_written)
            if n == 0:
                return np.zeros(0, dtype=np.float64)
            start = self.write_pos - n
            if start >= 0:
                return self.buf[start : self.write_pos].copy()
            return np.concatenate([self.buf[start:], self.buf[: self.write_pos]])

    def clear(self) -> None:
        """Drop everything captured so far, keeping the sample counter
        monotonic.

        Not currently on the resume-after-transmit path: both callers
        that need to keep our own sidetone out of the decoder (the
        "start receiving" button and resume-after-transmit) instead
        discard the whole `RingBuffer` and construct a fresh one, which
        starts `decode_loop` over from scratch -- so `blind_acc` and
        every other loop-local accumulator get a clean slate too, not
        just the audio. That leaves `total_written` restarting at 0
        rather than staying monotonic through the gap, which is fine
        because nothing survives the restart that could still be
        indexing against the old count. This method stays as a lower-
        overhead primitive (no reallocation, counter stays meaningful)
        for a caller that wants to wipe the audio in place without
        restarting the loop -- but pairing it with such a resume would
        need its own explicit reset of `blind_acc`/`blind_acc_pushed`
        and the other `decode_loop` locals, since none of those are
        reachable from here.
        """
        with self.lock:
            self.buf[:] = 0.0
            self.write_pos = self.total_written % self.n

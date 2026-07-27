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

        Used when resuming receive after a transmission: the buffer is
        full of our own sidetone, and the decode loop would happily lock
        onto it and 'receive' the picture we just sent. Leaving
        `total_written` alone keeps every absolute sample position the
        loop has recorded (finished_starts) still meaningful.
        """
        with self.lock:
            self.buf[:] = 0.0
            self.write_pos = self.total_written % self.n

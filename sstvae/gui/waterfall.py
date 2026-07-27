"""Scrolling spectrum display, plus the input level meter beside it.

Drawn straight into a numpy RGB buffer wrapped as a QImage rather than
going through a plotting library: this repaints ~20 times a second for
as long as the application is running, and it has to cost essentially
nothing next to the decode loop it shares a machine with.

The audio comes from the same `RingBuffer` the decoder reads, via
`tail()` -- a display-sized slice rather than the whole 130-second
buffer.
"""

import numpy as np
from PySide6.QtCore import QSize, Qt, QTimer
from PySide6.QtGui import QColor, QImage, QPainter, QPen
from PySide6.QtWidgets import QSizePolicy, QWidget

from ..config import CARRIER0, FS, NC, RS

NFFT = 1024
BIN_HZ = FS / NFFT
DISPLAY_HZ = 3000.0  # an SSB receiver's passband; the signal lives inside it
N_BINS = int(DISPLAY_HZ / BIN_HZ)
# Rows of history kept. Sized for the tall, narrow column the widget now
# lives in down the right-hand side of the receive panel: at ~20 fps this
# is around half a minute of history, and enough rows that a full-height
# pane isn't drawn from a handful of stretched ones.
HISTORY = 640

# The occupied band, marked so the operator can see whether the signal is
# sitting where the modem expects it.
BAND_LO_HZ = CARRIER0 - RS
BAND_HI_HZ = CARRIER0 + NC * RS

DB_FLOOR, DB_CEIL = -95.0, -20.0


def _colormap() -> np.ndarray:
    """256-entry black -> blue -> green -> yellow -> white ramp."""
    stops = [
        (0.00, (0, 0, 0)),
        (0.25, (0, 0, 140)),
        (0.50, (0, 170, 90)),
        (0.75, (245, 235, 40)),
        (1.00, (255, 255, 255)),
    ]
    lut = np.zeros((256, 3), dtype=np.uint8)
    xs = np.linspace(0, 1, 256)
    for ch in range(3):
        lut[:, ch] = np.interp(
            xs, [s[0] for s in stops], [s[1][ch] for s in stops]
        ).astype(np.uint8)
    return lut


_LUT = _colormap()
_WINDOW = np.hanning(NFFT)


class WaterfallWidget(QWidget):
    """Spectrum history, newest row at the top."""

    def __init__(self, ring, parent=None, fps: int = 20):
        super().__init__(parent)
        self._ring = ring
        self._rows = np.zeros((HISTORY, N_BINS, 3), dtype=np.uint8)
        self._peak = 0.0
        self._clipping = False
        # Narrow minimum: the frequency axis is scaled to whatever width
        # the splitter gives it, and demanding all N_BINS pixels would
        # stop the operator from shrinking the column in favour of the
        # picture.
        self.setMinimumWidth(160)
        self.setMinimumHeight(140)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(max(1, int(1000 / fps)))

    def sizeHint(self):
        return QSize(280, 600)

    # --- data ----------------------------------------------------------
    def set_ring(self, ring) -> None:
        self._ring = ring

    def clear(self) -> None:
        self._rows[:] = 0
        self._peak = 0.0
        self.update()

    def _tick(self) -> None:
        if self._ring is None:
            return
        block = self._ring.tail(NFFT)
        if len(block) < NFFT:
            return

        self._peak = float(np.max(np.abs(block)))
        # PortAudio hands back floats; anything at or over unity has
        # already been clipped somewhere upstream in the capture chain.
        self._clipping = self._peak >= 0.99

        spec = np.abs(np.fft.rfft(block * _WINDOW))[:N_BINS]
        db = 20.0 * np.log10(spec / NFFT + 1e-12)
        norm = np.clip((db - DB_FLOOR) / (DB_CEIL - DB_FLOOR), 0.0, 1.0)
        row = _LUT[(norm * 255).astype(np.uint8)]

        self._rows[1:] = self._rows[:-1]
        self._rows[0] = row
        self.update()

    # --- painting -------------------------------------------------------
    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        buf = np.ascontiguousarray(self._rows)
        img = QImage(buf.data, N_BINS, HISTORY, 3 * N_BINS, QImage.Format_RGB888)
        painter.drawImage(self.rect(), img)

        self._draw_band_markers(painter)
        self._draw_level_meter(painter)

    def _draw_band_markers(self, painter: QPainter) -> None:
        w = self.width()
        scale = w / DISPLAY_HZ
        pen = QPen(QColor(255, 255, 255, 110))
        pen.setStyle(Qt.DashLine)
        painter.setPen(pen)
        for hz in (BAND_LO_HZ, BAND_HI_HZ):
            x = int(hz * scale)
            painter.drawLine(x, 0, x, self.height())

        # The column is user-resizable now, so the caption has to earn its
        # place: drop it rather than let it run off the edge or overprint
        # the spectrum.
        label = f"SSTVAE {BAND_LO_HZ:.0f}-{BAND_HI_HZ:.0f} Hz"
        x_label = int(BAND_LO_HZ * scale) + 4
        if x_label + painter.fontMetrics().horizontalAdvance(label) < w - 12:
            painter.setPen(QColor(0, 0, 0, 160))
            painter.drawText(x_label + 1, 15, label)  # shadow, for contrast
            painter.setPen(QColor(255, 255, 255, 220))
            painter.drawText(x_label, 14, label)
        # A 1 kHz grid, so the operator can read where a signal sits.
        # Drawn with a shadow: the ticks sit over whatever the spectrum
        # happens to be doing at the bottom of the pane, which is often
        # bright.
        h = self.height()
        for hz in range(1000, int(DISPLAY_HZ), 1000):
            x = int(hz * scale)
            painter.setPen(QColor(0, 0, 0, 150))
            painter.drawLine(x + 1, h - 8, x + 1, h)
            painter.drawText(x + 4, h - 3, f"{hz // 1000}k")
            painter.setPen(QColor(255, 255, 255, 190))
            painter.drawLine(x, h - 8, x, h)
            painter.drawText(x + 3, h - 4, f"{hz // 1000}k")

    def _draw_level_meter(self, painter: QPainter) -> None:
        """A thin bar down the right edge: enough to set soundcard gain,
        which is the one audio adjustment that actually matters."""
        w, h = self.width(), self.height()
        bar_w = 8
        x0 = w - bar_w - 2
        painter.fillRect(x0, 2, bar_w, h - 4, QColor(0, 0, 0, 140))

        # dBFS, so the useful range isn't crushed into the top of a
        # linear bar.
        db = 20.0 * np.log10(max(self._peak, 1e-6))
        frac = float(np.clip((db + 60.0) / 60.0, 0.0, 1.0))
        filled = int((h - 4) * frac)
        if self._clipping:
            color = QColor(255, 60, 60)
        elif frac > 0.85:
            color = QColor(255, 190, 60)
        else:
            color = QColor(90, 220, 120)
        painter.fillRect(x0, h - 2 - filled, bar_w, filled, color)

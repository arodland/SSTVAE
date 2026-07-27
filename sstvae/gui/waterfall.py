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

# The backing image is kept exactly the widget's size, so the painter
# never rescales it.
#
# Scrolling is clean when the destination is an integer multiple of the
# source: at a k-times upscale, source row i lands at exactly i*k and a
# one-row shift moves the picture by exactly k pixels. The bad direction
# is downscaling, which is where this widget was -- 640 rows into a pane
# a few hundred pixels tall. There, source row i lands at
# floor(i * height / rows), a one-row shift moves the picture by a
# fraction of a pixel, and every frame re-quantises differently, so the
# rows crawl and shimmer.
#
# k = 1 is the simplest member of that family and the best one here: the
# scroll advances a single pixel per tick (k > 1 would jump k pixels and
# hold k times less history for the same pane).
#
# The frequency axis has the same problem in the same direction -- 384
# bins squeezed into a ~280 px column -- so rows are reduced to the
# widget's width when they are computed rather than by the painter; see
# `reduce_to_width`.
#
# History depth therefore follows the widget height: at ~20 fps a
# 700-pixel pane holds a bit over half a minute.

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


def reduce_to_width(values: np.ndarray, width: int) -> np.ndarray:
    """Map a spectrum onto exactly `width` columns.

    Peak-hold rather than point-sampling when shrinking: the carriers are
    one or two bins wide and about six bins apart, so taking every k'th
    bin drops some of them outright and leaves a ragged comb where the
    signal should be a solid block. Taking the maximum over each output
    column's bins keeps every carrier visible.
    """
    n = len(values)
    if width <= 0:
        return values[:0]
    if width == n:
        return values
    if width > n:  # upscaling: interpolate rather than repeat, to avoid
        # a blocky frequency axis on a wide pane
        return np.interp(np.linspace(0, n - 1, width), np.arange(n), values)
    # Group boundaries are strictly increasing because n >= width.
    return np.maximum.reduceat(values, (np.arange(width) * n) // width)


class WaterfallWidget(QWidget):
    """Spectrum history, newest row at the top."""

    def __init__(self, ring, parent=None, fps: int = 20):
        super().__init__(parent)
        self._ring = ring
        # Allocated on first use / resize, always exactly widget-sized.
        self._rgb: np.ndarray | None = None
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
        if self._rgb is not None:
            self._rgb[:] = 0
        self._peak = 0.0
        self.update()

    def _ensure_buffer(self) -> np.ndarray:
        """The backing image, resized to the widget if it has changed.

        Existing history is carried across so a resize doesn't blank the
        display: rows are kept as-is (they are already one pixel each)
        and columns are point-resampled, which is good enough for pixels
        that are only scrolling off anyway.
        """
        h, w = max(1, self.height()), max(1, self.width())
        old = self._rgb
        if old is not None and old.shape[:2] == (h, w):
            return old

        new = np.zeros((h, w, 3), dtype=np.uint8)
        if old is not None:
            rows = min(h, old.shape[0])
            if old.shape[1] == w:
                new[:rows] = old[:rows]
            else:
                cols = ((np.arange(w) * old.shape[1]) // w).clip(0, old.shape[1] - 1)
                new[:rows] = old[:rows][:, cols]
        self._rgb = new
        return new

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._ensure_buffer()

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

        buf = self._ensure_buffer()
        spec = np.abs(np.fft.rfft(block * _WINDOW))[:N_BINS]
        db = 20.0 * np.log10(spec / NFFT + 1e-12)
        # Reduced to the display width here, not by the painter: see
        # reduce_to_width.
        db = reduce_to_width(db, buf.shape[1])
        norm = np.clip((db - DB_FLOOR) / (DB_CEIL - DB_FLOOR), 0.0, 1.0)

        buf[1:] = buf[:-1]  # one row = one pixel, so this is a 1 px scroll
        buf[0] = _LUT[(norm * 255).astype(np.uint8)]
        self.update()

    # --- painting -------------------------------------------------------
    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        buf = np.ascontiguousarray(self._ensure_buffer())
        h, w = buf.shape[:2]
        img = QImage(buf.data, w, h, 3 * w, QImage.Format_RGB888)
        # 1:1 by construction, so this is a blit and not a rescale.
        painter.drawImage(0, 0, img)

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

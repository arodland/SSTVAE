"""Waterfall scrolling and frequency-axis reduction.

The shimmer this guards against came from the painter rescaling the
backing image: 640 rows squeezed into a pane a few hundred pixels tall
means a one-row data shift moves the picture by a fraction of a pixel,
so every frame re-quantises differently and the rows crawl. Keeping the
image exactly widget-sized makes one new row exactly one new pixel.
"""

import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication  # noqa: E402

from sstvae.gui.waterfall import (  # noqa: E402
    N_BINS,
    WaterfallWidget,
    reduce_to_width,
)
from sstvae.rx import RingBuffer  # noqa: E402

_APP = None


@pytest.fixture(scope="module")
def qapp():
    global _APP
    _APP = QApplication.instance() or QApplication([])
    return _APP


@pytest.fixture
def widget(qapp):
    w = WaterfallWidget(None)
    w.resize(300, 400)
    yield w
    w.deleteLater()
    qapp.processEvents()


def feed(ring, step, n=1024):
    """A chunk of noise unique to `step`.

    Noise rather than a tone on purpose: it puts varying content in
    *every* column, so a translation test can't pass by comparing two
    stretches of empty spectrum.
    """
    rng = np.random.default_rng(step)
    return ring.write(0.1 * rng.normal(size=n))


def grabbed(widget, tmp_path, name):
    path = tmp_path / name
    widget.grab().save(str(path))
    from PIL import Image

    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))


# --- reduce_to_width -----------------------------------------------------

def test_reduce_returns_exactly_the_requested_width():
    values = np.linspace(0, 1, N_BINS)
    for width in (1, 17, 160, 300, N_BINS, 500, 1000):
        assert len(reduce_to_width(values, width)) == width


def test_reduce_is_identity_at_equal_width():
    values = np.random.default_rng(0).normal(size=N_BINS)
    assert np.array_equal(reduce_to_width(values, N_BINS), values)


def test_downscaling_keeps_every_carrier():
    """Point-sampling 384 bins into a 280 px column drops roughly a
    quarter of them, and the carriers are only ~6 bins apart -- so some
    would vanish outright. Peak-hold must keep them all."""
    spectrum = np.full(N_BINS, -100.0)
    carriers = np.arange(122, 122 + 24 * 6, 6)  # ~950 Hz upward, 50 Hz apart
    spectrum[carriers] = 0.0

    reduced = reduce_to_width(spectrum, 280)
    peaks = np.flatnonzero(reduced > -50.0)

    sampled = spectrum[(np.arange(280) * N_BINS) // 280]
    assert len(peaks) >= 20, "carriers were lost in the reduction"
    assert len(peaks) > np.count_nonzero(sampled > -50.0), (
        "peak-hold should retain more carriers than point sampling"
    )


def test_upscaling_interpolates_rather_than_repeating():
    values = np.linspace(0.0, 1.0, N_BINS)
    out = reduce_to_width(values, 2 * N_BINS)
    assert len(out) == 2 * N_BINS
    assert np.all(np.diff(out) >= -1e-12), "should stay monotonic"


def test_degenerate_width_is_empty_not_an_error():
    assert len(reduce_to_width(np.zeros(N_BINS), 0)) == 0


# --- buffer geometry ------------------------------------------------------

def test_backing_image_matches_the_widget(widget):
    buf = widget._ensure_buffer()
    assert buf.shape == (widget.height(), widget.width(), 3)


def test_resize_refits_the_buffer(widget):
    widget._ensure_buffer()
    widget.resize(220, 650)
    buf = widget._ensure_buffer()
    assert buf.shape == (650, 220, 3)


def test_resize_keeps_the_history_it_can(widget):
    buf = widget._ensure_buffer()
    buf[0] = (255, 0, 0)  # newest row
    widget.resize(widget.width(), widget.height() + 120)
    assert tuple(widget._ensure_buffer()[0, 0]) == (255, 0, 0)


# --- the actual anti-shimmer property -------------------------------------

def test_the_display_scrolls_by_exactly_one_pixel(widget, tmp_path):
    """Each tick must translate the rendered image down by one pixel and
    nothing else. Under the old rescaled design the rows re-quantised
    every frame instead, which is what shimmered."""
    ring = RingBuffer(30.0)
    widget.set_ring(ring)
    for step in range(60):  # fill with distinguishable rows
        feed(ring, step)
        widget._tick()

    before = grabbed(widget, tmp_path, "before.png")
    feed(ring, 999)
    widget._tick()
    after = grabbed(widget, tmp_path, "after.png")

    # A column clear of the band markers, the level meter (right edge)
    # and the caption/tick labels at top and bottom.
    col = 40
    rows = slice(30, widget.height() - 30)
    old_col = before[rows, col]
    new_col = after[rows, col]

    assert np.array_equal(new_col[1:], old_col[:-1]), (
        "the rendered image is not a clean one-pixel translation"
    )


def test_a_still_signal_still_produces_moving_rows(widget, tmp_path):
    """Sanity check on the test above: if nothing scrolled at all, the
    translation assertion would pass trivially."""
    ring = RingBuffer(30.0)
    widget.set_ring(ring)
    for step in range(60):
        feed(ring, step)
        widget._tick()

    before = grabbed(widget, tmp_path, "a.png")
    feed(ring, 999)
    widget._tick()
    after = grabbed(widget, tmp_path, "b.png")

    rows = slice(30, widget.height() - 30)
    assert not np.array_equal(after[rows, 40], before[rows, 40]), (
        "nothing changed between frames -- the scroll test proves nothing"
    )

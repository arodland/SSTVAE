"""Shared visual vocabulary for the SSTVAE talk.

Every scene imports from here, which is what keeps the deck coherent:
one color per *role* (never per slide), one screen position for each
recurring shape, and the same primitives reused from the first phasor
to the last resource grid.
"""

from manim import *
import numpy as np

# ---- colors, by role ------------------------------------------------------
GOLD = "#F2C14E"   # carrier: rotating phasors, carrier planes
BLUE = "#4FA3F7"   # information: m(t), constellation points, latents
GREY = "#8A8F98"   # structure: axes, circles, connections, labels
RED = "#E5533D"    # impairment: the channel, and nothing else
INK = "#ECECEC"    # the signal going out: the scrolling trace

# ---- fixed positions ------------------------------------------------------
MOD_C = np.array([-5.9, 0.5, 0.0])    # the modulator plane m(t)
MOD_R = 0.8
MAIN_C = np.array([-3.3, 0.5, 0.0])   # the main (carrier / product) plane
MAIN_R = 1.3
TRACE_X0, TRACE_X1 = -1.6, 6.9        # the scrolling trace, leading edge at X0
PX_PER_S = 1.7                        # scroll speed: screen units per second
TRACE_WINDOW = (TRACE_X1 - TRACE_X0) / PX_PER_S

# ---- rates ----------------------------------------------------------------
OMEGA = TAU * 1.0      # carrier: one revolution per second on screen
F_MOD = 0.125          # every modulating signal is this slow (8 s period)
T_SYM = 1.0            # symbol period for the digital runs

LABEL_SIZE = 30


def label(text, color=GREY, size=LABEL_SIZE, **kw):
    """A label. Pango markup, so <sup>/<i> work and no LaTeX is needed."""
    return MarkupText(text, font_size=size, color=color, **kw)


def make_plane(center, r, color=GREY):
    """Unit circle with faint axes: the complex plane every phasor lives on."""
    ax = VGroup(
        Line(center + LEFT * r * 1.2, center + RIGHT * r * 1.2),
        Line(center + DOWN * r * 1.2, center + UP * r * 1.2),
    ).set_stroke(color, width=1, opacity=0.4)
    circle = Circle(radius=r, color=color, stroke_width=1.5,
                    stroke_opacity=0.6).move_to(center)
    return VGroup(ax, circle)


def to_point(center, r, z):
    return np.array(center) + np.array([z.real, z.imag, 0.0]) * r


class Phasor(VGroup):
    """Arrow + tip dot on a plane, driven by a time tracker and z(t)."""

    def __init__(self, center, r, t, z, color=GOLD, tip=BLUE,
                 opacity=1.0, width=4, tip_radius=0.07):
        super().__init__()
        self.c, self.r, self.t, self.z = np.array(center), r, t, z

        def arrow():
            tip_pt = self.tip_point()
            if np.linalg.norm(tip_pt - self.c) < 0.03:   # AM null: keep a stub
                tip_pt = self.c + RIGHT * 0.03
            a = Arrow(self.c, tip_pt, buff=0, color=color, stroke_width=width,
                      max_tip_length_to_length_ratio=0.18,
                      max_stroke_width_to_length_ratio=6)
            return a.set_opacity(opacity)

        self.arrow = always_redraw(arrow)
        self.dot = always_redraw(
            lambda: Dot(self.tip_point(), radius=tip_radius, color=tip)
            .set_opacity(opacity))
        self.add(self.arrow, self.dot)

    def tip_point(self):
        return to_point(self.c, self.r, self.z(self.t.get_value()))


class ScrollingTrace(VMobject):
    """Re of a signal, drawn as history flowing right from a leading edge.

    f must be vectorized over numpy arrays of time.  Points older than
    t_start are not drawn, so a trace can start empty and fill in.
    """

    def __init__(self, t, f, y0, amp, x0=TRACE_X0, x1=TRACE_X1,
                 px_per_s=PX_PER_S, t_start=0.0, color=INK, width=3, n=500):
        super().__init__()
        self.t, self.f, self.y0, self.amp = t, f, y0, amp
        self.x0, self.x1, self.pps, self.t_start, self.n = x0, x1, px_per_s, t_start, n
        self.set_stroke(color, width=width)
        self.refresh()
        self.add_updater(lambda m: m.refresh())

    def refresh(self):
        now = self.t.get_value()
        oldest = max(now - (self.x1 - self.x0) / self.pps, self.t_start)
        if now - oldest < 1e-6:
            p = np.array([self.x0, self.y0 + self.amp * float(self.f(np.array([now]))[0]), 0.0])
            self.set_points_as_corners([p, p])
            return
        taus = np.linspace(now, oldest, self.n)
        xs = self.x0 + (now - taus) * self.pps
        ys = self.y0 + self.amp * self.f(taus)
        self.set_points_as_corners(np.column_stack([xs, ys, np.zeros_like(xs)]))


def trace_axis(y0, x0=TRACE_X0, x1=TRACE_X1):
    """The time axis under a trace, with its arrowhead."""
    return Arrow([x0, y0, 0], [x1 + 0.3, y0, 0], buff=0, color=GREY,
                 stroke_width=1.5, max_tip_length_to_length_ratio=0.03,
                 stroke_opacity=0.6)


def projection(phasor, y0, x0=TRACE_X0):
    """Dashed line from a phasor's tip to the trace's leading edge, plus the dot there."""
    def line():
        tip = phasor.tip_point()
        return DashedLine(tip, [x0, tip[1], 0], dash_length=0.12,
                          color=GREY, stroke_width=1.5, stroke_opacity=0.7)

    def dot():
        tip = phasor.tip_point()
        return Dot([x0, tip[1], 0], radius=0.07, color=INK)

    return VGroup(always_redraw(line), always_redraw(dot))


def symbol_ruler(t, y, x0=TRACE_X0, x1=TRACE_X1, px_per_s=PX_PER_S, t_sym=T_SYM,
                 color=GREY):
    """Ticks under a trace at every symbol boundary, scrolling with it."""
    def ticks():
        now = t.get_value()
        g = VGroup(Line([x0, y, 0], [x1, y, 0], color=color, stroke_width=1,
                        stroke_opacity=0.5))
        k = np.floor(now / t_sym)
        while True:
            x = x0 + (now - k * t_sym) * px_per_s
            if x > x1:
                break
            if x >= x0:
                g.add(Line([x, y - 0.1, 0], [x, y + 0.1, 0], color=color,
                           stroke_width=1.5))
            k -= 1
        return g
    return always_redraw(ticks)


class UnitColumn(VGroup):
    """A layer of a neural network: a column of units, fill = activation."""

    def __init__(self, n, x, color=GREY, radius=0.13, spacing=0.3, y=0.0):
        super().__init__()
        self.color = color
        for i in range(n):
            yy = y + (i - (n - 1) / 2) * spacing
            self.add(Circle(radius=radius, color=color, stroke_width=1.5,
                            fill_color=color, fill_opacity=0.0).move_to([x, yy, 0]))

    def set_activation(self, values):
        for c, v in zip(self, values):
            c.set_fill(opacity=float(np.clip(v, 0, 1)))
        return self

    def activate(self, values, lag=0.05, run_time=0.6):
        return AnimationGroup(
            *[c.animate.set_fill(opacity=float(np.clip(v, 0, 1)))
              for c, v in zip(self, values)],
            lag_ratio=lag, run_time=run_time)


def connect(a, b, opacity=0.15):
    """Fully connect two UnitColumns with thin grey lines."""
    g = VGroup()
    for ca in a:
        for cb in b:
            g.add(Line(ca.get_center(), cb.get_center(), color=GREY,
                       stroke_width=0.8, stroke_opacity=opacity))
    return g


def run_clock(scene, t, seconds):
    """Advance the shared time tracker in real time."""
    scene.play(t.animate.increment_value(seconds), run_time=seconds,
               rate_func=linear)


# ---- the modulating signals, vectorized ----------------------------------
def carrier(t):
    return np.exp(1j * OMEGA * t)


def m_am(t):
    return 0.5 + 0.5 * np.sin(TAU * F_MOD * t) + 0j


def m_pm(t):
    return np.exp(1j * 1.2 * np.sin(TAU * F_MOD * t))


def m_fm(t):
    # a slow phasor whose *rate* varies: it speeds up, slows, and reverses
    return np.exp(1j * 3.0 * np.sin(TAU * F_MOD * t))


QAM = np.array([1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j]) / np.sqrt(2)
QAM_SEQ = np.array([0, 1, 3, 2, 0, 2, 1, 3, 3, 0, 2, 1, 1, 0, 3, 2])


def m_qam(t):
    idx = np.floor(np.asarray(t) / T_SYM).astype(int) % len(QAM_SEQ)
    return QAM[QAM_SEQ[idx]]


def product(m):
    return lambda t: m(t) * carrier(t)


def re(z):
    return lambda t: np.real(z(t))

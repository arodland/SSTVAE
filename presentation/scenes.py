"""SSTVAE for undergraduates: one scene per build step.

    manim -pql scenes.py PhasorSine        # quick preview of one scene
    manim -qh  scenes.py                   # render every scene, 1080p

Each scene ends on a hold frame so it drops into a slide as its own clip.
"""

from manim import *
import numpy as np
from style import *


# =============================================================================
# 1. Phasor -> sine
# =============================================================================
class PhasorSine(Scene):
    def construct(self):
        t = ValueTracker(0.0)
        plane = make_plane(MAIN_C, MAIN_R)
        ph = Phasor(MAIN_C, MAIN_R, t, carrier)
        lbl = label("e<sup>jωt</sup>", GOLD).next_to(plane, UP, buff=0.25)

        self.play(Create(plane), FadeIn(lbl))
        self.add(ph)
        run_clock(self, t, 2.0)

        # the Re / Im beat: a phasor is two numbers
        re_lbl = label("Re", size=24).next_to(plane[0][0].get_end(), DOWN + RIGHT, buff=0.05)
        im_lbl = label("Im", size=24).next_to(plane[0][1].get_end(), LEFT, buff=0.1)
        im_line = always_redraw(lambda: DashedLine(
            ph.tip_point(), [MAIN_C[0], ph.tip_point()[1], 0],
            dash_length=0.1, color=GREY, stroke_width=1.5, stroke_opacity=0.7))
        re_line = always_redraw(lambda: DashedLine(
            ph.tip_point(), [ph.tip_point()[0], MAIN_C[1], 0],
            dash_length=0.1, color=GREY, stroke_width=1.5, stroke_opacity=0.7))
        self.play(FadeIn(re_lbl, im_lbl))
        self.add(im_line, re_line)
        run_clock(self, t, 2.0)
        self.remove(im_line, re_line)

        # the projection onto one axis becomes the trace
        axis = trace_axis(MAIN_C[1])
        proj = projection(ph, MAIN_C[1])
        trace = ScrollingTrace(t, im(carrier), MAIN_C[1], MAIN_R,
                               t_start=t.get_value())
        t_lbl = label("t", size=24).next_to(axis.get_end(), DOWN, buff=0.1)
        # "Im" moves to head the column the leading dot travels in, clear of its whole path
        self.play(Create(axis), FadeIn(t_lbl), FadeOut(re_lbl),
                  im_lbl.animate.move_to([TRACE_X0, MAIN_C[1] + MAIN_R + 0.35, 0]))
        self.add(proj, trace)
        run_clock(self, t, TRACE_WINDOW + 2.0)
        self.wait(0.5)


# =============================================================================
# 2. Modulation: the same layout four times
# =============================================================================
class _Modulation(Scene):
    """m(t) on the small plane, times the carrier, on the main plane."""
    m = staticmethod(m_am)
    m_label = "m(t)"
    seconds = 10.0
    ruler = False

    def modulator_extras(self, t):
        """Anything drawn on the modulator plane besides the blue value."""
        return VGroup()

    def construct(self):
        t = ValueTracker(0.0)
        y0 = MAIN_C[1]

        main = make_plane(MAIN_C, MAIN_R)
        small = make_plane(MOD_C, MOD_R)
        times = label("×", size=32).move_to(
            [(MOD_C[0] + MOD_R * 1.15 + MAIN_C[0] - MAIN_R * 1.15) / 2, MAIN_C[1], 0])
        ghost = Phasor(MAIN_C, MAIN_R, t, carrier, opacity=0.3, tip_radius=0.0)
        z = product(self.m)
        prod = Phasor(MAIN_C, MAIN_R, t, z)
        mod = Phasor(MOD_C, MOD_R, t, self.m, color=BLUE, tip=BLUE, width=3,
                     tip_radius=0.08)
        axis = trace_axis(y0)
        proj = projection(prod, y0)
        trace = ScrollingTrace(t, im(z), y0, MAIN_R, t_start=-TRACE_WINDOW)

        l_small = label(self.m_label, BLUE).next_to(small, UP, buff=0.25)
        l_main = label(f"{self.m_label} · e<sup>jωt</sup>", GOLD).next_to(main, UP, buff=0.25)

        self.add(small, times, main, axis, ghost, proj, trace, prod, mod,
                 l_small, l_main, self.modulator_extras(t))
        if self.ruler:
            self.add(symbol_ruler(t, y0 - MAIN_R - 0.4))
        run_clock(self, t, self.seconds)
        self.wait(0.5)


class ModAM(_Modulation):
    m = staticmethod(m_am)
    m_label = "m(t)"

    def modulator_extras(self, t):
        # the envelope, drawn faintly over the trace
        y0 = MAIN_C[1]
        up = ScrollingTrace(t, lambda tt: np.real(m_am(tt)), y0, MAIN_R,
                            t_start=-TRACE_WINDOW, color=BLUE, width=1.5)
        dn = ScrollingTrace(t, lambda tt: -np.real(m_am(tt)), y0, MAIN_R,
                            t_start=-TRACE_WINDOW, color=BLUE, width=1.5)
        return VGroup(up, dn).set_stroke(opacity=0.6)


class ModPM(_Modulation):
    m = staticmethod(m_pm)
    m_label = "e<sup>jφ(t)</sup>"


class ModFM(_Modulation):
    m = staticmethod(m_fm)
    m_label = "e<sup>jθ(t)</sup>"

    def modulator_extras(self, t):
        # a faint arc showing where the slow phasor has been: its *rate* is the point
        def arc():
            th = fm_angle(t.get_value())
            return Arc(radius=MOD_R * 0.75, start_angle=0, angle=th,
                       arc_center=MOD_C, color=BLUE, stroke_width=2,
                       stroke_opacity=0.5)
        return always_redraw(arc)


class ModQAM(_Modulation):
    m = staticmethod(m_qam)
    m_label = "m[k]"
    ruler = True

    def modulator_extras(self, t):
        pts = VGroup(*[Dot(to_point(MOD_C, MOD_R, q), radius=0.08, color=BLUE)
                       .set_opacity(0.35) for q in QAM])
        return pts


# =============================================================================
# 3. Autoencoder
# =============================================================================
LAYERS = [12, 8, 4, 8, 12]
LAYER_X = [-4.0, -2.0, 0.0, 2.0, 4.0]
RNG = np.random.default_rng(7)


def build_network():
    cols = [UnitColumn(n, x, color=(BLUE if n == 4 else GREY))
            for n, x in zip(LAYERS, LAYER_X)]
    links = [connect(a, b) for a, b in zip(cols, cols[1:])]
    return cols, links


def propagate(scene, cols, x, run_time=0.5):
    """Light the input with x and sweep an activation through the layers."""
    acts = [x]
    for n in LAYERS[1:]:
        acts.append(np.clip(RNG.uniform(0.2, 1.0, n), 0, 1))
    acts[-1] = x  # a perfect reconstruction, for now
    scene.play(LaggedStart(*[c.activate(a, run_time=run_time) for c, a in zip(cols, acts)],
                           lag_ratio=0.5))
    return acts


class AutoencoderWhole(Scene):
    def construct(self):
        cols, links = build_network()
        self.play(*[Create(l) for l in links], *[FadeIn(c) for c in cols], run_time=1.5)
        x = RNG.uniform(0.1, 1.0, LAYERS[0])
        propagate(self, cols, x)
        self.wait(0.5)

        enc = Brace(VGroup(cols[0], cols[1]), DOWN, color=GREY)
        dec = Brace(VGroup(cols[3], cols[4]), DOWN, color=GREY)
        # the latent column is narrow and short: brace a box around it, at the others' height
        lat = Brace(Rectangle(width=1.2, height=0.1).move_to(cols[2]), DOWN, color=BLUE)
        lat.align_to(enc, UP)
        labels = VGroup(label("encoder").next_to(enc, DOWN, buff=0.15),
                        label("latent", BLUE).next_to(lat, DOWN, buff=0.15),
                        label("decoder").next_to(dec, DOWN, buff=0.15))
        self.play(GrowFromCenter(enc), GrowFromCenter(lat), GrowFromCenter(dec),
                  FadeIn(labels))
        self.wait(1.0)


class AutoencoderSplit(Scene):
    def construct(self):
        cols, links = build_network()
        x = RNG.uniform(0.1, 1.0, LAYERS[0])
        acts = [x] + [RNG.uniform(0.2, 1.0, n) for n in LAYERS[1:-1]] + [x]
        for c, a in zip(cols, acts):
            c.set_activation(a)

        # a second latent column, exactly over the first, so the split can duplicate it
        lat2 = UnitColumn(4, LAYER_X[2], color=BLUE).set_activation(acts[2])
        enc = VGroup(links[0], links[1], cols[0], cols[1], cols[2])
        dec = VGroup(links[2], links[3], lat2, cols[3], cols[4])
        # re-anchor the decoder's inbound links to its own latent copy
        self.add(enc, dec)
        self.wait(0.5)

        gap = 1.6
        self.play(enc.animate.shift(LEFT * gap), dec.animate.shift(RIGHT * gap),
                  run_time=1.2)
        dots = VGroup(*[DashedLine(a.get_center() + RIGHT * 0.13,
                                   b.get_center() + LEFT * 0.13,
                                   dash_length=0.1, color=BLUE, stroke_width=2)
                        for a, b in zip(cols[2], lat2)])
        self.play(Create(dots))
        self.wait(0.5)

        # a fresh input crosses the gap
        x2 = RNG.uniform(0.1, 1.0, LAYERS[0])
        a2 = [x2] + [RNG.uniform(0.2, 1.0, n) for n in LAYERS[1:-1]] + [x2]
        self.play(LaggedStart(cols[0].activate(a2[0]), cols[1].activate(a2[1]),
                              cols[2].activate(a2[2]), lag_ratio=0.5))
        self.play(ShowPassingFlash(dots.copy().set_stroke(BLUE, 4), time_width=0.6),
                  run_time=0.6)
        self.play(LaggedStart(lat2.activate(a2[2]), cols[3].activate(a2[3]),
                              cols[4].activate(a2[4]), lag_ratio=0.5))
        self.wait(1.0)


class AutoencoderChannel(Scene):
    def construct(self):
        gap = 1.6
        cols, links = build_network()
        x = RNG.uniform(0.1, 1.0, LAYERS[0])
        acts = [x] + [RNG.uniform(0.2, 1.0, n) for n in LAYERS[1:-1]] + [x]
        for c, a in zip(cols, acts):
            c.set_activation(a)
        lat2 = UnitColumn(4, LAYER_X[2], color=BLUE).set_activation(acts[2])
        enc = VGroup(links[0], links[1], cols[0], cols[1], cols[2]).shift(LEFT * gap)
        dec = VGroup(links[2], links[3], lat2, cols[3], cols[4]).shift(RIGHT * gap)
        dots = VGroup(*[DashedLine(a.get_center() + RIGHT * 0.13,
                                   b.get_center() + LEFT * 0.13,
                                   dash_length=0.1, color=BLUE, stroke_width=2)
                        for a, b in zip(cols[2], lat2)])
        self.add(enc, dec, dots)
        self.wait(0.5)

        box = RoundedRectangle(width=1.6, height=2.0, corner_radius=0.15,
                               color=RED, stroke_width=3,
                               fill_color=config.background_color, fill_opacity=1.0)
        box_lbl = label("channel", RED).next_to(box, UP, buff=0.15)
        self.play(FadeIn(box, shift=DOWN * 0.5), FadeIn(box_lbl))
        self.wait(0.5)

        # a value crosses the channel and comes out worse for it
        x2 = RNG.uniform(0.1, 1.0, LAYERS[0])
        a2 = [x2] + [RNG.uniform(0.2, 1.0, n) for n in LAYERS[1:-1]] + [x2]
        self.play(LaggedStart(cols[0].activate(a2[0]), cols[1].activate(a2[1]),
                              cols[2].activate(a2[2]), lag_ratio=0.5))
        self.play(ShowPassingFlash(dots.copy().set_stroke(RED, 4), time_width=0.6),
                  run_time=0.6)
        noisy = np.clip(a2[2] + RNG.normal(0, 0.25, 4), 0, 1)
        noisy[1] = 0.0                                   # one latent erased outright
        out = np.clip(a2[4] + RNG.normal(0, 0.12, LAYERS[4]), 0.05, 1)
        self.play(LaggedStart(lat2.activate(noisy), cols[3].activate(a2[3]),
                              cols[4].activate(out), lag_ratio=0.5))
        cross = Cross(lat2[1], stroke_color=RED, stroke_width=2).scale(0.9)
        self.play(FadeIn(cross), run_time=0.3)
        self.wait(1.5)


# =============================================================================
# 4. Latents on carriers
# =============================================================================
N_LAT = 8
LAT_X = -6.3            # the latent column, where Carriers keeps it
PAIR_X = -1.6           # where LatentPairs composes it, centred
PAIR_PLANE_X = 0.6
PAIR_R = 0.6
LAT_RNG = np.random.default_rng(11)


def latent_values(n):
    return LAT_RNG.uniform(-1, 1, n)


def pair_to_complex(v):
    return complex(v[0], v[1]) * 0.7      # stays inside the unit circle


class LatentPairs(Scene):
    def construct(self):
        v = latent_values(N_LAT)
        col = UnitColumn(N_LAT, PAIR_X, color=BLUE, radius=0.15, spacing=0.48)
        col.set_activation((v + 1) / 2)
        lbl = label("latents", BLUE).next_to(col, UP, buff=0.3)
        self.play(FadeIn(col), FadeIn(lbl))
        self.wait(0.5)

        ys = [2.4, 0.8, -0.8, -2.4]
        for i in range(N_LAT // 2):
            pair = VGroup(col[N_LAT - 1 - 2 * i], col[N_LAT - 2 - 2 * i])  # top-down
            brace = Brace(pair, RIGHT, color=BLUE, buff=0.1)
            c = np.array([PAIR_PLANE_X, ys[i], 0])
            plane = make_plane(c, PAIR_R)
            zval = pair_to_complex([v[N_LAT - 1 - 2 * i], v[N_LAT - 2 - 2 * i]])
            dot = Dot(to_point(c, PAIR_R, zval), radius=0.08, color=BLUE)
            anims = [GrowFromCenter(brace), Create(plane)]
            if i == 0:
                re_l = label("Re", size=22).next_to(plane[0][0], RIGHT, buff=0.05)
                im_l = label("Im", size=22).next_to(plane[0][1], UP, buff=0.05)
                anims += [FadeIn(re_l), FadeIn(im_l)]
            self.play(*anims, run_time=0.6)
            self.play(Transform(pair.copy(), dot), run_time=0.7)
        self.wait(1.0)


class LatentOnCarrier(_Modulation):
    """The first pair's value, held constant, modulating the carrier."""
    seconds = 6.0
    m_label = "m"

    def construct(self):
        v = latent_values(N_LAT)
        z0 = pair_to_complex([v[N_LAT - 1], v[N_LAT - 2]])
        self.m = lambda t: z0 * np.ones_like(np.asarray(t, dtype=float))
        # the plane this pair lived on in LatentPairs slides into the modulator slot
        c = np.array([PAIR_PLANE_X, 2.4, 0])
        plane = make_plane(c, PAIR_R)
        dot = Dot(to_point(c, PAIR_R, z0), radius=0.08, color=BLUE)
        self.add(plane, dot)
        self.wait(0.3)
        self.play(VGroup(plane, dot).animate.move_to(MOD_C).scale(MOD_R / PAIR_R),
                  run_time=1.0)
        self.remove(plane, dot)
        super().construct()


N_CAR = 6
CAR_X = MAIN_C[0]
CAR_R = 0.42
CAR_YS = np.linspace(2.6, -2.6, N_CAR)
CAR_OMEGA = TAU * np.array([0.5, 0.75, 1.0, 1.25, 1.5, 1.75])
FEED_X = -5.6         # the per-carrier values, in a column aligned with the rows
ROW_X0, ROW_X1 = -2.6, 1.3
SUM_X0, SUM_X1 = 2.3, 6.5
T_OFDM = 2.0          # one OFDM symbol on screen
SYM_RNG = np.random.default_rng(3)
# value tables, [symbol, carrier]: what each carrier is told to carry
QAM_TABLE = QAM[SYM_RNG.integers(0, 4, (40, N_CAR))]
_lat = SYM_RNG.uniform(-1, 1, (40, N_CAR, 2))
LAT_TABLE = (_lat[..., 0] + 1j * _lat[..., 1]) * 0.7
LAT_PAIRS = _lat                                   # the same values, two per carrier


def symbol_index(t):
    return np.floor(np.asarray(t, dtype=float) / T_OFDM).astype(int)


def symbol_value(table, k):
    return lambda t: table[symbol_index(t) % len(table), k]


def row_signal(table, k):
    return lambda t: symbol_value(table, k)(t) * np.exp(1j * CAR_OMEGA[k] * t)


def sum_signal(table):
    # /4.2 keeps six full-magnitude QAM carriers, all aligned, inside the frame
    return lambda t: sum(row_signal(table, k)(t) for k in range(N_CAR)) / 4.2


def paired_column(x, ys, gap=0.42, radius=0.11):
    """A latent column laid out two units per carrier row, top row first."""
    col = UnitColumn(0, x, color=BLUE)
    for y in ys:
        for dy in (+gap / 2, -gap / 2):
            col.add(Circle(radius=radius, color=BLUE, stroke_width=1.5,
                           fill_color=BLUE, fill_opacity=0.0).move_to([x, y + dy, 0]))
    return col


class _CarrierBank(Scene):
    """One modulated carrier becomes six, summed into one OFDM symbol.

    Shown twice with the same picture: fed with 4-QAM symbols (OFDM) and
    fed with latent pairs (Carriers). Subclasses supply the feed.
    """
    table = QAM_TABLE

    def feed_source(self, t):
        """What sits on the modulator plane in the opening frame."""
        raise NotImplementedError

    def feed(self, t):
        """The per-carrier values in a column aligned with the rows."""
        raise NotImplementedError

    def construct(self):
        t = ValueTracker(0.0)
        y0 = MAIN_C[1]
        z0 = row_signal(self.table, 0)
        # the opening is the modulation layout: one value times one carrier
        main = make_plane(MAIN_C, MAIN_R)
        small = make_plane(MOD_C, MOD_R)
        times = label("×", size=32).move_to(
            [(MOD_C[0] + MOD_R * 1.15 + MAIN_C[0] - MAIN_R * 1.15) / 2, y0, 0])
        source = self.feed_source(t)
        prod = Phasor(MAIN_C, MAIN_R, t, lambda tt: symbol_value(self.table, 0)(tt) * carrier(tt))
        trace = ScrollingTrace(t, im(lambda tt: symbol_value(self.table, 0)(tt) * carrier(tt)),
                               y0, MAIN_R, t_start=-TRACE_WINDOW)
        axis = trace_axis(y0)
        self.add(small, times, main, axis, trace, prod, source)
        run_clock(self, t, 1.0)
        self.remove(prod, trace)

        # the plane shrinks and multiplies into a stack of carriers,
        # and the one value on the left into one value per carrier
        planes = VGroup(*[make_plane([CAR_X, y, 0], CAR_R) for y in CAR_YS])
        column = self.feed(t)
        self.play(ReplacementTransform(main, planes),
                  ReplacementTransform(VGroup(small, source), column),
                  FadeOut(axis), FadeOut(times), run_time=1.0)
        phasors = [Phasor([CAR_X, y, 0], CAR_R, t, row_signal(self.table, k),
                          width=2.5, tip_radius=0.05) for k, y in enumerate(CAR_YS)]
        rows = [ScrollingTrace(t, im(row_signal(self.table, k)), y, CAR_R, x0=ROW_X0, x1=ROW_X1,
                               px_per_s=0.75, t_start=t.get_value(), width=2)
                for k, y in enumerate(CAR_YS)]
        projs = [projection(p, y, x0=ROW_X0) for p, y in zip(phasors, CAR_YS)]
        car_lbl = label("carriers", GOLD).next_to(planes, UP, buff=0.2)
        omega_lbls = VGroup(*[label(f"ω<sub>{k+1}</sub>", GOLD, size=22)
                              .next_to(p, LEFT, buff=0.1) for k, p in enumerate(planes)])
        self.play(FadeIn(car_lbl), FadeIn(omega_lbls))
        self.add(*phasors, *projs, *rows)
        run_clock(self, t, 2.0)

        # ... and their sum: one OFDM symbol
        plus = label("Σ", size=44).move_to([(ROW_X1 + SUM_X0) / 2, 0, 0])
        sum_axis = trace_axis(0.0, x0=SUM_X0, x1=SUM_X1)
        ofdm_lbl = label("OFDM").move_to([(SUM_X0 + SUM_X1) / 2, CAR_YS[0] + CAR_R + 0.5, 0])
        ruler = symbol_ruler(t, -3.4, x0=SUM_X0, x1=SUM_X1, px_per_s=0.75, t_sym=T_OFDM)
        total = ScrollingTrace(t, im(sum_signal(self.table)), 0.0, 2.4, x0=SUM_X0, x1=SUM_X1,
                               px_per_s=0.75, t_start=t.get_value(), width=2.5)
        self.play(FadeIn(plus), Create(sum_axis), FadeIn(ofdm_lbl))
        self.add(total, ruler)
        # successive symbols: every T_OFDM the feed changes and the dots jump
        run_clock(self, t, 3 * T_OFDM + 1.0)
        self.wait(0.5)


class OFDM(_CarrierBank):
    """From ModQAM: six 4-QAM symbols at once, one per carrier."""
    table = QAM_TABLE

    @staticmethod
    def constellation(t, center, r, k, dot_r):
        faint = VGroup(*[Dot(to_point(center, r, q), radius=dot_r, color=BLUE)
                         .set_opacity(0.35) for q in QAM])
        active = always_redraw(lambda: Dot(
            to_point(center, r, symbol_value(QAM_TABLE, k)(t.get_value())),
            radius=dot_r * 1.25, color=BLUE))
        return VGroup(faint, active)

    def feed_source(self, t):
        return self.constellation(t, MOD_C, MOD_R, 0, 0.08)

    def feed(self, t):
        r = 0.3
        return VGroup(*[VGroup(make_plane([FEED_X, y, 0], r),
                               self.constellation(t, [FEED_X, y, 0], r, k, 0.045))
                        for k, y in enumerate(CAR_YS)])


class Carriers(_CarrierBank):
    """From LatentOnCarrier: the same bank, fed with latent pairs."""
    table = LAT_TABLE

    def feed_source(self, t):
        return always_redraw(lambda: Dot(
            to_point(MOD_C, MOD_R, symbol_value(LAT_TABLE, 0)(t.get_value())),
            radius=0.08, color=BLUE))

    def feed(self, t):
        col = paired_column(FEED_X, CAR_YS)
        braces = VGroup()
        for k in range(N_CAR):
            pair = VGroup(col[2 * k], col[2 * k + 1])
            braces.add(Brace(pair, RIGHT, color=BLUE, buff=0.05).scale(0.8))

        def relight(m):
            idx = int(t.get_value() // T_OFDM) % len(LAT_PAIRS)
            m.set_activation((LAT_PAIRS[idx].reshape(-1) + 1) / 2)
        col.add_updater(relight)
        return VGroup(col, braces)


class ResourceGrid(Scene):
    """Carriers down, time across: where each latent pair ends up."""
    CELL_W, CELL_H = 0.62, float(CAR_YS[0] - CAR_YS[1])
    N_SYM = 10
    GRID_X0 = -1.4

    def construct(self):
        t = ValueTracker(0.0)
        planes = VGroup(*[make_plane([CAR_X, y, 0], CAR_R) for y in CAR_YS])
        phasors = [Phasor([CAR_X, y, 0], CAR_R, t, row_signal(LAT_TABLE, k), width=2.5, tip_radius=0.05)
                   for k, y in enumerate(CAR_YS)]
        omega_lbls = VGroup(*[label(f"ω<sub>{k+1}</sub>", GOLD, size=22)
                              .next_to(p, LEFT, buff=0.1) for k, p in enumerate(planes)])
        car_lbl = label("carriers", GOLD).next_to(planes, UP, buff=0.2)
        self.add(planes, omega_lbls, car_lbl, *phasors)

        # rows of the grid sit exactly on the carrier planes' rows
        ys = np.linspace(CAR_YS[0], CAR_YS[-1], N_CAR)
        cols = []
        for s in range(self.N_SYM):
            x = self.GRID_X0 + s * self.CELL_W
            cells = VGroup()
            for k, y in enumerate(ys):
                mag = abs(LAT_TABLE[s, k]) / (0.7 * np.sqrt(2))
                cells.add(Rectangle(width=self.CELL_W, height=self.CELL_H,
                                    color=GREY, stroke_width=1, stroke_opacity=0.5,
                                    fill_color=BLUE, fill_opacity=0.15 + 0.8 * mag)
                          .move_to([x + self.CELL_W / 2, y, 0]))
            cols.append(cells)

        # first three symbols one at a time, each flown out of the carrier stack
        for s in range(3):
            src = VGroup(*[Dot(p.tip_point(), radius=0.05, color=BLUE) for p in phasors])
            self.play(ReplacementTransform(src, cols[s]), run_time=0.8)
            run_clock(self, t, 0.6)
        # then the rest quickly
        self.play(LaggedStart(*[FadeIn(cols[s], shift=LEFT * 0.2) for s in range(3, self.N_SYM)],
                              lag_ratio=0.3), run_time=2.0)

        grid = VGroup(*cols)
        time_axis = Arrow(grid.get_corner(DL) + DOWN * 0.25, grid.get_corner(DR) + DOWN * 0.25,
                          buff=0, color=GREY, stroke_width=1.5, max_tip_length_to_length_ratio=0.04)
        t_lbl = label("time").next_to(time_axis, RIGHT, buff=0.15)
        self.play(Create(time_axis), FadeIn(t_lbl))
        self.wait(1.5)

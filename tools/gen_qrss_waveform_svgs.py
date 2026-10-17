#!/usr/bin/env python3
"""Draw the two animated waveform figures in docs/qrss/images/.

    tools/gen_qrss_waveform_svgs.py            # write waveform-ce.svg and waveform-l.svg
    tools/gen_qrss_waveform_svgs.py --check    # exit 1 unless the files match

Each figure is a self-contained SVG with SMIL animation and no script, so
GitHub shows it in Markdown like any picture, and its colours follow the
reader's light or dark setting. The numbers it states (phase per latent,
carrier and pilot shares, reference period, L's band edges, symbol rates)
come from `sstvae/qrss/constants.py`. The latents are random draws from a
fixed seed. The spectra are drawn from a short simulation of each
waveform's pulse-shaped baseband signal (no clipping, scrambler or
precoder), so they show the shape, not the measured 99% widths, which are
quoted from the docs.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sstvae.qrss import constants as C  # noqa: E402

OUT = ROOT / "docs" / "qrss" / "images"
BETA = float(C.BETA)                               # rad per unit latent
CE_RATE = 1 / C.T_SYM                              # 32.99 Bd
L_RATE = 1 / C.L_T_SYM                             # 16.49 Bd
L_HI = -float(C.L_PILOT_HZ - C.L_EDGE_HZ)          # data band, Hz from the pilot: -3.10
L_LO = -float(C.L_PILOT_HZ + C.L_EDGE_HZ)          # -22.07
CE_WIDTH_99 = 51.5                                 # measured, docs/qrss/README.md
L_WIDTH_99 = 21.5                                  # measured, docs/qrss/waveform-l.md

N = 16            # symbols per loop
DUR = 12.0        # seconds per loop
rng = np.random.default_rng(7)

def fmt(v): return f"{v:.1f}".rstrip('0').rstrip('.') if abs(v-round(v))>1e-9 else str(int(round(v)))

def smooth_series(vals, n_out):
    """Values at knots k/N, smoothstep between (wraps). Returns samples on [0,1)."""
    out = []
    for i in range(n_out):
        t = i / n_out * N
        k = int(math.floor(t)); f = t - k
        a, b = vals[k % N], vals[(k + 1) % N]
        s = f * f * (3 - 2 * f)
        out.append(a + (b - a) * s)
    return np.array(out)

STYLE = """
  svg.{p} {{ --bg:#f7f8fa; --fg:#1c2230; --muted:#5d6678; --grid:#d5d9e2;
            --data:#2f6fd0; --pilot:#c2410c; --ref:#c2410c; --hi:#e8eefb; }}
  @media (prefers-color-scheme: dark) {{
    svg.{p} {{ --bg:#141821; --fg:#e4e8f0; --muted:#98a1b3; --grid:#2c3343;
              --data:#7aa7f0; --pilot:#f08a4b; --ref:#f08a4b; --hi:#1f2a40; }} }}
  .{p} text {{ font-family: system-ui, -apple-system, 'Segoe UI', sans-serif; fill: var(--fg); }}
  .{p} .t {{ font-size: 17px; font-weight: 600; }}
  .{p} .lab {{ font-size: 11px; letter-spacing: .12em; text-transform: uppercase; fill: var(--muted); }}
  .{p} .note {{ font-size: 12px; fill: var(--muted); }}
  .{p} .axis {{ stroke: var(--grid); stroke-width: 1; fill: none; }}
  .{p} .data {{ fill: var(--data); }}
  .{p} .ref {{ fill: var(--ref); }}
  .{p} .sdata {{ stroke: var(--data); stroke-width: 2; fill: none; }}
  .{p} .spilot {{ stroke: var(--pilot); stroke-width: 2.5; fill: none; }}
  .{p} .hi {{ fill: var(--hi); }}
"""

def head(p, title, sub):
    return [f'<svg class="{p}" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 760 450" role="img" aria-label="{title}">',
            f'<style>{STYLE.format(p=p)}</style>',
            '<rect width="760" height="450" rx="10" fill="var(--bg)"/>',
            f'<text class="t" x="24" y="32">{title}</text>',
            f'<text class="note" x="24" y="52">{sub}</text>']

def discrete_x(xs):
    kt = ";".join(fmt(k / N) if k else "0" for k in range(N))
    vals = ";".join(fmt(x) for x in xs)
    return f'<animate attributeName="x" dur="{DUR}s" repeatCount="indefinite" calcMode="discrete" keyTimes="{kt}" values="{vals}"/>'

def trace_path(series, x0, w, yc, scale):
    n = len(series)
    pts = [f"{x0 + w * i / (n - 1):.1f},{yc - scale * v:.1f}" for i, v in enumerate(series)]
    return "M" + " L".join(pts)

def cursor(x0, w, y0, y1):
    return (f'<line x1="{x0}" x2="{x0}" y1="{y0}" y2="{y1}" stroke="var(--fg)" stroke-width="1" opacity=".5">'
            f'<animate attributeName="x1" dur="{DUR}s" repeatCount="indefinite" values="{x0};{x0+w}"/>'
            f'<animate attributeName="x2" dur="{DUR}s" repeatCount="indefinite" values="{x0};{x0+w}"/></line>')

def ce():
    lat = rng.normal(size=N); lat = lat / np.sqrt(np.mean(lat**2))
    lat = np.clip(lat, -2.2, 2.2)
    ph = BETA * lat                           # rad
    o = head("ce", "Waveform CE: each latent turns the carrier's phase",
             f"One latent per symbol ({CE_RATE:.0f} a second, shown slowed down). The arrow keeps its length: constant envelope.")
    # latents panel
    x0, y0, w, h = 24, 80, 240, 150
    yc = y0 + h / 2; bw = w / N
    o.append(f'<text class="lab" x="{x0}" y="{y0-6}">latents</text>')
    o.append(f'<rect class="hi" x="{x0}" y="{y0}" width="{bw:.1f}" height="{h}" rx="3">{discrete_x([x0 + k*bw for k in range(N)])}</rect>')
    o.append(f'<line class="axis" x1="{x0}" x2="{x0+w}" y1="{yc}" y2="{yc}"/>')
    for k, v in enumerate(lat):
        bh = abs(v) * 30
        y = yc - bh if v > 0 else yc
        o.append(f'<rect class="data" x="{x0 + k*bw + 2:.1f}" y="{y:.1f}" width="{bw-4:.1f}" height="{bh:.1f}" rx="1.5"/>')
    o.append(f'<text class="note" x="{x0}" y="{y0+h+18}">phase = {BETA:g} rad x latent</text>')
    # phasor
    cx, cy, r = 392, 155, 70
    o.append(f'<text class="lab" x="{cx-r}" y="{y0-6}">phase</text>')
    o.append(f'<circle class="axis" cx="{cx}" cy="{cy}" r="{r}"/>')
    o.append(f'<line class="axis" x1="{cx-r-10}" x2="{cx+r+10}" y1="{cy}" y2="{cy}"/>')
    o.append(f'<line class="axis" x1="{cx}" x2="{cx}" y1="{cy-r-10}" y2="{cy+r+10}"/>')
    degs = [-math.degrees(p) for p in ph]
    kt = ";".join([fmt(k / N) if k else "0" for k in range(N)] + ["1"])
    vals = ";".join(f"{d:.1f} {cx} {cy}" for d in degs + [degs[0]])
    ks = ";".join(["0.42 0 0.58 1"] * N)
    o.append(f'<g><line x1="{cx}" y1="{cy}" x2="{cx+r}" y2="{cy}" stroke="var(--data)" stroke-width="3" stroke-linecap="round"/>'
             f'<circle cx="{cx+r}" cy="{cy}" r="5" class="data"/>'
             f'<animateTransform attributeName="transform" type="rotate" dur="{DUR}s" repeatCount="indefinite" calcMode="spline" keyTimes="{kt}" values="{vals}" keySplines="{ks}"/></g>')
    o.append(f'<line x1="{cx}" y1="{cy}" x2="{cx + r*math.exp(-BETA**2 / 2):.1f}" y2="{cy}" stroke="var(--pilot)" stroke-width="3" stroke-linecap="round" opacity=".85"/>')
    o.append(f'<text class="note" x="{cx-r-6}" y="{cy+r+30}">orange: the average, a steady carrier</text>')
    o.append(f'<text class="note" x="{cx-r-6}" y="{cy+r+46}">({100*C.CARRIER_FRAC:.0f}% of the power; CE\'s built-in pilot)</text>')
    # traces
    tx, tw = 506, 230
    s_ph = smooth_series(ph, 240)
    o.append(f'<text class="lab" x="{tx}" y="{y0-6}">phase over time</text>')
    o.append(f'<line class="axis" x1="{tx}" x2="{tx+tw}" y1="125" y2="125"/>')
    o.append(f'<path class="sdata" d="{trace_path(s_ph, tx, tw, 125, 22)}"/>')
    o.append(f'<text class="lab" x="{tx}" y="182">amplitude</text>')
    o.append(f'<line class="axis" x1="{tx}" x2="{tx+tw}" y1="225" y2="225"/>')
    o.append(f'<line class="sdata" x1="{tx}" x2="{tx+tw}" y1="200" y2="200"/>')
    o.append(f'<text class="note" x="{tx}" y="244">never changes</text>')
    o.append(cursor(tx, tw, 85, 228))
    # spectrum
    o += spectrum_ce()
    o.append('</svg>')
    return "\n".join(o)

def spec_axis(o, x0, x1, y, ticks, unit_px, origin):
    o.append(f'<line class="axis" x1="{x0}" x2="{x1}" y1="{y}" y2="{y}"/>')
    for hz in ticks:
        x = origin + hz * unit_px
        o.append(f'<line class="axis" x1="{x:.1f}" x2="{x:.1f}" y1="{y}" y2="{y+4}"/>')
        lab = ("+" if hz > 0 else "") + str(hz) + (" Hz" if hz == ticks[-1] else "")
        o.append(f'<text class="note" x="{x:.1f}" y="{y+17}" text-anchor="middle">{lab}</text>')

def bracket(xa, xb, y, label):
    return [f'<path class="axis" style="stroke:var(--muted)" d="M{xa:.1f},{y+6} L{xa:.1f},{y} L{xb:.1f},{y} L{xb:.1f},{y+6}"/>',
            f'<text class="note" x="{(xa+xb)/2:.1f}" y="{y-6}" text-anchor="middle">{label}</text>']

def rrc(t, a):
    """Root-raised-cosine pulse at t symbols."""
    t = np.asarray(t, dtype=float)
    out = np.empty_like(t)
    for i, x in enumerate(t):
        if abs(x) < 1e-9:
            out[i] = 1 - a + 4 * a / math.pi
        elif abs(abs(x) - 1 / (4 * a)) < 1e-9:
            out[i] = a / math.sqrt(2) * ((1 + 2 / math.pi) * math.sin(math.pi / (4 * a))
                                         + (1 - 2 / math.pi) * math.cos(math.pi / (4 * a)))
        else:
            out[i] = ((math.sin(math.pi * x * (1 - a)) + 4 * a * x * math.cos(math.pi * x * (1 + a)))
                      / (math.pi * x * (1 - (4 * a * x) ** 2)))
    return out


def ce_psd():
    """CE's spectrum without its carrier line: phase = BETA x RRC-shaped
    unit latents, carrier removed, averaged periodograms; +-40 Hz."""
    sps, n_sym, seg = 12, 24000, 2048
    fs = CE_RATE * sps
    g = np.random.default_rng(1)
    up = np.zeros(n_sym * sps)
    up[::sps] = g.normal(size=n_sym)
    phase = np.convolve(up, rrc(np.arange(-8 * sps, 8 * sps + 1) / sps, float(C.ALPHA)), "same")
    phase *= BETA / phase.std()
    x = np.exp(1j * phase)
    x -= x.mean()
    segs = x[: len(x) // seg * seg].reshape(-1, seg) * np.hanning(seg)
    psd = np.fft.fftshift(np.mean(np.abs(np.fft.fft(segs, axis=1)) ** 2, axis=0))
    f = np.fft.fftshift(np.fft.fftfreq(seg, 1 / fs))
    keep = np.abs(f) <= 40
    f, psd = f[keep], psd[keep]
    psd = np.convolve(psd, np.ones(15) / 15, "same")
    return f, psd / psd.max()


def spectrum_ce():
    o = [f'<text class="lab" x="24" y="300">spectrum</text>']
    base, origin, upx = 415, 380, 7.0
    spec_axis(o, 60, 700, base, [-40, -20, 0, 20, 40], upx, origin)
    f, p = ce_psd()
    pts = " L".join(f"{origin + fi*upx:.1f},{base - 40*pi:.1f}" for fi, pi in zip(f, p))
    o.append(f'<path d="M{origin-40*upx},{base} L{pts} L{origin+40*upx},{base} Z" class="data" opacity=".35"/>')
    o.append(f'<line class="spilot" x1="{origin}" x2="{origin}" y1="{base}" y2="{base-90}"/>')
    o.append(f'<text class="note" x="{origin+8}" y="{base-70}">carrier</text>')
    o.append(f'<text class="note" x="{origin+26*upx}" y="{base-26}">latents, both sides</text>')
    o += bracket(origin - CE_WIDTH_99/2*upx, origin + CE_WIDTH_99/2*upx, base - 100, f"99% of the power in {CE_WIDTH_99:g} Hz")
    return o

def l_wave():
    lat = rng.normal(size=2 * N); lat = lat / np.sqrt(np.mean(lat**2))
    lat = np.clip(lat, -2.2, 2.2)
    I = lat[0::2] / math.sqrt(2); Q = lat[1::2] / math.sqrt(2)
    I[0] = Q[0] = 1 / math.sqrt(2)          # symbol 0: a reference (known QPSK)
    o = head("l", "Waveform L: two latents per symbol, beside a pilot",
             f"Each symbol puts one latent on I and the next on Q ({L_RATE:.1f} a second, shown slowed down). Every {C.REF_PERIOD}th is a known reference.")
    x0, y0, w, h = 24, 80, 240, 150
    yc = y0 + h / 2; bw = w / N
    o.append(f'<text class="lab" x="{x0}" y="{y0-6}">latents, in pairs</text>')
    o.append(f'<rect class="hi" x="{x0}" y="{y0}" width="{bw:.1f}" height="{h}" rx="3">{discrete_x([x0 + k*bw for k in range(N)])}</rect>')
    o.append(f'<line class="axis" x1="{x0}" x2="{x0+w}" y1="{yc}" y2="{yc}"/>')
    for k in range(N):
        for j, v in enumerate((I[k] * math.sqrt(2), Q[k] * math.sqrt(2))):
            bh = abs(v) * 30; y = yc - bh if v > 0 else yc
            cls = "ref" if k == 0 else "data"
            op = "1" if j == 0 else ".55"
            o.append(f'<rect class="{cls}" opacity="{op}" x="{x0 + k*bw + 1.5 + j*(bw/2-1):.1f}" y="{y:.1f}" width="{bw/2-2.5:.1f}" height="{bh:.1f}" rx="1"/>')
    o.append(f'<text class="note" x="{x0}" y="{y0+h+18}">dark: I, light: Q, orange: reference</text>')
    # IQ plane
    cx, cy, r = 392, 155, 70; sc = r / 1.6
    o.append(f'<text class="lab" x="{cx-r}" y="{y0-6}">I and Q</text>')
    o.append(f'<line class="axis" x1="{cx-r-10}" x2="{cx+r+10}" y1="{cy}" y2="{cy}"/>')
    o.append(f'<line class="axis" x1="{cx}" x2="{cx}" y1="{cy-r-10}" y2="{cy+r+10}"/>')
    o.append(f'<text class="note" x="{cx+r+4}" y="{cy+16}">I</text><text class="note" x="{cx-4}" y="{cy-r-14}">Q</text>')
    for sx in (-1, 1):
        for sy in (-1, 1):
            px, py = cx + sx*sc/math.sqrt(2), cy - sy*sc/math.sqrt(2)
            o.append(f'<rect x="{px-4:.1f}" y="{py-4:.1f}" width="8" height="8" fill="none" stroke="var(--ref)" stroke-width="1.2"/>')
    # trail: each point fades in when its symbol arrives, stays for the loop
    for k in range(N):
        px, py = cx + I[k]*sc, cy - Q[k]*sc
        cls = "ref" if k == 0 else "data"
        t = k / N
        kt, vals = (f"0;{t:.4f}", "0;.55") if k else ("0", ".55")
        o.append(f'<circle class="{cls}" cx="{px:.1f}" cy="{py:.1f}" r="3"><animate attributeName="opacity" dur="{DUR}s" repeatCount="indefinite" calcMode="discrete" keyTimes="{kt}" values="{vals}"/></circle>')
    xs = [cx + v*sc for v in I] + [cx + I[0]*sc]
    ys = [cy - v*sc for v in Q] + [cy - Q[0]*sc]
    kt = ";".join([fmt(k / N) if k else "0" for k in range(N)] + ["1"])
    ks = ";".join(["0.42 0 0.58 1"] * N)
    def an(attr, vs):
        return f'<animate attributeName="{attr}" dur="{DUR}s" repeatCount="indefinite" calcMode="spline" keyTimes="{kt}" keySplines="{ks}" values="{";".join(f"{v:.1f}" for v in vs)}"/>'
    o.append(f'<line x1="{cx}" y1="{cy}" x2="{xs[0]:.1f}" y2="{ys[0]:.1f}" stroke="var(--data)" stroke-width="3" stroke-linecap="round">{an("x2", xs)}{an("y2", ys)}</line>')
    o.append(f'<circle r="5" class="data" cx="{xs[0]:.1f}" cy="{ys[0]:.1f}">{an("cx", xs)}{an("cy", ys)}</circle>')
    o.append(f'<text class="note" x="{cx-r-6}" y="{cy+r+30}">angle 0 comes from the pilot;</text>')
    o.append(f'<text class="note" x="{cx-r-6}" y="{cy+r+46}">squares: the reference\'s known spots</text>')
    # traces
    tx, tw = 506, 230
    sI, sQ = smooth_series(I, 240), smooth_series(Q, 240)
    o.append(f'<text class="lab" x="{tx}" y="{y0-6}">I and Q over time</text>')
    o.append(f'<line class="axis" x1="{tx}" x2="{tx+tw}" y1="125" y2="125"/>')
    o.append(f'<path class="sdata" d="{trace_path(sI, tx, tw, 125, 20)}"/>')
    o.append(f'<path class="sdata" opacity=".5" d="{trace_path(sQ, tx, tw, 125, 20)}"/>')
    o.append(f'<text class="lab" x="{tx}" y="182">amplitude</text>')
    o.append(f'<line class="axis" x1="{tx}" x2="{tx+tw}" y1="225" y2="225"/>')
    o.append(f'<path class="sdata" d="{trace_path(np.hypot(sI, sQ), tx, tw, 225, 22)}"/>')
    o.append(f'<text class="note" x="{tx}" y="244">goes up and down: needs a linear rig</text>')
    o.append(cursor(tx, tw, 85, 228))
    # spectrum
    o.append(f'<text class="lab" x="24" y="300">spectrum</text>')
    base, upx = 415, 14.0
    origin = 470                      # pilot at 0 Hz
    o.append(f'<line class="axis" x1="60" x2="700" y1="{base}" y2="{base}"/>')
    for hz in (-25, -20, -15, -10, -5, 0, 5):
        x = origin + hz * upx
        o.append(f'<line class="axis" x1="{x:.1f}" x2="{x:.1f}" y1="{base}" y2="{base+4}"/>')
        lab = ("+" if hz > 0 else "") + str(hz) + (" Hz" if hz == 5 else "")
        o.append(f'<text class="note" x="{x:.1f}" y="{base+17}" text-anchor="middle">{lab}</text>')
    lo, hi = L_LO, L_HI
    c, half = (lo + hi) / 2, L_RATE / 2
    roll = float(C.ALPHA) * half                     # flat to (1-a)/2T, edges to (1+a)/2T
    f = np.linspace(lo - 0.5, hi + 0.5, 300)
    d = np.abs(f - c)
    p = np.where(d <= half - roll, 1.0, np.where(d >= half + roll, 0.0, 0.5 * (1 + np.cos(np.pi * (d - (half - roll)) / (2 * roll)))))
    pts = " L".join(f"{origin + fi*upx:.1f},{base - 34*pi:.1f}" for fi, pi in zip(f, p))
    o.append(f'<path d="M{origin + f[0]*upx:.1f},{base} L{pts} Z" class="data" opacity=".35"/>')
    o.append(f'<line class="spilot" x1="{origin}" x2="{origin}" y1="{base}" y2="{base-90}">'
             f'<animate attributeName="opacity" dur="2s" repeatCount="indefinite" values="1;.55;1"/></line>')
    o.append(f'<text class="note" x="{origin+8}" y="{base-70}">pilot: a steady tone,</text>')
    o.append(f'<text class="note" x="{origin+8}" y="{base-55}">{100*float(C.L_PILOT_FRAC):.0f}% of the power</text>')
    o.append(f'<text class="note" x="{origin + c*upx:.1f}" y="{base-44}" text-anchor="middle">latents, all below the pilot</text>')
    o += bracket(origin + L_LO*upx, origin + 0.3*upx, base - 100, f"about {L_WIDTH_99:.0f} Hz in all")
    o.append('</svg>')
    return "\n".join(o)



def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="exit 1 unless the committed files match")
    args = ap.parse_args()
    figs = {"waveform-ce.svg": ce(), "waveform-l.svg": l_wave()}
    stale = []
    for name, text in figs.items():
        path = OUT / name
        if args.check:
            if not path.exists() or path.read_text() != text + "\n":
                stale.append(name)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text + "\n")
            print(f"wrote {path.relative_to(ROOT)}")
    if stale:
        print("stale: " + ", ".join(stale) + "; run tools/gen_qrss_waveform_svgs.py", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

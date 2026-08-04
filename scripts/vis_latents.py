#!/usr/bin/env python3
"""Look at what the codec did to one picture.

Six views of a single image. Four are decoder-only -- no torch, no
training, just the published ONNX artifacts:

  sheet    all 132 latent channels as 40x30 tiles, grouped coarse/mid/fine
  energy   per-channel RMS: which channels the encoder actually uses
  ablate   zero one channel, decode, diff -- what each channel is *for*
  ladder   groups 0, 0+1, 0+1+2 -- the mode A/B/C quality ladder

plus two that run the *whole* path -- modulate, channel, demodulate,
decode -- because the clean `ladder` measures the one axis a longer mode
can only lose on:

  noisy    received pictures on a grid of SNR x mode
  acquire  tune in N seconds late, blind-acquire, and see what you get

    python scripts/vis_latents.py photo.jpg --out /tmp/vis
    python scripts/vis_latents.py photo.jpg --views noisy --channel mpp

Two caveats that belong on every plot this produces, and are printed on
them:

* **Nothing trains these channels to be disentangled.** Ablation shows
  what a channel is *used for*, not a concept it "means". A channel whose
  removal smears the whole frame is not a "global brightness" unit.
* **These are image-domain differences.** Latent-domain distances flatter
  themselves by roughly 2x (docs/latent-optimization.md), so every number
  here is measured after the decoder, against 8-bit subpixels.

Ablation is exactly an erasure -- the latent is zeroed *and* its weight
plane goes to 0 -- so it is the same operation the modem performs on a
latent that never arrived, not an approximation of one.
"""

import argparse
import csv
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from sstvae.checkpoint import PRECISIONS
from sstvae.codec import MODEL_HELP, PRECISION_HELP, load_codec, pad_to_full
from sstvae.config import (
    CHANNELS_PER_GROUP,
    FRAMES_PER_GROUP,
    FRAME_SAMPLES,
    FS,
    LATENT_CHANNELS,
    LATENT_GROUPS,
    LATENT_H,
    LATENT_W,
    MODES,
    SNR_REF_BW_HZ,
)
from sstvae.images import load_image
from sstvae.latents import latents_to_flat

# Group identity, in fixed order -- never cycled, never re-assigned by
# rank. Group 0 is transmitted by every mode, group 2 only by mode C, so
# the colours also order the modes.
GROUP_NAMES = ("group 0 (coarse) - mode A", "group 1 - mode B", "group 2 - mode C")
GROUP_SHORT = ("group 0", "group 1", "group 2")
GROUP_COLORS = ("#2a78d6", "#eb6834", "#1baf7a")

# Diverging pair for signed latents (blue <-> red, neutral gray midpoint)
# and a single-hue sequential ramp for magnitudes. No rainbow anywhere:
# a rainbow invents structure in a field that has none.
DIVERGING = ("#184f95", "#2a78d6", "#9ec5f4", "#f0efec", "#f0a5a4", "#e34948", "#a02020")
SEQUENTIAL = ("#fcfcfb", "#cde2fb", "#9ec5f4", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b")

INK = "#0b0b0b"
INK_MUTED = "#898781"
SURFACE = "#fcfcfb"
GRID = "#e1e0d9"


def _mpl():
    """matplotlib, configured once. Imported lazily so `--views sheet`
    works on a box with only PIL installed."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": "#52514e",
        "axes.titlecolor": INK,
        "text.color": INK,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "grid.color": GRID,
        "axes.grid": True,
        "axes.axisbelow": True,
        "font.size": 9,
    })
    return plt


def _cmap(name, colors):
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list(name, list(colors))


def colorize(arr: np.ndarray, cmap, vmin: float, vmax: float) -> np.ndarray:
    """(H, W) floats -> (H, W, 3) uint8 through a matplotlib colormap."""
    span = max(vmax - vmin, 1e-12)
    norm = np.clip((arr - vmin) / span, 0.0, 1.0)
    return (cmap(norm)[..., :3] * 255).round().astype(np.uint8)


# ---------------------------------------------------------------- sheets

def tile_sheet(tiles, cmap, vmin, vmax, *, title, subtitle, scale, cols=11,
               labels=None, footer=""):
    """A grouped contact sheet of `LATENT_CHANNELS` small tiles.

    Laid out as one block per latent group rather than one long grid,
    because the groups are the thing the modem treats differently: a
    mode-A receiver gets the first block and nothing else.
    """
    th, tw = tiles.shape[1:3]
    tw, th = tw * scale, th * scale
    gap, pad, head, label_h = 6, 28, 26, 14
    rows = CHANNELS_PER_GROUP // cols
    block_h = head + rows * (th + label_h + gap)
    width = pad * 2 + cols * (tw + gap) - gap
    height = pad + 52 + LATENT_GROUPS * block_h + 16 + (24 if footer else 0)

    sheet = Image.new("RGB", (width, height), SURFACE)
    draw = ImageDraw.Draw(sheet)
    draw.text((pad, pad), title, fill=INK)
    draw.text((pad, pad + 15), subtitle, fill=INK_MUTED)

    y = pad + 52
    for g in range(LATENT_GROUPS):
        draw.rectangle([pad, y + 6, pad + 10, y + 16], fill=GROUP_COLORS[g])
        draw.text((pad + 16, y + 5), GROUP_NAMES[g], fill=INK)
        y += head
        for i in range(CHANNELS_PER_GROUP):
            c = g * CHANNELS_PER_GROUP + i
            r, col = divmod(i, cols)
            x0 = pad + col * (tw + gap)
            y0 = y + r * (th + label_h + gap)
            rgb = colorize(tiles[c], cmap, vmin, vmax)
            sheet.paste(Image.fromarray(rgb).resize((tw, th), Image.NEAREST), (x0, y0))
            text = labels[c] if labels else f"ch {c}"
            draw.text((x0 + 1, y0 + th + 1), text, fill=INK_MUTED)
        y += rows * (th + label_h + gap)

    if footer:
        draw.text((pad, height - 20), footer, fill=INK_MUTED)
    return sheet


def view_sheet(z, out, args):
    """Every channel of the latent tensor, on one shared symmetric scale."""
    vmax = float(np.percentile(np.abs(z), 99.5))
    cmap = _cmap("sstvae_div", DIVERGING)
    rms = np.sqrt((z ** 2).mean(axis=(1, 2)))
    sheet = tile_sheet(
        z, cmap, -vmax, vmax, scale=args.scale,
        labels=[f"ch {c}  rms {rms[c]:.2f}" for c in range(LATENT_CHANNELS)],
        title=f"Latent tensor - {Path(args.image).name}",
        subtitle=f"{LATENT_CHANNELS} channels x {LATENT_H}x{LATENT_W}, "
                 f"diverging scale +/-{vmax:.2f} (99.5th pct of |z|); "
                 "the whole tensor is normalized to unit RMS, not each channel",
        footer="blue negative, red positive, gray near zero",
    )
    path = out / "latent_sheet.png"
    sheet.save(path)
    print(f"  wrote {path}  ({sheet.width}x{sheet.height})")


# ---------------------------------------------------------------- energy

def view_energy(z, out, args):
    """Per-channel RMS. Unit-RMS is enforced over the *whole* tensor, so
    the encoder is free to spend that budget unevenly -- and a channel it
    declines to use is airtime spent on nothing."""
    plt = _mpl()
    rms = np.sqrt((z ** 2).mean(axis=(1, 2)))
    mean = float(rms.mean())
    dead = np.flatnonzero(rms < 0.1 * mean)

    fig, ax = plt.subplots(figsize=(13, 4.2))
    for g in range(LATENT_GROUPS):
        lo, hi = g * CHANNELS_PER_GROUP, (g + 1) * CHANNELS_PER_GROUP
        ax.bar(np.arange(lo, hi), rms[lo:hi], width=0.78,
               color=GROUP_COLORS[g], label=GROUP_SHORT[g], zorder=3)
    ax.axhline(mean, color=INK_MUTED, lw=1, ls="--", zorder=4)
    ax.annotate(f"mean {mean:.3f}", (LATENT_CHANNELS - 1, mean),
                xytext=(-4, 4), textcoords="offset points",
                ha="right", color="#52514e")
    for g in range(1, LATENT_GROUPS):
        ax.axvline(g * CHANNELS_PER_GROUP - 0.5, color=GRID, lw=1, zorder=2)

    ax.set_xlim(-1, LATENT_CHANNELS)
    ax.set_xlabel("latent channel")
    ax.set_ylabel("RMS")
    ax.set_title(f"Per-channel latent energy - {Path(args.image).name}", loc="left")
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, loc="upper right", ncols=3)
    note = (f"{len(dead)} channel(s) below 10% of mean"
            + (f": {list(dead)}" if 0 < len(dead) <= 12 else ""))
    fig.text(0.007, 0.005, note + "   |   group budget: "
             + ", ".join(f"{GROUP_SHORT[g]} {rms[g * CHANNELS_PER_GROUP:(g + 1) * CHANNELS_PER_GROUP].mean():.3f}"
                         for g in range(LATENT_GROUPS)),
             color=INK_MUTED, fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    path = out / "latent_energy.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"  wrote {path}")
    print(f"    mean RMS {mean:.4f}, min {rms.min():.4f} (ch {rms.argmin()}), "
          f"max {rms.max():.4f} (ch {rms.argmax()}), {len(dead)} near-dead")


# -------------------------------------------------------------- ablation

def decode_masked(codec, z, mask):
    """Decode with `mask` (per-channel 0/1) applied as an erasure.

    Zeroing the latent *and* its weight plane is what the receiver does
    with a latent that never arrived, so this is the real operation, not
    a stand-in for it.
    """
    w = np.broadcast_to(mask[:, None, None], z.shape).astype(np.float32)
    flat_z = latents_to_flat((z * w)[None])[0]
    flat_w = latents_to_flat(w[None])[0]
    return codec.decode(flat_z, flat_w)


def channel_rms(z: np.ndarray) -> np.ndarray:
    return np.sqrt((z ** 2).mean(axis=(1, 2)))


def ablation_impacts(codec, z, *, progress=False):
    """Per-channel (mean |delta|, pooled delta map, intact decode).

    Split out of `view_ablate` so the before/after comparison runs the
    identical measurement on both latent tensors -- two code paths that
    were "the same computation" would be the easiest way to manufacture
    a difference that isn't there.
    """
    ones = np.ones(LATENT_CHANNELS, dtype=np.float32)
    base = np.asarray(decode_masked(codec, z, ones), dtype=np.float32)

    t0 = time.time()
    impact = np.zeros(LATENT_CHANNELS)
    maps = np.zeros((LATENT_CHANNELS, LATENT_H, LATENT_W))
    for c in range(LATENT_CHANNELS):
        mask = ones.copy()
        mask[c] = 0.0
        delta = np.abs(np.asarray(decode_masked(codec, z, mask), np.float32) - base)
        impact[c] = delta.mean()
        # Pool to the latent grid so the sheet's tiles line up with the
        # tensor the sheet view shows -- 16x down is exactly the codec's
        # own stride, so a tile is one latent's worth of picture.
        d = delta.mean(axis=2)
        maps[c] = d.reshape(LATENT_H, d.shape[0] // LATENT_H,
                            LATENT_W, d.shape[1] // LATENT_W).mean(axis=(1, 3))
        if progress and (c + 1) % 20 == 0:
            print(f"    ablated {c + 1}/{LATENT_CHANNELS}", flush=True)
    print(f"    {LATENT_CHANNELS} decodes in {time.time() - t0:.1f} s")
    return impact, maps, base


def view_ablate(codec, z, out, args, source):
    """Zero each channel in turn and measure what leaves the picture."""
    plt = _mpl()
    ones = np.ones(LATENT_CHANNELS, dtype=np.float32)
    impact, maps, base = ablation_impacts(codec, z, progress=args.progress)

    order = np.argsort(-impact)
    with (out / "ablation_impact.csv").open("w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["channel", "group", "mean_abs_delta_8bit", "peak_delta_8bit", "rank"])
        for rank, c in enumerate(order):
            wr.writerow([c, c // CHANNELS_PER_GROUP, f"{impact[c]:.4f}",
                         f"{maps[c].max():.4f}", rank])

    # Ranked magnitudes -- a sequential job, but coloured by group so the
    # question "do the coarse channels dominate?" is answerable by eye.
    fig, ax = plt.subplots(figsize=(13, 4.2))
    for g in range(LATENT_GROUPS):
        sel = [i for i, c in enumerate(order) if c // CHANNELS_PER_GROUP == g]
        ax.bar(sel, impact[order][sel], width=0.9, color=GROUP_COLORS[g],
               label=GROUP_SHORT[g], zorder=3)
    # Selective direct labels: only the channels that are visibly clear of
    # the pack. Labelling eight of them stacks four legends on top of each
    # other in the tail, and the CSV carries the full ranking anyway.
    for rank in range(min(6, LATENT_CHANNELS)):
        v = impact[order[rank]]
        if rank and v > 0.93 * impact[order[rank - 1]]:
            break
        ax.annotate(f"ch {order[rank]}", (rank, v),
                    xytext=(2, 4), textcoords="offset points",
                    ha="left", fontsize=8, color="#52514e")
    ax.set_xlim(-1, LATENT_CHANNELS)
    ax.set_xlabel("channels, ranked by impact")
    ax.set_ylabel("mean |delta| (8-bit levels)")
    ax.set_title(f"What each latent channel is worth - {Path(args.image).name}",
                 loc="left")
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, loc="upper right", ncols=3)
    fig.text(0.007, 0.005,
             "erasing one channel (latent and weight plane both zeroed), decoding, "
             "and differencing against the intact decode. "
             "Nothing trains these channels to be disentangled -- this is what a "
             "channel is used for, not what it means.",
             color=INK_MUTED, fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out / "ablation_impact.png", dpi=140)
    plt.close(fig)
    print(f"  wrote {out / 'ablation_impact.png'} and .csv")

    vmax = float(np.percentile(maps, 99.8))
    sheet = tile_sheet(
        maps, _cmap("sstvae_seq", SEQUENTIAL), 0.0, vmax, scale=args.scale,
        labels=[f"ch {c}  {impact[c]:.2f}" for c in range(LATENT_CHANNELS)],
        title=f"Ablation atlas - {Path(args.image).name}",
        subtitle="where the picture changes when each channel is erased, pooled to "
                 f"the latent grid; shared scale 0..{vmax:.2f} of 255",
        footer="tile label is the channel's mean |delta| over the whole frame",
    )
    sheet.save(out / "ablation_atlas.png")
    print(f"  wrote {out / 'ablation_atlas.png'}  ({sheet.width}x{sheet.height})")

    # The top offenders, at full resolution, where you can actually see it.
    n = min(args.top, LATENT_CHANNELS)
    strip_w, strip_h = base.shape[1] // 2, base.shape[0] // 2
    head, pad, cap = 46, 18, 16
    canvas = Image.new("RGB", (pad * 2 + 3 * strip_w + 2 * 8,
                               head + n * (strip_h + cap + 10)), SURFACE)
    draw = ImageDraw.Draw(canvas)
    draw.text((pad, pad), f"Top {n} channels by impact - {Path(args.image).name}",
              fill=INK)
    draw.text((pad, pad + 15),
              "intact decode  |  that channel erased  |  |difference|, x8",
              fill=INK_MUTED)
    base_img = Image.fromarray(base.round().clip(0, 255).astype(np.uint8))
    base_small = base_img.resize((strip_w, strip_h), Image.LANCZOS)
    for row in range(n):
        c = int(order[row])
        mask = ones.copy()
        mask[c] = 0.0
        ab = np.asarray(decode_masked(codec, z, mask), np.float32)
        diff = np.clip(np.abs(ab - base) * 8, 0, 255).astype(np.uint8)
        y = head + row * (strip_h + cap + 10)
        canvas.paste(base_small, (pad, y))
        canvas.paste(Image.fromarray(ab.round().clip(0, 255).astype(np.uint8))
                     .resize((strip_w, strip_h), Image.LANCZOS), (pad + strip_w + 8, y))
        canvas.paste(Image.fromarray(diff).resize((strip_w, strip_h), Image.LANCZOS),
                     (pad + 2 * (strip_w + 8), y))
        draw.text((pad, y + strip_h + 2),
                  f"ch {c} ({GROUP_SHORT[c // CHANNELS_PER_GROUP]}), "
                  f"mean |delta| {impact[c]:.2f}, peak {maps[c].max():.2f} of 255",
                  fill="#52514e")
    canvas.save(out / "ablation_top.png")
    print(f"  wrote {out / 'ablation_top.png'}")

    top = ", ".join(f"ch {int(c)} ({impact[c]:.2f})" for c in order[:5])
    print(f"    hardest-working: {top}")
    print(f"    softest: " + ", ".join(f"ch {int(c)} ({impact[c]:.3f})"
                                       for c in order[-3:]))
    return base


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return float("inf") if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)


# ---------------------------------------------------------------- ladder

def view_ladder(codec, z, out, args, source):
    """Groups 0, 0+1, 0+1+2 -- what each mode's extra airtime buys.

    The modes are nested, so this is the *whole* difference between them
    at the codec level: a mode-A receiver has the same decoder and simply
    holds weight 0 over the groups that were never sent.
    """
    # `load_image` already hands back (3, H, W) in [0,1], fitted to 640x480.
    ref = (source.transpose(1, 2, 0) * 255).round().clip(0, 255)
    modes = sorted(MODES)
    imgs, notes = [], []
    for name in modes:
        mask = np.zeros(LATENT_CHANNELS, dtype=np.float32)
        mask[: MODES[name].groups * CHANNELS_PER_GROUP] = 1.0
        img = decode_masked(codec, z, mask)
        imgs.append(img)
        spec = MODES[name]
        notes.append((f"mode {name} - {spec.groups} group(s), "
                      f"{spec.n_frames} frames",
                      f"PSNR {psnr(np.asarray(img), ref):.2f} dB vs source"))

    w, h = imgs[0].size
    sw, sh = w // 2, h // 2
    pad, head, cap = 18, 46, 30
    cols = len(imgs) + 1
    canvas = Image.new("RGB", (pad * 2 + cols * sw + (cols - 1) * 8,
                               head + sh + cap), SURFACE)
    draw = ImageDraw.Draw(canvas)
    draw.text((pad, pad), f"Progressive group ladder - {Path(args.image).name}",
              fill=INK)
    draw.text((pad, pad + 15),
              "the same decoder each time; later groups simply carry weight 0",
              fill=INK_MUTED)
    src_small = Image.fromarray(ref.astype(np.uint8)).resize((sw, sh), Image.LANCZOS)
    canvas.paste(src_small, (pad, head))
    draw.text((pad, head + sh + 4), "source (fitted 640x480)", fill="#52514e")
    for i, (img, (line1, line2)) in enumerate(zip(imgs, notes)):
        x = pad + (i + 1) * (sw + 8)
        canvas.paste(img.resize((sw, sh), Image.LANCZOS), (x, head))
        draw.rectangle([x, head + sh + 5, x + 10, head + sh + 15],
                       fill=GROUP_COLORS[min(i, LATENT_GROUPS - 1)])
        draw.text((x + 16, head + sh + 4), line1, fill="#52514e")
        draw.text((x, head + sh + 17), line2, fill="#52514e")
    canvas.save(out / "mode_ladder.png")
    print(f"  wrote {out / 'mode_ladder.png'}")
    for name, (_, line2) in zip(modes, notes):
        print(f"    mode {name}: {line2}")


# -------------------------------------------- before/after optimization

# Two series (encoder vs optimized), so two categorical hues, fixed.
BEFORE_COLOR, AFTER_COLOR = "#2a78d6", "#eb6834"


def _group_rules(ax, labels=True):
    """Latent-group boundaries as recessive rules, so the group identity
    is still readable when the hues are carrying before/after instead."""
    for g in range(1, LATENT_GROUPS):
        ax.axvline(g * CHANNELS_PER_GROUP - 0.5, color=GRID, lw=1, zorder=2)
    if labels:
        for g in range(LATENT_GROUPS):
            ax.annotate(GROUP_SHORT[g],
                        ((g + 0.5) * CHANNELS_PER_GROUP, 1.0),
                        xycoords=("data", "axes fraction"),
                        xytext=(0, -12), textcoords="offset points",
                        ha="center", color=INK_MUTED, fontsize=8)


def _paired_panel(plt, before, after, *, title, subtitle, ylabel, dlabel,
                  path, footer=""):
    """Two stacked panels on the same measure: the pair, then the change.

    Stacked rather than a twin y-axis: two scales on one frame is the
    single most misread chart there is, and here the second panel is a
    *derived* quantity, which is exactly the case that tempts it.
    """
    fig, (ax, ad) = plt.subplots(2, 1, figsize=(13, 6.4), sharex=True,
                                 gridspec_kw={"height_ratios": [2, 1]})
    x = np.arange(LATENT_CHANNELS)
    ax.plot(x, before, drawstyle="steps-mid", lw=2, color=BEFORE_COLOR,
            label="encoder", zorder=3)
    ax.plot(x, after, drawstyle="steps-mid", lw=2, color=AFTER_COLOR,
            label="optimized", zorder=4)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left")
    # On the title line, not inside the frame: the in-frame upper right is
    # where the group-2 label lives, and a legend over a label is worse
    # than either alone.
    ax.legend(frameon=False, ncols=2, loc="lower right",
              bbox_to_anchor=(1.0, 1.01))
    _group_rules(ax)

    delta = after - before
    ad.bar(x, delta, width=0.8, zorder=3,
           color=[AFTER_COLOR if d >= 0 else BEFORE_COLOR for d in delta])
    ad.axhline(0, color="#c3c2b7", lw=1, zorder=4)
    ad.set_ylabel(dlabel)
    ad.set_xlabel("latent channel")
    ad.set_xlim(-1, LATENT_CHANNELS)
    ad.grid(axis="x", visible=False)
    _group_rules(ad, labels=False)
    ax.grid(axis="x", visible=False)

    top = np.argsort(-np.abs(delta))[:5]
    for c in top:
        ad.annotate(f"ch {c}", (c, delta[c]),
                    xytext=(0, 4 if delta[c] >= 0 else -11),
                    textcoords="offset points", ha="center", fontsize=8,
                    color="#52514e")
    fig.text(0.007, 0.005, subtitle + ("   |   " + footer if footer else ""),
             color=INK_MUTED, fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"  wrote {path}")
    return delta


def ablation_impacts_rx(codec, flat, source, args, spec):
    """Per-channel impact measured on the *received* picture, in dB of PSNR.

    Two changes from `ablation_impacts`, and both matter:

    * The latents are the ones that came back off the air -- modulated,
      faded, demodulated, with the modem's own confidence weights and
      whatever it already erased. So an ablation is one more lost
      channel on top of a real channel's damage, which is the situation
      the question is actually about.
    * The measure is **PSNR against the source**, not |delta| against the
      unablated decode. `ablation_impacts` cannot compare two latent
      vectors whose decodes differ in sharpness -- a smoother picture
      has smaller deltas whatever its quality -- and that confound is
      exactly what made the clean comparison unreadable. Both sides here
      are scored against the same fixed source, so "losing this channel
      costs 0.4 dB" means the same thing on both.

    One demodulation per (latent set, seed); the 132 erasures then happen
    on the received vector, which is why this costs decodes and not
    channel runs.
    """
    from sstvae import hfchannel
    from sstvae.latents import flat_to_latents
    from sstvae.modem import Modem, SyncError

    modem = Modem()
    fading = None if args.channel == "awgn" else args.channel
    snr = float(args.snr.split(",")[0])
    ref = (source.transpose(1, 2, 0) * 255).round().clip(0, 255)

    wave = modem.modulate(flat[: spec.n_latents], spec, callsign=args.callsign)
    impact = np.zeros(LATENT_CHANNELS)
    bases, used = [], 0
    t0 = time.time()
    for rep in range(args.channel_reps):
        y = hfchannel.apply_channel(wave, snr_db=snr, fading_preset=fading,
                                    seed=args.seed + rep)
        try:
            d = modem.demodulate(y)
        except SyncError:
            print(f"    rep {rep}: no sync, skipped")
            continue
        used += 1
        zr = flat_to_latents(pad_to_full(d.latents)[None].astype(np.float32))[0]
        wr = flat_to_latents(pad_to_full(d.weights)[None].astype(np.float32))[0]
        base = psnr(np.asarray(codec.decode(latents_to_flat(zr[None])[0],
                                            latents_to_flat(wr[None])[0])), ref)
        bases.append(base)
        for c in range(LATENT_CHANNELS):
            zc, wc = zr.copy(), wr.copy()
            zc[c] = 0.0
            wc[c] = 0.0
            q = psnr(np.asarray(codec.decode(latents_to_flat(zc[None])[0],
                                             latents_to_flat(wc[None])[0])), ref)
            impact[c] += base - q
            if args.progress and (c + 1) % 40 == 0:
                print(f"    rep {rep}: ablated {c + 1}/{LATENT_CHANNELS}", flush=True)
    if not used:
        raise SystemExit("no channel realization acquired sync; raise --snr")
    print(f"    {used} realization(s), {used * LATENT_CHANNELS} decodes "
          f"in {time.time() - t0:.1f} s")
    return impact / used, float(np.mean(bases))


def view_optim(codec, flat, out, args, source):
    """Every study, before and after a latent-optimization pass.

    The optimizer moves the latents but changes nothing on air and
    nothing at the receiver, so "what did it actually do to the code"
    has no answer anywhere else -- the end-to-end figures say only that
    it helped.
    """
    from sstvae.latent_optim import (
        FADING_PROFILES, OBJECTIVE_SNR_DB, optimize)
    from sstvae.latents import flat_to_latents

    plt = _mpl()
    spec = MODES[args.optimize_mode]
    ref = (source.transpose(1, 2, 0) * 255).round().clip(0, 255)

    # The objective's own channel. `fading` samples the measured
    # (confidence, error) joint instead of assuming every latent arrives
    # at full confidence -- which also means its `objective_snr_db` is
    # not on the same scale as the flat one's (it carries 5-7 dB more
    # noise at the same nominal number), so the two are not comparable
    # by that setting and only by what comes back off the air.
    kw = {}
    if args.objective == "fading":
        kw["fading_profile"] = FADING_PROFILES["measured"]
    if args.objective_snr is not None:
        kw["objective_snr_db"] = args.objective_snr
    snr_note = (f"{args.objective_snr:g} dB" if args.objective_snr is not None
                else f"{OBJECTIVE_SNR_DB:g} dB (default)")
    print(f"  optimizing for mode {spec.name}, {args.objective} objective at "
          f"{snr_note} (up to {args.optimize:g} s, plateau ~90 s)...")
    t0 = time.time()
    r = optimize(flat, source, spec.name, model=args.model,
                 time_budget_s=args.optimize, max_steps=args.optimize_steps, **kw)
    print(f"    {r.steps} steps in {r.seconds:.1f} s ({r.stop_reason}), "
          f"objective gain {r.gain_db:.2f} dB "
          f"-- an objective value, ~3x what the receiver sees, and not "
          f"comparable across objectives")

    # The optimizer returns mode-length latents and renormalizes over the
    # transmitted groups only, so a mode A/B run leaves the untransmitted
    # tail at the encoder's values. Padding with the encoder's own tail
    # keeps the two tensors comparable channel by channel.
    after_flat = flat.copy()
    after_flat[: spec.n_latents] = r.latents
    # Keep the latents. Two minutes of optimization that can only be
    # re-examined by spending another two minutes is the kind of result
    # that never gets a second question asked of it.
    np.savez(out / f"optim_latents_{args.objective}.npz", before=flat, after=after_flat,
             mode=spec.name, steps=r.steps, seconds=r.seconds,
             objective_gain_db=r.gain_db)
    print(f"  wrote {out / f'optim_latents_{args.objective}.npz'}")

    z0 = flat_to_latents(flat[None].astype(np.float32))[0]
    z1 = flat_to_latents(after_flat[None].astype(np.float32))[0]
    if spec.groups < LATENT_GROUPS:
        print(f"    note: mode {spec.name} transmits "
              f"{spec.groups * CHANNELS_PER_GROUP} of {LATENT_CHANNELS} "
              "channels; the rest are the encoder's, untouched")

    # --- energy -------------------------------------------------------
    rms0, rms1 = channel_rms(z0), channel_rms(z1)
    d_rms = _paired_panel(
        plt, rms0, rms1,
        title=f"Per-channel energy, encoder vs optimized - {Path(args.image).name}",
        subtitle=f"mode {spec.name} objective, {r.steps} steps, "
                 f"{r.seconds:.0f} s; unit RMS is enforced over the transmitted "
                 "groups as a whole, so channels trade against each other",
        ylabel="RMS", dlabel="change", path=out / f"optim_energy_{args.objective}.png",
        footer=f"overall RMS {np.sqrt((z0 ** 2).mean()):.4f} -> "
               f"{np.sqrt((z1 ** 2).mean()):.4f}")

    # --- ablation -----------------------------------------------------
    if args.through_channel:
        chan = "AWGN" if args.channel == "awgn" else f"{args.channel} + AWGN"
        snr = float(args.snr.split(",")[0])
        print("  ablating the encoder's latents, through the channel...")
        imp0, rx0 = ablation_impacts_rx(codec, flat, source, args, spec)
        print("  ablating the optimized latents, through the channel...")
        imp1, rx1 = ablation_impacts_rx(codec, after_flat, source, args, spec)
        print(f"    received PSNR {rx0:.2f} -> {rx1:.2f} dB ({rx1 - rx0:+.2f}) "
              f"-- this one is the real gain, not the objective's")
        d_imp = _paired_panel(
            plt, imp0, imp1,
            title=f"Per-channel erasure cost through the channel, encoder vs "
                  f"optimized - {Path(args.image).name}",
            subtitle=f"{chan} at {snr:g} dB, {args.channel_reps} realization(s); "
                     "dB of received PSNR lost when that channel goes on top of "
                     "the channel's own damage",
            ylabel="PSNR lost (dB)", dlabel="change",
            path=out / f"optim_impact_rx_{args.objective}.png",
            footer=f"received PSNR {rx0:.2f} -> {rx1:.2f} dB. Scored against the "
                   "source, so a smoother picture earns nothing here")
        maps0 = maps1 = None
        base0 = np.asarray(decode_masked(codec, z0, np.ones(LATENT_CHANNELS, np.float32)),
                           np.float32)
        base1 = np.asarray(decode_masked(codec, z1, np.ones(LATENT_CHANNELS, np.float32)),
                           np.float32)
    else:
        print("  ablating the encoder's latents...")
        imp0, maps0, base0 = ablation_impacts(codec, z0, progress=args.progress)
        print("  ablating the optimized latents...")
        imp1, maps1, base1 = ablation_impacts(codec, z1, progress=args.progress)
        d_imp = _paired_panel(
            plt, imp0, imp1,
            title=f"Per-channel ablation impact, encoder vs optimized - "
                  f"{Path(args.image).name}",
            subtitle="mean |delta| in 8-bit levels when that channel is erased; "
                     "each side is measured against its own intact decode",
            ylabel="mean |delta| (8-bit)", dlabel="change",
            path=out / f"optim_impact_{args.objective}.png",
            footer="scale-dependent: a smoother decode has smaller deltas at "
                   "equal quality, so use --through-channel to compare "
                   "robustness")

    # --- what moved, spatially ---------------------------------------
    dz = z1 - z0
    vmax = float(np.percentile(np.abs(dz), 99.5)) or 1e-6
    sheet = tile_sheet(
        dz, _cmap("sstvae_div", DIVERGING), -vmax, vmax, scale=args.scale,
        labels=[f"ch {c}  d{d_rms[c]:+.2f}" for c in range(LATENT_CHANNELS)],
        title=f"What the optimizer moved - {Path(args.image).name}",
        subtitle=f"optimized minus encoder, per channel; diverging scale "
                 f"+/-{vmax:.3f} (99.5th pct of |delta|)",
        footer="tile label is the change in that channel's RMS",
    )
    sheet.save(out / f"optim_latent_delta_{args.objective}.png")
    print(f"  wrote {out / 'optim_latent_delta.png'}")

    # --- and what it did to the picture -------------------------------
    q0, q1 = psnr(base0, ref), psnr(base1, ref)
    sw, sh = 320, 240
    pad, head, cap, gap = 18, 52, 30, 8
    canvas = Image.new("RGB", (pad * 2 + 4 * sw + 3 * gap, head + sh + cap), SURFACE)
    draw = ImageDraw.Draw(canvas)
    draw.text((pad, pad), f"Clean decode, encoder vs optimized - {Path(args.image).name}",
              fill=INK)
    draw.text((pad, pad + 15),
              "the optimizer's objective is the decode *through the channel*, so a "
              "clean decode is not what it was maximizing", fill=INK_MUTED)
    diff = np.clip(np.abs(base1 - base0) * 8, 0, 255).astype(np.uint8)
    panels = [
        (Image.fromarray(ref.astype(np.uint8)), ["source (fitted 640x480)"]),
        (Image.fromarray(base0.round().clip(0, 255).astype(np.uint8)),
         [f"encoder latents", f"PSNR {q0:.2f} dB clean"]),
        (Image.fromarray(base1.round().clip(0, 255).astype(np.uint8)),
         [f"optimized latents", f"PSNR {q1:.2f} dB clean ({q1 - q0:+.2f})"]),
        (Image.fromarray(diff), ["|difference|, x8",
                                 f"mean {np.abs(base1 - base0).mean():.2f} of 255"]),
    ]
    for i, (img, lines) in enumerate(panels):
        x = pad + i * (sw + gap)
        canvas.paste(img.resize((sw, sh), Image.LANCZOS), (x, head))
        for j, line in enumerate(lines):
            draw.text((x, head + sh + 2 + j * 12), line, fill="#52514e")
    canvas.save(out / f"optim_pictures_{args.objective}.png")
    print(f"  wrote {out / 'optim_pictures.png'}")

    with (out / f"optim_channels_{args.objective}.csv").open("w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["channel", "group", "rms_before", "rms_after", "rms_delta",
                     "impact_before", "impact_after", "impact_delta"])
        for c in range(LATENT_CHANNELS):
            wr.writerow([c, c // CHANNELS_PER_GROUP, f"{rms0[c]:.4f}",
                         f"{rms1[c]:.4f}", f"{d_rms[c]:+.4f}",
                         f"{imp0[c]:.4f}", f"{imp1[c]:.4f}", f"{d_imp[c]:+.4f}"])
    print(f"  wrote {out / 'optim_channels.csv'}")

    def _top(d, label):
        idx = np.argsort(-np.abs(d))[:5]
        print(f"    largest {label}: "
              + ", ".join(f"ch {c} {d[c]:+.3f}" for c in idx))

    print(f"    RMS spread {rms0.max() - rms0.min():.3f} -> "
          f"{rms1.max() - rms1.min():.3f}  "
          f"(peak ch {rms0.argmax()} {rms0.max():.3f} -> ch {rms1.argmax()} "
          f"{rms1.max():.3f})")
    print(f"    impact spread {imp0.max() - imp0.min():.3f} -> "
          f"{imp1.max() - imp1.min():.3f}")
    _top(d_rms, "energy moves")
    _top(d_imp, "impact moves")
    # Does the optimizer take energy out of the channels the picture
    # leans on hardest? That is the shape "spread the risk" would have.
    if np.std(imp0) > 0 and np.std(d_rms) > 0:
        cc = float(np.corrcoef(imp0, d_rms)[0, 1])
        print(f"    corr(encoder impact, energy change) = {cc:+.3f}")


# ------------------------------------------------------- the on-air views

def _grid(rows, cols, cell, *, title, subtitle, footer, row_label_w=96):
    """A labelled table of half-size pictures. `cell(r, c)` returns
    (PIL image or None, list of caption lines)."""
    sw, sh = 320, 240
    pad, head, cap, gap = 18, 52, 30, 8
    hdr = 20
    width = pad * 2 + row_label_w + len(cols) * (sw + gap) - gap
    height = head + hdr + len(rows) * (sh + cap + gap) + 20
    canvas = Image.new("RGB", (width, height), SURFACE)
    draw = ImageDraw.Draw(canvas)
    draw.text((pad, pad), title, fill=INK)
    draw.text((pad, pad + 15), subtitle, fill=INK_MUTED)
    for c, name in enumerate(cols):
        draw.text((pad + row_label_w + c * (sw + gap), head), name, fill=INK)
    for r, rname in enumerate(rows):
        y = head + hdr + r * (sh + cap + gap)
        draw.text((pad, y + sh // 2 - 4), rname, fill=INK)
        for c in range(len(cols)):
            x = pad + row_label_w + c * (sw + gap)
            img, lines = cell(r, c)
            if img is None:
                draw.rectangle([x, y, x + sw, y + sh], fill="#f0efec")
            else:
                canvas.paste(img.resize((sw, sh), Image.LANCZOS), (x, y))
            for i, line in enumerate(lines):
                draw.text((x, y + sh + 2 + i * 12), line, fill="#52514e")
    draw.text((pad, height - 16), footer, fill=INK_MUTED)
    return canvas


def _reconstruct(codec, r):
    return codec.decode(pad_to_full(r.latents), pad_to_full(r.weights))


def view_noisy(codec, flat, out, args, source):
    """Received pictures on an SNR x mode grid, through the real path.

    Every mode in a row sees the **same channel seed**, so the columns
    differ by mode and not by which noise happened to be drawn. The
    waveforms have different lengths, so the realizations are the same
    process rather than literally the same samples -- which is the
    closest honest comparison available.
    """
    from sstvae import hfchannel
    from sstvae.modem import Modem, SyncError

    ref = (source.transpose(1, 2, 0) * 255).round().clip(0, 255)
    modem = Modem()
    fading = None if args.channel == "awgn" else args.channel
    modes = [MODES[m] for m in args.modes]
    snrs = [float(s) for s in args.snr.split(",")]

    # One waveform per mode, reused at every SNR (the modes are nested
    # prefixes of one latent vector, so this is also one encode).
    waves = {spec.name: modem.modulate(flat[: spec.n_latents], spec,
                                       callsign=args.callsign)
             for spec in modes}
    results = {}
    for r, snr in enumerate(snrs):
        for spec in modes:
            y = hfchannel.apply_channel(waves[spec.name], snr_db=snr,
                                        fading_preset=fading, seed=args.seed + r)
            try:
                d = modem.demodulate(y)
            except SyncError as e:
                results[(r, spec.name)] = (None, [f"no sync ({e})"])
                print(f"    {spec.name} @ {snr:g} dB: no sync")
                continue
            img = _reconstruct(codec, d)
            q = psnr(np.asarray(img), ref)
            results[(r, spec.name)] = (img, [f"PSNR {q:.2f} dB",
                                             f"{d.frames_received}/{spec.n_frames} "
                                             f"frames, est SNR {d.snr_db:.1f} dB"])
            print(f"    {spec.name} @ {snr:g} dB: PSNR {q:.2f}, "
                  f"{d.frames_received}/{spec.n_frames} frames")

    chan = "AWGN" if fading is None else f"{args.channel} fading + AWGN"
    canvas = _grid(
        [f"{s:g} dB" for s in snrs],
        [f"mode {spec.name}  ({len(waves[spec.name]) / FS:.0f} s on air)"
         for spec in modes],
        lambda r, c: results[(r, modes[c].name)],
        title=f"Received pictures - {Path(args.image).name}",
        subtitle=f"{chan}, SNR in a {SNR_REF_BW_HZ:.0f} Hz noise bandwidth; "
                 "encode -> modulate -> channel -> demodulate -> decode",
        footer="one channel seed per row, shared by every mode in it; "
               f"seed base {args.seed}. A single realization per cell, "
               "so read the failure modes, not the decimals.",
    )
    canvas.save(out / "noisy_ladder.png")
    print(f"  wrote {out / 'noisy_ladder.png'}")


def view_acquire(codec, flat, out, args, source):
    """Tune in late and blind-acquire: which arrival times still pay.

    This is the axis a longer mode wins on and `ladder` cannot see. The
    preamble is at the start only, so a late arrival depends entirely on
    `demodulate_blind` -- which needs `MIN_FRAMES_FOR_SYNC` (~73 frames,
    ~10.5 s) of *remaining* transmission to be guaranteed a full beacon
    superframe. A longer mode simply has more arrival times that still
    leave that much.
    """
    from sstvae import hfchannel
    from sstvae.modem import Modem
    from sstvae.modem.beacon import MIN_FRAMES_FOR_SYNC

    sync_s = MIN_FRAMES_FOR_SYNC * FRAME_SAMPLES / FS
    ref = (source.transpose(1, 2, 0) * 255).round().clip(0, 255)
    modem = Modem()
    fading = None if args.channel == "awgn" else args.channel
    snr = float(args.snr.split(",")[0])
    modes = [MODES[m] for m in args.modes]
    joins = [float(j) for j in args.joins.split(",")]

    results = {}
    for spec in modes:
        wave = modem.modulate(flat[: spec.n_latents], spec, callsign=args.callsign)
        y = hfchannel.apply_channel(wave, snr_db=snr, fading_preset=fading,
                                    seed=args.seed)
        dur = len(wave) / FS
        locked = in_range = 0
        for c, join in enumerate(joins):
            if join >= dur:
                results[(spec.name, c)] = (None, ["transmission over"])
                continue
            in_range += 1
            tail = y[int(join * FS):]
            try:
                d = modem.demodulate_blind(tail)
            except Exception as e:  # acquisition itself can fail outright
                results[(spec.name, c)] = (None, [f"no lock ({type(e).__name__})"])
                print(f"    {spec.name} join {join:g}s: no lock")
                continue
            if d.frame_offset is None:
                # Frames demodulated, but no beacon copy decoded -- so
                # there is no absolute position and the latents cannot be
                # placed. That is a failure with a picture's worth of
                # work already done, which is worth showing as its own
                # outcome rather than folding into "no lock".
                results[(spec.name, c)] = (
                    None, [f"no beacon ({d.n_frames} frames)",
                           f"{dur - join:.0f} s left, {sync_s:.1f} s guarantees it"])
                print(f"    {spec.name} join {join:g}s: {d.n_frames} frames, no beacon")
                continue
            locked += 1
            img = _reconstruct(codec, d)
            q = psnr(np.asarray(img), ref)
            # Which latent groups those frames actually covered. This is
            # the explanatory variable for the cliff in this view: the
            # groups go out in order, so a late enough join misses the
            # *coarse* group outright and no amount of fine detail
            # substitutes for it. Locking and getting a picture worth
            # having are two different things.
            first = max(d.frame_offset, 0) // FRAMES_PER_GROUP
            last = min(d.frame_offset + d.n_frames - 1,
                       spec.n_frames - 1) // FRAMES_PER_GROUP
            groups = f"group {first}" if first == last else f"groups {first}-{last}"
            results[(spec.name, c)] = (img, [f"PSNR {q:.2f} dB",
                                             f"{d.n_frames} frames from #{d.frame_offset}",
                                             f"covers {groups}"])
            print(f"    {spec.name} join {join:g}s: PSNR {q:.2f}, "
                  f"{d.n_frames} frames from #{d.frame_offset}, {groups}")
        print(f"    mode {spec.name} ({dur:.0f} s): locked from "
              f"{locked}/{in_range} arrival times inside the transmission")

    chan = "AWGN" if fading is None else f"{args.channel} fading + AWGN"
    canvas = _grid(
        [f"mode {spec.name}\n{MODES[spec.name].n_frames} fr" for spec in modes],
        [f"tune in at {j:g} s" for j in joins],
        lambda r, c: results[(modes[r].name, c)],
        title=f"Tuning in late - {Path(args.image).name}",
        subtitle=f"{chan} at {snr:g} dB; blind acquisition only (the preamble "
                 "is long gone), so each cell needs a whole beacon superframe "
                 "in what remains",
        footer=f"a guaranteed lock needs {MIN_FRAMES_FOR_SYNC} frames ({sync_s:.1f} s) "
               "of transmission left; shorter windows can still get lucky. "
               "The groups go out in order, so a join past group 0's airtime "
               "locks fine and still decodes badly. "
               "One realization per cell -- near threshold this is a coin, not a "
               "measurement (use scripts/late_join_sweep.py for rates).",
    )
    canvas.save(out / "acquisition.png")
    print(f"  wrote {out / 'acquisition.png'}")


# ------------------------------------------------------------------ main

VIEWS = ("sheet", "energy", "ablate", "ladder", "noisy", "acquire", "optim")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image")
    ap.add_argument("--out", default="latent-vis", type=Path,
                    help="directory for the PNGs (created if absent)")
    ap.add_argument("--views", default="all",
                    help=f"comma-separated subset of {','.join(VIEWS)} (default all)")
    ap.add_argument("--model", default=None, help=MODEL_HELP)
    ap.add_argument("--precision", choices=PRECISIONS, default=None,
                    help=PRECISION_HELP)
    ap.add_argument("--top", type=int, default=8,
                    help="how many channels the ablation close-ups show")
    ap.add_argument("--scale", type=int, default=4,
                    help="pixel scale of a contact-sheet tile (40x30 each)")
    ap.add_argument("--progress", action="store_true",
                    help="print progress through the 132 ablation decodes")
    o = ap.add_argument_group("the before/after view (optim)")
    o.add_argument("--optimize", type=float, default=120.0, metavar="SECONDS",
                   help="latent-optimization budget; the plateau is ~90 s on a "
                        "fast desktop, so the default runs past it and stops "
                        "early when the loss flattens")
    o.add_argument("--optimize-steps", type=int, default=1000,
                   help="hard step cap, whichever comes first")
    o.add_argument("--objective", default="flat", choices=("flat", "fading"),
                   help="the channel the optimizer optimizes *through*. "
                        "flat is the shipping one (every latent at full "
                        "confidence, one SNR); fading samples the measured "
                        "confidence/error joint. Their --objective-snr scales "
                        "differ by 5-7 dB, so compare them on received PSNR "
                        "and never on objective gain")
    o.add_argument("--objective-snr", type=float, default=None,
                   help="override the objective's SNR (default "
                        "OBJECTIVE_SNR_DB); unswept for the fading objective")
    o.add_argument("--through-channel", action="store_true",
                   help="measure erasure cost on the *received* picture in dB "
                        "of PSNR against the source, rather than |delta| on a "
                        "clean decode. Slower, and the only one of the two that "
                        "can compare two latent vectors of different sharpness")
    o.add_argument("--channel-reps", type=int, default=2,
                   help="--through-channel: channel realizations to average")
    o.add_argument("--optimize-mode", default="C", choices=sorted(MODES),
                   help="mode the objective optimizes for; C touches all "
                        f"{LATENT_CHANNELS} channels, A only the first group")
    g = ap.add_argument_group("the on-air views (noisy, acquire)")
    g.add_argument("--modes", default="ABC", help="modes to put in the grid")
    g.add_argument("--snr", default="10,6,3,0",
                   help="comma-separated SNRs in dB, referenced to "
                        f"{SNR_REF_BW_HZ:.0f} Hz. `acquire` uses the first only")
    g.add_argument("--channel", default="awgn",
                   choices=("awgn", "mpg", "mpp", "mpd"),
                   help="awgn, or a Watterson preset (good/poor/disturbed) "
                        "with AWGN on top")
    g.add_argument("--joins", default="0,10,20,40,60,80",
                   help="`acquire`: seconds into the transmission to tune in at")
    g.add_argument("--callsign", default="VIS", help="beacon callsign to send")
    g.add_argument("--seed", type=int, default=1000, help="channel seed base")
    args = ap.parse_args()

    # "all" excludes `optim`: it fetches an extra 18 MB gradient artifact
    # and runs the optimizer twice over, which is not what someone typing
    # no flags is asking for.
    default = tuple(v for v in VIEWS if v != "optim")
    views = default if args.views == "all" else tuple(args.views.split(","))
    unknown = set(views) - set(VIEWS)
    if unknown:
        raise SystemExit(f"unknown view(s): {', '.join(sorted(unknown))}")

    args.out.mkdir(parents=True, exist_ok=True)
    codec = load_codec(args.model, precision=args.precision)
    source = load_image(args.image)
    flat = codec.encode(source)
    # (132, 30, 40): channels 0-43 coarse, 44-87 mid, 88-131 fine.
    from sstvae.latents import flat_to_latents
    z = flat_to_latents(flat[None].astype(np.float32))[0]
    print(f"encoded {args.image}: {z.shape} latents, "
          f"overall RMS {np.sqrt((z ** 2).mean()):.4f}")

    if "sheet" in views:
        view_sheet(z, args.out, args)
    if "energy" in views:
        view_energy(z, args.out, args)
    if "ablate" in views:
        view_ablate(codec, z, args.out, args, source)
    if "ladder" in views:
        view_ladder(codec, z, args.out, args, source)
    if "noisy" in views:
        view_noisy(codec, flat, args.out, args, source)
    if "acquire" in views:
        view_acquire(codec, flat, args.out, args, source)
    if "optim" in views:
        view_optim(codec, flat, args.out, args, source)


if __name__ == "__main__":
    main()

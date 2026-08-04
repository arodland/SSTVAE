#!/usr/bin/env python3
"""Re-derive `latent_optim.FADING_PROFILES` and check it is still stable.

    python scripts/derive_fading_profile.py

The table is 27 numbers describing what a reported confidence weight
actually promises, and the optimizer's fading objective samples it. It
is a property of the *demodulator* under fading, not of the codec, so a
new checkpoint does not invalidate it -- but any change to the modem,
the pilot spacing or the equalizer does, and nothing else would notice.

The shipped candidate is 27 numbers taken from one image on mpd. Before
freezing that into two implementations, check the obvious ways it could
be an accident: a different picture, a different fading preset, a
different SNR. The error column is normalised by the top bin, which is
what makes pooling across SNR legitimate -- an SNR change should scale
every bin together and cancel.
"""
import sys
from pathlib import Path

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sstvae import hfchannel
from sstvae.codec import load_codec
from sstvae.config import MODES
from sstvae.images import load_image
from sstvae.latent_optim import FADING_PROFILES
from sstvae.modem import Modem, SyncError

IMAGES = ["damselfly.png", "woods.png", "w0nycert.png", "xchat1.png",
          "n2jqsl.png", "wonder_wheel.jpg"]
PRESETS, SNRS, SEEDS = ("mpp", "mpd"), (3.0, 6.0, 12.0), range(3)
EDGES = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.65, 0.8, 0.95, 1.001])
spec, modem = MODES["C"], Modem()
codec = load_codec()

cells = {}   # (preset, image) -> list of (fracs, rels)
for name in IMAGES:
    src = load_image(Path(__file__).resolve().parent.parent / "data/optim_corpus" / name)
    flat = codec.encode(src)[: spec.n_latents]
    tx = flat / np.sqrt(np.mean(flat ** 2))
    wave = modem.modulate(flat, spec, callsign="VIS")
    for preset in PRESETS:
        for snr in SNRS:
            W, E = [], []
            for seed in SEEDS:
                y = hfchannel.apply_channel(wave, snr_db=snr, fading_preset=preset,
                                            seed=11000 + seed)
                try:
                    d = modem.demodulate(y)
                except SyncError:
                    continue
                got = d.weights > 0
                W.append(d.weights[got]); E.append(d.latents[got] - tx[got])
            if not W:
                continue
            W, E = np.concatenate(W), np.concatenate(E)
            top = W >= EDGES[-2]
            ref = np.sqrt(np.mean(E[top] ** 2))
            fr, rl = [], []
            for lo, hi in zip(EDGES[:-1], EDGES[1:]):
                m = (W >= lo) & (W < hi)
                fr.append(m.mean())
                rl.append(np.sqrt(np.mean(E[m] ** 2)) / ref if m.sum() >= 50 else np.nan)
            cells.setdefault(preset, []).append((np.array(fr), np.array(rl), name, snr))

ship_p, _, ship_r = FADING_PROFILES["measured"]
print(f"{'bin':>12} {'shipped p':>10} {'p mean':>8} {'p range':>14} "
      f"{'shipped rel':>12} {'rel mean':>9} {'rel range':>16}")
allc = [c for v in cells.values() for c in v]
P = np.array([c[0] for c in allc]); R = np.array([c[1] for c in allc])
for i, (lo, hi) in enumerate(zip(EDGES[:-1], EDGES[1:])):
    r = R[:, i][~np.isnan(R[:, i])]
    print(f"[{lo:.2f},{hi:.2f}) {ship_p[i]:10.3f} {P[:, i].mean():8.3f} "
          f"{P[:, i].min():6.3f}-{P[:, i].max():<6.3f} {ship_r[i]:12.2f} "
          f"{r.mean():9.2f} {r.min():7.2f}-{r.max():<7.2f}")

print("\nby preset (mean over images x SNRs):")
for preset, v in cells.items():
    p = np.array([c[0] for c in v]).mean(0); r = np.nanmean([c[1] for c in v], axis=0)
    print(f"  {preset} p:   " + " ".join(f"{x:5.3f}" for x in p))
    print(f"  {preset} rel: " + " ".join(f"{x:5.2f}" for x in r))
print(f"\nfrac of latents below w=0.2, by condition: "
      f"{np.min([c[0][:2].sum() for c in allc]):.3f}-{np.max([c[0][:2].sum() for c in allc]):.3f}")

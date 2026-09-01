#!/usr/bin/env python3
"""Cross-decode compatibility: every encoder against every decoder.

    python scripts/cross_decode.py v4.pt d50.pt --dump-dir /tmp/xd
    python scripts/cross_decode.py v4.pt d50.pt --conditions "clean,mpp 6"

Two stations only interoperate if the latents one sends mean the same
thing to the other's decoder. A fine-tune keeps the architecture and the
unit-RMS latent contract, so a mismatched pair *runs* -- it cannot fail
loudly -- and the only question is how much picture is lost. That makes
this a silent failure by construction, and the number below is the bar a
rolling upgrade has to clear before a new checkpoint can be published to
a population that will not all update at once.

Two regimes, and the difference between them is the point:

* **Codec only** (the default, no --conditions). Encode -> per-mode
  truncation -> decode, the same idiom as scripts/eval_nonphoto.py. No
  modem, so the number is the model's own contribution, isolated.
* **Through the modem** (--conditions). encode -> modulate -> simulated
  channel -> demodulate -> decode, so the latents the far decoder sees
  are the ones that actually survived the air, complete with pilot-EQ
  residual, the clipper's ~0.794 latent gain and per-latent confidence
  weights. `clean` here still means through the modem and is therefore
  **not** the same measurement as codec-only.

The modulate/channel/demodulate step depends only on the *encoder*, so
it is run once per (encoder, mode, condition, image) and its output fed
to every decoder -- which is also what makes the comparison exact rather
than merely paired: every decoder sees the identical received latents,
not its own draw from the same distribution.

Reported per content category, because the classes do not have to
degrade together: a decoder that has drifted in how it renders text can
be fine on photographs, and pooling them would average that away.

`own` is each encoder paired with its own decoder; `penalty` is
cross-minus-own in dB, so it is <= 0 and more negative is worse.

SNRs are in `config.SNR_REF_BW_HZ`.
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sstvae import hfchannel  # noqa: E402
from sstvae.codec import load_codec, pad_to_full, reconstruct  # noqa: E402
from sstvae.config import MODES, SNR_REF_BW_HZ  # noqa: E402
from sstvae.images import fit_image  # noqa: E402
from sstvae.modem import Modem, SyncError  # noqa: E402

# (label, snr_db or None, fading preset or None) -- same set as
# scripts/ab_checkpoint_sweep.py, so the two tools' conditions match.
CONDITIONS = [
    ("clean", None, None),
    ("awgn 20 dB", 20.0, None),
    ("awgn 10 dB", 10.0, None),
    ("awgn 6 dB", 6.0, None),
    ("awgn 3 dB", 3.0, None),
    ("awgn 0 dB", 0.0, None),
    ("mpg 10 dB", 10.0, "mpg"),
    ("mpp 10 dB", 10.0, "mpp"),
    ("mpp 6 dB", 6.0, "mpp"),
    ("mpd 6 dB", 6.0, "mpd"),
    ("mpp 0 dB", 0.0, "mpp"),
]

CODEC_ONLY = "codec"


def psnr(a, b) -> float:
    x, y = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    mse = np.mean((x - y) ** 2)
    return float("inf") if mse == 0 else 10 * np.log10(255.0**2 / mse)


def load_coco(repo: str, split: str, n: int):
    from datasets import load_dataset

    ds = load_dataset(repo, split=split)
    return [("coco", fit_image(ds[i]["image"])) for i in range(min(n, len(ds)))]


def load_nonphoto(per_class: int, salt: str = "eval"):
    from sstvae import nonphoto

    return [
        (cls, nonphoto.generate(cls, i, salt=salt))
        for cls in nonphoto.CLASSES
        for i in range(per_class)
    ]


def load_dir(path: Path, n: int):
    from PIL import Image

    files = sorted(
        p for p in path.iterdir()
        if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp")
    )[:n]
    return [(path.name, fit_image(Image.open(p))) for p in files]


def save_grid(out_path: Path, panels, n: int):
    """n x n grid of decodes: rows are encoders, columns are decoders.

    No source panel, and a grid rather than a row, because the comparison
    this test exists to make is between the decodes themselves: the
    diagonal is own-decode and everything off it is a cross, so reading a
    column top to bottom is one decoder's view of every encoder and
    reading a row is one encoder seen by every decoder. A single row of
    panels puts the two members of each of those pairs far apart.
    """
    from PIL import Image, ImageDraw

    if len(panels) != n * n:
        raise RuntimeError(f"expected {n * n} panels for a {n}x{n} grid, "
                           f"got {len(panels)}")
    size = next((t.size for _l, t in panels if t is not None), None)
    if size is None:
        return  # nothing decoded at all here; no grid worth writing
    w, h = size
    grid = Image.new("RGB", (w * n, (h + 14) * n), (0, 0, 0))
    d = ImageDraw.Draw(grid)
    for k, (lab, t) in enumerate(panels):
        row, col = divmod(k, n)
        y = row * (h + 14)
        if t is not None:
            grid.paste(t, (col * w, y + 14))
        d.text((col * w + 4, y + 2),
               lab if t is not None else f"{lab} (no sync)",
               fill=(255, 255, 255))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(out_path)


def received_latents(modem, wave_lat, spec, snr_db, fading, seed):
    """Latents+weights as they arrive at a receiver, or None on sync loss."""
    y = modem.modulate(wave_lat, spec)
    if snr_db is not None or fading is not None:
        y = hfchannel.apply_channel(y, snr_db=snr_db, fading_preset=fading,
                                    seed=seed)
    try:
        r = modem.demodulate(y)
    except SyncError:
        return None
    return pad_to_full(r.latents), pad_to_full(r.weights)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("models", nargs="+", help="2+ checkpoints (.pt or .onnx dir)")
    ap.add_argument("--labels", default=None, help="comma-separated names")
    ap.add_argument("--dataset", default="arodland/coco640-sstvae")
    ap.add_argument("--split", default="validation")
    ap.add_argument("--n-images", type=int, default=16)
    ap.add_argument("--content", default="coco,nonphoto")
    ap.add_argument("--nonphoto-per-class", type=int, default=4)
    ap.add_argument("--modes", default="ABC")
    ap.add_argument(
        "--conditions",
        default=None,
        help="comma-separated substrings of "
        f"{[c[0] for c in CONDITIONS]}; passing this routes the test "
        "through the modem and the simulated channel instead of "
        "measuring the codec alone. Note 'clean' here is through the "
        "modem, which is not the same as the codec-only default",
    )
    ap.add_argument("--seed", type=int, default=1000,
                    help="channel seed base; seed+image index per image")
    ap.add_argument("--csv", type=Path, default=None)
    ap.add_argument("--dump-dir", type=Path, default=None)
    ap.add_argument("--dump-per-class", type=int, default=1)
    ap.add_argument(
        "--dump-mode", default="C",
        help="which mode the sample grids are rendered at (default C)",
    )
    ap.add_argument(
        "--dump-condition", default=None,
        help="which condition the sample grids are rendered at "
        "(default: the first selected condition)",
    )
    args = ap.parse_args()

    if len(args.models) < 2:
        ap.error("need at least two checkpoints to cross")
    labels = (args.labels.split(",") if args.labels
              else [Path(m).stem for m in args.models])
    if len(labels) != len(args.models):
        ap.error(f"--labels has {len(labels)} names for {len(args.models)} models")

    if args.conditions:
        keep = [c.strip() for c in args.conditions.split(",")]
        conds = [c for c in CONDITIONS if any(k in c[0] for k in keep)]
        if not conds:
            ap.error(f"--conditions matched nothing in "
                     f"{[c[0] for c in CONDITIONS]}")
    else:
        conds = [(CODEC_ONLY, None, None)]
    dump_cond = args.dump_condition or conds[0][0]

    content = []
    for spec in args.content.split(","):
        spec = spec.strip()
        if spec == "coco":
            content += load_coco(args.dataset, args.split, args.n_images)
        elif spec == "nonphoto":
            content += load_nonphoto(args.nonphoto_per_class)
        elif spec:
            content += load_dir(Path(spec), args.n_images)
    if not content:
        ap.error("no images selected by --content")
    classes = sorted({c for c, _ in content})
    images = [im for _c, im in content]
    img_class = np.array([c for c, _im in content])

    codecs = [load_codec(m) for m in args.models]
    nm = len(codecs)
    via = ("codec only, no modem" if conds[0][0] == CODEC_ONLY
           else f"through the modem, SNR referenced to {SNR_REF_BW_HZ:.0f} Hz")
    print(f"{len(content)} images ({', '.join(classes)}), "
          f"{nm} checkpoints -> {nm * nm} encoder/decoder pairs; {via}",
          file=sys.stderr)

    modem = Modem() if conds[0][0] != CODEC_ONLY else None

    # Encode once per (model, image): the encoder is what varies, and the
    # same latent vector is then handed to every decoder.
    latents = [[c.encode(im) for im in images] for c in codecs]

    rows, grids = [], {}
    for spec in (MODES[m] for m in args.modes):
        n = spec.n_latents
        for label, snr_db, fading in conds:
            for ei in range(nm):
                # One transmission per (encoder, mode, condition, image),
                # shared by every decoder: the received latents are then
                # identical across decoders rather than merely comparable.
                rx = []
                for i in range(len(images)):
                    if modem is None:
                        rx.append((pad_to_full(latents[ei][i][:n]),
                                   pad_to_full(np.ones(n))))
                    else:
                        rx.append(received_latents(
                            modem, latents[ei][i][:n], spec, snr_db, fading,
                            args.seed + i))
                for di in range(nm):
                    got = np.full(len(images), np.nan)
                    dumping = (args.dump_dir and spec.name == args.dump_mode
                               and label == dump_cond)
                    want = (_dump_idx(img_class, classes, args.dump_per_class)
                            if dumping else set())
                    for i in range(len(images)):
                        # None on sync loss, and it is still appended, so
                        # the grid keeps its row-major (encoder, decoder)
                        # order instead of silently shifting up a cell.
                        out = None
                        if rx[i] is not None:
                            out = reconstruct(codecs[di], *rx[i])
                            got[i] = psnr(images[i], out)
                        if i in want:
                            grids.setdefault(i, []).append(
                                (f"{labels[ei]}->{labels[di]}", out))
                    for cls in classes:
                        sel = img_class == cls
                        vals = got[sel]
                        rows.append({
                            "mode": spec.name, "condition": label,
                            "content": cls,
                            "encoder": labels[ei], "decoder": labels[di],
                            "psnr": (float(np.nanmean(vals))
                                     if np.any(~np.isnan(vals)) else float("nan")),
                            "sync": f"{int(np.count_nonzero(~np.isnan(vals)))}/"
                                    f"{int(np.count_nonzero(sel))}",
                                    "own": ei == di,
                                    })

    own = {(r["mode"], r["condition"], r["content"], r["encoder"]): r["psnr"]
           for r in rows if r["own"]}
    for r in rows:
        r["penalty"] = r["psnr"] - own[
            (r["mode"], r["condition"], r["content"], r["encoder"])]

    if args.dump_dir:
        for i, panels in grids.items():
            save_grid(args.dump_dir /
                      f"{args.dump_mode}_{dump_cond.replace(' ', '_')}"
                      f"_{img_class[i]}_{i}.png", panels, nm)
        print(f"wrote {len(grids)} {nm}x{nm} grids to {args.dump_dir}",
              file=sys.stderr)

    print("\n".join(f"{labels[i]} = {m}" for i, m in enumerate(args.models)))
    print(f"\nPSNR dB, {via}. penalty = this pair minus the same encoder's "
          "own decoder.\n")
    print("| Mode | Condition | Content | Encoder | Decoder | PSNR | penalty | sync |")
    print("|---|---|---|---|---|---|---|---|")
    for r in rows:
        tag = "own" if r["own"] else f"{r['penalty']:+.2f}"
        print(f"| {r['mode']} | {r['condition']} | {r['content']} | "
              f"{r['encoder']} | {r['decoder']} | {r['psnr']:.2f} | {tag} | "
              f"{r['sync']} |")

    print("\nWorst cross-decode penalty per mode and condition:")
    for m in args.modes:
        for label, _s, _f in conds:
            bad = [r for r in rows if r["mode"] == m
                   and r["condition"] == label and not r["own"]
                   and not np.isnan(r["penalty"])]
            if bad:
                w = min(bad, key=lambda r: r["penalty"])
                print(f"  mode {m} {label:>10}: {w['penalty']:+.2f} dB  "
                      f"({w['encoder']}->{w['decoder']} on {w['content']})")

    if args.csv:
        with open(args.csv, "w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
            wr.writeheader()
            wr.writerows(rows)
        print(f"\nwrote {args.csv}", file=sys.stderr)


def _dump_idx(img_class, classes, per_class):
    """First `per_class` image indices of each content class."""
    out = set()
    for cls in classes:
        out.update(np.flatnonzero(img_class == cls)[:per_class].tolist())
    return out


if __name__ == "__main__":
    main()

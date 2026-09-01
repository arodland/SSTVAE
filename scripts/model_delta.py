#!/usr/bin/env python3
"""How far did each half of the codec actually move between two checkpoints?

    python scripts/model_delta.py v4.pt v5.pt
    python scripts/model_delta.py v4.pt v5.pt --content coco,nonphoto

Not a quality question -- `scripts/ab_checkpoint_sweep.py` answers that.
This is the plainer one: **on the encoder side, how much do the latents
change for the same image; on the decoder side, how much does the image
change for the same latents.** A version bump that rewrites the latent
space is a different kind of event from one that polishes the decoder,
and the quality delta alone does not distinguish them.

## Putting the two in the same units

The two changes are natively in different spaces -- latents and pixels --
so they are not comparable as measured. They are made commensurable by
**referring both to the output image**, which is the one domain both
halves land in:

    encoder-induced   PSNR( D(z_A(x)), D(z_B(x)) )   decoder held fixed
    decoder-induced   PSNR( D_A(z), D_B(z) )         latents held fixed

Both are "how far did the picture move, in dB", and **higher means less
change**. Each is averaged over both choices of the held-fixed half
(both decoders, both encoders' latents) so neither checkpoint is
privileged as the reference.

The yardstick is `recon`, the codec's own reconstruction PSNR against
the source image. A change 20 dB above `recon` is far below the error
the codec already makes and is a polish; one at or below `recon` is a
rewrite. The `vs_recon` columns are that comparison, in dB -- **bigger
is more negligible**.

The raw latent-domain number is reported too, since it is what the
question literally asks for: `dz` is RMS(z_B - z_A) in units of the
unit-RMS latent contract, so 0.10 means the latents moved by 10% of
their own RMS.

## When the latent number is meaningless

`corr` is the correlation between z_A and z_B. Two checkpoints that
share a lineage keep their latent channels aligned and correlate near
1. Two independently trained models need not: nothing pins channel
order or sign, so their latent spaces can be a permutation apart, `dz`
comes out enormous, and it means nothing. Below `CORR_WARN` the tool
says so. The image-domain columns stay valid either way -- they never
assume the two latent spaces are the same space.

Codec-only by default: no modem, no channel, because this is a question
about the models. `--conditions` routes it through the modem if you
want the change as a receiver would see it.
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sstvae.codec import MODEL_HELP, load_codec, pad_to_full, reconstruct  # noqa: E402
from sstvae.config import MODES, SNR_REF_BW_HZ  # noqa: E402
from sstvae.modem import Modem  # noqa: E402

from cross_decode import (  # noqa: E402
    CODEC_ONLY,
    CONDITIONS,
    load_coco,
    load_dir,
    load_nonphoto,
    psnr,
    received_latents,
)

# Below this, the two latent spaces are not obviously the same space and
# `dz` stops being interpretable. Lineage fine-tunes sit far above it.
CORR_WARN = 0.9


def latent_delta(za: np.ndarray, zb: np.ndarray) -> tuple[float, float]:
    """(RMS change in units of the latent RMS, correlation)."""
    rms_a = float(np.sqrt(np.mean(za**2)))
    dz = float(np.sqrt(np.mean((zb - za) ** 2))) / (rms_a or 1.0)
    sa, sb = za - za.mean(), zb - zb.mean()
    denom = float(np.linalg.norm(sa) * np.linalg.norm(sb))
    corr = float(np.dot(sa, sb) / denom) if denom else float("nan")
    return dz, corr


def mean_sem(v: list[float]) -> tuple[float, float]:
    a = np.array([x for x in v if np.isfinite(x)], dtype=float)
    if a.size == 0:
        return float("nan"), float("nan")
    sem = float(a.std(ddof=1) / np.sqrt(a.size)) if a.size > 1 else float("nan")
    return float(a.mean()), sem


def self_check() -> None:
    """The parts with an answer known in advance."""
    rng = np.random.default_rng(0)
    z = rng.standard_normal(512)

    dz, corr = latent_delta(z, z)
    assert dz == 0.0 and abs(corr - 1.0) < 1e-12, (dz, corr)

    # A change of exactly 10% of the latent RMS reads as 0.10.
    z2 = z + 0.1 * np.sqrt(np.mean(z**2)) * rng.standard_normal(512)
    dz, _ = latent_delta(z, z2)
    assert abs(dz - 0.1) < 0.02, dz

    # Scale invariance: dz is relative, so scaling both sides cannot
    # move it. (It would if the normalisation were dropped.)
    dz_scaled, _ = latent_delta(7.0 * z, 7.0 * z2)
    assert abs(dz_scaled - dz) < 1e-12, (dz, dz_scaled)

    # A sign flip is near-perfectly anti-correlated and must not read as
    # "aligned" -- this is the permutation/sign hazard the warning is for.
    _, corr = latent_delta(z, -z)
    assert corr < -0.99, corr

    print("self-check passed: dz is relative and scale-free, corr detects misalignment")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--self-check", action="store_true",
                    help="verify the metrics and exit; no checkpoint or network")
    if "--self-check" in sys.argv:
        self_check()
        return
    ap.add_argument("model_a", help=MODEL_HELP)
    ap.add_argument("model_b", help="the newer checkpoint; deltas are B - A")
    ap.add_argument("--labels", default=None, help="comma-separated names")
    ap.add_argument("--dataset", default="arodland/coco640-sstvae")
    ap.add_argument("--split", default="validation")
    ap.add_argument("--n-images", type=int, default=16)
    ap.add_argument("--content", default="coco,nonphoto")
    ap.add_argument("--nonphoto-per-class", type=int, default=4)
    ap.add_argument("--modes", default="C")
    ap.add_argument(
        "--conditions", default=None,
        help="comma-separated substrings of "
        f"{[c[0] for c in CONDITIONS]}; routes through the modem and the "
        "simulated channel instead of measuring the models alone",
    )
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--csv", type=Path, default=None)
    args = ap.parse_args()

    labels = (args.labels.split(",") if args.labels
              else [Path(args.model_a).stem, Path(args.model_b).stem])
    if len(labels) != 2:
        ap.error(f"--labels needs 2 names, got {len(labels)}")

    if args.conditions:
        keep = [c.strip() for c in args.conditions.split(",")]
        conds = [c for c in CONDITIONS if any(k in c[0] for k in keep)]
        if not conds:
            ap.error(f"--conditions matched nothing in "
                     f"{[c[0] for c in CONDITIONS]}")
    else:
        conds = [(CODEC_ONLY, None, None)]

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

    codecs = [load_codec(args.model_a), load_codec(args.model_b)]
    via = ("codec only, no modem" if conds[0][0] == CODEC_ONLY
           else f"through the modem, SNR referenced to {SNR_REF_BW_HZ:.0f} Hz")
    print(f"{len(content)} images ({', '.join(classes)}); "
          f"A = {labels[0]}, B = {labels[1]}; {via}", file=sys.stderr)

    modem = Modem() if conds[0][0] != CODEC_ONLY else None
    latents = [[c.encode(im) for im in images] for c in codecs]

    rows = []
    for spec in (MODES[m] for m in args.modes):
        n = spec.n_latents
        for label, snr_db, fading in conds:
            # rx[ei][i] -- what a receiver holds after encoder ei's
            # transmission. One transmission per (encoder, image), reused
            # by both decoders, so a decoder comparison is exact rather
            # than two draws from one channel.
            rx = [[None] * len(images) for _ in range(2)]
            for ei in range(2):
                for i in range(len(images)):
                    if modem is None:
                        rx[ei][i] = (pad_to_full(latents[ei][i][:n]),
                                     pad_to_full(np.ones(n)))
                    else:
                        rx[ei][i] = received_latents(
                            modem, latents[ei][i][:n], spec, snr_db, fading,
                            args.seed + i)

            per = {k: [] for k in
                   ("dz", "corr", "enc_img", "dec_img", "recon")}
            keep_class = []
            for i in range(len(images)):
                if rx[0][i] is None or rx[1][i] is None:
                    continue
                # Only the latents this mode actually carries: the rest
                # is zero padding in both, and averaging it in would
                # deflate dz by the padding fraction rather than by
                # anything either model did.
                za, zb = rx[0][i][0][:n], rx[1][i][0][:n]
                dz, corr = latent_delta(za, zb)

                # Four decodes: [encoder][decoder].
                dec = [[reconstruct(codecs[di], *rx[ei][i])
                        for di in range(2)] for ei in range(2)]

                # Encoder-induced: decoder held fixed, averaged over both.
                enc_img = np.mean([psnr(dec[0][di], dec[1][di])
                                   for di in range(2)])
                # Decoder-induced: latents held fixed, averaged over both.
                dec_img = np.mean([psnr(dec[ei][0], dec[ei][1])
                                   for ei in range(2)])
                # The yardstick: each codec's own error, averaged.
                recon = np.mean([psnr(images[i], dec[m][m]) for m in range(2)])

                for k, v in (("dz", dz), ("corr", corr), ("enc_img", enc_img),
                             ("dec_img", dec_img), ("recon", recon)):
                    per[k].append(v)
                keep_class.append(img_class[i])

            keep_class = np.array(keep_class)
            for cls in [*classes, "all"]:
                sel = (np.ones(len(keep_class), bool) if cls == "all"
                       else keep_class == cls)
                if not np.any(sel):
                    continue
                pick = {k: list(np.array(v)[sel]) for k, v in per.items()}
                dz, dz_sem = mean_sem(pick["dz"])
                corr, _ = mean_sem(pick["corr"])
                enc_img, enc_sem = mean_sem(pick["enc_img"])
                dec_img, dec_sem = mean_sem(pick["dec_img"])
                recon, _ = mean_sem(pick["recon"])
                rows.append({
                    "mode": spec.name, "condition": label, "content": cls,
                    "dz": dz, "dz_sem": dz_sem, "corr": corr,
                    "enc_img_db": enc_img, "enc_img_sem": enc_sem,
                    "dec_img_db": dec_img, "dec_img_sem": dec_sem,
                    "recon_db": recon,
                    "enc_vs_recon_db": enc_img - recon,
                    "dec_vs_recon_db": dec_img - recon,
                    "n": int(np.count_nonzero(sel)),
                })
                print(
                    f"  mode {spec.name} {label:>10} {cls:>9}: "
                    f"dz {dz:.3f} corr {corr:.4f}  "
                    f"enc {enc_img:5.1f} dB  dec {dec_img:5.1f} dB  "
                    f"recon {recon:5.1f} dB  n {int(np.count_nonzero(sel))}",
                    file=sys.stderr, flush=True,
                )

    print(f"\nA = {args.model_a}\nB = {args.model_b}\n")
    print("How far each half moved. `enc` and `dec` are the same units --\n"
          "PSNR in dB between two decodes that differ in only that half, so\n"
          "**higher means less change**. `recon` is each codec's own error,\n"
          "the yardstick: `vs recon` well above 0 means the change is small\n"
          "next to the error the codec already makes.\n"
          "`dz` is RMS(z_B - z_A) in units of the unit-RMS latent contract.\n")
    print("| Mode | Condition | Content | dz | corr | enc | dec | recon | "
          "enc vs recon | dec vs recon |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(
            f"| {r['mode']} | {r['condition']} | {r['content']} | "
            f"{r['dz']:.3f} | {r['corr']:.4f} | "
            f"{r['enc_img_db']:.1f} ±{r['enc_img_sem']:.1f} | "
            f"{r['dec_img_db']:.1f} ±{r['dec_img_sem']:.1f} | "
            f"{r['recon_db']:.1f} | "
            f"{r['enc_vs_recon_db']:+.1f} | {r['dec_vs_recon_db']:+.1f} |"
        )

    overall = [r for r in rows if r["content"] == "all"]
    enc = float(np.nanmean([r["enc_img_db"] for r in overall]))
    dec = float(np.nanmean([r["dec_img_db"] for r in overall]))
    recon = float(np.nanmean([r["recon_db"] for r in overall]))
    dz = float(np.nanmean([r["dz"] for r in overall]))
    corr = float(np.nanmean([r["corr"] for r in overall]))
    print(f"\nOver {len(overall)} (mode, condition) cells, all content pooled:")
    print(f"  latents moved   dz {dz:.3f} of latent RMS   (corr {corr:.4f})")
    print(f"  encoder-induced image change  {enc:.1f} dB "
          f"({enc - recon:+.1f} vs recon)")
    print(f"  decoder-induced image change  {dec:.1f} dB "
          f"({dec - recon:+.1f} vs recon)")
    print(f"  codec's own reconstruction    {recon:.1f} dB")
    louder = "encoder" if enc < dec else "decoder"
    print(f"  -> the {louder} moved more, by {abs(enc - dec):.1f} dB")

    if np.isfinite(corr) and corr < CORR_WARN:
        print(
            f"\n  NOTE: latent correlation {corr:.3f} is below {CORR_WARN}.\n"
            "  These two latent spaces may not be the same space -- nothing\n"
            "  pins channel order or sign across independent training runs --\n"
            "  so `dz` is not interpretable. The image-domain columns are\n"
            "  unaffected; they never assume a shared latent basis."
        )

    if args.csv:
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {args.csv}", file=sys.stderr)


if __name__ == "__main__":
    main()

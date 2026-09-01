#!/usr/bin/env python3
"""Split the quality change between two checkpoints into an encoder
share and a decoder share.

    python scripts/encoder_decoder_split.py v4.pt v5.pt
    python scripts/encoder_decoder_split.py v4.pt v5.pt --conditions "clean,mpp 6"

`scripts/ab_checkpoint_sweep.py` says *how much* a version bump moved
the picture; this says *which half moved it*. Both halves ship together,
so the question is not idle: an encoder-side gain is paid for by every
receiver of that station whether or not they upgrade, while a
decoder-side gain is claimable unilaterally by upgrading your own
receiver. They also fail differently on a population that upgrades
piecemeal, which is what scripts/cross_decode.py measures.

## The decomposition

Four combinations of (encoder, decoder) over A and B -- the same 2x2
grid cross_decode.py builds, scored per image so the pairing is exact:

      AA = enc A, dec A      BA = enc B, dec A
      AB = enc A, dec B      BB = enc B, dec B

    encoder share  E = 1/2 [ (BA - AA) + (BB - AB) ]
    decoder share  D = 1/2 [ (AB - AA) + (BB - BA) ]
    interaction    I = (BB - AB) - (BA - AA)

`E` is the effect of swapping the encoder, averaged over both decoders;
`D` likewise. **The split is exact: E + D is identically BB - AA**,
which is the total the A/B sweep reports. Nothing is left over, and no
variance needs apportioning -- this is an algebraic identity, not a
regression.

## Read the interaction before you quote the split

`I` is how much the encoder's contribution *depends on which decoder it
is paired with*. The two halves are trained jointly and co-adapt, so a
mismatched pair is not a neutral probe: it is an off-manifold
combination neither half was trained for. When `|I|` is large next to
the total, the mixed cells are dominated by that mismatch and the
encoder/decoder attribution is not trustworthy -- the honest report is
then "the change is joint", not a split. The tool prints the ratio and
says so rather than leaving it to be noticed.

`cross` below is the mean of the two mismatched cells minus the mean of
the two matched ones: how much picture is lost purely to pairing halves
that were not trained together, in dB. It is the same quantity
cross_decode.py reports as a penalty, restated here so the interaction
has its magnitude beside it.

Two regimes, inherited from cross_decode.py: **codec only** (the
default -- encode, per-mode truncation, decode, no modem) isolates the
model's own contribution, and **through the modem** (`--conditions`)
feeds each decoder the latents that actually survived the air. `clean`
under `--conditions` still goes through the modem and is not the same
measurement as the codec-only default.

SNRs are in `config.SNR_REF_BW_HZ`.
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

# How large |I| may get, relative to |total|, before the split stops
# meaning anything. Not a statistical threshold -- a reading aid. At 1.0
# the interaction is the size of the whole effect being attributed.
INTERACTION_WARN = 0.5


def paired(d: np.ndarray) -> tuple[float, float, int]:
    """Mean, standard error and count over the images that scored."""
    v = d[~np.isnan(d)]
    if v.size == 0:
        return float("nan"), float("nan"), 0
    sem = float(v.std(ddof=1) / np.sqrt(v.size)) if v.size > 1 else float("nan")
    return float(v.mean()), sem, int(v.size)


def decompose(aa, ba, ab, bb):
    """The 2x2 split, per image. Arrays in, arrays out.

    Kept separate from the measurement so the algebra can be checked
    against hand-computed cases without a codec, a modem or an image --
    see `--self-check`, which is the only part of this script with a
    right answer known in advance.
    """
    return {
        "total": bb - aa,
        "encoder": 0.5 * ((ba - aa) + (bb - ab)),
        "decoder": 0.5 * ((ab - aa) + (bb - ba)),
        "interaction": (bb - ab) - (ba - aa),
        # Mismatched pairs minus matched ones: what pairing halves that
        # were not trained together costs, which is <= 0 whenever the
        # halves have co-adapted at all.
        "cross_penalty": 0.5 * (ba + ab) - 0.5 * (aa + bb),
    }


def self_check() -> None:
    """Assert the decomposition on cases whose answers are arithmetic."""
    a = np.array

    # Purely additive: encoder worth +1, decoder worth +2, no interaction.
    d = decompose(a([10.0]), a([11.0]), a([12.0]), a([13.0]))
    assert np.allclose(d["encoder"], 1.0), d
    assert np.allclose(d["decoder"], 2.0), d
    assert np.allclose(d["interaction"], 0.0), d
    assert np.allclose(d["cross_penalty"], 0.0), d

    # Purely joint: both matched pairs good, both crosses bad. Neither
    # half moves anything on its own, and the whole effect is
    # interaction -- the case the warning exists for.
    d = decompose(a([10.0]), a([8.0]), a([8.0]), a([10.0]))
    assert np.allclose(d["total"], 0.0), d
    assert np.allclose(d["encoder"], 0.0), d
    assert np.allclose(d["decoder"], 0.0), d
    assert np.allclose(d["interaction"], 4.0), d
    assert np.allclose(d["cross_penalty"], -2.0), d

    # The identity that makes this a decomposition rather than a fit,
    # on a deliberately lopsided random grid.
    rng = np.random.default_rng(0)
    aa, ba, ab, bb = (rng.normal(25, 3, 64) for _ in range(4))
    d = decompose(aa, ba, ab, bb)
    assert np.allclose(d["encoder"] + d["decoder"], d["total"]), \
        "encoder + decoder must equal total exactly"

    print("self-check passed: split is exact and the joint case reads as joint")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--self-check", action="store_true",
                    help="verify the decomposition algebra and exit; needs "
                         "no checkpoint, dataset or network")
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
    ap.add_argument("--modes", default="ABC")
    ap.add_argument(
        "--conditions", default=None,
        help="comma-separated substrings of "
        f"{[c[0] for c in CONDITIONS]}; passing this routes the test "
        "through the modem and the simulated channel instead of "
        "measuring the codec alone",
    )
    ap.add_argument("--seed", type=int, default=1000,
                    help="channel seed base; seed+image index per image")
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
            # cell[ei][di][image] -- one transmission per encoder, shared
            # by both decoders, so the two decoders see *identical*
            # received latents rather than two draws from one channel.
            cell = np.full((2, 2, len(images)), np.nan)
            for ei in range(2):
                rx = []
                for i in range(len(images)):
                    if modem is None:
                        rx.append((pad_to_full(latents[ei][i][:n]),
                                   pad_to_full(np.ones(n))))
                    else:
                        rx.append(received_latents(
                            modem, latents[ei][i][:n], spec, snr_db, fading,
                            args.seed + i))
                for di in range(2):
                    for i in range(len(images)):
                        if rx[i] is None:
                            continue  # sync loss; stays NaN in every cell
                        out = reconstruct(codecs[di], *rx[i])
                        cell[ei][di][i] = psnr(images[i], out)

            for cls in [*classes, "all"]:
                sel = (np.ones(len(images), bool) if cls == "all"
                       else img_class == cls)
                aa, ba = cell[0][0][sel], cell[1][0][sel]
                ab, bb = cell[0][1][sel], cell[1][1][sel]

                d = decompose(aa, ba, ab, bb)
                total, total_sem, n_ok = paired(d["total"])
                enc, enc_sem, _ = paired(d["encoder"])
                dec, dec_sem, _ = paired(d["decoder"])
                inter, inter_sem, _ = paired(d["interaction"])
                cross, _, _ = paired(d["cross_penalty"])

                rows.append({
                    "mode": spec.name, "condition": label, "content": cls,
                    "psnr_AA": np.nanmean(aa), "psnr_BA": np.nanmean(ba),
                    "psnr_AB": np.nanmean(ab), "psnr_BB": np.nanmean(bb),
                    "total": total, "total_sem": total_sem,
                    "encoder": enc, "encoder_sem": enc_sem,
                    "decoder": dec, "decoder_sem": dec_sem,
                    "interaction": inter, "interaction_sem": inter_sem,
                    "cross_penalty": cross,
                    "n": n_ok, "n_images": int(np.count_nonzero(sel)),
                })
                print(
                    f"  mode {spec.name} {label:>10} {cls:>9}: "
                    f"total {total:+5.2f}  enc {enc:+5.2f}  dec {dec:+5.2f}  "
                    f"interaction {inter:+5.2f}  cross {cross:+5.2f}  "
                    f"n {n_ok}/{int(np.count_nonzero(sel))}",
                    file=sys.stderr, flush=True,
                )

    print(f"\nA = {args.model_a}\nB = {args.model_b}\n")
    print("PSNR dB. total = B/B - A/A, split into the share attributable to\n"
          "each half. total = encoder + decoder exactly (an identity, not a fit).\n"
          "+/- are paired standard errors over images.\n")
    print("| Mode | Condition | Content | total | encoder | decoder | "
          "interaction | cross | n |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(
            f"| {r['mode']} | {r['condition']} | {r['content']} | "
            f"{r['total']:+.2f} ±{r['total_sem']:.2f} | "
            f"{r['encoder']:+.2f} ±{r['encoder_sem']:.2f} | "
            f"{r['decoder']:+.2f} ±{r['decoder_sem']:.2f} | "
            f"{r['interaction']:+.2f} | {r['cross_penalty']:+.2f} | "
            f"{r['n']}/{r['n_images']} |"
        )

    overall = [r for r in rows if r["content"] == "all"]
    tot = float(np.nanmean([r["total"] for r in overall]))
    enc = float(np.nanmean([r["encoder"] for r in overall]))
    dec = float(np.nanmean([r["decoder"] for r in overall]))
    inter = float(np.nanmean([r["interaction"] for r in overall]))
    print(f"\nOver {len(overall)} (mode, condition) cells, all content pooled:")
    print(f"  total       {tot:+.3f} dB")
    if abs(tot) > 1e-9:
        print(f"  encoder     {enc:+.3f} dB  ({100 * enc / tot:.0f}% of total)")
        print(f"  decoder     {dec:+.3f} dB  ({100 * dec / tot:.0f}% of total)")
    else:
        print(f"  encoder     {enc:+.3f} dB\n  decoder     {dec:+.3f} dB")
    print(f"  interaction {inter:+.3f} dB")

    # The split is only as meaningful as the mixed cells are legitimate.
    if abs(tot) > 1e-9 and abs(inter) > INTERACTION_WARN * abs(tot):
        print(
            f"\n  NOTE: |interaction| is {abs(inter) / abs(tot):.1f}x the total.\n"
            "  The encoder's contribution depends strongly on which decoder it\n"
            "  is paired with, which is what co-adapted halves look like when\n"
            "  you mix them. Report this change as joint rather than quoting\n"
            "  the encoder/decoder split.",
        )

    if args.csv:
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {args.csv}", file=sys.stderr)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Decode a QRSSTVAE picture from the multi-pass store (or one pass) to a PNG.

    python qrss_decode.py out.png --key K1ABC:1a2b3c4d
    python qrss_decode.py out.png --key K1ABC:1a2b3c4d --em 3
    python qrss_decode.py out.png --pass ~/.local/share/qrsstvae/passes/<uid>.npz
    python qrss_decode.py --list

--key CALL:PID renders the accumulator of that callsign and picture ID
(hexadecimal) in the store (--store DIR, default $QRSSTVAE_HOME or
~/.local/share/qrsstvae). --em N first runs N rounds of leave-one-out EM
over its passes (each pass re-tracked from its stored capture with the
others as its reference; `sstvae.qrss.em.em_refine`) and stores the
improved passes. --pass renders a single stored pass, which needs a
decoded header to place its latents. --list prints the store's
accumulators and exits.

Prints, per segment of the picture, the effective SNR (10 log10 median
W) and the members, then the mean W before and after each EM round.
The codec is the published v5 decoder (--model DIR, --precision).
"""

import argparse
import sys
from pathlib import Path

import numpy as np


def parse_key(s: str) -> tuple[str, int]:
    """("K1ABC", 0x1a2b3c4d) from "K1ABC:1a2b3c4d"."""
    call, sep, pid = s.rpartition(":")
    if not sep or not call:
        raise SystemExit(f"--key {s!r}: expected CALL:PICTUREID (hexadecimal)")
    try:
        return call.upper(), int(pid, 16)
    except ValueError:
        raise SystemExit(f"--key {s!r}: picture ID {pid!r} is not hexadecimal") from None


def acc_lines(acc) -> list[str]:
    from sstvae.qrss import em, picture

    out = [f"picture {acc.key[0]} {acc.key[1]:08x}  mode {picture.MODE_NAMES[acc.mode]}  "
           f"codec {acc.codec_id:04x}  {len(acc.members)} passes"
           + (f", {len(acc.foreign)} merged" if acc.foreign else "")
           + f"  mean W {em.acc_mean_w_db(acc):+.2f} dB"]
    for g, snr in enumerate(acc.seg_snr_db()):
        out.append(f"  segment {g}: " + ("no data" if snr is None else
                                         f"effective SNR {snr:+.2f} dB (median W)"))
    for m in acc.members:
        out.append(f"    {m['uid']}  segment {m['segment']}  by {m['method']}  "
                   f"Z_ref {m['z_ref']:.1f}  mean W {m['mean_w_db']:+.2f} dB  "
                   f"EM round {m['em_round']}")
    return out


def planes_from_pass(p) -> tuple[np.ndarray, np.ndarray, int]:
    """(S, W, mode) of one pass placed at its segment's canonical offsets."""
    from sstvae.qrss import picture
    from sstvae.qrss.store import canonical_index

    h = p.header
    if h is None:
        raise SystemExit("the pass has no decoded header: its latents cannot be placed "
                         "(store it and decode the accumulator instead)")
    S = np.zeros((picture.N_GROUPS, picture.GROUP_LATENTS), dtype=np.float64)
    W = np.zeros_like(S)
    idx = canonical_index(h.segment, len(p.z))
    w = np.asarray(p.w, dtype=np.float64)
    S[h.segment, idx] = w * np.asarray(p.z, dtype=np.float64)
    W[h.segment, idx] = w
    return S, W, h.mode


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("output", nargs="?", type=Path, help="PNG to write")
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--key", default=None, help="CALL:PICTUREID (hex) of an accumulator")
    src.add_argument("--pass", dest="pass_file", type=Path, default=None,
                     help="a single stored pass (.npz)")
    src.add_argument("--list", action="store_true", help="list the store's pictures")
    ap.add_argument("--store", default=None, help="multi-pass store directory")
    ap.add_argument("--em", type=int, default=0, metavar="ROUNDS",
                    help="leave-one-out EM rounds before decoding (default 0)")
    ap.add_argument("--model", default=None, help="codec model directory")
    ap.add_argument("--precision", choices=("fp32", "fp16", "int8"), default="fp32",
                    help="codec precision")
    args = ap.parse_args()

    from sstvae.qrss import em, render
    from sstvae.qrss.store import Store

    if args.list:
        store = Store(args.store)
        keys = store.open_keys()
        if not keys:
            print("no pictures in the store")
        for key in keys:
            for line in acc_lines(store.accumulator(key)):
                print(line)
        return
    if args.output is None or (args.key is None and args.pass_file is None):
        ap.error("give OUT.png and one of --key or --pass (or --list)")

    if args.pass_file is not None:
        from sstvae.qrss.types import PassResult

        if args.em:
            print("--em needs an accumulator (--key); ignored for a single pass")
        p = PassResult.load(args.pass_file)
        S, W, mode = planes_from_pass(p)
        print(f"pass {p.uid}: {p.frame} frame, mean W "
              f"{10 * np.log10(max(float(np.mean(p.w)), 1e-30)):+.2f} dB")
    else:
        store = Store(args.store)
        key = parse_key(args.key)
        if not store.has_accumulator(key):
            sys.exit(f"no picture {key[0]} {key[1]:08x} in {store.root}")
        if args.em > 0:
            hist = em.em_refine(store, key, rounds=args.em, log=print)
            print("EM: mean W " + " -> ".join(f"{h:+.2f}" for h in hist) + " dB (as claimed: "
                  "a re-receive raises W by up to about twice its true gain)")
        acc = store.accumulator(key)
        for line in acc_lines(acc):
            print(line)
        S, W, mode = acc.S, acc.W, acc.mode
    codec = render.load(args.precision, args.model)
    render.render(codec, S, W, mode).save(args.output)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()

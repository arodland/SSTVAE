#!/usr/bin/env python3
"""End-to-end confirmation of an LR profile, through the real modem.

`latent_optim_lr_sweep.py` ranks profiles on the *objective*, which is
cheap and which `docs/latent-optimization.md` records as overstating
recovered PSNR by roughly 3x -- and as having once produced a design
decision that this measurement overturned. So a profile that wins the
sweep is a candidate, not a default. This asks the radio.

encode -> optimize -> modulate -> HF channel -> demodulate -> decode,
PSNR of the picture that comes out against the one that went in. Every
config goes through the **same channel realizations** (seeds 0..N-1), so
a difference is between latents rather than between noise draws.

Two departures from `latent_optim_roundtrip.py`, which this borrows its
channel evaluation from rather than reimplementing:

- it drives `sstvae.latent_optim.optimize`, the **shipped** ONNX loop,
  rather than the torch prototype -- the LR profile is a property of the
  code that would change, so testing the other implementation would
  prove the wrong thing; and
- it sweeps *step counts* rather than objective SNR, because the open
  question is which profile is best at the budget a real station gets,
  and that budget is unknown. Early stopping is off so the step count is
  the independent variable.

    python scripts/latent_optim_lr_roundtrip.py \\
        data/optim_corpus/wonder_wheel.jpg data/optim_corpus/w0nycert.png
"""

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from latent_optim_roundtrip import run_cell  # noqa: E402

from sstvae import latent_optim  # noqa: E402
from sstvae.codec import load_codec, load_torch_model  # noqa: E402
from sstvae.config import MODES  # noqa: E402
from sstvae.images import fit_image, image_to_array  # noqa: E402
from sstvae.modem import Modem, dsp  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "_lrsweep", Path(__file__).resolve().parent / "latent_optim_lr_sweep.py")
_sweep = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_sweep)
PROFILES = _sweep.PROFILES


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("images", nargs="+")
    ap.add_argument("--model", default=None)
    ap.add_argument("--mode", choices=sorted(MODES), default="B")
    ap.add_argument("--profiles", nargs="+",
                    default=["const-0.02", "const-0.05", "warm2-0.10"])
    ap.add_argument("--steps", type=int, nargs="+", default=[5, 20, 50])
    ap.add_argument("--snr", type=float, nargs="+", default=[3.0, 9.0])
    ap.add_argument("--fading", nargs="+", default=["none", "mpp"])
    ap.add_argument("--seeds", type=int, default=25)
    ap.add_argument("--callsign", default="")
    args = ap.parse_args()

    model = load_torch_model(args.model)
    codec = load_codec(args.model)
    spec = MODES[args.mode]
    modem = Modem()
    cells = [(f, s) for f in args.fading for s in args.snr]

    def evaluate(wave, target):
        out, fails = {}, {}
        for fading, snr in cells:
            fp = None if fading == "none" else fading
            vals, bad = [], 0
            for seed in range(args.seeds):
                r = run_cell(model, wave, target, snr, fp, seed)
                if r["ok"]:
                    vals.append(r["psnr"])
                else:
                    bad += 1
            out[(fading, snr)] = float(np.mean(vals)) if vals else float("nan")
            fails[(fading, snr)] = bad
        return out, fails

    for path in args.images:
        name = Path(path).stem
        target = image_to_array(fit_image(Image.open(path)))
        enc = codec.encode(target)
        print(f"\n{'=' * 78}\n{name}  (mode {spec.name}, {args.seeds} seeds, "
              f"{len(cells)} channels)\n{'=' * 78}")

        base_wave = modem.modulate(enc[: spec.n_latents].astype(np.float32),
                                   spec, callsign=args.callsign)
        base, base_fails = evaluate(base_wave, target)
        base_papr = dsp.papr_db(base_wave)
        print(f"\n  encoder baseline (PAPR {base_papr:.2f} dB)")
        print("   " + "  ".join(f"{f}@{s:g}".rjust(9) for f, s in cells))
        print("   " + "  ".join(f"{base[c]:9.2f}" for c in cells)
              + "   absolute PSNR, dB")
        if any(base_fails.values()):
            print("   " + "  ".join(f"{base_fails[c] or '':>9}" for c in cells)
                  + "   no-lock")

        print(f"\n  delta vs encoder latents:")
        print("   " + "profile".ljust(13) + "steps".rjust(6) + "   "
              + "  ".join(f"{f}@{s:g}".rjust(9) for f, s in cells)
              + "     mean    PAPR")
        for prof in args.profiles:
            for n in args.steps:
                # +1: the objective at a step belongs to the latents
                # that went into it, so N updates need N+1 iterates for
                # the last to be evaluated. Plateau and clock off.
                r = latent_optim.optimize(
                    enc, target, args.mode, model=args.model,
                    max_steps=n + 1, patience=10**9, time_budget_s=1e9,
                    lr=PROFILES[prof])
                wave = modem.modulate(r.latents, spec,
                                      callsign=args.callsign)
                got, fails = evaluate(wave, target)
                d = [got[c] - base[c] for c in cells]
                print("   " + prof.ljust(13) + str(n).rjust(6) + "   "
                      + "  ".join(f"{x:+9.2f}" for x in d)
                      + f"  {np.nanmean(d):+7.2f}"
                      + f"  {dsp.papr_db(wave) - base_papr:+6.2f}"
                      + ("   (no-lock: %d)" % sum(fails.values())
                         if any(fails.values()) else ""))


if __name__ == "__main__":
    main()

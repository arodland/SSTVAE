#!/usr/bin/env python3
"""Apply a simulated HF channel to a transmit WAV file.

Example:
    python sstvae_simulate.py tx.wav rx.wav --snr 3 --fading mpp --freq-offset 43

--protect-sync keeps the preamble and header clean (spliced back in
from the original), so acquisition and header decode always succeed
regardless of how extreme --snr/--fading are. Per-frame pilots are
deliberately corrupted along with their frame's data (not protected) —
equalization needs both to see the same channel, so a clean pilot next
to corrupted data would make it apply a confidently wrong correction
rather than show real degradation. Requires the input to be exactly
what sstvae_encode.py produced (unchanged timing/length); --mode
overrides auto-detection if needed.
"""

import argparse

from sstvae import hfchannel, wavio
from sstvae.config import MODES


def parse_span(s: str) -> tuple[float, float]:
    a, b = s.split(":")
    return float(a), float(b)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input", help="input WAV (transmit audio)")
    ap.add_argument("output", help="output WAV (received audio)")
    ap.add_argument("--snr", type=float, default=None, help="SNR dB in 2500 Hz")
    ap.add_argument("--freq-offset", type=float, default=0.0, help="Hz")
    ap.add_argument("--ppm", type=float, default=0.0, help="sample clock error")
    ap.add_argument(
        "--fading", choices=sorted(hfchannel.FADING_PRESETS), default=None
    )
    ap.add_argument(
        "--zero-span",
        action="append",
        type=parse_span,
        default=[],
        metavar="START:END",
        help="blank a time span in seconds (repeatable)",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--protect-sync",
        action="store_true",
        help="only corrupt image data, keeping preamble/header/pilots "
        "clean so sync always succeeds (see module docstring); "
        "incompatible with --ppm",
    )
    ap.add_argument(
        "--mode",
        choices=sorted(MODES),
        default=None,
        help="override auto-detected mode for --protect-sync",
    )
    args = ap.parse_args()

    x = wavio.read_wav(args.input)
    if args.protect_sync:
        if args.ppm:
            ap.error("--protect-sync doesn't support --ppm (see docstring)")
        mode = MODES[args.mode] if args.mode else None
        y = hfchannel.apply_channel_data_only(
            x,
            mode=mode,
            snr_db=args.snr,
            freq_offset_hz=args.freq_offset,
            fading_preset=args.fading,
            spans=args.zero_span,
            seed=args.seed,
        )
    else:
        y = hfchannel.apply_channel(
            x,
            snr_db=args.snr,
            freq_offset_hz=args.freq_offset,
            ppm=args.ppm,
            fading_preset=args.fading,
            spans=args.zero_span,
            seed=args.seed,
        )
    wavio.write_wav(args.output, y)
    print(f"wrote {args.output} ({len(y)} samples)")


if __name__ == "__main__":
    main()

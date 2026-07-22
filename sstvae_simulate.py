#!/usr/bin/env python3
"""Apply a simulated HF channel to a transmit WAV file.

Example:
    python sstvae_simulate.py tx.wav rx.wav --snr 3 --fading mpp --freq-offset 43
"""

import argparse

from sstvae import hfchannel, wavio


def parse_span(s: str) -> tuple[float, float]:
    a, b = s.split(":")
    return float(a), float(b)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input", help="input WAV (transmit audio)")
    ap.add_argument("output", help="output WAV (received audio)")
    ap.add_argument("--snr", type=float, default=None, help="SNR dB in 3000 Hz")
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
    args = ap.parse_args()

    x = wavio.read_wav(args.input)
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

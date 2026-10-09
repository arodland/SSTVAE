#!/usr/bin/env python3
"""Write one QRSSTVAE CE pass as transmit audio (8 kHz WAV).

    python qrss_transmit.py seg0.bin tx.wav --slot 2026-10-09T06:00Z
    python qrss_transmit.py pic.qrsp tx.wav --slot 2026-10-09T06:00Z --callsign K1ABC

The WAV starts 12 s before t0 = slot + 1 s (11 s before the quarter
hour) and ends 3 s after the frame. The slot fixes the scrambler, so a
WAV is for that quarter hour only. Every pass keys the callsign windows
(a legal requirement), so a .qrsp needs --callsign. --lead-in sends up to
10 s of plain carrier before the preamble, for transmitters that drift
when keyed. --float writes float32 with no normalisation (exact round
trips); otherwise the file is peak-normalised int16.
"""

import argparse

from sstvae import wavio
from sstvae.qrss import beaconfile, frame, picture, sequences, tx
from sstvae.qrss.constants import LEAD_IN_MAX_S
from sstvae.qrss.header import HeaderFields


def parse_slot(s: str) -> int:
    """q for an ISO 8601 UTC time on a quarter hour, e.g. 2026-10-09T06:00Z."""
    try:
        return sequences.parse_slot(s)
    except ValueError as e:
        raise SystemExit(f"--slot {s!r}: {e}") from None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input", help="stored picture (.qrsp) or beacon file (.bin)")
    ap.add_argument("output", help="WAV file")
    ap.add_argument("--slot", required=True, help="quarter hour, ISO UTC, e.g. 2026-10-09T06:00Z")
    ap.add_argument("--callsign", default=None, help="for a .qrsp: 1-8 of A-Z, 0-9, /")
    ap.add_argument("--grid", default=None, help="for a .qrsp: 4-character locator")
    ap.add_argument("--segment", type=int, default=0, help="for a .qrsp")
    ap.add_argument("--freq", type=float, default=1500.0, help="audio carrier, Hz")
    ap.add_argument("--lead-in", type=float, default=0.0,
                    help=f"seconds of plain carrier before the preamble (0-{LEAD_IN_MAX_S})")
    ap.add_argument("--ook", action="store_true",
                    help="on-off keyed callsign windows for a .qrsp (a .bin carries its own "
                         "choice: --ook with an FSK .bin is refused)")
    ap.add_argument("--frame", choices=sorted(frame.PRESETS), default="full",
                    help="frame shape; anything but full is a test frame")
    ap.add_argument("--float", action="store_true", help="float32 WAV, not normalised")
    args = ap.parse_args()

    q = parse_slot(args.slot)
    spec = frame.get(args.frame)
    header = None
    try:
        with open(args.input, "rb") as f:
            is_bin = f.read(4) == beaconfile.MAGIC
        if is_bin:
            src = beaconfile.read(args.input)
            if args.callsign is not None and args.callsign != src.callsign:
                raise SystemExit(f"the beacon file is for {src.callsign}, not {args.callsign}")
            if args.ook and not src.ook:
                raise SystemExit("the beacon file keys its callsign windows FSK, and a beacon "
                                 "sends what its file says: make the .bin with "
                                 "qrss_beacon.py --ook instead of passing --ook here")
            ook = None
        else:
            src = picture.load_qrsp(args.input)
            if args.callsign is None:
                raise SystemExit("a .qrsp needs --callsign: every pass sends it in the "
                                 "header and in the callsign windows")
            header = HeaderFields(callsign=args.callsign, grid=args.grid,
                                  picture_id=src.picture_id, mode=src.mode,
                                  segment=args.segment, codec_id=src.codec_id)
            ook = args.ook
        x = tx.transmit_audio(src, q, args.freq, spec=spec, segment=None if is_bin
                              else args.segment, header=header, lead_in_s=args.lead_in,
                              ook=ook)
    except ValueError as e:
        raise SystemExit(str(e)) from None
    (wavio.write_wav_float if args.float else wavio.write_wav)(args.output, x)
    call = src.callsign if is_bin else args.callsign
    keyed = (src.ook if is_bin else args.ook)
    lead = f", {args.lead_in:g} s lead-in" if args.lead_in else ""
    print(f"wrote {args.output}: slot q={q}, frame {spec.name} "
          f"({spec.duration_s:.1f} s), {call} ({'OOK' if keyed else 'FSK'} callsign windows), "
          f"carrier {args.freq:g} Hz{lead}, "
          f"{len(x) / 8000:.1f} s of audio starting 12 s before t0")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Write a beacon file for a clock-chip (Si5351) QRSSTVAE beacon.

    python qrss_beacon.py pic.qrsp seg0.bin --callsign K1ABC --grid FN42

One segment of a stored picture as the beacon sends it: the header's
coded bits, the callsign's Morse keying, and the latents in air order as
int8 (x20). A full segment is 50,970 bytes, which fits a 24LC512 EEPROM.
The beacon computes the preamble, scrambler, precoder and phase itself.
"""

import argparse

from sstvae.qrss import beaconfile, picture
from sstvae.qrss.header import HeaderFields


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input", help="stored picture (.qrsp)")
    ap.add_argument("output", help="beacon file (.bin)")
    ap.add_argument("--callsign", required=True,
                    help="1-8 characters of A-Z, 0-9 and /; sent in the header and "
                    "in Morse in every callsign window")
    ap.add_argument("--grid", default=None, help="4-character Maidenhead locator")
    ap.add_argument("--segment", type=int, default=0)
    ap.add_argument("--ook", action="store_true",
                    help="identify by on-off keying instead of frequency-shift keying")
    args = ap.parse_args()

    sp = picture.load_qrsp(args.input)
    try:
        h = HeaderFields(callsign=args.callsign, grid=args.grid, picture_id=sp.picture_id,
                         mode=sp.mode, segment=args.segment, codec_id=sp.codec_id)
        bf = beaconfile.from_picture(sp, args.segment, h, ook=args.ook)
    except ValueError as e:
        raise SystemExit(str(e)) from None
    beaconfile.write(args.output, bf)
    print(f"wrote {args.output}: picture {sp.id_hex} segment {args.segment}, "
          f"{bf.callsign} {bf.grid or '-'}, {'OOK' if bf.ook else 'FSK'} ID, "
          f"{beaconfile.file_size(len(bf.latents_i8))} bytes")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Pass one QRSSTVAE CE slot through a simulated HF channel.

    python qrss_simulate.py seg0.bin rx.wav --slot 2026-10-09T06:00Z --snr -12 --preset quiet
    python qrss_simulate.py pic.qrsp rx.npz --slot 2026-10-09T06:00Z --callsign K1ABC \\
        --snr -20 --drift 1 --wander 0.1 --tx-ppm 100 --rx-ppm -100 --agc
    python qrss_simulate.py tx.wav rx.wav --slot 2026-10-09T06:00Z --snr -15 --impulse-rate 10

A .qrsp or .bin input is synthesised directly at the receiver's front-end
rate (4 kHz complex about 1500 Hz), callsign windows included; a WAV
input (as `qrss_transmit.py` writes it, starting --pre seconds before
t0) is converted to the front end, impaired and converted back. A .wav
output is 8 kHz audio starting --pre s before t0 on the receiver's
clock; an .npz output holds the front-end samples (`fe`, `fs`,
`t0_index`, `q`, `frame`) and the channel's truth arrays, for scoring a
receiver (`sstvae.qrss.channel.load_npz`).

SNR is in 2500 Hz on the wanted signal's keyed samples. Presets set the
Watterson Doppler spread and path delay: steady, quiet (0.1 Hz, 0.5 ms),
moderate (0.5, 1), disturbed (1, 2), mps (0.15, 2). --neighbour DF:DB
adds a CE station DF Hz away at DB relative to this one (repeatable;
DF:DB:LEADIN:DRIFT for its lead-in seconds and drift in Hz/min).
"""

import argparse
from datetime import datetime

from sstvae import wavio
from sstvae.qrss import beaconfile, channel, frame, picture, tx
from sstvae.qrss.channel import Agc, ChannelConfig, Impulses, Neighbour
from sstvae.qrss.constants import LEAD_IN_MAX_S
from sstvae.qrss.header import HeaderFields
from sstvae.qrss.sequences import quarter_hour_count


def parse_slot(s: str) -> int:
    """q for an ISO 8601 UTC time on a quarter hour, e.g. 2026-10-09T06:00Z."""
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        return quarter_hour_count(dt)
    except ValueError as e:
        raise SystemExit(f"--slot {s!r}: {e}") from None


def parse_neighbour(s: str) -> Neighbour:
    """DF:DB[:LEADIN[:DRIFT]] -> Neighbour (seeded by its position on the line)."""
    parts = [float(p) for p in s.split(":")]
    if not 2 <= len(parts) <= 4:
        raise argparse.ArgumentTypeError(f"--neighbour {s!r}: want DF:DB[:LEADIN[:DRIFT]]")
    return Neighbour(*parts[:2], *parts[2:])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="stored picture (.qrsp), beacon file (.bin) or transmit WAV")
    ap.add_argument("output", help="received WAV, or .npz with the front end and truth")
    ap.add_argument("--slot", required=True, help="quarter hour, ISO UTC, e.g. 2026-10-09T06:00Z")
    ap.add_argument("--callsign", default=None, help="for a .qrsp: 1-8 of A-Z, 0-9, /")
    ap.add_argument("--grid", default=None, help="for a .qrsp: 4-character locator")
    ap.add_argument("--segment", type=int, default=0, help="for a .qrsp")
    ap.add_argument("--freq", type=float, default=1500.0, help="audio carrier, Hz")
    ap.add_argument("--frame", choices=sorted(frame.PRESETS), default="full",
                    help="frame shape; anything but full is a test frame")
    ap.add_argument("--lead-in", type=float, default=0.0,
                    help=f"seconds of plain carrier before the preamble (0-{LEAD_IN_MAX_S})")
    ap.add_argument("--ook", action="store_true",
                    help="on-off keyed callsign windows (a .bin carries its own choice)")
    ap.add_argument("--snr", type=float, default=None, help="SNR in 2500 Hz, dB (default: none)")
    ap.add_argument("--preset", choices=sorted(channel.PRESETS), default=None)
    ap.add_argument("--doppler", type=float, default=0.0, help="Doppler spread (2 sigma), Hz")
    ap.add_argument("--delay", type=float, default=0.0, help="second path delay, ms")
    ap.add_argument("--offset", type=float, default=0.0, help="frequency offset, Hz")
    ap.add_argument("--drift", type=float, default=0.0, help="drift, Hz/min")
    ap.add_argument("--drift-tau", type=float, default=None,
                    help="warm-up settling time, s (drift is then the initial rate)")
    ap.add_argument("--wander", type=float, default=0.0, help="OU frequency wander, Hz rms")
    ap.add_argument("--wander-tau", type=float, default=10.0, help="wander correlation time, s")
    ap.add_argument("--tx-ppm", type=float, default=0.0, help="transmit clock error")
    ap.add_argument("--rx-ppm", type=float, default=0.0, help="receive clock error")
    ap.add_argument("--impulse-rate", type=float, default=0.0, help="clicks per second")
    ap.add_argument("--impulse-db", type=float, default=30.0, help="click level over the signal")
    ap.add_argument("--crashes", type=float, default=0.0, help="static crashes per minute")
    ap.add_argument("--agc", action="store_true", help="fast receive AGC (2 ms / 200 ms)")
    ap.add_argument("--neighbour", action="append", type=parse_neighbour, default=[],
                    metavar="DF:DB", help="a CE neighbour (repeatable)")
    ap.add_argument("--pre", type=float, default=12.0, help="seconds before t0 in the output")
    ap.add_argument("--post", type=float, default=3.0, help="seconds after the frame")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--float", action="store_true", help="float32 WAV, not normalised")
    args = ap.parse_args()

    q = parse_slot(args.slot)
    spec = frame.get(args.frame)
    impulses = None
    if args.impulse_rate > 0 or args.crashes > 0:
        impulses = Impulses(rate_hz=args.impulse_rate, amp_db=args.impulse_db,
                            crash_per_min=args.crashes)
    neighbours = tuple(Neighbour(nb.df_hz, nb.rel_db, nb.lead_in_s, nb.drift_hz_per_min,
                                 seed=args.seed * 1000 + i + 1)
                       for i, nb in enumerate(args.neighbour))
    cfg = ChannelConfig(
        snr_db=args.snr, preset=args.preset, doppler_hz=args.doppler, delay_ms=args.delay,
        freq_offset_hz=args.offset, drift_hz_per_min=args.drift, drift_tau_s=args.drift_tau,
        wander_rms_hz=args.wander, wander_tau_s=args.wander_tau, tx_ppm=args.tx_ppm,
        rx_ppm=args.rx_ppm, impulses=impulses, agc=Agc() if args.agc else None,
        neighbours=neighbours, seed=args.seed)

    try:
        with open(args.input, "rb") as f:
            magic = f.read(4)
        if magic == b"RIFF":
            x = wavio.read_wav(args.input)
            sim = channel.simulate_audio(x, spec, q, cfg, carrier_hz=args.freq,
                                         lead_in_s=args.lead_in, pre_s=args.pre)
            what = "transmit audio"
        else:
            if magic == beaconfile.MAGIC:
                src = beaconfile.read(args.input)
                sym = tx.slot_symbols(src, q, spec)
                keying = tx.slot_keying(src)
                ook = True if args.ook else src.ook
                what = f"beacon file, {src.callsign}"
            else:
                src = picture.load_qrsp(args.input)
                if args.callsign is None:
                    raise SystemExit("a .qrsp needs --callsign: every pass sends it in the "
                                     "header and in the callsign windows")
                header = HeaderFields(callsign=args.callsign, grid=args.grid,
                                      picture_id=src.picture_id, mode=src.mode,
                                      segment=args.segment, codec_id=src.codec_id)
                sym = tx.slot_symbols(src, q, spec, segment=args.segment, header=header)
                keying = tx.slot_keying(src, header)
                ook = args.ook
                what = f"stored picture, {args.callsign}"
            sim = channel.simulate(sym, spec, q, cfg, carrier_hz=args.freq,
                                   lead_in_s=args.lead_in, keying=keying, ook=ook,
                                   pre_s=args.pre, post_s=args.post)
    except ValueError as e:
        raise SystemExit(str(e)) from None

    if args.output.lower().endswith(".npz"):
        channel.save_npz(args.output, sim)
    else:
        y = channel.fe_to_audio(sim.fe)
        (wavio.write_wav_float if args.float else wavio.write_wav)(args.output, y)
    tr = sim.truth
    snr = "none" if args.snr is None else f"{args.snr:g} dB"
    print(f"wrote {args.output}: slot q={q}, frame {spec.name}, from {what}; "
          f"{len(sim.fe) / sim.fs:.1f} s, SNR {snr}, signal power {tr.signal_power:.3g}, "
          f"carrier {tr.f_hz[0]:.2f} -> {tr.f_hz[-1]:.2f} Hz, "
          f"clock x{tr.time_scale:.7f}")


if __name__ == "__main__":
    main()

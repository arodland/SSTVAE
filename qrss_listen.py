#!/usr/bin/env python3
"""Listen for QRSSTVAE live: every CE signal in the passband becomes a tile
whose picture fills in as its pass arrives.

    arecord -q -f S16_LE -r 8000 -c 1 -t raw | python qrss_listen.py --format s16
    parec --rate 8000 --channels 1 --format float32le | python qrss_listen.py
    python qrss_listen.py --wav rx.wav --start 2026-10-09T05:59:49Z    # replay

Audio is raw mono 8 kHz on stdin (--format f32, the default, or s16,
little-endian), stamped with the computer's clock as it arrives, so the
clock must be right to a second or so (NTP). The desktop app's QRSS
window starts this with its own capture audio piped in. --wav replays a
recording instead (from --start, default 12 s before the slot given by
--slot, as `qrss_transmit.py` writes it), as fast as the receiver goes
unless --speed is given.

Tiles go to --state DIR (default STORE/live): `state.json`, rewritten
after every change, and `tiles/<id>.png`. A signal is looked for on its
preamble (every --refresh seconds until 5 minutes into the slot), then
received every --refresh seconds with what has arrived so far; when its
frame ends the whole slot is received again (round B, and the whole-slot
search for signals too weak to see on the preamble), stored and
associated in the multi-pass store (--store DIR, default $QRSSTVAE_HOME
or ~/.local/share/qrsstvae; off with --no-store), and the tile shows the
picture accumulated over every pass of it. The front-end stream also
goes to the 48 h passband store (--passband DIR, default STORE/passband;
off with --no-passband or --no-store) for `qrss_decode.py --retro`.

A live receive of a FULL slot costs roughly 10-30 s of CPU per signal on
a desktop; slow machines want a longer --refresh.
"""

import argparse
import sys
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--format", choices=("f32", "s16"), default="f32",
                    help="stdin sample format (default f32)")
    ap.add_argument("--wav", type=Path, default=None, help="replay this 8 kHz recording")
    ap.add_argument("--start", default=None, help="UTC time of the replay's first sample")
    ap.add_argument("--slot", default=None,
                    help="with --wav and no --start: the replay starts 12 s before this slot")
    ap.add_argument("--speed", type=float, default=0.0,
                    help="replay at this many times real time (default: as fast as possible)")
    ap.add_argument("--frame", default="full", help="frame shape (default full)")
    ap.add_argument("--refresh", type=float, default=60.0,
                    help="seconds between live receives of a slot (default 60)")
    ap.add_argument("--state", type=Path, default=None, help="tile directory (default STORE/live)")
    ap.add_argument("--store", default=None, help="multi-pass store directory")
    ap.add_argument("--no-store", action="store_true", help="do not store or associate")
    ap.add_argument("--passband", default=None, help="48 h passband store directory")
    ap.add_argument("--no-passband", action="store_true", help="do not keep the front-end stream")
    ap.add_argument("--model", default=None, help="codec model directory")
    ap.add_argument("--precision", choices=("fp32", "fp16", "int8"), default="fp32",
                    help="codec precision (default fp32)")
    ap.add_argument("--no-pictures", action="store_true", help="tiles without pictures")
    args = ap.parse_args()

    from sstvae.qrss import frame, frontend, live, sequences
    from sstvae.qrss.store import Store, default_root

    try:
        spec = frame.get(args.frame)
    except (KeyError, ValueError):
        raise SystemExit(f"--frame {args.frame!r}: one of {', '.join(sorted(frame.PRESETS))}")
    if args.refresh <= 0:
        raise SystemExit("--refresh must be positive")
    root = Path(args.store) if args.store else default_root()
    state = args.state if args.state is not None else root / "live"
    # One listener per directory. A second one on the same tiles would
    # overwrite the first's state file, so it stops; on the same store it
    # runs without one (no combining with earlier passes, no passband).
    locks = [live.dir_lock(state)]
    if locks[0] is None:
        raise SystemExit(f"another listener is writing tiles to {state}; stop it, "
                         "or give this one its own --state DIR")
    store = None
    if not args.no_store:
        locks.append(live.dir_lock(root))
        if locks[-1] is None:
            print(f"another listener is using the store in {root}; this one runs "
                  "without a store (pass --store DIR for one of its own)",
                  file=sys.stderr, flush=True)
        else:
            store = Store(root)
    passband = None
    if store is not None and not args.no_passband:
        pb = Path(args.passband) if args.passband else store.root / "passband"
        locks.append(live.dir_lock(pb))
        if locks[-1] is None:
            print(f"another listener is writing the passband store in {pb}; "
                  "this one keeps none", file=sys.stderr, flush=True)
        else:
            passband = frontend.PassbandStore(pb)
    cfg = live.LiveConfig(spec=spec, refresh_s=args.refresh, precision=args.precision,
                          model=args.model, render=not args.no_pictures)
    listener = live.LiveListener(state, cfg, store=store, passband=passband)

    if args.wav is not None:
        if args.start:
            start = sequences.parse_utc(args.start).timestamp()
        elif args.slot:
            start = frontend.slot_t0(sequences.parse_slot(args.slot)) - 12.0
        else:
            raise SystemExit("--wav needs --start or --slot")
        if not args.wav.is_file():
            raise SystemExit(f"{args.wav}: no such file")
        source = live.WavSource(args.wav, start, speed=args.speed)
        realtime = False
    else:
        if sys.stdin.isatty():
            raise SystemExit("pipe 8 kHz mono audio into stdin (see --help), or use --wav")
        source = live.StdinSource(args.format)
        realtime = True
    print(f"tiles in {state}", file=sys.stderr, flush=True)
    try:
        live.run(listener, source, realtime=realtime)
    except KeyboardInterrupt:
        listener.log("stopped")
        listener.write_state(listening=False)


if __name__ == "__main__":
    main()

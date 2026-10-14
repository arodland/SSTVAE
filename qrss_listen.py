#!/usr/bin/env python3
"""Listen for QRSSTVAE live: every signal in the passband, of either
waveform, becomes a tile whose picture fills in as its pass arrives.

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

Both waveforms are listened for, CE and L (the linear one, for SSB
rigs), unless --just-ce or --just-l names one: --frame names the shape,
in either waveform's name for it (full or l-full), and each waveform
listens for its own frame of that shape, so a CE-only shape such as
full-v1 needs --just-ce. --just-ce is the listener as it was before L,
which knows nothing of L: an L signal in the slot can come out as a
weak, headerless CE tile. The desktop app starts this with no waveform
flag, so it hears both. Two receiver rules changed on 2026-10-12, the
noise floor near 300 and 2700 Hz and which candidates are verified
around a strong signal (`sstvae.qrss.rules`); --old-edge-noise and
--old-gating set them back, and with both --just-ce is the listener of
2026-10-11 bit for bit.

Tiles go to --state DIR (default STORE/live): `state.json`, rewritten
after every change, and `tiles/<id>.png`; every log line goes to
`listener.log` there too. With rig control the desktop
app writes the radio's dial there too (`dial_changes.jsonl`), and a slot
heard on more than one dial is received separately on each. A signal is looked for on its
preamble (every --refresh seconds until 5 minutes into the slot, then
every 5 minutes while none has been found), then received every
--refresh seconds (default 2 minutes) with what has arrived so far; when
its frame ends the whole slot is received again (round B, and the
whole-slot search for signals too weak to see on the preamble) in a
second process, so the other slots' tiles keep updating, stored and
associated in the multi-pass store (--store DIR, default $QRSSTVAE_HOME
or ~/.local/share/qrsstvae; off with --no-store), and the tile shows the
picture accumulated over every pass of it. The front-end stream also
goes to the 48 h passband store (--passband DIR, default STORE/passband;
off with --no-passband or --no-store) for `qrss_decode.py --retro`.
With the soundfile package installed (`uv pip install soundfile`; a
`uv sync` removes it again), each finished hour is packed into FLAC.

A live receive of a FULL slot costs roughly 10-30 s of CPU per signal on
a desktop; slow machines want a longer --refresh.
"""

import argparse
import os
import sys
import time
from pathlib import Path

# One thread for numpy's linear algebra. The receiver's arrays are too
# small for a thread pool to speed anything up, and an idle pool's
# threads spin, which is CPU spent for nothing on every core it starts.
# It has to be set before numpy is first imported; a value already in
# the environment wins.
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")


def _parser() -> argparse.ArgumentParser:
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
    ap.add_argument("--frame", default="full",
                    help="frame shape, either waveform's name for it: full or l-full "
                         "(default full)")
    wf = ap.add_mutually_exclusive_group()
    wf.add_argument("--just-ce", action="store_true",
                    help="receive waveform CE only (default: both)")
    wf.add_argument("--just-l", action="store_true",
                    help="receive waveform L only (default: both)")
    ap.add_argument("--refresh", type=float, default=120.0,
                    help="seconds between live receives of a slot (default 120)")
    ap.add_argument("--state", type=Path, default=None, help="tile directory (default STORE/live)")
    ap.add_argument("--store", default=None, help="multi-pass store directory")
    ap.add_argument("--no-store", action="store_true", help="do not store or associate")
    ap.add_argument("--passband", default=None, help="48 h passband store directory")
    ap.add_argument("--no-passband", action="store_true", help="do not keep the front-end stream")
    ap.add_argument("--recover-hours", type=float, default=3.0,
                    help="on start, receive from the passband store every slot that ended "
                         "this recently while no listener finished it (default 3; 0 turns "
                         "it off)")
    ap.add_argument("--model", default=None, help="codec model directory")
    ap.add_argument("--precision", choices=("fp32", "fp16", "int8"), default="fp32",
                    help="codec precision (default fp32)")
    ap.add_argument("--no-pictures", action="store_true", help="tiles without pictures")
    from sstvae.qrss import rules

    rules.add_arguments(ap)
    return ap


def build_config(argv=None):
    """(arguments, `live.LiveConfig`) for a command line, with nothing
    started: what a command line listens for, readable without a listener."""
    args = _parser().parse_args(argv)

    from sstvae.qrss import frame, lframe, live
    from sstvae.qrss.constants import WAVEFORM_CE, WAVEFORM_L

    try:                     # (an L name, as qrss_receive.py takes, names its CE twin)
        spec = lframe.twin(lframe.get_any(args.frame), WAVEFORM_CE)
    except (KeyError, ValueError):
        raise SystemExit(f"--frame {args.frame!r}: one of "
                         f"{', '.join(sorted(frame.PRESETS) + sorted(lframe.PRESETS))}")
    waveforms = ((WAVEFORM_CE,) if args.just_ce else (WAVEFORM_L,) if args.just_l
                 else (WAVEFORM_CE, WAVEFORM_L))
    if WAVEFORM_L in waveforms:
        try:
            lframe.twin(spec, WAVEFORM_L)
        except ValueError:
            raise SystemExit(f"--frame {args.frame!r} is CE only: waveform L has no frame of "
                             f"that shape (L frames: {', '.join(lframe.PRESETS)}); "
                             "add --just-ce to receive it")
    if args.refresh <= 0:
        raise SystemExit("--refresh must be positive")
    cfg = live.LiveConfig(spec=spec, waveforms=waveforms, refresh_s=args.refresh,
                          precision=args.precision, model=args.model,
                          render=not args.no_pictures, background=True, finish_process=True)
    return args, cfg


def main(argv=None) -> None:
    args, cfg = build_config(argv)

    from sstvae.qrss import rules

    rules.apply(args)        # before the end-of-slot process is spawned: it inherits them

    from sstvae.qrss import frontend, live, sequences
    from sstvae.qrss.store import Store, default_root

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
            passband = frontend.PassbandStore(pb, pack=True)
    listener = live.LiveListener(state, cfg, store=store, passband=passband)

    if args.wav is not None:
        # a bad time is the operator's typo, not a traceback (as qrss_receive.py)
        if args.start:
            try:
                start = sequences.parse_utc(args.start).timestamp()
            except ValueError as e:
                raise SystemExit(f"--start {args.start!r}: {e}") from None
        elif args.slot:
            try:
                start = frontend.slot_t0(sequences.parse_slot(args.slot)) - 12.0
            except ValueError as e:
                raise SystemExit(f"--slot {args.slot!r}: {e}") from None
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
        listener.backfill(time.time())
        listener.recover(time.time(), args.recover_hours)
    print(f"tiles in {state}", file=sys.stderr, flush=True)
    try:
        live.run(listener, source, realtime=realtime)
    except KeyboardInterrupt:
        listener.log("stopped")
        listener.write_state(listening=False)
    finally:
        listener.close()


if __name__ == "__main__":
    main()

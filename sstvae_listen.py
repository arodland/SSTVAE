#!/usr/bin/env python3
"""Continuously listen on an audio input device for an SSTVAE
transmission and decode it, including the case where listening starts
mid-transmission.

    python sstvae_listen.py --out-dir received
    python sstvae_listen.py --list-devices
    python sstvae_listen.py --device pulse --no-gui

Keeps a rolling buffer of the last --buffer-seconds of audio (long
enough to cover a full mode-C transmission) and repeatedly tries to
decode it, falling back to the preamble-free blind path so a
mid-transmission lock still recovers the frames that arrived before it.
The reception state machine itself lives in `sstvae/rx/engine.py`; this
is just its command-line front end. For a windowed version with rig
control, a waterfall, and transmit, use the desktop app (`native/`).

--low-cpu drops the blind fallback (and with it, retrospective
mid-stream decoding) in exchange for much lower idle CPU use: it only
ever looks for the preamble, restricts that search to the audio that's
newly arrived since the last poll (instead of rescanning the whole
buffer), and once the header locks it just sleeps until the whole
transmission has been captured and decodes it once, rather than
repeatedly re-decoding for progress updates.

Requires sounddevice (PortAudio): pip install -e .[listen]
"""

import argparse
import threading

import numpy as np

from sstvae.audio import list_devices, open_input_stream
from sstvae.checkpoint import PRECISIONS
from sstvae.codec import (  # noqa: F401  (re-exported)
    MODEL_HELP,
    PRECISION_HELP,
    load_codec,
    pad_to_full,
    reconstruct,
)
from sstvae.config import FS, MODES
from sstvae.rx import (  # noqa: F401
    Reception,
    RingBuffer,
    RxConfig,
    SaveToDirSink,
    SharedState,
    decode_loop,
    decode_loop_low_cpu,
    fmt_snr,
)


def _status_line(state: SharedState) -> str:
    with state.lock:
        status = state.status
        mode_name = state.mode_name
        frames_received = state.frames_received
        n_frames_expected = state.n_frames_expected
        progress_frac = state.progress_frac
        callsign = state.callsign
        snr_db = state.snr_db
        seconds_captured = state.seconds_captured
        saved_path = state.saved_path

    if status == "listening":
        return f"listening... ({seconds_captured:.0f}s captured)"
    if status == "receiving":
        if n_frames_expected is not None:
            line = (
                f"receiving mode {mode_name}: frame "
                f"{frames_received}/{n_frames_expected} ({100 * progress_frac:.0f}%)"
            )
        else:
            line = f"receiving (blind sync): {100 * progress_frac:.0f}% of latents"
        line += fmt_snr(snr_db)
        if callsign:
            line += f"  de {callsign}"
        return line
    return f"done -- saved {saved_path}" + fmt_snr(snr_db)


def run_gui(state: SharedState, stop_event: threading.Event):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(np.zeros((30, 40, 3)))
    ax.axis("off")
    title = ax.set_title("listening...")
    fig.canvas.mpl_connect("close_event", lambda e: stop_event.set())

    def update(_frame):
        with state.lock:
            img = state.image
        if img is not None:
            im.set_data(np.asarray(img))
            im.set_extent((0, img.width, img.height, 0))
        title.set_text(_status_line(state))
        if stop_event.is_set():
            plt.close(fig)
        return im, title

    # Kept referenced: a FuncAnimation that gets garbage collected stops
    # animating.
    _anim = FuncAnimation(fig, update, interval=500, cache_frame_data=False)
    plt.show()
    stop_event.set()


def run_console(state: SharedState, stop_event: threading.Event):
    last_status = None
    while not stop_event.is_set():
        line = _status_line(state)
        if line != last_status:
            print(line)
            last_status = line
        stop_event.wait(1.0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=None, help=MODEL_HELP)
    ap.add_argument("--precision", choices=PRECISIONS, default=None,
                    help=PRECISION_HELP)
    ap.add_argument("--out-dir", default="received", help="directory for saved images")
    ap.add_argument("--device", default=None, help="input device name/index (see --list-devices)")
    ap.add_argument(
        "--buffer-seconds", type=float, default=130.0,
        help="rolling audio buffer length; must exceed the longest mode duration "
        "(mode C is ~95s) with margin for retrospective decode",
    )
    ap.add_argument("--poll-interval", type=float, default=5.0, help="seconds between decode attempts")
    ap.add_argument(
        "--blind-search-seconds", type=float, default=MODES["C"].duration_s,
        help="a cap, not a fixed window: the blind CFO/timing search runs "
        "one decay timescale per mode, each capped at min(mode duration, "
        "this value), so audio older than roughly a mode's own duration "
        "decays out of that mode's search rather than being scanned fresh "
        "each poll (folding in only new audio is what makes this cheap "
        "regardless of how long it is). Default is above every mode's own "
        "duration, so nothing is capped by default -- lower it only to "
        "shrink below a mode's duration (e.g. for a quick test), since "
        "there is no reliability benefit to integrating past a mode's own "
        "length. Must exceed MIN_FRAMES_FOR_SYNC's ~10.5s with margin; the "
        "retrospective decode itself still covers the full buffer once "
        "locked, this only bounds how long stale audio keeps influencing "
        "acquisition.",
    )
    ap.add_argument(
        "--end-grace", type=float, default=8.0,
        help="seconds of no further progress (blind-sync case only, true length unknown) "
        "before a reception is considered finished",
    )
    ap.add_argument("--size", default=None, help="resize saved image, e.g. 320x240")
    ap.add_argument("--no-gui", action="store_true", help="print status instead of a matplotlib window")
    ap.add_argument("--once", action="store_true", help="exit after the first successful reception")
    ap.add_argument("--list-devices", action="store_true", help="list audio devices and exit")
    ap.add_argument(
        "--low-cpu", action="store_true",
        help="header-sync only: no blind fallback, no retrospective mid-stream "
        "decode. Searches only newly-arrived audio each poll instead of the "
        "whole buffer, and decodes once at the end of a locked reception "
        "instead of repeatedly for progress updates.",
    )
    args = ap.parse_args()

    if args.list_devices:
        for d in list_devices("input"):
            print(d.label())
        return

    config = RxConfig(
        out_dir=args.out_dir,
        poll_interval=args.poll_interval,
        end_grace=args.end_grace,
        size=args.size,
        once=args.once,
        blind_search_seconds=args.blind_search_seconds,
    )

    model = load_codec(args.model, precision=args.precision)
    ring = RingBuffer(args.buffer_seconds)
    state = SharedState()
    stop_event = threading.Event()

    stream, actual_rate = open_input_stream(args.device, ring, FS)
    print(f"listening at {actual_rate} Hz, buffer {args.buffer_seconds:.0f}s -- Ctrl+C to stop")

    target = decode_loop_low_cpu if args.low_cpu else decode_loop
    worker = threading.Thread(
        target=target, args=(ring, model, state, config, stop_event), daemon=True
    )
    worker.start()

    try:
        if args.no_gui:
            run_console(state, stop_event)
        else:
            run_gui(state, stop_event)
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        stream.stop()
        stream.close()
        worker.join(timeout=2.0)


if __name__ == "__main__":
    main()

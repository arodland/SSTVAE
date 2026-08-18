#!/usr/bin/env python3
"""Drive the whole aggregation path: two stations, one transmission.

Synthesizes (or reads) a transmission, degrades it independently per
station the way two receivers in different places would hear it, and
then runs each station through the *real* receive path -- decode_loop,
UploadSink, the reception payload, HTTP -- into a server that combines
them. Nothing here shortcuts to the combining function: the point is to
exercise what a skimmer actually does.

    # one station
    python scripts/aggregate_demo.py --snr 5 --seed 11 \
        --upload-url http://127.0.0.1:8000 --upload-key-file key1 \
        --station-call STA1 --frequency 14233000

    # ...then another, at a different seed, and watch the server's
    # picture improve as the second arrives.

With --wav it uses a real transmission from sstvae_encode.py; without,
it synthesizes unit latents, which needs no model and no download. The
latent SNR it prints is the same metric the test suite is calibrated
against, so "combined beats both" is checkable from the output.
"""

import argparse
import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sstvae import hfchannel  # noqa: E402
from sstvae.codec import load_codec  # noqa: E402
from sstvae.config import FS, MODES  # noqa: E402
from sstvae.modem.modem import Modem  # noqa: E402
from sstvae.rx import (  # noqa: E402
    RingBuffer,
    RxConfig,
    SaveToDirSink,
    SharedState,
    decode_loop,
)
from sstvae.upload import UploadSink  # noqa: E402
from sstvae.wavio import read_wav  # noqa: E402


class _StubDecoder:
    """A decoder that returns a flat picture.

    The station's own decode is incidental to an upload -- the latents
    are what goes on the wire, and the server makes the picture that
    matters -- so this lets the demo run with no model download at all.
    Real stations use the real codec; this exists so the path can be
    exercised offline.
    """

    def decode(self, latents, weights):
        from PIL import Image

        return Image.new("RGB", (640, 480), (32, 32, 32))


def latent_snr_db(sent, got, w=None) -> float:
    mask = np.ones_like(sent, dtype=bool) if w is None else (w > 0)
    err = np.mean((sent[mask] - got[mask]) ** 2)
    return 10 * np.log10(np.mean(sent[mask] ** 2) / err)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--wav", default=None,
                    help="a transmission from sstvae_encode.py (default: synthesize)")
    ap.add_argument("--mode", default="A", choices=sorted(MODES))
    ap.add_argument("--snr", type=float, default=5.0, help="channel SNR for this station")
    ap.add_argument("--fading", default=None, choices=[None, "mpg", "mpp", "mpd"],
                    help="fading preset (default: AWGN only)")
    ap.add_argument("--seed", type=int, default=1,
                    help="this station's noise seed -- different per station, which "
                         "is what makes the branches independent and the combining "
                         "worth anything")
    ap.add_argument("--callsign", default="BALLOON1", help="the transmitting station")
    ap.add_argument("--out-dir", default="demo-received")
    ap.add_argument("--model", default=None)
    ap.add_argument("--stub-decoder", action="store_true",
                    help="skip the local decode entirely (no model download). The "
                         "upload carries latents, so the server's picture is "
                         "unaffected.")
    ap.add_argument("--upload-url", required=True)
    ap.add_argument("--upload-key-file", required=True)
    ap.add_argument("--station-call", required=True)
    ap.add_argument("--frequency", type=float, default=None)
    ap.add_argument(
        "--utc-start", type=float, default=None, metavar="EPOCH",
        help="when the transmission began, as a unix timestamp. Two stations "
        "demonstrating aggregation must pass the SAME value: on the air one "
        "balloon transmits once and every receiver hears it at that moment, "
        "but this script fabricates its capture, so without this each run "
        "would stamp its own 'now' and the server would rightly file them as "
        "different transmissions. Defaults to now.",
    )
    args = ap.parse_args()

    modem = Modem()
    spec = MODES[args.mode]

    if args.wav:
        clean = read_wav(args.wav)
        sent_latents = None
        print(f"transmission from {args.wav} ({len(clean)/FS:.1f}s)")
    else:
        rng = np.random.default_rng(0)  # the *picture* is the same for everyone
        lat = rng.normal(size=spec.n_latents)
        sent_latents = lat / np.sqrt(np.mean(lat**2))
        clean = modem.modulate(sent_latents, args.mode, callsign=args.callsign)
        print(f"synthesized a mode-{args.mode} transmission from {args.callsign} "
              f"({len(clean)/FS:.1f}s)")

    heard = hfchannel.apply_channel(
        clean, snr_db=args.snr, fading_preset=args.fading, seed=args.seed
    )
    label = f"{args.snr:g} dB" + (f" {args.fading}" if args.fading else " AWGN")
    print(f"station {args.station_call.upper()} hears it at {label} (seed {args.seed})")

    # Into a ring buffer, and then the real decode loop over it. Some
    # silence in front so the reception does not start at sample zero.
    lead_in = int(2.0 * FS)
    ring = RingBuffer(200.0)
    ring.write(np.zeros(lead_in))
    ring.write(heard)

    # A real station's ring is stamped by its audio callback as the
    # audio arrives, so two receivers of one transmission agree on when
    # it started without arranging anything. Here the capture is
    # fabricated, so say when it happened: anchor the ring's clock so
    # the preamble's sample position dates to --utc-start.
    utc_start = time.time() if args.utc_start is None else args.utc_start
    ring.last_wall = utc_start + (ring.total_written - lead_in) / FS
    print(f"transmission dated {time.strftime('%H:%M:%SZ', time.gmtime(utc_start))} "
          f"({utc_start:.3f})")
    if args.utc_start is None:
        print("  note: pass the same --utc-start to every station, or the server "
              "will file them as separate transmissions")

    model = _StubDecoder() if args.stub_decoder else load_codec(
        args.model, precision=None
    )
    key = Path(args.upload_key_file).read_text().strip()
    sink = UploadSink(
        SaveToDirSink(args.out_dir),
        url=args.upload_url,
        key=key,
        station_callsign=args.station_call,
        dial_freq_hz=args.frequency,
        queue_dir=Path(args.out_dir) / "upload-queue",
    )

    state = SharedState()
    stop = threading.Event()
    config = RxConfig(out_dir=args.out_dir, poll_interval=1.0, end_grace=3.0, once=True)
    worker = threading.Thread(
        target=decode_loop, args=(ring, model, state, config, stop, sink), daemon=True
    )
    worker.start()
    deadline = time.time() + 180
    while worker.is_alive() and time.time() < deadline:
        time.sleep(0.2)
    stop.set()
    worker.join(timeout=15)

    with state.lock:
        if state.status == "listening" and not state.saved_path:
            print("no reception -- the channel may be too poor to acquire")
            return 1

    if sent_latents is not None:
        r = modem.demodulate(heard)
        print(f"this station alone: {latent_snr_db(sent_latents, r.latents, r.weights):+.2f} dB "
              "latent SNR")
        print("the server's combined figure is in its reply above, and on its gallery")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

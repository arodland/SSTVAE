#!/usr/bin/env python3
"""Measure what the capture path actually delivers.

For the case where a recording decodes cleanly with `sstvae_decode.py`
but the same audio played into a loopback decodes as garbage. That
pattern -- syncs, reports every frame, mangled picture -- means the
samples reaching the ring buffer are not the samples that were played,
and the interesting question is *how* they differ.

Three things can do it, and they are distinguishable:

* **Dropped samples.** A PortAudio input overflow discards a block. The
  stream stays continuous-looking but has a hole in it.
* **Clock error.** Playback and capture running on different clocks, or
  a resampler in the chain converting at a slightly wrong rate. The
  modem tracks sample-clock drift, but only to well under 0.1
  sample/frame; a loopback chain can be orders of magnitude past that.
* **Level or format problems.** Clipping, a silent channel, an unwanted
  gain stage.

    scripts/diagnose_capture.py --list
    scripts/diagnose_capture.py --device "Loopback" --seconds 70 \
        --out /tmp/captured.wav --reference /home/andrew/n6mts-sstvae5.wav

Start it, then play the file into the loopback. It reports what arrived,
saves it, and decodes it.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sstvae import wavio  # noqa: E402
from sstvae.audio import list_devices, open_input_stream  # noqa: E402
from sstvae.config import FS  # noqa: E402
from sstvae.rx import RingBuffer  # noqa: E402


class CountingRing:
    """A RingBuffer that also keeps everything, and counts writes."""

    def __init__(self, seconds: float):
        self._ring = RingBuffer(seconds)
        self.chunks: list[np.ndarray] = []
        self.first_write = None
        self.last_write = None

    def write(self, chunk) -> None:
        now = time.monotonic()
        if self.first_write is None:
            self.first_write = now
        self.last_write = now
        arr = np.asarray(chunk, dtype=np.float64).reshape(-1)
        self.chunks.append(arr)
        self._ring.write(arr)

    def __getattr__(self, name):
        return getattr(self._ring, name)

    def collected(self) -> np.ndarray:
        return (np.concatenate(self.chunks) if self.chunks
                else np.empty(0, dtype=np.float64))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true", help="list input devices and exit")
    ap.add_argument("--device", default=None, help="input device name or index")
    ap.add_argument("--seconds", type=float, default=70.0)
    ap.add_argument("--out", type=Path, default=None, help="write captured audio here")
    ap.add_argument("--reference", type=Path, default=None,
                    help="the file being played, for a like-for-like comparison")
    args = ap.parse_args()

    if args.list:
        for d in list_devices("input"):
            print(d)
        return 0

    statuses: list[str] = []
    ring = CountingRing(max(args.seconds + 10, 130.0))
    stream, rate = open_input_stream(args.device, ring, FS,
                                     on_error=statuses.append)
    print(f"capturing from {args.device or 'default'}: device stream at "
          f"{rate} Hz" + ("" if rate == FS else f", resampled to {FS} Hz here"))
    print(f"play the file now -- listening for {args.seconds:.0f} s")

    t0 = time.monotonic()
    try:
        time.sleep(args.seconds)
    except KeyboardInterrupt:
        pass
    elapsed = time.monotonic() - t0
    try:
        stream.stop()
        stream.close()
    except Exception:
        pass

    got = ring.collected()
    print(f"\n--- what arrived ---")
    print(f"  wall clock          {elapsed:.2f} s")
    print(f"  samples into ring   {len(got)} ({len(got) / FS:.2f} s at {FS} Hz)")
    if len(got) == 0:
        print("  NOTHING CAPTURED -- wrong device?")
        return 1

    effective = len(got) / elapsed
    ppm = (effective / FS - 1) * 1e6
    print(f"  effective rate      {effective:.1f} Hz  ({ppm:+.0f} ppm vs {FS})")
    if abs(ppm) > 1000:
        print("    ^^ more than 0.1% off. The modem's clock tracker handles far")
        print("       less than this; it will sync and then smear. Suspect a")
        print("       resampler or a rate mismatch in the loopback chain.")
    print(f"  peak level          {np.abs(got).max():.4f}"
          + ("   CLIPPING" if np.abs(got).max() >= 0.999 else ""))
    print(f"  silent fraction     {np.mean(np.abs(got) < 1e-6):.1%}")

    if statuses:
        print(f"  callback warnings   {len(statuses)}  <-- dropped audio")
        for s in dict.fromkeys(statuses):
            print(f"      {s}")
    else:
        print("  callback warnings   none")

    if args.out:
        wavio.write_wav(str(args.out), got)
        print(f"  saved to            {args.out}")

    # --- decode it ---------------------------------------------------------
    from sstvae.codec import load_codec, pad_to_full, reconstruct
    from sstvae.modem import Modem

    print("\n--- decoding what arrived ---")
    try:
        r = Modem().demodulate(got)
        print(f"  mode {r.mode.name}, {r.frames_received}/{r.mode.n_frames} frames, "
              f"freq offset {r.freq_offset:+.1f} Hz, SNR {r.snr_db:.1f} dB, "
              f"callsign {r.callsign!r}")
        if args.out:
            codec = load_codec()
            img = reconstruct(codec, pad_to_full(r.latents), pad_to_full(r.weights))
            png = args.out.with_suffix(".png")
            img.save(png)
            print(f"  picture saved to    {png}")
    except Exception as e:
        print(f"  demodulate failed: {type(e).__name__}: {e}")
        r = None

    if args.reference and r is not None:
        ref = wavio.read_wav(str(args.reference))
        rr = Modem().demodulate(ref)
        print("\n--- the same file, decoded straight from disk ---")
        print(f"  mode {rr.mode.name}, {rr.frames_received}/{rr.mode.n_frames} frames, "
              f"SNR {rr.snr_db:.1f} dB")
        print(f"\n  SNR through the loopback is {r.snr_db - rr.snr_db:+.1f} dB "
              "relative to the file.")
        print("  A few tenths is normal. Several dB means the loopback chain is")
        print("  damaging the signal, not the decoder.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

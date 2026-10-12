#!/usr/bin/env python3
"""Receive the QRSSTVAE passes, CE and L, in one slot recording.

    python qrss_receive.py rx.wav --slot 2026-10-09T06:00Z
    python qrss_receive.py rx.npz --slot 2026-10-09T06:00Z --frame short --no-store
    python qrss_receive.py rx.wav --slot 2026-10-09T06:00Z --just-l
    python qrss_receive.py rx.wav --slot 2026-10-09T06:00Z --start 2026-10-09T05:59:49Z \\
        --image pass.png

A WAV is 8 kHz audio starting 12 s before t0 (as `qrss_transmit.py`
writes it) unless --start gives its first sample's UTC time; an .npz is
`qrss_simulate.py`'s front-end output (its own slot and frame are used
unless --slot/--frame are given). Every signal in the slot is found
(preamble search, then the whole-slot track-before-detect), verified,
tracked and demodulated (`sstvae.qrss.receiver.receive_slot`), and for
each pass this prints its waveform and frame, its frequency at t0 (an L
pass's is its pilot's), drift, wander, Doppler, clock error, Z_ref, SNR
in 2500 Hz, the header, the callsign windows' text and match, the mean
weight W in dB and the association result.

Both waveforms are listened for by default; --just-ce or --just-l
listens for one. A CE-only receive is the receiver's original
single-waveform path, unchanged, and knows nothing of L: an L signal in
the recording can come out as a weak, headerless CE pass. --frame names
the frame shape in either waveform ("short" or "l-short"), and the
other waveform listens for the same shape (`lframe.twin`); a frame with
no twin (the CE version-1 and test frames) needs the flag that names its
own waveform.
An .npz's own frame fixes the waveform it holds, so a flag excluding
that waveform is refused rather than reporting an empty slot.

Passes are stored and associated in the multi-pass store
(--store DIR, default $QRSSTVAE_HOME or ~/.local/share/qrsstvae)
unless --no-store. The slot's whole front-end stream is also kept in the
48 h passband store (--passband DIR, default STORE/passband; off with
--no-passband or --no-store), where `qrss_decode.py --retro` later
template-searches it for passes too weak to detect now. --image renders the single pass (FULL frames with a
decoded header only) with the codec (--model DIR, --precision).
"""

import argparse
import sys
from pathlib import Path

import numpy as np

from sstvae.qrss import frame, frontend, lframe, receiver, sequences
from sstvae.qrss.constants import WAVEFORM_CE, WAVEFORM_L, WAVEFORM_NAMES


def parse_slot(s: str) -> int:
    """q for an ISO 8601 UTC time on a quarter hour, e.g. 2026-10-09T06:00Z."""
    try:
        return sequences.parse_slot(s)
    except ValueError as e:
        raise SystemExit(f"--slot {s!r}: {e}") from None


def parse_time(s: str) -> float:
    """Unix seconds of an ISO 8601 UTC time."""
    try:
        return sequences.parse_utc(s).timestamp()
    except ValueError as e:
        raise SystemExit(f"--start {s!r}: {e}") from None


def load_capture(path: Path, q: int | None, start: float | None):
    """(Capture, frame name or None) from a WAV or a simulator .npz."""
    if not path.is_file():
        raise SystemExit(f"{path}: no such file")
    if path.suffix.lower() == ".npz":
        from sstvae.qrss import channel

        sim = channel.load_npz(path)
        qq = sim.q if q is None else q
        return frontend.Capture(fe=sim.fe, q=qq, t0_index=sim.t0_index, fs=sim.fs), sim.spec.name
    if q is None:
        raise SystemExit("--slot is required for a WAV input")
    return receiver.capture_from_wav(path, q, start), None


def waveforms(args) -> tuple[int, ...]:
    """The waveforms listened for: one with --just-ce/--just-l, else both."""
    if args.just_ce:
        return (WAVEFORM_CE,)
    if args.just_l:
        return (WAVEFORM_L,)
    return (WAVEFORM_CE, WAVEFORM_L)


def select_frames(args, npz_frame: str | None = None):
    """The frame selection for `receiver.receive_slot`: a bare spec when one
    waveform is listened for, else (CE, L) twins of one shape.

    The shape is --frame's, else the .npz's own, else full.
    """
    wf = waveforms(args)
    base = lframe.get_any(args.frame or npz_frame or "full")
    if npz_frame is not None:
        own = lframe.get_any(npz_frame).waveform
        if own not in wf:
            flag = "--just-ce" if wf == (WAVEFORM_CE,) else "--just-l"
            raise SystemExit(f"the .npz holds frame {npz_frame}, waveform "
                             f"{WAVEFORM_NAMES[own]}, which {flag} does not listen for")
    specs = []
    for w in wf:
        try:
            specs.append(lframe.twin(base, w))
        except ValueError:
            flag = "--just-ce" if base.waveform == WAVEFORM_CE else "--just-l"
            raise SystemExit(f"frame {base.name} has no {WAVEFORM_NAMES[w]} twin; "
                             f"add {flag}") from None
    return specs[0] if len(specs) == 1 else tuple(specs)


def pass_lines(p) -> list[str]:
    r = p.report
    wf = int(getattr(p, "waveform", WAVEFORM_CE))
    out = [f"pass {p.uid}   {WAVEFORM_NAMES.get(wf, wf)} {p.frame}",
           f"  {'pilot    ' if wf == WAVEFORM_L else 'frequency'}  {r.offset_hz:9.3f} Hz at t0   "
           f"drift {r.drift_hz_per_min:+.3f} Hz/min   wander {r.wander_hz_rms:.3f} Hz rms",
           f"  doppler    {r.doppler_hz:.3f} Hz   clock {r.ppm:+.1f} ppm   "
           f"Z_ref {r.z_ref:.1f}   SNR2500 {r.snr2500_db:+.1f} dB   kappa {r.kappa:.2f}"
           + ("   SUSPECT" if r.suspect else "") + ("" if r.slip_free else "   phase slips")]
    h = p.header
    if h is not None:
        out.append(f"  header     {h.callsign} {h.grid} picture {h.picture_id:08x} "
                   f"mode {'ABC'[h.mode]} segment {h.segment} codec {h.codec_id:04x}")
    else:
        out.append("  header     not decoded")
    cw = p.cw
    if cw is not None:
        agrees = "" if cw.agrees is None else ("   agrees" if cw.agrees else "   DISAGREES")
        out.append(f"  callsign   {cw.text or '(not read)'}   match Z {cw.z_match:.1f}   "
                   f"keying {cw.keying}{agrees}")
    out.append(f"  mean W     {receiver.mean_w_db(p):+.2f} dB")
    return out


def render_pass(p, image: Path, model: str | None, precision: str) -> str:
    """Write the single pass's picture; returns a status line."""
    from sstvae.qrss import picture, render

    h = p.header
    if h is None:
        return "image: no header decoded, cannot place the latents"
    if len(p.z) != picture.SENT:
        return f"image: a {p.frame} frame holds {len(p.z)} latents, not a picture segment"
    S = np.zeros((picture.N_GROUPS, picture.GROUP_LATENTS), dtype=np.float64)
    W = np.zeros_like(S)
    idx = picture.air_to_canonical(h.segment)
    w = np.asarray(p.w, dtype=np.float64)
    S[h.segment, idx] = w * np.asarray(p.z, dtype=np.float64)
    W[h.segment, idx] = w
    try:
        codec = render.load(precision, model, codec_id=h.codec_id, warn=print)
    except render.CodecMismatch as e:
        return f"image: not written: {e}"
    render.render(codec, S, W, h.mode).save(image)
    return f"image: {image}"


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="slot recording: 8 kHz WAV, or a qrss_simulate .npz")
    ap.add_argument("--slot", default=None, help="quarter hour, ISO UTC, e.g. 2026-10-09T06:00Z")
    ap.add_argument("--start", default=None,
                    help="UTC time of the WAV's first sample (default: t0 - 12 s)")
    ap.add_argument("--frame", choices=sorted(frame.PRESETS) + sorted(lframe.PRESETS),
                    default=None,
                    help="frame shape, either waveform's name for it "
                         "(default full, or the .npz's own)")
    wf = ap.add_mutually_exclusive_group()
    wf.add_argument("--just-ce", action="store_true",
                    help="listen for waveform CE only (default: CE and L)")
    wf.add_argument("--just-l", action="store_true",
                    help="listen for waveform L only (default: CE and L)")
    ap.add_argument("--store", default=None, help="multi-pass store directory")
    ap.add_argument("--no-store", action="store_true", help="do not store or associate")
    ap.add_argument("--passband", default=None,
                    help="48 h passband store directory (default STORE/passband)")
    ap.add_argument("--no-passband", action="store_true",
                    help="do not keep the slot's front-end stream")
    ap.add_argument("--estimator", choices=("joint", "plain"), default="joint",
                    help="per-block data estimator (default joint)")
    ap.add_argument("--image", type=Path, default=None,
                    help="render the (first) pass's picture to this PNG")
    ap.add_argument("--model", default=None, help="codec model directory (for --image)")
    ap.add_argument("--precision", choices=("fp32", "fp16", "int8"), default="fp32",
                    help="codec precision (for --image)")
    return ap.parse_args(argv)


def main() -> None:
    args = parse_args()

    q = parse_slot(args.slot) if args.slot else None
    start = parse_time(args.start) if args.start else None
    cap, npz_frame = load_capture(Path(args.input), q, start)
    spec = select_frames(args, npz_frame)

    store = None
    if not args.no_store:
        from sstvae.qrss.store import Store

        store = Store(args.store)
        if not args.no_passband and cap.fs == frontend.FE_FS:
            pb_dir = Path(args.passband) if args.passband else store.root / "passband"
            frontend.PassbandStore(pb_dir).write_capture(cap)

    passes = receiver.receive_slot(cap, spec, estimator=args.estimator)
    if not passes:
        names = " or ".join(WAVEFORM_NAMES[s.waveform] for s in receiver.frames(spec))
        print(f"no {names} signal found" + (" (slot kept for a later --retro search)"
                                            if store is not None and not args.no_passband
                                            else ""))
        sys.exit(1)
    for p in passes:
        for line in pass_lines(p):
            print(line)
        if store is not None:
            from sstvae.qrss.associate import associate

            key, rule = associate(p, store)
            if key is None:
                print(f"  stored     provisional ({rule})")
            else:
                print(f"  stored     {key[0]} {key[1]:08x} by {rule or 'existing membership'}")
    if args.image is not None:
        print(render_pass(passes[0], args.image, args.model, args.precision))


if __name__ == "__main__":
    main()

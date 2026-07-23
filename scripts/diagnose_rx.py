#!/usr/bin/env python3
"""Diagnose sync/header failures on a received recording.

    python scripts/diagnose_rx.py rx.wav [--invert]

Reports the autocorrelation metric timeline (where a preamble-like
region exists, if any), the acquisition result, header soft-bit
quality, and retries with an inverted spectrum (LSB reception).
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sstvae import wavio
from sstvae.config import (
    FS,
    M,
    NC,
    NCP,
    NSYM,
    PREAMBLE_CP,
    PREAMBLE_SAMPLES,
    DEMOD_BACKOFF,
)
from sstvae.modem import framing, ofdm
from sstvae.modem.dsp import to_baseband, freq_correct, sync_lowpass
from sstvae.modem.sync import _autocorr_metric, acquire, SyncError


def spectral_invert(x: np.ndarray) -> np.ndarray:
    """LSB<->USB flip around FCENTER... implemented as full-band inversion
    (multiply by Nyquist carrier); combined with retuning this recovers
    an LSB recording well enough to test the hypothesis."""
    return x * np.cos(np.pi * np.arange(len(x)))


def report(x: np.ndarray, label: str, search=None) -> bool:
    print(f"\n=== {label} ===")
    z = to_baseband(x)
    zf = sync_lowpass(z)
    metric, _ = _autocorr_metric(zf)
    if search is not None:
        lo, hi = int(search[0] * FS), int(search[1] * FS)
        m = np.full_like(metric, -1.0)
        m[lo:hi] = metric[lo:hi]
        metric = m
    n_best = int(np.argmax(metric))
    print(f"peak autocorr metric {metric[n_best]:.3f} at sample {n_best} "
          f"(t={n_best/FS:.2f} s); >0.5 expected for a real preamble")
    top = np.argsort(metric)[-5:][::-1]
    print("top candidates:", ", ".join(f"{n} ({metric[n]:.2f})" for n in top))

    try:
        acq = acquire(
            z,
            search=None
            if search is None
            else (int(search[0] * FS), int(search[1] * FS)),
        )
    except SyncError as e:
        print(f"acquire failed: {e}")
        return False
    print(f"acquired: preamble at {acq.preamble_start} "
          f"(t={acq.preamble_start/FS:.2f} s), freq offset {acq.freq_offset:+.1f} Hz, "
          f"metric {acq.metric:.3f}")

    zc = freq_correct(z, acq.freq_offset)
    u0 = acq.preamble_start + PREAMBLE_CP
    h_pre = (
        ofdm.demod_window(zc, u0, DEMOD_BACKOFF)
        + ofdm.demod_window(zc, u0 + M, DEMOD_BACKOFF)
    ) / (2 * ofdm.pilot_sequence())
    snr_proxy = np.abs(h_pre).mean() / (np.abs(h_pre).std() + 1e-9)
    print(f"|H_pre| mean/std ratio: {snr_proxy:.1f} (higher = cleaner)")

    h0 = acq.preamble_start + PREAMBLE_SAMPLES
    soft = np.zeros(NC)
    for s in range(2):
        y = ofdm.demod_window(zc, h0 + s * NSYM + NCP, DEMOD_BACKOFF)
        soft += np.real(y * np.conj(h_pre))
    margin = np.abs(soft).mean() / (np.abs(soft).std() + 1e-9)
    spec = framing.decode_header(soft)
    print(f"header soft bits: |mean|/std {margin:.2f}, "
          f"decode -> {spec.name if spec else 'FAILED'}")
    return spec is not None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("wav")
    ap.add_argument("--invert", action="store_true", help="only try inverted")
    ap.add_argument("--search-start", type=float, default=None, help="seconds")
    ap.add_argument("--search-end", type=float, default=None, help="seconds")
    args = ap.parse_args()

    x = wavio.read_wav(args.wav)
    print(f"{args.wav}: {len(x)} samples ({len(x)/FS:.1f} s at {FS} Hz), "
          f"rms {np.sqrt(np.mean(x**2)):.4f}")

    search = None
    if args.search_start is not None or args.search_end is not None:
        search = (args.search_start or 0.0, args.search_end or len(x) / FS)
        print(f"preamble search limited to {search[0]:.1f}-{search[1]:.1f} s")

    ok = False
    if not args.invert:
        ok = report(x, "as recorded (USB assumed)", search)
    if not ok:
        report(spectral_invert(x), "spectrum inverted (LSB hypothesis)", search)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Generate the committed golden-vector corpus.

Known inputs and the Python reference's outputs at every module
boundary. Both test suites read the same files, so "the C++ port is
correct" is a claim `pytest` and `ctest` check against identical bytes
rather than against two independently written expectations.

    tools/gen_golden_vectors.py            # write the corpus
    tools/gen_golden_vectors.py --check    # exit 1 if it would change

## Why this format

Arrays are plain `.npy` -- a documented, stable, trivially parseable
container that numpy writes for free and ~90 lines of C++ reads (see
`native/core/testing/npy.hpp`). The alternative, `.npz`, is a zip and
would drag a zip reader into the C++ test build for no benefit.

Alongside them, `manifest.json` lists every vector with its shape and
dtype. That file is the reviewable part: regenerating the corpus
produces a diff that names exactly which vectors moved, which is the
property that makes deliberate regeneration safe. A directory of binary
blobs alone would not be reviewable at all, and "just trust the
regeneration" is how a format change gets blessed by accident.

## Bitwise vectors and tolerance vectors

**Not every vector is byte-reproducible, and pretending otherwise breaks
CI on machines that are working correctly.** Found the hard way: the
first CI run failed `--check` on three vectors, on runners whose only
sin was having a different BLAS kernel.

The dividing line is **what IEEE 754 actually guarantees**:

* **Bitwise** vectors use only operations that are guaranteed
  reproducible -- integer arithmetic, `+ - * /` (correctly rounded by
  the standard), and seeded numpy RNG draws (whose streams numpy commits
  to keeping stable). These compare **byte for byte** and carry a
  SHA-256.
* **Tolerance** vectors touch something that is *not* guaranteed:
  - a **transcendental** (`exp`, `sin`, `cos`), which no standard
    requires to be correctly rounded, so results differ between libms
    and between SIMD paths -- x86-64 and Apple silicon disagree here;
  - a **`@` (BLAS)** or an **FFT**, which associate their sums according
    to the BLAS build, the CPU's instruction set and the library version.

  These compare **by value**, and instead of a churning SHA-256 the
  manifest carries a fingerprint of value-domain statistics -- stable
  across platforms, but moved visibly by any real change.

The first cut of this file drew the line at BLAS/FFT alone and asserted
that "elementwise transcendentals proved identical across every runner".
That was an over-generalisation from one green Linux run: the next CI
round failed `MOD_MATRIX` and `DEMOD_MATRIX` on macOS and Windows. The
rule above is the one that follows from the standard rather than from a
sample of one.

Note the interaction with the unreduced-argument problem in
`docs/todo.md`: these tables are built on arguments up to 262 rad, where
libm implementations diverge *most*, because they differ in how far they
carry argument reduction. Fixing that would shrink the cross-platform
spread from ~6e-14 to ~1e-16 -- but it still would not make them
bitwise, so the classification here is right either way.

The tolerances are ~1e11 times tighter than anything that could affect a
decode, so this weakens the check by nothing that matters. What it buys
is that a `--check` failure now means *the reference changed*, which is
the only thing the check was ever supposed to detect.

`golay/soft_expected` is a deliberate edge case: it is produced through
a BLAS matmul but its output is an integer message index, so it is kept
bitwise. Flipping one would require two codewords whose correlation
scores are within ~1e-13 of each other, which does not happen at these
noise levels -- and if it ever did, that is a genuine near-tie worth
seeing rather than noise worth suppressing.

## Why inputs are committed too

Every input is committed next to its expected output. A test that
generated its own inputs from a seeded RNG would need numpy's exact
generator in C++ -- the same dependency `gen_config_header.py` refuses
to take on for the pilot sequence -- and would silently test different
data on the two sides the moment anything about the draw changed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DEFAULT_OUT = ROOT / "native" / "tests" / "golden"


# Tolerance for anything a platform is free to compute differently: a
# transcendental, a BLAS product, or an FFT.
#
# Sized by the larger of the two mechanisms. Sums reassociate by ~1e-15.
# Transcendentals are worse here than they look, because these phasors
# are evaluated at arguments up to 262 rad where one ulp is 5.7e-14, and
# implementations differ in argument reduction -- so the observed
# cross-platform spread is ~6e-14. 1e-11 gives two orders of margin over
# that, and is still ~1e11 tighter than anything a decode could notice
# (1e-11 on a unit phasor is -220 dB).
PLATFORM_TOL = 1e-11


class Corpus:
    """Collects arrays, writes them as .npy, and records the manifest."""

    def __init__(self) -> None:
        self.entries: dict[str, dict] = {}
        self.blobs: dict[str, bytes] = {}
        self.arrays: dict[str, np.ndarray] = {}

    def add(self, name: str, array, note: str = "", *, tol: float = 0.0) -> None:
        """Record one vector.

        `tol` of 0 means the vector must reproduce byte for byte. Pass
        PLATFORM_TOL for anything computed through `@` or an FFT -- see
        the module docstring for why those cannot be bitwise.
        """
        arr = np.ascontiguousarray(array)
        if arr.dtype == np.float32:
            raise ValueError(f"{name}: emit float64, not float32")
        import io

        buf = io.BytesIO()
        # allow_pickle=False is the default, but be explicit: a pickled
        # object array would be unreadable from C++ and a security
        # problem in a file the test suite loads.
        np.save(buf, arr, allow_pickle=False)
        blob = buf.getvalue()
        self.blobs[name] = blob
        self.arrays[name] = arr
        entry: dict = {"dtype": arr.dtype.str, "shape": list(arr.shape)}
        if tol:
            entry["tol"] = tol
            entry["fingerprint"] = fingerprint(arr)
        else:
            entry["sha256"] = hashlib.sha256(blob).hexdigest()
        if note:
            entry["note"] = note
        self.entries[name] = entry

    def add_scalar(self, name: str, value) -> None:
        """A single number, kept in the manifest itself.

        Scalars are the interesting ones to eyeball in a review, so they
        live in the readable file rather than in a 128-byte .npy.
        """
        self.entries[name] = {"scalar": value}

    def tol_for(self, name: str) -> float:
        return float(self.entries.get(name, {}).get("tol", 0.0))

    def manifest(self) -> str:
        doc = {
            "_comment": [
                "GENERATED by tools/gen_golden_vectors.py -- do not hand-edit.",
                "Arrays live beside this file as .npy.",
                "Entries with 'sha256' use only operations IEEE 754 makes",
                "reproducible, so they must match byte for byte.",
                "Entries with 'tol' touch a transcendental, a BLAS product",
                "or an FFT -- none of which any standard requires to give",
                "identical bits on every machine -- so they are checked by",
                "value; 'fingerprint' is a platform-stable summary that",
                "still moves visibly if the reference's output changes.",
                "Either way, a --check failure means the Python reference",
                "changed: a deliberate format change, or a bug. Not noise.",
            ],
            "vectors": self.entries,
        }
        return json.dumps(doc, indent=2, sort_keys=True) + "\n"

    def files(self) -> dict[str, bytes]:
        out = {f"{name}.npy": blob for name, blob in self.blobs.items()}
        out["manifest.json"] = self.manifest().encode("utf-8")
        return out


def fingerprint(arr: np.ndarray) -> dict:
    """A BLAS-stable summary of an array's values.

    Ten significant figures: far above the ~1e-15 that reassociation
    moves these by, and far below any change worth calling a change. The
    point is a manifest diff a human can read -- the numeric gate is the
    tolerance comparison in `--check`, not this.
    """
    flat = np.asarray(arr).ravel()
    mag = np.abs(flat)
    return {
        "rms": f"{float(np.sqrt(np.mean(mag ** 2))):.10e}",
        "max_abs": f"{float(mag.max()) if mag.size else 0.0:.10e}",
        "mean_abs": f"{float(mag.mean()) if mag.size else 0.0:.10e}",
    }


# --- the vectors themselves ------------------------------------------------


def build_golay(c: Corpus) -> None:
    from sstvae.modem import golay

    c.add_scalar("golay/min_distance", golay.min_distance())

    # The whole codebook. It is only 4096 entries and it pins the
    # generator polynomial, the systematic layout and the parity bit in
    # one artifact -- there is no cheaper way to be sure two encoders
    # agree everywhere rather than on the messages someone thought to try.
    c.add("golay/all_codewords",
          np.array([golay.encode(m) for m in range(4096)], dtype=np.int64),
          "encode(m) for every 12-bit message")

    messages = np.array([0, 1, 2, 0x555, 0xAAA, 0x7FF, 0x800, 0xABC, 0xFFE, 0xFFF],
                        dtype=np.int64)
    c.add("golay/bits_messages", messages)
    c.add("golay/bits_expected",
          np.array([golay.codeword_bits(int(m)) for m in messages], dtype=np.int64),
          "codeword_bits, MSB first")

    # Soft-decision cases, in three regimes: clean, at the correction
    # limit (3 hard errors, which the code must always fix), and noisy
    # enough that some cases legitimately fail -- the expected outputs
    # record what the reference actually decides, including its mistakes,
    # because a port that "fixes" one has diverged.
    rng = np.random.default_rng(20260728)
    soft_cases = []
    for m in range(0, 4096, 97):
        bits = golay.codeword_bits(m).astype(float)
        soft_cases.append(1.0 - 2.0 * bits)
    for _ in range(200):
        m = int(rng.integers(0, 4096))
        bits = golay.codeword_bits(m).copy()
        bits[rng.choice(24, size=3, replace=False)] ^= 1
        soft_cases.append(1.0 - 2.0 * bits.astype(float))
    for scale in (0.3, 0.7, 1.2):
        for _ in range(200):
            m = int(rng.integers(0, 4096))
            soft_cases.append(1.0 - 2.0 * golay.codeword_bits(m)
                              + rng.normal(scale=scale, size=24))
    soft = np.array(soft_cases, dtype=np.float64)
    c.add("golay/soft_inputs", soft, "clean, 3-error, and noisy soft values")
    c.add("golay/soft_expected",
          np.array([golay.decode_soft(s) for s in soft], dtype=np.int64),
          "decode_soft output, including its errors on noisy input")


def build_ofdm(c: Corpus) -> None:
    from sstvae.config import M, NC, NCP
    from sstvae.modem import ofdm
    from sstvae.modem.dsp import to_baseband

    c.add("ofdm/carrier_freqs", ofdm.CARRIER_FREQS.astype(np.float64))
    c.add("ofdm/baseband_freqs", ofdm.BASEBAND_FREQS.astype(np.float64))
    # Elementwise np.exp, and therefore *not* bitwise across platforms:
    # x86-64 and Apple silicon disagree in the last bits, worst here
    # because the arguments run up to 262 rad.
    c.add("ofdm/mod_matrix", ofdm.MOD_MATRIX.astype(np.complex128),
          "(NSYM, NC) passband modulation matrix", tol=PLATFORM_TOL)
    c.add("ofdm/demod_matrix", ofdm.DEMOD_MATRIX.astype(np.complex128),
          "(NC, M) baseband demodulation matrix", tol=PLATFORM_TOL)

    c.add("ofdm/pilot_sequence", ofdm.pilot_sequence().astype(np.complex128),
          "the on-air pilot constant; see config.hpp PILOT_QUADRANTS",
          tol=PLATFORM_TOL)
    # The three replicas are `e @ p` -- a BLAS matvec, so not bitwise.
    c.add("ofdm/preamble_waveform", ofdm.preamble_waveform().astype(np.float64),
          tol=PLATFORM_TOL)
    c.add("ofdm/preamble_template", ofdm.preamble_template().astype(np.complex128),
          tol=PLATFORM_TOL)
    c.add("ofdm/pilot_template", ofdm.pilot_template().astype(np.complex128),
          tol=PLATFORM_TOL)

    rng = np.random.default_rng(20260728)
    n_sym = 12
    symbols = ((rng.normal(size=(n_sym, NC)) + 1j * rng.normal(size=(n_sym, NC)))
               / np.sqrt(2)).astype(np.complex128)
    c.add("ofdm/modulate_input", symbols)
    c.add("ofdm/modulate_expected",
          ofdm.modulate_symbols(symbols).astype(np.float64), tol=PLATFORM_TOL)

    # demod_window is exercised through a real baseband signal rather
    # than on synthetic input, so the vector covers the window placement
    # and the CP backoff the way the modem actually uses them.
    pad = np.zeros((2, NC), dtype=complex)
    wave = ofdm.modulate_symbols(np.vstack([pad, symbols, pad]))
    z = to_baseband(wave)
    c.add("ofdm/demod_baseband", z.astype(np.complex128),
          "to_baseband() of the modulated test symbols", tol=PLATFORM_TOL)
    starts, backoffs, outputs = [], [], []
    for i in range(n_sym):
        for backoff in (0, 6):
            start = (2 + i) * (M + NCP) + NCP
            starts.append(start)
            backoffs.append(backoff)
            outputs.append(ofdm.demod_window(z, start, backoff))
    c.add("ofdm/demod_starts", np.array(starts, dtype=np.int64))
    c.add("ofdm/demod_backoffs", np.array(backoffs, dtype=np.int64))
    c.add("ofdm/demod_expected", np.array(outputs, dtype=np.complex128),
          "one row per (start, backoff) pair", tol=PLATFORM_TOL)

    # Past the end of the buffer: demod_window zero-pads a short window,
    # and that edge decides what happens at the tail of a recording.
    tail_start = len(z) - M // 2
    c.add("ofdm/demod_tail_start", np.array([tail_start], dtype=np.int64))
    c.add("ofdm/demod_tail_expected",
          ofdm.demod_window(z, tail_start).astype(np.complex128),
          "window running off the end of the signal; zero-padded",
          tol=PLATFORM_TOL)


def build_dsp(c: Corpus) -> None:
    from scipy import signal

    from sstvae.config import FS, TX_BANDPASS
    from sstvae.modem import dsp

    # FIR designs. These are part of the waveform (the transmit bandpass
    # shapes what goes on air) and of acquisition (the sync lowpass sets
    # what the preamble detector sees), so the port has to reproduce
    # scipy's firwin rather than design "a reasonable" filter.
    #
    # Tolerance rather than bitwise: sinc and the Hamming window are
    # transcendental, and the scale normalization sums 129 or 201 terms.
    c.add("dsp/firwin_sync", signal.firwin(129, 850.0, fs=FS),
          "firwin(129, 850, fs=FS) -- the sync lowpass", tol=PLATFORM_TOL)
    c.add("dsp/firwin_tx",
          signal.firwin(201, TX_BANDPASS, fs=FS, pass_zero=False),
          "firwin(201, TX_BANDPASS, pass_zero=False) -- the transmit bandpass",
          tol=PLATFORM_TOL)

    rng = np.random.default_rng(20260728)

    # A signal with real structure rather than noise alone, so the
    # filters and the analytic transform are exercised on something with
    # the spectrum they were designed for.
    n = 4096
    t = np.arange(n) / FS
    x = (np.sin(2 * np.pi * 1200 * t) + 0.5 * np.sin(2 * np.pi * 1900 * t)
         + 0.2 * rng.normal(size=n))
    c.add("dsp/signal_input", x, "test signal: two tones plus noise")

    c.add("dsp/to_baseband", dsp.to_baseband(x),
          "heterodyne by FCENTER; exactly periodic in 16 samples",
          tol=PLATFORM_TOL)
    c.add("dsp/hilbert", signal.hilbert(x), "analytic signal", tol=PLATFORM_TOL)
    c.add("dsp/sync_lowpass", dsp.sync_lowpass(dsp.to_baseband(x)),
          tol=PLATFORM_TOL)
    c.add("dsp/papr_db", np.array([dsp.papr_db(x)]), tol=PLATFORM_TOL)
    c.add("dsp/tx_condition", dsp.tx_condition(x, 0.5),
          "clip-and-filter at the configured headroom", tol=PLATFORM_TOL)
    c.add("dsp/to_int16", dsp.to_int16(x).astype(np.int64),
          "np.round is half-to-even; std::round would differ")

    # freq_correct at offsets spanning the acquisition range, including
    # a negative one and a deliberately awkward fractional value.
    z = dsp.to_baseband(x)
    offsets = np.array([0.0, 1.0, -1.0, 12.5, 37.5, -55.0, 7.3125], dtype=np.float64)
    c.add("dsp/freq_correct_offsets", offsets)
    c.add("dsp/freq_correct",
          np.array([dsp.freq_correct(z, f) for f in offsets]),
          "one row per offset", tol=PLATFORM_TOL)

    # An odd length too: hilbert's frequency mask differs for odd n, and
    # a recording is not going to be a nice round number of samples.
    x_odd = x[:1001]
    c.add("dsp/signal_input_odd", x_odd)
    c.add("dsp/hilbert_odd", signal.hilbert(x_odd),
          "odd length takes the other branch of the mask", tol=PLATFORM_TOL)


BUILDERS = {"dsp": build_dsp, "golay": build_golay, "ofdm": build_ofdm}


def build() -> Corpus:
    c = Corpus()
    for name in sorted(BUILDERS):
        BUILDERS[name](c)
    return c


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--check", action="store_true",
                    help="exit 1 if any vector differs from the committed one")
    args = ap.parse_args()

    corpus = build()
    files = corpus.files()

    if args.check:
        stale, missing, drifted = [], [], []
        worst = 0.0
        for name, blob in sorted(files.items()):
            path = args.out / name
            if not path.exists():
                missing.append(name)
                continue
            committed_bytes = path.read_bytes()
            if committed_bytes == blob:
                continue

            # The manifest is the one text file here, so it is the one
            # thing a checkout can rewrite. Compare it by *content*:
            # git for Windows converts LF to CRLF by default, which
            # otherwise fails only on Windows and looks like a corpus
            # problem rather than a checkout one. .gitattributes should
            # prevent it; this makes it self-diagnosing if it recurs.
            if name == "manifest.json":
                if committed_bytes.replace(b"\r\n", b"\n") == blob:
                    print("note: manifest.json differs only in line endings "
                          "-- your checkout rewrote it. Check .gitattributes; "
                          "the content is correct.", file=sys.stderr)
                    continue
                stale.append(name)
                continue

            vector = name[:-4] if name.endswith(".npy") else None
            tol = corpus.tol_for(vector) if vector else 0.0
            if not tol:
                stale.append(name)  # bitwise vector
                continue

            # A tolerance vector: the bytes may legitimately differ
            # because BLAS summed in another order. Compare the values.
            committed = np.load(path)
            fresh = corpus.arrays[vector]
            if committed.shape != fresh.shape or committed.dtype != fresh.dtype:
                stale.append(f"{name} (shape/dtype changed)")
                continue
            diff = float(np.max(np.abs(committed - fresh))) if fresh.size else 0.0
            worst = max(worst, diff)
            if diff > tol:
                drifted.append(f"{name}: max |diff| = {diff:.3e} > tol {tol:.0e}")

        extra = sorted(
            str(p.relative_to(args.out)).replace("\\", "/")
            for p in args.out.rglob("*")
            if p.is_file()
            and str(p.relative_to(args.out)).replace("\\", "/") not in files
        )
        if missing or stale or drifted or extra:
            for n in missing:
                print(f"missing: {n}", file=sys.stderr)
            for n in stale:
                print(f"out of date: {n}", file=sys.stderr)
            for n in drifted:
                print(f"drifted beyond tolerance: {n}", file=sys.stderr)
            for n in extra:
                print(f"not generated by this script: {n}", file=sys.stderr)
            print("\nThe golden corpus disagrees with the Python reference.\n"
                  "If this is a deliberate format change, re-run\n"
                  "tools/gen_golden_vectors.py and review the manifest diff.",
                  file=sys.stderr)
            return 1
        print(f"{len(files)} golden files are up to date"
              + (f" (largest value drift {worst:.2e}, within tolerance)"
                 if worst else ""))
        return 0

    for name, blob in sorted(files.items()):
        path = args.out / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(blob)
    total = sum(len(b) for b in files.values())
    print(f"wrote {len(files)} files to {args.out} ({total / 1024:.0f} KiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

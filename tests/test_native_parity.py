"""Compare the C++ core against the Python reference, side by side.

This is the complement to `pytest --native`. That mode *substitutes* the
C++ implementations and runs the whole suite through them, answering
"does the port satisfy everything we require of the modem?". These tests
instead hold both implementations in one process and diff them
directly, answering "where exactly do they differ, and by how much?" --
which is the question you need answered when the first mode fails.

Skipped entirely when the extension module has not been built, so a
normal `pytest` run is unaffected. Build it with tools/build_native.sh.

The tolerances here are the same ones justified in
native/tests/test_golden.cpp; see that file for why they are not zero.
The short version: both sides now reduce the phasor argument exactly in
integer arithmetic before calling exp(), so the only residual is that no
standard requires a transcendental to be correctly rounded -- about one
ulp. Measured 9.6e-16 on the OFDM matrices, 0 on the pilot sequence.
"""

import numpy as np
import pytest

from sstvae.config import M, NC, NCP
from sstvae.modem import golay, ofdm
from sstvae.modem.dsp import to_baseband

PHASOR_TOL = 1e-14
PHASOR_SUM_TOL = 1e-13


def max_abs_diff(a, b) -> float:
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))


# --- the two implementations must have been built from one config ----------

def test_config_agrees(native):
    """A C++ build from a different config.py would make every other
    check here meaningless, so it is the first thing verified."""
    from sstvae import config as cfg

    for name in ("FS", "RS", "NC", "M", "NCP", "NSYM", "CARRIER0", "FCENTER",
                 "FRAME_SAMPLES", "LATENTS_PER_FRAME", "BEACON_CARRIER",
                 "FRAMES_PER_GROUP", "GROUP_LATENTS", "PREAMBLE_SAMPLES",
                 "PROTOCOL_VERSION"):
        assert getattr(native.config, name) == getattr(cfg, name), (
            f"{name} differs: C++ has {getattr(native.config, name)}, "
            f"Python has {getattr(cfg, name)}. Re-run "
            "tools/gen_config_header.py and rebuild."
        )


# --- golay: integer arithmetic, so equality is exact -----------------------

def test_golay_encodes_every_message_identically(native):
    py = np.array([golay.encode(m) for m in range(4096)])
    cpp = np.array([native.golay.encode(m) for m in range(4096)])
    assert np.array_equal(py, cpp)


def test_golay_codeword_bits(native):
    for m in (0, 1, 0x555, 0xAAA, 0xABC, 0xFFF):
        assert np.array_equal(golay.codeword_bits(m), native.golay.codeword_bits(m))


def test_golay_min_distance(native):
    assert golay.min_distance() == native.golay.min_distance() == 8


def test_golay_soft_decode_agrees_including_its_mistakes(native):
    """Noise levels from clean to hopeless.

    The high-noise cases matter most: there the decoder is often wrong,
    and the two implementations have to be wrong *in the same way*. A
    port that broke ties differently would pass a clean-input test and
    fail here.
    """
    rng = np.random.default_rng(4242)
    disagreements = 0
    total = 0
    for scale in (0.0, 0.5, 1.0, 2.0, 5.0):
        for _ in range(200):
            m = int(rng.integers(0, 4096))
            soft = 1.0 - 2.0 * golay.codeword_bits(m)
            if scale:
                soft = soft + rng.normal(scale=scale, size=NC)
            total += 1
            if golay.decode_soft(soft) != native.golay.decode_soft(soft):
                disagreements += 1
    assert disagreements == 0, f"{disagreements}/{total} soft decodes differ"


def test_golay_tie_breaking_matches(native):
    """An all-zero soft vector scores every codeword identically.

    The answer is arbitrary, but it must be the *same* arbitrary answer:
    np.argmax returns the first maximum, and the C++ scans upward with a
    strict comparison to match.
    """
    zeros = np.zeros(24)
    assert golay.decode_soft(zeros) == native.golay.decode_soft(zeros)


# --- ofdm: transcendentals, so tolerances apply ----------------------------

def test_ofdm_frequency_tables_are_exact(native):
    """Small integers, exactly representable: no tolerance is defensible."""
    assert np.array_equal(ofdm.CARRIER_FREQS, native.ofdm.carrier_freqs())
    assert np.array_equal(ofdm.BASEBAND_FREQS, native.ofdm.baseband_freqs())


def test_ofdm_matrices(native):
    assert max_abs_diff(ofdm.MOD_MATRIX, native.ofdm.mod_matrix()) < PHASOR_TOL
    assert max_abs_diff(ofdm.DEMOD_MATRIX, native.ofdm.demod_matrix()) < PHASOR_TOL


def test_both_sides_range_reduce_their_phasors(native):
    """Not parity -- a check that neither side has regressed to building
    phasors on an unreduced argument.

    Every entry is a unit phasor in exact arithmetic, so `|z| - 1`
    measures each implementation's own error without needing a
    high-precision reference here. An unreduced argument reaching 262 rad
    costs ~3e-14; a reduced one costs ~1e-16. A few ulp is the pass mark,
    and the gap between the two regimes is two orders of magnitude, so
    this cannot fail marginally.

    This replaced an earlier test asserting the C++ was the *more*
    accurate side, which was true only while sstvae/modem/ofdm.py still
    computed the unreduced form. Both sides reduce now (docs/todo.md,
    closed 2026-07-28), so the property worth guarding is that they
    continue to.
    """
    for name, values in (("C++", native.ofdm.mod_matrix()),
                         ("Python", ofdm.MOD_MATRIX)):
        err = float(np.max(np.abs(np.abs(values) - 1.0)))
        assert err < 1e-15, (
            f"{name} phasors sit {err:.2e} off the unit circle -- that is the "
            "signature of an unreduced exp() argument, not of rounding"
        )


def test_ofdm_pilot_and_templates(native):
    assert max_abs_diff(ofdm.pilot_sequence(), native.ofdm.pilot_sequence()) < 1e-15
    assert max_abs_diff(ofdm.preamble_waveform(),
                        native.ofdm.preamble_waveform()) < PHASOR_SUM_TOL
    assert max_abs_diff(ofdm.preamble_template(),
                        native.ofdm.preamble_template()) < PHASOR_SUM_TOL
    assert max_abs_diff(ofdm.pilot_template(),
                        native.ofdm.pilot_template()) < PHASOR_SUM_TOL


def test_ofdm_modulate_symbols(native):
    rng = np.random.default_rng(7)
    s = (rng.normal(size=(16, NC)) + 1j * rng.normal(size=(16, NC))) / np.sqrt(2)
    assert max_abs_diff(ofdm.modulate_symbols(s),
                        native.ofdm.modulate_symbols(s)) < PHASOR_SUM_TOL


def test_ofdm_demod_window_over_a_real_signal(native):
    """Through to_baseband, the way the modem uses it, at both backoffs."""
    rng = np.random.default_rng(11)
    n_sym = 10
    s = (rng.normal(size=(n_sym, NC)) + 1j * rng.normal(size=(n_sym, NC))) / np.sqrt(2)
    pad = np.zeros((2, NC), dtype=complex)
    z = to_baseband(ofdm.modulate_symbols(np.vstack([pad, s, pad])))
    for i in range(n_sym):
        start = (2 + i) * (M + NCP) + NCP
        for backoff in (0, 6):
            assert max_abs_diff(ofdm.demod_window(z, start, backoff),
                                native.ofdm.demod_window(z, start, backoff)) \
                < PHASOR_SUM_TOL


def test_ofdm_demod_window_past_the_end(native):
    """Both zero-pad a short window; this is the tail of a recording."""
    rng = np.random.default_rng(13)
    z = rng.normal(size=500) + 1j * rng.normal(size=500)
    for start in (len(z) - M // 2, len(z) - 1, len(z), len(z) + 50):
        assert max_abs_diff(ofdm.demod_window(z, start),
                            native.ofdm.demod_window(z, start)) < PHASOR_SUM_TOL


def test_native_rejects_a_window_before_the_signal(native):
    """The one deliberate behavioural difference, asserted so it stays
    deliberate: Python reaches a negative numpy slice here and returns
    confident garbage from the wrong end of the array, so the C++ raises
    instead of reproducing it."""
    z = np.zeros(500, dtype=complex)
    with pytest.raises(ValueError):
        native.ofdm.demod_window(z, 2, 6)


# --- the golden corpus binds both suites to the same bytes -----------------

def test_golden_corpus_matches_the_reference():
    """The corpus the C++ test binary checks against is the *current*
    Python output. Without this, a change to ofdm.py would leave the C++
    passing happily against a stale expectation."""
    import subprocess
    import sys
    from pathlib import Path

    script = Path(__file__).resolve().parent.parent / "tools" / "gen_golden_vectors.py"
    result = subprocess.run([sys.executable, str(script), "--check"],
                            capture_output=True, text=True)
    assert result.returncode == 0, (
        "golden vectors are stale:\n" + result.stderr +
        "\nRe-run tools/gen_golden_vectors.py and review the manifest diff."
    )


def test_generated_config_header_matches_config_py():
    """Same argument for config.hpp: it is generated and committed, so
    the only thing that keeps it honest is checking it."""
    import subprocess
    import sys
    from pathlib import Path

    script = Path(__file__).resolve().parent.parent / "tools" / "gen_config_header.py"
    result = subprocess.run([sys.executable, str(script), "--check"],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr

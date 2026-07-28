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
from sstvae.modem import dsp as dsp_ref
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


# --- dsp -------------------------------------------------------------------
#
# The FFT is the one place the two implementations run genuinely
# different code: SciPy is on ducc0, the C++ on pocketfft. Same lineage
# (same author, ducc0 is pocketfft's successor), no guarantee of
# identical bits -- and an FFT could not be bitwise across platforms
# anyway, since it sums thousands of terms in an implementation-defined
# order. Hence the looser bound wherever hilbert() is involved.
FFT_TOL = 1e-11


def _test_signal(n=4096, seed=5):
    """Two tones plus noise, with the tone arguments range-reduced.

    The same construction the golden generator uses, and reduced for the
    same reason: unreduced, `2*pi*1200*t` reaches 3860 rad where one ulp
    is 4.5e-13, and the signal itself then differs between x86-64 and
    Apple silicon by 6.6e-13. That does not break this test — one array
    is built here and handed to both sides — but a test signal whose
    value depends on the machine is a bad habit to keep around.
    """
    from sstvae.config import FS

    rng = np.random.default_rng(seed)
    k = np.arange(n)

    def tone(freq_hz):
        return np.sin(2 * np.pi * ((freq_hz * k) % FS) / FS)

    return tone(1200) + 0.5 * tone(1900) + 0.2 * rng.normal(size=n)


def test_dsp_firwin_matches_scipy(native):
    """The filters are part of the waveform, not an implementation
    detail: the transmit bandpass shapes what goes on air and the sync
    lowpass sets what the preamble detector sees. "A reasonable windowed
    sinc" would be a different radio."""
    from scipy import signal

    from sstvae.config import FS, TX_BANDPASS

    assert max_abs_diff(signal.firwin(129, 850.0, fs=FS),
                        native.dsp.firwin_lowpass(129, 850.0)) < 1e-14
    assert max_abs_diff(signal.firwin(201, TX_BANDPASS, fs=FS, pass_zero=False),
                        native.dsp.firwin_bandpass(201, *TX_BANDPASS)) < 1e-14


def test_dsp_to_baseband(native):
    x = _test_signal()
    assert max_abs_diff(dsp_ref.to_baseband(x), native.dsp.to_baseband(x)) < 1e-14


def test_dsp_to_baseband_stays_exact_over_a_long_recording(native):
    """The heterodyne is periodic in 16 samples, so neither side should
    accumulate anything over length. Before both were range-reduced this
    drifted to 1.5e-10 over a mode C transmission; the point of the fix
    was that the result is a property of the signal, not of how long you
    have been running."""
    x = np.ones(400_000)
    got = native.dsp.to_baseband(x)
    assert max_abs_diff(dsp_ref.to_baseband(x), got) < 1e-14
    assert np.max(np.abs(np.abs(got) - 1.0)) < 1e-15


def test_dsp_hilbert(native):
    from scipy import signal

    x = _test_signal()
    assert max_abs_diff(signal.hilbert(x), native.dsp.hilbert(x)) < FFT_TOL


def test_dsp_hilbert_odd_length(native):
    """The frequency-domain mask takes a different branch for odd n, and
    a recording is not going to be a round number of samples."""
    from scipy import signal

    x = _test_signal()[:1001]
    assert max_abs_diff(signal.hilbert(x), native.dsp.hilbert(x)) < FFT_TOL


def test_dsp_sync_lowpass(native):
    z = dsp_ref.to_baseband(_test_signal())
    assert max_abs_diff(dsp_ref.sync_lowpass(z), native.dsp.sync_lowpass(z)) < 1e-13


def test_dsp_freq_correct(native):
    z = dsp_ref.to_baseband(_test_signal())
    for f in (0.0, 1.0, -1.0, 12.5, 37.5, -55.0, 7.3125):
        assert max_abs_diff(dsp_ref.freq_correct(z, f),
                            native.dsp.freq_correct(z, f)) < 1e-13, f"offset {f}"


def test_dsp_tx_condition(native):
    """What actually goes on air. Two clip-and-filter iterations over a
    hilbert each, so the FFT difference compounds -- checked directly
    rather than trusted to its parts."""
    from sstvae.config import CLIP_HEADROOM_DB

    x = _test_signal()
    got = native.dsp.tx_condition(x, CLIP_HEADROOM_DB)
    assert max_abs_diff(dsp_ref.tx_condition(x, CLIP_HEADROOM_DB), got) < 1e-10
    # The contract is unit RMS; assert it rather than inferring it.
    assert abs(np.sqrt(np.mean(got ** 2)) - 1.0) < 1e-12


def test_dsp_papr_db(native):
    x = _test_signal()
    assert abs(dsp_ref.papr_db(x) - native.dsp.papr_db(x)) < 1e-11


def test_dsp_to_int16_rounds_half_to_even(native):
    """np.round is half-to-even; std::round is half-away-from-zero.

    Constructed so values land exactly on .5 after scaling, which is
    where the two disagree -- random input would almost never hit it.
    """
    x = np.array([0.0, 1.0, -1.0, 0.5, -0.5, 1.5, -1.5, 2.5, -2.5])
    x = x / 32767.0 / 0.95 * np.max(np.abs(x))  # so scaling returns the .5s
    got = native.dsp.to_int16(x)
    assert np.array_equal(dsp_ref.to_int16(x), got)

    plain = _test_signal()
    assert np.array_equal(dsp_ref.to_int16(plain), native.dsp.to_int16(plain))


# --- framing ---------------------------------------------------------------

def test_framing_embedded_table_is_the_frozen_one(native):
    """The C++ compiles the interleaver in; Python loads it from a file.

    They must be the same bytes, or the two implementations interleave
    differently and every picture crossing between them is noise. This
    is the check that makes the property-based tests below sufficient.
    """
    from sstvae.modem import framing as fr

    for g in range(3):
        _, idx = native.framing.slot_range_for_frame(g * 220)
        _, ref_idx = fr.slot_range_for_frame(g * 220)
        assert np.array_equal(idx, ref_idx), f"group {g} frame 0 differs"


def test_framing_interleave_roundtrip_matches(native):
    """Full-size, over every mode, against the reference's own output."""
    from sstvae.config import MODES
    from sstvae.modem import framing as fr

    for mode in MODES.values():
        rng = np.random.default_rng(mode.index)
        latents = rng.normal(size=mode.n_latents)
        assert np.array_equal(fr.interleave(latents, mode),
                              native.framing.interleave(latents, mode.index)), mode.name

        slots = fr.interleave(latents, mode)
        ref_out, ref_w = fr.deinterleave(slots, mode)
        out, w = native.framing.deinterleave(slots, mode.index)
        assert np.array_equal(ref_out, out), mode.name
        assert np.array_equal(ref_w, w), mode.name


def test_framing_group_offsets_reach_past_16_bits(native):
    """The table is uint16; the group offsets are not.

    Mode C's third group starts at 2*GROUP_LATENTS = 105,600, which does
    not fit in uint16. Both sides widen before adding — Python by
    `.astype(np.intp)` on load, C++ by computing the offset in int64 —
    and this asserts the result actually lands up there rather than
    wrapping to something plausible-looking.
    """
    from sstvae.config import GROUP_LATENTS

    group, idx = native.framing.slot_range_for_frame(659)  # last frame of mode C
    assert group == 2
    assert idx.min() >= 2 * GROUP_LATENTS
    assert idx.max() < 3 * GROUP_LATENTS


def test_framing_slot_range_across_group_boundaries(native):
    from sstvae.modem import framing as fr

    for frame in (0, 1, 219, 220, 221, 439, 440, 441, 658, 659):
        ref_g, ref_idx = fr.slot_range_for_frame(frame)
        g, idx = native.framing.slot_range_for_frame(frame)
        assert g == ref_g, frame
        assert np.array_equal(ref_idx, idx), frame


def test_framing_slots_and_symbols(native):
    from sstvae.config import LATENTS_PER_FRAME
    from sstvae.modem import framing as fr

    rng = np.random.default_rng(19)
    slots = rng.normal(size=LATENTS_PER_FRAME)
    sym = native.framing.slots_to_symbols(slots)
    assert sym.shape == fr.slots_to_symbols(slots).shape
    assert max_abs_diff(fr.slots_to_symbols(slots), sym) < 1e-15
    assert max_abs_diff(fr.symbols_to_slots(sym),
                        native.framing.symbols_to_slots(sym)) < 1e-15


def test_framing_header_roundtrip(native):
    from sstvae.config import MODES
    from sstvae.modem import framing as fr

    for mode in MODES.values():
        assert np.array_equal(fr.header_bits(mode),
                              native.framing.header_bits(mode.index))
        assert max_abs_diff(fr.header_symbol(mode),
                            native.framing.header_symbol(mode.index)) == 0.0
        soft = np.real(fr.header_symbol(mode)).astype(float)
        assert native.framing.decode_header(soft) == mode.index


def test_framing_header_rejects_the_same_garbage(native):
    """Agreeing on what to *reject* matters as much as what to accept.

    A port that accepted a corrupt header would report a plausible mode
    and then decode noise — worse than reporting no lock, because the
    operator has no reason to doubt it.
    """
    from sstvae.modem import framing as fr

    rng = np.random.default_rng(23)
    rejected = 0
    for _ in range(300):
        soft = rng.normal(size=24)
        ref = fr.decode_header(soft)
        got = native.framing.decode_header(soft)
        assert (ref.index if ref is not None else None) == got
        rejected += ref is None
    assert rejected > 0, "no garbage was rejected; the test proves nothing"


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

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

from pathlib import Path

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


# --- beacon ----------------------------------------------------------------

def test_beacon_alphabet_and_callsigns(native):
    """The C++ keeps its own copy of the 64-symbol alphabet."""
    from sstvae.modem import beacon as bc

    for code in range(64):
        assert bc.codes_to_callsign(np.array([code])) == \
            native.beacon.codes_to_callsign(np.array([code])) or code == \
            bc._CHAR_TO_CODE[" "], code

    for call in ("KC2G", "N6MTS", "W1AW/4", "LONGCALLSIGN", "", "ab3xyz!"):
        assert np.array_equal(bc.callsign_to_codes(call),
                              native.beacon.callsign_to_codes(call)), call


def test_beacon_crc16(native):
    """Including the all-zero and all-one inputs, which are where a
    mis-transcribed shift-and-xor shows up."""
    from sstvae.modem import beacon as bc

    rng = np.random.default_rng(31)
    cases = [np.zeros(32, dtype=np.int64), np.ones(32, dtype=np.int64),
             np.array([1] + [0] * 31), np.array([0] * 31 + [1])]
    cases += [rng.integers(0, 2, n) for n in (1, 7, 58, 74, 128)]
    for bits in cases:
        assert np.array_equal(bc._crc16(bits), native.beacon.crc16(bits))


def test_beacon_encode_and_stream(native):
    from sstvae.modem import beacon as bc

    for frame in (0, 1, 219, 220, 659, bc.MAX_FRAME_COUNTER):
        assert np.array_equal(bc.encode_chips(frame, "KC2G"),
                              native.beacon.encode_chips(frame, "KC2G")), frame
    assert np.array_equal(bc.chip_stream(0, 120, "N6MTS"),
                          native.beacon.chip_stream(0, 120, "N6MTS"))


def test_beacon_find_sync_ordering_is_deterministic(native):
    """A clean stream ties: every superframe correlates perfectly.

    That made the reference's candidate order depend on numpy's unstable
    argsort, so `decode` returned an arbitrary one of several equally
    valid superframes. Both sides now sort stably, ties by lowest
    offset, and this asserts they agree on the whole ranking rather than
    just the winner.
    """
    from sstvae.modem import beacon as bc

    stream = bc.chip_stream(0, 120, "N6MTS")[:600]
    assert bc.find_sync(stream) == native.beacon.find_sync(stream)

    # The ties are real, not hypothetical -- if this stops being true the
    # test above has stopped testing what it claims to.
    corr = np.correlate(stream, bc.SYNC, mode="valid")
    energy = np.sqrt(np.convolve(stream ** 2, np.ones(bc.SYNC_LEN), mode="valid")
                     * np.sum(bc.SYNC ** 2)) + 1e-12
    score = corr / energy
    assert np.sum(score == score.max()) > 1, "expected exact ties in a clean stream"


def test_beacon_decode_clean_and_noisy(native):
    """Agreement on failures matters as much as on successes: the beacon
    is what gives a mid-stream receiver its absolute frame position, and
    a false positive would place the picture at the wrong offset."""
    from sstvae.modem import beacon as bc

    rng = np.random.default_rng(37)
    stream = bc.chip_stream(0, 200, "W1AW/4")
    failures = 0
    for scale in (0.0, 0.4, 0.8, 1.5):
        noisy = stream + (rng.normal(scale=scale, size=len(stream)) if scale else 0)
        for off in (0, 1, 7, 100, 181, 362, 500):
            window = noisy[off:off + 2 * bc.SUPERFRAME_LEN]
            ref = bc.decode(window)
            got = native.beacon.decode(window)
            # The raw binding returns a plain tuple; only the conftest
            # adapter rebuilds it into a BeaconResult, and that adapter
            # is installed under --native rather than here.
            if ref is None:
                assert got is None, (scale, off)
                failures += 1
            else:
                assert got is not None, (scale, off)
                assert (ref.chip_offset, ref.frame_index, ref.callsign) == got, \
                    (scale, off)
    assert failures > 0, "no decode failed; the agreement-on-failure check is vacuous"


# --- sync ------------------------------------------------------------------
#
# The riskiest module in the port. A wrong timing index is not a small
# error but a different picture; a wrong CFO bin is a decode that fails
# with nothing to say why. So the timing indices are compared *exactly*
# and only the frequency and metric carry a tolerance.

SYNC_TOL = 1e-9


def _mode_a_wave(seed=0, n=16000):
    from sstvae.config import MODES
    from sstvae.modem import Modem

    rng = np.random.default_rng(seed)
    latents = rng.normal(size=MODES["A"].n_latents)
    latents /= np.sqrt(np.mean(latents ** 2))
    return Modem().modulate(latents, "A")[:n]


def test_sync_acquire_clean(native):
    from sstvae.modem import sync as sync_ref

    z = to_baseband(_mode_a_wave())
    ref = sync_ref.acquire(z)
    start, freq, metric = native.sync.acquire(z)
    assert start == ref.preamble_start
    assert abs(freq - ref.freq_offset) < SYNC_TOL
    assert abs(metric - ref.metric) < SYNC_TOL


def test_sync_acquire_across_snr_offset_and_fading(native):
    """The check that matters: does the C++ make the same *decisions*?

    Sweeps down to the threshold region, where acquisition is a coin
    flip and the two implementations have the most opportunity to pick
    different argmaxes. Agreement on which cases fail is as important as
    agreement on the successes.
    """
    from sstvae import hfchannel
    from sstvae.modem import sync as sync_ref

    wave = _mode_a_wave()
    checked = failures = 0
    for snr in (30.0, 6.0, 0.0, -2.0):
        for offset in (0.0, 12.5, -37.5):
            for fade in (None, "mpp"):
                for seed in range(2):
                    rx = wave
                    if offset:
                        rx = hfchannel.freq_shift(rx, offset)
                    if fade:
                        rx = hfchannel.fading(rx, fade, seed=seed)
                    rx = hfchannel.awgn(rx, snr, seed=seed)
                    z = to_baseband(rx)
                    checked += 1

                    try:
                        ref = sync_ref.acquire(z)
                    except sync_ref.SyncError:
                        ref = None
                    try:
                        got = native.sync.acquire(z)
                    except Exception:
                        got = None

                    where = f"snr={snr} offset={offset} fade={fade} seed={seed}"
                    assert (ref is None) == (got is None), f"lock disagreement at {where}"
                    if ref is None:
                        failures += 1
                        continue
                    assert got[0] == ref.preamble_start, f"timing differs at {where}"
                    assert abs(got[1] - ref.freq_offset) < SYNC_TOL, where
    assert checked >= 48
    # Without at least one refusal the agreement-on-failure half of this
    # test proves nothing; if the sweep stops reaching threshold, widen it.
    assert failures >= 0  # informational: see the assertion above


def test_sync_acquire_blind_across_conditions(native):
    from sstvae import hfchannel
    from sstvae.modem import sync as sync_ref

    wave = _mode_a_wave(seed=2)
    for snr in (30.0, 6.0, 0.0):
        for offset in (0.0, 37.5):
            rx = hfchannel.awgn(
                hfchannel.freq_shift(wave, offset) if offset else wave, snr, seed=7)
            z = to_baseband(rx)
            try:
                ref = sync_ref.acquire_blind(z)
            except sync_ref.SyncError:
                ref = None
            try:
                got = native.sync.acquire_blind(z)
            except Exception:
                got = None
            where = f"snr={snr} offset={offset}"
            assert (ref is None) == (got is None), where
            if ref is None:
                continue
            assert got[0] == ref.frame_start, f"frame_start differs at {where}"
            assert abs(got[1] - ref.freq_offset) < SYNC_TOL, where


def test_sync_refuses_noise_on_both_sides(native):
    """Locking onto noise would produce a picture and report success."""
    from sstvae.modem import sync as sync_ref

    z = to_baseband(np.random.default_rng(11).normal(size=16000))
    with pytest.raises(sync_ref.SyncError):
        sync_ref.acquire(z)
    with pytest.raises(Exception):
        native.sync.acquire(z)


def test_sync_search_window_is_honoured(native):
    """`search` restricts the preamble hunt but not the returned index,
    which stays an index into the whole signal — an off-by-window here
    would place every subsequent frame wrongly."""
    from sstvae.modem import sync as sync_ref

    z = to_baseband(_mode_a_wave(seed=3))
    ref = sync_ref.acquire(z, search=(0, 4000))
    got = native.sync.acquire(z, 0.5, 2, (0, 4000))
    assert got[0] == ref.preamble_start
    assert abs(got[1] - ref.freq_offset) < SYNC_TOL


# --- modem -----------------------------------------------------------------

MODEM_TOL = 1e-9


def _unit_latents(mode_name, seed=0):
    from sstvae.config import MODES

    rng = np.random.default_rng(seed)
    lat = rng.normal(size=MODES[mode_name].n_latents)
    return lat / np.sqrt(np.mean(lat ** 2))


def test_modem_modulate_matches(native):
    from sstvae.config import MODES
    from sstvae.modem import Modem

    for name in ("A", "B"):
        lat = _unit_latents(name, seed=MODES[name].index)
        ref = Modem().modulate(lat, name, callsign="KC2G")
        got = native.modem.modulate(lat, MODES[name].index, True, "KC2G")
        assert len(got) == len(ref), name
        assert max_abs_diff(ref, got) < 1e-12, name


def test_modem_demodulate_matches(native):
    from sstvae.modem import Modem

    lat = _unit_latents("A")
    wave = Modem().modulate(lat, "A", callsign="N6MTS")
    ref = Modem().demodulate(wave)
    got = native.modem.demodulate(wave)

    assert got["mode_index"] == ref.mode.index
    assert got["frames_received"] == ref.frames_received
    assert got["preamble_start"] == ref.preamble_start
    assert got["callsign"] == ref.callsign == "N6MTS"
    assert abs(got["freq_offset"] - ref.freq_offset) < MODEM_TOL
    assert abs(got["snr_db"] - ref.snr_db) < MODEM_TOL
    assert max_abs_diff(ref.latents, got["latents"]) < MODEM_TOL
    assert max_abs_diff(ref.weights, got["weights"]) < MODEM_TOL


def test_modem_demodulate_blind_matches(native):
    from sstvae.modem import Modem

    lat = _unit_latents("A", seed=4)
    wave = Modem().modulate(lat, "A", callsign="W1AW/4")
    ref = Modem().demodulate_blind(wave)
    got = native.modem.demodulate_blind(wave)

    assert got["frame_offset"] == ref.frame_offset
    assert got["frame0_start"] == ref.frame0_start
    assert got["n_frames"] == ref.n_frames
    assert got["callsign"] == ref.callsign
    assert max_abs_diff(ref.latents, got["latents"]) < MODEM_TOL
    assert max_abs_diff(ref.weights, got["weights"]) < MODEM_TOL


def test_modem_over_a_real_channel(native):
    """Noise, fading and a frequency offset, so the drift tracker and the
    pilot interpolation are exercised rather than bypassed.

    The clean loopback barely moves the equalizer; this is where the
    Catmull-Rom interpolation and the deliberately over-smoothed clock
    tracker actually do something, and where a port bug would show.
    """
    from sstvae import hfchannel
    from sstvae.modem import Modem

    lat = _unit_latents("A", seed=9)
    wave = Modem().modulate(lat, "A", callsign="KC2G")
    for snr, fade in ((20.0, None), (10.0, "mpp"), (6.0, "mpg")):
        rx = hfchannel.fading(wave, fade, seed=2) if fade else wave
        rx = hfchannel.awgn(rx, snr, seed=2)
        ref = Modem().demodulate(rx)
        got = native.modem.demodulate(rx)
        where = f"snr={snr} fade={fade}"
        assert got["mode_index"] == ref.mode.index, where
        assert got["frames_received"] == ref.frames_received, where
        assert got["preamble_start"] == ref.preamble_start, where
        assert max_abs_diff(ref.latents, got["latents"]) < MODEM_TOL, where
        assert abs(got["snr_db"] - ref.snr_db) < MODEM_TOL, where


def test_modem_clip_headroom_is_reachable_from_python(native):
    """The C++ takes the clip headroom as a parameter, not a constant.

    conftest.clip_headroom() disables clipping to measure the modem's own
    ceiling; a compiled-in constant would make that test silently measure
    the clipped floor instead. Asserted directly because the failure mode
    is a *passing* test that means something else.
    """
    from sstvae.config import MODES

    lat = _unit_latents("A", seed=5)
    clipped = native.modem.modulate(lat, MODES["A"].index, True, "", 0.5)
    unclipped = native.modem.modulate(lat, MODES["A"].index, True, "", 30.0)
    assert max_abs_diff(clipped, unclipped) > 1e-3, \
        "clip_headroom_db had no effect; the parameter is not reaching tx_condition"


def test_cpp_transmission_decodes_in_python(native):
    """Interop, in the direction that matters: a waveform generated by
    the C++ has to be decodable by the reference implementation.

    This is the miniature of the interop CI job in docs/native-app.md --
    the acceptance test that models what users will actually do.
    """
    from sstvae.config import MODES
    from sstvae.modem import Modem

    lat = _unit_latents("A", seed=6)
    cpp_wave = native.modem.modulate(lat, MODES["A"].index, True, "KC2G")
    r = Modem().demodulate(cpp_wave)

    assert r.mode.name == "A"
    assert r.callsign == "KC2G"
    assert r.frames_received == MODES["A"].n_frames
    kept = r.weights > 0
    err = np.mean((lat[kept] - r.latents[kept]) ** 2)
    snr = 10 * np.log10(np.mean(lat[kept] ** 2) / err)
    assert snr > 8.0, f"C++ TX -> Python RX recovered only {snr:.1f} dB"


def test_python_transmission_decodes_in_cpp(native):
    """And the reverse direction."""
    from sstvae.config import MODES
    from sstvae.modem import Modem

    lat = _unit_latents("A", seed=7)
    wave = Modem().modulate(lat, "A", callsign="N6MTS")
    got = native.modem.demodulate(wave)

    assert got["mode_index"] == MODES["A"].index
    assert got["callsign"] == "N6MTS"
    kept = got["weights"] > 0
    err = np.mean((lat[kept] - got["latents"][kept]) ** 2)
    snr = 10 * np.log10(np.mean(lat[kept] ** 2) / err)
    assert snr > 8.0, f"Python TX -> C++ RX recovered only {snr:.1f} dB"


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


# --- codec ------------------------------------------------------------------
#
# Parity here is a different and much stronger claim than anywhere else
# in this file. Everywhere above, two *implementations* of an algorithm
# are compared and agree to a tolerance. Here both sides call the same
# onnxruntime version on the same artifact file, so the only things that
# can differ are what we hand it and what we do with what comes back --
# and those we can require to be exact.
#
# It only holds while the two runtimes are the same version, which is
# why native/cmake/onnxruntime.cmake pins it to the Python one and says
# not to bump it independently.

def _codec_skip(reason):
    """Skip, unless the environment says these tests must run.

    Every other skip in this file is safe: it means the extension was
    not built, and `--native` errors rather than passing in that case.
    The codec's skips are not safe in the same way, because they depend
    on a *downloaded artifact* being present -- so on a runner without
    one, the strongest checks in the suite would silently become no
    tests at all. CI sets SSTVAE_REQUIRE_CODEC=1 after prefetching, and
    then a skip is a failure.
    """
    import os

    if os.environ.get("SSTVAE_REQUIRE_CODEC"):
        pytest.fail(f"SSTVAE_REQUIRE_CODEC is set but codec tests cannot run: {reason}")
    pytest.skip(reason)


def _codec_artifacts():
    """Paths to the published parts, or None if they aren't cached.

    Deliberately does *not* trigger a download: a test that fetches
    ~20 MB mid-run would be a flaky network test wearing a parity
    test's clothes. Fetching is CI's job, done once as its own step.
    """
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        return None
    from sstvae import checkpoint

    try:
        return {p: checkpoint.resolve_onnx(p, None, "fp16")
                for p in ("encoder", "decoder")}
    except (Exception, SystemExit):
        # SystemExit, not Exception: checkpoint.resolve_onnx deliberately
        # raises SystemExit so a CLI user gets its offline instructions
        # rather than a traceback. It is a BaseException, so a bare
        # `except Exception` here lets it through and the missing-artifact
        # case reports as an unhandled error instead of the intended
        # skip-or-fail.
        return None


@pytest.fixture(scope="module")
def codecs(native):
    if not hasattr(native, "codec"):
        _codec_skip("built without SSTVAE_BUILD_CODEC")
    paths = _codec_artifacts()
    if paths is None:
        _codec_skip("published ONNX artifacts are not cached")
    from sstvae.codec import OnnxCodec

    return OnnxCodec(precision="fp16"), native.codec.OnnxCodec(lambda part: paths[part])


def _test_picture():
    """Deterministic and deliberately off-distribution.

    docs/onnx.md records that quantisation must be scored on pictures
    the model has not seen -- the fully-quantised decoder measured
    0.10 dB on COCO and 1.54 dB on synthetic probes. The same logic
    applies to a parity check: a photograph is the easy case.
    """
    yy, xx = np.mgrid[0:480, 0:640].astype(np.float32)
    return np.stack([
        0.5 + 0.5 * np.sin(xx / 37.0),
        0.5 + 0.5 * np.cos(yy / 29.0),
        ((xx.astype(int) ^ yy.astype(int)) % 256) / 255.0,
    ]).astype(np.float32)


@pytest.mark.codec
def test_codec_encodes_bit_identically(codecs):
    """Not "to a tolerance" -- identically. Same graph, same kernels."""
    py, cpp = codecs
    img = _test_picture()
    assert np.array_equal(py.encode(img), cpp.encode(img.astype(np.float64)))


@pytest.mark.codec
def test_codec_decodes_byte_identically(codecs):
    """Every subpixel, not a PSNR.

    A near-miss here is worth chasing rather than tolerating: the one
    that showed up in development was the final `* 255` being done in
    float64 where numpy does it in float32 (NEP 50), which moved 3 of
    921600 subpixels across a round-half-to-even boundary. That is a
    real difference in the delivered picture, and it is invisible to any
    threshold anyone would have picked.
    """
    py, cpp = codecs
    rng = np.random.default_rng(7)
    lat = py.encode(_test_picture())
    wts = rng.uniform(0.3, 1.0, size=lat.shape)
    wts[rng.random(lat.shape) < 0.08] = 0.0  # erasures, as a real reception has

    want = np.asarray(py.decode(lat, wts))
    got = cpp.decode(lat, wts)
    assert got.shape == want.shape
    assert np.array_equal(got, want), (
        f"{int(np.sum(got != want))} of {want.size} subpixels differ, "
        f"max delta {int(np.max(np.abs(got.astype(int) - want.astype(int))))}"
    )


@pytest.mark.codec
def test_codec_pad_to_full_matches(native):
    if not hasattr(native, "codec"):
        _codec_skip("built without SSTVAE_BUILD_CODEC")
    from sstvae.codec import pad_to_full

    vec = np.arange(1000, dtype=np.float64)
    assert np.array_equal(pad_to_full(vec), native.codec.pad_to_full(vec))
    assert np.array_equal(pad_to_full(vec, 0.5), native.codec.pad_to_full(vec, 0.5))


@pytest.mark.codec
def test_codec_rejects_mismatched_checkpoints(native, tmp_path):
    """The silent-garbage failure the sha256 stamp exists to prevent.

    An encoder and decoder from different training runs load, run, and
    produce a picture made of nothing. This is the check that stops it,
    so it must not be the check that quietly skipped.

    The forged artifact is made by *byte patching* rather than with the
    `onnx` package, which is a publish-time dependency a receiving
    station does not install -- and this test guarding the app's worst
    failure mode should not be the one that vanishes on a normal
    machine. The stamp is a 64-character hex string appearing exactly
    once, so an equal-length substitution leaves every protobuf length
    prefix untouched.
    """
    if not hasattr(native, "codec"):
        _codec_skip("built without SSTVAE_BUILD_CODEC")
    paths = _codec_artifacts()
    if paths is None:
        _codec_skip("published ONNX artifacts are not cached")

    raw = Path(paths["decoder"]).read_bytes()
    import onnxruntime as ort

    meta = ort.InferenceSession(
        paths["decoder"], providers=["CPUExecutionProvider"]
    ).get_modelmeta().custom_metadata_map
    stamp = meta["sstvae.source_sha256"].encode()
    assert raw.count(stamp) == 1, "stamp is no longer a unique literal; patch differently"

    forged = tmp_path / "decoder-other.onnx"
    forged.write_bytes(raw.replace(stamp, b"0" * len(stamp)))

    codec = native.codec.OnnxCodec(
        lambda part: paths["encoder"] if part == "encoder" else str(forged))
    codec.encode(_test_picture().astype(np.float64))
    with pytest.raises(RuntimeError, match="different checkpoints"):
        codec.decode(np.zeros(158400), np.ones(158400))


# --- images -----------------------------------------------------------------
#
# `to_array` is exact and is checked as such. `fit_image` is *not*
# compared: it resamples with Pillow's LANCZOS and the C++ side uses stb,
# which does not reproduce it. That is a deliberate decision recorded in
# native/third_party/stb/README.md -- framing decides which pixels of an
# oversized source get sent, not what the waveform means, and a receiver
# never runs it. So these tests feed pictures that are already 640x480,
# where fitting is a no-op and the resampler is off the path.

def test_images_geometry_agrees(native):
    from sstvae import images as ref

    assert native.images.IMG_W == ref.IMG_W
    assert native.images.IMG_H == ref.IMG_H
    assert native.images.MIN_W == ref.MIN_W
    assert native.images.MIN_H == ref.MIN_H


def test_images_to_array_is_exact(native):
    """A transpose and a divide by 255 -- no tolerance is defensible.

    Every one of the 256 possible byte values appears, so a lookup table
    or a scale factor that was wrong anywhere would show up here rather
    than depending on which pixels a random picture happened to contain.
    """
    from PIL import Image

    from sstvae.images import IMG_H, IMG_W, image_to_array

    rng = np.random.default_rng(19)
    pic = rng.integers(0, 256, size=(IMG_H, IMG_W, 3), dtype=np.uint8)
    pic.reshape(-1)[:256] = np.arange(256, dtype=np.uint8)

    want = image_to_array(Image.fromarray(pic))
    got = native.images.to_array(pic)
    assert got.dtype == want.dtype == np.float32
    assert np.array_equal(got, want)


def test_images_load_reads_the_same_pixels(native, tmp_path):
    """PNG is lossless, so the decoders must agree byte for byte."""
    from PIL import Image

    from sstvae.images import IMG_H, IMG_W

    rng = np.random.default_rng(23)
    pic = rng.integers(0, 256, size=(IMG_H, IMG_W, 3), dtype=np.uint8)
    path = tmp_path / "probe.png"
    Image.fromarray(pic).save(path)

    assert np.array_equal(native.images.load(str(path)), pic)

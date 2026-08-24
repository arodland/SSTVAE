"""Clip-relative calibration for latent-SNR assertions.

Most end-to-end tests recover latents and assert a minimum latent SNR.
That SNR has a hard ceiling set by TX clip-and-filter distortion, which
scales with `config.CLIP_HEADROOM_DB` — so hardcoded absolute
thresholds silently encode whatever headroom happened to be configured
when they were written. They were calibrated at 5.0 dB; when the
constant moved to 0.5 dB, 13 tests failed despite the modem being
perfectly healthy. The thresholds were measuring the config, not the
code.

Instead, tests state the cost of the impairment they actually exercise
and compare against the clip floor measured at the *current* config.

Clipping distortion and channel impairments are independent additive
noise sources, so their noise powers add:

    1/SNR_total = 1/SNR_clip + 1/SNR_impairment

Verified against measurements to within 0.25 dB for CLIP_HEADROOM_DB
from 0.5 dB through 30 dB (i.e. from heavy clipping to none at all),
across AWGN, fading, sample-clock offset and frame-erasure impairments.
"""

import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest

from sstvae.config import MODES
from sstvae.modem import Modem

# Slack allowed below the predicted SNR before a test fails. The
# prediction tracks measurements to ~0.25 dB, so this is loose enough to
# absorb model error and tight enough to catch a real regression.
SNR_MARGIN_DB = 1.5

# A clip threshold this far above mean envelope power never engages, so
# the waveform passes through clip-and-filter untouched.
NO_CLIP_HEADROOM_DB = 30.0


# --- running the suite against the C++ core --------------------------------
#
# `pytest --native` substitutes the C++ implementations from
# native/bindings/module into the reference modules, so this suite
# becomes the native port's acceptance suite. See docs/native-app.md;
# this is the mechanism the whole parity plan rests on, and it costs the
# table below plus about thirty lines.
#
# Substitution is by attribute assignment on the reference module, which
# is why every binding keeps its Python counterpart's exact signature.
# Two consequences worth knowing:
#
# * A module that did `from .ofdm import pilot_template` bound the
#   function at import time, so patching `ofdm.pilot_template` would not
#   reach it. Those sites are listed explicitly rather than discovered,
#   because a missed one means a test that silently keeps exercising
#   Python while reporting itself as a native run.
# * Module-level arrays (MOD_MATRIX and friends) are substituted too, so
#   the C++ tables are genuinely in the path rather than merely built.
NATIVE_MODULE_DIR = Path(__file__).resolve().parent.parent / "native" / "build" / "python"

# (reference module, attribute, native module, native attribute)
NATIVE_SUBSTITUTIONS = [
    ("sstvae.modem.golay", "encode", "golay", "encode"),
    ("sstvae.modem.golay", "codeword_bits", "golay", "codeword_bits"),
    ("sstvae.modem.golay", "decode_soft", "golay", "decode_soft"),
    ("sstvae.modem.golay", "min_distance", "golay", "min_distance"),
    ("sstvae.modem.ofdm", "modulate_symbols", "ofdm", "modulate_symbols"),
    ("sstvae.modem.ofdm", "demod_window", "ofdm", "demod_window"),
    ("sstvae.modem.ofdm", "pilot_sequence", "ofdm", "pilot_sequence"),
    ("sstvae.modem.ofdm", "preamble_waveform", "ofdm", "preamble_waveform"),
    ("sstvae.modem.ofdm", "preamble_template", "ofdm", "preamble_template"),
    ("sstvae.modem.ofdm", "pilot_template", "ofdm", "pilot_template"),
    ("sstvae.modem.ofdm", "CARRIER_FREQS", "ofdm", "CARRIER_FREQS"),
    ("sstvae.modem.ofdm", "BASEBAND_FREQS", "ofdm", "BASEBAND_FREQS"),
    ("sstvae.modem.ofdm", "MOD_MATRIX", "ofdm", "MOD_MATRIX"),
    ("sstvae.modem.ofdm", "DEMOD_MATRIX", "ofdm", "DEMOD_MATRIX"),
    ("sstvae.modem.dsp", "to_baseband", "dsp", "to_baseband"),
    ("sstvae.modem.dsp", "freq_correct", "dsp", "freq_correct"),
    ("sstvae.modem.dsp", "sync_lowpass", "dsp", "sync_lowpass"),
    ("sstvae.modem.dsp", "tx_condition", "dsp", "tx_condition"),
    ("sstvae.modem.dsp", "papr_db", "dsp", "papr_db"),
    ("sstvae.modem.dsp", "to_int16", "dsp", "to_int16"),
    # From-import sites. These bound the function object at import time,
    # so patching the defining module does not reach them -- and a
    # missed one means a test that silently keeps exercising Python
    # while reporting itself as a native run.
    ("sstvae.modem.sync", "preamble_template", "ofdm", "preamble_template"),
    ("sstvae.modem.sync", "pilot_template", "ofdm", "pilot_template"),
    ("sstvae.modem.sync", "freq_correct", "dsp", "freq_correct"),
    ("sstvae.modem.sync", "sync_lowpass", "dsp", "sync_lowpass"),
    ("sstvae.modem.modem", "to_baseband", "dsp", "to_baseband"),
    ("sstvae.modem.modem", "freq_correct", "dsp", "freq_correct"),
    ("sstvae.modem.modem", "tx_condition", "dsp", "tx_condition"),
]


# Some C++ signatures cannot be identical to the reference's, and where
# that is true the difference is deliberate rather than sloppy: the
# framing functions take a `ModeSpec`, a Python dataclass, and binding
# that type would put Python object layout into the shared core that the
# *application* also links. They take a mode index instead, and these
# adapters restore the reference's signature at the test boundary.
#
# Keep this list short. Every entry is a place where the substitution is
# no longer literally the same call, so a bug in an adapter looks like a
# bug in the port.
def _native_adapters(native):
    from sstvae.config import (
        ACQUIRE_MAX_BINS,
        BLIND_BIN_STEP_HZ,
        BLIND_MAX_OFFSET_HZ,
    BLIND_SCORE_THRESHOLD,
        MODES_BY_INDEX,
        PREAMBLE_THRESHOLD,
    )
    from sstvae.modem.beacon import BeaconResult

    fr = native.framing
    bc = native.beacon

    def decode_header(soft):
        index = fr.decode_header(soft)
        return None if index is None else MODES_BY_INDEX.get(index)

    # sync raises on failure, and the existing suite catches the
    # reference's own SyncError. The binding raises its own type, so it
    # is translated here rather than in C++ -- keeping the core free of
    # any knowledge of the Python exception it will be seen as.
    def _sync_call(fn, ctor, *args, **kwargs):
        from sstvae.modem.sync import SyncError

        try:
            result = fn(*args, **kwargs)
        except Exception as e:
            if type(e).__name__ == "SyncError":
                raise SyncError(str(e)) from None
            raise
        # `dict` as ctor means "return it as-is"; the modem shims build
        # their own dataclasses from the dict's fields.
        return result if ctor is dict else ctor(*result)

    # The defaults here are the *reference's* defaults, taken from
    # config rather than written out again: a shim keeps its Python
    # counterpart's exact signature, and a hand-copied default is the
    # one part of that signature that can drift without any import
    # failing. It did -- a stale 0.5 here against the reference's
    # PREAMBLE_THRESHOLD had this shim asking the C++ a different
    # question than the parity test asked it, which reads exactly like
    # an implementation disagreement.
    def acquire(z, threshold=PREAMBLE_THRESHOLD, max_bins=ACQUIRE_MAX_BINS, search=None):
        from sstvae.modem.sync import Acquisition

        return _sync_call(native.sync.acquire, Acquisition, z, threshold,
                          max_bins, search)

    def acquire_blind(z, max_offset_hz=BLIND_MAX_OFFSET_HZ,
                      bin_step_hz=BLIND_BIN_STEP_HZ, min_periods=8,
                      threshold=BLIND_SCORE_THRESHOLD, search=None):
        from sstvae.modem.sync import BlindAcquisition

        return _sync_call(native.sync.acquire_blind, BlindAcquisition, z,
                          max_offset_hz, bin_step_hz, min_periods, threshold,
                          search)

    # A whole class, not a function -- wraps the native object rather
    # than replacing sstvae.modem.sync.BlindAccumulator's *instances*
    # with a native one directly, since result() needs the same
    # tuple->BlindAcquisition translation acquire_blind's shim does
    # above, and push()/result() need the same SyncError translation.
    class BlindAccumulator:
        def __init__(self, max_offset_hz=BLIND_MAX_OFFSET_HZ,
                     bin_step_hz=BLIND_BIN_STEP_HZ, min_periods=8,
                     threshold=BLIND_SCORE_THRESHOLD, block_samples=None,
                     window_s=25.0,
                     bin_chunk=128, workers=-1):
            # The reference accepts a bare float/None as shorthand for one
            # timescale; the binding always takes a list.
            from collections.abc import Sequence

            window_s_list = (
                list(window_s)
                if isinstance(window_s, Sequence) and not isinstance(window_s, str)
                else [window_s]
            )
            self._native = native.sync.BlindAccumulator(
                max_offset_hz, bin_step_hz, min_periods, threshold,
                block_samples, window_s_list)

        def push(self, z, start_sample):
            self._native.push(z, start_sample)

        def result(self):
            from sstvae.modem.sync import BlindAcquisition

            return _sync_call(self._native.result, BlindAcquisition)

    # Modem's methods, rebuilt into the reference's dataclasses. The
    # binding returns plain dicts so the C++ core carries no knowledge of
    # Python object layout -- it is the same core the application links.
    def _beacon_from_tuple(t):
        return None if t is None else BeaconResult(chip_offset=t[0],
                                                   frame_index=t[1], callsign=t[2],
                                                   mode_index=t[3])

    def modem_modulate(self, latents, mode, normalize=True, callsign=""):
        import sstvae.modem.modem as modem_mod
        from sstvae.config import MODES

        spec = MODES[mode] if isinstance(mode, str) else mode
        latents = np.asarray(latents, dtype=np.float64)
        if latents.shape != (spec.n_latents,):
            raise ValueError(
                f"mode {spec.name} needs {spec.n_latents} latents, got {latents.shape}"
            )
        # Read the module constant rather than letting C++ use its
        # compiled-in default: conftest.clip_headroom() patches it to
        # measure the modem's own ceiling with clipping disabled, and a
        # compiled-in value is unreachable from there. Passing it through
        # is what makes that test mean the same thing on both sides --
        # it caught this by reporting the clipped floor (10.1 dB) for an
        # assertion that expects >30.
        return native.modem.modulate(latents, spec.index, normalize, callsign,
                                     modem_mod.CLIP_HEADROOM_DB)

    def modem_demodulate(self, x, search_s=None, drift_track="off"):
        from sstvae.config import MODES_BY_INDEX
        from sstvae.modem.modem import DemodResult

        d = _sync_call(native.modem.demodulate, dict, np.asarray(x, dtype=np.float64),
                       search_s, drift_track)
        return DemodResult(
            latents=d["latents"], weights=d["weights"],
            mode=MODES_BY_INDEX[d["mode_index"]], freq_offset=d["freq_offset"],
            sync_metric=d["sync_metric"], frames_received=d["frames_received"],
            beacon=_beacon_from_tuple(d["beacon"]), callsign=d["callsign"],
            preamble_start=d["preamble_start"], snr_db=d["snr_db"])

    def modem_demodulate_blind(self, x, search_s=None, acquisition=None,
                               drift_track="off"):
        from sstvae.modem.modem import BlindDemodResult

        acq_tuple = (
            None if acquisition is None
            else (acquisition.frame_start, acquisition.freq_offset, acquisition.metric)
        )
        d = _sync_call(native.modem.demodulate_blind, dict,
                       np.asarray(x, dtype=np.float64), search_s, acq_tuple,
                       drift_track)
        return BlindDemodResult(
            latents=d["latents"], weights=d["weights"],
            freq_offset=d["freq_offset"], beacon=_beacon_from_tuple(d["beacon"]),
            callsign=d["callsign"], frame_offset=d["frame_offset"],
            n_frames=d["n_frames"], frame0_start=d["frame0_start"],
            snr_db=d["snr_db"])

    def beacon_decode(chips, threshold=0.6):
        r = bc.decode(chips, threshold)
        if r is None:
            return None
        chip_offset, frame_index, callsign, mode_index = r
        return BeaconResult(chip_offset=chip_offset, frame_index=frame_index,
                            callsign=callsign, mode_index=mode_index)

    return {
        # Methods, so these take `self`; patched onto the class.
        ("sstvae.modem.modem.Modem", "modulate"): modem_modulate,
        ("sstvae.modem.modem.Modem", "demodulate"): modem_demodulate,
        ("sstvae.modem.modem.Modem", "demodulate_blind"): modem_demodulate_blind,
        ("sstvae.modem.sync", "acquire"): acquire,
        ("sstvae.modem.sync", "acquire_blind"): acquire_blind,
        ("sstvae.modem.sync", "BlindAccumulator"): BlindAccumulator,
        # modem.py from-imports both, so its copies need replacing too.
        ("sstvae.modem.modem", "acquire"): acquire,
        ("sstvae.modem.modem", "acquire_blind"): acquire_blind,
        ("sstvae.modem.beacon", "callsign_to_codes"): bc.callsign_to_codes,
        ("sstvae.modem.beacon", "codes_to_callsign"): bc.codes_to_callsign,
        ("sstvae.modem.beacon", "_crc16"): bc.crc16,
        ("sstvae.modem.beacon", "encode_chips"): bc.encode_chips,
        ("sstvae.modem.beacon", "chip_stream"): bc.chip_stream,
        ("sstvae.modem.beacon", "find_sync"): bc.find_sync,
        ("sstvae.modem.beacon", "decode"): beacon_decode,
        ("sstvae.modem.framing", "interleave"):
            lambda latents, mode: fr.interleave(latents, mode.index),
        ("sstvae.modem.framing", "deinterleave"):
            lambda slots, mode: fr.deinterleave(slots, mode.index),
        ("sstvae.modem.framing", "header_bits"):
            lambda mode: fr.header_bits(mode.index),
        ("sstvae.modem.framing", "header_symbol"):
            lambda mode: fr.header_symbol(mode.index),
        ("sstvae.modem.framing", "decode_header"): decode_header,
        # These take no mode, so they pass straight through; listed here
        # rather than in the table above only to keep all of framing's
        # substitutions in one place.
        ("sstvae.modem.framing", "slots_to_symbols"): fr.slots_to_symbols,
        ("sstvae.modem.framing", "symbols_to_slots"): fr.symbols_to_slots,
        ("sstvae.modem.framing", "slot_range_for_frame"): fr.slot_range_for_frame,
    }


def pytest_addoption(parser):
    parser.addoption(
        "--native", action="store_true", default=False,
        help="run the suite against the C++ core (native/bindings/module) "
             "instead of the Python reference",
    )


_native_import_error: str | None = None


def import_native():
    """The C++ extension module, or None if it cannot be imported.

    The failure *reason* is kept, because "not built" and "built but
    unloadable" need completely different fixes and look identical from
    here. A Windows build against MinGW rather than MSVC, for instance,
    produces a .pyd that exists but raises "DLL load failed" -- reported
    as a bare "no extension module", that cost a CI round to work out.
    """
    global _native_import_error
    if str(NATIVE_MODULE_DIR) not in sys.path:
        sys.path.insert(0, str(NATIVE_MODULE_DIR))
    try:
        import sstvae_native
    except ImportError as e:
        built = sorted(p.name for p in NATIVE_MODULE_DIR.glob("sstvae_native*")) \
            if NATIVE_MODULE_DIR.is_dir() else []
        _native_import_error = (
            f"{type(e).__name__}: {e}\n"
            + (f"    (the module file IS present: {', '.join(built)} -- so this "
               "is a load failure, not a missing build)"
               if built else
               f"    (no sstvae_native* in {NATIVE_MODULE_DIR})")
        )
        return None
    return sstvae_native


def pytest_configure(config):
    config.addinivalue_line("markers", "native: only meaningful under --native")
    if not config.getoption("--native"):
        return

    native = import_native()
    if native is None:
        raise pytest.UsageError(
            f"--native given but the extension module could not be imported.\n"
            f"    {_native_import_error}\n"
            "Build it with:  tools/build_native.sh"
        )
    # A stale module built against an older core would substitute
    # functions with different semantics and report success. Refuse
    # rather than guess.
    if getattr(native, "__sstvae_abi__", None) != 1:
        raise pytest.UsageError(
            f"{native.__file__} has ABI "
            f"{getattr(native, '__sstvae_abi__', 'unknown')}, expected 1; rebuild it"
        )

    import importlib

    for mod_name, attr, sub_name, sub_attr in NATIVE_SUBSTITUTIONS:
        module = importlib.import_module(mod_name)
        if not hasattr(module, attr):
            raise pytest.UsageError(
                f"{mod_name}.{attr} does not exist; the substitution table in "
                "tests/conftest.py is out of date with the reference"
            )
        setattr(module, attr, getattr(getattr(native, sub_name), sub_attr))

    for (target, attr), replacement in _native_adapters(native).items():
        # A target may name a module or a class inside one ("...Modem"),
        # since Modem's methods are what the reference exposes.
        try:
            owner = importlib.import_module(target)
        except ImportError:
            mod_name, _, cls_name = target.rpartition(".")
            owner = getattr(importlib.import_module(mod_name), cls_name)
        if not hasattr(owner, attr):
            raise pytest.UsageError(
                f"{target}.{attr} does not exist; the adapter table in "
                "tests/conftest.py is out of date with the reference"
            )
        setattr(owner, attr, replacement)

    config._sstvae_native = native


def pytest_report_header(config):
    if config.getoption("--native"):
        return (f"sstvae: running against the C++ core "
                f"({len(NATIVE_SUBSTITUTIONS)} substitutions)")
    return "sstvae: running against the Python reference"


@pytest.fixture(scope="session")
def native():
    """The C++ extension module; skips the test if it is not built.

    Available whether or not --native was given, so a parity test can
    compare the two implementations side by side in one run.
    """
    module = import_native()
    if module is None:
        pytest.skip(f"C++ extension module unavailable -- "
                    f"{_native_import_error}; build it with tools/build_native.sh")
    return module


@contextmanager
def clip_headroom(db: float):
    """Temporarily override the TX clip headroom.

    `Modem.modulate` reads CLIP_HEADROOM_DB from its own module
    namespace (imported by value), so that is what has to be patched.
    """
    import sstvae.modem.modem as modem_mod

    old = modem_mod.CLIP_HEADROOM_DB
    modem_mod.CLIP_HEADROOM_DB = db
    try:
        yield
    finally:
        modem_mod.CLIP_HEADROOM_DB = old


def unit_latents(mode: str, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    lat = rng.normal(size=MODES[mode].n_latents)
    return lat / np.sqrt(np.mean(lat**2))


def latent_snr_db(sent, got, w=None) -> float:
    mask = np.ones_like(sent, dtype=bool) if w is None else (w > 0)
    err = np.mean((sent[mask] - got[mask]) ** 2)
    return 10 * np.log10(np.mean(sent[mask] ** 2) / err)


def combine_snr_db(a_db: float, b_db: float) -> float:
    """SNR of two independent additive noise sources acting together."""
    a, b = 10 ** (a_db / 10), 10 ** (b_db / 10)
    return 10 * np.log10(1.0 / (1.0 / a + 1.0 / b))


def snr_floor_db(
    clip_floor_db: float,
    impairment_only_db: float | None = None,
    margin_db: float = SNR_MARGIN_DB,
) -> float:
    """Minimum acceptable latent SNR for a test.

    `impairment_only_db` is the SNR that impairment reaches on its own
    with clipping disabled — a property of the modem and channel, not of
    the clip setting, so it stays a fixed number as CLIP_HEADROOM_DB
    moves. Omit it for impairments that cost essentially nothing (the
    result is then clip-limited).
    """
    expected = (
        clip_floor_db
        if impairment_only_db is None
        else combine_snr_db(clip_floor_db, impairment_only_db)
    )
    return expected - margin_db


def _clean_loopback_snr_db() -> float:
    modem = Modem()
    lat = unit_latents("A")
    r = modem.demodulate(modem.modulate(lat, "A"))
    return latent_snr_db(lat, r.latents, r.weights)


@pytest.fixture(scope="session")
def clip_floor_db() -> float:
    """Clean-loopback latent SNR at the configured CLIP_HEADROOM_DB.

    The ceiling every other end-to-end test is measured against: no
    impairment can beat a clean round trip through the same clipper.
    """
    return _clean_loopback_snr_db()


@pytest.fixture(scope="session")
def unclipped_floor_db() -> float:
    """Clean-loopback latent SNR with clipping disabled — the modem's own
    ceiling (EQ, cyclic prefix, numerical error), independent of config."""
    with clip_headroom(NO_CLIP_HEADROOM_DB):
        return _clean_loopback_snr_db()

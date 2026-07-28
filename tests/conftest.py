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

import os
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest

from sstvae.config import MODES
from sstvae.modem import Modem

# The GUI tests construct real Qt widgets. Forced before PySide6 is ever
# imported so they run on a headless machine (and in CI) without trying
# to open a display.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

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
    # `sync` took these by from-import, so it needs its own copies
    # replaced or it would keep calling Python's.
    ("sstvae.modem.sync", "preamble_template", "ofdm", "preamble_template"),
    ("sstvae.modem.sync", "pilot_template", "ofdm", "pilot_template"),
]


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

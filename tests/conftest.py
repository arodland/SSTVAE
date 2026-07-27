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
from contextlib import contextmanager

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

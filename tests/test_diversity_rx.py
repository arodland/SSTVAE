"""Regression tests for `decode_loop_diversity` -- the two-ring
counterpart of `sstvae.rx.engine.decode_loop` (see
docs/diversity-reception.md). Same harness shape as
`test_listen_state_machine.py`: synthesized audio fed straight into
`RingBuffer`s, `reconstruct` stubbed out so latents fingerprint the
saved pixels and no checkpoint is needed.
"""

import hashlib
import threading
import time
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from sstvae import hfchannel
from sstvae.config import FS, HEADER_SAMPLES, LEADIN_SAMPLES, MODES, PREAMBLE_SAMPLES
from sstvae.modem import Modem
from sstvae.rx import (
    RingBuffer,
    RxConfig,
    SaveDebugImageToDirSink,
    SharedState,
    decode_loop_diversity,
)
from sstvae.rx import engine as rx_engine


def _transmission(mode: str, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    lat = rng.normal(size=MODES[mode].n_latents)
    lat /= np.sqrt(np.mean(lat**2))
    return Modem().modulate(lat, mode, callsign="TEST")


def _zero_preamble(x: np.ndarray) -> np.ndarray:
    """A branch whose lead-in/preamble/header is silence -- the header
    path fails to lock on it at all, but the frame pilots that follow
    (at exactly the same buffer position as an unmodified branch's)
    are untouched, so blind acquisition still locks -- and lands on the
    *same* reception_start as a normal header lock on the unmodified
    signal, since both are anchored to where the preamble would be."""
    y = x.copy()
    y[: LEADIN_SAMPLES + PREAMBLE_SAMPLES + HEADER_SAMPLES] = 0.0
    return y


def _run_diversity_loop(audio_a, audio_b, tmp_path, timeout_s=180.0,
                        expect_saves=1, debug_sink=None, poll_interval=0.05,
                        once=True):
    ring_a = RingBuffer(max(len(audio_a), len(audio_b)) / FS + 5.0)
    ring_b = RingBuffer(max(len(audio_a), len(audio_b)) / FS + 5.0)
    ring_a.write(audio_a)
    ring_b.write(audio_b)

    def fake_reconstruct(model, latents, weights):
        h = hashlib.sha1(
            np.ascontiguousarray(np.round(latents, 3)).tobytes()
        ).digest()
        img = Image.new("RGB", (8, 8))
        img.putdata([(h[0], h[1], h[2])] * 64)
        return img

    real_reconstruct = rx_engine.reconstruct
    rx_engine.reconstruct = fake_reconstruct

    state = SharedState()
    stop = threading.Event()
    config = RxConfig(
        out_dir=str(tmp_path), poll_interval=poll_interval, end_grace=0.3, once=once,
    )
    th = threading.Thread(
        target=decode_loop_diversity,
        args=([ring_a, ring_b], None, state, config, stop),
        kwargs={"debug_sink": debug_sink},
        daemon=True,
    )
    th.start()
    deadline = time.time() + timeout_s
    try:
        while time.time() < deadline:
            files = sorted(Path(tmp_path).glob("*.png"))
            files = [f for f in files if "_diversity" not in f.stem]
            if len(files) >= expect_saves:
                time.sleep(1.0)
                break
            if not th.is_alive():
                break
            time.sleep(0.2)
    finally:
        stop.set()
        th.join(timeout=10.0)
        rx_engine.reconstruct = real_reconstruct

    saves = sorted(f for f in Path(tmp_path).glob("*.png") if "_diversity" not in f.stem)
    return saves, state


@pytest.mark.slow
def test_diversity_combines_two_branches_into_one_save(tmp_path):
    x = _transmission("A", seed=1)
    a = hfchannel.apply_channel(x, snr_db=8.0, seed=11)
    b = hfchannel.apply_channel(x, snr_db=8.0, seed=22)

    saves, state = _run_diversity_loop(a, b, tmp_path)

    assert len(saves) == 1, f"expected 1 saved image, got {len(saves)}: {saves}"
    with state.lock:
        assert state.status == "done"
        assert state.frames_received == state.n_frames_expected
        assert state.callsign == "TEST"


@pytest.mark.slow
def test_diversity_falls_back_to_single_branch_when_the_other_is_dead(tmp_path):
    """One branch never acquires (pure noise); the other is clean. The
    reception must still complete using the surviving branch alone."""
    x = _transmission("A", seed=2)
    good = hfchannel.apply_channel(x, snr_db=10.0, seed=33)
    dead = np.random.default_rng(0).normal(scale=0.05, size=len(x))

    saves, state = _run_diversity_loop(good, dead, tmp_path)

    assert len(saves) == 1
    with state.lock:
        assert state.status == "done"


@pytest.mark.slow
def test_diversity_two_transmissions_save_two_distinct_images(tmp_path):
    """Same dedup guard as decode_loop's own test, now with two ring
    buffers: two complete transmissions must produce two saves, not the
    same one decoded and written out twice."""
    gap = np.zeros(int(2.0 * FS))
    t1 = _transmission("A", seed=3)
    t2 = _transmission("A", seed=4)
    x = np.concatenate([np.zeros(int(0.5 * FS)), t1, gap, t2, gap])
    a = hfchannel.apply_channel(x, snr_db=10.0, seed=44)
    b = hfchannel.apply_channel(x, snr_db=10.0, seed=55)

    saves, _ = _run_diversity_loop(a, b, tmp_path, expect_saves=2, once=False)

    assert len(saves) == 2, f"expected 2 saved images, got {len(saves)}: {saves}"
    digests = {hashlib.sha1(p.read_bytes()).hexdigest() for p in saves}
    assert len(digests) == 2, "the same image was saved twice"


@pytest.mark.slow
def test_diversity_writes_debug_contribution_image_when_both_branches_lock(tmp_path):
    x = _transmission("A", seed=5)
    a = hfchannel.apply_channel(x, snr_db=8.0, seed=66)
    b = hfchannel.apply_channel(x, snr_db=8.0, seed=77)

    debug_sink = SaveDebugImageToDirSink(verbose=False)
    saves, _ = _run_diversity_loop(a, b, tmp_path, debug_sink=debug_sink)

    assert len(saves) == 1
    debug_files = list(Path(tmp_path).glob("*_diversity.png"))
    assert len(debug_files) == 1, debug_files
    img = Image.open(debug_files[0])
    assert img.size[1] % 132 == 0  # LATENT_CHANNELS rows, at whatever scale


@pytest.mark.slow
def test_diversity_skips_debug_image_when_only_one_branch_locked(tmp_path):
    x = _transmission("A", seed=6)
    good = hfchannel.apply_channel(x, snr_db=10.0, seed=88)
    dead = np.random.default_rng(1).normal(scale=0.05, size=len(x))

    debug_sink = SaveDebugImageToDirSink(verbose=False)
    saves, _ = _run_diversity_loop(good, dead, tmp_path, debug_sink=debug_sink)

    assert len(saves) == 1
    assert list(Path(tmp_path).glob("*_diversity.png")) == []


# --- blind-fallback branches -----------------------------------------------

@pytest.mark.slow
def test_diversity_combines_a_header_branch_with_a_blind_only_branch(tmp_path):
    """Branch B's preamble/header is silence (as if too faded to detect
    at all), so it can only ever blind-lock -- but the frames behind it
    are clean, so the reception should still complete, combining a
    header-locked branch A with a blind-locked branch B, rather than
    branch B contributing nothing."""
    x = _transmission("A", seed=20)
    a = hfchannel.apply_channel(x, snr_db=10.0, seed=200)
    b = _zero_preamble(hfchannel.apply_channel(x, snr_db=10.0, seed=300))

    saves, state = _run_diversity_loop(a, b, tmp_path, poll_interval=0.1)

    assert len(saves) == 1, f"expected 1 saved image, got {len(saves)}: {saves}"
    with state.lock:
        assert state.status == "done"


@pytest.mark.slow
def test_diversity_completes_when_both_branches_are_blind_only(tmp_path):
    """Neither branch's preamble/header survives -- both can only
    blind-lock. Completion has to fall back to progress-stall detection
    (config.end_grace), the same as decode_loop's own all-blind path,
    since neither branch knows the true frame count."""
    x = _transmission("A", seed=21)
    a = _zero_preamble(hfchannel.apply_channel(x, snr_db=10.0, seed=210))
    b = _zero_preamble(hfchannel.apply_channel(x, snr_db=10.0, seed=320))

    config = RxConfig(poll_interval=0.1, end_grace=0.5, once=True)
    ring_a = RingBuffer(len(a) / FS + 5.0)
    ring_b = RingBuffer(len(b) / FS + 5.0)
    ring_a.write(a)
    ring_b.write(b)

    def fake_reconstruct(model, latents, weights):
        img = Image.new("RGB", (8, 8))
        img.putdata([(1, 2, 3)] * 64)
        return img

    real_reconstruct = rx_engine.reconstruct
    rx_engine.reconstruct = fake_reconstruct
    state = SharedState()
    stop = threading.Event()
    config.out_dir = str(tmp_path)

    class _CaptureSink:
        def __init__(self):
            self.received = []

        def on_reception(self, rec):
            self.received.append(rec)
            return None

    sink = _CaptureSink()

    th = threading.Thread(
        target=decode_loop_diversity,
        args=([ring_a, ring_b], None, state, config, stop, sink),
        daemon=True,
    )
    th.start()
    deadline = time.time() + 60.0
    try:
        while time.time() < deadline and not sink.received:
            time.sleep(0.1)
    finally:
        stop.set()
        th.join(timeout=10.0)
        rx_engine.reconstruct = real_reconstruct

    assert len(sink.received) == 1, "an all-blind diversity combine should still finish once"
    assert sink.received[0].mode_name is None  # true mode/duration never became known
    with state.lock:
        assert state.status == "done"

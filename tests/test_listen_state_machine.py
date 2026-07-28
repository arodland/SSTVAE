"""Regression tests for sstvae_listen's reception state machine.

These exercise the bookkeeping that decides *which* reception a decode
belongs to and whether it has already been saved — the logic that
governs multiple transmissions sitting in the ring buffer at once. The
audio is synthesized and fed straight into the RingBuffer, so no audio
device is involved; `reconstruct` is stubbed out so no checkpoint is
needed and each save can be fingerprinted back to the latents that
produced it.
"""

import hashlib
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

# No torch guard here: `reconstruct` is stubbed out below, so this file
# never loads a model. It used to import torch and skip without it,
# which meant the whole slow suite silently vanished on any machine
# without torch installed -- including CI, where it reported 13 skips
# and a green tick. Removed 2026-07-28.

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import sstvae_listen  # noqa: E402
from sstvae.config import FS, MODES  # noqa: E402
from sstvae.modem import Modem  # noqa: E402
from sstvae.rx import engine as rx_engine  # noqa: E402


def _transmission(mode: str, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    lat = rng.normal(size=MODES[mode].n_latents)
    lat /= np.sqrt(np.mean(lat**2))
    return Modem().modulate(lat, mode, callsign="TEST")


class _Args:
    """Stand-in for the argparse namespace decode_loop consumes."""

    def __init__(self, out_dir, poll_interval=0.05, end_grace=0.3):
        self.out_dir = str(out_dir)
        self.poll_interval = poll_interval
        self.end_grace = end_grace
        self.size = None
        self.once = False
        self.blind_search_seconds = 25.0


def _run_decode_loop(loop_fn, audio, tmp_path, timeout_s=180.0, expect_saves=2):
    """Prefill a ring buffer with `audio`, run `loop_fn` until it has
    saved `expect_saves` images (or times out), and report what it saved."""
    ring = sstvae_listen.RingBuffer(len(audio) / FS + 5.0)
    ring.write(audio)

    saves = []  # (path, fingerprint of the latents that produced it)
    from PIL import Image

    def fake_reconstruct(model, latents, weights):
        # Fingerprint the latents into the *pixels*, so two saves of the
        # same reception are byte-identical PNGs and two saves of
        # different receptions are not.
        h = hashlib.sha1(
            np.ascontiguousarray(np.round(latents, 3)).tobytes()
        ).digest()
        img = Image.new("RGB", (8, 8))
        img.putdata([(h[0], h[1], h[2])] * 64)
        return img

    # Patched where the decode loop looks it up, which is the engine
    # module -- sstvae_listen only re-exports it for the CLI.
    real_reconstruct = rx_engine.reconstruct
    rx_engine.reconstruct = fake_reconstruct

    state = sstvae_listen.SharedState()
    stop = threading.Event()
    args = _Args(tmp_path)
    th = threading.Thread(
        target=loop_fn, args=(ring, None, state, args, stop), daemon=True
    )
    th.start()
    deadline = time.time() + timeout_s
    try:
        while time.time() < deadline:
            files = sorted(Path(tmp_path).glob("*.png"))
            if len(files) >= expect_saves:
                time.sleep(1.0)  # let any duplicate save land too
                break
            if not th.is_alive():
                break
            time.sleep(0.2)
    finally:
        stop.set()
        th.join(timeout=10.0)
        rx_engine.reconstruct = real_reconstruct

    for p in sorted(Path(tmp_path).glob("*.png")):
        saves.append(p)
    return saves, state


@pytest.mark.slow
def test_two_transmissions_in_buffer_save_two_distinct_images(tmp_path):
    """Two complete mode-A transmissions sitting in the buffer must
    produce two saved images, not the same one twice.

    The failure this guards against: the dedup check ran against the
    preamble found by a *floored* sync_acquire, while the decode itself
    used an unfloored global argmax, so the two could disagree — the
    already-saved transmission got decoded and saved again while the
    bookkeeping recorded the other one's position.
    """
    gap = np.zeros(int(2.0 * FS))
    a = _transmission("A", seed=1)
    b = _transmission("A", seed=2)
    audio = np.concatenate([np.zeros(int(0.5 * FS)), a, gap, b, gap])

    saves, _ = _run_decode_loop(sstvae_listen.decode_loop, audio, tmp_path)

    assert len(saves) == 2, f"expected 2 saved images, got {len(saves)}: {saves}"
    digests = {hashlib.sha1(p.read_bytes()).hexdigest() for p in saves}
    assert len(digests) == 2, "the same image was saved twice"


@pytest.mark.slow
def test_low_cpu_does_not_resave_the_same_transmission(tmp_path):
    """The low-CPU loop resumes its preamble search from the position it
    held *before* waiting out the transmission, so without an explicit
    guard it re-finds and re-saves the reception it just completed."""
    a = _transmission("A", seed=3)
    audio = np.concatenate([np.zeros(int(0.5 * FS)), a, np.zeros(int(2.0 * FS))])

    saves, _ = _run_decode_loop(
        sstvae_listen.decode_loop_low_cpu, audio, tmp_path,
        timeout_s=120.0, expect_saves=2,
    )
    assert len(saves) == 1, f"one transmission should save once, got {len(saves)}"

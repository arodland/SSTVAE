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


def _Args(out_dir, poll_interval=0.05, end_grace=0.3):
    """The loops' config, built as the real `RxConfig` rather than as a
    duck-typed stand-in. It used to be a hand-written class mimicking the
    argparse namespace `decode_loop` once took, and that shape breaks
    silently and confusingly the moment a field is added to RxConfig: the
    loop raises AttributeError inside its worker thread, the thread dies,
    and every test here fails as "saved 0 images" with no hint that a
    field is missing rather than that decoding regressed."""
    return sstvae_listen.RxConfig(
        out_dir=str(out_dir),
        poll_interval=poll_interval,
        end_grace=end_grace,
        size=None,
        once=False,
        blind_search_seconds=25.0,
    )


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


def _frames_only(mode: str, seed: int) -> np.ndarray:
    """A transmission with no preamble/header at all -- forces
    decode_loop past `_find_new_reception` (which needs one) and into
    the blind path on every poll, the situation these tests target."""
    x = _transmission(mode, seed)
    from sstvae.config import LEADIN_SAMPLES, PREAMBLE_SAMPLES, HEADER_SAMPLES

    return x[LEADIN_SAMPLES + PREAMBLE_SAMPLES + HEADER_SAMPLES :]


def _run_decode_loop_incremental(loop_fn, chunks, tmp_path, args,
                                 timeout_s=180.0, expect_saves=1,
                                 ring_seconds=None):
    """Like `_run_decode_loop`, but the audio arrives as a sequence of
    `chunks` written to the ring buffer one at a time rather than all up
    front -- needed to reproduce bugs that only show up while the buffer
    is still growing. Returns (saves, state, handle) once `chunks` has
    been written and either `expect_saves` images have appeared or
    `timeout_s` has elapsed.

    The decode_loop thread is left running: the caller owns its
    lifecycle from here and must call `handle.stop()` when done (whether
    or not more is fed via `handle.feed()` first) -- stopping it here
    instead would join the thread before a caller-fed follow-up chunk
    ever reached it.

    `ring_seconds`, if given, must cover everything that will ever be
    fed, including via the handle's `feed` after this returns -- a ring
    too small for a later feed silently evicts the reception's own
    earlier audio instead of erroring."""
    total_len = sum(len(c) for c in chunks)
    ring = sstvae_listen.RingBuffer(
        ring_seconds if ring_seconds is not None else total_len / FS + 5.0
    )

    from PIL import Image

    def fake_reconstruct(model, latents, weights):
        img = Image.new("RGB", (8, 8))
        img.putdata([(1, 2, 3)] * 64)
        return img

    real_reconstruct = rx_engine.reconstruct
    rx_engine.reconstruct = fake_reconstruct

    state = sstvae_listen.SharedState()
    stop = threading.Event()
    th = threading.Thread(
        target=loop_fn, args=(ring, None, state, args, stop), daemon=True
    )
    th.start()

    class _Handle:
        def feed(self, chunk):
            ring.write(chunk)

        def wait_for_saves(self, n, timeout_s):
            deadline = time.time() + timeout_s
            while time.time() < deadline:
                files = sorted(Path(tmp_path).glob("*.png"))
                if len(files) >= n:
                    return files
                if not th.is_alive():
                    break
                time.sleep(0.05)
            return sorted(Path(tmp_path).glob("*.png"))

        def stop(self):
            stop.set()
            th.join(timeout=10.0)
            rx_engine.reconstruct = real_reconstruct

    h = _Handle()
    for c in chunks:
        h.feed(c)
    saves = h.wait_for_saves(expect_saves, timeout_s)
    return saves, state, h


@pytest.mark.slow
def test_blind_reception_completes_promptly_despite_a_growing_buffer(tmp_path):
    """A short (mode A) blind-only reception must finish shortly after
    end_grace once the real transmission is over -- not stay in
    "receiving" while trailing silence keeps accumulating.

    demodulate_blind's weight for a frame it demodulates is nonzero
    (just small, for noise) essentially always, and it demodulates every
    frame the *whole current buffer* can hold. A bare nonzero count as
    the stall-detector's progress metric therefore keeps climbing every
    poll as the buffer grows and touches more of the legal frame range —
    real signal or not — so the "progress stopped changing" condition
    was never met until buffer growth had mapped the *entire* range,
    tens of seconds after the real (32 s) transmission was long over.
    Regression for a report of receptions sitting in "receiving"
    seemingly forever.
    """
    sig = _frames_only("A", seed=11)
    pre = np.random.default_rng(1).normal(scale=0.01, size=int(5.0 * FS))
    trailing = np.random.default_rng(2).normal(scale=0.01, size=int(40.0 * FS))

    args = _Args(tmp_path, poll_interval=0.3, end_grace=2.0)
    real_end_s = (len(pre) + len(sig)) / FS

    t0 = time.time()
    saves, state, h = _run_decode_loop_incremental(
        sstvae_listen.decode_loop,
        [pre, sig, trailing],
        tmp_path,
        args,
        timeout_s=60.0,
        expect_saves=1,
    )
    h.stop()
    elapsed_after_end = time.time() - t0 - real_end_s
    assert len(saves) == 1, f"expected 1 saved image, got {len(saves)}: {saves}"
    # Generous margin over end_grace for scheduling/poll-interval slop,
    # but nowhere near the ~55 s of trailing silence available, let
    # alone mode C's ~95 s deadline -- this is what pins the fix rather
    # than merely re-checking the (much weaker) "it saved eventually"
    # deadline test below.
    assert elapsed_after_end < 15.0, (
        f"reception took {elapsed_after_end:.1f}s past the real transmission's end "
        f"to finish (end_grace={args.end_grace}s) -- still climbing on buffer growth?"
    )


@pytest.mark.slow
def test_blind_reception_has_a_hard_deadline_even_if_progress_never_settles(tmp_path):
    """Even if the stall detector's "progress stopped changing" never
    fires -- end_grace set absurdly high here, to isolate this from the
    prompt-completion behavior tested above -- a blind reception must
    still finish once the buffer holds audio past mode C's own duration
    from the transmission's known start (from the beacon). Before this
    backstop existed, a reception stuck for any reason simply never
    ended -- observed sitting in "receiving" for many minutes.
    """
    from sstvae.config import MODES, FRAME_SAMPLES

    sig = _frames_only("A", seed=12)
    pre = np.random.default_rng(3).normal(scale=0.01, size=int(5.0 * FS))
    # Enough trailing silence after this to lock and sit in "receiving",
    # but nowhere near mode C's duration past the transmission's start.
    short_trailing = np.random.default_rng(4).normal(scale=0.01, size=int(10.0 * FS))

    args = _Args(tmp_path, poll_interval=0.1, end_grace=1e9)
    needed_total_s = len(pre) / FS + MODES["C"].n_frames * FRAME_SAMPLES / FS

    saves, state, h = _run_decode_loop_incremental(
        sstvae_listen.decode_loop,
        [pre, sig, short_trailing],
        tmp_path,
        args,
        timeout_s=20.0,
        expect_saves=1,
        ring_seconds=needed_total_s + 15.0,
    )
    try:
        assert len(saves) == 0, (
            "reception finished before its deadline could possibly be reached -- "
            "end_grace=1e9 should have made that impossible; the fix under test "
            "isn't isolated"
        )
        assert state.status != "done"

        # Push the buffer well past frame 0's position + mode C's duration.
        # reception_start lands close to len(pre) (see decode_loop's blind
        # branch); pad generously past exactness rather than reproduce its
        # arithmetic here.
        have_s = (len(pre) + len(sig) + len(short_trailing)) / FS
        more = np.random.default_rng(5).normal(
            scale=0.01, size=int((needed_total_s - have_s + 5.0) * FS)
        )
        h.feed(more)
        saves = h.wait_for_saves(1, timeout_s=60.0)
        assert len(saves) == 1, (
            f"expected 1 saved image past the deadline, got {len(saves)}: {saves}"
        )
    finally:
        h.stop()


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


@pytest.mark.slow
def test_fresh_session_does_not_inherit_blind_evidence_from_a_prior_one(tmp_path):
    """The app's 'start receiving' button and resume_after_transmit both
    work by discarding the whole RingBuffer and calling decode_loop
    again from scratch (see native/gui/rx_panel.cpp's start() /
    resume_after_transmit(), and RingBuffer.clear()'s docstring, which
    records that clear() is *not* how that path works) rather than
    clearing state in place. blind_acc is a plain local inside
    decode_loop, so a brand-new call gets a clean one for free -- but
    that's a property of scoping, and nothing pins it against a future
    change that keeps decode_loop (and its locals) running across a
    'session' boundary instead of restarting it, which is exactly the
    scenario the indeterminate gap in real captured audio -- silence
    while transmitting, or whatever a fresh capture stream first hands
    back -- has no way to signal to a *stale* accumulator.

    Session 1 gets a real, complete mode-A transmission and is left to
    lock and finish, so it isn't just idling. Session 2 is a brand-new
    RingBuffer + decode_loop call, exactly as start()/resume_after_
    transmit() produce, fed nothing but noise: it must never save a
    reception. If blind_acc's folded evidence ever leaked across that
    boundary, session 2 would begin already most of the way to session
    1's lock rather than from nothing.
    """
    sig = _frames_only("A", seed=31)
    pre = np.random.default_rng(8).normal(scale=0.01, size=int(2.0 * FS))
    trailing = np.random.default_rng(9).normal(scale=0.01, size=int(10.0 * FS))

    session1_dir = tmp_path / "session1"
    session1_dir.mkdir()
    args1 = _Args(session1_dir, poll_interval=0.1, end_grace=1.0)
    saves1, _, h1 = _run_decode_loop_incremental(
        sstvae_listen.decode_loop, [pre, sig, trailing], session1_dir, args1,
        timeout_s=60.0, expect_saves=1,
    )
    h1.stop()
    assert len(saves1) == 1, (
        "session 1 should have locked on and saved a real transmission -- "
        "otherwise this isn't exercising real blind-accumulator evidence"
    )

    # A brand-new session: fresh RingBuffer, fresh decode_loop call, same
    # shape as what start()/resume_after_transmit() actually do. Fed
    # nothing but noise -- a fixed seed, so this is deterministic rather
    # than a rare-false-lock flake.
    session2_dir = tmp_path / "session2"
    session2_dir.mkdir()
    noise = np.random.default_rng(10).normal(scale=0.01, size=int(10.0 * FS))
    args2 = _Args(session2_dir, poll_interval=0.1, end_grace=1.0)
    saves2, _, h2 = _run_decode_loop_incremental(
        sstvae_listen.decode_loop, [noise], session2_dir, args2,
        timeout_s=5.0, expect_saves=1,
    )
    try:
        assert len(saves2) == 0, (
            "a brand-new session fed only noise saved a reception -- "
            "blind_acc leaked evidence across a session boundary"
        )
    finally:
        h2.stop()

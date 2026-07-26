"""End-to-end through the application's own transmit and receive paths.

Compose an overlay, run it through `TxEngine`, feed the resulting
waveform into the receiver's `RingBuffer` instead of a soundcard, and
let the real decode loop recover it. This is the check that the pieces
the GUI is assembled from actually fit together -- the panels above them
add only widgets.

Needs a model checkpoint, so it is skipped unless one is available:
point SSTVAE_TEST_CHECKPOINT at a .pt file, or drop one in the repo
root. Marked slow -- a mode A round trip is ~30 s of audio and the
decode is not real-time.
"""

import os
import threading
import time
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from PIL import Image  # noqa: E402

from sstvae.config import FS  # noqa: E402
from sstvae.overlay import ImageItem, OverlayDoc, TextItem, render  # noqa: E402
from sstvae.rx import RingBuffer, RxConfig, SharedState, decode_loop  # noqa: E402
from sstvae.tx import TxConfig, TxEngine  # noqa: E402

REPO = Path(__file__).resolve().parent.parent


def find_checkpoint() -> str | None:
    explicit = os.environ.get("SSTVAE_TEST_CHECKPOINT")
    if explicit and Path(explicit).exists():
        return explicit
    for name in ("ckpt_640_final.pt", "checkpoint.pt"):
        if (REPO / name).exists():
            return str(REPO / name)
    return None


@pytest.fixture(scope="module")
def model():
    path = find_checkpoint()
    if path is None:
        pytest.skip("no local checkpoint; set SSTVAE_TEST_CHECKPOINT")
    from sstvae.codec import load_model

    return load_model(path)


class FakePtt:
    def __init__(self):
        self.events = []

    def set_ptt(self, on):
        self.events.append(on)


def ring_player(ring):
    """A 'soundcard' whose output is the receiver's input buffer."""

    def play(device, wave, samplerate=FS, on_progress=None, should_stop=None,
             on_error=None):
        for i in range(0, len(wave), 4096):
            if should_stop and should_stop():
                return False
            ring.write(wave[i : i + 4096])
            if on_progress:
                on_progress(min(1.0, (i + 4096) / len(wave)))
        return True

    return play


def composed_picture() -> Image.Image:
    """A picture put together the way the transmit panel does it."""
    from sstvae.data import fit_image

    base = fit_image(Image.effect_mandelbrot((640, 480), (-2, -1.2, 0.8, 1.2), 40)
                     .convert("RGB"))
    last_rx = Image.new("RGB", (320, 240), (0, 160, 220))
    doc = OverlayDoc(items=[
        TextItem(text="N0CALL FN20", x=0.03, y=0.03, size=0.10),
        ImageItem(source="last_rx", x=0.68, y=0.66, width=0.28),
    ])
    return render(base, doc, last_rx)


@pytest.mark.slow
def test_transmit_then_receive_recovers_the_picture(model, tmp_path):
    picture = composed_picture()
    ring = RingBuffer(130.0)
    ptt = FakePtt()

    engine = TxEngine(ptt=ptt, player=ring_player(ring), model=model)
    ok = engine.transmit(
        picture,
        TxConfig(mode="A", callsign="N0CALL", ptt_lead_s=0.0, ptt_tail_s=0.0),
    )

    assert ok, "transmission did not complete"
    assert ptt.events == [True, False], "PTT must bracket the transmission"

    received = []

    class Sink:
        def on_reception(self, rec):
            path = tmp_path / f"rx_{len(received)}.png"
            rec.image.save(path)
            received.append(rec)
            return str(path)

    stop = threading.Event()
    worker = threading.Thread(
        target=decode_loop,
        args=(ring, model, SharedState(), RxConfig(poll_interval=1.0, end_grace=4.0),
              stop, Sink()),
        daemon=True,
    )
    worker.start()
    deadline = time.time() + 240
    try:
        while time.time() < deadline and not received:
            time.sleep(0.5)
    finally:
        stop.set()
        worker.join(timeout=15)

    assert received, "the transmission was never decoded"
    rec = received[0]
    assert rec.mode_name == "A"
    assert rec.callsign == "N0CALL", "the beacon callsign did not survive"
    assert rec.frames_received == rec.n_frames_expected

    sent = np.asarray(picture, dtype=float)
    got = np.asarray(rec.image.resize(picture.size), dtype=float)
    psnr = 10 * np.log10(255.0**2 / np.mean((sent - got) ** 2))
    # Mode A over a noiseless channel: the only losses are the codec's
    # own and the modem's clip floor. Well above this in practice (~23 dB
    # on a photograph); the threshold is here to catch a broken pipeline,
    # not to track codec quality.
    assert psnr > 15.0, f"recovered picture is too far off ({psnr:.1f} dB PSNR)"

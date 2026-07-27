"""The transmit sequence: encode, modulate, key the rig, play, unkey.

    compose -> fit 640x480 -> encode -> modulate -> PTT on -> lead delay
    -> play -> tail delay -> PTT off

The one rule this module exists to enforce: **PTT always comes back
down.** A cancelled transmission, an exception anywhere in the chain, an
audio device that stops calling back with the USB cable half out -- all
of them must unkey. So the keyed region is wrapped in try/finally, and a
watchdog thread independently drops PTT if the transmission runs past
its known duration by a margin. A stuck transmitter is a hazard to the
band and to the radio's finals, and it is not acceptable to rely on the
happy path for that.

Everything here is headless and synchronous: `TxEngine.transmit()`
blocks for the duration of the transmission (~32-95 s), so callers run
it on a worker thread and watch it through the `on_state` callback.
"""

import threading
from dataclasses import dataclass
from enum import Enum

import numpy as np
import torch
from PIL import Image

from ..codec import load_model
from ..config import FS, MODES, ModeSpec
from ..models import SSTVAE
from ..modem import Modem

# How long after the audio should have finished before the watchdog
# concludes something is wedged and unkeys anyway. Generous, because a
# resampled stream on a busy machine can legitimately lag by seconds.
WATCHDOG_MARGIN_S = 15.0


class TxPhase(str, Enum):
    IDLE = "idle"
    ENCODING = "encoding"  # neural encoder; no progress fraction available
    MODULATING = "modulating"
    KEYING = "keying"  # PTT up, waiting out the lead delay
    SENDING = "sending"  # audio playing; progress is meaningful here
    UNKEYING = "unkeying"
    DONE = "done"
    CANCELLED = "cancelled"
    FAILED = "failed"


@dataclass
class TxState:
    phase: TxPhase = TxPhase.IDLE
    progress: float = 0.0  # 0..1, only meaningful during SENDING
    message: str = ""

    @property
    def active(self) -> bool:
        return self.phase not in (
            TxPhase.IDLE, TxPhase.DONE, TxPhase.CANCELLED, TxPhase.FAILED
        )


@dataclass
class TxConfig:
    mode: str = "B"
    callsign: str = ""
    device: object = None  # audio output device (index, name, or None)
    level: float = 0.9  # output peak, 0..1
    ptt_lead_s: float = 0.3  # PTT up -> audio start (relay + ALC settling)
    ptt_tail_s: float = 0.3  # audio end -> PTT down
    model_path: str | None = None


def condition_for_output(x: np.ndarray, level: float) -> np.ndarray:
    """Scale the modulator's output to the configured peak.

    Deliberately a plain peak scale and nothing else. `Modem.modulate`
    has already done the envelope clipping and band-limiting that sets
    the waveform's ~4.2 dB PAPR (see config.CLIP_HEADROOM_DB and
    dsp.tx_condition); anything further here -- another clip, a
    compressor, a normalize to full scale -- would undo that
    conditioning and spray splatter into the adjacent channel.
    """
    peak = float(np.max(np.abs(x)))
    if peak <= 0:
        return x
    return x * (level / peak)


class TxEngine:
    """One transmission at a time.

    `ptt` is anything with `set_ptt(bool)` (a `RigctldClient`, or None
    for no rig control). `player` defaults to `sstvae.audio.play` and is
    injectable so tests can drive the sequence without a soundcard.
    """

    def __init__(self, ptt=None, player=None, model=None, on_state=None,
                 on_error=None):
        self._ptt = ptt
        self._player = player
        self._model = model
        self._on_state = on_state or (lambda state: None)
        self._on_error = on_error or (lambda msg: None)
        self._cancel = threading.Event()
        self._state = TxState()

    # --- state ---------------------------------------------------------
    @property
    def state(self) -> TxState:
        return self._state

    def _set(self, phase: TxPhase, progress: float = None, message: str = "") -> None:
        self._state = TxState(
            phase=phase,
            progress=self._state.progress if progress is None else progress,
            message=message,
        )
        self._on_state(self._state)

    def cancel(self) -> None:
        """Ask an in-flight transmission to stop. Safe from any thread."""
        self._cancel.set()

    @property
    def cancelled(self) -> bool:
        return self._cancel.is_set()

    # --- the sequence ---------------------------------------------------
    def prepare(self, image: Image.Image, config: TxConfig) -> np.ndarray:
        """Image -> transmit waveform. Pure computation, no rig, no audio;
        separable so a UI can precompute while the operator is still
        deciding, and so tests can check the waveform without keying
        anything."""
        spec: ModeSpec = MODES[config.mode]

        self._set(TxPhase.ENCODING, 0.0, "encoding image")
        if self._model is None:
            self._model = load_model(config.model_path)
        x = _image_to_tensor(image)[None]
        with torch.no_grad():
            z = self._model.encoder(x)
        flat = SSTVAE.latents_to_flat(z)[0].numpy().astype(np.float64)

        if self._cancel.is_set():
            raise _Cancelled()

        self._set(TxPhase.MODULATING, 0.0, f"modulating mode {spec.name}")
        wave = Modem().modulate(
            flat[: spec.n_latents], spec, callsign=config.callsign
        )
        return condition_for_output(wave, config.level)

    def transmit(self, image: Image.Image, config: TxConfig) -> bool:
        """Run the whole sequence. Returns True if the transmission
        completed, False if it was cancelled. Never raises for ordinary
        failures -- the phase becomes FAILED and `on_error` is called."""
        self._cancel.clear()
        try:
            wave = self.prepare(image, config)
        except _Cancelled:
            self._set(TxPhase.CANCELLED, 0.0, "cancelled")
            return False
        except Exception as e:
            self._on_error(f"could not prepare transmission: {e}")
            self._set(TxPhase.FAILED, 0.0, str(e))
            return False

        if self._cancel.is_set():
            self._set(TxPhase.CANCELLED, 0.0, "cancelled")
            return False

        return self._keyed_send(wave, config)

    def _keyed_send(self, wave: np.ndarray, config: TxConfig) -> bool:
        duration_s = len(wave) / FS
        watchdog = _PttWatchdog(
            self._ptt,
            timeout_s=config.ptt_lead_s + duration_s + config.ptt_tail_s
            + WATCHDOG_MARGIN_S,
            on_fire=lambda: self._on_error(
                "PTT watchdog fired: transmission overran its expected "
                "duration, forcing the rig back to receive"
            ),
        )
        completed = False
        try:
            self._set(TxPhase.KEYING, 0.0, "keying rig")
            self._key(True)
            watchdog.start()
            if self._cancel.wait(config.ptt_lead_s):
                return self._cancelled_result()

            self._set(TxPhase.SENDING, 0.0, f"sending ({duration_s:.0f} s)")
            completed = self._play(wave, config)
            if not completed:
                return self._cancelled_result()

            self._set(TxPhase.UNKEYING, 1.0, "unkeying")
            self._cancel.wait(config.ptt_tail_s)
        except Exception as e:
            self._on_error(f"transmission failed: {e}")
            self._set(TxPhase.FAILED, self._state.progress, str(e))
            return False
        finally:
            # The one guarantee this class makes.
            watchdog.cancel()
            self._key(False)

        self._set(TxPhase.DONE, 1.0, "sent")
        return True

    def _cancelled_result(self) -> bool:
        self._set(TxPhase.CANCELLED, self._state.progress, "cancelled")
        return False

    def _key(self, on: bool) -> None:
        if self._ptt is None:
            return
        try:
            self._ptt.set_ptt(on)
        except Exception as e:
            # Failing to key is a normal, reportable problem. Failing to
            # *unkey* is an emergency the operator has to know about now.
            if on:
                self._on_error(f"PTT on failed: {e}")
            else:
                self._on_error(
                    f"PTT OFF FAILED: {e} -- the rig may still be "
                    "transmitting. Unkey it manually."
                )

    def _play(self, wave: np.ndarray, config: TxConfig) -> bool:
        player = self._player
        if player is None:
            from ..audio import play as player

        return player(
            config.device,
            wave,
            samplerate=FS,
            on_progress=self._report_progress,
            should_stop=self._cancel.is_set,
            on_error=self._on_error,
        )

    def _report_progress(self, frac: float) -> None:
        # Called from the audio callback: keep it to a field write plus
        # the caller's (Qt-signal-emitting) callback. No allocation, no
        # locks, no I/O -- underruns live here.
        self._state.progress = frac
        self._on_state(self._state)


class _Cancelled(Exception):
    """Internal: cancellation noticed inside prepare()."""


class _PttWatchdog:
    """Drops PTT unconditionally if it is still up after `timeout_s`.

    Independent of the transmit path on purpose: it exists precisely for
    the cases where that path is stuck and its `finally` is never going
    to run.
    """

    def __init__(self, ptt, timeout_s: float, on_fire=None):
        self._ptt = ptt
        self._timeout_s = timeout_s
        self._on_fire = on_fire or (lambda: None)
        self._done = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._ptt is None:
            return
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="ptt-watchdog")
        self._thread.start()

    def _run(self) -> None:
        if self._done.wait(self._timeout_s):
            return
        try:
            self._ptt.set_ptt(False)
        except Exception:
            pass
        self._on_fire()

    def cancel(self) -> None:
        self._done.set()


def _image_to_tensor(image: Image.Image) -> torch.Tensor:
    # Imported lazily: sstvae.data pulls in torchvision and the training
    # dataset machinery, which a transmit-only install has no use for.
    from ..data import fit_image, image_to_tensor

    return image_to_tensor(fit_image(image))

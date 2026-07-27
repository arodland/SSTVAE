"""Receive panel: waterfall, live picture, and the reception controls.

Threading: the decode loop runs on a plain worker thread and reaches
this widget only through Qt signals, which Qt queues onto the GUI
thread. Nothing here touches a widget from the decode thread.

The sink deliberately does its *saving* on the decode thread rather than
signalling the GUI to save: writing a PNG is file I/O, and the UI thread
should not stall on a slow or full disk mid-reception.
"""

import threading
from pathlib import Path

from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ..audio import AudioUnavailable, open_input_stream
from ..rx import RingBuffer, RxConfig, SharedState, decode_loop, decode_loop_low_cpu, fmt_snr
from ..rx.engine import parse_size
from .overlay_editor import pil_to_pixmap
from .settings import format_filename
from .waterfall import WaterfallWidget


class _Signals(QObject):
    """Signal carrier for the decode thread. Emitting a Qt signal from a
    non-Qt thread is safe -- Qt queues it to the receiver's thread."""

    status = Signal(str)
    progress = Signal(float)
    image = Signal(object)  # PIL image
    reception = Signal(object, object)  # (Reception, saved path or None)
    error = Signal(str)


class _GuiSink:
    """Receives finished receptions from the decode thread.

    Autosave is read through a callable rather than captured, so
    toggling the checkbox takes effect on the very next reception
    without restarting the loop.
    """

    def __init__(self, signals: _Signals, settings_fn):
        self._signals = signals
        self._settings = settings_fn

    def on_reception(self, rec):
        cfg, freq_hz = self._settings()
        saved = None
        if cfg.receive.autosave:
            try:
                saved = self._save(rec, cfg, freq_hz)
            except OSError as e:
                self._signals.error.emit(f"could not save received image: {e}")
        self._signals.reception.emit(rec, saved)
        return saved

    def _save(self, rec, cfg, freq_hz) -> str:
        out_dir = Path(cfg.folders.receive_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = format_filename(
            cfg.receive.filename_template,
            callsign=rec.callsign,
            freq_hz=freq_hz,
            mode=rec.mode_name,
        )
        path = _unique(out_dir / f"{stem}.png")
        img = rec.image
        size = parse_size(cfg.receive.save_size)
        if size:
            img = img.resize(size)
        img.save(path)
        return str(path)


def _unique(path: Path) -> Path:
    """Never overwrite: two receptions can finish in the same second
    with the same callsign and frequency."""
    if not path.exists():
        return path
    for n in range(2, 1000):
        candidate = path.with_stem(f"{path.stem}_{n}")
        if not candidate.exists():
            return candidate
    return path


class ReceivePanel(QWidget):
    receptionSaved = Signal(str)
    imageReceived = Signal(object)  # newest complete picture, for TX insets
    listeningChanged = Signal(bool)

    def __init__(self, app_state, parent=None):
        super().__init__(parent)
        self._app = app_state  # provides .config, .model, .current_frequency_hz
        self._ring = None
        self._stream = None
        self._thread = None
        self._stop = threading.Event()
        self._state = SharedState()
        self._last_reception = None
        self._last_saved_path = None
        self._suspended_for_tx = False

        self._signals = _Signals()
        self._signals.status.connect(self._on_status)
        self._signals.image.connect(self._show_image)
        self._signals.reception.connect(self._on_reception)
        self._signals.error.connect(self._on_error)

        self._build_ui()
        self._poll = self._make_status_timer()

    # --- ui -------------------------------------------------------------
    def _build_ui(self) -> None:
        # Picture on the left, waterfall down the right — the same shape
        # as the transmit panel. The pictures are 4:3, so on the wide
        # monitor most people have, stacking the waterfall on top would
        # leave the sides empty and squeeze the thing you actually want
        # to look at.
        layout = QVBoxLayout(self)
        splitter = QSplitter(Qt.Horizontal, self)

        left = QWidget(splitter)
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)

        preview_box = QGroupBox("Picture", left)
        pv = QVBoxLayout(preview_box)
        self.preview = QLabel("Nothing received yet", left)
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setMinimumHeight(240)
        self.preview.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.preview.setStyleSheet("background:#202024; color:#888;")
        pv.addWidget(self.preview)
        left_layout.addWidget(preview_box, 1)

        self.status = QLabel("Stopped", left)
        left_layout.addWidget(self.status)

        self.progress = QProgressBar(left)
        self.progress.setRange(0, 100)
        left_layout.addWidget(self.progress)

        self.waterfall = WaterfallWidget(None, splitter)
        splitter.addWidget(left)
        splitter.addWidget(self.waterfall)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)
        layout.addWidget(splitter, 1)

        controls = QHBoxLayout()
        self.start_btn = QPushButton("Start receiving", self)
        self.start_btn.clicked.connect(self.start)
        self.stop_btn = QPushButton("Stop", self)
        self.stop_btn.clicked.connect(self.stop)
        self.stop_btn.setEnabled(False)
        self.save_btn = QPushButton("Save image", self)
        self.save_btn.clicked.connect(self.save_current)
        self.save_btn.setEnabled(False)
        self.autosave = QCheckBox("Autosave", self)
        self.autosave.setChecked(self._app.config.receive.autosave)
        self.autosave.toggled.connect(self._on_autosave_toggled)

        for w in (self.start_btn, self.stop_btn, self.save_btn):
            controls.addWidget(w)
        controls.addWidget(self.autosave)
        controls.addStretch(1)
        layout.addLayout(controls)

    def _make_status_timer(self):
        from PySide6.QtCore import QTimer

        t = QTimer(self)
        t.timeout.connect(self._refresh_status)
        t.start(500)
        return t

    # --- lifecycle -------------------------------------------------------
    @property
    def listening(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> bool:
        if self.listening:
            return True
        cfg = self._app.config
        model = self._app.model
        if model is None:
            QMessageBox.warning(
                self, "Model still loading",
                "The codec checkpoint is still loading. Try again in a moment.",
            )
            return False

        self._ring = RingBuffer(cfg.receive.buffer_seconds)
        self.waterfall.set_ring(self._ring)
        try:
            self._stream, rate = open_input_stream(
                cfg.audio.input_device or None, self._ring, cfg.audio.samplerate,
                on_error=self._signals.error.emit,
            )
        except AudioUnavailable as e:
            QMessageBox.critical(self, "Audio unavailable", str(e))
            return False
        except Exception as e:
            QMessageBox.critical(
                self, "Could not open the input device",
                f"{e}\n\nCheck the input device in Settings.",
            )
            return False

        rx_config = RxConfig(
            out_dir=cfg.folders.receive_dir,
            poll_interval=cfg.receive.poll_interval,
            end_grace=cfg.receive.end_grace,
            size=cfg.receive.save_size,
            once=False,
            blind_search_seconds=cfg.receive.blind_search_seconds,
        )
        sink = _GuiSink(self._signals, lambda: (self._app.config, self._app.current_frequency_hz))
        target = decode_loop_low_cpu if cfg.receive.low_cpu else decode_loop
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run_loop, args=(target, rx_config, model, sink),
            daemon=True, name="sstvae-decode",
        )
        self._thread.start()

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.status.setText(f"Listening at {rate} Hz")
        self.listeningChanged.emit(True)
        return True

    def _run_loop(self, target, rx_config, model, sink) -> None:
        try:
            target(self._ring, model, self._state, rx_config, self._stop, sink)
        except Exception as e:  # a crashed loop must not vanish silently
            self._signals.error.emit(f"receive loop stopped: {e}")

    def stop(self) -> None:
        self._stop.set()
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        self.waterfall.set_ring(None)
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status.setText("Stopped")
        self.listeningChanged.emit(False)

    # --- transmit interlock ------------------------------------------------
    def suspend_for_transmit(self) -> None:
        """Stop receiving while we transmit. Our own audio would
        otherwise be decoded straight back into a 'received' picture."""
        if self.listening:
            self._suspended_for_tx = True
            self.stop()
            self.status.setText("Paused -- transmitting")

    def resume_after_transmit(self) -> None:
        if not self._suspended_for_tx:
            return
        self._suspended_for_tx = False
        # start() allocates a fresh ring buffer, so the tail of our own
        # transmission is dropped rather than decoded back as a reception.
        self.start()
        self.waterfall.clear()

    # --- slots -------------------------------------------------------------
    def _refresh_status(self) -> None:
        if not self.listening:
            return
        with self._state.lock:
            status = self._state.status
            mode = self._state.mode_name
            got = self._state.frames_received
            want = self._state.n_frames_expected
            frac = self._state.progress_frac
            callsign = self._state.callsign
            snr = self._state.snr_db
            captured = self._state.seconds_captured
            image = self._state.image

        if status == "listening":
            text = f"Listening... ({captured:.0f}s captured)"
        elif status == "receiving":
            if want is not None:
                text = f"Receiving mode {mode}: frame {got}/{want} ({100 * frac:.0f}%)"
            else:
                text = f"Receiving (blind sync): {100 * frac:.0f}% of latents"
            text += fmt_snr(snr)
            if callsign:
                text += f"  de {callsign}"
        else:
            text = f"Complete{fmt_snr(snr)}"
            if self._last_saved_path:
                text += f" -- saved {Path(self._last_saved_path).name}"

        self.status.setText(text)
        self.progress.setValue(int(100 * frac))
        if image is not None and image is not getattr(self, "_shown_image", None):
            self._shown_image = image
            self._show_image(image)

    def _on_status(self, text: str) -> None:
        self.status.setText(text)

    def _show_image(self, img) -> None:
        pix = pil_to_pixmap(img)
        self.preview.setPixmap(
            pix.scaled(self.preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )
        self._preview_pixmap = pix

    def _on_reception(self, rec, saved_path) -> None:
        self._last_reception = rec
        self._last_saved_path = saved_path
        self.save_btn.setEnabled(True)
        self._show_image(rec.image)
        self.imageReceived.emit(rec.image)
        if saved_path:
            self.receptionSaved.emit(saved_path)

    def _on_error(self, message: str) -> None:
        self.status.setText(message)

    def _on_autosave_toggled(self, on: bool) -> None:
        self._app.config.receive.autosave = on
        self._app.save_config()

    def save_current(self) -> None:
        if self._last_reception is None:
            return
        cfg = self._app.config
        stem = format_filename(
            cfg.receive.filename_template,
            callsign=self._last_reception.callsign,
            freq_hz=self._app.current_frequency_hz,
            mode=self._last_reception.mode_name,
        )
        suggested = str(Path(cfg.folders.receive_dir) / f"{stem}.png")
        Path(cfg.folders.receive_dir).mkdir(parents=True, exist_ok=True)
        path, _ = QFileDialog.getSaveFileName(
            self, "Save received image", suggested, "Images (*.png *.jpg)"
        )
        if not path:
            return
        try:
            self._last_reception.image.save(path)
        except (OSError, ValueError) as e:
            QMessageBox.critical(self, "Could not save", str(e))
            return
        self._last_saved_path = path
        self.receptionSaved.emit(path)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        pix = getattr(self, "_preview_pixmap", None)
        if pix is not None:
            self.preview.setPixmap(
                pix.scaled(self.preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            )

"""The application window, and the shared state the panels hang off.

`AppState` owns the things both panels need -- the configuration, the
model, and the rig -- so neither panel has to reach into the other.

The model is loaded on a worker thread: resolving the published
checkpoint can mean an HTTP download on first run, and a window that
takes thirty seconds to appear looks broken.
"""

import sys
import threading

from PySide6.QtCore import QObject, QTimer, Signal
from PySide6.QtWidgets import (
    QApplication,
    QLabel,
    QMainWindow,
    QMessageBox,
    QStatusBar,
    QTabWidget,
)

from ..codec import load_model
from ..rig import RigError, RigctldClient, spawn_rigctld
from .rx_panel import ReceivePanel
from .settings import Config
from .settings_dialog import SettingsDialog
from .tx_panel import TransmitPanel

APP_NAME = "SSTVAE"


class AppState(QObject):
    """Shared, mutable application state."""

    modelLoaded = Signal()
    rigStatus = Signal(str)

    def __init__(self):
        super().__init__()
        self.config = Config.load()
        self.model = None
        self.current_frequency_hz: float | None = None
        self._rig: RigctldClient | None = None
        self._rigctld_proc = None
        self._model_error: str | None = None

    # --- model ----------------------------------------------------------
    def load_model_async(self) -> None:
        def run():
            try:
                self.model = load_model(self.config.model_path)
            except (SystemExit, Exception) as e:
                self._model_error = str(e)
            self.modelLoaded.emit()

        threading.Thread(target=run, daemon=True, name="sstvae-model-load").start()

    @property
    def model_error(self) -> str | None:
        return self._model_error

    # --- rig ------------------------------------------------------------
    def ptt(self):
        """The object the transmit engine keys, or None if rig control is
        off. Returning None (rather than a no-op stub) is what tells the
        engine there is nothing to key -- VOX or manual PTT."""
        return self._rig if self.config.rig.enabled else None

    def connect_rig(self) -> None:
        self.disconnect_rig()
        if not self.config.rig.enabled:
            self.rigStatus.emit("Rig control off")
            return
        cfg = self.config.rig
        if cfg.spawn_local:
            try:
                self._rigctld_proc = spawn_rigctld(
                    cfg.model, cfg.device, cfg.baud, cfg.port
                )
            except RigError as e:
                self.rigStatus.emit(str(e))
                return
        self._rig = RigctldClient(cfg.host, cfg.port)
        try:
            self._rig.connect()
            self.current_frequency_hz = self._rig.get_frequency_hz()
            self.rigStatus.emit(f"Rig: {self.current_frequency_hz / 1e6:.4f} MHz")
        except RigError as e:
            self.rigStatus.emit(str(e).splitlines()[0])

    def disconnect_rig(self) -> None:
        if self._rig is not None:
            self._rig.close()
            self._rig = None
        if self._rigctld_proc is not None:
            self._rigctld_proc.terminate()
            self._rigctld_proc = None
        self.current_frequency_hz = None

    def poll_frequency(self) -> None:
        """Cache the dial frequency for status and filenames. Quiet on
        failure: a rig that has gone away should degrade the frequency
        field, not nag during a reception."""
        if self._rig is None or not self.config.rig.enabled:
            return
        try:
            self.current_frequency_hz = self._rig.get_frequency_hz()
            self.rigStatus.emit(f"Rig: {self.current_frequency_hz / 1e6:.4f} MHz")
        except RigError as e:
            self.current_frequency_hz = None
            self.rigStatus.emit(str(e).splitlines()[0])

    def save_config(self) -> None:
        try:
            self.config.save()
        except OSError:
            pass  # a read-only config dir must not break the session


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.state = AppState()
        self.setWindowTitle(APP_NAME)
        self.resize(1100, 800)

        self.rx_panel = ReceivePanel(self.state, self)
        self.tx_panel = TransmitPanel(self.state, self)

        tabs = QTabWidget(self)
        tabs.addTab(self.rx_panel, "Receive")
        tabs.addTab(self.tx_panel, "Transmit")
        self.setCentralWidget(tabs)
        self._tabs = tabs

        self._build_menu()
        self._build_status_bar()

        # Half duplex: our own transmission must not be decoded back into
        # a received picture.
        self.tx_panel.transmitStarted.connect(self.rx_panel.suspend_for_transmit)
        self.tx_panel.transmitFinished.connect(self.rx_panel.resume_after_transmit)
        # The most recent picture becomes available as a transmit inset.
        self.rx_panel.imageReceived.connect(self.tx_panel.set_last_rx_image)
        self.rx_panel.receptionSaved.connect(
            lambda p: self.statusBar().showMessage(f"Saved {p}", 5000)
        )

        self.state.modelLoaded.connect(self._on_model_loaded)
        self.state.rigStatus.connect(self._rig_label.setText)
        self.state.load_model_async()
        self.state.connect_rig()

        self._rig_timer = QTimer(self)
        self._rig_timer.timeout.connect(self.state.poll_frequency)
        self._rig_timer.start(int(self.state.config.rig.poll_interval_s * 1000))

        self._update_station_label()

    def _build_menu(self) -> None:
        menu = self.menuBar().addMenu("&File")
        settings_action = menu.addAction("&Settings...")
        settings_action.triggered.connect(self.open_settings)
        menu.addSeparator()
        quit_action = menu.addAction("&Quit")
        quit_action.triggered.connect(self.close)

    def _build_status_bar(self) -> None:
        bar = QStatusBar(self)
        self.setStatusBar(bar)
        self._station_label = QLabel("", self)
        self._rig_label = QLabel("Rig control off", self)
        self._model_label = QLabel("Loading model...", self)
        for w in (self._station_label, self._rig_label, self._model_label):
            bar.addPermanentWidget(w)

    def _update_station_label(self) -> None:
        cs = self.state.config.callsign or "(no callsign set)"
        self._station_label.setText(f"Callsign: {cs}")

    def _on_model_loaded(self) -> None:
        if self.state.model is None:
            self._model_label.setText("Model failed to load")
            QMessageBox.critical(
                self, "Could not load the model",
                f"{self.state.model_error}\n\n"
                "Set a checkpoint path in Settings, or check your network "
                "connection for the published checkpoint.",
            )
            return
        self._model_label.setText("Model ready")

    def open_settings(self) -> None:
        dialog = SettingsDialog(self.state.config, self)
        if dialog.exec() != SettingsDialog.Accepted:
            return
        previous_model = self.state.config.model_path
        dialog.apply_to(self.state.config)
        self.state.save_config()
        self._update_station_label()
        self.state.connect_rig()
        self._rig_timer.setInterval(int(self.state.config.rig.poll_interval_s * 1000))
        self.rx_panel.autosave.setChecked(self.state.config.receive.autosave)
        if self.state.config.model_path != previous_model:
            self._model_label.setText("Loading model...")
            self.state.model = None
            self.state.load_model_async()

    def closeEvent(self, event) -> None:
        if self.tx_panel.transmitting:
            answer = QMessageBox.question(
                self, "Transmitting",
                "A transmission is in progress. Stop it and quit?",
            )
            if answer != QMessageBox.Yes:
                event.ignore()
                return
            self.tx_panel.cancel()
        self.rx_panel.stop()
        self.state.disconnect_rig()
        self.state.save_config()
        super().closeEvent(event)


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())

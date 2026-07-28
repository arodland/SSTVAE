"""The application window, and the shared state the panels hang off.

`AppState` owns the things both panels need -- the configuration, the
model, and the rig -- so neither panel has to reach into the other.

The model is loaded on a worker thread: resolving the published
checkpoint can mean an HTTP download on first run, and a window that
takes thirty seconds to appear looks broken.
"""

import sys
import threading

from PySide6.QtCore import QObject, Signal
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QLabel,
    QMainWindow,
    QMessageBox,
    QStatusBar,
    QTabWidget,
)

from ..codec import load_codec
from .rig_controller import RigController
from .rx_panel import ReceivePanel
from .settings import Config, codec_precision
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
        self._model_error: str | None = None
        self._rig = RigController(self)
        self._rig.statusChanged.connect(self.rigStatus)

    # --- model ----------------------------------------------------------
    def load_model_async(self) -> None:
        def run():
            try:
                self.model = load_codec(self.config.model_path,
                                        precision=codec_precision(self.config))
            except (SystemExit, Exception) as e:
                self._model_error = str(e)
            self.modelLoaded.emit()

        threading.Thread(target=run, daemon=True, name="sstvae-model-load").start()

    @property
    def model_error(self) -> str | None:
        return self._model_error

    # --- rig ------------------------------------------------------------
    # All rigctld I/O lives on the RigController's own thread. Nothing
    # here may call into the rig: every operation is a blocking socket
    # request, and a rigctld that is down costs seconds per attempt.
    @property
    def current_frequency_hz(self) -> float | None:
        """Last polled dial frequency — a cached value, not a request."""
        return self._rig.current_frequency_hz

    def ptt(self):
        """The object the transmit engine keys, or None if rig control is
        off. Returning None (rather than a no-op stub) is what tells the
        engine there is nothing to key -- VOX or manual PTT."""
        return self._rig.ptt() if self.config.rig.enabled else None

    def connect_rig(self) -> None:
        self._rig.apply_config(self.config.rig)

    def disconnect_rig(self) -> None:
        self._rig.stop()

    def pause_rig_polling(self) -> None:
        self._rig.pause()

    def resume_rig_polling(self) -> None:
        self._rig.resume()

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
        # Frequency polling pauses too: the answer is not interesting
        # mid-over, and it keeps CAT chatter off the wire while keyed.
        self.tx_panel.transmitStarted.connect(self.state.pause_rig_polling)
        self.tx_panel.transmitFinished.connect(self.state.resume_rig_polling)
        # The most recent picture becomes available as a transmit inset.
        self.rx_panel.imageReceived.connect(self.tx_panel.set_last_rx_image)
        self.rx_panel.receptionSaved.connect(
            lambda p: self.statusBar().showMessage(f"Saved {p}", 5000)
        )

        self.state.modelLoaded.connect(self._on_model_loaded)
        self.state.rigStatus.connect(self._rig_label.setText)
        self.state.load_model_async()
        self.state.connect_rig()

        self._update_station_label()

    def _build_menu(self) -> None:
        """Build the File menu.

        The explicit `NoRole` calls are load-bearing on macOS. Qt's Cocoa
        plugin pattern-matches action text and moves anything looking
        like Preferences or Quit into the application menu. Both of this
        menu's actions match, so Qt emptied the File menu -- and macOS
        hides an empty menu, leaving no way to reach Settings at all.
        Pinning the roles keeps them where they were put.

        The shortcuts are the belt to that braces: `Preferences` and
        `Quit` are the platform-correct sequences (Cmd+, and Cmd+Q on
        macOS, Ctrl+Q on everything else), so Settings stays reachable
        even if a platform menu bar misbehaves again.
        """
        menu = self.menuBar().addMenu("&File")
        settings_action = menu.addAction("&Settings...")
        settings_action.setMenuRole(QAction.MenuRole.NoRole)
        settings_action.setShortcut(QKeySequence.StandardKey.Preferences)
        settings_action.triggered.connect(self.open_settings)
        menu.addSeparator()
        quit_action = menu.addAction("&Quit")
        quit_action.setMenuRole(QAction.MenuRole.NoRole)
        quit_action.setShortcut(QKeySequence.StandardKey.Quit)
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

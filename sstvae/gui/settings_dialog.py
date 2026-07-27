"""The settings dialog.

Edits a copy of the configuration and only writes it back on OK, so
Cancel really cancels. The device pickers store the device *name*
rather than its PortAudio index: indices are renumbered whenever a USB
device is plugged or unplugged, so a saved index silently comes back
pointing at a different soundcard.
"""

from pathlib import Path

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ..audio import AudioUnavailable, list_devices
from ..rig import RigError, RigctldClient


class _FolderRow(QWidget):
    def __init__(self, value: str, caption: str, parent=None):
        super().__init__(parent)
        self._caption = caption
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.edit = QLineEdit(value, self)
        browse = QPushButton("Browse...", self)
        browse.clicked.connect(self._browse)
        layout.addWidget(self.edit, 1)
        layout.addWidget(browse)

    def _browse(self) -> None:
        path = QFileDialog.getExistingDirectory(self, self._caption, self.edit.text())
        if path:
            self.edit.setText(path)

    def value(self) -> str:
        return self.edit.text()


class SettingsDialog(QDialog):
    def __init__(self, config, parent=None):
        super().__init__(parent)
        self.setWindowTitle("SSTVAE settings")
        self.resize(560, 480)
        self._config = config

        tabs = QTabWidget(self)
        tabs.addTab(self._station_tab(), "Station")
        tabs.addTab(self._audio_tab(), "Audio")
        tabs.addTab(self._rig_tab(), "Rig control")
        tabs.addTab(self._folders_tab(), "Folders")
        tabs.addTab(self._receive_tab(), "Receive")

        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel, parent=self
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addWidget(tabs)
        layout.addWidget(buttons)

    # --- tabs -------------------------------------------------------------
    def _station_tab(self) -> QWidget:
        w = QWidget(self)
        form = QFormLayout(w)
        self.callsign = QLineEdit(self._config.callsign, w)
        self.callsign.setMaxLength(8)  # the beacon's callsign field
        self.callsign.setPlaceholderText("N0CALL")
        form.addRow("Callsign", self.callsign)
        form.addRow(QLabel(
            "Up to 8 characters. Sent continuously on the beacon carrier,\n"
            "so a receiver can identify you even from a partial reception."
        ))

        self.model_path = QLineEdit(self._config.model_path or "", w)
        self.model_path.setPlaceholderText("(published checkpoint)")
        browse = QPushButton("Browse...", w)
        browse.clicked.connect(self._browse_model)
        row = QHBoxLayout()
        row.addWidget(self.model_path, 1)
        row.addWidget(browse)
        holder = QWidget(w)
        holder.setLayout(row)
        form.addRow("Model checkpoint", holder)
        form.addRow(QLabel(
            "Leave blank to use the published checkpoint. Both stations must\n"
            "run the same checkpoint to exchange pictures."
        ))
        return w

    def _audio_tab(self) -> QWidget:
        w = QWidget(self)
        form = QFormLayout(w)
        self.input_device = self._device_combo("input", self._config.audio.input_device)
        self.output_device = self._device_combo("output", self._config.audio.output_device)
        form.addRow("Input (from radio)", self.input_device)
        form.addRow("Output (to radio)", self.output_device)

        self.tx_level = QDoubleSpinBox(w)
        self.tx_level.setRange(0.05, 1.0)
        self.tx_level.setSingleStep(0.05)
        self.tx_level.setDecimals(2)
        self.tx_level.setValue(self._config.transmit.level)
        form.addRow("Transmit level", self.tx_level)
        form.addRow(QLabel(
            "Set the level so the radio's ALC barely moves. The waveform is\n"
            "already conditioned for a ~4 dB envelope peak; driving it into\n"
            "ALC compression will spread it across the band."
        ))
        return w

    def _device_combo(self, kind: str, current: str | None) -> QComboBox:
        combo = QComboBox(self)
        combo.addItem("System default", None)
        try:
            for d in list_devices(kind):
                combo.addItem(d.label(), d.name)
        except AudioUnavailable as e:
            combo.addItem(f"(audio unavailable: {e})", None)
            combo.setEnabled(False)
        if current:
            idx = combo.findData(current)
            combo.setCurrentIndex(idx if idx >= 0 else 0)
        return combo

    def _rig_tab(self) -> QWidget:
        w = QWidget(self)
        form = QFormLayout(w)
        rig = self._config.rig

        self.rig_enabled = QCheckBox("Use rig control (PTT and frequency)", w)
        self.rig_enabled.setChecked(rig.enabled)
        form.addRow(self.rig_enabled)

        self.rig_host = QLineEdit(rig.host, w)
        self.rig_port = QSpinBox(w)
        self.rig_port.setRange(1, 65535)
        self.rig_port.setValue(rig.port)
        form.addRow("rigctld host", self.rig_host)
        form.addRow("rigctld port", self.rig_port)

        self.rig_spawn = QCheckBox("Start a local rigctld myself", w)
        self.rig_spawn.setChecked(rig.spawn_local)
        form.addRow(self.rig_spawn)
        form.addRow(QLabel(
            "Leave unchecked if rigctld is already running (shared with\n"
            "WSJT-X, fldigi, ...). Two programs cannot both hold the\n"
            "serial port."
        ))

        self.rig_model = QLineEdit(rig.model, w)
        self.rig_device = QLineEdit(rig.device, w)
        self.rig_baud = QSpinBox(w)
        self.rig_baud.setRange(300, 921600)
        self.rig_baud.setValue(rig.baud)
        form.addRow("Rig model (rigctl -l)", self.rig_model)
        form.addRow("Serial device", self.rig_device)
        form.addRow("Baud", self.rig_baud)

        self.ptt_lead = QDoubleSpinBox(w)
        self.ptt_lead.setRange(0.0, 3.0)
        self.ptt_lead.setSingleStep(0.05)
        self.ptt_lead.setValue(rig.ptt_lead_s)
        self.ptt_tail = QDoubleSpinBox(w)
        self.ptt_tail.setRange(0.0, 3.0)
        self.ptt_tail.setSingleStep(0.05)
        self.ptt_tail.setValue(rig.ptt_tail_s)
        form.addRow("PTT lead (s)", self.ptt_lead)
        form.addRow("PTT tail (s)", self.ptt_tail)

        test_row = QHBoxLayout()
        test_conn = QPushButton("Test connection", w)
        test_conn.clicked.connect(self._test_connection)
        test_ptt = QPushButton("Test PTT (0.5 s)", w)
        test_ptt.clicked.connect(self._test_ptt)
        test_row.addWidget(test_conn)
        test_row.addWidget(test_ptt)
        holder = QWidget(w)
        holder.setLayout(test_row)
        form.addRow(holder)
        return w

    def _folders_tab(self) -> QWidget:
        w = QWidget(self)
        form = QFormLayout(w)
        f = self._config.folders
        self.receive_dir = _FolderRow(f.receive_dir, "Received images", w)
        self.transmit_dir = _FolderRow(f.transmit_dir, "Images to send", w)
        self.template_dir = _FolderRow(f.template_dir, "Overlay templates", w)
        form.addRow("Received images", self.receive_dir)
        form.addRow("Images to send", self.transmit_dir)
        form.addRow("Overlay templates", self.template_dir)
        form.addRow(QLabel(
            "Saving and reusing overlay templates is not implemented yet;\n"
            "this is where they will go."
        ))
        return w

    def _receive_tab(self) -> QWidget:
        w = QWidget(self)
        form = QFormLayout(w)
        r = self._config.receive

        self.autosave = QCheckBox("Save every completed reception automatically", w)
        self.autosave.setChecked(r.autosave)
        form.addRow(self.autosave)

        self.low_cpu = QCheckBox("Low-CPU mode", w)
        self.low_cpu.setChecked(r.low_cpu)
        form.addRow(self.low_cpu)
        form.addRow(QLabel(
            "Low-CPU mode only looks for the start of a transmission, so it\n"
            "cannot pick up one already in progress or decode retrospectively."
        ))

        self.filename_template = QLineEdit(r.filename_template, w)
        form.addRow("Filename", self.filename_template)
        form.addRow(QLabel(
            "Fields: {date} {time} {freq} {callsign} {mode}.\n"
            "Fields with no value are dropped from the name."
        ))

        self.save_size = QLineEdit(r.save_size or "", w)
        self.save_size.setPlaceholderText("640x480 (as received)")
        form.addRow("Saved size", self.save_size)

        self.buffer_seconds = QDoubleSpinBox(w)
        self.buffer_seconds.setRange(100.0, 600.0)
        self.buffer_seconds.setValue(r.buffer_seconds)
        form.addRow("Buffer (s)", self.buffer_seconds)
        form.addRow(QLabel(
            "Must exceed the longest mode (C, ~95 s) with margin for\n"
            "retrospective decoding."
        ))

        self.poll_interval = QDoubleSpinBox(w)
        self.poll_interval.setRange(1.0, 30.0)
        self.poll_interval.setValue(r.poll_interval)
        form.addRow("Decode every (s)", self.poll_interval)
        return w

    # --- actions ------------------------------------------------------------
    def _browse_model(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Model checkpoint", str(Path.home()), "Checkpoints (*.pt *.ckpt)"
        )
        if path:
            self.model_path.setText(path)

    def _client(self) -> RigctldClient:
        return RigctldClient(self.rig_host.text(), self.rig_port.value(), timeout=2.0)

    def _test_connection(self) -> None:
        try:
            with self._client() as rig:
                freq = rig.get_frequency_hz()
        except RigError as e:
            QMessageBox.warning(self, "Rig control", str(e))
            return
        QMessageBox.information(
            self, "Rig control", f"Connected. Dial frequency: {freq / 1e6:.4f} MHz"
        )

    def _test_ptt(self) -> None:
        import time

        try:
            with self._client() as rig:
                rig.set_ptt(True)
                try:
                    time.sleep(0.5)
                finally:
                    rig.set_ptt(False)
        except RigError as e:
            QMessageBox.warning(self, "Rig control", str(e))
            return
        QMessageBox.information(self, "Rig control", "PTT keyed and released.")

    # --- result --------------------------------------------------------------
    def apply_to(self, config) -> None:
        """Copy the dialog's values into `config`."""
        config.callsign = self.callsign.text().strip().upper()
        config.model_path = self.model_path.text().strip() or None

        config.audio.input_device = self.input_device.currentData()
        config.audio.output_device = self.output_device.currentData()
        config.transmit.level = self.tx_level.value()

        config.rig.enabled = self.rig_enabled.isChecked()
        config.rig.host = self.rig_host.text().strip() or "127.0.0.1"
        config.rig.port = self.rig_port.value()
        config.rig.spawn_local = self.rig_spawn.isChecked()
        config.rig.model = self.rig_model.text().strip() or "1"
        config.rig.device = self.rig_device.text().strip()
        config.rig.baud = self.rig_baud.value()
        config.rig.ptt_lead_s = self.ptt_lead.value()
        config.rig.ptt_tail_s = self.ptt_tail.value()

        config.folders.receive_dir = self.receive_dir.value()
        config.folders.transmit_dir = self.transmit_dir.value()
        config.folders.template_dir = self.template_dir.value()

        config.receive.autosave = self.autosave.isChecked()
        config.receive.low_cpu = self.low_cpu.isChecked()
        config.receive.filename_template = self.filename_template.text()
        config.receive.save_size = self.save_size.text().strip() or None
        config.receive.buffer_seconds = self.buffer_seconds.value()
        config.receive.poll_interval = self.poll_interval.value()

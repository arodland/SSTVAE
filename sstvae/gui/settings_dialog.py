"""The settings dialog.

Edits a copy of the configuration and only writes it back on OK, so
Cancel really cancels. The device pickers store the device *name*
rather than its PortAudio index: indices are renumbered whenever a USB
device is plugged or unplugged, so a saved index silently comes back
pointing at a different soundcard.
"""

import re
from pathlib import Path

from PySide6.QtCore import Qt
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
from ..checkpoint import PRECISIONS
from ..rig import RigError, RigctldClient, list_models


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
        self.model_path.setPlaceholderText("(published model)")
        self.model_path.textChanged.connect(self._sync_precision_enabled)
        browse_dir = QPushButton("Folder...", w)
        browse_dir.clicked.connect(self._browse_model_folder)
        browse_file = QPushButton("File...", w)
        browse_file.clicked.connect(self._browse_model_file)
        row = QHBoxLayout()
        row.addWidget(self.model_path, 1)
        row.addWidget(browse_dir)
        row.addWidget(browse_file)
        holder = QWidget(w)
        holder.setLayout(row)
        form.addRow("Model", holder)
        form.addRow(QLabel(
            "Leave blank for the published model, downloaded once and cached.\n"
            "Otherwise a folder of exported .onnx files, a single .onnx, or a\n"
            ".pt checkpoint. Both stations must run the same model to exchange\n"
            "pictures — but not the same precision, which is a local choice."
        ))

        self.precision = QComboBox(w)
        for p in PRECISIONS:
            self.precision.addItem(p, p)
        idx = self.precision.findData(self._config.precision)
        self.precision.setCurrentIndex(idx if idx >= 0 else 0)
        form.addRow("Precision", self.precision)
        self.precision_note = QLabel("", w)
        self.precision_note.setWordWrap(True)
        self.precision_note.setStyleSheet("color: palette(link-visited);")
        form.addRow("", self.precision_note)
        self._sync_precision_enabled()
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

    def _model_combo(self, current: str, parent: QWidget) -> QComboBox:
        """Picker over `rigctld -l`.

        Editable on purpose: the list can fail to load (no Hamlib), and a
        configuration written by an older/newer Hamlib may name a model
        this one doesn't list. In both cases the number must still be
        typeable, and the saved value must survive a round trip through
        the dialog rather than being silently reset.
        """
        combo = QComboBox(parent)
        combo.setEditable(True)
        combo.setInsertPolicy(QComboBox.NoInsert)
        self._model_error = None
        try:
            models = list_models()
        except RigError as e:
            self._model_error = str(e).splitlines()[0]
            models = []
        for m in models:
            combo.addItem(m.label(), m.number)
        # 300-odd entries, labelled "<mfg> <model>". Qt's default
        # completer anchors at the start of the label, so a user typing
        # the only part they know ("FT-847", "IC-7300") would match
        # nothing; match anywhere in the label instead.
        completer = combo.completer()
        if completer is not None:
            completer.setCaseSensitivity(Qt.CaseInsensitive)
            completer.setFilterMode(Qt.MatchContains)

        try:
            number = int(current)
        except (TypeError, ValueError):
            number = None
        idx = combo.findData(number) if number is not None else -1
        if idx >= 0:
            combo.setCurrentIndex(idx)
        else:
            combo.setEditText(current or "1")
        return combo

    def _rig_model_number(self) -> str:
        """The model number behind the picker's current state.

        A chosen item carries its number as item data. Free text is
        either a bare number or a label that was typed/completed to match
        an item, so fall back to matching the text, then to the leading
        digits of a label like "Yaesu FT-847 (1001)".
        """
        text = self.rig_model.currentText().strip()
        idx = self.rig_model.findText(text)
        if idx >= 0:
            data = self.rig_model.itemData(idx)
            if data is not None:
                return str(data)
        if text.isdigit():
            return text
        m = re.search(r"\((\d+)\)\s*(?:\[[^]]*\])?$", text)
        return m.group(1) if m else "1"

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

        self.rig_model = self._model_combo(rig.model, w)
        self.rig_model_note = QLabel("", w)
        self.rig_device = QLineEdit(rig.device, w)
        self.rig_baud = QSpinBox(w)
        self.rig_baud.setRange(300, 921600)
        self.rig_baud.setValue(rig.baud)
        form.addRow("Rig model", self.rig_model)
        if self._model_error:
            self.rig_model_note.setText(
                f"{self._model_error}\nEnter a model number by hand, or install "
                "Hamlib and reopen this dialog."
            )
            self.rig_model_note.setStyleSheet("color: palette(link-visited);")
            # Disabled, but deliberately not unchecked: Hamlib may just be
            # missing on the machine editing the config, and clearing a
            # setting the user did not touch is worse than a spawn that
            # fails loudly later (rig_controller already reports that).
            self.rig_spawn.setEnabled(False)
            form.addRow("", self.rig_model_note)
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

        self.save_audio = QCheckBox(
            "Also save the captured audio (diagnostic)", w)
        self.save_audio.setChecked(r.save_audio)
        form.addRow(self.save_audio)
        form.addRow(QLabel(
            "Writes a .wav beside each received picture, exactly as captured.\n"
            "Use it when a picture decodes badly: run sstvae_decode.py on the\n"
            "dump to see whether the audio or the decoder was at fault."
        ))

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
    def _browse_model_folder(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self, "Folder of exported .onnx files", str(Path.home())
        )
        if path:
            self.model_path.setText(path)

    def _browse_model_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Model file", str(Path.home()),
            "Model files (*.onnx *.pt *.ckpt);;ONNX (*.onnx);;"
            "Checkpoints (*.pt *.ckpt)",
        )
        if path:
            self.model_path.setText(path)

    def _sync_precision_enabled(self) -> None:
        """Precision only means something for a folder or the published model.

        A `.onnx` filename already names its precision -- picking
        `v1-encoder-int8.onnx` selects int8 no matter what this combo
        says, and the decoder is resolved beside it at the same
        precision. A `.pt` is the torch backend, which has no precision
        at all. Greying the control out in those cases shows the user the
        same rule `settings.codec_precision` applies.
        """
        suffix = Path(self.model_path.text().strip()).suffix.lower()
        if suffix == ".onnx":
            note = "Set by the file name — that artifact is already one precision."
        elif suffix in (".pt", ".ckpt"):
            note = "Not applicable to a .pt checkpoint (that runs on torch)."
        else:
            note = ("fp16 is the default and measures identical to fp32. "
                    "Purely local: it never has to match the far end.")
        self.precision.setEnabled(suffix not in (".onnx", ".pt", ".ckpt"))
        self.precision_note.setText(note)

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
        config.precision = self.precision.currentData()

        config.audio.input_device = self.input_device.currentData()
        config.audio.output_device = self.output_device.currentData()
        config.transmit.level = self.tx_level.value()

        config.rig.enabled = self.rig_enabled.isChecked()
        config.rig.host = self.rig_host.text().strip() or "127.0.0.1"
        config.rig.port = self.rig_port.value()
        config.rig.spawn_local = self.rig_spawn.isChecked()
        config.rig.model = self._rig_model_number()
        config.rig.device = self.rig_device.text().strip()
        config.rig.baud = self.rig_baud.value()
        config.rig.ptt_lead_s = self.ptt_lead.value()
        config.rig.ptt_tail_s = self.ptt_tail.value()

        config.folders.receive_dir = self.receive_dir.value()
        config.folders.transmit_dir = self.transmit_dir.value()
        config.folders.template_dir = self.template_dir.value()

        config.receive.autosave = self.autosave.isChecked()
        config.receive.save_audio = self.save_audio.isChecked()
        config.receive.low_cpu = self.low_cpu.isChecked()
        config.receive.filename_template = self.filename_template.text()
        config.receive.save_size = self.save_size.text().strip() or None
        config.receive.buffer_seconds = self.buffer_seconds.value()
        config.receive.poll_interval = self.poll_interval.value()

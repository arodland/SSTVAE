"""Transmit panel: pick a picture, compose an overlay, send it."""

import threading
from pathlib import Path

from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtWidgets import (
    QColorDialog,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ..config import MODES
from ..images import fit_image
from ..overlay import ImageItem, TextItem
from ..tx import TxConfig, TxEngine, TxPhase
from .overlay_editor import OverlayEditor
from .settings import codec_precision

IMAGE_FILTER = "Images (*.png *.jpg *.jpeg *.webp *.bmp *.gif);;All files (*)"


class _Signals(QObject):
    state = Signal(object)
    error = Signal(str)
    finished = Signal(bool)


class TransmitPanel(QWidget):
    transmitStarted = Signal()
    transmitFinished = Signal()

    def __init__(self, app_state, parent=None):
        super().__init__(parent)
        self._app = app_state
        self._engine = None
        self._thread = None

        self._signals = _Signals()
        self._signals.state.connect(self._on_state)
        self._signals.error.connect(self._on_error)
        self._signals.finished.connect(self._on_finished)

        self._build_ui()

    # --- ui ---------------------------------------------------------------
    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        splitter = QSplitter(Qt.Horizontal, self)

        self.editor = OverlayEditor(self)
        self.editor.selectionChanged.connect(self._on_selection)
        splitter.addWidget(self.editor)
        splitter.addWidget(self._build_side_panel())
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)
        layout.addWidget(splitter, 1)

        layout.addLayout(self._build_send_bar())

    def _build_side_panel(self) -> QWidget:
        panel = QWidget(self)
        v = QVBoxLayout(panel)

        source = QGroupBox("Picture", panel)
        sv = QVBoxLayout(source)
        self.choose_btn = QPushButton("Choose image...", panel)
        self.choose_btn.clicked.connect(self.choose_image)
        self.image_label = QLabel("No image selected", panel)
        self.image_label.setWordWrap(True)
        sv.addWidget(self.choose_btn)
        sv.addWidget(self.image_label)
        v.addWidget(source)

        overlay = QGroupBox("Overlay", panel)
        ov = QVBoxLayout(overlay)
        add_text = QPushButton("Add text", panel)
        add_text.clicked.connect(self._add_text)
        self.add_rx_btn = QPushButton("Add last received image", panel)
        self.add_rx_btn.clicked.connect(self.editor.add_last_rx_inset)
        add_img = QPushButton("Add image from file...", panel)
        add_img.clicked.connect(self._add_image_file)
        remove = QPushButton("Remove selected", panel)
        remove.clicked.connect(self.editor.remove_selected)
        for w in (add_text, self.add_rx_btn, add_img, remove):
            ov.addWidget(w)
        v.addWidget(overlay)

        self.props = self._build_properties(panel)
        v.addWidget(self.props)
        v.addStretch(1)
        return panel

    def _build_properties(self, parent) -> QGroupBox:
        box = QGroupBox("Selected item", parent)
        box.setEnabled(False)
        form = QFormLayout(box)

        # Multi-line: a station's callsign, grid and name belong to one
        # item, not three stacked by hand. Enter inserts a newline, so
        # Tab has to be what leaves the field.
        self.text_edit = QPlainTextEdit(box)
        self.text_edit.setTabChangesFocus(True)
        self.text_edit.setFixedHeight(80)
        self.text_edit.textChanged.connect(self._apply_text)
        form.addRow("Text", self.text_edit)

        self.align_combo = QComboBox(box)
        for label, value in (("Left", "left"), ("Centre", "center"), ("Right", "right")):
            self.align_combo.addItem(label, value)
        self.align_combo.currentIndexChanged.connect(self._apply_align)
        form.addRow("Align", self.align_combo)

        self.size_spin = QDoubleSpinBox(box)
        self.size_spin.setRange(0.01, 1.5)
        self.size_spin.setSingleStep(0.01)
        self.size_spin.setDecimals(3)
        self.size_spin.valueChanged.connect(self._apply_size)
        form.addRow("Size", self.size_spin)

        self.rotation_spin = QDoubleSpinBox(box)
        self.rotation_spin.setRange(-180.0, 180.0)
        self.rotation_spin.setSingleStep(1.0)
        self.rotation_spin.valueChanged.connect(self._apply_rotation)
        form.addRow("Rotation", self.rotation_spin)

        self.color_btn = QPushButton("Colour...", box)
        self.color_btn.clicked.connect(self._pick_color)
        form.addRow("Colour", self.color_btn)
        return box

    def _build_send_bar(self) -> QHBoxLayout:
        bar = QHBoxLayout()
        self.mode_combo = QComboBox(self)
        for name in sorted(MODES):
            spec = MODES[name]
            self.mode_combo.addItem(f"Mode {name} — {spec.duration_s:.0f} s", name)
        self.mode_combo.setCurrentIndex(
            max(0, self.mode_combo.findData(self._app.config.transmit.mode))
        )
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)

        self.send_btn = QPushButton("Send", self)
        self.send_btn.clicked.connect(self.send)
        self.cancel_btn = QPushButton("Cancel", self)
        self.cancel_btn.clicked.connect(self.cancel)
        self.cancel_btn.setEnabled(False)

        self.tx_progress = QProgressBar(self)
        self.tx_progress.setRange(0, 100)
        self.tx_status = QLabel("Ready", self)

        bar.addWidget(QLabel("Mode:", self))
        bar.addWidget(self.mode_combo)
        bar.addWidget(self.send_btn)
        bar.addWidget(self.cancel_btn)
        bar.addWidget(self.tx_progress, 1)
        bar.addWidget(self.tx_status)
        return bar

    # --- content -----------------------------------------------------------
    def choose_image(self) -> None:
        start = self._app.config.folders.transmit_dir
        path, _ = QFileDialog.getOpenFileName(self, "Choose an image", start, IMAGE_FILTER)
        if not path:
            return
        self.load_image(path)

    def load_image(self, path: str) -> None:
        from PIL import Image

        try:
            img = fit_image(Image.open(path))
        except (OSError, ValueError) as e:
            QMessageBox.critical(self, "Could not open image", str(e))
            return
        self.editor.set_base_image(img)
        self.image_label.setText(Path(path).name)
        self._app.config.folders.transmit_dir = str(Path(path).parent)
        self._app.save_config()

    def set_last_rx_image(self, img) -> None:
        self.editor.set_last_rx(img)

    def _add_text(self) -> None:
        self.editor.add_text(self._app.config.callsign or "TEXT")

    def _add_image_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Choose an inset image", self._app.config.folders.transmit_dir,
            IMAGE_FILTER,
        )
        if path:
            self.editor.add_image_inset(path)

    # --- property editing ---------------------------------------------------
    def _on_selection(self, item) -> None:
        self.props.setEnabled(item is not None)
        if item is None:
            return
        is_text = isinstance(item, TextItem)
        self._loading_props = True
        self.text_edit.setEnabled(is_text)
        self.align_combo.setEnabled(is_text)
        self.color_btn.setEnabled(is_text)
        self.text_edit.setPlainText(item.text if is_text else "")
        if is_text:
            self.align_combo.setCurrentIndex(
                max(0, self.align_combo.findData(item.align))
            )
        self.size_spin.setValue(item.size if is_text else item.width)
        self.rotation_spin.setValue(item.rotation)
        self._loading_props = False

    def _selected(self):
        if getattr(self, "_loading_props", False):
            return None
        return self.editor.selected_item()

    def _apply_text(self) -> None:
        item = self._selected()
        if isinstance(item, TextItem):
            item.text = self.text_edit.toPlainText()
            self.editor.refresh_item()

    def _apply_align(self) -> None:
        item = self._selected()
        if isinstance(item, TextItem):
            item.align = self.align_combo.currentData()
            self.editor.refresh_item()

    def _apply_size(self, value: float) -> None:
        item = self._selected()
        if isinstance(item, TextItem):
            item.size = value
        elif isinstance(item, ImageItem):
            item.width = value
        else:
            return
        self.editor.refresh_item()

    def _apply_rotation(self, value: float) -> None:
        item = self._selected()
        if item is None:
            return
        item.rotation = value
        self.editor.refresh_item()

    def _pick_color(self) -> None:
        item = self._selected()
        if not isinstance(item, TextItem):
            return
        color = QColorDialog.getColor(parent=self)
        if color.isValid():
            item.color = color.name()
            self.editor.refresh_item()

    def _on_mode_changed(self) -> None:
        self._app.config.transmit.mode = self.mode_combo.currentData()
        self._app.save_config()

    # --- transmitting --------------------------------------------------------
    @property
    def transmitting(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def send(self) -> None:
        if self.transmitting:
            return
        image = self.editor.composed_image()
        if image is None:
            QMessageBox.information(
                self, "No picture", "Choose an image to transmit first."
            )
            return
        if self._app.model is None:
            QMessageBox.warning(
                self, "Model still loading",
                "The codec checkpoint is still loading. Try again in a moment.",
            )
            return

        cfg = self._app.config
        tx_config = TxConfig(
            mode=self.mode_combo.currentData(),
            callsign=cfg.callsign,
            device=cfg.audio.output_device or None,
            level=cfg.transmit.level,
            ptt_lead_s=cfg.rig.ptt_lead_s,
            ptt_tail_s=cfg.rig.ptt_tail_s,
            model_path=cfg.model_path,
            precision=codec_precision(cfg),
        )
        self._engine = TxEngine(
            ptt=self._app.ptt(),
            model=self._app.model,
            on_state=self._signals.state.emit,
            on_error=self._signals.error.emit,
        )

        self.send_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.transmitStarted.emit()

        def run():
            ok = self._engine.transmit(image, tx_config)
            self._signals.finished.emit(ok)

        self._thread = threading.Thread(target=run, daemon=True, name="sstvae-tx")
        self._thread.start()

    def cancel(self) -> None:
        if self._engine is not None:
            self._engine.cancel()
            self.tx_status.setText("Cancelling...")

    def _on_state(self, state) -> None:
        self.tx_status.setText(state.message or state.phase.value)
        if state.phase is TxPhase.SENDING:
            self.tx_progress.setRange(0, 100)
            self.tx_progress.setValue(int(100 * state.progress))
        elif state.phase in (TxPhase.ENCODING, TxPhase.MODULATING):
            self.tx_progress.setRange(0, 0)  # indeterminate: no useful fraction
        else:
            self.tx_progress.setRange(0, 100)

    def _on_error(self, message: str) -> None:
        self.tx_status.setText(message)

    def _on_finished(self, ok: bool) -> None:
        self.send_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.tx_progress.setRange(0, 100)
        self.tx_progress.setValue(100 if ok else 0)
        self.tx_status.setText("Sent" if ok else self.tx_status.text())
        self._thread = None
        self.transmitFinished.emit()

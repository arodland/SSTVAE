"""Interactive overlay composition.

The trick that keeps this honest: **the preview is the real render.**
The background of the scene is `overlay.render()`'s output, not a
Qt-drawn approximation of it, so what the operator arranges is exactly
the picture that gets encoded -- no font metric or stroke-width
mismatch between preview and transmission. The Qt graphics items on top
are invisible interaction handles that only carry positions back into
the `OverlayDoc`.

Re-rendering on every mouse-move event would be wasteful, so moves are
coalesced onto a short timer; the handle follows the cursor at Qt speed
and the underlying picture catches up a few milliseconds later.
"""

import io

from PySide6.QtCore import QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QPen, QPixmap
from PySide6.QtWidgets import (
    QGraphicsItem,
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsView,
)

from ..overlay import CANVAS_H, CANVAS_W, ImageItem, OverlayDoc, TextItem, item_bbox, render

HANDLE_PEN = QColor(80, 200, 255)
GRIP_PX = 12


def pil_to_pixmap(img) -> QPixmap:
    """PIL image -> QPixmap, via PNG bytes.

    Going through an encoder looks wasteful next to wrapping the buffer
    in a QImage, but it sidesteps stride and lifetime bugs (a QImage
    over a numpy buffer is only valid while that buffer lives), and at a
    few frames a second on a 640x480 picture it does not matter.
    """
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG")
    pix = QPixmap()
    pix.loadFromData(buf.getvalue(), "PNG")
    return pix


class _Handle(QGraphicsRectItem):
    """A movable, selectable rectangle standing in for one overlay item."""

    def __init__(self, editor, index: int, rect: QRectF):
        super().__init__(rect)
        self._editor = editor
        self.index = index
        self.setFlags(
            QGraphicsItem.ItemIsMovable
            | QGraphicsItem.ItemIsSelectable
            | QGraphicsItem.ItemSendsGeometryChanges
        )
        pen = QPen(HANDLE_PEN)
        pen.setWidth(1)
        pen.setCosmetic(True)
        self.setPen(pen)
        self.setBrush(QColor(80, 200, 255, 22))

    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemPositionHasChanged:
            self._editor._handle_moved(self)
        elif change == QGraphicsItem.ItemSelectedHasChanged:
            self._editor._selection_changed()
        return super().itemChange(change, value)


class _Grip(QGraphicsRectItem):
    """Corner grip that resizes the selected item."""

    def __init__(self, editor):
        super().__init__(QRectF(-GRIP_PX / 2, -GRIP_PX / 2, GRIP_PX, GRIP_PX))
        self._editor = editor
        self.setFlags(
            QGraphicsItem.ItemIsMovable | QGraphicsItem.ItemSendsGeometryChanges
        )
        self.setBrush(QColor(80, 200, 255))
        pen = QPen(QColor(10, 10, 10))
        pen.setCosmetic(True)
        self.setPen(pen)
        self.setZValue(10)
        self.setCursor(Qt.SizeFDiagCursor)
        self._suppress = False

    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemPositionHasChanged and not self._suppress:
            self._editor._grip_moved(self.pos())
        return super().itemChange(change, value)

    def place(self, x: float, y: float) -> None:
        self._suppress = True
        self.setPos(x, y)
        self._suppress = False


class OverlayEditor(QGraphicsView):
    """Edits an `OverlayDoc` over a base picture."""

    docChanged = Signal()
    selectionChanged = Signal(object)  # the selected item, or None

    def __init__(self, parent=None):
        super().__init__(parent)
        self._doc = OverlayDoc()
        self._base = None  # PIL, already fitted to CANVAS_W x CANVAS_H
        self._last_rx = None
        self._handles: list[_Handle] = []
        # Set before any handle exists: tearing down and re-adding handles
        # makes Qt fire selection changes that aren't real ones.
        self._suppress_selection = False

        self._scene = QGraphicsScene(0, 0, CANVAS_W, CANVAS_H, self)
        self.setScene(self._scene)
        self.setRenderHints(self.renderHints())
        self.setDragMode(QGraphicsView.RubberBandDrag)
        self.setBackgroundBrush(QColor(30, 30, 34))

        self._pixmap_item = QGraphicsPixmapItem()
        self._pixmap_item.setZValue(-10)
        self._scene.addItem(self._pixmap_item)

        self._grip = _Grip(self)
        self._grip.setVisible(False)
        self._scene.addItem(self._grip)

        # Coalesces the re-render triggered by a stream of move events.
        self._render_timer = QTimer(self)
        self._render_timer.setSingleShot(True)
        self._render_timer.setInterval(30)
        self._render_timer.timeout.connect(self._rerender)

        self._refresh_preview()

    # --- content --------------------------------------------------------
    @property
    def doc(self) -> OverlayDoc:
        return self._doc

    def set_doc(self, doc: OverlayDoc) -> None:
        self._doc = doc
        self._rebuild_handles()
        self._refresh_preview()
        self.docChanged.emit()

    def set_base_image(self, img) -> None:
        """`img` must already be framed to the transmit size."""
        self._base = img
        self._refresh_preview()

    def set_last_rx(self, img) -> None:
        self._last_rx = img
        self._refresh_preview()

    @property
    def has_base(self) -> bool:
        return self._base is not None

    def composed_image(self):
        """The picture as it would be transmitted, or None if no base
        image has been chosen."""
        if self._base is None:
            return None
        return render(self._base, self._doc, self._last_rx)

    # --- items ----------------------------------------------------------
    def add_text(self, text: str) -> None:
        self._doc.items.append(TextItem(text=text))
        self._after_item_change(select_last=True)

    def add_last_rx_inset(self) -> None:
        self._doc.items.append(ImageItem())
        self._after_item_change(select_last=True)

    def add_image_inset(self, path: str) -> None:
        self._doc.items.append(ImageItem(source=path))
        self._after_item_change(select_last=True)

    def remove_selected(self) -> None:
        idx = self.selected_index()
        if idx is None:
            return
        del self._doc.items[idx]
        self._after_item_change()

    def selected_index(self) -> int | None:
        for h in self._handles:
            if h.isSelected():
                return h.index
        return None

    def selected_item(self):
        idx = self.selected_index()
        return self._doc.items[idx] if idx is not None else None

    def refresh_item(self) -> None:
        """Called after a property panel edits the selected item."""
        self._rebuild_handles(keep_selection=True)
        self._refresh_preview()
        self.docChanged.emit()

    def _after_item_change(self, select_last: bool = False) -> None:
        self._rebuild_handles()
        if select_last and self._handles:
            self._handles[-1].setSelected(True)
        self._refresh_preview()
        self.docChanged.emit()
        self._selection_changed()

    # --- handles --------------------------------------------------------
    def _rebuild_handles(self, keep_selection: bool = False) -> None:
        """Re-create the interaction handles from the document.

        Emits nothing while it works. Removing a selected handle and
        re-selecting its replacement both raise Qt selection changes, but
        the *selection* has not changed -- and a spurious
        `selectionChanged` here is not harmless: it makes the property
        panel reload the text box mid-keystroke, which drops the caret to
        the end of the line and makes editing anything but the last
        character impossible.
        """
        selected = self.selected_index() if keep_selection else None
        self._suppress_selection = True
        try:
            for h in self._handles:
                self._scene.removeItem(h)
            self._handles = []
            for i, item in enumerate(self._doc.items):
                x, y, w, h = item_bbox((CANVAS_W, CANVAS_H), item, self._last_rx)
                handle = _Handle(self, i, QRectF(0, 0, w, h))
                handle.setPos(x, y)
                self._scene.addItem(handle)
                self._handles.append(handle)
            if selected is not None and selected < len(self._handles):
                self._handles[selected].setSelected(True)
        finally:
            self._suppress_selection = False
        self._position_grip()

    def _handle_moved(self, handle: _Handle) -> None:
        """Write a dragged handle's position back into the document, in
        the document's normalized coordinates."""
        if handle.index >= len(self._doc.items):
            return
        item = self._doc.items[handle.index]
        pos = handle.pos()
        # item_bbox reports the drawn box, which for anchored text is
        # offset from the item's own (x, y); preserve that offset so
        # dragging doesn't make the item jump on the first pixel.
        bx, by, _, _ = item_bbox((CANVAS_W, CANVAS_H), item, self._last_rx)
        dx = bx - round(item.x * CANVAS_W)
        dy = by - round(item.y * CANVAS_H)
        item.x = (pos.x() - dx) / CANVAS_W
        item.y = (pos.y() - dy) / CANVAS_H
        self._position_grip()
        self._render_timer.start()

    def _selection_changed(self) -> None:
        self._position_grip()
        if self._suppress_selection:
            return
        self.selectionChanged.emit(self.selected_item())

    def _position_grip(self) -> None:
        idx = self.selected_index()
        if idx is None or idx >= len(self._handles):
            self._grip.setVisible(False)
            return
        h = self._handles[idx]
        r = h.sceneBoundingRect()
        self._grip.place(r.right(), r.bottom())
        self._grip.setVisible(True)

    def _grip_moved(self, pos) -> None:
        """Resize the selected item so its bottom-right corner follows
        the grip."""
        idx = self.selected_index()
        if idx is None:
            return
        item = self._doc.items[idx]
        x, y, w, h = item_bbox((CANVAS_W, CANVAS_H), item, self._last_rx)
        new_w = max(8.0, pos.x() - x)
        if isinstance(item, ImageItem):
            item.width = max(0.02, min(1.5, new_w / CANVAS_W))
        else:
            scale = new_w / max(w, 1)
            item.size = max(0.01, min(1.0, item.size * scale))
        self._rebuild_handles(keep_selection=True)
        self._render_timer.start()

    # --- preview --------------------------------------------------------
    def _refresh_preview(self) -> None:
        self._render_timer.start()

    def _rerender(self) -> None:
        from PIL import Image

        base = self._base or Image.new("RGB", (CANVAS_W, CANVAS_H), (44, 44, 50))
        self._pixmap_item.setPixmap(pil_to_pixmap(render(base, self._doc, self._last_rx)))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.fitInView(self._scene.sceneRect(), Qt.KeepAspectRatio)

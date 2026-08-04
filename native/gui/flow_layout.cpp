#include "flow_layout.hpp"

#include <QWidget>

#include <algorithm>

namespace sstvae::gui {

FlowLayout::FlowLayout(QWidget* parent, int margin, int hspace, int vspace)
    : QLayout(parent), hspace_(hspace), vspace_(vspace) {
    setContentsMargins(margin, margin, margin, margin);
}

FlowLayout::~FlowLayout() {
    while (QLayoutItem* item = takeAt(0)) delete item;
}

void FlowLayout::addItem(QLayoutItem* item) { items_.append(item); }

int FlowLayout::count() const { return static_cast<int>(items_.size()); }

QLayoutItem* FlowLayout::itemAt(int index) const {
    return (index >= 0 && index < items_.size()) ? items_.at(index) : nullptr;
}

QLayoutItem* FlowLayout::takeAt(int index) {
    return (index >= 0 && index < items_.size()) ? items_.takeAt(index) : nullptr;
}

int FlowLayout::heightForWidth(int width) const {
    return reflow(QRect(0, 0, width, 0), /*apply=*/false);
}

void FlowLayout::setGeometry(const QRect& rect) {
    QLayout::setGeometry(rect);
    reflow(rect, /*apply=*/true);
}

QSize FlowLayout::minimumSize() const {
    // **The widest single item, not the sum.** That is the whole point:
    // a row that can wrap only needs room for its largest control, so
    // the window's floor stops being the total width of every button in
    // the strip.
    QSize widest(0, 0);
    for (const QLayoutItem* item : items_) {
        widest = widest.expandedTo(item->minimumSize());
    }
    const QMargins m = contentsMargins();
    return widest + QSize(m.left() + m.right(), m.top() + m.bottom());
}

// What to lay an item out at.
//
// **Not simply `item->sizeHint()`.** `QWidgetItem::sizeHint()` returns
// *zero* on any axis whose size policy is `Ignored` -- which is a
// sensible answer to a layout that distributes stretch, and the wrong
// one here, because this layout has no stretch to give. Several widgets
// carry `Ignored` horizontally precisely so they cannot pin the
// window's width, and they came out 0 px wide: the transmit status
// label (which is where the refiner reports its progress) and the send
// progress bar were both laid out at zero and simply were not there.
// Reported by an operator who noticed the refiner had gone quiet.
//
// Asking the widget directly restores a real width. The floor is
// unaffected, because `minimumSize()` still uses `item->minimumSize()`,
// which stays zero for an Ignored axis -- so these widgets are visible
// and still cannot stop the window being narrowed.
QSize FlowLayout::item_size(const QLayoutItem* item) {
    QSize want = item->sizeHint();
    if (const QWidget* w = item->widget()) {
        if (want.width() <= 0) want.setWidth(w->sizeHint().width());
        if (want.height() <= 0) want.setHeight(w->sizeHint().height());
    }
    return want;
}

int FlowLayout::reflow(const QRect& rect, bool apply) const {
    const QMargins m = contentsMargins();
    const QRect inner = rect.adjusted(m.left(), m.top(), -m.right(), -m.bottom());
    int x = inner.x();
    int y = inner.y();
    int line_height = 0;

    for (QLayoutItem* item : items_) {
        const QSize want = item_size(item);
        int next = x + want.width();
        // Wrap when this item would cross the right edge -- unless it is
        // the first on the line, in which case there is nowhere better
        // for it to go and it is allowed to overhang rather than be
        // dropped somewhere invisible.
        if (next > inner.right() + 1 && line_height > 0) {
            x = inner.x();
            y += line_height + vspace_;
            next = x + want.width();
            line_height = 0;
        }
        if (apply) item->setGeometry(QRect(QPoint(x, y), want));
        x = next + hspace_;
        line_height = std::max(line_height, want.height());
    }
    return y + line_height - rect.y() + m.bottom();
}

}  // namespace sstvae::gui

// A layout that wraps its items onto as many lines as they need.
//
// **This exists because a narrow window mangled the controls rather
// than reflowing them.** A `QHBoxLayout` has exactly one line: below
// the width its items want, it crushes them -- a caption collapses to a
// sliver, a combo loses its text, the field at the end is cut off by
// the edge. And because several of those items are
// `QSizePolicy::Ignored` (so they cannot pin the window's width), the
// *reported* minimum understates what the row actually needs, so the
// window happily resizes to a size at which its own controls are
// broken. Reported by an operator, and reproduced at exactly the
// minimum width the window would accept.
//
// Wrapping fixes both halves at once: the controls stay legible at any
// width, and the floor drops to the widest single item rather than the
// sum of all of them -- which is what makes the window shrink usefully
// far, which is the other half of the same report.
//
// `heightForWidth` is the whole mechanism here and is *safe* in this
// direction: this layout's height grows as it gets narrower, so it can
// never act as the width-following ratchet that
// `overlay_editor.hpp` records. Qt's own Flow Layout example is the
// shape; the spacing and the comments are ours.

#ifndef SSTVAE_GUI_FLOW_LAYOUT_HPP
#define SSTVAE_GUI_FLOW_LAYOUT_HPP

#include <QLayout>
#include <QList>
#include <QRect>
#include <QLayoutItem>
#include <QSize>

namespace sstvae::gui {

class FlowLayout : public QLayout {
public:
    explicit FlowLayout(QWidget* parent = nullptr, int margin = 0, int hspace = 6,
                        int vspace = 4);
    ~FlowLayout() override;

    void addItem(QLayoutItem* item) override;
    int count() const override;
    QLayoutItem* itemAt(int index) const override;
    QLayoutItem* takeAt(int index) override;

    Qt::Orientations expandingDirections() const override { return {}; }
    bool hasHeightForWidth() const override { return true; }
    int heightForWidth(int width) const override;
    QSize sizeHint() const override { return minimumSize(); }
    QSize minimumSize() const override;
    void setGeometry(const QRect& rect) override;

private:
    // `apply` false measures, true also moves the items. One routine for
    // both so a measured height cannot disagree with a laid-out one.
    // See the .cpp: an `Ignored` axis reports zero through the item,
    // which this layout must not take literally.
    static QSize item_size(const QLayoutItem* item);
    int reflow(const QRect& rect, bool apply) const;

    QList<QLayoutItem*> items_;
    int hspace_;
    int vspace_;
};

}  // namespace sstvae::gui

#endif

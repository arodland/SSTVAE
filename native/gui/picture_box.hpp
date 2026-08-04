// The received picture: a dark viewport that fills its share of the
// pane, with the 4:3 picture centred inside it.
//
// **The box is not the picture** (decided 2026-08-03). It used to be --
// the widget was sized to exactly 4:3, so pane colour showed at the
// sides whenever the pane was wider than the picture needed. That reads
// as a gap someone forgot to close rather than as a choice, and it
// forced this widget to impose a 4:3 *height cap* on the layout, which
// is where a long run of sizing bugs came from: a cap is one more thing
// that can fight another cap, and with two on one property whichever
// `resizeEvent` ran last won.
//
// Filling the area retires all of it. This widget asks for nothing in
// particular, takes what the layout gives it, and centres a 4:3
// rectangle in the middle. Two of these come out the same size when
// their panes are the same width and the controls below them are the
// same height -- a property of the *layout*, checkable in one place,
// rather than an agreement between two widgets that has to be
// maintained.
//
// No Q_OBJECT: plain virtual overrides, nothing for moc to do.

#ifndef SSTVAE_GUI_PICTURE_BOX_HPP
#define SSTVAE_GUI_PICTURE_BOX_HPP

#include <QPixmap>
#include <QString>
#include <QWidget>

class QLabel;

namespace sstvae::gui {

class PictureBox : public QWidget {
public:
    // Small, and constant. The floor stops the box vanishing in a very
    // short window; it deliberately does not follow the width, because
    // a minimum that follows the width is a ratchet -- see
    // `overlay_editor.hpp` for what that one cost.
    static constexpr int MIN_W = 160;
    static constexpr int MIN_H = 120;

    explicit PictureBox(const QString& text, QWidget* parent = nullptr);

    // What a 4:3 picture would like at this width. Preferred only: the
    // layout may give more (the surplus becomes viewport) or less (the
    // picture narrows to stay 4:3).
    QSize sizeHint() const override;
    QSize minimumSizeHint() const override { return QSize(MIN_W, MIN_H); }

    // The picture at its own size. Kept unscaled, so a resize rescales
    // from the original rather than compounding losses.
    void set_picture(const QPixmap& picture);

    // The 4:3 rectangle inside this widget -- what a viewer would call
    // "the picture", as distinct from the viewport around it. For tests:
    // an inverted axis or a dropped centring offset still looks like a
    // working preview until something is in it.
    QRect picture_rect() const;

protected:
    void resizeEvent(QResizeEvent* event) override;
    // Draws the viewport and, inside it, the 4:3 frame the picture will
    // occupy. **The frame has to be visible before there is a picture**:
    // once the box started filling the pane there was nothing on screen
    // showing where a 640x480 would land, so an empty pane gave no clue
    // what shape it was going to be.
    void paintEvent(QPaintEvent* event) override;

private:
    void rescale();

    QLabel* label_ = nullptr;
    QPixmap source_;
};

}  // namespace sstvae::gui

#endif

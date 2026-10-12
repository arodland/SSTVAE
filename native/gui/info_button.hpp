// A small round (i) that shows a card of rich text beside it (Jeff,
// 2026-10-12): hovering opens it after the tooltip delay and leaving
// closes it; a click pins it open until a second click, Esc, or a click
// anywhere else. The card is a tooltip-style window, so it never takes
// focus or the mouse from the window under it, and it is not clipped to
// the dialog.
//
// The text comes from a function, asked each time the card opens, so it
// always describes what is chosen now; `refresh()` redraws an open card
// after the choice changes.

#ifndef SSTVAE_GUI_INFO_BUTTON_HPP
#define SSTVAE_GUI_INFO_BUTTON_HPP

#include <QToolButton>

#include <functional>

class QLabel;
class QTimer;

namespace sstvae::gui {

class InfoCard;

class InfoButton : public QToolButton {
    Q_OBJECT

public:
    static constexpr int SHOW_DELAY_MS = 350;
    static constexpr int HIDE_DELAY_MS = 150;
    static constexpr int CARD_W = 400;

    InfoButton(std::function<QString()> html, QWidget* parent = nullptr);
    ~InfoButton() override;

    // The card's text as it would be shown now.
    QString html() const { return html_ ? html_() : QString(); }
    bool card_shown() const;
    bool pinned() const { return pinned_; }
    void show_card();
    void hide_card();
    // Re-reads the text into an open card.
    void refresh();
    // The card window, for tests; null until first shown.
    QWidget* card() const;

    QSize sizeHint() const override;
    QSize minimumSizeHint() const override { return sizeHint(); }

protected:
    void paintEvent(QPaintEvent* event) override;
#if QT_VERSION >= QT_VERSION_CHECK(6, 0, 0)
    void enterEvent(QEnterEvent* event) override;
#else
    void enterEvent(QEvent* event) override;
#endif
    void leaveEvent(QEvent* event) override;
    bool eventFilter(QObject* watched, QEvent* event) override;

private:
    void place_card();
    std::function<QString()> html_;
    InfoCard* card_ = nullptr;
    QTimer* show_timer_ = nullptr;
    QTimer* hide_timer_ = nullptr;
    bool pinned_ = false;
};

}  // namespace sstvae::gui

#endif

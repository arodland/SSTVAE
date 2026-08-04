// A sticky, dismissable error strip for the top of a panel.
//
// This is the error tier of the status model: a message that must not
// be lost to the next routine status write. The transmit and receive
// panels each funnel every status they show through a single QLabel,
// so before this existed "PTT OFF FAILED ... unkey it manually" was
// guaranteed to be replaced -- by "Sent", within a second, on the same
// label (the review's F3). Errors now land here and *stay* here until
// the operator dismisses them or the owning panel clears them on a
// clean restart of the same activity.
//
// Deliberately plain: the platform style's critical icon, bold text,
// and a standard button. Palette only, no stylesheet -- see CLAUDE.md
// on QStyleSheetStyle -- and no custom chrome.

#ifndef SSTVAE_GUI_BANNER_HPP
#define SSTVAE_GUI_BANNER_HPP

#include <QFrame>
#include <QString>

class QLabel;

namespace sstvae::gui {

class ErrorBanner : public QFrame {
    Q_OBJECT

public:
    explicit ErrorBanner(QWidget* parent = nullptr);

    // Show (or replace) the message. A later error replaces an earlier
    // one -- the log keeps the history; the banner shows the newest.
    void show_error(const QString& message);

    // Programmatic clear, for "the same activity started over cleanly".
    void clear();

    QString message() const;

private:
    QLabel* icon_ = nullptr;
    QLabel* text_ = nullptr;
};

}  // namespace sstvae::gui

#endif

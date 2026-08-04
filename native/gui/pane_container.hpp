// Holds the receive and transmit halves, either side by side or in tabs.
//
// **Why this is a widget of its own rather than four lines in
// `MainWindow`.** The switch is the part that can be wrong, and every
// way it can be wrong is invisible: a panel deleted along with its old
// container (taking a decode thread with it), a panel that survives but
// comes back hidden, or a "small screen" mode whose minimum width is no
// smaller than the one it replaced. None of those look like bugs in a
// screenshot -- the first two crash or blank later, and the third is a
// feature that silently does nothing. Behind a class they are all
// reachable from a test with two plain `QWidget`s and no engines,
// which is what `tests/test_pane_container.cpp` does.
//
// **The structural fact this exists for:** a `QSplitter`'s minimum
// width is the *sum* of its children's and a collapsed pane still
// counts (it is not hidden); a `QTabWidget`'s is the *max*. So tabs are
// the only arrangement that fits both halves on a narrow panel, and no
// amount of scrolling inside the panes changes that. Side by side stays
// the default because it is the better layout -- on this mode you
// prepare the next picture while listening to the current reception.
//
// **The order of operations in `set_mode` is load-bearing**: the new
// container is built *first*, which reparents the two content widgets
// out of the old one, and only then is the old container deleted. The
// obvious alternative -- detach with `setParent(nullptr)`, delete, then
// re-add -- sets Qt's explicit-hidden flag on both panels, and a widget
// hidden that way stays hidden when it is added to a visible layout.

#ifndef SSTVAE_GUI_PANE_CONTAINER_HPP
#define SSTVAE_GUI_PANE_CONTAINER_HPP

#include <QPointer>
#include <QSize>
#include <QString>
#include <QWidget>

#include <string>

class QTabWidget;
class QVBoxLayout;

namespace sstvae::gui {

enum class PaneLayout {
    Split,  // side by side in a splitter
    Tabs,   // one at a time
};

// Which layout a config setting asks for, given the room available.
//
// `setting` is `ui.layout`: "split" and "tabs" are the operator's
// explicit choice and are honoured whatever the screen says -- an
// operator who wants both halves on a small panel may scroll the window
// off the edge if they like, and one who prefers tabs on a large screen
// is not overruled. Only "auto" measures.
//
// **Both axes.** It compared width only, which was defensible while the
// split layout's height floor was fixed. It is not once the control rows
// wrap: the height floor then *rises* as the window narrows, so a screen
// that is wide enough and short enough would get side by side and not
// fit. Either axis failing is enough to choose tabs.
//
// **"auto" is resolved against the screen, once, and never against the
// window's own width.** A live breakpoint reads like the obvious
// implementation and cannot work: while side by side is in force, the
// splitter's minimum is exactly what stops the window reaching the
// narrow width that would trigger a switch away from it. The downward
// transition is unreachable, so the layout would only ever get *more*
// cramped, never less.
PaneLayout resolve_layout(const std::string& setting, QSize available,
                          QSize split_minimum);

class PaneContainer : public QWidget {
    Q_OBJECT

public:
    // Lower bound on each pane's width when side by side. The actual
    // floor is the larger of this and what either pane asks for -- both
    // panes then share one minimum, which is what makes a 1:1 stretch
    // produce genuinely equal widths. See `build_split`.
    static constexpr int PANE_MIN_W = 380;

    // `first` and `second` are shown in that order, left to right or as
    // tabs. Ownership passes to this widget's current container, and
    // survives every mode switch -- see the class comment.
    PaneContainer(QWidget* first, QString first_title, QWidget* second,
                  QString second_title, QWidget* parent = nullptr);

    // The two panes' control strips, held to the same height.
    //
    // **This is how the pictures end up equal**, and it is the whole
    // mechanism. The panes are already locked to the same width; make
    // the controls beneath them the same height and each picture gets
    // `pane - strip`, the same number on both sides, without anything
    // being cut down to match.
    //
    // The approach this replaced measured the two *pictures* and shrank
    // the larger to the smaller. That produced equal pictures and threw
    // away the space it took -- 650x480 where 970x727 was available,
    // the difference showing as grey margin. Equalising the thing that
    // actually differs is both smaller and better.
    //
    // Optional: a test may drive this class with plain widgets.
    void set_control_strips(QWidget* first_strip, QWidget* second_strip);

    PaneLayout mode() const { return mode_; }
    void set_mode(PaneLayout mode);

    // Append a parenthesised note to the first tab's label, e.g.
    // "Receive (receiving)". No-op in split mode, where the pane is on
    // screen and needs no summary. Empty clears it.
    void set_first_note(const QString& note);

signals:
    void modeChanged(PaneLayout mode);

protected:
    void resizeEvent(QResizeEvent* event) override;


private:
    // Give both strips the taller one's height. Cheap and idempotent,
    // so it can be called again whenever a caption or a font could have
    // changed the answer.
    void equalise_strips();


    QWidget* build_split();
    QWidget* build_tabs();
    // Make `container` the one on screen. The explicit `show()` inside
    // is load-bearing on every switch after the first -- see the .cpp.
    void install(QWidget* container);
    // A pane wrapped in a titled box, so which half is which is read
    // rather than inferred from a button caption. Split mode only: a
    // tab label already says it, and a second copy inside the tab would
    // spend a line of height saying nothing.
    QWidget* titled(QWidget* content, const QString& title);

    QVBoxLayout* layout_ = nullptr;
    QWidget* container_ = nullptr;
    QTabWidget* tabs_ = nullptr;  // null unless mode_ == Tabs

    QWidget* first_ = nullptr;
    QWidget* second_ = nullptr;
    QPointer<QWidget> first_strip_;
    QPointer<QWidget> second_strip_;
    QString first_title_;
    QString second_title_;
    QString first_note_;

    bool equalise_queued_ = false;
    PaneLayout mode_ = PaneLayout::Split;
};

}  // namespace sstvae::gui

#endif

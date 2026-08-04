// The status log pane: a scrolling, timestamped record of everything
// the app has said, docked at the bottom of the main window.
//
// Fed from `log::StatusLog` via `AppState::logEntry` -- a queued
// signal, so entries may originate on any thread. On construction the
// pane backfills from `snapshot()`, which is how startup entries
// (config validation notes, the first rig status) appear even though
// they were logged before the pane existed.
//
// Filtering is by source, and it re-renders from the log rather than
// hiding blocks: the log is the truth, the pane is a view. The Copy
// button copies the *unfiltered* log in the same one-line format the
// file uses, so a pasted bug report and the on-disk log read the same.

#ifndef SSTVAE_GUI_LOG_PANE_HPP
#define SSTVAE_GUI_LOG_PANE_HPP

#include <QWidget>

#include "log/log.hpp"

class QComboBox;
class QDockWidget;
class QLabel;
class QPlainTextEdit;

namespace sstvae::gui {

class LogPane : public QWidget {
    Q_OBJECT

public:
    // Hand the filter/Copy row back, with a name and a close button
    // folded in, so it can serve as the dock's title bar. Saves a whole
    // row of height at the bottom of the window -- see main_window.cpp.
    QWidget* take_title_row(const QString& title, QDockWidget* dock);

    explicit LogPane(const log::StatusLog* log, QWidget* parent = nullptr);

    // ~5 log lines tall by default (the settled Q4 decision); the dock
    // is user-resizable from there.
    QSize sizeHint() const override;

    // Shown next to the filter so a read-only config directory is not
    // silent: pass FileWriter::error() when it is set.
    void set_file_note(const QString& note);

public slots:
    void append(qlonglong ms, const QString& source, int severity,
                const QString& text);

private slots:
    void refill();
    void copy_all();

private:
    bool passes_filter(const std::string& source) const;
    void append_line(const log::Entry& entry);

    const log::StatusLog* log_ = nullptr;
    QWidget* header_ = nullptr;
    QComboBox* filter_ = nullptr;
    QLabel* file_note_ = nullptr;
    QPlainTextEdit* text_ = nullptr;
};

}  // namespace sstvae::gui

#endif

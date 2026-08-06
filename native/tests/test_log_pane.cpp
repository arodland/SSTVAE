// The log pane and the error banner, in the parts with a right answer.
//
// What needs eyes stays with sstvae-gui-shot; what does not: the pane
// backfills from the log's snapshot (startup entries logged before the
// pane existed must appear), the filter re-renders from the log rather
// than hiding blocks, error lines render bold, appends follow the tail
// without dragging the view right, and the banner is hidden until an
// error arrives and stays until cleared.

#include <QApplication>
#include <QComboBox>
#include <QPlainTextEdit>
#include <QScrollBar>
#include <QTextBlock>

#include <string>

#include "banner.hpp"
#include "check.hpp"
#include "log/log.hpp"
#include "log_pane.hpp"

namespace check = sstvae::check;
using sstvae::gui::ErrorBanner;
using sstvae::gui::LogPane;
using sstvae::log::Severity;
using sstvae::log::StatusLog;

namespace {

QPlainTextEdit* text_of(LogPane& pane) {
    auto* text = pane.findChild<QPlainTextEdit*>();
    check::is_true(text != nullptr, "pane has a text view");
    return text;
}

void test_backfill_from_snapshot() {
    StatusLog log;
    log.append("app", Severity::Warning, "config: bad key");
    log.append("rig", Severity::Info, "Rig: 14.2300 MHz");

    // Entries logged *before* the pane existed -- the startup case.
    LogPane pane(&log);
    const QString shown = text_of(pane)->toPlainText();
    check::is_true(shown.contains("config: bad key"),
                   "pane/backfill: startup entries appear");
    check::is_true(shown.contains("14.2300"),
                   "pane/backfill: all of them");
}

void test_append_and_filter() {
    StatusLog log;
    log.append("rig", Severity::Info, "rig line");
    LogPane pane(&log);

    // Live append through the slot, as AppState::logEntry delivers it.
    log.append("tx", Severity::Error, "tx error line");
    // By value: `snapshot()` returns a vector, so binding a
    // reference to its `back()` leaves one dangling at the end of
    // the full expression -- GCC's -Wdangling-reference is right.
    const auto entry = log.snapshot().back();
    pane.append(entry.ms, "tx", static_cast<int>(Severity::Error),
                "tx error line");
    QPlainTextEdit* text = text_of(pane);
    check::is_true(text->toPlainText().contains("tx error line"),
                   "pane/append: live entries appear");

    // The error line is bold; the info line is not.
    const QTextBlock last = text->document()->lastBlock();
    check::is_true(last.begin().fragment().charFormat().fontWeight() ==
                       QFont::Bold,
                   "pane/append: error lines are bold");

    // Filtering to another source re-renders from the log.
    auto* filter = pane.findChild<QComboBox*>();
    check::is_true(filter != nullptr, "pane has a filter");
    // **By item data, not by display text.** The entries are labelled
    // for a reader ("Rig", "Receive", "Transmit") and carry the log's
    // source key ("rig", "rx", "tx") in their data, like every other
    // combo in this application -- so selecting by the visible string
    // finds nothing and the filter silently stays on "All", which is a
    // pass-shaped way for this check to test nothing.
    const int rig_index = filter->findData(QStringLiteral("rig"));
    check::is_true(rig_index >= 0, "pane/filter: the rig source is offered");
    filter->setCurrentIndex(rig_index);
    check::is_true(!text->toPlainText().contains("tx error line"),
                   "pane/filter: other sources drop out");
    check::is_true(text->toPlainText().contains("rig line"),
                   "pane/filter: the wanted source stays");
    filter->setCurrentIndex(filter->findData(QString()));
    check::is_true(text->toPlainText().contains("tx error line"),
                   "pane/filter: All restores everything");
}

void test_follow_never_scrolls_right() {
    StatusLog log;
    LogPane pane(&log);
    pane.resize(400, 160);
    pane.show();
    QPlainTextEdit* text = text_of(pane);

    const std::string long_text(300, 'x');
    for (int i = 0; i < 8; ++i) {
        log.append("rx", Severity::Info, long_text);
        const auto entry = log.snapshot().back();
        pane.append(entry.ms, "rx", static_cast<int>(Severity::Info),
                    QString::fromStdString(long_text));
    }
    check::equal(text->horizontalScrollBar()->value(), 0,
                 "pane/follow: long lines never drag the view right");
}

void test_banner_lifecycle() {
    ErrorBanner banner;
    check::is_true(banner.isHidden(), "banner/initial: hidden");
    check::is_true(banner.message().isEmpty(), "banner/initial: empty");

    banner.show_error(QStringLiteral("PTT OFF FAILED"));
    check::is_true(!banner.isHidden(), "banner/error: shown");
    check::equal(banner.message().toStdString(), std::string("PTT OFF FAILED"),
                 "banner/error: carries the message");

    // A later error replaces, never appends.
    banner.show_error(QStringLiteral("second"));
    check::equal(banner.message().toStdString(), std::string("second"),
                 "banner/error: newest message wins");

    banner.clear();
    check::is_true(banner.isHidden(), "banner/clear: hidden again");
    check::is_true(banner.message().isEmpty(), "banner/clear: and empty");
}

}  // namespace

int main(int argc, char** argv) {
    check::report_crashes_instead_of_prompting();
    qputenv("QT_QPA_PLATFORM", "offscreen");
    const QApplication app(argc, argv);

    test_backfill_from_snapshot();
    test_append_and_filter();
    test_follow_never_scrolls_right();
    test_banner_lifecycle();

    return check::report("log pane + banner");
}

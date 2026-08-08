#include "log_pane.hpp"

#include <QClipboard>
#include <QComboBox>
#include <QFontDatabase>
#include <QGuiApplication>
#include <QDockWidget>
#include <QHBoxLayout>
#include <QStyle>
#include <QToolButton>
#include <QLabel>
#include <QPlainTextEdit>
#include <QPushButton>
#include <QScrollBar>
#include <QTextCharFormat>
#include <QTextCursor>
#include <QVBoxLayout>

#include <string>

namespace sstvae::gui {

namespace {

// "All" plus the sources AppState hands out. A source not in this list
// still logs and still shows under All; it just has no dedicated
// filter entry, which is the right default for a source added later.
//
// **The label is not the key.** These read `rig, rx, tx, opt, app` --
// lowercase source keys beside a capitalised "All", naming panes the
// window itself calls Receive and Transmit, and one ("opt") that is not
// a word the operator has seen anywhere. The key goes in the item's
// data, like every other combo in this application.
struct Filter {
    const char* key;    // matches log::Entry::source; empty means "all"
    const char* label;
};
const Filter FILTERS[] = {
    {"", QT_TRANSLATE_NOOP("LogPane", "All")},
    {"rig", QT_TRANSLATE_NOOP("LogPane", "Rig")},
    {"rx", QT_TRANSLATE_NOOP("LogPane", "Receive")},
    {"tx", QT_TRANSLATE_NOOP("LogPane", "Transmit")},
    {"opt", QT_TRANSLATE_NOOP("LogPane", "Refinement")},
    {"app", QT_TRANSLATE_NOOP("LogPane", "Application")},
};

}  // namespace

LogPane::LogPane(const log::StatusLog* log, QWidget* parent)
    : QWidget(parent), log_(log) {
    auto* layout = new QVBoxLayout(this);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->setSpacing(2);

    // Kept as a widget rather than a bare layout, so it can be lifted
    // out and used as the dock's title bar (see `take_title_row`).
    header_ = new QWidget(this);
    auto* header = new QHBoxLayout(header_);
    header->setContentsMargins(4, 2, 4, 2);
    header->addWidget(new QLabel(tr("Filter:"), header_));
    filter_ = new QComboBox(header_);
    for (const Filter& entry : FILTERS) {
        filter_->addItem(tr(entry.label), QString::fromLatin1(entry.key));
    }
    connect(filter_, &QComboBox::currentIndexChanged, this, &LogPane::refill);
    header->addWidget(filter_);

    file_note_ = new QLabel(header_);
    file_note_->setWordWrap(false);
    header->addWidget(file_note_, 1);

    auto* copy = new QPushButton(tr("Copy"), header_);
    copy->setToolTip(tr("Copy the whole log to the clipboard, ignoring the filter"));
    connect(copy, &QPushButton::clicked, this, &LogPane::copy_all);
    header->addWidget(copy);
    layout->addWidget(header_);

    text_ = new QPlainTextEdit(this);
    text_->setReadOnly(true);
    text_->setFont(QFontDatabase::systemFont(QFontDatabase::FixedFont));
    text_->setLineWrapMode(QPlainTextEdit::NoWrap);
    // Matches StatusLog's own retention, so the view cannot outgrow
    // the model it claims to show.
    text_->setMaximumBlockCount(2000);
    text_->setMinimumHeight(text_->fontMetrics().lineSpacing() * 3);
    layout->addWidget(text_, 1);

    refill();
}

QSize LogPane::sizeHint() const {
    const int line = text_->fontMetrics().lineSpacing();
    return {QWidget::sizeHint().width(), line * 5 + filter_->sizeHint().height() + 10};
}

void LogPane::set_file_note(const QString& note) {
    file_note_->setText(note);
}

bool LogPane::passes_filter(const std::string& source) const {
    const QString wanted = filter_->currentData().toString();
    return wanted.isEmpty() || wanted == QString::fromStdString(source);
}

void LogPane::append_line(const log::Entry& entry) {
    // Follow the tail only when the operator is already at it, and
    // never through their cursor: setTextCursor would clear a
    // selection mid-drag, and unconditional following yanks the view
    // away from the older line they scrolled up to read.
    QScrollBar* vbar = text_->verticalScrollBar();
    const bool at_bottom = vbar->value() >= vbar->maximum();

    QTextCursor cursor(text_->document());
    cursor.movePosition(QTextCursor::End);
    QTextCharFormat format;
    // Bold rather than coloured (the traditional-look decision): an
    // error line stands out in every theme without inventing chrome.
    format.setFontWeight(entry.severity == log::Severity::Error ? QFont::Bold
                                                                : QFont::Normal);
    if (cursor.position() != 0) cursor.insertBlock();
    cursor.setCharFormat(format);
    cursor.insertText(QString::fromStdString(log::format_entry(entry)));

    if (at_bottom) {
        vbar->setValue(vbar->maximum());
        // A long line would otherwise drag the view right and take the
        // timestamp column off the left edge -- seen in the first
        // gui-shot render. Follow vertically, never horizontally.
        text_->horizontalScrollBar()->setValue(0);
    }
}

void LogPane::append(qlonglong ms, const QString& source, int severity,
                     const QString& text) {
    if (!passes_filter(source.toStdString())) return;
    log::Entry entry;
    entry.ms = ms;
    entry.source = source.toStdString();
    entry.severity = static_cast<log::Severity>(severity);
    entry.text = text.toStdString();
    append_line(entry);
}

void LogPane::refill() {
    text_->clear();
    if (log_ == nullptr) return;
    for (const log::Entry& entry : log_->snapshot()) {
        if (passes_filter(entry.source)) append_line(entry);
    }
}

QWidget* LogPane::take_title_row(const QString& title, QDockWidget* dock) {
    auto* row = qobject_cast<QHBoxLayout*>(header_->layout());
    if (row == nullptr) return nullptr;
    // The dock's name goes at the front, bold, where a title bar's would
    // be; the close button at the end, where one belongs.
    auto* name = new QLabel(title, header_);
    QFont bold = name->font();
    bold.setBold(true);
    name->setFont(bold);
    row->insertWidget(0, name);

    auto* close = new QToolButton(header_);
    close->setAutoRaise(true);
    close->setIcon(style()->standardIcon(QStyle::SP_TitleBarCloseButton));
    close->setToolTip(tr("Hide the status log (View > Status log brings it back)"));
    connect(close, &QToolButton::clicked, dock, &QDockWidget::close);
    row->addWidget(close);

    // Reparented to the dock by setTitleBarWidget; taking it out of our
    // own layout first keeps the pane from reserving space for a row it
    // no longer owns.
    header_->setParent(nullptr);
    return header_;
}

void LogPane::copy_all() {
    if (log_ == nullptr) return;
    QString all;
    for (const log::Entry& entry : log_->snapshot()) {
        all += QString::fromStdString(log::format_entry(entry));
        all += QLatin1Char('\n');
    }
    QGuiApplication::clipboard()->setText(all);
}

}  // namespace sstvae::gui

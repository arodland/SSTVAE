#include "qrss_schedule_window.hpp"

#include <QComboBox>
#include <QDateEdit>
#include <QDateTime>
#include <QFileDialog>
#include <QFileInfo>
#include <QFormLayout>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QHeaderView>
#include <QInputDialog>
#include <QLabel>
#include <QListWidget>
#include <QMenu>
#include <QMessageBox>
#include <QPushButton>
#include <QScrollArea>
#include <QScrollBar>
#include <QSpinBox>
#include <QSplitter>
#include <QTimeZone>
#include <QTimer>
#include <QTreeWidget>
#include <QVBoxLayout>
#include <QWheelEvent>

#include <algorithm>
#include <cmath>
#include <exception>
#include <map>
#include <set>
#include <utility>

#include "compose_tools.hpp"
#include "images/images.hpp"
#include "qrss_schedule.hpp"
#include "qrss_sends.hpp"
#include "qrss_timeline.hpp"
#include "qrss_tx.hpp"
#include "style.hpp"

namespace sstvae::gui {

namespace qs = qrss_schedule;

namespace {

constexpr int THUMB_W = 128;
constexpr int THUMB_H = 96;
constexpr int LIST_ICON_W = 96;
constexpr int LIST_ICON_H = 72;

QDateTime utc_of(double t) {
    return QDateTime::fromSecsSinceEpoch(static_cast<qint64>(std::llround(t)), QTimeZone::utc());
}

QString every_label(int minutes) {
    if (minutes == 0) return QObject::tr("Back to back");
    if (minutes == 120) return QObject::tr("2 hours (alternating hours)");
    if (minutes % 1440 == 0) {
        return minutes == 1440 ? QObject::tr("1 day") : QObject::tr("%1 days").arg(minutes / 1440);
    }
    if (minutes % 60 == 0) {
        return minutes == 60 ? QObject::tr("1 hour") : QObject::tr("%1 hours").arg(minutes / 60);
    }
    if (minutes < 60) return QObject::tr("%1 minutes").arg(minutes);
    return QObject::tr("%1 hours").arg(minutes / 60.0, 0, 'g', 3);
}

QString pass_word(int n) { return n == 1 ? QObject::tr("pass") : QObject::tr("passes"); }

QString duration_text(int minutes) {
    if (minutes < 60) return QObject::tr("%1 minutes").arg(minutes);
    if (minutes % 60 == 0) {
        return minutes == 60 ? QObject::tr("1 hour") : QObject::tr("%1 hours").arg(minutes / 60);
    }
    return QObject::tr("%1 hours").arg(minutes / 60.0, 0, 'g', 3);
}

QPixmap thumbnail(const std::filesystem::path& png, int w, int h) {
    const QPixmap pix(QString::fromStdString(png.string()));
    if (pix.isNull()) return {};
    return pix.scaled(w, h, Qt::KeepAspectRatio, Qt::SmoothTransformation);
}

qs::Entry form_entry(const QComboBox* mode, const QSpinBox* freq, double slot,
                     const QSpinBox* count, const QComboBox* every) {
    qs::Entry e;
    e.mode = mode->currentData().toString().toStdString();
    e.freq_hz = freq->value();
    e.first_slot = slot;
    e.count = count->value();
    e.every_min = every->currentData().toInt();
    return e;
}

const qs::Entry* entry_by_id(const std::vector<qs::Entry>& entries, const std::string& id) {
    for (const qs::Entry& e : entries) {
        if (e.id == id) return &e;
    }
    return nullptr;
}

}  // namespace

QrssScheduleWindow::QrssScheduleWindow(QrssScheduler* scheduler, QrssSends* sends,
                                       std::function<Defaults()> defaults,
                                       QrssEditor::Context editor_context, QWidget* parent)
    : QWidget(parent, Qt::Window), scheduler_(scheduler), sends_(sends),
      defaults_(std::move(defaults)),
      editor_context_(std::move(editor_context)) {
    setObjectName(QStringLiteral("qrss_schedule_window"));
    setWindowTitle(tr("QRSS schedule"));
    build();
    const Defaults d = defaults_ ? defaults_() : Defaults{};
    mode_->setCurrentIndex(std::max(0, mode_->findData(QString::fromStdString(d.mode))));
    freq_->setValue(static_cast<int>(std::lround(d.freq_hz)));
    date_->setMinimumDate(utc_of(scheduler_->now()).date());
    set_chosen_slot(qs::earliest_slot(scheduler_->now()));
    connect(scheduler_, &QrssScheduler::changed, this, &QrssScheduleWindow::refresh);
    connect(sends_, &QrssSends::changed, this, &QrssScheduleWindow::refresh_sends);
    clock_timer_ = new QTimer(this);
    clock_timer_->setInterval(30000);
    connect(clock_timer_, &QTimer::timeout, this, [this] {
        refresh();
        update_summary();
    });
    clock_timer_->start();
    refresh_sends();
    refresh();
    on_send_selected();
    resize(980, 760);
}

void QrssScheduleWindow::build() {
    auto* outer = new QVBoxLayout(this);

    // --- Upcoming: the timeline --------------------------------------
    auto* upcoming = new QGroupBox(tr("Upcoming"), this);
    auto* up_layout = new QVBoxLayout(upcoming);
    timeline_ = new QrssTimeline(upcoming);
    timeline_scroll_ = new QScrollArea(upcoming);
    timeline_scroll_->setObjectName(QStringLiteral("schedule_timeline_scroll"));
    timeline_scroll_->setWidget(timeline_);
    timeline_scroll_->setWidgetResizable(false);
    timeline_scroll_->setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    timeline_scroll_->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOn);
    timeline_scroll_->setFrameShape(QFrame::NoFrame);
    timeline_scroll_->setFixedHeight(timeline_->sizeHint().height() +
                                     timeline_scroll_->horizontalScrollBar()->sizeHint().height() +
                                     2);
    // A mouse wheel scrolls the timeline sideways: it has no other way.
    timeline_scroll_->viewport()->installEventFilter(this);
    up_layout->addWidget(timeline_scroll_);
    message_ = new QLabel(upcoming);
    message_->setObjectName(QStringLiteral("schedule_message"));
    message_->setWordWrap(true);
    message_->setText(tr("Drag a send to move it; right-click it to remove it, change its mode "
                         "or repeat it. With a Send selected, click a free slot to start "
                         "there."));
    up_layout->addWidget(message_);
    outer->addWidget(upcoming);
    connect(timeline_, &QrssTimeline::blockClicked, this, &QrssScheduleWindow::on_block_clicked);
    connect(timeline_, &QrssTimeline::slotClicked, this, [this](double slot) {
        if (selected_send().empty()) return;
        set_chosen_slot(slot);
    });
    connect(timeline_, &QrssTimeline::moveRequested, this,
            [this](const QString& entry, int send, double slot) {
                move_send(entry.toStdString(), send, slot);
            });
    connect(timeline_, &QrssTimeline::menuRequested, this,
            [this](const QString& entry, int send, const QPoint& at) {
                QMenu* menu = menu_for(entry.toStdString(), send);
                menu->setAttribute(Qt::WA_DeleteOnClose);
                menu->popup(at);
            });

    // --- Sends | Schedule -------------------------------------------
    auto* split = new QSplitter(Qt::Horizontal, this);
    split->addWidget(build_sends(split));
    split->addWidget(build_form(split));
    split->setStretchFactor(0, 2);
    split->setStretchFactor(1, 3);
    split->setChildrenCollapsible(false);
    outer->addWidget(split, 1);

    outer->addWidget(style::note(
        tr("Scheduled sends go out only while this app is running, through the same radio, "
           "level and callsign as Send. Receiving pauses from a few minutes before each send "
           "until it ends."),
        this));
}

QWidget* QrssScheduleWindow::build_sends(QWidget* parent) {
    auto* box = new QGroupBox(tr("Sends"), parent);
    auto* layout = new QVBoxLayout(box);
    sends_list_ = new QListWidget(box);
    sends_list_->setObjectName(QStringLiteral("schedule_sends"));
    sends_list_->setIconSize(QSize(LIST_ICON_W, LIST_ICON_H));
    sends_list_->setSelectionMode(QAbstractItemView::SingleSelection);
    sends_list_->setWordWrap(true);
    sends_list_->setToolTip(tr("Pictures ready to schedule, kept on disk. Tinted ones are on "
                               "the schedule. Double-click to edit."));
    connect(sends_list_, &QListWidget::itemSelectionChanged, this,
            &QrssScheduleWindow::on_send_selected);
    connect(sends_list_, &QListWidget::itemDoubleClicked, this,
            [this](QListWidgetItem*) { edit_send(selected_send()); });
    layout->addWidget(sends_list_, 1);

    auto* new_button = new QPushButton(tr("&New..."), box);
    new_button->setObjectName(QStringLiteral("sends_new"));
    new_button->setToolTip(tr("Open the Editor on a blank picture"));
    connect(new_button, &QPushButton::clicked, this, &QrssScheduleWindow::new_send);
    auto* open_file = new QPushButton(tr("&Open file..."), box);
    open_file->setObjectName(QStringLiteral("sends_open_file"));
    open_file->setToolTip(tr("Open the Editor on a picture file, framed first unless it is "
                             "exactly 640 x 480"));
    connect(open_file, &QPushButton::clicked, this, [this] {
        const QString path = QFileDialog::getOpenFileName(
            this, tr("Picture for a Send"), QString::fromStdString(editor_context_.picture_dir),
            QString::fromLatin1(compose::IMAGE_FILTER));
        if (!path.isEmpty()) new_from_file(path);
    });
    edit_send_ = new QPushButton(tr("&Edit..."), box);
    edit_send_->setObjectName(QStringLiteral("sends_edit"));
    connect(edit_send_, &QPushButton::clicked, this, [this] { edit_send(selected_send()); });
    delete_send_ = new QPushButton(tr("&Delete"), box);
    delete_send_->setObjectName(QStringLiteral("sends_delete"));
    delete_send_->setToolTip(tr("Delete the Send. What is already on the schedule still goes "
                                "out: it has its own copy."));
    connect(delete_send_, &QPushButton::clicked, this, [this] {
        const std::string id = selected_send();
        if (id.empty()) return;
        const QrssSend* s = sends_->find(id);
        if (QMessageBox::question(this, tr("Delete Send"),
                                  tr("Delete \"%1\"?").arg(QString::fromStdString(
                                      s ? s->label : id))) != QMessageBox::Yes) {
            return;
        }
        delete_send(id);
    });
    layout->addWidget(style::row(box, {new_button}));
    layout->addWidget(style::row(box, {open_file, edit_send_, delete_send_}));
    return box;
}

QWidget* QrssScheduleWindow::build_form(QWidget* parent) {
    auto* right = new QWidget(parent);
    auto* column = new QVBoxLayout(right);
    column->setContentsMargins(0, 0, 0, 0);

    form_box_ = new QGroupBox(tr("Schedule the selected Send"), right);
    form_box_->setObjectName(QStringLiteral("schedule_form"));
    auto* form = new QFormLayout(form_box_);
    thumb_ = new QLabel(form_box_);
    thumb_->setObjectName(QStringLiteral("schedule_thumb"));
    thumb_->setFixedSize(THUMB_W, THUMB_H);
    thumb_->setAlignment(Qt::AlignCenter);
    thumb_->setFrameShape(QFrame::StyledPanel);
    send_name_ = new QLabel(form_box_);
    send_name_->setObjectName(QStringLiteral("schedule_picture_name"));
    send_name_->setWordWrap(true);
    auto* pic_row = new QHBoxLayout;
    pic_row->addWidget(thumb_);
    pic_row->addWidget(send_name_, 1);
    form->addRow(pic_row);

    mode_ = new QComboBox(form_box_);
    mode_->setObjectName(QStringLiteral("schedule_mode"));
    for (const char* m : {"A", "B", "C"}) {
        const std::string s(m);
        mode_->addItem(tr("QRSS CE Mode %1 - %2 min (%3 %4)")
                           .arg(QString::fromStdString(s))
                           .arg(qrss_tx::minutes_for(s))
                           .arg(qrss_tx::passes_for(s))
                           .arg(pass_word(qrss_tx::passes_for(s))),
                       QString::fromStdString(s));
    }
    form->addRow(tr("Mode:"), mode_);

    freq_ = new QSpinBox(form_box_);
    freq_->setObjectName(QStringLiteral("schedule_freq"));
    freq_->setRange(static_cast<int>(qrss_tx::FREQ_MIN_HZ), static_cast<int>(qrss_tx::FREQ_MAX_HZ));
    freq_->setSingleStep(10);
    freq_->setSuffix(tr(" Hz"));
    freq_->setToolTip(tr("The audio frequency of the carrier, 300-2700 Hz, as on the "
                         "QRSS carrier slider."));
    form->addRow(tr("Carrier:"), freq_);

    date_ = new QDateEdit(form_box_);
    date_->setObjectName(QStringLiteral("schedule_date"));
    date_->setCalendarPopup(true);
    date_->setDisplayFormat(QStringLiteral("ddd d MMM yyyy"));
    date_->setToolTip(tr("The UTC date of the first send."));
    time_ = new QComboBox(form_box_);
    time_->setObjectName(QStringLiteral("schedule_time"));
    time_->setToolTip(tr("Sends start on a quarter hour, UTC (Z); your local time is in "
                         "brackets. Clicking a free slot on the timeline sets this too."));
    auto* when_row = new QHBoxLayout;
    when_row->addWidget(date_);
    when_row->addWidget(time_, 1);
    form->addRow(tr("First send (UTC):"), when_row);

    count_ = new QSpinBox(form_box_);
    count_->setObjectName(QStringLiteral("schedule_count"));
    count_->setRange(0, 999);
    count_->setSpecialValueText(tr("Until removed"));
    count_->setValue(2);
    count_->setToolTip(tr("How many times to send the picture. Every send is the same "
                          "picture ID, so receivers add them together."));
    every_ = new QComboBox(form_box_);
    every_->setObjectName(QStringLiteral("schedule_every"));
    for (int m : qs::EVERY_CHOICES_MIN) every_->addItem(every_label(m), m);
    every_->setToolTip(tr("From the start of one send to the start of the next. Back to "
                          "back starts the next send the half hour after the last pass "
                          "of the one before."));
    auto* repeat_row = new QHBoxLayout;
    repeat_row->addWidget(count_);
    repeat_row->addWidget(new QLabel(tr("sends, every"), form_box_));
    repeat_row->addWidget(every_, 1);
    form->addRow(tr("Repeat:"), repeat_row);

    summary_ = new QLabel(form_box_);
    summary_->setObjectName(QStringLiteral("schedule_summary"));
    summary_->setWordWrap(true);
    summary_->setTextInteractionFlags(Qt::TextSelectableByMouse);
    form->addRow(summary_);
    add_ = new QPushButton(tr("&Add to schedule"), form_box_);
    add_->setObjectName(QStringLiteral("schedule_add"));
    connect(add_, &QPushButton::clicked, this, &QrssScheduleWindow::add);
    form->addRow(style::row(form_box_, {add_}));
    column->addWidget(form_box_);

    for (QComboBox* c : {mode_, time_, every_}) {
        connect(c, &QComboBox::currentIndexChanged, this, &QrssScheduleWindow::update_summary);
    }
    for (QSpinBox* s : {freq_, count_}) {
        connect(s, &QSpinBox::valueChanged, this, &QrssScheduleWindow::update_summary);
    }
    connect(date_, &QDateEdit::dateChanged, this, [this] {
        fill_times();
        update_summary();
    });

    auto* list_box = new QGroupBox(tr("Scheduled"), right);
    auto* list_layout = new QVBoxLayout(list_box);
    table_ = new QTreeWidget(list_box);
    table_->setObjectName(QStringLiteral("schedule_table"));
    table_->setRootIsDecorated(false);
    table_->setIconSize(QSize(48, 36));
    table_->setHeaderLabels({tr("Picture"), tr("Repeats"), tr("Carrier"), tr("Next send"),
                             tr("Status")});
    table_->header()->setSectionResizeMode(QHeaderView::ResizeToContents);
    table_->header()->setStretchLastSection(true);
    table_->setMinimumHeight(table_->fontMetrics().height() * 6);
    list_layout->addWidget(table_);
    pause_ = new QPushButton(tr("Pause"), list_box);
    pause_->setObjectName(QStringLiteral("schedule_pause"));
    resume_ = new QPushButton(tr("Resume"), list_box);
    resume_->setObjectName(QStringLiteral("schedule_resume"));
    remove_ = new QPushButton(tr("Remove"), list_box);
    remove_->setObjectName(QStringLiteral("schedule_remove"));
    remove_->setToolTip(tr("Take it off the schedule. A send already on the air carries on; "
                           "Cancel on the Transmit pane stops it."));
    connect(pause_, &QPushButton::clicked, this, [this] { set_selected_enabled(false); });
    connect(resume_, &QPushButton::clicked, this, [this] { set_selected_enabled(true); });
    connect(remove_, &QPushButton::clicked, this, &QrssScheduleWindow::remove_selected);
    connect(table_, &QTreeWidget::itemSelectionChanged, this, [this] {
        const QList<QTreeWidgetItem*> sel = table_->selectedItems();
        if (sel.isEmpty()) return;
        select_entry(sel.front()->data(0, Qt::UserRole).toString().toStdString());
    });
    list_layout->addWidget(style::row(list_box, {pause_, resume_, remove_}));
    column->addWidget(list_box, 1);
    return right;
}

bool QrssScheduleWindow::eventFilter(QObject* watched, QEvent* event) {
    if (watched == timeline_scroll_->viewport() && event->type() == QEvent::Wheel) {
        auto* wheel = static_cast<QWheelEvent*>(event);
        const int delta = wheel->angleDelta().y() != 0 ? wheel->angleDelta().y()
                                                        : wheel->angleDelta().x();
        QScrollBar* bar = timeline_scroll_->horizontalScrollBar();
        bar->setValue(bar->value() - delta);
        return true;
    }
    return QWidget::eventFilter(watched, event);
}

void QrssScheduleWindow::showEvent(QShowEvent* event) {
    QWidget::showEvent(event);
    refresh_sends();
    refresh();
    update_summary();
}

// --- the Sends ---------------------------------------------------------------

std::string QrssScheduleWindow::selected_send() const {
    const QList<QListWidgetItem*> sel = sends_list_->selectedItems();
    if (sel.isEmpty()) return {};
    return sel.front()->data(Qt::UserRole).toString().toStdString();
}

void QrssScheduleWindow::select_send(const std::string& id) {
    for (int i = 0; i < sends_list_->count(); ++i) {
        QListWidgetItem* item = sends_list_->item(i);
        if (item->data(Qt::UserRole).toString().toStdString() == id) {
            sends_list_->setCurrentItem(item);
            item->setSelected(true);
            sends_list_->scrollToItem(item);
            return;
        }
    }
}

void QrssScheduleWindow::refresh_sends() {
    const std::string keep = selected_send();
    // Which Sends are on the schedule, and their next send.
    std::map<std::string, double> next;
    for (const qs::Entry& e : scheduler_->entries()) {
        if (e.send_id.empty() || qs::finished(e)) continue;
        const double slot = qs::send_slot(e, e.next);
        auto it = next.find(e.send_id);
        if (it == next.end() || slot < it->second) next[e.send_id] = slot;
    }
    QColor tint = palette().color(QPalette::Highlight);
    tint.setAlpha(55);
    {
        const QSignalBlocker block(sends_list_);
        sends_list_->clear();
        for (const QrssSend& s : sends_->sends()) {
            auto* item = new QListWidgetItem(sends_list_);
            item->setData(Qt::UserRole, QString::fromStdString(s.id));
            QString text = QString::fromStdString(s.label);
            const auto it = next.find(s.id);
            if (it != next.end()) {
                text += tr("\nscheduled, next %1").arg(local_and_utc(it->second));
                item->setBackground(tint);
                item->setData(Qt::UserRole + 1, true);
            }
            item->setText(text);
            const QPixmap pix = thumbnail(s.picture, LIST_ICON_W, LIST_ICON_H);
            if (!pix.isNull()) item->setIcon(QIcon(pix));
            if (s.id == keep) item->setSelected(true);
        }
    }
    if (selected_send() != keep) on_send_selected();
}

void QrssScheduleWindow::on_send_selected() {
    const std::string id = selected_send();
    const QrssSend* s = id.empty() ? nullptr : sends_->find(id);
    edit_send_->setEnabled(s != nullptr);
    delete_send_->setEnabled(s != nullptr);
    form_box_->setEnabled(s != nullptr);
    timeline_->set_highlighted_send(id);
    timeline_->set_hover_slot_hint(s != nullptr);
    if (s == nullptr) {
        thumb_->setPixmap(QPixmap());
        thumb_->setText(tr("No Send"));
        send_name_->setText(sends_->sends().empty()
                                ? tr("Make a Send first: New or Open file, "
                                     "on the left.")
                                : tr("Choose a Send on the left to schedule it."));
    } else {
        thumb_->setPixmap(thumbnail(s->picture, THUMB_W, THUMB_H));
        send_name_->setText(QString::fromStdString(s->label));
    }
    update_summary();
}

void QrssScheduleWindow::open_editor(const std::function<bool(QrssEditor&)>& prepare) {
    QrssEditor editor(sends_, editor_context_, this);
    if (!prepare(editor)) return;
    QString saved;
    connect(&editor, &QrssEditor::saved, this, [&saved](const QString& id) { saved = id; });
    editor.exec();
    if (!saved.isEmpty()) select_send(saved.toStdString());
}

void QrssScheduleWindow::new_send() {
    open_editor([](QrssEditor& e) {
        e.start_blank();
        return true;
    });
}

void QrssScheduleWindow::new_from_file(const QString& path) {
    open_editor([&path](QrssEditor& e) { return e.load_picture(path); });
}

void QrssScheduleWindow::edit_send(const std::string& id) {
    if (id.empty()) return;
    open_editor([&id](QrssEditor& e) { return e.open_send(id); });
}

void QrssScheduleWindow::delete_send(const std::string& id) {
    sends_->remove(id);
}

// --- the form -------------------------------------------------------------

QString QrssScheduleWindow::local_and_utc(double slot) const {
    const QDateTime local = utc_of(slot).toLocalTime();
    return tr("%1 (%2 local)")
        .arg(QString::fromStdString(qs::when(slot, scheduler_->now())),
             local.toString(QStringLiteral("HH:mm")));
}

void QrssScheduleWindow::fill_times() {
    const QVariant keep = time_->currentData();
    const QSignalBlocker block(time_);
    time_->clear();
    const QDateTime midnight(date_->date(), QTime(0, 0), QTimeZone::utc());
    for (int q = 0; q < 96; ++q) {
        const QDateTime t = midnight.addSecs(q * 900);
        time_->addItem(tr("%1Z (%2 local)")
                           .arg(t.toString(QStringLiteral("HH:mm")),
                                t.toLocalTime().toString(QStringLiteral("HH:mm"))),
                       q * 900);
    }
    time_->setCurrentIndex(keep.isValid() ? std::max(0, time_->findData(keep)) : 0);
}

double QrssScheduleWindow::chosen_slot() const {
    const QDateTime midnight(date_->date(), QTime(0, 0), QTimeZone::utc());
    return static_cast<double>(midnight.toSecsSinceEpoch()) + time_->currentData().toInt();
}

void QrssScheduleWindow::set_chosen_slot(double slot) {
    const QDateTime t = utc_of(slot);
    {
        const QSignalBlocker block(date_);
        date_->setDate(t.date());
    }
    fill_times();
    time_->setCurrentIndex(std::max(0, time_->findData(t.time().msecsSinceStartOfDay() / 1000)));
    update_summary();
}

void QrssScheduleWindow::update_summary() {
    if (summary_ == nullptr) return;
    const qs::Entry e = form_entry(mode_, freq_, chosen_slot(), count_, every_);
    const double now = scheduler_->now();
    QString why;
    if (selected_send().empty()) why = tr("choose a Send first.");
    if (why.isEmpty()) why = QString::fromStdString(qs::problem(e));
    if (why.isEmpty() && e.first_slot < qs::earliest_slot(now)) {
        why = tr("that is too soon: the earliest a send added now can start is %1.")
                  .arg(local_and_utc(qs::earliest_slot(now)));
    }
    if (why.isEmpty()) {
        if (const auto c = qs::clash(e, scheduler_->entries(), now)) {
            why = tr("the %1 send would overlap the %2 send of \"%3\": the transmitter sends "
                     "one thing at a time.")
                      .arg(QString::fromStdString(qs::when(c->slot, now)),
                           QString::fromStdString(qs::when(c->other_slot, now)),
                           QString::fromStdString(c->other));
        }
    }
    add_->setEnabled(why.isEmpty());
    if (!why.isEmpty()) {
        summary_->setText(tr("Can't add yet: %1").arg(why));
        return;
    }
    const int n = qrss_tx::passes_for(e.mode);
    QStringList times;
    const int shown = e.count == 0 ? 3 : std::min(e.count, 4);
    for (int k = 0; k < shown; ++k) times << local_and_utc(qs::send_slot(e, k));
    QString when = times.join(tr(", "));
    if (e.count == 0 || e.count > shown) when += tr(", ...");
    QString total;
    if (e.count > 0) {
        total = tr(" In all: %1 %2, %3 on the air.")
                    .arg(e.count * n)
                    .arg(pass_word(e.count * n))
                    .arg(duration_text(e.count * n * 30));
    }
    QString text = tr("%1. %2 at %3.%4")
                       .arg(QString::fromStdString(qs::describe(e)),
                            e.count == 1 ? tr("Sends") : tr("Sends start"), when, total);
    if (e.count != 1) {
        text += tr(" Every send has the same picture ID, so receivers add them up.");
    }
    summary_->setText(text.left(1).toUpper() + text.mid(1));
}

void QrssScheduleWindow::add() {
    const std::string id = selected_send();
    const QrssSend* s = sends_->find(id);
    if (s == nullptr) return;
    images::Picture picture;
    try {
        picture = images::load(s->picture.string());
    } catch (const std::exception& ex) {
        summary_->setText(tr("Can't add: the Send's picture would not load (%1).")
                              .arg(QString::fromUtf8(ex.what())));
        return;
    }
    qs::Entry e = form_entry(mode_, freq_, chosen_slot(), count_, every_);
    e.label = s->label;
    e.send_id = s->id;
    const std::string why = scheduler_->add(e, picture);
    if (!why.empty()) {
        summary_->setText(tr("Can't add: %1").arg(QString::fromStdString(why)));
        return;
    }
    summary_->setText(tr("Added \"%1\": %2.")
                          .arg(QString::fromStdString(s->label),
                               QString::fromStdString(qs::describe(e))));
    add_->setEnabled(false);
}

// --- what is scheduled ----------------------------------------------------

void QrssScheduleWindow::set_selected_enabled(bool on) {
    const QList<QTreeWidgetItem*> sel = table_->selectedItems();
    if (sel.isEmpty()) return;
    const std::string why =
        scheduler_->set_enabled(sel.front()->data(0, Qt::UserRole).toString().toStdString(), on);
    if (!why.empty()) show_result(why, {});
}

void QrssScheduleWindow::remove_selected() {
    const QList<QTreeWidgetItem*> sel = table_->selectedItems();
    if (sel.isEmpty()) return;
    remove_entry(sel.front()->data(0, Qt::UserRole).toString().toStdString());
}

void QrssScheduleWindow::select_entry(const std::string& entry) {
    selected_entry_ = entry;
    timeline_->set_selected_entry(entry);
    {
        const QSignalBlocker block(table_);
        for (int i = 0; i < table_->topLevelItemCount(); ++i) {
            QTreeWidgetItem* item = table_->topLevelItem(i);
            item->setSelected(item->data(0, Qt::UserRole).toString().toStdString() == entry);
        }
    }
    const qs::Entry* e = entry_by_id(scheduler_->entries(), entry);
    pause_->setEnabled(e != nullptr && e->enabled && !qs::finished(*e));
    resume_->setEnabled(e != nullptr && !e->enabled && !qs::finished(*e));
    remove_->setEnabled(e != nullptr);
    // Its Send, back on the left.
    if (e != nullptr && !e->send_id.empty() && sends_->find(e->send_id) != nullptr &&
        selected_send() != e->send_id) {
        select_send(e->send_id);
    }
}

void QrssScheduleWindow::on_block_clicked(const QString& entry, int) {
    select_entry(entry.toStdString());
}

void QrssScheduleWindow::show_result(const std::string& why, const QString& done) {
    if (!why.empty()) {
        message_->setText(tr("Can't do that: %1.").arg(QString::fromStdString(why)));
    } else if (!done.isEmpty()) {
        message_->setText(done);
    }
}

void QrssScheduleWindow::refresh() {
    const auto& entries = scheduler_->entries();
    {
        const QSignalBlocker block(table_);
        table_->clear();
        for (const qs::Entry& e : entries) {
            auto* item = new QTreeWidgetItem(table_);
            const QString id = QString::fromStdString(e.id);
            item->setData(0, Qt::UserRole, id);
            item->setText(0, QString::fromStdString(e.label));
            const QPixmap pix = thumbnail(e.picture, 48, 36);
            if (!pix.isNull()) item->setIcon(0, QIcon(pix));
            item->setText(1, QString::fromStdString(qs::describe(e)));
            item->setText(2, tr("%1 Hz").arg(e.freq_hz, 0, 'f', 0));
            item->setText(3, qs::finished(e) ? tr("none") : local_and_utc(qs::send_slot(e, e.next)));
            QString status;
            if (scheduler_->active() == e.id) {
                status = tr("Sending");
            } else if (qs::finished(e)) {
                status = tr("Done");
            } else if (!e.enabled) {
                status = tr("Paused");
            } else {
                status = e.next == 0   ? tr("Scheduled")
                         : e.count > 0 ? tr("%1 of %2 done").arg(e.next).arg(e.count)
                                       : tr("%1 done").arg(e.next);
            }
            item->setText(4, status);
            if (e.id == selected_entry_) item->setSelected(true);
        }
    }
    if (entry_by_id(entries, selected_entry_) == nullptr) selected_entry_.clear();
    select_entry(selected_entry_);
    refresh_timeline();
    refresh_sends();
    update_summary();
}

void QrssScheduleWindow::refresh_timeline() {
    const double now = scheduler_->now();
    // From the quarter hour before the current one, so a pass on the air
    // now is still on screen.
    const double start = std::floor(now / QrssTimeline::CELL_S) * QrssTimeline::CELL_S -
                         QrssTimeline::CELL_S;
    const double end = start + TIMELINE_HOURS * 3600.0;
    const int cells = static_cast<int>(std::lround((end - start) / QrssTimeline::CELL_S));
    timeline_->set_view(start, cells, now);
    std::vector<QrssTimeline::Block> blocks;
    std::map<std::string, QPixmap> thumbs;
    for (const qs::Entry& e : scheduler_->entries()) {
        const int n = qrss_tx::passes_for(e.mode);
        if (n == 0) continue;
        const bool active = scheduler_->active() == e.id;
        // The send on the air is just before `next`; show it too.
        int k = std::max(0, e.next - (active ? std::max(1, qs::MAX_RUN_PASSES / n) : 0));
        auto& thumb = thumbs[e.id];
        if (thumb.isNull()) thumb = QPixmap(QString::fromStdString(e.picture));
        for (; e.count == 0 || k < e.count; ++k) {
            const double slot = qs::send_slot(e, k);
            if (slot > end) break;
            const double stop = slot + n * qrss_tx::PASS_SPACING_S;
            if (stop < start) continue;
            if (k < e.next && !(active && stop > now)) continue;
            QrssTimeline::Block b;
            b.entry = e.id;
            b.send_id = e.send_id;
            b.send = k;
            b.count = e.count;
            b.slot = slot;
            b.end = stop;
            b.mode = e.mode;
            b.thumb = thumb;
            b.label = QString::fromStdString(e.label);
            b.paused = !e.enabled;
            b.on_air = active && k < e.next;
            blocks.push_back(std::move(b));
        }
    }
    timeline_->set_blocks(std::move(blocks));
}

// --- the timeline's right-click menu -------------------------------------

std::vector<int> QrssScheduleWindow::repeat_choices(const std::string& mode) {
    const int n = qrss_tx::passes_for(mode);
    std::vector<int> out;
    if (n == 0) return out;
    for (int m : {30, 45, 60, 90, 120, 180, 240, 360, 480, 720, 1440, 2880}) {
        if (m >= 30 * n) out.push_back(m);
    }
    return out;
}

QMenu* QrssScheduleWindow::menu_for(const std::string& entry, int send) {
    auto* menu = new QMenu(this);
    menu->setObjectName(QStringLiteral("schedule_block_menu"));
    const qs::Entry* e = entry_by_id(scheduler_->entries(), entry);
    if (e == nullptr) {
        menu->addAction(tr("No longer scheduled"))->setEnabled(false);
        return menu;
    }
    const std::string id = entry;
    const QString label = QString::fromStdString(e->label);
    menu->addSection(e->count == 1 ? label
                                   : tr("%1, send %2").arg(label).arg(send + 1));

    QAction* remove = menu->addAction(
        e->count == 1 ? tr("Remove") : tr("Remove all its sends"));
    remove->setObjectName(QStringLiteral("menu_remove"));
    connect(remove, &QAction::triggered, this, [this, id] { remove_entry(id); });
    if (e->count != 1 && send > e->next) {
        QAction* from = menu->addAction(tr("Remove this send and the ones after it"));
        from->setObjectName(QStringLiteral("menu_remove_from"));
        connect(from, &QAction::triggered, this, [this, id, send] { remove_from(id, send); });
    }

    QMenu* modes = menu->addMenu(tr("Change Mode"));
    modes->setObjectName(QStringLiteral("menu_change_mode"));
    for (const char* m : {"A", "B", "C"}) {
        const std::string mode(m);
        QAction* a = modes->addAction(tr("Mode %1 - %2 min")
                                          .arg(QString::fromStdString(mode))
                                          .arg(qrss_tx::minutes_for(mode)));
        a->setCheckable(true);
        a->setChecked(mode == e->mode);
        connect(a, &QAction::triggered, this, [this, id, mode] { change_mode(id, mode); });
    }

    QAction* offset = menu->addAction(tr("Change Offset... (now %1 Hz)").arg(e->freq_hz, 0, 'f', 0));
    offset->setObjectName(QStringLiteral("menu_change_offset"));
    connect(offset, &QAction::triggered, this, [this, id, hz = e->freq_hz] {
        bool ok = false;
        const int chosen = QInputDialog::getInt(
            this, tr("Change Offset"),
            tr("Audio frequency of the carrier, %1 to %2 Hz:")
                .arg(qrss_tx::FREQ_MIN_HZ, 0, 'f', 0)
                .arg(qrss_tx::FREQ_MAX_HZ, 0, 'f', 0),
            static_cast<int>(std::lround(hz)), static_cast<int>(qrss_tx::FREQ_MIN_HZ),
            static_cast<int>(qrss_tx::FREQ_MAX_HZ), 10, &ok);
        if (ok) change_offset(id, chosen);
    });

    QMenu* repeat = menu->addMenu(tr("Repeat in"));
    repeat->setObjectName(QStringLiteral("menu_repeat_in"));
    const int current = e->every_min == 0 ? qs::period_min(*e) : e->every_min;
    for (int m : repeat_choices(e->mode)) {
        QAction* a = repeat->addAction(m < 60 ? tr("%1 minutes").arg(m)
                                              : tr("%1 minutes (%2)").arg(m).arg(duration_text(m)));
        a->setData(m);
        a->setCheckable(true);
        a->setChecked(e->count != 1 && m == current);
        connect(a, &QAction::triggered, this, [this, id, m] { repeat_every(id, m); });
    }

    menu->addSeparator();
    QAction* pause = menu->addAction(e->enabled ? tr("Pause") : tr("Resume"));
    connect(pause, &QAction::triggered, this, [this, id, on = !e->enabled] {
        show_result(scheduler_->set_enabled(id, on), {});
    });
    return menu;
}

std::string QrssScheduleWindow::remove_entry(const std::string& entry) {
    const qs::Entry* e = entry_by_id(scheduler_->entries(), entry);
    if (e == nullptr) return "it is no longer on the schedule";
    const QString label = QString::fromStdString(e->label);
    scheduler_->remove(entry);
    show_result({}, tr("Removed \"%1\".").arg(label));
    return {};
}

std::string QrssScheduleWindow::remove_from(const std::string& entry, int send) {
    const qs::Entry* e = entry_by_id(scheduler_->entries(), entry);
    if (e == nullptr) return "it is no longer on the schedule";
    const QString label = QString::fromStdString(e->label);
    scheduler_->truncate(entry, send);
    show_result({}, tr("\"%1\" now stops before send %2.").arg(label).arg(send + 1));
    return {};
}

std::string QrssScheduleWindow::change_mode(const std::string& entry, const std::string& mode) {
    std::string why = scheduler_->change(entry, [&mode](qs::Entry& e) {
        e.mode = mode;
        // A repeat interval shorter than the new mode's passes cannot
        // stand: make it back to back instead.
        if (e.every_min != 0 && e.every_min < qrss_tx::minutes_for(mode)) e.every_min = 0;
    });
    show_result(why, tr("Changed to mode %1.").arg(QString::fromStdString(mode)));
    return why;
}

std::string QrssScheduleWindow::change_offset(const std::string& entry, double hz) {
    // Every send of the entry moves; the picture and its ID stay the same.
    std::string why = scheduler_->change(entry, [hz](qs::Entry& e) { e.freq_hz = hz; });
    show_result(why, tr("Moved to %1 Hz.").arg(hz, 0, 'f', 0));
    return why;
}

std::string QrssScheduleWindow::repeat_every(const std::string& entry, int minutes) {
    std::string why = scheduler_->change(entry, [minutes](qs::Entry& e) {
        e.every_min = minutes;
        // A single send repeats once more; a series keeps its count.
        if (e.count == 1) e.count = 2;
    });
    show_result(why, tr("Repeats every %1.").arg(duration_text(minutes)));
    return why;
}

std::string QrssScheduleWindow::move_send(const std::string& entry, int send, double slot) {
    const qs::Entry* e = entry_by_id(scheduler_->entries(), entry);
    if (e == nullptr) return "it is no longer on the schedule";
    // The whole series moves with the send dragged.
    const double shift = slot - qs::send_slot(*e, send);
    std::string why = scheduler_->change(entry, [shift](qs::Entry& x) { x.first_slot += shift; });
    show_result(why, tr("Moved; the send now starts %1.").arg(local_and_utc(slot)));
    return why;
}

}  // namespace sstvae::gui

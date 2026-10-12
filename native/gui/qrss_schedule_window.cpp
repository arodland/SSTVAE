#include "qrss_schedule_window.hpp"

#include <QApplication>
#include <QComboBox>
#include <QDateTime>
#include <QDateTimeEdit>
#include <QDialog>
#include <QDialogButtonBox>
#include <QDoubleSpinBox>
#include <QFileDialog>
#include <QFileInfo>
#include <QFormLayout>
#include <QGridLayout>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QInputDialog>
#include <QLabel>
#include <QListWidget>
#include <QMenu>
#include <QMimeData>
#include <QMessageBox>
#include <QPainter>
#include <QPushButton>
#include <QScrollArea>
#include <QScrollBar>
#include <QSpinBox>
#include <QSplitter>
#include <QStyledItemDelegate>
#include <QTimeZone>
#include <QTimer>
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
#include "info_button.hpp"
#include "qrss_clock.hpp"
#include "qrss_schedule.hpp"
#include "qrss_sends.hpp"
#include "qrss_timeline.hpp"
#include "qrss_tx.hpp"
#include "style.hpp"

namespace sstvae::gui {

namespace qs = qrss_schedule;

namespace {

// The Sends list, whose items drag onto the timeline as their Send's id.
class SendsList : public QListWidget {
public:
    using QListWidget::QListWidget;

protected:
    QStringList mimeTypes() const override {
        return {QString::fromLatin1(QrssTimeline::SEND_MIME)};
    }
    QMimeData* mimeData(const QList<QListWidgetItem*>& items) const override {
        if (items.isEmpty()) return nullptr;
        auto* data = new QMimeData;
        data->setData(QString::fromLatin1(QrssTimeline::SEND_MIME),
                      items.front()->data(Qt::UserRole).toString().toUtf8());
        return data;
    }
};

// Draws a Send's name, and under it in smaller, quieter type the rest
// of its text (when it is scheduled). The item's text keeps both lines,
// so its size and what a screen reader says are unchanged.
class SendDelegate : public QStyledItemDelegate {
public:
    using QStyledItemDelegate::QStyledItemDelegate;
    static QFont status_font(QFont f) {
        f.setPointSizeF(f.pointSizeF() > 0 ? f.pointSizeF() * 0.82 : 8.0);
        return f;
    }
    void paint(QPainter* p, const QStyleOptionViewItem& option,
               const QModelIndex& index) const override {
        QStyleOptionViewItem o(option);
        initStyleOption(&o, index);
        const QString all = o.text;
        // The view hands a newline over as a line separator.
        qsizetype cut = all.indexOf(QChar::LineSeparator);
        if (cut < 0) cut = all.indexOf(QLatin1Char('\n'));
        if (cut < 0) {
            QStyledItemDelegate::paint(p, option, index);
            return;
        }
        const QString name = all.left(cut);
        const QString status = all.mid(cut + 1);
        const QWidget* w = option.widget;
        QStyle* style = w ? w->style() : QApplication::style();
        // Background, selection, icon and focus, with the text left out.
        o.text.clear();
        style->drawControl(QStyle::CE_ItemViewItem, &o, p, w);
        o.text = all;
        // The same margin the plain items get, and the two lines
        // centred together as one plain item's text is.
        const int margin = style->pixelMetric(QStyle::PM_FocusFrameHMargin, nullptr, w) + 1;
        const QRect area =
            style->subElementRect(QStyle::SE_ItemViewItemText, &o, w).adjusted(margin, 0, -margin, 0);
        const QFont small_font = status_font(o.font);
        const QRect name_box = QFontMetrics(o.font).boundingRect(
            area, Qt::AlignLeft | Qt::TextWordWrap, all.left(cut));
        const QRect status_box = QFontMetrics(small_font).boundingRect(
            area, Qt::AlignLeft | Qt::TextWordWrap, all.mid(cut + 1));
        const int h = name_box.height() + 1 + status_box.height();
        const QRect text(area.left(), area.top() + std::max(0, (area.height() - h) / 2),
                         area.width(), h);
        const bool selected = o.state & QStyle::State_Selected;
        const QPalette::ColorGroup group =
            (o.state & QStyle::State_Enabled) ? QPalette::Normal : QPalette::Disabled;
        QColor ink = o.palette.color(group, selected ? QPalette::HighlightedText : QPalette::Text);
        p->save();
        p->setFont(o.font);
        p->setPen(ink);
        QRect used;
        p->drawText(text, Qt::AlignLeft | Qt::AlignTop | Qt::TextWordWrap, name, &used);
        p->setFont(small_font);
        QColor quiet = ink;
        quiet.setAlphaF(selected ? 0.85 : 0.7);
        p->setPen(quiet);
        p->drawText(text.adjusted(0, used.height() + 1, 0, 0),
                    Qt::AlignLeft | Qt::AlignTop | Qt::TextWordWrap, status);
        p->restore();
    }
};

}  // namespace

namespace {

// The Repeat spin box's top value, shown as "until removed".
constexpr int TIMES_FOREVER = 100;
constexpr int LIST_ICON_W = 96;
constexpr int LIST_ICON_H = 72;

QDateTime utc_of(double t) {
    return QDateTime::fromSecsSinceEpoch(static_cast<qint64>(std::llround(t)), QTimeZone::utc());
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

// Under the waveform: only L's one caution, since getting it wrong
// splatters; the rest is on the (i) card (Jeff, 2026-10-12).
QString waveform_note(const std::string& waveform) {
    if (waveform == qrss_tx::WAVEFORM_L) return QObject::tr("Linear: keep the ALC out.");
    return {};
}

// The (i) cards' colours: the window's own, with a green and an amber
// dark enough to read on a light window and light enough on a dark one.
QString hex(const QColor& c) { return c.name(QColor::HexRgb); }
bool dark_theme(const QPalette& pal) { return pal.color(QPalette::Base).lightness() < 128; }
QString good_colour(const QPalette& pal) {
    return dark_theme(pal) ? QStringLiteral("#5cc983") : QStringLiteral("#23824a");
}
QString cost_colour(const QPalette& pal) {
    return dark_theme(pal) ? QStringLiteral("#f0a648") : QStringLiteral("#a85d00");
}

QString bullet_list(const QStringList& items) {
    QString out;
    // A table, not <ul>: a label's list indent is a fixed 40 px.
    for (const QString& t : items) {
        out += QStringLiteral("<tr><td valign=\"top\" style=\"padding-right:4px;\">\u2022</td>"
                              "<td valign=\"top\" style=\"padding-bottom:3px;\">%1</td></tr>")
                   .arg(t.toHtmlEscaped());
    }
    return QStringLiteral("<table cellspacing=\"0\" cellpadding=\"0\" style=\"margin-top:3px;\">%1</table>")
        .arg(out);
}

// Title, one line of what it is, "Better at" beside "Costs", then
// whatever `extra` adds (a table, a footnote).
QString info_html(const QPalette& pal, const QString& title, const QString& what,
                  const QStringList& good, const QStringList& cost, const QString& extra) {
    const QString muted = hex(style::secondary_text(pal));
    const auto head = [](const QString& colour, const QString& text) {
        return QStringLiteral("<p style=\"margin:0;\"><span style=\"color:%1; font-size:small; "
                              "font-weight:600; letter-spacing:1px;\">%2</span></p>")
            .arg(colour, text.toUpper().toHtmlEscaped());
    };
    return QStringLiteral(
               "<p style=\"margin:0; font-weight:700;\">%1</p>"
               "<p style=\"margin:2px 0 8px 0; color:%2;\">%3</p>"
               "<table width=\"100%\" cellspacing=\"0\" cellpadding=\"0\"><tr>"
               "<td width=\"50%\" valign=\"top\">%4%5</td>"
               "<td width=\"50%\" valign=\"top\" style=\"padding-left:10px;\">%6%7</td>"
               "</tr></table>%8")
        .arg(title.toHtmlEscaped(), muted, what.toHtmlEscaped(),
             head(good_colour(pal), QObject::tr("Better at")), bullet_list(good),
             head(cost_colour(pal), QObject::tr("Costs")), bullet_list(cost), extra);
}

QString footnote(const QPalette& pal, const QString& text) {
    return QStringLiteral("<p style=\"margin:8px 0 0 0; color:%1; font-size:small;\">%2</p>")
        .arg(hex(style::secondary_text(pal)), text.toHtmlEscaped());
}

// The Waveform (i): the chosen waveform, then L beside CE. Numbers from
// the spec (qrss-spec rev 13 and docs/qrss/waveform-l.md).
QString waveform_card(const QPalette& pal, const std::string& waveform) {
    const bool l = waveform == qrss_tx::WAVEFORM_L;
    QString body;
    if (l) {
        body = info_html(
            pal, QObject::tr("L: linear, about 22 Hz"),
            QObject::tr("A pilot tone with the data just below it. The Carrier is the pilot; "
                        "the data sits 3 to 22 Hz under it."),
            {QObject::tr("Needs 3.2 dB less SNR at the same average power"),
             QObject::tr("Less than half CE's width, so more stations fit"),
             QObject::tr("Header decodes to about -28 dB")},
            {QObject::tr("Needs a linear chain: ALC out, linear amplifier"),
             QObject::tr("At the same peak power it only ties with CE"),
             QObject::tr("Older listeners, and any on --just-ce, don't hear it"),
             QObject::tr("Not for Si5351 beacons")},
            {});
    } else {
        body = info_html(
            pal, QObject::tr("CE: constant envelope, about 50 Hz"),
            QObject::tr("A steady-amplitude, phase-modulated carrier. Nothing the transmitter "
                        "does to amplitude can hurt it."),
            {QObject::tr("Any amplifier, class C or D included"), QObject::tr("ALC is harmless"),
             QObject::tr("Every QRSSTVAE listener hears it"),
             QObject::tr("Si5351 beacons can send it")},
            {QObject::tr("Needs 3.2 dB more SNR at the same average power"),
             QObject::tr("More than twice as wide as L")},
            {});
    }
    const QString line = hex(pal.color(QPalette::Mid));
    const QString muted = hex(style::secondary_text(pal));
    const auto cell = [l](bool col_l, const QString& t) {
        const bool sel = col_l == l;
        return QStringLiteral("<td align=\"right\" style=\"font-family:monospace;%1\">%2</td>")
            .arg(sel ? QStringLiteral(" font-weight:700;") : QString(), t.toHtmlEscaped());
    };
    QString rows;
    const struct { const char* what; const char* l; const char* ce; } table[] = {
        {"Width (99%)", "21.5 Hz", "51.5 Hz"},
        {"Good picture, 1 pass", "-14.0 dB", "-10.9 dB"},
        {"Good picture, 16 passes", "-29.6 dB", "-26.4 dB"},
        {"Amplifier", "linear", "any"},
    };
    for (const auto& r : table) {
        rows += QStringLiteral("<tr><td style=\"color:%1;\">%2</td>%3%4</tr>")
                    .arg(muted, QObject::tr(r.what).toHtmlEscaped(),
                         cell(true, QObject::tr(r.l)), cell(false, QObject::tr(r.ce)));
    }
    body += QStringLiteral(
                "<table width=\"100%\" cellspacing=\"0\" cellpadding=\"2\" "
                "style=\"margin-top:10px; border-top:1px solid %1;\">"
                "<tr><td></td>%2%3</tr>%4</table>")
                .arg(line, cell(true, QStringLiteral("L")), cell(false, QStringLiteral("CE")), rows);
    body += footnote(pal, QObject::tr("SNR in 2500 Hz on a quiet path, at the same average "
                                      "power. Passes of either waveform add into the same "
                                      "picture."));
    return body;
}

// The Mode (i): what the chosen mode sends and what it trades.
QString mode_card(const QPalette& pal, const std::string& mode) {
    QString body;
    if (mode == "C") {
        body = info_html(pal, QObject::tr("Mode C: 90 minutes, 3 passes"),
                         QObject::tr("All three groups of the picture, one pass each."),
                         {QObject::tr("The most detail"),
                          QObject::tr("Pass 1 alone still gives a rougher picture")},
                         {QObject::tr("Three times the time on the air"),
                          QObject::tr("Full detail needs all three passes heard")},
                         {});
    } else if (mode == "B") {
        body = info_html(pal, QObject::tr("Mode B: 60 minutes, 2 passes"),
                         QObject::tr("Mode A's pass, then a second pass adding the next group."),
                         {QObject::tr("More detail than A"),
                          QObject::tr("Pass 1 alone still gives a rougher picture")},
                         {QObject::tr("Twice the time on the air"),
                          QObject::tr("The detail needs both passes heard")},
                         {});
    } else {
        body = info_html(pal, QObject::tr("Mode A: 30 minutes, 1 pass"),
                         QObject::tr("The picture's first group, in one pass."),
                         {QObject::tr("Shortest time on the air"),
                          QObject::tr("Easiest to repeat, so receivers can add the passes")},
                         {QObject::tr("The least detail of the three")}, {});
    }
    return body + footnote(pal, QObject::tr("Each pass is 30 minutes on a quarter hour."));
}

// A box's heading, as in the mockups: small, letter-spaced capitals in
// the quieter text colour (Jeff, 2026-10-12: "the fonts in your mockups
// are better"). A label rather than the group box's own title, whose
// font would also be every child's.
QFont heading_font(QFont f) {
    f.setCapitalization(QFont::AllUppercase);
    f.setLetterSpacing(QFont::PercentageSpacing, 108);
    f.setWeight(QFont::DemiBold);
    if (f.pointSizeF() > 0) f.setPointSizeF(f.pointSizeF() * 0.85);
    return f;
}

QLabel* heading(const QString& text, QWidget* parent) {
    auto* l = new QLabel(text, parent);
    l->setFont(heading_font(l->font()));
    style::dim(l);
    return l;
}

// `times` repeats after the first send (TIMES_FOREVER: until removed),
// `every` minutes from the start of one to the next; at the mode's own
// length that is back to back.
qs::Entry form_entry(const QComboBox* mode, const QComboBox* waveform, const QSpinBox* freq,
                     double slot, int times, int every) {
    qs::Entry e;
    e.mode = mode->currentData().toString().toStdString();
    e.waveform = waveform->currentData().toString().toStdString();
    e.freq_hz = freq->value();
    e.first_slot = slot;
    e.count = times >= TIMES_FOREVER ? 0 : times + 1;
    e.every_min = (e.count == 1 || every <= qrss_tx::minutes_for(e.mode)) ? 0 : every;
    return e;
}

// "[0] times", whose top value reads "until removed".
class TimesSpin : public QSpinBox {
public:
    using QSpinBox::QSpinBox;

protected:
    QString textFromValue(int v) const override {
        return v >= TIMES_FOREVER ? QStringLiteral("∞") : QSpinBox::textFromValue(v);
    }
    int valueFromText(const QString& text) const override {
        return text.trimmed() == QStringLiteral("∞") ? TIMES_FOREVER
                                                         : QSpinBox::valueFromText(text);
    }
    QValidator::State validate(QString& text, int& pos) const override {
        if (text.trimmed() == QStringLiteral("∞")) return QValidator::Acceptable;
        return QSpinBox::validate(text, pos);
    }
};

const qs::Entry* entry_by_id(const std::vector<qs::Entry>& entries, const std::string& id) {
    for (const qs::Entry& e : entries) {
        if (e.id == id) return &e;
    }
    return nullptr;
}

}  // namespace

// The Send's picture under its choices: the column's full width, 4:3,
// no frame (Jeff, 2026-10-12). It takes only the height left over, so
// the choices above it are never squeezed; short of 3/4 of its width,
// the picture narrows instead.
class SendPicture : public QWidget {
public:
    explicit SendPicture(QWidget* parent) : QWidget(parent) {
        setObjectName(QStringLiteral("schedule_picture"));
        setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Ignored);
    }
    void set_pixmap(const QPixmap& pix) {
        pix_ = pix;
        update();
    }
    const QPixmap& pixmap() const { return pix_; }
    QSize sizeHint() const override { return {320, 240}; }
    QSize minimumSizeHint() const override { return {120, 90}; }

protected:
    void paintEvent(QPaintEvent*) override {
        if (pix_.isNull()) return;
        QPainter p(this);
        p.setRenderHint(QPainter::SmoothPixmapTransform);
        const QSize sz = pix_.size().scaled(size(), Qt::KeepAspectRatio);
        p.drawPixmap(QRect(QPoint(0, 0), sz), pix_);
    }

private:
    QPixmap pix_;
};

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
    waveform_->setCurrentIndex(
        std::max(0, waveform_->findData(QString::fromStdString(d.waveform))));
    waveform_note_->setText(waveform_note(d.waveform));
    set_chosen_slot(qs::earliest_slot(scheduler_->now()));
    connect(scheduler_, &QrssScheduler::changed, this, &QrssScheduleWindow::refresh);
    connect(sends_, &QrssSends::changed, this, &QrssScheduleWindow::refresh_sends);
    clock_timer_ = new QTimer(this);
    clock_timer_->setInterval(30000);
    connect(clock_timer_, &QTimer::timeout, this, [this] {
        // A first send the clock has passed moves up to the earliest.
        const double earliest = qs::earliest_slot(scheduler_->now());
        if (chosen_slot_ < earliest) set_chosen_slot(earliest);
        refresh();
        update_summary();
    });
    clock_timer_->start();
    refresh_sends();
    refresh();
    on_send_selected();
    resize(1040, 800);
}

void QrssScheduleWindow::build() {
    auto* outer = new QVBoxLayout(this);

    // --- Upcoming: the timeline --------------------------------------
    auto* upcoming = new QGroupBox(this);
    upcoming->setObjectName(QStringLiteral("schedule_upcoming"));
    auto* up_layout = new QVBoxLayout(upcoming);
    up_layout->addWidget(heading(tr("Upcoming"), upcoming));
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
    message_->setText(tr("Drag a send to move it; right-click it to pause or remove it, or "
                         "change its mode, offset or repeat. Drag a Send here from below to schedule it, or "
                         "with one selected, click a free slot to start there. Green lines are frequency changes: drag one to "
                         "move it, right-click it to change its frequency or remove it, or right-click "
                         "free time to add one."));
    auto* message_row = new QHBoxLayout;
    message_row->addWidget(message_, 1);
    auto* add_change = new QPushButton(tr("Add frequency change..."), upcoming);
    add_change->setObjectName(QStringLiteral("add_freq_change"));
    connect(add_change, &QPushButton::clicked, this, [this] {
        const double now = scheduler_->now();
        double at = std::ceil((now + 300.0) / 60.0) * 60.0;
        double hz = defaults_().dial_hz;
        if (ask_freq_change(tr("Add frequency change"), &at, &hz)) add_freq_change(at, hz);
    });
    message_row->addWidget(add_change, 0, Qt::AlignTop);
    up_layout->addLayout(message_row);
    outer->addWidget(upcoming);
    connect(timeline_, &QrssTimeline::blockClicked, this, &QrssScheduleWindow::on_block_clicked);
    connect(timeline_, &QrssTimeline::slotClicked, this, [this](double slot) {
        if (selected_send().empty()) return;
        set_chosen_slot(slot);
    });
    connect(timeline_, &QrssTimeline::sendDropped, this,
            [this](const QString& send_id, double slot) {
                drop_send(send_id.toStdString(), slot);
            });
    connect(timeline_, &QrssTimeline::moveRequested, this,
            [this](const QString& entry, int send, double slot) {
                move_send(entry.toStdString(), send, slot);
            });
    connect(timeline_, &QrssTimeline::markMoveRequested, this,
            [this](const QString& id, double at) { move_freq_change(id.toStdString(), at); });
    connect(timeline_, &QrssTimeline::markMenuRequested, this,
            [this](const QString& id, const QPoint& at) {
                QMenu* menu = change_menu_for(id.toStdString());
                menu->setAttribute(Qt::WA_DeleteOnClose);
                menu->popup(at);
            });
    connect(timeline_, &QrssTimeline::emptyMenuRequested, this,
            [this](double t, const QPoint& at) {
                QMenu* menu = empty_menu_for(t);
                menu->setAttribute(Qt::WA_DeleteOnClose);
                menu->popup(at);
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
           "level and callsign as Send. Each gets ready a minute ahead and keys just before its "
           "quarter hour; receiving pauses only while it is on the air."),
        this));
}

QWidget* QrssScheduleWindow::build_sends(QWidget* parent) {
    auto* box = new QGroupBox(parent);
    box->setObjectName(QStringLiteral("schedule_sends_box"));
    auto* layout = new QVBoxLayout(box);
    {
        auto* title = new QHBoxLayout;
        title->addWidget(heading(tr("Sends"), box));
        auto* how = new QLabel(tr("click, or click and drag to add to timeline"), box);
        how->setObjectName(QStringLiteral("schedule_sends_how"));
        style::dim(how);
        title->addWidget(how, 1);
        layout->addLayout(title);
    }
    sends_list_ = new SendsList(box);
    sends_list_->setObjectName(QStringLiteral("schedule_sends"));
    // Drag one onto the timeline to schedule it there.
    sends_list_->setDragEnabled(true);
    sends_list_->setDragDropMode(QAbstractItemView::DragOnly);
    sends_list_->setIconSize(QSize(LIST_ICON_W, LIST_ICON_H));
    sends_list_->setSelectionMode(QAbstractItemView::SingleSelection);
    sends_list_->setWordWrap(true);
    sends_list_->setItemDelegate(new SendDelegate(sends_list_));
    sends_list_->setToolTip(tr("Pictures ready to schedule, kept on disk. Tinted ones are on "
                               "the schedule. Double-click to edit; drag one onto the timeline above to schedule it "
                               "there."));
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
    // Two columns: the Send's choices on the left, At Time on the right,
    // and under At Time the Add button, its bottom level with the left
    // column's, since both columns feed it (Jeff, 2026-10-12).
    auto* both = new QWidget(parent);
    auto* columns = new QHBoxLayout(both);
    columns->setContentsMargins(0, 0, 0, 0);

    // --- the Send ----------------------------------------------------
    form_box_ = new QGroupBox(both);
    form_box_->setObjectName(QStringLiteral("schedule_form"));
    auto* left = new QVBoxLayout(form_box_);
    left->addWidget(heading(tr("Send"), form_box_));
    auto* grid = new QGridLayout;
    grid->setColumnStretch(1, 1);
    int row = 0;
    send_name_ = new QLabel(form_box_);
    send_name_->setObjectName(QStringLiteral("schedule_picture_name"));
    send_name_->setWordWrap(true);
    QFont name_font = send_name_->font();
    name_font.setBold(true);
    send_name_->setFont(name_font);
    grid->addWidget(send_name_, row++, 0, 1, 2);

    mode_ = new QComboBox(form_box_);
    mode_->setObjectName(QStringLiteral("schedule_mode"));
    for (const char* m : {"A", "B", "C"}) {
        const std::string s(m);
        mode_->addItem(tr("QRSS Mode %1 - %2 min (%3 %4)")
                           .arg(QString::fromStdString(s))
                           .arg(qrss_tx::minutes_for(s))
                           .arg(qrss_tx::passes_for(s))
                           .arg(pass_word(qrss_tx::passes_for(s))),
                       QString::fromStdString(s));
    }
    grid->addWidget(new QLabel(tr("Mode:"), form_box_), row, 0);
    mode_info_ = new InfoButton(
        [this] { return mode_card(palette(), mode_->currentData().toString().toStdString()); },
        form_box_);
    mode_info_->setObjectName(QStringLiteral("schedule_mode_info"));
    auto* mode_row = new QHBoxLayout;
    mode_row->addWidget(mode_, 1);
    mode_row->addWidget(mode_info_);
    grid->addLayout(mode_row, row++, 1);

    waveform_ = new QComboBox(form_box_);
    waveform_->setObjectName(QStringLiteral("schedule_waveform"));
    waveform_->addItem(tr("CE - constant envelope, 50 Hz"),
                       QString::fromLatin1(qrss_tx::WAVEFORM_CE));
    waveform_->addItem(tr("L - linear, 22 Hz"), QString::fromLatin1(qrss_tx::WAVEFORM_L));
    waveform_->setToolTip(tr("How the picture goes on the air. Both carry the same picture, "
                             "and passes of either add into it at a receiver."));
    grid->addWidget(new QLabel(tr("Waveform:"), form_box_), row, 0);
    waveform_info_ = new InfoButton(
        [this] {
            return waveform_card(palette(), waveform_->currentData().toString().toStdString());
        },
        form_box_);
    waveform_info_->setObjectName(QStringLiteral("schedule_waveform_info"));
    auto* waveform_row = new QHBoxLayout;
    waveform_row->addWidget(waveform_, 1);
    waveform_row->addWidget(waveform_info_);
    grid->addLayout(waveform_row, row++, 1);
    waveform_note_ = new QLabel(form_box_);
    waveform_note_->setObjectName(QStringLiteral("schedule_waveform_note"));
    waveform_note_->setWordWrap(true);
    style::dim(waveform_note_);
    grid->addWidget(waveform_note_, row++, 1);

    freq_ = new QSpinBox(form_box_);
    freq_->setObjectName(QStringLiteral("schedule_freq"));
    freq_->setRange(static_cast<int>(qrss_tx::FREQ_MIN_HZ), static_cast<int>(qrss_tx::FREQ_MAX_HZ));
    freq_->setSingleStep(10);
    freq_->setSuffix(tr(" Hz"));
    freq_->setToolTip(tr("The audio frequency of the carrier, 300-2700 Hz, as on the "
                         "QRSS carrier slider."));
    grid->addWidget(new QLabel(tr("Carrier:"), form_box_), row, 0);
    grid->addWidget(freq_, row++, 1, Qt::AlignLeft);

    // Repeat: "[0] times", then "every [30] minutes", greyed until there
    // is a repeat; the minutes start at the mode's own length, back to
    // back, and go up in quarter hours.
    times_ = new TimesSpin(form_box_);
    times_->setObjectName(QStringLiteral("schedule_times"));
    times_->setRange(0, TIMES_FOREVER);
    times_->setValue(0);
    times_->setToolTip(tr("How many more times to send the picture after the first; at the "
                          "top (∞) until removed. Every send has the same picture ID, so "
                          "receivers add them together."));
    auto* times_row = new QHBoxLayout;
    times_row->addWidget(times_);
    times_row->addWidget(new QLabel(tr("times"), form_box_));
    times_row->addStretch(1);
    grid->addWidget(new QLabel(tr("Repeat:"), form_box_), row, 0);
    grid->addLayout(times_row, row++, 1);

    every_label_ = new QLabel(tr("every"), form_box_);
    every_label_->setObjectName(QStringLiteral("schedule_every_label"));
    every_ = new QSpinBox(form_box_);
    every_->setObjectName(QStringLiteral("schedule_every"));
    every_->setRange(qrss_tx::minutes_for("A"), 7 * 24 * 60);
    every_->setSingleStep(static_cast<int>(qs::QUARTER_S / 60));
    every_->setToolTip(tr("Minutes from the start of one send to the start of the next. At "
                          "the mode's own length the sends go back to back."));
    every_unit_ = new QLabel(tr("minutes"), form_box_);
    auto* every_row = new QHBoxLayout;
    every_row->addWidget(every_);
    every_row->addWidget(every_unit_);
    every_row->addStretch(1);
    grid->addWidget(every_label_, row, 0, Qt::AlignRight | Qt::AlignVCenter);
    grid->addLayout(every_row, row++, 1);
    left->addLayout(grid);

    picture_ = new SendPicture(form_box_);
    left->addWidget(picture_, 1);
    columns->addWidget(form_box_, 1);

    // --- At Time ------------------------------------------------------
    auto* right = new QVBoxLayout;
    at_box_ = new QGroupBox(both);
    at_box_->setObjectName(QStringLiteral("schedule_at_time"));
    auto* at = new QVBoxLayout(at_box_);
    at->addWidget(heading(tr("At Time"), at_box_));
    auto* faces = new QHBoxLayout;
    auto clock_column = [this, faces](QrssClockFace* face, QLabel* caption, QLabel* readout) {
        auto* col = new QVBoxLayout;
        caption->setAlignment(Qt::AlignHCenter);
        caption->setFont(heading_font(caption->font()));
        style::dim(caption);
        readout->setAlignment(Qt::AlignHCenter);
        readout->setTextFormat(Qt::RichText);
        col->addWidget(caption);
        col->addWidget(face, 0, Qt::AlignHCenter);
        col->addWidget(readout);
        faces->addLayout(col);
        connect(face, &QrssClockFace::stepRequested, this, &QrssScheduleWindow::step_slot);
    };
    clock_utc_ = new QrssClockFace(true, at_box_);
    clock_local_ = new QrssClockFace(false, at_box_);
    utc_readout_ = new QLabel(at_box_);
    utc_readout_->setObjectName(QStringLiteral("at_time_utc_readout"));
    local_caption_ = new QLabel(at_box_);
    local_caption_->setObjectName(QStringLiteral("at_time_local_caption"));
    local_readout_ = new QLabel(at_box_);
    local_readout_->setObjectName(QStringLiteral("at_time_local_readout"));
    clock_column(clock_utc_, new QLabel(tr("UTC"), at_box_), utc_readout_);
    clock_column(clock_local_, local_caption_, local_readout_);
    at->addLayout(faces);

    auto* hint = new QHBoxLayout;
    hint->setSpacing(4);
    auto* wheel = new QLabel(tr("Wheel: ±15 min ·"), at_box_);
    auto* shift_wheel = new QLabel(tr("+wheel: ±1 hour"), at_box_);
    for (QLabel* l : {wheel, shift_wheel}) style::dim(l);
    hint->addStretch(1);
    hint->addWidget(wheel, 0, Qt::AlignVCenter);
    hint->addWidget(new ShiftKeyCap(at_box_), 0, Qt::AlignVCenter);
    hint->addWidget(shift_wheel, 0, Qt::AlignVCenter);
    hint->addStretch(1);
    at->addLayout(hint);
    right->addWidget(at_box_);
    right->addStretch(1);

    summary_ = new QLabel(both);
    summary_->setObjectName(QStringLiteral("schedule_summary"));
    summary_->setWordWrap(true);
    summary_->setTextInteractionFlags(Qt::TextSelectableByMouse);
    right->addWidget(summary_);

    add_ = new QPushButton(tr("&Add to schedule"), both);
    add_->setObjectName(QStringLiteral("schedule_add"));
    add_->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
    add_->setMinimumHeight(add_->sizeHint().height() * 3 / 2);
    {
        // The accent, while it can be clicked; greyed as usual when not.
        QPalette pal = add_->palette();
        for (QPalette::ColorGroup g : {QPalette::Active, QPalette::Inactive}) {
            pal.setColor(g, QPalette::Button, pal.color(g, QPalette::Highlight));
            pal.setColor(g, QPalette::ButtonText, pal.color(g, QPalette::HighlightedText));
        }
        add_->setPalette(pal);
        QFont f = add_->font();
        f.setBold(true);
        add_->setFont(f);
    }
    connect(add_, &QPushButton::clicked, this, [this] { add(); });
    right->addWidget(add_);
    columns->addLayout(right, 1);

    for (QComboBox* c : {mode_, waveform_}) {
        connect(c, &QComboBox::currentIndexChanged, this, &QrssScheduleWindow::sync_repeat);
    }
    connect(times_, &QSpinBox::valueChanged, this, &QrssScheduleWindow::sync_repeat);
    for (QSpinBox* s : {freq_, every_}) {
        connect(s, &QSpinBox::valueChanged, this, &QrssScheduleWindow::update_summary);
    }
    // A typed interval lands on a quarter hour.
    connect(every_, &QSpinBox::editingFinished, this, [this] {
        const int q = static_cast<int>(qs::QUARTER_S / 60);
        every_->setValue(std::max(every_->minimum(), (every_->value() + q / 2) / q * q));
    });
    every_->setValue(qrss_tx::minutes_for("A"));
    sync_repeat();
    return both;
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
    at_box_->setEnabled(s != nullptr);
    timeline_->set_highlighted_send(id);
    timeline_->set_hover_slot_hint(s != nullptr);
    if (s == nullptr) {
        picture_->set_pixmap(QPixmap());
        send_name_->setText(sends_->sends().empty()
                                ? tr("Make a Send first: New or Open file, "
                                     "on the left.")
                                : tr("Choose a Send on the left to schedule it."));
    } else {
        picture_->set_pixmap(QPixmap(QString::fromStdString(s->picture.string())));
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

double QrssScheduleWindow::chosen_slot() const { return chosen_slot_; }

void QrssScheduleWindow::set_chosen_slot(double slot) {
    chosen_slot_ = slot;
    clock_utc_->set_time(slot);
    clock_local_->set_time(slot);
    const QDateTime local = utc_of(slot).toLocalTime();
    local_caption_->setText(tr("Local (%1)").arg(local.timeZoneAbbreviation()));
    const auto readout = [](const QrssClockFace* face) {
        return QStringLiteral("<span style=\"font-family:monospace; font-size:large;\">%1</span>"
                              "<br><small>%2</small>")
            .arg(face->readout().toHtmlEscaped(), face->date_text().toHtmlEscaped());
    };
    utc_readout_->setText(readout(clock_utc_));
    local_readout_->setText(readout(clock_local_));
    update_summary();
}

void QrssScheduleWindow::step_slot(int seconds) {
    set_chosen_slot(std::max(qs::earliest_slot(scheduler_->now()), chosen_slot_ + seconds));
}

void QrssScheduleWindow::sync_repeat() {
    if (every_ == nullptr) return;
    // The interval can't be shorter than the mode's passes; at that
    // length the sends go back to back.
    every_->setMinimum(qrss_tx::minutes_for(mode_->currentData().toString().toStdString()));
    const bool repeats = times_->value() > 0;
    for (QWidget* w : {static_cast<QWidget*>(every_label_), static_cast<QWidget*>(every_),
                       static_cast<QWidget*>(every_unit_)}) {
        w->setEnabled(repeats);
    }
    update_summary();
}

void QrssScheduleWindow::update_summary() {
    if (summary_ == nullptr) return;
    const qs::Entry e =
        form_entry(mode_, waveform_, freq_, chosen_slot(), times_->value(), every_->value());
    if (waveform_note_ != nullptr) {
        waveform_note_->setText(waveform_note(e.waveform));
        waveform_note_->setVisible(!waveform_note_->text().isEmpty());
    }
    if (mode_info_ != nullptr) mode_info_->refresh();
    if (waveform_info_ != nullptr) waveform_info_->refresh();
    // Only "choose a Send" is said unasked. Whether the form's send can
    // go where it says (too soon, an overlap, an impossible repeat) is
    // said when Add is clicked: shown live, it appeared unprovoked
    // whenever a click on the timeline filled the form (Jeff, 2026-10-12).
    if (selected_send().empty()) {
        add_->setEnabled(false);
        summary_->setText(tr("Choose a Send first."));
        timeline_->clear_proposal();
        return;
    }
    add_->setEnabled(true);
    // On the timeline: a dashed outline for each send the form would
    // add, and the blinking bug at the first (Jeff, 2026-10-12).
    {
        const int passes = qrss_tx::passes_for(e.mode);
        std::vector<std::pair<double, double>> spans;
        const bool ok = qs::problem(e).empty();
        const int shown = !ok ? 1 : e.count == 0 ? 48 : std::min(e.count, 48);
        for (int k = 0; k < shown; ++k) {
            const double t = qs::send_slot(e, k);
            spans.emplace_back(t, t + passes * qrss_tx::PASS_SPACING_S);
        }
        timeline_->set_proposal(e.first_slot, std::move(spans));
        if (!ok) {
            summary_->clear();
            return;
        }
    }
    const int n = qrss_tx::passes_for(e.mode);
    // When is on the clocks and the timeline; this says what and how much.
    QString text = QString::fromStdString(qs::describe(e)) + tr(".");
    if (e.count > 0) {
        text += tr(" In all: %1 %2, %3 on the air.")
                    .arg(e.count * n)
                    .arg(pass_word(e.count * n))
                    .arg(duration_text(e.count * n * 30));
    }
    if (e.count != 1) {
        text += tr(" Every send has the same picture ID, so receivers add them up.");
    }
    summary_->setText(text.left(1).toUpper() + text.mid(1));
}

std::string QrssScheduleWindow::add() {
    const std::string id = selected_send();
    const QrssSend* s = sends_->find(id);
    if (s == nullptr) return "choose a Send first";
    images::Picture picture;
    try {
        picture = images::load(s->picture.string());
    } catch (const std::exception& ex) {
        const std::string why = std::string("the Send's picture would not load (") + ex.what() + ")";
        summary_->setText(tr("Can't add: %1.").arg(QString::fromStdString(why)));
        return why;
    }
    qs::Entry e =
        form_entry(mode_, waveform_, freq_, chosen_slot(), times_->value(), every_->value());
    e.label = s->label;
    e.send_id = s->id;
    // The scheduler checks it, and logs a refusal with its reason.
    const std::string why = scheduler_->add(e, picture);
    if (!why.empty()) {
        summary_->setText(tr("Can't add: %1.").arg(QString::fromStdString(why)));
        return why;
    }
    // The form moves on to the quarter hour after the send just added,
    // so it does not sit on a slot that is now taken.
    if (e.count > 0) {
        set_chosen_slot(qs::send_slot(e, e.count - 1) +
                        qrss_tx::passes_for(e.mode) * qrss_tx::PASS_SPACING_S);
    }
    summary_->setText(tr("Added \"%1\": %2.")
                          .arg(QString::fromStdString(s->label),
                               QString::fromStdString(qs::describe(e))));
    add_->setEnabled(false);
    return {};
}

std::string QrssScheduleWindow::drop_send(const std::string& send_id, double slot) {
    if (sends_->find(send_id) == nullptr) return "that Send is gone";
    // As if chosen in the list and clicked on the timeline, then Add:
    // mode, carrier and repeats are the form's, and every rule Add keeps
    // (no overlap, frequency changes kept clear) still holds.
    select_send(send_id);
    set_chosen_slot(slot);
    const std::string why = add();
    message_->setText(summary_->text());
    return why;
}

// --- what is scheduled ----------------------------------------------------

void QrssScheduleWindow::form_from(const qs::Entry& e) {
    // Adding its Send again repeats what was last chosen for it, not
    // whatever the form held: Jeff changed an entry to L on the timeline
    // and his re-adds went out as the form's 2 x CE (2026-10-12).
    mode_->setCurrentIndex(std::max(0, mode_->findData(QString::fromStdString(e.mode))));
    waveform_->setCurrentIndex(
        std::max(0, waveform_->findData(QString::fromStdString(e.waveform))));
    freq_->setValue(static_cast<int>(std::lround(e.freq_hz)));
    times_->setValue(e.count == 0 ? TIMES_FOREVER : std::min(TIMES_FOREVER - 1, e.count - 1));
    every_->setValue(e.every_min == 0 ? qrss_tx::minutes_for(e.mode) : e.every_min);
    sync_repeat();
}

void QrssScheduleWindow::select_entry(const std::string& entry) {
    selected_entry_ = entry;
    timeline_->set_selected_entry(entry);
    const qs::Entry* e = entry_by_id(scheduler_->entries(), entry);
    if (e != nullptr) form_from(*e);
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
        const bool active = scheduler_->on_air(e.id);
        // The send on the air is just before `next`; show it too.
        int k = std::max(0, e.next - (active ? std::max(1, qs::MAX_RUN_PASSES / n) : 0));
        double run_first = -1.0;  // the slot of the run's first block shown
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
            // An L send says so on its block; CE goes unsaid.
            b.mode = e.waveform == qrss_tx::WAVEFORM_L ? "L " + e.mode : e.mode;
            b.thumb = thumb;
            b.label = QString::fromStdString(e.label);
            b.paused = !e.enabled;
            b.on_air = active && k < e.next;
            // A run is started ahead of its quarter hour to get ready;
            // until its first pass keys it is not on the air, and says so.
            if (b.on_air) {
                if (run_first < 0.0) run_first = slot;
                const double first_keys = qrss_tx::audio_start(run_first);
                if (now < first_keys) {
                    b.on_air = false;
                    if (slot == run_first) b.keys_at = first_keys;
                }
            }
            blocks.push_back(std::move(b));
        }
    }
    timeline_->set_blocks(std::move(blocks));
    std::vector<QrssTimeline::Mark> marks;
    for (const qs::FreqChange& c : scheduler_->changes()) {
        marks.push_back({c.id, c.at, c.dial_hz, c.done});
    }
    timeline_->set_marks(std::move(marks));
}

// --- frequency changes ----------------------------------------------------

namespace {

const qs::FreqChange* change_by_id(const std::vector<qs::FreqChange>& changes,
                                   const std::string& id) {
    for (const qs::FreqChange& c : changes) {
        if (c.id == id) return &c;
    }
    return nullptr;
}

}  // namespace

std::string QrssScheduleWindow::add_freq_change(double at, double dial_hz) {
    qs::FreqChange c;
    c.at = at;
    c.dial_hz = dial_hz;
    const std::string why = scheduler_->add_change(c);
    show_result(why, tr("Frequency change to %1 at %2.")
                         .arg(QString::fromStdString(qs::mhz(dial_hz)), local_and_utc(at)));
    return why;
}

std::string QrssScheduleWindow::move_freq_change(const std::string& id, double at) {
    const std::string why =
        scheduler_->edit_change(id, [at](qs::FreqChange& c) { c.at = at; });
    show_result(why, tr("Frequency change moved to %1.").arg(local_and_utc(at)));
    return why;
}

std::string QrssScheduleWindow::retune_freq_change(const std::string& id, double dial_hz) {
    const std::string why =
        scheduler_->edit_change(id, [dial_hz](qs::FreqChange& c) { c.dial_hz = dial_hz; });
    show_result(why, tr("Frequency change now to %1.")
                         .arg(QString::fromStdString(qs::mhz(dial_hz))));
    return why;
}

void QrssScheduleWindow::remove_freq_change(const std::string& id) {
    scheduler_->remove_change(id);
    show_result({}, tr("Frequency change removed."));
}

QMenu* QrssScheduleWindow::change_menu_for(const std::string& id) {
    auto* menu = new QMenu(this);
    menu->setObjectName(QStringLiteral("freq_change_menu"));
    const qs::FreqChange* c = change_by_id(scheduler_->changes(), id);
    if (c == nullptr) {
        menu->addAction(tr("No longer scheduled"))->setEnabled(false);
        return menu;
    }
    menu->addSection(tr("Frequency change to %1, %2")
                         .arg(QString::fromStdString(qs::mhz(c->dial_hz)), local_and_utc(c->at)));
    const double at0 = c->at;
    const double hz0 = c->dial_hz;
    QAction* move = menu->addAction(tr("Reschedule..."));
    move->setObjectName(QStringLiteral("menu_reschedule_change"));
    connect(move, &QAction::triggered, this, [this, id, at0, hz0] {
        double at = at0;
        double hz = hz0;
        if (ask_freq_change(tr("Reschedule frequency change"), &at, &hz)) {
            const std::string why = scheduler_->edit_change(id, [at, hz](qs::FreqChange& x) {
                x.at = at;
                x.dial_hz = hz;
            });
            show_result(why, tr("Frequency change to %1 at %2.")
                                 .arg(QString::fromStdString(qs::mhz(hz)), local_and_utc(at)));
        }
    });
    QAction* retune = menu->addAction(tr("Change frequency..."));
    retune->setObjectName(QStringLiteral("menu_change_frequency"));
    connect(retune, &QAction::triggered, this, [this, id, at0, hz0] {
        double at = at0;
        double hz = hz0;
        if (ask_freq_change(tr("Change frequency"), &at, &hz, false)) retune_freq_change(id, hz);
    });
    QAction* remove = menu->addAction(tr("Remove"));
    remove->setObjectName(QStringLiteral("menu_remove_change"));
    connect(remove, &QAction::triggered, this, [this, id] { remove_freq_change(id); });
    if (c->done) {
        move->setEnabled(false);
        move->setText(tr("Reschedule... (already made)"));
        retune->setEnabled(false);
    }
    return menu;
}

QMenu* QrssScheduleWindow::empty_menu_for(double t) {
    auto* menu = new QMenu(this);
    menu->setObjectName(QStringLiteral("schedule_empty_menu"));
    const double at0 = std::round(t / 60.0) * 60.0;
    QAction* add = menu->addAction(tr("Add frequency change here..."));
    add->setObjectName(QStringLiteral("menu_add_change"));
    connect(add, &QAction::triggered, this, [this, at0] {
        double at = at0;
        double hz = defaults_().dial_hz;
        if (ask_freq_change(tr("Add frequency change"), &at, &hz)) add_freq_change(at, hz);
    });
    return menu;
}

bool QrssScheduleWindow::ask_freq_change(const QString& title, double* at, double* dial_hz,
                                         bool with_time) {
    QDialog dialog(this);
    dialog.setWindowTitle(title);
    auto* form = new QFormLayout(&dialog);
    auto* when = new QDateTimeEdit(&dialog);
    when->setObjectName(QStringLiteral("freq_change_time"));
    // setTimeZone arrived in Qt 6.7 and deprecates setTimeSpec there.
#if QT_VERSION >= QT_VERSION_CHECK(6, 7, 0)
    when->setTimeZone(QTimeZone::utc());
#else
    when->setTimeSpec(Qt::UTC);
#endif
    when->setDisplayFormat(QStringLiteral("yyyy-MM-dd HH:mm:ss 'UTC'"));
    when->setDateTime(utc_of(*at));
    if (with_time) {
        form->addRow(tr("When:"), when);
    } else {
        when->hide();
    }
    auto* freq = new QDoubleSpinBox(&dialog);
    freq->setObjectName(QStringLiteral("freq_change_mhz"));
    freq->setDecimals(6);
    freq->setRange(0.001, 10000.0);
    freq->setSingleStep(0.001);
    freq->setSuffix(tr(" MHz"));
    freq->setValue(*dial_hz > 0.0 ? *dial_hz / 1e6 : 10.141);
    form->addRow(tr("Dial frequency:"), freq);
    form->addRow(style::note(tr("Keep it at least 30 s clear of any send. With \"Perform "
                                "tuning after frequency change\" on (Settings > Rig control), "
                                "the tuner starts 5 s later."),
                             &dialog));
    auto* buttons = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel, &dialog);
    connect(buttons, &QDialogButtonBox::accepted, &dialog, &QDialog::accept);
    connect(buttons, &QDialogButtonBox::rejected, &dialog, &QDialog::reject);
    form->addRow(buttons);
    if (dialog.exec() != QDialog::Accepted) return false;
    *at = static_cast<double>(when->dateTime().toSecsSinceEpoch());
    *dial_hz = std::round(freq->value() * 1e6);
    return true;
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

    if (!qs::finished(*e)) {
        // What the scheduled list's Pause and Resume did.
        QAction* pause = menu->addAction(e->enabled ? tr("Pause") : tr("Resume"));
        pause->setObjectName(QStringLiteral("menu_pause"));
        const bool on = !e->enabled;
        connect(pause, &QAction::triggered, this, [this, id, on] {
            const std::string why = scheduler_->set_enabled(id, on);
            if (!why.empty()) show_result(why, {});
        });
    }
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

    QMenu* waveforms = menu->addMenu(tr("Change Waveform"));
    waveforms->setObjectName(QStringLiteral("menu_change_waveform"));
    for (const char* w : {qrss_tx::WAVEFORM_CE, qrss_tx::WAVEFORM_L}) {
        const std::string waveform(w);
        QAction* a = waveforms->addAction(waveform == qrss_tx::WAVEFORM_CE
                                              ? tr("CE - constant envelope")
                                              : tr("L - linear (ALC out)"));
        a->setCheckable(true);
        a->setChecked(waveform == e->waveform);
        connect(a, &QAction::triggered, this,
                [this, id, waveform] { change_waveform(id, waveform); });
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
    // The form takes the entry as it now is, for the next add.
    if (const qs::Entry* e = entry_by_id(scheduler_->entries(), entry); why.empty() && e) {
        form_from(*e);
    }
    show_result(why, tr("Changed to mode %1.").arg(QString::fromStdString(mode)));
    return why;
}

std::string QrssScheduleWindow::change_offset(const std::string& entry, double hz) {
    // Every send of the entry moves; the picture and its ID stay the same.
    std::string why = scheduler_->change(entry, [hz](qs::Entry& e) { e.freq_hz = hz; });
    // The form takes the entry as it now is, for the next add.
    if (const qs::Entry* e = entry_by_id(scheduler_->entries(), entry); why.empty() && e) {
        form_from(*e);
    }
    show_result(why, tr("Moved to %1 Hz.").arg(hz, 0, 'f', 0));
    return why;
}

std::string QrssScheduleWindow::change_waveform(const std::string& entry,
                                                const std::string& waveform) {
    // The picture and its ID stay the same: receivers add passes of both.
    std::string why =
        scheduler_->change(entry, [&waveform](qs::Entry& e) { e.waveform = waveform; });
    // The form takes the entry as it now is, for the next add.
    if (const qs::Entry* e = entry_by_id(scheduler_->entries(), entry); why.empty() && e) {
        form_from(*e);
    }
    show_result(why, tr("Now sent as waveform %1.").arg(QString::fromStdString(waveform)));
    return why;
}

std::string QrssScheduleWindow::repeat_every(const std::string& entry, int minutes) {
    std::string why = scheduler_->change(entry, [minutes](qs::Entry& e) {
        e.every_min = minutes;
        // A single send repeats once more; a series keeps its count.
        if (e.count == 1) e.count = 2;
    });
    // The form takes the entry as it now is, for the next add.
    if (const qs::Entry* e = entry_by_id(scheduler_->entries(), entry); why.empty() && e) {
        form_from(*e);
    }
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

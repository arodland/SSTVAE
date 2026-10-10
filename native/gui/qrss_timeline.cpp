#include "qrss_timeline.hpp"

#include <QApplication>
#include <QContextMenuEvent>
#include <QDateTime>
#include <QDragEnterEvent>
#include <QDropEvent>
#include <QMimeData>
#include <QMouseEvent>
#include <QPainter>
#include <QTimeZone>

#include <algorithm>
#include <cmath>
#include <utility>

#include "style.hpp"

namespace sstvae::gui {

namespace {

constexpr int CAPTION_H = 18;
constexpr int PAD = 4;

QDateTime utc_of(double t) {
    return QDateTime::fromSecsSinceEpoch(static_cast<qint64>(std::llround(t)), QTimeZone::utc());
}

}  // namespace

QrssTimeline::QrssTimeline(QWidget* parent) : QWidget(parent) {
    setAcceptDrops(true);
    setObjectName(QStringLiteral("schedule_timeline"));
    setMouseTracking(true);
    setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);
}

void QrssTimeline::set_view(double start, int cells, double now) {
    start_ = start;
    cells_ = std::max(1, cells);
    now_ = now;
    updateGeometry();
    resize(sizeHint());
    update();
}

void QrssTimeline::set_blocks(std::vector<Block> blocks) {
    blocks_ = std::move(blocks);
    pressed_ = -1;
    dragging_ = false;
    update();
}

QColor QrssTimeline::mark_color() { return QColor(0x1f, 0xa8, 0x4a); }

void QrssTimeline::set_marks(std::vector<Mark> marks) {
    marks_ = std::move(marks);
    update();
}

int QrssTimeline::mark_at(const QPoint& p) const {
    int best = -1;
    int best_d = MARK_GRAB_PX + 1;
    for (std::size_t i = 0; i < marks_.size(); ++i) {
        const int d = std::abs(p.x() - x_of(marks_[i].at));
        if (d < best_d) {
            best = static_cast<int>(i);
            best_d = d;
        }
    }
    return best;
}

double QrssTimeline::time_at(int x) const {
    return start_ + static_cast<double>(x) / CELL_W * CELL_S;
}

void QrssTimeline::set_selected_entry(const std::string& entry) {
    selected_entry_ = entry;
    update();
}

void QrssTimeline::set_highlighted_send(const std::string& send_id) {
    highlighted_send_ = send_id;
    update();
}

QSize QrssTimeline::sizeHint() const {
    return {cells_ * CELL_W + 1, HEADER_H + 2 * PAD + THUMB_H + CAPTION_H + 2};
}

QSize QrssTimeline::minimumSizeHint() const { return sizeHint(); }

int QrssTimeline::x_of(double t) const {
    return static_cast<int>(std::lround((t - start_) / CELL_S * CELL_W));
}

double QrssTimeline::slot_at(int x) const {
    return start_ + std::floor(static_cast<double>(x) / CELL_W) * CELL_S;
}

QRect QrssTimeline::block_rect(const Block& b) const {
    const int x0 = x_of(b.slot) + 1;
    const int x1 = x_of(b.end) - 1;
    return {x0, HEADER_H + 1, std::max(4, x1 - x0), height() - HEADER_H - 2};
}

int QrssTimeline::block_at(const QPoint& p) const {
    for (int i = static_cast<int>(blocks_.size()) - 1; i >= 0; --i) {
        if (block_rect(blocks_[i]).contains(p)) return i;
    }
    return -1;
}

QString QrssTimeline::block_caption(const Block& b) const {
    const QString mode = QString::fromStdString(b.mode);
    if (b.count == 1) return mode;
    if (b.count == 0) return tr("%1 #%2").arg(mode).arg(b.send + 1);
    return tr("%1 %2/%3").arg(mode).arg(b.send + 1).arg(b.count);
}

void QrssTimeline::paintEvent(QPaintEvent*) {
    QPainter p(this);
    const QPalette& pal = palette();
    p.fillRect(rect(), pal.color(QPalette::Base));
    const QColor grid = pal.color(QPalette::Mid);
    const QColor faint(grid.red(), grid.green(), grid.blue(), 70);
    // A light grey line down every quarter hour, one per slot (Jeff,
    // 2026-10-10); the hour lines stay darker.
    QColor slot_line = grid;
    slot_line.setAlpha(110);
    const QColor text = pal.color(QPalette::Text);
    const QColor muted = style::secondary_text(pal);

    // Columns, an hour label every four, the day at midnight.
    QFont small = font();
    small.setPointSizeF(std::max(6.0, small.pointSizeF() * 0.85));
    p.setFont(small);
    for (int c = 0; c <= cells_; ++c) {
        const double t = start_ + c * CELL_S;
        const int x = c * CELL_W;
        const QDateTime u = utc_of(t);
        const bool hour = u.time().minute() == 0;
        const bool half = u.time().minute() == 30;
        p.setPen(hour ? grid : slot_line);
        p.drawLine(x, hour ? 0 : HEADER_H - 4, x, height());
        if (half) {
            p.setPen(faint);
            p.drawLine(x, HEADER_H - 6, x, HEADER_H);
        }
        if (hour && c < cells_) {
            p.setPen(u.time().hour() == 0 ? text : muted);
            const QString label = u.time().hour() == 0
                                      ? u.toString(QStringLiteral("ddd d"))
                                      : u.toString(QStringLiteral("HH:mm'Z'"));
            p.drawText(QRect(x + 3, 0, 4 * CELL_W - 4, HEADER_H - 2),
                       Qt::AlignLeft | Qt::AlignVCenter, label);
        }
    }
    p.setPen(grid);
    p.drawLine(0, HEADER_H, width(), HEADER_H);

    // Where a send added from the selected Send would go.
    if ((slot_hint_ || dropping_) && hover_x_ >= 0 &&
        block_at(QPoint(hover_x_, HEADER_H + 4)) < 0) {
        const int x = x_of(slot_at(hover_x_));
        QColor h = pal.color(QPalette::Highlight);
        h.setAlpha(40);
        p.fillRect(QRect(x + 1, HEADER_H + 1, 2 * CELL_W - 2, height() - HEADER_H - 2), h);
    }

    p.setFont(font());
    for (std::size_t i = 0; i < blocks_.size(); ++i) {
        const Block& b = blocks_[i];
        const QRect r = block_rect(b);
        const bool selected = !selected_entry_.empty() && b.entry == selected_entry_;
        const bool tinted = !highlighted_send_.empty() && b.send_id == highlighted_send_;
        QColor fill = pal.color(QPalette::Button);
        if (tinted) {
            fill = pal.color(QPalette::Highlight);
            fill.setAlpha(90);
        }
        p.fillRect(r, fill);
        const QRect thumb_box(r.left() + PAD, r.top() + PAD, r.width() - 2 * PAD, THUMB_H);
        if (!b.thumb.isNull()) {
            const QPixmap scaled = b.thumb.scaled(thumb_box.size(), Qt::KeepAspectRatio,
                                                  Qt::SmoothTransformation);
            const QPoint at(thumb_box.left() + (thumb_box.width() - scaled.width()) / 2,
                            thumb_box.top() + (thumb_box.height() - scaled.height()) / 2);
            p.drawPixmap(at, scaled);
        }
        const QRect caption(r.left() + 2, thumb_box.bottom() + 2, r.width() - 4, CAPTION_H);
        p.setPen(text);
        p.drawText(caption, Qt::AlignCenter, block_caption(b));
        if (b.paused) {
            QColor veil = pal.color(QPalette::Base);
            veil.setAlpha(160);
            p.fillRect(r, veil);
            p.setPen(muted);
            p.drawText(thumb_box, Qt::AlignCenter, tr("paused"));
        }
        QPen edge(b.on_air ? style::color::danger() : selected ? pal.color(QPalette::Highlight) : grid);
        edge.setWidth(b.on_air || selected ? 3 : 1);
        p.setPen(edge);
        p.setBrush(Qt::NoBrush);
        p.drawRect(r.adjusted(0, 0, -1, -1));
    }

    // The block being dragged, where it would land.
    if (dragging_ && pressed_ >= 0 && pressed_ < static_cast<int>(blocks_.size())) {
        Block ghost = blocks_[pressed_];
        const double len = ghost.end - ghost.slot;
        ghost.slot = drag_slot_;
        ghost.end = drag_slot_ + len;
        QPen dash(pal.color(QPalette::Highlight));
        dash.setStyle(Qt::DashLine);
        dash.setWidth(2);
        p.setPen(dash);
        p.drawRect(block_rect(ghost).adjusted(0, 0, -1, -1));
    }

    // Frequency changes: a line as wide as "now", and the new dial
    // frequency in the header beside it.
    for (const Mark& m : marks_) {
        const int x = x_of(m.at);
        if (x < -40 || x > width() + 40) continue;
        QColor c = mark_color();
        if (m.done) c.setAlpha(110);
        p.setPen(QPen(c, 2));
        p.drawLine(x, HEADER_H - 6, x, height());
        p.setFont(small);
        const QString label = QString::number(m.dial_hz / 1e6, 'f', 3);
        const int w = p.fontMetrics().horizontalAdvance(label) + 4;
        const QRect box(x + 3, HEADER_H - 1 - p.fontMetrics().height(), w,
                        p.fontMetrics().height());
        QColor ground = pal.color(QPalette::Base);
        ground.setAlpha(220);
        p.fillRect(box, ground);
        p.setPen(c);
        p.drawText(box, Qt::AlignCenter, label);
    }

    // Now.
    const int xn = x_of(now_);
    if (xn >= 0 && xn <= width()) {
        p.setPen(QPen(style::color::danger(), 2));
        p.drawLine(xn, 0, xn, height());
    }
}

void QrssTimeline::mousePressEvent(QMouseEvent* event) {
    if (event->button() != Qt::LeftButton) return;
    pressed_ = block_at(event->position().toPoint());
    press_pos_ = event->position().toPoint();
    dragging_ = false;
}

void QrssTimeline::mouseMoveEvent(QMouseEvent* event) {
    const QPoint pos = event->position().toPoint();
    hover_x_ = pos.x();
    if (pressed_ >= 0 && (event->buttons() & Qt::LeftButton)) {
        if (!dragging_ &&
            (pos - press_pos_).manhattanLength() >= QApplication::startDragDistance()) {
            dragging_ = true;
        }
        if (dragging_) {
            const Block& b = blocks_[pressed_];
            // Keep the grab point under the pointer, snapped to a column.
            const int offset = press_pos_.x() - x_of(b.slot);
            drag_slot_ = slot_at(pos.x() - offset + CELL_W / 2);
            setCursor(Qt::ClosedHandCursor);
        }
    } else {
        setCursor(block_at(pos) >= 0 ? Qt::OpenHandCursor : Qt::ArrowCursor);
    }
    update();
}

void QrssTimeline::mouseReleaseEvent(QMouseEvent* event) {
    if (event->button() != Qt::LeftButton) return;
    const int i = pressed_;
    const bool was_drag = dragging_;
    pressed_ = -1;
    dragging_ = false;
    unsetCursor();
    update();
    if (i >= 0 && i < static_cast<int>(blocks_.size())) {
        const Block b = blocks_[i];
        if (was_drag) {
            if (drag_slot_ != b.slot) {
                emit moveRequested(QString::fromStdString(b.entry), b.send, drag_slot_);
            }
        } else {
            emit blockClicked(QString::fromStdString(b.entry), b.send);
        }
        return;
    }
    if (event->position().y() > HEADER_H) emit slotClicked(slot_at(event->position().toPoint().x()));
}

void QrssTimeline::contextMenuEvent(QContextMenuEvent* event) {
    // A mark first: it is narrow and sits at a block's edge.
    if (const int m = mark_at(event->pos()); m >= 0) {
        emit markMenuRequested(QString::fromStdString(marks_[m].id), event->globalPos());
        return;
    }
    const int i = block_at(event->pos());
    if (i < 0) {
        if (event->pos().y() > HEADER_H) emit emptyMenuRequested(time_at(event->pos().x()), event->globalPos());
        return;
    }
    const Block& b = blocks_[i];
    emit blockClicked(QString::fromStdString(b.entry), b.send);
    emit menuRequested(QString::fromStdString(b.entry), b.send, event->globalPos());
}

void QrssTimeline::dragEnterEvent(QDragEnterEvent* event) {
    if (!event->mimeData()->hasFormat(QString::fromLatin1(SEND_MIME))) return;
    dropping_ = true;
    hover_x_ = event->position().toPoint().x();
    event->acceptProposedAction();
    update();
}

void QrssTimeline::dragMoveEvent(QDragMoveEvent* event) {
    if (!event->mimeData()->hasFormat(QString::fromLatin1(SEND_MIME))) return;
    hover_x_ = event->position().toPoint().x();
    event->acceptProposedAction();
    update();
}

void QrssTimeline::dragLeaveEvent(QDragLeaveEvent* event) {
    dropping_ = false;
    hover_x_ = -1;
    update();
    QWidget::dragLeaveEvent(event);
}

void QrssTimeline::dropEvent(QDropEvent* event) {
    dropping_ = false;
    hover_x_ = -1;
    update();
    const QByteArray id = event->mimeData()->data(QString::fromLatin1(SEND_MIME));
    if (id.isEmpty()) return;
    event->acceptProposedAction();
    // The quarter hour under the pointer, as a click there would choose.
    emit sendDropped(QString::fromUtf8(id), slot_at(event->position().toPoint().x()));
}

void QrssTimeline::leaveEvent(QEvent* event) {
    hover_x_ = -1;
    update();
    QWidget::leaveEvent(event);
}

}  // namespace sstvae::gui

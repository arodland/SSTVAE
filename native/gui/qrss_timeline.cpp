#include "qrss_timeline.hpp"

#include <QApplication>
#include <QContextMenuEvent>
#include <QDateTime>
#include <QDragEnterEvent>
#include <QDropEvent>
#include <QMimeData>
#include <QMouseEvent>
#include <QPainter>
#include <QPainterPath>
#include <QTimer>
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
    blink_ = new QTimer(this);
    blink_->setInterval(530);
    connect(blink_, &QTimer::timeout, this, [this] {
        blink_on_ = !blink_on_;
        // The whole of what is in view: the arrow at its edge blinks too.
        update(visibleRegion().boundingRect());
    });
}

void QrssTimeline::set_proposal(double first, std::vector<std::pair<double, double>> spans) {
    if (proposal_ && first == proposal_at_ && spans == spans_) return;
    proposal_ = true;
    proposal_at_ = first;
    spans_ = std::move(spans);
    blink_on_ = true;
    if (!blink_->isActive()) blink_->start();
    update();
}

int QrssTimeline::proposal_off_view() const {
    if (!proposal_) return 0;
    const QRect view = visibleRegion().boundingRect();
    if (view.isEmpty()) return 0;
    int lo = x_of(proposal_at_);
    int hi = lo;
    for (const auto& [from, to] : spans_) {
        lo = std::min(lo, x_of(from));
        hi = std::max(hi, x_of(to));
    }
    if (hi < view.left()) return -1;
    if (lo > view.right()) return 1;
    return 0;
}

void QrssTimeline::clear_proposal() {
    if (!proposal_) return;
    proposal_ = false;
    spans_.clear();
    blink_->stop();
    update();
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
    return {cells_ * CELL_W + 1, HEADER_H + 2 * PAD + THUMB_H + CAPTION_H + 2 + BUG_H};
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
    return {x0, HEADER_H + 1, std::max(4, x1 - x0), rows_bottom() - HEADER_H - 2};
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
    // The grid is the text colour mixed into the background, not the
    // style's Mid and Dark: some themes make those barely differ from
    // Base, and the quarter-hour lines vanished on Hamlet (Jeff,
    // 2026-10-12). So it has contrast in a light theme and a dark one.
    const QColor base = pal.color(QPalette::Base);
    const QColor ink = pal.color(QPalette::Text);
    const auto mix = [&base, &ink](double a) {
        return QColor::fromRgbF(base.redF() + (ink.redF() - base.redF()) * a,
                                base.greenF() + (ink.greenF() - base.greenF()) * a,
                                base.blueF() + (ink.blueF() - base.blueF()) * a);
    };
    const QColor grid = mix(0.35);
    const QColor faint = mix(0.25);
    // A line down every quarter hour, one per slot (Jeff, 2026-10-10),
    // solid since "rather subtle" (2026-10-12); the hour lines are twice
    // as wide and darker.
    const QColor slot_line = grid;
    const QPen hour_line(mix(0.6), 2);
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
        if (hour) {
            p.setPen(hour_line);
        } else {
            p.setPen(slot_line);
        }
        p.drawLine(x, hour ? 0 : HEADER_H - 4, x, rows_bottom());
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
    p.drawLine(0, rows_bottom(), width(), rows_bottom());

    // Where a send added from the selected Send would go.
    if ((slot_hint_ || dropping_) && hover_x_ >= 0 &&
        block_at(QPoint(hover_x_, HEADER_H + 4)) < 0) {
        const int x = x_of(slot_at(hover_x_));
        QColor h = pal.color(QPalette::Highlight);
        h.setAlpha(40);
        p.fillRect(QRect(x + 1, HEADER_H + 1, 2 * CELL_W - 2, rows_bottom() - HEADER_H - 2), h);
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
        if (b.keys_at > 0.0) {
            // Getting ready: say when it keys, on a veil like "paused".
            QColor veil = pal.color(QPalette::Base);
            veil.setAlpha(170);
            p.fillRect(thumb_box, veil);
            p.setPen(text);
            p.drawText(thumb_box, Qt::AlignCenter | Qt::TextWordWrap,
                       tr("getting ready\nkeys %1")
                           .arg(utc_of(b.keys_at).toString(QStringLiteral("HH:mm:ss'Z'"))));
        }
        QPen edge(b.on_air ? style::color::danger() : selected ? pal.color(QPalette::Highlight) : grid);
        if (b.keys_at > 0.0) {
            edge = QPen(style::color::danger());
            edge.setStyle(Qt::DashLine);
        }
        edge.setWidth(b.on_air || selected || b.keys_at > 0.0 ? 3 : 1);
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
        p.drawLine(x, HEADER_H - 6, x, rows_bottom());
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

    // The change being dragged, where it would land, and when.
    if (dragging_ && pressed_mark_ >= 0 && pressed_mark_ < static_cast<int>(marks_.size())) {
        const int x = x_of(drag_at_);
        QPen dash(mark_color(), 2);
        dash.setStyle(Qt::DashLine);
        p.setPen(dash);
        p.drawLine(x, HEADER_H - 6, x, rows_bottom());
        p.setFont(small);
        const QString label = utc_of(drag_at_).toString(QStringLiteral("HH:mm'Z'"));
        const int w = p.fontMetrics().horizontalAdvance(label) + 4;
        const QRect box(x + 3, HEADER_H + 2, w, p.fontMetrics().height());
        QColor ground = pal.color(QPalette::Base);
        ground.setAlpha(230);
        p.fillRect(box, ground);
        p.setPen(mark_color());
        p.drawText(box, Qt::AlignCenter, label);
    }

    // The send the form would add: dashed outlines, and the bug.
    if (proposal_) {
        const QColor red = style::color::danger();
        QPen dash(red, 2);
        dash.setStyle(Qt::DashLine);
        p.setPen(dash);
        p.setBrush(Qt::NoBrush);
        for (const auto& [from, to] : spans_) {
            Block b;
            b.slot = from;
            b.end = to;
            p.drawRect(block_rect(b).adjusted(1, 1, -1, -1));
        }
        const int x = x_of(proposal_at_);
        if (blink_on_) {
            p.setRenderHint(QPainter::Antialiasing);
            p.setPen(QPen(red, 2));
            const int y = rows_bottom();
            p.drawLine(x, y - 2, x, y + 3);
            QPainterPath tri;
            tri.moveTo(x, y + 3);
            tri.lineTo(x - 6, y + 12);
            tri.lineTo(x + 6, y + 12);
            tri.closeSubpath();
            p.fillPath(tri, red);
            p.setRenderHint(QPainter::Antialiasing, false);
        }
        QFont small = font();
        small.setPointSizeF(std::max(6.0, small.pointSizeF() * 0.85));
        p.setFont(small);
        p.setPen(red);
        p.drawText(QRect(x + 9, rows_bottom() + 1, 4 * CELL_W, BUG_H - 2),
                   Qt::AlignLeft | Qt::AlignVCenter,
                   utc_of(proposal_at_).toString(QStringLiteral("HH:mm'Z'")));
    }

    // Now.
    const int xn = x_of(now_);
    if (xn >= 0 && xn <= width()) {
        p.setPen(QPen(style::color::danger(), 2));
        p.drawLine(xn, 0, xn, rows_bottom());
    }

    // The proposal scrolled out of view: an arrow at that edge of the
    // view, pointing to it, blinking with the bug.
    if (const int side = proposal_off_view(); side != 0 && blink_on_) {
        const QRect view = visibleRegion().boundingRect();
        const int cy = (HEADER_H + rows_bottom()) / 2;
        const int tip = side < 0 ? view.left() + 5 : view.right() - 5;
        const int base = tip - side * 16;
        QPainterPath arrow;
        arrow.moveTo(tip, cy);
        arrow.lineTo(base, cy - 11);
        arrow.lineTo(base, cy + 11);
        arrow.closeSubpath();
        p.setRenderHint(QPainter::Antialiasing);
        p.setPen(QPen(pal.color(QPalette::Base), 3));
        p.setBrush(Qt::NoBrush);
        p.drawPath(arrow);
        p.fillPath(arrow, style::color::danger());
        p.setRenderHint(QPainter::Antialiasing, false);
    }
}

void QrssTimeline::mousePressEvent(QMouseEvent* event) {
    if (event->button() != Qt::LeftButton) return;
    press_pos_ = event->position().toPoint();
    dragging_ = false;
    // A change first, as for the right-click: it is narrow and sits at a
    // block's edge. One already made stays where it was made.
    pressed_mark_ = mark_at(press_pos_);
    if (pressed_mark_ >= 0 && marks_[pressed_mark_].done) pressed_mark_ = -1;
    pressed_ = pressed_mark_ >= 0 ? -1 : block_at(press_pos_);
}

void QrssTimeline::mouseMoveEvent(QMouseEvent* event) {
    const QPoint pos = event->position().toPoint();
    hover_x_ = pos.x();
    if (pressed_mark_ >= 0 && (event->buttons() & Qt::LeftButton)) {
        if (!dragging_ &&
            (pos - press_pos_).manhattanLength() >= QApplication::startDragDistance()) {
            dragging_ = true;
        }
        if (dragging_) {
            const double moved = time_at(pos.x()) - time_at(press_pos_.x());
            drag_at_ = std::round((marks_[pressed_mark_].at + moved) / 60.0) * 60.0;
            setCursor(Qt::SizeHorCursor);
        }
    } else if (pressed_ >= 0 && (event->buttons() & Qt::LeftButton)) {
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
        const int m = mark_at(pos);
        setCursor(m >= 0 && !marks_[m].done ? Qt::SizeHorCursor
                  : block_at(pos) >= 0      ? Qt::OpenHandCursor
                                            : Qt::ArrowCursor);
    }
    update();
}

void QrssTimeline::mouseReleaseEvent(QMouseEvent* event) {
    if (event->button() != Qt::LeftButton) return;
    const int i = pressed_;
    const int m = pressed_mark_;
    const bool was_drag = dragging_;
    pressed_ = -1;
    pressed_mark_ = -1;
    dragging_ = false;
    unsetCursor();
    update();
    if (m >= 0 && m < static_cast<int>(marks_.size())) {
        if (was_drag && drag_at_ != marks_[m].at) {
            emit markMoveRequested(QString::fromStdString(marks_[m].id), drag_at_);
        }
        return;
    }
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

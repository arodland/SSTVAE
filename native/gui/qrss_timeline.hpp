// The QRSS schedule as a horizontal timeline: one column per quarter
// hour, the slots a QRSS send can start on, and one block per scheduled
// send spanning the half hours its passes are on the air. Each block
// shows the picture, scaled down, with its mode and which repeat it is
// underneath ("A 2/5").
//
// A scheduled frequency change is a green line at its moment, as wide as
// the red "now" line, with the new dial frequency in the header; a
// right-click within MARK_GRAB_PX of it opens its menu, and a drag moves
// it (to the minute) unless it has already been made.
//
// The timeline only draws and reports: a click on a block or an empty
// slot, a right-click on a block, a block dragged to another slot. The
// schedule window decides what each means, so the widget holds no
// schedule of its own and a test can drive it with made-up blocks.

#ifndef SSTVAE_GUI_QRSS_TIMELINE_HPP
#define SSTVAE_GUI_QRSS_TIMELINE_HPP

#include <QColor>
#include <QPixmap>
#include <QPoint>
#include <QRect>
#include <QString>
#include <QWidget>

#include <string>
#include <vector>

namespace sstvae::gui {

class QrssTimeline : public QWidget {
    Q_OBJECT

public:
    static constexpr double CELL_S = 900.0;   // one column: a quarter hour
    static constexpr int CELL_W = 40;
    static constexpr int HEADER_H = 20;
    static constexpr int THUMB_H = 54;

    struct Block {
        std::string entry;      // the schedule entry's id
        std::string send_id;    // the Send it was made from, if any
        int send = 0;           // which send of the entry, from 0
        int count = 1;          // the entry's sends; 0 = until removed
        double slot = 0.0;      // its first pass's quarter hour
        double end = 0.0;       // the end of its last half hour
        std::string mode;
        QPixmap thumb;
        QString label;
        bool paused = false;
        bool on_air = false;
    };

    // A frequency change.
    struct Mark {
        std::string id;
        double at = 0.0;
        double dial_hz = 0.0;
        bool done = false;
    };
    // How far either side of a mark's line a right-click still finds it.
    static constexpr int MARK_GRAB_PX = 6;
    static QColor mark_color();
    // The drag-and-drop type of a Send dragged from the Sends list: its
    // id, as UTF-8.
    static constexpr const char* SEND_MIME = "application/x-sstvae-send";

    explicit QrssTimeline(QWidget* parent = nullptr);

    // `start` is the first column's quarter hour; `cells` columns.
    void set_view(double start, int cells, double now);
    void set_blocks(std::vector<Block> blocks);
    const std::vector<Block>& blocks() const { return blocks_; }
    void set_marks(std::vector<Mark> marks);
    const std::vector<Mark>& marks() const { return marks_; }
    // -1 when no mark is there.
    int mark_at(const QPoint& p) const;
    // The moment under x (not snapped).
    double time_at(int x) const;
    // Outlines the blocks of one entry, and tints those of one Send.
    void set_selected_entry(const std::string& entry);
    void set_highlighted_send(const std::string& send_id);
    // The slot a column under the pointer would start a send on.
    void set_hover_slot_hint(bool on) { slot_hint_ = on; }

    double start() const { return start_; }
    int x_of(double t) const;
    double slot_at(int x) const;
    // -1 when no block is there.
    int block_at(const QPoint& p) const;
    QRect block_rect(const Block& b) const;
    QString block_caption(const Block& b) const;

    QSize sizeHint() const override;
    QSize minimumSizeHint() const override;

signals:
    void blockClicked(const QString& entry, int send);
    void slotClicked(double slot);
    void menuRequested(const QString& entry, int send, const QPoint& global_pos);
    void moveRequested(const QString& entry, int send, double slot);
    // A frequency change dragged to `at` (whole minutes).
    void markMoveRequested(const QString& id, double at);
    // A right-click on a frequency change, and on empty time.
    void markMenuRequested(const QString& id, const QPoint& global_pos);
    void emptyMenuRequested(double t, const QPoint& global_pos);
    // A Send dropped on the quarter hour `slot`.
    void sendDropped(const QString& send_id, double slot);

protected:
    void paintEvent(QPaintEvent* event) override;
    void mousePressEvent(QMouseEvent* event) override;
    void mouseMoveEvent(QMouseEvent* event) override;
    void mouseReleaseEvent(QMouseEvent* event) override;
    void contextMenuEvent(QContextMenuEvent* event) override;
    void leaveEvent(QEvent* event) override;
    void dragEnterEvent(QDragEnterEvent* event) override;
    void dragMoveEvent(QDragMoveEvent* event) override;
    void dragLeaveEvent(QDragLeaveEvent* event) override;
    void dropEvent(QDropEvent* event) override;

private:
    double start_ = 0.0;
    double now_ = 0.0;
    int cells_ = 96;
    std::vector<Block> blocks_;
    std::vector<Mark> marks_;
    std::string selected_entry_;
    std::string highlighted_send_;
    bool slot_hint_ = false;
    bool dropping_ = false;   // a Send is being dragged over
    int hover_x_ = -1;

    int pressed_ = -1;
    QPoint press_pos_;
    bool dragging_ = false;
    double drag_slot_ = 0.0;
    int pressed_mark_ = -1;
    double drag_at_ = 0.0;     // where a dragged mark would land
};

}  // namespace sstvae::gui

#endif

#include "tx_panel.hpp"

#include <QColor>
#include <QComboBox>
#include <QCoreApplication>
#include <QDateTime>
#include <QDialog>
#include <QDialogButtonBox>
#include <QDragEnterEvent>
#include <QDropEvent>
#include <QFileDialog>
#include <QFont>
#include <QFileInfo>
#include <QFormLayout>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QIcon>
#include <QInputDialog>
#include <QLabel>
#include <QLineEdit>
#include <QMessageBox>
#include <QMimeData>
#include <QPainter>
#include <QPainterPath>
#include <QPixmap>
#include <QPlainTextEdit>
#include <QPolygonF>
#include <QTextDocument>
#include <QProgressBar>
#include <QPushButton>
#include <QResizeEvent>
#include <QSlider>
#include <QSplitter>
#include <QStyle>
#include <QTimer>
#include <QToolButton>
#include <QUrl>
#include <QVBoxLayout>

#include <algorithm>
#include <cmath>
#include <exception>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <system_error>
#include <vector>

#include "app_state.hpp"
#include "audio/qt/qtaudio.hpp"
#include "banner.hpp"
#include "checkpoint/checkpoint.hpp"
#include "codec/codec.hpp"
#include "codec/grad_session.hpp"
#include "config.hpp"
#include "crop_dialog.hpp"
#include "images/images.hpp"
#include "flow_layout.hpp"
#include "item_menu.hpp"
#include "overlay/template.hpp"
#include "overlay/template_catalog.hpp"
#include "overlay_editor.hpp"
#include "share_dialog.hpp"
#include "style.hpp"
#include "settings/settings.hpp"

namespace sstvae::gui {

namespace {

const char* IMAGE_FILTER =
    "Images (*.png *.jpg *.jpeg *.webp *.bmp *.gif);;All files (*)";

// --- the tool palette's icons -------------------------------------------
//
// Hand-drawn rather than shipped as asset files: four tiny glyphs are
// cheaper to draw once than to source, license and keep in step with
// the platform's icon theme (which several of this project's target
// platforms do not reliably have one of). `PM_LargeIconSize` and
// `devicePixelRatioF`, so a HiDPI screen gets a crisp glyph.

QIcon draw_tool_icon(const QWidget* metrics,
                     const std::function<void(QPainter&, int, const QColor&)>& paint) {
    const int size = metrics->style()->pixelMetric(QStyle::PM_LargeIconSize);
    const qreal dpr = metrics->devicePixelRatioF();
    QPixmap pixmap(static_cast<int>(std::lround(size * dpr)),
                   static_cast<int>(std::lround(size * dpr)));
    pixmap.setDevicePixelRatio(dpr);
    pixmap.fill(Qt::transparent);
    QPainter painter(&pixmap);
    painter.setRenderHint(QPainter::Antialiasing, true);
    // Palette, not a literal color: a hand-drawn icon that ignored the
    // palette would go invisible on a dark theme.
    paint(painter, size, metrics->palette().color(QPalette::WindowText));
    return QIcon(pixmap);
}

// The "picture" glyph shared by the image-inset and last-received tools
// -- a frame, a sun and a mountain range, the generic photo icon every
// platform already uses this shape for.
void draw_picture_glyph(QPainter& p, int size, const QColor& ink) {
    const QRectF frame(size * 0.12, size * 0.12, size * 0.76, size * 0.76);
    p.setPen(QPen(ink, std::max(1.0, size / 16.0)));
    p.setBrush(Qt::NoBrush);
    p.drawRoundedRect(frame, size * 0.06, size * 0.06);

    p.setPen(Qt::NoPen);
    p.setBrush(ink);
    p.drawEllipse(QPointF(frame.left() + frame.width() * 0.3,
                          frame.top() + frame.height() * 0.32),
                 size * 0.07, size * 0.07);

    QPainterPath mountains;
    mountains.moveTo(frame.left() + frame.width() * 0.06, frame.bottom() - frame.height() * 0.1);
    mountains.lineTo(frame.left() + frame.width() * 0.38, frame.bottom() - frame.height() * 0.55);
    mountains.lineTo(frame.left() + frame.width() * 0.58, frame.bottom() - frame.height() * 0.3);
    mountains.lineTo(frame.left() + frame.width() * 0.78, frame.bottom() - frame.height() * 0.52);
    mountains.lineTo(frame.right() - frame.width() * 0.06, frame.bottom() - frame.height() * 0.1);
    mountains.closeSubpath();
    p.drawPath(mountains);
}

QIcon text_tool_icon(const QWidget* metrics) {
    return draw_tool_icon(metrics, [](QPainter& p, int size, const QColor& ink) {
        QFont font = p.font();
        font.setBold(true);
        font.setPixelSize(static_cast<int>(size * 0.72));
        p.setFont(font);
        p.setPen(ink);
        p.drawText(QRect(0, 0, size, size), Qt::AlignCenter, QStringLiteral("T"));
    });
}

QIcon image_tool_icon(const QWidget* metrics) {
    return draw_tool_icon(metrics, [](QPainter& p, int size, const QColor& ink) {
        draw_picture_glyph(p, size, ink);
    });
}

QIcon last_rx_tool_icon(const QWidget* metrics) {
    return draw_tool_icon(metrics, [](QPainter& p, int size, const QColor& ink) {
        draw_picture_glyph(p, size, ink);
        // A small curved "history" arrow badge over the bottom-right
        // corner, so the glyph reads as "that picture again" rather
        // than a second, redundant photo icon.
        const QRectF badge(size * 0.48, size * 0.48, size * 0.46, size * 0.46);
        QPainterPath arrow;
        arrow.arcMoveTo(badge, 30);
        arrow.arcTo(badge, 30, 260);
        p.setPen(QPen(ink, std::max(1.0, size / 14.0)));
        p.setBrush(Qt::NoBrush);
        p.drawPath(arrow);
        const QPointF tip = arrow.currentPosition();
        QPolygonF head;
        head << tip << QPointF(tip.x() - size * 0.09, tip.y() - size * 0.02)
             << QPointF(tip.x() - size * 0.01, tip.y() + size * 0.09);
        p.setPen(Qt::NoPen);
        p.setBrush(ink);
        p.drawPolygon(head);
    });
}

QIcon rect_tool_icon(const QWidget* metrics) {
    return draw_tool_icon(metrics, [](QPainter& p, int size, const QColor& ink) {
        const QRectF box(size * 0.15, size * 0.26, size * 0.7, size * 0.48);
        p.setPen(Qt::NoPen);
        p.setBrush(QColor(ink.red(), ink.green(), ink.blue(), 90));
        p.drawRect(box);
        p.setBrush(Qt::NoBrush);
        p.setPen(QPen(ink, std::max(1.0, size / 14.0)));
        p.drawRect(box);
    });
}

}  // namespace

double level_to_db(double level) {
    if (level <= 0.0) return LEVEL_MIN_DB;
    return std::max(LEVEL_MIN_DB, 20.0 * std::log10(level));
}

double db_to_level(double db) { return std::pow(10.0, db / 20.0); }

TransmitPanel::TransmitPanel(AppState* state, QWidget* parent)
    : QWidget(parent), app_(state) {
    build_ui();

    connect(this, &TransmitPanel::stateChanged, this, &TransmitPanel::on_state,
            Qt::QueuedConnection);
    connect(this, &TransmitPanel::errorOccurred, this, &TransmitPanel::on_error,
            Qt::QueuedConnection);
    connect(this, &TransmitPanel::sendFinished, this, &TransmitPanel::on_finished,
            Qt::QueuedConnection);
    connect(this, &TransmitPanel::optimizerProgressed, this,
            &TransmitPanel::on_optimizer_progress, Qt::QueuedConnection);

    // Polls the optimizer while Send waits for it. A timer rather than
    // a blocking wait because nothing on the GUI thread may block --
    // the same rule rig control follows.
    wait_timer_ = new QTimer(this);
    wait_timer_->setInterval(100);
    connect(wait_timer_, &QTimer::timeout, this, [this] {
        if (!awaiting_optimizer_ || optimizer_ == nullptr) return;
        if (!optimizer_->ready()) return;
        wait_timer_->stop();
        awaiting_optimizer_ = false;
        if (!send_picture_) return;
        // The picture captured at the click, and latents for the
        // generation that was current then -- edits since were
        // deferred, so the two still describe the same composition.
        const images::Picture picture = *send_picture_;
        send_picture_.reset();
        begin_transmit(picture, optimizer_->take_result());
    });

    // Coalesces edits before the composite is rebuilt. See
    // `EDIT_DEBOUNCE_MS`, and `send()` for why it has to be flushable.
    edit_timer_ = new QTimer(this);
    edit_timer_->setObjectName(QStringLiteral("edit_debounce"));
    edit_timer_->setSingleShot(true);
    edit_timer_->setInterval(EDIT_DEBOUNCE_MS);
    connect(edit_timer_, &QTimer::timeout, this,
            &TransmitPanel::schedule_optimization);

    rebuild_optimizer();

    // Templates (docs/overlay-templates.md). After `rebuild_optimizer`
    // so the first `refresh_fields` (which `on_template_selected` would
    // trigger, and which `refresh_templates` does not by itself) has an
    // optimizer to hand its result to, same as any other edit.
    refresh_templates();
    sync_custom_field_rows();
    refresh_fields();
}

TransmitPanel::~TransmitPanel() {
    // Before the engine, because the optimizer's worker holds an ORT
    // session and its own thread; stopping it first keeps teardown in
    // one obvious order.
    if (optimizer_) optimizer_->stop();
    if (engine_) engine_->cancel();
    if (thread_.joinable()) thread_.join();
}

void TransmitPanel::sync_from_config() {
    // Turning refinement off must take effect immediately, including
    // dropping any refined latents already in hand: destroying the
    // optimizer discards them, and `send` then falls through to the
    // plain encoder exactly as it did before the feature existed.
    // Turning it on starts a run for the current composition rather
    // than waiting for the operator to touch something.
    rebuild_optimizer();
    // The callsign, grid or name may have just changed, and any of the
    // three can feed a template.
    refresh_fields();
}

void TransmitPanel::rebuild_optimizer() {
    const bool was_running = optimizer_ != nullptr;
    optimizer_.reset();

    if (awaiting_optimizer_) {
        // A send was waiting on a refinement that no longer exists (or
        // is being restarted). Stop waiting and hand the operator the
        // button back rather than silently sending something they did
        // not ask for -- the settings dialog is not a place anyone
        // expects to trigger a transmission.
        wait_timer_->stop();
        awaiting_optimizer_ = false;
        send_picture_.reset();
        send_button_->setEnabled(true);
        set_picture_controls_enabled(true);
        progress_->setRange(0, 100);
        progress_->setValue(0);
    }

    if (!app_->config().transmit.optimize) {
        // Clear a stale "Picture refined: +2.4 dB" that no longer
        // describes what would be sent.
        if (was_running) status_->clear();
        return;
    }
    codec::OnnxCodec* model = app_->model();
    if (model == nullptr) return;  // armed by modelLoaded, or by the next edit

    optimize::SpeculativeConfig cfg;
    const std::string model_path = app_->config().model_path;
    // **One session for the life of this optimizer, not one per run.**
    // Building a `GradSession` loads a ~9 MB artifact from disk and
    // stands up an `Ort::Env` and an `Ort::Session`; only the target
    // changes between runs, and the factory is asked for one on every
    // edit -- per mouse-move burst, after the debounce -- and on every
    // reception if the composition carries a "last received" inset.
    //
    // Held through a shared slot rather than a mutable capture:
    // `Speculative` *copies* the factory before calling it, so anything
    // assigned to a by-value capture would be written into the copy and
    // thrown away, and the cache would silently never hit. A fresh slot
    // per optimizer is also what makes a checkpoint change take effect
    // -- `rebuild_optimizer` is what runs on one.
    //
    // Not a data race despite being shared: the factory is called only
    // from `Speculative`'s single worker, and `optimizer_.reset()`
    // above joined the previous one before this line could run.
    auto session_slot = std::make_shared<std::shared_ptr<codec::GradSession>>();
    optimizer_ = std::make_unique<optimize::Speculative>(
        [model_path, session_slot](const images::ImageArray& target) {
            if (!*session_slot) {
                // Resolved on first use rather than at startup: the
                // artifact is only fetched when the feature is actually
                // used, and `resolve_onnx` pins it to fp32 whatever the
                // codec's precision is.
                const std::string path = checkpoint::resolve_onnx(
                    std::string(checkpoint::GRAD_PART), model_path);
                *session_slot = std::make_shared<codec::GradSession>(path);
            }
            std::shared_ptr<codec::GradSession> session = *session_slot;
            session->set_target(target);
            optimize::GradFn fn = session->fn();
            // Keep the session alive for as long as the gradient
            // function that borrows it.
            return [session, fn](const std::vector<float>& z,
                                 const std::vector<float>& w,
                                 std::vector<float>& grad, double& mse) {
                fn(z, w, grad, mse);
            };
        },
        cfg, [this] { emit optimizerProgressed(); });
    schedule_optimization();
}

void TransmitPanel::schedule_optimization_debounced() {
    // `start()` on a running single-shot timer restarts it, so a drag
    // rebuilds the composite once after the pointer stops rather than
    // once per mouse move.
    edit_timer_->start();
}

void TransmitPanel::schedule_optimization() {
    // Whatever brought us here, the pending edit is now being handled.
    // Leaving the timer armed would rebuild the same composite again a
    // fraction of a second later.
    edit_timer_->stop();
    if (optimizer_ == nullptr) {
        // The model may have finished loading since the last attempt.
        if (app_->config().transmit.optimize && app_->model() != nullptr) {
            rebuild_optimizer();
        }
        return;
    }
    // Send commits to the picture that was on screen when it was
    // pressed, so an edit arriving now belongs to the *next* send.
    // Deferring it also keeps the generation still, which is what lets
    // the latents already in flight stay valid for the committed
    // picture.
    if (transmitting() || awaiting_optimizer_) {
        restart_after_send_ = true;
        return;
    }
    const std::optional<images::Picture> image = editor_->composed_image();
    if (!image) {
        optimizer_->clear();
        return;
    }
    codec::OnnxCodec* model = app_->model();
    if (model == nullptr) return;

    images::ImageArray array = images::to_array(*image);
    const std::string mode_name =
        mode_combo_->currentData().toString().toStdString();
    const config::ModeSpec* mode = &config::MODES[0];
    for (const config::ModeSpec& m : config::MODES) {
        if (mode_name == m.name) mode = &m;
    }
    // The encode runs on the optimizer's worker, after the debounce --
    // so a drag that produces twenty of these pays for none of them.
    optimizer_result_logged_ = false;  // a fresh run gets a fresh log line
    optimizer_->picture_changed(
        array, [model, array] { return model->encode(array); }, *mode);
}

void TransmitPanel::on_optimizer_progress() {
    if (optimizer_ == nullptr || transmitting()) return;
    const optimize::SpeculativeStatus st = optimizer_->status();

    // An *estimate*, and labelled as one -- but a defensible one since
    // 2026-08-04. It is the clean-decode PSNR gain (measured, not a
    // proxy) times `optimize::RETENTION`, the fraction that survives to
    // the receiver. It replaced `objective_gain_db`, which overstated by
    // roughly 3x and could rank two objectives backwards -- on one
    // picture it reported 5.58 dB for a delivered 1.84 and 3.97 for a
    // delivered 3.29.
    //
    // Still approximate, and the tilde says so rather than the code
    // pretending otherwise: retention splits almost additively into an
    // image term the sender can see and a channel term it cannot, and
    // the channel term alone runs 0.39 (mpp at 3 dB) to 0.77 (AWGN at
    // 12 dB). RETENTION is anchored at mpp/6 dB so this under-promises
    // on a good path rather than over-promising on a bad one. It is
    // monotone in everything, so it is honest about *more* or *less*
    // even where the number itself is off by a factor.
    //
    // Shown because a number that climbs is worth having while the
    // operator waits, and kept afterwards because the finished figure
    // is the interesting one (Andrew, 2026-07-31).
    const QString gain =
        QString::asprintf("~%+.1f dB", st.progress.estimated_gain_db);

    if (st.running) {
        status_->setText(tr("Refining image... %1 on air").arg(gain));
    } else if (st.finished && awaiting_optimizer_) {
        status_->setText(tr("Refining image... finishing"));
    } else if (st.finished && st.progress.step > 0) {
        // Whatever ended it -- plateau, either budget, or Send cutting
        // it short -- the gain it did reach stays on screen.
        status_->setText(tr("Image refined: %1 on air").arg(gain));
        if (!optimizer_result_logged_) {
            // The status label cannot say *why* it stopped, so plateau
            // and out-of-time looked identical -- and the figure
            // itself was erased the moment transmit started. The log
            // keeps both.
            optimizer_result_logged_ = true;
            app_->log_event(
                "opt", log::Severity::Info,
                tr("refined %1 on air (%2, %3 steps, %4 s)")
                    .arg(gain)
                    .arg(QString::fromStdString(optimize::to_string(st.stop)))
                    .arg(st.progress.step)
                    .arg(st.progress.elapsed_s, 0, 'f', 1));
        }
    } else if (st.finished) {
        // Finished without a single measured step: the artifact was
        // missing or the encode failed, and the encoder's own latents
        // are what will be sent.
        status_->setText(tr("Sending unrefined"));
        if (!optimizer_result_logged_) {
            optimizer_result_logged_ = true;
            app_->log_event("opt", log::Severity::Warning,
                            tr("refinement produced no result; sending the "
                               "encoder's own latents"));
        }
    }
}

QWidget* TransmitPanel::picture_area() const { return editor_; }

QWidget* TransmitPanel::control_strip() const { return strip_; }

void TransmitPanel::build_ui() {
    // Dropping a file on the composer loads it -- the gesture every
    // other picture application answers, and the one an operator
    // reaches for before finding "Choose image...".
    setAcceptDrops(true);

    auto* layout = new QVBoxLayout(this);

    // The error tier. Everything the transmit path says shares one
    // status label at the end of the send bar, so before this existed
    // "PTT OFF FAILED ... unkey it manually" was replaced by "Sent"
    // within a second. Errors now also land here and stay until
    // dismissed or the next send starts cleanly.
    // **Over the picture, not above it.** In the layout it displaced
    // everything below it -- so an error in one pane pushed that pane's
    // picture down and the two stopped lining up, which is precisely
    // what the equal-size work exists to prevent. Floating it costs no
    // layout at all: it appears, it is dismissed, and nothing moves.
    banner_ = new ErrorBanner(this);
    banner_->hide();

    editor_ = new OverlayEditor(this);
    connect(editor_, &OverlayEditor::selectionChanged, this,
            &TransmitPanel::on_selection);
    // The debounced form: this signal is emitted per mouse move during
    // a drag, and the slot behind it renders the whole composite.
    connect(editor_, &OverlayEditor::documentChanged, this,
            &TransmitPanel::schedule_optimization_debounced);
    // Formatting lives on a right-click menu over the item itself, not on
    // a panel that appears with every selection: a left-click selects
    // and shows handles, and nothing else. Parented to the panel, so it
    // is destroyed with it; it holds no item, only the editor.
    item_menu_ = new ItemMenu(editor_, this);
    connect(editor_, &OverlayEditor::contextMenuRequested, item_menu_,
            &ItemMenu::popup_for);
    // Tools *under* the canvas, not beside it. Beside it they cost the
    // composer ~195 px of width that the receive preview does not pay,
    // so the two pictures could never be the same size however the
    // splitter was weighted -- and the received picture is the one you
    // cannot get back, so it is the one that should not lose.
    // Stretch 10, matching the receive pane's picture: without it the
    // trailing spacer below took the surplus and the canvas stopped at
    // its 640x480 sizeHint, so the two pictures agreed at every window
    // size and both stopped growing at 480 px. Bounded either way --
    // the editor caps itself at 4:3.
    // Stretch 1 and no trailing spacer: the canvas is what the spare
    // room is for. Everything else goes in the strip, whose height is
    // matched against the receive pane's.
    layout->addWidget(editor_);

    strip_ = new QWidget(this);
    auto* strip_layout = new QVBoxLayout(strip_);
    strip_layout->setContentsMargins(0, 0, 0, 0);
    strip_layout->setSpacing(4);
    strip_layout->addWidget(build_tool_row());
    properties_ = build_properties(strip_);
    // **Always present, disabled when nothing is selected** -- not
    // hidden. A row that appears on selection moves the picture under
    // the cursor every time an item is clicked, which is the one thing
    // a composing surface must not do. Its height is part of the strip
    // and therefore part of what the receive pane matches.
    properties_->setEnabled(false);
    strip_layout->addWidget(properties_);
    fields_box_ = build_fields_box(strip_);
    strip_layout->addWidget(fields_box_);
    // The send bar is built first because it is what constructs
    // `progress_`, but the bar goes in *below* it -- a full-width row
    // of its own, the same shape the receive pane gives its progress.
    QWidget* send_bar = build_send_bar();
    strip_layout->addWidget(progress_);
    strip_layout->addWidget(send_bar);
    layout->addWidget(strip_);
}

// One horizontal row of tools under the canvas, replacing the column
// that used to sit beside it. The groupings the column had (Picture /
// Overlay) are carried by separator lines rather than by boxes: three
// nested titled boxes in a row would cost more height than the buttons.
QWidget* TransmitPanel::build_tool_row() {
    auto* panel = new QWidget(this);
    // Wraps rather than crushes -- see flow_layout.hpp.
    auto* column = new FlowLayout(panel);

    auto* source = panel;
    auto* source_layout = column;
    choose_button_ = new QPushButton(tr("&Image..."), source);
    connect(choose_button_, &QPushButton::clicked, this,
            &TransmitPanel::choose_image);
    image_label_ = new style::ElidingLabel(tr("No image selected"), source);
    image_label_->setWordWrap(true);
    frame_button_ = new QPushButton(tr("&Framing..."), source);
    frame_button_->setToolTip(
        tr("Choose which part of the image is sent, for anything that is "
           "not 4:3"));
    frame_button_->setEnabled(false);
    connect(frame_button_, &QPushButton::clicked, this,
            &TransmitPanel::choose_framing);
    // The caption is the one thing here that is text rather than a
    // control, and it is the widest; it gets the row's slack and elides
    // rather than setting a width floor for the whole window.
    image_label_->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Preferred);
    source_layout->addWidget(choose_button_);
    source_layout->addWidget(frame_button_);
    source_layout->addWidget(image_label_);

    // **The rule needs a height of its own.** `FlowLayout` lays every
    // item out at its own size hint rather than stretching it to the
    // row, and a bare `QFrame`'s hint is a few pixels -- so this
    // painted as an invisible dot and the Picture/Overlay grouping the
    // comment above describes did not exist on screen at all. Matched
    // to a button, which is what it stands between.
    auto* rule = new QFrame(panel);
    rule->setFrameShape(QFrame::VLine);
    rule->setFrameShadow(QFrame::Sunken);
    rule->setFixedHeight(choose_button_->sizeHint().height());
    column->addWidget(rule);

    auto* overlay_box = panel;
    auto* overlay_layout = column;
    // **A tool palette, not four sentence-length buttons.** Icon-only
    // `QToolButton`s condense "Add text"/"Add last received"/"Add
    // image..."/"Add rectangle" into a strip that reads as a palette;
    // each still adds its item immediately at a default position and
    // selects it, exactly as the text buttons did -- the click still
    // does the same thing, it is only condensed on screen. The full
    // sentence survives as each button's tooltip.
    add_text_button_ = new QToolButton(overlay_box);
    add_text_button_->setIcon(text_tool_icon(this));
    add_text_button_->setToolTip(tr("Add text"));
    connect(add_text_button_, &QToolButton::clicked, this, [this] {
        const std::string& callsign = app_->config().callsign;
        editor_->add_text(callsign.empty() ? std::string("TEXT") : callsign);
    });

    add_rx_button_ = new QToolButton(overlay_box);
    add_rx_button_->setIcon(last_rx_tool_icon(this));
    // Nothing to insert until something has been received: the item it
    // adds resolves at render time, so before the first reception it
    // drew nothing and looked like a button that did not work.
    add_rx_button_->setEnabled(false);
    add_rx_button_->setToolTip(
        tr("Add last received: inserts the last received image. Available "
           "once one has arrived."));
    connect(add_rx_button_, &QToolButton::clicked, editor_,
            &OverlayEditor::add_last_rx_inset);

    add_image_button_ = new QToolButton(overlay_box);
    add_image_button_->setIcon(image_tool_icon(this));
    add_image_button_->setToolTip(tr("Add image..."));
    connect(add_image_button_, &QToolButton::clicked, this, [this] {
        const QString path = QFileDialog::getOpenFileName(
            this, tr("Choose an inset image"),
            QString::fromStdString(app_->config().folders.transmit_dir),
            QString::fromLatin1(IMAGE_FILTER));
        if (!path.isEmpty()) editor_->add_image_inset(path.toStdString());
    });

    add_rect_button_ = new QToolButton(overlay_box);
    add_rect_button_->setObjectName(QStringLiteral("add_rect_button"));
    add_rect_button_->setIcon(rect_tool_icon(this));
    add_rect_button_->setToolTip(
        tr("Add rectangle: a box, filled and/or stroked with a solid "
           "color or a gradient -- see the floating panel that appears "
           "beside it once it is selected."));
    connect(add_rect_button_, &QToolButton::clicked, editor_,
            &OverlayEditor::add_rect);

    // **Remove is not here; it is in the "Selected item" box.** It is
    // the only control in this row that acts on the *selection* rather
    // than adding something, and the box below is where the selection
    // lives -- so it was both misfiled and, at 85 px, the button that
    // tipped this row onto a second line once the captions were spelled
    // out. Two rows here cost 30 px of picture height on *both* panes,
    // the strips being locked equal, and the received image is the one
    // that cannot be asked for again.
    for (QToolButton* button :
        {add_text_button_, add_rx_button_, add_image_button_, add_rect_button_}) {
        button->setToolButtonStyle(Qt::ToolButtonIconOnly);
        overlay_layout->addWidget(button);
    }

    // **Templates (docs/overlay-templates.md).** A third rule, matching
    // the two above: the label on the vertical rule that follows below
    // is the group Picture/Overlay already imply.
    auto* rule2 = new QFrame(panel);
    rule2->setFrameShape(QFrame::VLine);
    rule2->setFrameShadow(QFrame::Sunken);
    rule2->setFixedHeight(choose_button_->sizeHint().height());
    column->addWidget(rule2);

    template_combo_ = new QComboBox(panel);
    // findChild<QComboBox*>() in test_tx_panel.cpp needs to tell this
    // one apart from `mode_combo_` and `align_combo_`.
    template_combo_->setObjectName(QStringLiteral("template_combo"));
    template_combo_->setToolTip(
        tr("Loads that template's layout onto the canvas, replacing "
           "whatever is there now. \"None\" clears the overlay."));
    connect(template_combo_, &QComboBox::currentIndexChanged, this,
            &TransmitPanel::on_template_selected);
    save_template_button_ = new QPushButton(tr("&Save as template..."), panel);
    save_template_button_->setToolTip(
        tr("Save the current text and layout as a template, with "
           "{placeholders} in it for whatever should vary per "
           "transmission -- {theircall}, {mycall}, {snr}, or a custom "
           "{field Label}."));
    connect(save_template_button_, &QPushButton::clicked, this,
            &TransmitPanel::save_as_template);
    share_template_button_ = new QPushButton(tr("S&hare..."), panel);
    share_template_button_->setObjectName(QStringLiteral("share_template_button"));
    share_template_button_->setToolTip(
        tr("Show the current composition as a QR code and as text, to "
           "copy it to another device."));
    connect(share_template_button_, &QPushButton::clicked, this, [this] {
        // The *canvas*, not the selected template: what is shared is
        // what the operator is looking at, which is the same thing
        // "Save as template..." would write.
        ShareDialog(editor_->doc(), this).exec();
    });
    delete_template_button_ = new QPushButton(tr("De&lete"), panel);
    delete_template_button_->setObjectName(QStringLiteral("delete_template_button"));
    delete_template_button_->setToolTip(
        tr("Delete the selected template's file. The built-ins ship with "
           "the app and cannot be deleted."));
    connect(delete_template_button_, &QPushButton::clicked, this,
            &TransmitPanel::delete_template);
    // One `style::row` item rather than two separate ones: a `FlowLayout`
    // wraps *between* items, and the combo and the button that saves to
    // it are one idea -- "Template: [pick one] [Save...]" -- not two
    // controls that happen to sit near each other. Split across a wrap
    // they read as unrelated.
    column->addWidget(style::row(
        panel, {new QLabel(tr("Template"), panel), template_combo_, save_template_button_,
                delete_template_button_, share_template_button_}));
    // **No `column->addWidget(overlay_box)` here.** `overlay_box` is an
    // alias for `panel`, whose layout `column` *is*, so that line asked
    // Qt to add a widget to its own child layout. Qt refuses and prints
    // "QLayout: Cannot add parent widget to its child layout" on every
    // construction -- of the app, of the screenshot tool, of any test --
    // while the four buttons had already been added by the loop above.
    // A permanently dead line and a permanent warning in our own
    // diagnostics.
    //
    // **The properties box is NOT built here either.** `build_ui` owns it, so
    // constructing a second one in this row left an orphan that nothing
    // ever updated -- and, because it sat in the row, it added 535 px to
    // the transmit pane's minimum width. That is most of the imbalance
    // that made the two pictures unequal: 1088 px against the receive
    // pane's 464.
    return panel;
}

QGroupBox* TransmitPanel::build_properties(QWidget* parent) {
    // **Text only, now.** Scaling, rotation, color, fill/stroke and
    // stacking order used to live here too, in as many as eleven rows --
    // "way too many buttons" on a box that stayed on screen and disabled
    // itself rather than getting out of the way. Scale and rotation
    // moved to on-canvas handles and keyboard shortcuts
    // (`OverlayEditor`); everything else is on the item's right-click
    // menu (`ItemMenu`). The text editor stays here, out of line, because
    // making it inline runs into template field substitution -- the box
    // would have to show the substituted text while still editing the
    // raw `{placeholder}` underneath it, which is not solved yet.
    auto* box = new QGroupBox(tr("Text"), parent);
    box->setEnabled(false);
    // Horizontal: a form stacks its rows, which under the canvas would
    // cost height. Side by side it is one -- and now there are only two.
    auto* form = new FlowLayout(box);

    // Multi-line: a station's callsign, grid and name belong to one
    // item, not three stacked by hand. Enter inserts a newline, so Tab
    // has to be what leaves the field.
    text_edit_ = new QPlainTextEdit(box);
    text_edit_->setTabChangesFocus(true);
    // The panel advertises "drop a picture here"; this widget would
    // answer that gesture by inserting the file's URL as overlay text
    // and sending it on the air. Let the drop fall through to the
    // panel, which loads it.
    text_edit_->setAcceptDrops(false);
    // Two lines rather than four: still multi-line (a callsign, a grid
    // and a name belong to one item), but in a row under the canvas the
    // height is the picture's.
    //
    // **Measured, not 46 px.** A pixel constant standing in for "two
    // lines" is two lines at exactly one font size: at 150% scaling it
    // is one and a bit, and the second line -- the reason the field is
    // multi-line at all -- is cut off. The rest of this project already
    // sizes from metrics (`level_label_`'s minimum width, the log
    // pane's minimum height); this was the exception.
    text_edit_->setFixedHeight(text_edit_->fontMetrics().lineSpacing() * 2 +
                               text_edit_->document()->documentMargin() * 2 +
                               text_edit_->frameWidth() * 2 + 4);
    // **`Preferred` with a small floor, not `Ignored`.** `Ignored` was
    // right while this widget was added to the `FlowLayout` directly:
    // that layout falls back to asking the widget for its hint, so the
    // field still got a width. Inside a `style::row` -- which it needs
    // to be, so a wrap cannot separate it from its label -- the hint
    // goes through a `QHBoxLayout`, which honours `Ignored` and gives it
    // *nothing*: the field disappeared entirely, leaving a "Text" label
    // captioning empty space. A modest minimum is not a window floor
    // either way, because `FlowLayout::minimumSize` takes the widest
    // single item and the button rows are wider than this.
    text_edit_->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Fixed);
    text_edit_->setMinimumWidth(120);
    text_edit_->setToolTip(
        tr("The text drawn on the image. Enter starts a new line; Tab leaves "
           "the field."));
    connect(text_edit_, &QPlainTextEdit::textChanged, this, [this] {
        if (auto* item = editing_item()) {
            if (auto* text = std::get_if<overlay::TextItem>(item)) {
                text->text = text_edit_->toPlainText().toStdString();
                editor_->refresh_item();
            }
        }
    });
    form->addWidget(style::row(box, {new QLabel(tr("Text"), box), text_edit_}, 1));

    align_combo_ = new QComboBox(box);
    align_combo_->addItem(tr("Left"), QStringLiteral("left"));
    align_combo_->addItem(tr("Centre"), QStringLiteral("center"));
    align_combo_->addItem(tr("Right"), QStringLiteral("right"));
    connect(align_combo_, &QComboBox::currentIndexChanged, this, [this] {
        if (auto* item = editing_item()) {
            if (auto* text = std::get_if<overlay::TextItem>(item)) {
                text->align = align_combo_->currentData().toString().toStdString();
                editor_->refresh_item();
            }
        }
    });
    form->addWidget(style::row(box, {new QLabel(tr("Align"), box), align_combo_}));

    return box;
}

QGroupBox* TransmitPanel::build_fields_box(QWidget* parent) {
    auto* box = new QGroupBox(tr("Reply fields"), parent);
    auto* form = new FlowLayout(box);

    theircall_edit_ = new QLineEdit(box);
    theircall_edit_->setObjectName(QStringLiteral("theircall_edit"));
    theircall_edit_->setPlaceholderText(QStringLiteral("W1XYZ"));
    theircall_edit_->setMaxLength(16);
    theircall_edit_->setToolTip(
        tr("Fills {theircall} in the current template. Disabled when it "
           "does not use one."));
    connect(theircall_edit_, &QLineEdit::textChanged, this,
            [this] { refresh_fields(); });
    form->addWidget(
        style::row(box, {new QLabel(tr("Their call"), box), theircall_edit_}));

    // **Live, inline, up to `MAX_INLINE_CUSTOM_FIELDS`** -- the rows
    // exist from construction on, whatever the template, and only their
    // label text, enabled state and value change; see the declaration's
    // comment for why that has to be true here just as it is for
    // `build_properties`. Each row writes straight into
    // `custom_field_values_` and calls `refresh_fields()` on every
    // keystroke, which is what makes the composite preview track typing
    // rather than needing a Done button.
    for (int i = 0; i < MAX_INLINE_CUSTOM_FIELDS; ++i) {
        custom_field_labels_[i] = new QLabel(box);
        custom_field_labels_[i]->setEnabled(false);
        custom_field_edits_[i] = new QLineEdit(box);
        custom_field_edits_[i]->setObjectName(
            QStringLiteral("custom_field_edit_%1").arg(i));
        custom_field_edits_[i]->setEnabled(false);
        connect(custom_field_edits_[i], &QLineEdit::textChanged, this,
                [this, i] { on_custom_field_edited(i); });
        form->addWidget(
            style::row(box, {custom_field_labels_[i], custom_field_edits_[i]}));
    }

    // The overflow pop-up: a template that declares more custom fields
    // than fit inline. Disabled and unlabelled with nothing to overflow,
    // which for every built-in template today is always -- they declare
    // one apiece.
    custom_fields_button_ = new QPushButton(tr("Custom fields..."), box);
    custom_fields_button_->setEnabled(false);
    connect(custom_fields_button_, &QPushButton::clicked, this,
            &TransmitPanel::open_custom_fields_dialog);
    form->addWidget(custom_fields_button_);

    return box;
}

QWidget* TransmitPanel::build_send_bar() {
    auto* bar = new QWidget(this);
    // Wraps: the mode name, the level slider, Send, Cancel and the
    // status field do not fit one line on a narrow pane.
    auto* layout = new FlowLayout(bar);
    layout->setContentsMargins(0, 0, 0, 0);

    mode_combo_ = new QComboBox(bar);
    for (const config::ModeSpec& spec : config::MODES) {
        const QString name = QString::fromUtf8(spec.name.data(),
                                               static_cast<int>(spec.name.size()));
        mode_combo_->addItem(
            tr("Mode %1 - %2 s").arg(name).arg(spec.duration_s, 0, 'f', 0), name);
    }
    const int mode_index =
        mode_combo_->findData(QString::fromStdString(app_->config().transmit.mode));
    mode_combo_->setCurrentIndex(std::max(0, mode_index));
    mode_combo_->setToolTip(
        tr("How long the transmission takes, and how much detail it carries. "
           "Longer modes send more latents, so they survive a poorer path."));
    connect(mode_combo_, &QComboBox::currentIndexChanged, this,
            &TransmitPanel::on_mode_changed);

    // The level belongs beside Send rather than in the settings dialog,
    // because setting it means watching the radio's ALC while
    // transmitting and a modal dialog covering the window makes that
    // awkward.
    level_slider_ = new QSlider(Qt::Horizontal, bar);
    level_slider_->setRange(static_cast<int>(std::lround(LEVEL_MIN_DB / LEVEL_STEP_DB)),
                            0);
    level_slider_->setSingleStep(1);
    level_slider_->setPageStep(2);  // one whole dB
    // Wide enough to aim with, but a *minimum* rather than a fixed
    // width so the send bar can be narrowed; the slider gives up its
    // extra length before anything starts clipping.
    level_slider_->setMinimumWidth(80);
    level_slider_->setMaximumWidth(140);
    // The short form. The full procedure is in Settings > Transmit,
    // where it can be read before the first send rather than only by
    // someone who already suspects the level is wrong -- and the two
    // must name the *same* target, because "barely moving ALC" and "no
    // ALC action" are different drive levels and an operator following
    // either should land in the same place.
    level_slider_->setToolTip(
        tr("Output level, dB relative to full scale.\n\n"
           "Set it so the radio shows no ALC action at all -- ALC is a "
           "compressor, and it flattens the peaks this waveform carries "
           "information in. Full procedure in Settings > Transmit."));
    // Set before connecting, so restoring the saved value is not itself
    // treated as an edit worth writing back.
    level_slider_->setValue(static_cast<int>(
        std::lround(level_to_db(app_->config().transmit.level) / LEVEL_STEP_DB)));
    connect(level_slider_, &QSlider::valueChanged, this,
            &TransmitPanel::on_level_changed);

    level_label_ = new QLabel(bar);
    level_label_->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
    level_label_->setMinimumWidth(
        level_label_->fontMetrics().horizontalAdvance(QStringLiteral("-30.0 dB")));

    save_level_timer_ = new QTimer(this);
    save_level_timer_->setSingleShot(true);
    save_level_timer_->setInterval(LEVEL_SAVE_DELAY_MS);
    connect(save_level_timer_, &QTimer::timeout, this,
            [this] { app_->save_config(); });
    update_level_label();

    send_button_ = new QPushButton(tr("&Send"), bar);
    connect(send_button_, &QPushButton::clicked, this, &TransmitPanel::send);
    send_button_->setToolTip(
        tr("Encode the composition, key the radio and send it. Any refinement "
           "still running is finished first."));
    cancel_button_ = new QPushButton(tr("&Cancel"), bar);
    cancel_button_->setToolTip(
        tr("Stop the transmission and unkey the radio."));
    connect(cancel_button_, &QPushButton::clicked, this, &TransmitPanel::cancel);
    cancel_button_->setEnabled(false);

    // **Not in this row.** Inline among the buttons, with `Ignored`
    // horizontally, it rendered as a blank rounded rectangle between
    // Cancel and the status text -- indistinguishable from a disabled
    // line edit, and nothing on screen said it was the transmission's
    // progress. The receive pane gives its bar a full-width row of its
    // own; `build_ui` now does the same here, and the two panes report
    // progress the same way. Built here because this is where its
    // siblings are configured.
    progress_ = new QProgressBar(strip_);
    progress_->setRange(0, 100);

    status_ = new style::ElidingLabel(tr("Ready"), bar);
    // Progress-tier text must not set a width floor: the two panes sit
    // in a splitter whose minimum is the *sum* of its children's, so
    // every pixel this label demands is a pixel the window cannot be
    // narrowed by. Errors go to the banner above, which wraps.
    status_->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Preferred);

    // No "Mode:" label: every item in the combo already begins with the
    // word, so the row read "Mode: Mode B - 64 s".
    layout->addWidget(mode_combo_);
    // **Caption, slider and readout are one item.** A `FlowLayout` wraps
    // between items, and these were three of them, so a narrow pane
    // could leave the slider on one line and the number it is showing
    // on the next -- or strand "Level:" above the thing it names. None
    // of the three means anything alone: the slider has no scale
    // printed on it, so the readout *is* how you know where it is set.
    // Grouped, they wrap as a unit or not at all.
    layout->addWidget(style::row(
        bar, {new QLabel(tr("Level:"), bar), level_slider_, level_label_}));
    layout->addWidget(send_button_);
    layout->addWidget(cancel_button_);
    layout->addWidget(status_);
    return bar;
}

void TransmitPanel::on_level_changed(int steps) {
    app_->config().transmit.level = db_to_level(steps * LEVEL_STEP_DB);
    update_level_label();
    save_level_timer_->start();
}

void TransmitPanel::update_level_label() {
    level_label_->setText(
        tr("%1 dB").arg(level_slider_->value() * LEVEL_STEP_DB, 0, 'f', 1));
}

void TransmitPanel::on_mode_changed() {
    // A different mode is a different latent budget, so any result in
    // hand is for the wrong transmission.
    schedule_optimization();
    app_->config().transmit.mode = mode_combo_->currentData().toString().toStdString();
    app_->save_config();
    // `{mode}` may be in the composition. `schedule_optimization` above
    // already rebuilds the composite, but from `editor_->composed_image()`
    // -- which substitutes -- so this must run *before* it would matter
    // and does no harm running after; ordered here for readability, not
    // correctness.
    refresh_fields();
}

// --- templates (docs/overlay-templates.md) -----------------------------------

std::filesystem::path TransmitPanel::builtin_templates_dir() {
    // Copied here at build time (`sstvae_copy_builtin_templates` in
    // native/CMakeLists.txt) from `sstvae/overlay/templates/`, the one
    // place the three ship from. A build tree and an installed tree
    // both put it beside the executable, so this is the only path that
    // has to be right.
    return (QCoreApplication::applicationDirPath() + QStringLiteral("/templates"))
        .toStdString();
}

void TransmitPanel::refresh_templates() {
    const QString previous = template_combo_->currentText();

    templates_.clear();
    template_paths_.clear();
    template_combo_->blockSignals(true);
    template_combo_->clear();

    // Index 0: not a file, not loaded from anywhere -- an empty
    // document is exactly today's "no overlay" behaviour.
    templates_.push_back(overlay::Doc());
    template_paths_.emplace_back();
    template_combo_->addItem(tr("None"));

    for (overlay::Doc& doc : overlay::load_builtin_templates(builtin_templates_dir())) {
        template_combo_->addItem(QString::fromStdString(doc.name));
        templates_.push_back(std::move(doc));
        // Deliberately pathless even though these *are* files: they sit
        // beside the executable, in a directory an installed app has no
        // business writing to, so "can this be deleted" is the same
        // question as "did it come from the operator's folder".
        template_paths_.emplace_back();
    }
    for (overlay::LoadedTemplate& loaded :
        overlay::load_templates(app_->config().folders.template_dir)) {
        const QString label = QString::fromStdString(
            loaded.doc.name.empty() ? loaded.path.stem().string() : loaded.doc.name);
        template_combo_->addItem(label);
        templates_.push_back(std::move(loaded.doc));
        template_paths_.push_back(std::move(loaded.path));
    }

    // Keep the same selection across a refresh (a saved template landed
    // in the same list it was chosen from) rather than silently
    // snapping back to "None".
    const int keep = template_combo_->findText(previous);
    template_combo_->setCurrentIndex(std::max(0, keep));
    template_combo_->blockSignals(false);
    // Signals are blocked above, so `on_template_selected` -- which is
    // the other place this is kept in step -- does not run here.
    update_delete_enabled();
}

// `setEnabled`, never `setVisible`: this row is in the control strip,
// whose height must not change (see the file header of
// test_tx_panel.cpp).
void TransmitPanel::update_delete_enabled() {
    const int index = template_combo_->currentIndex();
    delete_template_button_->setEnabled(
        index >= 0 && index < static_cast<int>(template_paths_.size()) &&
        !template_paths_[index].empty());
}

void TransmitPanel::on_template_selected(int index) {
    if (index < 0 || index >= static_cast<int>(templates_.size())) return;
    // Replaces the composition outright -- see the combo's tooltip.
    // `set_doc` emits `documentChanged`, already connected to the
    // debounced optimizer, so nothing further is needed for that half.
    editor_->set_doc(templates_[index]);
    update_delete_enabled();
    // The *set* of custom fields can only change here (a template
    // switch), never on a keystroke -- see `sync_custom_field_rows`'s
    // own comment for why that split matters.
    sync_custom_field_rows();
    refresh_fields();
}

void TransmitPanel::sync_custom_field_rows() {
    const overlay::Placeholders used = overlay::placeholders(editor_->doc());
    for (int i = 0; i < MAX_INLINE_CUSTOM_FIELDS; ++i) {
        if (i < static_cast<int>(used.custom.size())) {
            const std::string& label = used.custom[i];
            custom_field_slots_[i] = label;
            custom_field_labels_[i]->setText(QString::fromStdString(label));
            custom_field_labels_[i]->setEnabled(true);
            custom_field_edits_[i]->setEnabled(true);
            const auto it = custom_field_values_.find(label);
            const QString value = it != custom_field_values_.end()
                                       ? QString::fromStdString(it->second)
                                       : QString();
            // Only when it actually differs: `setText` moves the cursor
            // to the end even when the content is unchanged, and this
            // runs from `on_template_selected` while the operator may
            // already be typing in a row that kept the same label.
            if (custom_field_edits_[i]->text() != value) {
                custom_field_edits_[i]->setText(value);
            }
        } else {
            custom_field_slots_[i].clear();
            custom_field_labels_[i]->setText(QString());
            custom_field_labels_[i]->setEnabled(false);
            custom_field_edits_[i]->setEnabled(false);
            if (!custom_field_edits_[i]->text().isEmpty()) custom_field_edits_[i]->clear();
        }
    }
    const int overflow =
        std::max(0, static_cast<int>(used.custom.size()) - MAX_INLINE_CUSTOM_FIELDS);
    custom_fields_button_->setEnabled(overflow > 0);
    custom_fields_button_->setText(overflow > 0
                                       ? tr("Custom fields (+%1)...").arg(overflow)
                                       : tr("Custom fields..."));
}

void TransmitPanel::on_custom_field_edited(int slot) {
    if (custom_field_slots_[slot].empty()) return;  // an inactive row; should not fire
    custom_field_values_[custom_field_slots_[slot]] =
        custom_field_edits_[slot]->text().toStdString();
    refresh_fields();
}

void TransmitPanel::refresh_fields() {
    const overlay::Placeholders used = overlay::placeholders(editor_->doc());
    const bool wants_theircall =
        std::find(used.builtin.begin(), used.builtin.end(), std::string("theircall")) !=
        used.builtin.end();
    theircall_edit_->setEnabled(wants_theircall);

    overlay::Fields fields;
    const settings::Config& cfg = app_->config();
    fields.builtin["mycall"] = cfg.callsign;
    fields.builtin["grid"] = cfg.grid;
    fields.builtin["name"] = cfg.operator_name;
    fields.builtin["theircall"] = theircall_edit_->text().trimmed().toStdString();
    fields.builtin["snr"] = overlay::format_snr(last_reception_snr_db_);
    const QDateTime now = QDateTime::currentDateTimeUtc();
    fields.builtin["utc"] = now.toString(QStringLiteral("HH:mm")).toStdString();
    fields.builtin["date"] = now.toString(QStringLiteral("yyyy-MM-dd")).toStdString();
    fields.builtin["mode"] = mode_combo_->currentData().toString().toStdString();
    for (const std::string& label : used.custom) {
        const auto it = custom_field_values_.find(label);
        if (it != custom_field_values_.end()) fields.custom[label] = it->second;
    }
    editor_->set_fields(std::move(fields));
}

void TransmitPanel::open_custom_fields_dialog() {
    // Only the overflow: labels 0..MAX_INLINE_CUSTOM_FIELDS-1 already
    // have their own live row in `fields_box_`, and duplicating them
    // here would mean two controls writing the same
    // `custom_field_values_` entry.
    const overlay::Placeholders used = overlay::placeholders(editor_->doc());
    if (static_cast<int>(used.custom.size()) <= MAX_INLINE_CUSTOM_FIELDS) return;

    QDialog dialog(this);
    dialog.setWindowTitle(tr("Custom fields"));
    auto* form = new QFormLayout(&dialog);
    form->addRow(style::note(
        tr("Never mandatory -- leave one blank and its line is left out."),
        &dialog));
    std::vector<std::pair<std::string, QLineEdit*>> edits;
    for (std::size_t i = MAX_INLINE_CUSTOM_FIELDS; i < used.custom.size(); ++i) {
        const std::string& label = used.custom[i];
        auto* edit = new QLineEdit(&dialog);
        const auto it = custom_field_values_.find(label);
        if (it != custom_field_values_.end()) {
            edit->setText(QString::fromStdString(it->second));
        }
        form->addRow(QString::fromStdString(label), edit);
        edits.emplace_back(label, edit);
    }
    auto* buttons =
        new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel, &dialog);
    connect(buttons, &QDialogButtonBox::accepted, &dialog, &QDialog::accept);
    connect(buttons, &QDialogButtonBox::rejected, &dialog, &QDialog::reject);
    form->addRow(buttons);

    if (dialog.exec() != QDialog::Accepted) return;
    for (auto& [label, edit] : edits) {
        custom_field_values_[label] = edit->text().toStdString();
    }
    refresh_fields();
}

void TransmitPanel::save_as_template() {
    // Defaults to the loaded template's own name, if one was loaded --
    // `editor_->doc().name` already carries it, since `on_template_selected`
    // loads the template verbatim (name included) onto the canvas.
    // Re-saving over the same template is then the zero-edit path, and
    // saving under a new name is a one-word change rather than a blank
    // field to retype every time.
    bool ok = false;
    const QString name = QInputDialog::getText(
        this, tr("Save as template"), tr("Template name"), QLineEdit::Normal,
        QString::fromStdString(editor_->doc().name), &ok);
    if (!ok || name.trimmed().isEmpty()) return;

    overlay::Doc doc = editor_->doc();
    doc.name = name.trimmed().toStdString();

    const std::filesystem::path dir = app_->config().folders.template_dir;
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    const std::filesystem::path path = dir / (overlay::slugify(doc.name) + ".json");

    if (std::filesystem::exists(path)) {
        const auto reply = QMessageBox::question(
            this, tr("Replace template?"),
            tr("A template file named \"%1\" already exists. Replace it?")
                .arg(QString::fromStdString(path.filename().string())));
        if (reply != QMessageBox::Yes) return;
    }

    try {
        std::ofstream out(path, std::ios::binary);
        if (!out) throw std::runtime_error("could not open the file for writing");
        out << overlay::to_json(doc);
        if (!out) throw std::runtime_error("write failed");
    } catch (const std::exception& e) {
        app_->log_event("tx", log::Severity::Error,
                        tr("could not save template \"%1\": %2")
                            .arg(name.trimmed(), QString::fromUtf8(e.what())));
        QMessageBox::critical(this, tr("Could not save template"),
                              QString::fromUtf8(e.what()));
        return;
    }
    app_->log_event("tx", log::Severity::Info,
                    tr("saved template \"%1\"").arg(name.trimmed()));
    refresh_templates();
}

void TransmitPanel::delete_template() {
    const int index = template_combo_->currentIndex();
    if (index < 0 || index >= static_cast<int>(template_paths_.size())) return;
    const std::filesystem::path path = template_paths_[index];
    if (path.empty()) return;  // "None" or a built-in; the button is disabled

    const QString label = template_combo_->itemText(index);
    if (QMessageBox::question(
            this, tr("Delete template?"),
            tr("Delete the template \"%1\"? This removes %2 from disk.")
                .arg(label, QString::fromStdString(path.filename().string()))) !=
        QMessageBox::Yes) {
        return;
    }

    std::error_code ec;
    std::filesystem::remove(path, ec);
    if (ec) {
        app_->log_event("tx", log::Severity::Error,
                        tr("could not delete template \"%1\": %2")
                            .arg(label, QString::fromStdString(ec.message())));
        QMessageBox::critical(this, tr("Could not delete template"),
                              QString::fromStdString(ec.message()));
        return;
    }
    app_->log_event("tx", log::Severity::Info, tr("deleted template \"%1\"").arg(label));
    // The canvas keeps what it holds: deleting the file is not a
    // request to throw away the composition in front of the operator,
    // who may well be about to re-save it under another name. The combo
    // falls back to "None" because the name it held is gone.
    refresh_templates();
}

// --- content ----------------------------------------------------------------

void TransmitPanel::choose_image() {
    const QString path = QFileDialog::getOpenFileName(
        this, tr("Choose an image"),
        QString::fromStdString(app_->config().folders.transmit_dir),
        QString::fromLatin1(IMAGE_FILTER));
    if (!path.isEmpty()) load_image(path);
}

void TransmitPanel::load_image(const QString& path) {
    if (picture_locked()) return;
    images::Picture loaded;
    try {
        loaded = images::load(path.toStdString());
    } catch (const std::exception& e) {
        app_->log_event("tx", log::Severity::Error,
                        tr("could not open image %1: %2")
                            .arg(path, QString::fromUtf8(e.what())));
        QMessageBox::critical(this, tr("Could not open image"),
                              QString::fromUtf8(e.what()));
        return;
    }

    // The *original* is kept, not the framed result: re-framing has to
    // start from the picture the operator chose, or each adjustment
    // would crop what the last one had already cropped.
    source_ = std::move(loaded);
    source_path_ = path;
    framing_ = images::Framing{};

    if (source_->width < images::MIN_W || source_->height < images::MIN_H) {
        // Upscaled without comment until now. It still is -- refusing
        // would be worse -- but the operator should know why the
        // picture looks soft.
        app_->log_event(
            "tx", log::Severity::Warning,
            tr("%1 is %2x%3, below the %4x%5 minimum; it will be upscaled")
                .arg(QFileInfo(path).fileName())
                .arg(source_->width)
                .arg(source_->height)
                .arg(images::MIN_W)
                .arg(images::MIN_H));
    }

    // 4:3 needs no decision, so it is not asked for. Compared as a
    // ratio of integers rather than a float equality, so 640x480 and
    // 1600x1200 both count as exact.
    const bool four_by_three =
        source_->width * images::IMG_H == source_->height * images::IMG_W;
    if (!four_by_three) {
        CropDialog dialog(*source_, framing_, this);
        if (dialog.exec() == QDialog::Accepted) framing_ = dialog.framing();
    }
    // Once, either way: cancelling the dialog means "the default
    // framing", not "do not show me the picture".
    apply_framing();

    app_->config().folders.transmit_dir =
        QFileInfo(path).absolutePath().toStdString();
    app_->save_config();
}

bool TransmitPanel::picture_locked() const {
    // Between the Send click and the end of the transmission the
    // picture is committed, so changing it is refused rather than
    // queued. The reason is sharper than tidiness: both entry points
    // can open a *modal* dialog, whose nested event loop keeps pumping
    // timers -- including `wait_timer_`, which starts the transmission.
    // The radio would key while a framing dialog covered the Cancel
    // button. The buttons are disabled to say so; this guard is what
    // makes it true for a dropped file, which reaches no button.
    return transmitting() || awaiting_optimizer_;
}

void TransmitPanel::set_picture_controls_enabled(bool on) {
    choose_button_->setEnabled(on);
    frame_button_->setEnabled(on && source_.has_value());
}

void TransmitPanel::choose_framing() {
    if (picture_locked()) return;
    if (!source_) return;
    CropDialog dialog(*source_, framing_, this);
    // Cancel changes nothing, so nothing is re-applied -- re-fitting
    // would also throw away a speculative optimization that is still
    // valid for the picture on screen.
    if (dialog.exec() != QDialog::Accepted) return;
    framing_ = dialog.framing();
    apply_framing();
}

void TransmitPanel::apply_framing() {
    if (!source_) return;
    images::Picture framed;
    try {
        // Framed to the transmit size here, not at send time: the
        // overlay's coordinates are fractions of the canvas, so the
        // operator has to be composing against the frame that will
        // actually go out.
        //
        // In the try because it allocates and resamples: a huge source
        // at a high zoom can throw, and this runs from a slot, where an
        // escaping exception takes the process down instead of showing
        // a message. It used to sit inside `load_image`'s handler.
        framed = images::fit(*source_, framing_);
    } catch (const std::exception& e) {
        app_->log_event("tx", log::Severity::Error,
                        tr("could not frame %1: %2")
                            .arg(QFileInfo(source_path_).fileName(),
                                 QString::fromUtf8(e.what())));
        QMessageBox::critical(this, tr("Could not frame the image"),
                              QString::fromUtf8(e.what()));
        return;
    }
    editor_->set_base_image(framed);

    // An honest caption. The old one said the filename and nothing
    // else, so a picture that had lost a quarter of its width looked
    // exactly like one that had not.
    QString caption = QFileInfo(source_path_).fileName();
    caption += tr("\n%1x%2").arg(source_->width).arg(source_->height);
    const bool four_by_three =
        source_->width * images::IMG_H == source_->height * images::IMG_W;
    // Zoomed out, nothing is cropped -- the canvas is padded instead,
    // and saying "cropped" there would be exactly as misleading as the
    // bare filename this replaced.
    if (framing_.zoom < 1.0) {
        caption += tr(", padded to 4:3");
    } else if (!four_by_three || framing_.zoom > 1.0) {
        caption += tr(", cropped to 4:3");
    }
    image_label_->setText(caption);
    frame_button_->setEnabled(true);
    // No schedule_optimization() here: `set_base_image` emits
    // documentChanged, which is already connected to it. Calling it
    // again would convert the picture twice and restart the debounce
    // the first call had just started.
}

void TransmitPanel::dragEnterEvent(QDragEnterEvent* event) {
    // Accept only a single local file. Multiple would be ambiguous --
    // one picture goes out at a time -- and a remote URL would mean
    // fetching, which this panel has no business doing.
    const QMimeData* mime = event->mimeData();
    if (!mime->hasUrls()) return;
    const QList<QUrl> urls = mime->urls();
    if (urls.size() != 1 || !urls.front().isLocalFile()) return;
    event->acceptProposedAction();
}

void TransmitPanel::dropEvent(QDropEvent* event) {
    const QList<QUrl> urls = event->mimeData()->urls();
    if (urls.size() != 1 || !urls.front().isLocalFile()) return;
    event->acceptProposedAction();

    // Deferred, not called here. `load_image` can open the framing
    // dialog or a message box, and a nested event loop inside
    // `dropEvent` means this handler has not returned -- so the
    // platform's drop handshake is unfinished and the *source*
    // application stays blocked for as long as the operator spends
    // choosing a crop. Returning first and loading on the next tick
    // costs nothing and ends the drag properly.
    const QString path = urls.front().toLocalFile();
    QTimer::singleShot(0, this, [this, path] { load_image(path); });
}

void TransmitPanel::set_last_reception_info(const QString& callsign, double snr_db) {
    Q_UNUSED(callsign);
    last_reception_snr_db_ =
        std::isnan(snr_db) ? std::nullopt : std::optional<double>(snr_db);
    refresh_fields();
}

void TransmitPanel::set_last_rx_image(const images::Picture& image) {
    editor_->set_last_rx(image);
    // Only now is there anything for the button to insert. Before the
    // first reception it added an item that rendered as nothing, which
    // reads as a broken button rather than as "not yet".
    add_rx_button_->setEnabled(true);
    add_rx_button_->setToolTip(QString());
}

// --- property editing -------------------------------------------------------

overlay::Item* TransmitPanel::editing_item() {
    // Null while the widgets are being filled from an item, so their
    // change signals do not write the value straight back.
    if (loading_properties_) return nullptr;
    return editor_->selected_item();
}

void TransmitPanel::on_selection(overlay::Item* item) {
    // **`properties_` (Text/Align): disabled, never hidden** -- see
    // `build_ui`, which this used to contradict outright while both
    // sites carried a comment asserting the opposite rule.
    //
    // Two things went wrong when it hid. The row is tall, so the canvas
    // jumped under the pointer on every select and deselect, which is
    // the one thing a composing surface must not do. And
    // `PaneContainer::equalise_strips` runs only from
    // `set_control_strips` and a resize -- nothing re-runs it when a
    // strip's *content* changes height -- so from the first click the
    // two strips held minimums computed for a layout that no longer
    // existed and the two pictures silently stopped matching until the
    // window was resized. `test_pane_container.cpp` cannot see that:
    // its stand-in panes have static strips. `test_tx_panel.cpp` is the
    // guard.
    properties_->setEnabled(item != nullptr);

    const auto* text = item != nullptr ? std::get_if<overlay::TextItem>(item) : nullptr;
    loading_properties_ = true;
    text_edit_->setEnabled(text != nullptr);
    align_combo_->setEnabled(text != nullptr);
    if (text != nullptr) {
        text_edit_->setPlainText(QString::fromStdString(text->text));
        align_combo_->setCurrentIndex(
            std::max(0, align_combo_->findData(QString::fromStdString(text->align))));
    } else {
        text_edit_->setPlainText(QString());
    }
    loading_properties_ = false;
}

// --- transmitting -----------------------------------------------------------

void TransmitPanel::resizeEvent(QResizeEvent* event) {
    QWidget::resizeEvent(event);
    place_banner();
}

void TransmitPanel::place_banner() { style::place_over(banner_, editor_); }

bool TransmitPanel::transmitting() const { return running_.load(); }

void TransmitPanel::send() {
    if (transmitting()) return;

    // **Flush any edit still sitting in the debounce, first.** The
    // generation counter inside `Speculative` is what guarantees the
    // latents that go on the air describe the picture that goes on the
    // air, and it can only know about an edit that reached it. An edit
    // made within `EDIT_DEBOUNCE_MS` of the click has not: without this,
    // `request_send()` below would wait on the *previous* generation,
    // find a result ready for it, and transmit the current composition
    // with latents refined for the one before it -- which is precisely
    // the failure the whole speculative design exists to prevent, and
    // it would look like nothing at all, because refined latents for
    // the wrong picture still decode to a picture.
    //
    // Synchronous on purpose: it is one composite render, and Send is
    // already a moment where a few milliseconds are invisible.
    if (edit_timer_->isActive()) schedule_optimization();

    const std::optional<images::Picture> image = editor_->composed_image();
    if (!image) {
        QMessageBox::information(this, tr("No image"),
                                 tr("Choose an image to transmit first."));
        return;
    }
    codec::OnnxCodec* model = app_->model();
    if (model == nullptr) {
        QMessageBox::warning(
            this, tr("Model still loading"),
            tr("The codec checkpoint is still loading. Try again in a moment."));
        return;
    }
    // The same predicate the engine skips the ID on, asked here so it
    // is a refusal the operator can act on rather than a CW ID that
    // silently never goes out. Blocking rather than warning, because
    // every way out of it is in Settings and none of them is something
    // to decide with the radio keyed.
    {
        const settings::Config& config = app_->config();
        const std::string problem = tx::cw_id_problem(
            config.transmit.cw_id, config.transmit.cw_message, config.callsign);
        if (!problem.empty()) {
            QMessageBox::warning(this, tr("CW ID is not set up"),
                                 QString::fromStdString(problem));
            return;
        }
    }
    if (thread_.joinable()) thread_.join();

    // A fresh attempt clears the previous one's error; the log keeps it.
    banner_->clear();

    // Optimization, if it is on, must be entirely finished before the
    // radio is keyed -- so Send shortens its deadline and then waits,
    // on a timer rather than by blocking. `Speculative::ready()` is
    // true immediately when there is nothing to wait for, so this costs
    // nothing when the feature is off.
    if (optimizer_ != nullptr && !awaiting_optimizer_) {
        optimizer_->request_send();
        if (!optimizer_->ready()) {
            send_picture_ = *image;
            awaiting_optimizer_ = true;
            send_button_->setEnabled(false);
            set_picture_controls_enabled(false);
            status_->setText(tr("Refining image..."));
            progress_->setRange(0, 0);
            wait_timer_->start();
            return;
        }
        begin_transmit(*image, optimizer_->take_result());
        return;
    }
    begin_transmit(*image, {});
}

void TransmitPanel::begin_transmit(const images::Picture& picture,
                                   std::vector<double> latents) {
    codec::OnnxCodec* model = app_->model();
    if (model == nullptr) return;

    const settings::Config& config = app_->config();
    tx::TxConfig tx_config;
    tx_config.mode = mode_combo_->currentData().toString().toStdString();
    tx_config.callsign = config.callsign;
    tx_config.device = config.audio.output_device;
    tx_config.level = config.transmit.level;
    tx_config.ptt_lead_s = config.rig.ptt_lead_s;
    tx_config.ptt_tail_s = config.rig.ptt_tail_s;
    tx_config.cw_id = config.transmit.cw_id;
    tx_config.cw_message = config.transmit.cw_message;
    tx_config.vox_lead_s = config.transmit.vox_lead_s;

    engine_ = std::make_unique<tx::TxEngine>(
        app_->ptt(),
        [](const std::string& device, std::span<const double> wave, int samplerate,
           const std::function<void(double)>& on_progress,
           const std::function<bool()>& should_stop,
           const std::function<void(const std::string&)>& on_error) {
            return audio::qt::play(device, wave, samplerate, on_progress,
                                   should_stop, on_error);
        },
        // The optimizer's output enters here, through the seam the
        // engine already had. Empty means it was off, unfinished, or
        // failed -- in which case this is the plain encoder and the
        // picture is exactly what it would always have been.
        [model, latents](const images::ImageArray& array) {
            return latents.empty() ? model->encode(array) : latents;
        },
        [this](const tx::TxState& state) {
            emit stateChanged(static_cast<int>(state.phase), state.progress,
                              QString::fromStdString(state.message));
        },
        [this](const std::string& message) {
            emit errorOccurred(QString::fromStdString(message));
        });

    send_button_->setEnabled(false);
    set_picture_controls_enabled(false);
    cancel_button_->setEnabled(true);
    // The level is captured in tx_config above, so moving the slider now
    // would change the reading without changing the transmission.
    level_slider_->setEnabled(false);
    last_logged_phase_ = -1;
    running_.store(true);
    emit transmitStarted();

    thread_ = std::thread([this, picture, tx_config] {
        bool ok = false;
        try {
            ok = engine_->transmit(picture, tx_config);
        } catch (const std::exception& e) {
            emit errorOccurred(QString::fromUtf8(e.what()));
        }
        running_.store(false);
        emit sendFinished(ok);
    });
}

void TransmitPanel::cancel() {
    if (!engine_) return;
    engine_->cancel();
    status_->setText(tr("Cancelling..."));
}

void TransmitPanel::on_state(int phase, double progress, const QString& message) {
    const auto tx_phase = static_cast<tx::TxPhase>(phase);
    // This slot fires on every playback progress tick; the log gets a
    // line only when the phase changes. Keying and unkeying are the
    // lines the on-air PTT shakedown will want timestamps for.
    if (phase != last_logged_phase_) {
        last_logged_phase_ = phase;
        switch (tx_phase) {
            case tx::TxPhase::Keying:
            case tx::TxPhase::Sending:
            case tx::TxPhase::Unkeying:
            case tx::TxPhase::Done:
            case tx::TxPhase::Cancelled:
                app_->log_event("tx", log::Severity::Info,
                                message.isEmpty()
                                    ? QString::fromLatin1(tx::phase_name(tx_phase))
                                    : message);
                break;
            case tx::TxPhase::Failed:
                // The text arrives through on_error too; the phase line
                // marks *when* the sequence gave up.
                app_->log_event("tx", log::Severity::Error,
                                tr("transmit failed"));
                break;
            default:
                break;  // Idle/Encoding/Modulating: routine, not events
        }
    }
    // Failed carries the exception text as its message; the banner and
    // the log have it in full, and a single-line label given a long
    // error would force the send bar's minimum width out. The label is
    // the progress tier: short phase words only.
    status_->setText(message.isEmpty() || tx_phase == tx::TxPhase::Failed
                         ? QString::fromLatin1(tx::phase_name(tx_phase))
                         : message);
    if (tx_phase == tx::TxPhase::Sending) {
        progress_->setRange(0, 100);
        progress_->setValue(static_cast<int>(100.0 * progress));
    } else if (tx_phase == tx::TxPhase::Encoding ||
               tx_phase == tx::TxPhase::Modulating) {
        // Indeterminate: there is no useful fraction to report.
        progress_->setRange(0, 0);
    } else {
        progress_->setRange(0, 100);
    }
}

void TransmitPanel::on_error(const QString& message) {
    // Sticky (the banner) plus durable (the log): "PTT OFF FAILED --
    // unkey it manually" must survive the "Sent" that follows it on
    // the status label. The label itself gets nothing: it cannot wrap,
    // so a long error there inflates the send bar's minimum width --
    // rendered proof: a 700 px gui-shot request came back 1204 px wide
    // with the PTT message in the label.
    place_banner();
    banner_->show_error(message);
    app_->log_event("tx", log::Severity::Error, message);
}

void TransmitPanel::on_finished(bool ok) {
    if (thread_.joinable()) thread_.join();
    // Only re-armed if an edit was deferred while the send was
    // committed. Otherwise the composition is unchanged and still has
    // its optimized latents -- `take_result` consumes nothing -- so a
    // second send reuses them, and re-arming would spend CPU
    // reproducing what is already in hand and overwrite "Sent" a
    // second later.
    if (restart_after_send_) {
        restart_after_send_ = false;
        schedule_optimization();
    }
    send_button_->setEnabled(true);
    set_picture_controls_enabled(true);
    cancel_button_->setEnabled(false);
    level_slider_->setEnabled(true);
    progress_->setRange(0, 100);
    progress_->setValue(ok ? 100 : 0);
    if (ok) {
        status_->setText(tr("Sent"));
    } else if (static_cast<tx::TxPhase>(last_logged_phase_) ==
               tx::TxPhase::Failed) {
        // A failed send previously wrote no terminal status at all --
        // whatever text happened to be there stood. A cancelled send
        // also lands here with ok=false, and its "cancelled" text is
        // already correct, so only a genuine failure is relabelled.
        status_->setText(tr("Failed"));
    }
    emit transmitFinished();
}

}  // namespace sstvae::gui

#include "tx_panel.hpp"

#include <QColor>
#include <QColorDialog>
#include <QComboBox>
#include <QDoubleSpinBox>
#include <QDragEnterEvent>
#include <QDropEvent>
#include <QFileDialog>
#include <QFileInfo>
#include <QFormLayout>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QIcon>
#include <QLabel>
#include <QMessageBox>
#include <QMimeData>
#include <QPainter>
#include <QPixmap>
#include <QPlainTextEdit>
#include <QTextDocument>
#include <QProgressBar>
#include <QPushButton>
#include <QResizeEvent>
#include <QSlider>
#include <QSplitter>
#include <QStyle>
#include <QTimer>
#include <QUrl>
#include <QVBoxLayout>

#include <algorithm>
#include <cmath>
#include <exception>
#include <map>
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
#include "overlay_editor.hpp"
#include "style.hpp"
#include "settings/settings.hpp"

namespace sstvae::gui {

namespace {

const char* IMAGE_FILTER =
    "Images (*.png *.jpg *.jpeg *.webp *.bmp *.gif);;All files (*)";

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

    rebuild_optimizer();
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
    optimizer_ = std::make_unique<optimize::Speculative>(
        [model_path](const images::ImageArray& target) {
            // Resolved per run rather than cached: the artifact is only
            // fetched when the feature is actually used, and
            // `resolve_onnx` pins it to fp32 whatever the codec's
            // precision is.
            const std::string path = checkpoint::resolve_onnx(
                std::string(checkpoint::GRAD_PART), model_path);
            auto session = std::make_shared<codec::GradSession>(path, target);
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

void TransmitPanel::schedule_optimization() {
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
    connect(editor_, &OverlayEditor::documentChanged, this,
            &TransmitPanel::schedule_optimization);
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
    auto* add_text = new QPushButton(tr("Add &text"), overlay_box);
    connect(add_text, &QPushButton::clicked, this, [this] {
        const std::string& callsign = app_->config().callsign;
        editor_->add_text(callsign.empty() ? std::string("TEXT") : callsign);
    });
    add_rx_button_ = new QPushButton(tr("Add &last received"), overlay_box);
    // Nothing to insert until something has been received: the item it
    // adds resolves at render time, so before the first reception it
    // drew nothing and looked like a button that did not work.
    add_rx_button_->setEnabled(false);
    add_rx_button_->setToolTip(
        tr("Insert the last received image. Available once one has arrived."));
    connect(add_rx_button_, &QPushButton::clicked, editor_,
            &OverlayEditor::add_last_rx_inset);
    auto* add_image = new QPushButton(tr("Add i&mage..."), overlay_box);
    connect(add_image, &QPushButton::clicked, this, [this] {
        const QString path = QFileDialog::getOpenFileName(
            this, tr("Choose an inset image"),
            QString::fromStdString(app_->config().folders.transmit_dir),
            QString::fromLatin1(IMAGE_FILTER));
        if (!path.isEmpty()) editor_->add_image_inset(path.toStdString());
    });
    // **Remove is not here; it is in the "Selected item" box.** It is
    // the only control in this row that acts on the *selection* rather
    // than adding something, and the box below is where the selection
    // lives -- so it was both misfiled and, at 85 px, the button that
    // tipped this row onto a second line once the captions were spelled
    // out. Two rows here cost 30 px of picture height on *both* panes,
    // the strips being locked equal, and the received image is the one
    // that cannot be asked for again.
    for (QPushButton* button : {add_text, add_rx_button_, add_image}) {
        overlay_layout->addWidget(button);
    }
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
    auto* box = new QGroupBox(tr("Selected item"), parent);
    box->setEnabled(false);
    // Horizontal: a form stacks its rows, which under the canvas would
    // cost five rows of height. Side by side it is one.
    // Wrapping, for the same reason as the tool row: five fields on a
    // half-width pane is exactly where a single line gives up.
    //
    // **Each label and its control go in as one item**, through
    // `style::row`. A `FlowLayout` has no notion that two adjacent
    // items belong together, so adding them separately let a wrap fall
    // between them: measured at 1360 px wide, "Rotation" ended line one
    // and its spin box started line two, with "Size 0.010" in between.
    // At 900 px the same row wrapped cleanly, which is why it survived
    // -- it is only wrong at some widths.
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

    size_spin_ = new QDoubleSpinBox(box);
    size_spin_->setRange(0.01, 1.5);
    size_spin_->setSingleStep(0.01);
    size_spin_->setDecimals(3);
    connect(size_spin_, &QDoubleSpinBox::valueChanged, this, [this](double value) {
        auto* item = editing_item();
        if (item == nullptr) return;
        if (auto* text = std::get_if<overlay::TextItem>(item)) {
            text->size = value;
        } else if (auto* image = std::get_if<overlay::ImageItem>(item)) {
            image->width = value;
        }
        editor_->refresh_item();
    });
    size_spin_->setToolTip(
        tr("A fraction of the image, so it means the same at any window size."));
    form->addWidget(style::row(box, {new QLabel(tr("Size"), box), size_spin_}));

    rotation_spin_ = new QDoubleSpinBox(box);
    rotation_spin_->setRange(-180.0, 180.0);
    rotation_spin_->setSingleStep(1.0);
    connect(rotation_spin_, &QDoubleSpinBox::valueChanged, this,
            [this](double value) {
                auto* item = editing_item();
                if (item == nullptr) return;
                std::visit([value](auto& i) { i.rotation = value; }, *item);
                editor_->refresh_item();
            });
    rotation_spin_->setToolTip(tr("Degrees, clockwise."));
    form->addWidget(
        style::row(box, {new QLabel(tr("Rotation"), box), rotation_spin_}));

    color_button_ = new QPushButton(tr("Colour..."), box);
    connect(color_button_, &QPushButton::clicked, this, [this] {
        auto* item = editing_item();
        if (item == nullptr) return;
        auto* text = std::get_if<overlay::TextItem>(item);
        if (text == nullptr) return;
        const QColor color = QColorDialog::getColor(
            QColor(QString::fromStdString(text->color)), this);
        if (!color.isValid()) return;
        text->color = color.name().toStdString();
        set_color_swatch(color);
        editor_->refresh_item();
    });
    // **Given its swatch now, before anything is selected.**
    //
    // A QPushButton grows when it is handed an icon: measured 80x22
    // without and 80x24 with, on the same style. The swatch used to
    // arrive on the first *text* selection, so the button silently got
    // 2 px taller at that moment and stayed there -- and on a platform
    // where this button is the tallest thing on its line of the
    // wrapping row, that is 4 px on the whole control strip, which the
    // panes are then locked to. It cost a Windows CI failure that
    // reproduced nowhere else, because on Linux a taller sibling on the
    // same line absorbed it.
    //
    // An empty swatch is transparent and draws nothing, so this is
    // invisible; what it buys is a button whose metrics never move.
    // `test_tx_panel.cpp` asserts that directly, which is a check every
    // platform can run.
    set_color_swatch(QColor());
    form->addWidget(style::row(box, {new QLabel(tr("Colour"), box), color_button_}));

    // Removing acts on the selection, so it belongs with the selection's
    // own controls rather than among the buttons that add things. The
    // box is disabled with nothing selected, which is exactly when
    // Remove has nothing to do -- something the tool row could not say
    // about it.
    auto* remove = new QPushButton(tr("&Remove"), box);
    remove->setToolTip(tr("Remove the selected item (or press Delete)."));
    connect(remove, &QPushButton::clicked, editor_,
            &OverlayEditor::remove_selected);
    form->addWidget(remove);
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

void TransmitPanel::set_last_rx_image(const images::Picture& image) {
    editor_->set_last_rx(image);
    // Only now is there anything for the button to insert. Before the
    // first reception it added an item that rendered as nothing, which
    // reads as a broken button rather than as "not yet".
    add_rx_button_->setEnabled(true);
    add_rx_button_->setToolTip(QString());
}

void TransmitPanel::set_color_swatch(const QColor& color) {
    // The button said "Colour..." and nothing else, so the current
    // colour was invisible -- the one thing a colour control has to
    // show. A filled square on the button, drawn at the icon size the
    // style asks for so it matches the platform's other buttons.
    // Dragging a text item emits selectionChanged on every mouse move,
    // so without this guard the whole rebuild -- parse, allocate, paint,
    // setIcon, and the layout invalidation setIcon triggers -- runs at
    // mouse-move rate on the app's most latency-sensitive path.
    //
    // **`swatch_set_`, not just the colour.** An invalid QColor equals
    // an invalid QColor, so before this flag existed the button could
    // not be given its *first* swatch at construction -- the guard ate
    // the call -- and it therefore acquired one only when a text item
    // was first selected. See `build_properties` for why that mattered.
    if (swatch_set_ && color == swatch_color_) return;
    swatch_set_ = true;
    swatch_color_ = color;

    const int size = style()->pixelMetric(QStyle::PM_SmallIconSize);
    // Device pixels, like the waterfall's backing image: a logical-sized
    // pixmap is upscaled on a HiDPI screen, giving a soft square with a
    // half-resolution border.
    const qreal dpr = devicePixelRatioF();
    QPixmap swatch(static_cast<int>(std::lround(size * dpr)),
                   static_cast<int>(std::lround(size * dpr)));
    swatch.setDevicePixelRatio(dpr);
    swatch.fill(color.isValid() ? color : Qt::transparent);
    if (color.isValid()) {
        QPainter painter(&swatch);
        painter.setPen(palette().color(QPalette::WindowText));
        painter.drawRect(0, 0, size - 1, size - 1);
    }
    color_button_->setIcon(QIcon(swatch));
}

// --- property editing -------------------------------------------------------

overlay::Item* TransmitPanel::editing_item() {
    // Null while the widgets are being filled from an item, so their
    // change signals do not write the value straight back.
    if (loading_properties_) return nullptr;
    return editor_->selected_item();
}

void TransmitPanel::on_selection(overlay::Item* item) {
    // **Disabled, never hidden** -- see `build_ui`, which this used to
    // contradict outright while both sites carried a comment asserting
    // the opposite rule.
    //
    // Two things went wrong when it hid. The row is ~90 px, so the
    // canvas jumped under the pointer on every select and deselect,
    // which is the one thing a composing surface must not do. And
    // `PaneContainer::equalise_strips` runs only from
    // `set_control_strips` and a resize -- nothing re-runs it when a
    // strip's *content* changes height -- so from the first click the
    // two strips held minimums computed for a layout that no longer
    // existed and the two pictures silently stopped matching until the
    // window was resized. `test_pane_container.cpp` cannot see that:
    // its stand-in panes have static strips. `test_tx_panel.cpp` is the
    // guard.
    properties_->setEnabled(item != nullptr);
    if (item == nullptr) {
        // Otherwise the last item's colour stays painted on a disabled
        // button, describing a selection that no longer exists.
        set_color_swatch(QColor());
        return;
    }

    const bool is_text = std::holds_alternative<overlay::TextItem>(*item);
    loading_properties_ = true;
    text_edit_->setEnabled(is_text);
    align_combo_->setEnabled(is_text);
    color_button_->setEnabled(is_text);
    if (is_text) {
        const overlay::TextItem& text = std::get<overlay::TextItem>(*item);
        text_edit_->setPlainText(QString::fromStdString(text.text));
        align_combo_->setCurrentIndex(std::max(
            0, align_combo_->findData(QString::fromStdString(text.align))));
        size_spin_->setValue(text.size);
        set_color_swatch(QColor(QString::fromStdString(text.color)));
    } else {
        text_edit_->setPlainText(QString());
        size_spin_->setValue(std::get<overlay::ImageItem>(*item).width);
        set_color_swatch(QColor());  // no colour on an image item
    }
    rotation_spin_->setValue(std::visit([](const auto& i) { return i.rotation; },
                                        *item));
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

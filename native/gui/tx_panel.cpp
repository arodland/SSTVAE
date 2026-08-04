#include "tx_panel.hpp"

#include <QColorDialog>
#include <QComboBox>
#include <QDoubleSpinBox>
#include <QFileDialog>
#include <QFileInfo>
#include <QFormLayout>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QMessageBox>
#include <QPlainTextEdit>
#include <QProgressBar>
#include <QPushButton>
#include <QSlider>
#include <QSplitter>
#include <QTimer>
#include <QVBoxLayout>

#include <algorithm>
#include <cmath>
#include <exception>
#include <map>
#include <vector>

#include "app_state.hpp"
#include "audio/qt/qtaudio.hpp"
#include "checkpoint/checkpoint.hpp"
#include "codec/codec.hpp"
#include "codec/grad_session.hpp"
#include "config.hpp"
#include "images/images.hpp"
#include "overlay_editor.hpp"
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
        status_->setText(tr("Refining picture... %1 on air").arg(gain));
    } else if (st.finished && awaiting_optimizer_) {
        status_->setText(tr("Refining picture... finishing"));
    } else if (st.finished && st.progress.step > 0) {
        // Whatever ended it -- plateau, either budget, or Send cutting
        // it short -- the gain it did reach stays on screen.
        status_->setText(tr("Picture refined: %1 on air").arg(gain));
    } else if (st.finished) {
        // Finished without a single measured step: the artifact was
        // missing or the encode failed, and the encoder's own latents
        // are what will be sent.
        status_->setText(tr("Sending unrefined"));
    }
}

void TransmitPanel::build_ui() {
    auto* layout = new QVBoxLayout(this);
    auto* splitter = new QSplitter(Qt::Horizontal, this);

    editor_ = new OverlayEditor(splitter);
    connect(editor_, &OverlayEditor::selectionChanged, this,
            &TransmitPanel::on_selection);
    connect(editor_, &OverlayEditor::documentChanged, this,
            &TransmitPanel::schedule_optimization);
    splitter->addWidget(editor_);
    splitter->addWidget(build_side_panel());
    splitter->setStretchFactor(0, 3);
    splitter->setStretchFactor(1, 1);
    layout->addWidget(splitter, 1);
    layout->addWidget(build_send_bar());
}

QWidget* TransmitPanel::build_side_panel() {
    auto* panel = new QWidget(this);
    auto* column = new QVBoxLayout(panel);

    auto* source = new QGroupBox(tr("Picture"), panel);
    auto* source_layout = new QVBoxLayout(source);
    choose_button_ = new QPushButton(tr("Choose image..."), source);
    connect(choose_button_, &QPushButton::clicked, this,
            &TransmitPanel::choose_image);
    image_label_ = new QLabel(tr("No image selected"), source);
    image_label_->setWordWrap(true);
    source_layout->addWidget(choose_button_);
    source_layout->addWidget(image_label_);
    column->addWidget(source);

    auto* overlay_box = new QGroupBox(tr("Overlay"), panel);
    auto* overlay_layout = new QVBoxLayout(overlay_box);
    auto* add_text = new QPushButton(tr("Add text"), overlay_box);
    connect(add_text, &QPushButton::clicked, this, [this] {
        const std::string& callsign = app_->config().callsign;
        editor_->add_text(callsign.empty() ? std::string("TEXT") : callsign);
    });
    add_rx_button_ = new QPushButton(tr("Add last received image"), overlay_box);
    connect(add_rx_button_, &QPushButton::clicked, editor_,
            &OverlayEditor::add_last_rx_inset);
    auto* add_image = new QPushButton(tr("Add image from file..."), overlay_box);
    connect(add_image, &QPushButton::clicked, this, [this] {
        const QString path = QFileDialog::getOpenFileName(
            this, tr("Choose an inset image"),
            QString::fromStdString(app_->config().folders.transmit_dir),
            QString::fromLatin1(IMAGE_FILTER));
        if (!path.isEmpty()) editor_->add_image_inset(path.toStdString());
    });
    auto* remove = new QPushButton(tr("Remove selected"), overlay_box);
    connect(remove, &QPushButton::clicked, editor_,
            &OverlayEditor::remove_selected);
    for (QPushButton* button : {add_text, add_rx_button_, add_image, remove}) {
        overlay_layout->addWidget(button);
    }
    column->addWidget(overlay_box);

    properties_ = build_properties(panel);
    column->addWidget(properties_);
    column->addStretch(1);
    return panel;
}

QGroupBox* TransmitPanel::build_properties(QWidget* parent) {
    auto* box = new QGroupBox(tr("Selected item"), parent);
    box->setEnabled(false);
    auto* form = new QFormLayout(box);

    // Multi-line: a station's callsign, grid and name belong to one
    // item, not three stacked by hand. Enter inserts a newline, so Tab
    // has to be what leaves the field.
    text_edit_ = new QPlainTextEdit(box);
    text_edit_->setTabChangesFocus(true);
    text_edit_->setFixedHeight(80);
    connect(text_edit_, &QPlainTextEdit::textChanged, this, [this] {
        if (auto* item = editing_item()) {
            if (auto* text = std::get_if<overlay::TextItem>(item)) {
                text->text = text_edit_->toPlainText().toStdString();
                editor_->refresh_item();
            }
        }
    });
    form->addRow(tr("Text"), text_edit_);

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
    form->addRow(tr("Align"), align_combo_);

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
    form->addRow(tr("Size"), size_spin_);

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
    form->addRow(tr("Rotation"), rotation_spin_);

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
        editor_->refresh_item();
    });
    form->addRow(tr("Colour"), color_button_);
    return box;
}

QWidget* TransmitPanel::build_send_bar() {
    auto* bar = new QWidget(this);
    auto* layout = new QHBoxLayout(bar);
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
    level_slider_->setFixedWidth(140);
    level_slider_->setToolTip(
        tr("Output level, dB relative to full scale.\n\n"
           "Set it so the radio's ALC barely moves. The waveform is already "
           "conditioned for a ~4 dB envelope peak; driving it into ALC "
           "compression will spread it across the band."));
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

    send_button_ = new QPushButton(tr("Send"), bar);
    connect(send_button_, &QPushButton::clicked, this, &TransmitPanel::send);
    cancel_button_ = new QPushButton(tr("Cancel"), bar);
    connect(cancel_button_, &QPushButton::clicked, this, &TransmitPanel::cancel);
    cancel_button_->setEnabled(false);

    progress_ = new QProgressBar(bar);
    progress_->setRange(0, 100);
    status_ = new QLabel(tr("Ready"), bar);

    layout->addWidget(new QLabel(tr("Mode:"), bar));
    layout->addWidget(mode_combo_);
    layout->addWidget(new QLabel(tr("Level:"), bar));
    layout->addWidget(level_slider_);
    layout->addWidget(level_label_);
    layout->addWidget(send_button_);
    layout->addWidget(cancel_button_);
    layout->addWidget(progress_, 1);
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
    try {
        // Framed to the transmit size here, not at send time: the
        // overlay's coordinates are fractions of the canvas, so the
        // operator has to be composing against the frame that will
        // actually go out.
        editor_->set_base_image(images::fit(images::load(path.toStdString())));
    } catch (const std::exception& e) {
        QMessageBox::critical(this, tr("Could not open image"),
                              QString::fromUtf8(e.what()));
        return;
    }
    image_label_->setText(QFileInfo(path).fileName());
    app_->config().folders.transmit_dir =
        QFileInfo(path).absolutePath().toStdString();
    app_->save_config();
}

void TransmitPanel::set_last_rx_image(const images::Picture& image) {
    editor_->set_last_rx(image);
}

// --- property editing -------------------------------------------------------

overlay::Item* TransmitPanel::editing_item() {
    // Null while the widgets are being filled from an item, so their
    // change signals do not write the value straight back.
    if (loading_properties_) return nullptr;
    return editor_->selected_item();
}

void TransmitPanel::on_selection(overlay::Item* item) {
    properties_->setEnabled(item != nullptr);
    if (item == nullptr) return;

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
    } else {
        text_edit_->setPlainText(QString());
        size_spin_->setValue(std::get<overlay::ImageItem>(*item).width);
    }
    rotation_spin_->setValue(std::visit([](const auto& i) { return i.rotation; },
                                        *item));
    loading_properties_ = false;
}

// --- transmitting -----------------------------------------------------------

bool TransmitPanel::transmitting() const { return running_.load(); }

void TransmitPanel::send() {
    if (transmitting()) return;

    const std::optional<images::Picture> image = editor_->composed_image();
    if (!image) {
        QMessageBox::information(this, tr("No picture"),
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
    if (thread_.joinable()) thread_.join();

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
            status_->setText(tr("Refining picture..."));
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
    cancel_button_->setEnabled(true);
    // The level is captured in tx_config above, so moving the slider now
    // would change the reading without changing the transmission.
    level_slider_->setEnabled(false);
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
    status_->setText(message.isEmpty()
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

void TransmitPanel::on_error(const QString& message) { status_->setText(message); }

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
    cancel_button_->setEnabled(false);
    level_slider_->setEnabled(true);
    progress_->setRange(0, 100);
    progress_->setValue(ok ? 100 : 0);
    if (ok) status_->setText(tr("Sent"));
    emit transmitFinished();
}

}  // namespace sstvae::gui

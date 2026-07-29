#include "rx_panel.hpp"

#include <QCheckBox>
#include <QFileDialog>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QImage>
#include <QLabel>
#include <QMessageBox>
#include <QProgressBar>
#include <QPushButton>
#include <QResizeEvent>
#include <QSizePolicy>
#include <QSplitter>
#include <QTimer>
#include <QVBoxLayout>

#include <cmath>
#include <exception>
#include <filesystem>
#include <utility>

#include "app_state.hpp"
#include "audio/qt/qtaudio.hpp"
#include "audio/wavio.hpp"
#include "codec/codec.hpp"
#include "images/images.hpp"
#include "settings/settings.hpp"
#include "waterfall.hpp"

namespace sstvae::gui {

namespace {

namespace fs = std::filesystem;

QPixmap to_pixmap(const images::Picture& picture) {
    if (picture.empty()) return QPixmap();
    const QImage view(picture.rgb.data(), picture.width, picture.height,
                      picture.width * 3, QImage::Format_RGB888);
    return QPixmap::fromImage(view.copy());
}

// Never overwrite: two receptions can finish in the same second with the
// same callsign and frequency.
fs::path unique_path(fs::path path) {
    if (!fs::exists(path)) return path;
    const fs::path parent = path.parent_path();
    const std::string stem = path.stem().string();
    const std::string ext = path.extension().string();
    for (int n = 2; n < 1000; ++n) {
        const fs::path candidate =
            parent / (stem + "_" + std::to_string(n) + ext);
        if (!fs::exists(candidate)) return candidate;
    }
    return path;
}

}  // namespace

ReceivePanel::ReceivePanel(AppState* state, QWidget* parent)
    : QWidget(parent), app_(state), shared_(std::make_unique<rx::SharedState>()) {
    build_ui();

    // Queued by construction: both are emitted from the decode thread,
    // and Qt delivers a cross-thread signal on the receiver's thread.
    connect(this, &ReceivePanel::receptionFinished, this,
            &ReceivePanel::on_reception, Qt::QueuedConnection);
    connect(this, &ReceivePanel::errorOccurred, this, &ReceivePanel::on_error,
            Qt::QueuedConnection);

    auto* timer = new QTimer(this);
    connect(timer, &QTimer::timeout, this, &ReceivePanel::refresh_status);
    timer->start(500);
}

ReceivePanel::~ReceivePanel() { stop(); }

void ReceivePanel::build_ui() {
    // Picture on the left, waterfall down the right -- the same shape as
    // the transmit panel. The pictures are 4:3, so on the wide monitor
    // most people have, stacking the waterfall on top would leave the
    // sides empty and squeeze the thing you actually want to look at.
    auto* layout = new QVBoxLayout(this);
    auto* splitter = new QSplitter(Qt::Horizontal, this);

    auto* left = new QWidget(splitter);
    auto* left_layout = new QVBoxLayout(left);
    left_layout->setContentsMargins(0, 0, 0, 0);

    auto* preview_box = new QGroupBox(tr("Picture"), left);
    auto* preview_layout = new QVBoxLayout(preview_box);
    preview_ = new QLabel(tr("Nothing received yet"), preview_box);
    preview_->setAlignment(Qt::AlignCenter);
    preview_->setMinimumHeight(240);
    preview_->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    preview_->setStyleSheet(QStringLiteral("background:#202024; color:#888;"));
    preview_layout->addWidget(preview_);
    left_layout->addWidget(preview_box, 1);

    status_ = new QLabel(tr("Stopped"), left);
    left_layout->addWidget(status_);

    progress_ = new QProgressBar(left);
    progress_->setRange(0, 100);
    left_layout->addWidget(progress_);

    waterfall_ = new Waterfall(splitter);
    splitter->addWidget(left);
    splitter->addWidget(waterfall_);
    splitter->setStretchFactor(0, 3);
    splitter->setStretchFactor(1, 1);
    layout->addWidget(splitter, 1);

    auto* controls = new QHBoxLayout();
    start_button_ = new QPushButton(tr("Start receiving"), this);
    connect(start_button_, &QPushButton::clicked, this, &ReceivePanel::start);
    stop_button_ = new QPushButton(tr("Stop"), this);
    connect(stop_button_, &QPushButton::clicked, this, &ReceivePanel::stop);
    stop_button_->setEnabled(false);
    save_button_ = new QPushButton(tr("Save image"), this);
    connect(save_button_, &QPushButton::clicked, this, &ReceivePanel::save_current);
    save_button_->setEnabled(false);
    autosave_ = new QCheckBox(tr("Autosave"), this);
    autosave_->setChecked(app_->config().receive.autosave);
    connect(autosave_, &QCheckBox::toggled, this,
            &ReceivePanel::on_autosave_toggled);

    controls->addWidget(start_button_);
    controls->addWidget(stop_button_);
    controls->addWidget(save_button_);
    controls->addWidget(autosave_);
    controls->addStretch(1);
    layout->addLayout(controls);
}

bool ReceivePanel::listening() const { return running_.load(); }

bool ReceivePanel::start() {
    if (listening()) return true;

    codec::OnnxCodec* model = app_->model();
    if (model == nullptr) {
        QMessageBox::warning(
            this, tr("Model still loading"),
            tr("The codec checkpoint is still loading. Try again in a moment."));
        return false;
    }

    const settings::Config& config = app_->config();
    ring_ = std::make_shared<rx::RingBuffer>(config.receive.buffer_seconds);
    waterfall_->set_ring(ring_);

    try {
        stream_ = std::make_unique<audio::qt::InputStream>(
            config.audio.input_device, *ring_, config::FS,
            [this](const std::string& message) {
                emit errorOccurred(QString::fromStdString(message));
            });
    } catch (const std::exception& e) {
        waterfall_->set_ring(nullptr);
        ring_.reset();
        QMessageBox::critical(
            this, tr("Could not open the input device"),
            tr("%1\n\nCheck the input device in Settings.")
                .arg(QString::fromUtf8(e.what())));
        return false;
    }

    rx::RxConfig rx_config;
    rx_config.out_dir = config.folders.receive_dir;
    rx_config.poll_interval = config.receive.poll_interval;
    rx_config.end_grace = config.receive.end_grace;
    rx_config.size = rx::parse_size(config.receive.save_size);
    rx_config.once = false;
    rx_config.blind_search_seconds = config.receive.blind_search_seconds;

    // The decoder seam: the loop lives in sstvae_core and never links
    // onnxruntime, so the codec arrives as a function.
    auto decoder = [model](std::span<const double> latents,
                           std::span<const double> weights) {
        return model->decode(std::vector<double>(latents.begin(), latents.end()),
                             std::vector<double>(weights.begin(), weights.end()));
    };
    auto sink = [this](const rx::Reception& reception) {
        return handle_reception(reception);
    };

    stop_flag_.clear();
    const bool low_cpu = config.receive.low_cpu;
    const int device_rate = stream_->device_rate();

    running_.store(true);
    thread_ = std::thread([this, rx_config, decoder, sink, low_cpu] {
        try {
            if (low_cpu) {
                rx::decode_loop_low_cpu(*ring_, decoder, *shared_, rx_config,
                                        stop_flag_, sink);
            } else {
                rx::decode_loop(*ring_, decoder, *shared_, rx_config, stop_flag_,
                                sink);
            }
        } catch (const std::exception& e) {
            // A crashed loop must not vanish silently.
            emit errorOccurred(
                tr("receive loop stopped: %1").arg(QString::fromUtf8(e.what())));
        }
        running_.store(false);
    });

    start_button_->setEnabled(false);
    stop_button_->setEnabled(true);
    status_->setText(tr("Listening at %1 Hz").arg(device_rate));
    emit listeningChanged(true);
    return true;
}

void ReceivePanel::stop() {
    stop_flag_.set();
    // The stream first: it is what feeds the loop, and stopping it means
    // the loop's next poll sees no new audio rather than racing us.
    stream_.reset();
    if (thread_.joinable()) thread_.join();
    running_.store(false);

    if (waterfall_ != nullptr) waterfall_->set_ring(nullptr);
    ring_.reset();

    if (start_button_ != nullptr) {
        start_button_->setEnabled(true);
        stop_button_->setEnabled(false);
        status_->setText(tr("Stopped"));
    }
    emit listeningChanged(false);
}

void ReceivePanel::suspend_for_transmit() {
    if (!listening()) return;
    suspended_for_tx_ = true;
    stop();
    status_->setText(tr("Paused -- transmitting"));
}

void ReceivePanel::resume_after_transmit() {
    if (!suspended_for_tx_) return;
    suspended_for_tx_ = false;
    // start() allocates a fresh ring buffer, so the tail of our own
    // transmission is dropped rather than decoded back as a reception.
    start();
    waterfall_->clear();
}

// --- the sink, on the decode thread -----------------------------------------

std::optional<std::string> ReceivePanel::handle_reception(
    const rx::Reception& reception) {
    std::optional<std::string> saved;
    // Read from the config now rather than captured at start, so the
    // checkbox takes effect on the next reception without a restart.
    if (app_->config().receive.autosave) {
        try {
            saved = save_reception(reception);
        } catch (const std::exception& e) {
            emit errorOccurred(tr("could not save received image: %1")
                                   .arg(QString::fromUtf8(e.what())));
        }
    }
    {
        const std::lock_guard<std::mutex> lock(reception_mutex_);
        last_reception_ = reception;
    }
    emit receptionFinished(saved ? QString::fromStdString(*saved) : QString());
    return saved;
}

std::optional<std::string> ReceivePanel::save_reception(
    const rx::Reception& reception) {
    const settings::Config& config = app_->config();
    const fs::path out_dir(config.folders.receive_dir);
    fs::create_directories(out_dir);

    settings::FilenameFields fields;
    fields.callsign = reception.callsign;
    fields.mode = reception.mode_name.value_or(std::string());
    if (const std::optional<double> hz = app_->current_frequency_hz()) {
        fields.freq_hz = *hz;
    }
    const std::string stem =
        settings::format_filename(config.receive.filename_template, fields);
    const fs::path path = unique_path(out_dir / (stem + ".png"));

    images::Picture image = reception.image;
    if (const auto size = rx::parse_size(config.receive.save_size)) {
        image = images::resize(image, size->first, size->second);
    }
    images::save_png(image, path.string());

    if (config.receive.save_audio) save_audio_beside(path.string());
    return path.string();
}

void ReceivePanel::save_audio_beside(const std::string& image_path) {
    // Deliberately the *whole* ring rather than a trimmed reception: the
    // point is to be able to re-run the real decoder over exactly what
    // the sound card delivered, and a trim would beg the question by
    // assuming we already know where the transmission was.
    const std::shared_ptr<rx::RingBuffer> ring = ring_;
    if (!ring) return;
    try {
        std::uint64_t total = 0;
        std::vector<double> samples = ring->snapshot(&total);
        if (total == 0) return;
        if (total < samples.size()) samples.resize(total);
        audio::write_wav_float(fs::path(image_path).replace_extension(".wav").string(),
                               samples);
    } catch (const std::exception& e) {
        // A diagnostic must never break receiving.
        emit errorOccurred(
            tr("could not save captured audio: %1").arg(QString::fromUtf8(e.what())));
    }
}

// --- the GUI thread ---------------------------------------------------------

void ReceivePanel::refresh_status() {
    if (!listening()) return;
    const rx::Progress progress = shared_->get();

    QString text;
    if (progress.status == rx::Status::Listening) {
        text = tr("Listening... (%1s captured)")
                   .arg(progress.seconds_captured, 0, 'f', 0);
    } else if (progress.status == rx::Status::Receiving) {
        if (progress.n_frames_expected) {
            text = tr("Receiving mode %1: frame %2/%3 (%4%)")
                       .arg(QString::fromStdString(
                           progress.mode_name.value_or(std::string())))
                       .arg(progress.frames_received.value_or(0))
                       .arg(*progress.n_frames_expected)
                       .arg(100.0 * progress.progress_frac, 0, 'f', 0);
        } else {
            text = tr("Receiving (blind sync): %1% of latents")
                       .arg(100.0 * progress.progress_frac, 0, 'f', 0);
        }
        text += QString::fromStdString(rx::fmt_snr(progress.snr_db));
        if (!progress.callsign.empty()) {
            text += tr("  de %1").arg(QString::fromStdString(progress.callsign));
        }
    } else {
        text = tr("Complete%1").arg(QString::fromStdString(rx::fmt_snr(progress.snr_db)));
        if (last_saved_path_) {
            text += tr(" -- saved %1")
                        .arg(QString::fromStdString(
                            fs::path(*last_saved_path_).filename().string()));
        }
    }

    status_->setText(text);
    progress_->setValue(static_cast<int>(100.0 * progress.progress_frac));

    if (progress.image && progress.image.get() != shown_) {
        shown_ = progress.image.get();
        set_displayed(*progress.image, progress.callsign, progress.mode_name);
    }
}

void ReceivePanel::on_reception(const QString& saved_path) {
    std::optional<rx::Reception> reception;
    {
        const std::lock_guard<std::mutex> lock(reception_mutex_);
        reception = std::move(last_reception_);
        last_reception_.reset();
    }
    if (!reception) return;

    last_saved_path_ =
        saved_path.isEmpty() ? std::nullopt
                             : std::optional<std::string>(saved_path.toStdString());
    set_displayed(reception->image, reception->callsign, reception->mode_name);
    emit imageReceived(reception->image);
    if (!saved_path.isEmpty()) emit receptionSaved(saved_path);
}

void ReceivePanel::on_error(const QString& message) { status_->setText(message); }

void ReceivePanel::on_autosave_toggled(bool on) {
    app_->config().receive.autosave = on;
    app_->save_config();
}

void ReceivePanel::show_image(const images::Picture& image) {
    preview_pixmap_ = to_pixmap(image);
    if (preview_pixmap_.isNull()) return;
    preview_->setPixmap(preview_pixmap_.scaled(preview_->size(), Qt::KeepAspectRatio,
                                               Qt::SmoothTransformation));
}

void ReceivePanel::set_displayed(const images::Picture& image,
                                 const std::string& callsign,
                                 const std::optional<std::string>& mode_name) {
    displayed_ = image;
    displayed_callsign_ = callsign;
    displayed_mode_ = mode_name;
    show_image(image);
    save_button_->setEnabled(true);
}

void ReceivePanel::save_current() {
    // Whatever is in the preview, complete or not.
    if (!displayed_) return;
    const settings::Config& config = app_->config();

    settings::FilenameFields fields;
    fields.callsign = displayed_callsign_;
    fields.mode = displayed_mode_.value_or(std::string());
    if (const std::optional<double> hz = app_->current_frequency_hz()) {
        fields.freq_hz = *hz;
    }
    const std::string stem =
        settings::format_filename(config.receive.filename_template, fields);

    const fs::path out_dir(config.folders.receive_dir);
    std::error_code ignored;
    fs::create_directories(out_dir, ignored);
    const QString suggested =
        QString::fromStdString((out_dir / (stem + ".png")).string());

    const QString path = QFileDialog::getSaveFileName(
        this, tr("Save received image"), suggested, tr("Images (*.png *.jpg)"));
    if (path.isEmpty()) return;

    try {
        images::save_png(*displayed_, path.toStdString());
    } catch (const std::exception& e) {
        QMessageBox::critical(this, tr("Could not save"),
                              QString::fromUtf8(e.what()));
        return;
    }
    last_saved_path_ = path.toStdString();
    emit receptionSaved(path);
}

void ReceivePanel::resizeEvent(QResizeEvent* event) {
    QWidget::resizeEvent(event);
    if (!preview_pixmap_.isNull()) {
        preview_->setPixmap(preview_pixmap_.scaled(
            preview_->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
    }
}

}  // namespace sstvae::gui

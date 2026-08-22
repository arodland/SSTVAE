#include "rx_panel.hpp"

#include <QCheckBox>
#include <QColor>
#include <QDesktopServices>
#include <QSignalBlocker>
#include <QFileDialog>
#include <QFont>
#include <QFrame>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QMessageBox>
#include <QPalette>
#include <QProgressBar>
#include <QPushButton>
#include <QResizeEvent>
#include <QSizePolicy>
#include <QSplitter>
#include <QTime>
#include <QTimer>
#include <QUrl>
#include <QVBoxLayout>

#include <algorithm>
#include <array>
#include <cmath>
#include <exception>
#include <filesystem>
#include <utility>

#include "app_state.hpp"
#include "audio/qt/qtaudio.hpp"
#include "audio/wavio.hpp"
#include "banner.hpp"
#include "codec/codec.hpp"
#include "images/images.hpp"
#include "flow_layout.hpp"
#include "picture_box.hpp"
#include "style.hpp"
#include "settings/settings.hpp"
#include "waterfall.hpp"

namespace sstvae::gui {

namespace {

namespace fs = std::filesystem;

// One field separator, shared by the live status line and the
// last-reception card. They describe the same reception a second apart
// and used to punctuate it differently -- the status line with a double
// hyphen, the card with a single one.
const QString SEP = QStringLiteral("  ·  ");

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

// SNR for display, with a placeholder when there is none.
//
// `rx::fmt_snr` returns an empty string for NaN, which is right for it
// -- it mirrors Python's and feeds the CLI. But on screen an absent
// field is indistinguishable from a field that was never going to be
// there: the operator cannot tell "no SNR estimate yet" from "this
// build does not show SNR". The engine formats; the panel decides how
// to show absence.
QString snr_text(double snr_db) {
    const QString formatted = QString::fromStdString(rx::fmt_snr(snr_db));
    return formatted.isEmpty() ? QStringLiteral("  SNR --") : formatted;
}

// A filled chip, not coloured text -- the same argument as the PTT lamp
// (main_window.cpp): a lock state is exactly the kind of thing that
// must read at a glance, and coloured text on the panel's own
// background is a contrast problem waiting for a dark theme. Locked is
// `style::color::ok()`/`on_ok()`, the same pair the waterfall's level
// meter already uses for "healthy". Not locked deliberately reuses the
// palette's own Button/ButtonText pair rather than a second hand-picked
// grey: every theme already keeps that pairing legible, and it is what
// makes the chip read as an inactive control without being a disabled
// one (`style::dim` is for text, not for a filled surface).
void set_lock_chip(QLabel* chip, bool locked) {
    QPalette pal = chip->palette();
    if (locked) {
        pal.setColor(QPalette::Window, style::color::ok());
        pal.setColor(QPalette::WindowText, style::color::on_ok());
    } else {
        pal.setColor(QPalette::Window, pal.color(QPalette::Button));
        pal.setColor(QPalette::WindowText, pal.color(QPalette::ButtonText));
    }
    chip->setPalette(pal);
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
    // Picture, then a control strip -- and nothing else. The waterfall
    // used to be at the top of this pane in its own vertical splitter;
    // it now spans the whole window and belongs to `MainWindow`, which
    // drives it through `attach_waterfall`.
    //
    // The strip is the unit whose height the container matches against
    // the transmit pane's, so everything below the picture has to be
    // inside it. The picture then takes the rest, which is the same
    // number on both sides.
    auto* layout = new QVBoxLayout(this);

    // The error tier: sticky, dismissable, and never overwritten by the
    // 500 ms status refresh below -- which is exactly what happened to
    // every receive error before (visible for at most half a second).
    // **Over the picture, not above it.** In the layout it displaced
    // everything below it -- so an error in one pane pushed that pane's
    // picture down and the two stopped lining up, which is precisely
    // what the equal-size work exists to prevent. Floating it costs no
    // layout at all: it appears, it is dismissed, and nothing moves.
    banner_ = new ErrorBanner(this);
    banner_->hide();

    // No inner "Picture" box: the pane itself is titled, and a group
    // box between the layout and the picture would swallow the room the
    // picture is supposed to be taking.
    preview_ = new PictureBox(tr("Nothing received yet"), this);
    // Stretch 1 and no trailing spacer: the picture *is* what the spare
    // room is for. Everything else lives in the strip below, whose
    // height is fixed by agreement with the transmit pane.
    layout->addWidget(preview_);

    strip_ = new QWidget(this);
    auto* lower_layout = new QVBoxLayout(strip_);
    lower_layout->setContentsMargins(0, 0, 0, 0);
    lower_layout->setSpacing(4);
    QWidget* lower = strip_;

    // A progress-tier surface must not set a width floor. Its longest
    // line ("Receiving mode C: frame 220/220 ... de KD8XYZ") is ~400 px,
    // and both panes are locked to one width, so every such floor is
    // paid twice. Errors go to the banner, so what gives way here is
    // progress text that is rebuilt twice a second anyway.
    //
    // `ElidingLabel`, because giving way must not mean being cut off
    // mid-word: a truncated "de KD8X" reads as a callsign that did not
    // decode. See the same argument in `settings_dialog.cpp`'s scroll
    // areas.
    status_ = new style::ElidingLabel(tr("Stopped"), lower);
    status_->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Preferred);
    lower_layout->addWidget(status_);

    // The diversity lock chips: which of the two receivers is currently
    // contributing to the combine. Hidden until a diversity session
    // starts (set_lock_chip below never runs otherwise, so these stay in
    // their "not locked" look, which would be misleading to show for an
    // ordinary single-receiver session).
    auto* lock_row = new QHBoxLayout();
    lock_row->setContentsMargins(0, 0, 0, 0);
    auto make_lock_chip = [this](const QString& text) {
        auto* chip = new QLabel(text, this);
        QFont f = chip->font();
        f.setBold(true);
        chip->setFont(f);
        chip->setAutoFillBackground(true);
        chip->setAlignment(Qt::AlignCenter);
        chip->setContentsMargins(6, 1, 6, 1);
        // A boundary, not just a fill: measured against this platform's
        // own light theme, the "not locked" state's Button colour sits
        // only ~20 levels from Window -- filled but not obviously a
        // chip. `ok()`/`on_ok()` is high-contrast by construction and
        // needs no help; a frame is what keeps the *other* state
        // reading as a chip rather than as bold text with an
        // almost-invisible box.
        chip->setFrameShape(QFrame::Box);
        chip->setFrameShadow(QFrame::Plain);
        chip->setLineWidth(1);
        chip->hide();
        return chip;
    };
    branch_a_chip_ = make_lock_chip(tr("Primary"));
    branch_b_chip_ = make_lock_chip(tr("Secondary"));
    set_lock_chip(branch_a_chip_, false);
    set_lock_chip(branch_b_chip_, false);
    lock_row->addWidget(branch_a_chip_);
    lock_row->addWidget(branch_b_chip_);
    lock_row->addStretch(1);
    lower_layout->addLayout(lock_row);

    progress_ = new QProgressBar(lower);
    progress_->setRange(0, 100);
    lower_layout->addWidget(progress_);

    // The last completed reception's details, which nothing kept
    // before: the engine wipes mode, callsign and SNR from its shared
    // state two seconds after a reception finishes, so the "Complete
    // SNR x" line erased itself while the operator was still looking at
    // the picture it described.
    last_card_ = new QLabel(lower);
    last_card_->setTextFormat(Qt::PlainText);
    last_card_->setWordWrap(true);
    // No `setEnabled(false)` to make it read as secondary. Disabling is
    // a state change standing in for an appearance one -- it also takes
    // the widget out of the accessibility tree as
    // interactive-but-unavailable, and it says "you cannot use this"
    // about a label nobody was going to click. An empty label shows
    // nothing anyway, so the dimming bought exactly nothing.
    lower_layout->addWidget(last_card_);

    // Wraps rather than crushes -- see flow_layout.hpp.
    auto* controls = new FlowLayout();
    start_button_ = new QPushButton(tr("&Start receiving"), this);
    connect(start_button_, &QPushButton::clicked, this, &ReceivePanel::start);
    stop_button_ = new QPushButton(tr("Sto&p"), this);
    connect(stop_button_, &QPushButton::clicked, this, &ReceivePanel::stop);
    stop_button_->setEnabled(false);
    save_button_ = new QPushButton(tr("Sa&ve image..."), this);
    connect(save_button_, &QPushButton::clicked, this, &ReceivePanel::save_current);
    save_button_->setEnabled(false);
    folder_button_ = new QPushButton(tr("Show f&older"), this);
    folder_button_->setToolTip(tr("Open the folder holding the last saved image"));
    connect(folder_button_, &QPushButton::clicked, this,
            &ReceivePanel::open_saved_folder);
    folder_button_->setEnabled(false);
    autosave_ = new QCheckBox(tr("&Autosave"), this);
    autosave_->setChecked(app_->config().receive.autosave);
    connect(autosave_, &QCheckBox::toggled, this,
            &ReceivePanel::on_autosave_toggled);

    controls->addWidget(start_button_);
    controls->addWidget(stop_button_);
    controls->addWidget(save_button_);
    controls->addWidget(folder_button_);
    controls->addWidget(autosave_);
    // Into the strip, not the panel: the strip is the unit whose height
    // is matched against the transmit pane's, so anything left outside
    // it would break the equality it exists to give.
    lower_layout->addLayout(controls);

    layout->addWidget(strip_);
}

void ReceivePanel::resizeEvent(QResizeEvent* event) {
    QWidget::resizeEvent(event);
    place_banner();
}

void ReceivePanel::place_banner() { style::place_over(banner_, preview_); }

bool ReceivePanel::listening() const { return running_.load(); }

void ReceivePanel::sync_from_config() {
    // Without blocking signals this would write the value straight back
    // to the config it just came from -- harmless today, but it makes
    // the checkbox the authority over the file it is meant to reflect.
    const QSignalBlocker blocker(autosave_);
    autosave_->setChecked(app_->config().receive.autosave);
}

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
    // decode_loop's own locals (blind_acc included) already start clean
    // just by being a fresh function call, but the Progress this panel
    // *displays* lives here, not in the loop -- and stop() (called by
    // both the Stop button and suspend_for_transmit) never touches it.
    // Without this, stopping while a reception was in progress leaves
    // shared_ holding a stale "Receiving" Progress -- old mode, old
    // frame count, old callsign -- and the 500 ms status timer shows it
    // immediately on the next Start, before the new loop has run a
    // single poll. A fresh SharedState is the rest of what "start() is
    // the one method that resets all receiver state" needs to be true.
    shared_ = std::make_unique<rx::SharedState>();
    if (Waterfall* w = fall()) w->set_ring(ring_);

    try {
        stream_ = std::make_unique<audio::qt::InputStream>(
            config.audio.input_device, *ring_, config::FS,
            // What opened, and at what rate: information. The resampler
            // being in the path is the normal case -- very little
            // hardware is natively 8 kHz -- and a sticky red banner
            // saying so told operators their setup was broken when it
            // was working.
            [this](const std::string& message) {
                app_->log_event("rx", log::Severity::Info,
                                QString::fromStdString(message));
            },
            // Capture actually failing: banner and log, as before. This
            // is the half that must not be quiet.
            [this](const std::string& message) {
                emit errorOccurred(QString::fromStdString(message));
            });
    } catch (const std::exception& e) {
        if (Waterfall* w = fall()) w->set_ring(nullptr);
        ring_.reset();
        app_->log_event("rx", log::Severity::Error,
                        tr("could not open the input device: %1")
                            .arg(QString::fromUtf8(e.what())));
        QMessageBox::critical(
            this, tr("Could not open the input device"),
            tr("%1\n\nCheck the input device in Settings.")
                .arg(QString::fromUtf8(e.what())));
        return false;
    }

    // Diversity reception's second branch. Failing to open it fails the
    // whole start, the same as the primary device above -- silently
    // falling back to single-branch reception would mean the operator's
    // "diversity reception" setting quietly stopped meaning what it
    // said, which is worse than a clear error naming the device that
    // did not open.
    if (config.receive.diversity_enabled) {
        ring2_ = std::make_shared<rx::RingBuffer>(config.receive.buffer_seconds);
        try {
            stream2_ = std::make_unique<audio::qt::InputStream>(
                config.receive.diversity_device, *ring2_, config::FS,
                [this](const std::string& message) {
                    app_->log_event("rx", log::Severity::Info,
                                    QString::fromStdString(message));
                },
                [this](const std::string& message) {
                    emit errorOccurred(QString::fromStdString(message));
                });
        } catch (const std::exception& e) {
            ring2_.reset();
            if (Waterfall* w = fall()) w->set_ring(nullptr);
            ring_.reset();
            stream_.reset();
            app_->log_event(
                "rx", log::Severity::Error,
                tr("could not open the diversity input device: %1")
                    .arg(QString::fromUtf8(e.what())));
            QMessageBox::critical(
                this, tr("Could not open the diversity input device"),
                tr("%1\n\nCheck the second input device in Settings, or turn "
                   "diversity reception off there.")
                    .arg(QString::fromUtf8(e.what())));
            return false;
        }
    }

    rx::RxConfig rx_config;
    rx_config.out_dir = config.folders.receive_dir;
    rx_config.poll_interval = config.receive.poll_interval;
    rx_config.end_grace = config.receive.end_grace;
    rx_config.size = rx::parse_size(config.receive.save_size);
    rx_config.once = false;
    rx_config.blind_search_seconds = config.receive.blind_search_seconds;
    rx_config.blind_wide = config.receive.blind_wide;
    rx_config.drift_track = modem::drift_track_from_name(config.receive.drift_track);

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
    const bool diversity = config.receive.diversity_enabled;
    const bool low_cpu = config.receive.low_cpu;
    const int device_rate = stream_->device_rate();

    // Empty (falsy) unless diversity reception's debug image is on --
    // decode_loop_diversity takes a pointer, so this has to outlive the
    // call, hence captured by value into the thread lambda below rather
    // than constructed inside it.
    rx::DebugImageSink debug_sink;
    if (diversity && config.receive.diversity_debug_image) {
        debug_sink = rx::save_debug_image_to_dir_sink();
    }

    running_.store(true);
    thread_ = std::thread([this, rx_config, decoder, sink, low_cpu, diversity,
                           debug_sink] {
        try {
            if (diversity) {
                const std::array<rx::RingBuffer*, 2> rings{ring_.get(), ring2_.get()};
                const rx::DebugImageSink* debug_ptr = debug_sink ? &debug_sink : nullptr;
                rx::decode_loop_diversity(rings, decoder, *shared_, rx_config,
                                          stop_flag_, sink, debug_ptr);
            } else if (low_cpu) {
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
    set_status(tr("Listening at %1 Hz").arg(device_rate));
    // A clean restart of the activity clears its error banner; the log
    // keeps the history.
    banner_->clear();
    was_receiving_ = false;
    diversity_active_ = diversity;
    branch_a_chip_->setVisible(diversity_active_);
    branch_b_chip_->setVisible(diversity_active_);
    set_lock_chip(branch_a_chip_, false);
    set_lock_chip(branch_b_chip_, false);
    app_->log_event("rx", log::Severity::Info,
                    tr("listening (device at %1 Hz)").arg(device_rate));
    emit listeningChanged(true);
    return true;
}

void ReceivePanel::stop() {
    // Idempotence matters here: closeEvent stops the panel and the
    // destructor stops it again, and only the transition from "was
    // actually listening" deserves a log line.
    const bool was_listening = thread_.joinable();

    stop_flag_.set();
    // The streams first: they feed the loop, and stopping them means
    // the loop's next poll sees no new audio rather than racing us.
    stream_.reset();
    stream2_.reset();
    if (thread_.joinable()) thread_.join();
    running_.store(false);

    if (Waterfall* w = fall()) w->set_ring(nullptr);
    ring_.reset();
    ring2_.reset();
    diversity_active_ = false;

    if (start_button_ != nullptr) {
        start_button_->setEnabled(true);
        stop_button_->setEnabled(false);
        set_status(tr("Stopped"));
        // `refresh_status` returns early when not listening, so without
        // this the bar keeps whatever fraction the reception had
        // reached: "Stopped" over a bar still reading 63%.
        progress_->setValue(0);
        branch_a_chip_->hide();
        branch_b_chip_->hide();
    }
    if (was_listening) {
        app_->log_event("rx", log::Severity::Info, tr("stopped"));
        emit listeningChanged(false);
    }
}

void ReceivePanel::suspend_for_transmit() {
    if (!listening()) return;
    suspended_for_tx_ = true;
    stop();
    // This pane is on screen throughout the over now, so a half-duplex
    // pause has to look deliberate rather than like a receiver that
    // died: the waterfall stops dead when the ring goes away, and
    // without this it is the same picture as a wedged capture.
    set_status(tr("Paused — transmitting"));
    preview_->setEnabled(false);
    if (Waterfall* w = fall()) w->setEnabled(false);
    // And the button that would undo half duplex. `stop()` re-enables
    // it on its way through, and this pane is now in front of the
    // operator for the whole over: clicking Start here would open
    // capture while the rig is keyed, feed our own transmission to the
    // decoder, and -- because `start()` returns early when already
    // listening -- deny `resume_after_transmit` the fresh ring buffer
    // that is what keeps the tail of our own signal out of a
    // "reception".
    start_button_->setEnabled(false);
}

void ReceivePanel::resume_after_transmit() {
    if (!suspended_for_tx_) return;
    suspended_for_tx_ = false;
    preview_->setEnabled(true);
    if (Waterfall* w = fall()) w->setEnabled(true);
    start_button_->setEnabled(true);  // start() disables it again
    // start() allocates a fresh ring buffer, so the tail of our own
    // transmission is dropped rather than decoded back as a reception.
    start();
    if (Waterfall* w = fall()) w->clear();
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

QWidget* ReceivePanel::picture_area() const { return preview_; }

QWidget* ReceivePanel::control_strip() const { return strip_; }

void ReceivePanel::set_status(const QString& text) {
    status_->setText(text);
    emit statusChanged(text);
}

void ReceivePanel::refresh_status() {
    if (!listening()) return;
    const rx::Progress progress = shared_->get();

    // The engine has no "sync acquired" event -- acquisition is just the
    // first poll whose status reads "receiving" -- so the edge is
    // detected and logged here (once per acquisition).
    if (progress.status == rx::Status::Receiving && !was_receiving_) {
        was_receiving_ = true;
        QString acquired = tr("sync acquired");
        if (progress.mode_name) {
            acquired += tr(": mode %1").arg(QString::fromStdString(*progress.mode_name));
        } else {
            acquired += tr(" (blind)");
        }
        if (!progress.callsign.empty()) {
            acquired += tr(" de %1").arg(QString::fromStdString(progress.callsign));
        }
        app_->log_event("rx", log::Severity::Info, acquired);
    } else if (progress.status == rx::Status::Listening) {
        was_receiving_ = false;
    }

    QString text;
    if (progress.status == rx::Status::Listening) {
        text = tr("Listening... (%1 s captured)")
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
            text = tr("Receiving (blind sync): %1%")
                       .arg(100.0 * progress.progress_frac, 0, 'f', 0);
        }
        text += SEP + style::fmt_snr_db(progress.snr_db);
        if (!progress.callsign.empty()) {
            text += SEP + tr("de %1").arg(QString::fromStdString(progress.callsign));
        }
    } else {
        text = tr("Complete") + SEP + style::fmt_snr_db(progress.snr_db);
        if (last_saved_path_) {
            text += SEP + tr("saved %1")
                              .arg(QString::fromStdString(
                                  fs::path(*last_saved_path_).filename().string()));
        }
    }

    set_status(text);
    progress_->setValue(static_cast<int>(100.0 * progress.progress_frac));

    if (diversity_active_) {
        set_lock_chip(branch_a_chip_, progress.branch_a_locked);
        set_lock_chip(branch_b_chip_, progress.branch_b_locked);
    }

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

    // The one durable record of a completed reception: mode, callsign,
    // SNR, frame count and the *full* saved path -- which was previously
    // shown only in a 5-second status bar flash before this line existed.
    QString line = tr("reception complete");
    if (reception->mode_name) {
        line += tr(": mode %1").arg(QString::fromStdString(*reception->mode_name));
    }
    if (!reception->callsign.empty()) {
        line += tr(" de %1").arg(QString::fromStdString(reception->callsign));
    }
    if (reception->n_frames_expected) {
        line += tr(", %1/%2 frames")
                    .arg(reception->frames_received.value_or(0))
                    .arg(*reception->n_frames_expected);
    }
    if (!std::isnan(reception->snr_db)) {
        line += tr(", %1").arg(style::fmt_snr_db(reception->snr_db));
    }
    if (!saved_path.isEmpty()) line += tr(" — saved %1").arg(saved_path);
    app_->log_event("rx", log::Severity::Info, line);

    // And the same facts on screen, where they stay. The engine resets
    // its shared state two seconds after a reception, so everything
    // below this point was previously unrecoverable from the UI.
    QString card = tr("Last: %1").arg(QTime::currentTime().toString(QStringLiteral("HH:mm")));
    if (reception->mode_name) {
        card += SEP + tr("mode %1").arg(QString::fromStdString(*reception->mode_name));
    }
    if (!reception->callsign.empty()) {
        card += SEP + tr("de %1").arg(QString::fromStdString(reception->callsign));
    }
    if (!std::isnan(reception->snr_db)) {
        card += SEP + style::fmt_snr_db(reception->snr_db);
    }
    if (reception->n_frames_expected) {
        card += SEP + tr("%1/%2 frames")
                          .arg(reception->frames_received.value_or(0))
                          .arg(*reception->n_frames_expected);
    }
    if (!saved_path.isEmpty()) {
        card += SEP + QString::fromStdString(
                          fs::path(saved_path.toStdString()).filename().string());
    }
    last_card_->setText(card);
    folder_button_->setEnabled(!saved_path.isEmpty());

    emit imageReceived(reception->image);
    if (!saved_path.isEmpty()) emit receptionSaved(saved_path);
}

void ReceivePanel::on_error(const QString& message) {
    // Sticky (the banner) plus durable (the log). Deliberately not the
    // status label: that is the progress tier, the next 500 ms refresh
    // would overwrite it anyway, and a single-line label given a long
    // error inflates the panel's minimum width.
    place_banner();
    banner_->show_error(message);
    app_->log_event("rx", log::Severity::Error, message);
}

void ReceivePanel::on_autosave_toggled(bool on) {
    app_->config().receive.autosave = on;
    app_->save_config();
}

void ReceivePanel::show_image(const images::Picture& image) {
    const QPixmap pixmap = style::to_pixmap(image);
    if (pixmap.isNull()) return;
    // The box keeps the original and rescales itself on every resize;
    // the panel no longer has to notice.
    preview_->set_picture(pixmap);
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
        app_->log_event("rx", log::Severity::Error,
                        tr("could not save %1: %2")
                            .arg(path, QString::fromUtf8(e.what())));
        QMessageBox::critical(this, tr("Could not save"),
                              QString::fromUtf8(e.what()));
        return;
    }
    last_saved_path_ = path.toStdString();
    folder_button_->setEnabled(true);
    // The card claims to name the file the last picture went to, so a
    // manual save has to update it -- otherwise the button points at
    // one file while the card names another (or names none at all,
    // with autosave off).
    const QString name = QString::fromStdString(
        fs::path(path.toStdString()).filename().string());
    if (last_card_->text().isEmpty()) {
        last_card_->setText(tr("Saved %1").arg(name));
    } else {
        last_card_->setText(last_card_->text() + SEP + tr("saved %1").arg(name));
    }
    emit receptionSaved(path);
}

void ReceivePanel::fill_for_screenshot() {
    // The longest line `refresh_status` can build, the longest card
    // `on_reception` can build, and the banner inside the layout.
    set_status(
        tr("Receiving mode C: frame 220/220 (100%)  ·  SNR 8.3 dB  ·  de KD8XYZ"));
    progress_->setValue(100);
    branch_a_chip_->show();
    branch_b_chip_->show();
    set_lock_chip(branch_a_chip_, true);
    set_lock_chip(branch_b_chip_, false);
    last_card_->setText(
        tr("Last: 14:32  ·  mode C  ·  de KD8XYZ  ·  SNR 8.3 dB"
           "  ·  220/220 frames  ·  KD8XYZ-C-14230000.png"));
    save_button_->setEnabled(true);
    folder_button_->setEnabled(true);
    place_banner();
    banner_->show_error(
        tr("could not save received image: read-only file system"));
}

void ReceivePanel::open_saved_folder() {
    if (!last_saved_path_) return;
    // The containing folder rather than the file: opening the picture
    // itself would launch an image viewer, and what the operator wants
    // here is the directory their captures are piling up in.
    const QString dir =
        QString::fromStdString(fs::path(*last_saved_path_).parent_path().string());
    // Logged unconditionally, before the call. `openUrl` reports only
    // whether a *handler could be launched*: on Unix it ends in a
    // detached `xdg-open`, which returns true as soon as the process
    // starts and says nothing about whether a file manager ever
    // appeared. So a false return is worth reporting, but a true one is
    // not evidence of success -- and the case this button exists to
    // survive (a minimal desktop with nothing registered) is precisely
    // the one that returns true and does nothing. The log line is what
    // makes that debuggable instead of baffling.
    app_->log_event("rx", log::Severity::Info, tr("opening %1").arg(dir));
    if (!QDesktopServices::openUrl(QUrl::fromLocalFile(dir))) {
        app_->log_event("rx", log::Severity::Error,
                        tr("could not open %1 — no handler for local "
                           "folders on this desktop")
                            .arg(dir));
    }
}

}  // namespace sstvae::gui

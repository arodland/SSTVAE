#include "app_state.hpp"

#include <exception>
#include <initializer_list>
#include <utility>

#include "checkpoint/checkpoint.hpp"
#include "checkpoint/qt_fetcher.hpp"
#include "rig_config.hpp"

namespace sstvae::gui {

namespace {

// Which artifact to load for a part, honouring an explicit --model /
// configured path and otherwise the published checkpoint. Same shape as
// `sstvae_decode.cpp`'s resolver, so the app and the CLI cannot
// disagree about where a model comes from.
codec::Resolver model_resolver(const std::string& path,
                               const std::string& precision) {
    return [path, precision](const std::string& part) {
        return checkpoint::resolve_onnx(part, path, precision);
    };
}

}  // namespace

AppState::AppState(QObject* parent)
    : QObject(parent),
      rig_(
          // Both callbacks arrive on the rig's worker thread. Emitting
          // a Qt signal across threads is the supported way to hand
          // that to the GUI -- the connection becomes queued, so the
          // slot runs on the receiving object's thread and nothing on
          // the GUI thread ever touches the rig. log_event is
          // thread-safe by design, so logging here is fine too.
          [](std::optional<double>) {},
          [this](const std::string& text, bool error) {
              const QString qtext = QString::fromStdString(text);
              log_event("rig", error ? log::Severity::Error : log::Severity::Info,
                        qtext);
              emit rigStatus(qtext, error);
          }) {
    // Every append reaches the pane (queued signal; non-blocking from
    // any thread) and, if it opened, the file. Registered before the
    // first append so nothing is ever visible in one place and not the
    // other.
    log_.add_sink([this](const log::Entry& e) {
        emit logEntry(static_cast<qlonglong>(e.ms),
                      QString::fromStdString(e.source),
                      static_cast<int>(e.severity),
                      QString::fromStdString(e.text));
    });
    try {
        file_log_ = std::make_unique<log::FileWriter>(
            settings::config_path().parent_path() / "sstvae.log");
        log_.add_sink([this](const log::Entry& e) {
            file_log_->write(e);
            // A write failure is permanent (FileWriter closes the
            // stream), so it must be reported, once, whenever it
            // happens -- a full disk hours in must not fail quietly.
            if (!file_log_reported_.load(std::memory_order_relaxed)) {
                const std::string err = file_log_->error();
                if (!err.empty() && !file_log_reported_.exchange(true)) {
                    emit fileLogFailed(QString::fromStdString(err));
                }
            }
        });
    } catch (const std::exception&) {
        // No config directory at all; the in-memory log still works and
        // log_file_note() reports the absence.
    }

    // The load happens here rather than in the member initializer so the
    // validation notes can be kept: a corrupt or out-of-range field is
    // coerced to its default *and reported*, where previously the notes
    // were discarded wholesale.
    const settings::LoadResult loaded = settings::load();
    config_ = loaded.config;
    for (const std::string& message : loaded.messages()) {
        log_event("app", log::Severity::Warning,
                  tr("config: %1").arg(QString::fromStdString(message)));
    }

    // The fetcher that downloads the published checkpoint, with live
    // progress. Installed here rather than in main() so the progress
    // hook has somewhere to land. Three threads fetch through it: the
    // model thread (decoder preload), the transmit worker (the lazily
    // loaded encoder) and the optimizer worker (the gradient graph).
    // The `this` capture outlives all of them because ~MainWindow
    // destroys the panels -- which join those workers -- *before* its
    // AppState child, and ~AppState then joins the model thread.
    checkpoint::install_qt_fetcher([this](std::int64_t received,
                                          std::int64_t total) {
        emit modelProgress(static_cast<qlonglong>(received),
                           static_cast<qlonglong>(total));
    });
}

AppState::~AppState() {
    // stop() detaches by design, so a worker may still be inside
    // libhamlib when this returns -- and it holds a `this` that is about
    // to stop existing. Waiting here is the one place that is correct;
    // see RigController::wait_for_shutdown.
    rig_.stop();
    rig_.wait_for_shutdown();
    if (model_thread_.joinable()) model_thread_.join();
}

void AppState::load_model_async() {
    if (model_thread_.joinable()) model_thread_.join();
    {
        const std::lock_guard<std::mutex> lock(model_mutex_);
        model_.reset();
        model_error_.clear();
    }
    const std::string path = config_.model_path;
    const std::string precision =
        settings::codec_precision(config_).value_or(std::string());

    model_thread_ = std::thread([this, path, precision] {
        std::unique_ptr<codec::OnnxCodec> loaded;
        QString error;
        try {
            loaded = std::make_unique<codec::OnnxCodec>(
                model_resolver(path, precision));
            // Force the decoder now rather than on the first reception.
            // The parts are lazy and independent on purpose -- a
            // receive-only station never fetches the encoder -- but
            // "the model is ready" has to mean something, and the
            // decoder is what a listening station needs first.
            loaded->preload("decoder");
        } catch (const std::exception& e) {
            loaded.reset();
            error = QString::fromUtf8(e.what());
        }
        {
            const std::lock_guard<std::mutex> lock(model_mutex_);
            model_ = std::move(loaded);
            model_error_ = error;
        }
        if (error.isEmpty()) {
            log_event("app", log::Severity::Info, tr("model ready"));
        } else {
            log_event("app", log::Severity::Error,
                      tr("model failed to load: %1").arg(error));
        }
        emit modelLoaded();
    });
}

codec::OnnxCodec* AppState::model() {
    const std::lock_guard<std::mutex> lock(model_mutex_);
    return model_.get();
}

QString AppState::model_error() const {
    const std::lock_guard<std::mutex> lock(model_mutex_);
    return model_error_;
}

std::optional<double> AppState::current_frequency_hz() const {
    return rig_.frequency_hz();
}

std::function<void(bool)> AppState::ptt() {
    // Nothing to key, in both of the cases where that is true: rig
    // control off entirely, and rig control on but the radio keyed by
    // its own VOX. Returning nothing rather than a no-op is what lets
    // the transmit engine tell "do not key" apart from "keying failed".
    if (!config_.rig.enabled || config_.rig.ptt_method == "vox") return nullptr;
    return rig_.ptt_function();
}

void AppState::connect_rig() {
    rig_.stop();
    if (!config_.rig.enabled) {
        emit rigStatus(QStringLiteral("Rig control off"), false);
        return;
    }
    rig_.start(make_backend(config_.rig), controller_config(config_.rig));
}

void AppState::disconnect_rig() { rig_.stop(); }

void AppState::pause_rig_polling() { rig_.pause_polling(); }

void AppState::resume_rig_polling() { rig_.resume_polling(); }

void AppState::save_config() {
    try {
        settings::save(config_);
    } catch (const std::exception& e) {
        // A read-only config directory must not break the session --
        // but the operator has to find out *somewhere* that their
        // settings are not sticking.
        log_event("app", log::Severity::Error,
                  tr("could not save settings: %1").arg(QString::fromUtf8(e.what())));
    }
}

void AppState::log_event(const char* source, log::Severity severity,
                         const QString& text) {
    log_.append(source, severity, text.toStdString());
}

QString AppState::log_file_path() const {
    if (!file_log_) return QString();
    return QString::fromStdString(file_log_->path().string());
}

QString AppState::log_file_note() const {
    if (!file_log_) return tr("(no log file: no config directory)");
    const std::string error = file_log_->error();
    if (!error.empty()) {
        return tr("(log file unavailable: %1)").arg(QString::fromStdString(error));
    }
    return QString();
}

}  // namespace sstvae::gui

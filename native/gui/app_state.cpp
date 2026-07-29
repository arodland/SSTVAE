#include "app_state.hpp"

#include <exception>
#include <initializer_list>
#include <utility>

#include "checkpoint/checkpoint.hpp"
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
      config_(settings::load().config),
      rig_(
          // Both callbacks arrive on the rig's worker thread. Emitting
          // a Qt signal across threads is the supported way to hand
          // that to the GUI -- the connection becomes queued, so the
          // slot runs on the receiving object's thread and nothing on
          // the GUI thread ever touches the rig.
          [](std::optional<double>) {},
          [this](const std::string& text) {
              emit rigStatus(QString::fromStdString(text));
          }) {}

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
        emit rigStatus(QStringLiteral("Rig control off"));
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
    } catch (const std::exception&) {
        // A read-only config directory must not break the session.
    }
}

}  // namespace sstvae::gui

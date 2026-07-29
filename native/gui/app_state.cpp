#include "app_state.hpp"

#include <exception>
#include <initializer_list>
#include <utility>

#include "checkpoint/checkpoint.hpp"
#include "rig/hamlib.hpp"

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

// The settings file spells the enumerated rig options in lowercase, so
// a config a human opens does not read like Hamlib's combo boxes. This
// is the one place the two vocabularies meet.
//
// An unrecognized value falls back to Default rather than being an
// error, and that direction is deliberate: Default means "do not set
// the token", so a typo leaves the backend on its own settings instead
// of forcing a wrong one onto a radio.
template <typename E>
E lookup(const std::string& value, std::initializer_list<std::pair<const char*, E>> table,
         E fallback) {
    for (const auto& [name, mapped] : table) {
        if (value == name) return mapped;
    }
    return fallback;
}

std::unique_ptr<rig::RigBackend> backend_for(const settings::RigConfig& config) {
    rig::HamlibConfig hamlib;
    hamlib.model = config.model;
    hamlib.device = config.device;
    hamlib.baud = config.baud;
    hamlib.data_bits = lookup<rig::DataBits>(
        config.data_bits,
        {{"seven", rig::DataBits::Seven}, {"eight", rig::DataBits::Eight}},
        rig::DataBits::Default);
    hamlib.stop_bits = lookup<rig::StopBits>(
        config.stop_bits,
        {{"one", rig::StopBits::One}, {"two", rig::StopBits::Two}},
        rig::StopBits::Default);
    hamlib.parity = lookup<rig::Parity>(config.parity,
                                        {{"none", rig::Parity::None},
                                         {"odd", rig::Parity::Odd},
                                         {"even", rig::Parity::Even}},
                                        rig::Parity::Default);
    hamlib.handshake = lookup<rig::Handshake>(
        config.handshake,
        {{"none", rig::Handshake::None},
         {"xonxoff", rig::Handshake::XonXoff},
         {"hardware", rig::Handshake::Hardware}},
        rig::Handshake::Default);
    hamlib.dtr = lookup<rig::LineState>(
        config.dtr, {{"high", rig::LineState::High}, {"low", rig::LineState::Low}},
        rig::LineState::Default);
    hamlib.rts = lookup<rig::LineState>(
        config.rts, {{"high", rig::LineState::High}, {"low", rig::LineState::Low}},
        rig::LineState::Default);
    // Cat rather than Vox as the fallback: an unreadable value must not
    // silently turn keying off, which would look like a dead PTT.
    hamlib.ptt_method = lookup<rig::PttMethod>(config.ptt_method,
                                               {{"vox", rig::PttMethod::Vox},
                                                {"cat", rig::PttMethod::Cat},
                                                {"dtr", rig::PttMethod::Dtr},
                                                {"rts", rig::PttMethod::Rts}},
                                               rig::PttMethod::Cat);
    hamlib.ptt_device = config.ptt_device;
    hamlib.mode = lookup<rig::RigMode>(
        config.mode, {{"usb", rig::RigMode::Usb}, {"pkt_usb", rig::RigMode::PktUsb}},
        rig::RigMode::None);
    return rig::make_hamlib_backend(hamlib);
}

rig::RigConfig controller_config(const settings::RigConfig& config) {
    rig::RigConfig out;
    out.poll_interval_s = config.poll_interval_s;
    return out;
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
    rig_.start(backend_for(config_.rig), controller_config(config_.rig));
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

#include "session.hpp"

#include <chrono>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <sstream>

#include "checkpoint/checkpoint.hpp"
#include "java_fetcher.hpp"
#include "config.hpp"
#include "images/images.hpp"
#include "images/types.hpp"

namespace sstvae::androidapp {

namespace {

std::string json_escape(const std::string& s) {
    std::string out;
    for (const char c : s) {
        if (c == '"' || c == '\\') out += '\\';
        out += c;
    }
    return out;
}

}  // namespace

Session& Session::instance() {
    static Session s;
    return s;
}

void Session::set_picture_dir(std::string dir) {
    std::lock_guard<std::mutex> lk(mu_);
    picture_dir_ = std::move(dir);
}

// Write the picture and its metadata together.
//
// **The sidecar is not optional.** `rx/engine` wipes mode, callsign,
// SNR and frame count from shared state two seconds after a reception,
// and on a phone the operator is usually not looking then -- so a
// picture saved without them is a picture whose provenance is simply
// gone. The desktop papered over this with an on-screen card; here the
// facts go to disk beside the image, where opening it later still
// answers "who sent this, and how well did it come through".
std::optional<std::string> Session::save_reception(const rx::Reception& r) {
    if (r.image.width <= 0 || r.image.height <= 0) return std::nullopt;

    std::string dir;
    {
        std::lock_guard<std::mutex> lk(mu_);
        dir = picture_dir_;
    }
    if (dir.empty()) return std::nullopt;

    const auto now = std::chrono::system_clock::now();
    const std::time_t t = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
#if defined(_WIN32)
    gmtime_s(&tm, &t);
#else
    gmtime_r(&t, &tm);
#endif
    char stamp[32];
    std::strftime(stamp, sizeof(stamp), "%Y-%m-%d_%H%M%SZ", &tm);

    const std::string base = dir + "/" + stamp;
    const std::string png = base + ".png";
    try {
        std::error_code ec;
        std::filesystem::create_directories(dir, ec);
        images::save_png(r.image, png);

        std::ostringstream meta;
        meta << "{\n";
        meta << "  \"received\": \"" << stamp << "\",\n";
        meta << "  \"callsign\": \"" << json_escape(r.callsign) << "\",\n";
        meta << "  \"mode\": " << (r.mode_name ? "\"" + json_escape(*r.mode_name) + "\""
                                               : "null")
             << ",\n";
        meta << "  \"snr_db\": " << r.snr_db << ",\n";
        meta << "  \"frames_received\": "
             << (r.frames_received ? std::to_string(*r.frames_received) : "null") << ",\n";
        meta << "  \"frames_expected\": "
             << (r.n_frames_expected ? std::to_string(*r.n_frames_expected) : "null")
             << "\n";
        meta << "}\n";

        std::ofstream out(base + ".json");
        out << meta.str();
    } catch (const std::exception& e) {
        std::lock_guard<std::mutex> lk(mu_);
        error_ = std::string("could not save the reception: ") + e.what();
        return std::nullopt;
    }
    return png;
}

Session::~Session() {
    stop();
    if (model_thread_.joinable()) model_thread_.join();
}

void Session::load_model_async() {
    {
        std::lock_guard<std::mutex> lk(model_mu_);
        if (model_state_ == ModelState::Downloading ||
            model_state_ == ModelState::Loading || model_state_ == ModelState::Ready) {
            return;
        }
        model_state_ = ModelState::Loading;
        model_error_.clear();
        model_received_ = 0;
        model_total_ = 0;
    }
    if (model_thread_.joinable()) model_thread_.join();

    model_thread_ = std::thread([this] {
        // The progress hook is also what distinguishes Downloading from
        // Loading: `resolve_onnx` is silent on a cache hit, so the state
        // only moves to Downloading if bytes actually start arriving.
        // The Java transport, not Qt's: Qt for Android has no TLS
        // backend. See java_fetcher.hpp for why the split falls where
        // it does.
        install_java_fetcher(
            [this](std::int64_t received, std::int64_t total) {
                std::lock_guard<std::mutex> lk(model_mu_);
                model_state_ = ModelState::Downloading;
                model_received_ = received;
                model_total_ = total;
            });

        std::shared_ptr<codec::OnnxCodec> loaded;
        std::string error;
        try {
            loaded = std::make_shared<codec::OnnxCodec>(
                [](const std::string& part) { return checkpoint::resolve_onnx(part); });
            // Force the decoder now. The parts are lazy and independent
            // on purpose -- a receive-only station never fetches the
            // encoder -- but "the model is ready" has to mean something,
            // and the alternative is discovering a missing artifact
            // during the one reception it was needed for.
            loaded->preload("decoder");
        } catch (const std::exception& e) {
            error = e.what();
            loaded.reset();
        }

        std::lock_guard<std::mutex> lk(model_mu_);
        codec_ = std::move(loaded);
        model_state_ = codec_ ? ModelState::Ready : ModelState::Failed;
        model_error_ = error;
        model_received_ = 0;
        model_total_ = 0;
    });
}

ModelState Session::model_state() const {
    std::lock_guard<std::mutex> lk(model_mu_);
    return model_state_;
}

std::string Session::model_error() const {
    std::lock_guard<std::mutex> lk(model_mu_);
    return model_error_;
}

std::pair<std::int64_t, std::int64_t> Session::model_progress() const {
    std::lock_guard<std::mutex> lk(model_mu_);
    return {model_received_, model_total_};
}

bool Session::running() const {
    std::lock_guard<std::mutex> lk(mu_);
    return stream_ != nullptr;
}

bool Session::start(const std::string& device_name) {
    std::lock_guard<std::mutex> lk(mu_);
    if (stream_) return true;
    error_.clear();

    ring_ = std::make_unique<rx::RingBuffer>(130.0);
    state_ = std::make_unique<rx::SharedState>();
    stop_ = std::make_unique<rx::StopFlag>();

    // **The decoder is looked up per call, not captured at start.**
    // That is what lets listening begin before the model has arrived:
    // the engine reports mode, frames, callsign and SNR from the first
    // poll, and pictures start appearing the moment the download
    // finishes, with no restart. Capturing the codec here would make a
    // session started during the download permanently pictureless --
    // and on a first run that is every session.
    rx::Decoder decoder = [this](std::span<const double> latents,
                                 std::span<const double> weights) {
        std::shared_ptr<codec::OnnxCodec> c;
        {
            std::lock_guard<std::mutex> lk(model_mu_);
            c = codec_;
        }
        if (!c) return images::Picture{};
        return c->decode(std::vector<double>(latents.begin(), latents.end()),
                         std::vector<double>(weights.begin(), weights.end()));
    };
    rx::RxConfig cfg;
    cfg.poll_interval = 5.0;
    rx::Sink sink = [this](const rx::Reception& r) -> std::optional<std::string> {
        return save_reception(r);
    };

    try {
        stream_ = std::make_unique<audio::android::InputStream>(
            device_name, *ring_, config::FS, audio::android::Report{},
            [this](const std::string& m) {
                std::lock_guard<std::mutex> lk2(mu_);
                error_ = m;
            });
    } catch (const std::exception& e) {
        error_ = e.what();
        ring_.reset();
        state_.reset();
        stop_.reset();
        return false;
    }

    thread_ = std::thread([this, decoder, cfg, sink] {
        try {
            rx::decode_loop(*ring_, decoder, *state_, cfg, *stop_, sink);
        } catch (const std::exception&) {
            // One bad session must not take the process down. It
            // surfaces as the loop no longer advancing `polls`, which is
            // why that counter is published at all.
        }
    });
    return true;
}

void Session::stop() {
    // Stop capture and signal the loop *before* taking the lock: the
    // join below can take a poll interval, and a view asking for
    // `progress()` meanwhile must not block behind it.
    std::unique_ptr<audio::android::InputStream> stream;
    std::unique_ptr<rx::StopFlag> flag;
    {
        std::lock_guard<std::mutex> lk(mu_);
        stream = std::move(stream_);
        flag = std::move(stop_);
    }
    if (stream) stream->stop();
    if (flag) flag->set();
    if (thread_.joinable()) thread_.join();

    std::lock_guard<std::mutex> lk(mu_);
    ring_.reset();
    state_.reset();
}

rx::Progress Session::progress() const {
    std::lock_guard<std::mutex> lk(mu_);
    if (!state_) return rx::Progress{};
    return state_->get();
}

std::string Session::last_error() const {
    std::lock_guard<std::mutex> lk(mu_);
    return error_;
}

int Session::device_rate() const {
    std::lock_guard<std::mutex> lk(mu_);
    return stream_ ? stream_->device_rate() : 0;
}

std::string Session::routed_device() const {
    std::lock_guard<std::mutex> lk(mu_);
    return stream_ ? stream_->routed_device() : std::string{};
}

std::string Session::routing_warning() const {
    std::lock_guard<std::mutex> lk(mu_);
    return stream_ ? stream_->routing_warning() : std::string{};
}

double Session::peak_level() const {
    std::lock_guard<std::mutex> lk(mu_);
    return stream_ ? stream_->peak_level() : 0.0;
}

double Session::near_zero_fraction() const {
    std::lock_guard<std::mutex> lk(mu_);
    return stream_ ? stream_->near_zero_fraction() : 0.0;
}

std::vector<double> Session::audio_tail(std::size_t n) const {
    std::lock_guard<std::mutex> lk(mu_);
    if (!ring_) return {};
    return ring_->tail(n);
}

double Session::capture_drift_ppm() const {
    std::lock_guard<std::mutex> lk(mu_);
    return stream_ ? stream_->capture_drift_ppm() : 0.0;
}

}  // namespace sstvae::androidapp

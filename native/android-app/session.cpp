#include "session.hpp"

#include <chrono>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>

#include "assets.hpp"
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

// **Deliberately immortal: never destroyed, so no destructor runs at
// exit.**
//
// A function-local `static Session s;` registers a destructor with
// `atexit`, and that destructor tears down an `OnnxCodec` — by which
// time onnxruntime's own statics may already be gone. Measured: the app
// aborted on every exit with `FORTIFY: pthread_mutex_lock called on a
// destroyed mutex` inside `~OnnxCodec`, from `~Session`, on the Qt main
// loop thread. Static destruction order across translation units and
// shared libraries is not something this code can arrange, and there is
// nothing to gain by trying: the process is ending, so the memory, the
// audio device and the threads all go back to the OS anyway.
//
// Same instinct as `check::Watchdog` calling `std::_Exit` rather than
// unwinding, and as `RigController::stop()` detaching rather than
// joining: at teardown, *not running code* is the reliable option.
// Anything that genuinely has to happen before the process ends —
// dropping PTT, closing the capture stream — happens on the service's
// stop path, while the world is still standing.
Session& Session::instance() {
    static Session* s = new Session();
    return *s;
}

void Session::set_picture_dir(std::string dir) {
    std::lock_guard<std::mutex> lk(mu_);
    picture_dir_ = std::move(dir);
}

void Session::set_gallery_error(std::string message) {
    std::lock_guard<std::mutex> lk(gallery_mu_);
    gallery_error_ = std::move(message);
}

std::string Session::gallery_error() const {
    std::lock_guard<std::mutex> lk(gallery_mu_);
    return gallery_error_;
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

    // A reception that recovered after a fade comes back with the path
    // its first delivery went to (rx::Reception::saved_path): the PNG
    // and the sidecar beside it are rewritten in place, so one
    // transmission stays one picture. The sidecar's timestamp then has
    // to come from the *existing* name rather than from the clock, or a
    // replacement would date the reception to when the fade ended while
    // the filename beside it still said when it started.
    const bool replaced = r.saved_path.has_value();
    const std::string png = replaced ? *r.saved_path : dir + "/" + stamp + ".png";
    const std::string base = png.size() > 4 ? png.substr(0, png.size() - 4) : png;
    const std::string received =
        replaced ? std::filesystem::path(base).filename().string() : stamp;
    try {
        std::error_code ec;
        std::filesystem::create_directories(dir, ec);
        images::save_png(r.image, png);

        std::ostringstream meta;
        meta << "{\n";
        meta << "  \"received\": \"" << received << "\",\n";
        meta << "  \"callsign\": \"" << json_escape(r.callsign) << "\",\n";
        meta << "  \"mode\": " << (r.mode_name ? "\"" + json_escape(*r.mode_name) + "\""
                                               : "null")
             << ",\n";
        meta << "  \"snr_db\": " << r.snr_db << ",\n";
        meta << "  \"frames_received\": "
             << (r.frames_received ? std::to_string(*r.frames_received) : "null") << ",\n";
        // How much of the transmission arrived against how much of it
        // decoded: without the second the sidecar cannot say whether a
        // picture came through or merely went by.
        meta << "  \"frames_decoded\": "
             << (r.frames_decoded ? std::to_string(*r.frames_decoded) : "null") << ",\n";
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
    {
        std::lock_guard<std::mutex> lk(mu_);
        // Only a *new* picture is offered to the service.
        // take_saved_picture() is consuming so that each reception is
        // posted and exported exactly once; re-arming it for a
        // replacement would post a second notification and insert a
        // second MediaStore row for the same transmission, the file
        // having been overwritten in place. The summary is refreshed
        // either way -- it is a display of the best decode so far, not
        // an event.
        if (!replaced) saved_picture_ = png;
        std::ostringstream sum;
        sum << (r.callsign.empty() ? "unknown" : r.callsign.c_str());
        if (r.mode_name) sum << "  mode " << *r.mode_name;
        sum << "  " << std::fixed << std::setprecision(1) << r.snr_db << " dB";
        if (r.n_frames_expected.value_or(0) > 0) {
            sum << "  " << r.frames_received.value_or(0) << "/"
                << *r.n_frames_expected;
            if (r.frames_decoded) {
                sum << "  " << (100 * *r.frames_decoded / *r.n_frames_expected)
                    << "% decoded";
            }
        }
        saved_summary_ = sum.str();
    }
    return png;
}

std::optional<std::string> Session::take_saved_picture() {
    std::lock_guard<std::mutex> lk(mu_);
    std::optional<std::string> out = saved_picture_;
    saved_picture_.reset();
    return out;
}

std::string Session::last_saved_summary() const {
    std::lock_guard<std::mutex> lk(mu_);
    return saved_summary_;
}

// **Never runs in this application** -- see `instance()`, which leaks
// the singleton on purpose. Kept because it is the correct teardown if
// a Session is ever owned by something with a real lifetime, and
// because deleting it would make the leak look accidental rather than
// decided.
Session::~Session() {
    cancel_transmit();
    join_tx();
    stop();
    // Before the threads below, and deliberately: `stop_rig` detaches
    // rather than joining, so a wedged rig cannot hold up teardown --
    // but a *keyed* rig must be released, and the departing worker's
    // destructor is what does it.
    stop_rig();
    if (model_thread_.joinable()) model_thread_.join();
    if (encoder_thread_.joinable()) encoder_thread_.join();
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
            // Bundled artifacts first; the fetcher above is the
            // fallback, not the plan. On a build that carries them this
            // never touches the network, which is the whole point --
            // see assets.hpp.
            loaded->set_blob_resolver(assets::model_blob);
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
    device_ = device_name;

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
    // **Half, on a phone.** The desktop leaves this at 1.0 because its
    // decode is ~1% of the interval; here a decode is seconds, and at
    // the fixed interval a slow device ends up decoding back to back --
    // which starves the UI (Andrew: "decoding was pretty laggy") and
    // buys nothing, since each poll re-decodes a picture that has only
    // grown by one interval. A fast phone never reaches the cap and
    // polls every 5 s exactly as before; a slow one stretches out on
    // its own measured cost rather than on a constant guessed from
    // here. Anything not yet decoded is still in the ring buffer, so
    // backing off delays a picture rather than losing one.
    cfg.max_decode_duty = 0.5;
    cfg.blind_wide = blind_wide_.load();
    cfg.drift_track = drift_track_.load();
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

// --- transmit ---------------------------------------------------------

void Session::preload_encoder_async() {
    {
        std::lock_guard<std::mutex> lk(model_mu_);
        if (encoder_state_ == ModelState::Downloading ||
            encoder_state_ == ModelState::Loading || encoder_state_ == ModelState::Ready) {
            return;
        }
        encoder_state_ = ModelState::Loading;
        encoder_error_.clear();
    }
    if (encoder_thread_.joinable()) encoder_thread_.join();

    encoder_thread_ = std::thread([this] {
        install_java_fetcher([this](std::int64_t received, std::int64_t total) {
            std::lock_guard<std::mutex> lk(model_mu_);
            encoder_state_ = ModelState::Downloading;
            model_received_ = received;
            model_total_ = total;
        });

        // The *same* codec object the decoder half lives in, so the two
        // parts are cross-checked against one stamped `source_sha256`.
        // Building a second one for the encoder would run an encoder and
        // a decoder from different checkpoints without complaint, which
        // OnnxCodec exists partly to prevent.
        std::shared_ptr<codec::OnnxCodec> c;
        {
            std::lock_guard<std::mutex> lk(model_mu_);
            c = codec_;
        }
        std::string error;
        try {
            if (!c) {
                c = std::make_shared<codec::OnnxCodec>(
                    [](const std::string& part) { return checkpoint::resolve_onnx(part); });
                c->set_blob_resolver(assets::model_blob);
            }
            c->preload("encoder");
        } catch (const std::exception& e) {
            error = e.what();
            c.reset();
        }

        std::lock_guard<std::mutex> lk(model_mu_);
        if (c) codec_ = c;
        encoder_state_ = c ? ModelState::Ready : ModelState::Failed;
        encoder_error_ = error;
        model_received_ = 0;
        model_total_ = 0;
    });
}

ModelState Session::encoder_state() const {
    std::lock_guard<std::mutex> lk(model_mu_);
    return encoder_state_;
}

std::string Session::encoder_error() const {
    std::lock_guard<std::mutex> lk(model_mu_);
    return encoder_error_;
}

bool Session::transmitting() const {
    std::lock_guard<std::mutex> lk(tx_mu_);
    return tx_engine_ != nullptr;
}

tx::TxState Session::tx_state() const {
    std::lock_guard<std::mutex> lk(tx_mu_);
    return tx_state_;
}

void Session::cancel_transmit() {
    std::lock_guard<std::mutex> lk(tx_mu_);
    if (tx_engine_) tx_engine_->cancel();
}

void Session::join_tx() {
    if (tx_thread_.joinable()) tx_thread_.join();
}

void Session::stage_transmit(TxRequest request) {
    std::lock_guard<std::mutex> lk(tx_mu_);
    staged_ = std::move(request);
}

bool Session::start_staged_transmit() {
    std::optional<TxRequest> req;
    {
        std::lock_guard<std::mutex> lk(tx_mu_);
        // **Consumed, not merely read.** The service may be handed the
        // same intent twice -- Android redelivers, and a double tap
        // reaches it as two starts -- and a staged request that survived
        // its own transmission would put a second copy of the picture on
        // the air.
        req.swap(staged_);
    }
    if (!req) return false;
    return start_transmit(std::move(*req));
}

bool Session::start_transmit(TxRequest request) {
    {
        std::lock_guard<std::mutex> lk(tx_mu_);
        if (tx_engine_) return false;
    }
    {
        std::lock_guard<std::mutex> lk(model_mu_);
        if (encoder_state_ != ModelState::Ready || !codec_) return false;
    }
    join_tx();  // the previous over's thread, already finished

    tx_thread_ = std::thread([this, request] { run_transmit(request); });
    return true;
}

// --- rig control ------------------------------------------------------------

bool Session::start_rig(const rig::HamlibConfig& config,
                        std::shared_ptr<rig::SerialTransport> transport) {
#ifndef SSTVAE_ANDROID_HAVE_RIG
    (void)config;
    (void)transport;
    std::lock_guard<std::mutex> lk(rig_mu_);
    rig_status_ = "this build has no rig control";
    rig_failed_ = true;
    return false;
#else
    std::unique_ptr<rig::RigBackend> backend;
    try {
        // Everything that can be rejected on the spot is rejected here,
        // on the caller's thread, so a settings screen gets an answer
        // rather than a status line arriving later: an impossible
        // control-line combination, a device with nothing to open it, a
        // typed host that Hamlib will not dial.
        backend = rig::make_bridged_backend(
            config, std::move(transport),
            [](const rig::HamlibConfig& c) { return rig::make_hamlib_backend(c); });
    } catch (const std::exception& e) {
        std::lock_guard<std::mutex> lk(rig_mu_);
        rig_status_ = e.what();
        rig_failed_ = true;
        rig_can_key_ = false;
        return false;
    }

    // **The controller is created once and never destroyed.** Two
    // reasons, and the second is a bug this replaced.
    //
    // `ptt_function()` returns a lambda capturing the *controller*, and
    // `TxEngine` holds that for the length of an over. Destroying the
    // controller while a transmission is in flight -- which is what
    // turning rig control off mid-over used to do -- left the engine
    // calling into freed memory to bring PTT back down, at the one
    // moment it matters most.
    //
    // And `start()` supersedes a session in place, so re-starting the
    // same controller with a freshly built backend is exactly what a
    // reconnect needs: the `Ptt` already handed out stays valid across
    // it. `Session` is immortal for its own reasons; this rides along
    // with them.
    rig::RigController* controller = nullptr;
    {
        std::lock_guard<std::mutex> lk(rig_mu_);
        if (!rig_) {
            rig_ = std::make_unique<rig::RigController>(
                [this](std::optional<double> hz) {
                    std::lock_guard<std::mutex> lock(rig_mu_);
                    rig_frequency_ = hz;
                },
                [this](const std::string& text, bool error) {
                    std::lock_guard<std::mutex> lock(rig_mu_);
                    rig_status_ = text;
                    rig_failed_ = error;
                });
        }
        controller = rig_.get();
        rig_frequency_.reset();
        rig_status_ = "connecting";
        rig_failed_ = false;
        rig_can_key_ = config.ptt_method != rig::PttMethod::Vox;
    }

    rig::RigConfig rig_config;
    // Slower than the desktop's 5 s. A frequency readout is a comfort
    // on a phone rather than a working instrument -- there is no rig
    // panel competing for it -- and every poll is a CAT round trip plus
    // a wakeup on a device whose battery is the whole reason
    // `decode_loop_low_cpu` exists.
    rig_config.poll_interval_s = 10.0;

    // **Outside the lock, and that is not incidental.** `start()`
    // publishes "connecting" through the status callback, which takes
    // `rig_mu_`; calling it with the lock held would deadlock on the
    // first line of every connection attempt.
    controller->start(std::move(backend), rig_config);
    return true;
#endif
}

void Session::stop_rig() {
    rig::RigController* controller = nullptr;
    {
        std::lock_guard<std::mutex> lk(rig_mu_);
        controller = rig_.get();
        rig_frequency_.reset();
        rig_status_.clear();
        rig_failed_ = false;
        rig_can_key_ = false;
    }
    // Stopped, never destroyed -- see `start_rig`. `stop()` detaches
    // rather than joining, so this returns immediately even against a
    // rig that has stopped answering, which is the whole point of the
    // controller's design.
    if (controller) controller->stop();
}

bool Session::rig_running() const {
    std::lock_guard<std::mutex> lk(rig_mu_);
    return rig_ != nullptr && rig_->running();
}

std::optional<double> Session::rig_frequency_hz() const {
    std::lock_guard<std::mutex> lk(rig_mu_);
    return rig_frequency_;
}

std::string Session::rig_status() const {
    std::lock_guard<std::mutex> lk(rig_mu_);
    return rig_status_;
}

bool Session::rig_failed() const {
    std::lock_guard<std::mutex> lk(rig_mu_);
    return rig_failed_;
}

bool Session::rig_can_key() const {
    std::lock_guard<std::mutex> lk(rig_mu_);
    return rig_ != nullptr && rig_can_key_;
}

void Session::run_transmit(const TxRequest& request) {
    // **Capture stops before anything else happens**, including the
    // encode -- not because the encode would disturb it, but because a
    // phone's microphone is going to hear our own transmission and the
    // ring buffer must not contain it. `start()` builds a fresh
    // RingBuffer, so the resume below cannot replay our own audio back
    // into the decoder.
    std::string resume_device;
    bool resume_capture = false;
    {
        std::lock_guard<std::mutex> lk(mu_);
        resume_capture = stream_ != nullptr;
        resume_device = device_;
    }
    if (resume_capture) stop();

    std::shared_ptr<codec::OnnxCodec> model;
    {
        std::lock_guard<std::mutex> lk(model_mu_);
        model = codec_;
    }

    tx::TxConfig cfg;
    cfg.mode = request.mode;
    cfg.callsign = request.callsign;
    cfg.device = request.output_device;
    cfg.level = request.level;
    cfg.cw_id = request.cw_id;
    if (!request.cw_message.empty()) cfg.cw_message = request.cw_message;
    cfg.vox_lead_s = request.vox_lead_s;

    // **Keying, which this platform does have now.** `docs/android.md`
    // said it could not: Hamlib opens a device path and Android gives an
    // app none. It takes a socket for any backend instead (see
    // `core/rig/transport.hpp`), so what reaches `TxEngine` here is an
    // ordinary `Ptt` -- exactly the shape that file predicted CAT would
    // arrive in, and the prediction held: nothing above this line
    // changed to accept it.
    //
    // Null when the operator has no rig session, or has it set to VOX.
    // That is still a supported configuration and not a degraded one:
    // `PttWatchdog` with a null `Ptt` stands itself down, so there is no
    // special case below either way.
    tx::Ptt ptt;
    if (request.use_ptt) {
        std::lock_guard<std::mutex> lk(rig_mu_);
        if (rig_ && rig_can_key_) ptt = rig_->ptt_function();
    }
    // **The leader is not suppressed when the rig is keyed directly**
    // (Andrew, on hardware). An earlier version zeroed it here, on the
    // theory that a swept tone into an already-keyed radio only delays
    // the picture. That is the app second-guessing a setting the
    // operator made: the leader is also a settling period for an
    // interface that wants audio flowing before the radio is properly
    // in transmit, and `ptt_lead_s` is a different quantity doing a
    // different job. If it is not wanted with CAT keying, the operator
    // sets it to zero.

    auto engine = std::make_unique<tx::TxEngine>(
        std::move(ptt),
        [](const std::string& device, std::span<const double> wave, int samplerate,
           const std::function<void(double)>& on_progress,
           const std::function<bool()>& should_stop,
           const std::function<void(const std::string&)>& on_error) {
            return audio::android::play(device, wave, samplerate, on_progress,
                                        should_stop, on_error);
        },
        [model](const images::ImageArray& array) { return model->encode(array); },
        [this](const tx::TxState& s) {
            std::lock_guard<std::mutex> lk(tx_mu_);
            tx_state_ = s;
        },
        [this](const std::string& message) {
            std::lock_guard<std::mutex> lk(mu_);
            error_ = message;
        });

    tx::TxEngine* raw = engine.get();
    {
        std::lock_guard<std::mutex> lk(tx_mu_);
        tx_engine_ = std::move(engine);
        tx_state_ = tx::TxState{};
    }

    try {
        raw->transmit(request.picture, cfg);
    } catch (const std::exception& e) {
        std::lock_guard<std::mutex> lk(mu_);
        error_ = std::string("transmission failed: ") + e.what();
    }

    // The final state is read out *before* the engine is dropped, so a
    // UI polling across this moment sees Done or Failed rather than the
    // default-constructed Idle -- which would read as "nothing
    // happened" at exactly the point something did.
    {
        std::lock_guard<std::mutex> lk(tx_mu_);
        tx_state_ = raw->state();
        tx_engine_.reset();
    }

    if (resume_capture) start(resume_device);
}

}  // namespace sstvae::androidapp

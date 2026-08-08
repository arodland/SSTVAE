#include "session.hpp"

#include "config.hpp"
#include "images/types.hpp"

namespace sstvae::androidapp {

Session& Session::instance() {
    static Session s;
    return s;
}

Session::~Session() { stop(); }

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

    // No codec yet. `rx::Decoder` is a seam, so the whole state machine
    // -- acquisition, the beacon, frame accounting, SNR -- runs and
    // reports without one, which is what let this layer be tested
    // before the model fetch existed.
    rx::Decoder decoder = [](std::span<const double>, std::span<const double>) {
        return images::Picture{};
    };
    rx::RxConfig cfg;
    cfg.poll_interval = 5.0;
    rx::Sink sink = [](const rx::Reception&) -> std::optional<std::string> {
        return std::nullopt;
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

}  // namespace sstvae::androidapp

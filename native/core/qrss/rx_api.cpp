// QRSSTVAE receive, in process (rx_api.hpp).

#include "qrss/rx_api.hpp"

#include <complex>
#include <optional>
#include <span>
#include <vector>

#include "qrss/receiver.hpp"
#include "qrss/receiver_slot.hpp"

namespace sstvae::qrss::api {

Capture capture_from_audio(std::span<const double> audio_8k, std::int64_t q, std::optional<double> start_unix) {
    return receiver::capture_from_audio(audio_8k, q, start_unix);
}

Capture capture_from_audio(std::span<const float> audio_8k, std::int64_t q, std::optional<double> start_unix) {
    const std::vector<double> x(audio_8k.begin(), audio_8k.end());
    return receiver::capture_from_audio(x, q, start_unix);
}

Capture capture_from_wav(const std::string& path, std::int64_t q, std::optional<double> start_unix) {
    return receiver::capture_from_wav(path, q, start_unix);
}

std::vector<types::PassResult> receive_slot(const Capture& capture, const ReceiveConfig& config) {
    const bool old_edge = config.old_edge_noise;
    receiver::SlotSeams seams;
    // every noise table the stages fill, with the EDGE_NOISE rule this call's
    // (chan.cpp's default_noise_psd and acquire_a12's lead-in default, the
    // rule given rather than read)
    seams.pass.noise_psd = [old_edge](std::span<const std::complex<float>> ch, std::int64_t fs,
                                      std::span<const double> exclude_hz, double win_s,
                                      std::optional<double> audio_hz) {
        frontend::NoisePsdOptions o;
        o.audio_hz = audio_hz;
        o.old_edge_noise = old_edge;
        return frontend::noise_psd(ch, fs, exclude_hz, win_s, o);
    };
    seams.acquire.noise_psd = [old_edge](std::span<const std::complex<float>> ch, std::int64_t fs, std::int64_t fs_in,
                                         double audio_hz) {
        frontend::NoisePsdOptions o;
        o.fs_in = fs_in;
        o.audio_hz = audio_hz;
        o.old_edge_noise = old_edge;
        return frontend::noise_psd(ch, fs, {}, 10.0, o);
    };
    seams.old_gating = config.old_gating;
    receiver::SlotOptions opt;
    opt.live_only = config.live_only;
    opt.estimator = config.estimator;
    opt.erase_s = config.erase_s;
    opt.round_b = config.round_b;
    opt.max_candidates = config.max_candidates;
    opt.budget_s = config.budget_s;
    opt.hints = config.hints;
    return receiver::receive_slot(receiver::CaptureRef(&capture), config.frames, opt, seams);
}

}  // namespace sstvae::qrss::api

#include "audio/wavio.hpp"

#include <cmath>
#include <numeric>
#include <stdexcept>

#include "config.hpp"
#include "dsp/dsp.hpp"

#define DR_WAV_IMPLEMENTATION
#include "dr_wav.h"

namespace sstvae::audio {

std::vector<double> read_wav(const std::string& path) {
    drwav wav;
    if (!drwav_init_file(&wav, path.c_str(), nullptr)) {
        throw std::runtime_error("cannot read " + path + ": not a readable WAV");
    }
    const unsigned channels = wav.channels;
    const unsigned rate = wav.sampleRate;
    const std::size_t frames = static_cast<std::size_t>(wav.totalPCMFrameCount);

    // dr_wav's f32 reader applies each format's own full-scale
    // normalization -- int16 by 32768, int24 by 2^23, and so on -- so
    // samples arrive in [-1, 1] whatever the file's encoding was. That
    // is the "scale before the mixdown" rule satisfied by construction:
    // the conversion happens per sample, before anything is averaged.
    std::vector<float> interleaved(frames * channels);
    const std::size_t got =
        static_cast<std::size_t>(drwav_read_pcm_frames_f32(&wav, frames, interleaved.data()));
    drwav_uninit(&wav);
    if (got != frames) interleaved.resize(got * channels);

    std::vector<double> mono(got);
    for (std::size_t i = 0; i < got; ++i) {
        double acc = 0.0;
        for (unsigned c = 0; c < channels; ++c) {
            acc += static_cast<double>(interleaved[i * channels + c]);
        }
        mono[i] = acc / static_cast<double>(channels);
    }

    if (rate != static_cast<unsigned>(config::FS)) {
        const int g = static_cast<int>(std::gcd(static_cast<int>(config::FS),
                                                static_cast<int>(rate)));
        mono = dsp::resample_poly(mono, config::FS / g, static_cast<int>(rate) / g);
    }
    return mono;
}

void write_wav_float(const std::string& path, std::span<const double> x) {
    drwav_data_format fmt{};
    fmt.container = drwav_container_riff;
    fmt.format = DR_WAVE_FORMAT_IEEE_FLOAT;
    fmt.channels = 1;
    fmt.sampleRate = static_cast<drwav_uint32>(config::FS);
    fmt.bitsPerSample = 32;

    drwav wav;
    if (!drwav_init_file_write(&wav, path.c_str(), &fmt, nullptr)) {
        throw std::runtime_error("cannot write " + path);
    }
    std::vector<float> f(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) f[i] = static_cast<float>(x[i]);
    drwav_write_pcm_frames(&wav, f.size(), f.data());
    drwav_uninit(&wav);
}

void write_wav(const std::string& path, std::span<const double> x, double peak) {
    double m = 0.0;
    for (double v : x) m = std::max(m, std::abs(v));

    std::vector<std::int16_t> q(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) {
        const double v = (m > 0) ? x[i] / m * peak : x[i];
        // np.round is half-to-even; std::round is not.
        q[i] = static_cast<std::int16_t>(std::nearbyint(v * 32767.0));
    }

    drwav_data_format fmt{};
    fmt.container = drwav_container_riff;
    fmt.format = DR_WAVE_FORMAT_PCM;
    fmt.channels = 1;
    fmt.sampleRate = static_cast<drwav_uint32>(config::FS);
    fmt.bitsPerSample = 16;

    drwav wav;
    if (!drwav_init_file_write(&wav, path.c_str(), &fmt, nullptr)) {
        throw std::runtime_error("cannot write " + path);
    }
    drwav_write_pcm_frames(&wav, q.size(), q.data());
    drwav_uninit(&wav);
}

}  // namespace sstvae::audio

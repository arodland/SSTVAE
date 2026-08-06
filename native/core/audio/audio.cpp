#include "audio/audio.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <stdexcept>

#include "dsp/dsp.hpp"

namespace sstvae::audio {

namespace {

// Ceiling division for positive values.
std::size_t ceil_div(std::size_t a, std::size_t b) { return (a + b - 1) / b; }

struct Scaling {
    double scale;   // divisor when reading
    double offset;  // zero point
    double lo;      // clip bounds in the integer domain
    double hi;
};

Scaling scaling(SampleFormat f) {
    switch (f) {
        case SampleFormat::Float:
            return {1.0, 0.0, -1.0, 1.0};
        case SampleFormat::Int16:
            return {32768.0, 0.0, -32768.0, 32767.0};
        case SampleFormat::Int32:
            return {2147483648.0, 0.0, -2147483648.0, 2147483647.0};
        case SampleFormat::UInt8:
            return {128.0, 128.0, 0.0, 255.0};
    }
    throw std::invalid_argument("unknown sample format");
}

// Read one sample of `fmt` from `p`, as a raw (unscaled) double.
double read_raw(const std::byte* p, SampleFormat f) {
    switch (f) {
        case SampleFormat::Float: {
            float v;
            std::memcpy(&v, p, sizeof v);
            return static_cast<double>(v);
        }
        case SampleFormat::Int16: {
            std::int16_t v;
            std::memcpy(&v, p, sizeof v);
            return static_cast<double>(v);
        }
        case SampleFormat::Int32: {
            std::int32_t v;
            std::memcpy(&v, p, sizeof v);
            return static_cast<double>(v);
        }
        case SampleFormat::UInt8:
            return static_cast<double>(std::to_integer<std::uint8_t>(*p));
    }
    throw std::invalid_argument("unknown sample format");
}

void write_raw(std::byte* p, SampleFormat f, double v) {
    switch (f) {
        case SampleFormat::Float: {
            const auto x = static_cast<float>(v);
            std::memcpy(p, &x, sizeof x);
            return;
        }
        case SampleFormat::Int16: {
            const auto x = static_cast<std::int16_t>(v);
            std::memcpy(p, &x, sizeof x);
            return;
        }
        case SampleFormat::Int32: {
            const auto x = static_cast<std::int32_t>(v);
            std::memcpy(p, &x, sizeof x);
            return;
        }
        case SampleFormat::UInt8:
            *p = static_cast<std::byte>(static_cast<std::uint8_t>(v));
            return;
    }
    throw std::invalid_argument("unknown sample format");
}

std::string lowered(std::string_view s) {
    std::string out(s);
    std::transform(out.begin(), out.end(), out.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return out;
}

}  // namespace

Ratio resample_ratio(int src_rate, int dst_rate) {
    if (src_rate <= 0 || dst_rate <= 0) {
        throw std::invalid_argument("resample_ratio: rates must be positive");
    }
    const int g = std::gcd(src_rate, dst_rate);
    return {dst_rate / g, src_rate / g};
}

StreamResampler::StreamResampler(int up, int down) : up_(up), down_(down) {
    if (up <= 0 || down <= 0) {
        throw std::invalid_argument("StreamResampler: up and down must be positive");
    }
    // The default filter is 20*max(up, down)+1 taps in the *upsampled*
    // domain; this is its span measured in input samples.
    const std::size_t span =
        ceil_div(static_cast<std::size_t>(20 * std::max(up, down) + 1),
                 static_cast<std::size_t>(up));
    // Rounded up to a whole number of `down`, so the samples skipped at
    // the head are an exact integer and no phase error accumulates.
    pad_ = ceil_div(span, static_cast<std::size_t>(down)) * static_cast<std::size_t>(down);
    buf_.assign(pad_, 0.0);
}

std::vector<double> StreamResampler::operator()(std::span<const double> chunk) {
    buf_.insert(buf_.end(), chunk.begin(), chunk.end());

    const std::size_t d = static_cast<std::size_t>(down_);
    const std::size_t usable = buf_.size() > 2 * pad_ ? buf_.size() - 2 * pad_ : 0;
    const std::size_t n = (usable / d) * d;
    if (n == 0) return {};

    const std::vector<double> y = dsp::resample_poly(
        std::span<const double>(buf_.data(), 2 * pad_ + n), up_, down_, &filter_taps_);
    const std::size_t skip = pad_ * static_cast<std::size_t>(up_) / d;
    const std::size_t take = n * static_cast<std::size_t>(up_) / d;

    buf_.erase(buf_.begin(), buf_.begin() + static_cast<std::ptrdiff_t>(n));
    return std::vector<double>(y.begin() + static_cast<std::ptrdiff_t>(skip),
                               y.begin() + static_cast<std::ptrdiff_t>(skip + take));
}

std::optional<SampleFormat> sample_format_from_name(std::string_view name) {
    if (name == "Float") return SampleFormat::Float;
    if (name == "Int16") return SampleFormat::Int16;
    if (name == "Int32") return SampleFormat::Int32;
    if (name == "UInt8") return SampleFormat::UInt8;
    return std::nullopt;
}

const char* sample_format_name(SampleFormat f) {
    switch (f) {
        case SampleFormat::Float: return "Float";
        case SampleFormat::Int16: return "Int16";
        case SampleFormat::Int32: return "Int32";
        case SampleFormat::UInt8: return "UInt8";
    }
    return "?";
}

int bytes_per_sample(SampleFormat f) {
    switch (f) {
        case SampleFormat::Float: return 4;
        case SampleFormat::Int16: return 2;
        case SampleFormat::Int32: return 4;
        case SampleFormat::UInt8: return 1;
    }
    return 0;
}

std::vector<double> bytes_to_mono(std::span<const std::byte> raw, SampleFormat fmt,
                                  int channels) {
    if (channels < 1) throw std::invalid_argument("bytes_to_mono: channels < 1");
    const auto bps = static_cast<std::size_t>(bytes_per_sample(fmt));
    const std::size_t frame = bps * static_cast<std::size_t>(channels);
    // A trailing partial frame is dropped: keeping it would misalign
    // every sample after it, which sounds like noise rather than an
    // error.
    const std::size_t frames = raw.size() / frame;
    const Scaling s = scaling(fmt);

    std::vector<double> out(frames);
    for (std::size_t i = 0; i < frames; ++i) {
        double acc = 0.0;
        for (int c = 0; c < channels; ++c) {
            acc += read_raw(raw.data() + i * frame + static_cast<std::size_t>(c) * bps,
                            fmt);
        }
        out[i] = (acc / channels - s.offset) / s.scale;
    }
    return out;
}

std::vector<std::byte> mono_to_bytes(std::span<const double> x, SampleFormat fmt,
                                     int channels) {
    if (channels < 1) throw std::invalid_argument("mono_to_bytes: channels < 1");
    const auto bps = static_cast<std::size_t>(bytes_per_sample(fmt));
    const std::size_t frame = bps * static_cast<std::size_t>(channels);
    const Scaling s = scaling(fmt);

    std::vector<std::byte> out(x.size() * frame);
    for (std::size_t i = 0; i < x.size(); ++i) {
        double v;
        if (fmt == SampleFormat::Float) {
            v = std::clamp(x[i], -1.0, 1.0);
        } else {
            // Scale by `scale - 1` and clip in the integer domain, so
            // exactly +1.0 lands on the positive full-scale code rather
            // than wrapping to the negative one. Half-to-even rounding,
            // like numpy's.
            v = std::clamp(std::nearbyint(x[i] * (s.scale - 1.0) + s.offset), s.lo, s.hi);
        }
        for (int c = 0; c < channels; ++c) {
            write_raw(out.data() + i * frame + static_cast<std::size_t>(c) * bps, fmt, v);
        }
    }
    return out;
}

std::optional<std::size_t> match_device(std::span<const std::string> descriptions,
                                        std::string_view wanted) {
    if (wanted.empty()) return std::nullopt;
    for (std::size_t i = 0; i < descriptions.size(); ++i) {
        if (descriptions[i] == wanted) return i;
    }
    const std::string low = lowered(wanted);
    std::optional<std::size_t> hit;
    for (std::size_t i = 0; i < descriptions.size(); ++i) {
        if (lowered(descriptions[i]).find(low) == std::string::npos) continue;
        if (hit) return std::nullopt;  // ambiguous: prefer the default
        hit = i;
    }
    return hit;
}

}  // namespace sstvae::audio

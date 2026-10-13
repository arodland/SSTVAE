// sstvae.wavio.read_wav over scipy.io.wavfile.read (wavread.hpp).

#include "qrss/wavread.hpp"

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <numeric>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "dsp/dsp.hpp"
#include "qrss/config.hpp"

namespace sstvae::qrss::wav {

namespace {

constexpr std::uint16_t FORMAT_PCM = 0x0001;
constexpr std::uint16_t FORMAT_IEEE_FLOAT = 0x0003;
constexpr std::uint16_t FORMAT_EXTENSIBLE = 0xFFFE;

std::string errno_text(int err) {
    return std::string("[Errno ") + std::to_string(err) + "] " + std::strerror(err);
}

[[noreturn]] void unsupported(const std::string& why) { throw std::invalid_argument("unsupported WAV: " + why); }

std::uint32_t u32(const std::uint8_t* p) {
    return static_cast<std::uint32_t>(p[0]) | static_cast<std::uint32_t>(p[1]) << 8 |
           static_cast<std::uint32_t>(p[2]) << 16 | static_cast<std::uint32_t>(p[3]) << 24;
}
std::uint16_t u16(const std::uint8_t* p) {
    return static_cast<std::uint16_t>(static_cast<std::uint16_t>(p[0]) | static_cast<std::uint16_t>(p[1]) << 8);
}

struct Fmt {
    std::uint16_t tag = 0;
    std::uint16_t channels = 0;
    std::uint32_t rate = 0;
    std::uint16_t block_align = 0;
    std::uint16_t bits = 0;
};

}  // namespace

ReadError::ReadError(const std::string& p, int e) : std::runtime_error(errno_text(e) + ": '" + p + "'"), path(p), err(e) {}

std::vector<double> read_wav(const std::string& path) {
    std::unique_ptr<std::FILE, int (*)(std::FILE*)> f(std::fopen(path.c_str(), "rb"), &std::fclose);
    if (!f) throw ReadError(path, errno);
    std::vector<std::uint8_t> bytes;
    {
        std::uint8_t buf[1 << 16];
        std::size_t n;
        while ((n = std::fread(buf, 1, sizeof buf, f.get())) > 0) bytes.insert(bytes.end(), buf, buf + n);
        if (std::ferror(f.get())) throw ReadError(path, errno ? errno : EIO);
    }
    if (bytes.size() < 12 || std::memcmp(bytes.data(), "RIFF", 4) != 0 || std::memcmp(bytes.data() + 8, "WAVE", 4) != 0)
        unsupported("not a little-endian RIFF/WAVE file");
    std::size_t at = 12;
    std::optional<Fmt> fmt;
    const std::uint8_t* data = nullptr;
    std::size_t data_size = 0;
    while (at + 8 <= bytes.size()) {
        const std::uint8_t* h = bytes.data() + at;
        const std::uint32_t size = u32(h + 4);
        const std::size_t body = at + 8;
        if (std::memcmp(h, "fmt ", 4) == 0) {
            if (size < 16 || body + 16 > bytes.size()) unsupported("a short fmt chunk");
            const std::uint8_t* b = bytes.data() + body;
            Fmt m;
            m.tag = u16(b);
            m.channels = u16(b + 2);
            m.rate = u32(b + 4);
            m.block_align = u16(b + 12);
            m.bits = u16(b + 14);
            if (m.tag == FORMAT_EXTENSIBLE) {
                if (size < 40 || body + 40 > bytes.size()) unsupported("a short extensible fmt chunk");
                m.tag = u16(b + 24);   // the subformat GUID's first two bytes
            }
            fmt = m;
        } else if (std::memcmp(h, "data", 4) == 0) {
            if (!fmt) unsupported("a data chunk before the fmt chunk");
            data = bytes.data() + body;
            data_size = std::min<std::size_t>(size, bytes.size() - body);
            break;
        }
        at = body + size + (size % 2);
    }
    if (!fmt) unsupported("no fmt chunk");
    if (!data) unsupported("no data chunk");
    const Fmt& m = *fmt;
    if (m.channels == 0 || m.block_align == 0 || m.block_align % m.channels != 0) unsupported("a bad block alignment");
    // numpy's mean over 2..8 channels is a running sum; past 8 it sums the
    // rest pairwise, which nothing here needs
    if (m.channels > 8) unsupported(std::to_string(m.channels) + " channels (8 at most)");
    const std::size_t bps = m.block_align / m.channels;   // bytes per sample (container)
    const std::size_t n = data_size / bps;                // whole samples, as np.fromfile counts them
    if (n % m.channels != 0) unsupported("a data chunk that is not whole frames");
    std::vector<double> x(n);
    if (m.tag == FORMAT_PCM) {
        if (m.bits >= 1 && m.bits <= 8) {
            if (bps != 1) unsupported("8-bit samples in a wider container");
            for (std::size_t i = 0; i < n; ++i) x[i] = static_cast<double>(data[i]) / 255.0;
        } else if (bps == 2) {
            for (std::size_t i = 0; i < n; ++i)
                x[i] = static_cast<double>(static_cast<std::int16_t>(u16(data + 2 * i))) / 32767.0;
        } else if (bps == 3) {
            for (std::size_t i = 0; i < n; ++i) {
                const std::uint8_t* s = data + 3 * i;
                const std::uint32_t v = static_cast<std::uint32_t>(s[0]) << 8 | static_cast<std::uint32_t>(s[1]) << 16 |
                                        static_cast<std::uint32_t>(s[2]) << 24;
                x[i] = static_cast<double>(static_cast<std::int32_t>(v)) / 2147483647.0;
            }
        } else if (bps == 4) {
            for (std::size_t i = 0; i < n; ++i)
                x[i] = static_cast<double>(static_cast<std::int32_t>(u32(data + 4 * i))) / 2147483647.0;
        } else {
            unsupported(std::to_string(m.bits) + "-bit integer samples in " + std::to_string(bps) + " bytes");
        }
    } else if (m.tag == FORMAT_IEEE_FLOAT) {
        if (m.bits == 32 && bps == 4) {
            for (std::size_t i = 0; i < n; ++i) {
                const std::uint32_t v = u32(data + 4 * i);
                float s;
                std::memcpy(&s, &v, 4);
                x[i] = static_cast<double>(s);
            }
        } else if (m.bits == 64 && bps == 8) {
            for (std::size_t i = 0; i < n; ++i) {
                const std::uint64_t v = static_cast<std::uint64_t>(u32(data + 8 * i)) |
                                        static_cast<std::uint64_t>(u32(data + 8 * i + 4)) << 32;
                double s;
                std::memcpy(&s, &v, 8);
                x[i] = s;
            }
        } else {
            unsupported(std::to_string(m.bits) + "-bit floating-point samples");
        }
    } else {
        unsupported("format tag " + std::to_string(m.tag));
    }
    if (m.channels > 1) {
        const std::size_t frames = n / m.channels;
        std::vector<double> mono(frames);
        for (std::size_t i = 0; i < frames; ++i) {
            double acc = x[i * m.channels];
            for (std::size_t c = 1; c < m.channels; ++c) acc += x[i * m.channels + c];
            mono[i] = acc / static_cast<double>(m.channels);
        }
        x = std::move(mono);
    }
    if (m.rate != static_cast<std::uint32_t>(config::FS)) {
        const auto rate = static_cast<std::int64_t>(m.rate);
        const std::int64_t g = std::gcd(static_cast<std::int64_t>(config::FS), rate);
        if (rate == 0) unsupported("a sample rate of 0");
        x = dsp::resample_poly(std::span<const double>(x), static_cast<int>(config::FS / g), static_cast<int>(rate / g));
    }
    return x;
}

}  // namespace sstvae::qrss::wav

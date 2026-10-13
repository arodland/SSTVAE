#include "qrss/wav.hpp"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace sstvae::qrss::wav {

namespace {

// Little-endian packing, as struct.pack('<...').
void put_u16(std::vector<std::uint8_t>& b, std::uint32_t v) {
    b.push_back(static_cast<std::uint8_t>(v & 0xFF));
    b.push_back(static_cast<std::uint8_t>((v >> 8) & 0xFF));
}
void put_u32(std::vector<std::uint8_t>& b, std::uint64_t v) {
    for (int i = 0; i < 4; ++i) b.push_back(static_cast<std::uint8_t>((v >> (8 * i)) & 0xFF));
}
void put_u64(std::vector<std::uint8_t>& b, std::uint64_t v) {
    for (int i = 0; i < 8; ++i) b.push_back(static_cast<std::uint8_t>((v >> (8 * i)) & 0xFF));
}
void put_tag(std::vector<std::uint8_t>& b, const char* tag) { b.insert(b.end(), tag, tag + 4); }

std::uint64_t itemsize(Format f) { return f == Format::Int16 ? 2 : 4; }

std::string errno_text(const std::string& path, int err) {
    return "[Errno " + std::to_string(err) + "] " + std::strerror(err) + ": '" + path + "'";
}

}  // namespace

WriteError::WriteError(const std::string& p, int e) : std::runtime_error(errno_text(p, e)), path(p), err(e) {}

std::vector<std::uint8_t> header(Format format, std::int64_t n, std::int64_t rate) {
    if (n < 0) throw std::invalid_argument("wav::header: negative sample count");
    if (rate <= 0 || rate > 0xFFFFFFFF) throw std::invalid_argument("wav::header: rate out of range");
    const bool pcm = format == Format::Int16;
    const std::uint64_t bytes = itemsize(format);
    const std::uint64_t nbytes = static_cast<std::uint64_t>(n) * bytes;
    // fmt: format tag (1 PCM, 3 IEEE float), channels, rate, bytes/s,
    // block align, bits; non-PCM adds cbSize = 0.
    std::vector<std::uint8_t> fmt;
    put_u16(fmt, pcm ? 1 : 3);
    put_u16(fmt, 1);
    put_u32(fmt, static_cast<std::uint64_t>(rate));
    put_u32(fmt, static_cast<std::uint64_t>(rate) * bytes);
    put_u16(fmt, static_cast<std::uint32_t>(bytes));
    put_u16(fmt, static_cast<std::uint32_t>(8 * bytes));
    if (!pcm) put_u16(fmt, 0);

    std::vector<std::uint8_t> h;
    // RIFF, WAVE and the fmt chunk; the data chunk's tag and size follow.
    const std::uint64_t riff_header = 4 + 4 + 4 + 4 + 4 + fmt.size();
    const std::uint64_t fact = pcm ? 0 : 12;
    // scipy: (len(header) + 4 + 4 + data.nbytes) - 8 > 0xFFFFFFFF -> RF64,
    // where header is RIFF..fmt (before fact).
    const bool rf64 = riff_header + 8 + nbytes - 8 > 0xFFFFFFFFULL;
    // The RIFF size scipy patches in afterwards: the whole file less 8.
    if (!rf64) {
        put_tag(h, "RIFF");
        put_u32(h, riff_header + fact + 8 + nbytes - 8);
        put_tag(h, "WAVE");
    } else {
        const std::uint64_t file = 12 + 36 + 8 + fmt.size() + fact + 8 + nbytes;
        put_tag(h, "RF64");
        put_u32(h, 0xFFFFFFFFULL);
        put_tag(h, "WAVE");
        put_tag(h, "ds64");
        put_u32(h, 28);
        put_u64(h, file - 8);
        put_u64(h, nbytes);
        put_u64(h, static_cast<std::uint64_t>(n));
        put_u32(h, 0);
    }
    put_tag(h, "fmt ");
    put_u32(h, fmt.size());
    h.insert(h.end(), fmt.begin(), fmt.end());
    if (!pcm) {
        put_tag(h, "fact");
        put_u32(h, 4);
        put_u32(h, static_cast<std::uint64_t>(n) & 0xFFFFFFFFULL);   // struct.pack('<I'): n < 2^32 here
    }
    put_tag(h, "data");
    put_u32(h, std::min<std::uint64_t>(nbytes, 0xFFFFFFFFULL));
    return h;
}

double peak_of(std::span<const double> x, double running) {
    // np.max propagates NaN; |x| of anything else compares as a float.
    for (const double v : x) {
        const double a = std::fabs(v);
        if (std::isnan(a) || std::isnan(running)) {
            running = std::numeric_limits<double>::quiet_NaN();
        } else if (a > running) {
            running = a;
        }
    }
    return running;
}

double peak_of(std::span<const double> x) {
    if (x.empty()) throw std::invalid_argument("zero-size array to reduction operation maximum which has no identity");
    return peak_of(x, 0.0);
}

std::int16_t to_int16(double x, double m, double peak) {
    const double v = (m > 0) ? x / m * peak : x;
    const double r = std::nearbyint(v * 32767.0);   // np.round: half to even
    if (std::isnan(r)) return 0;
    // astype(int16) is unspecified past the range; normalised audio never
    // gets there, and here it saturates.
    const double c = std::clamp(r, -32768.0, 32767.0);
    return static_cast<std::int16_t>(c);
}

Writer::Writer(const std::string& path, Format format, std::int64_t n, std::int64_t rate)
    : path_(path), format_(format), n_(n) {
    const std::vector<std::uint8_t> h = header(format, n, rate);
    f_ = std::fopen(path.c_str(), "wb");
    if (!f_) throw WriteError(path, errno);
    put(h.data(), h.size());
}

Writer::~Writer() {
    if (f_) std::fclose(f_);
}

void Writer::put(const void* p, std::size_t bytes) {
    if (!f_) throw std::logic_error("wav::Writer: closed");
    if (bytes != 0 && std::fwrite(p, 1, bytes, f_) != bytes) throw WriteError(path_, errno);
}

void Writer::write(std::span<const std::int16_t> x) {
    if (format_ != Format::Int16) throw std::logic_error("wav::Writer: int16 samples for a float32 file");
    if (static_cast<std::int64_t>(x.size()) > n_ - written_) throw std::logic_error("wav::Writer: past the sample count");
    std::vector<std::uint8_t> b(2 * x.size());
    for (std::size_t i = 0; i < x.size(); ++i) {
        const auto u = static_cast<std::uint16_t>(x[i]);
        b[2 * i] = static_cast<std::uint8_t>(u & 0xFF);
        b[2 * i + 1] = static_cast<std::uint8_t>(u >> 8);
    }
    put(b.data(), b.size());
    written_ += static_cast<std::int64_t>(x.size());
}

void Writer::write(std::span<const float> x) {
    if (format_ != Format::Float32) throw std::logic_error("wav::Writer: float32 samples for an int16 file");
    if (static_cast<std::int64_t>(x.size()) > n_ - written_) throw std::logic_error("wav::Writer: past the sample count");
    std::vector<std::uint8_t> b(4 * x.size());
    for (std::size_t i = 0; i < x.size(); ++i) {
        std::uint32_t u = 0;
        std::memcpy(&u, &x[i], 4);
        for (int k = 0; k < 4; ++k) b[4 * i + static_cast<std::size_t>(k)] = static_cast<std::uint8_t>((u >> (8 * k)) & 0xFF);
    }
    put(b.data(), b.size());
    written_ += static_cast<std::int64_t>(x.size());
}

void Writer::close() {
    if (!f_) return;
    if (written_ != n_) {
        std::fclose(f_);
        f_ = nullptr;
        throw std::logic_error("wav::Writer: " + std::to_string(written_) + " of " + std::to_string(n_) +
                               " samples written");
    }
    const bool flushed = std::fflush(f_) == 0;
    const int err = errno;
    const bool closed = std::fclose(f_) == 0;
    f_ = nullptr;
    if (!flushed || !closed) throw WriteError(path_, flushed ? errno : err);
}

void write_int16(const std::string& path, std::span<const double> x, double peak) {
    const double m = peak_of(x);
    Writer w(path, Format::Int16, static_cast<std::int64_t>(x.size()));
    constexpr std::size_t BLOCK = 1 << 16;
    std::vector<std::int16_t> q;
    for (std::size_t a = 0; a < x.size(); a += BLOCK) {
        const std::size_t len = std::min(BLOCK, x.size() - a);
        q.resize(len);
        for (std::size_t i = 0; i < len; ++i) q[i] = to_int16(x[a + i], m, peak);
        w.write(std::span<const std::int16_t>(q));
    }
    w.close();
}

void write_float32(const std::string& path, std::span<const double> x) {
    Writer w(path, Format::Float32, static_cast<std::int64_t>(x.size()));
    constexpr std::size_t BLOCK = 1 << 16;
    std::vector<float> f;
    for (std::size_t a = 0; a < x.size(); a += BLOCK) {
        const std::size_t len = std::min(BLOCK, x.size() - a);
        f.resize(len);
        for (std::size_t i = 0; i < len; ++i) f[i] = static_cast<float>(x[a + i]);
        w.write(std::span<const float>(f));
    }
    w.close();
}

}  // namespace sstvae::qrss::wav

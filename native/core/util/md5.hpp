// MD5 (RFC 1321), for the one place a format asks for it: a FLAC
// STREAMINFO block carries the MD5 of the unencoded samples, and
// soundfile/libFLAC (and `flac -t`) check a file against it.
//
// Not a security primitive and not used as one. Header-only, like
// fileio.hpp, because sstvae_core lists its sources by hand while
// sstvae_qrss is the one that wants it. Incremental: update() any number
// of times, then digest() once. A Md5 object is a value with no shared
// state, so separate objects may run on separate threads.

#ifndef SSTVAE_UTIL_MD5_HPP
#define SSTVAE_UTIL_MD5_HPP

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <string_view>

namespace sstvae::util {

class Md5 {
public:
    using Digest = std::array<std::uint8_t, 16>;

    void update(const void* data, std::size_t n) {
        const auto* p = static_cast<const std::uint8_t*>(data);
        bytes_ += n;
        if (fill_) {
            const std::size_t take = (n < 64 - fill_) ? n : 64 - fill_;
            std::memcpy(buf_ + fill_, p, take);
            fill_ += take;
            p += take;
            n -= take;
            if (fill_ < 64) return;
            block(buf_);
            fill_ = 0;
        }
        for (; n >= 64; p += 64, n -= 64) block(p);
        if (n) {
            std::memcpy(buf_, p, n);
            fill_ = n;
        }
    }

    void update(std::string_view s) { update(s.data(), s.size()); }

    // The digest of everything updated so far. Call once: it pads in place.
    Digest digest() {
        const std::uint64_t bits = bytes_ * 8u;
        static constexpr std::uint8_t pad[64] = {0x80};
        const std::size_t r = static_cast<std::size_t>(bytes_ % 64);
        update(pad, r < 56 ? 56 - r : 120 - r);
        std::uint8_t len[8];
        for (int i = 0; i < 8; ++i) len[i] = static_cast<std::uint8_t>(bits >> (8 * i));
        update(len, 8);
        Digest out{};
        for (int i = 0; i < 4; ++i)
            for (int k = 0; k < 4; ++k) out[4 * i + k] = static_cast<std::uint8_t>(h_[i] >> (8 * k));
        return out;
    }

    static std::string hex(const Digest& d) {
        static constexpr char digits[] = "0123456789abcdef";
        std::string s;
        for (const std::uint8_t b : d) {
            s += digits[b >> 4];
            s += digits[b & 15];
        }
        return s;
    }

private:
    static std::uint32_t rotl(std::uint32_t x, int c) { return (x << c) | (x >> (32 - c)); }

    void block(const std::uint8_t* p) {
        static constexpr std::uint32_t K[64] = {
            0xd76aa478, 0xe8c7b756, 0x242070db, 0xc1bdceee, 0xf57c0faf, 0x4787c62a, 0xa8304613, 0xfd469501,
            0x698098d8, 0x8b44f7af, 0xffff5bb1, 0x895cd7be, 0x6b901122, 0xfd987193, 0xa679438e, 0x49b40821,
            0xf61e2562, 0xc040b340, 0x265e5a51, 0xe9b6c7aa, 0xd62f105d, 0x02441453, 0xd8a1e681, 0xe7d3fbc8,
            0x21e1cde6, 0xc33707d6, 0xf4d50d87, 0x455a14ed, 0xa9e3e905, 0xfcefa3f8, 0x676f02d9, 0x8d2a4c8a,
            0xfffa3942, 0x8771f681, 0x6d9d6122, 0xfde5380c, 0xa4beea44, 0x4bdecfa9, 0xf6bb4b60, 0xbebfbc70,
            0x289b7ec6, 0xeaa127fa, 0xd4ef3085, 0x04881d05, 0xd9d4d039, 0xe6db99e5, 0x1fa27cf8, 0xc4ac5665,
            0xf4292244, 0x432aff97, 0xab9423a7, 0xfc93a039, 0x655b59c3, 0x8f0ccc92, 0xffeff47d, 0x85845dd1,
            0x6fa87e4f, 0xfe2ce6e0, 0xa3014314, 0x4e0811a1, 0xf7537e82, 0xbd3af235, 0x2ad7d2bb, 0xeb86d391};
        static constexpr int S[64] = {7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22,
                                      5, 9,  14, 20, 5, 9,  14, 20, 5, 9,  14, 20, 5, 9,  14, 20,
                                      4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23,
                                      6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21};
        std::uint32_t m[16];
        for (int i = 0; i < 16; ++i)
            m[i] = static_cast<std::uint32_t>(p[4 * i]) | (static_cast<std::uint32_t>(p[4 * i + 1]) << 8) |
                   (static_cast<std::uint32_t>(p[4 * i + 2]) << 16) | (static_cast<std::uint32_t>(p[4 * i + 3]) << 24);
        std::uint32_t a = h_[0], b = h_[1], c = h_[2], d = h_[3];
        for (int i = 0; i < 64; ++i) {
            std::uint32_t f;
            int g;
            if (i < 16) {
                f = (b & c) | (~b & d);
                g = i;
            } else if (i < 32) {
                f = (d & b) | (~d & c);
                g = (5 * i + 1) % 16;
            } else if (i < 48) {
                f = b ^ c ^ d;
                g = (3 * i + 5) % 16;
            } else {
                f = c ^ (b | ~d);
                g = (7 * i) % 16;
            }
            const std::uint32_t t = d;
            d = c;
            c = b;
            b = b + rotl(a + f + K[i] + m[g], S[i]);
            a = t;
        }
        h_[0] += a;
        h_[1] += b;
        h_[2] += c;
        h_[3] += d;
    }

    std::uint32_t h_[4] = {0x67452301, 0xefcdab89, 0x98badcfe, 0x10325476};
    std::uint64_t bytes_ = 0;
    std::uint8_t buf_[64] = {};
    std::size_t fill_ = 0;
};

}  // namespace sstvae::util

#endif  // SSTVAE_UTIL_MD5_HPP

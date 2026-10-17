#include "util/sha256.hpp"

#include <cstring>

namespace sstvae::util {

namespace {

constexpr std::array<std::uint32_t, 64> K = {
    0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu, 0x59f111f1u,
    0x923f82a4u, 0xab1c5ed5u, 0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u,
    0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u, 0xe49b69c1u, 0xefbe4786u,
    0x0fc19dc6u, 0x240ca1ccu, 0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
    0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u, 0xc6e00bf3u, 0xd5a79147u,
    0x06ca6351u, 0x14292967u, 0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u,
    0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u, 0xa2bfe8a1u, 0xa81a664bu,
    0xc24b8b70u, 0xc76c51a3u, 0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
    0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au,
    0x5b9cca4fu, 0x682e6ff3u, 0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
    0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u,
};

constexpr std::array<std::uint32_t, 8> H0 = {
    0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
    0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u,
};

inline std::uint32_t ror(std::uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }

void compress(std::array<std::uint32_t, 8>& h, const std::uint8_t* blk) {
    std::uint32_t w[64];
    for (int i = 0; i < 16; ++i)
        w[i] = std::uint32_t(blk[4 * i]) << 24 | std::uint32_t(blk[4 * i + 1]) << 16 |
               std::uint32_t(blk[4 * i + 2]) << 8 | std::uint32_t(blk[4 * i + 3]);
    for (int i = 16; i < 64; ++i) {
        std::uint32_t s0 = ror(w[i - 15], 7) ^ ror(w[i - 15], 18) ^ (w[i - 15] >> 3);
        std::uint32_t s1 = ror(w[i - 2], 17) ^ ror(w[i - 2], 19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    std::uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], k = h[7];
    for (int i = 0; i < 64; ++i) {
        std::uint32_t t1 = k + (ror(e, 6) ^ ror(e, 11) ^ ror(e, 25)) + ((e & f) ^ (~e & g)) + K[i] + w[i];
        std::uint32_t t2 = (ror(a, 2) ^ ror(a, 13) ^ ror(a, 22)) + ((a & b) ^ (a & c) ^ (b & c));
        k = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
    }
    h[0] += a; h[1] += b; h[2] += c; h[3] += d; h[4] += e; h[5] += f; h[6] += g; h[7] += k;
}

}  // namespace

Sha256::Sha256() { reset(); }

void Sha256::reset() {
    h_ = H0;
    n_bytes_ = 0;
    n_buf_ = 0;
}

Sha256& Sha256::update(std::span<const std::uint8_t> data) {
    const std::uint8_t* p = data.data();
    std::size_t len = data.size();
    n_bytes_ += len;
    while (len) {
        std::size_t take = 64 - n_buf_;
        if (take > len) take = len;
        std::memcpy(buf_.data() + n_buf_, p, take);
        n_buf_ += static_cast<std::uint32_t>(take);
        p += take;
        len -= take;
        if (n_buf_ == 64) {
            compress(h_, buf_.data());
            n_buf_ = 0;
        }
    }
    return *this;
}

Sha256& Sha256::update(std::string_view data) {
    return update(std::span<const std::uint8_t>(
        reinterpret_cast<const std::uint8_t*>(data.data()), data.size()));
}

Sha256Digest Sha256::digest() {
    const std::uint64_t bits = n_bytes_ * 8;
    buf_[n_buf_++] = 0x80;
    if (n_buf_ > 56) {
        std::memset(buf_.data() + n_buf_, 0, 64 - n_buf_);
        compress(h_, buf_.data());
        n_buf_ = 0;
    }
    std::memset(buf_.data() + n_buf_, 0, 56 - n_buf_);
    for (int i = 0; i < 8; ++i) buf_[56 + i] = static_cast<std::uint8_t>(bits >> (56 - 8 * i));
    compress(h_, buf_.data());
    Sha256Digest out{};
    for (int i = 0; i < 8; ++i) {
        out[4 * i] = static_cast<std::uint8_t>(h_[i] >> 24);
        out[4 * i + 1] = static_cast<std::uint8_t>(h_[i] >> 16);
        out[4 * i + 2] = static_cast<std::uint8_t>(h_[i] >> 8);
        out[4 * i + 3] = static_cast<std::uint8_t>(h_[i]);
    }
    return out;
}

Sha256Digest sha256(std::span<const std::uint8_t> data) { return Sha256().update(data).digest(); }

Sha256Digest sha256(std::string_view data) { return Sha256().update(data).digest(); }

std::string to_hex(std::span<const std::uint8_t> bytes) {
    static constexpr char digits[] = "0123456789abcdef";
    std::string s;
    s.reserve(bytes.size() * 2);
    for (std::uint8_t b : bytes) {
        s.push_back(digits[b >> 4]);
        s.push_back(digits[b & 15]);
    }
    return s;
}

}  // namespace sstvae::util

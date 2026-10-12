// SHA-256 (FIPS 180-4), Qt-free.
//
// QRSSTVAE defines every on-air sequence as SHA-256 of a domain string
// and a counter (sstvae/qrss/sequences.py, decision D5), and its picture
// ID is a SHA-256 of the stored latents, so the core needs a hash that
// does not come from Qt: `sstvae_core` links no Qt at all, and
// QCryptographicHash is only reachable from the fetcher. This is
// qrss_beacon_c/sha256.c moved into C++ unchanged in substance -- the
// C reference is FIPS-tested by tests/test_qrss_c_ref.py, and keeping
// the same code means one implementation has been checked twice.

#ifndef SSTVAE_UTIL_SHA256_HPP
#define SSTVAE_UTIL_SHA256_HPP

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>

namespace sstvae::util {

using Sha256Digest = std::array<std::uint8_t, 32>;

class Sha256 {
public:
    Sha256();
    Sha256& update(std::span<const std::uint8_t> data);
    Sha256& update(std::string_view data);
    // Finishes the hash. The object is spent afterwards: call reset()
    // before reusing it.
    Sha256Digest digest();
    void reset();

private:
    std::array<std::uint32_t, 8> h_{};
    std::uint64_t n_bytes_ = 0;
    std::array<std::uint8_t, 64> buf_{};
    std::uint32_t n_buf_ = 0;
};

Sha256Digest sha256(std::span<const std::uint8_t> data);
Sha256Digest sha256(std::string_view data);

// Lowercase hex, as Python's hexdigest().
std::string to_hex(std::span<const std::uint8_t> bytes);

}  // namespace sstvae::util

#endif  // SSTVAE_UTIL_SHA256_HPP

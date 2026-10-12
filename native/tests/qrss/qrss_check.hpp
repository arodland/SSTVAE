// Helpers shared by the QRSSTVAE ctests (test_qrss_*.cpp).
//
// Most of the transmit chain is bit-exact against Python, so the cheapest
// strong check a ctest can make without Python is a SHA-256 of an output's
// bytes against a digest the Python reference produced -- pinned in the
// test, with the Python one-liner that made it beside it.

#pragma once

#include <complex>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "check.hpp"
#include "util/sha256.hpp"

namespace sstvae::qrss_check {

// SHA-256 of a contiguous array's bytes, as hashlib.sha256(a.tobytes())
// sees a little-endian numpy array of the same dtype.
template <typename T>
std::string digest(const std::vector<T>& v) {
    const auto* p = reinterpret_cast<const std::uint8_t*>(v.data());
    return util::to_hex(util::sha256(std::span<const std::uint8_t>(p, v.size() * sizeof(T))));
}

template <typename T>
void digest_is(const std::vector<T>& v, const std::string& want, const std::string& what) {
    check::equal(digest(v), want, what);
}

}  // namespace sstvae::qrss_check

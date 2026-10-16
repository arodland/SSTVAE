// Helpers shared by the QRSSTVAE ctests (test_qrss_*.cpp).
//
// Most of the transmit chain is bit-exact against Python, so the cheapest
// strong check a ctest can make without Python is a SHA-256 of an output's
// bytes against a digest the Python reference produced -- pinned in the
// test, with the Python one-liner that made it beside it.

#pragma once

#include <complex>
#include <cstdint>
#include <cstdio>
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

// The pins were made by numpy and scipy on glibc. A value computed through
// the C library's sin, cos, exp, lgamma and the like can differ in its last
// bit under another C library (Apple's, Microsoft's), so off glibc these
// checks print a mismatch rather than fail it. Linux CI holds them on two
// architectures; everything pinned with plain digest_is -- the transmitter's
// output among it -- is held on every platform.
#if defined(__GLIBC__)
inline constexpr bool LIBM_PINS_HOLD = true;
#else
inline constexpr bool LIBM_PINS_HOLD = false;
#endif

inline void libm_pin(bool ok, const std::string& what, const std::string& detail = "expected true") {
    if (LIBM_PINS_HOLD) {
        ++check::checks;
        if (!ok) check::fail(what, detail);
    } else if (!ok) {
        std::fprintf(stderr, "note: %s: %s (a glibc libm pin, not held on this C library)\n", what.c_str(),
                     detail.c_str());
    }
}

template <typename A, typename B>
void libm_equal(const A& got, const B& want, const std::string& what) {
    libm_pin(got == want, what, "got " + check::to_text(got) + ", want " + check::to_text(want));
}

template <typename T>
void libm_digest_is(const std::vector<T>& v, const std::string& want, const std::string& what) {
    libm_equal(digest(v), want, what);
}

}  // namespace sstvae::qrss_check
